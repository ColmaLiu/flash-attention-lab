# 44 · FP8 前向：E4M3 的 Q/K/V、descale、P 的 FP8 量化与误差

> 部分 4 · FA4 — Blackwell 上的完整优化　｜　前置：[40_fa4_pipeline](../40_fa4_pipeline/README.md)　｜　代码：[`students/44_fa4_fp8/kernel.cu`](../../students/44_fa4_fp8/kernel.cu)

## 1. 本关目标

40 关搭好了 FA4 的流水结构：两个 Q tile ping-pong、P 写回 TMEM、correction warpgroup。本关把它的输入换成 **FP8**：

| | 40 关 | 本关 |
|---|---|---|
| Q、K、V | FP16/BF16，2 字节 | **E4M3**，1 字节，外加每个张量一个 FP32 的 descale |
| 真实值 | 存的值 | `q_descale · Q₈`、`k_descale · K₈`、`v_descale · V₈` |
| Q·Kᵀ、P·V | `tcgen05.mma .kind::f16`，一条 K = 16 | **`.kind::f8f6f4`，一条 K = 32** |
| P | 16 位，TMEM 每列 2 个 | **E4M3，先乘 2⁸ 再量化**，TMEM 每列 4 个 |
| O | 与输入同 dtype | **BF16** |
| 误差 | 输出舍入为主 | **P 的量化为主**，本关要测量它 |

流水结构（warp 分工、barrier、TMEM 列）与 40 关步骤 3 **完全相同**，骨架里已经写好。本关三步：

1. **FP8 的 Q·Kᵀ**：8 位操作数的描述符与 instruction descriptor，两个 descale 折进 softmax 的 scale；S 写入 probe 检查；
2. **FP8 的 P·V**：P 乘 2⁸ 量化为 E4M3、四个一列写回 TMEM，TS 形式的 8 位 MMA，epilogue 撤销缩放；
3. **误差分析**：测量各误差来源，把 O 的归一化改为“实际参与 P·V 的量化值之和”。

做完后你应该能：

- 写出 E4M3 的位布局、最大值、最小正规数和单位舍入误差，解释为什么 P 要乘 2⁸ 而不是直接量化；
- 说清 `kind::f8f6f4` 的 instruction descriptor 与 16 位版本差在哪里，8 位 tile 的 swizzle 为什么是“一行 D 字节”；
- 推导 FP8 attention 的误差界，用 `./lab check` 的 `max|err|` 对上它；
- 解释为什么 Tensor Core 快了一倍，整个 kernel 只快约 1.2 倍。

## 2. 动机：FP8 能带来什么

B200 的 Tensor Core 做 FP8 乘法的吞吐是 BF16 的 **2 倍**（标称稠密约 4.5 PFLOP/s 对 2.25 PFLOP/s）：同样约 64 个周期，一条 `kind::f16` MMA 做 128×128×16，一条 `kind::f8f6f4` MMA 做 128×128×32。数据也只有一半大：Q、K、V 各 1 字节/元素，shared memory 的 K/V 环、TMA 的流量都减半。

FlashAttention-3 论文 §3.3 第一次在 Hopper 上把 attention 做成 FP8，并指出两个难点：

1. **布局**：Hopper 的 FP8 `wgmma` 要求两个操作数都是 K-major，而 P·V 中的 V 是 MN-major（D 连续），FA3 在 kernel 里用 `ldmatrix`/`stmatrix` 把 V tile 转置；
2. **精度**：E4M3 只有 3 位尾数。FA3 用 **block quantization**（每块一个 scale）和 **incoherent processing**（Q、K 先乘同一个随机正交矩阵，打散离群值）降低 Q、K 的量化误差。

在 B200 上第 1 点不存在了：`tcgen05.mma` 对 E4M3 允许 MN-major 的操作数（§4.6），V 可以照 40 关的方式直接用。第 2 点中的 Q/K/V 量化发生在**输入**里，本关的 harness 已经给了量化好的张量和 per-tensor descale；kernel 自己引入的量化只有 **P**。FA4（`flash_fwd_sm100.py`）的做法是把 P 乘 2⁸ 后量化成 E4M3，本关照做，并测量它的误差。

一个提醒：40 关 §7.2 的结论是**瓶颈在 softmax（MUFU）**，Tensor Core 只忙约 57%。FP8 只让 GEMM 变快，所以不要期待 2 倍——本关 §7.2 会先算出预期，再用实测对上。

## 3. 数学

### 3.1 E4M3

8 位：1 位符号、4 位指数（偏置 7）、3 位尾数，**没有无穷大**，只有 `S.1111.111` 是 NaN（PyTorch 的 `float8_e4m3fn`）。

| | 位模式 | 值 |
|---|---|---|
| 最大值 | `0.1111.110` | 1.75 × 2⁸ = **448** |
| 1 | `0.0111.000` | 1 |
| 最小正规数 | `0.0001.000` | 2⁻⁶ = 0.015625 |
| 最小次正规数 | `0.0000.001` | 2⁻⁹ ≈ 0.00195 |

- 每个 2 的幂区间 [2ᵉ, 2ᵉ⁺¹) 里只有 8 个值，间距 2ᵉ⁻³。就近舍入的**相对误差至多 u = 2⁻⁴ = 6.25%**（u 叫单位舍入误差；BF16 是 2⁻⁸，FP16 是 2⁻¹¹）；
- 次正规区（< 2⁻⁶）间距固定为 2⁻⁹，越小的数相对误差越大：0.003 只能变成 0.00195 或 0.0039；
- 转换成 E4M3 的 `cvt.rn.satfinite.e4m3x2.f32` 中 `.satfinite` 是**必须的**（ptxas 不接受没有它的 `.e4m3x2` 目标）：超出范围的值饱和到 ±448，而不是变成 NaN。

E5M2（2 位尾数，最大 57344）范围大、精度更低，常用于梯度；前向用 E4M3。

输入怎么来的：`make_inputs` 先生成 FP32 的随机数（真实值），Q、K、V 分别乘 2、8、4 后转成 E4M3 存储，对应 `q_descale = 0.5`、`k_descale = 0.125`、`v_descale = 0.25`。真实值 = descale × 存储值。三个 descale 故意取不同的 2 的幂：既让存储值落在 E4M3 的正规区（标准正态乘 8 后最大约 40，远小于 448），又让“用错 descale”的错误（例如 `q_descale·q_descale`）一定被检查出来。

### 3.2 descale 折进 scale

真实分数是

```
s_real = scale · (q_descale · Q₈) · (k_descale · K₈)ᵀ = (scale · q_descale · k_descale) · (Q₈ · K₈ᵀ)
                                                        └──────── qk_scale ────────┘
```

Tensor Core 算的是 `Q₈·K₈ᵀ`：两个 E4M3 的乘积有效位不超过 8 位，在 FP32 中**精确**，累加用 FP32。所以只要把 40 关的 `scale` 换成 `qk_scale`（`scale_log2 = qk_scale · log₂e`），online softmax 的代码一个字都不用改，m 仍然存原始分数。每个元素没有任何额外乘法。

步骤 1 的 probe 写的是 `qk_scale · s`（真实单位），对照 FP64 的 `scale · (q_descale·Q₈)(k_descale·K₈)ᵀ`，容差 1e-3。实测误差约 1e-7–1e-5（只有 FP32 累加的舍入）。

`v_descale` 不能这样折：它乘在 V 上，而 O = Σ P·V 是线性的，所以**在 epilogue 每个输出元素乘一次**即可。

### 3.3 P 乘 2⁸ 再量化

40 关中 `p = 2^(s·scale_log2 − m·scale_log2) ∈ (0, 1]`。本关写回 TMEM 的是

```
p̂ = E4M3(2⁸ · p)            即 exp2 的参数多减 8：2^(s·scale_log2 − (m·scale_log2 − 8))
```

为什么乘 2⁸：

- **最大值 256 < 448**，不会饱和；
- 原来 p < 2⁻⁶ 就掉进次正规区，乘 2⁸ 后要到 p < 2⁻¹⁴ 才会；
- **2 的幂只改指数，不改尾数**：正规区内 E4M3(2⁸·p) = 2⁸·E4M3(p)，不多引入任何舍入。所以 2⁸ 优于 448（448 = 1.75 × 2⁸ 会改变每个值在区间里的位置，误差只是“换了一组随机数”，§3.5 实测还略差）。

怎样撤销 2⁸：ℓ 用同样乘过 2⁸ 的 p 累加（同一次 `exp2` 的结果），`O / ℓ` 中两个 2⁸ 正好约掉。LSE 要减掉它：

```
O_real[i] = v_descale · O[i] / ℓ
LSE[i]    = qk_scale · m + ln ℓ − 8·ln 2
```

FA4 的同一做法见 `flash_fwd_sm100.py` 的 Note [Low Precision Scaling]：`max_offset = 8`，epilogue 中 `scale = rcp(row_sum) * v_descale`，LSE 中 `log2(row_sum) − max_offset`。

### 3.4 手算：什么时候 2⁸ 有用

一行有 1 个 key 的 p = 1，另外 1000 个 key 的 p = e⁻⁶ ≈ 0.00248（“长尾”）。

| | 尾部一个 p 量化后 | 相对误差 | 尾部总和（真值 2.48） | 最大 key 的权重（真值 1/3.48 = 0.287） |
|---|---|---|---|---|
| 直接量化 | 0.00248 落在次正规区，间距 2⁻⁹：→ 0.00195 | −21% | 1.95 | 1/2.95 = 0.339 |
| 乘 2⁸ | 0.635 落在 [0.5, 1)，间距 2⁻⁴：→ 0.625 | −1.5% | 2.44 | 1/3.44 = 0.291 |

直接量化时 O 偏离约 0.05 × |v_top − v̄_tail|，乘 2⁸ 后约 0.003 ×。

`long_tail` case 把这个例子推到极端：每个 head 一个热 key（logit ≈ 8），其余 4095 个 key 的 logit ≈ 0，即 p ≈ e⁻⁸ ≈ 3.4e-4，尾部总和约 1.4，比热 key 还重；V 的均值是 1。

| `long_tail` [1,4096,2,128] | 尾部一个 p | `max|err|`（步骤 2 / 步骤 3） |
|---|---|---|
| 直接量化 | 3.4e-4 < 2⁻¹⁰（最小次正规数的一半）→ **0**：整个尾部消失 | 0.61 / 2.0 |
| 乘 2⁸ | 0.086，正规数 → 相对误差 ≤ 2⁻⁴ | 5.7e-3 / 5.7e-3 |

（步骤 3 更糟：它用 Σp̂ 归一化，尾部为 0 时 O 就等于热 key 的 V。）

对其余随机数据的 case，p < 2⁻⁶ 的 key 对 ℓ 的贡献很小：把 `kPLog2` 改成 0，它们的误差和乘 2⁸ 相同。2⁸ 几乎没有代价（多一次减法）。

### 3.5 误差分析

O 的误差有这些来源（括号里是本关测得的量级）：

| 来源 | 大小 | 是否计入本关的检查 |
|---|---|---|
| Q、K、V 量化成 E4M3 | 每个元素相对 ≤ 2⁻⁴ | 否：oracle 用的就是反量化后的输入（`descale × 存储值`）。它属于输入；减小它要靠 FA3 的 block quantization / incoherent processing |
| Q₈·K₈ᵀ 的 FP32 累加 | 相对约 1e-7 | 是（步骤 1 的 probe） |
| `ex2.approx` | 相对约 2⁻²² | 是，可忽略 |
| **P 量化成 E4M3** | 每个 p 相对 ≤ u = 2⁻⁴ | **是：主要来源** |
| O 舍入成 BF16 | 相对 ≤ 2⁻⁹ | 是 |

**P 量化的误差界。** 记 p̂_j = p_j(1 + δ_j)，|δ_j| ≤ u。若归一化也用 p̂（步骤 3），O 是 V 行的**凸组合**，权重 ŵ_j = p̂_j / Σp̂：

```
Ô − O = Σ_j (ŵ_j − w_j) v_j = Σ_j (ŵ_j − w_j)(v_j − O)        （因为 Σŵ = Σw = 1）
|ŵ_j − w_j| ≤ w_j · 2u/(1 − u)   ⇒   |Ô − O| ≤ 2u/(1−u) · Σ_j w_j |v_j − O|
```

- 权重分散在很多 key 上时（随机数据），δ_j 正负抵消，误差约 u·√(Σw_j²)·σ_V，很小（`medium` 约 7e-3；P 保持 BF16 时约 1e-3，见下表）；
- 权重集中在一两个 key 上时（`large_logits`、causal 的前几行），抵消不了，误差约 u·|v_a − v_b| 量级——`peaked`、`causal_peaked` 的 5e-2–7.5e-2 就是这样来的。

**ℓ 用 FP32 的 p 还是量化后的 p̂？** 步骤 2（与 FA4 相同）用 FP32 的 p 求 ℓ，分子却是 Σp̂·v，两者不一致：V = 1 时 O = Σp̂/Σp ≠ 1。步骤 3 改用 Σp̂ 归一化，O 重新成为凸组合，`ones_v` 精确为 1。**但 LSE 必须仍用 FP32 的 ℓ**：LSE 描述的是真实的 softmax，Σp̂ 带着最大约 1% 的偏差（LSE 的容差是 1e-3）。所以 softmax 线程维护两个和：`l`（FP32，给 LSE）和 `lo`（给 O）。

**实测**（`max|err|`，与 FP64 oracle 比；“BF16 P”是 PyTorch 中按同样分块逐块把 P 舍入为 BF16 的模拟，不是 kernel；其余是 kernel 实测）：

| case | BF16 P（模拟） | 步骤 2：E4M3 P，ℓ = Σp | 步骤 3：E4M3 P，O 用 Σp̂ | FA4 FP8 |
|---|---|---|---|---|
| `medium` | 9.8e-4 | 7.1e-3 | 7.1e-3 | 6.6e-3 |
| `position` | 2.0e-3 | 4.5e-3 | 3.7e-3 | 4.6e-3 |
| `causal_position` | 1.9e-3 | 4.5e-3 | 2.2e-3 | 3.6e-3 |
| `ones_v` | ≈ 0 | 7.8e-3 | ≈ 1e-6 | 7.8e-3 |
| `causal_ones_v` | ≈ 0 | 2.0e-2 | ≈ 1e-6 | 2.0e-2 |
| `peaked` | 1.2e-2 | 4.7e-2 | 4.5e-2 | 6.6e-2 |
| `causal_peaked` | 1.4e-2 | 7.5e-2 | 5.1e-2 | 7.5e-2 |

- E4M3 的 P 让误差比 BF16 P 大约 4–7 倍，正是 u 从 2⁻⁸ 变成 2⁻⁴ 的结果（集中权重的行）；
- 步骤 3 消除了 Σp̂/Σp 的偏差，`ones_v` 从 7.8e-3 变成 ≈ 0；但最坏的行（`peaked`）几乎不变，因为那里误差来自权重本身的相对误差，不来自归一化；
- `kPLog2` 改为 0（不缩放）：随机数据的 case 误差与 2⁸ 相同，`long_tail`、`causal_long_tail` 失败（§3.4）；改为 log₂448：`peaked` 变成 6.1e-2（更差）。

**容差。** 本关 O 的默认容差是 `|err| ≤ u + 2u·|expected|`，即 (0.0625, 0.125)。它按 u 取量级、用实测校准，**不是**推导出的上界（上面的界要宽得多；两个 key 平分权重的行误差也可能超过 u·|v_a − v_b|/2）：本关的答案和 FA4 用掉这个预算的约 75%，P scale 取 448 的变体约 95%。

答案已知的结构化 case 用 `o_tol` 收紧（取实测误差的约 2–3 倍）：`uniform`、`causal_uniform`（P 全为 256，精确）为 (1e-3, 4e-3)，只剩 BF16 输出的舍入；`position` 类 case 为 1.5e-2；`ones_v` 为 2e-2、`causal_ones_v` 为 4e-2（步骤 2 的 Σp̂/Σp 偏差）；只在步骤 3 检查的 `ones_v_exact`、`causal_ones_v_exact` 为 1e-5。这样“截断而不是就近舍入 P”这类错误在步骤 2 的十几个 case 上都会失败。

## 4. 硬件与指令

### 4.1 `tcgen05.mma .kind::f8f6f4`

```
tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;     // SS
tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;   // TS
```

| | `kind::f16` | `kind::f8f6f4`（E4M3） |
|---|---|---|
| 一条的 K | 16 个元素 = **32 字节** | 32 个元素 = **32 字节** |
| M × N | 128 × 8..256 | 相同 |
| 累加 | FP32（或 FP16） | FP32 |
| 一条约多少周期（128×128） | 约 64 | 约 64：每周期 FLOP 翻倍 |

头文件新增 `lab::mma_f8f6f4_ss`、`lab::mma_f8f6f4_ts`。注意 K 片的步长：**按字节算和 16 位完全一样**，都是 32 字节；按元素算是 32 个。D = 128 时一行 Q 只要 4 条 MMA（40 关是 8 条）。

### 4.2 instruction descriptor

位的位置与 `kind::f16` 相同（31 关），只有 A/B format 字段的含义变了：

| 位 | 含义 | 本关 |
|---|---|---|
| 4–5 | D 格式：1 = F32 | 1 |
| 7–9 / 10–12 | A / B 格式：**0 = E4M3**、1 = E5M2、3 = E2M3、4 = E3M2、5 = E2M1 | 0 / 0 |
| 15 / 16 | A / B 为 MN-major | QKᵀ：0 / 0；PV：0 / 1 |
| 17–22 | N >> 3 | |
| 24–28 | M >> 4 | |

`lab::make_idesc_f8f6f4(m, n, a_fmt, b_fmt, a_mn, b_mn)`。一个容易出错的巧合：E4M3 的格式码 0 和 F16 的格式码 0 相同，所以 `make_idesc_f16(128, 128, /*bf16=*/false)` 的位和 E4M3 的 idesc **一模一样**——但它配 `kind::f16` 用就是把字节当 FP16 读，结果全错。决定数据类型的是 **kind 加格式码**，不是格式码本身。

### 4.3 8 位 tile 的 shared memory 布局

40 关的 tile 是 `[D/64 个 panel][行][64 个 16 位元素]`，一个 panel 的一行正好 128 字节 = 一个 128B swizzle 行。8 位时 **一行 D 个元素就是 D 字节**：

| D | 一行 | swizzle | tile 布局 | 一个 atom（8 行） | K-major 描述符 |
|---|---|---|---|---|---|
| 128 | 128 字节 | 128B | `[128 行][128]`，一个 panel | 1024 字节 | `make_smem_desc_kmajor_sw128`，SBO = 1024 |
| 64 | 64 字节 | **64B** | `[128 行][64]`，一个 panel | 512 字节 | `make_smem_desc_kmajor_sw64`，SBO = 512 |

TMA 的 tensor map 用 `CU_TENSOR_MAP_DATA_TYPE_UINT8`（TMA 只搬字节，不关心 E4M3），box `[1][行][1][D]`，swizzle 与上表一致（骨架的 `fp8_map` 已写好）。64B swizzle 的 XOR 模式每 512 字节重复一次，所以 atom 要 512 字节对齐；本关所有 tile 的偏移都是 1024 的倍数。

K 片 kk 的起始地址 = tile 起点 + 32·kk 字节，描述符的 start address 字段以 16 字节为单位，所以是 `desc + 2·kk`。D = 64 时一行只有两片：0 和 32 字节。

### 4.4 P 写成 E4M3、每列 4 个

```
cvt.rn.satfinite.e4m3x2.f32 d16, a, b;     // a → d 的高字节，b → 低字节
```

**操作数顺序是反的**：第一个源操作数进高字节。四个 key 打包成一个 32 位字，key 4c 在最低字节：

```
TMEM 列 p_addr + c（c = 0..31）的 32 位 = { 字节0: p̂[4c], 字节1: p̂[4c+1], 字节2: p̂[4c+2], 字节3: p̂[4c+3] }
```

一行 128 个 key 只占 **32 列**（40 关的 16 位 P 占 64 列），仍放在 S 的第 64 列起（与 FA4 的 `tmem_s_to_p_offset = n_block_size // 2` 相同），用两次 `tcgen05.st.32x32b.x16`。

### 4.5 TS 形式的 8 位 MMA

A（P）从 TMEM 读：第 i 行在 lane i，一条 MMA 的 K = 32 个 key = **8 列**。所以 key 片 kk 从 `p_tmem + 8·kk` 开始（40 关是 16 位、每列 2 个，同样是 8 列）。

### 4.6 V 不用转置

P·V 的 B 操作数是 V_j `[key][D]`，D 连续，即 **MN-major**。CUTLASS `mma_sm100_desc.hpp` 中 `InstrDescriptor` 的注释：“Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats”——E4M3 可以 MN-major，只有 6 位、4 位格式不行。FA4 对 FP8 的 V 也用 `OperandMajorMode.MN`。

8 位 MN-major 的 swizzle 行是“一个 key 的 D 个字节”：D = 128 时一行 128 字节（N = D = 128 正好一行，LBO 用不到），8 个 key 一个 1024 字节的 atom，SBO = 1024；D = 64 用 64B swizzle，SBO = 512。key 片 kk 前进 32 个 key = 32·D 字节。

所以本关**没有转置**，也就没有“转置算不算进计时”的问题：V 由 TMA 原样装入，计时边界与 40 关相同。Hopper 上的 FA3 必须在 kernel 里转置 V（计时内）。

### 4.7 步骤 3：把量化后的 P 加起来

```
cvt.rn.f16x2.e4m3x2 h01, w16;          // 两个 E4M3 → 两个 FP16（精确）
add.rn.f32.f16 acc, h, acc;            // FP32 += FP16（PTX ISA 8.6，sm_100），SASS 中一条 FHADD
```

E4M3 → FP16 精确，混合精度加法只做一次 FP32 舍入，结果与“先转 FP32 再加”相同，但每个元素只要一条 FHADD，而不是 `HADD2.F32` + `FADD` 两条。骨架已提供 `add_e4m3x4(acc, w)`。用几个独立的部分和，避免 128 条加法串成一条依赖链。

注意这样 O 的分母是 Σp̂，而 LSE 用的是 FP32 的 Σp，所以 exp(LSE) 并不是 O 实际用的归一化常数（相差约 1%）。对前向输出这没有问题；若反向传播用 LSE 重算 P，要知道它与前向的 P̂ 不完全一致。

### 4.8 常见误用

| 误用 | 后果 |
|---|---|
| 用 `kind::f16` 加 E4M3 的 idesc（位相同！） | 把字节对当 FP16 读：S 全错 |
| K 片步长写成 16 个元素（照抄 40 关的元素数） | 只用了一半的 D，另一半读到下一片：S 错 |
| D = 64 仍用 128B swizzle 描述符 | 与 TMA 写入的 64B swizzle 不一致：D = 64 的 case 错 |
| `cvt.e4m3x2` 的两个源操作数顺序反了 | 相邻 key 的 P 交换：`n1`、`peaked*` 错 |
| P 的缩放超过 448（如乘 2¹⁰） | `.satfinite`（`.e4m3x2` 必须带）把大的 p 饱和到 448，O 偏小 |
| P 乘了 2⁸，LSE 没减 8·ln2 | LSE 全部大 5.545 |
| 步骤 3 把 LSE 也改用 Σp̂ | LSE 误差约 1e-2，超过 1e-3 的容差 |
| 忘了乘 `v_descale` | O 大 4 倍 |

## 5. 调度设计

角色、barrier、MMA 发射顺序与 40 关步骤 3 相同（40 关 §5，16 个 warp，`setmaxnreg` 192 / 80 / 48）。变化只在存储预算：

**shared memory**（D = 128 / D = 64）：

| | 40 关步骤 3（BF16） | 本关（E4M3） |
|---|---|---|
| Q⁰、Q¹ | 64 KB / 32 KB | 32 KB / 16 KB |
| O 暂存区（BF16，128 × D × 2 字节） | 复用 Q^t（大小相同） | **复用 Q⁰ + Q¹ 两块合起来**，两个 tile 先后使用 |
| K/V 环 | 4 × 32 KB / 8 × 16 KB | 8 × 16 KB / 12 × 8 KB |
| 统计量 | 3 KB | 4 KB（多了 O 的归一化和 `fin_lo`） |
| 合计 | ≈ 195 KB / ≈ 163 KB | ≈ 165 KB / ≈ 117 KB |

O 暂存为什么可以覆盖两块 Q：`o_full[0]` 由最后一个 PV⁰ 之后的 commit 完成，而全部 QK¹ 都在它之前发出，commit 跟踪同一线程此前的全部 MMA（40 关 §4.3），所以那时 Q⁰、Q¹ 都不再被读。correction warpgroup 先写 tile 0 的 O，每个 warp 的 lane 0 `bulk_wait_group_read<0>` 之后 `__syncwarp`，再写 tile 1 的同一行（每个 warp 只碰自己的 32 行）。

K/V 环的级数：一块只有 16 KB，同样的字节能放 8 级。实测 4 级与 8 级的耗时差别在噪声内（数据只占 HBM 带宽的约 2%）。

**TMEM**：列号与 40 关相同，只是 P^t 只用 S^t 第 64–95 列。

| 列 | D = 128 | D = 64 |
|---|---|---|
| S⁰（P⁰ 在 64–95） | 0–127 | 0–127 |
| S¹（P¹ 在 192–223） | 128–255 | 128–255 |
| O⁰ / O¹ | 256–383 / 384–511 | 256–319 / 320–383 |

**softmax 线程每块多做的事**（相对 40 关）：64 条 `F2FP`（E4M3 打包，两个一条），步骤 3 再加 64 条 `F2FP`（解包）和 128 条 `FHADD`。都不在 MUFU 上。

## 6. 分步练习

骨架中 40 关的全部流水（TMA、MMA 调度、softmax、correction、barrier）、8 位 tensor map、O 暂存、`add_e4m3x4` 都已写好。`p.step` 选择三个配置之一（`Config<D, 1|2|3>`），三步是同一个 kernel 逐步补全。

### 步骤 1 · TODO(1a)–(1c)：FP8 的 Q·Kᵀ

- 1a（host，`launch`）：`qk_scale`、`scale_log2`（§3.2）；
- 1b：`desc_kmajor` 与 `mma_qk`（§4.1–4.3）；
- 1c：`write_probe`：真实单位的 S 写进 probe（§3.2）。

本步 MMA warp 不发 P·V，epilogue 不写 O，只检查 probe（非 causal 的小 case）。

```bash
./lab check 44 --step 1 --case n1
./lab check 44 --step 1
```

### 步骤 2 · TODO(2a)–(2c)：FP8 的 P·V

- 2a：`pack4_e4m3` 与 `write_p_tmem`（§4.4）。`exp2` 参数中的 −8 已经写好（`base - kPLog2`）；
- 2b：`mma_pv_ts`（§4.5–4.6）；
- 2c：`store_o_and_lse` 中的 `f` 和 LSE（§3.3）。

```bash
./lab check 44 --step 2 --case tile1_position
./lab check 44 --step 2
./lab sass 44             # UTCQMMA：A 操作数为 tmem[...] 的就是 TS 形式的 P·V
./lab bench 44 --step 2 --case perf
```

### 步骤 3 · TODO(3)：误差分析

1. 先记下步骤 2 的 `max|err|`：`./lab check 44 --step 2`，特别是 `ones_v`、`causal_ones_v`、`position`、`peaked`；
2. 在 `write_p_tmem` 的 `kQuantL` 分支里返回量化值之和（§4.7）。kernel 中 `lo` 的更新和 epilogue 的 `lo`/`l` 分工已经写好——读懂它们（§3.5）；
3. 对比误差，并用 §3.5 的误差界解释：哪些 case 变好了、哪些没变、为什么。

```bash
./lab check 44 --step 3
./lab bench 44 --case perf
./lab bench 44 --case perf_d64
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 所有 probe 都差一个常数倍（如 4 倍或 1/4） | `qk_scale` 用错了 descale（三个 descale 各不相同：0.5、0.125、0.25），或多乘了 log₂e |
| 步骤 1 只有 D = 64 的 case 错（`one_block`、`scale`、`peaked_d64`） | D = 64 用了 128B swizzle 的描述符；SBO 不是 512 |
| 步骤 1 D = 128 错、`n1` 也错 | K 片的字节偏移；idesc 的 N/M；用了 `mma_f16_ss` |
| 步骤 2 `n1` 的 O 错、LSE 对 | P 的字节顺序（`n1` 只有 key 0，它必须在字节 0） |
| 步骤 2 O 对、LSE 全部差 5.545 | LSE 没减 8·ln 2 |
| 步骤 2 O 全部差 4 倍 | 忘了 `v_descale` |
| 步骤 2 D = 64 的 O 错、D = 128 对 | V 的描述符：64B swizzle、SBO = 512、key 片步长 32·D 字节 |
| 步骤 3 LSE 错约 1e-2 | 把 LSE 也改成了 Σp̂（§3.5） |
| 步骤 3 `ones_v_exact` 失败（O 不是精确的 1） | `lo` 仍累加 FP32 的 p；或部分和漏了一个 |
| 步骤 2 `position` 类、`ones_v` 误差约 2.5e-2–6e-2 | P 被截断而不是就近舍入（例如自己拼位而不用 `cvt.rn`） |
| 步骤 2 `long_tail` 误差约 0.6 | P 没有乘 2⁸（`base - kPLog2` 被改掉） |

## 7. 性能

### 7.1 相对 FA4 FP8 与 40 关

实测（CUDA Graph，中位数；`vs FA4 FP8` = FA4 FP8 耗时 / 本版本耗时；范围来自 2–3 次运行，单次之间约有 ±3% 的波动）：

| 版本 | `perf` [4,4096,16,128] | `perf_causal` | `perf_d64` [4,4096,32,64] | `perf_8k` [2,8192,16,128] |
|---|---|---|---|---|
| 步骤 2（E4M3 P，ℓ = Σp） | ≈ 0.68–0.69× | ≈ 0.71× | ≈ 0.73× | ≈ 0.68× |
| 步骤 3（答案） | ≈ 0.69–0.73× | ≈ 0.73× | ≈ 0.74× | ≈ 0.70–0.72× |
| FA4 FP8（`flash_attn.cute`） | 1× | 1× | 1× | 1× |
| 本关步骤 3 相对 40 关步骤 3（BF16/FP16） | ≈ 1.23× | ≈ 1.15× | ≈ 1.12× | ≈ 1.24× |
| FA4 FP8 相对 FA4 BF16/FP16 | ≈ 1.27× | ≈ 1.12× | ≈ 1.05× | ≈ 1.37× |

- 你的步骤 3 在四个 perf case 上都达到 FA4 FP8 的约 0.63 倍以上，就说明本关的机制成立；cuDNN 没有 FP8 的 SDPA 基线（`bench` 显示“unavailable”）；
- 步骤 3 多了解包和 128 条 FHADD，却和步骤 2 一样快甚至略快（约 1–4%，可重复）。我们没有查明原因（可能是指令调度的变化）；结论是“一致的归一化”几乎不花时间；
- FA4 比我们快约 1.4 倍，因为它已经用上 41–43 关的优化（多项式 exp2、条件 rescale、persistent 调度等），这些不属于本关。

### 7.2 为什么只快约 1.2 倍

40 关 §3.1 的链模型：一个 tile 每块的周期 `T ≈ SM + PV + QK + 交接`。40 关（D = 128）实测 SM ≈ 1900–2200 周期，PV + QK ≈ 1100 周期，T ≈ 3450。FP8 让 PV + QK 减半到约 550 周期，softmax 不变：

```
T_fp8 ≈ 3450 − 550 ≈ 2900 周期    →   3450 / 2900 ≈ 1.19×
```

与 `perf`、`perf_8k` 实测的约 1.23× 一致。D = 64 时 GEMM 在链中占得更少，所以只快约 1.12×。

Nsight Compute（`./lab profile 44 --solution --case perf`）：

| 指标 | 40 关步骤 3 | 本关 |
|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 57% | **≈ 29%** |
| `sm__inst_executed_pipe_xu`（MUFU） | ≈ 57% | ≈ 58% |
| `dram__throughput` | — | ≈ 2% |

Tensor Core 的忙碌比例正好减半，MUFU 的比例不变：瓶颈仍然是 softmax 的 `exp2`。HBM 几乎闲着，所以“字节减半”对本 kernel 的耗时没有直接作用（它的作用是让 shared memory 能放更多级、为更大的 tile 留空间）。FP8 attention 是 **MUFU-bound** 的，FA4 源码中 `_FP8_TUNING_CONFIG` 的注释也这样说（“FP8 fwd is MUFU/ex2-bound”），因此 41 关的多项式 `exp2` 对 FP8 更重要。

```bash
./lab profile 44 --case perf
```

## 8. 自测问题

1. E4M3 能表示 300 吗？0.004 呢？它们分别会被舍入成什么？
2. D = 64 时，Q tile 的第 r 行第 40 个元素在 shared memory 中的字节偏移是多少（相对 tile 起点，考虑 64B swizzle：16 字节块 q 存在 q ^ ((r / 2) % 4)）？为什么不能用 128B swizzle？
3. PV 的第 kk = 3 条 MMA：A 的 TMEM 列（相对 base，t = 1）是多少？V 的描述符 start address 前进了多少字节（D = 128）？
4. 如果 P 改成乘 2¹⁰，会发生什么？乘 2⁸ 时 p̂ 最大是多少、离 448 还有多远？FA4 允许 softmax 的最大值“滞后”（41 关的条件 rescale）时，为什么要把这段余量算进去？
5. 为什么步骤 3 中 O 用 Σp̂ 归一化、LSE 却要用 FP32 的 Σp？如果反过来会怎样？
6. FP8 让 Tensor Core 的工作时间减半，为什么 `perf` 只快约 1.2 倍？如果 softmax 也快一倍（41 关的目标之一），按 §7.2 的模型会快多少？

<details>
<summary>参考答案</summary>

1. 300 在 [256, 448] 区间，间距 32：300 → 288（离 288 差 12，离 320 差 20）。0.004 在次正规区，间距 2⁻⁹ ≈ 0.00195：0.004 ≈ 2.05 × 2⁻⁹ → 2 × 2⁻⁹ = 0.0039。
2. 第 40 字节在 16 字节块 q = 2（字节 32–47）中，块内偏移 8；64B swizzle 把块 q 放到 q ^ ((r / 2) % 4)，所以偏移 = 64·r + 16·(2 ^ ((r/2) % 4)) + 8，例如 r = 5 时 = 320 + 16·0 + 8 = 328。`tcgen05` 的 128B swizzle K-major 布局假定每行 128 字节、8 行一个 1024 字节的 atom；D = 64 的一行只有 64 字节，按 128B 布局去读就会把两行当成一行。描述符的 swizzle 必须与 TMA 写入时的 swizzle 一致。
3. P¹ 从 S¹ 的第 64 列开始：`128 + 64 + 8·3 = 216`（lane 字段 0），对应 key 96–127。V 前进 3 × 32 个 key × 128 字节 = 12288 字节（描述符加 768）。
4. 2¹⁰ = 1024 > 448：p 接近 1 的值饱和到 448（`satfinite`），P·V 中最大的权重被压小，而 FP32 的 ℓ 仍按 1024 计，O 偏小（FA4 #2716 的问题）。乘 2⁸ 时最大 256，余量 448/256 = 1.75，即 log₂ 约 0.8。若允许最大值滞后 τ 个 log₂ 单位，p 可以到 2^τ，乘 2⁸ 后到 2^(8+τ)，必须 < 448，所以 FA4 断言 `max_offset + rescale_threshold < log2(448)`，FP8 时 rescale_threshold 取 0。
5. O = Σp̂v / Σp̂ 是精确的凸组合：权重之和为 1，`ones_v` 精确为 1，偏差不会以 v 的均值的形式进入 O。LSE 是真实 softmax 的归一化常数，应该是 ln Σexp(s)，Σp̂ 带着每项最多 6.25% 的相对误差，平均下来约 1e-2 量级，超过 LSE 的容差。反过来（O 用 Σp、LSE 用 Σp̂）两者都变差。
6. 链 T ≈ SM + PV + QK + 交接，只有 PV + QK 减半（约 1100 → 550 周期），SM ≈ 2000 不变：3450 / 2900 ≈ 1.19。若 SM 减半到约 1000，链只剩约 1000 + 550 + 交接 ≈ 1700 周期；但两个 tile 共用一个 MUFU：每块 2 × 128 × 128 个 `exp2`、每周期 16 个，至少 2048 周期。所以只要 `exp2` 全在 MUFU 上，T 不会低于约 2048，相对本关最多约 2900 / 2048 ≈ 1.4 倍——FA4 FP8 正好快约 1.4 倍。要突破这个下限，必须把一部分 `exp2` 搬到 FMA 上（41 关）。
</details>

## 9. 延伸阅读

- Jay Shah 等，*FlashAttention-3*（2024）§3.3：FP8 的布局要求（V 转置、P 的寄存器排列）、block quantization、incoherent processing 及其误差实验。
- Zadouri 等，[FlashAttention-4](https://arxiv.org/abs/2603.05451v1)（2026）；上游实现 `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`：
  - Note [Low Precision Scaling] 与 `_LOG2_DTYPE_MAX`：P 乘 2^max_offset、与 rescale_threshold 的约束；
  - `_load_effective_descales`：`qk_descale = q_descale · k_descale` 折进 `softmax_scale_log2`；
  - `correction_epilogue` 附近：`scale = rcp(row_sum) * v_descale`，LSE 中的 `− max_offset`；
  - `_FP8_TUNING_CONFIG`：“FP8 fwd is MUFU/ex2-bound”；
  - `flash_attn/cute/interface.py` 的 `_flash_attn_fwd`（`q_descale` 等参数，形状 `[B, H_kv]`；公开的 `flash_attn_func` 不接受 descale）。
- CUTLASS：`third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp`（`MXF8F6F4Format`、`InstrDescriptor` 中 MN-major 的适用格式）；`third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp`（`SM100_MMA_F8F6F4_SS`、`SM100_MMA_F8F6F4_TS`：“A from TMEM can't be transposed”）。
- PTX ISA 8.7：[tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)（`.kind::f8f6f4`、instruction descriptor 表）、[cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)（`.e4m3x2`、`.satfinite`）、[add](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)（混合精度 `add.f32.f16`）。
- OCP 8-bit Floating Point Specification (OFP8)：E4M3/E5M2 的定义。
- 后续：41 关（softmax 吞吐，FP8 更受益）、42、43 关。
