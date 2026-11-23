# 41 · softmax 吞吐：多项式 exp2、MUFU/FMA 混合、条件 rescale

> 部分 4 · FA4 — Blackwell 上的完整优化　｜　前置：[40_fa4_pipeline](../40_fa4_pipeline/README.md)、[01_softmax](../01_softmax/README.md)　｜　代码：[`students/41_fa4_softmax/kernel.cu`](../../students/41_fa4_softmax/kernel.cu)

## 1. 本关目标

40 关搭好了 FA4 的流水结构，最后的结论是：一个 tile 每块的“链”里最长的一段是 128 个 `exp2`，**MUFU（特殊函数单元）的吞吐成了瓶颈**。本关不动流水结构（warp 分工、barrier、TMEM 布局全部沿用 40 关步骤 3），只改 softmax 的算术，做 FA4 论文里关于 softmax 的两件事：

| | 40 关步骤 3 | 本关 |
|---|---|---|
| 指数 | 全部 `MUFU.EX2` | 一部分在 **FMA 流水线**上用多项式算（软件 `exp2`），其余仍用 MUFU，两条流水线同时工作 |
| 打包运算 | 标量 `FFMA`/`FADD` | **`FFMA2`/`FADD2`**：一条指令做两个 FP32 运算 |
| O 的修正 | 每块都修正 | **条件 rescale**：最大值增长超过 τ = 8（log₂ 单位）才移动基准、修正 O |

接口不变：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]`，D = 64 或 128，causal 与非 causal，输出 O 和 FP32 的 LSE。三步：

1. **软件 `exp2`**：范围约简 + 3 次多项式 + 指数位拼接，全部指数都换成它（预期变慢）；
2. **混合**：每 32 列中固定 6 列用多项式、26 列用 MUFU；
3. **条件 rescale**：softmax 只在必要时移动基准，correction warp 用 warp 投票跳过不需要的修正。

做完后你应该能：

- 用实测的吞吐（每 SM 每周期多少个结果）算出一块 softmax 的 MUFU 下限，以及软件 `exp2` 能分担多少；
- 写出 `2^x = 2^⌊x⌋ · 2^f` 的位运算实现，说明每一步的误差，以及为什么常数项必须恰好是 1；
- 从 BF16/FP16 的舍入误差出发选多项式的次数，并解释讲义中穷举实测的最大相对误差从哪里来；
- 证明条件 rescale 的结果与每块都 rescale 在数学上相同，并给出不溢出的条件。

## 2. 动机：40 关卡在哪里

40 关 §7.2 的插桩（`perf` case，稳态，SM 周期）：

| 段 | 周期 |
|---|---|
| 一块的周期 T（两个 tile 各一次 softmax） | ≈ 3450 |
| softmax WG：拿到 S → arrive `p_ready` | ≈ 1900–2200 |
| 　其中 128 个 `exp2`（含求和） | ≈ 1450–1700 |
| 一个 tile 的 PV + QK | ≈ 1100 |

Tensor Core 和 MUFU 都只忙约 57%：两边都没满，是一个 tile 的链（softmax → PV → QK → softmax）太长。链上最长的一段是指数。

FA4（Zadouri、Hoehnerbach、Shah、Liu、Thakkar、Dao，2026，*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*）对此的分析是：Blackwell 的 Tensor Core 吞吐比 Hopper 翻了一倍，MUFU 没有变，于是“非矩阵乘”部分的比重变大了。论文给出两个办法，正是本关的内容：

| 问题 | FA4 的办法 | 本关步骤 |
|---|---|---|
| MUFU 吞吐（每 SM 每周期 16 个 `exp2`）跟不上 Tensor Core | 一部分 `exp2` 用 FMA 上的多项式算（Cody–Waite 范围约简 + 多项式） | 1、2 |
| 每块都要把 O 从 TMEM 读出、乘 α、写回 | 最大值变化不大时不修正（“conditional softmax rescaling”） | 3 |

## 3. 数学

本关的记号与 40 关相同：`s` 是原始分数（`Q·Kᵀ` 的一个元素），`scale_log2 = scale · log₂e`，一个元素的指数参数是

```
x = s · scale_log2 − m · scale_log2        （m：当前基准，raw 分数单位）
p = 2^x
```

### 3.1 一块 softmax 的吞吐下限

一个 softmax warpgroup 每块处理 128 行 × 128 列 = 16384 个指数；一个 CTA 有两个 tile，所以每块 32768 个。B200 的 MUFU **每 SM 每周期 16 个** `ex2`（§4.1 实测），于是

```
MUFU 下限 = 32768 / 16 = 2048 周期 / 块（两个 tile）
Tensor Core（D = 128）：每 tile 每块 PV + QK = 2 × 128×128×128×2 FLOP = 8.4 MFLOP，
                        两个 tile 16.8 MFLOP / 每 SM 每周期 8192 FLOP ≈ 2048 周期
```

D = 128 时两者恰好一样多；D = 64 时 Tensor Core 只要约 1024 周期，MUFU 仍是 2048——**softmax 更是瓶颈**。只要指数全在 MUFU 上，一块就不可能快于 2048 周期，何况 40 关里一块是约 3450 周期（链的延迟）。

FMA 流水线的吞吐是每 SM 每周期 128 个 FP32 FMA（§4.1），是 MUFU 的 8 倍。如果用约 8 条 FMA 级的指令算出一个 `exp2`，FMA 流水线就能提供和 MUFU 差不多的指数吞吐，两者加起来约翻倍。

### 3.2 范围约简：`2^x = 2^n · 2^f`

任何实数 x 可以写成 `x = n + f`，`n = ⌊x⌋` 是整数，`f ∈ [0, 1)`：

```
2^x = 2^n · 2^f,      2^f ∈ [1, 2)
```

- `2^f` 在一个固定的小区间上，可以用多项式逼近（§3.3）；
- 乘 `2^n` 不需要乘法：FP32 的值是 `(−1)^sign · 1.mantissa · 2^(exponent − 127)`，`2^f ∈ [1, 2)` 的 exponent 字段恰好是 127，把 n 加到 exponent 字段（位 23–30）上就得到 `2^n · 2^f`。

手算：x = −2.75。n = −3，f = 0.25，`2^0.25 ≈ 1.1892`，位模式 `0x3F9837F0`；加 `−3 << 23 = 0xFE800000`（32 位补码）得 `0x3E1837F0` = 0.14865 = 2^−2.75。✓

#### 不用 `floorf` 求 n：加一个“魔数”

`floorf(x)` 在 SASS 中是 `FRND.FLOOR`，它和 MUFU 一样每 SM 每周期只有 16 个（§4.1 实测）——用它就白费了。FA4 用一个 FP32 加法：

```
r = x + 1.5 · 2^23     （向下舍入，add.rm）
```

`1.5 · 2^23 = 12582912` 在 `[2^23, 2^24)` 之间，这个范围的 FP32 相邻两个数相差恰好 1，所以 r 是一个整数，向下舍入保证 `r = 12582912 + ⌊x⌋`（只要 |x| < 2^22）。它的位模式是 `0x4B400000 + ⌊x⌋`：**n 就躺在 r 的低位里**（补码，借位进不到高位，因为有 1.5 中的 0.5 垫着）。于是：

```
n_float = r − 12582912        精确（同一个 binade 里的两个整数相减）
f       = x − n_float         x ≥ 0 或 x ≤ −0.5 时精确；x ∈ (−0.5, 0) 时 f 接近 1，
                              舍入的绝对误差 ≤ 2^−25，远小于多项式的误差
bits(2^x) = bits(p) + (bits(r) << 23)
```

最后一行：`bits(r) << 23` 把 r 的低 9 位移进 exponent 字段，高位（`0x4B4…` 那部分）全部移出 32 位，剩下的恰好是 `n << 23`（模 2^32）。

#### 边界：−∞ 和很小的 x

mask 掉的 key 的 s 是 −∞，x = −∞。软件 `exp2` 先做 `x = max(x, −127)`：

- x = −127：n = −127，f = 0，`p(0) = c0`。若 c0 恰好是 1.0（位模式 `0x3F800000 = 127 << 23`），加上 `−127 << 23` 正好是 **0x00000000 = +0**。若 c0 = 0.9999（位模式比 `0x3F800000` 小），相加后借位到符号位，得到一个**负的 NaN 或负数**。所以多项式的常数项被固定为 1（§3.3）。
- −127 < x < −126：拼出来的是次正规数 (p − 1)·2^−126，**并不等于** 2^x（次正规数没有隐含的 1）。但它 < 2^−126 ≈ 1.2·10⁻³⁸，与行和（≥ 1）相比可以忽略，转成 BF16/FP16 后也是 0 或次正规数，所以错了也无妨。
- x 的上限：40 关里 x ≤ 0；步骤 3 的条件 rescale 让 x ≤ τ = 8（§3.5）。n ≤ 8 不会让 exponent 字段溢出（要求 n ≤ 127）。

### 3.3 多项式与误差预算

**误差预算由 P 的数据类型决定。** P 在写回 TMEM 时被舍入成 16 位，这一步的相对误差是单位舍入 u：

| P 的类型 | 有效位（含隐含位） | u = 2^−p |
|---|---|---|
| BF16 | 8 | 2^−8 ≈ 3.9·10⁻³ |
| FP16 | 11 | 2^−11 ≈ 4.9·10⁻⁴ |

多项式的最大相对误差 ε 应明显小于 u，P 的误差才仍由舍入主导。同一份代码要同时服务 FP16，所以按 FP16 的 u 选。

**系数怎么来：** 在 `[0, 1)` 上找 `p(f) = 1 + c1·f + … + cd·f^d`（常数项固定为 1）使**相对**误差 `max |p(f)/2^f − 1|` 最小——这是一个 minimax（极小化最大误差）问题，Sollya 的 `fpminimax` 或 Remez 算法都能解。本关用 numpy 的 Lawson 迭代（加权最小二乘，权重按误差反复放大）在 20000 个点上求解，再把系数舍入成 FP32。结果（FA4 `flash_attn/cute/utils.py` 的 `POLY_EX2` 用同样的形式，系数相差在第 3 位）：

| 次数 d | 系数 c1, c2, …（FP32） | 最大相对误差（实测） | 与 FP16 的 u 相比 |
|---|---|---|---|
| 2 | 0.66596067, 0.32993290 | 2.05·10⁻³ = 2^−8.9 | 4.2 u：不够 |
| **3** | **0.69511682, 0.22764485, 0.07706717** | **8.57·10⁻⁵ = 2^−13.5** | **0.18 u** |
| 4 | 0.69304484, 0.24128021, 0.05224245, 0.01342670 | 2.97·10⁻⁶ = 2^−18.4 | 0.006 u |
| `MUFU.EX2`（对照） | — | 1.46·10⁻⁷ = 2^−22.7 | — |

“实测”是对 `[−127, 8]` 内**每一个** FP32 输入（约 22 亿个）在 GPU 上算软件 `exp2`，与 FP64 的 `exp2` 比较（输出为正规数的部分取相对误差；输出为次正规数的部分只检查结果非负且小于 2^−126）。3 次多项式的误差就是 `[0, 1)` 上多项式本身的误差：范围约简只在 x ∈ (−0.5, 0) 时引入 ≤ 2^−25 的绝对误差，指数拼接是精确的。

选 **3 次**：对 BF16 它比舍入误差小 45 倍，对 FP16 小 5 倍；每个指数只要 3 条 FMA。2 次对 BF16 其实够用（0.5 u），但对 FP16 不够。FA4 也用 3 次。

### 3.4 对 O 和 LSE 的影响

P 的每个元素有相对误差 εᵢ（|εᵢ| ≤ 8.6·10⁻⁵ + 舍入）。O = Σ pᵢ(1+εᵢ) vᵢ / Σ pᵢ(1+εᵢ)，分子分母用的是同一批带误差的 p（ℓ 用的是舍入前的 FP32 p，差别只在 16 位舍入），所以 O 的误差约为 `max|εᵢ| · max|vᵢ|` 量级，LSE 的误差约为 `max|εᵢ|`（ln(1+ε) ≈ ε）。两者都远小于本关的容差（O：FP16 `2e-3 + 5e-3·|ref|`，BF16 `1e-2 + 2e-2·|ref|`；LSE：`1e-3 + 1e-4·|ref|`）。§7.3 给出实测。

### 3.5 条件 rescale 为什么是精确的

online softmax 中 m 有两个作用：（a）让指数不溢出；（b）作为 `ℓ`、O 的公共基准。最终输出

```
O   = Σ_j 2^(x_j) v_j / Σ_j 2^(x_j)，   x_j = s_j·scale_log2 − m·scale_log2
LSE = m·scale + ln ℓ
```

里，**m 是什么值都不影响 O**（分子分母同乘 `2^(−m·scale_log2)`，约掉），也不影响 LSE（`m·scale + ln(Σ e^(scale(s_j − m))) = ln Σ e^(scale·s_j)`）。唯一的要求是：**同一行的 ℓ、O、P 始终用同一个基准**，基准改变时三者一起乘 α。

所以基准不必是真正的最大值。条件 rescale 的规则（FA4 的 `rescale_threshold`）：

```
m_new = max(m, 本块的行最大值)
若 (m_new − m)·scale_log2 > τ：  α = 2^((m − m_new)·scale_log2)，m ← m_new   （移动基准，修正 O、ℓ）
否则：                          α = 1，m 不变                                （P 用旧基准）
```

**不变式：** 任何时刻 `真最大值 − m ≤ τ / scale_log2`，即 `x ≤ τ`，所以每个 `p ≤ 2^τ`。

**不溢出的条件**（τ = 8，`p ≤ 256`）：

| 量 | 上界 | 类型的上限 |
|---|---|---|
| P（写成 16 位） | 256 | FP16 65504 ≈ 2^16，BF16 ≈ 2^128 |
| ℓ（FP32） | 256 · N，N = 10⁶ 时 ≈ 2.6·10⁸ | FP32 ≈ 3.4·10³⁸ |
| O 的累加（FP32） | 256 · N · max|v| | 同上 |

**精度不变：** 浮点数的相对精度与大小无关，P 乘上 2 的幂（`p ∈ [2^−τ·…, 2^τ]`）不改变它的相对舍入误差；若要说影响，只是更少的小 P 被下溢成 0。也不改变多项式的误差：f 和 p(f) 的计算与 n 无关。

**τ 不能太大：** FP16 的 P 必须 ≤ 65504，所以 τ < 16；软件 `exp2` 要求 x ≤ 127。FA4 对 16 位输入取 τ = 8（`flash_fwd_sm100.py`，`rescale_threshold = 8.0`），并断言 `max_offset + τ < log₂(dtype 最大值)`。

**手算：** 一行在块 0 的最大值（log₂ 单位）是 3.0，块 1 是 9.5，块 2 是 12.0。
- 块 0：m = −∞ → 差 = +∞ > 8，移动：基准 3.0，α = 0；
- 块 1：差 = 6.5 ≤ 8，不动：α = 1，这块最大的 p = 2^6.5 ≈ 90.5；
- 块 2：差（与基准 3.0 比，不是与上一块的 9.5 比）= 9.0 > 8，移动：基准 12.0，α = 2^−9。

如果错误地与“上一块的最大值”比（块 2：12 − 9.5 = 2.5，不动），基准会一路落后，p 最终超过 FP16 的上限——`growing` case 专门检查这一点。

**省掉了什么：** 对随机数据，S 的一行在 log₂ 单位下的标准差约 1.4（scale = 1/√D、Q、K 为标准正态时），一块的最大值比全行最大值通常只差 1–2；块 0 之后最大值再上涨 8 的概率可以忽略。于是**块 0 之后几乎不再修正 O**：每个 tile 每块省掉 D 列的 `tcgen05.ld` + D 次 FMUL + D 列的 `tcgen05.st`（每个 correction 线程），correction warp 腾出的发射槽和 TMEM 带宽留给了同一个 SMSP 上的 softmax warp。

### 3.6 混合比例：两条流水线的平衡点

一个 SMSP（SM 的四分之一，一个 warp 调度器）上，每个 warp 处理一个元素（32 个 lane 各一个）要占用的流水线周期（§4.1 实测换算）：

| 路径 | MUFU 流水线 | FMA 流水线 |
|---|---|---|
| `MUFU.EX2` | 8（每 SMSP 每周期 4 个） | — |
| 多项式（打包，每个元素） | — | ≈ 7.4（每 SM 每周期 17.4 个结果） |
| 其余 softmax（`FFMA2` 算 x、`FADD2` 求和，每个元素） | — | ≈ 2 |

设比例 φ 的元素走多项式，每个元素的时间约为

```
T(φ) = max( 8 · (1 − φ),  2 + 7.4 · φ )
平衡点：8 − 8φ = 2 + 7.4φ  →  φ ≈ 0.39，T ≈ 4.9（全 MUFU 时 8）
```

这个模型只算了两条流水线的吞吐，还没算：每个 SMSP 每周期只能发射一条指令（多项式每对元素约 10 条指令）；每个 SMSP 上只有两个 softmax warp 加一个 correction warp，多项式 7 条相互依赖的指令（每条约 4 周期延迟）很难被别的 warp 掩盖；多项式需要的常数和临时值会挤占寄存器。所以模型给出的只是上限的方向，真正的最优点要实测（§7.2）。本答案的扫描中 k = 3–5（φ ≈ 19–31%）的速度在测量波动内相同，但 k ≥ 4 时 softmax 的寄存器开始溢出（§5.4），所以取不溢出的 k = 3（φ ≈ 19%）。寄存器溢出让扫描无法检验模型的平衡点 φ ≈ 0.39 本身是否成立。FA4 的默认配置在 12–19% 之间（`ex2_emu_freq`，§9）。

## 4. 硬件与指令

### 4.1 实测：各种指令每 SM 每周期多少个结果

B200 上的实测（微基准：148 个 SM，每 SM 32 个 warp，每线程 8 条独立的依赖链，`clock64()` 计时；你可以用同样的方法自己写一个来验证）：

| 操作 | SASS | 每 SM 每周期 | 相对 MUFU |
|---|---|---|---|
| `ex2.approx.ftz.f32` | `MUFU.EX2` | 16.0 | 1× |
| `fma.rn.f32` | `FFMA` | 128（125–128） | 8× |
| `fma.rn.f32x2`（每条 2 个 FMA） | `FFMA2` | 128 个 FMA（= 64 条指令） | 8× |
| `cvt.rmi.f32.f32`（floor） | `FRND.FLOOR` | 16 | 1× |
| 软件 `exp2`，标量（§4.3） | 约 9 条 | 14.5 | 0.9× |
| 软件 `exp2`，打包（§4.3） | 每对约 10 条 | 17.4 | 1.09× |
| 混合：25% 多项式 | | 21.3 | 1.33×（仍受 MUFU 限制：16 / 0.75） |
| 混合：50% 多项式 | | 28.9 | 1.81× |
| 混合：75% 多项式 | | 20.8 | 1.3×（受 FMA 限制） |

要点：

- **MUFU 和 FMA 是两条独立的流水线**：混合 50% 时总吞吐接近两者之和（16 + 17.4 = 33.4 的 87%）；
- `FFMA2` 没有提高 FMA 的**运算**吞吐（还是每 SM 每周期 128 个 FMA），但**指令数减半**：每个 SMSP 每周期只能发射一条指令，打包让出的发射槽可以给 MUFU、`FMNMX`、`tcgen05.ld/st` 用；
- 取整（`FRND`）和 MUFU 一样慢，所以范围约简用 §3.2 的加法技巧。

### 4.2 打包的 FP32 指令：`FFMA2`、`FADD2`

SM100 新增了一次处理两个 FP32 的指令（PTX ISA 8.6+，`sm_100` 起）：

```
fma.rn.f32x2  d, a, b, c;     // d.x = a.x*b.x + c.x，d.y = a.y*b.y + c.y；a、b、c、d 为 .b64
add.rm.f32x2  d, a, b;        // 两个加法，向 −∞ 舍入
```

CUDA 12.8 起 `<cuda_runtime.h>` 提供对应的内建函数（`crt/sm_100_rt.h`），操作数是 `float2`：

```cpp
float2 __ffma2_rn(float2 a, float2 b, float2 c);   // FFMA2
float2 __fadd2_rn(float2 a, float2 b);             // FADD2
float2 __fadd2_rd(float2 a, float2 b);             // FADD2.RM（round down）
```

两个分量要在一对相邻的寄存器里（SASS 里写作 `R8.F32x2.HI_LO`）；`make_float2(s[c], s[c+1])` 这样连续的数组元素，ptxas 通常能直接分配成寄存器对。常数操作数（如系数）可以是立即数，两个分量相同。

### 4.3 软件 `exp2` 的代码骨架

下面是一个元素的标量写法，留空的地方就是 TODO(1) 要你填的（打包版本把每个运算换成 §4.2 的 `float2` 内建函数，两个元素一起算）：

```cpp
x = fmaxf(x, -127.f);                         // 1. 钳位（FMNMX）
float r = __fadd_rd(x, 12582912.f);           // 2. 魔数加法，向下舍入（FADD.RM）
float n = r - 12582912.f;                     // 3. ⌊x⌋ 作为浮点数
float f = /* x − n，用一条 FMA 写 */;          //    f ∈ [0, 1]
float p = /* 3 次 Horner，从 c3 开始 */;        // 4. 3 条 FMA
return __int_as_float(__float_as_int(p) + /* r 的位模式怎样变成 n << 23？ */);   // 5.
```

两个细节：

- f 写成 `fma(n, −1, x)`（打包版 `__ffma2_rn(n, make_float2(-1.f, -1.f), x)`）而不是 `x + (−n)`：后者要先对 n 的两个分量各做一次取负（两条标量指令），每对元素多 2 条指令；
- ptxas 把第 5 步的移位加法合成一条 `LEA`（`a + (b << 23)`）。

打包后每对元素：2 `FMNMX` + 2 `FADD2` + 4 `FFMA2` + 2 `LEA` = 10 条指令，其中 6 条在 FMA 流水线上（打包指令各占 2 个周期）。

### 4.4 warp 投票：`__any_sync`

`tcgen05.ld`/`tcgen05.st` 是 `.sync.aligned` 指令：**一个 warp 的 32 个 lane 必须一起执行**（32x32b 形状一次读写 32 个 lane）。所以“这一行要不要修正 O”不能每个线程各自决定，只能以 warp 为单位：只要 32 行中有一行要修正，整个 warp 就一起修正，α = 1 的行乘 1，结果不变（精确）。

`__any_sync(mask, pred)` 返回“warp 中是否有任何一个 lane 的 pred 为真”，所有 lane 得到同一个结果，编译成一条 `VOTE.ANY`。FA4 的 `correction_loop` 用的是同样的写法（`vote_ballot_sync(scale < 1.0)`）。

### 4.5 常见误用

| 误用 | 后果 |
|---|---|
| 用 `floorf` / `__float2int_rd` 求 n | 结果对，但 `FRND`/`F2I` 和 MUFU 一样每 SM 每周期 16 个：步骤 1 更慢 |
| 魔数加法用默认的就近舍入 | r 是最近的整数，f ∈ [−0.5, 0.5]：多项式在拟合区间外，`exp2` 本身的最大相对误差约 1.05·10⁻²（在 f = −0.5 处）；平均到 LSE 上约 2–3·10⁻³，超出容差 |
| 常数项写成拟合出的 0.99998… | x = −∞（mask）时得到负数或 NaN：`causal_*`、`short_tail` 等有 mask 的 case 出 NaN |
| 忘了 `max(x, −127)` | x = −∞ 时 r = −∞，位运算得到垃圾：有 mask 的 case（如 `n1`）错 |
| 每个线程按自己的 α 决定是否 `rescale_o_row` | `tcgen05.ld` 在分叉的 warp 里执行：未定义行为（挂起或结果错） |
| 条件判断和上一块的最大值比，而不是和当前基准比 | 基准一路落后，P 超出 FP16 的上限：`growing` 出 inf/NaN |
| 不移动基准时仍把 α 算成 `2^((m − m_new)·…)` | O 被修正、基准没动：O 和 ℓ 不一致 |

## 5. 调度设计

### 5.1 不变的部分

warp 分工、TMEM（S⁰/P⁰、S¹/P¹、O⁰、O¹，512 列）、shared memory（Q⁰、Q¹、K/V 环、统计量）、全部 mbarrier 及其 parity 与 40 关步骤 3 **完全相同**，见 40 关 §5。本关只改 softmax warpgroup 里“求 P”的几十行，以及 correction warpgroup 里“要不要修正”的一个条件。

```
16 个 warp：
  warp 0–3   softmax WG 0（tile 0）   setmaxnreg 192
  warp 4–7   softmax WG 1（tile 1）   setmaxnreg 192
  warp 8–11  correction WG            setmaxnreg 80
  warp 12    MMA   warp 13 TMA   warp 14–15 空闲   setmaxnreg 48
```

### 5.2 softmax 一块里的指令流（步骤 2–3）

```
wait s_full[t]; tcgen05.ld × 4（S 的一行，128 个 FP32）
mask（只在需要的块）
row max：127 条 FMNMX（8 路部分最大值）
m、α（步骤 3：带阈值的判断，见 §3.5）；α → shared；arrive stats_full[t]
对 c = 0, 2, …, 126：                                  ← 64 次，完全展开
    x = FFMA2(s[c..c+1], scale_log2, −base)
    第 (c % 32)/2 对是否“多项式对”（编译期决定）：
        是：exp2_poly2(x)                   10 条，FMA/ALU
        否：MUFU.EX2 × 2
    sum[(c/2) % 4] = FADD2(sum, p)
ℓ = α·ℓ + Σ
pack 成 16 位，tcgen05.st × 4，wait::st；fence；arrive p_ready[t]
```

每 32 列中哪几对用多项式（`use_poly<kPolyPairs>(q)`，q = 0..15）在编译期决定，均匀分布在 32 列里（不要全挤在一端）。这样 MUFU 指令和多项式指令在指令流中交错，调度器每个周期都能找到一条可以发射到空闲流水线的指令。答案的 SASS（`./lab sass`）：步骤 2–3 每个 kernel 有 105 条 `MUFU.EX2`（104 个 P + 1 个 α）、112 条 `FFMA2`（64 条算 x + 12 对 × 4 条多项式：f 与 3 步 Horner）；步骤 1 只有 1 条 `MUFU.EX2`（α）、320 条 `FFMA2`。

### 5.3 correction warp（步骤 3）

```
对 j，对 t = 0, 1：
    wait stats_full[t]（parity j&1）；fence
    j > 0 且 __any_sync(α ≠ 1)：rescale_o_row（ld / ×α / st / wait::st）
    fence；arrive p_ready[t]                 ← 跳过修正时也必须 arrive
```

跳过修正时**仍然**要 arrive `p_ready`：MMA warp 等的是 256 次 arrive（softmax 128 + correction 128），少一次就挂起。同步的论证与 40 关 §5.6 完全相同：correction 读 α 前等 `stats_full`，修正（或跳过）之后 arrive `p_ready`。

### 5.4 寄存器

softmax 线程的上限仍是 192（`setmaxnreg`）。整行 S（128 个 FP32）放在寄存器里，再加上多项式的系数、魔数、部分和。实测：每 16 对中 4 对及以上用多项式时，ptxas 在 192 个寄存器里放不下，`./lab sass` 显示 8–16 字节的 `STACK`（溢出），性能明显下降（§7.2）。k = 3 时无溢出。

## 6. 分步练习

骨架中流水线（TMA、MMA、barrier、TMEM、correction 的 epilogue、host 端）全部写好，与 40 关答案相同。`exp_and_sum` 已经写好了打包的循环；`p.step` 选择 `Config<D, 1|2|3>`。三步是同一个 kernel 逐步补全，后面的步骤用到前面的 TODO。

### 步骤 1 · TODO(1)：软件 `exp2`

实现 `exp2_poly2`（§3.2、§4.3）。系数 `kExp2C0..3` 已给出（§3.3 的 3 次多项式）。步骤 1 的配置让**全部** P 的指数走它。

```bash
./lab check 41 --step 1 --case short_tail     # 有 mask：检查 −∞ → 0
./lab check 41 --step 1 --case peaked_fp16    # scale = 1：大量有限的 x < −127，检查钳位
./lab check 41 --step 1
./lab bench 41 --step 1 --case perf           # 预期比 40 关慢（§7.1）
./lab sass 41     # 在 artifacts/sass/ 的完整 SASS 中，步骤 1 的 kernel（Config<D, 1>）只剩 1 条 MUFU.EX2（α）
```

### 步骤 2 · TODO(2)：混合

实现 `use_poly`：16 对中恰好 `kPolyPairs` 对返回 true，均匀分布；再在 `Config` 里选 `kPolyPairs`（从 §3.6 的平衡点出发，用 `bench` 调，注意 `./lab sass` 的 `STACK`）。

```bash
./lab check 41 --step 2
./lab bench 41 --step 2 --case perf
./lab bench 41 --step 2 --case perf_d64
./lab sass 41     # STACK 必须为 0；在完整 SASS 中数 Config<D, 2> 的 MUFU.EX2 与 FFMA2（§5.2）
```

### 步骤 3 · TODO(3a)–(3b)：条件 rescale

- 3a：softmax 里带阈值的基准更新（§3.5）；
- 3b：correction warp 用 `__any_sync` 决定整个 warp 是否修正（§4.4、§5.3）。

```bash
./lab check 41 --step 3 --case growing        # 缓慢增长：和基准比，不是和上一块比
./lab check 41 --step 3 --case jump           # 跳变：必须移动基准
./lab check 41 --step 3
./lab bench 41 --case perf
./lab bench 41 --case perf_causal
./lab bench 41 --case perf_d64
./lab bench 41 --case perf_8k
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 只有带 mask 的 case 出 NaN（`short_tail`、`causal_*`） | 没有钳位到 −127，或常数项不是恰好 1（§3.2） |
| 步骤 1 O 对、LSE 超差（误差约 2–3·10⁻³） | 魔数加法用了就近舍入（`__fadd2_rn`）：f ∈ [−0.5, 0.5]，多项式在区间外，单个 `exp2` 误差可达 10⁻²；系数顺序反了（Horner 从最高次开始）也会这样 |
| 步骤 1 几乎全部错 | 指数位的移位不是 23；`r` 与 `p` 的位模式加反了类型（要用 `__float_as_int` 取位，不是 `(int)x` 转换） |
| 步骤 2 正确但比步骤 1 还慢 | `use_poly` 不是编译期常量（循环里留下了分支）；或 `kPolyPairs` 太大，`STACK` > 0 |
| 步骤 3 `growing` 出 inf/NaN，其他都对 | 与上一块的最大值比较，而不是与当前基准 m 比较 |
| 步骤 3 `jump`、`jump_causal`、`growing` 出 NaN | 从不移动基准（条件写反或只在块 0 移动）；或移动基准时 α 用错 |
| 步骤 3 `TIMEOUT` | 跳过修正的 warp 没有 arrive `p_ready` |
| 步骤 3 挂起或偶发错误 | 按线程而不是按 warp 决定是否调用 `rescale_o_row`（`tcgen05.ld` 必须整个 warp 执行） |
| 步骤 3 `growing`、`growing_causal`、`jump_causal` 错但没有 NaN | 投票后用了某一个 lane 的 α 去乘所有行（应各乘各的 α） |

## 7. 性能

### 7.1 相对库基线

B200 实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时，越大越好）：

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| 40 关步骤 3（全 MUFU，对照） | ≈ 0.85× | ≈ 0.94× | ≈ 0.80× | ≈ 0.83× |
| 步骤 1（全部软件 `exp2`） | ≈ 0.58× | ≈ 0.69× | ≈ 0.61× | ≈ 0.56× |
| 步骤 2（每 32 列 6 列多项式） | ≈ 0.89× | ≈ 0.99× | ≈ 0.94× | ≈ 0.87× |
| 步骤 3（+ 条件 rescale，τ = 8） | ≈ 0.97× | ≈ 1.03× | ≈ 0.96× | ≈ 0.93× |
| FA4（`flash_attn.cute`） | ≈ 1.15× | ≈ 1.33× | ≈ 1.13–1.15× | ≈ 1.03–1.10× |

cuDNN 自身在多次运行之间有几个百分点的波动；同一行的数字来自同一次 `bench`。

- **步骤 1 比 40 关慢约 30%**：全部指数挪到 FMA 流水线（MUFU 几乎闲着，`xu` 0.3%），每对元素 10 条指令，FMA/ALU 流水线和发射槽成了新瓶颈——单独的软件 `exp2` 只比 MUFU 快 9%（§4.1），而 softmax 的其他指令也要用 FMA。
- **步骤 2** 只把约 19% 的指数挪过去，MUFU 和 FMA 同时工作：这是本关最大的一步。D = 64 的收益更大，因为它的 softmax 相对更重（§3.1）。
- **步骤 3** 在随机数据上几乎完全省掉了块 0 之后的 O 修正（§3.5），又快了 2–9%。对照实验：只保留 warp 投票、τ = 0（最大值完全不变才跳过）时 `perf` ≈ 0.91×、`perf_d64` ≈ 0.94×，只比步骤 2 好一点——随机数据上一个 warp 的 32 行在同一块里**全部**不涨的概率不高；τ = 8 才让修正真正消失。

本关的目标：步骤 3 在四个 perf case 上都达到 cuDNN 的约 0.85 倍以上。

### 7.2 混合比例的扫描

步骤 2，每 32 列中多项式的对数 k（φ = k/16）：

| k | φ | `perf` | `perf_d64` | `./lab sass` 的 STACK |
|---|---|---|---|---|
| 0（全 MUFU，但用 `FFMA2`/`FADD2`） | 0 | ≈ 0.81× | ≈ 0.82× | 0 |
| 2 | 12.5% | ≈ 0.87× | ≈ 0.91× | 0 |
| **3（答案）** | **18.75%** | **≈ 0.90×** | **≈ 0.93×** | **0** |
| 4 | 25% | ≈ 0.89× | ≈ 0.95× | 4 个 kernel 8 字节 |
| 5 | 31% | ≈ 0.89× | ≈ 0.95× | 6 个 kernel 8 字节 |
| 6 | 37.5% | ≈ 0.81× | ≈ 0.83× | 16 字节 |
| 8 | 50% | ≈ 0.72× | ≈ 0.74× | 24 字节 |
| 16（步骤 1） | 100% | ≈ 0.58× | ≈ 0.61× | 0 |

- k = 0 是“40 关 + 打包的 `FFMA2`/`FADD2`”：它和 40 关不是同一次 bench 测的，只能说两者大致相当；瓶颈仍是 MUFU；
- k = 3、4、5 三行用答案的最终代码测（f 用一条 `FFMA2`，§4.3）；其余行是更早一版代码（f = x + (−n)，每对多 2 条标量取负指令）测的，那一版 k = 4 只有 0.82×/0.88×、溢出也更多。对比说明：每对元素省两条指令，就让更大的 k 不再吃亏；
- 最终代码中 k = 3–5 的差别在测量波动内；k ≥ 4 时 `./lab sass` 显示部分 kernel 的 `STACK`（softmax 线程 192 个寄存器放不下，溢出到 local memory），所以答案取不溢出的 k = 3。**扫描时一定同时看 `STACK`**；
- 因为溢出从 k = 4 就开始，这个扫描不能说明 §3.6 的模型（φ ≈ 0.39）对或错，只能说明在 192 个寄存器的约束下 φ ≈ 19–31% 都差不多；
- 本表每格只测了一次，相邻的 k 差 1–2% 以内的不要过度解读。

### 7.3 误差分析

每种变体相对 FP64 oracle 的最大误差（`./lab check` 打印的 `max|err|`，O 的绝对误差；同一份输入）：

| case | 40 关（全 MUFU） | 步骤 1（全多项式） | 步骤 2（混合） | 步骤 3（混合 + 条件 rescale） |
|---|---|---|---|---|
| `medium` BF16 D=128 | 8.4·10⁻⁴ / 1.3·10⁻⁶ | 8.4·10⁻⁴ / 1.6·10⁻⁵ | 8.4·10⁻⁴ / 7.5·10⁻⁶ | 1.3·10⁻³ / 7.5·10⁻⁶ |
| `medium_d64` FP16 | 1.2·10⁻⁴ / 1.4·10⁻⁶ | 1.3·10⁻⁴ / 1.6·10⁻⁵ | 1.2·10⁻⁴ / 6.1·10⁻⁶ | 1.5·10⁻⁴ / 7.0·10⁻⁶ |
| `peaked` BF16 | 1.3·10⁻² / 1.6·10⁻⁵ | 1.3·10⁻² / 9.4·10⁻⁵ | 1.3·10⁻² / 9.4·10⁻⁵ | 1.5·10⁻² / 9.4·10⁻⁵ |
| `peaked_fp16`（scale = 1，LSE 数百） | 1.4·10⁻³ / 1.9·10⁻⁴ | 1.4·10⁻³ / 2.4·10⁻⁴ | 1.4·10⁻³ / 2.4·10⁻⁴ | 2.0·10⁻³ / 2.4·10⁻⁴ |
| `causal_medium` BF16 | 8.0·10⁻³ / 1.2·10⁻⁶ | 8.0·10⁻³ / 8.6·10⁻⁵ | 8.0·10⁻³ / 3.0·10⁻⁵ | 8.0·10⁻³ / 3.0·10⁻⁵ |
| `growing` FP16 | 4.3·10⁻⁴ / 1.1·10⁻⁵ | 4.3·10⁻⁴ / 5.9·10⁻⁵ | 4.3·10⁻⁴ / 6.1·10⁻⁵ | 4.8·10⁻⁴ / 6.1·10⁻⁵ |
| `jump_causal` BF16 | 7.3·10⁻³ / 7.9·10⁻⁵ | 7.3·10⁻³ / 1.3·10⁻⁴ | 7.3·10⁻³ / 1.2·10⁻⁴ | 7.3·10⁻³ / 1.2·10⁻⁴ |
| `ones_v` FP16 | 2·10⁻¹⁵ / 1.1·10⁻⁶ | 2·10⁻¹⁵ / 1.9·10⁻⁵ | 2·10⁻¹⁵ / 8.4·10⁻⁶ | 2·10⁻¹⁵ / 8.4·10⁻⁶ |

每格是 “O 的最大绝对误差 / LSE 的最大绝对误差”（同一份输入分别运行四个 kernel）。容差：O 为 FP16 `2e-3 + 5e-3·|ref|`、BF16 `1e-2 + 2e-2·|ref|`，LSE 为 `1e-3 + 1e-4·|ref|`（`peaked` 的 O 大于 1，所以 1.5·10⁻² 仍在容差内）。

- **O**：误差由 O 写回 16 位时的舍入主导，四种变体几乎相同；`ones_v` 的 O 恰好是 1：分子用 16 位的 P、分母 ℓ 用 FP32 的 p，两者的差（≤ 16 位舍入）在 O 写回 FP16 时被舍入吸收；
- **LSE**：从全 MUFU 的约 10⁻⁶ 增加到约 10⁻⁵–10⁻⁴，正好是多项式的相对误差 8.6·10⁻⁵ 的量级（LSE 的误差 ≈ ε，§3.4）。混合时只有 19% 的元素带这个误差，平均下来更小。仍比 LSE 容差 10⁻³ 小 10 倍以上；
- 条件 rescale 不改变误差的量级（§3.5：相对精度与大小无关）；
- 容差与 40 关完全相同，没有放宽。

### 7.4 看哪些指标

```bash
./lab profile 41 --case perf
```

Nsight Compute（锁频，只用来比较指标；`perf` case）：

| 版本 | `gpu__time_duration` | Tensor Core 忙（`sm__pipe_tensor_cycles_active`） | MUFU（`sm__inst_executed_pipe_xu`） | FMA 流水线（`pipe_fma`） | `smsp__inst_executed_pipe_xu.sum` |
|---|---|---|---|---|---|
| 40 关步骤 3 | 893 µs | 57% | 57% | — | 3382 万 |
| 步骤 1 | 1185 µs | 40% | 0.3% | 32% | 27 万（只剩 α） |
| 步骤 2 | 864 µs | 60% | 49% | 24% | 2753 万（= 104/128 + α） |
| 步骤 3 | 812 µs | 65% | 53% | 18% | 2753 万 |

`perf_d64`：40 关 1488 µs、Tensor Core 30%、MUFU 61%；步骤 3 1287 µs、Tensor Core 36%、MUFU 59%。

- `smsp__inst_executed_pipe_xu.sum` 验证混合比例：步骤 2–3 是 40 关的 81%（每行 104 个 MUFU 指数 + 1 个 α，对 128 + 1）；
- 步骤 3 的 Tensor Core 忙碌比例升到约 65%，但 MUFU 仍只有约一半时间在忙：两条流水线都没满，链的延迟（40 关 §3.1）仍然是限制。FA4 进一步缩短链的办法（2-CTA、更好的调度）在 42、43 关。

## 8. 自测问题

1. D = 64 时，一块（两个 tile）的 MUFU 下限和 Tensor Core 下限各是多少周期？为什么本关在 `perf_d64` 上的收益比 `perf` 大？
2. x = −0.3 时，`r = x + 1.5·2^23`（向下舍入）的位模式是什么？n、f 各是多少？最后要给 p 的位模式加多少？
3. 为什么多项式的常数项必须恰好是 1.0？如果用拟合出的 c0 = 0.99999，x = −∞ 时结果的位模式是什么？
4. 某一行连续 5 块的最大值（log₂ 单位）是 0、5、9、14、30，τ = 8。每块的 α 和基准各是多少？这一行最大的 P 出现在哪一块，是多少？
5. 若把 τ 提高到 20，FP16 和 BF16 分别会出什么问题？
6. 为什么 correction warp 跳过修正时仍然要 arrive `p_ready`？为什么“要不要修正”必须整个 warp 一起决定？

<details>
<summary>参考答案</summary>

1. MUFU：32768 / 16 = 2048 周期（与 D 无关）；Tensor Core：两个 tile 的 PV + QK = 4 × 128×128×64×2 FLOP = 8.4 MFLOP，/ 8192 ≈ 1024 周期。D = 64 时 softmax 占比更大，把指数分一部分给 FMA 的收益更大。
2. x + 12582912 = 12582911.7，向下舍入得 12582911 = 12582912 − 1，位模式 `0x4B3FFFFF`；n = −1，f = 0.7。`bits(r) << 23` = `0xFF800000`（即 −1 << 23），加到 p ≈ 1.6245 的位模式上，得到 0.81225 = 2^−0.3。
3. x 钳位成 −127，f = 0，p = c0，要求 `bits(c0) + (−127 << 23) = 0`，即 `bits(c0) = 0x3F800000`。c0 = 0.99999 的位模式是 `0x3F7FFF58`，加上 `0xC0800000` 得 `0xFFFFFF58`：一个 NaN。
4. 块 0：基准 0，α = 0；块 1：5 − 0 = 5 ≤ 8，不动，α = 1；块 2：9 > 8，基准 9，α = 2^−9；块 3：14 − 9 = 5，不动，α = 1；块 4：21 > 8，基准 30，α = 2^−21。最大的 P 在块 3：2^5 = 32（块 1 也是 32）。
5. P 可达 2^20 > 65504：FP16 的 P 溢出成 inf，O、ℓ 出 NaN 或偏小；BF16 不溢出（上限 2^128），结果仍对（但 FA4 的断言 `τ < log₂(dtype max)` 对 FP16 不成立）。
6. MMA warp 等 `p_ready` 的 256 次 arrive，correction 少 arrive 就永远凑不齐：挂起。`tcgen05.ld/st` 是 warp 级的 `.sync.aligned` 指令，必须 32 个 lane 一起执行，所以只能按 warp 决定。
</details>

## 9. 延伸阅读

- Zadouri 等，*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*（2026）：软件指数（多项式模拟 `exp2`）与条件 softmax rescaling 两节。
- 上游实现（`third_party/flash-attention`，提交 `1fc0221`）：
  - `flash_attn/cute/utils.py`：`POLY_EX2`（各次数的系数）、`ex2_emulation_2`（打包的范围约简：`fmax(x, −127)`、`add_packed_f32x2(..., rnd="rm")`、`combine_int_frac_ex2` 的 `shl 23` + `add`）；
  - `flash_attn/cute/softmax.py`：`SoftmaxSm100.apply_exp2_convert`（`ex2_emu_freq`、`ex2_emu_res`、`ex2_emu_start_frg`：每 32 个元素中哪些走多项式）、`update_row_max`（`rescale_threshold`）；
  - `flash_attn/cute/flash_fwd_sm100.py`：`_TUNING_CONFIG`（各配置的 `ex2_emu_freq`）、`rescale_threshold = 8.0` 与 Note [Low Precision Scaling]、`correction_loop` 中的 `vote_ballot_sync(scale < 1.0)`。
- PTX ISA：[ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2)、[fma（含 `.f32x2`）](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)、[add（含 `.f32x2` 与舍入模式）](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)、[vote.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync)。
- CUDA C++ Best Practices Guide 的算术指令吞吐表（[Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#arithmetic-instructions)）；本关 §4.1 的数字是实测的，以实测为准。
- 多项式逼近：J.-M. Muller，*Elementary Functions: Algorithms and Implementation*（范围约简、minimax 逼近）；Sollya 的 `fpminimax`（https://www.sollya.org/）。
- 后续：42 关（2-CTA attention）。
