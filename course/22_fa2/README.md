# 22 · FlashAttention-2：Tensor Core、循环交换与 warp 划分

> 部分 2 · FA2　｜　前置：[12_fa1](../12_fa1/README.md)、[20_mma_gemm](../20_mma_gemm/README.md)、[21_pipeline_gemm](../21_pipeline_gemm/README.md)　｜　代码：[`students/22_fa2/kernel.cu`](../../students/22_fa2/kernel.cu)

## 1. 本关目标

用 `mma.sync` 写出 FlashAttention-2 的前向 kernel：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]` 布局，D = 64 或 128，causal 和非 causal 都支持，输出 O 和 FP32 的 LSE。

12 关的 FA1 用 SIMT FP32 实现。本关把矩阵乘换成 20、21 关学过的 Tensor Core，并按 FA2 论文第 3 节改造整个算法：

| | 12 关（FA1） | 本关（FA2） |
|---|---|---|
| 循环顺序 | K/V 块在外层，Q 块在内层 | **Q 块在外层**，K/V 块在内层 |
| O、ℓ、m 存在哪 | HBM，每个块读写一次 | **寄存器**，最后写一次 |
| 归一化 | 每个块都除以 ℓ | **最后除一次** |
| 并行 | 一个 (batch, head) 一个 CTA | **一个 (Q 块, head, batch) 一个 CTA** |
| CTA 内分工 | 线程分 S 的行和列 | **每个 warp 独占 16·k 行 query** |
| P 怎样进入 P·V | 写进 shared memory | **留在寄存器，直接作 mma 的 A 操作数** |

做完本关你应该能：

- 解释 FA2 的三个改动各省掉了什么（HBM 往返、非矩阵乘运算、warp 间通信）；
- 从 `m16n8k16` 的 fragment 表推出“C fragment 可以直接当 A fragment 用”；
- 说清楚 V 为什么要用 `ldmatrix.trans` 读；
- 在 causal 下只算需要的块，只给需要的块加 mask；
- 解释为什么本关在 B200 上只有 cuDNN 的约 0.26–0.4 倍，以及剩下的差距要靠什么补。

## 2. 动机：FA1 在 Tensor Core 上暴露的三个问题

12 关 §7.2 的结论是：在 SIMT FP32 上，FA1 多出的 HBM 往返不是瓶颈，计算才是。换成 Tensor Core 以后，矩阵乘快了十几倍，原来藏在计算后面的开销都露了出来。FA2 论文（Dao 2023）第 3 节针对的就是这三点：

1. **O、ℓ、m 的 HBM 往返**（论文 3.1、3.2 节）。FA1 的 K/V 外层循环让每个 Q 块的状态在 HBM 里进出 Tc 次。把 Q 放到外层，一个 CTA 处理完一个 Q 块的全部 K/V，状态就能一直留在寄存器里。同时，Q 块之间互不相关，可以按 Q 块再并行一层：12 关的 `few_heads`（B=1、N=4096、H=16）只有 16 个 CTA；同样的 B、N、H 在本关是 16 个 head × 32 个 Q 块（D=64 时每块 128 行）= 512 个 CTA。
2. **非矩阵乘运算**（论文 3.1 节）。Tensor Core 的吞吐比普通 FP32 单元高一个数量级，所以“每个 O 元素多乘一次、多除一次”都不便宜。FA2 让 O 在循环中保持**未归一化**，最后只除一次；反向传播只需要 LSE，不再分别保存 ℓ 和 m。
3. **warp 之间的通信**（论文 3.3 节）。FA1 的 CUDA 实现把 K/V 分给不同 warp（“split-K”），每个 warp 算出部分 S·V 后，要经过 shared memory 和同步把各 warp 的结果加起来，softmax 的行最大值也要跨 warp 合并。FA2 改成按 **Q 行**分给 warp：一行的全部 key 都在同一个 warp 里，行最大值和行和只需要 warp 内的 shuffle，PV 的结果也不用合并。

## 3. 数学

### 3.1 循环交换后的不变式

一个 CTA 负责 query 行 `q0 .. q0+Br−1`（下面只看其中一行 i）。它按顺序处理 K/V 块 `j = 0, 1, …`，每个块有 Bc 个 key。处理完前 j 个块后，寄存器里保存：

```
m  = max_{k 已处理} s_k                         （行最大值，s_k = q_i · k_k）
ℓ  = Σ_{k 已处理} exp(scale·(s_k − m))
O~ = Σ_{k 已处理} exp(scale·(s_k − m)) · v_k    （未归一化！）
```

处理新块 j 时，先算新的最大值 `m' = max(m, max_{k∈j} s_k)`，然后：

```
α  = exp(scale·(m − m'))
P~ = exp(scale·(S_j − m'))           每个元素一次 exp
ℓ  ← α·ℓ + rowsum(P~)
O~ ← α·O~ + P~ · V_j                 一次乘法 + 一次 mma
m  ← m'
```

全部块处理完后，`O = O~ / ℓ`，`LSE = scale·m + ln ℓ`。和 12 关的合并定理相比，每个块对每个 O 元素只做一次乘法（乘 α），除法从“每块一次”变成“总共一次”。

### 3.2 手算：一行、两个块

scale = 1，D = 1。块 0 的 key 分数 `s = (2, 0)`、值 `v = (1, 0)`；块 1 的分数 `(4, 2)`、值 `(2, 1)`。

| | m | P~ | ℓ | O~ |
|---|---|---|---|---|
| 块 0 后 | 2 | (1, e⁻² = 0.1353) | 1.1353 | 1·1 + 0.1353·0 = 1 |
| 块 1：α = e^(2−4) = 0.1353 | 4 | (1, 0.1353) | 0.1353·1.1353 + 1.1353 = 1.2889 | 0.1353·1 + 1·2 + 0.1353·1 = 2.2707 |

结果 `O = 2.2707 / 1.2889 = 1.7617`，`LSE = 4 + ln 1.2889 = 4.2538`。直接计算：`e² + e⁰ + e⁴ + e² = 70.376`，`ln 70.376 = 4.2538`；`(e²·1 + e⁴·2 + e²·1)/70.376 = 1.7616`。两者一致（差别来自四舍五入）。

### 3.3 用 2 的幂代替 e 的幂，m 不乘 scale

硬件只有 `2^x`（`ex2.approx`，SASS 中是 `MUFU.EX2`）。因为 `exp(x) = 2^(x·log₂e)`，令 `scale_log2 = scale·log₂e`：

```
exp(scale·(s − m)) = 2^(s·scale_log2 − m·scale_log2)
```

括号里是一条 FFMA（`fmaf(s, scale_log2, −m·scale_log2)`），每行的 `m·scale_log2` 只算一次。

m 保存的是**未乘 scale 的原始分数**的最大值。这只在 scale > 0 时成立：`max(scale·s) = scale·max(s)`。本课程的支持域是 scale > 0 且有限，harness 会拒绝其他值（v1 的答案在 scale ≤ 0 时得到 NaN，就是忽略了这一点）。最后 `LSE = m·scale + ln ℓ`。

**全被 mask 的行。** 如果某行到目前为止看到的 key 全被 mask，`m' = −∞`，`m'·scale_log2 − m'·scale_log2` 是 `(−∞) − (−∞) = NaN`。做法是：`m' = −∞` 时用 0 代替被减数，于是 `α = 2^(−∞) = 0`、`P~ = 2^(−∞) = 0`，状态保持全零。本关的块划分保证真实行不会出现这种情况（块 0 的 key 0 对每一行都可见），但写成防御性的代码，换块大小或加 mask 种类时不会出错。

**ℓ 的 quad 内部分和。** 下一节会看到，一行的 Bc 个分数分在 4 个 lane 上，每个 lane 持有 Bc/4 个。行最大值必须 4 个 lane 一起求（否则 4 个 lane 用不同的 m，P~ 就不再可比）；但 ℓ 不必：只要 4 个 lane 用同一个 α，每个 lane 可以只累加自己那部分 P~，最后再把 4 个部分和加起来，因为 `α·(a+b+c+d) = α·a + α·b + α·c + α·d`。这样每个块少做两次 shuffle。`uniform` case（Q = 0，LSE 应为 ln N）专门检查最后有没有忘了这一步：忘了会得到 ln(N/4)。

## 4. 硬件与指令

本关没有新指令。`mma.sync`、`ldmatrix` 和 swizzle 来自 20 关，`cp.async` 来自 21 关，都由头文件提供：[`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh)、[`include/lab/cp_async.cuh`](../../include/lab/cp_async.cuh)。新的是怎样把它们组合起来。

### 4.1 回顾：m16n8k16 的 fragment

记 `g = lane / 4`（0..7，“group”），`q = lane % 4`（0..3，“quad 内编号”）：

| 操作数 | 寄存器 i | 行 | 列 |
|---|---|---|---|
| A（16×16，每个寄存器两个 16 位值） | 0, 1, 2, 3 | g, g+8, g, g+8 | 2q, 2q, 2q+8, 2q+8（及其 +1） |
| B（16×8，k × n） | 0, 1 | k = 2q, 2q+8（及其 +1） | n = g |
| C/D（16×8，FP32） | 0, 1, 2, 3 | g, g, g+8, g+8 | 2q, 2q+1, 2q, 2q+1 |

C 的每一行（例如第 g 行）由 lane `4g, 4g+1, 4g+2, 4g+3` 共同持有，这 4 个 lane 叫一个 **quad**。行归约只需要在 quad 内 shuffle：

```cpp
x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));   // lane ^ 1：同一 quad
x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));   // lane ^ 2：同一 quad
```

### 4.2 S = Q·Kᵀ：和 GEMM 完全一样

`S[i][j] = Σ_d Q[i][d]·K[j][d]`。对照 GEMM 的 `C = A·Bᵀ`：A = Q（`[行][D]`，D 连续），B = K（`[key][D]`，D 连续），也就是 20 关的“K-major 的 B”。所以 Q 的 A fragment 和 K 的 B fragment 都用普通的 `ldmatrix_x4`，地址公式和 21 关的 `compute_stage` 相同。

不同的是 **Q 只读一次**：一个 warp 的 16 行 × D 列 Q 在整个 K/V 循环中不变，所以在循环开始前把它的 A fragment 全部读进寄存器（D = 128 时每个 lane 32 个寄存器），之后每个块只需要读 K。这是循环交换带来的另一个好处：Q 成了“常驻”的操作数。

### 4.3 P 不经过 shared memory：C fragment 就是 A fragment

P·V 的 A 操作数是 P（16 行 × Bc 个 key）。mma 的 A 是 16×16，所以每次取 16 个 key，也就是**两个相邻的 n8 S tile**（tile 2t 和 2t+1）。把 §4.1 的表逐个寄存器对照：

| A 寄存器 | 需要的元素（行, key） | 在 C fragment 中的位置 |
|---|---|---|
| 0 | (g, 16t+2q), (g, 16t+2q+1) | tile 2t 的 float 0, 1 |
| 1 | (g+8, 16t+2q), (g+8, 16t+2q+1) | tile 2t 的 float 2, 3 |
| 2 | (g, 16t+8+2q), (g, 16t+8+2q+1) | tile 2t+1 的 float 0, 1 |
| 3 | (g+8, 16t+8+2q), (g+8, 16t+8+2q+1) | tile 2t+1 的 float 2, 3 |

右边每一格都在**同一个 lane** 自己的寄存器里：A 和 C 都把行放在 `g = lane/4`、把相邻两列放在 `2q, 2q+1`；A 的 16 列只是两个 C tile 的 8 列并排放。所以只需要把两个 FP32 转成一对 FP16/BF16（`__floats2bfloat162_rn` / `__floats2half2_rn`），不需要任何 shuffle 或 shared memory。

```cpp
a[0] = pack2(s[2t][0],   s[2t][1]);    a[1] = pack2(s[2t][2],   s[2t][3]);
a[2] = pack2(s[2t+1][0], s[2t+1][1]);  a[3] = pack2(s[2t+1][2], s[2t+1][3]);
```

`pack2(x, y)` 把 x 放在低 16 位：低半是列号较小的元素（`mma_sync.cuh` 文件头的约定）。

P 转成 16 位时会舍入（BF16 有 8 位有效二进制位，约 2–3 位有效十进制数字）。ℓ 用的是舍入前的 FP32 值，所以分子分母有微小的不一致。这是上游 FA2 的做法，误差远小于 BF16 输出本身的舍入，`ones_v` case（V = 1，O 应为 1）可以看到这个误差有多小。

### 4.4 V 是 MN-major：`ldmatrix.trans`

P·V 的 B 操作数是 V，k 方向是 key，n 方向是 d。mma 要的 B 寄存器 i 是 `(k = 2q+8i, n = g)` 和 `(k+1, n = g)`：**同一个 d、相邻两个 key**。但 V 在 shared memory 里是 `[key][D]`，d 连续，相邻两个 key 相隔一整行，不能作为一个 32 位字读出。用 GEMM 的话说，V 是“MN-major”的 B（n 方向连续），而 K 是 K-major 的。

`ldmatrix.trans` 在读入时把每个 8×8 矩阵转置：lane l 得到矩阵第 `2(l%4)` 和 `2(l%4)+1` **行**、第 `l/4` **列**的两个元素。把 V 的 8 个 key 行作为矩阵的行，正好得到 `(key 2q, d g)`、`(key 2q+1, d g)`，就是 B 寄存器要的东西。

一条 `ldmatrix_x4_trans` 读 4 个 8×8 矩阵：

```
lane 0–7   → key 行 16t+0..7,  d 列块 c      → r[0]：d tile c 的 B 寄存器 0
lane 8–15  → key 行 16t+8..15, d 列块 c      → r[1]：d tile c 的 B 寄存器 1
lane 16–23 → key 行 16t+0..7,  d 列块 c+1    → r[2]：d tile c+1 的 B 寄存器 0
lane 24–31 → key 行 16t+8..15, d 列块 c+1    → r[3]：d tile c+1 的 B 寄存器 1
```

即 lane l 给出 key 行 `16t + l%16`、列块 `c + l/16` 的地址。

### 4.5 一行 256 字节时的 swizzle

`lab::swizzle_chunk<8>` 处理 128 字节（8 个 16 字节 chunk）的行。D = 128 的 FP16 行是 256 字节。bank 每 128 字节重复一次，所以把一行看成两个 128 字节的段，每段单独 swizzle 就够了：

```cpp
offset(row, col) = row*D + (col/64)*64 + swizzle_chunk<8>(row, (col%64)/8)*8 + col%8
```

同一段里，8 个连续行的同一个逻辑 chunk 落在 8 个不同的物理 chunk 上，`ldmatrix`（包括 `.trans`）和 `cp.async` 的写都没有 bank conflict。

### 4.6 常见误用

| 误用 | 后果 |
|---|---|
| V 用普通 `ldmatrix` | B 拿到的是同一个 key 的相邻两个 d，O 完全错误 |
| 越界的 K/V 行不 zero-fill | 尾部块的 shared memory 里是旧数据，可能是 NaN；`0 × NaN = NaN`，被 mask 的 P 也救不回来 |
| m 乘了 scale 再比较、却用原始分数算指数 | 两套单位混用，`large_logits` 和 `scale` case 失败 |
| 在 online softmax 里对 ℓ 做 quad 求和，epilogue 再求一次 | ℓ 变成 4 倍（或 16 倍），O 偏小 |
| 用 `exp2f`/`expf` | 正确但慢：不加 `--use_fast_math` 时它们不是单条 `MUFU.EX2` |

## 5. 调度设计

### 5.1 网格与 CTA

```
grid  = (⌈N / Br⌉, H, B)       blockIdx.x = Q 块，.y = head，.z = batch
block = 4 个 warp（128 线程）
warp w 负责 CTA 内的第 w·16·kMTiles .. 行，每个 lane 负责其中 2·kMTiles 行
```

| D | kMTiles | Br（Q 行/CTA） | Bc（key/块） | 每 lane 的 Q / O / S 寄存器 | shared memory（步骤 2–3 / 步骤 4） |
|---|---|---|---|---|---|
| 128 | 1 | 64 | 64 | 32 / 64 / 32 | 48 KB / 80 KB |
| 64 | 2 | 128 | 32 | 32 / 64 / 32 | 24 KB / 32 KB |

两种配置都在数组上花 128 个寄存器，剩下的给地址和临时值。D = 64 时如果用 Bc = 64，S 要 64 个寄存器，数组共 160 个，编译器在步骤 2 的 kernel 中开始溢出到 local memory；所以改用 Bc = 32。D = 128 的 kernel 每线程约 220–240 个寄存器，每个 SM 能放 2 个 CTA（8 个 warp）；D = 64 约 170 个，能放 3 个。

### 5.2 一个 K/V 块的流程（步骤 2）

```
          所有 warp                                    每个 warp 自己（无 barrier）
 ┌──────────────────────────────┐   ┌─────────────────────────────────────────────────────┐
 │ cp.async K_j, V_j → smem      │   │ S = Q·K_jᵀ        Q 在寄存器，K_j 用 ldmatrix        │
 │ wait_group 0; __syncthreads() │ → │ mask（只有需要的块）                                  │
 └──────────────────────────────┘   │ m', α, P~ = 2^(…), ℓ, O~ *= α   quad shuffle          │
                                    │ O~ += P~·V_j      P~ 作 A，V_j 用 ldmatrix.trans     │
                                    └─────────────────────────────────────────────────────┘
                                      __syncthreads()   所有 warp 读完 K_j、V_j 才能覆盖
```

warp 之间唯一的交互是共享 K/V tile。softmax 的一切都在 warp 内完成，这就是 FA2 的 “split-Q”。

### 5.3 causal：跳块与只在对角块加 mask（步骤 3）

query i 只能看 key j ≤ i。CTA 的行是 `q0 .. q0+Br−1`，于是：

- **跳过的块**：起始 key `k0 > q0+Br−1` 的块对本 CTA 的每一行都全被 mask，根本不加载。循环上限从 `⌈N/Bc⌉` 变成 `min(⌈N/Bc⌉, ⌈(q0+Br)/Bc⌉)`。
- **需要 mask 的块**：块的最后一个 key `k0+Bc−1` 大于 CTA 的第一行 `q0`，块内就有某个 (行, key) 要被 mask。此外越过 N 的最后一个块总要 mask。
- **其余的块**完全可见，直接算。

D = 128（Br = Bc = 64）时，Q 块 i 处理 i+1 个块，只有最后一个（对角块）加 mask。`perf_causal` 中 64 个 Q 块共处理 `64·65/2 = 2080` 个块，是非 causal 的 4096 个的 51%；其中需要 mask 的只有 64 个。D = 64（Br = 128，Bc = 32）时对角线跨过 4 个 K 块，这 4 个都要 mask。

mask 本身只是把 S 设成 −∞。C fragment 的元素 e 对应行 `g + 8·(e/2)`、key `2q + e%2`（加上 tile 和块的偏移）。

### 5.4 cp.async 双缓冲（步骤 4）

和 21 关步骤 1 相同，只是一个 stage 装的是一对 (K_j, V_j)：

```
prologue: cp.async Q 与 K_0/V_0（stage 0），commit        ← 一个 group
for j:
    wait_group 0; __syncthreads()       ① 块 j 对所有 warp 可见  ② 所有 warp 都做完了块 j−1
    if j+1 存在: cp.async K_{j+1}/V_{j+1} → stage (j+1)%2; commit
    计算块 j（stage j%2）                ← 此时块 j+1 在路上
```

`__syncthreads()` 的第②点保证 stage (j+1)%2（上一轮的块 j−1 用过）可以被覆盖。每个块只需要一次 wait 和一次 barrier。

## 6. 分步练习

所有命令在仓库根目录运行。骨架中 load/launch、`issue_rows`、mask 的尾部部分、dtype 转换和 quad 归约都已写好。

### 步骤 1 · TODO(1)：Q fragment 与 S = Q·Kᵀ

实现 `load_q`（1a）和 `qk_block`（1b）。step 1 的 kernel `fa2_scores` 已经写好：它循环所有 K 块，把 `scale·S` 写进 probe（`[B,H,N,N]`，FP32），由 `fa2_scores` oracle 对照 FP64 的 `scale·QKᵀ`。此时还没有 mask 和 softmax。

```bash
./lab check 22 --step 1
```

- 报告中的维度名是 `[b, h, i, j]`：`i` 错说明 Q 的行（A fragment）取错，`j` 错说明 K 的行（B fragment）取错。
- 只有 D = 128 的 case 错：检查是否用了 `tile_offset`（两段 swizzle），而不是自己写 `row*D + col`。
- 单块的 case（`n1`、`one_block`、`short_tail`）对，多块的错：`qk_block` 每个块都要从 0 开始累加 S。

### 步骤 2 · TODO(2)：online softmax、P·V 和 epilogue

依次实现 `online_softmax`（2a）、`pv_block`（2b）和 `fa2_forward` 末尾的 epilogue（2c）。step 2 只有非 causal 的 case。

```bash
./lab check 22 --step 2
./lab bench 22 --step 2
```

建议先只跑 `./lab check 22 --step 2 --case n1` 和 `--case one_block`，再跑全部。

### 步骤 3 · TODO(3)：causal

三处：3a 减少循环的块数，3b 在 `apply_mask` 中加 causal 条件，3c 决定哪些块需要 mask。

```bash
./lab check 22 --step 3
./lab bench 22 --step 3 --case perf_causal
```

做完后可以做个实验：把 3a 改回不跳块（只保留 mask），再 bench `perf_causal`，对比 §7 的表。

### 步骤 4 · TODO(4)：cp.async 双缓冲

4a 在 prologue 发出块 0 的拷贝，4b 写循环开头的 wait、barrier 和下一块的拷贝。

```bash
./lab check 22
./lab bench 22 --case perf
./lab bench 22 --case perf_causal
./lab bench 22 --case perf_d64
./lab sass 22            # 看 HMMA / LDSM / LDGSTS / MUFU.EX2，以及有没有 spill
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| step 1 全部错，误差像随机数 | `ldmatrix` 的 lane → 行/chunk 映射错；或 Q 与 K 的 fragment 角色弄反 |
| step 2 `uniform` 的 LSE 都差 ln 4 ≈ 1.386 | epilogue 忘了对 ℓ 做 quad 求和 |
| `ones_v` 偏离 1 很多 | 循环中也做了归一化，或者 ℓ 和 O 用了不同的 α |
| `position` 错，其他随机 case 误差也大 | P 的 A 寄存器顺序错（寄存器 1 和 2 弄反），或 V 没有用 `.trans` |
| 只有 `tail`/`short_tail` 出 NaN | 越界的 K/V 行没有 zero-fill，或尾部块没有 mask |
| `large_logits` 类 case 出 inf/NaN | 指数里没减最大值，或 m 和 S 的单位不一致（§3.3） |
| 第一个 Q 块的 O 为 NaN、LSE 为 −inf，其余 Q 块全错 | 3a 的块数少算了一块（对角块），常见写法是 `(q0 + kBM - 1) / kBN` 少了 `+1` |
| step 4 偶尔错 | 少了 barrier，或在 barrier 之前就覆盖了另一个 stage |
| `TIMEOUT` 或挂起 | 不同 warp 走了不同次数的循环、少走了 `__syncthreads()`：块数必须对整个 CTA 一致 |

## 7. 性能

### 7.1 相对库基线

7 号卡实测（CUDA Graph；`vs cudnn` 是 cuDNN 耗时除以本版本耗时，越大越好）：

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 步骤 2/3（同步加载） | ≈ 0.23× | ≈ 0.30× | ≈ 0.36× |
| 步骤 3，causal 不跳块 | — | ≈ 0.15× | — |
| 步骤 4（双缓冲） | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4（`flash_attn.cute`） | ≈ 1.1× | ≈ 1.3× | ≈ 1.15× |

步骤 4 在三个 perf case 上都达到 cuDNN 的 0.2 倍以上，就说明机制成立。相对 FA4，步骤 4 约为 0.23×、0.25×、0.34×。`--cold`（每次清空 L2）下比例基本不变（`perf` 约 0.28×）：Q、K、V 共 200 MB，本来就大于 L2。

几点观察：

- **跳块让 causal 快了约一倍**（0.15× → 0.32×），和 §5.3 算出的 51% 工作量一致。不跳块时比非 causal 还慢，因为每个块都要做 mask。
- causal 的 TFLOP/s 比非 causal 低约 10%：Q 块 i 的工作量正比于 i+1，按 `blockIdx.x` 顺序发射时最重的 CTA 最后才开始，最后一波里很多 SM 闲着。43 关的 LPT 调度处理这个问题。
- 双缓冲只带来约 10–15%。原因是每个 SM 上有 2–3 个 CTA，一个 CTA 在等数据时，另一个在算（和 21 关“步骤 2 几乎没有变化”是同一个道理）。

### 7.2 为什么只有 cuDNN 的约 1/4：`mma.sync` 的上限

20 关 §4.1 测得 B200 上 `mma.sync` 的 BF16 上限约 560 TFLOP/s，而 cuDNN 和 FA4 用的是 `tcgen05`（部分 3、4）。本关步骤 4 在 `perf` 上约 340 TFLOP/s，即 **`mma.sync` 上限的约 61%**；cuDNN 约 1300 TFLOP/s，已经是 `mma.sync` 上限的 2.3 倍。所以：

| | TFLOP/s（`perf`） | 相对 cuDNN |
|---|---|---|
| 本关步骤 4 | ≈ 340 | ≈ 0.26× |
| `mma.sync` 上限（20 关微基准，无访存、无 softmax） | ≈ 560 | ≈ 0.43× |
| cuDNN（`tcgen05`） | ≈ 1300 | 1× |

即使把本关优化到完美，也只能到 cuDNN 的 0.43 倍左右。真正的差距是指令代际，不是写法。

### 7.3 和 12 关比

12 关的 FA1 是 FP32 SIMT，`perf`（[4,4096,64,64]）约 25 TFLOP/s。本关 D = 64 的 `perf_d64` 约 320 TFLOP/s，约 **13 倍**。两者 dtype 不同（FP32 vs FP16），这个比例主要来自 Tensor Core；但如果仍按 FA1 的循环顺序把 O/ℓ/m 每块写回 HBM，12 关 §7.2 那份“不是瓶颈”的额外流量（读写约为 Q/K/V/O 大小的几十倍，见 12 关 §2 与 §7.2）在这个速度下就会成为瓶颈。本关 profile 中 `dram__bytes_read` ≈ 200 MB，正好是 Q、K、V 各读一次（K/V 被同一 head 的 64 个 Q 块重复读，命中 L2）。

### 7.4 时间花在哪里

```bash
./lab profile 22 --case perf             # 步骤 4
./lab profile 22 --case perf --step 2
```

本机 Nsight Compute 结果（`perf`）：

| 指标 | 步骤 2 | 步骤 4 | 说明 |
|---|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 60% | ≈ 66% | Tensor Core 的忙碌比例 |
| `sm__warps_active` | ≈ 12% | ≈ 12% | 每 SM 8 个 warp（2 个 CTA） |
| `sm__inst_executed_pipe_tensor_subpipe_hmma` | 1.34 亿 | 1.34 亿 | = 4·B·H·N²·D / (2·16·8·16)，一条不多 |
| `smsp__inst_executed_pipe_xu` | — | ≈ 0.36 亿 | 略多于 B·H·N² / 32 ≈ 0.34 亿：每个分数一次 `MUFU.EX2`，加上 α、log 等 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` | 0 | 0 | swizzle 后 `cp.async` 写无冲突 |

（Nsight Compute 锁定时钟，`gpu__time_duration` 比 bench 长，只用它比较指标。）

剩下的约 1/3 空闲来自：每个块一次 `__syncthreads()`；softmax 的 FFMA、`MUFU.EX2`、shuffle 与 mma 在同一个 warp 里串行，没有别的 warp 时 Tensor Core 只能等；每个 SM 只有 8 个 warp，寄存器不允许更多。部分 3 的 FA3 思想就是针对这里：用专门的 warp 搬运、让 softmax 和下一个块的 QKᵀ 重叠。

可以自己试的实验：

- D = 128 改成 8 个 warp、Br = 128（`Config<128, 8, 1, 64, …>`）。本机上这样**更慢**（步骤 4 约 0.22×）：每个 SM 只放得下一个 CTA，它的 barrier 没有别的 CTA 来填补。
- D = 64 改成 Bc = 64，用 `./lab sass 22` 看 spill，再 bench。

## 8. 自测问题

1. FA1 的 K/V 外层循环中，每个 Q 块的 O、ℓ、m 要在 HBM 往返多少次？交换循环后呢？交换循环后，K/V 被读了多少次？
2. 一个 warp 处理 16 行、Bc = 64 时，lane 13 持有 S 的哪两行、每行哪些列？它要和哪些 lane 做 shuffle？
3. 为什么 m 要在 quad 内求最大值，而 ℓ 可以留到最后再求和？
4. 如果把 FA2 改成 “split-K”（4 个 warp 各算 16 个 key），每个块需要哪些额外的 shared memory 读写和同步？
5. 为什么 V 需要 `ldmatrix.trans`，而 K 不需要？如果 V 在 HBM 里存成 `[B,H,D,N]`，还需要吗？
6. causal、N = 4096、Br = 64、Bc = 128 时，Q 块 5（行 320–383）要处理几个 K 块？哪些要加 mask？

<details>
<summary>参考答案</summary>

1. FA1：每个 K/V 块读写一次，共 Tc = ⌈N/Bc⌉ 次。FA2：O 和 LSE 只写一次，ℓ、m 从不进 HBM。代价是每个 Q 块都要读全部 K/V，即 K/V 从全局内存读 ⌈N/Br⌉ 次；它们大多命中 L2（§7.3 中 DRAM 读只有 Q/K/V 各一次）。
2. g = 13/4 = 3，q = 1：行 3 和行 11；每行的列是每个 n8 tile 中的 2q, 2q+1 = 2, 3，即 key 2, 3, 10, 11, 18, 19, …, 58, 59（共 16 个）。与同一 quad 的 lane 12、14、15 shuffle（xor 1 和 xor 2）。
3. 指数 `2^(s − m)` 中所有 lane 必须用同一个 m，否则同一行的 P~ 被不同的因子缩放、不能相加。ℓ 只是被 α 缩放后再加上本 lane 的 P~，乘法对加法分配，所以 4 个部分和可以各自缩放，最后相加（§3.3）。
4. 每个 warp 只看到部分 key：行最大值要先在 warp 内求、写进 shared memory、`__syncthreads()`、再读回所有 warp 的值求总最大值；行和同理；每个 warp 的 P·V 只是部分和，要通过 shared memory 加到一起，又一次同步。FA2 的 split-Q 这些都不需要。
5. mma 的 B 寄存器要同一个 n、相邻两个 k。QKᵀ 中 k 是 D，K 的 D 连续，直接读；PV 中 k 是 key，V 的 key 方向不连续，要转置。若 V 存成 `[D][N]`（key 连续），它就是 K-major，用普通 `ldmatrix`。
6. 需要 key ≤ 383：`min(32, ⌈384/128⌉) = 3` 个块（key 0–383）。块的最后一个 key 大于 q0 = 320 的要 mask：块 2（key 256–383）的最后一个 key 383 > 320，要 mask；块 0、1 不用。注意此时 q0 不是 Bc 的倍数，对角线落在块 2 的中间。
</details>

## 9. 延伸阅读

- Tri Dao，*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*（2023）：3.1 节（减少非矩阵乘运算、只保存 logsumexp）、3.2 节（按序列长度并行）、3.3 节（warp 间按 Q 划分）、Algorithm 1
- PTX ISA：[Matrix Fragments for mma.m16n8k16 with floating point type](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)、[ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)、[cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)、[ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2)
- 上游 FA2（`third_party/flash-attention`，提交 `1fc0221`）：
  - `csrc/flash_attn/src/flash_fwd_kernel.h`：`compute_attn_1rowblock`（一个 CTA 处理一个 Q 块；`n_block_max` 是 causal 跳块；先处理需要 mask 的块，再处理其余块。本关按 K 块从前往后遍历，让同一个循环也能用于步骤 4 的流水，见 NOTES.md）
  - `csrc/flash_attn/src/softmax.h`：`Softmax::softmax_rescale_o`（§3.1 的更新）、`normalize_softmax_lse`（最后的除法与 LSE）、`quad_allreduce_`（quad shuffle）
  - `csrc/flash_attn/src/utils.h`：`convert_layout_acc_Aregs`（§4.3：C fragment 重新解释为 A fragment）
  - `csrc/flash_attn/src/mask.h`：`apply_mask_causal`
  - `csrc/flash_attn/src/flash_fwd_launch_template.h`：各 head 维的块大小与 warp 数
- 后续：部分 3（`30_tma_mbarrier` 起）用 TMA 和 `tcgen05` 跨过 §7.2 的上限
