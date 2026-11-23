# 12 · FlashAttention-1：分块与 online softmax

> 部分 1 · FA1　｜　前置：[01_softmax](../01_softmax/README.md)、[10_simt_gemm](../10_simt_gemm/README.md)、[11_naive_attention](../11_naive_attention/README.md)　｜　代码：[`students/12_fa1/kernel.cu`](../../students/12_fa1/kernel.cu)

## 1. 本关目标

按 FlashAttention 论文的 **Algorithm 1**，用 FP32 SIMT 指令写出一个完整的 attention 前向 kernel：

- K/V 按 `Bc` 行分块，Q 按 `Br` 行分块；
- **外层**循环遍历 K/V 块 `j`，**内层**循环遍历 Q 块 `i`；
- 每处理一个 `(i, j)`，都从 HBM 读出 `O_i, l_i, m_i`，更新后再写回 HBM；
- 输出 O 和 LSE（自然对数），支持 causal mask，并跳过整块被 mask 的 tile。

做完后你应该能：

- 证明“分块 + 运行中的 (m, l, O)”给出的结果和整行 softmax **完全相同**（不是近似）；
- 用论文的方法推出 HBM 访问量 Θ(N²d²/M)，并用 B200 的 shared memory 预算选 `Br`、`Bc`；
- 解释“一个 (batch, head) 一个 CTA”的并行方式什么时候填不满 148 个 SM；
- 用 `./lab bench` 和 `./lab profile` 看出这一版的时间花在哪里，以及它为什么比库慢约 2 倍。

本关刻意保留论文的循环顺序（K/V 在外层）。下一部分的 `22_fa2` 会把两层循环交换，你可以拿本关的数字直接对比。

## 2. 动机：去掉 N×N 的中间矩阵

[11_naive_attention](../11_naive_attention/README.md) 把 attention 拆成三个 kernel：`S = scale·QKᵀ`、`P = softmax(S)`、`O = PV`。S 和 P 各有 N² 个元素，必须写到 HBM 再读回来。N = 4096 时，一个 head 的 S 就是 64 MiB（FP32），而 Q、K、V、O 合起来只有 4 MiB。HBM 流量和显存占用都被 N² 项主导。

FlashAttention 论文第 3.1 节的思路是：**不生成完整的 S 和 P**。每次只把一块 K/V 和一块 Q 放进 SRAM（GPU 上就是 shared memory 和寄存器），算出这一小块的分数、概率和对 O 的贡献，然后丢掉。难点在 softmax：一行的归一化需要这一行**所有** key 的最大值和指数和，而分块时后面的块还没看到。[01_softmax](../01_softmax/README.md) 的 online softmax 已经解决了这个问题的一半（m 和 l）；本关把它推广到带 V 的加权和，并证明它是精确的。

## 3. 数学

### 3.1 一个 query、一个 key 集合的状态

固定 batch、head 和一个 query 行。对 key 的任意集合 A，定义：

| 记号 | 定义 | 含义 |
|---|---|---|
| `S_j` | `scale · q·k_j` | 第 j 个 key 的分数 |
| `m_A` | `max_{j∈A} S_j` | A 上的最大分数 |
| `l_A` | `Σ_{j∈A} exp(S_j − m_A)` | 以 `m_A` 为基准的指数和 |
| `O_A` | `Σ_{j∈A} exp(S_j − m_A) · v_j / l_A` | **只在 A 上**做 softmax 后对 V 的加权平均 |

A 取全部 key 时，`O_A` 就是要求的输出 O，`m_A + ln(l_A)` 就是 LSE（`ln Σ_j exp(S_j)`）。

空集的状态约定为 `m = −∞, l = 0, O = 0`。这正是论文 Algorithm 1 第 2 行在 HBM 里写的初值。

论文把 O 存成**已经归一化**的形式（除过 `l`），本关照做。FA2 会改成“只在最后除一次”，那是下一部分的优化。

### 3.2 合并定理

设 A、B 是两个不相交的 key 集合，它们的状态分别是 `(m_A, l_A, O_A)` 和 `(m_B, l_B, O_B)`。令

```
m   = max(m_A, m_B)
α   = exp(m_A − m)          β = exp(m_B − m)
l   = α·l_A + β·l_B
O   = (α·l_A·O_A + β·l_B·O_B) / l
```

则 `(m, l, O)` 正好是 A∪B 的状态。

**证明。** 第一行：`max` 对并集显然成立。对 A 中的每个 j，

```
exp(S_j − m_A) · α = exp(S_j − m_A) · exp(m_A − m) = exp(S_j − m)
```

所以 `α·l_A = Σ_{j∈A} exp(S_j − m)`，同理 `β·l_B = Σ_{j∈B} exp(S_j − m)`，两者相加就是 `l_{A∪B}`。再看 O：`l_A·O_A = Σ_{j∈A} exp(S_j − m_A)·v_j` 是“未归一化的分子”，乘 α 后基准换成 m；B 同理。相加得到 A∪B 的分子，除以 `l` 即 `O_{A∪B}`。∎

**归纳到任意多块。** 把 key 分成 `K_1, K_2, …, K_Tc`。令 `A_0 = ∅`，`A_j = A_{j−1} ∪ K_j`。`A_0` 的状态就是初值；若 `A_{j−1}` 的状态正确，则用合并定理把它和 `K_j` 这一块的状态合并，得到 `A_j` 的状态。归纳到 `j = Tc` 得到全集的状态，也就是 O 和 LSE。整个过程只用到实数恒等式，所以**结果是精确的**；浮点下只是舍入次序不同。

**为什么合并的顺序无关。** 合并对应集合的并，而并满足交换律和结合律。因此 K/V 块可以按任何顺序、甚至分成几组并行处理后再合并（后者就是以后 split-KV 的思路）。

### 3.3 一个 K/V 块的状态：Algorithm 1 第 10 行

对一个 `Br × Bc` 的分数块 `S_ij`，每一行分别计算：

```
m~ = rowmax(S_ij)
P~ = exp(S_ij − m~)        （逐元素）
l~ = rowsum(P~)
```

这一块的未归一化分子是 `P~·V_j`。代入合并定理（A = 已处理的块，B = 第 j 块，`l_B·O_B = P~·V_j`），就是论文第 11–12 行：

```
m_new = max(m_i, m~)
l_new = e^(m_i − m_new)·l_i + e^(m~ − m_new)·l~
O_i  ← ( l_i·e^(m_i − m_new)·O_i + e^(m~ − m_new)·P~·V_j ) / l_new
```

### 3.4 手算：一个 query、两个 K/V 块

`Bc = 2`，4 个 key，分数（已乘 scale）和标量 V 为：

| key | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| S | 1 | 2 | 4 | 3 |
| V | 10 | 20 | 30 | 40 |

**块 1**（key 0、1）：`m~ = 2`，`P~ = (e⁻¹, 1) = (0.36788, 1)`，`l~ = 1.36788`，`P~V = 23.6788`。和初值 `(−∞, 0, 0)` 合并：`α = e^(−∞) = 0`，所以 `m = 2`，`l = 1.36788`，`O = 23.6788 / 1.36788 = 17.3106`。

**块 2**（key 2、3）：`m~ = 4`，`P~ = (1, e⁻¹)`，`l~ = 1.36788`，`P~V = 30 + 40·0.36788 = 44.7152`。最大值变大了，旧状态要缩小：

```
m_new = max(2, 4) = 4
α     = e^(2−4) = 0.135335        β = e^(4−4) = 1
l_new = 0.135335·1.36788 + 1.36788 = 0.185122 + 1.36788 = 1.553002
O     = (0.185122·17.3106 + 44.7152) / 1.553002 = (3.20458 + 44.7152) / 1.553002 = 30.8562
LSE   = 4 + ln(1.553002) = 4.44019
```

**直接算**：`exp(S − 4) = (0.049787, 0.135335, 1, 0.367879)`，和为 1.553002；加权和 `0.49787 + 2.70671 + 30 + 14.7152 = 47.9198`，`O = 47.9198 / 1.553002 = 30.8562`。两者一致。

注意块 1 的 `l = 1.36788` 被乘上 0.135335 后变成 0.185122：这正是 key 0、1 以新基准 4 计的指数和 `0.049787 + 0.135335`。`large_logits` 这类 case 会让最大值在后面的块里变大，少乘一次 α 立刻就错。

### 3.5 Algorithm 1 的循环顺序

```
 1  Bc = ⌈M / 4d⌉,  Br = min(⌈M / 4d⌉, d)
 2  在 HBM 中初始化 O = 0, l = 0, m = −∞
 3  Q 分成 Tr = ⌈N/Br⌉ 块，K、V 分成 Tc = ⌈N/Bc⌉ 块
 5  for j = 1 .. Tc:                    ← 外层：K/V 块
 6      把 K_j, V_j 读进 SRAM
 7      for i = 1 .. Tr:                ← 内层：Q 块
 8          把 Q_i, O_i, l_i, m_i 读进 SRAM
 9          S_ij = Q_i K_jᵀ
10          m~, P~, l~（3.3 节）
11          m_new, l_new
12          O_i ← …，写回 HBM
13          l_i ← l_new, m_i ← m_new，写回 HBM
16  返回 O
```

论文的块号从 1 数起；代码（和 3.7 节）从 0 数起，即 `j = 0 .. Tc−1`、`i = 0 .. Tr−1`。

每个 K/V 块只读一次；但对每个 `j`，所有 Q 块和 O 块都要从 HBM 读一遍，O 还要写一遍。这是 FA1 和 FA2 最大的区别：FA2 把 Q 放到外层，`O_i` 一直留在寄存器里，只在最后写一次。

### 3.6 HBM 访问量：Θ(N²d²/M)

记 M 为 SRAM 能放下的元素个数，只看一个 head：

| 数据 | 每个 j 的读写量 | 次数 | 总量 |
|---|---|---|---|
| K、V | — | 一次 | 2Nd |
| Q | 读 Nd | Tc | Tc·Nd |
| O | 读 Nd + 写 Nd | Tc | 2·Tc·Nd |
| l、m | 读 2N + 写 2N | Tc | 4·Tc·N |

合计 Θ(Tc·Nd) = Θ(N²d / Bc)。论文取 `Bc = Θ(M/d)`，得到 **Θ(N²d²/M)**（论文定理 2）；标准 attention 是 Θ(Nd + N²)。d = 64、M 约为几万个元素时，`d²/M` 远小于 1，所以 FA1 的访问量比标准实现少很多倍。

为什么是 `M/4d`：一个 SRAM 要同时放下 `K_j`、`V_j`（各 `Bc·d`）、`Q_i`、`O_i`（各 `Br·d`）和 `S_ij`（`Br·Bc`）。取 `Bc = M/4d` 让 K_j、V_j 各占 M/4；取 `Br ≤ d` 让 `Br·Bc ≤ M/4`；`Br ≤ M/4d` 让 Q_i、O_i 各不超过 M/4。

**放到 B200 上。** 每个 SM 有 228 KB shared memory，一个 CTA 最多用 227 KB，即 M ≈ 58,000 个 FP32。论文公式给出：

| d | `Bc = ⌈M/4d⌉` | `Br = min(Bc, d)` |
|---|---|---|
| 64 | 227 | 64 |
| 128 | 114 | 114 |

本关实际用 `Br = Bc = 64`（第 5 节解释原因）。代入表格，perf case（N = 4096，d = 64，Tc = 64）每个 head 约读 `64 × 2 MiB`、写 `64 × 1 MiB`，是 Q/K/V/O 本身（4 MiB）的 40–60 倍。第 7 节用 Nsight Compute 实测验证这个数字。

### 3.7 LSE 与 causal

**LSE。** 外层循环结束后，每行的 `(m, l)` 就是全集的状态，`LSE = m + ln(l)`。注意是自然对数；如果你内部用 `exp2`，要在这里换算回来。

**causal mask。** query `r` 只能看 key `c ≤ r`，其余分数设为 −∞，`exp(−∞) = 0`，等价于把这些 key 从集合里拿掉。尾部也一样：最后一块中 `c ≥ N` 的 key 是填充，同样设为 −∞。

**跳过整块。** Q 块 i 包含 query `i·Br … i·Br+Br−1`，K/V 块 j 包含 key `j·Bc … j·Bc+Bc−1`。如果 Q 块的最后一个 query 小于 K/V 块的第一个 key，即 `i·Br + Br − 1 < j·Bc`，整块全被 mask：这一块的状态是空集，合并空集不改变任何东西，所以可以**整块跳过**，连 Q_i、O_i 都不用读。对固定的 j，需要处理的 Q 块从 `i = ⌊j·Bc / Br⌋` 开始。causal 的工作量因此约为非 causal 的一半。

**空集的陷阱。** 一行在某块里全被 mask 时 `m~ = −∞`，`exp(S − m~) = exp(−∞ − (−∞)) = NaN`。本关 `Br = Bc` 且跳过了整块 mask 的 tile，真实 query 行不会遇到这种情况，但代码仍应把这时的减数换成 0（得到 `P~ = 0, l~ = 0`），并在 `m_new = −∞` 时不更新这一行。这两行防护不花什么代价，换别的块大小时就是必需的。

## 4. 硬件与指令

本关没有新指令，用到的都是前面学过的：shared memory tile（00）、warp shuffle 归约（01）、寄存器分块 GEMM（10）。新的是**预算**和**同步**。

### 4.1 存储预算

| 资源 | B200 数值 | 本关用量（d = 64 / 128） |
|---|---|---|
| shared memory | 每 SM 228 KB，每 CTA 最多 227 KB | 67 KB / 116 KB |
| 寄存器 | 每 SM 64K 个 32 bit，每线程最多 255 | 约 128 / 189 个（`./lab sass 12 --solution` 中多块路径 `kSingleBlock = false` 的那一项） |
| 线程 | 每 SM 最多 2048 | 每 CTA 256 |
| SM | 148 | — |

动态 shared memory 超过 48 KB 时，必须先对 kernel 调用 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`，否则 launch 直接失败。骨架在 `make_plan` 里做了这一步（不计时）。

每个 SM 能同时驻留几个 CTA，取决于最紧的那项资源。d = 64：寄存器 `128 × 256 = 32K`，可放 2 个 CTA；shared memory 可放 3 个；所以是 2 个。d = 128：shared memory 和寄存器都只够 1 个。

### 4.2 HBM 状态与 `__syncthreads()`

本关的 O、l、m 放在 **global memory** 里，同一个 CTA 的不同线程会读写同一行的 l、m。`__syncthreads()` 不仅等所有线程到达，也保证之前对 global 和 shared memory 的写对 CTA 内其他线程可见。这就是为什么“一个 head 一个 CTA”能让 HBM 状态的同步这么简单：同一行的所有读写都在一个 CTA 里。

一行的 64 个分数分在 16 个线程上，每个线程都需要旧的 `m_i, l_i`，但只让 `tx == 0` 的线程写回新值。如果有线程在别的线程读取之前就写了新值，就会出错。骨架的顺序是：

```
load_state（每个线程读旧的 O、m、l）  →  __syncthreads()  →  …  →  merge_state（写新值）
```

中间的 barrier 保证写一定发生在所有读之后。

### 4.3 16 线程一组的归约

线程 `(ty, tx) = (tid/16, tid%16)`，一行的 16 个线程是一个 warp 里连续的 16 个 lane。用 `__shfl_xor_sync` 依次异或 1、2、4、8，就在这 16 个 lane 内做完了 max 或 sum，而且 16 个线程都拿到结果（骨架提供 `group16_max`、`group16_sum`）。异或 16 会跨到另一行，不能用。

### 4.4 指数与 −∞

- `expf(x)` 对 FP32 足够准；`expf(-INFINITY) = 0`。
- `fmaxf(-INFINITY, x) = x`，所以 m 的初值可以直接参与 max。
- `-INFINITY − (-INFINITY)` 是 NaN，`0 × NaN` 也是 NaN。这就是 O 必须初始化为 0、空集要特殊处理的原因。
- `lab::Scratch` 分配的 m、l 缓冲初始全是 `0xFF` 字节（NaN），O 输出也被 harness 填成 NaN：忘记初始化会立刻在结果里看到 NaN。

## 5. 调度设计

### 5.1 一个 (batch, head) 一个 CTA

```
grid  = B·H 个 CTA，blockIdx.x = b·H + h
block = 256 个线程 = 16 × 16
```

这是原始 FA1 kernel 的并行方式（FA2 论文第 3.2 节的描述）：一个 CTA 从头到尾负责一个 head 的全部 N 行，所以所有 HBM 状态只在 CTA 内共享，不需要跨 CTA 同步。代价是**并行度等于 B·H**：

| case | B·H | 每 SM 2 个 CTA 时占用 |
|---|---|---|
| `few_heads` [1, 4096, 16, 64] | 16 | 16 个 SM 有活干，132 个空闲 |
| `perf` [4, 4096, 64, 64] | 256 | 108 个 SM 跑 2 个，40 个 SM 跑 1 个 |

长序列、小 batch 的场景（推理很常见）B·H 很小，GPU 大部分闲着。FA2 改为再按 Q 块切分 CTA，就是为了解决这个问题。

### 5.2 线程分工

线程 `(ty, tx)` 负责 Q 块中的第 `ty·4 … ty·4+3` 行：

| 数据 | 线程持有的部分 | 存在哪里 |
|---|---|---|
| `S_ij`、`P~`（64×64） | 4 行 × 第 `tx·4 … tx·4+3` 列 | 寄存器 `s[4][4]` |
| `m~`、`l~` | 4 行，16 个线程各存一份相同的值 | 寄存器 |
| `P~·V_j`（64×d） | 4 行 × 第 `g·64 + tx·4 … +3` 列，`g < d/64` | 寄存器 `acc[4][d/16]` |
| 旧的 `O_i`、`m_i`、`l_i` | 与上面相同的位置 | 寄存器（从 HBM 读来） |

### 5.3 shared memory 布局（d = 64）

| 数组 | 形状 | 为什么这样存 |
|---|---|---|
| `Kt` | `[d][Bc+4]` | K 转置：算 `S` 时一个线程用一条 float4 读 4 个相邻 key 的同一维 |
| `Vs` | `[Bc][d]` | 行主序：算 `P~V` 时 float4 读 4 个相邻的 d |
| `Qs` | `[Br][d+4]` | `+4` 让同一 warp 里相隔 4 行的两次 float4 读落在不同 bank |
| `Ps` | `[Br][Bc+4]` | `P~` 从寄存器写回，供 `P~V` 使用 |

K_j、V_j 每个 j 只加载一次，在内层循环中被所有 Q 块复用；Q_i 和 P~ 每个 (i, j) 都要重新写。

### 5.4 块大小为什么是 64，而不是论文的 227

论文的 `Bc = M/4d` 只考虑了 SRAM 容量，目标是让 HBM 访问量最小。在 SIMT 实现里还有别的约束：

- `S_ij` 在寄存器里，`Br × Bc / 256` 是每个线程要存的分数个数。`64 × 227` 需要每线程约 57 个，加上 O 的累加器会溢出到 local memory。
- 一个 CTA 占满 227 KB 时每个 SM 只能驻留 1 个 CTA（8 个 warp），延迟更难隐藏。
- 本关的 kernel 不是 HBM 带宽受限（第 7 节实测），减少 HBM 访问的收益有限。

所以本关选 `Br = Bc = 64`：4×4 的线程 tile 正好覆盖 64×64，`Br = Bc` 还让 causal 的块分类变得简单（对角块、完全可见块、完全 mask 块）。这是“IO 最优”与“计算效率”之间的取舍。

### 5.5 一个 (i, j) 的时序

```
         所有 256 个线程
 j 开始 ─ sync ─ load K_j → Kt, V_j → Vs
 (i, j) ─ sync ─ load Q_i → Qs ; load_state: O_i, m_i, l_i → 寄存器
        ─ sync ─ tile_scores: s = Q_i K_jᵀ                     （读 Qs, Kt）
                 softmax_tile: s ← P~, m~, l~                   （寄存器 + shuffle）
                 store_p: P~ → Ps
        ─ sync ─ tile_pv: acc = P~ V_j                         （读 Ps, Vs）
                 merge_state: 写 O_i, m_i, l_i 回 HBM
 下一个 i ─ sync ─ …（Qs、Ps 可以被覆盖了）
```

每个 `sync` 保护的内容：第一个保证上一个 j 的 `tile_pv` 不再读 `Vs`；第二个保证上一个 tile 不再读 `Qs`、`Ps`；第三个让 `Qs`（和本 j 的 `Kt`、`Vs`）全部写好，也把 `load_state` 排在 `merge_state` 之前；第四个让 `Ps` 全部写好。

## 6. 分步练习

所有命令在仓库根目录运行。骨架已经写好 tile 加载、`tile_scores`（S 的点积）、`tile_pv`（P~V）、所有 barrier 和两层循环；你写的是 attention 特有的部分。

### 步骤 1 · TODO(1)：单个 K/V 块

本步只有 N ≤ 64 的 case，只有一个 K/V 块和一个 Q 块，不需要合并。`run()` 在 step 1 发射 `fa1_forward<D, true>`，它不调用任何 step 2 的函数。

1. `softmax_tile`：乘 scale、mask 掉 `col ≥ n` 的填充 key、16 线程组内求行最大值、算 `P~` 和 `l~`。
2. `write_single_block`：`O = acc / l~`，`LSE = m~ + ln(l~)`，只写 `row < n` 的行。

```bash
./lab check 12 --step 1
./lab bench 12 --step 1 --case one_block
```

`n1` case 只有一个 key：P 必须是 1，O 等于 V。如果 `n1` 得到 NaN，多半是把唯一的 key 也当成了填充，或者空集的减数没有处理。step 1 只是正确性练习，bench 的 case 太小，时间主要是 launch 开销。

### 步骤 2 · TODO(2)：完整的外层 K/V 循环

实现 Algorithm 1 的第 2、8、11–13 行和最后的 LSE：

1. `init_state`：O = 0，l = 0，m = −∞；
2. `load_state`：读出本线程负责的 O、m、l；
3. `merge_state`：3.3 节的公式，写回 O、m、l；
4. `write_lse`：`lse = m + ln(l)`。

```bash
./lab check 12 --step 2
./lab check 12 --step 2 --case two_blocks      # 先调通最小的两块 case
./lab bench 12 --step 2
```

只写 `--case` 时，只运行列出了这个 case 的 step；想指定检查哪一步，就同时写 `--step`。

写完先在纸上检查：第一块时 `m_old = −∞, l_old = 0, o_old = 0`，你的 `merge_state` 应该退化成 `write_single_block` 的结果。

### 步骤 3 · TODO(3)：causal

1. 在 `softmax_tile` 里加上 causal mask（`col > row` 设为 −∞），删掉 `if (causal) LAB_TODO(3);`；
2. `first_q_block`：返回第一个需要处理的 Q 块号。

```bash
./lab check 12                                 # 全部 step
./lab bench 12 --step 3 --case perf_causal
```

先只做第 1 项：把 `first_q_block` 里的 `if (causal) LAB_TODO(3);` 也删掉，让它对 causal 同样返回 0。bench 一次 `perf_causal`，再做第 2 项。前者和非 causal 的时间几乎一样；后者约快一倍。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| step 1 全是 NaN | 尾部 mask 把有效 key 也屏蔽了；或者全 mask 行用了 `−∞` 作减数 |
| `short_tail` 失败，`one_block` 通过 | 没有 mask 掉 `col ≥ n` 的填充 key（它们的分数是 0，不是 −∞） |
| step 2 结果全是 NaN | O 没有初始化为 0（`0 × NaN = NaN`），或 l、m 没初始化 |
| `two_blocks`、`rescale` 失败，`uniform` 通过 | 合并时少乘了 `e^(m_old − m_new)`，或乘到了 O 但没乘到 l |
| `ones_v` 的 O 不是 1 | O 多除或少除了一次 `l_new`；注意论文的 O 是已经归一化的 |
| 偶尔失败、结果不稳定 | `tx == 0` 以外的线程也写 m、l，或者读旧值的位置不在 barrier 之前 |
| `uniform` 的 LSE 不等于 `ln(N)` | LSE 用了 `log2`，或者在 `write_lse` 之前没有等所有块处理完 |
| `causal_*` 的第 0 行错 | mask 写成了 `col >= row`（把对角线也屏蔽了） |
| `causal_peaked` 失败，`causal_tail` 通过 | 跳过的块算错了：`first_q_block` 跳过了还需要处理的 Q 块 |

## 7. 性能

### 7.1 相对库基线

实测（CUDA Graph 计时，FP32，`vs efficient` 是 PyTorch memory-efficient attention 的耗时除以本版本耗时）：

| case | 版本 | `vs efficient` | % FP32 标称峰值 |
|---|---|---|---|
| `perf` [4, 4096, 64, 64] | 步骤 2 / 3 | ≈ 0.5× | ≈ 33% |
| `perf_causal` | 步骤 3，**不跳块** | ≈ 0.26× | ≈ 17% |
| `perf_causal` | 步骤 3，跳块 | ≈ 0.5× | ≈ 33% |
| `few_heads` [1, 4096, 16, 64] | 步骤 3 | ≈ 0.06× | ≈ 4% |

`% peak` 按 B200 FP32 标称 75 TFLOP/s 计。步骤 3 在 `perf` 和 `perf_causal` 上都达到基线的 0.4× 以上，说明机制成立。`--cold` 下比例不变：数据量大于 L2，本来就主要来自 HBM。

几点观察：

- **跳块让 causal 快约一倍**，正好是被跳过的工作量。
- **`few_heads` 只有 16 个 CTA**，148 个 SM 里只有 16 个在工作。库的 kernel 按 64 行的 Q 块切分，同一个 case 有 16 × 64 = 1024 个 CTA，所以差距被拉大到约 8 倍（0.5× → 0.06×）。
- **一个 SM 上 2 个 CTA 不等于 2 倍吞吐**：`few_heads` 中每个 SM 只跑 1 个 CTA，单个 head 的耗时约是 `perf`（大部分 SM 跑 2 个）的 0.57 倍，而不是 0.5 倍。

### 7.2 时间花在哪里

```bash
./lab profile 12 --case perf
```

Nsight Compute 实测结果（`perf`，比例形式）：

| 指标 | 结果 | 说明 |
|---|---|---|
| `dram__bytes_read.sum` | ≈ 44 × Q、K、V 的总大小 | 每个 j 都重读 Q 和 O，和 3.6 节的估算一致 |
| `dram__bytes_write.sum` | ≈ 67 × O 的大小 | O 被写了 Tc + 1 = 65 次，加上 m、l |
| HBM 带宽占用 | 约 15% 标称峰值 | 读写总字节数除以 `bench` 的时间；多出的流量远没有用满 HBM |
| `sm__throughput` | 约 50% | 主要是 FMA 和 shared memory 读 |
| `sm__warps_active` | 约 22% | 每个 SM 只有 8–16 个 warp |

结论：在 SIMT FP32 上，**FA1 多出的 HBM 往返不是瓶颈**，计算才是。原因是 SIMT FP32 的算力（标称 75 TFLOP/s）比 tensor core 低一个数量级：同样的 HBM 流量，摊到很长的计算时间里就不显眼了。到第 2 部分换成 tensor core 后，计算快了十几倍，这些往返就会变成主要开销，FA2 的循环交换才真正显出价值。

### 7.3 为什么比库慢约 2 倍

PyTorch 的 FP32 memory-efficient attention 在 B200 上运行的是 CUTLASS kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80`。和本关相比：

| | 本关 | 库 |
|---|---|---|
| 矩阵乘 | SIMT FFMA | TF32 tensor core，用 3 次 TF32 乘法补偿到 FP32 精度（“3xTF32”，由源码配置推断，见第 9 节；Nsight 中可以看到 tensor pipe 指令） |
| 循环顺序 | K/V 外层，O 往返 HBM | Q 块外层，O 留在寄存器（`rf`），FA2 式 |
| CTA 数 | B·H | B·H·⌈N/64⌉ |

所以库快 2 倍并不是因为它“更会写 SIMT”，而是它用了本关还没学的硬件（tensor core）和下一部分的调度。这也是本课程后面要学的内容。

### 7.4 可以自己试的实验

- `few_heads` 与 `perf` 对比：时间 × CTA 数 / SM 数，算出每个 SM 的有效吞吐。
- 把 `Bc` 加倍到 128，3.6 节的 HBM 流量减半；预测一下时间会怎样变化，再想想为什么（提示：7.2 节）。

## 8. 自测问题

1. 合并定理里，如果 `m_A = m_B = −∞`（两个空集），公式会算出什么？代码应该怎么处理？
2. 论文第 12 行先乘 `l_i` 再除以 `l_new`。如果 HBM 里存的是未归一化的分子 `l·O`，第 12 行该怎么写？省掉了哪些运算？
3. perf case（N = 4096，d = 64，`Bc = 64`）中，一个 head 的 O 被从 HBM 读了多少次？如果 `Bc = 227`（论文公式），是多少次？
4. causal、`Br = Bc = 64`、N = 4096 时，一共处理多少个 `(i, j)` tile？非 causal 呢？比例是多少？
5. 为什么 `load_state` 必须在 `__syncthreads()` 之前、`merge_state` 必须在它之后？如果 l、m 由每个线程各自保存一份（16 倍空间），还需要这个约束吗？
6. B = 1、H = 8、N = 32768 时，本关的 kernel 会用到多少个 SM？FA2 的哪个改动解决这个问题？

<details>
<summary>参考答案</summary>

1. `m = −∞`，`α = β = exp(−∞ − (−∞)) = NaN`，结果全是 NaN。合并两个空集的结果应该仍是空集：代码在 `m_new = −∞` 时保持原状态（本关的 `merge_state` 直接跳过这一行）。
2. 存 `u = l·O` 时，第 12 行变成 `u_i ← e^(m_i − m_new)·u_i + e^(m~ − m_new)·P~V_j`，不再需要乘 `l_i` 和除 `l_new`，只在最后一次 `O = u/l`。这就是 FA2 的“延迟归一化”。
3. 读 Tc = 4096/64 = 64 次（每个 j 一次）；写 65 次，即每个 j 一次再加上初始化的一次（与 7.2 节的 Tc + 1 一致）。`Bc = 227` 时 Tc = ⌈4096/227⌉ = 19，读 19 次、写 20 次。
4. Tr = Tc = 64。非 causal：64 × 64 = 4096 个。causal：对每个 j 处理 `i = j … 63`，共 64 + 63 + … + 1 = 2080 个，比例 2080/4096 ≈ 0.51，和实测的加速比一致。
5. 一行的 16 个线程都要读旧的 m、l，但只有一个线程写新值；barrier 保证所有读发生在写之前。若每个线程各存一份并只读写自己的那份，就没有跨线程的读写，不再需要这个 barrier（但 HBM 流量变成 16 倍）。
6. 8 个 CTA，只用 8 个 SM（约 5%）。FA2 把每个 head 再按 Q 块切成 ⌈N/Br⌉ 个 CTA，CTA 之间没有共享状态（每个 CTA 拥有自己的 O_i），并行度变成 B·H·⌈N/Br⌉。
</details>

## 9. 延伸阅读

- Tri Dao 等，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（2022）：第 3.1 节与 Algorithm 1（本关的算法），第 3.2 节定理 2（IO 复杂度）与命题 3（下界），附录 C（证明）。
- Tri Dao，[FlashAttention-2](https://arxiv.org/abs/2307.08691)（2023）：第 3.1–3.2 节描述了 FA1 的并行方式（每个 head 一个 thread block）和循环交换的动机，是下一部分的预习。
- Milakov 与 Gimelshein，[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（2018）：online softmax 的原始论文。
- CUDA C++ Programming Guide：[Synchronization Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-functions)（`__syncthreads()` 对 global memory 的可见性保证），[Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)，[Shared Memory（动态分配超过 48 KB）](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)。
- 上游源码（`third_party/flash-attention`，固定版本）：原始 FA1 CUDA 实现属于 flash-attn 1.x，不在本仓库固定的版本中。`csrc/flash_attn/src/flash_fwd_kernel.h` 和 `csrc/flash_attn/src/softmax.h`（`softmax_rescale_o`）是 FA2 的实现，可以对照本关的 `merge_state` 阅读“循环交换后状态留在寄存器里”是什么样子。
- 库基线的源码：`third_party/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h`（PyTorch memory-efficient attention 的来源），FP32 走 tensor core 的配置见同目录 `gemm_kernel_utils.h` 中的 `DefaultGemmType`（`OpMultiplyAddFastF32`）。
