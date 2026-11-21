# 01 · 归约与数值稳定：行 softmax

> 部分 0 · 预备　｜　前置：[00_transpose](../00_transpose/README.md)　｜　代码：[`students/01_softmax/kernel.cu`](../../students/01_softmax/kernel.cu)

## 1. 本关目标

对一个 FP32 矩阵的每一行做 softmax：`y[r, :] = softmax(x[r, :])`。softmax 需要先知道整行的最大值和指数和，所以它是一个**归约**问题：很多线程各算一部分，再把结果合在一起。本关用四个版本把它从“短行上比 torch 慢约十倍”做到在三个 perf case 上都快于 torch。做完后你应该能：

- 用 shared memory 和 `__shfl_xor_sync` 写出 block 级和 warp 级归约，说清每个 `__syncthreads()` 保护什么；
- 解释为什么 softmax 要减去最大值，以及 `ex2.approx` 的精度在什么时候够用、什么时候不够用；
- 推导 **online softmax** 的状态 `(m, l)` 和它的合并规则，证明合并满足结合律；
- 把整行留在寄存器里，用 `float4` 访存，按行宽选择“一个 warp 一行”或“一个 block 一行”。

online softmax 是整个课程最重要的一个公式。从 [12_fa1](../12_fa1/README.md) 开始，每一个 attention kernel 都在用它。

## 2. 动机：softmax 为什么挡住了分块

attention 是 `O = softmax(S) V`，其中 `S = Q Kᵀ` 的每一行有 N 个元素。标准实现把整个 `S`（N × N）写回 HBM，再读回来做 softmax，再写 `P`，再读 `P` 做 `P V`。N = 8192 时一个 head 的 FP32 `S` 就有 256 MB。

FlashAttention 想把 K、V 分成小块，每次只在片上处理一块，永远不把 `S` 写回 HBM。障碍就是 softmax：分母 `Σ_j e^{s_j}` 需要**整行**，而处理第一块时还没见过后面的块。FlashAttention 论文第 3.1 节的回答是：维护每行的两个数（最大值 `m` 和指数和 `l`），每来一块就把旧结果按新的最大值**重新缩放**。这个技巧来自 Milakov 与 Gimelshein 2018 年的论文 *Online normalizer calculation for softmax*。

本关先在最简单的场景——普通的行 softmax——把这件事做透：

| 场景 | 一行的数据从哪来 | online softmax 的作用 |
|---|---|---|
| 本关步骤 3 | 一个线程沿行扫描 x | 一遍读出 `(m, l)`，少读一遍 x |
| 本关步骤 2–4 的归约 | 很多线程各持有一段 | 各段的 `(m, l)` 按任意顺序合并 |
| FlashAttention | K/V 一块一块算出 `S` 的一段 | 不用保存整行 `S` |

## 3. 数学

### 3.1 定义与数值稳定

对一行 `x_0, …, x_{n-1}`：

```
y_j = e^{x_j} / Σ_k e^{x_k}
```

FP32 能表示的最大数约 3.4 × 10³⁸ ≈ e^{88.7}。`large` case 里 `|x|` 可达几千，直接算 `e^{x}` 会得到 `inf`，`inf / inf = NaN`。

softmax 对整行加同一个常数不变：`e^{x_j - c} / Σ_k e^{x_k - c}` 与原式相等。取 `c = m = max_k x_k`，则每个指数 `x_j - m ≤ 0`，`e^{x_j - m} ∈ [0, 1]`，分母 `l = Σ_k e^{x_k - m} ∈ [1, n]`（最大的那一项恰好是 1）。这叫 **safe softmax**：

```
m = max_k x_k
l = Σ_k e^{x_k - m}
y_j = e^{x_j - m} / l
```

`-inf` 的元素（`masked` case，相当于 attention 里被 mask 掉的位置）贡献 `e^{-inf} = 0`。每行至少有一个有限值，所以 `m` 有限，`l ≥ 1`。

### 3.2 三遍算法与访存次数

按上面的式子直接写，每行要扫三遍：

| 遍 | 读 | 写 | 得到 |
|---|---|---|---|
| 1 | x | — | m |
| 2 | x | — | l |
| 3 | x | y | y |

如果每一遍都从 HBM 读，总流量是 `3 × 读 + 1 × 写`，下界（读一遍写一遍）的 2 倍。第 2、3 遍能不能命中缓存，取决于“同时在处理的行”能不能装进 L2，第 7 节会实测。

### 3.3 online softmax：一遍得到 (m, l)

定义一个元素集合 `S` 的**状态**：

```
state(S) = (m_S, l_S),   m_S = max_{j∈S} x_j,   l_S = Σ_{j∈S} e^{x_j - m_S}
```

空集的状态规定为 `(-inf, 0)`，只有一个元素 `x` 的集合是 `(x, e^{x-x}) = (x, 1)`。

**合并规则。** 设 `A`、`B` 不相交，`m = max(m_A, m_B)`。把 `A` 的每一项改写成以 `m` 为基准：

```
e^{x_j - m} = e^{x_j - m_A} · e^{m_A - m}
```

对 `j ∈ A` 求和得到 `l_A · e^{m_A - m}`，`B` 同理。于是

```
merge((m_A, l_A), (m_B, l_B)) = ( m,  l_A · e^{m_A - m} + l_B · e^{m_B - m} ),   m = max(m_A, m_B)
```

**逐元素更新**就是和单元素集合 `(x, 1)` 合并：

```
m_new = max(m, x)
l_new = l · e^{m - m_new} + e^{x - m_new}
```

`e^{m - m_new}` 是“旧的和要缩小多少”：新元素比旧最大值大时，旧的每一项都要按新基准变小；否则它等于 1，`l` 只是加上新的一项。

**为什么可以任意顺序合并。** 上面的推导说明 `merge(state(A), state(B)) = state(A ∪ B)`：合并的结果只取决于集合本身，与集合是怎样拆分的无关。因此

```
merge(merge(a, b), c) = state(A ∪ B ∪ C) = merge(a, merge(b, c))     （结合律）
merge(a, b) = state(A ∪ B) = merge(b, a)                             （交换律）
merge((-inf, 0), a) = a                                              （单位元）
```

结合律与交换律让我们可以用任何形状的树来合并：每个线程先顺序扫描自己的元素，再在 warp 内用 butterfly 合并，再在 block 内合并。这和 `+`、`max` 能做并行归约是同一个道理。（在浮点下，和普通的求和一样，结合律只是近似成立：不同的合并顺序舍入不同，差别在几个 ulp 以内。）

**`-inf` 的陷阱。** 若 `m_A = m_B = -inf`（两个空集，或者都是被 mask 的元素），`m = -inf`，`m_A - m = (-inf) - (-inf) = NaN`。所以 `merge` 必须单独处理：新的 `m` 为 `-inf` 时直接返回 `(-inf, 0)`。`m_A = -inf` 而 `m` 有限时，`e^{-inf} = 0`，不用特殊处理。

**最后一步**与三遍算法相同：`y_j = e^{x_j - m} / l`。online 版本只是把前两遍合成了一遍：读 x 两次，而不是三次。

### 3.4 手算小例

一行 `x = [2, 1, 4, 3]`。顺序扫描：

| 读入 x | m_new | l_new = l · e^{m - m_new} + e^{x - m_new} |
|---|---|---|
| （初始） | -inf | 0 |
| 2 | 2 | 0 · 0 + e⁰ = 1 |
| 1 | 2 | 1 · 1 + e⁻¹ = 1.367879 |
| 4 | 4 | 1.367879 · e⁻² + e⁰ = 0.185122 + 1 = 1.185122 |
| 3 | 4 | 1.185122 · 1 + e⁻¹ = 1.553002 |

检验：`e⁻² + e⁻³ + e⁰ + e⁻¹ = 0.135335 + 0.049787 + 1 + 0.367879 = 1.553002`。

换一种拆法：两个线程分别处理 `A = [2, 1]`、`B = [4, 3]`，得到 `(2, 1.367879)` 和 `(4, 1.367879)`。合并：

```
m = max(2, 4) = 4
l = 1.367879 · e^{2-4} + 1.367879 · e^{4-4} = 0.185122 + 1.367879 ≈ 1.553002
```

结果相同（表中各数已四舍五入到 6 位小数，末位可能差 1）。输出 `y = e^{x - 4} / 1.553002 = [0.087144, 0.032059, 0.643914, 0.236883]`。

### 3.5 延迟缩放：步骤 4 与 FlashAttention 的同一个手法

步骤 4 中每个线程持有若干元素。它先用**自己的**最大值 `m_t` 算出 `v_j = e^{x_j - m_t}` 和 `l_t = Σ v_j`，然后把所有线程的 `(m_t, l_t)` 合并成整行的 `(M, L)`。输出时不必重算指数：

```
y_j = e^{x_j - M} / L = v_j · ( e^{m_t - M} / L )
```

括号里是每个线程一个的标量。这样每个元素只算一次指数，整行只做一次归约（而不是 max 和 sum 两次）。FlashAttention 对输出累加器做的是同一件事：`O ← O · e^{m_old - m_new} + P V`，先按旧的最大值累加，最大值变了再整体乘一个缩放因子。

## 4. 硬件与指令

### 4.1 block 归约：shared memory 树

256 个线程各有一个值 `v`，要得到它们的最大值（或和）：

```
smem:  v0 v1 v2 v3 v4 v5 v6 v7        （8 个线程的例子）
s=4:   t<4 做 smem[t] = op(smem[t], smem[t+4])
       a0 a1 a2 a3 .. .. .. ..
s=2:   b0 b1 .. ..
s=1:   c0                             所有线程读 smem[0]
```

写入 `smem` 之后、每一层之后都要 `__syncthreads()`：下一层读的是别的线程刚写的结果。256 个线程共 log₂256 = 8 层，加上写入后的一次，共 9 次。读完 `smem[0]` 之后还要再同步一次：同一个 `smem` 马上要被下一次归约复用，不同步的话，快的线程可能在慢线程读 `smem[0]` 之前就把它覆盖掉。

### 4.2 warp shuffle

同一个 warp 的 32 个线程可以直接读彼此的寄存器，不经过 shared memory：

```cpp
float other = __shfl_xor_sync(0xffffffff, v, offset);  // 读 lane (lane ^ offset) 的 v
```

- 第一个参数是参与的 lane 掩码。`0xffffffff` 表示 32 个 lane 都要执行这条指令；如果有 lane 因为分支没有执行到，结果未定义。所以没有元素的线程也要参与 shuffle，带着单位元（`-inf` 或 0）。
- `offset` 依次取 16、8、4、2、1，每轮 lane `i` 与 lane `i ^ offset` 交换并合并。这叫 butterfly：5 轮后**每个** lane 都持有全部 32 个值的归约结果。

```
offset 4:  0↔4  1↔5  2↔6  3↔7      （8 个 lane 的例子，3 轮）
offset 2:  0↔2  1↔3  4↔6  5↔7
offset 1:  0↔1  2↔3  4↔5  6↔7
```

shuffle 不需要 `__syncthreads()`：warp 内的数据交换由这一条指令完成。block 归约因此变成“warp 内 5 轮 shuffle → 每个 warp 的 lane 0 把结果写进 `smem[warp]` → 一次同步 → 再用 shuffle 归约 8 个部分结果”，barrier 从 10 次降到 2 次。

合并 `(m, l)` 状态时，两个字段各 shuffle 一次，再调用 `merge`。

### 4.3 寄存器

| 资源 | B200 每 SM |
|---|---|
| 寄存器文件 | 65536 个 32 位寄存器 = 256 KB |
| 每线程最多 | 255 个寄存器 |
| 线程 | 最多 2048 个（64 个 warp），每 block 最多 1024 个 |
| shared memory | 最多 228 KB |

一个 SM 能同时驻留多少 block，受寄存器、shared memory、线程数三者中最紧的那个限制。例如每线程 50 个寄存器、每 block 1024 线程时，一个 block 就用掉 51200 个寄存器以上，每 SM 只能驻留 1 个 block。

数组要留在寄存器里，所有下标都必须在编译期确定：循环要用 `#pragma unroll` 完全展开，下标里不能出现运行时变量。否则编译器把数组放到 local memory（实际在 HBM/L1 上），`./lab sass 01` 的 `LOCAL` 一栏会变成非零。

`__launch_bounds__(N)` 告诉编译器这个 kernel 每个 block 最多 N 个线程，编译器据此限制寄存器用量（1024 线程时每线程最多 64 个）。

### 4.4 指数：`ex2.approx` 与精度

GPU 的特殊函数单元（SFU，SASS 中的 `MUFU`）直接支持以 2 为底的指数，但它的吞吐比 FMA 单元低得多（第 41 关会细算这个瓶颈）。利用 `e^{t} = 2^{t · log₂e}`，一次指数只需要一条 `MUFU.EX2`：

```cpp
__device__ float ex2(float x) {           // 骨架已提供
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ float exp_shifted(float x, float m) { return ex2((x - m) * kLog2e); }
```

- PTX ISA 对 `ex2.approx.f32` 的说明是：在全部输入范围内，与正确舍入的结果相比最大误差为 2 ulp（“The maximum ulp error is 2 ulp from correctly rounded result across the full range of inputs”），相对误差不超过约 2.4 × 10⁻⁷；`ex2(-inf) = +0`。
- `.ftz`（flush to zero）：结果小于 2⁻¹²⁶ ≈ 1.2 × 10⁻³⁸ 时直接返回 0，省掉次正规数的处理。
- 本关容差是 `|err| ≤ 1e-8 + 1e-4 · |expected|`（FP64 的 `torch.softmax` 作参考），这两项误差都远小于它。

三种写法编译出的 SASS（`x - m` 已算好）：

| 写法 | 指令 | 说明 |
|---|---|---|
| `ex2((x - m) * kLog2e)` | `FADD`、`FMUL`、`MUFU.EX2` | 本关用法 |
| `exp2f((x - m) * kLog2e)` | 再加 `FSETP` 和 2 条带谓词的 `FMUL` | 正确处理次正规结果 |
| `expf(x - m)` | 约 8 条 `FFMA`/`FADD`/`SHF` + `MUFU.EX2` + `FMUL` | 先把指数拆成整数和小数两部分，最大误差 2 ulp |

`-use_fast_math` 会把 `expf(t)` 换成 `__expf(t)`，即 `ex2(t * log₂e)`，CUDA 文档给出的误差界是 `2 + ⌊1.173·|t|⌋` ulp：参数越大越不准。先减去最大值之后，对结果有贡献的元素的 `t = x - m` 都接近 0（`t < -100` 的项小于 e⁻¹⁰⁰，可以忽略），误差界也就很小。

**先减后乘。** `shifted` case 每行加了 5000。`x - m` 两个相近的数相减是精确的（Sterbenz 引理：`m/2 ≤ x ≤ 2m` 时 `x - m` 没有舍入），乘以 `log₂e` 只有一次舍入，相对误差约 6 × 10⁻⁸。很多代码（包括 FlashAttention 的源码）写成一条 FMA：`fmaf(x, log2e, -m * log2e)`，省一条指令。这时 `m · log₂e ≈ 7200` 先被舍入成 FP32，记舍入误差为 `δ(m)`，`|δ(m)|` 最多是半个 ulp，即 2⁻¹² ≈ 2.4 × 10⁻⁴。于是 FMA 写法算出的是 `e^{x-m} · 2^{-δ(m)}`：多了一个只依赖平移量 `m` 的因子。

- **步骤 1、2**：整行只用一个平移量 `M`，分子 `e^{x-M}` 和分母 `Σ e^{x-M}` 带着同一个因子 `2^{-δ(M)}`，相除后正好抵消。FMA 写法在这两步也能通过 `shifted`。
- **步骤 3、4**：部分状态各用自己的平移量 `m_t`（步骤 3 是每个线程逐元素更新的 running max，步骤 4 是线程的局部最大值），`l_t` 里带着 `2^{-δ(m_t)}`；合并时乘的 `e^{m_t - M}` 也用 FMA 写法算，只多出 `2^{-δ(M)}`，消不掉 `2^{-δ(m_t)}`。不同的 `m_t` 舍入误差不同，分母就成了带不同权重的和，与分子不再抵消。本机实测这种写法在步骤 3、4 的 `shifted` 上失败，最大相对误差约 2.5–4 × 10⁻⁴。

attention 里 logit 通常只有几十，`δ` 在 10⁻⁶ 量级，输出又是 BF16，所以 FA 用 FMA 没问题；本关输出是 FP32、容差 1e-4，就用先减后乘。另一种写法“先把 `x` 乘以 `log₂e` 再减”更糟：每个元素的 `x · log₂e` 各自舍入，误差连步骤 1、2 也抵消不掉。

### 4.5 向量化访存与缓存提示

`float4` 一次读写 16 字节（SASS 中 `LDG.E.128` / `STG.E.128`）。一个 warp 的一条指令就搬 512 字节，指令数是标量的 1/4，同样的指令数能让 4 倍的数据在路上（第 00 关的 Little 定律）。条件是地址 16 字节对齐：每行起点 `x + r * cols` 对齐要求 `cols % 4 == 0` 且基地址对齐。`run()` 已经做了这个判断，不满足时用标量版本。

`__ldcs` / `__stcs`（PTX `ld.global.cs` / `st.global.cs`，cache streaming）告诉缓存“这个数据只用一次，优先淘汰”。x 和 y 在步骤 4 中都只碰一次。

## 5. 调度设计

### 步骤 1–3：一个 block 一行

```
grid = rows, block = 256 线程
线程 t 处理第 t, t+256, t+512, … 列      ← warp 内 32 个线程读连续的 32 个 float，合并
```

| 步骤 | 扫描 x 的次数 | 每行 barrier | shared memory |
|---|---|---|---|
| 1 | 3 | 2 次归约 × 10 = 20 | 256 个 float |
| 2 | 3 | 2 次归约 × 2 = 4 | 32 个 float |
| 3 | 2 | 1 次合并 × 2 = 2 | 32 个 `State` |

### 步骤 4：整行在寄存器

`kThreads` 个线程负责一行，每线程 `kVecs` 个 chunk，每个 chunk 是 `V` 个连续 float（`V = 4` 时是一个 `float4`）。chunk `k` 的起始列是 `(k · kThreads + t) · V`：

```
kThreads = 32, kVecs = 2, V = 4（cols = 256）：
列     0   4   8  ...  124 | 128 132 ... 252
chunk  k=0: lane 0 1 2 ... 31 | k=1: lane 0 1 ... 31
       ↑ 一条 LDG.E.128 读 512 字节        ↑ 第二条
```

`run()` 选能装下一行的最小配置（`chunks = cols / V`，向上取整）：

| chunks | kThreads | kVecs | 每 block | 每线程 float 数（V = 4） |
|---|---|---|---|---|
| ≤ 32 / 64 / 128 / 256 | 32（一个 warp 一行） | 1 / 2 / 4 / 8 | 4 行，128 线程 | 4–32 |
| ≤ 512 / 1024 / 2048 | 128 / 256 / 512 | 4 | 1 行 | 16 |
| ≤ 4096 | 512 | 8 | 1 行 | 32 |
| ≤ 32768 / V | 1024 | 32 / V | 1 行 | 32 |
| 更大 | 退回步骤 3 | | | |

三个 perf case 的配置：

| case | 配置 | 每线程寄存器 | 每 SM 驻留 |
|---|---|---|---|
| rows256 | warp 一行，`kVecs = 2` | 27 | 16 个 block = 64 行 |
| rows4096 | 256 线程一行，`kVecs = 4` | 40 | 6 个 block = 6 行 |
| rows32768 | 1024 线程一行，`kVecs = 8` | 50 | 1 个 block = 1 行 |

一个 block 的时间线是“全部 load → 归约 → 全部 store”。SM 上驻留多个 block 时，一个 block 在归约或 store，另一个 block 的 load 正在路上，HBM 一直有活干。rows32768 每 SM 只有一行（128 KB 的行已经占了半个寄存器文件），三段无法重叠，这就是它比另两个 case 慢的原因（第 7 节）。

## 6. 分步练习

所有命令在仓库根目录运行。每一步只要求对应的 TODO。

### 步骤 1 · TODO(1)：三遍 softmax，shared memory 树形归约

实现 `block_reduce_tree`（TODO(1a)）和 `softmax_three_pass` 的三遍循环（TODO(1b)）。

```bash
./lab check 01 --step 1
./lab bench 01 --step 1
./lab bench 01 --step 1 --case rows256
```

- 先用最小的 case 调试：`./lab check 01 --step 1 --case tiny`。
- 循环写成 `for (int c = threadIdx.x; c < cols; c += kBlock)`；`one_col` 中 255 个线程一个元素都没有，它们的局部最大值是 `-inf`、局部和是 0，正好是单位元。
- 步骤 2 会以 `kShuffle = true` 复用这个 kernel，所以三遍里都调用 `block_reduce<kShuffle>`，不要直接调用 `block_reduce_tree`。

### 步骤 2 · TODO(2)：warp shuffle 归约

实现 `warp_reduce` 和 `block_reduce_shfl`。kernel 本身不用改。

```bash
./lab check 01 --step 2
./lab bench 01 --step 2 --case rows256
./lab profile 01 --step 2 --case rows256
```

对比步骤 1、2 在 profile 中的 `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`：平均每发射一条指令，有多少个 warp 正卡在 barrier 上。我们测到步骤 1 约 2.6、步骤 2 约 1.4。`rows256` 每行只有 256 个元素，时间主要花在归约上，所以这个 case 对归约的写法最敏感。

### 步骤 3 · TODO(3)：online softmax

实现 `merge`、`warp_merge`、`block_merge` 和 `softmax_online`。

```bash
./lab check 01 --step 3 --case masked
./lab check 01 --step 3
./lab bench 01 --step 3 --case rows32768
./lab profile 01 --step 3 --case rows32768
```

- 第一遍循环写 `s = merge(s, State{xr[c], 1.f});`。看上去可以省一次指数：`x > m` 时只算 `e^{m-x}`，否则只算 `e^{x-m}`。我们实测这种写法在 `rows4096` 上慢 1.6 倍以上：分支让编译器不敢把后面几次 load 提前发出（SASS 中每次只有 1–2 个 `LDG` 在路上，无分支写法是连续 8 个）。这个 kernel 的瓶颈是在途访存，不是指数运算。
- profile 中 `dram__bytes_read.sum` 对 rows32768 应约为 x 的 2 倍（512 MB × 2），步骤 1 约为 3 倍。

### 步骤 4 · TODO(4)：整行留在寄存器

实现 `softmax_regs`。配置选择、`load_chunk`/`store_chunk`、`row_merge` 都已写好。

```bash
./lab check 01 --step 4 --case short_rows
./lab check 01
./lab sass 01                         # 每个 softmax_regs 的 LOCAL 应为 0
./lab bench 01 --case rows4096
./lab bench 01 --case rows32768
./lab bench 01 --case rows256
./lab bench 01 --case rows256 --cold
```

- 局部最大值为 `-inf` 的线程（`one_col` 中没有元素的线程，`masked_short` 中元素全是 `-inf` 的线程）：平移量取 0，`v` 全部变成 `e^{-inf} = 0`，`l = 0`。最后的缩放因子 `e^{m_t - M}` 也是 0。
- `huge_row` 的 65537 列放不进寄存器，`run()` 会改用步骤 3 的 kernel，所以这个 case 也依赖 TODO(3)。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `large` 出现 NaN | 没有减去最大值，`e^{x}` 溢出 |
| `masked`、`one_col`、`masked_short` 出现 NaN | `(-inf) - (-inf)`：`merge` 或步骤 4 的局部平移没有处理 `m = -inf` |
| 步骤 3、4 只有 `shifted` 失败，相对误差约 2e-4–4e-4 | 用了 `fmaf(x, log2e, -m * log2e)`：各部分状态平移量的舍入误差在合并时消不掉（4.4 节） |
| 所有步骤只有 `shifted` 失败 | 先把 `x` 乘以 `log2e` 再减：每个元素的舍入误差各不相同 |
| 结果偶尔错、不稳定 | 读完 `smem` 后缺少 `__syncthreads()`，下一次归约覆盖了还没读的值 |
| `TIMEOUT` 或结果错，只在 block 一行的配置 | `__syncthreads()` 放在了部分线程不执行的分支里（例如没有元素的线程提前 `return`） |
| 步骤 2 只有 lane 0 结果对 | 用了 `__shfl_down_sync` 的归约，结果只在 lane 0；本关的写法要求每个 lane 都拿到结果 |
| 只有 `short_rows` 在步骤 4 失败 | 最后一个 block 中 `row >= rows` 的 warp 访问了越界的行 |
| 只有 `tail` 或 `tiny` 在步骤 4 失败 | 越界的 chunk 没有填 `-inf`，或存储时没有判断 `c < cols` |
| 步骤 4 正确但很慢，`LOCAL` 非零 | 循环没有完全展开，`v` 被放进了 local memory |

## 7. 性能

本机实测（CUDA Graph 计时；`vs torch` 是 PyTorch softmax（`aten::_softmax`）的耗时除以本版本耗时）：

| 版本 | rows4096 `[16384, 4096]` | rows32768 `[4096, 32768]` | rows256 `[262144, 256]` |
|---|---|---|---|
| 步骤 1 | ≈ 1.9× | ≈ 0.6–0.75× | ≈ 0.1× |
| 步骤 2 | ≈ 1.9× | ≈ 0.75× | ≈ 0.18× |
| 步骤 3 | ≈ 2.0× | ≈ 1.0× | ≈ 0.15× |
| 步骤 4 | ≈ 2.4×（HBM 标称峰值的 84%） | ≈ 1.2×（68%） | ≈ 1.08×（84%） |

torch 在这三个 case 上分别约为标称峰值的 35%、56%、77%。步骤 4 在三个 case 上都达到 torch 的 0.9 倍以上，即说明本关的机制成立。较慢的版本上 `vs torch` 偶尔会跳动（同一输入下 torch 有时会慢 1.6 倍左右），多跑一次即可；比较你的步骤之间的耗时更可靠。

**用 profile 解释这张表。** 下面是 `dram__bytes_read.sum`（HBM 读）和 `lts__t_sectors_srcunit_tex_op_read.sum × 32 B`（L2 读），以 x 的大小为单位：

| 版本 | rows4096 HBM 读 | rows4096 L2 读 | rows32768 HBM 读 | rows32768 L2 读 |
|---|---|---|---|---|
| 步骤 1 | 1.0 | 2.7 | 3.0 | 3.0 |
| 步骤 3 | 1.0 | 1.8 | 2.0 | 2.0 |
| 步骤 4 | 1.0 | 1.0 | 1.0 | 1.0 |

- **rows4096**：一行 16 KB，步骤 1–3 每 SM 驻留 8 个 256 线程的 block，全 GPU 同时在处理的行约 148 × 8 × 16 KB ≈ 19 MB，远小于 126 MB 的 L2。第 2、3 遍命中 L2，HBM 只读一遍。三遍算法在这里并不吃亏，步骤 4 的收益来自更少的指令和 L2 访问。
- **rows32768**：一行 128 KB，同时在处理的约 148 × 8 × 128 KB ≈ 155 MB，超过 L2，重读回到 HBM。这时 online softmax 少读一遍的价值就体现出来了：步骤 3 比步骤 1 快约 1.3–1.6 倍。FlashAttention 面对的正是这种情况：一行 `S` 太大，放不进片上存储。
- **rows256**：步骤 1–3 中 256 个线程每人只分到 1 个元素，大部分时间在等 barrier 和调度新 block。步骤 4 一个 warp 一行，没有 barrier（barrier stall 为 0），每行只要 2 条 `float4` load 指令；`sm__sass_inst_executed_op_global_ld.sum` 从步骤 1 的约 629 万条降到约 52 万条。它比步骤 2 快约 6 倍。
- **rows32768 的步骤 4** 只到 68%：`sm__warps_active` 约 43%，每 SM 只驻留一个 1024 线程的 block，load、归约、store 三段不能和其他 block 重叠（第 5 节）。

`./lab profile` 会锁定时钟，`gpu__time_duration` 比 `bench` 慢，只用它比较指标。

## 8. 自测问题

1. `x = [1000, 1000]` 时，不减最大值直接算 softmax，FP32 会得到什么？减去最大值后呢？
2. 在 3.4 节的例子中，按 `A = [2, 4]`、`B = [1, 3]` 拆分，写出 `state(A)`、`state(B)` 和合并结果。
3. 合并 `(m_A, l_A) = (-inf, 0)` 与 `(m_B, l_B) = (5, 2)` 时，`merge` 的公式里会出现哪些 `e^{…}`？会产生 NaN 吗？如果两个都是 `(-inf, 0)` 呢？
4. warp shuffle butterfly 有 5 轮。如果一个 warp 中只有 20 个 lane 执行到 `__shfl_xor_sync(0xffffffff, …)`，会怎样？本关的代码如何避免这种情况？
5. 步骤 4 中 rows32768 用 1024 线程 × 32 个 float。为什么不用 512 线程 × 64 个 float 来让每 SM 驻留更多行？
6. FlashAttention 的源码用 `exp2f(s * scale_log2 - max_scaled)`（一条 FMA）。为什么它在 attention 中可以接受，在本关步骤 3、4 的 `shifted` case 上却不行？步骤 1、2 为什么又没问题？

<details>
<summary>参考答案</summary>

1. `e^{1000} = inf`，`inf / (inf + inf) = NaN`。减去最大值后两个指数都是 `e⁰ = 1`，结果 `[0.5, 0.5]`。
2. `state(A) = (4, e^{2-4} + 1) = (4, 1.135335)`，`state(B) = (3, e^{1-3} + 1) = (3, 1.135335)`。合并：`m = 4`，`l = 1.135335 · 1 + 1.135335 · e^{-1} = 1.135335 + 0.417666 ≈ 1.553002`，与 3.4 节相同。
3. `m = 5`，`l = 0 · e^{-inf - 5} + 2 · e^{0} = 0 · 0 + 2 = 2`，没有 NaN。两个都是 `(-inf, 0)` 时 `m = -inf`，`e^{(-inf) - (-inf)} = e^{NaN}`，`0 · NaN = NaN`，所以 `merge` 要在 `m = -inf` 时直接返回 `(-inf, 0)`。
4. 掩码声明 32 个 lane 都参与，而有 lane 没执行到，行为未定义（可能读到垃圾或挂起）。本关所有线程都执行 shuffle，没有元素的线程带着单位元 `-inf`、0 或 `(-inf, 0)` 参与。
5. 寄存器文件每 SM 只有 256 KB，一行 128 KB 的数据本身就占了一半；加上地址、循环变量等，两行放不下。512 × 64 的配置每线程要 100 个以上寄存器，每 SM 仍只能驻留一行，而且每个 SM 上的 warp 更少。要多驻留一行，得把一部分数据放进 shared memory，或者把一行拆给多个 SM（第 33 关的 cluster）。
6. FMA 写法先把 `m · log₂e` 舍入成 FP32，误差 `δ(m)` 最多是它所在区间的半个 ulp，算出的是 `e^{x-m} · 2^{-δ(m)}`。步骤 1、2 整行只有一个平移量 `M`，分子分母带同一个因子 `2^{-δ(M)}`，相除抵消。步骤 3、4 的部分状态用不同的平移量 `m_t`，各自带 `2^{-δ(m_t)}`，合并时乘的 `e^{m_t - M}` 消不掉它们，分母成了带不同权重的和。`shifted` 中 `m · log₂e ≈ 7200`，半个 ulp 是 2⁻¹² ≈ 2.4 × 10⁻⁴，这些权重的差别让输出相对误差达到 10⁻⁴ 量级（实测约 2.5–4 × 10⁻⁴），超过本关 1e-4 的相对容差。FA 同样是 online softmax，但 attention 的 logit 通常只有几十，`δ` 在 10⁻⁶ 量级，而输出是 BF16（相对精度约 4 × 10⁻³），可以忽略。

</details>

## 9. 延伸阅读

- Tri Dao 等，*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（2022，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)），第 3.1 节：分块与 softmax 的重新缩放。
- Maxim Milakov, Natalia Gimelshein，*Online normalizer calculation for softmax*（2018，[arXiv:1805.02867](https://arxiv.org/abs/1805.02867)）：online softmax 的原始论文，含单遍算法与并行合并。
- Tri Dao，*FlashAttention-2*（2023，[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)），第 3.1 节：把输出的除以 `l` 推迟到最后（与本关步骤 4 的延迟缩放相同）。
- CUDA Programming Guide：[Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)，[Mathematical Functions（各函数的 ulp 误差界）](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- PTX ISA：[ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2)，[ld 的缓存操作符（.cs）](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
- Mark Harris，*Optimizing Parallel Reduction in CUDA*（NVIDIA 经典讲稿）：树形归约的各种写法。
- 上游源码（`third_party/flash-attention`，提交 `1fc0221`）：
  - `csrc/flash_attn/src/softmax.h`：FA2 的 `softmax_rescale_o`，`Check_inf` 与 `max == -INFINITY ? 0 : max` 的处理；
  - `flash_attn/cute/softmax.py`：FA4（CuTe DSL）的 `Softmax.online_softmax`、`rescale_O` 与 `finalize`（LSE = `m + log l`）。
