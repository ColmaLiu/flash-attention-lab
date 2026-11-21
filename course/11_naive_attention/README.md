# 11 · 标准 attention 与 IO 分析

> 部分 1 · FA1　｜　前置：[01_softmax](../01_softmax/README.md)、[10_simt_gemm](../10_simt_gemm/README.md)　｜　代码：[`students/11_naive_attention/kernel.cu`](../../students/11_naive_attention/kernel.cu)

## 1. 本关目标

用三个 kernel 拼出最直接的 attention：先算分数矩阵 S，再按行做 softmax 得到 P，最后算 O = P·V。S 和 P 都是 N×N，完整地写进 HBM。这就是 FlashAttention 论文里的 “standard attention”（Algorithm 0）。做完后你应该能：

- 写出 attention 的定义，包括 scale、causal mask、数值稳定的 softmax 和自然对数的 LSE；
- 在 `[B, N, H, D]` 布局里用 stride 取出一个 head，不复制、不转置；
- 逐个 kernel 数出 HBM 读写字节数，说清楚 S/P 的 Θ(N²) 流量和 Q/K/V/O 的 Θ(N·D) 流量差多少；
- 用 `./lab profile` 的 DRAM 计数验证这个分析，并解释它在什么时候会和估算不一致。

矩阵乘法用的是现成的 GEMM（[10_simt_gemm](../10_simt_gemm/README.md) 学过的分块 SIMT GEMM，骨架已提供），softmax 用 [01_softmax](../01_softmax/README.md) 的 online 归约。本关的新内容是**把它们正确地组合起来，并量出组合的代价**。

## 2. 动机：为什么需要 FlashAttention

FlashAttention 论文第 2.2 节先给出标准实现：

```
Algorithm 0（标准 attention）
1. 从 HBM 读 Q、K，计算 S = Q Kᵀ，把 S 写回 HBM
2. 从 HBM 读 S，计算 P = softmax(S)，把 P 写回 HBM
3. 从 HBM 读 P、V，计算 O = P V，把 O 写回 HBM
```

论文的论点是：S 和 P 有 N² 个元素，而 Q、K、V、O 只有 N·D 个；当 N 远大于 D 时，读写 S/P 的 HBM 流量占绝大部分，时间也被它决定。另一个更直接的问题是显存：本关的 `perf` case（B=1，H=16，N=4096）光 S 就要 1 GiB；N 增大到 64K 时要 256 GiB，超过 B200 的 HBM 容量（约 179 GiB）。

本关先把 Algorithm 0 写出来，然后测量这句话在 B200 上、在 FP32 下到底成立多少。[12_fa1](../12_fa1/README.md) 再把 S/P 留在片上。

## 3. 数学

### 3.1 定义

固定一个 batch `b` 和一个 head `h`，Q、K、V 都是 N×D 矩阵（N 行是序列位置，D 列是特征）。

```
S[i][j] = scale · Σ_d Q[i][d] · K[j][d]                  （i, j = 0..N-1）
可见：  j ≤ i（causal）   或   所有 j（非 causal）
m_i   = max_{j 可见} S[i][j]
l_i   = Σ_{j 可见} exp(S[i][j] − m_i)
P[i][j] = exp(S[i][j] − m_i) / l_i     （j 可见）；   0   （j 被 mask）
O[i][d] = Σ_j P[i][j] · V[j][d]
LSE[i]  = m_i + ln(l_i) = ln Σ_{j 可见} exp(S[i][j])
```

- **scale** 来自 `p.scale`，默认 1/√D，但 case 可以指定别的值（`scale` 和 `large_logits` case 就是这样）。
- **causal mask** 让 query `i` 只看 key `0..i`。对角线 `j = i` 总是可见，所以每行至少有一个可见 key，`l_i ≥ 1`，不会除以 0。
- **softmax 沿 key 维 j 归一化**，每行 P 之和为 1。被 mask 的位置要写成 0，因为第 3 步的 GEMM 会把整行乘上 V。
- **减去最大值**不改变 P（分子分母同乘 `exp(−m_i)`），但保证指数的参数 ≤ 0。FP32 的 `exp(x)` 在 x > 88.7 时溢出成 inf，`large_logits` case 中九成以上的行最大分数超过这个值。
- **LSE 用自然对数**（`logf`，不是 `log2f`）。反向传播和 FlashAttention 的分块合并都要用它，所以公共接口要求输出它，形状 `[B, H, N]`。

### 3.2 手算小例

取 N = 3，D = 2，Q 全为 0，V 的三行是 `(1,0)`、`(0,2)`、`(3,4)`。S 全为 0，所以每个可见 key 的权重相同：

| query i | 非 causal：P 的行 | O | LSE | causal：P 的行 | O | LSE |
|---|---|---|---|---|---|---|
| 0 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (1, 0, 0) | (1, 0) | 0 |
| 1 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (½, ½, 0) | (½, 1) | ln 2 |
| 2 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 |

`uniform_causal` case 就是这个例子放大到 N = 100：第 i 行的 O 是前 i+1 个 V 的平均，LSE = ln(i+1)。

### 3.3 布局：在 `[B, N, H, D]` 里找到一个 head

公共接口的 Q、K、V、O 都是 contiguous `[B, N, H, D]`，最后一维最快：

```
Q(b, i, h, d) 的地址 = q + ((b·N + i)·H + h)·D + d
                     = q + b·(N·H·D) + h·D + i·(H·D) + d·1
                           └─ batch ─┘ └head┘ └─ 行 ─┘  └列┘
```

固定 (b, h) 后，一个 head 是一个 N×D 矩阵：相邻两行（序列位置 i 和 i+1）之间隔了 **H·D** 个 float，因为中间夹着其他 head 的同一位置。常见错误是把它当成 `[B, H, N, D]`，把行跨度写成 D；当 H = 1 时两者相同，所以这个错误只在 H > 1 的 case（`bnhd`）上暴露。

骨架提供的 GEMM 用 4 个 stride 描述一个操作数：

```cpp
struct Strides { int64_t b, h, row, col; };
// 第 (b, h) 个矩阵的元素 (r, c) 在  x[b*s.b + h*s.h + r*s.row + c*s.col]
// gemm(a, sa, b, sb, c, sc, M, N, K, batch, heads, alpha, stream) 计算
//   C(m, n) = alpha · Σ_k A(m, k) · B(n, k)        对每个 (b, h)
```

S 和 P 是本关自己选的中间布局 `[B, H, N, N]`：每个 head 一个稠密的 N×N 矩阵。注意 S 的第 `row = (b·H + h)·N + i` 行恰好也是 `lse[b, h, i]` 的下标。

QKᵀ 正好是 GEMM 的 “A·Bᵀ” 形式：A = Q（m = i，k = d），B = K（n = j，k = d），两者都是 K-major（k 方向连续）。PV 不一样：`O(i, d) = Σ_j P(i, j) V(j, d)`，GEMM 要的 B(n, k) 是 V(k, n)，也就是按 (d, j) 读 V。在内存里 V 沿 d 连续（MN-major）。**不需要转置 V**，只要换一组 stride，GEMM 的 loader 会按 stride 为 1 的方向合并访存。

### 3.4 HBM 流量：逐个 kernel 计数

记 BH = B·H，每个元素 4 字节（FP32）。假设每个 kernel 读的输入都来自 HBM、写的输出都进入 HBM，复用全部发生在片上：

| kernel | 读 | 写 | 数量级 |
|---|---|---|---|
| ① S = scale·QKᵀ | Q、K：2·BH·N·D·4 | S：BH·N²·4 | Θ(N²) 写 |
| ② softmax | S：BH·N²·4 | P：BH·N²·4，LSE：BH·N·4 | Θ(N²) 读写 |
| ③ O = PV | P：BH·N²·4，V：BH·N·D·4 | O：BH·N·D·4 | Θ(N²) 读 |
| 合计 | | | 4·BH·N²·4 + 4·BH·N·D·4 |

S/P 的流量与 Q/K/V/O 的流量之比是 **N/D**。`perf` case（BH = 16，N = 4096，D = 64）：

```
S 或 P 一次：16 × 4096² × 4 B = 1.07 GB          4 次 → 4.29 GB
Q/K/V/O 一次：16 × 4096 × 64 × 4 B = 16.8 MB      4 次 → 0.067 GB
```

S/P 的流量是 Q/K/V/O 的 64 倍。一个融合的 kernel（例如 12 关的 FlashAttention）原则上只需要后面那 0.067 GB。

### 3.5 算力还是带宽：算术强度

两个 GEMM 一共 `4·BH·N²·D` 次浮点运算（乘加算 2 次），`perf` case 是 68.7 GFLOP。除以 S/P 的流量：

```
算术强度 = 4·N²·D / (4·N²·4 B) = D/4 FLOP/B = 16 FLOP/B   （D = 64，FP32 的 S/P）
```

拿它和 B200 的 “屋脊点”（峰值算力 / 峰值带宽）比较：

| 计算单元 | 标称峰值 | ÷ 8 TB/s | 16 FLOP/B 落在 |
|---|---|---|---|
| FP32 CUDA core | 75 TFLOP/s | ≈ 9 FLOP/B | 屋脊点右侧：**算力受限** |
| BF16 Tensor Core | 2250 TFLOP/s | ≈ 280 FLOP/B | 远在左侧：**带宽受限** |

所以要分开说：在本关的 FP32 SIMT 实现里，GEMM 的 FMA 本身就很慢，S/P 的 HBM 流量只是时间的一部分；一旦 GEMM 换成 Tensor Core（第 2–4 部分），同样的 4 GB 流量就成了主导。FlashAttention 论文在 A100 上用 FP16 Tensor Core，正是后一种情况。第 7 节用实测验证这两点。

## 4. 硬件与指令

本关没有新指令，新东西是 “怎样观察内存层级”。

### 4.1 B200 内存层级

| 层级 | 容量 | 本关的角色 |
|---|---|---|
| 寄存器 / shared memory | 每 SM 64K 个寄存器、228 KB smem | GEMM 的 tile 和累加器；softmax 的块内归约 |
| L2 | 126 MB（全芯片） | N = 1024 时 S（64 MiB）能整个放进去 |
| HBM3e | 约 179 GiB，标称 8 TB/s | N = 4096 时 S（1 GiB）只能放这里 |

### 4.2 DRAM 计数器

`./lab profile` 会为每个 kernel 打印 Nsight Compute 的两个指标：

- `dram__bytes_read.sum`：这个 kernel 运行期间从 HBM 读进 L2 的字节数；
- `dram__bytes_write.sum`：从 L2 写回 HBM 的字节数。

它们数的是**真正经过 HBM 接口**的字节，不是 kernel 发出的 load/store 字节。两者在两种情况下会不同：

1. L2 命中：多个 block 读同一个 K tile，只有第一次从 HBM 来；
2. L2 是 write-back：kernel 结束时还留在 L2 里的脏数据**不算在这个 kernel 上**，它会在之后被换出时写回（也可能被下一个 kernel 直接读到，根本不进 HBM）。

所以当中间结果小于 L2 时，计数会明显小于第 3.4 节的估算。第 7 节的 `n1k` case 会看到这一点。

`./lab profile` 运行时 Nsight Compute 会锁定时钟，`gpu__time_duration` 比 `bench` 慢；只用它比较 kernel 之间的比例和字节数。

### 4.3 64 位下标

`perf` case 的 S 有 16 × 4096² ≈ 2.7 亿个元素，离 2³¹ 只差 8 倍；B·H 或 N 再大一点 `int` 就溢出。`run()` 里把 n、h、d 转成 `int64_t` 后再相乘，`Strides` 的字段也都是 `int64_t`。

## 5. 调度设计

三个 kernel 在同一个 stream 上顺序执行，前一个的输出通过 HBM（或 L2）交给下一个：

```
  HBM:  Q,K ──► [① GEMM] ──► S ──► [② softmax] ──► P (覆盖 S) ──► [③ GEMM] ──► O
                                         │                          ▲
                                         └──► LSE            V ─────┘
```

| kernel | grid（`perf` case） | block | 每个 block 做什么 | smem |
|---|---|---|---|---|
| ① `gemm_kernel` | (⌈N/64⌉, ⌈N/64⌉, B·H) = (64, 64, 16) | 256 | C 的一个 64×64 tile，每线程 4×4 | 2 × 16 × 65 × 4 B ≈ 8.1 KB |
| ② `softmax_rows` | B·H·N = 65536 | 128 | S 的一行：online (m, l)，再写回 P | 几十字节（warp 部分结果） |
| ③ `gemm_kernel` | (⌈D/64⌉, ⌈N/64⌉, B·H) = (1, 64, 16) | 256 | O 的一个 64×64 tile，K 循环长 N | 同 ① |

几点设计说明：

- **P 原地覆盖 S。** softmax 的每个 block 只读写自己那一行，读完整行（第 1 遍）之后才写（第 2 遍），中间有 `block_merge` 里的 `__syncthreads()`，所以原地写是安全的。这样只需要一块 N×N 的 scratch。
- **scratch 在 `make_plan` 里分配**（`lab::Scratch`，不计时，初值是 NaN）。`run()` 只发射 kernel。
- **steps 1–2 把 S/P 写进 `p.probe`。** probe 也是 `[B, H, N, N]`，harness 拿它和 FP64 oracle 比较，这样 O 还没算之前就能检查 S 和 P。
- **causal 并没有省掉任何工作。** 两个 GEMM 仍然算完整的 N×N；softmax 把上三角写成 0，PV 还要把这些 0 乘一遍。库实现只算下三角附近的块，这是第 7 节 `perf_causal` 差距更大的原因，也是 43_fa4_schedule 关的主题。

## 6. 分步练习

所有命令在仓库根目录运行。骨架里 GEMM、归约 helper、launch 配置和 `make_plan` 都已写好；你要写的是 3 个 TODO。

### 步骤 1 · TODO(1)：S = scale·QKᵀ

在 `run()` 中写出 Q/K 的 `Strides`（一个 head 是 N×D 矩阵）和 S 的 `Strides`（`[B, H, N, N]`），然后调用一次 `gemm()`。scale 作为 `alpha` 传进去。

```bash
./lab check 11 --step 1
```

这一步检查的是 probe 里的 S，坐标显示为 `[b, h, i, j]`。S 不带 mask：causal case 的上三角也要有正确的分数。

### 步骤 2 · TODO(2)：带 mask 的行 softmax 与 LSE

在 `softmax_rows` 中：由 `blockIdx.x` 算出 query 位置 `i` 和可见长度 `len`，第 1 遍把可见的分数折叠成 `State {m, l}` 并用 `block_merge()` 合并，第 2 遍原地写 P（被 mask 的位置写 0），最后一个线程写 `lse[row]`。

```bash
./lab check 11 --step 2
```

这一步检查 probe 里的 P。LSE 要到步骤 3 才检查，但请现在就写对。

### 步骤 3 · TODO(3)：O = P·V

写出 P、V（按 (d, j) 读）和 O 的 `Strides`，调用 `gemm()`，`alpha = 1`。

```bash
./lab check 11
./lab bench 11
./lab bench 11 --case perf_causal
./lab profile 11
```

`./lab bench 11 --step 1` 和 `--step 2` 只计时到该步为止已完成的 kernel，而 TFLOP/s 仍按完整 attention 的计算量算、`vs` 仍是与完整 attention 库基线的耗时之比。所以这两步 bench 输出的 TFLOP/s 和 `vs` 都对应一个没做完的 kernel，没有意义，只看耗时本身。`./lab profile 11 --step 1` 或 `--step 2` 会列出每个 kernel 的时间和字节数。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 只有 `bnhd` 及之后的 case 失败 | 把 `[B,N,H,D]` 当成了 `[B,H,N,D]`：行跨度应是 H·D，head 偏移是 D |
| 步骤 1 `scale` case 失败、其他都过 | scale 写死成 1/√D 或 1，没有用 `p.scale` |
| 步骤 2 causal case 的 probe 在上三角不为 0 | mask 位置没写；P 原地覆盖 S，不写就留下了 S 的值 |
| 步骤 2 `large_logits` 出现 NaN/inf | 没有减最大值：`exp` 的参数超过 88.7 就溢出 |
| 步骤 2 只有 causal case 失败 | 可见长度差一：query i 看 `0..i`，共 i+1 个 key |
| 步骤 3 O 对、LSE 偏大 | 用了 `log2f`：log₂ x = 1.443·ln x |
| 步骤 3 LSE 对、O 错 | V 的 `Strides`：GEMM 的 n 对应 d，k 对应 j，分别想清楚它们在 `[B,N,H,D]` 里的跨度；再核对 M、N、K |
| `S would need more than 32 GiB` | 这正是本关要说明的问题：S 放不下 |
| `illegal memory access` | stride 乘积溢出（用 `int64_t`），或者 B 和 H 的 batch stride 写反；用 `./lab sanitize 11 --case bnhd` 定位 |

## 7. 性能

### 7.1 时间与库基线

本机实测（CUDA Graph 计时；`vs efficient` 是 PyTorch memory-efficient attention 的耗时除以本实现的耗时）：

| case | 形状 [B, N, H, D] | vs efficient |
|---|---|---|
| `perf` | [1, 4096, 16, 64] | ≈ 0.32× |
| `perf_causal` | [1, 4096, 16, 64]，causal | ≈ 0.21× |

efficient 是一个融合 kernel（不写 S/P），而且源码显示它的 FP32 版本用 Tensor Core 做 3xTF32 模拟（`OpMultiplyAddFastF32`，见第 9 节的源码位置）。所以这里的差距同时来自 “少了 4 GB 流量” 和 “GEMM 用了 Tensor Core” 两件事。本关的目标只是理解代价，答案达到 0.15× 以上即可。

### 7.2 时间都花在哪

`./lab profile 11`（`perf` case）每个 kernel 的时间占比和 DRAM 字节数：

| kernel | 时间占比 | DRAM 读 | DRAM 写 | 第 3.4 节估算（读 / 写） |
|---|---|---|---|---|
| ① QKᵀ | ≈ 41% | 0.034 GB | 1.02 GB | 0.034 / 1.07 GB |
| ② softmax | ≈ 9% | 1.07 GB | 1.01 GB | 1.07 / 1.07 GB |
| ③ PV | ≈ 50% | 1.10 GB | 0.019 GB | 1.09 / 0.017 GB |
| 合计 | | ≈ 4.25 GB | | 4.36 GB |

- 实测字节和估算对得上：几乎全部流量都是 S/P。写入略少于 1.07 GB，是因为 kernel 结束时还有几十 MB 留在 L2 里（第 4.2 节）。
- softmax 只做读写，是纯粹带宽受限的 kernel，`dram__throughput` 是三者中最高的。
- 两个 GEMM 占了 90% 的时间，`sm__throughput` 高而 `dram__throughput` 低：它们是**算力受限**的，与第 3.5 节 “FP32 SIMT 在屋脊点右侧” 一致。PV 的计算量和 QKᵀ 相同却更慢。一个可能的原因是它的 grid 只有 1024 个 block（D = 64 只够一个 64 列的 tile），`sm__warps_active` 约 54%，而 QKᵀ 约 74%。

作为对照，用 Nsight Compute 测 efficient kernel（同一形状）的 DRAM 读大约 0.05 GB，基本就是 Q、K、V 各读一次。

### 7.3 时间随 N 的变化

同样 B=1、H=16、D=64，只改 N（`./lab bench 11 --case n1k`、`--case n2k`、`--case perf`）：

| N | S 的大小 | 相对上一行的时间 | S 能否放进 L2 |
|---|---|---|---|
| 1024（`n1k`） | 64 MiB | — | 能 |
| 2048（`n2k`） | 256 MiB | ≈ 3.1× | 不能 |
| 4096（`perf`） | 1 GiB | ≈ 3.8× | 不能 |

N 翻倍，时间接近 4 倍：计算量和 S/P 流量都是 Θ(N²)。从 1024 到 2048 不足 4 倍，一个可能的原因是 N = 1024 时 grid 较小、GPU 没有被填满（未验证）。

在 `n1k` 上运行 `./lab profile 11 --case n1k`：①号 kernel 只把 64 MiB 中的约 15 MB 写进 HBM，其余留在 L2 里直接被 softmax 读到。中间结果比 L2 小时，“写 S 再读 S” 的代价会被 L2 掩盖一部分；N 一旦变大，它就全部落到 HBM 上。

### 7.4 应该得出什么结论

1. 标准 attention 的 HBM 流量是 Θ(BH·N²)，比 Q/K/V/O 大 N/D 倍，实测与估算一致。
2. 在 FP32 SIMT 下，这些流量只占总时间的一小部分：4.25 GB 按 8 TB/s 标称带宽至少要花总时间的约 11%，GEMM 的 FMA 才是大头；在 Tensor Core 上它会成为主导（第 3.5 节）。
3. 更硬的限制是显存：S 的大小随 N² 增长。

FlashAttention 同时解决 1 和 3：S、P 只在片上存在，HBM 流量回到 Θ(N·D) 量级（加上分块带来的重复读 K/V）。

## 8. 自测问题

1. Q 的布局改成 `[B, H, N, D]` 时，一个 head 的 `Strides{b, h, row, col}` 应该是什么？
2. B = 2，N = 1000，H = 4，D = 64，FP32。标准实现的 S/P 流量和 Q/K/V/O 流量各是多少字节？
3. causal 时，`softmax_rows` 为什么必须把 `j > i` 的位置写成 0，而不是跳过它们？
4. 如果 softmax 第 1 遍只算 max、第 2 遍算 sum、第 3 遍写 P（三遍），HBM 读流量会变成三倍吗？用 `./lab profile` 怎么验证？
5. 把本关的 GEMM 换成 BF16 Tensor Core（2250 TFLOP/s），S/P 存成 BF16，`perf` case 的计算时间和 S/P 流量时间下界各大约是多少？
6. 为什么 `perf` case 的 DRAM 写字节数略小于 S 的大小，而 `n1k` 小得多？

<details>
<summary>参考答案</summary>

1. 元素 (b, h, i, d) 在 `((b·H + h)·N + i)·D + d`，所以 `{H·N·D, N·D, D, 1}`。
2. S/P：4 × 2·4·1000²·4 B = 128 MB；Q/K/V/O：4 × 2·1000·4·64·4 B ≈ 8.2 MB；比值 N/D ≈ 15.6。
3. 第 3 步的 GEMM 对整行 j = 0..N−1 求和。P 和 S 共用一块缓冲，不写的话上三角留着的是 S 的值（或 scratch 的 NaN），会被乘进 O。
4. 一般不会。一行最多 16 KB，同一个 block 紧接着再读一遍，多半命中 L1/L2，HBM 只读一次。用 `./lab profile` 看 `softmax_rows` 的 `dram__bytes_read.sum` 是否仍约等于 S 的大小。
5. 计算：68.7 GFLOP / 2250 TFLOP/s ≈ 0.03 ms。流量：4 × 16·4096²·2 B ≈ 2.1 GB，/ 8 TB/s ≈ 0.27 ms。流量时间约是计算的 9 倍，完全带宽受限。
6. L2 是 write-back：kernel 结束时还在 L2 里的脏行不计入这个 kernel。`perf` 中 S（1 GiB）远大于 L2，只有最后几十 MB 留下；`n1k` 的 S 只有 64 MiB，大部分都留在 L2 里，直接被下一个 kernel 读走。

</details>

## 9. 延伸阅读

- FlashAttention 论文（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135v2)）：第 2.2 节与附录 Algorithm 0（标准 attention），第 3.2 节（IO 复杂度分析，定理 2）。
- Rabe & Staats，*Self-attention Does Not Need O(n²) Memory*（[arXiv:2112.05682](https://arxiv.org/abs/2112.05682)）：显存为什么可以不随 N² 增长。
- Williams, Waterman & Patterson，*Roofline: An Insightful Visual Performance Model*（CACM 2009）：第 3.5 节屋脊点的来源。
- Nsight Compute：[Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-workload-analysis)，DRAM 与 L2 指标的含义。
- PyTorch memory-efficient attention（xFormers / CUTLASS FMHA）：`third_party/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h`；FP32 选择 Tensor Core 3xTF32 的位置在同目录 `gemm_kernel_utils.h` 的 `DefaultGemmType<ArchTag, float>`。
