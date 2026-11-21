# attention 的数学：从三个算子到流式计算

本文把散在 [01](../course/01_softmax/README.md)、[11](../course/11_naive_attention/README.md)、[12](../course/12_fa1/README.md)、[22](../course/22_fa2/README.md) 中的数学放在一起。各关讲义有更完整的推导和手算小例；这里讲它们之间的联系。

## 1. 把一个 head 看成矩阵

固定 batch b 和 head h，Q、K、V 各是 N×D 矩阵。第 i 个 query 对第 j 个 key 的分数是：

```
S[i,j] = scale · Σ_d Q[i,d] · K[j,d]
```

softmax **沿 key 方向 j** 归一化：

```
P[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])
O[i,d] = Σ_j P[i,j] · V[j,d]
```

不同 head 互相独立。`[B,N,H,D]` 只是内存中的排列顺序，不改变归一化的维度（地址计算见 [11 §3.3](../course/11_naive_attention/README.md)）。

**一个检查用的小例子**：N = 3，Q 全为 0，V 的三行是 `(1,0)`、`(0,2)`、`(3,4)`。

| | 每行的 P | O | LSE |
|---|---|---|---|
| 非 causal | 都是 `(1/3, 1/3, 1/3)` | 都是 `(4/3, 2)` | 都是 `ln 3` |
| causal | `(1,0,0)`、`(1/2,1/2,0)`、`(1/3,1/3,1/3)` | `(1,0)`、`(1/2,1)`、`(4/3,2)` | `0`、`ln 2`、`ln 3` |

这个例子同时检查 mask、归一化方向和 V 的寻址。如果把 V 也设成 0，很多错误会被藏起来。

## 2. 为什么要减最大值

直接算 `exp(S)`，一个很大的分数就会溢出成 `inf`。令 `m = max_j S[j]`，分子分母同乘 `exp(−m)`：

```
P[j] = exp(S[j] − m) / ℓ，   ℓ = Σ_k exp(S[k] − m)
```

至少一个分数等于 m，它的指数是 1；其他指数在 `[0, 1]` 内。所以分母 ℓ ≥ 1，不会因为全部下溢而变成 0。

输出的 **LSE**（log-sum-exp）是归一化常数的自然对数：

```
LSE = m + ln(ℓ)
```

详见 [01 §3.1](../course/01_softmax/README.md)。

## 3. 标准实现为什么需要 N² 的空间和流量

标准实现（[11](../course/11_naive_attention/README.md)）分三步，把 S 和 P 写进 HBM：

```
S = scale · Q Kᵀ   →  写 N×N
P = softmax(S)     →  读 N×N，写 N×N
O = P V            →  读 N×N
```

每个 head 有 N² 个分数。N 翻倍，S/P 的空间和流量变成 4 倍；而 Q、K、V、O 只有 N·D 个元素。

N 远大于 D 时，S/P 的读写占绝大部分 HBM 流量。它是不是时间瓶颈，取决于计算有多快：

| 计算单元 | 屋脊点 | S/P 的算术强度（D = 64，FP32 S/P） | 结论 |
|---|---|---|---|
| FP32 CUDA core | 约 9 FLOP/B | 16 FLOP/B | 算力受限，流量只占一部分时间 |
| BF16 Tensor Core | 约 280 FLOP/B | 16 FLOP/B | 带宽受限，流量决定时间 |

所以 Tensor Core 越快，消除 S/P 流量的价值越大（[11 §3.5](../course/11_naive_attention/README.md)、[B200 架构概览 §3.1](b200.md#31-屋脊点算力还是带宽)）。

## 4. 三个状态足以合并任意两个 key 集合

输出只需要 N×D。能不能一次只处理一部分 key，处理完就丢掉对应的分数？难点是：后面的 key 可能出现更大的最大值，改变前面指数的尺度。

对某个 query，已处理的 key 集合 A 定义三个状态：

```
m_A = max_{j∈A} S[j]
ℓ_A = Σ_{j∈A} exp(S[j] − m_A)
u_A = Σ_{j∈A} exp(S[j] − m_A) · V[j]      （长度 D 的向量）
```

对另一个不相交的集合 B 同样定义。合并时取 `m = max(m_A, m_B)`，两个换算系数 `a = exp(m_A − m)`、`b = exp(m_B − m)`：

```
ℓ = a·ℓ_A + b·ℓ_B
u = a·u_A + b·u_B
```

**证明**：A 中每一项 `exp(S[j] − m_A)` 乘以 a，正好得到 `exp(S[j] − m)`；B 同理。所以合并后的状态仍然符合同一个定义，只是集合变成了 A ∪ B。

全部 key 处理完后，`O = u / ℓ`，`LSE = m + ln ℓ`。

这不是近似：数学上与标准 attention 完全相等。浮点运算顺序不同会带来不同的舍入误差，但不改变定义。

| 用法 | 哪一关 |
|---|---|
| 一行 softmax，一次处理一个元素：online softmax | [01 §3.3](../course/01_softmax/README.md) |
| 一个 K/V 块一次：FA1 Algorithm 1，状态存在 HBM | [12 §3.2–3.3](../course/12_fa1/README.md) |
| 状态留在寄存器，最后才除以 ℓ（延迟归一化） | [22 §3.1](../course/22_fa2/README.md) |
| O 在 TMEM，由 softmax 或 correction warp 乘以 a | [34 §3.2](../course/34_fa3/README.md)、40 |
| 只在 m 增长超过阈值时才重新缩放 | [41](../course/41_fa4_softmax/README.md) |

**两个边界情况**：

- 初始空集合：`m = −∞`、`ℓ = 0`。不能机械地算 `exp(−∞ − (−∞))`，那是 NaN。
- causal 中，某个 K/V 块可能对某些 query 完全被 mask。这时这一块对它们没有贡献；但整行没有合法 key 不会发生，因为对角线总是合法的（[12 §3.7](../course/12_fa1/README.md)）。

## 5. 数学不规定 GPU 上怎么分工

合并定理没有规定一行交给一个线程还是一个 warp，也没有规定 Q、K、V 放在寄存器还是 shared memory。不同的 FA 版本做了不同的选择：

| 版本 | 外层循环 | 状态放在哪 | 主要代价 |
|---|---|---|---|
| FA1（[12](../course/12_fa1/README.md)） | K/V 块 | O、ℓ、m 每块读写一次 HBM | 每个 K/V 块都要往返 O |
| FA2（[22](../course/22_fa2/README.md)） | Q 块 | O、ℓ、m 在寄存器，只写一次 | 每个 Q 块重读全部 K/V（多数命中 L2） |
| FA3 思想（[34](../course/34_fa3/README.md)） | Q 块 | S、O 在 TMEM | softmax 与 MMA 需要重叠 |

如果多个 warp 处理同一个 query 的不同 key 子集，最后还要用第 4 节的规则合并它们的状态。FA2 让每个 warp 处理不同的 query 行，避免了这种合并。

## 6. IO 复杂度与测量

FA1 论文给出 HBM 访问量 `Θ(N²d²/M)`，M 是片上存储大小（[12 §3.6](../course/12_fa1/README.md)）。这说明消除 S/P 后，访存量随片上存储增大而减少。但它是一个模型，实际流量还受下面这些影响：

- 分块大小与遍历顺序；
- L2 是否命中（例如 FA2 中同一 head 的 K/V 被多个 Q 块重读，大多来自 L2）；
- 事务粒度（32 字节 sector）、spill、重放。

所以分析时把三样东西分开写：

| 内容 | 怎么得到 |
|---|---|
| 算法模型：每种张量读写几次 | 按源码手算 |
| 实际 DRAM 字节数 | `./lab profile` 的 `dram__bytes_read.sum`、`dram__bytes_write.sum`（[11 §4.2](../course/11_naive_attention/README.md)） |
| 时间 | `./lab bench` |

消除了 N² 的存储，不保证一定更快。CTA 太少、同步太多、寄存器 spill、GEMM 太慢都会拖慢它。12 关在 FP32 SIMT 上就看到：FA1 多出的 HBM 往返不是瓶颈，计算才是（[12 §7.2](../course/12_fa1/README.md)）。

## 7. 怎样判断结果正确

**数学 oracle**：`labkit` 用 PyTorch 的 FP64 直接按定义计算 S、softmax、PV 和 LSE。它不分块、不用 online softmax，也不依赖 cuDNN 或 FA 的结果。大输入按 query 行分组计算以控制显存。低精度输入先转成 FP64 再算，所以输入本身的量化误差不算作 kernel 的错误。

**判据**：每个元素都要满足

```
|实际 − 参考| ≤ atol + rtol · |参考|
```

NaN 和 Inf 一律算错。失败时报告最大误差、失败个数和第一个失败的坐标。

**容差**按算子、dtype 和归约长度定义在 `labkit/kinds/<kind>.py` 的 `tolerance()` 中。大致是：

| 输出 | 容差的来源 |
|---|---|
| transpose | 0：没有任何计算 |
| GEMM | FP32 点积的舍入误差约随 √K 增长；Tensor Core 对 BF16/FP16 的乘积截断，误差更大 |
| attention O | FP32 最紧；FP16、BF16 依次放宽，因为 P 要转回 16 位才能送进 Tensor Core |
| attention LSE | 与 O 分开设定 |

**其他检查**：

- 输出和 scratch 先填 NaN（或垃圾字节），漏写的元素会被发现；
- 检查输入没有被改写；
- 在默认 stream 和非默认 stream 上各跑一次；
- 部分 step 不检查最终输出，而是检查 `probe` 中的中间结果（例如 S）。

**出错时**：固定 case，看第一个失败的坐标，核对 scale、底数、mask 和寻址，再缩小 case。**不要靠调大容差来通过。**

## 8. 指数的底数

硬件只直接支持 `2^x`（`ex2.approx`，SASS 中是 `MUFU.EX2`）。因为 `e^x = 2^(x·log₂e)`，令 `scale_log2 = scale · log₂e`：

```
exp(scale·(s − m)) = 2^(s·scale_log2 − m·scale_log2)
```

其中 s 是未乘 scale 的分数，m 是它们的最大值。指数里的减法和乘法合成一条 FFMA。这要求 scale > 0：只有这时 `max(scale·s) = scale·max(s)`。课程的支持域规定 scale > 0 且有限（[22 §3.3](../course/22_fa2/README.md)）。

内部可以换底，但**输出的 LSE 必须是自然对数**：`LSE = m·scale + ln ℓ`。

`ex2.approx` 的误差界和它在什么时候够用，见 [01 §4.4](../course/01_softmax/README.md)。

## 延伸阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135v2)：§2.2 GPU 内存层级，§3.1 Algorithm 1，§3.2 定理 2（IO 复杂度）；
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691v1)：§3.1 减少非矩阵运算，§3.2 并行划分；
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608v2)：异步生产者、GEMM 与 softmax 重叠；
- [FlashAttention-4 论文](https://arxiv.org/abs/2603.05451v1)：Blackwell 上的流水、指数与 rescale。
