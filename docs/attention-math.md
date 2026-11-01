# 从三个算子到流式 attention

本章贯穿 s01、s11、s12 和 s22。先弄清每个状态表示的数学对象，再阅读完整 kernel。改变线程划分时，这些对象应保持不变。

## 先把一个 head 看成矩阵

固定 batch 和 head，Q/K/V 各为 N×D 矩阵。第 i 个 query 的分数为

`S[i,j] = scale * sum_d Q[i,d] * K[j,d]`。

softmax **沿 key 位置 j** 归一化，最后 `O[i,d] = sum_j P[i,j] * V[j,d]`。多个 head 是相互独立的这类计算。BNHD 只是全局存储顺序；不能因此把 head 当作归一化维度。

取 N=3，Q 全为零，V 的三行为 `(1,0)`、`(0,2)`、`(3,4)`。非 causal 时每个 query 的概率都是 `(1/3,1/3,1/3)`，O 都是 `(4/3,2)`，LSE 都是 `ln(3)`。causal 时三行分别为 `(1,0)`、`(1/2,1)`、`(4/3,2)`，LSE 为 `0,ln(2),ln(3)`。这个小例同时检查 mask、归一化维度和 V 寻址；如果把 V 也置零，很多错误都会被隐藏。

## 为什么要减最大值

直接计算 `exp(S)`，一个很大的正分数就可能溢出。令 `m=max_j S[j]`，分子分母同乘 `exp(-m)`：

`P[j] = exp(S[j]-m) / sum_k exp(S[k]-m)`。

至少一个有效分数等于 m，其指数为 1；其他指数在 `[0,1]`。分母非零，避免“所有指数都下溢成零”。最终自然对数归一化常数为 `LSE = m + ln(l)`，其中 `l=sum exp(S-m)`。

s01 的核心是算出 m 和 l 并让使用它们的线程得到正确值。s11 则将这个操作放在 QK 与 PV 两个矩阵乘之间。先分别检查三步，才能判断最终 O 的错误来自哪里。

## s11 为什么占 N² 空间

显式实现把 S 和 P 写到全局内存。每个 head 有 N² 个分数，FP32 的一个中间矩阵占 `4*N*N` 字节；N 翻倍，矩阵空间变成四倍。Q/K/V/O 的空间只与 N 成正比。

但输出只需要 N×D。是否能按 key 子集处理分数，在每个子集用完后立即释放？难点是后面的 key 可能出现更大的最大值，从而改变前面指数的尺度。

## 三个状态足以合并任意两个 key 集合

对某个 query，已处理集合 A 的状态定义为：

- `m_A = max(S[j], j in A)`；
- `l_A = sum(exp(S[j]-m_A), j in A)`；
- 向量 `u_A[d] = sum(exp(S[j]-m_A)*V[j,d], j in A)`。

对另一个不相交集合 B 同样定义状态。取 `m=max(m_A,m_B)`，旧尺度转换系数为 `a=exp(m_A-m)`、`b=exp(m_B-m)`，则

`l = a*l_A + b*l_B`，`u = a*u_A + b*u_B`。

证明只需要把 A 中每个 `exp(S[j]-m_A)` 乘上 a，得到 `exp(S[j]-m)`；B 同理。因此合并后的状态仍表示全集的同一个定义。最后一次才计算 `O=u/l`。这不是近似 attention；浮点运算次序可能改变舍入误差，但不改变实数数学定义。

把这个证明对应到 [s12 完整答案](../solutions/s12_online/kernel.cu)：`maximum` 是 m，`denominator` 是 l，`numerator` 是 u。每次处理 key tile，`correction` 把旧状态换到新尺度，当前 tile 的概率再累积进来。先用两块数据手算一次，再用 `rising_max` case 检查后来的最大值改变时状态有没有同步更新。

初始空集合不能机械地代入 `exp(-inf - -inf)`，它是 NaN。causal 的某个 tile 也可能对某些 query 完全无效。阅读 s22 优化答案中的全 mask 分支，区分“这一块无贡献”和“整个 query 没有合法 key”；本课程 self-attention 的对角线保证后者不会发生。

## 数学合并不规定 GPU 映射

上述证明没规定一行交给一个 warp 还是多个 warp，也没规定 Q/K/V 放在寄存器还是 shared memory。s12 先展示 SIMT 版本的数据流；s22 把两次矩阵乘换成 Tensor Core，同时保持每个 warp 对 query 行与输出累加器的 ownership。

如果多个 warp 分别处理同一 query 的不同 key 子集，最终还需要合并输出状态。若每个 warp 处理独立的 query 行，就可以避免这类跨 warp 输出合并，但可能改变 K/V 复用方式。FA2 的并行与非矩阵操作优化必须从实际源码证明，不能仅以“使用 online softmax”命名。

## 精度与底数也属于契约

低精度输入已经在送入 kernel 前量化。oracle 使用同一量化值转 FP64，因此不把输入量化误差算成 kernel 错误。QK 累加、softmax 状态和 O 累加使用 FP32；P 若为 Tensor Core 转回 BF16/FP16，会引入新的舍入点。O 最后也会转换成输入 dtype。逐个分离这些误差来源，比一次增大容差更能帮助定位。

使用 `exp2(x)` 时应把自然指数的参数乘 `log2(e)`；输出 LSE 仍必须是自然对数。s42 的近似指数和 rescale 实验必须维持这个公共接口，不可因为内部表示改变就改变输出含义。

## 用什么证据判断优化

流式算法消除了完整 S/P 的显存存储，但这不保证你的实现更快：较少 CTA、过多同步、寄存器溢出或低效矩阵乘仍会拖慢它。先比较相同 dtype/shape/mask 的 s11 与 s12，再比较 s22 的两份答案和快速库。算法所需字节、NCU 实际 DRAM 计数和 CUDA event 时间分别回答不同问题，应分开记录。

下一步阅读：[FA1 原论文](https://arxiv.org/abs/2205.14135v2)、[FA2 原论文](https://arxiv.org/abs/2307.08691v1) 与 [本地优化答案](../solutions/s22_fa2/README.md)。
