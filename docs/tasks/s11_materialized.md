# s11_materialized · 显式中间矩阵 attention

阶段 1。前置：[s01_softmax](s01_softmax.md)、[s10_gemm](s10_gemm.md)

先读：[从三个算子到流式 attention](../attention-math.md)，包含可手算的例子、状态合并推导及精度边界。

## 把三个已掌握的操作连接起来

attention 先比较 query 和 key，再把归一化后的权重用于 value 的加权和。多头只改变独立问题的数量；序列位置仍是归一化维度。接口采用库常用的 BNHD，不能直接把它当作连续的 BHND 矩阵。先通过多 batch、多 head 的小案例确认寻址，再看性能。

本关允许显式保存分数和概率。它提供容易观察的中间结果，也暴露 N² 存储成本。causal 的条件作用于 query/key 的位置，不作用于 head；self-attention 的对角线始终有效。LSE 帮助定位归一化问题，并且其定义采用自然对数。

## 定义、代码与参考

S[b,h,i,j]=scale·dot(Q[b,i,h,:],K[b,j,h,:])；P=softmax(S+mask)；O=PV；LSE=logsumexp(S+mask)。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s11_materialized/kernel.cu](../../students/cuda/s11_materialized/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s11_materialized/](../../solutions/s11_materialized/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS 矩阵乘 + native softmax 的 FP32 显式 attention；包含其实际中间矩阵成本。

源码阅读：官方 FA2 README 的 flash_attn_func 输入约定；本地 solutions/s11_materialized/kernel.cu；快速参考 materialized_attention。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 17, 1, 64)` | fp32/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 33, 3, 64)` | fp32/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 31, 2, 64)` | fp32/full | Zero Q/K and nonzero V: verify mean of V, not vacuous zero output. |
| `large_logits` | stress | `(1, 65, 2, 128)` | fp32/full | 较大 QK，检查数值稳定性和输出舍入。 |
| `scale` | debug | `(1, 17, 2, 64)` | fp32/full | 显式 scale，防止硬编码 1/sqrt(D)。 |
| `single` | edge | `(1, 1, 2, 64)` | fp32/full | 单 token 无归约长度，输出应等于 V。 |
| `tail_causal` | edge | `(2, 129, 3, 64)` | fp32/causal | query/key 尾部、含对角线的 causal 及多头索引。 |
| `tail_full` | edge | `(1, 257, 2, 128)` | fp32/full | 非 causal 尾部，检查掩码不会错误地依赖 causal。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | fp32/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | fp32/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | fp32/full | 长 key 循环，观察融合和流水线的稳态作用。 |
| `causal_uniform` | edge | `(1, 31, 2, 64)` | fp32/causal | Uniform causal weights require the exact prefix mean of nonzero V. |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 cuBLAS/softmax 组合基线同输入比较，记录峰值 workspace 随 N 的变化。

画出 QK、P、PV 的数据流，列出临时内存与每一步的 dtype。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s11_materialized -v
./lab reference s11_materialized --case 0
./lab solution s11_materialized --case 0
./lab check s11_materialized --case 0
./lab check s11_materialized
./lab bench s11_materialized --case 0
./lab sanitize s11_materialized --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
