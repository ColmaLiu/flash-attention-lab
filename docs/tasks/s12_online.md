# s12_online · 流式 attention 与 IO

阶段 1。前置：[s11_materialized](s11_materialized.md)

先读：[从三个算子到流式 attention](../attention-math.md)，包含可手算的例子、状态合并推导及精度边界。

## 改变存储方式，保持同一数学结果

N² 的分数/概率矩阵不是最终输出，为什么一定要写回显存？本关以已处理 key 集合为单位描述统计量：最大值、相对该最大值的指数和、未归一化的 value 加权和。集合扩大时，这些量仍须表示同一个数学对象。先用任意两个不相交集合证明可合并性，再设计设备上的数据流。

减少临时存储并不自动减少每一种访存。Q 与 K/V 的外层遍历次序、输出状态驻留位置和 CTA 数量会改变重读与并行度。FA1 的贡献应从 IO 模型解释；你可以先实现一个清晰的流式版本，再将源码与原论文的循环组织逐项比较，不能只凭在线 softmax 的出现就宣称复现 FA1。

## 定义、代码与参考

对已处理集合 J，m_J=max_{j∈J} S_j，l_J=sum_{j∈J} exp(S_j-m_J)，u_J=sum_{j∈J} exp(S_j-m_J)V_j；最终 O=u_J/l_J。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s12_online/kernel.cu](../../students/cuda/s12_online/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s12_online/](../../solutions/s12_online/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS 矩阵乘 + native softmax 的 FP32 显式 attention；包含其实际中间矩阵成本。

源码阅读：FA1 论文 §3.1–3.2；本地 solutions/s12_online/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

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
| `rising_max` | stress | `(1, 512, 2, 128)` | fp32/full | 后续 key 的最大值上升，检验历史状态 rescale。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | fp32/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | fp32/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | fp32/full | 长 key 循环，观察融合和流水线的稳态作用。 |
| `causal_uniform` | edge | `(1, 31, 2, 64)` | fp32/causal | Uniform causal weights require the exact prefix mean of nonzero V. |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s11 保持 FP32 比较，区分显存占用下降与实际 DRAM 流量/速度变化。

证明已处理 key 集合的统计含义，分析遍历次序与 O 状态存储；不能只以数值通过命名 FA1。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s12_online -v
./lab reference s12_online --case 0
./lab solution s12_online --case 0
./lab check s12_online --case 0
./lab check s12_online
./lab bench s12_online --case 0
./lab sanitize s12_online --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
