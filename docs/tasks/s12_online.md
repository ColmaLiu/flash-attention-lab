# s12_online · 流式 attention 与 IO

阶段 1。前置：[s11_materialized](s11_materialized.md)

先读：[从三个算子到流式 attention](../attention-math.md)，包含可手算的例子、状态合并推导及精度边界。

## 改变存储方式，保持同一数学结果

N² 的分数/概率矩阵不是最终输出，为什么一定要写回显存？本关以已处理 key 集合为单位描述统计量：最大值、相对该最大值的指数和、未归一化的 value 加权和。集合扩大时，这些量仍须表示同一个数学对象。先用任意两个不相交集合证明可合并性，再设计设备上的数据流。

减少临时存储并不自动减少每一种访存。Q 与 K/V 的外层遍历次序、输出状态驻留位置和 CTA 数量会改变重读与并行度。FA1 的贡献应从 IO 模型解释；你可以先实现一个清晰的流式版本，再将源码与原论文的循环组织逐项比较，不能只凭在线 softmax 的出现就宣称复现 FA1。

## SIMT 分工与 shared-memory 布局

参考 CTA 使用 `dim3(16,16)`：16 个 half-warp 各拥有一个 query 行，共享 32 个 key 的 tile。每个 lane 计算两个 key 的完整 QK 点积，以 16 线程 shuffle 求行最大值与分母；随后各 lane 持有 O 的 `lane+16*part` 列。最大值、分母和未归一化 O 跨 key tile 驻留在 FP32 寄存器，不将 N² 的 S/P 写回显存。

布局应从消费者的访问方式推导。QK 内层固定特征列、lane 沿 key 变化，所以 K 以 `ks[D][33]` 存储，读取 `ks[col][key]` 连续；搬入时线程仍合并读取全局的连续特征列，再将其写到转置位置。shared 的 33 列步长让这些转置写入分散到不同 bank，避免简单转置引入另一处冲突。Q 使用 `qs[16][D+16]`：同一 warp 的两个 half-warp 分别读取相邻 query 的同一特征，16 个 float 的 padding 将两次广播放到不同 bank。V 保持 `vs[32][D]`，与 PV 阶段沿输出特征分布的 lane 对应。

K/V 搬运后由 CTA barrier 发布；score 归一化后把当前 tile 的概率写入 `weights[16][32]`，再经 barrier 供 PV 使用。所有 PV 消费者完成后才能覆盖下一块 K/V/weights。QK 使用带残差补偿的 FP32 点积，PV 使用 FP32 FMA；shared 布局优化不改变这些累加步骤。检验性能时区分 bank 行为、补偿算术、指数/归约与同步成本，不能把不再保存 N² 张量等同于整个 attention 已达到带宽或计算峰值。

## 定义、代码与参考

对已处理集合 J，m_J=max_{j∈J} S_j，l_J=sum_{j∈J} exp(S_j-m_J)，u_J=sum_{j∈J} exp(S_j-m_J)V_j；最终 O=u_J/l_J。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s12_online/kernel.cu](../../students/cuda/s12_online/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s12_online/](../../solutions/s12_online/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS 矩阵乘 + native softmax 的 FP32 显式 attention；包含其实际中间矩阵成本。

源码阅读：FA1 论文 §3.1–3.2；本地 solutions/s12_online/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 17, 1, 64)` | fp32/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 33, 3, 64)` | fp32/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 31, 2, 64)` | fp32/full | Q/K 为零而 V 非零，检查 V 的均值与完整输出写入。 |
| `large_logits` | stress | `(1, 65, 2, 128)` | fp32/full | 较大 QK，检查数值稳定性和输出舍入。 |
| `scale` | debug | `(1, 17, 2, 64)` | fp32/full | 显式 scale，防止硬编码 1/sqrt(D)。 |
| `single` | edge | `(1, 1, 2, 64)` | fp32/full | 单 token 无归约长度，输出应等于 V。 |
| `tail_causal` | edge | `(2, 129, 3, 64)` | fp32/causal | query/key 尾部、含对角线的 causal 及多头索引。 |
| `tail_full` | edge | `(1, 257, 2, 128)` | fp32/full | 非 causal 尾部，检查掩码不会错误地依赖 causal。 |
| `rising_max` | stress | `(1, 512, 2, 128)` | fp32/full | 后续 key 的最大值上升，检验历史状态 rescale。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | fp32/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | fp32/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | fp32/full | 长 key 循环，观察融合和流水线的稳态作用。 |
| `causal_uniform` | edge | `(1, 31, 2, 64)` | fp32/causal | 均匀 causal 权重要求输出为非零 V 的精确前缀均值。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s11 保持 FP32 比较，区分显存占用下降与实际 DRAM 流量/速度变化。

证明已处理 key 集合的统计含义，分析遍历次序与 O 状态存储；不能只以数值通过命名 FA1。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

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
