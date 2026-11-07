# s41_overlap · Matmul、softmax 与 correction overlap

阶段 4。前置：[s40_attention](s40_attention.md)

## 把关键路径上的非矩阵工作移开

Blackwell 上矩阵算力提升后，softmax、数据交换和 correction 的成本更显眼。优化的对象是依赖链：某块矩阵计算何时开始，另一块 softmax 是否已经具备输入，历史 O 的修正何时必须完成。只让多个 warp 同时存在，不等于计算已经重叠。

参考答案为 KV 和 score 设置双缓冲，先在等待当前 QK 完成之前预取下一块 KV，并在处理当前块的 CUDA softmax/correction 之前发起下一块 QK。这样既保留当前分数，又让矩阵引擎有机会与非矩阵工作并行。先从 s40 的完整源码中标记各引擎发起、等待与 buffer 释放位置。分析相邻 key tile 的交错是否减少空闲时间，同时检查 accumulator 或概率存储是否因交错而增加。性能与设计证据必须相互支持；缩短总时间不能单独证明某一种 overlap，时间线出现并行也不保证端到端更快。

## 定义、代码与参考

与 s40 使用完全相同的输入域与数学定义；本关改变执行安排，不改变 attention 语义。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s41_overlap/kernel.cu](../../students/cuda/s41_overlap/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s41_overlap/](../../solutions/s41_overlap/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：强制 cuDNN fused attention，返回 O+LSE；拒绝静默回退到数学实现。

源码阅读：[kernel.cu](../../solutions/s41_overlap/kernel.cu) 与 [sm100_attention.cuh](../../solutions/s40_attention/sm100_attention.cuh)；对照 s40 的发起、等待与 correction 位置，并结合 FA4 论文 §3.1.2 分析依赖。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 128, 1, 64)` | bf16/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 256, 3, 64)` | bf16/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 128, 2, 64)` | bf16/full | Q/K 为零而 V 非零，检查 V 的均值与完整输出写入。 |
| `large_logits` | stress | `(1, 256, 2, 128)` | bf16/full | 较大 QK，检查数值稳定性和输出舍入。 |
| `scale` | debug | `(1, 128, 2, 64)` | bf16/full | 显式 scale，防止硬编码 1/sqrt(D)。 |
| `rising_max` | stress | `(1, 512, 2, 128)` | bf16/full | 后续 key 的最大值上升，检验历史状态 rescale。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | bf16/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | bf16/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | bf16/full | 长 key 循环，观察融合和流水线的稳态作用。 |
| `scale_overflow_uniform` | stress | `(1, 128, 2, 64)` | bf16/full | 正有限 scale=3e38、Q/K 为零而 V 非零；检查 base-2 缩放溢出不会产生 NaN，O 为 V 均值且 LSE=log(N)。 |
| `scale_subnormal` | edge | `(1, 256, 2, 128)` | bf16/full | 最小正 FP32 次正规 scale 与非零 Q/K，检查 host/device 传递、FTZ 处理和多 key tile 的在线归一化。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s40 固定 shape/dtype/数学语义比较，记录瓶颈是否从 MMA 转移。

给出跨 tile 的就绪条件与不会覆盖活跃数据的证据；用时间线验证 overlap。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s41_overlap -v
./lab reference s41_overlap --case 0
./lab solution s41_overlap --case 0
./lab check s41_overlap --case 0
./lab check s41_overlap
./lab bench s41_overlap --case 0
./lab sanitize s41_overlap --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
