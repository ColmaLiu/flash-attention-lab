# s43_schedule · 边界、调度与性能结业实验

阶段 4。前置：[s42_numerics](s42_numerics.md)

## 让最慢的工作块也进入性能分析

causal attention 的不同 query 区域拥有不同有效 key 数，工作量呈三角分布。尾部 CTA 和不均匀任务可能延长总完成时间。调度策略、资源用量、tile 大小和驻留 CTA 数互相影响，所以单个吞吐数字不足以决定最佳方案。

结业实验面向一个明确工作负载集合：少 heads、多 heads、长序列、causal，以及合法尾部。先为每一种变更提出可被结果推翻的假设，再在相同代码指纹和环境下运行正确性、sanitizer 与双边基线。展示所有目标 case 的结果及损失，说明实现何时快、何时慢、为什么仍有差距。

## 定义、代码与参考

扩展到 FP16/BF16、任意正 N、D64/128、causal/noncausal。其余范围仍为 contiguous self-attention forward，无 dropout/bias/GQA。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s43_schedule/kernel.cu](../../students/cuda/s43_schedule/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s43_schedule/](../../solutions/s43_schedule/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：强制 cuDNN fused attention，返回 O+LSE；拒绝静默回退到数学实现。

源码阅读：FA4 论文 §3.3；本地 tile_scheduler.py、flash_fwd_sm100.py 与 solutions/s43_schedule/。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 128, 1, 64)` | bf16/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 256, 3, 64)` | bf16/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 128, 2, 64)` | bf16/full | Zero Q/K and nonzero V: verify mean of V, not vacuous zero output. |
| `large_logits` | stress | `(1, 256, 2, 128)` | bf16/full | 较大 QK，检查数值稳定性和输出舍入。 |
| `scale` | debug | `(1, 128, 2, 64)` | bf16/full | 显式 scale，防止硬编码 1/sqrt(D)。 |
| `single` | edge | `(1, 1, 2, 64)` | bf16/full | 单 token 无归约长度，输出应等于 V。 |
| `tail_causal` | edge | `(2, 129, 3, 64)` | bf16/causal | query/key 尾部、含对角线的 causal 及多头索引。 |
| `tail_full` | edge | `(1, 257, 2, 128)` | bf16/full | 非 causal 尾部，检查掩码不会错误地依赖 causal。 |
| `rising_max` | stress | `(1, 512, 2, 128)` | bf16/full | 后续 key 的最大值上升，检验历史状态 rescale。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | bf16/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | bf16/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | bf16/full | 长 key 循环，观察融合和流水线的稳态作用。 |
| `fp16_tail` | edge | `(2, 257, 2, 128)` | fp16/causal | FP16 动态范围、causal 与尾部。 |
| `causal_perf` | perf | `(1, 4096, 8, 128)` | bf16/causal | 三角工作量下的尾部调度。 |
| `causal_uniform` | edge | `(1, 128, 2, 64)` | bf16/causal | Uniform causal weights require the exact prefix mean of nonzero V. |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

至少比较两种调度/资源配置；以全部目标 case 的结果说明取舍，不只展示最快一个点。

提供完整正确性、内存安全、设计证据与同机参考性能，分别说明吞吐和尾延迟。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s43_schedule -v
./lab reference s43_schedule --case 0
./lab solution s43_schedule --case 0
./lab check s43_schedule --case 0
./lab check s43_schedule
./lab bench s43_schedule --case 0
./lab sanitize s43_schedule --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
