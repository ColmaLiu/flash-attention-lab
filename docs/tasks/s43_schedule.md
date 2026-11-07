# s43_schedule · 边界、调度与性能结业实验

阶段 4。前置：[s42_numerics](s42_numerics.md)

## 让最慢的工作块也进入性能分析

causal attention 的不同 query 区域拥有不同有效 key 数，工作量呈三角分布。尾部 CTA 和不均匀任务可能延长总完成时间。调度策略、资源用量、tile 大小和驻留 CTA 数互相影响，所以单个吞吐数字不足以决定最佳方案。

参考答案按任务数量选择两条原生流水线：paired 将相邻两个 query tile 放入同一 CTA 复用 KV；single 使用一个 query tile 和两个交替 score 槽，增加少任务输入的并行度。两者都可使用 static persistent 分配，只有任务数超过 grid 上限时同一 CTA 才继续处理下一任务。causal 任务跨所有 batch/head 按 query 位置从后向前排序，先安排较长工作块。prepare 配置 `persistent=False` 同时关闭网格限额与该排序，保留相同路径选择和数值选项；该对比对应整个调度策略，不能单独证明任务排序的收益。

结业实验面向一个明确工作负载集合：少 heads、多 heads、长序列、causal，以及合法尾部。先为每一种变更提出可被结果推翻的假设，再在相同代码指纹和环境下运行正确性、sanitizer 与双边基线。展示所有目标 case 的结果及损失，说明实现何时快、何时慢、为什么仍有差距。

## 定义、代码与参考

扩展到 FP16/BF16、任意正 N、D64/128、causal/noncausal。其余范围仍为 contiguous self-attention forward，无 dropout/bias/GQA。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s43_schedule/kernel.cu](../../students/cuda/s43_schedule/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s43_schedule/](../../solutions/s43_schedule/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：强制 cuDNN fused attention，返回 O+LSE；拒绝静默回退到数学实现。

源码阅读：[kernel.cu](../../solutions/s43_schedule/kernel.cu) 选择路径，[attention_common.cuh](../../solutions/s43_schedule/attention_common.cuh) 准备 tensor maps；两个完整设备入口为 [attention_dense.cuh](../../solutions/s43_schedule/attention_dense.cuh) 和 [attention_single.cuh](../../solutions/s43_schedule/attention_single.cuh)，共享 [attention_exp.cuh](../../solutions/s43_schedule/attention_exp.cuh) 的原生 packed 四阶指数。详细布局、逐项 phase、源码来源与实验入口见 [答案说明](../../solutions/s43_schedule/README.md)。固定版本与官方链接见 [sources](../sources.md)，课程源码索引见 [参考实现与源码](../reference-implementations.md)。

## 从前置流水线到两种任务粒度

令 `P=B*H*ceil(N/256)`，`SM` 为设备 SM 数量。causal 且 `P<SM`，或 noncausal 且 `2*P<SM`，选择 single；其余选择 paired。每次执行只发射一个本地 CUDA/PTX kernel。两条路径都使用 128-query、128-key 的 MMA tile，FP32 行统计保存在寄存器，P 与跨 key tile 累加的 O 放在 TMEM。

| 路径 | 线程、缓冲与依赖 |
|---|---|
| paired | 384 线程，两个 softmax WG 与一个控制 WG；两个 query 共用两组完整 KV pair。每个 WG 有独立 S/P/O，MMA 专用线程发 QK/PV。PV(t) 完成后才为该 WG 发 QK(t+1)，保护 S/P 别名复用；两个 WG 的计算交错。 |
| single | 256 线程，一个 softmax WG 与一个控制 WG；一个 query、两个 S/P 槽、三组完整 KV pair。MMA 预先发 QK(0/1)，之后 PV(t) 完成再发 QK(t+2)。softmax 可先计算 t+1 的分数与概率，但修改 O 前必须等待 PV(t)。 |

paired 的 score/Pready/PV phase 为每个 WG 的 `t&1`，KV phase 为 `(t/2)&1`、empty count 为 2；single 的 score/Pready/PV 使用槽 `t%2`、phase `(t/2)&1`，KV 使用槽 `t%3`、phase `(t/3)&1`、empty count 为 1。producer 复用 KV 时等上一代 empty，consumer 发布 Pready 前完成 TMEM stores 与 WG fence/barrier 交接。PTX 8.7 的同 accumulator/shape MMA 顺序保证不覆盖这里不同 S/O 的复用关系，因此显式 PV 完成等待不能仅凭“同一线程发射”删除。

默认启用混合四阶指数与 delayed rescale。公共 scale 必须可表示为正且有限的 FP32。令 `scale2=scale*log2(e)`；`isnormal(scale) && isnormal(scale2) && scale2<=1` 时使用原始 QK 最大值归约与 packed `f32x2` 独立乘法、减法，使 score 与最大值采用相同的乘法舍入，其余合法 scale 保留原始行参考值，在 FP64 中计算分数差乘 scale、correction 与 LSE 的参考值项，避免 scale2 溢出或次正规数被 flush-to-zero。原始 QK 的 FP32 累加与最终 FP32 LSE 仍受表示范围限制，不保证任意有限输入的全域精度。

`polynomial_exp=True` 在正常 scale 分支中将每行前 96 个 score 每四个中的第一对交给 packed 四阶多项式，最后 32 个全走 SFU，即 48 个四阶、80 个 SFU；关闭该开关后全走 SFU。极端 scale 分支与 correction 始终走 SFU。四阶 helper 用 round-down 偏置归约、四条 packed FMA 和指数位重建，末尾显式 FTZ 保证 masked `-inf` 严格变成零。先舍入 score 的缩放乘法再减参考值，才保留最大值指数为零的性质；对已舍入 maximum 使用 FMA 会让较大有限 QK 的乘法残差使 P 溢出或全零。delayed rescale 把有效指数输入限制到至多 4；关闭后至多为 0。系数来自固定 FA4 源码，helper 保留其 BSD-3-Clause 许可。本关不再使用 s42 的六阶实现；CPU 误差扫描也不替代原容差下的完整 CUDA 检查，尤其要验证近零 LSE、mask 和尺度切换。

同一 warp 的 correction 全为 1 时跳过 O 的读乘写，最终归一化用一次倒数和 `int4` shared 重排输出。descriptor 基址与增量按 16 bytes 编码，MMA warp 先 `elect.sync` 选出发射线程，减少逐条 descriptor/选举开销。混合指数与 delayed rescale 可以独立消融，性能对照也必须使用相同数值选项。

TMEM 在 CTA 内跨任务保留，每个任务重置统计与 barriers。paired 由 TMA 线程 256 初始化 mbarrier，再通过 async proxy fence 与全 CTA barrier 发布；single 由 thread 0 初始化。两条路径都由 thread 0 在任务末尾排空各 KV 槽最后一代 empty、invalidate，再经全 CTA barrier 交给下一任务。Q 的 generic shared 输出访问经过每线程 async proxy fence 和汇合后，才可被下次 TMA 覆盖。

手工角色同步使用无 `.aligned` 的 `barrier.sync`，named barrier 只汇合同一角色的 128 个线程，公共 CTA barrier 只出现在初始化与任务边界；不能把分支中的角色同步替换为隐含全 CTA 同指令参与承诺的 legacy `bar.sync`。paired 还动态调整 softmax/control 寄存器预算，并在恢复控制 WG 寄存器前先汇合；single 使用静态寄存器分配。`try_wait` 轮询、TMEM allocation 和寄存器重分配各有不同目的，不能当成同一种调度成本。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 128, 1, 64)` | bf16/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 256, 3, 64)` | bf16/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 128, 2, 64)` | bf16/full | Q/K 为零而 V 非零，检查 V 的均值与完整输出写入。 |
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
| `causal_uniform` | edge | `(1, 128, 2, 64)` | bf16/causal | 均匀 causal 权重要求输出为非零 V 的精确前缀均值。 |
| `persistent_waves` | perf | `(1, 8192, 8, 128)` | bf16/causal | 512 个 query tile 对应 256 个 paired tasks；检查长 causal 循环与排序，是否跨任务须比较实际 task 数和 grid 上限。 |
| `persistent_reuse` | stress | `(1, 128, 512, 64)` | bf16/causal | 512 个短任务，检查跨任务 barrier/TMEM/Q 复用；N128 只用一个 KV 槽，适合逐访存 sanitizer。 |
| `scale_overflow_uniform` | stress | `(1, 128, 2, 64)` | bf16/full | 正有限 scale=3e38、Q/K 为零而 V 非零；检查 base-2 缩放溢出不会产生 NaN，O 为 V 均值且 LSE=log(N)。 |
| `scale_subnormal` | edge | `(1, 256, 2, 128)` | bf16/full | 最小正 FP32 次正规 scale 与非零 Q/K，检查 host/device 传递、FTZ 处理和多 key tile 的在线归一化。 |
| `scaled_max_fp16` | stress | `(1, 128, 1, 64)` | fp16/full | 较大且精确的有限 QK、默认 scale，防止 FMA 残差使 P 溢出；O=1。 |
| `scaled_max_fp16_multi` | stress | `(1, 256, 128, 64)` | fp16/full | 相同数值反例，增加任务数及双 key tile。 |
| `scaled_max_bf16` | stress | `(1, 128, 1, 64)` | bf16/full | 较大且精确的有限 QK，防止 FMA 残差使 P 全零；O=1。 |
| `scaled_max_bf16_multi` | stress | `(1, 256, 128, 64)` | bf16/full | 相同数值反例，增加任务数及双 key tile。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

跨任务复用要求所选路径 `total_tasks>2*SM`，不能把 128-query tile 数直接当成 paired task 数。为补充两槽交接的验证，可手工构造 causal `[1,256,H,64]`、`H>2*SM`（满足条件时例如 H=384），让 paired 的两个 KV 槽在同一 CTA 的后续任务中再次使用；此控制例不增加或替换上表的固定案例。

## 实验与完成标准

至少比较两种调度/资源配置；以全部目标 case 的结果说明取舍，不只展示最快一个点。

提供完整正确性、内存安全、设计证据与同机参考性能，分别说明吞吐和尾延迟。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。90% 是参考答案维护的验收门槛，不作为学生完成本关的强制性能线；学生仍需提交受控计时与差距分析。最终参考答案的五个 perf case（9、10、11、13、15）必须逐例达到库吞吐的 90%，即 `library_graph_median / solution_graph_median >= 0.90`；默认配置、O+LSE 契约与 Graph 捕获批量一致，不用平均值掩盖某例失败。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化；测量与机器信息只放入忽略目录。

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
