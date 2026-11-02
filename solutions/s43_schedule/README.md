# CUDA/PTX：静态 persistent 调度与完整边界

[kernel.cu](kernel.cu) 默认选择 `Overlap=true, Numerics=true, Persistent=true, Polynomial=false`，使用 [共享设备代码](../s40_attention/sm100_attention.cuh)。本关扩展到 FP16/BF16、任意正 N、D64/128、causal/noncausal；输入输出仍为 contiguous BNHD，LSE 为 BHN FP32。

## 任务映射与资源复用

每个任务对应 `(batch,head,query_tile)`。网格大小为 `min(total_tasks, 2*SM_count)`，每个 CTA 依次处理 `blockIdx.x + k*gridDim.x` 的任务。causal 时跨所有 batch/head 按 query 位置从后向前分配：令 `batch_heads=B*H`，则 `query_tile=qtiles-1-task/batch_heads`、`bh=task%batch_heads`。相同 query 位置的所有 head 连续出现，再转到更早的 query；这样较长任务在全局任务序列中靠前。它是静态映射，没有全局任务队列，也不保证达到动态调度的负载平衡。

同一 CTA 跨任务保留 TMEM allocation，每个任务重置行统计、O 寄存器和 barriers。结束任务前完成所有异步工作、读取与输出；PV 的 TMEM 读以 `wait::ld` 和 before/after-thread-sync fence 建立 CTA 交接，epilogue 对 Q 的 generic shared 访问由每线程 async proxy fence 再汇合，才允许下一任务 TMA 覆盖。invalid barriers 后再进入下一任务。否则即使单个 query tile 正确，persistent 复用仍可能读到旧 phase 或覆写活跃数据。

TMA 对 query/key 尾部零填充；score 用 `key<n` 和 `key<=row` 施加有效域与含对角线的 causal mask。O/LSE store 检查 query 是否有效。所有集体同步仍由完整 CTA/warp 参与，不能让尾部线程提前 return。

## 比较与完成标准

```bash
./lab check s43_schedule --solution
./lab solution s43_schedule --case 9
./lab solution s43_schedule --case 10
./lab solution s43_schedule --case 11
./lab solution s43_schedule --case 13
./lab solution s43_schedule --case 15
./lab sanitize s43_schedule --solution --tool memcheck --case 12
./lab sanitize s43_schedule --solution --tool memcheck --case 16
./lab sanitize s43_schedule --solution --tool racecheck --case 16
./lab sanitize s43_schedule --solution --tool synccheck --case 16
```

公共 noncausal BF16 perf case 与 s42 比较，causal 新域与 cuDNN 比较。`entry.py` 默认 `persistent=True`；通过 `prepare("s43_schedule", inputs, config={**cfg, "persistent": False}, solution=True)` 可保留相同数值选项和输入域，切换到普通 grid。这个开关同时关闭 grid 限额与跨 batch/head 的 causal 长任务优先映射，比较结果对应整个调度策略，不能单独归因为任务排序。要分别衡量这两项仍需进一步控制变量。少任务可能根本没有跨任务循环，所以只比较一个小例不能验证调度机制。`persistent_waves` 使用 512 个 query tile，在 B200 上超过两倍 SM 的 grid 上限，实际进入跨任务复用；更换设备时还需核对 `total_tasks > 2*SM_count` 是否成立。`persistent_reuse` 保留 512 个任务但将每个任务缩短为 N128，用于逐访存 sanitizer 检查相同的跨任务资源交接；长 `persistent_waves` 仍用于性能比较。

网格上限的 `2*SM_count` 是配置选择，不保证每个 SM 同时驻留两个 CTA；shared、寄存器与 TMEM 决定实际驻留。局部更长 CTA 生命周期、头间工作量与尾部任务分布会影响总耗时。记录所有目标输入的增益与退化，再依据时间线和资源用量修改调度。

两个独立数值开关与消融接口见 [s42](../s42_numerics/README.md)，完整数据布局、同步及来源见 [s40](../s40_attention/README.md)。工作量排序思路可对照 [FA4 论文](https://arxiv.org/abs/2603.05451v1)，本实现的映射与资源协议以本地源码为准。
