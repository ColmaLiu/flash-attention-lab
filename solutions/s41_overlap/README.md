# CUDA/PTX：相邻 key tile 的 QK 与 softmax 交错

[kernel.cu](kernel.cu) 选择 `Overlap=true, Numerics=false, Persistent=false`，设备实现见 [sm100_attention.cuh](../s40_attention/sm100_attention.cuh)。数据契约、128×64 query/key tile 和 FP32 状态与 [s40](../s40_attention/README.md) 相同，新增两套 KV 和 score 存储。

特殊 scale 的 FP64 raw-score 差值与自然对数 LSE 路径同样来自该共享 header；host 分派条件、FP32 累加及 LSE 表示范围限制见 [s40 数值边界](../s40_attention/README.md#共享数值路径的边界)。执行交错不改变这些数值条件。

## 相邻迭代如何交接

等待当前 score 完成之前，线程 0 发起下一块 KV 的 TMA，使搬运有机会与仍在进行的当前 QK 重叠；score 完成后，四个 warp 将其从 TMEM 读到寄存器。下一块 KV 完成后，发起下一块 QK，再执行当前块 softmax、历史 O 修正和 PV。score 的两个 TMEM 区域分别位于列 0 和 64，PV 使用列 128 起的 D 列。

`slot=tile%2` 选择存储，`(tile/2)&1` 选择每个 slot 的 barrier phase。不同槽中的 K/V 和 score 可以同时存活。每轮尾部的 `before_thread_sync` fence、CTA barrier 和 `after_thread_sync` fence 将各 warp 经 `wait::ld` 完成的 PV 读取交接给发起线程，之后才允许回绕到旧槽；最后一块不再预取。请逐项追踪首次填充、奇偶回绕与 drain，尤其区分等待 TMA 和等待 MMA。

这一安排给 QK 与 CUDA softmax/correction 提供重叠机会。下一块 KV 的等待仍在当前 softmax 之前，PV 也仍逐块等待，所以它不是所有引擎都始终并行的完整生产流水线。时间线和相同输入的测量用于确定哪些等待真正缩短。

## 资源与验证

D64/D128 的动态 shared memory 从 s40 的 48/80 KiB 增至 64/112 KiB；TMEM 仍分配 256 列。额外存储可能减少驻留 CTA，少 heads、长序列与多 heads 的收益不能互相替代。

```bash
./lab check s41_overlap --solution
./lab solution s41_overlap --case 6
./lab solution s41_overlap --case 7
./lab solution s41_overlap --case 8
./lab sanitize s41_overlap --solution --tool racecheck --case 5
./lab sanitize s41_overlap --solution --tool synccheck --case 1
```

使用 s40 和 cuDNN 作为同契约比较对象；graph/eager 分别报告。算法、指令与上游设计来源见 [s40 README](../s40_attention/README.md)。本地双缓冲协议是可直接修改的 CUDA/PTX 实验，性能结论必须绑定实际测量。
