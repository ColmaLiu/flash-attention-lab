# s40_attention · SM100 attention 基线

阶段 4。前置：[s22_fa2](s22_fa2.md)、[s32_specialize](s32_specialize.md)

## 先把新的硬件资源接成正确 attention

GEMM 的结果可以直接写输出，attention 的 QK 结果还要经过 softmax 后成为 PV 的输入。因此 SM100 GEMM 并不能机械拼接成高效 attention。需要追踪 S、P、O 在 TMEM、SMEM、寄存器间的流转，明确每次精度转换和 buffer 重用。

本阶段答案采用 `third_party/flash-attention` 中固定的官方 CuTe DSL forward 源码。先按单个 query 块阅读数据生命周期，再阅读跨块 overlap。学生使用自己的 CUDA C++/PTX 入口逐步实现和优化。

## 定义、代码与参考

BF16 BNHD，D=64/128，N 是 128 的倍数，先只支持 noncausal。返回 O 与自然对数 LSE。

接口：`[B,N,H,D] → O 同 shape/dtype，LSE [B,H,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s40_attention/kernel.cu](../../students/cuda/s40_attention/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s40_attention/](../../solutions/s40_attention/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：强制 cuDNN fused attention，返回 O+LSE；拒绝静默回退到数学实现。

源码阅读：third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py；对应 solutions/s40_attention/。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(1, 128, 1, 64)` | bf16/full | 单头基线，先核对 O 与自然对数 LSE。 |
| `heads` | debug | `(2, 256, 3, 64)` | bf16/full | B、H 均非一，识别 BNHD 与 BHND 混淆。 |
| `uniform` | debug | `(1, 128, 2, 64)` | bf16/full | Zero Q/K and nonzero V: verify mean of V, not vacuous zero output. |
| `large_logits` | stress | `(1, 256, 2, 128)` | bf16/full | 较大 QK，检查数值稳定性和输出舍入。 |
| `scale` | debug | `(1, 128, 2, 64)` | bf16/full | 显式 scale，防止硬编码 1/sqrt(D)。 |
| `rising_max` | stress | `(1, 512, 2, 128)` | bf16/full | 后续 key 的最大值上升，检验历史状态 rescale。 |
| `few_heads` | perf | `(1, 1024, 2, 64)` | bf16/full | 少量 heads，观察 query 维并行度。 |
| `many_heads` | perf | `(2, 1024, 16, 128)` | bf16/full | 较多独立 heads，比较吞吐与资源占用。 |
| `long_sequence` | perf | `(1, 4096, 8, 128)` | bf16/full | 长 key 循环，观察融合和流水线的稳态作用。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s22 及 cuDNN 同一 BF16 输入比较，建立后续优化共同基线。

标明 S、P、O 及 LSE 的存储位置、精度与生命周期；不得包装官方 attention。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s40_attention -v
./lab reference s40_attention --case 0
./lab solution s40_attention --case 0
./lab check s40_attention --case 0
./lab check s40_attention
./lab bench s40_attention --case 0
./lab sanitize s40_attention --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
