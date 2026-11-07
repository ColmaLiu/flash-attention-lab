# s32_specialize · Warp specialization 流水线

阶段 3。前置：[s21_pipeline](s21_pipeline.md)、[s31_tcgen](s31_tcgen.md)

## 让不同 warp 服务不同执行引擎

TMA、矩阵引擎和普通线程执行单元可以并行工作，但它们对数据的就绪条件不同。warp specialization 将角色分开，让各自推进适合的任务；这同时使同步参与者、buffer 释放时机和寄存器需求更难推理。

答案使用独立设备实现，增加多组 operand 缓冲和独立的 TMA producer warp，并按短 K、宽 N 长 K、窄 N 长 K 选择 tile。宽 N 长 K 默认 128×64×128 四缓冲，满足面板对齐条件时使用 rank-3 TMA；窄 N 使用 64×32×256 四缓冲，以实际 32 列 B 布局增加 CTA 数并避免默认 split-K。详细布局与显式消融选项见 [答案说明](../../solutions/s32_specialize/README.md)。TMA 完成将 full barrier 推进，MMA 完成将 empty barrier 推进；producer 只有等到 empty 才能覆盖旧数据，MMA 只有等到 full 才能消费新数据。先把 s31 的依赖画成有向图，区分必要依赖与保守等待，再比较参考代码如何组织生产与消费。检查第一次使用、每次回绕及最后一次释放；最终 TMEM 读取还须以 `wait::ld` 和 before/after-thread-sync fence 包围的 CTA 交接，保证释放发生在所有读取之后。性能比较必须包含短 K 与长 K：稳态更快的实现也可能因准备开销或驻留 CTA 减少而输在短任务。

## 定义、代码与参考

数学与 dtype 与 s31 相同。与 s21 的相同之处是生命周期，不同之处是 MMA 自身也异步。

接口：`[M,K] @ [K,N] → [M,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s32_specialize/kernel.cu](../../students/cuda/s32_specialize/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s32_specialize/](../../solutions/s32_specialize/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；对已经量化的相同输入执行 FP64 矩阵乘，输出按公共契约比较。
- 快速库基线：cuBLAS GEMM；保持输入 dtype 与 FP32 输出一致。

源码阅读：[kernel.cu](../../solutions/s32_specialize/kernel.cu) 与独立的 [kernel_impl.cuh](../../solutions/s32_specialize/kernel_impl.cuh)；PTX tcgen05 commit/completion。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(64, 16, 32)` | bf16/full | 非方矩阵，区分 M/N/K 与两个输入的行列。 |
| `zero` | debug | `(64, 64, 64)` | bf16/full | 零乘积及完整输出写入。 |
| `coordinates` | debug | `(128, 64, 32)` | bf16/full | 可辨认坐标模式，定位 operand/accumulator layout。 |
| `long_k` | stress | `(128, 128, 2048)` | bf16/full | 多次累加与缓冲复用，避免只验证一次 MMA。 |
| `rectangular` | edge | `(192, 136, 80)` | bf16/full | 合法整倍数的非方尺寸，避免只支持方阵固定 tile。 |
| `square` | perf | `(1024, 1024, 1024)` | bf16/full | 多 CTA 持续 GEMM，观察复用和计算吞吐。 |
| `qk_shape` | perf | `(2048, 2048, 128)` | bf16/full | QK 风格短 K、大输出矩阵。 |
| `pv_shape` | perf | `(2048, 128, 2048)` | bf16/full | PV 风格窄 N、长 K，与 QK 使用不同瓶颈。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s31 同一 GEMM 对比，多缓冲数量与资源占用作为变量，不预设越多越快。总体阶段比较同时包含 tile 选择；消融时固定 tile 与 tensor-map 路线，再改变 stages 或 issuer 分工。参考答案的维护验收另要求全部三个 perf case 逐项满足同组 CUDA Graph median 的 `library_ms / solution_ms >= 0.9`，不得用平均值或遗漏案例代替；这不是学生通过本关的门槛。

用状态图证明所有缓冲读写与复用有顺序，短 K/长 K 均能 drain。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s32_specialize -v
./lab reference s32_specialize --case 0
./lab solution s32_specialize --case 0
./lab check s32_specialize --case 0
./lab check s32_specialize
./lab bench s32_specialize --case 0
./lab sanitize s32_specialize --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
