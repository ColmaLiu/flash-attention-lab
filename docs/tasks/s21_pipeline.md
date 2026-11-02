# s21_pipeline · 异步搬运的 Tensor Core GEMM

阶段 2。前置：[s20_mma](s20_mma.md)

## 异步发起与有效重叠不同

异步 copy 让发起线程在数据尚未到达时继续执行。若下一条操作立即等待所有数据，程序依然近似串行。流水线需要让未来数据的传输与当前数据的计算重叠，同时证明没有读未完成的数据，也没有覆盖仍被使用的 buffer。

把每个 buffer 的状态写成生命周期：可写、传输中、可读、消费中、可复用。初始填充与最后排空不是稳态循环的特例可以忽略。缓冲更多会扩大 shared memory 或寄存器需求，所以性能问题是延迟隐藏与资源占用之间的取舍。短 K 不能摊销准备成本，长 K 则能暴露错误的 phase 回绕。

## 定义、代码与参考

数学契约与 s20 完全相同。参考保留相同 128×64×32 tile、XOR shared layout、PTX ldmatrix/mma.sync fragment 和输出 ownership，改用 cp.async 双缓冲。s20 提供同步对照；实验固定 shape、输入值和 dtype，分别检查异步搬运、重叠与资源变化。

接口：`[M,K] @ [K,N] → [M,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s21_pipeline/kernel.cu](../../students/cuda/s21_pipeline/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s21_pipeline/](../../solutions/s21_pipeline/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；对已经量化的相同输入执行 FP64 矩阵乘，输出按公共契约比较。
- 快速库基线：cuBLAS GEMM；保持输入 dtype 与 FP32 输出一致。

源码阅读：PTX ISA cp.async completion groups；本地 solutions/s21_pipeline/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(32, 48, 16)` | bf16/full | 非方矩阵，区分 M/N/K 与两个输入的行列。 |
| `zero` | debug | `(64, 64, 64)` | bf16/full | 零乘积及完整输出写入。 |
| `coordinates` | debug | `(128, 64, 32)` | bf16/full | 可辨认坐标模式，定位 operand/accumulator layout。 |
| `long_k` | stress | `(128, 128, 2048)` | bf16/full | 多次累加与缓冲复用，避免只验证一次 MMA。 |
| `rectangular` | edge | `(192, 144, 80)` | bf16/full | 合法整倍数的非方尺寸，避免只支持方阵固定 tile。 |
| `square` | perf | `(1024, 1024, 1024)` | bf16/full | 多 CTA 持续 GEMM，观察复用和计算吞吐。 |
| `qk_shape` | perf | `(2048, 2048, 128)` | bf16/full | QK 风格短 K、大输出矩阵。 |
| `pv_shape` | perf | `(2048, 128, 2048)` | bf16/full | PV 风格窄 N、长 K，与 QK 使用不同瓶颈。 |
| `fp16` | debug | `(128, 96, 64)` | fp16/full | FP16 特化及 FP32 累加路径。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s20 相同形状和 dtype，比较短/长 K，记录是否有真实 overlap 与 spills。

给出 producer/consumer 生命周期与完成事件；检验 prologue、稳态、drain。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s21_pipeline -v
./lab reference s21_pipeline --case 0
./lab solution s21_pipeline --case 0
./lab check s21_pipeline --case 0
./lab check s21_pipeline
./lab bench s21_pipeline --case 0
./lab sanitize s21_pipeline --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
