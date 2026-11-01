# s10_gemm · SIMT 分块 GEMM

阶段 1。前置：[s00_layout](s00_layout.md)

## 先理解数据复用，再使用矩阵引擎

矩阵乘的每个输出都是一个点积。逐输出独立读取虽然容易写，但同一 A 或 B 元素会被反复访问。分块的意义是让一批输出共享已搬入的数据，而不是把循环机械地切成小块。用一个矩形问题追踪读入元素被哪些输出消费，并计算每搬入一个字节完成多少乘加。

复用可以发生在 cache、shared memory 和寄存器。寄存器保存更多输出会减少重复工作，也会限制活跃线程。SIMT FP32 路线让这些代价直接可见；cuBLAS 是本关的快速外部基线，允许它明显快于你的第一版。短 K 的 QK 形状与长 K 的 PV 形状要分开观察。

## 定义、代码与参考

C=A B；A[M,K]、B[K,N]、C[M,N]；C 使用 FP32。名义计算量为 2MNK FLOPs。

接口：`[M,K] @ [K,N] → [M,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s10_gemm/kernel.cu](../../students/cuda/s10_gemm/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s10_gemm/](../../solutions/s10_gemm/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS GEMM；保持输入 dtype 与 FP32 输出一致。

源码阅读：NVIDIA CUTLASS efficient GEMM 文档的数据复用层次；本地 solutions/s10_gemm/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(7, 11, 13)` | fp32/full | 非方矩阵，区分 M/N/K 与两个输入的行列。 |
| `zero` | debug | `(64, 64, 64)` | fp32/full | 零乘积及完整输出写入。 |
| `coordinates` | debug | `(128, 64, 32)` | fp32/full | 可辨认坐标模式，定位 operand/accumulator layout。 |
| `long_k` | stress | `(128, 128, 2048)` | fp32/full | 多次累加与缓冲复用，避免只验证一次 MMA。 |
| `tails` | edge | `(129, 65, 257)` | fp32/full | 三个维度独立尾部，检查 predication。 |
| `square` | perf | `(1024, 1024, 1024)` | fp32/full | 多 CTA 持续 GEMM，观察复用和计算吞吐。 |
| `qk_shape` | perf | `(2048, 2048, 128)` | fp32/full | QK 风格短 K、大输出矩阵。 |
| `pv_shape` | perf | `(2048, 128, 2048)` | fp32/full | PV 风格窄 N、长 K，与 QK 使用不同瓶颈。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

对比不复用与复用版本及 cuBLAS；短 K 和长 K 分开解释。

从源码说明复用层次、线程输出 ownership、尾部和寄存器开销。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s10_gemm -v
./lab reference s10_gemm --case 0
./lab solution s10_gemm --case 0
./lab check s10_gemm --case 0
./lab check s10_gemm
./lab bench s10_gemm --case 0
./lab sanitize s10_gemm --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
