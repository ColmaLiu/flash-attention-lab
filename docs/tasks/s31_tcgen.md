# s31_tcgen · Blackwell tcgen05 与 TMEM

阶段 3。前置：[s20_mma](s20_mma.md)、[s30_tma](s30_tma.md)

## TMEM 让 accumulator 成为显式资源

Blackwell 的 tcgen05 MMA 与传统 warp 同步矩阵指令具有不同的完成语义，accumulator 存在 TMEM。SMEM operand descriptor 描述输入，TMEM 地址描述结果，它们都不是普通的全局指针。理解指令合法域与资源 lifetime，比先追求复杂流水线更重要。

本关参考使用固定版本 CUTLASS 的完整 Blackwell 源码。沿着 host 参数、collective、MMA、epilogue 阅读，定位真实的 tcgen05 发起、完成通知和 TMEM 访问。先让完整 GEMM 正确覆盖多 CTA 和长 K，再用 s32 讨论角色划分与 overlap。不要把能够编译一个 intrinsic 当作完整验证。

## 定义、代码与参考

BF16 A[M,K]、B[K,N]，FP32 C[M,N]。M 是 64 的倍数、N 是 8 的倍数、K 是 16 的倍数。

接口：`[M,K] @ [K,N] → [M,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s31_tcgen/kernel.cu](../../students/cuda/s31_tcgen/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s31_tcgen/](../../solutions/s31_tcgen/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS GEMM；保持输入 dtype 与 FP32 输出一致。

源码阅读：references/cutlass-sm100/examples/cute/tutorial/blackwell/ 与 include/cute/arch/ 的 SM100 源码；solutions/s31_tcgen/。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

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

与 s20、cuBLAS 在共同形状比较，解释架构变化与性能差距。

说明 SMEM descriptor、TMEM lifetime、发布/等待和输出交接；检查实际 tcgen05 指令。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s31_tcgen -v
./lab reference s31_tcgen --case 0
./lab solution s31_tcgen --case 0
./lab check s31_tcgen --case 0
./lab check s31_tcgen
./lab bench s31_tcgen --case 0
./lab sanitize s31_tcgen --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
