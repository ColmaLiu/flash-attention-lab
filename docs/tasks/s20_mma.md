# s20_mma · Warp Tensor Core GEMM

阶段 2。前置：[s10_gemm](s10_gemm.md)

## 矩阵指令改变数据的拥有方式

Tensor Core 不是把普通点积换一个函数名。一次矩阵指令由线程集体提供分布式 operands，并将 accumulator 分布在线程寄存器中。数学矩阵坐标、shared-memory 布局和寄存器 fragment 是三种不同视图，必须画出它们之间的对应。先用坐标模式识别交换、重复与漏写，再用随机输入比较数值。

FP16/BF16 输入与 FP32 accumulation 是两个独立精度选择。参考必须从已经量化的相同输入计算，不能拿未量化的 FP32 输入作对照。小矩阵验证指令语义，多 CTA、长 K 验证完整 GEMM。WMMA 可作为起点；PTX mma.sync/ldmatrix 适合进一步理解显式 fragment，但不要求使用 CuTe。

## 定义、代码与参考

C=AB；输入 FP16/BF16、输出 FP32。输出舍入预算不包含输入量化误差，因为 oracle 使用相同输入值。

接口：`[M,K] @ [K,N] → [M,N] FP32`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s20_mma/kernel.cu](../../students/cuda/s20_mma/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s20_mma/](../../solutions/s20_mma/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：cuBLAS GEMM；保持输入 dtype 与 FP32 输出一致。

源码阅读：PTX ISA 的 mma.sync 与 ldmatrix；本地 solutions/s20_mma/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

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

同一量化输入与 cuBLAS 比较；一次 MMA 的正确性不代替完整 GEMM。

解释 fragment/ldmatrix/SMEM 的对应；用生成指令确认 Tensor Core 执行。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s20_mma -v
./lab reference s20_mma --case 0
./lab solution s20_mma --case 0
./lab check s20_mma --case 0
./lab check s20_mma
./lab bench s20_mma --case 0
./lab sanitize s20_mma --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
