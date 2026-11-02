# 参考答案：从代码定位困难，再验证优化

参考答案是在学生编写遇到困难时可以直接学习、修改和测量的 CUDA C++/PTX 实现。数学公式、线程分工、内存布局、指令与同步必须能沿着本地源码逐项对应；核心计算不能隐藏在库调用或编译 DSL 后面。答案还必须具备有用的性能：通过代表性输入与相邻阶段、优化库对比，确认其优化机制生效，并解释剩余差距。

`solutions/` 保存完整设备源码和 host 准备代码。每课从 `entry.py` 进入 `kernel.cu`；需要复用时只共享本仓库内可直接阅读的 helper。输出与学生入口使用相同契约。FP64 oracle 用于判断数学正确性，优化库用于判断性能差距；两者均有独立入口，不能代替参考答案。

## 阅读顺序

先运行 `./lab check ID --solution --case 0`，再结合输入 shape 阅读下面的设备入口。跟踪一个线程、一个 warp 或一个 CTA 拥有哪些元素，逐项核对地址、有效范围、同步参与者与数据释放时机。遇到共用 helper 时继续读到实际指令，不止于 host 参数或 kernel launch。

| 课程 | 本地设备入口 | 阅读重点 |
|---|---|---|
| s00_layout | [kernel.cu](../solutions/s00_layout/kernel.cu) | 线程与坐标、合并访存、shared-memory 转置 |
| s01_softmax | [kernel.cu](../solutions/s01_softmax/kernel.cu) | 最大值与求和归约、宽行分工、稳定归一化 |
| s10_gemm | [kernel.cu](../solutions/s10_gemm/kernel.cu) | SIMT 分块、寄存器输出块、尾部处理 |
| s11_materialized | [kernel.cu](../solutions/s11_materialized/kernel.cu) | QK、softmax、PV 三步的临时内存与 dtype |
| s12_online | [kernel.cu](../solutions/s12_online/kernel.cu) | half-warp 行 ownership、转置/padding shared 布局、行统计与 O 的 rescale |
| s20_mma | [kernel.cu](../solutions/s20_mma/kernel.cu) | PTX mma.sync/ldmatrix、CTA 复用、显式 fragment 与 FP32 accumulation |
| s21_pipeline | [kernel.cu](../solutions/s21_pipeline/kernel.cu) | `cp.async`、缓冲回绕与完成等待 |
| s22_fa2 | [kernel.cu](../solutions/s22_fa2/kernel.cu)、[fa2_optimized.cuh](../solutions/s22_fa2/fa2_optimized.cuh) | `mma.sync`、`ldmatrix`、shared-memory swizzle，寄存器 softmax 与 P fragment |
| s30_tma | [kernel.cu](../solutions/s30_tma/kernel.cu) | TMA descriptor、事务 barrier 与数据可见性 |
| s31_tcgen | [kernel.cu](../solutions/s31_tcgen/kernel.cu)、[kernel_impl.cuh](../solutions/s31_tcgen/kernel_impl.cuh) | 单缓冲 TMA→MMA→等待、SMEM descriptor、TMEM lifetime 和输出交接 |
| s32_specialize | [kernel.cu](../solutions/s32_specialize/kernel.cu)、[共享设备实现](../solutions/s31_tcgen/kernel_impl.cuh) | 三缓冲 producer/consumer、TMA full 与 MMA empty barrier 的独立 phase 回绕 |
| s40_attention | [kernel.cu](../solutions/s40_attention/kernel.cu)、[sm100_attention.cuh](../solutions/s40_attention/sm100_attention.cuh) | TMA、tcgen05 QK/PV、TMEM 分数，寄存器统计与 O，顺序完成每个 key tile |
| s41_overlap | [kernel.cu](../solutions/s41_overlap/kernel.cu)、[共享设备实现](../solutions/s40_attention/sm100_attention.cuh) | KV 与 score 双缓冲，下一块 QK 与当前 softmax/correction 交错 |
| s42_numerics | [kernel.cu](../solutions/s42_numerics/kernel.cu)、[共享设备实现](../solutions/s40_attention/sm100_attention.cuh) | 默认 SFU+conditional rescale；混合多项式指数独立消融，分别研究误差和吞吐 |
| s43_schedule | [kernel.cu](../solutions/s43_schedule/kernel.cu)、[共享设备实现](../solutions/s40_attention/sm100_attention.cuh) | static persistent、跨 batch/head 的 causal 长任务优先、FP16/BF16 与尾部 |

s40–s43 共享一个完整 CUDA C++/PTX 实现，各课 `kernel.cu` 明确选择本课的编译配置。公共 tile 为 128 个 query、64 个 key，head dimension 为 64 或 128。TMA 直接描述 BNHD；QK 分数和 PV 更新使用 TMEM，online 行统计与跨 key tile 的 O 累加保存在 FP32 寄存器中。阅读共享代码时始终对照当前课的配置，区分硬件机制与尚未启用的优化。

## 怎样判断答案能否帮助优化

单个小输入能通过数值检查，只说明该输入计算正确。先通过整课 correctness cases、非默认 stream 与重复调用检查，再运行每个 `perf` case：

```bash
./lab task s22_fa2 -v
./lab check s22_fa2 --solution
./lab solution s22_fa2 --case 9
./lab solution s22_fa2 --case 10
./lab solution s22_fa2 --case 11
```

记录答案与优化库在相同输入、输出集合和计时模式下的延迟；阶段间比较使用共同支持的输入与 dtype。指出优化减少了什么工作或等待、增加了什么资源，以及哪些输入受益。不能将“代码较容易阅读”作为严重性能缺口的解释，也不能把某个 case 的收益外推到整课。维护答案时须定位影响主要 perf case 的缺口并改进实现，再提交可复现的结果。

prepare 完成描述符、输出/workspace 分配和编译，计时只覆盖准备后的执行。attention 输入/输出是 BNHD，LSE 是 BHN。CUDA Graph 与 eager 分别比较。验证方法见 [validation](validation.md)，计时边界见 [benchmarking](benchmarking.md)。

## 上游源码的用途

固定上游用于核对论文与生产实现的关系、指令语义及设计取舍。课程的可执行答案在 `solutions/`；阅读一个本地 kernel 不需要先理解整个上游框架。

| 上游 | 固定版本 | 源码与许可 |
|---|---|---|
| NVIDIA CUTLASS | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | [submodule](../third_party/cutlass/)、[BSD-3-Clause](../third_party/cutlass/LICENSE.txt) |
| Dao-AILab FlashAttention | beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | [submodule](../third_party/flash-attention/)、[BSD-3-Clause](../third_party/flash-attention/LICENSE) |

父仓库 gitlinks 固定上游版本，`bash scripts/bootstrap.sh` 完成初始化。算法与指令资料见 [sources](sources.md)。直接改编的代码须在文件与答案目录中保留来源和许可；独立实现也应明确设计所依据的论文或指令文档。
