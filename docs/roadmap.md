# 路线与源码验收

课程围绕完整算子推进。一个 lesson 可以先完成简单正确版本，再在同一入口的实验分支上优化；指令级试验属于该课的调试步骤，不单独包装成通关。对照实验的实现分别命名，放在个人实验目录或学生 helper 中。

| 阶段 | 课程 | 前置 | 主要设计证据 |
|---|---|---|
| 0 | [s00_layout · 内存、线程与转置](tasks/s00_layout.md) | 无 | 说明 ownership、全局访存方向及 shared-memory bank 行为；提供至少两种映射的实测对比。 |
| 0 | [s01_softmax · 行归约与稳定 softmax](tasks/s01_softmax.md) | s00_layout | 给出最大值、分母的参与集合及有效线程边界；解释短行和宽行的资源差异。 |
| 1 | [s10_gemm · SIMT 分块 GEMM](tasks/s10_gemm.md) | s00_layout | 从源码说明复用层次、线程输出 ownership、尾部和寄存器开销。 |
| 1 | [s11_materialized · 显式中间矩阵 attention](tasks/s11_materialized.md) | s01_softmax, s10_gemm | 画出 QK、P、PV 的数据流，列出临时内存与每一步的 dtype。 |
| 1 | [s12_online · 流式 attention 与 IO](tasks/s12_online.md) | s11_materialized | 证明已处理 key 集合的统计含义，分析遍历次序与 O 状态存储；不能只以数值通过命名 FA1。 |
| 2 | [s20_mma · Warp Tensor Core GEMM](tasks/s20_mma.md) | s10_gemm | 解释 fragment/ldmatrix/SMEM 的对应；用生成指令确认 Tensor Core 执行。 |
| 2 | [s21_pipeline · 异步搬运的 Tensor Core GEMM](tasks/s21_pipeline.md) | s20_mma | 给出 producer/consumer 生命周期与完成事件；检验 prologue、稳态、drain。 |
| 2 | [s22_fa2 · FA2 并行与 Tensor Core 融合](tasks/s22_fa2.md) | s12_online, s21_pipeline | 核对 query grid、CTA 内输出 ownership、跨 warp 通信及非矩阵工作；标明 P 的精度转换。 |
| 3 | [s30_tma · TMA 与生产消费协议](tasks/s30_tma.md) | s21_pipeline | 描述事务完成与数据可见性的区别；追踪至少一次 buffer/phase 回绕。 |
| 3 | [s31_tcgen · Blackwell tcgen05 与 TMEM](tasks/s31_tcgen.md) | s20_mma, s30_tma | 说明 SMEM descriptor、TMEM lifetime、发布/等待和输出交接；检查实际 tcgen05 指令。 |
| 3 | [s32_specialize · Warp specialization 流水线](tasks/s32_specialize.md) | s21_pipeline, s31_tcgen | 用状态图证明所有缓冲读写与复用有顺序，短 K/长 K 均能 drain。 |
| 4 | [s40_attention · SM100 attention 基线](tasks/s40_attention.md) | s22_fa2, s32_specialize | 标明 S、P、O 及 LSE 的存储位置、精度与生命周期；不得包装官方 attention。 |
| 4 | [s41_overlap · Matmul、softmax 与 correction overlap](tasks/s41_overlap.md) | s40_attention | 给出跨 tile 的就绪条件与不会覆盖活跃数据的证据；用时间线验证 overlap。 |
| 4 | [s42_numerics · Softmax 吞吐与数值预算](tasks/s42_numerics.md) | s41_overlap | 区分指数近似误差、概率舍入与状态 rescale；保持 LSE 自然对数约定。 |
| 4 | [s43_schedule · 边界、调度与性能结业实验](tasks/s43_schedule.md) | s42_numerics | 提供完整正确性、内存安全、设计证据与同机参考性能，分别说明吞吐和尾延迟。 |

每课运行 `./lab task ID -v` 查看测试的序号与意图。数值验收之后按上表检查设计证据：FA1 的 IO 与遍历组织，FA2 的 query 并行、CTA 分工及非矩阵操作，SM100 attention 的异步 pipeline、softmax 与调度。

阶段 0 从矩形转置开始解释线程和地址，再在可变宽度 softmax 中引入归约。阶段 1 先学 GEMM 再组合 attention，避免在不会数据复用时直接要求实现完整 fused kernel。阶段 2 先单独验证 Tensor Core 和搬运，随后融合。阶段 3 使用相同数学工作负载理解 Blackwell 新资源；阶段 4 在已有完整源码上研究依赖、误差与调度。

## 阶段 4 的源码阅读与消融

阶段 4 共享 [sm100_attention.cuh](../solutions/s40_attention/sm100_attention.cuh) 中的完整 CUDA C++/PTX attention。s40 顺序执行每个 key tile 的 QK、softmax 和 PV；s41 用双缓冲提前发起下一块 QK；s42 研究指数与 conditional rescale；s43 增加 persistent 调度、causal 和尾部。每课先阅读对应 `kernel.cu`，再跟踪本地共享 helper 中的设备代码；所有入口的调用链都终止于本地 CUDA kernel。实验必须说明相对前一课具体改变了哪些代码、哪条依赖以及哪些资源，源码入口见 [参考实现](reference-implementations.md)。

课程不扩展 backward、varlen、paged attention、GQA、FP8/FP4。用于对照阅读的上游源码包含这些模块，它们不属于课程实现与验收范围。
