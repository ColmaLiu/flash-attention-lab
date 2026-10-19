# 全阶段路线与验收索引

课程覆盖阶段 0–4，核心设备端计算全部留给学生。顺序按下表：阶段 0 建立 FP32 数值与流程；阶段 1 FA1 IO；阶段 2 FA2 并行、sliced-Q、Tensor Core；阶段 3 DSL/异步/SM100；阶段 4 逐变量加入 FA4 forward 设计。

学习依赖是 DAG；s22 可在进入 Tensor Core 之前独立学习，s30 可独立启动 DSL，s36 在 TMA 之后并非强依赖，但进入 s37 前需要同时完成 TMA 与 TMEM。运行独立入口不要求先实现所有前置，只有明确声明的代码复用是 runtime dependency。

| 阶段 | 任务 | 核心入口 | 学习前置 | 全关验收 |
|---|---|---|---|---|
| 0 | [s00_softmax](tasks/s00_softmax.md) · 固定契约稳定 row-wise softmax | [学生文件](../students/cuda/s00_softmax/kernel.cu) | 无 | `./lab check s00_softmax` |
| 0 | [s01_reduce](tasks/s01_reduce.md) · 独立 reduction 与尾部元素 | [学生文件](../students/cuda/s01_reduce/kernel.cu) | s00_softmax | `./lab check s01_reduce` |
| 0 | [s02_naive](tasks/s02_naive.md) · 朴素 attention forward | [学生文件](../students/cuda/s02_naive/kernel.cu) | s00_softmax, s01_reduce | `./lab check s02_naive` |
| 0 | [s03_online](tasks/s03_online.md) · online softmax 与状态不变量 | [学生文件](../students/cuda/s03_online/kernel.cu) | s00_softmax, s01_reduce | `./lab check s03_online` |
| 1 | [s10_fa1](tasks/s10_fa1.md) · FA1：KV 外层循环与 IO 复用 | [学生文件](../students/cuda/s10_fa1/kernel.cu) | s02_naive, s03_online | `./lab check s10_fa1` |
| 1 | [s11_fa1_edges](tasks/s11_fa1_edges.md) · FA1：尾块与 causal 扩展 | [学生文件](../students/cuda/s11_fa1_edges/kernel.cu) | s10_fa1 | `./lab check s11_fa1_edges` |
| 2 | [s20_fa2_grid](tasks/s20_fa2_grid.md) · FA2 第一变量：query 维 grid 并行 | [学生文件](../students/cuda/s20_fa2_grid/kernel.cu) | s10_fa1 | `./lab check s20_fa2_grid` |
| 2 | [s21_fa2_sliced](tasks/s21_fa2_sliced.md) · FA2：sliced-Q 与非矩阵运算 | [学生文件](../students/cuda/s21_fa2_sliced/kernel.cu) | s20_fa2_grid | `./lab check s21_fa2_sliced` |
| 2 | [s22_mma](tasks/s22_mma.md) · CUDA Tensor Core 与大矩阵 GEMM | [学生文件](../students/cuda/s22_mma/kernel.cu) | s01_reduce | `./lab check s22_mma` |
| 2 | [s23_fa2_tensor](tasks/s23_fa2_tensor.md) · FA2 接入 Tensor Core | [学生文件](../students/cuda/s23_fa2_tensor/kernel.cu) | s21_fa2_sliced, s22_mma | `./lab check s23_fa2_tensor` |
| 2 | [s24_fa2_edges](tasks/s24_fa2_edges.md) · FA2 低精度边界验证 | [学生文件](../students/cuda/s24_fa2_edges/kernel.cu) | s23_fa2_tensor | `./lab check s24_fa2_edges` |
| 3 | [s30_layout](tasks/s30_layout.md) · CuTe DSL layout 与线程映射 | [学生文件](../students/cute/s30_layout/entry.py) | 无 | `./lab check s30_layout` |
| 3 | [s31_port](tasks/s31_port.md) · CUDA → CuTe DSL 迁移 | [学生文件](../students/cute/s31_port/entry.py) | s00_softmax, s30_layout | `./lab check s31_port` |
| 3 | [s32_async](tasks/s32_async.md) · 异步搬运与完成可见性 | [学生文件](../students/cute/s32_async/entry.py) | s30_layout | `./lab check s32_async` |
| 3 | [s33_tma](tasks/s33_tma.md) · TMA 与 mbarrier 单缓冲 | [学生文件](../students/cute/s33_tma/entry.py) | s32_async | `./lab check s33_tma` |
| 3 | [s34_warp_specialize](tasks/s34_warp_specialize.md) · warp specialization 生产消费 | [学生文件](../students/cute/s34_warp_specialize/entry.py) | s33_tma | `./lab check s34_warp_specialize` |
| 3 | [s35_pipeline](tasks/s35_pipeline.md) · 多缓冲流水线与 drain | [学生文件](../students/cute/s35_pipeline/entry.py) | s34_warp_specialize | `./lab check s35_pipeline` |
| 3 | [s36_tmem](tasks/s36_tmem.md) · TMEM 分配、访问与释放 | [学生文件](../students/cute/s36_tmem/entry.py) | s30_layout, s32_async | `./lab check s36_tmem` |
| 3 | [s37_tcgen05](tasks/s37_tcgen05.md) · tcgen05 单次异步 MMA | [学生文件](../students/cute/s37_tcgen05/entry.py) | s33_tma, s36_tmem | `./lab check s37_tcgen05` |
| 3 | [s38_async_gemm](tasks/s38_async_gemm.md) · TMA + tcgen05 流水线 GEMM | [学生文件](../students/cute/s38_async_gemm/entry.py) | s35_pipeline, s37_tcgen05 | `./lab check s38_async_gemm` |
| 4 | [s40_fa4_base](tasks/s40_fa4_base.md) · FA4 基础：受限 SM100 forward | [学生文件](../students/cute/s40_fa4_base/entry.py) | s31_port, s38_async_gemm | `./lab check s40_fa4_base` |
| 4 | [s41_fa4_pipeline](tasks/s41_fa4_pipeline.md) · FA4：matmul/softmax/correction overlap | [学生文件](../students/cute/s41_fa4_pipeline/entry.py) | s40_fa4_base | `./lab check s41_fa4_pipeline` |
| 4 | [s42_exp](tasks/s42_exp.md) · FA4：独立指数近似误差实验 | [学生文件](../students/cute/s42_exp/entry.py) | s31_port | `./lab check s42_exp` |
| 4 | [s43_fa4_exp](tasks/s43_fa4_exp.md) · FA4：指数吞吐优化接入 | [学生文件](../students/cute/s43_fa4_exp/entry.py) | s41_fa4_pipeline, s42_exp | `./lab check s43_fa4_exp` |
| 4 | [s44_fa4_rescale](tasks/s44_fa4_rescale.md) · FA4：条件重缩放与数值范围 | [学生文件](../students/cute/s44_fa4_rescale/entry.py) | s43_fa4_exp | `./lab check s44_fa4_rescale` |
| 4 | [s45_fa4_edges](tasks/s45_fa4_edges.md) · FA4：尾部、causal 与精度扩展 | [学生文件](../students/cute/s45_fa4_edges/entry.py) | s44_fa4_rescale | `./lab check s45_fa4_edges` |
| 4 | [s46_fa4_schedule](tasks/s46_fa4_schedule.md) · FA4：调度与资源控制实验 | [学生文件](../students/cute/s46_fa4_schedule/entry.py) | s45_fa4_edges | `./lab check s46_fa4_schedule` |

阶段独立检查：`./lab doctor --stage 0`、`./lab smoke --stage 0`、`./lab status --stage 0`、`./lab list --stage 0`，把数字替换为 1–4。每阶段每个入口都有相同 check/bench/profile/sanitize 命令；未实现必须非零退出，不会 fallback。

## 版本命名的验收层次

| 称谓 | 本仓库要求 | 不据此声称 |
|---|---|---|
| 算法基础 | s00–s03 稳定 softmax、矩阵定义、online 不变量 | online softmax 单独不叫 FA1 |
| FA1 教学结构 | s10 的 KV 外层、分块复用、明确 O/m/l 存储更新与 IO 推导 | 不代表历史官方功能或性能 |
| FA2 关键结构 | s20 query grid + s21 CTA 内 sliced-Q + 非矩阵操作分析；s23 再加入 Tensor Core | 仅增加 CTA 数不叫完成 FA2 |
| SM100 forward 基线 | s40 的合法 TMA/tcgen05/TMEM 与稳定 attention | 在 B200 上运行不叫完成 FA4 |
| 受限 FA4 forward 学习版 | s41 overlap/correction、s42–43 指数、s44 conditional rescale、s45 边界及 s46 调度证据 | 不等于官方全面工程实现或性能 |

每关通关都需数值 + 内存安全 + 设计证据。机器状态记录只表示测试结果，`experiments/` 中另记录设计 review。Backward、dropout、varlen、GQA、paged attention 只作为未来方向，没有为其搭建空框架。

## FA4 论文设计 → 硬件 → 练习 → 验证

| 论文设计 | 硬件机制 | 练习 | 验证与简化 |
|---|---|---|---|
| 异步 forward pipeline | TMA、tcgen05、TMEM、warp specialization | s32–s38，s40–s41 | 数据生命周期、短循环/wrap、数值、sanitizer、时间线；先单 CTA 与受限整块 |
| softmax 与 MMA/correction 解耦 | TMEM S/P/O 交接、不同执行引擎 | s41 | 依赖图与实际 overlap；不预给 buffer layout |
| 指数吞吐优化 | MUFU 与 FP32 FMA 资源 | s42–s43 | 独立指数误差、相同 attention 输入消融；系数由学生设计 |
| 条件重缩放 | FP32 状态与低精度 P 动态范围 | s44–s45 | 递增最大值、阈值单位、BF16/FP16 精度与重缩放计数 |
| 调度与资源分配 | CTA ownership、register/SMEM/TMEM 容量 | s46 | 两种配置同 shape 对比，尾部负载与 spills |
| 2-CTA backward/atomic 优化 | cluster 与协作 MMA | 课程范围外 | forward 学习版不要求实现，避免把 backward 设计冒充 forward 必需 |
