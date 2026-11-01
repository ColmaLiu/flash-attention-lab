# Blackwell 学习边界

主线是 CUDA C++ + PTX。s20/s21 先使用 warp MMA 和异步 copy，建立可迁移的布局与生命周期概念；s30/s31/s32 再引入 TMA、mbarrier、tcgen05 和 TMEM。不要把语言迁移作为硬件知识的前置关卡。

TMA 描述符指定多维全局内存布局；SMEM descriptor 描述矩阵引擎的输入；TMEM 用于 tcgen05 accumulator。三者具有不同地址空间、合法域和生命周期。发起、完成通知、跨代理可见性与 buffer 复用需要分别证明。完整 GEMM 的多 CTA、长 K、短 K 和尾部检验比只跑一次 MMA 更有意义。

固定 CUTLASS 源码保存在 `references/cutlass-sm100`。从 `examples/cute/tutorial/blackwell/01_mma_sm100.cu` 阅读 TMEM/MMA，再看 `02_mma_tma_sm100.cu` 的 TMA 与 barrier；生产式流水线还需阅读 `include/cutlass/gemm/collective/` 与 `include/cutlass/pipeline/` 的 SM100 实现。答案集成范围见 `references/README.md`。

阶段 4 使用固定官方 FA4 forward 源码阅读真实 pipeline、softmax 与 scheduling。2-CTA backward 的 atomic 优化属于原论文的 backward 内容，不作为本课程 forward 的要求。完整源码包含的其他功能不自动扩展学生契约。

权威约束查 NVIDIA PTX ISA、Blackwell tuning guide 与固定源码；在线手册可能比本机 Toolkit 新。先确认指令所需 target，再编译与实测。仅看到新指令不能证明流水线有效，必须结合生命周期审查及时间线。
