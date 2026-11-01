# Blackwell 编程与参考源码

学生主线使用 CUDA C++/PTX。s20/s21 用 warp MMA 和异步 copy 建立布局与生命周期概念；s30/s31/s32 引入 TMA、mbarrier、tcgen05 和 TMEM。s22/s31/s32 的完整答案使用 C++ CuTe，阶段 4 使用上游 CuTe DSL 生产源码；这些语言各自表达的硬件操作可沿 source map 对照阅读。

TMA 描述符指定多维全局内存布局，SMEM descriptor 描述矩阵引擎输入，TMEM 保存 tcgen05 accumulator。三者的地址空间、合法域和生命周期分别定义。需要证明发起、完成通知、跨代理可见性与 buffer 复用的顺序，并用完整 GEMM 的多 CTA、长 K、短 K 和边界案例检验。

CUTLASS submodule 位于 `third_party/cutlass`。从 `examples/cute/tutorial/blackwell/01_mma_sm100.cu` 阅读 TMEM/MMA，再看 `02_mma_tma_sm100.cu` 的 TMA 与 barrier，随后阅读 `include/cutlass/gemm/collective/` 与 `include/cutlass/pipeline/` 的 SM100 实现。[参考实现](reference-implementations.md) 列出可执行入口与具体模块。

阶段 4 的 FA4 源码位于 `third_party/flash-attention/flash_attn/cute`，学习内容覆盖 forward pipeline、softmax 与 scheduling。课程采用 forward、contiguous self-attention；原论文中的 2-CTA backward 与 atomic 优化留作后续主题。

指令合法域查 NVIDIA PTX ISA，资源限制查 Blackwell tuning guide，实际调用对照 gitlink 固定源码。在线手册可能比本机 Toolkit 新，编译与实测应使用所选工具链支持的 target。流水线效果通过生命周期审查、时间线和双边计时共同验证。
