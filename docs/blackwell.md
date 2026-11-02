# Blackwell 编程与参考源码

学生与参考答案都使用 CUDA C++/PTX。s20/s21 用 warp MMA 和异步 copy 建立布局与生命周期概念；s30/s31/s32 引入 TMA、mbarrier、tcgen05 和 TMEM。阶段 4 将这些机制组合为本地完整 attention，并继续研究依赖、数值与调度。实际设备源码入口见 [参考答案](reference-implementations.md)。

TMA 描述符指定多维全局内存布局，SMEM descriptor 描述矩阵引擎输入，TMEM 保存 tcgen05 accumulator。三者的地址空间、合法域和生命周期分别定义。需要证明发起、完成通知、跨代理可见性与 buffer 复用的顺序，并用完整 GEMM 的多 CTA、长 K、短 K 和边界案例检验。

阅读本地 CUDA kernel 时，对每个内联 PTX helper 标明调用者、参与线程、输入/输出空间与完成事件。先确认地址和 descriptor 编码，再确认异步操作完成后谁可以读、谁可以覆盖。执行一次 MMA 只能验证一小段数据通路；长循环、回绕和非默认 stream 才能检验完整协议。

`third_party/cutlass` 中的 `include/cute/arch/` 与 Blackwell 教程提供指令封装和生产布局的对照；`third_party/flash-attention/flash_attn/cute` 提供 attention pipeline、softmax 和调度的生产设计对照。它们通过 submodule 保留完整源码与许可证。课程答案的矩阵运算、softmax 和流水线直接在本地 CUDA C++/PTX 中实现。

指令合法域查 NVIDIA PTX ISA，资源限制查 Blackwell tuning guide，设计比较对照 gitlink 固定源码。在线手册可能比本机 Toolkit 新，编译与实测应使用所选工具链支持的 target。流水线效果通过生命周期审查、时间线和相同输入的计时共同验证。

完成等待、线程汇合与跨代理顺序需要分别证明。`tcgen05.wait::ld` 确认 TMEM 读取完成；让其他线程接续覆写或释放 TMEM，还需 `tcgen05.fence::before_thread_sync → CTA barrier → tcgen05.fence::after_thread_sync`。普通 CUDA 对 shared 的读写之后若要交给 TMA 覆盖，每个相关线程先执行 `fence.proxy.async.shared::cta` 再汇合，发起线程单独执行 fence 不能代替其他访问者。分别参照 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 8.7 的 §9.7.16.5.4.4、§9.7.9.25.2，并沿 s30、s31/s32、s40–s43 的缓冲复用路径检查这些义务。
