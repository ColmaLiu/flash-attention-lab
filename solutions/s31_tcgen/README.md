# CUDA/PTX：TMA、tcgen05 与 TMEM 的完整 GEMM

入口是 `entry.py → binding.cpp → kernel.cu → kernel_impl.cuh`。答案自行实现设备算法，只包含 CUDA headers，不调用 CUTLASS/CuTe collective 或库 GEMM。输入 BF16 `A[M,K]`、`B[K,N]`，输出 FP32 `C[M,N]`；支持课程声明的 M%64=0、N%8=0、K%16=0，不要求整个矩阵恰好覆盖 CTA tile。

## 从数据到指令

每个 CTA 计算 128×128 的输出，按 64 个 K 元素推进。TMA 将 A 的 128×64 tile 和 B 的两个 64×64 面板搬到 SW128 shared memory。B 在 shared memory 中按 `[N/64,K,64]` 放置，符合 MN-major MMA descriptor；全局 B 仍是普通 `[K,N]`，没有额外的全局转置。

`descriptor` 直接编码地址、leading/stride byte offsets、version 和 swizzle 位。`multiply` 直接编码 FP32 accumulator、BF16 两输入、M/N 和 B transpose 位；四条 K=16 的 `tcgen05.mma` 完成一个 K tile。首条 MMA 清零累加器，后续 MMA 保持累加。请先解释每个字段，再修改 tile。

本课只有一个 operand slot，明确执行以下顺序：

1. TMA 对 full barrier 承诺字节数，搬运完成后解除等待。
2. 单线程发起四条异步 MMA，commit 到 empty barrier。
3. 等待 MMA 完成后才能覆盖 operands；同一 barrier 的 phase 每轮翻转。
4. 最后一次 MMA 完成并经过 CTA 同步后，四个 warp 用 `tcgen05.ld` 读取 TMEM。每线程负责一行，暂存至带 padding 的 shared tile，然后以连续 float4 写回全局内存。
5. 每次 TMEM load 后以 `tcgen05.wait::ld` 确认本线程读取完成。读取阶段结束后，全体线程按 `tcgen05.fence::before_thread_sync → __syncthreads → tcgen05.fence::after_thread_sync` 交接；随后完成输出写回并汇合，分配 TMEM 的完整 warp 才能释放它。

128-column TMEM allocation、异步指令、barrier 与 SMEM 生命周期均在源码中显式可见。TMA tensor maps 在 `prepare` 中编码并作为 kernel 参数传递；每次计时调用包含所有输入搬运、计算及输出，没有隐藏预处理结果或临时全局 workspace。

## 边界与性能

TMA 对越界行/列填零；输出 store 按 M/N 判断有效性。N 是 8 的倍数，因此 float4 store 不会跨越有效尾部。K 尾部也由 TMA 填零。测试中的 `(192,136,80)` 同时检验三个方向的边界。

这是建立资源协议的完整 Tensor Core 基线。每个 K tile 等待搬运和计算的串行依赖限制吞吐；s32 在相同 tile、算术和 epilogue 上加入 producer warp 与三个缓冲，使这一开销可直接测量。小矩阵受 launch/低占用限制，短 K 矩阵的输出流量可能主导；不要根据一个尺寸推广速度结论。用 `./lab solution s31_tcgen --case 5` 等 perf 案例比较独立 cuBLAS 基线，graph 与 eager 分别解释。

## 阅读依据

指令语义与位字段来自 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)。可同时对照固定 submodule 中 `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp`、`mma_sm100_umma.hpp`、`copy_sm100.hpp` 的指令定义；它们是源码阅读资料，本答案不编译或调用这些实现。PTX 文档说明 `tcgen05` 与普通线程指令的完成语义，单纯 `__syncthreads()` 不能替代 MMA 的完成通知；`wait::ld` 也不能独自建立跨线程的 TMEM 复用/释放顺序。后者需结合 PTX 8.7 §9.7.16.5.4.4 的 before/after-thread-sync fence 协议核对。
