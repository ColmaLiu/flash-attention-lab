# CUDA/PTX：生产者与 MMA 消费者的三缓冲 GEMM

本关 `kernel.cu` 选择 `LAB_MULTISTAGE=1`，复用 [完整 CUDA/PTX 实现](../s31_tcgen/kernel_impl.cuh)。输入/输出、128×128×64 tile、BF16 MMA、TMEM accumulator、边界和 epilogue 与 s31 相同。可直接阅读并修改 producer/consumer 循环，无 CUTLASS/CuTe 或库 GEMM 调用。

## 每个 slot 的所有权

CTA 有五个 warp。第 5 个 warp 的一个线程负责 TMA；第 1 个 warp 的一个线程负责发起 MMA；前四个 warp 在最终 epilogue 中读取 TMEM。没有“发起异步指令就算完成”的假设：

- `full[s]` 是 TMA 完成 barrier。producer 对 A 与两个 B 面板的总字节数执行 expect_tx，MMA 线程等待后才读取它们。
- `empty[s]` 是 MMA 完成 barrier。MMA 线程提交当前 slot 的最后一条矩阵指令后 commit；producer 等待这一通知才复用相同 slot。
- 两条循环各自使用 `i % 3` 选 slot、`i / 3` 选择 phase。第一轮 producer 不等待 empty，因为该缓冲没有活跃读者；以后等待前一轮的 phase。

producer 最多领先三块 K。消费者在上一块 MMA 执行时可以等待下一块 TMA；两者独立推进，无逐 tile 的 CTA barrier。首轮、短 K、回绕和最后一次 MMA 的 drain 都走同一协议。最后一次 completion 之后才允许 epilogue 覆盖 operand shared memory。s31 的串行循环保留为对照。epilogue 的 TMEM load 各自完成 `wait::ld` 后，全体线程执行 `before_thread_sync` fence、CTA barrier 和 `after_thread_sync` fence，再完成输出与 TMEM 释放；完成等待和跨线程交接是两个不同义务。

## 实验重点

三缓冲增加 shared memory 占用到 96 KiB；多一个 producer warp 也消耗线程资源。长 K 工作负载通常更能摊薄 prologue/epilogue，但小矩阵、短 K 或低 tile 数仍可能由 launch、输出或占用率限制。应在 `square`、`qk_shape`、`pv_shape` 三种 perf case 上分别比较 s31、s32 与 cuBLAS，记录 graph/eager 同计时边界的差距；增加缓冲并不保证每个案例更快。

使用独立数学 oracle 验证数值，再运行 memcheck/racecheck/synccheck。`long_k` 与 `rectangular` 分别验证多次 phase 回绕和三方向边界。通过检查全部数值只是必要条件；应在源码中指出每次 TMA 写入、MMA 读取、缓冲再利用的完整 happens-before 链。

指令来源与完整数据布局说明见 [s31 README](../s31_tcgen/README.md) 及 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)。
