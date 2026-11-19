# CUDA/PTX：SM100 完整 attention 基线

入口为 `entry.py → binding.cpp → kernel.cu → sm100_attention.cuh`，核心实现是本地 [sm100_attention.cuh](sm100_attention.cuh)。本课选择 `Overlap=false, Numerics=false, Persistent=false`；s41、s42 用同一设备实现研究执行交错与数值，s43 使用独立的两路实现。`entry.py` 负责编译，并通过 TVM FFI 零拷贝传入预分配张量；准备状态保留张量引用，数学计算由 CUDA/PTX 完成。

## 从 GEMM 连接到 attention

每 CTA 有四个 warp，处理 128 个 query，每次前进 64 个 key。每个 lane 拥有一个 query 行；QK 与当前 PV 增量存于 TMEM，行最大值、分母与跨 tile 的 O 累加在 FP32 寄存器。Q/K/V/P 为输入低精度，矩阵累加为 FP32。步骤依次是 TMA→QK→softmax/rescale→PV→O 更新，随后才装载下一块 KV。

Q/K/V 的 TMA map 按 `[D,H,N,B]` 描述物理 contiguous BNHD，无需全局转置。shared memory 将 D 拆成 64 列面板，地址为 `[D/64, rows, 64]`，每面板采用 128 字节 swizzle。`offset` 是 CUDA 线程访问同一布局的显式公式；1024 字节基址对齐使八行 swizzle atom 与 descriptor 一致。QK 的 A/B 都按 K-major 描述，PV 的 V 使用 MN-major 和相应 transpose 位。

TMEM 一次分配 256 列：score 使用列 0–63，PV 使用列 128 起的 D 列；s41 会用列 64–127 保存另一块 score。`tcgen05.ld.32x32b` 的地址高位选择 warp 对应的 32 行，每 lane 取回自己行的值。完整 warp 参与 allocation、load 和 deallocation，不能将 collective 指令放进仅一个 lane 的分支。

## 完成事件与生命周期

`qbar`、`kvbar` 由 TMA 完成承诺字节数；`scorebar`、`pvbar` 由对应 MMA 的 commit 推进。等待正确 phase 后再执行 `tcgen05.fence::after_thread_sync`，才进入相关读取。CUDA 写 P 后先执行 async proxy fence，再由 CTA barrier 汇合所有 P 生产者，MMA 才能消费。PV 完成后所有 warp 必须读完 TMEM，下一轮才能复用相关存储。每个 warp 的 `tcgen05.wait::ld` 仅处理读取完成；读取阶段末尾还要执行 `tcgen05.fence::before_thread_sync → CTA barrier → tcgen05.fence::after_thread_sync`，将所有消费者的完成顺序传递给下一次 MMA 的发起线程。

key 循环结束后复用 Q 的 shared memory 做连续向量 O 写回，同时输出自然对数 LSE。epilogue 中所有线程通过普通 CUDA 读写 Q 的 shared 空间，结束后各自执行 `fence.proxy.async.shared::cta` 再汇合，保证 persistent 下一任务的 TMA 覆盖排在这些 generic 访问之后。只有 TMA 发起线程执行 proxy fence 不足以覆盖其他线程的访问。所有消费者结束才释放 TMEM。描述符编码、编译和输出分配在 prepare 阶段完成；计时包括执行时的搬运、矩阵运算、softmax、同步和 O/LSE 写回。

## 共享数值路径的边界

s40–s42 的公共 scale 必须可表示为正且有限的 FP32。令 `scale2=scale*log2(e)`；scale、scale2 均为 normal 且 `scale2<=1` 时保留普通 FP32 路径，其行参考值使用 log2 单位。其余合法 scale 由 host 选择独立 specialization：最大值保留为 raw QK，先在 FP64 中计算分数差，再乘 scale 和 log2(e)，最后交给原有 exp2 实现；LSE 使用自然对数表达式 `double(raw_max)*scale+log(double(denominator))` 后转回 FP32。这样避免额外的 base-2 缩放溢出，以及微小差值在放大前丢失。

s42 在这条特殊路径中也先把 raw maximum 的变化转换为缩放后的 log2 差值，再判断延迟 rescale 的阈值；SFU/多项式开关保持独立。FP64 只用于这些缩放与 LSE 步骤，QK/PV 仍为 FP32 累加，P 仍舍入到输入 dtype，输出 LSE 仍为 FP32，因此不保证任意有限输入下的全域精度。QK 累加或数学 LSE 超出对应表示范围是独立限制。

## 支持域与性能实验

本课对外域是 BF16、D64/128、N 为 128 的倍数、noncausal。共享 kernel 的其他 dtype、mask 或尾部分支可用于底层诊断，但 s43 的独立实现通过扩展域验收不能替代本共享 kernel 的验证。代码可编译其他分支不表示本课接口自动接受它们。

```bash
./lab check s40_attention --solution
./lab solution s40_attention --case 6
./lab solution s40_attention --case 7
./lab solution s40_attention --case 8
./lab sanitize s40_attention --solution --tool memcheck --case 1
```

对照 s22 和 cuDNN 时使用相同 BF16 BNHD、O+LSE 和计时模式。单缓冲强制逐 key tile 等待，FP32 O 寄存器、score 读取、P 的 shared 写入和 CTA 同步都可能限制吞吐；需要测量确定主要差距。D64/D128 的动态 shared 存储分别是 48/80 KiB，TMEM 与寄存器还会影响驻留。引入 tcgen05 本身不能证明端到端更快。

算法依据 [FA1](https://arxiv.org/abs/2205.14135v2) 的 online softmax，指令依据 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)，同步分别对照 PTX 8.7 §9.7.16.5.4.4 的 tcgen05 跨线程顺序与 §9.7.9.25.2 的异步 copy 内存一致性。[固定 FA4 源码](../../third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py) 与 [FA4 论文](https://arxiv.org/abs/2603.05451v1) 用于比较生产设计，课程实现不要求与其 tile、warp 分工或流水线相同。
