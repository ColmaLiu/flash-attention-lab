# 术语表

按主题分组。“首次出现”指第一次讲解这个概念的关卡；

## 线程与执行

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| kernel | kernel | 由 host 发起、在 GPU 上被许多线程同时执行的 `__global__` 函数 | [从零开始](starting-from-zero.md) |
| host | host | 发起 GPU 工作的 CPU 代码 | [从零开始](starting-from-zero.md) |
| 线程块 / CTA | thread block / Cooperative Thread Array | 一组共享 shared memory、可以用 `__syncthreads()` 同步的线程。CTA 是 PTX 中的叫法 | [00](../course/00_transpose/README.md) |
| grid | grid | 一次 launch 的全部 block | [00](../course/00_transpose/README.md) |
| warp | warp | 32 个线程一组，一起执行同一条指令 | [00](../course/00_transpose/README.md) |
| lane | lane | 线程在 warp 内的编号 0–31。注意 TMEM 的 “lane” 是另一个意思（见下） | [00](../course/00_transpose/README.md) |
| warpgroup | warpgroup | 连续 4 个 warp（128 个线程），起始 warp 编号是 4 的倍数 | [32](../course/32_ws_gemm/README.md) |
| SM | Streaming Multiprocessor | 执行 warp 的硬件单元，B200 有 148 个 | [00](../course/00_transpose/README.md) |
| SMSP | SM sub-partition | SM 的四分之一，有自己的 warp 调度器 | [20](../course/20_mma_gemm/README.md) |
| 驻留 / 占用率 | residency / occupancy | 一个 SM 上同时存在的 warp 数（占最多 64 个的比例），受寄存器、shared memory、线程数限制 | [00](../course/00_transpose/README.md) |
| 波 / 轮 | wave | grid 中的 block 数除以能同时驻留的 block 数。最后一轮不满时有 SM 空闲 | [10](../course/10_simt_gemm/README.md) |
| stream | stream | GPU 工作队列。同一个 stream 内按顺序执行 | [从零开始](starting-from-zero.md) |
| CUDA Graph | CUDA Graph | 把一串 launch 录制下来、整体回放，去掉逐次 launch 的开销 | [工具指南](tools.md) |
| persistent kernel | persistent kernel | grid 只有“能驻留的 block 数”那么大，每个 block 循环处理多个 tile | [30](../course/30_tma_mbarrier/README.md) |
| warp specialization | warp specialization | 同一个 block 中不同 warp 担任不同角色（TMA、MMA、epilogue、softmax） | [32](../course/32_ws_gemm/README.md) |
| 生产者 / 消费者 | producer / consumer | 往 buffer 里放数据的一方 / 取数据的一方 | [30](../course/30_tma_mbarrier/README.md) |
| `elect.sync` | elect | warp 中所有活跃线程一起执行，恰好选出一个线程 | [30](../course/30_tma_mbarrier/README.md) |
| `setmaxnreg` | — | 运行时在 warpgroup 之间重新分配寄存器 | [32](../course/32_ws_gemm/README.md)（介绍）、40 |
| LPT 调度 | Longest Processing Time first | 先发射工作量最大的块，减少最后一轮的空闲 | [43](../course/43_fa4_schedule/README.md) |

## 存储

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| global memory / HBM | global memory / High Bandwidth Memory | 所有线程可访问的显存。B200 是 HBM3e，标称 8 TB/s | [00](../course/00_transpose/README.md) |
| shared memory（smem） | shared memory | 同一 block 的线程共享的片上存储，每 SM 228 KB | [00](../course/00_transpose/README.md) |
| 寄存器 | register | 每个线程私有，每 SM 共 65536 个 32 位寄存器 | [01](../course/01_softmax/README.md) |
| local memory / spill | local memory / register spill | 寄存器放不下时，编译器把变量放进 local memory（实际在显存中） | [01](../course/01_softmax/README.md) |
| L2 | L2 cache | 全芯片共享的缓存，B200 为 126 MB | [00](../course/00_transpose/README.md) |
| sector | sector | 内存在 L1/L2/HBM 之间移动的最小单位，32 字节 | [00](../course/00_transpose/README.md) |
| 合并访存 | coalescing | 一个 warp 的访问落在连续地址上，只需要很少的 sector | [00](../course/00_transpose/README.md) |
| bank / bank conflict | bank / bank conflict | shared memory 分成 32 个 bank；一个 warp 的多个线程访问同一 bank 的不同地址时要排队 | [00](../course/00_transpose/README.md) |
| padding | padding | 给 shared memory 数组每行多留几个元素，改变行间距来避开 bank conflict | [00](../course/00_transpose/README.md) |
| swizzle | swizzle | 用 XOR 打乱 shared memory 中 16 字节块的位置，避开 bank conflict | [20](../course/20_mma_gemm/README.md)、[30](../course/30_tma_mbarrier/README.md) |
| 在途访存 | bytes in flight | 已发出、尚未返回的访存量。带宽 = 在途字节数 / 延迟（Little 定律） | [00](../course/00_transpose/README.md) |
| contiguous | contiguous | 张量按行主序紧密排列，最后一维连续 | [00](../course/00_transpose/README.md) |
| K-major / MN-major | K-major / MN-major | 矩阵操作数中连续的那一维是 K（归约维）还是 M/N | [10](../course/10_simt_gemm/README.md)、[34](../course/34_fa3/README.md) |
| scratch | scratch | `make_plan` 中分配的临时 global buffer（`lab::Scratch`），初始内容是垃圾值 | [11](../course/11_naive_attention/README.md) |
| TMEM | Tensor Memory | 每 SM 128 lane × 512 列 × 32 bit 的存储，放 `tcgen05.mma` 的累加器 | [31](../course/31_tcgen05_gemm/README.md) |
| TMEM lane / column | lane / column | TMEM 的行 / 列。warp w 只能访问 lane `32·(w%4)` 起的 32 行 | [31](../course/31_tcgen05_gemm/README.md) |
| DSMEM | distributed shared memory | cluster 内的 CTA 访问彼此的 shared memory | [33](../course/33_2cta_gemm/README.md) |

## 异步搬运与同步

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| `__syncthreads()` | CTA barrier | block 内所有线程到齐后才继续 | [00](../course/00_transpose/README.md) |
| warp shuffle | warp shuffle | warp 内线程之间直接交换寄存器的值（`__shfl_xor_sync`） | [01](../course/01_softmax/README.md) |
| `cp.async` | asynchronous copy | global → shared 的异步拷贝，不经过寄存器 | [21](../course/21_pipeline_gemm/README.md) |
| commit group / wait_group | — | 把已发出的 `cp.async` 打包成组；等待到最多 N 组未完成 | [21](../course/21_pipeline_gemm/README.md) |
| 流水 / 多级 buffer | pipeline / multi-stage buffer | 用多块 buffer 轮流装数据，让搬运和计算重叠 | [21](../course/21_pipeline_gemm/README.md) |
| 环形 buffer | ring buffer | S 块 buffer 按 0, 1, …, S−1, 0, … 循环使用 | [21](../course/21_pipeline_gemm/README.md) |
| TMA | Tensor Memory Accelerator | 一个线程发起、硬件整块搬运多维数据的单元 | [30](../course/30_tma_mbarrier/README.md) |
| tensor map | tensor map / `CUtensorMap` | 128 字节的 TMA 描述符，在 host 上编码 | [30](../course/30_tma_mbarrier/README.md) |
| box | box | 一次 TMA 搬运的数据块 | [30](../course/30_tma_mbarrier/README.md) |
| mbarrier | mbarrier | shared memory 中的 8 字节同步对象，记录 phase、待到达次数、待完成字节数 | [30](../course/30_tma_mbarrier/README.md) |
| phase / parity | phase / parity | mbarrier 每完成一轮，phase 加 1；等待者用 phase 的奇偶（parity）判断是否完成 | [30](../course/30_tma_mbarrier/README.md) |
| expect_tx / complete_tx | transaction count | 告诉 mbarrier 还要等多少字节 / TMA 写完后扣除这些字节 | [30](../course/30_tma_mbarrier/README.md) |
| full / empty barrier | — | 环形 buffer 中“数据已装好”/“buffer 已用完可以覆盖”的两组 mbarrier | [30](../course/30_tma_mbarrier/README.md) |
| proxy / proxy fence | proxy / `fence.proxy.async` | 普通 ld/st 属于 generic proxy，TMA 与 tcgen05 属于 async proxy；两者交接 shared memory 时要 fence | [30](../course/30_tma_mbarrier/README.md) |
| cluster | thread block cluster | 一组保证同时运行、在同一 GPC 上的 CTA | [33](../course/33_2cta_gemm/README.md) |
| multicast | TMA multicast | 一次 TMA 读取写进 cluster 中多个 CTA 的 shared memory | [33](../course/33_2cta_gemm/README.md) |

## Tensor Core

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| Tensor Core | Tensor Core | 专门做小矩阵乘加的硬件单元 | [20](../course/20_mma_gemm/README.md) |
| `mma.sync` | warp-level MMA | 一个 warp 同步完成一次 16×8×16 的矩阵乘加，操作数和累加器都在寄存器 | [20](../course/20_mma_gemm/README.md) |
| fragment | fragment | `mma.sync` 的一个操作数分散在 32 个线程寄存器中的那一份 | [20](../course/20_mma_gemm/README.md) |
| `ldmatrix` | ldmatrix | 从 shared memory 按 fragment 布局装 8×8 矩阵 | [20](../course/20_mma_gemm/README.md) |
| `tcgen05` | 5th-generation Tensor Core | Blackwell 的 Tensor Core 指令族：一个线程发起、异步完成、累加器在 TMEM | [31](../course/31_tcgen05_gemm/README.md) |
| smem descriptor | shared memory descriptor | 64 位，描述 shared memory 中操作数的起始地址与 swizzle 布局 | [31](../course/31_tcgen05_gemm/README.md) |
| instruction descriptor | instruction descriptor | 32 位，描述 MMA 的形状与数据类型 | [31](../course/31_tcgen05_gemm/README.md) |
| `tcgen05.commit` | commit | MMA 全部完成后到达指定 mbarrier | [31](../course/31_tcgen05_gemm/README.md) |
| SS-MMA / TS-MMA | — | A 操作数来自 shared memory（SS）或 TMEM（TS）的 `tcgen05.mma` | [31](../course/31_tcgen05_gemm/README.md)、40 |
| CTA pair / 2-CTA MMA | `cta_group::2` | cluster 中相邻两个 CTA 合作完成一条 M = 256 的 MMA | [33](../course/33_2cta_gemm/README.md) |
| epilogue | epilogue | 主循环之后把累加器转换、写回 global memory 的阶段 | [31](../course/31_tcgen05_gemm/README.md) |
| tile | tile | 一次处理的矩阵子块 | [10](../course/10_simt_gemm/README.md) |
| 光栅化 | rasterization | tile 编号到 (行, 列) 的映射顺序，影响 L2 复用 | [21](../course/21_pipeline_gemm/README.md)、[32](../course/32_ws_gemm/README.md) |

## 性能分析

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| 算术强度 | arithmetic intensity | 每搬 1 字节做多少次浮点运算（FLOP/B） | [10](../course/10_simt_gemm/README.md) |
| roofline / 屋脊点 | roofline / ridge point | 性能上限 = min(峰值算力, 算术强度 × 带宽)；两者相等的点叫屋脊点 | [10](../course/10_simt_gemm/README.md) |
| 标称峰值 | nominal peak | 数据手册给的峰值，实际 kernel 达不到 | [00](../course/00_transpose/README.md) |
| ILP / TLP | instruction / thread-level parallelism | 一个线程内能同时进行的独立指令 / 同时驻留的线程 | [00](../course/00_transpose/README.md) |
| SASS | SASS | GPU 真正执行的机器指令 | [工具指南](tools.md) |
| PTX | Parallel Thread Execution | NVIDIA 的虚拟指令集，由 `ptxas` 翻译成 SASS | [01](../course/01_softmax/README.md) |
| MUFU / SFU | multi-function unit / special function unit | 计算 `exp2`、`rcp` 等特殊函数的单元，每 SM 每周期 16 个 | [01](../course/01_softmax/README.md) |
| uniform 寄存器 | uniform register | warp 内共享一份的标量寄存器，TMA、tcgen05 的地址与描述符放在这里 | [工具指南](tools.md) |

## attention 与数值

| 术语 | 英文 | 解释 | 首次出现 |
|---|---|---|---|
| attention | scaled dot-product attention | `O = softmax(scale·QKᵀ)·V` | [11](../course/11_naive_attention/README.md) |
| S / P / O | scores / probabilities / output | 分数矩阵、softmax 后的概率矩阵、输出 | [11](../course/11_naive_attention/README.md) |
| scale | softmax scale | 分数的缩放系数，通常为 `1/√D` | [11](../course/11_naive_attention/README.md) |
| causal mask | causal mask | 第 i 个 query 只看位置 ≤ i 的 key | [11](../course/11_naive_attention/README.md) |
| LSE | log-sum-exp | 每行 `m + ln(ℓ)`，即 softmax 分母的自然对数；输出形状 `[B,H,N]` FP32 | [11](../course/11_naive_attention/README.md) |
| safe softmax | safe softmax | 先减去行最大值再求指数，避免溢出 | [01](../course/01_softmax/README.md) |
| online softmax | online softmax | 一遍扫描同时维护最大值 m 与和 ℓ，遇到更大的值时重新缩放 | [01](../course/01_softmax/README.md) |
| 重新缩放 / correction | rescale / correction | 最大值变大时，把已累加的 ℓ 和 O 乘以 `exp(m_old − m_new)` | [12](../course/12_fa1/README.md) |
| 延迟归一化 | deferred normalization | 循环中不除以 ℓ，最后只除一次 | [12](../course/12_fa1/README.md)、[22](../course/22_fa2/README.md) |
| 循环交换 | loop interchange | FA2 把 Q 块放在外层循环，O 留在片上 | [22](../course/22_fa2/README.md) |
| IO 复杂度 | IO complexity | 算法需要的 HBM 读写量 | [12](../course/12_fa1/README.md) |
| oracle | oracle | 用 FP64 直接按定义计算的参考结果，判断正确性 | [11](../course/11_naive_attention/README.md) |
| atol / rtol | absolute / relative tolerance | 判定通过的条件 `|实际 − 参考| ≤ atol + rtol·|参考|` | [attention 的数学](attention-math.md#7-怎样判断结果正确) |
| ulp | unit in the last place | 浮点数最后一位的单位，用来描述 `ex2.approx` 等的误差 | [01](../course/01_softmax/README.md) |
| BF16 / FP16 / FP8（E4M3） | — | 16 位 / 16 位 / 8 位浮点格式。BF16 指数位多、尾数位少 | [20](../course/20_mma_gemm/README.md)、44 |
| descale | descale | FP8 输入的缩放系数，乘回才得到真实值 | [44](../course/44_fa4_fp8/README.md) |
