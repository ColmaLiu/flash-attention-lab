# Glossary

Grouped by topic. "First appears" is the lab that first explains the concept.

## Threads and execution

| Term | Meaning | First appears |
|---|---|---|
| kernel | A `__global__` function launched by the host and executed by many GPU threads at the same time | [Starting from zero](starting-from-zero.md) |
| host | The CPU code that launches GPU work | [Starting from zero](starting-from-zero.md) |
| thread block / CTA (Cooperative Thread Array) | A group of threads that share shared memory and can synchronize with `__syncthreads()`. CTA is the PTX name | [00](../course/00_transpose/README.md) |
| grid | All blocks of one launch | [00](../course/00_transpose/README.md) |
| warp | A group of 32 threads that execute the same instruction together | [00](../course/00_transpose/README.md) |
| lane | A thread's index 0–31 within its warp. The TMEM "lane" means something else (see below) | [00](../course/00_transpose/README.md) |
| warpgroup | 4 consecutive warps (128 threads) whose first warp index is a multiple of 4 | [32](../course/32_ws_gemm/README.md) |
| SM (Streaming Multiprocessor) | The hardware unit that executes warps; B200 has 148 | [00](../course/00_transpose/README.md) |
| SMSP (SM sub-partition) | One quarter of an SM, with its own warp scheduler | [20](../course/20_mma_gemm/README.md) |
| residency / occupancy | The number of warps present on one SM at the same time (as a fraction of the maximum 64), limited by registers, shared memory and thread count | [00](../course/00_transpose/README.md) |
| wave | The number of blocks in the grid divided by the number of blocks that can be resident at once. When the last wave is not full, some SMs sit idle | [10](../course/10_simt_gemm/README.md) |
| stream | A GPU work queue. Work in the same stream runs in order | [Starting from zero](starting-from-zero.md) |
| CUDA Graph | Records a sequence of launches and replays it as a whole, removing the per-launch overhead | [Tools guide](tools.md) |
| persistent kernel | The grid is only as large as "the number of blocks that can be resident", and each block loops over several tiles | [30](../course/30_tma_mbarrier/README.md) |
| warp specialization | Different warps in the same block take different roles (TMA, MMA, epilogue, softmax) | [32](../course/32_ws_gemm/README.md) |
| producer / consumer | The side that puts data into a buffer / the side that takes data out | [30](../course/30_tma_mbarrier/README.md) |
| `elect.sync` (elect) | All active threads of a warp execute it together, and it picks exactly one thread | [30](../course/30_tma_mbarrier/README.md) |
| `setmaxnreg` | Redistributes registers between warpgroups at run time | [32](../course/32_ws_gemm/README.md) (introduced), 40 |
| LPT scheduling (Longest Processing Time first) | Issue the blocks with the most work first, to reduce idle time in the last wave | [43](../course/43_fa4_schedule/README.md) |

## Memory

| Term | Meaning | First appears |
|---|---|---|
| global memory / HBM (High Bandwidth Memory) | Device memory that all threads can access. B200 has HBM3e, nominal 8 TB/s | [00](../course/00_transpose/README.md) |
| shared memory (smem) | On-chip memory shared by the threads of one block, 228 KB per SM | [00](../course/00_transpose/README.md) |
| register | Private to each thread; 65536 32-bit registers per SM | [01](../course/01_softmax/README.md) |
| local memory / register spill | When registers run out, the compiler puts variables in local memory (which actually lives in device memory) | [01](../course/01_softmax/README.md) |
| L2 (L2 cache) | The cache shared by the whole chip, 126 MB on B200 | [00](../course/00_transpose/README.md) |
| sector | The smallest unit moved between L1/L2/HBM, 32 bytes | [00](../course/00_transpose/README.md) |
| coalesced access (coalescing) | The accesses of one warp fall on contiguous addresses and need only a few sectors | [00](../course/00_transpose/README.md) |
| bank / bank conflict | shared memory is split into 32 banks; when several threads of one warp access different addresses in the same bank, they queue | [00](../course/00_transpose/README.md) |
| padding | Leave a few extra elements at the end of each row of a shared memory array, changing the row stride to avoid bank conflicts | [00](../course/00_transpose/README.md) |
| swizzle | Permute the positions of 16-byte chunks in shared memory by XOR to avoid bank conflicts | [20](../course/20_mma_gemm/README.md), [30](../course/30_tma_mbarrier/README.md) |
| bytes in flight | The amount of memory traffic issued but not yet returned. Bandwidth = bytes in flight / latency (Little's law) | [00](../course/00_transpose/README.md) |
| contiguous | The tensor is packed in row-major order, with the last dimension contiguous | [00](../course/00_transpose/README.md) |
| K-major / MN-major | Whether the contiguous dimension of a matrix operand is K (the reduction dimension) or M/N | [10](../course/10_simt_gemm/README.md), [34](../course/34_fa3/README.md) |
| scratch | A temporary global buffer allocated in `make_plan` (`lab::Scratch`); its initial contents are garbage | [11](../course/11_naive_attention/README.md) |
| TMEM (Tensor Memory) | Storage of 128 lanes × 512 columns × 32 bit per SM that holds the accumulators of `tcgen05.mma` | [31](../course/31_tcgen05_gemm/README.md) |
| TMEM lane / column | A row / column of TMEM. Warp w can access only the 32 rows starting at lane `32·(w%4)` | [31](../course/31_tcgen05_gemm/README.md) |
| DSMEM (distributed shared memory) | CTAs in a cluster access each other's shared memory | [33](../course/33_2cta_gemm/README.md) |

## Asynchronous copies and synchronization

| Term | Meaning | First appears |
|---|---|---|
| `__syncthreads()` (CTA barrier) | Continues only after all threads of the block have arrived | [00](../course/00_transpose/README.md) |
| warp shuffle | Threads in a warp exchange register values directly (`__shfl_xor_sync`) | [01](../course/01_softmax/README.md) |
| `cp.async` (asynchronous copy) | Asynchronous global → shared copy that does not pass through registers | [21](../course/21_pipeline_gemm/README.md) |
| commit group / wait_group | Packs the issued `cp.async` into a group; waits until at most N groups are incomplete | [21](../course/21_pipeline_gemm/README.md) |
| pipeline / multi-stage buffer | Several buffers take turns holding data so that copying and computing overlap | [21](../course/21_pipeline_gemm/README.md) |
| ring buffer | S buffers used in the cycle 0, 1, …, S−1, 0, … | [21](../course/21_pipeline_gemm/README.md) |
| TMA (Tensor Memory Accelerator) | A unit that moves a whole block of multidimensional data in hardware, launched by one thread | [30](../course/30_tma_mbarrier/README.md) |
| tensor map (`CUtensorMap`) | A 128-byte TMA descriptor, encoded on the host | [30](../course/30_tma_mbarrier/README.md) |
| box | The block of data moved by one TMA copy | [30](../course/30_tma_mbarrier/README.md) |
| mbarrier | An 8-byte synchronization object in shared memory that records the phase, the pending arrival count and the pending byte count | [30](../course/30_tma_mbarrier/README.md) |
| phase / parity | Each time an mbarrier completes a round, its phase increases by 1; waiters use whether the phase is odd or even (the parity) to tell whether it has completed | [30](../course/30_tma_mbarrier/README.md) |
| expect_tx / complete_tx (transaction count) | Tells the mbarrier how many more bytes to wait for / subtracts those bytes after TMA finishes writing | [30](../course/30_tma_mbarrier/README.md) |
| full / empty barrier | The two sets of mbarriers in a ring buffer: "data is loaded" / "buffer is used up and may be overwritten" | [30](../course/30_tma_mbarrier/README.md) |
| proxy / proxy fence (`fence.proxy.async`) | Ordinary ld/st belong to the generic proxy, TMA and tcgen05 belong to the async proxy; handing shared memory between the two needs a fence | [30](../course/30_tma_mbarrier/README.md) |
| cluster (thread block cluster) | A group of CTAs that are guaranteed to run at the same time on the same GPC | [33](../course/33_2cta_gemm/README.md) |
| multicast (TMA multicast) | One TMA read is written into the shared memory of several CTAs in the cluster | [33](../course/33_2cta_gemm/README.md) |

## Tensor Core

| Term | Meaning | First appears |
|---|---|---|
| Tensor Core | A hardware unit dedicated to small matrix multiply-adds | [20](../course/20_mma_gemm/README.md) |
| `mma.sync` (warp-level MMA) | One warp synchronously completes one 16×8×16 matrix multiply-add; operands and accumulators are all in registers | [20](../course/20_mma_gemm/README.md) |
| fragment | The share of one `mma.sync` operand held in the registers of 32 threads | [20](../course/20_mma_gemm/README.md) |
| `ldmatrix` | Loads 8×8 matrices from shared memory in fragment layout | [20](../course/20_mma_gemm/README.md) |
| `tcgen05` (5th-generation Tensor Core) | The Blackwell Tensor Core instruction family: launched by one thread, completes asynchronously, accumulators in TMEM | [31](../course/31_tcgen05_gemm/README.md) |
| smem descriptor (shared memory descriptor) | 64 bits; describes the start address and swizzle layout of an operand in shared memory | [31](../course/31_tcgen05_gemm/README.md) |
| instruction descriptor | 32 bits; describes the shape and data types of the MMA | [31](../course/31_tcgen05_gemm/README.md) |
| `tcgen05.commit` (commit) | Arrives on the given mbarrier after all MMAs complete | [31](../course/31_tcgen05_gemm/README.md) |
| SS-MMA / TS-MMA | A `tcgen05.mma` whose A operand comes from shared memory (SS) or TMEM (TS) | [31](../course/31_tcgen05_gemm/README.md), 40 |
| CTA pair / 2-CTA MMA (`cta_group::2`) | Two adjacent CTAs in a cluster cooperate on one MMA with M = 256 | [33](../course/33_2cta_gemm/README.md) |
| epilogue | The stage after the main loop that converts the accumulators and writes them back to global memory | [31](../course/31_tcgen05_gemm/README.md) |
| tile | The matrix sub-block processed at one time | [10](../course/10_simt_gemm/README.md) |
| rasterization | The order in which tile indices map to (row, column); affects L2 reuse | [21](../course/21_pipeline_gemm/README.md), [32](../course/32_ws_gemm/README.md) |

## Performance analysis

| Term | Meaning | First appears |
|---|---|---|
| arithmetic intensity | How many floating-point operations are performed per byte moved (FLOP/B) | [10](../course/10_simt_gemm/README.md) |
| roofline / ridge point | Performance upper bound = min(peak compute, arithmetic intensity × bandwidth); the point where the two are equal is the ridge point | [10](../course/10_simt_gemm/README.md) |
| nominal peak | The peak from the datasheet; real kernels do not reach it | [00](../course/00_transpose/README.md) |
| ILP / TLP (instruction / thread-level parallelism) | Independent instructions that can be in progress at once within one thread / threads resident at once | [00](../course/00_transpose/README.md) |
| SASS | The machine code the GPU actually executes | [Tools guide](tools.md) |
| PTX (Parallel Thread Execution) | NVIDIA's virtual instruction set, translated into SASS by `ptxas` | [01](../course/01_softmax/README.md) |
| MUFU / SFU (multi-function unit / special function unit) | The unit that computes special functions such as `exp2` and `rcp`, 16 per SM per cycle | [01](../course/01_softmax/README.md) |
| uniform register | A scalar register shared as one copy by the whole warp; addresses and descriptors for TMA and tcgen05 go here | [Tools guide](tools.md) |

## Attention and numerics

| Term | Meaning | First appears |
|---|---|---|
| attention (scaled dot-product attention) | `O = softmax(scale·QKᵀ)·V` | [11](../course/11_naive_attention/README.md) |
| S / P / O (scores / probabilities / output) | The score matrix, the probability matrix after softmax, and the output | [11](../course/11_naive_attention/README.md) |
| scale (softmax scale) | The scaling factor of the scores, usually `1/√D` | [11](../course/11_naive_attention/README.md) |
| causal mask | Query i sees only keys at positions ≤ i | [11](../course/11_naive_attention/README.md) |
| LSE (log-sum-exp) | `m + ln(ℓ)` for each row, that is, the natural log of the softmax denominator; output shape `[B,H,N]` FP32 | [11](../course/11_naive_attention/README.md) |
| safe softmax | Subtract the row maximum before taking exponentials, to avoid overflow | [01](../course/01_softmax/README.md) |
| online softmax | One pass that maintains both the maximum m and the sum ℓ, rescaling when a larger value appears | [01](../course/01_softmax/README.md) |
| rescale / correction | When the maximum grows, multiply the accumulated ℓ and O by `exp(m_old − m_new)` | [12](../course/12_fa1/README.md) |
| deferred normalization | Do not divide by ℓ inside the loop; divide only once at the end | [12](../course/12_fa1/README.md), [22](../course/22_fa2/README.md) |
| loop interchange | FA2 puts the Q blocks in the outer loop, keeping O on chip | [22](../course/22_fa2/README.md) |
| IO complexity | The amount of HBM reads and writes an algorithm needs | [12](../course/12_fa1/README.md) |
| oracle | A reference result computed directly from the definition in FP64, used to judge correctness | [11](../course/11_naive_attention/README.md) |
| atol / rtol (absolute / relative tolerance) | The pass condition `|actual − reference| ≤ atol + rtol·|reference|` | [Attention math](attention-math.md#7-how-to-tell-whether-a-result-is-correct) |
| ulp (unit in the last place) | The unit of the last digit of a floating-point number, used to describe the error of `ex2.approx` and the like | [01](../course/01_softmax/README.md) |
| BF16 / FP16 / FP8 (E4M3) | 16-bit / 16-bit / 8-bit floating-point formats. BF16 has more exponent bits and fewer mantissa bits | [20](../course/20_mma_gemm/README.md), 44 |
| descale | The scaling factor of an FP8 input; multiply it back to get the true value | [44](../course/44_fa4_fp8/README.md) |
