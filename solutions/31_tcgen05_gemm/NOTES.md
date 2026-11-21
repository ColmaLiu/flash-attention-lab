# 31_tcgen05_gemm solution notes

## Design choices

- **Supported domain**: A and B are BF16, C is FP32; `K % 8 == 0`, `N % 4 == 0` (the TMA row strides of A, B and C must be multiples of 16 bytes); M is arbitrary. All tails are left to TMA: loads fill out-of-bounds elements with zero (both rows and K), and stores skip out-of-bounds elements. Step 1 additionally requires `M ≤ 128, N ≤ 128, K ≤ 64` (checked by make_plan).
- **Primitives**: `include/lab/tcgen05.cuh` (new) + `mbarrier.cuh` and `tma.cuh` from lab 30. The bit fields were checked against `SmemDescriptor` / `InstrDescriptor` in CUTLASS `mma_sm100_desc.hpp`: descriptor version = 1 (bits 46–47), layout 2 = SWIZZLE_128B (bits 61–63), SBO = 1024; in idesc, M>>4 is in bits 24–28 and N>>3 in bits 17–22. In v1 (s31 on the main branch) B was MN-major; in v2, B is `[N,K]` K-major, A and B use the same descriptor function, and `b_major = 0`.
- **Thread organization**: 128 threads = 4 warps = the 4 lane sections of TMEM. Thread 0 issues both TMA and MMA (no warp specialization; that is the topic of lab 32). The other threads wait on `mma_done`, and then all threads run the epilogue.
- **step 1**: single tile 128×128×64: one expect_tx (32 KB), two TMAs, 4 MMAs, `tcgen05.commit` → `mma_done`. The epilogue gives each thread one row (`tcgen05.ld.32x32b.x32` + `wait::ld`) and writes global memory element by element.
- **step 2**: `gemm_pipelined<Config<128, 4, false>>`. The prologue fills S stages. In the loop, the thread first issues MMA t and commits it to `empty[t%S]`, then waits on the `empty` of MMA t−1 and refills its stage (tile t−1+S), so the Tensor Core queue always holds MMA t during the wait. The step 2 grid already covers all M×N tiles, so step 2 runs all correctness cases. This differs slightly from the task description ("step 2 only does a single-CTA tile"): multiple CTAs are only one line of `blockIdx` mapping, which has no teaching value in step 3, so step 3 instead focuses on the epilogue and the tile shape.
- **step 3**: `Config<256, 4, true>`: 192 KB shared memory, 256 TMEM columns, 1 CTA per SM. The epilogue reuses the stage buffers for staging (all loads have been waited on, all MMAs have completed). Each warp uses one 32×32 box per 32 columns and writes it with 128B swizzle (16-byte stores without bank conflicts; NCU counts 0). After `fence.proxy.async` + `__syncwarp`, lane 0 issues the TMA store; no synchronization between warps is needed. Before exit, `cp.async.bulk.wait_group.read 0` (the same as CUTLASS's `tma_store_wait<0>`).
- **TMEM lifecycle**: warp 0 executes alloc and relinquish. After `fence::before → __syncthreads → fence::after`, all threads read the base. Before dealloc, the same fence sequence guarantees that all `tcgen05.ld` have completed.

## Performance per step (GPU 3, CUDA Graph, relative to cuBLAS BF16→FP32)

| Configuration | 4096³ | 8192×8192×4096 |
|---|---|---|
| step 2: `<128,4>` per-row write-back | ≈ 0.37× | ≈ 0.33× |
| `<128,4>` TMA write-back | ≈ 0.51× | ≈ 0.42× |
| `<256,4>` per-row write-back | ≈ 0.50× | ≈ 0.45× |
| `<256,3>` TMA write-back | ≈ 0.62× | ≈ 0.55× |
| `<128,6>` TMA write-back | ≈ 0.52× | ≈ 0.46× |
| `<128,3>` TMA write-back (2 CTA/SM) | ≈ 0.34–0.6× (unstable) | ≈ 0.60× |
| step 3: `<256,4>` TMA write-back | ≈ 0.81–0.83× | ≈ 0.74–0.77× |

- `selftest --perf`: square 0.830, wide 0.751, perf gate 0.6 (the task description expected 0.5–0.7; the measured values are higher, and the gate leaves a margin because this machine's GPU occasionally slows down across the board between runs).
- NCU (square): `sm__pipe_tensor_cycles_active` step 2 ≈ 26%, step 3 ≈ 68%; DRAM writes step 2 ≈ 99 MB (C is 67 MB; per-row stores produce scattered partial-sector writes), step 3 ≈ 42 MB; the epilogue's shared stores have 0 bank conflicts.
- Remaining gap (inferred, not measured item by item): the epilogue does not overlap the main loop (1 CTA per SM), a single thread waits on TMA and MMA serially, the 3.46-wave tail of square, and every CTA re-reads A/B from L2. These are handled by lab 32 (warp specialization, TMEM double buffering, persistent) and lab 33 (2-CTA, multicast).

## Verification

- `./lab check 31 --solution`: all cases of all 3 steps pass (`int` cases exactly equal, `random` max error 2.4e-4, within tolerance).
- `selftest 31 --perf` passes; the skeleton reports `TODO(1)` on the GPU (warp 0's `tmem_alloc` traps first, so there is no deadlock).
- compute-sanitizer: memcheck / racecheck / synccheck report 0 errors on `small` (step 1) and `tail` (steps 2, 3); racecheck (step 3) and synccheck (step 2) report 0 errors on `long_k`. The sanitizer printed no message about tcgen05/TMA being unsupported, but the experiments in lab 30 show that racecheck does not track async proxy accesses; TMEM accesses and MMA reads of shared memory are also outside its scope (inferred). So 0 errors only means the generic proxy part has no problem; the synchronization correctness of tcgen05 rests on reasoning from the specification and on exact-comparison cases (`long_k` wraps around the ring 4 times).
- `./lab sass 31 --solution`: no spill (56/46/44 registers); contains `UTCHMMA`, `UTCBAR`, `UTMALDG`, `UTMASTG`, `LDTM`, `SYNCS`.
- Skeleton: in a temporary copy, TODO(1), (2), (3) were filled in step by step; step n depends only on the first n groups of TODOs. After filling, the student version runs at the same speed as the solution (square ≈ 0.81×). The skeleton has been restored to its original state.

## Known limitations

- `K % 8 ≠ 0` or `N % 4 ≠ 0` is not supported (TMA row stride limit).
- BF16 → FP32 only; FP16 only needs the A/B format bits of idesc changed.
- No rasterization: `blockIdx.x` runs along N. On wide the L2 hit rate is about 84%; the kernel is limited by the Tensor Core and the epilogue, and this was not optimized.
- In step 3, `<128,3>` (2 CTAs per SM) gives highly variable results on square; the cause has not been investigated.
