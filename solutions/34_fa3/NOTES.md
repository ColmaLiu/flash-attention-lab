# 34_fa3 reference solution notes

## Design choices

**One CTA per (Q tile, head, batch), 6 warps with fixed roles.** Warps 0–3 are the softmax warpgroup (thread r owns
row q0 + r = TMEM lane r); lane 0 of warp 4 issues TMA; lane 0 of warp 5 issues every `tcgen05.mma`. BM = BN = 128:
M = 128 is the single-CTA `tcgen05.mma` shape with one row per lane; BN = 128 makes S exactly 128 columns and the K of P·V = 128.
No persistent kernel, no LPT launch order (lab 43), and no 2-CTA (lab 42).

**S and O in TMEM, P through shared memory (SS MMA).** This is the most direct mapping of the FA3 ideas onto SM100: on Hopper, P is in registers
as the A operand of `wgmma` (RS), while the A of SM100 `tcgen05.mma` can come only from shared memory or TMEM. Writing P back to TMEM
(TS MMA) belongs to FA4 and is left to lab 40. P is written with the same 128B swizzle that TMA uses for Q, so the 16-byte stores have no conflicts.

**V is an MN-major B operand.** Bit 16 of idesc is set to 1; the descriptor has SBO = 1024 (8 key rows) and LBO = one 64-column d panel
(128 rows × 128 B = 16 KB); a K slice advances 16 rows. The layout follows the canonical layout of CUTLASS `make_umma_desc<Major::MN>` (`mma_traits_sm100.hpp`);
v1 (`s40_attention` on the main branch) uses the same encoding.

**The softmax warps correct O synchronously, every block.** For block j > 0, a softmax thread waits for `o_ready(j−1)`,
does `tcgen05.ld` → multiply by α → `tcgen05.st`, and arrives on `p_ready` only after `wait::st`. A separate correction warpgroup and conditional rescale
belong to labs 40–41 and are not done here.

**Separate full/empty barriers for K and V, with the same stage index.** The MMA warp needs K_j first (QKᵀ) and V_j later (PV), and releases them at different times.
D = 128 uses 2 stages and D = 64 uses 3; both fit only one CTA per SM (192 KB / 144 KB): step 5 uses all 512 TMEM columns,
so a second CTA would wait idle at `tcgen05.alloc` while holding shared memory (`Config` has a static_assert).

**S double buffering (step 5).** The MMA order is QK_0, QK_1, then per block "wait p_ready(j) → PV_j → QK_{j+2}". PV_j is issued first,
so the `o_ready(j)` that softmax waits on before writing P_{j+1} is not delayed by QK_{j+2}. Softmax finishes the exponentials first and then waits for `o_ready(j−1)`, so the exponentials
overlap PV_{j−1}. Steps 1–4 are the same kernel with `kSBuf = 1`.

**O is written back with a TMA store.** After the last `o_ready`, every MMA (including the QKᵀ that reads Q) has completed, so the shared memory of Q serves as
the staging area; each softmax warp writes its own 32 rows with a 4-D map with box `[32][64]`; TMA skips rows past N, so no checks are needed.
LSE is one float per thread, written directly; adjacent threads write adjacent addresses.

**4-D tensor map `{D, H, N, B}`.** B and N cannot be merged: the tail block would read the next batch, and the store of O would corrupt the next batch.

**Draining before exit.** After the main loop, the TMA warp waits on the last `k_empty/v_empty` of each stage, so that the CTA exits only after the arrives of all `tcgen05.commit`s
have landed (the producer tail of CUTLASS does the same).

**Ordering of TMEM reads.** The maintainer pointed out that in the header there is no data dependency between `tmem_ld_*` and `tmem_wait_ld()`, so the compiler may schedule
computation on the result registers before the wait. The solution always uses `tmem_ld_32x32b_x32_sync` from the header (ld + wait in one asm statement).
For one row of S (4 x32 loads), a local helper with "4 loads in flight at once + one wait that takes the 32 registers as `+f` operands" was also tried:
in the same bench run, `perf` was about 3% faster and `perf_d64` about 1% faster. The difference is small, so it was not adopted, to keep the lesson simple.

**Numerics.** Same as lab 22: m stores raw scores (scale > 0), a local wrapper of `ex2.approx.ftz`, and 0 as the subtrahend when `m' = −∞`.
max and sum each use 8 independent accumulators: each SMSP has only one softmax warp, so a dependency chain of length 128 would expose latency.

**New shared primitives.** This lab adds to the headers (pure additions, no existing signatures changed): in `include/lab/tma.cuh`,
`make_tensor_map_4d`, `tma_load_4d`, `tma_store_4d`; in `include/lab/tcgen05.cuh`,
`make_smem_desc_mnmajor_sw128(smem, lbo_bytes, sbo_bytes = 1024)`. Only `exp2_approx` and `pack2<T>` remain local (same as lab 22).

**The S double-buffering protocol lives in two functions.** `s_buffer<kSBuf>(i)` and `s_phase<kSBuf>(i)` are used both by the MMA warp (which buffer to write,
which barrier to commit) and by the softmax warps (which buffer to read, which parity to wait for); the one-buffer branch is given in the skeleton, and the two-buffer
branch plus the early issue in the MMA (QK_1, QK_{j+2}) are TODO(5).

## Performance per step (vs cuDNN, CUDA Graph, GPU 6)

| step | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 1, 2 (single block, `one_block`, verification only) | — | — | — |
| 3 (non-causal) | ≈ 0.41× | — | ≈ 0.41× |
| 4 (one S buffer) | ≈ 0.41× | ≈ 0.51× | ≈ 0.41× |
| 5 (S double buffering) | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× |
| Lab 22 step 4 (for comparison) | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4 / cuDNN | ≈ 1.07–1.13× | ≈ 1.28–1.34× | ≈ 1.13–1.16× |

`--cold`: step 5 `perf` ≈ 0.65× (Q/K/V are about 200 MB, larger than L2 anyway; cuDNN is relatively slower with a cold L2).
`perf_target` is set to 0.48× of cuDNN: above steps 3/4 without overlap (≈ 0.41×), below the worst `perf` of step 5 (≈ 0.54–0.56×), a margin of about 15%.

Nsight Compute (`perf`): `sm__pipe_tensor_cycles_active` ≈ 27% in step 3, ≈ 38% in step 5;
`smsp__inst_executed_pipe_xu` ≈ 33.83 M (≈ B·H·N²/32, one MUFU.EX2 per score); `dram__bytes_read` ≈ 201 MB.
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` is about 0.95 M in step 3 and about 11.31 M in step 5, although the write
addresses of P and of the O staging are exactly the same in both steps. Conjecture (unverified): this counter includes cycles where stores contend for shared memory with concurrent async-proxy reads (MMA reading operands, TMA writes);
in step 5 the MMA and softmax run at the same time, so there are more. Not broken down further.

Registers: set by the softmax path, 164 per thread (140 for the step 1 probe kernel), no spills, no stack.

## Known limitations

- The bottleneck is the softmax warpgroup: per block, 128×128 MUFU.EX2, elementwise FFMA/FMNMX/FADD, writing 32 KB of P, and correcting O all run serially in one
  warpgroup; the Tensor Core is idle about 60% of the time. FA4's remedies (ping-pong over 2 Q tiles, correction warpgroup,
  P in TMEM, polynomial exp2, conditional rescale) are left to labs 40–41.
- O is corrected unconditionally every block (at D = 128, 128 ld/multiply/st per thread per block).
- causal Q tiles launch in `blockIdx.x` order, so the heaviest tile starts last.
- Not persistent: every CTA re-initializes barriers, allocates TMEM, and loads Q.
- Supports only D ∈ {64, 128}, FP16/BF16, O with the same dtype as the inputs; steps 1–2 support only N ≤ 128 (checked by `make_plan`).
