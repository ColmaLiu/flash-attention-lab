# 11_naive_attention solution notes

## Design choices

- **Three separate kernels, with S/P in global memory.** This is Algorithm 0 of the FlashAttention paper. The purpose of this lab is to measure its cost, so it deliberately does no fusion.
- **One batched SIMT GEMM with arbitrary strides serves both QKᵀ and PV.** The single interface `C(m,n) = alpha·Σ_k A(m,k)·B(n,k)` (consistent with the course's GEMM convention) describes each operand with four strides `{b, h, row, col}`. One head in BNHD, S in `[B,H,N,N]`, and "V read as (d, j)" are then just different stride values. Students practice exactly this mapping, and V needs no transpose or copy.
- **The GEMM uses 64×64×16 tiles and 4×4 per thread, not the 128×128 and 8×8 of lab 10.** The n dimension of PV is D (64 or 128), so a 128-wide tile would waste half of itself at D = 64. Also, the loader must support arbitrary strides (it assigns lanes along the direction with stride 1), so it cannot assume float4 alignment. A thread owns rows `ty + 16i` and columns `tx + 16j`, which makes shared memory reads free of bank conflicts and the writes of C coalesced. This code is "not the focus of this lab", so readability comes first.
- **The softmax processes one row per block (128 threads), in two passes: online (m, l), then write P.** It uses `State`/`merge`/`block_merge` from lab 01. The reads of the second pass mostly hit L1/L2, so S is read from HBM only once (in the profile, the DRAM reads of `softmax_rows` ≈ the size of S). P overwrites S in place, so only one scratch buffer is needed.
- **Steps 1–2 write S/P into `p.probe`; step 3 writes into the `lab::Scratch` allocated by `make_plan`.** The three steps are the same code completed step by step, with an early return on `p.step`; no branching versions are needed.
- **Causal is handled only in the softmax.** Both GEMMs still compute the full N×N, and the upper triangle of P is written as 0. This makes the time of `perf_causal` almost the same as `perf`, which shows exactly that the standard implementation wastes half the work; skipping blocks belongs to later labs.
- **`make_plan` reports an explicit error for inputs where S exceeds 32 GiB.** This is the memory problem this lab demonstrates, instead of a silent failure.

## Verification

- `./lab check 11 --solution`: all cases of all three steps pass (default stream and non-default stream).
- Compute Sanitizer memcheck / racecheck / synccheck report nothing on `bnhd_causal`; memcheck reports nothing on `single` (N = 1) and `position_causal`.
- Filling the TODOs one by one in a copy of the skeleton: after TODO(1) is filled, step 1 passes and step 2 reports TODO(2); and so on.
- In `large_logits` (D = 32, scale = 1), more than 90% of the rows have a maximum score above 88.7; a version without the max subtraction gets NaN in step 2. With D raised to 128 and scores around 200, the rounding error of the FP32 GEMM itself (about 1e-4) already exceeds the tolerance of O, so this case uses D = 32.
- SASS: `gemm_kernel` uses 38 registers and `softmax_rows` 31, both without spills.

## Performance (ratios, CUDA Graph)

| case | vs efficient |
|---|---|
| `perf` [1, 4096, 16, 64] | ≈ 0.32× |
| `perf_causal` | ≈ 0.21× |

- Time split (`perf`): QKᵀ ≈ 40%, softmax ≈ 9%, PV ≈ 50%. The two GEMMs are compute-bound (SIMT FMA, about 15–19 TFLOP/s), and the softmax is bandwidth-bound.
- DRAM bytes (`perf`): measured total ≈ 4.25 GB, Section 3.4 estimate 4.36 GB; the difference is dirty data still in L2 when the kernels end. The efficient baseline measures ≈ 0.05 GB of DRAM reads for the same shape.
- Doubling N multiplies the time by about 3.1 (1024→2048) and 3.8 (2048→4096).
- `perf_target` is set to 0.15 (relative to efficient). It only shows that the implementation has no obvious performance bug; speed is not the goal of this lab.

## Known limitations

- The FP32 path of the efficient baseline uses Tensor Core 3xTF32 (inferred from source: `DefaultGemmType<ArchTag, float>` in `gemm_kernel_utils.h` selects `OpMultiplyAddFastF32`), so not all of its gap to this solution comes from IO.
- At D = 64 the PV grid has only B·H·⌈N/64⌉ blocks, so SM occupancy is low; split-K or a smaller M tile would help, but that is not part of this lab.
- The softmax still uses 128 threads for short rows such as N = 1, leaving most threads idle; this has no effect at the sizes in this lab.
