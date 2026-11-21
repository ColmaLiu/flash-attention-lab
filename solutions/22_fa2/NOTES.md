# 22_fa2 reference solution notes

## Design choices

**One CTA per (Q block, head, batch), K/V in the inner loop.** This is sections 3.1/3.2 of the FA2 paper.
O, m, ℓ stay in registers from start to finish; O and LSE are each written once in the epilogue. The grid is
`(⌈N/Br⌉, H, B)`, with `blockIdx.x` varying fastest. No LPT/reverse launch order: causal load imbalance is left to lab 43.

**4 warps, partitioned by Q rows (split-Q, paper section 3.3).** Each warp owns `16·kMTiles` rows,
the softmax row reductions use only shuffles within a quad, and warps share only the K/V tiles.

| D | kMTiles | Br | Bc | Array registers per lane (Q/O/S) | Registers/thread | CTAs/SM |
|---|---|---|---|---|---|---|
| 128 | 1 | 64 | 64 | 32 / 64 / 32 | 218–237 | 2 |
| 64 | 2 | 128 | 32 | 32 / 64 / 32 | 168 | 3 |

- D = 64 uses Bc = 32 rather than upstream's 64/128: with Bc = 64 the arrays need 160 registers in total, and the non-causal
  kernels of steps 2/3 got a 16-byte stack (one LDL per block). Precomputing the index of "the blocks that need a mask"
  and moving the synchronization were both tried; they only moved the spill to another instantiation, or made the D = 128 kernel
  about 15% slower (changed register allocation). Bc = 32 is only about 8% slower in step 4 (0.43× → 0.39× cuDNN), and
  the register budget is the same for both D, which is easier to explain in the lesson.
- Tried 8 warps with Br = 128 for D = 128: step 4 dropped from ≈0.26× to ≈0.22×. Only one CTA fits per SM,
  so no other CTA fills in at the barriers.
- **Q stays resident in registers** (upstream's `Is_Q_in_regs` approach); each block only runs ldmatrix on K and V.

**shared memory.** `[rows][D]`, with `lab::swizzle_chunk<8>` applied independently to each 128-byte segment (a D = 128 row is two segments).
Both the `cp.async` writes and the `ldmatrix(.trans)` reads are conflict-free (the profile shows 0 `ldgsts` conflicts). Out-of-bounds
Q/K/V rows are written as zeros with `cp_async_16_zfill`: if a padding row of V were NaN, `0·NaN` would pollute O.

**P skips shared memory.** Two adjacent n8 S C-fragments form exactly one m16n8k16 A fragment
(lesson §4.3); the only work is packing FP32 → 16 bits. ℓ uses the FP32 P~ before packing (same as upstream).

**V uses `ldmatrix_x4_trans`.** V is `[key][D]`, which is MN-major for the B operand of P·V.

**Numerics.** m stores the raw scores (valid only for scale > 0, guaranteed by the binding); the exponent uses `exp2_approx`, a local
wrapper of `ex2.approx.ftz` (without `--use_fast_math`, `exp2f` is not a single MUFU), and the exponent argument is one
`fmaf(s, scale_log2, −m·scale_log2)`. When `m' = −∞`, 0 is used as the subtrahend to avoid `−∞ − (−∞)`; with this solution's
block partitioning, real rows never trigger it (key 0 of block 0 is visible to every row), so it is only defensive. ℓ stays a per-lane partial sum,
and is quad-summed only in the epilogue. The FP64 "extreme scale" path and the NaN problem for scale ≤ 0 of v1 (`s22_fa2` on the main branch)
have both been removed.

**causal.** The block count is `min(⌈N/Bc⌉, ⌈(q0+Br)/Bc⌉)`; a block needs a mask if and only if it extends past N, or
`k0 + Bc − 1 > q0`. Blocks are visited front to back (upstream goes back to front, processing the masked blocks first), so that the
same loop is shared with the step 4 pipeline.

**The step 4 pipeline.** Same structure as lab 21 step 1: Q and K/V block 0 both belong to group 0; each block has one
`wait_group 0` + one `__syncthreads()`, after which the next block is issued to the other stage. The finer upstream pipeline, with
K and V in separate groups and the V copy overlapped with QKᵀ, was not tried; based on lab 21's findings a 3-stage pipeline is expected to gain little, and it was not measured either.

**Local helpers (not in the headers):** `exp2_approx` (`ex2.approx.ftz.f32`), `pack2<T>`,
`mma<T>` (selects `lab::mma_m16n8k16_{f16,bf16}` by dtype), `quad_max/quad_sum`, and the two-segment swizzle
`tile_offset`. None of them is a new PTX primitive, so they were not put into `include/lab`.

## Performance per step (vs cuDNN, CUDA Graph, GPU 7)

| step | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 1 (probe, not meaningful) | — | — | — |
| 2 / 3 (synchronous loads) | ≈ 0.23× | ≈ 0.30× | ≈ 0.36× |
| 3, no block skipping (experiment) | — | ≈ 0.15× | — |
| 4 (double buffering) | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4 / cuDNN | ≈ 1.1× | ≈ 1.27× | ≈ 1.15× |

`perf_target` is set to 0.2× of cuDNN, leaving about 25% margin on the worst case, `perf`. On `perf`, step 4 reaches about 61% of the
`mma.sync` upper bound measured in lab 20 (≈ 560 TFLOP/s); `sm__pipe_tensor_cycles_active` is about 60% for step 2 and about 66% for step 4.
The ratios barely change under `--cold`. Compared with lab 12 (FP32 SIMT, `perf` about 25 TFLOP/s), `perf_d64` is about 13× higher
(the dtypes differ; the gain comes mainly from the Tensor Cores).

## Known limitations

- The upper bound of `mma.sync` itself is about 0.43× of cuDNN; going faster requires `tcgen05` (Part 3).
- Each SM has only 8–12 warps, softmax and mma run serially in the same warp, and there is no warp specialization.
- Causal does no LPT ordering; the heaviest Q blocks are launched last.
- O is written back directly from the fragments (4 bytes per lane), without merging into 16-byte writes through shared memory; O is only a small part of the traffic,
  and no effect was measured.
- Only D ∈ {64, 128} and FP16/BF16 are supported, with O in the same dtype as the inputs (checked in `make_plan`).
