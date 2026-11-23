# 40_fa4_pipeline reference solution notes

## Design choices

**Two Q tiles (256 rows) and two softmax warpgroups per CTA.** The structure comes from FA4 (`flash_fwd_sm100.py`, `q_stage = 2`):
warps 0–3 are the softmax of tile 0, warps 4–7 the softmax of tile 1; thread r ↔ row r of its tile ↔ TMEM lane r. The two tiles share
K_j and V_j. No persistent kernel, LPT (lab 43), 2-CTA (lab 42), polynomial exp2 or conditional rescale (lab 41).

**The 512 TMEM columns split exactly between the two tiles.** S⁰ 0–127, S¹ 128–255, O⁰/O¹ from 256 on, D columns each. Each tile has only one S region, and P is written into the last
64 columns of S (FA4's `tmem_s_to_p_offset = n_block_size // 2`). The "S double buffering within one tile" of lab 34 does not fit, so the overlap comes instead from the two
tiles.

**MMA issue order (FA4's non-`s_ping_pong` branch).** Prologue QK⁰₀, QK¹₀; per block `PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1}`. Consequences:
- phase j of `s_full[t]`, which softmax waits for, also means PV^t_{j−1} has completed (the commit tracks all earlier MMAs of the same thread, guaranteed by PTX), so
  neither softmax writing P nor correction reading O needs an extra `o_ready`;
- QK^t_{j+1} overwrites the P^t that PV^t_j is reading, which **relies on MMAs from the same thread executing in issue order**. PTX 8.7 guarantees this only for MMA pairs with "the same accumulator and
  shape", which does not hold here; FA4 and CUTLASS 77 both do it, this solution follows them, and lesson §4.3 says so explicitly. Step 1
  (P in shared memory) does not rely on this. There is no performance comparison with a "strictly by the spec" version (wait for the PV commit before QK).

**K and V share one ring.** item u: even is K_{u/2}, odd is V_{u/2}, matching the order in which the MMA consumes them; K is released after the QKs of both tiles,
V after both PVs. Once P is no longer in shared memory (steps 2–3), D = 128 goes from 3 to 4 stages and D = 64 from 6 to 8 stages. With 5 stages at D = 128 in
steps 2–3, shared memory exceeds 227 KB (counting the correction statistics).

**correction warpgroup (step 3).** As soon as softmax has m′ and α, it hands them to correction through shared memory (`Stats::alpha`) + `stats_full`,
then computes exp2; correction corrects O and arrives on `p_ready` (count 256 = softmax 128 + correction 128).
`stats_full` needs no empty barrier (the causal chain in lesson §5.6); the final m and ℓ use a separate `final_full`, otherwise `stats_full` could
complete two phases in a row. The epilogue (O/ℓ → smem of Q^t → TMA store, LSE) is done by correction, the same as FA4's
`use_correction_warps_for_epi`. The correction is unconditional (lab 41 does conditional rescale).

**setmaxnreg.** 16 warps (512 threads, 128 per thread at launch): softmax `inc` 192, correction `dec` 80, MMA/TMA/2 idle
warps `dec` 48, 2·192 + 80 + 48 = 512. The values match FA4's defaults (`num_regs_softmax = 192`, `num_regs_correction = 80`).
ptxas allocates registers for each code region according to the value after `setmaxnreg`: the MMA region has only 48, and at first the compiler hoisted the 16 Q descriptors of the two tiles out of the loop,
spilling about 160 bytes; with `#pragma unroll 1` on the tile loop of the MMA there are no spills. Steps 1–2 have 10 warps and no `setmaxnreg`;
the ptxas limit is 168 (320 threads counted as 384); in step 2, P is written with the `x16` `tcgen05.st` (`x32` spills 8–16 bytes).

**The whole MMA warp runs the schedule; one elected lane issues.** Instrumenting one CTA with `clock64()` (not committed) showed: when the MMA runs only on lane 0
(the other 31 lanes park at the final `__syncthreads`), the MMA thread takes about 250–550 cycles after `p_ready` has completed to issue PV; with the whole
warp waiting and one lane chosen by `elect.sync` issuing, this drops to about 40 cycles, and `perf` is about 6% faster (0.79× → 0.84×). The cause (scheduling of divergent warps)
was not verified further. One way of writing the TMA warp as "whole warp waits + leader issues Q and K/V" was instead about 10% slower (0.84× → 0.75×, reproducible), while another way
(a per-warp arrive variant) made no difference; the cause was not found, so the TMA warp keeps the lane-0 style (it loads several stages ahead and is not on the critical path).

**MMA descriptors: computed once, offset per K slice.** The start address field is in the low 14 bits (in 16-byte units); shared addresses < 256 KB never carry,
so `desc + (bytes >> 4)` is enough, which saves instructions in the MMA thread. Measured on its own, the difference from recomputing the descriptor per slice is within noise (≈1–2%); kept to keep the MMA thread's
instruction stream short.

**Attempts not adopted** (compared in the same run, differences within noise): keeping the 4 `tcgen05.ld` of S in flight together with a single wait; removing the
`tcgen05.fence::after_thread_sync` in `acquire`; one `p_ready` arrive per warp (count 4/8); interleaving exp2 and writing P in chunks.

## Performance per step (vs cuDNN, CUDA Graph, range over 3 runs each)

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| Lab 34 step 5 (for comparison) | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× | — |
| Step 1 (ping-pong, P through smem) | ≈ 0.77–0.78× | ≈ 0.88–0.89× | ≈ 0.80× | ≈ 0.75–0.76× |
| Step 2 (P in TMEM, TS MMA) | ≈ 0.80× | ≈ 0.89–0.90× | ≈ 0.73–0.75× | ≈ 0.77–0.78× |
| Step 3 (correction WG + setmaxnreg) | ≈ 0.84× | ≈ 0.93–0.94× | ≈ 0.79–0.81× | ≈ 0.82× |
| FA4 (`flash_attn.cute`) | ≈ 1.15× | ≈ 1.33× | ≈ 1.13× | ≈ 1.10× |

Instrumented timeline (`perf`, steady state, SM cycles): period of one block T ≈ 3450 (≈ 3700 with the MMA on lane 0 only); softmax WG from receiving S to arriving
≈ 1900–2200, of which the 128 exp2 take ≈ 1450–1700; PV + QK of one tile ≈ 1100. T ≈ softmax + PV + QK + handoff, consistent with the chain model of lesson §3.1.
The MUFU lower bound is 1024 cycles per tile per block, so the polynomial exp2 of lab 41 is the next step.

## Verification

- `./lab check 40 --solution`: all 3 steps × 25 cases pass; `./lab selftest 40` and `selftest 40 --perf` (target 0.72) pass;
- memcheck, racecheck and synccheck of `./lab sanitize --solution` report 0 errors on `two_tiles` and `causal_tail` (all steps);
  racecheck cannot see TMA/TMEM accesses; the synchronization argument is the table in lesson §5.6;
- `./lab sass 40 --solution`: STACK is 0 for all 24 kernels; steps 1–2 use 164–168 registers per thread, step 3 launches with 128 (each role's count is set by
  `setmaxnreg`, ptxas allocates per region, no spills);
- skeleton: generated from the solution; after replacing the TODOs of steps 1..n with solution code one step at a time, `--step n` passes and `--step n+1` reports TODO(n+1); filling only 1a,
  or filling up to 3a, also stops at a TODO instead of hanging; restored to the original skeleton afterwards (byte-identical).
- `perf_target` is 0.72 rather than 0.75: the worst case, `perf_d64`, is at ≈ 0.79–0.81×, and cuDNN itself varies by about 4%.

## Known limitations

- Relies on in-order execution of MMAs from the same thread (above).
- In causal mode both tiles go through the same number of blocks, so per pair tile 0 computes one extra fully masked block (same as FA4); Q tiles launch in `blockIdx.x` order (lab 43).
- When N is very small, a fully out-of-bounds tile 1 is still computed (the results are not written back).
- Not persistent; every CTA re-initializes barriers, allocates TMEM, and loads Q.
- Supports only D ∈ {64, 128}, FP16/BF16, O with the same dtype as the inputs.
