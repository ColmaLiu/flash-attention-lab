# 43_fa4_schedule reference solution notes

## Design choices

**The starting point is the lab 41 solution, step 3; the inside of the CTA does not change.** The roles of the 16 warps, the TMEM layout, the K/V ring, the softmax arithmetic (MUFU/polynomial mix with k = 3,
conditional rescale with τ = 8) and all existing mbarriers are kept. The changes from the lab 41 review are ported: in `exp2_poly2`, f = x − n uses a single
`__ffma2_rn(n, (−1, −1), x)` (bit-identical, two fewer negations). The three steps are the same kernel, selected by two compile-time switches in `Config<D, Step>`: `kPersistent` and
`kLpt`.

**Work tile and block classification (step 1).** A work tile is still (one pair of Q tiles, head, batch), and the two tiles share one K/V stream, so the block range is computed per pair
(same as FA4's `cta_tiler[0] = 2·128`). `num_unmasked_blocks(qt, n) = min(⌊n/128⌋, qt/128)` gives how many leading blocks of each tile
need no mask at all; the softmax loop is split into two segments, `softmax_block<false>` and `softmax_block<true>`. The last block of tile 0 (fully masked for it)
is still computed, same as FA4; skipping it exactly needs a separate phase count for each tile, so it is a self-check question in the lesson and is not implemented.

**A single tile loop.** Every role calls `for_each_tile(p, body)`, which provides the tile coordinates, the block count, `it` (number of earlier tiles) and `nb`
(number of earlier blocks). Phases: barriers used once per block use `nb + j`, barriers used once per tile use `it`, and ring buffers use item numbers that continue across tiles.
The static schedule depends only on (blockIdx.x, gridDim.x, k), so roles need no communication. In step 1 the grid = total tile count, and the loop runs only once.

**Resource handover with persistent (step 2).**
- Q^t's shared memory: MMA does `tcgen05.commit(q_empty[t])` after the tile's last Q^t Kᵀ (in the prologue for tiles with only 1 block),
  and TMA waits on it before loading Q^t. The TMA issue order changes to K_0, Q⁰, Q¹, V_0 …, so the next tile's K_0 is already in flight while TMA waits for Q to be released.
- O no longer goes through Q's shared memory + TMA store. Instead each thread writes its own row directly with 16-byte `st.global`. Reasons: with persistent, Q's
  shared memory should be handed to the next tile as early as possible; at D = 128 the shared memory (≈ 198 KB) has no room for a separate 32 KB O staging buffer (about 1 KB short).
  Step 1 also uses this epilogue, so the difference between step 1 and lab 41 includes it (measured within noise).
- O^t's TMEM: needs no separate barrier. Before the next tile's first PV, MMA waits on `p_ready` (count 256, including 128 correction
  threads), and correction arrives for block 0 of the next tile only after it has finished the previous tile's epilogue (`tcgen05.ld` completed and fenced).
  The first version had an `o_empty` barrier; the independent review pointed out that it is implied by `p_ready` (10/10 passed after removing it), so it was deleted.
- `final_full` and `st.fin_m/fin_l` are **both** double-buffered by tile parity (`final_full[it & 1][t]`, parity `(it >> 1) & 1`). In tiles with only 1 block
  (N ≤ 128), softmax writing the final values for the next tile and arriving `final_full` depends only on the next tile's S_0, not on correction,
  so it can be one tile ahead. The first version double-buffered only the data: the barrier could complete two phases in a row before correction waited, and correction never got its phase.
  The review reproduced the hang on `[4,128,500,128]` (3/3); it is now case `many_one_block`. The lead never exceeds one tile (lesson §5.4).
- drain: before TMA exits, it waits until the last commits of the ring buffer and of `q_empty` have arrived.

**LPT and snake (step 3).** For causal, idx → layer = idx / (B·H), pair = P − 1 − layer, bh = idx % (B·H); `tile_index` deals forward in even rounds
and in reverse in odd rounds. Non-causal keeps "pair varies fastest" (tiles of the same head are adjacent, L2-friendly); all tiles weigh the same, so snake does not affect balance.

**Why no head sections (FA4's L2 swizzle) and no dynamic scheduling.** Simulation (each tile adds 0.5 blocks of switching overhead, see lesson §3.4) and measurement both
show: static snake requires per-round work to be monotonic, and FA4's section order breaks that (measured variant D is 26% slower). Without persistent, relying on hardware greedy dispatch,
sections are usable (variant C), but on this lab's cases L2 is not the bottleneck, and sections are not faster than no sections. Dynamic scheduling (atomic counter + broadcast between roles, or SM100's
CLC) matches snake in simulation and is not implemented.

## Performance per step (relative to cuDNN, CUDA Graph, each row from the same bench run)

| Version | `perf` | `perf_causal` | `perf_d64` | `perf_8k` | `perf_causal_bh16` |
|---|---|---|---|---|---|
| lab 41 solution | ≈ 0.96× | ≈ 1.04× | ≈ 0.96× | ≈ 0.94× | ≈ 0.73× |
| step 1 | ≈ 0.97× | ≈ 1.04× | ≈ 0.95× | ≈ 0.95× | ≈ 0.78× |
| step 2 | ≈ 1.02× | ≈ 1.04× | ≈ 1.02× | ≈ 0.97× | ≈ 0.69× |
| step 3 | ≈ 1.03× | ≈ 1.27× | ≈ 1.01× | ≈ 0.99× | ≈ 0.97× |
| FA4 | ≈ 1.17× | ≈ 1.31–1.33× | ≈ 1.15× | ≈ 1.10× | ≈ 1.14× |

`./lab selftest 43 --perf` (target 0.9) passes (re-measured after the review fixes): `perf` 1.02×, `perf_causal` 1.27×, `perf_d64` 1.02×,
`perf_8k` 0.98×, `perf_causal_bh16` 1.03× (FA4 in the same run 1.16×, 1.32×, 1.15×, 1.11×, 1.14×). The tightest is `perf_8k`.

Nsight Compute (locked clocks) `sm__cycles_active.max / avg`: `perf_causal_bh16` steps 1/2/3 = 1.33/1.52/1.04 (FA4 1.07);
`perf_causal` 1.16/1.27/1.03 (FA4 1.09). Tensor Core busy: step 3 `perf` 68% (FA4 83%; here FA4 combines persistent, 2-CTA and other tuning, and lab 42 measured that 2-CTA alone does not pay off, so the gap cannot be attributed to 2-CTA),
`perf_causal` 63% (FA4 68%), `perf_causal_bh16` 66% (FA4 75%). The remaining gap is in per-block efficiency, not in scheduling.

Control experiments (the step 3 kernel with only the schedule changed, time relative to step 3): LPT + round robin 1.09/1.13 (`perf_causal`/`bh16`); not persistent + LPT
(FA4's causal approach) 1.07/0.97; plus FA4's head sections 1.08/0.98; persistent snake + sections 1.27/1.26. On
`perf_causal`, step 3 reads 535 MB from DRAM (steps 1/2 and FA4 about 201 MB), but HBM throughput is only about 15%, so time is not affected.

## Verification

- `./lab check 43 --solution`: 3 steps × 37 cases all pass (`many_one_block` added after the review; `many_one_block` and `multi_tile*` each repeated 5 times, all pass) (including the new `bh1_causal`, `bh1_long`, `uneven_causal`,
  `multi_tile` (450 tiles), `multi_tile_d128` (480 tiles, V = 1), `multi_tile_causal` (640 tiles, jump_max)).
- Skeleton: in a temporary copy, fill TODO(1), (2) (first only the host side: on the GPU it stops at TODO(2) and does not hang), (3) in turn: `--step 1` passes and step 2
  reports TODO(2); `--step 2` passes and step 3 reports TODO(3); everything passes. Afterwards the skeleton is restored byte-identical.
- `./lab sass 43 --solution`: all 24 kernels REG 128 (at launch; `setmaxnreg` redistributes), STACK 0.
- `./lab sanitize --solution` (all steps): memcheck, racecheck, synccheck on `multi_tile`, `two_tiles`, `bh1_causal`:
  0 errors (after the review, all three tools were also run on `many_one_block`: 0 errors). racecheck cannot see TMA and TMEM accesses; the correctness of the handover across tiles rests on the argument in lesson §5.2–5.4 and on the `multi_tile*` cases.
- `./lab selftest 43` and `./lab selftest 43 --perf` pass; `python -m pytest -q` 12 tests pass.
- `bench` with the skeleton filled in: student version and solution run at the same speed (`perf_causal_bh16` both ≈ 1.02×).

## Known limitations

- Snake is static and estimates work by "block count"; when work varies a lot within one round (very small B·H) the cancellation is incomplete. On `perf_causal_bh16`, hardware greedy
  (not persistent + LPT) is about 3% faster than it.
- tile 0's `X` blocks are computed anyway (about 1/(2(p+1)) extra work for causal).
- No 2-CTA (lab 42) and no FP8 (lab 44).
- Wave quantization (tile count not a multiple of 148) is not handled.
