# 41_fa4_softmax reference solution notes

## Design choices

**The pipeline structure is taken over unchanged from lab 40 step 3.** 16 warps (2 × softmax WG, correction WG, MMA, TMA, 2 idle), the TMEM layout, the single K/V ring,
all mbarriers and parities, and `setmaxnreg` 192/80/48 are all unchanged; the lab 40 step 1–2 configurations (P through shared memory, no correction WG) were removed.
This lab changes only the arithmetic that computes P in softmax and one condition in correction, so the differences among the three steps can all be attributed to softmax.

**Software `exp2` (`exp2_poly2`).** The same approach as `ex2_emulation_2` in FA4's `utils.py`: clamp with `max(x, −127)`, add `x + 1.5·2^23` rounded down
(`__fadd2_rd`) to put ⌊x⌋ into the low bits, `f = x − ⌊x⌋` exactly, 3-step Horner, and splice the exponent with `bits(p) + (bits(r) << 23)` (ptxas fuses it into `LEA`).
It uses the CUDA 12.8 `float2` intrinsics (`__ffma2_rn`, `__fadd2_rn/rd`) instead of inline PTX. f = x − n is written as `__ffma2_rn(n, (−1, −1), x)`: written as `x + (−n)`, `make_float2(-n.x, -n.y)` needs two scalar negations, 2 more instructions per pair (corrected after review). The intrinsics are more readable, and the SASS is the same as the inline PTX version
(`FFMA2`, `FADD2`, `FADD2.RM`). The coefficients were computed by us with numpy (a Lawson iteration for the relative-error minimax, with c0 fixed to 1),
not copied from upstream's `POLY_EX2`; the two differ in the 3rd significant digit, and the maximum errors are of the same order.

- c0 must be exactly 1: x = −∞ is clamped to −127, so f = 0 and the result's bit pattern is `bits(c0) − (127 << 23)`, which is +0 only when c0 = 1.0
  (c0 = 0.99999 gives NaN; confirmed by a mutation test).
- The degree is chosen from the unit roundoff 2^−11 of FP16 P: degree 3 (error 2^−13.5). Degree 2 (2^−8.9) is enough for BF16 but not for FP16; there is no version that
  "chooses the degree by dtype".

**Mix ratio.** In each 32 columns (one `tcgen05.ld`), `kPolyPairs = 3` of the 16 pairs use the polynomial (φ = 18.75%), spread evenly at compile time by `use_poly`
(q = 5, 10, 15). First sweep (old version, f = x + (−n)) with k = 0, 2, 3, 4, 5, 6, 8, 16: k = 3 was best, and k ≥ 4 spilled. After review, f was changed to one FFMA2, and
k = 3/4/5 were measured again: `perf` 0.90/0.89/0.89×, `perf_d64` 0.93/0.95/0.95×, differences within noise; k = 4 and 5 still spill 8 bytes in 4 and 6 kernels,
so we keep k = 3, which does not spill. Because spills start at k = 4, the sweep cannot test the balance point φ ≈ 0.39 of the model in lesson §3.6. FA4's `ex2_emu_freq` configurations correspond to
12.5–19%. We did not try FA4's "do not emulate the first and last fragments"
(`ex2_emu_start_frg`) variant.

**Packed `FFMA2`/`FADD2`.** Computing x (`s·scale_log2 − base`) and the sum also use packed instructions (four `float2` partial sums). k = 0 (all MUFU + packing)
is on par with lab 40, which shows that packing itself is not the source of the gain; its role is to free issue slots for the polynomial.

**Conditional rescale (step 3).** The m of softmax is a "reference", not the true max: only when `(m_new − m)·scale_log2 > τ = 8` does it move and give
α = 2^((m − m_new)·scale_log2); otherwise α = 1. In block 0, m = −∞ makes the difference +∞, so it always moves; if everything is −∞, NaN > τ is false, α = 1, and no
NaN appears (this does not happen in practice: block 0 always has visible keys). LSE is computed with the reference m (`m·scale + ln ℓ`), which is exact for any reference. Correction uses
`__any_sync(α ≠ 1)` to decide per warp (`tcgen05.ld/st` are `.sync.aligned`), and still arrives on `p_ready` when it skips.
This is the same as FA4's `SoftmaxSm100.update_row_max` (`rescale_threshold`) and `correction_loop` (`vote_ballot_sync(scale < 1.0)`).
τ = 8 satisfies P ≤ 256 < 65504 for FP16 and x ≤ 127 for the polynomial.

Control: keeping only the vote with τ = 0 gives `perf` ≈ 0.91× and `perf_d64` ≈ 0.94× (step 2 is 0.89×, 0.94×; τ = 8 is 0.97×, 0.96×).

**New patterns (`labkit/kinds/attention.py`, only new branches added).** `growing_max`: q_i = a_i u + noise, k_j = g(j) u + noise,
g(j) = 2j/128; each row's max rises by about 1.4–4.3 per block (log₂ units), never more than τ in a single block, but more than τ in total; `jump_max`: g jumps to 200 at key 3n/5 + 50 (the middle of a block),
a jump of about 145–430. a_i/√d ∈ [0.5, 1.5) differs per row, so the rows of one warp get corrected in different blocks. The pattern list in `docs/authoring.md`
now includes these two names.

## Performance of each step (relative to cuDNN, CUDA Graph, GPU 7, same bench run)

| Version | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| Lab 40 step 3 | ≈ 0.85× | ≈ 0.94× | ≈ 0.80× | ≈ 0.83× |
| Step 1 (all polynomial) | ≈ 0.58× | ≈ 0.69× | ≈ 0.61× | ≈ 0.56× |
| Step 2 (k = 3) | ≈ 0.89× | ≈ 0.99× | ≈ 0.94× | ≈ 0.87× |
| Step 3 (+ τ = 8) | ≈ 0.97× | ≈ 1.03× | ≈ 0.96× | ≈ 0.93× |
| FA4 | ≈ 1.15× | ≈ 1.33× | ≈ 1.13–1.15× | ≈ 1.03–1.10× |

Nsight Compute (`perf`): Tensor Core busy, lab 40 57% → step 1 40% → step 2 60% → step 3 65%; `pipe_xu` 57% → 0.3% →
49% → 53%; `smsp__inst_executed_pipe_xu.sum` for steps 2–3 is 81% of lab 40 (104/128 for P + α), which confirms the mix ratio.

`perf_target` is 0.85 (the worst case, `perf_8k`, is about 0.93×, leaving margin for cuDNN variation).

## Error (against the FP64 oracle, same input)

`exp2` itself (exhaustive over all FP32 inputs in [−127, 8] on the GPU): `MUFU.EX2` 1.46·10⁻⁷, degree-3 polynomial 8.57·10⁻⁵ (= the error of the coefficients themselves on [0,1);
range reduction and splicing add no error), degree 2 2.05·10⁻³, degree 4 2.97·10⁻⁶; −∞ and −127 give +0, and (−127, −126) gives non-negative subnormals.

Attention output: the maximum error of O is nearly the same for the four variants (dominated by the 16-bit rounding of O); the maximum error of LSE rises from about 10⁻⁶ with all MUFU to
10⁻⁵–1.3·10⁻⁴ (≈ the relative error of the polynomial), still an order of magnitude below the LSE tolerance 10⁻³. Per-case numbers are in lesson §7.3. The tolerances were not changed.

## Verification

- `./lab check 41 --solution`: 3 steps × 30 cases all pass (including the new `peaked_fp16`, `growing`, `growing_causal`,
  `jump`, `jump_causal`, and `tile1_position`, `causal_n_bound`, which were added after the lab 40 review).
- Mutation tests (in a temporary copy; afterwards the skeleton was restored byte for byte): round-to-nearest for the magic-number add → LSE out of tolerance (`one_block`, `medium`, etc., error
  ≈ 2–3·10⁻³); c0 = 0.99999 → cases with a mask produce NaN; no clamp → `n1` fails; condition compared with the previous block's max → `growing` (FP16)
  produces NaN (`growing_causal` is BF16, does not overflow, and passes); the reference never moves → `jump`, `jump_causal`, `growing` produce NaN; using
  lane 0's α after the vote → `growing`, `growing_causal`, `jump_causal` fail.
- Skeleton: after replacing TODO(1), (2), (3a), (3b) one at a time with solution code, `--step n` passes and `--step n+1` reports the TODO; with only 3a filled in, step 3 stops at
  TODO(3) (the trap in the correction warp) and does not hang.
- `./lab sass 41 --solution`: STACK is 0 in all 24 kernels; step 1 has 1 `MUFU.EX2` (α) and 320 `FFMA2` per kernel;
  steps 2–3 have 105 `MUFU.EX2` and 112 `FFMA2` per kernel.
- `./lab sanitize --solution`: memcheck, racecheck and synccheck report 0 errors on `two_tiles`, `growing_causal` (step 3) and `causal_tail` (steps 1, 2);
  racecheck cannot see TMA/TMEM accesses, and the synchronization is exactly the same as lab 40 (lab 40 lesson §5.6).
- `./lab selftest 41` passes; `./lab selftest 41 --perf` (target 0.85) passes, with that run measuring `perf_causal` 1.03×, `perf_d64` 0.96×,
  `perf_8k` 0.95× (`perf` ≈ 0.97×, see the table above).
- `python -m pytest -q`: the tests related to this lab pass; the only failure comes from lab 42, which was being written at the same time (`course/42_fa4_2cta/README.md` did not exist yet).

## Known limitations

- The mix ratio is k = 3 for D = 64/128 and for causal/non-causal alike, without separate tuning; each sweep point was measured only once.
- The softmax register limit of 192 limits k; we did not try changing the register allocation (for example lowering correction to 64) to allow a larger k.
- Reliance on in-order execution of MMAs from the same thread, tile 0 computing one extra block for causal, etc., are the same as lab 40.
- Conditional rescale almost never corrects on random data; for inputs whose max keeps rising (`growing_max`) it still corrects once every 2–5 blocks.
