# 44_fa4_fp8 reference solution notes

## Design choices

**The pipeline structure is taken unchanged from lab 40 step 3.** 16 warps (2 × softmax, correction, MMA, TMA, 2 idle); the barrier protocol, the MMA issue order and the
TMEM column numbers all stay the same. This lab only changes the data type, so all three steps use the step 3 structure (including correction and `setmaxnreg` 192/80/48),
instead of splitting out lab 40's intermediate forms again.

**FP8 tile = one panel, swizzle span = D bytes.** D = 128: a row is 128 bytes, 128B swizzle; D = 64: a row is 64 bytes, 64B swizzle
(new `lab::make_smem_desc_kmajor_sw64`; V uses `lab::make_smem_desc_mnmajor_sw64`, added in lab 42, whose byte semantics apply to 8 bits as well).
TMA uses `UINT8`, box `[1][rows][1][D]`. The K-slice stride is 32 bytes (= 32 E4M3), the same byte stride as the 16-bit version.

**V is not transposed.** The `tcgen05` instruction descriptor allows MN-major for E4M3/E5M2 (comment in CUTLASS `mma_sm100_desc.hpp`;
FA4 also uses `OperandMajorMode.MN` for FP8 V). So there is no in-kernel transpose as in FA3 (Hopper), and no pre-transpose in `make_plan`
either (that would move work out of the timed region). The timing boundary is the same as lab 40.

**descale folded into scale.** `qk_scale = scale · q_descale · k_descale`, `scale_log2 = qk_scale · log₂e`, computed on the host;
the softmax code is the same as lab 40; `v_descale` is multiplied once per element in the epilogue.

**P is multiplied by 2⁸ and quantized to E4M3** (FA4's `max_offset = 8`): the `exp2` base is reduced by 8; `cvt.rn.satfinite.e4m3x2.f32` packs pairs, four per column,
into columns 64–95 of S; each TS-form `kind::f8f6f4` MMA reads 8 columns. ℓ carries the same 2⁸, O / ℓ undoes it automatically, and LSE subtracts 8·ln2.

**Two normalization quantities (step 3).** `l`: the FP32 Σp, used only for LSE; `lo`: the denominator of O. In step 2, `lo = l` (same as FA4); in step 3,
`lo = Σp̂` (the sum of the E4M3 values that actually enter P·V), so O becomes an exact convex combination. The first version also used Σp̂ for LSE, and the LSE error of `one_block`,
9.8e-3, exceeded 1e-3: LSE must describe the true softmax. The sum uses `cvt.rn.f16x2.e4m3x2` (exact) + mixed-precision `add.rn.f32.f16`
(one FHADD in SASS). The first version used `__half22float2` + FADD (HADD2.F32 + FADD per element), and step 3 was 6–16% slower than step 2 (most at D = 64);
after switching to FHADD it is no longer slower.

**O staging lives on Q⁰ + Q¹.** An FP8 Q tile is only 128·D bytes, while a BF16 O tile needs 2·128·D; the two Q buffers together are exactly one O tile.
When `o_full[0]` completes, all QKs (including QK¹) have completed (they were issued before the last PV⁰), so both Q buffers are free. Correction writes tile 0 first;
in each warp, lane 0 does `bulk_wait_group_read<0>`, then `__syncwarp`, and then the warp writes the same rows of tile 1.

**The empty asm in `align_smem`.** The first version had an 8-byte STACK in the step 2–3 kernels: the 1024-byte-aligned smem base computed in the prologue
was saved to local memory and loaded back with 4 LDL in the TMA branch (after `setmaxnreg.dec` to 48). Changing the register budget to (184, 80, 64),
(192, 72, 56), (192, 64, 64), (184, 96, 48), (192, 88, 40) did not remove it (some made it worse). Recomputing the base after `setmaxnreg.dec` with an empty asm that is opaque to the compiler
(a few integer instructions) brings STACK to 0 in all 12 kernels. The cause (ptxas's allocation strategy on either side of `setmaxnreg`)
was not investigated further.

**Number of K/V ring stages**: D = 128 uses 8 stages (the same 128 KB as lab 40), D = 64 uses 12 stages. The difference between 4/6 stages and 8/12 stages on `perf` and `perf_d64`
is within noise (HBM is only about 2% used).

## Error and tolerance

A measurement script (not committed) compares O of every non-perf case with the FP64 oracle and records `max |err| / (atol + rtol·|expected|)` (the fraction of the
tolerance budget used):

| Method | (5e-2, 1e-1) | (2⁻⁴, 2⁻³) |
|---|---|---|
| step 2 (E4M3 P × 2⁸, ℓ = Σp) | 0.91 (`causal_peaked`) | 0.73 |
| step 3 (O uses Σp̂) | 0.94 (`causal_peaked`) | 0.75 |
| FA4 FP8 (`_flash_attn_fwd`) | 0.91 | 0.73 |
| P scale = 1 (`kPLog2 = 0`), kernel measured | 0.91 / 0.95 | 0.73 / 0.76 |
| P scale = 448, kernel measured | **1.19 / 1.17 (`peaked`)** | 0.95 / 0.94 |
| BF16 P (PyTorch simulation) | 0.06 | 0.05 |

The original tolerance (5e-2, 1e-1) left only 6% headroom for the solution and rejected P scale = 448, a reasonable variant, so it was changed to (u, 2u), where u = 2⁻⁴ is the unit roundoff
of E4M3. This value takes its magnitude from u and is calibrated by measurement (about 75% used); it is not a derived upper bound. Fault injection confirms that the looser tolerance still catches bugs (`./lab check`
stops at the first failing case): P byte swaps (0↔1 → `n1`; 2↔3 → `one_block`), fully reversed byte order, PV TMEM column stride ×2, P¹ using tile 0's
columns (`two_tiles`), D = 64 V SBO wrongly set to 1024 (`one_block`), causal mask seeing one key too many (`causal_pair`), LSE not subtracting 8·ln2 or P
not scaled (LSE of `n1`) all fail under both tolerances.

P scale = 1 and 2⁸ give the same error on random-data cases (keys with p < 2⁻⁶ contribute very little to ℓ). For this reason a new `long_tail` pattern was added (generation method
from the independent review: one hot key with logit≈8 per head, the rest ≈0, mean of V is 1) along with two cases: with P scale = 1, `long_tail` has error 0.61
(step 2) / 2.0 (step 3), `causal_long_tail` 0.44 / 0.99; with 2⁸, 5.7e-3 and 5.3e-2 / 3.7e-2.

**The descales all differ.** The review found that when all three descales were 0.25, mix-ups such as `scale·q_descale·q_descale` or `q_descale/lo` passed. Now
the stored values are Q×2, K×8, V×4 (descales 0.5, 0.125, 0.25; true values unchanged; K's stored values peak at about 40 ≪ 448). Both mix-ups fail on all 27
cases (steps 2/3); the FA4 FP8 baseline still agrees with the oracle.

**Tightened structured cases (`o_tol`, about 2–3× the measured error).** `uniform`/`causal_uniform` (1e-3, 4e-3); `two_tiles`,
`tile1_position`, `short_tail`, `position`, `ring`, `causal_position` 1.5e-2 (measured ≤ 6.9e-3); `ones_v` 2e-2 (7.8e-3),
`causal_ones_v` 4e-2 (2.0e-2); `long_tail` (1.2e-2, 4e-3). In addition, `ones_v_exact` and `causal_ones_v_exact` are checked only in step 3
(1e-5; step 3 measured ≈1e-6), so the step 2 case list explicitly excludes them. Fault "P truncated instead of rounded to nearest": 16 cases fail in step 2
(`position`-type, `ones_v*`, `peaked*`, `long_tail`, and so on), 6 in step 3 (`peaked*`, `causal_medium*`, `long_tail`):
the consistent normalization of step 3 cancels most of the truncation bias, which is visible only in rows with concentrated weight and long-tail rows.

## Performance per step (CUDA Graph, GPU 4, range of 2–3 runs)

| Version | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| step 2 vs FA4 FP8 | ≈ 0.68–0.69× | ≈ 0.71× | ≈ 0.73× | ≈ 0.68× |
| step 3 vs FA4 FP8 | ≈ 0.69–0.73× | ≈ 0.73× | ≈ 0.74× | ≈ 0.70–0.72× |
| step 3 relative to lab 40 step 3 | ≈ 1.23× | ≈ 1.15× | ≈ 1.12× | ≈ 1.24× |
| FA4 FP8 relative to FA4 16-bit | ≈ 1.27× | ≈ 1.12× | ≈ 1.05× | ≈ 1.37× |

Step 1 is a probe step (outputs S as [B,H,N,N]); the probe for the perf cases would need 4 GiB, so it is not timed.

Nsight Compute (`perf`): Tensor pipe ≈ 29% (lab 40 ≈ 57%), XU (MUFU) ≈ 58% (lab 40 ≈ 57%), DRAM ≈ 2%. This agrees with the chain model:
PV + QK halved (≈ 1100 → 550 cycles), softmax unchanged, 3450 / 2900 ≈ 1.19×. Step 3 is about 1–4% faster than step 2 (reproducible); the cause has not been found.

`perf_target`: 0.63 relative to `fa4` (FA4 FP8); the worst cases, `perf_8k` and `perf`, are at ≈ 0.69–0.72×, with single-run variation of about ±3%. cuDNN has no FP8 baseline.

## FA4 baseline

In the pinned version, the public `flash_attn_func` does not accept descale arguments (the original labkit passed `q_descale=` for e4m3, which raised `TypeError`,
while `bench` only catches `EnvironmentError_/RuntimeError/NotImplementedError`). For e4m3, `_fa4` in `labkit/kinds/attention.py` now uses
`flash_attn.cute.interface._flash_attn_fwd` instead (the same as upstream `benchmark_flash_attention_fp8.py`), with descale shape `[B, H]` FP32 and
O in BF16. Its O/LSE has been checked to agree with the FP64 oracle (errors in the table above; LSE ≤ 1e-4).

## Verification

- `./lab check 44 --solution`: 3 steps (8 + 27 + 29 cases) all pass; `./lab selftest 44` and `selftest 44 --perf` (0.63) pass;
- `./lab sanitize --solution`: memcheck, racecheck, synccheck on `two_tiles` (steps 1–3), `tile1_tail` (step 1), `causal_tail`
  (steps 2–3): 0 errors; racecheck cannot see TMA/TMEM, and the synchronization argument follows lab 40 §5.6;
- `./lab sass 44 --solution`: all 12 kernels have STACK 0, and there are 104 `UTCQMMA` (the FP8 form of tcgen05.mma);
- Skeleton: generated from the solution; after replacing the TODOs of steps 1..n with the solution step by step, `--step n` passes and `--step n+1` reports TODO(n+1); with only 1a filled in it
  stops at TODO(1) instead of hanging; after verification it is restored to the original skeleton (byte-identical).

## Known limitations

- Inherits all limitations of lab 40 (dependencies from in-order MMA execution within one thread, wasted blocks under causal, not persistent).
- Only per-tensor descale is supported; FA4 supports per-(batch, kv head) descale, and the block quantization of the FA3 paper is outside the scope of this lab.
- O is output only in BF16; E5M2 input is not supported (it needs only a change of the idesc format code and the input checks).
- The quantization error of Q/K/V is not counted by the checks (the oracle uses the dequantized inputs).
