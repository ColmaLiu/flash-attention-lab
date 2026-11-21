# 01_softmax reference solution notes

Lesson: [course/01_softmax/README.md](../../course/01_softmax/README.md). This file explains why the solution is written this way, how to verify it, and what limitations remain. All ratios are ratios of medians from the same CUDA Graph timing run (`torch time / time of this version`); no absolute times are given.

## Design choices

### Shared by all steps

- **One `exp_shifted(x, m) = ex2.approx.ftz((x - m) * log2 e)`.** Subtract first, then multiply: `x - m` is exact when `x` and `m` are close (Sterbenz), and the multiplication introduces only one rounding. We tried `fmaf(x, log2e, -m * log2e)`, common in the FlashAttention source: it rounds `m * log2e ≈ 7200` to FP32 with an error `δ(m)` of at most 2⁻¹², so the result is `e^(x-m) · 2^(-δ(m))`. In steps 1 and 2 the whole row has one shift, so this factor cancels in `e / l` and `shifted` still passes. Steps 3 and 4 merge partial states with different shifts `m_t`, and `e^(m_t - M)` cannot cancel each one's `2^(-δ(m_t))`; the measured relative error on `shifted` is about 2.5–4e-4, above the 1e-4 tolerance. Attention logits are generally not large (`δ` about 1e-6) and the output is BF16, so FA's FMA form is fine. This lab's output is FP32, so all four steps subtract first, then multiply.
- **Write `ex2.approx.ftz.f32` directly** instead of `exp2f`: to handle subnormal results correctly, `exp2f` adds an `FSETP` and two predicated `FMUL`s. `ftz` flushes results below 2⁻¹²⁶ to 0, which is negligible relative to the 1e-8 absolute tolerance.
- **`-inf` handling is concentrated in two places**: in `merge()`, when the new maximum is `-inf`, return the empty state; in step 4, when a thread's local maximum is `-inf`, use 0 as the shift. Both correspond to `Check_inf` / `max == -INFINITY ? 0 : max` in the FlashAttention source.

### Steps 1–3: one block (256 threads) per row

- Thread `t` handles columns `t, t+256, …`, contiguous within a warp, so memory access is coalesced.
- The tree reduction in step 1 needs `blockDim` to be a power of 2: 8 levels, one `__syncthreads()` per level, plus one more at the end to protect the reuse of `smem`.
- Step 2 first does a butterfly within each warp, then **every warp** reduces the 8 partial results, so all threads get the result directly without a second broadcast through shared memory.
- Step 3 updates element by element with `s = merge(s, {x, 1})`, two `ex2` per element. We tried a branchy form (when `x > m` compute only `e^(m-x)`, otherwise only `e^(x-m)`) that halves the exp count, and measured it as about 1.6–2.9 times slower on `rows4096`: the branch turns the loop into a dependency chain, the compiler no longer issues later loads early, and in SASS only 1–2 `LDG` are issued at a time, while the branch-free form unrolls the loop and issues 8 `LDG` back to back. This kernel is limited by memory traffic in flight, not by MUFU throughput, so we keep the branch-free form.

### Step 4: the whole row in registers

- `kThreads` threads handle one row, and each thread has `kVecs` chunks. Each chunk is `V` consecutive floats (`float4` for `V = 4`). Chunk `k` starts at column `(k * kThreads + t) * V`, so one warp instruction covers `32 * V` consecutive floats.
- All chunks are read into registers before any computation, so `kVecs` loads are in flight at once.
- Each thread first computes its local `(m, l)` and turns `v` in place into `e^(x - m_local)`; one `row_merge` gives the whole row's `(M, L)`; the output multiplies by `e^(m_local - M) / L`. This way each element does only one `ex2`, there is only one reduction (instead of two, for max and sum), and it is the same trick as FA's deferred rescale of the output accumulator.
- Configuration choice: at most 32 floats per thread (no spill to local memory); short rows (≤ 256 chunks) use one warp per row and 4 rows per block; long rows use one block per row with 128–1024 threads. Beyond 32768 columns, fall back to the online kernel of step 3.
- `__ldcs` / `__stcs` (evict-first): x and y are each touched only once.
- When `cols % 4 != 0` or a pointer is not 16-byte aligned, use the scalar version with `V = 1`, which is correct but slower.

## Verification

```bash
./lab check 01 --solution
./lab selftest 01 --perf
./lab sanitize 01 --solution --step N --case tiny|masked_short|tail --tool memcheck|racecheck|synccheck   # N = 1..4
./lab sass 01 --solution          # LOCAL:0 for all kernels
```

- `masked_short` and `tiny` cover warp-per-row; `tail` covers block-per-row (512 threads, scalar path). All three sanitizers report no errors on all four steps.
- Skeleton check: replace the skeleton's TODO(1..n) with solution code one at a time; `./lab check 01 --step n` passes and `--step n+1` reports `TODO(n+1)`. Restore the skeleton after the check.

## Performance change per step

| step | rows4096 `[16384, 4096]` | rows32768 `[4096, 32768]` | rows256 `[262144, 256]` |
|---|---|---|---|
| 1 three-pass, tree reduction | ≈ 1.9× | ≈ 0.6–0.75× | ≈ 0.07–0.12× |
| 2 warp shuffle | ≈ 1.9× | ≈ 0.75× | ≈ 0.18× |
| 3 online | ≈ 2.0× | ≈ 1.0× | ≈ 0.15× |
| 4 registers + float4 | ≈ 2.4× (84% of nominal HBM) | ≈ 1.2× (68%) | ≈ 1.08× (84%) |

(torch itself reaches about 35%, 56% and 77% of nominal HBM peak on the three cases. Under `--cold` the step 4 ratios barely change: ≈ 2.25×, 1.22×, 1.07×.)

Why (Nsight Compute, `dram__bytes_read.sum` and `lts__t_sectors_srcunit_tex_op_read.sum`, in units of the size of x):

| step | rows4096 DRAM reads | rows4096 L2 reads | rows32768 DRAM reads | rows32768 L2 reads |
|---|---|---|---|---|
| 1 | 1.0 | 2.7 | 3.0 | 3.0 |
| 3 | 1.0 | 1.8 | 2.0 | 2.0 |
| 4 | 1.0 | 1.0 | 1.0 | 1.0 |

- 16 KB rows: the rows resident at once take about 148 × 8 × 16 KB ≈ 19 MB, so rereads hit in L2. Three passes and one pass have the same DRAM traffic; the difference is only L2 bandwidth and instruction count.
- 128 KB rows: the rows resident at once take about 148 × 8 × 128 KB ≈ 155 MB, more than the 126 MB L2, so rereads go back to HBM. Here online (step 3) reads x one fewer time, a clear gain.
- Short rows of 256 columns: in steps 1–3 each of the 256 threads has only 1 element, and the time goes to block scheduling and the reduction's barriers; step 4 switches to one warp per row and has no barriers.
- Step 4 on wide rows reaches only ≈ 68%: 50 registers per thread × 1024 threads means one SM holds only 1 block (`sm__warps_active` ≈ 43%), so the load, reduction and store phases cannot overlap with other blocks.

## Known limitations

- Step 4 on `rows32768` is limited by "one row per SM". Possible improvements: split a row across 2 CTAs of a cluster and merge `(m, l)` through distributed shared memory (the mechanism of lab 33), or put half a row in shared memory so each SM holds two rows. These belong to later labs and are not done here.
- Beyond 32768 columns, fall back to step 3 (x read twice); when the column count is not a multiple of 4, use scalar memory access.
- The "vs torch" ratio occasionally jumps on the slow versions: on the same input, torch's `rows4096` sometimes measures about 1.65 times the time (a bimodal behavior reproducible across processes), which inflates the ratios of steps 1–3. Step 4 and its perf gate result are stable across many runs.
