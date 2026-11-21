# 21_pipeline_gemm solution notes

## Design choices

- **Supported domain**: A and B are BF16, C is FP32, `K % 8 == 0` (a 16-byte `cp.async` never crosses a row end); M and N are arbitrary. Out-of-bounds chunks are zero-filled with the `cp.async` `src-size = 0`, and the source address is replaced by the matrix base address, so no out-of-bounds data is read.
- **One kernel template, three configurations**: `Config<BM, BN, BK, WarpsM, WarpsN, Stages>`. `compute_stage` (ldmatrix + mma) and `store_c` are the same as in lab 20, only parameterized; all shared tiles use `lab::swizzle_chunk<BK/8>`.
- **step 1 (double buffering)**: written separately as `gemm_double_buffer`, so the learner first sees the simplest "wait → barrier → issue next tile → compute this tile". Each K tile has one `wait_group 0` and one `__syncthreads()`; the barrier covers both RAW (the new tile is visible to all warps) and WAR (all readers of the previous tile have left, so its stage can be overwritten).
- **step 2 (ring buffer)**: `gemm_multistage`, with the invariant "commit group g is exactly tile g". Empty groups are committed when K runs out, so every round is `wait_group<S−2>`, with no special case for the tail. S = 3 (48 KB), two CTAs per SM.
- **step 3 (tuning)**: BK = 64 (128-byte rows, 8-chunk swizzle), 4 warps of 64×64, 3 stages (96 KB, two CTAs per SM; about 254 registers × 256 threads fills the register file exactly), grouped rasterization `kGroupM = 8`.
- **shared memory carveout**: besides the dynamic shared memory limit, `allow_smem` also sets `PreferredSharedMemoryCarveout = MaxShared`. On this machine, without it, the 64–96 KB configurations occasionally had whole runs about 2× slower (bimodal samples); with it, step 3 is stable. The presumed cause is that the carveout chosen by the driver fits only one CTA; this was not confirmed further.

## Performance per step (vs cuBLAS BF16→FP32, CUDA Graph)

| step | 4096³ | 8192×8192×4096 | % of mma.sync upper bound (≈560 TFLOP/s) |
|---|---|---|---|
| lab 20 step 3 | ≈ 0.17× | ≈ 0.17× | ≈ 45% |
| 1 double buffering | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 2 three-stage pipeline | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 3 tuned | ≈ 0.28× | ≈ 0.26× | ≈ 72% |

- step 1 → 2 gives almost no gain. By the Little's law estimate (lesson §3.1), two CTAs per SM, each one tile ahead, are already close to the amount of data needed in flight. The 4-stage (64 KB) version was not faster on this machine and its samples varied widely (see above), so step 2 uses 3 stages.
- The step 3 gain comes from BK = 64 and the 64×64 warp tile (fewer `ldmatrix` per mma, fewer barriers). Configuration experiments (4096³): `128,128,64,2,4,3` ≈ 0.27×, `128,256,32,2,4,3` ≈ 0.19× (170 registers, one CTA per SM), `128,128,64,2,2,4` ≈ 0.23× (128 KB, one CTA per SM).
- Grouped rasterization: on `wide`, DRAM reads drop from about 716 MB → 556 MB and the L2 hit rate rises from about 79% → 86%, with no change in time; the kernel is bound by the Tensor Cores (NCU `sm__pipe_tensor_cycles_active` about 79%, DRAM throughput about 5%). It stays for teaching: the lesson uses it to show "optimize the actual bottleneck".
- `perf_target` is set to 0.22; the reference solution reaches about 0.26–0.28, and on `wide` our kernel varies by about ±10% per run (cuBLAS is stable), so the target leaves a margin.

## Verification

- `./lab check 21 --solution`: all steps and all cases pass.
- compute-sanitizer: memcheck / racecheck / synccheck report no errors on `tail`; racecheck and synccheck report no errors on `aligned` (16 K tiles, the ring buffer wraps several times).
- `./lab sass 21 --solution`: no spill (LOCAL 0); SASS contains `LDGSTS`, `LDSM`, `HMMA`.
- Skeleton: TODOs filled in step by step; step n depends only on TODO(1..n), and the completed version matches the reference performance; the skeleton has been restored.

## Known limitations

- The remaining ≈30% to the `mma.sync` upper bound: one CTA barrier per tile, the `ldmatrix`→mma dependence, and the tail effect of 1024 CTAs over 296 resident slots in 3.46 waves. No fragment double buffering, persistent kernel or split-K.
- The write-back is still an element-wise FP32 store.
- Only BF16 inputs and FP32 output are supported.
