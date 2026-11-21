# 30_tma_mbarrier solution notes

## Design choices

- **Supported domain**: FP32, `rows % 4 == 0`, `cols % 4 == 0`. This is a hard TMA requirement: the row pitches of x and y (`cols·4` and `rows·4` bytes) must both be multiples of 16. Tail tiles are left entirely to the TMA: loads fill out-of-bounds elements with 0 (and count them in the tx bytes), stores skip them, and the kernel has no bounds checks. Not extended to BF16: transposing 2-byte elements needs half-word shuffling in registers (`prmt`), which has nothing to do with the mechanism of this lab.
- **Primitives**: `include/lab/mbarrier.cuh` and `include/lab/tma.cuh`; the solution only `#include`s them. The address conversion function is named `lab::shared_addr` to avoid a redefinition with `lab::smem_u32` of `mma_sync.cuh` in the same translation unit.
- **step 1**: one 32×32 tile per CTA, 4 warps. Shows the minimal protocol: init + `fence.mbarrier_init` + `__syncthreads`; `arrive.expect_tx` + TMA load; `try_wait.parity 0`; after the generic writes to `out`, every thread does `fence.proxy.async` and then `__syncthreads`; TMA store + commit + `wait_group 0`. The transpose is scalar and has 32-way bank conflicts, deliberately left for step 3.
- **step 2**: persistent. One thread elected with `elect.sync` in warp 0 is the producer; `kRows/32` consumer warps each own 32 rows of the tile, each stores its own 32×32 output box and manages two out buffers with its own bulk group (`wait_group.read 1`). So after initialization the kernel has no CTA-level barrier: `full[s]` (count 1 + tx) and `empty[s]` (count = number of consumer warps; lane 0 of each warp arrives once after `__syncwarp`) are the only synchronization. The producer waits for parity 1 on `empty` in its first lap, relying on "the previous phase counts as complete", the same as CUTLASS `PipelineTmaAsync`.
- **step 3**: both maps, x and y, use 128B swizzle; each lane handles two 4×4 blocks `(R, C) = (l%8, (l%8 + l/8 + 4t)%8)`: float4 reads, register transpose, float4 writes. Verified one by one with a script: in groups of 8 lanes, the physical chunks of both reads and writes are pairwise distinct. Configuration `Config<256, 2, true>` (box 32×256, 2 stages, 9 warps, 128 KB, 1 CTA per SM).
- **No proxy fence for generic read → TMA overwrite**: the consumer reads `in[s]`, hands it to the producer through an mbarrier arrive/wait, and only then is the TMA issued, consistent with CUTLASS's consumer_release. For generic write → TMA store read, every writing thread executes `fence.proxy.async.shared::cta`.
- **`bulk_wait_group<0>` before exit** (not `.read`): a conservative choice that makes sure all stores complete; the cost is one extra write latency at the end of each consumer warp, negligible for a persistent kernel. Lab 31 uses CUTLASS's `.read` form; the lessons explain this in both places.

## Performance per step (GPU 4, CUDA Graph, each step measured 2–3 times)

| step | `big` GB/s | `wide` GB/s | `vs copy` | Notes |
|---|---|---|---|---|
| 1 | ≈ 1950 | ≈ 1950 | ≈ 0.30 | Limited by bank conflicts (about 31 extra wavefronts per LDS) |
| 2 | ≈ 2100 | ≈ 2100 | ≈ 0.33 | Same as above: the pipeline gains only about 8%; the transpose itself is the bottleneck |
| 3 (`<128,4>`, earlier measurement) | ≈ 5600 | ≈ 5600 | ≈ 0.87 | Only swizzle + 4×4 transpose added |
| 3 (solution `<256,2>`) | ≈ 6190 | ≈ 6095 | ≈ 0.96 / 0.95 | |
| `copy` (`y.copy_(x)`) | ≈ 6430 | ≈ 6430 | 1 | Practical HBM upper bound |
| `torch` (`y.copy_(x.t())`) | ≈ 1860–1905 | ≈ 1885–1905 | ≈ 0.29 | |

- On the same machine, lab 00 step 4 reaches about 5910 GB/s (`vs copy` ≈ 0.92), so TMA leads in bandwidth by only about 5% (about 2% with `--cold`). The lesson states this honestly and describes the gain as "1 issuing thread and 9 warps reach the same bandwidth".
- `perf_target` is `{"baseline": "copy", "ratio": 0.85}`: `copy` and the solution are both HBM-bound, so the ratio is stable. `torch` is not suitable as a gate: in an earlier measurement (GPU 3), step 1/2 reached only ≈ 1030/1060 GB/s and torch jumped between about 1.0 and 1.8 TB/s, while step 3 and copy did not change. Hypothesis (unverified): the GPU was at a lower SM clock that time, and kernels limited by bank conflicts / latency (the torch transpose, step 1/2) slowed down with the SM clock, while HBM-bound kernels were not affected. During this measurement the SM clock was 1965 MHz (the maximum), and all results were stable.
- Tuning (`big`): `<32,4>` ≈ 5180, `<64,4>` ≈ 5650, `<128,2>` ≈ 5720, `<128,3>` ≈ 5720, `<128,4>` ≈ 5630, `<128,6>` ≈ 5600, `<192,3>` ≈ 5790, `<256,2>` ≈ 6200, `<256,3>` ≈ 5770, `<256,4>` ≈ 5750. More bytes in flight is not faster; the cause was not found.
- NCU (`big`): step 1/2 shared load bank conflicts ≈ 6.6e7 (LDS ≈ 2.1e6 instructions); step 3 load ≈ 7.3e5, store ≈ 1.0e6 (LDS/STS ≈ 5.2e5 instructions each). A different lane assignment barely changes the counts; the residual counts presumably come from arbitration with TMA accesses, not confirmed.

## Verification

- `./lab check 30 --solution`: 3 steps × 7 cases all pass exactly.
- `selftest 30 --perf` passes; the skeleton reports `TODO(1)` at make_plan.
- compute-sanitizer: memcheck / racecheck / synccheck on `tail` (all 3 steps) and racecheck on `ring` (steps 2, 3) report no errors.
- **Sanitizer blind spot** (experiment): after deleting the consumer's `fence_proxy_async`, the tests (`ring` × 3 runs) and racecheck all pass; after deleting the producer's wait on `empty`, `ring` gives wrong results, but racecheck still reports 0 hazards. racecheck does not track shared memory accesses by the async proxy (TMA). Lesson §4.7 and self-check question 6 use this finding.
- `./lab sass 30 --solution`: no spill; contains `UTMALDG`, `UTMASTG`, `SYNCS`.
- Skeleton: in a temporary copy, TODO(1), (2), (3) were filled in step by step; step n depends only on the first n groups of TODOs, and once filled in it runs as fast as the solution; the skeleton has been restored to its original state.

## Known limitations

- Only FP32, with rows and columns both multiples of 4.
- The tile order is simple row-major (`t → (t / tiles_c, t % tiles_c)`); DRAM page locality was not studied.
- Why 2 stages beat 3–4 stages was not found out.
