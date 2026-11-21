# 10_simt_gemm reference solution notes

## Design choices

- **Step 1 maps `threadIdx.x` to N.** Writes to C are coalesced, reads of A are broadcasts, and reads of B span 32 rows per warp. With two K-major operands, whatever the mapping, the naive version always has one operand that is not coalesced; this mapping was chosen to match lab 00 (contiguous writes along x). This version is extremely slow (about 1% of cuBLAS), and the lesson warns the learner that bench takes a while.
- **Step 2 uses 1024 threads, one output per thread.** This makes "tiling" the only change, which is easy to compare with step 1. The `[32][33]` padding is carried over directly from lab 00, with no separate "with conflicts" step: the new topic of this lab is reuse, and the bank conflict analysis belongs to the layout design of steps 3 and 4.
- **Step 3 uses a strided assignment (`ty + 16i`, `tx + 16j`) and keeps K-major smem.** This keeps the K-major layout conflict-free too (`stride 9` is coprime with 32), so all of step 3's gain comes from register reuse, separate from step 4's "layout + vectorization". The skeleton provides the tile load function `stage_kmajor`; the learner writes only the outer product and the write-back.
- **Step 4 uses transposed storage, two 4×4 groups, padding 4.** The transpose makes the m/n for a fixed `kk` contiguous, which `LDS.128` requires; the two groups 64 apart put the `float4`s of the 16 values of `tx` in 256 consecutive bytes; padding 4 removes the 2-way conflict of the transposed write and keeps 16-byte alignment.
- **Tail handling lives in `load4` / `store4`.** When `K % 4 != 0` they fall back to scalar reads, when `N % 4 != 0` to scalar writes, and out-of-bounds M/N/K are filled with 0 or skipped. These details are not the focus of this lab and are provided in the skeleton.
- **No global → smem prefetch or multi-stage pipeline.** That is the topic of `21_pipeline_gemm`; this lab's solution covers only reuse, register tiling and vectorization.
- **BM = BN = 128, BK = 8, TM = TN = 8, 256 threads**: the classic SIMT SGEMM configuration. About 127 registers per thread, 2 blocks per SM; smem is about 8.4 KB per block and is not the limiting factor.

## Performance per step (CUDA Graph, relative to cuBLAS FP32 with TF32 off)

| Step | `square4k` | `wide` | Main reason |
|---|---|---|---|
| 1 | ≈ 0.01× | not measured (too slow) | 2 global loads per multiply-add, B not coalesced |
| 2 | ≈ 0.14× | ≈ 0.13× | 2 LDS per multiply-add, FFMA takes at most 1/3 of issue slots |
| 3 | ≈ 0.56× | ≈ 0.53× | 0.25 LDS per multiply-add; smem wavefronts and FFMA exactly balanced |
| 4 | ≈ 0.72× | ≈ 0.68× | LDS instructions cut to another 1/4, wavefronts down 25%, transposed write conflict-free |

cuBLAS picks `cutlass3x_sm100_simt_sgemm_..._128x128x16_..._3_tnn_align1` here, at about 35–39% of nominal FP32 peak. An independent-FFMA microbenchmark reaches about 97% of nominal peak, so the gap between GEMM and peak is inside the kernel, not in the clock.

`perf_target` is set to 0.6: this is an intermediate lab, and 0.6 is enough to show that register tiling and vectorization work, while leaving about 10% margin for the two perf cases (`wide` measures ≈ 0.68). Under `--cold` the ratio is about 0.84, higher than in graph mode. The cause has not been investigated: both are faster in cold mode, possibly because of the gaps between iterations.

## Verification

- `./lab check 10 --solution`: all steps and all cases pass.
- `./lab sanitize 10 --solution --step n --case tail` (n = 1..4) and `--case tail_vec`: memcheck, racecheck and synccheck report no errors.
- `./lab sass 10 --solution`: no local memory, no spills.
- Replace the skeleton's TODOs with solution code one at a time: after filling the nth, `--step n` passes and `--step n+1` reports `TODO(n+1)`.

## Known limitations

- Global memory latency is not hidden (no prefetch, no double buffering), and each SM has only 16 warps; this is the main gap to cuBLAS, left for lab 21.
- The 4096³ grid has 1024 blocks; with 2 per SM that is about 3.5 waves, and the last wave is only half full; no tile shape or split-K tuning was done.
- The vector path of `load4` assumes the base addresses of A and B are 16-byte aligned (PyTorch allocations satisfy this).
