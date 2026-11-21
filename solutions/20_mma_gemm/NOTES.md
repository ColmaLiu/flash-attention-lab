# 20_mma_gemm solution notes

## Design choices

- **Supported domain**: A and B are BF16, C is FP32, `K % 8 == 0`. When K is a multiple of 8, every row is 16-byte aligned, so a 16-byte chunk is either entirely inside the row or entirely outside it, and `load_tile` can zero-fill whole chunks. Step 1 additionally requires `M % 16`, `N % 8`, `K % 16` to be 0: this step only practices fragment coordinates and does not handle tails, `make_plan` explicitly rejects other shapes, and in the spec step 1 runs only the aligned cases.
- **Step 1**: one 16×8 block per warp; each register reads one `uint32_t` (two adjacent BF16 values) from global. The fragment coordinate functions correspond directly to the tables in the PTX documentation, and `one_atom` (integer inputs, one mma) checks every coordinate exactly.
- **Steps 2/3**: a 128×128×32 CTA tile, 8 warps arranged 2×4, warp tile 64×32 (4×4 m16n8). The B tile stays `[n][k]` in shared memory, which is exactly the layout of the `.col` operand, so B also uses the non-transposed `ldmatrix.x4`, one covering two n8 blocks. `load_tile` first issues all global loads and then writes shared memory in one go, so each thread has 2 16-byte loads in flight.
- **Swizzle**: 64-byte rows (4 chunks) use `chunk ^ ((row >> 1) & 3)`. The same logical chunk of 8 consecutive rows falls into 8 different 16-byte slots, so each 8×8 matrix of `ldmatrix` takes one wavefront; 16-byte stores are also conflict-free. Reads and writes both go through the same `tile_offset`, so correctness does not depend on whether the swizzle is applied.
- The write-back writes global element by element straight from registers (with a bounds check per element). This is not the bottleneck (C is written only once); vectorization is left to later labs.

## Performance change per step (relative to cuBLAS BF16→FP32, CUDA Graph)

| step | 4096³ | 8192×8192×4096 | reason |
|---|---|---|---|
| 1 | ≈ 0.02× | ≈ 0.01× | 6 global reads of 4 bytes per mma, no reuse |
| 2 | ≈ 0.09× | ≈ 0.09× | shared memory reuse; but `ldmatrix` has 4-way bank conflicts |
| 3 | ≈ 0.17× | ≈ 0.17× | conflicts removed: shared load wavefronts drop to exactly 1/4 (NCU: about 200 million → 50 million, conflicts about 150 million → 10 thousand) |

Reference point: the `mma.sync` microbenchmark upper bound is about 560 TFLOP/s, about 0.36–0.39 times cuBLAS; step 3 is about 45% of that upper bound. The remaining gap is mainly the synchronous global→shared data movement (the topic of the next lab). `perf_target` is set to 0.12; the solution reaches about 0.165 on both perf cases.

## Verification

- `./lab check 20 --solution`: all steps and all cases pass (error 0 on the integer cases).
- compute-sanitizer memcheck / racecheck / synccheck: `atoms` for step 1, `tail` for steps 2 and 3, all with no reports.
- `./lab sass 20 --solution`: no spills; `mma_cta` uses about 125 registers; the SASS contains `HMMA` and `LDSM`.
- Skeleton: replacing the TODOs with solution code step by step, step n depends only on TODO(1..n), and once complete the performance matches the solution; the skeleton has been restored.

## Known limitations

- Supports only BF16 input and FP32 output (the header has an FP16 mma wrapper, which this lab does not use).
- With random inputs and large K, the error of Tensor Core accumulation grows roughly as K^1.5 (cuBLAS too), so the gemm tolerance adds a `2e-8·K^1.5` term for 16-bit inputs (see `labkit/kinds/gemm.py`).
