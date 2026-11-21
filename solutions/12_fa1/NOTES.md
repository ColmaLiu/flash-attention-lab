# 12_fa1 reference solution notes

## Design choices

**Faithful to Algorithm 1 of the paper.** The outer loop runs over K/V blocks and the inner loop over Q
blocks; `O`, `l`, `m` live in HBM and are read once and written once per `(i, j)`; `O` in HBM is always
normalized (line 12 first multiplies by `l_i`, then divides by `l_new`). These inefficiencies are kept on
purpose: `22_fa2` swaps the loops, keeps `O` in registers and delays normalization, and learners can
compare with the same set of perf cases.

**One CTA per (batch, head)**, which is how the original FA1 kernel parallelizes (Section 3.2 of the FA2
paper). A CTA owns all rows of one head, so `O/l/m` in HBM are shared only within the CTA, and
`__syncthreads()` is enough to guarantee visibility: no cross-CTA synchronization and no atomics are
needed. `O` uses the output tensor directly; `l` and `m` use `lab::Scratch` (one `[B, H, N]` FP32 buffer
each), initialized at the start of the kernel as in line 2.

**Block size `Br = Bc = 64`, 256 threads, a 4×4 S tile per thread.** It does not use the paper's
`Bc = ⌈M/4d⌉` (227 at d = 64): S lives in registers, and `64×227` would give each thread too many scores
and spill; a block that takes the full 227 KB leaves only 1 CTA per SM; and this kernel is not limited by
HBM bandwidth (see below). `Br = Bc` also reduces the causal blocks to three kinds (fully visible,
diagonal, fully masked), and guarantees that a real query row is never fully masked in a block that is
processed.

**Shared memory layout.** `Kt[d][Bc+4]` (K transposed, a float4 reads 4 keys), `Vs[Bc][d]`,
`Qs[Br][d+4]`, `Ps[Br][Bc+4]`. The `+4` keeps 16-byte alignment and puts two groups of float4 reads 4
rows apart in the same warp into different banks. The transposed write of K_j has bank conflicts, but it
happens only once per j and is amortized over all Q blocks. At d = 64 / 128 the kernel uses 67 KB /
116 KB, which needs `cudaFuncSetAttribute` to lift the 48 KB limit.

**Row reduction.** The 16 threads of a row are 16 consecutive lanes of the same warp; shuffles with xor
1/2/4/8 complete the max/sum, and all 16 threads get the result.

**Order of HBM state reads and writes.** `load_state` (each thread reads the old `O/m/l` of its own 4
rows) sits at the position of line 8, that is, after the load of `Q_i` and before the barrier; in
`merge_state` only `tx == 0` writes `m/l` back. At least two `__syncthreads()` lie between them, so the
other 15 threads of the same row always read the old values first. Each element of `O` is always read and
written by the same thread, so there is no cross-thread dependency.

**Numerics.** Uses `expf` (accurate enough for the FP32 tolerance `atol 2e-5, rtol 1e-4`). A fully masked
row uses 0 as the subtrahend, and the row is not updated when `m_new = −∞`; with `Br = Bc` real rows never
trigger these two branches, but they keep the kernel correct when the block size changes.

**Step 1** is the `kSingleBlock = true` version of the same kernel: no initialization, no HBM state reads
or writes; it writes `O = P~V/l~` and `LSE` directly. `make_plan` rejects N > 64 for step 1. Step 2 and
step 3 are the same kernel (none of the step 2 cases is causal).

## Performance per step (relative to the `efficient` baseline, CUDA Graph)

| case | step | ratio | notes |
|---|---|---|---|
| `one_block` | 1 | ≈ 1.2× | only 6 CTAs, almost all launch overhead, meaningless |
| `perf` [4,4096,64,64] | 2 / 3 | ≈ 0.49× | about 33% of FP32 nominal peak; same under `--cold` |
| `perf_causal` | 3 (no block skipping) | ≈ 0.26× | same time as non-causal |
| `perf_causal` | 3 | ≈ 0.50× | with block skipping, 2080/4096 tiles, time about halved |
| `few_heads` [1,4096,16,64] | 3 | ≈ 0.06× | 16 CTAs, only 16 SMs working |

`perf_target` is set to 0.4 (relative to `efficient`); both perf cases have about 20% margin.

**Nsight Compute (`perf`)**: DRAM reads are about 44 times the size of Q+K+V, and writes about 67 times
the size of O, consistent with the estimate `Tc · (read Q + read O + write O)`; relative to the bench
time this is only about 15% of the HBM nominal bandwidth. `sm__throughput` is about 50% and
`sm__warps_active` about 22% (128 registers at d = 64, 2 CTAs per SM). So the SIMT version is
compute/latency-bound, not HBM-bound.

**Where the gap to the baseline comes from**: PyTorch's FP32 memory-efficient attention is CUTLASS
`fmha_cutlassF_f32_aligned_64x64_rf_sm80`. According to `DefaultGemmType<ArchTag, float>` in
`third_party/cutlass/examples/41_fused_multi_head_attention/gemm_kernel_utils.h`,
FP32 on SM80+ uses `OpClassTensorOp` with `OpMultiplyAddFastF32` (3xTF32); Nsight measurements show
tensor pipe instructions. It splits CTAs by Q blocks of 64 rows (`getBlocksGrid`), and the output
accumulators stay in registers. These are all topics of later labs, and this lab's solution does not
use them early.

## Verification

- `./lab check 12 --solution`: all cases of all steps pass.
- `compute-sanitizer` memcheck / racecheck / synccheck (`causal_tail` step 3, `tail` step 3,
  `rescale` step 2, `short_tail` step 1): 0 errors.
- Skeleton: after each TODO is filled with solution code, the corresponding step passes, and step n
  depends only on the first n TODOs.

## Known limitations

- Supports only FP32 and d ∈ {64, 128}.
- `Br` and `Bc` are fixed at 64 and not adjusted for d or the SRAM budget; at d = 128 only 1 CTA can
  reside per SM.
- The parallelism is B·H; when B·H is far below 148, most of the GPU is idle (this is the effect this
  lab demonstrates).
- K_j is reloaded and transposed for every j, and the loads of Q_i and O_i are not pipelined with
  computation (`cp.async` comes in lab 21).
