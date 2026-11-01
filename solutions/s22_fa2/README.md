# s22: two complete attention implementations

The default answer in `fa2_optimized.cuh` is CUDA C++ with CuTe headers for
compile-time layouts and instruction atoms. It uses `mma.sync`, `ldmatrix`,
`cp.async`, register softmax state, and register-resident probabilities passed
to the PV multiply. It supports the public contiguous **BNHD** input/output
layout, BHN natural-log LSE, FP16/BF16, D=64/128, arbitrary positive sequence
lengths, and causal/noncausal masks.

`wmma_readable.cuh` is the preserved, complete CUDA/WMMA teaching alternative.
It stages score and PV tiles through shared memory to expose each mathematical
phase. Set `CONFIG = {"readable_wmma": 1}` in this solution's `entry.py` to select
it, and rerun validation before measuring. This version has extra shared-memory
traffic and synchronization; it is not described as production FA2 performance.
The student entry never selects either answer automatically.

The optimized version masks **fragment coordinates**, zero-fills invalid rows
on asynchronous loads, and predicates final stores. During reverse causal
traversal, a query row may initially see only masked keys. Its maximum must stay
negative infinity until a valid key arrives; the rescaling and exponentials
explicitly handle that state. Each warp retains its query rows across QK,
softmax, and PV, so no cross-warp output reduction is needed.

## Build and compare

The solution declares a private include path to the bundled, versioned
`references/cutlass-sm100/include` tree. The student build remains plain CUDA/PTX
unless the learner explicitly selects optional headers. No Python CuTe DSL or
installed FlashAttention package is involved in this solution.

```bash
./lab check s22 --solution
./lab solution s22
./lab reference s22
./lab sanitize s22 --solution --tool memcheck --case 6
./lab sanitize s22 --solution --tool racecheck --case 12
```

The optimized independent library anchor is explicit cuDNN attention including
O and LSE. Both implementations are validated against the independent FP64
oracle before graph timing; the measured ratio, rather than the implementation's
name, determines how close a case is to the library. Inspect common perf cases
as well as causal and tail cases. No hardware-independent speed claim is made.

## Source provenance

The optimized fragment transformations and GEMM staging were adapted from
[FlashAttention v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793),
commit `5231d95fe13733fb534c01895f7ea88c6a6c7793` (BSD-3-Clause). Relevant upstream
files are `csrc/flash_attn/src/kernel_traits.h`, `utils.h`, `softmax.h`, and
`flash_fwd_kernel.h`. `LICENSE.flash-attention` and `AUTHORS.flash-attention`
retain upstream license and attribution. This local implementation adds BNHD
addressing, predicated transfers/stores, and causal/tail masks. It is not an
unmodified official kernel. The bundled CUTLASS revision and license are recorded
under `references/cutlass-sm100/`.
