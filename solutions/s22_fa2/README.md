# s22: two complete attention implementations

The default answer in `fa2_optimized.cuh` is CUDA C++ with CuTe headers for
compile-time layouts and instruction atoms. It uses `mma.sync`, `ldmatrix`,
`cp.async`, register softmax state, and register-resident probabilities passed
to the PV multiply. It supports the public contiguous **BNHD** input/output
layout, BHN natural-log LSE, FP16/BF16, D=64/128, arbitrary positive sequence
lengths, and causal/noncausal masks.

`wmma_readable.cuh` is a complete CUDA/WMMA teaching alternative.
It stages score and PV tiles through shared memory to expose each mathematical
phase. Set `CONFIG = {"readable_wmma": 1}` in this solution's `entry.py` to select
it, and rerun validation before measuring. This version exposes intermediate
tiles through shared memory, which adds traffic and synchronization. Both
answers have independent validation paths.

The optimized version masks **fragment coordinates**, zero-fills invalid rows
on asynchronous loads, and predicates final stores. During reverse causal
traversal, a query row may initially see only masked keys. Its maximum must stay
negative infinity until a valid key arrives; the rescaling and exponentials
explicitly handle that state. Each warp retains its query rows across QK,
softmax, and PV, so no cross-warp output reduction is needed.

## Build and compare

The solution uses C++ CuTe headers from the gitlink-pinned
`third_party/cutlass` submodule and compiles with nvcc. `bootstrap` installs the
complete course environment and initializes its upstream sources.

```bash
./lab check s22 --solution
./lab solution s22
./lab reference s22
./lab sanitize s22 --solution --tool memcheck --case 6
./lab sanitize s22 --solution --tool racecheck --case 12
```

The optimized independent library anchor is explicit cuDNN attention including
O and LSE. Both implementations are validated against the independent FP64
oracle before graph timing. Inspect the measured ratios on common perf cases
as well as causal and tail cases, with the full shape, dtype and GPU recorded.

## Source provenance

The optimized fragment transformations and GEMM staging were adapted from
[FlashAttention v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793),
commit `5231d95fe13733fb534c01895f7ea88c6a6c7793` (BSD-3-Clause). Relevant upstream
files are `csrc/flash_attn/src/kernel_traits.h`, `utils.h`, `softmax.h`, and
`flash_fwd_kernel.h`. `LICENSE.flash-attention` and `AUTHORS.flash-attention`
retain upstream license and attribution. This local implementation adds BNHD
addressing, predicated transfers/stores, and causal/tail masks. The CUTLASS
revision is locked by the gitlink; its license is available in
`third_party/cutlass/LICENSE.txt`.
