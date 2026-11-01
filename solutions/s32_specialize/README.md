# s32_specialize: SM100 GEMM reference

This C++ CuTe answer uses a K-buffer count selected from the shared-memory budget. A/B are BF16 row-major and C is FP32 row-major, with a 128×128×64 MMA tile and one-CTA cluster. CUTLASS is provided by the gitlink-pinned `third_party/cutlass` submodule.

Read [the host integration](../s31_tcgen/kernel_impl.cuh), the SM100 collective, its pipeline implementation and tcgen05 atoms in that order. [The source map](../../docs/reference-implementations.md) links the complete device code and license. The 01/02 Blackwell tutorials introduce MMA/TMEM and TMA before the production collective.

Both s31 and s32 use producer/consumer warp roles. Their executable comparison isolates buffering. `prepare` creates descriptors and allocates workspace before timing; subsequent launches run the prepared CUTLASS operator on the supplied stream with the public row-major layouts.

```bash
bash scripts/bootstrap.sh
./lab check s32_specialize --solution
./lab solution s32_specialize --case 5
```

The commands validate and measure this answer. Student acceptance uses the separate `students/` entry.
