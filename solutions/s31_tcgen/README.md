# s31_tcgen: complete SM100 GEMM reference

The executable integration uses 固定两个 K buffers. A/B are BF16 row-major, C is FP32 row-major, with a 128×128×64 MMA tile and one-CTA cluster. The complete device implementation is vendored under `references/cutlass-sm100/include`, not downloaded at execution time. Read `kernel_impl.cuh` in s31, the SM100 collective, its pipeline implementation, and the tcgen05 atom in that order; the exact paths and license are in [the source map](../../references/README.md).

Both configurations already use producer/consumer warp roles. The s31/s32 executable comparison isolates buffering, not the first introduction of warp specialization. `prepare` creates descriptors, queries hardware and allocates workspace before timing. Repeated launches invoke the prepared CUTLASS operator on the provided stream. No tensor layout conversion is performed.

```bash
./lab check s31_tcgen --solution
./lab solution s31_tcgen --case 5
```

These commands validate the reference independently. They do not mark the student's implementation complete.
