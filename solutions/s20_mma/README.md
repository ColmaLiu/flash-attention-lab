# s20: the MMA instruction as a complete program

`kernel.cu` is a complete CUDA/PTX answer with FP16/BF16 operands and FP32
accumulation/output. A warp owns a 16×8 output tile. The register packing follows
PTX's `mma.sync.aligned.m16n8k16.row.col` fragment contract: compare each `ar`,
`br`, and accumulator index with the instruction's lane/value diagram.

This first version deliberately reloads operands from global memory for each
warp. It is an inspectable instruction baseline; s21 adds CTA reuse and overlaps
copies. The independent cuBLAS baseline remains the performance target.

```bash
./lab check s20 --solution
./lab solution s20
./lab sanitize s20 --solution --tool memcheck --case 4
```

Dimensions must be positive multiples of 16. No fragment ABI is inferred from
WMMA in this version. See the [PTX MMA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
