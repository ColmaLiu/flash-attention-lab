# 10 · Tiled GEMM: data reuse and arithmetic intensity

> Part 1 · FA1 | Prerequisites: [00_transpose](../00_transpose/README.md) | Code: [`students/10_simt_gemm/kernel.cu`](../../students/10_simt_gemm/kernel.cu)

## 1. Goals

Use CUDA cores (not Tensor Cores) to compute the FP32 matrix product

```
C[M, N] = A[M, K] · B[N, K]ᵀ        C[m][n] = Σ_k A[m][k] · B[n][k]
```

B is stored as `[N, K]`: A and B are both **K-major** (the K dimension is contiguous), exactly like Q and K in attention's `S = Q·Kᵀ`. This lab builds four versions and raises the same GEMM from about 1% of cuBLAS to about 70%. When you finish, you should be able to:

- compute a kernel's **arithmetic intensity** (FLOP per byte moved), and use the roofline to decide whether it is compute-bound or memory-bound;
- state **how many times** each level of tiling (block tile, K slice, thread micro-tile) reuses the data;
- estimate the shared memory and register budgets, and explain how they limit the tile size;
- explain why a K-major operand must be transposed when it is placed in shared memory, so that it can be read with 128-bit accesses.

The FlashAttention paper uses IO complexity to explain where standard attention is slow. The next lab, [11_naive_attention](../11_naive_attention/README.md), uses this lab's "reuse count" and "arithmetic intensity" to count the HBM traffic of attention.

## 2. Motivation: from "moving data" to "computing enough"

[00_transpose](../00_transpose/README.md) only moves data and does no computation, so it can only chase bandwidth. GEMM is the opposite: each input element takes part in `N` (elements of A) or `M` (elements of B) multiply-adds. As long as each element is read from HBM once and used many times on chip, GEMM can be bound by **compute** rather than by memory access.

The difficulty is that on-chip storage is small: one SM has only 228 KB of shared memory and 256 KB of registers, which cannot hold a whole matrix. So we **tile**: bring only a small tile of A and B on chip at a time, and use it as many times as possible inside the tile. FlashAttention does the same thing for attention: it moves tiles of Q, K and V into SRAM and avoids writing the N×N intermediate matrix back to HBM (paper Section 3.1; the IO complexity analysis is in Section 3.2 and Theorem 2).

## 3. Math: reuse and arithmetic intensity

### 3.1 A small worked example

`M = 2, N = 2, K = 3`:

```
A = [1  2  0]     B = [1  0  1]     C[0][0] = 1·1 + 2·0 + 0·1 = 1
    [0  1 -1]         [2  1  0]     C[0][1] = 1·2 + 2·1 + 0·0 = 4
                                    C[1][0] = 0·1 + 1·0 - 1·1 = -1
                                    C[1][1] = 0·2 + 1·1 - 1·0 = 1
```

`C[m][n]` is the dot product of **row m of A and row n of B**. Both rows are contiguous in memory; this is what K-major means.

The cases `tiny`, `one_tile`, `tail`, `tail_vec` and `k1` use the `int` pattern: the inputs are small integers in −2…2, so every product and partial sum is exactly representable in FP32, and the tolerance is 0. When something fails, you can check the reported values by hand.

### 3.2 FLOP and minimum bytes

GEMM does `M·N·K` multiply-adds, and `./lab bench` counts **2 FLOP per multiply-add**:

```
FLOP          = 2·M·N·K
min HBM bytes = 4·(M·K + N·K + M·N)      (A and B read once each, C written once, FP32 is 4 bytes)
AI            = FLOP / bytes             (arithmetic intensity)
```

| case | FLOP | Minimum bytes | AI (FLOP/B) |
|---|---|---|---|
| `square4k` 4096×4096×4096 | 137.4 G | 201 MB | 683 |
| `wide` 8192×8192×2048 | 274.9 G | 403 MB | 683 |

### 3.3 Roofline

The two "roofs" of B200:

- **FP32 CUDA core peak**: 148 SMs × 128 FP32 lanes per SM × 2 FLOP (one FFMA) × 1.965 GHz (Max Clocks in `nvidia-smi -q -d CLOCK`; the measured clock under full-load timing is still 1965 MHz) ≈ **74.4 TFLOP/s**. The `% peak` of `./lab bench` uses the rounded value of 75 TFLOP/s. A microbenchmark that does only independent FFMAs reaches about 97% of this value on this machine.
- **HBM bandwidth**: nominal 8 TB/s.

The ratio of the two is the **ridge point**: `74.4 / 8 ≈ 9.3 FLOP/B`. A kernel with AI below 9.3 is bandwidth-bound; one above 9.3 is compute-bound.

```
 TFLOP/s
   74 ┤                 ____________________________  FP32 peak
      │               /
      │             /   slope = 8 TB/s
      │           /
      │         /
      │       /
      └──────┴──────────────────────────────────────── AI (FLOP/B)
            9.3                                 683
```

The AI of `square4k` is 683, more than 70 times the ridge point: if each piece of data really is "read only once", reading HBM takes 0.025 ms and computing takes 1.85 ms. **The question is how many times the kernel actually reads it.**

### 3.4 Reuse at each level of tiling

Let one thread block own a `BM × BN` tile of C and move a K slice of width `BK` at a time. For this tile of C, the `BM` rows of A and the `BN` rows of B must each be read in full once, so:

```
floats one block reads from global memory = (BM + BN) · K
total over all blocks                     = M·N·K · (1/BN + 1/BM)
times each element of A is read           = N / BN   (ideally 1)
arithmetic intensity of global reads      = 2·M·N·K / (4·M·N·K·(1/BM + 1/BN)) = BM·BN / (2·(BM + BN))
```

One level further down, the thread's `TM × TN` micro-tile: for one `kk`, the thread reads `TM` values of A and `TN` values of B from shared memory and does `TM·TN` multiply-adds.

| Version | block tile | Outputs per thread | Global read AI | Total global reads (square4k) | Shared loads per multiply-add |
|---|---|---|---|---|---|
| step 1 naive | — | 1 | 0.25 | 550 GB (mostly from L1/L2) | — (reads global directly) |
| step 2 smem tile | 32×32 | 1 | 8 | 17 GB | 2 instructions (the A one is a broadcast) |
| step 3 register tiling | 128×128 | 8×8 | 32 | 4.3 GB | (8 + 8) / 64 = 0.25 |
| step 4 vectorized | 128×128 | 8×8 | 32 | 4.3 GB | same, but 4 128-bit instructions instead of 16 32-bit ones |

Measured with `./lab profile` on the `square` case (1024³): the number of global load sectors in step 2 is exactly 8.4 M. A and B each need `1024³ / 32` floats read (each element is read once by each of `1024 / 32` blocks), `2·1024³ / 32` floats in total, and each 32-byte sector holds 8 floats, so `2·1024³ / 32 / 8 = 8.4 M`. Step 3 is 1/4 of that.

The global read AI of step 2 is 8, below the ridge point of 9.3: even if every read hits in L2, feeding 74 TFLOP/s needs about 9.3 TB/s of read bandwidth. Step 3 needs only about 2.3 TB/s. **The larger the block tile, the less the kernel is bound by global memory access**, but the block tile is constrained by shared memory and register capacity (Section 5).

### 3.5 Invariant

All tiled versions maintain the same invariant: after processing the slice that starts at `k0`,

```
acc[i][j] = Σ_{k < k0 + BK} A[row(i)][k] · B[col(j)][k]
```

Positions past `K` are filled with 0 in shared memory, so their products are 0 and do not affect the result. The tail slice therefore needs no special loop; only the step "move into shared memory" needs a bounds check.

## 4. Hardware and instructions

### 4.1 FFMA and issue slots

One `FFMA d, a, b, c` computes `d = a·b + c`, which is 2 FLOP. Each B200 SM has 4 sub-partitions (SMSPs). Each sub-partition **issues at most one warp instruction** per cycle, and its FP32 unit executes one warp's FFMA (32 lanes) per cycle. So:

- to keep the FFMA units busy, nearly every issued instruction must be an FFMA;
- every `LDS` (shared memory read), address computation and `BAR` (`__syncthreads`) takes an issue slot that should have gone to an FFMA.

This is the most direct benefit of register tiling:

| Version | Instructions per thread per `kk` | Upper bound of FFMA share of issue slots |
|---|---|---|
| step 2 | 1 FFMA + 2 LDS | ≤ 33% |
| step 3 | 64 FFMA + 16 LDS | ≤ 80% |
| step 4 | 64 FFMA + 4 LDS.128 | ≤ 94% |

### 4.2 Shared memory bandwidth

Shared memory processes one **wavefront** per SM per cycle: at most 4 bytes from each of the 32 banks, or 128 B. How many wavefronts a warp access needs depends on bank conflicts (see [00 §4.3](../00_transpose/README.md#43-shared-memory-and-bank-conflicts)) and on the access width. The 4 SMSPs of one SM share this one shared memory pipeline.

An SM does at most 4 warp FFMAs per cycle. In step 3 each warp does 64 FFMAs per `kk` and needs 16 wavefronts: `64 / 16 = 4`, exactly the balance point between FFMA and shared memory throughput, so any conflict or stall makes shared memory the bottleneck. Step 4 measures 12 wavefronts per `kk`, which leaves some margin.

### 4.3 128-bit access: `float4`

```cpp
float4 v = *reinterpret_cast<const float4 *>(p);   // p must be 16-byte aligned
```

This compiles to one `LDG.E.128` (global) or `LDS.128` (shared), which reads 4 consecutive floats at once. The benefit is that **the instruction count drops to 1/4**; the cost is two preconditions:

1. **The address is aligned to 16 bytes.** Row `r` of a global matrix starts at the base address plus `r·K·4` bytes, so every row is aligned only if the base address is 16-byte aligned and `K % 4 == 0`. `run()` checks these two conditions (for reading A and B along K it checks `K` and the base addresses of A and B; for writing C along N it checks `N` and the base address of C), computes `k_vec` and `n_vec`, and passes them to the kernel. When the conditions fail, `load4`/`store4` fall back to 4 scalar accesses (the `tail` case specifically tests the `K % 4 != 0` path).
2. **The 4 elements are adjacent in memory.** This determines the layout of the tile in shared memory (next subsection).

A common misuse: a `float4` read of `&tile[kk][4]` in `float tile[8][130]`. The row length 130·4 = 520 bytes is not a multiple of 16, so the row `kk = 1` is no longer aligned, and the run reports a misaligned address. This lab uses `BM + 4 = 132`.

### 4.4 Why a K-major tile is stored transposed

During computation, for a fixed `kk`, a thread needs 8 values from one column of A (8 different m) and 8 values from one column of B (8 different n).

```
stored as in global (K-major): as[m][kk]      stored transposed: as[kk][m]
        kk →                                        m →
   m  ┌ a00 a01 … a07 ┐                       kk ┌ a00 a10 a20 a30 … ┐
   ↓  │ a10 a11 … a17 │                        ↓ │ a01 a11 a21 a31 … │
      │ a20 a21 … a27 │                          │ …                 │
      └ …             ┘                          └                   ┘
   column (fixed kk): BK+1 floats apart          row (fixed kk) is contiguous: float4 reads work
```

With K-major storage, adjacent m for the same `kk` are a whole row apart, so they can only be read one at a time. If the tile is **transposed while it is moved into shared memory**, the 128 m for a fixed `kk` form one contiguous row, and `LDS.128` reads 4 of them at once. The transpose adds no global traffic; the cost is that each thread does 4 scalar `STS` when writing shared memory (writes are far fewer than reads: each element is written once and read 16 times).

If B is MN-major (`[K, N]`, as V is in attention's `P·V`), a global row is already the contiguous n for a fixed k, so a direct copy works and no transpose is needed. The Tensor Core labs handle both layouts with `ldmatrix` and smem descriptors.

## 5. Schedule design

### Step 1: one thread per output

```
block = (32, 8), grid = (ceil(N/32), ceil(M/8))
col = blockIdx.x·32 + threadIdx.x       ← contiguous within a warp, C writes coalesced
row = blockIdx.y·8  + threadIdx.y
```

The 32 threads of a warp have the same `row` and consecutive `col`: reading `A[row][k]` is a broadcast, and reading `B[col][k]` touches 32 addresses `K·4` bytes apart, so each load needs 32 sectors.

### Step 2: 32×32 block tile, BK = 32

```
block = (32, 32) = 1024 threads, thread (tx, ty) handles C[m0+ty][n0+tx]
for each K slice:
  as[ty][tx] = A[m0+ty][k0+tx]      bs[ty][tx] = B[n0+ty][k0+tx]     ← global reads: coalesced
  __syncthreads()
  acc += as[ty][kk] · bs[tx][kk], kk = 0..31                         ← smem reads
  __syncthreads()
```

During computation the warp has the same `ty` and `tx = 0..31`: `as[ty][kk]` is a broadcast, and `bs[tx][kk]` is a **column read**, which, as in lab 00, needs `+1` padding, so the tile is declared as `[32][33]`.

### Step 3: 128×128 block tile, 8×8 micro-tile

```
block = 256 threads, viewed as 16×16: tx = tid % 16, ty = tid / 16
BM = BN = 128, BK = 8, TM = TN = 8
thread (tx, ty) owns acc[i][j] = C[m0 + ty + 16i][n0 + tx + 16j]       (a "strided" assignment with stride 16)

        n0 → BN = 128
   m0 ┌──┬──┬──┬── … ──┐      thread (0,0) owns the 8×8 positions marked *:
    ↓ │* │  │  │       │      rows 0, 16, 32, …, 112
      │  │  │  │       │      cols 0, 16, 32, …, 112
      │* │  │  │       │
      └──┴──┴──┴── … ──┘

for each K slice:
  stage_kmajor(as) / stage_kmajor(bs): copy the 128×8 slices of A and B into smem as is (provided)
  __syncthreads()
  for kk in 0..7:
      af[i] = as[ty + 16i][kk]   (i = 0..7)
      bf[j] = bs[tx + 16j][kk]   (j = 0..7)
      acc[i][j] += af[i] · bf[j]            ← 8×8 outer product: 16 loads, 64 multiply-adds
  __syncthreads()
```

**Why a strided assignment plus `[BK + 1]` padding.** A warp is tid 0..31, which is `tx = 0..15` and two values of `ty`. When reading `bs[tx + 16j][kk]`, the address is `(tx + 16j)·9 + kk`, and the bank is `(9·tx + constant) % 32`. 9 and 32 are coprime, so the 16 values of `tx` fall in 16 different banks: no conflict. The two addresses of `as` (the two values of `ty`) are also in different banks. With a contiguous assignment instead (`tx·8 + j`), the bank becomes `(72·tx + …) % 32 = (8·tx + …) % 32`, which has only 4 distinct values: a 4-way conflict.

### Step 4: transposed storage + 128-bit access

```
as[BK][BM + 4], bs[BK][BN + 4]: as[kk][r] = A[m0 + r][k0 + kk]
thread (tx, ty) owns two groups of 4 rows and two groups of 4 columns:
  row(i) = m0 + (i/4)·64 + ty·4 + i%4        col(j) = n0 + (j/4)·64 + tx·4 + j%4

load:    tid reads one float4: A[m0 + tid/2][k0 + (tid%2)·4 .. +3], writes it transposed into one column of as (4 STS)
compute: af[0..3] = as[kk][ty·4 ..], af[4..7] = as[kk][64 + ty·4 ..]    ← 2 LDS.128
         bf[0..3] = bs[kk][tx·4 ..], bf[4..7] = bs[kk][64 + tx·4 ..]    ← 2 LDS.128
store:   two float4 stores per row
```

Two details:

- **Why two groups, 64 apart.** If a thread owned 8 consecutive columns `tx·8 .. tx·8+7`, the `float4` addresses of the 16 values of `tx` would be 32 bytes apart, spread over 512 bytes with only half of them used, and `tx` and `tx + 4` would land on the same group of banks. With two groups, the first group that the 16 values of `tx` read is exactly 64 consecutive floats (256 bytes), and no bank is taken by two different addresses. `./lab profile` measures an average of 3 wavefronts per `LDS.128` in step 4, with no bank conflicts: one read of `bs` (16 different `float4`s, 256 bytes in total) needs 4, and one read of `as` (each half warp reads the same address, a broadcast) needs 2. One wavefront is at most 128 bytes, and a 128-bit load in which 32 threads each read 16 bytes needs at least 4 wavefronts, so 4 is already ideal (on the Nsight Compute source page, the `L1 Wavefronts Shared` and `Ideal` columns are equal). Each `kk` needs 2 × 4 + 2 × 2 = 12, matching Section 4.2.
- **Why the padding is 4.** During the transposed write, `lr = tid/2` takes 16 consecutive values within a warp, and `lk` is 0 or 4. With a row length of 128, the two groups of writes land on the same 16 banks: a 2-way conflict. With a row length of 132, the bank is `(4·(lk + i) + lr) % 32`; the `lk = 4` group is offset by 16, and the 32 threads use all 32 banks. The padding must be a multiple of 4 so that `float4` reads stay 16-byte aligned.

### Budget

| Resource | step 2 | step 3 | step 4 | B200 per SM |
|---|---|---|---|---|
| threads / block | 1024 | 256 | 256 | at most 2048 |
| shared memory / block | 2·32·33·4 = 8448 B | 2·128·9·4 = 9216 B | 2·8·132·4 = 8448 B | 228 KB |
| registers / thread (`./lab sass`) | 32 | ≈ 100 | ≈ 127 | 64 K per SM, at most 255 per thread |
| resident blocks / SM | 2 | 2 | 2 | — |

- The `SHARED` that `./lab sass` shows is 1 KB more than the table: each block has 1 KB of shared memory reserved for the system.
- Steps 3 and 4 are limited by **registers**: 64 accumulators + 16 operands + addresses come to about 128 registers; 256 threads × 128 = 32 K, so one SM holds only 2 blocks (16 warps). Shared memory use is under 10%.
- The micro-tile cannot grow without limit: 16×16 needs 256 accumulator registers, above the limit of 255 per thread, so it would spill to local memory.
- The block tile cannot grow without limit either: the larger the tile, the fewer the blocks. `square4k` has 32×32 = 1024 blocks of 128×128, and 148 SMs × 2 = 296 slots, so it takes 3.5 waves, and the last wave uses only half the SMs.

## 6. Step-by-step exercises

Run all commands from the repository root. Each step asks only for its own TODO; the rest of the code is already written.

### Step 1 · TODO(1): naive GEMM

In `gemm_naive`, compute `(row, col)`, check bounds, and compute a dot product of length K.

```bash
./lab check 10 --step 1
./lab bench 10 --step 1          # very slow: about 100 times slower than cuBLAS, so wait a while
```

- `tiny` is a 7×5×3 integer input with tolerance 0. The report gives the value at `[m=…, n=…]`; check it by hand with the method of Section 3.1.
- The `col` case (N = 1) fails while `row` passes: most likely B was treated as `[K, N]`. Row n of B is `b + n·K`.
- The report "elements are NaN … never written": the grid does not cover everything, or a bounds check is reversed.

### Step 2 · TODO(2): shared memory tile

In `gemm_smem`, implement the K loop "load → synchronize → 32 multiply-adds → synchronize" and the final write-back.

```bash
./lab check 10 --step 2
./lab bench 10 --step 2
./lab profile 10 --step 2 --case square
```

In the profile output, compare `smsp__inst_executed_op_shared_ld.sum` (shared load instruction count) and `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (global load sector count). Do they match the formulas in Section 3.4?

### Step 3 · TODO(3): register tiling

`stage_kmajor` is already written. You write two parts: the 8×8 outer product for each `kk`, and the write-back after the loop.

```bash
./lab check 10 --step 3
./lab bench 10 --step 3
./lab sass 10                    # REG of gemm_regblock should be around 100, LOCAL 0
```

- Add `#pragma unroll` to every loop. The indices of `acc[i][j]` must be known at compile time; otherwise the array is put in local memory (`LOCAL` is nonzero in `./lab sass`), and performance drops by an order of magnitude.
- `one_tile` passes, `tail` fails: the write-back does not check `row < M` and `col < N`.

### Step 4 · TODO(4): transposed storage and vectorization

In `gemm_vec`, write the full K loop (`load4` → transposed write → synchronize → `float4` reads → outer product → synchronize) and the write-back with `store4`.

```bash
./lab check 10
./lab bench 10 --step 4
./lab bench 10 --step 4 --case wide
./lab profile 10 --step 4 --case square
```

Compare `smsp__inst_executed_op_shared_ld.sum` (should drop to 1/4) and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (conflicts of the transposed write) between steps 3 and 4.

### Common mistakes

| Symptom | Likely cause |
|---|---|
| errors in `int` pattern cases are integers, for example off by 2 or 4 | some K slice is added twice or skipped: check the step of `k0` and the second `__syncthreads()` |
| occasional failures, unstable results | missing `__syncthreads()`: the next slice's load overwrites a tile that others have not finished reading |
| `tail` fails, `tail_vec` passes | `float4` is still used when `K % 4 != 0`, or out-of-bounds K columns are not filled with 0 |
| `misaligned address` | the `float4` pointer is not 16-byte aligned: check the padding and the `vec` arguments |
| step 3 is slower than step 2 | `acc` was put in local memory (see `LOCAL` in `./lab sass`), or reads of `bs` have bank conflicts |
| `k1` fails | with K = 1 there is only one slice, and only column 0 is valid: check the zero fill past the bounds |

## 7. Performance

Measured on this machine (CUDA Graph timing; `vs cublas` is the time of cuBLAS SGEMM (TF32 off) divided by the time of this version):

| Version | `square4k` vs cublas | `wide` vs cublas | `square4k` % FP32 peak |
|---|---|---|---|
| step 1 | ≈ 0.01× | (too slow, not measured) | ≈ 0.3% |
| step 2 | ≈ 0.14× | ≈ 0.13× | ≈ 5% |
| step 3 | ≈ 0.56× | ≈ 0.53× | ≈ 19% |
| step 4 | ≈ 0.72× | ≈ 0.68× | ≈ 25% |
| cuBLAS | 1× | 1× | ≈ 35% |

Your numbers will differ slightly with GPU state, but the ratios should match. If step 4 reaches 0.6× cuBLAS or better, the mechanism works.

Metrics worth reading in `./lab profile` (measured on `square`, step 2 → 3 → 4):

| Metric | step 2 | step 3 | step 4 | Explanation |
|---|---|---|---|---|
| `smsp__inst_executed_op_shared_ld.sum` | 67 M | 8.4 M | 2.1 M | shared load instructions per multiply-add: 2 → 1/4 → 1/16 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | 67 M | 8.4 M | 6.3 M | actual work of the shared memory pipeline |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈ 0 | 0 | 0 | effect of padding and thread assignment |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 8.4 M | 2.1 M | 2.1 M | block tile grows from 32 to 128, global reads drop to 1/4 |

`sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` on `square4k` rises by about a quarter from step 3 to step 4; it is good for relative comparisons.

**Where the remaining gap is.**

- On each slice, this lab's kernels "load everything first, then compute everything". During the load the FFMA units sit idle, and only another block on the same SM can fill the gap (each SM has only 16 warps). Here cuBLAS uses CUTLASS's 3-stage pipelined SIMT kernel, which is already moving later slices while it computes the current one. This is exactly the topic of [21_pipeline_gemm](../21_pipeline_gemm/README.md).
- Even cuBLAS reaches only about 35% of peak, while a microbenchmark that does only independent FFMAs reaches about 97%. The gap comes from other overheads in the GEMM inner loop (LDS, address computation, synchronization, contention for register read ports); this lab did not measure the share of each item.
- More fundamentally, the dense BF16 Tensor Core peak of B200 is about 2250 TFLOP/s, 30 times that of the FP32 CUDA cores. Real attention kernels all use Tensor Cores, starting with [20_mma_gemm](../20_mma_gemm/README.md). This lab's analysis of reuse, tiling and budgets applies there unchanged.

## 8. Self-check questions

1. What is the global read arithmetic intensity of a 64×64 block tile? Is it above or below the ridge point?
2. What happens if you delete the second `__syncthreads()` in the K loop of step 2? Why do `int` pattern cases expose this problem more easily than random cases?
3. If step 3 used a contiguous assignment instead (a thread owns columns `tx·8 .. tx·8+7`) and kept `bs[128][9]`, how many ways would the bank conflict be when reading `bs`?
4. Why not grow the micro-tile to 16×16?
5. If B were stored as `[K, N]`, would step 4 still need to transpose B when loading it?
6. How much time do the minimum HBM traffic and the computation of `square4k` each take? If the total global reads of step 2 (table in Section 3.4) all came from HBM, how much time would they take?

<details>
<summary>Answers</summary>

1. `64·64 / (2·(64 + 64)) = 16 FLOP/B`, above 9.3 but by less than 2 times: a slightly worse L2 hit rate makes it memory-bound.
2. A fast warp moves on to the next slice and overwrites `as` and `bs` while a slow warp is still reading them, so some products use data from the wrong slice. The `int` pattern has tolerance 0, so any wrong term is reported; random data also reports an error, but with the `int` pattern the error is an integer, which makes it easier to see that "some term is wrong".
3. The address is `(tx·8 + j)·9 + kk` and the bank is `(72·tx + …) % 32 = (8·tx + …) % 32`. The 16 values of `tx` fall in only 4 banks, with 4 different addresses per bank: a 4-way conflict.
4. 16×16 needs 256 accumulator registers; with the operands this exceeds the limit of 255 registers per thread, so it spills to local memory. Even without a spill, doubling the registers per thread means one SM holds only 1 block, which makes the load latency harder to hide.
5. No. In `[K, N]`, a global row is the contiguous n for a fixed k, so a direct copy gives `bs[kk][n]`. A is still K-major and still needs the transpose.
6. Minimum traffic: 201 MB / 8 TB/s ≈ 0.025 ms; computation: 137.4 GFLOP / 74.4 TFLOP/s ≈ 1.85 ms. The global reads of step 2 are about 17 GB; if they all came from HBM they would take about 2.1 ms, already more than the computation time. This is why the block tile must be large enough, and why the kernel relies on L2 hits.
</details>

## 9. Further reading

- FlashAttention paper (Dao et al., 2022), Sections 2.1–2.2 (GPU memory hierarchy, arithmetic intensity) and Section 3.2 (IO complexity)
- CUDA C++ Programming Guide: [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory) (includes a tiled matrix multiplication example)
- CUDA C++ Best Practices Guide: [Shared Memory in Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
- Samuel Williams et al., *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (CACM 2009)
- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM): steps 1–4 of this lab correspond to kernels 1–6 of that article
- CUTLASS (`third_party/cutlass`, commit `4370102`):
  - `media/docs/cpp/efficient_gemm.md`: the tiling hierarchy (block tile, warp tile, thread tile) and pipelining
  - `include/cutlass/gemm/warp/mma_simt.h`, `include/cutlass/gemm/threadblock/default_mma_core_simt.h`: thread assignment and shared memory layout of the SIMT GEMM
  - `include/cutlass/gemm/collective/builders/sm100_simt_builder.inl`: how the SM100 SIMT kernel that cuBLAS picks on B200 is built
