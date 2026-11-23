# 00 · Memory access and thread mapping: matrix transpose

> Part 0 · Preliminaries | Prerequisites: none | Code: [`students/00_transpose/kernel.cu`](../../students/00_transpose/kernel.cu)

## 1. Goals

Transpose an FP32 matrix: `y[c][r] = x[r][c]`. A transpose does no arithmetic, so its speed depends entirely on **how data moves between the levels of the memory hierarchy**. This lab builds four versions of the same transpose and raises it from 15% of nominal HBM bandwidth to about 74% (about 90% of the speed of a contiguous copy). When you finish, you should be able to:

- map thread coordinates to array indices, and decide whether a warp's memory access is coalesced;
- use shared memory to change the access order, and count bank conflicts;
- use Little's law to explain "why each thread should move more data";
- check all of the above with `./lab bench` and `./lab profile` instead of guessing.

The attention kernels in every later lab do these same things over and over: move tiles of Q, K and V from HBM into shared memory, read them back in a different order, and keep enough data in flight at once.

## 2. Motivation: why start with a transpose

The core observation of the FlashAttention paper is that the bottleneck of attention is not floating-point arithmetic but HBM reads and writes (paper Section 2.2). To understand this claim, you first need to know what "reading and writing HBM" means on a GPU: how many threads issue requests, how the requests are coalesced, and how many requests must be in flight at once. A transpose is the smallest kernel that does only memory access, so it is the best place to see these things clearly.

## 3. Math: indices and lower bound

Both matrices are row-major: the last dimension is contiguous.

```
x: [rows, cols]   address of x[r][c] = x + (r * cols + c)
y: [cols, rows]   address of y[c][r] = y + (c * rows + r)
```

A transpose must read `rows*cols*4` bytes and write the same number of bytes. The `big` case is 8192×8192, so reads plus writes total 512 MiB. B200 HBM3e has a nominal bandwidth of 8 TB/s, so the lower bound on time is about 0.067 ms. The simplest `y.copy_(x)` (a contiguous copy, no transpose) measures about 6.5 TB/s, which is a more realistic upper bound.

The `GB/s` that `./lab bench` reports is `2 * rows * cols * 4 / time`, and `% peak` is that value as a fraction of 8 TB/s.

## 4. Hardware and instructions

### 4.1 Thread hierarchy

| Level | Use in this lab | Numbers on B200 |
|---|---|---|
| grid | the grid of blocks that covers the whole matrix | — |
| block (CTA) | 32×8 = 256 threads that share one shared memory region | up to 2048 threads and 228 KB shared memory per SM |
| warp | 32 threads with consecutive `threadIdx.x`, **executing the same instruction together** | up to 64 warps per SM |
| thread | moves one or a few elements | 148 SMs |

The linear index of a thread inside a block is `threadIdx.x + threadIdx.y * blockDim.x`, and every 32 consecutive indices form one warp. In this lab `blockDim.x = 32`, so **the 32 threads with the same `threadIdx.y` are one warp**. Remember this; it drives all the memory-access analysis below.

### 4.2 Coalescing global memory access

When a warp executes a load/store instruction, the hardware handles the addresses of all 32 threads together:

- memory moves between L1/L2/HBM in units of **32-byte sectors**;
- when 32 threads access 32 consecutive `float`s (128 bytes), only **4 sectors** are needed: this is coalesced access;
- when 32 threads each access addresses far apart, **32 sectors** are needed, and only 4 bytes of each are used.

In the naive transpose, the 32 threads of a warp have the same `r` and consecutive `c`:

- reading `x[r][c]`: the addresses are contiguous, so the access is coalesced;
- writing `y[c][r]`: adjacent threads' addresses differ by `rows * 4` bytes, so the access is not coalesced.

Below is a measured result of `./lab profile 00 --solution --step 1` (8192×8192). Each store request maps to 32 sectors:

| Metric | read x | write y |
|---|---|---|
| requests (warp instructions) | 2,097,152 | 2,097,152 |
| sectors | 8,388,608 (4 per request) | 67,108,864 (32 per request) |

Interestingly, the number of bytes actually read from and written to HBM (`dram__bytes_*`) is almost the same as for the optimized versions: other warps soon write the rest of the same 32-byte sector, and L2 merges these fragmented writes before writing back to HBM. So the naive version is not slow because "HBM moves 8 times as much data". It is slow because **L1/L2 must process 8 times as many sector transactions**, and transaction throughput becomes the bottleneck. This is why you measure instead of reasoning only from textbook conclusions.

### 4.3 Shared memory and bank conflicts

Shared memory sits on each SM. Its latency and bandwidth are far better than HBM's, and threads in a block can read data that other threads in the block wrote. It is split into **32 banks**, and each bank serves one 4-byte word per cycle:

```
bank(address) = (byte address / 4) % 32
```

If the 32 threads of a warp access 32 different banks, the access completes in one cycle. If k threads access **different addresses** in the same bank, the access is serialized into k passes; this is a k-way bank conflict (accessing the same address is a broadcast, not a conflict).

With the declaration `float tile[32][32]`, the bank of `tile[i][j]` is `(i*32 + j) % 32 = j`. Reading by row (fixed `i`, `j = lane`) has no conflict. **Reading by column** (fixed `j`, `i = lane`) puts all 32 threads in bank `j`: a 32-way conflict.

Pad each row with one float: with `float tile[32][33]`, the bank of `tile[i][j]` becomes `(i*33 + j) % 32 = (i + j) % 32`. A column read with `i = 0..31` hits 32 different banks, and the conflict disappears. The cost is 128 extra bytes of shared memory per block.

(The later tensor core labs use another method, XOR swizzle, which wastes no space. The idea is the same: spread one column of data across different banks.)

### 4.4 `__syncthreads()`

All threads in the block must reach this instruction before any continues, and it guarantees that earlier writes to shared memory are visible to the other threads in the block. In the transpose, a thread reads elements that **other threads** wrote into the tile, so this barrier must sit between the load and the store. Without it, the result is sometimes right and sometimes wrong, depending on the order in which warps are scheduled.

### 4.5 Little's law: how much data must be in flight

```
bandwidth = bytes in flight / latency
```

HBM access latency is on the order of a few hundred nanoseconds to 1 microsecond. To reach 6.5 TB/s with an estimate of 700 ns, the whole GPU needs about 4.5 MB in flight at once, or about 30 KB per SM on average.

At most 2048 threads are resident on one SM. Data in flight = resident threads × loads in flight per thread × bytes per load. The last two factors depend on how you write the code:

- **The loads must be able to issue together.** After a thread issues a load, it does not stop to wait for the data. It keeps executing later instructions that do not depend on that data, and waits only when it uses the data. So the more independent loads a thread can issue back to back, the more data is in flight at once. This is called instruction-level parallelism (ILP).
- **How you write the loop affects whether the compiler can do this.** With `for (k = 0; k < 4; ++k)`, the trip count is a compile-time constant, so the compiler fully unrolls the loop and issues the 4 loads back to back. With `for (i = threadIdx.y; i < 32; i += 8)`, the trip count depends on `threadIdx.y`, so the compiler generates code with branches: after each load it immediately writes the result into shared memory, and the next load waits for the previous one to return. In measurements, this one change moves step 3 from about 2.8 TB/s to about 5.0 TB/s. You can check it in the SASS file that `./lab sass 00` generates: count the `LDG` instructions before the first `STS` in the kernel.

In step 3, each thread has 4 loads in flight. A fully loaded SM then has about 2048 × 4 × 4 B = 32 KB in flight, right around the estimate above. Also, each block reads no data in its write phase after `__syncthreads()`. Step 4 keeps the tile the same and cuts each block's threads from 32×8 to 32×2: each thread handles 16 elements and issues 16 loads back to back, which raises the data in flight to 4 times the previous amount. This is common on GPUs: **per-thread ILP and the number of threads on the SM (TLP) together determine the memory traffic in flight**, and fewer threads that each do more work are often faster.

## 5. Schedule design

### Step 1: one thread per element

```
block = (32, 8), grid = (ceil(cols/32), ceil(rows/8))
c = blockIdx.x * 32 + threadIdx.x      ← contiguous within a warp
r = blockIdx.y * 8  + threadIdx.y
```

### Steps 2–4: a tile through shared memory

```
            one tile of x (kTile × kTile)                   matching place in y
   c0 ────────────────────► cols                     r0 ──────────────────► cols
r0 ┌──────────────────────┐                      c0 ┌──────────────────────┐
   │ warp 0 reads row 0   │  ── write tile ──►      │ warp 0 writes row 0  │
   │ warp 1 reads row 1   │     __sync              │ = column c0 of x     │
   │ ...                  │  ── read by col ─►      │ ...                  │
   └──────────────────────┘                         └──────────────────────┘
   global read: coalesced                           global write: coalesced
```

- Read phase: warp `threadIdx.y` reads rows `threadIdx.y`, `threadIdx.y+kThreadsY`, … of the tile, with lanes mapped to columns, so the reads are coalesced. It writes `tile[i][j]` by row, with no conflict.
- Write phase: row `c0+i` of y is column `c0+i` of x. A warp writes one row of y; lane `j` writes `y[c0+i][r0+j]` and needs `tile[j][i]`, which means **reading the tile by column**.

| Version | tile | block | elements per thread | shared memory / block |
|---|---|---|---|---|
| step 2 | `float[32][32]` | 32×8 | 4 | 4096 B |
| step 3 | `float[32][33]` | 32×8 | 4 | 4224 B |
| step 4 | `float[32][33]` | 32×2 | 16 | 4224 B |

In step 4 each block has only 64 threads. At most 32 blocks are resident on one SM, so the SM can still hold the full 2048 threads, and the shared memory use is 32 × 4224 B ≈ 135 KB, within 228 KB.

## 6. Step-by-step exercises

Run all commands from the repository root. Each step asks only for its own TODO; the rest of the code is already written.

### Step 1 · TODO(1): naive transpose

In `transpose_naive`, compute `(r, c)`, check bounds, and copy one element.

```bash
./lab check 00 --step 1
./lab bench 00 --step 1
```

- The first case, `tiny`, is 7×11, and each input value equals its linear index in x (`coordinate` mode). On an error, the report says "at `[col=…, row=…]` got v, expected w", and `v` tells you directly which position of x you read.
- The report "elements are NaN … never written" means some outputs were never written: check the grid coverage and whether the bounds checks are swapped.
- `rows * cols` exceeds 2³¹ in the big case, so use `int64_t` for indices.

### Step 2 · TODO(2): shared memory tile

In `transpose_tiled`, implement the three phases: read, synchronize, write. Following the skeleton comments, write the loops with a compile-time trip count, `for (k = 0; k < kTile / kThreadsY; ++k)` (see Section 4.5 for why), so that step 4 can reuse them directly.

```bash
./lab check 00 --step 2
./lab bench 00 --step 2
./lab profile 00 --step 2
```

In the profile output, find `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: this is the number of bank conflicts on shared memory loads. On 8192×8192 it is in the tens of millions. Does that match the analysis in Section 4.3?

### Step 3 · TODO(3): remove bank conflicts

In `case 3` of `run()`, choose `kPad` and launch the kernel. The change is only one line, but first work out on paper which banks the 32 elements of a column hit when `kPad = 1`.

```bash
./lab check 00 --step 3
./lab bench 00 --step 3
./lab profile 00 --step 3
```

### Step 4 · TODO(4): each thread moves more data

In `case 4`, choose `kThreadsY` so that each thread handles 16 elements, and launch the kernel. If you wrote the step 2 loops in terms of `kTile / kThreadsY`, the kernel itself needs no change. First work it out: in step 3 and in step 4, how many bytes of loads are in flight on a fully loaded SM?

```bash
./lab check 00
./lab bench 00 --step 4
./lab bench 00 --step 4 --cold
```

`--cold` flushes L2 before each timed run, so it measures performance when the data really comes from HBM. The 8192×8192 input (256 MiB) is already larger than L2 (126 MB), so the two timing modes differ little. On small matrices they differ a lot; think about why.

### Common mistakes

| Symptom | Likely cause |
|---|---|
| most cases pass, `tail` fails | the bounds check of the tail tile uses the wrong dimension (reads use `rows/cols`, writes use `cols/rows`) |
| occasional failures, unstable results | missing `__syncthreads()` |
| `row` case fails | y was treated as `[rows, cols]` |
| step 3 is not faster | check that the bank conflict metric really dropped; check that you did not forget to change `case 3` |
| `TIMEOUT` | a wrong loop variable causes an infinite loop |

## 7. Performance

Measured (8192×8192, CUDA Graph timing; `vs torch` is the time of PyTorch `y.copy_(x.t())` divided by the time of this version):

| Version | `vs torch` | % nominal HBM peak |
|---|---|---|
| step 1 | ≈ 0.65× | ≈ 15% |
| step 2 | ≈ 1.0× | ≈ 24% |
| step 3 | ≈ 2.7× | ≈ 62% |
| step 4 | ≈ 3.2× | ≈ 74% |

Your numbers may differ slightly with GPU clock and load, but the ratios should match. If step 3 reaches about 2.5× `vs torch` and step 4 is another 15% or more faster, the mechanism works. If step 3 is only about 1.5×, check that the loops use a compile-time trip count (Section 4.5).

During `./lab profile`, Nsight Compute locks the clocks, so `gpu__time_duration` is somewhat slower than `bench`. Use it only to compare metrics; do not compare it directly with bench times.

Step 4 is already close to the 6.5 TB/s contiguous-copy speed. Most of the remaining gap comes from the block structure that alternates reads and writes: a block issues no new read requests during its write phase. [30_tma_mbarrier](../30_tma_mbarrier/README.md) returns to this problem when it does the transpose with TMA.

## 8. Self-check questions

1. With `blockDim = (32, 8)`, which warp of the block does the thread with `threadIdx = (5, 3)` belong to?
2. In step 1, if `threadIdx.x` maps to rows and `threadIdx.y` maps to columns, which is coalesced, the read or the write?
3. In `float tile[32][32]`, which banks do the 32 threads access when they read column 7? If you change it to `double tile[32][33]`, is it still conflict-free?
4. In step 4 a block has only 2 warps. Is reading the tile by column still free of bank conflicts? Why?
5. A 4096×4096 FP32 matrix fits entirely in L2. If you time it with graph timing and with `--cold` timing, which is faster, and why?

<details>
<summary>Answers</summary>

1. The linear index is `5 + 3*32 = 101`, and `101 / 32 = 3`, so it is warp 3 (counting from 0).
2. The write is coalesced and the read is not: the 32 threads of a warp have consecutive row indices `r`, so `y[c][r]` is contiguous while `x[r][c]` crosses rows.
3. All in bank 7: a 32-way conflict. A `double` occupies two adjacent banks, and 32 threads need 256 bytes in total, so the hardware already serves them in two passes (half a warp, 16 threads, per pass). In `double tile[32][33]`, the starting bank of row `i`, column `j` is `(66i + 2j) % 32 = (2i + 2j) % 32`. Within one half warp, `i` takes 16 consecutive values and the bank pairs they occupy are all different, so there is no conflict beyond the two required passes. You can verify this with the bank conflict metric from `./lab profile`.
4. Conflict-free. A bank conflict happens only within one access by one warp. In the write phase, the 32 lanes of a warp read `tile[lane][i]`, whose bank is `(lane*33 + i) % 32 = (lane + i) % 32`, and the 32 lanes are pairwise different. This does not depend on how many warps the block has.
5. Graph timing is faster: across repeated runs the input stays in L2, while `--cold` flushes L2 before each run, so the data must be read from HBM.
</details>

## 9. Further reading

- CUDA Programming Guide: [Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access), [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts), and on the same page, [Matrix Transpose Example Using Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-shared-memory)
- CUDA C++ Best Practices Guide: [Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) (a classic article, the source of steps 1–3 in this lab)
- Vasily Volkov, *Understanding Latency Hiding on GPUs* (PhD thesis, 2016): analyzes GPU memory access with Little's law
- FlashAttention paper, Sections 2.1–2.2: the GPU memory hierarchy and IO awareness
