# 21 · Async copy pipeline: cp.async multistage buffers

> Part 2 · FA2 | Prerequisites: [20_mma_gemm](../20_mma_gemm/README.md) | Code: [`students/21_pipeline_gemm/kernel.cu`](../../students/21_pipeline_gemm/kernel.cu)

## 1. Goals

We keep computing `C[M,N] = A[M,K] · B[N,K]ᵀ` (BF16 inputs, FP32 output). In the lab 20 kernel, the Tensor Cores sit idle while data moves, and nothing moves data while they compute. This lab uses **`cp.async`** to load the next tile while the current tile is computed:

1. **Double buffering**: two shared memory buffers; compute from one while loading the other;
2. **Multistage ring buffer**: S buffers, at most S−1 tiles in flight at once, counted with commit groups;
3. **Tuning**: a larger K tile, 64×64 warp tiles, grouped rasterization, and profiling to explain why each change helps (or does not).

When you finish, you should be able to:

- write wrappers for `cp.async`, `commit_group` and `wait_group`, and explain their completion semantics;
- name the two things that each `__syncthreads()` in the pipeline loop protects at the same time;
- estimate the number of pipeline stages you need with Little's law;
- raise lab 20's ≈0.17× of cuBLAS to ≈0.28×, about 70% of the `mma.sync` upper bound.

[`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh) provides `ldmatrix`, `mma`, fragment coordinates and swizzle (the code you wrote in lab 20). From lab 22 on, [`include/lab/cp_async.cuh`](../../include/lab/cp_async.cuh) provides the `cp.async` wrappers you write in this lab.

## 2. Motivation: stop the Tensor Cores from waiting for data

The K loop of lab 20 step 3 is:

```
load_tile(A), load_tile(B)  ← all warps read global, Tensor Cores idle
__syncthreads()
ldmatrix + mma              ← all warps compute, no memory access in flight
__syncthreads()
```

A normal global load goes into registers first. The thread must wait for the data to arrive before it can store it to shared memory, so "load" and "compute" can only alternate. Other CTAs resident on the same SM can fill part of the idle time, but each CTA on its own is serial.

The FlashAttention-2 implementation (`third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`) prefetches the next K/V block with `cp.async` while it computes `Q·Kᵀ` for the current K/V block, and waits with `cp_async_wait<1>`. FA2 in lab 22 reuses the pipeline structure of this lab: the K dimension of the GEMM corresponds to the KV sequence dimension of attention.

## 3. Math: how much data must be in flight

### 3.1 Little's law

As in [lab 00 §4.5](../00_transpose/README.md#45-littles-law-how-much-data-must-be-in-flight):

```
bytes in flight needed = bandwidth needed × latency
```

**Bandwidth needed**: per K tile, one CTA reads `(BM + BN) · BK · 2` bytes and performs `2 · BM · BN · BK` floating-point operations. With BM = BN = 128 and BK = 32, that is 16 KB for 1 MFLOP. On B200, the `mma.sync` upper bound is about 1920 FLOP/cycle/SM (lab 20 §4.1). For one SM to compute at that rate, it must be fed `1920 / 64 ≈ 30` bytes per cycle, that is, **each SM must finish reading one 16 KB tile about every 550 cycles**.

**Latency**: most data comes from L2 (at 4096³, A and B total 64 MB and fit in the 126 MB L2). An L2 hit takes a few hundred cycles; a miss adds HBM latency, on the order of 1000 cycles (this is an estimate; this lab does not measure it).

**Conclusion**: each SM needs about `30 B/cycle × 1000 cycles ≈ 30 KB` in flight, about two 16 KB tiles. With double buffering, each CTA is one tile ahead, so two CTAs on an SM together give about two tiles; each extra pipeline stage adds one tile of margin. The measurements in section 7 test this estimate.

### 3.2 The group-counting invariant

In a multistage pipeline, tile t goes into stage `t % S`. Each thread commits one commit group per "round" and maintains:

> **Commit group g contains exactly the copies of tile g** (an empty group when tile g does not exist).

So "tile t has arrived" is equivalent to "the first t+1 groups have completed". If `t + S − 1` groups have been committed at that point (tiles 0 .. t+S−2), allowing `S − 2` of them to be pending guarantees exactly that tiles 0..t are complete. This is where `wait_group<S − 2>` comes from. The empty groups keep the count invariant at the end of the K loop: without them, `wait_group<S−2>` in the last few rounds would let through tiles that have not arrived.

| S | prologue commits | wait in each loop round | tiles in flight at once |
|---|---|---|---|
| 2 (double buffering) | tile 0 | `wait_group 0` | 1 |
| 3 | tile 0, 1 | `wait_group 1` | 2 |
| 4 | tile 0, 1, 2 | `wait_group 2` | 3 |

## 4. Hardware and instructions

### 4.1 `cp.async`

```
cp.async.cg.shared.global [dst], [src], 16, src_size;
```

| Part | Meaning |
|---|---|
| `cp.async` | Copies from global to shared **without going through registers**; the thread continues right after issuing it (`LDGSTS` in SASS) |
| `.cg` | Caches only in L2 (bypasses L1); only cp-size = 16 is allowed. The other option is `.ca` (also caches in L1), with cp-size 4, 8 or 16 |
| `16` | The copy size (cp-size): only 4, 8 or 16; `.cg` allows only 16. `dst` and `src` must both be aligned to the size |
| `src_size` | Optional. Reads only `src_size` bytes from `src` and **fills the remaining `16 − src_size` bytes with zeros** |

This lab uses `src_size = valid ? 16 : 0` to handle the tail: an out-of-bounds chunk is written as 16 zero bytes, which contribute nothing to the dot product, and not a single byte is read. `src` must still be a "legal-looking" address; passing the matrix base address works.

Why require `K % 8 == 0`? A 16-byte chunk is 8 BF16 values. When K is a multiple of 8, every row starts 16-byte aligned, and each chunk is either fully inside the row or fully outside it, so a `src_size` of 0 or 16 is enough. (Supporting any K would require filling `src_size` with a byte count and handling misaligned rows; this lab does not do that.)

### 4.2 commit group and wait_group

```
cp.async.commit_group;      packs this thread's earlier, not-yet-grouped cp.async ops into one group
cp.async.wait_group N;      waits until at most N of this thread's most recent groups are pending
cp.async.wait_all;          = commit_group + wait_group 0
```

```
Timeline of thread t:
  cp.async × 8 ─commit─┐ cp.async × 8 ─commit─┐ cp.async × 8 ─commit─┐
                  group 0                group 1                group 2
  wait_group 1  ⇒  group 0 complete (groups 1 and 2 may still be pending)
```

**It covers only the calling thread.** When `wait_group` returns, the copies issued by this thread are written to shared memory and visible to this thread; copies issued by **other threads** are not necessarily complete or visible.

### 4.3 Each `__syncthreads()` protects two things

The pipeline loop has only one `__syncthreads()` per K tile, placed after `wait_group` and before the next batch of copies is issued. It guarantees two things at once:

1. **RAW (read after write)**: every thread's copies of tile t are complete and visible, so any warp may `ldmatrix` tile t;
2. **WAR (write after read)**: every warp has finished computing the previous round's tile (t−1), so its stage may be overwritten.

| What is removed | Consequence |
|---|---|
| `wait_group` | Copies are not complete at the barrier; `ldmatrix` reads stale data |
| `__syncthreads()` | Only this thread's own copies are waited for; other threads' chunks may not have arrived, and a stage that another warp is still reading may be overwritten |
| Barrier placed after the issue | WAR is broken: new copies may overwrite a stage that a warp is still reading |

These bugs often show up only in some runs. `./lab sanitize 21 --case aligned --tool racecheck` can find races on shared memory.

### 4.4 Shared memory above 48 KB

A static `__shared__` array is limited to 48 KB. For larger buffers, use dynamic shared memory (`extern __shared__`, with the size as the third argument of `<<<grid, block, smem>>>`), and declare the limit before launch:

```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
```

B200 has 228 KB of shared memory per SM, and a single CTA can use at most 227 KB. Shared memory and L1 share the same storage on each SM; the driver picks the split (the carveout) per kernel. The skeleton's `allow_smem` also sets `cudaFuncAttributePreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared`: on this machine, without it, the 3–4 stage versions occasionally had runs that were twice as slow (as if only one CTA fit per SM); with it, those runs disappeared. This part is already written.

### 4.5 Common misuses

| Misuse | Consequence |
|---|---|
| asm without a `"memory"` clobber | The compiler may move ordinary shared memory reads and writes before `wait_group` |
| Skipping out-of-bounds chunks with `if (ok) cp_async(...)` | Shared memory keeps stale values from the previous tile; the tail results are wrong |
| `src` still points to the computed out-of-bounds address | With `src_size = 0` nothing is read, but the out-of-bounds address may fall in another allocation or even an unmapped page; passing the matrix base address is safest |
| A variable as the `wait_group` argument | Compile error: N must be an immediate (template parameter + `"n"` constraint) |
| No empty groups committed in the prologue when K is short | `wait_group<S−2>` lets through tiles that have not arrived in the last few rounds |

## 5. Schedule design

### 5.1 Step 1: double buffering

The CTA tile is the same as in lab 20: 128×128, BK = 32, 8 warps (2×4, each 64×32), and the shared tile uses lab 20's 4-chunk swizzle. Each stage is 16 KB; two stages are 32 KB.

```
prologue:  issue(tile 0 → stage 0); commit
t = 0:     wait 0; sync │ issue(tile 1 → stage 1); commit │ compute(stage 0)
t = 1:     wait 0; sync │ issue(tile 2 → stage 0); commit │ compute(stage 1)
t = 2:     wait 0; sync │ issue(tile 3 → stage 1); commit │ compute(stage 0)
                 ▲                                              │
                 └── copy of tile t+1 overlaps compute(t) ──────┘
```

Overwriting stage `(t+1) % 2` in round t is safe: it last held tile t−1, and all warps finished computing tile t−1 before the barrier of round t (the WAR case of §4.3).

### 5.2 Step 2: S-stage ring buffer

```
prologue:  for s in 0 .. S−2:  issue(tile s → stage s) (if it exists); commit
for t in 0 .. k_tiles−1:
    wait_group<S−2>; __syncthreads()
    issue(tile t+S−1 → stage (t+S−1) % S) (if it exists); commit   ← may be an empty group
    compute(stage t % S)
```

Stage `(t+S−1) % S = (t−1) % S` last held tile t−1; the same argument applies. With S = 3, shared memory is 48 KB, and two CTAs fit per SM.

### 5.3 Step 3: tuning

| Change | Purpose |
|---|---|
| BK = 32 → 64 | 128 bytes per row (8 chunks), using `swizzle_chunk<8>`; the work per tile doubles, and `__syncthreads()` and loop overhead are halved |
| 8 warps of 64×32 → 4 warps of 64×64 | Per k16: 4 A `ldmatrix` + 4 B `ldmatrix` feed 32 mma (before: 6 feed 16); shared memory reads relative to compute drop by 1/3 |
| 3 stages × 32 KB = 96 KB | Two CTAs share one SM (192 KB) |
| Grouped rasterization | CTAs that run at the same time share A and B tiles, which reduces L2 misses |

A 64×64 warp tile has 128 float accumulators; the reference solution uses about 254 registers. A CTA has 128 threads, so two CTAs use 65536 registers, exactly one SM's register file.

**Rasterization**: `blockIdx.x` is one-dimensional, and `tile_coords` maps it to (tm, tn). With a row-major mapping, the ≈296 CTAs that run at the same time (148 SMs × 2) cover several full rows of C and need every B tile. A grouped mapping first walks kGroup rows along M, then moves to the next column:

```
Row-major (kGroup = 1)          Grouped (kGroup = 4, tiles_n = 4)
pid: 0  1  2  3                 pid: 0  4  8 12
     4  5  6  7                      1  5  9 13
     8  9 10 11                      2  6 10 14
    12 13 14 15                      3  7 11 15
                                    16 20 ...       ← next group
```

The figure is only an illustration; kGroup = 4 keeps it small enough to draw. The reference solution uses `kGroupM = 8`. The last group may have fewer than kGroup rows (`raster` case: 11 rows of tiles; with kGroup = 8 the last group has 3 rows).

## 6. Step-by-step exercises

### Step 1 · TODO(1): cp.async and double buffering

- TODO(1a): the three wrappers `cp_async_16_zfill`, `cp_async_commit`, `cp_async_wait_group<N>` (§4.1, §4.2);
- TODO(1b): `issue_tile`: the same thread mapping as lab 20's `load_tile`, but writing shared memory directly with `cp_async_16_zfill` (the tail handling of §4.1);
- TODO(1c): the prologue and K loop of `gemm_double_buffer` (§5.1).

```bash
./lab check 21 --step 1
./lab bench 21 --step 1
./lab sanitize 21 --step 1 --case aligned --tool racecheck
```

### Step 2 · TODO(2): multistage ring buffer

Implement the prologue and K loop of `gemm_multistage` (§3.2, §5.2). `run()` calls it with S = 3; step 3 uses the same kernel.

```bash
./lab check 21 --step 2
./lab bench 21 --step 2
```

Try changing the stage count of `Step2` to 2 and 4. Check whether `short_k` and `tiny` still pass and how performance changes, then change it back to 3.

### Step 3 · TODO(3): tuning

- TODO(3a): grouped rasterization in `tile_coords` (§5.3);
- TODO(3b): choose the configuration of `Step3`. Start your experiments from the table below; after each change, run `check`, then `bench`:

| Experiment | `Config<BM, BN, BK, WarpsM, WarpsN, Stages>` |
|---|---|
| Starting point (= step 2) | `128, 128, 32, 2, 4, 3` |
| Larger K tile | `128, 128, 64, 2, 4, 3` |
| Larger warp tile | `128, 128, 64, 2, 2, 3` |
| Larger CTA tile | `128, 256, 32, 2, 4, 3` |
| More stages | `128, 128, 64, 2, 2, 4` |

The default configuration of `Step3` is the step 2 configuration, so `./lab check 21 --step 3` passes as soon as TODO(3a) is written. Passing only shows that the rasterization is correct; the point of this step is tuning: compare `bench` across configurations and use profiles to explain what is fast and what is slow.

```bash
./lab check 21
./lab bench 21 --step 3 --case square
./lab bench 21 --step 3 --case wide
./lab sass 21          # register count, any spill (LOCAL > 0)
./lab profile 21 --step 3 --case wide
```

Change `kGroupM` to 1, profile again, and compare `dram__bytes_read.sum` and `lts__t_sector_hit_rate.pct`.

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `tail` or `tiny` fails, and the wrong values look like data from elsewhere | Out-of-bounds chunks are not zero-filled (§4.5) |
| Results are occasionally wrong, different on each run | Missing `wait_group` or `__syncthreads()`, or the barrier is in the wrong place (§4.3); use racecheck |
| `short_k` and `tiny` fail, others pass | Group counting in the prologue or tail is wrong: commit every round even when K is short |
| `aligned` fails, `tail` passes | Wrong stage index (`t % S`); after the ring buffer wraps around, a stale tile is read |
| Compile error `asm operand must be an immediate` | The N of `wait_group` is a runtime variable |
| `raster` fails (step 3 only) | `tm` and `tn` are computed wrong when the last group has fewer than kGroup rows |
| `too many resources requested for launch` | Registers × threads or shared memory of the configuration exceed the limit |

## 7. Performance

Measured on this machine (CUDA Graph; larger `vs cublas` is better; "% of mma.sync upper bound" is TFLOP/s divided by the 560 TFLOP/s measured in lab 20 §4.1):

| Version | `square` `vs cublas` | `wide` `vs cublas` | % of mma.sync upper bound |
|---|---|---|---|
| Lab 20 step 3 (synchronous loads) | ≈ 0.17× | ≈ 0.17× | ≈ 45% |
| Step 1: double buffering | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| Step 2: 3-stage pipeline | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| Step 3: tuned | ≈ 0.28× | ≈ 0.26× | ≈ 72% |
| mma.sync upper bound | ≈ 0.39× | ≈ 0.36× | 100% |

If step 3 reaches 0.22× of cuBLAS or more (about 55% of the mma.sync upper bound), the mechanism works. On `wide`, our kernel varies by about ±10% from run to run (cuBLAS is stable); run it several times and look at the median.

**Reading this table**:

- **Step 1 gives the largest gain**: loads and compute overlap for the first time.
- **Step 2 changes almost nothing**: this matches the §3.1 estimate: with double buffering, each of the two CTAs on an SM has one tile in flight, which is already close to the amount needed. Multistage pipelines matter when "only one CTA fits per SM" or "latency is longer", for example the TMA pipelines of Part 3.
- **The step 3 gain comes from the tile shape**, not from rasterization. Step 3 configuration experiments (`square`, this machine):

| Configuration | `vs cublas` |
|---|---|
| `128,128,32, 2,4, 3` (starting point) | ≈ 0.24× |
| `128,128,64, 2,4, 3` | ≈ 0.27× |
| `128,128,64, 2,2, 3` (reference solution) | ≈ 0.28× |
| `128,256,32, 2,4, 3` | ≈ 0.19× |
| `128,128,64, 2,2, 4` (128 KB, one CTA per SM) | ≈ 0.23× |

A `128×256` CTA needs 170 registers per thread, so only one CTA (8 warps) fits per SM, and no other CTA fills its idle time; the same holds for a 64-wide, 4-stage pipeline. **The number of resident warps and the pipeline depth together decide whether latency can be hidden.**

**Rasterization** (`wide`, `./lab profile`, reference configuration):

| Metric | `kGroupM = 1` | `kGroupM = 8` |
|---|---|---|
| `dram__bytes_read.sum` | ≈ 716 MB | ≈ 556 MB |
| `lts__t_sector_hit_rate.pct` | ≈ 79% | ≈ 86% |
| `dram__throughput` | ≈ 5% | ≈ 4% |
| Time | same | same |

Grouping cuts DRAM reads by about 22%, but the time does not change: HBM uses only about 5% of its bandwidth, and the kernel is bound by the Tensor Cores (`sm__pipe_tensor_cycles_active` about 79%). Optimize the real bottleneck; the L2 hit rate starts to matter only when the Part 3 kernels are 3–4× faster.

The remaining ≈30% gap to the `mma.sync` upper bound comes mainly from the `__syncthreads()` per tile, the dependence between `ldmatrix` and mma, and the uneven waves of 1024 CTAs over 296 slots (3.46 waves). Part 3 removes thread-driven loads and CTA-level barriers with TMA + mbarrier, and goes past the `mma.sync` upper bound with `tcgen05`.

## 8. Self-check questions

1. In the double-buffered loop, what goes wrong if you move `__syncthreads()` after `issue_stage` and before `compute_stage`?
2. With S = 4, after `wait_group<2>` returns in loop round t, which tiles are guaranteed to have arrived for this thread? Which may still be in flight?
3. With K = 8 and S = 3 (the `tiny` case), how many groups are committed in total? How many of them are empty?
4. Why must an out-of-bounds chunk be written as zeros with `src_size = 0` instead of skipping the copy?
5. Estimate with the method of §3.1: if B200 ran at `tcgen05` throughput (about 4× that of `mma.sync`), how much data in flight per SM would the same 128×128×32 tile need?

<details>
<summary>Answers</summary>

1. WAR is broken: the issue writes to the previous round's stage, and some warp may still be running `ldmatrix` on that stage (it has not reached the barrier yet). RAW still holds, but the new tile overwrites data that another warp is reading.
2. Tiles 0 .. t+2 have been committed, t+3 groups in total, with at most 2 pending, so tiles 0 .. t have arrived; tiles t+1 and t+2 may still be in flight. This covers only this thread's own copies, which is why `__syncthreads()` must follow.
3. k_tiles = 1, so the prologue commits 2 groups (tile 0 and an empty tile 1), and the loop commits 1 more at t = 0 (tile 2, empty). 3 in total, 2 of them empty.
4. Shared memory stages are reused. Skipping the copy leaves the previous tile's data, which then enters the dot product and gives wrong results. (On first use, it would be uninitialized values.)
5. The bytes needed per cycle grow about 4×, to about 120 B/cycle/SM. With the same latency (about 1000 cycles), about 120 KB must be in flight, that is, 7–8 tiles (16 KB each). This is why Part 3 uses larger tiles and deeper pipelines (TMA multistage buffers).
</details>

## 9. Further reading

- PTX ISA: [Data Movement and Conversion Instructions: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async), plus `cp.async.commit_group` and `cp.async.wait_group` in the same section
- CUDA C++ Programming Guide: [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) (`cuda::memcpy_async` is the C++ interface to the same hardware); [Shared Memory and carveout](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)
- Vasily Volkov, *Understanding Latency Hiding on GPUs* (2016): Little's law and occupancy
- Upstream source (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cute/arch/copy_sm80.hpp`: `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL`, the equivalent of this lab's `cp_async_16_zfill`;
  - `third_party/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h`: the multistage pipelined GEMM main loop of CUTLASS 2.x;
  - `third_party/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h`: CTA rasterization;
  - `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`: the K/V prefetch with `cp_async_fence` / `cp_async_wait<1>` in the FA2 forward pass.
