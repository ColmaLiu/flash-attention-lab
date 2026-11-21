# 30 · TMA and mbarrier: transpose with asynchronous copies

> Part 3 · FA3 ideas on SM100 | Prerequisites: [00_transpose](../00_transpose/README.md), [21_pipeline_gemm](../21_pipeline_gemm/README.md) | Code: [`students/30_tma_mbarrier/kernel.cu`](../../students/30_tma_mbarrier/kernel.cu)

## 1. Goals

Do lab 00's FP32 transpose `y[c][r] = x[r][c]` once more, but this time the threads do not move the data themselves; a dedicated copy engine, the **TMA** (Tensor Memory Accelerator), does. Threads hand off to each other with **mbarrier**: has the data arrived, is the buffer free. This lab has three steps:

1. **Single tile**: one thread issues a TMA load and waits on an mbarrier for the data to arrive, all threads transpose in shared memory, then one thread issues a TMA store;
2. **Persistent ring buffer**: one producer thread keeps issuing loads, consumer warps consume them, and two sets of barriers, `full`/`empty`, manage S buffers;
3. **128B swizzle**: have the TMA write shared memory in an interleaved layout and combine it with a 4×4 register transpose to remove bank conflicts, then tune the tile shape and stage count.

When you finish, you should be able to:

- encode a tensor map with `cuTensorMapEncodeTiled`, and explain what the box, out-of-bounds zero fill and swizzle mode each mean;
- write wrappers for `cp.async.bulk.tensor` load/store and for `mbarrier` init / arrive.expect_tx / try_wait.parity, and draw the phase timeline;
- explain the generic proxy and the async proxy, and know where `fence.proxy.async` must go and which threads execute it;
- push the transpose to about 6.2 TB/s, close to the upper bound of about 6.4 TB/s of `y.copy_(x)` (a contiguous copy without transpose) on the same machine, while using only 9 warps per SM, with all loads issued by 1 thread.

From lab 31 on, [`include/lab/mbarrier.cuh`](../../include/lab/mbarrier.cuh) and [`include/lab/tma.cuh`](../../include/lab/tma.cuh) provide the wrappers you write in this lab. This is the last transpose lab.

## 2. Motivation: free the threads from moving data

On the same GPU of this machine, lab 00 step 4 reaches about 5.9 TB/s, already close to a contiguous copy. But it pays a price: 32 CTAs (64 threads each) are resident per SM, 2048 threads in total, and every byte passes through a thread:

```
LDG (global → registers) → STS (registers → shared) → __syncthreads → LDS → STG
```

To keep enough bytes in flight (Little's law, lab 00 §4.5), the only option is more threads and more loads per thread, at the cost of registers, address arithmetic and waiting in `__syncthreads()`. A transpose has nothing else to do, so it can devote the whole SM to moving data; GEMM and attention cannot: threads must compute MMA and softmax, and registers must hold accumulators. Lab 21's `cp.async` removes the hop through registers, but each thread still issues one instruction and computes one address per 16 bytes.

Since Hopper, the GPU has a separate copy unit, the TMA: **one thread** says with one instruction "move a box of this tensor with origin (x, y) into shared memory", and the hardware generates all the addresses, handles out-of-bounds accesses, and reports the byte count to an mbarrier when it is done. After issuing the instruction, the thread can go do other work.

The first idea of section 3.1 of the FlashAttention-3 paper is exactly this: **producer–consumer asynchrony**. Dedicated producer warps only issue TMA, consumer warps only compute, and the two hand off through a ring buffer in shared memory and barriers, so data movement and compute overlap completely. This lab drills the mechanism on the simplest operator (transpose); lab 31 connects it to the Tensor Cores, and lab 34 uses it in attention.

## 3. Math: bytes, boxes and parity

### 3.1 Lower bound and bytes in flight

A transpose reads and writes `rows·cols·4` bytes each. `big` (8192×8192) totals 512 MiB; the lower bound is about 0.067 ms at the nominal 8 TB/s peak, and about 0.083 ms at the ≈6.4 TB/s measured for `y.copy_(x)` on this machine.

Little's law (lab 00 §4.5): to run at 6.4 TB/s with an estimated latency of about 700 ns, the whole GPU needs about 4.5 MB in flight, about 30 KB per SM. With TMA, the number of read bytes in flight is set directly by the number of buffers:

```
reads in flight per SM ≈ stages per CTA × bytes per stage × CTAs per SM
```

The reference solution for step 3 uses 2 stages × 32 KB × 1 CTA = 64 KB; together with the stores being written back, this is enough to cover the latency.

### 3.2 Box and out of bounds

The TMA moves data in units of a **box**. A box is a fixed-size rectangle of the tensor; its origin coordinates come from the instruction and can be any integers (including ones that make the box stick out of the tensor):

```
x: [rows = 36, cols = 68], box = 32 rows × 32 cols, coordinates written as (x = col, y = row)

        x=0        x=32       x=64 68    x=96
   y=0  ┌──────────┬──────────┬────┐┄┄┄┄┄┐
        │ box(0,0) │ box(32,0)│    │ OOB ┆   out-of-bounds part: load fills 0, store skips it
  y=32  ├──────────┼──────────┼────┤ = 0 ┆
   36   └──────────┴──────────┴────┘     ┆
        ┆ OOB, filled with 0             ┆
  y=64  └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
```

- A load **always** writes the whole box to shared memory (32×32×4 = 4096 bytes), writes 0 for out-of-bounds elements, and **counts all 4096 bytes toward the mbarrier's tx-count**. So `expect_tx` is always the byte size of the box, whether or not it is out of bounds.
- A store writes only the elements inside the tensor; the out-of-bounds part is dropped.

So the kernel of this lab has no `if (r < rows)` at all: the TMA handles the tail completely.

### 3.3 Phase and parity: one ring by hand

Each round of an mbarrier is called a **phase**, numbered 0, 1, 2, …; a wait gives only the parity (odd or even) of the phase number. In a ring buffer, tile number `it` uses buffer `s = it % S`, and it is fill number `n = it / S` of that buffer. So:

- the consumer waits for phase n of `full[s]` to complete: parity = `n & 1`;
- before fill n, the producer must wait for the consumer to release fill n−1, that is, phase n−1 of `empty[s]`: parity = `(n − 1) & 1 = (n & 1) ^ 1`.

For n = 0, "phase −1" does not exist. The PTX rule is: **the phase before the current phase counts as complete**. A freshly initialized barrier is in phase 0, so a wait for parity 1 returns immediately, and the producer's first lap needs no special case. With S = 2:

| it | s | n | parity the producer waits for on `empty[s]` | parity the consumer waits for on `full[s]` |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 (new barrier, passes immediately) | 0 |
| 1 | 1 | 0 | 1 (passes immediately) | 0 |
| 2 | 0 | 1 | 0 (consumer's first release of buffer 0) | 1 |
| 3 | 1 | 1 | 0 | 1 |
| 4 | 0 | 2 | 1 | 0 |

Parity has only 1 bit: a waiter must start waiting before the barrier advances two phases, otherwise the parity "wraps around" and a phase that has not happened yet is taken as complete. The ring buffer guarantees this by construction: the producer does not refill the same buffer before `empty` completes, so `full[s]` can never be two phases ahead of the consumer.

## 4. Hardware and instructions

### 4.1 Tensor map: describing the tensor on the host

`CUtensorMap` is an opaque 128-byte structure. It is encoded once on the host with the Driver API (untimed, in `make_plan`), passed into the kernel as a `const __grid_constant__ CUtensorMap` parameter, and TMA instructions take its address.

```cpp
cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*rank=*/2, base,
                       dims, strides, box, element_strides,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

| Parameter | Value in this lab (x: [rows, cols]) | Rule |
|---|---|---|
| `dims` | `{cols, rows}` | **Innermost (contiguous) dimension first**, in elements |
| `strides` | `{cols * 4}` | rank − 1 entries, in bytes; the stride of dimension 0 is implicitly the element size. **Must be a multiple of 16** |
| `box` | `{32, 32}` or `{32, 256}` | Each dimension ≤ 256; `box[0] × element size` is a multiple of 16, and with swizzle it is at most the swizzle width |
| `element_strides` | `{1, 1}` | Take one element every so many; always 1 in this course |
| `swizzle` | `NONE` or `128B` | See §4.3 |
| `OOB_FILL_NONE` | | Out of bounds filled with 0 (the other option fills NaN) |
| `base` | | 16-byte aligned |

**The supported domain comes from the second row**: the row pitch of x, `cols·4` bytes, and the row pitch of y, `rows·4` bytes, must both be multiples of 16, so this lab requires `rows % 4 == 0` and `cols % 4 == 0`. Lab 00's 7×11 is not legal here.

The coordinate order in TMA instructions is the same as in `dims`: `(x, y)` = (column, row). This is one of the most common mistakes in this lab.

### 4.2 TMA load working with an mbarrier

```
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst], [map, {x, y}], [bar];
```

| Part | Meaning |
|---|---|
| `.2d` | Tensor rank (1d–5d) |
| `.shared::cluster.global` | Destination shared, source global. The CTA's own shared address is also a legal cluster address |
| `.mbarrier::complete_tx::bytes` | After the data is written, performs complete_tx (the byte size of the box) on `bar` |
| `dst` | Shared address; 128-byte aligned without swizzle, **1024-byte aligned** with 128B swizzle |

After issuing it, the thread does not wait. The only way to know whether the data has arrived is through the mbarrier.

### 4.3 Swizzle: why and what

The TMA can only write a **dense** box; it cannot pad each row with one float to shift the banks, as lab 00 did. A 32×32 FP32 tile has exactly 128 bytes = 32 banks per row, so when reading a column, all 32 threads land in the same bank: a 32-way conflict.

Swizzle makes the TMA move each row's 16-byte chunks to different positions as it writes. The rule acts on shared memory **byte addresses** (address bits from bit 4 are the chunk number, bits from bit 7 are the 128-byte "row" number):

| Mode | Rule (byte address) | Repeat period | Max box inner size |
|---|---|---|---|
| `NONE` | unchanged | — | 256 elements |
| `32B` | bit 4 ^= bit 7 | 256 B | 32 B |
| `64B` | bits[4:5] ^= bits[7:8] | 512 B | 64 B |
| `128B` | bits[4:6] ^= bits[7:9] | 1024 B | 128 B |

For a tile with 128-byte rows, `128B` means: **chunk q of row r is stored at chunk position `q ^ (r % 8)`**. The numbers in the table are the logical chunk numbers stored at each position:

```
            physical chunk position (each cell 16 B = 4 floats = 4 banks)
             0   1   2   3   4   5   6   7
   row 0:    0   1   2   3   4   5   6   7
   row 1:    1   0   3   2   5   4   7   6
   row 2:    2   3   0   1   6   7   4   5
   row 3:    3   2   1   0   7   6   5   4
   row 4:    4   5   6   7   0   1   2   3
   row 5:    5   4   7   6   1   0   3   2
   row 6:    6   7   4   5   2   3   0   1
   row 7:    7   6   5   4   3   2   1   0
   row 8:    same as row 0 (repeats every 8 rows = 1024 B)
```

Logical chunk 0 (columns 0–3) sits at 8 different physical positions in the 8 rows, covering all 32 banks. As long as the access pattern makes the threads of one wave (wavefront) read "the same column of different rows", there are no more conflicts. The repeat period is 1024 bytes, so the buffer must be 1024-byte aligned; otherwise the TMA and your index formula disagree.

Why does the hardware also support the 32B/64B widths (three widths in all)? When the box inner dimension is only 32 or 64 bytes (for example 16 BF16 values), a row does not fill 128 bytes, and the swizzle of the matching width is used. The Tensor Core descriptors of lab 31 understand the same set of modes, so the layout written by the TMA can be handed directly to the MMA.

### 4.4 mbarrier

An mbarrier is an 8-byte object in shared memory. Internally it has:

- the current **phase**;
- the **pending arrival count** (reset at the start of each phase to the count given at init);
- the **tx-count** (pending transactions, in bytes).

When **the pending arrival count is 0 and the tx-count is 0**, the current phase completes and the barrier moves to the next phase.

| Wrapper | PTX | Effect |
|---|---|---|
| `mbar_init(bar, count)` | `mbarrier.init.shared::cta.b64 [bar], count;` | Phase 0; each phase expects count arrivals. Executed by one thread |
| `fence_barrier_init()` | `fence.mbarrier_init.release.cluster;` | Makes the init visible to the async proxy (TMA); a later `__syncthreads()` makes it visible to the other threads |
| `mbar_arrive_expect_tx(bar, bytes)` | `mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes;` | tx-count += bytes, then arrive once |
| `mbar_arrive(bar)` | `mbarrier.arrive.shared::cta.b64 _, [bar];` | Arrive once (release semantics) |
| `mbar_wait(bar, parity)` | loop over `mbarrier.try_wait.parity.shared::cta.b64 p, [bar], parity;` | Wait until the phase with that parity completes (acquire semantics) |
| (when the TMA completes) | complete_tx | tx-count −= bytes |

Timeline of one phase of a `full` barrier (count = 1):

```
time →
 producer:   arrive.expect_tx(4096) ──► issue TMA
                   │
 barrier:   phase 0 [pending 1→0, tx 0→4096] ··· TMA done, complete_tx(4096): tx→0 ⇒ phase 0 complete, enter phase 1
                                                                                   │
 consumer:   try_wait.parity(0) returns false … false … ───────────────────────────┴─► true, start reading the tile
```

The order of `expect_tx` and complete_tx does not matter: the tx-count may be temporarily negative, as long as both the arrival count and the byte count match by the end of the phase. But **the arrive must happen before the phase ends**, so by convention you do `arrive.expect_tx` first and then issue the TMA.

The difference between `try_wait` and `test_wait`: `try_wait` may suspend the thread for a short time before returning false, so it does not have to fill the issue slots, and it is cheaper than spinning on `test_wait`; it still has to sit in a loop.

### 4.5 The two proxies and TMA store

The GPU memory model divides accesses into different **proxies**: ordinary `ld`/`st` go through the **generic proxy**, TMA and tcgen05 go through the **async proxy**. When the same shared memory is accessed by both proxies, ordinary synchronization (`__syncthreads`, mbarrier) does **not guarantee** that one proxy's writes are visible to the other; a proxy fence is required:

```
threads that write out:    st.shared out[...]       (generic proxy write)
                           fence.proxy.async.shared::cta   ← executed by every writing thread itself
                           __syncthreads / __syncwarp
thread that issues store:  cp.async.bulk.tensor ... [out]  (async proxy read)
```

A fence executed only by the one thread that issues the store is **not enough**: a fence applies only to the writes of the thread that executes it.

TMA store and its completion:

```
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [map, {x, y}], [src];
cp.async.bulk.commit_group;           packs this thread's uncommitted stores into one group
cp.async.bulk.wait_group.read N;      at most N groups still reading src → src of older groups may be overwritten
cp.async.bulk.wait_group N;           at most N groups not yet complete (including the global writes)
```

As with lab 21's `cp.async.wait_group`, the count covers only the issuing thread. The `.read` version waits only until "shared memory has been read", which is what buffer reuse needs; before the CTA exits, it must also wait until the stores have read their source buffers, because the CTA's shared memory is freed with it.

The opposite direction (a generic read, then a TMA overwrite) needs no fence in this lab: the consumer arrives on `empty[s]` after reading `in[s]`, and the producer issues the load only after its wait succeeds, so the mbarrier's release/acquire already orders them (CUTLASS's `PipelineTmaAsync` does the same).

### 4.6 elect.sync

```
elect.sync %r|%p, 0xffffffff;
```

All active threads of the warp execute it together, and exactly one gets `p = true`. Using it to pick the producer thread expresses "pick one in the warp" more clearly than `lane == 0`.

### 4.7 Common misuses

| Misuse | Consequence |
|---|---|
| Coordinates written as `(row, col)` | Reads the box at the transposed position; may "happen" to work on square matrices, but `tail` and `tall` fail |
| `expect_tx` computed from the number of valid elements | The tail tile never receives the remaining bytes: **hang** (`TIMEOUT`) |
| No `fence_barrier_init` + `__syncthreads` after `mbar_init` | Other threads or the TMA see an uninitialized barrier; occasional hangs or wrong results |
| Wrong parity when waiting on `full` | Hangs on the first lap (waiting for a phase that never comes), or passes immediately and reads stale data |
| Producer's first lap does not use `(n & 1) ^ 1` | Hang: nobody will perform release number −1 on `empty` |
| No `fence.proxy.async` after writing `out` | The TMA store may read stale values. **Neither the tests on this machine nor racecheck found this bug**, so correctness here must come from reasoning, not testing |
| Only lane 0 executes the fence | Same as above: the writes of the other lanes are not fenced |
| No `wait_group` before the CTA exits | The store has not finished reading shared memory when the buffer is freed |
| 128B swizzle buffer not 1024-byte aligned | TMA error, or swizzle indices that do not match |

## 5. Schedule design

### 5.1 Step 1: one tile per CTA

```
grid = (ceil(cols/32), ceil(rows/32)), block = 128 threads (4 warps)
shared: in[32][32], out[32][32] (4 KB each, 128 B aligned), full (8 B)

thread 0:  mbar_init(full, 1); fence_barrier_init
all:       __syncthreads
thread 0:  arrive_expect_tx(full, 4096); tma_load(in ← x box (c0, r0))
all:       mbar_wait(full, 0)
all:       out[i][j] = in[j][i]          (4 warps, 8 rows each)
all:       fence_proxy_async; __syncthreads
thread 0:  tma_store(y box (r0, c0) ← out); commit; wait_group 0
```

Each CTA moves only 4 KB, and load, transpose and store run serially, so the latency is fully exposed; the multiple CTAs resident on an SM are the only source of overlap.

### 5.2 Step 2: persistent + ring buffer

```
block = 32 × (1 + C) threads, C = kRows / 32 consumer warps
grid  = min(number of tiles, number of SMs × CTAs that fit per SM)      ← persistent
tile  = a kRows × 32 block of x; a CTA processes t = blockIdx.x, +gridDim.x, ...

           ┌──────── shared memory ────────────────────────────────────────────────────┐
           │ in[0] in[1] ... in[S-1]      each kRows×32×4 B                            │
 warp 0    │   ▲  TMA load                                                             │
 producer ─┼───┘  full[s] ◄── complete_tx      empty[s] ◄── C arrives                  │
 (elect 1) │                                                                           │
 warp 1..C │ out[cw][0], out[cw][1]  two 32×32 output buffers per warp                 │
 consumer ─┼── read 32 rows of in[s] → write out → TMA store (each warp independently) │
           └───────────────────────────────────────────────────────────────────────────┘
```

| Role | Work per tile |
|---|---|
| producer (1 thread elected in warp 0) | `wait(empty[s], (n&1)^1)` → `arrive_expect_tx(full[s], kRows·128)` → TMA load |
| consumer warp cw | lane 0 `bulk_wait_group_read<1>` + `__syncwarp` (the store from two rounds ago has finished reading its out buffer) → `wait(full[s], n&1)` → transpose its own 32×32 slice → `fence_proxy_async` + `__syncwarp` → lane 0: `arrive(empty[s])`, TMA store, commit |

The whole kernel has only one `__syncthreads()`, after initialization. Consumer warps never wait for each other: each warp stores its own 32×32 box and manages its own two out buffers with its own bulk group. The count of `empty[s]` is C: a buffer can be refilled only after all C consumer warps have read it.

Timeline (S = 4):

```
producer: L0 L1 L2 L3 ·wait e0· L4 ·wait e1· L5 ...
consumer:    ·wait f0· T0 S0 ·f1· T1 S1 ·f2· T2 S2 ...     L = load, T = transpose, S = store
                         └ arrive e0 ┘
```

The step 2 smem budget (kRows = 128, S = 4): in 4 × 16 KB + out 4 warps × 2 × 4 KB = 96 KB, 2 CTAs per SM.

### 5.3 Step 3: conflict-free 4×4 register transpose

In the scalar transpose of step 2, lane j reads `in[j][i]`: all 32 lanes land in the same bank, and each LDS is split into 32 accesses. Instead:

1. Use `128B` swizzle in both the x and y tensor maps, so `in` and `out` both have the layout of §4.3;
2. Treat a warp's 32×32 slice as 8×8 small blocks of 4×4. Lane l handles two small blocks `(R, C) = (l % 8, (l % 8 + l / 8 + 4t) % 8)`, t = 0, 1, which covers each of the 64 small blocks exactly once;
3. For each small block: read rows 4R..4R+3 of `in` at chunk C (4 float4), transpose in registers, and write rows 4C..4C+3 of `out` at chunk R (4 float4).

16-byte accesses are processed in groups of 8 lanes (one wavefront, 8 × 16 B = 128 B = 32 banks). Within one group of 8 lanes (same l / 8):

- read number k: row `4R + k`, physical chunk `C ^ ((4R + k) % 8)`;
- write number k: row `4C + k`, physical chunk `R ^ ((4C + k) % 8)`.

R takes each value 0..7 once, and C = (R + constant) % 8. You can check one by one (or with a small script) that for each k, the physical chunks of the 8 lanes are pairwise distinct, so both reads and writes are conflict-free. The instruction count also drops to 1/4 (each instruction moves 16 bytes instead of 4).

**Tuning**: the tile height kRows (32–256; the box height is at most 256) and the stage count S. The reference solution uses kRows = 256, S = 2: 9 warps per CTA, in 2 × 32 KB + out 8 × 8 KB = 128 KB, one CTA per SM.

## 6. Step-by-step exercises

Run all commands from the repository root.

### Step 1 · TODO(1a)(1b)(1c): wrappers, encoding, single tile

- TODO(1a): nine PTX wrappers (§4.2, §4.4, §4.5). `mbar_wait` needs a loop inside the asm; use `%=` in the labels to keep them unique;
- TODO(1b): the host-side `encode_2d` (the table in §4.1);
- TODO(1c): `transpose_one_tile` (§5.1).

```bash
./lab check 30 --step 1
./lab bench 30 --step 1
```

### Step 2 · TODO(2a)(2b): ring buffer

- TODO(2a): `mbar_arrive`, `bulk_wait_group_read<N>`, `elect_one`;
- TODO(2b): the initialization, producer loop and consumer loop of `transpose_pipelined` (§3.3, §5.2).

```bash
./lab check 30 --step 2
./lab bench 30 --step 2
./lab sanitize 30 --step 2 --case tail --tool racecheck
```

Get `--case tiny` and `--case tail` working first (one tile per CTA), then run `ring` (about 7 tiles per CTA, the ring buffer wraps around) and `ring_tail` (wrapping around with tail tiles at the same time).

### Step 3 · TODO(3a)(3b)(3c): swizzle and tuning

- TODO(3a): `tile_offset<true>` (§4.3);
- TODO(3b): the swizzled version of `transpose_slice` (§5.3);
- TODO(3c): choose the `Config<Rows, Stages, true>` of `Step3`.

```bash
./lab check 30
./lab bench 30 --step 3
./lab bench 30 --step 3 --case wide
./lab profile 30 --step 2
./lab profile 30 --step 3
```

Compare `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `smsp__inst_executed_op_shared_ld.sum` between the two profiles.

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `TODO(1) is not implemented yet (make_plan)` | `encode_2d` is not written yet, or `LAB_TODO(1)` is still there |
| `cuTensorMapEncodeTiled failed` | dims/box in the wrong order (innermost first), stride given in elements, box larger than 256 or than the swizzle width |
| `TIMEOUT` | Wrong `expect_tx` byte count, wrong parity, producer waiting on `empty` in its first lap, init count of `full` not 1 |
| `tiny` and `square` pass, `tail` and `tall` fail | Wrong TMA coordinate order (x, y), or the box origin of y swapped |
| In step 2, `tiny`/`tail` pass and `ring` fails | The number of arrives on `empty` does not match its init count, or no `bulk_wait_group_read` before an out buffer is reused |
| Step 3 wrong values appear in blocks, at columns that differ by multiples of 4 | The swizzle index uses a row number other than the row within the whole tile, or the buffer is not 1024-aligned |
| `misaligned address` / `illegal instruction` | TMA destination address not aligned to 128/1024 bytes |

## 7. Performance

Measured on this machine (CUDA Graph). `./lab bench 30` prints two baselines:

- `torch`: PyTorch `y.copy_(x.t())`, a general-purpose transposing copy;
- `copy`: `y.copy_(x)`, which moves the same number of bytes but reads and writes fully contiguously, without transposing. It is the **practical upper bound** of HBM on this machine.

The `vs` column is the ratio against the first baseline, `torch`. The main yardstick of this lab is **`vs copy`** (`copy` time / your time, the last column of the table below): a transpose can at best be as fast as a contiguous copy, and the closer the ratio is to 1, the better.

| Version | `big` 8192×8192 GB/s | `wide` 4096×16384 GB/s | % of nominal HBM peak | `vs copy` |
|---|---|---|---|---|
| Step 1: single tile | ≈ 1950 | ≈ 1950 | ≈ 24% | ≈ 0.30× |
| Step 2: ring buffer (scalar, no swizzle) | ≈ 2100 | ≈ 2100 | ≈ 26% | ≈ 0.33× |
| Step 3: swizzle, `Config<128, 4>` | ≈ 5600 | ≈ 5600 | ≈ 70% | ≈ 0.87× |
| Step 3: swizzle, `Config<256, 2>` (reference solution) | ≈ 6200 | ≈ 6100 | ≈ 77% | ≈ 0.95× |
| Reference: lab 00 step 4 | ≈ 5900 | — | ≈ 74% | ≈ 0.92× |
| Reference: `torch` (`y.copy_(x.t())`) | ≈ 1860–1900 | ≈ 1890–1910 | ≈ 23% | ≈ 0.29× |
| Reference: `copy` (`y.copy_(x)`, no transpose) | ≈ 6430 | ≈ 6430 | ≈ 80% | 1 |

The reference solution reaches about 95% of a contiguous copy. It is only about 5% faster than lab 00 step 4 (about 2% with `--cold`): a transpose could already use the whole SM for data movement, so the gain from TMA here is not bandwidth but **doing with 1 thread and 9 warps what took lab 00 2048 threads**. In later labs, the threads and registers saved go to Tensor Core compute and softmax. The perf gate of this lab is that step 3 reaches at least 0.85× `vs copy` on both perf cases (`copy` time / your time ≥ 0.85); `vs torch` is usually above 3×, but it is not a good yardstick, for the reasons below.

**Reading this table**:

- **Steps 1 and 2 are both slow, and differ by less than 10%**: the bottleneck is bank conflicts, not data movement. In `./lab profile`, the shared loads of steps 1 and 2 have about 66 million bank conflicts for about 2.1 million LDS instructions: about 31 extra wavefronts per instruction, exactly a 32-way conflict. The step 2 pipeline overlaps loads with the transpose but gains only about 8%, because the transpose itself is the limit.
- **Step 3 is suddenly about 3× faster**: about 520 thousand LDS and 520 thousand STS (one per 16 bytes, 1/4 of before), with about 700 thousand (load) and 1 million (store) bank conflicts, fewer than 2 per instruction on average. The remaining conflicts are not caused by the index formula (a different lane assignment barely changes them); they presumably come from arbitration with the TMA reading and writing the same shared memory, which was not confirmed further.
- **Why `vs torch` is not used as the yardstick**: `torch`'s transpose, like steps 1 and 2, is not limited by HBM. In an earlier, separate measurement, steps 1 and 2 reached only about 1030 and 1060 GB/s (about half of the table above), and `torch` jumped between about 1.0 TB/s and about 1.8 TB/s, while step 3 and `copy` matched the table above. An **unverified hypothesis** is that the GPU was at a lower SM clock at the time: the speed of kernels limited by bank conflicts or latency scales with the SM clock, while HBM-limited kernels (`copy`, step 3) are barely affected. Whatever the cause, `copy` and the reference solution are both set by HBM, so their ratio is stable under such fluctuations.
- **Tuning** (`big`, swizzled version):

| `Config<Rows, Stages>` | smem / CTA | CTA / SM | GB/s |
|---|---|---|---|
| `<32, 4>` | 25 KB | 8 | ≈ 5180 |
| `<64, 4>` | 49 KB | 4 | ≈ 5650 |
| `<128, 2>` | 65 KB | 3 | ≈ 5720 |
| `<128, 4>` | 97 KB | 2 | ≈ 5630 |
| `<128, 6>` | 129 KB | 1 | ≈ 5600 |
| `<256, 2>` | 129 KB | 1 | ≈ 6200 |
| `<256, 3>` | 161 KB | 1 | ≈ 5770 |

More stages (more bytes in flight) are not faster; the fastest is "large box, 2 stages". We have not found the exact reason; a possible explanation is that with too many read requests in flight, HBM read/write turnarounds and page conflicts cancel out the gain. Conclusions like this can only come from measurement; do not pick the stage count by intuition.

During `./lab profile`, Nsight Compute locks the clocks and times are longer than in `bench`; use it only to compare metrics.

## 8. Self-check questions

1. In `tail` (36×68), a CTA processes the x box with origin (x = 64, y = 32) (step 1). How many valid elements does this box contain? How many bytes should `expect_tx` be given?
2. An mbarrier is initialized with count = 2, two threads each execute one `arrive`, and the tx-count is 0. Has phase 0 completed? What if one of the threads used `arrive.expect_tx(1024)` instead, and there is no TMA at all?
3. In step 2, what happens if the count of `empty[s]` is set to the number of consumer threads (128) while each warp still arrives only once?
4. Why does a consumer use `wait_group.read 1` rather than `wait_group.read 0` before reusing an out buffer?
5. With 128B swizzle, what is the byte offset (relative to a 1024-aligned buffer) of row 13, column 22 (float) of `in`?
6. After deleting `fence_proxy_async()`, all tests on this machine still pass. Why does this not show that the fence can be omitted?

<details>
<summary>Answers</summary>

1. Rows 32..35 are valid (4 rows) and columns 64..67 are valid (4 columns), 16 elements in total; `expect_tx` is still the whole box, 32 × 32 × 4 = 4096 bytes, because the out-of-bounds part is written as 0 and counted.
2. In the first case, yes: the arrival count reaches zero and the tx-count is 0. In the second case, no: the tx-count is 1024, with no complete_tx to cancel it, so waiters hang forever.
3. Each phase gets only C arrivals and never reaches 128: the producer hangs on `empty[s]` in its second lap.
4. Buffer `it % 2` was last stored at `it − 2`; at this point the most recently committed group is that of `it − 1`, and allowing it to be still reading (1 pending) guarantees that the group of `it − 2` has finished reading. Using 0 is also correct, but it waits for one more store and reduces overlap.
5. Row 13: `13 % 8 = 5`; column 22 is in chunk `22 / 4 = 5`, float 2 within the chunk. Physical chunk = `5 ^ 5 = 0`. Offset = `13 × 128 + 0 × 16 + 2 × 4 = 1672` bytes.
6. The memory model does not guarantee visibility without the fence; passing tests only show that no reordering happened to occur on this machine under this load. racecheck does not track async proxy accesses either. Correctness of this kind must be reasoned from the specification; tests can find only some of the bugs.
</details>

## 9. Further reading

- PTX ISA 8.7: [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor), [cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group), [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier) (including the definitions of phase and tx-count), [Tensor Memory Swizzling Modes / Tensor copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes), [elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync), [Memory Consistency Model: Proxies](https://docs.nvidia.com/cuda/parallel-thread-execution/#proxies)
- CUDA Driver API: [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- CUDA C++ Programming Guide: [Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#async-copies-tma)
- FlashAttention-3 paper (Shah et al., 2024), section 3.1: Producer-Consumer asynchrony
- Upstream source (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp`: `SM90_TMA_LOAD_2D`, `SM90_TMA_STORE_2D`, `tma_store_arrive` / `tma_store_wait`;
  - `third_party/cutlass/include/cute/arch/copy_sm90_desc.hpp`: `TMA::SmemSwizzleBits` and tensor map encoding;
  - `third_party/cutlass/include/cutlass/arch/barrier.h`: `ClusterTransactionBarrier` (mbarrier wrapper), `fence_view_async_shared`;
  - `third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp`: `PipelineTmaAsync`, the full/empty protocol of step 2 in this lab (the producer starts at phase 1).
