# 32 · warp specialization: a persistent GEMM with a split-role pipeline

> Part 3 · FA3 ideas on SM100 | Prerequisites: [31_tcgen05_gemm](../31_tcgen05_gemm/README.md) (and the full/empty ring buffer of [30_tma_mbarrier](../30_tma_mbarrier/README.md)) | Code: [`students/32_ws_gemm/kernel.cu`](../../students/32_ws_gemm/kernel.cu)

## 1. Goals

This is still the GEMM of lab 31: `C[M,N] = A[M,K] · B[N,K]ᵀ`, BF16 inputs, FP32 output, the same supported domain (`K % 8 == 0`, `N % 4 == 0`). No new instruction is added: TMA, mbarrier and `tcgen05.mma/commit/ld` all come from lab 31 and are provided by [`include/lab/tcgen05.cuh`](../../include/lab/tcgen05.cuh), [`tma.cuh`](../../include/lab/tma.cuh) and [`mbarrier.cuh`](../../include/lab/mbarrier.cuh). This lab changes **who does what, and when**:

| | Lab 31 | This lab |
|---|---|---|
| Who issues TMA | thread 0 | A dedicated **producer warp** |
| Who issues MMA | The same thread 0 | A dedicated **MMA warp** (one lane) |
| Who writes C back | All 4 warps, after the main loop ends | Dedicated **epilogue warps**, concurrently with the main loop of the next tile |
| How many tiles one CTA computes | 1 | Many (**persistent**, grid = number of SMs) |
| TMEM accumulators | 1 | 2 (**double buffering**) |

This lab has three steps:

1. **Role split**: one tile per CTA; a producer warp + an MMA warp + 4 epilogue warps hand stages over through full/empty barriers;
2. **persistent + accumulator double buffering**: each CTA loops over many tiles; while the epilogue reads accumulator 0, the MMA is already computing the next tile into accumulator 1;
3. **Tile order**: grouped rasterization, so the 148 tiles computed at the same time share less A and B data, which reduces HBM reads.

After this lab you should be able to:

- Draw the timelines of the three roles and say whom each wait is waiting for;
- For each of the 4 kinds of barrier (`full`, `empty`, `acc_full`, `acc_empty`), say who arrives, who waits, what the init count is, and which parity the first round waits on;
- Keep the phases of the ring buffer and the accumulators continuous across tiles instead of resetting them for each tile;
- Raise the GEMM from about 0.8× cuBLAS in lab 31 to about 0.9×, and explain what the remaining gap needs (lab 33).

## 2. Motivation: the serialization in lab 31

In the lab 31 kernel, one thread plays two roles, and all warps run the epilogue only after the main loop. Here is the timeline of one CTA:

```
Lab 31 (1 CTA per SM)
thread 0  : ld ld ld ld ─wait f─mma─commit─wait e─ld─wait f─mma ... mma─commit(done)
warps 0-3 : ································· idle ·································│ld TMEM│write C│
Tensor    :            ══ MMA ══      ══ MMA ══      ══ MMA ══ ... ══ MMA ══════════│     idle      │
                                                                                    ↑ the next CTA must wait for this CTA to exit
                                                                                      before it starts: it refills from empty buffers
```

Three problems:

1. **One thread waits on two things**. While it waits for TMA data (`full`), it cannot refill buffers; while it waits for an MMA to release a buffer (`empty`), it cannot issue the next MMA. Lab 31 softens this with the "refill the stage of t−1" order, but it is still one serial instruction stream.
2. **The epilogue does not overlap the main loop**. A 128×256 FP32 tile has 128 KB to write back, and the Tensor Core is idle during the write-back. Lab 31 uses 192 KB of shared memory, so each SM holds only 1 CTA, and "another CTA fills the gap" is not possible either.
3. **Every tile starts from scratch**. CTA launch, barrier init, TMEM allocation, and filling the pipeline from empty are paid once per tile.

Section 3.1 of the FlashAttention-3 paper calls the solution **producer–consumer asynchrony**: warp specialization gives "moving data" and "computing" to different warps, which hand work to each other only through barriers and each move at their own pace; on top of that, pingpong scheduling lets one group of warps run a GEMM while another group runs softmax. FA3's mapping on Hopper, compared with this lab's mapping on Blackwell:

| FA3 (Hopper) | This lab (Blackwell) | Labs 34 / 40 (attention) |
|---|---|---|
| producer warp: TMA reads Q, K, V | producer warp: TMA reads A, B | Same as left |
| consumer warpgroup issues `wgmma`, accumulator in registers | **one lane of the MMA warp** issues `tcgen05.mma`, accumulator in TMEM | Same as left |
| consumer warpgroup does its own softmax and write-back | **epilogue warps** read TMEM and write back | softmax / correction warpgroups read S and write P |
| pingpong: two warpgroups alternate GEMM and softmax | Similar overlap (different mechanism): **TMEM double buffering**; the epilogue reads accumulator 0 while the MMA writes accumulator 1 | S double buffering: QKᵀ overlaps softmax |
| `setmaxnreg`: the producer gives up registers | Not needed in this lab (§4.5) | Used in lab 40 |

The split goes further on Blackwell: one thread issues the MMA and the accumulator does not occupy registers, so "issuing MMAs" and "consuming the accumulator" naturally become two roles. This lab builds that structure on a GEMM; lab 34 replaces the epilogue with softmax.

## 3. Math: when overlap pays off

### 3.1 The two time segments of a tile

A 128×256 tile has `k_tiles = K / 64` K tiles along K, and each K tile has 4 MMAs of 128×256×16. With the estimate of lab 31 §3.2 (about 8192 FLOP per cycle per SM, about 1.9 GHz):

```
T_main ≈ k_tiles × 4 × 128 cycles = k_tiles × 512 cycles
T_epi  ≈ 128 KB of C / HBM write bandwidth per SM (8 TB/s ÷ 148 ≈ 54 GB/s) ≈ 2.4 µs ≈ 4600 cycles
```

| case | K | k_tiles | T_main | T_epi / T_main |
|---|---|---|---|---|
| `square` 4096³, `wide` | 4096 | 64 | ≈ 32800 cycles ≈ 17 µs | ≈ 14% |
| `flat` 8192×8192×1024 | 1024 | 16 | ≈ 8200 cycles ≈ 4.3 µs | ≈ 56% |

Without overlap each tile takes `T_main + T_epi`; with overlap (steady state) it takes `max(T_main, T_epi)`. The smaller K is, the more the overlap is worth. This is why this lab adds the `flat` perf case (this is only an order-of-magnitude estimate; Section 7 has measurements).

### 3.2 The phase invariant of the ring buffer

The rule from lab 30: in a ring with S slots, the u-th use (u counts from 0, accumulated over all tiles) falls on

```
slot  = u % S
phase = (u / S) & 1           the waiting side (consumer) waits on parity = phase
                              the releasing side (producer) waits on the opposite barrier, parity = phase ^ 1
```

`phase ^ 1` is 1 in the first round (phase = 0): a freshly initialized barrier is in phase 0, and the "previous phase" (parity 1) counts as completed, so the first round does not block. This is exactly "all slots start empty".

**A parity wait can only tell two phases apart**: the "current phase" (not completed; the wait blocks) and "the phase right before it" (counts as completed; the wait returns immediately). The barrier does not remember any older history, so the waiting side must know exactly which phase's parity it wants.

**Phases do not reset across tiles.** Example: S = 4, 6 K tiles per tile. The 1st tile uses u = 0..5: slots 0, 1, 2, 3, 0, 1. At this point `full[0]` and `full[1]` have each completed 2 times (current phase 2, parity 0), and `full[2]` and `full[3]` have each completed 1 time (current phase 1, parity 1).

- Correct: the 3rd K tile of the 2nd tile is use u = 8: slot 0, phase (8/4) & 1 = 0.
- If each tile resets `RingPos` to (slot 0, phase 0): the 1st and 2nd K tiles of the 2nd tile fall on slots 0 and 1 and wait on parity 0, which is exactly their current phase, so they block normally until the data arrives. This is right by luck. The 3rd K tile falls on slot 2 and waits on parity 0, but the current phase of `full[2]` is 1, so parity 0 refers to the "previous phase" (the load of the 1st tile). **The wait returns immediately**, and the MMA reads the stale data of the 1st tile.

So this bug does not hang; it silently computes some of the K tiles wrong. The `persistent` case catches it.

In the code, each role holds a `RingPos{index, phase}`, calls `advance()` after each use, and keeps moving forward across tiles:

```cpp
template <int N> struct RingPos {
  int index = 0;  uint32_t phase = 0;
  void advance() { if (++index == N) { index = 0; phase ^= 1; } }
};
```

The producer and the MMA warp each have a `RingPos` for the stage ring, and the MMA warp and the epilogue each have a `RingPos` for the accumulator ring (N = 2). They do not communicate: everyone processes the same sequence of tiles with the same number of K tiles, so their use counts agree naturally.

### 3.3 Tile order and L2

In a persistent kernel, CTA b processes tiles `b, b + 148, b + 296, ...` in turn. At any moment, the 148 SMs are computing about 148 consecutive tile indices. The mapping from tile index to `(mb, nb)` decides which row blocks of A (each 128 × K) and row blocks of B (each 256 × K) these 148 tiles need.

`wide` (M = N = 8192, K = 4096): one row block of A is 1 MB, one row block of B is 2 MB, and there are 64 × 32 tiles.

| Order | Shape of the 148 tiles | A needed | B needed | Total |
|---|---|---|---|---|
| Row-major (nb changes fastest) | ≈ 4.6 rows × 32 columns | 4.6 MB | 64 MB (all of B) | ≈ 69 MB |
| Grouped G = 8 (mb changes fastest within a group) | 8 rows × 18.5 columns | 8 MB | 37 MB | ≈ 45 MB |
| Grouped G = 32 | 32 rows × 4.6 columns | 32 MB | 9.3 MB | ≈ 41 MB |

L2 is 126 MB, but it must hold the data being read, the data just read, and the write-back of C at the same time; the smaller the working set, the more A and B reads hit in L2. With row-major order, after each row of tiles all of B (64 MB) is scanned again from the start, and it has likely been evicted from L2. With grouping, the data in use at the same time is much smaller (Section 7 has the measured HBM read volume).

For `square` (4096³), A + B is only 64 MB in total; any order fits in L2, and grouping gives no benefit.

## 4. Hardware and instructions

This lab has no new instructions. Below are the things to watch when "existing instructions are split across different warps".

### 4.1 Picking one lane: `elect.sync`

The producer and MMA loops need only one thread. `lab::elect_one()` (`elect.sync`) is called by the whole warp together, and exactly one lane gets true:

```cpp
if (warp == kProducerWarp) {
  if (lab::elect_one()) {       // one lane runs the whole loop
    ...
  }
  __syncwarp();                 // the other 31 lanes wait for it here, then go to __syncthreads together
}
```

This is equivalent to `if (threadIdx.x == 128)`, but it does not hard-code a thread index, and the compiler also knows that the condition is true for exactly one lane in the warp.

### 4.2 A barrier can have several arrivers

The count in `mbar_init(bar, count)` is the number of arrivals each phase needs. The four kinds of barrier in this lab:

| barrier | count | Who arrives | How it arrives | Who waits |
|---|---|---|---|---|
| `full[s]` | 1 | producer | `arrive_expect_tx(48 KB)`, then TMA does `complete_tx` | MMA |
| `empty[s]` | 1 | MMA | `tcgen05.commit`: arrives when the MMAs of that stage finish reading | producer |
| `acc_full[a]` | 1 | MMA | `tcgen05.commit`: arrives when all MMAs of that tile complete | 128 epilogue threads |
| `acc_empty[a]` | **128** | each epilogue thread | `tcgen05.fence::before_thread_sync` + `mbar_arrive` | MMA |

`acc_empty` counts as empty only after **all** 128 threads have finished reading their own rows, so count = 128 (CUTLASS also uses the number of epilogue threads).

A barrier can have many waiters: 128 threads wait on `acc_full[a]` at the same time, and each of them sees the same phase complete.

### 4.3 `tcgen05.commit` is an asynchronous "release"

The MMA warp never waits for MMAs to complete. It issues the 4 MMAs of a K tile, then `tcgen05.commit(&empty[s])`, and immediately goes to wait on the next `full`. The commit only makes `empty[s]` remember "arrive once when these MMAs have all completed". The producer waits on `empty[s]` in another warp and is woken up when the MMAs finish reading stage s. `tcgen05.commit(&acc_full[a])` at the end of a tile works the same way: it tracks **all MMAs this thread issued before it**, so when it arrives the whole tile is complete.

### 4.4 TMEM double buffering and fences

Two 128×256 FP32 accumulators = 512 columns = all of TMEM (128 lanes × 512 columns × 32 bits). Allocate 512 columns once; accumulator a is at column `base + 256a`. Each SM holds only 1 CTA of this kernel (225 KB of shared memory), so it never competes with its own other CTAs for TMEM.

When TMEM is accessed across roles, the fence rule of lab 31 §4.7 still holds:

```
epilogue thread: tcgen05.ld ... wait::ld;  tcgen05.fence::before_thread_sync;  mbar_arrive(acc_empty[a])
MMA thread     : mbar_wait(acc_empty[a]);  tcgen05.fence::after_thread_sync;   tcgen05.mma → accumulator a
```

**Early release**: the epilogue arrives on `acc_empty` as soon as it has finished reading TMEM, while its TMA store is still writing C. This is safe: the store reads the staging box in shared memory and does not touch TMEM any more.

This lab uses `lab::tmem_ld_32x32b_x32_sync`: `tcgen05.ld` and `tcgen05.wait::ld` are in the same asm block, so the registers are valid when it returns. If they are written as two asm blocks, the compiler does not know that the registers become valid only after `wait::ld`, and it may move computation that uses only those registers before the wait.

### 4.5 Registers and `setmaxnreg`

FA3 uses `setmaxnreg` to let the producer warpgroup give its registers to the consumer. This lab does not need it: the 6 warps use 56–72 registers per thread, far below the register file (64K × 32 bits / 192 threads ≈ 340 per thread), and there is no spill (check with `./lab sass 32`). `setmaxnreg` adjusts registers in units of warpgroups; it is covered in lab 40 (the softmax warpgroups need many registers).

### 4.6 Common misuses

| Misuse | Consequence |
|---|---|
| A new `RingPos ring;` for each tile (phase reset to zero) | Stale data from the 2nd tile on, or a hang; the `persistent` case catches it immediately |
| The producer waits on `empty` with parity 0 in the first round | Waits forever (`TIMEOUT`): no MMA has released anything yet |
| `acc_empty` initialized to 1 while all 128 threads arrive | It is "empty" after the first thread finishes reading, and the MMA overwrites rows other threads have not read yet; the other 127 arrivals also advance later phases |
| The epilogue arrives without `fence::before_thread_sync` | The TMEM read and the next tile's MMA write are unordered; sporadic errors |
| The MMA warp uses only `if (lane == 0)` while the other lanes go straight to `__syncthreads` | Relies on independent thread scheduling behavior and is hard to read; use `elect_one()` + `__syncwarp()` |
| The thread that issues MMAs changes before `tcgen05.commit` | A commit only tracks MMAs issued by the thread that executes it |
| A staging box is overwritten by the next block while TMA store is still reading it | Values of the next block appear at random in C; `bulk_wait_group_read<1>` first |
| The epilogue exits without `bulk_wait_group_read<0>` | After the CTA exits its shared memory is reclaimed, and the store reads garbage. (`.read` only guarantees that the store has finished reading its shared memory source; the write to global completes before the kernel ends, so there is no need to wait for it) |
| The tile index mapping is not one-to-one (when the last group has fewer rows) | Some tiles are never computed and some are computed twice; the `persistent` case (31 rows of tiles; with G = 8 the last group has 7 rows) catches it |

## 5. Schedule design

### 5.1 Roles and storage budget

```
block = 192 threads = 6 warps, 1 CTA per SM, tile 128 × 256, BK = 64

 warp 0..3  epilogue   warp w reads TMEM lanes 32w..32w+31 = rows 32w.. of C
 warp 4     producer   elect one lane: TMA
 warp 5     MMA        tmem_alloc(512) / dealloc; elect one lane: MMA + commit

shared memory (dynamic, 1024-aligned)
 stages    4 × (A 128×64 + B 256×64) BF16 = 4 × 48 KB = 192 KB
 staging   4 warps × 2 32×32 FP32 boxes = 32 KB
 total     225 KB (limit 227 KB); barriers and the TMEM address slot are static shared memory

TMEM        2 accumulators × 256 columns = 512 columns (all)
```

The budget is already full: 5 stages (240 KB) or 3 accumulators (768 columns) do not fit. The lab 31 epilogue reuses the stage buffers to stage C; in a persistent kernel the main loop uses the stage buffers all the time, so the epilogue needs its own staging.

### 5.2 Step 1: one tile per CTA

```
setup   : thread 0 initializes barriers; warp 5 tmem_alloc; fence / __syncthreads / fence
producer: for t in 0..k_tiles-1:  wait empty[s] (phase^1); arm full[s]; TMA A, B → stage s; advance
MMA     : for t in 0..k_tiles-1:  wait full[s] (phase); fence::after; 4 × mma; commit empty[s]; advance
          commit acc_full[0]
epilogue: wait acc_full[0] (parity 0); fence::after; 8 blocks of 32 columns: ld TMEM → staging → TMA store
teardown: fence::before; __syncthreads; warp 5: fence::after; dealloc
```

```
Step 1, one CTA
producer : ld0 ld1 ld2 ld3 ┄wait e0┄ ld4 ┄wait e1┄ ld5 ...  ld63
MMA      :     ┄wait f0┄ mma0 c  mma1 c  mma2 c  mma3 c  mma4 ...  mma63 c(acc_full)
epilogue : ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ wait acc_full ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│drain + store│
Tensor   :           ═══════════════════════════════════════════════════════════════│    idle     │
```

The producer and the MMA no longer wait for each other: the producer always runs 4 stages ahead of the MMA, and the MMA stops only when data has not arrived. But the epilogue still comes after the main loop, and the next CTA still has to wait for this CTA to exit.

### 5.3 Step 2: persistent + accumulator double buffering

```
grid = min(number of tiles, number of SMs); CTA b processes tiles b, b + gridDim.x, ...

producer: ring continuous across tiles; for each tile: produce_tile
MMA     : ring and acc continuous across tiles; for each tile:
            wait acc_empty[acc] (acc.phase ^ 1); fence::after
            mma_tile → accumulator acc, commit acc_full[acc] at the end; acc.advance
epilogue: acc continuous across tiles; for each tile:
            wait acc_full[acc] (acc.phase); fence::after
            drain: after the last ld, fence::before + arrive acc_empty[acc]; continue the stores
            acc.advance
```

```
Step 2, one CTA (3 tiles)
producer : │tile0 k0..k63 ·······│tile1 k0..k63 ··········│tile2 k0..k63 ···
MMA      :  │═════ tile0 → acc0 ═════│═════ tile1 → acc1 ═════│═════ tile2 → acc0 ═════│
                                     ↓ acc_full[0]            ↓ acc_full[1]          ↑ needs acc_empty[0]:
epilogue : ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│ld acc0│store t0 ···│┄┄┄│ld acc1│store t1 ···│   tile0's epilogue
                                             ↑ acc_empty[0] (early release)            finished reading long ago
Tensor   :   ══════════════════════════════════════════════════════════════════════════  never stops
```

- While the MMA is still computing the last few K tiles of tile 0, the producer already starts loading the first 4 stages of tile 1: the pipeline no longer "fills from empty".
- While the MMA computes tile 1 into acc1, the epilogue reads acc0 at the same time: this is the `max(T_main, T_epi)` of §3.1.
- The MMA stops on `acc_empty` only when the epilogue falls two tiles behind.

### 5.4 Step 3: grouped rasterization

```
tile index → (mb, nb), m_tiles = 5, n_tiles = 3, G = 2

            nb=0  nb=1  nb=2
   mb=0      0     2     4      ┐ group 0: down first (mb+1), then right
   mb=1      1     3     5      ┘
   mb=2      6     8    10      ┐ group 1
   mb=3      7     9    11      ┘
   mb=4     12    13    14        group 2: only 1 row left; rows in group = min(G, m_tiles − first_m)
```

```
per_group = G · n_tiles;  group = tile / per_group;  first_m = group · G
rows      = min(G, m_tiles − first_m);  in_group = tile − group · per_group
mb = first_m + in_group % rows;  nb = in_group / rows
```

All three roles use the same `tile_coords`, so they stay consistent when the order changes.

### 5.5 Why BN and the stage count are not tuned

Lab 31 already showed that BN = 256 is better than 128 (shared memory bandwidth), and this lab's budget is already full. The control table in Section 7 gives a few smaller configurations so you can confirm this.

## 6. Step-by-step exercises

### Step 1 · TODO(1a)(1b)(1c): role split

- TODO(1a): `produce_tile` (the producer line of §5.2; for the first-round parity see §3.2);
- TODO(1b): `mma_tile` (the MMA line; for which K tile is `accumulate` false? What releases a stage?);
- TODO(1c): the epilogue branch of `gemm_ws_tile`: wait on `acc_full[0]`, then call the provided `epilogue_tile` (the TMA store write-back of lab 31 step 3, changed to rotate between two staging boxes per warp).

Barrier init, TMEM allocation, role dispatch and teardown are already written.

```bash
./lab check 32 --step 1 --case one_tile
./lab check 32 --step 1
./lab bench 32 --step 1 --case square
./lab sass 32          # no spill; UTMALDG, UTCHMMA, UTCBAR, LDTM, UTMASTG
```

### Step 2 · TODO(2a)–(2d): persistent and accumulator double buffering

- TODO(2a): persistent producer: one `RingPos` for all tiles;
- TODO(2b): persistent MMA: the wait on `acc_empty` (which parity?), the fence, and writing into accumulator `acc.index`;
- TODO(2c): persistent epilogue: wait on `acc_full[acc.index]`, call `epilogue_tile` and pass in `acc_empty`;
- TODO(2d): the early release inside `epilogue_tile`: fence + arrive.

```bash
./lab check 32 --step 2 --case persistent
./lab check 32 --step 2
./lab bench 32 --step 2 --case square
./lab bench 32 --step 2 --case flat
```

The `persistent` case has 372 tiles: some CTAs process 3 tiles, with accumulators 0 → 1 → 0; each tile has 6 K tiles, so the ring buffer position shifts between tiles. It passes only if the phases are truly continuous across tiles.

### Step 3 · TODO(3a)(3b): tile order

- TODO(3a): the grouped branch of `tile_coords` (§5.4);
- TODO(3b): choose `kGroupM`. The skeleton starts at 16; try 4, 8, 16 and 32, measure all three perf cases, and compare with step 2 (row-major).

```bash
./lab check 32 --step 3 --case persistent
./lab check 32
./lab bench 32 --step 3 --case wide
./lab profile 32 --step 3 --case wide     # look at dram__bytes_read.sum
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `TIMEOUT`, step 1 | The producer waits on `empty` with parity 0 in the first round; `expect_tx` byte count; the MMA does not commit `acc_full` |
| `TIMEOUT`, step 2, small cases fine | The MMA waits on `acc_empty` with parity 0 in the first round; the epilogue does not arrive on `acc_empty`, or the arrival count does not match count |
| Step 1 all correct, step 2 wrong only on `persistent` | `RingPos` reset for each tile; the accumulator index differs from the epilogue's; the three roles call `tile_coords` with different arguments |
| Step 2 has sporadic errors, results differ between runs | `acc_empty` count is not 128; `tcgen05.fence::before_thread_sync` missing before arrive; arrive before `wait::ld` |
| Step 3 has some blocks of NaN (the output is pre-filled with NaN) | The grouped mapping misses tiles: check `rows` of the last group |
| Step 3 correct but slower than step 2 | Normal on `square` and `flat` (§7); look at `wide` |

## 7. Performance

Measured on this machine (B200, CUDA Graph; larger `vs cublas` is better):

| Version | `square` 4096³ | `wide` 8192×8192×4096 | `flat` 8192×8192×1024 |
|---|---|---|---|
| Lab 31 solution | ≈ 0.84× | ≈ 0.75× | — |
| Step 1: role split | ≈ 0.89× | ≈ 0.8× | ≈ 0.78× |
| Control: persistent, **single** accumulator | ≈ 0.92× | ≈ 0.85× | ≈ 0.89× |
| Step 2: persistent + double buffering | ≈ 0.93× | ≈ 0.87× | ≈ 0.95–0.98× |
| Step 3: grouped G = 8 | ≈ 0.92× | ≈ 0.91× | ≈ 0.95× |
| cuBLAS | 1 | 1 | 1 |

The GPU on this machine occasionally slows down across the board (cuBLAS slows down too), so read the ratios. Step 3 passes when it reaches 0.85× cuBLAS or more on all perf cases.

**Reading this table**:

- **Step 1**: the same tile and the same number of stages, only the roles are split, and `square` goes from 0.84 to 0.89. In `./lab profile 32 --step 1 --case square`, `sm__pipe_tensor_cycles_active` is about 80% (about 68% in lab 31).
- **What persistent and double buffering each handle**: the persistent version with a single accumulator already gets most of the gain (it saves CTA launches and pipeline refills, and the accumulator is released as soon as it has been read). Double buffering adds only about 1% on `square` and about 10% on `flat`: this matches the `T_epi / T_main` of §3.1 (14% vs 56%).
- **Grouped rasterization** (time change relative to step 2):

| G | `square` | `wide` | `flat` | HBM reads on `wide` |
|---|---|---|---|---|
| 0 (step 2) | — | — | — | ≈ 981 MB |
| 8 | about 1.5% slower | about 6% faster | about 1–2% slower | ≈ 507 MB |
| 16 | about 5% slower | about 8% faster | about 4% slower | |
| 32 | about 3% slower | about 10% faster | about 4% slower | ≈ 280 MB |

On `wide`, A + B is 128 MB, more than L2: with row-major order the HBM reads are about 7.7 times A + B, and with G = 32 about 2.2 times. On `square` and `flat`, A + B fits in L2, and grouping is slightly slower instead; we have not found the reason (do not treat it as a conclusion). G = 8 gives the best worst-case ratio across the three cases.

- **Control: smaller configurations** (the step 3 kernel; this set of experiments used G = 32, so the first row differs from the G = 8 solution above):

| `Config<BN, Stages, AccStages>` | `square` | `wide` |
|---|---|---|
| `<256, 4, 2>` (the solution's shape, G = 32) | ≈ 0.91× | ≈ 0.93× |
| `<256, 3, 2>` | ≈ 0.81× | ≈ 0.79× |
| `<128, 6, 2>` | ≈ 0.89× | ≈ 0.81× |
| `<128, 4, 4>` | ≈ 0.68× | ≈ 0.63× |

- **The remaining ~8%**: (1) for each tile, each K tile reads 48 KB from L2, and the 148 SMs together approach the L2 bandwidth; (2) `square` has 512 tiles, 3.46 waves, and about half of the SMs are idle in the last wave. Lab 33 handles (1) with 2-CTA MMA (256×256 tile, B shared between two SMs) and TMA multicast.

## 8. Self-check questions

1. If `acc_empty` is initialized to 1 while all 128 epilogue threads arrive, what happens?
2. Why does the producer use parity 1 the first time it waits on `empty[s]`? What about the MMA's first wait on `acc_empty[a]`?
3. S = 4, 6 K tiles per tile. For the 2nd K tile of a CTA's 3rd tile, which `full` slot and which parity does the MMA wait on?
4. The epilogue releases the accumulator before the TMA store has finished writing C. Why is this safe? What if it released it **before the last `tcgen05.ld`**?
5. Step 1 already splits the roles. Why can the epilogue still not overlap the main loop?
6. On `wide` with G = 64 (a whole column first), how much A and B data do the 148 concurrent tiles need? Compare with G = 32.

<details>
<summary>Answers</summary>

1. The phase completes as soon as the first thread arrives, the MMA starts overwriting the accumulator, and other threads may read partial sums of the next tile. The remaining 127 arrivals also accumulate into later phases, making later `acc_empty` phases complete early, so errors keep growing.
2. A freshly initialized barrier is in phase 0; waiting on parity 1 means waiting for the "previous phase", which counts as completed, so the wait passes immediately: all stages start empty. `acc_empty` works the same way: both accumulators start empty, and the MMA's first wait on each accumulator uses `phase ^ 1 = 1`.
3. This is use u = 2·6 + 1 = 13: slot 13 % 4 = 1, phase (13 / 4) & 1 = 1, so it waits on `full[1]` with parity 1.
4. The TMA store reads the staging box (shared memory), not TMEM; this thread no longer accesses TMEM once the last `wait::ld` returns. If it released before the last ld, the MMA could overwrite those columns before the ld reads them.
5. Each CTA has only one tile, so the epilogue can only run after this tile's main loop. 225 KB of shared memory allows only 1 CTA per SM, so the next tile (the next CTA) cannot start until this CTA exits.
6. G = 64: 64 rows × 2.3 columns: all of A, 64 MB, + about 4.6 MB of B ≈ 69 MB. This mirrors row-major order and is large again. G = 32 needs about 41 MB. The working set is smallest near G ≈ √(148 × 2) ≈ 17; measurements are also affected by other factors (§7).
</details>

## 9. Further reading

- FlashAttention-3 paper (Shah et al., 2024) §3.1 (producer–consumer asynchrony: warp specialization and pingpong scheduling); the intra-warpgroup overlap of §3.2 is used in lab 34.
- PTX ISA 8.7: [elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync), [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier), [tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit), [tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model), [cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
- Upstream sources (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`: `WarpCategory` (MMA, Sched, MainloopLoad, EpilogueLoad, Epilogue) and the arrival counts of each pipeline;
  - `third_party/cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp`: the load / mma roles of the mainloop;
  - `third_party/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp`: `PipelineTmaUmmaAsync` (this lab's full/empty) and `PipelineUmmaAsync` (this lab's acc_full/acc_empty); `consumer_release` uses `tcgen05.commit`;
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp`: persistent static scheduling and rasterization (raster order, swizzle).
