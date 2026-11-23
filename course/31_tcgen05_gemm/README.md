# 31 · tcgen05 and TMEM: fifth-generation Tensor Core GEMM

> Part 3 · FA3 ideas on SM100 | Prerequisites: [21_pipeline_gemm](../21_pipeline_gemm/README.md), [30_tma_mbarrier](../30_tma_mbarrier/README.md) | Code: [`students/31_tcgen05_gemm/kernel.cu`](../../students/31_tcgen05_gemm/kernel.cu)

## 1. Goals

We still compute `C[M,N] = A[M,K] · B[N,K]ᵀ` (BF16 inputs, FP32 output), but with a different Tensor Core instruction: Blackwell's **`tcgen05.mma`**. It is completely different from the `mma.sync` of labs 20 and 21:

| | `mma.sync` (labs 20, 21) | `tcgen05.mma` (this lab) |
|---|---|---|
| Who issues it | The whole warp; each of the 32 threads supplies part of a fragment | **One thread** |
| Shape of one instruction | 16×8×16 | Up to 128×256×16 (single CTA) |
| Where A and B come from | Registers (`ldmatrix` first) | **shared memory**, layout described by a 64-bit descriptor |
| Where the accumulator lives | Registers | **Tensor Memory (TMEM)** |
| When it completes | Synchronous: usable when the instruction returns | **Asynchronous**: `tcgen05.commit` signals an mbarrier |

This lab has three steps:

1. **A single tile**: TMA loads a 128×64 A and a 128×64 B, 4 MMAs run, and a 128×128 C is read out of TMEM;
2. **Full K loop**: an S-stage TMA/mbarrier ring buffer; one thread issues both TMA and MMA, and `tcgen05.commit` releases buffers;
3. **TMA store epilogue and 128×256 tile**: write the accumulator back through swizzled shared memory with TMA, then tune the tile shape.

After this lab you should be able to:

- Draw the lane/column layout of TMEM and say which warp can read which 32 lanes;
- Write the shared memory descriptor and the instruction descriptor bit by bit, and explain why the K direction only needs the start address to move by 32 bytes;
- Write the full lifecycle alloc → MMA → commit → wait → ld → dealloc, and the purpose of each `tcgen05.fence`;
- Raise the GEMM from about 0.28× cuBLAS in lab 21 to about 0.8×.

The `mbarrier` and TMA wrappers come from [`include/lab/mbarrier.cuh`](../../include/lab/mbarrier.cuh) and [`include/lab/tma.cuh`](../../include/lab/tma.cuh) (written in lab 30). From lab 32 on, the tcgen05 wrappers you write in this lab are provided by [`include/lab/tcgen05.cuh`](../../include/lab/tcgen05.cuh).

## 2. Motivation: the ceiling of mma.sync

The lab 21 solution reaches about 72% of the `mma.sync` upper bound, and that upper bound itself is only about 0.39× cuBLAS (lab 21 §7). The reason is that `mma.sync` is a "register to register" instruction: for every 16×8×16, the warp first reads A and B into registers with `ldmatrix`, and the accumulator also occupies registers. The faster the Tensor Core, the more the register bandwidth and issue slots that feed it become the bottleneck.

Hopper's `wgmma` reads operands directly from shared memory. Blackwell's `tcgen05` goes one step further: the accumulator also leaves the registers and moves into a dedicated 256 KB **Tensor Memory** on each SM. One thread issues a 128×256×16 MMA; the hardware reads A and B from shared memory by itself and accumulates the result into TMEM. The other threads are completely idle during this time and can do other work (from lab 32 they run the epilogue, from lab 34 they run softmax).

This is the hardware precondition for FlashAttention-3's second idea (overlapping GEMM with softmax) on Blackwell: the MMA is asynchronous, and the accumulator does not occupy registers.

## 3. Math: decomposing one tile

### 3.1 Tiles and instructions

A CTA computes one `BM × BN` block of C (BM = 128). The K direction is cut into K tiles of `BK = 64` (a row of 64 BF16 = 128 bytes, exactly one 128B swizzle row). Each K tile is cut again into 4 slices of `K = 16`, one MMA per slice:

```
C[m0:m0+128, n0:n0+BN] = Σ_t Σ_kk  A[m0:+128, 64t+16kk : +16] · B[n0:+BN, 64t+16kk : +16]ᵀ
                         │   └─ kk: 4 tcgen05.mma per K tile
                         └─ t: K tile
```

The `int` pattern inputs (integers in −2..2) make all products and partial sums exact in FP32, so any mismatch is a bug, not rounding.

### 3.2 How much data the Tensor Core needs

Estimating with the B200 BF16 dense peak of 2250 TFLOP/s, 148 SMs and about 1.9 GHz, each SM does about 8192 FLOP per cycle. One 128×N×16 MMA is `2·128·N·16` FLOP:

| N | FLOP per MMA | About how many cycles | shared memory read per MMA | shared memory read per cycle |
|---|---|---|---|---|
| 128 | 524,288 | ≈ 64 | A 4 KB + B 4 KB | ≈ 128 B |
| 256 | 1,048,576 | ≈ 128 | A 4 KB + B 8 KB | ≈ 96 B |

shared memory delivers about 128 bytes per cycle per SM. At N = 128 the MMA saturates it exactly, and any other access (TMA writes, epilogue) makes the Tensor Core wait. At N = 256, A is reused twice as much, which leaves headroom. This is why step 3 picks a 128×256 tile (an estimate; Section 7 verifies it with measurements).

One K tile (N = 256) has 4 MMAs, about 512 cycles, and needs 48 KB delivered by TMA. With the method of lab 21 §3.1, a latency of about 1000 cycles needs about 2 K tiles in flight; the solution uses 4 stages.

## 4. Hardware and instructions

### 4.1 Tensor Memory

Each SM has one TMEM: **128 lanes (rows) × 512 columns × 32 bits**, 256 KB in total. Addresses are 32 bits:

```
TMEM address = (lane << 16) | column

            column →  0   1   2  ...  127 128 ... 255 256 ... 511
 lane   0          ┌──────────────────────┬─────────────┬─────────┐
 lane   1          │  CTA X accumulator   │ CTA Y       │ free    │
  ...              │  128 × 128 FP32      │ accumulator │         │
 lane 127          └──────────────────────┴─────────────┴─────────┘
                    ↑ base returned by alloc (lane 0, column c0)
```

- **Allocation is by column**: one allocation takes `ncols` columns (a power of 2, 32 ≤ ncols ≤ 512) and always includes all 128 lanes. An FP32 accumulator element takes one 32-bit cell, so a 128×N accumulator needs N columns; when N is not a power of 2, round up (N = 96 allocates 128 columns).
- **One warp executes** `tcgen05.alloc` (`.sync.aligned`, all 32 threads together). It writes the base address to the shared memory location you give it.
- `tcgen05.relinquish_alloc_permit`: this CTA gives up the right to allocate more TMEM. After it, any thread of this CTA that executes `tcgen05.alloc` is illegal (the PTX ISA's wording).
- `tcgen05.dealloc`: before the CTA exits it **must** free all its columns; otherwise later CTAs on this SM wait forever when they allocate.
- For this lab's `cta_group::1`, M = 128 MMA: **accumulator row i is in lane i, and column j is in column base + j**.

### 4.2 Which warp can read which lanes

TMEM lanes are split into 4 sections, and each section can be accessed by only one warp: warp w can only access lanes `32·(w % 4)` to `32·(w % 4) + 31`. So this lab uses 4 warps (128 threads), and warp w owns exactly rows `32w .. 32w+31` of C:

```
              TMEM lane          C row (in tile) thread
 warp 0  →    0 ..  31     =     0 ..  31        lane i ↔ row i
 warp 1  →   32 ..  63     =    32 ..  63
 warp 2  →   64 ..  95     =    64 ..  95
 warp 3  →   96 .. 127     =    96 .. 127
```

### 4.3 Operands in shared memory: K-major + 128B swizzle

A K tile of A is 128 rows × 64 BF16; a K tile of B is BN rows × 64 BF16. TMA loads them with box = {64, rows} and `SWIZZLE_128B`, 128 bytes per row, with the rule of lab 30 §4.3: the 16-byte chunk q of row r is stored at `q ^ (r % 8)`.

```
              one row = 64 K = 8 chunks (8 BF16 each)
  row 0   │ k0-7 │ k8-15│k16-23│ ...                       │  ┐
  row 1   │k8-15 │ k0-7 │k24-31│ ...  (chunk q at q^1)     │  │ one swizzle atom:
  ...                                                         │ 8 rows × 128 B = 1024 B
  row 7   │k56-63│ ...                   (chunk q at q^7)  │  ┘
  row 8   │ k0-7 │ ...        (next atom, same as row 0)   │  ← exactly 1024 B = SBO from row 0
```

When the MMA reads operands, this "canonical layout" is all it needs to know: the swizzle mode and the distance between two adjacent 8-row atoms (SBO).

### 4.4 Shared memory descriptor (64 bits)

| Bits | Field | Value in this lab |
|---|---|---|
| 0–13 | start address >> 4 | `shared_addr(tile + K slice offset) >> 4` |
| 16–29 | leading dimension byte offset (LBO) >> 4 | 1 (unused for K-major + swizzle; 1 by convention) |
| 32–45 | stride dimension byte offset (SBO) >> 4 | 1024 >> 4 = 64 |
| 46–47 | version | 1 (must be 1 on SM100) |
| 49–51 | base offset | 0 (when the atom is 1024-byte aligned) |
| 52 | LBO mode | 0 |
| 61–63 | layout | 2 = `SWIZZLE_128B` (0 none, 1 128B_BASE32B, 4 64B, 6 32B) |

Example: if a K tile of A is at shared address `0x400`, the descriptor is `0x4000404000010040`.

**The 4 slices along K**: slice kk starts at K index `16·kk`, that is, at byte `32·kk`. The descriptor only needs its start address increased by `32·kk` bytes (the encoded value by `2·kk`), for example `0x...0040` → `0x...0042`. Why is this correct inside a swizzled row? Because the swizzle is an XOR applied to the **full address** (address bits[4:6] ^= bits[7:9]). The MMA computes "unswizzled logical address = start + row·128 + chunk·16" and then applies the same XOR; TMA uses the same function when it writes. As long as the atom is 1024-byte aligned, adding 32 bytes to the logical address lands exactly on K + 16. This is also the precondition for setting base offset to 0.

### 4.5 Instruction descriptor (32 bits)

| Bits | Field | Value in this lab |
|---|---|---|
| 0–1 | sparsity selector | 0 |
| 2 | sparsity | 0 |
| 3 | saturate | 0 |
| 4–5 | D format: 0 F16, 1 F32 | 1 |
| 7–9 | A format: 0 F16, 1 BF16 | 1 |
| 10–12 | B format | 1 |
| 13, 14 | negate A, B | 0 |
| 15 | A major: 0 K-major, 1 MN-major | 0 |
| 16 | B major | 0 |
| 17–22 | N >> 3 | 16 (N = 128) or 32 (N = 256) |
| 24–28 | M >> 4 | 8 (M = 128) |
| 30–31 | max shift (`.ws` only) | 0 |

M = 128, N = 256: `0x08400490`; N = 128: `0x08200490`. M starts at **bit 24** (bit 23 is reserved). Putting M at bit 23 is a common mistake.

### 4.6 MMA, commit and TMEM reads

```
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, p;
```

`D = A · Bᵀ + (p ? D : 0)`. `kind::f16` means 16-bit inputs (F16/BF16). B is given as N×K, K-major, which is exactly this course's `B[N,K]` layout, so no transpose is needed. One thread issues it, and the instruction returns immediately.

```
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];
```

This makes the mbarrier track **all tcgen05 async operations this thread issued before it**: when they have all completed, it performs one arrive on bar. "Completed" includes having finished reading the operands from shared memory, so after the commit arrives:

- the A and B buffers of that stage can be overwritten by the next TMA (`empty[s]`);
- the accumulator can be read (`mma_done`).

```
tcgen05.ld.sync.aligned.32x32b.x32.b32 {r0, ..., r31}, [taddr];
tcgen05.wait::ld.sync.aligned;
```

The `32x32b` shape: the warp reads 32 lanes, and each thread reads 32 bits × N consecutive columns of one lane (`.x32` means 32 columns; `.x8`, `.x16`, … work the same way). Thread i gets columns `taddr.col .. taddr.col + 31` of lane `taddr.lane + i`. The lane field of `taddr` must be the start of the section this warp may access (`32·(w%4)`). `tcgen05.ld` is also asynchronous: the registers are valid only after `wait::ld`.

### 4.7 tcgen05 synchronization: fence::before / after_thread_sync

tcgen05 instructions are asynchronous, and ordinary thread synchronization (`__syncthreads`, mbarrier) does not order them by itself. The rule:

```
Thread A: tcgen05 ops ...;  tcgen05.fence::before_thread_sync;
          sync (__syncthreads, or mbarrier arrive → thread B's wait)
Thread B: tcgen05.fence::after_thread_sync;  tcgen05 ops ...
```

The four places this lab uses it:

| Where | before (signaling side) | Sync | after (receiving side) |
|---|---|---|---|
| After alloc, all threads read the base address | warp 0's alloc | `__syncthreads` | all threads |
| Issue MMA after TMA data arrives | (TMA is completed by the mbarrier) | `wait(full[s])` | the MMA-issuing thread (optional, defensive, see below) |
| Read the accumulator after the MMA completes | commit (implied) | `wait(mma_done)` | all threads, then `tcgen05.ld` |
| dealloc after all lds complete | all threads | `__syncthreads` | warp 0, then dealloc |

The before column is what orders the signaling side: in rows 1 and 4 those threads execute fence::before; rows 2 and 3 need no explicit fence, since mbarrier completion and the commit provide the ordering. The after column lists the threads that must execute fence::after. `setup_tmem` has every thread execute fence::before only to keep the code uniform; outside warp 0 it is a no-op.

The after fence in the second row is not required: the TMA write to shared memory and the MMA read of shared memory both belong to the async proxy, and completion of the `full[s]` mbarrier already orders them. CUTLASS's `PipelineTmaUmmaAsync::consumer_wait` only waits on the mbarrier here, with no fence. This lab's solution keeps it as a defensive measure; the cost is negligible.

### 4.8 Common misuses

| Misuse | Consequence |
|---|---|
| M field at bit 23, or N not shifted >> 3 | All results wrong, or illegal instruction |
| SBO written as 128 (one row) instead of 1024 (8 rows) | Only the first 8 rows correct, or all wrong |
| Descriptor version bits set to 0 | Illegal on SM100 |
| A unit error in the K slice offset, such as "element count × 2 added to the encoded value before >> 4" | Every K slice reads from the wrong place; the exact comparison of `one_tile` catches it immediately |
| Shared tile not 1024-byte aligned | The swizzle does not match; alignment holds by accident in some launches only, so results are sometimes right and sometimes wrong |
| MMA issued without waiting on `full`, or a stage reused before commit | Reads stale or half-written data |
| A warp reads a lane section it does not own | Illegal access |
| Registers used after `tcgen05.ld` without `wait::ld` | Reads random values |
| Lane 0 of warp 0 is inside `if (tid == 0)` while the other lanes are already at `tcgen05.ld` | `.sync.aligned` requires a converged warp; `__syncwarp()` first |
| No dealloc before exit | Later CTAs wait forever at alloc (`TIMEOUT`) |
| `expect_tx` counts only A, or only the valid part | Hang |

## 5. Schedule design

### 5.1 Step 1: one tile

```
grid = 1, block = 128 threads; shared: A[128][64], B[128][64] (16 KB each, 1024-aligned)
barriers: full, mma_done; tmem_slot

thread 0 : mbar_init(full, 1); mbar_init(mma_done, 1); fence_barrier_init
warp 0   : tmem_alloc(&tmem_slot, 128); relinquish
all      : fence::before; __syncthreads; fence::after;  tmem = tmem_slot
thread 0 : arrive_expect_tx(full, 32 KB); TMA A(0,0); TMA B(0,0)
           wait(full, 0); fence::after (optional)
           4 × mma(tmem, descA + 32kk B, descB + 32kk B, idesc, kk > 0)
           commit(mma_done)
all      : __syncwarp; wait(mma_done, 0); fence::after
warp w   : for j in 0, 32, 64, 96:  ld 32x32b.x32 (lane 32w, col j); wait::ld; write C row 32w+lane
all      : fence::before; __syncthreads;  warp 0: fence::after; dealloc
```

The step 1 write-back is "each thread writes its own row": adjacent threads write different rows, so the stores are not coalesced. Step 3 fixes this.

### 5.2 Step 2: a one-thread TMA/MMA pipeline

```
prologue:  load(0) .. load(S−1)
for t in 0 .. k_tiles−1:
    s = t % S
    wait(full[s], (t/S) & 1); fence::after (optional)
    mma_k_tile(stage s, accumulate = t > 0)
    commit(empty[s])                       ← arrives when MMA t has finished reading stage s
    if t ≥ 1 and t−1+S < k_tiles:
        wait(empty[(t−1) % S], ((t−1)/S) & 1)   ← wait for MMA t−1, not MMA t
        load(t−1+S → stage (t−1) % S)
commit(mma_done)
```

Why refill the stage of **t−1** and not that of t? If the thread waits for MMA t to complete before issuing the next load, the Tensor Core queue runs empty: MMA t+1 cannot be issued until this thread comes back. If it issues MMA t first and then waits for MMA t−1, MMA t is queued in hardware and the Tensor Core never stops:

```
Thread 0: ─mma t─commit─┬wait e(t−1)┬load t+3─wait f(t+1)─mma t+1─commit─┬wait e(t)┬load t+4 ...
Tensor:   ═══ MMA t−1 ══════════════╪═══════════════════ MMA t ════════════════════╪═ MMA t+1 ═...
                                    └ MMA t−1 done, e arrives                      └ MMA t done
TMA:      ... tiles t+1, t+2 in flight ... tile t+3 in flight ...
```

Within one thread, `empty[s]` is both the commit target and the wait target. The n-th completion of `empty[s]` corresponds to the n-th MMA on stage s, so the parity is `((t−1)/S) & 1`.

A single thread acts as both producer and MMA issuer: any wait (for TMA data, or for an MMA to release a buffer) delays the other job. Lab 32 splits them into a TMA warp and an MMA warp, and hands the epilogue to other warps.

Step 2 configuration: BN = 128, 4 stages, 32 KB per stage, 128 KB in total (1 CTA per SM), 128 TMEM columns. The step 2 grid already covers all M×N tiles (`blockIdx` → `(m0, n0)` is already written); the write-back is still step 1's per-row store.

### 5.3 Step 3: TMA store epilogue

After all MMAs complete, the stage buffers are no longer needed (every TMA load has been waited on, and every MMA has finished reading), so they are reused to stage C:

```
warp w, column block j (32 columns): one 32×32 FP32 box, 4 KB, at staging + (w·BN/32 + j)·4 KB

   tcgen05.ld 32x32b.x32 (lane 32w, col 32j) → thread r holds the 32 values of row r
   thread r writes row r of the box: 8 float4, chunk q goes to position q ^ (r % 8)   ← C's tensor map also uses 128B swizzle
   fence.proxy.async; __syncwarp
   lane 0: TMA store box → C at (x = n0 + 32j, y = m0 + 32w)
finally: lane 0 commit; wait_group.read 0
```

- **bank conflict**: 16-byte stores take one wavefront per 8 lanes. In one wavefront, the 8 threads r = 0..7 write chunk q of their own rows; the physical positions `q ^ r` are all different and cover 32 banks. Without swizzle, the 8 threads write the same chunk position: an 8-way conflict.
- **tail**: TMA store clips out-of-bounds rows and columns; no bounds check is needed.
- **warps are independent**: each warp writes its own box and issues its own store; no CTA-level barrier is needed.

C's tensor map requires N·4 bytes to be a multiple of 16; this is where `N % 4 == 0` comes from. The row stride of A and B requires `K % 8 == 0`.

**Tuning**: BN = 256 (the shared memory bandwidth estimate of §3.2), 4 stages, 48 KB per stage, 192 KB in total; 256 TMEM columns. The staging area needs 128 KB, which fits into the stage buffers.

## 6. Step-by-step exercises

### Step 1 · TODO(1a)–(1f): wrappers and a single tile

- TODO(1a): TMEM alloc / relinquish / dealloc (§4.1);
- TODO(1b): the two descriptors (§4.4, §4.5). Work out the two examples of those sections on paper first;
- TODO(1c): mma, commit, ld, wait, and the two fences (§4.6, §4.7);
- TODO(1d): the 4 MMAs of `mma_k_tile` (the K slice offset of §4.4);
- TODO(1e): `store_direct` (§4.2);
- TODO(1f): `gemm_one_tile` (§5.1).

```bash
./lab check 31 --step 1
./lab sass 31          # should show UTCHMMA, UTMALDG, LDTM
```

`one_tile` is an exact comparison with integer inputs. Look at the first error position in the error report:

| What the error looks like | Check first |
|---|---|
| All NaN | C was not written, or `tcgen05.ld` read the wrong lane section |
| Repeats every 8 rows, or only the first 8 rows are correct | SBO |
| Regular errors every 16 K (values close but not equal) | K slice offset |
| All 0 | The `accumulate` logic, or the MMA did not execute (idesc fields misplaced) |

### Step 2 · TODO(2): K loop pipeline

```bash
./lab check 31 --step 2 --case long_k
./lab check 31 --step 2
./lab bench 31 --step 2
```

`long_k` has only one CTA and 16 K tiles, so the ring buffer wraps around 4 times. `short_k` has only 3 K tiles, so the prologue must not load a 4th.

### Step 3 · TODO(3a)(3b): TMA store epilogue and tuning

```bash
./lab check 31
./lab bench 31 --step 3 --case square
./lab bench 31 --step 3 --case wide
./lab profile 31 --step 3 --case square
```

For TODO(3b), start experimenting from the table below; run `check` and then `bench` each time:

| Experiment | `Config<BN, Stages, true>` |
|---|---|
| Starting point | `128, 4` |
| Wider N | `256, 3`, `256, 4` |
| More stages | `128, 6` |

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `TIMEOUT` | Wrong `expect_tx` byte count, parity, or commit target; or some CTA did not dealloc, and later CTAs wait at alloc |
| `illegal instruction` | Descriptor version bits, the M/N fields of idesc, the lane section of `tcgen05.ld` |
| `one_tile` right, `small` wrong | `expect_tx` used the valid byte count; the out-of-bounds check of the write-back |
| `long_k` wrong, `one_tile` and `small` right | Stage index, parity, or a stage refilled before its commit |
| In step 3 only `tail` and `raster` are wrong | Coordinate order of the TMA store (x = column n, y = row m) |
| Step 3 results misplaced within blocks of 4 columns | Wrong row number for the swizzle during staging (it should be the row r within the box) |

## 7. Performance

Measured on this machine (GPU 3, CUDA Graph; larger `vs cublas` is better):

| Version | `square` 4096³ `vs cublas` | `wide` 8192×8192×4096 `vs cublas` | `square` TFLOP/s |
|---|---|---|---|
| Lab 21 step 3 (`mma.sync`) | ≈ 0.28× | ≈ 0.26× | ≈ 400 |
| Step 2: BN = 128, per-row write-back | ≈ 0.37× | ≈ 0.33× | ≈ 540 |
| Step 3 starting point: BN = 128, TMA write-back | ≈ 0.5× | ≈ 0.42× | ≈ 740 |
| Control: BN = 256, per-row write-back | ≈ 0.5× | ≈ 0.45× | ≈ 730 |
| Step 3 (solution): BN = 256, 4 stages, TMA write-back | ≈ 0.8× | ≈ 0.75× | ≈ 1150–1200 |
| cuBLAS | 1 | 1 | ≈ 1440 |

The absolute speed of this machine's GPU sometimes drops across the board between runs (cuBLAS slows down as well), while the ratios stay relatively stable, so read the ratios. If step 3 reaches 0.6× cuBLAS or more, the mechanism works.

**Reading this table**:

- **Write-back and tile shape each account for half**. In `./lab profile`, step 2's per-row stores show up as about 99 MB of DRAM writes (C is only 67 MB: scattered partial-sector writes); with TMA write-back it is about 42 MB (the rest stays in L2). BN = 256 lowers shared memory bandwidth from "exactly saturated" to about 75% (§3.2). With both changes, `sm__pipe_tensor_cycles_active` rises from about 26% to about 68%.
- **Tuning** (`square` / `wide`, TMA write-back):

| `Config<BN, Stages>` | smem | `square` | `wide` |
|---|---|---|---|
| `<128, 3>` | 97 KB (2 CTAs per SM) | ≈ 0.34–0.6× (large variance) | ≈ 0.6× |
| `<128, 4>` | 129 KB | ≈ 0.5× | ≈ 0.42× |
| `<128, 6>` | 193 KB | ≈ 0.52× | ≈ 0.46× |
| `<256, 3>` | 145 KB | ≈ 0.62× | ≈ 0.55× |
| `<256, 4>` (solution) | 193 KB | ≈ 0.8× | ≈ 0.75× |

- **The remaining ~20%**: (1) the epilogue does not overlap the main loop: with one CTA per SM, the Tensor Core is idle while TMEM is read and C is written; (2) a single thread waits on both TMA and MMA; (3) `square` has 512 tiles on 148 SMs, 3.46 waves, so the last wave is not full; (4) every CTA re-reads A and B from L2. Lab 32 solves (1)–(3) with warp specialization, accumulator double buffering and persistent scheduling; lab 33 solves (4) with 2-CTA MMA and TMA multicast.

## 8. Self-check questions

1. With BN = 256, how many TMEM columns does one CTA's accumulator take? At most how many such CTAs can hold an accumulator on one SM at the same time?
2. Warp 2 uses `tcgen05.ld.32x32b.x32` to read row 70, columns 40–71 of the accumulator. What should taddr be (relative to base)? Which thread gets it?
3. A K tile of A is at shared address `0x8000`. Write the low 16 bits of the descriptor for the 3rd K slice (kk = 3).
4. In step 2, if "refill the stage of tile t−1" is changed to "refill the stage of tile t, waiting on `empty[t % S]` first", is the result still correct? What happens to performance?
5. Why can step 3's staging area reuse the stage buffers directly, without worrying that TMA is still writing into them?
6. If the M field of idesc is put at bit 23 by mistake (`8 << 23`), what M and N does the hardware see?

<details>
<summary>Answers</summary>

1. 256 columns (one FP32 per column). TMEM has 512 columns in total, so at most 2; but this lab's 192 KB of shared memory already limits each SM to 1 CTA.
2. Row 70 is in lane 70, which belongs to warp 2's section (64–95); taddr = `(64 << 16) | 40`, and thread 6 (70 − 64) gets row 70.
3. Start address `0x8000 + 3 × 32 = 0x8060`, `>> 4` = `0x806`; the low 16 bits are `0x0806` (14-bit address field).
4. The result is still correct, but slower: waiting on `empty[t % S]` means waiting for the MMA t that was just issued to complete. During that time the thread cannot issue MMA t+1, so the Tensor Core sits idle for one round trip between two MMAs.
5. Thread 0 issues loads only for K tiles that exist, and every load is waited on with `wait(full)` before the corresponding MMA is issued. The arrival of `mma_done` means all MMAs (and their shared memory reads) have completed. So when the staging area is written, there is no TMA write in flight and no MMA reading.
6. Bit 23 of `8 << 23` is a reserved bit, and the M field (bits 24–28) gets 4, that is, M = 64; the N field is unaffected. With M = 64 only half of the rows of C are computed, and the TMEM layout is also different, so the result is wrong.
</details>

## 9. Further reading

- PTX ISA 8.7, Section 9.7.16 TensorCore 5th Generation Family Instructions: [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory), [Matrix Descriptors (Shared Memory Descriptor)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor), [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor), [tcgen05.alloc / dealloc / relinquish_alloc_permit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions), [tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma), [tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit), [tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld), [tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)
- FlashAttention-3 paper, Sections 3.1–3.2: the hardware preconditions for asynchrony and GEMM–softmax overlap
- Upstream sources (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp`: bit fields of `SmemDescriptor` and `InstrDescriptor` (the source of the two tables in this lab);
  - `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp`: `make_umma_desc`, the canonical K-major SW128 layout and SBO;
  - `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp`: the inline PTX of `SM100_MMA_F16BF16_SS`;
  - `third_party/cutlass/include/cute/arch/tmem_allocator_sm100.hpp`: alloc / dealloc / relinquish;
  - `third_party/cutlass/include/cute/arch/copy_sm100.hpp`: `SM100_TMEM_LOAD_32dp32b*` (`tcgen05.ld.32x32b`);
  - `third_party/cutlass/include/cutlass/arch/barrier.h`: `umma_arrive` (`tcgen05.commit`).
