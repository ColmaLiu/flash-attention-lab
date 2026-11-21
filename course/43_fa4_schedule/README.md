# 43 · Scheduling and causal: block classification, persistent, LPT ordering

> Part 4 · FA4 — full optimization on Blackwell | Prerequisites: [41_fa4_softmax](../41_fa4_softmax/README.md), [32_ws_gemm](../32_ws_gemm/README.md) | Code: [`students/43_fa4_schedule/kernel.cu`](../../students/43_fa4_schedule/kernel.cu)

## 1. Goals

This is the last attention lab. Lab 41 brought the pipeline inside one CTA (warp roles, TMEM, softmax arithmetic) to about 0.95× of cuDNN. This lab no longer changes **how** a CTA computes. It changes **which CTA computes which tile, and in what order**:

| | Lab 41 | This lab |
|---|---|---|
| causal mask | each block uses a runtime check to decide whether to mask | each (Q tile, K block) is classified up front: the main loop splits into an **unmasked segment** and a **masked tail segment**, and the hot loop has no mask code |
| grid | one CTA per work tile (hundreds to thousands of CTAs) | **persistent**: grid = SM count (148), each CTA loops over several tiles |
| tile order | hardware dispatches in blockIdx order | for causal, **heavy tiles go first** (LPT), and odd rounds deal in reverse |

The interface does not change: FP16/BF16 Q, K, V, O, `[B,N,H,D]`, D = 64 or 128, causal and non-causal, outputs O and FP32 LSE. Three steps:

1. **Block classification**: compute how many blocks of each Q tile need no mask, and split the softmax loop into two segments;
2. **Persistent**: grid = min(tile count, SM count). Every role walks the same tile sequence. Q's shared memory and O's TMEM are handed over between two adjacent tiles;
3. **LPT**: for causal, sort so that "the last tile pair goes first", and deal tiles to CTAs in a "snake" order (odd rounds reversed).

Acceptance for this lab: **all 5 perf cases reach 0.9× of cuDNN** (median of the same CUDA Graph timing run). The new `perf_causal_bh16` (B·H = 16, N = 8192, causal) is the case where scheduling decides the result: lab 41 reaches only about 0.73× of cuDNN on it.

After this lab you should be able to:

- for any (N, causal), write down which blocks each tile visits and which of them need a mask, and compute each tile's work;
- explain what a persistent kernel saves and what it does not save (for example, it cannot remove the "wave quantization" that comes from a tile count that is not a multiple of the SM count);
- derive the completion time (makespan) upper bounds of four assignments: round robin, greedy, LPT and snake, and verify them with `sm__cycles_active.max / .avg`;
- use Nsight Compute metrics to explain the remaining gap between this lab's solution and FA4.

## 2. Motivation: where lab 41 gets stuck

Measurements of the lab 41 solution (step 3) (GPU 5, CUDA Graph; `vs cudnn` = cuDNN time / time of this version):

| case | shape | lab 41 vs cuDNN | FA4 vs cuDNN |
|---|---|---|---|
| `perf` | [4,4096,16,128] BF16 | ≈ 0.96× | ≈ 1.16× |
| `perf_causal` | same, causal | ≈ 1.04× | ≈ 1.33× |
| `perf_d64` | [4,4096,32,64] FP16 | ≈ 0.96× | ≈ 1.15× |
| `perf_8k` | [2,8192,16,128] BF16 | ≈ 0.94× | ≈ 1.10× |
| **`perf_causal_bh16`** | **[1,8192,16,128] BF16, causal** | **≈ 0.73×** | ≈ 1.14× |

On the first four cases lab 41 is already close to cuDNN. On the last one it is a quarter behind. The only difference from `perf_causal` is **small B·H, large N**: 512 work tiles; the heaviest tile has 64 K/V blocks, the lightest only 2. Nsight Compute gives a direct clue (locked clocks):

```
Lab 41 structure (step 1), perf_causal_bh16:
  sm__cycles_active.avg  ≈ 363,000 cycles
  sm__cycles_active.max  ≈ 482,000 cycles      max / avg ≈ 1.33
```

The busiest SM does 33% more work than the average: **some SMs go idle early, and the kernel waits for the slowest one**. The CTA is not computing slowly inside; the work is split unevenly.

The FlashAttention-2 paper (Dao, 2023), Section 3.2, discusses parallelizing along the sequence length and skipping blocks for causal. The FA4 implementation (`flash_attn/cute/tile_scheduler.py`) uses a dedicated `SingleTileLPTScheduler` for causal: it sorts by "longest processing time first" and splits heads into sections by L2 capacity. This lab builds these scheduling techniques one by one and measures what each one is worth.

## 3. Math

Notation: a **work tile** is one pair of Q tiles (256 rows, rows q0 .. q0 + 255, q0 = 256·p, p is the pair index) × one head × one batch. K/V block j is keys 128 j .. 128 j + 127. Each head has `P = ⌈N / 256⌉` work tiles (one per pair), `T = P · B · H` in total.

### 3.1 Causal block classification

For Q tile rows [qt, qt + 127] and K block j (keys [k0, k0 + 127], k0 = 128 j), the causal visibility condition is key ≤ row:

| Class | Condition | Handling |
|---|---|---|
| fully visible | k0 + 127 ≤ qt (largest key in the block ≤ smallest row), and k0 + 128 ≤ N | no mask needed |
| diagonal / out of bounds | the above fails, but at least one (row, key) is visible | needs a mask |
| fully masked | k0 > qt + 127 (smallest key in the block > largest row), or k0 ≥ N | skip (not loaded) |

qt is a multiple of 128, so the "fully visible" blocks are exactly **a run of consecutive blocks at the start**: once a block's largest key exceeds qt or reaches N, every later block does too. `num_unmasked_blocks` in the skeleton returns this count. Verify it yourself with the two conditions in the table and the diagram below.

The two Q tiles of a work tile share one K/V stream (lab 40): pair p visits blocks 0 .. `min(⌈N/128⌉, 2p + 2) − 1`. Below is the class of each block for each Q tile with N = 1024, causal (`F` no mask, `D` needs a mask, `X` fully masked for tile 0 but computed anyway because tile 1 uses it, `.` skipped):

```
               K block j →  0 1 2 3 4 5 6 7    blocks   masked
pair 0  tile 0 (row   0)    D X . . . . . .      2        2
        tile 1 (row 128)    F D . . . . . .      2        1
pair 1  tile 0 (row 256)    F F D X . . . .      4        2
        tile 1 (row 384)    F F F D . . . .      4        1
pair 2  tile 0 (row 512)    F F F F D X . .      6        2
        tile 1 (row 640)    F F F F F D . .      6        1
pair 3  tile 0 (row 768)    F F F F F F D X      8        2
        tile 1 (row 896)    F F F F F F F D      8        1
```

- **Work is proportional to the pair index**: pair p has 2p + 2 blocks. Pair 3 is 4 times pair 0. At N = 8192 the last pair has 64 blocks, 32 times the first pair.
- **Very few blocks need a mask**: each pair has 3 (tile, block) combinations that need a mask, independent of N. `perf_causal` (N = 4096) averages 17 blocks per pair, and the masked ones are only 3/34 ≈ 9%. So moving the mask out of the hot loop mainly makes the hot loop cleaner (registers, instructions); the performance gain is small (§7.1). Lab 41 already skips most mask code with a runtime branch that is "uniform over the whole block".
- The `X` in the table: tile 0's last block is fully masked for it, so softmax gives P = 0 and the result does not change. We still compute it (FA4 does the same), which costs about 1/(2·(p+1)) extra work for causal. Skipping it exactly requires the two tiles to have different block counts; see §8 question 5.

When N is not a multiple of 128, the last block is "out of bounds" for both tiles and also goes into the masked tail segment (cases `causal_n_bound`, `causal_tail`, and so on).

### 3.2 Work, lower bound, and "wave quantization"

Let the time of one block (one QKᵀ, softmax and PV for each of the two tiles) be 1 unit. Total work W = Σ blocks over all tiles, m = 148 SMs. The completion time C of any assignment satisfies

```
C ≥ max( W / m ,  w_max )        (w_max: the heaviest single tile)
```

| case | T (tile count) | blocks per tile | W / m | w_max |
|---|---|---|---|---|
| `perf` | 1024 | 32 | 221.4 | 32 |
| `perf_d64` | 2048 | 32 | 442.8 | 32 |
| `perf_8k` | 1024 | 64 | 442.8 | 64 |
| `perf_causal` | 1024 | 2, 4, …, 32 (64 of each) | 117.6 | 32 |
| `perf_causal_bh16` | 512 | 2, 4, …, 64 (16 of each) | 114.2 | 64 |

**Wave quantization**: for non-causal, all tiles weigh the same. 1024 tiles over 148 SMs is 1024 / 148 = 6.92, so some SMs must do 7. C = 7 × 32 = 224 > 221.4, a 1.2% loss.

```
perf: 1024 tiles, 148 SMs, each cell = one tile (32 blocks)
SM   0 │■■■■■■■│
...    │■■■■■■■│        136 SMs do 7
SM 135 │■■■■■■■│
SM 136 │■■■■■■ │        12 SMs do 6 and sit idle for the last 1/7 of the time
...    │■■■■■■ │
SM 147 │■■■■■■ │
```

Note: **persistent does not remove this**. The granularity of work is still one tile, and with 1024 tiles 136 SMs still have to do 7. What persistent saves is the overhead between one tile and the next (§3.5). The only ways to remove quantization are smaller tiles or splitting one tile's work across SMs (split-KV); this course does not do either.

### 3.3 Greedy (list scheduling) and its upper bound

Without persistent (lab 41, step 1 of this lab), each CTA processes one tile. When an SM frees up, the hardware dispatches the next CTA in blockIdx order. This amounts to **greedy scheduling**: in the given order, each task goes to the first idle machine. Its classic upper bound (Graham, 1966):

```
C_greedy ≤ W / m + w_last        (w_last: work of the last task to start on the machine that determines C)
         ≤ W / m + w_max
```

Proof in one line: when the last-finishing task starts, every machine is busy (otherwise it would have started earlier), so its start time ≤ W / m.

In the natural order (blockIdx.x = pair varies fastest), the last tiles dispatched are the heavy tiles of the last head, so w_last can be close to w_max. For `perf_causal_bh16` the bound is 114 + 64 = 178, almost 1.6 times the lower bound. The measured max / avg ≈ 1.33 (§2).

### 3.4 LPT and snake dealing

**LPT** (longest processing time first): sort by work from largest to smallest, then run greedy. The tasks that start last are all light, so w_last is small. Graham proved that LPT gives C ≤ (4/3 − 1/(3m))·optimal. Under causal the sort is very simple: pair p has work 2p + 2, so **"layer"** ℓ = 0 holds the last pair (p = P − 1) of every (batch, head), ℓ = 1 holds the second-to-last pair, and so on.

After going persistent, no hardware does the "greedy" for us: CTA b decides by itself which tile is its k-th. The simplest is **round robin**, `idx = k·G + b` (G = gridDim.x). Round robin over the sorted sequence does not work well: in every round CTA 0 takes the heaviest tile of that round,

```
Round robin: total of CTA b = Σ_r w(rG + b)
CTA 0 − CTA G−1 = Σ_r [w(rG) − w(rG + G − 1)] ≤ w_max − w_min     (telescoping: w(rG+G−1) ≥ w((r+1)G))
```

For `perf_causal_bh16`: w_max − w_min = 62, more than half the average.

**Snake** (boustrophedon, "as the ox plows"): even rounds deal forward, odd rounds deal in reverse. The CTA that gets the lightest tile in one round gets the heaviest in the next. Over the two rounds, CTA b gets `x_b + y_(G−1−b)`: x decreases with b, y_(G−1−b) increases with b, and their differences cancel. Let Δx be the range of x in its round and Δy the range of y. Then the difference in total work of any two CTAs over these two rounds lies in [−Δy, Δx]. If the work of the two rounds decreases roughly linearly (exactly the causal case: each layer has 2 blocks fewer than the one before), the sum is almost constant.

Worked example by hand: 3 SMs, 2 heads × 4 pairs (work 2, 4, 6, 8; lowercase is head 0, uppercase is head 1), W / m = 40 / 3 ≈ 13.3:

```
Hardware greedy, natural order (lab 41)     Static round robin, natural order (step 2)
SM0 |aaddddddddDDDDDDDD| 18                 SM0 |aaddddddddCCCCCC| 16
SM1 |bbbbAABBBB        | 10                 SM1 |bbbbAADDDDDDDD  | 14
SM2 |ccccccCCCCCC      | 12                 SM2 |ccccccBBBB      | 10

LPT + round robin                           LPT + snake (step 3)
SM0 |ddddddddCCCCCCaa| 16                   SM0 |ddddddddBBBBaa| 14
SM1 |DDDDDDDDbbbbAA  | 14                   SM1 |DDDDDDDDbbbbAA| 14
SM2 |ccccccBBBB      | 10                   SM2 |ccccccCCCCCC  | 12
```

The same simulation for this lab's perf cases (each tile adds 0.5 blocks of switching overhead, 1.5 blocks without persistent; the numbers are C / lower bound):

| Assignment | `perf_causal` | `perf_causal_bh16` | Corresponds to |
|---|---|---|---|
| hardware greedy, natural order | 1.24 | 1.39 | lab 41, step 1 |
| static round robin, natural order | 1.28 | 1.54 | step 2 |
| static round robin, LPT order | 1.14 | 1.29 | §7.2 variant A |
| hardware greedy, LPT order | 1.08 | 1.07 | FA4's causal approach; §7.2 variant B |
| **static snake, LPT order** | **1.02** | **1.05** | **step 3** |
| dynamic greedy (atomic counter), LPT order | 1.02 | 1.05 | not implemented, for reference |

The measured max / avg (§7.2) follows the trend of this table. On these two cases snake is as good as true greedy LPT. It is static: each role computes the tile sequence by itself, with no atomics and no communication between roles (§5.5). Its weakness: if work varies a lot within one round (for example, B·H is very small and one round spans many layers), the cancellation is incomplete.

### 3.5 What persistent saves

When a CTA finishes one tile and moves to the next, without persistent it goes through:

```
CTA ends:  wait for all MMAs and for the epilogue to write O → TMEM freed → CTA exits
New CTA:   launch → barrier init, TMEM alloc, tensor map prefetch → TMA loads Q, K_0 (full latency) → first QKᵀ …
```

During this time the Tensor Cores and MUFU sit idle. With persistent: the TMA warp starts loading the next tile's K_0 and Q while the last few blocks of the current tile are still computing (as long as the ring buffer has a free slot and the previous user of Q has finished). While the correction warpgroup writes the previous tile's O, MMA is already computing the next tile's QKᵀ. §7.1: the three non-causal cases get about 4–5% faster from this, which is about 1.5 blocks of time saved per tile.

### 3.6 L2: can one head's K/V stay in L2

One head's K + V is `2 · N · D · 2` bytes: 2 MB for `perf` / `perf_causal`, 4 MB for `perf_causal_bh16`. All P tiles of the same head read it (a prefix of it for causal). If these tiles run at the same time, K/V is read from HBM only once, and the rest comes from L2 (126 MB).

- Natural order (pair varies fastest): the 16 tiles of one head have adjacent indices and run at the same time. All non-causal cases use this order.
- LPT order: layer 0 is the last pair of **every** head, so every round reads the K/V of every head. `perf_causal` has 64 heads in total, 128 MB, which does not fit in L2, so every round re-reads from HBM.
- FA4's `SingleTileLPTScheduler` splits heads into sections: the K/V of one section is at most about 50 MB (16 heads for `perf_causal`), LPT within a section, sections run one after another. It relies on hardware greedy dispatch to balance load (FA4's causal path is not persistent).

There is a real conflict here: **perfect LPT wants the heavy tiles of every head first; L2 locality wants the tiles of one head together**. For static snake dealing, sections break the precondition that "per-round work is monotonic" (in the `perf_causal` simulation C / lower bound goes from 1.02 to 1.43). §7.2 measures this cost: on this lab's perf cases, the extra DRAM reads (`perf_causal` goes from 201 MB to 535 MB) do not slow the kernel down, because HBM bandwidth is far from saturated; sections instead slow it down through load imbalance. So step 3 does not use sections; within a layer it only orders tiles by (batch, head).

## 4. Hardware and instructions

This lab has no new instructions; everything it uses was covered earlier:

### 4.1 SM count and CTAs per SM

```cpp
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);   // B200: 148
```

In this lab one CTA uses about 198 KB of shared memory (D = 128: Q 64 KB + K/V ring 128 KB + statistics 5 KB) and all 512 TMEM columns, so each SM fits only 1 CTA, and the persistent grid = min(T, 148). If the grid is larger than the SM count, the extra CTAs must wait for others to finish **entirely** before they can start, and persistent degrades into two waves.

### 4.2 How the hardware dispatches CTAs

Without persistent, CTAs are dispatched to idle SMs in linear blockIdx order. This is observed behavior (FA4's LPT scheduling relies on it); the CUDA programming guide does not guarantee the execution order of CTAs. The static schedule in step 3 of this lab does not rely on it: each CTA computes its own tile sequence.

### 4.3 Releasing shared memory with `tcgen05.commit`

`tcgen05.commit` makes an mbarrier receive one arrive **after all MMAs issued earlier by this thread have completed** (labs 31, 32). So the MMA warp can commit `q_empty[t]` as soon as it has issued a tile's last Q^t Kᵀ: when this arrive lands, every MMA that reads Q^t has completed, and TMA can load the next tile's Q into Q^t. There is no need to wait for PV.

### 4.4 Phases across tiles

Each time an mbarrier completes a phase, its parity flips (lab 30). With persistent, the same barrier is used by many tiles inside one CTA, so phases must be **counted continuously**:

| barrier | uses per tile | which phase | parity to wait on |
|---|---|---|---|
| `kv_full` / `kv_empty` | 2 × blocks (ring) | item number u in the ring (continuous across tiles) | `(u / S) & 1`, and `^ 1` for empty |
| `s_full`, `p_ready`, `stats_full` | blocks | `nb + j` (nb = total blocks of all earlier tiles) | `(nb + j) & 1` |
| `q_full`, `o_full` | 1 | `it` (number of tiles done before) | `it & 1` |
| `final_full[2]` (double-buffered by tile parity, §5.4) | 1 (the two alternate) | tile `it` uses phase `it >> 1` of `final_full[it & 1]` | `(it >> 1) & 1` |
| `q_empty` (new in this lab) | 1 | waits for "the release by the previous tile" | derive it yourself (next paragraph) |

The first wait on an "empty"-type barrier (`kv_empty` included) must return immediately: for a freshly initialized mbarrier, "phase −1" counts as completed, so a wait on parity 1 returns immediately (lab 30). Tile `it` waits for release number `it − 1`. In the skeleton `for_each_tile` already computes `it` and `nb`, and every role calls it, so all roles see the same tile sequence and phases.

### 4.5 O is written directly to global memory

The epilogue in labs 34–41 writes O into Q's shared memory and then uses a TMA store. With persistent, that shared memory has already been handed to the next tile's Q (§4.3). This lab's epilogue has each thread write its own row (D 16-bit values = 128 or 256 bytes, contiguous) directly with 16-byte `st.global`. The 32 lanes of one warp instruction land on 32 different rows, so each 32-byte sector is only half written (LSU sector utilization 50%). The other half of the same sector is written by the next instruction, and L2 merges the two halves before writing back to HBM, so no HBM traffic is wasted. O takes only a small part of the epilogue time, so this LSU efficiency is acceptable.

### 4.6 Common misuses

| Misuse | Consequence |
|---|---|
| persistent grid larger than the SM count | the extra CTAs start only after the first wave ends; the tail doubles |
| phases restart from 0 across tiles (for example `j & 1`) | wrong parity from the second tile on: a wait passes too early or never completes (hang); shows up in the `multi_tile*` cases |
| loading the next tile's Q without waiting for `q_empty` | the previous tile's last QKᵀ reads the new Q: errors appear at random |
| not resetting m, ℓ at the start of each tile | the reference of the conditional rescale carries over from the previous tile: `multi_tile_causal` (jump_max) fails |
| different roles compute different tile sequences | immediate hang or reading the wrong head |

## 5. Schedule design

### 5.1 What stays the same

Warp roles, TMEM layout (S⁰/P⁰, S¹/P¹, O⁰, O¹, 512 columns), the K/V ring, and the softmax arithmetic (lab 41 step 3: MUFU/polynomial mix, conditional rescale) all stay the same:

```
16 warps:
  warp 0–3   softmax WG 0 (tile 0)   setmaxnreg 192
  warp 4–7   softmax WG 1 (tile 1)   setmaxnreg 192
  warp 8–11  correction WG (corrects O + epilogue)   setmaxnreg 80
  warp 12    MMA   warp 13 TMA   warp 14–15 idle   setmaxnreg 48
```

### 5.2 Resource lifetimes

With persistent, every resource has to answer "when may the next tile overwrite it, and who signals that":

| Resource | Last use by the previous tile | First write by the next tile | Handover |
|---|---|---|---|
| K/V ring stage | the last MMA that reads it | TMA loads a new item | `kv_empty` (same as lab 40, except item numbers continue across tiles) |
| Q^t (shared) | the last Q^t Kᵀ | TMA loads the next Q^t | **`q_empty[t]`**: MMA commits after the last QKᵀ; TMA waits on it |
| S^t / P^t (TMEM) | the last P^t V | the next tile's Q^t K₀ᵀ | the same thread executes in issue order (lab 40 §4.3); no barrier needed |
| O^t (TMEM) | the epilogue's `tcgen05.ld` | the next tile's first P^t V (overwrites) | **already implied by `p_ready`**, no new barrier needed: before this PV, MMA waits on `p_ready[t]` (256 arrives, 128 of them from correction), and correction is a sequential program: it arrives for block 0 of the next tile only after it has finished the previous tile's epilogue (`tcgen05.ld` completed and fenced) |
| `st.alpha` | correction reads α | softmax writes the next block's α | same as lab 40: softmax is at most one block ahead (§5.4) |
| `st.fin_m/fin_l` and `final_full` | epilogue reads / waits | softmax writes the next tile's final values / arrives | **both data and barrier are double-buffered by tile parity** (§5.4) |

### 5.3 Timeline: handover between two adjacent tiles (schematic, not to scale)

```
          ┌─────── last block of tile w ────────┐┌──────────── tile w+1 ────────────────
TMA     : … K_last V_last │K_0(w+1)│ wait q_empty[0] → Q⁰(w+1)  Q¹(w+1) │V_0 K_1 …
MMA     : PV⁰ QK⁰ PV¹ QK¹ ──── commit q_empty ──┐ PV⁰_last o_full[0] PV¹_last o_full[1] │QK⁰_0 QK¹_0 │ wait p_ready │PV⁰_0 …
softmax0: … softmax⁰(last) │ fin → final_full │ wait s_full │ softmax⁰(block 0) …
correct : … correct │ epilogue O⁰(w) │ epilogue O¹(w) │ tile w+1, j = 0: arrive p_ready …
```

- Q^t is released right after the last QKᵀ, and TMA loads the next Q while PV_last is still computing, so the next tile's QKᵀ_0 starts right after PV_last;
- the next tile's softmax of block 0 runs at the same time as the previous tile's epilogue;
- the only wait: the next tile's first PV waits for the correction warpgroup to finish both epilogues of the previous tile (correction is a sequential program: epilogue first, then it enters the next tile's block loop and arrives `p_ready` for block 0).

### 5.4 Why both fin and final_full are double-buffered

At the end of a tile, softmax writes the final m, ℓ into `st.fin_*` and arrives `final_full`; the epilogue later waits on it and reads the values. When does the next tile's softmax write and arrive again? It first needs the next tile's S_0, and S_0 depends only on Q and K_0: **it does not depend on correction**. If the next tile has only 1 block (N ≤ 128), softmax may write new values for tile w + 1 and arrive while correction is still writing the O of tile w. Two problems:

1. **Data**: the m, ℓ of tile w are overwritten;
2. **Barrier**: with only one `final_full`, it completes two phases in a row (w and w + 1). Correction then waits on parity `w & 1`, and this parity now refers to "phase w + 2", which never comes: **hang** (`many_one_block`: 2000 tiles of 1 block each). An mbarrier's parity can tell apart only two adjacent phases, so **the producer may be at most one phase ahead of the waiter**.

So both data and barrier have two copies selected by tile parity (`it & 1`): tile `it` uses `final_full[it & 1]`, and that is its phase `it >> 1`. Are two copies enough? Before writing tile w + 2, softmax must get the S_0 of tile w + 2. MMA issues it after the PV of tile w + 1, and that PV waits for correction to arrive `p_ready` for tile w + 1; by then correction has already waited for and read the values of tile w. Softmax is at most one tile ahead.

`st.alpha` does not need it: before softmax writes the α of the next block (including block 0 of the next tile) it waits on `s_full`. The corresponding QKᵀ comes after this block's PV in the MMA program, and that PV waits on `p_ready`, which correction arrives after it has read α.

### 5.5 Tile sequence

```cpp
// every role:
for_each_tile<C, kCausal>(p, [&](const Tile &w, int blocks, int it, int nb) { ... });

for k = 0, 1, … (k·G < T):
    idx = tile_index<C>(k)          step 1: G = T, only k = 0; step 2: round robin; step 3: snake
    idx ≥ T: skip (when the snake's last round is not full, reverse dealing lands outside T)
    w = tile_coords(idx)            step 3 + causal: LPT mapping; otherwise pair varies fastest
    body(w, num_kv_blocks(w.q0), it, nb); it += 1; nb += blocks
```

## 6. Step-by-step exercises

The skeleton already contains the whole pipeline (TMA, MMA, one softmax block, correction, epilogue, host side). Every role already loops with `for_each_tile`, and phases are already counted continuously across tiles. The three steps complete the same kernel bit by bit; later steps use the TODOs of earlier ones.

### Step 1 · TODO(1a)–(1b): block classification and loop split

- 1a: `num_unmasked_blocks(qt, n)` (§3.1): how many leading blocks of the tile need no mask at all;
- 1b: in the softmax warpgroup, first run `softmax_block<false, …>` for the first `min(unmasked, blocks)` blocks, then run `softmax_block<true, …>` for the rest.

```bash
./lab check 43 --step 1 --case causal_pair       # diagonal block + the fully masked block of tile 0
./lab check 43 --step 1 --case causal_n_bound    # diagonal mask and N tail at the same time
./lab check 43 --step 1 --case short_tail        # non-causal, N tail only
./lab check 43 --step 1
./lab bench 43 --step 1 --case perf_causal_bh16
```

### Step 2 · TODO(2a)–(2c): persistent

- 2a (host): grid = min(T, SM count);
- 2b (TMA): wait on `q_empty[t]` before loading Q^t; after the loop, wait for the last `q_empty` once more (drain);
- 2c (MMA): commit `q_empty[t]` after the last QKᵀ that reads Q^t (two places: tiles with only 1 block are in the prologue).

O^t's TMEM needs no separate handover (§5.2). The double buffering of `final_full` (§5.4) is already in the skeleton; read it and understand why it is needed.

```bash
./lab check 43 --step 2 --case multi_tile        # 450 tiles, some CTAs do 4
./lab check 43 --step 2 --case many_one_block    # 2000 tiles of 1 block: q_empty in the prologue
./lab check 43 --step 2 --case bh1_causal        # 8 tiles < 148 SMs
./lab check 43 --step 2
./lab bench 43 --step 2 --case perf
./lab bench 43 --step 2 --case perf_causal_bh16  # expected to be slower than step 1 (§3.4)
```

### Step 3 · TODO(3a)–(3b): LPT and snake

- 3a: snake dealing in `tile_index` (§3.4);
- 3b: LPT mapping in `tile_coords` (the "layers" of §3.4).

```bash
./lab check 43 --step 3 --case uneven_causal     # V[j] = j/N: a wrong pair reads another position
./lab check 43 --step 3
./lab bench 43 --case perf_causal_bh16
./lab bench 43 --case perf_causal
./lab profile 43 --case perf_causal_bh16         # look at sm__cycles_active.max / .avg
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| step 1 causal cases show values from "future positions" (`causal_position`) | `num_unmasked_blocks` uses an upper bound such as `(qt + 127) / 128`, which counts the diagonal block as unmasked |
| step 1 `short_tail` and `causal_n_bound` fail, others pass | forgot the N limit (`⌊N / 128⌋`), or did not clamp unmasked to blocks |
| step 1 `TIMEOUT` | the two loop segments skip or repeat a block: MMA and softmax disagree on the block count |
| step 2 only `multi_tile*` and `many_one_block` hang | wrong `q_empty` parity (the first tile must not block; later ones wait for the previous release) |
| step 2 only `n1`, `one_block`, `causal_n1`, `many_one_block` hang | tiles with only 1 block (N ≤ 128) do not commit `q_empty` in the prologue |
| step 3 `uneven_causal` fails | the LPT pair is not reversed (layer 0 should be the last pair of each head), or the split into layer and bh is swapped |
| step 3 passes but `perf_causal_bh16` is not faster | did LPT but not snake ("LPT + round robin" in §3.4) |

## 7. Performance

### 7.1 Against the library baselines

Measured on GPU 5 (CUDA Graph; `vs cudnn` = cuDNN time / time of this version, larger is better; each row comes from one `bench` run; cuDNN itself varies by about ±4% between runs):

| Version | `perf` | `perf_causal` | `perf_d64` | `perf_8k` | `perf_causal_bh16` |
|---|---|---|---|---|---|
| lab 41 solution | ≈ 0.96× | ≈ 1.04× | ≈ 0.96× | ≈ 0.94× | ≈ 0.73× |
| step 1 (block classification) | ≈ 0.97× | ≈ 1.04× | ≈ 0.95× | ≈ 0.95× | ≈ 0.78× |
| step 2 (persistent, round robin) | ≈ 1.02× | ≈ 1.04× | ≈ 1.02× | ≈ 0.97× | ≈ 0.69× |
| step 3 (LPT + snake) | ≈ 1.03× | ≈ 1.27× | ≈ 1.01× | ≈ 0.99× | ≈ 0.97× |
| FA4 (`flash_attn.cute`) | ≈ 1.17× | ≈ 1.31–1.33× | ≈ 1.15× | ≈ 1.10× | ≈ 1.14× |

Target for this lab: step 3 reaches 0.9× of cuDNN or better on all 5 perf cases.

- **Step 1** is within noise of lab 41 (the 0.73 → 0.78 on `perf_causal_bh16` is mostly the difference between two cuDNN measurements; the absolute time is only about 2% faster). The reason is in §3.1: masked blocks are only about 9%, and lab 41 already skips the mask code of the other blocks with a branch that is uniform over the whole block. The value of this step is to make the classification clear and to prepare "the block count of each tile" for the next two steps.
- **Step 2**: the three non-causal cases get 4–5% faster (§3.5: the overhead between tiles is overlapped). `perf_causal_bh16` gets slower instead: hardware dispatch was a greedy "give it to whoever is idle", and static round robin turns it into a fixed assignment (1.39 → 1.54 in the §3.4 table).
- **Step 3**: `perf_causal_bh16` 0.69 → 0.97, `perf_causal` 1.04 → 1.27. The non-causal cases are not affected (all tiles weigh the same, and the order does not change).

### 7.2 Load balance: measured max / avg and control experiments

`sm__cycles_active.max / sm__cycles_active.avg` from `./lab profile` (locked clocks):

| Version | `perf` | `perf_causal` | `perf_causal_bh16` |
|---|---|---|---|
| step 1 | 1.02 | 1.16 | 1.33 |
| step 2 | 1.03 | 1.27 | 1.52 |
| step 3 | 1.03 | 1.03 | 1.04 |
| FA4 | 1.03 | 1.09 | 1.07 |

`sm__cycles_active.avg` drops by about 4–7% from step 1 to step 2 (same work, less idle time between tiles). This is the gain of §3.5; `.max` is what decides the kernel time.

Control experiments (only the schedule changes, the rest is the same as step 3; ratio of absolute times, smaller is better, step 3 = 1):

| Variant | `perf_causal` | `perf_causal_bh16` | Notes |
|---|---|---|---|
| step 3: persistent, LPT + snake | 1 | 1 | |
| A: persistent, LPT + round robin | 1.09 | 1.13 | §3.4: CTA 0 takes the heaviest tile every round |
| B: not persistent, LPT order (hardware greedy) | 1.07 | 0.97 | FA4's causal approach (no sections) |
| C: B + FA4's head sections (16 / 8 heads per section) | 1.08 | 0.98 | better L2, slightly worse balance |
| D: persistent, snake + head sections | 1.27 | 1.26 | sections break per-round monotonicity, so snake stops working (§3.6) |

- On `perf_causal_bh16`, hardware greedy (B) is slightly better than snake, by about 3%: true greedy adapts to the actual time each tile takes, while a static assignment can only estimate by "block count". The cost is relaunching a CTA for every tile. On `perf_causal` (1024 shorter tiles) the launch overhead takes a bigger share, and B is 7% slower than step 3.
- L2: on `perf_causal`, step 3 reads 535 MB from HBM (L2 hit rate 52%), while steps 1, 2 and FA4 read only about 201 MB (the minimum: 64 MB each for Q, K, V; steps 1 and 2 have hit rates of about 72–74%). But 535 MB / 0.44 ms ≈ 1.2 TB/s, only about 15% of nominal HBM bandwidth, so the extra reads are hidden behind compute: C is only 1% worse than B. **On this lab's cases, load balance matters much more than L2**. With larger B·H·N (K/V far larger than L2, and shorter compute), the conclusion could differ.

### 7.3 Where the remaining gap is (compared with FA4)

Nsight Compute (locked clocks, `perf` and the causal cases):

| | Tensor Core busy (`sm__pipe_tensor_cycles_active`) | MUFU (`sm__inst_executed_pipe_xu`) | max / avg |
|---|---|---|---|
| step 3, `perf` | 68% | 56% | 1.03 |
| FA4, `perf` | **83%** | **73%** | 1.03 |
| step 3, `perf_causal` | 63% | 52% | 1.03 |
| FA4, `perf_causal` | 68% | 60% | 1.09 |
| step 3, `perf_causal_bh16` | 66% | 55% | 1.04 |
| FA4, `perf_causal_bh16` | 75% | 66% | 1.07 |

How to read this table:

- On scheduling this lab no longer loses to FA4: its causal max / avg is even lower than FA4's. The remaining gap is entirely in **per-block efficiency**: the time per block per SM. FA4's Tensor Cores are busier than ours: 15 percentage points more on `perf` (MUFU 17 more), 9 more on `perf_causal_bh16` (MUFU 11 more), 5 more on `perf_causal` (MUFU 8 more). This shows that its "chain" (softmax → PV → QKᵀ → softmax, lab 40 §3.1) is shorter.
- How large the per-block efficiency gap is: Tensor Core busy 68% vs FA4's 83% on `perf`, 66% vs 75% on `perf_causal_bh16`, 63% vs 68% on `perf_causal`. For non-causal with D = 128, FA4 combines persistent scheduling, 2-CTA MMA (`use_2cta_instrs` in `interface.py`) and other tuning. Lab 42 measured 2-CTA added **alone** on top of lab 41's softmax (not persistent): the main loop broke even and the whole kernel was about 2.5% slower. This course has not measured persistent + 2-CTA. So from our measurements we **cannot** attribute the roughly 12–15% gap to FA4 on `perf` and `perf_8k` to 2-CTA. Candidate causes (all hypotheses inferred from source, none verified one by one): the gain of 2-CTA combined with persistent; FA4's register allocation (softmax 176–200, correction 64–80, see `_TUNING_CONFIG`); where the exponential emulation is placed (`ex2_emu_start_frg`); for causal, iterating over blocks in **reverse** order so that masked blocks go first. FA4's causal and D = 64 paths do not use 2-CTA; there the gap is about 3% on `perf_causal`, about 15% on `perf_causal_bh16`, and about 13% on `perf_d64`.
- On `perf_causal` this lab is already within about 3% of FA4. `perf_causal_bh16` is much further behind because its tiles are longer (up to 64 blocks), which magnifies the per-block efficiency difference, while the balance advantage is only a few percentage points.

```bash
./lab profile 43 --case perf_causal_bh16
```

Look at the ratio of `sm__cycles_active.max` to `.avg` (load balance), at `sm__pipe_tensor_cycles_active` and `sm__inst_executed_pipe_xu` (per-block efficiency), and at `lts__t_sector_hit_rate` and `dram__bytes_read` (L2).

## 8. Self-check questions

1. For N = 1000, causal: how many unmasked blocks do tile 0 and tile 1 of pair 1 (rows 256..511) each have? How many blocks does this pair visit in total?
2. How many work tiles does `perf_d64` have? With persistent, how many tiles does each SM do? How many SMs are idle in the last round?
3. Assign 6 tasks (work 10, 9, 8, 3, 2, 1) to 3 machines by round robin and by snake (both over the sorted sequence). What is each completion time? What is the lower bound?
4. Why did step 2 make `perf_causal_bh16` slower? What is the essential difference between hardware CTA dispatch and static round robin?
5. To skip the `X` blocks in the §3.1 table (blocks fully masked for tile 0), how must the loops of the MMA warp, softmax 0 and correction change? Which barriers can no longer share one phase count between the two tiles?
6. If `fin` is double-buffered but there is only one `final_full`, which case fails, and how? Why do `alpha` and `stats_full` not need double buffering?

<details>
<summary>Answers</summary>

1. ⌈1000/128⌉ = 8, 2p + 2 = 4, so it visits 4 blocks (0..3). Tile 0 (qt = 256): min(256/128, ⌊1000/128⌋) = min(2, 7) = 2; tile 1 (qt = 384): min(3, 7) = 3.
2. P = 4096/256 = 16, B·H = 4·32 = 128, T = 2048. 2048 = 13 × 148 + 124: 124 SMs do 14 and 24 SMs do 13; in the last round 24 SMs are idle (2048 / 148 = 13.84 rounds).
3. W = 33, lower bound 11. Round robin: 10 + 3 = 13, 9 + 2 = 11, 8 + 1 = 9, C = 13. Snake: round 2 is reversed, 10 + 1 = 11, 9 + 2 = 11, 8 + 3 = 11, C = 11 (optimal).
4. The hardware gives the next CTA in order to the **first idle** SM (greedy), so SMs that finish light tiles pick up a few more. Static round robin fixes each CTA's tiles in advance: in natural order, in round k CTA b takes pair (k·148 + b) % 32, so some CTAs keep getting heavy tiles, and nothing compensates (§3.4, simulation 1.39 → 1.54, measured max / avg 1.33 → 1.52).
5. Give tile 0 a block count of blocks − 1 (when causal and the last block is tile 0's `X` block): at j = blocks − 2, MMA no longer issues QK⁰_(j+1), and commits `o_full[0]` (and `q_empty[0]`) right after PV⁰_(j); j = blocks − 1 does only tile 1, and K_last is released after QK¹, V_last after PV¹ (tile 1 is the last reader anyway); softmax 0 runs one block fewer; correction handles only t = 1 at j = blocks − 1. The phases of `s_full`, `p_ready`, `stats_full` must be counted separately per tile (nb⁰, nb¹) and can no longer share `nb + j`.
6. When 1-block tiles come one after another and there are more tiles than SMs (`many_one_block`: N = 128, B·H = 2000): for softmax to arrive `final_full` for the next tile it needs only that tile's S_0, which does not go through correction. So `final_full` completes two phases in a row before correction waits on it, and the parity correction waits on has already "flipped over and back": hang (§5.4). The next write of alpha and the next arrive of `stats_full` both first wait on `s_full`, and that QKᵀ comes after the PV that waits for correction, so they are at most one phase ahead.
</details>

## 9. Further reading

- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023), §3.2 (parallelism and causal block skipping).
- Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling* (2026): the section on scheduling and causal.
- R. L. Graham, *Bounds on Multiprocessing Timing Anomalies*, SIAM J. Appl. Math. 17(2), 1969: the 2 − 1/m bound for list scheduling and the 4/3 − 1/(3m) bound for LPT.
- Upstream implementation (`third_party/flash-attention`, commit `1fc0221`):
  - `flash_attn/cute/tile_scheduler.py`: `StaticPersistentTileScheduler` (grid = SM count, `tile_idx += grid_dim`), `SingleTileLPTScheduler` (head sections `l2_minor` with `size_l2 = 50 MB`, LPT reversal `block = num_block − 1 − block`), CLC (`clusterlaunchcontrol`, hardware dynamic scheduling);
  - `flash_attn/cute/flash_fwd_sm100.py`: scheduler selection logic (causal → `SingleTileLPTScheduler`, non-causal → `StaticPersistentTileScheduler`), `_TUNING_CONFIG`;
  - `flash_attn/cute/interface.py`: `is_static_persistent` (false for causal), `use_2cta_instrs` (on for non-causal hdim 128);
  - `flash_attn/cute/block_info.py`: `get_n_block_min_max` (block range for causal), `get_n_block_min_causal_local_mask` (boundary between masked and unmasked blocks).
- CUDA C++ Programming Guide: [Thread Block Clusters and Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (hardware support for dynamic persistent scheduling); [`cudaDevAttrMultiProcessorCount`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) of `cudaDeviceGetAttribute`.
- PTX ISA: [tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit), [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier).
- Lab 32 §5.3–5.4 (persistent GEMM, grouped rasterization): in GEMM, L2 grouping pays off (HBM reads of `wide` drop by 2.5×); compare with §7.2 of this lab and think about why attention is different.
