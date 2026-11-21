# 40 · FA4 pipeline structure: two-Q-tile ping-pong, P written back to TMEM, correction warpgroup

> Part 4 · FA4 — full optimization on Blackwell | Prerequisites: [34_fa3](../34_fa3/README.md), [32_ws_gemm](../32_ws_gemm/README.md) | Code: [`students/40_fa4_pipeline/kernel.cu`](../../students/40_fa4_pipeline/kernel.cu)

## 1. Goals

Lab 34 moved the ideas of FA3 onto `tcgen05`: one TMA warp, one MMA warp, one softmax warpgroup; S and O in TMEM; P through shared memory. It ended at about 0.55× of cuDNN, and the bottleneck was that single softmax warpgroup.

This lab builds the **pipeline structure** of FlashAttention-4 (FA4) on B200. The interface does not change: FP16/BF16 Q, K, V, O, `[B,N,H,D]`, D = 64 or 128, causal and non-causal, outputs O and an FP32 LSE.

| | Lab 34 (FA3 ideas) | This lab (FA4 structure) |
|---|---|---|
| Query rows per CTA | 128 (one Q tile) | **256 (two Q tiles)** |
| softmax warpgroups | 1 | **2, ping-pong**: while one does softmax, the Tensor Core computes the GEMMs of the other tile |
| S buffers in TMEM | 2 S buffers (same tile) | 1 S buffer per tile |
| How P enters P·V | Written to shared memory (SS MMA) | **Written back to TMEM, over the last 64 columns of this tile's S**; the A operand of P·V is read directly from TMEM (**TS MMA**) |
| O correction | Done by the softmax threads themselves, in front of P·V | **A separate correction warpgroup** |
| Registers | Same for every thread | **`setmaxnreg`**: softmax 192, correction 80, everyone else 48 |
| K/V shared memory | One ring each for K and V | K and V share **one** ring (K₀, V₀, K₁, V₁, ...) |

This lab has three steps, and every step supports causal:

1. **Two Q tiles, two softmax warpgroups ping-pong** (P still goes through shared memory, softmax still corrects O itself), plus the causal block count;
2. **P aliased back into TMEM**; P·V becomes a TS-form `tcgen05.mma`;
3. **correction warpgroup and `setmaxnreg`**.

When you finish, you should be able to:

- draw the timeline of the six roles (2 × softmax, correction, MMA, TMA, idle) and the producer, consumer and parity of every mbarrier;
- explain why the single barrier `s_full` guarantees at once that "S is computed", "the P·V of the previous block is done", and "the P of the previous block has been read";
- write the layout of an A operand in TMEM, and how the TS MMA address advances per K slice;
- list how 512 TMEM columns, 64K registers and 227 KB of shared memory are split among the roles;
- use `./lab sass` and an instrumented timeline to explain the gain of each step, and name the bottleneck the next lab (41) attacks: MUFU.

## 2. Motivation: where lab 34 stalls

The conclusion of lab 34 §7.2: after S double buffering, the Tensor Core is still idle about 60% of the time, waiting for the **only** softmax warpgroup. In one block, the softmax warpgroup has to do, in order: read S, find the max, 128×128 `exp2`, write 32 KB of P, and read O, multiply by α and write it back.

The remedies of FA4 (Zadouri et al., 2026; code in `flash_attn/cute/flash_fwd_sm100.py`) match the first three rows of the table in lab 34 §7.2:

| Remaining problem from lab 34 | FA4's solution | Step in this lab |
|---|---|---|
| Nobody fills in while the one softmax warpgroup waits for the MMA, and the MMA is idle while it waits for softmax | Two Q tiles, two softmax warpgroups ping-pong (the inter-warpgroup ping-pong of FA3 paper §3.2, moved onto `tcgen05`) | 1 |
| P is written to shared memory and read back by the MMA, which also needs `fence.proxy.async`; P takes 64 KB of shared memory | P written back to TMEM over S; TS MMA | 2 |
| The O correction is on the critical path of softmax | A separate correction warpgroup | 3 |
| With more roles in one CTA, there are not enough registers to go around | `setmaxnreg` redistributes them by role | 3 |
| MUFU throughput, O corrected every block | Polynomial `exp2`, conditional rescale | Lab 41 |

## 3. Math

The online softmax formulas are exactly the same as in labs 22 and 34 (m stores raw scores, `scale_log2 = scale·log₂e`):

```
m'  = max(m, max_c s_c)
α   = 2^(m·scale_log2 − m'·scale_log2)         block 0: m = −∞, α = 0
p_c = 2^(s_c·scale_log2 − m'·scale_log2)
ℓ   ← α·ℓ + Σ_c p_c
O   ← α·O + P·V_j                              finally O/ℓ, LSE = m·scale + ln ℓ
```

The two tiles are independent: tile 0 is query rows `q0 .. q0+127`, tile 1 is `q0+128 .. q0+255`. They share the same K_j and V_j, and each has its own m, ℓ, S, P, O. The only new math in this lab is **how long one block takes**.

### 3.1 The "chain" of one tile

For one tile, these four things must happen in order (arrows are data dependencies):

```
softmax_j (reads S_j, writes P_j) → P_j·V_j → S_{j+1} = Q·K_{j+1}ᵀ → softmax_{j+1} → …
```

Why must `S_{j+1}` come after `P_j·V_j`? Because each tile has only one S region in TMEM (the column budget of §5.1), and P_j is written into the second half of that S; as soon as `S_{j+1}` is written, P_j is gone. So the period per block of one tile is at least

```
T_tile ≥ SM + PV + QK      (SM = time for the softmax of one block)
```

At D = 128, PV and QK are each 8 MMAs of 128×128×16, each about 64 cycles (B200 does about 8192 FLOP per SM per cycle), about 1024 cycles in total; `exp2` runs at 16 per SM per cycle, and one block of one tile has 128×128 of them, about 1024 cycles.

### 3.2 Two tiles ping-pong

Lab 34 has only one tile: during softmax the Tensor Core is mostly idle (apart from computing the next S early). With two tiles, while tile 0 does softmax, the Tensor Core can do the PV and QK of tile 1, and the other way around:

```
Tensor Core │PV¹ QK¹│PV⁰ QK⁰│PV¹ QK¹│PV⁰ QK⁰│
softmax 0   │ SM⁰_j │       │SM⁰_j+1│       │
softmax 1   │       │ SM¹_j │       │SM¹_j+1│
```

Ideally the two chains are offset by half a period, and `T = max(2·(PV+QK), 2·SM, SM + PV + QK)`. At D = 128 all three terms are about 2048 cycles: the Tensor Core and MUFU are both fully loaded. In practice SM is much longer than 1024 (measured about 1900–2200 in §7.2), so `T ≈ SM + PV + QK + handoff latency`, and the chain length decides everything. This is what every step of this lab shortens.

### 3.3 By hand: how long one block takes per SM

The `perf` case (B·H = 64, N = 4096, D = 128): each CTA has two tiles and 32 K/V blocks; 1024 CTAs on 148 SMs is about 7 rounds. Counting in cycles (independent of the clock frequency):

| | T per block | Total per SM (7 rounds × 32 blocks × T) |
|---|---|---|
| Ideal (§3.2) | 2048 | ≈ 0.46 M cycles |
| Step 3 of this lab (`sm__cycles_elapsed.avg` ≈ 0.83 M, measured with Nsight Compute without locked clocks) | ≈ 3700 (including the prologue and epilogue of each CTA) | ≈ 0.83 M cycles |

At full load the SM clock is about 1.64 GHz (`sm__cycles_elapsed.avg.per_second` from the same measurement, lower than the nominal maximum frequency), so 0.83 M cycles ≈ 0.50 ms (bench measures about 0.55 ms, including the gaps between kernels). §7.2 uses instrumentation to break down one steady-state block (≈ 3450 cycles).

## 4. Hardware and instructions

### 4.1 Writing P back to TMEM: packed 16-bit data with `tcgen05.st`

Softmax thread r owns row r of S (TMEM lane r, lab 34 §4.1). After reading the whole row of S (128 FP32 values, 128 columns), S is no longer needed, so P is written back into **the last 64 columns of the same row**:

```
TMEM columns:  S^t: 128 columns          c = 0 .. 127
               P^t: 64 columns           c = 64 .. 127 (overwrites the second half of S)
32 bits of P^t column c' (c' = 0..63) = { low 16 bits: p[2c'], high 16 bits: p[2c'+1] }
```

A `tcgen05.st` with the 32x32b shape does not care about the data type; it only writes 32-bit cells. The header adds a `uint32_t` version:

```cpp
uint32_t w[16];
for (int c = 0; c < 16; ++c) w[c] = pack2<T>(p[32h + 2c], p[32h + 2c + 1]);
lab::tmem_st_32x32b_x16(p_addr + 16 * h, w);      // h = 0..3
lab::tmem_wait_st();                               // finish all writes before handing off to the MMA
```

Why overwriting the second half of S is safe: before writing P, the same thread has already read the whole row of S into registers (`tcgen05.ld` + `wait::ld`), and only this thread reads this row of S. Why not `x32`: 32 words at once, plus the S row of 128 floats, would spill past 168 registers (the limit in step 2, §4.5).

### 4.2 TS-form `tcgen05.mma`: A read from TMEM

```
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
```

Compared with the SS form of lab 31, the only change is that A is a **TMEM address** instead of a shared memory descriptor (header `lab::mma_f16_ts`). Layout requirements for A in TMEM (PTX ISA "tcgen05.mma", `cta_group::1`, M = 128):

| Requirement | P in this lab |
|---|---|
| Row i of A is in lane i (128 rows fill all 128 lanes); the lane field of the address is 0 | Row r of P^t = lane r, written by softmax thread r |
| The K direction runs along columns, 16-bit data **two per column**, even index in the low 16 bits | `pack2(p[2c], p[2c+1])` |
| Must be K-major: the A-major bit of idesc is 0 | Same as Q and P (lab 34) |
| One MMA has K = 16, i.e. 8 columns | The kk-th K slice starts at `p_tmem + 8·kk` |

B (V, MN-major) does not change at all; it is still the descriptor from lab 34 §4.3. The synchronization for handing off P also becomes simpler:

| | Lab 34 / step 1 of this lab (P in shared memory) | Step 2 (P in TMEM) |
|---|---|---|
| Write P | Ordinary store + `fence.proxy.async` | `tcgen05.st` + `tcgen05.wait::st` |
| Hand off to the MMA | `tcgen05.fence::before_thread_sync` + `mbarrier.arrive` | Same as left |
| 64 KB of P in shared memory | Needed | **Not needed**: the K/V ring grows from 3 to 4 stages (D = 128) |

### 4.3 MMAs from the same thread execute in issue order

In step 2 the MMA warp issues, for the same tile, in order:

```
PV^t_j    : reads  P^t (TMEM columns 128t+64 .. 128t+127)
QK^t_{j+1}: writes S^t (TMEM columns 128t .. 128t+127)      ← overwrites the P^t that PV^t_j still reads
```

There is no barrier between the two MMAs. This relies on: **`tcgen05.mma`s issued by the same thread execute in issue order in the Tensor Core**, so when the later one writes TMEM, the earlier one has already read its operands. FA4 (`mma()` in `flash_fwd_sm100.py`: "this gemm is scheduled after the PV gemm, which guaranteed that Si has been read") and the SM100 FMHA of CUTLASS (`sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`, where P and S share TMEM and the next QK is issued right after PV) both do this.

To be honest about it: the "tcgen05 Memory Consistency Model" in PTX ISA 8.7 lists as guaranteed-in-order pipelined pairs only two `tcgen05.mma`s with "**the same accumulator and the same shape**"; PV (accumulator O) and QK (accumulator S) are not such a pair. This lab, like upstream, relies on the actual behavior of the hardware; all correctness cases of this lab (including large random data) pass, but this is not a guarantee from the specification. The strictly-by-the-spec approach is to wait for the commit of PV to complete before QK, at the cost of draining the Tensor Core twice per block.

This lab relies heavily on another fact that **is guaranteed by the specification**: `tcgen05.commit` makes the mbarrier track **all** `tcgen05` asynchronous operations **previously issued by that thread**. So once the commit after QK^t_{j+1} arrives, PV^t_j has certainly completed too (§5.4).

### 4.4 `setmaxnreg`: moving registers between warpgroups

Each SM has 64K 32-bit registers. When a 512-thread CTA launches, each thread gets at most 128 (`__launch_bounds__(512, 1)`). But a softmax thread keeps a whole row of S (128 floats) in registers, plus addresses, m, ℓ and 8 partial sums; lab 34 measured about 164 for it. MMA and TMA threads need only a few dozen.

```
setmaxnreg.inc.sync.aligned.u32 N;   // raise the limit of every thread in this warp to N, taking from the CTA's pool
setmaxnreg.dec.sync.aligned.u32 N;   // lower it to N, returning the surplus to the pool
```

- N is a multiple of 8, 24 ≤ N ≤ 256;
- all 4 warps of a warpgroup must execute the same `setmaxnreg`, so the MMA warp, the TMA warp and two idle warps together form a complete warpgroup 3 (the block has 16 warps);
- `dec` takes effect immediately; `inc` waits until the pool has enough registers. In this lab the roles that `dec` first (correction, MMA/TMA) do not depend on any `inc`, so there is no deadlock;
- ptxas allocates registers for a code region according to the N after its `setmaxnreg`: in the MMA code with N = 48, if the compiler hoists all 16 descriptors of the two tiles out of the loop, they spill to local memory. The solution writes `#pragma unroll 1` on the tile loop of the MMA. The `STACK` column of `./lab sass` must be 0.

The header adds `lab::setmaxnreg_inc<N>()` / `lab::setmaxnreg_dec<N>()`.

### 4.5 Register and thread budget

| Version | Warps | Role | Register limit per thread | Total |
|---|---|---|---|---|
| Steps 1–2 | 10 (320 threads) | 2 × softmax, MMA, TMA | ptxas gives 168 (the 64K registers of an SM are split into 4 subpartitions of 16K each; warp i is in subpartition i % 4, so with 10 warps some subpartitions hold 3 warps: 16384 / (3·32) ≈ 170, rounded down to a multiple of 8 gives 168) | — |
| Step 3 | 16 (512 threads) | softmax ×2 | 192 | 2 × 128 × 192 = 49152 |
| | | correction | 80 | 128 × 80 = 10240 |
| | | MMA, TMA, 2 idle warps | 48 | 128 × 48 = 6144 |
| | | | | **65536 = 64K** |

In step 3 without `setmaxnreg`, the softmax code can use only 128 registers, and `./lab sass` shows hundreds of bytes of `STACK` (spills).

### 4.6 Running the MMA schedule on the whole warp

In lab 34 only lane 0 of the MMA warp runs (`if (lane == 0)`); the other 31 lanes go straight to the `__syncthreads` at the end of the kernel and wait. This lab switches the MMA warp to the CUTLASS/FA4 style: **all 32 lanes wait on the barrier together, and one lane chosen by `elect.sync` issues the MMA and the commit**:

```cpp
const bool leader = lab::elect_one();
lab::mbar_wait(&bar.p_ready[t], j & 1);       // the whole warp waits
lab::tcgen05_fence_after_thread_sync();
if (leader) { mma...; lab::tcgen05_commit(...); }
__syncwarp();
```

Instrumentation on this machine (§7.2): with only lane 0 running, the MMA thread takes several hundred cycles after `p_ready` has completed to issue the next MMA; with the whole warp, this drops to a few dozen cycles, and the whole kernel is about 5–8% faster. This is a measured effect; we have not verified its hardware cause (how a divergent warp is scheduled). The TMA warp loads several stages ahead, so a slower reaction does not matter there, and it keeps the lane-0 style.

### 4.7 Common misuses

| Misuse | Consequence |
|---|---|
| P written to the first 64 columns of S before the whole row of S is read | Reads back its own freshly written P: wrong results |
| The A address of the TS MMA advances 16 columns per K slice | Only half of P is used; the other half reads leftovers of S: O wrong |
| The two arguments of `pack2` swapped | The P values of two adjacent keys are swapped: `causal_position`, `position` fail |
| arrive after `tcgen05.st` without `wait::st` | The MMA occasionally reads a stale P |
| Only some warps of a warpgroup execute `setmaxnreg` | Undefined behavior (a PTX ISA requirement; this lab did not test what actually happens) |
| `setmaxnreg` totals more than 512 | `inc` never gets its registers: hang |
| The count of `p_ready` is still 128 in step 3 | The arrives of correction and softmax overflow one phase: parity gets scrambled, hang or wrong results |
| Handing the final m and ℓ to correction through `stats_full` as well | Two phases may complete back to back, and the parity correction waits for never comes (§5.6) |

## 5. Schedule design

### 5.1 Roles and storage budget

```
grid  = (⌈N/256⌉, H, B), one CTA = one pair of Q tiles (256 rows) × one head × one batch
block = steps 1–2: 10 warps; step 3: 16 warps
  warp 0–3   softmax WG 0: thread r ↔ row r of tile 0 ↔ TMEM lane r
  warp 4–7   softmax WG 1: thread r ↔ row r of tile 1 ↔ TMEM lane r (warp 4 accesses lanes 0–31)
  warp 8–11  correction WG (step 3): thread r handles row r of O for both tiles
  MMA warp   warp 8 (steps 1–2) or 12 (step 3)
  TMA warp   MMA warp + 1
```

**TMEM** (512 columns × 128 lanes × 32 bits), one S/P and one O for each of the two tiles:

| Columns | D = 128 | D = 64 |
|---|---|---|
| S⁰ (P⁰ in 64–127 of it) | 0–127 | 0–127 |
| S¹ (P¹ in 192–255 of it) | 128–255 | 128–255 |
| O⁰ | 256–383 | 256–319 |
| O¹ | 384–511 | 320–383 |
| Used / allocated | **512 / 512** | 384 / 512 |

At D = 128 it is exactly full. This is why each tile can have only one S region: the "S double buffering" of lab 34 does not fit here, and the overlap comes instead from **the two tiles**.

**shared memory** (all in 128B swizzle `[panel][row][64]`, lab 34 §4.6):

| | Step 1, D = 128 | Steps 2–3, D = 128 | Step 1, D = 64 | Steps 2–3, D = 64 |
|---|---|---|---|---|
| Q⁰, Q¹ (the O staging area at the end) | 64 KB | 64 KB | 32 KB | 32 KB |
| P⁰, P¹ | 64 KB | — | 64 KB | — |
| K/V ring | 3 × 32 KB | 4 × 32 KB | 6 × 16 KB | 8 × 16 KB |
| correction statistics (step 3) | — | 3 KB | — | 3 KB |
| Total | ≈ 224 KB | ≈ 195 KB | ≈ 192 KB | ≈ 163 KB |

### 5.2 One K/V ring

K_j and V_j no longer have a ring each. Instead they form one sequence in the order the MMA warp uses them, **item u**:

```
u = 0    1    2    3    4    5   …
    K₀   V₀   K₁   V₁   K₂   V₂  …       item u goes in stage u % S, the ⌊u/S⌋-th use of that stage
```

For each item, the TMA warp waits on `kv_empty[u%S]` with parity `((u/S)&1)^1` (the standard ring buffer of lab 30), arms `kv_full`, and loads K or V. The MMA warp commits `kv_empty` once after it is done with K_j (after the QKs of both tiles are issued) and once after it is done with V_j (after the PVs of both tiles are issued).

### 5.3 Issue order of the MMA warp

```
Prologue: wait K₀; for t = 0, 1: wait Q^t, QK^t_0 → S^t, commit s_full[t]; commit kv_empty(K₀)
Block j:  wait V_j, K_{j+1}
          for t = 0, 1:
            wait p_ready[t] (parity j&1)
            PV^t_j: O^t [+]= P^t_j · V_j            (no accumulation when j = 0)
            t = 1: commit kv_empty(V_j)
            if there is a next block: QK^t_{j+1} → S^t, commit s_full[t]
            otherwise:                commit o_full[t]
          if there is a next block: commit kv_empty(K_{j+1})
```

The sequence on the Tensor Core is `PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1} PV⁰_{j+1} …`. While the softmax of tile 0 works on S⁰_{j+1}, the Tensor Core does PV¹_j and QK¹_{j+1}: this is the ping-pong. The phase offset between the two tiles forms naturally in the prologue: S⁰_0 completes one QK earlier than S¹_0, and after that each tile's chain loops on its own (§3.2).

### 5.4 One barrier, `s_full`, guarantees three things

At the start of block j, softmax WG t waits only for phase j of `s_full[t]`. When it arrives:

1. S^t_j is in TMEM (the commit follows QK^t_j);
2. PV^t_{j−1} has completed and O^t includes blocks 0..j−1: PV^t_{j−1} was issued by the same thread **before** QK^t_j, and the commit tracks all earlier MMAs (§4.3);
3. P^t_{j−1} has been fully read (same as 2).

So unlike lab 34, softmax **does not need to wait for `o_ready`** before writing P^t_j and correcting O^t.

### 5.5 causal

The last row of a tile pair is `q0 + 255`, so the block count is `min(⌈N/128⌉, ⌈(q0 + 256)/128⌉)`, and both tiles go through **the same number** of blocks (the MMA schedule does not need to distinguish tiles). The cost: the last block of tile 0 is fully masked (rows `q0..q0+127` cannot see keys `q0+128..q0+255`). In a fully masked block, `m' = m` (not −∞, because block 0 always contains key 0), `α = 1`, P is all 0, and the result does not change. FA4 does the same; the schedule of lab 43 deals with this kind of waste.

The blocks that need a mask and the number of visible keys per row are the same as in lab 34, except that `q0` is replaced by the first row of this tile, `qt = q0 + 128t`.

### 5.6 correction warpgroup (step 3)

As soon as a softmax thread has m' and α, it **immediately** hands α to correction, and only then does the 128 `exp2`:

```
softmax WG t (block j)                      correction WG (for each j, for t = 0, 1)
──────────────────────                      ────────────────────────────────────────
wait s_full[t]; ld S; mask; max
α → st.alpha[t][r]; arrive stats_full[t] ─▶ wait stats_full[t] (parity j&1)
exp2, ℓ, write P (TMEM)                     j > 0: O^t row r ×= α (ld/×/st, wait::st)
fence::before; arrive p_ready[t] ──┐        fence::before; arrive p_ready[t] ─┐
                                   └───▶ MMA: wait p_ready[t] (count 256) ◀───┘
(after all blocks) m, ℓ → st.fin_*; arrive final_full[t] ─▶ wait final_full[t], o_full[t]
                                                            O/ℓ → smem → TMA store; LSE
```

correction can read O^t safely: softmax arrives on `stats_full` only after phase j of `s_full[t]`, and at that point PV^t_{j−1} has completed (§5.4). The correction runs in parallel with `exp2` and is no longer on the critical path of softmax.

**Why `stats_full` needs no empty barrier**: before writing α^t_{j+1}, softmax must wait for phase j+1 of `s_full[t]`, which requires the MMA to have waited for phase j of `p_ready[t]`, which in turn requires every correction thread to arrive after reading α^t_j.

**Why the final m and ℓ use another barrier, `final_full`**: after its arrive on `stats_full` in the last block, softmax no longer waits for anything related to correction. If it arrived on `stats_full` again right away, this barrier could complete two phases back to back, the parity would return to its old value, and correction would "never" see the phase it is waiting for.

**Every mbarrier** (S = number of ring stages):

| barrier | count | Who arrives | Who waits | Parity for block j / item u | What it protects |
|---|---|---|---|---|---|
| `q_full[t]` | 1 + tx | TMA | MMA | 0 | Q^t has arrived |
| `kv_full[s]` | 1 + tx | TMA | MMA | `(u/S)&1` | item u has arrived |
| `kv_empty[s]` | 1 | `tcgen05.commit` (K: after both QKs; V: after both PVs) | TMA | `((u/S)&1)^1` | The stage can be overwritten |
| `s_full[t]` | 1 | commit (after QK^t_j) | softmax WG t | `j&1` | The three things of §5.4 |
| `p_ready[t]` | 128 (steps 1–2) / 256 (step 3) | Every thread of softmax WG t (+ every correction thread) | MMA | `j&1` | P^t_j is written, S^t_j has been read, O^t is corrected |
| `o_full[t]` | 1 | commit (after the last PV^t) | epilogue | 0 | Final value of O^t; all MMAs that read Q^t have completed (Q^t can be the staging area) |
| `stats_full[t]` | 128 | softmax WG t | correction | `j&1` | α^t_j is in `st.alpha[t]` |
| `final_full[t]` | 128 | softmax WG t | correction | 0 | Final m, ℓ |

Parity never laps for the same reason as in lab 34: phase j+1 of every barrier requires the waiter to have seen phase j first (for example, phase j+1 of `p_ready` requires softmax to have received phase j+1 of `s_full`, which can only appear after the MMA has waited for phase j of `p_ready`). `final_full` and `o_full` have only one phase.

**racecheck cannot see TMA and TMEM accesses**: the table above is the entire argument for the synchronization correctness of this lab, so check it row by row.

### 5.7 Timeline of all roles (D = 128, steady state, schematic, not to scale)

```
TMA          │K₂ V₂ …… (2 blocks ahead) …………………………………………………………………………………………………………
MMA/TC       │ PV⁰₀ QK⁰₁ │ PV¹₀ QK¹₁ │   ·    │ PV⁰₁ QK⁰₂ │ PV¹₁ QK¹₂ │   ·    │
softmax 0    │           │ld max exp exp exp P│           │ld max exp exp exp P│
softmax 1    │… exp exp P│           │ld max exp exp exp P│           │ld max …
correction   │           │    ×α⁰    │  ×α¹   │           │    ×α⁰    │  ×α¹   │
                         ↑p_ready¹            ↑p_ready⁰   ↑p_ready¹
```

Each tile's period = its own softmax + its own PV and QK + handoff latency; the two tiles are offset by about half a period (§3.2).

## 6. Step-by-step exercises

The skeleton already contains the host side (tensor maps, launch, dispatch), barrier initialization, TMEM allocation, and every helper learned in lab 34 (`mma_qk`, `mma_pv_ss`, `load_s_row`, mask, `row_max`/`exp_and_sum`, `write_p_smem`, `rescale_o_row`, `store_o_and_lse`). `p.step` selects one of three configurations (`Config<D, 1|2|3>`); the three steps **complete the same kernel incrementally**, and later steps depend on earlier TODOs.

### Step 1 · TODO(1a)–(1e): two-tile ping-pong (including causal)

- 1a: the TMA warp loads the two Q tiles and the K/V ring (§5.2);
- 1b: the prologue of the MMA warp (§5.3);
- 1c: per block and per tile, the PV, QK and commits of the MMA warp (§5.3, §4.3);
- 1d: each softmax WG picks its tile and TMEM addresses, and waits on `s_full` (§5.1, §5.4);
- 1e: the causal block count (§5.5).

```bash
./lab check 40 --step 1 --case two_tiles
./lab check 40 --step 1
./lab bench 40 --step 1 --case perf
```

### Step 2 · TODO(2a)–(2b): P written back to TMEM

- 2a: `write_p_tmem` (§4.1); 2b: `mma_pv_ts` (§4.2). The branch of the kernel that calls them (`C::kPInTmem`) is already written.

```bash
./lab check 40 --step 2 --case position
./lab check 40 --step 2
./lab sass 40           # the A operand of UTCHMMA becomes tmem[...]
```

### Step 3 · TODO(3a)–(3d): correction warpgroup and setmaxnreg

- 3a: the three register counts in `Config`, and the `setmaxnreg` at the start of each role (§4.4, §4.5);
- 3b: softmax hands off α and the final m, ℓ; 3c: the main loop of correction; 3d: the epilogue of correction (§5.6).

```bash
./lab check 40 --step 3
./lab sass 40           # step 3 kernel: REG 128, STACK 0
./lab bench 40 --case perf
./lab bench 40 --case perf_causal
./lab bench 40 --case perf_d64
./lab bench 40 --case perf_8k
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `TIMEOUT` (step 1) | Ring parity; MMA and TMA number the items differently (K is even, V is odd); `p_ready` parity; forgot the "continued" parts of TODO(1b)/(1c): releasing K₀ after the prologue and releasing K_{j+1} at the end of each block (these two places have no `LAB_TODO`; if you miss them, the TMA warp never gets an empty stage) |
| Only rows of tile 1 are wrong (cases with N > 128) | Coordinates of Q¹ (`q0 + 128`); TMEM columns of S¹/O¹; `qt` of softmax WG 1 |
| `n1`, `one_block` right, `two_tiles` wrong | Same as above; or tile 1 uses the P or S of tile 0 |
| `ring`, `position` wrong, single-block cases right | K/V item index or stage |
| causal: only tiles from the second pair on are wrong | Block count uses `q0 + 128` instead of `q0 + 256` |
| Step 2: O all wrong, LSE right | Packing order or column offset of P; A address stride of the TS MMA (8 columns) |
| Step 3 `TIMEOUT` | `setmaxnreg` totals > 512; or some warpgroup did not all execute it; `stats_full` parity |
| Step 3 `peaked*` wrong, `uniform` right | correction uses the α of the wrong tile, or multiplies by α also at j = 0 (where α = 0) |
| Step 3 correct but much slower than step 2 | Check STACK in `./lab sass`: register counts not set, softmax spills |

## 7. Performance

### 7.1 Against the library baselines

Measured on GPU 7 (CUDA Graph; `vs cudnn` = cuDNN time / this version's time, higher is better; each entry is the range over 3 runs):

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| Lab 34 step 5 (for comparison) | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× | — |
| Step 1 (ping-pong, P through smem) | ≈ 0.77–0.78× | ≈ 0.88–0.89× | ≈ 0.80× | ≈ 0.75–0.76× |
| Step 2 (P in TMEM, TS MMA) | ≈ 0.80× | ≈ 0.89–0.90× | ≈ 0.73–0.75× | ≈ 0.77–0.78× |
| Step 3 (correction WG + setmaxnreg) | ≈ 0.84× | ≈ 0.93–0.94× | ≈ 0.79–0.81× | ≈ 0.82× |
| FA4 (`flash_attn.cute`) | ≈ 1.15× | ≈ 1.33× | ≈ 1.13× | ≈ 1.10× |

If your step 3 reaches at least about 0.72× of cuDNN on all four perf cases, the mechanisms of this lab work; cuDNN itself varies by a few percent between runs, so look at the range. Reaching 0.9× of cuDNN is the goal of lab 43.

- **ping-pong** (lab 34 → step 1): about 1.35–1.45×, the biggest step of this lab. On `perf`, the fraction of time the Tensor Core is busy (`sm__pipe_tensor_cycles_active`) rises from about 38% in lab 34 to about 53%.
- **P written back to TMEM** (step 1 → 2) and **correction warpgroup** (step 2 → 3): only a few percent each (in step 3 the Tensor Core is about 57% busy on `perf`). The next section explains why: they shorten the softmax chain by a few hundred cycles, but the longest segment of the chain, the 128 `exp2`, does not change.
- On `perf_d64`, step 2 is actually about 7% slower than step 1, and step 3 returns to the level of step 1. At D = 64 the GEMMs are only half of those at D = 128, the Tensor Core is only about 30% busy, and the bottleneck is entirely on the softmax side (MUFU about 57–61%). We have not found the specific reason step 2 slows down (K/V ring depths of 4, 6 and 8 give the same result); this is an effect left for you to study with `./lab profile`.
- Another gain of step 3 is registers: the registers given up by correction and MMA/TMA go to softmax, which later (lab 41) also has to hold the code of the polynomial `exp2`.

### 7.2 Where the bottleneck is: an instrumented timeline

Record the time of each event inside one CTA with `clock64()` (a maintainer experiment; the code is not in the repository; `perf` case, steady-state blocks, in SM cycles):

| Segment | Step 3, MMA on lane 0 only | Step 3 (solution) |
|---|---|---|
| Period T of one block | ≈ 3700 | ≈ 3450 |
| softmax WG: S received → arrive `p_ready` | ≈ 1900 | ≈ 1900–2200 |
| &emsp;of which: 128 `exp2` (including the sum) | ≈ 1450 | ≈ 1450–1700 |
| Last thread arrives → MMA issues PV | ≈ 250–550 | ≈ 40 |
| PV + QK (one tile) on the Tensor Core | ≈ 1100–1300 | ≈ 1100 |

Conclusions:

1. The chain of one tile ≈ softmax + handoff + PV + QK, exactly the model of §3.1;
2. Most of softmax is `exp2`: one warp issues 128 `MUFU.EX2`; at a throughput of 16 per SM per cycle that is at least 1024 cycles, and about 1450 are measured: **MUFU is now the bottleneck**;
3. The Tensor Core works about 57% of the time (`sm__pipe_tensor_cycles_active` in `ncu`), and MUFU (`sm__inst_executed_pipe_xu`) is also only about 57%: neither is saturated; the chain is too long.

The next step in shortening the chain is a faster `exp2`: lab 41 replaces part of the `exp2` with a polynomial on the FMA units (FMA throughput is 8× that of MUFU), and corrects O only when the maximum changes significantly.

```bash
./lab profile 40 --case perf
```

Look at `sm__pipe_tensor_cycles_active`, `sm__inst_executed_pipe_xu`, and in Warp State the `long_sb` stalls of softmax waiting on `s_full` (mbarrier waits). Nsight Compute locks the clocks; use it only to compare metrics.

## 8. Self-check questions

1. How many TMEM columns are used at D = 64? If you wanted to give each tile two S regions at D = 64 (as in lab 34), would there be enough?
2. Warp 6 of softmax WG 1 reads columns 32–63 of S¹. What is the taddr of `tcgen05.ld` (relative to the TMEM base)? What about writing words 16–31 of P¹ (`tmem_st_32x32b_x16`)?
3. In step 2, for the kk = 5 MMA of PV^t_j, what is the TMEM address of A (relative to the base, t = 1)? Which keys does it cover?
4. Why can softmax write P^t_j and correct O^t without waiting for `o_ready` as in lab 34? What goes wrong if the MMA issues QK^t_{j+1} **before** PV^t_j?
5. Why is the count of `p_ready` 256 in step 3? What happens if correction arrives only when j > 0?
6. If you change `setmaxnreg` to softmax 200, correction 64, everyone else 48, what is the total? What happens?

<details>
<summary>Answers</summary>

1. S⁰ and S¹ take 256 columns + O⁰ and O¹ take 64 columns each = 384 columns (512 allocated). One more S region per tile needs another 256 columns, 640 > 512 in total: not enough.
2. Warp 6 is warp 2 of its warpgroup (counting from 0), whose lane range starts at 64; S¹ starts at column 128: `(64 << 16) | (128 + 32)`. P¹ starts at column 64 of S¹, and word 16 is: `(64 << 16) | (128 + 64 + 16)`.
3. `128 + 64 + 8·5 = 232` (lane field 0), covering keys 80–95.
4. §5.4: phase j of `s_full[t]`, which softmax waits for, is completed by the commit after QK^t_j, and PV^t_{j−1} was issued by the same thread before QK^t_j, so the commit tracks it. If QK^t_{j+1} were issued first instead, it would overwrite the P^t_j that PV^t_j has not read yet (P lives in the columns of S): wrong results.
5. Every softmax thread (128) and every correction thread (128) arrives once per block; the MMA must confirm both "P is written" and "O is corrected". If correction did not arrive at j = 0, phase 0 would never collect 256 arrivals: hang.
6. 2·200 + 64 + 48 = 512, which is legal. Lowering correction to 64 may cause spills (check `./lab sass`), and 8 more registers for softmax do not help in this lab. Above 512, `inc` never gets its registers and hangs.
</details>

## 9. Further reading

- Zadouri et al., [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) (2026); upstream implementation `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`:
  - `__init__`: warp roles (`softmax0_warp_ids` … `empty_warp_ids`), `tmem_s_offset` / `tmem_o_offset` / `tmem_s_to_p_offset = n_block_size // 2` (the table in §5.1 of this lab), register budgets such as `num_regs_softmax`;
  - `mma()`: the issue order of the two Q stages (§5.3 of this lab) and the comment "S is written only after PV";
  - `softmax_step()`: read S, max, hand `acc_scale` to correction, write P back to TMEM;
  - `correction_loop()`: `sm_stats_barrier`, rescale, epilogue (§5.6 of this lab; FA4 also has conditional rescale: lab 41).
- Jay Shah et al., *FlashAttention-3* (2024) §3.2: inter-warpgroup ping-pong scheduling.
- `third_party/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`: `TmemAllocation` (S0, S1, P0, P1, O0, O1), the alternation of QK/PV in `mma()`, `softmax()` and `correction()`; `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`: `NumRegsSoftmax` and related.
- PTX ISA 8.7: [tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma) (the form with A in TMEM), [tcgen05.st](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st), [tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model) (pipelined instruction pairs), [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg), [elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync).
- Next: lab 41 (softmax throughput: polynomial `exp2`, conditional rescale).
