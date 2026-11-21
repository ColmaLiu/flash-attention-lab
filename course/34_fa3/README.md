# 34 · FA3 ideas on SM100: warp roles, S/O in TMEM, and softmax–MMA overlap

> Part 3 · FA3 ideas on SM100 | Prerequisites: [22_fa2](../22_fa2/README.md), [30_tma_mbarrier](../30_tma_mbarrier/README.md), [31_tcgen05_gemm](../31_tcgen05_gemm/README.md) (recommended first: [32_ws_gemm](../32_ws_gemm/README.md), warp specialization) | Code: [`students/34_fa3/kernel.cu`](../../students/34_fa3/kernel.cu)

## 1. Goals

Move the FA2 of lab 22 onto `tcgen05` and apply the ideas of FlashAttention-3. The interface does not change: FP16/BF16 Q, K, V, O, `[B,N,H,D]`, D = 64 or 128, causal and non-causal, outputs O and an FP32 LSE.

Compared with lab 22, every change is about "who does what, and when":

| | Lab 22 (FA2, `mma.sync`) | This lab (FA3 ideas, `tcgen05`) |
|---|---|---|
| Data movement | All threads `cp.async`, then `__syncthreads` | **One TMA warp** feeds data through an mbarrier ring buffer |
| Matrix multiply | Each warp issues its own `mma.sync` | **One thread of one MMA warp** issues every `tcgen05.mma` |
| Where S lives | Registers | **TMEM**; the softmax warps read it with `tcgen05.ld` |
| How P enters P·V | Stays in registers as the A operand | **Written to shared memory** (SS MMA) |
| How V is read | `ldmatrix.trans` | **MN-major descriptor** + the transpose bit of the instruction descriptor |
| Where O lives | Registers | **TMEM**; for every block (except block 0) the softmax warps read O, multiply by α, and write it back, whether or not the maximum changed (conditional rescale is in lab 41) |
| Overlap | None (one warp runs everything in order) | **S double buffering**: MMA computes S_{j+1} while softmax processes S_j |

This lab has five steps:

1. **Roles and S = Q·Kᵀ**: the TMA warp, the MMA warp and the softmax warpgroup take their places; S is written to TMEM, and the softmax warps read it and write it to the probe (single block, N ≤ 128);
2. **Single-block attention**: P goes through shared memory, O += P·V (V is MN-major), normalize, TMA-store O, write LSE;
3. **Full K/V loop**: online softmax; the softmax warps correct O inside TMEM; the K/V ring buffer wraps around;
4. **causal**: skip blocks, mask only the diagonal block and the tail block;
5. **S double buffering**: QKᵀ_{j+1} overlaps softmax_j.

When you finish, you should be able to:

- name the three ideas of FA3, and say what each one becomes when moving from Hopper (`wgmma`, register accumulators) to SM100 (`tcgen05`, TMEM);
- draw the producer, consumer and parity of every mbarrier in this lab, and explain the role of each `tcgen05.fence` and `fence.proxy.async`;
- write the descriptor of an MN-major operand;
- list how the 512 TMEM columns are allocated;
- explain why softmax becomes the bottleneck after overlapping, and how FA4 (Part 4) continues from there.

## 2. Motivation: where FA2 stalls on B200

The conclusion of lab 22 §7.2: FA2 reaches only about 0.26× of cuDNN, because `mma.sync` itself has an upper bound of only about 0.43× of cuDNN. In lab 31, switching to `tcgen05` raised GEMM from 0.28× to about 0.8× of cuBLAS. Is it enough to replace the two matrix multiplies of attention with `tcgen05`?

No. Each block of attention is **GEMM → softmax → GEMM**. The softmax in the middle depends on the result of the first GEMM, and the second GEMM depends on the result of the softmax. Done in order, the Tensor Cores sit idle during softmax, and the softmax units sit idle during the GEMMs. §3.3 computes that on B200, at D = 128, the softmax of one block takes **about as long** as the two GEMMs. Done serially, you can use at most half of the Tensor Core.

FlashAttention-3 (Shah et al., 2024) proposes three ideas for Hopper:

1. **Producer–consumer asynchrony** (paper §3.1):
   - *warp specialization*: some warps only move data with TMA (producers), the other warps only compute (consumers). They hand off through ring buffers and barriers in shared memory, instead of "all threads load together, then compute together";
   - *ping-pong scheduling*: two consumer warpgroups each compute one Q tile, ordered by barriers, so that the softmax of one warpgroup runs at the same time as the GEMM of the other (overlap **between** warpgroups).
2. **Overlapping softmax with GEMM** (paper §3.2): *a two-stage pipeline inside a warpgroup*: inside one warpgroup, first issue the QKᵀ of the next block (asynchronous `wgmma`), then do the softmax of the current block, and finally wait for the QKᵀ to finish (overlap **within** a warpgroup).
3. **Low precision** (paper §3.3): FP8 QKᵀ and PV, with block quantization and incoherent processing. This course does FP8 in lab 44; this lab does not cover it.

How they map onto SM100:

| FA3 on Hopper | This lab on SM100 | Deferred to |
|---|---|---|
| producer warpgroup (TMA) + consumer warpgroups, `setmaxnreg` redistributes registers | TMA warp + MMA warp + softmax warpgroup, one role each | Register redistribution: lab 40 |
| The `wgmma` accumulator is in **registers**; S and O both take consumer registers | The `tcgen05.mma` accumulator is in **TMEM**; neither S nor O takes registers | — |
| The warpgroup that issues `wgmma` is the one that does softmax; "issue the next QKᵀ, then do softmax" happens in the same thread | The MMA is issued by **another warp**; overlap comes from **two S buffers** in TMEM | — |
| P is in registers, used as the A operand of `wgmma` (RS) | P is written to **shared memory** (SS MMA) | P aliased back into TMEM (TS MMA): lab 40 |
| Rescaling O: the consumer multiplies its registers directly | softmax warps `tcgen05.ld` → multiply by α → `tcgen05.st`, every block, synchronously | Separate correction warpgroup, conditional rescale: labs 40–41 |
| Two warpgroups ping-pong | One softmax warpgroup, one Q tile | Ping-pong over 2 Q tiles: lab 40 |
| FP8 | Not done | Lab 44 |

This lab deliberately stops at "one softmax warpgroup + S double buffering". That way you can measure clearly how much the overlap gains and where the remaining bottleneck is (§7); Part 4 then removes the bottlenecks one by one.

## 3. Math

### 3.1 online softmax: same as lab 22

One CTA handles 128 query rows. When processing the j-th K/V block (128 keys), for each row:

```
m'  = max(m, max_c s_c)                        (s is the raw score, not multiplied by scale)
α   = 2^((m − m')·scale_log2)                  scale_log2 = scale·log₂e
p_c = 2^(s_c·scale_log2 − m'·scale_log2)
ℓ   ← α·ℓ + Σ_c p_c
O   ← α·O + P·V_j
m   ← m'
```

At the end, `O ← O/ℓ` and `LSE = m·scale + ln ℓ`. The derivation, the worked example, the reason "m is not multiplied by scale" (scale > 0), and the handling of fully masked rows (use 0 in place of m'·scale_log2) are all exactly as in lab 22 §3.

The only difference is **where the data lives**: S and O are in TMEM; m, ℓ and the current block's P are in the registers of the softmax threads. In this lab one thread owns **a whole row** (one TMEM lane), so the row max and row sum need no shuffle: the quad reduction of lab 22 disappears here.

### 3.2 Why the O correction must come before P·V

In `O ← α·O + P_j·V_j`, the MMA accumulates `P_j·V_j` directly into O in TMEM (`enable_input_d = 1`). So "multiply by α" must finish before the MMA reads O, and O can be read only after the previous `P_{j−1}·V_{j−1}` has finished writing it:

```
PV_{j−1} done  →  softmax threads: ld O, O *= α_j, st O  →  PV_j starts
(o_ready)                                                   (p_ready)
```

Block 0 has no old state: `PV_0` uses `enable_input_d = 0` to overwrite O (TMEM holds garbage after allocation), and needs no correction.

### 3.3 How many cycles one block takes: why overlap is required

B200 does about 8192 FLOP per SM per cycle (BF16 dense, lab 31 §3.2). At D = 128, for one block of one CTA:

| Work | Amount | Approx. cycles |
|---|---|---|
| S = Q·Kᵀ: 128×128×128 | 8 MMAs of 128×128×16 | ≈ 512 |
| O += P·V: 128×128×128 | 8 MMAs of 128×128×16 | ≈ 512 |
| Exponentials: 128×128 `MUFU.EX2` | 16 per SM per cycle | ≈ 1024 |
| Rest of softmax (FFMA, FMNMX, FADD, packing, writing P, correcting O) | about 600 instructions per thread | ≈ 600 (partly issued in parallel with MUFU) |

At D = 64 each of the two GEMMs takes about 256 cycles; the exponentials still take 1024 cycles.

Conclusions:
- Serial (GEMM → softmax → GEMM): about 1024 + 1000+ cycles per block; the Tensor Core is busy at most half the time;
- Perfect overlap: about max(1024, softmax) cycles per block; softmax is as long as the GEMMs or longer, so **after overlapping, softmax becomes the bottleneck**.

This is where FA4's work comes from: two softmax warpgroups (lab 40), and emulating part of `exp2` with FMA (lab 41).

## 4. Hardware and instructions

The mbarrier, TMA, `tcgen05.mma/commit/ld` and TMEM allocation used in this lab all come from labs 30 and 31 (headers [`mbarrier.cuh`](../../include/lab/mbarrier.cuh), [`tma.cuh`](../../include/lab/tma.cuh), [`tcgen05.cuh`](../../include/lab/tcgen05.cuh)). There are five new points.

### 4.1 S in TMEM: one thread per row

S is a 128×128 FP32 accumulator and takes 128 columns (column = key, lane = query row). The softmax warpgroup is warps 0–3: warp w can access only lanes `32w .. 32w+31` (lab 31 §4.2), so thread `r = threadIdx.x` (0..127) owns exactly row r:

```
TMEM lane r  ←→  query row q0 + r  ←→  softmax thread r
tcgen05.ld.32x32b.x32 (lane 32w, col c): thread r gets S[r][c .. c+31]
one row of 128 columns = 4 x32 loads, into float s[128]
```

**No computation between `tcgen05.ld` and `wait::ld`.** `tcgen05.ld` is asynchronous: the registers are valid only after `tcgen05.wait::ld`. In lab 31, `tmem_ld_32x32b_x32` + `tmem_wait_ld()` are two separate asm statements. The `wait` has no register operands, so the compiler does not know that `v[]` depends on it, and it may schedule "pure register arithmetic on v[]" (taking the max, multiplying by α) before the wait, reading registers that are not written yet. The epilogue of lab 31 stores to memory right after the load, which happens to be safe; this lab computes right after the load, so it always uses **`lab::tmem_ld_32x32b_x32_sync`** from the header: ld and wait are in the same asm statement, so the output registers "exist" only after the wait. The cost is that the 4 loads of one thread cannot be in flight at the same time (measured on this machine: about 1–3% difference, see NOTES).

### 4.2 S = Q·Kᵀ: the same as the GEMM of lab 31

Q (`[128][D]`) and K_j (`[128][D]`) are both contiguous in D, i.e. the K-major A and B of lab 31. At D = 128 one row is 256 bytes, so TMA loads it as two **panels** with one 64-column box each: `[D/64][128 rows][64]`; each panel has the 128B swizzle layout of lab 31. The kk-th K slice (16 values of d) is in panel `16kk/64`, at the panel start address plus `2·(16kk % 64)` bytes:

```cpp
idesc = lab::make_idesc_f16(128, 128, is_bf16);
for kk in 0 .. D/16−1:
  mma(S, desc_K(Q + panel·128·64 + col), desc_K(K + panel·128·64 + col), idesc, kk > 0)
```

### 4.3 O += P·V: V is an MN-major B operand

`tcgen05.mma` computes `D = A·Bᵀ`, with B given as "N × K". In P·V:

| | M | N | K |
|---|---|---|---|
| Meaning | query row (128) | d (D) | key (128) |
| A = P | row | — | key contiguous → **K-major** |
| B = V | — | d | V is stored as `[key][D]`, **d (N) contiguous → MN-major** |

Lab 22 solved the same problem with `ldmatrix.trans` (lab 22 §4.4). `tcgen05` lets the hardware read MN-major operands directly. You only need:

1. **Set bit 16 (B major) of the instruction descriptor to 1**: `lab::make_idesc_f16(128, D, is_bf16, true, false, /*b_mn_major=*/true)`. CUTLASS calls it `b_major_ = MN`; it is often called the "transpose bit";
2. **An MN-major shared memory descriptor**, `lab::make_smem_desc_mnmajor_sw128(smem, lbo_bytes)` from the header. The bit fields are the same as K-major (lab 31 §4.4), but LBO and SBO mean different things. The canonical 128B swizzle layout given by CUTLASS (`mma_traits_sm100.hpp`, `make_umma_desc<Major::MN>`), in units of elements:

```
MN-major, SWIZZLE_128B:  ((8, 8, m), (8, k)) : ((1, 8, LBO), (64, SBO))
                          └── MN ─┘  └ K ─┘

   MN (d) direction: 64 elements (128 bytes) contiguous = one swizzle row; the second 64-column panel is at +LBO bytes
   K (key) direction: 8 key rows (128 bytes each) = one 1024-byte atom; the next group of 8 rows is at +SBO bytes
```

TMA loads V_j as `[D/64 panels][128 key rows][64 d]`, so:

| Field | Value | Reason |
|---|---|---|
| SBO | 1024 bytes | 8 key rows form one atom; atoms are laid out contiguously |
| LBO | 128 × 128 = 16384 bytes | The size of one d panel (at D = 64 there is only one panel; unused) |
| kk-th K slice (16 keys) | start address + kk·16 rows·128 bytes | Crosses two atoms |

Compare with K-major (lab 31): in K-major the K slice moves **within a row** (+32 bytes), and SBO is the distance between groups of 8 rows in the M/N direction; in MN-major the K slice moves **across rows** (+2048 bytes), and SBO is the distance between groups of 8 rows in the K direction. The same V tile, with no transposing copy at all.

### 4.4 P through shared memory: writing the TMA layout by hand

The A operand of P·V is read from shared memory, so the softmax threads must write P (converted to FP16/BF16) in **the canonical layout the MMA understands**: exactly the same K-major 128B swizzle that TMA writes for Q, `[2 panels][128 rows][64 keys]`. Thread r writes row r, and the q-th 16-byte chunk (8 keys) goes to `q ^ (r % 8)`:

```cpp
*(uint4*)(sP + panel·128·64 + r·64 + (q ^ (r % 8))·8) = pack(p[8q .. 8q+7]);
```

A 16-byte store is processed in phases (wavefronts) of 8 threads: threads r..r+7 write the same logical chunk q of their own rows; the physical positions `q ^ (r%8)` are all different and cover all 32 banks, with no conflict.

After writing, each thread executes **`fence.proxy.async`** (lab 30 §4.7): ordinary stores go through the generic proxy, while `tcgen05.mma` reads shared memory through the async proxy. Without this fence, the MMA may read a stale P.

### 4.5 `tcgen05.st`: writing the corrected O back to TMEM

```
tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], {r0, ..., r31};
tcgen05.wait::st.sync.aligned;
```

The mirror of `tcgen05.ld`: thread i writes 32 registers to the 32 columns of lane `taddr.lane + i`. It is also asynchronous and counts as complete only after `wait::st`. The header provides `lab::tmem_st_32x32b_x32` and `lab::tmem_wait_st`. Correcting one row of O:

```cpp
for c in 0, 32, ..., D−32:
  tmem_ld_32x32b_x32_sync(O + c, o);  o[i] *= α;  tmem_st_32x32b_x32(O + c, o);
tmem_wait_st();
```

### 4.6 A 4D tensor map over [B,N,H,D]

Q, K, V, O are `[B,N,H,D]`: two adjacent rows of the same head are `H·D` elements apart, and different batches are `N·H·D` apart. Describe it as a 4D tensor (fastest dimension first):

| Dim | Size | Byte stride | box |
|---|---|---|---|
| 0: d | D | 2 | 64 |
| 1: head | H | D·2 | 1 |
| 2: row | N | H·D·2 | 128 (32 for O) |
| 3: batch | B | N·H·D·2 | 1 |

One box is 128 rows × 64 columns of one (batch, head), and TMA writes it as `[128][64]` with `SWIZZLE_128B`. The coordinate order is `(x = d, y = head, z = row, w = batch)`.

Why not merge B and N into a single dimension `B·N`? Then rows past N would read rows of **the next batch** (in the tail blocks of K/V), and storing O would even **overwrite** the next batch's output. With 4 dimensions, rows past N are out of bounds in dimension 2: loads fill zeros, stores skip them. This is why the tail needs no checks at all:

- Out-of-bounds Q rows are 0: the results of these rows are never written back;
- Out-of-bounds K rows are 0: S = 0, and then it is masked to −∞;
- Out-of-bounds V rows are 0: P = 0 times 0, O is not affected (stale data could be NaN, and `0·NaN = NaN`);
- O store: out-of-bounds rows are not written.

The header provides `lab::make_tensor_map_4d` (host; arguments in `[B, N, H, D]` order from outer to inner: `d3 = B, d2 = N, d1 = H, d0 = D`) and `lab::tma_load_4d` / `lab::tma_store_4d` (device). They differ from the 2d/3d versions of lab 30 only by one coordinate.

### 4.7 Two kinds of fence for handing off TMEM

| Case | Sender | Sync | Receiver |
|---|---|---|---|
| MMA finished writing S; softmax reads | `tcgen05.commit` (implicit) | `s_full` | `tcgen05.fence::after_thread_sync`, then `tcgen05.ld` |
| softmax finished reading S and correcting O; MMA writes again | `tcgen05.wait::ld/st`, `tcgen05.fence::before_thread_sync` | `mbarrier.arrive(p_ready)` → MMA warp `wait` | `tcgen05.fence::after_thread_sync`, then `tcgen05.mma` |
| softmax finished writing P (shared memory); MMA reads | `fence.proxy.async` | Same as above (`p_ready`) | — |

`p_ready` hands off three things at once: P is written, this thread's read of S is complete (the S buffer can be overwritten), and this thread's row of O is corrected.

### 4.8 Common misuses

| Misuse | Consequence |
|---|---|
| Using a K-major B descriptor for V, or forgetting bit 16 of idesc | O is completely wrong; `one_position` catches it immediately |
| Swapping LBO/SBO in the MN-major descriptor | Wrong for every D: one 16-key K slice spans two 8-row atoms, so SBO is used even at D = 64; at D = 128 the second d panel also uses LBO |
| Wrong row number in the P swizzle | `one_position`, `one_ones_v` fail |
| No `fence.proxy.async` after writing P | Intermittent errors, mostly on large cases |
| softmax writes P before `o_ready` | P_{j−1}·V_{j−1} reads a P that is half overwritten: intermittent errors on multi-block cases |
| arrive after correcting O without `wait::st` | MMA may read O before the st: intermittent errors |
| `tcgen05.ld` and `wait::ld` written separately, with computation between them | Intermittent errors after compiler scheduling (§4.1) |
| A tensor map that merges B·N into one dimension | The tail reads the next batch; O corrupts the next batch |
| expect_tx counts only valid rows | Hang (the zero-filled out-of-bounds part also counts bytes) |

## 5. Schedule design

### 5.1 Roles and storage budget

```
grid  = (⌈N/128⌉, H, B), one CTA per (Q tile, head, batch)
block = 6 warps = 192 threads
  warp 0–3  softmax warpgroup: thread r ↔ row q0+r ↔ TMEM lane r
  warp 4    TMA warp (lane 0)
  warp 5    MMA warp (lane 0)
```

shared memory (1024-byte aligned, all in 128B swizzle `[panel][row][64]`):

| | D = 128 | D = 64 |
|---|---|---|
| Q (reused as the O staging area at the end) | 32 KB | 16 KB |
| P (128×128, 16-bit) | 32 KB | 32 KB |
| K stage × number of stages | 32 KB × 2 | 16 KB × 3 |
| V stage × number of stages | 32 KB × 2 | 16 KB × 3 |
| Total | 192 KB | 144 KB |

Both values of D give one CTA per SM (228 KB cannot hold two). This is intentional: step 5 uses all 512 TMEM columns, so a second CTA would wait idle at `tcgen05.alloc` while holding shared memory.

**TMEM budget** (512 columns per SM, each column 128 lanes × 32 bits):

| | Steps 1–4 | Step 5 |
|---|---|---|
| S buffer 0 | columns 0–127 | columns 0–127 |
| S buffer 1 | — | columns 128–255 |
| O (FP32, D columns) | from column 128 | from column 256 |
| Actually used (D = 128 / 64) | 256 / 192 | 384 / 320 |
| Allocated (power of 2) | 256 | 512 |

FA4 also has to fit the S and O of a second Q tile (lab 40), so it writes P back into the TMEM columns of S and drops P from shared memory.

### 5.2 Handoffs for one block (step 2: single block)

```
TMA warp          MMA warp                      softmax warpgroup (128 threads)
─────────         ─────────                     ───────────────────────────────
Q → q_full ─────▶ wait q_full
K_0 → k_full ───▶ wait k_full; QKᵀ → S
                  commit k_empty, s_full ─────▶ wait s_full; ld S; mask; max; exp; ℓ
V_0 → v_full                                    write P (swizzle); fence.proxy.async
                                                fence::before; arrive p_ready
                  wait p_ready, v_full ◀──────┘
                  P·V → O
                  commit v_empty, o_ready ────▶ wait o_ready; ld O; ×1/ℓ; write staging
                                                fence.proxy.async; TMA store O; write LSE
```

**epilogue**: `o_ready` tracks all earlier `tcgen05` operations of the MMA thread, so when it arrives, QKᵀ has long finished reading Q. The 32 KB of Q becomes the staging area for O: each thread writes its own row of O/ℓ into it with the swizzle, and each warp uses one TMA store (two at D = 128) to write back its own 32 rows (box `[32][64]`); TMA skips rows past N. LSE is one float per thread; adjacent threads write adjacent addresses, so the access is naturally coalesced.

### 5.3 The full loop (step 3)

```
MMA warp:                                softmax thread (block j):
  issue_qk(0)                              wait s_full; ld S_j; mask; m', α, P, ℓ
  for j:                                   if j > 0: wait o_ready (PV_{j−1} done)
    wait p_ready (j-th time)               write P_j; fence.proxy.async
    wait v_full; PV_j → O                  if j > 0: O *= α (ld/st, wait::st)
    commit v_empty, o_ready                fence::before; arrive p_ready
    next issue_qk (which block? issue only if the block exists)
```

The next QKᵀ goes **after** PV_j: there is only one S buffer for it to write, and softmax must finish reading S_j first (guaranteed by `p_ready`). As a result, when softmax processes the next block, its S has to wait for both PV_j and this QKᵀ to finish: the Tensor Core and softmax take turns, with **no overlap**.

Softmax waits for `o_ready` only **after** computing the exponentials: during the exponentials PV_{j−1} can still be running. In step 3 this bit of overlap is almost useless (S_j is computed only after PV_{j−1} anyway); in step 5 it matters a lot.

### 5.4 causal (step 4)

BM = BN = 128; the rows of Q tile i are `128i .. 128i+127`:

- Number of blocks: `min(⌈N/128⌉, ⌈(q0 + 128)/128⌉)`, i.e. blocks 0..i; later blocks are neither loaded nor computed;
- Blocks that need a mask: blocks that go past N, and blocks whose last key satisfies `k0 + 127 > q0`. With BM = BN, that is only the diagonal block (block i);
- Row `q0 + r` sees `min(N, q0 + r + 1) − k0` keys; the rest are set to −∞.

### 5.5 S double buffering (step 5): overlap

Put S into two TMEM buffers (which buffer block j uses is decided by `s_buffer(j)`, which you implement in step 5), and the MMA warp can compute S **one block ahead**:

```
MMA warp:
  issue_qk(0)
  issue_qk(?)                                 ← new in step 5: starts while softmax is still on S_0
  for j:
    wait p_ready(j)                            S_j has been read, P_j is written, O is corrected
    PV_j; commit v_empty, o_ready
    issue_qk(?) (if the block exists)         ← reuses the buffer S_j just freed
```

Timeline sketch (D = 128; `QK0` = QKᵀ_0, `SM2` = softmax of block 2; per §3.3, each QK and PV takes about 512 cycles and one softmax takes 1000+ cycles; cell widths show only ordering, not scale):

```
Tensor Core  │QK0│QK· │  ··  │PV0│QK· │  ··  │PV1│QK· │  ··  │PV2│QK· │ ··
softmax WG   │   │SM0 ·······│ SM1 ··········│ SM2 ··········│ SM3
                 └─p_ready(0)┴─────p_ready(1)┴─────p_ready(2)┘
                    QK·: in step 5 you decide which block. When SM1 starts, S_1 is long done; before writing P, SM1 waits for o_ready(0) (PV0 is done)
```

Why PV_j goes **before** the QKᵀ in the loop: when softmax processes block j+1, it must wait for PV_j (`o_ready`) before writing P_{j+1}. If that QKᵀ were issued first, PV_j would queue behind it, and softmax would wait about 512 cycles for nothing.

**Every mbarrier** (S = number of K/V stages, `kSBuf` = 1 or 2):

| barrier | count | Who arrives | Who waits | Parity used for block j | What it protects |
|---|---|---|---|---|---|
| `q_full` | 1 + tx | TMA warp (expect_tx) + TMA | MMA | 0 | Q has arrived |
| `k_full[s]` | 1 + tx | TMA warp + TMA | MMA (`issue_qk`) | `(j/S)&1` | K_j has arrived in stage s |
| `v_full[s]` | 1 + tx | TMA warp + TMA | MMA (before PV) | `(j/S)&1` | V_j has arrived |
| `k_empty[s]` | 1 | `tcgen05.commit` (after QKᵀ_j) | TMA warp (before loading K_{j+S}) | `((j/S)&1)^1` (when loading block j) | The K in stage s has been fully read |
| `v_empty[s]` | 1 | `tcgen05.commit` (after PV_j) | TMA warp | Same as above | The V in stage s has been fully read |
| `s_full[b]` | 1 | `tcgen05.commit` (after QKᵀ_j) | 128 softmax threads | `s_phase(j)`: `j&1` with one buffer; derived in step 5 for two buffers | S_j is in buffer `s_buffer(j)` |
| `p_ready` | 128 | Each softmax thread | MMA | `j&1` | P_j is written, S_j has been read, O is corrected |
| `o_ready` | 1 | `tcgen05.commit` (after PV_j) | 128 softmax threads | `j&1` | O includes block j; P can be overwritten |

Why parity never "laps": if a barrier completes phase j+1 before the waiter has seen phase j, the parity is back to its old value and the waiter never sees its phase. Check each one: phase j+1 of `p_ready` requires softmax to have waited on `o_ready(j)`, which can only happen after MMA waited on `p_ready(j)`; phase j+1 of `o_ready` requires `p_ready(j+1)`, which comes after softmax waited on `o_ready(j)`; the next commit of `s_full[b]` comes after `p_ready`; empty/full is the standard ring buffer of lab 30.

**Draining before exit**: after the loop, the TMA warp waits once more on the last `empty` of every stage. Otherwise the CTA may exit while the arrive of some `tcgen05.commit` has not yet landed in shared memory, and the next CTA initializes its barriers in the same shared memory.

## 6. Step-by-step exercises

The skeleton already contains the launch, the host-side tensor maps, barrier initialization, TMEM allocation/deallocation, writing the probe, and the tail part of the mask. New header functions used in this lab: `lab::make_tensor_map_4d`, `lab::tma_load_4d/tma_store_4d`, `lab::make_smem_desc_mnmajor_sw128`, `lab::tmem_ld_32x32b_x32_sync`, `lab::tmem_st_32x32b_x32/tmem_wait_st`. The comment before each `TODO` states its inputs, outputs and available helpers.

### Step 1 · TODO(1a)–(1d): roles and S = Q·Kᵀ

- 1a: the TMA warp loads Q, and K_j inside the loop (wait `k_empty`, arm `k_full`, `tma_load_4d` panel by panel);
- 1b: `mma_qk` (§4.2);
- 1c: `load_s_row` (§4.1);
- 1d: `issue_qk` of the MMA warp.

The softmax threads write `scale·S` into the probe (`[B,H,N,N]`, already written), which is compared against the `fa3_scores` oracle. Steps 1 and 2 handle only one block (N ≤ 128).

```bash
./lab check 34 --step 1
./lab sass 34          # UTMALDG, UTCHMMA, LDTM
```

The report names the dimensions `[b, h, i, j]`: if `i` is wrong, check Q (coordinates, panel); if `j` is wrong, check K; if only the D = 128 cases fail, check the offset of the second panel.

### Step 2 · TODO(2a)–(2h): single-block attention

- 2a: `mma_pv` (§4.3; use `lab::make_smem_desc_mnmajor_sw128` for the descriptor);
- 2b: `softmax_block` (max, exp, row sum); 2c: `write_p_row` (§4.4);
- 2d: `store_o_and_lse` (§5.2 epilogue); 2e: the TMA warp loads V_j;
- 2f: PV in the MMA warp; 2g: softmax hands off P (§4.7); 2h: wait for the last PV before the epilogue.

```bash
./lab check 34 --step 2 --case one_position
./lab check 34 --step 2
```

### Step 3 · TODO(3a)–(3d): full loop and O correction

- 3a: `rescale_o_row` (§4.5); 3b: the MMA warp issues the next QKᵀ (§5.3: which block, under what condition, and why after PV);
- 3c: α; 3d: softmax waits for `o_ready` before writing P and correcting O.

```bash
./lab check 34 --step 3 --case ring
./lab check 34 --step 3
./lab bench 34 --step 3 --case perf
```

### Step 4 · TODO(4a)–(4c): causal

```bash
./lab check 34 --step 4
./lab bench 34 --step 4 --case perf_causal
```

### Step 5 · TODO(5a)–(5d): S double buffering

In steps 1–4, both the MMA warp and the softmax warps find S_i through two functions: `s_buffer<kSBuf>(i)` (which TMEM buffer S_i is in) and `s_phase<kSBuf>(i)` (which parity of `s_full[buffer]` to wait for). The one-buffer branch is given. In this step you implement the two-buffer protocol (§5.5):

- 5a: the MMA warp issues QKᵀ_1 early, before the loop;
- 5b: after PV_j, which block to issue into the buffer that was just freed;
- 5c: the two-buffer branch of `s_buffer`;
- 5d: the two-buffer branch of `s_phase`. How to derive it: `s_full[b]` completes one phase per commit it receives (one per block that uses buffer b). If block i is the k-th block (counting from 0) to use its buffer, its S completes the k-th phase, and waiting for it means waiting for parity `k & 1`.

Both sides use the same pair of functions, so the buffer the MMA writes and the barrier it commits automatically match the buffer softmax reads and the parity it waits for. When this is wrong, the most common symptom is a hang (waiting for a phase that never completes), followed by reading the S of the previous block.

After passing, try an experiment: swap the issue order of "PV_j" and "the next QKᵀ" in the loop, compare with `./lab bench 34 --case perf`, and explain the difference with the timeline of §5.5 (which barrier does softmax wait on before writing P_{j+1}? What is queued in front of it?). Change it back before continuing.

```bash
./lab check 34
./lab bench 34 --case perf
./lab bench 34 --case perf_causal
./lab bench 34 --case perf_d64
./lab profile 34 --case perf
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| `TIMEOUT` | expect_tx byte count; the parity of some barrier; `p_ready` count is not 128 (some thread did not arrive); the MMA warp and softmax compute different block counts |
| step 1 fails only at D = 128 | Address of the second panel of Q or K (`panel·128·64`) |
| step 2 `one_ones_v` greater than 1 | Padding keys are not masked (P ≠ 0) |
| step 2 only the last 64 columns of O are wrong at D = 128 | LBO of the MN-major descriptor |
| step 2 `one_position` wrong, `one_uniform` right | K slice offset of V, or the B major bit of idesc |
| step 2 LSE right, O all wrong | Swizzle or packing order of P |
| step 3 `ring` wrong or hangs, others right | K/V stage index, empty parity |
| step 3 `peaked*` wrong, `uniform`/`ones_v` right | O not corrected, or the α used for correction differs from the one used for ℓ |
| step 3 intermittent errors | Missing one of: the `o_ready` wait, `wait::st`, `fence.proxy.async` |
| step 4 first Q tile right, all later ones wrong | The block count misses the diagonal block |
| step 5 hangs | The two-buffer branch of `s_phase` is wrong; or the block number issued in 5b makes some buffer be written twice in a row |
| step 5 wrong results without a hang | `s_buffer` disagrees with 5b: softmax reads the S of another block |

## 7. Performance

### 7.1 Against the library baselines

Measured on GPU 6 (CUDA Graph; `vs cudnn` = cuDNN time / this version's time, higher is better):

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| Lab 22 step 4 (FA2, `mma.sync`) | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| Steps 3/4 (one S buffer, no overlap) | ≈ 0.41× | ≈ 0.51× | ≈ 0.41× |
| Step 5 (S double buffering) | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× |
| FA4 (`flash_attn.cute`) | ≈ 1.1× | ≈ 1.3× | ≈ 1.15× |

Only when step 5 reaches at least 0.48× of cuDNN on all three perf cases does it show that the overlap works (steps 3/4 without overlap are about 0.41×). cuDNN itself varies by a few percent between runs, so look at the range of the ratio.

- **Switching to `tcgen05` + warp roles** (lab 22 → step 3): about 1.6×. At this point the Tensor Core and softmax still take turns.
- **S double buffering** (step 3 → 5): about 1.3–1.4×, more at D = 64. The estimate of §3.3 is "serial ≈ MMA + softmax, overlapped ≈ max(MMA, softmax)". At D = 128, MMA ≈ softmax, so in theory it can be at most 2× faster; the measured 1.3× is lower because softmax itself takes longer than estimated (next section).
- **The causal ratio is higher than the non-causal one**: cuDNN is relatively slower on causal; after block skipping this lab does about 51% of the work. Q tiles launch in `blockIdx.x` order, so the heaviest tile starts last; lab 43 handles this.

### 7.2 Where the bottleneck is: softmax

```bash
./lab profile 34 --case perf --step 3
./lab profile 34 --case perf
```

Nsight Compute on this machine (`perf`):

| Metric | Step 3 | Step 5 | Notes |
|---|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 27% | ≈ 38% | Fraction of time the Tensor Core is busy |
| `smsp__inst_executed_pipe_xu` | 33.83 M | 33.83 M | ≈ B·H·N²/32: one `MUFU.EX2` (warp instruction) per score, plus a few for α and log |
| `dram__bytes_read` | ≈ 201 MB | ≈ 201 MB | Q, K, V each read once (K/V hit in L2 for the 32 Q tiles of the same head) |
| `sm__warps_active` | ≈ 9% | ≈ 9% | 6 warps per SM |

(Nsight Compute locks the clocks, so `gpu__time_duration` is longer than in bench; use it only to compare metrics.)

After step 5 the Tensor Core is still idle about 60% of the time: it is waiting for softmax. Per block, one softmax warpgroup has to do 128×128 `exp2` (MUFU does 16 per SM per cycle, ≈ 1024 cycles), the per-row max/sum/FFMA, write 32 KB of P, and also read O, multiply by α and write it back every block (128 values per thread at D = 128). All of this runs serially in one warpgroup. FA4 in Part 4 handles these one by one:

| Remaining problem | FA4's solution | Lab |
|---|---|---|
| There is only one softmax warpgroup, and nobody fills in while it waits for the MMA | Two Q tiles, two softmax warpgroups ping-pong | 40 |
| The O correction is on the critical path of softmax | A separate correction warpgroup | 40 |
| P is written to shared memory (32 KB/block) and read back by the MMA | P written back to TMEM, TS-MMA | 40 |
| MUFU throughput | Part of `exp2` done with an FMA polynomial | 41 |
| O corrected every block | Correct only when the maximum grows past a threshold | 41 |

**Experiments you can try yourself**:

- Use only 1 max/sum accumulator in `softmax_block`, and look at the dependency chain in SASS and the change in bench.

## 8. Self-check questions

1. In step 5 at D = 128, how many TMEM columns are actually used? Why are 512 columns allocated? If you wanted to fit the S and O of another Q tile into the same CTA, how many more columns would you need?
2. Softmax thread 70 reads columns 96–127 of S. What is the taddr of `tcgen05.ld` (relative to the TMEM base, in buffer 1 of step 5)?
3. V_j is at shared address `0x10000`, D = 128. Write the start address field and the LBO field (values after shifting right by 4) of the B descriptor for the kk = 5 K slice (keys 80–95).
4. If softmax waited for `o_ready(j−1)` before computing the exponentials (instead of after), how would the performance of step 3 and step 5 change?
5. Why is the count of `p_ready` 128, while `s_full` and `o_ready` have count 1? Could each softmax warp arrive only once?
6. In step 5, when the MMA warp issues the next QKᵀ into the buffer used by S_j, why is it certain that softmax has finished reading S_j? Which instructions together guarantee this?

<details>
<summary>Answers</summary>

1. Two S buffers take 256 columns + O takes 128 columns = 384 columns; the allocation must be a power of 2, so 512. Another Q tile needs another 256 (S double buffering) + 128 (O) = 384 columns, more than the 128 columns left. FA4's solution is to give S only one buffer per Q tile and let P reuse the columns of S (lab 40).
2. Thread 70 is in warp 2, whose lane range starts at 64: taddr = `(64 << 16) | (128 + 96)` = `(64 << 16) | 224`. Thread 70 gets lane 70 = row 70.
3. Key 80 is in row 80: start address `0x10000 + 80·128 = 0x12800`, field value `0x1280`; LBO = 16384 bytes, field value `0x400`.
4. Step 3: almost unchanged; S_j is computed only after PV_{j−1} anyway, so waiting for `o_ready` costs no extra time. Step 5: slower; when softmax starts, PV_{j−1} may still be running, and the exponentials can no longer overlap with it.
5. `tcgen05.commit` itself arrives only once, so barriers driven by the MMA have count 1. `p_ready` must guarantee that **every one** of the 128 threads has written its row of P, finished reading its row of S, and finished correcting O; each thread's fence covers only its own accesses. You could have each warp do `__syncwarp()` and then arrive from one lane (count = 4), provided every thread first executes its own `fence.proxy.async` and `tcgen05.fence::before_thread_sync` before the `__syncwarp`.
6. This QKᵀ is issued only after the MMA warp has waited on `p_ready(j)`. Before arriving, each softmax thread did: `tcgen05.ld` to read S_j, `tcgen05.wait::ld` (the read is complete), `tcgen05.fence::before_thread_sync`; after the `mbarrier` wait, the MMA thread executes `tcgen05.fence::after_thread_sync` and then issues the MMA. The count of `p_ready` is 128, so all 128 rows have been read.
</details>

## 9. Further reading

- Jay Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024): §3.1 (producer–consumer, warp specialization, ping-pong scheduling), §3.2 (two-stage softmax–GEMM pipeline inside a warpgroup), §3.3 (FP8, lab 44 of this course)
- PTX ISA 8.7: [tcgen05.st](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st), [tcgen05.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait), [Shared Memory Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor), [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor), [tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model), [cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- Upstream sources (pinned versions in `third_party/`):
  - `third_party/flash-attention/hopper/flash_fwd_kernel_sm90.h`: FA3's producer/consumer warpgroup split; `mainloop_fwd_sm90_tma_gmma_ws.hpp`: the barriers of the K/V pipeline, `IntraWGOverlap` (the two-stage pipeline of §3.2);
  - `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp`: `make_umma_desc`, the canonical MN-major and K-major layouts (the table in §4.3);
  - `third_party/cutlass/examples/77_blackwell_fmha/`: CUTLASS's SM100 FMHA; role split, TMEM allocation and S/O handoff in `collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`;
  - `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`: the full structure of FA4. Reading it now, you can use the table in §7.2 to find what this lab does not have yet (two Q tiles, correction warpgroup, P in TMEM, software `exp2`).
- Next: lab 40 (FA4's pipeline structure).
