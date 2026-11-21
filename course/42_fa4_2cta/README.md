# 42 · 2-CTA attention: a CTA pair shares a pair MMA, and each loads half of K/V

> Part 4 · FA4 — full optimization on Blackwell | Prerequisites: [40_fa4_pipeline](../40_fa4_pipeline/README.md), [41_fa4_softmax](../41_fa4_softmax/README.md), [33_2cta_gemm](../33_2cta_gemm/README.md) | Code: [`students/42_fa4_2cta/kernel.cu`](../../students/42_fa4_2cta/kernel.cu)

## 1. Goals

Lab 33 taught 2-CTA MMA on GEMM: the two CTAs of a cluster form a pair, one leader thread issues `tcgen05.mma.cta_group::2` with M = 256, A and B are each read half from the shared memory of each of the two CTAs, and the result is written to the TMEM of both CTAs. Lab 40 built the FA4 pipeline: two Q tiles ping-pong in one CTA, P is written back to TMEM, and a correction warpgroup; lab 41 sped up softmax: software `exp2` mixed with `MUFU.EX2`, and conditional rescale.

This lab starts from the lab 41 solution and adds 2-CTA, which is exactly what FA4 does with `use_2cta_instrs=True`:

| | Labs 40, 41 (1-CTA) | This lab (2-CTA) |
|---|---|---|
| Query rows per cluster | 256 (cluster = 1 CTA) | **512** (2 CTAs, 256 each) |
| QKᵀ | Each CTA runs its own M = 128 MMA | **Pair MMA, M = 256**: tile t of the leader and tile t of the peer are computed together |
| P·V | TS MMA, M = 128 | **TS pair MMA**, A = P in each CTA's own TMEM |
| K_j loaded per CTA | 128 keys × D | **64 keys × D** (the other half is in the peer) |
| V_j loaded per CTA | 128 keys × D | **128 keys × D/2 columns** (the other half of the columns is in the peer) |
| MMA warp | One per CTA | **Only in the leader**; it issues all MMAs for both CTAs |
| `p_ready` | 256 arrives in this CTA | **The leader's barrier collects 512 arrives from both CTAs** |
| softmax, correction, epilogue | Each CTA on its own (lab 41 arithmetic) | Unchanged: each CTA handles the rows in its own TMEM; the polynomial ratio stays as in lab 41 (§7.3 explains why it was not changed) |

The interface is unchanged: FP16/BF16 Q, K, V, O, `[B,N,H,D]`, D = 64 or 128, causal and non-causal, O and an FP32 LSE.

Three steps:

1. **Pair QKᵀ**: cluster launch, pair TMEM, split K load, leader-side full barriers, multicast commit, drain and cluster teardown; checks only S of the first K/V block (probe);
2. **Full 2-CTA pipeline**: split V load by columns, TS pair MMA, cross-CTA `p_ready`, causal;
3. **The cost of the cross-CTA handoff**: change the remote arrive on `p_ready` from `.release.cluster` to `.release.cta`.

When you finish, you should be able to:

- draw the data layout of the pair: which row is in which TMEM lane of which CTA, and which half of K and V is in which CTA;
- list, for each mbarrier, which CTA's copy is used, who arrives and who waits;
- derive the shared memory, L2→SM traffic and shared memory read bandwidth that 2-CTA saves;
- explain from measurements: when softmax is MUFU-bound (the lab 40 softmax), 2-CTA is actually about 2–12% slower; with the lab 41 softmax, the non-causal main loop ties with 1-CTA, but the whole kernel is still about 1–3% slower (the upper end is close to noise), and D = 64 and causal are about 5–7% slower; and the basis for choosing 1-CTA / 2-CTA per configuration.

## 2. Motivation: what is left after lab 41

Conclusion of lab 40 §7.2: the chain of one tile is softmax (≈ 2000 cycles, mostly the 128 `exp2`s) + handoff + PV + QK (≈ 1100 cycles). Lab 41 moved some `exp2`s to the FMA pipeline and skips most O corrections, so the softmax segment got shorter and the MMA side takes a larger share of the chain.

The FA4 paper and code have two more kinds of optimization:

| Direction | Method | Lab |
|---|---|---|
| Shorten softmax | Polynomial `exp2` shares the MUFU load, conditional rescale | 41 (done, reused in this lab) |
| Lighten the MMA side | **2-CTA MMA**: the B operands (K, V) are split in half between two CTAs | **This lab** |
| Scheduling | Persistent, LPT, classification of causal blocks | 43 |

2-CTA lightens three things (derived in §3): the K/V shared memory per CTA, the L2→SM traffic per output row, and the bytes the MMA reads from shared memory per cycle. It does **not** reduce the softmax work. So one important question of this lab is: how fast must softmax be for 2-CTA to pay off? §7 answers with measurements (first with the lab 40 softmax to see why it loses, then with the lab 41 softmax).

## 3. Math

The attention math is exactly the same as lab 40 (online softmax, P written back to TMEM, correction of O). This lab changes only "who computes which rows, and who loads which half of K/V".

### 3.1 Data layout of the pair

Cluster c (`%clusterid.x`) owns the 512 rows starting at q0 = 512c. Tile t of CTA r (`%cluster_ctarank`, 0 = leader, 1 = peer) is

```
rows(r, t) = q0 + 256 t + 128 r  ..  + 127
```

So pair tile t (tile t of both CTAs together) is exactly the 256 contiguous rows `q0 + 256t .. q0 + 256t + 255`:

```
             leader (rank 0)                    peer (rank 1)
             ───────────────                    ─────────────
tile 0       rows q0      .. q0+127             rows q0+128 .. q0+255     ← pair tile 0 (MMA rows 0..255)
tile 1       rows q0+256  .. q0+383             rows q0+384 .. q0+511     ← pair tile 1
TMEM         lane r of S⁰ S¹ O⁰ O¹ = row r of this CTA's tile (same column addresses in both CTAs)

K_j (128 keys × D)      keys 0..63  × D           keys 64..127 × D          ← B of QKᵀ, N = keys
V_j (128 keys × D)      keys 0..127 × cols 0..D/2-1   keys 0..127 × cols D/2..D-1   ← B of PV, N = D
```

Rules of `tcgen05.mma.cta_group::2` (lab 33 §4.4): rows 0–127 of A come from the leader and rows 128–255 from the peer; **B is split by N**: the first half of N comes from the leader, the second half from the peer; rows 0–127 of D go to the leader's TMEM, rows 128–255 to the peer's TMEM.

- QKᵀ: A = Q (M is the query), B = K (N is the key). So K is split in half by **key**.
- PV: A = P (M is the query), B = V (N is the column of D, K is the key). So V is split in half by **column**: each CTA needs all 128 keys but only D/2 columns.

Small example (D = 128, N = 4096, cluster 0): the leader loads Q rows 0–127 and 256–383, keys 128j..128j+63 of K_j, and columns 0–63 of V_j; the peer loads Q rows 128–255 and 384–511, keys 128j+64..128j+127 of K_j, and columns 64–127 of V_j. After the pair MMA, the leader's S⁰ holds the scores of rows 0–127 against 128 keys, and the peer's S⁰ holds those of rows 128–255.

### 3.2 What it saves

Bytes per K/V block (128 keys) per CTA (16-bit elements):

| | Lab 40 | This lab | Ratio |
|---|---|---|---|
| K_j + V_j into shared memory | 2 · 128 · D · 2 B (D = 128: 64 KB) | Half (32 KB) | 0.5 |
| Blocks that fit in the same ring bytes | 128 KB / 64 KB = 2 blocks | 128 KB / 32 KB = 4 blocks | 2 |
| L2→SM K/V traffic (per output row) | 64 KB / 256 rows = 256 B/row | 32 KB / 256 rows = 128 B/row | 0.5 |

For the whole `perf` case (B·H = 64, N = 4096, D = 128): lab 40 moves 1024 CTAs × 32 blocks × 64 KB ≈ 2.15 GB of K/V plus 67 MB of Q; this lab ≈ 1.07 GB + 67 MB.

Bandwidth for the MMA to read operands from shared memory (estimated at 128 B per SM per cycle, and about 8192 BF16 FLOP per SM per cycle):

| MMA (the share of each SM) | Bytes read by one K = 16 MMA | Cycles it takes | Read per cycle |
|---|---|---|---|
| Lab 40 QKᵀ, M = 128, N = 128 (A and B both in smem) | A 4 KB + B 4 KB | 128·128·16·2 / 8192 = 64 | 128 B, **exactly saturated** |
| This lab QKᵀ (pair, half of B per SM) | A 4 KB + B 2 KB | 64 | 96 B (0.75) |
| Lab 40 PV, N = D = 128 (A in TMEM) | B 4 KB | 64 | 64 B |
| This lab PV (pair) | B 2 KB | 64 | 32 B (0.5) |

So the 1-CTA QKᵀ has no headroom in shared memory bandwidth; 2-CTA lowers it to 75%. This helps only when the MMA side becomes the bottleneck (§7).

### 3.3 Costs

- **Two SMs are tied together**: the PV of pair tile t must wait until the softmax of **both** CTAs has handed over P, and S is written into both CTAs at the same time. The chain of each block gains two cross-SM handoffs (the multicast commit to the peer, and the peer's arrive to the leader).
- **The diagonal waste of causal doubles**: a pair tile is 256 rows, the two tiles of one cluster advance together (ping-pong), and the cluster's time is set by tile 1, which goes furthest: cluster c walks `4c + 4` blocks. In lab 40 the same 512 rows are handled by two CTAs that walk `4c + 2` and `4c + 4` blocks. At N = 4096 the average SM time per 512 rows is 2·(4c+4) ≈ 36 blocks vs (8c+6) ≈ 34 blocks, about 6% more. Tile 0 walking two fewer blocks (`tile_blocks` in step 2) does not win it back: tile 1 still has to walk to the end, and stopping tile 0 early only leaves the last two blocks of tile 1 without a ping-pong partner.
- **Cluster launch**: a CTA pair needs two idle SMs at the same time; the `cta_group::2` TMEM allocation is a cross-CTA software protocol (lab 33 NOTES). Measured (lab 41 softmax, `perf`): the whole setup grows from about 0.45 µs to about 1.0 µs, and the teardown at the end of a CTA from about 1.1 µs to about 1.8 µs (CTA lifetime about 65 µs). How much of the extra setup is the allocation protocol was not measured separately.

## 4. Hardware and instructions

### 4.1 New header functions in this lab

| Function (header) | PTX | Purpose |
|---|---|---|
| `lab::tma_load_4d_cg2` (`cluster.cuh`) | `cp.async.bulk.tensor.4d.cta_group::2...` | 4-D version of lab 33's `tma_load_2d_cg2`: the box is written into this CTA, and complete_tx lands on the leader's barrier |
| `lab::mma_f16_ts_cg2` (`cluster.cuh`) | `tcgen05.mma.cta_group::2.kind::f16 [d], [a_tmem], b_desc, idesc, p` | TS-form pair MMA: A comes from the same address in each CTA's own TMEM |
| `lab::mbar_arrive_remote` (`cluster.cuh`) | `mbarrier.arrive.shared::cluster.b64` (default `.release.cta`) | The cheap remote arrive of step 3 (§4.4) |
| `lab::make_smem_desc_mnmajor_sw64` (`tcgen05.cuh`) | Descriptor, layout = 4 (SWIZZLE_64B) | At D = 64 a half of V has only 32 columns = 64 B per row |

The other pair primitives (`tmem_alloc_cg2`, `mma_f16_ss_cg2`, `tcgen05_commit_cg2_multicast`, `map_shared_rank`, `mbar_wait_cluster`, `mbar_arrive_cluster`, `cluster_sync`, `launch_cluster`) were written in lab 33. All tcgen05 instructions in one kernel must use the same `cta_group`, so in this lab even the commits use the `cg2` multicast version.

### 4.2 Pair QKᵀ

```cpp
idesc = make_idesc_f16(256 /*M of the pair*/, 128 /*N = keys*/, bf16);
da = make_smem_desc_kmajor_sw128(sq);   // leader's Q^t; the peer's is at the same offset
db = make_smem_desc_kmajor_sw128(sk);   // leader's half of K: [D/64 panel][64 key][64]
for kk: mma_f16_ss_cg2(s_tmem, da + a_off(kk), db + b_off(kk), idesc, kk > 0);
```

The only difference from lab 40's `mma_qk` is that a K panel has only 64 rows (`kHalfKeys`), so `b_off = (panel · 64 · 64 + col) · 2 / 16`. The descriptor describes only the leader's buffer, and the hardware reads the other half at the same offset in the peer, so **the shared memory layout of the two CTAs must be exactly the same**.

### 4.3 TS pair PV and 64B swizzle

```cpp
idesc = make_idesc_f16(256, D /*N = all D columns*/, bf16, true, false, /*b_mn_major=*/true);
db = D == 128 ? make_smem_desc_mnmajor_sw128(sv, ...)   // a row of 64 columns = 128 B
              : make_smem_desc_mnmajor_sw64(sv, ...);   // a row of 32 columns = 64 B
for kk: mma_f16_ts_cg2(o_tmem, p_tmem + 8 kk, db + kk · 16 · rowbytes / 16, idesc, acc || kk > 0);
```

- The N of idesc is the **whole** D: the hardware splits N in half between the two CTAs. Each CTA's half of V is [128 key][D/2], MN-major.
- At D = 128 a half = 64 columns = 128 bytes, exactly one 128B swizzle row (the lab 40 layout). At D = 64 a half has only 32 columns = 64 bytes, so 128B swizzle cannot be used and you need **64B swizzle**: a TMA box 32 elements wide with `CU_TENSOR_MAP_SWIZZLE_64B`, descriptor layout field = 4, 8 rows per 512-byte atom (SBO = 512), and advancing 16 rows in the K direction = 16 · 64 bytes.
- The address rule for A is the same as lab 40 (8 columns per 16 keys); the P of the leader and of the peer are in the same column of each one's own TMEM.
- Each CTA has only one MN panel, so LBO is not used.

### 4.4 Cross-CTA arrive: `.release.cluster` vs `.release.cta`

After the peer's softmax and correction threads finish writing P into their own TMEM and correcting O, they must tell the **leader's** MMA thread. They arrive remotely on the leader's `p_ready` (the address comes from `mapa`):

```cpp
tcgen05_fence_before_thread_sync();       // order this thread's tcgen05.ld/st before the arrive
arrive(map_shared_rank(&bar.p_ready[t], 0));
```

Lab 33 used `mbar_arrive_cluster`: `mbarrier.arrive.release.cluster`. Compiled to SASS it is (compile a kernel that contains only this instruction yourself, then `cuobjdump -sass`):

```
.release.cluster:  MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; SYNCS.ARRIVE.TRANS64.RED.A1T0 [addr]
.release.cta:      SYNCS.ARRIVE.TRANS64.RED.A1T0 [addr]
```

`.release.cluster` makes **all** memory writes before this thread's arrive visible at cluster scope; the hardware implements it with a GPU-scope `MEMBAR`, which waits for all of this thread's in-flight memory accesses to complete. It happens once per block, per tile, in each of 512 threads. Instrumentation (data in `solutions/42_fa4_2cta/NOTES.md`): in step 2, the leader's MMA thread sees `p_ready` complete about 650–900 ns after the last thread arrives; in step 3, about 100 ns.

`.release.cta` is formally **not guaranteed** to be correct, even though the data is in TMEM:

- In the PTX memory model, the scopes of the peer thread's `.release.cta` arrive and the leader's `try_wait.acquire.cluster` do not both include the two threads, so the pair is not morally strong and there is **no synchronizes-with**. The tcgen05 ordering rule (`fence::before_thread_sync` … thread synchronization … `fence::after_thread_sync`) depends on exactly this thread synchronization, so "the data is in TMEM" does not close the formal gap.
- Only two things are guaranteed: the arrive is always counted (the phase of `p_ready` waits for all 512 arrives); and this thread's `tcgen05.wait::st` / `wait::ld` has completed its own TMEM accesses before the arrive.
- The rest relies on the behavior of the current hardware and compiler. This is a hardware-level argument: the TMEM write has completed when `wait::st` returns, `SYNCS.ARRIVE` is issued after it, and the leader issues the MMA only after it sees the phase complete. CUTLASS (`ClusterBarrier::arrive(cta_id, pred)`, `include/cutlass/arch/barrier.h`) and FA4 (the pipeline's `consumer_release` arrives remotely on the leader) both do this. The comment in the CuTe DSL's `cute/arch/mbar.py` says it plainly: a remote arrive should formally use cluster scope; CTA is the default because existing protocols rely on barrier counting and current hardware/compiler behavior, and cluster scope is "measured to be slower".

So step 3 is a **deliberate trade-off**: give up the formal guarantee for about 1.2 times the speed, the same as CUTLASS/FA4. For formal correctness, keep the `.release.cluster` of step 2; when handing off data in shared/global memory (for example writing the other CTA's smem through DSMEM), you must not drop it.

### 4.5 Common misuses

| Misuse | Consequence |
|---|---|
| The peer also issues MMAs | Every block is computed twice, S/O are written twice: wrong results or a hang |
| The peer also does `arrive_expect_tx` on the same leader barrier | The tx byte count doubles and there is one more arrive: the phase completes early or never completes |
| The commit is not multicast (`tcgen05_commit` of `cta_group::1`) | Mixing cta_groups is not allowed at compile time; even with a mask that contains only the leader, the peer's softmax never sees `s_full` |
| K split by D, V split by key | The directions are swapped: B is always split by **N**; the N of QKᵀ is the key, the N of PV is the column of D |
| The V half at D = 64 still uses 128B swizzle | A 32-element box is only 64 B wide, so the 128B swizzle addressing is wrong, and O is wrong |
| `__syncthreads` instead of a cluster barrier | The peer's TMA or the leader's commit can arrive before the other CTA's `mbarrier.init` |
| One CTA exits early | The leader's MMA is still reading its smem and writing its TMEM; or a multicast lands on a CTA that has exited |

## 5. Schedule design

### 5.1 Roles and storage budget

The roles are the same as lab 40 step 3 (16 warps, `setmaxnreg` 192/80/48), except that the MMA warp works only in the leader:

| warp | leader | peer |
|---|---|---|
| 0–3, 4–7 | softmax tile 0, tile 1 | Same (its own rows) |
| 8–11 | correction + epilogue | Same |
| 12 | **MMA: all MMAs of both CTAs** | Idle (goes straight to the cluster barrier of the teardown) |
| 13 | TMA: its own Q, half of K/V | Same |
| 14–15 | Idle | Idle |

Shared memory per CTA (the 1024 B alignment slack is extra):

| | D = 128 | D = 64 |
|---|---|---|
| Q (2 tiles; also the O staging buffer in the epilogue) | 64 KB | 32 KB |
| K/V ring: one stage = one "half" | 8 stages × 16 KB = 128 KB | 16 stages × 8 KB = 128 KB |
| Correction statistics | 3 KB | 3 KB |

The ring has the same bytes as lab 40, but twice as many stages (lab 40 at D = 128 has 4 stages × 32 KB): the same space can prefetch 4 blocks instead of 2. TMEM is exactly the same as lab 40, and both CTAs use the same columns (the `cta_group::2` alloc guarantees that both sides get the same base address).

Host side (already written): the Q box is [128 rows][64], the K box is **[64 rows][64]**, the V box is [128 rows][64] for D = 128 and **[128 rows][32] + 64B swizzle** for D = 64, and the O box is [32 rows][64]. grid = (2 · ⌈N/512⌉, H, B), cluster = (2, 1, 1).

### 5.2 TMA: each CTA loads its own half, completion on the leader's barrier

```
for t: (leader) arrive_expect_tx(q_full[t], 2 × Q bytes of this CTA)
       tma_load_4d_cg2(sq(t), Q, rows(rank, t), → leader's q_full[t])
for u = 0 .. 2·blocks − 1:                     # even: K_{u/2}, odd: V_{u/2}
       wait this CTA's kv_empty[u % S]
       (leader) arrive_expect_tx(kv_full[s], 2 × bytes of one half)
       K: tma_load_4d_cg2(ring(s) + panel, K, x = 64·panel, row = 128j + 64·rank) × D/64
       V: tma_load_4d_cg2(ring(s), V, x = rank · D/2, row = 128j)
drain: wait S more times on kv_empty (exit only after all of the leader's multicast commits have arrived)
```

Only the leader arms, and the byte count is the sum over both CTAs. The peer's complete_tx may arrive before the leader's `arrive_expect_tx`: a temporarily negative tx count is allowed, and the phase still waits for the leader's 1 arrive (lab 33 NOTES).

### 5.3 MMA: one leader thread issues for both CTAs

The order is the same as lab 40 (`PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1}`); each one is a pair MMA, and each commit is multicast to both CTAs (mask `0b11`):

```
prologue: wait kv_full(K₀); for t: wait q_full[t]; QK^t_0 → commit s_full[t] (both)
          commit kv_empty[0] (both)
for j:
  wait kv_full(V_j), kv_full(K_{j+1})
  for t (j < blocks_t):
    wait_cluster p_ready[t] (phase j)      ← 512 arrives in total from both CTAs
    PV^t_j = P^t (TMEM of both) · V_j (each side holds half the columns)
    (t = 1) commit kv_empty[V_j] (both)
    j+1 < blocks_t ? QK^t_{j+1} + commit s_full[t] (both) : commit o_full[t] (both)
  commit kv_empty[K_{j+1}] (both)
```

### 5.4 Barrier map

The same `Barriers` struct has one copy at the same offset in each of the two CTAs; in the table, L = leader and P = peer.

| barrier | Which copy | count / per phase | Who arrives | Who waits |
|---|---|---|---|---|
| `q_full[t]` | L | 1 + tx | L's TMA (expect_tx with the bytes of both sides); complete_tx of both sides' TMA | L's MMA |
| `kv_full[s]` | L | 1 + tx | Same as above | L's MMA |
| `kv_empty[s]` | L and P | 1 | L's MMA: `commit.cta_group::2.multicast 0b11` | Each side's own TMA |
| `s_full[t]` | L and P | 1 | L's MMA: multicast commit | Each side's own softmax t |
| `p_ready[t]` | L | 2 × 256 | Softmax t (128) + correction (128) of both sides, remote arrive | L's MMA (`try_wait.acquire.cluster`) |
| `o_full[t]` | L and P | 1 | L's MMA: multicast commit | Each side's own correction (epilogue) |
| `stats_full[t]`, `final_full[t]` | Each its own | 128 | This CTA's softmax t | This CTA's correction |

```
             leader CTA                                     peer CTA
 ┌────────────────────────────────────┐       ┌────────────────────────────────────┐
 │ TMA ──expect_tx──► q/kv_full ◄─────┼───────┼── complete_tx ── TMA               │
 │                       │ wait       │       │                                    │
 │  ┌────────────────────┘            │       │                                    │
 │  ▼                                 │       │                                    │
 │ MMA ═══════ pair MMA ══════════════╪═══════╪══► reads Q and the K/V half        │
 │ │ ▲                                │       │    from smem; writes TMEM          │
 │ │ │ wait                           │       │                                    │
 │ │ └──── p_ready ◄══════════════════╪═══════╪═══ remote arrive ═══ softmax/corr  │
 │ │        ▲                         │       │                                    │
 │ │        └─arrive─ softmax/corr    │       │                                    │
 │ │ commit multicast                 │       │                                    │
 │ ├──► s_full, o_full, kv_empty      │       │                                    │
 │ └──────────────────────────────────┼───────┼──► s_full, o_full, kv_empty        │
 │                                    │       │    (same commit also reaches peer) │
 └────────────────────────────────────┘       └────────────────────────────────────┘
```

**Setup**: thread 0 initializes + `fence.mbarrier_init.release.cluster`; warp 0 on both sides does `tcgen05.alloc.cta_group::2` (512 columns) + relinquish; `fence::before` → `barrier.cluster` → `fence::after`. The cluster barrier guarantees that nobody signals the other CTA's barriers before they are initialized.

**Teardown (drain)**: before it exits, each CTA must receive everything others send to it:

| Sent to | What | Who guarantees it has all arrived |
|---|---|---|
| peer, leader | Multicast commits of `kv_empty` | Each side's TMA drain (waits S more times) |
| peer, leader | Multicast commits of `s_full`, `o_full` | Softmax has waited on every `s_full` phase, correction on `o_full` |
| leader | The peer's remote arrives (`p_ready`), the peer's TMA complete_tx | The leader's MMA has waited on every `p_ready` and `kv_full` phase |
| peer | The leader's MMA reads the peer's smem and writes the peer's TMEM | `o_full` is the last commit and tracks all earlier MMAs |

Then `fence::before` → `barrier.cluster` → warp 0 does `fence::after` + `tcgen05.dealloc.cta_group::2`.

### 5.5 Causal

The last row of pair tile t is `q0 + 256t + 255`, and the number of blocks it walks is `min(⌈N/128⌉, ⌊(q0 + 256t + 255)/128⌋ + 1)`; the two CTAs must agree on it (they share the MMA). Tile 0 walks two fewer blocks than tile 1; the MMA skips tile 0 when `j ≥ blocks_0`. In each pair tile, the last block of the leader's 128 rows is fully masked (the same as tile 0 in lab 40; m still comes from earlier blocks, so no NaN appears). The mask itself is taken over from lab 40 unchanged and is computed from this CTA's `qt` and row index.

### 5.6 Timeline (steady state, D = 128, one cell ≈ 250 ns)

```
leader MMA  │PV⁰QK⁰│      │PV¹QK¹│      │PV⁰QK⁰│      │PV¹QK¹│
Tensor(L,P) │██████│      │██████│      │██████│      │██████│     ← both SMs execute the same pair MMA at the same time
softmax 0 L │      │ld max exp exp exp P│      │ld max exp exp exp P│
softmax 0 P │      │ld max exp exp exp P│      │ld max exp exp exp P│  ← PV⁰ can issue only after both sides arrive
softmax 1 L │  … exp exp P│      │ld max exp exp exp P│      │ld max …
softmax 1 P │  … exp exp P│      │ld max exp exp exp P│      │ld max …
                                        ↑ p_ready⁰ = the later of the two CTAs + remote arrive latency
```

## 6. Step-by-step exercises

Everything from lab 40 in the skeleton (softmax, correction, epilogue, role split, `setmaxnreg`, host-side tensor maps, dispatch) is already written. `p.step` selects `Config<D, 1|2|3>`; the three steps **complete the same kernel one piece at a time**.

### Step 1 · TODO(1a)–(1f): pair QKᵀ

- 1a: the host-side grid and `lab::launch_cluster` (§5.1);
- 1b: setup and teardown: allocation/release of pair TMEM, cluster barriers (§5.4);
- 1c: `tile_row`: the first row of tile t of CTA r (§3.1);
- 1d: TMA loads of Q and the half of K (§5.2);
- 1e: `mma_qk_pair` (§4.2);
- 1f: the multicast commit in the MMA prologue (§5.3).

This step computes only the first K/V block: the softmax threads write `scale · S` into the probe (already written), and the check is on S of shape `[B, H, N, min(N,128)]`, covering all four tiles of a cluster and both halves of K.

```bash
./lab check 42 --step 1 --case two_tiles      # the first pair tile = leader + peer
./lab check 42 --step 1                       # includes cases such as 3 clusters, or only 3 valid tiles
```

### Step 2 · TODO(2a)–(2e): the full 2-CTA pipeline

- 2a: TMA load of the half of V (§5.2);
- 2b: `mma_pv_pair`, including the 64B swizzle for D = 64 (§4.3);
- 2c: the MMA main loop (§5.3);
- 2d: the step-2 branch of `arrive_leader` (`.release.cluster`) and `kPReadyArrivals` (§4.4, §5.4);
- 2e: the causal block count (§5.5).

```bash
./lab check 42 --step 2 --case peer_position   # the peer's P, the half of the V columns
./lab check 42 --step 2 --case position        # D = 64: 64B swizzle
./lab check 42 --step 2
./lab bench 42 --step 2 --case perf
```

### Step 3 · TODO(3): the cost of the cross-CTA handoff

First look at the symptom:

```bash
./lab sass 42        # in the full SASS under artifacts/sass/, find where MEMBAR.ALL.GPU appears
./lab bench 42 --step 2 --case perf
```

Then, in the step-3 branch of `arrive_leader`, switch to a remote arrive without the GPU-scope fence (§4.4), and say why this lab can do so.

```bash
./lab check 42 --step 3
./lab bench 42 --case perf
./lab bench 42 --case perf_causal
./lab bench 42 --case perf_d64
./lab bench 42 --case perf_8k
./lab bench 41 --solution --case perf          # compare with 1-CTA (lab 41)
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| Step 1 `TIMEOUT` | Only the leader loads K/Q, or the peer's TMA uses its own barrier (it must be `map_shared_rank(…, 0)`); the byte count of `arrive_expect_tx` is not the sum over both sides; the peer also arms |
| Step 1: only rows 128–255, 384–511 are wrong | The `tile_row` mapping; or the peer's Q is loaded at a different offset |
| Step 1: the half with keys 64–127 is wrong | The row coordinate of the K half (`+ 64·rank`); `b_off` uses a 128-row panel |
| Step 1/2 hangs at the end of the kernel or reports an illegal address | The teardown is missing a cluster barrier or the drain; the peer exits early |
| Step 2 `TIMEOUT` | `kPReadyArrivals` is not 512; the leader uses `mbar_wait`, or the peer's arrive goes to its own CTA's `p_ready`; `o_full`/`s_full` are not multicast |
| Step 2: D = 128 is right, O at D = 64 is wrong | The swizzle or descriptor of the V half: 64B swizzle, 64 B row width, advancing 1024 B per 16 rows in the K direction |
| Step 2: the second half of the O columns is wrong | The x coordinate of V (`rank · D/2`); the N of idesc uses D/2 |
| Causal: only later clusters are wrong | The block count uses the CTA's rows instead of the last row of the pair tile |
| Step 3 is not faster | Still using `.release.cluster`; check MEMBAR with `./lab sass 42` |

## 7. Performance

### 7.1 Against the library baseline and lab 41

Measured on GPU 3 (CUDA Graph; `vs cudnn` = cuDNN time / time of this version, higher is better; range over two runs):

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| Lab 41 step 3 (1-CTA, reference) | ≈ 0.94–0.95× | ≈ 1.04–1.05× | ≈ 0.95× | ≈ 0.92× |
| Step 2 (2-CTA, `.release.cluster`) | ≈ 0.73–0.74× | ≈ 0.79–0.80× | ≈ 0.80–0.81× | ≈ 0.71–0.75× |
| Step 3 (2-CTA, `.release.cta`) | ≈ 0.90–0.92× | ≈ 0.96× | ≈ 0.87–0.88× | ≈ 0.89–0.90× |
| Speed of step 3 relative to lab 41 | ≈ 0.96–0.99× | ≈ 0.93–0.95× | ≈ 0.93–0.94× | ≈ 0.98–0.99× |
| FA4 (`flash_attn.cute`) | ≈ 1.15–1.16× | ≈ 1.33–1.35× | ≈ 1.14–1.16× | ≈ 1.08–1.10× |

If your step 3 reaches at least about 0.8× cuDNN on all four perf cases, your 2-CTA pipeline is right. Key points:

- Step 2 → 3: about 1.2 times, all from the fence of §4.4.
- Non-causal D = 128 (`perf`, `perf_8k`) roughly ties with lab 41; D = 64 and causal still lose (§7.4).
- 2-CTA by itself does not make this lab beat lab 41. The next three sections explain why, and what it needs to win.

### 7.2 Why 2-CTA loses when softmax is MUFU-bound

First, the same 2-CTA with the lab 40 softmax (all 128 `exp2`s on MUFU, and no conditional rescale) (a maintainer experiment; the instrumentation code is not in the repository; `perf`, averaged over all CTAs):

| Metric | Lab 40 (1-CTA) | Lab 40 softmax + 2-CTA |
|---|---|---|
| Relative to cuDNN | ≈ 0.84× | ≈ 0.75× |
| CTA main loop (32 blocks) | ≈ 69.5 µs | ≈ 78 µs |
| Cycles per block (clock ≈ 1.62 / 1.66 GHz) | ≈ 3500 | ≈ 4050 |
| Tensor Core busy (NCU, locked clocks) | ≈ 57% | ≈ 51% |
| MUFU instruction count | 33.8 M | 33.8 M |
| L2→SM reads (`lts__t_sectors_srcunit_tex_op_read`) | ≈ 1.49 GB | ≈ 1.01 GB (≈ 0.68×; derived 0.53×) |

When `exp2` stretches the chain, the resources saved on the MMA side go unused, while the pair MMA locks the two SMs together: the PV of a tile has to wait for the later of the two CTAs' softmax, and jitter on either side delays the interleaving of both tiles. We did not fully locate the extra ~500 cycles per block (a small amount of instrumentation already shrinks the gap to about 3%); the above is a hypothesis, not a measured conclusion.

An experiment that is **for timing only and gives wrong results** points the way: softmax does `MUFU.EX2` on only half of the elements and replaces the other half with one FMA. The lab 40 main loop drops from 69.6 to 61.2 µs and then stops getting shorter (doing only 1/4 or 1/128 is the same), while 2-CTA drops from 78.2 to 56.4 µs and then to about 55 µs: once softmax is fast enough, 1-CTA is held back by the MMA side (QKᵀ saturates shared memory bandwidth, §3.2), 2-CTA is not, and it is about 9–12% faster.

### 7.3 Switching to the lab 41 softmax

The lab 41 softmax is the real version: 3 of every 16 column pairs take the polynomial path (about 10 instructions per pair on the FMA pipeline), the rest take MUFU; conditional rescale means correction does nothing in most blocks. Main loop (`perf`, averaged over all CTAs):

| | Lab 41 (1-CTA) | This lab, step 3 |
|---|---|---|
| CTA main loop (32 blocks), D = 128 | ≈ 61.4 µs | ≈ 61.4 µs (tie) |
| Same, D = 64 (`perf_d64`) | ≈ 52.5 µs | ≈ 53.8 µs |
| Setup / teardown | ≈ 0.45 / 1.1 µs | ≈ 1.0 / 1.8 µs |

- Compared with lab 41, the main-loop gap shrinks from about 12% (2-CTA with the lab 40 softmax) to 0. The software `exp2` still has work to do on the FMA pipeline, and softmax is not as fast as in the fake experiment, so the headroom on the MMA side is only enough to offset the coupling of the pair, not to turn into a gain.
- The remaining ~1–3% (the upper end is close to noise) comes from one-time costs per CTA: the whole setup takes about 0.55 µs more (presumably mostly the `cta_group::2` TMEM allocation protocol and the cluster barrier; not measured separately), the teardown adds one cluster barrier and `dealloc.cta_group::2`, and a CTA pair must exit together before the next cluster can be placed. The CTA lifetime is about 65 µs, and each CTA costs about 1.2 µs more. A persistent kernel (lab 43) pays this cost only once.
- **Should the polynomial ratio be retuned?** With 2-CTA the MMA and shared memory load per SM is lighter, so the balance point may move. Measured (`kPolyPairs` = 2, 3, 4, 5, 6): at D = 128, 3, 4 and 5 are within noise, while 2 and 6 are slower; at D = 64, 4 is about 3% faster than 3 (main loop 53.8 → 51.8 µs, `perf_d64` ≈ 0.81 → 0.785 ms), but 4 makes the causal D = 64 kernel spill under 192 registers (STACK 24 in `./lab sass`). The solution keeps lab 41's 3. τ = 8 is not changed: it is set by the FP16 upper bound (lab 41 §3.5), not a performance parameter.

### 7.4 1-CTA or 2-CTA: choose by configuration

| Configuration | Measured (step 3 relative to lab 41) | Reason | Choice |
|---|---|---|---|
| Non-causal, D = 128 | ≈ 0.96–0.99× | The main loop ties, and each CTA pays about 1.2 µs of fixed cost | Without persistence the difference is close to noise; with persistence it may favor 2-CTA (an untested prediction; this course does not measure persistent + 2-CTA) |
| Non-causal, D = 64 | ≈ 0.93–0.94× | Each block has only half the MMA, so the MMA side is even less of a bottleneck | 1-CTA |
| Causal | ≈ 0.93–0.95× | The 512 rows of a cluster advance together, so the diagonal waste is about 2 blocks / 512 rows more than 1-CTA (§3.3) | 1-CTA |

This matches FA4's choice (`use_2cta_instrs` in `flash_attn/cute/interface.py`): it is turned on only for **non-causal**, head_dim padded to 128 or 192, head_dim_v = 128 (plus a dedicated kernel for hdim 256); it is not used for D = 64 or causal (the code only states the conditions; the table above is our interpretation). The first item of the key of FA4's `_TUNING_CONFIG` is `use_2cta_instrs`: the 2-CTA D = 128 non-causal configuration uses 184 softmax registers and `ex2_emu_freq = 10`. FA4 also has parts this course does not do (persistent, handing P to the MMA in segments, etc.); we did not separately measure the difference between FA4's own 1-CTA and 2-CTA.

This lab's `run()` does **not** switch by configuration: every case goes through the pair path so that it gets verified; by the table above, a production `run()` would consider 2-CTA only for non-causal D = 128, and the difference measured in this lab is not enough to prove that it is worth it.

```bash
./lab profile 42 --case perf
```

Look at `sm__pipe_tensor_cycles_active`, `lts__t_sectors_srcunit_tex_op_read` (compare with lab 41) and `smsp__inst_executed_pipe_xu`.

## 8. Self-check questions

1. At D = 128, how many bytes does the leader's TMA arm on `kv_full[s]`? And on `q_full[t]`? What happens if the peer also arms once?
2. In the idesc of the pair PV, is N written as D or D/2? Why is K for QKᵀ split by key, but V for PV split by column?
3. Why does `p_ready` use only the leader's copy, while `s_full` uses both? Why is the count of `p_ready` 512?
4. The peer's MMA warp does nothing and goes straight to the cluster barrier of the teardown. After what point can the peer safely dealloc TMEM and exit?
5. Why is switching the arrive on `p_ready` to `.release.cta` acceptable in this lab? Give an example where `.release.cluster` is required.
6. In this lab, causal loses more relative to lab 40 than non-causal. Explain with the number of blocks that cluster c walks.

<details>
<summary>Answers</summary>

1. One half = 64 keys × 128 × 2 B = 16 KB; both CTAs together 32 KB. `q_full[t]`: 2 × 128 × 128 × 2 B = 64 KB. If the peer also arms, there is one extra arrive (a phase with count 1 is completed early by the peer's arrive, or the next phase gets an extra one) and extra tx bytes: reading data that has not finished loading, or a hang.
2. N is the whole D: the hardware splits N in half between the two CTAs. B of `cta_group::2` is always split by N; the N of QKᵀ is the key, and the N of PV is the column of D (the key is the K dimension of PV, and each CTA needs all 128 keys).
3. The only consumer of `p_ready` is the leader's MMA thread, so only the leader's copy is needed, and the producers on both sides arrive there; the consumers of `s_full` are each side's own softmax, so the commit is multicast to one copy on each side. count = 2 CTAs × (128 softmax threads + 128 correction threads).
4. After it has received every signal sent to it and the leader's MMA no longer reads or writes it: its TMA has drained `kv_empty`, softmax has waited on every `s_full`, correction has waited on `o_full` (the last commit, which tracks all earlier MMAs, including reading the peer's Q, K/V and writing the peer's TMEM); then both sides pass the cluster barrier.
5. Formally it is not acceptable: a cross-CTA `.release.cta` and `acquire.cluster` do not form synchronizes-with, and TMEM is no exception. All that is guaranteed is that the arrive is counted and that this thread's `wait::st` has completed its TMEM writes; the rest relies on hardware behavior, CUTLASS/FA4 do the same, and it buys about 1.2 times. If the peer writes data into the leader's shared memory (DSMEM) with ordinary stores and then arrives to tell the leader to read it, it must use `.release.cluster` (paired with the leader's `.acquire.cluster`).
6. The two pair tiles of cluster c advance together, and the time is set by tile 1: 4c + 4 blocks, with both SMs occupied. In lab 40 the same 512 rows are two CTAs walking 4c + 2 and 4c + 4 blocks. At N = 4096 that is about 2 more blocks per 512 rows on average (≈ 6%); stopping tile 0 early does not win it back.
</details>

## 9. Further reading

- Zadouri et al., [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) (2026); upstream implementation `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`:
  - `__init__`: `use_2cta_instrs`, `cta_group_size`, `mma_tiler_qk = (cta_group_size · m_block_size, n_block_size, D)`, `mma_tiler_pv`, `cluster_shape_mn = (2, 1)`;
  - `_setup_attributes`: `smem_size_kv_per_stage = max(K, V) // cta_group_size`, `kv_stage`;
  - `tma_copy_bytes[...] *= cta_group_size` (the leader arms the bytes of both sides at once);
  - `softmax_correction_threads_cluster` (the number of pipeline producers for `p_ready` includes both CTAs);
  - `_TUNING_CONFIG` at the top of the file (the first item of the key is `use_2cta_instrs`);
  - the condition of `use_2cta_instrs = (...)` in `flash_attn/cute/interface.py`.
- CUTLASS: `third_party/cutlass/include/cutlass/arch/barrier.h` (the remote arrive of `ClusterBarrier::arrive(smem_ptr, cta_id, pred)` uses the default semantics); the FMHA forward in `examples/77_blackwell_fmha` is 1-SM; 2-SM is used only for MLA (`kernel/sm100_fmha_mla_tma_warpspecialized.hpp`, `kIs2Sm`, `Allocator2Sm`).
- CuTe DSL (`nvidia_cutlass_dsl/.../cutlass/cute/arch/mbar.py` in the conda environment, the `scope` comment of `mbarrier_arrive`).
- PTX ISA 8.7: [tcgen05.mma (cta_group::2, A in TMEM)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma), [cp.async.bulk.tensor (.cta_group)](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor), [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier), [tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model), [Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model) (scope and morally strong).
- This repository: lab 33 lesson §4–§5 and `solutions/33_2cta_gemm/NOTES.md` (pair barriers, drain, racecheck false positives); lab 40 lesson §3, §5, §7.2.
