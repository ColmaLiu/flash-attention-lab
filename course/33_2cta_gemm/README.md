# 33 · 2-CTA MMA: a tcgen05 GEMM with clusters and paired CTAs

> Part 3 · FA3 ideas on SM100 | Prerequisites: [32_ws_gemm](../32_ws_gemm/README.md) (warp specialization, persistent, accumulator double buffering) | Code: [`students/33_2cta_gemm/kernel.cu`](../../students/33_2cta_gemm/kernel.cu)

## 1. Goals

This is still the GEMM of lab 32: `C[M,N] = A[M,K] · B[N,K]ᵀ`, BF16 inputs, FP32 output, the same supported domain (`K % 8 == 0`, `N % 4 == 0`, `K > 0`). The structure of lab 32 (producer warp, MMA warp, 4 epilogue warps, persistent, accumulator double buffering, grouped rasterization) stays as it is. This lab changes **how many SMs do one MMA together**:

| | Lab 32 | This lab |
|---|---|---|
| One output tile | 128 × 256, one CTA | 256 × 256, **two CTAs (one pair)** |
| MMA instruction | `tcgen05.mma.cta_group::1`, M = 128 | `tcgen05.mma.cta_group::2`, M = 256, **issued only by the leader CTA** |
| Where A and B are | This CTA's shared memory | Half of A and half of B in each of the two CTAs' shared memory |
| L2 reads per CTA per K tile | 48 KB | 32 KB (with step 3's multicast: 24 KB) |
| Between CTAs | No interaction | **cluster**: can read and write the other's shared memory and arrive on the other's mbarriers |

This lab has three steps:

1. **cluster and pair MMA**: launch a 2-CTA cluster with `cudaLaunchKernelEx`. Each of the two CTAs loads half of A and half of B, the leader issues the `cta_group::2` MMA, and the result is split across the TMEM of the two CTAs. One cluster computes one tile.
2. **persistent pair**: move lab 32's persistent + accumulator double buffering onto the pair. `acc_empty` must collect 256 arrivals from the two CTAs, half of them from another SM.
3. **TMA multicast**: grow the cluster to 4 CTAs (two pairs side by side), so the same block of A is read from L2 once and written into two CTAs at the same time; then decide, from the number of "rounds", when this is worth doing.

After this lab you should be able to:

- Draw the data layout of a pair: which half of A and which half of B is in which CTA, and which rows of C land in which TMEM;
- For each barrier, say which CTA it is in, who arrives, who waits, what its count is, and why it must be "drained" before exit;
- Derive how much shared memory traffic and L2 reading per SM the pair saves;
- Explain why a larger cluster is not always faster (a cluster must fit inside one GPC).

FA4 (lab 42) uses the same pair for attention: both QKᵀ and PV are 2-CTA MMAs.

## 2. Motivation: the remaining gap of lab 32

The lab 32 solution runs at about 0.91× cuBLAS on B200. Its NOTES infer that the remaining gap comes from the data supply: for each 128×256 tile and each K tile, the CTA reads A 16 KB + B 32 KB = 48 KB from L2, and the 148 SMs together approach the L2 bandwidth; within a K tile, the Tensor Core also reads the same 48 KB from shared memory.

Can one MMA become "bigger" while each SM moves less data?

- On a single SM, the MMA's M is at most 128 (TMEM has only 128 lanes) and N is at most 256. A larger tile needs more MMAs and more accumulators, and TMEM and shared memory are both already full.
- Blackwell's answer: **two SMs do one M = 256 MMA together**. Each SM still holds only 128 rows of the accumulator, but each needs only half of B (256 columns): B is split between the two SMs, and each holds its own 128 rows of A.

There is no directly matching section in the FlashAttention papers; this is one of the ways FA4 raises attention throughput on Blackwell (FA4's 2-CTA QKᵀ / PV, see lab 42). First, get cluster, pair and cross-CTA synchronization clear on a GEMM.

## 3. Math: what the pair saves

### 3.1 Data per K tile

Let BK = 64 (one 128B swizzle row); BF16 is 2 bytes per element.

| | Output of one "logical MMA group" | A held per SM | B held per SM | shared memory per SM per K tile |
|---|---|---|---|---|
| Lab 32, 1 CTA | 128 × 256 | 128 × 64 = 16 KB | 256 × 64 = 32 KB | 48 KB |
| This lab, pair | 256 × 256 (128 × 256 per SM) | 128 × 64 = 16 KB | 128 × 64 = 16 KB | 32 KB |
| This lab step 3, two pairs share A | 128 × 256 per SM | 16 KB (only 8 KB read from L2) | 16 KB | 32 KB |

Each SM does the same number of multiply-adds per K tile: `2 × 128 × 256 × 64 ≈ 4.2 MFLOP`. So the "FLOP per byte" (arithmetic intensity relative to L2 and shared memory) is:

```
Lab 32    : 4.2 MFLOP / 48 KB ≈ 85 FLOP/B
pair      : 4.2 MFLOP / 32 KB ≈ 128 FLOP/B     (×1.5)
pair + A multicast (L2 side): 4.2 MFLOP / 24 KB ≈ 171 FLOP/B   (×2)
```

Intuitively: in lab 32 each element of B is reused by 128 rows of A; in the pair it is reused by 256 rows of A (two SMs share one copy of B, and each SM stores, and reads from L2, only half of it).

### 3.2 Global L2 → SM traffic

`square` (M = N = K = 4096): along K, each tile reads BM rows of A and BN rows of B:

```
Lab 32: 512 tiles of 128×256 × (128 + 256) rows × 4096 × 2 B ≈ 1.61 GB
pair  : 256 tiles of 256×256 × (256 + 256) rows × 4096 × 2 B ≈ 1.07 GB    (×0.67)
pair + A multicast (256×512 cluster tile):
        128 × (256 + 512) rows × 4096 × 2 B ≈ 0.81 GB                      (×0.5)
```

Measured with NCU's `lts__t_sectors_srcunit_tex_op_read`: step 2 ≈ 0.99 GB (a bit below the derived 1.07 GB), step 3 ≈ 0.82 GB, close to the derivation.

### 3.3 shared memory budget

```
per stage (per CTA)  : A 16 KB + B 16 KB = 32 KB
6 stages             : 192 KB
epilogue staging     : 32 KB (same as lab 32)
total                : 225 KB (limit 227 KB)
```

With the same budget, lab 32 fits only 4 stages of 48 KB; in this lab each stage is smaller, so 6 stages fit and the pipeline is deeper.

### 3.4 Rounds (waves)

In a persistent kernel, each cluster processes tiles one after another; all clusters each computing one tile at the same time is called a "round". Time ≈ number of rounds × time per round.

```
square: 256 pair tiles, 74 pairs resident at once  →  ⌈256 / 74⌉ = 4 rounds (3.46: only 34 pairs compute in the last round, 54% idle)
wide  : 1024 pair tiles                            →  ⌈1024 / 74⌉ = 14 rounds
```

Section 5.5 uses this to choose the cluster shape of step 3.

## 4. Hardware and instructions

### 4.1 thread block cluster

A **cluster** is a group of CTAs that are guaranteed to run at the same time and are placed in the same GPC (Graphics Processing Cluster, a group of SMs). CTAs of the same cluster can:

- access each other's shared memory (**distributed shared memory, DSMEM**);
- synchronize as a whole with `barrier.cluster`;
- arrive on each other's mbarriers, and receive TMA multicasts and `tcgen05.commit` multicasts issued by the other side.

Each CTA has a **rank** in the cluster (`%cluster_ctarank`, 0 .. size−1). With a 1-D grid and a 1-D cluster, the cluster's index is `%clusterid.x` and the total number of clusters is `%nclusterid.x`.

**Launch**. `<<<grid, block>>>` cannot specify a cluster shape at run time; use `cudaLaunchKernelEx`:

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim = grid;            // gridDim.x must be a multiple of cluster_x
cfg.blockDim = block;
cfg.dynamicSmemBytes = smem;
cfg.stream = stream;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim.x = cluster_x;  attr[0].val.clusterDim.y = 1;  attr[0].val.clusterDim.z = 1;
cfg.attrs = attr;  cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, kernel, args...);
```

**Constraints**:

| Constraint | Explanation |
|---|---|
| Size | Portable limit of 8 CTAs; larger needs `cudaFuncAttributeNonPortableClusterSizeAllowed` set as well. This lab uses 2 and 4 |
| Placement | All CTAs of a cluster must fit into the same GPC at the same time; if they do not fit, they wait |
| Resident count | Query with `cudaOccupancyMaxActiveClusters`. For this lab's kernel on B200 (225 KB smem, 1 CTA per SM): cluster 2 → **74** (all 148 SMs used), cluster 4 → **33** (only 132 SMs used) |
| persistent grid | grid = cluster size × min(number of tiles, number of resident clusters); it cannot be computed directly from the SM count |

Cluster 4 loses 16 SMs because the number of usable SMs in each GPC is not a multiple of 4 (the 148 SMs of B200 are spread over several GPCs, and the exact counts vary from chip to chip). CUTLASS uses a **preferred cluster** (`cudaLaunchAttributePreferredClusterDimension`, `examples/73_blackwell_gemm_preferred_cluster`) so that GPCs that cannot fit a large cluster fall back to a small one, but this requires the kernel to handle two cluster sizes and dynamic scheduling. This lab does not do it.

### 4.2 DSMEM addresses and cross-CTA mbarriers

shared memory has two "windows": `.shared::cta` (this CTA) and `.shared::cluster` (the whole cluster). A `.shared::cta` address of this CTA is also a valid `.shared::cluster` address (pointing to itself). To get the address of **the same variable in another CTA**, use `mapa`:

```cpp
// p points to this CTA's shared memory; returns the .shared::cluster address of the same offset in CTA `rank`
uint32_t remote = lab::map_shared_rank(&sm.acc_empty[a], /*rank=*/0);   // mapa.shared::cluster.u32
```

The two CTAs run the same kernel, so the offsets of static `__shared__` and of aligned dynamic shared memory are identical; "the same offset" is therefore "the other CTA's same variable".

Arriving and waiting across CTAs:

```cpp
// arrive on another CTA's barrier (release at cluster scope)
lab::mbar_arrive_cluster(remote);         // mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]
// wait on our own barrier whose arriver is in another CTA (acquire at cluster scope)
lab::mbar_wait_cluster(&bar, parity);     // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64
```

`mbarrier.try_wait` can only wait on a barrier in **its own** CTA. When another CTA needs to know something, that CTA arrives on your barrier; you do not wait on theirs.

**cluster barrier**:

```cpp
lab::cluster_arrive();   // barrier.cluster.arrive.release.aligned
lab::cluster_wait();     // barrier.cluster.wait.acquire.aligned
```

The wait returns only after **all threads of all CTAs** in the cluster have arrived. It replaces the `__syncthreads()` in lab 32's setup / teardown (Section 4.6).

### 4.3 Pair TMA: completion signals go to the leader

The `.cta_group::2` form of the TMA load writes the box into **this CTA's** shared memory, but `complete_tx` can go to an mbarrier of **the other CTA in the pair** (the operand is a `.shared::cluster` address):

```cpp
// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [map, {x, y}], [bar]
lab::tma_load_2d_cg2(sm.a(s), &a_map, lab::map_shared_rank(&sm.full[s], leader), x, y);
```

So the data of both CTAs "reports" to the leader's `full[s]`, and the thread that issues the MMA is in the leader.

### 4.4 Pair MMA: `tcgen05.mma.cta_group::2`

**pair**: the two CTAs of rank 2p and 2p+1 in a cluster; the even one is the **leader**. PTX only requires that **one** thread in the pair issues this MMA; by convention (CUTLASS does the same) one thread of the leader issues it:

```cpp
constexpr uint32_t idesc = lab::make_idesc_f16(/*M=*/256, /*N=*/256, /*bf16=*/true);
lab::mma_f16_ss_cg2(d_tmem, a_desc, b_desc, idesc, accumulate);
// tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, p;
```

Semantics (the descriptors describe only **the leader's own** buffers; the hardware reads the other half at the **same offset** in the peer):

```
                 leader (rank 0)                        peer (rank 1)
smem A    A[m0 +   0 .. 127][k tile]  ← a_desc      A[m0 + 128 .. 255][k tile]  (same offset)
smem B    B[n0 +   0 .. 127][k tile]  ← b_desc      B[n0 + 128 .. 255][k tile]  (same offset)

D = A(256 × 16) · B(256 × 16)ᵀ:
                          ┌──────────── N = 256 columns ─────────┐
TMEM (leader) lane 0..127 │ C[m0 + 0..127,   n0 + 0..255]        │  ← uses B from both sides
TMEM (peer)   lane 0..127 │ C[m0 + 128..255, n0 + 0..255]        │
                          └──────────────────────────────────────┘
```

B is split **in half along N**: the leader's accumulator needs **all** 256 columns of B, and half of them are in the peer's shared memory. This is exactly "two SMs share one copy of B".

**A small example by hand**: the `one_tile` case (M = N = 256, K = 64). C[200][10] belongs to the peer (200 ≥ 128) and is in lane 72, column 10 of the peer's TMEM; it needs row 200 of A (in the peer's smem, row 72) and row 10 of B (in the leader's smem, row 10). C[5][250] is in lane 5, column 250 of the leader's TMEM, and uses row 5 of the leader's A and row 122 of the peer's B.

**All tcgen05 instructions of one kernel must use the same `.cta_group`**. So allocation, deallocation and commit must also use the `cta_group::2` form:

| Instruction | Who executes it | Notes |
|---|---|---|
| `tcgen05.alloc.cta_group::2` | One warp in **each** CTA of the pair (by this lab's convention the same index: warp 5) | Both sides get the same column address |
| `tcgen05.relinquish_alloc_permit.cta_group::2` | Same as above | |
| `tcgen05.mma.cta_group::2` | One thread of the leader | |
| `tcgen05.commit.cta_group::2 ... .multicast::cluster [bar], mask` | The leader thread that issues the MMA | After the MMA completes, arrives once on `bar` at the same offset in each CTA in mask |
| `tcgen05.dealloc.cta_group::2` | That warp in both CTAs | The leader's earlier MMAs must no longer write the peer's TMEM |
| `tcgen05.ld`, `tcgen05.fence` | Each CTA by itself | No cta_group; accesses only this CTA's TMEM |

### 4.5 TMA multicast

A `.multicast::cluster` TMA load reads the box from L2 once and writes it to the same offset in every CTA in `cta_mask`:

```cpp
// ...complete_tx::bytes.multicast::cluster [dst], [map, {x, y}], [bar], cta_mask
lab::tma_load_2d_cg2_multicast(dst, &a_map, bar, /*cta_mask=*/0b0101, x, y);
```

Each copy received by a target CTA does its own `complete_tx`. Combined with `.cta_group::2`, the completion signal of that copy goes to the same offset in **the leader of the pair the target CTA belongs to** (`bar` gives the address of this pair's leader; the hardware translates it for the target pair).

When is this useful? Only when **several CTAs need the same block of data**. Inside one pair, the two halves of A and of B are all different, so there is nothing to share. When two pairs compute `(m0, n0)` and `(m0, n0 + 256)` side by side, they need exactly the same rows of A:

```
cluster of 4 (step 3):  rank 0, 1 = pair 0 → C[m0 .. m0+255, n0       .. n0+255]
                        rank 2, 3 = pair 1 → C[m0 .. m0+255, n0 + 256 .. n0+511]
rank 0 and rank 2 both need A[m0 .. m0+127]; rank 1 and rank 3 both need A[m0+128 .. m0+255]

rank 0: reads A[m0 +  0 .. 63]  multicast → rank 0, 2 (mask 0b0101)
rank 2: reads A[m0 + 64 ..127]  multicast → rank 0, 2
rank 1: reads A[m0 +128 ..191]  multicast → rank 1, 3 (mask 0b1010)
rank 3: reads A[m0 +192 ..255]  multicast → rank 1, 3
```

Each CTA reads only 8 KB of A from L2, but its shared memory still holds the full 16 KB.

### 4.6 Common misuses

| Misuse | Consequence |
|---|---|
| Launch with `<<<grid, block>>>` without the cluster attribute | Each CTA is its own cluster, `%cluster_ctarank` is always 0, and every CTA thinks it is the leader; cta_group::2 instructions have no peer, and the behavior is undefined |
| gridDim.x is not a multiple of the cluster size | Launch fails (a `cudaErrorInvalidValue`-type error) |
| setup uses only `__syncthreads()` | The peer's TMA / the leader's commit may arrive before the other side's `mbarrier.init`: random hangs or errors |
| The peer's TMA uses its own `full[s]` address | The leader never gets these 32 KB: `TIMEOUT`; the peer's `full` count is also messed up |
| The leader's `expect_tx` covers only this CTA's 32 KB | The leader's MMA may start before the peer's data arrives: random errors in the peer's half of the result |
| The peer also executes `arrive_expect_tx` on its own `full[s]` | That barrier never receives `complete_tx`, so its phase never completes; arriving again in the next round is an excess arrival on an incomplete phase (undefined behavior). Only the leader arms |
| Both CTAs issue MMAs | PTX only requires that **one** thread in the pair issues it (either CTA is allowed; "the even rank is the leader" is a convention). If both issue, every K tile is computed twice, and the two issuers also compete on the same TMEM / smem: wrong results |
| tcgen05 instructions mix `cta_group::1` and `::2` | PTX requires one kernel to be consistent: compile or run-time error |
| `commit` without multicast | Only the leader's `empty` / `acc_full` arrive; the peer's producer / epilogue wait forever |
| `acc_empty` count still 128 | It counts as empty once the leader's 128 threads finish reading, and the MMA overwrites an accumulator the peer has not finished reading |
| The peer's epilogue arrives on its own `acc_empty` | The leader's MMA waits on the leader's copy: hang |
| The leader waits on `acc_empty` with `.acquire.cta` (`lab::mbar_wait`) | Does not pair with the peer's `release.cluster`; no memory-ordering guarantee |
| Exit without draining | One CTA has exited, and a commit / arrive from the other CTA lands in reclaimed shared memory (see 5.4) |
| The two CTAs have different shared memory layouts (for example, different sizes allocated by rank) | "The same offset" is no longer the same variable |

## 5. Schedule design

### 5.1 Roles and storage budget (per CTA)

```
block = 192 threads = 6 warps, 1 CTA per SM, cluster = 2 (can be 4 in step 3)

 warp 0..3  epilogue   in both CTAs: read this CTA's TMEM (this CTA's 128 rows of C) → staging → TMA store
 warp 4     producer   in both CTAs: elect one lane; TMA loads this CTA's halves of A and B
 warp 5     MMA        both CTAs tmem_alloc_cg2(512) / dealloc; only one lane of the leader issues MMA and commit

shared memory: 6 stages × 32 KB + 32 KB staging + 1 KB alignment = 225 KB
TMEM         : 2 accumulators × 256 columns = 512 columns (each CTA's own TMEM, 128 lanes each)
```

Who does what (one K tile):

```
            leader (rank 0)                                  peer (rank 1)
producer    wait empty[s]                                    wait empty[s]
            arrive_expect_tx(full[s], 64 KB)
            TMA A half → own smem, complete_tx → L.full[s]   TMA A half → own smem, complete_tx → L.full[s]
            TMA B half → own smem, complete_tx → L.full[s]   TMA B half → own smem, complete_tx → L.full[s]
MMA         wait full[s]; fence::after                       (idle)
            4 × mma.cta_group::2 (reads both smem, writes both TMEM)
            commit.multicast(empty[s], 0b11) ──────────┬──→ P.empty[s]
                                                       └──→ L.empty[s]
```

### 5.2 Barrier map

Each barrier has one copy at the same offset in each of the two CTAs; some barriers use only one of the copies.

| barrier | count | Which copy is used | Who arrives | Who waits | First-round parity |
|---|---|---|---|---|---|
| `full[s]` | 1 | **Only the leader's** | the leader producer's `arrive_expect_tx(64 KB)` + `complete_tx` from both sides' TMA | leader MMA | `phase` (0) |
| `empty[s]` | 1 (step 3: 2) | Both copies | the leader MMA's multicast commit (step 3: once from each of the two leaders) | each CTA's producer | `phase ^ 1` (1) |
| `acc_full[a]` | 1 | Both copies | the leader MMA's multicast commit (both CTAs of this pair) | each CTA's epilogue | `phase` |
| `acc_empty[a]` | **256** | **Only the leader's** | 256 epilogue threads of the two CTAs, `fence::before` + `arrive.release.cluster` | leader MMA, `wait.acquire.cluster` + `fence::after` | `phase ^ 1` |
| cluster barrier | all threads | — | end of setup, teardown | all threads | — |

```
                 leader (rank 0)                                          peer (rank 1)
   ┌──────────────────────────────────────────┐            ┌──────────────────────────────────────────┐
   │ full[s]   ◄── L.producer expect_tx       │◄───────────┼── P's TMA complete_tx                    │
   │           ◄── L's TMA complete_tx        │            │ full[s]   (unused)                       │
   │ empty[s]  ◄─┐                            │            │ empty[s]  ◄─┐                            │
   │             ├─ L.MMA commit multicast ───┼────────────┼─────────────┘                            │
   │ acc_full  ◄─┘  (0b11)                    │            │ acc_full  ◄── the same commit            │
   │ acc_empty ◄── L.epilogue × 128           │◄───────────┼── P.epilogue × 128                       │
   │                                          │            │   (mapa address, release.cluster)        │
   └──────────────────────────────────────────┘            └──────────────────────────────────────────┘
```

### 5.3 Step 2: persistent pair

The structure is the same as lab 32 step 2; the tile becomes 256 × 256, and scheduling is per cluster:

```
grid = 2 × min(number of tiles, number of resident clusters); cluster c processes tiles c, c + %nclusterid.x, ...
both CTAs walk the same sequence of tiles with the same number of K tiles: RingPos agrees without communication

producer (both): ring continuous across tiles; for each tile: produce_tile (this CTA's two halves); finally producer_drain
MMA (leader)   : for each tile:
                   mbar_wait_cluster(acc_empty[acc], acc.phase ^ 1); fence::after
                   mma_tile → accumulator acc; commit multicast acc_full[acc]; acc.advance
                 at the end, wait on acc_empty kAccStages more times (drain)
epilogue (both): for each tile:
                   wait on this CTA's acc_full[acc]; fence::after
                   epilogue_tile: after the last ld, fence::before + arrive_cluster(the leader's acc_empty[acc])
```

```
One cluster (3 tiles)
L.producer : │t0 k0..│t1 k0..             │t2 ...
P.producer : │t0 k0..│t1 k0..             │t2 ...         (each waits on its own empty, advancing at the pace of the multicast)
L.MMA      :  │══ t0 → acc0 ══│══ t1 → acc1 ══│┄ wait acc_empty[0] ┄│══ t2 → acc0 ══│
L.epilogue :                  │ld acc0 ·························│store│ld acc1 → store│
P.epilogue :                  │ld acc0 ·····························│store│ld acc1 → store│
                                                                    ↑ the later CTA's last ld of acc0 completes the 256 arrivals on L's acc_empty[0]
                                    the epilogues on both sides each read their own CTA's TMEM, at the same time
```

### 5.4 Startup and exit: why "drain"

**Startup**:

```
thread 0     : mbar_init(...); fence.mbarrier_init.release.cluster
warp 5       : tmem_alloc_cg2(slot, 512); relinquish (both CTAs do this)
all threads  : tcgen05.fence::before; cluster_sync(); tcgen05.fence::after; read slot
```

Why it must be a cluster barrier: the peer's TMA does `complete_tx` on the leader's `full`, and the leader's commit arrives on the peer's `empty`. If one CTA starts signaling before the other has initialized, the result is undefined.

**Exit**: after a CTA exits, its shared memory and TMEM can be handed to the next CTA immediately. You must guarantee that after this point there is **no access at all from another CTA**:

| Cross-CTA access | How to guarantee it has ended |
|---|---|
| The leader's MMA reads the peer's smem and writes the peer's TMEM | The peer's epilogue has waited for the last `acc_full`: all MMAs have completed |
| The leader's multicast commit arrives on the peer's (and its own) `empty[s]` | **Producer drain**: after the last tile, "acquire" kStages more slots (wait on `empty`, load no data). Every `empty` has then received its last commit |
| The peer's epilogue arrives remotely on the leader's `acc_empty` | **Leader MMA drain**: wait on `acc_empty` kAccStages more times |
| The peer's TMA does `complete_tx` on the leader's `full` | The leader's MMA has waited on every `full` |
| (step 3) The other pair's multicast writes this CTA's A | This CTA's `empty[s]` waits for the commits of both leaders |

Then:

```
all threads : tcgen05.fence::before; cluster_sync()
warp 5      : tcgen05.fence::after; tmem_dealloc_cg2(tmem, 512)
```

In lab 32, not draining was fine: a commit only arrives on this CTA's own barriers, and by the time the CTA has waited on `acc_full` those MMAs have all completed. In this lab, commits land on **another SM**. The CUTLASS counterparts are `producer_tail` and `tmem_deallocation_result_barrier`.

### 5.5 Step 3: multicast of A and the cluster shape

Cluster 4 = two pairs side by side along N, a 256 × 512 cluster tile. Differences from step 2:

| | Step 2 | Step 3 |
|---|---|---|
| Box of A's tensor map | 128 rows | 64 rows (each CTA reads half, multicast) |
| count of `empty[s]` | 1 | 2: this CTA's A is written jointly by two pairs, and can be rewritten only after the MMAs of both pairs have finished reading it |
| mask of the `empty` commit | 0b11 | 0b1111 |
| mask of the `acc_full` commit | 0b11 | this pair: `0b11 << 2·pair` |
| `acc_empty` address | rank 0 | this pair's leader (rank 0 or 2) |

The two pairs share `empty`, so they move in lockstep: if one pair is slow, the other has to wait for it.

**The number of rounds decides whether it pays off** (Section 3.4; for resident counts see Section 4.1): with both shapes, each pair computes one 256×256 tile per round, and time ≈ number of rounds × time per round.

| case | cluster 2: tiles / 74 | cluster 4: tiles / 33 | Choice |
|---|---|---|---|
| `square` | 256 / 74 → 4 rounds | 128 / 33 → 4 rounds | cluster 4 (same rounds, fewer A reads) |
| `wide`, `flat` | 1024 / 74 → 14 rounds | 512 / 33 → 16 rounds | cluster 2 |

So step 3 of `run()` computes these two round counts on the host and uses the cluster 4 multicast kernel when `rounds3 ≤ rounds2` (rounds3 is the round count of step 3's cluster 4 shape), and the step 2 kernel otherwise.

## 6. Step-by-step exercises

### Step 1 · TODO(1a)–(1h): cluster and pair MMA

`./lab check 33` / `./lab bench 33` without `--step` run the last step (step 3) and hit TODO(3c) on the host first; while working on steps 1 and 2, always add `--step 1` / `--step 2`.

- TODO(1a): cluster identity, cluster barrier, `mapa` (Sections 4.1, 4.2);
- TODO(1b): the pair versions of TMA, TMEM alloc/dealloc, MMA, multicast commit (Sections 4.3, 4.4);
- TODO(1c): the host's `launch_cluster` (the `cudaLaunchKernelEx` of Section 4.1);
- TODO(1d): `mma_k_tile_pair`: what are M and N in idesc?
- TODO(1e): `produce_tile`: who arms `full[s]`, with how many bytes, and where the TMA completion signals go;
- TODO(1f): `producer_drain` (Section 5.4);
- TODO(1g): `mma_tile`: commit with multicast;
- TODO(1h): setup / teardown synchronization (Section 5.4).

The `gemm_pair_tile` kernel itself (role dispatch, which rows the epilogue writes) is already written; `epilogue_tile` is the same as in lab 32.

```bash
./lab check 33 --step 1 --case one_tile
./lab check 33 --step 1 --case leader_only
./lab check 33 --step 1
./lab bench 33 --step 1 --case square
./lab sass 33          # UTCHMMA.2CTA, UTCBAR.2CTA.MULTICAST, UTMALDG.2D.2CTA, UTCATOMSWS.2CTA; no spill
```

### Step 2 · TODO(2a)–(2d): persistent pair

- TODO(2a): `mbar_arrive_cluster`, `mbar_wait_cluster` (Section 4.2);
- TODO(2b): persistent MMA: which kind of wait to use on `acc_empty`, and the drain at the end;
- TODO(2c): the release inside `epilogue_tile`: fence + remote arrive;
- TODO(2d): persistent epilogue: `acc_free` is which barrier in which CTA?

```bash
./lab check 33 --step 2 --case persistent
./lab check 33 --step 2
./lab bench 33 --step 2 --case wide
./lab bench 33 --step 2 --case flat
```

### Step 3 · TODO(3a)–(3c): multicast and cluster shape

- TODO(3a): `tma_load_2d_cg2_multicast`;
- TODO(3b): the multicast branch of `load_a` (the table of Section 4.5);
- TODO(3c): choosing the cluster shape by round count in `run()` (Section 5.5).

```bash
./lab check 33 --step 3 --case one_tile
./lab check 33
./lab bench 33 --step 3 --case square
./lab profile 33 --step 3 --case square    # look at lts__t_sectors_srcunit_tex_op_read.sum, compare with step 2
```

All correctness cases of step 3 have very few rounds, so they all take cluster 4.

### Common mistakes

| Symptom | Likely cause |
|---|---|
| Step 1 launch error | gridDim is not a multiple of the cluster size; `clusterDim.y/z` not set to 1; `numAttrs` is 0 |
| `TIMEOUT`, `one_tile` | The peer's TMA signals go to its own `full`; the leader's expect_tx byte count is not 64 KB; the commit is not multicast to the peer; the rank in `mapa` is its own |
| Half of the rows of `one_tile` wrong | M in idesc is still 128; B's box is still 256 rows (it should be 128); the peer's starting row of B is not `n0 + 128` |
| `one_tile` correct, `long_k` has sporadic errors | The leader expects only this CTA's 32 KB; the peer also executed `arrive_expect_tx` |
| `leader_only` hangs | The peer thinks its A and B are entirely out of bounds and skips the TMA: out-of-bounds boxes must also be issued, and zero fill still counts toward the bytes |
| Step 2 `TIMEOUT` | `acc_empty` count is not 256; the peer's epilogue arrives on its own `acc_empty` |
| Step 2 sporadic errors | The leader waits on `acc_empty` with cta scope; `fence::before_thread_sync` missing before arrive |
| Sporadic "illegal address" or an error at the next launch | No drain before exit (Section 5.4), or no cluster synchronization in teardown |
| Step 3 hangs | `empty` count still 1, or the commit mask does not include the other pair; wrong multicast mask bits |
| Step 3 correct but `wide` slower | The cluster shape is not chosen by round count (Section 5.5) |

## 7. Performance

Measured (B200, CUDA Graph median; larger `vs cublas` is better):

| Version | `square` 4096³ | `wide` 8192×8192×4096 | `flat` 8192×8192×1024 |
|---|---|---|---|
| Lab 32 solution (same GPU) | ≈ 0.91× | ≈ 0.91× | ≈ 0.99× |
| Step 1: pair, one tile per cluster | ≈ 0.91× | ≈ 0.86× | ≈ 0.73× |
| Step 2: persistent pair | ≈ 0.98× | ≈ 1.12× | ≈ 1.09× |
| Step 3: shape chosen by round count | ≈ 1.00× | ≈ 1.16× (takes cluster 2) | ≈ 1.03–1.05× (takes cluster 2) |
| Control: forced cluster 4 multicast | ≈ 1.00× | about 8% slower than step 2 | about 5% slower than step 2 |
| Step 3, `--cold` (L2 flushed before each run) | ≈ 1.01× | ≈ 1.05× | ≈ 1.10× |

cuBLAS on `wide` varies a lot (0.39–0.43 ms on the same GPU), so the ratio moves between 1.0 and 1.16 with it; the time of this lab relative to lab 32 is more stable: step 2 takes about 0.83–0.94 times the time of lab 32 on all three cases. The perf gate is 0.9× cuBLAS on all perf cases.

**Reading this table**:

- **Step 1 is not faster than lab 32**: one cluster computes only one tile, so every tile pays for cluster launch, init, alloc and pipeline fill, and the epilogue does not overlap the main loop; this is most visible on `flat` (where the epilogue's share is largest). `sm__pipe_tensor_cycles_active` on `square` is about 76% (about 85% in lab 32).
- **Only step 2 shows the value of the pair**: Tensor Core activity is about 88%, each CTA reads one third less per K tile (Section 3.1), and the pipeline has 6 stages. On `wide` it is even faster than cuBLAS.
- **Step 3**: `lts__t_sectors_srcunit_tex_op_read` drops from about 0.99 GB to about 0.82 GB (Section 3.2). But only 33 cluster-4 clusters can be resident (132 SMs), which adds 2 rounds on `wide` and `flat`, so forcing it is slower; with the round-count choice, `square` gets the ~1.5% from multicast and the other cases keep the step 2 speed. With clocks locked in NCU, cluster 4 is actually slightly slower on `square` (about 2%): these differences depend on power and clock frequency, not only on L2 traffic.

**Metrics to look at** (`./lab profile 33 --step n --case square`):

| Metric | Meaning |
|---|---|
| `sm__pipe_tensor_cycles_active` | Fraction of time the Tensor Core is busy: the main change from step 1 → 2 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | Sectors (32 B) the SMs read from L2: the change from step 2 → 3 |
| `dram__bytes_read.sum` | HBM reads: look at grouped rasterization on `wide` |

## 8. Self-check questions

1. In the pair MMA, which row of B in which CTA does the leader's accumulator element C[m0 + 3][n0 + 200] use?
2. Why does the leader's `full[s]` expect 64 KB, while the peer's `full[s]` is not used at all?
3. Why is `acc_empty` used only in the leader, and why is its count 256? Why must the leader's wait use `acquire.cluster`?
4. Lab 32 does not need to drain `empty` before exit; why must this lab? What goes wrong if teardown only does the cluster synchronization without draining?
5. Why is the count of `empty[s]` 2 in step 3? What happens if it stays 1?
6. If a GPU could keep 37 cluster-4 clusters resident (all 148 SMs), which shape should `wide` choose?

<details>
<summary>Answers</summary>

1. B is split in half along N: column 200 of n is ≥ 128, so it is in the peer's shared memory; row n0 + 200 of B is stored in row 72 of the peer's B buffer.
2. The MMA is issued by a leader thread, which can only wait on its own CTA's barriers. The TMA of both CTAs sends completion signals to the leader's `full[s]`, so it must wait for 2 × 32 KB from both CTAs. The peer has no MMA thread, so nobody waits on its `full`.
3. The thread that issues the MMA is in the leader, and it can only wait on the leader's barriers. Each of the two CTAs has 128 epilogue threads, each reading this CTA's 128 rows; the accumulator is empty only after all of them have finished reading. The peer's arrival is `release.cluster` and must pair with `acquire.cluster` to guarantee that the peer's TMEM reads (ordered by the fence) happen before the leader's next MMA write.
4. In lab 32, a commit only arrives on this CTA's barriers. In this lab, the leader's commit arrives on the peer's `empty`, and the peer's epilogue arrives on the leader's `acc_empty`. The last few commits/arrives have no waiter and may land after the other CTA has exited, writing into shared memory that has been reclaimed (or even given to a new CTA). The cluster synchronization orders only ordinary thread operations; it does not wait for asynchronous commits to arrive.
5. The `empty[s]` of CTA X gates X's multicast, which writes the A buffers of both X and X ± 2 (the CTA at the same position in the other pair). These two copies are read by the MMAs of the two pairs respectively, so they can be rewritten only after both leaders have committed. With count 1, the commit of whichever pair finishes reading first lets it through while the other pair may still be reading: sporadic errors; the extra arrival also advances the next phase early.
6. Cluster 4: 512 / 37 → 14 rounds, the same as the 14 rounds of cluster 2, so there is no loss in rounds and fewer A reads. The round-count comparison in this lab's `run()` picks it automatically.
</details>

## 9. Further reading

- FlashAttention-4 / Blackwell: the FA4 write-ups by Tri Dao and others; 2-CTA QKᵀ and PV (lab 42).
- PTX ISA 8.7:
  [Thread Block Clusters and Distributed Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays),
  [barrier.cluster](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster),
  [mapa](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mapa),
  [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier),
  [cp.async.bulk.tensor (.cta_group, .multicast::cluster)](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor),
  [tcgen05.mma (cta_group::2)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma),
  [tcgen05.alloc](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions),
  [tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit).
- CUDA Programming Guide: [Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters), [Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory); Runtime API `cudaLaunchKernelEx`, `cudaOccupancyMaxActiveClusters`.
- Upstream sources (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp`: `SM100_MMA_F16BF16_2x1SM_SS` (the PTX form of the pair MMA);
  - `third_party/cutlass/include/cute/arch/copy_sm100_tma.hpp`: `SM100_TMA_2SM_LOAD_2D`, `SM100_TMA_2SM_LOAD_MULTICAST_2D` (`Sm100MmaPeerBitMask`);
  - `third_party/cutlass/include/cute/arch/tmem_allocator_sm100.hpp`: `Allocator2Sm`;
  - `third_party/cutlass/include/cutlass/arch/barrier.h`: `umma_arrive_multicast_2x1SM`, `ClusterBarrier`;
  - `third_party/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp`: `PipelineTmaUmmaAsync` (only the leader does `arrive_and_expect_tx`; the multicast mask computation);
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`: `is_mma_leader_cta`, `tmem_deallocation_result_barrier`, `producer_tail`;
  - `third_party/cutlass/examples/70_blackwell_gemm`, `73_blackwell_gemm_preferred_cluster`.
