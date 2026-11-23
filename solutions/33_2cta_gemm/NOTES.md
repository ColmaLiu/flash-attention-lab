# 33_2cta_gemm solution notes

## Design choices

- **Supported domain**: the same as labs 31 and 32. A and B are BF16, C is FP32; `K % 8 == 0`, `N % 4 == 0`, `K > 0` (with K = 0 the kernel would commit `acc_full` without any MMA and read uninitialized TMEM); M arbitrary; all tails are left to TMA. All three steps cover all shapes, so every step runs all correctness cases.
- **Primitives**: new `include/lab/cluster.cuh` (students implement it in this lab's skeleton; it is provided from lab 34 on): cluster identity (`%cluster_ctarank`, `%clusterid.x`, `%nclusterid.x`), `barrier.cluster`, `mapa`, cluster-scope mbarrier arrive / wait, the `.cta_group::2` TMA load (including multicast), TMEM alloc/relinquish/dealloc, `tcgen05.mma`, multicast `tcgen05.commit`, plus the host-side `launch_cluster` (`cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`) and `max_active_clusters` (`cudaOccupancyMaxActiveClusters`). `tcgen05.cuh` / `tma.cuh` are unchanged: all tcgen05 instructions in a kernel must use the same `.cta_group`, so the pair versions of alloc/mma/commit live in `cluster.cuh`, while `tcgen05.ld`, fences and descriptors still come from `tcgen05.cuh`.
- **Pair layout**: rank 0 of the cluster is the leader and rank 1 is the peer. CTA r uses TMA to read 128 rows of A starting at row `m0 + 128r` and 128 rows of B starting at row `n0 + 128r` (B box of 128 rows), placed at the same shared memory offsets. One leader thread issues `tcgen05.mma.cta_group::2` with idesc M = 256, N = 256; the descriptors point only to the leader's own buffers, and the hardware reads the other half at the same offset in the peer. The accumulator takes 128 lanes × 256 columns in the TMEM of each of the two CTAs, and the epilogue of CTA r writes the 128 rows of C starting at row `m0 + 128r`, all 256 columns.
- **barriers** (one copy at the same offset in each of the two CTAs):
  - `full[s]` (count 1): only the leader's is used. The leader's producer does `arrive_expect_tx(64 KB)` (the bytes of both CTAs), and the TMAs of both CTAs `complete_tx` on the leader address obtained with `mapa`. The peer's TMA may arrive before the leader's expect_tx, making the tx-count temporarily negative; this is allowed. The phase cannot complete until the leader's 1 arrival.
  - `empty[s]` (count = number of pairs per cluster): the leader's MMA thread does `tcgen05.commit.cta_group::2 ... multicast` to all CTAs in the cluster; each CTA's producer waits only on its own copy.
  - `acc_full[a]` (count 1): the leader commits with multicast to both CTAs of this pair; each epilogue waits on its own copy.
  - `acc_empty[a]` (count 256): only the leader's is used. The 128 epilogue threads of each of the two CTAs do `fence::before_thread_sync` + `mbarrier.arrive.release.cluster` after their last `tcgen05.ld` (the peer through the `mapa` address); the leader waits with `try_wait.acquire.cluster`, then `fence::after_thread_sync`.
- **Startup and exit** (problems pointed out in the maintainer's review; this lab must handle them):
  - setup: thread 0 initializes the barriers + `fence.mbarrier_init.release.cluster`; warp 5 of both CTAs executes `tcgen05.alloc.cta_group::2` (512 columns; both sides get the same column address); `fence::before` + `barrier.cluster.arrive.release/wait.acquire` + `fence::after`. `__syncthreads` is not enough: the peer's TMA and the leader's commit write the other side's barriers, which must happen after the other side has initialized them.
  - Drain: after the last tile, the producers of both CTAs "acquire" kStages more slots (wait on `empty`, load no data), which guarantees that all multicast commits the leader sent to this CTA have arrived; the leader's MMA thread likewise waits on `acc_empty` kAccStages more times, which guarantees that all of the peer's remote arrives have arrived.
  - teardown: `fence::before` + cluster barrier + `fence::after`, then each side does `tcgen05.dealloc.cta_group::2`. At this point all of the leader's MMAs have completed (both epilogues have waited on the last `acc_full`), and no CTA's smem / TMEM is accessed by the other side any more.
- **Resources**: 32 KB per stage per CTA (A 16 KB + B 16 KB; lab 32 has 48 KB), 6 stages 192 KB + 32 KB staging + 1 KB alignment = 225 KB; TMEM 2 × 256 columns = 512 columns. 56–72 registers, no spill.
- **step 1** (`gemm_pair_tile`): grid = 2 × number of tiles, cluster 2, one 256×256 tile per cluster, single accumulator. **step 2** (`gemm_pair_persistent<Config<6,2,8>>`): grid = 2 × min(number of tiles, number of resident clusters), static round-robin over `%clusterid.x` / `%nclusterid.x`, accumulator double buffering, grouped rasterization G = 8 (in units of tile rows of 256 rows). **step 3**: cluster 4 = two pairs side by side along N (cluster tile 256×512). For each half of A, the two CTAs at the same position in the two pairs each read 64 rows and multicast them to each other (mask = bit r | bit r+2); `empty` count = 2 (the commits of the two leaders); `acc_full` is multicast only to this pair. The host chooses between cluster 4 and cluster 2 by the number of "rounds".
- **Barrier address of a multicast**: a `.cta_group::2` + `.multicast::cluster` TMA takes only one barrier operand (the `mapa` address of this CTA's leader), and each target CTA's copy does complete_tx at the same offset in "the leader of the pair the target CTA belongs to". This was confirmed by measurement on `persistent` and other cases (if the signal lands on the wrong leader, one leader receives extra bytes and the other never gets enough, i.e. `TIMEOUT`), and it matches the way CUTLASS `SM100_TMA_2SM_LOAD_MULTICAST` clears the peer bit.

## Performance per step (CUDA Graph median, relative to cuBLAS BF16→FP32)

| Version | square 4096³ | wide 8192×8192×4096 | flat 8192×8192×1024 |
|---|---|---|---|
| Lab 32 solution (128×256, single CTA) | ≈ 0.91× | ≈ 0.91× | ≈ 0.99× |
| step 1: pair tile, not persistent | ≈ 0.91× | ≈ 0.86× | ≈ 0.73× |
| step 2: persistent pair + double buffering + G = 8 | ≈ 0.98–0.99× | ≈ 1.12× | ≈ 1.09× |
| step 3: chosen by rounds (square takes cluster 4 multicast, wide/flat take cluster 2) | ≈ 1.00× | ≈ 1.16× | ≈ 1.03–1.05× |
| Control: forced cluster 4 multicast | ≈ 1.00× | ≈ 8% slower than step 2 | ≈ 5% slower than step 2 |
| step 3, `--cold` | ≈ 1.01× | ≈ 1.05× | ≈ 1.10× |

- cuBLAS on `wide` varies a lot (0.39–0.43 ms over several runs on the same GPU), so the ratio moves between 1.0 and 1.16 with it; this solution's own time is very stable (steps 2/3 take about 0.83 times the time of the lab 32 solution).
- `selftest --perf` (perf gate 0.9): square 1.00, wide 1.16, flat 1.05. The worst, square, leaves about 10% margin.
- NCU (clocks locked, so only ratios matter), square: `sm__pipe_tensor_cycles_active` lab 32 ≈ 84.5%, step 1 ≈ 75.7%, step 2 ≈ 87.7%, step 3 ≈ 88.7%. `lts__t_sectors_srcunit_tex_op_read`: step 2 ≈ 0.99 GB, step 3 ≈ 0.82 GB; the derived values are lab 32 1.61 GB, pair 1.07 GB, pair + A multicast 0.81 GB. With locked clocks step 3 is actually about 2% slower than step 2, and without locked clocks about 1.5% faster: the difference comes mainly from the round count and power/frequency, not from L2 bandwidth.
- **Why step 3 chooses the shape on the host**: `cudaOccupancyMaxActiveClusters` reports that on B200 74 cluster-2 clusters can be resident (all 148 SMs used), but only 33 cluster-4 clusters (132 SMs): a cluster must fit inside one GPC, and the SM count of a GPC is not a multiple of 4. With both shapes each pair computes one 256×256 tile per round, so time ≈ number of rounds × time per round. square: 256 pair tiles / 74 = 3.46 → 4 rounds; 128 cluster tiles / 33 = 3.88 → 4 rounds; the round counts are equal, and multicast reads less A, so it is slightly faster. wide/flat: 1024 / 74 → 14 rounds, 512 / 33 → 16 rounds, and cluster 4 is about 8% slower (the measurement differs from the 14% of 16/14; multicast offsets part of it). CUTLASS's solution is the preferred cluster (`cudaLaunchAttributePreferredClusterDimension`, examples/73), which lets GPCs that cannot fit 4 use 2; it needs dynamic scheduling and is outside the scope of this course.
- step 1 is worse than lab 32's step 1 on wide/flat: each cluster computes only one tile, so 2048/1024 clusters repeatedly launch, build the pipeline and do TMEM alloc, and the epilogue does not overlap (the epilogue's share is largest on flat). The advantage of the pair shows only after persistent.

## Verification

- `./lab check 33 --solution`: 3 steps × 9 cases pass (`int` cases exactly equal, `random` max error 1.8e-4). All correctness cases of step 3 take cluster 4 (none of them has more rounds than with cluster 2); the `persistent` case makes some clusters process 3 tiles; step 2's `persistent` covers the persistent path of cluster 2.
- compute-sanitizer:
  - memcheck, synccheck: 0 errors on `tail` for all 3 steps and on `persistent` for steps 2/3.
  - racecheck: every run reports 2 hazards, all of them "a write at an unknown PC" against "a read at the `tmem_alloc_cg2` line". From the SASS: the compiler expands `tcgen05.alloc.cta_group::2` into a software protocol (`UTCATOMSWS.2CTA.FIND_AND_SET`, a `SYNCS` mbarrier in the reserved shared memory 0x40–0x7f, remote `STAS` writes to the peer), and racecheck does not understand cross-CTA mbarrier synchronization. A minimal kernel that only does alloc/dealloc (148 CTAs, cluster 2, warp 5 alloc) reports the same hazard, so it is unrelated to this lab's code; lab 32 with cta_group::1 does not have it. Apart from this, racecheck reports nothing. racecheck itself does not track TMA, tcgen05, multicast or DSMEM remote arrives (async proxy / remote accesses); their ordering is reasoned through in the table below:

| Access | Before | After | Guarantee |
|---|---|---|---|
| barrier init → use by another CTA or the async proxy | `mbarrier.init` | the peer's TMA complete_tx, the leader's commit, remote arrives | `fence.mbarrier_init.release.cluster` + `barrier.cluster` |
| TMA writes stage s of the leader / peer → pair MMA reads | `complete_tx` from both sides | MMA issued | wait on the leader's `full[s]` (64 KB) + `fence::after` |
| MMA reads stage s on both sides → TMA on both sides rewrites | MMA completes | next-round TMA | multicast commit → each CTA's `empty[s]`, each producer waits |
| MMA writes accumulator a on both sides → each epilogue reads | MMA completes | `tcgen05.ld` | multicast commit → each CTA's `acc_full[a]`, wait + `fence::after` |
| epilogues on both sides read accumulator a → MMA rewrites | `wait::ld` | the next tile's MMA | `fence::before` + 256 `arrive.release.cluster` → leader `acc_empty[a]`, `try_wait.acquire.cluster` + `fence::after` |
| (step 3) the other pair's multicast writes this CTA's A → rewrite | the MMAs of both pairs finished reading | next-round multicast | `empty[s]` count 2: the commits of both leaders |
| leader commit → this CTA's `empty` / `acc_full` | arrival | CTA exits | producer drains kStages times; the epilogue has waited on every `acc_full` |
| peer remote arrive → the leader's `acc_empty` | arrival | leader exits | leader MMA drains kAccStages times |
| all TMEM accesses → dealloc | MMA completes, `wait::ld` | `dealloc.cta_group::2` | `fence::before` + `barrier.cluster` + `fence::after` |
| generic write to staging → TMA store; store finished reading → rewrite / exit | | | Same as lab 32 |

- `./lab sass 33 --solution`: 56 / 72 / 72 registers, no spill, no local memory; `UTCHMMA.2CTA`, `UTCBAR.2CTA.MULTICAST`, `UTMALDG.2D.2CTA`, `UTMALDG.2D.MULTICAST.2CTA` (step 3) and `UTCATOMSWS.2CTA` (alloc) are all present.
- Skeleton: the solution code was filled in place group by group for TODO(1), (2), (3); `--step n` passes in order, and step n depends only on the first n groups of TODOs. After filling, the student version runs at the same speed as the solution; the skeleton has been restored byte for byte. When unfilled: steps 1/2 report an error at the host's `launch_cluster` (TODO(1)); with only the launch implemented, both CTAs trap at the setup TODO(1) (`tmem_alloc_cg2` / `cluster_arrive`, before any wait); with step 1 complete, step 2 traps at the leader MMA's TODO(2) (reported on the GPU, no hang); step 3 reports an error at the host's cluster-shape choice TODO(3).
- `./lab selftest 33 --perf` passes; `pytest` passes (after the lesson was written).

## Known limitations

- Static round-robin scheduling; the 3.46-wave tail of square is not handled. Cluster 4 can only be used when it does not cost extra rounds; preferred cluster / dynamic scheduling (CLC) is not implemented.
- `kGroupM = 8` reuses the conclusion of lab 32 (with the unit changed to tile rows of 256 rows); it was not re-swept for the pair.
- Multicast is used only for A (two pairs side by side along N); the symmetric shape that shares B along M is not implemented.
- BF16 → FP32 only.
