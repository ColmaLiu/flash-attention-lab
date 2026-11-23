# 32_ws_gemm solution notes

## Design choices

- **Supported domain**: the same as lab 31. A and B are BF16, C is FP32; `K % 8 == 0`, `N % 4 == 0`, M arbitrary; all tails are left to TMA (loads fill zeros, stores clip). All three steps cover all shapes, so every step runs all correctness cases.
- **Primitives**: only `include/lab/{mbarrier,tma,tcgen05}.cuh`. The epilogue uses `tmem_ld_32x32b_x32_sync` (ld and `wait::ld` are in the same asm block, so the registers are valid on return). No local PTX wrappers were added.
- **Roles**: 6 warps (192 threads). Warps 0–3 run the epilogue: warp w can only access TMEM lanes `32(w%4)..`; taking w = 0..3 matches lab 31's "warp w ↔ C from row 32w". Warp 4 is the TMA producer; warp 5 is the MMA issuer and also handles TMEM alloc/dealloc (as in CUTLASS, the MMA warp owns TMEM). The producer and the MMA each use `elect_one()` to pick one lane that runs the whole loop; the other lanes wait at `__syncwarp`. CUTLASS (`sm100_gemm_tma_warpspecialized.hpp`) has one more scheduler warp and one epilogue-load warp; static scheduling does not need the former, and C is only written, never read, so the latter is not needed either.
- **barriers**: `full[s]` (1 arrival + tx), `empty[s]` (1, tcgen05.commit), `acc_full[a]` (1, tcgen05.commit), `acc_empty[a]` (128: each epilogue thread does `fence::before_thread_sync` + arrive after its own last `tcgen05.ld` completes; the same as CUTLASS `consumer_arv_count = NumEpilogueThreads`). The producer side (the producer waiting on `empty`, the MMA waiting on `acc_empty`) uses `phase ^ 1` in the first round, relying on "the previous phase of a freshly initialized barrier counts as completed".
- **Phases across tiles**: each role holds its own `RingPos` (index, phase) and advances it once per use, without resetting it across tiles. The three roles walk the same static tile sequence and use the same number of slots, so they stay consistent without communicating.
- **`empty` arrivals that nobody waits for**: the commits to `empty[s]` for the last few K tiles of a CTA's last tile have no waiter. With a single CTA this is harmless (they only arrive on this CTA's barriers, and `acc_full` has already confirmed that the MMAs completed). With clusters in lab 33, CUTLASS drains them with `producer_tail` before letting the CTA exit.
- **`K = 0`**: rejected by make_plan (otherwise the MMA warp commits `acc_full` directly, and the epilogue writes uninitialized TMEM).
- **barrier storage**: `Barriers<Cfg>` (all barriers + the TMEM address slot) is a static `__shared__` variable declared explicitly in each kernel; `carve()` splits the dynamic shared memory into stages and staging. No macros.
- **Early accumulator release**: the epilogue arrives on `acc_empty` right after it finishes reading TMEM, while the TMA store is still in progress. This is safe because the store reads staging (shared memory), not TMEM.
- **epilogue staging**: in a persistent kernel the main loop uses the stage buffers at the same time, so they cannot be reused as in lab 31. Each epilogue warp has two 32×32 FP32 boxes (4 KB, 128B swizzle), 32 KB in total; each box is one bulk group. Before rewriting a box, lane 0 does `bulk_wait_group_read<1>` (the store from two rounds ago has finished reading), then `__syncwarp`.
- **Resources**: BN = 256, 4 stages (192 KB) + 32 KB staging + 1 KB alignment = 225 KB of dynamic shared memory, plus about 1.1 KB static, within the 227 KB limit; TMEM 2 × 256 = 512 columns, all of TMEM. One CTA per SM, so the 512-column alloc never competes with other CTAs of this kernel. Neither 5 stages nor 3 accumulators fit.
- **step 1** uses a separate `gemm_ws_tile` (grid = all tiles, single accumulator, the same tile and stage count as lab 31 step 3) and only changes "who issues TMA, who issues MMA, who runs the epilogue". **steps 2/3** use `gemm_ws_persistent`, grid = min(number of tiles, number of SMs), tiles `blockIdx.x + i·gridDim.x`. Step 3 only changes the tile order (grouped rasterization, `kGroupM = 8`).
- **No `setmaxnreg`**: 56–72 registers, no spill; each SM has only 6 warps, the register file is far from full, and redistributing gives no benefit. It is left to FA4 (lab 40), where the softmax warpgroups need many registers.

## Performance per step (CUDA Graph, relative to cuBLAS BF16→FP32)

| Version | square 4096³ | wide 8192×8192×4096 | flat 8192×8192×1024 |
|---|---|---|---|
| Lab 31 solution (single-thread TMA+MMA, not persistent) | ≈ 0.84× | ≈ 0.75× | — |
| step 1: role split, one tile per CTA | ≈ 0.89× | ≈ 0.80× | ≈ 0.78× |
| Control: persistent, single accumulator (`Config<256,4,1,0>`) | ≈ 0.92× | ≈ 0.85× | ≈ 0.88–0.9× |
| step 2: persistent + accumulator double buffering | ≈ 0.93× | ≈ 0.86–0.88× | ≈ 0.95–0.98× |
| step 3: + grouped rasterization G = 8 | ≈ 0.92× | ≈ 0.91× | ≈ 0.95× |

Group size (step 3, change in absolute time relative to step 2, at most 3 repetitions):

| G | square | wide | flat |
|---|---|---|---|
| 0 (row-major = step 2) | 1 | 1 | 1 |
| 8 | −1.5% | +6% | ≈ −1–2% |
| 16 | −5% | +8% | −4% |
| 32 | −3% | +10% | −4% |

- NCU (clocks locked, so only ratios matter): `sm__pipe_tensor_cycles_active` on square is ≈ 80% for step 1 and ≈ 85% for steps 2/3 (lab 31 step 3 ≈ 68%). `dram__bytes_read` on wide: row-major ≈ 981 MB, G = 8 ≈ 507 MB, G = 32 ≈ 280 MB (A + B itself is 128 MB).
- Accumulator double buffering by itself is worth only about 1% on square and about 10% on flat: with K = 4096 the main loop of a tile is about 6–8 times the epilogue, and the single accumulator is also "released as soon as TMEM is read", so the MMA only waits for the TMEM→register part. With K = 1024 the epilogue and the main loop are of the same order, and only then does overlap show its value. The `flat` perf case was added for this.
- G > 0 is slightly slower on cases where A and B both fit in L2; the reason has not been found (inferred to be related to the separate L2 partitions of the two B200 dies, or to the same panel being read by more CTAs at the same time; not verified). G = 8 is the choice with the best worst-case ratio across the three perf cases.
- `--cold` (L2 flushed before each iteration) step 3: square ≈ 0.91×, wide ≈ 0.87×, flat ≈ 1.0×.
- `selftest --perf` perf gate 0.85: the measured worst case is about 0.91 (square, wide), leaving about 6% margin. The GPU occasionally has periods of slowing down across the board (cuBLAS slows down too); the ratios stay basically the same.
- Remaining gap (inferred): each 128×256 tile reads 48 KB from L2 per K tile, and the 148 SMs together approach the L2 bandwidth; square has 512 tiles = 3.46 waves, and 46% of the SMs are idle in the last wave. The former is handled by lab 33's 2-CTA MMA and TMA multicast.

## Verification

- `./lab check 32 --solution`: 3 steps × 9 cases pass (`int` cases exactly equal, `random` max error 1.9e-4). The `persistent` case (372 tiles) makes some CTAs process 3 tiles, rotating accumulators 0→1→0; 6 K tiles per tile shift the ring buffer position across tiles, and the last group has only 7 rows of tiles.
- compute-sanitizer memcheck / racecheck / synccheck: 0 errors on `tail` for all 3 steps; 0 errors from all three tools on step 3 `persistent`. racecheck does not track TMA and tcgen05 (async proxy) accesses, so 0 errors only covers the generic proxy (staging writes). The ordering of the async part is reasoned through in the table below:

| Access | Before | After | Guarantee |
|---|---|---|---|
| TMA writes stage s → MMA reads | `complete_tx` | MMA issued | `full[s]` wait + `fence::after_thread_sync` |
| MMA reads stage s → TMA rewrites | MMA completes | next-round TMA | `tcgen05.commit → empty[s]`, producer wait |
| MMA writes accumulator a → epilogue reads | MMA completes | `tcgen05.ld` | `commit → acc_full[a]`, wait + `fence::after` |
| epilogue reads accumulator a → MMA rewrites | `wait::ld` | the next tile's MMA | `fence::before` + 128 arrivals on `acc_empty[a]`, MMA wait + `fence::after` |
| generic write to staging → TMA store reads | `st.shared` | `cp.async.bulk.tensor` | `fence.proxy.async` + `__syncwarp` |
| TMA store reads staging → rewrite | store finished reading | next write to the box | lane 0 `wait_group.read<1>` + `__syncwarp` |
| all TMEM accesses → dealloc | MMA completes (the epilogue has waited on `acc_full`), `wait::ld` | dealloc | `fence::before` + `__syncthreads` + `fence::after` |
| staging lifetime | store finished reading | CTA exits | lane 0 `wait_group.read<0>` |

- `./lab sass 32 --solution`: 56 / 70 / 72 registers, no spill, no local memory; contains `UTCHMMA`, `UTCBAR`, `UTMALDG`, `UTMASTG`, `LDTM`.
- Skeleton: the solution code was filled in place step by step for TODO(1), (2), (3); `--step n` passes in order, and step n depends only on the first n groups of TODOs. After filling, the student version runs at the same speed as the solution; the skeleton has been restored byte for byte. When unfilled, step 1 traps in the producer (TODO(1)) and steps 2/3 trap in the persistent producer (TODO(2)), without deadlock: every role reaches its own TODO before any mbarrier wait.

## Known limitations

- Static round-robin scheduling: the wave tail of square (3.46 waves) is not handled (stream-K / split-K are outside the scope of this course).
- `kGroupM` is a compile-time constant and is not chosen automatically by shape.
- The epilogue waits right after the ld of each 32-column block (`_sync`) and does not overlap the next block's ld with this block's smem writes; the epilogue already overlaps the main loop, and measurements show it is not the bottleneck.
- BF16 → FP32 only.
