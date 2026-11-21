# 42_fa4_2cta reference solution notes

## Design choices

**The starting point is the lab 41 step 3 solution** (the lab 40 pipeline + the lab 41 softmax: the MUFU/polynomial mix with `kPolyPairs = 3`, conditional
rescale with τ = 8, and correction skipping corrections by warp vote), with only 2-CTA added; all three steps use this softmax. The first version started from lab 40 (the maintainer
later decided that lab 42 builds on lab 41), and the measurements of that version are kept below in the section "2-CTA with the lab 40 softmax". There is no persistent /
LPT from lab 43. The independent review of lab 40 (`5b4157a`) changed only comments and cases; this lab took over its `tile1_position` (renamed `peer_position`) and added
lab 41's `peaked_fp16`, `growing`, `growing_causal`, `jump`, `jump_causal` (conditional rescale votes separately in each of the two CTAs).

**Pair layout (the same as FA4's `use_2cta_instrs`).** cluster = 2 CTAs, owning 512 rows; tile t of CTA r is the 128 rows starting at
`q0 + 256t + 128r`, so pair tile t is 256 contiguous rows. QKᵀ: idesc M = 256, N = 128, B = K split in half by key
(64 keys per CTA, box [64 rows][64]). PV: TS pair MMA, idesc M = 256, N = D, B = V split in half by **column** (each CTA has all
128 keys × D/2 columns). At D = 64 a half of V has only 32 columns = 64 bytes per row, so it uses 64B swizzle (TMA `SWIZZLE_64B`, the new
`make_smem_desc_mnmajor_sw64`). FA4's `smem_size_kv_per_stage // cta_group_size` agrees with this.

**Ring.** One stage = one "half" (a half of K and a half of V have the same byte count: 16 KB for D = 128, 8 KB for D = 64); the ring's total bytes are the same as lab 40
(128 KB), and the stage count doubles: 8 stages for D = 128, 16 stages for D = 64.

**Barriers.** The structure is the same as lab 40, with one copy at the same offset in each of the two CTAs: `q_full`/`kv_full` use only the leader's copy (the leader's TMA
arms the bytes of both sides, and the `cta_group::2` TMA of both sides does complete_tx on the leader); `kv_empty`/`s_full`/`o_full` use both copies (the leader's
`tcgen05.commit.cta_group::2 ... multicast 0b11`); `p_ready` uses only the leader's copy, count 512 = softmax t (128) + correction (128) of both CTAs,
by remote arrive; `stats_full`/`final_full` stay local to each CTA. The MMA warp works only in the leader.

**Setup / drain / teardown (the rules of lab 33).** `fence.mbarrier_init.release.cluster` + warp 0 on both sides does
`tcgen05.alloc.cta_group::2` (512 columns) + `fence::before` / `barrier.cluster` / `fence::after`. Before exiting, each CTA receives all
signals sent to it by others: the TMA drains `kv_empty` (waits S more times); softmax has waited on every `s_full` phase; correction has waited on `o_full` (the last
commit, which tracks all of the leader's earlier MMAs, including reading the peer's smem and writing the peer's TMEM); the leader's MMA has waited on every `p_ready` and
`kv_full` phase (including the peer's remote arrives and complete_tx). Finally, after the cluster barrier, `dealloc.cta_group::2`.

**Causal.** The blocks pair tile t needs to reach `q0 + 256t + 255`: tile 0 walks two fewer blocks than tile 1, and the MMA skips
tile 0 when `j ≥ blocks_0`. Measured, this saves almost no time (`perf_causal` ≈ 1%): in the ping-pong, the cluster's time is set by tile 1. Compared with lab 40, 2-CTA
causal costs about 2 more blocks of SM time per 512 rows (lesson §3.3).

**Steps.** Step 1: pair QKᵀ of the first K/V block only; the softmax threads write `scale·S` into the probe (new oracle
`probe_fa4c_scores`: `[B,H,N,min(N,128)]`, covering the four tiles of a cluster and both halves of K). Step 2: the full pipeline; the cross-CTA
arrive uses `.release.cluster` (lab 33's `mbar_arrive_cluster`). Step 3: switch to `.release.cta` (the new
`mbar_arrive_remote`).

**Why step 3 changes only one arrive.** The first complete version (lab 40 softmax, `.release.cluster`) was about 25% slower than lab 40. Instrumentation (`globaltimer`, one
cluster) found: the two CTAs received `s_full` within ≤ 32 ns of each other, the last thread arrived at about 1350 ns, yet the leader's MMA passed `p_ready` only at
about 2000–2300 ns. SASS: `mbarrier.arrive.release.cluster` = `MEMBAR.ALL.CTA; MEMBAR.ALL.GPU;
ERRBAR; CGAERRBAR; SYNCS.ARRIVE...`, once per thread per block per tile; the default `.release.cta` is only `SYNCS.ARRIVE`.
CUTLASS (`ClusterBarrier::arrive(cta_id)`) and FA4 (pipeline `consumer_release`) both use the default semantics, and the comment in CuTe DSL's
`mbar.py` explains that cluster scope is "measured to be slower" and that the CTA default relies on barrier counting and current hardware/compiler behavior. Formally, a cross-CTA
`.release.cta` and `acquire.cluster` do not form synchronizes-with, and TMEM is no exception; all that is guaranteed is that the arrive is counted and that this thread's
`wait::st`/`wait::ld` has completed. Step 3 deliberately gives up this formal guarantee for speed (lesson §4.4), the same as CUTLASS/FA4.

**Attempts not adopted** (compared in the same run): one remote arrive per warp (count 16) is about 1% faster, but per-thread arrives were kept for consistency with FA4; the leader's
own threads using a local `mbarrier.arrive.shared::cta` with only the peer arriving remotely: slightly slower; the MMA using a plain `try_wait` (without
`.acquire.cluster`): no difference; adding `__syncthreads` before the cluster barrier in the teardown so idle warps park on a hardware barrier: no difference;
registers 192/72/56, 184/80/64: slower or more spills.

**Registers.** The causal D = 128 kernel at first spilled 8 bytes in the 48-register TMA section: `blocks` was computed before the role split and stayed live across
`setmaxnreg`; recomputing it inside each role and adding `#pragma unroll 1` to the panel loop over the K half removed the spill.

## Performance of each step (GPU 3, CUDA Graph, relative to cuDNN; range over two runs)

| Version | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| Lab 41 step 3 (1-CTA) | ≈ 0.94–0.95× | ≈ 1.04–1.05× | ≈ 0.95× | ≈ 0.92× |
| Step 2 (`.release.cluster`) | ≈ 0.73–0.74× | ≈ 0.79–0.80× | ≈ 0.80–0.81× | ≈ 0.71–0.75× |
| Step 3 (`.release.cta`) | ≈ 0.90–0.92× | ≈ 0.96× | ≈ 0.87–0.88× | ≈ 0.89–0.90× |
| Speed of step 3 relative to lab 41 | ≈ 0.96–0.99× | ≈ 0.93–0.95× | ≈ 0.93–0.94× | ≈ 0.98–0.99× |
| FA4 | ≈ 1.15–1.16× | ≈ 1.33–1.35× | ≈ 1.14–1.16× | ≈ 1.08–1.10× |

Step 1 computes only the first block and writes the probe, so timing it is meaningless. `perf_target` = cuDNN × 0.8; `selftest --perf`: perf 0.920, perf_causal 0.951, perf_d64 0.884, perf_8k 0.892 (the worst leaves about 10% margin).

**The main loop ties; the fixed costs lose.** Averaged over all CTAs (`globaltimer` instrumentation, not committed): `perf` main loop lab 41 ≈ 61.4 µs, this lab
≈ 61.4 µs; `perf_d64` ≈ 52.5 / 53.8 µs. Setup ≈ 0.45 / 1.0 µs (presumably mostly the `cta_group::2` TMEM allocation protocol; not measured separately),
teardown ≈ 1.1 / 1.8 µs (cluster barrier, `dealloc.cta_group::2`). About 1.2 µs more in a CTA lifetime of about 65 µs, consistent with the 1–3% gap for non-causal
D = 128 (the upper end is close to noise); persistent (lab 43) pays it only once. Causal also has the diagonal waste of §3.3 (≈ 6% SM time).

**Polynomial ratio.** Under 2-CTA, `kPolyPairs` = 2, 3, 4, 5, 6 were swept: at D = 128, 3–5 are within noise, and 2 and 6 are slower; at D = 64, 4 is about 3% faster than 3
(`perf_d64` ≈ 0.81 → 0.785 ms), but the causal D = 64 kernel spills 24 bytes under 192 softmax registers, so we keep
lab 41's 3. τ is not tuned (lab 41 derives it from the FP16 upper bound; it is not a performance parameter).

**No per-configuration choice in `run()`.** By the measurements, a production policy would choose 1-CTA (lab 41) for causal and D = 64, and non-causal D = 128 is
a tie; this lab needs every case to verify the pair path, so the solution always takes 2-CTA. Lesson §7.4 gives the per-configuration choice table, which agrees with the conditions of FA4's
`use_2cta_instrs`.

### 2-CTA with the lab 40 softmax (the data of lesson §7.2)

The same pair pipeline with the lab 40 softmax: relative to cuDNN ≈ 0.75 / 0.81–0.83 / 0.75–0.78 / 0.72–0.74 (step 2 ≈ 0.66–0.74),
that is ≈ 0.88–0.98× of lab 40. Main loop 69.5 → 78 µs (per block ≈ 3500 → 4050 cycles, clock ≈ 1.62 / 1.66 GHz); NCU (locked clocks)
Tensor Core ≈ 57% / 51%, `smsp__inst_executed_pipe_xu` the same, `lts__t_sectors_srcunit_tex_op_read` ≈ 1.49 / 1.01 GB
(≈ 0.68×; derived 0.53×, cause of the difference not investigated). The extra ~500 cycles per block were not fully located: a small amount of `clock64` instrumentation shrinks the gap to about 3%
(the instrumentation slowed lab 40 by about 10% and barely affected 2-CTA); the hypothesis (unverified) is that the pair locks the two SMs together and jitter delays the ping-pong interleaving.
A timing-only experiment with wrong results (one FMA instead of `MUFU.EX2` for half of the elements): the lab 40 main loop goes 69.6 → 61.2 µs and then stops getting shorter, while 2-CTA goes
78.2 → 56.4 → about 55 µs (about 9–12% faster). The real lab 41 softmax is not that cheap (the polynomial still takes about 10 instructions per pair
on the FMA pipeline), so the measurement only reaches a tie.

## Verification

- `./lab check 42 --solution`: the 10 probe cases of step 1 and the 35 cases of steps 2–3 all pass; `./lab selftest 42` and
  `selftest 42 --perf` pass; the tests in `pytest` related to this lab pass (`test_every_spec_parses_and_has_sources` currently fails because lab 43,
  being written in parallel, has no README yet).
- compute-sanitizer: memcheck, synccheck and racecheck report 0 errors and 0 hazards on `odd_tiles`, `two_tiles` of step 1 and on `odd_tiles`,
  `causal_tail`, `growing_causal` of steps 2–3 (the 2 false positives seen in lab 33 on the cta_group::2 TMEM alloc do not appear here).
  racecheck cannot see TMA, tcgen05, multicast commits or remote arrives; their ordering is reasoned out from the barrier map and drain table in lesson §5.4.
- `./lab sass 42 --solution`: all 20 kernels have REG 128 (`setmaxnreg` sections) and STACK 0; there are `UTCHMMA.2CTA`,
  `UTCBAR.2CTA.MULTICAST`, `UTMALDG.4D.2CTA`, `UTCATOMSWS.2CTA` (alloc). In the step 3 kernels there is
  no `MEMBAR.ALL.GPU` before the `p_ready` arrive (the remaining ones are in the barrier initialization and the cluster barrier).
- Skeleton: generated from the solution; after putting TODO(1), (1)+(2), (1)+(2)+(3) back to solution code step by step, `--step 1/2/3` pass in turn, and the next step reports
  TODO(n+1); with only 1a, 1a–1b, …, 1a–1e filled in, all report TODO(1) on the GPU, and with step 2 partly filled in they report TODO(2), with no hang (including
  the peer: a trap in either CTA terminates the whole grid); with `kPReadyArrivals` unset the kernel reports TODO(2) after setup instead of hanging. Once filled in,
  the student version runs at the same speed as the solution (±2%). After verification the skeleton was restored byte for byte.

## Known limitations

- Without persistence, 2-CTA is not faster than lab 41 (non-causal D = 128 ties to −4%, D = 64 and causal −5–7%); see above.
- Relies on in-order execution of MMAs from the same thread (lab 40).
- The `.release.cta` cross-CTA arrive of step 3 is formally not guaranteed to be correct under the PTX memory model (even though the data is in TMEM); it relies on barrier counting
  and current hardware behavior, the same as CUTLASS/FA4. The formally correct version is step 2 (about 1.2 times slower).
- Causal: the 512 rows of a cluster advance together, so the diagonal waste is larger than in lab 40; stopping tile 0 early saves no time.
- Not persistent; when N is very small, a peer or tile 1 that is fully out of bounds still takes part in the pair MMA.
