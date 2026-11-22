// 33_2cta_gemm: C[M,N] = A[M,K] . B[N,K]^T with the 2-CTA ("pair")
// tcgen05.mma: two CTAs of a thread block cluster compute one 256 x 256
// output tile together.
// Lesson: course/33_2cta_gemm/README.md
//
// A is [M, K], B is [N, K], both BF16, row-major (K-major); C is [M, N] FP32.
// Supported (as labs 31, 32): K % 8 == 0, N % 4 == 0, K > 0; M arbitrary.
// Tails need no code: TMA loads zero-fill, TMA stores clip.
//
// Pair layout (lesson §4.4): cluster rank 0 = leader, rank 1 = peer.
//   CTA r loads A rows m0 + 128r .. +127 and B rows n0 + 128r .. +127;
//   the leader's MMA (M = 256, N = 256) reads both CTAs' shared memory and
//   writes C rows m0 + 128r .. into CTA r's TMEM; CTA r stores those rows.
// One CTA = 6 warps as in lab 32: warps 0-3 epilogue, warp 4 producer,
// warp 5 MMA (TMEM alloc in both CTAs, MMAs only in the leader).
// Barriers (lesson §5.2; L = leader's copy, P = peer's copy):
//   full[s]      used in L  L arms with both CTAs' bytes; L's and P's TMA
//   empty[s]     L and P    L's multicast tcgen05.commit
//   acc_full[a]  L and P    L's multicast tcgen05.commit
//   acc_empty[a] used in L  256 arrivals: epilogue threads of L and P
// run() picks a version by p.step:  ./lab bench 33 --step 1 / 2 / 3
//
// mbarrier, TMA and tcgen05 helpers come from include/lab/ (labs 30-32).
// The cluster / pair primitives are yours to write (below); from lab 34 on
// they come from include/lab/cluster.cuh.
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <algorithm>
#include <cuda_bf16.h>

// ===========================================================================
// Cluster and pair primitives (lesson §4). You write them here; from lab 34
// on they come from include/lab/cluster.cuh. One PTX instruction each
// (the waits: a retry loop around one). Shared addresses: lab::shared_addr
// ("r" operand); 16-bit masks are "h" operands; "memory" clobber on
// everything except reading special registers and mapa.
// ===========================================================================
namespace lab {

// ---------------------------------------------------------------------------
// TODO(1a) — cluster identity, cluster barrier, DSMEM address (lesson §4.1,
// §4.2):
//   cluster_ctarank:  mov.u32 %0, %%cluster_ctarank;
//   cluster_id_x:     mov.u32 %0, %%clusterid.x;
//   num_clusters_x:   mov.u32 %0, %%nclusterid.x;
//   cluster_arrive:   barrier.cluster.arrive.release.aligned;
//   cluster_wait:     barrier.cluster.wait.acquire.aligned;
//   map_shared_rank:  mapa.shared::cluster.u32 dst, src, rank;
//                     src = shared_addr(p); returns the .shared::cluster
//                     address of the same offset in CTA `rank`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cluster_ctarank() {
  LAB_TODO(1);
  return 0;
}
__device__ __forceinline__ uint32_t cluster_id_x() {
  LAB_TODO(1);
  return 0;
}
__device__ __forceinline__ uint32_t num_clusters_x() {
  LAB_TODO(1);
  return 0;
}
__device__ __forceinline__ void cluster_arrive() {
  LAB_TODO(1);
}
__device__ __forceinline__ void cluster_wait() {
  LAB_TODO(1);
}
__device__ __forceinline__ void cluster_sync() { // given
  cluster_arrive();
  cluster_wait();
}
__device__ __forceinline__ uint32_t map_shared_rank(const void *p,
                                                    uint32_t rank) {
  LAB_TODO(1);
  return 0;
}

// ---------------------------------------------------------------------------
// TODO(1b) — pair forms of TMA and tcgen05 (lesson §4.3, §4.4):
//   tma_load_2d_cg2: like lab::tma_load_2d, but the barrier operand `bar`
//     is a .shared::cluster address (it may be the peer's barrier):
//     cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
//         .mbarrier::complete_tx::bytes [dst], [map, {x, y}], [bar];
//   tmem_alloc_cg2 / tmem_relinquish_alloc_permit_cg2 / tmem_dealloc_cg2:
//     the lab 31 instructions with .cta_group::2 instead of ::1;
//   mma_f16_ss_cg2: tcgen05.mma.cta_group::2.kind::f16 [d], a, b, idesc, p;
//     (predicate p from `accumulate`, as lab::mma_f16_ss);
//   tcgen05_commit_cg2_multicast:
//     tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster
//         .multicast::cluster.b64 [bar], cta_mask;
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
tma_load_2d_cg2(void *dst, const CUtensorMap *map, uint32_t bar, int x, int y) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_alloc_cg2(uint32_t *dst, uint32_t ncols) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_relinquish_alloc_permit_cg2() {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t taddr,
                                                 uint32_t ncols) {
  LAB_TODO(1);
}
__device__ __forceinline__ void mma_f16_ss_cg2(uint32_t d_tmem, uint64_t a_desc,
                                               uint64_t b_desc, uint32_t idesc,
                                               bool accumulate) {
  LAB_TODO(1);
}
__device__ __forceinline__ void
tcgen05_commit_cg2_multicast(uint64_t *bar, uint16_t cta_mask) {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(2a) — mbarrier across CTAs (lesson §4.2):
//   mbar_arrive_cluster: mbarrier.arrive.release.cluster.shared::cluster.b64
//     _, [addr];   (addr from map_shared_rank)
//   mbar_wait_cluster: lab::mbar_wait with
//     mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64
//     (our own barrier, but the arrivals come from other CTAs).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr) {
  LAB_TODO(2);
}
__device__ __forceinline__ void mbar_wait_cluster(uint64_t *bar,
                                                  uint32_t parity) {
  LAB_TODO(2);
}

// ---------------------------------------------------------------------------
// TODO(3a) — multicast TMA (lesson §4.5): tma_load_2d_cg2 plus
// .multicast::cluster and a 16-bit cta_mask operand after [bar]:
//   cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
//       .mbarrier::complete_tx::bytes.multicast::cluster
//       [dst], [map, {x, y}], [bar], cta_mask;
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
tma_load_2d_cg2_multicast(void *dst, const CUtensorMap *map, uint32_t bar,
                          uint16_t cta_mask, int x, int y) {
  LAB_TODO(3);
}

// ---------------------------------------------------------------------------
// TODO(1c) — host: launch `kernel` with clusters of cluster_x CTAs along x
// (lesson §4.1). Fill a cudaLaunchConfig_t (gridDim, blockDim,
// dynamicSmemBytes, stream) with one cudaLaunchAttribute
// {cudaLaunchAttributeClusterDimension, clusterDim = (cluster_x, 1, 1)}, then
//   LAB_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
//                                     static_cast<KArgs>(args)...));
// ---------------------------------------------------------------------------
template <class... KArgs, class... Args>
void launch_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block,
                    int smem_bytes, cudaStream_t stream, int cluster_x,
                    Args &&...args) {
  LAB_TODO(1);
}

// How many clusters of cluster_x CTAs can be resident at once (given).
template <class... KArgs>
int max_active_clusters(void (*kernel)(KArgs...), dim3 block, int smem_bytes,
                        int cluster_x) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster_x, 1, 1);
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_x;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  LAB_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &n, reinterpret_cast<const void *>(kernel), &cfg));
  return n;
}

} // namespace lab

namespace {

using bf16 = __nv_bfloat16;

constexpr int kBM = 128;        // rows of C per CTA (= TMEM lanes)
constexpr int kPairM = 2 * kBM; // MMA M of the pair
constexpr int kBN = 256;        // MMA N (columns of C per tile)
constexpr int kHalfN = kBN / 2; // B rows held by each CTA
constexpr int kBK = 64;         // 64 BF16 = one 128-byte swizzle row
constexpr int kUmmaK = 16;      // K of one tcgen05.mma for 16-bit inputs
constexpr int kPairSize = 2;    // CTAs per pair

constexpr int kEpilogueWarps = 4;
constexpr int kProducerWarp = 4;
constexpr int kMmaWarp = 5;
constexpr int kThreads = 32 * 6;
constexpr int kEpilogueThreads = 32 * kEpilogueWarps;

constexpr int kBoxFloats = 32 * 32;
constexpr int kStagingBytes = kEpilogueWarps * 2 * kBoxFloats * 4; // 32 KB

// Bytes one CTA's TMA loads bring per stage: A half + B half.
constexpr int kStageBytes = (kBM + kHalfN) * kBK * 2; // 32 KB

template <int Stages, int AccStages, int GroupM, int Pairs = 1> struct Config {
  static constexpr int kStages = Stages, kAccStages = AccStages;
  static constexpr int kGroupM = GroupM; // 0: row-major tile order
  // Pairs = 2: a cluster of 4 CTAs = two pairs side by side along N (same
  // A rows): the A halves are multicast across the two pairs.
  static constexpr int kPairs = Pairs, kClusterSize = kPairSize * Pairs;
  static constexpr int kClusterN = kBN * Pairs; // C columns per cluster tile
  static_assert(Pairs == 1 || Pairs == 2, "1 or 2 pairs per cluster");
  static constexpr int kSmemBytes = 1024 + Stages * kStageBytes + kStagingBytes;
  static constexpr int kTmemCols = AccStages * kBN;
  static_assert(kTmemCols == 256 || kTmemCols == 512, "TMEM: power of two");
  static_assert(kSmemBytes <= 227 * 1024, "dynamic shared memory limit");
};

__device__ __forceinline__ uint8_t *align_1024(uint8_t *p) {
  return p + ((1024 - (lab::shared_addr(p) & 1023)) & 1023);
}

// D (both CTAs' TMEM) [+]= A[256 x 64] . B[256 x 64]^T: four pair MMAs of
// K = 16. sa / sb are the LEADER's stage buffers; the hardware reads the
// peer's halves at the same offsets.
__device__ __forceinline__ void mma_k_tile_pair(uint32_t d_tmem, const bf16 *sa,
                                                const bf16 *sb,
                                                bool accumulate) {
  // -------------------------------------------------------------------------
  // TODO(1d) — four pair MMAs of K = 16 (lesson §4.4). As lab 32's
  // mma_k_tile, but the instruction descriptor describes the WHOLE pair
  // MMA (M = ?, N = ?) and you call lab::mma_f16_ss_cg2. The descriptors
  // point at the leader's buffers only.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

template <class Cfg> struct Smem {
  uint8_t *stages; // kStages x (A [128][64], B [128][64]) BF16
  float *staging;  // kEpilogueWarps x 2 boxes of 32 x 32 FP32
  uint64_t *full, *empty, *acc_full, *acc_empty;
  __device__ bf16 *a(int s) const {
    return reinterpret_cast<bf16 *>(stages + s * kStageBytes);
  }
  __device__ bf16 *b(int s) const {
    return a(s) + kBM * kBK;
  }
};

template <int N> struct RingPos {
  int index = 0;
  uint32_t phase = 0;
  __device__ __forceinline__ void advance() {
    if (++index == N) {
      index = 0;
      phase ^= 1;
    }
  }
};

// Who am I in the cluster.
struct PairInfo {
  uint32_t rank;        // 0 = leader, 1 = peer (rank inside the pair)
  uint32_t pair;        // which pair of the cluster (0 when Pairs = 1)
  uint32_t leader_rank; // cluster rank of our pair's leader
  bool leader;
  __device__ static PairInfo get() {
    const uint32_t r = lab::cluster_ctarank();
    return PairInfo{r % kPairSize, r / kPairSize, r & ~1u, r % kPairSize == 0};
  }
};

// ---------------------------------------------------------------------------
// The A half of one stage. Pairs = 1: this CTA loads its 128 rows.
// Pairs = 2 (cluster of 4, pairs side by side along N): CTA r and CTA r + 2
// need the same A half. Each loads 64 of the 128 rows and multicasts them
// to both (mask bit r and bit r + 2); each copy completes on the full
// barrier of its DESTINATION pair's leader (same offset as `full_l`).
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void load_a(bf16 *dst, const CUtensorMap *a_map,
                                       uint32_t full_l, const PairInfo &pair,
                                       int x, int a_row) {
  if constexpr (Cfg::kPairs == 1) {
    lab::tma_load_2d_cg2(dst, a_map, full_l, x, a_row);
  } else {
    // -----------------------------------------------------------------------
    // TODO(3b) — multicast A (lesson §4.5, §5.5). Here a_map's box has
    // kBM / Cfg::kPairs rows. Questions to answer first:
    //   - the A half this CTA needs is also needed by which CTA of the other
    //     pair? Which part of it does THIS CTA load, and where in `dst`?
    //   - which bits must cta_mask have so the box lands in both CTAs?
    //   - which barrier address do you pass, and on which leader's barrier
    //     does the copy for the other pair complete?
    // Then one call to lab::tma_load_2d_cg2_multicast.
    // -----------------------------------------------------------------------
    (void)pair;
    LAB_TODO(3);
  }
}

// ---------------------------------------------------------------------------
// Producer (one lane of warp 4, in EVERY CTA): this CTA's halves of the K
// tiles of one output tile. Every load completes on our pair LEADER's
// full[s] (a .shared::cluster address); only the leader arms it, with the
// bytes of both CTAs. The slot is reused only after the MMAs that read it
// are done: empty[s] here gets the leader's multicast commit (with
// Pairs = 2, both leaders' commits: the other pair writes A into it too).
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
produce_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring,
             const PairInfo &pair, const CUtensorMap *a_map,
             const CUtensorMap *b_map, int m0, int n0, int k_tiles) {
  // -------------------------------------------------------------------------
  // TODO(1e) — pair producer (lesson §5.2). For each K tile t, slot
  // s = ring.index:
  //   - wait until THIS CTA's slot is free: empty[s], parity ring.phase ^ 1;
  //   - only the leader arms ITS full[s] -- with how many bytes? (Both CTAs'
  //     loads complete on it.)
  //   - full_l = the .shared::cluster address of full[s] in the leader
  //     (pair.leader_rank);
  //   - load_a<Cfg>(sm.a(s), a_map, full_l, pair, x = 64t, a_row = this
  //     CTA's first A row) and the B half with lab::tma_load_2d_cg2 (which
  //     B rows does this CTA hold?);
  //   - ring.advance().
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

// After the last tile: wait until every slot has been released once more,
// i.e. every multicast commit aimed at THIS CTA's empty[] has landed. A CTA
// must not exit while the leader's commits can still arrive in its shared
// memory (lab 32 could skip this: a CTA only signalled itself).
template <class Cfg>
__device__ __forceinline__ void producer_drain(const Smem<Cfg> &sm,
                                               RingPos<Cfg::kStages> &ring) {
  // -------------------------------------------------------------------------
  // TODO(1f) — drain (lesson §5.4): wait for the release of each of the
  // last kStages uses of the ring, i.e. acquire kStages more slots exactly
  // as the producer would (empty, parity phase ^ 1, advance), without
  // loading anything.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// MMA issuer (one lane of warp 5, LEADER only). Same as lab 32 but every
// commit is multicast to both CTAs of the pair.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
mma_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring, uint32_t acc_tmem,
         uint64_t *acc_done, uint16_t pair_mask, int k_tiles) {
  // -------------------------------------------------------------------------
  // TODO(1g) — pair MMA issuer (lesson §5.2): lab 32's mma_tile with
  //   - mma_k_tile_pair instead of mma_k_tile;
  //   - every commit multicast: the slot is freed in every CTA of the
  //     cluster (mask (1 << Cfg::kClusterSize) - 1: both CTAs load into
  //     the stage the MMA read), the accumulator is final in our pair only
  //     (pair_mask).
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// Epilogue (warps 0-3 of BOTH CTAs): drain this CTA's 128 x 256 part of the
// accumulator to C rows m0.. (lab 32). If `release`, `acc_free` (the
// .shared::cluster address of our pair leader's acc_empty) receives one
// arrival per thread after its last TMEM read.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
epilogue_tile(uint32_t acc_tmem, float *warp_staging, int &box, bool release,
              uint32_t acc_free, const CUtensorMap *c_map, int m0, int n0) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll 1
  for (int j = 0; j < kBN / 32; ++j) {
    float v[32];
    lab::tmem_ld_32x32b_x32_sync(lab::tmem_addr(acc_tmem, warp * 32, j * 32),
                                 v);
    if (j == kBN / 32 - 1 && release) {
      // ---------------------------------------------------------------------
      // TODO(2c) — release the accumulator to the LEADER's MMA thread
      // (lesson §5.3): lab 32's fence + arrive, but acc_free is a
      // .shared::cluster address (possibly in the other CTA).
      // ---------------------------------------------------------------------
      LAB_TODO(2);
    }
    if (lane == 0)
      lab::bulk_wait_group_read<1>();
    __syncwarp();
    float4 *dst = reinterpret_cast<float4 *>(warp_staging + box * kBoxFloats);
#pragma unroll
    for (int q = 0; q < 8; ++q)
      dst[lane * 8 + (q ^ (lane % 8))] =
          make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    lab::fence_proxy_async();
    __syncwarp();
    if (lane == 0) {
      lab::tma_store_2d(c_map, dst, n0 + j * 32, m0 + warp * 32);
      lab::bulk_commit_group();
    }
    box ^= 1;
  }
}

// ---------------------------------------------------------------------------
// Setup, shared by both kernels:
//   1. thread 0 initializes this CTA's barriers; fence.mbarrier_init
//      .release.cluster makes the init visible to the other CTA and to the
//      async proxy (the peer's TMA, the leader's multicast commit).
//   2. warp 5 of BOTH CTAs allocates TMEM with cta_group::2.
//   3. TODO(1h): synchronize before any role starts.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ uint32_t setup(Smem<Cfg> &sm, uint32_t *tmem_slot,
                                          const CUtensorMap *a_map,
                                          const CUtensorMap *b_map,
                                          const CUtensorMap *c_map) {
  const int warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < Cfg::kStages; ++s) {
      lab::mbar_init(&sm.full[s], 1); // leader's arrive_expect_tx (+ bytes)
      // multicast commit of every pair leader that reads this CTA's loads
      lab::mbar_init(&sm.empty[s], Cfg::kPairs);
    }
    for (int a = 0; a < Cfg::kAccStages; ++a) {
      lab::mbar_init(&sm.acc_full[a], 1); // leader's multicast commit
      // Every epilogue thread of both CTAs (used in the leader only).
      lab::mbar_init(&sm.acc_empty[a], kPairSize * kEpilogueThreads);
    }
    lab::fence_barrier_init();
  }
  if (warp == kProducerWarp && lab::elect_one()) {
    lab::prefetch_tensor_map(a_map);
    lab::prefetch_tensor_map(b_map);
    lab::prefetch_tensor_map(c_map);
  }
  if (warp == kMmaWarp) {
    __syncwarp();
    lab::tmem_alloc_cg2(tmem_slot, Cfg::kTmemCols);
    lab::tmem_relinquish_alloc_permit_cg2();
  }
  // -------------------------------------------------------------------------
  // TODO(1h) — make the barrier inits and the TMEM address visible (lesson
  // §5.4). In lab 32: fence::before; __syncthreads(); fence::after. Why is
  // __syncthreads() no longer enough? (Who signals our barriers?)
  // -------------------------------------------------------------------------
  LAB_TODO(1);
  return *tmem_slot;
}

// Teardown: every role has drained its barriers (no arrival can still be on
// its way to this CTA) and every TMEM access has completed. Which threads
// must agree on that before a CTA frees its TMEM or exits?
template <class Cfg> __device__ __forceinline__ void teardown(uint32_t tmem) {
  // -------------------------------------------------------------------------
  // TODO(1h) — the same synchronization as setup, then warp kMmaWarp frees
  // the TMEM (lab::tmem_dealloc_cg2, Cfg::kTmemCols columns).
  // -------------------------------------------------------------------------
  (void)tmem;
  LAB_TODO(1);
}

#define LAB33_SMEM_DECL(Cfg)                                                   \
  extern __shared__ uint8_t smem_raw[];                                        \
  __shared__ __align__(8) uint64_t bar_full[Cfg::kStages],                     \
      bar_empty[Cfg::kStages], bar_acc_full[Cfg::kAccStages],                  \
      bar_acc_empty[Cfg::kAccStages];                                          \
  __shared__ uint32_t tmem_slot;                                               \
  Smem<Cfg> sm;                                                                \
  sm.stages = align_1024(smem_raw);                                            \
  sm.staging =                                                                 \
      reinterpret_cast<float *>(sm.stages + Cfg::kStages * kStageBytes);       \
  sm.full = bar_full;                                                          \
  sm.empty = bar_empty;                                                        \
  sm.acc_full = bar_acc_full;                                                  \
  sm.acc_empty = bar_acc_empty

// ===========================================================================
// Step 1: one 256 x 256 tile per cluster.
// ===========================================================================
template <class Cfg>
__global__ void __launch_bounds__(kThreads, 1)
    gemm_pair_tile(const __grid_constant__ CUtensorMap a_map,
                   const __grid_constant__ CUtensorMap b_map,
                   const __grid_constant__ CUtensorMap c_map, int n, int k) {
  LAB33_SMEM_DECL(Cfg);
  const uint32_t tmem = setup(sm, &tmem_slot, &a_map, &b_map, &c_map);
  const int warp = threadIdx.x / 32;
  static_assert(Cfg::kPairs == 1, "step 1: one pair per cluster");
  const PairInfo pair = PairInfo::get();
  const int n_tiles = (n + kBN - 1) / kBN;
  const int tile = lab::cluster_id_x();
  const int m0 = (tile / n_tiles) * kPairM, n0 = (tile % n_tiles) * kBN;
  const int k_tiles = (k + kBK - 1) / kBK;

  if (warp == kProducerWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      produce_tile(sm, ring, pair, &a_map, &b_map, m0, n0, k_tiles);
      producer_drain(sm, ring);
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    if (pair.leader && lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      mma_tile(sm, ring, tmem, &sm.acc_full[0], 0b11, k_tiles);
    }
    __syncwarp();
  } else {
    // acc_full[0] of THIS CTA: the leader's commit is multicast to both.
    lab::mbar_wait(&sm.acc_full[0], 0);
    lab::tcgen05_fence_after_thread_sync();
    int box = 0;
    epilogue_tile(tmem, sm.staging + warp * 2 * kBoxFloats, box, false, 0,
                  &c_map, m0 + pair.rank * kBM, n0);
    if (threadIdx.x % 32 == 0)
      lab::bulk_wait_group_read<0>();
    __syncwarp();
  }
  teardown<Cfg>(tmem);
}

// ===========================================================================
// Steps 2 and 3: persistent clusters, double-buffered accumulator.
// ===========================================================================
template <class Cfg>
__device__ __forceinline__ void tile_coords(int tile, int m_tiles, int n_tiles,
                                            int &m0, int &n0) {
  int mb, nb;
  if constexpr (Cfg::kGroupM == 0) {
    mb = tile / n_tiles;
    nb = tile % n_tiles;
  } else {
    const int per_group = Cfg::kGroupM * n_tiles;
    const int group = tile / per_group;
    const int first_m = group * Cfg::kGroupM;
    const int rows = min(Cfg::kGroupM, m_tiles - first_m);
    const int in_group = tile - group * per_group;
    mb = first_m + in_group % rows;
    nb = in_group / rows;
  }
  m0 = mb * kPairM;
  n0 = nb * Cfg::kClusterN;
}

template <class Cfg>
__global__ void __launch_bounds__(kThreads, 1)
    gemm_pair_persistent(const __grid_constant__ CUtensorMap a_map,
                         const __grid_constant__ CUtensorMap b_map,
                         const __grid_constant__ CUtensorMap c_map, int m,
                         int n, int k) {
  LAB33_SMEM_DECL(Cfg);
  const uint32_t tmem = setup(sm, &tmem_slot, &a_map, &b_map, &c_map);
  const int warp = threadIdx.x / 32;
  const PairInfo pair = PairInfo::get();
  const int m_tiles = (m + kPairM - 1) / kPairM;
  const int n_tiles = (n + Cfg::kClusterN - 1) / Cfg::kClusterN;
  const int num_tiles = m_tiles * n_tiles;
  const int k_tiles = (k + kBK - 1) / kBK;
  const int first = lab::cluster_id_x(), stride = lab::num_clusters_x();

  if (warp == kProducerWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      for (int tile = first; tile < num_tiles; tile += stride) {
        int m0, n0;
        tile_coords<Cfg>(tile, m_tiles, n_tiles, m0, n0);
        produce_tile(sm, ring, pair, &a_map, &b_map, m0, n0 + pair.pair * kBN,
                     k_tiles);
      }
      producer_drain(sm, ring);
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    if (pair.leader && lab::elect_one()) {
      // ---------------------------------------------------------------------
      // TODO(2b) — persistent pair MMA issuer (lesson §5.3). Start from
      // lab 32's persistent MMA loop (one RingPos for the stages, one for
      // the accumulators; per tile wait until the accumulator is free, fence,
      // mma_tile, advance). What changes on a pair?
      //   - who arrives on acc_empty now, and which wait pairs with them?
      //   - which CTAs must acc_full reach (pair_mask, also with 2 pairs)?
      //   - after the last tile: which arrivals into OUR shared memory may
      //     still be in flight, and how do you wait for them (cf.
      //     producer_drain)?
      // ---------------------------------------------------------------------
      LAB_TODO(2);
    }
    __syncwarp();
  } else {
    RingPos<Cfg::kAccStages> acc;
    int box = 0;
    float *warp_staging = sm.staging + warp * 2 * kBoxFloats;
    for (int tile = first; tile < num_tiles; tile += stride) {
      int m0, n0;
      tile_coords<Cfg>(tile, m_tiles, n_tiles, m0, n0);
      lab::mbar_wait(&sm.acc_full[acc.index], acc.phase);
      lab::tcgen05_fence_after_thread_sync();
      // ---------------------------------------------------------------------
      // TODO(2d) — drain accumulator acc.index (lesson §5.3) with
      // epilogue_tile(..., release = true, acc_free, ...). acc_free: the
      // .shared::cluster address of acc_empty[acc.index] in our pair's
      // LEADER. Rows: this CTA's half; columns: our pair's tile.
      // ---------------------------------------------------------------------
      (void)m0;
      (void)n0;
      LAB_TODO(2);
      acc.advance();
    }
    if (threadIdx.x % 32 == 0)
      lab::bulk_wait_group_read<0>();
    __syncwarp();
  }
  teardown<Cfg>(tmem);
}

// 32 KB stages: 6 fit next to the 32 KB staging (lab 32: 4 x 48 KB).
using Step1 = Config<6, 1, 0>;
using Step2 = Config<6, 2, 8>;
// Step 3: two pairs per cluster (cluster of 4), A multicast across them.
using Step3 = Config<6, 2, 8, 2>;

template <class Kernel> void allow_smem(Kernel kernel, int bytes) {
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

CUtensorMap operand_map(const void *base, int rows, int k, int box_rows) {
  return lab::make_tensor_map_2d(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rows,
                                 k, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
}

int num_tiles(const lab::GemmProblem &p) {
  return ((p.m + kPairM - 1) / kPairM) * ((p.n + kBN - 1) / kBN);
}

} // namespace

struct Plan {
  CUtensorMap a_map, a_map_mc, b_map, c_map; // a_map_mc: multicast A box
  int clusters2, clusters3; // resident clusters for the persistent kernels
};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::BF16, "this lab uses BF16 A and B");
  LAB_REQUIRE(p.c_dtype == lab::DType::F32, "this lab writes an FP32 C");
  LAB_REQUIRE(p.k > 0, "K must be positive");
  LAB_REQUIRE(p.k % 8 == 0, "K must be a multiple of 8 (TMA: 16-byte rows)");
  LAB_REQUIRE(p.n % 4 == 0, "N must be a multiple of 4 (TMA: 16-byte rows)");
  Plan plan{};
  allow_smem(gemm_pair_tile<Step1>, Step1::kSmemBytes);
  allow_smem(gemm_pair_persistent<Step2>, Step2::kSmemBytes);
  allow_smem(gemm_pair_persistent<Step3>, Step3::kSmemBytes);
  plan.clusters2 =
      lab::max_active_clusters(gemm_pair_persistent<Step2>, dim3(kThreads),
                               Step2::kSmemBytes, Step2::kClusterSize);
  plan.clusters3 =
      lab::max_active_clusters(gemm_pair_persistent<Step3>, dim3(kThreads),
                               Step3::kSmemBytes, Step3::kClusterSize);
  LAB_REQUIRE(plan.clusters2 > 0 && plan.clusters3 > 0,
              "no 2-CTA cluster fits on this GPU");
  plan.a_map = operand_map(p.a, p.m, p.k, kBM);
  plan.a_map_mc = operand_map(p.a, p.m, p.k, kBM / Step3::kPairs);
  plan.b_map = operand_map(p.b, p.n, p.k, kHalfN);
  plan.c_map =
      lab::make_tensor_map_2d(p.c, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, p.m, p.n,
                              32, 32, CU_TENSOR_MAP_SWIZZLE_128B);
  return plan;
}

template <class Cfg>
void launch_persistent(const Plan &plan, const CUtensorMap &a_map, int clusters,
                       const lab::GemmProblem &p, cudaStream_t stream) {
  const int tiles = ((p.m + kPairM - 1) / kPairM) *
                    ((p.n + Cfg::kClusterN - 1) / Cfg::kClusterN);
  const int grid = Cfg::kClusterSize * std::min(tiles, clusters);
  lab::launch_cluster(gemm_pair_persistent<Cfg>, dim3(grid), dim3(kThreads),
                      Cfg::kSmemBytes, stream, Cfg::kClusterSize, a_map,
                      plan.b_map, plan.c_map, p.m, p.n, p.k);
}

void run(const Plan &plan, const lab::GemmProblem &p, cudaStream_t stream) {
  switch (p.step) {
  case 1:
    lab::launch_cluster(
        gemm_pair_tile<Step1>, dim3(Step1::kClusterSize * num_tiles(p)),
        dim3(kThreads), Step1::kSmemBytes, stream, Step1::kClusterSize,
        plan.a_map, plan.b_map, plan.c_map, p.n, p.k);
    break;
  case 2:
    launch_persistent<Step2>(plan, plan.a_map, plan.clusters2, p, stream);
    break;
  default: {
    // -----------------------------------------------------------------------
    // TODO(3c) — choose the cluster shape (lesson §5.5). A 4-CTA cluster
    // (Step3: two pairs, A multicast) must fit in one GPC, so fewer of them
    // are resident: plan.clusters3 clusters of 4 vs plan.clusters2 pairs.
    // Decide from the number of tile rounds each shape needs (lesson §3.4,
    // §5.5) when the multicast kernel is worth it, then launch either
    // launch_persistent<Step3> (A map: plan.a_map_mc) or
    // launch_persistent<Step2> (plan.a_map) with the matching cluster count.
    // -----------------------------------------------------------------------
    LAB_TODO(3);
    break;
  }
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
