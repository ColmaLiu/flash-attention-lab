// Reference solution for 33_2cta_gemm: C[M,N] = A[M,K] . B[N,K]^T with the
// 2-CTA ("pair") tcgen05.mma: two CTAs of a cluster compute one 256 x 256
// output tile together.
//
// Pair layout (cluster of 2 CTAs, rank 0 = leader, rank 1 = peer):
//   CTA r loads  A rows m0 + 128r .. +127  (its half of the 256 tile rows)
//                B rows n0 + 128r .. +127  (its half of the 256 tile cols)
//   leader issues tcgen05.mma.cta_group::2 M = 256, N = 256: reads A and B
//   from both CTAs' shared memory (same offsets) and writes
//   C rows m0 + 128r .. into CTA r's TMEM (lanes 0..127, all 256 columns).
//   CTA r's epilogue stores rows m0 + 128r .. +127, columns n0 .. n0 + 255.
//
// One CTA = 6 warps (192 threads), roles as in lab 32:
//   warps 0-3 epilogue (both CTAs), warp 4 producer (both CTAs: each loads
//   its own halves), warp 5 MMA (both CTAs allocate TMEM with cta_group::2;
//   only the leader's elected lane issues MMAs and commits).
// Barriers (offsets are the same in both CTAs; "L" = leader's, "P" = peer's):
//   full[s]      used in L  arrive_expect_tx by L's producer (bytes of BOTH
//                           CTAs) + complete_tx of L's and P's TMA loads
//   empty[s]     L and P    multicast tcgen05.commit from L's MMA thread;
//                           each CTA's producer waits on its own
//   acc_full[a]  L and P    multicast commit; each epilogue waits its own
//   acc_empty[a] used in L  256 arrivals: every epilogue thread of L and P
//                           (P's through a .shared::cluster address)
// Before exit every CTA drains the barriers other CTAs signal (producer:
// empty[], leader MMA: acc_empty[]) and the cluster synchronizes, so no
// CTA's shared memory or TMEM disappears while its partner still uses it.
//
// step 1: one 256 x 256 tile per cluster (grid = 2 x tiles), one
//         accumulator: cluster launch, pair MMA, cross-CTA barriers.
// step 2: persistent clusters (grid = 2 x max active clusters) + two
//         accumulators + grouped rasterization (lab 32 steps 2-3 on pairs).
// step 3: clusters of 4 CTAs = 2 pairs side by side along N; each A half is
//         loaded once per cluster and multicast to both pairs. The host
//         picks this shape only when it needs no more tile rounds than
//         pairs of 2 (fewer 4-CTA clusters fit on the GPU).
//
// Supported domain (as labs 31, 32): BF16 A/B, FP32 C, K % 8 == 0,
// N % 4 == 0, K > 0; M arbitrary. TMA loads zero-fill and stores clip tails.
#include "lab/cluster.cuh"
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <algorithm>
#include <cuda_bf16.h>

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
  constexpr uint32_t idesc = lab::make_idesc_f16(kPairM, kBN, /*bf16=*/true);
#pragma unroll
  for (int kk = 0; kk < kBK / kUmmaK; ++kk) {
    const uint64_t da = lab::make_smem_desc_kmajor_sw128(sa + kk * kUmmaK);
    const uint64_t db = lab::make_smem_desc_kmajor_sw128(sb + kk * kUmmaK);
    lab::mma_f16_ss_cg2(d_tmem, da, db, idesc, accumulate || kk > 0);
  }
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
    constexpr int kRows = kBM / Cfg::kPairs; // 64 rows per multicast box
    const uint16_t mask =
        uint16_t((1u << pair.rank) | (1u << (pair.rank + kPairSize)));
    lab::tma_load_2d_cg2_multicast(dst + pair.pair * kRows * kBK, a_map, full_l,
                                   mask, x, a_row + pair.pair * kRows);
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
  for (int t = 0; t < k_tiles; ++t) {
    const int s = ring.index;
    lab::mbar_wait(&sm.empty[s], ring.phase ^ 1);
    if (pair.leader)
      lab::mbar_arrive_expect_tx(&sm.full[s], kPairSize * kStageBytes);
    const uint32_t full_l = lab::map_shared_rank(&sm.full[s], pair.leader_rank);
    load_a<Cfg>(sm.a(s), a_map, full_l, pair, t * kBK, m0 + pair.rank * kBM);
    lab::tma_load_2d_cg2(sm.b(s), b_map, full_l, t * kBK,
                         n0 + pair.rank * kHalfN);
    ring.advance();
  }
}

// After the last tile: wait until every slot has been released once more,
// i.e. every multicast commit aimed at THIS CTA's empty[] has landed. A CTA
// must not exit while the leader's commits can still arrive in its shared
// memory (lab 32 could skip this: a CTA only signalled itself).
template <class Cfg>
__device__ __forceinline__ void producer_drain(const Smem<Cfg> &sm,
                                               RingPos<Cfg::kStages> &ring) {
  for (int i = 0; i < Cfg::kStages; ++i) {
    lab::mbar_wait(&sm.empty[ring.index], ring.phase ^ 1);
    ring.advance();
  }
}

// ---------------------------------------------------------------------------
// MMA issuer (one lane of warp 5, LEADER only). Same as lab 32 but every
// commit is multicast to both CTAs of the pair.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
mma_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring, uint32_t acc_tmem,
         uint64_t *acc_done, uint16_t pair_mask, int k_tiles) {
  for (int t = 0; t < k_tiles; ++t) {
    lab::mbar_wait(&sm.full[ring.index], ring.phase);
    lab::tcgen05_fence_after_thread_sync();
    mma_k_tile_pair(acc_tmem, sm.a(ring.index), sm.b(ring.index), t > 0);
    // Free the slot in every CTA that loads into our pair's smem (all CTAs
    // of the cluster).
    lab::tcgen05_commit_cg2_multicast(&sm.empty[ring.index],
                                      (1u << Cfg::kClusterSize) - 1);
    ring.advance();
  }
  // The accumulator lives in our pair only.
  lab::tcgen05_commit_cg2_multicast(acc_done, pair_mask);
}

// ---------------------------------------------------------------------------
// Epilogue (warps 0-3 of BOTH CTAs): drain this CTA's 128 x 256 part of the
// accumulator to C rows m0.. (lab 32). `acc_free` (a .shared::cluster
// address of the leader's acc_empty, or 0 for none) receives one arrival
// per thread after its last TMEM read.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
epilogue_tile(uint32_t acc_tmem, float *warp_staging, int &box,
              uint32_t acc_free, const CUtensorMap *c_map, int m0, int n0) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll 1
  for (int j = 0; j < kBN / 32; ++j) {
    float v[32];
    lab::tmem_ld_32x32b_x32_sync(lab::tmem_addr(acc_tmem, warp * 32, j * 32),
                                 v);
    if (j == kBN / 32 - 1 && acc_free != 0) {
      // Our TMEM reads are complete (wait::ld). Order them before the
      // arrival; the leader's MMA thread pairs this with
      // mbar_wait_cluster + fence::after_thread_sync.
      lab::tcgen05_fence_before_thread_sync();
      lab::mbar_arrive_cluster(acc_free);
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
//   3. cluster barrier: no CTA may signal a barrier in the other CTA before
//      that CTA has initialized it; the tcgen05 fences order the alloc
//      before everyone reading the TMEM address.
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
  lab::tcgen05_fence_before_thread_sync();
  lab::cluster_sync();
  lab::tcgen05_fence_after_thread_sync();
  return *tmem_slot;
}

// Teardown: every role has drained its barriers (no arrival can still be on
// its way to either CTA) and every TMEM access of the pair has completed.
// The cluster barrier makes both CTAs agree on that before either frees its
// TMEM or exits (the leader's MMAs read the peer's smem and write the
// peer's TMEM; the peer's threads arrive on the leader's barriers).
template <class Cfg> __device__ __forceinline__ void teardown(uint32_t tmem) {
  lab::tcgen05_fence_before_thread_sync();
  lab::cluster_sync();
  if (threadIdx.x / 32 == kMmaWarp) {
    lab::tcgen05_fence_after_thread_sync();
    lab::tmem_dealloc_cg2(tmem, Cfg::kTmemCols);
  }
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
    epilogue_tile(tmem, sm.staging + warp * 2 * kBoxFloats, box, 0, &c_map,
                  m0 + pair.rank * kBM, n0);
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
      RingPos<Cfg::kStages> ring;
      RingPos<Cfg::kAccStages> acc;
      for (int tile = first; tile < num_tiles; tile += stride) {
        // Both CTAs' epilogues must have read this accumulator (256
        // arrivals, some from the peer: cluster-scope acquire).
        lab::mbar_wait_cluster(&sm.acc_empty[acc.index], acc.phase ^ 1);
        lab::tcgen05_fence_after_thread_sync();
        mma_tile(sm, ring, tmem + acc.index * kBN, &sm.acc_full[acc.index],
                 uint16_t(0b11u << (kPairSize * pair.pair)), k_tiles);
        acc.advance();
      }
      // Drain: wait for the epilogues' last releases, so no arrival from
      // the peer can still be on its way into our shared memory.
      for (int i = 0; i < Cfg::kAccStages; ++i) {
        lab::mbar_wait_cluster(&sm.acc_empty[acc.index], acc.phase ^ 1);
        acc.advance();
      }
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
      epilogue_tile(
          tmem + acc.index * kBN, warp_staging, box,
          lab::map_shared_rank(&sm.acc_empty[acc.index], pair.leader_rank),
          &c_map, m0 + pair.rank * kBM, n0 + pair.pair * kBN);
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
    // Choose the cluster shape by counting rounds ("waves") of pair tiles.
    // A 4-CTA cluster must fit in one GPC, so far fewer of them are
    // resident (B200: 33 x 4 = 132 SMs instead of 74 x 2 = 148). Every
    // round each pair computes one 256 x 256 tile in both shapes, so the
    // multicast kernel wins only if it needs no more rounds; then it also
    // reads half as much A from L2 per CTA.
    const int m_tiles = (p.m + kPairM - 1) / kPairM;
    const int pair_tiles = m_tiles * ((p.n + kBN - 1) / kBN);
    const int quad_tiles =
        m_tiles * ((p.n + Step3::kClusterN - 1) / Step3::kClusterN);
    const int rounds2 = (pair_tiles + plan.clusters2 - 1) / plan.clusters2;
    const int rounds3 = (quad_tiles + plan.clusters3 - 1) / plan.clusters3;
    if (rounds3 <= rounds2)
      launch_persistent<Step3>(plan, plan.a_map_mc, plan.clusters3, p, stream);
    else
      launch_persistent<Step2>(plan, plan.a_map, plan.clusters2, p, stream);
    break;
  }
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
