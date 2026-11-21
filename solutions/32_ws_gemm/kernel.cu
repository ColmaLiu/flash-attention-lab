// Reference solution for 32_ws_gemm: C[M,N] = A[M,K] . B[N,K]^T with
// tcgen05.mma, split into warp-specialized roles.
//
// One CTA = 6 warps (192 threads), each warp has exactly one job:
//   warps 0-3  epilogue: TMEM -> registers -> swizzled smem -> TMA store.
//              Warp w may only read TMEM lanes 32w..32w+31 (= C rows).
//   warp  4    producer: one elected lane issues every TMA load.
//   warp  5    MMA issuer: allocates TMEM; one elected lane issues every
//              tcgen05.mma and tcgen05.commit.
// The roles talk only through mbarriers:
//   full[s]      producer -> MMA      stage s holds A/B (TMA complete_tx)
//   empty[s]     MMA -> producer      MMAs reading stage s are done (commit)
//   acc_full[a]  MMA -> epilogue      accumulator a is final (commit)
//   acc_empty[a] epilogue -> MMA      all 128 epilogue threads have read
//                                     accumulator a (arrive x 128)
//
// step 1: one output tile per CTA (grid = all tiles), one accumulator.
//         The role split alone: TMA and MMA no longer wait for each other,
//         but the epilogue still runs after the CTA's main loop.
// step 2: persistent CTAs (grid = #SMs, static round-robin tile schedule)
//         and two TMEM accumulators: the epilogue of tile i overlaps the
//         main loop of tile i + 1. Ring and accumulator phases run on across
//         tiles.
// step 3: grouped rasterization of the tile order (L2 reuse of A and B).
//
// Supported domain (same as lab 31): BF16 A/B, FP32 C, K % 8 == 0 and
// N % 4 == 0 (16-byte TMA row pitches); M arbitrary. TMA loads zero-fill and
// TMA stores clip all tails.
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kBM = 128;   // MMA M: one TMEM lane per row of C
constexpr int kBK = 64;    // 64 BF16 = 128 bytes: one 128B swizzle row
constexpr int kUmmaK = 16; // K of one tcgen05.mma for 16-bit inputs

constexpr int kEpilogueWarps = 4; // warps 0..3: the 4 TMEM lane quarters
constexpr int kProducerWarp = 4;
constexpr int kMmaWarp = 5;
constexpr int kThreads = 32 * 6;
constexpr int kEpilogueThreads = 32 * kEpilogueWarps;

// Epilogue staging: every epilogue warp owns two 32 x 32 FP32 boxes (4 KB
// each, 128B swizzle) and alternates between them, so it can fill one while
// the TMA store of the other is still reading shared memory.
constexpr int kBoxFloats = 32 * 32;
constexpr int kStagingBytes = kEpilogueWarps * 2 * kBoxFloats * 4; // 32 KB

template <int BN, int Stages, int AccStages, int GroupM> struct Config {
  static constexpr int kBN = BN, kStages = Stages, kAccStages = AccStages;
  // GroupM = 0: row-major tile order (n fastest). GroupM = g > 0: groups of
  // g tile rows, m fastest inside a group.
  static constexpr int kGroupM = GroupM;
  static constexpr int kStageBytes = (kBM + BN) * kBK * 2;
  static constexpr int kSmemBytes = 1024 + Stages * kStageBytes + kStagingBytes;
  static constexpr int kTmemCols = AccStages * BN; // FP32: one column per n
  static_assert(BN % 32 == 0 && BN <= 256, "MMA N: multiple of 16, <= 256");
  static_assert(kTmemCols == 128 || kTmemCols == 256 || kTmemCols == 512,
                "TMEM allocations are powers of two, 32..512 columns");
  static_assert(kSmemBytes <= 227 * 1024, "dynamic shared memory limit");
};

__device__ __forceinline__ uint8_t *align_1024(uint8_t *p) {
  return p + ((1024 - (lab::shared_addr(p) & 1023)) & 1023);
}

// D (TMEM) [+]= A_tile[128 x 64] . B_tile[BN x 64]^T as 4 MMAs of K = 16
// (lab 31): K slice kk starts 32 * kk bytes into the swizzled rows.
template <int BN>
__device__ __forceinline__ void mma_k_tile(uint32_t d_tmem, const bf16 *sa,
                                           const bf16 *sb, bool accumulate) {
  constexpr uint32_t idesc = lab::make_idesc_f16(kBM, BN, /*bf16=*/true);
#pragma unroll
  for (int kk = 0; kk < kBK / kUmmaK; ++kk) {
    const uint64_t da = lab::make_smem_desc_kmajor_sw128(sa + kk * kUmmaK);
    const uint64_t db = lab::make_smem_desc_kmajor_sw128(sb + kk * kUmmaK);
    lab::mma_f16_ss(d_tmem, da, db, idesc, accumulate || kk > 0);
  }
}

// Shared-memory layout and barriers of one CTA.
template <class Cfg> struct Smem {
  uint8_t *stages; // kStages x (A [128][64], B [BN][64]) BF16, 1024-aligned
  float *staging;  // kEpilogueWarps x 2 boxes of 32 x 32 FP32
  uint64_t *full, *empty, *acc_full, *acc_empty;
  __device__ bf16 *a(int s) const {
    return reinterpret_cast<bf16 *>(stages + s * Cfg::kStageBytes);
  }
  __device__ bf16 *b(int s) const {
    return a(s) + kBM * kBK;
  }
};

// Position in a ring of N slots: slot index and the parity of the current
// round. Every role keeps its own copy and advances it once per use, so the
// copies stay in step without any communication.
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

// ---------------------------------------------------------------------------
// Producer (one lane of warp 4): the K tiles of one output tile.
// A slot is refilled only after the MMAs that read its previous contents
// have completed (empty[s], parity phase ^ 1: on the first round the
// "previous phase" of a fresh barrier counts as complete).
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
produce_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring,
             const CUtensorMap *a_map, const CUtensorMap *b_map, int m0, int n0,
             int k_tiles) {
  for (int t = 0; t < k_tiles; ++t) {
    lab::mbar_wait(&sm.empty[ring.index], ring.phase ^ 1);
    lab::mbar_arrive_expect_tx(&sm.full[ring.index], Cfg::kStageBytes);
    lab::tma_load_2d(sm.a(ring.index), a_map, &sm.full[ring.index], t * kBK,
                     m0);
    lab::tma_load_2d(sm.b(ring.index), b_map, &sm.full[ring.index], t * kBK,
                     n0);
    ring.advance();
  }
}

// ---------------------------------------------------------------------------
// MMA issuer (one lane of warp 5): the K loop of one output tile into the
// accumulator at TMEM column `acc_tmem`. Never waits for an MMA to finish:
// commit(empty[s]) hands stage s back to the producer asynchronously, and
// commit(acc_done) tells the epilogue when the whole tile is done.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
mma_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring, uint32_t acc_tmem,
         uint64_t *acc_done, int k_tiles) {
  for (int t = 0; t < k_tiles; ++t) {
    lab::mbar_wait(&sm.full[ring.index], ring.phase);
    lab::tcgen05_fence_after_thread_sync();
    mma_k_tile<Cfg::kBN>(acc_tmem, sm.a(ring.index), sm.b(ring.index), t > 0);
    lab::tcgen05_commit(&sm.empty[ring.index]);
    ring.advance();
  }
  lab::tcgen05_commit(acc_done);
}

// ---------------------------------------------------------------------------
// Epilogue (warps 0-3, all lanes): drain one accumulator to C.
// Warp w reads TMEM lanes 32w.. (rows m0 + 32w + lane) in 32-column blocks
// and writes each block through a 128B-swizzled staging box with a TMA
// store (lab 31 step 3). `acc_free`, if not null, is arrived on as soon as
// this thread's last tcgen05.ld has completed: from then on the MMA warp may
// overwrite the accumulator while we are still storing.
// `box` alternates between the warp's two staging boxes, across tiles.
// ---------------------------------------------------------------------------
template <int BN>
__device__ __forceinline__ void
epilogue_tile(uint32_t acc_tmem, float *warp_staging, int &box,
              uint64_t *acc_free, const CUtensorMap *c_map, int m0, int n0) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll 1
  for (int j = 0; j < BN / 32; ++j) {
    float v[32];
    // tcgen05.ld + wait::ld in one asm block: v is valid on return.
    lab::tmem_ld_32x32b_x32_sync(lab::tmem_addr(acc_tmem, warp * 32, j * 32),
                                 v);
    if (j == BN / 32 - 1 && acc_free != nullptr) {
      // Order our completed TMEM reads before the arrival (lab 31 §4.7);
      // the MMA warp pairs this with fence::after_thread_sync.
      lab::tcgen05_fence_before_thread_sync();
      lab::mbar_arrive(acc_free);
    }
    // Box `box` was last used two stores ago: wait until that store has
    // read it (at most one younger bulk group may still be reading).
    if (lane == 0)
      lab::bulk_wait_group_read<1>();
    __syncwarp();
    float4 *dst = reinterpret_cast<float4 *>(warp_staging + box * kBoxFloats);
#pragma unroll
    for (int q = 0; q < 8; ++q)
      dst[lane * 8 + (q ^ (lane % 8))] =
          make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    lab::fence_proxy_async(); // generic writes -> visible to the TMA store
    __syncwarp();
    if (lane == 0) {
      lab::tma_store_2d(c_map, dst, n0 + j * 32, m0 + warp * 32);
      lab::bulk_commit_group();
    }
    box ^= 1;
  }
}

// Barrier initialization and TMEM allocation, shared by both kernels.
template <class Cfg>
__device__ __forceinline__ uint32_t setup(Smem<Cfg> &sm, uint32_t *tmem_slot,
                                          const CUtensorMap *a_map,
                                          const CUtensorMap *b_map,
                                          const CUtensorMap *c_map) {
  const int warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < Cfg::kStages; ++s) {
      lab::mbar_init(&sm.full[s], 1);  // producer's arrive_expect_tx (+ bytes)
      lab::mbar_init(&sm.empty[s], 1); // MMA warp's tcgen05.commit
    }
    for (int a = 0; a < Cfg::kAccStages; ++a) {
      lab::mbar_init(&sm.acc_full[a], 1); // MMA warp's tcgen05.commit
      lab::mbar_init(&sm.acc_empty[a], kEpilogueThreads); // every epi thread
    }
    lab::fence_barrier_init();
  }
  if (warp == kProducerWarp && lab::elect_one()) {
    lab::prefetch_tensor_map(a_map);
    lab::prefetch_tensor_map(b_map);
    lab::prefetch_tensor_map(c_map);
  }
  if (warp == kMmaWarp) {
    __syncwarp(); // .sync.aligned: the whole warp must be converged
    lab::tmem_alloc(tmem_slot, Cfg::kTmemCols);
    lab::tmem_relinquish_alloc_permit();
  }
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  lab::tcgen05_fence_after_thread_sync();
  return *tmem_slot;
}

// Every MMA has completed (the epilogue waited for acc_full) and every
// tcgen05.ld has completed (wait_ld): order them before the dealloc.
template <class Cfg> __device__ __forceinline__ void teardown(uint32_t tmem) {
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (threadIdx.x / 32 == kMmaWarp) {
    lab::tcgen05_fence_after_thread_sync();
    lab::tmem_dealloc(tmem, Cfg::kTmemCols);
  }
}

#define LAB32_SMEM_DECL(Cfg)                                                   \
  extern __shared__ uint8_t smem_raw[];                                        \
  __shared__ __align__(8) uint64_t bar_full[Cfg::kStages],                     \
      bar_empty[Cfg::kStages], bar_acc_full[Cfg::kAccStages],                  \
      bar_acc_empty[Cfg::kAccStages];                                          \
  __shared__ uint32_t tmem_slot;                                               \
  Smem<Cfg> sm;                                                                \
  sm.stages = align_1024(smem_raw);                                            \
  sm.staging =                                                                 \
      reinterpret_cast<float *>(sm.stages + Cfg::kStages * Cfg::kStageBytes);  \
  sm.full = bar_full;                                                          \
  sm.empty = bar_empty;                                                        \
  sm.acc_full = bar_acc_full;                                                  \
  sm.acc_empty = bar_acc_empty

// ===========================================================================
// Step 1: warp-specialized, one output tile per CTA.
// ===========================================================================
template <class Cfg>
__global__ void __launch_bounds__(kThreads, 1)
    gemm_ws_tile(const __grid_constant__ CUtensorMap a_map,
                 const __grid_constant__ CUtensorMap b_map,
                 const __grid_constant__ CUtensorMap c_map, int k) {
  LAB32_SMEM_DECL(Cfg);
  const uint32_t tmem = setup(sm, &tmem_slot, &a_map, &b_map, &c_map);
  const int warp = threadIdx.x / 32;
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * Cfg::kBN;
  const int k_tiles = (k + kBK - 1) / kBK;

  if (warp == kProducerWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      produce_tile(sm, ring, &a_map, &b_map, m0, n0, k_tiles);
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      mma_tile(sm, ring, tmem, &sm.acc_full[0], k_tiles);
    }
    __syncwarp();
  } else {
    lab::mbar_wait(&sm.acc_full[0], 0);
    lab::tcgen05_fence_after_thread_sync();
    int box = 0;
    epilogue_tile<Cfg::kBN>(tmem, sm.staging + warp * 2 * kBoxFloats, box,
                            nullptr, &c_map, m0, n0);
    // Shared memory disappears when the CTA exits: the stores must have
    // read their staging boxes first.
    if (threadIdx.x % 32 == 0)
      lab::bulk_wait_group_read<0>();
    __syncwarp();
  }
  teardown<Cfg>(tmem);
}

// ===========================================================================
// Steps 2 and 3: persistent CTAs, double-buffered accumulator.
// ===========================================================================

// Static tile schedule: CTA b handles tiles b, b + gridDim.x, ... Every role
// walks the same sequence, so they agree on (m0, n0) without communicating.
template <class Cfg>
__device__ __forceinline__ void tile_coords(int tile, int m_tiles, int n_tiles,
                                            int &m0, int &n0) {
  int mb, nb;
  if constexpr (Cfg::kGroupM == 0) {
    mb = tile / n_tiles; // row-major: neighbours share an A row panel
    nb = tile % n_tiles;
  } else {
    // Groups of kGroupM tile rows; inside a group, walk down a column of
    // kGroupM tiles before moving right. 148 consecutive tiles then cover
    // a ~kGroupM x (148 / kGroupM) block of C, which reads far fewer A and B
    // panels than 148 tiles of one long row.
    const int per_group = Cfg::kGroupM * n_tiles;
    const int group = tile / per_group;
    const int first_m = group * Cfg::kGroupM;
    const int rows = min(Cfg::kGroupM, m_tiles - first_m);
    const int in_group = tile - group * per_group;
    mb = first_m + in_group % rows;
    nb = in_group / rows;
  }
  m0 = mb * kBM;
  n0 = nb * Cfg::kBN;
}

template <class Cfg>
__global__ void __launch_bounds__(kThreads, 1)
    gemm_ws_persistent(const __grid_constant__ CUtensorMap a_map,
                       const __grid_constant__ CUtensorMap b_map,
                       const __grid_constant__ CUtensorMap c_map, int m, int n,
                       int k) {
  LAB32_SMEM_DECL(Cfg);
  const uint32_t tmem = setup(sm, &tmem_slot, &a_map, &b_map, &c_map);
  const int warp = threadIdx.x / 32;
  const int m_tiles = (m + kBM - 1) / kBM;
  const int n_tiles = (n + Cfg::kBN - 1) / Cfg::kBN;
  const int num_tiles = m_tiles * n_tiles;
  const int k_tiles = (k + kBK - 1) / kBK;

  if (warp == kProducerWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring; // runs on across tiles
      for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int m0, n0;
        tile_coords<Cfg>(tile, m_tiles, n_tiles, m0, n0);
        produce_tile(sm, ring, &a_map, &b_map, m0, n0, k_tiles);
      }
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    if (lab::elect_one()) {
      RingPos<Cfg::kStages> ring;
      RingPos<Cfg::kAccStages> acc;
      for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        // The epilogue must have read the previous contents of this
        // accumulator (parity ^ 1: free on the first round).
        lab::mbar_wait(&sm.acc_empty[acc.index], acc.phase ^ 1);
        lab::tcgen05_fence_after_thread_sync();
        mma_tile(sm, ring, tmem + acc.index * Cfg::kBN, &sm.acc_full[acc.index],
                 k_tiles);
        acc.advance();
      }
    }
    __syncwarp();
  } else {
    RingPos<Cfg::kAccStages> acc;
    int box = 0;
    float *warp_staging = sm.staging + warp * 2 * kBoxFloats;
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      int m0, n0;
      tile_coords<Cfg>(tile, m_tiles, n_tiles, m0, n0);
      lab::mbar_wait(&sm.acc_full[acc.index], acc.phase);
      lab::tcgen05_fence_after_thread_sync();
      epilogue_tile<Cfg::kBN>(tmem + acc.index * Cfg::kBN, warp_staging, box,
                              &sm.acc_empty[acc.index], &c_map, m0, n0);
      acc.advance();
    }
    if (threadIdx.x % 32 == 0)
      lab::bulk_wait_group_read<0>();
    __syncwarp();
  }
  teardown<Cfg>(tmem);
}

using Step1 = Config<256, 4, 1, 0>;
using Step2 = Config<256, 4, 2, 0>;
// GroupM = 8: on `wide` (A + B = 128 MB > L2) it cuts DRAM reads ~2.5x and
// time by ~6%; on `square` / `flat` (A + B fit in L2) it costs 1-2% against
// row-major. Larger groups help `wide` more and hurt the others more
// (NOTES.md).
using Step3 = Config<256, 4, 2, 8>;

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

template <class Cfg> int num_tiles(const lab::GemmProblem &p) {
  return ((p.m + kBM - 1) / kBM) * ((p.n + Cfg::kBN - 1) / Cfg::kBN);
}

} // namespace

struct Plan {
  CUtensorMap a_map, b_map, c_map;
  int num_sms;
};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::BF16, "this lab uses BF16 A and B");
  LAB_REQUIRE(p.c_dtype == lab::DType::F32, "this lab writes an FP32 C");
  LAB_REQUIRE(p.k % 8 == 0, "K must be a multiple of 8 (TMA: 16-byte rows)");
  LAB_REQUIRE(p.n % 4 == 0, "N must be a multiple of 4 (TMA: 16-byte rows)");
  Plan plan{};
  allow_smem(gemm_ws_tile<Step1>, Step1::kSmemBytes);
  allow_smem(gemm_ws_persistent<Step2>, Step2::kSmemBytes);
  allow_smem(gemm_ws_persistent<Step3>, Step3::kSmemBytes);
  int dev = 0;
  LAB_CUDA_CHECK(cudaGetDevice(&dev));
  LAB_CUDA_CHECK(cudaDeviceGetAttribute(&plan.num_sms,
                                        cudaDevAttrMultiProcessorCount, dev));
  static_assert(Step1::kBN == Step2::kBN && Step2::kBN == Step3::kBN,
                "one B box shape for all steps");
  plan.a_map = operand_map(p.a, p.m, p.k, kBM);
  plan.b_map = operand_map(p.b, p.n, p.k, Step3::kBN);
  plan.c_map =
      lab::make_tensor_map_2d(p.c, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, p.m, p.n,
                              32, 32, CU_TENSOR_MAP_SWIZZLE_128B);
  return plan;
}

template <class Cfg>
void launch_persistent(const Plan &plan, const lab::GemmProblem &p,
                       cudaStream_t stream) {
  const int grid = std::min(num_tiles<Cfg>(p), plan.num_sms);
  gemm_ws_persistent<Cfg><<<grid, kThreads, Cfg::kSmemBytes, stream>>>(
      plan.a_map, plan.b_map, plan.c_map, p.m, p.n, p.k);
}

void run(const Plan &plan, const lab::GemmProblem &p, cudaStream_t stream) {
  switch (p.step) {
  case 1: {
    const dim3 grid((p.n + Step1::kBN - 1) / Step1::kBN, (p.m + kBM - 1) / kBM);
    gemm_ws_tile<Step1><<<grid, kThreads, Step1::kSmemBytes, stream>>>(
        plan.a_map, plan.b_map, plan.c_map, p.k);
    break;
  }
  case 2:
    launch_persistent<Step2>(plan, p, stream);
    break;
  default:
    launch_persistent<Step3>(plan, p, stream);
    break;
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
