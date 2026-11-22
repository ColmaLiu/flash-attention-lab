// 32_ws_gemm: C[M,N] = A[M,K] . B[N,K]^T with tcgen05.mma, split into
// warp-specialized roles that talk only through mbarriers.
// Lesson: course/32_ws_gemm/README.md
//
// A is [M, K], B is [N, K], both BF16, row-major (K-major); C is [M, N] FP32.
// Supported (as in lab 31): K % 8 == 0 and N % 4 == 0; M arbitrary. Tails
// need no code: TMA loads zero-fill, TMA stores clip.
//
// One CTA = 6 warps (192 threads):
//   warps 0-3  epilogue  TMEM -> registers -> swizzled smem -> TMA store
//                        (warp w may only read TMEM lanes 32w..32w+31)
//   warp  4    producer  one elected lane issues every TMA load
//   warp  5    MMA       allocates TMEM; one elected lane issues every
//                        tcgen05.mma / tcgen05.commit
// Barriers (lesson §4.2):
//   full[s]      producer -> MMA       stage s holds A/B   (expect_tx + TMA)
//   empty[s]     MMA -> producer       stage s was read    (tcgen05.commit)
//   acc_full[a]  MMA -> epilogue       accumulator a final (tcgen05.commit)
//   acc_empty[a] epilogue -> MMA       accumulator a read  (128 arrivals)
// run() picks a version by p.step:  ./lab bench 32 --step 1 / 2 / 3
//
// mbarrier, TMA and tcgen05 helpers come from include/lab/ (labs 30, 31).
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
// have completed.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
produce_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring,
             const CUtensorMap *a_map, const CUtensorMap *b_map, int m0, int n0,
             int k_tiles) {
  // -------------------------------------------------------------------------
  // TODO(1a) — producer (lesson §5.2). For each K tile t = 0 .. k_tiles - 1,
  // in slot s = ring.index:
  //   - wait until the slot is free: empty[s] with which parity? (On the
  //     first round nobody has consumed anything yet -- lab 30 §4.)
  //   - arm full[s] with Cfg::kStageBytes and TMA-load the A box at
  //     (x = 64t, y = m0) into sm.a(s) and the B box at (x = 64t, y = n0)
  //     into sm.b(s), both completing on full[s];
  //   - ring.advance().
  // Do not reset `ring`: in step 2 it continues into the next tile.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// MMA issuer (one lane of warp 5): the K loop of one output tile into the
// accumulator at TMEM column `acc_tmem`. Never waits for an MMA to finish.
// ---------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void
mma_tile(const Smem<Cfg> &sm, RingPos<Cfg::kStages> &ring, uint32_t acc_tmem,
         uint64_t *acc_done, int k_tiles) {
  // -------------------------------------------------------------------------
  // TODO(1b) — MMA issuer (lesson §5.2). For each K tile t, in slot
  // s = ring.index:
  //   - wait for the TMA data: full[s], parity ring.phase;
  //     tcgen05_fence_after_thread_sync();
  //   - mma_k_tile<Cfg::kBN>(acc_tmem, sm.a(s), sm.b(s), accumulate = ?);
  //   - hand the slot back WITHOUT waiting for the MMAs: which instruction
  //     arrives on empty[s] once they have read it?
  //   - ring.advance().
  // After the loop, the same mechanism on acc_done tells the epilogue that
  // the accumulator is final. This thread never waits for an MMA.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
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
      // ---------------------------------------------------------------------
      // TODO(2d) — release the accumulator (lesson §5.3). This thread's last
      // TMEM read has completed. Tell the MMA warp (acc_free, initialized
      // with 128 arrivals) that it may overwrite the accumulator -- with the
      // tcgen05 fence that orders TMEM reads before a thread sync
      // (lab 31 §4.7). The stores below may still be running; why is it
      // safe to release now?
      // ---------------------------------------------------------------------
      LAB_TODO(2);
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

// The mbarriers and the TMEM address slot: small, so they live in static
// shared memory (8-byte aligned through the uint64_t members). Each kernel
// declares one:  __shared__ Barriers<Cfg> bars;
template <class Cfg> struct Barriers {
  uint64_t full[Cfg::kStages], empty[Cfg::kStages];
  uint64_t acc_full[Cfg::kAccStages], acc_empty[Cfg::kAccStages];
  uint32_t tmem_slot; // tcgen05.alloc writes the TMEM base address here
};

// Carve the dynamic shared memory: stage ring first (1024-aligned for the
// 128B swizzle), then the epilogue staging boxes.
template <class Cfg>
__device__ __forceinline__ Smem<Cfg> carve(uint8_t *smem_raw,
                                           Barriers<Cfg> &bars) {
  Smem<Cfg> sm;
  sm.stages = align_1024(smem_raw);
  sm.staging =
      reinterpret_cast<float *>(sm.stages + Cfg::kStages * Cfg::kStageBytes);
  sm.full = bars.full;
  sm.empty = bars.empty;
  sm.acc_full = bars.acc_full;
  sm.acc_empty = bars.acc_empty;
  return sm;
}

// ===========================================================================
// Step 1: warp-specialized, one output tile per CTA.
// ===========================================================================
template <class Cfg>
__global__ void __launch_bounds__(kThreads, 1)
    gemm_ws_tile(const __grid_constant__ CUtensorMap a_map,
                 const __grid_constant__ CUtensorMap b_map,
                 const __grid_constant__ CUtensorMap c_map, int k) {
  extern __shared__ uint8_t smem_raw[]; // stages + staging (dynamic)
  __shared__ Barriers<Cfg> bars;
  Smem<Cfg> sm = carve(smem_raw, bars);
  const uint32_t tmem = setup(sm, &bars.tmem_slot, &a_map, &b_map, &c_map);
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
    // -----------------------------------------------------------------------
    // TODO(1c) — epilogue warps (lesson §5.2): wait until the accumulator
    // is final, order the TMEM reads after that wait (lab 31 §4.7), then
    // drain it with epilogue_tile (read its comment: which staging boxes
    // belong to this warp, and what is acc_free in step 1?).
    // -----------------------------------------------------------------------
    LAB_TODO(1);
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
    // -----------------------------------------------------------------------
    // TODO(3a) — grouped rasterization (lesson §5.4). Cut the tile rows
    // into groups of Cfg::kGroupM rows (the last group may be shorter:
    // m_tiles is not always a multiple). Number the tiles group by group;
    // inside a group go DOWN first (consecutive tiles: same nb, mb + 1),
    // then to the next column. Example: m_tiles = 5, n_tiles = 3,
    // kGroupM = 2 gives the order
    //   (0,0) (1,0) (0,1) (1,1) (0,2) (1,2) (2,0) (3,0) ... (4,0) (4,1) (4,2)
    // Set mb, nb. Every tile index 0 .. m_tiles * n_tiles - 1 must map to a
    // distinct tile.
    // -----------------------------------------------------------------------
    mb = nb = 0;
    LAB_TODO(3);
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
  extern __shared__ uint8_t smem_raw[]; // stages + staging (dynamic)
  __shared__ Barriers<Cfg> bars;
  Smem<Cfg> sm = carve(smem_raw, bars);
  const uint32_t tmem = setup(sm, &bars.tmem_slot, &a_map, &b_map, &c_map);
  const int warp = threadIdx.x / 32;
  const int m_tiles = (m + kBM - 1) / kBM;
  const int n_tiles = (n + Cfg::kBN - 1) / Cfg::kBN;
  const int num_tiles = m_tiles * n_tiles;
  const int k_tiles = (k + kBK - 1) / kBK;

  // Every role walks the same static schedule:
  //   for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
  // and gets the tile origin from tile_coords<Cfg>(tile, m_tiles, n_tiles,
  // m0, n0). Accumulator a lives at TMEM columns tmem + a * Cfg::kBN.
  if (warp == kProducerWarp) {
    if (lab::elect_one()) {
      // ---------------------------------------------------------------------
      // TODO(2a) — persistent producer (lesson §5.3): one RingPos for the
      // whole kernel, produce_tile for every tile of this CTA.
      // ---------------------------------------------------------------------
      LAB_TODO(2);
    }
    __syncwarp();
  } else if (warp == kMmaWarp) {
    if (lab::elect_one()) {
      // ---------------------------------------------------------------------
      // TODO(2b) — persistent MMA issuer (lesson §5.3). Keep a
      // RingPos<Cfg::kStages> for the smem ring and a
      // RingPos<Cfg::kAccStages> for the accumulators. Per tile:
      //   - wait until the epilogue has drained accumulator acc.index:
      //     acc_empty, which parity? (Both accumulators start empty.)
      //   - tcgen05_fence_after_thread_sync (the epilogue's TMEM reads
      //     happen before our MMAs overwrite it);
      //   - mma_tile into that accumulator, signalling acc_full[acc.index];
      //   - acc.advance().
      // ---------------------------------------------------------------------
      LAB_TODO(2);
    }
    __syncwarp();
  } else {
    int box = 0; // alternates between this warp's two staging boxes
    float *warp_staging = sm.staging + warp * 2 * kBoxFloats;
    // -----------------------------------------------------------------------
    // TODO(2c) — persistent epilogue (lesson §5.3): the step-1 epilogue
    // inside the tile loop, for accumulator acc.index of a
    // RingPos<Cfg::kAccStages> that runs on across tiles. Which parity?
    // This time epilogue_tile must release the accumulator (TODO(2d)).
    // -----------------------------------------------------------------------
    (void)box;
    (void)warp_staging;
    LAB_TODO(2);
    if (threadIdx.x % 32 == 0)
      lab::bulk_wait_group_read<0>();
    __syncwarp();
  }
  teardown<Cfg>(tmem);
}

using Step1 = Config<256, 4, 1, 0>;
using Step2 = Config<256, 4, 2, 0>;
// ---------------------------------------------------------------------------
// TODO(3b) — choose kGroupM for step 3 (lesson §6 step 3): start at 16, try
// 4, 8, 32 on `square`, `wide` and `flat` (step 2 = row-major). Stages
// and BN are already at the shared-memory / TMEM limit (lesson §5.5).
// ---------------------------------------------------------------------------
using Step3 = Config<256, 4, 2, 16>;

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
  LAB_REQUIRE(p.k > 0, "K must be positive (K = 0 would store an "
                       "accumulator no MMA ever wrote)");
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
