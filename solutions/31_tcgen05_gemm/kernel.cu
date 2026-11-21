// Reference solution for 31_tcgen05_gemm: C[M,N] = A[M,K] . B[N,K]^T with
// tcgen05.mma (accumulator in Tensor Memory) fed by TMA.
//
// step 1: one 128 x 128 x 64 tile. Thread 0 TMA-loads A and B (128B swizzle),
//         waits on the mbarrier, issues 4 tcgen05.mma (K = 16 each) and
//         commits them to a second mbarrier. Every warp then reads its 32
//         TMEM lanes (= 32 rows of C) with tcgen05.ld and stores them.
// step 2: the full K loop with a ring of kStages shared-memory stages. Still
//         one thread for both roles: it keeps kStages - 1 TMA loads ahead of
//         the MMAs; tcgen05.commit to empty[s] tells it when an MMA has
//         finished reading stage s so the stage can be refilled.
// step 3: a grid of CTAs over the M x N tiles, a 128 x 256 tile, and an
//         epilogue that stages each warp's 32 x 32 piece of C in 128B-swizzled
//         shared memory and writes it with a TMA store (which also clips the
//         M and N tails).
//
// Supported domain: BF16 A/B, FP32 C, K % 8 == 0 and N % 4 == 0 (TMA needs
// 16-byte row pitches for A, B and C); M arbitrary. Tails: TMA loads fill
// out-of-bounds rows and K columns with zeros, which add nothing to the dot
// products; TMA stores skip out-of-bounds elements.
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kBM = 128;      // MMA M: one TMEM lane per row of C
constexpr int kBK = 64;       // 64 BF16 = 128 bytes: one 128B swizzle row
constexpr int kUmmaK = 16;    // K of one tcgen05.mma for 16-bit inputs
constexpr int kThreads = 128; // 4 warps = the 128 TMEM lanes

// Shared memory starts on a 1024-byte boundary: the 128B swizzle atom is
// 8 rows x 128 B and both TMA and the MMA descriptor assume it is aligned.
__device__ __forceinline__ uint8_t *align_1024(uint8_t *p) {
  return p + ((1024 - (lab::shared_addr(p) & 1023)) & 1023);
}

// D (TMEM) [+]= A_tile[128 x 64] . B_tile[bn x 64]^T as 4 MMAs of K = 16.
// Inside the 128-byte swizzled row, K slice kk starts kk * 16 * 2 bytes
// further: advancing the descriptor's start address by 32 B selects it (the
// hardware applies the swizzle to the full address).
template <int kBN>
__device__ __forceinline__ void mma_k_tile(uint32_t d_tmem, const bf16 *sa,
                                           const bf16 *sb, bool accumulate) {
  constexpr uint32_t idesc = lab::make_idesc_f16(kBM, kBN, /*bf16=*/true);
#pragma unroll
  for (int kk = 0; kk < kBK / kUmmaK; ++kk) {
    const uint64_t da = lab::make_smem_desc_kmajor_sw128(sa + kk * kUmmaK);
    const uint64_t db = lab::make_smem_desc_kmajor_sw128(sb + kk * kUmmaK);
    lab::mma_f16_ss(d_tmem, da, db, idesc, accumulate || kk > 0);
  }
}

// Epilogue of steps 1-2: warp w owns TMEM lanes 32w..32w+31 = rows
// m0 + 32w + lane. Each thread holds 32 consecutive columns of ONE row per
// tcgen05.ld, and stores them directly (simple; neighbouring threads write
// different rows, so the stores are not coalesced -- step 3 fixes that).
template <int kBN>
__device__ __forceinline__ void store_direct(uint32_t tmem, float *c, int m,
                                             int n, int m0, int n0) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = m0 + warp * 32 + lane;
#pragma unroll 1
  for (int j = 0; j < kBN; j += 32) {
    float v[32];
    lab::tmem_ld_32x32b_x32(lab::tmem_addr(tmem, warp * 32, j), v);
    lab::tmem_wait_ld();
    if (row < m) {
#pragma unroll
      for (int i = 0; i < 32; ++i)
        if (n0 + j + i < n)
          c[static_cast<int64_t>(row) * n + n0 + j + i] = v[i];
    }
  }
}

// Allocation and release of the accumulator columns by warp 0, with the
// fence / barrier / fence sequence that orders TMEM use across threads.
__device__ __forceinline__ uint32_t setup_tmem(uint32_t *slot, int ncols) {
  if (threadIdx.x / 32 == 0) {
    __syncwarp(); // .sync.aligned: the whole warp must be converged
    lab::tmem_alloc(slot, ncols);
    lab::tmem_relinquish_alloc_permit();
  }
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  lab::tcgen05_fence_after_thread_sync();
  return *slot;
}

__device__ __forceinline__ void release_tmem(uint32_t tmem, int ncols) {
  // All tcgen05.ld are complete (wait_ld); order them before the dealloc.
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    lab::tcgen05_fence_after_thread_sync();
    lab::tmem_dealloc(tmem, ncols);
  }
}

// ===========================================================================
// Step 1: a single 128 x 128 x 64 tile (M <= 128, N <= 128, K <= 64).
// ===========================================================================
constexpr int kOneBN = 128;
constexpr int kOneSmem = 1024 + (kBM + kOneBN) * kBK * 2;

__global__ void __launch_bounds__(kThreads)
    gemm_one_tile(const __grid_constant__ CUtensorMap a_map,
                  const __grid_constant__ CUtensorMap b_map, float *c, int m,
                  int n) {
  extern __shared__ uint8_t smem_raw[];
  bf16 *sa = reinterpret_cast<bf16 *>(align_1024(smem_raw)); // [128][64]
  bf16 *sb = sa + kBM * kBK;                                 // [128][64]
  __shared__ __align__(8) uint64_t full, mma_done;
  __shared__ uint32_t tmem_slot;

  if (threadIdx.x == 0) {
    lab::mbar_init(&full, 1);
    lab::mbar_init(&mma_done, 1);
    lab::fence_barrier_init();
  }
  const uint32_t tmem =
      setup_tmem(&tmem_slot, kOneBN); // includes __syncthreads

  if (threadIdx.x == 0) {
    lab::mbar_arrive_expect_tx(&full, (kBM + kOneBN) * kBK * 2);
    lab::tma_load_2d(sa, &a_map, &full, 0, 0); // (x = k, y = row)
    lab::tma_load_2d(sb, &b_map, &full, 0, 0);
    lab::mbar_wait(&full, 0);
    // Defensive: TMA and MMA are both async proxy, ordered by the mbarrier;
    // CUTLASS (PipelineTmaUmmaAsync::consumer_wait) has no fence here.
    lab::tcgen05_fence_after_thread_sync();
    mma_k_tile<kOneBN>(tmem, sa, sb, /*accumulate=*/false);
    lab::tcgen05_commit(&mma_done); // arrives when the 4 MMAs are complete
  }
  __syncwarp(); // reconverge warp 0 before the .sync.aligned tcgen05.ld
  lab::mbar_wait(&mma_done, 0);
  lab::tcgen05_fence_after_thread_sync();
  store_direct<kOneBN>(tmem, c, m, n, 0, 0);
  release_tmem(tmem, kOneBN);
}

// ===========================================================================
// Steps 2 and 3: full K loop, ring of stages.
// ===========================================================================
// tcgen05.alloc takes a power of 2 between 32 and 512 columns.
constexpr int tmem_alloc_cols(int ncols) {
  int c = 32;
  while (c < ncols)
    c *= 2;
  return c;
}

template <int BN, int Stages, bool TmaEpilogue> struct Config {
  static constexpr int kBN = BN, kStages = Stages;
  static constexpr bool kTmaEpilogue = TmaEpilogue;
  static constexpr int kStageBytes = (kBM + BN) * kBK * 2;
  static constexpr int kTxBytes = kStageBytes; // bytes TMA delivers per stage
  static constexpr int kSmemBytes = 1024 + Stages * kStageBytes;
  // FP32 accumulator: one column per n, rounded up to a legal allocation
  // size (BN = 96 allocates 128 columns).
  static constexpr int kTmemCols = tmem_alloc_cols(BN);
  static_assert(BN % 32 == 0 && BN <= 256,
                "BN: multiple of 32 (32-column tcgen05.ld epilogue), <= 256");
  static_assert(!TmaEpilogue || kBM * BN * 4 <= Stages * kStageBytes,
                "the C staging area reuses the stage buffers");
};

template <class Cfg>
__global__ void __launch_bounds__(kThreads)
    gemm_pipelined(const __grid_constant__ CUtensorMap a_map,
                   const __grid_constant__ CUtensorMap b_map,
                   const __grid_constant__ CUtensorMap c_map, float *c, int m,
                   int n, int k) {
  constexpr int S = Cfg::kStages, BN = Cfg::kBN;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *stages = align_1024(smem_raw); // S x (A [128][64], B [BN][64])
  __shared__ __align__(8) uint64_t full[S], empty[S], mma_done;
  __shared__ uint32_t tmem_slot;
  auto stage_a = [&](int s) {
    return reinterpret_cast<bf16 *>(stages + s * Cfg::kStageBytes);
  };
  auto stage_b = [&](int s) { return stage_a(s) + kBM * kBK; };

  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * BN;
  const int k_tiles = (k + kBK - 1) / kBK;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      lab::mbar_init(&full[s], 1);  // TMA: arrive_expect_tx + bytes
      lab::mbar_init(&empty[s], 1); // tcgen05.commit after the stage's MMAs
    }
    lab::mbar_init(&mma_done, 1);
    lab::fence_barrier_init();
  }
  const uint32_t tmem = setup_tmem(&tmem_slot, Cfg::kTmemCols);

  if (threadIdx.x == 0) {
    lab::prefetch_tensor_map(&a_map);
    lab::prefetch_tensor_map(&b_map);
    auto load = [&](int t) { // K tile t -> stage t % S
      const int s = t % S;
      lab::mbar_arrive_expect_tx(&full[s], Cfg::kTxBytes);
      lab::tma_load_2d(stage_a(s), &a_map, &full[s], t * kBK, m0);
      lab::tma_load_2d(stage_b(s), &b_map, &full[s], t * kBK, n0);
    };
    // Prologue: fill every stage.
    for (int t = 0; t < S && t < k_tiles; ++t)
      load(t);
    for (int t = 0; t < k_tiles; ++t) {
      const int s = t % S;
      lab::mbar_wait(&full[s], (t / S) & 1);  // fill number t / S of stage s
      lab::tcgen05_fence_after_thread_sync(); // defensive, see step 1
      mma_k_tile<BN>(tmem, stage_a(s), stage_b(s), t > 0);
      lab::tcgen05_commit(&empty[s]); // stage s is free once these MMAs end
      // Refill the stage of tile t - 1 with tile t - 1 + S. MMA t is already
      // queued behind MMA t - 1, so the Tensor Core stays busy while we wait
      // for MMA t - 1 to release its operands.
      const int next = t - 1 + S;
      if (t >= 1 && next < k_tiles) {
        lab::mbar_wait(&empty[(t - 1) % S], ((t - 1) / S) & 1);
        load(next);
      }
    }
    lab::tcgen05_commit(&mma_done); // all MMAs of this tile complete
  }
  __syncwarp();
  lab::mbar_wait(&mma_done, 0);
  lab::tcgen05_fence_after_thread_sync();

  if constexpr (!Cfg::kTmaEpilogue) {
    store_direct<BN>(tmem, c, m, n, m0, n0);
  } else {
    // Every MMA has finished reading the stage buffers and every TMA load
    // into them was waited for: reuse them to stage C. Warp w, column block
    // j owns a 32 x 32 FP32 box (4 KB, 1024-aligned) laid out as TMA's 128B
    // swizzle expects: 16-byte chunk q of row r at chunk q ^ (r % 8). Lane
    // r writes row r; in each 8-lane wavefront the 8 chunks q ^ r are
    // distinct, so the float4 stores are bank-conflict free.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *staging = reinterpret_cast<float *>(stages);
#pragma unroll 1
    for (int j = 0; j < BN / 32; ++j) {
      float v[32];
      lab::tmem_ld_32x32b_x32(lab::tmem_addr(tmem, warp * 32, j * 32), v);
      lab::tmem_wait_ld();
      float4 *box =
          reinterpret_cast<float4 *>(staging + (warp * (BN / 32) + j) * 1024);
#pragma unroll
      for (int q = 0; q < 8; ++q)
        box[lane * 8 + (q ^ (lane % 8))] =
            make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
      lab::fence_proxy_async(); // generic writes -> visible to the TMA store
      __syncwarp();
      if (lane == 0)
        lab::tma_store_2d(&c_map, box, n0 + j * 32, m0 + warp * 32);
    }
    if (lane == 0) {
      lab::bulk_commit_group();
      // The staging buffers live in this CTA's shared memory: wait until the
      // stores have read them before the CTA may exit (as CUTLASS does; the
      // global writes complete before the kernel is considered finished).
      lab::bulk_wait_group_read<0>();
    }
  }
  release_tmem(tmem, Cfg::kTmemCols);
}

using Step2 = Config<128, 4, false>;
using Step3 = Config<256, 4, true>;

template <class Kernel> void allow_smem(Kernel kernel, int bytes) {
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

// K-major BF16 operand [rows, k]: box of box_rows rows x 64 K elements
// (128 bytes), 128B swizzle -- the canonical layout the MMA descriptor reads.
CUtensorMap operand_map(const void *base, int rows, int k, int box_rows) {
  return lab::make_tensor_map_2d(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rows,
                                 k, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
}

} // namespace

struct Plan {
  CUtensorMap a_map, b_map, c_map;
};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::BF16, "this lab uses BF16 A and B");
  LAB_REQUIRE(p.c_dtype == lab::DType::F32, "this lab writes an FP32 C");
  LAB_REQUIRE(p.k % 8 == 0, "K must be a multiple of 8 (TMA: 16-byte rows)");
  LAB_REQUIRE(p.n % 4 == 0, "N must be a multiple of 4 (TMA: 16-byte rows)");
  Plan plan{};
  int bn = 0;
  if (p.step == 1) {
    LAB_REQUIRE(p.m <= kBM && p.n <= kOneBN && p.k <= kBK,
                "step 1 handles a single 128 x 128 x 64 tile");
    allow_smem(gemm_one_tile, kOneSmem);
    bn = kOneBN;
  } else if (p.step == 2) {
    allow_smem(gemm_pipelined<Step2>, Step2::kSmemBytes);
    bn = Step2::kBN;
  } else {
    allow_smem(gemm_pipelined<Step3>, Step3::kSmemBytes);
    bn = Step3::kBN;
  }
  plan.a_map = operand_map(p.a, p.m, p.k, kBM);
  plan.b_map = operand_map(p.b, p.n, p.k, bn);
  // C: FP32 [m, n], 32 x 32 boxes (128-byte rows), 128B swizzle.
  plan.c_map =
      lab::make_tensor_map_2d(p.c, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, p.m, p.n,
                              32, 32, CU_TENSOR_MAP_SWIZZLE_128B);
  return plan;
}

template <class Cfg> dim3 grid_of(const lab::GemmProblem &p) {
  return dim3((p.n + Cfg::kBN - 1) / Cfg::kBN, (p.m + kBM - 1) / kBM);
}

void run(const Plan &plan, const lab::GemmProblem &p, cudaStream_t stream) {
  auto c = static_cast<float *>(p.c);
  switch (p.step) {
  case 1:
    gemm_one_tile<<<1, kThreads, kOneSmem, stream>>>(plan.a_map, plan.b_map, c,
                                                     p.m, p.n);
    break;
  case 2:
    gemm_pipelined<Step2>
        <<<grid_of<Step2>(p), kThreads, Step2::kSmemBytes, stream>>>(
            plan.a_map, plan.b_map, plan.c_map, c, p.m, p.n, p.k);
    break;
  default:
    gemm_pipelined<Step3>
        <<<grid_of<Step3>(p), kThreads, Step3::kSmemBytes, stream>>>(
            plan.a_map, plan.b_map, plan.c_map, c, p.m, p.n, p.k);
    break;
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
