// 31_tcgen05_gemm: C[M,N] = A[M,K] . B[N,K]^T with tcgen05.mma, the
// accumulator in Tensor Memory (TMEM), operands brought in by TMA.
// Lesson: course/31_tcgen05_gemm/README.md
//
// A is [M, K], B is [N, K], both BF16, row-major (K fastest, "K-major");
// C is [M, N] FP32, row-major.
// Supported: K % 8 == 0 and N % 4 == 0 (TMA needs 16-byte row pitches for A,
// B and C); M arbitrary. make_plan rejects anything else. Tails need no
// code: TMA loads fill out-of-bounds elements with zeros, TMA stores skip
// them.
// Shared tiles are what TMA writes with a 64-element (128-byte) box row and
// the 128B swizzle: [rows][64] BF16, 16-byte chunk q of row r stored at
// chunk q ^ (r % 8), each 8-row group a 1024-byte "swizzle atom".
// run() picks a version by p.step:  ./lab bench 31 --step 2 ... --step 3
//
// mbarrier and TMA helpers come from include/lab/mbarrier.cuh and
// include/lab/tma.cuh (what you wrote in lab 30).
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tma.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

// ===========================================================================
// tcgen05 / TMEM primitives. You write them here; from lab 32 on they come
// from include/lab/tcgen05.cuh. One PTX instruction (or one descriptor word)
// each; shared addresses via lab::shared_addr(ptr) ("r" operand), 64-bit
// descriptors are "l" operands, TMEM addresses and idesc are "r".
// "memory" clobber on everything except tcgen05.ld (its outputs are
// registers; tmem_wait_ld carries the clobber).
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO(1a) — TMEM allocation (lesson §4.1). Executed by one whole warp.
//   tmem_alloc:  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], ncols;
//                ncols: power of 2, 32..512; the TMEM base address is
//                written to shared *dst.
//   tmem_relinquish_alloc_permit:
//                tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
//   tmem_dealloc: tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, ncols;
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tmem_alloc(uint32_t *dst, uint32_t ncols) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  LAB_TODO(1);
}

// TMEM address of (lane, column) relative to an allocated base (given).
__device__ __forceinline__ uint32_t tmem_addr(uint32_t base, int lane, int col) {
  return base + (static_cast<uint32_t>(lane) << 16) + static_cast<uint32_t>(col);
}

// ---------------------------------------------------------------------------
// TODO(1b) — descriptors (lesson §4.3, §4.4; bit tables there).
//   make_smem_desc: shared-memory descriptor of a K-major, 128B-swizzled
//     operand starting at `smem`: start address >> 4, LBO = 16 bytes
//     (unused), SBO = 1024 bytes (distance between 8-row swizzle atoms),
//     version 1 (bits 46-47), layout SWIZZLE_128B = 2 (bits 61-63).
//   make_idesc: instruction descriptor for .kind::f16 with BF16 A and B,
//     FP32 D, both K-major, M = m, N = n.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc(const void *smem) {
  LAB_TODO(1);
  return 0;
}
__device__ __forceinline__ uint32_t make_idesc(int m, int n) {
  LAB_TODO(1);
  return 0;
}

// ---------------------------------------------------------------------------
// TODO(1c) — MMA, commit, TMEM loads, fences (lesson §4.5–§4.7).
//   mma_f16_ss: tcgen05.mma.cta_group::1.kind::f16 [d], a_desc, b_desc, idesc, p;
//     p is a predicate: true -> D += A.B^T, false -> D = A.B^T (set it from
//     `accumulate` with setp inside the asm block).
//   tcgen05_commit: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];
//   tmem_ld_32x32b_x32: tcgen05.ld.sync.aligned.32x32b.x32.b32 {32 regs}, [taddr];
//     ("=f" outputs v[0..31]).
//   tmem_wait_ld: tcgen05.wait::ld.sync.aligned;
//   tcgen05_fence_before_thread_sync: tcgen05.fence::before_thread_sync;
//   tcgen05_fence_after_thread_sync:  tcgen05.fence::after_thread_sync;
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_f16_ss(uint32_t d_tmem, uint64_t a_desc,
                                           uint64_t b_desc, uint32_t idesc,
                                           bool accumulate) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tcgen05_commit(uint64_t *bar) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr,
                                                   float (&v)[32]) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tmem_wait_ld() {
  LAB_TODO(1);
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  LAB_TODO(1);
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  LAB_TODO(1);
}

constexpr int kBM = 128;  // MMA M: one TMEM lane per row of C
constexpr int kBK = 64;   // 64 BF16 = 128 bytes: one 128B swizzle row
constexpr int kUmmaK = 16; // K of one tcgen05.mma for 16-bit inputs
constexpr int kThreads = 128; // 4 warps = the 128 TMEM lanes

__device__ __forceinline__ uint8_t *align_1024(uint8_t *p) {
  return p + ((1024 - (lab::shared_addr(p) & 1023)) & 1023);
}

// ---------------------------------------------------------------------------
// TODO(1d) — D (TMEM) [+]= A_tile[128 x 64] . B_tile[kBN x 64]^T as
// kBK / kUmmaK = 4 MMAs, K slice kk = 0..3 covering K elements
// 16kk .. 16kk + 15 (lesson §4.4: how far do you move the descriptors'
// start address for the next slice, and why is that correct although the
// rows are swizzled?). The first MMA accumulates only if `accumulate`.
// ---------------------------------------------------------------------------
template <int kBN>
__device__ __forceinline__ void mma_k_tile(uint32_t d_tmem, const bf16 *sa,
                                           const bf16 *sb, bool accumulate) {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(1e) — epilogue of steps 1-2: copy the accumulator rows owned by this
// warp to C (lesson §4.6, §5.1). Warp w may read TMEM lanes 32w..32w+31,
// which hold rows m0 + 32w + lane of C; columns 0..kBN-1 of the tile.
// For each block of 32 columns: tmem_ld_32x32b_x32, tmem_wait_ld, then
// store this thread's 32 values to C row (m0 + 32w + lane), columns
// n0 + j .. n0 + j + 31 -- only those inside the m x n matrix.
// ---------------------------------------------------------------------------
template <int kBN>
__device__ __forceinline__ void store_direct(uint32_t tmem, float *c, int m,
                                             int n, int m0, int n0) {
  LAB_TODO(1);
}

// Warp 0 allocates the accumulator columns; the fence / barrier / fence
// sequence makes the address (written to shared memory by the tensor core
// unit) and the allocation visible to all threads (lesson §4.1, §4.7).
__device__ __forceinline__ uint32_t setup_tmem(uint32_t *slot, int ncols) {
  if (threadIdx.x / 32 == 0) {
    __syncwarp(); // .sync.aligned: the whole warp must be converged
    tmem_alloc(slot, ncols);
    tmem_relinquish_alloc_permit();
  }
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  tcgen05_fence_after_thread_sync();
  return *slot;
}

// Every warp's tcgen05.ld has completed (tmem_wait_ld); order them before
// warp 0 frees the columns. A CTA must free its TMEM before exiting.
__device__ __forceinline__ void release_tmem(uint32_t tmem, int ncols) {
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    tcgen05_fence_after_thread_sync();
    tmem_dealloc(tmem, ncols);
  }
}

// ===========================================================================
// Step 1: a single 128 x 128 x 64 tile (M <= 128, N <= 128, K <= 64).
// ===========================================================================
constexpr int kOneBN = 128;
constexpr int kOneSmem = 1024 + (kBM + kOneBN) * kBK * 2;

// ---------------------------------------------------------------------------
// TODO(1f) — after setup_tmem (lesson §5.1):
//   Thread 0: arm `full` with the bytes of both boxes, TMA-load the A box
//     (origin x = 0, y = 0) into sa and the B box into sb, wait for phase 0,
//     tcgen05_fence_after_thread_sync (optional here, lesson §4.7),
//     mma_k_tile (no accumulation),
//     tcgen05_commit(&mma_done).
//   Everyone: __syncwarp (warp 0 must reconverge before the .sync.aligned
//     tcgen05.ld), wait for phase 0 of mma_done,
//     tcgen05_fence_after_thread_sync, store_direct for (m0, n0) = (0, 0).
//   Finally release_tmem.
// ---------------------------------------------------------------------------
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
  const uint32_t tmem = setup_tmem(&tmem_slot, kOneBN); // includes __syncthreads
  LAB_TODO(1);
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
    // -----------------------------------------------------------------------
    // TODO(2) — one thread issues both the TMA loads and the MMAs
    // (lesson §5.2). K tile t (columns 64t .. 64t + 63) lives in stage
    // s = t % S; the A box origin is (x = 64t, y = m0), the B box
    // (x = 64t, y = n0); each fill delivers Cfg::kTxBytes to full[s].
    //   prologue: load tiles 0 .. min(S, k_tiles) - 1.
    //   for t in 0 .. k_tiles - 1:
    //     - wait for fill number t / S of stage s (parity?),
    //       tcgen05_fence_after_thread_sync (optional, lesson §4.7);
    //     - mma_k_tile on stage s (accumulate unless t == 0);
    //     - tcgen05_commit(&empty[s]): arrives when these MMAs finished
    //       reading stage s;
    //     - refill: tile t - 1 + S (if it exists, t >= 1) goes into the
    //       stage of tile t - 1. Wait until MMA t - 1 released it (empty,
    //       parity?), then load. Why refill the stage of t - 1 and not of t?
    //   after the loop: tcgen05_commit(&mma_done).
    // -----------------------------------------------------------------------
    LAB_TODO(2);
  }
  __syncwarp();
  lab::mbar_wait(&mma_done, 0);
  tcgen05_fence_after_thread_sync();

  if constexpr (!Cfg::kTmaEpilogue) {
    store_direct<BN>(tmem, c, m, n, m0, n0);
  } else {
    // -----------------------------------------------------------------------
    // TODO(3a) — TMA-store epilogue (lesson §5.3). The stage buffers are
    // free now: reuse them as `staging`. Warp w, column block j (32 columns)
    // owns the 32 x 32 FP32 box at staging + (w * (BN / 32) + j) * 1024
    // floats; c_map has 32 x 32 boxes with the 128B swizzle.
    // For each j:
    //   - tmem_ld_32x32b_x32 of lanes 32w.., columns 32j..; tmem_wait_ld;
    //   - lane r writes its 32 values as row r of the box: 8 float4, chunk q
    //     (values 4q .. 4q+3) at chunk position q ^ (r % 8);
    //   - make the writes visible to the TMA store (which fence? lab 30),
    //     __syncwarp, lane 0 stores the box at origin (x = n0 + 32j,
    //     y = m0 + 32w).
    // Then lane 0 commits and waits until the stores have READ the staging
    // buffers (the CTA's shared memory disappears when it exits).
    // -----------------------------------------------------------------------
    LAB_TODO(3);
  }
  release_tmem(tmem, Cfg::kTmemCols);
}

using Step2 = Config<128, 4, false>;
// ---------------------------------------------------------------------------
// TODO(3b) — choose the step-3 configuration Config<BN, Stages, true>
// (lesson §6 step 3): start from <128, 4, true>, then try BN = 256 and
// different stage counts (shared memory: Stages * (128 + BN) * 128 bytes
// must fit in 227 KB). Keep the fastest on both perf cases.
// ---------------------------------------------------------------------------
using Step3 = Config<128, 4, true>;

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
  plan.c_map = lab::make_tensor_map_2d(p.c, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                       p.m, p.n, 32, 32,
                                       CU_TENSOR_MAP_SWIZZLE_128B);
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
    gemm_pipelined<Step2><<<grid_of<Step2>(p), kThreads, Step2::kSmemBytes,
                            stream>>>(plan.a_map, plan.b_map, plan.c_map, c,
                                      p.m, p.n, p.k);
    break;
  default:
    gemm_pipelined<Step3><<<grid_of<Step3>(p), kThreads, Step3::kSmemBytes,
                            stream>>>(plan.a_map, plan.b_map, plan.c_map, c,
                                      p.m, p.n, p.k);
    break;
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
