// 21_pipeline_gemm: C[M,N] = A[M,K] . B[N,K]^T with mma.sync, and cp.async
// to overlap global -> shared copies with Tensor Core math.
// Lesson: course/21_pipeline_gemm/README.md
//
// A is [M, K], B is [N, K], both BF16, row-major (K fastest, "K-major");
// C is [M, N] FP32, row-major:
//   A[m][k] is at a[m * K + k],  B[n][k] at b[n * K + k],  C[m][n] at c[m * N + n].
// Supported: K % 8 == 0 (a 16-byte copy never straddles the end of a row).
// make_plan rejects anything else. Shared tiles are K-major with XOR-swizzled
// 16-byte chunks (lab 20), one "stage" = one A tile followed by one B tile.
// run() picks a version by p.step:  ./lab bench 21 --step 1 ... --step 3
//
// ldmatrix, mma.sync, frag_c_coord and swizzle_chunk come from
// include/lab/mma_sync.cuh (what you wrote in lab 20).
#include "lab/lab.h"
#include "lab/mma_sync.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

// ===========================================================================
// cp.async primitives. You write them here; from lab 22 on they come from
// include/lab/cp_async.cuh. A shared-memory address for the asm operand is
// lab::smem_u32(ptr) (a 32-bit "r" operand); a global pointer is an "l"
// operand. Give every asm a "memory" clobber so the compiler does not move
// ordinary loads/stores across it.
// ===========================================================================

// ---------------------------------------------------------------------------
// TODO(1a) — three wrappers (lesson §4.1):
//   cp_async_16_zfill: cp.async.cg.shared.global [dst], [src], 16, src_size;
//     copies 16 bytes global -> shared without blocking; src_size =
//     valid ? 16 : 0, and the bytes after src_size are filled with zeros.
//   cp_async_commit:   cp.async.commit_group;  closes this thread's group.
//   cp_async_wait_group<N>: cp.async.wait_group N;  waits until at most N
//     of this thread's groups are pending ("n" constraint for N).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16_zfill(void *dst, const void *src,
                                                  bool valid) {
  LAB_TODO(1);
}
__device__ __forceinline__ void cp_async_commit() {
  LAB_TODO(1);
}
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
  LAB_TODO(1);
}

// Tile configuration. Shared memory holds kStages copies of a kBM x kBK A
// tile and a kBN x kBK B tile, both K-major with XOR-swizzled 16-byte chunks.
template <int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
struct Config {
  static constexpr int kBM = BM, kBN = BN, kBK = BK, kStages = Stages;
  static constexpr int kWarpsM = WarpsM, kWarpsN = WarpsN;
  static constexpr int kThreads = WarpsM * WarpsN * 32;
  static constexpr int kWM = BM / WarpsM, kWN = BN / WarpsN; // warp tile
  static constexpr int kMI = kWM / 16, kNJ = kWN / 8;        // mma tiles
  static constexpr int kChunks = BK / 8; // 16-byte chunks per tile row
  static constexpr int kStageElems = (BM + BN) * BK;
  static constexpr int kSmemBytes = Stages * kStageElems * 2;
  static_assert(kNJ % 2 == 0, "B fragments are loaded two n8 tiles at a time");
  static_assert(BM * kChunks % kThreads == 0 && BN * kChunks % kThreads == 0);
};

// Element offset of (row, col) in a swizzled [rows][kBK] tile (col % 8 == 0).
template <class C> __device__ __forceinline__ int tile_offset(int row, int col) {
  return row * C::kBK + lab::swizzle_chunk<C::kChunks>(row, col / 8) * 8 +
         col % 8;
}

// ---------------------------------------------------------------------------
// TODO(1b) — start copying rows row0 .. row0 + kRows - 1, columns
// k0 .. k0 + kBK - 1 of the K-major matrix g ([rows][k]) into the shared tile
// s, without waiting. Same thread -> chunk mapping as load_tile in lab 20:
// iteration i of thread t copies chunk idx = t + i * kThreads, i.e. tile row
// idx / kChunks, columns (idx % kChunks) * 8 .. +7, to s + tile_offset<C>.
// A chunk outside the matrix (row0 + r >= rows or k0 + col >= k) must become
// 16 zero bytes, and must not read out of bounds: pass valid = false and a
// harmless source address such as g itself. Do not commit here.
// ---------------------------------------------------------------------------
template <class C, int kRows>
__device__ __forceinline__ void issue_tile(bf16 *s, const bf16 *g, int rows,
                                           int k, int row0, int k0) {
  LAB_TODO(1);
}

template <class C>
__device__ __forceinline__ void issue_stage(bf16 *stage, const bf16 *a,
                                            const bf16 *b, int m, int n,
                                            int k, int m0, int n0, int k0) {
  issue_tile<C, C::kBM>(stage, a, m, k, m0, k0);
  issue_tile<C, C::kBN>(stage + C::kBM * C::kBK, b, n, k, n0, k0);
}

// ldmatrix + mma over one K tile held in shared memory (from lab 20).
template <class C>
__device__ __forceinline__ void compute_stage(const bf16 *stage,
                                              float (&acc)[C::kMI][C::kNJ][4],
                                              int wm, int wn, int lane) {
  const bf16 *sa = stage, *sb = stage + C::kBM * C::kBK;
#pragma unroll
  for (int kk = 0; kk < C::kBK; kk += 16) {
    uint32_t af[C::kMI][4], bfr[C::kNJ][2];
#pragma unroll
    for (int i = 0; i < C::kMI; ++i)
      lab::ldmatrix_x4(af[i], sa + tile_offset<C>(wm + i * 16 + lane % 16,
                                                  kk + lane / 16 * 8));
#pragma unroll
    for (int j = 0; j < C::kNJ; j += 2) {
      uint32_t r[4];
      lab::ldmatrix_x4(r, sb + tile_offset<C>(wn + j * 8 + lane % 8 +
                                                  lane / 16 * 8,
                                              kk + (lane / 8) % 2 * 8));
      bfr[j][0] = r[0];
      bfr[j][1] = r[1];
      bfr[j + 1][0] = r[2];
      bfr[j + 1][1] = r[3];
    }
#pragma unroll
    for (int i = 0; i < C::kMI; ++i)
#pragma unroll
      for (int j = 0; j < C::kNJ; ++j)
        lab::mma_m16n8k16_bf16(acc[i][j], af[i], bfr[j]);
  }
}

template <class C>
__device__ __forceinline__ void store_c(float *c, int m, int n, int m0,
                                        int n0, int wm, int wn, int lane,
                                        const float (&acc)[C::kMI][C::kNJ][4]) {
#pragma unroll
  for (int i = 0; i < C::kMI; ++i)
#pragma unroll
    for (int j = 0; j < C::kNJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        lab::FragCoord rc = lab::frag_c_coord(lane, e);
        int row = m0 + wm + i * 16 + rc.row;
        int col = n0 + wn + j * 8 + rc.col;
        if (row < m && col < n)
          c[static_cast<int64_t>(row) * n + col] = acc[i][j][e];
      }
}

// Map the linear CTA index to a (tile_m, tile_n).
template <int kGroup>
__device__ __forceinline__ void tile_coords(int tiles_m, int tiles_n,
                                            int &tm, int &tn) {
  const int pid = blockIdx.x;
  if constexpr (kGroup <= 1) {
    // Row by row: consecutive CTAs walk along N.
    tm = pid / tiles_n;
    tn = pid % tiles_n;
  } else {
    // -----------------------------------------------------------------------
    // TODO(3a) — grouped rasterization (lesson §5.3). Split the tile rows
    // into groups of kGroup. Inside a group, consecutive pid walk DOWN a
    // column of tiles (tm changes fastest), then move to the next column;
    // after the last column, the next group starts. The last group may have
    // fewer than kGroup rows (tiles_m % kGroup). Every (tm, tn) must be
    // produced exactly once for pid = 0 .. tiles_m * tiles_n - 1.
    // -----------------------------------------------------------------------
    LAB_TODO(3);
    tm = tn = 0;
  }
}

// ===========================================================================
// Step 1: double buffering.
// ===========================================================================
template <class C>
__global__ void __launch_bounds__(C::kThreads)
    gemm_double_buffer(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
                       float *__restrict__ c, int m, int n, int k) {
  static_assert(C::kStages == 2);
  extern __shared__ __align__(128) unsigned char smem_raw[];
  bf16 *smem = reinterpret_cast<bf16 *>(smem_raw); // 2 stages, kStageElems each
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tm, tn;
  tile_coords<1>((m + C::kBM - 1) / C::kBM, (n + C::kBN - 1) / C::kBN, tm, tn);
  const int m0 = tm * C::kBM, n0 = tn * C::kBN;
  const int wm = warp / C::kWarpsN * C::kWM, wn = warp % C::kWarpsN * C::kWN;
  const int k_tiles = (k + C::kBK - 1) / C::kBK;

  float acc[C::kMI][C::kNJ][4] = {};
  // -------------------------------------------------------------------------
  // TODO(1c) — K tile t lives in stage t % 2 (smem + (t % 2) * kStageElems).
  //   prologue: issue_stage(tile 0) into stage 0, commit.
  //   for t in 0 .. k_tiles-1:
  //     - wait for your own copies of tile t, then __syncthreads(). Write
  //       down the TWO things this barrier guarantees (lesson §5.1).
  //     - if tile t+1 exists: issue it into the other stage, commit.
  //     - compute_stage(stage of tile t).
  // Exactly one wait and one __syncthreads per tile is enough.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
  store_c<C>(c, m, n, m0, n0, wm, wn, lane, acc);
}

// ===========================================================================
// Steps 2 and 3: a ring of kStages buffers.
// ===========================================================================
template <class C, int kGroup>
__global__ void __launch_bounds__(C::kThreads)
    gemm_multistage(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
                    float *__restrict__ c, int m, int n, int k) {
  constexpr int S = C::kStages;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  bf16 *smem = reinterpret_cast<bf16 *>(smem_raw); // S stages
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tm, tn;
  tile_coords<kGroup>((m + C::kBM - 1) / C::kBM, (n + C::kBN - 1) / C::kBN, tm,
                      tn);
  const int m0 = tm * C::kBM, n0 = tn * C::kBN;
  const int wm = warp / C::kWarpsN * C::kWM, wn = warp % C::kWarpsN * C::kWN;
  const int k_tiles = (k + C::kBK - 1) / C::kBK;

  float acc[C::kMI][C::kNJ][4] = {};
  // -------------------------------------------------------------------------
  // TODO(2) — K tile t lives in stage t % S. Keep the invariant
  //   "commit group number g contains exactly K tile g"
  // by committing one group per tile slot, even when the tile is past the
  // end of K (an empty group completes at once).
  //   prologue: tiles 0 .. S-2, one commit each.
  //   for t in 0 .. k_tiles-1:
  //     - wait until tile t is complete: how many of the groups committed
  //       so far may still be pending? (wait_group needs a constant.)
  //     - __syncthreads()
  //     - issue tile t + S - 1 (if it exists) into its stage; commit.
  //       Which tile used that stage before, and why is it free now?
  //     - compute_stage(stage of tile t).
  // -------------------------------------------------------------------------
  LAB_TODO(2);
  store_c<C>(c, m, n, m0, n0, wm, wn, lane, acc);
}

using Step1 = Config<128, 128, 32, 2, 4, 2>;
using Step2 = Config<128, 128, 32, 2, 4, 3>;
// ---------------------------------------------------------------------------
// TODO(3b) — tune the step-3 configuration Config<BM, BN, BK, WarpsM,
// WarpsN, Stages>. Start from step 2's and try the experiments in lesson §6
// step 3 (BK = 64, 64x64 warp tiles, number of stages); keep the fastest.
// ---------------------------------------------------------------------------
using Step3 = Config<128, 128, 32, 2, 4, 3>;
constexpr int kGroupM = 8; // tile rows per rasterization group (step 3)

template <class C> int grid_size(const lab::GemmProblem &p) {
  return ((p.m + C::kBM - 1) / C::kBM) * ((p.n + C::kBN - 1) / C::kBN);
}

template <class Kernel> void allow_smem(Kernel kernel, int bytes) {
  // More than 48 KB of dynamic shared memory needs an explicit opt-in.
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  // L1 and shared memory share one 256 KB array per SM. Prefer the largest
  // shared part so two CTAs always fit (with the default choice we measured
  // occasional 2x slower runs).
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

} // namespace

struct Plan {};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::BF16, "this lab uses BF16 A and B");
  LAB_REQUIRE(p.c_dtype == lab::DType::F32, "this lab writes an FP32 C");
  LAB_REQUIRE(p.m > 0 && p.n > 0 && p.k > 0, "empty problem");
  LAB_REQUIRE(p.k % 8 == 0, "K must be a multiple of 8 (16-byte cp.async)");
  allow_smem(gemm_double_buffer<Step1>, Step1::kSmemBytes);
  allow_smem(gemm_multistage<Step2, 1>, Step2::kSmemBytes);
  allow_smem(gemm_multistage<Step3, kGroupM>, Step3::kSmemBytes);
  return {};
}

void run(const Plan &, const lab::GemmProblem &p, cudaStream_t stream) {
  auto a = static_cast<const bf16 *>(p.a);
  auto b = static_cast<const bf16 *>(p.b);
  auto c = static_cast<float *>(p.c);
  switch (p.step) {
  case 1:
    gemm_double_buffer<Step1>
        <<<grid_size<Step1>(p), Step1::kThreads, Step1::kSmemBytes, stream>>>(
            a, b, c, p.m, p.n, p.k);
    break;
  case 2:
    gemm_multistage<Step2, 1>
        <<<grid_size<Step2>(p), Step2::kThreads, Step2::kSmemBytes, stream>>>(
            a, b, c, p.m, p.n, p.k);
    break;
  default:
    gemm_multistage<Step3, kGroupM>
        <<<grid_size<Step3>(p), Step3::kThreads, Step3::kSmemBytes, stream>>>(
            a, b, c, p.m, p.n, p.k);
    break;
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
