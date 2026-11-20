// Reference solution for 21_pipeline_gemm: C[M,N] = A[M,K] . B[N,K]^T with
// mma.sync, and cp.async to overlap the global -> shared copies with math.
//
// step 1: double buffering. While the warps run mma on buffer s, cp.async
//         fills buffer s^1 with the next K tile. One wait_group 0 and one
//         __syncthreads per K tile.
// step 2: a ring of kStages = 3 buffers: up to two K tiles are in flight
//         while one is consumed; wait_group<kStages - 2>.
// step 3: tuning. kBK = 64 (128-byte rows, 8-chunk swizzle): half as many
//         barriers per FLOP. 4 warps with 64x64 warp tiles: each ldmatrix
//         feeds twice as many mma. 3 stages x 32 KB lets 2 CTAs share an SM.
//         Grouped rasterization makes CTAs that run together share A and B
//         tiles in L2 (fewer DRAM bytes; the kernel is compute bound, so the
//         time barely changes).
//
// Supported domain: BF16 A/B, FP32 C, K % 8 == 0 (a 16-byte cp.async never
// straddles the end of a row). M and N are arbitrary.
#include "lab/cp_async.cuh"
#include "lab/lab.h"
#include "lab/mma_sync.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

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

template <class C>
__device__ __forceinline__ int tile_offset(int row, int col) {
  return row * C::kBK + lab::swizzle_chunk<C::kChunks>(row, col / 8) * 8 +
         col % 8;
}

// Issue the cp.async for one K tile (cols k0 .. k0 + kBK) of A and B into
// one stage. Chunks outside the matrix are zero-filled (src-size 0) and read
// nothing, so the tail adds zeros to the dot products. Does not commit.
template <class C, int kRows>
__device__ __forceinline__ void issue_tile(bf16 *s, const bf16 *g, int rows,
                                           int k, int row0, int k0) {
#pragma unroll
  for (int i = 0; i < kRows * C::kChunks / C::kThreads; ++i) {
    int idx = threadIdx.x + i * C::kThreads;
    int r = idx / C::kChunks, col = idx % C::kChunks * 8;
    bool ok = row0 + r < rows && k0 + col < k;
    const bf16 *src =
        ok ? g + static_cast<int64_t>(row0 + r) * k + k0 + col : g;
    lab::cp_async_16_zfill(s + tile_offset<C>(r, col), src, ok);
  }
}

template <class C>
__device__ __forceinline__ void issue_stage(bf16 *stage, const bf16 *a,
                                            const bf16 *b, int m, int n, int k,
                                            int m0, int n0, int k0) {
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
      lab::ldmatrix_x4(
          r, sb + tile_offset<C>(wn + j * 8 + lane % 8 + lane / 16 * 8,
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
__device__ __forceinline__ void store_c(float *c, int m, int n, int m0, int n0,
                                        int wm, int wn, int lane,
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

// Map the linear CTA index to a (tile_m, tile_n). Without grouping,
// consecutive CTAs walk along N. With grouping, they walk down a column of
// kGroup tile rows first, so the CTAs that run together touch kGroup A
// tile-rows and few B tile-columns: both stay in L2.
template <int kGroup>
__device__ __forceinline__ void tile_coords(int tiles_m, int tiles_n, int &tm,
                                            int &tn) {
  const int pid = blockIdx.x;
  if constexpr (kGroup <= 1) {
    tm = pid / tiles_n;
    tn = pid % tiles_n;
  } else {
    const int per_group = kGroup * tiles_n;
    const int first_m = pid / per_group * kGroup;
    const int rows = min(tiles_m - first_m, kGroup); // last group may be short
    const int in_group = pid % per_group;
    tm = first_m + in_group % rows;
    tn = in_group / rows;
  }
}

// ---------------------------------------------------------------- step 1
template <class C>
__global__ void __launch_bounds__(C::kThreads)
    gemm_double_buffer(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
                       float *__restrict__ c, int m, int n, int k) {
  static_assert(C::kStages == 2);
  extern __shared__ __align__(128) unsigned char smem_raw[];
  bf16 *smem = reinterpret_cast<bf16 *>(smem_raw);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tm, tn;
  tile_coords<1>((m + C::kBM - 1) / C::kBM, (n + C::kBN - 1) / C::kBN, tm, tn);
  const int m0 = tm * C::kBM, n0 = tn * C::kBN;
  const int wm = warp / C::kWarpsN * C::kWM, wn = warp % C::kWarpsN * C::kWN;
  const int k_tiles = (k + C::kBK - 1) / C::kBK;

  float acc[C::kMI][C::kNJ][4] = {};
  issue_stage<C>(smem, a, b, m, n, k, m0, n0, 0);
  lab::cp_async_commit();
  for (int t = 0; t < k_tiles; ++t) {
    // This thread's copies of tile t are done; the barrier makes everyone
    // else's visible AND guarantees that all warps finished reading buffer
    // (t+1)%2 (tile t-1) before we overwrite it below.
    lab::cp_async_wait_group<0>();
    __syncthreads();
    if (t + 1 < k_tiles) {
      issue_stage<C>(smem + (t + 1) % 2 * C::kStageElems, a, b, m, n, k, m0, n0,
                     (t + 1) * C::kBK);
      lab::cp_async_commit();
    }
    // Tile t + 1 is now in flight while we compute tile t.
    compute_stage<C>(smem + t % 2 * C::kStageElems, acc, wm, wn, lane);
  }
  store_c<C>(c, m, n, m0, n0, wm, wn, lane, acc);
}

// ---------------------------------------------------------------- step 2, 3
template <class C, int kGroup>
__global__ void __launch_bounds__(C::kThreads)
    gemm_multistage(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
                    float *__restrict__ c, int m, int n, int k) {
  constexpr int S = C::kStages;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  bf16 *smem = reinterpret_cast<bf16 *>(smem_raw);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tm, tn;
  tile_coords<kGroup>((m + C::kBM - 1) / C::kBM, (n + C::kBN - 1) / C::kBN, tm,
                      tn);
  const int m0 = tm * C::kBM, n0 = tn * C::kBN;
  const int wm = warp / C::kWarpsN * C::kWM, wn = warp % C::kWarpsN * C::kWN;
  const int k_tiles = (k + C::kBK - 1) / C::kBK;

  float acc[C::kMI][C::kNJ][4] = {};
  // Prologue: tiles 0 .. S-2 into stages 0 .. S-2, one group per tile.
  // A group is committed even when the tile does not exist (short K), so
  // "group g holds tile g" stays true and wait_group counts stay uniform.
#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    if (s < k_tiles)
      issue_stage<C>(smem + s * C::kStageElems, a, b, m, n, k, m0, n0,
                     s * C::kBK);
    lab::cp_async_commit();
  }
  for (int t = 0; t < k_tiles; ++t) {
    // Groups committed so far: tiles 0 .. t+S-2. Allowing S-2 pending means
    // tiles 0 .. t are complete (for this thread).
    lab::cp_async_wait_group<S - 2>();
    // Publishes tile t to all warps, and every warp has left compute of
    // tile t-1, whose stage (t-1)%S = (t+S-1)%S is refilled next.
    __syncthreads();
    const int next = t + S - 1;
    if (next < k_tiles)
      issue_stage<C>(smem + next % S * C::kStageElems, a, b, m, n, k, m0, n0,
                     next * C::kBK);
    lab::cp_async_commit(); // possibly empty: keeps one group per tile
    compute_stage<C>(smem + t % S * C::kStageElems, acc, wm, wn, lane);
  }
  // Only empty groups can still be pending here; no smem is read after.
  store_c<C>(c, m, n, m0, n0, wm, wn, lane, acc);
}

using Step1 = Config<128, 128, 32, 2, 4, 2>;
using Step2 = Config<128, 128, 32, 2, 4, 3>;
using Step3 = Config<128, 128, 64, 2, 2, 3>;
constexpr int kGroupM = 8;

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
