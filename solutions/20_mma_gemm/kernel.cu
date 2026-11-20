// Reference solution for 20_mma_gemm: C[M,N] = A[M,K] . B[N,K]^T on Tensor
// Cores with mma.sync.m16n8k16 (BF16 in, FP32 accumulate and out).
//
// step 1: one warp computes one 16x8 tile of C; it loads its A and B
//         fragments straight from global memory using the fragment
//         coordinates, then calls mma once per 16-wide K slice.
// step 2: a 128x128 CTA tile, 8 warps of 64x32. A 128x32 slice of A and B is
//         staged through shared memory; warps read fragments with ldmatrix.
//         Rows of the shared tile are 64 bytes, so ldmatrix hits 4-way bank
//         conflicts.
// step 3: same kernel with an XOR-swizzled shared layout: conflict-free.
//
// Supported domain: BF16 A/B, FP32 C, K % 8 == 0 (one 16-byte chunk is
// either fully inside or fully outside a row). M and N are arbitrary from
// step 2 on; step 1 needs M % 16 == 0, N % 8 == 0, K % 16 == 0.
#include "lab/lab.h"
#include "lab/mma_sync.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------- step 1
constexpr int kNaiveWarps = 4; // warps per block, stacked along M

__global__ void __launch_bounds__(kNaiveWarps * 32)
    mma_naive(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
              float *__restrict__ c, int m, int n, int k) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int m0 = (blockIdx.y * kNaiveWarps + warp) * 16;
  const int n0 = blockIdx.x * 8;
  if (m0 >= m)
    return; // whole warp leaves together: mma needs all 32 lanes
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k0 = 0; k0 < k; k0 += 16) {
    uint32_t af[4], bfr[2];
    // Each register holds two neighbours along K, so one 32-bit load each.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      lab::FragCoord rc = lab::frag_a_coord(lane, i);
      af[i] = *reinterpret_cast<const uint32_t *>(
          a + static_cast<int64_t>(m0 + rc.row) * k + k0 + rc.col);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      lab::FragCoord kn = lab::frag_b_coord(lane, i); // {row = k, col = n}
      bfr[i] = *reinterpret_cast<const uint32_t *>(
          b + static_cast<int64_t>(n0 + kn.col) * k + k0 + kn.row);
    }
    lab::mma_m16n8k16_bf16(d, af, bfr);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lab::FragCoord rc = lab::frag_c_coord(lane, i);
    c[static_cast<int64_t>(m0 + rc.row) * n + n0 + rc.col] = d[i];
  }
}

// ---------------------------------------------------------------- step 2, 3
constexpr int kBM = 128, kBN = 128, kBK = 32;
constexpr int kWarpsM = 2, kWarpsN = 4;      // 8 warps
constexpr int kWM = kBM / kWarpsM;           // 64 rows per warp
constexpr int kWN = kBN / kWarpsN;           // 32 cols per warp
constexpr int kMI = kWM / 16, kNJ = kWN / 8; // 4 x 4 mma tiles per warp
constexpr int kThreads = kWarpsM * kWarpsN * 32;
constexpr int kChunks = kBK / 8; // 16-byte chunks per tile row (= 4)

// Element offset of (row, col) in a [rows][kBK] shared tile. col is a
// multiple of 8 for every caller: we only address whole 16-byte chunks.
template <bool kSwizzle>
__device__ __forceinline__ int tile_offset(int row, int col) {
  int chunk = col / 8;
  if constexpr (kSwizzle)
    chunk = lab::swizzle_chunk<kChunks>(row, chunk);
  return row * kBK + chunk * 8 + col % 8;
}

// Copy a kRows x kBK slice (rows row0.., cols k0..) of a K-major matrix into
// shared memory, 16 bytes per thread per iteration. Out-of-range chunks are
// written as zeros so they add nothing to the dot products.
template <int kRows, bool kSwizzle>
__device__ __forceinline__ void load_tile(bf16 *s, const bf16 *g, int rows,
                                          int k, int row0, int k0) {
  constexpr int kIters = kRows * kChunks / kThreads;
  uint4 v[kIters];
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    int idx = threadIdx.x + i * kThreads;
    int r = idx / kChunks, col = idx % kChunks * 8;
    bool ok = row0 + r < rows && k0 + col < k;
    v[i] = ok ? *reinterpret_cast<const uint4 *>(
                    g + static_cast<int64_t>(row0 + r) * k + k0 + col)
              : make_uint4(0, 0, 0, 0);
  }
  // All loads are issued before the first store, so they overlap.
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    int idx = threadIdx.x + i * kThreads;
    int r = idx / kChunks, col = idx % kChunks * 8;
    *reinterpret_cast<uint4 *>(s + tile_offset<kSwizzle>(r, col)) = v[i];
  }
}

template <bool kSwizzle>
__global__ void __launch_bounds__(kThreads)
    mma_cta(const bf16 *__restrict__ a, const bf16 *__restrict__ b,
            float *__restrict__ c, int m, int n, int k) {
  __shared__ __align__(128) bf16 sa[kBM * kBK];
  __shared__ __align__(128) bf16 sb[kBN * kBK];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  const int wm = warp / kWarpsN * kWM; // warp's first row inside the tile
  const int wn = warp % kWarpsN * kWN; // warp's first col inside the tile

  float acc[kMI][kNJ][4] = {};
  for (int k0 = 0; k0 < k; k0 += kBK) {
    load_tile<kBM, kSwizzle>(sa, a, m, k, m0, k0);
    load_tile<kBN, kSwizzle>(sb, b, n, k, n0, k0);
    __syncthreads(); // tile complete before any warp reads it

#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t af[kMI][4], bfr[kNJ][2];
      // A 16x16 = four 8x8 matrices; lanes 0-15 give rows 0-15 at col kk,
      // lanes 16-31 the same rows at col kk + 8. Register order matches
      // the a0..a3 fragment order (rows +0/+8, cols +0/+8).
#pragma unroll
      for (int i = 0; i < kMI; ++i)
        lab::ldmatrix_x4(af[i],
                         sa + tile_offset<kSwizzle>(wm + i * 16 + lane % 16,
                                                    kk + lane / 16 * 8));
      // B tile is [n][k] in shared memory, which is exactly the .col
      // operand layout: no .trans. One x4 covers two n8 tiles:
      // matrices (n0-7,k0-7) (n0-7,k8-15) (n8-15,k0-7) (n8-15,k8-15).
#pragma unroll
      for (int j = 0; j < kNJ; j += 2) {
        uint32_t r[4];
        lab::ldmatrix_x4(
            r, sb + tile_offset<kSwizzle>(wn + j * 8 + lane % 8 + lane / 16 * 8,
                                          kk + (lane / 8) % 2 * 8));
        bfr[j][0] = r[0];
        bfr[j][1] = r[1];
        bfr[j + 1][0] = r[2];
        bfr[j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < kMI; ++i)
#pragma unroll
        for (int j = 0; j < kNJ; ++j)
          lab::mma_m16n8k16_bf16(acc[i][j], af[i], bfr[j]);
    }
    __syncthreads(); // every warp is done reading before the next load
  }

  // Epilogue: write the accumulators straight from registers.
#pragma unroll
  for (int i = 0; i < kMI; ++i)
#pragma unroll
    for (int j = 0; j < kNJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        lab::FragCoord rc = lab::frag_c_coord(lane, e);
        int row = m0 + wm + i * 16 + rc.row;
        int col = n0 + wn + j * 8 + rc.col;
        if (row < m && col < n)
          c[static_cast<int64_t>(row) * n + col] = acc[i][j][e];
      }
}

template <bool kSwizzle>
void launch_cta(const bf16 *a, const bf16 *b, float *c, int m, int n, int k,
                cudaStream_t stream) {
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  mma_cta<kSwizzle><<<grid, kThreads, 0, stream>>>(a, b, c, m, n, k);
}

} // namespace

struct Plan {};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::BF16, "this lab uses BF16 A and B");
  LAB_REQUIRE(p.c_dtype == lab::DType::F32, "this lab writes an FP32 C");
  LAB_REQUIRE(p.m > 0 && p.n > 0 && p.k > 0, "empty problem");
  LAB_REQUIRE(p.k % 8 == 0, "K must be a multiple of 8 (16-byte rows)");
  if (p.step == 1)
    LAB_REQUIRE(p.m % 16 == 0 && p.n % 8 == 0 && p.k % 16 == 0,
                "step 1 needs M % 16 == 0, N % 8 == 0, K % 16 == 0");
  return {};
}

void run(const Plan &, const lab::GemmProblem &p, cudaStream_t stream) {
  auto a = static_cast<const bf16 *>(p.a);
  auto b = static_cast<const bf16 *>(p.b);
  auto c = static_cast<float *>(p.c);
  switch (p.step) {
  case 1: {
    dim3 grid(p.n / 8, (p.m / 16 + kNaiveWarps - 1) / kNaiveWarps);
    mma_naive<<<grid, kNaiveWarps * 32, 0, stream>>>(a, b, c, p.m, p.n, p.k);
    break;
  }
  case 2:
    launch_cta<false>(a, b, c, p.m, p.n, p.k, stream);
    break;
  default:
    launch_cta<true>(a, b, c, p.m, p.n, p.k, stream);
    break;
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
