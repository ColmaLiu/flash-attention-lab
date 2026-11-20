// Reference solution for 10_simt_gemm: C[M,N] = A[M,K] . B[N,K]^T, FP32.
// Lesson: course/10_simt_gemm/README.md
//
// step 1: one thread per C element, operands straight from global memory.
//         Every A and B element is fetched from L1/L2 again for every output
//         that uses it.
// step 2: a 32x32 block tile. A 32x32 slice of A and of B (BK = 32) is staged
//         in shared memory once and reused by 32 threads each.
// step 3: register blocking. A 128x128 block tile, BK = 8, 256 threads, each
//         thread accumulates an 8x8 micro-tile in registers. One value loaded
//         from shared memory now feeds 8 FMAs instead of 1.
// step 4: the same tiling, vectorized. Tiles are staged transposed
//         ([BK][BM + 4]) so a thread's micro-tile column is contiguous and can
//         be read with 128-bit shared-memory loads; global loads and C stores
//         use float4 too.
#include "lab/lab.h"

namespace {

// ---------------------------------------------------------------------------
// Step 1: naive.
// ---------------------------------------------------------------------------
constexpr int kNaiveX = 32; // threadIdx.x -> n: C writes are coalesced
constexpr int kNaiveY = 8;

__global__ void gemm_naive(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x; // n
  int row = blockIdx.y * blockDim.y + threadIdx.y; // m
  if (row >= m || col >= n)
    return;
  const float *a_row = a + static_cast<int64_t>(row) * k;
  const float *b_row = b + static_cast<int64_t>(col) * k;
  float acc = 0.f;
  for (int kk = 0; kk < k; ++kk)
    acc += a_row[kk] * b_row[kk];
  c[static_cast<int64_t>(row) * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Step 2: shared-memory tiling, one output per thread.
// A 32x32 block of threads computes the 32x32 C tile at (m0, n0). For each
// K slice of width 32 it stages A[m0:m0+32, k0:k0+32] and B[n0:n0+32,
// k0:k0+32] in shared memory; each staged value is then read by 32 threads.
// ---------------------------------------------------------------------------
constexpr int kTile = 32;

__global__ void gemm_smem(const float *__restrict__ a,
                          const float *__restrict__ b, float *__restrict__ c,
                          int m, int n, int k) {
  // Both tiles keep the global K-major layout: tile[row][kk]. The compute
  // loop reads bs[tx][kk] down a column (tx varies across the warp), so pad
  // each row by one float to spread that column over 32 banks (lab 00).
  __shared__ float as[kTile][kTile + 1];
  __shared__ float bs[kTile][kTile + 1];
  int tx = threadIdx.x, ty = threadIdx.y;
  int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;

  float acc = 0.f;
  for (int k0 = 0; k0 < k; k0 += kTile) {
    // Load: lanes read consecutive k of one row (coalesced). Out-of-range
    // elements are zero so they add nothing to the dot product.
    int kc = k0 + tx;
    int ar = m0 + ty, br = n0 + ty;
    as[ty][tx] =
        (ar < m && kc < k) ? a[static_cast<int64_t>(ar) * k + kc] : 0.f;
    bs[ty][tx] =
        (br < n && kc < k) ? b[static_cast<int64_t>(br) * k + kc] : 0.f;
    __syncthreads(); // tiles complete before anyone reads them

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk)
      acc += as[ty][kk] * bs[tx][kk]; // as: broadcast in a warp; bs: 32 banks
    __syncthreads(); // everyone done reading before the next k0 overwrites
  }
  int row = m0 + ty, col = n0 + tx;
  if (row < m && col < n)
    c[static_cast<int64_t>(row) * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Steps 3-4: register blocking.
// Block tile BM x BN = 128 x 128, K slice BK = 8, 256 threads as 16 x 16.
// Each thread owns an 8 x 8 micro-tile of C held in 64 registers.
// ---------------------------------------------------------------------------
constexpr int kBM = 128, kBN = 128, kBK = 8;
constexpr int kTM = 8, kTN = 8;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN); // 256
static_assert(kBM * kBK == kThreads * 4, "each thread loads 4 A elements");
static_assert(kBN * kBK == kThreads * 4, "each thread loads 4 B elements");

// Step 3 tile loader: copy src[r0 : r0+128, k0 : k0+8] of a [rows, k]
// K-major matrix into tile[128][kBK + 1], keeping the K-major layout and
// writing zeros outside the matrix. Element e = tid + 256 * l is tile row
// e / 8, column e % 8, so 8 consecutive lanes read one 32-byte row segment.
__device__ __forceinline__ void stage_kmajor(float (*tile)[kBK + 1],
                                             const float *__restrict__ src,
                                             int r0, int k0, int rows, int k) {
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    int e = threadIdx.x + l * kThreads;
    int r = e / kBK, kk = e % kBK;
    tile[r][kk] = (r0 + r < rows && k0 + kk < k)
                      ? src[static_cast<int64_t>(r0 + r) * k + k0 + kk]
                      : 0.f;
  }
}

// Step 3. Tiles stay K-major in shared memory, padded to BK + 1 floats per
// row. Thread (tx, ty) owns rows ty + 16*i and columns tx + 16*j (strided):
// for one kk, the 16 different tx of a warp read bs[tx + 16*j][kk], whose
// banks (tx * 9 + const) % 32 are all different, so there is no conflict.
__global__ void __launch_bounds__(kThreads)
    gemm_regblock(const float *__restrict__ a, const float *__restrict__ b,
                  float *__restrict__ c, int m, int n, int k) {
  __shared__ float as[kBM][kBK + 1];
  __shared__ float bs[kBN][kBK + 1];
  int tid = threadIdx.x;
  int tx = tid % 16, ty = tid / 16;
  int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;

  float acc[kTM][kTN] = {};
  for (int k0 = 0; k0 < k; k0 += kBK) {
    stage_kmajor(as, a, m0, k0, m, k);
    stage_kmajor(bs, b, n0, k0, n, k);
    __syncthreads(); // tiles complete before the compute loop reads them

    // Outer product per kk: 8 A values x 8 B values -> 64 FMAs from 16 loads.
#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      float af[kTM], bf[kTN];
#pragma unroll
      for (int i = 0; i < kTM; ++i)
        af[i] = as[ty + 16 * i][kk];
#pragma unroll
      for (int j = 0; j < kTN; ++j)
        bf[j] = bs[tx + 16 * j][kk];
#pragma unroll
      for (int i = 0; i < kTM; ++i)
#pragma unroll
        for (int j = 0; j < kTN; ++j)
          acc[i][j] += af[i] * bf[j];
    }
    __syncthreads(); // all reads done before the next slice overwrites
  }

#pragma unroll
  for (int i = 0; i < kTM; ++i) {
    int row = m0 + ty + 16 * i;
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      int col = n0 + tx + 16 * j;
      if (row < m && col < n)
        c[static_cast<int64_t>(row) * n + col] = acc[i][j];
    }
  }
}

// Four consecutive elements src[row][col .. col+3] of a [rows, cols]
// row-major matrix, zero outside it. `vec` says whether 16-byte loads are
// allowed (cols % 4 == 0 keeps every row 16-byte aligned).
__device__ __forceinline__ float4 load4(const float *__restrict__ src, int row,
                                        int col, int rows, int cols, bool vec) {
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (row >= rows)
    return v;
  const float *p = src + static_cast<int64_t>(row) * cols + col;
  if (vec && col + 3 < cols)
    return *reinterpret_cast<const float4 *>(p);
  if (col + 0 < cols)
    v.x = p[0];
  if (col + 1 < cols)
    v.y = p[1];
  if (col + 2 < cols)
    v.z = p[2];
  if (col + 3 < cols)
    v.w = p[3];
  return v;
}

// Store v to dst[row][col .. col+3] of a [rows, cols] matrix, skipping
// elements outside it.
__device__ __forceinline__ void store4(float *__restrict__ dst, int row,
                                       int col, int rows, int cols, bool vec,
                                       float4 v) {
  if (row >= rows)
    return;
  float *p = dst + static_cast<int64_t>(row) * cols + col;
  if (vec && col + 3 < cols) {
    *reinterpret_cast<float4 *>(p) = v;
    return;
  }
  if (col + 0 < cols)
    p[0] = v.x;
  if (col + 1 < cols)
    p[1] = v.y;
  if (col + 2 < cols)
    p[2] = v.z;
  if (col + 3 < cols)
    p[3] = v.w;
}

// Step 4. Tiles are stored transposed, as[kk][row], so for one kk the values a
// thread needs along M (or N) are next to each other. Rows are padded to
// BM + 4 floats: the padding keeps 16-byte alignment and moves the two
// column halves written by one warp to different banks.
//
// Thread (tx, ty) owns rows {ty*4 .. ty*4+3} and {64 + ty*4 .. 64 + ty*4+3},
// columns likewise with tx: two float4 per operand per kk. Splitting the 8
// values into two groups 64 apart makes the 16 tx of a warp read one
// contiguous 256-byte run (2 wavefronts, the minimum) instead of 16 runs
// 32 bytes apart.
constexpr int kPadT = 4;

__global__ void __launch_bounds__(kThreads)
    gemm_vec(const float *__restrict__ a, const float *__restrict__ b,
             float *__restrict__ c, int m, int n, int k) {
  __shared__ __align__(16) float as[kBK][kBM + kPadT];
  __shared__ __align__(16) float bs[kBK][kBN + kPadT];
  int tid = threadIdx.x;
  int tx = tid % 16, ty = tid / 16;
  int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  bool k_vec = (k % 4) == 0;
  bool n_vec = (n % 4) == 0;

  // Loader mapping: thread tid reads one float4 of A (and one of B):
  // tile row tid / 2, k columns (tid % 2) * 4 .. +3.
  int lr = tid / 2, lk = (tid % 2) * 4;

  float acc[kTM][kTN] = {};
  for (int k0 = 0; k0 < k; k0 += kBK) {
    float4 av = load4(a, m0 + lr, k0 + lk, m, k, k_vec);
    float4 bv = load4(b, n0 + lr, k0 + lk, n, k, k_vec);
    // Transpose on the way in: four scalar stores down one smem column.
    as[lk + 0][lr] = av.x;
    as[lk + 1][lr] = av.y;
    as[lk + 2][lr] = av.z;
    as[lk + 3][lr] = av.w;
    bs[lk + 0][lr] = bv.x;
    bs[lk + 1][lr] = bv.y;
    bs[lk + 2][lr] = bv.z;
    bs[lk + 3][lr] = bv.w;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      float af[kTM], bf[kTN];
      // Two 128-bit loads per operand.
      *reinterpret_cast<float4 *>(&af[0]) =
          *reinterpret_cast<const float4 *>(&as[kk][ty * 4]);
      *reinterpret_cast<float4 *>(&af[4]) =
          *reinterpret_cast<const float4 *>(&as[kk][64 + ty * 4]);
      *reinterpret_cast<float4 *>(&bf[0]) =
          *reinterpret_cast<const float4 *>(&bs[kk][tx * 4]);
      *reinterpret_cast<float4 *>(&bf[4]) =
          *reinterpret_cast<const float4 *>(&bs[kk][64 + tx * 4]);
#pragma unroll
      for (int i = 0; i < kTM; ++i)
#pragma unroll
        for (int j = 0; j < kTN; ++j)
          acc[i][j] += af[i] * bf[j];
    }
    __syncthreads();
  }

  // acc[i][j] is C[row(i)][col(j)] with row(i) = m0 + (i / 4) * 64 + ty*4 +
  // i % 4 and col(j) likewise: write each group of 4 columns as one float4.
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
    int row = m0 + (i / 4) * 64 + ty * 4 + i % 4;
#pragma unroll
    for (int jg = 0; jg < 2; ++jg) {
      int col = n0 + jg * 64 + tx * 4;
      float4 v = make_float4(acc[i][jg * 4 + 0], acc[i][jg * 4 + 1],
                             acc[i][jg * 4 + 2], acc[i][jg * 4 + 3]);
      store4(c, row, col, m, n, n_vec, v);
    }
  }
}

} // namespace

struct Plan {};

Plan make_plan(const lab::GemmProblem &p) {
  LAB_REQUIRE(p.ab_dtype == lab::DType::F32 && p.c_dtype == lab::DType::F32,
              "this lab uses FP32 A, B and C");
  LAB_REQUIRE(p.m > 0 && p.n > 0 && p.k > 0, "M, N, K must be positive");
  return {};
}

void run(const Plan &, const lab::GemmProblem &p, cudaStream_t stream) {
  auto a = static_cast<const float *>(p.a);
  auto b = static_cast<const float *>(p.b);
  auto c = static_cast<float *>(p.c);
  switch (p.step) {
  case 1: {
    dim3 block(kNaiveX, kNaiveY);
    dim3 grid((p.n + kNaiveX - 1) / kNaiveX, (p.m + kNaiveY - 1) / kNaiveY);
    gemm_naive<<<grid, block, 0, stream>>>(a, b, c, p.m, p.n, p.k);
    break;
  }
  case 2: {
    dim3 block(kTile, kTile);
    dim3 grid((p.n + kTile - 1) / kTile, (p.m + kTile - 1) / kTile);
    gemm_smem<<<grid, block, 0, stream>>>(a, b, c, p.m, p.n, p.k);
    break;
  }
  case 3: {
    dim3 grid((p.n + kBN - 1) / kBN, (p.m + kBM - 1) / kBM);
    gemm_regblock<<<grid, kThreads, 0, stream>>>(a, b, c, p.m, p.n, p.k);
    break;
  }
  default: {
    dim3 grid((p.n + kBN - 1) / kBN, (p.m + kBM - 1) / kBM);
    gemm_vec<<<grid, kThreads, 0, stream>>>(a, b, c, p.m, p.n, p.k);
    break;
  }
  }
}

LAB_REGISTER(lab::GemmProblem, Plan, make_plan, run);
