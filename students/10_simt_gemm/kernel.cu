// 10_simt_gemm: C[M,N] = A[M,K] . B[N,K]^T, FP32, CUDA cores only.
// Lesson: course/10_simt_gemm/README.md
//
// All matrices are row-major (last dimension contiguous):
//   A is [M, K]: A[m][k] is at a[m * K + k]
//   B is [N, K]: B[n][k] is at b[n * K + k]   (K-major, like K in Q . K^T)
//   C is [M, N]: C[m][n] is at c[m * N + n]
//   C[m][n] = sum over k of A[m][k] * B[n][k]   (row m of A dot row n of B)
// run() picks a version by p.step, so all versions stay side by side and you
// can compare them:  ./lab bench 10 --step 1   ...   ./lab bench 10 --step 4
#include "lab/lab.h"

namespace {

// ---------------------------------------------------------------------------
// TODO(1) — step 1: one thread per element of C.
// run() launches 32 x 8-thread blocks covering C, threadIdx.x along N.
//   1. Compute this thread's (row, col) = (m, n); return if outside C.
//   2. acc = dot product of row `row` of A and row `col` of B (length K).
//   3. Write acc to C[row][col].
// Use int64_t when you form row * K or row * N.
// ---------------------------------------------------------------------------
constexpr int kNaiveX = 32;
constexpr int kNaiveY = 8;

__global__ void gemm_naive(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int m, int n, int k) {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(2) — step 2: shared-memory tiling, still one output per thread.
// A block of 32 x 32 threads (tx = threadIdx.x, ty = threadIdx.y) computes
// the 32 x 32 tile of C whose top-left element is
// (m0, n0) = (blockIdx.y * 32, blockIdx.x * 32); thread (tx, ty) owns
// C[m0 + ty][n0 + tx]. Loop over K in slices of width kTile = 32:
//   1. Load as[ty][tx] = A[m0 + ty][k0 + tx] and bs[ty][tx] = B[n0 + ty][k0 + tx]
//      (lanes read consecutive k: coalesced). Store 0 for elements outside
//      A or B, so the tail slice adds nothing.
//   2. __syncthreads()
//   3. acc += as[ty][kk] * bs[tx][kk] for kk in 0..31
//   4. __syncthreads() — why is this second one needed?
// After the loop, write acc if (m0 + ty, n0 + tx) is inside C.
// The tiles are padded to 33 floats per row: step 3 of the compute reads a
// column of bs across the warp (tx varies). Work out the banks (lab 00 §4.3).
// ---------------------------------------------------------------------------
constexpr int kTile = 32;

__global__ void gemm_smem(const float *__restrict__ a,
                          const float *__restrict__ b, float *__restrict__ c,
                          int m, int n, int k) {
  __shared__ float as[kTile][kTile + 1];
  __shared__ float bs[kTile][kTile + 1];
  LAB_TODO(2);
}

// ---------------------------------------------------------------------------
// Steps 3-4: register blocking.
// Block tile BM x BN = 128 x 128, K slice BK = 8, 256 threads viewed as
// 16 x 16: tx = threadIdx.x % 16, ty = threadIdx.x / 16.
// Each thread accumulates an 8 x 8 micro-tile of C in registers (acc).
// ---------------------------------------------------------------------------
constexpr int kBM = 128, kBN = 128, kBK = 8;
constexpr int kTM = 8, kTN = 8;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN); // 256

// Provided: copy src[r0 : r0+128, k0 : k0+8] of a [rows, k] K-major matrix
// into tile[128][kBK + 1] with the same K-major layout, zeros outside the
// matrix. All 256 threads must call it (each copies 4 elements).
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

// ---------------------------------------------------------------------------
// TODO(3) — step 3: the register-blocked compute and the epilogue.
// Ownership (strided): thread (tx, ty) owns
//   acc[i][j] = C[m0 + ty + 16*i][n0 + tx + 16*j],   i, j in 0..7.
// For each K slice, after the tiles are staged:
//   1. For each kk in 0..7, load the 8 A values as[ty + 16*i][kk] into a
//      register array af[8] and the 8 B values bs[tx + 16*j][kk] into bf[8],
//      then do the 8 x 8 outer product acc[i][j] += af[i] * bf[j].
//      Use #pragma unroll on every loop: acc must stay in registers, which
//      needs compile-time indices.
// After the K loop:
//   2. Write every acc[i][j] whose (row, col) is inside C.
// Invariant: after slice k0, acc[i][j] = sum over k < k0 + 8 of A[row][k] *
// B[col][k] (zeros past K add nothing).
// ---------------------------------------------------------------------------
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

    LAB_TODO(3); // 1. outer products over kk

    __syncthreads(); // all reads done before the next slice overwrites
  }
  LAB_TODO(3); // 2. write acc to C
}

// Provided: four consecutive elements src[row][col .. col+3] of a
// [rows, cols] row-major matrix, zero outside it. With vec == true and all
// four inside, this is one 16-byte load; vec requires cols % 4 == 0 so that
// every row starts 16-byte aligned (col must be a multiple of 4).
__device__ __forceinline__ float4 load4(const float *__restrict__ src, int row,
                                        int col, int rows, int cols, bool vec) {
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (row >= rows)
    return v;
  const float *p = src + static_cast<int64_t>(row) * cols + col;
  if (vec && col + 3 < cols)
    return *reinterpret_cast<const float4 *>(p);
  if (col + 0 < cols) v.x = p[0];
  if (col + 1 < cols) v.y = p[1];
  if (col + 2 < cols) v.z = p[2];
  if (col + 3 < cols) v.w = p[3];
  return v;
}

// Provided: store v to dst[row][col .. col+3], skipping elements outside the
// [rows, cols] matrix; one 16-byte store when vec and all four are inside.
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
  if (col + 0 < cols) p[0] = v.x;
  if (col + 1 < cols) p[1] = v.y;
  if (col + 2 < cols) p[2] = v.z;
  if (col + 3 < cols) p[3] = v.w;
}

// ---------------------------------------------------------------------------
// TODO(4) — step 4: same tiling, transposed tiles and 128-bit accesses.
// Shared memory now holds the tiles transposed: as[kk][r] = A[m0 + r][k0 + kk]
// (likewise bs with B and n0), rows padded to 128 + 4 floats.
// Ownership (two groups of 4, 64 apart):
//   row(i) = m0 + (i / 4) * 64 + ty * 4 + i % 4,   i in 0..7
//   col(j) = n0 + (j / 4) * 64 + tx * 4 + j % 4,   j in 0..7
// For each K slice:
//   1. Load: thread tid reads ONE float4 of A with load4: tile row
//      lr = tid / 2, k columns lk = (tid % 2) * 4 .. +3, i.e. A[m0 + lr][k0 +
//      lk ..]; same for B. Pass vec = (k % 4 == 0). Store the four values
//      transposed: as[lk + 0][lr] = v.x, ..., as[lk + 3][lr] = v.w.
//   2. __syncthreads()
//   3. For each kk: af[0..3] = as[kk][ty*4 .. +3] and af[4..7] =
//      as[kk][64 + ty*4 .. +3], each as ONE float4 read (reinterpret_cast
//      to const float4 *); bf likewise from bs with tx. Then the same 8 x 8
//      outer product as step 3.
//   4. __syncthreads()
// After the loop, write each row's two groups of 4 columns with store4
// (vec = (n % 4 == 0)).
// ---------------------------------------------------------------------------
constexpr int kPadT = 4;

__global__ void __launch_bounds__(kThreads)
    gemm_vec(const float *__restrict__ a, const float *__restrict__ b,
             float *__restrict__ c, int m, int n, int k) {
  __shared__ __align__(16) float as[kBK][kBM + kPadT];
  __shared__ __align__(16) float bs[kBK][kBN + kPadT];
  int tid = threadIdx.x;
  int tx = tid % 16, ty = tid / 16;
  int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  LAB_TODO(4);
}

} // namespace

// Nothing to prepare in this lab.
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
