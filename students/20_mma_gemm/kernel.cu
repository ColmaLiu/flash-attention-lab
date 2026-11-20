// 20_mma_gemm: C[M,N] = A[M,K] . B[N,K]^T on Tensor Cores (mma.sync).
// Lesson: course/20_mma_gemm/README.md
//
// A is [M, K], B is [N, K], both BF16, row-major (K fastest, "K-major");
// C is [M, N] FP32, row-major:
//   A[m][k] is at a[m * K + k],  B[n][k] at b[n * K + k],  C[m][n] at c[m * N + n].
// Supported: K % 8 == 0 (rows are whole 16-byte chunks); step 1 also needs
// M % 16 == 0, N % 8 == 0, K % 16 == 0. make_plan rejects anything else.
// run() picks a version by p.step:  ./lab bench 20 --step 1 ... --step 3
#include "lab/lab.h"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;

// ===========================================================================
// Primitives. You write them here; from lab 21 on they come from
// include/lab/mma_sync.cuh. Notation: lane = threadIdx.x % 32,
// group = lane / 4 (0..7), quad = lane % 4 (0..3). A 32-bit register holding
// two 16-bit values keeps the one with the smaller column (or k) in its low
// 16 bits -- the same order as two consecutive bf16 in memory.
// ===========================================================================

struct FragCoord {
  int row, col;
};

// ---------------------------------------------------------------------------
// TODO(1a) — d[16x8] += a[16x16] * b[16x8] for the whole warp:
//   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//     {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
// Use inline asm with d as both C and D: "+f" constraints for d[0..3],
// "r" for the six 32-bit operand registers. (Lesson §4.2.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(1b) — fragment coordinates of mma.m16n8k16 (lesson §4.3, PTX ISA
// "Matrix Fragments for mma.m16n8k16 with floating point type").
//   frag_a_coord(lane, i): A register i (0..3) holds A[row][col] (low half)
//                          and A[row][col + 1] (high half). Return {row, col}.
//   frag_b_coord(lane, i): B register i (0..1) holds B[k][n] and B[k+1][n]
//                          of the 16x8 (k x n) operand. Return {row=k, col=n}.
//   frag_c_coord(lane, i): accumulator float i (0..3) is D[row][col].
// Each is one line of arithmetic on lane / 4, lane % 4 and i.
// ---------------------------------------------------------------------------
__device__ __forceinline__ FragCoord frag_a_coord(int lane, int i) {
  LAB_TODO(1);
  return {0, 0};
}
__device__ __forceinline__ FragCoord frag_b_coord(int lane, int i) {
  LAB_TODO(1);
  return {0, 0};
}
__device__ __forceinline__ FragCoord frag_c_coord(int lane, int i) {
  LAB_TODO(1);
  return {0, 0};
}

// ---------------------------------------------------------------------------
// TODO(2a) — ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
// Loads four 8x8 bf16 matrices from shared memory. Lanes 8i .. 8i+7 pass
// the addresses of rows 0..7 of matrix i (each row: 16 contiguous bytes,
// 16-byte aligned). Afterwards r[i] of lane l holds row l/4, elements
// 2*(l%4) and 2*(l%4)+1 of matrix i. `addr` is a 32-bit shared-memory
// address: static_cast<uint32_t>(__cvta_generic_to_shared(p)). (Lesson §4.4.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  LAB_TODO(2);
}

// ---------------------------------------------------------------------------
// TODO(3) — XOR swizzle for a shared tile whose rows are 64 bytes (4 chunks
// of 16 bytes; kBK = 32 bf16). Return the physical chunk (0..3) that stores
// logical chunk `chunk` of row `row`, such that the same logical chunk of 8
// consecutive rows falls into 8 different 16-byte bank groups (lesson §4.5).
// It must be a permutation of 0..3 within each row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swizzle_chunk4(int row, int chunk) {
  LAB_TODO(3);
  return chunk;
}

// ===========================================================================
// Step 1: one warp computes one 16x8 tile of C.
// ===========================================================================
constexpr int kNaiveWarps = 4; // warps per block, stacked along M

// ---------------------------------------------------------------------------
// TODO(1c) — warp `warp` of block (blockIdx.x, blockIdx.y) owns the C tile
// with top-left (m0, n0) computed below. For k0 = 0, 16, 32, ... < k:
//   1. fill uint32_t af[4] from A: register i is the pair at
//      A[m0 + row][k0 + col], {row, col} = frag_a_coord(lane, i). The two
//      bf16 are adjacent in memory, so read them as one uint32_t.
//   2. fill uint32_t bfr[2] from B the same way with frag_b_coord. Careful:
//      B is stored as [N][K], i.e. B[n][k] is at b[n * K + k].
//   3. mma_m16n8k16_bf16(d, af, bfr).
// Then write the 4 floats of d to C with frag_c_coord.
// Use int64_t for linear indices.
// ---------------------------------------------------------------------------
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
  LAB_TODO(1);
}

// ===========================================================================
// Steps 2 and 3: a 128x128 CTA tile, 8 warps of 64x32, K tiles of 32.
// ===========================================================================
constexpr int kBM = 128, kBN = 128, kBK = 32;
constexpr int kWarpsM = 2, kWarpsN = 4;      // 8 warps
constexpr int kWM = kBM / kWarpsM;           // 64 rows per warp
constexpr int kWN = kBN / kWarpsN;           // 32 cols per warp
constexpr int kMI = kWM / 16, kNJ = kWN / 8; // 4 x 4 mma tiles per warp
constexpr int kThreads = kWarpsM * kWarpsN * 32;
constexpr int kChunks = kBK / 8; // 16-byte chunks per tile row (= 4)

// Element offset of (row, col) in a [rows][kBK] shared tile; col is a
// multiple of 8 (we only address whole 16-byte chunks).
template <bool kSwizzle> __device__ __forceinline__ int tile_offset(int row,
                                                                   int col) {
  int chunk = col / 8;
  if constexpr (kSwizzle)
    chunk = swizzle_chunk4(row, chunk);
  return row * kBK + chunk * 8 + col % 8;
}

// Copy a kRows x kBK slice (rows row0.., cols k0..) of a K-major matrix into
// shared memory, 16 bytes per thread per iteration. Chunks outside the
// matrix are written as zeros, so they add nothing to the dot products.
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
  __shared__ __align__(128) bf16 sa[kBM * kBK]; // A tile: [128][32]
  __shared__ __align__(128) bf16 sb[kBN * kBK]; // B tile: [128][32] = [n][k]
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

    // -----------------------------------------------------------------------
    // TODO(2b) — for kk = 0, 16 (the two k16 slices of this K tile):
    //   1. A fragments: for i < kMI, one ldmatrix_x4 loads the 16x16 block
    //      rows wm + 16i .. +15, cols kk .. kk+15 into uint32_t af[i][4].
    //      Which (row, col) must lane l point at so that r[0..3] come out in
    //      the a0..a3 order of frag_a_coord? (lesson §5.2)
    //   2. B fragments: sb holds B as [n][k], which already is the layout the
    //      .col operand wants (no .trans). One ldmatrix_x4 covers two n8
    //      tiles j and j+1: bfr[j][0..1] and bfr[j+1][0..1].
    //   3. kMI x kNJ calls of mma_m16n8k16_bf16(acc[i][j], af[i], bfr[j]).
    //   Always address shared memory through tile_offset<kSwizzle>(row, col).
    // -----------------------------------------------------------------------
    LAB_TODO(2);
    __syncthreads(); // every warp is done reading before the next load
  }

  // -------------------------------------------------------------------------
  // TODO(2c) — epilogue: acc[i][j][e] is C[m0 + wm + 16i + row][n0 + wn + 8j
  // + col] with {row, col} = frag_c_coord(lane, e). Skip elements outside
  // the matrix (row >= m or col >= n).
  // -------------------------------------------------------------------------
  LAB_TODO(2);
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
