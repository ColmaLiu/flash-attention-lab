// 00_transpose: y[c][r] = x[r][c]
// Lesson: course/00_transpose/README.md
//
// x is [rows, cols], y is [cols, rows], both FP32 and row-major:
//   x[r][c] is at x[r * cols + c],   y[c][r] is at y[c * rows + r].
// run() picks a version by p.step, so all versions stay side by side and you
// can compare them:  ./lab bench 00 --step 1   ...   ./lab bench 00 --step 4
#include "lab/lab.h"

namespace {

constexpr int kTile = 32; // the tile is kTile x kTile elements

// ---------------------------------------------------------------------------
// TODO(1) — step 1: one thread per element.
// run() launches 32x8-thread blocks that cover x, threadIdx.x along columns.
//   1. Compute this thread's (r, c) from blockIdx, blockDim and threadIdx.
//   2. Return if (r, c) is outside the matrix (the grid is rounded up).
//   3. Copy x[r][c] to y[c][r].
// Use int64_t for linear indices: rows * cols can exceed 2^31.
// ---------------------------------------------------------------------------
__global__ void transpose_naive(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(2) — step 2: transpose through a tile in shared memory.
// One block of 32 x kThreadsY threads owns the 32 x 32 tile of x whose
// top-left element is (r0, c0) = (blockIdx.y * kTile, blockIdx.x * kTile).
// Each thread moves kTile / kThreadsY elements (4 when kThreadsY = 8).
//   1. Load the tile: a warp (fixed threadIdx.y) reads consecutive columns of
//      one x row. Write the loop with a compile-time trip count:
//          #pragma unroll
//          for (int k = 0; k < kTile / kThreadsY; ++k) {
//            int i = threadIdx.y + k * kThreadsY;   // tile row
//            ...
//          }
//      (lesson 4.5 explains why `for (i = threadIdx.y; i < kTile; ...)` is
//      slower: the compiler cannot issue all loads back to back.)
//   2. __syncthreads(): every element must be in shared memory before any
//      thread reads another thread's element.
//   3. Store: a warp writes consecutive columns of one y row. Which tile
//      element does y[c0 + i][r0 + threadIdx.x] need?
//   Check r < rows and c < cols for every load and store (tail tiles).
// kPad extra floats per row: only the declaration of `tile` uses it.
// ---------------------------------------------------------------------------
template <int kPad, int kThreadsY>
__global__ void transpose_tiled(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  __shared__ float tile[kTile][kTile + kPad];
  LAB_TODO(2);
}

template <int kPad, int kThreadsY>
void launch_tiled(const float *x, float *y, int rows, int cols,
                  cudaStream_t stream) {
  dim3 block(32, kThreadsY);
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<kPad, kThreadsY><<<grid, block, 0, stream>>>(x, y, rows,
                                                               cols);
}

} // namespace

// Nothing to prepare in this lab.
struct Plan {};

Plan make_plan(const lab::TransposeProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F32, "this lab uses FP32");
  // gridDim.y is at most 65535; step 1 uses one block row per 8 rows.
  LAB_REQUIRE(p.rows <= 65535 * 8, "rows must be at most 524280");
  return {};
}

void run(const Plan &, const lab::TransposeProblem &p, cudaStream_t stream) {
  auto x = static_cast<const float *>(p.x);
  auto y = static_cast<float *>(p.y);
  switch (p.step) {
  case 1: {
    dim3 block(32, 8);
    dim3 grid((p.cols + 31) / 32, (p.rows + 7) / 8);
    transpose_naive<<<grid, block, 0, stream>>>(x, y, p.rows, p.cols);
    break;
  }
  case 2:
    launch_tiled</*kPad=*/0, /*kThreadsY=*/8>(x, y, p.rows, p.cols, stream);
    break;
  case 3:
    // TODO(3) — step 3: choose kPad so that reading a tile column touches
    // 32 different banks, then replace LAB_TODO(3) with the launch.
    LAB_TODO(3); // launch_tiled<kPad, 8>(x, y, p.rows, p.cols, stream);
    break;
  default:
    // TODO(4) — step 4: keep the tile and the padding, but use fewer threads
    // per block so that each thread has 16 loads in flight (lesson 4.5).
    // Choose kThreadsY and replace LAB_TODO(4) with the launch.
    LAB_TODO(4);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
