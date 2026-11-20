// 00_transpose: y[c][r] = x[r][c]
// Lesson: course/00_transpose/README.md
//
// x is [rows, cols], y is [cols, rows], both FP32 and row-major:
//   x[r][c] is at x[r * cols + c],   y[c][r] is at y[c * rows + r].
// run() picks a version by p.step, so all versions stay side by side and you
// can compare them:  ./lab bench 00 --step 1   ...   ./lab bench 00 --step 4
#include "lab/lab.h"

namespace {

constexpr int kThreadsX = 32; // one warp along a tile row
constexpr int kThreadsY = 8;  // 8 warps per block

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
// One block (32 x 8 threads) owns the kTile x kTile tile of x whose top-left
// element is (r0, c0) = (blockIdx.y * kTile, blockIdx.x * kTile).
//   1. Load the tile: a warp (fixed threadIdx.y) reads consecutive columns of
//      one x row. With kTile = 32, each thread loads 32 / 8 = 4 elements;
//      write the loops so they also work for kTile = 64 (step 4).
//   2. __syncthreads(): every element must be in shared memory before any
//      thread reads another thread's element.
//   3. Store: a warp writes consecutive columns of one y row. Which tile
//      element does y[c0 + i][r0 + j] need?
//   Check r < rows and c < cols for every load and store (tail tiles).
// kPad extra floats per row: only the declaration of `tile` uses it.
// ---------------------------------------------------------------------------
template <int kTile, int kPad>
__global__ void transpose_tiled(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  __shared__ float tile[kTile][kTile + kPad];
  LAB_TODO(2);
}

template <int kTile, int kPad>
void launch_tiled(const float *x, float *y, int rows, int cols,
                  cudaStream_t stream) {
  dim3 block(kThreadsX, kThreadsY);
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<kTile, kPad><<<grid, block, 0, stream>>>(x, y, rows, cols);
}

} // namespace

// Nothing to prepare in this lab.
struct Plan {};

Plan make_plan(const lab::TransposeProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F32, "this lab uses FP32");
  return {};
}

void run(const Plan &, const lab::TransposeProblem &p, cudaStream_t stream) {
  auto x = static_cast<const float *>(p.x);
  auto y = static_cast<float *>(p.y);
  switch (p.step) {
  case 1: {
    dim3 block(kThreadsX, kThreadsY);
    dim3 grid((p.cols + kThreadsX - 1) / kThreadsX,
              (p.rows + kThreadsY - 1) / kThreadsY);
    transpose_naive<<<grid, block, 0, stream>>>(x, y, p.rows, p.cols);
    break;
  }
  case 2:
    launch_tiled<32, 0>(x, y, p.rows, p.cols, stream);
    break;
  case 3:
    // TODO(3) — step 3: choose kPad so that reading a tile column touches
    // 32 different banks, then replace LAB_TODO(3) with the launch.
    LAB_TODO(3); // launch_tiled<32, kPad>(x, y, p.rows, p.cols, stream);
    break;
  default:
    // TODO(4) — step 4: a 64 x 64 tile with the same 256 threads, so each
    // thread has 16 loads in flight. Replace LAB_TODO(4) with the launch.
    LAB_TODO(4);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
