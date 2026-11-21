// Reference solution for 00_transpose: y[c][r] = x[r][c].
//
// step 1: one thread per element. Reads are coalesced, writes are strided.
// step 2: a 32x32 tile goes through shared memory, so both the global read
//         and the global write are coalesced. Reading a tile column from
//         shared memory hits one bank 32 times (32-way bank conflict).
// step 3: pad each shared-memory row to 33 floats; a tile column now spans
//         32 different banks.
// step 4: the same tile with 32x2 instead of 32x8 threads: each thread now
//         issues 16 independent loads back to back instead of 4, so more
//         bytes are in flight per SM (Little's law).
#include "lab/lab.h"

namespace {

constexpr int kTile = 32; // the tile is kTile x kTile elements

__global__ void transpose_naive(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols)
    y[static_cast<int64_t>(c) * rows + r] =
        x[static_cast<int64_t>(r) * cols + c];
}

// The block has 32 x kThreadsY threads; each thread moves kTile / kThreadsY
// elements. kPad extra floats per shared-memory row move tile columns to
// other banks.
template <int kPad, int kThreadsY>
__global__ void transpose_tiled(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  __shared__ float tile[kTile][kTile + kPad];
  int r0 = blockIdx.y * kTile;
  int c0 = blockIdx.x * kTile;
  int j = threadIdx.x; // column inside the tile (load) / row of y (store)

  // Load: lanes of a warp read consecutive columns of one x row.
  // The trip count is a compile-time constant, so the loop is fully unrolled
  // and all kTile / kThreadsY loads are issued before the first one returns.
#pragma unroll
  for (int k = 0; k < kTile / kThreadsY; ++k) {
    int i = threadIdx.y + k * kThreadsY;
    int r = r0 + i, c = c0 + j;
    if (r < rows && c < cols)
      tile[i][j] = x[static_cast<int64_t>(r) * cols + c];
  }
  __syncthreads();

  // Store: lanes write consecutive columns of one y row. y row (c0 + i)
  // holds x column (c0 + i), so we read down column i of the tile.
#pragma unroll
  for (int k = 0; k < kTile / kThreadsY; ++k) {
    int i = threadIdx.y + k * kThreadsY;
    int yr = c0 + i, yc = r0 + j; // y is [cols, rows]
    if (yr < cols && yc < rows)
      y[static_cast<int64_t>(yr) * rows + yc] = tile[j][i];
  }
}

template <int kPad, int kThreadsY>
void launch_tiled(const float *x, float *y, int rows, int cols,
                  cudaStream_t stream) {
  dim3 block(32, kThreadsY);
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<kPad, kThreadsY>
      <<<grid, block, 0, stream>>>(x, y, rows, cols);
}

} // namespace

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
    launch_tiled<0, 8>(x, y, p.rows, p.cols, stream);
    break;
  case 3:
    launch_tiled<1, 8>(x, y, p.rows, p.cols, stream);
    break;
  default:
    launch_tiled<1, 2>(x, y, p.rows, p.cols, stream);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
