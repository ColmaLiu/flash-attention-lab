// Reference solution for 00_transpose: y[c][r] = x[r][c].
//
// step 1: one thread per element. Reads are coalesced, writes are strided.
// step 2: a 32x32 tile goes through shared memory, so both the global read
//         and the global write are coalesced. Reading a tile column from
//         shared memory hits one bank 32 times (32-way bank conflict).
// step 3: pad each shared-memory row to 33 floats; a tile column now spans
//         32 different banks.
// step 4: a 64x64 tile with the same 256 threads: each thread keeps 16 loads
//         in flight instead of 4, which is what HBM needs to stay busy.
#include "lab/lab.h"

namespace {

constexpr int kThreadsX = 32; // one warp along a tile row
constexpr int kThreadsY = 8;  // 8 warps per block

__global__ void transpose_naive(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols)
    y[static_cast<int64_t>(c) * rows + r] =
        x[static_cast<int64_t>(r) * cols + c];
}

// One block transposes the kTile x kTile tile of x starting at (r0, c0).
// kPad extra floats per shared-memory row move tile columns to other banks.
template <int kTile, int kPad>
__global__ void transpose_tiled(const float *__restrict__ x,
                                float *__restrict__ y, int rows, int cols) {
  __shared__ float tile[kTile][kTile + kPad];
  int r0 = blockIdx.y * kTile;
  int c0 = blockIdx.x * kTile;

  // Load: lanes of a warp read consecutive columns of one x row.
#pragma unroll
  for (int i = threadIdx.y; i < kTile; i += kThreadsY) {
#pragma unroll
    for (int j = threadIdx.x; j < kTile; j += kThreadsX) {
      int r = r0 + i, c = c0 + j;
      if (r < rows && c < cols)
        tile[i][j] = x[static_cast<int64_t>(r) * cols + c];
    }
  }
  __syncthreads();

  // Store: lanes write consecutive columns of one y row. y row (c0 + i)
  // holds x column (c0 + i), so we read down column i of the tile.
#pragma unroll
  for (int i = threadIdx.y; i < kTile; i += kThreadsY) {
#pragma unroll
    for (int j = threadIdx.x; j < kTile; j += kThreadsX) {
      int yr = c0 + i, yc = r0 + j; // y is [cols, rows]
      if (yr < cols && yc < rows)
        y[static_cast<int64_t>(yr) * rows + yc] = tile[j][i];
    }
  }
}

template <int kTile, int kPad>
void launch_tiled(const float *x, float *y, int rows, int cols,
                  cudaStream_t stream) {
  dim3 block(kThreadsX, kThreadsY);
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  transpose_tiled<kTile, kPad><<<grid, block, 0, stream>>>(x, y, rows, cols);
}

} // namespace

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
    launch_tiled<32, 1>(x, y, p.rows, p.cols, stream);
    break;
  default:
    launch_tiled<64, 1>(x, y, p.rows, p.cols, stream);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
