#include "cuda_api.h"

// A square shared tile turns strided output columns into coalesced rows.
// The extra column changes the bank stride when the tile is read transposed.
namespace {
__global__ void transpose(const float *x, float *y, int rows, int cols) {
  __shared__ float tile[32][33];
  int row = blockIdx.y * 32 + threadIdx.y;
  int col = blockIdx.x * 32 + threadIdx.x;
  for (int offset = 0; offset < 32; offset += 8)
    if (row + offset < rows && col < cols)
      tile[threadIdx.y + offset][threadIdx.x] =
          x[(int64_t)(row + offset) * cols + col];
  __syncthreads();
  row = blockIdx.x * 32 + threadIdx.y;
  col = blockIdx.y * 32 + threadIdx.x;
  for (int offset = 0; offset < 32; offset += 8)
    if (row + offset < cols && col < rows)
      y[(int64_t)(row + offset) * rows + col] =
          tile[threadIdx.x][threadIdx.y + offset];
}
} // namespace

void student_launch(const TensorMap &inputs, const TensorMap &outputs,
                    const TensorMap &, const Options &, cudaStream_t stream) {
  const auto &x = inputs.at("x");
  if (x.dtype != DType::F32)
    throw std::runtime_error("s00 solution requires FP32");
  int rows = x.shape[0], cols = x.shape[1];
  transpose<<<dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8), 0,
              stream>>>(static_cast<const float *>(x.data),
                        static_cast<float *>(outputs.at("y").data), rows, cols);
}
