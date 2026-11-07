#include "cuda_api.h"
#include <cmath>

namespace {
__device__ float warp_max(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
  return x;
}
__device__ float warp_sum(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, offset);
  return x;
}
// Short rows use independent warps, so several rows share each CTA.
template <int VALUES>
__global__ void small_rows(const float *x, float *y, int rows, int cols) {
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return; // Uniform within each warp.
  x += (int64_t)row * cols;
  y += (int64_t)row * cols;
  float values[VALUES];
  float maximum = -INFINITY;
  for (int i = 0; i < VALUES; ++i) {
    int col = lane + 32 * i;
    values[i] = col < cols ? x[col] : -INFINITY;
    maximum = fmaxf(maximum, values[i]);
  }
  maximum = warp_max(maximum);
  float denominator = 0;
  for (int i = 0; i < VALUES; ++i) {
    values[i] = expf(values[i] - maximum);
    denominator += values[i];
  }
  denominator = warp_sum(denominator);
  for (int i = 0; i < VALUES; ++i)
    if (lane + 32 * i < cols)
      y[lane + 32 * i] = values[i] / denominator;
}
// Cache a bounded wide row in registers: one global read and one exp per
// element. The unbounded fallback below retains support for arbitrary widths.
template <int VALUES>
__global__ void cached_wide_rows(const float *x, float *y, int cols) {
  __shared__ float partial[8];
  x += (int64_t)blockIdx.x * cols;
  y += (int64_t)blockIdx.x * cols;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float values[VALUES], maximum = -INFINITY;
#pragma unroll
  for (int i = 0; i < VALUES; ++i) {
    int c = threadIdx.x + i * 256;
    values[i] = c < cols ? x[c] : -INFINITY;
    maximum = fmaxf(maximum, values[i]);
  }
  maximum = warp_max(maximum);
  if (lane == 0)
    partial[warp] = maximum;
  __syncthreads();
  maximum = warp_max(lane < 8 ? partial[lane] : -INFINITY);
  __syncthreads();
  float denominator = 0.f;
#pragma unroll
  for (int i = 0; i < VALUES; ++i) {
    values[i] = expf(values[i] - maximum);
    denominator += values[i];
  }
  denominator = warp_sum(denominator);
  if (lane == 0)
    partial[warp] = denominator;
  __syncthreads();
  denominator = warp_sum(lane < 8 ? partial[lane] : 0.f);
#pragma unroll
  for (int i = 0; i < VALUES; ++i) {
    int c = threadIdx.x + i * 256;
    if (c < cols)
      y[c] = values[i] / denominator;
  }
}
// Wide rows use a CTA. Reloading the input bounds register storage
// independently of the supported width; shared storage contains only inter-warp
// partials.
__global__ void wide_rows(const float *x, float *y, int cols) {
  __shared__ float partial[8];
  x += (int64_t)blockIdx.x * cols;
  y += (int64_t)blockIdx.x * cols;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < cols; c += 256)
    maximum = fmaxf(maximum, x[c]);
  maximum = warp_max(maximum);
  if (lane == 0)
    partial[warp] = maximum;
  __syncthreads();
  maximum = warp_max(lane < 8 ? partial[lane] : -INFINITY);
  __syncthreads();
  float denominator = 0;
  for (int c = threadIdx.x; c < cols; c += 256)
    denominator += expf(x[c] - maximum);
  denominator = warp_sum(denominator);
  if (lane == 0)
    partial[warp] = denominator;
  __syncthreads();
  denominator = warp_sum(lane < 8 ? partial[lane] : 0.0f);
  for (int c = threadIdx.x; c < cols; c += 256)
    y[c] = expf(x[c] - maximum) / denominator;
}
} // namespace
void student_launch(const TensorMap &inputs, const TensorMap &outputs,
                    const TensorMap &, const Options &, cudaStream_t stream) {
  const auto &x = inputs.at("x");
  if (x.dtype != DType::F32)
    throw std::runtime_error("s01 solution requires FP32");
  int rows = x.shape[0], cols = x.shape[1];
  const float *in = static_cast<const float *>(x.data);
  float *out = static_cast<float *>(outputs.at("y").data);
  if (cols <= 32)
    small_rows<1><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 64)
    small_rows<2><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 128)
    small_rows<4><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 256)
    small_rows<8><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 512)
    small_rows<16><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 1024)
    small_rows<32><<<(rows + 3) / 4, 128, 0, stream>>>(in, out, rows, cols);
  else if (cols <= 2048)
    cached_wide_rows<8><<<rows, 256, 0, stream>>>(in, out, cols);
  else if (cols <= 4096)
    cached_wide_rows<16><<<rows, 256, 0, stream>>>(in, out, cols);
  else if (cols <= 8192)
    cached_wide_rows<32><<<rows, 256, 0, stream>>>(in, out, cols);
  else if (cols <= 16384)
    cached_wide_rows<64><<<rows, 256, 0, stream>>>(in, out, cols);
  else
    wide_rows<<<rows, 256, 0, stream>>>(in, out, cols);
}
