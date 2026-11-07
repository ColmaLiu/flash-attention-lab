#pragma once
// Host-side tensor maps and shared launch state. Preparation is outside timing.
// Device synchronization and descriptors follow NVIDIA PTX ISA 8.7:
// https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace sm100_attention {
constexpr int BM = 128, BN = 128;
struct Prepared {
  CUtensorMap q, k, v;
  void *output;
  float *lse;
  int batch, n, heads, dim, sms;
  bool half, causal;
  bool polynomial = false, delayed_rescale = true, persistent = true;
  float scale;
  cudaStream_t stream;
};
inline void check(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
inline void encode(CUtensorMap *map, uintptr_t data, int b, int n, int h, int d,
                   int rows, bool half) {
  cuuint64_t dims[] = {static_cast<cuuint64_t>(d), static_cast<cuuint64_t>(h),
                       static_cast<cuuint64_t>(n), static_cast<cuuint64_t>(b)};
  cuuint64_t strides[] = {static_cast<cuuint64_t>(d * 2),
                          static_cast<cuuint64_t>(h * d * 2),
                          static_cast<cuuint64_t>(n) * h * d * 2};
  cuuint32_t box[] = {64, 1, static_cast<cuuint32_t>(rows), 1},
             steps[] = {1, 1, 1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      map,
      half ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      4, reinterpret_cast<void *>(data), dims, strides, box, steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(rc, &message);
    throw std::runtime_error(message ? message : "attention TMA encode failed");
  }
}
} // namespace sm100_attention
