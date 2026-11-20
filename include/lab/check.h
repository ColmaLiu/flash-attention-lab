// Error checking and the LAB_TODO marker.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Throw a readable error if a CUDA runtime call fails (host code only).
#define LAB_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t lab_err_ = (expr);                                             \
    if (lab_err_ != cudaSuccess)                                               \
      throw std::runtime_error(                                                \
          std::string(#expr) + " failed: " + cudaGetErrorString(lab_err_) +    \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));                 \
  } while (0)

// Throw if a host-side condition does not hold, e.g. an unsupported shape.
#define LAB_REQUIRE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond))                                                               \
      throw std::runtime_error(std::string("requirement failed: ") + (msg) +   \
                               " (" #cond ")");                                \
  } while (0)

#ifdef __CUDACC__
// Set by LAB_REGISTER to host-mapped memory, so the harness can read which
// TODO was hit even after the kernel traps.
static __device__ int *lab_todo_flag_;
#endif

namespace lab {

// Marks unfinished code. On the host it throws; on the device it records the
// TODO number and traps, so the harness reports "TODO(n) not implemented"
// instead of a wrong answer or a hang.
__host__ __device__ inline void todo(int n) {
#ifdef __CUDA_ARCH__
  if (lab_todo_flag_ != nullptr) {
    *reinterpret_cast<volatile int *>(lab_todo_flag_) = n;
    __threadfence_system();
  }
  __trap();
#else
  throw std::runtime_error("LAB_TODO(" + std::to_string(n) +
                           "): not implemented yet");
#endif
}

} // namespace lab

#define LAB_TODO(n) ::lab::todo(n)
