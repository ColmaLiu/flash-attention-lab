// Global-memory scratch buffers owned by a Plan.
#pragma once

#include <cstddef>
#include <utility>
#include "lab/check.h"

namespace lab {

// Allocated once in make_plan (not timed) and freed with the Plan.
// The contents start as garbage (all bytes 0xFF, i.e. NaN for floats):
// initialize whatever your algorithm needs inside run().
class Scratch {
public:
  Scratch() = default;
  explicit Scratch(size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0)
      return;
    LAB_CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
    LAB_CUDA_CHECK(cudaMemset(ptr_, 0xFF, bytes_));
  }
  Scratch(const Scratch &) = delete;
  Scratch &operator=(const Scratch &) = delete;
  Scratch(Scratch &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {
  }
  Scratch &operator=(Scratch &&other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }
  ~Scratch() {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
  }

  template <class T> T *as() const {
    return static_cast<T *>(ptr_);
  }
  size_t bytes() const {
    return bytes_;
  }

private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
};

} // namespace lab
