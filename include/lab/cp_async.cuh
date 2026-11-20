// Asynchronous global -> shared copies (cp.async, sm_80+). Written by learners
// in 21_pipeline_gemm; provided from 22_fa2 on.
//
// Every function wraps one PTX instruction and contains no algorithm logic.
// PTX section names refer to the PTX ISA (version 8.7),
// https://docs.nvidia.com/cuda/parallel-thread-execution/
//
// Model: each thread issues copies, then groups all its not-yet-committed
// copies with commit_group. wait_group<N> blocks until at most N of this
// thread's groups are still in flight. Completion is per thread: to read data
// copied by other threads, wait_group must be followed by __syncthreads().
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lab {

// ---------------------------------------------------------------------------
// cp.async.cg.shared.global [dst], [src], 16;
// PTX: "Data Movement and Conversion Instructions: cp.async".
// Starts copying 16 bytes from global `src` to shared `dst` and returns
// immediately; the data is not visible until a matching wait_group.
// Both addresses must be 16-byte aligned. .cg caches in L2 only (bypasses
// L1), the only cache mode allowed for 16-byte copies besides .ca.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src)
               : "memory");
}

// cp.async.cg.shared.global [dst], [src], 16, src_size;
// Same, with src-size = valid ? 16 : 0. The bytes past src-size are filled
// with zeros, so valid = false writes 16 zero bytes and reads nothing.
// `src` must still be a valid-looking global address (pass the tensor base).
__device__ __forceinline__ void cp_async_16_zfill(void *dst, const void *src,
                                                  bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src), "r"(valid ? 16 : 0)
               : "memory");
}

// cp.async.commit_group;
// PTX: "Data Movement and Conversion Instructions: cp.async.commit_group".
// Closes the group of all cp.async issued by this thread since the last
// commit. Committing with no pending copies creates an empty group, which
// completes immediately (useful to keep group counting uniform).
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// cp.async.wait_group N;
// PTX: "Data Movement and Conversion Instructions: cp.async.wait_group /
// cp.async.wait_all".
// Waits until at most N of this thread's most recent groups are pending;
// all older groups are complete and their data is visible to this thread.
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
  static_assert(N >= 0, "N is a non-negative immediate");
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// cp.async.wait_all;  (= commit_group + wait_group 0)
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

} // namespace lab
