// TMA (Tensor Memory Accelerator): tensor maps on the host, bulk tensor
// copies between global and shared memory on the device. Written by learners
// in 30_tma_mbarrier; provided from 31_tcgen05_gemm on.
//
// Every function wraps one driver call or one PTX instruction and contains no
// algorithm logic. PTX section names refer to the PTX ISA (version 8.7),
// https://docs.nvidia.com/cuda/parallel-thread-execution/
//
// Model: a CUtensorMap (128 bytes, encoded once on the host) describes a
// tensor in global memory (base, sizes, strides), the box one copy moves,
// and how the box is laid out in shared memory (swizzle). A single thread
// issues a copy with box coordinates; the hardware moves the whole box.
//   load : global -> shared, completion reported in bytes to an mbarrier
//          (complete_tx); out-of-bounds elements are filled with zeros and
//          still count toward the bytes.
//   store: shared -> global, completion tracked per thread with bulk
//          commit/wait groups; out-of-bounds elements are not written.
#pragma once

#include <cstdint>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lab/check.h"
#include "lab/mbarrier.cuh"

namespace lab {

// ===========================================================================
// Host: encoding tensor maps (CUDA Driver API, "Tensor Map Object Management",
// cuTensorMapEncodeTiled).
// ===========================================================================

inline int tma_elem_bytes(CUtensorMapDataType t) {
  switch (t) {
  case CU_TENSOR_MAP_DATA_TYPE_UINT8:
    return 1;
  case CU_TENSOR_MAP_DATA_TYPE_UINT16:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
  case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    return 2;
  case CU_TENSOR_MAP_DATA_TYPE_UINT32:
  case CU_TENSOR_MAP_DATA_TYPE_INT32:
  case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    return 4;
  default:
    return 0;
  }
}

// Tiled tensor map of rank 1..5. dims[0] is the contiguous (innermost)
// dimension, in elements. strides_bytes[i] is the byte distance between
// consecutive indices of dimension i + 1 (rank - 1 entries; each a multiple
// of 16). box[i] <= 256 elements; box[0] * element size must be a multiple
// of 16 bytes, and with a swizzle at most the swizzle span (32/64/128 B).
// The base must be 16-byte aligned. Out-of-bounds elements of a load are
// zero-filled (FLOAT_OOB_FILL_NONE).
template <int kRank>
CUtensorMap
make_tensor_map(const void *base, CUtensorMapDataType dtype,
                const uint64_t (&dims)[kRank],
                const uint64_t (&strides_bytes)[kRank > 1 ? kRank - 1 : 1],
                const uint32_t (&box)[kRank], CUtensorMapSwizzle swizzle) {
  static_assert(kRank >= 1 && kRank <= 5, "tiled tensor maps have rank 1..5");
  CUtensorMap map;
  cuuint64_t d[kRank], s[kRank > 1 ? kRank - 1 : 1];
  cuuint32_t b[kRank], e[kRank];
  for (int i = 0; i < kRank; ++i) {
    d[i] = dims[i];
    b[i] = box[i];
    e[i] = 1; // element strides: take every element
  }
  for (int i = 0; i + 1 < kRank; ++i)
    s[i] = strides_bytes[i];
  CUresult rc = cuTensorMapEncodeTiled(
      &map, dtype, kRank, const_cast<void *>(base), d, s, b, e,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(rc, &msg);
    throw std::runtime_error(
        std::string("cuTensorMapEncodeTiled failed: ") +
        (msg ? msg : "unknown error") +
        " (check: base 16-byte aligned, strides multiple of 16 bytes, "
        "box <= 256, box[0] bytes multiple of 16 and <= swizzle span)");
  }
  return map;
}

// The common case: a row-major [rows, cols] matrix with a [box_rows,
// box_cols] box. Coordinates in device calls are (x = col, y = row).
inline CUtensorMap make_tensor_map_2d(const void *base,
                                      CUtensorMapDataType dtype, uint64_t rows,
                                      uint64_t cols, uint32_t box_rows,
                                      uint32_t box_cols,
                                      CUtensorMapSwizzle swizzle) {
  const uint64_t dims[2] = {cols, rows};
  const uint64_t strides[1] = {cols * tma_elem_bytes(dtype)};
  const uint32_t box[2] = {box_cols, box_rows};
  return make_tensor_map<2>(base, dtype, dims, strides, box, swizzle);
}

// A row-major [d2, d1, d0] tensor with a [box2, box1, box0] box.
// Coordinates in device calls are (x = i0, y = i1, z = i2).
inline CUtensorMap make_tensor_map_3d(const void *base,
                                      CUtensorMapDataType dtype, uint64_t d2,
                                      uint64_t d1, uint64_t d0, uint32_t box2,
                                      uint32_t box1, uint32_t box0,
                                      CUtensorMapSwizzle swizzle) {
  const uint64_t e = tma_elem_bytes(dtype);
  const uint64_t dims[3] = {d0, d1, d2};
  const uint64_t strides[2] = {d0 * e, d1 * d0 * e};
  const uint32_t box[3] = {box0, box1, box2};
  return make_tensor_map<3>(base, dtype, dims, strides, box, swizzle);
}

// ===========================================================================
// Device: bulk tensor copies.
// Pass the map as `const __grid_constant__ CUtensorMap` kernel parameter and
// take its address (&map): the instruction reads the 128-byte map from the
// kernel-parameter space.
// ===========================================================================

// prefetch.tensormap [map];
// PTX: "Data Movement and Conversion Instructions: prefetch, prefetchu".
// Brings the map into the descriptor cache ahead of the first copy.
__device__ __forceinline__ void prefetch_tensor_map(const CUtensorMap *map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
//     [dst], [map, {x, y}], [bar];
// PTX: "Data Movement and Conversion Instructions: cp.async.bulk.tensor".
// Copies the box whose first element is (x, y) into shared `dst` (1024-byte
// aligned for 128B swizzle, else 128-byte aligned), then performs
// complete_tx(box bytes) on `bar`. Issued by ONE thread. The destination
// of a CTA's own shared memory is a valid .shared::cluster address.
__device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *map,
                                            uint64_t *bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(shared_addr(dst)),
      "l"(map), "r"(x), "r"(y), "r"(shared_addr(bar))
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(void *dst, const CUtensorMap *map,
                                            uint64_t *bar, int x, int y,
                                            int z) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
                   shared_addr(dst)),
               "l"(map), "r"(x), "r"(y), "r"(z), "r"(shared_addr(bar))
               : "memory");
}

// cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [map, {x, y}], [src];
// PTX: "Data Movement and Conversion Instructions: cp.async.bulk.tensor".
// Copies the box in shared `src` to global at box origin (x, y); elements
// outside the tensor are skipped. Issued by ONE thread, which tracks it with
// bulk_commit_group / bulk_wait_group*. Writers of `src` must have executed
// fence_proxy_async() and synchronized with the issuing thread first.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap *map,
                                             const void *src, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], "
      "[%3];" ::"l"(map),
      "r"(x), "r"(y), "r"(shared_addr(src))
      : "memory");
}

__device__ __forceinline__ void
tma_store_3d(const CUtensorMap *map, const void *src, int x, int y, int z) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
               "[%0, {%1, %2, %3}], [%4];" ::"l"(map),
               "r"(x), "r"(y), "r"(z), "r"(shared_addr(src))
               : "memory");
}

// cp.async.bulk.commit_group;
// PTX: "cp.async.bulk.commit_group". Closes the group of this thread's
// not-yet-committed bulk stores.
__device__ __forceinline__ void bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// cp.async.bulk.wait_group.read N;
// PTX: "cp.async.bulk.wait_group". Waits until at most N of this thread's
// bulk groups still have to READ their shared-memory source: older groups'
// source buffers may be overwritten. (The global writes may still be in
// flight.)
template <int N> __device__ __forceinline__ void bulk_wait_group_read() {
  static_assert(N >= 0, "N is a non-negative immediate");
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// cp.async.bulk.wait_group N;
// Waits until at most N groups are pending entirely (global writes done).
template <int N> __device__ __forceinline__ void bulk_wait_group() {
  static_assert(N >= 0, "N is a non-negative immediate");
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

} // namespace lab
