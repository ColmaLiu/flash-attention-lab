// Warp-level Tensor Core primitives: ldmatrix, mma.sync m16n8k16 and the
// fragment layouts they use. Written by learners in 20_mma_gemm; provided
// from 21_pipeline_gemm on.
//
// Every function wraps one PTX instruction (or a fixed formula) and contains
// no algorithm logic. PTX section names refer to the PTX ISA (version 8.7),
// https://docs.nvidia.com/cuda/parallel-thread-execution/
//
// Notation for one warp: lane = threadIdx.x % 32, group = lane / 4 (0..7),
// quad = lane % 4 (0..3). A 32-bit register holding two 16-bit values keeps
// the element with the smaller column (or k) index in its low half.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lab {

// Shared-memory address of a generic pointer, as ldmatrix/cp.async expect.
// (PTX: "Generic Addressing"; CUDA: __cvta_generic_to_shared.)
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// ldmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans].shared.b16
// PTX: "Warp-level matrix load instruction: ldmatrix".
//
// Loads 1, 2 or 4 matrices of 8x8 16-bit elements from shared memory.
// Each matrix row is 16 contiguous bytes (16-byte aligned); rows may be
// anywhere. Lanes 8*i .. 8*i+7 supply the addresses of rows 0..7 of matrix i
// (for .x1 only lanes 0-7 are used, for .x2 lanes 0-15; the other lanes'
// addresses are ignored but must still be passed).
// Result: register r[i] of lane l holds row l/4, elements 2*(l%4) and
// 2*(l%4)+1 of matrix i -- exactly the layout of one mma operand register.
// With .trans each matrix is transposed while loading: r[i] of lane l holds
// elements (row 2*(l%4), col l/4) and (row 2*(l%4)+1, col l/4).
// All 32 lanes must execute it together (.sync.aligned).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x1(uint32_t &r, const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
               : "=r"(r)
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x1_trans(uint32_t &r, const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(r)
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const void *p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_u32(p)));
}

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32
// PTX: "Warp Level Matrix Multiply-Accumulate Instructions: mma",
//      "Matrix Fragments for mma.m16n8k16 with floating point type".
//
// d[16x8] += a[16x16] * b[16x8], FP32 accumulate, computed by the whole warp.
// .row/.col: A is read as row-major (k fastest), B as column-major (k
// fastest), i.e. b is the 8x16 tile B[n][k] of a K-major B.
// Each lane holds (see frag_*_coord below for the exact coordinates):
//   a: 4 registers = 8 elements,  b: 2 registers = 4 elements,
//   d: 4 floats.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}
__device__ __forceinline__ void mma_m16n8k16_f16(float (&d)[4],
                                                 const uint32_t (&a)[4],
                                                 const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

// ---------------------------------------------------------------------------
// Fragment coordinates of mma.m16n8k16 (PTX: "Matrix Fragments for
// mma.m16n8k16 with floating point type"). group = lane/4, quad = lane%4.
//
//   A (16x16), register i = 0..3 holds A[row][col], A[row][col+1]:
//     i:    0        1        2        3
//     row:  group    group+8  group    group+8
//     col:  2quad    2quad    2quad+8  2quad+8
//   B (16x8 as k x n), register i = 0..1 holds B[k][n], B[k+1][n]:
//     k:    2quad, 2quad+8       n: group
//   C/D (16x8), float i = 0..3 is D[row][col]:
//     row:  group, group, group+8, group+8
//     col:  2quad, 2quad+1, 2quad, 2quad+1
// ---------------------------------------------------------------------------
struct FragCoord {
  int row, col;
};

// A register i (0..3): coordinates of its low element; the high element is
// at col + 1.
__device__ __forceinline__ FragCoord frag_a_coord(int lane, int i) {
  return {lane / 4 + (i & 1) * 8, (lane % 4) * 2 + (i >> 1) * 8};
}
// B register i (0..1): {row = k, col = n} of its low element; the high
// element is at k + 1.
__device__ __forceinline__ FragCoord frag_b_coord(int lane, int i) {
  return {(lane % 4) * 2 + i * 8, lane / 4};
}
// C/D float i (0..3).
__device__ __forceinline__ FragCoord frag_c_coord(int lane, int i) {
  return {lane / 4 + (i >> 1) * 8, (lane % 4) * 2 + (i & 1)};
}

// ---------------------------------------------------------------------------
// XOR swizzle of 16-byte chunks in a shared-memory tile whose rows are
// kChunks * 16 bytes (kChunks = 4: 64-byte rows, 8: 128-byte rows).
// Returns the physical chunk index of logical chunk `chunk` in row `row`.
// Every 128 bytes of shared memory cover all 32 banks once, i.e. 8 chunk
// slots. The XOR makes the same logical chunk of 8 consecutive rows land in
// 8 different slots, so ldmatrix (8 rows x 16 bytes per matrix) and 16-byte
// stores are conflict-free. Values inside a chunk are not moved.
// (Same idea as CUTLASS Swizzle<3,4,3> for 128-byte rows; not a PTX
// instruction.)
// ---------------------------------------------------------------------------
template <int kChunks>
__device__ __forceinline__ int swizzle_chunk(int row, int chunk) {
  static_assert(kChunks == 4 || kChunks == 8, "rows of 64 or 128 bytes");
  if constexpr (kChunks == 8)
    return chunk ^ (row & 7);
  else
    return chunk ^ ((row >> 1) & 3);
}

} // namespace lab
