// Complete CUDA/PTX reference: synchronous CTA tiling, ldmatrix and mma.sync.
#include "cuda_api.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
constexpr int BM = 128, BN = 64, BK = 32;
template <bool BF16>
__device__ void mma(float (&c)[4], const unsigned (&a)[4],
                    const unsigned (&b)[2]) {
  if constexpr (BF16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
  }
}
// Swizzle whole 16-byte vectors without disturbing values within each vector.
template <int Stride> __device__ __forceinline__ int offset(int row, int col) {
  return row * Stride + (col ^ ((row & (Stride / 8 - 1)) * 8));
}
__device__ __forceinline__ unsigned address(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void load_a(unsigned (&a)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address(p)));
}
__device__ __forceinline__ void load_b(unsigned (&b)[2], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address(p)));
}

template <bool BF16>
__global__ __launch_bounds__(128) void gemm(const unsigned short *a,
                                            const unsigned short *b, float *c,
                                            int m, int n, int k) {
  __shared__ __align__(128) unsigned short sa[BM * BK];
  __shared__ __align__(128) unsigned short sb[BK * BN];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.y * BM, col = blockIdx.x * BN;
  const int wr = warp / 2 * 64, wc = warp % 2 * 32;
  // Warp output tile = 4x4 MMA atoms, each 16x8. The accumulator ownership
  // is row=lane/4 + {0,8}, column=2*(lane%4) + {0,1} within each atom.
  float accum[4][4][4] = {};
  for (int base = 0; base < k; base += BK) {
// Synchronous global -> registers -> shared copies. Each vector stays
// inside one row; the contract's multiples of 16 make tails vector-safe.
#pragma unroll
    for (int vec = threadIdx.x; vec < BM * BK / 8; vec += 128) {
      const int r = vec / (BK / 8), x = vec % (BK / 8) * 8;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (row + r < m && base + x < k)
        value = *reinterpret_cast<const uint4 *>(a + (row + r) * k + base + x);
      *reinterpret_cast<uint4 *>(sa + offset<BK>(r, x)) = value;
    }
#pragma unroll
    for (int vec = threadIdx.x; vec < BK * BN / 8; vec += 128) {
      const int r = vec / (BN / 8), x = vec % (BN / 8) * 8;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (base + r < k && col + x < n)
        value = *reinterpret_cast<const uint4 *>(b + (base + r) * n + col + x);
      *reinterpret_cast<uint4 *>(sb + offset<BN>(r, x)) = value;
    }
    __syncthreads(); // Every producer has published this tile.
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      unsigned ar[4][4], br[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        load_a(ar[i],
               sa + offset<BK>(wr + i * 16 + lane % 16, inner + lane / 16 * 8));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        load_b(br[j], sb + offset<BN>(inner + lane % 16, wc + j * 8));
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma<BF16>(accum[i][j], ar[i], br[j]);
    }
    __syncthreads(); // All consumers finish before the next overwrite.
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int rr = row + wr + i * 16 + lane / 4;
      int cc = col + wc + j * 8 + lane % 4 * 2;
      if (rr < m && cc < n)
        *reinterpret_cast<float2 *>(c + rr * n + cc) =
            make_float2(accum[i][j][0], accum[i][j][1]);
      if (rr + 8 < m && cc < n)
        *reinterpret_cast<float2 *>(c + (rr + 8) * n + cc) =
            make_float2(accum[i][j][2], accum[i][j][3]);
    }
  }
}
} // namespace

void student_launch(const TensorMap &inputs, const TensorMap &outputs,
                    const TensorMap &, const Options &, cudaStream_t stream) {
  const auto &a = inputs.at("a");
  const auto &b = inputs.at("b");
  const int m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (m % 16 || n % 16 || k % 16)
    throw std::invalid_argument("M,N,K must be multiples of 16");
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  const auto *ap = static_cast<const unsigned short *>(a.data);
  const auto *bp = static_cast<const unsigned short *>(b.data);
  auto *cp = static_cast<float *>(outputs.at("c").data);
  if (a.dtype == DType::BF16)
    gemm<true><<<grid, 128, 0, stream>>>(ap, bp, cp, m, n, k);
  else if (a.dtype == DType::F16)
    gemm<false><<<grid, 128, 0, stream>>>(ap, bp, cp, m, n, k);
  else
    throw std::invalid_argument("expected FP16/BF16 operands");
  const auto error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}
