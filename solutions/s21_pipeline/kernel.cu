// Complete reference: CTA reuse plus cp.async double buffering.
// WMMA expresses the fragment operations already exposed explicitly in s20.
#include "cuda_api.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace {
constexpr int BM = 64, BN = 64, BK = 32, AS = 40, BS = 72;
__device__ void copy16(void* dst, const void* src, bool valid) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
template<class T>
__device__ void stage(T* sa, T* sb, const T* a, const T* b,
                      int row, int col, int base, int m, int n, int k) {
    for (int segment = threadIdx.x; segment < BM * BK / 8; segment += 128) {
        const int r = segment / (BK / 8), x = segment % (BK / 8) * 8;
        const bool valid = row + r < m && base + x < k;
        copy16(sa + r * AS + x, valid ? a + (row + r) * k + base + x : a, valid);
    }
    for (int segment = threadIdx.x; segment < BK * BN / 8; segment += 128) {
        const int r = segment / (BN / 8), x = segment % (BN / 8) * 8;
        const bool valid = base + r < k && col + x < n;
        copy16(sb + r * BS + x, valid ? b + (base + r) * n + col + x : b, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<class T>
__global__ void gemm(const T* a, const T* b, float* c, int m, int n, int k) {
    using namespace nvcuda;
    __shared__ __align__(32) T sa[2][BM * AS];
    __shared__ __align__(32) T sb[2][BK * BS];
    const int warp = threadIdx.x / 32;
    const int wr = warp / 2 * 32, wc = warp % 2 * 32;
    const int row = blockIdx.y * BM, col = blockIdx.x * BN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(accum[i][j], 0.0f);
    stage(sa[0], sb[0], a, b, row, col, 0, m, n, k);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int base = 0, tile = 0; base < k; base += BK, ++tile) {
        const int current = tile & 1;
        // The next producer writes a different buffer while MMA consumes this one.
        if (base + BK < k) stage(sa[current ^ 1], sb[current ^ 1], a, b, row, col, base + BK, m, n, k);
        #pragma unroll
        for (int inner = 0; inner < BK; inner += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], sa[current] + (wr + i * 16) * AS + inner, AS);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], sb[current] + inner * BS + wc + j * 16, BS);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(accum[i][j], af[i], bf[j], accum[i][j]);
        }
        // All old consumers finish, and every thread waits for its own copies;
        // the following CTA barrier publishes the complete next buffer.
        __syncthreads();
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            if (row + wr + i * 16 < m && col + wc + j * 16 < n)
                wmma::store_matrix_sync(c + (row + wr + i * 16) * n + col + wc + j * 16,
                                        accum[i][j], n, wmma::mem_row_major);
}
}  // namespace

void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap&, const Options&, cudaStream_t stream) {
    const auto& a = inputs.at("a");
    const auto& b = inputs.at("b");
    const int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    if (m % 16 || n % 16 || k % 16) throw std::invalid_argument("M,N,K must be multiples of 16");
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    auto* cp = static_cast<float*>(outputs.at("c").data);
    if (a.dtype == DType::BF16) gemm<<<grid, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data), cp, m, n, k);
    else if (a.dtype == DType::F16) gemm<<<grid, 128, 0, stream>>>(static_cast<const __half*>(a.data), static_cast<const __half*>(b.data), cp, m, n, k);
    else throw std::invalid_argument("expected FP16/BF16 operands");
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
