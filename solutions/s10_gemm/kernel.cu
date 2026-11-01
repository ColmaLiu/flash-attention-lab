#include "cuda_api.h"
#include <cmath>

namespace {
// Each CTA computes 32x32 output values. Its 64 threads retain 4x4 register
// subtiles; a 16-wide K tile provides reuse across both output dimensions.
__global__ void gemm(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[32][16], bs[16][32];
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 8 + tx;
    int row0 = blockIdx.y * 32, col0 = blockIdx.x * 32;
    float accum[4][4] = {};
    for (int start = 0; start < k; start += 16) {
        for (int index = tid; index < 32 * 16; index += 64) {
            int row = index / 16, inner = index % 16;
            as[row][inner] = row0 + row < m && start + inner < k
                ? a[(int64_t)(row0 + row) * k + start + inner] : 0.0f;
        }
        for (int index = tid; index < 16 * 32; index += 64) {
            int inner = index / 32, col = index % 32;
            bs[inner][col] = start + inner < k && col0 + col < n
                ? b[(int64_t)(start + inner) * n + col0 + col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { av[i] = as[ty + i * 8][inner]; bv[i] = bs[inner][tx + i * 8]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) accum[i][j] = fmaf(av[i], bv[j], accum[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = row0 + ty + 8 * i, col = col0 + tx + 8 * j;
            if (row < m && col < n) c[(int64_t)row * n + col] = accum[i][j];
        }
}
}
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap&, const Options&, cudaStream_t stream) {
    const auto& a = inputs.at("a"); const auto& b = inputs.at("b");
    if (a.dtype != DType::F32 || b.dtype != DType::F32) throw std::runtime_error("s10 solution requires FP32");
    int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    gemm<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 8), 0, stream>>>(
        static_cast<const float*>(a.data), static_cast<const float*>(b.data), static_cast<float*>(outputs.at("c").data), m, n, k);
}
