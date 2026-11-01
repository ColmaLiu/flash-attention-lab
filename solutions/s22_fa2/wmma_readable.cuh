#pragma once
// Complete CUDA/WMMA teaching reference: sliced-Q, online softmax, and BNHD tails.
// Adapted from the previous local WMMA exercise, without a library fallback.
// Shared-memory score/PV staging makes this readable but is not production FA2.
// Compare against the independent cuDNN anchor; read docs for the staged traffic cost.
#include "cuda_api.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>

namespace s22_reference {
constexpr int block_m = 64;
constexpr int block_n = 64;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(kWarpMask, x, offset);
    return __shfl_sync(kWarpMask, x, 0);
}

__device__ float warp_max(float x) {
    for (int offset = 16; offset > 0; offset /= 2)
        x = fmaxf(x, __shfl_down_sync(kWarpMask, x, offset));
    return __shfl_sync(kWarpMask, x, 0);
}

template<typename T> __device__ T from_float(float x);
template<> __device__ __half from_float<__half>(float x) {
    return __float2half_rn(x);
}
template<> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

template<typename T, int D>
struct SharedStorage {
    __align__(32) T q[block_m * D];
    // K and V are consumed in separate phases and reuse the same storage.
    __align__(32) T kv[block_n * D];
    __align__(32) T p[block_m * block_n];
    // Each warp owns 16 rows. QK and PV staging never overlap in time.
    union {
        __align__(32) float scores[block_m * block_n];
        __align__(32) float pv[block_m * D];
    } scratch;
};

template<typename T, int D>
__global__ void fa2(const T* q, const T* k, const T* v, T* o, float* lse,
                    int n, int heads, float scale, bool causal) {
    using namespace nvcuda;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wr = warp * 16;
    const int64_t head_row = static_cast<int64_t>(blockIdx.y) * n;
    const int64_t query_row = head_row + blockIdx.x * block_m;
    const int batch = blockIdx.y / heads, head = blockIdx.y % heads;
    extern __shared__ __align__(32) unsigned char storage[];
    auto& sm = *reinterpret_cast<SharedStorage<T, D>*>(storage);

    // Sliced-Q: each warp owns all key/output columns of its 16 query rows.
    // No cross-warp softmax or output reduction is required.
    float row_max[16], row_sum[16];
    float out[16][D / 32];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        row_max[r] = -CUDART_INF_F;
        row_sum[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < D / 32; ++c) out[r][c] = 0.0f;
    }
    // Linear 16-byte segments work for both D=64 and D=128.
    for (int segment = tid; segment < block_m * D / 8; segment += 128) {
        const int row = segment / (D / 8), col = segment % (D / 8) * 8;
        const int qi = blockIdx.x * block_m + row;
        reinterpret_cast<uint4*>(sm.q)[segment] = qi < n ?
            *reinterpret_cast<const uint4*>(q + ((static_cast<int64_t>(batch) * n + qi) * heads + head) * D + col) :
            make_uint4(0, 0, 0, 0);
    }
    __syncthreads();

    const int loop_range = causal ?
        min((n + block_n - 1) / block_n, static_cast<int>(blockIdx.x + 1)) : (n + block_n - 1) / block_n;
    for (int kb = 0; kb < loop_range; ++kb) {
        for (int segment = tid; segment < block_n * D / 8; segment += 128) {
            const int row = segment / (D / 8), col = segment % (D / 8) * 8;
            const int ki = kb * block_n + row;
            reinterpret_cast<uint4*>(sm.kv)[segment] = ki < n ?
                *reinterpret_cast<const uint4*>(k + ((static_cast<int64_t>(batch) * n + ki) * heads + head) * D + col) :
                make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // QK^T: K's row-major [64,D] storage is a column-major [D,64]
        // view. Accumulate in FP32 and materialize only this score tile.
        for (int col = 0; col < block_n; col += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int d = 0; d < D; d += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bf;
                wmma::load_matrix_sync(af, sm.q + wr * D + d, D);
                wmma::load_matrix_sync(bf, sm.kv + col * D + d, D);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(sm.scratch.scores + wr * block_n + col,
                                    acc, block_n, wmma::mem_row_major);
        }
        // Finish all reads of K before its storage is reused for V.
        __syncthreads();

        for (int r = 0; r < 16; ++r) {
            const int row = wr + r;
            float x0 = sm.scratch.scores[row * block_n + lane] * scale;
            float x1 = sm.scratch.scores[row * block_n + lane + 32] * scale;
            const int qi = blockIdx.x * block_m + row;
            if (kb * block_n + lane >= n || (causal && kb * block_n + lane > qi)) x0 = -CUDART_INF_F;
            if (kb * block_n + lane + 32 >= n || (causal && kb * block_n + lane + 32 > qi)) x1 = -CUDART_INF_F;
            const float next_max = fmaxf(row_max[r], warp_max(fmaxf(x0, x1)));
            const float alpha = expf(row_max[r] - next_max);
            const float p0 = expf(x0 - next_max);
            const float p1 = expf(x1 - next_max);
            row_sum[r] = alpha * row_sum[r] + warp_sum(p0 + p1);
            row_max[r] = next_max;
            #pragma unroll
            for (int c = 0; c < D / 32; ++c) out[r][c] *= alpha;
            // Round P only at the MMA input boundary. The denominator above
            // uses FP32 probabilities, not these rounded values.
            sm.p[row * block_n + lane] = from_float<T>(p0);
            sm.p[row * block_n + lane + 32] = from_float<T>(p1);
        }
        for (int segment = tid; segment < block_n * D / 8; segment += 128) {
            const int row = segment / (D / 8), col = segment % (D / 8) * 8;
            const int ki = kb * block_n + row;
            reinterpret_cast<uint4*>(sm.kv)[segment] = ki < n ?
                *reinterpret_cast<const uint4*>(v + ((static_cast<int64_t>(batch) * n + ki) * heads + head) * D + col) :
                make_uint4(0, 0, 0, 0);
        }
        // Publish P/V and finish score reads before the union becomes PV.
        __syncthreads();

        for (int col = 0; col < D; col += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < block_n; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sm.p + wr * block_n + kk, block_n);
                wmma::load_matrix_sync(bf, sm.kv + kk * D + col, D);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(sm.scratch.pv + wr * D + col,
                                    acc, D, wmma::mem_row_major);
        }
        __syncwarp(kWarpMask);
        #pragma unroll
        for (int r = 0; r < 16; ++r)
            #pragma unroll
            for (int c = 0; c < D / 32; ++c)
                out[r][c] += sm.scratch.pv[(wr + r) * D + lane + c * 32];
        // Finish consuming V and scratch before either is overwritten.
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int qi = blockIdx.x * block_m + wr + r;
        if (qi < n) {
            const int64_t output_row = (static_cast<int64_t>(batch) * n + qi) * heads + head;
            #pragma unroll
            for (int c = 0; c < D / 32; ++c)
                o[output_row * D + lane + c * 32] = from_float<T>(out[r][c] / row_sum[r]);
            if (lane == 0) lse[query_row + wr + r] = row_max[r] + logf(row_sum[r]);
        }
    }
}

template<typename T, int D>
void launch_specialized(const T* q, const T* k, const T* v, T* o, float* lse,
                        int n, int heads, float scale, bool causal, dim3 grid, dim3 block,
                        cudaStream_t stream) {
    constexpr int bytes = sizeof(SharedStorage<T, D>);
    const auto error = cudaFuncSetAttribute(fa2<T, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("s22_fa2 shared memory: ") +
                                 cudaGetErrorString(error));
    fa2<T, D><<<grid, block, bytes, stream>>>(q, k, v, o, lse, n, heads, scale, causal);
}

template<typename T>
void launch_typed(const TensorView& q, const TensorView& k, const TensorView& v,
                  const TensorView& o, const TensorView& lse,
                  float scale, bool causal, cudaStream_t stream) {
    const T* qp = static_cast<const T*>(q.data);
    const T* kp = static_cast<const T*>(k.data);
    const T* vp = static_cast<const T*>(v.data);
    T* op = static_cast<T*>(o.data);
    float* lp = static_cast<float*>(lse.data);

    const int64_t b = q.shape[0], n = q.shape[1], h = q.shape[2], d = q.shape[3];
    const dim3 grid(static_cast<unsigned>((n + block_m - 1) / block_m), static_cast<unsigned>(b * h));
    const dim3 block(128);
    switch (d) {
        case 64: launch_specialized<T, 64>(qp, kp, vp, op, lp, n, h, scale, causal, grid, block, stream); break;
        case 128: launch_specialized<T, 128>(qp, kp, vp, op, lp, n, h, scale, causal, grid, block, stream); break;
    }
}

// All pointers are device pointers; launch on the supplied stream.
// ENTRY CONTRACT (case-sensitive keys; contiguous tensors on one GPU):
// inputs.at("q"): [B,N,H,D], DType::BF16 / DType::F16; input Q; same shape/dtype/device, read-only.
// inputs.at("k"): [B,N,H,D], DType::BF16 / DType::F16; input K; same shape/dtype/device, read-only.
// inputs.at("v"): [B,N,H,D], DType::BF16 / DType::F16; input V; same shape/dtype/device, read-only.
// outputs.at("o"): [B,N,H,D], DType::BF16 / DType::F16; attention output O; input dtype.
// outputs.at("lse"): [B,H,N], DType::F32; natural-log logsumexp of scaled, masked scores.
// options.at("scale"): positive double (default 1/sqrt(D)); options.at("causal"): 0.0 or 1.0.
// workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
// Outputs are preallocated, not initialized; input tensors must not be modified.
// Full contract and supported shapes: docs/tasks/s22_fa2.md.
void launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream) {
    const auto& q = inputs.at("q");
    const auto& k = inputs.at("k");
    const auto& v = inputs.at("v");
    if (q.shape.size() != 4 || (q.dtype != DType::BF16 && q.dtype != DType::F16) ||
        k.shape != q.shape || v.shape != q.shape ||
        k.dtype != q.dtype || v.dtype != q.dtype)
        throw std::runtime_error("s22_fa2 requires matching BF16/FP16 [B,N,H,D] inputs");
    const int64_t b = q.shape[0], n = q.shape[1], h = q.shape[2], d = q.shape[3];
    if (!(d == 64 || d == 128) || b <= 0 || h <= 0 || b > 65535 / h ||
        n <= 0 || n > 2147483584)
        throw std::runtime_error("s22_fa2 supports only its curriculum shapes");
    const double scale = options.at("scale");
    const bool causal = options.at("causal") != 0.0;
    if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(static_cast<float>(scale)) ||
        static_cast<float>(scale) <= 0.0f)
        throw std::runtime_error("s22_fa2 requires a finite positive FP32 scale");
    const auto& o = outputs.at("o");
    const std::vector<int64_t> row_shape{b, h, n};
    if (o.shape != q.shape || o.dtype != q.dtype)
        throw std::runtime_error("s22_fa2: invalid output");
    const auto& lse = outputs.at("lse");
    if (lse.shape != row_shape || lse.dtype != DType::F32)
        throw std::runtime_error("s22_fa2: invalid row state");
    if (q.dtype == DType::F16) {
        launch_typed<__half>(q, k, v, o, lse, static_cast<float>(scale), causal, stream);
    } else if (q.dtype == DType::BF16) {
        launch_typed<__nv_bfloat16>(q, k, v, o, lse, static_cast<float>(scale), causal, stream);
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("s22_fa2 launch: ")+ cudaGetErrorString(error));
}

} // namespace s22_reference
