#include "cuda_api.h"
#include <cmath>

namespace {
// SIMT tiled attention: 16 queries share a 32-key tile. Each half warp owns
// one query row and retains its output numerator across all key tiles.
// This deliberately exposes the online state before introducing Tensor Cores.
template<int D>
__global__ void attention(const float* q, const float* k, const float* v,
                          float* o, float* lse, int n, int heads,
                          float scale, bool causal) {
    // A warp consists of two half-warps owning adjacent query rows. Sixteen
    // padding floats put their Q broadcasts (and contiguous initial writes)
    // in disjoint bank halves instead of different words of the same bank.
    __shared__ float qs[16][D + 16];
    // QK lanes vary in key while the reduction column is fixed: transpose K
    // so ks[col][key] is contiguous. The extra column keeps global-coalesced
    // tile stores bank-conflict-free as their lanes vary in col instead.
    __shared__ float ks[D][33], vs[32][D], weights[16][32];
    int lane = threadIdx.x, query_local = threadIdx.y;
    int tid = query_local * 16 + lane;
    int query = blockIdx.x * 16 + query_local;
    int batch = blockIdx.y / heads, head = blockIdx.y % heads;
    int64_t base = ((int64_t)batch * n * heads + head) * D;
    #pragma unroll
    for (int part = 0; part < D / 16; ++part) {
        int col = part * 16 + lane;
        qs[query_local][col] = query < n ? q[base + (int64_t)query * heads * D + col] : 0;
    }
    float numerator[D / 16] = {};
    float maximum = -INFINITY, denominator = 0;
    int limit = causal ? min(n, (int)(blockIdx.x + 1) * 16) : n;
    for (int start = 0; start < limit; start += 32) {
        for (int index = tid; index < 32 * D; index += 256) {
            int row = index / D, col = index % D;
            int64_t offset = base + (int64_t)(start + row) * heads * D + col;
            ks[col][row] = start + row < n ? k[offset] : 0;
            vs[row][col] = start + row < n ? v[offset] : 0;
        }
        __syncthreads();
        float score[2];
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            int key = lane + part * 16;
            float dot = 0, error = 0;
            #pragma unroll
            for (int col = 0; col < D; ++col) {
                float a = qs[query_local][col], b = ks[col][key];
                float product = __fmul_rn(a, b), next = __fadd_rn(dot, product);
                float residual = fabsf(dot) >= fabsf(product) ? __fadd_rn(__fsub_rn(dot, next), product) : __fadd_rn(__fsub_rn(product, next), dot);
                error += residual + fmaf(a, b, -product);
                dot = next;
            }
            score[part] = start + key < n && (!causal || start + key <= query) ? (dot + error) * scale : -INFINITY;
        }
        float tile_max = fmaxf(score[0], score[1]);
        for (int offset = 8; offset; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset, 16));
        float next_max = fmaxf(maximum, tile_max);
        float correction = expf(maximum - next_max);
        float p0 = expf(score[0] - next_max), p1 = expf(score[1] - next_max);
        weights[query_local][lane] = p0;
        weights[query_local][lane + 16] = p1;
        float tile_sum = p0 + p1;
        for (int offset = 8; offset; offset >>= 1)
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, offset, 16);
        denominator = denominator * correction + tile_sum;
        maximum = next_max;
        __syncthreads();
        #pragma unroll
        for (int part = 0; part < D / 16; ++part) {
            float update = 0;
            #pragma unroll
            for (int key = 0; key < 32; ++key)
                update = fmaf(weights[query_local][key], vs[key][lane + 16 * part], update);
            numerator[part] = numerator[part] * correction + update;
        }
        __syncthreads();
    }
    if (query < n) {
        #pragma unroll
        for (int part = 0; part < D / 16; ++part)
            o[base + (int64_t)query * heads * D + lane + 16 * part] = numerator[part] / denominator;
        if (lane == 0) lse[(int64_t)blockIdx.y * n + query] = maximum + logf(denominator);
    }
}
template<int D>
void launch(const TensorMap& inputs, const TensorMap& outputs, const Options& options, cudaStream_t stream) {
    const auto& q = inputs.at("q");
    attention<D><<<dim3((q.shape[1] + 15) / 16, q.shape[0] * q.shape[2]), dim3(16, 16), 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(inputs.at("k").data),
        static_cast<const float*>(inputs.at("v").data), static_cast<float*>(outputs.at("o").data),
        static_cast<float*>(outputs.at("lse").data), q.shape[1], q.shape[2], options.at("scale"), options.at("causal") != 0);
}
}
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap&, const Options& options, cudaStream_t stream) {
    const auto& q = inputs.at("q");
    if (q.dtype != DType::F32) throw std::runtime_error("s12 solution requires FP32");
    switch (q.shape[3]) {
        case 32: launch<32>(inputs, outputs, options, stream); break;
        case 64: launch<64>(inputs, outputs, options, stream); break;
        case 128: launch<128>(inputs, outputs, options, stream); break;
        default: throw std::runtime_error("s12 solution requires D=32/64/128");
    }
}
