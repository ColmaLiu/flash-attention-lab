#include "cuda_api.h"
#include <cmath>

namespace {
__global__ void qk(const float* q, const float* k, float* scores,
                   int n, int heads, int d, float scale, bool causal) {
    __shared__ float qs[16][16], ks[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
    int batch = blockIdx.z / heads, head = blockIdx.z % heads;
    int64_t base = ((int64_t)batch * n * heads + head) * d;
    float sum = 0, correction = 0;
    for (int start = 0; start < d; start += 16) {
        qs[ty][tx] = row < n && start + tx < d ? q[base + (int64_t)row * heads * d + start + tx] : 0;
        int key = blockIdx.x * 16 + ty;
        ks[ty][tx] = key < n && start + tx < d ? k[base + (int64_t)key * heads * d + start + tx] : 0;
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            float a = qs[ty][inner], b = ks[tx][inner];
            float product = __fmul_rn(a, b), next = __fadd_rn(sum, product);
            float residual = fabsf(sum) >= fabsf(product) ? __fadd_rn(__fsub_rn(sum, next), product) : __fadd_rn(__fsub_rn(product, next), sum);
            correction += residual + fmaf(a, b, -product);
            sum = next;
        }
        __syncthreads();
    }
    if (row < n && col < n) scores[((int64_t)blockIdx.z * n + row) * n + col] = causal && col > row ? -INFINITY : (sum + correction) * scale;
}
__device__ float warp_max(float x) {
    for (int off = 16; off; off >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, off));
    return x;
}
__device__ float warp_sum(float x) {
    for (int off = 16; off; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off);
    return x;
}
__global__ void normalize(const float* scores, float* probabilities, float* lse, int n) {
    __shared__ float partial[8];
    int64_t row = blockIdx.x;
    scores += row * n; probabilities += row * n;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float maximum = -INFINITY;
    for (int col = threadIdx.x; col < n; col += 256) maximum = fmaxf(maximum, scores[col]);
    maximum = warp_max(maximum);
    if (lane == 0) partial[warp] = maximum;
    __syncthreads();
    maximum = warp_max(lane < 8 ? partial[lane] : -INFINITY);
    __syncthreads();
    float denominator = 0;
    for (int col = threadIdx.x; col < n; col += 256) denominator += expf(scores[col] - maximum);
    denominator = warp_sum(denominator);
    if (lane == 0) partial[warp] = denominator;
    __syncthreads();
    denominator = warp_sum(lane < 8 ? partial[lane] : 0);
    for (int col = threadIdx.x; col < n; col += 256) probabilities[col] = expf(scores[col] - maximum) / denominator;
    if (threadIdx.x == 0) lse[row] = maximum + logf(denominator);
}
__global__ void pv(const float* probabilities, const float* v, float* o, int n, int heads, int d) {
    __shared__ float ps[16][16], vs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
    int batch = blockIdx.z / heads, head = blockIdx.z % heads;
    int64_t base = ((int64_t)batch * n * heads + head) * d;
    float sum = 0;
    for (int start = 0; start < n; start += 16) {
        ps[ty][tx] = row < n && start + tx < n ? probabilities[((int64_t)blockIdx.z * n + row) * n + start + tx] : 0;
        vs[ty][tx] = start + ty < n && col < d ? v[base + (int64_t)(start + ty) * heads * d + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < 16; ++inner) sum = fmaf(ps[ty][inner], vs[inner][tx], sum);
        __syncthreads();
    }
    if (row < n && col < d) o[base + (int64_t)row * heads * d + col] = sum;
}
}
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options, cudaStream_t stream) {
    const auto& q = inputs.at("q");
    if (q.dtype != DType::F32) throw std::runtime_error("s11 solution requires FP32");
    int batch = q.shape[0], n = q.shape[1], heads = q.shape[2], d = q.shape[3];
    float* scores = static_cast<float*>(workspace.at("scores").data);
    float* probabilities = static_cast<float*>(workspace.at("probabilities").data);
    qk<<<dim3((n + 15) / 16, (n + 15) / 16, batch * heads), dim3(16, 16), 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(inputs.at("k").data), scores,
        n, heads, d, options.at("scale"), options.at("causal") != 0);
    normalize<<<batch * heads * n, 256, 0, stream>>>(scores, probabilities, static_cast<float*>(outputs.at("lse").data), n);
    pv<<<dim3((d + 15) / 16, (n + 15) / 16, batch * heads), dim3(16, 16), 0, stream>>>(
        probabilities, static_cast<const float*>(inputs.at("v").data), static_cast<float*>(outputs.at("o").data), n, heads, d);
}
