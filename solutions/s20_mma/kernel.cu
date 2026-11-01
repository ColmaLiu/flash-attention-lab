// Complete reference: one warp owns a 16x8 output tile, explicit PTX fragments.
// Performance anchor: cuBLAS; this stage exposes the instruction mapping first.
#include "cuda_api.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
template<bool BF16>
__device__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    if constexpr (BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}
__device__ unsigned pack(unsigned short lo, unsigned short hi) {
    return static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
}

template<bool BF16>
__global__ void gemm(const unsigned short* a, const unsigned short* b, float* c, int n, int k) {
    const int lane = threadIdx.x;
    const int group = lane / 4, pair = (lane % 4) * 2;
    const int row = blockIdx.y * 16, col = blockIdx.x * 8;
    float accum[4] = {0, 0, 0, 0};
    for (int base = 0; base < k; base += 16) {
        unsigned ar[4], br[2];
        ar[0] = pack(a[(row + group) * k + base + pair], a[(row + group) * k + base + pair + 1]);
        ar[1] = pack(a[(row + group + 8) * k + base + pair], a[(row + group + 8) * k + base + pair + 1]);
        ar[2] = pack(a[(row + group) * k + base + pair + 8], a[(row + group) * k + base + pair + 9]);
        ar[3] = pack(a[(row + group + 8) * k + base + pair + 8], a[(row + group + 8) * k + base + pair + 9]);
        br[0] = pack(b[(base + pair) * n + col + group], b[(base + pair + 1) * n + col + group]);
        br[1] = pack(b[(base + pair + 8) * n + col + group], b[(base + pair + 9) * n + col + group]);
        mma<BF16>(accum, ar, br);
    }
    c[(row + group) * n + col + pair] = accum[0];
    c[(row + group) * n + col + pair + 1] = accum[1];
    c[(row + group + 8) * n + col + pair] = accum[2];
    c[(row + group + 8) * n + col + pair + 1] = accum[3];
}
}  // namespace

void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap&, const Options&, cudaStream_t stream) {
    const auto& a = inputs.at("a");
    const auto& b = inputs.at("b");
    const int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    if (m % 16 || n % 16 || k % 16) throw std::invalid_argument("M,N,K must be multiples of 16");
    const dim3 grid(n / 8, m / 16);
    const auto* ap = static_cast<const unsigned short*>(a.data);
    const auto* bp = static_cast<const unsigned short*>(b.data);
    auto* cp = static_cast<float*>(outputs.at("c").data);
    if (a.dtype == DType::BF16) gemm<true><<<grid, 32, 0, stream>>>(ap, bp, cp, n, k);
    else if (a.dtype == DType::F16) gemm<false><<<grid, 32, 0, stream>>>(ap, bp, cp, n, k);
    else throw std::invalid_argument("expected FP16/BF16 operands");
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
