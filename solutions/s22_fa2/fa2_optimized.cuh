#pragma once
// Native CUDA/PTX sliced-Q attention. No CUTLASS/CuTe types or kernel calls.
// Algorithm and ownership follow FlashAttention-2; see README for provenance.
#include "cuda_api.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <type_traits>

namespace s22_native {
constexpr unsigned FULL = 0xffffffffu;

// A 16-byte vector is unchanged by the XOR; consecutive rows select different
// shared-memory bank groups. Both copies and ldmatrix use this same mapping.
template<int D> __device__ __forceinline__ int swizzle(int row, int col) {
    return row * D + (col ^ ((row & 7) * 8));
}
__device__ __forceinline__ unsigned smem(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(smem(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void commit_copy() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait_copy() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void load_a(unsigned (&a)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem(p)));
}
__device__ __forceinline__ void load_b(unsigned (&b)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(smem(p)));
}
__device__ __forceinline__ void load_bt(unsigned (&b)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(smem(p)));
}
template<class T> __device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}
template<class T> __device__ __forceinline__ unsigned pack(float x, float y) {
    if constexpr (std::is_same_v<T, __half>) {
        __half2 pair = __floats2half2_rn(x, y);
        return reinterpret_cast<const unsigned&>(pair);
    } else {
        __nv_bfloat162 pair = __floats2bfloat162_rn(x, y);
        return reinterpret_cast<const unsigned&>(pair);
    }
}
__device__ __forceinline__ float quad_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
    return fmaxf(x, __shfl_xor_sync(FULL, x, 1));
}
__device__ __forceinline__ float quad_sum(float x) {
    x += __shfl_xor_sync(FULL, x, 2);
    return x + __shfl_xor_sync(FULL, x, 1);
}

template<class T, int D, int Rows, bool Masked>
__device__ __forceinline__ void copy_rows(T* dst, const T* src, int first, int n, int stride) {
    #pragma unroll
    for (int vec = threadIdx.x; vec < Rows * D / 8; vec += 128) {
        int r = vec / (D / 8), c = vec % (D / 8) * 8;
        bool valid = !Masked || first + r < n;
        // Even a zero-byte copy receives a pointer inside the input allocation.
        const T* p = src + static_cast<int64_t>(valid ? first + r : 0) * stride + c;
        copy16(dst + swizzle<D>(r, c), p, valid);
    }
}

template<class T, int D, int BN, bool Masked>
__global__ __launch_bounds__(128) void attention(const T* q, const T* k, const T* v,
    T* o, float* lse, int n, int heads, float scale, bool causal) {
    constexpr int MG = D == 128 ? 1 : 2;
    constexpr int BM = 64 * MG;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int qr = blockIdx.x * BM;
    const int batch = blockIdx.y / heads, head = blockIdx.y % heads;
    const int stride = heads * D;
    const int64_t base = (static_cast<int64_t>(batch) * n * heads + head) * D;
    q += base; k += base; v += base; o += base;
    extern __shared__ __align__(128) unsigned char raw[];
    T* sq = reinterpret_cast<T*>(raw);
    T* sk = sq + BM * D;
    T* sv = sk + BN * D;
    // Each warp owns rows warp*16+[0,16) and 64+warp*16+[0,16).
    // Each lane owns rows lane/4 and lane/4+8 within a 16-row tile;
    // c[0:2] are two columns of the first row, c[2:4] of the second.
    float out[MG][D / 8][4] = {};
    float maximum[MG][2];
    #pragma unroll
    for (int m = 0; m < MG; ++m) maximum[m][0] = maximum[m][1] = -INFINITY;
    float total[MG][2] = {};
    const float scale2 = scale * 1.4426950408889634f;
    const int end = causal ? min(n, qr + BM) : n;
    const int last = (end + BN - 1) / BN - 1;
    copy_rows<T, D, BM, Masked>(sq, q, qr, n, stride);
    copy_rows<T, D, BN, Masked>(sk, k, last * BN, n, stride);
    commit_copy();

    for (int block = last; block >= 0; --block) {
        float score[MG][BN / 8][4] = {};
        wait_copy();
        __syncthreads();
        // V transfer and QK use different buffers and execute concurrently.
        copy_rows<T, D, BN, Masked>(sv, v, block * BN, n, stride);
        commit_copy();
        #pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
            unsigned a[MG][4];
            #pragma unroll
            for (int m = 0; m < MG; ++m)
                load_a(a[m], sq + swizzle<D>(m * 64 + warp * 16 + lane % 16, kk + lane / 16 * 8));
            #pragma unroll
            for (int col = 0; col < BN / 8; ++col) {
                unsigned b[2];
                load_b(b, sk + swizzle<D>(col * 8 + lane % 8, kk + (lane / 8 % 2) * 8));
                #pragma unroll
                for (int m = 0; m < MG; ++m) mma<T>(score[m][col], a[m], b);
            }
        }
        wait_copy();
        __syncthreads(); // K consumed, V ready. Refill K while computing softmax/PV.
        if (block > 0) {
            copy_rows<T, D, BN, Masked>(sk, k, (block - 1) * BN, n, stride);
            commit_copy();
        }
        #pragma unroll
        for (int m = 0; m < MG; ++m) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = qr + m * 64 + warp * 16 + lane / 4 + r * 8;
                float next = maximum[m][r];
                #pragma unroll
                for (int col = 0; col < BN / 8; ++col) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int key = block * BN + col * 8 + lane % 4 * 2 + j;
                        if constexpr (Masked)
                            if (row >= n || key >= n || (causal && key > row)) score[m][col][2 * r + j] = -INFINITY;
                        next = fmaxf(next, score[m][col][2 * r + j]);
                    }
                }
                next = quad_max(next);
                const float alpha = maximum[m][r] == -INFINITY ? 0.f : exp2f((maximum[m][r] - next) * scale2);
                maximum[m][r] = next;
                total[m][r] *= alpha;
                #pragma unroll
                for (int col = 0; col < D / 8; ++col) {
                    out[m][col][r * 2] *= alpha;
                    out[m][col][r * 2 + 1] *= alpha;
                }
                #pragma unroll
                for (int col = 0; col < BN / 8; ++col) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float p = next == -INFINITY ? 0.f : exp2f((score[m][col][r * 2 + j] - next) * scale2);
                        score[m][col][r * 2 + j] = p;
                        total[m][r] += p; // Quad-local denominator, reduced once at exit.
                    }
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            unsigned a[MG][4];
            #pragma unroll
            for (int m = 0; m < MG; ++m) {
                // Two adjacent 16x8 score fragments become one 16x16 MMA A.
                // No shared-memory round trip and no lane exchange are required.
                a[m][0] = pack<T>(score[m][kk * 2][0], score[m][kk * 2][1]);
                a[m][1] = pack<T>(score[m][kk * 2][2], score[m][kk * 2][3]);
                a[m][2] = pack<T>(score[m][kk * 2 + 1][0], score[m][kk * 2 + 1][1]);
                a[m][3] = pack<T>(score[m][kk * 2 + 1][2], score[m][kk * 2 + 1][3]);
            }
            #pragma unroll
            for (int col = 0; col < D / 8; ++col) {
                unsigned b[2];
                load_bt(b, sv + swizzle<D>(kk * 16 + lane % 16, col * 8));
                #pragma unroll
                for (int m = 0; m < MG; ++m) mma<T>(out[m][col], a[m], b);
            }
        }
        // The next iteration's CTA barrier completes all V readers before V is reused.
    }
    // Reuse Q storage to turn fragment stores into coalesced 16-byte output stores.
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < MG; ++m) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int row = m * 64 + warp * 16 + lane / 4 + r * 8;
            float sum = quad_sum(total[m][r]);
            float inv = sum > 0.f ? 1.f / sum : 0.f;
            if (lane % 4 == 0 && qr + row < n)
                lse[static_cast<int64_t>(blockIdx.y) * n + qr + row] = maximum[m][r] * scale + logf(sum);
            #pragma unroll
            for (int col = 0; col < D / 8; ++col)
                *reinterpret_cast<unsigned*>(sq + swizzle<D>(row, col * 8 + lane % 4 * 2)) =
                    pack<T>(out[m][col][r * 2] * inv, out[m][col][r * 2 + 1] * inv);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int vec = threadIdx.x; vec < BM * D / 8; vec += 128) {
        int row = vec / (D / 8), col = vec % (D / 8) * 8;
        if (qr + row < n)
            *reinterpret_cast<uint4*>(o + static_cast<int64_t>(qr + row) * stride + col) =
                *reinterpret_cast<uint4*>(sq + swizzle<D>(row, col));
    }
}

template<class T, int D, int BN, bool Masked>
void launch(const TensorView& q, const TensorView& k, const TensorView& v,
            const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    constexpr int BM = D == 128 ? 64 : 128;
    constexpr int bytes = (BM + 2 * BN) * D * sizeof(T);
    auto err = cudaFuncSetAttribute(attention<T, D, BN, Masked>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    const dim3 grid((q.shape[1] + BM - 1) / BM, q.shape[0] * q.shape[2]);
    attention<T, D, BN, Masked><<<grid, 128, bytes, stream>>>(
        static_cast<const T*>(q.data), static_cast<const T*>(k.data), static_cast<const T*>(v.data),
        static_cast<T*>(o.data), static_cast<float*>(lse.data), q.shape[1], q.shape[2], scale, causal);
}
template<class T, int D, int BN>
void launch_shape(const TensorView& q, const TensorView& k, const TensorView& v,
                  const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    if (!causal && q.shape[1] % (D == 128 ? 64 : 128) == 0) launch<T, D, BN, false>(q,k,v,o,lse,scale,causal,stream);
    else launch<T, D, BN, true>(q,k,v,o,lse,scale,causal,stream);
}
template<class T>
void dispatch(const TensorView& q, const TensorView& k, const TensorView& v,
              const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    if (q.shape[3] == 64) launch_shape<T,64,64>(q,k,v,o,lse,scale,causal,stream);
    else launch_shape<T,128,64>(q,k,v,o,lse,scale,causal,stream);
}
} // namespace s22_native
