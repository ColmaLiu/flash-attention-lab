#pragma once
// A complete SM100 forward kernel. No tensor library or DSL is used.
// The PTX instruction and descriptor layouts are specified by NVIDIA's PTX ISA.
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace sm100_attention {
constexpr int BM = 128, BN = 64;

// Artifact diagnostic: preserve the ordinary specialization unchanged. Convert
// each raw operand to FP64 before subtraction/multiplication so a subnormal
// difference is not rounded/flushed before a large scale can amplify it.
__device__ __forceinline__ float extreme_exp_input(float value, float reference,
                                                   float scale) {
  return static_cast<float>((double(value) - double(reference)) *
                            double(scale) * 1.4426950408889634);
}
inline bool needs_extreme_scale(float scale) {
  const float scale2 = scale * 1.4426950408889634f;
  return !std::isnormal(scale) || !std::isnormal(scale2) || scale2 > 1.f;
}

__device__ __forceinline__ unsigned shared(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init_bar(uint64_t *p) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(shared(p))
               : "memory");
}
__device__ __forceinline__ void wait_bar(uint64_t *p, int phase) {
  asm volatile(
      "{ .reg .pred done; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 "
      "done, [%0], %1; @!done bra wait_loop; }" ::"r"(shared(p)),
      "r"(phase)
      : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void expect_bytes(uint64_t *p, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(shared(p)),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void copy_tile(const CUtensorMap *map, void *dst,
                                          uint64_t *bar, int col, int head,
                                          int row, int batch) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2,%3,%4,%5}], [%6];" ::"r"(shared(dst)),
      "l"(map), "r"(col), "r"(head), "r"(row), "r"(batch), "r"(shared(bar))
      : "memory");
}
// Each 64-element row is 128 bytes. Eight rows form a swizzle atom.
// Panels of 64 columns are stored separately: [D/64, Rows, 64].
__device__ __forceinline__ int offset(int rows, int row, int col) {
  return (col / 64) * rows * 64 + row * 64 + ((col % 64) ^ ((row & 7) * 8));
}
__device__ __forceinline__ uint64_t descriptor(const void *p,
                                               int leading = 16) {
  return (uint64_t(shared(p)) >> 4) | (uint64_t(leading / 16) << 16) |
         (uint64_t(1024 / 16) << 32) | (1ull << 46) | (2ull << 61);
}
template <class T, int N, bool BTransposed = false>
__device__ __forceinline__ void mma(unsigned output, uint64_t a, uint64_t b,
                                    bool accumulate) {
  constexpr unsigned type = std::is_same_v<T, __half> ? 0 : 1;
  constexpr unsigned instruction = (1u << 4) | (type << 7) | (type << 10) |
                                   (unsigned(BTransposed) << 16) |
                                   ((N / 8) << 17) | ((BM / 16) << 24);
  asm volatile("{ .reg .pred accumulate; setp.ne.b32 accumulate, %4, 0; "
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
               "{%5,%5,%5,%5}, accumulate; }" ::"r"(output),
               "l"(a), "l"(b), "r"(instruction), "r"(int(accumulate)), "r"(0)
               : "memory");
}
__device__ __forceinline__ void commit_mma(uint64_t *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(shared(bar))
               : "memory");
}
__device__ __forceinline__ void load16(float *x, unsigned address) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]),
                 "=f"(x[5]), "=f"(x[6]), "=f"(x[7]), "=f"(x[8]), "=f"(x[9]),
                 "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]),
                 "=f"(x[14]), "=f"(x[15])
               : "r"(address)
               : "memory");
}
__device__ __forceinline__ void wait_loads() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <class T> __device__ __forceinline__ unsigned pack(float a, float b) {
  if constexpr (std::is_same_v<T, __half>) {
    __half2 x = __floats2half2_rn(a, b);
    return reinterpret_cast<const unsigned &>(x);
  } else {
    __nv_bfloat162 x = __floats2bfloat162_rn(a, b);
    return reinterpret_cast<const unsigned &>(x);
  }
}
__device__ __forceinline__ float exp2_sfu(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Range reduction to [-0.5,0.5], then degree-6 exp polynomial. This path
// trades SFU pressure for FP32 FMAs; it is measured, never assumed faster.
__device__ __forceinline__ float exp2_polynomial(float x) {
  if (x < -126.f)
    return 0.f;
  int exponent = __float2int_rn(x);
  float r = x - float(exponent);
  float p = 0.0001540353039f;
  p = fmaf(p, r, 0.001333355815f);
  p = fmaf(p, r, 0.009618129108f);
  p = fmaf(p, r, 0.05550410866f);
  p = fmaf(p, r, 0.2402265070f);
  p = fmaf(p, r, 0.6931471806f);
  p = fmaf(p, r, 1.f);
  return p * __int_as_float((exponent + 127) << 23);
}

template <class T, int D>
__device__ __forceinline__ void load_q(const CUtensorMap *qmap, T *q,
                                       uint64_t *bar, int head, int row,
                                       int batch) {
  expect_bytes(bar, BM * D * 2);
#pragma unroll
  for (int d = 0; d < D; d += 64)
    copy_tile(qmap, q + (d / 64) * BM * 64, bar, d, head, row, batch);
}
template <class T, int D>
__device__ __forceinline__ void
load_kv(const CUtensorMap *kmap, const CUtensorMap *vmap, T *k, T *v,
        uint64_t *bar, int head, int row, int batch) {
  expect_bytes(bar, 2 * BN * D * 2);
#pragma unroll
  for (int d = 0; d < D; d += 64) {
    copy_tile(kmap, k + (d / 64) * BN * 64, bar, d, head, row, batch);
    copy_tile(vmap, v + (d / 64) * BN * 64, bar, d, head, row, batch);
  }
}
template <class T, int D>
__device__ __forceinline__ void qk(T *q, T *k, unsigned out, uint64_t *bar) {
#pragma unroll
  for (int d = 0; d < D; d += 16) {
    auto qa = descriptor(q + (d / 64) * BM * 64 + d % 64);
    auto kb = descriptor(k + (d / 64) * BN * 64 + d % 64);
    mma<T, BN>(out, qa, kb, d != 0);
  }
  commit_mma(bar);
}
template <class T, int D>
__device__ __forceinline__ void pv(T *p, T *v, unsigned out, uint64_t *bar) {
#pragma unroll
  for (int k = 0; k < BN; k += 16) {
    mma<T, D, true>(out, descriptor(p + k), descriptor(v + k * 64, BN * 128),
                    k != 0);
  }
  commit_mma(bar);
}

// Four consumer warps own one query row per lane. QK and PV accumulate in
// TMEM, while running max/denominator/output live in FP32 registers. Each PTX
// load is warp-collective, with the warp's first row encoded in address[31:16].
template <class T, int D, bool Overlap, bool Numerics, bool Persistent,
          bool Polynomial = false, bool ExtremeScale = false>
__global__
__launch_bounds__(128) void attention(const __grid_constant__ CUtensorMap qmap,
                                      const __grid_constant__ CUtensorMap kmap,
                                      const __grid_constant__ CUtensorMap vmap,
                                      T *out, float *lse, int n, int heads,
                                      int total, float scale, bool causal) {
  constexpr int Buffers = Overlap ? 2 : 1;
  extern __shared__ __align__(1024) unsigned char raw[];
  T *q = reinterpret_cast<T *>(raw);
  T *k = q + BM * D;
  T *v = k + Buffers * BN * D;
  T *p = v + Buffers * BN * D;
  __shared__ uint64_t qbar, kvbar[Buffers], scorebar[Buffers], pvbar;
  __shared__ unsigned tmem;
  if (threadIdx.x < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
            "r"(shared(&tmem))
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const unsigned row_address = tmem + ((warp * 32) << 16);
  const int qtiles = (n + BM - 1) / BM;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int query_tile = task % qtiles;
    // Longest causal rows first. Every CTA uses the same static stride;
    // neither a global counter nor inter-CTA synchronization is needed.
    int bh = task / qtiles;
    if constexpr (Persistent)
      if (causal) {
        // Sort by query length across all heads, not within each head.
        // Contiguous tasks have comparable work before the cyclic stride.
        int batch_heads = total / qtiles;
        query_tile = qtiles - 1 - task / batch_heads;
        bh = task % batch_heads;
      }
    int head = bh % heads, batch = bh / heads;
    int query = query_tile * BM, row = query + threadIdx.x;
    int tiles = ((causal ? min(n, query + BM) : n) + BN - 1) / BN;
    if (threadIdx.x == 0) {
      init_bar(&qbar);
      init_bar(&pvbar);
#pragma unroll
      for (int i = 0; i < Buffers; ++i) {
        init_bar(&kvbar[i]);
        init_bar(&scorebar[i]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      load_q<T, D>(&qmap, q, &qbar, head, query, batch);
      load_kv<T, D>(&kmap, &vmap, k, v, &kvbar[0], head, 0, batch);
    }
    wait_bar(&qbar, 0);
    wait_bar(&kvbar[0], 0);
    if (threadIdx.x == 0)
      qk<T, D>(q, k, tmem, &scorebar[0]);
    float maximum = -CUDART_INF_F, denominator = 0.f;
    float output[D] = {};
    const float scale2 = scale * 1.4426950408889634f;
    for (int tile = 0; tile < tiles; ++tile) {
      int slot = tile % Buffers;
      // Prefetch while the current QK is still in flight. Waiting for
      // score completion first would serialize the TMA and QK engines.
      if constexpr (Overlap) {
        if (threadIdx.x == 0 && tile + 1 < tiles) {
          int next = (tile + 1) % Buffers;
          load_kv<T, D>(&kmap, &vmap, k + next * BN * D, v + next * BN * D,
                        &kvbar[next], head, (tile + 1) * BN, batch);
        }
      }
      wait_bar(&scorebar[slot], (tile / Buffers) & 1);
      float scores[BN];
#pragma unroll
      for (int j = 0; j < BN; j += 16)
        load16(scores + j, row_address + slot * BN + j);
      wait_loads();
      if constexpr (Overlap) {
        // Launch next QK before CUDA softmax/correction for this tile.
        // Separate score and K/V buffers keep both generations alive.
        if (tile + 1 < tiles) {
          int next = (tile + 1) % Buffers;
          wait_bar(&kvbar[next], ((tile + 1) / Buffers) & 1);
          if (threadIdx.x == 0)
            qk<T, D>(q, k + next * BN * D, tmem + next * BN, &scorebar[next]);
        }
      }
      float maxima[4] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F,
                         -CUDART_INF_F};
#pragma unroll
      for (int j = 0; j < BN; ++j) {
        int key = tile * BN + j;
        if constexpr (ExtremeScale)
          scores[j] =
              key < n && (!causal || key <= row) ? scores[j] : -CUDART_INF_F;
        else
          scores[j] = key < n && (!causal || key <= row) ? scores[j] * scale2
                                                         : -CUDART_INF_F;
        maxima[j % 4] = fmaxf(maxima[j % 4], scores[j]);
      }
      float next_max = fmaxf(maximum, fmaxf(fmaxf(maxima[0], maxima[1]),
                                            fmaxf(maxima[2], maxima[3])));
      if constexpr (Numerics) {
        // Delayed rescale: keep the old reference when all probabilities
        // remain <= 16. This is not skipping contributions or masking.
        if constexpr (ExtremeScale) {
          if ((double(next_max) - double(maximum)) * double(scale) *
                  1.4426950408889634 <
              4.)
            next_max = maximum;
        } else if (next_max - maximum < 4.f)
          next_max = maximum;
      }
      float correction;
      if constexpr (ExtremeScale)
        correction = exp2_sfu(extreme_exp_input(maximum, next_max, scale));
      else
        correction = exp2_sfu(maximum - next_max);
      maximum = next_max;
      float sums[4] = {};
#pragma unroll
      for (int j = 0; j < BN; ++j) {
        float x;
        if constexpr (ExtremeScale)
          x = extreme_exp_input(scores[j], maximum, scale);
        else
          x = scores[j] - maximum;
        if constexpr (Polynomial)
          scores[j] = (j & 1) ? exp2_polynomial(x) : exp2_sfu(x);
        else
          scores[j] = exp2_sfu(x);
        sums[j % 4] += scores[j];
      }
      denominator =
          denominator * correction + (sums[0] + sums[1]) + (sums[2] + sums[3]);
      // With delayed rescaling most tiles keep the same maximum.
      // Skip the entire D-wide correction, not just its exponential.
      if (!Numerics || correction != 1.f) {
#pragma unroll
        for (int d = 0; d < D; ++d)
          output[d] *= correction;
      }
#pragma unroll
      for (int j = 0; j < BN; j += 2)
        *reinterpret_cast<unsigned *>(p + offset(BM, threadIdx.x, j)) =
            pack<T>(scores[j], scores[j + 1]);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (threadIdx.x == 0)
        pv<T, D>(p, v + slot * BN * D, tmem + 128, &pvbar);
      wait_bar(&pvbar, tile & 1);
#pragma unroll
      for (int d = 0; d < D; d += 16) {
        float values[16];
        load16(values, row_address + 128 + d);
        wait_loads();
#pragma unroll
        for (int j = 0; j < 16; ++j)
          output[d + j] += values[j];
      }
      // PTX 8.7 9.7.16.5.4.4: completion (wait::ld) and cross-thread
      // ordering are separate. Publish every warp's completed TMEM reads
      // before the issuer can overwrite this allocation in another MMA.
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if constexpr (!Overlap) {
        if (tile + 1 < tiles) {
          if (threadIdx.x == 0)
            load_kv<T, D>(&kmap, &vmap, k, v, &kvbar[0], head, (tile + 1) * BN,
                          batch);
          wait_bar(&kvbar[0], (tile + 1) & 1);
          if (threadIdx.x == 0)
            qk<T, D>(q, k, tmem, &scorebar[0]);
        }
      }
    }
// Reuse Q's dead shared allocation for a coalesced output epilogue.
#pragma unroll
    for (int d = 0; d < D; d += 2)
      *reinterpret_cast<unsigned *>(q + offset(BM, threadIdx.x, d)) =
          pack<T>(output[d] / denominator, output[d + 1] / denominator);
    if (row < n) {
      if constexpr (ExtremeScale)
        lse[static_cast<int64_t>(bh) * n + row] = static_cast<float>(
            double(maximum) * double(scale) + log(double(denominator)));
      else
        lse[static_cast<int64_t>(bh) * n + row] =
            maximum * 0.6931471805599453f + logf(denominator);
    }
    __syncthreads();
#pragma unroll
    for (int vec = threadIdx.x; vec < BM * D / 8; vec += 128) {
      int r = vec / (D / 8), d = (vec % (D / 8)) * 8;
      if (query + r < n) {
        auto dst =
            out +
            ((static_cast<int64_t>(batch) * n + query + r) * heads + head) * D +
            d;
        *reinterpret_cast<int4 *>(dst) =
            *reinterpret_cast<int4 *>(q + offset(BM, r, d));
      }
    }
    // All threads used Q through generic shared accesses in the epilogue.
    // Order those reads/writes before the next task's async TMA overwrite;
    // a fence executed only by the TMA issuer would not cover its peers.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&qbar))
                   : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&pvbar))
                   : "memory");
#pragma unroll
      for (int i = 0; i < Buffers; ++i) {
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&kvbar[i]))
            : "memory");
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&scorebar[i]))
            : "memory");
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 32)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem)
        : "memory");
}

struct Prepared {
  CUtensorMap q, k, v;
  void *output;
  float *lse;
  int batch, n, heads, dim, sms;
  bool half, causal;
  bool polynomial = false, delayed_rescale = true, persistent = true;
  float scale;
  cudaStream_t stream;
};
inline void check(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
inline void encode(CUtensorMap *map, uintptr_t data, int b, int n, int h, int d,
                   int rows, bool half) {
  cuuint64_t dims[] = {static_cast<cuuint64_t>(d), static_cast<cuuint64_t>(h),
                       static_cast<cuuint64_t>(n), static_cast<cuuint64_t>(b)};
  cuuint64_t strides[] = {static_cast<cuuint64_t>(d * 2),
                          static_cast<cuuint64_t>(h * d * 2),
                          static_cast<cuuint64_t>(n) * h * d * 2};
  cuuint32_t box[] = {64, 1, static_cast<cuuint32_t>(rows), 1},
             steps[] = {1, 1, 1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      map,
      half ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      4, reinterpret_cast<void *>(data), dims, strides, box, steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(rc, &message);
    throw std::runtime_error(message ? message : "attention TMA encode failed");
  }
}
template <class T, int D, bool Overlap, bool Numerics, bool Persistent,
          bool Polynomial, bool ExtremeScale>
inline void launch_variant(const Prepared &s) {
  constexpr int Buffers = Overlap ? 2 : 1;
  constexpr int SharedBytes = (BM * D + 2 * Buffers * BN * D + BM * BN) * 2;
  auto kernel =
      attention<T, D, Overlap, Numerics, Persistent, Polynomial, ExtremeScale>;
  check(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
  int total = s.batch * s.heads * ((s.n + BM - 1) / BM);
  int grid = Persistent ? std::min(total, s.sms * 2) : total;
  kernel<<<grid, 128, SharedBytes, s.stream>>>(
      s.q, s.k, s.v, static_cast<T *>(s.output), s.lse, s.n, s.heads, total,
      s.scale, s.causal);
  check(cudaGetLastError());
}
template <class T, int D, bool Overlap, bool Numerics, bool Persistent,
          bool Polynomial = false>
inline void launch(const Prepared &s) {
  if (needs_extreme_scale(s.scale))
    launch_variant<T, D, Overlap, Numerics, Persistent, Polynomial, true>(s);
  else
    launch_variant<T, D, Overlap, Numerics, Persistent, Polynomial, false>(s);
}
// Independent compile-time ablations keep the measured launch free of
// device-side mode branches. The default uses SFU plus delayed rescaling.
template <class T, int D, bool Persistent>
inline void launch_numerics(const Prepared &s) {
  if (s.polynomial) {
    if (s.delayed_rescale)
      launch<T, D, true, true, Persistent, true>(s);
    else
      launch<T, D, true, false, Persistent, true>(s);
  } else {
    if (s.delayed_rescale)
      launch<T, D, true, true, Persistent, false>(s);
    else
      launch<T, D, true, false, Persistent, false>(s);
  }
}
} // namespace sm100_attention
