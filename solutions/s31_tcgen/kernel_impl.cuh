// Native CUDA/PTX SM100 GEMM. No CUTLASS/CuTe headers or library kernel calls.
// Instruction encodings follow NVIDIA PTX ISA tcgen05 matrix descriptors.
#include "cuda_api.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <stdexcept>

namespace {
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int Stages = LAB_MULTISTAGE ? 3 : 1;
__device__ __forceinline__ unsigned smem(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void barrier_init(uint64_t *b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem(b))
               : "memory");
}
__device__ __forceinline__ void wait(uint64_t *b, int phase) {
  asm volatile(
      "{ .reg .pred p; again: mbarrier.try_wait.parity.shared::cta.b64 p, "
      "[%0], %1; @!p bra again; }" ::"r"(smem(b)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void commit(uint64_t *b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(smem(b))
               : "memory");
}
__device__ __forceinline__ uint64_t descriptor(const void *p, int leading,
                                               int stride) {
  return uint64_t(smem(p) >> 4) | (uint64_t(leading >> 4) << 16) |
         (uint64_t(stride >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void tma(const CUtensorMap *map, void *dst,
                                    uint64_t *bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(smem(dst)),
               "l"(map), "r"(x), "r"(y), "r"(smem(bar))
               : "memory");
}
__device__ __forceinline__ void load_tile(const CUtensorMap *a,
                                          const CUtensorMap *b,
                                          __nv_bfloat16 *tile, uint64_t *full,
                                          int m, int n, int k) {
  constexpr int Bytes = (BM * BK + BK * BN) * 2;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem(full)),
               "r"(Bytes)
               : "memory");
  tma(a, tile, full, k, m);
  // Each N64 panel is contiguous [K,64] in SMEM. This is the MN-major
  // tcgen05 layout; a plain [K,128] TMA box has a different row stride.
  tma(b, tile + BM * BK, full, n, k);
  tma(b, tile + BM * BK + BK * 64, full, n + 64, k);
}
__device__ __forceinline__ void multiply(__nv_bfloat16 *tile, unsigned accum,
                                         bool add) {
  // FP32 accumulator, BF16 inputs, K-major A, MN-major B, M=N=128.
  constexpr unsigned instr = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) |
                             (16u << 17) | (8u << 24);
#pragma unroll
  for (int k = 0; k < BK; k += 16) {
    uint64_t da = descriptor(tile + k, 16, 1024);
    uint64_t db = descriptor(tile + BM * BK + k * 64, BK * 128, 1024);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                 "{%5,%5,%5,%5}, p; }" ::"r"(accum),
                 "l"(da), "l"(db), "r"(instr), "r"(int(add || k > 0)), "r"(0)
                 : "memory");
  }
}
__device__ __forceinline__ void read8(unsigned addr, float (&v)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
        "=f"(v[6]), "=f"(v[7])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ void gemm_native(const __grid_constant__ CUtensorMap a,
                            const __grid_constant__ CUtensorMap b, float *c,
                            int M, int N, int K) {
  // Operand storage is reused by the padded epilogue only after all MMA reads
  // complete. Padding lets row-owned TMEM fragments be staged without 32-way
  // bank conflicts; the final global writes are contiguous vectors.
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *tiles = reinterpret_cast<__nv_bfloat16 *>(storage);
  __shared__ __align__(8) uint64_t full[Stages], empty[Stages];
  __shared__ unsigned accum;
  int tid = threadIdx.x, warp = tid / 32;
  if (tid == 0) {
    for (int s = 0; s < Stages; ++s) {
      barrier_init(&full[s]);
      barrier_init(&empty[s]);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" ::
            "r"(smem(&accum))
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  __syncthreads();
  int m = blockIdx.y * BM, n = blockIdx.x * BN, steps = (K + BK - 1) / BK;
#if LAB_MULTISTAGE
  // A dedicated producer warp may be three tiles ahead of the MMA issuer.
  // full[s] is completed by TMA; empty[s] by tcgen05.commit after its final
  // read of that slot. Independent parity counters prevent early buffer reuse.
  if (tid == 128) {
    for (int i = 0; i < steps; ++i) {
      int s = i % Stages, round = i / Stages;
      if (round)
        wait(&empty[s], (round - 1) & 1);
      load_tile(&a, &b, tiles + s * (BM * BK + BK * BN), &full[s], m, n,
                i * BK);
    }
  }
  if (tid == 0) {
    for (int i = 0; i < steps; ++i) {
      int s = i % Stages;
      wait(&full[s], (i / Stages) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      multiply(tiles + s * (BM * BK + BK * BN), accum, i != 0);
      commit(&empty[s]);
    }
    wait(&empty[(steps - 1) % Stages], ((steps - 1) / Stages) & 1);
  }
#else
  // First establish the full TMA -> MMA -> completion protocol with one slot.
  if (tid == 0) {
    for (int i = 0; i < steps; ++i) {
      load_tile(&a, &b, tiles, &full[0], m, n, i * BK);
      wait(&full[0], i & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      multiply(tiles, accum, i != 0);
      commit(&empty[0]);
      wait(&empty[0], i & 1);
    }
  }
#endif
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  float *epilogue = reinterpret_cast<float *>(storage);
  if (warp < 4) {
#pragma unroll
    for (int col = 0; col < BN; col += 8) {
      float v[8];
      read8(accum + ((warp * 32) << 16) + col, v);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        epilogue[tid * (BN + 4) + col + j] = v[j];
    }
  }
  // Publish completed TMEM reads across warps before eventual deallocation.
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  for (int vec = tid; vec < BM * BN / 4; vec += blockDim.x) {
    int row = vec / (BN / 4), col = (vec % (BN / 4)) * 4;
    if (m + row < M && n + col < N)
      *reinterpret_cast<float4 *>(c + int64_t(m + row) * N + n + col) =
          *reinterpret_cast<float4 *>(epilogue + row * (BN + 4) + col);
  }
  __syncthreads();
  if (warp == 0)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(accum)
        : "memory");
}

struct Prepared {
  CUtensorMap a, b;
  float *c;
  int m, n, k;
  cudaStream_t stream;
};
void encode(CUtensorMap *map, uintptr_t pointer, int rows, int cols,
            int tile_rows) {
  cuuint64_t dims[] = {static_cast<cuuint64_t>(cols),
                       static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[] = {static_cast<cuuint64_t>(cols * 2)};
  cuuint32_t box[] = {64, static_cast<cuuint32_t>(tile_rows)}, steps[] = {1, 1};
  auto rc = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      reinterpret_cast<void *>(pointer), dims, strides, box, steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(rc, &msg);
    throw std::runtime_error(msg ? msg : "TMA descriptor encoding failed");
  }
}
constexpr int SharedBytes = Stages * (BM * BK + BK * BN) * 2 > BM *(BN + 4) * 4
                                ? Stages *(BM *BK + BK * BN) * 2
                                : BM *(BN + 4) * 4;
} // namespace
void *prepare_gemm(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k,
                   uintptr_t stream) {
  if (m % 64 || n % 8 || k % 16)
    throw std::runtime_error("SM100 GEMM requires M%64=N%8=K%16=0");
  auto *s = new Prepared{};
  try {
    encode(&s->a, a, m, k, BM);
    encode(&s->b, b, k, n, BK);
  } catch (...) {
    delete s;
    throw;
  }
  s->c = reinterpret_cast<float *>(c);
  s->m = m;
  s->n = n;
  s->k = k;
  s->stream = reinterpret_cast<cudaStream_t>(stream);
  auto rc = cudaFuncSetAttribute(
      gemm_native, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
  if (rc != cudaSuccess) {
    delete s;
    throw std::runtime_error(cudaGetErrorString(rc));
  }
  return s;
}
void run_gemm(void *p) {
  auto *s = static_cast<Prepared *>(p);
  gemm_native<<<dim3((s->n + BN - 1) / BN, (s->m + BM - 1) / BM),
                LAB_MULTISTAGE ? 160 : 128, SharedBytes, s->stream>>>(
      s->a, s->b, s->c, s->m, s->n, s->k);
  auto rc = cudaGetLastError();
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
void delete_gemm(void *p) {
  delete static_cast<Prepared *>(p);
}
size_t gemm_workspace_bytes(void *) {
  return 0;
}
