// Native CUDA/PTX SM100 GEMM with occupancy-aware N tiling and split K.
// See s31 for the serial teaching baseline. This implementation owns its
// complete device path; no CUTLASS/CuTe or library GEMM is used.
#include "cuda_api.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <stdexcept>

namespace {

// Every lane of the selected warp must enter this election. Keeping the
// election outside the issue loop lets ptxas retain uniform descriptors.
__device__ __forceinline__ bool elected() {
  unsigned result;
  asm volatile(
      "{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
      : "=r"(result));
  return result;
}
__device__ __forceinline__ unsigned smem(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init(uint64_t *b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem(b))
               : "memory");
}
// A blocking wait gives the scheduler a long suspend hint. The no-hint form
// is useful for a one-shot probe, but repeatedly probing consumes issue slots.
__device__ __forceinline__ void wait(uint64_t *b, int phase) {
  asm volatile(
      "{ .reg .pred p; again: mbarrier.try_wait.parity.shared::cta.b64 p, "
      "[%0], %1, 0x989680; @!p bra again; }" ::"r"(smem(b)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void commit(uint64_t *b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(smem(b))
               : "memory");
}
__device__ __forceinline__ uint64_t descriptor(const void *p, int leading,
                                               int width = 64) {
  return uint64_t((smem(p) >> 4) & 0x3fff) |
         (uint64_t((leading >> 4) & 0x3fff) << 16) | (uint64_t(width) << 32) |
         (1ull << 46) | (uint64_t(width == 64 ? 2 : 4) << 61);
}
__device__ __forceinline__ void tma(const CUtensorMap *map, void *dst,
                                    uint64_t *bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(smem(dst)),
               "l"(map), "r"(x), "r"(y), "r"(smem(bar))
               : "memory");
}
__device__ __forceinline__ void tma3(const CUtensorMap *map, void *dst,
                                     uint64_t *bar, int x, int y, int z) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2,%3,%4}], [%5];" ::"r"(smem(dst)),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem(bar))
      : "memory");
}
template <int BM, int BN, int BK>
__device__ __forceinline__ void
load_tile(const CUtensorMap *a, const CUtensorMap *b, __nv_bfloat16 *tile,
          uint64_t *full, int m, int n, int k, bool packed) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem(full)),
               "r"((BM * BK + BK * BN) * 2)
               : "memory");
  if (packed) {
    // The third coordinate selects adjacent 64-element panels. Two TMA
    // instructions now cover the entire A/B stage, without a transpose.
    tma3(a, tile, full, 0, m, k / 64);
    tma3(b, tile + BM * BK, full, 0, k, n / 64);
  } else {
#pragma unroll
    for (int panel = 0; panel < BK / 64; ++panel)
      tma(a, tile + panel * BM * 64, full, k + panel * 64, m);
#pragma unroll
    for (int panel = 0; panel < (BN + 63) / 64; ++panel)
#pragma unroll
      for (int offset = 0; offset < BK; offset += 256)
        tma(b, tile + BM * BK + panel * BK * 64 + offset * (BN < 64 ? BN : 64),
            full, n + panel * 64, k + offset);
  }
}
template <int BM, int BN, int BK, bool FastIssue>
__device__ __forceinline__ void multiply(__nv_bfloat16 *tile, unsigned accum,
                                         bool add) {
  constexpr unsigned instr = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) |
                             ((BN / 8) << 17) | ((BM / 16) << 24);
  constexpr int BWidth = BN < 64 ? BN : 64;
  auto base_a = descriptor(tile, 16),
       base_b = descriptor(tile + BM * BK, BK * BWidth * 2, BWidth);
#pragma unroll
  for (int k = 0; k < BK; k += 16) {
    uint64_t da, db;
    if constexpr (FastIssue) {
      // Only the encoded start address changes. The shared-memory resource
      // limit keeps this increment inside its 14-bit field. Spell the low
      // word update explicitly rather than carrying through descriptor flags.
      da = (base_a & 0xffffffff00000000ull) |
           uint32_t(uint32_t(base_a) + ((k / 64) * BM * 64 + k % 64) / 8);
      db = (base_b & 0xffffffff00000000ull) |
           uint32_t(uint32_t(base_b) + k * BWidth / 8);
    } else {
      da = descriptor(tile + (k / 64) * BM * 64 + k % 64, 16);
      db = descriptor(tile + BM * BK + k * BWidth, BK * BWidth * 2, BWidth);
    }
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                 "{%5,%5,%5,%5}, p; }" ::"r"(accum),
                 "l"(da), "l"(db), "r"(instr), "r"(int(add || k > 0)), "r"(0)
                 : "memory");
  }
}
// The default producer/consumer route retains one descriptor pair for slot 0.
// Each ring slot and K16 step only changes the encoded shared start address.
// Addresses use 16-byte units: eight BF16 elements. Legal shared allocations
// stay below 256 KiB, so these offsets cannot carry out of the 14-bit start
// field. Keep the high word explicit to preserve its layout/swizzle flags.
template <int BM, int BN, int BK>
__device__ __forceinline__ void multiply_desc(uint64_t base_a, uint64_t base_b,
                                              unsigned accum, bool add) {
  constexpr int BWidth = BN < 64 ? BN : 64;
  constexpr unsigned instr = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) |
                             ((BN / 8) << 17) | ((BM / 16) << 24);
#pragma unroll
  for (int k = 0; k < BK; k += 16) {
    uint64_t da =
        (base_a & 0xffffffff00000000ull) |
        uint32_t(uint32_t(base_a) + ((k / 64) * BM * 64 + k % 64) / 8);
    uint64_t db = (base_b & 0xffffffff00000000ull) |
                  uint32_t(uint32_t(base_b) + k * BWidth / 8);
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                 "{%5,%5,%5,%5}, p; }" ::"r"(accum),
                 "l"(da), "l"(db), "r"(instr), "r"(int(add || k > 0)), "r"(0)
                 : "memory");
  }
}
__device__ __forceinline__ void read32(unsigned addr, float (&v)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%"
      "20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
        "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]),
        "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]),
        "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]),
        "=f"(v[21]), "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]),
        "=f"(v[26]), "=f"(v[27]), "=f"(v[28]), "=f"(v[29]), "=f"(v[30]),
        "=f"(v[31])
      : "r"(addr)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// 16x256b gives each lane two adjacent columns in rows lane/4 and
// lane/4+8, repeated every 8 columns. This ownership permits sector-coalesced
// float2 global stores without a shared-memory transpose.
__device__ __forceinline__ void read16x32(unsigned addr, float (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]),
                 "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
                 "=f"(v[14]), "=f"(v[15])
               : "r"(addr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

template <int BM, int BN, int BK, int Stages, bool Single>
__global__ __launch_bounds__(Single ? 128 : 160) void gemm(
    const __grid_constant__ CUtensorMap a,
    const __grid_constant__ CUtensorMap b, float *c, int M, int N, int K,
    int tiles_per_split, bool packed) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *tiles = reinterpret_cast<__nv_bfloat16 *>(storage);
  __shared__ uint64_t full[Stages], empty[Stages];
  __shared__ unsigned accum;
  int tid = threadIdx.x, warp = tid / 32;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < Stages; ++s) {
      init(&full[s]);
      init(&empty[s]);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(smem(&accum)),
        "n"(BN)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  __syncthreads();
  int m = blockIdx.y * BM, n = blockIdx.x * BN;
  int first = blockIdx.z * tiles_per_split;
  int steps = min(tiles_per_split, (K + BK - 1) / BK - first);
  if constexpr (Single) {
    // One issuer owns both asynchronous engines. Preload the ring, then refill
    // a retired slot while the most recently issued MMA is still executing.
    // Stages==1 is the explicit serial control; >=2 retains TMA/MMA overlap.
    if (tid == 0) {
      for (int i = 0; i < min(steps, Stages); ++i)
        load_tile<BM, BN, BK>(&a, &b, tiles + i * (BM * BK + BK * BN), &full[i],
                              m, n, (first + i) * BK, packed);
      for (int i = 0; i < steps; ++i) {
        int slot = i % Stages;
        wait(&full[slot], (i / Stages) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        multiply<BM, BN, BK, (BK >= 256 || Stages >= 4)>(
            tiles + slot * (BM * BK + BK * BN), accum, i != 0);
        commit(&empty[slot]);
        int next = Stages == 1 ? i + 1 : i + Stages - 1;
        if ((Stages == 1 || i > 0) && next < steps) {
          int refill = next % Stages;
          wait(&empty[refill], (next / Stages - 1) & 1);
          load_tile<BM, BN, BK>(&a, &b, tiles + refill * (BM * BK + BK * BN),
                                &full[refill], m, n, (first + next) * BK,
                                packed);
        }
      }
      wait(&empty[(steps - 1) % Stages], ((steps - 1) / Stages) & 1);
    }
  } else {
    if (warp == 4 && ((BK >= 256 || Stages >= 4) ? elected() : tid == 128)) {
      for (int i = 0; i < steps; ++i) {
        int s = i % Stages, round = i / Stages;
        if (round)
          wait(&empty[s], (round - 1) & 1);
        load_tile<BM, BN, BK>(&a, &b, tiles + s * (BM * BK + BK * BN), &full[s],
                              m, n, (first + i) * BK, packed);
      }
    }
    if (warp == 0 && ((BK >= 256 || Stages >= 4) ? elected() : tid == 0)) {
      constexpr int BWidth = BN < 64 ? BN : 64;
      // Reuse the fixed leading/stride/swizzle fields across ring generations.
      // The short-K teaching issue path below still builds each K16 descriptor.
      auto first_a = descriptor(tiles, 16);
      auto first_b = descriptor(tiles + BM * BK, BK * BWidth * 2, BWidth);
      for (int i = 0; i < steps; ++i) {
        int s = i % Stages;
        wait(&full[s], (i / Stages) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if constexpr (BK >= 256 || Stages >= 4) {
          unsigned offset = s * (BM * BK + BK * BN) / 8;
          auto da = (first_a & 0xffffffff00000000ull) |
                    uint32_t(uint32_t(first_a) + offset);
          auto db = (first_b & 0xffffffff00000000ull) |
                    uint32_t(uint32_t(first_b) + offset);
          multiply_desc<BM, BN, BK>(da, db, accum, i != 0);
        } else
          multiply<BM, BN, BK, false>(tiles + s * (BM * BK + BK * BN), accum,
                                      i != 0);
        commit(&empty[s]);
      }
      wait(&empty[(steps - 1) % Stages], ((steps - 1) / Stages) & 1);
    }
  }
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  c += int64_t(blockIdx.z) * M * N;
  if (warp < 4) {
#pragma unroll
    for (int half = 0; half < BM / 64; ++half) {
#pragma unroll
      for (int col = 0; col < BN; col += 32) {
        float v[16];
        read16x32(accum + ((warp * 32 + half * 16) << 16) + col, v);
#pragma unroll
        for (int panel = 0; panel < 4; ++panel) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            int row = warp * (BM / 4) + half * 16 + (tid % 32) / 4 + r * 8;
            int column = col + (tid % 4) * 2 + panel * 8;
            if (m + row < M && n + column < N)
              *reinterpret_cast<float2 *>(c + int64_t(m + row) * N + n +
                                          column) =
                  make_float2(v[panel * 4 + r * 2], v[panel * 4 + r * 2 + 1]);
          }
        }
      }
    }
  }
  // All warps finish TMEM reads before its allocation is released.
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 0)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(accum),
        "n"(BN)
        : "memory");
}
__global__ void reduce_splits(const float *partial, float *c, int count,
                              int splits) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i < count) {
    float4 sum = make_float4(0, 0, 0, 0);
    for (int s = 0; s < splits; ++s) {
      auto v =
          *reinterpret_cast<const float4 *>(partial + int64_t(s) * count + i);
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    *reinterpret_cast<float4 *>(c + i) = sum;
  }
}
struct Prepared {
  CUtensorMap a, b;
  float *c;
  float *partial = nullptr;
  int m, n, k, bm, bn, bk, stages, splits, tiles;
  bool single, packed;
  cudaStream_t stream;
};
void check(cudaError_t rc) {
  if (rc != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(rc));
}
void encode(CUtensorMap *map, uintptr_t p, int rows, int cols, int tile_rows,
            int width = 64) {
  cuuint64_t dims[] = {static_cast<cuuint64_t>(cols),
                       static_cast<cuuint64_t>(rows)},
             strides[] = {static_cast<cuuint64_t>(cols) * 2};
  cuuint32_t box[] = {static_cast<cuuint32_t>(width),
                      static_cast<cuuint32_t>(tile_rows)},
             steps[] = {1, 1};
  auto rc = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void *>(p),
      dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
      width == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(rc, &msg);
    throw std::runtime_error(msg ? msg : "TMA encode failed");
  }
}
void encode3(CUtensorMap *map, uintptr_t ptr, int rows, int cols, int box_y,
             int box_z) {
  cuuint64_t dims[] = {64, cuuint64_t(rows), cuuint64_t(cols / 64)},
             strides[] = {cuuint64_t(cols) * 2, 128};
  cuuint32_t box[] = {64, cuuint32_t(box_y), cuuint32_t(box_z)},
             step[] = {1, 1, 1};
  auto rc = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, reinterpret_cast<void *>(ptr),
      dims, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS)
    throw std::runtime_error("rank3 input TMA map failed");
}
template <int BM, int BN, int BK, int Stages> constexpr int shared_bytes() {
  return Stages * (BM * BK + BK * BN) * 2;
}
template <int BM, int BN, int BK, int Stages> void setup(bool single) {
  if (single)
    check(cudaFuncSetAttribute(gemm<BM, BN, BK, Stages, true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shared_bytes<BM, BN, BK, Stages>()));
  else
    check(cudaFuncSetAttribute(gemm<BM, BN, BK, Stages, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shared_bytes<BM, BN, BK, Stages>()));
}
template <int BM, int BN, int BK, int Stages> void launch(const Prepared &s) {
  dim3 grid((s.n + BN - 1) / BN, (s.m + BM - 1) / BM, s.splits);
  if (s.single)
    gemm<BM, BN, BK, Stages, true>
        <<<grid, 128, shared_bytes<BM, BN, BK, Stages>(), s.stream>>>(
            s.a, s.b, s.partial ? s.partial : s.c, s.m, s.n, s.k, s.tiles,
            s.packed);
  else
    gemm<BM, BN, BK, Stages, false>
        <<<grid, 160, shared_bytes<BM, BN, BK, Stages>(), s.stream>>>(
            s.a, s.b, s.partial ? s.partial : s.c, s.m, s.n, s.k, s.tiles,
            s.packed);
}
template <bool Run> void dispatch(const Prepared &s) {
  // A true 32-column B tile gives the narrow-N path 128 independent CTAs for
  // the representative PV shape. Its 64B swizzle avoids padding the B traffic
  // and lets four K256 stages fit without a separate split-K reduction.
  if (s.bm == 64 && s.bn == 32 && s.bk == 256 && s.stages == 4) {
    if constexpr (Run)
      launch<64, 32, 256, 4>(s);
    else
      setup<64, 32, 256, 4>(s.single);
    return;
  }
// Only resource-valid combinations are instantiated. The explicit branches
// make each teaching/ablation configuration visible in ordinary CUDA C++.
#define CONFIG(N, K, S)                                                        \
  if (s.bn == N && s.bk == K && s.stages == S) {                               \
    if (s.bm == 64) {                                                          \
      if constexpr (Run)                                                       \
        launch<64, N, K, S>(s);                                                \
      else                                                                     \
        setup<64, N, K, S>(s.single);                                          \
    } else {                                                                   \
      if constexpr (Run)                                                       \
        launch<128, N, K, S>(s);                                               \
      else                                                                     \
        setup<128, N, K, S>(s.single);                                         \
    }                                                                          \
    return;                                                                    \
  }
  CONFIG(64, 64, 2)
  CONFIG(64, 64, 3)
  CONFIG(128, 64, 2)
  CONFIG(128, 64, 3)
  CONFIG(64, 128, 1)
  CONFIG(64, 128, 2)
  CONFIG(64, 128, 3)
  CONFIG(64, 128, 4)
  CONFIG(128, 128, 1)
  CONFIG(128, 128, 2)
  CONFIG(128, 128, 3)
  CONFIG(64, 256, 1)
  CONFIG(64, 256, 2)
  CONFIG(128, 256, 1)
  CONFIG(64, 512, 1)
#undef CONFIG
  throw std::runtime_error(
      "unsupported tile_n/tile_k/stages shared-memory configuration");
}
} // namespace
void *prepare_gemm(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k,
                   uintptr_t stream, int tile_m, int tile_n, int tile_k,
                   int split_k, int stages, int single_issuer) {
  if (m % 64 || n % 8 || k % 16)
    throw std::runtime_error("SM100 GEMM requires M%64=N%8=K%16=0");
  auto *s = new Prepared{};
  try {
    s->m = m;
    s->n = n;
    s->k = k;
    s->c = reinterpret_cast<float *>(c);
    s->stream = reinterpret_cast<cudaStream_t>(stream);
    bool narrow = n <= 128 && k >= 512;
    bool wide = k >= 512 && !narrow;
    s->bm = tile_m ? tile_m : (narrow ? 64 : 128);
    if (s->bm != 64 && s->bm != 128)
      throw std::runtime_error("tile_m must be 64 or 128");
    s->bn = tile_n ? tile_n
                   : (narrow                           ? 32
                      : wide || std::min(m, n) <= 1024 ? 64
                                                       : 128);
    s->bk = tile_k ? tile_k : (wide || k <= 128 ? 128 : 256);
    if (!tile_k && s->bn == 128 && k > 128)
      s->bk = 128;
    s->stages = stages ? stages
                       : (s->bn == 32 || (wide && s->bn == 64) ? 4
                          : k <= s->bk                         ? 1
                                                               : 2);
    s->single = single_issuer != 0;
    s->splits = split_k ? split_k : 1;
    if (s->splits < 1)
      throw std::runtime_error("split_k must be positive");
    // dispatch validates the tiling and resource limits before descriptor use.
    dispatch<false>(*s);
    int total = (k + s->bk - 1) / s->bk;
    s->tiles = (total + s->splits - 1) / s->splits;
    s->splits = (total + s->tiles - 1) / s->tiles;
    s->packed = wide && s->bn >= 64 && s->bk <= 256 && m % 64 == 0 &&
                n % 64 == 0 && k % 64 == 0;
    if (s->packed) {
      encode3(&s->a, a, m, k, s->bm, s->bk / 64);
      encode3(&s->b, b, k, n, s->bk, s->bn / 64);
    } else {
      encode(&s->a, a, m, k, s->bm);
      encode(&s->b, b, k, n, std::min(s->bk, 256), std::min(s->bn, 64));
    }
    if (s->splits > 1)
      check(
          cudaMalloc(&s->partial, int64_t(s->splits) * m * n * sizeof(float)));
  } catch (...) {
    if (s->partial)
      cudaFree(s->partial);
    delete s;
    throw;
  }
  return s;
}
void run_gemm(void *p) {
  auto &s = *static_cast<Prepared *>(p);
  dispatch<true>(s);
  if (s.partial)
    reduce_splits<<<(s.m * s.n + 1023) / 1024, 256, 0, s.stream>>>(
        s.partial, s.c, s.m * s.n, s.splits);
  check(cudaGetLastError());
}
void delete_gemm(void *p) {
  auto *s = static_cast<Prepared *>(p);
  if (s->partial)
    cudaFree(s->partial);
  delete s;
}
size_t gemm_workspace_bytes(void *p) {
  auto *s = static_cast<Prepared *>(p);
  return s->partial ? size_t(s->splits) * s->m * s->n * sizeof(float) : 0;
}
