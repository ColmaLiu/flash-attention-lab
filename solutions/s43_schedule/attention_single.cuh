#pragma once
// Native CUDA C++ / PTX attention for SM100. All device operations are local.
// One query warp group overlaps softmax with the next QK using two
// score slots and three K/V pairs. This increases query parallelism.
// See README.md for buffer ownership, phase accounting and timing boundaries.
#include "attention_common.cuh"
#include "attention_exp.cuh"

namespace sm100_attention_single {
constexpr int BM = 128, BN = 128;
constexpr int TmemColumns = 512, QueryGroups = 1, ScoreSlots = 2, Threads = 256;
static_assert(BM == 128 && BN == 128, "pingpong kernel requires 128x128 tiles");
__device__ __forceinline__ unsigned shared(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ bool elected() {
  unsigned result;
  asm volatile(
      "{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
      : "=r"(result));
  return result;
}
__device__ __forceinline__ void init_bar(uint64_t *p, unsigned arrivals = 1) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared(p)),
               "r"(arrivals)
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
// The producer waits for MMA completion without accessing TMEM itself.
__device__ __forceinline__ void wait_empty(uint64_t *p, int phase) {
  asm volatile("{ .reg .pred done; wait_empty_loop: "
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
               "@!done bra wait_empty_loop; }" ::"r"(shared(p)),
               "r"(phase)
               : "memory");
}
// ID 1 belongs to the 128 softmax lanes; ID 3 to the 128 control lanes.
// ID 0 is reserved for common CTA barriers at task boundaries.
__device__ __forceinline__ void group_sync(int group) {
  if (group == 0)
    asm volatile("barrier.sync 1, 128;" ::: "memory");
  else
    asm volatile("barrier.sync 2, 128;" ::: "memory");
}
// Called by one lane only after all consumer TMEM stores have completed
// and the before/barrier.sync/after handoff has collected the WG's writes.
__device__ __forceinline__ void arrive_bar(uint64_t *p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(shared(p))
               : "memory");
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
template <class T, int N>
__device__ __forceinline__ void mma_ts(unsigned output, unsigned a, uint64_t b,
                                       bool accumulate) {
  constexpr unsigned type = std::is_same_v<T, __half> ? 0 : 1;
  constexpr unsigned instruction = (1u << 4) | (type << 7) | (type << 10) |
                                   (1u << 16) | ((N / 8) << 17) |
                                   ((BM / 16) << 24);
  asm volatile("{ .reg .pred accumulate; setp.ne.b32 accumulate, %4, 0; "
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
               "{%5,%5,%5,%5}, accumulate; }" ::"r"(output),
               "r"(a), "l"(b), "r"(instruction), "r"(int(accumulate)), "r"(0)
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
__device__ __forceinline__ void store16(const float *x, unsigned address) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
          address),
      "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]),
      "f"(x[6]), "f"(x[7]), "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]),
      "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15])
      : "memory");
}
__device__ __forceinline__ void wait_stores() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wait_loads() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Native SM100 packed FP32 arithmetic, PTX ISA 8.7 section 9.7.3.
__device__ __forceinline__ void mul2(float *p, float factor) {
  float2 x{p[0], p[1]}, f{factor, factor}, y;
  asm("mul.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t &>(y))
      : "l"(reinterpret_cast<const uint64_t &>(x)),
        "l"(reinterpret_cast<const uint64_t &>(f)));
  p[0] = y.x;
  p[1] = y.y;
}
__device__ __forceinline__ void add2(float *p, const float *q) {
  float2 x{p[0], p[1]}, v{q[0], q[1]}, y;
  asm("add.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t &>(y))
      : "l"(reinterpret_cast<const uint64_t &>(x)),
        "l"(reinterpret_cast<const uint64_t &>(v)));
  p[0] = y.x;
  p[1] = y.y;
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
  const uint64_t qa_base = descriptor(q), kb_base = descriptor(k);
#pragma unroll
  for (int d = 0; d < D; d += 16) {
    constexpr int BytesPerElement = sizeof(T);
    const uint64_t qa =
        qa_base + ((d / 64) * BM * 64 + d % 64) * BytesPerElement / 16;
    const uint64_t kb =
        kb_base + ((d / 64) * BN * 64 + d % 64) * BytesPerElement / 16;
    mma<T, BN>(out, qa, kb, d != 0);
  }
  commit_mma(bar);
}
template <class T, int D>
__device__ __forceinline__ void pv(unsigned p, T *v, unsigned out,
                                   uint64_t *bar, bool accumulate) {
  const uint64_t vb_base = descriptor(v, BN * 128);
#pragma unroll
  for (int k = 0; k < BN; k += 16) {
    mma_ts<T, D>(out, p + k / 2, vb_base + k * 64 * sizeof(T) / 16,
                 k != 0 || accumulate);
  }
  commit_mma(bar);
}

// One softmax WG owns Q and O. S/P alternate between two TMEM regions.
// QK(t+2) waits for PV(t) before reusing its S/P slot. Softmax(t+1) may
// compute early but explicitly waits for PV(t) before rescaling O.
template <class T, int D, bool Overlap, bool Numerics, bool Persistent,
          bool Polynomial = false, bool Masked = true,
          bool PositiveScale = true>
__global__ __launch_bounds__(Threads, 1) void attention(
    const __grid_constant__ CUtensorMap qmap,
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, T *out, float *lse, int n,
    int heads, int total, float scale, bool causal) {
  static_assert(Overlap, "pingpong kernel requires its three-pair KV ring");
  constexpr int Buffers = 3;
  extern __shared__ __align__(1024) unsigned char raw[];
  T *q_all = reinterpret_cast<T *>(raw);
  T *k = q_all + QueryGroups * BM * D;
  T *v = k + Buffers * BN * D;
  __shared__ uint64_t qbar[QueryGroups], scorebar[ScoreSlots],
      pvbar[ScoreSlots], p_ready[ScoreSlots];
  __shared__ uint64_t kv_full[Buffers], kv_empty[Buffers];
  __shared__ unsigned tmem;
  if (threadIdx.x < 32) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(shared(&tmem)),
        "n"(TmemColumns)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  __syncthreads();
  const int qtiles = (n + QueryGroups * BM - 1) / (QueryGroups * BM);
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int query_tile = task % qtiles, bh = task / qtiles;
    if constexpr (Persistent)
      if (causal) {
        int batch_heads = total / qtiles;
        query_tile = qtiles - 1 - task / batch_heads;
        bh = task % batch_heads;
      }
    int head = bh % heads, batch = bh / heads;
    int pair_query = query_tile * QueryGroups * BM;
    int tiles =
        ((causal ? min(n, pair_query + QueryGroups * BM) : n) + BN - 1) / BN;
    if (threadIdx.x == 0) {
      init_bar(&qbar[0]);
#pragma unroll
      for (int i = 0; i < ScoreSlots; ++i) {
        init_bar(&scorebar[i]);
        init_bar(&pvbar[i]);
        init_bar(&p_ready[i]);
      }
#pragma unroll
      for (int i = 0; i < Buffers; ++i) {
        init_bar(&kv_full[i]);
        init_bar(&kv_empty[i]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x >= QueryGroups * BM) {
      if (threadIdx.x == QueryGroups * BM) {
#pragma unroll
        for (int i = 0; i < QueryGroups; ++i)
          load_q<T, D>(&qmap, q_all + i * BM * D, &qbar[i], head,
                       pair_query + i * BM, batch);
        for (int tile = 0; tile < tiles; ++tile) {
          int slot = tile % Buffers, generation = tile / Buffers;
          // A single arrival releases this pair after PV is complete.
          if (generation)
            wait_empty(&kv_empty[slot], (generation - 1) & 1);
          load_kv<T, D>(&kmap, &vmap, k + slot * BN * D, v + slot * BN * D,
                        &kv_full[slot], head, tile * BN, batch);
        }
      }
      if (threadIdx.x / 32 == QueryGroups * BM / 32 + 1 && elected()) {
        // The two initial scores use distinct S/P slots but the same Q.
        wait_bar(&qbar[0], 0);
        wait_bar(&kv_full[0], 0);
        qk<T, D>(q_all, k, tmem, &scorebar[0]);
        if (tiles > 1) {
          wait_bar(&kv_full[1], 0);
          qk<T, D>(q_all, k + BN * D, tmem + BN, &scorebar[1]);
        }
        for (int tile = 0; tile < tiles; ++tile) {
          int slot = tile % Buffers, score_slot = tile % ScoreSlots;
          int phase = (tile / ScoreSlots) & 1;
          wait_bar(&p_ready[score_slot], phase);
          pv<T, D>(tmem + score_slot * BN + BN / 2, v + slot * BN * D,
                   tmem + 256, &pvbar[score_slot], tile > 0);
          // S/P reuse has a real WAR dependency across different
          // accumulators. Keep the explicit PTX 8.7 completion wait.
          wait_bar(&pvbar[score_slot], phase);
          commit_mma(&kv_empty[slot]);
          if (tile + ScoreSlots < tiles) {
            int next_tile = tile + ScoreSlots;
            int next_slot = next_tile % Buffers;
            wait_bar(&kv_full[next_slot], (next_tile / Buffers) & 1);
            qk<T, D>(q_all, k + next_slot * BN * D, tmem + score_slot * BN,
                     &scorebar[score_slot]);
          }
        }
      }
      asm volatile("barrier.sync 3, 128;" ::: "memory");
    } else {
      const int group = threadIdx.x / BM, local_row = threadIdx.x % BM;
      const int warp = (threadIdx.x / 32) % 4;
      T *q = q_all + group * BM * D;
      const unsigned row_address = tmem + ((warp * 32) << 16);
      const unsigned output_address = row_address + 256;
      int query = pair_query + group * BM, row = query + local_row;
      float maximum = -CUDART_INF_F, denominator = 0.f;
      const float scale2 = scale * 1.4426950408889634f;
      for (int tile = 0; tile < tiles; ++tile) {
        const int score_slot = tile % ScoreSlots;
        const unsigned score_address = row_address + score_slot * BN;
        // This wait only protects S/P. It does not imply PV(t-1)
        // completed: the other score slot was intentionally issued early.
        wait_bar(&scorebar[score_slot], (tile / ScoreSlots) & 1);
        float scores[BN];
#pragma unroll
        for (int j = 0; j < BN; j += 16)
          load16(scores + j, score_address + j);
        wait_loads();
        float maxima[4] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F,
                           -CUDART_INF_F};
        // Reduce raw scores before scaling. Extreme positive finite scales
        // retain a raw maximum and use FP64 products of score differences;
        // forming scale*log2(e) in FP32 can otherwise overflow or flush.
        if constexpr (Masked) {
          if (tile * BN + BN > n || (causal && tile * BN + BN > query)) {
#pragma unroll
            for (int j = 0; j < BN; ++j) {
              int key = tile * BN + j;
              if (key >= n || (causal && key > row))
                scores[j] = -CUDART_INF_F;
            }
          }
        }
#pragma unroll
        for (int j = 0; j < BN; ++j)
          maxima[j % 4] = fmaxf(maxima[j % 4], scores[j]);
        float tile_max =
            fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
        if constexpr (PositiveScale)
          tile_max *= scale2;
        float next_max = fmaxf(maximum, tile_max);
        if constexpr (Numerics) {
          // Delayed rescale: keep the old reference when all probabilities
          // remain <= 16. This is not skipping contributions or masking.
          if constexpr (PositiveScale) {
            if (next_max - maximum < 4.f)
              next_max = maximum;
          } else {
            if ((double(next_max) - double(maximum)) * double(scale) *
                    1.4426950408889634 <
                4.0)
              next_max = maximum;
          }
        }
        float correction;
        if constexpr (PositiveScale)
          correction = exp2_sfu(maximum - next_max);
        else
          correction = exp2_sfu(float((double(maximum) - double(next_max)) *
                                      double(scale) * 1.4426950408889634));
        maximum = next_max;
        float sums[4] = {};
        if constexpr (PositiveScale) {
#pragma unroll
          for (int j = 0; j < BN; j += 2) {
            // Match tile_max's rounded multiplication before subtracting.
            // An FMA against an already rounded maximum leaves a large
            // residual for large finite QK, overflowing/underflowing P.
            mul2(scores + j, scale2);
            float reference[2] = {-maximum, -maximum};
            add2(scores + j, reference);
          }
        }
        if constexpr (Polynomial && PositiveScale) {
// 48 of 128 exponents use the packed quartic: first pair in
// each group of four, with the final 32 scores all on SFU.
// Correction remains SFU; delayed rescaling is independent.
#pragma unroll
          for (int j = 0; j < BN; j += 2) {
            if (j < BN - 32 && j % 4 == 0)
              sm100_attention_exp::exp2_packed_quartic(scores + j);
            else {
              scores[j] = exp2_sfu(scores[j]);
              scores[j + 1] = exp2_sfu(scores[j + 1]);
            }
          }
        } else {
// The disabled approximation and FP64-difference extreme-
// scale path both keep all score exponentials on SFU.
#pragma unroll
          for (int j = 0; j < BN; ++j) {
            float x;
            if constexpr (PositiveScale)
              x = scores[j];
            else
              x = float((double(scores[j]) - double(maximum)) * double(scale) *
                        1.4426950408889634);
            scores[j] = exp2_sfu(x);
          }
        }
#pragma unroll
        for (int j = 0; j < BN; j += 2)
          add2(sums + j % 4, scores + j);
        denominator = denominator * correction + (sums[0] + sums[1]) +
                      (sums[2] + sums[3]);
#pragma unroll
        for (int j = 0; j < BN; j += 32) {
          float packed[16];
#pragma unroll
          for (int t = 0; t < 16; ++t)
            packed[t] = __uint_as_float(
                pack<T>(scores[j + 2 * t], scores[j + 2 * t + 1]));
          // Two input values occupy each 32-bit TMEM cell. Alias the
          // upper half of the consumed score slot as the MMA A operand.
          store16(packed, score_address + BN / 2 + j / 2);
        }
        wait_stores();
        // Preserve the sequential online-softmax O recurrence while its
        // score/max/exp/P work overlaps the previous PV and next QK.
        if (tile > 0)
          wait_bar(&pvbar[(tile - 1) % ScoreSlots],
                   ((tile - 1) / ScoreSlots) & 1);
        // Keep the running O numerator in TMEM. Most tiles need no
        // correction at all when the softmax reference is unchanged.
        // TMEM loads/stores are warp-collective even if one row rescales.
        if (tile > 0 && __any_sync(0xffffffffu, correction != 1.f)) {
#pragma unroll
          for (int d = 0; d < D; d += 16) {
            float values[16];
            load16(values, output_address + d);
            wait_loads();
#pragma unroll
            for (int j = 0; j < 16; j += 2)
              mul2(values + j, correction);
            store16(values, output_address + d);
          }
          wait_stores();
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        group_sync(group);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (local_row == 0)
          arrive_bar(&p_ready[score_slot]);
        // Continue immediately into the other score slot. The final
        // epilogue alone needs this tile's PV completion here.
        if (tile + 1 == tiles) {
          wait_bar(&pvbar[score_slot], (tile / ScoreSlots) & 1);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          group_sync(group);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
      }
      // Reuse Q's dead shared allocation for a coalesced output epilogue.
      const float inverse_denominator = 1.f / denominator;
#pragma unroll
      for (int d = 0; d < D; d += 16) {
        float values[16];
        load16(values, output_address + d);
        wait_loads();
#pragma unroll
        for (int j = 0; j < 16; j += 8) {
          int4 packed;
          packed.x = pack<T>(values[j] * inverse_denominator,
                             values[j + 1] * inverse_denominator);
          packed.y = pack<T>(values[j + 2] * inverse_denominator,
                             values[j + 3] * inverse_denominator);
          packed.z = pack<T>(values[j + 4] * inverse_denominator,
                             values[j + 5] * inverse_denominator);
          packed.w = pack<T>(values[j + 6] * inverse_denominator,
                             values[j + 7] * inverse_denominator);
          *reinterpret_cast<int4 *>(q + offset(BM, local_row, d + j)) = packed;
        }
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      if (row < n) {
        if constexpr (PositiveScale)
          lse[static_cast<int64_t>(bh) * n + row] =
              maximum * 0.6931471805599453f + logf(denominator);
        else
          lse[static_cast<int64_t>(bh) * n + row] = float(
              double(maximum) * double(scale) + double(logf(denominator)));
      }
      group_sync(group);
#pragma unroll
      for (int vec = local_row; vec < BM * D / 8; vec += BM) {
        int r = vec / (D / 8), d = (vec % (D / 8)) * 8;
        if (query + r < n) {
          auto dst =
              out +
              ((static_cast<int64_t>(batch) * n + query + r) * heads + head) *
                  D +
              d;
          *reinterpret_cast<int4 *>(dst) =
              *reinterpret_cast<int4 *>(q + offset(BM, r, d));
        }
      }
      // All threads used Q through generic shared accesses in the epilogue.
      // Order those reads/writes before the next task's async TMA overwrite;
      // a fence executed only by the TMA issuer would not cover its peers.
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      group_sync(group);
    }
    // All control work and the softmax epilogue finish before another task.
    __syncthreads();
    if (threadIdx.x == 0) {
// Drain each used slot's last empty generation before invalidating:
// its asynchronous commit may arrive later than the WG's pvbar.
#pragma unroll
      for (int slot = 0; slot < Buffers; ++slot) {
        if (slot < tiles) {
          int last_tile = tiles - 1;
          last_tile -= (last_tile - slot) % Buffers;
          wait_empty(&kv_empty[slot], (last_tile / Buffers) & 1);
        }
      }
      asm volatile(
          "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&qbar[0]))
          : "memory");
#pragma unroll
      for (int i = 0; i < ScoreSlots; ++i) {
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&scorebar[i]))
            : "memory");
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&pvbar[i]))
            : "memory");
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&p_ready[i]))
            : "memory");
      }
#pragma unroll
      for (int i = 0; i < Buffers; ++i) {
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&kv_full[i]))
            : "memory");
        asm volatile(
            "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared(&kv_empty[i]))
            : "memory");
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 32)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
        "n"(TmemColumns)
        : "memory");
}

using sm100_attention::check;
using sm100_attention::Prepared;

template <class T, int D, bool Overlap, bool Numerics, bool Persistent,
          bool Polynomial = false>
inline void launch(const Prepared &s) {
  static_assert(Overlap, "pingpong kernel uses three KV pair buffers");
  constexpr int Buffers = 3;
  constexpr int SharedBytes = (QueryGroups * BM * D + 2 * Buffers * BN * D) * 2;
  // Entire noncausal key tiles need no per-score key/causal predicates.
  const float scale2 = s.scale * 1.4426950408889634f;
  const bool positive_scale =
      std::isnormal(s.scale) && std::isnormal(scale2) && scale2 <= 1.f;
  auto kernel =
      !s.causal && s.n % BN == 0
          ? (positive_scale ? attention<T, D, Overlap, Numerics, Persistent,
                                        Polynomial, false, true>
                            : attention<T, D, Overlap, Numerics, Persistent,
                                        Polynomial, false, false>)
          : (positive_scale ? attention<T, D, Overlap, Numerics, Persistent,
                                        Polynomial, true, true>
                            : attention<T, D, Overlap, Numerics, Persistent,
                                        Polynomial, true, false>);
  check(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
  int total =
      s.batch * s.heads * ((s.n + QueryGroups * BM - 1) / (QueryGroups * BM));
  int grid = Persistent ? std::min(total, s.sms * 2) : total;
  kernel<<<grid, Threads, SharedBytes, s.stream>>>(
      s.q, s.k, s.v, static_cast<T *>(s.output), s.lse, s.n, s.heads, total,
      s.scale, s.causal);
  check(cudaGetLastError());
}
// Independent compile-time ablations keep the measured launch free of
// device-side mode branches. The default uses mixed quartic/SFU plus delayed
// rescaling.
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
} // namespace sm100_attention_single
