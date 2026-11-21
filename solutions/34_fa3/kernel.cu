// Reference solution for 34_fa3: FlashAttention-3's ideas on SM100 --
// warp-specialized TMA producer, S and O accumulated in Tensor Memory, P
// through shared memory, and QK^T of the next block overlapped with softmax.
// Lesson: course/34_fa3/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous. LSE: [B, H, N] FP32 (ln).
// D in {64, 128}, scale > 0 finite. One CTA per (Q tile of 128 rows, head,
// batch) = (blockIdx.x, .y, .z), 6 warps with fixed roles:
//
//   warps 0-3  softmax warpgroup: thread r owns query row q0 + r = TMEM lane r.
//              Reads S with tcgen05.ld, online softmax in registers, writes P
//              (16-bit) to shared memory, rescales O in TMEM (tcgen05.ld/st),
//              finally normalizes O, TMA-stores it and writes LSE.
//   warp 4     TMA load warp (one lane): Q once, then K_j / V_j into a ring of
//              kStages stages, each with its own full/empty mbarriers.
//   warp 5     MMA warp (one lane): S_j = Q K_j^T (SS, both K-major) and
//              O += P_j V_j (SS, P K-major, V MN-major), committing every
//              group of MMAs to an mbarrier.
//
// step 1: one K block (N <= 128); the softmax warps write scale * S to the
//         probe instead of doing softmax.
// step 2: one K/V block, full attention: P -> smem, P.V into TMEM, final
//         normalization, O via TMA store, LSE.
// step 3: full K/V loop: online softmax, O rescaled in TMEM by the softmax
//         warps before every P.V after the first, stage ring wraps.
// step 4: causal: K blocks after the diagonal are skipped, the diagonal block
//         (and the N tail) is masked.
// step 5: S double-buffered in TMEM: the MMA warp computes S_{j+1} while the
//         softmax warps work on S_j.
//
// Everything every barrier protects is spelled out at its declaration
// (struct Barriers) and in the lesson's timeline (section 5).
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

using bf16 = __nv_bfloat16;
using fp16 = __half;

constexpr int kBM = 128;     // query rows per CTA = MMA M = TMEM lanes
constexpr int kBN = 128;     // keys per K/V block = N of S = K of P.V
constexpr int kPanel = 64;   // 64 16-bit elements = one 128-byte swizzle row
constexpr int kUmmaK = 16;   // K of one tcgen05.mma for 16-bit inputs
constexpr int kLoadWarp = 4; // warp roles
constexpr int kMmaWarp = 5;
constexpr int kThreads = 6 * 32;

// ===========================================================================
// Small local helpers (the same as lab 22).
// ===========================================================================

// 2^x with MUFU.EX2 (ex2.approx.ftz.f32; 2^-inf = +0).
__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Two FP32 -> one 32-bit word of two 16-bit values (x in the low half).
template <class T> __device__ __forceinline__ uint32_t pack2(float x, float y) {
  if constexpr (std::is_same_v<T, fp16>) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
  } else {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
  }
}

// ===========================================================================
// Configuration and shared-memory layout.
//
// Every tile is stored as TMA writes it with a 64-element box row and the
// 128B swizzle: [D/64 panels][rows][64], 16-byte chunk q of row r at chunk
// q ^ (r % 8). Q, K, V, P and the O staging area all use this layout.
// ===========================================================================
constexpr int pow2_at_least(int x) {
  int p = 32;
  while (p < x)
    p *= 2;
  return p;
}

template <int D_, int SBuf> struct Config {
  static constexpr int kD = D_;
  static constexpr int kPanels = D_ / kPanel;
  static constexpr int kSBuf = SBuf; // S buffers in TMEM (1, or 2 in step 5)
  // K/V ring depth: as many as fit next to Q and P while one CTA uses the SM.
  static constexpr int kStages = D_ == 128 ? 2 : 3;
  static constexpr int kQBytes = kBM * D_ * 2;
  static constexpr int kKVBytes = kBN * D_ * 2; // one K (or V) block
  static constexpr int kPBytes = kBM * kBN * 2;
  static constexpr int kQOff = 0, kPOff = kQBytes, kKOff = kPOff + kPBytes,
                       kVOff = kKOff + kStages * kKVBytes;
  static constexpr int kSmemBytes = 1024 + kVOff + kStages * kKVBytes;
  // TMEM columns: S buffer b at b * kBN, O (FP32, D columns) after them.
  static constexpr int kOCol = SBuf * kBN;
  static constexpr int kTmemCols = pow2_at_least(kOCol + D_);
  static_assert(kTmemCols <= 512, "TMEM has 512 columns");
  static_assert(kSmemBytes <= 227 * 1024, "shared memory per CTA");
  // One CTA per SM: a second CTA would block in tcgen05.alloc (step 5 uses
  // all 512 columns) while holding shared memory.
  static_assert(2 * kSmemBytes > 228 * 1024, "keep one CTA per SM");
};

// Every mbarrier of the kernel. "count" is the arrival count per phase.
template <int Stages> struct Barriers {
  // TMA -> MMA. count 1 (arrive_expect_tx by the load warp) + tx bytes.
  uint64_t q_full;         // Q tile landed (one phase only)
  uint64_t k_full[Stages]; // K_j landed in stage j % Stages
  uint64_t v_full[Stages]; // V_j landed
  // MMA -> TMA. count 1 (tcgen05.commit): the MMAs reading the stage are
  // complete, so TMA may overwrite it.
  uint64_t k_empty[Stages]; // after Q K_j^T
  uint64_t v_empty[Stages]; // after P_j V_j
  // MMA -> softmax. count 1 (tcgen05.commit after Q K_j^T): S_j is in TMEM
  // buffer s_buffer(j).
  uint64_t s_full[2];
  // softmax -> MMA. count 128 (every softmax thread): P_j is in shared
  // memory (fence.proxy.async done), the thread's tcgen05.ld of S_j are
  // complete (its S buffer may be overwritten), and its O row has been
  // rescaled (tcgen05.st complete).
  uint64_t p_ready;
  // MMA -> softmax. count 1 (tcgen05.commit after P_j V_j): O includes
  // block j, and the MMA has finished reading P_j (P may be overwritten).
  uint64_t o_ready;
  uint32_t tmem_slot; // TMEM base address written by tcgen05.alloc
};

struct Params {
  int n, h;
  float scale, scale_log2; // scale_log2 = scale * log2(e)
  float *lse, *probe;
};

// K/V blocks this Q tile needs. Causal: query rows q0 .. q0 + kBM - 1 see
// keys <= q0 + kBM - 1, so later blocks are skipped (never loaded).
template <bool kCausal>
__device__ __forceinline__ int num_kv_blocks(int q0, int n) {
  int blocks = (n + kBN - 1) / kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + kBM + kBN - 1) / kBN);
  return blocks;
}

// The S double-buffer protocol (lesson section 5.5). S_i lives in TMEM
// buffer s_buffer(i); its tcgen05.commit completes phase number
// (number of earlier blocks in the same buffer) of s_full[s_buffer(i)], and
// waiting for that phase means waiting for its parity. Both the MMA warp
// (where to write S_i, which barrier to commit) and the softmax warps (where
// to read S_i, which parity to wait for) use these two functions.
template <int kSBuf> __device__ __forceinline__ int s_buffer(int i) {
  if constexpr (kSBuf == 1) {
    return 0;
  } else {
    return i % 2;
  }
}
template <int kSBuf> __device__ __forceinline__ uint32_t s_phase(int i) {
  if constexpr (kSBuf == 1) {
    return i & 1;
  } else {
    return (i / 2) & 1;
  }
}

// --------------------------------------------------------------------- MMA
// S (TMEM, 128 x 128 FP32) = Q[128 x D] . K_j[128 x D]^T: D / 16 MMAs, both
// operands K-major (D contiguous). K slice kk lies in panel 16kk / 64, at
// byte 2 * (16kk % 64) of the swizzled row (lab 31, section 4.4).
template <class T, class C>
__device__ __forceinline__ void mma_qk(uint32_t s_tmem, const T *sq,
                                       const T *sk) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kBM, kBN, std::is_same_v<T, bf16>);
#pragma unroll
  for (int kk = 0; kk < C::kD / kUmmaK; ++kk) {
    const int panel = kk * kUmmaK / kPanel, col = kk * kUmmaK % kPanel;
    const uint64_t da =
        lab::make_smem_desc_kmajor_sw128(sq + panel * kBM * kPanel + col);
    const uint64_t db =
        lab::make_smem_desc_kmajor_sw128(sk + panel * kBN * kPanel + col);
    lab::mma_f16_ss(s_tmem, da, db, idesc, kk > 0);
  }
}

// O (TMEM, 128 x D FP32) [+]= P[128 x 128 keys] . V_j[128 keys x D].
// In the C = A . B^T convention, B is "N x K" = D x keys. V is stored
// [keys][D] (D contiguous): the N index is contiguous, i.e. MN-major. So
//   - idesc bit 16 (B major) = 1 ("transpose B");
//   - the B descriptor is MN-major: SBO = 1024 (8 key rows), LBO = distance
//     between the 64-wide D panels = kBN rows * 128 B;
//   - key slice kk (keys 16kk .. 16kk + 15) starts 16 rows = 2 atoms further:
//     + 16kk * 64 elements. P is K-major (keys contiguous) like Q.
template <class T, class C>
__device__ __forceinline__ void mma_pv(uint32_t o_tmem, const T *sp,
                                       const T *sv, bool accumulate) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kBM, C::kD, std::is_same_v<T, bf16>, true,
                          /*a_mn_major=*/false, /*b_mn_major=*/true);
#pragma unroll
  for (int kk = 0; kk < kBN / kUmmaK; ++kk) {
    const int panel = kk * kUmmaK / kPanel, col = kk * kUmmaK % kPanel;
    const uint64_t da =
        lab::make_smem_desc_kmajor_sw128(sp + panel * kBM * kPanel + col);
    const uint64_t db = lab::make_smem_desc_mnmajor_sw128(
        sv + kk * kUmmaK * kPanel, kBN * kPanel * 2);
    lab::mma_f16_ss(o_tmem, da, db, idesc, accumulate || kk > 0);
  }
}

// ===========================================================================
// Softmax-warpgroup helpers. Thread r of warps 0-3 owns query row q0 + r,
// which is TMEM lane r; `lane_addr` = tmem base + (32 * warp << 16).
// ===========================================================================

// The thread's row of S (128 FP32 columns starting at column `col`) into
// registers. The _sync loads put tcgen05.wait::ld in the same asm statement,
// so no arithmetic on s[] can be scheduled before the data has arrived.
__device__ __forceinline__ void load_s_row(uint32_t lane_addr, int col,
                                           float (&s)[kBN]) {
#pragma unroll
  for (int c = 0; c < kBN; c += 32)
    lab::tmem_ld_32x32b_x32_sync(lane_addr + col + c,
                                 *reinterpret_cast<float (*)[32]>(s + c));
}

// Does K block k0 .. k0 + kBN - 1 contain a key that some row of the tile
// q0 .. q0 + kBM - 1 must not see? Only the block that crosses N and, causal,
// the block(s) crossing the diagonal.
template <bool kCausal>
__device__ __forceinline__ bool block_needs_mask(int k0, int q0, int n) {
  if (k0 + kBN > n)
    return true;
  if constexpr (kCausal)
    return k0 + kBN - 1 > q0;
  return false;
}

// Number of keys of block k0 that query `row` may see (keys k0 .. k0 +
// result - 1; may be <= 0 or > kBN).
template <bool kCausal>
__device__ __forceinline__ int visible_keys(int row, int k0, int n) {
  int end = n;
  if constexpr (kCausal)
    end = min(end, row + 1);
  return end - k0;
}

// One block of the online softmax (lab 22, section 3.1), in raw score units
// (scale > 0, so max(scale * s) = scale * max(s)):
//   m_new = max(m_old, max_c s[c])
//   base  = m_new * scale_log2      (0 if m_new = -inf: never -inf - -inf)
//   s[c]  = 2^(s[c] * scale_log2 - base) = exp(scale * (s[c] - m_new))
//   returns row_sum = sum_c s[c]
// Eight independent partial maxima / sums: one warp per SMSP, so a single
// dependent chain of 128 FMNMX/FADD would expose their latency.
__device__ __forceinline__ float softmax_block(float (&s)[kBN], float m_old,
                                               float scale_log2, float &m_new,
                                               float &base) {
  float mx[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    mx[i] = s[i];
#pragma unroll
  for (int c = 8; c < kBN; ++c)
    mx[c % 8] = fmaxf(mx[c % 8], s[c]);
  m_new = m_old;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    m_new = fmaxf(m_new, mx[i]);
  base = m_new == -INFINITY ? 0.f : m_new * scale_log2;
  float sum[8] = {};
#pragma unroll
  for (int c = 0; c < kBN; ++c) {
    s[c] = exp2_approx(fmaf(s[c], scale_log2, -base));
    sum[c % 8] += s[c];
  }
  return ((sum[0] + sum[1]) + (sum[2] + sum[3])) +
         ((sum[4] + sum[5]) + (sum[6] + sum[7]));
}

// Row r of P (16-bit) -> shared memory, 128B-swizzled K-major, exactly as
// TMA would have written it: keys 64pn .. 64pn + 63 in panel pn, 16-byte
// chunk q (keys 8q .. 8q + 7) at chunk q ^ (r % 8). Within one 8-thread
// phase of a 16-byte store, rows r .. r + 7 hit 8 distinct chunks: no bank
// conflicts. Then the fence that hands generic-proxy writes to the MMA.
template <class T>
__device__ __forceinline__ void write_p_row(T *sp, int r,
                                            const float (&s)[kBN]) {
#pragma unroll
  for (int pn = 0; pn < kBN / kPanel; ++pn)
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float *x = s + pn * kPanel + q * 8;
      uint4 v;
      v.x = pack2<T>(x[0], x[1]);
      v.y = pack2<T>(x[2], x[3]);
      v.z = pack2<T>(x[4], x[5]);
      v.w = pack2<T>(x[6], x[7]);
      *reinterpret_cast<uint4 *>(sp + pn * kBM * kPanel + r * kPanel +
                                 (q ^ (r % 8)) * 8) = v;
    }
  lab::fence_proxy_async();
}

// O row *= alpha in TMEM: 32 columns at a time, ld -> scale -> st, and wait
// until the stores are complete. Synchronous and unconditional: FA4 moves
// this to a separate correction warpgroup and skips it when the max barely
// changes (labs 40-41).
template <class C>
__device__ __forceinline__ void rescale_o_row(uint32_t o_addr, float alpha) {
#pragma unroll
  for (int c = 0; c < C::kD; c += 32) {
    float o[32];
    lab::tmem_ld_32x32b_x32_sync(o_addr + c, o);
#pragma unroll
    for (int i = 0; i < 32; ++i)
      o[i] *= alpha;
    lab::tmem_st_32x32b_x32(o_addr + c, o);
  }
  lab::tmem_wait_st();
}

// Epilogue for row r: O / l staged in the (now unused) Q tile in the same
// swizzled layout, then lane 0 TMA-stores this warp's 32 rows (box
// [32][64] per panel; rows >= N are skipped by TMA), and the thread writes
// its LSE = scale * m + ln(l).
template <class T, class C>
__device__ __forceinline__ void
store_o_and_lse(const CUtensorMap *o_map, T *sq, uint32_t o_addr, int r,
                int head, int batch, int q0, float m, float l,
                const Params &p) {
  const int warp = r / 32, lane = r % 32, row = q0 + r;
  const float inv = 1.f / l; // l >= 1: the max element contributes 2^0
#pragma unroll
  for (int c = 0; c < C::kD; c += 32) {
    float o[32];
    lab::tmem_ld_32x32b_x32_sync(o_addr + c, o);
    T *dst = sq + (c / kPanel) * kBM * kPanel + r * kPanel;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int chunk = (c % kPanel) / 8 + q;
      uint4 v;
      v.x = pack2<T>(o[8 * q + 0] * inv, o[8 * q + 1] * inv);
      v.y = pack2<T>(o[8 * q + 2] * inv, o[8 * q + 3] * inv);
      v.z = pack2<T>(o[8 * q + 4] * inv, o[8 * q + 5] * inv);
      v.w = pack2<T>(o[8 * q + 6] * inv, o[8 * q + 7] * inv);
      *reinterpret_cast<uint4 *>(dst + (chunk ^ (r % 8)) * 8) = v;
    }
  }
  lab::fence_proxy_async(); // generic O writes -> visible to the TMA store
  __syncwarp();
  if (lane == 0) {
    for (int pn = 0; pn < C::kPanels; ++pn)
      lab::tma_store_4d(o_map, sq + pn * kBM * kPanel + warp * 32 * kPanel,
                        pn * kPanel, head, q0 + warp * 32, batch);
    lab::bulk_commit_group();
    lab::bulk_wait_group_read<0>(); // smem read before the CTA may exit
  }
  if (row < p.n)
    p.lse[(static_cast<int64_t>(batch) * p.h + head) * p.n + row] =
        m * p.scale + logf(l);
}

// ===========================================================================
// The kernel.
// ===========================================================================
template <class T, class C, bool kCausal, bool kProbe>
__global__ void __launch_bounds__(kThreads, 1)
    fa3_forward(const __grid_constant__ CUtensorMap q_map,
                const __grid_constant__ CUtensorMap k_map,
                const __grid_constant__ CUtensorMap v_map,
                const __grid_constant__ CUtensorMap o_map, Params p) {
  constexpr int S = C::kStages;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem =
      smem_raw + ((1024 - (lab::shared_addr(smem_raw) & 1023)) & 1023);
  T *sq = reinterpret_cast<T *>(smem + C::kQOff); // Q, later O staging
  T *sp = reinterpret_cast<T *>(smem + C::kPOff); // P [2][128][64]
  auto sk = [&](int s) {
    return reinterpret_cast<T *>(smem + C::kKOff + s * C::kKVBytes);
  };
  auto sv = [&](int s) {
    return reinterpret_cast<T *>(smem + C::kVOff + s * C::kKVBytes);
  };
  __shared__ Barriers<S> bar;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int q0 = blockIdx.x * kBM, head = blockIdx.y, batch = blockIdx.z;
  const int n = p.n;
  const int blocks = num_kv_blocks<kCausal>(q0, n);

  // ---- setup: barriers (thread 0), TMEM (warp 0), then one CTA barrier.
  if (threadIdx.x == 0) {
    lab::mbar_init(&bar.q_full, 1);
    for (int s = 0; s < S; ++s) {
      lab::mbar_init(&bar.k_full[s], 1);
      lab::mbar_init(&bar.v_full[s], 1);
      lab::mbar_init(&bar.k_empty[s], 1);
      lab::mbar_init(&bar.v_empty[s], 1);
    }
    lab::mbar_init(&bar.s_full[0], 1);
    lab::mbar_init(&bar.s_full[1], 1);
    lab::mbar_init(&bar.p_ready, kBM);
    lab::mbar_init(&bar.o_ready, 1);
    lab::fence_barrier_init();
  }
  if (warp == 0) {
    lab::tmem_alloc(&bar.tmem_slot, C::kTmemCols);
    lab::tmem_relinquish_alloc_permit();
  }
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  lab::tcgen05_fence_after_thread_sync();
  const uint32_t tmem = bar.tmem_slot;

  if (warp == kLoadWarp) {
    // ===================================================== TMA load warp
    if (lane == 0) {
      lab::prefetch_tensor_map(&q_map);
      lab::prefetch_tensor_map(&k_map);
      lab::prefetch_tensor_map(&v_map);
      // Q: kPanels boxes of [128 rows][64]; rows >= N are zero-filled and
      // still count toward the transaction bytes.
      lab::mbar_arrive_expect_tx(&bar.q_full, C::kQBytes);
      for (int pn = 0; pn < C::kPanels; ++pn)
        lab::tma_load_4d(sq + pn * kBM * kPanel, &q_map, &bar.q_full,
                         pn * kPanel, head, q0, batch);
      // K_j and V_j share stage j % S but have separate barriers: the MMA
      // warp needs K_j for S_j long before it needs V_j for P_j V_j, and
      // releases them at different times.
      for (int j = 0; j < blocks; ++j) {
        const int s = j % S;
        const uint32_t phase = (j / S) & 1;
        // Wait until the MMAs of block j - S released the stage. For
        // j < S the "previous phase" of a fresh barrier counts as complete.
        lab::mbar_wait(&bar.k_empty[s], phase ^ 1);
        lab::mbar_arrive_expect_tx(&bar.k_full[s], C::kKVBytes);
        for (int pn = 0; pn < C::kPanels; ++pn)
          lab::tma_load_4d(sk(s) + pn * kBN * kPanel, &k_map, &bar.k_full[s],
                           pn * kPanel, head, j * kBN, batch);
        if constexpr (!kProbe) {
          lab::mbar_wait(&bar.v_empty[s], phase ^ 1);
          lab::mbar_arrive_expect_tx(&bar.v_full[s], C::kKVBytes);
          for (int pn = 0; pn < C::kPanels; ++pn)
            lab::tma_load_4d(sv(s) + pn * kBN * kPanel, &v_map, &bar.v_full[s],
                             pn * kPanel, head, j * kBN, batch);
        }
      }
      // Drain: wait for the last release of every stage, so that no
      // tcgen05.commit arrival on our barriers is still pending at exit.
      for (int j = blocks; j < blocks + S; ++j) {
        lab::mbar_wait(&bar.k_empty[j % S], ((j / S) & 1) ^ 1);
        if constexpr (!kProbe)
          lab::mbar_wait(&bar.v_empty[j % S], ((j / S) & 1) ^ 1);
      }
    }
  } else if (warp == kMmaWarp) {
    // ======================================================== MMA warp
    if (lane == 0) {
      lab::mbar_wait(&bar.q_full, 0);
      auto issue_qk = [&](int i) { // S_i -> TMEM buffer s_buffer(i)
        const int s = i % S, buf = s_buffer<C::kSBuf>(i);
        lab::mbar_wait(&bar.k_full[s], (i / S) & 1);
        lab::tcgen05_fence_after_thread_sync();
        mma_qk<T, C>(tmem + buf * kBN, sq, sk(s));
        lab::tcgen05_commit(&bar.k_empty[s]);  // K_i may be overwritten
        lab::tcgen05_commit(&bar.s_full[buf]); // S_i is ready
      };
      issue_qk(0);
      if constexpr (!kProbe) {
        if constexpr (C::kSBuf == 2) {
          // Step 5: S_1 goes into the second buffer right away, so the
          // Tensor Core works on it while the softmax warps handle S_0.
          if (blocks > 1)
            issue_qk(1);
        }
        for (int j = 0; j < blocks; ++j) {
          const int s = j % S;
          // P_j written, S_j read, O rescaled by all 128 softmax threads.
          lab::mbar_wait(&bar.p_ready, j & 1);
          lab::mbar_wait(&bar.v_full[s], (j / S) & 1);
          lab::tcgen05_fence_after_thread_sync();
          mma_pv<T, C>(tmem + C::kOCol, sp, sv(s), j > 0);
          lab::tcgen05_commit(&bar.v_empty[s]); // V_j may be overwritten
          lab::tcgen05_commit(&bar.o_ready);    // O has block j; P is free
          // The S buffer of block j was read by the softmax warps (p_ready):
          // reuse it for the next block that maps to it. P_j V_j was issued
          // first, so the softmax warps waiting for o_ready are not delayed
          // by this QK.
          if constexpr (C::kSBuf == 1) {
            if (j + 1 < blocks)
              issue_qk(j + 1);
          } else {
            if (j + 2 < blocks)
              issue_qk(j + 2);
          }
        }
      }
    }
  } else {
    // =================================================== softmax warpgroup
    const int r = threadIdx.x; // row in the tile = TMEM lane
    const int row = q0 + r;
    const uint32_t lane_addr = tmem + (static_cast<uint32_t>(warp * 32) << 16);
    float m = -INFINITY; // running max of the raw scores S (not scaled)
    float l = 0.f;       // running sum of exp(scale * (S - m))
    for (int j = 0; j < blocks; ++j) {
      const int buf = s_buffer<C::kSBuf>(j);
      lab::mbar_wait(&bar.s_full[buf], s_phase<C::kSBuf>(j));
      lab::tcgen05_fence_after_thread_sync();
      float s[kBN];
      load_s_row(lane_addr, buf * kBN, s);

      if constexpr (kProbe) {
        // Step 1 (given): scale * S of the single block to the probe.
        if (row < n) {
          float *dst =
              p.probe +
              ((static_cast<int64_t>(batch) * p.h + head) * n + row) * n;
#pragma unroll
          for (int c = 0; c < kBN; ++c)
            if (c < n)
              dst[c] = s[c] * p.scale;
        }
        // (The teardown fence orders these tcgen05.ld before dealloc.)
        break;
      } else {
        const int k0 = j * kBN;
        if (block_needs_mask<kCausal>(k0, q0, n)) {
          const int limit = visible_keys<kCausal>(row, k0, n);
#pragma unroll
          for (int c = 0; c < kBN; ++c)
            if (c >= limit)
              s[c] = -INFINITY;
        }
        float m_new, base;
        const float row_sum = softmax_block(s, m, p.scale_log2, m_new, base);
        // Correction factor for the old state. Block 0: there is no old
        // state (l = 0, P_0 V_0 overwrites O), so alpha = 0 is harmless.
        float alpha = 0.f;
        if (j > 0)
          alpha = exp2_approx(m * p.scale_log2 - base);
        l = l * alpha + row_sum;
        m = m_new;

        // P_{j-1} V_{j-1} must be complete before P is overwritten and O is
        // rescaled. Everything above overlapped with it.
        if (j > 0) {
          lab::mbar_wait(&bar.o_ready, (j - 1) & 1);
          lab::tcgen05_fence_after_thread_sync();
        }
        write_p_row<T>(sp, r, s);
        if (j > 0)
          rescale_o_row<C>(lane_addr + C::kOCol, alpha);
        // Order this thread's tcgen05.ld (S, O) and tcgen05.st (O) before
        // the MMA warp's next tcgen05.mma, then hand P_j over.
        lab::tcgen05_fence_before_thread_sync();
        lab::mbar_arrive(&bar.p_ready);
      }
    }
    if constexpr (!kProbe) {
      // Last P V complete. o_ready tracks every earlier tcgen05 op of the
      // MMA thread, so nobody reads Q any more: it becomes the O staging.
      lab::mbar_wait(&bar.o_ready, (blocks - 1) & 1);
      lab::tcgen05_fence_after_thread_sync();
      store_o_and_lse<T, C>(&o_map, sq, lane_addr + C::kOCol, r, head, batch,
                            q0, m, l, p);
    }
  }

  // ---- teardown: all TMEM accesses ordered before warp 0 frees it.
  lab::tcgen05_fence_before_thread_sync();
  __syncthreads();
  if (warp == 0) {
    lab::tcgen05_fence_after_thread_sync();
    lab::tmem_dealloc(tmem, C::kTmemCols);
  }
}

// ===========================================================================
// Host side.
// ===========================================================================

// [B, N, H, D] as a 4-D tensor map (d fastest): box [1][rows][1][64].
CUtensorMap bnhd_map(const void *base, const lab::AttentionProblem &p,
                     uint32_t box_rows) {
  const CUtensorMapDataType dt = p.dtype == lab::DType::F16
                                     ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                     : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  return lab::make_tensor_map_4d(base, dt, p.b, p.n, p.h, p.d, 1, box_rows, 1,
                                 kPanel, CU_TENSOR_MAP_SWIZZLE_128B);
}

struct Maps {
  CUtensorMap q, k, v, o;
};

template <class T, int D, int SBuf, bool kCausal, bool kProbe>
void launch(const Maps &maps, const lab::AttentionProblem &pr,
            cudaStream_t stream, bool setup) {
  using C = Config<D, SBuf>;
  auto kernel = fa3_forward<T, C, kCausal, kProbe>;
  if (setup) {
    LAB_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    return;
  }
  Params prm{pr.n,   pr.h,    pr.scale, pr.scale * 1.4426950408889634f,
             pr.lse, pr.probe};
  const dim3 grid((pr.n + kBM - 1) / kBM, pr.h, pr.b);
  kernel<<<grid, kThreads, C::kSmemBytes, stream>>>(maps.q, maps.k, maps.v,
                                                    maps.o, prm);
  LAB_CUDA_CHECK(cudaGetLastError());
}

template <class T, int D>
void dispatch_step(const Maps &m, const lab::AttentionProblem &p,
                   cudaStream_t s, bool setup) {
  if (p.step == 1)
    launch<T, D, 1, false, true>(m, p, s, setup);
  else if (p.step <= 4)
    p.causal ? launch<T, D, 1, true, false>(m, p, s, setup)
             : launch<T, D, 1, false, false>(m, p, s, setup);
  else
    p.causal ? launch<T, D, 2, true, false>(m, p, s, setup)
             : launch<T, D, 2, false, false>(m, p, s, setup);
}

void dispatch(const Maps &m, const lab::AttentionProblem &p, cudaStream_t s,
              bool setup) {
  const bool half = p.dtype == lab::DType::F16;
  if (p.d == 64)
    half ? dispatch_step<fp16, 64>(m, p, s, setup)
         : dispatch_step<bf16, 64>(m, p, s, setup);
  else
    half ? dispatch_step<fp16, 128>(m, p, s, setup)
         : dispatch_step<bf16, 128>(m, p, s, setup);
}

} // namespace

struct Plan {
  Maps maps;
};

Plan make_plan(const lab::AttentionProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F16 || p.dtype == lab::DType::BF16,
              "this lab uses FP16 or BF16 Q/K/V");
  LAB_REQUIRE(p.o_dtype == p.dtype, "O has the dtype of Q/K/V");
  LAB_REQUIRE(p.d == 64 || p.d == 128, "D must be 64 or 128");
  LAB_REQUIRE(p.n > 0 && p.b > 0 && p.h > 0, "empty problem");
  LAB_REQUIRE(p.step > 2 || p.n <= kBN, "steps 1-2 handle one K/V block");
  LAB_REQUIRE(p.step != 1 || p.probe != nullptr, "step 1 writes the probe");
  Plan plan{};
  plan.maps.q = bnhd_map(p.q, p, kBM);
  plan.maps.k = bnhd_map(p.k, p, kBN);
  plan.maps.v = bnhd_map(p.v, p, kBN);
  plan.maps.o = bnhd_map(p.o, p, 32); // one box per softmax warp
  dispatch(plan.maps, p, nullptr, true);
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  dispatch(plan.maps, p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
