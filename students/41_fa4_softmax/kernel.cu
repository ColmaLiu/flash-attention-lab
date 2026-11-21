// 41_fa4_softmax: softmax throughput in the FA4 pipeline of lab 40 -- a
// software exp2 on the FMA pipe, a MUFU/FMA mix, conditional rescaling.
// Lesson: course/41_fa4_softmax/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous: element (b, i, h, d) at
//   ((b * N + i) * H + h) * D + d. LSE: [B, H, N] FP32, natural log.
// Supported: D in {64, 128}, scale > 0 finite, causal or not.
//
// The pipeline is lab 40's step 3 and is given complete: one CTA per (pair
// of Q tiles = 256 query rows, head, batch); 16 warps:
//   warps 0-3   softmax warpgroup 0 (tile 0): thread r owns row r = TMEM lane r
//   warps 4-7   softmax warpgroup 1 (tile 1)
//   warps 8-11  correction warpgroup (O rescale, epilogue)
//   warp 12     MMA, warp 13 TMA, warps 14-15 idle
//
// run() picks a version by p.step (one kernel, three configurations):
//   step 1  Config<D, 1>  every exponential of P with exp2_poly2 (TODO 1)
//   step 2  Config<D, 2>  kPolyPairs of 16 pairs polynomial, rest MUFU (TODO 2)
//   step 3  Config<D, 3>  conditional rescale with kTau = 8 (TODO 3a, 3b)
// The steps build on each other: step 2 uses TODO(1), step 3 uses 1 and 2.
//
// New CUDA intrinsics (sm_100, <cuda_runtime.h>): float2 __ffma2_rn(a, b, c),
// __fadd2_rn(a, b), __fadd2_rd(a, b) (round down) -- two FP32 operations in
// one FFMA2 / FADD2 instruction (lesson section 4.2).
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

constexpr int kBM = 128;   // query rows per tile = MMA M = TMEM lanes
constexpr int kBN = 128;   // keys per K/V block = N of S = K of P.V
constexpr int kTiles = 2;  // Q tiles per CTA
constexpr int kPanel = 64; // 64 16-bit elements = one 128-byte swizzle row
constexpr int kUmmaK = 16; // K of one tcgen05.mma for 16-bit inputs
constexpr int kPCol = 64;  // P^t (128 x 16-bit = 64 columns) at S^t + 64

// ===========================================================================
// Exponentials (lesson sections 3-4).
// ===========================================================================

// 2^x with MUFU.EX2 (ex2.approx.ftz.f32; 2^-inf = +0). Max relative error
// measured over every FP32 input in [-126, 8]: 1.5e-7.
__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Minimax coefficients of 2^f on [0, 1) for the relative error, with
// c0 = 1 exactly (lesson section 3.3). FP32 Horner evaluation, max relative
// error measured over every FP32 f in [0, 1): 8.57e-5 = 2^-13.5.
constexpr float kExp2C0 = 1.0f;
constexpr float kExp2C1 = 0.6951168179512024f;  // 0x3F31F32D
constexpr float kExp2C2 = 0.22764484584331512f; // 0x3E691BBB
constexpr float kExp2C3 = 0.07706717401742935f; // 0x3D9DD565

// 2^x for two values at once, on the FMA pipe only (no MUFU):
//   x = max(x, -127)                         (2^-inf and tiny x -> +0)
//   r = x + 1.5 * 2^23, rounded DOWN         the low mantissa bits of r hold
//                                            the integer n = floor(x)
//   f = x - (r - 1.5 * 2^23) in [0, 1)       exact
//   p = ((c3 f + c2) f + c1) f + c0 in [1, 2)
//   2^x = 2^n * p: add n to p's exponent field (bits 23-30)
// Valid for x <= 127 (here x <= kTau = 8). x = -127 gives exactly +0
// because p(0) = c0 = 1 has the bit pattern 127 << 23. Two lanes per
// instruction: FADD2 / FFMA2 do two FP32 operations each (lesson 4.2).
__device__ __forceinline__ float2 exp2_poly2(float2 x) {
  // TODO(1) -- lesson sections 3.2-3.4 and 4.2-4.3. For each of x.x, x.y:
  //  1. clamp: x = max(x, -127) (masked keys arrive as -inf);
  //  2. r = x + 1.5 * 2^23 rounded DOWN (__fadd2_rd): the low mantissa
  //     bits of r now hold n = floor(x) as an integer;
  //  3. n = r - 1.5 * 2^23 (exact), f = x - n in [0, 1] (one FFMA2: n * -1 + x);
  //  4. p = ((c3 f + c2) f + c1) f + c0 (Horner, __ffma2_rn, kExp2C*);
  //  5. add n to p's exponent field (bits 23-30) with integer arithmetic
  //     on the bit patterns (__float_as_int / __int_as_float); n is
  //     already in the low bits of r (lesson section 3.2).
  // Why must c0 be exactly 1? (Try x = -inf.)
  LAB_TODO(1);
  return x;
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
// Configuration: warp roles, shared memory, TMEM (lab 40, step 3).
//
// Shared tiles are what TMA writes with a 64-element box row and the 128B
// swizzle: [D/64 panels][rows][64], 16-byte chunk q of row r at q ^ (r % 8).
// ===========================================================================

// Per-row statistics handed from the softmax to the correction warpgroup,
// in shared memory after the K/V ring.
struct Stats {
  float alpha[kTiles][kBM]; // alpha^t_j of the current block
  float fin_m[kTiles][kBM], fin_l[kTiles][kBM]; // final m and l
};

template <int D_, int Step> struct Config {
  static constexpr int kD = D_;
  static constexpr int kPanels = D_ / kPanel;

  // Softmax arithmetic (the subject of this lab). Of every 16 pairs of
  // columns (32 columns = one tcgen05.ld chunk), kPolyPairs pairs use
  // exp2_poly2 and the rest MUFU.EX2. Step 1: all 16. Steps 2-3: the
  // balance point measured in lesson section 7 (D = 64 has half the MMA
  // work per block, so the softmax is relatively more expensive there,
  // but the same split was best for both).
  // TODO(2): choose the step 2-3 value from the balance point (lesson
  // section 3.6), then tune it with ./lab bench (section 7).
  static constexpr int kPolyPairs = Step == 1 ? 16 : 2;
  static constexpr bool kLazy = Step >= 3; // conditional rescale
  static constexpr float kTau = 8.f;       // log2(256)

  // Warp roles.
  static constexpr int kMmaWarp = 12;
  static constexpr int kLoadWarp = kMmaWarp + 1;
  static constexpr int kThreads = 512;

  // Register budget. 64K registers / 512 threads = 128 at launch; the four
  // warpgroups must add up to 4 * 128 = 512.
  static constexpr int kRegSoftmax = 192, kRegCorrection = 80, kRegOther = 48;
  static_assert(2 * kRegSoftmax + kRegCorrection + kRegOther == 512,
                "register pool: 64K registers per SM");

  // One ring of shared-memory stages for K_0, V_0, K_1, V_1, ... (item u:
  // K_{u/2} if u is even, V_{u/2} if odd).
  static constexpr int kQBytes = kBM * D_ * 2;  // one Q tile
  static constexpr int kKVBytes = kBN * D_ * 2; // one K or V block
  static constexpr int kStages = D_ == 128 ? 4 : 8;
  static constexpr int kQOff = 0;
  static constexpr int kRingOff = kQOff + kTiles * kQBytes;
  static constexpr int kStatsOff = kRingOff + kStages * kKVBytes;
  static constexpr int kSmemBytes = 1024 + kStatsOff + int(sizeof(Stats));
  static_assert(kSmemBytes + 256 <= 227 * 1024, "shared memory per CTA");

  // TMEM (columns of 128 lanes x 32 bit):
  //   S^t: columns 128 t .. 128 t + 127 (FP32 scores of tile t)
  //   P^t: columns 128 t + 64 .. 128 t + 127 (16-bit pairs, over S^t)
  //   O^t: columns 256 + D t .. 256 + D t + D - 1 (FP32)
  static constexpr int kOCol = kTiles * kBN;
  static constexpr int kTmemCols = 512;
  static_assert(kOCol + kTiles * D_ <= kTmemCols, "TMEM has 512 columns");
};

// Every mbarrier of the kernel (lab 40). "count" = arrivals per phase.
template <int Stages> struct Barriers {
  // TMA -> MMA. count 1 (arrive_expect_tx by the TMA lane) + tx bytes.
  uint64_t q_full[kTiles];  // Q^t landed (one phase)
  uint64_t kv_full[Stages]; // ring item u landed in stage u % Stages
  // MMA -> TMA. count 1 (tcgen05.commit): the MMAs that read the stage are
  // complete. K_j is released after Q^1 K_j^T, V_j after P^1_j V_j.
  uint64_t kv_empty[Stages];
  // MMA -> softmax t. count 1 (commit after Q^t K_j^T): S^t_j is in TMEM,
  // and (commit tracks every earlier MMA of the thread) P^t_{j-1} V_{j-1}
  // has completed. Phase j, parity j & 1.
  uint64_t s_full[kTiles];
  // softmax t + correction -> MMA. count 256. Phase j, parity j & 1:
  // P^t_j written, S^t_j read, O^t rescaled (or the rescale skipped).
  uint64_t p_ready[kTiles];
  // MMA -> epilogue. count 1 (commit after the last P^t V): O^t is final.
  uint64_t o_full[kTiles];
  // softmax t -> correction. count 128. Phase j: alpha^t_j is in
  // st.alpha[t][].
  uint64_t stats_full[kTiles];
  // softmax t -> correction. count 128, one phase: final m and l.
  uint64_t final_full[kTiles];
  uint32_t tmem_slot; // TMEM base address written by tcgen05.alloc
};

struct Params {
  int n, h;
  float scale, scale_log2; // scale_log2 = scale * log2(e)
  float *lse;
};

// K/V blocks of the pair of tiles starting at query row q0 (lab 40).
template <bool kCausal>
__device__ __forceinline__ int num_kv_blocks(int q0, int n) {
  int blocks = (n + kBN - 1) / kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + kTiles * kBM + kBN - 1) / kBN);
  return blocks;
}

// --------------------------------------------------------------------- MMA
// S (TMEM, 128 x 128 FP32) = Q[128 x D] . K_j[128 x D]^T (lab 34).
template <class T, class C>
__device__ __forceinline__ void mma_qk(uint32_t s_tmem, const T *sq,
                                       const T *sk) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kBM, kBN, std::is_same_v<T, bf16>);
  const uint64_t da = lab::make_smem_desc_kmajor_sw128(sq);
  const uint64_t db = lab::make_smem_desc_kmajor_sw128(sk);
#pragma unroll
  for (int kk = 0; kk < C::kD / kUmmaK; ++kk) {
    const int panel = kk * kUmmaK / kPanel, col = kk * kUmmaK % kPanel;
    const uint32_t a_off = (panel * kBM * kPanel + col) * 2 / 16;
    const uint32_t b_off = (panel * kBN * kPanel + col) * 2 / 16;
    lab::mma_f16_ss(s_tmem, da + a_off, db + b_off, idesc, kk > 0);
  }
}

// O (TMEM) [+]= P . V_j with P in TMEM (TS form; lab 40).
template <class T, class C>
__device__ __forceinline__ void mma_pv_ts(uint32_t o_tmem, uint32_t p_tmem,
                                          const T *sv, bool accumulate) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kBM, C::kD, std::is_same_v<T, bf16>, true,
                          /*a_mn_major=*/false, /*b_mn_major=*/true);
  const uint64_t db = lab::make_smem_desc_mnmajor_sw128(sv, kBN * kPanel * 2);
#pragma unroll
  for (int kk = 0; kk < kBN / kUmmaK; ++kk)
    lab::mma_f16_ts(o_tmem, p_tmem + kk * (kUmmaK / 2),
                    db + kk * kUmmaK * kPanel * 2 / 16, idesc,
                    accumulate || kk > 0);
}

// ===========================================================================
// Softmax-warpgroup helpers. Thread r of a softmax warpgroup owns row r of
// its tile = TMEM lane r.
// ===========================================================================
__device__ __forceinline__ void load_s_row(uint32_t lane_addr, int col,
                                           float (&s)[kBN]) {
#pragma unroll
  for (int c = 0; c < kBN; c += 32)
    lab::tmem_ld_32x32b_x32_sync(lane_addr + col + c,
                                 *reinterpret_cast<float (*)[32]>(s + c));
}

template <bool kCausal>
__device__ __forceinline__ bool block_needs_mask(int k0, int q0, int n) {
  if (k0 + kBN > n)
    return true;
  if constexpr (kCausal)
    return k0 + kBN - 1 > q0;
  return false;
}

template <bool kCausal>
__device__ __forceinline__ int visible_keys(int row, int k0, int n) {
  int end = n;
  if constexpr (kCausal)
    end = min(end, row + 1);
  return end - k0;
}

// max_c s[c] with eight independent partial maxima.
__device__ __forceinline__ float row_max(const float (&s)[kBN]) {
  float mx[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    mx[i] = s[i];
#pragma unroll
  for (int c = 8; c < kBN; ++c)
    mx[c % 8] = fmaxf(mx[c % 8], s[c]);
  return fmaxf(fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3])),
               fmaxf(fmaxf(mx[4], mx[5]), fmaxf(mx[6], mx[7])));
}

// Does pair q (columns 2q, 2q + 1 of a 32-column chunk, q = 0..15) use the
// polynomial? kPolyPairs of the 16 pairs, spread evenly over the chunk so
// that MUFU and FMA instructions interleave in the instruction stream.
template <int kPolyPairs> __device__ __forceinline__ bool use_poly(int q) {
  if constexpr (kPolyPairs == 16)
    return true; // step 1: everything on the FMA pipe
  // TODO(2) -- lesson sections 3.6 and 5.2. Return true for exactly
  // kPolyPairs of the 16 values q = 0..15, spread over the chunk (not all
  // at one end). The expression must be a compile-time function of q: the
  // loop in exp_and_sum is fully unrolled, so no branch remains in the SASS.
  (void)q;
  LAB_TODO(2);
  return false;
}

// s[c] = 2^(s[c] * scale_log2 - base); returns sum_c s[c]. Two columns at
// a time: one FFMA2 for the argument, then either exp2_poly2 or two
// MUFU.EX2, and one FADD2 into one of four packed partial sums.
template <int kPolyPairs>
__device__ __forceinline__ float exp_and_sum(float (&s)[kBN], float scale_log2,
                                             float base) {
  const float2 mul = make_float2(scale_log2, scale_log2);
  const float2 sub = make_float2(-base, -base);
  float2 sum[4] = {};
#pragma unroll
  for (int c = 0; c < kBN; c += 2) {
    float2 x = __ffma2_rn(make_float2(s[c], s[c + 1]), mul, sub);
    if constexpr (kPolyPairs > 0) {
      if (use_poly<kPolyPairs>((c % 32) / 2)) {
        x = exp2_poly2(x);
      } else {
        x.x = exp2_approx(x.x);
        x.y = exp2_approx(x.y);
      }
    } else {
      x.x = exp2_approx(x.x);
      x.y = exp2_approx(x.y);
    }
    s[c] = x.x;
    s[c + 1] = x.y;
    sum[(c / 2) % 4] = __fadd2_rn(sum[(c / 2) % 4], x);
  }
  const float2 t = __fadd2_rn(__fadd2_rn(sum[0], sum[1]),
                              __fadd2_rn(sum[2], sum[3]));
  return t.x + t.y;
}

// Row of P -> TMEM over the upper half of its own S columns (lab 40).
template <class T>
__device__ __forceinline__ void write_p_tmem(uint32_t p_addr,
                                             const float (&s)[kBN]) {
#pragma unroll
  for (int h = 0; h < kBN / 32; ++h) {
    uint32_t w[16];
#pragma unroll
    for (int c = 0; c < 16; ++c)
      w[c] = pack2<T>(s[32 * h + 2 * c], s[32 * h + 2 * c + 1]);
    lab::tmem_st_32x32b_x16(p_addr + 16 * h, w);
  }
  lab::tmem_wait_st();
}

// O row *= alpha in TMEM (lab 34): ld -> scale -> st, then wait::st.
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

// Epilogue for row r of the tile starting at q0 (lab 34): O / l staged in
// the tile's Q buffer, each warp TMA-stores its 32 rows, the thread writes
// LSE = scale * m + ln(l). m is the base the exponentials used (with the
// conditional rescale it may be below the true row max; LSE is still exact,
// lesson section 3.5).
template <class T, class C>
__device__ __forceinline__ void
store_o_and_lse(const CUtensorMap *o_map, T *sq, uint32_t o_addr, int r,
                int head, int batch, int q0, float m, float l,
                const Params &p) {
  const int warp = r / 32, lane = r % 32, row = q0 + r;
  const float inv = 1.f / l; // l >= 1: the max element contributes >= 2^0
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
template <class T, class C, bool kCausal>
__global__ void __launch_bounds__(C::kThreads, 1)
    fa4_forward(const __grid_constant__ CUtensorMap q_map,
                const __grid_constant__ CUtensorMap k_map,
                const __grid_constant__ CUtensorMap v_map,
                const __grid_constant__ CUtensorMap o_map, Params p) {
  constexpr int S = C::kStages;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem =
      smem_raw + ((1024 - (lab::shared_addr(smem_raw) & 1023)) & 1023);
  auto sq = [&](int t) { // Q^t, later the O^t staging area
    return reinterpret_cast<T *>(smem + C::kQOff + t * C::kQBytes);
  };
  auto ring = [&](int s) {
    return reinterpret_cast<T *>(smem + C::kRingOff + s * C::kKVBytes);
  };
  Stats &st = *reinterpret_cast<Stats *>(smem + C::kStatsOff);
  __shared__ Barriers<S> bar;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int q0 = blockIdx.x * (kTiles * kBM), head = blockIdx.y,
            batch = blockIdx.z;
  const int n = p.n;
  const int blocks = num_kv_blocks<kCausal>(q0, n);

  // ---- setup: barriers (thread 0), TMEM (warp 0), then one CTA barrier.
  if (threadIdx.x == 0) {
    for (int t = 0; t < kTiles; ++t) {
      lab::mbar_init(&bar.q_full[t], 1);
      lab::mbar_init(&bar.s_full[t], 1);
      lab::mbar_init(&bar.p_ready[t], 2 * kBM);
      lab::mbar_init(&bar.o_full[t], 1);
      lab::mbar_init(&bar.stats_full[t], kBM);
      lab::mbar_init(&bar.final_full[t], kBM);
    }
    for (int s = 0; s < S; ++s) {
      lab::mbar_init(&bar.kv_full[s], 1);
      lab::mbar_init(&bar.kv_empty[s], 1);
    }
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
  // This warp's TMEM lane slice (only meaningful for warpgroups 0-2).
  const uint32_t lane_addr =
      tmem + (static_cast<uint32_t>(32 * (warp % 4)) << 16);

  if (warp >= C::kMmaWarp) {
    // ============================================ MMA, TMA (and idle) warps
    lab::setmaxnreg_dec<C::kRegOther>();
    if (warp == C::kLoadWarp && lane == 0) {
      // ------------------------------------------------------ TMA load
      lab::prefetch_tensor_map(&q_map);
      lab::prefetch_tensor_map(&k_map);
      lab::prefetch_tensor_map(&v_map);
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_arrive_expect_tx(&bar.q_full[t], C::kQBytes);
        for (int pn = 0; pn < C::kPanels; ++pn)
          lab::tma_load_4d(sq(t) + pn * kBM * kPanel, &q_map, &bar.q_full[t],
                           pn * kPanel, head, q0 + t * kBM, batch);
      }
      for (int u = 0; u < 2 * blocks; ++u) {
        const int s = u % S;
        lab::mbar_wait(&bar.kv_empty[s], ((u / S) & 1) ^ 1);
        lab::mbar_arrive_expect_tx(&bar.kv_full[s], C::kKVBytes);
        const CUtensorMap *map = (u & 1) ? &v_map : &k_map;
        for (int pn = 0; pn < C::kPanels; ++pn)
          lab::tma_load_4d(ring(s) + pn * kBN * kPanel, map, &bar.kv_full[s],
                           pn * kPanel, head, (u / 2) * kBN, batch);
      }
      // Drain: the last release of every stage has arrived before exit.
      for (int u = 2 * blocks; u < 2 * blocks + S; ++u)
        lab::mbar_wait(&bar.kv_empty[u % S], ((u / S) & 1) ^ 1);
    } else if (warp == C::kMmaWarp) {
      // ----------------------------------------------------------- MMA
      const bool leader = lab::elect_one();
      auto acquire = [&](int u) { // ring item u has landed
        lab::mbar_wait(&bar.kv_full[u % S], (u / S) & 1);
        lab::tcgen05_fence_after_thread_sync();
        return ring(u % S);
      };
      auto s_col = [&](int t) { return tmem + t * kBN; };
      auto o_col = [&](int t) { return tmem + C::kOCol + t * C::kD; };

      // Prologue: S^0_0 = Q^0 K_0^T, S^1_0 = Q^1 K_0^T.
      const T *k0 = acquire(0);
#pragma unroll 1
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.q_full[t], 0);
        lab::tcgen05_fence_after_thread_sync();
        if (leader) {
          mma_qk<T, C>(s_col(t), sq(t), k0);
          lab::tcgen05_commit(&bar.s_full[t]);
        }
        __syncwarp();
      }
      if (leader)
        lab::tcgen05_commit(&bar.kv_empty[0]); // K_0 read by both tiles
      // Block j: PV^0_j QK^0_{j+1} PV^1_j QK^1_{j+1} (lab 40, section 5.3).
      for (int j = 0; j < blocks; ++j) {
        const bool more = j + 1 < blocks;
        const T *v = acquire(2 * j + 1);                  // V_j
        const T *k = more ? acquire(2 * j + 2) : nullptr; // K_{j+1}
#pragma unroll 1
        for (int t = 0; t < kTiles; ++t) {
          lab::mbar_wait(&bar.p_ready[t], j & 1);
          lab::tcgen05_fence_after_thread_sync();
          if (leader) {
            mma_pv_ts<T, C>(o_col(t), s_col(t) + kPCol, v, j > 0);
            if (t == kTiles - 1) // V_j read by both tiles
              lab::tcgen05_commit(&bar.kv_empty[(2 * j + 1) % S]);
            if (more) {
              // Overwrites S^t and the P^t that PV^t_j reads: the MMAs of
              // one thread execute in issue order (lab 40, section 4.3).
              mma_qk<T, C>(s_col(t), sq(t), k);
              lab::tcgen05_commit(&bar.s_full[t]);
            } else {
              lab::tcgen05_commit(&bar.o_full[t]); // O^t final
            }
          }
          __syncwarp();
        }
        if (more && leader) // K_{j+1} read by both tiles
          lab::tcgen05_commit(&bar.kv_empty[(2 * j + 2) % S]);
      }
    }
  } else if (warp < 4 * kTiles) {
    // ================================================ softmax warpgroups
    lab::setmaxnreg_inc<C::kRegSoftmax>();
    const int t = warp / 4;          // tile
    const int r = threadIdx.x % kBM; // row in the tile = TMEM lane
    const int qt = q0 + t * kBM;     // first row of the tile
    const int row = qt + r;
    const uint32_t s_addr = lane_addr + t * kBN;
    // m: the base of the exponentials, in raw-score units. Without the
    // conditional rescale it is the running max; with it (step 3) it may
    // lag the running max by up to kTau / scale_log2 (lesson section 3.5).
    float m = -INFINITY;
    float l = 0.f; // running sum of 2^(scale_log2 * (S - m))
    for (int j = 0; j < blocks; ++j) {
      // S^t_j is ready; P^t_{j-1} V_{j-1} is complete (see s_full).
      lab::mbar_wait(&bar.s_full[t], j & 1);
      lab::tcgen05_fence_after_thread_sync();
      float s[kBN];
      load_s_row(s_addr, 0, s);
      const int k0 = j * kBN;
      if (block_needs_mask<kCausal>(k0, qt, n)) {
        const int limit = visible_keys<kCausal>(row, k0, n);
#pragma unroll
        for (int c = 0; c < kBN; ++c)
          if (c >= limit)
            s[c] = -INFINITY;
      }
      const float m_new = fmaxf(m, row_max(s));
      float alpha;
      if constexpr (C::kLazy) {
        // Move the base only if the max grew by more than kTau (log2
        // units). Block 0: m = -inf, the difference is +inf, so it always
        // moves (alpha = 2^-inf = 0). (Both -inf: NaN > kTau is false.)
        // TODO(3a) -- lesson section 3.5: move the base m (and set alpha)
        // only when the max grew by more than C::kTau in log2 units;
        // otherwise alpha = 1. Compare with the current base m, not with
        // the previous block's max. Block 0 (m = -inf) must move.
        alpha = 1.f;
        LAB_TODO(3);
      } else {
        // Always move the base (labs 22-40). Block 0: alpha = 0.
        alpha = m_new == -INFINITY ? 1.f
                                   : exp2_approx((m - m_new) * p.scale_log2);
        m = m_new;
      }
      const float base = m == -INFINITY ? 0.f : m * p.scale_log2;
      // Hand alpha to the correction warpgroup now: it rescales O^t (if
      // needed) while this thread computes the exponentials.
      st.alpha[t][r] = alpha;
      lab::mbar_arrive(&bar.stats_full[t]);
      l = l * alpha + exp_and_sum<C::kPolyPairs>(s, p.scale_log2, base);
      write_p_tmem<T>(s_addr + kPCol, s);
      // Order this thread's tcgen05.ld/st before the MMA warp's next MMAs.
      lab::tcgen05_fence_before_thread_sync();
      lab::mbar_arrive(&bar.p_ready[t]);
    }
    st.fin_m[t][r] = m;
    st.fin_l[t][r] = l;
    lab::mbar_arrive(&bar.final_full[t]);
  } else {
    // ============================================== correction warpgroup
    lab::setmaxnreg_dec<C::kRegCorrection>();
    const int r = threadIdx.x % kBM; // row = TMEM lane, in both tiles
    for (int j = 0; j < blocks; ++j) {
      for (int t = 0; t < kTiles; ++t) {
        // alpha^t_j published; S^t_j was ready when softmax read it, so
        // P^t_{j-1} V_{j-1} has completed and O^t may be read.
        lab::mbar_wait(&bar.stats_full[t], j & 1);
        lab::tcgen05_fence_after_thread_sync();
        if (j > 0) {
          const float alpha = st.alpha[t][r];
          // tcgen05.ld/st are warp-collective (.sync.aligned): the whole
          // warp rescales its 32 rows or none. Rows with alpha = 1 are
          // multiplied by 1 (exact). Without kLazy alpha is almost never 1
          // for every row, so lab 40 rescaled unconditionally.
          if constexpr (C::kLazy) {
            // TODO(3b) -- lesson sections 4.4 and 5.3. Skip the rescale of
            // this warp's 32 rows of O^t when none of them needs it. The
            // decision must be the same for all 32 lanes: why?
            LAB_TODO(3);
          } else {
            rescale_o_row<C>(lane_addr + C::kOCol + t * C::kD, alpha);
          }
        }
        lab::tcgen05_fence_before_thread_sync();
        lab::mbar_arrive(&bar.p_ready[t]);
      }
    }
    for (int t = 0; t < kTiles; ++t) {
      lab::mbar_wait(&bar.final_full[t], 0);
      lab::mbar_wait(&bar.o_full[t], 0);
      lab::tcgen05_fence_after_thread_sync();
      store_o_and_lse<T, C>(&o_map, sq(t), lane_addr + C::kOCol + t * C::kD,
                            r, head, batch, q0 + t * kBM, st.fin_m[t][r],
                            st.fin_l[t][r], p);
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
// Host side (lab 40).
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

template <class T, int D, int Step, bool kCausal>
void launch(const Maps &maps, const lab::AttentionProblem &pr,
            cudaStream_t stream, bool setup) {
  using C = Config<D, Step>;
  auto kernel = fa4_forward<T, C, kCausal>;
  if (setup) {
    LAB_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    return;
  }
  Params prm{pr.n, pr.h, pr.scale, pr.scale * 1.4426950408889634f, pr.lse};
  const dim3 grid((pr.n + kTiles * kBM - 1) / (kTiles * kBM), pr.h, pr.b);
  kernel<<<grid, C::kThreads, C::kSmemBytes, stream>>>(maps.q, maps.k, maps.v,
                                                       maps.o, prm);
  LAB_CUDA_CHECK(cudaGetLastError());
}

template <class T, int D>
void dispatch_step(const Maps &m, const lab::AttentionProblem &p,
                   cudaStream_t s, bool setup) {
  if (p.step == 1)
    p.causal ? launch<T, D, 1, true>(m, p, s, setup)
             : launch<T, D, 1, false>(m, p, s, setup);
  else if (p.step == 2)
    p.causal ? launch<T, D, 2, true>(m, p, s, setup)
             : launch<T, D, 2, false>(m, p, s, setup);
  else
    p.causal ? launch<T, D, 3, true>(m, p, s, setup)
             : launch<T, D, 3, false>(m, p, s, setup);
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
  LAB_REQUIRE(p.step >= 1 && p.step <= 3, "steps 1-3");
  Plan plan{};
  plan.maps.q = bnhd_map(p.q, p, kBM);
  plan.maps.k = bnhd_map(p.k, p, kBN);
  plan.maps.v = bnhd_map(p.v, p, kBN);
  plan.maps.o = bnhd_map(p.o, p, 32); // one box per 32-row warp slice
  dispatch(plan.maps, p, nullptr, true);
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  dispatch(plan.maps, p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
