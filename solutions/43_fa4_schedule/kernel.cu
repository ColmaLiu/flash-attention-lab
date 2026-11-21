// Reference solution for 43_fa4_schedule: work scheduling for the FA4
// forward of labs 40-41 -- causal block classification with a mask-free
// main loop, persistent CTAs with a static tile schedule, and LPT
// (longest-processing-time-first) ordering for causal attention.
// Lesson: course/43_fa4_schedule/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous. LSE: [B, H, N] FP32 (ln).
// D in {64, 128}, scale > 0 finite.
//
// Work tile = (pair of Q tiles = 256 query rows, head, batch). Tile t (0, 1)
// of the pair holds rows q0 + 128 t .. q0 + 128 t + 127. The warp roles,
// TMEM layout and softmax arithmetic are lab 41's step 3:
//   warps 0-3   softmax warpgroup 0: thread r owns row r of tile 0
//   warps 4-7   softmax warpgroup 1: thread r owns row r of tile 1
//   warps 8-11  correction warpgroup: rescales O^0 / O^1 in TMEM, then the
//               epilogue (normalize, O straight to global memory, LSE)
//   warp 12     MMA (all 32 lanes wait, one elected lane issues)
//   warp 13     TMA (lane 0): K_0, Q^0, Q^1, V_0, K_1, V_1, ...
//   warps 14-15 idle (warpgroup 3 must be complete for setmaxnreg)
//
// step 1: one work tile per CTA (grid = number of tiles, as in lab 41). Each
//         (Q tile, K block) is classified: fully visible, needs a mask
//         (diagonal or past N), or fully masked (not visited). The softmax
//         loop runs the unmasked blocks first with no mask code at all, then
//         the masked tail.
// step 2: persistent: grid = min(#tiles, #SMs), CTA b processes tiles
//         b, b + G, b + 2G, ... Every role walks the same sequence; ring
//         positions and barrier phases run on across tiles. Q^t smem and
//         O^t TMEM get "empty" barriers so tile w + 1 can start while the
//         epilogue of tile w still runs.
// step 3: causal: tiles ordered heaviest first (LPT: last Q pair of every
//         head first) and dealt out boustrophedon (odd rounds reversed), so
//         every CTA gets about the same number of K/V blocks.
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
constexpr int kTiles = 2;  // Q tiles per work tile
constexpr int kPanel = 64; // 64 16-bit elements = one 128-byte swizzle row
constexpr int kUmmaK = 16; // K of one tcgen05.mma for 16-bit inputs
constexpr int kPCol = 64;  // P^t (128 x 16-bit = 64 columns) at S^t + 64

// ===========================================================================
// Exponentials (lab 41).
// ===========================================================================

__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

constexpr float kExp2C0 = 1.0f;
constexpr float kExp2C1 = 0.6951168179512024f;  // 0x3F31F32D
constexpr float kExp2C2 = 0.22764484584331512f; // 0x3E691BBB
constexpr float kExp2C3 = 0.07706717401742935f; // 0x3D9DD565

// 2^x for two values on the FMA pipe (lab 41, section 4.3).
__device__ __forceinline__ float2 exp2_poly2(float2 x) {
  const float2 magic = make_float2(12582912.f, 12582912.f); // 1.5 * 2^23
  x.x = fmaxf(x.x, -127.f);
  x.y = fmaxf(x.y, -127.f);
  const float2 r = __fadd2_rd(x, magic); // round down: floor(x) in the bits
  const float2 n = __fadd2_rn(r, make_float2(-12582912.f, -12582912.f));
  const float2 f = __ffma2_rn(n, make_float2(-1.f, -1.f), x); // x - n
  float2 p = __ffma2_rn(make_float2(kExp2C3, kExp2C3), f,
                        make_float2(kExp2C2, kExp2C2));
  p = __ffma2_rn(p, f, make_float2(kExp2C1, kExp2C1));
  p = __ffma2_rn(p, f, make_float2(kExp2C0, kExp2C0));
  return make_float2(
      __int_as_float(__float_as_int(p.x) + (__float_as_int(r.x) << 23)),
      __int_as_float(__float_as_int(p.y) + (__float_as_int(r.y) << 23)));
}

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
// Configuration.
// ===========================================================================

// Per-row statistics handed from the softmax to the correction warpgroup.
// The final m, l are double-buffered by tile parity: with persistent CTAs
// the softmax may finish tile w + 1 (if it has one K/V block) before the
// correction warpgroup has read tile w's values (lesson section 5.4).
struct Stats {
  float alpha[kTiles][kBM];                           // alpha^t_j
  float fin_m[2][kTiles][kBM], fin_l[2][kTiles][kBM]; // [tile parity]
};

template <int D_, int Step> struct Config {
  static constexpr int kD = D_;
  static constexpr int kPanels = D_ / kPanel;

  // Softmax arithmetic: lab 41 step 3.
  static constexpr int kPolyPairs = 3;
  static constexpr float kTau = 8.f;

  // Scheduling (this lab).
  static constexpr bool kPersistent = Step >= 2;
  static constexpr bool kLpt = Step >= 3;

  static constexpr int kMmaWarp = 12;
  static constexpr int kLoadWarp = kMmaWarp + 1;
  static constexpr int kThreads = 512;
  static constexpr int kRegSoftmax = 192, kRegCorrection = 80, kRegOther = 48;
  static_assert(2 * kRegSoftmax + kRegCorrection + kRegOther == 512,
                "register pool: 64K registers per SM");

  static constexpr int kQBytes = kBM * D_ * 2;
  static constexpr int kKVBytes = kBN * D_ * 2;
  static constexpr int kStages = D_ == 128 ? 4 : 8;
  static constexpr int kQOff = 0;
  static constexpr int kRingOff = kQOff + kTiles * kQBytes;
  static constexpr int kStatsOff = kRingOff + kStages * kKVBytes;
  static constexpr int kSmemBytes = 1024 + kStatsOff + int(sizeof(Stats));
  static_assert(kSmemBytes + 512 <= 227 * 1024, "shared memory per CTA");

  // TMEM: S^t at 128 t, P^t at 128 t + 64, O^t at 256 + D t.
  static constexpr int kOCol = kTiles * kBN;
  static constexpr int kTmemCols = 512;
  static_assert(kOCol + kTiles * D_ <= kTmemCols, "TMEM has 512 columns");
};

// Every mbarrier of the kernel. "count" = arrivals per phase. Barriers used
// once per K/V block have phase = number of blocks this CTA has processed
// before (over all earlier tiles too); barriers used once per work tile have
// phase = number of earlier tiles of this CTA.
template <int Stages> struct Barriers {
  // TMA -> MMA, count 1 + tx bytes. Per tile.
  uint64_t q_full[kTiles];
  // MMA -> TMA, count 1 (tcgen05.commit after the last Q^t K^T of a tile):
  // Q^t may be overwritten by the next tile's Q^t. Per tile. (Step 2+.)
  uint64_t q_empty[kTiles];
  // K/V ring: item u in stage u % Stages, u counts over all tiles.
  uint64_t kv_full[Stages];
  uint64_t kv_empty[Stages];
  // MMA -> softmax t, count 1: S^t_j in TMEM. Per block.
  uint64_t s_full[kTiles];
  // softmax t + correction -> MMA, count 256. Per block.
  uint64_t p_ready[kTiles];
  // MMA -> epilogue, count 1: O^t final. Per tile.
  uint64_t o_full[kTiles];
  // epilogue -> MMA, count 128: O^t read out of TMEM, the next tile's first
  // P V may overwrite it. Per tile. (Step 2+.)
  uint64_t o_empty[kTiles];
  // softmax t -> correction, count 128: alpha^t_j published. Per block.
  uint64_t stats_full[kTiles];
  // softmax t -> correction, count 128: final m, l published. Per tile.
  uint64_t final_full[kTiles];
  uint32_t tmem_slot;
};

struct Params {
  int b, n, h;
  float scale, scale_log2; // scale_log2 = scale * log2(e)
  void *o;
  float *lse;
  int pairs; // Q tile pairs per (batch, head) = ceil(N / 256)
  int total; // work tiles = pairs * B * H
};

// ===========================================================================
// Scheduling (lesson sections 3 and 5).
// ===========================================================================

struct Tile {
  int q0, head, batch;
};

// Index of the k-th work tile of this CTA in the schedule order, or a value
// >= total if the CTA has no k-th tile. Static: computed by every role from
// (blockIdx.x, gridDim.x, k) alone, so all roles agree without talking.
// Step 1: gridDim.x = total, so k = 0 is the only tile. Step 2: round robin.
// Step 3: boustrophedon -- odd rounds are dealt in reverse, so the CTA that
// got the lightest tile of a round gets the heaviest of the next one.
template <class C> __device__ __forceinline__ int tile_index(int k) {
  const int g = gridDim.x, b = blockIdx.x;
  if constexpr (C::kLpt)
    return k * g + ((k & 1) ? g - 1 - b : b); // boustrophedon
  else
    return k * g + b; // round robin
}

// Schedule index -> coordinates. Default (steps 1-2, and every non-causal
// case): pair fastest, then head, then batch -- the 16 pairs of a head are
// neighbours, so concurrently running CTAs share that head's K/V in L2.
// Step 3, causal: LPT -- index / (B H) is the "level": level 0 is the last
// (heaviest) pair of every (batch, head), then the next-to-last, ...; inside
// a level the B H heads are in order.
template <class C, bool kCausal>
__device__ __forceinline__ Tile tile_coords(int idx, const Params &p) {
  int pair, bh;
  if constexpr (C::kLpt && kCausal) {
    const int bhs = p.b * p.h;
    pair = p.pairs - 1 - idx / bhs;
    bh = idx % bhs;
  } else {
    pair = idx % p.pairs;
    bh = idx / p.pairs;
  }
  return {pair * (kTiles * kBM), bh % p.h, bh / p.h};
}

// Block classification (lesson section 3.1). For a pair starting at q0:
// blocks j >= num_kv_blocks are fully masked for both tiles (never loaded).
template <bool kCausal>
__device__ __forceinline__ int num_kv_blocks(int q0, int n) {
  int blocks = (n + kBN - 1) / kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + kTiles * kBM + kBN - 1) / kBN);
  return blocks;
}

// For the Q tile starting at row qt: blocks 0 .. unmasked - 1 need no mask
// for any of its 128 rows: every key is < N, and (causal) every key
// k <= 128 j + 127 <= qt <= row. Blocks unmasked .. blocks - 1 are the
// masked tail: the diagonal block, the block past N, and (tile 0 of a
// causal pair) the block that is fully masked for tile 0 but needed by
// tile 1.
template <bool kCausal>
__device__ __forceinline__ int num_unmasked_blocks(int qt, int n) {
  int full = n / kBN;
  if constexpr (kCausal)
    full = min(full, qt / kBN);
  return full;
}

// Walks this CTA's work tiles in schedule order and calls
// body(tile, blocks, it, nb): `it` = tiles this CTA did before, `nb` = K/V
// blocks it did before (over all earlier tiles). Every role calls this with
// the same arguments, so all roles see the same tile sequence and derive
// the same barrier phases. Returns the number of tiles.
template <class C, bool kCausal, class Body>
__device__ __forceinline__ int for_each_tile(const Params &p, Body &&body) {
  int it = 0, nb = 0;
  for (int k = 0; k * static_cast<int>(gridDim.x) < p.total; ++k) {
    const int idx = tile_index<C>(k);
    if (idx >= p.total) // last round of step 3: reversed, may be partial
      continue;
    const Tile w = tile_coords<C, kCausal>(idx, p);
    const int blocks = num_kv_blocks<kCausal>(w.q0, p.n);
    body(w, blocks, it, nb);
    ++it;
    nb += blocks;
  }
  return it;
}

// --------------------------------------------------------------------- MMA
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
// Softmax-warpgroup helpers (lab 41).
// ===========================================================================
__device__ __forceinline__ void load_s_row(uint32_t lane_addr,
                                           float (&s)[kBN]) {
#pragma unroll
  for (int c = 0; c < kBN; c += 32)
    lab::tmem_ld_32x32b_x32_sync(lane_addr + c,
                                 *reinterpret_cast<float (*)[32]>(s + c));
}

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

template <int kPolyPairs> constexpr bool use_poly(int q) {
  return (q + 1) * kPolyPairs / 16 != q * kPolyPairs / 16;
}

template <int kPolyPairs>
__device__ __forceinline__ float exp_and_sum(float (&s)[kBN], float scale_log2,
                                             float base) {
  const float2 mul = make_float2(scale_log2, scale_log2);
  const float2 sub = make_float2(-base, -base);
  float2 sum[4] = {};
#pragma unroll
  for (int c = 0; c < kBN; c += 2) {
    float2 x = __ffma2_rn(make_float2(s[c], s[c + 1]), mul, sub);
    if (use_poly<kPolyPairs>((c % 32) / 2)) {
      x = exp2_poly2(x);
    } else {
      x.x = exp2_approx(x.x);
      x.y = exp2_approx(x.y);
    }
    s[c] = x.x;
    s[c + 1] = x.y;
    sum[(c / 2) % 4] = __fadd2_rn(sum[(c / 2) % 4], x);
  }
  const float2 t =
      __fadd2_rn(__fadd2_rn(sum[0], sum[1]), __fadd2_rn(sum[2], sum[3]));
  return t.x + t.y;
}

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

// Running state of one row of one tile.
struct RowState {
  float m; // base of the exponentials (raw-score units, lab 41 step 3)
  float l; // running sum
};

// One K/V block of the softmax loop (lab 41's loop body). kMask selects the
// masked-tail version; the unmasked version contains no mask code at all.
// `ph` is the phase of s_full / stats_full / p_ready for this block.
template <bool kMask, bool kCausal, class T, class C>
__device__ __forceinline__ void
softmax_block(RowState &rs, int ph, int k0, int row, int t, int r,
              uint32_t s_addr, Barriers<C::kStages> &bar, Stats &st,
              const Params &p) {
  lab::mbar_wait(&bar.s_full[t], ph & 1);
  lab::tcgen05_fence_after_thread_sync();
  float s[kBN];
  load_s_row(s_addr, s);
  if constexpr (kMask) {
    // Visible keys of this row in this block: k < N and (causal) k <= row.
    int end = p.n;
    if constexpr (kCausal)
      end = min(end, row + 1);
    const int limit = end - k0; // may be <= 0 (block fully masked for row)
#pragma unroll
    for (int c = 0; c < kBN; ++c)
      if (c >= limit)
        s[c] = -INFINITY;
  }
  const float m_new = fmaxf(rs.m, row_max(s));
  float alpha = 1.f;
  if ((m_new - rs.m) * p.scale_log2 > C::kTau) { // conditional rescale
    alpha = exp2_approx((rs.m - m_new) * p.scale_log2);
    rs.m = m_new;
  }
  const float base = rs.m == -INFINITY ? 0.f : rs.m * p.scale_log2;
  st.alpha[t][r] = alpha;
  lab::mbar_arrive(&bar.stats_full[t]);
  rs.l = rs.l * alpha + exp_and_sum<C::kPolyPairs>(s, p.scale_log2, base);
  write_p_tmem<T>(s_addr + kPCol, s);
  lab::tcgen05_fence_before_thread_sync();
  lab::mbar_arrive(&bar.p_ready[t]);
}

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

// Epilogue for row r of the Q tile starting at qt: O / l from TMEM straight
// to global memory (each thread writes its row's D contiguous elements, in
// 16-byte stores), LSE = scale * m + ln(l). Unlike labs 34-41, O is not
// staged in the Q buffer: the next tile's Q may already be loading there.
// Returns after the last tcgen05.ld; `o_empty` (if given) is arrived as
// soon as O^t has been read, before the global stores.
template <class T, class C>
__device__ __forceinline__ void
store_o_and_lse(uint32_t o_addr, int qt, int r, int head, int batch, float m,
                float l, uint64_t *o_empty, const Params &p) {
  const int row = qt + r;
  const bool valid = row < p.n;
  const float inv = 1.f / l;
  T *dst = static_cast<T *>(p.o) +
           ((static_cast<int64_t>(batch) * p.n + row) * p.h + head) * C::kD;
#pragma unroll
  for (int c = 0; c < C::kD; c += 32) {
    float o[32];
    lab::tmem_ld_32x32b_x32_sync(o_addr + c, o);
    if (c + 32 == C::kD && o_empty != nullptr) {
      // All of this thread's reads of O^t are complete (ld + wait::ld).
      lab::tcgen05_fence_before_thread_sync();
      lab::mbar_arrive(o_empty);
    }
    if (valid) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint4 v;
        v.x = pack2<T>(o[8 * q + 0] * inv, o[8 * q + 1] * inv);
        v.y = pack2<T>(o[8 * q + 2] * inv, o[8 * q + 3] * inv);
        v.z = pack2<T>(o[8 * q + 4] * inv, o[8 * q + 5] * inv);
        v.w = pack2<T>(o[8 * q + 6] * inv, o[8 * q + 7] * inv);
        *reinterpret_cast<uint4 *>(dst + c + 8 * q) = v;
      }
    }
  }
  if (valid)
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
                const __grid_constant__ CUtensorMap v_map, Params p) {
  constexpr int S = C::kStages;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem =
      smem_raw + ((1024 - (lab::shared_addr(smem_raw) & 1023)) & 1023);
  auto sq = [&](int t) {
    return reinterpret_cast<T *>(smem + C::kQOff + t * C::kQBytes);
  };
  auto ring = [&](int s) {
    return reinterpret_cast<T *>(smem + C::kRingOff + s * C::kKVBytes);
  };
  Stats &st = *reinterpret_cast<Stats *>(smem + C::kStatsOff);
  __shared__ Barriers<S> bar;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n = p.n;

  if (threadIdx.x == 0) {
    for (int t = 0; t < kTiles; ++t) {
      lab::mbar_init(&bar.q_full[t], 1);
      lab::mbar_init(&bar.q_empty[t], 1);
      lab::mbar_init(&bar.s_full[t], 1);
      lab::mbar_init(&bar.p_ready[t], 2 * kBM);
      lab::mbar_init(&bar.o_full[t], 1);
      lab::mbar_init(&bar.o_empty[t], kBM);
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
  const uint32_t lane_addr =
      tmem + (static_cast<uint32_t>(32 * (warp % 4)) << 16);

  if (warp >= C::kMmaWarp) {
    lab::setmaxnreg_dec<C::kRegOther>();
    if (warp == C::kLoadWarp && lane == 0) {
      // ------------------------------------------------------ TMA load
      lab::prefetch_tensor_map(&q_map);
      lab::prefetch_tensor_map(&k_map);
      lab::prefetch_tensor_map(&v_map);
      int u = 0; // ring item, over all tiles
      const int tiles = for_each_tile<C, kCausal>(
          p, [&](const Tile &w, int blocks, int it, int) {
            for (int i = 0; i < 2 * blocks; ++i, ++u) {
              const int s = u % S;
              lab::mbar_wait(&bar.kv_empty[s], ((u / S) & 1) ^ 1);
              lab::mbar_arrive_expect_tx(&bar.kv_full[s], C::kKVBytes);
              const CUtensorMap *map = (i & 1) ? &v_map : &k_map;
              for (int pn = 0; pn < C::kPanels; ++pn)
                lab::tma_load_4d(ring(s) + pn * kBN * kPanel, map,
                                 &bar.kv_full[s], pn * kPanel, w.head,
                                 (i / 2) * kBN, w.batch);
              if (i == 0) {
                // Q^0, Q^1 after K_0: K_0 of the next tile can be in flight
                // while we wait for the previous tile to release Q.
                for (int t = 0; t < kTiles; ++t) {
                  if constexpr (C::kPersistent)
                    lab::mbar_wait(&bar.q_empty[t], (it & 1) ^ 1);
                  lab::mbar_arrive_expect_tx(&bar.q_full[t], C::kQBytes);
                  for (int pn = 0; pn < C::kPanels; ++pn)
                    lab::tma_load_4d(sq(t) + pn * kBM * kPanel, &q_map,
                                     &bar.q_full[t], pn * kPanel, w.head,
                                     w.q0 + t * kBM, w.batch);
                }
              }
            }
          });
      // Drain: every commit to a ring stage / q_empty has arrived before
      // the CTA exits (an arrival on freed shared memory is undefined).
      for (int i = 0; i < S; ++i, ++u)
        lab::mbar_wait(&bar.kv_empty[u % S], ((u / S) & 1) ^ 1);
      if constexpr (C::kPersistent)
        for (int t = 0; t < kTiles; ++t)
          lab::mbar_wait(&bar.q_empty[t], (tiles & 1) ^ 1);
    } else if (warp == C::kMmaWarp) {
      // ----------------------------------------------------------- MMA
      const bool leader = lab::elect_one();
      auto acquire = [&](int u) {
        lab::mbar_wait(&bar.kv_full[u % S], (u / S) & 1);
        lab::tcgen05_fence_after_thread_sync();
        return ring(u % S);
      };
      auto s_col = [&](int t) { return tmem + t * kBN; };
      auto o_col = [&](int t) { return tmem + C::kOCol + t * C::kD; };
      int u = 0; // ring item of K_0 of the current tile
      for_each_tile<C, kCausal>(
          p, [&](const Tile &w, int blocks, int it, int nb) {
            // Prologue: S^t_0 = Q^t K_0^T. S^t / P^t of the previous tile were
            // last read by its final P^t V, issued earlier by this thread.
            const T *k0 = acquire(u);
#pragma unroll 1
            for (int t = 0; t < kTiles; ++t) {
              lab::mbar_wait(&bar.q_full[t], it & 1);
              lab::tcgen05_fence_after_thread_sync();
              if (leader) {
                mma_qk<T, C>(s_col(t), sq(t), k0);
                lab::tcgen05_commit(&bar.s_full[t]);
                if (C::kPersistent && blocks == 1) // last read of Q^t
                  lab::tcgen05_commit(&bar.q_empty[t]);
              }
              __syncwarp();
            }
            if (leader)
              lab::tcgen05_commit(&bar.kv_empty[u % S]);
            for (int j = 0; j < blocks; ++j) {
              const bool more = j + 1 < blocks;
              const T *v = acquire(u + 2 * j + 1);
              const T *k = more ? acquire(u + 2 * j + 2) : nullptr;
#pragma unroll 1
              for (int t = 0; t < kTiles; ++t) {
                lab::mbar_wait(&bar.p_ready[t], (nb + j) & 1);
                if (C::kPersistent && j == 0) // previous O^t read out of TMEM
                  lab::mbar_wait(&bar.o_empty[t], (it & 1) ^ 1);
                lab::tcgen05_fence_after_thread_sync();
                if (leader) {
                  mma_pv_ts<T, C>(o_col(t), s_col(t) + kPCol, v, j > 0);
                  if (t == kTiles - 1)
                    lab::tcgen05_commit(&bar.kv_empty[(u + 2 * j + 1) % S]);
                  if (more) {
                    mma_qk<T, C>(s_col(t), sq(t), k);
                    lab::tcgen05_commit(&bar.s_full[t]);
                    if (C::kPersistent && j + 2 == blocks) // last read of Q^t
                      lab::tcgen05_commit(&bar.q_empty[t]);
                  } else {
                    lab::tcgen05_commit(&bar.o_full[t]);
                  }
                }
                __syncwarp();
              }
              if (more && leader)
                lab::tcgen05_commit(&bar.kv_empty[(u + 2 * j + 2) % S]);
            }
            u += 2 * blocks;
          });
    }
  } else if (warp < 4 * kTiles) {
    // ================================================ softmax warpgroups
    lab::setmaxnreg_inc<C::kRegSoftmax>();
    const int t = warp / 4;
    const int r = threadIdx.x % kBM;
    const uint32_t s_addr = lane_addr + t * kBN;
    for_each_tile<C, kCausal>(
        p, [&](const Tile &w, int blocks, int it, int nb) {
          const int qt = w.q0 + t * kBM;
          const int row = qt + r;
          const int unmasked = min(num_unmasked_blocks<kCausal>(qt, n), blocks);
          RowState rs{-INFINITY, 0.f}; // fresh for every tile
          int j = 0;
          for (; j < unmasked; ++j) // hot loop: no mask code
            softmax_block<false, kCausal, T, C>(rs, nb + j, j * kBN, row, t, r,
                                                s_addr, bar, st, p);
          for (; j < blocks; ++j) // masked tail
            softmax_block<true, kCausal, T, C>(rs, nb + j, j * kBN, row, t, r,
                                               s_addr, bar, st, p);
          st.fin_m[it & 1][t][r] = rs.m;
          st.fin_l[it & 1][t][r] = rs.l;
          lab::mbar_arrive(&bar.final_full[t]);
        });
  } else {
    // ============================================== correction warpgroup
    lab::setmaxnreg_dec<C::kRegCorrection>();
    const int r = threadIdx.x % kBM;
    for_each_tile<C, kCausal>(p, [&](const Tile &w, int blocks, int it,
                                     int nb) {
      for (int j = 0; j < blocks; ++j) {
        for (int t = 0; t < kTiles; ++t) {
          lab::mbar_wait(&bar.stats_full[t], (nb + j) & 1);
          lab::tcgen05_fence_after_thread_sync();
          if (j > 0) {
            const float alpha = st.alpha[t][r];
            if (__any_sync(0xffffffffu, alpha != 1.f))
              rescale_o_row<C>(lane_addr + C::kOCol + t * C::kD, alpha);
          }
          lab::tcgen05_fence_before_thread_sync();
          lab::mbar_arrive(&bar.p_ready[t]);
        }
      }
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.final_full[t], it & 1);
        lab::mbar_wait(&bar.o_full[t], it & 1);
        lab::tcgen05_fence_after_thread_sync();
        store_o_and_lse<T, C>(lane_addr + C::kOCol + t * C::kD, w.q0 + t * kBM,
                              r, w.head, w.batch, st.fin_m[it & 1][t][r],
                              st.fin_l[it & 1][t][r],
                              C::kPersistent ? &bar.o_empty[t] : nullptr, p);
      }
    });
  }

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

CUtensorMap bnhd_map(const void *base, const lab::AttentionProblem &p,
                     uint32_t box_rows) {
  const CUtensorMapDataType dt = p.dtype == lab::DType::F16
                                     ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                     : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  return lab::make_tensor_map_4d(base, dt, p.b, p.n, p.h, p.d, 1, box_rows, 1,
                                 kPanel, CU_TENSOR_MAP_SWIZZLE_128B);
}

struct Maps {
  CUtensorMap q, k, v;
  int num_sms;
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
  const int pairs = (pr.n + kTiles * kBM - 1) / (kTiles * kBM);
  const int total = pairs * pr.b * pr.h;
  Params prm{pr.b, pr.n,   pr.h,  pr.scale, pr.scale * 1.4426950408889634f,
             pr.o, pr.lse, pairs, total};
  // Step 1: one CTA per work tile. Steps 2-3: one CTA per SM (1 CTA fits).
  const int grid = C::kPersistent ? std::min(total, maps.num_sms) : total;
  kernel<<<grid, C::kThreads, C::kSmemBytes, stream>>>(maps.q, maps.k, maps.v,
                                                       prm);
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
  int dev = 0;
  LAB_CUDA_CHECK(cudaGetDevice(&dev));
  LAB_CUDA_CHECK(cudaDeviceGetAttribute(&plan.maps.num_sms,
                                        cudaDevAttrMultiProcessorCount, dev));
  dispatch(plan.maps, p, nullptr, true);
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  dispatch(plan.maps, p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
