// Reference solution for 42_fa4_2cta: the FA4 kernel of lab 41 (lab 40's
// pipeline -- two Q tiles per CTA, softmax ping-pong, P in TMEM, correction
// warpgroup, setmaxnreg -- with lab 41's softmax: software exp2 mixed with
// MUFU.EX2, conditional rescale) run by a CTA PAIR with the 2-CTA
// tcgen05.mma (cta_group::2): both CTAs' Q tiles go through one M = 256 MMA,
// and each CTA loads only HALF of every K and V block.
// Lesson: course/42_fa4_2cta/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous. LSE: [B, H, N] FP32 (ln).
// D in {64, 128}, scale > 0 finite, causal or not.
//
// Pair layout. A cluster of 2 CTAs (rank 0 = leader, rank 1 = peer) owns
// 512 query rows starting at q0 = 512 * cluster index (grid = (2 * clusters,
// H, B)). CTA r's tile t (r, t in {0, 1}) holds rows
//     q0 + 256 t + 128 r .. + 127
// so the pair MMA of tile t covers the 256 consecutive rows q0 + 256 t ..
//     S^t (M = 256, N = 128 keys) = [Q^t_leader; Q^t_peer] . K_j^T
//     O^t (M = 256, N = D)       += [P^t_leader; P^t_peer] . V_j
// and writes rows 0..127 into the leader's TMEM, 128..255 into the peer's.
// The B operand is split along N between the two CTAs' shared memory:
//     K_j: CTA r holds keys 64 r .. 64 r + 63 (all D columns, K-major)
//     V_j: CTA r holds columns r D/2 .. r D/2 + D/2 - 1 (all 128 keys,
//          MN-major; 128-byte swizzle for D = 128, 64-byte for D = 64)
//
// Roles (each CTA, 16 warps, as labs 40-41):
//   warps 0-3   softmax warpgroup 0 (tile 0): thread r owns row r = TMEM lane
//   warps 4-7   softmax warpgroup 1 (tile 1)
//   warps 8-11  correction warpgroup: rescales O^0, O^1; epilogue
//   warp 12     MMA: LEADER only; issues every pair MMA and commit
//   warp 13     TMA (lane 0): this CTA's Q tiles and K/V halves (both CTAs)
//   warps 14-15 idle (warpgroup 3 must be complete for setmaxnreg)
//
// step 1: pair QK^T only: S of the first K/V block, written to the probe.
//         Cluster launch, pair TMEM allocation, split K, leader-side
//         full barriers, multicast commits, drain + cluster teardown.
// step 2: the full pair pipeline: split V, P^t from both CTAs' TMEM (TS pair
//         MMA), p_ready completed by the arrivals of BOTH CTAs (arrive
//         .release.cluster, as in lab 33), causal (pair tile t walks the
//         blocks its last row q0 + 256 t + 255 needs).
// step 3: the cross-CTA handoff without a GPU-wide fence: p_ready arrivals
//         with .release.cta (lab::mbar_arrive_remote), as CUTLASS and FA4.
//         .release.cluster costs a MEMBAR.ALL.GPU per arriving thread.
//
// Every barrier's protocol is spelled out at its declaration (struct
// Barriers) and in the lesson (section 5).
#include "lab/cluster.cuh"
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

constexpr int kBM = 128;   // query rows per tile = TMEM lanes per CTA
constexpr int kBN = 128;   // keys per K/V block = N of S = K of P.V
constexpr int kTiles = 2;  // Q tiles per CTA
constexpr int kPanel = 64; // 64 16-bit elements = one 128-byte swizzle row
constexpr int kUmmaK = 16; // K of one tcgen05.mma for 16-bit inputs
constexpr int kPCol = 64;  // P^t (128 x 16-bit = 64 columns) at S^t + 64

constexpr int kPair = 2;                          // CTAs per cluster
constexpr int kPairM = kPair * kBM;               // 256 = M of a pair MMA
constexpr int kHalfKeys = kBN / kPair;            // keys of K_j per CTA
constexpr int kClusterRows = kTiles * kPairM;     // 512 query rows
constexpr uint16_t kPairMask = (1u << kPair) - 1; // multicast to both CTAs

// ===========================================================================
// Exponentials (lab 41) and small helpers (lab 40).
// ===========================================================================
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
//   f = x - (r - 1.5 * 2^23) in [0, 1]       abs. error <= 2^-25 (only
//                                            for x in (-0.5, 0)): far below
//                                            the polynomial error
//   p = ((c3 f + c2) f + c1) f + c0 in [1, 2)
//   2^x = 2^n * p: add n to p's exponent field (bits 23-30)
// Valid for x <= 127 (here x <= kTau = 8). x = -127 gives exactly +0
// because p(0) = c0 = 1 has the bit pattern 127 << 23. Two lanes per
// instruction: FADD2 / FFMA2 do two FP32 operations each (lesson 4.2).
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
  // (r's bits) << 23 = n << 23 modulo 2^32: the 1.5 * 2^23 part shifts out.
  return make_float2(
      __int_as_float(__float_as_int(p.x) + (__float_as_int(r.x) << 23)),
      __int_as_float(__float_as_int(p.y) + (__float_as_int(r.y) << 23)));
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
// Configuration.
// ===========================================================================
struct Stats {
  float alpha[kTiles][kBM];
  float fin_m[kTiles][kBM], fin_l[kTiles][kBM];
};

template <int D_, int Step> struct Config {
  static constexpr int kD = D_;
  static constexpr int kPanels = D_ / kPanel;
  static constexpr bool kProbe = Step == 1;       // S of block 0 only
  static constexpr bool kCheapArrive = Step >= 3; // .release.cta arrivals
  // Softmax arithmetic of lab 41 step 3 in every step: kPolyPairs of every
  // 16 column pairs use the software exp2, the rest MUFU.EX2; conditional
  // rescale with threshold kTau (log2 units). Lab 41's split is kept (NOTES:
  // 4 pairs is ~3% faster for non-causal D = 64 but spills in the causal
  // D = 64 kernels at 192 registers).
  static constexpr int kPolyPairs = 3;
  static constexpr bool kLazy = true;
  static constexpr float kTau = 8.f;

  static constexpr int kMmaWarp = 12, kLoadWarp = 13, kThreads = 512;
  static constexpr int kRegSoftmax = 192, kRegCorrection = 80, kRegOther = 48;
  static_assert(2 * kRegSoftmax + kRegCorrection + kRegOther == 512,
                "register pool: 64K registers per SM");

  // One ring for the halves K_0, V_0, K_1, V_1, ... (item u: K_{u/2} if u
  // is even, V_{u/2} if odd). Every half has the same size: 64 keys x D
  // (K) = 128 keys x D/2 (V) = 8 KB (D = 64) or 16 KB (D = 128) -- half of
  // lab 40's stage, so twice as many stages fit.
  static constexpr int kQBytes = kBM * D_ * 2;        // one Q tile
  static constexpr int kHalfBytes = kBN * D_ * 2 / 2; // one K or V half
  static constexpr int kVCols = D_ / kPair;           // V columns per CTA
  static constexpr int kVRowBytes = kVCols * 2;       // 128 (D=128), 64 (64)
  static constexpr int kStages = D_ == 128 ? 8 : 16;
  static constexpr int kQOff = 0;
  static constexpr int kRingOff = kQOff + kTiles * kQBytes;
  static constexpr int kStatsOff = kRingOff + kStages * kHalfBytes;
  static constexpr int kSmemBytes = 1024 + kStatsOff + int(sizeof(Stats));
  static_assert(kSmemBytes + 256 <= 227 * 1024, "shared memory per CTA");

  // TMEM, the same columns in both CTAs (lab 40):
  //   S^t: 128 t .. ; P^t: 128 t + 64 .. ; O^t: 256 + D t ..
  static constexpr int kOCol = kTiles * kBN;
  static constexpr int kTmemCols = 512;
  static_assert(kOCol + kTiles * D_ <= kTmemCols, "TMEM has 512 columns");
};

// Every mbarrier of the kernel. The struct has the same offset in both CTAs.
// "L" = only the leader's copy is used; "both" = each CTA uses its own.
template <int Stages> struct Barriers {
  // L. TMA (both CTAs) -> MMA. count 1: the leader's arrive_expect_tx with
  // the bytes of BOTH CTAs; complete_tx from both CTAs' cta_group::2 loads.
  uint64_t q_full[kTiles];
  uint64_t kv_full[Stages];
  // both. MMA -> TMA. count 1: the leader's multicast tcgen05.commit; each
  // CTA's TMA lane waits on its own copy before overwriting its half.
  uint64_t kv_empty[Stages];
  // both. MMA -> softmax t. count 1, multicast commit after Q^t K_j^T: S^t_j
  // is in this CTA's TMEM (and, as in lab 40, P^t_{j-1} V_{j-1} is complete).
  uint64_t s_full[kTiles];
  // L. softmax t + correction of BOTH CTAs -> MMA. count 2 x 256: every
  // softmax-t and correction thread of both CTAs arrives (arrive_leader)
  // once per block: P^t_j written into its TMEM, S^t_j read, O^t rescaled.
  uint64_t p_ready[kTiles];
  // both. MMA -> correction (epilogue). count 1, multicast commit after the
  // last P^t V: O^t is final and every MMA reading Q^t (both CTAs) is done.
  uint64_t o_full[kTiles];
  // both, local as in lab 40: softmax t -> correction of the same CTA.
  uint64_t stats_full[kTiles];
  uint64_t final_full[kTiles];
  uint32_t tmem_slot;
};

struct Params {
  int n, h;
  float scale, scale_log2;
  float *lse;
  float *probe; // step 1: [B, H, N, probe_cols]
  int probe_cols;
};

// First query row of this CTA's tile t (r = rank in the pair).
__device__ __forceinline__ int tile_row(int q0, int t, int rank) {
  return q0 + t * kPairM + rank * kBM;
}

// Number of K/V blocks that pair tile t walks (the same in both CTAs: the
// pair MMA is shared). Non-causal: all of them. Causal: those that contain
// a key the last row of the PAIR tile, q0 + 256 t + 255, may see -- so tile
// 0 stops two blocks before tile 1, and the leader's rows of each pair tile
// get one fully masked block (as tile 0 in lab 40).
template <bool kCausal>
__device__ __forceinline__ int tile_blocks(int q0, int t, int n) {
  int blocks = (n + kBN - 1) / kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + (t + 1) * kPairM - 1) / kBN + 1);
  return blocks;
}

// One arrival of this thread on the LEADER's p_ready (a .shared::cluster
// address). Step 2: .release.cluster (lab 33's mbar_arrive_cluster); step 3:
// .release.cta (lesson section 4.4): the handoff is TMEM, already complete
// (tcgen05.wait::st) and ordered by tcgen05.fence::before_thread_sync.
template <class C>
__device__ __forceinline__ void arrive_leader(uint32_t addr) {
  if constexpr (C::kCheapArrive)
    lab::mbar_arrive_remote(addr);
  else
    lab::mbar_arrive_cluster(addr);
}

// --------------------------------------------------------------------- MMA
// S^t (both CTAs' TMEM) = [Q^t_L; Q^t_P] (256 x D) . K_j (128 x D)^T.
// sq: this tile's Q in the leader (the peer's at the same offset); sk: the
// leader's K half [D/64 panels][64 keys][64] (the peer holds keys 64..127).
template <class T, class C>
__device__ __forceinline__ void mma_qk_pair(uint32_t s_tmem, const T *sq,
                                            const T *sk) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kPairM, kBN, std::is_same_v<T, bf16>);
  const uint64_t da = lab::make_smem_desc_kmajor_sw128(sq);
  const uint64_t db = lab::make_smem_desc_kmajor_sw128(sk);
#pragma unroll
  for (int kk = 0; kk < C::kD / kUmmaK; ++kk) {
    const int panel = kk * kUmmaK / kPanel, col = kk * kUmmaK % kPanel;
    const uint32_t a_off = (panel * kBM * kPanel + col) * 2 / 16;
    const uint32_t b_off = (panel * kHalfKeys * kPanel + col) * 2 / 16;
    lab::mma_f16_ss_cg2(s_tmem, da + a_off, db + b_off, idesc, kk > 0);
  }
}

// O^t (both CTAs' TMEM) [+]= [P^t_L; P^t_P] (256 x 128) . V_j (128 x D).
// A = P from each CTA's own TMEM (TS form); B = V_j, N = D split in two:
// sv is the leader's half [128 keys][D/2] (MN-major), the peer holds the
// other D/2 columns at the same offset. Key slice kk = 16 keys = 16 rows.
template <class T, class C>
__device__ __forceinline__ void mma_pv_pair(uint32_t o_tmem, uint32_t p_tmem,
                                            const T *sv, bool accumulate) {
  constexpr uint32_t idesc =
      lab::make_idesc_f16(kPairM, C::kD, std::is_same_v<T, bf16>, true,
                          /*a_mn_major=*/false, /*b_mn_major=*/true);
  // One MN panel per CTA (64 columns with 128-byte rows, or 32 columns with
  // 64-byte rows): LBO is never used; pass the half's size.
  uint64_t db;
  if constexpr (C::kD == 128)
    db = lab::make_smem_desc_mnmajor_sw128(sv, C::kHalfBytes);
  else
    db = lab::make_smem_desc_mnmajor_sw64(sv, C::kHalfBytes);
#pragma unroll
  for (int kk = 0; kk < kBN / kUmmaK; ++kk)
    lab::mma_f16_ts_cg2(o_tmem, p_tmem + kk * (kUmmaK / 2),
                        db + kk * kUmmaK * C::kVRowBytes / 16, idesc,
                        accumulate || kk > 0);
}

// ===========================================================================
// Softmax / correction helpers (labs 40-41, unchanged).
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
template <int kPolyPairs> constexpr bool use_poly(int q) {
  return (q + 1) * kPolyPairs / 16 != q * kPolyPairs / 16;
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
  lab::fence_proxy_async();
  __syncwarp();
  if (lane == 0) {
    for (int pn = 0; pn < C::kPanels; ++pn)
      lab::tma_store_4d(o_map, sq + pn * kBM * kPanel + warp * 32 * kPanel,
                        pn * kPanel, head, q0 + warp * 32, batch);
    lab::bulk_commit_group();
    lab::bulk_wait_group_read<0>();
  }
  if (row < p.n)
    p.lse[(static_cast<int64_t>(batch) * p.h + head) * p.n + row] =
        m * p.scale + logf(l);
}

// Step 1: row `row` of S = scale * Q K_0^T -> probe[b, h, row, 0..cols-1].
__device__ __forceinline__ void write_probe(const Params &p, int batch,
                                            int head, int row,
                                            const float (&s)[kBN]) {
  if (row >= p.n)
    return;
  float *dst =
      p.probe +
      ((static_cast<int64_t>(batch) * p.h + head) * p.n + row) * p.probe_cols;
#pragma unroll
  for (int c = 0; c < kBN; ++c)
    if (c < p.probe_cols)
      dst[c] = s[c] * p.scale;
}

// ===========================================================================
// The kernel (launched with clusters of 2 CTAs along x).
// ===========================================================================
template <class T, class C, bool kCausal>
__global__ void __launch_bounds__(C::kThreads, 1)
    fa4_2cta_forward(const __grid_constant__ CUtensorMap q_map,
                     const __grid_constant__ CUtensorMap k_map,
                     const __grid_constant__ CUtensorMap v_map,
                     const __grid_constant__ CUtensorMap o_map, Params p) {
  constexpr int S = C::kStages;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem =
      smem_raw + ((1024 - (lab::shared_addr(smem_raw) & 1023)) & 1023);
  auto sq = [&](int t) {
    return reinterpret_cast<T *>(smem + C::kQOff + t * C::kQBytes);
  };
  auto ring = [&](int s) {
    return reinterpret_cast<T *>(smem + C::kRingOff + s * C::kHalfBytes);
  };
  Stats &st = *reinterpret_cast<Stats *>(smem + C::kStatsOff);
  __shared__ Barriers<S> bar;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int rank = static_cast<int>(lab::cluster_ctarank()); // 0 = leader
  const bool leader = rank == 0;
  const int q0 = static_cast<int>(lab::cluster_id_x()) * kClusterRows;
  const int head = blockIdx.y, batch = blockIdx.z;
  const int n = p.n;
  // K/V blocks of pair tile t (identical in both CTAs: the pair MMA is
  // shared). Tile 1 always walks the most; the ring and TMA follow it.
  // Recomputed where needed (a few integer ops): an array indexed by a
  // runtime t would live on the stack, and a value computed here and kept
  // across setmaxnreg into the 48-register roles is spilled.
  auto tblocks = [&](int t) {
    return C::kProbe ? 1 : tile_blocks<kCausal>(q0, t, n);
  };

  // ---- setup: barriers (thread 0), pair TMEM (warp 0 of BOTH CTAs), then
  // a cluster barrier: no CTA may signal the other's barriers (TMA
  // complete_tx, multicast commit, remote arrive) before they are
  // initialized, and the alloc must be ordered before every TMEM access.
  if (threadIdx.x == 0) {
    for (int t = 0; t < kTiles; ++t) {
      lab::mbar_init(&bar.q_full[t], 1);
      lab::mbar_init(&bar.s_full[t], 1);
      lab::mbar_init(&bar.p_ready[t], kPair * 2 * kBM);
      lab::mbar_init(&bar.o_full[t], 1);
      lab::mbar_init(&bar.stats_full[t], kBM);
      lab::mbar_init(&bar.final_full[t], kBM);
    }
    for (int s = 0; s < S; ++s) {
      lab::mbar_init(&bar.kv_full[s], 1);
      lab::mbar_init(&bar.kv_empty[s], 1);
    }
    lab::fence_barrier_init(); // .release.cluster: visible to the peer
  }
  if (warp == 0) {
    lab::tmem_alloc_cg2(&bar.tmem_slot, C::kTmemCols);
    lab::tmem_relinquish_alloc_permit_cg2();
  }
  lab::tcgen05_fence_before_thread_sync();
  lab::cluster_sync();
  lab::tcgen05_fence_after_thread_sync();
  const uint32_t tmem = bar.tmem_slot;
  const uint32_t lane_addr =
      tmem + (static_cast<uint32_t>(32 * (warp % 4)) << 16);

  if (warp >= C::kMmaWarp) {
    // ============================================ MMA, TMA (and idle) warps
    lab::setmaxnreg_dec<C::kRegOther>();
    if (warp == C::kLoadWarp && lane == 0) {
      // ------------------------------------------- TMA load (both CTAs)
      // Every load completes on the LEADER's barrier: the leader's MMA
      // thread is the only consumer. Only the leader arms it -- with the
      // bytes of both CTAs. A peer's complete_tx may arrive before the
      // leader's arrive_expect_tx; the phase still needs that one arrival.
      lab::prefetch_tensor_map(&q_map);
      lab::prefetch_tensor_map(&k_map);
      lab::prefetch_tensor_map(&v_map);
      for (int t = 0; t < kTiles; ++t) {
        if (leader)
          lab::mbar_arrive_expect_tx(&bar.q_full[t], kPair * C::kQBytes);
        const uint32_t full_l = lab::map_shared_rank(&bar.q_full[t], 0);
        for (int pn = 0; pn < C::kPanels; ++pn)
          lab::tma_load_4d_cg2(sq(t) + pn * kBM * kPanel, &q_map, full_l,
                               pn * kPanel, head, tile_row(q0, t, rank), batch);
      }
      const int items = 2 * tblocks(kTiles - 1) - (C::kProbe ? 1 : 0);
      for (int u = 0; u < items; ++u) {
        const int s = u % S, j = u / 2;
        lab::mbar_wait(&bar.kv_empty[s], ((u / S) & 1) ^ 1);
        if (leader)
          lab::mbar_arrive_expect_tx(&bar.kv_full[s], kPair * C::kHalfBytes);
        const uint32_t full_l = lab::map_shared_rank(&bar.kv_full[s], 0);
        if ((u & 1) == 0) {
          // K_j half: keys j * 128 + 64 rank .. + 63, all D columns.
#pragma unroll 1
          for (int pn = 0; pn < C::kPanels; ++pn)
            lab::tma_load_4d_cg2(ring(s) + pn * kHalfKeys * kPanel, &k_map,
                                 full_l, pn * kPanel, head,
                                 j * kBN + rank * kHalfKeys, batch);
        } else {
          // V_j half: all 128 keys, columns rank * D/2 .. + D/2 - 1.
          lab::tma_load_4d_cg2(ring(s), &v_map, full_l, rank * C::kVCols, head,
                               j * kBN, batch);
        }
      }
      // Drain: every multicast release aimed at this CTA's kv_empty has
      // arrived before the CTA may exit.
      for (int u = items; u < items + S; ++u)
        lab::mbar_wait(&bar.kv_empty[u % S], ((u / S) & 1) ^ 1);
    } else if (warp == C::kMmaWarp && leader) {
      // ------------------------------------------- MMA (leader only)
      const bool elected = lab::elect_one();
      auto acquire = [&](int u) {
        lab::mbar_wait(&bar.kv_full[u % S], (u / S) & 1);
        lab::tcgen05_fence_after_thread_sync();
        return ring(u % S);
      };
      auto s_col = [&](int t) { return tmem + t * kBN; };
      auto o_col = [&](int t) { return tmem + C::kOCol + t * C::kD; };

      // Prologue: S^0_0, S^1_0 (each = 256 rows of the pair).
      const T *k0 = acquire(0);
#pragma unroll 1
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.q_full[t], 0);
        lab::tcgen05_fence_after_thread_sync();
        if (elected) {
          mma_qk_pair<T, C>(s_col(t), sq(t), k0);
          lab::tcgen05_commit_cg2_multicast(&bar.s_full[t], kPairMask);
        }
        __syncwarp();
      }
      if (elected) // K_0 read in both CTAs by both tiles
        lab::tcgen05_commit_cg2_multicast(&bar.kv_empty[0], kPairMask);

      if constexpr (!C::kProbe) {
        // Lab 40's order PV^0_j QK^0_{j+1} PV^1_j QK^1_{j+1}, with pair
        // MMAs. Causal: tile 0 finishes two blocks before tile 1, which
        // then runs alone (no ping-pong partner) for its last blocks.
        const int blocks = tblocks(kTiles - 1), blocks0 = tblocks(0);
        for (int j = 0; j < blocks; ++j) {
          const bool more = j + 1 < blocks;
          const T *v = acquire(2 * j + 1);
          const T *k = more ? acquire(2 * j + 2) : nullptr;
#pragma unroll 1
          for (int t = 0; t < kTiles; ++t) {
            const int tb = t == 0 ? blocks0 : blocks; // this tile's blocks
            if (j >= tb)
              continue;
            // Both CTAs' P^t_j written, S^t_j read, O^t rescaled. The
            // arrivals come from the peer too: acquire at cluster scope.
            lab::mbar_wait_cluster(&bar.p_ready[t], j & 1);
            lab::tcgen05_fence_after_thread_sync();
            if (elected) {
              mma_pv_pair<T, C>(o_col(t), s_col(t) + kPCol, v, j > 0);
              if (t == kTiles - 1) // V_j: tile 1 is always the last reader
                lab::tcgen05_commit_cg2_multicast(
                    &bar.kv_empty[(2 * j + 1) % S], kPairMask);
              if (j + 1 < tb) {
                mma_qk_pair<T, C>(s_col(t), sq(t), k);
                lab::tcgen05_commit_cg2_multicast(&bar.s_full[t], kPairMask);
              } else {
                lab::tcgen05_commit_cg2_multicast(&bar.o_full[t], kPairMask);
              }
            }
            __syncwarp();
          }
          if (more && elected)
            lab::tcgen05_commit_cg2_multicast(&bar.kv_empty[(2 * j + 2) % S],
                                              kPairMask);
        }
      }
    }
  } else if (warp < 4 * kTiles) {
    // ================================== softmax warpgroups (both CTAs)
    lab::setmaxnreg_inc<C::kRegSoftmax>();
    const int t = warp / 4;
    const int r = threadIdx.x % kBM;
    const int qt = tile_row(q0, t, rank);
    const int row = qt + r;
    const uint32_t s_addr = lane_addr + t * kBN;
    // p_ready[t] of the leader, as a .shared::cluster address.
    const uint32_t p_ready_l = lab::map_shared_rank(&bar.p_ready[t], 0);
    // m: the base of the exponentials (raw-score units); with the
    // conditional rescale it may lag the running max (lab 41, section 3.5).
    float m = -INFINITY, l = 0.f;
    for (int j = 0; j < tblocks(t); ++j) {
      // S^t_j is in OUR TMEM: the leader's commit is multicast to both.
      lab::mbar_wait(&bar.s_full[t], j & 1);
      lab::tcgen05_fence_after_thread_sync();
      float s[kBN];
      load_s_row(s_addr, 0, s);
      if constexpr (C::kProbe) {
        write_probe(p, batch, head, row, s);
        break;
      }
      const int k0 = j * kBN;
      if (block_needs_mask<kCausal>(k0, qt, n)) {
        const int limit = visible_keys<kCausal>(row, k0, n);
#pragma unroll
        for (int c = 0; c < kBN; ++c)
          if (c >= limit)
            s[c] = -INFINITY;
      }
      // Online softmax with the conditional rescale of lab 41: the base m
      // moves only if the max grew by more than kTau (log2 units). Block 0:
      // m = -inf, so it always moves (alpha = 0).
      const float m_new = fmaxf(m, row_max(s));
      float alpha = 1.f;
      if ((m_new - m) * p.scale_log2 > C::kTau) {
        alpha = exp2_approx((m - m_new) * p.scale_log2);
        m = m_new;
      }
      const float base = m == -INFINITY ? 0.f : m * p.scale_log2;
      st.alpha[t][r] = alpha;
      lab::mbar_arrive(&bar.stats_full[t]);
      l = l * alpha + exp_and_sum<C::kPolyPairs>(s, p.scale_log2, base);
      write_p_tmem<T>(s_addr + kPCol, s);
      // Our tcgen05.ld/st are complete (wait::ld / wait::st) and ordered
      // before the arrival on the leader's p_ready: the leader's MMA reads
      // P from our TMEM and accumulates into our O.
      lab::tcgen05_fence_before_thread_sync();
      arrive_leader<C>(p_ready_l);
    }
    if constexpr (!C::kProbe) {
      st.fin_m[t][r] = m;
      st.fin_l[t][r] = l;
      lab::mbar_arrive(&bar.final_full[t]);
    }
  } else if (warp < 12) {
    // ================================== correction warpgroup (both CTAs)
    lab::setmaxnreg_dec<C::kRegCorrection>();
    if constexpr (!C::kProbe) {
      const int r = threadIdx.x % kBM;
      const int blocks = tblocks(kTiles - 1);
      for (int j = 0; j < blocks; ++j) {
        for (int t = 0; t < kTiles; ++t) {
          if (j >= tblocks(t))
            continue;
          lab::mbar_wait(&bar.stats_full[t], j & 1);
          lab::tcgen05_fence_after_thread_sync();
          if (j > 0) {
            // Lab 41: skip the warp's rescale when none of its 32 rows moved
            // (tcgen05.ld/st are warp-collective).
            const float alpha = st.alpha[t][r];
            if (__any_sync(0xffffffffu, alpha != 1.f))
              rescale_o_row<C>(lane_addr + C::kOCol + t * C::kD, alpha);
          }
          lab::tcgen05_fence_before_thread_sync();
          arrive_leader<C>(lab::map_shared_rank(&bar.p_ready[t], 0));
        }
      }
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.final_full[t], 0);
        lab::mbar_wait(&bar.o_full[t], 0);
        lab::tcgen05_fence_after_thread_sync();
        store_o_and_lse<T, C>(&o_map, sq(t), lane_addr + C::kOCol + t * C::kD,
                              r, head, batch, tile_row(q0, t, rank),
                              st.fin_m[t][r], st.fin_l[t][r], p);
      }
    }
  }

  // ---- teardown. Each CTA has drained what the other sends it (TMA:
  // kv_empty; softmax/correction: every s_full and o_full; the leader's MMA
  // waited every p_ready phase, i.e. every remote arrival). The cluster
  // barrier makes both agree before either frees TMEM or exits: the
  // leader's MMAs read the peer's smem and write the peer's TMEM.
  lab::tcgen05_fence_before_thread_sync();
  lab::cluster_sync();
  if (warp == 0) {
    lab::tcgen05_fence_after_thread_sync();
    lab::tmem_dealloc_cg2(tmem, C::kTmemCols);
  }
}

// ===========================================================================
// Host side.
// ===========================================================================
CUtensorMap bnhd_map(const void *base, const lab::AttentionProblem &p,
                     uint32_t box_rows, uint32_t box_cols,
                     CUtensorMapSwizzle swizzle) {
  const CUtensorMapDataType dt = p.dtype == lab::DType::F16
                                     ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                     : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  return lab::make_tensor_map_4d(base, dt, p.b, p.n, p.h, p.d, 1, box_rows, 1,
                                 box_cols, swizzle);
}

struct Maps {
  CUtensorMap q, k, v, o;
};

template <class T, int D, int Step, bool kCausal>
void launch(const Maps &maps, const lab::AttentionProblem &pr,
            cudaStream_t stream, bool setup) {
  using C = Config<D, Step>;
  auto kernel = fa4_2cta_forward<T, C, kCausal>;
  if (setup) {
    LAB_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    return;
  }
  Params prm{pr.n,
             pr.h,
             pr.scale,
             pr.scale * 1.4426950408889634f,
             pr.lse,
             pr.probe,
             pr.n < kBN ? pr.n : kBN};
  const int clusters = (pr.n + kClusterRows - 1) / kClusterRows;
  const dim3 grid(kPair * clusters, pr.h, pr.b);
  lab::launch_cluster(kernel, grid, dim3(C::kThreads), C::kSmemBytes, stream,
                      kPair, maps.q, maps.k, maps.v, maps.o, prm);
}

template <class T, int D>
void dispatch_step(const Maps &m, const lab::AttentionProblem &p,
                   cudaStream_t s, bool setup) {
  if (p.step == 1)
    launch<T, D, 1, false>(m, p, s, setup); // no mask in the probe
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
  if (p.step == 1)
    LAB_REQUIRE(p.probe != nullptr, "step 1 writes the probe");
  Plan plan{};
  plan.maps.q = bnhd_map(p.q, p, kBM, kPanel, CU_TENSOR_MAP_SWIZZLE_128B);
  plan.maps.k = bnhd_map(p.k, p, kHalfKeys, kPanel, CU_TENSOR_MAP_SWIZZLE_128B);
  // V half: 128 keys x D/2 columns; one swizzled row = D/2 elements.
  plan.maps.v = p.d == 128
                    ? bnhd_map(p.v, p, kBN, 64, CU_TENSOR_MAP_SWIZZLE_128B)
                    : bnhd_map(p.v, p, kBN, 32, CU_TENSOR_MAP_SWIZZLE_64B);
  plan.maps.o = bnhd_map(p.o, p, 32, kPanel, CU_TENSOR_MAP_SWIZZLE_128B);
  dispatch(plan.maps, p, nullptr, true);
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  dispatch(plan.maps, p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
