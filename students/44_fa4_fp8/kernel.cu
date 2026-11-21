// 44_fa4_fp8: the lab 40 FlashAttention-4 pipeline with FP8 (E4M3) Q, K, V:
// tcgen05.mma .kind::f8f6f4, per-tensor descales, P quantized to E4M3.
// Lesson: course/44_fa4_fp8/README.md
//
// Q, K, V: [B, N, H, D] E4M3 contiguous (one byte per element): element
//   (b, i, h, d) at ((b * N + i) * H + h) * D + d; the real values are
//   p.q_descale * Q, p.k_descale * K, p.v_descale * V (FP32 descales).
// O: [B, N, H, D] BF16. LSE: [B, H, N] FP32, natural log, of the real scores.
// Supported: D in {64, 128}, scale > 0 finite, causal or not.
//
// The pipeline is lab 40's step 3, unchanged and given: one CTA per (pair
// of Q tiles = 256 rows, head, batch), 16 warps:
//   warps 0-3 / 4-7  softmax warpgroups of tile 0 / 1 (thread r = row r)
//   warps 8-11       correction warpgroup (O rescale, epilogue)
//   warp 12 MMA, warp 13 TMA, warps 14-15 idle.
// run() picks a version by p.step (one kernel, three configurations):
//   step 1  Config<D, 1>  Q K^T in FP8, S (real units) -> probe; no P V
//   step 2  Config<D, 2>  P quantized to E4M3 in TMEM, FP8 P V, epilogue
//   step 3  Config<D, 3>  O normalized by the sum of the quantized P
//
// What changed from lab 40 (all given unless marked TODO):
//   - 8-bit tiles: one row of D bytes = one swizzle row (128B swizzle for
//     D = 128, 64B for D = 64), a single panel per tile; UINT8 tensor maps;
//   - O is BF16 and is staged over Q^0 + Q^1 (2 * 128 * D bytes);
//   - new header functions (include/lab/tcgen05.cuh): lab::mma_f8f6f4_ss,
//     lab::mma_f8f6f4_ts, lab::make_idesc_f8f6f4, lab::make_smem_desc_kmajor_sw64,
//     lab::make_smem_desc_mnmajor_sw64.
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tcgen05.cuh"
#include "lab/tma.cuh"

#include <cuda_bf16.h>

namespace {

using bf16 = __nv_bfloat16;
using fp8 = uint8_t; // E4M3 bit pattern

constexpr int kBM = 128;    // query rows per tile = MMA M = TMEM lanes
constexpr int kBN = 128;    // keys per K/V block = N of S = K of P.V
constexpr int kTiles = 2;   // Q tiles per CTA
constexpr int kUmmaK = 32;  // K of one kind::f8f6f4 MMA (8-bit elements)
constexpr int kPCol = 64;   // P^t (128 keys x 8 bit = 32 columns) at S^t + 64
constexpr int kOPanel = 64; // BF16 O: 64 elements = one 128-byte swizzle row

// P is stored as E4M3(2^kPLog2 * p) with p = exp(s - m) in (0, 1]: the
// largest value is 256 (< 448, the E4M3 maximum), and probabilities down to
// 2^-14 stay in the normal range of E4M3 (2^-6 .. 448) instead of 2^-6.
// A power of two changes only the exponent, so it adds no rounding.
constexpr float kPLog2 = 8.f;

// ===========================================================================
// Small local helpers.
// ===========================================================================

// 2^x with MUFU.EX2 (ex2.approx.ftz.f32; 2^-inf = +0).
__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Two FP32 -> one 32-bit word of two BF16 (x in the low half). For O.
__device__ __forceinline__ uint32_t pack2_bf16(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t *>(&h);
}

// Four FP32 -> one 32-bit word of four E4M3, x0 in the least significant
// byte. cvt.rn.satfinite.e4m3x2.f32 d, a, b puts a in the UPPER byte of d
// and b in the lower one (PTX: "cvt"); satfinite clamps to +-448 instead of
// producing NaN.
__device__ __forceinline__ uint32_t pack4_e4m3(float x0, float x1, float x2,
                                               float x3) {
  // TODO(2a) -- lesson section 4.4. Two instructions
  //   asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(u16) : "f"(a), "f"(b));
  // each convert two floats into a 16-bit pair -- mind which operand lands
  // in which byte (see the comment above) -- then combine the two pairs
  // into one word with x0 in byte 0 and x3 in byte 3.
  (void)x0; (void)x1; (void)x2; (void)x3;
  LAB_TODO(2);
  return 0;
}

// acc + (sum of the four E4M3 values of word w), in FP32. E4M3 -> FP16 is
// exact (cvt.rn.f16x2.e4m3x2, one F2FP per two values), and the mixed-
// precision add.rn.f32.f16 (PTX ISA 8.6, sm_100: one FHADD) adds an FP16
// value to an FP32 one with a single FP32 rounding -- the same result as
// converting to FP32 first, in half the instructions.
__device__ __forceinline__ float add_e4m3x4(float acc, uint32_t w) {
  float r;
  asm("{\n\t.reg .b32 h01, h23;\n\t.reg .b16 a, b, c, d;\n\t"
      ".reg .b16 w0, w1;\n\t.reg .f32 t0, t1, t2;\n\t"
      "mov.b32 {w0, w1}, %1;\n\t"
      "cvt.rn.f16x2.e4m3x2 h01, w0;\n\t"
      "cvt.rn.f16x2.e4m3x2 h23, w1;\n\t"
      "mov.b32 {a, b}, h01;\n\t"
      "mov.b32 {c, d}, h23;\n\t"
      "add.rn.f32.f16 t0, a, %2;\n\t"
      "add.rn.f32.f16 t1, b, t0;\n\t"
      "add.rn.f32.f16 t2, c, t1;\n\t"
      "add.rn.f32.f16 %0, d, t2;\n\t}"
      : "=f"(r)
      : "r"(w), "f"(acc));
  return r;
}

// ===========================================================================
// Configuration: warp roles, shared memory, TMEM.
//
// FP8 tiles in shared memory: one row of D elements is D bytes, so a whole
// row is one swizzle row -- 128-byte swizzle for D = 128, 64-byte swizzle for
// D = 64 -- and a tile is a single panel [rows][D] (lab 40: D/64 panels of
// 64 16-bit elements). TMA writes exactly this with a box row of D bytes and
// the matching CU_TENSOR_MAP_SWIZZLE_*.
// ===========================================================================

// Per-row statistics handed from the softmax to the correction warpgroup.
struct Stats {
  float alpha[kTiles][kBM]; // alpha^t_j of the current block
  float fin_m[kTiles][kBM], fin_l[kTiles][kBM]; // final m and l
  float fin_lo[kTiles][kBM]; // final normalizer of O (step 3: sum of E4M3 P)
};

template <int D_, int Step> struct Config {
  static constexpr int kD = D_;
  static constexpr bool kProbe = Step == 1;  // step 1: S to the probe only
  static constexpr bool kQuantL = Step >= 3; // step 3: l from E4M3 P
  static constexpr int kSwizzle = D_;        // swizzle span in bytes

  static constexpr int kMmaWarp = 12, kLoadWarp = 13, kThreads = 512;
  static constexpr int kRegSoftmax = 192, kRegCorrection = 80, kRegOther = 48;
  static_assert(2 * kRegSoftmax + kRegCorrection + kRegOther == 512,
                "register pool: 64K registers per SM");

  // Q^0, Q^1 (FP8) at the start; at the end the same 2 * 128 * D bytes hold
  // ONE BF16 O tile (128 * D * 2 bytes) as the TMA-store staging area, used
  // by tile 0 and then tile 1. Then the K/V ring (item u: K_{u/2} if u is
  // even, V_{u/2} if odd), then the correction statistics.
  static constexpr int kQBytes = kBM * D_;  // one Q tile, FP8
  static constexpr int kKVBytes = kBN * D_; // one K or V block, FP8
  static constexpr int kStages = D_ == 128 ? 8 : 12;
  static constexpr int kQOff = 0;
  static constexpr int kRingOff = kQOff + kTiles * kQBytes;
  static constexpr int kStatsOff = kRingOff + kStages * kKVBytes;
  static constexpr int kSmemBytes = 1024 + kStatsOff + int(sizeof(Stats));
  static_assert(kTiles * kQBytes == kBM * D_ * 2, "O staging = Q^0 + Q^1");
  static_assert(kSmemBytes + 256 <= 227 * 1024, "shared memory per CTA");

  // TMEM: S^t columns 128 t .. + 127; P^t (E4M3, 4 per column) columns
  // 128 t + 64 .. + 95; O^t columns 256 + D t .. + D - 1 (FP32).
  static constexpr int kOCol = kTiles * kBN;
  static constexpr int kTmemCols = 512;
  static_assert(kOCol + kTiles * D_ <= kTmemCols, "TMEM has 512 columns");
};

// Every mbarrier of the kernel; protocol as in lab 40 (step 3).
template <int Stages> struct Barriers {
  uint64_t q_full[kTiles];     // TMA -> MMA: Q^t landed
  uint64_t kv_full[Stages];    // TMA -> MMA: ring item u landed
  uint64_t kv_empty[Stages];   // MMA (commit) -> TMA: stage free
  uint64_t s_full[kTiles];     // MMA (commit after QK^t_j) -> softmax t
  uint64_t p_ready[kTiles];    // softmax t + correction (256) -> MMA
  uint64_t o_full[kTiles];     // MMA (commit after last PV^t) -> epilogue
  uint64_t stats_full[kTiles]; // softmax t -> correction: alpha^t_j
  uint64_t final_full[kTiles]; // softmax t -> correction: final m, l
  uint32_t tmem_slot;
};

struct Params {
  int n, h;
  float qk_scale;   // raw score Q8 . K8 -> real scaled score (TODO(1a))
  float scale_log2; // qk_scale * log2(e) (TODO(1a))
  float v_descale;  // applied once per output element in the epilogue
  float *lse;
  float *probe; // step 1: [B, H, N, N] FP32
};

template <bool kCausal>
__device__ __forceinline__ int num_kv_blocks(int q0, int n) {
  int blocks = (n + kBN - 1) / kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + kTiles * kBM + kBN - 1) / kBN);
  return blocks;
}

// K-major descriptor of an FP8 tile [rows][D bytes] (one swizzle row each).
template <class C>
__device__ __forceinline__ uint64_t desc_kmajor(const fp8 *tile) {
  // TODO(1b) -- lesson section 4.3. A tile is [rows][D bytes] written by TMA
  // with a D-byte swizzle (C::kSwizzle = 128 or 64). Return the K-major
  // descriptor that matches: lab::make_smem_desc_kmajor_sw128 (8 rows = one
  // 1024-byte atom) or lab::make_smem_desc_kmajor_sw64 (8 rows = 512 bytes).
  (void)tile;
  LAB_TODO(1);
  return 0;
}

// --------------------------------------------------------------------- MMA
// S (TMEM, 128 x 128 FP32) = Q8[128 x D] . K8_j[128 x D]^T, kind::f8f6f4.
template <class C>
__device__ __forceinline__ void mma_qk(uint32_t s_tmem, const fp8 *sq,
                                       const fp8 *sk) {
  // TODO(1b) -- lesson sections 4.1-4.3. S = Q8 . K8^T with C::kD / kUmmaK
  // MMAs lab::mma_f8f6f4_ss(s_tmem, a_desc, b_desc, idesc, accumulate):
  //  - idesc: lab::make_idesc_f8f6f4(M, N) -- which M and N? (both operands
  //    E4M3 and K-major are the defaults);
  //  - descriptors from desc_kmajor<C>; slice kk covers K elements 32 kk ..
  //    32 kk + 31 of every row: how many BYTES further does it start, and
  //    what is that in descriptor start-address units (16 bytes)?
  //  - the first MMA overwrites S, the others accumulate.
  (void)s_tmem; (void)sq; (void)sk;
  LAB_TODO(1);
}

// O (TMEM) [+]= P8[128 x 128] . V8_j with P in TMEM (TS form). B = V_j
// [keys][D] is MN-major (D contiguous), one D-byte swizzle row per key.
template <class C>
__device__ __forceinline__ void mma_pv_ts(uint32_t o_tmem, uint32_t p_tmem,
                                          const fp8 *sv, bool accumulate) {
  // TODO(2b) -- lesson sections 4.5-4.6. O [+]= P8 . V8_j, kBN / kUmmaK
  // TS MMAs lab::mma_f8f6f4_ts(o_tmem, a_tmem, b_desc, idesc, accumulate):
  //  - idesc: M = kBM, N = C::kD, E4M3 x E4M3, A K-major, B (V) MN-major;
  //  - B descriptor: lab::make_smem_desc_mnmajor_sw128 or _sw64 (as in
  //    desc_kmajor); LBO is unused here (pass kBN * C::kD), SBO = default;
  //  - key slice kk (keys 32 kk .. 32 kk + 31): how many TMEM columns
  //    further does it start in P (four keys per column), and how many
  //    bytes further in V (one D-byte row per key)?
  //  - the first MMA accumulates only if `accumulate`.
  (void)o_tmem; (void)p_tmem; (void)sv; (void)accumulate;
  LAB_TODO(2);
}

// ===========================================================================
// Softmax-warpgroup helpers (lab 40).
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

// s[c] = 2^(s[c] * scale_log2 - base); returns sum_c s[c].
__device__ __forceinline__ float exp_and_sum(float (&s)[kBN], float scale_log2,
                                             float base) {
  float sum[8] = {};
#pragma unroll
  for (int c = 0; c < kBN; ++c) {
    s[c] = exp2_approx(fmaf(s[c], scale_log2, -base));
    sum[c % 8] += s[c];
  }
  return ((sum[0] + sum[1]) + (sum[2] + sum[3])) +
         ((sum[4] + sum[5]) + (sum[6] + sum[7]));
}

// Step 1: row `row` of S (raw Q8 . K8 in s[]) in real units -> probe,
// [B, H, N, N], only the keys < n of rows < n.
__device__ __forceinline__ void write_probe(const Params &p, int batch,
                                            int head, int row, int k0,
                                            const float (&s)[kBN]) {
  // TODO(1c) -- lesson section 3.2. s[c] is the raw score Q8 . K8 of row
  // `row` and key k0 + c. Write it in real units (scale * Q . K, see
  // TODO(1a)) to p.probe[b][h][row][key] ([B, H, N, N] FP32, p.h heads,
  // p.n rows and keys), only for row < p.n and key < p.n. No mask.
  (void)p; (void)batch; (void)head; (void)row; (void)k0; (void)s;
  LAB_TODO(1);
}

// Row of P (s[] holds 2^8 p, FP32) -> E4M3 -> TMEM: column p_addr + c holds
// keys 4c .. 4c + 3 (key 4c in the least significant byte), the A-from-TMEM
// layout of the 8-bit TS MMA; 32 columns, over S columns 64..95 (the whole S
// row is already in registers). Then wait::st. kQuantL (step 3): also
// returns the FP32 sum of the E4M3 values just stored; otherwise 0.
template <bool kQuantL>
__device__ __forceinline__ float write_p_tmem(uint32_t p_addr,
                                              const float (&s)[kBN]) {
  // TODO(2a) -- lesson section 4.4. Convert the 128 values s[] (already
  // 2^8 p, FP32) into 32 words with pack4_e4m3 (word c = keys 4c .. 4c + 3)
  // and store them at columns p_addr .. p_addr + 31 with
  // lab::tmem_st_32x32b_x16 (uint32_t overload), 16 columns at a time.
  // Finally lab::tmem_wait_st().
  //
  // TODO(3) -- lesson section 3.5 (only when kQuantL): also return the
  // FP32 sum of the E4M3 values just stored (add_e4m3x4 adds the four
  // values of one word to an FP32 accumulator; use a few independent
  // partial sums). Leave the step-2 path returning 0.
  (void)p_addr; (void)s;
  LAB_TODO(2);
  if constexpr (kQuantL) {
    LAB_TODO(3);
  }
  return 0.f;
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

// Epilogue for row r of the tile starting at q0: the real O row (TODO(2c))
// staged as BF16 in `so` ([D/64 panels][128 rows][64], 128-byte swizzle),
// each warp TMA-stores its 32 rows; the thread writes the LSE (TODO(2c)).
template <class C>
__device__ __forceinline__ void
store_o_and_lse(const CUtensorMap *o_map, bf16 *so, uint32_t o_addr, int r,
                int head, int batch, int q0, float m, float l, float lo,
                const Params &p) {
  const int warp = r / 32, lane = r % 32, row = q0 + r;
  // TODO(2c) -- lesson section 3.3. f converts the FP32 accumulator O into
  // the real output: O was accumulated from 2^8-scaled P and raw V8, and lo
  // carries the same 2^8 (it is >= 2^8: the row maximum contributes 2^8).
  float f = 0.f;
  LAB_TODO(2);
#pragma unroll
  for (int c = 0; c < C::kD; c += 32) {
    float o[32];
    lab::tmem_ld_32x32b_x32_sync(o_addr + c, o);
    bf16 *dst = so + (c / kOPanel) * kBM * kOPanel + r * kOPanel;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int chunk = (c % kOPanel) / 8 + q;
      uint4 v;
      v.x = pack2_bf16(o[8 * q + 0] * f, o[8 * q + 1] * f);
      v.y = pack2_bf16(o[8 * q + 2] * f, o[8 * q + 3] * f);
      v.z = pack2_bf16(o[8 * q + 4] * f, o[8 * q + 5] * f);
      v.w = pack2_bf16(o[8 * q + 6] * f, o[8 * q + 7] * f);
      *reinterpret_cast<uint4 *>(dst + (chunk ^ (r % 8)) * 8) = v;
    }
  }
  lab::fence_proxy_async(); // generic O writes -> visible to the TMA store
  __syncwarp();
  if (lane == 0) {
    for (int pn = 0; pn < C::kD / kOPanel; ++pn)
      lab::tma_store_4d(o_map, so + pn * kBM * kOPanel + warp * 32 * kOPanel,
                        pn * kOPanel, head, q0 + warp * 32, batch);
    lab::bulk_commit_group();
    lab::bulk_wait_group_read<0>(); // smem read before reuse / exit
  }
  __syncwarp(); // the next tile reuses these 32 staging rows
  if (row < p.n) {
    // TODO(2c, continued): LSE of the REAL scores from the raw maximum m
    // and the FP32 sum l (which also carries 2^8). ln 2 = 0.6931471805599453.
    (void)m; (void)l;
    LAB_TODO(2);
  }
}

// ===========================================================================
// The kernel (lab 40 step 3 with the FP8 changes marked "FP8").
// ===========================================================================
// The 1024-byte aligned start of the dynamic shared memory. The empty asm
// makes the result opaque, so a call after setmaxnreg.dec recomputes it (a
// few integer instructions) instead of keeping the prologue's value alive:
// with 48 registers ptxas otherwise spilled that one value to local memory.
__device__ __forceinline__ uint8_t *align_smem(uint8_t *raw) {
  uint32_t a = lab::shared_addr(raw);
  asm volatile("" : "+r"(a));
  return raw + ((1024 - (a & 1023)) & 1023);
}

template <class C, bool kCausal>
__global__ void __launch_bounds__(C::kThreads, 1)
    fa4_fp8_forward(const __grid_constant__ CUtensorMap q_map,
                    const __grid_constant__ CUtensorMap k_map,
                    const __grid_constant__ CUtensorMap v_map,
                    const __grid_constant__ CUtensorMap o_map, Params p) {
  constexpr int S = C::kStages;
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = align_smem(smem_raw);
  auto sq = [&](int t) { // Q^t (FP8)
    return reinterpret_cast<fp8 *>(smem + C::kQOff + t * C::kQBytes);
  };
  auto ring = [&](int s) {
    return reinterpret_cast<fp8 *>(smem + C::kRingOff + s * C::kKVBytes);
  };
  Stats &st = *reinterpret_cast<Stats *>(smem + C::kStatsOff);
  __shared__ Barriers<S> bar;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int q0 = blockIdx.x * (kTiles * kBM), head = blockIdx.y,
            batch = blockIdx.z;
  const int n = p.n;
  const int blocks = num_kv_blocks<kCausal>(q0, n);

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
  const uint32_t lane_addr =
      tmem + (static_cast<uint32_t>(32 * (warp % 4)) << 16);

  if (warp >= C::kMmaWarp) {
    // ============================================ MMA, TMA (and idle) warps
    lab::setmaxnreg_dec<C::kRegOther>();
    smem = align_smem(smem_raw); // recomputed: see align_smem
    if (warp == C::kLoadWarp && lane == 0) {
      // ------------------------------------------------------ TMA load
      lab::prefetch_tensor_map(&q_map);
      lab::prefetch_tensor_map(&k_map);
      lab::prefetch_tensor_map(&v_map);
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_arrive_expect_tx(&bar.q_full[t], C::kQBytes);
        lab::tma_load_4d(sq(t), &q_map, &bar.q_full[t], 0, head, q0 + t * kBM,
                         batch); // FP8: one box of [128 rows][D bytes]
      }
      for (int u = 0; u < 2 * blocks; ++u) {
        const int s = u % S;
        lab::mbar_wait(&bar.kv_empty[s], ((u / S) & 1) ^ 1);
        lab::mbar_arrive_expect_tx(&bar.kv_full[s], C::kKVBytes);
        const CUtensorMap *map = (u & 1) ? &v_map : &k_map;
        lab::tma_load_4d(ring(s), map, &bar.kv_full[s], 0, head, (u / 2) * kBN,
                         batch);
      }
      for (int u = 2 * blocks; u < 2 * blocks + S; ++u)
        lab::mbar_wait(&bar.kv_empty[u % S], ((u / S) & 1) ^ 1);
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

      const fp8 *k0 = acquire(0);
#pragma unroll 1
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.q_full[t], 0);
        lab::tcgen05_fence_after_thread_sync();
        if (leader) {
          mma_qk<C>(s_col(t), sq(t), k0);
          lab::tcgen05_commit(&bar.s_full[t]);
        }
        __syncwarp();
      }
      if (leader)
        lab::tcgen05_commit(&bar.kv_empty[0]);
      for (int j = 0; j < blocks; ++j) {
        const bool more = j + 1 < blocks;
        const fp8 *v = acquire(2 * j + 1);
        const fp8 *k = more ? acquire(2 * j + 2) : nullptr;
#pragma unroll 1
        for (int t = 0; t < kTiles; ++t) {
          lab::mbar_wait(&bar.p_ready[t], j & 1);
          lab::tcgen05_fence_after_thread_sync();
          if (leader) {
            if constexpr (!C::kProbe) // step 1 stops after Q K^T
              mma_pv_ts<C>(o_col(t), s_col(t) + kPCol, v, j > 0);
            if (t == kTiles - 1)
              lab::tcgen05_commit(&bar.kv_empty[(2 * j + 1) % S]);
            if (more) {
              mma_qk<C>(s_col(t), sq(t), k);
              lab::tcgen05_commit(&bar.s_full[t]);
            } else {
              lab::tcgen05_commit(&bar.o_full[t]);
            }
          }
          __syncwarp();
        }
        if (more && leader)
          lab::tcgen05_commit(&bar.kv_empty[(2 * j + 2) % S]);
      }
    }
  } else if (warp < 4 * kTiles) {
    // ================================================ softmax warpgroups
    lab::setmaxnreg_inc<C::kRegSoftmax>();
    const int t = warp / 4;
    const int r = threadIdx.x % kBM;
    const int qt = q0 + t * kBM;
    const int row = qt + r;
    const uint32_t s_addr = lane_addr + t * kBN;
    float m = -INFINITY; // running max of the raw scores Q8 . K8
    float l = 0.f;  // running sum of p = 2^8 exp(qk_scale (S - m)), FP32: LSE
    float lo = 0.f; // running normalizer of O: l (step 2), or the sum of
                    // the E4M3 values P V really used (step 3)
    for (int j = 0; j < blocks; ++j) {
      lab::mbar_wait(&bar.s_full[t], j & 1);
      lab::tcgen05_fence_after_thread_sync();
      float s[kBN];
      load_s_row(s_addr, 0, s);
      const int k0 = j * kBN;
      if constexpr (C::kProbe)
        write_probe(p, batch, head, row, k0, s);
      if (block_needs_mask<kCausal>(k0, qt, n)) {
        const int limit = visible_keys<kCausal>(row, k0, n);
#pragma unroll
        for (int c = 0; c < kBN; ++c)
          if (c >= limit)
            s[c] = -INFINITY;
      }
      const float m_new = fmaxf(m, row_max(s));
      const float base = m_new == -INFINITY ? 0.f : m_new * p.scale_log2;
      const float alpha = exp2_approx(m * p.scale_log2 - base);
      m = m_new;
      st.alpha[t][r] = alpha;
      lab::mbar_arrive(&bar.stats_full[t]);
      // FP8: p = 2^(s scale_log2 - base + 8) in (0, 256].
      const float sum = exp_and_sum(s, p.scale_log2, base - kPLog2);
      l = l * alpha + sum;
      if constexpr (!C::kProbe) {
        const float qsum = write_p_tmem<C::kQuantL>(s_addr + kPCol, s);
        lo = lo * alpha + (C::kQuantL ? qsum : sum);
      }
      lab::tcgen05_fence_before_thread_sync();
      lab::mbar_arrive(&bar.p_ready[t]);
    }
    st.fin_m[t][r] = m;
    st.fin_l[t][r] = l;
    st.fin_lo[t][r] = lo;
    lab::mbar_arrive(&bar.final_full[t]);
  } else {
    // =============================================== correction warpgroup
    lab::setmaxnreg_dec<C::kRegCorrection>();
    const int r = threadIdx.x % kBM;
    for (int j = 0; j < blocks; ++j) {
      for (int t = 0; t < kTiles; ++t) {
        lab::mbar_wait(&bar.stats_full[t], j & 1);
        lab::tcgen05_fence_after_thread_sync();
        if (j > 0 && !C::kProbe)
          rescale_o_row<C>(lane_addr + C::kOCol + t * C::kD, st.alpha[t][r]);
        lab::tcgen05_fence_before_thread_sync();
        lab::mbar_arrive(&bar.p_ready[t]);
      }
    }
    for (int t = 0; t < kTiles; ++t) {
      lab::mbar_wait(&bar.final_full[t], 0);
      lab::mbar_wait(&bar.o_full[t], 0); // also: every MMA has completed
      lab::tcgen05_fence_after_thread_sync();
      if constexpr (!C::kProbe)
        // FP8: one BF16 O tile over Q^0 + Q^1 (free once o_full[0] has
        // fired: every QK^t was issued before the last PV^0).
        store_o_and_lse<C>(&o_map, reinterpret_cast<bf16 *>(sq(0)),
                           lane_addr + C::kOCol + t * C::kD, r, head, batch,
                           q0 + t * kBM, st.fin_m[t][r], st.fin_l[t][r],
                           st.fin_lo[t][r], p);
    }
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

// FP8 [B, N, H, D] as a 4-D tensor map of bytes: box [1][rows][1][D], i.e.
// one D-byte row per (b, i, h), with the swizzle span equal to the row.
CUtensorMap fp8_map(const void *base, const lab::AttentionProblem &p,
                    uint32_t box_rows) {
  const CUtensorMapSwizzle swz =
      p.d == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
  return lab::make_tensor_map_4d(base, CU_TENSOR_MAP_DATA_TYPE_UINT8, p.b, p.n,
                                 p.h, p.d, 1, box_rows, 1, p.d, swz);
}

// BF16 O: box [1][32][1][64], 128-byte swizzle (lab 40).
CUtensorMap o_map_bf16(const lab::AttentionProblem &p) {
  return lab::make_tensor_map_4d(p.o, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.b,
                                 p.n, p.h, p.d, 1, 32, 1, kOPanel,
                                 CU_TENSOR_MAP_SWIZZLE_128B);
}

struct Maps {
  CUtensorMap q, k, v, o;
};

template <int D, int Step, bool kCausal>
void launch(const Maps &maps, const lab::AttentionProblem &pr,
            cudaStream_t stream, bool setup) {
  using C = Config<D, Step>;
  auto kernel = fa4_fp8_forward<C, kCausal>;
  if (setup) {
    LAB_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    return;
  }
  Params prm{};
  prm.n = pr.n;
  prm.h = pr.h;
  // TODO(1a) -- lesson section 3.2. The MMA computes the raw Q8 . K8 (exact
  // in FP32). Fold the descales into ONE factor: prm.qk_scale converts a
  // raw score into scale * (real Q) . (real K); prm.scale_log2 is the same
  // factor times log2(e) (the softmax uses exp2). Inputs: pr.scale,
  // pr.q_descale, pr.k_descale.
  LAB_TODO(1);
  prm.v_descale = pr.v_descale;
  prm.lse = pr.lse;
  prm.probe = pr.probe;
  const dim3 grid((pr.n + kTiles * kBM - 1) / (kTiles * kBM), pr.h, pr.b);
  kernel<<<grid, C::kThreads, C::kSmemBytes, stream>>>(maps.q, maps.k, maps.v,
                                                       maps.o, prm);
  LAB_CUDA_CHECK(cudaGetLastError());
}

template <int D>
void dispatch_step(const Maps &m, const lab::AttentionProblem &p,
                   cudaStream_t s, bool setup) {
  if (p.step == 1)
    p.causal ? launch<D, 1, true>(m, p, s, setup)
             : launch<D, 1, false>(m, p, s, setup);
  else if (p.step == 2)
    p.causal ? launch<D, 2, true>(m, p, s, setup)
             : launch<D, 2, false>(m, p, s, setup);
  else
    p.causal ? launch<D, 3, true>(m, p, s, setup)
             : launch<D, 3, false>(m, p, s, setup);
}

void dispatch(const Maps &m, const lab::AttentionProblem &p, cudaStream_t s,
              bool setup) {
  if (p.d == 64)
    dispatch_step<64>(m, p, s, setup);
  else
    dispatch_step<128>(m, p, s, setup);
}

} // namespace

struct Plan {
  Maps maps;
};

Plan make_plan(const lab::AttentionProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::E4M3, "this lab uses E4M3 Q/K/V");
  LAB_REQUIRE(p.o_dtype == lab::DType::BF16, "O is BF16");
  LAB_REQUIRE(p.d == 64 || p.d == 128, "D must be 64 or 128");
  LAB_REQUIRE(p.n > 0 && p.b > 0 && p.h > 0, "empty problem");
  LAB_REQUIRE(p.step >= 1 && p.step <= 3, "steps 1-3");
  LAB_REQUIRE(p.step != 1 || p.probe != nullptr, "step 1 writes the probe");
  Plan plan{};
  plan.maps.q = fp8_map(p.q, p, kBM);
  plan.maps.k = fp8_map(p.k, p, kBN);
  plan.maps.v = fp8_map(p.v, p, kBN);
  plan.maps.o = o_map_bf16(p);
  dispatch(plan.maps, p, nullptr, true);
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  dispatch(plan.maps, p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
