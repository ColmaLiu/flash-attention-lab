// Reference solution for 12_fa1: FlashAttention-1 forward (paper Algorithm 1)
// on SIMT FP32 cores.
//
// Layout: q, k, v, o are [B, N, H, D] FP32; row n of head (b, h) starts at
// ((b * N + n) * H + h) * D, so consecutive rows of one head are H * D floats
// apart. lse is [B, H, N] FP32 (natural log). D is 64 or 128.
//
// Parallelism (as in the original FA1 kernel): one CTA per (batch, head).
// Inside the CTA the loops follow Algorithm 1 exactly:
//
//   initialize O = 0, l = 0, m = -inf in HBM                    (line 2)
//   for j over K/V blocks (Bc rows):                            (line 5)
//     load K_j, V_j into shared memory                          (line 6)
//     for i over Q blocks (Br rows):                            (line 7)
//       load Q_i (smem) and O_i, l_i, m_i (registers) from HBM  (line 8)
//       S = scale * Q_i K_j^T                                   (line 9)
//       m~ = rowmax(S), P~ = exp(S - m~), l~ = rowsum(P~)       (line 10)
//       m_new = max(m_i, m~)
//       l_new = e^(m_i - m_new) l_i + e^(m~ - m_new) l~         (line 11)
//       O_i = (l_i e^(m_i - m_new) O_i + e^(m~ - m_new) P~ V_j) / l_new
//       write O_i, l_new, m_new back to HBM                     (lines 12-13)
//   lse = m + log(l)
//
// step 1: N <= Bc, a single K/V block. There is nothing to merge, so the tile
//         result is written directly: O = P~ V / l~, lse = m~ + log(l~).
// step 2: the full outer K/V loop with O, l, m resident in HBM.
// step 3: causal mask, and skipping Q blocks that are fully masked.
#include "lab/lab.h"

#include <cmath>

namespace {

constexpr int kBr = 64;       // rows of a Q block
constexpr int kBc = 64;       // rows of a K/V block
constexpr int kThreads = 256; // 16 x 16 threads, each owns a 4 x 4 S tile
constexpr int kPad = 4;       // keeps float4 alignment, shifts rows by 16 banks

// Thread (ty, tx) = (tid / 16, tid % 16) owns
//   S rows  ty*4 .. ty*4+3   and S columns  tx*4 .. tx*4+3,
//   O rows  ty*4 .. ty*4+3   and O columns  g*64 + tx*4 .. +3, g < D/64.
// The 16 threads of one S row are 16 consecutive lanes of one warp, so row
// reductions are 4 shuffle steps (xor 1, 2, 4, 8).
template <int D> struct Smem {
  static constexpr int kKtStride = kBc + kPad; // Kt[D][Bc+4]: K transposed
  static constexpr int kVStride = D;           // Vs[Bc][D]
  static constexpr int kQStride = D + kPad;    // Qs[Br][D+4]
  static constexpr int kPStride = kBc + kPad;  // Ps[Br][Bc+4]
  static constexpr int kKt = 0;
  static constexpr int kV = kKt + D * kKtStride;
  static constexpr int kQ = kV + kBc * kVStride;
  static constexpr int kP = kQ + kBr * kQStride;
  static constexpr int kFloats = kP + kBr * kPStride;
  static constexpr size_t kBytes = kFloats * sizeof(float);
};

struct Params {
  const float *q, *k, *v;
  float *o, *lse;
  float *m, *l; // [B, H, N] scratch: running max and running sum
  int n, h;
  float scale;
  bool causal;
};

__device__ __forceinline__ float group16_max(float x) {
#pragma unroll
  for (int off = 1; off < 16; off <<= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
  return x;
}

__device__ __forceinline__ float group16_sum(float x) {
#pragma unroll
  for (int off = 1; off < 16; off <<= 1)
    x += __shfl_xor_sync(0xffffffffu, x, off);
  return x;
}

// K_j -> Kt (transposed, so a thread reads 4 consecutive keys with a float4),
// V_j -> Vs (row-major). Rows at or beyond N are zero: they must not inject
// NaN into P V (0 * NaN = NaN), and their scores are masked anyway.
template <int D>
__device__ void load_kv_tile(float *smem, const float *k, const float *v,
                             int row_stride, int key0, int n) {
  float *kt = smem + Smem<D>::kKt;
  float *vs = smem + Smem<D>::kV;
  constexpr int kVecPerRow = D / 4;
  for (int idx = threadIdx.x; idx < kBc * kVecPerRow; idx += kThreads) {
    int c = idx / kVecPerRow, d4 = (idx % kVecPerRow) * 4;
    float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
    if (key0 + c < n) {
      int64_t off = static_cast<int64_t>(key0 + c) * row_stride + d4;
      kv = *reinterpret_cast<const float4 *>(k + off);
      vv = *reinterpret_cast<const float4 *>(v + off);
    }
    kt[(d4 + 0) * Smem<D>::kKtStride + c] = kv.x;
    kt[(d4 + 1) * Smem<D>::kKtStride + c] = kv.y;
    kt[(d4 + 2) * Smem<D>::kKtStride + c] = kv.z;
    kt[(d4 + 3) * Smem<D>::kKtStride + c] = kv.w;
    *reinterpret_cast<float4 *>(vs + c * Smem<D>::kVStride + d4) = vv;
  }
}

template <int D>
__device__ void load_q_tile(float *smem, const float *q, int row_stride,
                            int row0, int n) {
  float *qs = smem + Smem<D>::kQ;
  constexpr int kVecPerRow = D / 4;
  for (int idx = threadIdx.x; idx < kBr * kVecPerRow; idx += kThreads) {
    int r = idx / kVecPerRow, d4 = (idx % kVecPerRow) * 4;
    float4 qv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (row0 + r < n)
      qv = *reinterpret_cast<const float4 *>(
          q + static_cast<int64_t>(row0 + r) * row_stride + d4);
    *reinterpret_cast<float4 *>(qs + r * Smem<D>::kQStride + d4) = qv;
  }
}

// s[rr][cc] = Q_i[ty*4+rr, :] . K_j[tx*4+cc, :]   (no scale, no mask)
// The register-tiled SIMT GEMM from 10_simt_gemm: per 4 k-steps a thread
// reads 4 float4 of Q (broadcast across the row's 16 threads) and 4 float4
// of Kt, then does 64 FMAs.
template <int D>
__device__ __forceinline__ void tile_scores(const float *smem, float s[4][4]) {
  const float *qs = smem + Smem<D>::kQ;
  const float *kt = smem + Smem<D>::kKt;
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr)
#pragma unroll
    for (int cc = 0; cc < 4; ++cc)
      s[rr][cc] = 0.f;
#pragma unroll
  for (int k = 0; k < D; k += 4) {
    float4 a[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
      a[rr] = *reinterpret_cast<const float4 *>(
          qs + (ty * 4 + rr) * Smem<D>::kQStride + k);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      float4 b = *reinterpret_cast<const float4 *>(
          kt + (k + kk) * Smem<D>::kKtStride + tx * 4);
#pragma unroll
      for (int rr = 0; rr < 4; ++rr) {
        float av = kk == 0   ? a[rr].x
                   : kk == 1 ? a[rr].y
                   : kk == 2 ? a[rr].z
                             : a[rr].w;
        s[rr][0] = fmaf(av, b.x, s[rr][0]);
        s[rr][1] = fmaf(av, b.y, s[rr][1]);
        s[rr][2] = fmaf(av, b.z, s[rr][2]);
        s[rr][3] = fmaf(av, b.w, s[rr][3]);
      }
    }
  }
}

// P~ (registers) -> Ps, row-major.
template <int D>
__device__ __forceinline__ void store_p(float *smem, const float p[4][4]) {
  float *ps = smem + Smem<D>::kP;
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr)
    *reinterpret_cast<float4 *>(ps + (ty * 4 + rr) * Smem<D>::kPStride +
                                tx * 4) =
        make_float4(p[rr][0], p[rr][1], p[rr][2], p[rr][3]);
}

// acc[rr][g*4+e] = sum_c P~[ty*4+rr, c] * V_j[c, g*64 + tx*4 + e]
template <int D>
__device__ __forceinline__ void tile_pv(const float *smem,
                                        float acc[4][D / 16]) {
  const float *ps = smem + Smem<D>::kP;
  const float *vs = smem + Smem<D>::kV;
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr)
#pragma unroll
    for (int e = 0; e < D / 16; ++e)
      acc[rr][e] = 0.f;
#pragma unroll
  for (int c = 0; c < kBc; c += 4) {
    float4 p[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
      p[rr] = *reinterpret_cast<const float4 *>(
          ps + (ty * 4 + rr) * Smem<D>::kPStride + c);
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
#pragma unroll
      for (int g = 0; g < D / 64; ++g) {
        float4 vv = *reinterpret_cast<const float4 *>(
            vs + (c + cc) * Smem<D>::kVStride + g * 64 + tx * 4);
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
          float pv = cc == 0   ? p[rr].x
                     : cc == 1 ? p[rr].y
                     : cc == 2 ? p[rr].z
                               : p[rr].w;
          acc[rr][g * 4 + 0] = fmaf(pv, vv.x, acc[rr][g * 4 + 0]);
          acc[rr][g * 4 + 1] = fmaf(pv, vv.y, acc[rr][g * 4 + 1]);
          acc[rr][g * 4 + 2] = fmaf(pv, vv.z, acc[rr][g * 4 + 2]);
          acc[rr][g * 4 + 3] = fmaf(pv, vv.w, acc[rr][g * 4 + 3]);
        }
      }
    }
  }
}

// Per-head view of the problem, set up once at the start of the kernel.
// Row `row` of this head's Q/K/V/O starts at  q + row * row_stride.
struct Head {
  const float *q, *k, *v;
  float *o;
  float *m, *l; // running max / running sum of this head, N floats each
  float *lse;   // N floats
  int n, row_stride;
};

// ---- Algorithm 1, line 2: O = 0, l = 0, m = -inf in HBM -------------------
// The CTA owns every row of its head, so after this a __syncthreads() is
// enough before anyone reads the state.
template <int D> __device__ void init_state(const Head &hd) {
  for (int64_t idx = threadIdx.x; idx < static_cast<int64_t>(hd.n) * (D / 4);
       idx += kThreads) {
    int64_t r = idx / (D / 4);
    int d4 = static_cast<int>(idx % (D / 4)) * 4;
    *reinterpret_cast<float4 *>(hd.o + r * hd.row_stride + d4) =
        make_float4(0.f, 0.f, 0.f, 0.f);
  }
  for (int r = threadIdx.x; r < hd.n; r += kThreads) {
    hd.m[r] = -INFINITY;
    hd.l[r] = 0.f;
  }
}

// ---- Algorithm 1, line 8 (O_i, l_i, m_i) ----------------------------------
// Each thread loads the O values it owns (rows ty*4+rr, columns
// g*64 + tx*4 .. +3) and m, l of its 4 rows into registers.
template <int D>
__device__ __forceinline__ void load_state(const Head &hd, int row0,
                                           float o_old[4][D / 16],
                                           float m_old[4], float l_old[4]) {
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    int row = row0 + ty * 4 + rr;
    if (row >= hd.n)
      continue;
#pragma unroll
    for (int g = 0; g < D / 64; ++g) {
      float4 x = *reinterpret_cast<const float4 *>(
          hd.o + static_cast<int64_t>(row) * hd.row_stride + g * 64 + tx * 4);
      o_old[rr][g * 4 + 0] = x.x;
      o_old[rr][g * 4 + 1] = x.y;
      o_old[rr][g * 4 + 2] = x.z;
      o_old[rr][g * 4 + 3] = x.w;
    }
    m_old[rr] = hd.m[row];
    l_old[rr] = hd.l[row];
  }
}

// ---- Algorithm 1, line 10: block-local softmax statistics -----------------
// In:  s = raw dot products of the tile; (row0, col0) = (first query, first
//      key) of the tile.
// Out: s = P~ = exp(S - m~) with S = scale * s (0 where masked),
//      m_t = m~ and l_t = l~ of the thread's 4 rows (same value in all 16
//      threads of a row).
// A row whose scores are all masked gets m~ = -inf, P~ = 0, l~ = 0: an empty
// key set. With Br == Bc and causal skipping this never happens for a real
// row, but the guard is free and keeps other block sizes safe.
__device__ __forceinline__ void softmax_tile(float s[4][4], int row0, int col0,
                                             int n, float scale, bool causal,
                                             float m_t[4], float l_t[4]) {
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    int row = row0 + ty * 4 + rr;
    float mx = -INFINITY;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      int col = col0 + tx * 4 + cc;
      float x = s[rr][cc] * scale;
      if (col >= n) // keys past the end of the sequence are padding
        x = -INFINITY;
      if (causal && col > row) // query `row` sees keys 0..row only
        x = -INFINITY;
      s[rr][cc] = x;
      mx = fmaxf(mx, x);
    }
    mx = group16_max(mx);
    // exp(-inf - (-inf)) would be NaN; an all-masked row uses 0 as the base.
    float base = mx == -INFINITY ? 0.f : mx;
    float sum = 0.f;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      s[rr][cc] = expf(s[rr][cc] - base);
      sum += s[rr][cc];
    }
    m_t[rr] = mx;
    l_t[rr] = group16_sum(sum);
  }
}

// ---- step 1 only: the whole sequence is one K/V block ---------------------
// (m~, l~, P~ V) already describe all keys: O = P~ V / l~, lse = m~ + log l~.
template <int D>
__device__ __forceinline__ void
write_single_block(const Head &hd, int row0, const float acc[4][D / 16],
                   const float m_t[4], const float l_t[4]) {
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    int row = row0 + ty * 4 + rr;
    if (row >= hd.n)
      continue;
    float inv = 1.f / l_t[rr];
    float *orow = hd.o + static_cast<int64_t>(row) * hd.row_stride + tx * 4;
#pragma unroll
    for (int g = 0; g < D / 64; ++g)
      *reinterpret_cast<float4 *>(orow + g * 64) =
          make_float4(acc[rr][g * 4 + 0] * inv, acc[rr][g * 4 + 1] * inv,
                      acc[rr][g * 4 + 2] * inv, acc[rr][g * 4 + 3] * inv);
    if (tx == 0)
      hd.lse[row] = m_t[rr] + logf(l_t[rr]);
  }
}

// ---- Algorithm 1, lines 11-13: merge and write back -----------------------
// Old state (m_i, l_i, O_i) and tile state (m~, l~, P~ V_j), both relative to
// their own max, are moved to the common scale m_new and added:
//   l_new = e^(m_i - m_new) l_i + e^(m~ - m_new) l~
//   O_i   = (l_i e^(m_i - m_new) O_i + e^(m~ - m_new) P~ V_j) / l_new
// O stays normalized in HBM after every K/V block, as in the paper.
template <int D>
__device__ __forceinline__ void
merge_state(const Head &hd, int row0, const float o_old[4][D / 16],
            const float m_old[4], const float l_old[4],
            const float acc[4][D / 16], const float m_t[4],
            const float l_t[4]) {
  int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    int row = row0 + ty * 4 + rr;
    if (row >= hd.n)
      continue;
    float m_new = fmaxf(m_old[rr], m_t[rr]);
    if (m_new == -INFINITY)
      continue; // no visible key so far and none in this tile
    float a = expf(m_old[rr] - m_new); // 0 when m_old = -inf (first block)
    float c = expf(m_t[rr] - m_new);   // 0 when m~ = -inf (all masked)
    float l_new = a * l_old[rr] + c * l_t[rr];
    float inv = 1.f / l_new;
    float w_old = a * l_old[rr] * inv, w_new = c * inv;
    float *orow = hd.o + static_cast<int64_t>(row) * hd.row_stride + tx * 4;
#pragma unroll
    for (int g = 0; g < D / 64; ++g)
      *reinterpret_cast<float4 *>(orow + g * 64) = make_float4(
          w_old * o_old[rr][g * 4 + 0] + w_new * acc[rr][g * 4 + 0],
          w_old * o_old[rr][g * 4 + 1] + w_new * acc[rr][g * 4 + 1],
          w_old * o_old[rr][g * 4 + 2] + w_new * acc[rr][g * 4 + 2],
          w_old * o_old[rr][g * 4 + 3] + w_new * acc[rr][g * 4 + 3]);
    if (tx == 0) { // one writer per row; all 16 threads hold the same value
      hd.m[row] = m_new;
      hd.l[row] = l_new;
    }
  }
}

// ---- after the last K/V block: lse = m + log(l) ---------------------------
__device__ void write_lse(const Head &hd) {
  for (int r = threadIdx.x; r < hd.n; r += kThreads)
    hd.lse[r] = hd.m[r] + logf(hd.l[r]);
}

// ---- causal: first Q block that can see K/V block j -----------------------
// Q block i holds queries i*Br .. i*Br+Br-1. It sees some key of block j
// (keys j*Bc ..) iff its last query >= j*Bc. Earlier Q blocks are fully
// masked and are skipped: no loads, no math, no write-back.
__device__ __forceinline__ int first_q_block(int j, bool causal) {
  return causal ? (j * kBc) / kBr : 0;
}

template <int D, bool kSingleBlock>
__global__ void __launch_bounds__(kThreads) fa1_forward(const Params prm) {
  extern __shared__ __align__(16) float smem[];
  const int n = prm.n;
  const int bh = blockIdx.x; // one CTA per (batch, head)
  const int b = bh / prm.h, h = bh % prm.h;
  const int row_stride = prm.h * D;
  const int64_t head_off =
      static_cast<int64_t>(b) * n * row_stride + static_cast<int64_t>(h) * D;
  const Head hd{prm.q + head_off,
                prm.k + head_off,
                prm.v + head_off,
                prm.o + head_off,
                prm.m + static_cast<int64_t>(bh) * n,
                prm.l + static_cast<int64_t>(bh) * n,
                prm.lse + static_cast<int64_t>(bh) * n,
                n,
                row_stride};
  const int t_r = (n + kBr - 1) / kBr; // number of Q blocks
  const int t_c = (n + kBc - 1) / kBc; // number of K/V blocks

  if constexpr (!kSingleBlock) {
    init_state<D>(hd); // line 2
    // m/l of a row are read by threads that did not write them.
    __syncthreads();
  }

  for (int j = 0; j < t_c; ++j) { // line 5: outer loop over K/V blocks
    const int key0 = j * kBc;
    __syncthreads(); // the previous tile_pv is done with Vs
    load_kv_tile<D>(smem, hd.k, hd.v, row_stride, key0, n); // line 6

    for (int i = first_q_block(j, prm.causal); i < t_r; ++i) { // line 7
      const int row0 = i * kBr;
      __syncthreads(); // Qs and Ps of the previous tile are no longer read
      load_q_tile<D>(smem, hd.q, row_stride, row0, n); // line 8: Q_i

      // Line 8: O_i, l_i, m_i into registers. Reading them here, before the
      // barriers below, guarantees every thread of a row has its old m/l
      // before the row's tx == 0 thread overwrites them in merge_state.
      float o_old[4][D / 16], m_old[4], l_old[4];
      if constexpr (!kSingleBlock)
        load_state<D>(hd, row0, o_old, m_old, l_old);
      // Q_i (and, on the first i of each j, K_j and V_j) are in smem.
      __syncthreads();

      float s[4][4], m_t[4], l_t[4];
      tile_scores<D>(smem, s); // line 9: S = Q_i K_j^T (scale comes next)
      softmax_tile(s, row0, key0, n, prm.scale, prm.causal, m_t, l_t); // 10

      store_p<D>(smem, s);
      __syncthreads(); // all of P~ is in shared memory
      float acc[4][D / 16];
      tile_pv<D>(smem, acc); // P~ V_j

      if constexpr (kSingleBlock)
        write_single_block<D>(hd, row0, acc, m_t, l_t);
      else
        merge_state<D>(hd, row0, o_old, m_old, l_old, acc, m_t, l_t); // 11-13
    }
  }

  if constexpr (!kSingleBlock) {
    __syncthreads(); // m and l of every row are final and visible
    write_lse(hd);
  }
}

template <int D, bool kSingleBlock>
void launch(const Params &prm, int grid, cudaStream_t stream) {
  fa1_forward<D, kSingleBlock>
      <<<grid, kThreads, Smem<D>::kBytes, stream>>>(prm);
}

// Shared memory above 48 KB per CTA must be requested explicitly.
template <int D> void allow_smem() {
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      fa1_forward<D, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      Smem<D>::kBytes));
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      fa1_forward<D, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      Smem<D>::kBytes));
}

} // namespace

struct Plan {
  lab::Scratch m, l; // [B, H, N] FP32 each; garbage until the kernel runs
};

Plan make_plan(const lab::AttentionProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F32 && p.o_dtype == lab::DType::F32,
              "this lab uses FP32 Q, K, V and O");
  LAB_REQUIRE(p.d == 64 || p.d == 128, "D must be 64 or 128");
  LAB_REQUIRE(p.step != 1 || p.n <= kBc,
              "step 1 handles a single K/V block only (N <= 64)");
  if (p.d == 64)
    allow_smem<64>();
  else
    allow_smem<128>();
  size_t rows = static_cast<size_t>(p.b) * p.h * p.n;
  Plan plan;
  plan.m = lab::Scratch(rows * sizeof(float));
  plan.l = lab::Scratch(rows * sizeof(float));
  return plan;
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  Params prm{static_cast<const float *>(p.q),
             static_cast<const float *>(p.k),
             static_cast<const float *>(p.v),
             static_cast<float *>(p.o),
             p.lse,
             plan.m.as<float>(),
             plan.l.as<float>(),
             p.n,
             p.h,
             p.scale,
             p.causal};
  int grid = p.b * p.h; // one CTA per (batch, head)
  // step 1: single K/V block, no HBM state. steps 2-3: Algorithm 1.
  bool single = p.step == 1;
  if (p.d == 64)
    single ? launch<64, true>(prm, grid, stream)
           : launch<64, false>(prm, grid, stream);
  else
    single ? launch<128, true>(prm, grid, stream)
           : launch<128, false>(prm, grid, stream);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
