// Reference solution for 22_fa2: FlashAttention-2 forward with mma.sync.
// Lesson: course/22_fa2/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous, row stride H*D elements.
// LSE: [B, H, N] FP32, natural log. D in {64, 128}, scale > 0 finite.
//
// One CTA per (Q block, head, batch); the K/V loop is inside (FA2 paper
// section 3.1/3.2). Each warp owns 16 * kMTiles query rows (section 3.3), so
// row max / row sum never leave the warp: a row lives in the 4 lanes of one
// quad. O, m, l stay in registers for the whole loop and O is divided by l
// once at the end (section 3.1).
//
// step 1: S = Q K^T in registers, written (times scale) to the probe.
// step 2: online softmax in registers, P reused as the A operand of P.V,
//         V read with ldmatrix.trans. Loads are synchronous (issue, wait,
//         compute).
// step 3: causal: K blocks above the diagonal are skipped, only blocks that
//         cross the diagonal (or the end of N) are masked.
// step 4: cp.async double buffering of K/V (lab 21): block j+1 is in flight
//         while block j is computed.
#include "lab/cp_async.cuh"
#include "lab/lab.h"
#include "lab/mma_sync.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

using bf16 = __nv_bfloat16;
using fp16 = __half;

// Tile configuration. kWarps warps, each owning kMTiles 16-row mma tiles,
// so a CTA covers kBM = 16 * kMTiles * kWarps query rows. K/V blocks have
// kBN keys. Shared memory: one Q tile plus kStages (K tile, V tile) pairs,
// all [rows][D] with XOR-swizzled 16-byte chunks.
template <int D_, int Warps, int MTiles, int BN, int Stages> struct Config {
  static constexpr int kD = D_, kWarps = Warps, kMTiles = MTiles;
  static constexpr int kBM = 16 * MTiles * Warps, kBN = BN, kStages = Stages;
  static constexpr int kThreads = Warps * 32;
  static constexpr int kChunks = D_ / 8; // 16-byte chunks per row
  static constexpr int kQElems = kBM * D_, kKVElems = BN * D_;
  static constexpr int kSmemBytes = (kQElems + 2 * Stages * kKVElems) * 2;
  static_assert(D_ % 64 == 0 && BN % 16 == 0);
};

// Element offset of (row, col) in a [rows][D] tile, col % 8 == 0 for the
// chunk address. Each 64-column (128-byte) segment of a row is swizzled on
// its own: banks repeat every 128 bytes, so this is enough to make 8 rows of
// the same logical chunk hit 8 different bank groups.
template <int D> __device__ __forceinline__ int tile_offset(int row, int col) {
  return row * D + (col / 64) * 64 +
         lab::swizzle_chunk<8>(row, col % 64 / 8) * 8 + col % 8;
}

// ---------------------------------------------------------------- dtype glue
template <class T>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2]) {
  if constexpr (std::is_same_v<T, fp16>)
    lab::mma_m16n8k16_f16(d, a, b);
  else
    lab::mma_m16n8k16_bf16(d, a, b);
}

// Two FP32 values -> one 32-bit register of two 16-bit values (x in the low
// half, i.e. the smaller column), rounded to nearest.
template <class T> __device__ __forceinline__ uint32_t pack2(float x, float y) {
  if constexpr (std::is_same_v<T, fp16>) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
  } else {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t *>(&h);
  }
}

// 2^x with the hardware MUFU.EX2 (ex2.approx.ftz.f32: max rel. error 2^-22,
// ex2(-inf) = +0). exp2f() without --use_fast_math is a longer sequence.
__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The 4 lanes of a quad (lane / 4 equal) hold the same two rows of every
// C fragment. XOR 1 and 2 stay inside the quad.
__device__ __forceinline__ float quad_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
__device__ __forceinline__ float quad_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x + __shfl_xor_sync(0xffffffffu, x, 2);
}

// ---------------------------------------------------------------- copies
// cp.async rows row0 .. row0 + kRows - 1 of a [N][stride] tensor (only the
// first D columns) into a swizzled [kRows][D] tile. Rows >= n become zeros:
// a zero K row gives S = 0 (masked later), a zero V row gives P * 0 = 0;
// stale smem could hold NaN, and 0 * NaN = NaN. Does not commit.
template <class C, int kRows, class T>
__device__ __forceinline__ void issue_rows(T *s, const T *g, int row0, int n,
                                           int stride) {
  static_assert(kRows * C::kChunks % C::kThreads == 0);
#pragma unroll
  for (int i = 0; i < kRows * C::kChunks / C::kThreads; ++i) {
    const int idx = threadIdx.x + i * C::kThreads;
    const int r = idx / C::kChunks, col = idx % C::kChunks * 8;
    const bool ok = row0 + r < n;
    const T *src = ok ? g + static_cast<int64_t>(row0 + r) * stride + col : g;
    lab::cp_async_16_zfill(s + tile_offset<C::kD>(r, col), src, ok);
  }
}

// ---------------------------------------------------------------- math
// Q fragments of this warp's rows, loaded once: qf[mi][kk] is the A operand
// for rows w0 + 16 mi .. +15, columns 16 kk .. +15.
template <class C>
__device__ __forceinline__ void
load_q(uint32_t (&qf)[C::kMTiles][C::kD / 16][4], const void *sq, int w0,
       int lane) {
  const auto *s = static_cast<const uint16_t *>(sq);
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
    for (int kk = 0; kk < C::kD / 16; ++kk)
      lab::ldmatrix_x4(qf[mi][kk],
                       s + tile_offset<C::kD>(w0 + mi * 16 + lane % 16,
                                              kk * 16 + lane / 16 * 8));
}

// s[mi][j] (16x8 C fragment) = Q rows (mi) . K rows 8j .. 8j+7 of the block.
// K is [keys][D] = [n][k] for the B operand: K-major, plain ldmatrix (as B
// in labs 20/21). One x4 gives the B fragments of two n8 tiles.
template <class T, class C>
__device__ __forceinline__ void
qk_block(float (&s)[C::kMTiles][C::kBN / 8][4],
         const uint32_t (&qf)[C::kMTiles][C::kD / 16][4], const T *sk,
         int lane) {
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
    for (int j = 0; j < C::kBN / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        s[mi][j][e] = 0.f;
#pragma unroll
  for (int kk = 0; kk < C::kD / 16; ++kk)
#pragma unroll
    for (int j = 0; j < C::kBN / 8; j += 2) {
      uint32_t r[4];
      lab::ldmatrix_x4(r,
                       sk + tile_offset<C::kD>(j * 8 + lane % 8 + lane / 16 * 8,
                                               kk * 16 + (lane / 8) % 2 * 8));
      const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
#pragma unroll
      for (int mi = 0; mi < C::kMTiles; ++mi) {
        mma<T>(s[mi][j], qf[mi][kk], b0);
        mma<T>(s[mi][j + 1], qf[mi][kk], b1);
      }
    }
}

// Set S to -inf where key >= n (tail) or, if causal, key > query.
// C fragment element e of tile j: row group + 8 (e / 2), col 2 quad + e % 2.
template <class C, bool kCausal>
__device__ __forceinline__ void
apply_mask(float (&s)[C::kMTiles][C::kBN / 8][4], int row0, int k0, int n,
           int lane) {
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
    for (int j = 0; j < C::kBN / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = row0 + mi * 16 + lane / 4 + (e / 2) * 8;
        const int key = k0 + j * 8 + (lane % 4) * 2 + e % 2;
        if (key >= n || (kCausal && key > row))
          s[mi][j][e] = -INFINITY;
      }
}

// Online softmax on one block, all in registers. m is kept in raw (unscaled)
// S units: scale > 0, so max(scale * s) = scale * max(s), and
//   exp(scale * (s - m)) = 2^(s * scale_log2 - m * scale_log2).
// l is this lane's partial row sum (its 2 columns of every n8 tile); the
// quad sum is taken only once, in the epilogue: alpha is the same for all
// 4 lanes of the quad, so the partial sums rescale correctly on their own.
// On return s holds P~ = exp(scale * (s - m_new)) and o, l are rescaled.
template <class C>
__device__ __forceinline__ void
online_softmax(float (&s)[C::kMTiles][C::kBN / 8][4],
               float (&o)[C::kMTiles][C::kD / 8][4], float (&m)[C::kMTiles][2],
               float (&l)[C::kMTiles][2], float scale_log2) {
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) { // h = 0: row group, h = 1: row group + 8
      float mx = m[mi][h];
#pragma unroll
      for (int j = 0; j < C::kBN / 8; ++j)
        mx = fmaxf(mx, fmaxf(s[mi][j][2 * h], s[mi][j][2 * h + 1]));
      mx = quad_max(mx);
      // A row that has seen only masked keys keeps m = -inf; subtract 0
      // instead so that (-inf) - (-inf) = NaN never appears.
      const float base = mx == -INFINITY ? 0.f : mx * scale_log2;
      const float alpha = exp2_approx(m[mi][h] * scale_log2 - base);
      m[mi][h] = mx;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < C::kBN / 8; ++j)
#pragma unroll
        for (int e = 2 * h; e < 2 * h + 2; ++e) {
          s[mi][j][e] = exp2_approx(fmaf(s[mi][j][e], scale_log2, -base));
          sum += s[mi][j][e];
        }
      l[mi][h] = l[mi][h] * alpha + sum;
#pragma unroll
      for (int dj = 0; dj < C::kD / 8; ++dj) {
        o[mi][dj][2 * h] *= alpha;
        o[mi][dj][2 * h + 1] *= alpha;
      }
    }
}

// o += P~ . V for one block. The C fragments of two neighbouring n8 tiles of
// P~ (keys 16t .. 16t+15) are exactly one A fragment of m16n8k16:
//   A reg 0 = (row g,   k 2q..2q+1)   = C tile 2t,   floats 0,1
//   A reg 1 = (row g+8, k 2q..2q+1)   = C tile 2t,   floats 2,3
//   A reg 2 = (row g,   k 2q+8..+9)   = C tile 2t+1, floats 0,1
//   A reg 3 = (row g+8, k 2q+8..+9)   = C tile 2t+1, floats 2,3
// so P never goes through shared memory. V is [keys][D] = [k][n]: n is the
// fastest dimension (MN-major), the transpose of what mma's B operand wants;
// ldmatrix.trans transposes each 8x8 matrix on the way.
template <class T, class C>
__device__ __forceinline__ void
pv_block(float (&o)[C::kMTiles][C::kD / 8][4],
         const float (&s)[C::kMTiles][C::kBN / 8][4], const T *sv, int lane) {
#pragma unroll
  for (int t = 0; t < C::kBN / 16; ++t) {
    uint32_t a[C::kMTiles][4];
#pragma unroll
    for (int mi = 0; mi < C::kMTiles; ++mi) {
      a[mi][0] = pack2<T>(s[mi][2 * t][0], s[mi][2 * t][1]);
      a[mi][1] = pack2<T>(s[mi][2 * t][2], s[mi][2 * t][3]);
      a[mi][2] = pack2<T>(s[mi][2 * t + 1][0], s[mi][2 * t + 1][1]);
      a[mi][3] = pack2<T>(s[mi][2 * t + 1][2], s[mi][2 * t + 1][3]);
    }
#pragma unroll
    for (int dj = 0; dj < C::kD / 8; dj += 2) {
      // Matrices: (keys 16t+0..7, d 8dj..), (keys 16t+8..15, d 8dj..),
      //           (keys 16t+0..7, d 8dj+8..), (keys 16t+8..15, d 8dj+8..).
      uint32_t r[4];
      lab::ldmatrix_x4_trans(
          r,
          sv + tile_offset<C::kD>(t * 16 + lane % 16, dj * 8 + lane / 16 * 8));
      const uint32_t b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
#pragma unroll
      for (int mi = 0; mi < C::kMTiles; ++mi) {
        mma<T>(o[mi][dj], a[mi], b0);
        mma<T>(o[mi][dj + 1], a[mi], b1);
      }
    }
  }
}

struct Params {
  const void *q, *k, *v;
  void *o;
  float *lse, *probe;
  int n, h;
  float scale, scale_log2;
};

// ---------------------------------------------------------------- step 1
template <class T, class C>
__global__ void __launch_bounds__(C::kThreads) fa2_scores(Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  T *sq = reinterpret_cast<T *>(smem_raw);
  T *sk = sq + C::kQElems;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int q0 = blockIdx.x * C::kBM, head = blockIdx.y, batch = blockIdx.z;
  const int stride = p.h * C::kD;
  const int64_t base = (static_cast<int64_t>(batch) * p.n * p.h + head) * C::kD;
  const T *q = static_cast<const T *>(p.q) + base;
  const T *k = static_cast<const T *>(p.k) + base;
  const int w0 = warp * 16 * C::kMTiles;

  issue_rows<C, C::kBM>(sq, q + static_cast<int64_t>(q0) * stride, 0, p.n - q0,
                        stride);
  lab::cp_async_commit();
  lab::cp_async_wait_group<0>();
  __syncthreads();
  uint32_t qf[C::kMTiles][C::kD / 16][4];
  load_q<C>(qf, sq, w0, lane);

  float *probe =
      p.probe + (static_cast<int64_t>(batch) * p.h + head) * p.n * p.n;
  const int blocks = (p.n + C::kBN - 1) / C::kBN;
  for (int jb = 0; jb < blocks; ++jb) {
    const int k0 = jb * C::kBN;
    __syncthreads(); // everyone finished reading the previous K block
    issue_rows<C, C::kBN>(sk, k + static_cast<int64_t>(k0) * stride, 0,
                          p.n - k0, stride);
    lab::cp_async_commit();
    lab::cp_async_wait_group<0>();
    __syncthreads();
    float s[C::kMTiles][C::kBN / 8][4];
    qk_block<T, C>(s, qf, sk, lane);
#pragma unroll
    for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
      for (int j = 0; j < C::kBN / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          lab::FragCoord rc = lab::frag_c_coord(lane, e);
          const int row = q0 + w0 + mi * 16 + rc.row, key = k0 + j * 8 + rc.col;
          if (row < p.n && key < p.n)
            probe[static_cast<int64_t>(row) * p.n + key] =
                s[mi][j][e] * p.scale;
        }
  }
}

// ---------------------------------------------------------------- steps 2-4
template <class T, class C, bool kCausal, bool kPipelined>
__global__ void __launch_bounds__(C::kThreads) fa2_forward(Params p) {
  static_assert(!kPipelined || C::kStages == 2);
  extern __shared__ __align__(128) unsigned char smem_raw[];
  T *sq = reinterpret_cast<T *>(smem_raw);
  T *skv = sq + C::kQElems; // stage s: K at skv + 2s kKVElems, V right after
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int q0 = blockIdx.x * C::kBM, head = blockIdx.y, batch = blockIdx.z;
  const int n = p.n, stride = p.h * C::kD;
  const int64_t base = (static_cast<int64_t>(batch) * n * p.h + head) * C::kD;
  const T *q = static_cast<const T *>(p.q) + base;
  const T *k = static_cast<const T *>(p.k) + base;
  const T *v = static_cast<const T *>(p.v) + base;
  const int w0 = warp * 16 * C::kMTiles; // first row of this warp in the CTA

  // K blocks this Q block needs. Causal: the last query row is q0 + kBM - 1,
  // so blocks starting after it are skipped entirely (never loaded).
  int blocks = (n + C::kBN - 1) / C::kBN;
  if constexpr (kCausal)
    blocks = min(blocks, (q0 + C::kBM + C::kBN - 1) / C::kBN);

  auto issue_kv = [&](int jb, int stage) {
    T *sk = skv + stage * 2 * C::kKVElems;
    const int64_t off = static_cast<int64_t>(jb) * C::kBN * stride;
    issue_rows<C, C::kBN>(sk, k + off, 0, n - jb * C::kBN, stride);
    issue_rows<C, C::kBN>(sk + C::kKVElems, v + off, 0, n - jb * C::kBN,
                          stride);
  };

  issue_rows<C, C::kBM>(sq, q + static_cast<int64_t>(q0) * stride, 0, n - q0,
                        stride);
  if constexpr (kPipelined)
    issue_kv(0, 0); // block 0 travels with Q in group 0
  lab::cp_async_commit();
  lab::cp_async_wait_group<0>();
  __syncthreads();
  uint32_t qf[C::kMTiles][C::kD / 16][4];
  load_q<C>(qf, sq, w0, lane); // Q stays in registers for the whole loop

  float o[C::kMTiles][C::kD / 8][4], m[C::kMTiles][2], l[C::kMTiles][2];
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi) {
#pragma unroll
    for (int dj = 0; dj < C::kD / 8; ++dj)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[mi][dj][e] = 0.f;
    m[mi][0] = m[mi][1] = -INFINITY;
    l[mi][0] = l[mi][1] = 0.f;
  }

  for (int jb = 0; jb < blocks; ++jb) {
    int stage = 0;
    if constexpr (kPipelined) {
      // Block jb's group is the newest one committed, so wait for all.
      // The barrier publishes it to every warp, and also means every warp
      // has finished block jb-1, whose stage (jb+1) % 2 is refilled now.
      stage = jb % 2;
      lab::cp_async_wait_group<0>();
      __syncthreads();
      if (jb + 1 < blocks) {
        issue_kv(jb + 1, (jb + 1) % 2);
        lab::cp_async_commit();
      }
    } else {
      issue_kv(jb, 0);
      lab::cp_async_commit();
      lab::cp_async_wait_group<0>();
      __syncthreads();
    }
    const T *sk = skv + stage * 2 * C::kKVElems, *sv = sk + C::kKVElems;

    float s[C::kMTiles][C::kBN / 8][4];
    qk_block<T, C>(s, qf, sk, lane);
    // Only blocks that cross the end of N or the diagonal need a mask: the
    // block's last key k0 + kBN - 1 is beyond the CTA's first query q0.
    const int k0 = jb * C::kBN;
    if (k0 + C::kBN > n || (kCausal && k0 + C::kBN - 1 > q0))
      apply_mask<C, kCausal>(s, q0 + w0, k0, n, lane);
    online_softmax<C>(s, o, m, l, p.scale_log2);
    pv_block<T, C>(o, s, sv, lane);
    if constexpr (!kPipelined)
      __syncthreads(); // all warps done with K/V before the next overwrite
  }

  // Epilogue: finish the row sums, normalize O once, write O and LSE.
  T *o_ptr = static_cast<T *>(p.o) + base;
  float *lse = p.lse + (static_cast<int64_t>(batch) * p.h + head) * n;
#pragma unroll
  for (int mi = 0; mi < C::kMTiles; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sum = quad_sum(l[mi][h]);
      const float inv = 1.f / sum; // sum >= 1: the max element gives 2^0
      const int row = q0 + w0 + mi * 16 + lane / 4 + h * 8;
      if (row >= n)
        continue;
      if (lane % 4 == 0)
        lse[row] = m[mi][h] * p.scale + logf(sum);
      T *dst = o_ptr + static_cast<int64_t>(row) * stride + (lane % 4) * 2;
#pragma unroll
      for (int dj = 0; dj < C::kD / 8; ++dj)
        *reinterpret_cast<uint32_t *>(dst + dj * 8) =
            pack2<T>(o[mi][dj][2 * h] * inv, o[mi][dj][2 * h + 1] * inv);
    }
}

// Per-lane register arrays: Q kMTiles*D/4 32-bit registers (two 16-bit
// values each), O kMTiles*D/2 floats, S kMTiles*kBN/2 floats. Both
// configurations spend 32 + 64 + 32 = 128 registers on them, leaving room
// for addresses and temporaries under the 255-register limit (no spills):
// D = 128: 4 warps x 1 tile = 64 query rows per CTA, kBN = 64.
// D = 64:  4 warps x 2 tiles = 128 query rows per CTA, kBN = 32 (kBN = 64
//          needs 160 registers of arrays and spills in the step 2/3 kernel).
template <int D, int Stages> struct Pick;
template <int Stages> struct Pick<128, Stages> {
  using type = Config<128, 4, 1, 64, Stages>;
};
template <int Stages> struct Pick<64, Stages> {
  using type = Config<64, 4, 2, 32, Stages>;
};

template <class Kernel> void allow_smem(Kernel kernel, int bytes) {
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

template <class T, int D>
void launch(const lab::AttentionProblem &pr, cudaStream_t stream, bool setup) {
  Params p{pr.q,     pr.k, pr.v, pr.o,     pr.lse,
           pr.probe, pr.n, pr.h, pr.scale, pr.scale * 1.4426950408889634f};
  using C1 = typename Pick<D, 1>::type;
  using C2 = typename Pick<D, 2>::type;
  auto grid = [&](int bm) { return dim3((pr.n + bm - 1) / bm, pr.h, pr.b); };
  if (pr.step == 1) {
    if (setup)
      allow_smem(fa2_scores<T, C1>, C1::kSmemBytes);
    else
      fa2_scores<T, C1>
          <<<grid(C1::kBM), C1::kThreads, C1::kSmemBytes, stream>>>(p);
    return;
  }
  auto go = [&](auto kernel, auto cfg) {
    using C = decltype(cfg);
    if (setup)
      allow_smem(kernel, C::kSmemBytes);
    else
      kernel<<<grid(C::kBM), C::kThreads, C::kSmemBytes, stream>>>(p);
  };
  if (pr.step <= 3) {
    if (pr.causal)
      go(fa2_forward<T, C1, true, false>, C1{});
    else
      go(fa2_forward<T, C1, false, false>, C1{});
  } else {
    if (pr.causal)
      go(fa2_forward<T, C2, true, true>, C2{});
    else
      go(fa2_forward<T, C2, false, true>, C2{});
  }
}

void dispatch(const lab::AttentionProblem &p, cudaStream_t s, bool setup) {
  const bool half = p.dtype == lab::DType::F16;
  if (p.d == 64)
    half ? launch<fp16, 64>(p, s, setup) : launch<bf16, 64>(p, s, setup);
  else
    half ? launch<fp16, 128>(p, s, setup) : launch<bf16, 128>(p, s, setup);
}

} // namespace

struct Plan {};

Plan make_plan(const lab::AttentionProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F16 || p.dtype == lab::DType::BF16,
              "this lab uses FP16 or BF16 Q/K/V");
  LAB_REQUIRE(p.o_dtype == p.dtype, "O has the dtype of Q/K/V");
  LAB_REQUIRE(p.d == 64 || p.d == 128, "D must be 64 or 128");
  LAB_REQUIRE(p.n > 0 && p.b > 0 && p.h > 0, "empty problem");
  LAB_REQUIRE(p.step != 1 || p.probe != nullptr, "step 1 writes the probe");
  dispatch(p, nullptr, true);
  return {};
}

void run(const Plan &, const lab::AttentionProblem &p, cudaStream_t stream) {
  dispatch(p, stream, false);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
