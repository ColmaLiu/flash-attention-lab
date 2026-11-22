// 22_fa2: FlashAttention-2 forward with mma.sync and cp.async.
// Lesson: course/22_fa2/README.md
//
// Q, K, V, O: [B, N, H, D] FP16/BF16 contiguous: element (b, i, h, d) is at
//   ((b * N + i) * H + h) * D + d, so consecutive rows i of one head are
//   H * D elements apart ("stride").
// LSE: [B, H, N] FP32, natural log: lse[(b * H + h) * N + i].
// Supported: D in {64, 128}, scale > 0 and finite (checked by the harness).
//
// One CTA per (Q block, head, batch) = (blockIdx.x, .y, .z); the K/V loop is
// inside the CTA. Each warp owns 16 * kMTiles query rows. Shared tiles are
// [rows][D] with XOR-swizzled 16-byte chunks (tile_offset).
// run() picks a version by p.step:
//   step 1: fa2_scores           (S = Q K^T into the probe)
//   step 2, 3: fa2_forward<.., kPipelined = false>  (synchronous loads)
//   step 4: fa2_forward<.., kPipelined = true>      (cp.async double buffer)
//
// Provided headers: include/lab/mma_sync.cuh (ldmatrix, mma, fragment
// coordinates, swizzle_chunk) and include/lab/cp_async.cuh (lab 21).
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
  // -------------------------------------------------------------------------
  // TODO(1a) — fill qf[mi][kk] (mi < kMTiles, kk < D/16) with the A fragment
  // of rows w0 + 16 mi .. +15, columns 16 kk .. +15 of the Q tile in sq
  // (lesson §4.2). One lab::ldmatrix_x4 per fragment, exactly as the A tile
  // in labs 20/21: lane l points at row (l % 16), column chunk (l / 16).
  // Element (row, col) of the tile is at
  //   static_cast<const uint16_t *>(sq) + tile_offset<C::kD>(row, col).
  // -------------------------------------------------------------------------
  LAB_TODO(1);
}

// s[mi][j] (16x8 C fragment) = Q rows (mi) . K rows 8j .. 8j+7 of the block.
// K is [keys][D] = [n][k] for the B operand: K-major, plain ldmatrix (as B
// in labs 20/21). One x4 gives the B fragments of two n8 tiles.
template <class T, class C>
__device__ __forceinline__ void
qk_block(float (&s)[C::kMTiles][C::kBN / 8][4],
         const uint32_t (&qf)[C::kMTiles][C::kD / 16][4], const T *sk,
         int lane) {
  // -------------------------------------------------------------------------
  // TODO(1b) — s[mi][j] = (Q rows of tile mi) . (K rows 8j .. 8j+7)^T, i.e.
  // the 16x8 C fragment of S for keys 8j .. 8j+7 of this block (lesson §4.2).
  // Start from zero. K is [keys][D]: for mma's B operand ([n][k], k fastest)
  // that is K-major, so plain ldmatrix works as for B in labs 20/21; one
  // ldmatrix_x4 gives the B fragments of two n8 tiles (j and j+1).
  // Use mma<T>(d, a, b): it picks the fp16 or bf16 instruction.
  // -------------------------------------------------------------------------
  LAB_TODO(1);
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
        bool masked = key >= n; // padding keys past the end of N
        if constexpr (kCausal) {
          // -----------------------------------------------------------------
          // TODO(3b) — causal: query `row` may only see keys <= row.
          // -----------------------------------------------------------------
          LAB_TODO(3);
        }
        if (masked)
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
  // -------------------------------------------------------------------------
  // TODO(2a) — for each of this lane's rows (mi, h), h = 0: row group,
  // h = 1: row group + 8 (C floats 2h and 2h+1 of every tile j):
  //   1. mx = max(m_old, max over this block's S of the row); the row is
  //      spread over the 4 lanes of a quad: finish with quad_max.
  //   2. alpha = 2^(m_old * scale_log2 - mx * scale_log2); if mx is still
  //      -inf (row fully masked so far), use 0 in place of mx * scale_log2.
  //   3. s <- 2^(s * scale_log2 - mx * scale_log2)  (exp2_approx, fmaf),
  //      using the same substituted value for mx * scale_log2 as in item 2.
  //   4. l <- l * alpha + (sum of this lane's new s of the row): keep l
  //      per lane, do NOT quad-reduce here (lesson §3.3).
  //   5. o <- o * alpha for the row; m <- mx.
  // -------------------------------------------------------------------------
  LAB_TODO(2);
}

// o += P~ . V for one block, with P~ (the s array after online_softmax)
// taken straight from registers and V from shared memory.
template <class T, class C>
__device__ __forceinline__ void
pv_block(float (&o)[C::kMTiles][C::kD / 8][4],
         const float (&s)[C::kMTiles][C::kBN / 8][4], const T *sv, int lane) {
  // -------------------------------------------------------------------------
  // TODO(2b) — o[mi][dj] += P~[rows of tile mi][keys 16t .. 16t+15] .
  // V[keys 16t .. 16t+15][d 8dj .. 8dj+7], for t < kBN/16, dj < D/8.
  //   A operand: built from s[mi][2t] and s[mi][2t+1] with pack2<T>, no
  //     shared memory (which floats go into which register? lesson §4.3).
  //   B operand: V tile in sv is [keys][D] = MN-major for B. Use
  //     lab::ldmatrix_x4_trans: lanes 0-15 point at key rows 16t + (l % 16),
  //     lanes 16-31 at the same rows one 8-column chunk further; r[0], r[1]
  //     are the B fragment of d tile dj, r[2], r[3] of dj + 1 (lesson §4.4).
  // -------------------------------------------------------------------------
  LAB_TODO(2);
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

  // K blocks this Q block needs.
  int blocks = (n + C::kBN - 1) / C::kBN;
  if constexpr (kCausal) {
    // -----------------------------------------------------------------------
    // TODO(3a) — causal: the CTA's last query row is q0 + kBM - 1. Reduce
    // `blocks` so that K blocks containing only keys beyond it are never
    // loaded or computed (lesson §5.3).
    // -----------------------------------------------------------------------
    LAB_TODO(3);
  }

  auto issue_kv = [&](int jb, int stage) {
    T *sk = skv + stage * 2 * C::kKVElems;
    const int64_t off = static_cast<int64_t>(jb) * C::kBN * stride;
    issue_rows<C, C::kBN>(sk, k + off, 0, n - jb * C::kBN, stride);
    issue_rows<C, C::kBN>(sk + C::kKVElems, v + off, 0, n - jb * C::kBN,
                          stride);
  };

  issue_rows<C, C::kBM>(sq, q + static_cast<int64_t>(q0) * stride, 0, n - q0,
                        stride);
  if constexpr (kPipelined) {
    // -----------------------------------------------------------------------
    // TODO(4a) — prologue of the pipeline: K/V block 0 must be in flight
    // before the loop starts (which stage? which commit group?).
    // -----------------------------------------------------------------------
    LAB_TODO(4);
  }
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
      // ---------------------------------------------------------------------
      // TODO(4b) — double buffering (lab 21 step 1): block jb lives in stage
      // jb % 2. Set `stage`; wait until block jb has arrived and is visible to
      // every warp; then start copying block jb + 1 (if it exists) into the
      // other stage and commit. Why is that stage free at this point?
      // Exactly one wait and one __syncthreads per block (lesson §5.4).
      // ---------------------------------------------------------------------
      LAB_TODO(4);
    } else {
      // Synchronous version: copy block jb, wait for it, then compute.
      issue_kv(jb, 0);
      lab::cp_async_commit();
      lab::cp_async_wait_group<0>();
      __syncthreads();
    }
    const T *sk = skv + stage * 2 * C::kKVElems, *sv = sk + C::kKVElems;

    float s[C::kMTiles][C::kBN / 8][4];
    qk_block<T, C>(s, qf, sk, lane);
    // Only blocks that need a mask pay for it. A block that runs past the
    // end of N always does.
    const int k0 = jb * C::kBN;
    bool need_mask = k0 + C::kBN > n;
    if constexpr (kCausal) {
      // ---------------------------------------------------------------------
      // TODO(3c) — causal: which blocks contain a key that is beyond some
      // query row of this CTA (rows q0 .. q0 + kBM - 1)? Set need_mask for
      // them, and only for them (lesson §5.3).
      // ---------------------------------------------------------------------
      LAB_TODO(3);
    }
    if (need_mask)
      apply_mask<C, kCausal>(s, q0 + w0, k0, n, lane);
    online_softmax<C>(s, o, m, l, p.scale_log2);
    pv_block<T, C>(o, s, sv, lane);
    if constexpr (!kPipelined)
      __syncthreads(); // all warps done with K/V before the next overwrite
  }

  // -------------------------------------------------------------------------
  // TODO(2c) — epilogue (lesson §3.3). For each row (mi, h) of this lane:
  //   - the row sum is still split over the 4 lanes of the quad;
  //   - divide O by it once (no other division anywhere);
  //   - LSE = natural log of sum_j exp(scale * s_j): m is in raw S units;
  //   - skip rows >= n; write O as pack2<T> pairs (this lane holds columns
  //     8dj + 2 (lane % 4) and +1), LSE from one lane of the quad.
  // O row `row` starts at static_cast<T *>(p.o) + base + row * stride;
  // LSE of row `row` is p.lse[(batch * p.h + head) * n + row].
  // -------------------------------------------------------------------------
  LAB_TODO(2);
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
