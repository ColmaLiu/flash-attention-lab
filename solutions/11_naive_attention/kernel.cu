// Reference solution for 11_naive_attention: standard attention that
// materializes S and P in global memory (FlashAttention paper, Algorithm 0).
//
// q, k, v, o: [B, N, H, D] FP32, contiguous; lse: [B, H, N] FP32 (natural log).
// Element (b, i, h, d) of Q is at q[((b * N + i) * H + h) * D + d], so one
// head (b, h) is an N x D matrix with row stride H * D and column stride 1.
//
// step 1: S = scale * Q K^T              (provided GEMM, S is [B, H, N, N])
// step 2: P = softmax_j(S + causal mask), in place over S; also writes LSE
// step 3: O = P V                         (provided GEMM, V read "MN-major")
// Steps 1 and 2 put S / P in p.probe so the harness can check them; step 3
// uses the scratch buffer allocated in make_plan.
#include "lab/lab.h"

#include <cfloat>

namespace {

// ---------------------------------------------------------------------------
// Batched SIMT GEMM with arbitrary strides (what lab 10 taught).
//   C(m, n) = alpha * sum_k A(m, k) * B(n, k)      for every (b, h)
// An operand X with strides s has element (r, c) of batch (b, h) at
//   x[b * s.b + h * s.h + r * s.row + c * s.col].
// ---------------------------------------------------------------------------
struct Strides {
  int64_t b, h, row, col;
};

constexpr int kBM = 64;       // rows of C per block
constexpr int kBN = 64;       // columns of C per block
constexpr int kBK = 16;       // K slice held in shared memory
constexpr int kThreads = 256; // 16 x 16 threads, each owns 4 x 4 outputs
constexpr int kT = 4;

// Copy rows [r0, r0 + kRows) x columns [k0, k0 + kBK) of X into tile[k][r]
// (K-major in shared memory, so the inner product loop reads a row of the
// tile). Out-of-range elements become 0 so they add nothing to the sum.
// Consecutive threads walk along whichever index has stride 1 in memory, so
// the global loads are coalesced for both K-major (Q, K, P) and MN-major (V)
// operands.
template <int kRows>
__device__ void load_tile(const float *__restrict__ x, Strides s, int64_t base,
                          int r0, int rows, int k0, int depth,
                          float (*tile)[kRows + 1]) {
  for (int idx = threadIdx.x; idx < kBK * kRows; idx += kThreads) {
    int r, k;
    if (s.col == 1) { // K contiguous: lanes go along k
      k = idx % kBK;
      r = idx / kBK;
    } else { // rows contiguous: lanes go along r
      r = idx % kRows;
      k = idx / kRows;
    }
    int gr = r0 + r, gk = k0 + k;
    tile[k][r] = (gr < rows && gk < depth)
                     ? x[base + gr * s.row + int64_t(gk) * s.col]
                     : 0.0f;
  }
}

__global__ void __launch_bounds__(kThreads)
    gemm_kernel(const float *__restrict__ a, Strides sa,
                const float *__restrict__ b, Strides sb, float *__restrict__ c,
                Strides sc, int m, int n, int depth, int heads, float alpha) {
  __shared__ float as[kBK][kBM + 1];
  __shared__ float bs[kBK][kBN + 1];
  int bi = blockIdx.z / heads, hi = blockIdx.z % heads;
  int64_t a_base = bi * sa.b + hi * sa.h;
  int64_t b_base = bi * sb.b + hi * sb.h;
  int64_t c_base = bi * sc.b + hi * sc.h;
  int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  // Thread (ty, tx) owns rows ty + 16 * i and columns tx + 16 * j, so a warp
  // reads 16 consecutive shared-memory words (no bank conflict) and writes
  // 16 consecutive C columns.
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  float acc[kT][kT] = {};
  for (int k0 = 0; k0 < depth; k0 += kBK) {
    load_tile<kBM>(a, sa, a_base, m0, m, k0, depth, as);
    load_tile<kBN>(b, sb, b_base, n0, n, k0, depth, bs);
    __syncthreads(); // tiles complete before anyone reads them
#pragma unroll
    for (int kk = 0; kk < kBK; ++kk) {
      float ra[kT], rb[kT];
#pragma unroll
      for (int i = 0; i < kT; ++i)
        ra[i] = as[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < kT; ++j)
        rb[j] = bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < kT; ++i)
#pragma unroll
        for (int j = 0; j < kT; ++j)
          acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads(); // everyone is done reading before the next load
  }
#pragma unroll
  for (int i = 0; i < kT; ++i) {
#pragma unroll
    for (int j = 0; j < kT; ++j) {
      int r = m0 + ty + 16 * i, col = n0 + tx + 16 * j;
      if (r < m && col < n)
        c[c_base + r * sc.row + int64_t(col) * sc.col] = alpha * acc[i][j];
    }
  }
}

// C = alpha * A B^T for all B * H heads; see the comment at Strides.
void gemm(const float *a, Strides sa, const float *b, Strides sb, float *c,
          Strides sc, int m, int n, int depth, int batch, int heads,
          float alpha, cudaStream_t stream) {
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM, batch * heads);
  gemm_kernel<<<grid, kThreads, 0, stream>>>(a, sa, b, sb, c, sc, m, n, depth,
                                             heads, alpha);
}

// ---------------------------------------------------------------------------
// Block-wide reductions (what lab 01 taught).
// ---------------------------------------------------------------------------
constexpr int kSoftmaxThreads = 128;

// Online-softmax state of a set of scores: m = max, l = sum exp(s - m).
struct State {
  float m, l;
};

// Merge two states (lab 01): rescale both sums to the larger maximum.
// An empty state is {-FLT_MAX, 0}; merging it changes nothing.
__device__ State merge(State x, State y) {
  float m = fmaxf(x.m, y.m);
  return {m, x.l * __expf(x.m - m) + y.l * __expf(y.m - m)};
}

// Every thread of the block gets the merged state of all threads.
__device__ State block_merge(State x) {
  __shared__ State partial[kSoftmaxThreads / 32];
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    State y = {__shfl_xor_sync(0xffffffff, x.m, offset),
               __shfl_xor_sync(0xffffffff, x.l, offset)};
    x = merge(x, y);
  }
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0)
    partial[warp] = x;
  __syncthreads();
  x = partial[0];
#pragma unroll
  for (int w = 1; w < kSoftmaxThreads / 32; ++w)
    x = merge(x, partial[w]);
  return x;
}

// One block per row of S. Row r = (b * H + h) * N + i is both the row of the
// [B, H, N, N] score matrix and the index of lse[b, h, i].
//   pass 1: online max/sum over the unmasked keys j < len
//   pass 2: P[j] = exp(S[j] - m) / l for j < len, 0 for masked keys
// The row is read twice; the second read mostly hits L1/L2 (a row is at most
// 16 KB at N = 4096), so HBM sees S read once and P written once.
__global__ void __launch_bounds__(kSoftmaxThreads)
    softmax_rows(float *__restrict__ s, float *__restrict__ lse, int n,
                 bool causal) {
  int64_t row = blockIdx.x;
  int i = static_cast<int>(row % n); // query position
  float *sr = s + row * n;
  // Causal: query i sees keys 0..i. The diagonal is always visible, so every
  // row has at least one key and l >= 1 below.
  int len = causal ? i + 1 : n;

  State st = {-FLT_MAX, 0.0f};
  for (int j = threadIdx.x; j < len; j += kSoftmaxThreads) {
    float x = sr[j];
    if (x > st.m) { // new maximum: rescale the old sum, one exp either way
      st.l = st.l * __expf(st.m - x) + 1.0f;
      st.m = x;
    } else {
      st.l += __expf(x - st.m);
    }
  }
  st = block_merge(st);

  float inv_l = 1.0f / st.l;
  for (int j = threadIdx.x; j < n; j += kSoftmaxThreads)
    sr[j] = j < len ? __expf(sr[j] - st.m) * inv_l : 0.0f;
  if (threadIdx.x == 0)
    lse[row] = st.m + logf(st.l); // natural log, like torch.logsumexp
}

} // namespace

struct Plan {
  lab::Scratch s; // S, then P in place: [B, H, N, N] FP32
};

Plan make_plan(const lab::AttentionProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F32 && p.o_dtype == lab::DType::F32,
              "this lab uses FP32 Q, K, V and O");
  LAB_REQUIRE(p.b * p.h <= 65535, "B * H must fit in gridDim.z");
  size_t bytes = sizeof(float) * size_t(p.b) * p.h * p.n * p.n;
  LAB_REQUIRE(bytes <= (size_t(32) << 30),
              "S would need more than 32 GiB: that is the point of this lab");
  if (p.step <= 2) {
    LAB_REQUIRE(p.probe != nullptr && size_t(p.probe_numel) * 4 == bytes,
                "steps 1 and 2 write S / P into the [B, H, N, N] probe");
    return {};
  }
  return {lab::Scratch(bytes)};
}

void run(const Plan &plan, const lab::AttentionProblem &p,
         cudaStream_t stream) {
  auto q = static_cast<const float *>(p.q);
  auto k = static_cast<const float *>(p.k);
  auto v = static_cast<const float *>(p.v);
  auto o = static_cast<float *>(p.o);
  int64_t n = p.n, h = p.h, d = p.d;
  float *s = p.step <= 2 ? p.probe : plan.s.as<float>();

  // Q, K, V, O as per-head N x D matrices inside [B, N, H, D].
  Strides head_rows{n * h * d, d, h * d, 1};
  // S and P: [B, H, N, N], each head a dense N x N matrix.
  Strides scores{h * n * n, n * n, n, 1};

  // step 1: S(i, j) = scale * sum_d Q(i, d) K(j, d).
  gemm(q, head_rows, k, head_rows, s, scores, p.n, p.n, p.d, p.b, p.h, p.scale,
       stream);
  if (p.step == 1)
    return;

  // step 2: P = softmax(S) row by row, in place; LSE alongside.
  softmax_rows<<<p.b * p.h * p.n, kSoftmaxThreads, 0, stream>>>(s, p.lse, p.n,
                                                                p.causal);
  if (p.step == 2)
    return;

  // step 3: O(i, d) = sum_j P(i, j) V(j, d). The GEMM wants B(n, k) with
  // n = d and k = j, i.e. V read with row stride 1 and column stride H * D.
  Strides v_by_d{n * h * d, d, 1, h * d};
  gemm(s, scores, v, v_by_d, o, head_rows, p.n, p.d, p.n, p.b, p.h, 1.0f,
       stream);
}

LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
