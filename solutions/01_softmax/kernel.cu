// Reference solution for 01_softmax: y[r, :] = softmax(x[r, :]).
//
// x and y are [rows, cols] FP32, row-major: x[r][c] is at x[r * cols + c].
//
// step 1: one block per row, three passes over the row (max, sum of exps,
//         write). Block reductions go through a shared-memory tree.
// step 2: the same kernel; reductions use __shfl_xor_sync inside each warp
//         and shared memory only for the 8 per-warp partials.
// step 3: online softmax. One pass keeps a running state (m, l); the states
//         of all threads are merged with the associative merge(). A second
//         pass writes y. x is read twice instead of three times.
// step 4: the whole row stays in registers, so x is read from HBM exactly
//         once. float4 loads/stores when cols % 4 == 0; a warp per row for
//         short rows, a block per row for long rows. Rows that do not fit in
//         registers fall back to step 3.
#include <cstdint>
#include "lab/lab.h"

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kBlock = 256; // steps 1-3: threads per block, one row per block
constexpr unsigned kFullMask = 0xffffffffu;

// 2^x with the MUFU.EX2 unit (PTX ex2.approx.ftz.f32: max error 2 ulp,
// results below 2^-126 flush to 0, ex2(-inf) = 0).
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// e^(x - m) = 2^((x - m) * log2 e). Subtract first: x - m is exact when x and
// m are close (Sterbenz lemma), so a large common offset (the `shifted` case,
// x ~ 5000) does not cost precision. fmaf(x, log2e, -m * log2e) rounds
// m * log2e ~ 7200 to FP32 and returns e^(x-m) * 2^(-d(m)), |d(m)| <= 2^-12.
// With one shift per row (steps 1-2) the factor cancels in e / l; when states
// with different shifts are merged (steps 3-4) it does not, and `shifted`
// fails with ~3e-4 relative error.
__device__ __forceinline__ float exp_shifted(float x, float m) {
  return ex2((x - m) * kLog2e);
}

struct MaxOp {
  static constexpr float kIdentity = kNegInf;
  __device__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

struct SumOp {
  static constexpr float kIdentity = 0.f;
  __device__ float operator()(float a, float b) const {
    return a + b;
  }
};

// ---------------------------------------------------------------------------
// Step 1: block reduction through a shared-memory tree.
// smem holds blockDim.x floats; blockDim.x must be a power of two.
// Every thread returns the reduction of all threads' v.
// ---------------------------------------------------------------------------
template <class Op>
__device__ float block_reduce_tree(float v, Op op, float *smem) {
  int t = threadIdx.x;
  smem[t] = v;
  __syncthreads(); // all inputs are in smem
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s)
      smem[t] = op(smem[t], smem[t + s]);
    __syncthreads(); // level done before the next level reads it
  }
  float r = smem[0];
  __syncthreads(); // everyone has read smem[0] before smem is reused
  return r;
}

// ---------------------------------------------------------------------------
// Step 2: warp shuffle. After the butterfly every lane holds the result.
// ---------------------------------------------------------------------------
template <class Op> __device__ float warp_reduce(float v, Op op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = op(v, __shfl_xor_sync(kFullMask, v, offset));
  return v;
}

// smem holds 32 floats (one per warp). blockDim.x is a multiple of 32.
template <class Op>
__device__ float block_reduce_shfl(float v, Op op, float *smem) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int warps = blockDim.x / 32;
  v = warp_reduce(v, op);
  if (lane == 0)
    smem[warp] = v;
  __syncthreads(); // every warp's partial is in smem
  // Every warp reduces the (at most 32) partials, so every thread gets the
  // result without a second broadcast through smem.
  v = warp_reduce(lane < warps ? smem[lane] : Op::kIdentity, op);
  __syncthreads(); // everyone has read smem before it is reused
  return v;
}

template <bool kShuffle, class Op>
__device__ float block_reduce(float v, Op op, float *smem) {
  if constexpr (kShuffle)
    return block_reduce_shfl(v, op, smem);
  else
    return block_reduce_tree(v, op, smem);
}

// Steps 1 and 2: one block per row, three passes over x.
template <bool kShuffle>
__global__ void __launch_bounds__(kBlock)
    softmax_three_pass(const float *__restrict__ x, float *__restrict__ y,
                       int cols) {
  __shared__ float smem[kBlock];
  const float *xr = x + static_cast<int64_t>(blockIdx.x) * cols;
  float *yr = y + static_cast<int64_t>(blockIdx.x) * cols;

  // Pass 1: m = max_j x_j.
  float m = kNegInf;
  for (int c = threadIdx.x; c < cols; c += kBlock)
    m = fmaxf(m, xr[c]);
  m = block_reduce<kShuffle>(m, MaxOp{}, smem);

  // Pass 2: l = sum_j e^(x_j - m). m is finite: every row has a finite entry.
  float l = 0.f;
  for (int c = threadIdx.x; c < cols; c += kBlock)
    l += exp_shifted(xr[c], m);
  l = block_reduce<kShuffle>(l, SumOp{}, smem);

  // Pass 3: y_j = e^(x_j - m) / l.
  float inv = 1.f / l;
  for (int c = threadIdx.x; c < cols; c += kBlock)
    yr[c] = exp_shifted(xr[c], m) * inv;
}

// ---------------------------------------------------------------------------
// Step 3: online softmax.
// A State summarizes a set S of elements: m = max_{j in S} x_j and
// l = sum_{j in S} e^(x_j - m). The empty set is (-inf, 0).
// ---------------------------------------------------------------------------
struct State {
  float m;
  float l;
};

// State of the union of two disjoint sets. Associative and commutative, so
// partial states can be merged in any order and any tree shape.
__device__ __forceinline__ State merge(State a, State b) {
  float m = fmaxf(a.m, b.m);
  if (m == kNegInf) // both sets are empty or all -inf: avoid (-inf) - (-inf)
    return {kNegInf, 0.f};
  return {m, a.l * exp_shifted(a.m, m) + b.l * exp_shifted(b.m, m)};
}

__device__ __forceinline__ State warp_merge(State s) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    State o{__shfl_xor_sync(kFullMask, s.m, offset),
            __shfl_xor_sync(kFullMask, s.l, offset)};
    s = merge(s, o);
  }
  return s;
}

// smem holds 32 States. blockDim.x is a multiple of 32.
__device__ State block_merge(State s, State *smem) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int warps = blockDim.x / 32;
  s = warp_merge(s);
  if (lane == 0)
    smem[warp] = s;
  __syncthreads(); // every warp's state is in smem
  s = warp_merge(lane < warps ? smem[lane] : State{kNegInf, 0.f});
  __syncthreads(); // everyone has read smem before it is reused
  return s;
}

__global__ void __launch_bounds__(kBlock)
    softmax_online(const float *__restrict__ x, float *__restrict__ y,
                   int cols) {
  __shared__ State smem[32];
  const float *xr = x + static_cast<int64_t>(blockIdx.x) * cols;
  float *yr = y + static_cast<int64_t>(blockIdx.x) * cols;

  // Pass 1: this thread's state over its elements, then the row's state.
  // The one-element set {x} has state (x, e^(x - x)) = (x, 1); for x = -inf
  // merge() still gives the right answer (it contributes 1 * e^-inf = 0).
  // This costs two exps per element. Branching on x > m to skip the e^0 = 1
  // looks cheaper, but the branch stops the compiler from issuing the next
  // loads early, and this kernel is limited by loads in flight, not by exp.
  State s{kNegInf, 0.f};
  for (int c = threadIdx.x; c < cols; c += kBlock)
    s = merge(s, State{xr[c], 1.f});
  s = block_merge(s, smem);

  // Pass 2: write.
  float inv = 1.f / s.l;
  for (int c = threadIdx.x; c < cols; c += kBlock)
    yr[c] = exp_shifted(xr[c], s.m) * inv;
}

// ---------------------------------------------------------------------------
// Step 4: the row lives in registers.
// kThreads threads own one row; each holds kVecs chunks of V consecutive
// floats. Chunk k of thread t starts at column (k * kThreads + t) * V, so a
// warp's 32 chunks are contiguous: one load instruction covers 32 * V floats.
// kThreads == 32: a warp per row, 4 rows per 128-thread block, no smem.
// kThreads >= 64: a block per row; the warps meet through smem.
// ---------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ void load_chunk(const float *p, float (&v)[V]) {
  if constexpr (V == 4) {
    float4 q = __ldcs(reinterpret_cast<const float4 *>(p)); // read once
    v[0] = q.x, v[1] = q.y, v[2] = q.z, v[3] = q.w;
  } else {
    v[0] = __ldcs(p);
  }
}

template <int V>
__device__ __forceinline__ void store_chunk(float *p, const float (&v)[V]) {
  if constexpr (V == 4)
    __stcs(reinterpret_cast<float4 *>(p), make_float4(v[0], v[1], v[2], v[3]));
  else
    __stcs(p, v[0]);
}

template <int kThreads>
__device__ __forceinline__ State row_merge(State s, State *smem) {
  if constexpr (kThreads == 32)
    return warp_merge(s);
  else
    return block_merge(s, smem);
}

template <int kThreads> constexpr int rows_per_block() {
  return kThreads < 128 ? 128 / kThreads : 1;
}

template <int kThreads> constexpr int block_threads() {
  return kThreads * rows_per_block<kThreads>();
}

template <int kThreads, int kVecs, int V>
__global__ void __launch_bounds__(block_threads<kThreads>())
    softmax_regs(const float *__restrict__ x, float *__restrict__ y, int rows,
                 int cols) {
  __shared__ State smem[32];
  int row = blockIdx.x * rows_per_block<kThreads>() + threadIdx.x / kThreads;
  int t = threadIdx.x % kThreads;
  // Only the warp-per-row version can have a row past the end, and then the
  // whole warp leaves together (no __syncthreads in that version).
  if (row >= rows)
    return;
  const float *xr = x + static_cast<int64_t>(row) * cols;
  float *yr = y + static_cast<int64_t>(row) * cols;

  // Load every chunk before using any: kVecs independent loads in flight.
  float v[kVecs][V];
#pragma unroll
  for (int k = 0; k < kVecs; ++k) {
    int c = (k * kThreads + t) * V;
    if (c < cols) { // cols % V == 0, so a chunk is all in or all out
      load_chunk<V>(xr + c, v[k]);
    } else {
#pragma unroll
      for (int i = 0; i < V; ++i)
        v[k][i] = kNegInf;
    }
  }

  // This thread's state. v becomes e^(x - m_local); a thread whose elements
  // are all -inf uses 0 as its shift so that exp gives 0, not NaN.
  float m = kNegInf;
#pragma unroll
  for (int k = 0; k < kVecs; ++k)
#pragma unroll
    for (int i = 0; i < V; ++i)
      m = fmaxf(m, v[k][i]);
  float shift = m == kNegInf ? 0.f : m;
  float l = 0.f;
#pragma unroll
  for (int k = 0; k < kVecs; ++k)
#pragma unroll
    for (int i = 0; i < V; ++i) {
      v[k][i] = exp_shifted(v[k][i], shift);
      l += v[k][i];
    }

  // One merge of (m, l) instead of two reductions (max, then sum). Then
  // e^(x - m_row) = e^(x - m_local) * e^(m_local - m_row): one exp per
  // element plus one per thread, the same rescaling FlashAttention applies
  // to its output accumulator.
  State row_state = row_merge<kThreads>(State{m, l}, smem);
  float scale = exp_shifted(m, row_state.m) / row_state.l; // 0 if m = -inf

#pragma unroll
  for (int k = 0; k < kVecs; ++k) {
    int c = (k * kThreads + t) * V;
    if (c < cols) {
#pragma unroll
      for (int i = 0; i < V; ++i)
        v[k][i] *= scale;
      store_chunk<V>(yr + c, v[k]);
    }
  }
}

template <int kThreads, int kVecs, int V>
void launch_regs(const float *x, float *y, int rows, int cols,
                 cudaStream_t stream) {
  constexpr int rpb = rows_per_block<kThreads>();
  int blocks = (rows + rpb - 1) / rpb;
  softmax_regs<kThreads, kVecs, V>
      <<<blocks, block_threads<kThreads>(), 0, stream>>>(x, y, rows, cols);
}

// Pick the smallest configuration whose kThreads * kVecs * V covers a row.
// Each thread keeps at most 32 floats, so kVecs * V <= 32 and nothing spills.
// Returns false if the row does not fit (more than 32768 floats).
template <int V>
bool launch_fast(const float *x, float *y, int rows, int cols,
                 cudaStream_t stream) {
  int chunks = (cols + V - 1) / V;
  if (chunks <= 32)
    launch_regs<32, 1, V>(x, y, rows, cols, stream);
  else if (chunks <= 64)
    launch_regs<32, 2, V>(x, y, rows, cols, stream);
  else if (chunks <= 128)
    launch_regs<32, 4, V>(x, y, rows, cols, stream);
  else if (chunks <= 256)
    launch_regs<32, 8, V>(x, y, rows, cols, stream);
  else if (chunks <= 512)
    launch_regs<128, 4, V>(x, y, rows, cols, stream);
  else if (chunks <= 1024)
    launch_regs<256, 4, V>(x, y, rows, cols, stream);
  else if (chunks <= 2048)
    launch_regs<512, 4, V>(x, y, rows, cols, stream);
  else if (chunks <= 4096)
    launch_regs<512, 8, V>(x, y, rows, cols, stream);
  else if (chunks * V <= 32768)
    launch_regs<1024, 32 / V, V>(x, y, rows, cols, stream);
  else
    return false;
  return true;
}

bool aligned16(const void *p) {
  return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

} // namespace

struct Plan {};

Plan make_plan(const lab::SoftmaxProblem &p) {
  LAB_REQUIRE(p.rows >= 1 && p.cols >= 1, "empty input");
  return {};
}

void run(const Plan &, const lab::SoftmaxProblem &p, cudaStream_t stream) {
  switch (p.step) {
  case 1:
    softmax_three_pass<false><<<p.rows, kBlock, 0, stream>>>(p.x, p.y, p.cols);
    break;
  case 2:
    softmax_three_pass<true><<<p.rows, kBlock, 0, stream>>>(p.x, p.y, p.cols);
    break;
  case 3:
    softmax_online<<<p.rows, kBlock, 0, stream>>>(p.x, p.y, p.cols);
    break;
  default: {
    // float4 needs every row start 16-byte aligned: cols % 4 == 0 and
    // aligned base pointers.
    bool vec4 = p.cols % 4 == 0 && aligned16(p.x) && aligned16(p.y);
    bool done = vec4 ? launch_fast<4>(p.x, p.y, p.rows, p.cols, stream)
                     : launch_fast<1>(p.x, p.y, p.rows, p.cols, stream);
    if (!done) // the row does not fit in registers
      softmax_online<<<p.rows, kBlock, 0, stream>>>(p.x, p.y, p.cols);
    break;
  }
  }
}

LAB_REGISTER(lab::SoftmaxProblem, Plan, make_plan, run);
