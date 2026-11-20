// 01_softmax: y[r, :] = softmax(x[r, :]) for every row r
// Lesson: course/01_softmax/README.md
//
// x and y are [rows, cols], FP32, row-major: x[r][c] is at x[r * cols + c].
// Every row has at least one finite entry; other entries may be -inf.
// run() picks a version by p.step, so all versions stay side by side and you
// can compare them:  ./lab bench 01 --step 1   ...   ./lab bench 01 --step 4
#include <cstdint>
#include "lab/lab.h"

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kBlock = 256; // steps 1-3: threads per block, one row per block
constexpr unsigned kFullMask = 0xffffffffu;

// 2^x with the MUFU.EX2 unit (PTX ex2.approx.ftz.f32: max error 2 ulp,
// results below 2^-126 flush to 0, ex2(-inf) = 0). Lesson section 4.4.
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// e^(x - m). Use this for every exponential in this lab.
// x - m is computed before the multiply on purpose (lesson section 4.4).
__device__ __forceinline__ float exp_shifted(float x, float m) {
  return ex2((x - m) * kLog2e);
}

// Reduction operators: op(a, b) and an identity e with op(e, a) = a.
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
// TODO(1a) — step 1: block reduction through a shared-memory tree.
// Input: every thread's v; smem has room for blockDim.x floats; blockDim.x is
// a power of two (kBlock = 256).
// Output: EVERY thread returns op(v_0, v_1, ..., v_{blockDim.x - 1}).
//   1. Each thread stores v into smem[threadIdx.x].
//   2. Halve the number of active threads each level: thread t < s combines
//      smem[t] and smem[t + s]. Which __syncthreads() does each level need?
//   3. Everyone reads the result from smem[0].
// smem is reused by the next call: make sure nobody can overwrite it while
// another thread is still reading smem[0].
// ---------------------------------------------------------------------------
template <class Op>
__device__ float block_reduce_tree(float v, Op op, float *smem) {
  LAB_TODO(1);
  return v;
}

// ---------------------------------------------------------------------------
// TODO(2) — step 2: warp shuffle reductions.
// warp_reduce: every lane of a full warp returns op over the 32 lanes' v.
//   Use __shfl_xor_sync(kFullMask, v, offset) with offset = 16, 8, 4, 2, 1
//   (a butterfly: afterwards every lane holds the same result).
// block_reduce_shfl: same contract as block_reduce_tree, but smem only has
//   room for 32 floats (one per warp). blockDim.x is a multiple of 32.
//   Reduce inside each warp, let lane 0 publish the warp's partial, then
//   reduce the partials. Pad lanes beyond the number of warps with
//   Op::kIdentity. Same smem reuse rule as in TODO(1a).
// ---------------------------------------------------------------------------
template <class Op> __device__ float warp_reduce(float v, Op op) {
  LAB_TODO(2);
  return v;
}

template <class Op>
__device__ float block_reduce_shfl(float v, Op op, float *smem) {
  LAB_TODO(2);
  return v;
}

template <bool kShuffle, class Op>
__device__ float block_reduce(float v, Op op, float *smem) {
  if constexpr (kShuffle)
    return block_reduce_shfl(v, op, smem);
  else
    return block_reduce_tree(v, op, smem);
}

// ---------------------------------------------------------------------------
// TODO(1b) — step 1 (step 2 reuses this kernel with kShuffle = true).
// One block of kBlock threads per row: blockIdx.x is the row.
// Thread t handles columns t, t + kBlock, t + 2 * kBlock, ... (< cols);
// a warp then touches consecutive columns, so loads are coalesced.
//   pass 1: m = max_j x_j            (per-thread max, then block_reduce)
//   pass 2: l = sum_j e^(x_j - m)    (exp_shifted, then block_reduce)
//   pass 3: y_j = e^(x_j - m) / l    (compute 1 / l once)
// Every reduction call passes the same smem array.
// ---------------------------------------------------------------------------
template <bool kShuffle>
__global__ void __launch_bounds__(kBlock)
    softmax_three_pass(const float *__restrict__ x, float *__restrict__ y,
                       int cols) {
  __shared__ float smem[kBlock];
  const float *xr = x + static_cast<int64_t>(blockIdx.x) * cols;
  float *yr = y + static_cast<int64_t>(blockIdx.x) * cols;
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(3) — step 3: online softmax.
// A State summarizes a set S of elements of the row:
//   m = max_{j in S} x_j,   l = sum_{j in S} e^(x_j - m).
// The empty set is {-inf, 0}; the one-element set {x} is {x, 1}.
//
// merge(a, b): the State of the union of the two (disjoint) sets.
//   Rescale each l to the new max (lesson section 3.3). If the new max is
//   -inf, return the empty state: (-inf) - (-inf) is NaN.
// warp_merge: like warp_reduce, but shuffle both fields and merge().
// block_merge: like block_reduce_shfl, with smem holding 32 States.
// softmax_online: one block per row, as in step 1.
//   pass 1: s = merge(s, {x_j, 1}) over this thread's columns, then
//           block_merge. Do not branch on x_j > s.m to save an exp; see the
//           lesson (section 6, step 3) for why that is slower.
//   pass 2: y_j = e^(x_j - m) / l.
// ---------------------------------------------------------------------------
struct State {
  float m;
  float l;
};

__device__ __forceinline__ State merge(State a, State b) {
  LAB_TODO(3);
  return a;
}

__device__ __forceinline__ State warp_merge(State s) {
  LAB_TODO(3);
  return s;
}

__device__ State block_merge(State s, State *smem) {
  LAB_TODO(3);
  return s;
}

__global__ void __launch_bounds__(kBlock)
    softmax_online(const float *__restrict__ x, float *__restrict__ y,
                   int cols) {
  __shared__ State smem[32];
  const float *xr = x + static_cast<int64_t>(blockIdx.x) * cols;
  float *yr = y + static_cast<int64_t>(blockIdx.x) * cols;
  LAB_TODO(3);
}

// ---------------------------------------------------------------------------
// Step 4 helpers (provided).
// A chunk is V consecutive floats: V = 4 is one 16-byte float4 access (needs
// cols % 4 == 0 and aligned pointers, run() checks), V = 1 is a plain float.
// __ldcs / __stcs are "streaming" loads/stores: x and y are touched once,
// so they should not push other data out of the caches.
// ---------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ void load_chunk(const float *p, float (&v)[V]) {
  if constexpr (V == 4) {
    float4 q = __ldcs(reinterpret_cast<const float4 *>(p));
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

// Merge the States of the kThreads threads that share a row; every one of
// them gets the row's State. A warp per row needs no shared memory.
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

// ---------------------------------------------------------------------------
// TODO(4) — step 4: the row lives in registers; x is read from HBM once.
// kThreads threads own one row (32: a warp per row; >= 128: a block per
// row). Thread t of the row holds kVecs chunks of V floats in v[kVecs][V]:
// chunk k starts at column (k * kThreads + t) * V, so the 32 lanes of a warp
// cover 32 * V consecutive floats with one load instruction.
// kThreads * kVecs * V >= cols (run() picks the configuration).
//   1. Load ALL chunks first (independent loads in flight). A chunk with
//      column >= cols is padding: fill it with -inf. Since cols % V == 0,
//      a chunk is either entirely inside the row or entirely outside.
//   2. Thread-local m = max of its v; turn v into e^(x - m) and sum them
//      into l. If m == -inf (all padding or masked), shift by 0 instead so
//      the exps are 0, not NaN.
//   3. ONE row_merge of {m, l} gives the row's (M, L). There is no second
//      reduction: e^(x - M) = e^(x - m) * e^(m - M).
//   4. Scale v by e^(m - M) / L and store the chunks inside the row.
// Use #pragma unroll on loops over k and i so v stays in registers
// (check with ./lab sass 01: LOCAL should be 0).
// ---------------------------------------------------------------------------
template <int kThreads, int kVecs, int V>
__global__ void __launch_bounds__(block_threads<kThreads>())
    softmax_regs(const float *__restrict__ x, float *__restrict__ y, int rows,
                 int cols) {
  __shared__ State smem[32];
  int row = blockIdx.x * rows_per_block<kThreads>() + threadIdx.x / kThreads;
  int t = threadIdx.x % kThreads; // this thread's index within the row
  // Only the warp-per-row version can have a row past the end, and then the
  // whole warp leaves together (that version has no __syncthreads).
  if (row >= rows)
    return;
  const float *xr = x + static_cast<int64_t>(row) * cols;
  float *yr = y + static_cast<int64_t>(row) * cols;
  float v[kVecs][V];
  LAB_TODO(4);
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
// Each thread keeps at most 32 floats, so nothing spills to local memory.
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

// Nothing to prepare in this lab.
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
    if (!done) // the row does not fit in registers: use step 3
      softmax_online<<<p.rows, kBlock, 0, stream>>>(p.x, p.y, p.cols);
    break;
  }
  }
}

LAB_REGISTER(lab::SoftmaxProblem, Plan, make_plan, run);
