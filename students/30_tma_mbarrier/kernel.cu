// 30_tma_mbarrier: y[c][r] = x[r][c], moved by TMA and synchronized with
// mbarriers.
// Lesson: course/30_tma_mbarrier/README.md
//
// x is [rows, cols], y is [cols, rows], both FP32 and row-major:
//   x[r][c] is at x[r * cols + c],   y[c][r] is at y[c * rows + r].
// Supported: rows % 4 == 0 and cols % 4 == 0 (TMA needs a row pitch that is
// a multiple of 16 bytes). make_plan rejects anything else.
// Tail tiles need no special code: TMA loads fill out-of-bounds elements with
// zeros and TMA stores skip them.
// run() picks a version by p.step:  ./lab bench 30 --step 1 ... --step 3
//
// A TMA coordinate is (x, y) = (column, row) of the box's first element:
// the innermost (contiguous) dimension comes first.
#include "lab/lab.h"

#include <algorithm>
#include <cuda.h>

namespace {

// ===========================================================================
// mbarrier and TMA primitives. You write them here; from lab 31 on they come
// from include/lab/mbarrier.cuh and include/lab/tma.cuh.
// Operands: a shared-memory address is a 32-bit "r" operand from
// shared_addr(ptr); a CUtensorMap pointer or global pointer is an "l"
// operand; byte counts and coordinates are "r". Give every asm a "memory"
// clobber. A wait loop needs a predicate register and a label: inside
// asm, "{ .reg .pred p; LAB_WAIT_%=: ...; @!p bra LAB_WAIT_%=; }" ("%="
// makes the label unique for every inlined copy).
// ===========================================================================
__device__ __forceinline__ uint32_t shared_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// TODO(1a) — step 1 primitives (lesson §4.2–§4.5). One PTX instruction each.
//   mbar_init:            mbarrier.init.shared::cta.b64 [bar], count;
//                         phase 0, `count` arrivals expected per phase.
//   fence_barrier_init:   fence.mbarrier_init.release.cluster;
//                         makes the init visible to the async proxy (TMA).
//   mbar_arrive_expect_tx: mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes;
//                         add `bytes` to the phase's tx-count, then arrive once.
//   mbar_wait:            loop on mbarrier.try_wait.parity.shared::cta.b64 p, [bar], parity;
//                         until the phase with this parity has completed.
//   fence_proxy_async:    fence.proxy.async.shared::cta;
//                         orders this thread's generic shared-memory writes
//                         before later async-proxy (TMA) reads.
//   tma_load_2d:   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
//                  [dst], [map, {x, y}], [bar];
//                  box with origin (x, y) -> shared dst; complete_tx on bar.
//   tma_store_2d:  cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [map, {x, y}], [src];
//   bulk_commit_group:   cp.async.bulk.commit_group;
//   bulk_wait_group<N>:  cp.async.bulk.wait_group N;   ("n" constraint)
//                  at most N of this thread's store groups still pending.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  LAB_TODO(1);
}
__device__ __forceinline__ void fence_barrier_init() {
  LAB_TODO(1);
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar,
                                                      uint32_t bytes) {
  LAB_TODO(1);
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  LAB_TODO(1);
}
__device__ __forceinline__ void fence_proxy_async() {
  LAB_TODO(1);
}
__device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *map,
                                            uint64_t *bar, int x, int y) {
  LAB_TODO(1);
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap *map,
                                             const void *src, int x, int y) {
  LAB_TODO(1);
}
__device__ __forceinline__ void bulk_commit_group() {
  LAB_TODO(1);
}
template <int N> __device__ __forceinline__ void bulk_wait_group() {
  LAB_TODO(1);
}

// ---------------------------------------------------------------------------
// TODO(2a) — step 2 primitives (lesson §4.2, §4.4, §5.2).
//   mbar_arrive:             mbarrier.arrive.shared::cta.b64 _, [bar];
//   bulk_wait_group_read<N>: cp.async.bulk.wait_group.read N;
//       at most N groups may still be READING their shared-memory source.
//   elect_one: elect.sync r|p, 0xffffffff;  all 32 lanes call it, exactly
//       one gets true (return the predicate as a bool, e.g. with selp).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  LAB_TODO(2);
}
template <int N> __device__ __forceinline__ void bulk_wait_group_read() {
  LAB_TODO(2);
}
__device__ __forceinline__ bool elect_one() {
  LAB_TODO(2);
  return false;
}

// ---------------------------------------------------------------------------
// TODO(1b) — host: encode the tensor map of a row-major [rows, cols] FP32
// matrix whose box is box_rows x box_cols (lesson §4.1). Call
//   cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank,
//       base, dims, strides, box, element_strides,
//       CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
//       CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// with the innermost dimension FIRST in dims and box, rank - 1 strides in
// bytes (dimension 0's stride is implied), element strides all 1.
// If it does not return CUDA_SUCCESS, throw (LAB_REQUIRE(false, ...)) with
// the message from cuGetErrorString.
// ---------------------------------------------------------------------------
CUtensorMap encode_2d(const void *base, int rows, int cols, int box_rows,
                      int box_cols, CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  LAB_TODO(1);
  return map;
}

constexpr int kTile = 32; // edge of a TMA box in floats: 32 * 4 B = 128 B

// ===========================================================================
// Step 1: one tile per CTA.
// ===========================================================================
constexpr int kOneTileThreads = 128;

// ---------------------------------------------------------------------------
// TODO(1c) — the CTA owns the 32x32 tile of x with origin
// (r0, c0) = (blockIdx.y * 32, blockIdx.x * 32). x_map and y_map have
// 32x32 boxes without swizzle (see make_plan).
//   1. Thread 0: init `full` for ONE arrival, fence_barrier_init();
//      then __syncthreads() so every thread sees an initialized barrier.
//   2. Thread 0: arm `full` with the number of bytes the load delivers, and
//      issue the load of the x box into `in`. (Which bytes count when the
//      box sticks out of the matrix?)
//   3. Everyone: wait for phase 0 of `full`.
//   4. out[i][j] = in[j][i] (4 warps; e.g. lane = j, warp strides over i).
//   5. Hand `out` to the TMA store: which fence, executed by whom, and which
//      barrier? (lesson §4.5)
//   6. Thread 0: store `out` as the y box with origin (x = r0, y = c0),
//      commit, and wait until the store is done before the CTA exits.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kOneTileThreads)
    transpose_one_tile(const __grid_constant__ CUtensorMap x_map,
                       const __grid_constant__ CUtensorMap y_map) {
  __shared__ __align__(128) float in[kTile][kTile];
  __shared__ __align__(128) float out[kTile][kTile];
  __shared__ __align__(8) uint64_t full;
  LAB_TODO(1);
}

// ===========================================================================
// Steps 2 and 3: persistent CTAs, a ring of input buffers.
// ===========================================================================
template <int Rows, int Stages, bool Swizzle> struct Config {
  static constexpr int kRows = Rows;       // input tile: kRows x 32 floats
  static constexpr int kStages = Stages;   // input buffers in the ring
  static constexpr bool kSwizzle = Swizzle;
  static constexpr int kConsumers = Rows / 32; // one 32x32 slice per warp
  static constexpr int kThreads = 32 * (1 + kConsumers);
  static constexpr int kInBytes = Rows * kTile * 4;
  static constexpr int kOutBytes = kTile * kTile * 4; // one 32x32 box
  static constexpr int kOutBufs = 2; // per consumer warp
  static constexpr int kSmemBytes = 1024 /* alignment slack */ +
                                    Stages * kInBytes +
                                    kConsumers * kOutBufs * kOutBytes +
                                    2 * Stages * 8 /* barriers */;
  static_assert(Rows % 32 == 0 && Rows <= 256, "TMA box height <= 256");
};

// ---------------------------------------------------------------------------
// TODO(3a) — float offset of element (row, col) in a tile whose rows are 128
// bytes (32 floats), stored by TMA with CU_TENSOR_MAP_SWIZZLE_128B
// (lesson §4.3): the 16-byte chunk q = col / 4 of row r is stored at chunk
// position q ^ (r % 8). Without swizzle it is simply row * 32 + col.
// ---------------------------------------------------------------------------
template <bool kSwizzle>
__device__ __forceinline__ int tile_offset(int row, int col) {
  if constexpr (kSwizzle) {
    LAB_TODO(3);
    return 0;
  } else {
    return row * kTile + col;
  }
}

// out[i][j] = in[j][i] for one 32x32 slice, executed by one warp. `in`
// points at row 0 of the slice (a multiple of 32 rows into a 1024-byte
// aligned tile, so row % 8 is the same as in the whole tile); `out` is a
// 32x32 box buffer laid out like the y box.
template <bool kSwizzle>
__device__ __forceinline__ void transpose_slice(const float *in, float *out,
                                                int lane) {
  if constexpr (!kSwizzle) {
    // Scalar version (given): lane j reads column i of `in` -- all 32 lanes
    // in the same bank, a 32-way conflict on every load.
#pragma unroll 8
    for (int i = 0; i < kTile; ++i)
      out[i * kTile + lane] = in[lane * kTile + i];
  } else {
    // -----------------------------------------------------------------------
    // TODO(3b) — conflict-free version for 128B-swizzled `in` and `out`
    // (lesson §5.3). View the slice as 8 x 8 blocks of 4 x 4 floats; lane l
    // handles blocks (R, C) = (l % 8, (l % 8 + l / 8 + 4 t) % 8), t = 0, 1.
    // For one block: load rows 4R..4R+3, columns 4C..4C+3 of `in` as four
    // float4 (use tile_offset<true>), transpose the 4x4 values in
    // registers, store them as four float4 to rows 4C..4C+3, columns
    // 4R..4R+3 of `out`.
    // -----------------------------------------------------------------------
    LAB_TODO(3);
  }
}

// ---------------------------------------------------------------------------
// TODO(2b) — persistent, pipelined transpose (lesson §5.2).
// Tiles: kRows x 32 of x, numbered t = tile_row * tiles_c + tile_col; this
// CTA handles t = blockIdx.x, blockIdx.x + gridDim.x, ...; `it` counts this
// CTA's tiles (0, 1, 2, ...). Tile number it uses input buffer s = it % S.
// Shared memory (carved out below): in[S][kRows][32], out[kConsumers][2]
// [32][32], full[S], empty[S].
//   Setup (thread 0): full[s] expects 1 arrival (the producer's
//     arrive_expect_tx); empty[s] expects kConsumers arrivals (one per
//     consumer warp). fence_barrier_init, then __syncthreads -- the only
//     CTA-wide barrier in the kernel.
//   Warp 0 = producer: one elected lane loops over the tiles:
//     wait until buffer s is empty (its fill n = it / S - 1 was consumed;
//     which parity? what happens for n = -1 on a fresh barrier?), arm
//     full[s] with kInBytes, TMA-load the tile into in[s].
//   Warps 1..kConsumers = consumers; warp cw = warp - 1 owns rows
//     32cw..32cw+31 of every tile and its two out buffers ob = out[cw][it % 2]:
//     - make sure the store issued from ob two tiles ago finished reading it
//       (lane 0: bulk_wait_group_read<?>, then __syncwarp);
//     - wait until full[s] completed fill n = it / S (parity?);
//     - transpose_slice<Cfg::kSwizzle>(slice of in[s], ob, lane);
//     - fence_proxy_async (all lanes), __syncwarp;
//     - lane 0: arrive on empty[s]; TMA-store ob as the y box with origin
//       (x = tile_row * kRows + 32cw, y = tile_col * 32); commit.
//   At the end every consumer lane 0 waits for all its stores
//   (bulk_wait_group<0>) -- the CTA's shared memory must outlive them.
// ---------------------------------------------------------------------------
template <class Cfg>
__global__ void __launch_bounds__(Cfg::kThreads)
    transpose_pipelined(const __grid_constant__ CUtensorMap x_map,
                        const __grid_constant__ CUtensorMap y_map, int rows,
                        int cols) {
  constexpr int S = Cfg::kStages;
  extern __shared__ uint8_t smem_raw[];
  // Round up to 1024 bytes: TMA requires 128B-swizzled buffers to start on
  // a 1024-byte boundary (the swizzle pattern repeats every 1024 bytes).
  uint8_t *base = smem_raw + ((1024 - (shared_addr(smem_raw) & 1023)) & 1023);
  float *in = reinterpret_cast<float *>(base); // S x [kRows][32]
  float *out = reinterpret_cast<float *>(base + S * Cfg::kInBytes);
  uint64_t *full = reinterpret_cast<uint64_t *>(
      base + S * Cfg::kInBytes +
      Cfg::kConsumers * Cfg::kOutBufs * Cfg::kOutBytes);
  uint64_t *empty = full + S;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tiles_c = (cols + kTile - 1) / kTile;
  const int total = (rows + Cfg::kRows - 1) / Cfg::kRows * tiles_c;
  LAB_TODO(2);
}

using Step2 = Config<128, 4, false>;
// ---------------------------------------------------------------------------
// TODO(3c) — tune Config<Rows, Stages, true> for step 3 (lesson §6 step 3):
// start from <128, 4, true>, then try other tile heights (32..256) and
// stage counts; keep the fastest on both perf cases.
// ---------------------------------------------------------------------------
using Step3 = Config<128, 4, true>;

template <class Kernel> void allow_smem(Kernel kernel, int bytes) {
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  LAB_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
}

// Persistent grid: as many CTAs as fit on the GPU at once, at most one per
// tile.
template <class Cfg> int persistent_grid(int rows, int cols) {
  int device = 0, sms = 0, per_sm = 0;
  LAB_CUDA_CHECK(cudaGetDevice(&device));
  LAB_CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  LAB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, transpose_pipelined<Cfg>, Cfg::kThreads, Cfg::kSmemBytes));
  LAB_REQUIRE(per_sm > 0, "configuration does not fit on an SM");
  const long tiles = static_cast<long>((rows + Cfg::kRows - 1) / Cfg::kRows) *
                     ((cols + kTile - 1) / kTile);
  return static_cast<int>(std::min<long>(tiles, static_cast<long>(sms) * per_sm));
}

template <class Cfg> CUtensorMapSwizzle swizzle_of() {
  return Cfg::kSwizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
}

} // namespace

struct Plan {
  CUtensorMap x_map, y_map; // encoded once, passed by value to the kernel
  int grid = 0;
};

Plan make_plan(const lab::TransposeProblem &p) {
  LAB_REQUIRE(p.dtype == lab::DType::F32, "this lab uses FP32");
  LAB_REQUIRE(p.rows % 4 == 0 && p.cols % 4 == 0,
              "rows and cols must be multiples of 4 (TMA: 16-byte row pitch)");
  Plan plan;
  if (p.step == 1) {
    plan.x_map = encode_2d(p.x, p.rows, p.cols, kTile, kTile,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
    plan.y_map = encode_2d(p.y, p.cols, p.rows, kTile, kTile,
                           CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (p.step == 2) {
    allow_smem(transpose_pipelined<Step2>, Step2::kSmemBytes);
    plan.x_map = encode_2d(p.x, p.rows, p.cols, Step2::kRows, kTile,
                           swizzle_of<Step2>());
    plan.y_map = encode_2d(p.y, p.cols, p.rows, kTile, kTile,
                           swizzle_of<Step2>());
    plan.grid = persistent_grid<Step2>(p.rows, p.cols);
  } else {
    allow_smem(transpose_pipelined<Step3>, Step3::kSmemBytes);
    plan.x_map = encode_2d(p.x, p.rows, p.cols, Step3::kRows, kTile,
                           swizzle_of<Step3>());
    plan.y_map = encode_2d(p.y, p.cols, p.rows, kTile, kTile,
                           swizzle_of<Step3>());
    plan.grid = persistent_grid<Step3>(p.rows, p.cols);
  }
  return plan;
}

void run(const Plan &plan, const lab::TransposeProblem &p,
         cudaStream_t stream) {
  switch (p.step) {
  case 1: {
    dim3 grid((p.cols + kTile - 1) / kTile, (p.rows + kTile - 1) / kTile);
    transpose_one_tile<<<grid, kOneTileThreads, 0, stream>>>(plan.x_map,
                                                             plan.y_map);
    break;
  }
  case 2:
    transpose_pipelined<Step2><<<plan.grid, Step2::kThreads,
                                 Step2::kSmemBytes, stream>>>(
        plan.x_map, plan.y_map, p.rows, p.cols);
    break;
  default:
    transpose_pipelined<Step3><<<plan.grid, Step3::kThreads,
                                 Step3::kSmemBytes, stream>>>(
        plan.x_map, plan.y_map, p.rows, p.cols);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
