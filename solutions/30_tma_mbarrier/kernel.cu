// Reference solution for 30_tma_mbarrier: y[c][r] = x[r][c] with TMA loads,
// TMA stores and mbarriers.
//
// step 1: one 32x32 tile per CTA. Thread 0 starts a TMA load and arms an
//         mbarrier with the expected byte count; everyone waits on parity 0,
//         transposes through shared memory, fences the generic writes into
//         the async proxy, and thread 0 issues a TMA store.
// step 2: persistent CTAs with a ring of kStages input buffers. Warp 0 is the
//         producer (one elected lane issues TMA loads); warps 1..4 are
//         consumers, each owning a 32-row slice of the 128x32 input tile and
//         storing its own 32x32 output box. full[s] (TMA bytes) and empty[s]
//         (one arrival per consumer warp) hand buffers back and forth.
// step 3: the same pipeline with 128-byte swizzled tiles and a 4x4 register
//         transpose (16-byte shared accesses, no bank conflicts), plus tuned
//         tile height and stage count.
//
// Supported domain: FP32, rows % 4 == 0 and cols % 4 == 0 (TMA needs the
// row pitch of x and y to be a multiple of 16 bytes). Tail tiles need no
// code: loads zero-fill and stores clip outside the tensor.
#include "lab/lab.h"
#include "lab/mbarrier.cuh"
#include "lab/tma.cuh"

#include <algorithm>

namespace {

constexpr int kTile = 32; // edge of a TMA box in floats: 32 * 4 B = 128 B

// ===========================================================================
// Step 1: one tile per CTA.
// ===========================================================================
constexpr int kOneTileThreads = 128;

__global__ void __launch_bounds__(kOneTileThreads)
    transpose_one_tile(const __grid_constant__ CUtensorMap x_map,
                       const __grid_constant__ CUtensorMap y_map) {
  // TMA without swizzle needs 128-byte aligned shared destinations.
  __shared__ __align__(128) float in[kTile][kTile];
  __shared__ __align__(128) float out[kTile][kTile];
  __shared__ __align__(8) uint64_t full;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int r0 = blockIdx.y * kTile, c0 = blockIdx.x * kTile;

  if (tid == 0) {
    lab::mbar_init(&full, 1);  // one arrival: thread 0's arrive_expect_tx
    lab::fence_barrier_init(); // visible to the async proxy (the TMA unit)
  }
  __syncthreads(); // ... and to the other threads

  if (tid == 0) {
    // Arm the barrier with the bytes the load will deliver (the whole box,
    // even if part of it is out of bounds and zero-filled), then start it.
    lab::mbar_arrive_expect_tx(&full, sizeof(in));
    lab::tma_load_2d(in, &x_map, &full, c0, r0); // (x = col, y = row)
  }
  // Phase 0 completes when the arrival is in and all 4096 bytes have landed.
  lab::mbar_wait(&full, 0);

  // out[i][j] = in[j][i]. Lane j reads column i of `in`: all 32 lanes hit
  // the same bank (32-way conflict); step 3 fixes this.
  for (int i = warp; i < kTile; i += kOneTileThreads / 32)
    out[i][lane] = in[lane][i];

  // The TMA store reads `out` through the async proxy: every thread that
  // wrote it fences its own generic-proxy writes, then the CTA barrier hands
  // them to thread 0.
  lab::fence_proxy_async();
  __syncthreads();
  if (tid == 0) {
    lab::tma_store_2d(&y_map, out, r0, c0); // y box origin: (x = r0, y = c0)
    lab::bulk_commit_group();
    // `out` must stay valid until the store has finished with it; the CTA's
    // shared memory is released when it exits.
    lab::bulk_wait_group<0>();
  }
}

// ===========================================================================
// Steps 2 and 3: persistent CTAs, a ring of input buffers.
// ===========================================================================
template <int Rows, int Stages, bool Swizzle> struct Config {
  static constexpr int kRows = Rows;     // input tile: kRows x 32 floats
  static constexpr int kStages = Stages; // input buffers in the ring
  static constexpr bool kSwizzle = Swizzle;
  static constexpr int kConsumers = Rows / 32; // one 32x32 slice per warp
  static constexpr int kThreads = 32 * (1 + kConsumers);
  static constexpr int kInBytes = Rows * kTile * 4;
  static constexpr int kOutBytes = kTile * kTile * 4; // one 32x32 box
  static constexpr int kOutBufs = 2;                  // per consumer warp
  static constexpr int kSmemBytes =
      1024 /* alignment slack */ + Stages * kInBytes +
      kConsumers * kOutBufs * kOutBytes + 2 * Stages * 8 /* barriers */;
  static_assert(Rows % 32 == 0 && Rows <= 256, "TMA box height <= 256");
};

// Float offset of element (row, col) in a tile of 128-byte rows. With the
// 128B swizzle, TMA stores 16-byte chunk q of row r at chunk q ^ (r % 8)
// (the buffer must be 1024-byte aligned).
template <bool kSwizzle>
__device__ __forceinline__ int tile_offset(int row, int col) {
  if constexpr (kSwizzle)
    return row * kTile + ((col / 4) ^ (row % 8)) * 4 + col % 4;
  else
    return row * kTile + col;
}

// out[i][j] = in[j][i] for one 32x32 slice. `in` points at row 0 of the
// slice (a multiple of 32 rows into a 1024-aligned tile, so row % 8 is the
// same as in the full tile).
template <bool kSwizzle>
__device__ __forceinline__ void transpose_slice(const float *in, float *out,
                                                int lane) {
  if constexpr (!kSwizzle) {
    // Scalar: lane j reads column i -> 32-way bank conflict on every load.
#pragma unroll 8
    for (int i = 0; i < kTile; ++i)
      out[i * kTile + lane] = in[lane * kTile + i];
  } else {
    // 4x4 blocks: lane l handles blocks (R, C) = (l % 8, (l % 8 + l / 8 +
    // 4 t) % 8), t = 0, 1, which covers all 64 blocks once. It loads rows
    // 4R..4R+3, chunk C of `in` (4 float4), transposes in registers and
    // stores rows 4C..4C+3, chunk R of `out`. Within each group of 8 lanes
    // (one shared-memory wavefront for 16-byte accesses) both the loaded
    // and the stored chunks land on 8 different swizzled positions, i.e.
    // all 32 banks: no conflict (lesson §5.3).
    const float4 *in4 = reinterpret_cast<const float4 *>(in);
    float4 *out4 = reinterpret_cast<float4 *>(out);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int R = lane % 8, C = (lane % 8 + lane / 8 + 4 * t) % 8;
      float4 a[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int row = 4 * R + k;
        a[k] = in4[row * 8 + (C ^ (row % 8))];
      }
      float4 b[4] = {{a[0].x, a[1].x, a[2].x, a[3].x},
                     {a[0].y, a[1].y, a[2].y, a[3].y},
                     {a[0].z, a[1].z, a[2].z, a[3].z},
                     {a[0].w, a[1].w, a[2].w, a[3].w}};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int row = 4 * C + k;
        out4[row * 8 + (R ^ (row % 8))] = b[k];
      }
    }
  }
}

template <class Cfg>
__global__ void __launch_bounds__(Cfg::kThreads)
    transpose_pipelined(const __grid_constant__ CUtensorMap x_map,
                        const __grid_constant__ CUtensorMap y_map, int rows,
                        int cols) {
  constexpr int S = Cfg::kStages;
  extern __shared__ uint8_t smem_raw[];
  // Round up to 1024 bytes: the 128B swizzle pattern repeats every 1024 B
  // and TMA requires swizzled buffers to start on that boundary.
  uint8_t *base =
      smem_raw + ((1024 - (lab::shared_addr(smem_raw) & 1023)) & 1023);
  float *in = reinterpret_cast<float *>(base); // S x [kRows][32]
  float *out = reinterpret_cast<float *>(base + S * Cfg::kInBytes);
  uint64_t *full = reinterpret_cast<uint64_t *>(
      base + S * Cfg::kInBytes +
      Cfg::kConsumers * Cfg::kOutBufs * Cfg::kOutBytes);
  uint64_t *empty = full + S;

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tiles_c = (cols + kTile - 1) / kTile;
  const int total = (rows + Cfg::kRows - 1) / Cfg::kRows * tiles_c;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      lab::mbar_init(&full[s], 1); // producer's arrive_expect_tx
      lab::mbar_init(&empty[s],
                     Cfg::kConsumers); // one arrival per consumer warp
    }
    lab::fence_barrier_init();
  }
  __syncthreads(); // the only CTA-wide barrier

  if (warp == 0) {
    // ---- producer: one elected lane issues every load ---------------------
    if (lab::elect_one()) {
      lab::prefetch_tensor_map(&x_map);
      int it = 0;
      for (int t = blockIdx.x; t < total; t += gridDim.x, ++it) {
        const int s = it % S;
        const uint32_t n = it / S; // how many times buffer s was filled before
        // Wait until the consumers released the previous contents of buffer
        // s (fill n-1). For n = 0 the parity is 1, which a fresh barrier
        // reports as complete at once.
        lab::mbar_wait(&empty[s], (n & 1) ^ 1);
        lab::mbar_arrive_expect_tx(&full[s], Cfg::kInBytes);
        lab::tma_load_2d(in + s * Cfg::kRows * kTile, &x_map, &full[s],
                         t % tiles_c * kTile, t / tiles_c * Cfg::kRows);
      }
    }
  } else {
    // ---- consumers: warp cw transposes rows 32cw..32cw+31 of each tile ----
    const int cw = warp - 1;
    if (lane == 0)
      lab::prefetch_tensor_map(&y_map);
    int it = 0;
    for (int t = blockIdx.x; t < total; t += gridDim.x, ++it) {
      const int s = it % S;
      const uint32_t n = it / S;
      float *ob =
          out + (cw * Cfg::kOutBufs + it % Cfg::kOutBufs) * kTile * kTile;
      // Output buffer `ob` was last stored at iteration it - 2: allow only
      // the most recent store group (it - 1) to still be reading.
      if (lane == 0)
        lab::bulk_wait_group_read<Cfg::kOutBufs - 1>();
      __syncwarp();
      lab::mbar_wait(&full[s], n & 1); // fill n of buffer s has landed

      transpose_slice<Cfg::kSwizzle>(in + (s * Cfg::kRows + cw * 32) * kTile,
                                     ob, lane);

      // Release buffer s: all lanes' reads are done (__syncwarp orders them
      // before lane 0's arrive, which has release semantics).
      // Make our generic writes to `ob` visible to the async proxy before the
      // TMA store reads them.
      lab::fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        lab::mbar_arrive(&empty[s]);
        const int r0 = t / tiles_c * Cfg::kRows + cw * 32,
                  c0 = t % tiles_c * kTile;
        lab::tma_store_2d(&y_map, ob, r0, c0);
        lab::bulk_commit_group();
      }
    }
    if (lane == 0)
      lab::bulk_wait_group<0>(); // shared memory must outlive the stores
  }
}

using Step2 = Config<128, 4, false>;
using Step3 = Config<256, 2, true>; // 256-row box, 2 stages: fastest measured

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
  return static_cast<int>(
      std::min<long>(tiles, static_cast<long>(sms) * per_sm));
}

template <class Cfg> CUtensorMap x_map_for(const lab::TransposeProblem &p) {
  return lab::make_tensor_map_2d(
      p.x, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, p.rows, p.cols, Cfg::kRows, kTile,
      Cfg::kSwizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE);
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
    plan.x_map = lab::make_tensor_map_2d(p.x, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                         p.rows, p.cols, kTile, kTile,
                                         CU_TENSOR_MAP_SWIZZLE_NONE);
    plan.y_map = lab::make_tensor_map_2d(p.y, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                         p.cols, p.rows, kTile, kTile,
                                         CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (p.step == 2) {
    allow_smem(transpose_pipelined<Step2>, Step2::kSmemBytes);
    plan.x_map = x_map_for<Step2>(p);
    plan.y_map = lab::make_tensor_map_2d(p.y, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                         p.cols, p.rows, kTile, kTile,
                                         CU_TENSOR_MAP_SWIZZLE_NONE);
    plan.grid = persistent_grid<Step2>(p.rows, p.cols);
  } else {
    allow_smem(transpose_pipelined<Step3>, Step3::kSmemBytes);
    plan.x_map = x_map_for<Step3>(p);
    plan.y_map = lab::make_tensor_map_2d(p.y, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                         p.cols, p.rows, kTile, kTile,
                                         CU_TENSOR_MAP_SWIZZLE_128B);
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
    transpose_pipelined<Step2>
        <<<plan.grid, Step2::kThreads, Step2::kSmemBytes, stream>>>(
            plan.x_map, plan.y_map, p.rows, p.cols);
    break;
  default:
    transpose_pipelined<Step3>
        <<<plan.grid, Step3::kThreads, Step3::kSmemBytes, stream>>>(
            plan.x_map, plan.y_map, p.rows, p.cols);
    break;
  }
}

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);
