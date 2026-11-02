#include "cuda_api.h"
#include <cuda.h>
#include <algorithm>

namespace {
__device__ uint32_t shared_address(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ void issue(const CUtensorMap* descriptor, float* tile, uint64_t* barrier, int col, int row) {
    uint32_t bar = shared_address(barrier), dst = shared_address(tile);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;" :: "r"(bar) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(descriptor), "r"(col), "r"(row), "r"(bar) : "memory");
}
__device__ void wait(uint64_t* barrier, int phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "poll: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra poll;\n}"
                 :: "r"(shared_address(barrier)), "r"(phase) : "memory");
}
// Persistent CTAs cycle two TMA buffers. The next transfer overlaps stores from
// the current tile. Per-buffer phase flips only after that buffer is consumed.
__global__ void transpose_tma(const CUtensorMap* descriptor, float* out, int rows, int cols) {
    __shared__ __align__(128) float tile[2][32][32];
    __shared__ float padded[32][33];
    __shared__ __align__(8) uint64_t barriers[2];
    int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(&barriers[0])) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(&barriers[1])) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    int columns = (cols + 31) / 32;
    int total = ((rows + 31) / 32) * columns;
    int phase[2] = {0, 0};
    int slot = 0;
    if (tid == 0) issue(descriptor, &tile[0][0][0], &barriers[0], (blockIdx.x % columns) * 32, (blockIdx.x / columns) * 32);
    for (int index = blockIdx.x; index < total; index += gridDim.x) {
        wait(&barriers[slot], phase[slot]);
        int next = index + gridDim.x;
        if (tid == 0 && next < total)
            issue(descriptor, &tile[slot ^ 1][0][0], &barriers[slot ^ 1], (next % columns) * 32, (next / columns) * 32);
        // Repacking the plain TMA tile into a padded tile avoids bank conflicts
        // during transpose. The extra copy is visible overhead, not hidden setup.
        for (int offset = 0; offset < 32; offset += 8)
            padded[threadIdx.y + offset][threadIdx.x] = tile[slot][threadIdx.y + offset][threadIdx.x];
        // Every reader orders its generic loads before this slot is eventually
        // overwritten by another TMA transaction through the async proxy.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        int input_row = (index / columns) * 32 + threadIdx.x;
        int input_col = (index % columns) * 32 + threadIdx.y;
        for (int offset = 0; offset < 32; offset += 8)
            if (input_row < rows && input_col + offset < cols)
                out[(int64_t)(input_col + offset) * rows + input_row] = padded[threadIdx.x][threadIdx.y + offset];
        __syncthreads();  // All consumers release this slot before its next issue.
        phase[slot] ^= 1;
        slot ^= 1;
    }
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(shared_address(&barriers[0])) : "memory");
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(shared_address(&barriers[1])) : "memory");
    }
}
}

// Called once during prepare. No descriptor encoding or transfer is timed.
void prepare_tma_descriptor(uintptr_t address, uintptr_t destination, int64_t rows, int64_t cols, uintptr_t stream_value) {
    alignas(64) CUtensorMap descriptor;
    cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols * sizeof(float))};
    cuuint32_t box[2] = {32, 32}, element_strides[2] = {1, 1};
    auto status = cuTensorMapEncodeTiled(&descriptor, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        reinterpret_cast<void*>(address), dims, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr; cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "cuTensorMapEncodeTiled failed");
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    auto error = cudaMemcpyAsync(reinterpret_cast<void*>(destination), &descriptor, sizeof(descriptor), cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    // The stack descriptor remains alive until the transfer has completed.
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options&, cudaStream_t stream) {
    const auto& x = inputs.at("x");
    int rows = x.shape[0], cols = x.shape[1];
    if (x.dtype != DType::F32 || cols % 4 != 0) throw std::runtime_error("TMA solution requires FP32 and C divisible by four");
    int tiles = ((rows + 31) / 32) * ((cols + 31) / 32);
    transpose_tma<<<std::min(128, tiles), dim3(32, 8), 0, stream>>>(
        static_cast<const CUtensorMap*>(workspace.at("descriptor").data),
        static_cast<float*>(outputs.at("y").data), rows, cols);
}
