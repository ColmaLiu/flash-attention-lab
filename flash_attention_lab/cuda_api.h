#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Host ABI only. No thread mapping, launch dimensions, workspace or device layout.
enum class DType : int { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, U8 = 4, I64 = 5 };
struct TensorView {
    void* data;
    std::vector<int64_t> shape;
    DType dtype;
};
using TensorMap = std::unordered_map<std::string, TensorView>;
using Options = std::unordered_map<std::string, double>;
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream);
