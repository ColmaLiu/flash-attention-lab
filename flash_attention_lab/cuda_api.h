#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Host ABI only. No thread mapping, launch dimensions, workspace or device
// layout.
enum class DType : int { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, U8 = 4, I64 = 5 };
struct TensorView {
  // Device address. Inputs are read-only by contract; output/scratch data is
  // writable.
  void *data;
  // Host metadata: logical extents in elements; tensors are contiguous, last
  // dim fastest.
  std::vector<int64_t> shape;
  DType dtype;
};
// Keys are case-sensitive and exercise-specific; use .at("key"), not iteration
// order. Each kernel.cu and docs/tasks/<id>.md lists the exact keys, shapes and
// dtypes. E.g. s00: inputs.at("x"), outputs.at("y"); attention: q/k/v -> o/lse.
using TensorMap = std::unordered_map<std::string, TensorView>;
// All values are double, including causal (0.0/1.0). scale/causal affect only
// attention. Student CONFIG supplies other numeric options; tensor dimensions
// are NOT implicit keys.
using Options = std::unordered_map<std::string, double>;
// workspace has only the keys declared by workspace_spec (none by default).
// Outputs/scratch are preallocated, with no initial algorithm state supplied by
// the host. See docs/contracts.md for configuration defaults and parameter
// ownership.
void student_launch(const TensorMap &inputs, const TensorMap &outputs,
                    const TensorMap &workspace, const Options &options,
                    cudaStream_t stream);
