#include "cuda_api.h"

// STUDENT_TODO[device-computation]: implement the exercise's device modules.
// STUDENT_TODO[launch-and-storage]: choose grid/block/shared memory and data
// ownership. All pointers are device pointers; launch on the supplied stream.
// ENTRY CONTRACT (case-sensitive keys; contiguous tensors on one GPU):
// inputs.at("a"): [M,K], DType::BF16 / DType::F16; left matrix A; read-only.
// inputs.at("b"): [K,N], DType::BF16 / DType::F16; right matrix B, not transposed; read-only.
// outputs.at("c"): [M,N], DType::F32; C = A B; no initial C input.
// options contains scale=1.0 and causal=0.0; neither affects this exercise.
// workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
// Outputs are preallocated, not initialized; input tensors must not be modified.
// Full contract and supported shapes: docs/tasks/s22_mma.md.
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream) {
    throw std::runtime_error("STUDENT_TODO s22_mma: device implementation and launch are missing");
}
