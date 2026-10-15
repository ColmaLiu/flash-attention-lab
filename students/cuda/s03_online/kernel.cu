#include "cuda_api.h"

// STUDENT_TODO[device-computation]: implement the exercise's device modules.
// STUDENT_TODO[launch-and-storage]: choose grid/block/shared memory and data
// ownership. All pointers are device pointers; launch on the supplied stream.
// ENTRY CONTRACT (case-sensitive keys; contiguous tensors on one GPU):
// inputs.at("x"): [R,C], DType::F32; input X; read-only.
// outputs.at("y"): [R,C], DType::F32; full-row softmax probabilities.
// outputs.at("m"): [R], DType::F32; final full-row maximum.
// outputs.at("l"): [R], DType::F32; final full-row sum of exp(X - m).
// options contains scale=1.0 and causal=0.0; neither affects this exercise.
// workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
// Outputs are preallocated, not initialized; input tensors must not be modified.
// Full contract and supported shapes: docs/tasks/s03_online.md.
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream) {
    throw std::runtime_error("STUDENT_TODO s03_online: device implementation and launch are missing");
}
