#include "cuda_api.h"

// STUDENT_TODO[device-computation]: implement the exercise's device modules.
// STUDENT_TODO[launch-and-storage]: choose grid/block/shared memory and data
// ownership. All pointers are device pointers; launch on the supplied stream.
// ENTRY CONTRACT (case-sensitive keys; contiguous tensors on one GPU):
// inputs.at("q"): [B,H,N,D], DType::BF16 / DType::F16; input Q; same shape/dtype/device, read-only.
// inputs.at("k"): [B,H,N,D], DType::BF16 / DType::F16; input K; same shape/dtype/device, read-only.
// inputs.at("v"): [B,H,N,D], DType::BF16 / DType::F16; input V; same shape/dtype/device, read-only.
// outputs.at("o"): [B,H,N,D], DType::BF16 / DType::F16; attention output O; input dtype.
// outputs.at("lse"): [B,H,N], DType::F32; natural-log logsumexp of scaled, masked scores.
// options.at("scale"): positive double (default 1/sqrt(D)); options.at("causal"): 0.0 or 1.0.
// workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
// Outputs are preallocated, not initialized; input tensors must not be modified.
// Full contract and supported shapes: docs/tasks/s24_fa2_edges.md.
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream) {
    throw std::runtime_error("STUDENT_TODO s24_fa2_edges: device implementation and launch are missing");
}
