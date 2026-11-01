#include "cuda_api.h"

// x [R,C] -> y [R,C], FP32; contiguous, one device, preallocated outputs.
// Implement device computation and choose launch/storage here.
// Use the supplied stream. No algorithm state is initialized by the host.
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options,
                    cudaStream_t stream) {
    throw std::runtime_error("STUDENT_TODO: s01_softmax");
}
