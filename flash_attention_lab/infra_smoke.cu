#include "cuda_api.h"
// Infrastructure-only sentinel. Not an exercise, not callable through student
// API.
__global__ void infrastructure_sentinel(int *out) {
  *out = 7;
}
void student_launch(const TensorMap &, const TensorMap &out, const TensorMap &,
                    const Options &, cudaStream_t stream) {
  infrastructure_sentinel<<<1, 1, 0, stream>>>(
      static_cast<int *>(out.at("sentinel").data));
}
