#include "cuda_api.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
// Compile the shared C++ dependency through the normal infrastructure build.
static_assert(decltype(cute::size(cute::Layout<cute::Shape<cute::_2, cute::_4>>{}))::value == 8);
// Infrastructure-only sentinel. Not an exercise, not callable through student API.
__global__ void infrastructure_sentinel(int* out) { *out = 7; }
void student_launch(const TensorMap&, const TensorMap& out, const TensorMap&,
                    const Options&, cudaStream_t stream) {
    infrastructure_sentinel<<<1, 1, 0, stream>>>(static_cast<int*>(out.at("sentinel").data));
}
