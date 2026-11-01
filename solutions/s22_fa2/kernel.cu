#include "cuda_api.h"
#include "fa2_optimized.cuh"
#include "wmma_readable.cuh"

// Explicit reference-version selection: optimized CuTe C++ by default;
// CONFIG["readable_wmma"]=1 selects the preserved, more heavily staged version.
void student_launch(const TensorMap& inputs, const TensorMap& outputs,
                    const TensorMap& workspace, const Options& options, cudaStream_t stream) {
    const auto readable = options.find("readable_wmma");
    if (readable != options.end() && readable->second != 0) {
        s22_reference::launch(inputs, outputs, workspace, options, stream);
        return;
    }
    const auto& q = inputs.at("q");
    const auto& k = inputs.at("k");
    const auto& v = inputs.at("v");
    const auto& o = outputs.at("o");
    const auto& lse = outputs.at("lse");
    if (q.shape.size() != 4 || q.shape[3] != 64 && q.shape[3] != 128)
        throw std::invalid_argument("s22 requires BNHD, D=64 or 128");
    const float scale = static_cast<float>(options.at("scale"));
    const bool causal = options.at("causal") != 0;
    if (q.dtype == DType::BF16)
        s22_optimized::dispatch<cutlass::bfloat16_t>(q, k, v, o, lse, scale, causal, stream);
    else if (q.dtype == DType::F16)
        s22_optimized::dispatch<cutlass::half_t>(q, k, v, o, lse, scale, causal, stream);
    else throw std::invalid_argument("s22 requires FP16/BF16");
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
