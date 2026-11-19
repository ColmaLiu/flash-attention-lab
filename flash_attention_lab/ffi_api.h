#pragma once

#include "cuda_api.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <limits>
#include <memory>

// Host bindings translate the FFI boundary into the learner-facing CUDA ABI.
// Device translation units do not include this header.
namespace lab_ffi {
namespace ffi = tvm::ffi;
using TensorArgs = ffi::Map<ffi::String, ffi::Tensor>;
using Config = ffi::Map<ffi::String, double>;

inline DType dtype(ffi::TensorView tensor) {
  auto type = tensor.dtype();
  if (type.lanes == 1) {
    if (type.code == kDLFloat && type.bits == 32)
      return DType::F32;
    if (type.code == kDLFloat && type.bits == 16)
      return DType::F16;
    if (type.code == kDLBfloat && type.bits == 16)
      return DType::BF16;
    if (type.code == kDLInt && type.bits == 32)
      return DType::I32;
    if (type.code == kDLUInt && type.bits == 8)
      return DType::U8;
    if (type.code == kDLInt && type.bits == 64)
      return DType::I64;
  }
  throw std::runtime_error("unsupported tensor dtype");
}

inline uintptr_t address(ffi::TensorView tensor) {
  // DLPack producers can supply a byte offset independently of data.
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) + tensor.byte_offset();
}

inline void check_tensor(ffi::TensorView tensor, int &device) {
  if (tensor.device().device_type != kDLCUDA || !tensor.IsContiguous())
    throw std::runtime_error("binding requires contiguous CUDA tensors");
  if (device < 0)
    device = tensor.device().device_id;
  if (tensor.device().device_id != device)
    throw std::runtime_error("binding tensors must use the same CUDA device");
  for (int i = 0; i < tensor.ndim(); ++i)
    if (tensor.size(i) <= 0)
      throw std::runtime_error("tensor dimensions must be positive");
  (void)dtype(tensor);
}

inline TensorMap unpack(const TensorArgs &args, int &device) {
  TensorMap result;
  for (const auto &[name, tensor] : args) {
    check_tensor(tensor, device);
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.ndim(); ++i)
      shape.push_back(tensor.size(i));
    result.emplace(std::string(name),
                   TensorView{reinterpret_cast<void *>(address(tensor)),
                              std::move(shape), dtype(tensor)});
  }
  return result;
}

inline void launch(TensorArgs inputs, TensorArgs outputs, TensorArgs workspace,
                   Config config, int64_t stream) {
  int device = -1;
  auto in = unpack(inputs, device);
  auto out = unpack(outputs, device);
  auto work = unpack(workspace, device);
  Options options;
  for (const auto &[name, value] : config)
    options.emplace(std::string(name), value);
  // Empty maps must still reach an unimplemented student's explicit error.
  if (device < 0) {
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(error));
  }
  ffi::CUDADeviceGuard guard(device);
  student_launch(in, out, work, options,
                 reinterpret_cast<cudaStream_t>(stream));
  auto error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

inline void check_shape(ffi::TensorView tensor,
                        std::initializer_list<int64_t> expected) {
  if (tensor.ndim() != static_cast<int>(expected.size()))
    throw std::runtime_error("binding tensor rank mismatch");
  int axis = 0;
  for (auto size : expected)
    if (tensor.size(axis++) != size)
      throw std::runtime_error("binding tensor shape mismatch");
}

inline int dimension(ffi::TensorView tensor, int axis) {
  if (axis >= tensor.ndim() || tensor.size(axis) <= 0 ||
      tensor.size(axis) > std::numeric_limits<int>::max())
    throw std::runtime_error("binding dimension is outside the int32 range");
  return static_cast<int>(tensor.size(axis));
}

inline int check_gemm(ffi::TensorView a, ffi::TensorView b, ffi::TensorView c) {
  int device = -1;
  for (auto tensor : {a, b, c}) {
    check_tensor(tensor, device);
    if (tensor.ndim() != 2 || address(tensor) % 16 != 0)
      throw std::runtime_error("GEMM requires 16-byte aligned matrices");
  }
  int m = dimension(a, 0), k = dimension(a, 1), n = dimension(b, 1);
  check_shape(b, {k, n});
  check_shape(c, {m, n});
  if (dtype(a) != DType::BF16 || dtype(b) != DType::BF16 ||
      dtype(c) != DType::F32 || m % 64 || n % 8 || k % 16)
    throw std::runtime_error(
        "GEMM requires BF16 A/B, FP32 C and M%64=N%8=K%16=0");
  return device;
}

inline int check_attention(ffi::TensorView q, ffi::TensorView k,
                           ffi::TensorView v, ffi::TensorView o,
                           ffi::TensorView lse) {
  int device = -1;
  for (auto tensor : {q, k, v, o, lse})
    check_tensor(tensor, device);
  if (q.ndim() != 4 || (dtype(q) != DType::F16 && dtype(q) != DType::BF16))
    throw std::runtime_error("attention requires FP16/BF16 BNHD tensors");
  int b = dimension(q, 0), n = dimension(q, 1);
  int h = dimension(q, 2), d = dimension(q, 3);
  for (auto tensor : {q, k, v, o}) {
    check_shape(tensor, {b, n, h, d});
    if (dtype(tensor) != dtype(q) || address(tensor) % 16 != 0)
      throw std::runtime_error(
          "attention requires matching dtypes and 16-byte alignment");
  }
  check_shape(lse, {b, h, n});
  if (dtype(lse) != DType::F32)
    throw std::runtime_error("attention LSE must be FP32");
  return device;
}

// Native state owns descriptors/scratch. FFI tensors keep all GPU allocations
// alive independently of the original Python tensor objects. The build-specific
// type key rejects states belonging to another loaded kernel.
class PreparedObj : public ffi::Object {
public:
  using Native = std::unique_ptr<void, void (*)(void *)>;
  ffi::Array<ffi::Tensor> tensors;
  Native native;
  void (*run)(void *);
  int device;
  int64_t workspace_bytes;

  PreparedObj(ffi::Array<ffi::Tensor> owners, Native state,
              void (*launcher)(void *), int device_id, int64_t bytes)
      : tensors(std::move(owners)), native(std::move(state)), run(launcher),
        device(device_id), workspace_bytes(bytes) {
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("flash_attention_lab." LAB_FFI_MODULE
                                    ".Prepared",
                                    PreparedObj, ffi::Object);
};
using Prepared = ffi::ObjectPtr<PreparedObj>;

inline Prepared prepared(ffi::Array<ffi::Tensor> tensors, void *native,
                         void (*destroy)(void *), void (*run)(void *),
                         int device, size_t bytes = 0) {
  PreparedObj::Native owned(native, destroy);
  return ffi::make_object<PreparedObj>(std::move(tensors), std::move(owned),
                                       run, device, bytes);
}

inline void run(Prepared state) {
  if (!state)
    throw std::runtime_error("prepared state must not be None");
  ffi::CUDADeviceGuard guard(state->device);
  state->run(state->native.get());
}

inline int64_t workspace_bytes(Prepared state) {
  if (!state)
    throw std::runtime_error("prepared state must not be None");
  return state->workspace_bytes;
}
} // namespace lab_ffi
