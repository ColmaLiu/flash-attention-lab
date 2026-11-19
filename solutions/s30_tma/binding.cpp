#include "ffi_api.h"

void prepare_tma_descriptor(uintptr_t, uintptr_t, int64_t, int64_t, uintptr_t);

void prepare_descriptor(tvm::ffi::TensorView input,
                        tvm::ffi::TensorView descriptor, int64_t stream) {
  int device = -1;
  lab_ffi::check_tensor(input, device);
  lab_ffi::check_tensor(descriptor, device);
  if (input.ndim() != 2 || lab_ffi::dtype(input) != DType::F32 ||
      input.size(1) % 4 || lab_ffi::address(input) % 16 ||
      lab_ffi::dtype(descriptor) != DType::U8 ||
      lab_ffi::address(descriptor) % 64)
    throw std::runtime_error("invalid TMA input or descriptor buffer");
  lab_ffi::check_shape(descriptor, {128});
  tvm::ffi::CUDADeviceGuard guard(device);
  prepare_tma_descriptor(lab_ffi::address(input), lab_ffi::address(descriptor),
                         input.size(0), input.size(1), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_descriptor, prepare_descriptor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, lab_ffi::launch);
