#include "ffi_api.h"

void *prepare_attention(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                        int, int, int, int, bool, float, bool, uintptr_t, bool,
                        bool);
void run_attention(void *);
void delete_attention(void *);

lab_ffi::Prepared prepare(tvm::ffi::Tensor q, tvm::ffi::Tensor k,
                          tvm::ffi::Tensor v, tvm::ffi::Tensor o,
                          tvm::ffi::Tensor lse, double scale, bool causal,
                          int64_t stream, bool polynomial,
                          bool delayed_rescale) {
  int device = lab_ffi::check_attention(q, k, v, o, lse);
  tvm::ffi::Array<tvm::ffi::Tensor> owners{q, k, v, o, lse};
  tvm::ffi::CUDADeviceGuard guard(device);
  void *native = prepare_attention(
      lab_ffi::address(q), lab_ffi::address(k), lab_ffi::address(v),
      lab_ffi::address(o), lab_ffi::address(lse), lab_ffi::dimension(q, 0),
      lab_ffi::dimension(q, 1), lab_ffi::dimension(q, 2),
      lab_ffi::dimension(q, 3), lab_ffi::dtype(q) == DType::F16,
      static_cast<float>(scale), causal, stream, polynomial, delayed_rescale);
  return lab_ffi::prepared(std::move(owners), native, delete_attention,
                           run_attention, device);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, lab_ffi::run);
