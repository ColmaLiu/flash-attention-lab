#include "ffi_api.h"

void *prepare_gemm(uintptr_t, uintptr_t, uintptr_t, int, int, int, uintptr_t,
                   int, int, int, int, int, int);
void run_gemm(void *);
void delete_gemm(void *);
size_t gemm_workspace_bytes(void *);

lab_ffi::Prepared prepare(tvm::ffi::Tensor a, tvm::ffi::Tensor b,
                          tvm::ffi::Tensor c, int64_t stream, int tile_m,
                          int tile_n, int tile_k, int split_k, int stages,
                          int single_issuer) {
  int device = lab_ffi::check_gemm(a, b, c);
  tvm::ffi::Array<tvm::ffi::Tensor> owners{a, b, c};
  tvm::ffi::CUDADeviceGuard guard(device);
  void *native =
      prepare_gemm(lab_ffi::address(a), lab_ffi::address(b),
                   lab_ffi::address(c), lab_ffi::dimension(a, 0),
                   lab_ffi::dimension(b, 1), lab_ffi::dimension(a, 1), stream,
                   tile_m, tile_n, tile_k, split_k, stages, single_issuer);
  return lab_ffi::prepared(std::move(owners), native, delete_gemm, run_gemm,
                           device, gemm_workspace_bytes(native));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, lab_ffi::run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_bytes, lab_ffi::workspace_bytes);
