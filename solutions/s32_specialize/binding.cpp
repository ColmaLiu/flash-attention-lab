#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;
void *prepare_gemm(uintptr_t, uintptr_t, uintptr_t, int, int, int, uintptr_t,
                   int, int, int, int, int, int);
void run_gemm(void *);
void delete_gemm(void *);
size_t gemm_workspace_bytes(void *);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", [](uintptr_t a, uintptr_t b, uintptr_t c, int M, int N,
                      int K, uintptr_t stream, int tile_m, int tile_n,
                      int tile_k, int split_k, int stages, int single_issuer) {
    return py::capsule(prepare_gemm(a, b, c, M, N, K, stream, tile_m, tile_n,
                                    tile_k, split_k, stages, single_issuer),
                       [](void *p) { delete_gemm(p); });
  });
  m.def("workspace_bytes",
        [](py::capsule p) { return gemm_workspace_bytes(p.get_pointer()); });
  m.def("run", [](py::capsule p) { run_gemm(p.get_pointer()); });
}
