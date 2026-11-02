#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;
void* prepare_attention(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, int, int, int, int, bool, float, bool, uintptr_t, bool, bool);
void run_attention(void*);
void delete_attention(void*);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", [](uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o, uintptr_t lse,
                        int b, int n, int h, int d, bool half, float scale, bool causal, uintptr_t stream, bool polynomial, bool delayed_rescale) {
        return py::capsule(prepare_attention(q,k,v,o,lse,b,n,h,d,half,scale,causal,stream,polynomial,delayed_rescale), [](void* p) { delete_attention(p); });
    });
    m.def("run", [](py::capsule state) { run_attention(state.get_pointer()); });
}
