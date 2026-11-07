#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cuda_api.h"
namespace py = pybind11;

TensorMap unpack(const py::dict &tensors) {
  TensorMap result;
  for (auto item : tensors) {
    auto desc = py::cast<py::tuple>(item.second);
    result.emplace(
        py::cast<std::string>(item.first),
        TensorView{reinterpret_cast<void *>(desc[0].cast<uintptr_t>()),
                   desc[1].cast<std::vector<int64_t>>(),
                   static_cast<DType>(desc[2].cast<int>())});
  }
  return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", [](py::dict in, py::dict out, py::dict work, Options opts,
                     uintptr_t stream) {
    student_launch(unpack(in), unpack(out), unpack(work), opts,
                   reinterpret_cast<cudaStream_t>(stream));
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
  });
}
