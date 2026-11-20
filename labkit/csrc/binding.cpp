// TVM FFI binding: Python tensors -> lab::*Problem -> the kernel's Plan.
// Compiled once per kernel.cu with exactly one of LAB_KIND_TRANSPOSE,
// LAB_KIND_SOFTMAX, LAB_KIND_GEMM, LAB_KIND_ATTENTION defined.
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/string.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "lab/register.h"

namespace {
namespace ffi = tvm::ffi;
using Tensors = ffi::Map<ffi::String, ffi::Tensor>;
using Options = ffi::Map<ffi::String, double>;

#if defined(LAB_KIND_TRANSPOSE)
using Problem = lab::TransposeProblem;
#elif defined(LAB_KIND_SOFTMAX)
using Problem = lab::SoftmaxProblem;
#elif defined(LAB_KIND_GEMM)
using Problem = lab::GemmProblem;
#elif defined(LAB_KIND_ATTENTION)
using Problem = lab::AttentionProblem;
#else
#error "define one LAB_KIND_*"
#endif

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error("binding: " + msg);
}

lab::DType dtype_of(const ffi::Tensor &t) {
  auto d = t.dtype();
  if (d.lanes == 1) {
    if (d.code == kDLFloat && d.bits == 32)
      return lab::DType::F32;
    if (d.code == kDLFloat && d.bits == 16)
      return lab::DType::F16;
    if (d.code == kDLBfloat && d.bits == 16)
      return lab::DType::BF16;
    if (d.code == kDLFloat8_e4m3fn && d.bits == 8)
      return lab::DType::E4M3;
  }
  fail("unsupported tensor dtype");
}

struct Unpacker {
  const Tensors &tensors;
  const Options &options;
  int device = -1;
  ffi::Array<ffi::Tensor> owners;

  ffi::Tensor get(const char *name) {
    auto it = tensors.find(name);
    if (it == tensors.end())
      fail(std::string("missing tensor '") + name + "'");
    ffi::Tensor t = (*it).second;
    if (t.device().device_type != kDLCUDA)
      fail(std::string(name) + " must be a CUDA tensor");
    if (!t.IsContiguous())
      fail(std::string(name) + " must be contiguous");
    if (device < 0)
      device = t.device().device_id;
    if (t.device().device_id != device)
      fail("all tensors must be on the same GPU");
    owners.push_back(t);
    return t;
  }
  void *ptr(const ffi::Tensor &t) {
    auto p = reinterpret_cast<uintptr_t>(t.data_ptr()) + t.byte_offset();
    if (p % 16 != 0)
      fail("tensors must be 16-byte aligned");
    return reinterpret_cast<void *>(p);
  }
  int dim(const ffi::Tensor &t, int axis) {
    if (axis >= t.ndim())
      fail("tensor rank too small");
    int64_t s = t.size(axis);
    if (s <= 0 || s > std::numeric_limits<int>::max())
      fail("tensor dimensions must be in [1, 2^31)");
    return static_cast<int>(s);
  }
  void expect_shape(const ffi::Tensor &t, std::initializer_list<int> shape,
                    const char *name) {
    if (t.ndim() != static_cast<int>(shape.size()))
      fail(std::string(name) + " has the wrong rank");
    int i = 0;
    for (int s : shape)
      if (t.size(i++) != s)
        fail(std::string(name) + " has the wrong shape");
  }
  double option(const char *name, double fallback) {
    auto it = options.find(name);
    return it == options.end() ? fallback : (*it).second;
  }
  float positive_finite(const char *name, double fallback) {
    double v = option(name, fallback);
    if (!std::isfinite(v) || v <= 0)
      fail(std::string(name) + " must be finite and > 0");
    return static_cast<float>(v);
  }
  template <class P> void common(P &p) {
    p.step = static_cast<int>(option("step", 0));
    p.probe = nullptr;
    p.probe_numel = 0;
    if (tensors.find("probe") != tensors.end()) {
      auto t = get("probe");
      if (dtype_of(t) != lab::DType::F32)
        fail("probe must be FP32");
      p.probe = static_cast<float *>(ptr(t));
      p.probe_numel = t.numel();
    }
  }

  Problem build();
};

#if defined(LAB_KIND_TRANSPOSE)
Problem Unpacker::build() {
  Problem p{};
  auto x = get("x"), y = get("y");
  p.rows = dim(x, 0);
  p.cols = dim(x, 1);
  expect_shape(x, {p.rows, p.cols}, "x");
  expect_shape(y, {p.cols, p.rows}, "y");
  p.dtype = dtype_of(x);
  if (dtype_of(y) != p.dtype)
    fail("x and y must have the same dtype");
  p.x = ptr(x);
  p.y = ptr(y);
  common(p);
  return p;
}
#elif defined(LAB_KIND_SOFTMAX)
Problem Unpacker::build() {
  Problem p{};
  auto x = get("x"), y = get("y");
  p.rows = dim(x, 0);
  p.cols = dim(x, 1);
  expect_shape(x, {p.rows, p.cols}, "x");
  expect_shape(y, {p.rows, p.cols}, "y");
  if (dtype_of(x) != lab::DType::F32 || dtype_of(y) != lab::DType::F32)
    fail("softmax uses FP32 tensors");
  p.x = static_cast<const float *>(ptr(x));
  p.y = static_cast<float *>(ptr(y));
  common(p);
  return p;
}
#elif defined(LAB_KIND_GEMM)
Problem Unpacker::build() {
  Problem p{};
  auto a = get("a"), b = get("b"), c = get("c");
  p.m = dim(a, 0);
  p.k = dim(a, 1);
  p.n = dim(b, 0);
  expect_shape(a, {p.m, p.k}, "a");
  expect_shape(b, {p.n, p.k}, "b");
  expect_shape(c, {p.m, p.n}, "c");
  p.ab_dtype = dtype_of(a);
  if (dtype_of(b) != p.ab_dtype)
    fail("a and b must have the same dtype");
  p.c_dtype = dtype_of(c);
  p.a = ptr(a);
  p.b = ptr(b);
  p.c = ptr(c);
  p.a_scale = positive_finite("a_scale", 1.0);
  p.b_scale = positive_finite("b_scale", 1.0);
  common(p);
  return p;
}
#elif defined(LAB_KIND_ATTENTION)
Problem Unpacker::build() {
  Problem p{};
  auto q = get("q"), k = get("k"), v = get("v"), o = get("o"), lse = get("lse");
  p.b = dim(q, 0);
  p.n = dim(q, 1);
  p.h = dim(q, 2);
  p.d = dim(q, 3);
  for (auto [t, name] : {std::pair{q, "q"}, {k, "k"}, {v, "v"}, {o, "o"}})
    expect_shape(t, {p.b, p.n, p.h, p.d}, name);
  expect_shape(lse, {p.b, p.h, p.n}, "lse");
  p.dtype = dtype_of(q);
  if (dtype_of(k) != p.dtype || dtype_of(v) != p.dtype)
    fail("q, k and v must have the same dtype");
  p.o_dtype = dtype_of(o);
  if (dtype_of(lse) != lab::DType::F32)
    fail("lse must be FP32");
  p.q = ptr(q);
  p.k = ptr(k);
  p.v = ptr(v);
  p.o = ptr(o);
  p.lse = static_cast<float *>(ptr(lse));
  p.scale = positive_finite("scale", 1.0 / std::sqrt(double(p.d)));
  p.causal = option("causal", 0.0) != 0.0;
  p.q_descale = positive_finite("q_descale", 1.0);
  p.k_descale = positive_finite("k_descale", 1.0);
  p.v_descale = positive_finite("v_descale", 1.0);
  common(p);
  return p;
}
#endif

class PlanObj : public ffi::Object {
public:
  ffi::Array<ffi::Tensor> owners; // keep tensors alive while the plan exists
  Problem problem;
  void *plan = nullptr;
  int device = 0;

  ~PlanObj() {
    if (plan != nullptr) {
      ffi::CUDADeviceGuard guard(device);
      lab::entry::destroy(plan, static_cast<const Problem *>(nullptr));
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("labkit." LAB_FFI_MODULE ".Plan", PlanObj,
                                    ffi::Object);
};

ffi::ObjectRef prepare(Tensors tensors, Options options) {
  Unpacker u{tensors, options};
  Problem p = u.build();
  auto obj = ffi::make_object<PlanObj>();
  obj->owners = u.owners;
  obj->problem = p;
  obj->device = u.device;
  ffi::CUDADeviceGuard guard(u.device);
  obj->plan = lab::entry::make(p);
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    fail(std::string("make_plan: ") + cudaGetErrorString(err));
  return ffi::ObjectRef(std::move(obj));
}

void run(ffi::ObjectRef ref, int64_t stream) {
  auto *obj = const_cast<PlanObj *>(ref.as<PlanObj>());
  if (obj == nullptr)
    fail("run() expects a plan from this module");
  ffi::CUDADeviceGuard guard(obj->device);
  lab::entry::run(obj->plan, obj->problem,
                  reinterpret_cast<cudaStream_t>(stream));
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    fail(std::string("launch failed: ") + cudaGetErrorString(err));
}

int64_t todo_value() {
  return lab::entry::todo_value();
}
} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(todo_value, todo_value);
