// LAB_REGISTER connects your kernel.cu to the Python harness.
//
// Every kernel.cu defines
//   struct Plan { ... };               // state prepared once
//   Plan make_plan(const Problem &p);  // host, NOT timed
//   void run(const Plan &plan, const Problem &p, cudaStream_t stream); // timed
// and ends with
//   LAB_REGISTER(lab::XxxProblem, Plan, make_plan, run);
//
// make_plan is the place for host-side preparation that should not be timed:
// encoding TMA descriptors, allocating scratch (lab::Scratch), choosing launch
// sizes. run must only launch work on `stream` (it is captured into a CUDA
// Graph for timing, so no synchronization or allocation inside run).
#pragma once

#include <cstddef>
#include "lab/check.h"
#include "lab/problem.h"

namespace lab::entry {
// Implemented by LAB_REGISTER inside the kernel translation unit.
void *make(const TransposeProblem &p);
void *make(const SoftmaxProblem &p);
void *make(const GemmProblem &p);
void *make(const AttentionProblem &p);
void run(void *plan, const TransposeProblem &p, cudaStream_t s);
void run(void *plan, const SoftmaxProblem &p, cudaStream_t s);
void run(void *plan, const GemmProblem &p, cudaStream_t s);
void run(void *plan, const AttentionProblem &p, cudaStream_t s);
void destroy(void *plan, const TransposeProblem *);
void destroy(void *plan, const SoftmaxProblem *);
void destroy(void *plan, const GemmProblem *);
void destroy(void *plan, const AttentionProblem *);
// TODO number recorded by a device-side LAB_TODO (0 if none).
int todo_value();
} // namespace lab::entry

#ifdef __CUDACC__
namespace lab::detail {
inline int *todo_host_flag() {
  static int *flag = [] {
    int *host = nullptr;
    int *device = nullptr;
    LAB_CUDA_CHECK(cudaHostAlloc(&host, sizeof(int), cudaHostAllocMapped));
    *host = 0;
    LAB_CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
    LAB_CUDA_CHECK(cudaMemcpyToSymbol(lab_todo_flag_, &device, sizeof(device)));
    return host;
  }();
  return flag;
}
} // namespace lab::detail

#define LAB_REGISTER(ProblemT, PlanT, make_fn, run_fn)                         \
  namespace lab::entry {                                                       \
  void *make(const ProblemT &p) {                                              \
    *::lab::detail::todo_host_flag() = 0;                                      \
    return new PlanT(::make_fn(p));                                            \
  }                                                                            \
  void run(void *plan, const ProblemT &p, cudaStream_t s) {                    \
    ::run_fn(*static_cast<const PlanT *>(plan), p, s);                         \
  }                                                                            \
  void destroy(void *plan, const ProblemT *) {                                 \
    delete static_cast<PlanT *>(plan);                                         \
  }                                                                            \
  int todo_value() {                                                           \
    return *reinterpret_cast<volatile int *>(::lab::detail::todo_host_flag()); \
  }                                                                            \
  }                                                                            \
  static_assert(true, "")
#endif
