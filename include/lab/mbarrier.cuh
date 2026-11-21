// mbarrier: the shared-memory barrier object that asynchronous copies (TMA)
// and tcgen05 operations report completion to. Written by learners in
// 30_tma_mbarrier; provided from 31_tcgen05_gemm on.
//
// Every function wraps one PTX instruction (or a fixed retry loop around one)
// and contains no algorithm logic. PTX section names refer to the PTX ISA
// (version 8.7), https://docs.nvidia.com/cuda/parallel-thread-execution/
//
// Model ("Parallel Synchronization and Communication Instructions: mbarrier"):
// an mbarrier is a 64-bit word in shared memory holding
//   - the current phase (its parity is 0 or 1),
//   - a pending-arrival count, reset to the expected count at every phase,
//   - a transaction count (tx-count) in bytes.
// The current phase completes when pending arrivals == 0 AND tx-count == 0;
// the barrier then moves to the next phase (parity flips) and reloads the
// arrival count. Threads wait for a phase by its parity.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lab {

// 32-bit shared-memory address of a generic pointer, the form every
// .shared::cta / .shared::cluster operand expects.
// (PTX: "Generic Addressing"; CUDA: __cvta_generic_to_shared.)
__device__ __forceinline__ uint32_t shared_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// mbarrier.init.shared::cta.b64 [bar], count;
// PTX: "mbarrier.init". Phase 0, `count` arrivals expected per phase,
// tx-count 0. One thread initializes; the barrier must be 8-byte aligned.
// Make the initialization visible (fence_barrier_init + __syncthreads)
// before any other thread or the async proxy (TMA, tcgen05) uses it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared_addr(bar)),
      "r"(count)
      : "memory");
}

// fence.mbarrier_init.release.cluster;
// PTX: "Parallel Synchronization and Communication Instructions: membar /
// fence". Orders prior mbarrier.init before later operations of other
// threads and of the async proxy (a TMA that will complete_tx on it).
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// mbarrier.inval.shared::cta.b64 [bar];
// PTX: "mbarrier.inval". Required before the 8 bytes are reused as ordinary
// shared memory; optional at kernel exit.
__device__ __forceinline__ void mbar_inval(uint64_t *bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared_addr(bar))
               : "memory");
}

// mbarrier.arrive.shared::cta.b64 _, [bar];
// PTX: "mbarrier.arrive". One arrival (release semantics: this thread's
// earlier memory accesses happen before the phase completes).
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(shared_addr(bar))
      : "memory");
}

// mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes;
// PTX: "mbarrier.expect_tx", "mbarrier.arrive". Adds `bytes` to the
// tx-count of the current phase, then arrives once. The phase can only
// complete after asynchronous copies have reported (complete_tx) exactly
// that many bytes. Issue it before (or together with) the copies it covers.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar,
                                                      uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   shared_addr(bar)),
               "r"(bytes)
               : "memory");
}

// mbarrier.try_wait.parity.shared::cta.b64 p, [bar], parity;
// PTX: "mbarrier.test_wait / mbarrier.try_wait". True once the phase with
// the given parity has completed (acquire semantics). It may suspend the
// thread for a system-dependent time before returning false.
// The phase *before* the current one counts as completed, so on a freshly
// initialized barrier (current phase 0) try_wait(parity 1) is true at once.
__device__ __forceinline__ bool mbar_try_wait(uint64_t *bar, uint32_t parity) {
  uint32_t done;
  asm volatile("{\n\t.reg .pred p;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, p;\n\t}"
               : "=r"(done)
               : "r"(shared_addr(bar)), "r"(parity)
               : "memory");
  return done != 0;
}

// Blocks until the phase with the given parity has completed.
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  asm volatile("{\n\t.reg .pred p;\n"
               "LAB_MBAR_WAIT_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
               "@!p bra LAB_MBAR_WAIT_%=;\n\t}" ::"r"(shared_addr(bar)),
               "r"(parity)
               : "memory");
}

// fence.proxy.async.shared::cta;
// PTX: "membar / fence" (proxy fences), "Memory Consistency Model: Proxies".
// Ordinary loads/stores use the generic proxy; TMA and tcgen05 access shared
// memory through the async proxy. A thread that WROTE shared memory with
// ordinary stores must execute this fence before the data is handed (via a
// barrier) to a TMA store or an MMA that reads it. Every writing thread
// fences its own writes; one thread's fence does not cover the others.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// elect.sync %r|%p, membermask;
// PTX: "Parallel Synchronization and Communication Instructions: elect.sync".
// All 32 lanes of the warp call it; exactly one active lane gets true.
__device__ __forceinline__ bool elect_one() {
  uint32_t pred;
  asm volatile("{\n\t.reg .pred p;\n\t.reg .b32 r;\n\t"
               "elect.sync r|p, 0xffffffff;\n\t"
               "selp.u32 %0, 1, 0, p;\n\t}"
               : "=r"(pred));
  return pred != 0;
}

} // namespace lab
