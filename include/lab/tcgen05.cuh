// tcgen05 (5th-generation Tensor Core) and Tensor Memory (TMEM) on SM100.
// Written by learners in 31_tcgen05_gemm; provided from 32_ws_gemm on.
//
// Every function wraps one PTX instruction (or builds one descriptor word)
// and contains no algorithm logic. PTX section names refer to the PTX ISA
// (version 8.7), https://docs.nvidia.com/cuda/parallel-thread-execution/,
// chapter "TensorCore 5th Generation Family Instructions"; bit layouts were
// checked against CUTLASS (third_party/cutlass/include/cute/arch/
// mma_sm100_desc.hpp: SmemDescriptor, InstrDescriptor).
//
// TMEM: per SM, 128 lanes x 512 columns of 32-bit cells. An address is
// (lane << 16) | column. A CTA allocates columns (all 128 lanes); for the
// cta_group::1 MMA with M = 128, accumulator row i lives in lane i and
// column j of the N columns in column base + j. Warp w of a warpgroup may
// only access lanes 32 * (w % 4) .. 32 * (w % 4) + 31.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "lab/mbarrier.cuh"

namespace lab {

// ===========================================================================
// TMEM allocation. PTX: "Tensor Memory Allocation and Management
// Instructions": tcgen05.alloc, tcgen05.dealloc,
// tcgen05.relinquish_alloc_permit. All three are .sync.aligned: one whole warp
// executes them together.
// ===========================================================================

// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], ncols;
// Allocates `ncols` columns (a power of 2, 32 <= ncols <= 512) and writes the
// TMEM base address to shared `dst`. Blocks if the columns are not free.
// Other threads may read *dst only after
//   tcgen05_fence_before_thread_sync(); __syncthreads();
//   tcgen05_fence_after_thread_sync();
__device__ __forceinline__ void tmem_alloc(uint32_t *dst, uint32_t ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          shared_addr(dst)),
      "r"(ncols)
      : "memory");
}

// tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
// The CTA relinquishes the right to allocate Tensor Memory: after any of its
// threads executes this, a tcgen05.alloc by the CTA is illegal (PTX ISA).
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                   : "memory");
}

// tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, ncols;
// Frees what tmem_alloc returned. Every TMEM access of the CTA must be
// complete (wait_ld / MMA commit) and ordered before it with the fence +
// barrier sequence. A CTA must free all its columns before it exits.
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(ncols)
      : "memory");
}

// TMEM address of (lane, column) relative to an allocated base.
__device__ __forceinline__ uint32_t tmem_addr(uint32_t base, int lane,
                                              int col) {
  return base + (static_cast<uint32_t>(lane) << 16) +
         static_cast<uint32_t>(col);
}

// ===========================================================================
// Shared-memory matrix descriptor. PTX: "Matrix Descriptors" / "Shared Memory
// Descriptor" (tcgen05).
//
//   bits  0-13  start address >> 4
//   bits 16-29  leading-dimension byte offset (LBO) >> 4
//   bits 32-45  stride-dimension byte offset (SBO) >> 4
//   bits 46-47  version: 1 on SM100
//   bits 49-51  base offset (0 when the swizzle atoms are 1024-byte aligned)
//   bit  52     LBO mode (0)
//   bits 61-63  layout: 0 none, 1 128B with 32B atom, 2 128B, 4 64B, 6 32B
//
// K-major operand with 128-byte swizzle (what TMA writes with a 64-element
// BF16 box row and CU_TENSOR_MAP_SWIZZLE_128B): each row (M or N index) is
// 128 bytes = 64 K elements; 8 rows form a 1024-byte swizzle atom; atoms
// follow each other along M/N, so SBO = 1024. LBO is unused for swizzled
// K-major layouts (encoded as 1 by convention). To select the K slice
// k0 .. k0 + 15 of the 64-wide atom, add k0 * 2 bytes to the start address.
// ===========================================================================
__device__ __forceinline__ uint64_t
make_smem_desc_kmajor_sw128(const void *smem, uint32_t sbo_bytes = 1024) {
  const uint64_t addr = shared_addr(smem);
  return ((addr >> 4) & 0x3FFF)                        // start address
         | (uint64_t(1) << 16)                         // LBO (ignored): 16 B
         | (uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32) // SBO
         | (uint64_t(1) << 46)                         // version = 1 (SM100)
         | (uint64_t(2) << 61);                        // SWIZZLE_128B
}

// MN-major operand with 128-byte swizzle: the M (or N) index is contiguous,
// e.g. V [keys][D] used as the B operand ("N x K" = D x keys) of P . V.
// Canonical layout (CUTLASS mma_traits_sm100.hpp, make_umma_desc<Major::MN>,
// LayoutType::B128), in elements:
//   ((8, 8, m), (8, k)) : ((1, 8, LBO), (64, SBO))
// i.e. 64 MN elements (128 bytes) form one swizzled row; 8 rows along K form
// a 1024-byte atom; atoms follow each other along K at SBO = 1024 bytes; the
// next 64-wide MN panel starts LBO bytes further (only used when the MMA's
// M or N exceeds 64). To select K rows k0 .. k0 + 15, add k0 * 128 bytes to
// the start address. Set the matching "A/B major" bit of the instruction
// descriptor (make_idesc_f16(..., a_mn_major / b_mn_major = true)).
// As for the K-major descriptor, base offset (bits 49-51) is 0: the swizzle
// atoms must be 1024-byte aligned, and a start address advanced by whole
// 128-byte rows (k0 * 128) keeps the swizzle phase consistent.
__device__ __forceinline__ uint64_t make_smem_desc_mnmajor_sw128(
    const void *smem, uint32_t lbo_bytes, uint32_t sbo_bytes = 1024) {
  const uint64_t addr = shared_addr(smem);
  return ((addr >> 4) & 0x3FFF)                        // start address
         | (uint64_t((lbo_bytes >> 4) & 0x3FFF) << 16) // LBO: MN panels
         | (uint64_t((sbo_bytes >> 4) & 0x3FFF) << 32) // SBO: 8-row K groups
         | (uint64_t(1) << 46)                         // version = 1 (SM100)
         | (uint64_t(2) << 61);                        // SWIZZLE_128B
}

// ===========================================================================
// Instruction descriptor for tcgen05.mma .kind::f16 (32 bits).
// PTX: "Instruction descriptor" (tcgen05).
//
//   bits  0-1  sparsity selector (0)   bit 2  sparse (0)   bit 3  saturate (0)
//   bits  4-5  D format: 0 F16, 1 F32
//   bits  7-9  A format: 0 F16, 1 BF16      bits 10-12  B format (same)
//   bit  13/14 negate A / B (0)
//   bit  15    A major: 0 K-major, 1 MN-major   bit 16  B major
//   bits 17-22 N >> 3                         bits 24-28  M >> 4
//   bits 30-31 max shift (.ws only, 0)
// ===========================================================================
constexpr uint32_t make_idesc_f16(int m, int n, bool bf16, bool d_f32 = true,
                                  bool a_mn_major = false,
                                  bool b_mn_major = false) {
  return (uint32_t(d_f32 ? 1 : 0) << 4) | (uint32_t(bf16 ? 1 : 0) << 7) |
         (uint32_t(bf16 ? 1 : 0) << 10) | (uint32_t(a_mn_major) << 15) |
         (uint32_t(b_mn_major) << 16) | (uint32_t(n >> 3) << 17) |
         (uint32_t(m >> 4) << 24);
}

// ===========================================================================
// MMA. PTX: "tcgen05.mma".
// tcgen05.mma.cta_group::1.kind::f16 [d], a_desc, b_desc, idesc,
// enable_input_d;
//   D[M x N] (TMEM, FP32) = A[M x 16] * B[N x 16]^T + (accumulate ? D : 0)
// A and B are described by shared-memory descriptors; K = 16 per
// instruction for 16-bit inputs. Issued by ONE thread; it runs
// asynchronously -- completion is observed only through tcgen05_commit.
// ===========================================================================
__device__ __forceinline__ void mma_f16_ss(uint32_t d_tmem, uint64_t a_desc,
                                           uint64_t b_desc, uint32_t idesc,
                                           bool accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
          d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate ? 1 : 0)
      : "memory");
}

// tcgen05.mma.cta_group::1.kind::f16 [d], [a_tmem], b_desc, idesc,
// enable_input_d;   ("TS" form; added in 40_fa4_pipeline)
// Same as mma_f16_ss, but the A operand (M x 16, 16-bit) is read from Tensor
// Memory instead of shared memory. PTX: "tcgen05.mma", operand [a-desc] /
// [a-tmem]. Layout of A in TMEM (cta_group::1, M = 128): row i of A is TMEM
// lane i; its 16 K elements are packed two per 32-bit column, element 2c in
// the low half and 2c + 1 in the high half of column a_tmem.col + c, i.e. 8
// columns per instruction. a_tmem's lane field is 0. A must be K-major (the
// A-major bit of idesc is 0); B still comes from a shared-memory descriptor.
// Before the MMA reads A, the threads that wrote it with tcgen05.st must
// tcgen05.wait::st, tcgen05.fence::before_thread_sync and synchronize with
// the issuing thread (which then executes tcgen05.fence::after_thread_sync).
__device__ __forceinline__ void mma_f16_ts(uint32_t d_tmem, uint32_t a_tmem,
                                           uint64_t b_desc, uint32_t idesc,
                                           bool accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::"r"(
          d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate ? 1 : 0)
      : "memory");
}

// tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];
// PTX: "tcgen05.commit". Makes the mbarrier track all prior async tcgen05
// operations (MMAs) issued by this thread: one arrival happens when they have
// all completed -- including their shared-memory reads, so the operand
// buffers may then be overwritten, and the accumulator may be read.
__device__ __forceinline__ void tcgen05_commit(uint64_t *bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(shared_addr(bar))
               : "memory");
}

// ===========================================================================
// TMEM -> registers. PTX: "tcgen05.ld", "tcgen05.wait".
// tcgen05.ld.sync.aligned.32x32b.xN.b32: the warp reads 32 lanes x N
// columns starting at taddr (taddr's lane field = 32 * (warp % 4)); thread i
// receives lane (taddr.lane + i), columns taddr.col .. taddr.col + N - 1, in
// registers v[0..N-1]. Asynchronous: the registers are valid only after
// tmem_wait_ld().
//
// Caveat: the compiler does not know that v[] depends on tmem_wait_ld(). The
// wait asm has no register operands, so nothing ties the output registers to
// it; "volatile"/"memory" only keep the two asm statements (and memory
// accesses) in order. Register-only math on v[] (e.g. v[i] * scale, max)
// could in principle be scheduled between the ld and the wait, i.e. before
// the values arrive. Storing v[] straight to memory after the wait is safe in
// practice. For anything else prefer the *_sync variants below, which issue
// tcgen05.ld and tcgen05.wait::ld in one asm statement, so the outputs only
// exist after the wait.
// ===========================================================================
__device__ __forceinline__ void tmem_ld_32x32b_x8(uint32_t taddr,
                                                  float (&v)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld_32x32b_x16(uint32_t taddr,
                                                   float (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15}, [%16];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]),
                 "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
                 "=f"(v[14]), "=f"(v[15])
               : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr,
                                                   float (&v)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "
      "%30, %31}, [%32];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
        "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]),
        "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]),
        "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]),
        "=f"(v[21]), "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]),
        "=f"(v[26]), "=f"(v[27]), "=f"(v[28]), "=f"(v[29]), "=f"(v[30]),
        "=f"(v[31])
      : "r"(taddr));
}

// tcgen05.wait::ld.sync.aligned;  all prior tcgen05.ld of this thread done.
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// tcgen05.ld.sync.aligned.32x32b.xN.b32 followed by tcgen05.wait::ld in the
// same asm statement: same data as tmem_ld_32x32b_xN + tmem_wait_ld, but the
// output registers are defined only after the wait (no compiler reordering
// of their uses before it). The wait also covers every earlier tcgen05.ld of
// this thread, so there is no overlap between several loads.
__device__ __forceinline__ void tmem_ld_32x32b_x8_sync(uint32_t taddr,
                                                       float (&v)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n\t"
               "tcgen05.wait::ld.sync.aligned;"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "r"(taddr)
               : "memory");
}

__device__ __forceinline__ void tmem_ld_32x32b_x16_sync(uint32_t taddr,
                                                        float (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15}, [%16];\n\t"
               "tcgen05.wait::ld.sync.aligned;"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]),
                 "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
                 "=f"(v[14]), "=f"(v[15])
               : "r"(taddr)
               : "memory");
}

__device__ __forceinline__ void tmem_ld_32x32b_x32_sync(uint32_t taddr,
                                                        float (&v)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
               "%26, %27, %28, %29, %30, %31}, [%32];\n\t"
               "tcgen05.wait::ld.sync.aligned;"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]),
                 "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
                 "=f"(v[14]), "=f"(v[15]), "=f"(v[16]), "=f"(v[17]),
                 "=f"(v[18]), "=f"(v[19]), "=f"(v[20]), "=f"(v[21]),
                 "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]),
                 "=f"(v[26]), "=f"(v[27]), "=f"(v[28]), "=f"(v[29]),
                 "=f"(v[30]), "=f"(v[31])
               : "r"(taddr)
               : "memory");
}

// ===========================================================================
// Registers -> TMEM. PTX: "tcgen05.st", "tcgen05.wait".
// tcgen05.st.sync.aligned.32x32b.xN.b32 [taddr], {v[0..N-1]}: the mirror of
// tcgen05.ld 32x32b -- thread i writes lane (taddr.lane + i), columns
// taddr.col .. taddr.col + N - 1 (same lane restriction per warp).
// Asynchronous: before a tcgen05.mma reads those cells, or before another
// thread uses them, call tmem_wait_st() (then tcgen05 fence + sync as usual).
// Syntax as in CUTLASS cute/arch/copy_sm100.hpp (SM100_TMEM_STORE_32dp32b*x)
// and cutlass/arch/barrier.h (fence_view_async_tmem_store).
// ===========================================================================
__device__ __forceinline__ void tmem_st_32x32b_x8(uint32_t taddr,
                                                  const float (&v)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 "
               "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
               :
               : "r"(taddr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                 "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7])
               : "memory");
}

__device__ __forceinline__ void tmem_st_32x32b_x16(uint32_t taddr,
                                                   const float (&v)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 "
               "[%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16};"
               :
               : "r"(taddr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                 "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]),
                 "f"(v[9]), "f"(v[10]), "f"(v[11]), "f"(v[12]), "f"(v[13]),
                 "f"(v[14]), "f"(v[15])
               : "memory");
}

__device__ __forceinline__ void tmem_st_32x32b_x32(uint32_t taddr,
                                                   const float (&v)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 "
               "[%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
               "%26, %27, %28, %29, %30, %31, %32};"
               :
               : "r"(taddr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                 "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]),
                 "f"(v[9]), "f"(v[10]), "f"(v[11]), "f"(v[12]), "f"(v[13]),
                 "f"(v[14]), "f"(v[15]), "f"(v[16]), "f"(v[17]), "f"(v[18]),
                 "f"(v[19]), "f"(v[20]), "f"(v[21]), "f"(v[22]), "f"(v[23]),
                 "f"(v[24]), "f"(v[25]), "f"(v[26]), "f"(v[27]), "f"(v[28]),
                 "f"(v[29]), "f"(v[30]), "f"(v[31])
               : "memory");
}

// The same stores with 32-bit integer sources, for packed data such as two
// 16-bit values per column (added in 40_fa4_pipeline: P written over S).
__device__ __forceinline__ void tmem_st_32x32b_x16(uint32_t taddr,
                                                   const uint32_t (&v)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 "
               "[%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16};"
               :
               : "r"(taddr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
                 "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
                 "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]),
                 "r"(v[14]), "r"(v[15])
               : "memory");
}

__device__ __forceinline__ void tmem_st_32x32b_x32(uint32_t taddr,
                                                   const uint32_t (&v)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 "
               "[%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
               "%26, %27, %28, %29, %30, %31, %32};"
               :
               : "r"(taddr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
                 "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]),
                 "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]),
                 "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]),
                 "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]),
                 "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]),
                 "r"(v[29]), "r"(v[30]), "r"(v[31])
               : "memory");
}

// tcgen05.wait::st.sync.aligned;  all prior tcgen05.st of this thread done.
__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ===========================================================================
// Ordering of tcgen05 operations across thread synchronization.
// PTX: "tcgen05.fence", "Memory Consistency Model for tcgen05 operations".
// tcgen05 instructions are asynchronous; a CTA barrier or mbarrier wait does
// not by itself order them. Pattern:
//   thread A: tcgen05 op ...; tcgen05_fence_before_thread_sync();
//   sync (__syncthreads / mbarrier arrive -> wait)
//   thread B: tcgen05_fence_after_thread_sync(); tcgen05 op ...
// ===========================================================================
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ===========================================================================
// Register reallocation between warpgroups (added in 40_fa4_pipeline).
// PTX: "Miscellaneous Instructions: setmaxnreg".
// setmaxnreg.inc/dec.sync.aligned.u32 N;  N in 24 .. 256, a multiple of 8.
// Every warp of a warpgroup must execute the same setmaxnreg (it is
// .sync.aligned per warp). dec releases registers above N to a per-CTA
// pool; inc blocks until N - current registers per thread are available in
// the pool. The kernel must be compiled for the launch register count
// (e.g. __launch_bounds__(512, 1) -> 128); ptxas then allocates up to N
// registers in code that follows an inc. Not tcgen05-specific, kept here
// with the other SM90+/SM100 warp-specialization helpers.
// ===========================================================================
template <uint32_t N> __device__ __forceinline__ void setmaxnreg_inc() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0, "setmaxnreg range");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N) : "memory");
}
template <uint32_t N> __device__ __forceinline__ void setmaxnreg_dec() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0, "setmaxnreg range");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N) : "memory");
}

} // namespace lab
