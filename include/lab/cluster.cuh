// Thread block clusters and the 2-CTA ("pair") forms of TMA and tcgen05 on
// SM100. Written by learners in 33_2cta_gemm; provided from 34_fa3 on.
//
// Every function wraps one PTX instruction (or a fixed retry loop around
// one, or one runtime call) and contains no algorithm logic. PTX section
// names refer to the PTX ISA (version 8.7),
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// Forms were checked against CUTLASS (third_party/cutlass/include/cute/arch/
// copy_sm100_tma.hpp, mma_sm100_umma.hpp, tmem_allocator_sm100.hpp;
// include/cutlass/arch/barrier.h).
//
// Model ("Thread Block Clusters", "Distributed Shared Memory"):
//   - a cluster is a group of CTAs that are co-scheduled on one GPC; each
//     CTA has a rank 0 .. size-1 (%cluster_ctarank);
//   - every CTA can address the shared memory of every CTA of its cluster
//     through the .shared::cluster window. mapa turns "my address of x"
//     into "the address of x in CTA r" (same offset, other CTA);
//   - a CTA's own .shared::cta address is also a valid .shared::cluster
//     address (it denotes the CTA's own shared memory);
//   - barrier.cluster synchronizes all threads of all CTAs of the cluster.
//
// Pair ("cta_group::2") model: CTAs 2p and 2p + 1 of a cluster form pair p;
// the even one is the leader. One thread of the leader issues
// tcgen05.mma.cta_group::2, which reads A and B from the SAME shared-memory
// offsets in both CTAs and writes D into BOTH CTAs' TMEM (rows 0..127 in the
// leader, 128..255 in the peer). All tcgen05 instructions of a kernel must
// use the same .cta_group, so a pair kernel also allocates, frees and
// commits with cta_group::2.
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lab/check.h"
#include "lab/mbarrier.cuh"

namespace lab {

// ===========================================================================
// Cluster identity. PTX: "Special Registers": %cluster_ctarank,
// %cluster_nctarank, %clusterid, %nclusterid.
// ===========================================================================

// Rank of this CTA inside its cluster (0 .. cluster size - 1).
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

// Index of this cluster in the grid along x, and the number of clusters
// along x (= gridDim.x / cluster size for a 1-D grid of 1-D clusters).
__device__ __forceinline__ uint32_t cluster_id_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t num_clusters_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(r));
  return r;
}

// ===========================================================================
// Cluster barrier. PTX: "barrier.cluster".
// barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;
// Every thread of every CTA of the cluster must arrive (.aligned: all
// threads of a warp execute it together). wait returns once all of them
// have arrived; release/acquire make each thread's earlier shared-memory
// writes (e.g. mbarrier.init + fence_barrier_init) visible cluster-wide.
// Splitting arrive and wait lets a thread do independent work in between.
// ===========================================================================
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// ===========================================================================
// Distributed shared memory. PTX: "mapa".
// mapa.shared::cluster.u32 dst, src, rank;
// `p` points into this CTA's shared memory; the result is the
// .shared::cluster address of the same offset in CTA `rank` of the cluster
// (rank may be our own). Use it as the address operand of *_cluster
// functions below; it is NOT a generic pointer.
// ===========================================================================
__device__ __forceinline__ uint32_t map_shared_rank(const void *p,
                                                    uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r)
               : "r"(shared_addr(p)), "r"(rank));
  return r;
}

// ===========================================================================
// mbarrier across the cluster. PTX: "mbarrier.arrive", "mbarrier.try_wait".
// ===========================================================================

// mbarrier.arrive.release.cluster.shared::cluster.b64 _, [addr];
// One arrival on the barrier at .shared::cluster address `addr` (possibly in
// another CTA, from map_shared_rank). Release at cluster scope: this
// thread's earlier memory accesses happen before the phase completes, for a
// waiter in any CTA of the cluster.
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr) {
  asm volatile(
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(addr)
      : "memory");
}

// mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 in a loop.
// Waits on a barrier in OUR shared memory whose arrivals may come from
// other CTAs of the cluster: acquire at cluster scope pairs with their
// release.cluster arrivals. (A wait can only target the own CTA's barrier.)
__device__ __forceinline__ void mbar_wait_cluster(uint64_t *bar,
                                                  uint32_t parity) {
  asm volatile("{\n\t.reg .pred p;\n"
               "LAB_MBAR_WAIT_CL_%=:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, "
               "[%0], %1;\n\t"
               "@!p bra LAB_MBAR_WAIT_CL_%=;\n\t}" ::"r"(shared_addr(bar)),
               "r"(parity)
               : "memory");
}

// ===========================================================================
// TMA loads for a CTA pair. PTX: "cp.async.bulk.tensor" (.cta_group,
// .multicast::cluster).
// ===========================================================================

// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
//     .mbarrier::complete_tx::bytes [dst], [map, {x, y}], [bar];
// Like tma_load_2d (box -> this CTA's shared `dst`), but the completion
// (complete_tx of the box bytes) may go to an mbarrier in the PEER CTA of
// the pair: `bar` is a .shared::cluster address (map_shared_rank(bar,
// leader)). This lets both CTAs' loads complete on the leader's barrier,
// which is the one the MMA issuer waits on.
__device__ __forceinline__ void
tma_load_2d_cg2(void *dst, const CUtensorMap *map, uint32_t bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
                   shared_addr(dst)),
               "l"(map), "r"(x), "r"(y), "r"(bar)
               : "memory");
}

// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
//     .mbarrier::complete_tx::bytes.multicast::cluster
//     [dst], [map, {x, y}], [bar], cta_mask;
// One read of the box from global/L2, written to the same offset `dst` in
// every CTA whose bit is set in `cta_mask` (bit r = cluster rank r). Each
// destination CTA's copy signals complete_tx on the barrier at offset `bar`
// in that destination CTA or in its pair peer: `bar` is the
// .shared::cluster address of the barrier in OUR pair's leader
// (map_shared_rank(bar, leader)); for a destination in another pair the
// hardware uses the same offset in THAT pair's leader (the peer bit of the
// address selects leader vs. follower, as CUTLASS's Sm100MmaPeerBitMask).
__device__ __forceinline__ void
tma_load_2d_cg2_multicast(void *dst, const CUtensorMap *map, uint32_t bar,
                          uint16_t cta_mask, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
               "mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
               "{%2, %3}], [%4], %5;" ::"r"(shared_addr(dst)),
               "l"(map), "r"(x), "r"(y), "r"(bar), "h"(cta_mask)
               : "memory");
}

// ===========================================================================
// TMEM allocation for a pair. PTX: "Tensor Memory Allocation and Management
// Instructions" with .cta_group::2.
// One warp in EACH CTA of the pair (same warp index in both) executes the
// instruction; the two warps cooperate and both receive the same base
// column in their `dst`. Same rules as the cta_group::1 forms otherwise.
// ===========================================================================
__device__ __forceinline__ void tmem_alloc_cg2(uint32_t *dst, uint32_t ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          shared_addr(dst)),
      "r"(ncols)
      : "memory");
}
__device__ __forceinline__ void tmem_relinquish_alloc_permit_cg2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                   : "memory");
}
// Before it: every access to this CTA's columns by ANY CTA of the pair
// (the leader's MMAs write the peer's TMEM) has completed and is ordered
// before it (tcgen05 fences around a cluster barrier).
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t taddr,
                                                 uint32_t ncols) {
  asm volatile(
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(ncols)
      : "memory");
}

// ===========================================================================
// Pair MMA. PTX: "tcgen05.mma" with .cta_group::2.
// tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, p;
// Issued by ONE thread of the LEADER CTA. With idesc M = 256, N = n:
//   A: rows 0..127 from the leader's smem at a_desc, rows 128..255 from the
//      peer's smem at the same offset;
//   B: rows 0..n/2-1 from the leader at b_desc, rows n/2..n-1 from the peer
//      at the same offset;
//   D: rows 0..127 -> leader TMEM lanes 0..127, rows 128..255 -> peer TMEM
//      lanes 0..127, columns d .. d + n - 1 in both.
// ===========================================================================
__device__ __forceinline__ void mma_f16_ss_cg2(uint32_t d_tmem, uint64_t a_desc,
                                               uint64_t b_desc, uint32_t idesc,
                                               bool accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
          d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate ? 1 : 0)
      : "memory");
}

// tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster
//     .multicast::cluster.b64 [bar], cta_mask;
// Like tcgen05_commit: once all earlier MMAs of this thread have completed
// (shared-memory reads of BOTH CTAs and TMEM writes of both), arrive once on
// the barrier at offset `bar` in every CTA whose bit is set in cta_mask
// (e.g. 0b11 for pair 0: leader and peer).
__device__ __forceinline__ void
tcgen05_commit_cg2_multicast(uint64_t *bar, uint16_t cta_mask) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::"
      "cluster.multicast::cluster.b64 [%0], %1;" ::"r"(shared_addr(bar)),
      "h"(cta_mask)
      : "memory");
}

// ===========================================================================
// Host: launching a kernel with a cluster shape. CUDA Runtime API
// "cudaLaunchKernelEx", launch attribute cudaLaunchAttributeClusterDimension.
// gridDim.x must be a multiple of cluster_x. The <<<>>> syntax cannot set a
// runtime cluster shape (only __cluster_dims__ at compile time).
// ===========================================================================
template <class... KArgs, class... Args>
void launch_cluster(void (*kernel)(KArgs...), dim3 grid, dim3 block,
                    int smem_bytes, cudaStream_t stream, int cluster_x,
                    Args &&...args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_x;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  LAB_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

// How many clusters of `cluster_x` CTAs (each `block` threads, `smem_bytes`
// dynamic shared memory) can be resident at once: the grid size of a
// persistent cluster kernel. CUDA Runtime API
// "cudaOccupancyMaxActiveClusters". Not always SMs / cluster_x: a cluster
// must fit inside one GPC.
template <class... KArgs>
int max_active_clusters(void (*kernel)(KArgs...), dim3 block, int smem_bytes,
                        int cluster_x) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster_x, 1, 1);
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_x;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  LAB_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &n, reinterpret_cast<const void *>(kernel), &cfg));
  return n;
}

} // namespace lab
