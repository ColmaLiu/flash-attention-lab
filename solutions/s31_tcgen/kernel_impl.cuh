// Host integration of the pinned NVIDIA CUTLASS SM100 implementation.
// Full device source and BSD-3-Clause license: references/cutlass-sm100/.
#include "cuda_api.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
using namespace cute;
namespace {
using Element = cutlass::bfloat16_t;
using Layout = cutlass::layout::RowMajor;
using Tile = Shape<_128,_128,_64>;
using Cluster = Shape<_1,_1,_1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
  cutlass::epilogue::collective::EpilogueTileAuto, float, float,
  float, Layout, 4, float, Layout, 4,
  cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
#if LAB_MULTISTAGE
using Stages = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>;
#else
using Stages = cutlass::gemm::collective::StageCount<2>;
#endif
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm100,cutlass::arch::OpClassTensorOp,
  Element,Layout,8,Element,Layout,8,float,Tile,Cluster,Stages,
  cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
void checked(cutlass::Status s) {
  if(s!=cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}
struct Prepared {
  Gemm gemm;
  void* workspace=nullptr;
  size_t workspace_bytes=0;
  cudaStream_t stream;
  Prepared(uintptr_t a,uintptr_t b,uintptr_t c,int m,int n,int k,uintptr_t raw_stream):
    stream(reinterpret_cast<cudaStream_t>(raw_stream)) {
    auto sa=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{},make_shape(m,k,1));
    auto sb=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{},make_shape(n,k,1));
    auto sc=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{},make_shape(m,n,1));
    auto sd=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{},make_shape(m,n,1));
    cutlass::KernelHardwareInfo hw{};
    cudaGetDevice(&hw.device_id);
    hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
      {reinterpret_cast<Element const*>(a),sa,reinterpret_cast<Element const*>(b),sb},
      {{1.0f,0.0f},reinterpret_cast<float const*>(c),sc,reinterpret_cast<float*>(c),sd},hw};
    checked(gemm.can_implement(args));
    auto bytes=Gemm::get_workspace_size(args);
    workspace_bytes=bytes;
    if(bytes && cudaMalloc(&workspace,bytes)!=cudaSuccess) throw std::runtime_error("CUTLASS workspace allocation failed");
    try { checked(gemm.initialize(args,workspace,stream)); }
    catch(...) {if(workspace)cudaFree(workspace);throw;}
  }
  ~Prepared(){if(workspace)cudaFree(workspace);}
};
}
void* prepare_gemm(uintptr_t a,uintptr_t b,uintptr_t c,int m,int n,int k,uintptr_t stream){return new Prepared(a,b,c,m,n,k,stream);}
void run_gemm(void* p){auto* s=static_cast<Prepared*>(p);checked(s->gemm.run(s->stream));}
void delete_gemm(void* p){delete static_cast<Prepared*>(p);}

size_t gemm_workspace_bytes(void* p){return static_cast<Prepared*>(p)->workspace_bytes;}
