#pragma once
// FA2-style teaching implementation for contiguous, BNHD forward with tail and causal predicates.
// Layout/fragment transformations and the GEMM staging pattern are adapted from
// FlashAttention v2.7.4.post1, copyright (c) 2024 Tri Dao, BSD-3-Clause.
// See README.md, LICENSE.flash-attention and AUTHORS.flash-attention for sources.
// Modified locally: BNHD addressing, zero-fill tail copies, causal masks, valid-row stores.
#include "cuda_api.h"
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cmath>

namespace s22_optimized {
using namespace cute;

template<class T, int D, int BN>
struct Traits {
    static constexpr int BM = 128;
    static constexpr int Threads = 128;
    using Atom = std::conditional_t<std::is_same_v<T, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
    using Mma = TiledMMA<Atom, Layout<Shape<_4, _1, _1>>, Tile<_64, _16, _16>>;
    using SmemAtom = decltype(composition(Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using QLayout = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BM>, Int<D>>{}));
    using KVLayout = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BN>, Int<D>>{}));
    using VtLayout = decltype(composition(KVLayout{},
        make_layout(Shape<Int<D>, Int<BN>>{}, GenRowMajor{})));
    using VtPlain = decltype(get_nonswizzle_portion(VtLayout{}));
    using Load = Copy_Atom<SM75_U32x4_LDSM_N, T>;
    using LoadTrans = Copy_Atom<SM75_U16x8_LDSM_T, T>;
    // Eight lanes per 64-element row segment, eight 16-bit elements per lane.
    using GmemCopy = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, T>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
    using GmemStore = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
    static constexpr int SmemBytes = (cosize(QLayout{}) + 2 * cosize(KVLayout{})) * sizeof(T);
};

// These views change the indexing of values already owned by a thread; they
// neither move data between lanes nor write it to shared memory.
template<class L>
CUTE_DEVICE auto rowcol(L layout) {
    static_assert(decltype(size<0>(layout))::value == 4);
    auto divided = logical_divide(layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                       make_layout(get<0, 0>(divided), get<2>(divided)));
}

template<class L>
CUTE_DEVICE auto as_operand_a(L layout) {
    auto divided = logical_divide(layout, Shape<Underscore, Underscore, _2>{});
    return make_layout(make_layout(get<0>(divided), get<2, 0>(divided)),
                       get<1>(divided), get<2, 1>(divided));
}

template<class T, class Tensor>
CUTE_DEVICE auto convert_fragment(Tensor const& src) {
    constexpr int count = decltype(size(src))::value;
    auto dst = make_tensor<T>(src.layout());
    cutlass::NumericArrayConverter<T, float, count> converter;
    *reinterpret_cast<cutlass::Array<T, count>*>(dst.data()) =
        converter(*reinterpret_cast<const cutlass::Array<float, count>*>(src.data()));
    return dst;
}

CUTE_DEVICE float quad_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
    return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
}
CUTE_DEVICE float quad_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x + __shfl_xor_sync(0xffffffffu, x, 1);
}

// Q/K are in SMEM. Preload the next register fragment before the current MMA.
template<class Acc, class RA, class RB, class SA, class SB, class Mma,
         class CopyA, class CopyB, class ThreadA, class ThreadB>
CUTE_DEVICE void qk_gemm(Acc& acc, RA& ra, RB& rb, SA const& sa, SB const& sb,
                        Mma mma, CopyA ca, CopyB cb, ThreadA ta, ThreadB tb) {
    auto av = ta.retile_D(ra);
    auto bv = tb.retile_D(rb);
    copy(ca, sa(_, _, _0{}), av(_, _, _0{}));
    copy(cb, sb(_, _, _0{}), bv(_, _, _0{}));
    CUTE_UNROLL
    for (int kk = 0; kk < size<2>(ra); ++kk) {
        if (kk + 1 < size<2>(ra)) {
            copy(ca, sa(_, _, kk + 1), av(_, _, kk + 1));
            copy(cb, sb(_, _, kk + 1), bv(_, _, kk + 1));
        }
        gemm(mma, ra(_, _, kk), rb(_, _, kk), acc);
    }
}

// P is already an A register fragment; only V is loaded from SMEM.
template<class Acc, class RA, class RB, class SB, class Mma, class CopyB, class ThreadB>
CUTE_DEVICE void pv_gemm(Acc& acc, RA const& ra, RB& rb, SB const& sb,
                        Mma mma, CopyB cb, ThreadB tb) {
    auto bv = tb.retile_D(rb);
    copy(cb, sb(_, _, _0{}), bv(_, _, _0{}));
    CUTE_UNROLL
    for (int kk = 0; kk < size<2>(ra); ++kk) {
        if (kk + 1 < size<2>(ra)) copy(cb, sb(_, _, kk + 1), bv(_, _, kk + 1));
        gemm(mma, ra(_, _, kk), rb(_, _, kk), acc);
    }
}

// A copy vector stays within one D row; D is always a multiple of eight.
// Invalid rows are zero-filled by cp.async; ordinary stores skip invalid rows.
template<class Copy, class Src, class Dst, class Coordinates>
CUTE_DEVICE void copy_valid_rows(Copy copy_atom, Src const& src, Dst& dst,
                                Coordinates const& coord, int start, int n) {
    auto predicate = make_tensor<bool>(make_shape(size<1>(src), size<2>(src)));
    CUTE_UNROLL
    for (int i = 0; i < size<1>(src); ++i)
        CUTE_UNROLL
        for (int j = 0; j < size<2>(src); ++j)
            predicate(i, j) = start + get<0>(coord(0, i, j)) < n;
    copy_if(copy_atom, predicate, src, dst);
}

template<class T, int D, int BN, bool Masked>
__global__ __launch_bounds__(128) void attention(
    const T* q, const T* k, const T* v, T* o, float* lse, int n, int heads, float scale, bool causal) {
    using Tr = Traits<T, D, BN>;
    constexpr int BM = Tr::BM;
    const int tid = threadIdx.x;
    const int64_t head_row = static_cast<int64_t>(blockIdx.y) * n;
    const int64_t qrow = head_row + blockIdx.x * BM;
    const int batch = blockIdx.y / heads, head = blockIdx.y % heads;
    const int64_t base = (static_cast<int64_t>(batch) * n * heads + head) * D;
    const int row_stride = heads * D;
    extern __shared__ __align__(128) unsigned char storage[];
    auto sq = make_tensor(make_smem_ptr(reinterpret_cast<T*>(storage)), typename Tr::QLayout{});
    auto sk = make_tensor(sq.data() + cosize(typename Tr::QLayout{}), typename Tr::KVLayout{});
    auto sv = make_tensor(sk.data() + cosize(typename Tr::KVLayout{}), typename Tr::KVLayout{});
    auto svt = make_tensor(sv.data(), typename Tr::VtLayout{});
    auto svt_plain = make_tensor(sv.data().get(), typename Tr::VtPlain{});
    auto gq = make_tensor(make_gmem_ptr(q + base + static_cast<int64_t>(blockIdx.x) * BM * row_stride),
                         make_layout(Shape<Int<BM>, Int<D>>{}, make_stride(row_stride, _1{})));
    typename Tr::GmemCopy gcopy;
    auto gt = gcopy.get_thread_slice(tid);
    auto tq = gt.partition_D(sq);
    auto tk = gt.partition_D(sk);
    auto tv = gt.partition_D(sv);
    auto qcoord = gt.partition_S(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
    auto kvcoord = gt.partition_S(make_identity_tensor(Shape<Int<BN>, Int<D>>{}));
    auto load_k = [&](int block) {
        auto gk = make_tensor(make_gmem_ptr(k + base + static_cast<int64_t>(block) * BN * row_stride),
                             make_layout(Shape<Int<BN>, Int<D>>{}, make_stride(row_stride, _1{})));
        if constexpr (Masked) copy_valid_rows(gcopy, gt.partition_S(gk), tk, kvcoord, block * BN, n);
        else copy(gcopy, gt.partition_S(gk), tk);
    };
    auto load_v = [&](int block) {
        auto gv = make_tensor(make_gmem_ptr(v + base + static_cast<int64_t>(block) * BN * row_stride),
                             make_layout(Shape<Int<BN>, Int<D>>{}, make_stride(row_stride, _1{})));
        if constexpr (Masked) copy_valid_rows(gcopy, gt.partition_S(gv), tv, kvcoord, block * BN, n);
        else copy(gcopy, gt.partition_S(gv), tv);
    };

    typename Tr::Mma mma;
    auto mt = mma.get_thread_slice(tid);
    auto rq = mt.partition_fragment_A(sq);
    auto rk = mt.partition_fragment_B(sk);
    auto rv = mt.partition_fragment_B(svt_plain);
    auto out = partition_fragment_C(mma, Shape<Int<BM>, Int<D>>{});
    clear(out);
    auto out_rows = make_tensor(out.data(), rowcol(out.layout()));
    auto maxs = make_tensor<float>(make_shape(size<0>(out_rows)));
    auto sums = make_tensor<float>(make_shape(size<0>(out_rows)));
    fill(maxs, -INFINITY);
    clear(sums);
    auto cq = make_tiled_copy_A(typename Tr::Load{}, mma);
    auto ck = make_tiled_copy_B(typename Tr::Load{}, mma);
    auto cv = make_tiled_copy_B(typename Tr::LoadTrans{}, mma);
    auto qt = cq.get_thread_slice(tid);
    auto kt = ck.get_thread_slice(tid);
    auto vt = cv.get_thread_slice(tid);
    auto qs = qt.partition_S(sq);
    auto ks = kt.partition_S(sk);
    auto vs = vt.partition_S(svt);

    if constexpr (Masked) copy_valid_rows(gcopy, gt.partition_S(gq), tq, qcoord, blockIdx.x * BM, n);
    else copy(gcopy, gt.partition_S(gq), tq);
    const int end = Masked && causal ? min(n, static_cast<int>((blockIdx.x + 1) * BM)) : n;
    const int last_block = (end + BN - 1) / BN - 1;
    load_k(last_block);
    cp_async_fence();
    const float scale_log2 = scale * 1.4426950408889634f;
    // Reverse traversal; mask invalid keys before the row maximum.
    for (int block = last_block; block >= 0; --block) {
        auto scores = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
        clear(scores);
        cp_async_wait<0>();
        __syncthreads();
        // V transfer overlaps QK; Q is retained in shared memory.
        load_v(block);
        cp_async_fence();
        qk_gemm(scores, rq, rk, qs, ks, mma, cq, ck, qt, kt);
        cp_async_wait<0>();
        __syncthreads();
        // All K readers are done. The next K transfer overlaps softmax/PV.
        if (block > 0) {
            load_k(block - 1);
            cp_async_fence();
        }
        if constexpr (Masked) {
            auto coordinates = mt.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
            CUTE_UNROLL
            for (int i = 0; i < size(scores); ++i) {
                const int qi = blockIdx.x * BM + get<0>(coordinates(i));
                const int ki = block * BN + get<1>(coordinates(i));
                if (qi >= n || ki >= n || (causal && ki > qi)) scores(i) = -INFINITY;
            }
        }
        auto rows = make_tensor(scores.data(), rowcol(scores.layout()));
        CUTE_UNROLL
        for (int r = 0; r < size<0>(rows); ++r) {
            float maximum = maxs(r);
            CUTE_UNROLL
            for (int c = 0; c < size<1>(rows); ++c) maximum = fmaxf(maximum, rows(r, c));
            maximum = quad_max(maximum);
            const float alpha = Masked && maxs(r) == -INFINITY ? 0.0f : exp2f((maxs(r) - maximum) * scale_log2);
            maxs(r) = maximum;
            sums(r) *= alpha;
            CUTE_UNROLL
            for (int c = 0; c < size<1>(out_rows); ++c) out_rows(r, c) *= alpha;
            CUTE_UNROLL
            for (int c = 0; c < size<1>(rows); ++c) {
                rows(r, c) = Masked && maximum == -INFINITY ? 0.0f : exp2f((rows(r, c) - maximum) * scale_log2);
                sums(r) += rows(r, c);
            }
        }
        // Scores -> low-precision P -> MMA A, without any SMEM round trip.
        auto p = convert_fragment<T>(scores);
        auto pa = make_tensor(p.data(), as_operand_a(p.layout()));
        pv_gemm(out, pa, rv, vs, mma, cv, vt);
    }

    // Complete the row sum across the quad only when normalization needs it.
    CUTE_UNROLL
    for (int r = 0; r < size<0>(out_rows); ++r) {
        const float sum = quad_sum(sums(r));
        const float inv = !Masked || sum > 0.0f ? 1.0f / sum : 0.0f;
        CUTE_UNROLL
        for (int c = 0; c < size<1>(out_rows); ++c) out_rows(r, c) *= inv;
        // rowcol's row mode is (2,MMA_M); warp spacing along M is 64.
        const int row = (tid / 32) * 16 + (tid % 32) / 4 + (r % 2) * 8 + (r / 2) * 64;
        if (tid % 4 == 0 && (!Masked || blockIdx.x * BM + row < n)) lse[qrow + row] = maxs(r) * scale + logf(sum);
    }
    auto ro = convert_fragment<T>(out);
    // Final output only: reuse Q's storage for coalesced vector stores.
    auto co = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{}, mma);
    auto ot = co.get_thread_slice(tid);
    __syncthreads();
    copy(co, ot.retile_S(ro), ot.partition_D(sq));
    __syncthreads();
    // Each vector is one aligned eight-element segment of a logical row.
    // Index the swizzled SMEM view explicitly; the external BNHD row stride
    // differs from the packed shared row stride.
    for (int segment = tid; segment < BM * D / 8; segment += Tr::Threads) {
        const int row = segment / (D / 8), col = (segment % (D / 8)) * 8;
        if (!Masked || blockIdx.x * BM + row < n) {
            const uint4 packed = *reinterpret_cast<const uint4*>(&sq(row, col));
            *reinterpret_cast<uint4*>(o + base + (static_cast<int64_t>(blockIdx.x) * BM + row) * row_stride + col) = packed;
        }
    }
}

template<class T, int D, int BN, bool Masked>
void launch(const TensorView& q, const TensorView& k, const TensorView& v,
            const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    constexpr int bytes = Traits<T, D, BN>::SmemBytes;
    const auto error = cudaFuncSetAttribute(attention<T, D, BN, Masked>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    const dim3 grid(static_cast<unsigned>((q.shape[1] + 127) / 128),
                    static_cast<unsigned>(q.shape[0] * q.shape[2]));
    attention<T, D, BN, Masked><<<grid, 128, bytes, stream>>>(
        static_cast<const T*>(q.data), static_cast<const T*>(k.data),
        static_cast<const T*>(v.data), static_cast<T*>(o.data),
        static_cast<float*>(lse.data), q.shape[1], q.shape[2], scale, causal);
}

template<class T, int D, int BN>
void launch_shape(const TensorView& q, const TensorView& k, const TensorView& v,
                  const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    if (!causal && q.shape[1] % 128 == 0)
        launch<T, D, BN, false>(q, k, v, o, lse, scale, causal, stream);
    else
        launch<T, D, BN, true>(q, k, v, o, lse, scale, causal, stream);
}

template<class T>
void dispatch(const TensorView& q, const TensorView& k, const TensorView& v,
              const TensorView& o, const TensorView& lse, float scale, bool causal, cudaStream_t stream) {
    if (q.shape[3] == 64) {
        launch_shape<T, 64, 128>(q, k, v, o, lse, scale, causal, stream);
    } else {
        int device = 0;
        cudaDeviceProp props{};
        auto status = cudaGetDevice(&device);
        if (status == cudaSuccess) status = cudaGetDeviceProperties(&props, device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        // Reuse the smaller FA2 tile on SM100 to reduce register pressure.
        if ((props.major == 8 && props.minor > 0) ||
            (props.major == 10 && props.minor == 0))
            launch_shape<T, 128, 32>(q, k, v, o, lse, scale, causal, stream);
        else
            launch_shape<T, 128, 64>(q, k, v, o, lse, scale, causal, stream);
    }
}
} // namespace s22_optimized
