// Problem descriptions passed from the Python harness to every kernel.
//
// A Problem holds raw device pointers plus sizes. All tensors are contiguous,
// row-major (last dimension fastest) and live on the same GPU. Inputs are
// read-only; outputs start filled with garbage (NaN) and every element must be
// written by the kernel.
//
// `step` is the step being checked (see the lab's README); it lets one
// kernel.cu keep several versions side by side, e.g. `switch (p.step)`.
// `probe` is an optional FP32 debug buffer some steps ask you to fill.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lab {

enum class DType : int { F32 = 0, F16 = 1, BF16 = 2, E4M3 = 3 };

inline const char *dtype_name(DType t) {
  switch (t) {
  case DType::F32:
    return "fp32";
  case DType::F16:
    return "fp16";
  case DType::BF16:
    return "bf16";
  case DType::E4M3:
    return "e4m3";
  }
  return "?";
}

// y[cols, rows] = x[rows, cols]^T
struct TransposeProblem {
  const void *x;
  void *y;
  int rows, cols;
  DType dtype;
  int step;
  float *probe;
  int64_t probe_numel;
};

// y[r, :] = softmax(x[r, :]) for every row r
struct SoftmaxProblem {
  const float *x;
  float *y;
  int rows, cols;
  int step;
  float *probe;
  int64_t probe_numel;
};

// c[M, N] = a_scale * b_scale * (a[M, K] @ b[N, K]^T)
// Both A and B are K-major (the same shape as Q @ K^T).
struct GemmProblem {
  const void *a;
  const void *b;
  void *c;
  int m, n, k;
  DType ab_dtype;         // dtype of A and B
  DType c_dtype;          // dtype of C
  float a_scale, b_scale; // 1.0 unless FP8
  int step;
  float *probe;
  int64_t probe_numel;
};

// O = softmax(scale * Q K^T [+ causal mask]) V, per (batch, head).
// q, k, v, o: [B, N, H, D]; lse: [B, H, N] FP32, natural log:
//   lse[b, h, i] = log(sum_j exp(scale * q_i . k_j))   (over unmasked j)
// Causal: query i attends to keys j <= i.
// FP8 (E4M3) inputs: the real values are q_descale * q etc.; O is BF16.
struct AttentionProblem {
  const void *q, *k, *v;
  void *o;
  float *lse;
  int b, n, h, d;
  DType dtype;   // dtype of Q, K, V
  DType o_dtype; // dtype of O
  float scale;   // > 0 and finite
  bool causal;
  float q_descale, k_descale, v_descale; // 1.0 unless FP8
  int step;
  float *probe;
  int64_t probe_numel;
};

} // namespace lab
