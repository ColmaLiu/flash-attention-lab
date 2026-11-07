#pragma once
// Copyright (c) 2025, Tri Dao.
// Native packed exp2 for SM100. Adapted from Dao-AILab FlashAttention:
// flash_attn/cute/utils.py, e2e_asm2 / ex2_emulation_2 and POLY_EX2[4].
// Pinned commit: 1fc0221183e799363001394a12dd9e78e15f5dda.
// https://github.com/Dao-AILab/flash-attention/blob/1fc0221183e799363001394a12dd9e78e15f5dda/flash_attn/cute/utils.py
// The final packed FTZ operation is local and makes underflow/masked zero
// explicit. See README.md for the mixed SFU schedule and numerical limits.
/*
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <cuda_runtime.h>

namespace sm100_attention_exp {
// Domain for this caller: x,y <= 4 (<=0 without delayed rescaling), with
// -infinity permitted for masks. A clamp to -127 plus exponent reconstruction
// yields exact +0 for -inf; final FTZ makes all x<-126 underflows explicit +0.
// For normal outputs this uses the upstream degree-4 minimax coefficients,
// four packed FMAs and biased-integer range reduction with round-down.
__device__ __forceinline__ void exp2_packed_quartic(float *xy) {
  float a, b;
  asm("{\n\t"
      ".reg .f32 x, y, bias, c0, c1, c2, c3, c4;\n\t"
      ".reg .b64 clamped, bias2, rounded, integer2, fraction2, p, coeff, "
      "joined, zero2;\n\t"
      ".reg .b32 ix, iy, px, py, ex, ey;\n\t"
      "max.ftz.f32 x, %2, 0fC2FE0000;\n\t"
      "max.ftz.f32 y, %3, 0fC2FE0000;\n\t"
      "mov.b64 clamped, {x, y};\n\t"
      "mov.f32 bias, 0f4B400000;\n\t"
      "mov.b64 bias2, {bias, bias};\n\t"
      "add.rm.ftz.f32x2 rounded, clamped, bias2;\n\t"
      "sub.rn.ftz.f32x2 integer2, rounded, bias2;\n\t"
      "sub.rn.ftz.f32x2 fraction2, clamped, integer2;\n\t"
      "mov.f32 c3, 0f3D55EA42;\n\t"
      "mov.f32 c2, 0f3E77150E;\n\t"
      "mov.f32 c1, 0f3F316B3F;\n\t"
      "mov.f32 c0, 0f3F800000;\n\t"
      "mov.f32 c4, 0f3C5C1E37;\n\t"
      "mov.b64 p, {c4, c4};\n\t"
      "mov.b64 coeff, {c3, c3};\n\t"
      "fma.rn.ftz.f32x2 p, fraction2, p, coeff;\n\t"
      "mov.b64 coeff, {c2, c2};\n\t"
      "fma.rn.ftz.f32x2 p, fraction2, p, coeff;\n\t"
      "mov.b64 coeff, {c1, c1};\n\t"
      "fma.rn.ftz.f32x2 p, p, fraction2, coeff;\n\t"
      "mov.b64 coeff, {c0, c0};\n\t"
      "fma.rn.ftz.f32x2 p, p, fraction2, coeff;\n\t"
      "mov.b64 {ix, iy}, rounded;\n\t"
      "mov.b64 {px, py}, p;\n\t"
      "shl.b32 ex, ix, 23;\n\t"
      "shl.b32 ey, iy, 23;\n\t"
      "add.s32 px, ex, px;\n\t"
      "add.s32 py, ey, py;\n\t"
      "mov.b64 joined, {px, py};\n\t"
      "mov.b64 zero2, 0;\n\t"
      "add.rn.ftz.f32x2 joined, joined, zero2;\n\t"
      "mov.b64 {%0, %1}, joined;\n\t"
      "}"
      : "=f"(a), "=f"(b)
      : "f"(xy[0]), "f"(xy[1]));
  xy[0] = a;
  xy[1] = b;
}
} // namespace sm100_attention_exp
