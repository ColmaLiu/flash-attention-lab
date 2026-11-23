# 44 · FP8 forward: E4M3 Q/K/V, descale, FP8 quantization of P, and error

> Part 4 · FA4 — full optimization on Blackwell | Prerequisites: [40_fa4_pipeline](../40_fa4_pipeline/README.md) | Code: [`students/44_fa4_fp8/kernel.cu`](../../students/44_fa4_fp8/kernel.cu)

## 1. Goals

Lab 40 built the FA4 pipeline structure: two Q tiles in ping-pong, P written back to TMEM, a correction warpgroup. This lab switches its inputs to **FP8**:

| | Lab 40 | This lab |
|---|---|---|
| Q, K, V | FP16/BF16, 2 bytes | **E4M3**, 1 byte, plus one FP32 descale per tensor |
| true value | the stored value | `q_descale · Q₈`, `k_descale · K₈`, `v_descale · V₈` |
| Q·Kᵀ, P·V | `tcgen05.mma .kind::f16`, K = 16 per instruction | **`.kind::f8f6f4`, K = 32 per instruction** |
| P | 16-bit, 2 per TMEM column | **E4M3, multiplied by 2⁸ before quantizing**, 4 per TMEM column |
| O | same dtype as the input | **BF16** |
| error | mostly output rounding | **mostly the quantization of P**; this lab measures it |

The pipeline structure (warp roles, barriers, TMEM columns) is **exactly the same** as lab 40 step 3 and is already in the skeleton. The three steps of this lab:

1. **FP8 Q·Kᵀ**: descriptors and the instruction descriptor for 8-bit operands, both descales folded into the softmax scale; S is written to the probe for checking;
2. **FP8 P·V**: P is multiplied by 2⁸, quantized to E4M3 and written back to TMEM four per column; the 8-bit MMA in TS form; the epilogue undoes the scaling;
3. **Error analysis**: measure each source of error, and change the normalization of O to "the sum of the quantized values that actually enter P·V".

After this lab you should be able to:

- write down the bit layout, maximum, smallest normal number and unit roundoff of E4M3, and explain why P is multiplied by 2⁸ instead of being quantized directly;
- explain how the instruction descriptor of `kind::f8f6f4` differs from the 16-bit version, and why the swizzle of an 8-bit tile is "one row = D bytes";
- derive the error bound of FP8 attention and match it against the `max|err|` from `./lab check`;
- explain why the Tensor Core gets 2× faster but the whole kernel only about 1.2× faster.

## 2. Motivation: what FP8 can bring

On B200 the Tensor Core's FP8 multiply throughput is **2×** that of BF16 (nominal dense about 4.5 PFLOP/s vs 2.25 PFLOP/s): in the same roughly 64 cycles, one `kind::f16` MMA computes 128×128×16, and one `kind::f8f6f4` MMA computes 128×128×32. The data is also half as large: Q, K, V are 1 byte/element each, so the K/V ring in shared memory and the TMA traffic are both halved.

The FlashAttention-3 paper, §3.3, was the first to do attention in FP8 on Hopper, and it points out two difficulties:

1. **Layout**: Hopper's FP8 `wgmma` requires both operands to be K-major, but V in P·V is MN-major (D contiguous); FA3 transposes the V tile inside the kernel with `ldmatrix`/`stmatrix`;
2. **Precision**: E4M3 has only 3 mantissa bits. FA3 uses **block quantization** (one scale per block) and **incoherent processing** (Q and K are first multiplied by the same random orthogonal matrix to spread out outliers) to reduce the quantization error of Q and K.

On B200 point 1 no longer exists: `tcgen05.mma` allows MN-major operands for E4M3 (§4.6), so V can be used directly the way lab 40 does. The Q/K/V quantization in point 2 happens in the **inputs**: this lab's harness already provides quantized tensors and per-tensor descales; the only quantization the kernel itself introduces is on **P**. FA4 (`flash_fwd_sm100.py`) multiplies P by 2⁸ and then quantizes it to E4M3. This lab does the same and measures its error.

A reminder: the conclusion of lab 40 §7.2 is that **the bottleneck is softmax (MUFU)**; the Tensor Core is busy only about 57% of the time. FP8 only makes the GEMMs faster, so do not expect 2×. §7.2 estimates the expected speedup and checks it against the measurements.

## 3. Math

### 3.1 E4M3

8 bits: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, **no infinity**; only `S.1111.111` is NaN (PyTorch's `float8_e4m3fn`).

| | Bit pattern | Value |
|---|---|---|
| maximum | `0.1111.110` | 1.75 × 2⁸ = **448** |
| 1 | `0.0111.000` | 1 |
| smallest normal | `0.0001.000` | 2⁻⁶ = 0.015625 |
| smallest subnormal | `0.0000.001` | 2⁻⁹ ≈ 0.00195 |

- Each power-of-two interval [2ᵉ, 2ᵉ⁺¹) holds only 8 values, spaced 2ᵉ⁻³ apart. Round-to-nearest has **relative error at most u = 2⁻⁴ = 6.25%** (u is called the unit roundoff; BF16 has 2⁻⁸, FP16 has 2⁻¹¹);
- in the subnormal range (< 2⁻⁶) the spacing is fixed at 2⁻⁹, so smaller numbers have larger relative error: 0.003 can only become 0.00195 or 0.0039;
- in `cvt.rn.satfinite.e4m3x2.f32`, which converts to E4M3, `.satfinite` is **required** (ptxas does not accept an `.e4m3x2` destination without it): out-of-range values saturate to ±448 instead of becoming NaN.

E5M2 (2 mantissa bits, maximum 57344) has a larger range and lower precision and is commonly used for gradients; the forward pass uses E4M3.

Where the inputs come from: `make_inputs` first generates random FP32 numbers (the true values); Q, K, V are multiplied by 2, 8, 4 respectively and then stored as E4M3, which corresponds to `q_descale = 0.5`, `k_descale = 0.125`, `v_descale = 0.25`. True value = descale × stored value. The three descales are deliberately different powers of two: this keeps the stored values in the E4M3 normal range (a standard normal times 8 peaks at about 40, far below 448), and it guarantees that a "wrong descale" bug (for example `q_descale·q_descale`) gets caught.

### 3.2 Folding descale into scale

The true score is

```
s_real = scale · (q_descale · Q₈) · (k_descale · K₈)ᵀ = (scale · q_descale · k_descale) · (Q₈ · K₈ᵀ)
                                                        └──────── qk_scale ────────┘
```

The Tensor Core computes `Q₈·K₈ᵀ`: the product of two E4M3 values has at most 8 significant bits, which is **exact** in FP32, and accumulation is in FP32. So replacing lab 40's `scale` with `qk_scale` (`scale_log2 = qk_scale · log₂e`) is enough; the online softmax code does not change at all, and m still holds raw scores. There is no extra multiply per element.

The probe in step 1 writes `qk_scale · s` (true units) and compares it with the FP64 `scale · (q_descale·Q₈)(k_descale·K₈)ᵀ`, tolerance 1e-3. The measured error is about 1e-7–1e-5 (only FP32 accumulation rounding).

`v_descale` cannot be folded this way: it multiplies V, and O = Σ P·V is linear, so **one multiply per output element in the epilogue** is enough.

### 3.3 Multiply P by 2⁸, then quantize

In lab 40, `p = 2^(s·scale_log2 − m·scale_log2) ∈ (0, 1]`. In this lab, what is written back to TMEM is

```
p̂ = E4M3(2⁸ · p)            i.e. subtract 8 more in the exp2 argument: 2^(s·scale_log2 − (m·scale_log2 − 8))
```

Why multiply by 2⁸:

- **the maximum 256 < 448**, so nothing saturates;
- before, p < 2⁻⁶ already fell into the subnormal range; after multiplying by 2⁸ that happens only for p < 2⁻¹⁴;
- **a power of two changes only the exponent, not the mantissa**: in the normal range E4M3(2⁸·p) = 2⁸·E4M3(p), so no extra rounding is introduced. That is why 2⁸ beats 448 (448 = 1.75 × 2⁸ changes where each value sits inside its interval; the error just becomes "a different set of random numbers", and §3.5 measures it as slightly worse).

How to undo the 2⁸: ℓ accumulates the same p that was multiplied by 2⁸ (the result of the same `exp2`), so the two 2⁸ factors cancel in `O / ℓ`. LSE has to subtract it:

```
O_real[i] = v_descale · O[i] / ℓ
LSE[i]    = qk_scale · m + ln ℓ − 8·ln 2
```

FA4 does the same; see Note [Low Precision Scaling] in `flash_fwd_sm100.py`: `max_offset = 8`, `scale = rcp(row_sum) * v_descale` in the epilogue, and `log2(row_sum) − max_offset` in LSE.

### 3.4 Worked example: when 2⁸ helps

One row has 1 key with p = 1, and 1000 other keys with p = e⁻⁶ ≈ 0.00248 (the "long tail").

| | One tail p after quantization | Relative error | Tail sum (true value 2.48) | Weight of the largest key (true value 1/3.48 = 0.287) |
|---|---|---|---|---|
| direct quantization | 0.00248 falls in the subnormal range, spacing 2⁻⁹: → 0.00195 | −21% | 1.95 | 1/2.95 = 0.339 |
| multiply by 2⁸ | 0.635 falls in [0.5, 1), spacing 2⁻⁴: → 0.625 | −1.5% | 2.44 | 1/3.44 = 0.291 |

With direct quantization O is off by about 0.05 × |v_top − v̄_tail|; with 2⁸ by about 0.003 ×.

The `long_tail` case pushes this example to the extreme: each head has one hot key (logit ≈ 8), and the other 4095 keys have logit ≈ 0, i.e. p ≈ e⁻⁸ ≈ 3.4e-4. The tail sum is about 1.4, heavier than the hot key; the mean of V is 1.

| `long_tail` [1,4096,2,128] | One tail p | `max|err|` (step 2 / step 3) |
|---|---|---|
| direct quantization | 3.4e-4 < 2⁻¹⁰ (half the smallest subnormal) → **0**: the whole tail vanishes | 0.61 / 2.0 |
| multiply by 2⁸ | 0.086, normal → relative error ≤ 2⁻⁴ | 5.7e-3 / 5.7e-3 |

(Step 3 is worse: it normalizes by Σp̂, so when the tail is 0, O equals the hot key's V.)

For the other cases with random data, keys with p < 2⁻⁶ contribute very little to ℓ: with `kPLog2` set to 0, their error is the same as with 2⁸. The 2⁸ costs almost nothing (one extra subtraction).

### 3.5 Error analysis

The error in O has these sources (measured magnitudes in this lab in parentheses):

| Source | Size | Counted by this lab's check? |
|---|---|---|
| Q, K, V quantized to E4M3 | relative ≤ 2⁻⁴ per element | no: the oracle uses the dequantized inputs (`descale × stored value`). This error belongs to the input; reducing it needs FA3's block quantization / incoherent processing |
| FP32 accumulation of Q₈·K₈ᵀ | relative about 1e-7 | yes (step 1 probe) |
| `ex2.approx` | relative about 2⁻²² | yes, negligible |
| **P quantized to E4M3** | relative ≤ u = 2⁻⁴ per p | **yes: the main source** |
| O rounded to BF16 | relative ≤ 2⁻⁹ | yes |

**Error bound for quantizing P.** Write p̂_j = p_j(1 + δ_j), |δ_j| ≤ u. If the normalization also uses p̂ (step 3), O is a **convex combination** of the rows of V with weights ŵ_j = p̂_j / Σp̂:

```
Ô − O = Σ_j (ŵ_j − w_j) v_j = Σ_j (ŵ_j − w_j)(v_j − O)        (because Σŵ = Σw = 1)
|ŵ_j − w_j| ≤ w_j · 2u/(1 − u)   ⇒   |Ô − O| ≤ 2u/(1−u) · Σ_j w_j |v_j − O|
```

- When the weight is spread over many keys (random data), the δ_j cancel out, and the error is about u·√(Σw_j²)·σ_V, which is small (`medium`: about 7e-3, against 1e-3 when P stays in BF16; see the table below);
- when the weight is concentrated on one or two keys (`large_logits`, the first few rows under causal), nothing cancels, and the error is on the order of u·|v_a − v_b|. This is where the 5e-2–7.5e-2 of `peaked` and `causal_peaked` comes from.

**Should ℓ use the FP32 p or the quantized p̂?** Step 2 (same as FA4) computes ℓ from the FP32 p, but the numerator is Σp̂·v, and the two disagree: for V = 1, O = Σp̂/Σp ≠ 1. Step 3 normalizes by Σp̂ instead, so O is again a convex combination and `ones_v` is exactly 1. **But LSE must still use the FP32 ℓ**: LSE describes the true softmax, and Σp̂ carries a bias of up to about 1% (the LSE tolerance is 1e-3). So the softmax threads keep two sums: `l` (FP32, for LSE) and `lo` (for O).

**Measured** (`max|err|`, compared with the FP64 oracle; "BF16 P" is a PyTorch simulation that rounds P to BF16 block by block with the same tiling, not a kernel; the others are kernel measurements):

| case | BF16 P (simulated) | step 2: E4M3 P, ℓ = Σp | step 3: E4M3 P, O uses Σp̂ | FA4 FP8 |
|---|---|---|---|---|
| `medium` | 9.8e-4 | 7.1e-3 | 7.1e-3 | 6.6e-3 |
| `position` | 2.0e-3 | 4.5e-3 | 3.7e-3 | 4.6e-3 |
| `causal_position` | 1.9e-3 | 4.5e-3 | 2.2e-3 | 3.6e-3 |
| `ones_v` | ≈ 0 | 7.8e-3 | ≈ 1e-6 | 7.8e-3 |
| `causal_ones_v` | ≈ 0 | 2.0e-2 | ≈ 1e-6 | 2.0e-2 |
| `peaked` | 1.2e-2 | 4.7e-2 | 4.5e-2 | 6.6e-2 |
| `causal_peaked` | 1.4e-2 | 7.5e-2 | 5.1e-2 | 7.5e-2 |

- An E4M3 P makes the error about 4–7× larger than a BF16 P, exactly the effect of u going from 2⁻⁸ to 2⁻⁴ (for rows with concentrated weight);
- step 3 removes the Σp̂/Σp bias, and `ones_v` goes from 7.8e-3 to ≈ 0; but the worst rows (`peaked`) barely change, because there the error comes from the relative error of the weights themselves, not from the normalization;
- `kPLog2` set to 0 (no scaling): the random-data cases have the same error as with 2⁸, and `long_tail` and `causal_long_tail` fail (§3.4); set to log₂448: `peaked` becomes 6.1e-2 (worse).

**Tolerance.** The default O tolerance in this lab is `|err| ≤ u + 2u·|expected|`, i.e. (0.0625, 0.125). It takes its magnitude from u and is calibrated by measurement; it is **not** a derived upper bound (the bound above is much looser; a row whose weight is split evenly between two keys can also exceed u·|v_a − v_b|/2). The solution of this lab and FA4 use about 75% of this budget; the variant with P scale 448 uses about 95%.

Structured cases with known answers tighten it with `o_tol` (about 2–3× the measured error): `uniform` and `causal_uniform` (P all 256, exact) use (1e-3, 4e-3), leaving only the rounding of the BF16 output; `position`-type cases 1.5e-2; `ones_v` 2e-2 and `causal_ones_v` 4e-2 (the Σp̂/Σp bias of step 2); `ones_v_exact` and `causal_ones_v_exact`, checked only in step 3, 1e-5. This way a bug such as "truncating P instead of rounding to nearest" fails on a dozen or more cases in step 2.

## 4. Hardware and instructions

### 4.1 `tcgen05.mma .kind::f8f6f4`

```
tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], a_desc, b_desc, idesc, enable_input_d;     // SS
tcgen05.mma.cta_group::1.kind::f8f6f4 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;   // TS
```

| | `kind::f16` | `kind::f8f6f4` (E4M3) |
|---|---|---|
| K per instruction | 16 elements = **32 bytes** | 32 elements = **32 bytes** |
| M × N | 128 × 8..256 | same |
| accumulation | FP32 (or FP16) | FP32 |
| approx. cycles per instruction (128×128) | about 64 | about 64: FLOP per cycle doubles |

The header adds `lab::mma_f8f6f4_ss` and `lab::mma_f8f6f4_ts`. Watch the K-slice stride: **in bytes it is exactly the same as 16-bit**, 32 bytes in both cases; in elements it is 32. At D = 128, one row of Q needs only 4 MMAs (8 in lab 40).

### 4.2 Instruction descriptor

The bit positions are the same as for `kind::f16` (lab 31); only the meaning of the A/B format fields changes:

| Bits | Meaning | This lab |
|---|---|---|
| 4–5 | D format: 1 = F32 | 1 |
| 7–9 / 10–12 | A / B format: **0 = E4M3**, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1 | 0 / 0 |
| 15 / 16 | A / B is MN-major | QKᵀ: 0 / 0; PV: 0 / 1 |
| 17–22 | N >> 3 | |
| 24–28 | M >> 4 | |

`lab::make_idesc_f8f6f4(m, n, a_fmt, b_fmt, a_mn, b_mn)`. A coincidence that is easy to get wrong: the E4M3 format code 0 is the same as the F16 format code 0, so the bits of `make_idesc_f16(128, 128, /*bf16=*/false)` are **identical** to the E4M3 idesc. But paired with `kind::f16` it reads the bytes as FP16, and the result is completely wrong. What decides the data type is **kind plus format code**, not the format code alone.

### 4.3 Shared memory layout of 8-bit tiles

A lab 40 tile is `[D/64 panels][rows][64 16-bit elements]`; one row of a panel is exactly 128 bytes = one 128B swizzle row. In 8 bits, **one row of D elements is D bytes**:

| D | One row | swizzle | Tile layout | One atom (8 rows) | K-major descriptor |
|---|---|---|---|---|---|
| 128 | 128 bytes | 128B | `[128 rows][128]`, one panel | 1024 bytes | `make_smem_desc_kmajor_sw128`, SBO = 1024 |
| 64 | 64 bytes | **64B** | `[128 rows][64]`, one panel | 512 bytes | `make_smem_desc_kmajor_sw64`, SBO = 512 |

The TMA tensor map uses `CU_TENSOR_MAP_DATA_TYPE_UINT8` (TMA only moves bytes and does not care about E4M3), box `[1][rows][1][D]`, with the swizzle from the table above (the skeleton's `fp8_map` already does this). The XOR pattern of the 64B swizzle repeats every 512 bytes, so an atom must be 512-byte aligned; in this lab all tile offsets are multiples of 1024.

The start address of K slice kk = tile start + 32·kk bytes. The descriptor's start address field is in units of 16 bytes, so it is `desc + 2·kk`. At D = 64 a row has only two slices: 0 and 32 bytes.

### 4.4 Writing P as E4M3, 4 per column

```
cvt.rn.satfinite.e4m3x2.f32 d16, a, b;     // a → high byte of d, b → low byte
```

**The operand order is reversed**: the first source operand goes to the high byte. Four keys are packed into one 32-bit word, with key 4c in the lowest byte:

```
32 bits of TMEM column p_addr + c (c = 0..31) = { byte0: p̂[4c], byte1: p̂[4c+1], byte2: p̂[4c+2], byte3: p̂[4c+3] }
```

A row of 128 keys takes only **32 columns** (lab 40's 16-bit P takes 64 columns). It still starts at column 64 of S (same as FA4's `tmem_s_to_p_offset = n_block_size // 2`), written with two `tcgen05.st.32x32b.x16`.

### 4.5 The 8-bit MMA in TS form

A (P) is read from TMEM: row i is in lane i, and the K = 32 keys of one MMA are **8 columns**. So key slice kk starts at `p_tmem + 8·kk` (lab 40 is 16-bit with 2 per column, also 8 columns).

### 4.6 V needs no transpose

The B operand of P·V is V_j `[key][D]` with D contiguous, i.e. **MN-major**. The comment on `InstrDescriptor` in CUTLASS `mma_sm100_desc.hpp`: "Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats". E4M3 may be MN-major; only the 6-bit and 4-bit formats may not. FA4 also uses `OperandMajorMode.MN` for FP8 V.

The swizzle row of 8-bit MN-major is "the D bytes of one key": at D = 128 a row is 128 bytes (N = D = 128 is exactly one row, so LBO is unused), 8 keys form one 1024-byte atom, SBO = 1024; D = 64 uses the 64B swizzle, SBO = 512. Key slice kk advances 32 keys = 32·D bytes.

So this lab has **no transpose**, and therefore no question of "whether the transpose counts toward the timing": TMA loads V as is, and the timing boundary is the same as in lab 40. FA3 on Hopper must transpose V inside the kernel (within the timing).

### 4.7 Step 3: summing the quantized P

```
cvt.rn.f16x2.e4m3x2 h01, w16;          // two E4M3 → two FP16 (exact)
add.rn.f32.f16 acc, h, acc;            // FP32 += FP16 (PTX ISA 8.6, sm_100), one FHADD in SASS
```

E4M3 → FP16 is exact, and the mixed-precision add rounds to FP32 only once, so the result is the same as "convert to FP32 first, then add", but each element needs only one FHADD instead of the two instructions `HADD2.F32` + `FADD`. The skeleton already provides `add_e4m3x4(acc, w)`. Use several independent partial sums so that the 128 adds do not form one long dependency chain.

Note that this way the denominator of O is Σp̂, while LSE uses the FP32 Σp, so exp(LSE) is not the normalization constant O actually uses (they differ by about 1%). This is fine for the forward outputs; if a backward pass recomputes P from LSE, be aware that it does not exactly match the forward P̂.

### 4.8 Common misuses

| Misuse | Consequence |
|---|---|
| `kind::f16` with the E4M3 idesc (same bits!) | byte pairs are read as FP16: S is completely wrong |
| K-slice stride of 16 elements (copying lab 40's element count) | only half of D is used, the other half reads the next slice: S is wrong |
| D = 64 still uses the 128B swizzle descriptor | does not match the 64B swizzle that TMA writes: D = 64 cases fail |
| the two source operands of `cvt.e4m3x2` in the wrong order | P of adjacent keys swapped: `n1` and `peaked*` fail |
| P scale above 448 (for example × 2¹⁰) | `.satfinite` (required with `.e4m3x2`) saturates large p to 448, O is too small |
| P multiplied by 2⁸ but LSE does not subtract 8·ln2 | every LSE is 5.545 too large |
| step 3 also switches LSE to Σp̂ | LSE error about 1e-2, above the 1e-3 tolerance |
| forgot to multiply by `v_descale` | O is 4× too large |

## 5. Schedule design

Roles, barriers and the MMA issue order are the same as lab 40 step 3 (lab 40 §5, 16 warps, `setmaxnreg` 192 / 80 / 48). Only the storage budget changes:

**shared memory** (D = 128 / D = 64):

| | Lab 40 step 3 (BF16) | This lab (E4M3) |
|---|---|---|
| Q⁰, Q¹ | 64 KB / 32 KB | 32 KB / 16 KB |
| O staging (BF16, 128 × D × 2 bytes) | reuses Q^t (same size) | **reuses Q⁰ + Q¹ together**, used by the two tiles one after the other |
| K/V ring | 4 × 32 KB / 8 × 16 KB | 8 × 16 KB / 12 × 8 KB |
| statistics | 3 KB | 4 KB (adds the O normalization and `fin_lo`) |
| total | ≈ 195 KB / ≈ 163 KB | ≈ 165 KB / ≈ 117 KB |

Why O staging may overwrite both Q buffers: `o_full[0]` is completed by the commit after the last PV⁰, all QK¹ are issued before it, and a commit tracks all earlier MMAs of the same thread (lab 40 §4.3), so by then neither Q⁰ nor Q¹ is read anymore. The correction warpgroup first writes tile 0's O; in each warp, lane 0 does `bulk_wait_group_read<0>`, then `__syncwarp`, and then the warp writes the same rows of tile 1 (each warp touches only its own 32 rows).

Number of K/V ring stages: one block is only 16 KB, so the same bytes hold 8 stages. Measured, 4 stages and 8 stages differ only within noise (the data uses only about 2% of HBM bandwidth).

**TMEM**: the column numbers are the same as lab 40, except that P^t uses only columns 64–95 of S^t.

| Columns | D = 128 | D = 64 |
|---|---|---|
| S⁰ (P⁰ at 64–95) | 0–127 | 0–127 |
| S¹ (P¹ at 192–223) | 128–255 | 128–255 |
| O⁰ / O¹ | 256–383 / 384–511 | 256–319 / 320–383 |

**Extra work per block for the softmax threads** (compared with lab 40): 64 `F2FP` (E4M3 packing, two per instruction); step 3 adds another 64 `F2FP` (unpacking) and 128 `FHADD`. None of these run on MUFU.

## 6. Step-by-step exercises

The skeleton already contains the whole lab 40 pipeline (TMA, MMA scheduling, softmax, correction, barriers), the 8-bit tensor maps, O staging, and `add_e4m3x4`. `p.step` selects one of three configurations (`Config<D, 1|2|3>`); the three steps complete the same kernel bit by bit.

### Step 1 · TODO(1a)–(1c): FP8 Q·Kᵀ

- 1a (host, `launch`): `qk_scale`, `scale_log2` (§3.2);
- 1b: `desc_kmajor` and `mma_qk` (§4.1–4.3);
- 1c: `write_probe`: write S in true units to the probe (§3.2).

In this step the MMA warp does not issue P·V and the epilogue does not write O; only the probe is checked (small non-causal cases).

```bash
./lab check 44 --step 1 --case n1
./lab check 44 --step 1
```

### Step 2 · TODO(2a)–(2c): FP8 P·V

- 2a: `pack4_e4m3` and `write_p_tmem` (§4.4). The −8 in the `exp2` argument is already written (`base - kPLog2`);
- 2b: `mma_pv_ts` (§4.5–4.6);
- 2c: `f` and LSE in `store_o_and_lse` (§3.3).

```bash
./lab check 44 --step 2 --case tile1_position
./lab check 44 --step 2
./lab sass 44             # UTCQMMA: the one whose A operand is tmem[...] is the TS-form P·V
./lab bench 44 --step 2 --case perf
```

### Step 3 · TODO(3): error analysis

1. First record the `max|err|` of step 2: `./lab check 44 --step 2`, especially `ones_v`, `causal_ones_v`, `position`, `peaked`;
2. in the `kQuantL` branch of `write_p_tmem`, return the sum of the quantized values (§4.7). The update of `lo` in the kernel and the split of roles between `lo`/`l` in the epilogue are already written; read and understand them (§3.5);
3. compare the errors and explain them with the error bound from §3.5: which cases improved, which did not change, and why.

```bash
./lab check 44 --step 3
./lab bench 44 --case perf
./lab bench 44 --case perf_d64
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| step 1: every probe is off by a constant factor (such as 4× or 1/4) | `qk_scale` uses the wrong descale (the three descales all differ: 0.5, 0.125, 0.25), or multiplies by log₂e once too often |
| step 1: only the D = 64 cases fail (`one_block`, `scale`, `peaked_d64`) | D = 64 uses the 128B swizzle descriptor; SBO is not 512 |
| step 1: D = 128 fails and `n1` also fails | byte offset of the K slice; N/M in the idesc; used `mma_f16_ss` |
| step 2: O of `n1` is wrong, LSE is right | byte order of P (`n1` has only key 0, and it must be in byte 0) |
| step 2: O is right, every LSE is off by 5.545 | LSE does not subtract 8·ln 2 |
| step 2: every O is off by 4× | forgot `v_descale` |
| step 2: O is wrong at D = 64, right at D = 128 | V descriptor: 64B swizzle, SBO = 512, key-slice stride 32·D bytes |
| step 3: LSE off by about 1e-2 | LSE was also switched to Σp̂ (§3.5) |
| step 3: `ones_v_exact` fails (O is not exactly 1) | `lo` still accumulates the FP32 p; or a partial sum is missing |
| step 2: `position`-type cases and `ones_v` have errors of about 2.5e-2–6e-2 | P is truncated instead of rounded to nearest (for example by assembling bits by hand instead of using `cvt.rn`) |
| step 2: `long_tail` error about 0.6 | P is not multiplied by 2⁸ (`base - kPLog2` was changed) |

## 7. Performance

### 7.1 Against FA4 FP8 and lab 40

Measured on this machine (CUDA Graph, median; `vs FA4 FP8` = FA4 FP8 time / time of this version; ranges come from 2–3 runs, with about ±3% variation between single runs):

| Version | `perf` [4,4096,16,128] | `perf_causal` | `perf_d64` [4,4096,32,64] | `perf_8k` [2,8192,16,128] |
|---|---|---|---|---|
| step 2 (E4M3 P, ℓ = Σp) | ≈ 0.68–0.69× | ≈ 0.71× | ≈ 0.73× | ≈ 0.68× |
| step 3 (solution) | ≈ 0.69–0.73× | ≈ 0.73× | ≈ 0.74× | ≈ 0.70–0.72× |
| FA4 FP8 (`flash_attn.cute`) | 1× | 1× | 1× | 1× |
| this lab's step 3 relative to lab 40 step 3 (BF16/FP16) | ≈ 1.23× | ≈ 1.15× | ≈ 1.12× | ≈ 1.24× |
| FA4 FP8 relative to FA4 BF16/FP16 | ≈ 1.27× | ≈ 1.12× | ≈ 1.05× | ≈ 1.37× |

- If your step 3 reaches about 0.63× of FA4 FP8 or better on all four perf cases, the mechanisms of this lab work. cuDNN has no FP8 SDPA baseline (`bench` shows "unavailable");
- step 3 adds unpacking and 128 FHADD, yet it is as fast as step 2 or slightly faster (about 1–4%, reproducible). We have not found the cause (possibly a change in instruction scheduling); the conclusion is that "consistent normalization" costs almost no time;
- FA4 is about 1.4× faster than us because it already uses the optimizations of labs 41–43 (polynomial exp2, conditional rescale, persistent scheduling, and so on), which are not part of this lab.

### 7.2 Why only about 1.2× faster

The chain model from lab 40 §3.1: the cycles per block of one tile are `T ≈ SM + PV + QK + handover`. Lab 40 (D = 128) measured SM ≈ 1900–2200 cycles, PV + QK ≈ 1100 cycles, T ≈ 3450. FP8 halves PV + QK to about 550 cycles, and softmax does not change:

```
T_fp8 ≈ 3450 − 550 ≈ 2900 cycles    →   3450 / 2900 ≈ 1.19×
```

This matches the roughly 1.23× measured on `perf` and `perf_8k`. At D = 64 the GEMMs take a smaller share of the chain, so the speedup is only about 1.12×.

Nsight Compute (`./lab profile 44 --solution --case perf`):

| Metric | Lab 40 step 3 | This lab |
|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 57% | **≈ 29%** |
| `sm__inst_executed_pipe_xu` (MUFU) | ≈ 57% | ≈ 58% |
| `dram__throughput` | — | ≈ 2% |

The Tensor Core busy fraction is exactly halved, and the MUFU fraction does not change: the bottleneck is still softmax's `exp2`. HBM is almost idle, so "half the bytes" has no direct effect on this kernel's time (its effect is that shared memory can hold more stages and leaves room for larger tiles). FP8 attention is **MUFU-bound**, as the comment on `_FP8_TUNING_CONFIG` in the FA4 source also says ("FP8 fwd is MUFU/ex2-bound"), so lab 41's polynomial `exp2` matters even more for FP8.

```bash
./lab profile 44 --case perf
```

## 8. Self-check questions

1. Can E4M3 represent 300? What about 0.004? What does each get rounded to?
2. At D = 64, what is the byte offset in shared memory of element 40 of row r of the Q tile (relative to the tile start, with the 64B swizzle: 16-byte chunk q is stored at q ^ ((r / 2) % 4))? Why can the 128B swizzle not be used?
3. For PV MMA number kk = 3: what is A's TMEM column (relative to base, t = 1)? How many bytes does V's descriptor start address advance (D = 128)?
4. What happens if P is multiplied by 2¹⁰ instead? With 2⁸, what is the largest p̂, and how far is it from 448? When FA4 lets the softmax max "lag" (lab 41's conditional rescale), why must this margin be accounted for?
5. Why does step 3 normalize O by Σp̂ but compute LSE with the FP32 Σp? What happens the other way round?
6. FP8 halves the Tensor Core work time; why is `perf` only about 1.2× faster? If softmax also became 2× faster (one of lab 41's goals), how much faster would it be under the §7.2 model?

<details>
<summary>Answers</summary>

1. 300 lies in the interval [256, 448], spacing 32: 300 → 288 (12 away from 288, 20 away from 320). 0.004 is in the subnormal range, spacing 2⁻⁹ ≈ 0.00195: 0.004 ≈ 2.05 × 2⁻⁹ → 2 × 2⁻⁹ = 0.0039.
2. Byte 40 is in 16-byte chunk q = 2 (bytes 32–47), at offset 8 within the chunk; the 64B swizzle puts chunk q at q ^ ((r / 2) % 4), so offset = 64·r + 16·(2 ^ ((r/2) % 4)) + 8; for example, for r = 5 it is 320 + 16·0 + 8 = 328. The `tcgen05` 128B-swizzle K-major layout assumes 128 bytes per row and 8 rows per 1024-byte atom; a D = 64 row has only 64 bytes, so reading it with the 128B layout treats two rows as one. The descriptor's swizzle must match the swizzle TMA used when writing.
3. P¹ starts at column 64 of S¹: `128 + 64 + 8·3 = 216` (lane field 0), which holds keys 96–127. V advances 3 × 32 keys × 128 bytes = 12288 bytes (descriptor + 768).
4. 2¹⁰ = 1024 > 448: values with p close to 1 saturate to 448 (`satfinite`), so the largest weights in P·V get squashed, while the FP32 ℓ still counts them at 1024, and O comes out too small (the problem in FA4 #2716). With 2⁸ the maximum is 256, and the margin is 448/256 = 1.75, i.e. about 0.8 in log₂. If the max may lag by τ log₂ units, p can reach 2^τ, which becomes 2^(8+τ) after multiplying by 2⁸ and must be < 448. So FA4 asserts `max_offset + rescale_threshold < log2(448)`, and for FP8 rescale_threshold is set to 0.
5. O = Σp̂v / Σp̂ is an exact convex combination: the weights sum to 1, `ones_v` is exactly 1, and the bias cannot enter O in the form of the mean of v. LSE is the normalization constant of the true softmax and should be ln Σexp(s); Σp̂ carries up to 6.25% relative error per term, which averages to an error on the order of 1e-2, above the LSE tolerance. The other way round (O uses Σp, LSE uses Σp̂), both get worse.
6. Chain T ≈ SM + PV + QK + handover; only PV + QK is halved (about 1100 → 550 cycles), while SM ≈ 2000 does not change: 3450 / 2900 ≈ 1.19. If SM were halved to about 1000, the chain would be only about 1000 + 550 + handover ≈ 1700 cycles; but the two tiles share one MUFU: each block has 2 × 128 × 128 `exp2`, at 16 per cycle, so at least 2048 cycles. So as long as every `exp2` runs on MUFU, T cannot drop below about 2048, which is at most about 2900 / 2048 ≈ 1.4× faster than this lab; FA4 FP8 is indeed about 1.4× faster. To get below this floor, part of the `exp2` work must move to the FMA units (lab 41).
</details>

## 9. Further reading

- Jay Shah et al., *FlashAttention-3* (2024) §3.3: the FP8 layout requirements (V transpose, register arrangement of P), block quantization, incoherent processing, and their error experiments.
- Zadouri et al., [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) (2026); upstream implementation `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`:
  - Note [Low Precision Scaling] and `_LOG2_DTYPE_MAX`: multiplying P by 2^max_offset, and the constraint with rescale_threshold;
  - `_load_effective_descales`: `qk_descale = q_descale · k_descale` folded into `softmax_scale_log2`;
  - around `correction_epilogue`: `scale = rcp(row_sum) * v_descale`, and `− max_offset` in LSE;
  - `_FP8_TUNING_CONFIG`: "FP8 fwd is MUFU/ex2-bound";
  - `_flash_attn_fwd` in `flash_attn/cute/interface.py` (`q_descale` and other parameters, shape `[B, H_kv]`; the public `flash_attn_func` does not accept descales).
- CUTLASS: `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp` (`MXF8F6F4Format`, the formats for which MN-major applies in `InstrDescriptor`); `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` (`SM100_MMA_F8F6F4_SS`, `SM100_MMA_F8F6F4_TS`: "A from TMEM can't be transposed").
- PTX ISA 8.7: [tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma) (`.kind::f8f6f4`, instruction descriptor table), [cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt) (`.e4m3x2`, `.satfinite`), [add](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add) (mixed-precision `add.f32.f16`).
- OCP 8-bit Floating Point Specification (OFP8): definition of E4M3/E5M2.
- Next: lab 41 (softmax throughput, which FP8 benefits from even more), labs 42 and 43.
