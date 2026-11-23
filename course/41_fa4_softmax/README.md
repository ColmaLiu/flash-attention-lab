# 41 · Softmax throughput: polynomial exp2, MUFU/FMA mix, conditional rescale

> Part 4 · FA4 — full optimization on Blackwell | Prerequisites: [40_fa4_pipeline](../40_fa4_pipeline/README.md), [01_softmax](../01_softmax/README.md) | Code: [`students/41_fa4_softmax/kernel.cu`](../../students/41_fa4_softmax/kernel.cu)

## 1. Goals

Lab 40 built the FA4 pipeline structure and ended with this conclusion: in the per-block "chain" of a tile, the longest segment is the 128 `exp2`s, and **the throughput of MUFU (the special function unit) became the bottleneck**. This lab leaves the pipeline structure alone (warp roles, barriers and the TMEM layout all carry over from lab 40 step 3) and changes only the softmax arithmetic. It does the two things the FA4 paper does for softmax:

| | Lab 40 step 3 | This lab |
|---|---|---|
| Exponentials | All `MUFU.EX2` | Some are computed with a polynomial on the **FMA pipeline** (software `exp2`), the rest still use MUFU, and both pipelines work at the same time |
| Packed arithmetic | Scalar `FFMA`/`FADD` | **`FFMA2`/`FADD2`**: one instruction does two FP32 operations |
| Correction of O | Every block | **Conditional rescale**: move the reference and correct O only when the max grows by more than τ = 8 (log₂ units) |

The interface is unchanged: FP16/BF16 Q, K, V, O, `[B,N,H,D]`, D = 64 or 128, causal and non-causal, output O and an FP32 LSE. Three steps:

1. **Software `exp2`**: range reduction + degree-3 polynomial + exponent-bit splicing; every exponential is replaced by it (expected to be slower);
2. **Mix**: in every 32 columns, a fixed 6 columns use the polynomial and 26 use MUFU;
3. **Conditional rescale**: softmax moves the reference only when needed, and the correction warp uses a warp vote to skip corrections that are not needed.

When you finish, you should be able to:

- use the measured throughput (results per SM per cycle) to compute the MUFU lower bound of one softmax block, and how much of it the software `exp2` can take over;
- write the bit-level implementation of `2^x = 2^⌊x⌋ · 2^f`, state the error of each step, and explain why the constant term must be exactly 1;
- choose the polynomial degree from the BF16/FP16 rounding error, and explain where the exhaustively measured maximum relative error in this lesson comes from;
- prove that conditional rescale gives a result mathematically identical to rescaling every block, and give the condition for no overflow.

## 2. Motivation: where lab 40 is stuck

Instrumentation from lab 40 §7.2 (`perf` case, steady state, SM cycles):

| Segment | Cycles |
|---|---|
| Period T of one block (one softmax for each of the two tiles) | ≈ 3450 |
| Softmax WG: S received → arrive `p_ready` | ≈ 1900–2200 |
| Of which, the 128 `exp2`s (including the sum) | ≈ 1450–1700 |
| PV + QK of one tile | ≈ 1100 |

The Tensor Core and MUFU are each busy only about 57% of the time: neither is saturated; the chain of one tile (softmax → PV → QK → softmax) is too long. The longest segment of the chain is the exponentials.

FA4 (Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, 2026, *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*) analyzes it this way: Blackwell doubled the Tensor Core throughput over Hopper, but MUFU did not change, so the "non-matmul" part takes a larger share. The paper gives two remedies, and they are the content of this lab:

| Problem | FA4's remedy | Step in this lab |
|---|---|---|
| MUFU throughput (16 `exp2` per SM per cycle) cannot keep up with the Tensor Core | Compute some `exp2` with a polynomial on the FMA units (Cody–Waite range reduction + polynomial) | 1, 2 |
| Every block must read O from TMEM, multiply by α and write it back | Skip the correction when the max changes little ("conditional softmax rescaling") | 3 |

## 3. Math

This lab uses the notation of lab 40: `s` is a raw score (one element of `Q·Kᵀ`), `scale_log2 = scale · log₂e`, and the exponent argument of one element is

```
x = s · scale_log2 − m · scale_log2        (m: current reference, in raw score units)
p = 2^x
```

### 3.1 Lower bound on the throughput of one softmax block

One softmax warpgroup handles 128 rows × 128 columns = 16384 exponentials per block; one CTA has two tiles, so 32768 per block. MUFU on B200 does **16 `ex2` per SM per cycle** (measured in §4.1), so

```
MUFU lower bound = 32768 / 16 = 2048 cycles / block (two tiles)
Tensor Core (D = 128): per tile per block PV + QK = 2 × 128×128×128×2 FLOP = 8.4 MFLOP,
                       two tiles 16.8 MFLOP / 8192 FLOP per SM per cycle ≈ 2048 cycles
```

At D = 128 the two are exactly equal; at D = 64 the Tensor Core needs only about 1024 cycles while MUFU still needs 2048, so **softmax is even more the bottleneck**. As long as every exponential runs on MUFU, one block cannot be faster than 2048 cycles, and in lab 40 a block takes about 3450 cycles (the chain latency).

The FMA pipeline does 128 FP32 FMAs per SM per cycle (§4.1), 8 times MUFU. If one `exp2` takes about 8 FMA-class instructions, the FMA pipeline supplies about the same exponential throughput as MUFU, and together they roughly double it.

### 3.2 Range reduction: `2^x = 2^n · 2^f`

Any real x can be written as `x = n + f`, where `n = ⌊x⌋` is an integer and `f ∈ [0, 1)`:

```
2^x = 2^n · 2^f,      2^f ∈ [1, 2)
```

- `2^f` lies in a fixed small interval and can be approximated by a polynomial (§3.3);
- multiplying by `2^n` needs no multiply: an FP32 value is `(−1)^sign · 1.mantissa · 2^(exponent − 127)`, the exponent field of `2^f ∈ [1, 2)` is exactly 127, and adding n to the exponent field (bits 23–30) gives `2^n · 2^f`.

Worked example: x = −2.75. n = −3, f = 0.25, `2^0.25 ≈ 1.1892`, bit pattern `0x3F9837F0`; adding `−3 << 23 = 0xFE800000` (32-bit two's complement) gives `0x3E1837F0` = 0.14865 = 2^−2.75. ✓

#### Computing n without `floorf`: add a "magic number"

`floorf(x)` is `FRND.FLOOR` in SASS, and like MUFU it does only 16 per SM per cycle (measured in §4.1), so using it wastes the gain. FA4 uses one FP32 addition:

```
r = x + 1.5 · 2^23     (round down, add.rm)
```

`1.5 · 2^23 = 12582912` lies in `[2^23, 2^24)`, where adjacent FP32 values differ by exactly 1, so r is an integer, and rounding down guarantees `r = 12582912 + ⌊x⌋` (as long as |x| < 2^22). Its bit pattern is `0x4B400000 + ⌊x⌋`: **n sits in the low bits of r** (two's complement; a borrow cannot reach the high bits because the 0.5 in 1.5 acts as padding). So:

```
n_float = r − 12582912        exact (difference of two integers in the same binade)
f       = x − n_float         exact for x ≥ 0 or x ≤ −0.5; for x ∈ (−0.5, 0) f is close to 1,
                              and the absolute rounding error is ≤ 2^−25, far below the polynomial error
bits(2^x) = bits(p) + (bits(r) << 23)
```

The last line: `bits(r) << 23` shifts the low 9 bits of r into the exponent field and shifts all high bits (the `0x4B4…` part) out of the 32 bits; what remains is exactly `n << 23` (mod 2^32).

#### Edge cases: −∞ and very small x

A masked key has s = −∞, so x = −∞. The software `exp2` first does `x = max(x, −127)`:

- x = −127: n = −127, f = 0, `p(0) = c0`. If c0 is exactly 1.0 (bit pattern `0x3F800000 = 127 << 23`), adding `−127 << 23` gives exactly **0x00000000 = +0**. If c0 = 0.9999 (bit pattern smaller than `0x3F800000`), the sum borrows into the sign bit and gives a **negative NaN or a negative number**. That is why the constant term of the polynomial is fixed to 1 (§3.3).
- −127 < x < −126: the spliced result is the subnormal (p − 1)·2^−126, which **is not equal** to 2^x (subnormals have no implicit 1). But it is < 2^−126 ≈ 1.2·10⁻³⁸, negligible next to the row sum (≥ 1), and after conversion to BF16/FP16 it is 0 or a subnormal anyway, so the error does no harm.
- Upper bound on x: in lab 40, x ≤ 0; the conditional rescale of step 3 allows x ≤ τ = 8 (§3.5). n ≤ 8 does not overflow the exponent field (that requires n ≤ 127).

### 3.3 The polynomial and the error budget

**The error budget is set by the data type of P.** P is rounded to 16 bits when it is written back to TMEM, and the relative error of this step is the unit roundoff u:

| Type of P | Significant bits (including the implicit bit) | u = 2^−p |
|---|---|---|
| BF16 | 8 | 2^−8 ≈ 3.9·10⁻³ |
| FP16 | 11 | 2^−11 ≈ 4.9·10⁻⁴ |

The maximum relative error ε of the polynomial should be clearly smaller than u, so that the error of P is still dominated by rounding. The same code must also serve FP16, so we choose by the u of FP16.

**Where the coefficients come from:** on `[0, 1)`, find `p(f) = 1 + c1·f + … + cd·f^d` (constant term fixed to 1) that minimizes the **relative** error `max |p(f)/2^f − 1|`. This is a minimax problem (minimize the maximum error); Sollya's `fpminimax` or the Remez algorithm can solve it. This lab uses a Lawson iteration in numpy (weighted least squares, with weights repeatedly scaled up by the error) on 20000 points, then rounds the coefficients to FP32. Result (FA4's `POLY_EX2` in `flash_attn/cute/utils.py` uses the same form; the coefficients differ in the 3rd digit):

| Degree d | Coefficients c1, c2, … (FP32) | Max relative error (measured) | Compared with FP16 u |
|---|---|---|---|
| 2 | 0.66596067, 0.32993290 | 2.05·10⁻³ = 2^−8.9 | 4.2 u: not enough |
| **3** | **0.69511682, 0.22764485, 0.07706717** | **8.57·10⁻⁵ = 2^−13.5** | **0.18 u** |
| 4 | 0.69304484, 0.24128021, 0.05224245, 0.01342670 | 2.97·10⁻⁶ = 2^−18.4 | 0.006 u |
| `MUFU.EX2` (reference) | — | 1.46·10⁻⁷ = 2^−22.7 | — |

"Measured" means: compute the software `exp2` on the GPU for **every** FP32 input in `[−127, 8]` (about 2.2 billion) and compare with the FP64 `exp2` (relative error where the output is normal; where the output is subnormal, only check that the result is non-negative and smaller than 2^−126). The error of the degree-3 polynomial is exactly the error of the polynomial itself on `[0, 1)`: range reduction adds an absolute error ≤ 2^−25 only for x ∈ (−0.5, 0), and the exponent splicing is exact.

We choose **degree 3**: it is 45 times smaller than the rounding error for BF16 and 5 times smaller for FP16, and each exponential needs only 3 FMAs. Degree 2 is actually enough for BF16 (0.5 u) but not for FP16. FA4 also uses degree 3.

### 3.4 Effect on O and LSE

Each element of P has a relative error εᵢ (|εᵢ| ≤ 8.6·10⁻⁵ + rounding). O = Σ pᵢ(1+εᵢ) vᵢ / Σ pᵢ(1+εᵢ). The numerator and denominator use the same set of perturbed p (ℓ uses the FP32 p before rounding; the difference is only the 16-bit rounding), so the error of O is on the order of `max|εᵢ| · max|vᵢ|`, and the error of LSE is about `max|εᵢ|` (ln(1+ε) ≈ ε). Both are far below the tolerances of this lab (O: FP16 `2e-3 + 5e-3·|ref|`, BF16 `1e-2 + 2e-2·|ref|`; LSE: `1e-3 + 1e-4·|ref|`). §7.3 gives measurements.

### 3.5 Why conditional rescale is exact

In online softmax, m has two roles: (a) keep the exponentials from overflowing; (b) serve as the common reference for `ℓ` and O. In the final output

```
O   = Σ_j 2^(x_j) v_j / Σ_j 2^(x_j),   x_j = s_j·scale_log2 − m·scale_log2
LSE = m·scale + ln ℓ
```

**the value of m does not affect O** (numerator and denominator are both multiplied by `2^(−m·scale_log2)`, which cancels), and does not affect LSE either (`m·scale + ln(Σ e^(scale(s_j − m))) = ln Σ e^(scale·s_j)`). The only requirement: **ℓ, O and P of the same row always use the same reference**, and when the reference changes, all three are multiplied by α together.

So the reference does not have to be the true maximum. The conditional rescale rule (FA4's `rescale_threshold`):

```
m_new = max(m, row max of this block)
if (m_new − m)·scale_log2 > τ:  α = 2^((m − m_new)·scale_log2), m ← m_new   (move the reference, correct O, ℓ)
otherwise:                      α = 1, m unchanged                          (P uses the old reference)
```

**Invariant:** at any time `true max − m ≤ τ / scale_log2`, that is `x ≤ τ`, so every `p ≤ 2^τ`.

**Condition for no overflow** (τ = 8, `p ≤ 256`):

| Quantity | Upper bound | Limit of the type |
|---|---|---|
| P (written as 16 bits) | 256 | FP16 65504 ≈ 2^16, BF16 ≈ 2^128 |
| ℓ (FP32) | 256 · N, ≈ 2.6·10⁸ for N = 10⁶ | FP32 ≈ 3.4·10³⁸ |
| O accumulator (FP32) | 256 · N · max|v| | Same as above |

**Precision is unchanged:** the relative precision of a floating-point number does not depend on its magnitude, so multiplying P by a power of 2 (`p ∈ [2^−τ·…, 2^τ]`) does not change its relative rounding error; if anything, fewer small P underflow to 0. It also does not change the polynomial error: computing f and p(f) does not depend on n.

**τ cannot be too large:** FP16 P must be ≤ 65504, so τ < 16; the software `exp2` requires x ≤ 127. FA4 uses τ = 8 for 16-bit inputs (`flash_fwd_sm100.py`, `rescale_threshold = 8.0`) and asserts `max_offset + τ < log₂(dtype max)`.

**Worked example:** a row has max (log₂ units) 3.0 in block 0, 9.5 in block 1 and 12.0 in block 2.
- Block 0: m = −∞ → difference = +∞ > 8, move: reference 3.0, α = 0;
- Block 1: difference = 6.5 ≤ 8, no move: α = 1, the largest p in this block = 2^6.5 ≈ 90.5;
- Block 2: difference (against the reference 3.0, not the previous block's 9.5) = 9.0 > 8, move: reference 12.0, α = 2^−9.

If you wrongly compare against "the previous block's max" (block 2: 12 − 9.5 = 2.5, no move), the reference falls further and further behind, and p eventually exceeds the FP16 upper bound. The `growing` case checks exactly this.

**What it saves:** for random data, the standard deviation of a row of S in log₂ units is about 1.4 (with scale = 1/√D and standard normal Q, K), and the max of one block is usually only 1–2 below the max of the whole row; the probability that the max rises by another 8 after block 0 is negligible. So **after block 0, O is almost never corrected again**: each tile saves, per block, D columns of `tcgen05.ld` + D FMULs + D columns of `tcgen05.st` (per correction thread), and the issue slots and TMEM bandwidth freed by the correction warp go to the softmax warps on the same SMSP.

### 3.6 Mix ratio: the balance point of the two pipelines

On one SMSP (a quarter of the SM, one warp scheduler), the pipeline cycles one warp occupies to process one element (one per lane, 32 lanes) are (converted from the measurements in §4.1):

| Path | MUFU pipeline | FMA pipeline |
|---|---|---|
| `MUFU.EX2` | 8 (4 per SMSP per cycle) | — |
| Polynomial (packed, per element) | — | ≈ 7.4 (17.4 results per SM per cycle) |
| Rest of softmax (`FFMA2` for x, `FADD2` for the sum, per element) | — | ≈ 2 |

If a fraction φ of the elements takes the polynomial path, the time per element is about

```
T(φ) = max( 8 · (1 − φ),  2 + 7.4 · φ )
balance point: 8 − 8φ = 2 + 7.4φ  →  φ ≈ 0.39, T ≈ 4.9 (8 with all MUFU)
```

This model counts only the throughput of the two pipelines. It leaves out: each SMSP issues only one instruction per cycle (the polynomial takes about 10 instructions per element pair); each SMSP has only two softmax warps plus one correction warp, so the polynomial's 7 dependent instructions (about 4 cycles of latency each) are hard to hide behind other warps; the constants and temporaries of the polynomial take up registers. So the model only points in the direction of the upper bound, and the real optimum has to be measured (§7.2). In this solution's sweep, k = 3–5 (φ ≈ 19–31%) run at the same speed within measurement noise, but at k ≥ 4 softmax starts to spill registers (§5.4), so we take k = 3 (φ ≈ 19%), which does not spill. Because of the register spills, the sweep cannot test whether the model's balance point φ ≈ 0.39 itself holds. FA4's default configurations lie between 12% and 19% (`ex2_emu_freq`, §9).

## 4. Hardware and instructions

### 4.1 Measured: results per SM per cycle for each instruction

Measured on B200 (microbenchmark: 148 SMs, 32 warps per SM, 8 independent dependency chains per thread, timed with `clock64()`; you can write one the same way to verify):

| Operation | SASS | Per SM per cycle | Relative to MUFU |
|---|---|---|---|
| `ex2.approx.ftz.f32` | `MUFU.EX2` | 16.0 | 1× |
| `fma.rn.f32` | `FFMA` | 128 (125–128) | 8× |
| `fma.rn.f32x2` (2 FMAs per instruction) | `FFMA2` | 128 FMAs (= 64 instructions) | 8× |
| `cvt.rmi.f32.f32` (floor) | `FRND.FLOOR` | 16 | 1× |
| Software `exp2`, scalar (§4.3) | About 9 instructions | 14.5 | 0.9× |
| Software `exp2`, packed (§4.3) | About 10 per pair | 17.4 | 1.09× |
| Mix: 25% polynomial | | 21.3 | 1.33× (still MUFU-bound: 16 / 0.75) |
| Mix: 50% polynomial | | 28.9 | 1.81× |
| Mix: 75% polynomial | | 20.8 | 1.3× (FMA-bound) |

Key points:

- **MUFU and FMA are two independent pipelines**: at a 50% mix the total throughput is close to the sum of the two (87% of 16 + 17.4 = 33.4);
- `FFMA2` does not raise the **arithmetic** throughput of FMA (still 128 FMAs per SM per cycle), but it **halves the instruction count**: each SMSP issues only one instruction per cycle, and the issue slots freed by packing can go to MUFU, `FMNMX` and `tcgen05.ld/st`;
- rounding (`FRND`) is as slow as MUFU, so range reduction uses the addition trick of §3.2.

### 4.2 Packed FP32 instructions: `FFMA2`, `FADD2`

SM100 adds instructions that process two FP32 values at once (PTX ISA 8.6+, from `sm_100`):

```
fma.rn.f32x2  d, a, b, c;     // d.x = a.x*b.x + c.x, d.y = a.y*b.y + c.y; a, b, c, d are .b64
add.rm.f32x2  d, a, b;        // two additions, rounding toward −∞
```

From CUDA 12.8, `<cuda_runtime.h>` provides matching intrinsics (`crt/sm_100_rt.h`) whose operands are `float2`:

```cpp
float2 __ffma2_rn(float2 a, float2 b, float2 c);   // FFMA2
float2 __fadd2_rn(float2 a, float2 b);             // FADD2
float2 __fadd2_rd(float2 a, float2 b);             // FADD2.RM (round down)
```

The two components must be in a pair of adjacent registers (written `R8.F32x2.HI_LO` in SASS); for consecutive array elements such as `make_float2(s[c], s[c+1])`, ptxas can usually allocate a register pair directly. A constant operand (such as a coefficient) can be an immediate, with both components equal.

### 4.3 Skeleton of the software `exp2`

Below is the scalar form for one element; the blanks are what TODO(1) asks you to fill in (the packed version replaces each operation with a `float2` intrinsic from §4.2 and computes two elements together):

```cpp
x = fmaxf(x, -127.f);                         // 1. clamp (FMNMX)
float r = __fadd_rd(x, 12582912.f);           // 2. magic-number add, round down (FADD.RM)
float n = r - 12582912.f;                     // 3. ⌊x⌋ as a float
float f = /* x − n, as one FMA */;            //    f ∈ [0, 1]
float p = /* degree-3 Horner, from c3 */;     // 4. 3 FMAs
return __int_as_float(__float_as_int(p) + /* how does the bit pattern of r become n << 23? */);   // 5.
```

Two details:

- write f as `fma(n, −1, x)` (packed: `__ffma2_rn(n, make_float2(-1.f, -1.f), x)`) rather than `x + (−n)`: the latter first negates each of the two components of n (two scalar instructions), adding 2 instructions per element pair;
- ptxas fuses the shift-and-add of step 5 into one `LEA` (`a + (b << 23)`).

After packing, each element pair takes: 2 `FMNMX` + 2 `FADD2` + 4 `FFMA2` + 2 `LEA` = 10 instructions, of which 6 are on the FMA pipeline (each packed instruction takes 2 cycles).

### 4.4 Warp vote: `__any_sync`

`tcgen05.ld`/`tcgen05.st` are `.sync.aligned` instructions: **all 32 lanes of a warp must execute them together** (the 32x32b shape reads or writes 32 lanes at once). So "does this row need an O correction" cannot be decided by each thread on its own, only per warp: if any one of the 32 rows needs a correction, the whole warp corrects together, and rows with α = 1 are multiplied by 1, which leaves the result unchanged (exact).

`__any_sync(mask, pred)` returns "is pred true for any lane in the warp"; all lanes get the same result, and it compiles to one `VOTE.ANY`. FA4's `correction_loop` uses the same pattern (`vote_ballot_sync(scale < 1.0)`).

### 4.5 Common misuses

| Misuse | Consequence |
|---|---|
| Use `floorf` / `__float2int_rd` to compute n | The result is right, but `FRND`/`F2I`, like MUFU, do only 16 per SM per cycle: step 1 gets slower |
| Use the default round-to-nearest for the magic-number add | r is the nearest integer and f ∈ [−0.5, 0.5]: the polynomial is outside its fit interval, the max relative error of `exp2` itself is about 1.05·10⁻² (at f = −0.5); averaged into LSE it is about 2–3·10⁻³, beyond tolerance |
| Write the constant term as the fitted 0.99998… | x = −∞ (mask) gives a negative number or NaN: cases with a mask such as `causal_*` and `short_tail` produce NaN |
| Forget `max(x, −127)` | For x = −∞, r = −∞ and the bit operations produce garbage: cases with a mask (such as `n1`) fail |
| Each thread decides on `rescale_o_row` from its own α | `tcgen05.ld` runs in a divergent warp: undefined behavior (hang or wrong result) |
| The condition compares with the previous block's max, not with the current reference | The reference falls further and further behind, and P exceeds the FP16 upper bound: `growing` produces inf/NaN |
| α is still computed as `2^((m − m_new)·…)` when the reference does not move | O is corrected but the reference did not move: O and ℓ disagree |

## 5. Schedule design

### 5.1 What stays the same

Warp roles, TMEM (S⁰/P⁰, S¹/P¹, O⁰, O¹, 512 columns), shared memory (Q⁰, Q¹, K/V ring, statistics), and all mbarriers with their parity are **exactly the same** as lab 40 step 3; see lab 40 §5. This lab changes only the few dozen lines that "compute P" in the softmax warpgroup, and one condition, "correct or not", in the correction warpgroup.

```
16 warps:
  warp 0–3   softmax WG 0 (tile 0)   setmaxnreg 192
  warp 4–7   softmax WG 1 (tile 1)   setmaxnreg 192
  warp 8–11  correction WG           setmaxnreg 80
  warp 12    MMA   warp 13 TMA   warp 14–15 idle   setmaxnreg 48
```

### 5.2 Instruction stream of one softmax block (steps 2–3)

```
wait s_full[t]; tcgen05.ld × 4 (one row of S, 128 FP32)
mask (only in blocks that need it)
row max: 127 FMNMX (8-way partial maxima)
m, α (step 3: thresholded test, see §3.5); α → shared; arrive stats_full[t]
for c = 0, 2, …, 126:                                  ← 64 iterations, fully unrolled
    x = FFMA2(s[c..c+1], scale_log2, −base)
    is pair (c % 32)/2 a "polynomial pair" (decided at compile time):
        yes: exp2_poly2(x)                  10 instructions, FMA/ALU
        no:  MUFU.EX2 × 2
    sum[(c/2) % 4] = FADD2(sum, p)
ℓ = α·ℓ + Σ
pack to 16 bits, tcgen05.st × 4, wait::st; fence; arrive p_ready[t]
```

Which pairs in every 32 columns use the polynomial (`use_poly<kPolyPairs>(q)`, q = 0..15) is decided at compile time, spread evenly over the 32 columns (not all bunched at one end). This interleaves the MUFU instructions and polynomial instructions in the instruction stream, so in every cycle the scheduler can find an instruction to issue to an idle pipeline. The solution's SASS (`./lab sass`): for steps 2–3, each kernel has 105 `MUFU.EX2` (104 for P + 1 for α) and 112 `FFMA2` (64 for x + 12 pairs × 4 for the polynomial: f and 3 Horner steps); step 1 has only 1 `MUFU.EX2` (α) and 320 `FFMA2`.

### 5.3 Correction warp (step 3)

```
for j, for t = 0, 1:
    wait stats_full[t] (parity j&1); fence
    j > 0 and __any_sync(α ≠ 1): rescale_o_row (ld / ×α / st / wait::st)
    fence; arrive p_ready[t]                 ← must arrive even when the correction is skipped
```

When the correction is skipped, the warp **still** has to arrive on `p_ready`: the MMA warp waits for 256 arrives (softmax 128 + correction 128), and one missing arrive means a hang. The synchronization argument is exactly the same as lab 40 §5.6: correction waits on `stats_full` before reading α, and arrives on `p_ready` after the correction (or after skipping it).

### 5.4 Registers

The softmax threads' limit is still 192 (`setmaxnreg`). A whole row of S (128 FP32) sits in registers, plus the polynomial coefficients, the magic number and the partial sums. Measured: when 4 or more of every 16 pairs use the polynomial, ptxas cannot fit in 192 registers, `./lab sass` shows a `STACK` of 8–16 bytes (spills), and performance drops clearly (§7.2). At k = 3 there are no spills.

## 6. Step-by-step exercises

In the skeleton, the pipeline (TMA, MMA, barriers, TMEM, the correction epilogue, the host side) is fully written and identical to the lab 40 solution. `exp_and_sum` already contains the packed loop; `p.step` selects `Config<D, 1|2|3>`. The three steps complete the same kernel one piece at a time, and later steps use the earlier TODOs.

### Step 1 · TODO(1): software `exp2`

Implement `exp2_poly2` (§3.2, §4.3). The coefficients `kExp2C0..3` are given (the degree-3 polynomial of §3.3). The step-1 configuration sends **all** exponentials of P through it.

```bash
./lab check 41 --step 1 --case short_tail     # has a mask: checks −∞ → 0
./lab check 41 --step 1 --case peaked_fp16    # scale = 1: many finite x < −127, checks the clamp
./lab check 41 --step 1
./lab bench 41 --step 1 --case perf           # expected to be slower than lab 40 (§7.1)
./lab sass 41     # in the full SASS under artifacts/sass/, the step-1 kernels (Config<D, 1>) have only 1 MUFU.EX2 (α)
```

### Step 2 · TODO(2): mixing

Implement `use_poly`: exactly `kPolyPairs` of the 16 pairs return true, spread evenly; then choose `kPolyPairs` in `Config` (start from the balance point in §3.6, tune with `bench`, and watch the `STACK` in `./lab sass`).

```bash
./lab check 41 --step 2
./lab bench 41 --step 2 --case perf
./lab bench 41 --step 2 --case perf_d64
./lab sass 41     # STACK must be 0; in the full SASS, count MUFU.EX2 and FFMA2 of Config<D, 2> (§5.2)
```

### Step 3 · TODO(3a)–(3b): conditional rescale

- 3a: the thresholded reference update in softmax (§3.5);
- 3b: the correction warp uses `__any_sync` to decide whether the whole warp corrects (§4.4, §5.3).

```bash
./lab check 41 --step 3 --case growing        # slow growth: compare with the reference, not the previous block
./lab check 41 --step 3 --case jump           # a jump: the reference must move
./lab check 41 --step 3
./lab bench 41 --case perf
./lab bench 41 --case perf_causal
./lab bench 41 --case perf_d64
./lab bench 41 --case perf_8k
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| Step 1: only cases with a mask produce NaN (`short_tail`, `causal_*`) | No clamp to −127, or the constant term is not exactly 1 (§3.2) |
| Step 1: O is right, LSE is out of tolerance (error about 2–3·10⁻³) | The magic-number add uses round-to-nearest (`__fadd2_rn`): f ∈ [−0.5, 0.5], the polynomial is outside its interval, and a single `exp2` error can reach 10⁻²; reversed coefficient order (Horner starts from the highest degree) also does this |
| Step 1: almost everything is wrong | The exponent-bit shift is not 23; the bit patterns of `r` and `p` are added with the wrong type (take the bits with `__float_as_int`, not an `(int)x` conversion) |
| Step 2 is correct but slower than step 1 | `use_poly` is not a compile-time constant (a branch remains in the loop); or `kPolyPairs` is too large and `STACK` > 0 |
| Step 3: `growing` produces inf/NaN, everything else is right | Comparing with the previous block's max instead of the current reference m |
| Step 3: `jump`, `jump_causal`, `growing` produce NaN | The reference never moves (condition inverted, or moves only in block 0); or α is wrong when the reference moves |
| Step 3 `TIMEOUT` | A warp that skips the correction does not arrive on `p_ready` |
| Step 3 hangs or fails intermittently | Deciding whether to call `rescale_o_row` per thread instead of per warp (`tcgen05.ld` must be executed by the whole warp) |
| Step 3: `growing`, `growing_causal`, `jump_causal` wrong but no NaN | After the vote, one lane's α is used to multiply all rows (each row should use its own α) |

## 7. Performance

### 7.1 Against the library baseline

Measured on B200 (CUDA Graph; `vs cudnn` = cuDNN time / time of this version, higher is better):

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| Lab 40 step 3 (all MUFU, reference) | ≈ 0.85× | ≈ 0.94× | ≈ 0.80× | ≈ 0.83× |
| Step 1 (all software `exp2`) | ≈ 0.58× | ≈ 0.69× | ≈ 0.61× | ≈ 0.56× |
| Step 2 (6 of every 32 columns polynomial) | ≈ 0.89× | ≈ 0.99× | ≈ 0.94× | ≈ 0.87× |
| Step 3 (+ conditional rescale, τ = 8) | ≈ 0.97× | ≈ 1.03× | ≈ 0.96× | ≈ 0.93× |
| FA4 (`flash_attn.cute`) | ≈ 1.15× | ≈ 1.33× | ≈ 1.13–1.15× | ≈ 1.03–1.10× |

cuDNN itself varies by a few percent between runs; the numbers in one row come from the same `bench` run.

- **Step 1 is about 30% slower than lab 40**: all exponentials move to the FMA pipeline (MUFU is nearly idle, `xu` 0.3%), each element pair takes 10 instructions, and the FMA/ALU pipeline and the issue slots become the new bottleneck. On its own, the software `exp2` is only 9% faster than MUFU (§4.1), and the other softmax instructions also need FMA.
- **Step 2** moves only about 19% of the exponentials, and MUFU and FMA work at the same time: this is the biggest step of this lab. The gain is larger for D = 64 because its softmax is relatively heavier (§3.1).
- **Step 3** almost entirely removes the O corrections after block 0 on random data (§3.5) and is another 2–9% faster. Control experiment: keeping only the warp vote with τ = 0 (skip only when the max does not change at all) gives `perf` ≈ 0.91× and `perf_d64` ≈ 0.94×, only a little better than step 2. On random data, the probability that **none** of a warp's 32 rows rises in the same block is low; only τ = 8 makes the corrections really disappear.

Target for this lab: step 3 reaches at least about 0.85× cuDNN on all four perf cases.

### 7.2 Sweeping the mix ratio

Step 2, with k polynomial pairs in every 32 columns (φ = k/16):

| k | φ | `perf` | `perf_d64` | STACK in `./lab sass` |
|---|---|---|---|---|
| 0 (all MUFU, but with `FFMA2`/`FADD2`) | 0 | ≈ 0.81× | ≈ 0.82× | 0 |
| 2 | 12.5% | ≈ 0.87× | ≈ 0.91× | 0 |
| **3 (solution)** | **18.75%** | **≈ 0.90×** | **≈ 0.93×** | **0** |
| 4 | 25% | ≈ 0.89× | ≈ 0.95× | 8 bytes in 4 kernels |
| 5 | 31% | ≈ 0.89× | ≈ 0.95× | 8 bytes in 6 kernels |
| 6 | 37.5% | ≈ 0.81× | ≈ 0.83× | 16 bytes |
| 8 | 50% | ≈ 0.72× | ≈ 0.74× | 24 bytes |
| 16 (step 1) | 100% | ≈ 0.58× | ≈ 0.61× | 0 |

- k = 0 is "lab 40 + packed `FFMA2`/`FADD2`": it was not measured in the same bench run as lab 40, so we can only say the two are about equal; the bottleneck is still MUFU;
- the rows k = 3, 4, 5 were measured with the solution's final code (f with one `FFMA2`, §4.3); the other rows were measured with an earlier version (f = x + (−n), 2 more scalar negation instructions per pair), in which k = 4 reached only 0.82×/0.88× and spilled more. The comparison shows: saving two instructions per element pair is enough to keep a larger k from losing;
- in the final code the differences among k = 3–5 are within measurement noise; at k ≥ 4, `./lab sass` shows a `STACK` for some kernels (the softmax threads do not fit in 192 registers and spill to local memory), so the solution takes k = 3, which does not spill. **Always watch `STACK` when sweeping**;
- because spills start at k = 4, this sweep cannot show whether the model of §3.6 (φ ≈ 0.39) is right or wrong; it only shows that, under the 192-register limit, φ ≈ 19–31% are all about the same;
- each cell of this table was measured once; do not over-interpret differences of 1–2% or less between adjacent k.

### 7.3 Error analysis

Maximum error of each variant against the FP64 oracle (`max|err|` printed by `./lab check`, absolute error of O; same input):

| case | Lab 40 (all MUFU) | Step 1 (all polynomial) | Step 2 (mix) | Step 3 (mix + conditional rescale) |
|---|---|---|---|---|
| `medium` BF16 D=128 | 8.4·10⁻⁴ / 1.3·10⁻⁶ | 8.4·10⁻⁴ / 1.6·10⁻⁵ | 8.4·10⁻⁴ / 7.5·10⁻⁶ | 1.3·10⁻³ / 7.5·10⁻⁶ |
| `medium_d64` FP16 | 1.2·10⁻⁴ / 1.4·10⁻⁶ | 1.3·10⁻⁴ / 1.6·10⁻⁵ | 1.2·10⁻⁴ / 6.1·10⁻⁶ | 1.5·10⁻⁴ / 7.0·10⁻⁶ |
| `peaked` BF16 | 1.3·10⁻² / 1.6·10⁻⁵ | 1.3·10⁻² / 9.4·10⁻⁵ | 1.3·10⁻² / 9.4·10⁻⁵ | 1.5·10⁻² / 9.4·10⁻⁵ |
| `peaked_fp16` (scale = 1, LSE in the hundreds) | 1.4·10⁻³ / 1.9·10⁻⁴ | 1.4·10⁻³ / 2.4·10⁻⁴ | 1.4·10⁻³ / 2.4·10⁻⁴ | 2.0·10⁻³ / 2.4·10⁻⁴ |
| `causal_medium` BF16 | 8.0·10⁻³ / 1.2·10⁻⁶ | 8.0·10⁻³ / 8.6·10⁻⁵ | 8.0·10⁻³ / 3.0·10⁻⁵ | 8.0·10⁻³ / 3.0·10⁻⁵ |
| `growing` FP16 | 4.3·10⁻⁴ / 1.1·10⁻⁵ | 4.3·10⁻⁴ / 5.9·10⁻⁵ | 4.3·10⁻⁴ / 6.1·10⁻⁵ | 4.8·10⁻⁴ / 6.1·10⁻⁵ |
| `jump_causal` BF16 | 7.3·10⁻³ / 7.9·10⁻⁵ | 7.3·10⁻³ / 1.3·10⁻⁴ | 7.3·10⁻³ / 1.2·10⁻⁴ | 7.3·10⁻³ / 1.2·10⁻⁴ |
| `ones_v` FP16 | 2·10⁻¹⁵ / 1.1·10⁻⁶ | 2·10⁻¹⁵ / 1.9·10⁻⁵ | 2·10⁻¹⁵ / 8.4·10⁻⁶ | 2·10⁻¹⁵ / 8.4·10⁻⁶ |

Each cell is "max absolute error of O / max absolute error of LSE" (the four kernels run on the same input). Tolerances: O is FP16 `2e-3 + 5e-3·|ref|`, BF16 `1e-2 + 2e-2·|ref|`; LSE is `1e-3 + 1e-4·|ref|` (O of `peaked` is larger than 1, so 1.5·10⁻² is still within tolerance).

- **O**: the error is dominated by the rounding of O to 16 bits, and the four variants are nearly identical; O of `ones_v` is exactly 1: the numerator uses the 16-bit P and the denominator ℓ uses the FP32 p, and their difference (≤ 16-bit rounding) is absorbed by the rounding when O is written back as FP16;
- **LSE**: rises from about 10⁻⁶ with all MUFU to about 10⁻⁵–10⁻⁴, which is exactly the order of the polynomial's relative error 8.6·10⁻⁵ (LSE error ≈ ε, §3.4). With the mix, only 19% of the elements carry this error, so the average is smaller. It is still more than 10 times below the LSE tolerance 10⁻³;
- conditional rescale does not change the order of the error (§3.5: relative precision does not depend on magnitude);
- the tolerances are exactly the same as lab 40, not relaxed.

### 7.4 Which metrics to watch

```bash
./lab profile 41 --case perf
```

Nsight Compute (locked clocks, used only to compare metrics; `perf` case):

| Version | `gpu__time_duration` | Tensor Core busy (`sm__pipe_tensor_cycles_active`) | MUFU (`sm__inst_executed_pipe_xu`) | FMA pipeline (`pipe_fma`) | `smsp__inst_executed_pipe_xu.sum` |
|---|---|---|---|---|---|
| Lab 40 step 3 | 893 µs | 57% | 57% | — | 33.82 M |
| Step 1 | 1185 µs | 40% | 0.3% | 32% | 0.27 M (only α left) |
| Step 2 | 864 µs | 60% | 49% | 24% | 27.53 M (= 104/128 + α) |
| Step 3 | 812 µs | 65% | 53% | 18% | 27.53 M |

`perf_d64`: lab 40 1488 µs, Tensor Core 30%, MUFU 61%; step 3 1287 µs, Tensor Core 36%, MUFU 59%.

- `smsp__inst_executed_pipe_xu.sum` confirms the mix ratio: steps 2–3 are 81% of lab 40 (104 MUFU exponentials + 1 α per row, versus 128 + 1);
- the Tensor Core busy fraction of step 3 rises to about 65%, but MUFU is still busy only about half the time: neither pipeline is saturated, and the chain latency (lab 40 §3.1) is still the limit. FA4's further ways to shorten the chain (2-CTA, better scheduling) are in labs 42 and 43.

## 8. Self-check questions

1. At D = 64, what are the MUFU lower bound and the Tensor Core lower bound, in cycles, for one block (two tiles)? Why is the gain of this lab larger on `perf_d64` than on `perf`?
2. For x = −0.3, what is the bit pattern of `r = x + 1.5·2^23` (rounded down)? What are n and f? How much is finally added to the bit pattern of p?
3. Why must the constant term of the polynomial be exactly 1.0? With a fitted c0 = 0.99999, what is the bit pattern of the result for x = −∞?
4. A row has maxima (log₂ units) 0, 5, 9, 14, 30 over 5 consecutive blocks, with τ = 8. What are α and the reference in each block? In which block is the largest P of this row, and what is it?
5. If τ is raised to 20, what goes wrong for FP16 and for BF16?
6. Why must the correction warp still arrive on `p_ready` when it skips the correction? Why must "correct or not" be decided by the whole warp together?

<details>
<summary>Answers</summary>

1. MUFU: 32768 / 16 = 2048 cycles (independent of D); Tensor Core: PV + QK of two tiles = 4 × 128×128×64×2 FLOP = 8.4 MFLOP, / 8192 ≈ 1024 cycles. At D = 64 softmax takes a larger share, so moving some exponentials to FMA gains more.
2. x + 12582912 = 12582911.7, rounded down to 12582911 = 12582912 − 1, bit pattern `0x4B3FFFFF`; n = −1, f = 0.7. `bits(r) << 23` = `0xFF800000` (that is, −1 << 23); added to the bit pattern of p ≈ 1.6245, it gives 0.81225 = 2^−0.3.
3. x is clamped to −127, f = 0, p = c0, and we need `bits(c0) + (−127 << 23) = 0`, that is `bits(c0) = 0x3F800000`. The bit pattern of c0 = 0.99999 is `0x3F7FFF58`; adding `0xC0800000` gives `0xFFFFFF58`: a NaN.
4. Block 0: reference 0, α = 0; block 1: 5 − 0 = 5 ≤ 8, no move, α = 1; block 2: 9 > 8, reference 9, α = 2^−9; block 3: 14 − 9 = 5, no move, α = 1; block 4: 21 > 8, reference 30, α = 2^−21. The largest P is in block 3: 2^5 = 32 (block 1 also gives 32).
5. P can reach 2^20 > 65504: FP16 P overflows to inf, and O, ℓ become NaN or too small; BF16 does not overflow (upper bound 2^128), and the result is still right (but FA4's assertion `τ < log₂(dtype max)` fails for FP16).
6. The MMA warp waits for 256 arrives on `p_ready`; if correction misses an arrive, the count is never reached: a hang. `tcgen05.ld/st` are warp-level `.sync.aligned` instructions that all 32 lanes must execute together, so the decision can only be made per warp.
</details>

## 9. Further reading

- Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling* (2026): the sections on software exponentials (polynomial emulation of `exp2`) and conditional softmax rescaling.
- Upstream implementation (`third_party/flash-attention`, commit `1fc0221`):
  - `flash_attn/cute/utils.py`: `POLY_EX2` (coefficients for each degree), `ex2_emulation_2` (packed range reduction: `fmax(x, −127)`, `add_packed_f32x2(..., rnd="rm")`, the `shl 23` + `add` of `combine_int_frac_ex2`);
  - `flash_attn/cute/softmax.py`: `SoftmaxSm100.apply_exp2_convert` (`ex2_emu_freq`, `ex2_emu_res`, `ex2_emu_start_frg`: which of every 32 elements take the polynomial path), `update_row_max` (`rescale_threshold`);
  - `flash_attn/cute/flash_fwd_sm100.py`: `_TUNING_CONFIG` (`ex2_emu_freq` of each configuration), `rescale_threshold = 8.0` and Note [Low Precision Scaling], `vote_ballot_sync(scale < 1.0)` in `correction_loop`.
- PTX ISA: [ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2), [fma (including `.f32x2`)](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma), [add (including `.f32x2` and rounding modes)](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add), [vote.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).
- The arithmetic instruction throughput table in the CUDA C++ Best Practices Guide ([Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#arithmetic-instructions)); the numbers in §4.1 of this lab are measured, and the measurements take precedence.
- Polynomial approximation: J.-M. Muller, *Elementary Functions: Algorithms and Implementation* (range reduction, minimax approximation); Sollya's `fpminimax` (https://www.sollya.org/).
- Next: lab 42 (2-CTA attention).
