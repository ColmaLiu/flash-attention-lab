# Attention math: from three operators to streaming computation

This doc gathers the math scattered across [01](../course/01_softmax/README.md), [11](../course/11_naive_attention/README.md), [12](../course/12_fa1/README.md) and [22](../course/22_fa2/README.md). The lessons have fuller derivations and small examples worked by hand; this doc covers how they connect.

## 1. Treat one head as matrices

Fix batch b and head h; Q, K and V are each N×D matrices. The score of query i against key j is:

```
S[i,j] = scale · Σ_d Q[i,d] · K[j,d]
```

softmax normalizes **along the key direction j**:

```
P[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])
O[i,d] = Σ_j P[i,j] · V[j,d]
```

Different heads are independent. `[B,N,H,D]` is only the order in memory; it does not change the dimension being normalized (address computation is in [11 §3.3](../course/11_naive_attention/README.md)).

**A small example for checking**: N = 3, Q is all 0, and the three rows of V are `(1,0)`, `(0,2)`, `(3,4)`.

| | P of each row | O | LSE |
|---|---|---|---|
| Non-causal | All `(1/3, 1/3, 1/3)` | All `(4/3, 2)` | All `ln 3` |
| Causal | `(1,0,0)`, `(1/2,1/2,0)`, `(1/3,1/3,1/3)` | `(1,0)`, `(1/2,1)`, `(4/3,2)` | `0`, `ln 2`, `ln 3` |

This example checks the mask, the normalization direction and the addressing of V at the same time. If you also set V to 0, many bugs stay hidden.

## 2. Why subtract the maximum

If you compute `exp(S)` directly, one large score overflows to `inf`. Let `m = max_j S[j]` and multiply the numerator and the denominator by `exp(−m)`:

```
P[j] = exp(S[j] − m) / ℓ,   ℓ = Σ_k exp(S[k] − m)
```

At least one score equals m, and its exponential is 1; the other exponentials are in `[0, 1]`. So the denominator ℓ ≥ 1, and it cannot become 0 because everything underflows.

The output **LSE** (log-sum-exp) is the natural log of the normalization constant:

```
LSE = m + ln(ℓ)
```

Details are in [01 §3.1](../course/01_softmax/README.md).

## 3. Why the standard implementation needs N² space and traffic

The standard implementation ([11](../course/11_naive_attention/README.md)) has three steps and writes S and P to HBM:

```
S = scale · Q Kᵀ   →  write N×N
P = softmax(S)     →  read N×N, write N×N
O = P V            →  read N×N
```

Each head has N² scores. When N doubles, the space and traffic of S/P grow 4×, while Q, K, V and O have only N·D elements.

When N is much larger than D, reading and writing S/P makes up most of the HBM traffic. Whether it is the time bottleneck depends on how fast the computation is:

| Compute unit | Ridge point | Arithmetic intensity of S/P (D = 64, FP32 S/P) | Conclusion |
|---|---|---|---|
| FP32 CUDA core | About 9 FLOP/B | 16 FLOP/B | Compute-bound; the traffic takes only part of the time |
| BF16 Tensor Core | About 280 FLOP/B | 16 FLOP/B | Bandwidth-bound; the traffic sets the time |

So the faster the Tensor Core, the more it is worth removing the S/P traffic ([11 §3.5](../course/11_naive_attention/README.md), [B200 architecture overview §3.1](b200.md#31-the-ridge-point-compute-or-bandwidth)).

## 4. Three states are enough to merge any two key sets

The output needs only N×D. Can we process only part of the keys at a time and throw away their scores when done? The difficulty: a later key may bring a larger maximum, which changes the scale of the earlier exponentials.

For one query and a processed key set A, define three states:

```
m_A = max_{j∈A} S[j]
ℓ_A = Σ_{j∈A} exp(S[j] − m_A)
u_A = Σ_{j∈A} exp(S[j] − m_A) · V[j]      (a vector of length D)
```

Define the same for another, disjoint set B. To merge, take `m = max(m_A, m_B)` and the two conversion factors `a = exp(m_A − m)`, `b = exp(m_B − m)`:

```
ℓ = a·ℓ_A + b·ℓ_B
u = a·u_A + b·u_B
```

**Proof**: multiplying each term `exp(S[j] − m_A)` of A by a gives exactly `exp(S[j] − m)`; the same holds for B. So the merged state still fits the same definition, only the set is now A ∪ B.

After all keys are processed, `O = u / ℓ` and `LSE = m + ln ℓ`.

This is not an approximation: mathematically it is exactly equal to standard attention. A different order of floating-point operations brings different rounding errors, but it does not change the definition.

| Use | Lab |
|---|---|
| One softmax row, one element at a time: online softmax | [01 §3.3](../course/01_softmax/README.md) |
| One K/V block at a time: FA1 Algorithm 1, states stored in HBM | [12 §3.2–3.3](../course/12_fa1/README.md) |
| States stay in registers, divide by ℓ only at the end (deferred normalization) | [22 §3.1](../course/22_fa2/README.md) |
| O in TMEM, multiplied by a by the softmax or correction warps | [34 §3.2](../course/34_fa3/README.md), 40 |
| Rescale only when m grows past a threshold | [41](../course/41_fa4_softmax/README.md) |

**Two edge cases**:

- The initial empty set: `m = −∞`, `ℓ = 0`. Do not mechanically compute `exp(−∞ − (−∞))`; that is NaN.
- With causal, a K/V block may be fully masked for some queries. That block then contributes nothing to them; but a whole row with no valid key never happens, because the diagonal is always valid ([12 §3.7](../course/12_fa1/README.md)).

## 5. The math does not fix the work split on the GPU

The merge theorem does not say whether a row goes to one thread or one warp, nor whether Q, K and V sit in registers or shared memory. Different FA versions made different choices:

| Version | Outer loop | Where the states live | Main cost |
|---|---|---|---|
| FA1 ([12](../course/12_fa1/README.md)) | K/V blocks | O, ℓ, m read and written to HBM once per block | O makes a round trip for every K/V block |
| FA2 ([22](../course/22_fa2/README.md)) | Q blocks | O, ℓ, m in registers, written only once | Each Q block rereads all of K/V (mostly L2 hits) |
| FA3 ideas ([34](../course/34_fa3/README.md)) | Q blocks | S, O in TMEM | softmax and MMA must overlap |

If several warps process different key subsets of the same query, their states must be merged at the end with the rule from section 4. FA2 gives each warp different query rows, which avoids this merge.

## 6. IO complexity and measurement

The FA1 paper gives the HBM access count `Θ(N²d²/M)`, where M is the on-chip memory size ([12 §3.6](../course/12_fa1/README.md)). This shows that, once S/P is removed, memory traffic shrinks as on-chip memory grows. But it is a model; real traffic also depends on:

- Tile size and traversal order;
- Whether L2 hits (for example, in FA2 the K/V of one head is reread by many Q blocks, mostly from L2);
- Transaction granularity (32-byte sectors), spills, replays.

So when you analyze, write three things down separately:

| Content | How to get it |
|---|---|
| Algorithm model: how many times each tensor is read and written | Compute by hand from the source |
| Actual DRAM bytes | `dram__bytes_read.sum`, `dram__bytes_write.sum` from `./lab profile` ([11 §4.2](../course/11_naive_attention/README.md)) |
| Time | `./lab bench` |

Removing the N² storage does not guarantee a speedup. Too few CTAs, too much synchronization, register spills or a slow GEMM can all slow it down. Lab 12 sees this on FP32 SIMT: the extra HBM round trips of FA1 are not the bottleneck; the computation is ([12 §7.2](../course/12_fa1/README.md)).

## 7. How to tell whether a result is correct

**Math oracle**: `labkit` uses PyTorch in FP64 to compute S, softmax, PV and LSE directly from the definition. It does no tiling, uses no online softmax, and does not depend on the results of cuDNN or FA. Large inputs are computed in groups of query rows to limit GPU memory. Low-precision inputs are first converted to FP64, so the quantization error of the inputs themselves does not count as a kernel error.

**Criterion**: every element must satisfy

```
|actual − reference| ≤ atol + rtol · |reference|
```

NaN and Inf always count as wrong. On failure, the check reports the maximum error, the number of failures and the coordinates of the first failure.

**Tolerances** are defined per operator, dtype and reduction length in `tolerance()` in `labkit/kinds/<kind>.py`. Roughly:

| Output | Where the tolerance comes from |
|---|---|
| transpose | 0: there is no computation |
| GEMM | The rounding error of an FP32 dot product grows roughly with √K; the Tensor Core truncates the products of BF16/FP16, so the error is larger |
| attention O | Tightest for FP32; looser for FP16, then BF16, because P must be converted back to 16 bits before it goes into the Tensor Core |
| attention LSE | Set separately from O |

**Other checks**:

- Outputs and scratch are first filled with NaN (or garbage bytes), so elements that were never written are caught;
- Inputs are checked for being overwritten;
- The kernel runs once on the default stream and once on a non-default stream;
- Some steps do not check the final output; they check an intermediate result in `probe` instead (for example S).

**When something fails**: fix the case, look at the coordinates of the first failure, check the scale, the exponential base, the mask and the addressing, then shrink the case. **Do not pass by raising the tolerance.**

## 8. The base of the exponential

The hardware directly supports only `2^x` (`ex2.approx`, `MUFU.EX2` in SASS). Because `e^x = 2^(x·log₂e)`, let `scale_log2 = scale · log₂e`:

```
exp(scale·(s − m)) = 2^(s·scale_log2 − m·scale_log2)
```

Here s is a score before multiplying by scale, and m is their maximum. The subtraction and multiplication in the exponent fuse into one FFMA. This requires scale > 0: only then does `max(scale·s) = scale·max(s)` hold. The course's supported domain requires scale > 0 and finite ([22 §3.3](../course/22_fa2/README.md)).

You can switch bases internally, but **the output LSE must be a natural log**: `LSE = m·scale + ln ℓ`.

For the error bound of `ex2.approx` and when it is good enough, see [01 §4.4](../course/01_softmax/README.md).

## Further reading

- [FlashAttention paper](https://arxiv.org/abs/2205.14135v2): §2.2 GPU memory hierarchy, §3.1 Algorithm 1, §3.2 Theorem 2 (IO complexity);
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691v1): §3.1 reducing non-matmul operations, §3.2 parallel partitioning;
- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608v2): asynchronous producers, overlapping GEMM and softmax;
- [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451v1): pipelines, exponentials and rescale on Blackwell.
