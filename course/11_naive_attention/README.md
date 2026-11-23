# 11 · Standard attention and IO analysis

> Part 1 · FA1 | Prerequisites: [01_softmax](../01_softmax/README.md), [10_simt_gemm](../10_simt_gemm/README.md) | Code: [`students/11_naive_attention/kernel.cu`](../../students/11_naive_attention/kernel.cu)

## 1. Goals

Build the most direct attention from three kernels: first compute the score matrix S, then apply a row-wise softmax to get P, and finally compute O = P·V. S and P are both N×N, and both are written in full to HBM. This is the "standard attention" of the FlashAttention paper (Algorithm 0). After this lab you should be able to:

- write down the definition of attention, including the scale, the causal mask, the numerically stable softmax, and the natural-log LSE;
- address one head inside the `[B, N, H, D]` layout with strides, without copying or transposing;
- count the HBM bytes read and written by each kernel, and state how much larger the Θ(N²) traffic of S/P is than the Θ(N·D) traffic of Q/K/V/O;
- verify this analysis with the DRAM counters of `./lab profile`, and explain when the counters disagree with the estimate.

The matrix multiplications use a ready-made GEMM (the tiled SIMT GEMM from [10_simt_gemm](../10_simt_gemm/README.md); the skeleton provides it), and the softmax uses the online reduction from [01_softmax](../01_softmax/README.md). The new part in this lab is **combining them correctly and measuring the cost of the combination**.

## 2. Motivation: why FlashAttention is needed

Section 2.2 of the FlashAttention paper first gives the standard implementation:

```
Algorithm 0 (standard attention)
1. Read Q, K from HBM, compute S = Q Kᵀ, write S back to HBM
2. Read S from HBM, compute P = softmax(S), write P back to HBM
3. Read P, V from HBM, compute O = P V, write O back to HBM
```

The paper's argument: S and P have N² elements, while Q, K, V, O have only N·D each. When N is much larger than D, the HBM traffic of reading and writing S/P makes up almost all of the traffic, and it also determines the time. A more immediate problem is memory capacity: in this lab's `perf` case (B=1, H=16, N=4096), S alone takes 1 GiB; at N = 64K it takes 256 GiB, more than the HBM capacity of a B200 (about 179 GiB).

This lab first writes Algorithm 0, then measures how much of this argument holds on a B200 in FP32. [12_fa1](../12_fa1/README.md) then keeps S/P on chip.

## 3. Math

### 3.1 Definition

Fix one batch `b` and one head `h`. Q, K, V are all N×D matrices (the N rows are sequence positions, the D columns are features).

```
S[i][j] = scale · Σ_d Q[i][d] · K[j][d]                  (i, j = 0..N-1)
visible:  j ≤ i (causal)   or   all j (non-causal)
m_i   = max_{j visible} S[i][j]
l_i   = Σ_{j visible} exp(S[i][j] − m_i)
P[i][j] = exp(S[i][j] − m_i) / l_i     (j visible);   0   (j masked)
O[i][d] = Σ_j P[i][j] · V[j][d]
LSE[i]  = m_i + ln(l_i) = ln Σ_{j visible} exp(S[i][j])
```

- **scale** comes from `p.scale`. Its default is 1/√D, but a case can set another value (the `scale` and `large_logits` cases do).
- **The causal mask** lets query `i` see only keys `0..i`. The diagonal `j = i` is always visible, so every row has at least one visible key, `l_i ≥ 1`, and there is no division by 0.
- **The softmax normalizes along the key dimension j**, so each row of P sums to 1. Masked positions must be written as 0, because the GEMM in step 3 multiplies the whole row by V.
- **Subtracting the maximum** does not change P (numerator and denominator are both multiplied by `exp(−m_i)`), but it guarantees that the argument of the exponential is ≤ 0. FP32 `exp(x)` overflows to inf when x > 88.7, and in the `large_logits` case more than 90% of the rows have a maximum score above this value.
- **LSE uses the natural log** (`logf`, not `log2f`). The backward pass and FlashAttention's block merging both need it, so the public interface requires it as an output, with shape `[B, H, N]`.

### 3.2 A small example by hand

Take N = 3, D = 2, Q all 0, and the three rows of V are `(1,0)`, `(0,2)`, `(3,4)`. S is all 0, so every visible key gets the same weight:

| query i | non-causal: row of P | O | LSE | causal: row of P | O | LSE |
|---|---|---|---|---|---|---|
| 0 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (1, 0, 0) | (1, 0) | 0 |
| 1 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (½, ½, 0) | (½, 1) | ln 2 |
| 2 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 | (⅓, ⅓, ⅓) | (4/3, 2) | ln 3 |

The `uniform_causal` case is this example scaled up to N = 100: O of row i is the mean of the first i+1 rows of V, and LSE = ln(i+1).

### 3.3 Layout: finding one head in `[B, N, H, D]`

Q, K, V, O in the public interface are all contiguous `[B, N, H, D]`, with the last dimension fastest:

```
address of Q(b, i, h, d) = q + ((b·N + i)·H + h)·D + d
                         = q + b·(N·H·D) + h·D + i·(H·D) + d·1
                               └─batch─┘   └┬┘   └─row─┘   └┬┘
                                           head            col
```

With (b, h) fixed, one head is an N×D matrix: two adjacent rows (sequence positions i and i+1) are **H·D** floats apart, because the same position of the other heads sits between them. A common mistake is to treat the layout as `[B, H, N, D]` and use D as the row stride. When H = 1 the two are the same, so this mistake only shows up in cases with H > 1 (`bnhd`).

The GEMM in the skeleton describes an operand with 4 strides:

```cpp
struct Strides { int64_t b, h, row, col; };
// element (r, c) of matrix (b, h) is at  x[b*s.b + h*s.h + r*s.row + c*s.col]
// gemm(a, sa, b, sb, c, sc, M, N, K, batch, heads, alpha, stream) computes
//   C(m, n) = alpha · Σ_k A(m, k) · B(n, k)        for each (b, h)
```

S and P use an intermediate layout that this lab chooses, `[B, H, N, N]`: one dense N×N matrix per head. Row `row = (b·H + h)·N + i` of S is also exactly the index of `lse[b, h, i]`.

QKᵀ has exactly the GEMM's "A·Bᵀ" form: A = Q (m = i, k = d), B = K (n = j, k = d), and both are K-major (contiguous along k). PV is different: `O(i, d) = Σ_j P(i, j) V(j, d)`, and the GEMM's B(n, k) is V(k, n), that is, V read as (d, j). In memory V is contiguous along d (MN-major). **V does not need a transpose**: a different set of strides is enough, and the GEMM's loader coalesces accesses along the direction with stride 1.

### 3.4 HBM traffic: counting kernel by kernel

Let BH = B·H, with 4 bytes per element (FP32). Assume every kernel reads its inputs from HBM and writes its outputs to HBM, and all reuse happens on chip:

| kernel | read | write | order |
|---|---|---|---|
| ① S = scale·QKᵀ | Q, K: 2·BH·N·D·4 | S: BH·N²·4 | Θ(N²) write |
| ② softmax | S: BH·N²·4 | P: BH·N²·4, LSE: BH·N·4 | Θ(N²) read and write |
| ③ O = PV | P: BH·N²·4, V: BH·N·D·4 | O: BH·N·D·4 | Θ(N²) read |
| total | | | 4·BH·N²·4 + 4·BH·N·D·4 |

The ratio of S/P traffic to Q/K/V/O traffic is **N/D**. For the `perf` case (BH = 16, N = 4096, D = 64):

```
S or P, once:   16 × 4096² × 4 B = 1.07 GB         4 times → 4.29 GB
Q/K/V/O, once:  16 × 4096 × 64 × 4 B = 16.8 MB     4 times → 0.067 GB
```

The S/P traffic is 64 times the Q/K/V/O traffic. A fused kernel (for example the FlashAttention of lab 12) in principle needs only the second number, 0.067 GB.

### 3.5 Compute or bandwidth: arithmetic intensity

The two GEMMs do `4·BH·N²·D` floating-point operations in total (a multiply-add counts as 2), which is 68.7 GFLOP for the `perf` case. Divide by the S/P traffic:

```
arithmetic intensity = 4·N²·D / (4·N²·4 B) = D/4 FLOP/B = 16 FLOP/B   (D = 64, FP32 S/P)
```

Compare it with the "ridge point" of the B200 (peak compute / peak bandwidth):

| compute unit | nominal peak | ÷ 8 TB/s | 16 FLOP/B falls |
|---|---|---|---|
| FP32 CUDA core | 75 TFLOP/s | ≈ 9 FLOP/B | right of the ridge point: **compute-bound** |
| BF16 Tensor Core | 2250 TFLOP/s | ≈ 280 FLOP/B | far to the left: **bandwidth-bound** |

So the two cases need separate answers. In this lab's FP32 SIMT implementation, the GEMM FMAs themselves are slow, and the HBM traffic of S/P is only part of the time. Once the GEMMs move to Tensor Cores (Parts 2–4), the same 4 GB of traffic dominates. The FlashAttention paper uses FP16 Tensor Cores on A100, which is the second case. Section 7 verifies both points with measurements.

## 4. Hardware and instructions

This lab has no new instructions. What is new is "how to observe the memory hierarchy".

### 4.1 B200 memory hierarchy

| level | capacity | role in this lab |
|---|---|---|
| registers / shared memory | 64K registers and 228 KB smem per SM | GEMM tiles and accumulators; softmax in-block reduction |
| L2 | 126 MB (whole chip) | at N = 1024, all of S (64 MiB) fits |
| HBM3e | about 179 GiB, nominal 8 TB/s | at N = 4096, S (1 GiB) only fits here |

### 4.2 DRAM counters

`./lab profile` prints two Nsight Compute metrics for each kernel:

- `dram__bytes_read.sum`: the bytes read from HBM into L2 while this kernel runs;
- `dram__bytes_write.sum`: the bytes written back from L2 to HBM.

They count the bytes that **actually cross the HBM interface**, not the load/store bytes the kernel issues. The two differ in two situations:

1. L2 hits: when several blocks read the same K tile, only the first read comes from HBM;
2. L2 is write-back: dirty data still in L2 when the kernel ends **is not counted for this kernel**. It is written back later when evicted (or the next kernel reads it directly, and it never goes to HBM at all).

So when the intermediate result is smaller than L2, the counters show clearly less than the estimate in Section 3.4. The `n1k` case in Section 7 shows this.

When `./lab profile` runs, Nsight Compute locks the clocks, so `gpu__time_duration` is slower than in `bench`. Use it only to compare ratios between kernels and byte counts.

### 4.3 64-bit indices

S in the `perf` case has 16 × 4096² ≈ 270 million elements, only a factor of 8 below 2³¹. A slightly larger B·H or N overflows an `int`. `run()` converts n, h, d to `int64_t` before multiplying, and the fields of `Strides` are all `int64_t`.

## 5. Schedule design

The three kernels run in order on one stream. Each passes its output to the next through HBM (or L2):

```
  HBM:  Q,K ──► [① GEMM] ──► S ──► [② softmax] ──► P (overwrites S) ──► [③ GEMM] ──► O
                                         │                                  ▲
                                         └──► LSE                    V ─────┘
```

| kernel | grid (`perf` case) | block | what each block does | smem |
|---|---|---|---|---|
| ① `gemm_kernel` | (⌈N/64⌉, ⌈N/64⌉, B·H) = (64, 64, 16) | 256 | one 64×64 tile of C, 4×4 per thread | 2 × 16 × 65 × 4 B ≈ 8.1 KB |
| ② `softmax_rows` | B·H·N = 65536 | 128 | one row of S: online (m, l), then write P back | a few dozen bytes (per-warp partial results) |
| ③ `gemm_kernel` | (⌈D/64⌉, ⌈N/64⌉, B·H) = (1, 64, 16) | 256 | one 64×64 tile of O, K loop of length N | same as ① |

Some design notes:

- **P overwrites S in place.** Each softmax block reads and writes only its own row. It writes (pass 2) only after it has read the whole row (pass 1), with the `__syncthreads()` inside `block_merge` in between, so the in-place write is safe. This way only one N×N scratch buffer is needed.
- **The scratch buffer is allocated in `make_plan`** (`lab::Scratch`, not timed, initialized to NaN). `run()` only launches kernels.
- **Steps 1–2 write S/P into `p.probe`.** The probe is also `[B, H, N, N]`. The harness compares it with an FP64 oracle, so S and P can be checked before O is computed.
- **Causal saves no work.** Both GEMMs still compute the full N×N; the softmax writes zeros into the upper triangle, and PV still multiplies those zeros. Library implementations compute only the blocks near and below the diagonal. This is why the gap is larger for `perf_causal` in Section 7, and it is the topic of lab 43_fa4_schedule.

## 6. Step-by-step exercises

Run all commands from the repository root. The skeleton already contains the GEMM, the reduction helpers, the launch configuration and `make_plan`. You write the 3 TODOs.

### Step 1 · TODO(1): S = scale·QKᵀ

In `run()`, write the `Strides` of Q/K (one head is an N×D matrix) and the `Strides` of S (`[B, H, N, N]`), then call `gemm()` once. Pass scale as `alpha`.

```bash
./lab check 11 --step 1
```

This step checks S in the probe, with coordinates shown as `[b, h, i, j]`. S has no mask: the upper triangle of causal cases must also hold the correct scores.

### Step 2 · TODO(2): masked row softmax and LSE

In `softmax_rows`: compute the query position `i` and the visible length `len` from `blockIdx.x`. In pass 1, fold the visible scores into `State {m, l}` and combine them with `block_merge()`. In pass 2, write P in place (0 at masked positions). Finally, one thread writes `lse[row]`.

```bash
./lab check 11 --step 2
```

This step checks P in the probe. LSE is checked only in step 3, but write it correctly now.

### Step 3 · TODO(3): O = P·V

Write the `Strides` of P, V (read as (d, j)) and O, and call `gemm()` with `alpha = 1`.

```bash
./lab check 11
./lab bench 11
./lab bench 11 --case perf_causal
./lab profile 11
```

`./lab bench 11 --step 1` and `--step 2` time only the kernels completed up to that step, while TFLOP/s is still computed from the FLOPs of the full attention and `vs` is still the time ratio against the full-attention library baseline. So the TFLOP/s and `vs` printed by bench for these two steps describe an unfinished kernel and mean nothing; look only at the time itself. `./lab profile 11 --step 1` or `--step 2` lists the time and bytes of each kernel.

### Common mistakes

| symptom | likely cause |
|---|---|
| in step 1 only `bnhd` and later cases fail | `[B,N,H,D]` was treated as `[B,H,N,D]`: the row stride should be H·D and the head offset D |
| in step 1 the `scale` case fails and all others pass | scale is hard-coded as 1/√D or 1 instead of using `p.scale` |
| in step 2 the probe of causal cases is not 0 in the upper triangle | masked positions were not written; P overwrites S in place, so skipping them leaves the values of S |
| NaN/inf in step 2 `large_logits` | the maximum was not subtracted: `exp` overflows when its argument exceeds 88.7 |
| in step 2 only causal cases fail | off-by-one in the visible length: query i sees `0..i`, which is i+1 keys |
| in step 3 O is right but LSE is too large | `log2f` was used: log₂ x = 1.443·ln x |
| in step 3 LSE is right but O is wrong | the `Strides` of V: n of the GEMM corresponds to d and k to j; work out their strides in `[B,N,H,D]` one by one; then check M, N, K again |
| `S would need more than 32 GiB` | this is exactly the problem this lab demonstrates: S does not fit |
| `illegal memory access` | stride product overflow (use `int64_t`), or the batch strides of B and H are swapped; locate it with `./lab sanitize 11 --case bnhd` |

## 7. Performance

### 7.1 Time and library baseline

Measured (CUDA Graph timing; `vs efficient` is the time of PyTorch memory-efficient attention divided by the time of this implementation):

| case | shape [B, N, H, D] | vs efficient |
|---|---|---|
| `perf` | [1, 4096, 16, 64] | ≈ 0.32× |
| `perf_causal` | [1, 4096, 16, 64], causal | ≈ 0.21× |

efficient is a fused kernel (it does not write S/P), and its source shows that its FP32 version emulates FP32 with 3xTF32 on Tensor Cores (`OpMultiplyAddFastF32`; see Section 9 for the source location). So the gap here comes from two things at once: "4 GB less traffic" and "GEMMs on Tensor Cores". The goal of this lab is only to understand the cost; reaching 0.15× or better is enough.

### 7.2 Where the time goes

Time share and DRAM bytes of each kernel from `./lab profile 11` (`perf` case):

| kernel | time share | DRAM read | DRAM write | Section 3.4 estimate (read / write) |
|---|---|---|---|---|
| ① QKᵀ | ≈ 41% | 0.034 GB | 1.02 GB | 0.034 / 1.07 GB |
| ② softmax | ≈ 9% | 1.07 GB | 1.01 GB | 1.07 / 1.07 GB |
| ③ PV | ≈ 50% | 1.10 GB | 0.019 GB | 1.09 / 0.017 GB |
| total | | ≈ 4.25 GB | | 4.36 GB |

- The measured bytes match the estimate: almost all traffic is S/P. The writes are slightly below 1.07 GB because a few dozen MB are still in L2 when the kernel ends (Section 4.2).
- The softmax only reads and writes. It is a purely bandwidth-bound kernel, and its `dram__throughput` is the highest of the three.
- The two GEMMs take 90% of the time, with high `sm__throughput` and low `dram__throughput`: they are **compute-bound**, which agrees with Section 3.5 ("FP32 SIMT is right of the ridge point"). PV does the same amount of computation as QKᵀ but is slower. One possible reason is that its grid has only 1024 blocks (D = 64 fills only one 64-column tile): `sm__warps_active` is about 54%, compared with about 74% for QKᵀ.

For comparison, Nsight Compute measures about 0.05 GB of DRAM reads for the efficient kernel (same shape), which is roughly one read each of Q, K and V.

### 7.3 Time as N changes

Same B=1, H=16, D=64, changing only N (`./lab bench 11 --case n1k`, `--case n2k`, `--case perf`):

| N | size of S | time relative to the previous row | does S fit in L2 |
|---|---|---|---|
| 1024 (`n1k`) | 64 MiB | — | yes |
| 2048 (`n2k`) | 256 MiB | ≈ 3.1× | no |
| 4096 (`perf`) | 1 GiB | ≈ 3.8× | no |

Doubling N makes the time close to 4 times longer: both the computation and the S/P traffic are Θ(N²). From 1024 to 2048 the factor is below 4. One possible reason is that at N = 1024 the grid is small and does not fill the GPU (unverified).

Run `./lab profile 11 --case n1k` on `n1k`: kernel ① writes only about 15 MB of its 64 MiB to HBM; the rest stays in L2 and the softmax reads it from there. When the intermediate result is smaller than L2, L2 hides part of the cost of "write S, then read S"; once N grows, the whole cost lands on HBM.

### 7.4 What to conclude

1. The HBM traffic of standard attention is Θ(BH·N²), N/D times larger than that of Q/K/V/O, and the measurements agree with the estimate.
2. In FP32 SIMT this traffic is only a small part of the total time: at the nominal 8 TB/s, 4.25 GB takes at least about 11% of the total time, and the GEMM FMAs are the bulk. On Tensor Cores the traffic becomes dominant (Section 3.5).
3. The harder limit is memory capacity: the size of S grows with N².

FlashAttention solves both 1 and 3: S and P exist only on chip, and the HBM traffic returns to the order of Θ(N·D) (plus the repeated K/V reads caused by tiling).

## 8. Self-check questions

1. If the layout of Q changes to `[B, H, N, D]`, what should the `Strides{b, h, row, col}` of one head be?
2. B = 2, N = 1000, H = 4, D = 64, FP32. How many bytes are the S/P traffic and the Q/K/V/O traffic of the standard implementation?
3. With causal, why must `softmax_rows` write 0 at positions `j > i` instead of skipping them?
4. If the softmax computed only the max in pass 1, the sum in pass 2, and wrote P in pass 3 (three passes), would the HBM read traffic triple? How do you verify it with `./lab profile`?
5. If this lab's GEMMs moved to BF16 Tensor Cores (2250 TFLOP/s) and S/P were stored in BF16, about what would the lower bounds of the compute time and the S/P traffic time be for the `perf` case?
6. Why is the DRAM write byte count of the `perf` case slightly smaller than the size of S, while for `n1k` it is much smaller?

<details>
<summary>Answers</summary>

1. Element (b, h, i, d) is at `((b·H + h)·N + i)·D + d`, so `{H·N·D, N·D, D, 1}`.
2. S/P: 4 × 2·4·1000²·4 B = 128 MB; Q/K/V/O: 4 × 2·1000·4·64·4 B ≈ 8.2 MB; the ratio is N/D ≈ 15.6.
3. The GEMM in step 3 sums over the whole row j = 0..N−1. P and S share one buffer; without the write, the upper triangle keeps the values of S (or the scratch NaN), and they get multiplied into O.
4. Usually not. A row is at most 16 KB, and the same block reads it again right away, so the reads mostly hit L1/L2 and HBM is read only once. Use `./lab profile` to check whether `dram__bytes_read.sum` of `softmax_rows` is still about the size of S.
5. Compute: 68.7 GFLOP / 2250 TFLOP/s ≈ 0.03 ms. Traffic: 4 × 16·4096²·2 B ≈ 2.1 GB, / 8 TB/s ≈ 0.27 ms. The traffic time is about 9 times the compute time: fully bandwidth-bound.
6. L2 is write-back: dirty lines still in L2 when the kernel ends are not counted for this kernel. In `perf`, S (1 GiB) is far larger than L2, and only the last few dozen MB stay behind. In `n1k`, S is only 64 MiB; most of it stays in L2 and the next kernel reads it directly.

</details>

## 9. Further reading

- The FlashAttention paper ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135v2)): Section 2.2 and Algorithm 0 in the appendix (standard attention), Section 3.2 (IO complexity analysis, Theorem 2).
- Rabe & Staats, *Self-attention Does Not Need O(n²) Memory* ([arXiv:2112.05682](https://arxiv.org/abs/2112.05682)): why memory does not have to grow with N².
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model* (CACM 2009): the origin of the ridge point in Section 3.5.
- Nsight Compute: [Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-workload-analysis), the meaning of the DRAM and L2 metrics.
- PyTorch memory-efficient attention (xFormers / CUTLASS FMHA): `third_party/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h`; the place where FP32 selects Tensor Core 3xTF32 is `DefaultGemmType<ArchTag, float>` in `gemm_kernel_utils.h` in the same directory.
