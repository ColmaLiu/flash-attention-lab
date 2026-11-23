# 01 · Reduction and numerical stability: row softmax

> Part 0 · Preliminaries | Prerequisites: [00_transpose](../00_transpose/README.md) | Code: [`students/01_softmax/kernel.cu`](../../students/01_softmax/kernel.cu)

## 1. Goals

Apply softmax to each row of an FP32 matrix: `y[r, :] = softmax(x[r, :])`. Softmax needs the maximum and the sum of exponentials of the whole row first, so it is a **reduction** problem: many threads each compute a part, and then the parts are combined. This lab builds four versions and takes softmax from "about ten times slower than torch on short rows" to faster than torch on all three perf cases. When you finish, you should be able to:

- write block-level and warp-level reductions with shared memory and `__shfl_xor_sync`, and state what each `__syncthreads()` protects;
- explain why softmax subtracts the maximum, and when the precision of `ex2.approx` is good enough and when it is not;
- derive the **online softmax** state `(m, l)` and its merge rule, and prove that the merge is associative;
- keep a whole row in registers, access memory with `float4`, and choose "one warp per row" or "one block per row" by row width.

Online softmax is the single most important formula in the whole course. Starting with [12_fa1](../12_fa1/README.md), every attention kernel uses it.

## 2. Motivation: why softmax blocks tiling

Attention is `O = softmax(S) V`, where each row of `S = Q Kᵀ` has N elements. The standard implementation writes the whole `S` (N × N) back to HBM, reads it back to compute softmax, writes `P`, then reads `P` to compute `P V`. At N = 8192, the FP32 `S` of one head alone is 256 MB.

FlashAttention wants to split K and V into small tiles, process one tile at a time on chip, and never write `S` back to HBM. The obstacle is softmax: the denominator `Σ_j e^{s_j}` needs the **whole row**, but while processing the first tile you have not yet seen the later tiles. The answer in Section 3.1 of the FlashAttention paper: keep two numbers per row (the maximum `m` and the sum of exponentials `l`), and each time a new tile arrives, **rescale** the old result to the new maximum. This trick comes from the 2018 paper by Milakov and Gimelshein, *Online normalizer calculation for softmax*.

This lab first works this idea out fully in the simplest setting, plain row softmax:

| Setting | Where a row's data comes from | What online softmax does |
|---|---|---|
| step 3 of this lab | one thread scans x along the row | gets `(m, l)` in one pass, reading x one fewer time |
| the reductions in steps 2–4 of this lab | many threads each hold a segment | merges the segments' `(m, l)` in any order |
| FlashAttention | K/V tiles produce `S` one segment at a time | no need to keep the whole row of `S` |

## 3. Math

### 3.1 Definition and numerical stability

For a row `x_0, …, x_{n-1}`:

```
y_j = e^{x_j} / Σ_k e^{x_k}
```

The largest number FP32 can represent is about 3.4 × 10³⁸ ≈ e^{88.7}. In the `large` case `|x|` reaches several thousand, so computing `e^{x}` directly gives `inf`, and `inf / inf = NaN`.

Softmax does not change when you add the same constant to the whole row: `e^{x_j - c} / Σ_k e^{x_k - c}` equals the original expression. Take `c = m = max_k x_k`. Then every exponent `x_j - m ≤ 0`, `e^{x_j - m} ∈ [0, 1]`, and the denominator `l = Σ_k e^{x_k - m} ∈ [1, n]` (the largest term is exactly 1). This is called **safe softmax**:

```
m = max_k x_k
l = Σ_k e^{x_k - m}
y_j = e^{x_j - m} / l
```

Elements equal to `-inf` (the `masked` case, like masked-out positions in attention) contribute `e^{-inf} = 0`. Each row has at least one finite value, so `m` is finite and `l ≥ 1`.

### 3.2 The three-pass algorithm and memory traffic

Writing the formulas above directly, each row needs three scans:

| Pass | Reads | Writes | Produces |
|---|---|---|---|
| 1 | x | — | m |
| 2 | x | — | l |
| 3 | x | y | y |

If every pass reads from HBM, the total traffic is `3 × read + 1 × write`, twice the lower bound (one read and one write). Whether passes 2 and 3 hit in cache depends on whether "the rows being processed at the same time" fit in L2; Section 7 measures this.

### 3.3 Online softmax: (m, l) in one pass

Define the **state** of a set of elements `S`:

```
state(S) = (m_S, l_S),   m_S = max_{j∈S} x_j,   l_S = Σ_{j∈S} e^{x_j - m_S}
```

The state of the empty set is defined as `(-inf, 0)`, and the state of a set with one element `x` is `(x, e^{x-x}) = (x, 1)`.

**Merge rule.** Let `A` and `B` be disjoint, and `m = max(m_A, m_B)`. Rewrite each term of `A` relative to `m`:

```
e^{x_j - m} = e^{x_j - m_A} · e^{m_A - m}
```

Summing over `j ∈ A` gives `l_A · e^{m_A - m}`, and likewise for `B`. Therefore

```
merge((m_A, l_A), (m_B, l_B)) = ( m,  l_A · e^{m_A - m} + l_B · e^{m_B - m} ),   m = max(m_A, m_B)
```

The **per-element update** is a merge with the one-element set `(x, 1)`:

```
m_new = max(m, x)
l_new = l · e^{m - m_new} + e^{x - m_new}
```

`e^{m - m_new}` is "how much the old sum must shrink": when the new element is larger than the old maximum, every old term must get smaller relative to the new reference; otherwise this factor equals 1 and `l` just gains the new term.

**Why merging in any order works.** The derivation above shows that `merge(state(A), state(B)) = state(A ∪ B)`: the result of a merge depends only on the set itself, not on how the set was split. Therefore

```
merge(merge(a, b), c) = state(A ∪ B ∪ C) = merge(a, merge(b, c))     (associativity)
merge(a, b) = state(A ∪ B) = merge(b, a)                             (commutativity)
merge((-inf, 0), a) = a                                              (identity)
```

Associativity and commutativity let us merge along a tree of any shape: each thread first scans its own elements sequentially, then merges within the warp with a butterfly, then merges within the block. This is the same reason `+` and `max` support parallel reduction. (In floating point, as with ordinary summation, associativity holds only approximately: different merge orders round differently, and the difference is within a few ulp.)

**The `-inf` trap.** If `m_A = m_B = -inf` (two empty sets, or both made of masked elements), then `m = -inf` and `m_A - m = (-inf) - (-inf) = NaN`. So `merge` must handle this case separately: when the new `m` is `-inf`, return `(-inf, 0)` directly. When `m_A = -inf` and `m` is finite, `e^{-inf} = 0`, and no special handling is needed.

**The last step** is the same as in the three-pass algorithm: `y_j = e^{x_j - m} / l`. The online version only fuses the first two passes into one: it reads x twice instead of three times.

### 3.4 A small worked example

A row `x = [2, 1, 4, 3]`. Sequential scan:

| x read | m_new | l_new = l · e^{m - m_new} + e^{x - m_new} |
|---|---|---|
| (initial) | -inf | 0 |
| 2 | 2 | 0 · 0 + e⁰ = 1 |
| 1 | 2 | 1 · 1 + e⁻¹ = 1.367879 |
| 4 | 4 | 1.367879 · e⁻² + e⁰ = 0.185122 + 1 = 1.185122 |
| 3 | 4 | 1.185122 · 1 + e⁻¹ = 1.553002 |

Check: `e⁻² + e⁻³ + e⁰ + e⁻¹ = 0.135335 + 0.049787 + 1 + 0.367879 = 1.553002`.

A different split: two threads process `A = [2, 1]` and `B = [4, 3]` and get `(2, 1.367879)` and `(4, 1.367879)`. Merge:

```
m = max(2, 4) = 4
l = 1.367879 · e^{2-4} + 1.367879 · e^{4-4} = 0.185122 + 1.367879 ≈ 1.553002
```

The result is the same (the numbers in the table are rounded to 6 decimal places, so the last digit may differ by 1). The output is `y = e^{x - 4} / 1.553002 = [0.087144, 0.032059, 0.643914, 0.236883]`.

### 3.5 Deferred scaling: the same trick in step 4 and FlashAttention

In step 4 each thread holds several elements. It first uses **its own** maximum `m_t` to compute `v_j = e^{x_j - m_t}` and `l_t = Σ v_j`, then merges all threads' `(m_t, l_t)` into the whole row's `(M, L)`. The output does not need to recompute the exponentials:

```
y_j = e^{x_j - M} / L = v_j · ( e^{m_t - M} / L )
```

The parenthesized factor is one scalar per thread. This way each element computes only one exponential, and the whole row does only one reduction (instead of two, one for max and one for sum). FlashAttention does the same thing to the output accumulator: `O ← O · e^{m_old - m_new} + P V`. It accumulates relative to the old maximum, and when the maximum changes it multiplies everything by one scaling factor.

## 4. Hardware and instructions

### 4.1 Block reduction: a shared memory tree

256 threads each hold one value `v`, and we want their maximum (or sum):

```
smem:  v0 v1 v2 v3 v4 v5 v6 v7        (example with 8 threads)
s=4:   t<4 does smem[t] = op(smem[t], smem[t+4])
       a0 a1 a2 a3 .. .. .. ..
s=2:   b0 b1 .. ..
s=1:   c0                             all threads read smem[0]
```

You need `__syncthreads()` after writing to `smem` and after every level: the next level reads results that other threads just wrote. 256 threads take log₂256 = 8 levels, plus one after the initial write, for 9 in total. After reading `smem[0]` you need one more synchronization: the same `smem` is reused right away by the next reduction, and without the barrier a fast thread could overwrite `smem[0]` before a slow thread has read it.

### 4.2 Warp shuffle

The 32 threads of one warp can read each other's registers directly, without going through shared memory:

```cpp
float other = __shfl_xor_sync(0xffffffff, v, offset);  // read v from lane (lane ^ offset)
```

- The first argument is the mask of participating lanes. `0xffffffff` means all 32 lanes must execute this instruction; if a lane does not reach it because of a branch, the result is undefined. So threads with no elements must also take part in the shuffle, carrying the identity element (`-inf` or 0).
- `offset` takes 16, 8, 4, 2, 1 in turn; in each round lane `i` exchanges with lane `i ^ offset` and merges. This is called a butterfly: after 5 rounds **every** lane holds the reduction of all 32 values.

```
offset 4:  0↔4  1↔5  2↔6  3↔7      (example with 8 lanes, 3 rounds)
offset 2:  0↔2  1↔3  4↔6  5↔7
offset 1:  0↔1  2↔3  4↔5  6↔7
```

A shuffle needs no `__syncthreads()`: this one instruction completes the data exchange within the warp. A block reduction therefore becomes "5 shuffle rounds within each warp → lane 0 of each warp writes its result into `smem[warp]` → one synchronization → reduce the 8 partial results with shuffles again", and the barrier count drops from 10 to 2.

To merge `(m, l)` states, shuffle each of the two fields once, then call `merge`.

### 4.3 Registers

| Resource | B200 per SM |
|---|---|
| register file | 65536 32-bit registers = 256 KB |
| per-thread maximum | 255 registers |
| threads | at most 2048 (64 warps), at most 1024 per block |
| shared memory | at most 228 KB |

How many blocks an SM can hold at once is limited by the tightest of three resources: registers, shared memory and threads. For example, with 50 registers per thread and 1024 threads per block, one block uses more than 51200 registers, so each SM can hold only 1 block.

For an array to stay in registers, every index must be known at compile time: loops must be fully unrolled with `#pragma unroll`, and indices cannot contain runtime variables. Otherwise the compiler puts the array in local memory (which actually lives in HBM/L1), and the `LOCAL` column of `./lab sass 01` becomes nonzero.

`__launch_bounds__(N)` tells the compiler that this kernel has at most N threads per block, and the compiler limits register use accordingly (at most 64 per thread for 1024 threads).

### 4.4 Exponentials: `ex2.approx` and precision

The GPU's special function unit (SFU, `MUFU` in SASS) directly supports base-2 exponentials, but its throughput is much lower than that of the FMA units (lab 41 works out this bottleneck in detail). Using `e^{t} = 2^{t · log₂e}`, one exponential needs only one `MUFU.EX2`:

```cpp
__device__ float ex2(float x) {           // provided by the skeleton
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ float exp_shifted(float x, float m) { return ex2((x - m) * kLog2e); }
```

- The PTX ISA says of `ex2.approx.f32`: "The maximum ulp error is 2 ulp from correctly rounded result across the full range of inputs". The relative error is at most about 2.4 × 10⁻⁷, and `ex2(-inf) = +0`.
- `.ftz` (flush to zero): when the result is below 2⁻¹²⁶ ≈ 1.2 × 10⁻³⁸, it returns 0 directly and skips subnormal handling.
- The tolerance of this lab is `|err| ≤ 1e-8 + 1e-4 · |expected|` (with FP64 `torch.softmax` as the reference); both of these errors are far below it.

The SASS that three ways of writing it compile to (with `x - m` already computed):

| Code | Instructions | Notes |
|---|---|---|
| `ex2((x - m) * kLog2e)` | `FADD`, `FMUL`, `MUFU.EX2` | used in this lab |
| `exp2f((x - m) * kLog2e)` | adds `FSETP` and 2 predicated `FMUL` | handles subnormal results correctly |
| `expf(x - m)` | about 8 `FFMA`/`FADD`/`SHF` + `MUFU.EX2` + `FMUL` | first splits the exponent into integer and fractional parts; maximum error 2 ulp |

`-use_fast_math` replaces `expf(t)` with `__expf(t)`, which is `ex2(t * log₂e)`. The CUDA documentation gives its error bound as `2 + ⌊1.173·|t|⌋` ulp: the larger the argument, the less accurate. After subtracting the maximum, every element that contributes to the result has `t = x - m` close to 0 (terms with `t < -100` are below e⁻¹⁰⁰ and can be ignored), so the error bound is small too.

**Subtract first, then multiply.** The `shifted` case adds 5000 to each row. Subtracting two close numbers, `x - m`, is exact (Sterbenz lemma: when `m/2 ≤ x ≤ 2m`, `x - m` has no rounding), and multiplying by `log₂e` adds only one rounding, a relative error of about 6 × 10⁻⁸. Much code (including the FlashAttention source) writes this as one FMA, `fmaf(x, log2e, -m * log2e)`, to save an instruction. In that form, `m · log₂e ≈ 7200` is first rounded to FP32. Call the rounding error `δ(m)`; `|δ(m)|` is at most half an ulp, which is 2⁻¹² ≈ 2.4 × 10⁻⁴. So the FMA form computes `e^{x-m} · 2^{-δ(m)}`: an extra factor that depends only on the shift `m`.

- **Steps 1 and 2**: the whole row uses one shift `M`. The numerator `e^{x-M}` and the denominator `Σ e^{x-M}` carry the same factor `2^{-δ(M)}`, which cancels exactly in the division. The FMA form also passes `shifted` in these two steps.
- **Steps 3 and 4**: the partial states each use their own shift `m_t` (in step 3, the running max that each thread updates element by element; in step 4, the thread's local maximum), so `l_t` carries `2^{-δ(m_t)}`. The factor `e^{m_t - M}` applied at merge time, also computed in FMA form, adds only `2^{-δ(M)}` and cannot cancel `2^{-δ(m_t)}`. Different `m_t` have different rounding errors, so the denominator becomes a sum with unequal weights and no longer cancels with the numerator. In measurements, this form fails `shifted` in steps 3 and 4, with a maximum relative error of about 2.5–4 × 10⁻⁴.

In attention, logits are usually only a few tens, `δ` is on the order of 10⁻⁶, and the output is BF16, so FA's use of FMA is fine. This lab's output is FP32 with a tolerance of 1e-4, so it subtracts first, then multiplies. Another form, "multiply `x` by `log₂e` first, then subtract", is even worse: each element's `x · log₂e` is rounded separately, and the errors do not cancel even in steps 1 and 2.

### 4.5 Vectorized memory access and cache hints

`float4` reads or writes 16 bytes at a time (`LDG.E.128` / `STG.E.128` in SASS). One warp instruction moves 512 bytes, so the instruction count is 1/4 of the scalar version, and the same number of instructions keeps 4 times as much data in flight (Little's law from lab 00). The condition is 16-byte alignment: for each row start `x + r * cols` to be aligned, `cols % 4 == 0` and an aligned base address are required. `run()` already checks this and falls back to the scalar version when the condition fails.

`__ldcs` / `__stcs` (PTX `ld.global.cs` / `st.global.cs`, cache streaming) tell the cache "this data is used once; evict it first". In step 4, x and y are each touched only once.

## 5. Schedule design

### Steps 1–3: one block per row

```
grid = rows, block = 256 threads
thread t handles columns t, t+256, t+512, …      ← the 32 threads of a warp read 32 consecutive floats: coalesced
```

| Step | Scans of x | Barriers per row | shared memory |
|---|---|---|---|
| 1 | 3 | 2 reductions × 10 = 20 | 256 floats |
| 2 | 3 | 2 reductions × 2 = 4 | 32 floats |
| 3 | 2 | 1 merge × 2 = 2 | 32 `State`s |

### Step 4: the whole row in registers

`kThreads` threads handle one row, and each thread has `kVecs` chunks. Each chunk is `V` consecutive floats (one `float4` when `V = 4`). Chunk `k` starts at column `(k · kThreads + t) · V`:

```
kThreads = 32, kVecs = 2, V = 4 (cols = 256):
column           0 4 8 ... 124 |           128 132 ... 252
chunk  k=0: lane 0 1 2 ... 31  | k=1: lane 0 1 ... 31
                 ↑ one LDG.E.128: 512 B    ↑ the second one
```

`run()` picks the smallest configuration that fits one row (`chunks = cols / V`, rounded up):

| chunks | kThreads | kVecs | Per block | floats per thread (V = 4) |
|---|---|---|---|---|
| ≤ 32 / 64 / 128 / 256 | 32 (one warp per row) | 1 / 2 / 4 / 8 | 4 rows, 128 threads | 4–32 |
| ≤ 512 / 1024 / 2048 | 128 / 256 / 512 | 4 | 1 row | 16 |
| ≤ 4096 | 512 | 8 | 1 row | 32 |
| ≤ 32768 / V | 1024 | 32 / V | 1 row | 32 |
| larger | fall back to step 3 | | | |

The configurations of the three perf cases:

| case | Configuration | Registers per thread | Resident per SM |
|---|---|---|---|
| rows256 | one warp per row, `kVecs = 2` | 27 | 16 blocks = 64 rows |
| rows4096 | 256 threads per row, `kVecs = 4` | 40 | 6 blocks = 6 rows |
| rows32768 | 1024 threads per row, `kVecs = 8` | 50 | 1 block = 1 row |

The timeline of one block is "all loads → reduction → all stores". When several blocks are resident on an SM, one block's loads are in flight while another block is reducing or storing, so HBM always has work. rows32768 has only one row per SM (a 128 KB row already takes half the register file), so the three phases cannot overlap, which is why it is slower than the other two cases (Section 7).

## 6. Step-by-step exercises

Run all commands from the repository root. Each step asks only for its own TODO.

### Step 1 · TODO(1): three-pass softmax, shared memory tree reduction

Implement `block_reduce_tree` (TODO(1a)) and the three pass loops of `softmax_three_pass` (TODO(1b)).

```bash
./lab check 01 --step 1
./lab bench 01 --step 1
./lab bench 01 --step 1 --case rows256
```

- Debug with the smallest case first: `./lab check 01 --step 1 --case tiny`.
- Write the loop as `for (int c = threadIdx.x; c < cols; c += kBlock)`. In `one_col`, 255 threads get no element at all; their local maximum is `-inf` and their local sum is 0, which are exactly the identity elements.
- Step 2 reuses this kernel with `kShuffle = true`, so call `block_reduce<kShuffle>` in all three passes; do not call `block_reduce_tree` directly.

### Step 2 · TODO(2): warp shuffle reduction

Implement `warp_reduce` and `block_reduce_shfl`. The kernel itself needs no change.

```bash
./lab check 01 --step 2
./lab bench 01 --step 2 --case rows256
./lab profile 01 --step 2 --case rows256
```

Compare `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` in the profiles of steps 1 and 2: on average, for each issued instruction, how many warps are stuck at a barrier. We measured about 2.6 for step 1 and about 1.4 for step 2. `rows256` has only 256 elements per row, so most of its time goes to the reduction, which makes this case the most sensitive to how the reduction is written.

### Step 3 · TODO(3): online softmax

Implement `merge`, `warp_merge`, `block_merge` and `softmax_online`.

```bash
./lab check 01 --step 3 --case masked
./lab check 01 --step 3
./lab bench 01 --step 3 --case rows32768
./lab profile 01 --step 3 --case rows32768
```

- In the first pass loop, write `s = merge(s, State{xr[c], 1.f});`. It may look like you can save one exponential: when `x > m` compute only `e^{m-x}`, otherwise only `e^{x-m}`. We measured this form as more than 1.6 times slower on `rows4096`: the branch keeps the compiler from hoisting the later loads (in SASS only 1–2 `LDG` are in flight at a time, versus 8 back to back for the branch-free form). This kernel is limited by memory traffic in flight, not by exponentials.
- In the profile, `dram__bytes_read.sum` for rows32768 should be about 2 times the size of x (512 MB × 2), and about 3 times for step 1.

### Step 4 · TODO(4): keep the whole row in registers

Implement `softmax_regs`. The configuration choice, `load_chunk`/`store_chunk` and `row_merge` are already written.

```bash
./lab check 01 --step 4 --case short_rows
./lab check 01
./lab sass 01                         # LOCAL should be 0 for every softmax_regs
./lab bench 01 --case rows4096
./lab bench 01 --case rows32768
./lab bench 01 --case rows256
./lab bench 01 --case rows256 --cold
```

- Threads whose local maximum is `-inf` (threads with no elements in `one_col`, threads whose elements are all `-inf` in `masked_short`): use 0 as the shift, so every `v` becomes `e^{-inf} = 0` and `l = 0`. The final scaling factor `e^{m_t - M}` is also 0.
- The 65537 columns of `huge_row` do not fit in registers, so `run()` switches to the step 3 kernel, and this case also depends on TODO(3).

### Common mistakes

| Symptom | Likely cause |
|---|---|
| NaN in `large` | the maximum was not subtracted, so `e^{x}` overflows |
| NaN in `masked`, `one_col`, `masked_short` | `(-inf) - (-inf)`: `merge` or the local shift in step 4 does not handle `m = -inf` |
| only `shifted` fails in steps 3 and 4, relative error about 2e-4–4e-4 | used `fmaf(x, log2e, -m * log2e)`: the rounding errors of the partial states' shifts do not cancel at merge time (Section 4.4) |
| only `shifted` fails, in all steps | multiplied `x` by `log2e` first, then subtracted: each element has a different rounding error |
| occasional wrong, unstable results | missing `__syncthreads()` after reading `smem`, so the next reduction overwrites values not yet read |
| `TIMEOUT` or wrong results, only in block-per-row configurations | `__syncthreads()` sits in a branch that some threads skip (for example, threads with no elements `return` early) |
| in step 2 only lane 0 is right | used a `__shfl_down_sync` reduction, which leaves the result only in lane 0; this lab requires every lane to get the result |
| only `short_rows` fails in step 4 | warps with `row >= rows` in the last block access out-of-bounds rows |
| only `tail` or `tiny` fails in step 4 | out-of-bounds chunks are not filled with `-inf`, or the store does not check `c < cols` |
| step 4 is correct but slow, `LOCAL` nonzero | a loop is not fully unrolled, so `v` was put in local memory |

## 7. Performance

Measured (CUDA Graph timing; `vs torch` is the time of PyTorch softmax (`aten::_softmax`) divided by the time of this version):

| Version | rows4096 `[16384, 4096]` | rows32768 `[4096, 32768]` | rows256 `[262144, 256]` |
|---|---|---|---|
| step 1 | ≈ 1.9× | ≈ 0.6–0.75× | ≈ 0.1× |
| step 2 | ≈ 1.9× | ≈ 0.75× | ≈ 0.18× |
| step 3 | ≈ 2.0× | ≈ 1.0× | ≈ 0.15× |
| step 4 | ≈ 2.4× (84% of nominal HBM peak) | ≈ 1.2× (68%) | ≈ 1.08× (84%) |

On these three cases torch reaches about 35%, 56% and 77% of nominal peak. If step 4 reaches 0.9 times torch or better on all three cases, the mechanisms of this lab work. On the slower versions, `vs torch` occasionally jumps (on the same input torch is sometimes about 1.6 times slower); run again. Comparing times between your own steps is more reliable.

**Explaining this table with the profiler.** Below are `dram__bytes_read.sum` (HBM reads) and `lts__t_sectors_srcunit_tex_op_read.sum × 32 B` (L2 reads), in units of the size of x:

| Version | rows4096 HBM reads | rows4096 L2 reads | rows32768 HBM reads | rows32768 L2 reads |
|---|---|---|---|---|
| step 1 | 1.0 | 2.7 | 3.0 | 3.0 |
| step 3 | 1.0 | 1.8 | 2.0 | 2.0 |
| step 4 | 1.0 | 1.0 | 1.0 | 1.0 |

- **rows4096**: a row is 16 KB. In steps 1–3 each SM holds 8 blocks of 256 threads, so the rows being processed at once across the GPU take about 148 × 8 × 16 KB ≈ 19 MB, far less than the 126 MB L2. Passes 2 and 3 hit in L2, and HBM is read only once. The three-pass algorithm loses nothing here; the gain of step 4 comes from fewer instructions and fewer L2 accesses.
- **rows32768**: a row is 128 KB, and the rows processed at once take about 148 × 8 × 128 KB ≈ 155 MB, more than L2, so rereads go back to HBM. Here the value of online softmax reading x one fewer time shows: step 3 is about 1.3–1.6 times faster than step 1. This is exactly the situation FlashAttention faces: a row of `S` is too large to fit in on-chip memory.
- **rows256**: in steps 1–3 each of the 256 threads gets only 1 element, and most of the time goes to waiting at barriers and scheduling new blocks. Step 4 uses one warp per row, has no barriers (barrier stall is 0), and needs only 2 `float4` load instructions per row; `sm__sass_inst_executed_op_global_ld.sum` drops from about 6.29 million in step 1 to about 0.52 million. It is about 6 times faster than step 2.
- **Step 4 on rows32768** reaches only 68%: `sm__warps_active` is about 43%, each SM holds only one block of 1024 threads, and the load, reduction and store phases cannot overlap with other blocks (Section 5).

`./lab profile` locks the clocks, so `gpu__time_duration` is slower than `bench`; use it only to compare metrics.

## 8. Self-check questions

1. For `x = [1000, 1000]`, what does FP32 give if you compute softmax without subtracting the maximum? And after subtracting it?
2. In the example of Section 3.4, split as `A = [2, 4]` and `B = [1, 3]`. Write down `state(A)`, `state(B)` and the merged result.
3. When merging `(m_A, l_A) = (-inf, 0)` with `(m_B, l_B) = (5, 2)`, which `e^{…}` terms appear in the `merge` formula? Does it produce NaN? What if both are `(-inf, 0)`?
4. The warp shuffle butterfly has 5 rounds. What happens if only 20 lanes of a warp reach `__shfl_xor_sync(0xffffffff, …)`? How does the code in this lab avoid this?
5. In step 4, rows32768 uses 1024 threads × 32 floats. Why not use 512 threads × 64 floats so that each SM holds more rows?
6. The FlashAttention source uses `exp2f(s * scale_log2 - max_scaled)` (one FMA). Why is that acceptable in attention, but not on the `shifted` case in steps 3 and 4 of this lab? And why are steps 1 and 2 fine?

<details>
<summary>Answers</summary>

1. `e^{1000} = inf`, and `inf / (inf + inf) = NaN`. After subtracting the maximum, both exponentials are `e⁰ = 1`, and the result is `[0.5, 0.5]`.
2. `state(A) = (4, e^{2-4} + 1) = (4, 1.135335)`, `state(B) = (3, e^{1-3} + 1) = (3, 1.135335)`. Merge: `m = 4`, `l = 1.135335 · 1 + 1.135335 · e^{-1} = 1.135335 + 0.417666 ≈ 1.553002`, the same as in Section 3.4.
3. `m = 5`, `l = 0 · e^{-inf - 5} + 2 · e^{0} = 0 · 0 + 2 = 2`, no NaN. When both are `(-inf, 0)`, `m = -inf`, `e^{(-inf) - (-inf)} = e^{NaN}`, and `0 · NaN = NaN`, so `merge` must return `(-inf, 0)` directly when `m = -inf`.
4. The mask declares that all 32 lanes take part, but some lanes do not reach the instruction, so the behavior is undefined (it may read garbage or hang). In this lab every thread executes the shuffle; threads with no elements take part carrying the identity `-inf`, 0 or `(-inf, 0)`.
5. The register file has only 256 KB per SM, and the data of one 128 KB row already takes half of it; with addresses, loop variables and so on, two rows do not fit. The 512 × 64 configuration needs more than 100 registers per thread, so each SM still holds only one row, with fewer warps per SM. To hold another row you would have to put part of the data in shared memory, or split a row across several SMs (the cluster of lab 33).
6. The FMA form first rounds `m · log₂e` to FP32, with an error `δ(m)` of at most half an ulp of its binade, so it computes `e^{x-m} · 2^{-δ(m)}`. In steps 1 and 2 the whole row has one shift `M`, so the numerator and denominator carry the same factor `2^{-δ(M)}`, which cancels in the division. In steps 3 and 4 the partial states use different shifts `m_t`, each carrying `2^{-δ(m_t)}`; the factor `e^{m_t - M}` applied at merge time cannot cancel them, and the denominator becomes a sum with unequal weights. In `shifted`, `m · log₂e ≈ 7200`, and half an ulp is 2⁻¹² ≈ 2.4 × 10⁻⁴. The differences between these weights push the output's relative error to the order of 10⁻⁴ (measured about 2.5–4 × 10⁻⁴), above this lab's relative tolerance of 1e-4. FA also uses online softmax, but attention logits are usually only a few tens, `δ` is on the order of 10⁻⁶, and the output is BF16 (relative precision about 4 × 10⁻³), so the effect is negligible.

</details>

## 9. Further reading

- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)), Section 3.1: tiling and softmax rescaling.
- Maxim Milakov, Natalia Gimelshein, *Online normalizer calculation for softmax* (2018, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)): the original online softmax paper, with the single-pass algorithm and parallel merging.
- Tri Dao, *FlashAttention-2* (2023, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)), Section 3.1: deferring the division of the output by `l` to the end (the same as the deferred scaling in step 4 of this lab).
- CUDA Programming Guide: [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions), [Mathematical Functions (ulp error bounds of each function)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- PTX ISA: [ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2), [cache operators of ld (.cs)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
- Mark Harris, *Optimizing Parallel Reduction in CUDA* (a classic NVIDIA presentation): the many ways to write a tree reduction.
- Upstream source (`third_party/flash-attention`, commit `1fc0221`):
  - `csrc/flash_attn/src/softmax.h`: FA2's `softmax_rescale_o`, and the handling of `Check_inf` and `max == -INFINITY ? 0 : max`;
  - `flash_attn/cute/softmax.py`: FA4 (CuTe DSL) `Softmax.online_softmax`, `rescale_O` and `finalize` (LSE = `m + log l`).
