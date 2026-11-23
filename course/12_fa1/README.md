# 12 · FlashAttention-1: tiling and online softmax

> Part 1 · FA1 | Prerequisites: [01_softmax](../01_softmax/README.md), [10_simt_gemm](../10_simt_gemm/README.md), [11_naive_attention](../11_naive_attention/README.md) | Code: [`students/12_fa1/kernel.cu`](../../students/12_fa1/kernel.cu)

## 1. Goals

Following **Algorithm 1** of the FlashAttention paper, write a complete attention forward kernel with FP32 SIMT instructions:

- K/V are tiled into blocks of `Bc` rows, Q into blocks of `Br` rows;
- the **outer** loop runs over K/V blocks `j`, the **inner** loop over Q blocks `i`;
- for every `(i, j)`, `O_i, l_i, m_i` are read from HBM, updated, and written back to HBM;
- the kernel outputs O and LSE (natural log), supports the causal mask, and skips tiles that are fully masked.

After this lab you should be able to:

- prove that "tiling + running (m, l, O)" gives **exactly the same** result as a full-row softmax (not an approximation);
- derive the HBM access count Θ(N²d²/M) with the paper's method, and choose `Br`, `Bc` from the shared memory budget of a B200;
- explain when the "one CTA per (batch, head)" parallelization fails to fill 148 SMs;
- use `./lab bench` and `./lab profile` to see where this version spends its time, and why it is about 2 times slower than the library.

This lab deliberately keeps the paper's loop order (K/V outer). In the next part, `22_fa2` swaps the two loops, and you can compare directly with the numbers from this lab.

## 2. Motivation: removing the N×N intermediate matrices

[11_naive_attention](../11_naive_attention/README.md) splits attention into three kernels: `S = scale·QKᵀ`, `P = softmax(S)`, `O = PV`. S and P have N² elements each and must be written to HBM and read back. At N = 4096, S for one head is 64 MiB (FP32), while Q, K, V, O together are only 4 MiB. The N² term dominates both HBM traffic and memory use.

The idea of Section 3.1 of the FlashAttention paper: **never materialize the full S and P**. Each time, put only one block of K/V and one block of Q into SRAM (on a GPU, that is shared memory and registers), compute the scores, probabilities and contribution to O of this small block, then discard them. The hard part is the softmax: normalizing a row needs the maximum and the exponential sum over **all** keys of the row, but with tiling the later blocks have not been seen yet. The online softmax of [01_softmax](../01_softmax/README.md) already solved half of this problem (m and l). This lab extends it to the weighted sum with V, and proves that it is exact.

## 3. Math

### 3.1 The state of one query over a set of keys

Fix the batch, the head and one query row. For any set A of keys, define:

| symbol | definition | meaning |
|---|---|---|
| `S_j` | `scale · q·k_j` | score of key j |
| `m_A` | `max_{j∈A} S_j` | maximum score over A |
| `l_A` | `Σ_{j∈A} exp(S_j − m_A)` | exponential sum with `m_A` as the reference |
| `O_A` | `Σ_{j∈A} exp(S_j − m_A) · v_j / l_A` | weighted average of V after a softmax **over A only** |

When A is the set of all keys, `O_A` is the required output O, and `m_A + ln(l_A)` is the LSE (`ln Σ_j exp(S_j)`).

By convention, the state of the empty set is `m = −∞, l = 0, O = 0`. These are exactly the initial values that line 2 of Algorithm 1 in the paper writes to HBM.

The paper stores O in **already normalized** form (divided by `l`), and this lab does the same. FA2 changes this to "divide only once at the end"; that is the optimization of the next part.

### 3.2 The merge theorem

Let A and B be two disjoint sets of keys with states `(m_A, l_A, O_A)` and `(m_B, l_B, O_B)`. Let

```
m   = max(m_A, m_B)
α   = exp(m_A − m)          β = exp(m_B − m)
l   = α·l_A + β·l_B
O   = (α·l_A·O_A + β·l_B·O_B) / l
```

Then `(m, l, O)` is exactly the state of A∪B.

**Proof.** First line: `max` over the union clearly holds. For every j in A,

```
exp(S_j − m_A) · α = exp(S_j − m_A) · exp(m_A − m) = exp(S_j − m)
```

so `α·l_A = Σ_{j∈A} exp(S_j − m)`, and likewise `β·l_B = Σ_{j∈B} exp(S_j − m)`; their sum is `l_{A∪B}`. Now O: `l_A·O_A = Σ_{j∈A} exp(S_j − m_A)·v_j` is the "unnormalized numerator", and multiplying by α changes its reference to m; the same holds for B. The sum is the numerator of A∪B, and dividing by `l` gives `O_{A∪B}`. ∎

**Induction to any number of blocks.** Split the keys into `K_1, K_2, …, K_Tc`. Let `A_0 = ∅` and `A_j = A_{j−1} ∪ K_j`. The state of `A_0` is the initial value. If the state of `A_{j−1}` is correct, merging it with the state of block `K_j` by the merge theorem gives the state of `A_j`. Induction up to `j = Tc` gives the state of the full set, that is, O and LSE. The whole process uses only identities over the reals, so **the result is exact**; in floating point only the rounding order differs.

**Why the merge order does not matter.** Merging corresponds to the union of sets, and union is commutative and associative. So the K/V blocks can be processed in any order, or even split into several groups processed in parallel and merged afterwards (the latter is the idea behind split-KV later).

### 3.3 The state of one K/V block: line 10 of Algorithm 1

For a `Br × Bc` score block `S_ij`, compute for each row:

```
m~ = rowmax(S_ij)
P~ = exp(S_ij − m~)        (elementwise)
l~ = rowsum(P~)
```

The unnormalized numerator of this block is `P~·V_j`. Substituting into the merge theorem (A = the blocks already processed, B = block j, `l_B·O_B = P~·V_j`) gives lines 11–12 of the paper:

```
m_new = max(m_i, m~)
l_new = e^(m_i − m_new)·l_i + e^(m~ − m_new)·l~
O_i  ← ( l_i·e^(m_i − m_new)·O_i + e^(m~ − m_new)·P~·V_j ) / l_new
```

### 3.4 By hand: one query, two K/V blocks

`Bc = 2`, 4 keys, with scores (already multiplied by scale) and scalar V:

| key | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| S | 1 | 2 | 4 | 3 |
| V | 10 | 20 | 30 | 40 |

**Block 1** (keys 0, 1): `m~ = 2`, `P~ = (e⁻¹, 1) = (0.36788, 1)`, `l~ = 1.36788`, `P~V = 23.6788`. Merge with the initial value `(−∞, 0, 0)`: `α = e^(−∞) = 0`, so `m = 2`, `l = 1.36788`, `O = 23.6788 / 1.36788 = 17.3106`.

**Block 2** (keys 2, 3): `m~ = 4`, `P~ = (1, e⁻¹)`, `l~ = 1.36788`, `P~V = 30 + 40·0.36788 = 44.7152`. The maximum grew, so the old state must shrink:

```
m_new = max(2, 4) = 4
α     = e^(2−4) = 0.135335        β = e^(4−4) = 1
l_new = 0.135335·1.36788 + 1.36788 = 0.185122 + 1.36788 = 1.553002
O     = (0.185122·17.3106 + 44.7152) / 1.553002 = (3.20458 + 44.7152) / 1.553002 = 30.8562
LSE   = 4 + ln(1.553002) = 4.44019
```

**Direct computation**: `exp(S − 4) = (0.049787, 0.135335, 1, 0.367879)`, with sum 1.553002; the weighted sum is `0.49787 + 2.70671 + 30 + 14.7152 = 47.9198`, and `O = 47.9198 / 1.553002 = 30.8562`. The two agree.

Look at how `l = 1.36788` from block 1 becomes 0.185122 after multiplication by 0.135335: this is exactly the exponential sum `0.049787 + 0.135335` of keys 0, 1 with the new reference 4. Cases such as `large_logits` make the maximum grow in later blocks, and missing one multiplication by α gives a wrong result immediately.

### 3.5 Loop order of Algorithm 1

```
 1  Bc = ⌈M / 4d⌉,  Br = min(⌈M / 4d⌉, d)
 2  initialize O = 0, l = 0, m = −∞ in HBM
 3  split Q into Tr = ⌈N/Br⌉ blocks, K and V into Tc = ⌈N/Bc⌉ blocks
 5  for j = 1 .. Tc:                    ← outer: K/V blocks
 6      load K_j, V_j into SRAM
 7      for i = 1 .. Tr:                ← inner: Q blocks
 8          load Q_i, O_i, l_i, m_i into SRAM
 9          S_ij = Q_i K_jᵀ
10          m~, P~, l~ (Section 3.3)
11          m_new, l_new
12          O_i ← …, write back to HBM
13          l_i ← l_new, m_i ← m_new, write back to HBM
16  return O
```

The paper numbers blocks from 1; the code (and Section 3.7) numbers them from 0, that is, `j = 0 .. Tc−1`, `i = 0 .. Tr−1`.

Each K/V block is read only once; but for each `j`, every Q block and O block is read from HBM once, and O is also written once. This is the biggest difference between FA1 and FA2: FA2 moves Q to the outer loop, so `O_i` stays in registers and is written only once at the end.

### 3.6 HBM accesses: Θ(N²d²/M)

Let M be the number of elements SRAM can hold, and look at one head only:

| data | reads/writes per j | times | total |
|---|---|---|---|
| K, V | — | once | 2Nd |
| Q | read Nd | Tc | Tc·Nd |
| O | read Nd + write Nd | Tc | 2·Tc·Nd |
| l, m | read 2N + write 2N | Tc | 4·Tc·N |

The total is Θ(Tc·Nd) = Θ(N²d / Bc). The paper takes `Bc = Θ(M/d)` and gets **Θ(N²d²/M)** (Theorem 2 of the paper); standard attention is Θ(Nd + N²). With d = 64 and M at a few tens of thousands of elements, `d²/M` is much smaller than 1, so FA1 makes many times fewer accesses than the standard implementation.

Why `M/4d`: one SRAM must hold `K_j`, `V_j` (`Bc·d` each), `Q_i`, `O_i` (`Br·d` each) and `S_ij` (`Br·Bc`) at the same time. Choosing `Bc = M/4d` lets K_j and V_j take M/4 each; `Br ≤ d` gives `Br·Bc ≤ M/4`; `Br ≤ M/4d` keeps Q_i and O_i at no more than M/4 each.

**On a B200.** Each SM has 228 KB of shared memory, and one CTA can use at most 227 KB, that is, M ≈ 58,000 FP32 values. The paper's formula gives:

| d | `Bc = ⌈M/4d⌉` | `Br = min(Bc, d)` |
|---|---|---|
| 64 | 227 | 64 |
| 128 | 114 | 114 |

This lab actually uses `Br = Bc = 64` (Section 5 explains why). Plugging into the table, the perf case (N = 4096, d = 64, Tc = 64) reads about `64 × 2 MiB` and writes about `64 × 1 MiB` per head, 40–60 times the size of Q/K/V/O themselves (4 MiB). Section 7 verifies this number with Nsight Compute measurements.

### 3.7 LSE and causal

**LSE.** After the outer loop ends, the `(m, l)` of each row is the state of the full set, and `LSE = m + ln(l)`. This is the natural log; if you use `exp2` internally, convert back here.

**Causal mask.** Query `r` can see only keys `c ≤ r`. The other scores are set to −∞, and `exp(−∞) = 0`, which is equivalent to removing those keys from the set. The tail works the same way: keys with `c ≥ N` in the last block are padding and are also set to −∞.

**Skipping whole blocks.** Q block i holds queries `i·Br … i·Br+Br−1`, and K/V block j holds keys `j·Bc … j·Bc+Bc−1`. If the last query of the Q block is smaller than the first key of the K/V block, that is, `i·Br + Br − 1 < j·Bc`, the whole block is masked: its state is the empty set, and merging the empty set changes nothing, so the block can be **skipped entirely**, without even reading Q_i and O_i. For a fixed j, the Q blocks to process start at `i = ⌊j·Bc / Br⌋`. So causal does about half the work of non-causal.

**The empty-set trap.** When a row is fully masked within a block, `m~ = −∞` and `exp(S − m~) = exp(−∞ − (−∞)) = NaN`. In this lab `Br = Bc` and fully masked tiles are skipped, so real query rows never hit this case. The code should still replace the subtrahend with 0 in this case (giving `P~ = 0, l~ = 0`), and not update the row when `m_new = −∞`. These two guards cost almost nothing, and with other block sizes they are required.

## 4. Hardware and instructions

This lab has no new instructions; everything it uses was covered earlier: shared memory tiles (00), warp shuffle reductions (01), register-tiled GEMM (10). What is new is the **budget** and the **synchronization**.

### 4.1 Storage budget

| resource | B200 value | used in this lab (d = 64 / 128) |
|---|---|---|
| shared memory | 228 KB per SM, at most 227 KB per CTA | 67 KB / 116 KB |
| registers | 64K 32-bit registers per SM, at most 255 per thread | about 128 / 189 (the entry for the multi-block path `kSingleBlock = false` in `./lab sass 12 --solution`) |
| threads | at most 2048 per SM | 256 per CTA |
| SMs | 148 | — |

When dynamic shared memory exceeds 48 KB, you must first call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` on the kernel, or the launch fails outright. The skeleton does this in `make_plan` (not timed).

How many CTAs can reside on one SM at the same time depends on the tightest resource. d = 64: registers `128 × 256 = 32K`, room for 2 CTAs; shared memory has room for 3; so it is 2. d = 128: both shared memory and registers have room for only 1.

### 4.2 HBM state and `__syncthreads()`

In this lab O, l, m live in **global memory**, and different threads of the same CTA read and write the l, m of the same row. `__syncthreads()` does not only wait for all threads to arrive; it also guarantees that earlier writes to global and shared memory are visible to the other threads of the CTA. This is why "one CTA per head" makes synchronizing the HBM state so simple: all reads and writes of a row happen inside one CTA.

The 64 scores of a row are spread over 16 threads. Every thread needs the old `m_i, l_i`, but only the thread with `tx == 0` writes back the new values. If some thread writes the new values before another thread reads, the result is wrong. The skeleton's order is:

```
load_state (each thread reads old O, m, l)  →  __syncthreads()  →  …  →  merge_state (writes new values)
```

The barrier in the middle guarantees that the write always happens after all reads.

### 4.3 Reduction over groups of 16 threads

Thread `(ty, tx) = (tid/16, tid%16)`, so the 16 threads of a row are 16 consecutive lanes of one warp. `__shfl_xor_sync` with XOR 1, 2, 4, 8 in turn completes the max or sum within these 16 lanes, and all 16 threads get the result (the skeleton provides `group16_max`, `group16_sum`). XOR 16 would cross into another row and must not be used.

### 4.4 Exponentials and −∞

- `expf(x)` is accurate enough for FP32; `expf(-INFINITY) = 0`.
- `fmaxf(-INFINITY, x) = x`, so the initial value of m can enter the max directly.
- `-INFINITY − (-INFINITY)` is NaN, and `0 × NaN` is also NaN. This is why O must be initialized to 0 and the empty set needs special handling.
- The m, l buffers allocated by `lab::Scratch` start as all `0xFF` bytes (NaN), and the harness also fills the O output with NaN: forgetting to initialize shows up as NaN in the result right away.

## 5. Schedule design

### 5.1 One CTA per (batch, head)

```
grid  = B·H CTAs, blockIdx.x = b·H + h
block = 256 threads = 16 × 16
```

This is how the original FA1 kernel parallelizes (as described in Section 3.2 of the FA2 paper): one CTA is responsible for all N rows of one head from start to finish, so all HBM state is shared only within the CTA and no cross-CTA synchronization is needed. The cost: **the parallelism equals B·H**:

| case | B·H | occupancy at 2 CTAs per SM |
|---|---|---|
| `few_heads` [1, 4096, 16, 64] | 16 | 16 SMs have work, 132 are idle |
| `perf` [4, 4096, 64, 64] | 256 | 108 SMs run 2, 40 SMs run 1 |

With long sequences and a small batch (common in inference), B·H is small and most of the GPU sits idle. FA2 also splits CTAs by Q block to solve this problem.

### 5.2 Thread work assignment

Thread `(ty, tx)` is responsible for rows `ty·4 … ty·4+3` of the Q block:

| data | part held by the thread | where it is stored |
|---|---|---|
| `S_ij`, `P~` (64×64) | 4 rows × columns `tx·4 … tx·4+3` | registers `s[4][4]` |
| `m~`, `l~` | 4 rows; each of the 16 threads stores an identical copy | registers |
| `P~·V_j` (64×d) | 4 rows × columns `g·64 + tx·4 … +3`, `g < d/64` | registers `acc[4][d/16]` |
| old `O_i`, `m_i`, `l_i` | the same positions as above | registers (read from HBM) |

### 5.3 Shared memory layout (d = 64)

| array | shape | why it is stored this way |
|---|---|---|
| `Kt` | `[d][Bc+4]` | K transposed: when computing `S`, one thread reads the same dimension of 4 adjacent keys with one float4 |
| `Vs` | `[Bc][d]` | row-major: when computing `P~V`, a float4 reads 4 adjacent d |
| `Qs` | `[Br][d+4]` | `+4` puts two float4 reads 4 rows apart in the same warp into different banks |
| `Ps` | `[Br][Bc+4]` | `P~` written back from registers, for use in `P~V` |

K_j and V_j are loaded only once per j and reused by all Q blocks in the inner loop; Q_i and P~ are rewritten for every (i, j).

### 5.4 Why the block size is 64 and not the paper's 227

The paper's `Bc = M/4d` considers only SRAM capacity, with the goal of minimizing HBM accesses. A SIMT implementation has other constraints:

- `S_ij` lives in registers, and `Br × Bc / 256` is the number of scores each thread must hold. `64 × 227` needs about 57 per thread, and together with the O accumulators it spills to local memory.
- When one CTA takes the full 227 KB, each SM can hold only 1 CTA (8 warps), and latency is harder to hide.
- The kernel in this lab is not HBM-bandwidth-bound (measured in Section 7), so reducing HBM accesses gains little.

So this lab chooses `Br = Bc = 64`: a 4×4 thread tile covers 64×64 exactly, and `Br = Bc` also makes the causal block classification simple (diagonal blocks, fully visible blocks, fully masked blocks). This is a trade-off between "IO-optimal" and "compute-efficient".

### 5.5 Timeline of one (i, j)

```
           all 256 threads
 j start  ─ sync ─ load K_j → Kt, V_j → Vs
 (i, j)   ─ sync ─ load Q_i → Qs ; load_state: O_i, m_i, l_i → registers
          ─ sync ─ tile_scores: s = Q_i K_jᵀ                 (reads Qs, Kt)
                   softmax_tile: s ← P~, m~, l~              (registers + shuffle)
                   store_p: P~ → Ps
          ─ sync ─ tile_pv: acc = P~ V_j                     (reads Ps, Vs)
                   merge_state: write O_i, m_i, l_i back to HBM
 next i   ─ sync ─ … (Qs, Ps may now be overwritten)
```

What each `sync` protects: the first guarantees that `tile_pv` of the previous j no longer reads `Vs`; the second guarantees that the previous tile no longer reads `Qs`, `Ps`; the third makes sure `Qs` (and `Kt`, `Vs` of this j) are fully written, and also orders `load_state` before `merge_state`; the fourth makes sure `Ps` is fully written.

## 6. Step-by-step exercises

Run all commands from the repository root. The skeleton already contains the tile loads, `tile_scores` (the dot products of S), `tile_pv` (P~V), all barriers and both loops; you write the parts specific to attention.

### Step 1 · TODO(1): a single K/V block

This step has only cases with N ≤ 64: there is only one K/V block and one Q block, and no merging is needed. In step 1, `run()` launches `fa1_forward<D, true>`, which calls none of the step 2 functions.

1. `softmax_tile`: multiply by scale, mask the padding keys with `col ≥ n`, compute the row maximum within the 16-thread group, compute `P~` and `l~`.
2. `write_single_block`: `O = acc / l~`, `LSE = m~ + ln(l~)`, writing only rows with `row < n`.

```bash
./lab check 12 --step 1
./lab bench 12 --step 1 --case one_block
```

The `n1` case has only one key: P must be 1 and O equals V. If `n1` gives NaN, most likely the only key was also treated as padding, or the subtrahend for the empty set was not handled. Step 1 is only a correctness exercise; the bench case is too small, and the time is mostly launch overhead.

### Step 2 · TODO(2): the full outer K/V loop

Implement lines 2, 8, 11–13 of Algorithm 1 and the final LSE:

1. `init_state`: O = 0, l = 0, m = −∞;
2. `load_state`: read the O, m, l this thread is responsible for;
3. `merge_state`: the formulas of Section 3.3, writing O, m, l back;
4. `write_lse`: `lse = m + ln(l)`.

```bash
./lab check 12 --step 2
./lab check 12 --step 2 --case two_blocks      # first get the smallest two-block case working
./lab bench 12 --step 2
```

With only `--case`, the lab runs only the steps that list this case; to choose which step to check, also pass `--step`.

After writing the code, first check it on paper: for the first block, `m_old = −∞, l_old = 0, o_old = 0`, and your `merge_state` should reduce to the result of `write_single_block`.

### Step 3 · TODO(3): causal

1. Add the causal mask in `softmax_tile` (set `col > row` to −∞), and delete `if (causal) LAB_TODO(3);`;
2. `first_q_block`: return the index of the first Q block that needs processing.

```bash
./lab check 12                                 # all steps
./lab bench 12 --step 3 --case perf_causal
```

Do only item 1 first: also delete the `if (causal) LAB_TODO(3);` in `first_q_block`, so it returns 0 for causal as well. Bench `perf_causal` once, then do item 2. The former takes almost the same time as non-causal; the latter is about twice as fast.

### Common mistakes

| symptom | likely cause |
|---|---|
| step 1 is all NaN | the tail mask also masked valid keys; or a fully masked row used `−∞` as the subtrahend |
| `short_tail` fails, `one_block` passes | the padding keys with `col ≥ n` were not masked (their score is 0, not −∞) |
| step 2 result is all NaN | O was not initialized to 0 (`0 × NaN = NaN`), or l, m were not initialized |
| `two_blocks`, `rescale` fail, `uniform` passes | the merge misses the multiplication by `e^(m_old − m_new)`, or applies it to O but not to l |
| O of `ones_v` is not 1 | O is divided by `l_new` one time too many or too few; the paper's O is already normalized |
| occasional failures, unstable results | threads other than `tx == 0` also write m, l, or the old values are not read before the barrier |
| LSE of `uniform` is not `ln(N)` | LSE used `log2`, or `write_lse` did not wait for all blocks to finish |
| row 0 of `causal_*` is wrong | the mask was written as `col >= row` (which also masks the diagonal) |
| `causal_peaked` fails, `causal_tail` passes | the skipped blocks are wrong: `first_q_block` skipped Q blocks that still need processing |

## 7. Performance

### 7.1 Relative to the library baseline

Measured (CUDA Graph timing, FP32; `vs efficient` is the time of PyTorch memory-efficient attention divided by the time of this version):

| case | version | `vs efficient` | % of FP32 nominal peak |
|---|---|---|---|
| `perf` [4, 4096, 64, 64] | step 2 / 3 | ≈ 0.5× | ≈ 33% |
| `perf_causal` | step 3, **no block skipping** | ≈ 0.26× | ≈ 17% |
| `perf_causal` | step 3, block skipping | ≈ 0.5× | ≈ 33% |
| `few_heads` [1, 4096, 16, 64] | step 3 | ≈ 0.06× | ≈ 4% |

`% peak` is relative to the B200 FP32 nominal 75 TFLOP/s. Step 3 reaches 0.4× of the baseline or better on both `perf` and `perf_causal`, which shows the mechanism works. Under `--cold` the ratios stay the same: the data is larger than L2, so it comes mainly from HBM anyway.

Some observations:

- **Block skipping makes causal about twice as fast**, exactly the amount of work skipped.
- **`few_heads` has only 16 CTAs**, so only 16 of the 148 SMs work. The library kernel splits by Q blocks of 64 rows and has 16 × 64 = 1024 CTAs for the same case, so the gap widens to about 8 times (0.5× → 0.06×).
- **2 CTAs on one SM does not mean 2 times the throughput**: in `few_heads` each SM runs only 1 CTA, and the time per head is about 0.57 times that of `perf` (where most SMs run 2), not 0.5 times.

### 7.2 Where the time goes

```bash
./lab profile 12 --case perf
```

Measured Nsight Compute results (`perf`, as ratios):

| metric | result | explanation |
|---|---|---|
| `dram__bytes_read.sum` | ≈ 44 × total size of Q, K, V | Q and O are reread for every j, consistent with the estimate in Section 3.6 |
| `dram__bytes_write.sum` | ≈ 67 × size of O | O is written Tc + 1 = 65 times, plus m, l |
| HBM bandwidth utilization | about 15% of nominal peak | total bytes read and written divided by the `bench` time; the extra traffic is far from saturating HBM |
| `sm__throughput` | about 50% | mainly FMA and shared memory reads |
| `sm__warps_active` | about 22% | only 8–16 warps per SM |

Conclusion: in SIMT FP32, **the extra HBM round trips of FA1 are not the bottleneck**; computation is. The reason is that SIMT FP32 compute (nominal 75 TFLOP/s) is an order of magnitude lower than tensor cores: the same HBM traffic, spread over a very long compute time, does not stand out. In Part 2, after the switch to tensor cores, computation becomes more than ten times faster, these round trips become the main cost, and the loop swap of FA2 shows its real value.

### 7.3 Why it is about 2 times slower than the library

PyTorch's FP32 memory-efficient attention runs the CUTLASS kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80` on B200. Compared with this lab:

| | this lab | library |
|---|---|---|
| matrix multiply | SIMT FFMA | TF32 tensor core, with 3 TF32 multiplications to recover FP32 accuracy ("3xTF32", inferred from the source configuration, see Section 9; tensor pipe instructions are visible in Nsight) |
| loop order | K/V outer, O round trips through HBM | Q blocks outer, O stays in registers (`rf`), FA2 style |
| CTA count | B·H | B·H·⌈N/64⌉ |

So the library is 2 times faster not because it "writes SIMT better", but because it uses hardware this lab has not covered yet (tensor cores) and the scheduling of the next part. These are also what the rest of the course teaches.

### 7.4 Experiments to try yourself

- Compare `few_heads` with `perf`: compute time × CTA count / SM count to get the effective throughput per SM.
- Double `Bc` to 128, which halves the HBM traffic of Section 3.6; predict how the time will change, then think about why (hint: Section 7.2).

## 8. Self-check questions

1. In the merge theorem, if `m_A = m_B = −∞` (two empty sets), what does the formula compute? How should the code handle it?
2. Line 12 of the paper first multiplies by `l_i` and then divides by `l_new`. If HBM stored the unnormalized numerator `l·O`, how should line 12 be written? Which operations does it save?
3. In the perf case (N = 4096, d = 64, `Bc = 64`), how many times is O of one head read from HBM? How many times if `Bc = 227` (the paper's formula)?
4. With causal, `Br = Bc = 64` and N = 4096, how many `(i, j)` tiles are processed in total? And for non-causal? What is the ratio?
5. Why must `load_state` come before `__syncthreads()` and `merge_state` after it? If each thread kept its own copy of l, m (16 times the space), would this constraint still be needed?
6. With B = 1, H = 8, N = 32768, how many SMs does this lab's kernel use? Which change in FA2 solves this problem?

<details>
<summary>Answers</summary>

1. `m = −∞`, `α = β = exp(−∞ − (−∞)) = NaN`, and the result is all NaN. Merging two empty sets should still give the empty set: the code keeps the old state when `m_new = −∞` (the `merge_state` of this lab skips the row).
2. When storing `u = l·O`, line 12 becomes `u_i ← e^(m_i − m_new)·u_i + e^(m~ − m_new)·P~V_j`. The multiplication by `l_i` and the division by `l_new` are no longer needed; `O = u/l` happens only once at the end. This is the "delayed normalization" of FA2.
3. Read Tc = 4096/64 = 64 times (once per j); written 65 times, that is, once per j plus once for initialization (consistent with Tc + 1 in Section 7.2). With `Bc = 227`, Tc = ⌈4096/227⌉ = 19: read 19 times, written 20 times.
4. Tr = Tc = 64. Non-causal: 64 × 64 = 4096. Causal: for each j, process `i = j … 63`, for a total of 64 + 63 + … + 1 = 2080; the ratio 2080/4096 ≈ 0.51 matches the measured speedup.
5. All 16 threads of a row read the old m, l, but only one thread writes the new values; the barrier guarantees that all reads happen before the write. If each thread kept its own copy and read and wrote only that copy, there would be no cross-thread reads and writes, and this barrier would no longer be needed (but the HBM traffic would be 16 times larger).
6. 8 CTAs, using only 8 SMs (about 5%). FA2 further splits each head into ⌈N/Br⌉ CTAs by Q block. The CTAs share no state (each CTA owns its own O_i), and the parallelism becomes B·H·⌈N/Br⌉.
</details>

## 9. Further reading

- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022): Section 3.1 and Algorithm 1 (the algorithm of this lab), Section 3.2 Theorem 2 (IO complexity) and Proposition 3 (lower bound), Appendix C (proofs).
- Tri Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023): Sections 3.1–3.2 describe how FA1 parallelizes (one thread block per head) and the motivation for the loop swap; a preview of the next part.
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018): the original online softmax paper.
- CUDA C++ Programming Guide: [Synchronization Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-functions) (the visibility guarantee of `__syncthreads()` for global memory), [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions), [Shared Memory (dynamic allocation above 48 KB)](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory).
- Upstream source (`third_party/flash-attention`, pinned version): the original FA1 CUDA implementation belongs to flash-attn 1.x and is not in the version pinned in this repository. `csrc/flash_attn/src/flash_fwd_kernel.h` and `csrc/flash_attn/src/softmax.h` (`softmax_rescale_o`) are the FA2 implementation; read them next to this lab's `merge_state` to see what "after the loop swap the state stays in registers" looks like.
- Source of the library baseline: `third_party/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h` (the origin of PyTorch memory-efficient attention); the configuration that sends FP32 to tensor cores is `DefaultGemmType` (`OpMultiplyAddFastF32`) in `gemm_kernel_utils.h` in the same directory.
