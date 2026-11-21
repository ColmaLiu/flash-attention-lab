# 22 · FlashAttention-2: Tensor Cores, loop interchange and warp partitioning

> Part 2 · FA2 | Prerequisites: [12_fa1](../12_fa1/README.md), [20_mma_gemm](../20_mma_gemm/README.md), [21_pipeline_gemm](../21_pipeline_gemm/README.md) | Code: [`students/22_fa2/kernel.cu`](../../students/22_fa2/kernel.cu)

## 1. Goals

Write the FlashAttention-2 forward kernel with `mma.sync`: FP16/BF16 Q, K, V, O in `[B,N,H,D]` layout, D = 64 or 128, both causal and non-causal, with O and an FP32 LSE as outputs.

The FA1 of lab 12 is SIMT FP32. This lab replaces the matrix multiplies with the Tensor Cores from labs 20 and 21, and reworks the whole algorithm following section 3 of the FA2 paper:

| | Lab 12 (FA1) | This lab (FA2) |
|---|---|---|
| Loop order | K/V blocks outer, Q blocks inner | **Q blocks outer**, K/V blocks inner |
| Where O, ℓ, m live | HBM, read and written once per block | **Registers**, written once at the end |
| Normalization | Divide by ℓ for every block | **Divide once at the end** |
| Parallelism | One CTA per (batch, head) | **One CTA per (Q block, head, batch)** |
| Work split inside a CTA | Threads split the rows and columns of S | **Each warp owns 16·k query rows** |
| How P enters P·V | Written to shared memory | **Stays in registers, used directly as the mma A operand** |

When you finish this lab, you should be able to:

- explain what each of FA2's three changes saves (HBM round trips, non-matmul operations, inter-warp communication);
- derive from the `m16n8k16` fragment table that "a C fragment can be used directly as an A fragment";
- explain why V must be read with `ldmatrix.trans`;
- under causal masking, compute only the blocks that are needed and mask only the blocks that need it;
- explain why this lab reaches only about 0.26–0.4× of cuDNN on B200, and what is needed to close the remaining gap.

## 2. Motivation: three problems FA1 exposes on Tensor Cores

The conclusion of lab 12 §7.2 is: on SIMT FP32, FA1's extra HBM round trips are not the bottleneck; compute is. With Tensor Cores, the matrix multiplies get more than ten times faster, and the overheads that used to hide behind compute become visible. Section 3 of the FA2 paper (Dao 2023) targets exactly these three points:

1. **HBM round trips of O, ℓ, m** (paper sections 3.1, 3.2). FA1's outer K/V loop moves the state of each Q block in and out of HBM Tc times. With Q in the outer loop, one CTA processes all of K/V for one Q block, and the state can stay in registers the whole time. Q blocks are also independent of each other, which adds one more level of parallelism over Q blocks: lab 12's `few_heads` (B=1, N=4096, H=16) has only 16 CTAs; the same B, N, H in this lab gives 16 heads × 32 Q blocks (128 rows per block at D=64) = 512 CTAs.
2. **Non-matmul operations** (paper section 3.1). Tensor Core throughput is an order of magnitude higher than that of the ordinary FP32 units, so "one more multiply and one more divide per O element" is not cheap. FA2 keeps O **unnormalized** in the loop and divides only once at the end; the backward pass needs only the LSE, so ℓ and m are no longer saved separately.
3. **Communication between warps** (paper section 3.3). FA1's CUDA implementation splits K/V across warps ("split-K"): after each warp computes a partial S·V, the warps' results are summed through shared memory with synchronization, and the softmax row maximum must also be merged across warps. FA2 instead assigns **Q rows** to warps: all keys of a row are in the same warp, the row maximum and row sum need only shuffles within the warp, and the PV results need no merging.

## 3. Math

### 3.1 The invariant after loop interchange

A CTA owns query rows `q0 .. q0+Br−1` (below we look at one of them, row i). It processes K/V blocks `j = 0, 1, …` in order, with Bc keys per block. After the first j blocks, the registers hold:

```
m  = max_{k seen} s_k                          (row maximum, s_k = q_i · k_k)
ℓ  = Σ_{k seen} exp(scale·(s_k − m))
O~ = Σ_{k seen} exp(scale·(s_k − m)) · v_k     (unnormalized!)
```

For a new block j, first compute the new maximum `m' = max(m, max_{k∈j} s_k)`, then:

```
α  = exp(scale·(m − m'))
P~ = exp(scale·(S_j − m'))           one exp per element
ℓ  ← α·ℓ + rowsum(P~)
O~ ← α·O~ + P~ · V_j                 one multiply + one mma
m  ← m'
```

After all blocks, `O = O~ / ℓ` and `LSE = scale·m + ln ℓ`. Compared with the merge theorem of lab 12, each block does only one multiply per O element (by α), and the division goes from "once per block" to "once in total".

### 3.2 By hand: one row, two blocks

scale = 1, D = 1. Block 0 has key scores `s = (2, 0)` and values `v = (1, 0)`; block 1 has scores `(4, 2)` and values `(2, 1)`.

| | m | P~ | ℓ | O~ |
|---|---|---|---|---|
| After block 0 | 2 | (1, e⁻² = 0.1353) | 1.1353 | 1·1 + 0.1353·0 = 1 |
| Block 1: α = e^(2−4) = 0.1353 | 4 | (1, 0.1353) | 0.1353·1.1353 + 1.1353 = 1.2889 | 0.1353·1 + 1·2 + 0.1353·1 = 2.2707 |

The result is `O = 2.2707 / 1.2889 = 1.7617` and `LSE = 4 + ln 1.2889 = 4.2538`. Direct computation: `e² + e⁰ + e⁴ + e² = 70.376`, `ln 70.376 = 4.2538`; `(e²·1 + e⁴·2 + e²·1)/70.376 = 1.7616`. The two agree (the difference comes from rounding).

### 3.3 Powers of 2 instead of powers of e, and m without scale

The hardware only has `2^x` (`ex2.approx`, `MUFU.EX2` in SASS). Since `exp(x) = 2^(x·log₂e)`, let `scale_log2 = scale·log₂e`:

```
exp(scale·(s − m)) = 2^(s·scale_log2 − m·scale_log2)
```

The exponent is one FFMA (`fmaf(s, scale_log2, −m·scale_log2)`), and `m·scale_log2` is computed only once per row.

m holds the maximum of the **raw scores, not multiplied by scale**. This is valid only for scale > 0: `max(scale·s) = scale·max(s)`. The supported domain of this course is a finite scale > 0, and the harness rejects other values (the v1 reference solution produced NaN for scale ≤ 0 because it ignored this point). At the end, `LSE = m·scale + ln ℓ`.

**Fully masked rows.** If every key a row has seen so far is masked, `m' = −∞`, and both `s·scale_log2 − m'·scale_log2` (for a masked s = −∞) and the α exponent `m·scale_log2 − m'·scale_log2` are `(−∞) − (−∞) = NaN`. The fix: when `m' = −∞`, use 0 in place of the subtrahend, so `α = 2^(−∞) = 0` and `P~ = 2^(−∞) = 0`, and the state stays all zero. The block partitioning of this lab guarantees that real rows never hit this case (key 0 of block 0 is visible to every row), but the code is written defensively so that it stays correct if the block size changes or new kinds of masks are added.

**Partial sums of ℓ within a quad.** As the next section shows, the Bc scores of a row are spread over 4 lanes, Bc/4 per lane. The row maximum must be computed by all 4 lanes together (otherwise the 4 lanes use different m, and their P~ are no longer comparable); ℓ need not be: as long as the 4 lanes use the same α, each lane can accumulate only its own part of P~, and the 4 partial sums are added at the end, because `α·(a+b+c+d) = α·a + α·b + α·c + α·d`. This saves two shuffles per block. The `uniform` case (Q = 0, LSE should be ln N) checks specifically that this final step is not forgotten: forgetting it gives ln(N/4).

## 4. Hardware and instructions

This lab has no new instructions. `mma.sync`, `ldmatrix` and swizzle come from lab 20, `cp.async` from lab 21, all provided by headers: [`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh), [`include/lab/cp_async.cuh`](../../include/lab/cp_async.cuh). What is new is how they are combined.

### 4.1 Review: the m16n8k16 fragments

Let `g = lane / 4` (0..7, "group") and `q = lane % 4` (0..3, "index within the quad"):

| Operand | Register i | Row | Column |
|---|---|---|---|
| A (16×16, two 16-bit values per register) | 0, 1, 2, 3 | g, g+8, g, g+8 | 2q, 2q, 2q+8, 2q+8 (and +1) |
| B (16×8, k × n) | 0, 1 | k = 2q, 2q+8 (and +1) | n = g |
| C/D (16×8, FP32) | 0, 1, 2, 3 | g, g, g+8, g+8 | 2q, 2q+1, 2q, 2q+1 |

Each row of C (for example row g) is held jointly by lanes `4g, 4g+1, 4g+2, 4g+3`; these 4 lanes are called a **quad**. A row reduction needs only shuffles within the quad:

```cpp
x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));   // lane ^ 1: same quad
x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));   // lane ^ 2: same quad
```

### 4.2 S = Q·Kᵀ: exactly like a GEMM

`S[i][j] = Σ_d Q[i][d]·K[j][d]`. Compared with the GEMM `C = A·Bᵀ`: A = Q (`[row][D]`, D contiguous), B = K (`[key][D]`, D contiguous), which is lab 20's "K-major B". So the A fragment of Q and the B fragment of K both use the plain `ldmatrix_x4`, with the same address formula as lab 21's `compute_stage`.

The difference is that **Q is read only once**: a warp's 16 rows × D columns of Q do not change during the whole K/V loop, so all of its A fragments are read into registers before the loop starts (32 registers per lane at D = 128), and afterwards each block only needs to read K. This is another benefit of the loop interchange: Q becomes a "resident" operand.

### 4.3 P skips shared memory: a C fragment is an A fragment

The A operand of P·V is P (16 rows × Bc keys). The mma A is 16×16, so each step takes 16 keys, that is, **two adjacent n8 S tiles** (tile 2t and 2t+1). Compare the table of §4.1 register by register:

| A register | Elements needed (row, key) | Location in the C fragment |
|---|---|---|
| 0 | (g, 16t+2q), (g, 16t+2q+1) | floats 0, 1 of tile 2t |
| 1 | (g+8, 16t+2q), (g+8, 16t+2q+1) | floats 2, 3 of tile 2t |
| 2 | (g, 16t+8+2q), (g, 16t+8+2q+1) | floats 0, 1 of tile 2t+1 |
| 3 | (g+8, 16t+8+2q), (g+8, 16t+8+2q+1) | floats 2, 3 of tile 2t+1 |

Every cell on the right is in the **same lane's** own registers: both A and C put the row at `g = lane/4` and two adjacent columns at `2q, 2q+1`; the 16 columns of A are the 8 columns of two C tiles side by side. So the only work is converting two FP32 values into an FP16/BF16 pair (`__floats2bfloat162_rn` / `__floats2half2_rn`); no shuffle or shared memory is needed.

```cpp
a[0] = pack2(s[2t][0],   s[2t][1]);    a[1] = pack2(s[2t][2],   s[2t][3]);
a[2] = pack2(s[2t+1][0], s[2t+1][1]);  a[3] = pack2(s[2t+1][2], s[2t+1][3]);
```

`pack2(x, y)` puts x in the low 16 bits: the low half is the element with the smaller column index (the convention at the top of `mma_sync.cuh`).

Converting P to 16 bits rounds it (BF16 has 8 significant bits, about 2–3 significant decimal digits). ℓ uses the FP32 values before rounding, so numerator and denominator are slightly inconsistent. This is what upstream FA2 does; the error is far smaller than the rounding of the BF16 output itself, and the `ones_v` case (V = 1, O should be 1) shows how small it is.

### 4.4 V is MN-major: `ldmatrix.trans`

The B operand of P·V is V; the k direction is the key and the n direction is d. B register i of the mma must hold `(k = 2q+8i, n = g)` and `(k+1, n = g)`: **the same d, two adjacent keys**. But V in shared memory is `[key][D]` with d contiguous; two adjacent keys are a whole row apart and cannot be read as one 32-bit word. In GEMM terms, V is an "MN-major" B (contiguous along n), while K is K-major.

`ldmatrix.trans` transposes each 8×8 matrix as it reads it: lane l gets the two elements in **rows** `2(l%4)` and `2(l%4)+1`, **column** `l/4` of the matrix. Taking 8 key rows of V as the matrix rows gives exactly `(key 2q, d g)` and `(key 2q+1, d g)`, which is what the B register needs.

One `ldmatrix_x4_trans` reads 4 matrices of 8×8:

```
lane 0–7   → key rows 16t+0..7,  d column block c    → r[0]: B register 0 of d tile c
lane 8–15  → key rows 16t+8..15, d column block c    → r[1]: B register 1 of d tile c
lane 16–23 → key rows 16t+0..7,  d column block c+1  → r[2]: B register 0 of d tile c+1
lane 24–31 → key rows 16t+8..15, d column block c+1  → r[3]: B register 1 of d tile c+1
```

That is, lane l supplies the address of key row `16t + l%16`, column block `c + l/16`.

### 4.5 Swizzle for 256-byte rows

`lab::swizzle_chunk<8>` handles rows of 128 bytes (8 chunks of 16 bytes). An FP16 row with D = 128 is 256 bytes. The banks repeat every 128 bytes, so it is enough to treat a row as two 128-byte segments and swizzle each segment separately:

```cpp
offset(row, col) = row*D + (col/64)*64 + swizzle_chunk<8>(row, (col%64)/8)*8 + col%8
```

Within a segment, the same logical chunk of 8 consecutive rows lands on 8 different physical chunks, so neither `ldmatrix` (including `.trans`) nor the `cp.async` writes have bank conflicts.

### 4.6 Common misuses

| Misuse | Consequence |
|---|---|
| Plain `ldmatrix` for V | B gets two adjacent d of the same key; O is completely wrong |
| Out-of-bounds K/V rows not zero-filled | Shared memory of the tail block holds stale data, possibly NaN; `0 × NaN = NaN`, and masking P cannot save it |
| m multiplied by scale before comparing, but the exponent uses raw scores | Two sets of units are mixed; the `large_logits` and `scale` cases fail |
| Quad-summing ℓ inside the online softmax and again in the epilogue | ℓ becomes 4× (or 16×) too large; O is too small |
| Using `exp2f`/`expf` | Correct but slow: without `--use_fast_math` they are not a single `MUFU.EX2` |

## 5. Schedule design

### 5.1 Grid and CTA

```
grid  = (⌈N / Br⌉, H, B)       blockIdx.x = Q block, .y = head, .z = batch
block = 4 warps (128 threads)
warp w owns CTA rows w·16·kMTiles .. ; each lane owns 2·kMTiles of them
```

| D | kMTiles | Br (Q rows/CTA) | Bc (keys/block) | Q / O / S registers per lane | shared memory (steps 2–3 / step 4) |
|---|---|---|---|---|---|
| 128 | 1 | 64 | 64 | 32 / 64 / 32 | 48 KB / 80 KB |
| 64 | 2 | 128 | 32 | 32 / 64 / 32 | 24 KB / 32 KB |

Both configurations spend 128 registers on arrays and leave the rest for addresses and temporaries. With Bc = 64 at D = 64, S would need 64 registers, 160 for all arrays, and the compiler starts spilling to local memory in the step 2 kernel; so Bc = 32 is used instead. The D = 128 kernel uses about 220–240 registers per thread, and 2 CTAs (8 warps) fit per SM; D = 64 uses about 170, and 3 fit.

### 5.2 Flow of one K/V block (step 2)

```
          all warps                            each warp on its own (no barrier)
 ┌───────────────────────────────┐   ┌────────────────────────────────────────────────────┐
 │ cp.async K_j, V_j → smem      │   │ S = Q·K_jᵀ        Q in registers, K_j via ldmatrix │
 │ wait_group 0; __syncthreads() │ → │ mask (only the blocks that need it)                │
 └───────────────────────────────┘   │ m', α, P~ = 2^(…), ℓ, O~ *= α   quad shuffle       │
                                     │ O~ += P~·V_j      P~ as A, V_j via ldmatrix.trans  │
                                     └────────────────────────────────────────────────────┘
                                       __syncthreads()   all warps must finish reading K_j, V_j before they are overwritten
```

The only interaction between warps is sharing the K/V tiles. Everything in the softmax happens inside a warp; this is FA2's "split-Q".

### 5.3 Causal: skipping blocks and masking only diagonal blocks (step 3)

Query i may only see keys j ≤ i. The CTA's rows are `q0 .. q0+Br−1`, so:

- **Skipped blocks**: a block whose first key `k0 > q0+Br−1` is fully masked for every row of this CTA and is not loaded at all. The loop bound changes from `⌈N/Bc⌉` to `min(⌈N/Bc⌉, ⌈(q0+Br)/Bc⌉)`.
- **Blocks that need a mask**: if the block's last key `k0+Bc−1` is greater than the CTA's first row `q0`, some (row, key) inside the block must be masked. In addition, the last block, which extends past N, always needs a mask.
- **All other blocks** are fully visible and are computed directly.

At D = 128 (Br = Bc = 64), Q block i processes i+1 blocks, and only the last one (the diagonal block) is masked. In `perf_causal`, the 64 Q blocks process `64·65/2 = 2080` blocks in total, 51% of the 4096 of the non-causal case; only 64 of them need a mask. At D = 64 (Br = 128, Bc = 32), the diagonal crosses 4 K blocks, and all 4 need a mask.

The mask itself only sets S to −∞. Element e of a C fragment corresponds to row `g + 8·(e/2)` and key `2q + e%2` (plus the tile and block offsets).

### 5.4 cp.async double buffering (step 4)

The same as lab 21 step 1, except that one stage holds a pair (K_j, V_j):

```
prologue: cp.async Q and K_0/V_0 (stage 0), commit           ← one group
for j:
    wait_group 0; __syncthreads()       ① block j visible to all warps  ② all warps done with block j−1
    if j+1 exists: cp.async K_{j+1}/V_{j+1} → stage (j+1)%2; commit
    compute block j (stage j%2)                              ← block j+1 is in flight meanwhile
```

Point ② of `__syncthreads()` guarantees that stage (j+1)%2 (used by block j−1 in the previous round) can be overwritten. Each block needs only one wait and one barrier.

## 6. Step-by-step exercises

Run all commands from the repository root. The skeleton already has load/launch, `issue_rows`, the tail part of the mask, dtype conversion and the quad reductions.

### Step 1 · TODO(1): Q fragments and S = Q·Kᵀ

Implement `load_q` (1a) and `qk_block` (1b). The step 1 kernel `fa2_scores` is already written: it loops over all K blocks and writes `scale·S` into the probe (`[B,H,N,N]`, FP32), which the `fa2_scores` oracle compares against an FP64 `scale·QKᵀ`. There is no mask and no softmax yet.

```bash
./lab check 22 --step 1
```

- The dimension names in the report are `[b, h, i, j]`: an error in `i` means the Q rows (A fragment) are taken wrong; an error in `j` means the K rows (B fragment) are taken wrong.
- Only the D = 128 cases fail: check that you use `tile_offset` (the two-segment swizzle) rather than your own `row*D + col`.
- Single-block cases (`n1`, `one_block`, `short_tail`) pass and multi-block cases fail: `qk_block` must start accumulating S from 0 for every block.

### Step 2 · TODO(2): online softmax, P·V and the epilogue

Implement `online_softmax` (2a), `pv_block` (2b) and the epilogue at the end of `fa2_forward` (2c), in that order. Step 2 has only non-causal cases.

```bash
./lab check 22 --step 2
./lab bench 22 --step 2
```

Run only `./lab check 22 --step 2 --case n1` and `--case one_block` first, then run everything.

### Step 3 · TODO(3): causal

Three places: 3a reduces the number of blocks in the loop, 3b adds the causal condition in `apply_mask`, and 3c decides which blocks need a mask.

```bash
./lab check 22 --step 3
./lab bench 22 --step 3 --case perf_causal
```

When you are done, try an experiment: change 3a back to not skipping blocks (keep only the mask), bench `perf_causal` again, and compare with the table in §7.

### Step 4 · TODO(4): cp.async double buffering

4a issues the copies of block 0 in the prologue; 4b writes the wait, the barrier and the copies of the next block at the top of the loop.

```bash
./lab check 22
./lab bench 22 --case perf
./lab bench 22 --case perf_causal
./lab bench 22 --case perf_d64
./lab sass 22            # look at HMMA / LDSM / LDGSTS / MUFU.EX2, and check for spills
```

### Common mistakes

| Symptom | Likely cause |
|---|---|
| All of step 1 fails, errors look random | Wrong lane → row/chunk mapping in `ldmatrix`; or the fragment roles of Q and K are swapped |
| In step 2, every LSE of `uniform` is off by ln 4 ≈ 1.386 | The epilogue forgot to quad-sum ℓ |
| `ones_v` is far from 1 | Normalization is also done inside the loop, or ℓ and O use different α |
| `position` fails, and other random cases have large errors | Wrong order of P's A registers (registers 1 and 2 swapped), or V is not read with `.trans` |
| Only `tail`/`short_tail` give NaN | Out-of-bounds K/V rows are not zero-filled, or the tail block is not masked |
| `large_logits`-type cases give inf/NaN | The maximum is not subtracted in the exponent, or m and S use different units (§3.3) |
| O of the first Q block is NaN, its LSE is −inf, and all other Q blocks are wrong | 3a counts one block too few (the diagonal block); a common form is `(q0 + kBM - 1) / kBN` missing the `+1` |
| Step 4 is occasionally wrong | A missing barrier, or the other stage is overwritten before the barrier |
| `TIMEOUT` or a hang | Different warps run the loop a different number of times and skip a `__syncthreads()`: the block count must be the same for the whole CTA |

## 7. Performance

### 7.1 Against the library baselines

Measured on GPU 7 (CUDA Graph; `vs cudnn` is cuDNN time divided by the time of this version, larger is better):

| Version | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| Steps 2/3 (synchronous loads) | ≈ 0.23× | ≈ 0.30× | ≈ 0.36× |
| Step 3, causal without block skipping | — | ≈ 0.15× | — |
| Step 4 (double buffering) | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4 (`flash_attn.cute`) | ≈ 1.1× | ≈ 1.3× | ≈ 1.15× |

If step 4 reaches 0.2× of cuDNN or more on all three perf cases, the mechanism works. Against FA4, step 4 is about 0.23×, 0.25× and 0.34×. Under `--cold` (L2 flushed before each run) the ratios barely change (`perf` about 0.28×): Q, K and V total 200 MB, which is larger than L2 anyway.

Some observations:

- **Block skipping makes causal about twice as fast** (0.15× → 0.32×), consistent with the 51% of work computed in §5.3. Without skipping it is even slower than non-causal, because every block needs a mask.
- Causal reaches about 10% fewer TFLOP/s than non-causal: the work of Q block i is proportional to i+1, and when CTAs are launched in `blockIdx.x` order the heaviest CTAs start last, so many SMs sit idle in the last wave. The LPT schedule of lab 43 deals with this.
- Double buffering gains only about 10–15%. Each SM has 2–3 CTAs, so while one CTA waits for data another computes (the same reason as "step 2 changes almost nothing" in lab 21).

### 7.2 Why only about 1/4 of cuDNN: the `mma.sync` upper bound

Lab 20 §4.1 measured the BF16 `mma.sync` upper bound on B200 at about 560 TFLOP/s, while cuDNN and FA4 use `tcgen05` (Parts 3 and 4). Step 4 of this lab reaches about 340 TFLOP/s on `perf`, that is, **about 61% of the `mma.sync` upper bound**; cuDNN reaches about 1300 TFLOP/s, already 2.3× the `mma.sync` upper bound. So:

| | TFLOP/s (`perf`) | vs cuDNN |
|---|---|---|
| Step 4 of this lab | ≈ 340 | ≈ 0.26× |
| `mma.sync` upper bound (lab 20 microbenchmark, no memory access, no softmax) | ≈ 560 | ≈ 0.43× |
| cuDNN (`tcgen05`) | ≈ 1300 | 1× |

Even a perfectly optimized version of this lab could reach only about 0.43× of cuDNN. The real gap is the instruction generation, not the code.

### 7.3 Compared with lab 12

The FA1 of lab 12 is FP32 SIMT and reaches about 25 TFLOP/s on `perf` ([4,4096,64,64]). The D = 64 `perf_d64` of this lab reaches about 320 TFLOP/s, about **13×**. The dtypes differ (FP32 vs FP16), and the ratio comes mainly from the Tensor Cores; but if O/ℓ/m were still written back to HBM per block in FA1's loop order, the extra traffic that lab 12 §7.2 called "not the bottleneck" (reads and writes of tens of times the size of Q/K/V/O; see lab 12 §2 and §7.2) would become the bottleneck at this speed. In this lab's profile, `dram__bytes_read` ≈ 200 MB, exactly one read each of Q, K and V (K/V are read again by the 64 Q blocks of the same head and hit in L2).

### 7.4 Where the time goes

```bash
./lab profile 22 --case perf             # step 4
./lab profile 22 --case perf --step 2
```

Nsight Compute results on this machine (`perf`):

| Metric | Step 2 | Step 4 | Notes |
|---|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 60% | ≈ 66% | Fraction of time the Tensor Cores are busy |
| `sm__warps_active` | ≈ 12% | ≈ 12% | 8 warps per SM (2 CTAs) |
| `sm__inst_executed_pipe_tensor_subpipe_hmma` | 1.34 × 10⁸ | 1.34 × 10⁸ | = 4·B·H·N²·D / (2·16·8·16), not one extra |
| `smsp__inst_executed_pipe_xu` | — | ≈ 0.36 × 10⁸ | Slightly more than B·H·N² / 32 ≈ 0.34 × 10⁸: one `MUFU.EX2` per score, plus α, log and so on |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` | 0 | 0 | With swizzle, `cp.async` writes have no conflicts |

(Nsight Compute locks the clocks, so `gpu__time_duration` is longer than in bench; use it only to compare metrics.)

The remaining ≈1/3 idle time comes from: one `__syncthreads()` per block; the softmax FFMA, `MUFU.EX2` and shuffles run serially with the mma in the same warp, so the Tensor Cores can only wait when no other warp is available; and each SM has only 8 warps, since registers do not allow more. The FA3 ideas of Part 3 target exactly this: dedicated warps for data movement, and overlapping the softmax with the QKᵀ of the next block.

Experiments to try yourself:

- Change D = 128 to 8 warps with Br = 128 (`Config<128, 8, 1, 64, …>`). On this machine this is **slower** (step 4 about 0.22×): only one CTA fits per SM, and no other CTA fills in at its barriers.
- Change D = 64 to Bc = 64, look for spills with `./lab sass 22`, then bench.

## 8. Self-check questions

1. In FA1's outer K/V loop, how many HBM round trips do the O, ℓ, m of each Q block make? And after the loop interchange? After the interchange, how many times is K/V read?
2. With one warp handling 16 rows and Bc = 64, which two rows of S does lane 13 hold, and which columns of each row? Which lanes does it shuffle with?
3. Why must m be maximized within the quad, while ℓ can be summed only at the end?
4. If FA2 were changed to "split-K" (4 warps each computing 16 keys), what extra shared memory reads/writes and synchronization would each block need?
5. Why does V need `ldmatrix.trans` but K does not? If V were stored in HBM as `[B,H,D,N]`, would it still be needed?
6. With causal, N = 4096, Br = 64, Bc = 128, how many K blocks does Q block 5 (rows 320–383) process? Which ones need a mask?

<details>
<summary>Answers</summary>

1. FA1: read and written once per K/V block, Tc = ⌈N/Bc⌉ times in total. FA2: O and LSE are written only once, and ℓ and m never go to HBM. The cost is that every Q block reads all of K/V, so K/V are read from global memory ⌈N/Br⌉ times; most of these reads hit in L2 (in §7.3, DRAM reads are only one each of Q/K/V).
2. g = 13/4 = 3, q = 1: rows 3 and 11; the columns of each row are 2q, 2q+1 = 2, 3 within each n8 tile, that is, keys 2, 3, 10, 11, 18, 19, …, 58, 59 (16 in total). It shuffles with lanes 12, 14 and 15 of the same quad (xor 1 and xor 2).
3. In the exponent `2^(s − m)`, all lanes must use the same m; otherwise the P~ of the same row are scaled by different factors and cannot be added. ℓ is only scaled by α and then increased by this lane's P~; multiplication distributes over addition, so the 4 partial sums can each be scaled and added at the end (§3.3).
4. Each warp sees only some of the keys: the row maximum must first be computed within the warp, written to shared memory, followed by `__syncthreads()`, and then all warps' values are read back to compute the overall maximum; the same for the row sum; each warp's P·V is only a partial sum and must be added together through shared memory, with another synchronization. FA2's split-Q needs none of this.
5. The mma B register needs the same n and two adjacent k. In QKᵀ, k is D, and K is contiguous in D, so it is read directly; in PV, k is the key, and V is not contiguous along the key, so it must be transposed. If V were stored as `[D][N]` (key contiguous), it would be K-major and the plain `ldmatrix` would work.
6. Keys ≤ 383 are needed: `min(32, ⌈384/128⌉) = 3` blocks (keys 0–383). A block needs a mask if its last key is greater than q0 = 320: block 2 (keys 256–383) has last key 383 > 320, so it needs a mask; blocks 0 and 1 do not. Here q0 is not a multiple of Bc, so the diagonal falls in the middle of block 2.
</details>

## 9. Further reading

- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023): section 3.1 (fewer non-matmul operations, saving only the logsumexp), section 3.2 (parallelism over the sequence length), section 3.3 (partitioning Q across warps), Algorithm 1
- PTX ISA: [Matrix Fragments for mma.m16n8k16 with floating point type](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float), [ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix), [cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async), [ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2)
- Upstream FA2 (`third_party/flash-attention`, commit `1fc0221`):
  - `csrc/flash_attn/src/flash_fwd_kernel.h`: `compute_attn_1rowblock` (one CTA processes one Q block; `n_block_max` implements causal block skipping; blocks that need a mask are processed first, then the rest. This lab walks the K blocks front to back so that the same loop also works for the step 4 pipeline; see NOTES.md)
  - `csrc/flash_attn/src/softmax.h`: `Softmax::softmax_rescale_o` (the update of §3.1), `normalize_softmax_lse` (the final division and LSE), `quad_allreduce_` (quad shuffle)
  - `csrc/flash_attn/src/utils.h`: `convert_layout_acc_Aregs` (§4.3: reinterpreting a C fragment as an A fragment)
  - `csrc/flash_attn/src/mask.h`: `apply_mask_causal`
  - `csrc/flash_attn/src/flash_fwd_launch_template.h`: block sizes and warp counts for each head dimension
- Next: Part 3 (starting with `30_tma_mbarrier`) goes past the upper bound of §7.2 with TMA and `tcgen05`
