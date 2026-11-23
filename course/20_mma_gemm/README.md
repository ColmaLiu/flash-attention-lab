# 20 · Warp-level Tensor Cores: mma.sync GEMM

> Part 2 · FA2 | Prerequisites: [10_simt_gemm](../10_simt_gemm/README.md) | Code: [`students/20_mma_gemm/kernel.cu`](../../students/20_mma_gemm/kernel.cu)

## 1. Goals

Compute `C[M,N] = A[M,K] · B[N,K]ᵀ` with Tensor Cores: A and B are BF16, C is FP32. Both A and B are stored contiguous along K (K-major), the same shape as `Q·Kᵀ` in attention.

This lab has three steps, which take the same GEMM from about 2% of cuBLAS to about 17%:

1. one warp computes one 16×8 output block, loading operands directly from global memory by **fragment coordinates**;
2. one CTA computes a 128×128 output block; operands first go to shared memory, then **`ldmatrix`** loads them into registers;
3. an **XOR swizzle** rearranges shared memory to remove the bank conflicts of `ldmatrix`.

After this lab you should be able to:

- state which elements of A, B, C each lane holds in `mma.sync.m16n8k16`;
- write the row address each lane must supply to `ldmatrix`;
- compute the `ldmatrix` bank conflicts of a shared memory layout, and design a swizzle that removes them;
- explain why `mma.sync` on B200 can reach at most about 0.37 times cuBLAS.

The `ldmatrix`, `mma`, fragment coordinate and swizzle functions you write in this lab are provided by [`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh) from lab 21 on.

## 2. Motivation: GEMM needs Tensor Cores

[10_simt_gemm](../10_simt_gemm/README.md) does GEMM with FP32 FMA instructions (CUDA cores); the FP32 nominal peak of B200 is 75 TFLOP/s. On the same GPU, the BF16 Tensor Core nominal peak is 2250 TFLOP/s, 30 times higher. The two matrix multiplications in attention, `S = Q·Kᵀ` and `O = P·V`, account for most of the floating-point operations and must go to Tensor Cores.

Section 3.1 of the FlashAttention-2 paper discusses exactly this: on A100, matmul throughput is 16 times that of non-matmul FP32 operations, so non-matmul operations such as the softmax should be minimized and the time should go to Tensor Cores. The FA2 kernel uses exactly this lab's `mma.sync` and `ldmatrix` (wrapped by CUTLASS/CuTe). The FA2 of lab 22 uses this lab's fragment layout directly: after the softmax, the accumulator fragment of `Q·Kᵀ` becomes the A operand of `P·V` in place.

## 3. Math

### 3.1 Tiling

`C[m][n] = Σ_k A[m][k] · B[n][k]`. Tile all three dimensions:

```
tile of C (BM × BN)  =  Σ over K tiles  tile of A (BM × BK) · tile of B (BN × BK)ᵀ
```

A Tensor Core instruction computes a small block of fixed shape at a time. This lab uses **m16n8k16**:

```
D[16×8] += A[16×16] · B[16×8]          (B here is viewed as k×n)
```

One instruction does 16·8·16 = 2048 multiply-adds, that is, 4096 FLOP. The 32 threads of a warp execute this instruction **together**, and each thread holds only a small part of the operands, called a **fragment**.

### 3.2 K-major B

Our B is stored as `[N][K]`. The `.col` of `mma` means the B operand is "stored by column": viewing B as a k×n matrix, the k elements of one column (fixed n) are contiguous. `B[n][k]` has exactly this layout. So **K-major B needs no transpose** to be the operand `mma` wants, just like the K matrix in attention. (In `P·V`, V is `[N][D]`; as the B operand its n dimension is contiguous, so it needs a transpose; see `.trans` in §4.4.)

### 3.3 Data reuse

A CTA computes a BM×BN tile of C. For each K tile it reads `(BM + BN) · BK` elements from global and does `2 · BM · BN · BK` floating-point operations. With BM = BN = 128 and BK = 32:

```
read (128 + 128) · 32 · 2 B = 16 KB,   compute 2 · 128 · 128 · 32 = 1 MFLOP,   64 FLOP/B
```

A 4096³ GEMM is 137 GFLOP in total, and at 64 FLOP/B it needs to read about 2.1 GB from L2. As in lab 10, larger tiles give more reuse.

## 4. Hardware and instructions

### 4.1 Three generations of Tensor Core instructions

| instruction | architecture | issued by | operands from | accumulator in |
|---|---|---|---|---|
| `mma.sync` | Ampere and later (including SM100) | one warp | registers | registers |
| `wgmma.mma_async` | Hopper only (SM90) | 4 warps | shared memory / registers | registers |
| `tcgen05.mma` | Blackwell (SM100) | one thread | shared memory / TMEM | TMEM |

B200 still supports `mma.sync`, but it is the "old path". A microbenchmark that issues only `mma.sync` measures:

| | BF16 throughput | relative to nominal 2250 TFLOP/s |
|---|---|---|
| `mma.sync` upper bound (microbenchmark, no memory access) | ≈ 560 TFLOP/s | ≈ 25% |
| cuBLAS (`tcgen05`), 4096³ | ≈ 1440 TFLOP/s | ≈ 64% |
| cuBLAS (`tcgen05`), 8192×8192×4096 | ≈ 1550 TFLOP/s | ≈ 69% |

In other words, **even a perfect `mma.sync` kernel can reach only about 0.36–0.39 times cuBLAS**. Read the performance of this lab and of lab 21 against this "`mma.sync` roofline". Part 3 teaches `tcgen05` precisely to get past this line. Then why learn `mma.sync` at all? Because fragment layouts, `ldmatrix`, swizzle and `cp.async` pipelines are the foundation of everything that follows, and `mma.sync` has the simplest synchronization model: when the instruction returns, the result is in registers.

Per SM: 560 TFLOP/s ÷ 148 SM ÷ 1.965 GHz ≈ 1920 FLOP/cycle/SM. Each SM has 4 warp schedulers (SMSPs), so **each SMSP completes roughly one m16n8k16 every 8 cycles**.

### 4.2 `mma.sync` syntax

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])   // D = C, read-write
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),        // A: 4 32-bit registers
      "r"(b[0]), "r"(b[1]));                              // B: 2 32-bit registers
```

| part | meaning |
|---|---|
| `.sync.aligned` | all 32 lanes of the warp must execute the same instruction together (it cannot run on only some lanes inside a branch) |
| `.m16n8k16` | shape |
| `.row.col` | A by row (k contiguous), B by column (k contiguous). BF16 supports only `.row.col` |
| `.f32.bf16.bf16.f32` | types of D, A, B, C |

Each 32-bit register holds two BF16 values: **the element with the smaller k (or column) is in the low 16 bits**. This is exactly the order of two adjacent BF16 values in memory, so reading one `uint32_t` from memory gives a valid operand register.

### 4.3 Fragment layout

Let `lane = threadIdx.x % 32`, `group = lane / 4` (0..7), `quad = lane % 4` (0..3).

**A (16×16, row-major)**: 4 registers per lane; register i holds `A[row][col]` and `A[row][col+1]`:

| register | a0 | a1 | a2 | a3 |
|---|---|---|---|---|
| row | group | group + 8 | group | group + 8 |
| col | 2·quad | 2·quad | 2·quad + 8 | 2·quad + 8 |

```
           col 0..7              col 8..15
         ┌─────────────────────┬─────────────────────┐
row 0..7 │ a0: lane 4r+q holds │ a2                  │
         │     (r, 2q..2q+1)   │                     │
row 8..15│ a1                  │ a3                  │
         └─────────────────────┴─────────────────────┘
```

**B (16×8, viewed as k×n)**: 2 registers per lane; register i holds `B[k][n]` and `B[k+1][n]`: `k = 2·quad + 8i`, `n = group`. In our storage `B[n][k]`: the lane reads the two elements starting at row `group`, column `2·quad + 8i`.

**C/D (16×8)**: 4 floats per lane; the i-th is `D[row][col]`:

| float | d0 | d1 | d2 | d3 |
|---|---|---|---|---|
| row | group | group | group + 8 | group + 8 |
| col | 2·quad | 2·quad + 1 | 2·quad | 2·quad + 1 |

**One lane by hand**: lane 5, group = 1, quad = 1.

| operand | elements held by lane 5 |
|---|---|
| a0, a1, a2, a3 | `A[1][2..3]`, `A[9][2..3]`, `A[1][10..11]`, `A[9][10..11]` |
| b0, b1 | `B[n=1][k=2..3]`, `B[n=1][k=10..11]` (stored as `[N][K]`) |
| d0..d3 | `D[1][2]`, `D[1][3]`, `D[9][2]`, `D[9][3]` |

The "rows = group, group+8" of A and of C follow the same rule: the rows a lane is responsible for in A are the rows it is responsible for in C. Lab 22 uses this to run the softmax row by row in registers.

The `one_atom` case (16×8×16, small integer inputs) has only one mma. If any coordinate is wrong, the result is not exactly equal, and the report gives the first wrong `[m, n]`.

### 4.4 `ldmatrix`

In step 1 each lane reads its fragment from global on its own: the 4 A registers of a lane come from 4 different locations, each read is only 4 bytes, and the accesses are very fragmented. `ldmatrix` lets the warp load fragments from shared memory together:

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

- It loads 1/2/4 (`.x1/.x2/.x4`) 8×8 matrices of 16-bit elements. One **row** of each matrix is 16 contiguous bytes, and each row address can be arbitrary (16-byte aligned).
- **Lanes 8i .. 8i+7 supply the addresses of rows 0..7 of matrix i**. `.x1` uses only the addresses of lanes 0–7, `.x2` uses lanes 0–15.
- Result: `r[i]` of lane l holds elements `2(l%4)` and `2(l%4)+1` of row `l/4` of matrix i.

Compare this result with §4.3: the distribution of an 8×8 matrix after `ldmatrix` is exactly the distribution of one A register (for example a0 = the top-left 8×8) or one B register. So one `.x4` fills a0..a3 of a 16×16 A, as long as the 4 matrices are top-left, bottom-left, top-right, bottom-right in that order.

**`.trans`**: transposes each 8×8 matrix during the load; `r[i]` of lane l holds column `l/4` of rows `2(l%4)`, `2(l%4)+1`. When the B operand is n-contiguous in shared memory (for example V in attention: `[N_kv][D]`, where D is the n dimension when V is the B of `P·V`), `.trans` gives a fragment in `.col` layout. The B of this lab is K-major and needs no `.trans`; lab 22 uses it.

**Common misuses**:

| misuse | consequence |
|---|---|
| executing on only some lanes (inside `if (lane < 16)`) | undefined behavior, may hang |
| passing a generic pointer as the address | wrong address, reads from elsewhere; convert to a 32-bit shared address with `__cvta_generic_to_shared` |
| row address not 16-byte aligned | undefined behavior |
| addresses of lanes 16–31 not set for `.x4` | matrices 2 and 3 are read wrong |

### 4.5 Bank conflicts and XOR swizzle

(For the definition of banks, see [lab 00 §4.3](../00_transpose/README.md#43-shared-memory-and-bank-conflicts).) Every 128 bytes of shared memory cover each of the 32 banks once. Divide the 128 bytes into 8 "slots" of 16 bytes (4 banks per slot). One 8×8 matrix of `ldmatrix` reads 8 rows × 16 bytes = 128 bytes: when **the 8 rows fall into 8 different slots**, it completes in one pass (1 wavefront); otherwise it takes several.

The shared tile of step 2 is a `[128][32]` BF16 array; each row is 64 bytes = 4 chunks of 16 bytes. The byte address of chunk c of row r is `64r + 16c`, and its slot is `(4r + c) % 8`. `ldmatrix` reads 8 rows of the same chunk column (fixed c):

| row r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| slot (c = 0, no swizzle) | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 4 |

The 8 rows use only 2 slots, each contended by 4 rows: a **4-way bank conflict**, so one matrix takes 4 wavefronts.

XOR swizzle: data inside a chunk does not move; logical chunk c of row r is stored in physical chunk `c ^ f(r)`. Choose `f(r) = (r >> 1) & 3`:

| row r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `(r>>1)&3` | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| physical chunk (c = 0) | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| slot = `4(r&1) + physical chunk` | 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |

The 8 rows fall into 8 different slots. This holds for any c, because XOR with a fixed value is a permutation of 0..3. Writes (each thread stores one 16-byte chunk) are also conflict-free.

Compared with the padding of lab 00, swizzle wastes no shared memory and keeps every row 16-byte aligned, which `ldmatrix` (and later `cp.async` and TMA) requires. CUTLASS writes this kind of layout as `Swizzle<B, M, S>`; the TMA of lab 30 also has hardware swizzle modes (128B swizzle corresponds to `c ^ (r & 7)` over 8 chunks), on the same principle.

## 5. Schedule design

### 5.1 Step 1: one 16×8 per warp

```
block = 4 warps (128 threads), stacked along M
grid  = (N/8, ceil(M/16 / 4))
warp w owns C[m0 .. m0+15][n0 .. n0+7], m0 = (blockIdx.y·4 + w)·16, n0 = blockIdx.x·8
for k0 in 0, 16, ...:   read af[4], bfr[2] from global; mma
write back d[4]
```

Each mma needs 6 global reads of 4 bytes, with almost no reuse: this step only verifies the fragment coordinates. It requires M, N, K to be multiples of 16, 8, 16 (`make_plan` checks this), so step 1 runs only the four cases `one_atom`, `atoms`, `long_k`, `aligned`.

### 5.2 Steps 2 and 3: 128×128 CTA tile

```
CTA tile 128×128, BK = 32, 8 warps arranged 2×4, each warp 64×32
           ┌────────┬────────┬────────┬────────┐
 rows 0-63 │ w0     │ w1     │ w2     │ w3     │   per warp: 4 (M) × 4 (N) m16n8 blocks
rows 64-127│ w4     │ w5     │ w6     │ w7     │   accumulators 4·4·4 = 64 floats / lane
           └────────┴────────┴────────┴────────┘
            32 cols  32 cols  32 cols  32 cols
for k0 in 0, 32, ...:
    load_tile: global → registers → shared (A, B each 128×32, 2 chunks of 16 bytes per thread)
    __syncthreads()
    for kk in 0, 16:
        A: 4 ldmatrix.x4 (one per m16)
        B: 2 ldmatrix.x4 (each covers two n8)
        16 mma
    __syncthreads()
write back: 64 floats per lane, located with frag_c_coord
```

| resource | usage |
|---|---|
| shared memory | A, B each 128×32×2 B = 8 KB, 16 KB total |
| registers (per lane) | accumulators 64 + A fragment 16 + B fragment 8 + addresses; the solution uses about 125 |
| per k16: ldmatrix / mma | 6 / 16 |

**ldmatrix addresses for A**: the i-th m16 block covers tile rows `wm + 16i ..` and columns `kk ..`. For r0..r3 to be the top-left, bottom-left, top-right, bottom-right 8×8 in that order (that is, a0..a3), lane l should point to:

| lane | 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| matrix | top-left (a0) | bottom-left (a1) | top-right (a2) | bottom-right (a3) |
| row | `l` | `l` | `l - 16` | `l - 16` |
| col | `kk` | `kk` | `kk + 8` | `kk + 8` |

(All rows are relative to `wm + 16i`.) Writing this as one expression is item 1 of TODO(2b).

**ldmatrix addresses for B**: the B tile in shared memory is `[n][k]`. b0 of an n8 block is the 8×8 with `n = 0..7, k = 0..7`, and b1 is `n = 0..7, k = 8..15`. One `.x4` loads two n8 blocks j, j+1: with the 4 matrices in the order `(n 0–7, k 0–7)`, `(n 0–7, k 8–15)`, `(n 8–15, k 0–7)`, `(n 8–15, k 8–15)`, you get `bfr[j][0], bfr[j][1], bfr[j+1][0], bfr[j+1][1]`. Work out the row (n) and column (k) for each lane yourself.

Every shared memory address goes through `tile_offset<kSwizzle>(row, col)`: in step 2 it is plain row-major, and in step 3 it calls the swizzle you write. The `load_tile` writes and the `ldmatrix` reads use the same function, so as long as the swizzle is a permutation within each row, the result is always correct; only the speed differs.

## 6. Step-by-step exercises

Run all commands from the repository root.

### Step 1 · TODO(1): fragment coordinates and mma

- TODO(1a): the inline assembly of `mma_m16n8k16_bf16` (§4.2);
- TODO(1b): `frag_a_coord`, `frag_b_coord`, `frag_c_coord` (the tables in §4.3, one line each);
- TODO(1c): the K loop and write-back of `mma_naive` (§5.1).

```bash
./lab check 20 --step 1
./lab bench 20 --step 1
```

### Step 2 · TODO(2): shared memory and ldmatrix

- TODO(2a): the inline assembly of `ldmatrix_x4` (§4.4);
- TODO(2b): the ldmatrix and mma for each k16 slice (the two address tables in §5.2);
- TODO(2c): write-back, skipping out-of-bounds elements.

```bash
./lab check 20 --step 2
./lab bench 20 --step 2
./lab profile 20 --step 2
```

In the profile output, look at `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (`ldmatrix` counts as a shared load) and `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`. By the analysis in §4.5, the wavefront count should be 4 times the "ideal value". You can compute the ideal value by hand: for 4096³, 1024 CTAs × 128 K tiles × 8 warps × 2 k16 × 6 `ldmatrix.x4` × 4 wavefronts in total.

### Step 3 · TODO(3): XOR swizzle

Implement `swizzle_chunk4(row, chunk)` (§4.5). Change only this one function.

```bash
./lab check 20
./lab bench 20 --step 3
./lab profile 20 --step 3
```

Bank conflicts should drop to nearly 0, and the wavefront count to 1/4 of step 2.

### Common mistakes

| symptom | likely cause |
|---|---|
| `one_atom` fails, with errors concentrated in the right half (columns 4–7) or the bottom half (rows 8–15) | group/quad swapped in the fragment coordinates, or +8 added to the wrong register |
| `one_atom` fails, and the result looks like A·B instead of A·Bᵀ | B addresses computed as `[K][N]`; our B is `[N][K]` |
| every two columns are swapped | the high/low order of the two BF16 values in a register is reversed (for example when assembling a `uint32_t` by hand) |
| step 2 `aligned` fails but `one_atom` passes | wrong ldmatrix lane addresses: check that lanes 16–31 point to `kk + 8` |
| step 2 `tail`, `tiny` fail | the write-back does not check bounds, or the zero fill in `load_tile` was broken |
| `illegal instruction` or hang | `mma`/`ldmatrix` placed in a branch executed by only some lanes |
| step 3 result wrong | the swizzle is not a permutation (two logical chunks map to the same physical chunk) |
| step 3 not faster | check in the profile whether bank conflicts dropped; check that the swizzle depends only on bits 1–2 of row |

## 7. Performance

Measured (CUDA Graph timing; `vs cublas` is the time of cuBLAS BF16→FP32 divided by the time of this version; "% of mma.sync limit" is TFLOP/s divided by the 560 TFLOP/s of §4.1):

| version | `square` 4096³ `vs cublas` | `wide` 8192×8192×4096 `vs cublas` | % of mma.sync limit |
|---|---|---|---|
| step 1 | ≈ 0.02× | ≈ 0.01× | ≈ 4% |
| step 2 | ≈ 0.09× | ≈ 0.09× | ≈ 24% |
| step 3 | ≈ 0.17× | ≈ 0.17× | ≈ 43–46% |
| mma.sync limit | ≈ 0.39× | ≈ 0.36× | 100% |

Step 3 reaching 0.12 times cuBLAS or better (about 30% of the mma.sync limit) shows the mechanism works.

Profile metrics (step 2 → step 3, `square`):

| metric | step 2 | step 3 |
|---|---|---|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | ≈ 200 million | ≈ 50 million (= ideal value) |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈ 150 million | ≈ 10 thousand |

Step 3 still reaches only about half of the limit. The reason is that the K loop is synchronous: all warps first wait for the global reads to finish, then compute together, and nobody reads global while they compute. The `sm__pipe_tensor_cycles_active` metric shows directly how much of the time the Tensor Cores sit idle. Lab 21 uses `cp.async` to overlap data movement with computation.

## 8. Self-check questions

1. Which 8 elements of A does lane 13 hold in m16n8k16? Which 4 elements of D?
2. Why does K-major B (`[N][K]`) need no `.trans` with `ldmatrix`, while V (`[N_kv][D]`) in `P·V` of attention does?
3. If each row of the shared tile were 128 bytes (BK = 64), how many ways would the `ldmatrix` conflict be without swizzle? Which swizzle should be used?
4. The wavefronts of step 2 are exactly 4 times the ideal value. Is the `ldmatrix` of B also a 4-way conflict? Why?
5. If you swizzle only the positions that `load_tile` writes and forget to swizzle the `ldmatrix` reads, what happens? Can `one_atom` detect it?

<details>
<summary>Answers</summary>

1. group = 3, quad = 1. A: `A[3][2..3]`, `A[11][2..3]`, `A[3][10..11]`, `A[11][10..11]`. D: `D[3][2]`, `D[3][3]`, `D[11][2]`, `D[11][3]`.
2. `.col` requires the k elements of one n to be contiguous. In `B[N][K]`, k is contiguous for fixed n, which satisfies it; when V is the B, k is `N_kv` and n is D, and in `V[N_kv][D]` the contiguous dimension is D (n), so it must be transposed during the load.
3. Chunk c of row r is in slot `(8r + c) % 8 = c`, so all 8 rows are in the same slot: an 8-way conflict. Use `c ^ (r & 7)`: the physical chunks of the 8 rows are pairwise distinct. This is `lab::swizzle_chunk<8>`, used in lab 21 with BK = 64.
4. Yes. The B tile is also `[128][32]` with 64-byte rows, and each 8×8 matrix of B likewise reads 8 consecutive rows of the same chunk column; the analysis is the same as for A.
5. The chunks read do not match the chunks written, and the result is wrong. `one_atom` has only 16 rows × 16 columns, and with K = 16 it uses only chunks 0 and 1. The XOR value of rows 0 and 1 is 0, so the first two rows are unaffected, but from row 2 on the result is wrong, so `one_atom` does detect it; `atoms` and `aligned` make the pattern easier to see.
</details>

## 9. Further reading

- PTX ISA: [Warp Level Matrix Multiply-Accumulate Instructions: mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma); its section *Matrix Fragments for mma.m16n8k16 with floating point type* has the original figures for §4.3 of this lab; [ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)
- CUDA C++ Programming Guide: [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory) (banks and wavefronts)
- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, Section 3.1 (why to reduce non-matmul operations)
- Upstream source (pinned versions in `third_party/`):
  - `third_party/cutlass/include/cute/arch/mma_sm80.hpp`: `SM80_16x8x16_F32BF16BF16F32_TN`, the mma of this lab;
  - `third_party/cutlass/include/cute/arch/copy_sm75.hpp`: `SM75_U32x4_LDSM_N` (`ldmatrix.x4`) and `SM75_U16x8_LDSM_T` (`.trans`);
  - `third_party/flash-attention/csrc/flash_attn/src/kernel_traits.h`: the FA2 shared memory layout `Swizzle<kSwizzle, 3, 3>`.
