# 20 · warp 级 Tensor Core：mma.sync GEMM

> 部分 2 · FA2　｜　前置：[10_simt_gemm](../10_simt_gemm/README.md)　｜　代码：[`students/20_mma_gemm/kernel.cu`](../../students/20_mma_gemm/kernel.cu)

## 1. 本关目标

用 Tensor Core 计算 `C[M,N] = A[M,K] · B[N,K]ᵀ`：A、B 为 BF16，C 为 FP32。A 和 B 都按 K 连续存放（K-major），形状和 attention 里的 `Q·Kᵀ` 一样。

本关分三步，把同一个 GEMM 从 cuBLAS 的约 2% 提到约 17%：

1. 一个 warp 算一个 16×8 的输出块，直接从 global memory 按 **fragment 坐标**装操作数；
2. 一个 CTA 算 128×128 的输出块，操作数先进 shared memory，再用 **`ldmatrix`** 装进寄存器；
3. 用 **XOR swizzle** 重新排列 shared memory，消除 `ldmatrix` 的 bank conflict。

做完后你应该能：

- 说出 `mma.sync.m16n8k16` 中每个 lane 持有 A、B、C 的哪些元素；
- 写出 `ldmatrix` 每个 lane 应该提供的行地址；
- 算出一种 shared memory 布局下 `ldmatrix` 的 bank conflict，并设计 swizzle 消除它；
- 解释为什么 B200 上 `mma.sync` 最多只能跑到 cuBLAS 的约 0.37 倍。

本关写的 `ldmatrix`、`mma`、fragment 坐标和 swizzle 函数，从 21 关起由 [`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh) 提供。

## 2. 动机：GEMM 要用 Tensor Core

[10_simt_gemm](../10_simt_gemm/README.md) 用 FP32 FMA 指令（CUDA core）做 GEMM，B200 上 FP32 的标称峰值是 75 TFLOP/s。同一块 GPU 上，BF16 Tensor Core 的标称峰值是 2250 TFLOP/s，高出 30 倍。attention 里的两次矩阵乘 `S = Q·Kᵀ` 和 `O = P·V` 占了绝大部分浮点运算，必须交给 Tensor Core。

FlashAttention-2 论文第 3.1 节专门讨论了这件事：在 A100 上，矩阵乘的吞吐是非矩阵乘 FP32 运算的 16 倍，所以要尽量减少 softmax 等非矩阵乘运算，让时间花在 Tensor Core 上。FA2 的 kernel 用的就是本关的 `mma.sync` 和 `ldmatrix`（通过 CUTLASS/CuTe 封装）。22 关的 FA2 会直接用本关的 fragment 布局：`Q·Kᵀ` 的累加器 fragment 经过 softmax 后，原地变成 `P·V` 的 A 操作数。

## 3. 数学

### 3.1 分块

`C[m][n] = Σ_k A[m][k] · B[n][k]`。把三个维度都切块：

```
C 的 tile (BM × BN)  =  Σ over K tiles  A 的 tile (BM × BK) · B 的 tile (BN × BK)ᵀ
```

Tensor Core 指令一次算一个固定形状的小块。本关用 **m16n8k16**：

```
D[16×8] += A[16×16] · B[16×8]          （B 在这里按 k×n 理解）
```

一条指令做 16·8·16 = 2048 次乘加，即 4096 FLOP。一个 warp 的 32 个线程**一起**执行这条指令，每个线程只持有操作数的一小部分，叫做 **fragment**。

### 3.2 K-major 的 B

我们的 B 存为 `[N][K]`。`mma` 的 `.col` 表示 B 操作数“按列存放”：B 作为 k×n 矩阵，同一列（固定 n）的 k 个元素连续。`B[n][k]` 正是这样的布局。所以 **K-major 的 B 不需要转置**就是 `mma` 要的操作数，这和 attention 里的 K 矩阵一样。（`P·V` 中 V 是 `[N][D]`，作为 B 操作数时 n 维连续，需要转置，见 §4.4 的 `.trans`。）

### 3.3 数据复用

一个 CTA 计算 BM×BN 的 C tile，每个 K tile 从 global 读 `(BM + BN) · BK` 个元素，做 `2 · BM · BN · BK` 次浮点运算。BM = BN = 128、BK = 32 时：

```
读入 (128 + 128) · 32 · 2 B = 16 KB，   计算 2 · 128 · 128 · 32 = 1 MFLOP，   64 FLOP/B
```

4096³ 的 GEMM 一共 137 GFLOP，按 64 FLOP/B 需要从 L2 读约 2.1 GB。和 10 关一样，tile 越大复用越多。

## 4. 硬件与指令

### 4.1 三代 Tensor Core 指令

| 指令 | 架构 | 谁发起 | 操作数从哪来 | 累加器在哪 |
|---|---|---|---|---|
| `mma.sync` | Ampere 起（含 SM100） | 一个 warp | 寄存器 | 寄存器 |
| `wgmma.mma_async` | 仅 Hopper（SM90） | 4 个 warp | shared memory / 寄存器 | 寄存器 |
| `tcgen05.mma` | Blackwell（SM100） | 一个线程 | shared memory / TMEM | TMEM |

B200 仍然支持 `mma.sync`，但它是“旧路径”。在本机上用一个只发 `mma.sync` 的微基准测得：

| | BF16 吞吐 | 相对标称 2250 TFLOP/s |
|---|---|---|
| `mma.sync` 上限（微基准，无访存） | ≈ 560 TFLOP/s | ≈ 25% |
| cuBLAS（`tcgen05`），4096³ | ≈ 1440 TFLOP/s | ≈ 64% |
| cuBLAS（`tcgen05`），8192×8192×4096 | ≈ 1550 TFLOP/s | ≈ 69% |

也就是说，**一个完美的 `mma.sync` kernel 也只能跑到 cuBLAS 的约 0.36–0.39 倍**。本关和 21 关的性能都应该对照这条“`mma.sync` roofline”来看。部分 3 学 `tcgen05` 就是为了跨过这条线。那为什么还要学 `mma.sync`？因为 fragment 布局、`ldmatrix`、swizzle、`cp.async` 流水这些概念是后面所有内容的基础，而且 `mma.sync` 的同步模型最简单：指令返回时结果就在寄存器里。

换算到每个 SM：560 TFLOP/s ÷ 148 SM ÷ 1.965 GHz ≈ 1920 FLOP/周期/SM。每个 SM 有 4 个 warp 调度器（SMSP），所以**每个 SMSP 大约每 8 个周期完成一条 m16n8k16**。

### 4.2 `mma.sync` 的语法

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])   // D = C，读写
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),        // A：4 个 32 位寄存器
      "r"(b[0]), "r"(b[1]));                              // B：2 个 32 位寄存器
```

| 部分 | 含义 |
|---|---|
| `.sync.aligned` | warp 内 32 个 lane 必须一起执行同一条指令（不能在分支里只让部分 lane 执行） |
| `.m16n8k16` | 形状 |
| `.row.col` | A 按行（k 连续），B 按列（k 连续）。BF16 只支持 `.row.col` |
| `.f32.bf16.bf16.f32` | D、A、B、C 的类型 |

每个 32 位寄存器装两个 BF16：**k（或列）较小的元素在低 16 位**。这正好是内存中两个相邻 BF16 的顺序，所以从内存读一个 `uint32_t` 就得到一个合法的操作数寄存器。

### 4.3 fragment 布局

记 `lane = threadIdx.x % 32`，`group = lane / 4`（0..7），`quad = lane % 4`（0..3）。

**A（16×16，行主序）**：每个 lane 4 个寄存器，寄存器 i 装 `A[row][col]` 和 `A[row][col+1]`：

| 寄存器 | a0 | a1 | a2 | a3 |
|---|---|---|---|---|
| row | group | group + 8 | group | group + 8 |
| col | 2·quad | 2·quad | 2·quad + 8 | 2·quad + 8 |

```
           col 0..7                col 8..15
         ┌───────────────────┬───────────────────┐
row 0..7 │ a0: lane 4r+q 拿   │ a2                │
         │     (r, 2q..2q+1)  │                   │
row 8..15│ a1                │ a3                │
         └───────────────────┴───────────────────┘
```

**B（16×8，按 k×n 看）**：每个 lane 2 个寄存器，寄存器 i 装 `B[k][n]` 和 `B[k+1][n]`：`k = 2·quad + 8i`，`n = group`。换成我们的存储 `B[n][k]`：lane 读的是第 `group` 行、第 `2·quad + 8i` 列开始的两个元素。

**C/D（16×8）**：每个 lane 4 个 float，第 i 个是 `D[row][col]`：

| float | d0 | d1 | d2 | d3 |
|---|---|---|---|---|
| row | group | group | group + 8 | group + 8 |
| col | 2·quad | 2·quad + 1 | 2·quad | 2·quad + 1 |

**手算一个 lane**：lane 5，group = 1，quad = 1。

| 操作数 | lane 5 持有的元素 |
|---|---|
| a0, a1, a2, a3 | `A[1][2..3]`，`A[9][2..3]`，`A[1][10..11]`，`A[9][10..11]` |
| b0, b1 | `B[n=1][k=2..3]`，`B[n=1][k=10..11]`（按 `[N][K]` 存储） |
| d0..d3 | `D[1][2]`，`D[1][3]`，`D[9][2]`，`D[9][3]` |

注意 A 和 C 的“行 = group、group+8”是同一个规律：一个 lane 在 A 中负责的行就是它在 C 中负责的行。22 关会利用这一点让 softmax 在寄存器里按行进行。

`one_atom` case（16×8×16，小整数输入）只有一条 mma。任何一个坐标写错，结果都不会精确相等，报告会给出第一个错的 `[m, n]`。

### 4.4 `ldmatrix`

step 1 里每个 lane 自己从 global 读 fragment：一个 lane 的 4 个 A 寄存器来自 4 个不同位置，每次只读 4 字节，访存非常零碎。`ldmatrix` 让 warp 一起从 shared memory 装 fragment：

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

- 装 1/2/4 个（`.x1/.x2/.x4`）8×8 的 16 位矩阵。每个矩阵的一**行**是 16 个连续字节，各行地址可以任意（16 字节对齐）。
- **lane 8i .. 8i+7 提供第 i 个矩阵第 0..7 行的地址**。`.x1` 只用 lane 0–7 的地址，`.x2` 用 lane 0–15。
- 结果：lane l 的 `r[i]` 装第 i 个矩阵的第 `l/4` 行、第 `2(l%4)` 和 `2(l%4)+1` 个元素。

把这个结果和 §4.3 比较：一个 8×8 矩阵经 `ldmatrix` 后的分布，正好是 A 的一个寄存器（例如 a0 = 左上 8×8）或 B 的一个寄存器的分布。所以 16×16 的 A 用一条 `.x4` 就能装满 a0..a3，只要让 4 个矩阵依次是左上、左下、右上、右下。

**`.trans`**：装载时把每个 8×8 矩阵转置，lane l 的 `r[i]` 装的是第 `2(l%4)`、`2(l%4)+1` 行的第 `l/4` 列。当 B 操作数在 shared memory 中是 n 连续（例如 attention 的 V：`[N_kv][D]`，作为 `P·V` 的 B 时 D 是 n 维）时，用 `.trans` 就能得到 `.col` 布局的 fragment。本关的 B 是 K-major，不需要 `.trans`；22 关会用到。

**常见误用**：

| 误用 | 后果 |
|---|---|
| 只在部分 lane 上执行（放在 `if (lane < 16)` 里） | 未定义行为，可能挂起 |
| 传 generic 指针当地址 | 地址错，读到别处；要用 `__cvta_generic_to_shared` 转成 32 位 shared 地址 |
| 行地址不是 16 字节对齐 | 未定义行为 |
| `.x4` 时 lane 16–31 的地址没有设好 | 第 2、3 个矩阵读错 |

### 4.5 bank conflict 与 XOR swizzle

（bank 的定义见 [00 关 §4.3](../00_transpose/README.md#43-shared-memory-与-bank-conflict)。）shared memory 每 128 字节覆盖 32 个 bank 各一次。把 128 字节分成 8 个 16 字节的“槽”（每槽 4 个 bank）。`ldmatrix` 的一个 8×8 矩阵读 8 行 × 16 字节 = 128 字节：**8 行落在 8 个不同的槽**时一次完成（1 个 wavefront），否则要分多次。

step 2 的 shared tile 是 `[128][32]` 的 BF16，每行 64 字节 = 4 个 16 字节 chunk。第 r 行第 c 个 chunk 的字节地址是 `64r + 16c`，所在的槽是 `(4r + c) % 8`。`ldmatrix` 读同一列 chunk（固定 c）的 8 行：

| 行 r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 槽（c = 0，不 swizzle） | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 4 |

8 行只用了 2 个槽，每个槽被 4 行争用：**4 路 bank conflict**，一个矩阵要 4 个 wavefront。

XOR swizzle：不移动 chunk 内的数据，只把第 r 行的逻辑 chunk c 存到物理 chunk `c ^ f(r)`。选 `f(r) = (r >> 1) & 3`：

| 行 r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `(r>>1)&3` | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| 物理 chunk（c = 0） | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| 槽 = `4(r&1) + 物理 chunk` | 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |

8 行落在 8 个不同的槽。对任意 c 都成立，因为 XOR 一个固定值是 0..3 上的置换。写入时（每个线程存一个 16 字节 chunk）同样无冲突。

和 00 关的 padding 相比，swizzle 不浪费 shared memory，而且保持每行 16 字节对齐，这是 `ldmatrix`（以及后面的 `cp.async`、TMA）的要求。CUTLASS 把这类布局写作 `Swizzle<B, M, S>`；30 关的 TMA 也有硬件 swizzle 模式（128B swizzle 对应 8 个 chunk 的 `c ^ (r & 7)`），原理相同。

## 5. 调度设计

### 5.1 step 1：一个 warp 一个 16×8

```
block = 4 warps（128 线程），沿 M 叠放
grid  = (N/8, ceil(M/16 / 4))
warp w 负责 C[m0 .. m0+15][n0 .. n0+7]，m0 = (blockIdx.y·4 + w)·16，n0 = blockIdx.x·8
for k0 in 0, 16, ...:   从 global 读 af[4]、bfr[2]；mma
写回 d[4]
```

每条 mma 需要 6 次 4 字节的 global 读，几乎没有复用：这一步只用来验证 fragment 坐标。它要求 M、N、K 是 16、8、16 的倍数（`make_plan` 会检查），所以 step 1 只运行 `one_atom`、`atoms`、`long_k`、`aligned` 四个 case。

### 5.2 step 2、3：128×128 CTA tile

```
CTA tile 128×128，BK = 32，8 个 warp 排成 2×4，每个 warp 64×32
           ┌──────┬──────┬──────┬──────┐
 rows 0-63 │ w0   │ w1   │ w2   │ w3   │   每个 warp：4 (M) × 4 (N) 个 m16n8 块
rows 64-127│ w4   │ w5   │ w6   │ w7   │   累加器 4·4·4 = 64 个 float / lane
           └──────┴──────┴──────┴──────┘
             32 列  32 列  32 列  32 列
for k0 in 0, 32, ...:
    load_tile：global → 寄存器 → shared（A、B 各 128×32，每线程 2 个 16 字节 chunk）
    __syncthreads()
    for kk in 0, 16:
        A：4 条 ldmatrix.x4（每个 m16 一条）
        B：2 条 ldmatrix.x4（每条覆盖两个 n8）
        16 条 mma
    __syncthreads()
写回：每个 lane 64 个 float，按 frag_c_coord 定位
```

| 资源 | 用量 |
|---|---|
| shared memory | A、B 各 128×32×2 B = 8 KB，共 16 KB |
| 寄存器（每 lane） | 累加器 64 + A fragment 16 + B fragment 8 + 地址；答案约 125 个 |
| 每个 k16：ldmatrix / mma | 6 / 16 |

**A 的 ldmatrix 地址**：第 i 个 m16 块是 tile 中的行 `wm + 16i ..`、列 `kk ..`。要让 r0..r3 依次是左上、左下、右上、右下 8×8（即 a0..a3），lane l 应指向：

| lane | 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| 矩阵 | 左上（a0） | 左下（a1） | 右上（a2） | 右下（a3） |
| 行 | `l` | `l` | `l - 16` | `l - 16` |
| 列 | `kk` | `kk` | `kk + 8` | `kk + 8` |

（行都相对 `wm + 16i`。）用一个式子写出来就是 TODO(2b) 的第 1 步。

**B 的 ldmatrix 地址**：B tile 在 shared memory 中是 `[n][k]`。一个 n8 块的 b0 是 `n = 0..7, k = 0..7` 的 8×8，b1 是 `n = 0..7, k = 8..15`。一条 `.x4` 装两个 n8 块 j、j+1，让 4 个矩阵依次是 `(n 0–7, k 0–7)`、`(n 0–7, k 8–15)`、`(n 8–15, k 0–7)`、`(n 8–15, k 8–15)`，就得到 `bfr[j][0], bfr[j][1], bfr[j+1][0], bfr[j+1][1]`。每个 lane 的行（n）和列（k）请自己推出来。

所有 shared memory 地址都经过 `tile_offset<kSwizzle>(row, col)`：step 2 是普通行主序，step 3 会调用你写的 swizzle。`load_tile` 写入和 `ldmatrix` 读取用的是同一个函数，所以只要 swizzle 是每行内的置换，结果就一定正确，只是快慢不同。

## 6. 分步练习

所有命令在仓库根目录运行。

### 步骤 1 · TODO(1)：fragment 坐标与 mma

- TODO(1a)：`mma_m16n8k16_bf16` 的内联汇编（§4.2）；
- TODO(1b)：`frag_a_coord`、`frag_b_coord`、`frag_c_coord`（§4.3 的表格，每个一行）；
- TODO(1c)：`mma_naive` 的 K 循环和写回（§5.1）。

```bash
./lab check 20 --step 1
./lab bench 20 --step 1
```

### 步骤 2 · TODO(2)：shared memory 与 ldmatrix

- TODO(2a)：`ldmatrix_x4` 的内联汇编（§4.4）；
- TODO(2b)：每个 k16 片的 ldmatrix 和 mma（§5.2 的两张地址表）；
- TODO(2c)：写回，跳过越界元素。

```bash
./lab check 20 --step 2
./lab bench 20 --step 2
./lab profile 20 --step 2
```

在 profile 输出里看 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`（`ldmatrix` 计入 shared load）和 `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`。按 §4.5 的分析，wavefront 数应该是“理想值”的 4 倍。理想值可以手算：4096³ 时一共 1024 个 CTA × 128 个 K tile × 8 个 warp × 2 个 k16 × 6 条 `ldmatrix.x4` × 4 个 wavefront。

### 步骤 3 · TODO(3)：XOR swizzle

实现 `swizzle_chunk4(row, chunk)`（§4.5）。只改这一个函数。

```bash
./lab check 20
./lab bench 20 --step 3
./lab profile 20 --step 3
```

bank conflict 应该降到接近 0，wavefront 数降为 step 2 的 1/4。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `one_atom` 失败，错误集中在右半（列 4–7）或下半（行 8–15） | fragment 坐标里 group/quad 用反了，或 +8 加在了错误的寄存器上 |
| `one_atom` 失败，结果像是 A·B 而不是 A·Bᵀ | B 的地址按 `[K][N]` 算了；我们的 B 是 `[N][K]` |
| 结果每两列互换 | 寄存器里两个 BF16 的高低位顺序弄反（例如手工拼 `uint32_t` 时） |
| step 2 `aligned` 失败但 `one_atom` 通过 | ldmatrix 的 lane 地址错：检查 lane 16–31 是否指向 `kk + 8` |
| step 2 `tail`、`tiny` 失败 | 写回没判断越界，或 `load_tile` 的补零被改坏 |
| `illegal instruction` 或挂起 | `mma`/`ldmatrix` 放在了只有部分 lane 执行的分支里 |
| step 3 结果错 | swizzle 不是置换（两个逻辑 chunk 映射到同一个物理 chunk） |
| step 3 没变快 | profile 看 bank conflict 是否下降；swizzle 是否只依赖 row 的第 1–2 位 |

## 7. 性能

本机实测（CUDA Graph 计时；`vs cublas` 是 cuBLAS BF16→FP32 的耗时除以本版本耗时；“占 mma.sync 上限”是 TFLOP/s 除以 §4.1 的 560 TFLOP/s）：

| 版本 | `square` 4096³ `vs cublas` | `wide` 8192×8192×4096 `vs cublas` | 占 mma.sync 上限 |
|---|---|---|---|
| 步骤 1 | ≈ 0.02× | ≈ 0.01× | ≈ 4% |
| 步骤 2 | ≈ 0.09× | ≈ 0.09× | ≈ 24% |
| 步骤 3 | ≈ 0.17× | ≈ 0.17× | ≈ 43–46% |
| mma.sync 上限 | ≈ 0.39× | ≈ 0.36× | 100% |

步骤 3 达到 cuBLAS 的 0.12 倍以上（即 mma.sync 上限的约 30%）就说明机制成立。

profile 指标（步骤 2 → 步骤 3，`square`）：

| 指标 | 步骤 2 | 步骤 3 |
|---|---|---|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | ≈ 2.0 亿 | ≈ 0.50 亿（= 理想值） |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈ 1.5 亿 | ≈ 1 万 |

步骤 3 仍只有上限的一半左右。原因是 K 循环是同步的：所有 warp 先等 global 读完，再一起算，算的时候没人读 global。`sm__pipe_tensor_cycles_active` 指标能直接看到 Tensor Core 有多少时间在空等。21 关用 `cp.async` 让搬运和计算重叠。

## 8. 自测问题

1. lane 13 在 m16n8k16 中持有 A 的哪 8 个元素？持有 D 的哪 4 个元素？
2. 为什么 K-major 的 B（`[N][K]`）用 `ldmatrix` 时不需要 `.trans`，而 attention 中 `P·V` 的 V（`[N_kv][D]`）需要？
3. 如果 shared tile 每行是 128 字节（BK = 64），不 swizzle 时 `ldmatrix` 是几路冲突？应该用什么 swizzle？
4. step 2 的 wavefront 恰好是理想值的 4 倍。B 的 `ldmatrix` 也是 4 路冲突吗？为什么？
5. 只把 `load_tile` 写入的位置做 swizzle、`ldmatrix` 读取时忘了 swizzle，会发生什么？`one_atom` 能发现吗？

<details>
<summary>参考答案</summary>

1. group = 3，quad = 1。A：`A[3][2..3]`、`A[11][2..3]`、`A[3][10..11]`、`A[11][10..11]`。D：`D[3][2]`、`D[3][3]`、`D[11][2]`、`D[11][3]`。
2. `.col` 要求同一个 n 的 k 个元素连续。`B[N][K]` 中固定 n、k 连续，满足；V 作为 B 时 k 是 `N_kv`、n 是 D，`V[N_kv][D]` 中连续的是 D（n），所以要在装载时转置。
3. 第 r 行第 c 个 chunk 在槽 `(8r + c) % 8 = c`，8 行全在同一个槽：8 路冲突。用 `c ^ (r & 7)`：8 行的物理 chunk 两两不同。这就是 `lab::swizzle_chunk<8>`，21 关 BK = 64 时用到。
4. 是。B tile 也是 `[128][32]` 的 64 字节行，B 的每个 8×8 矩阵同样读同一 chunk 列的 8 个连续行，分析和 A 一样。
5. 读到的 chunk 和写入的不对应，结果错误。`one_atom` 只有 16 行 × 16 列，K = 16 时只用了 chunk 0、1，而第 0、1 行的 XOR 值是 0，前两行不受影响，但第 2 行起就会错，所以 `one_atom` 能发现；用 `atoms` 和 `aligned` 更容易看出规律。
</details>

## 9. 延伸阅读

- PTX ISA：[Warp Level Matrix Multiply-Accumulate Instructions: mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)，其中 *Matrix Fragments for mma.m16n8k16 with floating point type* 一节有本关 §4.3 的原图；[ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)
- CUDA C++ Programming Guide：[Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)（bank 与 wavefront）
- Tri Dao，*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*，第 3.1 节（为什么要减少非矩阵乘运算）
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cute/arch/mma_sm80.hpp`：`SM80_16x8x16_F32BF16BF16F32_TN`，即本关的 mma；
  - `third_party/cutlass/include/cute/arch/copy_sm75.hpp`：`SM75_U32x4_LDSM_N`（`ldmatrix.x4`）与 `SM75_U16x8_LDSM_T`（`.trans`）；
  - `third_party/flash-attention/csrc/flash_attn/src/kernel_traits.h`：FA2 的 shared memory 布局 `Swizzle<kSwizzle, 3, 3>`。
