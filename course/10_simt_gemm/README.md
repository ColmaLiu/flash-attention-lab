# 10 · 分块 GEMM：数据复用与算术强度

> 部分 1 · FA1　｜　前置：[00_transpose](../00_transpose/README.md)　｜　代码：[`students/10_simt_gemm/kernel.cu`](../../students/10_simt_gemm/kernel.cu)

## 1. 本关目标

用 CUDA core（不用 Tensor Core）计算 FP32 矩阵乘

```
C[M, N] = A[M, K] · B[N, K]ᵀ        C[m][n] = Σ_k A[m][k] · B[n][k]
```

B 按 `[N, K]` 存储：A 和 B 都是 **K-major**（K 维连续），和 attention 中 `S = Q·Kᵀ` 的 Q、K 完全一样。本关用四个版本，把同一个 GEMM 从 cuBLAS 的约 1% 提到约 70%。做完后你应该能：

- 算出一个 kernel 的**算术强度**（每搬 1 字节做多少 FLOP），并用 roofline 判断它受计算还是受访存限制；
- 说出每一级分块（block tile、K 切片、线程 micro-tile）把数据**复用了多少次**；
- 估算 shared memory、寄存器的预算，并解释它们怎样限制 tile 大小；
- 解释为什么 K-major 的操作数放进 shared memory 时要转置，才能用 128 位访存读出。

FlashAttention 论文用 IO 复杂度说明标准 attention 慢在哪里。下一关 [11_naive_attention](../11_naive_attention/README.md) 会用本关的“复用次数”和“算术强度”来数 attention 的 HBM 流量。

## 2. 动机：从“搬数据”到“算得够多”

[00_transpose](../00_transpose/README.md) 只搬数据，不计算，所以只能追求带宽。GEMM 正好相反：每个输入元素要参与 `N` 次（A 的元素）或 `M` 次（B 的元素）乘加。只要每个元素从 HBM 读一次、在片上用很多次，GEMM 就能受**计算**限制，而不是受访存限制。

难点在于片上存储很小：一个 SM 只有 228 KB shared memory 和 256 KB 寄存器，装不下整个矩阵。所以要**分块**：每次只把一小块 A、B 搬上来，在块内尽量多用几次。FlashAttention 对 attention 做的是同一件事：把 Q、K、V 分块搬进 SRAM，避免把 N×N 的中间矩阵写回 HBM（论文第 3.1 节，IO 复杂度分析见第 3.2 节和定理 2）。

## 3. 数学：复用与算术强度

### 3.1 一个手算小例

`M = 2, N = 2, K = 3`：

```
A = [1  2  0]     B = [1  0  1]     C[0][0] = 1·1 + 2·0 + 0·1 = 1
    [0  1 -1]         [2  1  0]     C[0][1] = 1·2 + 2·1 + 0·0 = 4
                                    C[1][0] = 0·1 + 1·0 - 1·1 = -1
                                    C[1][1] = 0·2 + 1·1 - 1·0 = 1
```

`C[m][n]` 是 **A 的第 m 行与 B 的第 n 行**的点积。两行都在内存中连续，这就是 K-major 的含义。

`tiny`、`one_tile`、`tail`、`tail_vec`、`k1` 这几个 case 使用 `int` 模式：输入是 −2…2 的小整数，所有乘积和部分和都能被 FP32 精确表示，所以容差为 0。出错时报告的数值可以直接手算核对。

### 3.2 FLOP 与最少字节数

GEMM 做 `M·N·K` 次乘加，`./lab bench` 按 **2 FLOP/乘加** 计算：

```
FLOP          = 2·M·N·K
最少 HBM 字节 = 4·(M·K + N·K + M·N)      （A、B 各读一次，C 写一次，FP32 4 字节）
算术强度 AI   = FLOP / 字节
```

| case | FLOP | 最少字节 | AI（FLOP/B） |
|---|---|---|---|
| `square4k` 4096×4096×4096 | 137.4 G | 201 MB | 683 |
| `wide` 8192×8192×2048 | 274.9 G | 403 MB | 683 |

### 3.3 Roofline

B200 的两个“屋顶”：

- **FP32 CUDA core 峰值**：148 个 SM × 每 SM 128 条 FP32 通道 × 2 FLOP（一条 FFMA）× 1.965 GHz（`nvidia-smi -q -d CLOCK` 中的 Max Clocks，满载计时时实测仍为 1965 MHz）≈ **74.4 TFLOP/s**。`./lab bench` 的 `% peak` 用取整的 75 TFLOP/s。一个只做独立 FFMA 的微基准实测能跑到这个值的约 97%。
- **HBM 带宽**：标称 8 TB/s。

两者之比是**屋脊点**：`74.4 / 8 ≈ 9.3 FLOP/B`。AI 低于 9.3 的 kernel 受带宽限制，高于 9.3 的受计算限制。

```
 TFLOP/s
   74 ┤                 ____________________________  FP32 峰值
      │               /
      │             /   斜率 = 8 TB/s
      │           /
      │         /
      │       /
      └──────┴──────────────────────────────────────── AI (FLOP/B)
            9.3                                 683
```

`square4k` 的 AI 是 683，比屋脊点高 70 多倍：只要数据真的“每个只读一次”，读 HBM 需要 0.025 ms，计算需要 1.85 ms。**问题是 kernel 实际上读了多少次。**

### 3.4 每一级分块的复用

设一个线程块负责 C 的 `BM × BN` 块，K 方向一次搬 `BK` 宽的切片。对 C 的这块来说，A 的 `BM` 行、B 的 `BN` 行都要完整读一遍，所以：

```
每个 block 从全局内存读的 float 数 = (BM + BN) · K
全部 block 合计                     = M·N·K · (1/BN + 1/BM)
A 的每个元素被读的次数               = N / BN   （理想是 1）
全局读的算术强度                     = 2·M·N·K / (4·M·N·K·(1/BM + 1/BN)) = BM·BN / (2·(BM + BN))
```

再往下一级，线程 `TM × TN` 的 micro-tile：在一个 `kk` 上，线程从 shared memory 读 `TM` 个 A 值和 `TN` 个 B 值，做 `TM·TN` 次乘加。

| 版本 | block tile | 每线程输出 | 全局读 AI | 全局读总量（square4k） | 每次乘加的 shared load 数 |
|---|---|---|---|---|---|
| 步骤 1 朴素 | — | 1 | 0.25 | 550 GB（主要来自 L1/L2） | —（直接读全局） |
| 步骤 2 smem tile | 32×32 | 1 | 8 | 17 GB | 2 条指令（A 那条是广播） |
| 步骤 3 寄存器分块 | 128×128 | 8×8 | 32 | 4.3 GB | (8 + 8) / 64 = 0.25 |
| 步骤 4 向量化 | 128×128 | 8×8 | 32 | 4.3 GB | 同上，但 4 条 128 位指令代替 16 条 32 位指令 |

在 `square` case（1024³）上用 `./lab profile` 实测：步骤 2 的全局 load sector 数正好是 8.4 M：A、B 各要读 `1024³ / 32` 个 float（每个元素被 `1024 / 32` 个 block 读一次），共 `2·1024³ / 32` 个 float，每个 sector 32 字节装 8 个 float，`2·1024³ / 32 / 8 = 8.4 M`，步骤 3 是它的 1/4。

步骤 2 的全局读 AI 为 8，低于屋脊点 9.3：即使每次都命中 L2，要喂饱 74 TFLOP/s 也需要约 9.3 TB/s 的读带宽。步骤 3 只需要约 2.3 TB/s。**block tile 越大，越不受全局访存限制**，但它受 shared memory 和寄存器容量约束（第 5 节）。

### 3.5 不变式

所有分块版本都维护同一个不变式：处理完从 `k0` 开始的切片后，

```
acc[i][j] = Σ_{k < k0 + BK} A[row(i)][k] · B[col(j)][k]
```

越过 `K` 的位置在 shared memory 里填 0，乘出来是 0，不影响结果。所以尾部切片不需要特殊的循环，只需要在“搬进 shared memory”时判断越界。

## 4. 硬件与指令

### 4.1 FFMA 与发射槽

一条 `FFMA d, a, b, c` 计算 `d = a·b + c`，就是 2 FLOP。B200 的每个 SM 有 4 个子分区（SMSP），每个子分区每周期最多**发射一条 warp 指令**，其中的 FP32 单元每周期执行一条 warp 的 FFMA（32 条通道）。所以：

- FFMA 要占满，几乎每个周期发射的都得是 FFMA；
- 每条 `LDS`（读 shared memory）、地址计算、`BAR`（`__syncthreads`）都会占用一个本该发 FFMA 的发射槽。

这是寄存器分块最直接的收益：

| 版本 | 每个 `kk` 每线程的指令 | FFMA 在发射槽中的上限 |
|---|---|---|
| 步骤 2 | 1 FFMA + 2 LDS | ≤ 33% |
| 步骤 3 | 64 FFMA + 16 LDS | ≤ 80% |
| 步骤 4 | 64 FFMA + 4 LDS.128 | ≤ 94% |

### 4.2 Shared memory 带宽

shared memory 每 SM 每周期处理一个 **wavefront**：最多 32 个 bank 各 4 字节，即 128 B。warp 访问需要多少个 wavefront，取决于 bank 冲突（见 [00 §4.3](../00_transpose/README.md#43-shared-memory-与-bank-conflict)）和访问宽度。同一 SM 的 4 个 SMSP 共用这一个 shared memory 流水线。

一个 SM 每周期最多做 4 条 warp FFMA。步骤 3 中每个 warp 每个 `kk` 做 64 条 FFMA，需要 16 个 wavefront：`64 / 16 = 4`，正好是 FFMA 与 shared memory 吞吐的平衡点，任何冲突或停顿都会让 shared memory 成为瓶颈。步骤 4 实测每个 `kk` 需要 12 个 wavefront，留出了余量。

### 4.3 128 位访存：`float4`

```cpp
float4 v = *reinterpret_cast<const float4 *>(p);   // p 必须 16 字节对齐
```

编译成一条 `LDG.E.128`（全局）或 `LDS.128`（shared），一次读 4 个连续 float。好处是**指令数变成 1/4**，坏处是有两个前提：

1. **地址对齐到 16 字节**。全局矩阵的第 `r` 行从基地址加 `r·K·4` 字节开始，只有基地址 16 字节对齐并且 `K % 4 == 0` 时每行都对齐。`run()` 检查这两个条件（沿 K 读 A、B 时看 `K` 和 A、B 的基地址，沿 N 写 C 时看 `N` 和 C 的基地址），算出 `k_vec`、`n_vec` 传给 kernel；不满足时 `load4`/`store4` 退回 4 次标量访问（`tail` case 专门测 `K % 4 != 0` 这条路径）。
2. **4 个元素在内存中相邻**。这决定了 shared memory 里 tile 的布局（下一小节）。

常见误用：对 `float tile[8][130]` 中的 `&tile[kk][4]` 做 `float4` 读：行长 130·4 = 520 字节不是 16 的倍数，`kk = 1` 的行就不对齐了，运行时报 misaligned address。本关用 `BM + 4 = 132`。

### 4.4 为什么 K-major 的 tile 要转置后存放

计算时，一个线程在固定的 `kk` 上需要 A 的一列中的 8 个值（8 个不同的 m）和 B 的一列中的 8 个值（8 个不同的 n）。

```
按全局布局存（K-major）：as[m][kk]          转置后存：as[kk][m]
        kk →                                        m →
   m  ┌ a00 a01 … a07 ┐                       kk ┌ a00 a10 a20 a30 … ┐
   ↓  │ a10 a11 … a17 │                        ↓ │ a01 a11 a21 a31 … │
      │ a20 a21 … a27 │                          │ …                 │
      └ …             ┘                          └                   ┘
   一列（固定 kk）相隔 BK+1 个 float             一行（固定 kk）连续：可以 float4 读
```

K-major 存放时，同一个 `kk` 下相邻的 m 相隔一整行，只能一个一个读。把 tile **在搬进 shared memory 的同时转置**，固定 `kk` 的 128 个 m 就是连续的一行，可以用 `LDS.128` 一次读 4 个。转置不增加全局流量，代价是写 shared memory 时每个线程做 4 次标量 `STS`（写的次数远少于读：每个元素写 1 次、被读 16 次）。

如果 B 是 MN-major（`[K, N]`，attention 中 `P·V` 的 V 就是这样），全局的一行本来就是固定 k 的连续 n，直接拷贝即可，不用转置。Tensor Core 关卡会用 `ldmatrix` 和 smem descriptor 处理这两种布局。

## 5. 调度设计

### 步骤 1：一个线程一个输出

```
block = (32, 8)，grid = (ceil(N/32), ceil(M/8))
col = blockIdx.x·32 + threadIdx.x       ← warp 内连续，写 C 合并
row = blockIdx.y·8  + threadIdx.y
```

一个 warp 的 32 个线程有相同的 `row`、连续的 `col`：读 `A[row][k]` 是广播，读 `B[col][k]` 是 32 个相距 `K·4` 字节的地址，每次 load 要 32 个 sector。

### 步骤 2：32×32 block tile，BK = 32

```
block = (32, 32) = 1024 线程，线程 (tx, ty) 负责 C[m0+ty][n0+tx]
每个 K 切片：
  as[ty][tx] = A[m0+ty][k0+tx]      bs[ty][tx] = B[n0+ty][k0+tx]     ← 读全局：合并
  __syncthreads()
  acc += as[ty][kk] · bs[tx][kk]，kk = 0..31                         ← 读 smem
  __syncthreads()
```

计算时 warp 内 `ty` 相同、`tx = 0..31`：`as[ty][kk]` 是广播；`bs[tx][kk]` 是**按列读**，和 00 关一样需要 `+1` 的 padding，tile 声明为 `[32][33]`。

### 步骤 3：128×128 block tile，8×8 micro-tile

```
block = 256 线程，看成 16×16：tx = tid % 16，ty = tid / 16
BM = BN = 128，BK = 8，TM = TN = 8
线程 (tx, ty) 拥有 acc[i][j] = C[m0 + ty + 16i][n0 + tx + 16j]       （间隔 16 的“跨步”分配）

        n0 → BN = 128
   m0 ┌──┬──┬──┬── … ──┐      线程 (0,0) 拥有标 * 的 8×8 个位置：
    ↓ │* │  │  │       │      行 0, 16, 32, …, 112
      │  │  │  │       │      列 0, 16, 32, …, 112
      │* │  │  │       │
      └──┴──┴──┴── … ──┘

每个 K 切片：
  stage_kmajor(as) / stage_kmajor(bs)：128×8 的 A、B 切片原样拷进 smem（已提供）
  __syncthreads()
  for kk in 0..7:
      af[i] = as[ty + 16i][kk]   (i = 0..7)
      bf[j] = bs[tx + 16j][kk]   (j = 0..7)
      acc[i][j] += af[i] · bf[j]            ← 8×8 外积：16 次 load，64 次乘加
  __syncthreads()
```

**为什么是跨步分配 + `[BK + 1]` padding。** warp 是 tid 0..31，即 `tx = 0..15`、两个 `ty`。读 `bs[tx + 16j][kk]` 时，地址是 `(tx + 16j)·9 + kk`，bank 是 `(9·tx + 常数) % 32`。9 和 32 互素，16 个 `tx` 落在 16 个不同的 bank：无冲突。`as` 的两个地址（两个 `ty`）也在不同 bank。如果改成连续分配（`tx·8 + j`），bank 变成 `(72·tx + …) % 32 = (8·tx + …) % 32`，只有 4 个不同值：4 路冲突。

### 步骤 4：转置存放 + 128 位访存

```
as[BK][BM + 4]，bs[BK][BN + 4]：as[kk][r] = A[m0 + r][k0 + kk]
线程 (tx, ty) 拥有两组 4 行、两组 4 列：
  row(i) = m0 + (i/4)·64 + ty·4 + i%4        col(j) = n0 + (j/4)·64 + tx·4 + j%4

装载：tid 读一个 float4：A[m0 + tid/2][k0 + (tid%2)·4 .. +3]，转置写进 as 的一列（4 次 STS）
计算：af[0..3] = as[kk][ty·4 ..]，af[4..7] = as[kk][64 + ty·4 ..]    ← 2 条 LDS.128
      bf[0..3] = bs[kk][tx·4 ..]，bf[4..7] = bs[kk][64 + tx·4 ..]    ← 2 条 LDS.128
写回：每行两次 float4 store
```

两处细节：

- **为什么分两组、相隔 64。** 如果线程拥有连续 8 列 `tx·8 .. tx·8+7`，16 个 `tx` 的 `float4` 地址相隔 32 字节，散布在 512 字节里、只用其中一半，`tx` 与 `tx + 4` 落在同一组 bank 上。分成两组后，16 个 `tx` 读的第一组正好是连续的 64 个 float（256 字节），没有 bank 被两个不同地址占用。`./lab profile` 实测步骤 4 平均每条 `LDS.128` 需要 3 个 wavefront、无 bank 冲突：读 `bs` 的一条（16 个不同的 `float4`，共 256 字节）需要 4 个，读 `as` 的一条（每半个 warp 读同一个地址，广播）需要 2 个。一个 wavefront 最多 128 字节，32 个线程各读 16 字节的 128 位 load 至少要 4 个 wavefront，所以 4 已是理想值（Nsight Compute 的 source 页面中 `L1 Wavefronts Shared` 与 `Ideal` 两列相等）。每个 `kk` 2 × 4 + 2 × 2 = 12 个，与 4.2 节一致。
- **为什么 padding 是 4。** 转置写入时，warp 内 `lr = tid/2` 取 16 个连续值，`lk` 取 0 或 4。行长为 128 时两组写入落在相同的 16 个 bank：2 路冲突。行长 132 时 bank 是 `(4·(lk + i) + lr) % 32`，`lk = 4` 那组偏移 16，32 个线程用满 32 个 bank。padding 必须是 4 的倍数，`float4` 读才保持 16 字节对齐。

### 预算

| 资源 | 步骤 2 | 步骤 3 | 步骤 4 | B200 每 SM |
|---|---|---|---|---|
| 线程 / block | 1024 | 256 | 256 | 最多 2048 |
| shared memory / block | 2·32·33·4 = 8448 B | 2·128·9·4 = 9216 B | 2·8·132·4 = 8448 B | 228 KB |
| 寄存器 / 线程（`./lab sass`） | 32 | ≈ 100 | ≈ 127 | 每 SM 64 K 个，每线程最多 255 |
| 驻留 block / SM | 2 | 2 | 2 | — |

- `./lab sass` 显示的 `SHARED` 比上表多 1 KB：每个 block 有 1 KB shared memory 留给系统。
- 步骤 3、4 受**寄存器**限制：64 个累加器 + 16 个操作数 + 地址，约 128 个寄存器；256 线程 × 128 = 32 K，一个 SM 只能放 2 个 block（16 个 warp）。shared memory 只用了不到 10%。
- micro-tile 不能无限放大：16×16 需要 256 个累加寄存器，超过每线程 255 个的上限，会 spill 到 local memory。
- block tile 也不能无限放大：tile 越大，block 数越少。`square4k` 有 32×32 = 1024 个 128×128 block，148 个 SM × 2 = 296 个位置，3.5 轮，最后一轮只用了一半 SM。

## 6. 分步练习

所有命令在仓库根目录运行。每一步只要求对应的 TODO，其余代码已经写好。

### 步骤 1 · TODO(1)：朴素 GEMM

在 `gemm_naive` 中算出 `(row, col)`、判断越界、做长度为 K 的点积。

```bash
./lab check 10 --step 1
./lab bench 10 --step 1          # 很慢：比 cuBLAS 慢约 100 倍，要等一会儿
```

- `tiny` 是 7×5×3 的整数输入，容差为 0。报告给出 `[m=…, n=…]` 处的值，用 3.1 节的方法手算核对。
- `col` case（N = 1）失败而 `row` 通过：多半把 B 当成了 `[K, N]`。B 的第 n 行是 `b + n·K`。
- 报告 “elements are NaN … never written”：grid 没覆盖到，或越界判断写反。

### 步骤 2 · TODO(2)：shared memory tile

在 `gemm_smem` 中实现“装载 → 同步 → 32 次乘加 → 同步”的 K 循环和最后的写回。

```bash
./lab check 10 --step 2
./lab bench 10 --step 2
./lab profile 10 --step 2 --case square
```

在 profile 输出中对比 `smsp__inst_executed_op_shared_ld.sum`（shared load 指令数）和 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`（全局 load sector 数），和 3.4 节的公式对得上吗？

### 步骤 3 · TODO(3)：寄存器分块

`stage_kmajor` 已经写好。你要写两段：每个 `kk` 的 8×8 外积，以及循环结束后的写回。

```bash
./lab check 10 --step 3
./lab bench 10 --step 3
./lab sass 10                    # gemm_regblock 的 REG 应在 100 左右，LOCAL 为 0
```

- 所有循环加 `#pragma unroll`。`acc[i][j]` 的下标必须在编译期确定，否则数组会被放进 local memory（`./lab sass` 中 `LOCAL` 不为 0），性能下降一个数量级。
- `one_tile` 通过、`tail` 失败：写回时没判断 `row < M`、`col < N`。

### 步骤 4 · TODO(4)：转置存放与向量化

在 `gemm_vec` 中写完整的 K 循环（`load4` → 转置写入 → 同步 → `float4` 读 → 外积 → 同步）和用 `store4` 的写回。

```bash
./lab check 10
./lab bench 10 --step 4
./lab bench 10 --step 4 --case wide
./lab profile 10 --step 4 --case square
```

对比步骤 3 和 4 的 `smsp__inst_executed_op_shared_ld.sum`（应变为 1/4）和 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`（转置写入的冲突）。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `int` 模式的 case 误差是整数，例如差 2 或 4 | 某个 K 切片被加了两次或漏了：检查 `k0` 的步长和第二个 `__syncthreads()` |
| 偶尔失败、结果不稳定 | 缺少 `__syncthreads()`：下一切片的装载覆盖了别人还没读完的 tile |
| `tail` 失败、`tail_vec` 通过 | `K % 4 != 0` 时还在用 `float4`，或越界的 K 列没有填 0 |
| `misaligned address` | `float4` 指针不是 16 字节对齐：检查 padding 和 `vec` 参数 |
| 步骤 3 比步骤 2 还慢 | `acc` 被放进 local memory（看 `./lab sass` 的 `LOCAL`），或 `bs` 读有 bank 冲突 |
| `k1` 失败 | K = 1 时只有一个切片，且只有第 0 列有效：检查越界填 0 |

## 7. 性能

实测（CUDA Graph 计时；`vs cublas` 是 cuBLAS SGEMM（关闭 TF32）的耗时除以本版本耗时）：

| 版本 | `square4k` vs cublas | `wide` vs cublas | `square4k` % FP32 峰值 |
|---|---|---|---|
| 步骤 1 | ≈ 0.01× | （太慢，未测） | ≈ 0.3% |
| 步骤 2 | ≈ 0.14× | ≈ 0.13× | ≈ 5% |
| 步骤 3 | ≈ 0.56× | ≈ 0.53× | ≈ 19% |
| 步骤 4 | ≈ 0.72× | ≈ 0.68× | ≈ 25% |
| cuBLAS | 1× | 1× | ≈ 35% |

你的数字会因 GPU 状态略有不同，但比例关系应当一致。步骤 4 达到 cuBLAS 的 0.6× 以上即说明机制成立。

`./lab profile` 中值得看的指标（在 `square` 上实测，步骤 2 → 3 → 4）：

| 指标 | 步骤 2 | 步骤 3 | 步骤 4 | 解释 |
|---|---|---|---|---|
| `smsp__inst_executed_op_shared_ld.sum` | 67 M | 8.4 M | 2.1 M | 每次乘加的 shared load 指令：2 → 1/4 → 1/16 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | 67 M | 8.4 M | 6.3 M | shared memory 流水线的实际工作量 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈ 0 | 0 | 0 | padding 与线程分配的效果 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 8.4 M | 2.1 M | 2.1 M | block tile 从 32 变 128，全局读变为 1/4 |

`sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` 在 `square4k` 上从步骤 3 到步骤 4 提高约四分之一，适合做相对比较。

**还差在哪里。**

- 本关的 kernel 在一个切片上“先全部装载、再全部计算”，装载时 FFMA 单元空闲，只能靠同一 SM 上另一个 block 填补（每 SM 只有 16 个 warp）。cuBLAS 这里用的是 CUTLASS 的 3 级流水 SIMT kernel，在计算当前切片时已经在搬后面的切片。这正是 [21_pipeline_gemm](../21_pipeline_gemm/README.md) 的主题。
- 连 cuBLAS 也只有峰值的约 35%，而只做独立 FFMA 的微基准能到约 97%。差距来自 GEMM 内层循环的其他开销（LDS、地址计算、同步、寄存器读端口的竞争）；具体占比本关没有逐项测量。
- 更根本的是，B200 的 BF16 Tensor Core 稠密峰值约 2250 TFLOP/s，是 FP32 CUDA core 的 30 倍。真正的 attention kernel 都用 Tensor Core，从 [20_mma_gemm](../20_mma_gemm/README.md) 开始。本关的复用、分块和预算分析在那里原样适用。

## 8. 自测问题

1. 一个 64×64 的 block tile，全局读的算术强度是多少？比屋脊点高还是低？
2. 步骤 2 的 K 循环里，第二个 `__syncthreads()` 删掉会怎样？为什么 `int` 模式的 case 比随机 case 更容易发现这个问题？
3. 步骤 3 如果改成连续分配（线程拥有列 `tx·8 .. tx·8+7`），保持 `bs[128][9]` 不变，读 `bs` 是几路 bank 冲突？
4. 为什么不把 micro-tile 增大到 16×16？
5. 如果 B 按 `[K, N]` 存储，步骤 4 装载 B 时还需要转置吗？
6. `square4k` 的最少 HBM 流量和计算量各需要多少时间？步骤 2 的全局读总量（第 3.4 节表格）如果都来自 HBM，又需要多少时间？

<details>
<summary>参考答案</summary>

1. `64·64 / (2·(64 + 64)) = 16 FLOP/B`，高于 9.3，但只高出不到 2 倍：L2 命中率稍差就会受访存限制。
2. 快的 warp 会进入下一个切片，在慢的 warp 还在读 `as`、`bs` 时把它们覆盖掉，部分乘积用错了切片的数据。`int` 模式的容差为 0，任何一项用错都会报错；随机数据下也会报错，但 `int` 模式下误差是整数，更容易看出是“某一项算错”。
3. 地址 `(tx·8 + j)·9 + kk`，bank 为 `(72·tx + …) % 32 = (8·tx + …) % 32`，16 个 `tx` 只落在 4 个 bank，每个 bank 4 个不同地址：4 路冲突。
4. 16×16 需要 256 个累加寄存器，加上操作数超过每线程 255 个寄存器的上限，会 spill 到 local memory。即使不 spill，每线程寄存器翻倍后一个 SM 只能驻留 1 个 block，更难掩盖装载的延迟。
5. 不需要。`[K, N]` 中全局的一行就是固定 k、连续的 n，直接拷贝就得到 `bs[kk][n]`。A 仍是 K-major，仍需转置。
6. 最少流量 201 MB / 8 TB/s ≈ 0.025 ms；计算 137.4 GFLOP / 74.4 TFLOP/s ≈ 1.85 ms。步骤 2 的全局读约 17 GB，如果都来自 HBM 需要约 2.1 ms，已经超过计算时间：这就是为什么 block tile 要足够大，并且要靠 L2 命中。
</details>

## 9. 延伸阅读

- FlashAttention 论文（Dao et al., 2022）第 2.1–2.2 节（GPU 内存层级、算术强度）和第 3.2 节（IO 复杂度）
- CUDA C++ Programming Guide：[Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)（含分块矩阵乘的例子）
- CUDA C++ Best Practices Guide：[Shared Memory in Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
- Samuel Williams et al., *Roofline: An Insightful Visual Performance Model for Multicore Architectures*（CACM 2009）
- Simon Boehm，[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)：本关步骤 1–4 的思路与该文的 kernel 1–6 对应
- CUTLASS（`third_party/cutlass`，提交 `4370102`）：
  - `media/docs/cpp/efficient_gemm.md`：分块层级（block tile、warp tile、thread tile）与流水
  - `include/cutlass/gemm/warp/mma_simt.h`、`include/cutlass/gemm/threadblock/default_mma_core_simt.h`：SIMT GEMM 的线程分配与 shared memory 布局
  - `include/cutlass/gemm/collective/builders/sm100_simt_builder.inl`：cuBLAS 在 B200 上选中的 SM100 SIMT kernel 的构建方式
