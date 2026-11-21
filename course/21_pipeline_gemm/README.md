# 21 · 异步拷贝流水：cp.async 多级 buffer

> 部分 2 · FA2　｜　前置：[20_mma_gemm](../20_mma_gemm/README.md)　｜　代码：[`students/21_pipeline_gemm/kernel.cu`](../../students/21_pipeline_gemm/kernel.cu)

## 1. 本关目标

继续算 `C[M,N] = A[M,K] · B[N,K]ᵀ`（BF16 输入，FP32 输出）。20 关的 kernel 在搬运数据时 Tensor Core 空等，计算时又没有人搬运。本关用 **`cp.async`** 让下一块数据的搬运和这一块的计算同时进行：

1. **双缓冲**：两份 shared memory buffer，算一份、搬另一份；
2. **多级环形 buffer**：S 份 buffer，最多 S−1 块数据同时在路上，用 commit group 计数；
3. **调参**：更大的 K tile、64×64 的 warp tile、分组光栅化，并用 profile 判断每个改动为什么有效（或无效）。

做完后你应该能：

- 写出 `cp.async`、`commit_group`、`wait_group` 的封装，并说清它们的完成语义；
- 说出流水循环中每个 `__syncthreads()` 同时保护的两件事；
- 用 Little 定律估算需要几级流水；
- 把 20 关 cuBLAS 的约 0.17 倍提到约 0.28 倍，即 `mma.sync` 上限的约 70%。

`ldmatrix`、`mma`、fragment 坐标和 swizzle 由 [`include/lab/mma_sync.cuh`](../../include/lab/mma_sync.cuh) 提供（就是你在 20 关写的）。本关写的 `cp.async` 封装从 22 关起由 [`include/lab/cp_async.cuh`](../../include/lab/cp_async.cuh) 提供。

## 2. 动机：让 Tensor Core 不再等数据

20 关步骤 3 的 K 循环是：

```
load_tile(A)、load_tile(B)  ← 所有 warp 读 global，Tensor Core 空闲
__syncthreads()
ldmatrix + mma             ← 所有 warp 计算，没有访存在路上
__syncthreads()
```

普通的 global load 要先进寄存器，线程必须等数据回来才能把它存进 shared memory，所以“搬”和“算”只能交替。SM 上同时驻留的其他 CTA 可以填补一部分空闲，但每个 CTA 自己是串行的。

FlashAttention-2 的实现（`third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`）在计算当前 K/V 块的 `Q·Kᵀ` 时就用 `cp.async` 预取下一个 K/V 块，并用 `cp_async_wait<1>` 等待。22 关的 FA2 会沿用本关的流水结构：GEMM 的 K 维对应 attention 的 KV 序列维。

## 3. 数学：需要多少数据在路上

### 3.1 Little 定律

和 [00 关 §4.5](../00_transpose/README.md#45-little-定律需要多少数据在路上) 一样：

```
需要的在途字节数 = 需要的带宽 × 延迟
```

**需要的带宽**：一个 CTA 每个 K tile 读 `(BM + BN) · BK · 2` 字节、做 `2 · BM · BN · BK` 次浮点运算。BM = BN = 128、BK = 32 时是 16 KB 对 1 MFLOP。B200 上 `mma.sync` 的上限约 1920 FLOP/周期/SM（20 关 §4.1）。一个 SM 上要以这个速度计算，就要每周期供给 `1920 / 64 ≈ 30` 字节，也就是**每个 SM 每 550 个周期左右要读完一个 16 KB 的 tile**。

**延迟**：数据大多来自 L2（4096³ 时 A、B 共 64 MB，能放进 126 MB 的 L2），L2 命中的延迟是几百个周期，未命中要再加上 HBM 的延迟，在 1000 周期量级（这里是估计值，本关不测量）。

**结论**：每个 SM 需要大约 `30 B/周期 × 1000 周期 ≈ 30 KB` 在路上，即约两个 16 KB 的 tile。双缓冲每个 CTA 提前一个 tile，SM 上两个 CTA 合起来正好约两个 tile；多一级流水就多一个 tile 的余量。第 7 节的实测会验证这个估计。

### 3.2 group 计数的不变式

多级流水中，tile t 放在 stage `t % S`。每个线程每“轮”提交一个 commit group，并维持：

> **第 g 个 commit group 恰好包含 tile g 的拷贝**（tile 不存在时是空 group）。

于是“tile t 已到达”等价于“前 t+1 个 group 都已完成”。若此时一共提交了 `t + S − 1` 个 group（tile 0 .. t+S−2），允许 `S − 2` 个未完成，正好保证 tile 0..t 都完成了。这就是 `wait_group<S − 2>` 的来历。空 group 的作用是保证 K 循环末尾的计数不变：否则最后几轮的 `wait_group<S−2>` 会放过还没到的 tile。

| S | prologue 提交 | 循环中每轮 wait | 同时在路上的 tile |
|---|---|---|---|
| 2（双缓冲） | tile 0 | `wait_group 0` | 1 |
| 3 | tile 0, 1 | `wait_group 1` | 2 |
| 4 | tile 0, 1, 2 | `wait_group 2` | 3 |

## 4. 硬件与指令

### 4.1 `cp.async`

```
cp.async.cg.shared.global [dst], [src], 16, src_size;
```

| 部分 | 含义 |
|---|---|
| `cp.async` | 从 global 拷贝到 shared，**不经过寄存器**，发出后线程立即继续执行（SASS 中是 `LDGSTS`） |
| `.cg` | 只缓存在 L2（绕过 L1），只允许 cp-size = 16。另一种是 `.ca`（也缓存在 L1），cp-size 可以是 4、8、16 |
| `16` | 拷贝大小（cp-size），只能是 4、8、16；`.cg` 只能是 16。`dst`、`src` 都要按大小对齐 |
| `src_size` | 可选。只从 `src` 读 `src_size` 个字节，剩下的 `16 − src_size` 个字节**填零** |

本关用 `src_size = valid ? 16 : 0` 处理尾部：越界的 chunk 写成 16 个零字节，对点积没有贡献，而且一个字节都不读。`src` 仍然要是一个“看起来合法”的地址，传矩阵的首地址即可。

为什么要求 `K % 8 == 0`？一个 16 字节 chunk 是 8 个 BF16。K 是 8 的倍数时，每行起点都 16 字节对齐，而且每个 chunk 要么完全在行内、要么完全在行外，用 0 或 16 的 `src_size` 就够了。（支持任意 K 需要按字节数填 `src_size` 并处理不对齐的行，本关不做。）

### 4.2 commit group 与 wait_group

```
cp.async.commit_group;      把本线程此前提交、尚未分组的 cp.async 打包成一个 group
cp.async.wait_group N;      等到本线程最近的 group 中最多还有 N 个没完成
cp.async.wait_all;          = commit_group + wait_group 0
```

```
线程 t 的时间线：
  cp.async × 8 ─commit─┐ cp.async × 8 ─commit─┐ cp.async × 8 ─commit─┐
                  group 0                group 1                group 2
  wait_group 1  ⇒  group 0 完成（group 1、2 可以未完成）
```

**只管本线程。** `wait_group` 返回时，本线程发出的拷贝已经写进 shared memory、对本线程可见；**其他线程**发出的拷贝不一定完成，也不一定可见。

### 4.3 每个 `__syncthreads()` 保护两件事

流水循环中每个 K tile 只有一个 `__syncthreads()`，放在 `wait_group` 之后、发出下一批拷贝之前。它同时保证：

1. **RAW（先写后读）**：所有线程对 tile t 的拷贝都已完成并可见，任何 warp 都可以 `ldmatrix` tile t；
2. **WAR（先读后写）**：所有 warp 都已经算完上一轮的 tile（t−1），它所在的 stage 可以被覆盖。

| 去掉了什么 | 后果 |
|---|---|
| `wait_group` | barrier 处拷贝还没完成，`ldmatrix` 读到旧数据 |
| `__syncthreads()` | 只等到了自己的拷贝；别的线程的 chunk 可能没到；还可能覆盖别的 warp 正在读的 stage |
| barrier 放在 issue 之后 | WAR 被破坏：新的拷贝可能覆盖一个还有 warp 在读的 stage |

这类错误往往只在部分运行中出现。`./lab sanitize 21 --case aligned --tool racecheck` 能发现 shared memory 上的竞争。

### 4.4 超过 48 KB 的 shared memory

静态 `__shared__` 数组最多 48 KB。更大的 buffer 用动态 shared memory（`extern __shared__`，大小作为 `<<<grid, block, smem>>>` 的第三个参数），并且要在 launch 前声明上限：

```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
```

B200 每个 SM 有 228 KB shared memory，单个 CTA 最多 227 KB。shared memory 和 L1 共用每个 SM 上的同一块存储，比例（carveout）由驱动按 kernel 选择。骨架中的 `allow_smem` 同时设置了 `cudaFuncAttributePreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared`：在本机上，不设置时 3–4 级流水的版本偶尔出现慢一倍的运行（像是每个 SM 只放下了一个 CTA），设置后消失。这部分已经写好。

### 4.5 常见误用

| 误用 | 后果 |
|---|---|
| asm 没有 `"memory"` clobber | 编译器可能把普通的 shared memory 读写移到 `wait_group` 前面 |
| 越界 chunk 用 `if (ok) cp_async(...)` 跳过 | shared memory 里留着上一个 tile 的旧值，尾部结果错 |
| 越界时 `src` 仍指向算出来的越界地址 | `src_size = 0` 时不会读，但越界地址可能落在别的分配甚至未映射的页上；传矩阵首地址最稳妥 |
| `wait_group` 的参数用变量 | 编译失败：N 必须是立即数（模板参数 + `"n"` 约束） |
| K 很短时 prologue 没有提交空 group | `wait_group<S−2>` 在最后几轮放过还没到的 tile |

## 5. 调度设计

### 5.1 步骤 1：双缓冲

CTA tile 与 20 关相同：128×128，BK = 32，8 个 warp（2×4，每个 64×32），shared tile 用 20 关的 4-chunk swizzle。每个 stage 16 KB，两个 stage 32 KB。

```
prologue:  issue(tile 0 → stage 0); commit
t = 0:     wait 0; sync │ issue(tile 1 → stage 1); commit │ compute(stage 0)
t = 1:     wait 0; sync │ issue(tile 2 → stage 0); commit │ compute(stage 1)
t = 2:     wait 0; sync │ issue(tile 3 → stage 1); commit │ compute(stage 0)
                 ▲                                              │
                 └── tile t+1 的拷贝与 compute(t) 同时进行 ───────┘
```

在 t 轮覆盖 stage `(t+1) % 2` 是安全的：它上一次装的是 tile t−1，而所有 warp 在 t 轮的 barrier 之前已经算完了 tile t−1（§4.3 的 WAR）。

### 5.2 步骤 2：S 级环形 buffer

```
prologue:  for s in 0 .. S−2:  issue(tile s → stage s)（若存在）; commit
for t in 0 .. k_tiles−1:
    wait_group<S−2>; __syncthreads()
    issue(tile t+S−1 → stage (t+S−1) % S)（若存在）; commit   ← 可能是空 group
    compute(stage t % S)
```

stage `(t+S−1) % S = (t−1) % S` 上一次装的是 tile t−1，理由同上。S = 3 时 shared memory 为 48 KB，每个 SM 可以放两个 CTA。

### 5.3 步骤 3：调参

| 改动 | 目的 |
|---|---|
| BK = 32 → 64 | 每行 128 字节（8 个 chunk），用 `swizzle_chunk<8>`；每个 tile 的计算量翻倍，`__syncthreads()` 和循环开销减半 |
| 8 个 64×32 warp → 4 个 64×64 warp | 每个 k16：4 条 A 的 `ldmatrix` + 4 条 B 的 `ldmatrix` 喂 32 条 mma（原来 6 条喂 16 条），shared memory 读取量相对计算减少 1/3 |
| 3 个 stage × 32 KB = 96 KB | 两个 CTA 共用一个 SM（192 KB） |
| 分组光栅化 | 让同时运行的 CTA 共享 A、B tile，减少 L2 未命中 |

64×64 的 warp tile 有 128 个 float 累加器，答案用了约 254 个寄存器。一个 CTA 128 个线程，两个 CTA 共 65536 个寄存器，正好是一个 SM 的寄存器文件。

**光栅化**：`blockIdx.x` 是一维的，`tile_coords` 把它映射为 (tm, tn)。按行映射时，同时运行的约 296 个 CTA（148 SM × 2）覆盖 C 的若干整行，需要所有 B tile。分组映射先沿 M 走 kGroup 行，再换列：

```
按行（kGroup = 1）              分组（kGroup = 4，tiles_n = 4）
pid: 0  1  2  3                 pid: 0  4  8 12
     4  5  6  7                      1  5  9 13
     8  9 10 11                      2  6 10 14
    12 13 14 15                      3  7 11 15
                                    16 20 ...       ← 下一组
```

上图只是示意，取 kGroup = 4 是为了画得下；答案用 `kGroupM = 8`。最后一组可能不满 kGroup 行（`raster` case：11 行 tile，kGroup = 8 时最后一组 3 行）。

## 6. 分步练习

### 步骤 1 · TODO(1)：cp.async 与双缓冲

- TODO(1a)：三个封装 `cp_async_16_zfill`、`cp_async_commit`、`cp_async_wait_group<N>`（§4.1、§4.2）；
- TODO(1b)：`issue_tile`：与 20 关 `load_tile` 相同的线程分工，但用 `cp_async_16_zfill` 直接写 shared memory（§4.1 的尾部处理）；
- TODO(1c)：`gemm_double_buffer` 的 prologue 与 K 循环（§5.1）。

```bash
./lab check 21 --step 1
./lab bench 21 --step 1
./lab sanitize 21 --step 1 --case aligned --tool racecheck
```

### 步骤 2 · TODO(2)：多级环形 buffer

实现 `gemm_multistage` 的 prologue 与 K 循环（§3.2、§5.2）。`run()` 用 S = 3 调用它；同一个 kernel 在步骤 3 中也会用到。

```bash
./lab check 21 --step 2
./lab bench 21 --step 2
```

试着把 `Step2` 的 stage 数改成 2、4，看 `short_k`、`tiny` 是否仍然通过、性能怎样变化，再改回 3。

### 步骤 3 · TODO(3)：调参

- TODO(3a)：`tile_coords` 中的分组光栅化（§5.3）；
- TODO(3b)：选择 `Step3` 的配置。从下表开始实验，每次改完运行 `check` 再 `bench`：

| 实验 | `Config<BM, BN, BK, WarpsM, WarpsN, Stages>` |
|---|---|
| 起点（= 步骤 2） | `128, 128, 32, 2, 4, 3` |
| 更大的 K tile | `128, 128, 64, 2, 4, 3` |
| 更大的 warp tile | `128, 128, 64, 2, 2, 3` |
| 更大的 CTA tile | `128, 256, 32, 2, 4, 3` |
| 更多 stage | `128, 128, 64, 2, 2, 4` |

`Step3` 的默认配置就是步骤 2 的配置，所以写完 TODO(3a) 后 `./lab check 21 --step 3` 就会通过。通过只说明光栅化写对了；这一步的重点是调参：比较各配置的 `bench`，用 profile 解释快慢。

```bash
./lab check 21
./lab bench 21 --step 3 --case square
./lab bench 21 --step 3 --case wide
./lab sass 21          # 寄存器数、有没有 spill（LOCAL > 0）
./lab profile 21 --step 3 --case wide
```

把 `kGroupM` 改成 1 再 profile 一次，比较 `dram__bytes_read.sum` 和 `lts__t_sector_hit_rate.pct`。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `tail` 或 `tiny` 失败，错误值像是别处的数据 | 越界 chunk 没有填零（§4.5） |
| 结果偶尔错、每次不一样 | 少了 `wait_group` 或 `__syncthreads()`，或 barrier 位置不对（§4.3）；用 racecheck |
| `short_k`、`tiny` 失败，其他通过 | prologue 或尾部的 group 计数不对：K 很短时也要每轮 commit |
| `aligned` 失败、`tail` 通过 | stage 下标算错（`t % S`），环形 buffer 绕圈后读到旧 tile |
| 编译错误 `asm operand must be an immediate` | `wait_group` 的 N 用了运行时变量 |
| `raster` 失败（只在步骤 3） | 最后一组不满 kGroup 行时 `tm`、`tn` 计算错误 |
| `too many resources requested for launch` | 配置的寄存器 × 线程数或 shared memory 超出上限 |

## 7. 性能

本机实测（CUDA Graph；`vs cublas` 越大越好；“占 mma.sync 上限”是 TFLOP/s 除以 20 关 §4.1 测得的 560 TFLOP/s）：

| 版本 | `square` `vs cublas` | `wide` `vs cublas` | 占 mma.sync 上限 |
|---|---|---|---|
| 20 关步骤 3（同步搬运） | ≈ 0.17× | ≈ 0.17× | ≈ 45% |
| 步骤 1：双缓冲 | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 步骤 2：3 级流水 | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 步骤 3：调参后 | ≈ 0.28× | ≈ 0.26× | ≈ 72% |
| mma.sync 上限 | ≈ 0.39× | ≈ 0.36× | 100% |

步骤 3 达到 cuBLAS 的 0.22 倍以上（mma.sync 上限的约 55%）就说明机制成立。`wide` 上我们的 kernel 每次运行之间有约 ±10% 的波动（cuBLAS 很稳定），请多跑几次、看中位数。

**读这张表**：

- **步骤 1 的提升最大**：搬运和计算第一次重叠。
- **步骤 2 几乎没有变化**：和 §3.1 的估计一致。双缓冲时每个 SM 上两个 CTA 各有一个 tile 在路上，已经接近所需的在途量。多级流水在“每个 SM 只能放一个 CTA”或“延迟更长”时才明显，例如部分 3 的 TMA 流水。
- **步骤 3 的提升来自 tile 形状**，不是光栅化。步骤 3 配置实验（`square`，本机）：

| 配置 | `vs cublas` |
|---|---|
| `128,128,32, 2,4, 3`（起点） | ≈ 0.24× |
| `128,128,64, 2,4, 3` | ≈ 0.27× |
| `128,128,64, 2,2, 3`（答案） | ≈ 0.28× |
| `128,256,32, 2,4, 3` | ≈ 0.19× |
| `128,128,64, 2,2, 4`（128 KB，每 SM 一个 CTA） | ≈ 0.23× |

`128×256` 的 CTA 每个线程要 170 个寄存器，每个 SM 只放得下一个 CTA（8 个 warp），没有别的 CTA 来填补它的空闲；4 级 64 宽的 stage 也一样。**驻留的 warp 数和流水深度一起决定延迟能否被隐藏。**

**光栅化**（`wide`，`./lab profile`，答案配置）：

| 指标 | `kGroupM = 1` | `kGroupM = 8` |
|---|---|---|
| `dram__bytes_read.sum` | ≈ 716 MB | ≈ 556 MB |
| `lts__t_sector_hit_rate.pct` | ≈ 79% | ≈ 86% |
| `dram__throughput` | ≈ 5% | ≈ 4% |
| 时间 | 相同 | 相同 |

分组让 DRAM 读取减少约 22%，但时间不变：HBM 只用了约 5% 的带宽，kernel 受限于 Tensor Core（`sm__pipe_tensor_cycles_active` 约 79%）。优化要针对真正的瓶颈；等到部分 3 的 kernel 快 3–4 倍时，L2 命中率才会变得重要。

剩下约 30% 到 `mma.sync` 上限的差距，主要来自每个 tile 的 `__syncthreads()`、`ldmatrix` 与 mma 之间的依赖、以及 1024 个 CTA 在 296 个位置上的波次不均（3.46 波）。部分 3 用 TMA + mbarrier 去掉了线程搬运和 CTA 级 barrier，用 `tcgen05` 跨过 `mma.sync` 的上限。

## 8. 自测问题

1. 双缓冲循环中，如果把 `__syncthreads()` 移到 `issue_stage` 之后、`compute_stage` 之前，会出什么问题？
2. S = 4 时，循环第 t 轮的 `wait_group<2>` 返回后，哪些 tile 保证已经到达本线程？哪些可能还在路上？
3. K = 8、S = 3 时（`tiny` case），一共提交几个 group？其中几个是空的？
4. 为什么越界 chunk 要用 `src_size = 0` 写零，而不是跳过拷贝？
5. 按 §3.1 的方法估算：如果把 B200 换成 `tcgen05` 的吞吐（约为 `mma.sync` 的 4 倍），同样 128×128×32 的 tile 每个 SM 需要多少在途数据？

<details>
<summary>参考答案</summary>

1. WAR 被破坏：issue 写的是上一轮的 stage，此时可能还有 warp 在 `ldmatrix` 那个 stage（它还没走到 barrier）。RAW 仍然成立，但新 tile 会覆盖别人正在读的数据。
2. 已提交 tile 0 .. t+2 共 t+3 个 group，最多 2 个未完成，所以 tile 0 .. t 已到达；tile t+1、t+2 可能还在路上。注意这只是本线程自己的拷贝，所以后面要有 `__syncthreads()`。
3. k_tiles = 1。prologue 提交 2 个 group（tile 0 和一个空的 tile 1），循环中 t = 0 再提交 1 个（tile 2，空）。共 3 个，2 个是空的。
4. shared memory 的 stage 是复用的，跳过拷贝会留下上一个 tile 的数据，参与点积就错了。（第一次使用时则是未初始化的值。）
5. 每周期要供给的字节数变成约 4 倍，约 120 B/周期/SM；延迟不变（约 1000 周期）时需要约 120 KB 在路上，即 7–8 个 16 KB 的 tile。这就是部分 3 用更大的 tile、更深的流水（TMA 多级 buffer）的原因。
</details>

## 9. 延伸阅读

- PTX ISA：[Data Movement and Conversion Instructions: cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)，以及同节的 `cp.async.commit_group`、`cp.async.wait_group`
- CUDA C++ Programming Guide：[Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)（`cuda::memcpy_async` 是同一硬件的 C++ 接口）；[Shared Memory 与 carveout](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- Vasily Volkov，*Understanding Latency Hiding on GPUs*（2016）：Little 定律与占用率
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cute/arch/copy_sm80.hpp`：`SM80_CP_ASYNC_CACHEGLOBAL_ZFILL`，即本关的 `cp_async_16_zfill`；
  - `third_party/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h`：CUTLASS 2.x 的多级流水 GEMM 主循环；
  - `third_party/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h`：CTA 光栅化；
  - `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`：FA2 前向中 K/V 的 `cp_async_fence` / `cp_async_wait<1>` 预取。
