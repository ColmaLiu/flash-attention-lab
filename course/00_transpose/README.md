# 00 · 访存与线程映射：矩阵转置

> 部分 0 · 预备　｜　前置：无　｜　代码：[`students/00_transpose/kernel.cu`](../../students/00_transpose/kernel.cu)

## 1. 本关目标

把一个 FP32 矩阵转置：`y[c][r] = x[r][c]`。转置没有任何计算，速度完全取决于**数据怎么在内存层级之间移动**。本关用四个版本，把同一个转置从 HBM 标称带宽的 15% 提到约 74%（约为连续拷贝速度的 90%）。做完后你应该能：

- 把线程坐标映射到数组下标，并判断一个 warp 的访存是否合并（coalesced）；
- 用 shared memory 改变访问顺序，并计算 bank conflict；
- 用 Little 定律解释“为什么每个线程要多搬一些数据”；
- 用 `./lab bench` 和 `./lab profile` 验证以上判断，而不是靠猜。

后面每一关的 attention kernel 都在反复做这几件事：把 Q、K、V 的 tile 从 HBM 搬进 shared memory，换一种顺序读出来，并让足够多的数据同时在路上。

## 2. 动机：为什么从转置开始

FlashAttention 论文的核心观察是：attention 的瓶颈不是浮点运算，而是 HBM 读写（论文第 2.2 节）。要理解这句话，得先知道“读写 HBM”在 GPU 上具体是什么：多少线程在发请求、请求怎样合并、需要多少请求同时在路上。转置是最小的、只有访存的 kernel，最适合把这些事情看清楚。

## 3. 数学：下标与下界

两个矩阵都是行主序（row-major），最后一维连续：

```
x: [rows, cols]   x[r][c] 的地址 = x + (r * cols + c)
y: [cols, rows]   y[c][r] 的地址 = y + (c * rows + r)
```

转置必须读 `rows*cols*4` 字节、写同样多字节。`big` case 是 8192×8192，读写共 512 MiB。B200 的 HBM3e 标称 8 TB/s，所以时间下界约 0.067 ms；在本机实测，最简单的 `y.copy_(x)`（连续拷贝，无转置）大约跑到 6.5 TB/s，这是更现实的上限。

`./lab bench` 报告的 `GB/s` 就是 `2 * rows * cols * 4 / 时间`，`% peak` 是它相对 8 TB/s 的比例。

## 4. 硬件与指令

### 4.1 线程层级

| 层级 | 本关的用法 | B200 上的数字 |
|---|---|---|
| grid | 覆盖整个矩阵的 block 网格 | — |
| block（CTA） | 32×8 = 256 个线程，共享一块 shared memory | 每个 SM 最多 2048 线程、228 KB shared memory |
| warp | 32 个连续 `threadIdx.x` 的线程，**一起执行同一条指令** | 每个 SM 最多 64 个 warp |
| thread | 搬一个或几个元素 | 148 个 SM |

一个 block 内线程的线性编号是 `threadIdx.x + threadIdx.y * blockDim.x`，每 32 个连续编号组成一个 warp。本关 `blockDim.x = 32`，所以**同一个 `threadIdx.y` 的 32 个线程就是一个 warp**。记住这一点，它决定了下面所有的访存分析。

### 4.2 全局内存访问的合并

warp 执行一条 load/store 指令时，硬件把 32 个线程的地址合在一起处理：

- 内存以 **32 字节的 sector** 为单位在 L1/L2/HBM 之间移动；
- 32 个线程访问连续的 32 个 `float`（128 字节）时，只需要 **4 个 sector**：这叫合并访问；
- 32 个线程各自访问相隔很远的地址时，需要 **32 个 sector**，每个只用到 4 字节。

朴素转置里，warp 的 32 个线程有相同的 `r` 和连续的 `c`：

- 读 `x[r][c]`：地址连续，合并；
- 写 `y[c][r]`：相邻线程地址相差 `rows * 4` 字节，不合并。

下面是本机 `./lab profile 00 --solution --step 1` 的结果（8192×8192），注意 store 每个请求对应 32 个 sector：

| 指标 | 读 x | 写 y |
|---|---|---|
| 请求数（warp 指令） | 2,097,152 | 2,097,152 |
| sector 数 | 8,388,608（每请求 4 个） | 67,108,864（每请求 32 个） |

有意思的是，HBM 实际读写字节数（`dram__bytes_*`）几乎和优化后的版本一样：同一 32 字节 sector 的其余部分很快会被别的 warp 写到，L2 把这些零碎写入合并后才写回 HBM。所以朴素版本慢的原因不是“HBM 多搬了 8 倍数据”，而是 **L1/L2 要处理 8 倍的 sector 事务**，事务处理速度成了瓶颈。这就是为什么要测量，而不是只凭教科书结论推断。

### 4.3 Shared memory 与 bank conflict

shared memory 在每个 SM 上，延迟和带宽都远好于 HBM，块内线程可以互相读对方写入的数据。它被分成 **32 个 bank**，每个 bank 每周期服务一个 4 字节字：

```
bank(地址) = (字节地址 / 4) % 32
```

一个 warp 的 32 个线程如果访问 32 个不同的 bank，一个周期完成；如果有 k 个线程访问同一个 bank 的**不同地址**，就要串行成 k 次，这叫 k 路 bank conflict（访问同一个地址是广播，不冲突）。

声明 `float tile[32][32]` 时，`tile[i][j]` 的 bank 是 `(i*32 + j) % 32 = j`。按行读（固定 `i`，`j = lane`）没有冲突；**按列读**（固定 `j`，`i = lane`）32 个线程全落在 bank `j`：32 路冲突。

把每行补一个 float：`float tile[32][33]`，`tile[i][j]` 的 bank 变成 `(i*33 + j) % 32 = (i + j) % 32`。按列读时 `i = 0..31` 落在 32 个不同的 bank，冲突消失。代价是每个 block 多用 128 字节 shared memory。

（后面 tensor core 关卡会用另一种方法：XOR swizzle，不浪费空间。思路相同：让一列数据分散到不同 bank。）

### 4.4 `__syncthreads()`

块内所有线程到达这条指令后才继续，并保证之前对 shared memory 的写入对块内其他线程可见。转置中一个线程会读**别的线程**写进 tile 的元素，所以 load 和 store 之间必须有它。少了它，结果有时对有时错，取决于 warp 的调度顺序。

### 4.5 Little 定律：需要多少数据在路上

```
带宽 = 在途字节数 / 延迟
```

HBM 的访问延迟大约在几百纳秒到 1 微秒量级。要达到 6.5 TB/s，按 700 ns 估算，全 GPU 需要约 4.5 MB 同时在路上，平均每个 SM 约 30 KB。

一个 SM 上最多驻留 2048 个线程。在途数据量 = 驻留线程数 × 每个线程同时在路上的 load 数 × 每次的字节数。后两项取决于代码怎么写：

- **load 必须能一起发出去。** 线程发出一条 load 后不会停下来等数据，而是继续执行后面不依赖这个数据的指令；只有用到数据时才等待。所以一个线程里互不依赖、能连续发出的 load 越多，同时在路上的数据就越多。这叫指令级并行（ILP）。
- **循环的写法会影响编译器能否这样做。** 写成 `for (k = 0; k < 4; ++k)` 时，循环次数是编译期常数，编译器把它完全展开，4 条 load 连续发出。写成 `for (i = threadIdx.y; i < 32; i += 8)` 时，循环次数取决于 `threadIdx.y`，编译器生成带分支的代码，每次 load 后立刻把结果写进 shared memory，下一条 load 要等上一条返回。本机实测这一个改动让步骤 3 从约 2.8 TB/s 变成约 5.0 TB/s。用 `./lab sass 00` 生成的 SASS 文件可以检查：找到 kernel 里第一条 `STS` 之前有几条 `LDG`。

步骤 3 中每个线程有 4 条 load 在路上，一个 SM 满载时约 2048 × 4 × 4 B = 32 KB，刚好在上面估算的需求附近；而且每个 block 在 `__syncthreads()` 之后的写阶段不再读数据。步骤 4 保持 tile 不变，把每个 block 的线程从 32×8 减到 32×2：每个线程负责 16 个元素，连续发出 16 条 load，在途数据量提高到原来的 4 倍。这在 GPU 上很常见：**单个线程的 ILP 和 SM 上的线程数（TLP）共同决定在途访存量**，线程少一些、每个线程多做一些，往往更快。

## 5. 调度设计

### 步骤 1：一个线程一个元素

```
block = (32, 8), grid = (ceil(cols/32), ceil(rows/8))
c = blockIdx.x * 32 + threadIdx.x      ← warp 内连续
r = blockIdx.y * 8  + threadIdx.y
```

### 步骤 2–4：经过 shared memory 的 tile

```
             x 的一个 tile（kTile × kTile）                 y 中对应的位置
   c0 ────────────────────► 列                    r0 ──────────────► 列
r0 ┌──────────────────────┐                   c0 ┌──────────────────┐
   │ warp 0 读第 0 行      │  ── 写入 tile ──►     │ warp 0 写第 0 行  │
   │ warp 1 读第 1 行      │     __sync          │ = x 的第 c0 列     │
   │ ...                  │  ── 按列读出 ──►      │ ...              │
   └──────────────────────┘                      └──────────────────┘
   全局读：合并                                       全局写：合并
```

- 读阶段：warp `threadIdx.y` 读 tile 的第 `threadIdx.y`、`threadIdx.y+kThreadsY`、… 行，lane 对应列，合并；写入 `tile[i][j]`，按行写，无冲突。
- 写阶段：y 的第 `c0+i` 行就是 x 的第 `c0+i` 列。warp 写 y 的一行，lane `j` 写 `y[c0+i][r0+j]`，需要 `tile[j][i]`，即**按列读 tile**。

| 版本 | tile | block | 每线程元素数 | shared memory / block |
|---|---|---|---|---|
| 步骤 2 | `float[32][32]` | 32×8 | 4 | 4096 B |
| 步骤 3 | `float[32][33]` | 32×8 | 4 | 4224 B |
| 步骤 4 | `float[32][33]` | 32×2 | 16 | 4224 B |

步骤 4 每个 block 只有 64 个线程。一个 SM 最多驻留 32 个 block，所以仍然能放满 2048 个线程，shared memory 用量为 32 × 4224 B ≈ 135 KB，在 228 KB 以内。

## 6. 分步练习

所有命令在仓库根目录运行。每一步只要求对应的 TODO；其余代码已经写好。

### 步骤 1 · TODO(1)：朴素转置

在 `transpose_naive` 里算出 `(r, c)`、判断越界、拷贝一个元素。

```bash
./lab check 00 --step 1
./lab bench 00 --step 1
```

- 第一个 case `tiny` 是 7×11，输入值等于它在 x 中的线性下标（`coordinate` 模式）。出错时报告会给出“在 `[col=…, row=…]` 得到 v、期望 w”，`v` 直接告诉你读到了 x 的哪个位置。
- 报告 “elements are NaN … never written” 说明有输出没写到：检查 grid 覆盖和越界条件是否写反。
- `rows * cols` 在大 case 中超过 2³¹，下标请用 `int64_t`。

### 步骤 2 · TODO(2)：shared memory tile

在 `transpose_tiled` 中实现读、同步、写三段。按骨架注释把循环写成编译期次数的形式 `for (k = 0; k < kTile / kThreadsY; ++k)`（原因见第 4.5 节），步骤 4 可以直接复用。

```bash
./lab check 00 --step 2
./lab bench 00 --step 2
./lab profile 00 --step 2
```

在 profile 输出中找 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`：这是 shared memory load 的 bank conflict 次数。8192×8192 上它是几千万量级，和第 4.3 节的分析对得上吗？

### 步骤 3 · TODO(3)：消除 bank conflict

在 `run()` 的 `case 3` 选择 `kPad` 并发射 kernel。改动只有一行，但请先在纸上算出 `kPad = 1` 时一列的 32 个元素落在哪些 bank。

```bash
./lab check 00 --step 3
./lab bench 00 --step 3
./lab profile 00 --step 3
```

### 步骤 4 · TODO(4)：每个线程多搬一些

在 `case 4` 选择 `kThreadsY`，使每个线程负责 16 个元素，并发射 kernel。如果步骤 2 的循环按 `kTile / kThreadsY` 写，kernel 本身不用改。先算一算：步骤 3 和步骤 4 中，一个 SM 满载时各有多少字节的 load 在路上？

```bash
./lab check 00
./lab bench 00 --step 4
./lab bench 00 --step 4 --cold
```

`--cold` 在每次计时前清空 L2，测的是数据真正来自 HBM 时的性能；8192×8192 的输入（256 MiB）本来就比 L2（126 MB）大，所以两种计时差别不大。小矩阵上两者会差很多，想想为什么。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 大部分 case 过了，`tail` 失败 | 尾部 tile 的越界判断用错了维度（读用 `rows/cols`，写用 `cols/rows`） |
| 偶尔失败、结果不稳定 | 缺少 `__syncthreads()` |
| `row` case 失败 | 把 y 当成了 `[rows, cols]` |
| 步骤 3 没变快 | 检查 bank conflict 指标是否真的下降；是否忘了改 `case 3` |
| `TIMEOUT` | 循环变量写错导致死循环 |

## 7. 性能

本机实测（8192×8192，CUDA Graph 计时；`vs torch` 是 PyTorch `y.copy_(x.t())` 的耗时除以本版本耗时）：

| 版本 | `vs torch` | % HBM 标称峰值 |
|---|---|---|
| 步骤 1 | ≈ 0.65× | ≈ 15% |
| 步骤 2 | ≈ 1.0× | ≈ 24% |
| 步骤 3 | ≈ 2.7× | ≈ 62% |
| 步骤 4 | ≈ 3.2× | ≈ 74% |

你的数字可能因 GPU 频率和负载略有不同，比例关系应当一致。步骤 3 达到 `vs torch` 2.5× 左右、步骤 4 再快 15% 以上，即说明机制成立。如果步骤 3 只有 1.5× 左右，检查循环是否写成了编译期次数（第 4.5 节）。

`./lab profile` 时 Nsight Compute 会锁定时钟，`gpu__time_duration` 比 `bench` 慢一些，只用它比较指标，不要和 bench 的时间直接比。

步骤 4 已经接近 6.5 TB/s 的连续拷贝速度。剩余差距主要来自读写交替的 block 结构：一个 block 在写阶段不发出新的读请求。到 [30_tma_mbarrier](../30_tma_mbarrier/README.md) 用 TMA 做转置时会再回到这个问题。

## 8. 自测问题

1. `blockDim = (32, 8)` 时，`threadIdx = (5, 3)` 的线程属于 block 中的第几个 warp？
2. 步骤 1 中，如果把 `threadIdx.x` 对应到行、`threadIdx.y` 对应到列，读和写哪个会合并？
3. `float tile[32][32]` 中，按列读第 7 列时 32 个线程访问哪些 bank？如果改成 `double tile[32][33]` 还无冲突吗？
4. 步骤 4 的 block 只有 2 个 warp。按列读 tile 时，是否仍然无 bank 冲突？为什么？
5. 一个 4096×4096 的 FP32 矩阵能全部放进 L2。对它做 graph 计时和 `--cold` 计时，哪个更快，为什么？

<details>
<summary>参考答案</summary>

1. 线性编号 `5 + 3*32 = 101`，`101 / 32 = 3`，第 3 个 warp（从 0 数）。
2. 写会合并，读不合并：warp 内 32 个线程会有连续的行号 `r`，`y[c][r]` 连续、`x[r][c]` 跨行。
3. bank 全是 7，32 路冲突。`double` 每个元素占两个相邻 bank，32 个线程共要 256 字节，硬件本来就分两次（每次半个 warp、16 个线程）服务。`double tile[32][33]` 中第 `i` 行第 `j` 列的起始 bank 是 `(66i + 2j) % 32 = (2i + 2j) % 32`；同一个半 warp 内 `i` 取 16 个连续值，占用的 bank 对互不相同，所以除了必需的两次之外没有额外冲突。可以用 `./lab profile` 的 bank conflict 指标验证。
4. 无冲突。bank conflict 只发生在同一个 warp 的一次访问内。写阶段一个 warp 的 32 个 lane 读 `tile[lane][i]`，bank 是 `(lane*33 + i) % 32 = (lane + i) % 32`，32 个 lane 两两不同；这和 block 里有几个 warp 无关。
5. graph 计时更快：反复运行时输入留在 L2，`--cold` 每次先清空 L2，数据必须从 HBM 读。
</details>

## 9. 延伸阅读

- CUDA Programming Guide：[Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access)，[Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-bank-conflicts)，以及同页的 [Matrix Transpose Example Using Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#matrix-transpose-example-using-shared-memory)
- CUDA C++ Best Practices Guide：[Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- Mark Harris，[An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)（经典文章，本关步骤 1–3 的来源）
- Vasily Volkov，*Understanding Latency Hiding on GPUs*（博士论文，2016）：用 Little 定律分析 GPU 访存
- FlashAttention 论文第 2.1–2.2 节：GPU 内存层级与 IO 感知
