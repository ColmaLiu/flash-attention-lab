# 30 · TMA 与 mbarrier：异步搬运的转置

> 部分 3 · FA3 的思想在 SM100 上　｜　前置：[00_transpose](../00_transpose/README.md)、[21_pipeline_gemm](../21_pipeline_gemm/README.md)　｜　代码：[`students/30_tma_mbarrier/kernel.cu`](../../students/30_tma_mbarrier/kernel.cu)

## 1. 本关目标

再做一次 00 关的 FP32 转置 `y[c][r] = x[r][c]`，但这次不再让每个线程自己去搬数据，而是交给 **TMA**（Tensor Memory Accelerator）这个专用的拷贝引擎。线程之间用 **mbarrier** 交接：数据到了没有、buffer 空了没有。本关分三步：

1. **单 tile**：一个线程发出 TMA load，用 mbarrier 等数据到达，全体线程在 shared memory 里转置，再由一个线程发出 TMA store；
2. **persistent 环形 buffer**：一个 producer 线程不停地发 load，consumer warp 消费，`full`/`empty` 两组 barrier 管理 S 个 buffer；
3. **128B swizzle**：让 TMA 按交错的布局写 shared memory，配合 4×4 寄存器转置消除 bank conflict，再调 tile 形状和级数。

做完后你应该能：

- 用 `cuTensorMapEncodeTiled` 编码 tensor map，说清 box、越界填零、swizzle 模式各自的含义；
- 写出 `cp.async.bulk.tensor` load/store、`mbarrier` 的 init / arrive.expect_tx / try_wait.parity 封装，并画出 phase 时间线；
- 解释 generic proxy 与 async proxy，知道 `fence.proxy.async` 必须放在哪里、由谁执行；
- 把转置推到约 6.2 TB/s，接近同机 `y.copy_(x)`（无转置的连续拷贝）约 6.4 TB/s 的上限；而每个 SM 只用 9 个 warp，全部 load 由 1 个线程发起。

本关写的封装从 31 关起由 [`include/lab/mbarrier.cuh`](../../include/lab/mbarrier.cuh) 和 [`include/lab/tma.cuh`](../../include/lab/tma.cuh) 提供。这是 transpose 的最后一关。

## 2. 动机：让搬运不再占用线程

00 关步骤 4 在本机同一张卡上约 5.9 TB/s，已经接近连续拷贝。但它付出的代价是：每个 SM 驻留 32 个 CTA（每个 64 线程）、共 2048 个线程，每一个字节都要经过线程：

```
LDG（global → 寄存器） → STS（寄存器 → shared） → __syncthreads → LDS → STG
```

要让足够多的字节同时在路上（00 关 §4.5 的 Little 定律），只能靠更多线程、每线程更多 load，代价是寄存器、地址计算和 `__syncthreads()` 的等待。转置没有别的事可做，可以把整个 SM 都用来搬运；GEMM 和 attention 不行：线程要算 MMA、softmax，寄存器要放累加器。21 关的 `cp.async` 省掉了寄存器这一跳，但每个线程仍要为每 16 字节发一条指令、算一个地址。

Hopper 起，GPU 有一个独立的拷贝单元 TMA：**一个线程**用一条指令说“把这个张量中以 (x, y) 为原点的一个 box 搬进 shared memory”，硬件自己生成全部地址、处理越界，完成后把字节数报告给一个 mbarrier。线程发完指令就可以去做别的事。

FlashAttention-3 论文第 3.1 节的第一个思想就是这件事：**生产者–消费者异步**。专门的 producer warp 只负责发 TMA，consumer warp 只负责计算，二者通过 shared memory 中的环形 buffer 和 barrier 交接，搬运与计算完全重叠。本关在最简单的算子（转置）上把这套机制练熟；31 关把它接到 Tensor Core 上，34 关用进 attention。

## 3. 数学：字节、box 与 parity

### 3.1 下界与在途字节

转置读写各 `rows·cols·4` 字节。`big`（8192×8192）共 512 MiB，按 8 TB/s 标称峰值的下界约 0.067 ms，按本机 `y.copy_(x)` 实测的约 6.4 TB/s 约 0.083 ms。

Little 定律（00 关 §4.5）：要跑 6.4 TB/s、延迟按约 700 ns 估算，全 GPU 需要约 4.5 MB 在路上，即每个 SM 约 30 KB。TMA 下，在途的读取字节数直接由 buffer 数决定：

```
每 SM 在途读取 ≈ 每 CTA 的 stage 数 × 每个 stage 的字节数 × 每 SM 的 CTA 数
```

步骤 3 的答案是 2 个 stage × 32 KB × 1 个 CTA = 64 KB，加上正在写回的 store，足够覆盖延迟。

### 3.2 box 与越界

TMA 以 **box** 为单位搬运。box 是张量中一个固定大小的矩形，起点坐标由指令给出，可以是任意整数（包括让 box 伸出张量边界）：

```
x: [rows = 36, cols = 68]，box = 32 行 × 32 列，坐标写作 (x = 列, y = 行)

        x=0        x=32       x=64  68
   y=0  ┌──────────┬──────────┬────┐┄┄┄┄┄┄┐
        │ box(0,0) │ box(32,0)│    │ 越界   ┆   越界部分：load 填 0，store 跳过
  y=32  ├──────────┼──────────┼────┤ 填 0   ┆
   36   └──────────┴──────────┴────┘┄┄┄┄┄┄┘
        ┆ 越界 填 0                          ┆
  y=64  └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
```

- 一次 load **总是**向 shared memory 写满整个 box（32×32×4 = 4096 字节），越界的元素写 0，并且**这 4096 字节全部计入 mbarrier 的 tx-count**。所以 `expect_tx` 永远是 box 的字节数，与是否越界无关。
- 一次 store 只写张量内的元素，越界的部分直接丢弃。

所以本关的 kernel 里没有任何 `if (r < rows)`：尾部完全由 TMA 处理。

### 3.3 phase 与 parity：手算一个环

mbarrier 的每一轮叫一个 **phase**，编号 0、1、2、…；等待时只给 phase 编号的奇偶（parity）。环形 buffer 中，第 `it` 个 tile 用 buffer `s = it % S`，是这个 buffer 的第 `n = it / S` 次装填。于是：

- consumer 等 `full[s]` 的第 n 个 phase 完成：parity = `n & 1`；
- producer 装第 n 次之前，要等 consumer 对第 n−1 次装填的释放，即 `empty[s]` 的第 n−1 个 phase：parity = `(n − 1) & 1 = (n & 1) ^ 1`。

n = 0 时“第 −1 个 phase”并不存在。PTX 规定：**当前 phase 的前一个 phase 视为已完成**。新初始化的 barrier 处在 phase 0，等 parity 1 会立刻返回，于是 producer 第一圈不用特判。S = 2 时：

| it | s | n | producer 等 `empty[s]` 的 parity | consumer 等 `full[s]` 的 parity |
|---|---|---|---|---|
| 0 | 0 | 0 | 1（新 barrier，立即通过） | 0 |
| 1 | 1 | 0 | 1（立即通过） | 0 |
| 2 | 0 | 1 | 0（consumer 第一次释放 buffer 0） | 1 |
| 3 | 1 | 1 | 0 | 1 |
| 4 | 0 | 2 | 1 | 0 |

parity 只有 1 位：等待者必须在 barrier 前进两个 phase 之前来等，否则 parity 会“绕回”，把还没发生的 phase 当成完成。环形 buffer 天然保证了这一点：producer 不等到 `empty` 就不会再装同一个 buffer，`full[s]` 不可能领先 consumer 两个 phase。

## 4. 硬件与指令

### 4.1 tensor map：在 host 上描述张量

`CUtensorMap` 是一个 128 字节的不透明结构，在 host 端用 Driver API 编码一次（不计时，放在 `make_plan`），作为 `const __grid_constant__ CUtensorMap` kernel 参数传进去，TMA 指令取它的地址。

```cpp
cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*rank=*/2, base,
                       dims, strides, box, element_strides,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

| 参数 | 本关的值（x: [rows, cols]） | 规则 |
|---|---|---|
| `dims` | `{cols, rows}` | **最内层（连续）维在前**，单位是元素 |
| `strides` | `{cols * 4}` | rank − 1 个，单位字节；第 0 维的步长隐含为元素大小。**必须是 16 的倍数** |
| `box` | `{32, 32}` 或 `{32, 256}` | 每维 ≤ 256；`box[0] × 元素大小` 是 16 的倍数，有 swizzle 时不超过 swizzle 宽度 |
| `element_strides` | `{1, 1}` | 每隔几个元素取一个，本课程总是 1 |
| `swizzle` | `NONE` 或 `128B` | 见 §4.3 |
| `OOB_FILL_NONE` | | 越界填 0（另一选项填 NaN） |
| `base` | | 16 字节对齐 |

**支持域由第二行来**：x 的行距 `cols·4` 字节和 y 的行距 `rows·4` 字节都必须是 16 的倍数，所以本关要求 `rows % 4 == 0`、`cols % 4 == 0`。00 关的 7×11 在这里不合法。

TMA 指令里的坐标顺序和 `dims` 一样：`(x, y)` = (列, 行)。这是本关最容易犯的错误之一。

### 4.2 TMA load 与 mbarrier 的配合

```
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst], [map, {x, y}], [bar];
```

| 部分 | 含义 |
|---|---|
| `.2d` | 张量维数（1d–5d） |
| `.shared::cluster.global` | 目的地 shared、源 global。CTA 自己的 shared 地址也是合法的 cluster 地址 |
| `.mbarrier::complete_tx::bytes` | 数据写完后，对 `bar` 执行 complete_tx（box 的字节数） |
| `dst` | shared 地址；无 swizzle 时 128 字节对齐，128B swizzle 时 **1024 字节对齐** |

发出后，线程不等待。数据是否到达只能通过 mbarrier 得知。

### 4.3 swizzle：为什么、是什么

TMA 只能写**稠密**的 box，不能像 00 关那样每行补一个 float 来错开 bank。一个 32×32 FP32 tile 每行正好 128 字节 = 32 个 bank，按列读时 32 个线程落在同一个 bank：32 路冲突。

swizzle 让 TMA 在写入时把每行的 16 字节 chunk 换个位置。规则作用在 shared memory **字节地址**上（地址 bit 4 起是 chunk 号，bit 7 起是 128 字节“行”号）：

| 模式 | 规则（字节地址） | 重复周期 | box 内层最多 |
|---|---|---|---|
| `NONE` | 不变 | — | 256 个元素 |
| `32B` | bit 4 ^= bit 7 | 256 B | 32 B |
| `64B` | bits[4:5] ^= bits[7:8] | 512 B | 64 B |
| `128B` | bits[4:6] ^= bits[7:9] | 1024 B | 128 B |

对 128 字节一行的 tile，`128B` 就是：**第 r 行的第 q 个 chunk 存在 chunk 位置 `q ^ (r % 8)`**。表中的数字是存在该位置上的逻辑 chunk 号：

```
            物理 chunk 位置（每格 16 B = 4 个 float = 4 个 bank）
             0   1   2   3   4   5   6   7
   row 0:    0   1   2   3   4   5   6   7
   row 1:    1   0   3   2   5   4   7   6
   row 2:    2   3   0   1   6   7   4   5
   row 3:    3   2   1   0   7   6   5   4
   row 4:    4   5   6   7   0   1   2   3
   row 5:    5   4   7   6   1   0   3   2
   row 6:    6   7   4   5   2   3   0   1
   row 7:    7   6   5   4   3   2   1   0
   row 8:    同 row 0（每 8 行 = 1024 B 重复一次）
```

逻辑 chunk 0（第 0–3 列）在 8 行中分别位于 8 个不同的物理位置，覆盖全部 32 个 bank。只要访问方式让同一波（wavefront）的线程读“不同行的同一列”，就不再冲突。重复周期是 1024 字节，所以 buffer 必须 1024 字节对齐，否则 TMA 与你的下标公式对不上。

为什么硬件要支持 32B/64B 三种宽度？box 内层只有 32 或 64 字节时（例如 16 个 BF16），一行放不满 128 字节，用对应宽度的 swizzle。31 关的 Tensor Core 描述符也认识同一套模式，TMA 写出的布局可以直接交给 MMA 读。

### 4.4 mbarrier

mbarrier 是 shared memory 中一个 8 字节对象，内部有：

- 当前 **phase**；
- **待到达计数**（每个 phase 开始时重置为 init 时给的 count）；
- **tx-count**（以字节计的待完成事务）。

**待到达计数为 0 且 tx-count 为 0** 时，当前 phase 完成，barrier 进入下一个 phase。

| 封装 | PTX | 作用 |
|---|---|---|
| `mbar_init(bar, count)` | `mbarrier.init.shared::cta.b64 [bar], count;` | phase 0，每个 phase 期望 count 次到达。一个线程执行 |
| `fence_barrier_init()` | `fence.mbarrier_init.release.cluster;` | 让 init 对 async proxy（TMA）可见；之后 `__syncthreads()` 让其他线程可见 |
| `mbar_arrive_expect_tx(bar, bytes)` | `mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes;` | tx-count += bytes，再到达一次 |
| `mbar_arrive(bar)` | `mbarrier.arrive.shared::cta.b64 _, [bar];` | 到达一次（release 语义） |
| `mbar_wait(bar, parity)` | 循环 `mbarrier.try_wait.parity.shared::cta.b64 p, [bar], parity;` | 等到该 parity 的 phase 完成（acquire 语义） |
| （TMA 完成时） | complete_tx | tx-count −= bytes |

一个 `full` barrier（count = 1）在一个 phase 内的时间线：

```
时间 →
 producer:   arrive.expect_tx(4096) ──► 发出 TMA
                   │
 barrier:   phase 0 [待到达 1→0, tx 0→4096] ··· TMA 写完 complete_tx(4096): tx→0 ⇒ phase 0 完成，进入 phase 1
                                                                │
 consumer:   try_wait.parity(0) 返回 false … false … ───────────┴─► true，开始读 tile
```

`expect_tx` 和 complete_tx 的先后不重要：tx-count 可以暂时为负，只要 phase 结束时到达数和字节数都对上。但 **arrive 必须在 phase 结束前发生**，所以惯例是先 `arrive.expect_tx` 再发 TMA。

`try_wait` 与 `test_wait` 的区别：`try_wait` 可能让线程挂起一小段时间再返回 false，不必占满发射槽，比空转的 `test_wait` 更省；仍然要放在循环里。

### 4.5 两个 proxy 与 TMA store

GPU 的内存模型把访问分成不同的 **proxy**：普通的 `ld`/`st` 走 **generic proxy**，TMA 和 tcgen05 走 **async proxy**。同一块 shared memory 被两个 proxy 访问时，普通的同步（`__syncthreads`、mbarrier）**不保证**一个 proxy 的写对另一个 proxy 可见，必须加 proxy fence：

```
所有写 out 的线程:   st.shared out[...]       (generic proxy 写)
                     fence.proxy.async.shared::cta   ← 每个写线程自己执行
                     __syncthreads / __syncwarp
发 store 的线程:     cp.async.bulk.tensor ... [out]  (async proxy 读)
```

只有发 store 的那一个线程执行 fence 是**不够的**：fence 只对执行它的线程自己的写生效。

TMA store 与它的完成：

```
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [map, {x, y}], [src];
cp.async.bulk.commit_group;           把本线程未提交的 store 打包成一组
cp.async.bulk.wait_group.read N;      最多 N 组还在读 src → 更早组的 src 可以覆盖
cp.async.bulk.wait_group N;           最多 N 组尚未完成（包括写 global）
```

和 21 关的 `cp.async.wait_group` 一样，计数只针对发起的线程。`.read` 版本只等“读完 shared memory”，用于 buffer 复用；CTA 退出前也必须等到 store 读完源 buffer，因为 CTA 的 shared memory 随它一起释放。

反方向（generic 读之后由 TMA 覆盖）在本关不需要 fence：consumer 读完 `in[s]` 后 arrive `empty[s]`，producer 等到后才发 load，mbarrier 的 release/acquire 已经排好了顺序（CUTLASS 的 `PipelineTmaAsync` 也是这样做的）。

### 4.6 elect.sync

```
elect.sync %r|%p, 0xffffffff;
```

warp 中所有活跃线程一起执行，恰好一个得到 `p = true`。用它选出 producer 线程，比 `lane == 0` 更明确地表达“warp 内选一个”。

### 4.7 常见误用

| 误用 | 后果 |
|---|---|
| 坐标写成 `(row, col)` | 读到转置位置的 box；方阵上可能“碰巧”对，`tail`、`tall` 会错 |
| `expect_tx` 按有效元素数计算 | 尾部 tile 永远等不到剩下的字节：**挂起**（`TIMEOUT`） |
| `mbar_init` 后没有 `fence_barrier_init` + `__syncthreads` | 其他线程或 TMA 看到未初始化的 barrier，偶发挂起或错误 |
| 等 `full` 用了错误的 parity | 第一圈就挂起（等一个不会来的 phase），或立刻通过、读到旧数据 |
| producer 第一圈没有用 `(n & 1) ^ 1` | 挂起：没有人会对 `empty` 做第 −1 次释放 |
| 写完 `out` 没有 `fence.proxy.async` | TMA store 可能读到旧值。**本机测试和 racecheck 都没有发现这个错误**，所以要靠推理而不是测试来保证 |
| 只有 lane 0 执行 fence | 同上：其他 lane 的写没有被 fence |
| CTA 退出前没有 `wait_group` | store 还没读完 shared memory，buffer 已被释放 |
| 128B swizzle 的 buffer 没有 1024 字节对齐 | TMA 报错或 swizzle 下标对不上 |

## 5. 调度设计

### 5.1 步骤 1：一个 CTA 一个 tile

```
grid = (ceil(cols/32), ceil(rows/32))，block = 128 线程（4 warp）
shared: in[32][32]、out[32][32]（各 4 KB，128 B 对齐）、full（8 B）

thread 0:  mbar_init(full, 1); fence_barrier_init
all:       __syncthreads
thread 0:  arrive_expect_tx(full, 4096); tma_load(in ← x box (c0, r0))
all:       mbar_wait(full, 0)
all:       out[i][j] = in[j][i]          （4 个 warp 各做 8 行）
all:       fence_proxy_async; __syncthreads
thread 0:  tma_store(y box (r0, c0) ← out); commit; wait_group 0
```

每个 CTA 只搬 4 KB，且 load、转置、store 串行，延迟完全暴露；SM 上同时驻留的多个 CTA 是唯一的重叠来源。

### 5.2 步骤 2：persistent + 环形 buffer

```
block = 32 × (1 + C) 线程，C = kRows / 32 个 consumer warp
grid  = min(tile 数, SM 数 × 每 SM 能放的 CTA 数)      ← persistent
tile  = x 的 kRows × 32 块；CTA 处理 t = blockIdx.x, +gridDim.x, ...

           ┌──────── shared memory ───────────────────────────────┐
           │ in[0] in[1] ... in[S-1]      每个 kRows×32×4 B        │
 warp 0    │   ▲  TMA load                                         │
 producer ─┼───┘  full[s] ◄── complete_tx      empty[s] ◄── C 次 arrive
 (elect 1) │                                                       │
 warp 1..C │ out[cw][0], out[cw][1]  每个 warp 两个 32×32 输出 buffer │
 consumer ─┼── 读 in[s] 的 32 行 → 写 out → TMA store（各 warp 独立） │
           └───────────────────────────────────────────────────────┘
```

| 角色 | 每个 tile 做的事 |
|---|---|
| producer（warp 0 中 elect 出的 1 个线程） | `wait(empty[s], (n&1)^1)` → `arrive_expect_tx(full[s], kRows·128)` → TMA load |
| consumer warp cw | lane 0 `bulk_wait_group_read<1>` + `__syncwarp`（它的 out buffer 已被两轮前的 store 读完）→ `wait(full[s], n&1)` → 转置自己的 32×32 片 → `fence_proxy_async` + `__syncwarp` → lane 0：`arrive(empty[s])`、TMA store、commit |

整个 kernel 只有初始化后的一个 `__syncthreads()`。consumer warp 之间完全不互相等待：每个 warp 自己存自己的 32×32 box，用自己的 bulk group 管理自己的两个 out buffer。`empty[s]` 的 count 是 C：C 个 consumer warp 都读完了，buffer 才能重装。

时序（S = 4）：

```
producer: L0 L1 L2 L3 ·wait e0· L4 ·wait e1· L5 ...
consumer:    ·wait f0· T0 S0 ·f1· T1 S1 ·f2· T2 S2 ...     L = load, T = 转置, S = store
                         └ arrive e0 ┘
```

步骤 2 的 smem 预算（kRows = 128，S = 4）：in 4 × 16 KB + out 4 warp × 2 × 4 KB = 96 KB，每 SM 2 个 CTA。

### 5.3 步骤 3：无冲突的 4×4 寄存器转置

步骤 2 的标量转置中，lane j 读 `in[j][i]`：32 个 lane 落在同一个 bank，每条 LDS 被拆成 32 次。改成：

1. x 和 y 的 tensor map 都用 `128B` swizzle，`in` 和 `out` 都是 §4.3 的布局；
2. 把一个 warp 的 32×32 片看成 8×8 个 4×4 小块。lane l 处理两个小块 `(R, C) = (l % 8, (l % 8 + l / 8 + 4t) % 8)`，t = 0, 1，正好覆盖 64 个小块各一次；
3. 每个小块：读 `in` 的第 4R..4R+3 行、chunk C（4 个 float4），在寄存器里转置，写 `out` 的第 4C..4C+3 行、chunk R（4 个 float4）。

16 字节的访问按 8 个 lane 一组（一个 wavefront，8 × 16 B = 128 B = 32 个 bank）处理。同一组的 8 个 lane（l / 8 相同）中：

- 读第 k 条：行 `4R + k`，物理 chunk `C ^ ((4R + k) % 8)`；
- 写第 k 条：行 `4C + k`，物理 chunk `R ^ ((4C + k) % 8)`。

R 取 0..7 各一次，C = (R + 常数) % 8。可以逐一验证（或写个小脚本）：对每个 k，8 个 lane 的物理 chunk 两两不同，读和写都无冲突。指令数也降到原来的 1/4（每条搬 16 字节而不是 4 字节）。

**调参**：tile 高度 kRows（32–256，box 高度上限 256）与级数 S。答案用 kRows = 256、S = 2：每个 CTA 9 个 warp、in 2 × 32 KB + out 8 × 8 KB = 128 KB，每 SM 一个 CTA。

## 6. 分步练习

所有命令在仓库根目录运行。

### 步骤 1 · TODO(1a)(1b)(1c)：封装、编码、单 tile

- TODO(1a)：九个 PTX 封装（§4.2、§4.4、§4.5）。注意 `mbar_wait` 要在 asm 里写循环，标签用 `%=` 保证唯一；
- TODO(1b)：host 端 `encode_2d`（§4.1 的表）；
- TODO(1c)：`transpose_one_tile`（§5.1）。

```bash
./lab check 30 --step 1
./lab bench 30 --step 1
```

### 步骤 2 · TODO(2a)(2b)：环形 buffer

- TODO(2a)：`mbar_arrive`、`bulk_wait_group_read<N>`、`elect_one`；
- TODO(2b)：`transpose_pipelined` 的初始化、producer 循环、consumer 循环（§3.3、§5.2）。

```bash
./lab check 30 --step 2
./lab bench 30 --step 2
./lab sanitize 30 --step 2 --case tail --tool racecheck
```

先用 `--case tiny`、`--case tail` 调通（一个 CTA 一个 tile），再跑 `ring`（每个 CTA 约 7 个 tile，环形 buffer 绕圈）和 `ring_tail`（绕圈的同时有尾部 tile）。

### 步骤 3 · TODO(3a)(3b)(3c)：swizzle 与调参

- TODO(3a)：`tile_offset<true>`（§4.3）；
- TODO(3b)：`transpose_slice` 的 swizzle 版本（§5.3）；
- TODO(3c)：选择 `Step3` 的 `Config<Rows, Stages, true>`。

```bash
./lab check 30
./lab bench 30 --step 3
./lab bench 30 --step 3 --case wide
./lab profile 30 --step 2
./lab profile 30 --step 3
```

比较两次 profile 中的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 和 `smsp__inst_executed_op_shared_ld.sum`。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `TODO(1) is not implemented yet (make_plan)` | `encode_2d` 还没写，或仍有 `LAB_TODO(1)` |
| `cuTensorMapEncodeTiled failed` | dims/box 顺序反了（最内层在前）、stride 写成了元素数、box 超过 256 或 swizzle 宽度 |
| `TIMEOUT` | `expect_tx` 字节数不对、parity 写错、producer 第一圈在等 `empty`、`full` 的 init count 不是 1 |
| `tiny`、`square` 对，`tail`、`tall` 错 | TMA 坐标 (x, y) 顺序或 y 的 box 原点写反 |
| 步骤 2 的 `tiny`/`tail` 对、`ring` 错 | `empty` 的 arrive 次数和 init count 不一致，或 out buffer 复用前没有 `bulk_wait_group_read` |
| 步骤 3 错误值成块出现、相差 4 的倍数列 | swizzle 下标用了全 tile 行号以外的行号，或 buffer 没有 1024 对齐 |
| `misaligned address` / `illegal instruction` | TMA 目的地址未按 128/1024 字节对齐 |

## 7. 性能

本机实测（CUDA Graph）。`./lab bench 30` 打印两条基线：

- `torch`：PyTorch `y.copy_(x.t())`，一个通用的转置拷贝；
- `copy`：`y.copy_(x)`，搬同样多的字节，但读写都是完全连续的，不做转置。它是本机 HBM 的**实际上限**。

`vs` 列是相对第一条基线 `torch` 的比值。本关的主要尺子是 **`vs copy`**（`copy` 耗时 / 你的耗时，即下表最后一列）：转置最多做到和连续拷贝一样快，比值越接近 1 越好。

| 版本 | `big` 8192×8192 GB/s | `wide` 4096×16384 GB/s | % HBM 标称峰值 | `vs copy` |
|---|---|---|---|---|
| 步骤 1：单 tile | ≈ 1950 | ≈ 1950 | ≈ 24% | ≈ 0.30× |
| 步骤 2：环形 buffer（标量、无 swizzle） | ≈ 2100 | ≈ 2100 | ≈ 26% | ≈ 0.33× |
| 步骤 3：swizzle，`Config<128, 4>` | ≈ 5600 | ≈ 5600 | ≈ 70% | ≈ 0.87× |
| 步骤 3：swizzle，`Config<256, 2>`（答案） | ≈ 6200 | ≈ 6100 | ≈ 77% | ≈ 0.95× |
| 参考：00 关步骤 4 | ≈ 5900 | — | ≈ 74% | ≈ 0.92× |
| 参考：`torch`（`y.copy_(x.t())`） | ≈ 1860–1900 | ≈ 1890–1910 | ≈ 23% | ≈ 0.29× |
| 参考：`copy`（`y.copy_(x)`，无转置） | ≈ 6430 | ≈ 6430 | ≈ 80% | 1 |

答案达到连续拷贝的约 95%。它只比 00 关步骤 4 快约 5%（`--cold` 时约 2%）：转置本来就可以用满整个 SM 来搬运，TMA 在这里的收益不是带宽，而是**用 1 个线程、9 个 warp 做到了 00 关 2048 个线程才做到的事**。后面的关卡里，省下来的线程和寄存器用来做 Tensor Core 计算和 softmax。本关的性能门槛是步骤 3 在两个 perf case 上都达到 `vs copy` 0.85 倍以上（`copy` 耗时 / 你的耗时 ≥ 0.85）；`vs torch` 通常在 3 倍以上，但它不是好尺子，原因见下。

**读这张表**：

- **步骤 1、2 都慢，而且相差不到 10%**：瓶颈是 bank conflict，而不是搬运。`./lab profile` 中，步骤 1、2 的 shared load 有约 6600 万次 bank conflict，对应约 210 万条 LDS：每条约 31 次额外的 wavefront，正是 32 路冲突。步骤 2 的流水让 load 与转置重叠，只换来约 8%，被转置本身卡住。
- **步骤 3 一下快了约 3 倍**：LDS/STS 各约 52 万条（16 字节一条，是原来的 1/4），bank conflict 约 70 万（load）和 100 万（store），平均每条不到 2 次。剩下这些不是下标公式造成的（换一种 lane 分配几乎不变），推测来自与 TMA 读写同一块 shared memory 的仲裁，未进一步证实。
- **为什么不用 `vs torch` 当尺子**：`torch` 的转置和步骤 1、2 一样，不是被 HBM 卡住的。早先另一次测量中，步骤 1、2 只有约 1030、1060 GB/s（约为上表的一半），`torch` 在约 1.0 TB/s 与约 1.8 TB/s 之间跳动，而步骤 3 和 `copy` 与上表相同。一个**未经验证的假设**是 GPU 当时处于较低的 SM 时钟：受 bank conflict 或延迟限制的 kernel 的速度随 SM 时钟变化，受 HBM 限制的 kernel（`copy`、步骤 3）几乎不受影响。不论原因是什么，`copy` 和答案都由 HBM 决定，二者的比值在这种波动下是稳定的。
- **调参**（`big`，swizzle 版）：

| `Config<Rows, Stages>` | smem / CTA | CTA / SM | GB/s |
|---|---|---|---|
| `<32, 4>` | 25 KB | 8 | ≈ 5180 |
| `<64, 4>` | 49 KB | 4 | ≈ 5650 |
| `<128, 2>` | 65 KB | 3 | ≈ 5720 |
| `<128, 4>` | 97 KB | 2 | ≈ 5630 |
| `<128, 6>` | 129 KB | 1 | ≈ 5600 |
| `<256, 2>` | 129 KB | 1 | ≈ 6200 |
| `<256, 3>` | 161 KB | 1 | ≈ 5770 |

更多 stage（更多在途字节）并没有更快，最快的是“大 box、2 级”。我们没有找到确切原因；可能的解释是在途读请求过多时，HBM 的读写切换和页冲突抵消了收益。这类结论只能靠测量，不要凭直觉选级数。

`./lab profile` 时 Nsight Compute 锁定时钟，时间比 `bench` 长，只用它比较指标。

## 8. 自测问题

1. `tail`（36×68）中，CTA 处理 x 的 box 原点 (x = 64, y = 32)（步骤 1）。这个 box 里有多少个有效元素？`expect_tx` 应该填多少字节？
2. mbarrier 初始化为 count = 2，两个线程各执行一次 `arrive`，同时 tx-count 为 0。此时 phase 0 完成了吗？如果其中一个线程用的是 `arrive.expect_tx(1024)`，而没有任何 TMA 呢？
3. 步骤 2 中，若把 `empty[s]` 的 count 设为 consumer 线程数（128）但仍然每个 warp 只 arrive 一次，会怎样？
4. 为什么 consumer 在复用 out buffer 之前用 `wait_group.read 1` 而不是 `wait_group.read 0`？
5. 128B swizzle 下，`in` 的第 13 行第 22 列（float）的字节偏移是多少（相对 1024 对齐的 buffer）？
6. 删掉 `fence_proxy_async()` 后本机测试仍然全部通过。为什么这不能说明 fence 可以省略？

<details>
<summary>参考答案</summary>

1. 行 32..35 有效（4 行），列 64..67 有效（4 列），共 16 个元素；`expect_tx` 仍是整个 box 的 32 × 32 × 4 = 4096 字节，越界部分以 0 写入并计数。
2. 第一种情况完成了：到达计数归零、tx-count 为 0。第二种情况不会完成：tx-count 是 1024，没有 complete_tx 来抵消，等待者永远挂起。
3. 每个 phase 只有 C 次到达，永远凑不够 128 次：producer 第二圈在 `empty[s]` 上挂起。
4. buffer `it % 2` 上一次是在 `it − 2` 被 store 的；此时最近提交的是 `it − 1` 的组，允许它还没读完（1 个未完成），就保证了 `it − 2` 的组已经读完。用 0 也正确，但会多等一个 store，减少重叠。
5. 行 13：`13 % 8 = 5`；列 22 在 chunk `22 / 4 = 5`，chunk 内第 2 个 float。物理 chunk = `5 ^ 5 = 0`。偏移 = `13 × 128 + 0 × 16 + 2 × 4 = 1672` 字节。
6. 内存模型不保证没有 fence 时的可见性；测试通过只说明在这台机器、这个负载下恰好没有出现重排。racecheck 也不跟踪 async proxy 的访问。这类正确性要按规范推理，测试只能发现一部分错误。
</details>

## 9. 延伸阅读

- PTX ISA 8.7：[cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)、[cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)、[mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)（含 phase、tx-count 的定义）、[Tensor Memory Swizzling Modes / Tensor copy](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes)、[elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync)、[Memory Consistency Model: Proxies](https://docs.nvidia.com/cuda/parallel-thread-execution/#proxies)
- CUDA Driver API：[cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- CUDA C++ Programming Guide：[Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)
- FlashAttention-3 论文（Shah et al., 2024）第 3.1 节：Producer-Consumer asynchrony
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp`：`SM90_TMA_LOAD_2D`、`SM90_TMA_STORE_2D`、`tma_store_arrive` / `tma_store_wait`；
  - `third_party/cutlass/include/cute/arch/copy_sm90_desc.hpp`：`TMA::SmemSwizzleBits` 与 tensor map 的编码；
  - `third_party/cutlass/include/cutlass/arch/barrier.h`：`ClusterTransactionBarrier`（mbarrier 封装）、`fence_view_async_shared`；
  - `third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp`：`PipelineTmaAsync`，即本关步骤 2 的 full/empty 协议（producer 从 phase 1 开始）。
