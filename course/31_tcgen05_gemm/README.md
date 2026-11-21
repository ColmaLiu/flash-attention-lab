# 31 · tcgen05 与 TMEM：第五代 Tensor Core GEMM

> 部分 3 · FA3 的思想在 SM100 上　｜　前置：[21_pipeline_gemm](../21_pipeline_gemm/README.md)、[30_tma_mbarrier](../30_tma_mbarrier/README.md)　｜　代码：[`students/31_tcgen05_gemm/kernel.cu`](../../students/31_tcgen05_gemm/kernel.cu)

## 1. 本关目标

继续算 `C[M,N] = A[M,K] · B[N,K]ᵀ`（BF16 输入，FP32 输出），但换一种 Tensor Core 指令：Blackwell 的 **`tcgen05.mma`**。它和 20、21 关的 `mma.sync` 完全不同：

| | `mma.sync`（20、21 关） | `tcgen05.mma`（本关） |
|---|---|---|
| 谁发起 | 整个 warp，32 个线程各出一份 fragment | **一个线程** |
| 一条指令的形状 | 16×8×16 | 最大 128×256×16（单 CTA） |
| A、B 从哪来 | 寄存器（先 `ldmatrix`） | **shared memory**，用 64 位描述符说明布局 |
| 累加器在哪 | 寄存器 | **Tensor Memory（TMEM）** |
| 何时完成 | 同步：指令返回即可用 | **异步**：用 `tcgen05.commit` 通知 mbarrier |

本关三步：

1. **单个 tile**：TMA 装 128×64 的 A 和 128×64 的 B，4 条 MMA，从 TMEM 读出 128×128 的 C；
2. **完整 K 循环**：S 级 TMA/mbarrier 环形 buffer，一个线程既发 TMA 又发 MMA，`tcgen05.commit` 释放 buffer；
3. **TMA store epilogue 与 128×256 tile**：把累加器经 swizzle 过的 shared memory 用 TMA 写回，再调 tile 形状。

做完后你应该能：

- 画出 TMEM 的 lane/column 布局，说出哪个 warp 能读哪 32 个 lane；
- 逐位写出 shared memory 描述符和 instruction descriptor，并解释 K 方向为什么只要把起始地址加 32 字节；
- 写出 alloc → MMA → commit → wait → ld → dealloc 的完整生命周期，以及每个 `tcgen05.fence` 的作用；
- 把 GEMM 从 21 关 cuBLAS 的约 0.28 倍提到约 0.8 倍。

`mbarrier` 与 TMA 封装来自 [`include/lab/mbarrier.cuh`](../../include/lab/mbarrier.cuh)、[`include/lab/tma.cuh`](../../include/lab/tma.cuh)（30 关写的）。本关写的 tcgen05 封装从 32 关起由 [`include/lab/tcgen05.cuh`](../../include/lab/tcgen05.cuh) 提供。

## 2. 动机：mma.sync 的天花板

21 关的答案达到 `mma.sync` 上限的约 72%，而这个上限本身只有 cuBLAS 的约 0.39 倍（21 关 §7）。原因是 `mma.sync` 是“寄存器到寄存器”的指令：每做一次 16×8×16，warp 要先用 `ldmatrix` 把 A、B 读进寄存器，累加器也占着寄存器。Tensor Core 越快，喂它的寄存器带宽和发射槽就越成为瓶颈。

Hopper 的 `wgmma` 让操作数直接从 shared memory 读；Blackwell 的 `tcgen05` 更进一步：累加器也离开寄存器，放进每个 SM 上一块专用的 256 KB **Tensor Memory**。一个线程发出一条 128×256×16 的 MMA，硬件自己从 shared memory 读 A、B，把结果累加进 TMEM；其他线程这段时间完全空闲，可以做别的事（32 关起它们去做 epilogue，34 关起去做 softmax）。

这正是 FlashAttention-3 第二个思想（GEMM 与 softmax 重叠）在 Blackwell 上成立的硬件前提：MMA 异步、累加器不占寄存器。

## 3. 数学：一个 tile 的分解

### 3.1 tile 与指令

CTA 计算 C 的一个 `BM × BN` 块（BM = 128）。K 方向按 `BK = 64` 切成 K 块（一行 64 个 BF16 = 128 字节，正好一个 128B swizzle 行），每个 K 块再切成 4 个 `K = 16` 的片，每片一条 MMA：

```
C[m0:m0+128, n0:n0+BN] = Σ_t Σ_kk  A[m0:+128, 64t+16kk : +16] · B[n0:+BN, 64t+16kk : +16]ᵀ
                         └ K 块 t ┘└ 4 条 tcgen05.mma ┘
```

`int` 模式的输入（−2..2 的整数）让所有乘积与部分和在 FP32 中精确，任何结果不等都是 bug，不是舍入。

### 3.2 Tensor Core 需要多少数据

按 B200 BF16 稠密峰值 2250 TFLOP/s、148 个 SM、约 1.9 GHz 估算，每个 SM 每周期约 8192 FLOP。一条 128×N×16 的 MMA 是 `2·128·N·16` FLOP：

| N | 每条 MMA 的 FLOP | 约多少周期 | 每条读 shared memory | 每周期读 shared memory |
|---|---|---|---|---|
| 128 | 524,288 | ≈ 64 | A 4 KB + B 4 KB | ≈ 128 B |
| 256 | 1,048,576 | ≈ 128 | A 4 KB + B 8 KB | ≈ 96 B |

shared memory 每个 SM 每周期约 128 字节。N = 128 时 MMA 刚好吃满它，任何其他访问（TMA 写入、epilogue）都会让 Tensor Core 等待；N = 256 时 A 被复用两倍，留出余量。这是步骤 3 选 128×256 tile 的理由（估算值，第 7 节实测验证）。

一个 K 块（N = 256）有 4 条 MMA，约 512 周期，需要 TMA 送来 48 KB。按 21 关 §3.1 的方法，延迟约 1000 周期时需要约 2 个 K 块在路上；答案用 4 级。

## 4. 硬件与指令

### 4.1 Tensor Memory

每个 SM 一块 TMEM：**128 个 lane（行）× 512 个 column（列）× 32 位**，共 256 KB。地址是 32 位：

```
TMEM 地址 = (lane << 16) | column

            column →  0   1   2  ...  127 128 ... 255 256 ... 511
 lane   0          ┌──────────────────────┬─────────────┬─────────┐
 lane   1          │  CTA X 的累加器       │ CTA Y 的     │ 空闲     │
  ...              │  128 × 128 FP32      │ 累加器       │         │
 lane 127          └──────────────────────┴─────────────┴─────────┘
                    ↑ alloc 返回的 base（lane 0, column c0）
```

- **分配按列**：一次分配 `ncols` 列（2 的幂，32 ≤ ncols ≤ 512），总是包含全部 128 个 lane。FP32 累加器一个元素占一个 32 位单元，所以 128×N 的累加器要 N 列；N 不是 2 的幂时向上取整（N = 96 分配 128 列）。
- **由一个 warp 执行** `tcgen05.alloc`（`.sync.aligned`，32 个线程一起），它把 base 地址写到 shared memory 中你给的位置。
- `tcgen05.relinquish_alloc_permit`：本 CTA 放弃继续分配 TMEM 的权利。执行之后，本 CTA 的任何线程再执行 `tcgen05.alloc` 都是非法的（PTX ISA 的表述）。
- `tcgen05.dealloc`：CTA 退出前**必须**释放全部列，否则后面在这个 SM 上的 CTA 分配时永远等待。
- 本关的 `cta_group::1`、M = 128 的 MMA：**累加器第 i 行在 lane i，第 j 列在 column base + j**。

### 4.2 哪个 warp 能读哪些 lane

TMEM 的 lane 被分成 4 段，每段只能被一个 warp 访问：warp w 只能访问 lane `32·(w % 4)` 到 `32·(w % 4) + 31`。所以本关用 4 个 warp（128 个线程），warp w 正好负责 C 的第 `32w .. 32w+31` 行：

```
              TMEM lane          C 的行（tile 内）     线程
 warp 0  →    0 ..  31     =     0 ..  31        lane i ↔ 第 i 行
 warp 1  →   32 ..  63     =    32 ..  63
 warp 2  →   64 ..  95     =    64 ..  95
 warp 3  →   96 .. 127     =    96 .. 127
```

### 4.3 shared memory 中的操作数：K-major + 128B swizzle

A 的 K 块是 128 行 × 64 个 BF16，B 的 K 块是 BN 行 × 64 个 BF16。TMA 用 box = {64, 行数}、`SWIZZLE_128B` 装进来，每行 128 字节，30 关 §4.3 的规则：第 r 行的 16 字节 chunk q 存在 `q ^ (r % 8)`。

```
              一行 = 64 个 K = 8 个 chunk（每个 8 个 BF16）
  row 0   │ k0-7 │ k8-15│k16-23│ ...                     │  ┐
  row 1   │k8-15 │ k0-7 │k24-31│ ...  (chunk q 在 q^1)   │  │ 一个 swizzle atom：
  ...                                                       │ 8 行 × 128 B = 1024 B
  row 7   │k56-63│ ...                   (chunk q 在 q^7) │  ┘
  row 8   │ k0-7 │ ...        （下一个 atom，同 row 0）   │  ← 距 row 0 正好 1024 B = SBO
```

MMA 读操作数时，需要知道的就是这个“规范布局”（canonical layout）：swizzle 模式、相邻两个 8 行 atom 之间的距离（SBO）。

### 4.4 shared memory 描述符（64 位）

| 位 | 字段 | 本关的值 |
|---|---|---|
| 0–13 | 起始地址 >> 4 | `shared_addr(tile + K 片偏移) >> 4` |
| 16–29 | leading dimension byte offset（LBO）>> 4 | 1（K-major + swizzle 时不使用，惯例填 1） |
| 32–45 | stride dimension byte offset（SBO）>> 4 | 1024 >> 4 = 64 |
| 46–47 | 版本 | 1（SM100 必须为 1） |
| 49–51 | base offset | 0（atom 1024 字节对齐时） |
| 52 | LBO 模式 | 0 |
| 61–63 | 布局 | 2 = `SWIZZLE_128B`（0 无，1 128B_BASE32B，4 64B，6 32B） |

例：A 的 K 块在 shared 地址 `0x400`，描述符是 `0x4000404000010040`。

**K 方向的 4 个片**：第 kk 片从第 `16·kk` 个 K 开始，即字节 `32·kk`。描述符只需把起始地址加 `32·kk` 字节（编码值加 `2·kk`），例如 `0x...0040` → `0x...0042`。为什么在 swizzle 过的行里这样做是对的？因为 swizzle 是作用在**完整地址**上的 XOR（地址 bits[4:6] ^= bits[7:9]）。MMA 按“未 swizzle 的逻辑地址 = 起始 + 行·128 + chunk·16”计算，再施加同样的 XOR；TMA 写入时用的是同一个函数。只要 atom 1024 字节对齐，逻辑地址加 32 字节就正好是 K + 16 的位置。这也是 base offset 能填 0 的前提。

### 4.5 instruction descriptor（32 位）

| 位 | 字段 | 本关的值 |
|---|---|---|
| 0–1 | 稀疏选择 | 0 |
| 2 | 稀疏 | 0 |
| 3 | 饱和 | 0 |
| 4–5 | D 格式：0 F16，1 F32 | 1 |
| 7–9 | A 格式：0 F16，1 BF16 | 1 |
| 10–12 | B 格式 | 1 |
| 13、14 | A、B 取负 | 0 |
| 15 | A 主序：0 K-major，1 MN-major | 0 |
| 16 | B 主序 | 0 |
| 17–22 | N >> 3 | 16（N = 128）或 32（N = 256） |
| 24–28 | M >> 4 | 8（M = 128） |
| 30–31 | 最大移位（只用于 `.ws`） | 0 |

M = 128、N = 256：`0x08400490`；N = 128：`0x08200490`。注意 M 从 **第 24 位**开始（第 23 位保留），把 M 放在第 23 位是常见错误。

### 4.6 MMA、commit 与 TMEM 读取

```
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, p;
```

`D = A · Bᵀ + (p ? D : 0)`。`kind::f16` 表示 16 位输入（F16/BF16）。B 以 N×K、K-major 的形式给出，正是本课程 `B[N,K]` 的布局，不需要转置。一个线程发出，指令立即返回。

```
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];
```

让 mbarrier 跟踪**本线程此前发出的全部** tcgen05 异步操作：它们全部完成后，对 bar 做一次 arrive。“完成”包括读完 shared memory 中的操作数，所以 commit 到达后：

- 该 stage 的 A、B buffer 可以被下一次 TMA 覆盖（`empty[s]`）；
- 累加器可以被读取（`mma_done`）。

```
tcgen05.ld.sync.aligned.32x32b.x32.b32 {r0, ..., r31}, [taddr];
tcgen05.wait::ld.sync.aligned;
```

`32x32b` 形状：warp 读 32 个 lane，每个线程读一个 lane 的 32 位 × N 个连续 column（`.x32` 就是 32 列，`.x8`、`.x16`… 同理）。线程 i 得到 lane `taddr.lane + i` 的第 `taddr.col .. taddr.col + 31` 列。`taddr` 的 lane 字段必须是该 warp 被允许的那一段的起点（`32·(w%4)`）。`tcgen05.ld` 也是异步的：寄存器在 `wait::ld` 之后才有效。

### 4.7 tcgen05 的同步：fence::before / after_thread_sync

tcgen05 指令是异步的，普通的线程同步（`__syncthreads`、mbarrier）本身不对它们排序。规则：

```
线程 A： tcgen05 操作 ...;  tcgen05.fence::before_thread_sync;
         同步（__syncthreads，或 mbarrier arrive → 线程 B 的 wait）
线程 B： tcgen05.fence::after_thread_sync;  tcgen05 操作 ...
```

本关用到的四处：

| 位置 | before（发出方） | 同步 | after（接收方） |
|---|---|---|---|
| alloc 后，所有线程读 base 地址 | warp 0 的 alloc | `__syncthreads` | 所有线程 |
| TMA 数据到达后发 MMA | （TMA 由 mbarrier 完成） | `wait(full[s])` | 发 MMA 的线程（可选，防御性，见下） |
| MMA 完成后读累加器 | commit（隐含） | `wait(mma_done)` | 所有线程，再 `tcgen05.ld` |
| 所有 ld 完成后 dealloc | 所有线程 | `__syncthreads` | warp 0，再 dealloc |

第二行的 after fence 不是必需的：TMA 写 shared memory 和 MMA 读 shared memory 都属于 async proxy，`full[s]` 的 mbarrier 完成已经给它们排好了顺序。CUTLASS 的 `PipelineTmaUmmaAsync::consumer_wait` 在这里只等 mbarrier、不加 fence。本关答案保留它作为防御性写法，代价可以忽略。

### 4.8 常见误用

| 误用 | 后果 |
|---|---|
| M 字段放在第 23 位，或 N 没有 >> 3 | 结果全错或 illegal instruction |
| SBO 写成 128（一行）而不是 1024（8 行） | 只有前 8 行对，或者全错 |
| 描述符的 version 位为 0 | SM100 上非法 |
| K 片偏移用了“元素数 × 2 再 >> 4 之前就加到编码值上”之类的单位错误 | 每个 K 片读错位置；`one_tile` 精确比较立刻发现 |
| shared tile 没有 1024 字节对齐 | swizzle 对不上；对齐只在部分 launch 里偶然成立，结果时对时错 |
| 发 MMA 前没有等 `full`，或 commit 前就复用 stage | 读到旧数据或半写的数据 |
| warp 读了不属于它的 lane 段 | 非法访问 |
| `tcgen05.ld` 后没有 `wait::ld` 就用寄存器 | 读到随机值 |
| warp 0 的 lane 0 在 `if (tid == 0)` 里，其他 lane 已到 `tcgen05.ld` | `.sync.aligned` 要求 warp 收敛，先 `__syncwarp()` |
| 退出前没有 dealloc | 后续 CTA 在 alloc 处永远等待（`TIMEOUT`） |
| `expect_tx` 只算 A 或只算有效部分 | 挂起 |

## 5. 调度设计

### 5.1 步骤 1：一个 tile

```
grid = 1，block = 128 线程；shared: A[128][64]、B[128][64]（各 16 KB，1024 对齐）
barriers: full、mma_done；tmem_slot

thread 0 : mbar_init(full, 1); mbar_init(mma_done, 1); fence_barrier_init
warp 0   : tmem_alloc(&tmem_slot, 128); relinquish
all      : fence::before; __syncthreads; fence::after;  tmem = tmem_slot
thread 0 : arrive_expect_tx(full, 32 KB); TMA A(0,0); TMA B(0,0)
           wait(full, 0); fence::after（可选）
           4 × mma(tmem, descA + 32kk B, descB + 32kk B, idesc, kk > 0)
           commit(mma_done)
all      : __syncwarp; wait(mma_done, 0); fence::after
warp w   : for j in 0, 32, 64, 96:  ld 32x32b.x32 (lane 32w, col j); wait::ld; 写 C 第 32w+lane 行
all      : fence::before; __syncthreads;  warp 0: fence::after; dealloc
```

步骤 1 的写回是“每个线程写自己那一行”：相邻线程写不同的行，store 不合并。步骤 3 会改掉它。

### 5.2 步骤 2：一个线程的 TMA/MMA 流水

```
prologue:  load(0) .. load(S−1)
for t in 0 .. k_tiles−1:
    s = t % S
    wait(full[s], (t/S) & 1); fence::after（可选）
    mma_k_tile(stage s, accumulate = t > 0)
    commit(empty[s])                       ← MMA t 读完 stage s 时到达
    if t ≥ 1 且 t−1+S < k_tiles:
        wait(empty[(t−1) % S], ((t−1)/S) & 1)   ← 等 MMA t−1，不是 MMA t
        load(t−1+S → stage (t−1) % S)
commit(mma_done)
```

为什么重装 **t−1** 的 stage 而不是 t 的？如果等 MMA t 完成再发下一个 load，Tensor Core 队列里就空了：MMA t+1 要等这个线程回来才能发出。先发出 MMA t、再等 MMA t−1，MMA t 在硬件里排队执行，Tensor Core 不停：

```
线程 0:  ─mma t─commit─┬wait e(t−1)┬load t+3─wait f(t+1)─mma t+1─commit─┬wait e(t)┬load t+4 ...
Tensor:  ══ MMA t−1 ══╪═══════════ MMA t ═══════════╪════════ MMA t+1 ══════...
                       └ MMA t−1 完成，e 到达 ┘        └ MMA t 完成 ┘
TMA:     ... tile t+1、t+2 在路上 ... tile t+3 在路上 ...
```

`empty[s]` 在一个线程内既是 commit 的目标也是 wait 的对象；`empty[s]` 第 n 次完成对应 stage s 上第 n 次 MMA，所以 parity 是 `((t−1)/S) & 1`。

单线程同时担任 producer 和 MMA 发射者：任何一次等待（等 TMA 数据、等 MMA 释放 buffer）都会推迟另一件事。32 关把它们拆成 TMA warp 和 MMA warp，并把 epilogue 交给另外的 warp。

步骤 2 的配置：BN = 128，4 级，每级 32 KB，共 128 KB（每 SM 1 个 CTA），TMEM 128 列。步骤 2 的 grid 已经覆盖全部 M×N tile（`blockIdx` → `(m0, n0)` 已写好），写回仍是步骤 1 的逐行 store。

### 5.3 步骤 3：TMA store epilogue

MMA 全部完成后，stage buffer 不再需要（所有 TMA load 都已被等待、所有 MMA 都已读完），拿来暂存 C：

```
warp w、列块 j（32 列）：一个 32×32 FP32 box，4 KB，放在 staging + (w·BN/32 + j)·4 KB

   tcgen05.ld 32x32b.x32 (lane 32w, col 32j) → 线程 r 有第 r 行的 32 个值
   线程 r 写 box 的第 r 行：8 个 float4，chunk q 放在位置 q ^ (r % 8)   ← C 的 tensor map 也用 128B swizzle
   fence.proxy.async; __syncwarp
   lane 0: TMA store box → C 的 (x = n0 + 32j, y = m0 + 32w)
最后：lane 0 commit; wait_group.read 0
```

- **bank conflict**：16 字节的 store 每 8 个 lane 一个 wavefront；同一波的 8 个线程 r = 0..7 写各自行的 chunk q，物理位置 `q ^ r` 两两不同，覆盖 32 个 bank。如果不 swizzle，8 个线程写同一个 chunk 位置：8 路冲突。
- **尾部**：越界的行、列由 TMA store 裁掉，不需要判断。
- **各 warp 独立**：每个 warp 写自己的 box、自己 store，不需要 CTA 级 barrier。

C 的 tensor map 要求 N·4 字节是 16 的倍数，这就是 `N % 4 == 0` 的来源；A、B 的行距要求 `K % 8 == 0`。

**调参**：BN = 256（§3.2 的 shared memory 带宽估算），4 级，每级 48 KB，共 192 KB；TMEM 256 列；暂存区需要 128 KB，可以放进 stage buffer。

## 6. 分步练习

### 步骤 1 · TODO(1a)–(1f)：封装与单 tile

- TODO(1a)：TMEM 的 alloc / relinquish / dealloc（§4.1）；
- TODO(1b)：两个描述符（§4.4、§4.5）。建议先在纸上算出本节的两个例子；
- TODO(1c)：mma、commit、ld、wait、两个 fence（§4.6、§4.7）；
- TODO(1d)：`mma_k_tile` 的 4 条 MMA（§4.4 的 K 片偏移）；
- TODO(1e)：`store_direct`（§4.2）；
- TODO(1f)：`gemm_one_tile`（§5.1）。

```bash
./lab check 31 --step 1
./lab sass 31          # 应当看到 UTCHMMA、UTMALDG、LDTM
```

`one_tile` 是整数输入的精确比较。看错误报告中的第一个错误位置：

| 错误的样子 | 先查 |
|---|---|
| 全部是 NaN | 没有写 C，或 `tcgen05.ld` 读错 lane 段 |
| 每 8 行重复、或只有前 8 行对 | SBO |
| 每 16 个 K 有规律的错误（值接近但不等） | K 片偏移 |
| 全部为 0 | `accumulate` 逻辑、MMA 没有执行（idesc 字段错位） |

### 步骤 2 · TODO(2)：K 循环流水

```bash
./lab check 31 --step 2 --case long_k
./lab check 31 --step 2
./lab bench 31 --step 2
```

`long_k` 只有一个 CTA、16 个 K 块，环形 buffer 绕 4 圈；`short_k` 只有 3 个 K 块，prologue 不能装第 4 个。

### 步骤 3 · TODO(3a)(3b)：TMA store epilogue 与调参

```bash
./lab check 31
./lab bench 31 --step 3 --case square
./lab bench 31 --step 3 --case wide
./lab profile 31 --step 3 --case square
```

TODO(3b) 从下表开始实验，每次 `check` 再 `bench`：

| 实验 | `Config<BN, Stages, true>` |
|---|---|
| 起点 | `128, 4` |
| 更宽的 N | `256, 3`、`256, 4` |
| 更多 stage | `128, 6` |

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `TIMEOUT` | `expect_tx` 字节数、parity、commit 目标写错；或者某个 CTA 没有 dealloc，后面的 CTA 在 alloc 处等待 |
| `illegal instruction` | 描述符的 version 位、idesc 的 M/N 字段、`tcgen05.ld` 的 lane 段 |
| `one_tile` 对，`small` 错 | `expect_tx` 用了有效字节数；写回越界判断 |
| `long_k` 错，`one_tile`、`small` 对 | stage 下标、parity，或在 commit 之前就重装了 stage |
| 步骤 3 只有 `tail`、`raster` 错 | TMA store 的坐标顺序（x = 列 n，y = 行 m） |
| 步骤 3 结果在 4 列的块内错位 | 暂存时 swizzle 的行号用错（应当是 box 内的行 r） |

## 7. 性能

本机实测（3 号卡，CUDA Graph；`vs cublas` 越大越好）：

| 版本 | `square` 4096³ `vs cublas` | `wide` 8192×8192×4096 `vs cublas` | `square` TFLOP/s |
|---|---|---|---|
| 21 关步骤 3（`mma.sync`） | ≈ 0.28× | ≈ 0.26× | ≈ 400 |
| 步骤 2：BN = 128，逐行写回 | ≈ 0.37× | ≈ 0.33× | ≈ 540 |
| 步骤 3 起点：BN = 128，TMA 写回 | ≈ 0.5× | ≈ 0.42× | ≈ 740 |
| 对照：BN = 256，逐行写回 | ≈ 0.5× | ≈ 0.45× | ≈ 730 |
| 步骤 3（答案）：BN = 256，4 级，TMA 写回 | ≈ 0.8× | ≈ 0.75× | ≈ 1150–1200 |
| cuBLAS | 1 | 1 | ≈ 1440 |

本机 GPU 的绝对速度在多次运行之间有时整体变慢（cuBLAS 同样变慢），比值相对稳定，请看比值。步骤 3 达到 cuBLAS 的 0.6 倍以上即说明机制成立。

**读这张表**：

- **写回和 tile 形状各占一半**。步骤 2 的逐行 store 在 `./lab profile` 里表现为 DRAM 写入约 99 MB（C 只有 67 MB：零散的部分 sector 写入）；TMA 写回后约 42 MB（其余留在 L2）。BN = 256 把 shared memory 带宽从“刚好吃满”降到约 75%（§3.2）。两件事都做，`sm__pipe_tensor_cycles_active` 从约 26% 升到约 68%。
- **调参**（`square` / `wide`，TMA 写回）：

| `Config<BN, Stages>` | smem | `square` | `wide` |
|---|---|---|---|
| `<128, 3>` | 97 KB（每 SM 2 个 CTA） | ≈ 0.34–0.6×（波动大） | ≈ 0.6× |
| `<128, 4>` | 129 KB | ≈ 0.5× | ≈ 0.42× |
| `<128, 6>` | 193 KB | ≈ 0.52× | ≈ 0.46× |
| `<256, 3>` | 145 KB | ≈ 0.62× | ≈ 0.55× |
| `<256, 4>`（答案） | 193 KB | ≈ 0.8× | ≈ 0.75× |

- **剩下的约 20%**：(1) epilogue 不与主循环重叠：每 SM 一个 CTA，读 TMEM、写 C 时 Tensor Core 空闲；(2) 单线程同时等 TMA 和 MMA；(3) `square` 有 512 个 tile、148 个 SM，3.46 波，最后一波不满；(4) 每个 CTA 都从 L2 重新读 A、B。32 关用 warp specialization、累加器双缓冲和 persistent 调度解决 (1)–(3)，33 关用 2-CTA MMA 和 TMA multicast 解决 (4)。

## 8. 自测问题

1. BN = 256 时，一个 CTA 的累加器占多少 TMEM 列？一个 SM 上最多能同时有几个这样的 CTA 持有累加器？
2. warp 2 用 `tcgen05.ld.32x32b.x32` 读累加器第 70 行、第 40–71 列。taddr 应该是多少（相对 base）？由哪个线程拿到？
3. A 的 K 块在 shared 地址 `0x8000`，写出第 3 个 K 片（kk = 3）的描述符的低 16 位。
4. 步骤 2 中，若把“重装 tile t−1 的 stage”改成“重装 tile t 的 stage，先等 `empty[t % S]`”，结果还对吗？性能会怎样？
5. 为什么步骤 3 的暂存区可以直接复用 stage buffer，而不用担心 TMA 还在往里写？
6. idesc 中 M 字段若误放在第 23 位（`8 << 23`），硬件看到的 M 和 N 分别是多少？

<details>
<summary>参考答案</summary>

1. 256 列（每列一个 FP32）。TMEM 共 512 列，最多 2 个；但本关的 192 KB shared memory 已经限制每 SM 只能放 1 个 CTA。
2. 第 70 行在 lane 70，属于 warp 2 的段（64–95）；taddr = `(64 << 16) | 40`，线程 6（70 − 64）拿到第 70 行。
3. 起始地址 `0x8000 + 3 × 32 = 0x8060`，`>> 4` = `0x806`；低 16 位就是 `0x0806`（14 位地址字段）。
4. 结果仍然对，但慢：等 `empty[t % S]` 就是等刚刚发出的 MMA t 完成，此时线程不能发 MMA t+1，Tensor Core 在两条 MMA 之间空等一个往返。
5. 线程 0 只为存在的 K 块发 load，而每个 load 都在发出对应 MMA 前被 `wait(full)` 等到；`mma_done` 到达说明所有 MMA（及其 shared memory 读取）都已完成。所以写暂存区时，既没有在途的 TMA 写，也没有 MMA 在读。
6. `8 << 23` 的第 23 位是保留位，M 字段（24–28 位）得到 4，即 M = 64；N 字段不受影响。M = 64 时只计算 C 的一半行，并且 TMEM 布局也不同，结果错误。
</details>

## 9. 延伸阅读

- PTX ISA 8.7，第 9.7.16 节 TensorCore 5th Generation Family Instructions：[Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)、[Matrix Descriptors（Shared Memory Descriptor）](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)、[Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)、[tcgen05.alloc / dealloc / relinquish_alloc_permit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)、[tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)、[tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit)、[tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)、[tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)
- FlashAttention-3 论文第 3.1–3.2 节：异步与 GEMM–softmax 重叠的硬件前提
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp`：`SmemDescriptor`、`InstrDescriptor` 的位域（本关两张表的来源）；
  - `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp`：`make_umma_desc`，K-major SW128 的规范布局与 SBO；
  - `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp`：`SM100_MMA_F16BF16_SS` 的 inline PTX；
  - `third_party/cutlass/include/cute/arch/tmem_allocator_sm100.hpp`：alloc / dealloc / relinquish；
  - `third_party/cutlass/include/cute/arch/copy_sm100.hpp`：`SM100_TMEM_LOAD_32dp32b*`（`tcgen05.ld.32x32b`）；
  - `third_party/cutlass/include/cutlass/arch/barrier.h`：`umma_arrive`（`tcgen05.commit`）。
