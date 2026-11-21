# 33 · 2-CTA MMA：cluster 与成对 CTA 的 tcgen05 GEMM

> 部分 3 · FA3 的思想在 SM100 上　｜　前置：[32_ws_gemm](../32_ws_gemm/README.md)（warp specialization、persistent、累加器双缓冲）　｜　代码：[`students/33_2cta_gemm/kernel.cu`](../../students/33_2cta_gemm/kernel.cu)

## 1. 本关目标

还是 32 关的 GEMM：`C[M,N] = A[M,K] · B[N,K]ᵀ`，BF16 输入，FP32 输出，支持域相同（`K % 8 == 0`、`N % 4 == 0`、`K > 0`）。32 关的结构（producer warp、MMA warp、4 个 epilogue warp、persistent、累加器双缓冲、分组光栅化）原样保留。本关改的是**一条 MMA 由几个 SM 一起做**：

| | 32 关 | 本关 |
|---|---|---|
| 一个输出 tile | 128 × 256，一个 CTA | 256 × 256，**两个 CTA（一个 pair）** |
| MMA 指令 | `tcgen05.mma.cta_group::1`，M = 128 | `tcgen05.mma.cta_group::2`，M = 256，**只由 leader CTA 发出** |
| A、B 在哪 | 本 CTA 的 shared memory | A、B 各一半在两个 CTA 的 shared memory |
| 每 CTA 每个 K 块从 L2 读 | 48 KB | 32 KB（step 3 的 multicast：24 KB） |
| CTA 之间 | 互不往来 | **cluster**：可以读写对方的 shared memory、到达对方的 mbarrier |

本关三步：

1. **cluster 与 pair MMA**：用 `cudaLaunchKernelEx` 启动 2-CTA cluster；两个 CTA 各装一半 A、一半 B，leader 发 `cta_group::2` 的 MMA，结果分落在两个 CTA 的 TMEM 里；一个 cluster 算一个 tile。
2. **persistent pair**：把 32 关的 persistent + 累加器双缓冲搬到 pair 上，`acc_empty` 要收齐两个 CTA 的 256 次到达，其中一半来自另一个 SM。
3. **TMA multicast**：cluster 扩到 4 个 CTA（两个 pair 并排），A 的同一块只从 L2 读一次、同时写进两个 CTA；再根据“轮数”决定什么时候值得这样做。

做完后你应该能：

- 画出 pair 的数据布局：哪一半 A、哪一半 B 在哪个 CTA，C 的哪些行落在哪个 TMEM；
- 说出每个 barrier 在哪个 CTA、谁到达、谁等待、count 是多少，以及为什么退出前要“排空”；
- 推导 pair 让每个 SM 的 shared memory 流量、L2 读取各省多少；
- 解释为什么 cluster 越大不一定越快（cluster 必须放进一个 GPC）。

FA4（42 关）会把同样的 pair 用在 attention 上：QKᵀ 与 PV 都是 2-CTA MMA。

## 2. 动机：32 关剩下的差距

32 关的答案在 B200 上约为 cuBLAS 的 0.91 倍。它的 NOTES 推断剩下的差距来自数据供给：每个 128×256 tile、每个 K 块，CTA 要从 L2 读 A 16 KB + B 32 KB = 48 KB，148 个 SM 合计接近 L2 带宽；Tensor Core 在一个 K 块里还要从 shared memory 读同样这 48 KB。

能不能让一条 MMA 更“大”、而每个 SM 搬的数据更少？

- 单个 SM 上 MMA 的 M 最大 128（TMEM 只有 128 个 lane）、N 最大 256。tile 再大就要多条 MMA、多份累加器，TMEM 和 shared memory 都已用满。
- Blackwell 的答案：**两个 SM 合做一条 M = 256 的 MMA**。每个 SM 仍只放 128 行累加器，但 B（256 列）只需各放一半：B 由两个 SM 分摊，A 各放自己的 128 行。

这与 FlashAttention 论文没有直接对应的一节；它是 FA4 在 Blackwell 上提升 attention 吞吐的手段之一（FA4 的 2-CTA QKᵀ / PV，见 42 关）。先在 GEMM 上把 cluster、pair、跨 CTA 的同步搞清楚。

## 3. 数学：pair 省了什么

### 3.1 一个 K 块的数据量

记 BK = 64（一个 128B swizzle 行），BF16 每元素 2 字节。

| | 一条“逻辑 MMA 组”的输出 | 每 SM 放的 A | 每 SM 放的 B | 每 SM 每 K 块 shared memory |
|---|---|---|---|---|
| 32 关，1 CTA | 128 × 256 | 128 × 64 = 16 KB | 256 × 64 = 32 KB | 48 KB |
| 本关，pair | 256 × 256（每 SM 128 × 256） | 128 × 64 = 16 KB | 128 × 64 = 16 KB | 32 KB |
| 本关 step 3，两个 pair 共享 A | 每 SM 128 × 256 | 16 KB（只从 L2 读 8 KB） | 16 KB | 32 KB |

每个 SM 每个 K 块做的乘加一样多：`2 × 128 × 256 × 64 ≈ 4.2 MFLOP`。所以“每字节 FLOP”（相对 L2 和 shared memory 的算术强度）：

```
32 关     : 4.2 MFLOP / 48 KB ≈ 85 FLOP/B
pair      : 4.2 MFLOP / 32 KB ≈ 128 FLOP/B     （×1.5）
pair + A 多播（L2 侧）: 4.2 MFLOP / 24 KB ≈ 171 FLOP/B   （×2）
```

直观地说：B 的每个元素在 32 关被 128 行 A 复用，在 pair 中被 256 行 A 复用（两个 SM 共用一份 B，每个 SM 只存、只从 L2 读一半）。

### 3.2 全局的 L2 → SM 流量

`square`（M = N = K = 4096），每个 tile 的 K 方向要读 A 的 BM 行与 B 的 BN 行：

```
32 关 : 512 个 128×256 tile × (128 + 256) 行 × 4096 × 2 B ≈ 1.61 GB
pair  : 256 个 256×256 tile × (256 + 256) 行 × 4096 × 2 B ≈ 1.07 GB    （×0.67）
pair + A 多播（256×512 的 cluster tile）:
        128 个 × (256 + 512) 行 × 4096 × 2 B ≈ 0.81 GB                   （×0.5）
```

NCU 的 `lts__t_sectors_srcunit_tex_op_read` 实测：step 2 ≈ 0.99 GB（略低于推导的 1.07 GB），step 3 ≈ 0.82 GB，与推导接近。

### 3.3 shared memory 预算

```
每级 stage（每 CTA）: A 16 KB + B 16 KB = 32 KB
6 级                 : 192 KB
epilogue staging     : 32 KB（与 32 关相同）
合计                 : 225 KB（上限 227 KB）
```

32 关同样的预算只放得下 4 级 48 KB；本关每级小了，可以放 6 级，流水更深。

### 3.4 轮数（wave）

persistent kernel 中，每个 cluster 依次处理 tile；所有 cluster 同时各算一个 tile 叫一“轮”。时间 ≈ 轮数 × 每轮时间。

```
square: 256 个 pair tile，74 个 pair 同时驻留  →  ⌈256 / 74⌉ = 4 轮（3.46：最后一轮只有 34 个 pair 在算，54% 空闲）
wide  : 1024 个 pair tile                      →  ⌈1024 / 74⌉ = 14 轮
```

第 5.5 节会用它决定 step 3 的 cluster 形状。

## 4. 硬件与指令

### 4.1 thread block cluster

**cluster** 是一组保证同时运行、而且放在同一个 GPC（Graphics Processing Cluster，若干 SM 的组）里的 CTA。同一 cluster 的 CTA 之间可以：

- 访问彼此的 shared memory（**distributed shared memory，DSMEM**）；
- 用 `barrier.cluster` 整体同步；
- 到达彼此的 mbarrier；接收对方发起的 TMA multicast 与 `tcgen05.commit` multicast。

每个 CTA 在 cluster 中有一个 **rank**（`%cluster_ctarank`，0 .. 大小−1）。一维 grid、一维 cluster 时，cluster 的编号是 `%clusterid.x`，cluster 总数是 `%nclusterid.x`。

**启动**。`<<<grid, block>>>` 不能在运行时指定 cluster 形状，要用 `cudaLaunchKernelEx`：

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim = grid;            // gridDim.x 必须是 cluster_x 的倍数
cfg.blockDim = block;
cfg.dynamicSmemBytes = smem;
cfg.stream = stream;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim.x = cluster_x;  attr[0].val.clusterDim.y = 1;  attr[0].val.clusterDim.z = 1;
cfg.attrs = attr;  cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, kernel, args...);
```

**约束**：

| 约束 | 说明 |
|---|---|
| 大小 | 可移植上限 8 个 CTA；更大要另设 `cudaFuncAttributeNonPortableClusterSizeAllowed`。本关用 2 和 4 |
| 放置 | 一个 cluster 的全部 CTA 必须同时放进同一个 GPC；放不下就等待 |
| 驻留数 | 用 `cudaOccupancyMaxActiveClusters` 查。B200 上本关的 kernel（225 KB smem，每 SM 1 个 CTA）：cluster 2 → **74 个**（148 SM 全用），cluster 4 → **33 个**（只用 132 SM） |
| persistent grid | grid = cluster 大小 × min(tile 数, 驻留 cluster 数)，不能直接按 SM 数算 |

cluster 4 少了 16 个 SM，是因为各 GPC 的可用 SM 数不是 4 的倍数（B200 的 148 个 SM 分布在若干 GPC 上，具体个数因芯片而异）。CUTLASS 用 **preferred cluster**（`cudaLaunchAttributePreferredClusterDimension`，`examples/73_blackwell_gemm_preferred_cluster`）让放不下大 cluster 的 GPC 退回小 cluster，但这要求 kernel 处理两种 cluster 大小与动态调度，本关不做。

### 4.2 DSMEM 地址与跨 CTA 的 mbarrier

shared memory 有两个“窗口”：`.shared::cta`（本 CTA）和 `.shared::cluster`（整个 cluster）。本 CTA 的 `.shared::cta` 地址也是合法的 `.shared::cluster` 地址（指自己）。要得到**另一个 CTA 里同一个变量**的地址，用 `mapa`：

```cpp
// p 指向本 CTA 的 shared memory；返回 CTA `rank` 中同一偏移的 .shared::cluster 地址
uint32_t remote = lab::map_shared_rank(&sm.acc_empty[a], /*rank=*/0);   // mapa.shared::cluster.u32
```

两个 CTA 运行同一个 kernel，静态 `__shared__` 与对齐后的动态 shared memory 偏移完全相同，所以“同一偏移”就是“对方的同一个变量”。

跨 CTA 到达与等待：

```cpp
// 到达另一个 CTA 的 barrier（release 到 cluster 作用域）
lab::mbar_arrive_cluster(remote);         // mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]
// 等待自己的 barrier，但到达者在别的 CTA（acquire 到 cluster 作用域）
lab::mbar_wait_cluster(&bar, parity);     // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64
```

`mbarrier.try_wait` 只能等**自己** CTA 的 barrier；需要别人知道的事，由别人来到达你的 barrier，而不是你去等别人的。

**cluster 屏障**：

```cpp
lab::cluster_arrive();   // barrier.cluster.arrive.release.aligned
lab::cluster_wait();     // barrier.cluster.wait.acquire.aligned
```

cluster 中**所有 CTA 的所有线程**都到达后，wait 才返回。它取代 32 关 setup / teardown 里的 `__syncthreads()`（4.6 节）。

### 4.3 pair 的 TMA：完成信号发到 leader

`.cta_group::2` 形式的 TMA load 把 box 写进**本 CTA** 的 shared memory，但 `complete_tx` 可以发到 **pair 中另一个 CTA** 的 mbarrier（操作数是 `.shared::cluster` 地址）：

```cpp
// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [map, {x, y}], [bar]
lab::tma_load_2d_cg2(sm.a(s), &a_map, lab::map_shared_rank(&sm.full[s], leader), x, y);
```

于是两个 CTA 的数据都“报到”leader 的 `full[s]`，而发 MMA 的线程就在 leader 里。

### 4.4 pair MMA：`tcgen05.mma.cta_group::2`

**pair**：cluster 中 rank 2p 与 2p+1 两个 CTA；偶数的叫 **leader**。PTX 只要求 pair 中**一个**线程发出这条 MMA；按约定（CUTLASS 也如此）由 leader 的一个线程发：

```cpp
constexpr uint32_t idesc = lab::make_idesc_f16(/*M=*/256, /*N=*/256, /*bf16=*/true);
lab::mma_f16_ss_cg2(d_tmem, a_desc, b_desc, idesc, accumulate);
// tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, p;
```

语义（descriptor 只描述 **leader 自己**的 buffer，硬件在 peer 的**同一偏移**读另一半）：

```
                 leader（rank 0）                       peer（rank 1）
smem A    A[m0 +   0 .. 127][k 块]  ← a_desc        A[m0 + 128 .. 255][k 块]  （同一偏移）
smem B    B[n0 +   0 .. 127][k 块]  ← b_desc        B[n0 + 128 .. 255][k 块]  （同一偏移）

D = A(256 × 16) · B(256 × 16)ᵀ：
                        ┌──────────── N = 256 列 ────────────┐
TMEM（leader）lane 0..127│ C[m0 + 0..127,   n0 + 0..255]       │  ← 用到两边的 B
TMEM（peer）  lane 0..127│ C[m0 + 128..255, n0 + 0..255]       │
                        └─────────────────────────────────────┘
```

注意 B 是按 **N 对半**分的：leader 的累加器需要**全部** 256 列 B，其中一半在 peer 的 shared memory 里。这正是“两个 SM 共用一份 B”。

**手算小例**：`one_tile` case（M = N = 256，K = 64）。C[200][10] 属于 peer（200 ≥ 128），在 peer TMEM 的 lane 72、列 10；它需要 A 的第 200 行（在 peer 的 smem，第 72 行）和 B 的第 10 行（在 leader 的 smem，第 10 行）。C[5][250] 在 leader TMEM 的 lane 5、列 250，用 leader 的 A 第 5 行与 peer 的 B 第 122 行。

**同一 kernel 的所有 tcgen05 指令必须使用同一个 `.cta_group`**。所以分配、释放、commit 也要用 `cta_group::2` 的形式：

| 指令 | 谁执行 | 说明 |
|---|---|---|
| `tcgen05.alloc.cta_group::2` | pair 中**每个** CTA 各一个 warp（本关约定用同一编号：warp 5） | 两边得到同一个列地址 |
| `tcgen05.relinquish_alloc_permit.cta_group::2` | 同上 | |
| `tcgen05.mma.cta_group::2` | leader 的一个线程 | |
| `tcgen05.commit.cta_group::2 ... .multicast::cluster [bar], mask` | leader 发 MMA 的线程 | MMA 完成后，在 mask 中每个 CTA 的同一偏移 `bar` 上各到达一次 |
| `tcgen05.dealloc.cta_group::2` | 两个 CTA 的那个 warp | 之前 leader 的 MMA 不能再写 peer 的 TMEM |
| `tcgen05.ld`、`tcgen05.fence` | 各 CTA 自己 | 没有 cta_group，只访问本 CTA 的 TMEM |

### 4.5 TMA multicast

`.multicast::cluster` 的 TMA load 从 L2 读一次 box，写进 `cta_mask` 中每个 CTA 的同一偏移：

```cpp
// ...complete_tx::bytes.multicast::cluster [dst], [map, {x, y}], [bar], cta_mask
lab::tma_load_2d_cg2_multicast(dst, &a_map, bar, /*cta_mask=*/0b0101, x, y);
```

每个目标 CTA 收到的那一份各自做一次 `complete_tx`。配合 `.cta_group::2`，那一份的完成信号发到**目标 CTA 所在 pair 的 leader** 的同一偏移（`bar` 给的是本 pair leader 的地址，硬件按目标 pair 换算）。

什么时候有用？只有当**多个 CTA 需要同一块数据**时。一个 pair 内部，A、B 两半各不相同，没有可共享的；两个 pair 并排算 `(m0, n0)` 与 `(m0, n0 + 256)` 时，它们需要完全相同的 A 行：

```
cluster of 4（step 3）:  rank 0、1 = pair 0 → C[m0 .. m0+255, n0       .. n0+255]
                         rank 2、3 = pair 1 → C[m0 .. m0+255, n0 + 256 .. n0+511]
rank 0 与 rank 2 都需要 A[m0 .. m0+127]；rank 1 与 rank 3 都需要 A[m0+128 .. m0+255]

rank 0：读 A[m0 +  0 .. 63]  multicast → rank 0、2（mask 0b0101）
rank 2：读 A[m0 + 64 ..127]  multicast → rank 0、2
rank 1：读 A[m0 +128 ..191]  multicast → rank 1、3（mask 0b1010）
rank 3：读 A[m0 +192 ..255]  multicast → rank 1、3
```

每个 CTA 从 L2 只读 8 KB 的 A，但它的 shared memory 里仍有完整的 16 KB。

### 4.6 常见误用

| 误用 | 后果 |
|---|---|
| 用 `<<<grid, block>>>` 启动，没有 cluster 属性 | 每个 CTA 自成一个 cluster，`%cluster_ctarank` 恒为 0，人人都以为自己是 leader；cta_group::2 指令没有 peer，行为未定义 |
| gridDim.x 不是 cluster 大小的倍数 | 启动失败（`cudaErrorInvalidValue` 一类） |
| setup 只用 `__syncthreads()` | peer 的 TMA / leader 的 commit 可能在对方 `mbarrier.init` 之前到达：随机挂起或错误 |
| peer 的 TMA 用自己的 `full[s]` 地址 | leader 永远等不到这 32 KB：`TIMEOUT`；peer 的 `full` 计数也乱了 |
| leader 的 `expect_tx` 只写本 CTA 的 32 KB | leader 的 MMA 可能在 peer 数据到达前开始：结果中 peer 那一半随机错误 |
| peer 也对自己的 `full[s]` 执行 `arrive_expect_tx` | 那份 barrier 永远收不到 `complete_tx`，相位不会完成；下一轮再到达就是对未完成相位的超额到达（未定义行为）。只有 leader arm |
| 两个 CTA 都发 MMA | PTX 只要求由 pair 中的**一个**线程发出（哪个 CTA 都可以，“偶数 rank 当 leader”是约定）。两边都发时每个 K 块算两遍，两个发出者还在同一块 TMEM / smem 上竞争：结果错误 |
| tcgen05 指令混用 `cta_group::1` 与 `::2` | PTX 规定同一 kernel 必须一致：编译或运行错误 |
| `commit` 不带 multicast | 只有 leader 的 `empty` / `acc_full` 到达，peer 的 producer / epilogue 永远等待 |
| `acc_empty` 的 count 仍是 128 | leader 的 128 个线程读完就算空，MMA 覆盖 peer 还没读完的累加器 |
| peer 的 epilogue 到达自己的 `acc_empty` | leader 的 MMA 等的是 leader 的那份：挂起 |
| leader 等 `acc_empty` 用 `.acquire.cta`（`lab::mbar_wait`） | 与 peer 的 `release.cluster` 不配对，内存序没有保证 |
| 不排空就退出 | 一个 CTA 已退出，另一个 CTA 发来的 commit / arrive 落到已回收的 shared memory（见 5.4） |
| 两个 CTA 的 shared memory 布局不同（例如按 rank 分配不同大小） | “同一偏移”不再是同一个变量 |

## 5. 调度设计

### 5.1 角色与存储预算（每个 CTA）

```
block = 192 线程 = 6 warp，每 SM 1 个 CTA，cluster = 2（step 3 可为 4）

 warp 0..3  epilogue   两个 CTA 都有：读本 CTA 的 TMEM（C 的本 CTA 128 行）→ staging → TMA store
 warp 4     producer   两个 CTA 都有：elect 一个 lane，TMA 装本 CTA 的 A 半与 B 半
 warp 5     MMA        两个 CTA 都 tmem_alloc_cg2(512) / dealloc；只有 leader 的一个 lane 发 MMA 和 commit

shared memory：6 级 × 32 KB + 32 KB staging + 1 KB 对齐 = 225 KB
TMEM       ：2 个累加器 × 256 列 = 512 列（每个 CTA 各自的 TMEM，各 128 lane）
```

谁做什么（一个 K 块）：

```
            leader（rank 0）                              peer（rank 1）
producer    wait empty[s]                                 wait empty[s]
            arrive_expect_tx(full[s], 64 KB)
            TMA A 半 → 本 smem，complete_tx → L.full[s]    TMA A 半 → 本 smem，complete_tx → L.full[s]
            TMA B 半 → 本 smem，complete_tx → L.full[s]    TMA B 半 → 本 smem，complete_tx → L.full[s]
MMA         wait full[s]; fence::after                    （空闲）
            4 × mma.cta_group::2（读两边 smem，写两边 TMEM）
            commit.multicast(empty[s], 0b11) ──────────┬──→ P.empty[s]
                                                        └──→ L.empty[s]
```

### 5.2 barrier 地图

每个 barrier 在两个 CTA 的同一偏移各有一份；有些只用其中一份。

| barrier | count | 用哪份 | 谁到达 | 谁等待 | 首轮 parity |
|---|---|---|---|---|---|
| `full[s]` | 1 | **只用 leader 的** | leader producer `arrive_expect_tx(64 KB)` + 两边 TMA 的 `complete_tx` | leader MMA | `phase`（0） |
| `empty[s]` | 1（step 3：2） | 两份都用 | leader MMA 的 multicast commit（step 3：两个 leader 各一次） | 各 CTA 的 producer | `phase ^ 1`（1） |
| `acc_full[a]` | 1 | 两份都用 | leader MMA 的 multicast commit（本 pair 两个 CTA） | 各 CTA 的 epilogue | `phase` |
| `acc_empty[a]` | **256** | **只用 leader 的** | 两个 CTA 共 256 个 epilogue 线程，`fence::before` + `arrive.release.cluster` | leader MMA，`wait.acquire.cluster` + `fence::after` | `phase ^ 1` |
| cluster 屏障 | 全部线程 | — | setup 末尾、teardown | 全部线程 | — |

```
              leader（rank 0）                                    peer（rank 1）
   ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
   │ full[s]  ◄── L.producer expect_tx     │◄── P 的 TMA complete_tx ───────────────────────────│
   │          ◄── L 的 TMA complete_tx      │            │ full[s]   （不用）                     │
   │ empty[s] ◄─┐                          │            │ empty[s] ◄─┐                          │
   │            ├─ L.MMA commit multicast ─┼────────────┼────────────┘                          │
   │ acc_full ◄─┘  （0b11）                 │            │ acc_full ◄─  同一条 commit             │
   │ acc_empty◄── L.epilogue × 128          │◄── P.epilogue × 128（mapa 地址，release.cluster）  │
   └──────────────────────────────────────┘            └──────────────────────────────────────┘
```

### 5.3 步骤 2：persistent pair

与 32 关步骤 2 的结构相同，tile 改为 256 × 256，以 cluster 为单位调度：

```
grid = 2 × min(tile 数, 驻留 cluster 数)；cluster c 处理 tile c, c + %nclusterid.x, ...
两个 CTA 走同一串 tile，用同样多的 K 块：RingPos 无需通信就一致

producer（两边）: ring 跨 tile 连续；对每个 tile：produce_tile（本 CTA 的两半）；最后 producer_drain
MMA（leader）   : 对每个 tile：
                    mbar_wait_cluster(acc_empty[acc], acc.phase ^ 1); fence::after
                    mma_tile → 累加器 acc；commit multicast acc_full[acc]；acc.advance
                  最后再等 kAccStages 次 acc_empty（排空）
epilogue（两边）: 对每个 tile：
                    wait 本 CTA 的 acc_full[acc]; fence::after
                    epilogue_tile：最后一次 ld 之后 fence::before + arrive_cluster(leader 的 acc_empty[acc])
```

```
一个 cluster（3 个 tile）
L.producer : │t0 k0..│t1 k0..                │t2 ...
P.producer : │t0 k0..│t1 k0..                │t2 ...         （各自等自己的 empty，按 multicast 节奏前进）
L.MMA      :  │══ t0 → acc0 ══│══ t1 → acc1 ══│┄ wait acc_empty[0] ┄│══ t2 → acc0 ══│
L.epilogue :                  │ld acc0 ·························│store│ld acc1 → store│
P.epilogue :                  │ld acc0 ·····························│store│ld acc1 → store│
                                                                    ↑ 较晚的 CTA 读完 acc0 的最后一次 ld，凑齐 L 的 acc_empty[0] 的 256 次到达
                                    两边 epilogue 各自读本 CTA 的 TMEM，同时进行
```

### 5.4 启动与退出：为什么要“排空”

**启动**：

```
thread 0     : mbar_init(...)；fence.mbarrier_init.release.cluster
warp 5       : tmem_alloc_cg2(slot, 512)；relinquish（两个 CTA 都做）
全部线程     : tcgen05.fence::before；cluster_sync()；tcgen05.fence::after；读 slot
```

为什么必须是 cluster 屏障：peer 的 TMA 会 `complete_tx` 到 leader 的 `full`，leader 的 commit 会到达 peer 的 `empty`。若某个 CTA 还没初始化，另一个 CTA 就开始发信号，结果未定义。

**退出**：一个 CTA 退出后，它的 shared memory 与 TMEM 可以立刻交给下一个 CTA。必须保证此后**没有任何来自另一个 CTA 的访问**：

| 跨 CTA 的访问 | 怎样保证已经结束 |
|---|---|
| leader 的 MMA 读 peer 的 smem、写 peer 的 TMEM | peer 的 epilogue 等到了最后一个 `acc_full`：全部 MMA 已完成 |
| leader 的 multicast commit 到达 peer（及自己）的 `empty[s]` | **producer 排空**：最后一个 tile 之后，再“获取” kStages 个槽（等 `empty`、不装数据）。每个 `empty` 都收到了最后一次 commit |
| peer 的 epilogue 远程到达 leader 的 `acc_empty` | **leader MMA 排空**：再等 kAccStages 次 `acc_empty` |
| peer 的 TMA `complete_tx` 到 leader 的 `full` | leader 的 MMA 等过每一个 `full` |
| （step 3）另一个 pair 的 multicast 写本 CTA 的 A | 本 CTA 的 `empty[s]` 等的是两个 leader 的 commit |

然后：

```
全部线程 : tcgen05.fence::before；cluster_sync()
warp 5   : tcgen05.fence::after；tmem_dealloc_cg2(tmem, 512)
```

32 关不排空也没事：commit 只到达本 CTA 自己的 barrier，CTA 等 `acc_full` 时这些 MMA 都已完成。本关的 commit 会落到**另一个 SM** 上。CUTLASS 中对应的是 `producer_tail` 与 `tmem_deallocation_result_barrier`。

### 5.5 步骤 3：A 的 multicast 与 cluster 形状

cluster 4 = 两个 pair 沿 N 并排，cluster tile 256 × 512。与步骤 2 的区别：

| | 步骤 2 | 步骤 3 |
|---|---|---|
| A 的 tensor map box | 128 行 | 64 行（每个 CTA 读一半，multicast） |
| `empty[s]` 的 count | 1 | 2：本 CTA 的 A 由两个 pair 共同写入，两个 pair 的 MMA 都读完才能重写 |
| commit `empty` 的 mask | 0b11 | 0b1111 |
| commit `acc_full` 的 mask | 0b11 | 本 pair：`0b11 << 2·pair` |
| `acc_empty` 地址 | rank 0 | 本 pair 的 leader（rank 0 或 2） |

两个 pair 共享 `empty`，因此步调一致：一个 pair 慢，另一个也要等它。

**轮数决定值不值得**（3.4 节，驻留数见 4.1 节）：两种形状下每个 pair 每轮都算一个 256×256 tile，时间 ≈ 轮数 × 每轮时间。

| case | cluster 2：tile / 74 | cluster 4：tile / 33 | 选择 |
|---|---|---|---|
| `square` | 256 / 74 → 4 轮 | 128 / 33 → 4 轮 | cluster 4（轮数相同，少读 A） |
| `wide`、`flat` | 1024 / 74 → 14 轮 | 512 / 33 → 16 轮 | cluster 2 |

所以 `run()` 的步骤 3 在 host 上算这两个轮数，`rounds3 ≤ rounds2`（rounds3 是步骤 3 的 cluster 4 形状的轮数） 时用 cluster 4 的 multicast kernel，否则用步骤 2 的 kernel。

## 6. 分步练习

### 步骤 1 · TODO(1a)–(1h)：cluster 与 pair MMA

不带 `--step` 的 `./lab check 33` / `./lab bench 33` 运行最后一步（步骤 3），在 host 上先遇到 TODO(3c)；做步骤 1、2 时请始终加 `--step 1` / `--step 2`。

- TODO(1a)：cluster 身份、cluster 屏障、`mapa`（4.1、4.2 节）；
- TODO(1b)：pair 版的 TMA、TMEM alloc/dealloc、MMA、multicast commit（4.3、4.4 节）；
- TODO(1c)：host 的 `launch_cluster`（4.1 节的 `cudaLaunchKernelEx`）；
- TODO(1d)：`mma_k_tile_pair`：idesc 的 M、N 是多少？
- TODO(1e)：`produce_tile`：谁 arm `full[s]`、arm 多少字节、TMA 的完成信号发到哪；
- TODO(1f)：`producer_drain`（5.4 节）；
- TODO(1g)：`mma_tile`：commit 带 multicast；
- TODO(1h)：setup / teardown 的同步（5.4 节）。

`gemm_pair_tile` kernel 本身（角色分派、epilogue 写哪些行）已写好；`epilogue_tile` 与 32 关相同。

```bash
./lab check 33 --step 1 --case one_tile
./lab check 33 --step 1 --case leader_only
./lab check 33 --step 1
./lab bench 33 --step 1 --case square
./lab sass 33          # UTCHMMA.2CTA、UTCBAR.2CTA.MULTICAST、UTMALDG.2D.2CTA、UTCATOMSWS.2CTA；无 spill
```

### 步骤 2 · TODO(2a)–(2d)：persistent pair

- TODO(2a)：`mbar_arrive_cluster`、`mbar_wait_cluster`（4.2 节）；
- TODO(2b)：persistent MMA：`acc_empty` 用哪种等待、末尾排空；
- TODO(2c)：`epilogue_tile` 里的释放：fence + 远程到达；
- TODO(2d)：persistent epilogue：`acc_free` 是哪个 CTA 的哪个 barrier。

```bash
./lab check 33 --step 2 --case persistent
./lab check 33 --step 2
./lab bench 33 --step 2 --case wide
./lab bench 33 --step 2 --case flat
```

### 步骤 3 · TODO(3a)–(3c)：multicast 与 cluster 形状

- TODO(3a)：`tma_load_2d_cg2_multicast`；
- TODO(3b)：`load_a` 的 multicast 分支（4.5 节的表）；
- TODO(3c)：`run()` 中按轮数选择 cluster 形状（5.5 节）。

```bash
./lab check 33 --step 3 --case one_tile
./lab check 33
./lab bench 33 --step 3 --case square
./lab profile 33 --step 3 --case square    # 看 lts__t_sectors_srcunit_tex_op_read.sum，与步骤 2 比
```

步骤 3 的全部 correctness case 轮数都很少，都会走 cluster 4。

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 启动报错 | gridDim 不是 cluster 大小的倍数；`clusterDim.y/z` 没设为 1；`numAttrs` 为 0 |
| `TIMEOUT`，`one_tile` | peer 的 TMA 信号发到了自己的 `full`；leader 的 expect_tx 字节数不是 64 KB；commit 没有 multicast 到 peer；`mapa` 的 rank 写成了自己 |
| `one_tile` 有一半行错误 | idesc 的 M 仍是 128；B 的 box 仍是 256 行（应为 128）；peer 的 B 起始行不是 `n0 + 128` |
| `one_tile` 对，`long_k` 偶发错误 | leader 只 expect 本 CTA 的 32 KB；peer 也执行了 `arrive_expect_tx` |
| `leader_only` 挂起 | peer 认为 A、B 全部越界就跳过了 TMA：越界的 box 也要发，零填充仍计入字节 |
| 步骤 2 `TIMEOUT` | `acc_empty` 的 count 不是 256；peer 的 epilogue 到达了自己的 `acc_empty` |
| 步骤 2 偶发错误 | leader 等 `acc_empty` 用了 cta 作用域；arrive 前缺 `fence::before_thread_sync` |
| 程序偶发 “illegal address” 或下一次启动出错 | 退出前没有排空（5.4 节）或 teardown 没有 cluster 同步 |
| 步骤 3 挂起 | `empty` 的 count 仍是 1，或 commit 的 mask 不含另一个 pair；multicast 的 mask 位算错 |
| 步骤 3 正确但 `wide` 变慢 | cluster 形状选择没有按轮数（5.5 节） |

## 7. 性能

本机实测（B200，2 号卡，CUDA Graph 中位数；`vs cublas` 越大越好）：

| 版本 | `square` 4096³ | `wide` 8192×8192×4096 | `flat` 8192×8192×1024 |
|---|---|---|---|
| 32 关答案（同卡） | ≈ 0.91× | ≈ 0.91× | ≈ 0.99× |
| 步骤 1：pair，一个 cluster 一个 tile | ≈ 0.91× | ≈ 0.86× | ≈ 0.73× |
| 步骤 2：persistent pair | ≈ 0.98× | ≈ 1.12× | ≈ 1.09× |
| 步骤 3：按轮数选形状 | ≈ 1.00× | ≈ 1.16×（走 cluster 2） | ≈ 1.03–1.05×（走 cluster 2） |
| 对照：强制 cluster 4 multicast | ≈ 1.00× | 比步骤 2 慢约 8% | 比步骤 2 慢约 5% |
| 步骤 3，`--cold`（每次前清 L2） | ≈ 1.01× | ≈ 1.05× | ≈ 1.10× |

cuBLAS 在 `wide` 上本机波动较大（同卡 0.39–0.43 ms），比值随之在 1.0–1.16 之间；看本关相对 32 关的时间更稳定：步骤 2 在三个 case 上都约为 32 关的 0.83–0.94 倍时间。在全部 perf case 上达到 cuBLAS 的 0.9 倍即达标。

**读这张表**：

- **步骤 1 不比 32 关快**：一个 cluster 只算一个 tile，每个 tile 都要启动 cluster、初始化、alloc、填流水，epilogue 不与主循环重叠；`flat`（epilogue 占比最大）最明显。`sm__pipe_tensor_cycles_active` 在 `square` 上约 76%（32 关约 85%）。
- **步骤 2 才显出 pair 的价值**：Tensor Core 活跃度约 88%，每 CTA 每 K 块少读三分之一（3.1 节），流水 6 级。`wide` 上它比 cuBLAS 还快。
- **步骤 3**：`lts__t_sectors_srcunit_tex_op_read` 从约 0.99 GB 降到约 0.82 GB（3.2 节）。但 cluster 4 只有 33 个能驻留（132 SM），在 `wide`、`flat` 上多出 2 轮，强制使用反而更慢；按轮数选择后，`square` 拿到 multicast 的约 1.5%，其余保持步骤 2 的速度。NCU 锁频时 `square` 上 cluster 4 反而略慢（约 2%）：这些差别与功耗、频率有关，不只取决于 L2 流量。

**要看的指标**（`./lab profile 33 --step n --case square`）：

| 指标 | 含义 |
|---|---|
| `sm__pipe_tensor_cycles_active` | Tensor Core 忙的比例：步骤 1 → 2 的主要变化 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | SM 从 L2 读的扇区（32 B）：步骤 2 → 3 的变化 |
| `dram__bytes_read.sum` | HBM 读：`wide` 上看分组光栅化 |

## 8. 自测问题

1. pair MMA 中 leader 的累加器 C[m0 + 3][n0 + 200] 用到哪个 CTA 的哪一行 B？
2. 为什么 leader 的 `full[s]` 要 expect 64 KB，而 peer 的 `full[s]` 完全不用？
3. `acc_empty` 为什么只在 leader 里用、count 为什么是 256？为什么 leader 的等待要用 `acquire.cluster`？
4. 32 关退出前不用排空 `empty`，本关为什么必须？如果只做 teardown 的 cluster 同步、不排空，会出什么问题？
5. 步骤 3 中 `empty[s]` 的 count 为什么是 2？如果仍是 1 会怎样？
6. 若一台 GPU 上 cluster 4 能驻留 37 个（全部 148 SM），`wide` 应该选哪种形状？

<details>
<summary>参考答案</summary>

1. B 按 N 对半：n 的第 200 列 ≥ 128，在 peer 的 shared memory，B 的第 n0 + 200 行存在 peer 的 B buffer 的第 72 行。
2. MMA 由 leader 的线程发出，它只能等自己 CTA 的 barrier；两个 CTA 的 TMA 都把完成信号发到 leader 的 `full[s]`，所以它要等两个 CTA 共 2 × 32 KB。peer 没有 MMA 线程，它的 `full` 没人等。
3. 发 MMA 的线程在 leader，它只能等 leader 的 barrier。两个 CTA 各有 128 个 epilogue 线程，各读本 CTA 的 128 行，全部读完累加器才空。peer 的到达是 `release.cluster`，要与 `acquire.cluster` 配对，才能保证 peer 的 TMEM 读（经 fence 排序）发生在 leader 的下一次 MMA 写之前。
4. 32 关的 commit 只到达本 CTA 的 barrier；本关 leader 的 commit 会到达 peer 的 `empty`，peer 的 epilogue 会到达 leader 的 `acc_empty`。最后几次 commit/arrive 没人等，可能在对方 CTA 退出后才落地，写进已回收（甚至已分给新 CTA）的 shared memory。cluster 同步只排序普通线程的操作，不等异步的 commit 到达。
5. CTA X 的 `empty[s]` 放行的是 X 的 multicast，它同时写 X 与 X ± 2（另一个 pair 中同位置的 CTA）的 A buffer。这两份分别被两个 pair 的 MMA 读，所以要两个 leader 都 commit 过才能重写。count 为 1 时，先读完的那个 pair 的 commit 就放行，另一个 pair 可能还在读：偶发错误；多出的那次到达还会提前推进下一个相位。
6. cluster 4：512 / 37 → 14 轮，与 cluster 2 的 14 轮相同，轮数不吃亏，还少读 A。本关 `run()` 的轮数比较会自动选它。
</details>

## 9. 延伸阅读

- FlashAttention-4 / Blackwell：Tri Dao 等人关于 FA4 的介绍，2-CTA 的 QKᵀ 与 PV（42 关）。
- PTX ISA 8.7：
  [Thread Block Clusters 与 Distributed Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays)、
  [barrier.cluster](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster)、
  [mapa](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mapa)、
  [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)、
  [cp.async.bulk.tensor（.cta_group、.multicast::cluster）](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)、
  [tcgen05.mma（cta_group::2）](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)、
  [tcgen05.alloc](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)、
  [tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)。
- CUDA Programming Guide：[Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)、[Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory)；Runtime API `cudaLaunchKernelEx`、`cudaOccupancyMaxActiveClusters`。
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp`：`SM100_MMA_F16BF16_2x1SM_SS`（pair MMA 的 PTX 形式）；
  - `third_party/cutlass/include/cute/arch/copy_sm100_tma.hpp`：`SM100_TMA_2SM_LOAD_2D`、`SM100_TMA_2SM_LOAD_MULTICAST_2D`（`Sm100MmaPeerBitMask`）；
  - `third_party/cutlass/include/cute/arch/tmem_allocator_sm100.hpp`：`Allocator2Sm`；
  - `third_party/cutlass/include/cutlass/arch/barrier.h`：`umma_arrive_multicast_2x1SM`、`ClusterBarrier`；
  - `third_party/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp`：`PipelineTmaUmmaAsync`（只有 leader `arrive_and_expect_tx`、multicast mask 的计算）；
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`：`is_mma_leader_cta`、`tmem_deallocation_result_barrier`、`producer_tail`；
  - `third_party/cutlass/examples/70_blackwell_gemm`、`73_blackwell_gemm_preferred_cluster`。
