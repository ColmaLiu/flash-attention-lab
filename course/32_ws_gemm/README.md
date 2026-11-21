# 32 · warp specialization：分工流水的 persistent GEMM

> 部分 3 · FA3 的思想在 SM100 上　｜　前置：[31_tcgen05_gemm](../31_tcgen05_gemm/README.md)（以及 [30_tma_mbarrier](../30_tma_mbarrier/README.md) 的 full/empty 环形 buffer）　｜　代码：[`students/32_ws_gemm/kernel.cu`](../../students/32_ws_gemm/kernel.cu)

## 1. 本关目标

还是 31 关的 GEMM：`C[M,N] = A[M,K] · B[N,K]ᵀ`，BF16 输入，FP32 输出，同样的支持域（`K % 8 == 0`、`N % 4 == 0`）。指令一条都不新增：TMA、mbarrier、`tcgen05.mma/commit/ld` 全部来自 31 关，由 [`include/lab/tcgen05.cuh`](../../include/lab/tcgen05.cuh)、[`tma.cuh`](../../include/lab/tma.cuh)、[`mbarrier.cuh`](../../include/lab/mbarrier.cuh) 提供。本关改的是**谁在什么时候做什么**：

| | 31 关 | 本关 |
|---|---|---|
| 谁发 TMA | thread 0 | 专门的 **producer warp** |
| 谁发 MMA | 同一个 thread 0 | 专门的 **MMA warp**（一个 lane） |
| 谁写回 C | 全部 4 个 warp，在主循环结束后 | 专门的 **epilogue warps**，与下一个 tile 的主循环同时进行 |
| 一个 CTA 算几个 tile | 1 | 多个（**persistent**，grid = SM 数） |
| TMEM 累加器 | 1 个 | 2 个（**双缓冲**） |

本关三步：

1. **角色分工**：一个 CTA 一个 tile，producer warp + MMA warp + 4 个 epilogue warp，通过 full/empty barrier 交接 stage；
2. **persistent + 累加器双缓冲**：每个 CTA 循环处理多个 tile，epilogue 读累加器 0 时，MMA 已经在往累加器 1 里算下一个 tile；
3. **tile 顺序**：分组光栅化，让同时在算的 148 个 tile 共用更少的 A、B 数据，减少 HBM 读取。

做完后你应该能：

- 画出三个角色的时间线，指出每一处等待在等谁；
- 说出 4 类 barrier（`full`、`empty`、`acc_full`、`acc_empty`）各由谁到达、谁等待、初始化计数是多少、首轮等哪个 parity；
- 让环形 buffer 与累加器的相位跨 tile 连续，而不是每个 tile 重置；
- 把 31 关 cuBLAS 的约 0.8 倍提到约 0.9 倍；并说清剩下的差距要靠什么（33 关）。

## 2. 动机：31 关的串行

31 关的 kernel 里，一个线程同时担任两个角色，所有 warp 在主循环之后才做 epilogue。把一个 CTA 的时间线画出来：

```
31 关（每 SM 1 个 CTA）
thread 0  : ld ld ld ld ─wait f─mma─commit─wait e─ld─wait f─mma ... mma─commit(done)
warps 0-3 : ······························· 空闲 ·······························│ld TMEM│写 C│
Tensor    :            ══ MMA ══      ══ MMA ══      ══ MMA ══ ... ══ MMA ══│   空闲    │
                                                                          ↑ 下一个 CTA 要等本 CTA 退出
                                                                            才能开始：再从空 buffer 填起
```

三个问题：

1. **一个线程等两件事**。等 TMA 数据（`full`）时不能去重装 buffer，等 MMA 释放 buffer（`empty`）时不能发下一条 MMA。31 关用“重装 t−1 的 stage”的顺序缓解，但仍然是一条串行的指令流。
2. **epilogue 不与主循环重叠**。128×256 的 FP32 tile 有 128 KB 要写回；写回期间 Tensor Core 空闲。31 关的 shared memory 用了 192 KB，每个 SM 只能放 1 个 CTA，靠“另一个 CTA 来填空”也做不到。
3. **每个 tile 都从头开始**。CTA 启动、barrier 初始化、TMEM 分配、流水线从空填满，每个 tile 都要付一次。

FlashAttention-3 论文 §3.1 把解决办法叫 **producer–consumer asynchrony**：用 warp specialization 把“搬数据”和“算”分给不同的 warp，二者只通过 barrier 交接，各自按自己的节奏前进；再用 pingpong 调度让一组 warp 做 GEMM 的同时另一组做 softmax。FA3 在 Hopper 上的对应关系，与本关在 Blackwell 上的对应关系：

| FA3（Hopper） | 本关（Blackwell） | 34 关 / 40 关（attention） |
|---|---|---|
| producer warp：TMA 读 Q、K、V | producer warp：TMA 读 A、B | 同左 |
| consumer warpgroup 发 `wgmma`，累加器在寄存器 | **MMA warp 的一个 lane** 发 `tcgen05.mma`，累加器在 TMEM | 同左 |
| consumer warpgroup 自己做 softmax、写回 | **epilogue warps** 读 TMEM、写回 | softmax / correction warpgroup 读 S、写 P |
| pingpong：两个 warpgroup 交替 GEMM 与 softmax | **TMEM 双缓冲**：MMA 写累加器 1 时，epilogue 读累加器 0 | S 双缓冲：QKᵀ 与 softmax 重叠 |
| `setmaxnreg`：producer 让出寄存器 | 本关不需要（§4.5） | 40 关使用 |

Blackwell 上分工更彻底：MMA 由一个线程发出、累加器不占寄存器，所以“发 MMA”与“消费累加器”自然分成两个角色。本关在 GEMM 上把这套结构搭好，34 关把 epilogue 换成 softmax。

## 3. 数学：什么时候重叠值得做

### 3.1 一个 tile 的两段时间

一个 128×256 的 tile，K 方向 `k_tiles = K / 64` 个 K 块，每块 4 条 128×256×16 的 MMA。按 31 关 §3.2 的估算（每 SM 每周期约 8192 FLOP，约 1.9 GHz）：

```
T_main ≈ k_tiles × 4 × 128 周期 = k_tiles × 512 周期
T_epi  ≈ 128 KB 的 C / 每 SM 分到的 HBM 写带宽（8 TB/s ÷ 148 ≈ 54 GB/s）≈ 2.4 µs ≈ 4600 周期
```

| case | K | k_tiles | T_main | T_epi / T_main |
|---|---|---|---|---|
| `square` 4096³、`wide` | 4096 | 64 | ≈ 32800 周期 ≈ 17 µs | ≈ 14% |
| `flat` 8192×8192×1024 | 1024 | 16 | ≈ 8200 周期 ≈ 4.3 µs | ≈ 56% |

不重叠时每个 tile 花 `T_main + T_epi`，重叠后（稳态）花 `max(T_main, T_epi)`。K 越小，重叠越值钱。这是本关加入 `flat` perf case 的原因（这只是量级估算，第 7 节有实测）。

### 3.2 环形 buffer 的相位不变式

30 关的规则：一个有 S 个槽的环，第 u 次使用（u 从 0 起，所有 tile 累计）落在

```
slot  = u % S
phase = (u / S) & 1           等待方（consumer）等 parity = phase
                              释放方（producer）等对面的 barrier，parity = phase ^ 1
```

`phase ^ 1` 在第一轮（phase = 0）时是 1：新初始化的 barrier 处于相位 0，“前一个相位”（parity 1）视为已完成，所以第一轮不阻塞，正是“所有槽一开始都是空的”。

**相位跨 tile 不清零。** 例：S = 4，每个 tile 6 个 K 块。CTA 的第 3 个 tile（tile 序号 2）的第一个 K 块是第 u = 12 次使用：slot 0，phase 1。如果每个 tile 都从 (slot 0, phase 0) 重新开始，第 2 个 tile 的第一块会落在 slot 0、等 parity 0，而这时 slot 0 已经完成了 2 次（phase 0 和 1），parity 0 的等待看到的是“第 0 相位已完成”，立刻返回，读到的是旧数据。

代码里每个角色持有一个 `RingPos{index, phase}`，每用一次 `advance()`，跨 tile 一直往前走：

```cpp
template <int N> struct RingPos {
  int index = 0;  uint32_t phase = 0;
  void advance() { if (++index == N) { index = 0; phase ^= 1; } }
};
```

producer 和 MMA warp 各有一份 stage 环的 `RingPos`，MMA warp 和 epilogue 各有一份累加器环（N = 2）的 `RingPos`。它们不通信，因为大家处理同一串 tile、同样多的 K 块，使用次数天然一致。

### 3.3 tile 顺序与 L2

persistent kernel 里，CTA b 依次处理 tile `b, b + 148, b + 296, ...`。任一时刻，148 个 SM 正在算的是连续的约 148 个 tile 序号。tile 序号到 `(mb, nb)` 的映射决定这 148 个 tile 需要哪些 A 的行块（每块 128 × K）与 B 的行块（每块 256 × K）。

`wide`（M = N = 8192，K = 4096）：A 的一个行块 1 MB，B 的一个行块 2 MB，共 64 × 32 个 tile。

| 顺序 | 148 个 tile 的形状 | 需要的 A | 需要的 B | 合计 |
|---|---|---|---|---|
| 行优先（nb 变得最快） | ≈ 4.6 行 × 32 列 | 4.6 MB | 64 MB（全部 B） | ≈ 69 MB |
| 分组 G = 8（组内 mb 变得最快） | 8 行 × 18.5 列 | 8 MB | 37 MB | ≈ 45 MB |
| 分组 G = 32 | 32 行 × 4.6 列 | 32 MB | 9.3 MB | ≈ 41 MB |

L2 是 126 MB，但它要同时装下正在读的、刚读过的和 C 的写回；工作集越小，越多的 A、B 读取命中 L2。行优先时，每算完一行 tile，整个 B（64 MB）要再从头扫一遍，很可能已被挤出 L2。分组后同时在用的数据小得多（第 7 节有实测的 HBM 读取量）。

`square`（4096³）的 A + B 一共只有 64 MB，任何顺序都放得进 L2，分组没有好处。

## 4. 硬件与指令

本关没有新指令。下面是“把已有指令分给不同 warp”时要注意的事。

### 4.1 选一个 lane：`elect.sync`

producer 与 MMA 的循环只要一个线程跑。`lab::elect_one()`（`elect.sync`）让整个 warp 一起调用、恰好一个 lane 得到 true：

```cpp
if (warp == kProducerWarp) {
  if (lab::elect_one()) {       // 一个 lane 跑整个循环
    ...
  }
  __syncwarp();                 // 其余 31 个 lane 在这里等它，再一起去 __syncthreads
}
```

与 `if (threadIdx.x == 128)` 等价，但不需要写死线程号，编译器也知道条件在 warp 内只有一个 lane 为真。

### 4.2 一个 barrier 可以有多个到达者

`mbar_init(bar, count)` 的 count 是每个相位需要的到达次数。本关的四类 barrier：

| barrier | count | 谁到达 | 怎么到达 | 谁等待 |
|---|---|---|---|---|
| `full[s]` | 1 | producer | `arrive_expect_tx(48 KB)`，TMA 再 `complete_tx` | MMA |
| `empty[s]` | 1 | MMA | `tcgen05.commit`：该 stage 的 MMA 读完时到达 | producer |
| `acc_full[a]` | 1 | MMA | `tcgen05.commit`：该 tile 全部 MMA 完成时到达 | 128 个 epilogue 线程 |
| `acc_empty[a]` | **128** | 每个 epilogue 线程 | `tcgen05.fence::before_thread_sync` + `mbar_arrive` | MMA |

`acc_empty` 必须等**全部** 128 个线程都读完各自那一行才算空，所以 count = 128（CUTLASS 同样用 epilogue 线程数）。

一个 barrier 可以有很多个等待者：`acc_full[a]` 被 128 个线程同时等待，各自都能看到同一个相位完成。

### 4.3 `tcgen05.commit` 是异步的“释放”

MMA warp 从不等 MMA 完成。它发出一个 K 块的 4 条 MMA，接着 `tcgen05.commit(&empty[s])`，然后立刻去等下一个 `full`。commit 只是让 `empty[s]` 记住“当这些 MMA 都完成时到达一次”。producer 在另一个 warp 里等 `empty[s]`，MMA 读完 stage s 时它就被唤醒。tile 结束时 `tcgen05.commit(&acc_full[a])` 同理：它跟踪**这个线程此前发出的全部** MMA，所以到达时整个 tile 都完成了。

### 4.4 TMEM 的双缓冲与 fence

两个 128×256 FP32 累加器 = 512 列 = 整块 TMEM（128 lane × 512 列 × 32 位）。分配一次 512 列，累加器 a 在列 `base + 256a`。每 SM 只能放 1 个本 kernel 的 CTA（shared memory 225 KB），所以不会和自己的其他 CTA 争 TMEM。

跨角色访问 TMEM 时，31 关 §4.7 的 fence 规则照样成立：

```
epilogue 线程：tcgen05.ld ... wait::ld;  tcgen05.fence::before_thread_sync;  mbar_arrive(acc_empty[a])
MMA 线程    ：mbar_wait(acc_empty[a]);  tcgen05.fence::after_thread_sync;   tcgen05.mma → 累加器 a
```

**早释放**：epilogue 读完 TMEM 就到达 `acc_empty`，这时它的 TMA store 还在写 C。这是安全的：store 读的是 shared memory 里的 staging box，不再碰 TMEM。

本关用 `lab::tmem_ld_32x32b_x32_sync`：`tcgen05.ld` 与 `tcgen05.wait::ld` 放在同一个 asm 块里，返回时寄存器就有效。分开写成两个 asm 时，编译器不知道寄存器要等 `wait::ld` 才有效，可能把只用寄存器的计算挪到 wait 之前。

### 4.5 寄存器与 `setmaxnreg`

FA3 用 `setmaxnreg` 让 producer warpgroup 把寄存器让给 consumer。本关不需要：6 个 warp 每线程 56–72 个寄存器，远小于寄存器文件（64K × 32 位 / 192 线程 ≈ 340 个/线程），也没有 spill（`./lab sass 32` 可查）。`setmaxnreg` 要求以 warpgroup 为单位调整，会在 40 关（softmax warpgroup 需要大量寄存器）讲。

### 4.6 常见误用

| 误用 | 后果 |
|---|---|
| 每个 tile 重新 `RingPos ring;`（相位清零） | 第 2 个 tile 起读到旧数据，或者挂起；`persistent` case 立刻发现 |
| producer 首轮等 `empty` 用 parity 0 | 永远等待（`TIMEOUT`）：还没有任何 MMA 释放过 |
| `acc_empty` 初始化为 1，128 个线程都 arrive | 第一个线程读完就“空”了，MMA 覆盖其他线程还没读的行；其余 127 次到达还会推进后面的相位 |
| epilogue 没有 `fence::before_thread_sync` 就 arrive | TMEM 读与下一 tile 的 MMA 写之间无序，偶发错误 |
| MMA warp 只用 `if (lane == 0)` 而其余 lane 直接去 `__syncthreads` | 依赖独立线程调度的行为，可读性差；用 `elect_one()` + `__syncwarp()` |
| 在 `tcgen05.commit` 前换了发 MMA 的线程 | commit 只跟踪执行它的线程发出的 MMA |
| 一个 staging box 还在被 TMA store 读，就被下一块覆盖 | C 中随机出现下一块的值；先 `bulk_wait_group_read<1>` |
| epilogue 退出前没有 `bulk_wait_group_read<0>` | CTA 退出后 shared memory 被回收，store 读到垃圾 |
| tile 序号映射不是一一对应（分组的最后一组行数不足时） | 有的 tile 没算、有的算两遍；`persistent` case（31 行 tile，G = 8 时最后一组 7 行）会发现 |

## 5. 调度设计

### 5.1 角色、存储预算

```
block = 192 线程 = 6 个 warp，每 SM 1 个 CTA，tile 128 × 256，BK = 64

 warp 0..3  epilogue   warp w 读 TMEM lane 32w..32w+31 = C 的第 32w.. 行
 warp 4     producer   elect 一个 lane：TMA
 warp 5     MMA        tmem_alloc(512) / dealloc；elect 一个 lane：MMA + commit

shared memory（动态，1024 对齐）
 stages    4 × (A 128×64 + B 256×64) BF16 = 4 × 48 KB = 192 KB
 staging   4 个 warp × 2 个 32×32 FP32 box = 32 KB
 合计      225 KB（上限 227 KB）；barrier 与 TMEM 地址槽是静态 shared memory

TMEM        2 个累加器 × 256 列 = 512 列（全部）
```

预算已经顶满：5 级 stage（240 KB）或 3 个累加器（768 列）都放不下。31 关的 epilogue 复用 stage buffer 暂存 C；persistent 时主循环一直在用 stage buffer，所以 epilogue 需要自己的 staging。

### 5.2 步骤 1：一个 CTA 一个 tile

```
setup   : thread 0 初始化 barrier；warp 5 tmem_alloc；fence / __syncthreads / fence
producer: for t in 0..k_tiles-1:  wait empty[s] (phase^1); arm full[s]; TMA A, B → stage s; advance
MMA     : for t in 0..k_tiles-1:  wait full[s] (phase); fence::after; 4 × mma; commit empty[s]; advance
          commit acc_full[0]
epilogue: wait acc_full[0] (parity 0); fence::after; 8 个 32 列块：ld TMEM → staging → TMA store
teardown: fence::before; __syncthreads; warp 5: fence::after; dealloc
```

```
步骤 1 的一个 CTA
producer : ld0 ld1 ld2 ld3 ┄wait e0┄ ld4 ┄wait e1┄ ld5 ...  ld63
MMA      :     ┄wait f0┄ mma0 c  mma1 c  mma2 c  mma3 c  mma4 ...  mma63 c(acc_full)
epilogue : ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ wait acc_full ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│drain + store│
Tensor   :           ════════════════════════════════════════════════════│   空闲      │
```

producer 和 MMA 不再互相等待：producer 一直跑在 MMA 前面 4 个 stage，MMA 只在数据没到时才停。但 epilogue 仍在主循环之后，而且下一个 CTA 要等本 CTA 退出。

### 5.3 步骤 2：persistent + 累加器双缓冲

```
grid = min(tile 数, SM 数)；CTA b 处理 tile b, b + gridDim.x, ...

producer: ring 跨 tile 连续；对每个 tile：produce_tile
MMA     : ring、acc 跨 tile 连续；对每个 tile：
            wait acc_empty[acc] (acc.phase ^ 1); fence::after
            mma_tile → 累加器 acc，结束时 commit acc_full[acc]；acc.advance
epilogue: acc 跨 tile 连续；对每个 tile：
            wait acc_full[acc] (acc.phase); fence::after
            drain：最后一次 ld 之后 fence::before + arrive acc_empty[acc]；继续 store
            acc.advance
```

```
步骤 2 的一个 CTA（3 个 tile）
producer : │tile0 k0..k63 ··························│tile1 k0..k63 ························│tile2 ...
MMA      :  │═════ tile0 → acc0 ═════│═════ tile1 → acc1 ═════│═════ tile2 → acc0 ═════│
                                     ↓ acc_full[0]            ↓ acc_full[1]          ↑ 需要 acc_empty[0]：
epilogue : ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│ld acc0│store t0 ···│┄┄┄┄┄┄┄┄┄┄│ld acc1│store t1 ···│   tile0 的 epilogue
                                             ↑ acc_empty[0]（早释放）                       早已读完
Tensor   :   ══════════════════════════════════════════════════════════════════════════  不再停
```

- producer 在 MMA 还在算 tile 0 的最后几个 K 块时，就开始装 tile 1 的前 4 个 stage：流水线不再“从空填起”。
- MMA 算 tile 1 时写 acc1，epilogue 同时读 acc0：这就是 §3.1 的 `max(T_main, T_epi)`。
- MMA 只有在 epilogue 落后两个 tile 时才会在 `acc_empty` 上停下。

### 5.4 步骤 3：分组光栅化

```
tile 序号 → (mb, nb)，m_tiles = 5，n_tiles = 3，G = 2

            nb=0  nb=1  nb=2
   mb=0      0     2     4      ┐ 组 0：先往下（mb+1），再往右
   mb=1      1     3     5      ┘
   mb=2      6     8    10      ┐ 组 1
   mb=3      7     9    11      ┘
   mb=4     12    13    14        组 2：只剩 1 行，组内行数 = min(G, m_tiles − first_m)
```

```
per_group = G · n_tiles;  group = tile / per_group;  first_m = group · G
rows      = min(G, m_tiles − first_m);  in_group = tile − group · per_group
mb = first_m + in_group % rows;  nb = in_group / rows
```

三个角色都用同一个 `tile_coords`，顺序变了它们仍然一致。

### 5.5 为什么不调 BN 和 stage 数

31 关已经说明 BN = 256 比 128 好（shared memory 带宽）；本关预算又已顶满。第 7 节的对照表给出几个更小的配置，供你确认这一点。

## 6. 分步练习

### 步骤 1 · TODO(1a)(1b)(1c)：角色分工

- TODO(1a)：`produce_tile`（§5.2 的 producer 行；首轮 parity 见 §3.2）；
- TODO(1b)：`mma_tile`（MMA 行；`accumulate` 在哪个 K 块为 false？用什么释放 stage？）；
- TODO(1c)：`gemm_ws_tile` 的 epilogue 分支：等 `acc_full[0]`，再调用给好的 `epilogue_tile`（31 关 step 3 的 TMA store 写回，改成每 warp 两个 staging box 轮换）。

barrier 初始化、TMEM 分配、角色分派、teardown 已写好。

```bash
./lab check 32 --step 1 --case one_tile
./lab check 32 --step 1
./lab bench 32 --step 1 --case square
./lab sass 32          # 无 spill；UTMALDG、UTCHMMA、UTCBAR、LDTM、UTMASTG
```

### 步骤 2 · TODO(2a)–(2d)：persistent 与累加器双缓冲

- TODO(2a)：persistent producer：一个 `RingPos` 管全部 tile；
- TODO(2b)：persistent MMA：`acc_empty` 的等待（parity？）、fence、写入第 `acc.index` 个累加器；
- TODO(2c)：persistent epilogue：等 `acc_full[acc.index]`，调用 `epilogue_tile` 并传入 `acc_empty`；
- TODO(2d)：`epilogue_tile` 里的早释放：fence + arrive。

```bash
./lab check 32 --step 2 --case persistent
./lab check 32 --step 2
./lab bench 32 --step 2 --case square
./lab bench 32 --step 2 --case flat
```

`persistent` case 有 372 个 tile：一部分 CTA 处理 3 个 tile，累加器 0 → 1 → 0；每个 tile 6 个 K 块，环形 buffer 的位置在 tile 之间错开。只有相位真正跨 tile 连续时它才对。

### 步骤 3 · TODO(3a)(3b)：tile 顺序

- TODO(3a)：`tile_coords` 的分组分支（§5.4）；
- TODO(3b)：选 `kGroupM`。骨架从 16 开始；分别试 4、8、16、32，三个 perf case 都测，与步骤 2（行优先）比较。

```bash
./lab check 32 --step 3 --case persistent
./lab check 32
./lab bench 32 --step 3 --case wide
./lab profile 32 --step 3 --case wide     # 看 dram__bytes_read.sum
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `TIMEOUT`，步骤 1 | producer 首轮等 `empty` 用了 parity 0；`expect_tx` 字节数；MMA 没有 commit `acc_full` |
| `TIMEOUT`，步骤 2，小 case 正常 | MMA 首轮等 `acc_empty` 用了 parity 0；epilogue 没有到达 `acc_empty`，或到达次数与 count 不符 |
| 步骤 1 全对，步骤 2 只有 `persistent` 错 | 每个 tile 重置了 `RingPos`；累加器下标与 epilogue 不一致；`tile_coords` 三个角色调用的参数不同 |
| 步骤 2 偶发错误、多跑几次结果不同 | `acc_empty` 的 count 不是 128；arrive 之前缺 `tcgen05.fence::before_thread_sync`；arrive 在 `wait::ld` 之前 |
| 步骤 3 有的块是 NaN（输出预先填了 NaN） | 分组映射漏了 tile：检查最后一组 `rows` |
| 步骤 3 正确但比步骤 2 慢 | 在 `square`、`flat` 上是正常的（§7）；看 `wide` |

## 7. 性能

本机实测（B200，5 号卡，CUDA Graph；`vs cublas` 越大越好）：

| 版本 | `square` 4096³ | `wide` 8192×8192×4096 | `flat` 8192×8192×1024 |
|---|---|---|---|
| 31 关答案 | ≈ 0.84× | ≈ 0.75× | — |
| 步骤 1：角色分工 | ≈ 0.89× | ≈ 0.8× | ≈ 0.78× |
| 对照：persistent，**单**累加器 | ≈ 0.92× | ≈ 0.85× | ≈ 0.89× |
| 步骤 2：persistent + 双缓冲 | ≈ 0.93× | ≈ 0.87× | ≈ 0.95–0.98× |
| 步骤 3：分组 G = 8 | ≈ 0.92× | ≈ 0.91× | ≈ 0.95× |
| cuBLAS | 1 | 1 | 1 |

本机 GPU 偶尔整体变慢（cuBLAS 同样变慢），请看比值。步骤 3 在全部 perf case 上达到 cuBLAS 的 0.85 倍以上即达标。

**读这张表**：

- **步骤 1**：同样的 tile、同样的 stage 数，只是分工，`square` 从 0.84 到 0.89。`./lab profile 32 --step 1 --case square` 中 `sm__pipe_tensor_cycles_active` 约 80%（31 关约 68%）。
- **persistent 与双缓冲各管什么**：单累加器的 persistent 版本已经拿到大部分收益（省掉 CTA 启动和流水线重新填充，且累加器读完就释放）。双缓冲在 `square` 上只多约 1%，在 `flat` 上多约 10%：正好对应 §3.1 的 `T_epi / T_main`（14% 对 56%）。
- **分组光栅化**（相对步骤 2 的时间变化）：

| G | `square` | `wide` | `flat` | `wide` 的 HBM 读取 |
|---|---|---|---|---|
| 0（步骤 2） | — | — | — | ≈ 981 MB |
| 8 | 慢约 1.5% | 快约 6% | 慢约 1–2% | ≈ 507 MB |
| 16 | 慢约 5% | 快约 8% | 慢约 4% | |
| 32 | 慢约 3% | 快约 10% | 慢约 4% | ≈ 280 MB |

`wide` 的 A + B 是 128 MB，超过 L2：行优先时 HBM 读取约为 A + B 的 7.7 倍，G = 32 时约 2.2 倍。`square`、`flat` 的 A + B 放得进 L2，分组反而稍慢，原因我们没有查明（不要把它当结论）。G = 8 让三个 case 中最差的比值最高。

- **对照：更小的配置**（步骤 3 的 kernel，G = 32）：

| `Config<BN, Stages, AccStages>` | `square` | `wide` |
|---|---|---|
| `<256, 4, 2>`（答案） | ≈ 0.91× | ≈ 0.93× |
| `<256, 3, 2>` | ≈ 0.81× | ≈ 0.79× |
| `<128, 6, 2>` | ≈ 0.89× | ≈ 0.81× |
| `<128, 4, 4>` | ≈ 0.68× | ≈ 0.63× |

- **剩下的约 8%**：(1) 每个 tile 每个 K 块从 L2 读 48 KB，148 个 SM 合计逼近 L2 带宽；(2) `square` 有 512 个 tile，3.46 波，最后一波约一半 SM 空闲。33 关用 2-CTA MMA（256×256 tile，B 由两个 SM 分担）与 TMA multicast 处理 (1)。

## 8. 自测问题

1. `acc_empty` 若初始化为 1，而 128 个 epilogue 线程都 arrive，会发生什么？
2. producer 第一次等 `empty[s]` 为什么用 parity 1？MMA 第一次等 `acc_empty[a]` 呢？
3. S = 4、每 tile 6 个 K 块。某 CTA 的第 3 个 tile 的第 2 个 K 块，MMA 等哪个 `full` 槽、哪个 parity？
4. epilogue 在 TMA store 写完 C 之前就释放累加器，为什么安全？如果它在**最后一次 `tcgen05.ld` 之前**就释放呢？
5. 步骤 1 已经分工，为什么 epilogue 仍不能与主循环重叠？
6. `wide` 上若 G = 64（整列先走完），148 个并发 tile 需要多少 A、B 数据？与 G = 32 比较。

<details>
<summary>参考答案</summary>

1. 第一个线程到达就完成相位，MMA 开始覆盖累加器，其余线程可能读到下一个 tile 的部分和；剩下 127 次到达还会累积到后面的相位里，让后续的 `acc_empty` 相位提前完成，错误越来越多。
2. 新初始化的 barrier 处于相位 0，等 parity 1 就是等“前一个相位”，视为已完成，立即通过：所有 stage 一开始都是空的。`acc_empty` 同理：两个累加器一开始都空，MMA 对每个累加器的第一次等待都用 `phase ^ 1 = 1`。
3. 这是第 u = 2·6 + 1 = 13 次使用：slot 13 % 4 = 1，phase (13 / 4) & 1 = 1，等 `full[1]` 的 parity 1。
4. TMA store 读的是 staging box（shared memory），不读 TMEM；TMEM 在最后一次 `wait::ld` 返回时就不再被本线程访问。若在最后一次 ld 之前释放，MMA 可能在 ld 读取前覆盖这些列。
5. 每个 CTA 只有一个 tile，epilogue 只能在本 tile 的主循环之后做；shared memory 225 KB 使每 SM 只能放 1 个 CTA，下一个 tile（下一个 CTA）要等本 CTA 退出才能开始。
6. G = 64：64 行 × 2.3 列：A 全部 64 MB + B 约 4.6 MB ≈ 69 MB，与行优先对称，又变大了。G = 32 约 41 MB。工作集在 G ≈ √(148 × 2) ≈ 17 附近最小；实测还受其他因素影响（§7）。
</details>

## 9. 延伸阅读

- FlashAttention-3 论文（Shah et al., 2024）§3.1（producer–consumer asynchrony：warp specialization 与 pingpong 调度）；§3.2 的 intra-warpgroup overlap 在 34 关使用。
- PTX ISA 8.7：[elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync)、[mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)、[tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit)、[tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)、[cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)。
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`：`WarpCategory`（MMA、Sched、MainloopLoad、EpilogueLoad、Epilogue）与各 pipeline 的到达计数；
  - `third_party/cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp`：mainloop 的 load / mma 两个角色；
  - `third_party/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp`：`PipelineTmaUmmaAsync`（本关的 full/empty）与 `PipelineUmmaAsync`（本关的 acc_full/acc_empty），`consumer_release` 用 `tcgen05.commit`；
  - `third_party/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp`：persistent 静态调度与光栅化（raster order、swizzle）。
