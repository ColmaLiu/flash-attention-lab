# 40 · FA4 的流水结构：两个 Q tile ping-pong、P 写回 TMEM、correction warpgroup

> 部分 4 · FA4 — Blackwell 上的完整优化　｜　前置：[34_fa3](../34_fa3/README.md)、[32_ws_gemm](../32_ws_gemm/README.md)　｜　代码：[`students/40_fa4_pipeline/kernel.cu`](../../students/40_fa4_pipeline/kernel.cu)

## 1. 本关目标

34 关把 FA3 的思想搬到了 `tcgen05` 上：一个 TMA warp、一个 MMA warp、一个 softmax warpgroup，S 和 O 在 TMEM 里，P 经 shared memory。它最后停在 cuDNN 的约 0.55 倍，瓶颈是那一个 softmax warpgroup。

本关搭出 FlashAttention-4（FA4）在 B200 上的**流水结构**。接口不变：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]`，D = 64 或 128，causal 与非 causal，输出 O 和 FP32 的 LSE。

| | 34 关（FA3 思想） | 本关（FA4 结构） |
|---|---|---|
| 一个 CTA 算几行 query | 128（一个 Q tile） | **256（两个 Q tile）** |
| softmax warpgroup | 1 个 | **2 个，ping-pong**：一个做 softmax 时，Tensor Core 在算另一个 tile 的 GEMM |
| S 在 TMEM 的缓冲 | 2 个 S buffer（同一个 tile） | 每个 tile 1 个 S buffer |
| P 怎样进入 P·V | 写 shared memory（SS MMA） | **写回 TMEM，覆盖本 tile S 的后 64 列**，P·V 的 A 操作数直接从 TMEM 读（**TS MMA**） |
| O 的修正 | softmax 线程自己做，挡在 P·V 前面 | **独立的 correction warpgroup** |
| 寄存器 | 每个线程一样多 | **`setmaxnreg`**：softmax 192，correction 80，其余 48 |
| K/V 的 shared memory | K、V 各一个环 | K、V 共用**一个**环（K₀、V₀、K₁、V₁…） |

本关三步，每一步都支持 causal：

1. **两个 Q tile、两个 softmax warpgroup ping-pong**（P 仍经 shared memory，softmax 自己修正 O），加上 causal 的块数；
2. **P 别名写回 TMEM**，P·V 改成 TS 形式的 `tcgen05.mma`；
3. **correction warpgroup 与 `setmaxnreg`**。

做完后你应该能：

- 画出六种角色（2 × softmax、correction、MMA、TMA、空闲）的时间线和每个 mbarrier 的生产者、消费者与 parity；
- 说清为什么 `s_full` 一个 barrier 就同时保证了“S 已算好”“上一块的 P·V 已完成”“上一块的 P 已读完”；
- 写出 A 操作数在 TMEM 中的布局，以及 TS MMA 的地址怎样逐 K 片前进；
- 列出 512 列 TMEM、64K 个寄存器、227 KB shared memory 怎样在角色之间分配；
- 用 `./lab sass` 和插桩时间线解释每一步的收益，并说出下一关（41）要解决的瓶颈：MUFU。

## 2. 动机：34 关卡在哪里

34 关 §7.2 的结论：S 双缓冲之后 Tensor Core 仍有约 60% 的时间空闲，它在等**唯一的** softmax warpgroup。一个块里 softmax warpgroup 要依次做：读 S、求最大值、128×128 次 `exp2`、写 32 KB 的 P、把 O 读出来乘 α 再写回。

FA4（Zadouri 等，2026，代码在 `flash_attn/cute/flash_fwd_sm100.py`）给出的办法正好对应 34 关 §7.2 表中的前三行：

| 34 关剩下的问题 | FA4 的办法 | 本关步骤 |
|---|---|---|
| 一个 softmax warpgroup 在等 MMA 时没人填补，MMA 在等 softmax 时也闲着 | 两个 Q tile、两个 softmax warpgroup ping-pong（FA3 论文 §3.2 的 inter-warpgroup ping-pong，换成 `tcgen05`） | 1 |
| P 写 shared memory 又被 MMA 读回，还要 `fence.proxy.async`；P 占 64 KB shared memory | P 写回 TMEM，覆盖 S；TS MMA | 2 |
| O 的修正在 softmax 的关键路径上 | 独立的 correction warpgroup | 3 |
| 一个 CTA 里角色多了，寄存器不够分 | `setmaxnreg` 按角色重新分配 | 3 |
| MUFU 吞吐、每块都修正 O | 多项式 `exp2`、条件 rescale | 41 关 |

## 3. 数学

online softmax 的公式与 22、34 关完全相同（m 存原始分数，`scale_log2 = scale·log₂e`）：

```
m'  = max(m, max_c s_c)
α   = 2^(m·scale_log2 − m'·scale_log2)        块 0：m = −∞，α = 0
p_c = 2^(s_c·scale_log2 − m'·scale_log2)
ℓ   ← α·ℓ + Σ_c p_c
O   ← α·O + P·V_j                              最后 O/ℓ，LSE = m·scale + ln ℓ
```

两个 tile 互不相关：tile 0 是 query 行 `q0 .. q0+127`，tile 1 是 `q0+128 .. q0+255`，它们共用同一个 K_j、V_j，各有自己的 m、ℓ、S、P、O。本关新的数学只有一个：**一个块要多少时间**。

### 3.1 一个 tile 的“链”

对一个 tile，下面四件事必须按顺序发生（箭头是数据依赖）：

```
softmax_j（读 S_j，写 P_j）→ P_j·V_j → S_{j+1} = Q·K_{j+1}ᵀ → softmax_{j+1} → …
```

为什么 `S_{j+1}` 必须排在 `P_j·V_j` 之后？因为每个 tile 只有一块 S 的 TMEM（§5.1 的列预算），而 P_j 就写在这块 S 的后半；`S_{j+1}` 一写，P_j 就没了。于是一个 tile 每块的周期至少是

```
T_tile ≥ SM + PV + QK      （SM = softmax 一块所需时间）
```

D = 128 时 PV、QK 各 8 条 128×128×16 的 MMA，每条约 64 周期（B200 每 SM 每周期约 8192 FLOP），合计约 1024 周期；`exp2` 每 SM 每周期 16 个，一个 tile 一块 128×128 个，约 1024 周期。

### 3.2 两个 tile ping-pong

34 关只有一个 tile：Tensor Core 在 softmax 期间（除了提前算的下一个 S）基本闲着。两个 tile 时，tile 0 在做 softmax 的那段时间，Tensor Core 可以做 tile 1 的 PV 和 QK，反过来也一样：

```
Tensor Core │PV¹ QK¹│PV⁰ QK⁰│PV¹ QK¹│PV⁰ QK⁰│
softmax 0   │ SM⁰_j │       │SM⁰_j+1│       │
softmax 1   │       │ SM¹_j │       │SM¹_j+1│
```

理想情况下两条链错开半个周期，`T = max(2·(PV+QK), 2·SM, SM + PV + QK)`。D = 128 时三项都约 2048 周期：Tensor Core 和 MUFU 同时满载。实际 SM 比 1024 长得多（§7.2 实测约 1900–2200），所以 `T ≈ SM + PV + QK + 交接延迟`，链长决定一切。这就是本关每一步都在缩短的东西。

### 3.3 手算：每 SM 一块要多久

`perf` case（B·H = 64，N = 4096，D = 128）：每个 CTA 两个 tile、32 个 K/V 块，共 1024 个 CTA、148 个 SM，约 7 轮。用周期来算（不依赖频率）：

| | 每块 T | 每 SM 合计（7 轮 × 32 块 × T） |
|---|---|---|
| 理想（§3.2） | 2048 | ≈ 46 万周期 |
| 本关步骤 3（Nsight Compute 不锁频时测得 `sm__cycles_elapsed.avg` ≈ 83 万） | ≈ 3700（含每个 CTA 的序幕、收尾） | ≈ 83 万周期 |

满载时 SM 时钟约 1.64 GHz（同一次测量的 `sm__cycles_elapsed.avg.per_second`，低于标称最高频率），83 万周期 ≈ 0.50 ms（bench 测得约 0.55 ms，含 kernel 之间的间隙）。§7.2 用插桩把稳态的一块（≈ 3450 周期）拆开。

## 4. 硬件与指令

### 4.1 P 写回 TMEM：`tcgen05.st` 打包的 16 位数据

softmax 线程 r 拥有 S 的第 r 行（TMEM lane r，34 关 §4.1）。读完整行 S（128 个 FP32，128 列）之后，S 就不需要了，于是把 P 写回**同一行的后 64 列**：

```
TMEM 列：  S^t 的 128 列           c = 0 .. 127
           P^t 的 64 列            c = 64 .. 127（覆盖 S 的后半）
P^t 第 c' 列（c' = 0..63）的 32 位 = { 低 16 位: p[2c'], 高 16 位: p[2c'+1] }
```

32x32b 形状的 `tcgen05.st` 不关心数据类型，它只写 32 位的格子。头文件新增了 `uint32_t` 版本：

```cpp
uint32_t w[16];
for (int c = 0; c < 16; ++c) w[c] = pack2<T>(p[32h + 2c], p[32h + 2c + 1]);
lab::tmem_st_32x32b_x16(p_addr + 16 * h, w);      // h = 0..3
lab::tmem_wait_st();                               // 全部写完再交给 MMA
```

为什么覆盖 S 的后半是安全的：同一个线程在写 P 之前已经把整行 S 读进了寄存器（`tcgen05.ld` + `wait::ld`），而 S 的这一行只有这个线程读。为什么不用 `x32`：一次 32 个字再加 128 个 float 的 S 行，会让 168 个寄存器（步骤 2 的上限，§4.5）溢出。

### 4.2 TS 形式的 `tcgen05.mma`：A 从 TMEM 读

```
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
```

和 31 关的 SS 形式比，只是 A 由 shared memory 描述符换成了 **TMEM 地址**（头文件 `lab::mma_f16_ts`）。A 在 TMEM 中的布局要求（PTX ISA “tcgen05.mma”，`cta_group::1`，M = 128）：

| 要求 | 本关的 P |
|---|---|
| A 的第 i 行在 lane i（128 行占满 128 个 lane），地址的 lane 字段为 0 | P^t 第 r 行 = softmax 线程 r 写的 lane r |
| K 方向沿列排列，16 位数据**每列两个**，偶数下标在低 16 位 | `pack2(p[2c], p[2c+1])` |
| 必须是 K-major：idesc 的 A-major 位为 0 | 与 Q、P（34 关）相同 |
| 一条 MMA 的 K = 16，即 8 列 | 第 kk 个 K 片从 `p_tmem + 8·kk` 开始 |

B（V，MN-major）完全不变，还是 34 关 §4.3 的描述符。交接 P 的同步也变简单了：

| | 34 关 / 本关步骤 1（P 在 shared memory） | 步骤 2（P 在 TMEM） |
|---|---|---|
| 写 P | 普通 store + `fence.proxy.async` | `tcgen05.st` + `tcgen05.wait::st` |
| 交给 MMA | `tcgen05.fence::before_thread_sync` + `mbarrier.arrive` | 同左 |
| shared memory 中 P 的 64 KB | 需要 | **不需要**：K/V 环从 3 级增到 4 级（D = 128） |

### 4.3 同一个线程的 MMA 按发射顺序执行

步骤 2 的 MMA warp 对同一个 tile 依次发出：

```
PV^t_j  ：读 P^t（TMEM 列 128t+64 .. 128t+127）
QK^t_{j+1}：写 S^t（TMEM 列 128t .. 128t+127）   ← 覆盖了 PV^t_j 还要读的 P^t
```

两条 MMA 之间没有任何 barrier。这依赖于：**同一个线程发出的 `tcgen05.mma` 在 Tensor Core 中按发射顺序执行**，后一条写 TMEM 时前一条已经读完了它的操作数。FA4（`flash_fwd_sm100.py` 的 `mma()`：“this gemm is scheduled after the PV gemm, which guaranteed that Si has been read”）和 CUTLASS 的 SM100 FMHA（`sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`，P 与 S 共用 TMEM，PV 之后直接发下一个 QK）都这样做。

要诚实地说明：PTX ISA 8.7 的 “tcgen05 Memory Consistency Model” 只把“**相同 accumulator、相同形状**”的两条 `tcgen05.mma` 列为保证按序的 pipelined 对；PV（accumulator 是 O）和 QK（accumulator 是 S）不属于这一条。本关和上游一样依赖硬件的实际行为；本关全部 correctness case（含大规模随机数据）都通过，但这不是规范给出的保证。严格按规范的做法是在 QK 之前等 PV 的 commit 完成，代价是每块两次 Tensor Core 排空。

另一个**有规范保证**的事实本关大量使用：`tcgen05.commit` 让 mbarrier 跟踪**该线程此前发出的全部** `tcgen05` 异步操作。所以 QK^t_{j+1} 后面的 commit 一旦到达，PV^t_j 也一定完成了（§5.4）。

### 4.4 `setmaxnreg`：在 warpgroup 之间挪寄存器

每个 SM 有 64K 个 32 位寄存器。512 个线程的 CTA 启动时每个线程最多 128 个（`__launch_bounds__(512, 1)`）。但 softmax 线程要把一整行 S（128 个 float）放在寄存器里，再加上地址、m、ℓ、8 个部分和，34 关量出来要约 164 个；MMA、TMA 线程只要几十个。

```
setmaxnreg.inc.sync.aligned.u32 N;   // 把本 warp 每个线程的上限提高到 N，从 CTA 的池里取
setmaxnreg.dec.sync.aligned.u32 N;   // 降到 N，多出来的还回池里
```

- N 是 8 的倍数，24 ≤ N ≤ 256；
- 一个 warpgroup 的 4 个 warp 必须执行同一条 `setmaxnreg`，所以 MMA warp、TMA warp 和两个空闲 warp 凑成一个完整的 warpgroup 3（block 为 16 个 warp）；
- `dec` 立即生效；`inc` 会等到池里有足够的寄存器。本关先 `dec` 的角色（correction、MMA/TMA）不依赖任何 `inc`，不会死锁；
- ptxas 按 `setmaxnreg` 之后的 N 给那段代码分配寄存器：在 N = 48 的 MMA 代码里，如果编译器把两个 tile 的 16 个描述符都提到循环外，就会溢出到 local memory。答案中 MMA 的 tile 循环写了 `#pragma unroll 1`。`./lab sass` 的 `STACK` 一栏必须为 0。

头文件新增 `lab::setmaxnreg_inc<N>()` / `lab::setmaxnreg_dec<N>()`。

### 4.5 寄存器和线程预算

| 版本 | warp 数 | 角色 | 每线程寄存器上限 | 合计 |
|---|---|---|---|---|
| 步骤 1–2 | 10（320 线程） | 2 × softmax、MMA、TMA | ptxas 给出 168（SM 的 64K 寄存器分在 4 个子分区，每个 16K；warp i 在子分区 i % 4，10 个 warp 时有的子分区要放 3 个 warp：16384 / (3·32) ≈ 170，按 8 取整为 168） | — |
| 步骤 3 | 16（512 线程） | softmax ×2 | 192 | 2 × 128 × 192 = 49152 |
| | | correction | 80 | 128 × 80 = 10240 |
| | | MMA、TMA、2 个空闲 warp | 48 | 128 × 48 = 6144 |
| | | | | **65536 = 64K** |

步骤 3 如果不写 `setmaxnreg`，softmax 代码只能用 128 个寄存器，`./lab sass` 会显示上百字节的 `STACK`（溢出）。

### 4.6 让整个 warp 跑 MMA 调度

34 关的 MMA warp 只有 lane 0 在跑（`if (lane == 0)`），其余 31 个 lane 直接走到 kernel 末尾的 `__syncthreads` 等着。本关把 MMA warp 改成 CUTLASS/FA4 的写法：**32 个 lane 一起等 barrier，`elect.sync` 选出的一个 lane 发 MMA 和 commit**：

```cpp
const bool leader = lab::elect_one();
lab::mbar_wait(&bar.p_ready[t], j & 1);       // 整个 warp 等
lab::tcgen05_fence_after_thread_sync();
if (leader) { mma...; lab::tcgen05_commit(...); }
__syncwarp();
```

插桩实测（§7.2）：只有 lane 0 在跑时，MMA 线程在 `p_ready` 已经完成之后还要几百个周期才能发出下一条 MMA；改成整个 warp 之后降到几十个周期，整体快约 5–8%。这个现象是实测的；它的硬件原因（发散的 warp 如何被调度）我们没有进一步验证。TMA warp 提前好几级装数据，反应慢一点无所谓，保留 lane 0 的写法。

### 4.7 常见误用

| 误用 | 后果 |
|---|---|
| P 写到 S 的前 64 列，但还没读完整行 S | 读到自己刚写的 P：结果错 |
| TS MMA 的 A 地址每个 K 片前进 16 列 | 只用到一半的 P，另一半读到 S 的残余：O 错 |
| `pack2` 两个参数顺序反了 | 相邻两个 key 的 P 交换：`causal_position`、`position` 错 |
| `tcgen05.st` 之后没有 `wait::st` 就 arrive | MMA 偶尔读到旧 P |
| 一个 warpgroup 只有部分 warp 执行 `setmaxnreg` | 未定义行为（PTX ISA 的要求；本关没有测试具体表现） |
| `setmaxnreg` 的总和超过 512 | `inc` 永远等不到寄存器：挂起 |
| `p_ready` 的 count 在步骤 3 仍是 128 | correction 和 softmax 的 arrive 超出一个阶段：parity 错乱，挂起或结果错 |
| 把最终的 m、ℓ 也用 `stats_full` 交给 correction | 两个阶段可能连着完成，correction 等的 parity 永远不来（§5.6） |

## 5. 调度设计

### 5.1 角色、存储预算

```
grid  = (⌈N/256⌉, H, B)，一个 CTA = 一对 Q tile（256 行）× 一个 head × 一个 batch
block = 步骤 1–2：10 个 warp；步骤 3：16 个 warp
  warp 0–3   softmax WG 0：线程 r ↔ tile 0 的第 r 行 ↔ TMEM lane r
  warp 4–7   softmax WG 1：线程 r ↔ tile 1 的第 r 行 ↔ TMEM lane r（warp 4 访问 lane 0–31）
  warp 8–11  correction WG（步骤 3）：线程 r 负责两个 tile 的 O 第 r 行
  MMA warp   warp 8（步骤 1–2）或 12（步骤 3）
  TMA warp   MMA warp + 1
```

**TMEM**（512 列 × 128 lane × 32 位），两个 tile 各一份 S/P 和 O：

| 列 | D = 128 | D = 64 |
|---|---|---|
| S⁰（P⁰ 在其中 64–127） | 0–127 | 0–127 |
| S¹（P¹ 在其中 192–255） | 128–255 | 128–255 |
| O⁰ | 256–383 | 256–319 |
| O¹ | 384–511 | 320–383 |
| 合计 / 分配 | **512 / 512** | 384 / 512 |

D = 128 时正好用满。这就是为什么每个 tile 只能有一块 S：34 关的“S 双缓冲”在这里放不下，重叠改由**两个 tile 之间**来提供。

**shared memory**（全部是 128B swizzle 的 `[panel][行][64]`，34 关 §4.6）：

| | 步骤 1，D = 128 | 步骤 2–3，D = 128 | 步骤 1，D = 64 | 步骤 2–3，D = 64 |
|---|---|---|---|---|
| Q⁰、Q¹（结束时作 O 的暂存区） | 64 KB | 64 KB | 32 KB | 32 KB |
| P⁰、P¹ | 64 KB | — | 64 KB | — |
| K/V 环 | 3 × 32 KB | 4 × 32 KB | 6 × 16 KB | 8 × 16 KB |
| correction 的统计量（步骤 3） | — | 3 KB | — | 3 KB |
| 合计 | ≈ 224 KB | ≈ 195 KB | ≈ 192 KB | ≈ 163 KB |

### 5.2 一个 K/V 环

K_j 和 V_j 不再各有一个环，而是按 MMA warp 使用它们的顺序排成一个序列，**item u**：

```
u = 0    1    2    3    4    5   …
    K₀   V₀   K₁   V₁   K₂   V₂  …       item u 放在 stage u % S，第 ⌊u/S⌋ 次使用该 stage
```

TMA warp 对每个 item：等 `kv_empty[u%S]` 的 parity `((u/S)&1)^1`（30 关的标准环形 buffer），arm `kv_full`，装 K 或 V。MMA warp 用完 K_j（两个 tile 的 QK 都发完后）和 V_j（两个 tile 的 PV 都发完后）各 commit 一次 `kv_empty`。

### 5.3 MMA warp 的发射顺序

```
序幕： 等 K₀；对 t = 0, 1：等 Q^t，QK^t_0 → S^t，commit s_full[t]；commit kv_empty(K₀)
块 j： 等 V_j、K_{j+1}
       对 t = 0, 1：
         等 p_ready[t]（parity j&1）
         PV^t_j：O^t [+]= P^t_j · V_j            （j = 0 时不累加）
         t = 1：commit kv_empty(V_j)
         还有下一块：QK^t_{j+1} → S^t，commit s_full[t]
         否则：     commit o_full[t]
       还有下一块：commit kv_empty(K_{j+1})
```

Tensor Core 上的序列是 `PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1} PV⁰_{j+1} …`。tile 0 的 softmax 在算 S⁰_{j+1} 的时候，Tensor Core 在做 PV¹_j 和 QK¹_{j+1}——这就是 ping-pong。两个 tile 的相位差是在序幕里自然形成的：S⁰_0 比 S¹_0 早一条 QK 完成，之后每个 tile 的链各自循环（§3.2）。

### 5.4 `s_full` 一个 barrier 保证三件事

softmax WG t 在块 j 开始时只等 `s_full[t]` 的第 j 个阶段。它到达时：

1. S^t_j 已在 TMEM 中（commit 跟在 QK^t_j 后面）；
2. PV^t_{j−1} 已完成，O^t 含块 0..j−1：PV^t_{j−1} 在 QK^t_j **之前**由同一线程发出，commit 跟踪此前全部 MMA（§4.3）；
3. P^t_{j−1} 已被读完（同 2）。

所以和 34 关不同，softmax 写 P^t_j、修正 O^t 之前**不需要再等 `o_ready`**。

### 5.5 causal

一对 tile 的最后一行是 `q0 + 255`，所以块数是 `min(⌈N/128⌉, ⌈(q0 + 256)/128⌉)`，两个 tile 走**同样多**的块（MMA 调度不用分 tile）。代价：tile 0 的最后一块全部被 mask（行 `q0..q0+127` 看不到 key `q0+128..q0+255`）。全被 mask 的块里 `m' = m`（不是 −∞，因为块 0 总含 key 0），`α = 1`，P 全为 0，结果不变。FA4 同样如此；43 关的调度会处理这类浪费。

需要 mask 的块和每行可见的 key 数与 34 关相同，只是 `q0` 换成本 tile 的首行 `qt = q0 + 128t`。

### 5.6 correction warpgroup（步骤 3）

softmax 线程算出 m' 和 α 之后**立刻**把 α 交给 correction，然后才做 128 个 `exp2`：

```
softmax WG t（块 j）                         correction WG（对 j，对 t = 0, 1）
──────────────────                          ─────────────────────────────
wait s_full[t]; ld S; mask; max
α → st.alpha[t][r]; arrive stats_full[t] ─▶ wait stats_full[t]（parity j&1）
exp2, ℓ, 写 P（TMEM）                         j > 0：O^t 第 r 行 ×= α（ld/×/st，wait::st）
fence::before; arrive p_ready[t] ──┐         fence::before; arrive p_ready[t] ─┐
                                   └──────────────▶ MMA：wait p_ready[t]（count 256）
（全部块之后）m, ℓ → st.fin_*; arrive final_full[t] ─▶ wait final_full[t], o_full[t]
                                                   O/ℓ → smem → TMA store；LSE
```

correction 能安全地读 O^t：softmax 是在 `s_full[t]` 第 j 阶段之后才 arrive `stats_full` 的，此时 PV^t_{j−1} 已完成（§5.4）。修正与 `exp2` 并行，不再在 softmax 的关键路径上。

**`stats_full` 为什么不需要 empty barrier**：softmax 写 α^t_{j+1} 之前要等 `s_full[t]` 的第 j+1 阶段，它需要 MMA 等到 `p_ready[t]` 的第 j 阶段，而这需要每个 correction 线程在读完 α^t_j 之后 arrive。

**为什么最终的 m、ℓ 要用另一个 barrier `final_full`**：softmax 在最后一块 arrive `stats_full` 之后不再等任何 correction 相关的事，如果它马上再 arrive 一次 `stats_full`，这个 barrier 可能连着完成两个阶段，parity 回到原值，correction 等的那一阶段就“永远”看不到了。

**每个 mbarrier**（S = 环的级数）：

| barrier | count | 谁 arrive | 谁 wait | 块 j / item u 的 parity | 保护什么 |
|---|---|---|---|---|---|
| `q_full[t]` | 1 + tx | TMA | MMA | 0 | Q^t 已到达 |
| `kv_full[s]` | 1 + tx | TMA | MMA | `(u/S)&1` | item u 已到达 |
| `kv_empty[s]` | 1 | `tcgen05.commit`（K：两个 QK 后；V：两个 PV 后） | TMA | `((u/S)&1)^1` | stage 可以覆盖 |
| `s_full[t]` | 1 | commit（QK^t_j 后） | softmax WG t | `j&1` | §5.4 的三件事 |
| `p_ready[t]` | 128（步骤 1–2）/ 256（步骤 3） | softmax WG t 每个线程（+ correction 每个线程） | MMA | `j&1` | P^t_j 已写好、S^t_j 已读完、O^t 已修正 |
| `o_full[t]` | 1 | commit（最后一个 PV^t 后） | epilogue | 0 | O^t 最终值；读 Q^t 的 MMA 全部完成（Q^t 可作暂存区） |
| `stats_full[t]` | 128 | softmax WG t | correction | `j&1` | α^t_j 在 `st.alpha[t]` |
| `final_full[t]` | 128 | softmax WG t | correction | 0 | 最终 m、ℓ |

parity 不会追尾的理由和 34 关相同：每个 barrier 的第 j+1 阶段都要求等待者先看到第 j 阶段（例如 `p_ready` 的第 j+1 阶段需要 softmax 先拿到 `s_full` 的第 j+1 阶段，而它只能在 MMA 等到 `p_ready` 第 j 阶段之后出现）。`final_full` 和 `o_full` 只有一个阶段。

**racecheck 看不见 TMA 和 TMEM 的访问**：上面的表就是本关同步正确性的全部论证，请逐行核对。

### 5.7 全部角色的时间线（D = 128，稳态，示意，不按比例）

```
TMA          │K₂ V₂ ……（提前 2 块）…………………………………………………………………………
MMA/TC       │PV⁰₀ QK⁰₁│PV¹₀ QK¹₁│  ·  │PV⁰₁ QK⁰₂│PV¹₁ QK¹₂│  ·  │
softmax 0    │         │ld max exp exp exp P│         │ld max exp exp exp P│
softmax 1    │ld max exp exp exp P│         │ld max exp exp exp P│
correction   │   ×α¹   │   ×α⁰   │         │   ×α¹   │   ×α⁰   │
              ↑p_ready⁰          ↑p_ready¹
```

每个 tile 的周期 = 自己的 softmax + 自己的 PV、QK + 交接延迟；两个 tile 错开约半个周期（§3.2）。

## 6. 分步练习

骨架中 host 端（tensor map、launch、dispatch）、barrier 初始化、TMEM 分配、34 关学过的全部辅助函数（`mma_qk`、`mma_pv_ss`、`load_s_row`、mask、`row_max`/`exp_and_sum`、`write_p_smem`、`rescale_o_row`、`store_o_and_lse`）都已写好。`p.step` 选择三个配置之一（`Config<D, 1|2|3>`）；三步是**同一个 kernel 逐步补全**，后面的步骤依赖前面的 TODO。

### 步骤 1 · TODO(1a)–(1e)：两个 tile ping-pong（含 causal）

- 1a：TMA warp 装两个 Q tile，以及 K/V 环（§5.2）；
- 1b：MMA warp 的序幕（§5.3）；
- 1c：MMA warp 每块对每个 tile 的 PV、QK 与各个 commit（§5.3、§4.3）；
- 1d：softmax WG 选自己的 tile、TMEM 地址，等 `s_full`（§5.1、§5.4）；
- 1e：causal 的块数（§5.5）。

```bash
./lab check 40 --step 1 --case two_tiles
./lab check 40 --step 1
./lab bench 40 --step 1 --case perf
```

### 步骤 2 · TODO(2a)–(2b)：P 写回 TMEM

- 2a：`write_p_tmem`（§4.1）；2b：`mma_pv_ts`（§4.2）。kernel 中调用它们的分支（`C::kPInTmem`）已经写好。

```bash
./lab check 40 --step 2 --case position
./lab check 40 --step 2
./lab sass 40           # UTCHMMA 的 A 操作数变成 tmem[...]
```

### 步骤 3 · TODO(3a)–(3d)：correction warpgroup 与 setmaxnreg

- 3a：`Config` 中的三个寄存器数，以及每个角色开头的 `setmaxnreg`（§4.4、§4.5）；
- 3b：softmax 交出 α 和最终的 m、ℓ；3c：correction 的主循环；3d：correction 的 epilogue（§5.6）。

```bash
./lab check 40 --step 3
./lab sass 40           # 步骤 3 的 kernel：REG 128、STACK 0
./lab bench 40 --case perf
./lab bench 40 --case perf_causal
./lab bench 40 --case perf_d64
./lab bench 40 --case perf_8k
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `TIMEOUT`（步骤 1） | 环的 parity；MMA 和 TMA 对 item 的编号不一致（K 是偶数、V 是奇数）；`p_ready` 的 parity；忘了 TODO(1b)/(1c) 的 “continued” 部分——序幕后释放 K₀、每块末释放 K_{j+1}（这两处没有 `LAB_TODO`，漏掉时 TMA warp 永远等不到空 stage） |
| 只有 tile 1 的行错（N > 128 的 case） | Q¹ 的坐标（`q0 + 128`）；S¹/O¹ 的 TMEM 列；softmax WG 1 的 `qt` |
| `n1`、`one_block` 对，`two_tiles` 错 | 同上；或 tile 1 用了 tile 0 的 P 或 S |
| `ring`、`position` 错，单块 case 对 | K/V 的 item 下标或 stage |
| causal 只有第二对以后的 tile 错 | 块数用了 `q0 + 128` 而不是 `q0 + 256` |
| 步骤 2 O 全错、LSE 对 | P 的打包顺序或列偏移；TS MMA 的 A 地址步长（8 列） |
| 步骤 3 `TIMEOUT` | `setmaxnreg` 总和 > 512；或某个 warpgroup 没有全部执行；`stats_full` 的 parity |
| 步骤 3 `peaked*` 错、`uniform` 对 | correction 用错 tile 的 α，或在 j = 0 也乘了 α（此时 α = 0） |
| 步骤 3 正确但比步骤 2 慢很多 | `./lab sass` 看 STACK：寄存器数没设，softmax 溢出 |

## 7. 性能

### 7.1 相对库基线

B200 实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时，越大越好；每项 3 次运行的范围）：

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| 34 关步骤 5（对照） | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× | — |
| 步骤 1（ping-pong，P 经 smem） | ≈ 0.77–0.78× | ≈ 0.88–0.89× | ≈ 0.80× | ≈ 0.75–0.76× |
| 步骤 2（P 在 TMEM，TS MMA） | ≈ 0.80× | ≈ 0.89–0.90× | ≈ 0.73–0.75× | ≈ 0.77–0.78× |
| 步骤 3（correction WG + setmaxnreg） | ≈ 0.84× | ≈ 0.93–0.94× | ≈ 0.79–0.81× | ≈ 0.82× |
| FA4（`flash_attn.cute`） | ≈ 1.15× | ≈ 1.33× | ≈ 1.13× | ≈ 1.10× |

你的步骤 3 在四个 perf case 上都达到 cuDNN 的约 0.72 倍以上，就说明本关的机制成立；cuDNN 本身在多次运行之间有几个百分点的波动，请看范围。达到 cuDNN 0.9 倍是 43 关的目标。

- **ping-pong**（34 关 → 步骤 1）：约 1.35–1.45 倍，这是本关最大的一步。`perf` 上 Tensor Core 忙碌的比例（`sm__pipe_tensor_cycles_active`）从 34 关的约 38% 升到约 53%。
- **P 写回 TMEM**（步骤 1 → 2）和 **correction warpgroup**（步骤 2 → 3）：各只有几个百分点（步骤 3 的 `perf` 上 Tensor Core 约 57%）。原因见下一节：它们缩短的是 softmax 链上的几百个周期，而链上最长的一段——128 个 `exp2`——没有变。
- `perf_d64` 上步骤 2 反而比步骤 1 慢约 7%，步骤 3 又回到步骤 1 的水平。D = 64 时 GEMM 只有 D = 128 的一半，Tensor Core 只忙约 30%，瓶颈完全在 softmax 一侧（MUFU 约 57–61%）。我们没有查明步骤 2 变慢的具体原因（K/V 环的级数 4、6、8 结果相同），这是一个留给你用 `./lab profile` 研究的现象。
- 步骤 3 的另一个收益是寄存器：correction 和 MMA/TMA 让出来的寄存器给了 softmax，以后（41 关）softmax 还要再加多项式 `exp2` 的代码。

### 7.2 瓶颈在哪：插桩时间线

在一个 CTA 里用 `clock64()` 记下每个事件的时间（维护者实验，代码不在仓库中；`perf` case，稳态块，单位是 SM 周期）：

| 段 | 步骤 3，MMA 只用 lane 0 | 步骤 3（答案） |
|---|---|---|
| 一块的周期 T | ≈ 3700 | ≈ 3450 |
| softmax WG：拿到 S → arrive `p_ready` | ≈ 1900 | ≈ 1900–2200 |
| 　其中 128 个 `exp2`（含求和） | ≈ 1450 | ≈ 1450–1700 |
| 最后一个线程 arrive → MMA 发出 PV | ≈ 250–550 | ≈ 40 |
| PV + QK（一个 tile）在 Tensor Core 上 | ≈ 1100–1300 | ≈ 1100 |

结论：

1. 一个 tile 的链 ≈ softmax + 交接 + PV + QK，正是 §3.1 的模型；
2. softmax 的大头是 `exp2`：一个 warp 128 条 `MUFU.EX2`，按每 SM 每周期 16 个的吞吐至少 1024 周期，实测约 1450——**MUFU 现在是瓶颈**；
3. Tensor Core 约 57% 时间在工作（`ncu` 的 `sm__pipe_tensor_cycles_active`），MUFU（`sm__inst_executed_pipe_xu`）也只有约 57%：两者都没满，是链太长。

缩短链的下一步是让 `exp2` 更快：41 关把一部分 `exp2` 换成 FMA 上的多项式（FMA 的吞吐是 MUFU 的 8 倍），并只在最大值明显变化时才修正 O。

```bash
./lab profile 40 --case perf
```

看 `sm__pipe_tensor_cycles_active`、`sm__inst_executed_pipe_xu` 和 Warp State 中 softmax 等 `s_full` 的 `long_sb` 停顿（mbarrier 等待）。Nsight Compute 锁定时钟，只用它比较指标。

## 8. 自测问题

1. D = 64 时 TMEM 用了多少列？如果想在 D = 64 时给每个 tile 两块 S（像 34 关那样），够不够？
2. softmax WG 1 的 warp 6 读 S¹ 的第 32–63 列，`tcgen05.ld` 的 taddr（相对 TMEM base）是多少？写 P¹ 的第 16–31 个字（`tmem_st_32x32b_x16`）呢？
3. 步骤 2 中 PV^t_j 的第 kk = 5 个 MMA，A 的 TMEM 地址（相对 base，t = 1）是多少？它对应哪些 key？
4. 为什么 softmax 不需要像 34 关那样等 `o_ready` 就可以写 P^t_j、修正 O^t？如果 MMA 把 QK^t_{j+1} 放在 PV^t_j **之前**发，会出什么问题？
5. 步骤 3 的 `p_ready` count 为什么是 256？如果 correction 只在 j > 0 时 arrive 会怎样？
6. 把 `setmaxnreg` 改成 softmax 200、correction 64、其余 48，总和是多少？会发生什么？

<details>
<summary>参考答案</summary>

1. S⁰、S¹ 256 列 + O⁰、O¹ 各 64 列 = 384 列（分配 512）。每个 tile 再加一块 S 要再 256 列，共 640 > 512，不够。
2. warp 6 在 warpgroup 中是第 2 个 warp，lane 段从 64 开始；S¹ 从第 128 列开始：`(64 << 16) | (128 + 32)`。P¹ 从 S¹ 的第 64 列开始，第 16 个字：`(64 << 16) | (128 + 64 + 16)`。
3. `128 + 64 + 8·5 = 232`（lane 字段 0），对应 key 80–95。
4. §5.4：softmax 等到的 `s_full[t]` 第 j 阶段由 QK^t_j 之后的 commit 完成，而 PV^t_{j−1} 在 QK^t_j 之前由同一线程发出，commit 跟踪它。反过来先发 QK^t_{j+1}，它会覆盖 PV^t_j 还没读的 P^t_j（P 在 S 的列里），结果错。
5. 每个 softmax 线程（128）和每个 correction 线程（128）每块各 arrive 一次，MMA 要同时确认“P 已写好”和“O 已修正”。如果 correction 在 j = 0 不 arrive，第 0 阶段永远凑不够 256：挂起。
6. 2·200 + 64 + 48 = 512，合法。correction 降到 64 可能溢出（看 `./lab sass`），softmax 多 8 个寄存器对本关没有好处。超过 512 时 `inc` 永远等不到寄存器而挂起。
</details>

## 9. 延伸阅读

- Zadouri 等，[FlashAttention-4](https://arxiv.org/abs/2603.05451v1)（2026）；上游实现 `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`：
  - `__init__`：warp 角色（`softmax0_warp_ids` … `empty_warp_ids`）、`tmem_s_offset` / `tmem_o_offset` / `tmem_s_to_p_offset = n_block_size // 2`（本关 §5.1 的表）、`num_regs_softmax` 等寄存器预算；
  - `mma()`：两个 Q stage 的发射顺序（本关 §5.3）与 “PV 之后才写 S” 的注释；
  - `softmax_step()`：读 S、最大值、把 `acc_scale` 交给 correction、写 P 回 TMEM；
  - `correction_loop()`：`sm_stats_barrier`、rescale、epilogue（本关 §5.6，FA4 还有条件 rescale：41 关）。
- Jay Shah 等，*FlashAttention-3*（2024）§3.2：inter-warpgroup ping-pong 调度。
- `third_party/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`：`TmemAllocation`（S0、S1、P0、P1、O0、O1）、`mma()` 中 QK/PV 的交替、`softmax()` 与 `correction()`；`kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`：`NumRegsSoftmax` 等。
- PTX ISA 8.7：[tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)（A 在 TMEM 的形式）、[tcgen05.st](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)、[tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)（pipelined 指令对）、[setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)、[elect.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync)。
- 后续：41 关（softmax 吞吐：多项式 `exp2`、条件 rescale）。
