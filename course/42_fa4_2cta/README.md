# 42 · 2-CTA attention：一对 CTA 共用 pair MMA，K/V 各装一半

> 部分 4 · FA4 — Blackwell 上的完整优化　｜　前置：[40_fa4_pipeline](../40_fa4_pipeline/README.md)、[41_fa4_softmax](../41_fa4_softmax/README.md)、[33_2cta_gemm](../33_2cta_gemm/README.md)　｜　代码：[`students/42_fa4_2cta/kernel.cu`](../../students/42_fa4_2cta/kernel.cu)

## 1. 本关目标

33 关在 GEMM 上学了 2-CTA MMA：一个 cluster 里的两个 CTA 组成一对（pair），leader 的一个线程发 `tcgen05.mma.cta_group::2`，M = 256，A 和 B 各从两个 CTA 的 shared memory 读一半，结果写进两个 CTA 的 TMEM。40 关搭好了 FA4 的流水：一个 CTA 两个 Q tile ping-pong、P 写回 TMEM、correction warpgroup；41 关加快了 softmax：软件 `exp2` 与 `MUFU.EX2` 混合、条件 rescale。

本关从 41 关的答案出发，加入 2-CTA，正是 FA4 在 `use_2cta_instrs=True` 时做的事：

| | 40、41 关（1-CTA） | 本关（2-CTA） |
|---|---|---|
| 一个 cluster 算几行 query | 256（cluster = 1 个 CTA） | **512**（2 个 CTA，每个 256） |
| QKᵀ | 每个 CTA 自己的 M = 128 MMA | **pair MMA，M = 256**：leader 的 tile t 和 peer 的 tile t 一起算 |
| P·V | TS MMA，M = 128 | **TS pair MMA**，A = 两个 CTA 各自 TMEM 里的 P |
| 每个 CTA 装的 K_j | 128 个 key × D | **64 个 key × D**（另一半在 peer） |
| 每个 CTA 装的 V_j | 128 个 key × D | **128 个 key × D/2 列**（另一半列在 peer） |
| MMA warp | 每个 CTA 一个 | **只有 leader 有**；它替两个 CTA 发全部 MMA |
| `p_ready` | 本 CTA 256 次 arrive | **leader 的 barrier 收两个 CTA 的 512 次 arrive** |
| softmax、correction、epilogue | 每个 CTA 自己（41 关的算术） | 不变：每个 CTA 处理自己 TMEM 里的行；D = 64 的多项式比例重新调过（§7.3） |

接口不变：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]`，D = 64 或 128，causal 与非 causal，O 与 FP32 的 LSE。

三步：

1. **pair QKᵀ**：cluster 启动、pair TMEM、K 对半装载、leader 侧的 full barrier、multicast commit、排空与 cluster 收尾；只检查第一个 K/V 块的 S（probe）；
2. **完整的 2-CTA 流水**：V 按列对半装载、TS pair MMA、跨 CTA 的 `p_ready`、causal；
3. **跨 CTA 交接的代价**：把 `p_ready` 的远程 arrive 从 `.release.cluster` 换成 `.release.cta`。

做完后你应该能：

- 画出 pair 的数据布局：哪一行在哪个 CTA 的哪个 TMEM lane，K、V 的哪一半在哪个 CTA；
- 列出每个 mbarrier 用的是哪个 CTA 的副本、谁 arrive、谁 wait；
- 推导 2-CTA 省下的 shared memory、L2→SM 流量和 shared memory 读带宽；
- 用实测解释：softmax 受 MUFU 限制时（40 关的 softmax）2-CTA 反而慢约 2–12%；换上 41 关的 softmax 后，非 causal 的主循环与 1-CTA 打平，整体仍差约 1–4%，D = 64 与 causal 差约 5–7%；以及按配置选择 1-CTA / 2-CTA 的依据。

## 2. 动机：41 关之后还剩什么

40 关 §7.2 的结论：一个 tile 的链是 softmax（≈ 2000 周期，大头是 128 个 `exp2`）+ 交接 + PV + QK（≈ 1100 周期）。41 关把一部分 `exp2` 移到 FMA 流水线、跳过大部分 O 的修正，softmax 段变短，链上 MMA 一侧的比重随之变大。

FA4 论文和代码里还有两类优化：

| 方向 | 做法 | 关卡 |
|---|---|---|
| 缩短 softmax | 多项式 `exp2` 分担 MUFU、条件 rescale | 41（已完成，本关沿用） |
| 减轻 MMA 一侧的负担 | **2-CTA MMA**：B 操作数（K、V）在两个 CTA 之间对半分 | **本关** |
| 调度 | persistent、LPT、causal 的块分类 | 43 |

2-CTA 减轻的是三样东西（§3 推导）：每个 CTA 的 K/V shared memory、每个输出行的 L2→SM 流量、MMA 每周期从 shared memory 读的字节数。它**不**减少 softmax 的工作量。所以本关的一个重要问题是：softmax 要快到什么程度，2-CTA 才能带来收益？§7 用实测回答（先用 40 关的 softmax 看它为什么输，再换上 41 关的 softmax）。

## 3. 数学

attention 的数学与 40 关完全相同（online softmax、P 写回 TMEM、O 的修正）。本关只改变“谁算哪几行、谁装哪一半 K/V”。

### 3.1 pair 的数据布局

cluster c（`%clusterid.x`）负责第 q0 = 512c 起的 512 行。CTA r（`%cluster_ctarank`，0 = leader，1 = peer）的 tile t 是

```
rows(r, t) = q0 + 256 t + 128 r  ..  + 127
```

于是 pair tile t（两个 CTA 的 tile t 合起来）正好是连续的 256 行 `q0 + 256t .. q0 + 256t + 255`：

```
             leader (rank 0)                    peer (rank 1)
             ───────────────                    ─────────────
tile 0       rows q0      .. q0+127             rows q0+128 .. q0+255     ← pair tile 0（MMA 行 0..255）
tile 1       rows q0+256  .. q0+383             rows q0+384 .. q0+511     ← pair tile 1
TMEM         S⁰ S¹ O⁰ O¹ 的 lane r = 本 CTA tile 的第 r 行（两个 CTA 列地址相同）

K_j (128 keys × D)      keys 0..63  × D           keys 64..127 × D          ← B of QKᵀ, N = keys
V_j (128 keys × D)      keys 0..127 × cols 0..D/2-1   keys 0..127 × cols D/2..D-1   ← B of PV, N = D
```

`tcgen05.mma.cta_group::2` 的规则（33 关 §4.4）：A 的第 0–127 行来自 leader、128–255 行来自 peer；**B 按 N 分**：N 的前一半来自 leader，后一半来自 peer；D 的第 0–127 行写 leader 的 TMEM，128–255 行写 peer 的 TMEM。

- QKᵀ：A = Q（M 是 query），B = K（N 是 key）。所以 K 按 **key** 对半分。
- PV：A = P（M 是 query），B = V（N 是 D 的列，K 是 key）。所以 V 按 **列** 对半分：每个 CTA 都要全部 128 个 key，但只要 D/2 列。

小例子（D = 128，N = 4096，第 0 个 cluster）：leader 装 Q 行 0–127、256–383，K_j 的 key 128j..128j+63，V_j 的第 0–63 列；peer 装 Q 行 128–255、384–511，K_j 的 key 128j+64..128j+127，V_j 的第 64–127 列。pair MMA 之后 leader 的 S⁰ 是行 0–127 对 128 个 key 的分数，peer 的 S⁰ 是行 128–255 的。

### 3.2 省下了什么

每块 K/V（128 个 key）、每个 CTA 的字节数（16 位元素）：

| | 40 关 | 本关 | 比例 |
|---|---|---|---|
| K_j + V_j 进 shared memory | 2 · 128 · D · 2 B（D = 128：64 KB） | 一半（32 KB） | 0.5 |
| 同样的环字节数能放几块 | 128 KB / 64 KB = 2 块 | 128 KB / 32 KB = 4 块 | 2 |
| L2→SM 的 K/V 流量（每个输出行） | 64 KB / 256 行 = 256 B/行 | 32 KB / 256 行 = 128 B/行 | 0.5 |

整个 `perf` case（B·H = 64，N = 4096，D = 128）：40 关 1024 个 CTA × 32 块 × 64 KB ≈ 2.15 GB 的 K/V 加 67 MB 的 Q；本关 ≈ 1.07 GB + 67 MB。

MMA 从 shared memory 读操作数的带宽（按每 SM 每周期 128 B、BF16 每 SM 每周期约 8192 FLOP 估算）：

| MMA（每个 SM 的份） | 一条 K = 16 的 MMA 读多少 | 用多少周期 | 每周期读 |
|---|---|---|---|
| 40 关 QKᵀ，M = 128，N = 128（A、B 都在 smem） | A 4 KB + B 4 KB | 128·128·16·2 / 8192 = 64 | 128 B，**正好占满** |
| 本关 QKᵀ（pair，每个 SM 一半 B） | A 4 KB + B 2 KB | 64 | 96 B（0.75） |
| 40 关 PV，N = D = 128（A 在 TMEM） | B 4 KB | 64 | 64 B |
| 本关 PV（pair） | B 2 KB | 64 | 32 B（0.5） |

所以 1-CTA 的 QKᵀ 在 shared memory 带宽上没有余量；2-CTA 把它降到 75%。这只在 MMA 一侧成为瓶颈时有用（§7）。

### 3.3 代价

- **两个 SM 被绑在一起**：pair tile t 的 PV 必须等**两个** CTA 的 softmax 都交出 P，S 也同时写进两个 CTA。每块的链上多了两次跨 SM 的交接（multicast commit 到 peer、peer 的 arrive 到 leader）。
- **causal 的对角浪费翻倍**：一个 pair tile 是 256 行，一个 cluster 的两个 tile 一起前进（ping-pong），cluster 的时间由走得最远的 tile 1 决定：cluster c 走 `4c + 4` 块。40 关同样的 512 行由两个 CTA 分别走 `4c + 2` 和 `4c + 4` 块。N = 4096 时平均每 512 行的 SM 时间是 2·(4c+4) ≈ 36 块 vs (8c+6) ≈ 34 块，多约 6%。tile 0 少走两块（步骤 2 的 `tile_blocks`）省不回来：tile 1 照样要走完，tile 0 早停只是让 tile 1 最后两块没有 ping-pong 伙伴。
- **cluster 启动**：一对 CTA 要同时有两个空闲 SM；TMEM 的 `cta_group::2` 分配是一段跨 CTA 的软件协议（33 关 NOTES）。实测 setup 从约 0.45 µs 变成约 1 µs，CTA 末尾的收尾从约 1.2 µs 变成约 1.9 µs（共 32 块、约 75 µs 的 CTA 寿命里很小）。

## 4. 硬件与指令

### 4.1 本关新增的头文件函数

| 函数（头文件） | PTX | 用途 |
|---|---|---|
| `lab::tma_load_4d_cg2`（`cluster.cuh`） | `cp.async.bulk.tensor.4d.cta_group::2...` | 33 关 `tma_load_2d_cg2` 的 4-D 版：box 写进本 CTA，complete_tx 落在 leader 的 barrier |
| `lab::mma_f16_ts_cg2`（`cluster.cuh`） | `tcgen05.mma.cta_group::2.kind::f16 [d], [a_tmem], b_desc, idesc, p` | TS 形式的 pair MMA：A 来自两个 CTA 各自 TMEM 的同一地址 |
| `lab::mbar_arrive_remote`（`cluster.cuh`） | `mbarrier.arrive.shared::cluster.b64`（默认 `.release.cta`） | 步骤 3 的廉价远程 arrive（§4.4） |
| `lab::make_smem_desc_mnmajor_sw64`（`tcgen05.cuh`） | 描述符，layout = 4（SWIZZLE_64B） | D = 64 时 V 的一半只有 32 列 = 64 B 一行 |

其余 pair 原语（`tmem_alloc_cg2`、`mma_f16_ss_cg2`、`tcgen05_commit_cg2_multicast`、`map_shared_rank`、`mbar_wait_cluster`、`mbar_arrive_cluster`、`cluster_sync`、`launch_cluster`）在 33 关写过。一个 kernel 里所有 tcgen05 指令必须用同一个 `cta_group`，所以本关连 commit 都用 `cg2` 的 multicast 版。

### 4.2 pair QKᵀ

```cpp
idesc = make_idesc_f16(256 /*M of the pair*/, 128 /*N = keys*/, bf16);
da = make_smem_desc_kmajor_sw128(sq);   // leader 的 Q^t；peer 的在同一偏移
db = make_smem_desc_kmajor_sw128(sk);   // leader 的 K 一半：[D/64 panel][64 key][64]
for kk: mma_f16_ss_cg2(s_tmem, da + a_off(kk), db + b_off(kk), idesc, kk > 0);
```

与 40 关 `mma_qk` 唯一的不同是 K 的 panel 只有 64 行（`kHalfKeys`），所以 `b_off = (panel · 64 · 64 + col) · 2 / 16`。描述符只描述 leader 的 buffer，硬件在 peer 的同一偏移读另一半——所以**两个 CTA 的 shared memory 布局必须完全相同**。

### 4.3 TS pair PV 与 64B swizzle

```cpp
idesc = make_idesc_f16(256, D /*N = 全部 D 列*/, bf16, true, false, /*b_mn_major=*/true);
db = D == 128 ? make_smem_desc_mnmajor_sw128(sv, ...)   // 一行 64 列 = 128 B
              : make_smem_desc_mnmajor_sw64(sv, ...);   // 一行 32 列 = 64 B
for kk: mma_f16_ts_cg2(o_tmem, p_tmem + 8 kk, db + kk · 16 · rowbytes / 16, idesc, acc || kk > 0);
```

- idesc 的 N 是**整个** D：硬件把 N 对半分给两个 CTA。每个 CTA 的 V 一半是 [128 key][D/2]，MN-major。
- D = 128 时一半 = 64 列 = 128 字节，正好一个 128B swizzle 行（40 关的布局）。D = 64 时一半只有 32 列 = 64 字节，128B swizzle 用不了，要用 **64B swizzle**：TMA 的 box 宽 32 个元素、`CU_TENSOR_MAP_SWIZZLE_64B`，描述符 layout 字段 = 4，8 行一个 512 字节的 atom（SBO = 512），K 方向前进 16 行 = 16 · 64 字节。
- A 的地址规则与 40 关相同（每 16 个 key 8 列）；leader 和 peer 的 P 都在各自 TMEM 的同一列。
- 每个 CTA 只有一个 MN panel，所以 LBO 用不到。

### 4.4 跨 CTA 的 arrive：`.release.cluster` 与 `.release.cta`

peer 的 softmax 和 correction 线程写完自己 TMEM 里的 P、修正完 O 后，要告诉 **leader** 的 MMA 线程。它们在 leader 的 `p_ready` 上远程 arrive（地址来自 `mapa`）：

```cpp
tcgen05_fence_before_thread_sync();       // 本线程的 tcgen05.ld/st 排在 arrive 之前
arrive(map_shared_rank(&bar.p_ready[t], 0));
```

33 关用的是 `mbar_arrive_cluster`：`mbarrier.arrive.release.cluster`。编译成 SASS 后是（自己编译一个只含这条指令的 kernel，`cuobjdump -sass`）：

```
.release.cluster:  MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; SYNCS.ARRIVE.TRANS64.RED.A1T0 [addr]
.release.cta:      SYNCS.ARRIVE.TRANS64.RED.A1T0 [addr]
```

`.release.cluster` 让这个线程之前的**所有**内存写在 cluster 范围可见，硬件用一次 GPU 范围的 `MEMBAR` 实现，要等本线程所有在途的访存完成。每块、每个 tile、512 个线程各做一次。§7.2 的插桩：步骤 2 里 leader 的 MMA 线程在最后一个线程 arrive 之后约 650–900 ns 才看到 `p_ready` 完成；步骤 3 约 100 ns。

为什么本关可以用 `.release.cta`：

- 交接的数据是 **TMEM**（P 和修正后的 O），不是 shared/global memory。它们由 `tcgen05.st` 写入，`tcgen05.wait::st` 等到写完，`tcgen05.fence::before_thread_sync` 把它们排在 arrive 之前；leader 一侧 wait 之后做 `tcgen05.fence::after_thread_sync`。
- arrive 本身照样被计数；`p_ready` 的相位要等 512 次 arrive 全部到达才完成。
- CUTLASS（`ClusterBarrier::arrive(cta_id, pred)`，`include/cutlass/arch/barrier.h`）和 FA4（pipeline 的 `consumer_release` 远程 arrive 到 leader）都用这个默认形式。CuTe DSL 的 `cute/arch/mbar.py` 里明确写着：远程 arrive 在形式上应该用 cluster 作用域，cluster 作用域“实测更慢”，所以默认 CTA，经 DSMEM 交接数据的协议才传 `scope=CLUSTER`。

需要说清楚的是：PTX 内存模型要求 release/acquire 的作用域同时包含两个线程，才算严格的 synchronizes-with。`.release.cta` 跨 CTA 使用，靠的是“数据在 TMEM、已由 wait::st 完成”这一点，与 CUTLASS/FA4 的做法一致。**如果你交接的是 shared memory 或 global memory 里的数据，必须用 `.release.cluster`**（例如 33 关之前讨论过的 DSMEM 写）。

### 4.5 常见误用

| 误用 | 后果 |
|---|---|
| peer 也发 MMA | 每块算两遍，S/O 被写两次，结果错或挂起 |
| peer 也 `arrive_expect_tx` 同一个 leader barrier | tx 字节数翻倍、arrive 多一次：相位提前完成或永远完不成 |
| commit 不 multicast（`cta_group::1` 的 `tcgen05_commit`） | 编译期就不允许混用 cta_group；即使改成 mask 只含 leader，peer 的 softmax 永远等不到 `s_full` |
| K 按 D 对半、V 按 key 对半 | 方向反了：B 总是按 **N** 对半，QKᵀ 的 N 是 key，PV 的 N 是 D 的列 |
| D = 64 的 V 一半仍用 128B swizzle | box 宽 32 元素只有 64 B，128B swizzle 的地址模式不对，O 错 |
| `__syncthreads` 代替 cluster barrier | peer 的 TMA 或 leader 的 commit 可能在对方 `mbarrier.init` 之前到达 |
| 一个 CTA 提前退出 | leader 的 MMA 还在读它的 smem、写它的 TMEM；或 multicast 落到已退出的 CTA |

## 5. 调度设计

### 5.1 角色与存储预算

角色与 40 关步骤 3 相同（16 个 warp，`setmaxnreg` 192/80/48），只有 MMA warp 只在 leader 里工作：

| warp | leader | peer |
|---|---|---|
| 0–3、4–7 | softmax tile 0、tile 1 | 同左（自己的行） |
| 8–11 | correction + epilogue | 同左 |
| 12 | **MMA：两个 CTA 的全部 MMA** | 空闲（直接到收尾的 cluster barrier） |
| 13 | TMA：自己的 Q、K/V 一半 | 同左 |
| 14–15 | 空闲 | 空闲 |

每个 CTA 的 shared memory（1024 B 对齐余量另计）：

| | D = 128 | D = 64 |
|---|---|---|
| Q（2 个 tile，epilogue 时兼作 O 的暂存） | 64 KB | 32 KB |
| K/V 环：一级 = 一个“一半” | 8 级 × 16 KB = 128 KB | 16 级 × 8 KB = 128 KB |
| correction 的统计量 | 3 KB | 3 KB |

环的字节数与 40 关一样，但级数翻倍（40 关 D = 128 是 4 级 × 32 KB）：同样的空间能提前装 4 块而不是 2 块。TMEM 与 40 关完全相同，两个 CTA 用同样的列（`cta_group::2` 的 alloc 保证两边拿到同一个基址）。

host 端（已写好）：Q 的 box [128 行][64]，K 的 box **[64 行][64]**，V 的 box D = 128 时 [128 行][64]、D = 64 时 **[128 行][32] + 64B swizzle**，O 的 box [32 行][64]。grid = (2 · ⌈N/512⌉, H, B)，cluster = (2, 1, 1)。

### 5.2 TMA：两个 CTA 装各自的一半，完成在 leader 的 barrier

```
for t: (leader) arrive_expect_tx(q_full[t], 2 × 本 CTA 的 Q 字节)
       tma_load_4d_cg2(sq(t), Q, rows(rank, t), → leader 的 q_full[t])
for u = 0 .. 2·blocks − 1:                     # 偶数 K_{u/2}，奇数 V_{u/2}
       wait 本 CTA 的 kv_empty[u % S]
       (leader) arrive_expect_tx(kv_full[s], 2 × 一半的字节)
       K: tma_load_4d_cg2(ring(s) + panel, K, x = 64·panel, row = 128j + 64·rank) × D/64
       V: tma_load_4d_cg2(ring(s), V, x = rank · D/2, row = 128j)
drain: 再等 S 次 kv_empty（leader 的 multicast commit 全部到达后才能退出）
```

只有 leader arm，字节数是两个 CTA 的和。peer 的 complete_tx 可能比 leader 的 `arrive_expect_tx` 先到：tx 计数暂时为负是允许的，相位仍要等 leader 那 1 次 arrive（33 关 NOTES）。

### 5.3 MMA：leader 一个线程替两个 CTA 发射

顺序与 40 关相同（`PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1}`），每条都是 pair MMA，每个 commit 都 multicast 到两个 CTA（mask `0b11`）：

```
序幕: wait kv_full(K₀); for t: wait q_full[t]; QK^t_0 → commit s_full[t] (both)
      commit kv_empty[0] (both)
for j:
  wait kv_full(V_j), kv_full(K_{j+1})
  for t (j < blocks_t):
    wait_cluster p_ready[t] (phase j)      ← 两个 CTA 共 512 次 arrive
    PV^t_j = P^t (两边 TMEM) · V_j (两边各一半列)
    (t = 1) commit kv_empty[V_j] (both)
    j+1 < blocks_t ? QK^t_{j+1} + commit s_full[t] (both) : commit o_full[t] (both)
  commit kv_empty[K_{j+1}] (both)
```

### 5.4 barrier 地图

同一个 `Barriers` 结构在两个 CTA 的同一偏移各有一份；表中 L = leader，P = peer。

| barrier | 用哪份 | count / 每相位 | 谁 arrive | 谁 wait |
|---|---|---|---|---|
| `q_full[t]` | L | 1 + tx | L 的 TMA（expect_tx 两边字节）；两边 TMA 的 complete_tx | L 的 MMA |
| `kv_full[s]` | L | 1 + tx | 同上 | L 的 MMA |
| `kv_empty[s]` | L 和 P | 1 | L 的 MMA：`commit.cta_group::2.multicast 0b11` | 各自的 TMA |
| `s_full[t]` | L 和 P | 1 | L 的 MMA：multicast commit | 各自的 softmax t |
| `p_ready[t]` | L | 2 × 256 | 两边的 softmax t（128）+ correction（128），远程 arrive | L 的 MMA（`try_wait.acquire.cluster`） |
| `o_full[t]` | L 和 P | 1 | L 的 MMA：multicast commit | 各自的 correction（epilogue） |
| `stats_full[t]`、`final_full[t]` | 各自 | 128 | 本 CTA 的 softmax t | 本 CTA 的 correction |

```
         leader CTA                                   peer CTA
 ┌──────────────────────────────┐            ┌──────────────────────────────┐
 │ TMA ──expect_tx──► q/kv_full ◄──complete_tx── TMA                        │
 │                     │ wait                │                              │
 │                    MMA ══ pair MMA ══════════► (smem 读 Q、K/V 一半；写 TMEM) │
 │  commit multicast ─┼─► s_full, o_full, kv_empty ◄─┼── 同一条 commit 也到 peer │
 │                     ▲                     │                              │
 │  softmax/corr ──arrive── p_ready ◄═══remote arrive═══ softmax/corr       │
 └──────────────────────────────┘            └──────────────────────────────┘
```

**setup**：thread 0 初始化 + `fence.mbarrier_init.release.cluster`；两边 warp 0 `tcgen05.alloc.cta_group::2`（512 列）+ relinquish；`fence::before` → `barrier.cluster` → `fence::after`。cluster barrier 保证对方的 barrier 已初始化之后才有人给它发信号。

**收尾（drain）**：每个 CTA 在退出前必须收完别人发给它的东西：

| 发给谁 | 什么 | 谁保证收完 |
|---|---|---|
| peer、leader | `kv_empty` 的 multicast commit | 各自 TMA 的 drain（多等 S 次） |
| peer、leader | `s_full`、`o_full` 的 multicast commit | softmax 等过每一个 `s_full` 相位，correction 等过 `o_full` |
| leader | peer 的远程 arrive（`p_ready`）、peer 的 TMA complete_tx | leader 的 MMA 等过每一个 `p_ready` 和 `kv_full` 相位 |
| peer | leader 的 MMA 读 peer 的 smem、写 peer 的 TMEM | `o_full` 是最后一个 commit，跟踪全部此前的 MMA |

然后 `fence::before` → `barrier.cluster` → warp 0 `fence::after` + `tcgen05.dealloc.cta_group::2`。

### 5.5 causal

pair tile t 的最后一行是 `q0 + 256t + 255`，它走的块数是 `min(⌈N/128⌉, ⌊(q0 + 256t + 255)/128⌋ + 1)`，两个 CTA 必须一致（它们共用 MMA）。tile 0 比 tile 1 少两块；MMA 在 `j ≥ blocks_0` 时跳过 tile 0。每个 pair tile 里 leader 的 128 行最后一块全被 mask（与 40 关 tile 0 相同，m 仍从前面的块得到，不会产生 NaN）。mask 本身完全沿用 40 关，按本 CTA 的 `qt` 与行号计算。

### 5.6 时间线（稳态，D = 128，一格约 250 ns）

```
leader MMA  │PV⁰QK⁰│      │PV¹QK¹│      │PV⁰QK⁰│      │PV¹QK¹│
Tensor(L,P) │██████│      │██████│      │██████│      │██████│     ← 两个 SM 同时执行同一条 pair MMA
softmax 0 L │      │ld max exp exp exp P │      │ld max exp exp exp P│
softmax 0 P │      │ld max exp exp exp P │      │ld max exp exp exp P│  ← 两边都 arrive 后 PV⁰ 才能发
softmax 1 L │ exp P│      │ld max exp exp exp P│      │ld max …
softmax 1 P │ exp P│      │ld max exp exp exp P│      │ld max …
              ↑p_ready⁰ = 两个 CTA 中较晚的那一个 + 远程 arrive 的延迟
```

## 6. 分步练习

骨架里 40 关的全部内容（softmax、correction、epilogue、角色划分、`setmaxnreg`、host 端 tensor map、dispatch）都已写好。`p.step` 选择 `Config<D, 1|2|3>`；三步是**同一个 kernel 逐步补全**。

### 步骤 1 · TODO(1a)–(1f)：pair QKᵀ

- 1a：host 端的 grid 与 `lab::launch_cluster`（§5.1）；
- 1b：setup 与收尾：pair TMEM 的分配/释放、cluster barrier（§5.4）；
- 1c：`tile_row`：CTA r 的 tile t 从哪一行开始（§3.1）；
- 1d：TMA 装 Q 和 K 的一半（§5.2）；
- 1e：`mma_qk_pair`（§4.2）；
- 1f：MMA 序幕的 multicast commit（§5.3）。

这一步只算第一个 K/V 块：softmax 线程把 `scale · S` 写进 probe（已写好），检查的是 `[B, H, N, min(N,128)]` 的 S，覆盖 cluster 中全部四个 tile 和 K 的两半。

```bash
./lab check 42 --step 1 --case two_tiles      # 第一个 pair tile = leader + peer
./lab check 42 --step 1                       # 含 3 个 cluster、只有 3 个 tile 有效等 case
```

### 步骤 2 · TODO(2a)–(2e)：完整的 2-CTA 流水

- 2a：TMA 装 V 的一半（§5.2）；
- 2b：`mma_pv_pair`，含 D = 64 的 64B swizzle（§4.3）；
- 2c：MMA 主循环（§5.3）；
- 2d：`arrive_leader` 的步骤 2 分支（`.release.cluster`）与 `kPReadyArrivals`（§4.4、§5.4）；
- 2e：causal 的块数（§5.5）。

```bash
./lab check 42 --step 2 --case peer_position   # peer 的 P、V 列的一半
./lab check 42 --step 2 --case position        # D = 64：64B swizzle
./lab check 42 --step 2
./lab bench 42 --step 2 --case perf
```

### 步骤 3 · TODO(3)：跨 CTA 交接的代价

先看现象：

```bash
./lab sass 42        # 在 artifacts/sass/ 的完整 SASS 里找 MEMBAR.ALL.GPU 出现在哪里
./lab bench 42 --step 2 --case perf
```

然后在 `arrive_leader` 的步骤 3 分支换成不带 GPU 范围 fence 的远程 arrive（§4.4），说出为什么本关可以这样做。

```bash
./lab check 42 --step 3
./lab bench 42 --case perf
./lab bench 42 --case perf_causal
./lab bench 42 --case perf_d64
./lab bench 42 --case perf_8k
./lab bench 41 --solution --case perf          # 与 1-CTA（41 关）对比
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 `TIMEOUT` | 只有 leader 装了 K/Q，或 peer 的 TMA 用了自己的 barrier（必须是 `map_shared_rank(…, 0)`）；`arrive_expect_tx` 的字节数不是两边之和；peer 也 arm 了 |
| 步骤 1 只有行 128–255、384–511 错 | `tile_row` 的映射；或 peer 的 Q 装到了别的偏移 |
| 步骤 1 key 64–127 那一半错 | K 的一半的行坐标（`+ 64·rank`）；`b_off` 用了 128 行的 panel |
| 步骤 1/2 在 kernel 结束时挂起或报非法地址 | 收尾少了 cluster barrier 或 drain；peer 提前退出 |
| 步骤 2 `TIMEOUT` | `kPReadyArrivals` 不是 512；leader 用 `mbar_wait`/peer 的 arrive 打到了本 CTA 的 `p_ready`；`o_full`/`s_full` 没 multicast |
| 步骤 2 D = 128 对、D = 64 的 O 错 | V 一半的 swizzle 或描述符：64B swizzle、行宽 64 B、K 方向每 16 行前进 1024 B |
| 步骤 2 O 的后一半列错 | V 的 x 坐标（`rank · D/2`）；idesc 的 N 用了 D/2 |
| causal 只有后面的 cluster 错 | 块数用了 CTA 的行而不是 pair tile 的最后一行 |
| 步骤 3 没有变快 | 仍在用 `.release.cluster`；`./lab sass 42` 看 MEMBAR |

## 7. 性能

### 7.1 相对库基线与 41 关

3 号卡实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时，越大越好；两次运行的范围）：

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| 41 关步骤 3（1-CTA，对照） | ≈ 0.94–0.95× | ≈ 1.04–1.05× | ≈ 0.95× | ≈ 0.92× |
| 步骤 2（2-CTA，`.release.cluster`） | ≈ 0.73–0.74× | ≈ 0.79–0.80× | ≈ 0.80–0.81× | ≈ 0.71–0.75× |
| 步骤 3（2-CTA，`.release.cta`） | ≈ 0.90–0.92× | ≈ 0.96× | ≈ 0.87–0.88× | ≈ 0.89–0.90× |
| 步骤 3 相对 41 关的速度 | ≈ 0.96–0.99× | ≈ 0.93–0.95× | ≈ 0.93–0.94× | ≈ 0.98–0.99× |
| FA4（`flash_attn.cute`） | ≈ 1.15–1.16× | ≈ 1.33–1.35× | ≈ 1.14–1.16× | ≈ 1.08–1.10× |

你的步骤 3 在四个 perf case 上都达到 cuDNN 的约 0.8 倍以上，就说明 2-CTA 的流水是对的。要点：

- 步骤 2 → 3：约 1.2 倍，全部来自 §4.4 的 fence。
- 非 causal 的 D = 128（`perf`、`perf_8k`）与 41 关基本打平；D = 64 和 causal 仍然输（§7.4）。
- 2-CTA 本身没有让本关超过 41 关。下面三节解释为什么，以及它需要什么条件才能赢。

### 7.2 为什么 softmax 受 MUFU 限制时 2-CTA 输

先用 40 关的 softmax（全部 128 个 `exp2` 都走 MUFU，条件 rescale 也没有）做同样的 2-CTA（维护者实验，插桩代码不在仓库中；`perf`，全部 CTA 平均）：

| 指标 | 40 关（1-CTA） | 40 关 softmax + 2-CTA |
|---|---|---|
| 相对 cuDNN | ≈ 0.84× | ≈ 0.75× |
| CTA 主循环（32 块） | ≈ 69.5 µs | ≈ 78 µs |
| 每块的周期（时钟 ≈ 1.62 / 1.66 GHz） | ≈ 3500 | ≈ 4050 |
| Tensor Core 忙碌（NCU，锁频） | ≈ 57% | ≈ 51% |
| MUFU 指令数 | 33.8 M | 33.8 M |
| L2→SM 读（`lts__t_sectors_srcunit_tex_op_read`） | ≈ 1.49 GB | ≈ 1.01 GB（≈ 0.68×；推导 0.53×） |

链被 `exp2` 拉长时，MMA 一侧省下的资源用不上，而两个 SM 被 pair MMA 锁在一起：一个 tile 的 PV 要等两个 CTA 中较晚的 softmax，任何一边的抖动都推迟两个 tile 的交错。每块多出的约 500 周期我们没有完全定位（少量插桩就让差距缩到约 3%），上面是假设，不是测量结论。

一个**只用于测时、结果不正确**的实验指出了方向：softmax 只对一半元素做 `MUFU.EX2`、另一半换成一条 FMA。40 关的主循环从 69.6 降到 61.2 µs 后不再变短（只做 1/4、1/128 也一样），2-CTA 从 78.2 降到 56.4 µs、再到约 55 µs：softmax 足够快时，1-CTA 被 MMA 一侧（QKᵀ 占满 shared memory 带宽，§3.2）卡住，2-CTA 不会，约快 9–12%。

### 7.3 换上 41 关的 softmax

41 关的 softmax 是真实的版本：16 对列中 3 对走多项式（FMA 流水线上约 10 条指令一对），其余走 MUFU；条件 rescale 让 correction 大多数块什么都不做。主循环（`perf`，全部 CTA 平均）：

| | 41 关（1-CTA） | 本关步骤 3 |
|---|---|---|
| CTA 主循环（32 块），D = 128 | ≈ 61.4 µs | ≈ 61.4 µs（打平） |
| 同上，D = 64（`perf_d64`） | ≈ 52.5 µs | ≈ 53.8 µs |
| setup / 收尾 | ≈ 0.45 / 1.1 µs | ≈ 1.0 / 1.8 µs |

- 与 41 关相比，主循环的差距从 2-CTA 版 40 关的约 12% 缩到 0。软件 `exp2` 仍要在 FMA 流水线上做事，softmax 没有快到假实验那样，所以 MMA 一侧的余量只够抵消 pair 的耦合，还不能变成收益。
- 剩下的约 1–4% 来自每个 CTA 一次性的开销：`cta_group::2` 的 TMEM 分配是跨 CTA 的软件协议，收尾多一次 cluster barrier，一对 CTA 要同时退出才能放进下一个 cluster。CTA 寿命约 65 µs，每个 CTA 多约 1.2 µs。persistent kernel（43 关）只付一次这笔开销。
- **多项式比例要不要重调**：2-CTA 每个 SM 的 MMA 与 shared memory 负担更轻，平衡点可能移动。实测（`kPolyPairs` = 2、3、4、5、6）：D = 128 时 3、4、5 在噪声内，2 和 6 变慢；D = 64 时 4 比 3 快约 3%（主循环 53.8 → 51.8 µs，`perf_d64` ≈ 0.81 → 0.785 ms），但 4 让 causal 的 D = 64 kernel 在 192 个寄存器下溢出（`./lab sass` 的 STACK 24）。答案保留 41 关的 3。τ = 8 不改：它由 FP16 的上限决定（41 关 §3.5），不是性能参数。

### 7.4 1-CTA 还是 2-CTA：按配置选择

| 配置 | 实测（步骤 3 相对 41 关） | 原因 | 选择 |
|---|---|---|---|
| 非 causal，D = 128 | ≈ 0.96–0.99× | 主循环打平，每 CTA 多约 1.2 µs 的固定开销 | 非 persistent 时差别在噪声附近；persistent 后（43 关）倾向 2-CTA |
| 非 causal，D = 64 | ≈ 0.93–0.94× | 每块 MMA 只有一半，MMA 一侧更不是瓶颈 | 1-CTA |
| causal | ≈ 0.93–0.95× | 一个 cluster 的 512 行一起前进，对角浪费比 1-CTA 多约 2 块 / 512 行（§3.3） | 1-CTA |

这与 FA4 的选择一致（`flash_attn/cute/interface.py` 中的 `use_2cta_instrs`）：只对**非 causal**、head_dim 补齐到 128 或 192、head_dim_v = 128（以及 hdim 256 的专用 kernel）打开；D = 64 与 causal 不用（代码只写了条件，上表是我们的解释）。FA4 的 `_TUNING_CONFIG` 的键第一项就是 `use_2cta_instrs`：2-CTA 的 D = 128 非 causal 用 softmax 184 个寄存器、`ex2_emu_freq = 10`。FA4 还有本课没有做的部分（persistent、P 分段交给 MMA 等）；我们没有单独测 FA4 自身 1-CTA 与 2-CTA 的差别。

本关的 `run()` **不**按配置切换：每个 case 都走 pair 路径，以便验证它；按上表，一个生产用的 `run()` 只会在非 causal 的 D = 128 上考虑 2-CTA，而本关测得的差别还不足以证明这样做值得。

```bash
./lab profile 42 --case perf
```

看 `sm__pipe_tensor_cycles_active`、`lts__t_sectors_srcunit_tex_op_read`（与 41 关比）和 `smsp__inst_executed_pipe_xu`。

## 8. 自测问题

1. D = 128 时，leader 的 TMA 为 `kv_full[s]` arm 多少字节？`q_full[t]` 呢？如果 peer 也 arm 一次，会发生什么？
2. pair PV 的 idesc 里 N 写 D 还是 D/2？为什么 QKᵀ 的 K 按 key 分、PV 的 V 却按列分？
3. 为什么 `p_ready` 只用 leader 的那份，而 `s_full` 两边都用？`p_ready` 的 count 为什么是 512？
4. peer 的 MMA warp 什么都不做就走到收尾的 cluster barrier。peer 在什么时候之后才能安全地 dealloc TMEM、退出？
5. 把 `p_ready` 的 arrive 换成 `.release.cta` 为什么在本关是可以接受的？举一个必须用 `.release.cluster` 的例子。
6. 本关的 causal 比非 causal 相对 40 关输得更多。用 cluster c 走的块数解释。

<details>
<summary>参考答案</summary>

1. 一半 = 64 key × 128 × 2 B = 16 KB；两个 CTA 共 32 KB。`q_full[t]`：2 × 128 × 128 × 2 B = 64 KB。peer 也 arm 会多一次 arrive（count 1 的相位会被 peer 的 arrive 提前完成，或下一相位多出一次）并多出 tx 字节：读到没装完的数据或挂起。
2. N 写整个 D：硬件把 N 对半分给两个 CTA。`cta_group::2` 的 B 总是按 N 分；QKᵀ 的 N 是 key，PV 的 N 是 D 的列（key 是 PV 的 K 维度，每个 CTA 都要全部 128 个 key）。
3. `p_ready` 的唯一消费者是 leader 的 MMA 线程，所以只需要 leader 那份，两边的生产者都往那里 arrive；`s_full` 的消费者是两边各自的 softmax，所以 commit multicast 到两边各一份。count = 2 个 CTA × (128 个 softmax 线程 + 128 个 correction 线程)。
4. 收完发给它的全部信号并且 leader 的 MMA 不再读写它：它的 TMA 排空了 `kv_empty`，softmax 等过每个 `s_full`，correction 等过 `o_full`（最后一个 commit，跟踪此前全部 MMA，包括读 peer 的 Q、K/V 和写 peer 的 TMEM），然后两边过 cluster barrier。
5. 交接的数据在 TMEM，已由 `tcgen05.wait::st` 完成并由 `tcgen05.fence::before_thread_sync` 排在 arrive 之前；arrive 只需要被计数。若 peer 用普通 store 把数据写进 leader 的 shared memory（DSMEM），再 arrive 通知 leader 去读，就必须 `.release.cluster`（配合 leader 的 `.acquire.cluster`）。
6. cluster c 的两个 pair tile 一起前进，时间由 tile 1 决定：4c + 4 块，两个 SM 都被占用。40 关同样的 512 行是两个 CTA，分别 4c + 2 和 4c + 4 块。N = 4096 平均每 512 行多约 2 块（≈ 6%）；tile 0 早停省不回来。
</details>

## 9. 延伸阅读

- Tri Dao 等，FlashAttention-4（2025）；上游实现 `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`：
  - `__init__`：`use_2cta_instrs`、`cta_group_size`、`mma_tiler_qk = (cta_group_size · m_block_size, n_block_size, D)`、`mma_tiler_pv`、`cluster_shape_mn = (2, 1)`；
  - `_setup_attributes`：`smem_size_kv_per_stage = max(K, V) // cta_group_size`、`kv_stage`；
  - `tma_copy_bytes[...] *= cta_group_size`（leader 一次 arm 两边的字节）；
  - `softmax_correction_threads_cluster`（`p_ready` 对应的 pipeline 生产者数含两个 CTA）；
  - 文件顶部 `_TUNING_CONFIG`（键的第一项是 `use_2cta_instrs`）；
  - `flash_attn/cute/interface.py` 中 `use_2cta_instrs = (...)` 的条件。
- CUTLASS：`third_party/cutlass/include/cutlass/arch/barrier.h`（`ClusterBarrier::arrive(smem_ptr, cta_id, pred)` 的远程 arrive 用默认语义）；`examples/77_blackwell_fmha` 的 FMHA 前向是 1-SM 的；2-SM 只用于 MLA（`kernel/sm100_fmha_mla_tma_warpspecialized.hpp`，`kIs2Sm`、`Allocator2Sm`）。
- CuTe DSL（conda 环境中的 `nvidia_cutlass_dsl/.../cutlass/cute/arch/mbar.py`，`mbarrier_arrive` 的 `scope` 注释）。
- PTX ISA 8.7：[tcgen05.mma（cta_group::2，A 在 TMEM）](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)、[cp.async.bulk.tensor（.cta_group）](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)、[mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)、[tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)、[Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)（scope 与 morally strong）。
- 本仓库：33 关讲义 §4–§5 与 `solutions/33_2cta_gemm/NOTES.md`（pair barrier、drain、racecheck 的误报）；40 关讲义 §3、§5、§7.2。
