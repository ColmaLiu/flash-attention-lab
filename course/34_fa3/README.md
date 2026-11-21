# 34 · FA3 的思想在 SM100 上：warp 分工、TMEM 中的 S/O 与 softmax–MMA 重叠

> 部分 3 · FA3 的思想在 SM100 上　｜　前置：[22_fa2](../22_fa2/README.md)、[30_tma_mbarrier](../30_tma_mbarrier/README.md)、[31_tcgen05_gemm](../31_tcgen05_gemm/README.md)（推荐先做 [32_ws_gemm](../32_ws_gemm/README.md)：warp specialization）　｜　代码：[`students/34_fa3/kernel.cu`](../../students/34_fa3/kernel.cu)

## 1. 本关目标

把 22 关的 FA2 搬到 `tcgen05` 上，并用上 FlashAttention-3 的思想。接口不变：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]`，D = 64 或 128，causal 与非 causal，输出 O 和 FP32 的 LSE。

和 22 关相比，变化都在“谁在什么时候做什么”：

| | 22 关（FA2，`mma.sync`） | 本关（FA3 思想，`tcgen05`） |
|---|---|---|
| 搬运 | 所有线程 `cp.async`，然后 `__syncthreads` | **一个 TMA warp**，经 mbarrier 环形 buffer 喂数据 |
| 矩阵乘 | 每个 warp 自己发 `mma.sync` | **一个 MMA warp** 的一个线程发全部 `tcgen05.mma` |
| S 在哪 | 寄存器 | **TMEM**，softmax warp 用 `tcgen05.ld` 读 |
| P 怎样进入 P·V | 留在寄存器作 A 操作数 | **写进 shared memory**（SS MMA） |
| V 怎样读 | `ldmatrix.trans` | **MN-major 描述符** + instruction descriptor 的转置位 |
| O 在哪 | 寄存器 | **TMEM**；每块（块 0 除外）softmax warp 都读出、乘 α、写回，不论最大值是否变化（条件 rescale 在 41 关） |
| 重叠 | 无（同一 warp 串行） | **S 双缓冲**：MMA 算 S_{j+1} 时 softmax 处理 S_j |

本关五步：

1. **角色分工与 S = Q·Kᵀ**：TMA warp、MMA warp、softmax warpgroup 各就各位，S 写进 TMEM，softmax warp 读出来写到 probe（单块，N ≤ 128）；
2. **单块 attention**：P 经 shared memory，O += P·V（V 为 MN-major），归一化、TMA 写回 O、写 LSE；
3. **完整 K/V 循环**：online softmax，softmax warp 在 TMEM 里修正 O，K/V 环形 buffer 绕圈；
4. **causal**：跳块、只给对角块和尾块加 mask；
5. **S 双缓冲**：QKᵀ_{j+1} 与 softmax_j 重叠。

做完后你应该能：

- 说出 FA3 的三个思想，以及它们从 Hopper（`wgmma`、寄存器累加器）到 SM100（`tcgen05`、TMEM）各变成了什么；
- 画出本关每个 mbarrier 的生产者、消费者和 parity，并解释每个 `tcgen05.fence` 与 `fence.proxy.async` 的作用；
- 写出 MN-major 操作数的描述符；
- 列出 TMEM 的 512 列怎样分配；
- 解释为什么重叠后瓶颈变成了 softmax，以及 FA4（部分 4）要怎样继续。

## 2. 动机：FA2 在 B200 上卡在哪里

22 关 §7.2 的结论：FA2 只有 cuDNN 的约 0.26 倍，因为 `mma.sync` 本身的上限只有 cuDNN 的约 0.43 倍。31 关换成 `tcgen05` 后，GEMM 从 cuBLAS 的 0.28 倍升到约 0.8 倍。直接把 attention 的两个矩阵乘换成 `tcgen05` 就够了吗？

不够。attention 的每个块是 **GEMM → softmax → GEMM**，中间的 softmax 依赖第一个 GEMM 的结果，第二个 GEMM 依赖 softmax 的结果。按顺序做，Tensor Core 在 softmax 期间闲着，softmax 单元在 GEMM 期间闲着。§3.3 会算出：在 B200 上，D = 128 时一个块的 softmax 所需时间和两个 GEMM **差不多长**。串行做，最多只能用到 Tensor Core 的一半。

FlashAttention-3（Shah 等，2024）针对 Hopper 提出三个思想：

1. **生产者–消费者异步**（论文 §3.1）：
   - *warp specialization*：一部分 warp 专门用 TMA 搬运（生产者），其余 warp 专门计算（消费者），二者通过 shared memory 中的环形 buffer 和 barrier 交接，而不是“所有线程一起搬、一起算”；
   - *ping-pong 调度*：两个消费者 warpgroup 各算一个 Q tile，用 barrier 排好顺序，让一个 warpgroup 的 softmax 和另一个的 GEMM 同时进行（warpgroup **之间**的重叠）。
2. **softmax 与 GEMM 重叠**（论文 §3.2）：*warpgroup 内的两级流水*：在同一个 warpgroup 里，先发出下一块的 QKᵀ（异步的 `wgmma`），再做当前块的 softmax，最后等 QKᵀ 完成（warpgroup **之内**的重叠）。
3. **低精度**（论文 §3.3）：FP8 的 QKᵀ 与 PV，配合块量化和非相干处理（incoherent processing）。本课程在 44 关做 FP8，这里不涉及。

它们在 SM100 上怎样对应：

| FA3 在 Hopper 上 | 本关在 SM100 上 | 留到哪里 |
|---|---|---|
| producer warpgroup（TMA）+ consumer warpgroup，`setmaxnreg` 重分配寄存器 | TMA warp + MMA warp + softmax warpgroup，各自一个角色 | 寄存器重分配：40 关 |
| `wgmma` 的累加器在**寄存器**，S 和 O 都占 consumer 的寄存器 | `tcgen05.mma` 的累加器在 **TMEM**，S、O 都不占寄存器 | — |
| 发 `wgmma` 的就是做 softmax 的 warpgroup；“先发下一块 QKᵀ 再做 softmax”在同一线程内完成 | 发 MMA 的是**另一个 warp**；重叠靠 TMEM 里的**两个 S buffer** | — |
| P 在寄存器，作 `wgmma` 的 A 操作数（RS） | P 写进 **shared memory**（SS MMA） | P 别名写回 TMEM（TS MMA）：40 关 |
| O 的 rescale：consumer 直接乘寄存器 | softmax warp `tcgen05.ld` → 乘 α → `tcgen05.st`，每块都做、同步完成 | 独立 correction warpgroup、条件 rescale：40–41 关 |
| 两个 warpgroup ping-pong | 一个 softmax warpgroup、一个 Q tile | 2 个 Q tile ping-pong：40 关 |
| FP8 | 不做 | 44 关 |

本关故意只做到“一个 softmax warpgroup + S 双缓冲”：这样能清楚地测出重叠带来多少、剩下的瓶颈在哪里（§7），部分 4 再逐个去掉。

## 3. 数学

### 3.1 online softmax：和 22 关相同

一个 CTA 负责 128 行 query。处理第 j 个 K/V 块（128 个 key）时，每一行：

```
m'  = max(m, max_c s_c)                        （s 是未乘 scale 的原始分数）
α   = 2^((m − m')·scale_log2)                  scale_log2 = scale·log₂e
p_c = 2^(s_c·scale_log2 − m'·scale_log2)
ℓ   ← α·ℓ + Σ_c p_c
O   ← α·O + P·V_j
m   ← m'
```

结束后 `O ← O/ℓ`，`LSE = m·scale + ln ℓ`。推导、手算例子、“m 不乘 scale”的理由（scale > 0）以及全被 mask 的行用 0 代替 m'·scale_log2 的处理，都和 22 关 §3 完全一样。

不同的只是**数据放在哪**：S 和 O 在 TMEM 里，m、ℓ 和当前块的 P 在 softmax 线程的寄存器里。本关一个线程负责**一整行**（TMEM 的一个 lane），所以行最大值和行和都不需要 shuffle：22 关的 quad 归约在这里消失了。

### 3.2 O 的修正为什么必须在 P·V 之前

`O ← α·O + P_j·V_j` 中，`P_j·V_j` 由 MMA 直接累加进 TMEM 中的 O（`enable_input_d = 1`）。所以“乘 α”必须在 MMA 读 O 之前完成，而且必须在上一个 `P_{j−1}·V_{j−1}` 写完 O 之后才能读 O：

```
PV_{j−1} 完成  →  softmax 线程：ld O，O *= α_j，st O  →  PV_j 开始
（o_ready）                                           （p_ready）
```

块 0 没有旧状态：`PV_0` 用 `enable_input_d = 0` 覆盖 O（TMEM 分配时是垃圾值），也不需要修正。

### 3.3 一个块要多少周期：为什么必须重叠

B200 每个 SM 每周期约 8192 FLOP（BF16 稠密，31 关 §3.2）。D = 128 时，一个 CTA 的一个块：

| 工作 | 数量 | 约多少周期 |
|---|---|---|
| S = Q·Kᵀ：128×128×128 | 8 条 128×128×16 的 MMA | ≈ 512 |
| O += P·V：128×128×128 | 8 条 128×128×16 的 MMA | ≈ 512 |
| 指数：128×128 个 `MUFU.EX2` | 每 SM 每周期 16 个 | ≈ 1024 |
| 其余 softmax（FFMA、FMNMX、FADD、打包、写 P、修正 O） | 每线程约 600 条 | ≈ 600（与 MUFU 并行发射一部分） |

D = 64 时两个 GEMM 各约 256 周期，指数仍是 1024 周期。

结论：
- 串行做（GEMM → softmax → GEMM），一块约 1024 + 1000+ 周期，Tensor Core 最多忙一半；
- 完美重叠，一块约 max(1024, softmax) 周期；softmax 和 GEMM 一样长甚至更长，**重叠之后 softmax 成为瓶颈**。

这就是 FA4 要做的事情的来源：用两个 softmax warpgroup（40 关）、用 FMA 模拟一部分 `exp2`（41 关）。

## 4. 硬件与指令

本关用到的 mbarrier、TMA、`tcgen05.mma/commit/ld`、TMEM 分配都来自 30、31 关（头文件 [`mbarrier.cuh`](../../include/lab/mbarrier.cuh)、[`tma.cuh`](../../include/lab/tma.cuh)、[`tcgen05.cuh`](../../include/lab/tcgen05.cuh)）。新的有五点。

### 4.1 S 在 TMEM 里：一行一个线程

S 是 128×128 的 FP32 累加器，占 128 列（列 = key，lane = query 行）。softmax warpgroup 是 warp 0–3：warp w 只能访问 lane `32w .. 32w+31`（31 关 §4.2），于是线程 `r = threadIdx.x`（0..127）正好拥有第 r 行：

```
TMEM lane r  ←→  query 行 q0 + r  ←→  softmax 线程 r
tcgen05.ld.32x32b.x32 (lane 32w, col c)：线程 r 得到 S[r][c .. c+31]
一行 128 列 = 4 次 x32 读，放进 float s[128]
```

**`tcgen05.ld` 与 `wait::ld` 之间不能有计算。** `tcgen05.ld` 是异步的：寄存器在 `tcgen05.wait::ld` 之后才有效。31 关的 `tmem_ld_32x32b_x32` + `tmem_wait_ld()` 是两条独立的 asm 语句，`wait` 没有寄存器操作数，编译器不知道 `v[]` 依赖它，可能把“对 v[] 的纯寄存器运算”（求 max、乘 α）调度到 wait 之前，读到还没写好的寄存器。31 关的 epilogue 读完直接存内存，碰巧安全；本关读完就要计算，所以统一用头文件里的 **`lab::tmem_ld_32x32b_x32_sync`**：ld 和 wait 在同一条 asm 语句里，输出寄存器只在 wait 之后才“存在”。代价是同一线程的 4 次读不能同时在路上（本机测得约 1–3% 的差别，见 NOTES）。

### 4.2 S = Q·Kᵀ：和 31 关的 GEMM 一样

Q（`[128][D]`）和 K_j（`[128][D]`）都是 D 连续，即 31 关的 K-major A、B。D = 128 时一行是 256 字节，TMA 按 64 列一个 box 装成两个 **panel**：`[D/64][128 行][64]`，每个 panel 是 31 关的 128B swizzle 布局。第 kk 个 K 片（16 个 d）在 panel `16kk/64`，panel 内起始地址加 `2·(16kk % 64)` 字节：

```cpp
idesc = lab::make_idesc_f16(128, 128, is_bf16);
for kk in 0 .. D/16−1:
  mma(S, desc_K(Q + panel·128·64 + col), desc_K(K + panel·128·64 + col), idesc, kk > 0)
```

### 4.3 O += P·V：V 是 MN-major 的 B 操作数

`tcgen05.mma` 计算 `D = A·Bᵀ`，B 以 “N × K” 的形式给出。P·V 中：

| | M | N | K |
|---|---|---|---|
| 含义 | query 行（128） | d（D） | key（128） |
| A = P | 行 | — | key 连续 → **K-major** |
| B = V | — | d | V 存成 `[key][D]`，**d（N）连续 → MN-major** |

22 关用 `ldmatrix.trans` 解决同一个问题（22 关 §4.4）；`tcgen05` 让硬件直接读 MN-major 的操作数，只需要：

1. **instruction descriptor 第 16 位（B major）置 1**：`lab::make_idesc_f16(128, D, is_bf16, true, false, /*b_mn_major=*/true)`。CUTLASS 叫它 `b_major_ = MN`，常被称为“转置位”；
2. **MN-major 的 shared memory 描述符**，头文件的 `lab::make_smem_desc_mnmajor_sw128(smem, lbo_bytes)`。位域和 K-major 一样（31 关 §4.4），但 LBO、SBO 的含义变了。CUTLASS（`mma_traits_sm100.hpp`，`make_umma_desc<Major::MN>`）给出的 128B swizzle 规范布局（单位为元素）：

```
MN-major, SWIZZLE_128B:  ((8, 8, m), (8, k)) : ((1, 8, LBO), (64, SBO))
                           └ MN 方向 ┘ └ K 方向 ┘

   MN（d）方向：64 个元素（128 字节）连续 = 一个 swizzle 行；第二个 64 列 panel 在 +LBO 字节
   K（key）方向：8 个 key 行（每行 128 字节）= 一个 1024 字节的 atom；下一组 8 行在 +SBO 字节
```

TMA 装进来的 V_j 是 `[D/64 panel][128 key 行][64 d]`，所以：

| 字段 | 值 | 理由 |
|---|---|---|
| SBO | 1024 字节 | 8 个 key 行一个 atom，atom 连续排列 |
| LBO | 128 × 128 = 16384 字节 | 一个 d panel 的大小（D = 64 时只有一个 panel，不使用） |
| 第 kk 个 K 片（16 个 key） | 起始地址 + kk·16 行·128 字节 | 跨过两个 atom |

对比 K-major（31 关）：K-major 时 K 片在**行内**移动（+32 字节），SBO 是 M/N 方向 8 行的距离；MN-major 时 K 片**跨行**移动（+2048 字节），SBO 是 K 方向 8 行的距离。同一个 V tile，不需要任何转置拷贝。

### 4.4 P 经 shared memory：手写 TMA 的布局

P·V 的 A 操作数从 shared memory 读，所以 softmax 线程要把 P（转成 FP16/BF16）写成 **MMA 认识的规范布局**：和 TMA 写 Q 完全一样的 K-major 128B swizzle，`[2 panel][128 行][64 key]`。线程 r 写第 r 行，第 q 个 16 字节 chunk（8 个 key）放在 `q ^ (r % 8)`：

```cpp
*(uint4*)(sP + panel·128·64 + r·64 + (q ^ (r % 8))·8) = pack(p[8q .. 8q+7]);
```

一条 16 字节 store 按 8 个线程一个阶段（wavefront）处理：线程 r..r+7 写各自行的同一个逻辑 chunk q，物理位置 `q ^ (r%8)` 两两不同，覆盖全部 32 个 bank，无冲突。

写完后每个线程执行 **`fence.proxy.async`**（30 关 §4.7）：普通 store 走 generic proxy，`tcgen05.mma` 通过 async proxy 读 shared memory，不加这个 fence，MMA 可能读到旧的 P。

### 4.5 `tcgen05.st`：把修正后的 O 写回 TMEM

```
tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], {r0, ..., r31};
tcgen05.wait::st.sync.aligned;
```

`tcgen05.ld` 的镜像：线程 i 把 32 个寄存器写到 lane `taddr.lane + i` 的 32 列。也是异步的，`wait::st` 之后才算完成。头文件提供 `lab::tmem_st_32x32b_x32` 和 `lab::tmem_wait_st`。修正 O 的一行：

```cpp
for c in 0, 32, ..., D−32:
  tmem_ld_32x32b_x32_sync(O + c, o);  o[i] *= α;  tmem_st_32x32b_x32(O + c, o);
tmem_wait_st();
```

### 4.6 [B,N,H,D] 上的 4 维 tensor map

Q、K、V、O 是 `[B,N,H,D]`：同一个 head 的相邻两行相隔 `H·D` 个元素，不同 batch 相隔 `N·H·D`。把它描述成 4 维 tensor（最快的维在前）：

| 维 | 大小 | 字节步长 | box |
|---|---|---|---|
| 0：d | D | 2 | 64 |
| 1：head | H | D·2 | 1 |
| 2：行 | N | H·D·2 | 128（O 用 32） |
| 3：batch | B | N·H·D·2 | 1 |

一个 box 是某个 (batch, head) 的 128 行 × 64 列，TMA 按 `SWIZZLE_128B` 写成 `[128][64]`。坐标顺序是 `(x = d, y = head, z = 行, w = batch)`。

为什么不能把 B 和 N 合成一维 `B·N`？那样越过 N 的行会读到**下一个 batch** 的行（K/V 的尾块），写 O 时更会**覆盖**下一个 batch 的输出。4 维时越过 N 的行在第 2 维越界：load 填零，store 跳过。这就是尾部不需要任何判断的原因：

- Q 的越界行是 0：这些行算出的结果不会被写回；
- K 的越界行是 0：S = 0，再被 mask 成 −∞；
- V 的越界行是 0：P = 0 乘 0，O 不受影响（如果是旧数据可能是 NaN，`0·NaN = NaN`）；
- O 的 store：越界行不写。

头文件提供 `lab::make_tensor_map_4d`（host，参数按 `[B, N, H, D]` 从外到内的顺序：`d3 = B, d2 = N, d1 = H, d0 = D`）和 `lab::tma_load_4d` / `lab::tma_store_4d`（device），它们和 30 关的 2d/3d 版本只差一个坐标。

### 4.7 交接 TMEM 的两种 fence

| 场景 | 发出方 | 同步 | 接收方 |
|---|---|---|---|
| MMA 写完 S，softmax 读 | `tcgen05.commit`（隐含） | `s_full` | `tcgen05.fence::after_thread_sync`，再 `tcgen05.ld` |
| softmax 读完 S、改完 O，MMA 再写 | `tcgen05.wait::ld/st`，`tcgen05.fence::before_thread_sync` | `mbarrier.arrive(p_ready)` → MMA warp `wait` | `tcgen05.fence::after_thread_sync`，再 `tcgen05.mma` |
| softmax 写完 P（shared memory），MMA 读 | `fence.proxy.async` | 同上（`p_ready`） | — |

`p_ready` 一次交接三件事：P 已写好、本线程对 S 的读已完成（S buffer 可以被覆盖）、本线程的 O 行已修正。

### 4.8 常见误用

| 误用 | 后果 |
|---|---|
| V 的 B 描述符用 K-major、或忘了 idesc 第 16 位 | O 完全错误，`one_position` 立刻发现 |
| MN-major 描述符的 LBO/SBO 写反 | 所有 D 都错：一个 16 key 的 K 片跨两个 8 行 atom，D = 64 时也要用 SBO；D = 128 时第二个 d panel 还要用 LBO |
| P 的 swizzle 用错行号 | `one_position`、`one_ones_v` 错 |
| 写 P 后没有 `fence.proxy.async` | 偶发错误，多在大 case 上出现 |
| softmax 在 `o_ready` 之前就写 P | P_{j−1}·V_{j−1} 读到一半被覆盖的 P：多块 case 偶发错误 |
| 修正 O 后没有 `wait::st` 就 arrive | MMA 可能先于 st 读 O：偶发错误 |
| `tcgen05.ld` 和 `wait::ld` 分开写、中间有计算 | 编译器调度后偶发错误（§4.1） |
| B·N 合成一维的 tensor map | 尾部读到下一个 batch；O 写坏下一个 batch |
| expect_tx 只算有效行 | 挂起（越界填零的部分也计字节） |

## 5. 调度设计

### 5.1 角色、存储预算

```
grid  = (⌈N/128⌉, H, B)，一个 CTA 一个 (Q tile, head, batch)
block = 6 个 warp = 192 线程
  warp 0–3  softmax warpgroup：线程 r ↔ 行 q0+r ↔ TMEM lane r
  warp 4    TMA warp（lane 0）
  warp 5    MMA warp（lane 0）
```

shared memory（1024 字节对齐，全部是 128B swizzle 的 `[panel][行][64]`）：

| | D = 128 | D = 64 |
|---|---|---|
| Q（结束时复用为 O 的暂存区） | 32 KB | 16 KB |
| P（128×128 的 16 位） | 32 KB | 32 KB |
| K stage × 级数 | 32 KB × 2 | 16 KB × 3 |
| V stage × 级数 | 32 KB × 2 | 16 KB × 3 |
| 合计 | 192 KB | 144 KB |

两种 D 都是每 SM 一个 CTA（228 KB 放不下两个）。这是有意的：步骤 5 用满 512 列 TMEM，第二个 CTA 会在 `tcgen05.alloc` 处空等，还占着 shared memory。

**TMEM 预算**（每 SM 512 列，每列 128 lane × 32 位）：

| | 步骤 1–4 | 步骤 5 |
|---|---|---|
| S buffer 0 | 列 0–127 | 列 0–127 |
| S buffer 1 | — | 列 128–255 |
| O（FP32，D 列） | 列 128 起 | 列 256 起 |
| 实际使用（D = 128 / 64） | 256 / 192 | 384 / 320 |
| 分配（2 的幂） | 256 | 512 |

FA4 还要放第二个 Q tile 的 S 和 O（40 关），所以它把 P 写回 S 所在的 TMEM 列，省掉 shared memory 里的 P。

### 5.2 一个块的交接（步骤 2：单块）

```
TMA warp          MMA warp                      softmax warpgroup（128 线程）
─────────         ─────────                     ─────────────────────────────
Q → q_full ─────▶ wait q_full
K_0 → k_full ───▶ wait k_full; QKᵀ → S
                  commit k_empty, s_full ─────▶ wait s_full; ld S; mask; max; exp; ℓ
V_0 → v_full                                    写 P（swizzle）; fence.proxy.async
                                                fence::before; arrive p_ready
                  wait p_ready, v_full ◀──────┘
                  P·V → O
                  commit v_empty, o_ready ────▶ wait o_ready; ld O; ×1/ℓ; 写暂存区
                                                fence.proxy.async; TMA store O; 写 LSE
```

**epilogue**：`o_ready` 跟踪 MMA 线程此前的全部 `tcgen05` 操作，所以它到达时 QKᵀ 也早已读完 Q。Q 的 32 KB 就成了 O 的暂存区：每个线程把自己那行 O/ℓ 按 swizzle 写进去，每个 warp 用一次（D = 128 时两次）TMA store 写回自己的 32 行（box `[32][64]`），越过 N 的行由 TMA 跳过。LSE 每个线程一个 float，相邻线程写相邻地址，自然合并。

### 5.3 完整循环（步骤 3）

```
MMA warp:                                softmax 线程（块 j）:
  issue_qk(0)                              wait s_full; ld S_j; mask; m', α, P, ℓ
  for j:                                   if j > 0: wait o_ready（PV_{j−1} 完成）
    wait p_ready（第 j 次）                写 P_j; fence.proxy.async
    wait v_full; PV_j → O                  if j > 0: O *= α（ld/st，wait::st）
    commit v_empty, o_ready                fence::before; arrive p_ready
    下一次 issue_qk（哪一块？块存在时才发）
```

下一次 QKᵀ 放在 PV_j **之后**：它要写的 S buffer 只有一个，softmax 必须先读完 S_j（`p_ready` 保证）。这样一来，softmax 处理下一块时，它的 S 必须等 PV_j 和这次 QKᵀ 都做完：Tensor Core 和 softmax 轮流工作，**没有重叠**。

注意 softmax 在算完指数**之后**才等 `o_ready`：指数这段时间里 PV_{j−1} 还可以在跑。步骤 3 里这点重叠几乎没有用（S_j 本来就在 PV_{j−1} 之后才算），步骤 5 里它很重要。

### 5.4 causal（步骤 4）

BM = BN = 128，Q tile i 的行是 `128i .. 128i+127`：

- 块数 `min(⌈N/128⌉, ⌈(q0 + 128)/128⌉)`，即块 0..i；之后的块不加载、不计算；
- 需要 mask 的块：越过 N 的块，以及最后一个 key `k0 + 127 > q0` 的块。BM = BN 时只有对角块（块 i）；
- 行 `q0 + r` 可见的 key 数是 `min(N, q0 + r + 1) − k0`，其余设为 −∞。

### 5.5 S 双缓冲（步骤 5）：重叠

把 S 放进两个 TMEM buffer（块 j 用哪个 buffer 由 `s_buffer(j)` 决定，步骤 5 中由你实现），MMA warp 就可以**提前一块**算 S：

```
MMA warp:
  issue_qk(0)
  issue_qk(?)                                 ← 步骤 5 新增：softmax 还在处理 S_0 时就开始
  for j:
    wait p_ready(j)                            S_j 已读完，P_j 已写好，O 已修正
    PV_j; commit v_empty, o_ready
    issue_qk(?)（块存在时）                    ← 复用 S_j 刚空出来的 buffer
```

时间线示意（D = 128；`QK0` = QKᵀ_0，`SM2` = 块 2 的 softmax；按 §3.3，每个 QK、PV 约 512 周期，一次 softmax 约 1000 周期以上，格宽只表示先后，不按比例）：

```
Tensor Core  │QK0│QK· │ ·· │PV0│QK· │ ·· │PV1│QK· │ ·· │PV2│QK· │ ··
softmax WG   │   │SM0 ··········│ SM1 ···········│ SM2 ···········│ SM3
                    └p_ready(0)┘      └p_ready(1)┘      └p_ready(2)┘
                    QK·：步骤 5 中由你决定是哪一块。SM1 开始时 S_1 早已算好；SM1 写 P 前等 o_ready(0)（PV0 已完成）
```

为什么 PV_j 排在循环里那次 QKᵀ **之前**：softmax 处理块 j+1 时，写 P_{j+1} 之前要等 PV_j（`o_ready`）；如果先发那次 QKᵀ，PV_j 要排在它后面，softmax 白等约 512 个周期。

**每个 mbarrier**（S = K/V 级数，`kSBuf` = 1 或 2）：

| barrier | count | 谁 arrive | 谁 wait | 第 j 块用的 parity | 保护什么 |
|---|---|---|---|---|---|
| `q_full` | 1 + tx | TMA warp（expect_tx）+ TMA | MMA | 0 | Q 已到达 |
| `k_full[s]` | 1 + tx | TMA warp + TMA | MMA（`issue_qk`） | `(j/S)&1` | K_j 已到达 stage s |
| `v_full[s]` | 1 + tx | TMA warp + TMA | MMA（PV 前） | `(j/S)&1` | V_j 已到达 |
| `k_empty[s]` | 1 | `tcgen05.commit`（QKᵀ_j 后） | TMA warp（装 K_{j+S} 前） | `((j/S)&1)^1`（装块 j 时） | stage s 的 K 已被读完 |
| `v_empty[s]` | 1 | `tcgen05.commit`（PV_j 后） | TMA warp | 同上 | stage s 的 V 已被读完 |
| `s_full[b]` | 1 | `tcgen05.commit`（QKᵀ_j 后） | softmax 128 线程 | `s_phase(j)`：单 buffer 为 `j&1`，双 buffer 在步骤 5 中推导 | S_j 已在 buffer `s_buffer(j)` |
| `p_ready` | 128 | 每个 softmax 线程 | MMA | `j&1` | P_j 已写好、S_j 已读完、O 已修正 |
| `o_ready` | 1 | `tcgen05.commit`（PV_j 后） | softmax 128 线程 | `j&1` | O 含块 j；P 可以覆盖 |

parity 为什么不会“追尾”：一个 barrier 如果在等待者看到第 j 个阶段之前就完成了第 j+1 个阶段，parity 又变回原值，等待者永远等不到。逐个检查：`p_ready` 的第 j+1 个阶段需要 softmax 等到 `o_ready(j)`，而它只能在 MMA 等到 `p_ready(j)` 之后出现；`o_ready` 的第 j+1 个阶段需要 `p_ready(j+1)`，而它在 softmax 等到 `o_ready(j)` 之后；`s_full[b]` 的下一次 commit 在 `p_ready` 之后；empty/full 是 30 关的标准环形 buffer。

**退出前的排空**：TMA warp 在循环后再等一遍每个 stage 的最后一次 `empty`。否则 CTA 可能在某个 `tcgen05.commit` 的 arrive 还没落到 shared memory 时就退出，下一个 CTA 在同一块 shared memory 上初始化 barrier。

## 6. 分步练习

骨架中 launch、host 端 tensor map、barrier 初始化、TMEM 分配/释放、probe 写出、mask 的尾部部分都已写好。本关新用到的头文件函数：`lab::make_tensor_map_4d`、`lab::tma_load_4d/tma_store_4d`、`lab::make_smem_desc_mnmajor_sw128`、`lab::tmem_ld_32x32b_x32_sync`、`lab::tmem_st_32x32b_x32/tmem_wait_st`。每个 `TODO` 前的注释写明了输入、输出和可用的 helper。

### 步骤 1 · TODO(1a)–(1d)：角色分工与 S = Q·Kᵀ

- 1a：TMA warp 装 Q，以及循环里的 K_j（等 `k_empty`、arm `k_full`、逐 panel `tma_load_4d`）；
- 1b：`mma_qk`（§4.2）；
- 1c：`load_s_row`（§4.1）；
- 1d：MMA warp 的 `issue_qk`。

softmax 线程把 `scale·S` 写进 probe（`[B,H,N,N]`，已写好），由 `fa3_scores` oracle 对照。步骤 1、2 只处理一个块（N ≤ 128）。

```bash
./lab check 34 --step 1
./lab sass 34          # UTMALDG、UTCHMMA、LDTM
```

报告的维度名是 `[b, h, i, j]`：`i` 错查 Q（坐标、panel），`j` 错查 K；只有 D = 128 的 case 错，查第二个 panel 的偏移。

### 步骤 2 · TODO(2a)–(2h)：单块 attention

- 2a：`mma_pv`（§4.3，描述符用 `lab::make_smem_desc_mnmajor_sw128`）；
- 2b：`softmax_block`（max、exp、行和）；2c：`write_p_row`（§4.4）；
- 2d：`store_o_and_lse`（§5.2 epilogue）；2e：TMA warp 装 V_j；
- 2f：MMA warp 的 PV；2g：softmax 交出 P（§4.7）；2h：epilogue 前等最后一个 PV。

```bash
./lab check 34 --step 2 --case one_position
./lab check 34 --step 2
```

### 步骤 3 · TODO(3a)–(3d)：完整循环与 O 修正

- 3a：`rescale_o_row`（§4.5）；3b：MMA warp 发下一个 QKᵀ（§5.3：哪一块、什么条件下发、为什么放在 PV 之后）；
- 3c：α；3d：softmax 在写 P、修正 O 之前等 `o_ready`。

```bash
./lab check 34 --step 3 --case ring
./lab check 34 --step 3
./lab bench 34 --step 3 --case perf
```

### 步骤 4 · TODO(4a)–(4c)：causal

```bash
./lab check 34 --step 4
./lab bench 34 --step 4 --case perf_causal
```

### 步骤 5 · TODO(5a)–(5d)：S 双缓冲

步骤 1–4 中，MMA warp 和 softmax warp 都通过 `s_buffer<kSBuf>(i)`（S_i 在哪个 TMEM buffer）和 `s_phase<kSBuf>(i)`（等 `s_full[buffer]` 的哪个 parity）这两个函数找到 S_i；单 buffer 的分支已经给出。本步实现两个 buffer 的协议（§5.5）：

- 5a：MMA warp 在循环之前提前发出 QKᵀ_1；
- 5b：PV_j 之后，往刚空出来的 buffer 里发哪一块；
- 5c：`s_buffer` 的两 buffer 分支；
- 5d：`s_phase` 的两 buffer 分支。推导方法：`s_full[b]` 每收到一次 commit（每个用 buffer b 的块）完成一个阶段；块 i 是第 k 个（从 0 数）使用它那个 buffer 的块，它的 S 就完成第 k 个阶段，等它就是等 parity `k & 1`。

两边用的是同一对函数，所以 MMA 写进哪个 buffer、commit 哪个 barrier，和 softmax 读哪个 buffer、等哪个 parity 自动一致。写错时最常见的现象是挂起（等一个永远不会完成的阶段），其次是读到上一块的 S。

通过之后做一个实验：把循环里“PV_j”和“下一次 QKᵀ”的发射顺序对调，`./lab bench 34 --case perf` 对比，并用 §5.5 的时间线解释差别（softmax 写 P_{j+1} 前等的是哪个 barrier？它前面排着什么？）。改回来再继续。

```bash
./lab check 34
./lab bench 34 --case perf
./lab bench 34 --case perf_causal
./lab bench 34 --case perf_d64
./lab profile 34 --case perf
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| `TIMEOUT` | expect_tx 字节数；某个 barrier 的 parity；`p_ready` 的 count 不是 128（有线程没 arrive）；MMA warp 和 softmax 算出的块数不同 |
| step 1 只有 D = 128 错 | Q 或 K 第二个 panel 的地址（`panel·128·64`） |
| step 2 `one_ones_v` 大于 1 | 填充 key 没有 mask（P ≠ 0） |
| step 2 只有 D = 128 的 O 后 64 列错 | MN-major 描述符的 LBO |
| step 2 `one_position` 错、`one_uniform` 对 | V 的 K 片偏移，或 idesc 的 B major 位 |
| step 2 LSE 对、O 全错 | P 的 swizzle 或打包顺序 |
| step 3 `ring` 错或挂起，其他对 | K/V 的 stage 下标、empty 的 parity |
| step 3 `peaked*` 错、`uniform`/`ones_v` 对 | O 没有修正，或修正用的 α 和 ℓ 用的不同 |
| step 3 偶发错误 | 少了 `o_ready` 等待、`wait::st`、`fence.proxy.async` 之一 |
| step 4 第一个 Q tile 正确、后面全错 | 块数少算了对角块 |
| step 5 挂起 | `s_phase` 的两 buffer 分支错；或 5b 发的块号使某个 buffer 被连续写两次 |
| step 5 结果错但不挂起 | `s_buffer` 与 5b 不一致：softmax 读到的是另一块的 S |

## 7. 性能

### 7.1 相对库基线

6 号卡实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时，越大越好）：

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 22 关步骤 4（FA2，`mma.sync`） | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| 步骤 3/4（单 S buffer，无重叠） | ≈ 0.41× | ≈ 0.51× | ≈ 0.41× |
| 步骤 5（S 双缓冲） | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× |
| FA4（`flash_attn.cute`） | ≈ 1.1× | ≈ 1.3× | ≈ 1.15× |

步骤 5 在三个 perf case 上都达到 cuDNN 的 0.48 倍以上，才说明重叠起了作用（没有重叠的步骤 3/4 约 0.41 倍）。多次运行之间 cuDNN 本身有几个百分点的波动，请看比例的范围。

- **换成 `tcgen05` + warp 分工**（22 关 → 步骤 3）：约 1.6 倍。此时 Tensor Core 和 softmax 还在轮流工作。
- **S 双缓冲**（步骤 3 → 5）：约 1.3–1.4 倍，D = 64 更多。§3.3 的估算是“串行 ≈ MMA + softmax，重叠 ≈ max(MMA, softmax)”。D = 128 时 MMA ≈ softmax，理论上最多快 2 倍；实测 1.3 倍，因为 softmax 本身比估算的更长（下一节）。
- **causal 比非 causal 的比例高**：cuDNN 的 causal 相对更慢；本关跳块后工作量约 51%。Q tile 按 `blockIdx.x` 顺序发射，最重的 tile 最后才开始，43 关处理。

### 7.2 瓶颈在哪：softmax

```bash
./lab profile 34 --case perf --step 3
./lab profile 34 --case perf
```

本机 Nsight Compute（`perf`）：

| 指标 | 步骤 3 | 步骤 5 | 说明 |
|---|---|---|---|
| `sm__pipe_tensor_cycles_active` | ≈ 27% | ≈ 38% | Tensor Core 忙碌比例 |
| `smsp__inst_executed_pipe_xu` | 3383 万 | 3383 万 | ≈ B·H·N²/32：每个分数一条 `MUFU.EX2`（warp 指令），加少量 α、log |
| `dram__bytes_read` | ≈ 201 MB | ≈ 201 MB | Q、K、V 各读一次（K/V 对同一 head 的 32 个 Q tile 在 L2 命中） |
| `sm__warps_active` | ≈ 9% | ≈ 9% | 每 SM 6 个 warp |

（Nsight Compute 锁定时钟，`gpu__time_duration` 比 bench 长，只用它比较指标。）

步骤 5 后 Tensor Core 仍有约 60% 的时间空闲：它在等 softmax。一个 softmax warpgroup 每块要做 128×128 次 `exp2`（MUFU 每 SM 每周期 16 个，≈ 1024 周期）、逐行的 max/sum/FFMA、写 32 KB 的 P，还要每块把 O 读出、乘 α、写回（D = 128 时每线程 128 个值）。这些都在一个 warpgroup 里串行。部分 4 的 FA4 逐项处理：

| 剩下的问题 | FA4 的办法 | 关卡 |
|---|---|---|
| 只有一个 softmax warpgroup，它在等 MMA 时也没人填补 | 两个 Q tile、两个 softmax warpgroup ping-pong | 40 |
| O 的修正在 softmax 的关键路径上 | 独立的 correction warpgroup | 40 |
| P 写 shared memory（32 KB/块）又被 MMA 读回 | P 写回 TMEM，TS-MMA | 40 |
| MUFU 吞吐 | 一部分 `exp2` 用 FMA 多项式 | 41 |
| 每块都修正 O | 只在最大值增长超过阈值时修正 | 41 |

**可以自己做的实验**：

- 在 `softmax_block` 里只用 1 个 max/sum 累加器，看 SASS 中的依赖链和 bench 的变化。

## 8. 自测问题

1. 步骤 5 中 D = 128，TMEM 实际用了多少列？为什么分配 512 列？如果想在同一个 CTA 里再放一个 Q tile 的 S 和 O，还差多少？
2. softmax 线程 70 读 S 的第 96–127 列，`tcgen05.ld` 的 taddr（相对 TMEM base、在步骤 5 的 buffer 1 中）是多少？
3. V_j 在 shared 地址 `0x10000`，D = 128。写出第 kk = 5 个 K 片（key 80–95）的 B 描述符的起始地址字段和 LBO 字段（右移 4 位之后的值）。
4. 如果 softmax 先等 `o_ready(j−1)` 再算指数（而不是算完再等），步骤 3 和步骤 5 的性能分别会怎样？
5. 为什么 `p_ready` 的 count 是 128，而 `s_full`、`o_ready` 是 1？能不能让每个 softmax warp 只 arrive 一次？
6. 步骤 5 中，MMA warp 往 S_j 用过的 buffer 里发下一次 QKᵀ 时，为什么可以确定 softmax 已经读完了 S_j？哪几条指令共同保证了这一点？

<details>
<summary>参考答案</summary>

1. S 两个 buffer 256 列 + O 128 列 = 384 列；分配必须是 2 的幂，所以 512。再放一个 Q tile 需要再 256（S 双缓冲）+ 128（O）= 384 列，超过剩下的 128 列。FA4 的办法是每个 Q tile 只给 S 一个 buffer，并让 P 复用 S 的列（40 关）。
2. 线程 70 在 warp 2，lane 段从 64 开始：taddr = `(64 << 16) | (128 + 96)` = `(64 << 16) | 224`。线程 70 拿到 lane 70 = 第 70 行。
3. key 80 在第 80 行：起始地址 `0x10000 + 80·128 = 0x12800`，字段值 `0x1280`；LBO = 16384 字节，字段值 `0x400`。
4. 步骤 3：几乎不变，S_j 本来就在 PV_{j−1} 之后才算完，等 `o_ready` 不需要额外时间。步骤 5：变慢，softmax 开始时 PV_{j−1} 可能还在跑，指数计算不能和它重叠。
5. `tcgen05.commit` 本身只 arrive 一次，所以由 MMA 产生的 barrier count 为 1。`p_ready` 要保证 128 个线程**每一个**都写完了自己那行 P、读完了自己那行 S、改完了 O；每个线程的 fence 只覆盖它自己的访问。可以改成每个 warp `__syncwarp()` 后由一个 lane arrive（count = 4），前提是每个线程先各自执行 `fence.proxy.async` 和 `tcgen05.fence::before_thread_sync`，再 `__syncwarp`。
6. 这次 QKᵀ 是在 MMA warp 等到 `p_ready(j)` 之后才发出的。每个 softmax 线程在 arrive 之前：`tcgen05.ld` 读 S_j、`tcgen05.wait::ld`（读已完成）、`tcgen05.fence::before_thread_sync`；MMA 线程 `mbarrier` wait 之后执行 `tcgen05.fence::after_thread_sync` 再发 MMA。`p_ready` 的 count 是 128，所以 128 行都读完了。
</details>

## 9. 延伸阅读

- Jay Shah 等，*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*（2024）：§3.1（producer–consumer、warp specialization、ping-pong 调度）、§3.2（warpgroup 内的 softmax–GEMM 两级流水）、§3.3（FP8，本课程 44 关）
- PTX ISA 8.7：[tcgen05.st](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)、[tcgen05.wait](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait)、[Shared Memory Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)、[Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)、[tcgen05 Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)、[cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- 上游源码（`third_party/` 中固定版本）：
  - `third_party/flash-attention/hopper/flash_fwd_kernel_sm90.h`：FA3 的 producer/consumer warpgroup 划分；`mainloop_fwd_sm90_tma_gmma_ws.hpp`：K/V 流水的 barrier、`IntraWGOverlap`（§3.2 的两级流水）；
  - `third_party/cutlass/include/cute/atom/mma_traits_sm100.hpp`：`make_umma_desc`，MN-major 与 K-major 的规范布局（§4.3 的表）；
  - `third_party/cutlass/examples/77_blackwell_fmha/`：CUTLASS 的 SM100 FMHA，`collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp` 中的角色划分、TMEM 分配与 S/O 的交接；
  - `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py`：FA4 的完整结构。现在读它，可以对照 §7.2 的表找出本关还没有的部分（两个 Q tile、correction warpgroup、P 在 TMEM、软件 `exp2`）。
- 后续：40 关（FA4 的流水结构）。
