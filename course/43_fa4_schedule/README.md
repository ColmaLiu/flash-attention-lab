# 43 · 调度与 causal：块分类、persistent、LPT 排序

> 部分 4 · FA4 — Blackwell 上的完整优化　｜　前置：[41_fa4_softmax](../41_fa4_softmax/README.md)、[32_ws_gemm](../32_ws_gemm/README.md)　｜　代码：[`students/43_fa4_schedule/kernel.cu`](../../students/43_fa4_schedule/kernel.cu)

## 1. 本关目标

这是 attention 的最后一关。41 关把一个 CTA 内部的流水（warp 分工、TMEM、softmax 的算术）做到了 cuDNN 的约 0.95 倍；本关不再改一个 CTA **怎么算**，而是改**哪个 CTA 算哪一块、按什么顺序算**：

| | 41 关 | 本关 |
|---|---|---|
| causal 的 mask | 每块用一个运行时判断决定要不要 mask | 每个 (Q tile, K 块) 预先分类：主循环分成**无 mask 段**和**mask 尾段**，热循环里没有 mask 代码 |
| grid | 每个 work tile 一个 CTA（几百到几千个 CTA） | **persistent**：grid = SM 数（148），每个 CTA 循环处理多个 tile |
| tile 顺序 | 硬件按 blockIdx 顺序派发 | causal 时**重的 tile 先做**（LPT），奇数轮反向发牌 |

接口不变：FP16/BF16 的 Q、K、V、O，`[B,N,H,D]`，D = 64 或 128，causal 与非 causal，输出 O 和 FP32 的 LSE。三步：

1. **块分类**：算出每个 Q tile 有几个块不需要 mask，softmax 循环拆成两段；
2. **persistent**：grid = min(tile 数, SM 数)，每个角色走同一个 tile 序列，Q 的 shared memory 和 O 的 TMEM 在相邻两个 tile 之间交接；
3. **LPT**：causal 时按“最后一对 tile 先做”排序，并按“蛇形”（奇数轮反向）分给 CTA。

本关的验收：**5 个 perf case 全部达到 cuDNN 的 0.9 倍**（同一次 CUDA Graph 计时的中位数），其中新加的 `perf_causal_bh16`（B·H = 16、N = 8192、causal）是调度起决定作用的 case：41 关在它上面只有 cuDNN 的约 0.73 倍。

做完后你应该能：

- 对任意 (N, causal) 写出每个 tile 访问哪些块、哪些块要 mask，并算出各 tile 的工作量；
- 说清 persistent kernel 省掉了什么、没省掉什么（例如 tile 数不是 SM 数整数倍造成的“波次量化”它省不掉）；
- 推导轮转、贪心、LPT、蛇形四种分配的完成时间（makespan）上界，并用 `sm__cycles_active.max / .avg` 验证；
- 用 Nsight Compute 的指标解释本关答案与 FA4 还差在哪里。

## 2. 动机：41 关卡在哪里

41 关答案（步骤 3）的实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时）：

| case | 形状 | 41 关 vs cuDNN | FA4 vs cuDNN |
|---|---|---|---|
| `perf` | [4,4096,16,128] BF16 | ≈ 0.96× | ≈ 1.16× |
| `perf_causal` | 同上，causal | ≈ 1.04× | ≈ 1.33× |
| `perf_d64` | [4,4096,32,64] FP16 | ≈ 0.96× | ≈ 1.15× |
| `perf_8k` | [2,8192,16,128] BF16 | ≈ 0.94× | ≈ 1.10× |
| **`perf_causal_bh16`** | **[1,8192,16,128] BF16，causal** | **≈ 0.73×** | ≈ 1.14× |

前四个 case 41 关已经接近 cuDNN；最后一个差了四分之一。它和 `perf_causal` 的区别只是 **B·H 小、N 大**：512 个 work tile，最重的 tile 有 64 个 K/V 块，最轻的只有 2 个。Nsight Compute 给出的线索很直接（锁频）：

```
41 关结构（步骤 1），perf_causal_bh16：
  sm__cycles_active.avg  ≈ 36.3 万周期
  sm__cycles_active.max  ≈ 48.2 万周期      max / avg ≈ 1.33
```

最忙的 SM 比平均多干了 33% 的活：**有的 SM 早早就闲了，kernel 要等最慢的那个**。这不是 CTA 内部算得慢，是工作分得不均。

FlashAttention-2 论文（Dao，2023）第 3.2 节讨论了按序列长度方向并行与 causal 跳块；FA4 的实现（`flash_attn/cute/tile_scheduler.py`）为 causal 专门用了 `SingleTileLPTScheduler`：按“最长处理时间优先”排序，并按 L2 容量把 head 分段。本关就是把这些调度手段一样一样做出来，并测量每一样到底值多少。

## 3. 数学

记号：一个 **work tile** 是一对 Q tile（256 行，行 q0 .. q0 + 255，q0 = 256·p，p 是 pair 下标）× 一个 head × 一个 batch；K/V 块 j 是 key 128 j .. 128 j + 127。每个 pair 每 head 有 `P = ⌈N / 256⌉` 个 work tile，总共 `T = P · B · H` 个。

### 3.1 causal 的块分类

对 Q tile 行 [qt, qt + 127] 和 K 块 j（key [k0, k0 + 127]，k0 = 128 j），causal 的可见条件是 key ≤ row：

| 类别 | 条件 | 处理 |
|---|---|---|
| 全部可见 | k0 + 127 ≤ qt（块里最大的 key ≤ 最小的行），且 k0 + 128 ≤ N | 不需要 mask |
| 对角 / 越界 | 以上不成立，但至少有一个 (row, key) 可见 | 需要 mask |
| 全部被遮 | k0 > qt + 127（块里最小的 key > 最大的行），或 k0 ≥ N | 跳过（不加载） |

qt 是 128 的倍数，所以“全部可见”的块恰好是**开头连续的若干块**：一旦某块的最大 key 超过 qt 或碰到 N，后面的块也一样。骨架里 `num_unmasked_blocks` 要返回的就是这个个数——用上表的两个条件、下面的图自己验证。

一个 work tile 的两个 Q tile 共用 K/V 流（40 关）：pair p 访问块 0 .. `min(⌈N/128⌉, 2p + 2) − 1`。下面是 N = 1024、causal 时每个 Q tile 每个块的类别（`F` 无 mask，`D` 要 mask，`X` 对 tile 0 全部被遮、但 tile 1 要用所以照算，`.` 跳过）：

```
              K 块 j →   0 1 2 3 4 5 6 7     访问块数   其中要 mask
pair 0  tile 0 (行   0)  D X . . . . . .        2          2
        tile 1 (行 128)  F D . . . . . .        2          1
pair 1  tile 0 (行 256)  F F D X . . . .        4          2
        tile 1 (行 384)  F F F D . . . .        4          1
pair 2  tile 0 (行 512)  F F F F D X . .        6          2
        tile 1 (行 640)  F F F F F D . .        6          1
pair 3  tile 0 (行 768)  F F F F F F D X        8          2
        tile 1 (行 896)  F F F F F F F D        8          1
```

- **工作量与 pair 下标成正比**：pair p 有 2p + 2 块。pair 3 是 pair 0 的 4 倍；N = 8192 时最后一对 64 块，是第一对的 32 倍。
- **只有很少的块要 mask**：每对 3 个 (tile, 块) 组合要 mask，与 N 无关。`perf_causal`（N = 4096）每对平均 17 块，要 mask 的只占 3/34 ≈ 9%。所以把 mask 从热循环里拿出去，主要好处是热循环更干净（寄存器、指令），性能收益很小（§7.1）——41 关用一个“整块一致”的运行时分支已经跳过了大部分 mask 代码。
- 表中的 `X`：tile 0 的最后一块对它整块被遮，所以 softmax 得到 P = 0，结果不变；我们仍然算它（FA4 也一样），代价是 causal 时约 1/(2·(p+1)) 的多余工作。精确跳过要让两个 tile 的块数不同，见 §8 第 5 题。

N 不是 128 的倍数时，最后一块对两个 tile 都“越界”，也归入 mask 尾段（`causal_n_bound`、`causal_tail` 等 case）。

### 3.2 工作量、下界与“波次量化”

把一块（两个 tile 各一次 QKᵀ、softmax、PV）的时间记为 1 个单位。总工作量 W = Σ 各 tile 的块数，m = 148 个 SM。任何分配的完成时间 C 都满足

```
C ≥ max( W / m ,  w_max )        （w_max：最重的一个 tile）
```

| case | T（tile 数） | 每 tile 块数 | W / m | w_max |
|---|---|---|---|---|
| `perf` | 1024 | 32 | 221.4 | 32 |
| `perf_d64` | 2048 | 32 | 442.8 | 32 |
| `perf_8k` | 1024 | 64 | 442.8 | 64 |
| `perf_causal` | 1024 | 2, 4, …, 32（各 64 个） | 117.6 | 32 |
| `perf_causal_bh16` | 512 | 2, 4, …, 64（各 16 个） | 114.2 | 64 |

**波次量化**（wave quantization）：非 causal 时所有 tile 一样重。1024 个 tile 分给 148 个 SM，1024 / 148 = 6.92，总有 SM 要做 7 个，C = 7 × 32 = 224 > 221.4，损失 1.2%。

```
perf：1024 个 tile，148 个 SM，每格 = 一个 tile（32 块）
SM   0 │■■■■■■■│
...    │■■■■■■■│        136 个 SM 做 7 个
SM 135 │■■■■■■■│
SM 136 │■■■■■■ │        12 个 SM 做 6 个，最后 1/7 的时间空闲
...    │■■■■■■ │
SM 147 │■■■■■■ │
```

注意：**persistent 并不能消除这一点**——工作的粒度还是一个 tile，1024 个 tile 仍然有 136 个 SM 要做 7 个。persistent 省掉的是 tile 与 tile 之间的开销（§3.5）。要消除量化只能把 tile 切小或让一个 tile 的工作跨 SM 分担（split-KV），本课程不做。

### 3.3 贪心（list scheduling）与它的上界

不 persistent 时（41 关、本关步骤 1），每个 CTA 处理一个 tile；SM 空出来时，硬件按 blockIdx 的顺序派发下一个 CTA。这相当于**贪心调度**：按给定顺序，每个任务交给最先空闲的机器。它的经典上界（Graham，1966）：

```
C_greedy ≤ W / m + w_last        （w_last：决定 C 的那台机器上最后开始的那个任务的工作量）
         ≤ W / m + w_max
```

证明一行：最后结束的任务开始时，所有机器都在忙（否则它会更早开始），所以它的开始时间 ≤ W / m。

自然顺序（blockIdx.x = pair 最快变化）里，最后派发的是最后一个 head 的重 tile，w_last 可能接近 w_max：`perf_causal_bh16` 的上界是 114 + 64 = 178，几乎是下界的 1.6 倍。实测 max / avg ≈ 1.33（§2）。

### 3.4 LPT 与蛇形发牌

**LPT**（longest processing time first）：先按工作量从大到小排序，再贪心。最后开始的任务都是轻的，所以 w_last 小；Graham 证明 LPT 的 C ≤ (4/3 − 1/(3m))·最优。causal 下排序非常简单：pair p 的工作量是 2p + 2，**“层”** ℓ = 0 放所有 (batch, head) 的最后一对（p = P − 1），ℓ = 1 放倒数第二对，……

persistent 之后没有硬件替我们“贪心”了：CTA b 自己决定它的第 k 个 tile 是谁。最简单的**轮转**（round robin）是 `idx = k·G + b`（G = gridDim.x）。在排好序的序列上轮转并不好：每一轮 CTA 0 都拿这一轮最重的，

```
轮转：CTA b 的总量 = Σ_r w(rG + b)
CTA 0 − CTA G−1 = Σ_r [w(rG) − w(rG + G − 1)] ≤ w_max − w_min     （伸缩求和：w(rG+G−1) ≥ w((r+1)G)）
```

`perf_causal_bh16`：w_max − w_min = 62，比平均值的一半还多。

**蛇形**（boustrophedon，“牛耕式”）：偶数轮正向发牌，奇数轮反向发牌。一轮里拿到最轻的 CTA，下一轮拿到最重的。两轮合起来，CTA b 得到 `x_b + y_(G−1−b)`：x 随 b 递减、y_(G−1−b) 随 b 递增，二者的差互相抵消。设 x 这一轮的极差为 Δx、y 为 Δy，则任意两个 CTA 在这两轮的总量差在 [−Δy, Δx] 之内；若两轮的工作量大致线性下降（causal 正是如此：每层比上一层少 2 块），和几乎是常数。

手算：3 个 SM，2 个 head × 4 对（工作量 2、4、6、8，小写 head 0、大写 head 1），W / m = 40 / 3 ≈ 13.3：

```
硬件贪心，自然顺序（41 关）                 静态轮转，自然顺序（步骤 2）
SM0 |aaddddddddDDDDDDDD| 18                 SM0 |aaddddddddCCCCCC| 16
SM1 |bbbbAABBBB        | 10                 SM1 |bbbbAADDDDDDDD  | 14
SM2 |ccccccCCCCCC      | 12                 SM2 |ccccccBBBB      | 10

LPT + 轮转                                   LPT + 蛇形（步骤 3）
SM0 |ddddddddCCCCCCaa| 16                   SM0 |ddddddddBBBBaa| 14
SM1 |DDDDDDDDbbbbAA  | 14                   SM1 |DDDDDDDDbbbbAA| 14
SM2 |ccccccBBBB      | 10                   SM2 |ccccccCCCCCC  | 12
```

对本关的 perf case 做同样的模拟（每个 tile 另加 0.5 块的切换开销，不 persistent 时加 1.5 块；数字是 C / 下界）：

| 分配方式 | `perf_causal` | `perf_causal_bh16` | 对应 |
|---|---|---|---|
| 硬件贪心，自然顺序 | 1.24 | 1.39 | 41 关、步骤 1 |
| 静态轮转，自然顺序 | 1.28 | 1.54 | 步骤 2 |
| 静态轮转，LPT 顺序 | 1.14 | 1.29 | §7.2 对照 A |
| 硬件贪心，LPT 顺序 | 1.08 | 1.07 | FA4 的 causal 做法；§7.2 对照 B |
| **静态蛇形，LPT 顺序** | **1.02** | **1.05** | **步骤 3** |
| 动态贪心（原子计数器），LPT 顺序 | 1.02 | 1.05 | 未实现，作参考 |

实测的 max / avg（§7.2）与这张表的趋势一致。蛇形在这两个 case 上和真正的贪心 LPT 一样好；它是静态的，每个角色自己就能算出 tile 序列，不需要原子操作和角色间通信（§5.5）。它的弱点：如果一轮内的工作量差别很大（例如 B·H 很小、一轮跨很多层），抵消就不完全。

### 3.5 persistent 省掉了什么

一个 CTA 处理完一个 tile 再换下一个 tile，不 persistent 时要经过：

```
CTA 结束：等全部 MMA、epilogue 写完 O → TMEM 释放 → CTA 退出
新 CTA：  启动 → barrier 初始化、TMEM 分配、tensor map 预取 → TMA 读 Q、K_0（全程延迟）→ 第一个 QKᵀ …
```

这段时间里 Tensor Core 和 MUFU 都闲着。persistent 后：TMA warp 在当前 tile 的最后几块还没算完时就开始装下一个 tile 的 K_0、Q（只要环形 buffer 有空位、Q 的上一个使用者已经结束）；correction warpgroup 写上一个 tile 的 O 时，MMA 已经在算下一个 tile 的 QKᵀ。§7.1：非 causal 的三个 case 因此快了约 4–5%，相当于每个 tile 省了约 1.5 块的时间。

### 3.6 L2：一个 head 的 K/V 能不能留在 L2 里

一个 head 的 K + V 是 `2 · N · D · 2` 字节：`perf` / `perf_causal` 是 2 MB，`perf_causal_bh16` 是 4 MB。同一个 head 的 P 个 tile 都要读它（causal 时读前缀）。如果这些 tile 同时在跑，K/V 只从 HBM 读一次，其余从 L2（126 MB）读。

- 自然顺序（pair 最快变化）：同一个 head 的 16 个 tile 下标相邻，同时运行。非 causal 的 case 都用这个顺序。
- LPT 顺序：第 0 层是**所有** head 的最后一对，所以每一轮都要读所有 head 的 K/V。`perf_causal` 共 64 个 head、128 MB，放不进 L2，每一轮都要重新从 HBM 读。
- FA4 的 `SingleTileLPTScheduler` 把 head 分段：一段的 K/V 不超过约 50 MB（`perf_causal` 是 16 个 head），段内 LPT，段与段依次执行。它依赖硬件贪心派发来均衡负载（FA4 的 causal 不是 persistent）。

这里有一个真实的冲突：**完美的 LPT 要求所有 head 的重 tile 先做，L2 局部性要求同一个 head 的 tile 一起做**。对静态蛇形发牌，分段会破坏“每轮工作量单调”的前提（`perf_causal` 的模拟中 C / 下界从 1.02 变成 1.43）。§7.2 实测了这个代价：在本关的 perf case 上，DRAM 读取量的增加（`perf_causal` 从 201 MB 到 535 MB）并没有让 kernel 变慢，HBM 带宽远没有用满；分段反而因负载不均变慢。所以步骤 3 不分段，只在层内按 (batch, head) 顺序排列。

## 4. 硬件与指令

本关没有新指令，用到的都是前面学过的：

### 4.1 SM 数与每 SM 的 CTA 数

```cpp
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);   // B200：148
```

本关一个 CTA 用约 198 KB shared memory（D = 128：Q 64 KB + K/V 环 128 KB + 统计量 5 KB）和全部 512 列 TMEM，每个 SM 只放得下 1 个 CTA，所以 persistent 的 grid = min(T, 148)。如果 grid 大于 SM 数，多出来的 CTA 要等别人**整个**结束才能开始，persistent 就退化成了两波。

### 4.2 硬件怎么派发 CTA

不 persistent 时，CTA 按线性 blockIdx 的顺序派发到空闲 SM 上。这是观察到的行为（FA4 的 LPT 调度就依赖它），CUDA 编程指南并不保证 CTA 的执行顺序。本关步骤 3 的静态调度不依赖它：每个 CTA 自己算自己的 tile 序列。

### 4.3 用 `tcgen05.commit` 释放 shared memory

`tcgen05.commit` 让 mbarrier 在**这个线程之前发射的全部 MMA 完成后**收到一次 arrive（31、32 关）。所以 MMA warp 发完一个 tile 的最后一个 Q^t Kᵀ 就可以 commit `q_empty[t]`：这个 arrive 到达时，读 Q^t 的 MMA 都完成了，TMA 可以往 Q^t 里装下一个 tile 的 Q。不需要等 PV。

### 4.4 跨 tile 的 phase

一个 mbarrier 每完成一个 phase，parity 翻一次（30 关）。persistent 后，同一个 barrier 在一个 CTA 里被很多个 tile 用，phase 要**连续计数**：

| barrier | 每个 tile 用几次 | 第几个 phase | 等待的 parity |
|---|---|---|---|
| `kv_full` / `kv_empty` | 2 × 块数（环形） | 环中的 item 号 u（跨 tile 连续） | `(u / S) & 1`，empty 再 `^ 1` |
| `s_full`、`p_ready`、`stats_full` | 块数 | `nb + j`（nb = 之前各 tile 的块数之和） | `(nb + j) & 1` |
| `q_full`、`o_full` | 1 | `it`（之前做过的 tile 数） | `it & 1` |
| `final_full[2]`（按 tile 奇偶双缓冲，§5.4） | 1（轮流用两个） | 第 `it` 个 tile 用 `final_full[it & 1]` 的第 `it >> 1` 个 phase | `(it >> 1) & 1` |
| `q_empty`（本关新增） | 1 | 等的是“上一个 tile 的释放” | 自己推（下一段） |

“empty” 一类的 barrier（`kv_empty` 也是）第一次等待必须立即返回：一个新初始化的 mbarrier，“phase −1” 视为已完成，等 parity 1 立即返回（30 关）。第 `it` 个 tile 要等的是第 `it − 1` 次释放。骨架里 `for_each_tile` 已经算好了 `it` 和 `nb`，每个角色都调用它，所以各角色看到的 tile 序列和 phase 一致。

### 4.5 O 直接写全局内存

34–41 关的 epilogue 把 O 写进 Q 的 shared memory，再用 TMA store。persistent 后这块 shared memory 已经交给下一个 tile 的 Q 了（§4.3）。本关的 epilogue 让每个线程把自己那一行（D 个 16 位数 = 128 或 256 字节，连续）用 16 字节的 `st.global` 直接写出。一条 warp 指令的 32 个 lane 落在 32 个不同的行上，每个 32 字节 sector 只写了一半（LSU 的 sector 利用率 50%）；同一 sector 的另一半由下一条指令写，L2 把两半合并后才写回 HBM，所以 HBM 流量不浪费。O 只占 epilogue 的一小部分时间，这点 LSU 效率可以接受。

### 4.6 常见误用

| 误用 | 后果 |
|---|---|
| persistent 的 grid 大于 SM 数 | 多出来的 CTA 等第一波结束才开始，尾部翻倍 |
| 跨 tile 时 phase 从 0 重新数（例如 `j & 1`） | 第二个 tile 起 parity 错：等过了头或者永远等不到（挂起），在 `multi_tile*` case 暴露 |
| 下一个 tile 的 Q 不等 `q_empty` 就装 | 上一个 tile 最后的 QKᵀ 读到新 Q：错误随机出现 |
| 每个 tile 开头不重置 m、ℓ | 条件 rescale 的基准从上一个 tile 带过来：`multi_tile_causal`（jump_max）错 |
| 不同角色算出的 tile 序列不同 | 立刻挂起或读错 head |

## 5. 调度设计

### 5.1 不变的部分

warp 分工、TMEM 布局（S⁰/P⁰、S¹/P¹、O⁰、O¹，512 列）、K/V 环、softmax 的算术（41 关步骤 3：MUFU/多项式混合、条件 rescale）全部不变：

```
16 个 warp：
  warp 0–3   softmax WG 0（tile 0）   setmaxnreg 192
  warp 4–7   softmax WG 1（tile 1）   setmaxnreg 192
  warp 8–11  correction WG（修正 O + epilogue）   setmaxnreg 80
  warp 12    MMA   warp 13 TMA   warp 14–15 空闲   setmaxnreg 48
```

### 5.2 资源的生命周期

persistent 之后，每样资源都要回答“下一个 tile 什么时候可以覆盖它、谁来通知”：

| 资源 | 上一个 tile 最后一次使用 | 下一个 tile 第一次写 | 交接方式 |
|---|---|---|---|
| K/V 环的 stage | 最后读它的 MMA | TMA 装新 item | `kv_empty`（与 40 关相同，只是 item 号跨 tile 连续） |
| Q^t（shared） | 最后一个 Q^t Kᵀ | TMA 装下一个 Q^t | **`q_empty[t]`**：MMA 在最后一个 QKᵀ 之后 commit；TMA 等它 |
| S^t / P^t（TMEM） | 最后一个 P^t V | 下一个 tile 的 Q^t K₀ᵀ | 同一个线程按发射顺序执行（40 关 §4.3），不需要 barrier |
| O^t（TMEM） | epilogue 的 `tcgen05.ld` | 下一个 tile 的第一个 P^t V（覆盖） | **已被 `p_ready` 蕴含**，不需要新 barrier：MMA 在这个 PV 之前等 `p_ready[t]`（256 次 arrive，其中 128 次来自 correction），而 correction 是顺序程序，只有做完上一个 tile 的 epilogue（`tcgen05.ld` 已完成、已 fence）才会为下一个 tile 的块 0 arrive |
| `st.alpha` | correction 读 α | softmax 写下一块的 α | 与 40 关相同：softmax 最多领先一块（§5.4） |
| `st.fin_m/fin_l` 与 `final_full` | epilogue 读 / 等 | softmax 写下一个 tile 的最终值 / arrive | **数据和 barrier 都按 tile 奇偶双缓冲**（§5.4） |

### 5.3 时间线：相邻两个 tile 的交接（示意，不按比例）

```
              ┌──────── tile w 的最后一块 ────────┐┌──────────── tile w+1 ────────────────
TMA     : … K_last V_last │K_0(w+1)│ 等 q_empty[0] → Q⁰(w+1)  Q¹(w+1) │V_0 K_1 …
MMA     : PV⁰ QK⁰ PV¹ QK¹ ──── commit q_empty ──┐ PV⁰_last o_full[0] PV¹_last o_full[1] │QK⁰_0 QK¹_0 │ 等 p_ready │PV⁰_0 …
softmax0: … softmax⁰(last) │ fin → final_full │ 等 s_full │ softmax⁰(block 0) …
correct : … 修正 │ epilogue O⁰(w) │ epilogue O¹(w) │ tile w+1 的 j = 0：arrive p_ready …
```

- Q^t 在最后一个 QKᵀ 之后就释放了，TMA 在 PV_last 还在算时就装下一个 Q，所以下一个 tile 的 QKᵀ_0 紧接着 PV_last 开始；
- 下一个 tile 的 softmax 块 0 与上一个 tile 的 epilogue 同时进行；
- 唯一的等待：下一个 tile 的第一个 PV 要等 correction warpgroup 做完上一个 tile 的两个 epilogue（correction 是一个顺序程序：先 epilogue，再进入下一个 tile 的块循环、为块 0 arrive `p_ready`）。

### 5.4 为什么 fin 和 final_full 都要双缓冲

softmax 在一个 tile 结束时把最终的 m、ℓ 写进 `st.fin_*` 并 arrive `final_full`，epilogue 之后等它、读值。下一个 tile 的 softmax 什么时候会再写、再 arrive？它要先拿到下一个 tile 的 S_0，而 S_0 只依赖 Q 和 K_0——**不依赖 correction**。如果下一个 tile 只有 1 块（N ≤ 128），softmax 可能在 correction 还在写第 w 个 tile 的 O 时，就为第 w + 1 个 tile 写了新值并 arrive。两个问题：

1. **数据**：第 w 个 tile 的 m、ℓ 被覆盖；
2. **barrier**：只有一个 `final_full` 时，它连续完成了两个 phase（w 和 w + 1）。correction 随后等 parity `w & 1`——这个 parity 现在指的是“第 w + 2 个 phase”，永远等不到：**挂起**（`many_one_block`：2000 个 1 块的 tile）。mbarrier 的 parity 只能区分相邻的两个 phase，**生产者最多只能领先等待者一个 phase**。

所以数据和 barrier 都按 tile 奇偶（`it & 1`）用两份：第 `it` 个 tile 用 `final_full[it & 1]`，那是它的第 `it >> 1` 个 phase。两份够不够？写第 w + 2 个 tile 之前，softmax 必须拿到第 w + 2 个 tile 的 S_0，它由 MMA 在第 w + 1 个 tile 的 PV 之后发射，而那个 PV 要等 correction 为第 w + 1 个 tile arrive `p_ready`——这时 correction 已经等过、读完了第 w 个 tile 的值。softmax 最多领先一个 tile。

`st.alpha` 不需要：softmax 写下一块（包括下一个 tile 的块 0）的 α 之前要等 `s_full`，对应的 QKᵀ 在 MMA 程序里排在本块的 PV 之后，而那个 PV 等 correction 读完 α 后的 `p_ready`。

### 5.5 tile 序列

```cpp
// 每个角色：
for_each_tile<C, kCausal>(p, [&](const Tile &w, int blocks, int it, int nb) { ... });

for k = 0, 1, …（k·G < T）：
    idx = tile_index<C>(k)          步骤 1：G = T，只有 k = 0；步骤 2：轮转；步骤 3：蛇形
    idx ≥ T：跳过（蛇形的最后一轮不满时，反向发牌会落在 T 之外）
    w = tile_coords(idx)            步骤 3 + causal：LPT 映射；否则 pair 最快变化
    body(w, num_kv_blocks(w.q0), it, nb)；it += 1；nb += blocks
```

## 6. 分步练习

骨架里流水线（TMA、MMA、softmax 的一块、correction、epilogue、host 端）全部写好；每个角色已经用 `for_each_tile` 循环、phase 已经跨 tile 连续计数。三步是同一个 kernel 逐步补全，后面的步骤用到前面的 TODO。

### 步骤 1 · TODO(1a)–(1b)：块分类与循环拆分

- 1a：`num_unmasked_blocks(qt, n)`（§3.1）：tile 前多少块完全不需要 mask；
- 1b：softmax warpgroup 里先用 `softmax_block<false, …>` 跑前 `min(unmasked, blocks)` 块，再用 `softmax_block<true, …>` 跑剩下的。

```bash
./lab check 43 --step 1 --case causal_pair       # 对角块 + tile 0 的整块被遮
./lab check 43 --step 1 --case causal_n_bound    # 对角 mask 与 N 尾部同时
./lab check 43 --step 1 --case short_tail        # 非 causal，只有 N 尾部
./lab check 43 --step 1
./lab bench 43 --step 1 --case perf_causal_bh16
```

### 步骤 2 · TODO(2a)–(2c)：persistent

- 2a（host）：grid = min(T, SM 数)；
- 2b（TMA）：装 Q^t 之前等 `q_empty[t]`；循环结束后再等最后一次 `q_empty`（drain）；
- 2c（MMA）：最后一个读 Q^t 的 QKᵀ 之后 commit `q_empty[t]`（两处：只有 1 块的 tile 在序幕里）。

O^t 的 TMEM 不需要单独交接（§5.2）；`final_full` 的双缓冲（§5.4）骨架里已经写好，读懂它为什么必要。

```bash
./lab check 43 --step 2 --case multi_tile        # 450 个 tile，有的 CTA 做 4 个
./lab check 43 --step 2 --case many_one_block    # 2000 个 1 块的 tile：序幕里的 q_empty
./lab check 43 --step 2 --case bh1_causal        # 8 个 tile < 148 个 SM
./lab check 43 --step 2
./lab bench 43 --step 2 --case perf
./lab bench 43 --step 2 --case perf_causal_bh16  # 预期比步骤 1 慢（§3.4）
```

### 步骤 3 · TODO(3a)–(3b)：LPT 与蛇形

- 3a：`tile_index` 的蛇形发牌（§3.4）；
- 3b：`tile_coords` 的 LPT 映射（§3.4 的“层”）。

```bash
./lab check 43 --step 3 --case uneven_causal     # V[j] = j/N：pair 算错会读到别的位置
./lab check 43 --step 3
./lab bench 43 --case perf_causal_bh16
./lab bench 43 --case perf_causal
./lab profile 43 --case perf_causal_bh16         # 看 sm__cycles_active.max / .avg
```

### 常见错误

| 现象 | 可能原因 |
|---|---|
| 步骤 1 causal 的 case 出现“未来位置”的值（`causal_position`） | `num_unmasked_blocks` 用了 `(qt + 127) / 128` 之类的上界，把对角块算成了无 mask |
| 步骤 1 `short_tail`、`causal_n_bound` 错，其他对 | 忘了 N 的限制（`⌊N / 128⌋`），或没有把 unmasked clamp 到 blocks |
| 步骤 1 `TIMEOUT` | 两段循环漏了或重复了一块：MMA 和 softmax 的块数对不上 |
| 步骤 2 只有 `multi_tile*`、`many_one_block` 挂起 | `q_empty` 的 parity 不对（第一个 tile 必须不阻塞，之后等上一次释放） |
| 步骤 2 只有 `n1`、`one_block`、`causal_n1`、`many_one_block` 挂起 | 只有 1 块的 tile（N ≤ 128）没有在序幕里 commit `q_empty` |
| 步骤 3 `uneven_causal` 错 | LPT 的 pair 没有反过来（层 0 应是每个 head 的最后一对），或层与 bh 的拆分取反了 |
| 步骤 3 正确但 `perf_causal_bh16` 没有变快 | 只做了 LPT 没做蛇形（§3.4 的“LPT + 轮转”） |

## 7. 性能

### 7.1 相对库基线

B200 实测（CUDA Graph；`vs cudnn` = cuDNN 耗时 / 本版本耗时，越大越好；同一行来自同一次 `bench`，cuDNN 自身在多次运行间有约 ±4% 的波动）：

| 版本 | `perf` | `perf_causal` | `perf_d64` | `perf_8k` | `perf_causal_bh16` |
|---|---|---|---|---|---|
| 41 关答案 | ≈ 0.96× | ≈ 1.04× | ≈ 0.96× | ≈ 0.94× | ≈ 0.73× |
| 步骤 1（块分类） | ≈ 0.97× | ≈ 1.04× | ≈ 0.95× | ≈ 0.95× | ≈ 0.78× |
| 步骤 2（persistent，轮转） | ≈ 1.02× | ≈ 1.04× | ≈ 1.02× | ≈ 0.97× | ≈ 0.69× |
| 步骤 3（LPT + 蛇形） | ≈ 1.03× | ≈ 1.27× | ≈ 1.01× | ≈ 0.99× | ≈ 0.97× |
| FA4（`flash_attn.cute`） | ≈ 1.17× | ≈ 1.31–1.33× | ≈ 1.15× | ≈ 1.10× | ≈ 1.14× |

本关的目标：步骤 3 在全部 5 个 perf case 上达到 cuDNN 的 0.9 倍以上。

- **步骤 1** 与 41 关在波动范围内（`perf_causal_bh16` 的 0.73 → 0.78 主要是 cuDNN 两次测量的差别；绝对时间只快了约 2%）。原因见 §3.1：要 mask 的块只占约 9%，41 关已经用整块一致的分支跳过了其余块的 mask 代码。这一步的价值在于把分类讲清楚，并为后两步准备好“每个 tile 的块数”。
- **步骤 2**：非 causal 的三个 case 快了 4–5%（§3.5，tile 间的开销被重叠掉）。`perf_causal_bh16` 反而变慢：硬件的派发本来是“谁空闲给谁”的贪心，静态轮转把它变成了固定分配（§3.4 表中 1.39 → 1.54）。
- **步骤 3**：`perf_causal_bh16` 0.69 → 0.97，`perf_causal` 1.04 → 1.27。非 causal 的 case 不受影响（所有 tile 一样重，顺序不变）。

### 7.2 负载均衡：实测 max / avg 与对照实验

`./lab profile`（锁频）的 `sm__cycles_active.max / sm__cycles_active.avg`：

| 版本 | `perf` | `perf_causal` | `perf_causal_bh16` |
|---|---|---|---|
| 步骤 1 | 1.02 | 1.16 | 1.33 |
| 步骤 2 | 1.03 | 1.27 | 1.52 |
| 步骤 3 | 1.03 | 1.03 | 1.04 |
| FA4 | 1.03 | 1.09 | 1.07 |

`sm__cycles_active.avg` 从步骤 1 到步骤 2 下降约 4–7%（同样的工作，tile 间的空闲少了），这就是 §3.5 的收益；`.max` 才决定 kernel 的时间。

对照实验（只改调度，其余与步骤 3 相同；绝对时间之比，越小越好，以步骤 3 为 1）：

| 变体 | `perf_causal` | `perf_causal_bh16` | 说明 |
|---|---|---|---|
| 步骤 3：persistent，LPT + 蛇形 | 1 | 1 | |
| A：persistent，LPT + 轮转 | 1.09 | 1.13 | §3.4：每轮 CTA 0 都拿最重的 |
| B：不 persistent，LPT 顺序（硬件贪心） | 1.07 | 0.97 | FA4 的 causal 做法（不分段） |
| C：B + FA4 的 head 分段（16 / 8 个 head 一段） | 1.08 | 0.98 | L2 更好，均衡略差 |
| D：persistent，蛇形 + head 分段 | 1.27 | 1.26 | 分段破坏了每轮单调，蛇形失效（§3.6） |

- `perf_causal_bh16` 上硬件贪心（B）比蛇形略好约 3%：真正的贪心能适应每个 tile 实际花的时间，静态分配只能按“块数”估计。代价是每个 tile 都要重新启动 CTA；`perf_causal`（1024 个更短的 tile）上启动开销占得更多，B 就比步骤 3 慢 7%。
- L2：步骤 3 在 `perf_causal` 上从 HBM 读 535 MB（L2 命中率 52%），步骤 1、2 和 FA4 只读约 201 MB（Q、K、V 各 64 MB 的最小值；步骤 1、2 的命中率约 72–74%）。但 535 MB / 0.44 ms ≈ 1.2 TB/s，只有 HBM 标称带宽的约 15%，这些多出来的读取被计算掩盖了——C 比 B 只差 1%。**在本关的 case 上，负载均衡比 L2 重要得多**。若 B·H·N 再大（K/V 远超 L2、且计算更短），结论可能不同。

### 7.3 剩下的差距在哪里（与 FA4 比）

Nsight Compute（锁频，`perf` 与 causal case）：

| | Tensor Core 忙（`sm__pipe_tensor_cycles_active`） | MUFU（`sm__inst_executed_pipe_xu`） | max / avg |
|---|---|---|---|
| 步骤 3，`perf` | 68% | 56% | 1.03 |
| FA4，`perf` | **83%** | **73%** | 1.03 |
| 步骤 3，`perf_causal` | 63% | 52% | 1.03 |
| FA4，`perf_causal` | 68% | 60% | 1.09 |
| 步骤 3，`perf_causal_bh16` | 66% | 55% | 1.04 |
| FA4，`perf_causal_bh16` | 75% | 66% | 1.07 |

读这张表：

- 调度上本关已经不输 FA4：causal 的 max / avg 比 FA4 还低。剩下的差距全在**一块的效率**——每 SM 每块的时间。FA4 的 Tensor Core 比我们忙：`perf` 多 15 个百分点（MUFU 多 17）、`perf_causal_bh16` 多 9（MUFU 多 11）、`perf_causal` 多 5（MUFU 多 8），说明它的“链”（softmax → PV → QKᵀ → softmax，40 关 §3.1）更短。
- 一块的效率差多少：`perf` 上 Tensor Core 忙 68% 对 FA4 的 83%，`perf_causal_bh16` 66% 对 75%，`perf_causal` 63% 对 68%。FA4 在非 causal、D = 128 时把 persistent 调度、2-CTA MMA（`interface.py` 中 `use_2cta_instrs`）和其他调参组合在一起；42 关实测 2-CTA **单独**加在 41 关的 softmax 上（不 persistent）时主循环打平、整体慢约 2.5%，本课程没有测过 persistent + 2-CTA。所以从我们的测量**不能**把 `perf`、`perf_8k` 与 FA4 约 12–15% 的差距归到 2-CTA 上。候选原因（都是源码推断的假设，没有逐一验证）：2-CTA 与 persistent 合用时的收益；FA4 的寄存器分配（softmax 176–200、correction 64–80，见 `_TUNING_CONFIG`）；指数模拟的位置（`ex2_emu_start_frg`）；causal 时按块号**倒序**遍历使 mask 块先做。FA4 的 causal 与 D = 64 不用 2-CTA，差距为 `perf_causal` 约 3%、`perf_causal_bh16` 约 15%、`perf_d64` 约 13%。
- `perf_causal` 本关已与 FA4 相差约 3%；`perf_causal_bh16` 差得多，因为它的 tile 更长（最多 64 块），一块的效率差异被放大，而均衡的优势只有几个百分点。

```bash
./lab profile 43 --case perf_causal_bh16
```

看 `sm__cycles_active.max` 与 `.avg` 的比（负载均衡）、`sm__pipe_tensor_cycles_active` 与 `sm__inst_executed_pipe_xu`（一块的效率）、`lts__t_sector_hit_rate` 与 `dram__bytes_read`（L2）。

## 8. 自测问题

1. N = 1000、causal 时，pair 1（行 256..511）的 tile 0 和 tile 1 各有几个无 mask 的块？这一对共访问几块？
2. `perf_d64` 有多少个 work tile？persistent 之后，每个 SM 做几个 tile？最后一轮有多少个 SM 空闲？
3. 6 个任务（工作量 10、9、8、3、2、1）分给 3 台机器：分别按轮转、蛇形（都在排好序的序列上）分配，完成时间各是多少？下界是多少？
4. 为什么步骤 2 让 `perf_causal_bh16` 变慢了？硬件派发 CTA 和静态轮转的本质区别是什么？
5. 表 §3.1 中的 `X` 块（tile 0 整块被遮的块）如果要跳过，MMA warp、softmax 0、correction 的循环各要怎么改？哪些 barrier 的 phase 计数不能再两个 tile 共用？
6. 如果 `fin` 双缓冲、`final_full` 只有一个，哪个 case 会出错，表现是什么？为什么 `alpha` 和 `stats_full` 不需要双缓冲？

<details>
<summary>参考答案</summary>

1. ⌈1000/128⌉ = 8，2p + 2 = 4，访问 4 块（0..3）。tile 0（qt = 256）：min(256/128, ⌊1000/128⌋) = min(2, 7) = 2；tile 1（qt = 384）：min(3, 7) = 3。
2. P = 4096/256 = 16，B·H = 4·32 = 128，T = 2048。2048 = 13 × 148 + 124：124 个 SM 做 14 个、24 个 SM 做 13 个；最后一轮 24 个 SM 空闲（2048 / 148 = 13.84 轮）。
3. W = 33，下界 11。轮转：10 + 3 = 13、9 + 2 = 11、8 + 1 = 9，C = 13。蛇形：第 2 轮反向，10 + 1 = 11、9 + 2 = 11、8 + 3 = 11，C = 11（最优）。
4. 硬件按顺序把下一个 CTA 交给**最先空闲**的 SM（贪心），轻 tile 做完的 SM 会多拿几个；静态轮转事先固定了每个 CTA 的 tile，自然顺序下第 k 轮 CTA b 拿 pair (k·148 + b) % 32，有的 CTA 总拿到重的，又没有机制补偿（§3.4，模拟 1.39 → 1.54，实测 max / avg 1.33 → 1.52）。
5. 让 tile 0 的块数为 blocks − 1（causal、且最后一块是 tile 0 的 `X` 块时）：MMA 在 j = blocks − 2 时不再发 QK⁰_(j+1)，PV⁰_(j) 之后就 commit `o_full[0]`（和 `q_empty[0]`）；j = blocks − 1 只做 tile 1，K_last 在 QK¹ 之后、V_last 在 PV¹ 之后释放（本来就是 tile 1 最后读）；softmax 0 少跑一块；correction 在 j = blocks − 1 时只处理 t = 1。`s_full`、`p_ready`、`stats_full` 每个 tile 的 phase 要分别计数（nb⁰、nb¹），不能再共用 `nb + j`。
6. 只有 1 块的 tile 连续出现、且 tile 数多于 SM 数时（`many_one_block`：N = 128、B·H = 2000）：softmax 为下一个 tile arrive `final_full` 只需要它的 S_0，不经过 correction，于是 `final_full` 在 correction 等它之前连续完成两个 phase，correction 等的 parity 已经“翻过去又翻回来”——挂起（§5.4）。alpha 的下一次写入和 `stats_full` 的下一次 arrive 都要先等 `s_full`，而那个 QKᵀ 排在等 correction 的 PV 之后，所以最多领先一个 phase。
</details>

## 9. 延伸阅读

- Dao，*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*（2023），§3.2（并行化与 causal 跳块）。
- Zadouri 等，*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*（2026）：调度与 causal 一节。
- R. L. Graham，*Bounds on Multiprocessing Timing Anomalies*，SIAM J. Appl. Math. 17(2), 1969：list scheduling 的 2 − 1/m 界与 LPT 的 4/3 − 1/(3m) 界。
- 上游实现（`third_party/flash-attention`，提交 `1fc0221`）：
  - `flash_attn/cute/tile_scheduler.py`：`StaticPersistentTileScheduler`（grid = SM 数、`tile_idx += grid_dim`）、`SingleTileLPTScheduler`（`size_l2 = 50 MB` 的 head 分段 `l2_minor`、`block = num_block − 1 − block` 的 LPT 反转）、CLC（`clusterlaunchcontrol`，硬件动态调度）；
  - `flash_attn/cute/flash_fwd_sm100.py`：选择调度器的逻辑（causal → `SingleTileLPTScheduler`，非 causal → `StaticPersistentTileScheduler`）、`_TUNING_CONFIG`；
  - `flash_attn/cute/interface.py`：`is_static_persistent`（causal 时为假）、`use_2cta_instrs`（非 causal 的 hdim 128 打开）；
  - `flash_attn/cute/block_info.py`：`get_n_block_min_max`（causal 的块范围）、`get_n_block_min_causal_local_mask`（mask 块与无 mask 块的分界）。
- CUDA C++ Programming Guide：[Thread Block Clusters 与 Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（动态 persistent 调度的硬件支持）；`cudaDeviceGetAttribute` 的 [`cudaDevAttrMultiProcessorCount`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)。
- PTX ISA：[tcgen05.commit](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit)、[mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)。
- 32 关 §5.3–5.4（persistent GEMM、分组光栅化）：GEMM 里 L2 分组是值得的（`wide` 的 HBM 读取降 2.5 倍）；对照本关 §7.2，想一想为什么 attention 里不一样。
