# 43_fa4_schedule 参考答案说明

## 设计选择

**起点是 41 关答案步骤 3，CTA 内部不变。** 16 个 warp 的分工、TMEM 布局、K/V 环、softmax 的算术（k = 3 的 MUFU/多项式
混合、τ = 8 的条件 rescale）、全部原有 mbarrier 都沿用。已移植 41 关审查后的修改：`exp2_poly2` 中 f = x − n 用一条
`__ffma2_rn(n, (−1, −1), x)`（位级相同，少两条取负）。三个 step 是同一个 kernel，用 `Config<D, Step>` 的 `kPersistent`、
`kLpt` 两个编译期开关区分。

**work tile 与块分类（步骤 1）。** work tile 仍是 (一对 Q tile, head, batch)，两个 tile 共用 K/V 流，所以块的范围按 pair
算（与 FA4 的 `cta_tiler[0] = 2·128` 相同）。`num_unmasked_blocks(qt, n) = min(⌊n/128⌋, qt/128)` 给出每个 tile 前多少块
完全不需要 mask；softmax 循环拆成 `softmax_block<false>` 与 `softmax_block<true>` 两段。tile 0 的最后一块（对它整块被遮）
仍然照算，与 FA4 相同；精确跳过需要两个 tile 各自计 phase，作为讲义自测题，没有实现。

**统一的 tile 循环。** 每个角色都调用 `for_each_tile(p, body)`，它给出 tile 坐标、块数、`it`（之前的 tile 数）和 `nb`
（之前的块数）。phase：每块一次的 barrier 用 `nb + j`，每 tile 一次的用 `it`，环形 buffer 用跨 tile 连续的 item 号。
静态调度只依赖 (blockIdx.x, gridDim.x, k)，角色之间不需要通信。步骤 1 的 grid = tile 总数，循环只执行一次。

**persistent（步骤 2）的资源交接。**
- Q^t 的 shared memory：MMA 在 tile 的最后一个 Q^t Kᵀ 之后 `tcgen05.commit(q_empty[t])`（只有 1 块的 tile 在序幕里），
  TMA 装 Q^t 前等它；TMA 发射顺序改成 K_0、Q⁰、Q¹、V_0 …，让下一个 tile 的 K_0 在等 Q 释放时已经在路上。
- O 不再经 Q 的 shared memory + TMA store，而是每个线程把自己那一行直接 16 字节 `st.global` 写出。原因：persistent 后 Q
  的 shared memory 尽早交给下一个 tile；D = 128 时 shared memory（≈ 198 KB）放不下单独的 32 KB O staging（差约 1 KB）。
  步骤 1 也用这个 epilogue，所以步骤 1 与 41 关的差别包含它（实测在波动内）。
- O^t 的 TMEM：不需要单独的 barrier。MMA 在下一个 tile 的第一个 PV 前等 `p_ready`（count 256，含 128 个 correction
  线程），correction 只在做完上一个 tile 的 epilogue（`tcgen05.ld` 已完成并 fence）之后才为下一个 tile 的块 0 arrive。
  初版有一个 `o_empty` barrier，独立审查指出它被 `p_ready` 蕴含（去掉后 10/10 通过），已删除。
- `final_full` 与 `st.fin_m/fin_l` **都**按 tile 奇偶双缓冲（`final_full[it & 1][t]`，parity `(it >> 1) & 1`）。只有 1 块
  的 tile（N ≤ 128）中，softmax 为下一个 tile 写最终值、arrive `final_full` 只依赖下一个 tile 的 S_0，不经过 correction，
  可以领先一个 tile。初版只双缓冲了数据：barrier 会在 correction 等待前连续完成两个 phase，correction 永远等不到——
  审查在 `[4,128,500,128]` 上复现了挂起（3/3），现为 case `many_one_block`。领先不会超过一个 tile（讲义 §5.4）。
- drain：TMA 退出前等环形 buffer 和 `q_empty` 的最后一次 commit 都到达。

**LPT 与蛇形（步骤 3）。** causal 时 idx → 层 = idx / (B·H)，pair = P − 1 − 层，bh = idx % (B·H)；`tile_index` 偶数轮
正向、奇数轮反向发牌。非 causal 保持“pair 最快变化”（同一 head 的 tile 相邻，L2 友好），所有 tile 一样重，蛇形不影响均衡。

**为什么不做 head 分段（FA4 的 L2 swizzle）、不做动态调度。** 模拟（每 tile 加 0.5 块的切换开销，见讲义 §3.4）和实测都
表明：静态蛇形要求每一轮的工作量单调，FA4 的分段顺序会破坏它（实测变体 D 慢 26%）；在不 persistent、靠硬件贪心派发
时分段可用（变体 C），但本关 case 上 L2 不是瓶颈，分段不比不分段快。动态调度（原子计数器 + 角色间广播，或 SM100 的
CLC）在模拟中与蛇形相同，没有实现。

## 各 step 的性能（相对 cuDNN，CUDA Graph，同一行同一次 bench）

| 版本 | `perf` | `perf_causal` | `perf_d64` | `perf_8k` | `perf_causal_bh16` |
|---|---|---|---|---|---|
| 41 关答案 | ≈ 0.96× | ≈ 1.04× | ≈ 0.96× | ≈ 0.94× | ≈ 0.73× |
| 步骤 1 | ≈ 0.97× | ≈ 1.04× | ≈ 0.95× | ≈ 0.95× | ≈ 0.78× |
| 步骤 2 | ≈ 1.02× | ≈ 1.04× | ≈ 1.02× | ≈ 0.97× | ≈ 0.69× |
| 步骤 3 | ≈ 1.03× | ≈ 1.27× | ≈ 1.01× | ≈ 0.99× | ≈ 0.97× |
| FA4 | ≈ 1.17× | ≈ 1.31–1.33× | ≈ 1.15× | ≈ 1.10× | ≈ 1.14× |

`./lab selftest 43 --perf`（目标 0.9）通过（审查修正后重测）：`perf` 1.02×、`perf_causal` 1.27×、`perf_d64` 1.02×、
`perf_8k` 0.98×、`perf_causal_bh16` 1.03×（FA4 同次 1.16×、1.32×、1.15×、1.11×、1.14×）。最紧的是 `perf_8k`。

Nsight Compute（锁频）`sm__cycles_active.max / avg`：`perf_causal_bh16` 步骤 1/2/3 = 1.33/1.52/1.04（FA4 1.07）；
`perf_causal` 1.16/1.27/1.03（FA4 1.09）。Tensor Core 忙碌：步骤 3 `perf` 68%（FA4 83%；FA4 此处组合了 persistent、2-CTA 与其他调参，42 关实测单独的 2-CTA 不划算，差距不能归因于 2-CTA），
`perf_causal` 63%（FA4 68%），`perf_causal_bh16` 66%（FA4 75%）。剩下的差距在一块的效率，不在调度。

对照（步骤 3 的 kernel 只改调度，时间相对步骤 3）：LPT + 轮转 1.09/1.13（`perf_causal`/`bh16`）；不 persistent + LPT
（FA4 的 causal 做法）1.07/0.97；再加 FA4 的 head 分段 1.08/0.98；persistent 蛇形 + 分段 1.27/1.26。步骤 3 在
`perf_causal` 上 DRAM 读 535 MB（步骤 1/2、FA4 约 201 MB），但 HBM 吞吐只有约 15%，时间不受影响。

## 验证

- `./lab check 43 --solution`：3 个 step × 37 个 case 全部通过（审查后加 `many_one_block`；`many_one_block`、`multi_tile*` 各重复 5 次全部通过）（含新增 `bh1_causal`、`bh1_long`、`uneven_causal`、
  `multi_tile`（450 tile）、`multi_tile_d128`（480 tile，V = 1）、`multi_tile_causal`（640 tile，jump_max））。
- 骨架：在临时副本里依次填 TODO(1)、(2)（先只填 host 端：GPU 上停在 TODO(2)，不挂起）、(3)：`--step 1` 通过且 step 2
  报 TODO(2)；`--step 2` 通过且 step 3 报 TODO(3)；全部通过。之后骨架恢复为字节一致。
- `./lab sass 43 --solution`：24 个 kernel 均 REG 128（launch 时；`setmaxnreg` 重新分配）、STACK 0。
- `./lab sanitize --solution`（全部 step）：memcheck、racecheck、synccheck 在 `multi_tile`、`two_tiles`、`bh1_causal` 上
  0 错误（审查后又在 `many_one_block` 上跑了三种工具，0 错误）。racecheck 看不见 TMA 与 TMEM 的访问，跨 tile 交接的正确性靠讲义 §5.2–5.4 的论证和 `multi_tile*` case。
- `./lab selftest 43` 与 `./lab selftest 43 --perf` 通过；`python -m pytest -q` 12 项通过。
- 骨架填满后 `bench`：学生版与答案同速（`perf_causal_bh16` 均 ≈ 1.02×）。

## 已知局限

- 蛇形是静态的，按“块数”估计工作量；一轮内工作量差别大时（B·H 很小）抵消不完全。`perf_causal_bh16` 上硬件贪心
  （不 persistent + LPT）比它快约 3%。
- tile 0 的 `X` 块照算（causal 约 1/(2(p+1)) 的多余工作）。
- 没有 2-CTA（42 关）与 FP8（44 关）。
- 波次量化（tile 数不是 148 的整数倍）未处理。
