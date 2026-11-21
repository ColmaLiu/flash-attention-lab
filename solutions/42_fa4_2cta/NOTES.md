# 42_fa4_2cta 参考答案说明

## 设计选择

**起点是 41 关步骤 3 的答案**（40 关的流水 + 41 关的 softmax：`kPolyPairs = 3` 的 MUFU/多项式混合、τ = 8 的条件
rescale、correction 按 warp 投票跳过修正），只加入 2-CTA；三个步骤都用这套 softmax。最初版本从 40 关出发（维护者
随后决定 42 关建立在 41 关之上），那一版的测量保留在下面“40 关 softmax 下的 2-CTA”一节。没有 43 关的 persistent /
LPT。40 关的独立审查（`5b4157a`）只改了注释与 case；本关吸收了它的 `tile1_position`（改名 `peer_position`），并加入
41 关的 `peaked_fp16`、`growing`、`growing_causal`、`jump`、`jump_causal`（条件 rescale 在两个 CTA 中各自投票）。

**pair 布局（与 FA4 `use_2cta_instrs` 相同）。** cluster = 2 个 CTA，负责 512 行；CTA r 的 tile t 是
`q0 + 256t + 128r` 起 128 行，pair tile t 因此是连续的 256 行。QKᵀ：idesc M = 256、N = 128，B = K 按 key 对半
（每个 CTA 64 个 key，box [64 行][64]）。PV：TS pair MMA，idesc M = 256、N = D，B = V 按 **列** 对半（每个 CTA 全部
128 个 key × D/2 列）。D = 64 时 V 的一半只有 32 列 = 64 字节一行，用 64B swizzle（TMA `SWIZZLE_64B`，新的
`make_smem_desc_mnmajor_sw64`）。FA4 的 `smem_size_kv_per_stage // cta_group_size` 与此一致。

**环。** 一级 = 一个“一半”（K 或 V 的一半字节数相同：D = 128 为 16 KB，D = 64 为 8 KB），环的总字节与 40 关相同
（128 KB），级数翻倍：D = 128 为 8 级、D = 64 为 16 级。

**barrier。** 结构与 40 关相同，同一偏移在两个 CTA 各有一份：`q_full`/`kv_full` 只用 leader 的（leader 的 TMA
arm 两边字节，两边的 `cta_group::2` TMA complete_tx 到 leader）；`kv_empty`/`s_full`/`o_full` 两边都用（leader
的 `tcgen05.commit.cta_group::2 ... multicast 0b11`）；`p_ready` 只用 leader 的，count 512 = 两个 CTA 的 softmax t
（128）+ correction（128），远程 arrive；`stats_full`/`final_full` 仍是各 CTA 本地。MMA warp 只在 leader 里工作。

**setup / drain / teardown（33 关的规则）。** `fence.mbarrier_init.release.cluster` + 两边 warp 0
`tcgen05.alloc.cta_group::2`（512 列）+ `fence::before` / `barrier.cluster` / `fence::after`。每个 CTA 退出前收完
别人发来的信号：TMA 排空 `kv_empty`（多等 S 次）；softmax 等过每个 `s_full` 相位；correction 等过 `o_full`（最后
一个 commit，跟踪 leader 此前全部 MMA，包括读 peer 的 smem、写 peer 的 TMEM）；leader 的 MMA 等过每个 `p_ready`、
`kv_full` 相位（包括 peer 的远程 arrive 和 complete_tx）。最后 cluster barrier 后 `dealloc.cta_group::2`。

**causal。** pair tile t 走到 `q0 + 256t + 255` 所需的块：tile 0 比 tile 1 少两块，MMA 在 `j ≥ blocks_0` 时跳过
tile 0。实测这几乎不省时间（`perf_causal` ≈ 1%）：ping-pong 中 cluster 的时间由 tile 1 决定。与 40 关相比，2-CTA
的 causal 每 512 行多约 2 块 SM 时间（讲义 §3.3）。

**步骤。** 步骤 1：只有第一个 K/V 块的 pair QKᵀ，softmax 线程把 `scale·S` 写进 probe（新 oracle
`probe_fa4c_scores`：`[B,H,N,min(N,128)]`，覆盖 cluster 的四个 tile 和 K 的两半）。步骤 2：完整流水，跨 CTA 的
arrive 用 `.release.cluster`（33 关的 `mbar_arrive_cluster`）。步骤 3：换成 `.release.cta`（新的
`mbar_arrive_remote`）。

**为什么步骤 3 只改一个 arrive。** 最初的完整版本（40 关 softmax、`.release.cluster`）比 40 关慢约 25%。插桩（`globaltimer`，一个
cluster）发现：两个 CTA 收到 `s_full` 的时间差 ≤ 32 ns，最后一个线程约 1350 ns 时 arrive，leader 的 MMA 却在
约 2000–2300 ns 才通过 `p_ready`。SASS：`mbarrier.arrive.release.cluster` = `MEMBAR.ALL.CTA; MEMBAR.ALL.GPU;
ERRBAR; CGAERRBAR; SYNCS.ARRIVE...`，每个线程每块每 tile 一次；默认的 `.release.cta` 只有 `SYNCS.ARRIVE`。
CUTLASS（`ClusterBarrier::arrive(cta_id)`）和 FA4（pipeline `consumer_release`）用的都是默认语义，CuTe DSL
`mbar.py` 注释里说明 cluster 作用域“实测更慢”，默认 CTA 依赖 barrier 计数与当前硬件/编译器行为。形式上跨 CTA 的
`.release.cta` 与 `acquire.cluster` 不构成 synchronizes-with，TMEM 也不例外；有保证的只是 arrive 被计数、本线程
的 `wait::st`/`wait::ld` 已完成。步骤 3 有意识地用这个形式保证换速度（讲义 §4.4），与 CUTLASS/FA4 相同。

**未采用的尝试**（同次对比）：每 warp 一个远程 arrive（count 16）快约 1%，为与 FA4 一致保留每线程 arrive；leader
本 CTA 的线程用本地 `mbarrier.arrive.shared::cta`、只有 peer 远程——略慢；MMA 用普通 `try_wait`（不带
`.acquire.cluster`）——无差别；收尾在 cluster barrier 前加 `__syncthreads` 让空闲 warp 停在硬件 barrier 上——无差别；
寄存器 192/72/56、184/80/64——更慢或溢出更多。

**寄存器。** causal 的 D = 128 kernel 最初在 48 寄存器的 TMA 段溢出 8 字节：`blocks` 在角色划分前计算、跨过
`setmaxnreg` 存活；改为在各角色内重算，并给 K 一半的 panel 循环加 `#pragma unroll 1` 后无溢出。

## 各 step 的性能（3 号卡，CUDA Graph，相对 cuDNN；两次运行的范围）

| 版本 | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| 41 关步骤 3（1-CTA） | ≈ 0.94–0.95× | ≈ 1.04–1.05× | ≈ 0.95× | ≈ 0.92× |
| 步骤 2（`.release.cluster`） | ≈ 0.73–0.74× | ≈ 0.79–0.80× | ≈ 0.80–0.81× | ≈ 0.71–0.75× |
| 步骤 3（`.release.cta`） | ≈ 0.90–0.92× | ≈ 0.96× | ≈ 0.87–0.88× | ≈ 0.89–0.90× |
| 步骤 3 相对 41 关的速度 | ≈ 0.96–0.99× | ≈ 0.93–0.95× | ≈ 0.93–0.94× | ≈ 0.98–0.99× |
| FA4 | ≈ 1.15–1.16× | ≈ 1.33–1.35× | ≈ 1.14–1.16× | ≈ 1.08–1.10× |

步骤 1 只算第一块、写 probe，计时没有意义。`perf_target` = cuDNN × 0.8；`selftest --perf`：perf 0.920、perf_causal 0.951、perf_d64 0.884、perf_8k 0.892（最差留约 10% 余量）。

**主循环打平，固定开销输。** 全部 CTA 平均（`globaltimer` 插桩，未提交）：`perf` 主循环 41 关 ≈ 61.4 µs、本关
≈ 61.4 µs；`perf_d64` ≈ 52.5 / 53.8 µs。setup ≈ 0.45 / 1.0 µs（推测主要是 `cta_group::2` 的 TMEM 分配协议，未单独测），
收尾 ≈ 1.1 / 1.8 µs（cluster barrier、`dealloc.cta_group::2`）。约 65 µs 的 CTA 寿命里多 ≈ 1.2 µs，与非 causal
D = 128 的 1–3% 差距（上限接近噪声）相符；persistent（43 关）只付一次。causal 另有 §3.3 的对角浪费（≈ 6% SM 时间）。

**多项式比例。** 2-CTA 下扫了 `kPolyPairs` = 2、3、4、5、6：D = 128 时 3–5 在噪声内、2 和 6 变慢；D = 64 时 4 比 3
快约 3%（`perf_d64` ≈ 0.81 → 0.785 ms），但 causal 的 D = 64 kernel 在 softmax 192 寄存器下溢出 24 字节，所以保留
41 关的 3。τ 不调（41 关由 FP16 上限推出，不是性能参数）。

**没有在 `run()` 中按配置选择。** 按实测，生产策略会对 causal 与 D = 64 选 1-CTA（41 关），非 causal D = 128 是
平手；本关需要每个 case 都验证 pair 路径，答案始终走 2-CTA。讲义 §7.4 给出按配置的选择表，与 FA4
`use_2cta_instrs` 的条件一致。

### 40 关 softmax 下的 2-CTA（讲义 §7.2 的数据）

同样的 pair 流水配 40 关的 softmax：相对 cuDNN ≈ 0.75 / 0.81–0.83 / 0.75–0.78 / 0.72–0.74（步骤 2 ≈ 0.66–0.74），
即 40 关的 ≈ 0.88–0.98×。主循环 69.5 → 78 µs（每块 ≈ 3500 → 4050 周期，时钟 ≈ 1.62 / 1.66 GHz）；NCU（锁频）
Tensor Core ≈ 57% / 51%，`smsp__inst_executed_pipe_xu` 相同，`lts__t_sectors_srcunit_tex_op_read` ≈ 1.49 / 1.01 GB
（≈ 0.68×；推导 0.53×，差异原因未查明）。每块多出的约 500 周期没有完全定位：少量 `clock64` 插桩把差距缩到约 3%
（插桩让 40 关慢了约 10%，几乎不影响 2-CTA）；假设（未验证）是 pair 把两个 SM 锁在一起，抖动推迟 ping-pong 的交错。
只测时、结果错误的实验（一半元素用一条 FMA 代替 `MUFU.EX2`）：40 关主循环 69.6 → 61.2 µs 后不再变短，2-CTA
78.2 → 56.4 → 约 55 µs（快约 9–12%）。41 关真实的 softmax 没有这么便宜（多项式在 FMA 流水线上仍要约 10 条指令
一对），所以实测只到打平。

## 验证

- `./lab check 42 --solution`：步骤 1 的 10 个 probe case、步骤 2–3 的 35 个 case 全部通过；`./lab selftest 42`、
  `selftest 42 --perf` 通过；`pytest` 中与本关有关的测试通过（`test_every_spec_parses_and_has_sources` 当前因并行
  编写中的 43 关缺 README 而失败）。
- compute-sanitizer：memcheck、synccheck、racecheck 在步骤 1 的 `odd_tiles`、`two_tiles`，步骤 2–3 的 `odd_tiles`、
  `causal_tail`、`growing_causal` 上 0 错误、0 hazard（33 关在 cta_group::2 的 TMEM alloc 上见过的 2 个误报这里没有出现）。
  racecheck 看不见 TMA、tcgen05、multicast commit 与远程 arrive，这些顺序按讲义 §5.4 的 barrier 地图与 drain 表推理。
- `./lab sass 42 --solution`：20 个 kernel 均 REG 128（`setmaxnreg` 分段）、STACK 0；有 `UTCHMMA.2CTA`、
  `UTCBAR.2CTA.MULTICAST`、`UTMALDG.4D.2CTA`、`UTCATOMSWS.2CTA`（alloc）。步骤 3 的 kernel 中 `p_ready` 的 arrive 前
  没有 `MEMBAR.ALL.GPU`（剩下的出现在 barrier 初始化与 cluster barrier）。
- 骨架：由答案生成；逐步把 TODO(1)、(1)+(2)、(1)+(2)+(3) 换回答案代码后 `--step 1/2/3` 依次通过，下一步报
  TODO(n+1)；只填 1a、1a–1b、……、1a–1e 时都在 GPU 上报 TODO(1)，步骤 2 只填一部分时报 TODO(2)，没有挂起（包括
  peer：任一 CTA 的 trap 终止整个 grid）；`kPReadyArrivals` 未设时 kernel 在 setup 后报 TODO(2) 而不是挂起。填完后
  学生版与答案同速（±2%）。验证后骨架恢复为逐字节相同。

## 已知局限

- 非 persistent 下 2-CTA 不比 41 关快（非 causal D = 128 打平到 −4%，D = 64 与 causal −5–7%），见上。
- 依赖同一线程 MMA 的按序执行（40 关）。
- 步骤 3 的 `.release.cta` 跨 CTA arrive 在 PTX 内存模型下形式上不保证正确（即使数据在 TMEM）；依赖 barrier 计数
  与当前硬件行为，与 CUTLASS/FA4 相同。形式正确的版本是步骤 2（约慢 1.2 倍）。
- causal：cluster 的 512 行一起前进，对角浪费比 40 关多；tile 0 早停不省时间。
- 非 persistent；N 很小时 peer 或 tile 1 全部越界仍参与 pair MMA。
