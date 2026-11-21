# 33_2cta_gemm 答案说明

## 设计选择

- **支持域**：与 31、32 关相同。A、B 为 BF16，C 为 FP32；`K % 8 == 0`、`N % 4 == 0`、`K > 0`（K = 0 时会在没有任何 MMA 的情况下 commit `acc_full`，读到未初始化的 TMEM），M 任意；尾部全部交给 TMA。三个 step 都覆盖全部形状，所以每个 step 跑全部 correctness case。
- **原语**：新增 `include/lab/cluster.cuh`（学生在本关骨架里实现，34 关起提供）：cluster 身份（`%cluster_ctarank`、`%clusterid.x`、`%nclusterid.x`）、`barrier.cluster`、`mapa`、cluster 作用域的 mbarrier arrive / wait、`.cta_group::2` 的 TMA load（含 multicast）、TMEM alloc/relinquish/dealloc、`tcgen05.mma`、multicast `tcgen05.commit`，以及 host 端 `launch_cluster`（`cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`）和 `max_active_clusters`（`cudaOccupancyMaxActiveClusters`）。没有修改 `tcgen05.cuh` / `tma.cuh`：kernel 内所有 tcgen05 指令必须用同一个 `.cta_group`，所以 pair 版本的 alloc/mma/commit 都放在 `cluster.cuh`，`tcgen05.ld`、fence、descriptor 仍来自 `tcgen05.cuh`。
- **pair 布局**：cluster 的 rank 0 是 leader，rank 1 是 peer。CTA r 用 TMA 读 A 的第 `m0 + 128r` 起 128 行和 B 的第 `n0 + 128r` 起 128 行（B box 128 行），放在相同的 shared memory 偏移。leader 的一个线程发 `tcgen05.mma.cta_group::2`，idesc M = 256、N = 256，descriptor 只指向 leader 自己的 buffer，硬件在 peer 的同一偏移读另一半。累加器在两个 CTA 的 TMEM 中各 128 lane × 256 列，CTA r 的 epilogue 写 C 的第 `m0 + 128r` 起 128 行、全部 256 列。
- **barrier**（同一偏移在两个 CTA 里各有一份）：
  - `full[s]`（count 1）：只用 leader 的。leader 的 producer `arrive_expect_tx(64 KB)`（两个 CTA 的字节），两个 CTA 的 TMA 都用 `mapa` 得到的 leader 地址 `complete_tx`。peer 的 TMA 可能先于 leader 的 expect_tx 到达，tx-count 暂时为负，这是允许的；相位要等 leader 的 1 次到达才能完成。
  - `empty[s]`（count = 每 cluster 的 pair 数）：leader 的 MMA 线程 `tcgen05.commit.cta_group::2 ... multicast` 到 cluster 内所有 CTA；每个 CTA 的 producer 只等自己的那份。
  - `acc_full[a]`（count 1）：leader commit multicast 到本 pair 的两个 CTA；各自的 epilogue 等自己的那份。
  - `acc_empty[a]`（count 256）：只用 leader 的。两个 CTA 的 128 个 epilogue 线程都在最后一次 `tcgen05.ld` 后 `fence::before_thread_sync` + `mbarrier.arrive.release.cluster`（peer 经 `mapa` 地址）；leader 用 `try_wait.acquire.cluster` 等待，再 `fence::after_thread_sync`。
- **启动与退出**（维护者 review 指出的问题，本关必须处理）：
  - setup：thread 0 初始化 barrier + `fence.mbarrier_init.release.cluster`；两个 CTA 的 warp 5 都执行 `tcgen05.alloc.cta_group::2`（512 列，两边得到同一个列地址）；`fence::before` + `barrier.cluster.arrive.release/wait.acquire` + `fence::after`。`__syncthreads` 不够：peer 的 TMA、leader 的 commit 会写对方的 barrier，必须在对方初始化之后。
  - 排空：两个 CTA 的 producer 在最后一个 tile 之后再“获取” kStages 个槽（等 `empty`，不装数据），保证 leader 发往本 CTA 的所有 multicast commit 都已到达；leader 的 MMA 线程同样再等 kAccStages 次 `acc_empty`，保证 peer 的远程 arrive 都已到达。
  - teardown：`fence::before` + cluster barrier + `fence::after`，然后各自 `tcgen05.dealloc.cta_group::2`。此时 leader 的 MMA 已全部完成（两边 epilogue 都等过最后的 `acc_full`），任何 CTA 都不再被另一方访问 smem / TMEM。
- **资源**：每 CTA 每级 32 KB（A 16 KB + B 16 KB，32 关是 48 KB），6 级 192 KB + 32 KB staging + 1 KB 对齐 = 225 KB；TMEM 2 × 256 列 = 512 列。寄存器 56–72，无 spill。
- **step 1**（`gemm_pair_tile`）：grid = 2 × tile 数，cluster 2，每 cluster 一个 256×256 tile、单累加器。**step 2**（`gemm_pair_persistent<Config<6,2,8>>`）：grid = 2 × min(tile 数, 可驻留 cluster 数)，按 `%clusterid.x` / `%nclusterid.x` 静态轮转，累加器双缓冲，分组光栅化 G = 8（单位是 256 行的 tile 行）。**step 3**：cluster 4 = 沿 N 并排的两个 pair（cluster tile 256×512），A 的每一半由两个 pair 中同位置的两个 CTA 各读 64 行并 multicast 给对方（mask = bit r | bit r+2），`empty` count = 2（两个 leader 的 commit），`acc_full` 只 multicast 给本 pair。host 端按“轮数”在 cluster 4 与 cluster 2 之间选择。
- **multicast 的 barrier 地址**：`.cta_group::2` + `.multicast::cluster` 的 TMA 只给一个 barrier 操作数（本 CTA 的 leader 的 `mapa` 地址），每个目标 CTA 的那份在“目标 CTA 所在 pair 的 leader”的同一偏移上 complete_tx。这一点由 `persistent` 等 case 实测确认（若信号落到错误的 leader，一个 leader 会多收字节、另一个永远等不到，即 `TIMEOUT`），与 CUTLASS `SM100_TMA_2SM_LOAD_MULTICAST` 清除 peer bit 的做法一致。

## 各 step 的性能（2 号卡，CUDA Graph 中位数，相对 cuBLAS BF16→FP32）

| 版本 | square 4096³ | wide 8192×8192×4096 | flat 8192×8192×1024 |
|---|---|---|---|
| 32 关答案（128×256，单 CTA） | ≈ 0.91× | ≈ 0.91× | ≈ 0.99× |
| step 1：pair tile，非 persistent | ≈ 0.91× | ≈ 0.86× | ≈ 0.73× |
| step 2：persistent pair + 双缓冲 + G = 8 | ≈ 0.98–0.99× | ≈ 1.12× | ≈ 1.09× |
| step 3：按轮数选择（square 走 cluster 4 multicast，wide/flat 走 cluster 2） | ≈ 1.00× | ≈ 1.16× | ≈ 1.03–1.05× |
| 对照：强制 cluster 4 multicast | ≈ 1.00× | 比 step 2 慢 ≈ 8% | 比 step 2 慢 ≈ 5% |
| step 3，`--cold` | ≈ 1.01× | ≈ 1.05× | ≈ 1.10× |

- cuBLAS 在 `wide` 上本机波动较大（同一卡多次测得 0.39–0.43 ms），比值随之在 1.0–1.16 之间变化；本答案自身的时间很稳定（step 2/3 约为 32 关答案的 0.83 倍）。
- `selftest --perf`（门槛 0.9）：square 1.00、wide 1.16、flat 1.05。最差的 square 留约 10% 余量。
- NCU（锁频，只看比例），square：`sm__pipe_tensor_cycles_active` 32 关 ≈ 84.5%，step 1 ≈ 75.7%，step 2 ≈ 87.7%，step 3 ≈ 88.7%。`lts__t_sectors_srcunit_tex_op_read`：step 2 ≈ 0.99 GB，step 3 ≈ 0.82 GB；推导值 32 关 1.61 GB、pair 1.07 GB、pair + A multicast 0.81 GB。锁频下 step 3 反而比 step 2 慢约 2%，不锁频时快约 1.5%：差别主要来自轮数与功耗/频率，不是 L2 带宽。
- **为什么 step 3 要在 host 上选形状**：`cudaOccupancyMaxActiveClusters` 给出 B200 上 cluster 2 可驻留 74 个（148 SM 全用），cluster 4 只有 33 个（132 SM）：cluster 必须放在一个 GPC 内，GPC 的 SM 数不是 4 的倍数。两种形状下每个 pair 每轮都算一个 256×256 tile，所以时间 ≈ 轮数 × 每轮时间。square：256 个 pair tile / 74 = 3.46 → 4 轮；128 个 cluster tile / 33 = 3.88 → 4 轮，轮数相同，multicast 少读 A 而稍快。wide/flat：1024 / 74 → 14 轮，512 / 33 → 16 轮，cluster 4 慢约 8%（实测与 16/14 的 14% 有差，multicast 抵消了一部分）。CUTLASS 的解法是 preferred cluster（`cudaLaunchAttributePreferredClusterDimension`，examples/73），让放不下 4 的 GPC 用 2，需要动态调度，不在本课范围。
- step 1 比 32 关的 step 1 在 wide/flat 上差：每个 cluster 只算一个 tile，2048/1024 个 cluster 反复启动、建流水、TMEM alloc，epilogue 不重叠（flat 的 epilogue 占比最大）。persistent 之后 pair 的优势才显出来。

## 验证

- `./lab check 33 --solution`：3 个 step × 9 个 case 通过（`int` case 精确相等，`random` 最大误差 1.8e-4）。step 3 的全部 correctness case 都走 cluster 4（轮数都不多于 cluster 2），`persistent` case 让部分 cluster 处理 3 个 tile；step 2 的 `persistent` 覆盖 cluster 2 的 persistent 路径。
- compute-sanitizer：
  - memcheck、synccheck：3 个 step 在 `tail` 上、step 2/3 在 `persistent` 上均 0 错误。
  - racecheck：每次运行报告 2 个 hazard，全部是“PC 未知的写”对“`tmem_alloc_cg2` 那一行的读”。查 SASS：`tcgen05.alloc.cta_group::2` 被编译器展开成一段软件协议（`UTCATOMSWS.2CTA.FIND_AND_SET`、在保留的 shared memory 0x40–0x7f 上的 `SYNCS` mbarrier、向 peer 的 `STAS` 远程写），racecheck 不理解跨 CTA 的 mbarrier 同步。一个只做 alloc/dealloc 的最小 kernel（148 个 CTA、cluster 2、warp 5 alloc）同样报告这个 hazard，与本关代码无关；cta_group::1 的 32 关没有。除此之外 racecheck 无报告。racecheck 本身不跟踪 TMA、tcgen05、multicast 与 DSMEM 远程 arrive（异步代理/远程访问），这些顺序按下表推理：

| 访问 | 先 | 后 | 保证 |
|---|---|---|---|
| barrier 初始化 → 另一个 CTA 或异步代理使用 | `mbarrier.init` | peer 的 TMA complete_tx、leader 的 commit、远程 arrive | `fence.mbarrier_init.release.cluster` + `barrier.cluster` |
| TMA 写 leader / peer 的 stage s → pair MMA 读 | 两边 `complete_tx` | MMA 发出 | leader 的 `full[s]`（64 KB）wait + `fence::after` |
| MMA 读两边 stage s → 两边 TMA 重写 | MMA 完成 | 下一轮 TMA | multicast commit → 各 CTA 的 `empty[s]`，各自 producer wait |
| MMA 写两边累加器 a → 各自 epilogue 读 | MMA 完成 | `tcgen05.ld` | multicast commit → 各 CTA 的 `acc_full[a]`，wait + `fence::after` |
| 两边 epilogue 读累加器 a → MMA 重写 | `wait::ld` | 下一 tile 的 MMA | `fence::before` + 256 次 `arrive.release.cluster` → leader `acc_empty[a]`，`try_wait.acquire.cluster` + `fence::after` |
| （step 3）另一 pair 的 multicast 写本 CTA 的 A → 重写 | 两个 pair 的 MMA 读完 | 下一轮 multicast | `empty[s]` count 2：两个 leader 的 commit |
| leader commit → 本 CTA 的 `empty` / `acc_full` | 到达 | CTA 退出 | producer 排空 kStages 次；epilogue 等过每个 `acc_full` |
| peer 远程 arrive → leader 的 `acc_empty` | 到达 | leader 退出 | leader MMA 排空 kAccStages 次 |
| 所有 TMEM 访问 → dealloc | MMA 完成、`wait::ld` | `dealloc.cta_group::2` | `fence::before` + `barrier.cluster` + `fence::after` |
| generic 写 staging → TMA store；store 读完 → 重写 / 退出 | | | 与 32 关相同 |

- `./lab sass 33 --solution`：寄存器 56 / 72 / 72，无 spill、无 local memory；`UTCHMMA.2CTA`、`UTCBAR.2CTA.MULTICAST`、`UTMALDG.2D.2CTA`、`UTMALDG.2D.MULTICAST.2CTA`（step 3）、`UTCATOMSWS.2CTA`（alloc）都在。
- 骨架：按 TODO(1)、(2)、(3) 逐组把答案代码填入原位，`--step n` 依次通过，第 n 步只依赖前 n 组 TODO；填完后学生版与答案同速；骨架已恢复为逐字节相同。未填时：step 1/2 在 host 的 `launch_cluster`（TODO(1)）处报错；只实现 launch 时，两个 CTA 都在 setup 的 TODO(1) 处 trap（`tmem_alloc_cg2` / `cluster_arrive`，任何等待之前）；step 1 完成后 step 2 在 leader MMA 的 TODO(2) 处 trap（GPU 上报告，不挂起）；step 3 在 host 的 cluster 形状选择 TODO(3) 处报错。
- `./lab selftest 33 --perf` 通过；`pytest` 通过（写完讲义后）。

## 已知局限

- 静态 round-robin 调度；square 的 3.46 波尾部没有处理。cluster 4 只能在轮数不吃亏时使用；没有实现 preferred cluster / 动态调度（CLC）。
- `kGroupM = 8` 沿用 32 关的结论（单位换成 256 行的 tile 行），没有对 pair 重新扫描。
- multicast 只用于 A（沿 N 并排的两个 pair）；沿 M 并排共享 B 的形状对称，未实现。
- 只有 BF16 → FP32。
