# 40_fa4_pipeline 参考答案说明

## 设计选择

**一个 CTA 两个 Q tile（256 行）、两个 softmax warpgroup。** 结构取自 FA4（`flash_fwd_sm100.py`，`q_stage = 2`）：
warp 0–3 是 tile 0 的 softmax，warp 4–7 是 tile 1 的 softmax，线程 r ↔ 本 tile 第 r 行 ↔ TMEM lane r。两个 tile 共用
K_j、V_j。没有 persistent、LPT（43 关）、2-CTA（42 关）、多项式 exp2 与条件 rescale（41 关）。

**TMEM 512 列正好分给两个 tile。** S⁰ 0–127、S¹ 128–255、O⁰/O¹ 从 256 起各 D 列。每个 tile 只有一块 S，P 写在 S 的后
64 列（FA4 的 `tmem_s_to_p_offset = n_block_size // 2`）。34 关的“同一 tile 的 S 双缓冲”放不下，重叠改由两个 tile
之间提供。

**MMA 发射顺序（FA4 非 `s_ping_pong` 分支）。** 序幕 QK⁰₀、QK¹₀；每块 `PV⁰_j QK⁰_{j+1} PV¹_j QK¹_{j+1}`。由此：
- softmax 等的 `s_full[t]` 第 j 阶段同时意味着 PV^t_{j−1} 完成（commit 跟踪同一线程此前全部 MMA，PTX 保证），所以
  softmax 写 P、correction 读 O 之前都不需要额外的 `o_ready`；
- QK^t_{j+1} 覆盖 PV^t_j 正在读的 P^t，**依赖同一线程的 MMA 按发射顺序执行**。PTX 8.7 只对“相同 accumulator 和
  形状”的 MMA 对保证这一点，这里不满足；FA4 与 CUTLASS 77 都这样做，本答案照做并在讲义 §4.3 明确说明。步骤 1
  （P 在 shared memory）不依赖这一点。未做“严格按规范”（QK 前等 PV commit）版本的性能对比。

**K/V 共用一个环。** item u：偶数为 K_{u/2}、奇数为 V_{u/2}，与 MMA 的消费顺序一致；K 在两个 tile 的 QK 后释放，
V 在两个 PV 后释放。P 不在 shared memory 后（步骤 2–3）D = 128 从 3 级增到 4 级、D = 64 从 6 级到 8 级。D = 128
步骤 2–3 用 5 级时 shared memory 超出 227 KB（加上 correction 的统计量）。

**correction warpgroup（步骤 3）。** softmax 求出 m′、α 后立刻经 shared memory（`Stats::alpha`）+ `stats_full`
交给 correction，再算 exp2；correction 修正 O 并 arrive `p_ready`（count 256 = softmax 128 + correction 128）。
`stats_full` 不需要 empty barrier（讲义 §5.6 的因果链）；最终 m、ℓ 用单独的 `final_full`，否则 `stats_full` 可能
连续完成两个阶段。epilogue（O/ℓ → Q^t 的 smem → TMA store、LSE）由 correction 做，与 FA4 的
`use_correction_warps_for_epi` 相同。修正是无条件的（41 关做条件 rescale）。

**setmaxnreg。** 16 个 warp（512 线程，启动时每线程 128）：softmax `inc` 192，correction `dec` 80，MMA/TMA/2 个空闲
warp `dec` 48，2·192 + 80 + 48 = 512。数值与 FA4 默认值（`num_regs_softmax = 192`、`num_regs_correction = 80`）一致。
ptxas 按 `setmaxnreg` 之后的值给各段代码分配寄存器：MMA 段只有 48 个，最初编译器把两个 tile 的 16 个 Q 描述符提到循环
外，溢出约 160 字节；MMA 的 tile 循环加 `#pragma unroll 1` 后无溢出。步骤 1–2 是 10 个 warp、无 `setmaxnreg`，
ptxas 上限 168（320 线程按 384 计）；步骤 2 写 P 时用 `x16` 的 `tcgen05.st`（`x32` 时溢出 8–16 字节）。

**MMA warp 整个 warp 跑调度、elect 一个 lane 发射。** 用 `clock64()` 插桩一个 CTA（未提交）发现：MMA 只由 lane 0 运行
（其余 31 个 lane 停在末尾的 `__syncthreads`）时，`p_ready` 已完成后还要约 250–550 周期 MMA 线程才发出 PV；改成全
warp 等待、`elect.sync` 一个 lane 发射后降到约 40 周期，`perf` 约快 6%（0.79× → 0.84×）。原因（发散 warp 的调度）
未进一步验证。TMA warp 改成“全 warp 等待 + leader 发 Q 和 K/V”的一种写法时反而慢约 10%（0.84× → 0.75×，可重复），而另一种写法
（per-warp arrive 的变体）不变；原因未查明，TMA warp 保留 lane 0 写法（它提前几级装数据，不在关键路径上）。

**MMA 描述符：一次计算、逐 K 片加偏移。** start address 字段在低 14 位（16 字节单位），shared 地址 < 256 KB 不会进位，
`desc + (bytes >> 4)` 即可，减少 MMA 线程的指令。单独测量时与逐片重算描述符的差别在噪声内（≈1–2%），为了 MMA 线程
指令短而保留。

**未采用的尝试**（同次对比，差别在噪声内）：S 的 4 次 `tcgen05.ld` 同时在路上再统一 wait；去掉 `acquire` 中的
`tcgen05.fence::after_thread_sync`；`p_ready` 改为每 warp 一个 arrive（count 4/8）；把 exp2 与写 P 分块交织。

## 各 step 的性能（相对 cuDNN，CUDA Graph，7 号卡，每项 3 次的范围）

| 版本 | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 | `perf_8k` [2,8192,16,128] BF16 |
|---|---|---|---|---|
| 34 关步骤 5（对照） | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× | — |
| 步骤 1（ping-pong，P 经 smem） | ≈ 0.77–0.78× | ≈ 0.88–0.89× | ≈ 0.80× | ≈ 0.75–0.76× |
| 步骤 2（P 在 TMEM，TS MMA） | ≈ 0.80× | ≈ 0.89–0.90× | ≈ 0.73–0.75× | ≈ 0.77–0.78× |
| 步骤 3（correction WG + setmaxnreg） | ≈ 0.84× | ≈ 0.93–0.94× | ≈ 0.79–0.81× | ≈ 0.82× |
| FA4（`flash_attn.cute`） | ≈ 1.15× | ≈ 1.33× | ≈ 1.13× | ≈ 1.10× |

插桩时间线（`perf`，稳态，SM 周期）：一块的周期 T ≈ 3450（MMA 只用 lane 0 时 ≈ 3700）；softmax WG 从拿到 S 到 arrive
≈ 1900–2200，其中 128 个 exp2 ≈ 1450–1700；一个 tile 的 PV + QK ≈ 1100。T ≈ softmax + PV + QK + 交接，与讲义 §3.1 的
链模型一致。MUFU 下限是每 tile 每块 1024 周期，所以 41 关的多项式 exp2 是下一步。

## 验证

- `./lab check 40 --solution`：3 个 step × 25 个 case 全部通过；`./lab selftest 40` 与 `selftest 40 --perf`（目标 0.72）通过；
- `./lab sanitize --solution` 的 memcheck、racecheck、synccheck 在 `two_tiles`、`causal_tail` 上（全部 step）0 错误；
  racecheck 看不见 TMA/TMEM 访问，同步论证见讲义 §5.6 的表；
- `./lab sass 40 --solution`：24 个 kernel STACK 均为 0；步骤 1–2 每线程 164–168 个寄存器，步骤 3 启动值 128（各角色由
  `setmaxnreg` 决定，ptxas 按段分配，无溢出）；
- 骨架：由答案生成，逐步把第 1..n 步的 TODO 换回答案代码后 `--step n` 通过、`--step n+1` 报 TODO(n+1)；只填 1a、
  或填到 3a 时也在 TODO 处停下而不挂起；验证后恢复为原骨架（字节一致）。
- `perf_target` 取 0.72 而不是 0.75：最差的 `perf_d64` 在 ≈ 0.79–0.81×，cuDNN 自身有约 4% 的波动。

## 已知局限

- 依赖同一线程 MMA 的按序执行（上文）。
- causal 时两个 tile 走同样多的块，tile 0 每对多算一个全被 mask 的块（FA4 相同）；Q tile 按 `blockIdx.x` 顺序发射（43 关）。
- N 很小时 tile 1 全部越界仍然被计算（结果不写回）。
- 非 persistent；每个 CTA 重新初始化 barrier、分配 TMEM、装 Q。
- 只支持 D ∈ {64, 128}、FP16/BF16，O 与输入同 dtype。
