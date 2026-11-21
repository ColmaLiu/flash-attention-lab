# 34_fa3 参考答案说明

## 设计选择

**一个 (Q tile, head, batch) 一个 CTA，6 个 warp 固定分工。** warp 0–3 是 softmax warpgroup（线程 r 拥有
行 q0 + r = TMEM lane r），warp 4 的 lane 0 发 TMA，warp 5 的 lane 0 发全部 `tcgen05.mma`。BM = BN = 128：
M = 128 是单 CTA `tcgen05.mma` 一个 lane 一行的形状；BN = 128 让 S 正好 128 列、P·V 的 K = 128。
没有 persistent、没有 LPT 发射顺序（43 关），也没有 2-CTA（42 关）。

**S、O 在 TMEM，P 经 shared memory（SS MMA）。** 这是 FA3 思想在 SM100 上最直接的映射：Hopper 上 P 在寄存器
作 `wgmma` 的 A 操作数（RS），SM100 的 `tcgen05.mma` 的 A 只能来自 shared memory 或 TMEM。P 写回 TMEM
（TS MMA）属于 FA4，留给 40 关。P 的写法与 TMA 写 Q 的 128B swizzle 一致，16 字节 store 无冲突。

**V 是 MN-major 的 B 操作数。** idesc 第 16 位置 1，描述符 SBO = 1024（8 个 key 行）、LBO = 一个 64 列 d panel
（128 行 × 128 B = 16 KB），K 片前进 16 行。布局依据 CUTLASS `make_umma_desc<Major::MN>`（`mma_traits_sm100.hpp`）
的规范布局；v1（main 分支 `s40_attention`）也是同样的编码。

**O 的修正由 softmax warp 同步完成，每块都做。** 块 j > 0 时，softmax 线程等 `o_ready(j−1)`，
`tcgen05.ld` → 乘 α → `tcgen05.st`，`wait::st` 后才 arrive `p_ready`。独立 correction warpgroup 与条件 rescale
是 40–41 关的内容，这里不做。

**K、V 分开的 full/empty barrier，同一个 stage 下标。** MMA warp 先要 K_j（QKᵀ），后要 V_j（PV），释放时间也不同。
D = 128 用 2 级、D = 64 用 3 级，两者都让每 SM 只放一个 CTA（192 KB / 144 KB）：步骤 5 用满 512 列 TMEM，
第二个 CTA 会在 `tcgen05.alloc` 处空等并占着 shared memory（`Config` 中有 static_assert）。

**S 双缓冲（步骤 5）。** MMA 的顺序是 QK_0、QK_1，然后每块“等 p_ready(j) → PV_j → QK_{j+2}”。PV_j 先发，
softmax 写 P_{j+1} 前等的 `o_ready(j)` 不被 QK_{j+2} 推迟。softmax 先算完指数，再等 `o_ready(j−1)`，指数计算与
PV_{j−1} 重叠。步骤 1–4 是同一个 kernel，`kSBuf = 1`。

**O 的写回用 TMA store。** 最后一个 `o_ready` 之后所有 MMA（含读 Q 的 QKᵀ）都已完成，Q 的 shared memory 作
暂存区；每个 softmax warp 用 box `[32][64]` 的 4-D map 写自己的 32 行，越过 N 的行由 TMA 跳过，不需要任何判断。
LSE 每线程一个 float，直接写，相邻线程地址相邻。

**4-D tensor map `{D, H, N, B}`。** 不能把 B 和 N 合并：尾块会读到下一个 batch，O 的 store 会写坏下一个 batch。

**退出前的排空。** TMA warp 在主循环后等每个 stage 最后一次 `k_empty/v_empty`，保证所有 `tcgen05.commit` 的
arrive 都已落地后 CTA 才退出（CUTLASS 的 producer tail 同理）。

**TMEM 读取的排序。** 维护者指出头文件中 `tmem_ld_*` 与 `tmem_wait_ld()` 之间没有数据依赖，编译器可以把对
结果寄存器的计算调度到 wait 之前。答案全部使用头文件的 `tmem_ld_32x32b_x32_sync`（ld + wait 同一条 asm）。
对 S 的一行（4 次 x32）也试过“4 次 ld 同时在路上 + 一个把 32 个寄存器作为 `+f` 操作数的 wait”的本地 helper：
同一次 bench 中 `perf` 约快 3%、`perf_d64` 约快 1%。差别小，为了讲义简单没有采用。

**数值。** 与 22 关相同：m 存原始分数（scale > 0），`ex2.approx.ftz` 的本地封装，`m' = −∞` 时以 0 作减数。
max 与 sum 各用 8 个独立累加器：每个 SMSP 只有一个 softmax warp，128 长的依赖链会暴露延迟。

**新增的公共原语。** 本关加入头文件（纯新增，未改已有签名）：`include/lab/tma.cuh` 的
`make_tensor_map_4d`、`tma_load_4d`、`tma_store_4d`；`include/lab/tcgen05.cuh` 的
`make_smem_desc_mnmajor_sw128(smem, lbo_bytes, sbo_bytes = 1024)`。本地只剩 `exp2_approx`、`pack2<T>`（与 22 关相同）。

**S 双缓冲的协议集中在两个函数里。** `s_buffer<kSBuf>(i)` 与 `s_phase<kSBuf>(i)` 同时被 MMA warp（写哪个 buffer、
commit 哪个 barrier）和 softmax warp（读哪个 buffer、等哪个 parity）使用；单 buffer 分支在骨架中给出，两 buffer
分支与 MMA 的提前发射（QK_1、QK_{j+2}）是 TODO(5)。

## 各 step 的性能（相对 cuDNN，CUDA Graph，6 号卡）

| step | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 1、2（单块，`one_block`，只用于验证） | — | — | — |
| 3（非 causal） | ≈ 0.41× | — | ≈ 0.41× |
| 4（单 S buffer） | ≈ 0.41× | ≈ 0.51× | ≈ 0.41× |
| 5（S 双缓冲） | ≈ 0.54–0.59× | ≈ 0.70× | ≈ 0.57–0.59× |
| 22 关步骤 4（对照） | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4 / cuDNN | ≈ 1.07–1.13× | ≈ 1.28–1.34× | ≈ 1.13–1.16× |

`--cold`：步骤 5 `perf` ≈ 0.65×（Q/K/V 约 200 MB，本来就大于 L2；cuDNN 在冷 L2 下相对慢一些）。
`perf_target` 设为 cuDNN 的 0.4×，最差的 `perf` 有约 35% 余量。

Nsight Compute（`perf`）：`sm__pipe_tensor_cycles_active` 步骤 3 ≈ 27%，步骤 5 ≈ 38%；
`smsp__inst_executed_pipe_xu` ≈ 3383 万（≈ B·H·N²/32，每个分数一条 MUFU.EX2）；`dram__bytes_read` ≈ 201 MB。
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` 步骤 3 约 95 万、步骤 5 约 1131 万，而 P 与 O 暂存的写入
地址在两个步骤中完全相同。推测（未验证）：这个计数包含了 store 与并发的 async-proxy 读（MMA 读操作数、TMA 写入）
争用 shared memory 的周期，步骤 5 中 MMA 与 softmax 同时进行，所以更多。没有进一步拆分。

寄存器：softmax 路径决定，每线程 164 个（步骤 1 的 probe kernel 140 个），无 spill、无 stack。

## 已知局限

- 瓶颈是 softmax warpgroup：每块 128×128 次 MUFU.EX2、逐元素 FFMA/FMNMX/FADD、写 32 KB 的 P、修正 O 都在一个
  warpgroup 里串行，Tensor Core 约 60% 时间空闲。FA4 的办法（2 个 Q tile ping-pong、correction warpgroup、
  P 在 TMEM、多项式 exp2、条件 rescale）留给 40–41 关。
- 每块都无条件修正 O（D = 128 时每线程每块 128 次 ld/乘/st）。
- causal 的 Q tile 按 `blockIdx.x` 顺序发射，最重的 tile 最后开始。
- 非 persistent：每个 CTA 都重新初始化 barrier、分配 TMEM、装 Q。
- 只支持 D ∈ {64, 128}、FP16/BF16，O 与输入同 dtype；步骤 1–2 只支持 N ≤ 128（`make_plan` 检查）。
