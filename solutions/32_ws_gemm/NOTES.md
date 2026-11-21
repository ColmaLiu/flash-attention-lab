# 32_ws_gemm 答案说明

## 设计选择

- **支持域**：与 31 关相同。A、B 为 BF16，C 为 FP32；`K % 8 == 0`、`N % 4 == 0`，M 任意；尾部全部交给 TMA（load 填零，store 裁剪）。三个 step 都覆盖全部形状，所以每个 step 跑全部 correctness case。
- **原语**：只用 `include/lab/{mbarrier,tma,tcgen05}.cuh`。epilogue 用 `tmem_ld_32x32b_x32_sync`（ld 与 `wait::ld` 在同一个 asm 块里，寄存器在返回时就有效）。没有在本地新增 PTX 封装。
- **角色**：6 个 warp（192 线程）。warp 0–3 做 epilogue：warp w 只能访问 TMEM lane `32(w%4)..`，取 w = 0..3，与 31 关的“warp w ↔ C 的第 32w 行起”一致。warp 4 是 TMA producer，warp 5 是 MMA issuer，同时负责 TMEM alloc/dealloc（与 CUTLASS 一样，由 MMA warp 持有 TMEM）。producer 和 MMA 各用 `elect_one()` 选一个 lane 跑整个循环，其余 lane 在 `__syncwarp` 等待。CUTLASS（`sm100_gemm_tma_warpspecialized.hpp`）多一个 scheduler warp 和一个 epilogue-load warp；静态调度不需要前者，C 只写不读不需要后者。
- **barrier**：`full[s]`（1 次到达 + tx）、`empty[s]`（1 次，tcgen05.commit）、`acc_full[a]`（1 次，tcgen05.commit）、`acc_empty[a]`（128 次：每个 epilogue 线程在自己最后一次 `tcgen05.ld` 完成后 `fence::before_thread_sync` + arrive；与 CUTLASS `consumer_arv_count = NumEpilogueThreads` 相同）。生产者一侧（producer 等 `empty`、MMA 等 `acc_empty`）首轮用 `phase ^ 1`，利用“新初始化 barrier 的前一相位视为已完成”。
- **相位跨 tile**：每个角色各持一份 `RingPos`（index、phase），每用一次 advance 一次，跨 tile 不清零。三个角色按同一个静态 tile 序列走，用同样多次，因此无需通信就保持一致。
- **未等待的 `empty` 到达**：CTA 最后一个 tile 的最后几个 K 块提交到 `empty[s]` 的 commit 没有人等待。单 CTA 时无害（只到达本 CTA 的 barrier，MMA 已由 `acc_full` 确认完成）；33 关用 cluster 时，CUTLASS 用 `producer_tail` 把它们排空，再让 CTA 退出。
- **`K = 0`**：make_plan 拒绝（否则 MMA warp 直接 commit `acc_full`，epilogue 写出未初始化的 TMEM）。
- **barrier 存放**：`Barriers<Cfg>`（各 barrier + TMEM 地址槽）是每个 kernel 里显式声明的静态 `__shared__` 变量，`carve()` 把动态 shared memory 切成 stage 与 staging；不用宏。
- **早释放累加器**：epilogue 在读完 TMEM 后立即到达 `acc_empty`，此时 TMA store 仍在进行。安全，因为 store 读的是 staging（shared memory），不是 TMEM。
- **epilogue staging**：persistent 时主循环同时在用 stage buffer，不能像 31 关那样复用。每个 epilogue warp 两个 32×32 FP32 box（4 KB，128B swizzle），共 32 KB；每个 box 一个 bulk group，重写 box 前由 lane 0 `bulk_wait_group_read<1>`（两次之前的 store 已读完），再 `__syncwarp`。
- **资源**：BN = 256、4 级 stage（192 KB）+ 32 KB staging + 1 KB 对齐 = 225 KB 动态 shared memory，加约 1.1 KB 静态，在 227 KB 上限之内；TMEM 2 × 256 = 512 列，整块 TMEM。每 SM 一个 CTA，所以 512 列的 alloc 不会与本 kernel 的其他 CTA 竞争。5 级或 3 个累加器都放不下。
- **step 1** 用独立的 `gemm_ws_tile`（grid = 全部 tile，单累加器，与 31 关 step 3 同 tile/同 stage 数），只改变“谁发 TMA、谁发 MMA、谁做 epilogue”。**step 2/3** 用 `gemm_ws_persistent`，grid = min(tile 数, SM 数)，tile `blockIdx.x + i·gridDim.x`。step 3 只改 tile 顺序（分组光栅化，`kGroupM = 8`）。
- **没有用 `setmaxnreg`**：寄存器 56–72、无 spill；每 SM 只有 6 个 warp，寄存器文件远没用满，重分配没有收益。它留到 FA4（40 关），那里 softmax warpgroup 需要大量寄存器。

## 各 step 的性能（5 号卡，CUDA Graph，相对 cuBLAS BF16→FP32）

| 版本 | square 4096³ | wide 8192×8192×4096 | flat 8192×8192×1024 |
|---|---|---|---|
| 31 关答案（单线程 TMA+MMA，非 persistent） | ≈ 0.84× | ≈ 0.75× | — |
| step 1：角色分工，一个 CTA 一个 tile | ≈ 0.89× | ≈ 0.80× | ≈ 0.78× |
| 对照：persistent，单累加器（`Config<256,4,1,0>`） | ≈ 0.92× | ≈ 0.85× | ≈ 0.88–0.9× |
| step 2：persistent + 累加器双缓冲 | ≈ 0.93× | ≈ 0.86–0.88× | ≈ 0.95–0.98× |
| step 3：+ 分组光栅化 G = 8 | ≈ 0.92× | ≈ 0.91× | ≈ 0.95× |

分组大小（step 3，绝对时间相对 step 2 的变化，3 次以内重复）：

| G | square | wide | flat |
|---|---|---|---|
| 0（行优先 = step 2） | 1 | 1 | 1 |
| 8 | −1.5% | +6% | ≈ −1–2% |
| 16 | −5% | +8% | −4% |
| 32 | −3% | +10% | −4% |

- NCU（时钟被锁定，只看比例）：`sm__pipe_tensor_cycles_active` square step 1 ≈ 80%，step 2/3 ≈ 85%（31 关 step 3 ≈ 68%）。wide 的 `dram__bytes_read`：行优先 ≈ 981 MB，G = 8 ≈ 507 MB，G = 32 ≈ 280 MB（A + B 本身 128 MB）。
- 累加器双缓冲本身在 square 上只值约 1%，在 flat 上约 10%：K = 4096 时一个 tile 的主循环约为 epilogue 的 6–8 倍，而且单累加器也是“读完 TMEM 就释放”，MMA 只等 TMEM→寄存器那一段；K = 1024 时 epilogue 与主循环同量级，重叠才显出价值。为此增加了 `flat` perf case。
- G > 0 在 A、B 都能放进 L2 的 case 上稍慢，原因未查明（推断与 B200 两个 die 各自的 L2 分区或同一 panel 被更多 CTA 同时读取有关，未验证）。G = 8 是三个 perf case 最差比值最高的选择。
- `--cold`（每次迭代前清 L2）step 3：square ≈ 0.91×、wide ≈ 0.87×、flat ≈ 1.0×。
- `selftest --perf` 门槛 0.85：实测最差约 0.91（square、wide），留约 6% 余量。本机 GPU 偶有整体降速时段（cuBLAS 同样变慢），比值基本不变。
- 剩余差距（推断）：每个 128×256 tile 每 K 块从 L2 读 48 KB，148 个 SM 合计接近 L2 带宽；square 512 个 tile = 3.46 波，最后一波 46% 的 SM 空闲。前者由 33 关的 2-CTA MMA 与 TMA multicast 处理。

## 验证

- `./lab check 32 --solution`：3 个 step × 10 个 case 通过（`int` case 精确相等，`random` 最大误差 1.9e-4）。`persistent` case（372 个 tile）让部分 CTA 处理 3 个 tile，累加器 0→1→0 轮换，每 tile 6 个 K 块使环形 buffer 位置跨 tile 错开，最后一组只有 7 行 tile。
- compute-sanitizer memcheck / racecheck / synccheck：`tail` 上 3 个 step 均 0 错误；step 3 `persistent` 上三种工具均 0 错误。racecheck 不跟踪 TMA 与 tcgen05（async proxy）访问，所以 0 错误只覆盖 generic proxy（staging 写入）。async 部分的顺序按下表推理：

| 访问 | 先 | 后 | 保证 |
|---|---|---|---|
| TMA 写 stage s → MMA 读 | `complete_tx` | MMA 发出 | `full[s]` wait + `fence::after_thread_sync` |
| MMA 读 stage s → TMA 重写 | MMA 完成 | 下一轮 TMA | `tcgen05.commit → empty[s]`，producer wait |
| MMA 写累加器 a → epilogue 读 | MMA 完成 | `tcgen05.ld` | `commit → acc_full[a]`，wait + `fence::after` |
| epilogue 读累加器 a → MMA 重写 | `wait::ld` | 下一 tile 的 MMA | `fence::before` + 128 次 arrive `acc_empty[a]`，MMA wait + `fence::after` |
| generic 写 staging → TMA store 读 | `st.shared` | `cp.async.bulk.tensor` | `fence.proxy.async` + `__syncwarp` |
| TMA store 读 staging → 重写 | store 读完 | 下一次写 box | lane 0 `wait_group.read<1>` + `__syncwarp` |
| 所有 TMEM 访问 → dealloc | MMA 完成（epilogue 已等 `acc_full`）、`wait::ld` | dealloc | `fence::before` + `__syncthreads` + `fence::after` |
| staging 生命周期 | store 读完 | CTA 退出 | lane 0 `wait_group.read<0>` |

- `./lab sass 32 --solution`：56 / 70 / 72 寄存器，无 spill、无 local memory；有 `UTCHMMA`、`UTCBAR`、`UTMALDG`、`UTMASTG`、`LDTM`。
- 骨架：在原位按 TODO(1)、(2)、(3) 逐步填入答案代码，`--step n` 依次通过，第 n 步只依赖前 n 组 TODO；填完后学生版与答案同速；骨架已恢复为逐字节相同。未填时 step 1 在 producer（TODO(1)）、step 2/3 在 persistent producer（TODO(2)）处 trap，不会死锁：每个角色在任何 mbarrier 等待之前先到达自己的 TODO。

## 已知局限

- 静态 round-robin 调度：square 的波次尾部（3.46 波）没有处理（stream-K / split-K 不在本课程范围）。
- `kGroupM` 是编译期常量，没有按形状自动选择。
- epilogue 每个 32 列块 ld 后立即等待（`_sync`），没有把下一块的 ld 与本块的 smem 写重叠；epilogue 已与主循环重叠，测得不是瓶颈。
- 只有 BF16 → FP32。
