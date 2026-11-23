# 31_tcgen05_gemm 答案说明

## 设计选择

- **支持域**：A、B 为 BF16，C 为 FP32；`K % 8 == 0`、`N % 4 == 0`（A、B、C 的 TMA 行距须为 16 字节的倍数），M 任意。尾部全部交给 TMA：load 越界填零（行与 K 都是），store 越界跳过。步骤 1 另外要求 `M ≤ 128, N ≤ 128, K ≤ 64`（make_plan 检查）。
- **原语**：`include/lab/tcgen05.cuh`（新）+ 30 关的 `mbarrier.cuh`、`tma.cuh`。位域按 CUTLASS `mma_sm100_desc.hpp` 的 `SmemDescriptor` / `InstrDescriptor` 核对：描述符 version = 1（bits 46–47）、layout 2 = SWIZZLE_128B（bits 61–63）、SBO = 1024；idesc 中 M>>4 在 bits 24–28、N>>3 在 bits 17–22。v1（main 分支 s31）的 B 是 MN-major；v2 的 B 是 `[N,K]` K-major，A、B 用同一个描述符函数，`b_major = 0`。
- **线程组织**：128 个线程 = 4 个 warp = TMEM 的 4 个 lane 段。thread 0 同时发 TMA 和 MMA（不做 warp specialization，那是 32 关的内容）；其余线程在 `mma_done` 上等待，然后全体做 epilogue。
- **step 1**：单 tile 128×128×64：一次 expect_tx（32 KB）、两次 TMA、4 条 MMA、`tcgen05.commit` → `mma_done`；epilogue 每线程一行（`tcgen05.ld.32x32b.x32` + `wait::ld`），直接逐元素写 global。
- **step 2**：`gemm_pipelined<Config<128, 4, false>>`。prologue 装满 S 个 stage；循环中先发 MMA t 并 commit 到 `empty[t%S]`，再等 MMA t−1 的 `empty` 并重装它的 stage（tile t−1+S），这样等待时 Tensor Core 队列里始终有 MMA t。step 2 的 grid 已经覆盖全部 M×N tile，所以它运行全部 correctness case；与任务说明“step 2 只做单 CTA tile”略有不同：多 CTA 只是一行 `blockIdx` 映射，放在 step 3 没有教学价值，step 3 改为专注于 epilogue 与 tile 形状。
- **step 3**：`Config<256, 4, true>`：192 KB shared memory、256 列 TMEM、每 SM 1 个 CTA。epilogue 复用 stage buffer 作暂存（所有 load 已等待、所有 MMA 已完成），每个 warp、每 32 列一个 32×32 box，按 128B swizzle 写入（16 字节 store 无 bank conflict，NCU 计数为 0），`fence.proxy.async` + `__syncwarp` 后由 lane 0 发 TMA store；warp 之间不需要同步。退出前 `cp.async.bulk.wait_group.read 0`（与 CUTLASS 的 `tma_store_wait<0>` 相同）。
- **TMEM 生命周期**：alloc 由 warp 0 执行并 relinquish；`fence::before → __syncthreads → fence::after` 后所有线程读 base；dealloc 前同样的 fence 序列保证所有 `tcgen05.ld` 已完成。

## 各 step 的性能（CUDA Graph，相对 cuBLAS BF16→FP32）

| 配置 | 4096³ | 8192×8192×4096 |
|---|---|---|
| step 2：`<128,4>` 逐行写回 | ≈ 0.37× | ≈ 0.33× |
| `<128,4>` TMA 写回 | ≈ 0.51× | ≈ 0.42× |
| `<256,4>` 逐行写回 | ≈ 0.50× | ≈ 0.45× |
| `<256,3>` TMA 写回 | ≈ 0.62× | ≈ 0.55× |
| `<128,6>` TMA 写回 | ≈ 0.52× | ≈ 0.46× |
| `<128,3>` TMA 写回（2 CTA/SM） | ≈ 0.34–0.6×（不稳定） | ≈ 0.60× |
| step 3：`<256,4>` TMA 写回 | ≈ 0.81–0.83× | ≈ 0.74–0.77× |

- `selftest --perf`：square 0.830、wide 0.751，门槛 0.6（任务说明预期 0.5–0.7；实测更高，门槛留出余量，因为本机 GPU 在多次运行之间偶尔整体降速）。
- NCU（square）：`sm__pipe_tensor_cycles_active` step 2 ≈ 26%，step 3 ≈ 68%；DRAM 写入 step 2 ≈ 99 MB（C 为 67 MB，逐行 store 产生零散的部分 sector 写），step 3 ≈ 42 MB；epilogue 的 shared store bank conflict 为 0。
- 剩余差距（推断，未逐项测量）：epilogue 不与主循环重叠（每 SM 1 个 CTA）、单线程串行地等 TMA 与 MMA、square 的 3.46 波尾部、A/B 每个 CTA 从 L2 重读。分别由 32 关（warp specialization、TMEM 双缓冲、persistent）与 33 关（2-CTA、multicast）处理。

## 验证

- `./lab check 31 --solution`：3 个 step 全部 case 通过（`int` case 精确相等，`random` 最大误差 2.4e-4，在容差内）。
- `selftest 31 --perf` 通过；骨架在 GPU 上报 `TODO(1)`（warp 0 的 `tmem_alloc` 先 trap，不会死锁）。
- compute-sanitizer：memcheck / racecheck / synccheck 在 `small`（step 1）、`tail`（step 2、3）上 0 错误；racecheck（step 3）与 synccheck（step 2）在 `long_k` 上 0 错误。sanitizer 没有输出任何关于 tcgen05/TMA 不受支持的提示，但从 30 关的实验看，racecheck 不跟踪 async proxy 的访问；TMEM 访问与 MMA 读 shared memory 也不在其检查范围内（推断）。因此 0 错误只说明 generic proxy 部分没有问题，tcgen05 的同步正确性由按规范推理和精确比较的 case（`long_k` 绕环 4 圈）保证。
- `./lab sass 31 --solution`：无 spill（56/46/44 寄存器）；有 `UTCHMMA`、`UTCBAR`、`UTMALDG`、`UTMASTG`、`LDTM`、`SYNCS`。
- 骨架：在临时副本中按 TODO(1)、(2)、(3) 逐步填入，第 n 步只依赖前 n 组 TODO，填完后学生版与答案同速（square ≈ 0.81×）；骨架已恢复为原样。

## 已知局限

- 不支持 `K % 8 ≠ 0` 或 `N % 4 ≠ 0`（TMA 行距限制）。
- 只有 BF16 → FP32；FP16 只需改 idesc 的 A/B 格式位。
- 没有光栅化：`blockIdx.x` 沿 N。wide 上 L2 命中率约 84%，kernel 受 Tensor Core 与 epilogue 限制，未优化。
- 步骤 3 的 `<128,3>`（每 SM 2 个 CTA）在 square 上结果波动很大，原因未查明。
