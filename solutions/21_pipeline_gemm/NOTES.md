# 21_pipeline_gemm 答案说明

## 设计选择

- **支持域**：A、B 为 BF16，C 为 FP32，`K % 8 == 0`（16 字节 `cp.async` 不跨行尾）；M、N 任意。越界 chunk 用 `cp.async` 的 `src-size = 0` 填零，源地址替换为矩阵首地址，不读任何越界数据。
- **一个 kernel 模板、三个配置**：`Config<BM, BN, BK, WarpsM, WarpsN, Stages>`。`compute_stage`（ldmatrix + mma）和 `store_c` 与 20 关相同，只是参数化；shared tile 统一用 `lab::swizzle_chunk<BK/8>`。
- **step 1（双缓冲）**：单独写成 `gemm_double_buffer`，让学习者先看清最简单的“wait → barrier → 发下一块 → 算这一块”。每个 K tile 一个 `wait_group 0` 和一个 `__syncthreads()`；barrier 同时承担 RAW（新 tile 对所有 warp 可见）和 WAR（上一 tile 的读者都已离开，它的 stage 可以覆盖）。
- **step 2（环形 buffer）**：`gemm_multistage`，不变式“第 g 个 commit group 恰好是 tile g”，K 不足时提交空 group，因此每轮都是 `wait_group<S−2>`，没有尾部特判。S = 3（48 KB），每个 SM 两个 CTA。
- **step 3（调参）**：BK = 64（128 字节行，8-chunk swizzle），4 个 64×64 warp，3 级（96 KB，每 SM 两个 CTA，约 254 寄存器 × 256 线程正好用满寄存器文件），分组光栅化 `kGroupM = 8`。
- **shared memory carveout**：`allow_smem` 除了设置动态 shared memory 上限，还设置 `PreferredSharedMemoryCarveout = MaxShared`。本机上不设时，64–96 KB 的配置偶尔出现整段慢约一倍的运行（样本呈双峰），设后 step 3 稳定。原因推测为驱动选的 carveout 只容得下一个 CTA，未进一步证实。

## 各 step 的性能变化（相对 cuBLAS BF16→FP32，CUDA Graph）

| step | 4096³ | 8192×8192×4096 | 占 mma.sync 上限（≈560 TFLOP/s） |
|---|---|---|---|
| 20 关 step 3 | ≈ 0.17× | ≈ 0.17× | ≈ 45% |
| 1 双缓冲 | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 2 三级流水 | ≈ 0.24× | ≈ 0.21× | ≈ 60% |
| 3 调参 | ≈ 0.28× | ≈ 0.26× | ≈ 72% |

- step 1 → 2 几乎没有提升。按 Little 定律估算（讲义 §3.1），每个 SM 两个 CTA 各提前一个 tile 已接近所需的在途量。4 级（64 KB）的版本在本机上没有更快，而且样本波动大（见上），所以 step 2 用 3 级。
- step 3 的提升来自 BK = 64 与 64×64 warp tile（每条 mma 需要的 `ldmatrix` 更少，barrier 更少）。配置实验（4096³）：`128,128,64,2,4,3` ≈ 0.27×，`128,256,32,2,4,3` ≈ 0.19×（170 寄存器，每 SM 一个 CTA），`128,128,64,2,2,4` ≈ 0.23×（128 KB，每 SM 一个 CTA）。
- 分组光栅化：`wide` 上 DRAM 读取约 716 MB → 556 MB，L2 命中率约 79% → 86%，时间不变；kernel 受限于 Tensor Core（NCU `sm__pipe_tensor_cycles_active` 约 79%，DRAM 吞吐约 5%）。保留它是为了教学：讲义用它说明“优化要对准瓶颈”。
- `perf_target` 设为 0.22；答案约 0.26–0.28。`wide` 上我们的 kernel 每次运行有约 ±10% 波动（cuBLAS 稳定），目标留了余量。

## 验证

- `./lab check 21 --solution`：全部 step、全部 case 通过。
- compute-sanitizer：memcheck / racecheck / synccheck 在 `tail` 上无报错；racecheck、synccheck 在 `aligned`（16 个 K tile，环形 buffer 绕多圈）上无报错。
- `./lab sass 21 --solution`：无 spill（LOCAL 0）；SASS 中有 `LDGSTS`、`LDSM`、`HMMA`。
- 骨架：逐步填入 TODO，第 n 步只依赖 TODO(1..n)，完成后与答案性能一致；骨架已恢复。

## 已知局限

- 剩余约 30% 到 `mma.sync` 上限：每 tile 一次 CTA barrier、`ldmatrix`→mma 依赖、1024 个 CTA 在 296 个驻留位置上 3.46 波的尾部效应。没有做 fragment 双缓冲、persistent kernel 或 split-K。
- 写回仍是逐元素 FP32 store。
- 只支持 BF16 输入、FP32 输出。
