# 10_simt_gemm 参考答案说明

## 设计选择

- **步骤 1 让 `threadIdx.x` 对应 N。** 写 C 合并，读 A 是广播，读 B 每个 warp 跨 32 行。两种 K-major 操作数无论怎样映射，朴素版本总有一个操作数不合并；选这种映射是为了和 00 关一致（x 方向连续写）。这个版本极慢（约 cuBLAS 的 1%），讲义提醒学生 bench 要等一会儿。
- **步骤 2 用 1024 线程、每线程一个输出。** 这样“分块”是唯一的变化，便于和步骤 1 对比。`[32][33]` 的 padding 直接沿用 00 关，不单独设一个“有冲突”的步骤：本关的新知识是复用，bank 冲突的分析放在步骤 3、4 的布局设计里。
- **步骤 3 用跨步分配（`ty + 16i`、`tx + 16j`）并保持 K-major smem。** 这样保持 K-major 布局也无 bank 冲突（`stride 9` 与 32 互素），步骤 3 的提升全部来自寄存器复用，和步骤 4 的“布局 + 向量化”分开。tile 装载函数 `stage_kmajor` 在骨架中提供，学生只写外积和写回。
- **步骤 4 转置存放、两组 4×4、padding 4。** 转置使固定 `kk` 的 m/n 连续，才能 `LDS.128`；两组相隔 64 使 16 个 `tx` 的 `float4` 落在连续 256 字节；padding 4 消除转置写入的 2 路冲突并保持 16 字节对齐。
- **尾部处理放在 `load4` / `store4` 里。** `K % 4 != 0` 时退回标量读，`N % 4 != 0` 时退回标量写，M/N/K 越界填 0 或跳过。这些是非本关重点的细节，骨架中已提供。
- **不做全局 → smem 的预取或多级流水。** 那是 `21_pipeline_gemm` 的主题；本关答案只包含复用、寄存器分块和向量化。
- **BM = BN = 128、BK = 8、TM = TN = 8、256 线程**：经典 SIMT SGEMM 配置。每线程约 127 个寄存器，每 SM 2 个 block；smem 每 block 约 8.4 KB，不是限制因素。

## 各步性能（CUDA Graph，相对 cuBLAS FP32、TF32 关闭）

| 步骤 | `square4k` | `wide` | 主要原因 |
|---|---|---|---|
| 1 | ≈ 0.01× | 未测（太慢） | 每次乘加 2 次全局 load，B 不合并 |
| 2 | ≈ 0.14× | ≈ 0.13× | 每次乘加 2 条 LDS，FFMA 最多占 1/3 的发射槽 |
| 3 | ≈ 0.56× | ≈ 0.53× | 每次乘加 0.25 条 LDS；smem wavefront 与 FFMA 刚好平衡 |
| 4 | ≈ 0.72× | ≈ 0.68× | LDS 指令再减为 1/4，wavefront 减 25%，转置写入无冲突 |

cuBLAS 在这里选中 `cutlass3x_sm100_simt_sgemm_..._128x128x16_..._3_tnn_align1`，约为 FP32 标称峰值的 35–39%。独立 FFMA 的微基准可达标称峰值约 97%，所以 GEMM 与峰值的差距在 kernel 内部，不在时钟。

`perf_target` 设为 0.6：本关是中间关卡，0.6 足以说明寄存器分块和向量化生效，同时给两个 perf case 留出约 10% 的余量（`wide` 实测 ≈ 0.68）。`--cold` 下比例约 0.84，比 graph 模式高。原因未查明：cold 模式下两者都更快，可能与迭代间的空隙有关。

## 验证

- `./lab check 10 --solution`：全部 step、全部 case 通过。
- `./lab sanitize 10 --solution --step n --case tail`（n = 1..4）以及 `--case tail_vec`：memcheck、racecheck、synccheck 无报错。
- `./lab sass 10 --solution`：无 local memory，无 spill。
- 把骨架的 TODO 依次换成答案代码：填到第 n 个时 `--step n` 通过，`--step n+1` 报 `TODO(n+1)`。

## 已知局限

- 没有隐藏全局访存延迟（无预取、无双缓冲），每 SM 只有 16 个 warp；这是和 cuBLAS 的主要差距，留给 21 关。
- 4096³ 的 grid 为 1024 个 block，每 SM 2 个时约 3.5 轮，最后一轮只填满一半；未做 tile 形状或 split-K 调整。
- `load4` 的向量路径假设 A、B 基址 16 字节对齐（PyTorch 分配满足）。
