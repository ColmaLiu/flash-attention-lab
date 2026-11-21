# 11_naive_attention 答案说明

## 设计选择

- **三个独立 kernel，S/P 落在全局内存。** 这是 FlashAttention 论文的 Algorithm 0，本关的目的就是测量它的代价，所以刻意不做融合。
- **一个带任意 stride 的 batched SIMT GEMM 同时服务 QKᵀ 和 PV。** 统一接口 `C(m,n) = alpha·Σ_k A(m,k)·B(n,k)`（与课程的 GEMM 约定一致），每个操作数用 `{b, h, row, col}` 四个 stride 描述。这样 BNHD 中的一个 head、`[B,H,N,N]` 的 S、以及 “按 (d, j) 读 V” 都只是 stride 的不同取值，学生练的就是这个映射，不需要转置或复制 V。
- **GEMM 用 64×64×16 tile、每线程 4×4，而不是 10 关的 128×128、8×8。** PV 的 n 维是 D（64 或 128），128 宽的 tile 在 D = 64 时会浪费一半；另外 loader 要支持任意 stride（按 stride 为 1 的方向分配 lane），不能假设 float4 对齐。线程拥有 `ty + 16i` 行、`tx + 16j` 列，使 shared memory 读无 bank conflict、C 的写合并。这是 “非本关重点” 的代码，可读性优先。
- **softmax 一个 block（128 线程）处理一行，两遍：online (m, l)，然后写 P。** 用的是 01 关的 `State`/`merge`/`block_merge`。第二遍的读基本命中 L1/L2，HBM 上 S 只读一次（profile 中 `softmax_rows` 的 DRAM 读 ≈ S 的大小）。P 原地覆盖 S，只需一块 scratch。
- **steps 1–2 把 S/P 写进 `p.probe`，step 3 写进 `make_plan` 分配的 `lab::Scratch`。** 三个 step 是同一段代码逐步补全，用 `p.step` 提前返回，不需要分支版本。
- **causal 只在 softmax 中处理。** 两个 GEMM 仍然计算完整的 N×N，P 的上三角写 0。这让 `perf_causal` 的耗时与 `perf` 几乎相同，正好用来说明标准实现浪费了一半工作；跳块属于后续关卡。
- **`make_plan` 对 S 超过 32 GiB 的输入明确报错。** 这是本关要说明的显存问题，而不是静默失败。

## 验证

- `./lab check 11 --solution`：三个 step 的全部 case 通过（默认流与非默认流）。
- Compute Sanitizer memcheck / racecheck / synccheck 在 `bnhd_causal` 上无报错；memcheck 在 `single`（N = 1）、`position_causal` 上无报错。
- 在骨架副本中逐个填入 TODO：填入 TODO(1) 后 step 1 通过、step 2 报 TODO(2)；以此类推。
- `large_logits`（D = 32，scale = 1）中九成以上的行最大分数超过 88.7；去掉减最大值的版本在 step 2 得到 NaN。把 D 提到 128、分数到 200 左右时，FP32 GEMM 本身的舍入误差（约 1e-4）就会超过 O 的容差，所以这个 case 用 D = 32。
- SASS：`gemm_kernel` 38 个寄存器，`softmax_rows` 31 个，均无 spill。

## 性能（比例，CUDA Graph）

| case | vs efficient |
|---|---|
| `perf` [1, 4096, 16, 64] | ≈ 0.32× |
| `perf_causal` | ≈ 0.21× |

- 时间分布（`perf`）：QKᵀ ≈ 40%，softmax ≈ 9%，PV ≈ 50%。两个 GEMM 算力受限（SIMT FMA，约 15–19 TFLOP/s），softmax 带宽受限。
- DRAM 字节（`perf`）：实测合计 ≈ 4.25 GB，第 3.4 节估算 4.36 GB；差值是 kernel 结束时仍留在 L2 的脏数据。efficient 基线同形状实测 DRAM 读 ≈ 0.05 GB。
- N 翻倍时间约 ×3.1（1024→2048）、×3.8（2048→4096）。
- `perf_target` 设为 0.15（相对 efficient），只说明实现没有明显的性能 bug；本关不以速度为目标。

## 已知局限

- efficient 基线的 FP32 路径用 Tensor Core 3xTF32（源码推断：`gemm_kernel_utils.h` 中 `DefaultGemmType<ArchTag, float>` 选 `OpMultiplyAddFastF32`），所以它和本答案的差距不全来自 IO。
- PV 的 grid 在 D = 64 时只有 B·H·⌈N/64⌉ 个 block，SM 占用偏低；split-K 或更小的 M tile 能改善，但不是本关内容。
- softmax 对 N = 1 之类的短行仍用 128 线程，大部分线程空闲；对本关的规模无影响。
