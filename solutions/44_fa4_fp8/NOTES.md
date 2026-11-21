# 44_fa4_fp8 参考答案说明

## 设计选择

**流水结构原样取自 40 关步骤 3。** 16 个 warp（2 × softmax、correction、MMA、TMA、2 个空闲），barrier 协议、MMA 发射顺序、
TMEM 列号都不变。本关只换数据类型，所以三个 step 都用步骤 3 的结构（含 correction 与 `setmaxnreg` 192/80/48），
不再拆出 40 关的中间形态。

**FP8 tile = 一个 panel，swizzle 跨度 = D 字节。** D = 128：一行 128 字节，128B swizzle；D = 64：一行 64 字节，64B swizzle
（新增 `lab::make_smem_desc_kmajor_sw64`；V 用 42 关新增的 `lab::make_smem_desc_mnmajor_sw64`，按字节语义同样适用于 8 位）。
TMA 用 `UINT8`，box `[1][rows][1][D]`。K 片步长 32 字节（= 32 个 E4M3），与 16 位版本的字节步长相同。

**V 不转置。** `tcgen05` 的 instruction descriptor 允许 E4M3/E5M2 的 MN-major（CUTLASS `mma_sm100_desc.hpp` 的注释，
FA4 对 FP8 的 V 同样用 `OperandMajorMode.MN`）。因此没有 FA3（Hopper）那样的 kernel 内转置，也没有在 `make_plan` 中预先
转置（那会把工作移出计时区）。计时边界与 40 关相同。

**descale 折进 scale。** `qk_scale = scale · q_descale · k_descale`，`scale_log2 = qk_scale · log₂e`，在 host 计算，
softmax 代码与 40 关相同；`v_descale` 在 epilogue 每个元素乘一次。

**P 乘 2⁸ 量化为 E4M3**（FA4 的 `max_offset = 8`）：`exp2` 的 base 减 8，`cvt.rn.satfinite.e4m3x2.f32` 两两打包、四个一列
写进 S 的第 64–95 列，TS 形式的 `kind::f8f6f4` MMA 每条读 8 列。ℓ 带同样的 2⁸，O / ℓ 自动撤销，LSE 减 8·ln2。

**两个归一化量（步骤 3）。** `l`：FP32 的 Σp，只用于 LSE；`lo`：O 的分母。步骤 2 中 `lo = l`（与 FA4 相同）；步骤 3
中 `lo = Σp̂`（实际参与 P·V 的 E4M3 值之和），O 成为精确的凸组合。最初版本让 LSE 也用 Σp̂，`one_block` 的 LSE 误差
9.8e-3 超过 1e-3：LSE 必须描述真实的 softmax。求和用 `cvt.rn.f16x2.e4m3x2`（精确）+ 混合精度 `add.rn.f32.f16`
（SASS 一条 FHADD）；第一版用 `__half22float2` + FADD（每元素 HADD2.F32 + FADD），步骤 3 比步骤 2 慢 6–16%（D = 64 最多），
换成 FHADD 后不再变慢。

**O 暂存在 Q⁰ + Q¹ 上。** FP8 的一个 Q tile 只有 128·D 字节，BF16 的 O tile 要 2·128·D；两块 Q 合起来正好一个 O tile。
`o_full[0]` 完成时全部 QK（含 QK¹）都已完成（它们在最后一个 PV⁰ 之前发出），所以两块 Q 都空闲；correction 先写 tile 0，
每个 warp 的 lane 0 `bulk_wait_group_read<0>` 后 `__syncwarp`，再写 tile 1 的同一批行。

**`align_smem` 中的空 asm。** 第一版在步骤 2–3 的 kernel 中有 8 字节 STACK：prologue 算出的 1024 字节对齐的 smem 基址
被保存到 local memory，在 TMA 分支（`setmaxnreg.dec` 到 48 之后）里 4 次 LDL。把寄存器预算改成 (184, 80, 64)、
(192, 72, 56)、(192, 64, 64)、(184, 96, 48)、(192, 88, 40) 都没有消除（有的更多）。在 `setmaxnreg.dec` 之后用一个对编译器
不透明的空 asm 重新计算基址（几条整数指令），全部 12 个 kernel STACK 为 0。原因（ptxas 在 `setmaxnreg` 两侧的分配
策略）未进一步研究。

**K/V 环的级数**：D = 128 用 8 级（与 40 关同样 128 KB），D = 64 用 12 级。4/6 级与 8/12 级在 `perf`、`perf_d64` 上的差别
在噪声内（HBM 只用了约 2%）。

## 误差与容差

测量脚本（未提交）把每个非 perf case 的 O 与 FP64 oracle 比较，记录 `max |err| / (atol + rtol·|expected|)`（容差预算
的使用率）：

| 方法 | (5e-2, 1e-1) | (2⁻⁴, 2⁻³) |
|---|---|---|
| 步骤 2（E4M3 P × 2⁸，ℓ = Σp） | 0.91（`causal_peaked`） | 0.73 |
| 步骤 3（O 用 Σp̂） | 0.94（`causal_peaked`） | 0.75 |
| FA4 FP8（`_flash_attn_fwd`） | 0.91 | 0.73 |
| P scale = 1（`kPLog2 = 0`），kernel 实测 | 0.91 / 0.95 | 0.73 / 0.76 |
| P scale = 448，kernel 实测 | **1.19 / 1.17（`peaked`）** | 0.95 / 0.94 |
| BF16 P（PyTorch 模拟） | 0.06 | 0.05 |

原容差 (5e-2, 1e-1) 只给答案留了 6% 的余量，并且拒绝 P scale = 448 这一合理变体，所以改为 (u, 2u)，u = 2⁻⁴ 是 E4M3
的单位舍入误差（依据见讲义 §3.5：集中权重的行误差约 u·|v_a − v_b|）。用故障注入确认放宽后仍能发现错误（`./lab check`
在第一个失败 case 停下）：P 的字节交换（0↔1 → `n1`；2↔3 → `one_block`）、整字节反序、PV 的 TMEM 列步长 ×2、P¹ 用了 tile 0
的列（`two_tiles`）、D = 64 的 V SBO 错为 1024（`one_block`）、causal mask 多看一个 key（`causal_pair`）、LSE 漏减 8·ln2 或 P
没有缩放（`n1` 的 LSE）在两种容差下都失败。

P scale = 1 与 2⁸ 在全部 case 上误差相同：本关的随机数据中 p < 2⁻⁶ 的 key 对 ℓ 贡献很小。讲义 §3.4 用长尾行的手算例子
说明 2⁸ 的作用，没有为此加 case。

## 各 step 的性能（CUDA Graph，4 号卡，2–3 次运行的范围）

| 版本 | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| 步骤 2 vs FA4 FP8 | ≈ 0.68–0.69× | ≈ 0.71× | ≈ 0.73× | ≈ 0.68× |
| 步骤 3 vs FA4 FP8 | ≈ 0.69–0.73× | ≈ 0.73× | ≈ 0.74× | ≈ 0.70–0.72× |
| 步骤 3 相对 40 关步骤 3 | ≈ 1.23× | ≈ 1.15× | ≈ 1.12× | ≈ 1.24× |
| FA4 FP8 相对 FA4 16 位 | ≈ 1.27× | ≈ 1.12× | ≈ 1.05× | ≈ 1.37× |

步骤 1 是 probe step（输出 [B,H,N,N] 的 S），perf case 的 probe 要 4 GiB，没有计时。

Nsight Compute（`perf`）：Tensor pipe ≈ 29%（40 关 ≈ 57%），XU（MUFU）≈ 58%（40 关 ≈ 57%），DRAM ≈ 2%。与链模型一致：
PV + QK 减半（≈ 1100 → 550 周期），softmax 不变，3450 / 2900 ≈ 1.19×。步骤 3 比步骤 2 快约 1–4%（可重复），原因未查明。

`perf_target`：相对 `fa4`（FA4 FP8）0.63；最差的 `perf_8k`、`perf` 在 ≈ 0.69–0.72×，单次运行波动约 ±3%。cuDNN 没有 FP8 基线。

## FA4 基线

固定版本中公开的 `flash_attn_func` 不接受 descale 参数（原 labkit 在 e4m3 时传 `q_descale=` 会触发 `TypeError`，
而 `bench` 只捕获 `EnvironmentError_/RuntimeError/NotImplementedError`）。`labkit/kinds/attention.py` 的 `_fa4` 在 e4m3 时改用
`flash_attn.cute.interface._flash_attn_fwd`（与上游 `benchmark_flash_attention_fp8.py` 相同），descale 形状 `[B, H]` FP32，
O 为 BF16。已核对它的 O/LSE 与 FP64 oracle 一致（误差见上表；LSE ≤ 1e-4）。

## 验证

- `./lab check 44 --solution`：3 个 step（8 + 25 + 25 个 case）全部通过；`./lab selftest 44`、`selftest 44 --perf`（0.63）通过；
- `./lab sanitize --solution`：memcheck、racecheck、synccheck 在 `two_tiles`（步骤 1–3）、`tile1_tail`（步骤 1）、`causal_tail`
  （步骤 2–3）上 0 错误；racecheck 看不见 TMA/TMEM，同步论证沿用 40 关 §5.6；
- `./lab sass 44 --solution`：12 个 kernel STACK 均为 0，`UTCQMMA`（tcgen05.mma 的 FP8 形式）104 条；
- 骨架：由答案生成；逐步把第 1..n 步的 TODO 换回答案后 `--step n` 通过、`--step n+1` 报 TODO(n+1)；只填 1a 时在
  TODO(1) 处停下而不挂起；验证后恢复为原骨架（字节一致）。

## 已知局限

- 继承 40 关的全部局限（同一线程 MMA 按序执行的依赖、causal 的浪费块、非 persistent）。
- 只支持 per-tensor descale；FA4 支持 per-(batch, kv head) descale，FA3 论文的 block quantization 不在本关范围。
- O 只输出 BF16；E5M2 输入未支持（只需改 idesc 格式码和输入检查）。
- Q/K/V 的量化误差不计入检查（oracle 用反量化后的输入）。
