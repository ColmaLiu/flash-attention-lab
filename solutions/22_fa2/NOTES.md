# 22_fa2 参考答案说明

## 设计选择

**一个 (Q 块, head, batch) 一个 CTA，K/V 在内层循环。** 即 FA2 论文 3.1/3.2 节。
O、m、ℓ 从头到尾在寄存器里；O 和 LSE 在 epilogue 中各写一次。grid 为
`(⌈N/Br⌉, H, B)`，`blockIdx.x` 最快变化。没有做 LPT/反向发射：causal 的负载不均衡留给 43 关。

**4 个 warp，按 Q 行划分（split-Q，论文 3.3 节）。** 每个 warp 负责 `16·kMTiles` 行，
softmax 的行归约只在 quad 内 shuffle，warp 之间只共享 K/V tile。

| D | kMTiles | Br | Bc | 每 lane 数组寄存器（Q/O/S） | 寄存器/线程 | CTA/SM |
|---|---|---|---|---|---|---|
| 128 | 1 | 64 | 64 | 32 / 64 / 32 | 218–237 | 2 |
| 64 | 2 | 128 | 32 | 32 / 64 / 32 | 168 | 3 |

- D = 64 用 Bc = 32 而不是上游的 64/128：Bc = 64 时数组共 160 个寄存器，步骤 2/3 的非 causal
  kernel 出现 16 字节的 stack（每块一次 LDL）。试过把“需要 mask 的块”改成预先算好的下标、
  调整同步位置，都只是把 spill 挪到别的实例化上，或让 D = 128 的 kernel 变慢约 15%
  （寄存器分配变化）。Bc = 32 在步骤 4 上只慢约 8%（0.43× → 0.39× cuDNN），且两种 D 的
  寄存器预算一致，讲义更好讲。
- 试过 D = 128 用 8 个 warp、Br = 128：步骤 4 从 ≈0.26× 降到 ≈0.22×。每 SM 只能放一个 CTA，
  barrier 时没有另一个 CTA 填补。
- **Q 常驻寄存器**（上游 `Is_Q_in_regs` 的做法），每个块只 ldmatrix K 和 V。

**shared memory。** `[rows][D]`，每 128 字节段独立用 `lab::swizzle_chunk<8>`（D = 128 的行是两个段）。
`cp.async` 写入和 `ldmatrix(.trans)` 读出都无冲突（profile 中 `ldgsts` 冲突为 0）。越界的
Q/K/V 行用 `cp_async_16_zfill` 写零：V 的填充行若是 NaN，`0·NaN` 会污染 O。

**P 不经过 shared memory。** 两个相邻 n8 的 S C-fragment 正好组成一个 m16n8k16 的 A fragment
（讲义 §4.3），只做 FP32 → 16 位的打包。ℓ 用打包前的 FP32 P~（与上游一致）。

**V 用 `ldmatrix_x4_trans`。** V 是 `[key][D]`，对 P·V 的 B 操作数是 MN-major。

**数值。** m 存原始分数（scale > 0 才成立，binding 保证）；指数用 `ex2.approx.ftz` 的本地封装
`exp2_approx`（不加 `--use_fast_math` 时 `exp2f` 不是单条 MUFU），指数参数是一条
`fmaf(s, scale_log2, −m·scale_log2)`。`m' = −∞` 时以 0 作减数，避免 `−∞ − (−∞)`；本答案的
块划分下真实行不会触发（块 0 的 key 0 对每一行可见），仅作防御。ℓ 保持每 lane 部分和，
epilogue 才 quad 求和。v1（main 分支 `s22_fa2`）中的 FP64“极端 scale”路径与 scale ≤ 0 的
NaN 问题都已去掉。

**causal。** 块数为 `min(⌈N/Bc⌉, ⌈(q0+Br)/Bc⌉)`；块需要 mask 当且仅当越过 N，或
`k0 + Bc − 1 > q0`。前向顺序遍历块（上游是从后往前，先处理 mask 块），以便和步骤 4 的流水
共用同一循环。

**步骤 4 的流水。** 与 21 关步骤 1 同构：Q 和 K/V 块 0 同属 group 0；每块一次
`wait_group 0` + 一次 `__syncthreads()`，之后发出下一块到另一个 stage。没有试上游那种
K、V 分别成组、V 的拷贝与 QKᵀ 重叠的更细流水；3 级流水按 21 关的结论预计收益很小，也未测。

**本地 helper（头文件中没有的）：** `exp2_approx`（`ex2.approx.ftz.f32`）、`pack2<T>`、
`mma<T>`（按 dtype 选择 `lab::mma_m16n8k16_{f16,bf16}`）、`quad_max/quad_sum`、两段 swizzle 的
`tile_offset`。都不是新的 PTX 原语，没有放进 `include/lab`。

## 各 step 的性能（相对 cuDNN，CUDA Graph，7 号卡）

| step | `perf` [4,4096,16,128] BF16 | `perf_causal` | `perf_d64` [4,4096,32,64] FP16 |
|---|---|---|---|
| 1（probe，无意义） | — | — | — |
| 2 / 3（同步加载） | ≈ 0.23× | ≈ 0.30× | ≈ 0.36× |
| 3，不跳块（实验） | — | ≈ 0.15× | — |
| 4（双缓冲） | ≈ 0.26× | ≈ 0.32× | ≈ 0.39× |
| FA4 / cuDNN | ≈ 1.1× | ≈ 1.27× | ≈ 1.15× |

`perf_target` 设为 cuDNN 的 0.2×，最差的 `perf` 有约 25% 余量。步骤 4 在 `perf` 上约为 20 关测得的
`mma.sync` 上限（≈ 560 TFLOP/s）的 61%；`sm__pipe_tensor_cycles_active` 步骤 2 约 60%、步骤 4 约 66%。
`--cold` 比例基本不变。与 12 关（FP32 SIMT，`perf` 约 25 TFLOP/s）相比，`perf_d64` 约高 13 倍
（dtype 不同，主要来自 Tensor Core）。

## 已知局限

- `mma.sync` 本身的上限约为 cuDNN 的 0.43×；要更快必须换 `tcgen05`（部分 3）。
- 每个 SM 只有 8–12 个 warp，softmax 与 mma 在同一 warp 内串行，没有 warp specialization。
- causal 不做 LPT 排序，最重的 Q 块最后发射。
- O 直接从 fragment 写回（每 lane 4 字节），没有经 shared memory 合并成 16 字节写；O 只占很小的流量，
  未测到影响。
- 只支持 D ∈ {64, 128}、FP16/BF16，O 与输入同 dtype（`make_plan` 检查）。
