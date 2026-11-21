# 41_fa4_softmax 参考答案说明

## 设计选择

**流水结构完全沿用 40 关步骤 3。** 16 个 warp（2 × softmax WG、correction WG、MMA、TMA、2 个空闲）、TMEM 布局、K/V 单环、全部
mbarrier 与 parity、`setmaxnreg` 192/80/48 都不变；40 关的步骤 1–2 配置（P 经 shared memory、无 correction WG）删掉了。
本关只改 softmax 求 P 的算术和 correction 的一个条件，所以三个 step 的差别都可以归因到 softmax。

**软件 `exp2`（`exp2_poly2`）。** 与 FA4 `utils.py` 的 `ex2_emulation_2` 同一做法：`max(x, −127)` 钳位，`x + 1.5·2^23` 向下舍入
（`__fadd2_rd`）把 ⌊x⌋ 放进低位，`f = x − ⌊x⌋` 精确，3 次 Horner，`bits(p) + (bits(r) << 23)` 拼指数（ptxas 合成 `LEA`）。
用 CUDA 12.8 的 `float2` 内建函数（`__ffma2_rn`、`__fadd2_rn/rd`）而不是内联 PTX，可读性更好，SASS 与内联 PTX 版本相同
（`FFMA2`、`FADD2`、`FADD2.RM`）。系数是自己用 numpy（Lawson 迭代求相对误差的 minimax，c0 固定为 1）算的，
没有抄上游的 `POLY_EX2`；两者在第 3 位有效数字上不同，最大误差同量级。

- c0 必须恰好为 1：x = −∞ 钳位成 −127 后 f = 0，结果的位模式是 `bits(c0) − (127 << 23)`，只有 c0 = 1.0 时得 +0
  （c0 = 0.99999 时得到 NaN；已用变异测试确认）。
- 次数按 FP16 P 的单位舍入 2^−11 选 3 次（误差 2^−13.5）。2 次（2^−8.9）对 BF16 够用、对 FP16 不够；未做“按 dtype
  选次数”的版本。

**混合比例。** 每 32 列（一次 `tcgen05.ld`）的 16 对中 `kPolyPairs = 3` 对用多项式（φ = 18.75%），由 `use_poly` 在编译期均匀
分布（q = 5、10、15）。扫描 k = 0、2、3、4、5、6、8、16（步骤 2，`perf` / `perf_d64` 各测一次）：k = 3 最好；k ≥ 4 时
softmax 段在 192 个寄存器内放不下，`STACK` 8–24 字节，性能不规则下降（k = 5 只有 6 个 kernel 溢出，比 k = 4 好）。
吞吐模型（讲义 §3.6）预测平衡点 φ ≈ 0.39，实测最优低得多：模型没有算发射槽、依赖延迟（每个 SMSP 只有 2 个 softmax
warp）和寄存器。FA4 的 `ex2_emu_freq` 配置对应 12.5–19%，与此一致。没有尝试 FA4 的“首尾 fragment 不模拟”
（`ex2_emu_start_frg`）变体。

**打包的 `FFMA2`/`FADD2`。** 算 x（`s·scale_log2 − base`）和求和也用打包指令（四个 `float2` 部分和）。k = 0（全 MUFU + 打包）
与 40 关相当，说明打包本身不是收益来源；它的作用是给多项式让出发射槽。

**条件 rescale（步骤 3）。** softmax 的 m 是“基准”而不是真最大值：`(m_new − m)·scale_log2 > τ = 8` 才移动并给出
α = 2^((m − m_new)·scale_log2)，否则 α = 1。块 0 的 m = −∞ 使差为 +∞，必然移动；全 −∞ 时 NaN > τ 为假，α = 1，不会
出 NaN（实际不会出现：块 0 总有可见 key）。LSE 用基准 m 计算（`m·scale + ln ℓ`），对任意基准都精确。correction 用
`__any_sync(α ≠ 1)` 以 warp 为单位决定（`tcgen05.ld/st` 是 `.sync.aligned`），跳过时仍 arrive `p_ready`。
与 FA4 的 `SoftmaxSm100.update_row_max`（`rescale_threshold`）和 `correction_loop`（`vote_ballot_sync(scale < 1.0)`）相同。
τ = 8 满足 FP16 的 P ≤ 256 < 65504 和多项式的 x ≤ 127。

对照：只保留投票、τ = 0 时 `perf` ≈ 0.91×、`perf_d64` ≈ 0.94×（步骤 2 为 0.89×、0.94×，τ = 8 为 0.97×、0.96×）。

**新增 pattern（`labkit/kinds/attention.py`，只增加分支）。** `growing_max`：q_i = a_i u + 噪声、k_j = g(j) u + 噪声，
g(j) = 2j/128，每行最大值每块上涨约 1.4–4.3（log₂ 单位），单块不超过 τ，累计超过；`jump_max`：g 在 key 3n/5 + 50（块中间）
处跳到 200，跳变约 145–430。a_i/√d ∈ [0.5, 1.5) 每行不同，使同一 warp 中的行在不同块修正。`docs/authoring.md` 的
pattern 列表已补上这两个名字。

## 各 step 的性能（相对 cuDNN，CUDA Graph，7 号卡，同一次 bench）

| 版本 | `perf` | `perf_causal` | `perf_d64` | `perf_8k` |
|---|---|---|---|---|
| 40 关步骤 3 | ≈ 0.85× | ≈ 0.94× | ≈ 0.80× | ≈ 0.83× |
| 步骤 1（全多项式） | ≈ 0.58× | ≈ 0.69× | ≈ 0.61× | ≈ 0.56× |
| 步骤 2（k = 3） | ≈ 0.89× | ≈ 0.99× | ≈ 0.94× | ≈ 0.87× |
| 步骤 3（+ τ = 8） | ≈ 0.97× | ≈ 1.03× | ≈ 0.96× | ≈ 0.93× |
| FA4 | ≈ 1.15× | ≈ 1.33× | ≈ 1.13–1.15× | ≈ 1.03–1.10× |

Nsight Compute（`perf`）：Tensor Core 忙碌 40 关 57% → 步骤 1 40% → 步骤 2 60% → 步骤 3 65%；`pipe_xu` 57% → 0.3% →
49% → 53%；`smsp__inst_executed_pipe_xu.sum` 步骤 2–3 为 40 关的 81%（104/128 的 P + α），验证了混合比例。

`perf_target` 取 0.85（最差的 `perf_8k` 约 0.93×，留出 cuDNN 波动的余量）。

## 误差（相对 FP64 oracle，同一份输入）

`exp2` 本身（GPU 上穷举 [−127, 8] 的全部 FP32 输入）：`MUFU.EX2` 1.46·10⁻⁷，3 次多项式 8.57·10⁻⁵（= 系数本身在 [0,1) 上的
误差，范围约简与拼接无额外误差），2 次 2.05·10⁻³，4 次 2.97·10⁻⁶；−∞ 与 −127 得 +0，(−127, −126) 得非负次正规数。

attention 输出：O 的最大误差四种变体几乎相同（由 O 的 16 位舍入主导）；LSE 的最大误差由全 MUFU 的约 10⁻⁶ 升到
10⁻⁵–1.3·10⁻⁴（≈ 多项式的相对误差），仍比 LSE 容差 10⁻³ 小一个数量级。逐 case 数字见讲义 §7.3。容差未改动。

## 验证

- `./lab check 41 --solution`：3 个 step × 30 个 case 全部通过（含新增的 `peaked_fp16`、`growing`、`growing_causal`、
  `jump`、`jump_causal`，以及 40 关审查后新增的 `tile1_position`、`causal_n_bound`）。
- 变异测试（在临时副本中，结束后骨架恢复为字节一致）：魔数加法改就近舍入 → LSE 超差（`one_block`、`medium` 等，误差
  ≈ 2–3·10⁻³）；c0 = 0.99999 → 有 mask 的 case 出 NaN；去掉钳位 → `n1` 错；条件与上一块最大值比较 → `growing`（FP16）
  出 NaN（`growing_causal` 为 BF16，不溢出，能通过）；从不移动基准 → `jump`、`jump_causal`、`growing` 出 NaN；投票后用
  lane 0 的 α → `growing`、`growing_causal`、`jump_causal` 错。
- 骨架：把 TODO(1)、(2)、(3a)、(3b) 依次换成答案代码后，`--step n` 通过、`--step n+1` 报 TODO；只填 3a 时 step 3 停在
  TODO(3)（correction warp 的 trap），不挂起。
- `./lab sass 41 --solution`：24 个 kernel STACK 均为 0；步骤 1 每个 kernel 1 条 `MUFU.EX2`（α）、256 条 `FFMA2`；
  步骤 2–3 每个 kernel 105 条 `MUFU.EX2`、100 条 `FFMA2`。
- `./lab sanitize --solution`：memcheck、racecheck、synccheck 在 `two_tiles`、`growing_causal`（步骤 3）和 `causal_tail`（步骤 1、2）
  上 0 错误；racecheck 看不见 TMA/TMEM 访问，同步与 40 关完全相同（40 关讲义 §5.6）。
- `./lab selftest 41` 通过；`./lab selftest 41 --perf`（目标 0.85）通过，该次实测 `perf_causal` 1.03×、`perf_d64` 0.96×、
  `perf_8k` 0.95×（`perf` 见上表 ≈ 0.97×）。
- `python -m pytest -q`：本关相关测试通过；唯一失败来自同时在写的 42 关（`course/42_fa4_2cta/README.md` 尚不存在）。

## 已知局限

- 混合比例对 D = 64/128、causal/非 causal 都用 k = 3，没有分别调；扫描每点只测一次。
- softmax 的寄存器上限 192 限制了 k；没有尝试改变寄存器分配（例如 correction 降到 64）来放开更大的 k。
- 依赖同一线程 MMA 的按序执行、causal 时 tile 0 多算一块等，与 40 关相同。
- 条件 rescale 对随机数据几乎不修正；对最大值持续上涨的输入（`growing_max`）仍需每 2–5 块修正一次。
