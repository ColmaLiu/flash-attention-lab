# s34_warp_specialize · warp specialization 生产消费

## 目的与联系

把搬运和计算交给不同 warp 角色，先保持单 stage，避免同时改变多种机制。

学习前置：`s33_tma`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

输入 X:[T,128,64] FP16，输出 Y 同 shape FP32，Y=2X+1。T 是独立数据 tile 的序列，不是线程数。

用 TMA producer 和独立 consumer 实现转换；自行决定角色、参与计数和 release 协议。

## 精确契约

X `[T,128,64]` FP16 contiguous；Y 同 shape FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(1, 128, 64)` | fp16 | False |
| `(3, 128, 64)` | fp16 | False |
| `(64, 128, 64)` | fp16 | False |
| `(257, 128, 64)` | fp16 | False |

case `0–3` 用于小规模调试；case `4–5` 用更长的 T 序列覆盖反复 buffer 复用和 phase wrap；T 是数据 tile 数，不是 buffer 数或 CTA 数。 `./lab check s34_warp_specialize` 默认验收全部 case。具体编号和输入模式见 `./lab task s34_warp_specialize -v`；规模设计见 [测试规模](../test_scales.md)。

案例模式：coordinate, random。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"x"` | `[T,128,64]` | FP16 | 输入 X；只读 |
| `outputs` | `"y"` | `[T,128,64]` | FP32 | Y = 2X + 1；T 是数据 tile 序列长度 |

CuTe 的 host 入口使用 `inputs["x"]` / `outputs["y"]`，值是 `torch.Tensor`；这不是 CUDA C++ 的 `TensorView`。转换为 CuTe 参数的方式由你设计。

`config` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cute/s34_warp_specialize/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s34_warp_specialize.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 3
./lab task s34_warp_specialize
./lab check s34_warp_specialize --case 0 --timeout 180
./lab check s34_warp_specialize --timeout 300
./lab sanitize s34_warp_specialize --tool memcheck --case 0
./lab sanitize s34_warp_specialize --tool racecheck --case 0
./lab sanitize s34_warp_specialize --tool synccheck --case 0
./lab bench s34_warp_specialize --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s34_warp_specialize --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s34_warp_specialize -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

画角色间依赖图，指出每个 buffer 被谁持有、何时可覆盖；说明退出前哪些异步动作必须完成。

## 性能实验与记录

比较同一任务统一 warp 分工与 specialization 的 register/SMEM 成本；不要求此小任务提速。

复制或填写 [本关实验记录](../../experiments/s34_warp_specialize.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

producer 等待永远不会发生的 release；consumer 读取上一 tile；角色分支跨越错误的 CTA 同步。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
