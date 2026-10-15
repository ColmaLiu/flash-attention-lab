# s35_pipeline · 多缓冲流水线与 drain

## 目的与联系

在同一计算上加入循环缓冲，独立检验 prologue、steady state、epilogue 与 phase wrap。

学习前置：`s34_warp_specialize`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

Y=2X+1 不变。buffer 的占用区间从 producer acquire 开始，到最后一个 consumer release 结束；环形索引相同不意味着 phase 相同。

自行设计至少两种 stage 数的多缓冲流水线；选择配置字段，保留 s34 单缓冲入口。

## 精确契约

X `[T,128,64]` FP16 contiguous；Y 同 shape FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(1, 128, 64)` | fp16 | False |
| `(2, 128, 64)` | fp16 | False |
| `(3, 128, 64)` | fp16 | False |
| `(7, 128, 64)` | fp16 | False |
| `(17, 128, 64)` | fp16 | False |

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

允许修改：`students/cute/s35_pipeline/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s35_pipeline.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 3
./lab task s35_pipeline
./lab check s35_pipeline --case 0 --timeout 180
./lab check s35_pipeline --timeout 300
./lab sanitize s35_pipeline --tool memcheck --case 0
./lab sanitize s35_pipeline --tool racecheck --case 0
./lab sanitize s35_pipeline --tool synccheck --case 0
./lab bench s35_pipeline --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s35_pipeline --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s35_pipeline -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

测试 T=1/2/3/7/17 以及多次启动；提交生命周期图，证明没有提前覆盖和未完成退出。

## 性能实验与记录

固定计算量研究 stage 数、SMEM 占用和 overlap；给出性能下降的一个可验证解释。

复制或填写 [本关实验记录](../../experiments/s35_pipeline.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

短循环死锁；T 大于 ring 长度才出错；第一次正确第二次错误；末尾输出遗漏。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
