# s01_reduce · 独立 reduction 与尾部元素

## 目的与联系

把第一关的 reduction 独立出来，加入非整 warp 和非整块长度，为后续分块状态验证提供基础。

学习前置：`s00_softmax`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

maximum_r=max_j x_rj；sum_r=Σ_j x_rj。空参与线程不等于真实的零输入；本关没有空行。浮点加法不满足结合律。

实现行最大值与行总和两个输出；选择线程分工、局部/跨线程 reduction 及尾部谓词。

## 精确契约

X `[R,C]` FP32 contiguous；maximum、sum 均 `[R]` FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(1, 1)` | fp32 | False |
| `(3, 257)` | fp32 | False |
| `(4, 128)` | fp32 | False |
| `(7, 31)` | fp32 | False |
| `(31, 4097)` | fp32 | False |
| `(128, 2048)` | fp32 | False |
| `(128, 8192)` | fp32 | False |

case `0–15` 用于小规模调试；case `16–18` 覆盖更宽的行、多行并行与尾部长度，检查 reduction/状态更新能否跨越更多数据块。 `./lab check s01_reduce` 默认验收全部 case。具体编号和输入模式见 `./lab task s01_reduce -v`；规模设计见 [测试规模](../test_scales.md)。

案例模式：cancellation, negative, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"x"` | `[R,C]` | FP32 | 输入 X；只读 |
| `outputs` | `"maximum"` | `[R]` | FP32 | 每行 max_j X[r,j] |
| `outputs` | `"sum"` | `[R]` | FP32 | 每行原始元素之和 sum_j X[r,j]；不是指数和 |

C++ 使用 `inputs.at("x")` / `outputs.at("maximum")` 取得 `TensorView`；维度从 `shape` 读取，数据地址在 `data`，dtype 使用 `DType` 枚举。输入只读，输出指针可写；`const TensorMap&` 限制的是描述符，不是输出显存。

`options` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。C++ 的 `Options` 值统一为 `double`。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cuda/s01_reduce/entry.py`、`students/cuda/s01_reduce/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s01_reduce.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 0
./lab task s01_reduce
./lab check s01_reduce --case 0 --timeout 180
./lab check s01_reduce --timeout 300
./lab sanitize s01_reduce --tool memcheck --case 0
./lab sanitize s01_reduce --tool racecheck --case 0
./lab sanitize s01_reduce --tool synccheck --case 0
./lab bench s01_reduce --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s01_reduce --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s01_reduce -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

说明所有有效元素恰好参与一次；记录 negative 输入和单元素输入的边界处理；不规定一个 CTA 必须对应一行。

## 性能实验与记录

比较宽度 31、128、257 的通信与空闲线程；用 NCU LaunchStats/MemoryWorkloadAnalysis 解释实际布局访问。

复制或填写 [本关实验记录](../../experiments/s01_reduce.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

负数行最大值变成 0；C=257 漏最后一个值；取消相加时误差异常；线程不一致到达同步点。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
