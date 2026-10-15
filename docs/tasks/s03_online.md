# s03_online · online softmax 与状态不变量

## 目的与联系

从完整行转向分块统计，为 FA1 的跨块更新做准备；此关完成不等于完成 FA1。

学习前置：`s00_softmax`, `s01_reduce`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

对已处理的索引集合 A，m_A=max_{j∈A}x_j，l_A=Σ_{j∈A}exp(x_j-m_A)。用不变量推导不相交集合合并后同一归一化基准下的表示。最终 y 是整行 softmax。

实现按块处理的 online 统计和最终归一化输出 y,m,l。块大小、扫描与输出组织均由你决定，不能退回一次性整行统计冒充 online。

## 精确契约

X `[R,C]` FP32；Y 同 shape；m/l `[R]` FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(2, 513)` | fp32 | False |
| `(3, 257)` | fp32 | False |
| `(4, 128)` | fp32 | False |

案例模式：large, negative, random, shifted。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"x"` | `[R,C]` | FP32 | 输入 X；只读 |
| `outputs` | `"y"` | `[R,C]` | FP32 | 完整行的 softmax 概率 |
| `outputs` | `"m"` | `[R]` | FP32 | 最终整行最大值 max_j X[r,j] |
| `outputs` | `"l"` | `[R]` | FP32 | 最终整行 sum_j exp(X[r,j] - m[r]) |

C++ 使用 `inputs.at("x")` / `outputs.at("y")` 取得 `TensorView`；维度从 `shape` 读取，数据地址在 `data`，dtype 使用 `DType` 枚举。输入只读，输出指针可写；`const TensorMap&` 限制的是描述符，不是输出显存。

`options` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。C++ 的 `Options` 值统一为 `double`。

`m/l` 是本次计算结束后的整行状态，不是某个中间 tile 的状态；中间状态的存储位置由你设计。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cuda/s03_online/entry.py`、`students/cuda/s03_online/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s03_online.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 0
./lab task s03_online
./lab check s03_online --case 0 --timeout 180
./lab check s03_online --timeout 300
./lab sanitize s03_online --tool memcheck --case 0
./lab sanitize s03_online --tool racecheck --case 0
./lab sanitize s03_online --tool synccheck --case 0
./lab bench s03_online --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s03_online --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s03_online -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

给出跨块状态不变量及初始化依据；至少演示两个不同块大小以及尾块。无需在 API 中返回全部中间状态。

## 性能实验与记录

固定输入比较块大小对舍入误差、读入次数和存储量的影响；解释输出概率是否要求再读取输入。

复制或填写 [本关实验记录](../../experiments/s03_online.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

新块最大值更大时旧统计失效；尾块影响分母；只归一化了最后一块。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
