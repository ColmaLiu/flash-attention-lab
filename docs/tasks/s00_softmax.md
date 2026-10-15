# s00_softmax · 固定契约稳定 row-wise softmax

## 目的与联系

第一关只处理 4 行、每行 128 个 FP32 数，建立从 CUDA 源码到误差报告的闭环。性能不是通关门槛。

学习前置：无；先运行 `./lab doctor` 与 `./lab smoke`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

对每行 x，m=max(x)，y_j=exp(x_j-m)/Σ_k exp(x_k-m)。对任意有限常数 c，softmax(x+c)=softmax(x)。从分子分母同时乘 exp(-m) 推导稳定性；不要求先支持所有宽度。

自行设计 reduction、线程到行/元素的映射、存储位置和 launch。实现完整的一行稳定 softmax 模块；不得调用现成 softmax。

## 精确契约

X `[R,C]` FP32 contiguous；Y 同 shape/dtype。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(4, 128)` | fp32 | False |

案例模式：constant, large, negative, random, shifted, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"x"` | `[4,128]` | FP32 | 输入 X；只读 |
| `outputs` | `"y"` | `[4,128]` | FP32 | 逐行 softmax 概率 |

C++ 使用 `inputs.at("x")` / `outputs.at("y")` 取得 `TensorView`；维度从 `shape` 读取，数据地址在 `data`，dtype 使用 `DType` 枚举。输入只读，输出指针可写；`const TensorMap&` 限制的是描述符，不是输出显存。

`options` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。C++ 的 `Options` 值统一为 `double`。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cuda/s00_softmax/entry.py`、`students/cuda/s00_softmax/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s00_softmax.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 0
./lab task s00_softmax
./lab check s00_softmax --case 0 --timeout 180
./lab check s00_softmax --timeout 300
./lab sanitize s00_softmax --tool memcheck --case 0
./lab sanitize s00_softmax --tool racecheck --case 0
./lab sanitize s00_softmax --tool synccheck --case 0
./lab bench s00_softmax --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s00_softmax --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s00_softmax -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

画出一行数据由哪些线程拥有；解释最大值和分母如何传播。所有输出非负、每行和约为 1，偏移 10000 仍有限。

## 性能实验与记录

记录一次正确实现的耗时与 launch 配置；解释小规模运行为何可能由 launch 开销主导。比较 constant 和 large 输入误差，不以速度决定正确性。

复制或填写 [本关实验记录](../../experiments/s00_softmax.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

大正数输入出现 NaN/Inf；每行和不是 1；只有第一行正确；运行之间随机变化。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
