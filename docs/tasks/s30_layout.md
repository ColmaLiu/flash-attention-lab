# s30_layout · CuTe DSL layout 与线程映射

## 目的与联系

独立进入 DSL：理解 shape/stride/layout 的域和值域，验证 JIT、DLPack 和线程映射。smoke 的标量写入不算完成此关。

学习前置：无；先运行 `./lab doctor` 与 `./lab smoke`。。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

输入 X:[R,C] row-major；输出 Y:[C,R] row-major；Y[j,i]=X[i,j]。CuTe layout 将逻辑坐标映射到线性偏移；线程 layout 是另一层映射。

实现 transpose；自行设计 layout、partition 和 grid/block。可先用普通 global load/store，再研究合并访问，不指定答案 layout。

## 精确契约

X `[R,C]` FP32 contiguous；Y `[C,R]` contiguous。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(17, 33)` | fp32 | False |
| `(32, 64)` | fp32 | False |

案例模式：coordinate, random。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"x"` | `[M,N]` | FP32 | 输入 X；只读 |
| `outputs` | `"y"` | `[N,M]` | FP32 | Y[j,i] = X[i,j]；独立 contiguous 输出 |

CuTe 的 host 入口使用 `inputs["x"]` / `outputs["y"]`，值是 `torch.Tensor`；这不是 CUDA C++ 的 `TensorView`。转换为 CuTe 参数的方式由你设计。

`config` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cute/s30_layout/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s30_layout.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 3
./lab task s30_layout
./lab check s30_layout --case 0 --timeout 180
./lab check s30_layout --timeout 300
./lab sanitize s30_layout --tool memcheck --case 0
./lab sanitize s30_layout --tool racecheck --case 0
./lab sanitize s30_layout --tool synccheck --case 0
./lab bench s30_layout --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s30_layout --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s30_layout -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

为一个小角落列出逻辑坐标、内存偏移和线程编号；证明一对一覆盖与尾部保护。

## 性能实验与记录

比较读合并与写合并的取舍；记录编译时间与重复执行时间，不把首次 JIT 纳入 kernel 耗时。

复制或填写 [本关实验记录](../../experiments/s30_layout.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

shape 正确但 stride 错误；转置两次；把线程编号误当 tensor offset；JIT 类型/动态参数混用。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
