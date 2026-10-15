# s22_mma · CUDA Tensor Core 小练习

## 目的与联系

独立学习 warp 级 Tensor Core 输入/输出组织，再把低精度引入 attention。

学习前置：`s01_reduce`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

C=A B，A:[M,K]，B:[K,N]，C:[M,N]。A/B 为 FP16 或 BF16；乘积累加与 C 为 FP32，不做 TF32。

自行选择 WMMA 或 mma.sync 的合法指令形状、fragment layout、加载和存储。不可调用 cuBLAS，也不预填 launch 和 SMEM。

## 精确契约

表内 shape 为 `(M,N,K)`；A `[M,K]`、B `[K,N]` row-major contiguous，C `[M,N]` FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。本关不应用 attention 的 scale 或 causal mask。

| shape | dtype | causal |
|---|---|---|
| `(16, 16, 16)` | bf16 | False |
| `(16, 16, 16)` | fp16 | False |
| `(32, 32, 64)` | bf16 | False |
| `(32, 32, 64)` | fp16 | False |

案例模式：cancellation, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

### 参数字典：精确键名

键名区分大小写，以下是本关完整的输入/输出集合；数学符号 `X/Y/Q/K/V` 不能直接用作字典键。所有 tensor 均位于同一 GPU、连续存储，最后一维连续；shape 描述完整逻辑 tensor，不预先规定 tile、CTA 或 warp 的划分。

| 参数 | 字符串键 | shape | dtype | 含义 |
|---|---|---|---|---|
| `inputs` | `"a"` | `[M,K]` | BF16 / FP16 | 左矩阵 A；只读 |
| `inputs` | `"b"` | `[K,N]` | BF16 / FP16 | 右矩阵 B；只读；接口不传入转置后的 B |
| `outputs` | `"c"` | `[M,N]` | FP32 | C = A B；无额外初值或累加输入 |

C++ 使用 `inputs.at("a")` / `outputs.at("c")` 取得 `TensorView`；维度从 `shape` 读取，数据地址在 `data`，dtype 使用 `DType` 枚举。输入只读，输出指针可写；`const TensorMap&` 限制的是描述符，不是输出显存。

`options` 中保留公共默认字段 `scale=1.0`、`causal=False`（C++ 中为 `0.0`），它们不参与本关数学运算，不应据此增加缩放或 mask。C++ 的 `Options` 值统一为 `double`。

支持域表中的 `(M,N,K)` 是 GEMM 问题尺寸；实际传入的两个 tensor 形状分别为 `[M,K]` 和 `[K,N]`。

`workspace` 没有预设键，初始为空；其键、shape、dtype 完全来自本关 `workspace_spec`。输出和 workspace 已由 host 分配，但算法状态未经初始化。自定义数值配置来自 `entry.CONFIG` 或调用参数；shape 不会自动作为配置键传入。完整参数合并和类型约定见 [公共接口](../contracts.md)。

## 学生文件与接口

允许修改：`students/cuda/s22_mma/entry.py`、`students/cuda/s22_mma/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s22_mma.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 2
./lab task s22_mma
./lab check s22_mma --case 0 --timeout 180
./lab check s22_mma --timeout 300
./lab sanitize s22_mma --tool memcheck --case 0
./lab sanitize s22_mma --tool racecheck --case 0
./lab sanitize s22_mma --tool synccheck --case 0
./lab bench s22_mma --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s22_mma --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s22_mma -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

在 B200 生成代码中确认 Tensor Core 指令；说明 collective 参与线程和对齐约束；不要把 CUDA WMMA 当作 Blackwell tcgen05。

## 性能实验与记录

对同一已量化输入测误差，分开记录量化与累加误差；记录选择的指令与 SM100 编译目标。

复制或填写 [本关实验记录](../../experiments/s22_mma.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

fragment 按普通连续矩阵访问；参与线程不完整；B layout 方向错误；输出被低精度累加。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
