# s24_fa2_edges · FA2 低精度边界验证

## 目的与联系

在 FA2 Tensor Core 版本扩展尾块和 causal，形成可供 DSL 移植比较的受限低精度基线。

学习前置：`s23_fa2_tensor`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

矩阵指令的物理 tile 与有效的 N 不同；只有有效且未 mask 的 pair 参与 softmax 的分母。

完成低精度 MMA tile 的合法 padding、mask 和写回，不让无效元素改变统计。

## 精确契约

Q/K/V 同 shape `[B,H,N,D]` contiguous；O 同 shape/dtype，LSE `[B,H,N]` FP32。输入 dtype 的舍入在生成时完成，数学参考基于已舍入输入。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(1, 1, 1, 64)` | bf16 | False |
| `(1, 1, 1, 64)` | bf16 | True |
| `(1, 1, 1, 64)` | fp16 | False |
| `(1, 1, 1, 64)` | fp16 | True |
| `(1, 2, 129, 64)` | bf16 | False |
| `(1, 2, 129, 64)` | bf16 | True |
| `(1, 2, 129, 64)` | fp16 | False |
| `(1, 2, 129, 64)` | fp16 | True |
| `(2, 2, 65, 64)` | bf16 | False |
| `(2, 2, 65, 64)` | bf16 | True |
| `(2, 2, 65, 64)` | fp16 | False |
| `(2, 2, 65, 64)` | fp16 | True |

案例模式：large, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cuda/s24_fa2_edges/entry.py`、`students/cuda/s24_fa2_edges/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s24_fa2_edges.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 2
./lab task s24_fa2_edges
./lab check s24_fa2_edges --case 0 --timeout 180
./lab check s24_fa2_edges --timeout 300
./lab sanitize s24_fa2_edges --tool memcheck --case 0
./lab sanitize s24_fa2_edges --tool racecheck --case 0
./lab sanitize s24_fa2_edges --tool synccheck --case 0
./lab bench s24_fa2_edges --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s24_fa2_edges --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s24_fa2_edges -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

输出 N=1 与 N=65/129 的最大误差；说明合法 MMA shape 与逻辑边界的区别。

## 性能实验与记录

比较 FP16/BF16 的 near-zero 误差与大 logits；对 causal 分开统计有效 FLOPs 和实际指令工作。

复制或填写 [本关实验记录](../../experiments/s24_fa2_edges.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

尾块仍使用未初始化 shared 数据；mask 在指数之后处理；首行输出不等于第一个 V。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
