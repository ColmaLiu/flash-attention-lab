# s11_fa1_edges · FA1：尾块与 causal 扩展

## 目的与联系

保留 s10 独立入口，给 FA1 增加 B/H/D 扩展、非整块 N 和 causal，用于与后续 FA2 控制变量比较。

学习前置：`s10_fa1`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

causal 只允许 j≤i；被屏蔽位置数学分数为 -∞。self-attention 的对角线有效，整个有效 query 行不会全屏蔽，但某个 KV tile 可能全屏蔽。

在新入口实现边界和 mask 规则；选择对全屏蔽 tile 和尾部元素的处理，不能改变 s10 的结果记录。

## 精确契约

Q/K/V 同 shape `[B,H,N,D]` contiguous；O 同 shape/dtype，LSE `[B,H,N]` FP32。输入 dtype 的舍入在生成时完成，数学参考基于已舍入输入。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(1, 1, 1, 32)` | fp32 | False |
| `(1, 1, 1, 32)` | fp32 | True |
| `(1, 2, 129, 32)` | fp32 | False |
| `(1, 2, 129, 32)` | fp32 | True |
| `(2, 3, 65, 64)` | fp32 | False |
| `(2, 3, 65, 64)` | fp32 | True |

案例模式：large, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cuda/s11_fa1_edges/entry.py`、`students/cuda/s11_fa1_edges/kernel.cu`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s11_fa1_edges.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `student_launch(inputs, outputs, workspace, options, stream)`；设备端模块和 launch 均为空。TensorView.data 是设备指针，shape 是逻辑形状，DType 见 `flash_attention_lab/cuda_api.h`。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 1
./lab task s11_fa1_edges
./lab check s11_fa1_edges --case 0 --timeout 180
./lab check s11_fa1_edges --timeout 300
./lab sanitize s11_fa1_edges --tool memcheck --case 0
./lab sanitize s11_fa1_edges --tool racecheck --case 0
./lab sanitize s11_fa1_edges --tool synccheck --case 0
./lab bench s11_fa1_edges --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s11_fa1_edges --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s11_fa1_edges -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

区分越界 padding 与有效但被 mask 的元素；展示 N=1、65、129 和 causal/noncausal 的通过结果。

## 性能实验与记录

分别报告实际有效 pair FLOPs 和 dense-equivalent FLOPs；分析对角 tile 的额外工作。

复制或填写 [本关实验记录](../../experiments/s11_fa1_edges.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

全屏蔽 tile 产生 -∞ 减 -∞；三角形方向反了；尾部 global 越界；不同 head 互相污染。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
