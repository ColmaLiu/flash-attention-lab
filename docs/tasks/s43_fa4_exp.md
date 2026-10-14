# s43_fa4_exp · FA4：指数吞吐优化接入

## 目的与联系

在已经正确的流水线中加入指数机制，观察 B200 非矩阵执行资源的约束。

学习前置：`s41_fa4_pipeline`, `s42_exp`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

exp(s)=2^(s log2(e))；attention 的 scale、最大值与缩放状态必须采用一致的底数。

自行把经过独立误差验证的指数实现接入 softmax，并选择软件/硬件计算比例；保持 s41 独立入口。

## 精确契约

Q/K/V 同 shape `[B,H,N,D]` contiguous；O 同 shape/dtype，LSE `[B,H,N]` FP32。输入 dtype 的舍入在生成时完成，数学参考基于已舍入输入。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(1, 1, 256, 128)` | bf16 | False |
| `(1, 2, 512, 128)` | bf16 | False |

案例模式：large, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cute/s43_fa4_exp/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s43_fa4_exp.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 4
./lab task s43_fa4_exp
./lab check s43_fa4_exp --case 0 --timeout 180
./lab check s43_fa4_exp --timeout 300
./lab sanitize s43_fa4_exp --tool memcheck --case 0
./lab sanitize s43_fa4_exp --tool racecheck --case 0
./lab sanitize s43_fa4_exp --tool synccheck --case 0
./lab bench s43_fa4_exp --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s43_fa4_exp --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s43_fa4_exp -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

解释近似误差如何影响 softmax 分母、P 转换和 O；提供与 s41 相同输入的误差分布。

## 性能实验与记录

固定 pipeline 和 tile，仅改变指数实现；检查 register 增加或 FMA 争用是否抵消收益。

复制或填写 [本关实验记录](../../experiments/s43_fa4_exp.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

exp 与 exp2 的系数乘两次；某些分支精度不同；指数优化后 spills 导致性能下降。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
