# s46_fa4_schedule · FA4：调度与资源控制实验

## 目的与联系

用固定功能比较 CTA 调度与资源分配，补齐论文 scheduling 思考；无需达到官方性能。

学习前置：`s45_fa4_edges`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

每个合法输出 query tile 恰好执行一次；调度改变执行顺序，不能改变数学依赖。causal 下不同 query tile 的有效 key 工作量不同。

实现并比较至少两种你选择的 tile 调度/资源配置；持久化 CTA 可选，不预先规定 scheduler。保留前一版。

## 精确契约

Q/K/V 同 shape `[B,H,N,D]` contiguous；O 同 shape/dtype，LSE `[B,H,N]` FP32。输入 dtype 的舍入在生成时完成，数学参考基于已舍入输入。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(1, 1, 256, 128)` | bf16 | False |
| `(1, 1, 256, 128)` | bf16 | True |
| `(2, 4, 512, 128)` | bf16 | False |
| `(2, 4, 512, 128)` | bf16 | True |

案例模式：large, random。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cute/s46_fa4_schedule/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s46_fa4_schedule.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 4
./lab task s46_fa4_schedule
./lab check s46_fa4_schedule --case 0 --timeout 180
./lab check s46_fa4_schedule --timeout 300
./lab sanitize s46_fa4_schedule --tool memcheck --case 0
./lab sanitize s46_fa4_schedule --tool racecheck --case 0
./lab sanitize s46_fa4_schedule --tool synccheck --case 0
./lab bench s46_fa4_schedule --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s46_fa4_schedule --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s46_fa4_schedule -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

给出无遗漏/无重复 ownership 证明；源码说明调度策略，测量 tail 与资源占用，而非仅给总时间。

## 性能实验与记录

固定 shape/dtype/数学路径比较调度；记录是否受 CTA 尾部、register、SMEM 或 TMEM 容量限制。官方结果只作独立对照。

复制或填写 [本关实验记录](../../experiments/s46_fa4_schedule.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

persistent 循环漏任务；causal 工作分配失衡；resource 配置导致 launch 失败；多 CTA 写同一输出。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
