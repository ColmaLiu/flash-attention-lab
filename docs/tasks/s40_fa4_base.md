# s40_fa4_base · FA4 基础：受限 SM100 forward

## 目的与联系

先把 TMA/tcgen05/TMEM 与稳定 attention 接通，形成 FA4 实验基线；仅此关不称为完成 FA4。

学习前置：`s31_port`, `s38_async_gemm`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

数学定义与 s23 相同。BF16 Q/K/V/O，FP32 softmax 统计、MMA 累加与 LSE；D=128，非 causal，整块 N。

自行实现单 CTA mode 的分块 forward，设计 S/P/O 的 TMEM、SMEM、寄存器交接。允许串行依赖以先保证正确，不预填 layout 或 tile。

## 精确契约

Q/K/V 同 shape `[B,H,N,D]` contiguous；O 同 shape/dtype，LSE `[B,H,N]` FP32。输入 dtype 的舍入在生成时完成，数学参考基于已舍入输入。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(1, 1, 128, 128)` | bf16 | False |
| `(1, 2, 256, 128)` | bf16 | False |

| `(1, 1, 256, 128)` | bf16 | False |
| `(1, 2, 512, 128)` | bf16 | False |

案例模式：large, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cute/s40_fa4_base/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s40_fa4_base.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 4
./lab task s40_fa4_base
./lab check s40_fa4_base --case 0 --timeout 180
./lab check s40_fa4_base --timeout 300
./lab sanitize s40_fa4_base --tool memcheck --case 0
./lab sanitize s40_fa4_base --tool racecheck --case 0
./lab sanitize s40_fa4_base --tool synccheck --case 0
./lab bench s40_fa4_base --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s40_fa4_base --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s40_fa4_base -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

记录 QK 与 PV 的输入来源、TMEM accumulator 和精度转换；不得通过 DSL 包装现成 attention。

## 性能实验与记录

与 s23 在共同的 (1,1,128,128) BF16 shape 上比较；先画本实现的依赖时间线。

复制或填写 [本关实验记录](../../experiments/s40_fa4_base.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

TMEM S/P 覆盖冲突；PV 读到未发布的 P；自然指数与 exp2 的 scale 不一致。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
