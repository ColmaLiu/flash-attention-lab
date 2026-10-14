# s38_async_gemm · TMA + tcgen05 流水线 GEMM

## 目的与联系

把多缓冲和异步 MMA 合并，在简单 GEMM 上确认不同执行引擎的同步，再进入 attention。

学习前置：`s35_pipeline`, `s37_tcgen05`。学习依赖不等于强制导入；默认每关独立实现。若复用自己前关代码，在 `RUNTIME_REQUIRES` 声明具体 ID，未实现时会报告缺哪个模块。

## 数学定义与推导任务

C=A B，K 被分块；每个累加片段必须包含所有 K 项恰好一次。SMEM buffer 只有在其最后一个 MMA 读者完成后才能复用。

实现 pipelined GEMM，不复用官方 GEMM 内核。自行安排 TMA→MMA 与 MMA→SMEM release 的跨引擎协议。

## 精确契约

表内 shape 为 `(M,N,K)`；A `[M,K]`、B `[K,N]` row-major contiguous，C `[M,N]` FP32。 只支持下表组合，有限输入，无空维度；没有 NaN/Inf 输入语义。attention 无 dropout/bias/GQA，自注意力 forward。scale 为有限正标量，默认 `1/sqrt(D)`，另测 0.25；mask 为下三角且包含对角线。

| shape | dtype | causal |
|---|---|---|
| `(128, 128, 256)` | bf16 | False |
| `(128, 128, 256)` | fp16 | False |
| `(128, 128, 512)` | bf16 | False |
| `(128, 128, 512)` | fp16 | False |

案例模式：cancellation, random, zero。FP32 累加；copy/transpose 必须逐元素精确。容差与近零处理见 [数值规则](../numerics.md)，不是自行提高阈值的许可。CLI `--case` 是 `curriculum.json` 中本关 cases 的零基索引。

## 学生文件与接口

允许修改：`students/cute/s38_async_gemm/entry.py`，也可在本关目录加入自己的 `.cuh`/Python helper，以及 `experiments/s38_async_gemm.md`。公共设施若需扩展，应另标明基础设施修改并保留其他关入口。

完成 `prepare(inputs, outputs, workspace, config, stream)` 以及你自己定义的 `@cute.kernel`/`@cute.jit`；返回无参数 launcher。DLPack 包装、compile 可在 prepare 做，设备 layout 与 launch 不预设。

`workspace_spec` 返回全局 scratch 规格；host 只负责分配，**不初始化算法状态**。所有输出必须由学生写入；每次 launch 必须能独立重复运行，不能依赖上次残留。完成后手动设置 `IMPLEMENTED=True`；这只开启入口，不表示正确性通过。默认 stream 参数是 PyTorch 当前 stream，禁止硬编码 stream 0。

## 可执行验收

```bash
./lab doctor --stage 3
./lab task s38_async_gemm
./lab check s38_async_gemm --case 0 --timeout 180
./lab check s38_async_gemm --timeout 300
./lab sanitize s38_async_gemm --tool memcheck --case 0
./lab sanitize s38_async_gemm --tool racecheck --case 0
./lab sanitize s38_async_gemm --tool synccheck --case 0
./lab bench s38_async_gemm --case 0 --warmup 5 --repeats 7 --iterations 10
./lab profile s38_async_gemm --case 0
```

未完成时 check/bench 非零退出（2），明确 `STUDENT_UNIMPLEMENTED`；环境问题退出 4，正确性失败退出 3，超时退出 5。首次 CUDA build/DSL JIT 允许更长 timeout，但不能无限等待。测试共享入口也可执行 `./lab python -m pytest tests/student --exercise s38_async_gemm -q`；该入口自动使用与 `./lab check` 相同的 Conda、单 GPU 和 Toolkit 配置。

**完成标准分两层：** 所有契约案例在普通/非默认 stream、重复调用、输出预填 NaN 的条件下通过且输入未变；适用 sanitizer 无错误。再完成以下设计证据，AI review 后才记为本关学习完成。数值通过并不自动证明算法/流水线结构。

提交每个 buffer 的生产/消费/释放图；说明至少两个 stage 的实际重叠证据，单看 launch 个数不够。

## 性能实验与记录

固定 K，比较单 stage 与多 stage；记录占用、SMEM bytes 和 stall，用 profiler 判断瓶颈而非猜测。

复制或填写 [本关实验记录](../../experiments/s38_async_gemm.md)。记录源代码 hash、GPU、软件、shape、dtype、scale、mask、配置、误差、median/min/max/stdev、raw artifact 与未解决问题。bench 先运行本 case 的完整检查；全关 check 是通关依据。未实现代码不存在真实性能。

## 调试与常见症状

K=64 正确而 K=512 失败；TMA 覆盖正在消费的数据；累加 flag 错误；drain 漏最后一项。

先 `--case 0`；失败时从报告的 case/index 固定随机种子复现，再缩到契约中最小 shape，记录第一次错误的坐标。用 `CUDA_LAUNCH_BLOCKING=1 ./lab check ... --case ...` 定位错误 launch；该模式不用于性能结论。超时终止的是本次子进程，不是其他 GPU 用户进程。参见 [调试](../debugging.md) 与 [Blackwell 接口](../blackwell.md)。
