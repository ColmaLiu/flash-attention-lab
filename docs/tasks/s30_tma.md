# s30_tma · TMA 与生产消费协议

阶段 3。前置：[s21_pipeline](s21_pipeline.md)

## 把搬运任务交给独立引擎

TMA 用 host 编码的 tensor descriptor 表达多维数据，减少线程逐元素计算地址与发起 copy 的工作。它并不消除对布局、对齐和边界的推理。输入描述符的 stride 合法性必须与实际连续存储一致，输出转置的 ownership 仍需由 kernel 决定。

本关沿用 s00 的数学语义，让你专注于同步协议：一个事务被发起、写入完成、对消费者可见、随后 buffer 可复用，是相互关联但不同的事件。mbarrier 的参与计数、事务字节和 phase 必须对应实际工作。短输入、多轮复用和尾部都需要验证；单次 copy 成功不足以证明协议正确。

## 定义、代码与参考

y[j,i]=x[i,j]，逐元素精确。为合法 TMA 行步长，本关 FP32 输入列数要求是 4 的倍数；行数可有尾部。

接口：`[R,C] → [C,R]`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s30_tma/kernel.cu](../../students/cuda/s30_tma/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s30_tma/](../../solutions/s30_tma/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；attention 使用 FP64 的直接数学定义，并按独立 query 行限制临时内存。
- 快速库基线：PyTorch 原生 CUDA 算子，预分配输出；它是可运行的性能比较对象。

源码阅读：PTX ISA cp.async.bulk.tensor、mbarrier；本地 solutions/s30_tma/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，上游答案的对应模块见 [references](../../references/README.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(64, 128)` | fp32/full | 单次传输与输出 ownership。 |
| `coordinates` | edge | `(257, 132)` | fp32/full | 满足描述符行步长对齐的行尾/列尾，检查边界。 |
| `short` | edge | `(1, 128)` | fp32/full | 短任务的初始化与 drain。 |
| `reuse` | stress | `(4097, 256)` | fp32/full | 多次 stage 循环，触发 phase/buffer 复用错误。 |
| `bandwidth` | perf | `(2048, 2048)` | fp32/full | 与 s00 相同 shape 对照异步传输开销。 |
| `wide` | perf | `(1024, 8192)` | fp32/full | 与 s00 相同长行，比较吞吐。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

与 s00 同形状及 native transpose 比较，解释短任务与稳态的不同成本。

描述事务完成与数据可见性的区别；追踪至少一次 buffer/phase 回绕。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。机械正确性不自动证明使用了目标机制，也不要求初学版本在所有 shape 打败厂商库。

```bash
./lab task s30_tma -v
./lab reference s30_tma --case 0
./lab solution s30_tma --case 0
./lab check s30_tma --case 0
./lab check s30_tma
./lab bench s30_tma --case 0
./lab sanitize s30_tma --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
