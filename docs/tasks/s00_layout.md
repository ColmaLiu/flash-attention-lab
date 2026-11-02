# s00_layout · 内存、线程与转置

阶段 0。前置：无；先完成环境和基础设施 smoke。

## 为什么从转置开始

GPU 执行的是许多线程的内存请求。一个线程连续访问，并不意味着一个 warp 的同一条指令访问相邻地址；后者才决定事务能否合并。转置保持元素数量不变，却交换连续方向，因此能把计算开销几乎排除，让你直接观察地址映射的代价。先在纸上写出行主序输入与输出的逻辑坐标关系，再检查同一时刻各 lane 的地址。

shared memory 可以把读写两侧的分工解耦，但它也有 bank 冲突和同步成本。更大的 tile 会增加复用机会，也可能增加 shared memory 占用并减少驻留 CTA。不要先背一个 tile 尺寸；先明确谁写、谁读、何时可复用。

## 定义、代码与参考

y[j,i] = x[i,j]。要求逐元素精确，不涉及浮点误差预算。

接口：`[R,C] → [C,R]`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s00_layout/kernel.cu](../../students/cuda/s00_layout/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s00_layout/](../../solutions/s00_layout/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；使用独立张量转置定义检查元素与坐标。
- 快速库基线：PyTorch 原生 CUDA 算子，预分配输出；它是可运行的性能比较对象。

源码阅读：NVIDIA CUDA Best Practices 的 coalesced global access、shared memory bank conflicts；本地完整答案 solutions/s00_layout/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(7, 11)` | fp32/full | 矩形转置，明确输入/输出的线性地址。 |
| `coordinates` | edge | `(33, 65)` | fp32/full | 跨 warp 边界和尾部，检查每个坐标恰好写入一次。 |
| `single_row` | edge | `(1, 257)` | fp32/full | 退化维度，区分 transpose 与 copy。 |
| `tall` | stress | `(1025, 17)` | fp32/full | 长矩形及尾部，观察映射覆盖。 |
| `bandwidth` | perf | `(2048, 2048)` | fp32/full | 较大矩阵的全局访存效率。 |
| `wide` | perf | `(1024, 8192)` | fp32/full | 长行及不同宽高比；报告有效字节率而非假定 HBM 带宽。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

预测哪种访问会增加事务，再用 profiler 和同形状 native transpose 对照。

说明 ownership、全局访存方向及 shared-memory bank 行为；提供至少两种映射的实测对比。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s00_layout -v
./lab reference s00_layout --case 0
./lab solution s00_layout --case 0
./lab check s00_layout --case 0
./lab check s00_layout
./lab bench s00_layout --case 0
./lab sanitize s00_layout --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
