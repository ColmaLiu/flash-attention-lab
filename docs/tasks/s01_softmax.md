# s01_softmax · 行归约与稳定 softmax

阶段 0。前置：[s00_layout](s00_layout.md)

先读：[从三个算子到流式 attention](../attention-math.md)，包含可手算的例子、状态合并推导及精度边界。

## 从局部值到一行共同的归一化

softmax 的输出依赖整行的最大值和总和，因此它不是逐元素 exp。线程需要合作传播信息，而不同宽度改变参与者数量、每个线程持有的值和通信成本。先考虑一行只有一个元素，再考虑不足一个 warp、刚超过一个 warp及很宽的行。无效线程不能作为真实输入参与最大值。

直接求 exp(x) 容易溢出；整体减去同一常数不改变归一化结果。最大值是保证所有指数参数非正的实用选择。求和次序会影响舍入，所以数值分析和线程划分要分别记录。宽行时性能可能由多次读取、寄存器压力或归约通信决定。

## 定义、代码与参考

m=max_j x_j；y_j=exp(x_j-m)/sum_k exp(x_k-m)。对有限常数 c，有 softmax(x+c)=softmax(x)。

接口：`[R,C] → [R,C]`。输入必须在同一 GPU 上连续存储。精确支持域是 [curriculum.json](../../curriculum.json) 中本关的 `domain`，测试例是采样而非 shape 白名单。所有维度为正；输入有限。

- 学生入口：[students/cuda/s01_softmax/kernel.cu](../../students/cuda/s01_softmax/kernel.cu)，host workspace 声明在同目录 `entry.py`。
- 完整可读答案：[solutions/s01_softmax/](../../solutions/s01_softmax/)。可以先独立实现，也可以阅读答案后做控制变量实验；学生入口不会调用答案。
- 独立数学 oracle：[reference.py](../../flash_attention_lab/reference.py) 的 `mathematical_reference`；使用 FP64 显式指数与归一化检查逐行 softmax。
- 快速库基线：PyTorch 原生 CUDA 算子，预分配输出；它是可运行的性能比较对象。

源码阅读：PyTorch ATen/native/cuda/SoftMax.cu 的 dispatch 与 persistent softmax；本地 solutions/s01_softmax/kernel.cu。 固定版本与官方链接见 [sources](../sources.md)，本地答案的设备源码入口见 [参考实现与源码](../reference-implementations.md)。

答案按行宽选择三条原生路径。宽度不超过 1024 时，一个 warp 处理一行，每个 CTA 包含四行；1025–16384 时，每行使用 256 线程，将输入和指数结果保存在寄存器中，每个元素只读一次、求一次指数。八个 warp 通过 shared memory 交换局部最大值与分母，复用这块 shared memory 前有完整 CTA 同步。更宽的行使用多次读取的通用路径，避免寄存器数量随行宽无界增长。这个取舍需要在短、中、宽三个性能案例上分别检查，不能把减少读取直接当作实测加速。

## 为什么选择这些测试

| case ID | 组 | shape | dtype/mask | 要排查的问题 |
|---|---|---|---|---|
| `small` | debug | `(3, 37)` | fp32/full | 跨 warp 非整倍数行，建立稳定 softmax。 |
| `single` | edge | `(7, 1)` | fp32/full | 宽度一应输出一。 |
| `negative` | edge | `(13, 129)` | fp32/full | 全负输入，识别错误最大值初始化。 |
| `shift` | stress | `(17, 257)` | fp32/full | 大整体偏移，检查 exp 前的稳定化。 |
| `uniform` | debug | `(5, 33)` | fp32/full | 均匀分布和每行和为一。 |
| `wide_tail` | stress | `(31, 8193)` | fp32/full | 宽行、多次处理及尾部。 |
| `many_short` | perf | `(4096, 128)` | fp32/full | 短行多行的并行度及 launch 成本。 |
| `medium` | perf | `(1024, 2048)` | fp32/full | 中等行宽的归约和访存平衡。 |
| `wide` | perf | `(128, 8192)` | fp32/full | 宽行的资源与归约成本。 |

`debug` 用于缩小问题，`edge` 检验边界，`stress` 检验数值或生命周期，`perf` 选择代表性工作负载；所有声明案例都必须通过正确性验收。case 0 是第一行；用 `task -v` 查看序号。

## 实验与完成标准

在短/中/宽行三个工作负载上比较 native softmax，不用一个固定宽度概括吞吐。

给出最大值、分母的参与集合及有效线程边界；解释短行和宽行的资源差异。

完成标准包含三项：全关数值/stream/重复调用检查通过；相关 memcheck/racecheck/synccheck 无问题；提交源码与设计证据，以及相同输入下答案、学生、快速库的计时。用源码确认目标机制，再用各个 perf case 的实测差距确定下一步优化。

```bash
./lab task s01_softmax -v
./lab reference s01_softmax --case 0
./lab solution s01_softmax --case 0
./lab check s01_softmax --case 0
./lab check s01_softmax
./lab bench s01_softmax --case 0
./lab sanitize s01_softmax --tool memcheck --case 0
```

先运行小例，再选择表中 perf 案例比较速度；不要把 debug 微小输入的 launch 时间当成吞吐。计时范围、graph/eager 与不等价比较见 [benchmarking](../benchmarking.md)。把观察记入忽略目录，使用 [实验模板](../../experiments/README.md)。
