# CUDA/PTX：指数与延迟 rescale 的数值实验

[kernel.cu](kernel.cu) 默认选择 `Overlap=true, Numerics=true, Persistent=false, Polynomial=false`，复用 [SM100 attention](../s40_attention/sm100_attention.cuh)。输入域与 s41 相同。数学目标仍是相同 attention，O 与自然对数 LSE 沿用原有容差。

## 指数与状态的尺度

普通路径先将 score 乘 `scale*log2(e)`。特殊 scale 使用共享 header 的 FP64 raw-score 差值与自然对数 LSE 路径，分派条件与 FP32 累加/输出表示范围限制见 [s40 数值边界](../s40_attention/README.md#共享数值路径的边界)。默认全部列使用 PTX `ex2.approx.ftz.f32`。设置 `polynomial_exp=True` 后，偶数列仍用 SFU，奇数列将 x 规约为整数指数与 `[-0.5,0.5]` 余量，对余量用六阶多项式，再乘二的整数幂。x 小于 -126 时返回零。这改变 SFU 与 FP32 FMA 的工作比例，同时引入近似及下溢截断误差；应先检查局部指数误差，再检查归一化后的 O/LSE。

当新的最大 score 与保存的参考值相差小于 4 个 log2 单位时，保留旧参考值。这样未归一化概率可以大于 1，但小于 16；分母与 O 必须一直使用同一个参考尺度。第一次从 `-inf` 建立参考值仍正常更新。max 上升超过阈值时恢复 rescale。该策略不会丢弃 key 的贡献，但低精度 P 舍入会改变数值误差。

特殊 scale 路径在 FP64 中把 raw maximum 差乘 scale 和 log2(e) 后再与 4 比较，不能直接把 raw 差当作 log2 差；这使两种 specialization 使用同一个概率幅度界。

`entry.py` 的两个配置彼此独立：`polynomial_exp=False`、`delayed_rescale=True` 为默认值；后者对应 `Numerics` 模板参数。当 correction 为 1 时，延迟 rescale 路径直接跳过 D 个 O 元素的乘法；需要更新参考值时仍执行完整修正。默认保留 SFU 指数，多项式作为受控实验，不能仅凭减少 SFU 工作就假定它更快。

## 独立控制两个优化

在仓库根目录通过 `./lab python` 执行下面代码；它复用同一输入，先检查四种组合的数值。更换到每个 perf case 后，可对这些 prepared launches 使用 `measure_graph_group` 做相同计时边界的交替采样，编译、准备和 oracle 均不在计时内。

```python
import torch
from flash_attention_lab.registry import EXERCISES
from flash_attention_lab.cases import make_inputs
from flash_attention_lab.api import prepare
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.accuracy import compare

e = EXERCISES["s42_numerics"]
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    inputs, cfg = make_inputs(e, e["cases"][0], "cuda")
    oracle = mathematical_reference(e["kind"], inputs, cfg)
    plans = []
    for polynomial, delayed in ((False, False), (True, False),
                                (False, True), (True, True)):
        variant = dict(cfg, polynomial_exp=polynomial, delayed_rescale=delayed)
        plan = prepare(e["id"], inputs, config=variant, solution=True)
        plan.launch()
        stream.synchronize()
        print(variant, compare(e, plan.outputs, oracle, inputs))
        plans.append(plan)
```

这段单例说明接口；完整验收还需重复调用、非默认 stream、输出污染与所有声明案例。独立指数消融与 rescale 消融分别报告，组合版本与 s41 的差异不能全部归为其中一项。

## 验证与性能

```bash
./lab check s42_numerics --solution
./lab solution s42_numerics --case 6
./lab solution s42_numerics --case 7
./lab solution s42_numerics --case 8
```

`large_logits`、`rising_max`、显式 scale 和均匀输入分别检查动态范围、跨 tile 状态重标定、尺度及归一化。s41 的 SFU/逐次 rescale 路径与 cuDNN 提供两类性能对照。多项式增加 FMA 指令，延迟 rescale 改变概率动态范围，两者都不保证更快或误差更小。所有实验保留原有容差和全部 perf case 结果。

数据布局、同步和指令来源见 [s40](../s40_attention/README.md)、[s41](../s41_overlap/README.md)。指数工作分配与 rescale 的设计讨论参见 [FA4 论文](https://arxiv.org/abs/2603.05451v1)；本地多项式与阈值由这里的显式 CUDA/PTX 实现定义，不应当作官方 kernel 的相同配置。
