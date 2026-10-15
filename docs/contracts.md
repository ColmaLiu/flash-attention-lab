# 接口与工程边界

所有学生入口通过 `flash_attention_lab.api.prepare(id, inputs, config)` 调用；返回 `Prepared(launch, outputs, workspace, inputs)`。`launch()` 必须只提交本次计算，不编译、不分配、不读取 CPU 参考答案。`run()` 是包含准备和分配的便捷入口，不是纯 kernel 计时接口。

## Tensor 字典键名

下面列出完整的输入/输出键集合，**全部为区分大小写的小写字符串**。数学符号 X、Q、LSE 等不代表实际键名。每关任务文档的“参数字典：精确键名”表还列出 shape、dtype 和含义；学生入口旁也有同样的键名提示，无需查测试代码。

| 关卡 | `inputs` 键 | `outputs` 键 |
|---|---|---|
| s00_softmax、s31_port | `"x"` | `"y"` |
| s01_reduce | `"x"` | `"maximum"`、`"sum"` |
| s03_online | `"x"` | `"y"`、`"m"`、`"l"` |
| s02_naive、s10_fa1、s11_fa1_edges | `"q"`、`"k"`、`"v"` | `"o"`、`"lse"` |
| s20_fa2_grid、s21_fa2_sliced、s23_fa2_tensor、s24_fa2_edges | `"q"`、`"k"`、`"v"` | `"o"`、`"lse"` |
| s40_fa4_base、s41_fa4_pipeline、s43_fa4_exp、s44_fa4_rescale、s45_fa4_edges、s46_fa4_schedule | `"q"`、`"k"`、`"v"` | `"o"`、`"lse"` |
| s22_mma、s37_tcgen05、s38_async_gemm | `"a"`、`"b"` | `"c"` |
| s30_layout、s32_async、s33_tma、s34_warp_specialize、s35_pipeline、s36_tmem、s42_exp | `"x"` | `"y"` |

`maximum/sum` 是原始输入的逐行最大值与逐行和；`m/l` 是 online softmax 最终的整行最大值与基于该最大值的指数和。attention 的 `lse` 使用自然对数。GEMM 的 `c` 为 FP32；s34/s35 的 `y` 也是 FP32，尽管其输入 `x` 为 FP16。不要从输入 dtype 推断所有输出 dtype。

CUDA C++ 中，`inputs`、`outputs` 和 `workspace` 均为 `TensorMap`，值是 `TensorView`。例如第一关通过 `inputs.at("x")`、`outputs.at("y")` 获取描述符；`.data` 是设备地址，`.shape` 是 host 侧的逻辑维度数组，单位为元素，`.dtype` 使用 `DType` 枚举。维度从 `.shape` 读取，不会自动添加 `R/C/N/D` 等 Options 字段。所有输入/输出为 contiguous，最后一维连续；不要依赖 unordered_map 的遍历顺序。`const TensorMap&` 约束的是描述符容器，输出显存仍可写；输入显存必须保持不变。

Python 的 `workspace_spec`（两种 backend 都适用）和 CuTe 的 `prepare` 接收 `dict[str, torch.Tensor]`，按 `inputs["x"]` 等键访问。CuTe 的 tensor 转换、device layout 和 launch 由学生设计。`workspace` 没有任何公共预设键，其键、shape、dtype 来自学生的 `workspace_spec`；默认返回空字典。

## 配置与 stream

| 字段 | CUDA C++ `options` | CuTe/Python `config` | 含义 |
|---|---|---|---|
| `"scale"` | `options.at("scale")`，double | `config["scale"]`，数值标量 | attention 的 logits 缩放；不参与其它关的数学运算 |
| `"causal"` | `options.at("causal")`，double，0.0/1.0 | `config["causal"]`，bool | attention 为 true 时保留 j ≤ i；其它关声明为 false |
| 学生自定义键 | double | 数值或 bool | 来自 `entry.CONFIG` 或调用参数，不预选 launch/tile/stage |

非 attention 默认 `scale=1.0`；attention 调用未显式传入 `config["scale"]` 时，host 取 `D**-0.5`（包括覆盖 `entry.CONFIG` 中的 scale）。其它字段按公共默认值 → `entry.CONFIG` → 调用参数的顺序合并；check/bench 的 case 参数覆盖 `entry.CONFIG` 中的 scale/causal。输入维度不会自动作为配置键传入。

CUDA C++ 的 `stream` 是 `cudaStream_t`；CuTe 的 `stream` 是当前 CUDA stream 的整数 handle。所有设备操作使用传入的 stream；禁止硬编码默认 stream。没有隐式 mask tensor、scale tensor 或算法状态 tensor。

## 计算与存储契约

attention：Q/K/V 均 contiguous `[B,H,N,D]`，三者同 dtype/device/shape，self-attention，O 同形状同 dtype，LSE `[B,H,N]` FP32。scale 为有限正数，缺省 `D**-0.5`；causal 保留 `j <= i`。LSE 使用自然对数。FP32 基础关不使用 TF32；低精度关使用 FP16/BF16 输入、FP32 点积与输出累加、FP32 softmax 状态。P 若在 Tensor Core 前转低精度，要在实验记录声明。

每关 `curriculum.json` 中列出的 shape/dtype/mask 组合就是受限支持域，host 会拒绝其他组合。有限输入，无空维度；NaN/Inf 输入不在范围中。所有测试输入生成固定 seed，再转换到目标 dtype；数学参考针对已经转换的输入。支持范围扩展必须同步增加案例和文档，不能仅放宽校验。

CUDA 只有一种构建方式：`torch.utils.cpp_extension.load` JIT，每关独立目录、独立模块，pybind11 传递设备地址与 shape/dtype，避免学生错误使整仓库无法编译。ABI 定义见 `flash_attention_lab/cuda_api.h`。可用 `cuda_fp16.h`/`cuda_bf16.h`，未预选任何学生 launch 或 layout。

CuTe 的 `entry.prepare` 接收 PyTorch tensors 与整数 stream handle。允许在这里使用 `cutlass.cute.runtime.from_dlpack`、`cuda.bindings.driver.CUstream` 与 `cute.compile`；具体 kernel/JIT 签名、layout、compile specialization 和 launch 由学生定义。每关 `entry.CONFIG` 是空的数值配置字典，由学生填写自己的 tile/stage 等参数；check/bench 会合并并记录这些参数，测试 case 的 scale/mask 优先。必须保留编译对象与参数的生命周期，返回闭包。若增加配置，记录 config 并在 entry 中校验；host 不替学生选 tile、stage 数或 TMEM 列数。

`workspace_spec(inputs,config)` 只声明全局 scratch 的 `{name:(shape,torch.dtype)}`。host 使用 `torch.empty` 分配，不为你初始化 m/l/O、barrier 或任何算法状态。SMEM/TMEM 的大小、布局与初始化属于学生 kernel。每次 launch 必须独立可重复：前一次结果不能意外参与下一次。输出不能 alias 输入；不支持 autograd。

`prerequisites` 是学习顺序，不阻止单独加载后面的任务。实际代码复用另在 `RUNTIME_REQUIRES` 声明；检查会给出具体未实现模块名。复用文件也纳入代码指纹。不要用 runtime dependency 作为跳过新练习实现的理由。

“算法 tile”是矩阵逻辑区域，“CTA/block”是共同驻留和同步的线程集合，“warp”是 32 个线程的执行组，“thread”持有自己的寄存器。一个 tile 可能由一个或多个 warp/CTA 操作，一个 CTA 也可能处理多个 tile。grid 并行决定 CTA 间 ownership；sliced-Q 决定 CTA 内 warp 间 ownership，二者必须分别说明。

本仓库以源码树方式开发，激活 `flash-attention-lab` conda 环境后，`python -m pip install -e .` 只做 editable 安装；不是可搬运的 wheel 分发包。官方 FA4 对照位于 `benchmarks/official_fa4.py`，不属于学生 API 的任何 fallback。
