# 接口与工程边界

所有学生入口通过 `flash_attention_lab.api.prepare(id, inputs, config)` 调用；返回 `Prepared(launch, outputs, workspace, inputs)`。`launch()` 必须只提交本次计算，不编译、不分配、不读取 CPU 参考答案。`run()` 是包含准备和分配的便捷入口，不是纯 kernel 计时接口。

attention：Q/K/V 均 contiguous `[B,H,N,D]`，三者同 dtype/device/shape，self-attention，O 同形状同 dtype，LSE `[B,H,N]` FP32。scale 为有限正数，缺省 `D**-0.5`；causal 保留 `j <= i`。LSE 使用自然对数。FP32 基础关不使用 TF32；低精度关使用 FP16/BF16 输入、FP32 点积与输出累加、FP32 softmax 状态。P 若在 Tensor Core 前转低精度，要在实验记录声明。

每关 `curriculum.json` 中列出的 shape/dtype/mask 组合就是受限支持域，host 会拒绝其他组合。有限输入，无空维度；NaN/Inf 输入不在范围中。所有测试输入生成固定 seed，再转换到目标 dtype；数学参考针对已经转换的输入。支持范围扩展必须同步增加案例和文档，不能仅放宽校验。

CUDA 只有一种构建方式：`torch.utils.cpp_extension.load` JIT，每关独立目录、独立模块，pybind11 传递设备地址与 shape/dtype，避免学生错误使整仓库无法编译。`cuda_api.h` 的 `TensorMap` 名称与上述输入/输出一致；dtype 取枚举 `DType`，`Options` 含 scale、causal 和自定义数值配置。`cudaStream_t` 来自 PyTorch 当前 stream。可用 `cuda_fp16.h`/`cuda_bf16.h`，未预选任何学生 launch 或 layout。

CuTe 的 `entry.prepare` 接收 PyTorch tensors 与整数 stream handle。允许在这里使用 `cutlass.cute.runtime.from_dlpack`、`cuda.bindings.driver.CUstream` 与 `cute.compile`；具体 kernel/JIT 签名、layout、compile specialization 和 launch 由学生定义。每关 `entry.CONFIG` 是空的数值配置字典，由学生填写自己的 tile/stage 等参数；check/bench 会合并并记录这些参数，测试 case 的 scale/mask 优先。必须保留编译对象与参数的生命周期，返回闭包。若增加配置，记录 config 并在 entry 中校验；host 不替学生选 tile、stage 数或 TMEM 列数。

`workspace_spec(inputs,config)` 只声明全局 scratch 的 `{name:(shape,torch.dtype)}`。host 使用 `torch.empty` 分配，不为你初始化 m/l/O、barrier 或任何算法状态。SMEM/TMEM 的大小、布局与初始化属于学生 kernel。每次 launch 必须独立可重复：前一次结果不能意外参与下一次。输出不能 alias 输入；不支持 autograd。

`prerequisites` 是学习顺序，不阻止单独加载后面的任务。实际代码复用另在 `RUNTIME_REQUIRES` 声明；检查会给出具体未实现模块名。复用文件也纳入代码指纹。不要用 runtime dependency 作为跳过新练习实现的理由。

“算法 tile”是矩阵逻辑区域，“CTA/block”是共同驻留和同步的线程集合，“warp”是 32 个线程的执行组，“thread”持有自己的寄存器。一个 tile 可能由一个或多个 warp/CTA 操作，一个 CTA 也可能处理多个 tile。grid 并行决定 CTA 间 ownership；sliced-Q 决定 CTA 内 warp 间 ownership，二者必须分别说明。

本仓库以源码树方式开发，激活 `flash-attention-lab` conda 环境后，`python -m pip install -e .` 只做 editable 安装；不是可搬运的 wheel 分发包。官方 FA4 对照位于 `benchmarks/official_fa4.py`，不属于学生 API 的任何 fallback。
