# FlashAttention GPU Kernel Lab

以 FlashAttention 为主线，从内存访问、归约和 GEMM，走到 B200 上的 Tensor Core、异步搬运与 attention 流水线。课程分 **5 个阶段、15 个完整实验**，默认使用 **CUDA C++ 与 PTX**；CuTe 是阅读和表达复杂布局的可选工具，不要求中途更换语言。

每课包含概念讲解、有目的的测试、独立 FP64 数学参考、完整可读答案和快速库性能基线。`students/` 是独立实现入口；`solutions/` 与 `references/` 提供答案及固定版本的上游源码。基础设施通过、答案通过与学生通过是不同状态。

第一次写 CUDA，请先读 [从零开始：读懂第一个 GPU kernel](docs/starting-from-zero.md)，再进入课程。

## 开始

```bash
bash scripts/bootstrap.sh
./lab doctor
./lab python -m pytest -q
./lab smoke
./lab task s00_layout -v
./lab reference s00_layout --case 0
./lab solution s00_layout --case 0
```

默认使用独立 Conda 环境 `flash-attention-lab`，可由 `LAB_CONDA_ENV` 配置。GPU 由 `LAB_GPU` 选择；尊重已有 `CUDA_VISIBLE_DEVICES`，默认只运行一张。安装和各编译链见 [environment](docs/environment.md)。Blackwell 专用实验针对 SM100，例如 B200；不把消费级 Blackwell 当成相同指令集。

打开 [第一课](docs/tasks/s00_layout.md)，先阅读实验目标并运行参考，再实现 `students/cuda/s00_layout/kernel.cu` 与必要的 workspace 声明。完成后设置本课 `entry.py` 中的 `IMPLEMENTED=True`：

```bash
./lab check s00_layout --case 0
./lab check s00_layout
./lab bench s00_layout --case 0
./lab sanitize s00_layout --tool memcheck --case 0
```

空学生入口以非零状态明确失败。`bench` 先检验当前输入，再自动比较学生与快速库；`solution` 检验独立答案。阅读答案后仍需能解释数据 ownership、同步与性能变化。先选小例定位，再选课程表中的 perf 案例测量；不要自动运行长 sweep。

## 学习路线

| 阶段 | 完整实验 | 这一阶段解决什么问题 |
|---|---|---|
| 0 | layout / softmax | 地址、线程协作、归约、数值与测量 |
| 1 | SIMT GEMM / materialized attention / online attention | 数据复用、组合算子、消除 N² 中间存储 |
| 2 | warp MMA / pipeline GEMM / FA2 | Tensor Core 数据布局、异步搬运、query 并行与融合 |
| 3 | TMA / tcgen05 GEMM / warp specialization | 描述符、TMEM、跨引擎同步与流水线 |
| 4 | SM100 attention / overlap / numerics / scheduling | 生产源码阅读、消融、数值预算、完整性能实验 |

完整顺序与源码阅读入口在 [roadmap](docs/roadmap.md)。每课任务文档解释为什么学、参考如何读、每个测试检查什么以及如何验证优化假设。

Attention 统一采用与 `flash_attn_func` 相同的 **`[B,N,H,D]`** 连续输入/输出布局；LSE 是 `[B,H,N]` FP32。仅 forward、contiguous self-attention，无 dropout/bias/GQA。测试是支持域中的代表性采样，合法 shape 不再由测试列表决定，详见 [contracts](docs/contracts.md)。

## 参考与性能

FP64 oracle 用于判断数学正确性。教学答案用于理解完整实现。快速库使用 native CUDA 算子、cuBLAS，以及低精度 attention 的强制 cuDNN fused 路径；这是每关的实际速度比较对象。阶段 4 的完整答案采用固定官方 FA4 源码，不声称另有独立手写教学 FA4。源码版本及许可证在 [sources](docs/sources.md) 和 [references](references/README.md)。

计时分别呈现 CUDA Graph 与 eager launch，报告采样分布和参考/学生比值；排除编译、输入生成及预分配，明确实际布局转换和 host 成本。算法 IO 模型不等于测量 DRAM 字节，教学版 B200 数据不代表历史官方 FA 性能。见 [benchmarking](docs/benchmarking.md)。

目录：`students/` 学生代码，`solutions/` 完整答案，`references/` 固定上游源码，`flash_attention_lab/` 公共设施，`docs/tasks/` 课程，`tests/` 基础设施测试与显式学生验收。个人记录、日志、编译缓存与运行结果写入被忽略的 `artifacts/`、`.cache/`、`progress/results/`；实验模板在 [experiments](experiments/README.md)。公共文档不保存特定机器的测试或性能结论。
