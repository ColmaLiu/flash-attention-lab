# FlashAttention GPU Kernel Lab

以 FlashAttention 为主线，从内存访问、归约和 GEMM，走到 B200 上的 Tensor Core、异步搬运与 attention 流水线。课程分 **5 个阶段、15 个完整实验**。学生和完整参考答案都使用 **CUDA C++ 与 PTX**。从地址计算到矩阵指令、异步搬运与同步，核心设备代码都能直接在本仓库中阅读、修改和运行。

每课包含概念讲解、有目的的测试、独立 FP64 数学 oracle、完整可读且经过性能验证的参考答案，以及优化库性能对照。参考答案供学生编写遇到困难时学习具体实现，并作为优化起点。`students/` 放学生入口，`solutions/` 放答案，`third_party/` 通过 Git submodule 固定用于对照阅读的上游源码。学生、答案和基础设施分别验证。

第一次写 CUDA，请先读 [从零开始：读懂第一个 GPU kernel](docs/starting-from-zero.md)，再进入课程。

## 安装与开始

```bash
git clone https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh
./lab doctor
./lab python -m pytest -q
./lab smoke
./lab task s00_layout -v
./lab reference s00_layout --case 0
./lab solution s00_layout --case 0
```

已有 checkout 可运行 `git submodule update --init -- third_party/cutlass third_party/flash-attention` 补齐上游源码。`bootstrap` 为整个课程安装声明依赖，统一使用独立 Conda 环境 `flash-attention-lab`；环境名由 `LAB_CONDA_ENV` 配置。

GPU 由 `LAB_GPU` 选择，尊重已有 `CUDA_VISIBLE_DEVICES`，默认只运行一张。Blackwell 专用实验针对 SM100，例如 B200。工具链与设备要求见 [环境与依赖](docs/environment.md)。

打开 [第一课](docs/tasks/s00_layout.md)，阅读实验目标并运行参考，再实现 `students/cuda/s00_layout/kernel.cu` 与必要的 workspace 声明。完成后设置本课 `entry.py` 中的 `IMPLEMENTED=True`：

```bash
./lab check s00_layout --case 0
./lab check s00_layout
./lab bench s00_layout --case 0
./lab sanitize s00_layout --tool memcheck --case 0
```

空学生入口以非零状态明确失败。`bench` 先检查当前输入，再比较学生与快速库；`solution` 检查并测量独立答案。阅读答案时重点解释数据 ownership、同步与性能变化。先用小例定位，再选择课程表中的 perf 案例测量。

## 学习路线

| 阶段 | 完整实验 | 这一阶段解决什么问题 |
|---|---|---|
| 0 | layout / softmax | 地址、线程协作、归约、数值与测量 |
| 1 | SIMT GEMM / materialized attention / online attention | 数据复用、组合算子、消除 N² 中间存储 |
| 2 | warp MMA / pipeline GEMM / FA2 | Tensor Core 数据布局、异步搬运、query 并行与融合 |
| 3 | TMA / tcgen05 GEMM / warp specialization | 描述符、TMEM、跨引擎同步与流水线 |
| 4 | SM100 attention / overlap / numerics / scheduling | 生产源码阅读、消融、数值预算、完整性能实验 |

完整顺序与源码阅读入口在 [roadmap](docs/roadmap.md)。每课解释为什么学、参考如何读、每个测试检查什么，以及如何验证优化假设。

Attention 采用与 `flash_attn_func` 相同的 **`[B,N,H,D]`** 连续输入/输出布局；LSE 是 `[B,H,N]` FP32。范围是 forward、contiguous self-attention，无 dropout/bias/GQA。支持域与代表性测试分别声明，详见 [contracts](docs/contracts.md)。

## 参考与性能

FP64 oracle 判断数学正确性，CUDA C++/PTX 答案提供可直接跟踪的实现与优化基线。优化库对照使用 native CUDA 算子、cuBLAS，以及低精度 attention 的 cuDNN fused 路径。阶段 4 在本地完整 SM100 attention kernel 上研究异步执行、指数、rescale 和调度。答案的性能应通过每课代表性 case 与相邻阶段、优化库的实测对比来判断。完整源码、版本与许可证见 [参考实现](docs/reference-implementations.md) 和 [来源](docs/sources.md)。

计时分别呈现 CUDA Graph 与 eager launch、采样分布和双方比值；排除编译、输入生成及预分配，说明布局转换和 host 成本。分析时分别列出算法 IO 模型、实际 DRAM 计数和待验证假设。测量结论绑定本机设备、输入与代码版本，方法见 [benchmarking](docs/benchmarking.md)。

目录：`students/` 学生代码，`solutions/` CUDA C++/PTX 完整答案，`third_party/` 上游源码 submodules，`flash_attention_lab/` 公共设施，`docs/tasks/` 课程，`tests/` 基础设施测试与显式学生验收。个人记录、日志、编译缓存与结果写入被忽略的 `artifacts/`、`.cache/`、`progress/results/`；[experiments](experiments/README.md) 提供记录模板。
