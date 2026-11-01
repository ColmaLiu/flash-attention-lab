# 来源、版本与阅读方法

课程同时提供独立数学 oracle、本地完整答案和快速库对照。完整答案可以直接阅读；生产源码包含超出课程的通用功能，学习时按每课 source map 定位目标模块，不需要复现全部 dispatch 或 backward。

| 来源 | 固定标识 | 用途 |
|---|---|---|
| [FA1 原论文](https://arxiv.org/abs/2205.14135v2) | arXiv:2205.14135v2 | IO 模型、分块与归一化状态 |
| [FA2 原论文](https://arxiv.org/html/2307.08691v1) | arXiv:2307.08691v1 | query 并行、CTA 内工作分配、非矩阵操作 |
| [FA2 官方源码](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793) | v2.7.4.post1 / `5231d95fe13733fb534c01895f7ea88c6a6c7793` | API 的 BNHD 与历史 forward 数据流 |
| [FA4 原论文](https://arxiv.org/html/2603.05451v1) | arXiv:2603.05451v1 | Blackwell forward pipeline、指数、rescale 与调度；backward 章节不作为本课要求 |
| [FA4 官方源码](https://github.com/Dao-AILab/flash-attention/tree/1fc0221183e799363001394a12dd9e78e15f5dda/flash_attn/cute) | beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | 本地完整 SM100 attention 答案 |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/tree/4370102f9dacab813282e1d67722fceb0b90a019) | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | 本地完整 C++ SM100 GEMM 答案 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 在线 NVIDIA 文档 | 指令合法域、集体参与和可见性 |
| [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 在线 NVIDIA 文档 | SM100 与其他 Blackwell 的资源差异 |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 在线 NVIDIA 文档 | 合并访存、bank conflicts、测量方法 |
| [PyTorch CUDA SoftMax](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu) | v2.10.0 | 原生 softmax 快速实现阅读 |

上游完整代码和许可证在 [references](../references/README.md)，具体函数、配置与本地 patch 在同页 source map。固定版本信息以 [sources.lock.json](sources.lock.json) 与 [manifest](../references/manifest.json) 为准。在线手册会更新，API 与指令最终以所选依赖和编译器支持为准。

C++ CUTLASS 源码版本与 Python CuTe DSL wheel 版本是两件事。主学生路线不强制引入 CuTe；阶段 3 的 C++ 答案使用这里固定的头文件，阶段 4 的官方答案使用 `.[fa4]` 声明的 Python 编译器依赖。普通课程命令不隐式联网安装官方库。
