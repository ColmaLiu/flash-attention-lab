# 来源、版本与阅读方法

课程提供独立数学 oracle、完整实现和快速库对照。上游源码通过 `third_party/cutlass` 与 `third_party/flash-attention` 两个 Git submodule 引入，父仓库的 gitlinks 锁定源码版本。按 [参考实现](reference-implementations.md) 的 source map 阅读入口、设备模块和实验配置。

| 来源 | 固定标识 | 用途 |
|---|---|---|
| [FA1 原论文](https://arxiv.org/abs/2205.14135v2) | arXiv:2205.14135v2 | IO 模型、分块与归一化状态 |
| [FA2 原论文](https://arxiv.org/html/2307.08691v1) | arXiv:2307.08691v1 | query 并行、CTA 内工作分配、非矩阵操作 |
| [FA2 forward 来源](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793) | v2.7.4.post1 / `5231d95fe13733fb534c01895f7ea88c6a6c7793` | FA2 并行划分、流水线与数值策略的生产实现对照 |
| [FA4 原论文](https://arxiv.org/html/2603.05451v1) | arXiv:2603.05451v1 | Blackwell forward pipeline、指数、rescale 与调度 |
| [FlashAttention submodule](../third_party/flash-attention/) | beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | SM100 attention 的生产实现与设计对照 |
| [CUTLASS submodule](../third_party/cutlass/) | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | SM100 指令封装、布局与完整 GEMM 的生产实现对照 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | NVIDIA 在线文档 | 指令合法域、集体参与和可见性 |
| [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | NVIDIA 在线文档 | SM100 的资源限制与优化建议 |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | NVIDIA 在线文档 | 合并访存、bank conflicts、测量方法 |
| [PyTorch CUDA SoftMax](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu) | v2.10.0 | 原生 softmax 实现阅读 |

查看父仓库记录的版本：

```bash
git ls-files --stage third_party/cutlass third_party/flash-attention
git submodule update --init -- third_party/cutlass third_party/flash-attention
```

[sources.lock.json](sources.lock.json) 记录资料来源，实际源码版本以 gitlinks 为准。许可证随 submodule 保存。本地 CUDA C++/PTX 答案中的直接改编部分另外保留来源与许可，不能用 submodule 的存在代替署名。

`bootstrap` 统一安装完整课程环境。本地参考答案由 nvcc 编译，编译与输入准备都排除在 kernel 计时之外。在线手册会更新，指令和 API 使用应对照实际编译器及锁定源码。
