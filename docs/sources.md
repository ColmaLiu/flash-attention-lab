# 来源、版本与阅读方法

课程提供独立数学 oracle、完整实现和快速库对照。上游源码通过 `third_party/cutlass` 与 `third_party/flash-attention` 两个 Git submodule 引入，父仓库的 gitlinks 锁定源码版本。按 [参考实现](reference-implementations.md) 的 source map 阅读入口、设备模块和实验配置。

| 来源 | 固定标识 | 用途 |
|---|---|---|
| [FA1 原论文](https://arxiv.org/abs/2205.14135v2) | arXiv:2205.14135v2 | IO 模型、分块与归一化状态 |
| [FA2 原论文](https://arxiv.org/html/2307.08691v1) | arXiv:2307.08691v1 | query 并行、CTA 内工作分配、非矩阵操作 |
| [FA2 forward 来源](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793) | v2.7.4.post1 / `5231d95fe13733fb534c01895f7ea88c6a6c7793` | s22 实现的设计与代码来源；署名保存在答案目录 |
| [FA4 原论文](https://arxiv.org/html/2603.05451v1) | arXiv:2603.05451v1 | Blackwell forward pipeline、指数、rescale 与调度 |
| [FlashAttention submodule](../third_party/flash-attention/) | beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | 完整 SM100 attention 生产答案 |
| [CUTLASS submodule](../third_party/cutlass/) | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | C++ CuTe 头文件及完整 SM100 GEMM |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | NVIDIA 在线文档 | 指令合法域、集体参与和可见性 |
| [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | NVIDIA 在线文档 | SM100 的资源限制与优化建议 |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | NVIDIA 在线文档 | 合并访存、bank conflicts、测量方法 |
| [PyTorch CUDA SoftMax](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu) | v2.10.0 | 原生 softmax 实现阅读 |

查看父仓库记录的版本：

```bash
git ls-files --stage third_party/cutlass third_party/flash-attention
git submodule update --init -- third_party/cutlass third_party/flash-attention
```

[sources.lock.json](sources.lock.json) 记录资料来源，实际源码版本以 gitlinks 为准。许可证随 submodule 保存；课程 patch 独立保存在 `patches/`，应用于忽略目录中的编译副本。源码工作树保持干净。

C++ CUTLASS 源码版本与 Python CuTe DSL 编译器版本分别管理，后者由课程依赖声明固定。`bootstrap` 统一安装完整课程环境。在线手册会更新，指令和 API 使用应对照实际编译器及锁定源码。
