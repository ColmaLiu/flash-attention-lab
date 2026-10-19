# 资料与固定版本

以下引用论文、作者仓库和 NVIDIA 官方资料。固定 commit 用于对照算法设计与 API；课程不提供官方 kernel 的复制模板或学生答案。

| 来源 | 核查内容 | 固定标识 |
|---|---|---|
| [FA1 原论文](https://arxiv.org/abs/2205.14135) / [PDF](https://arxiv.org/pdf/2205.14135) | §3.1/算法1、§3.2 IO、附录 C 状态证明 | arXiv:2205.14135 |
| [FA2 原论文](https://arxiv.org/abs/2307.08691) / [全文](https://arxiv.org/html/2307.08691) | §3.1 非矩阵操作、§3.2 query 并行、§3.3 sliced-Q | arXiv:2307.08691 |
| [FA4 原论文 v1](https://arxiv.org/html/2603.05451v1) | §2.2 Blackwell；§3.1 forward；§3.3 scheduling；§4 DSL；§5 的序列长度、head dimension 与 batch 设置；区分 §3.2 backward | arXiv:2603.05451v1 |
| [FA1 历史作者代码 v1.0.9](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3) | 历史版本对照；教学版以原论文循环定义为准 | `6d48e14a6c2f551db96f0badc658a6279a929df3` |
| [FA2 官方 forward](https://github.com/Dao-AILab/flash-attention/blob/1fc0221183e799363001394a12dd9e78e15f5dda/csrc/flash_attn/src/flash_fwd_kernel.h) | `compute_attn_1rowblock`、query block 与 key 遍历、softmax 交接 | `1fc0221183e799363001394a12dd9e78e15f5dda` |
| [FA4 官方文档](https://github.com/Dao-AILab/flash-attention/blob/1fc0221183e799363001394a12dd9e78e15f5dda/flash_attn/cute/README.md) / [依赖](https://github.com/Dao-AILab/flash-attention/blob/1fc0221183e799363001394a12dd9e78e15f5dda/flash_attn/cute/pyproject.toml) | CuTe DSL 路线、安装与独立可选对照；该版本依赖 DSL >=4.6.2 | 同上 |
| [FA4 SM100 forward](https://github.com/Dao-AILab/flash-attention/blob/1fc0221183e799363001394a12dd9e78e15f5dda/flash_attn/cute/flash_fwd_sm100.py) | TmemAllocator、S/P/O 的 lifetime、pipeline 类型、指数与 rescale 控制 | 同上；源码 SHA256 见 sources.lock.json |
| [CUTLASS v4.4.1](https://github.com/NVIDIA/cutlass/tree/4370102f9dacab813282e1d67722fceb0b90a019) | 固定 DSL API 与对应 setup；当前项目不追踪 main | `4370102f9dacab813282e1d67722fceb0b90a019` |
| [NVIDIA DSL 概览](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) / [安装](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) | Python/Linux/driver 与 toolkit 系列关系 | 在线文档会漂移，实际签名以安装 4.4.1 为准 |
| [tcgen05 API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html) | MMA/TMEM 操作的参数与 collective 约束 | 对照固定 4.4.1 API |
| [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 架构与资源约束 | 在线 NVIDIA 文档，访问时版本可能比所选 Toolkit 新 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | TMA/mbarrier/tcgen05/TMEM 指令与内存一致性 | 以所选编译器支持的指令子集为最终约束 |

FA1 的关键是存储层次与循环复用；FA2 还改变 grid 和 CTA 内分工；FA4 针对 Blackwell 非矩阵执行资源与全异步 Tensor Core 的比例调整 forward pipeline。具体论文设计到本关验证的对应表见 [roadmap](roadmap.md)。这里只保存引用和教学契约，不重印完整论文算法或官方代码。

“本仓库学习完成”不等于“官方工程功能完成”。课程限制 forward、contiguous self-attention、少量 shape；不做 backward、dropout、varlen、GQA、paged attention、FP8/FP4 或全量官方 dispatch。2-CTA backward 的通信/atomic 优化明确列为范围外。
