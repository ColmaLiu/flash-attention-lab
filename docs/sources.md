# 上游来源与固定版本

课程的参考答案全部是本仓库中可读的 CUDA C++ / PTX，不调用 CUTLASS、CuTe 或 FlashAttention 的代码。上游源码只用来**对照阅读**：看生产实现怎样处理同一个问题。

## 1. 论文

| 论文 | 固定版本 | 对应关卡 |
|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135v2) | arXiv:2205.14135v2 | 11、12：IO 分析、分块、online softmax |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691v1) | arXiv:2307.08691v1 | 22：循环交换、warp 划分、减少非矩阵运算 |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608v2) | arXiv:2407.08608v2 | 34：异步生产者、GEMM/softmax 重叠、pingpong |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) | arXiv:2603.05451v1 | 部分 4：Blackwell 流水、指数、rescale、2-CTA、调度 |

## 2. 上游源码（Git submodule）

两个 submodule 的版本由父仓库的 gitlink 锁定。`bootstrap.sh` 会初始化它们；也可以手动运行：

```bash
git submodule update --init -- third_party/cutlass third_party/flash-attention
git ls-files --stage third_party/cutlass third_party/flash-attention   # 查看锁定的 commit
```

| 路径 | 版本 | 许可证 | 用途 |
|---|---|---|---|
| `third_party/cutlass` | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | BSD-3-Clause（`LICENSE.txt`） | SM100 指令封装、描述符位域、流水线类、Blackwell GEMM 与 FMHA 示例 |
| `third_party/flash-attention` | fa4-v4.0.0.beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | BSD-3-Clause（`LICENSE`） | FA4（CuTe DSL）的 SM100 前向实现；`--with-fa4` 从这里安装性能基线 |

各关讲义的“延伸阅读”给出具体文件。常用的入口：

| 文件 | 内容 | 对照关卡 |
|---|---|---|
| `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp` | smem descriptor、instruction descriptor 的位域 | 31 |
| `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` | `tcgen05.mma` 的 inline PTX | 31、33 |
| `third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp`、`copy_sm100_tma.hpp` | TMA 指令 | 30、33 |
| `third_party/cutlass/include/cutlass/arch/barrier.h` | mbarrier、cluster barrier | 30、33 |
| `third_party/cutlass/examples/77_blackwell_fmha/` | CUTLASS SM100 FMHA 示例 | 34、部分 4 |
| `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py` | FA4 SM100 前向 kernel | 34、部分 4 |
| `third_party/flash-attention/flash_attn/cute/interface.py` | FA4 的 Python 接口（`flash_attn_func`） | `bench` 的 `fa4` 基线 |

本仓库答案中直接改编自上游的部分，在代码注释中注明出处；许可证随 submodule 保存。

## 3. 其他固定来源

[sources.lock.json](sources.lock.json) 以机器可读的格式记录论文版本、上游文件的 commit 与 sha256。其中还包括两份不在 submodule 中的历史版本：

| 来源 | 固定版本 | 用途 |
|---|---|---|
| [FA1 实现](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3) | v1.0.9 | FA1 的原始 CUDA 实现 |
| [FA2 forward kernel](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/csrc/flash_attn/src/flash_fwd_kernel.h) | v2.7.4.post1 | FA2 的并行划分、流水与数值处理 |

## 4. NVIDIA 文档

在线文档会更新。指令用法以课程使用的 CUDA 12.8 对应的 PTX ISA 8.7 为准；`include/lab/*.cuh` 中的封装注明了 PTX ISA 8.7 的章节名。

| 文档 | 用途 |
|---|---|
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 指令语义、合法参数、内存模型 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | 编程模型、cluster、异步拷贝、各计算能力的资源上限 |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 合并访存、bank conflict、指令吞吐表 |
| [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | SM100 的资源与调优建议 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 性能指标的定义 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | memcheck、racecheck、synccheck、initcheck |
| [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | `cuobjdump`、`nvdisasm` |
