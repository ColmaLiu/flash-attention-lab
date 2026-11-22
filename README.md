# flash-attention-lab

在 NVIDIA B200（SM100）上，从一个矩阵转置开始，一步步写出接近库性能的 FlashAttention。

本课程用 **CUDA C++ 和内联 PTX** 编写全部 kernel，以 FlashAttention 1→4 为主线：每一代 FlashAttention 解决一个瓶颈，每一关先在简单算子上练一个新的硬件机制，再把它用进 attention。

## 适合谁

| 你需要会的 | 不需要会的 |
|---|---|
| C/C++ 基础、Python 基础 | Tensor Core、TMA、TMEM 等硬件细节（讲义从头讲） |
| 写过一个 elementwise CUDA kernel | 读过 FlashAttention 论文或 PTX 手册 |
| 知道 grid / block / thread 和 shared memory 是什么 | CUTLASS、CuTe 或 Triton |
| 矩阵乘法、softmax 的数学定义 | |

如果你还没写过 CUDA，先读 [从零开始](docs/starting-from-zero.md)。

学完后你应该能：

1. 从数学定义推导 attention 的 HBM 访存量，说清 FA1→FA4 每一代解决了什么瓶颈；
2. 独立写出 `mma.sync`、TMA、mbarrier、`tcgen05`/TMEM、warp specialization、2-CTA MMA 的正确代码，并说清每条同步的作用；
3. 用 SASS、Nsight Compute 和受控计时验证优化假设；
4. 读懂上游 FA4（CuTe DSL）与 CUTLASS SM100 FMHA 的源码结构。

## 环境要求

| 项目 | 要求 |
|---|---|
| GPU | B200 或其他计算能力 10.0（SM100）的 GPU。消费级 Blackwell（SM120）没有 `tcgen05`，不支持 |
| 系统 | Linux x86_64，NVIDIA 驱动支持 CUDA 12.8 |
| 软件 | Conda；CUDA Toolkit 12.8（可由安装脚本装进环境） |

## 安装

```bash
git clone -b v2 https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh            # 可加 --with-cuda、--with-fa4，见下表
```

安装脚本会创建独立的 Conda 环境 `flash-attention-lab`，初始化 `third_party/` 中固定版本的上游源码，并安装 PyTorch 等 Python 依赖。两个可选参数：

| 参数 | 作用 | 什么时候需要 |
|---|---|---|
| `--with-cuda` | 往环境里装 CUDA 12.8 和 GCC 13 | 系统里没有 CUDA 12.8 的 `nvcc` |
| `--with-fa4` | 安装 FlashAttention-4（固定版本），作为性能对照 | 想在部分 3、4 的 `bench` 里看到 FA4 一列 |

详细说明见 [环境与依赖](docs/environment.md)。

## 快速开始

```bash
./lab doctor                 # 检查 GPU、nvcc、PyTorch 和工具
./lab list                   # 课程列表与你的进度
```

然后打开第一关的讲义 [course/00_transpose/README.md](course/00_transpose/README.md)，在 `students/00_transpose/kernel.cu` 里完成 `TODO(1)`，运行：

```bash
./lab check 00 --step 1      # 第 1 步的正确性检查
./lab bench 00 --step 1      # 和参考答案、PyTorch 比速度
```

还没写代码时，`check` 会明确报告 `TODO(1) is not implemented yet`。这是预期行为。

机器上有多张 GPU 时，用 `LAB_GPU` 选一张空闲的卡，例如 `LAB_GPU=3 ./lab check 00 --step 1`。

## 课程地图

关卡 ID 的第一位数字是“部分”：部分 0 是预备，部分 1–4 分别对应 FA1–FA4。

| ID | 主题 | 这一关学什么 |
|---|---|---|
| **部分 0 · 预备** | | |
| [00_transpose](course/00_transpose/README.md) | 矩阵转置 | 线程到地址的映射、合并访存、shared memory、bank conflict、在途访存 |
| [01_softmax](course/01_softmax/README.md) | 行 softmax | warp shuffle、block 归约、数值稳定、online softmax |
| **部分 1 · FA1：IO 感知的分块** | | |
| [10_simt_gemm](course/10_simt_gemm/README.md) | FP32 分块 GEMM | 数据复用、算术强度、roofline、shared memory 与寄存器分块 |
| [11_naive_attention](course/11_naive_attention/README.md) | 标准 attention | attention 数学、causal mask、LSE、HBM 流量计数 |
| [12_fa1](course/12_fa1/README.md) | FlashAttention-1 | 分块 + online softmax 的正确性证明、论文 Algorithm 1 |
| **部分 2 · FA2：Tensor Core 与并行划分** | | |
| [20_mma_gemm](course/20_mma_gemm/README.md) | `mma.sync` GEMM | warp 级 Tensor Core、fragment 布局、`ldmatrix`、XOR swizzle |
| [21_pipeline_gemm](course/21_pipeline_gemm/README.md) | 异步拷贝流水 | `cp.async`、commit/wait group、多级 buffer |
| [22_fa2](course/22_fa2/README.md) | FlashAttention-2 | 循环交换、P 留在寄存器、causal 跳块 |
| **部分 3 · FA3 的思想在 SM100 上：异步与分工** | | |
| [30_tma_mbarrier](course/30_tma_mbarrier/README.md) | TMA 转置 | tensor map、TMA load/store、mbarrier 的 phase、环形 buffer |
| [31_tcgen05_gemm](course/31_tcgen05_gemm/README.md) | `tcgen05` GEMM | Tensor Memory（TMEM）、smem/instruction descriptor、`tcgen05.mma/commit/ld` |
| [32_ws_gemm](course/32_ws_gemm/README.md) | warp specialization | 生产者/消费者分工、TMEM 双缓冲、persistent kernel、L2 光栅化 |
| [33_2cta_gemm](course/33_2cta_gemm/README.md) | 2-CTA MMA | cluster、distributed shared memory、`cta_group::2`、TMA multicast |
| [34_fa3](course/34_fa3/README.md) | FA3 式 attention | S/O 放在 TMEM、softmax 与 MMA 重叠 |
| **部分 4 · FA4：Blackwell 上的完整优化** | | |
| [40_fa4_pipeline](course/40_fa4_pipeline/README.md) | FA4 流水结构 | 两个 Q tile ping-pong、P 写回 TMEM、correction warpgroup |
| [41_fa4_softmax](course/41_fa4_softmax/README.md) | softmax 吞吐 | MUFU 瓶颈、多项式 `exp2`、条件 rescale |
| [42_fa4_2cta](course/42_fa4_2cta/README.md) | 2-CTA attention | 2-CTA 的 QKᵀ/PV 划分、cluster 同步 |
| [43_fa4_schedule](course/43_fa4_schedule/README.md) | 调度与 causal | 块分类、persistent、LPT 排序 |
| [44_fa4_fp8](course/44_fa4_fp8/README.md) | FP8 前向 | E4M3 量化、descale、`kind::f8f6f4` |

建议按顺序学。每关讲义开头写了前置关卡。

## 一关怎么学

每关都有同样的结构：一份讲义、一个带 `TODO` 的骨架、一组按 step 拆开的检查。

```
读讲义 ──► 填 TODO(n) ──► ./lab check ID --step n ──► ./lab bench ID --step n ──► sass / profile / sanitize
 §1–§5      §6 分步练习      正确性（小 case 先挂）       与答案、库比速度            解释为什么快或慢
                ▲                                                                        │
                └──────────────────────────── 下一个 step ◄──────────────────────────────┘
```

| 环节 | 在哪里 | 说明 |
|---|---|---|
| 讲义 | `course/<ID>/README.md` | 九节：目标、动机、数学、硬件与指令、调度设计、分步练习、性能、自测、延伸阅读 |
| 骨架 | `students/<ID>/kernel.cu` | 你只改这个文件。launch 配置等非重点部分已写好；每个 `TODO(n)` 写明输入、输出和不变式 |
| 正确性 | `./lab check ID --step n` | 与 FP64 数学 oracle 比较。输出先填 NaN，漏写会被发现；还会在非默认 stream 上再跑一次 |
| 性能 | `./lab bench ID --step n` | 同一输入、同一计时方式下，比较你的 kernel、参考答案和库（cuBLAS、cuDNN、FA4） |
| 分析 | `./lab sass` / `profile` / `sanitize` | 寄存器与指令、Nsight Compute 指标、Compute Sanitizer。见 [工具指南](docs/tools.md) |
| 参考答案 | `solutions/<ID>/` | 可读的 CUDA C++/PTX 实现和设计说明 `NOTES.md`。卡住时可以读；加 `--solution` 可运行它 |

所有命令：

```
./lab list                                     课程列表与进度
./lab check ID [--step N] [--case NAME]        正确性
./lab bench ID [--step N] [--case NAME] [--cold]   计时
./lab sass ID                                  寄存器、spill、指令统计
./lab profile ID [--step N] [--case NAME]      Nsight Compute
./lab sanitize ID --case NAME [--tool memcheck|racecheck|synccheck|initcheck]
./lab doctor                                   环境检查
./lab clean [--yes]                            删除编译缓存和 bench/profile/sass 输出（不动 students/ 和进度）
```

在 check、bench、sass、profile、sanitize 后加 `--solution`，就改为运行参考答案。ID 可以写唯一前缀，例如 `00` 或 `31_tc`。

## 仓库结构

```
course/<ID>/          讲义 README.md 与 spec.json（case、step、性能目标）
students/<ID>/        你的练习：kernel.cu
solutions/<ID>/       参考答案：kernel.cu（及 .cuh）与 NOTES.md
include/lab/          公共头文件：接口、错误检查，以及前面关卡写过的原语（mma_sync、tma、tcgen05…）
labkit/               Python 基础设施：造数据、调用 kernel、对答案、计时
third_party/          固定版本的 CUTLASS 与 FlashAttention 源码，供对照阅读
docs/                 全局文档
scripts/              bootstrap.sh（安装）、activate.sh（激活环境）
tests/                基础设施测试
artifacts/  .cache/  progress/              bench/profile 结果、编译缓存、进度（Git 忽略）
```

`include/lab/` 中的原语遵循“前面写过的，后面提供”：例如你在 20 关自己实现 `ldmatrix` 和 `mma.sync` 的封装，从 21 关起直接 `#include "lab/mma_sync.cuh"`。

## 文档

| 文档 | 内容 |
|---|---|
| [从零开始](docs/starting-from-zero.md) | 第一次写 CUDA：线程、地址、存储层级，以及第一关的第一步 |
| [B200 架构概览](docs/b200.md) | SM、寄存器、shared memory、L2、HBM、Tensor Core、TMEM、TMA、cluster 的数字，以及哪一关讲 |
| [工具指南](docs/tools.md) | `./lab sass/profile/sanitize`、SASS 指令、计时模式与 `vs` 列 |
| [术语表](docs/glossary.md) | 中英文术语，链接到引入它的关卡 |
| [attention 的数学](docs/attention-math.md) | attention、online softmax、IO 复杂度、数值检查 |
| [环境与依赖](docs/environment.md) | 安装参数、`LAB_GPU`、CUDA 版本、FA4 基线 |
| [上游来源](docs/sources.md) | 论文、固定版本的上游源码 |
| [代码格式](docs/formatting.md) | clang-format、Ruff、pre-commit |
