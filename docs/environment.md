# 环境与依赖

## 1. 需要什么

| 项目 | 要求 | 说明 |
|---|---|---|
| GPU | 计算能力 10.0（SM100），例如 B200 | 部分 3、4 的 `tcgen05`、TMEM 只在 SM100 上有。消费级 Blackwell（SM120）不支持 |
| 系统 | Linux x86_64 | 安装脚本只支持这个平台 |
| 驱动 | 支持 CUDA 12.8 或更新 | `nvidia-smi` 右上角显示驱动支持的最高 CUDA 版本 |
| CUDA Toolkit | 12.8 | 可以用系统的，也可以让安装脚本装进 Conda 环境 |
| Conda | 任意发行版 | 课程使用独立环境，不碰 base |

Python 3.12、PyTorch 2.10.0（CUDA 12.8 版）、`apache-tvm-ffi` 等由安装脚本安装。版本以 `environment.yml`、`requirements.txt`、`pyproject.toml` 为准。

## 2. 安装

```bash
git clone -b v2 https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh [--with-cuda] [--with-fa4]
```

脚本依次做这些事：

1. 创建 Conda 环境 `flash-attention-lab`（已存在则复用）；
2. 按参数安装可选组件（见下表）；
3. 初始化两个固定版本的 submodule：`third_party/cutlass`、`third_party/flash-attention`；
4. 安装 `requirements.txt` 中的 Python 依赖。

| 参数 | 作用 |
|---|---|
| `--with-cuda` | 往环境里装 CUDA Toolkit 12.8 和 GCC 13。系统没有合适的 `nvcc` 或 host 编译器时使用 |
| `--with-fa4` | 从 `third_party/flash-attention/flash_attn/cute` 安装 FlashAttention-4，作为 attention 关卡的性能对照 |

已有的 checkout 可以随时补跑 `bash scripts/bootstrap.sh --with-fa4`。

## 3. 使用

`./lab` 会自动激活 Conda 环境，不需要手动 `conda activate`。要在同一个环境里运行别的命令（例如 pytest）：

```bash
source scripts/activate.sh
python -m pytest -q tests/test_labkit.py
```

### 3.1 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `LAB_GPU` | 选择一张 GPU（编号或 UUID） | `CUDA_VISIBLE_DEVICES` 中的第一张；都没设时是 0 |
| `LAB_CONDA_ENV` | Conda 环境名，或绝对路径 | `flash-attention-lab` |
| `LAB_CONDA_EXE` | conda 可执行文件 | `$CONDA_EXE` 或 PATH 中的 `conda` |
| `LAB_CUDA_HOME` | 使用哪个 CUDA Toolkit | `$CUDA_HOME`，再找 PATH 中的 `nvcc` |
| `LAB_NCU`、`LAB_SANITIZER` | 指定 `ncu`、`compute-sanitizer` 的路径 | 先找 Toolkit 目录，再找 PATH |
| `LAB_NVCC_FLAGS` | 额外的 nvcc 参数，例如 `-Xptxas -v` | 空 |

**一次只用一张 GPU**。`./lab` 把 `CUDA_VISIBLE_DEVICES` 设为选中的那一张。如果你已经设了 `CUDA_VISIBLE_DEVICES`，`LAB_GPU` 必须是其中之一。多人共用机器时，先用 `nvidia-smi` 找一张空闲的卡：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
LAB_GPU=3 ./lab check 00
```

别人的 kernel 和你的 kernel 在同一张卡上运行时，`bench` 的数字不可信。

### 3.2 检查环境

```bash
./lab doctor
```

它分别检查：选中的 GPU 与计算能力、`nvcc` 版本、PyTorch 及其 CUDA 版本、`apache-tvm-ffi`、FA4 基线（可选）、`ninja`、`compute-sanitizer`、`ncu`、`cuobjdump`。

注意区分三个“CUDA 版本”：

| 版本 | 怎么看 | 含义 |
|---|---|---|
| 驱动支持的版本 | `nvidia-smi` | 驱动能运行的最高 CUDA 版本 |
| Toolkit | `nvcc --version` | 编译 kernel 用的编译器 |
| PyTorch runtime | `torch.version.cuda` | PyTorch 自带的 CUDA 运行库 |

`./lab doctor` 显示的 `nvcc` 是编译课程 kernel 实际使用的那个。

## 4. 编译与缓存

- kernel 由 `nvcc` 以 `-O3 -lineinfo -gencode=arch=compute_100a,code=sm_100a` 编译，通过 [TVM FFI](https://github.com/apache/tvm-ffi) 与 Python 交换张量。不使用 PyTorch C++ 扩展构建。
- host 端绑定代码使用 C++20，设备代码使用 C++17。`--with-cuda` 提供的 GCC 13 两者都支持。
- 编译产物缓存在 `.cache/build/`，按源码、它 `#include` 的本仓库头文件、编译参数和 FFI 版本区分。改了代码会自动重新编译。
- 第一次编译可能要几十秒，之后的 `check`、`bench` 直接使用缓存。

## 5. 输出写在哪里

这些目录都被 Git 忽略：

| 目录 | 内容 |
|---|---|
| `.cache/` | 编译缓存、CUDA JIT 缓存 |
| `artifacts/bench/` | 每次 `bench` 的完整结果（JSON，含源码指纹） |
| `artifacts/profiles/` | Nsight Compute 报告 |
| `artifacts/sass/` | `./lab sass` 导出的 SASS |
| `progress/` | 你的每关、每步通过状态（`./lab list` 读取） |

进度记录包含源码指纹。改了代码之后，`./lab list` 会提示 “code changed since last check”。
