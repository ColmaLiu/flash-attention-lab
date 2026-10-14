# 环境与安装

## 支持范围与依赖选择

| 项目 | 项目要求 |
|---|---|
| 平台 | Linux x86_64、glibc ≥ 2.28；Bash、Conda、联网安装能力 |
| GPU | compute capability 10.0 / SM100，例如 B200；一次选择一张 GPU |
| NVIDIA 驱动 | 能运行所选 GPU 和 CUDA 12 系列；CuTe DSL 建议至少 575.51.03 |
| Python | 专用 Conda 环境中的 Python 3.12 |
| PyTorch | `2.10.0+cu128`，来自 PyTorch 官方 cu128 wheel 源 |
| CuTe DSL | `nvidia-cutlass-dsl==4.4.1`，CUDA 12 系列组件 |
| CUDA Python | `cuda-python==12.9.4`、`cuda-bindings==12.9.4` |
| CUDA C++ Toolkit | 12.8；运行时检查 GPU 后构建 `compute_100a/sm_100a` |
| Host C++ | 支持 C++17、兼容 CUDA 12.8 的编译器；Conda 安装选用 GCC 13 |
| 调试工具 | Compute Sanitizer、Nsight Compute；按需使用，缺少时明确报错 |

这是一组项目依赖选择，不是某台机器的环境导出。`pyproject.toml` 固定直接 Python 依赖，`environment.yml` 声明 Conda 基础环境；传递依赖由包管理器按这些约束解析，不承诺不同日期逐包一致。需要复现某次实验的完整包清单时自行导出到 `artifacts/`，不要把机器快照当作公共安装配置。

SM100 架构依据 [NVIDIA GPU compute capability 表](https://developer.nvidia.com/cuda/gpus)；不能把 RTX Blackwell 或其它 compute capability 当作 SM100。PyTorch 的 CUDA wheel 选择见 [官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。DSL 包自带代码生成组件，其驱动要求见 [NVIDIA DSL 安装文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)；API 对应的固定源码版本见 [sources](sources.md)。

## 创建独立 Conda 环境

从仓库根目录选择一种安装方式：

```bash
# 已有可发现的 CUDA 12.8 与 C++ 编译器：
bash scripts/bootstrap.sh

# 或在项目专用 Conda 环境中同时安装 Toolkit 与 GCC：
bash scripts/bootstrap.sh --with-cuda
```

`--with-cuda` 使用 NVIDIA 的 CUDA 12.8.1 channel 和 conda-forge，只安装用户态工具，不安装或修改驱动。Conda 的 CUDA 安装方法及 host compiler 兼容范围见 [NVIDIA CUDA 12.8 安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#conda-installation)。下载体积较大；已有合适工具链时使用第一种方式。

默认环境名为 `flash-attention-lab`。`LAB_CONDA_ENV` 可设为另一个名称或绝对前缀；安装和运行需使用同一个值：

```bash
export LAB_CONDA_ENV=my-fa-lab
bash scripts/bootstrap.sh
./lab doctor
```

已有同名环境会被复用；Python 版本不匹配时明确报错，请选择新的专用环境。脚本拒绝在 `base` 中安装。`conda` 不在 PATH 时设置 `LAB_CONDA_EXE` 为其可执行文件路径。也可以手动执行：

```bash
conda env create -f environment.yml
conda activate flash-attention-lab
python -m pip install -r requirements.txt
python -m pip check
```

`./lab` 自动激活环境。直接使用 Python 时可 `source scripts/activate.sh`；更方便的统一入口是 `./lab python -m pytest -q`，它使用与学生检查相同的 GPU 和工具链配置。项目以 editable checkout 运行，移动仓库后重新运行安装脚本；不把仅含 Python 包的 wheel 当作完整课程发行物。

## 工具链与设备配置

| 配置 | 行为 |
|---|---|
| `LAB_CUDA_HOME` | 首选 Toolkit 根目录 |
| `CUDA_HOME` | 未设置 LAB_CUDA_HOME 时采用 |
| `PATH` 中的 `nvcc` | 两者都未设置时自动推导 Toolkit 根目录 |
| `CC` / `CXX` | CUDA host / C++ 编译器可执行文件；沿用用户或 Conda 激活配置 |
| `LAB_GPU` | 单个设备索引或 UUID；不得超出已有 CUDA_VISIBLE_DEVICES 列表 |
| `CUDA_VISIBLE_DEVICES` | 继承调度器/容器设置，默认选列表第一张；空值保持无 GPU |
| `LAB_SANITIZER` / `LAB_NCU` | 可选工具路径或 PATH 中的命令名；未设置时从 PATH、选定 Toolkit/bin 查找 |
| `MAX_JOBS` | CUDA 扩展编译并发，默认 2 |

显式路径错误不会悄悄换用另一套工具。没有安装 Toolkit 或发现的是其它版本时，`./lab doctor` 给出原因；设置合适的路径，或使用 `--with-cuda`。如果切换为 Conda 工具链，注意移除指向其它 Toolkit 的显式覆盖。

构建入口支持系统 Toolkit 的 `include/lib64` 和 NVIDIA Conda Toolkit 的 `targets/x86_64-linux/include/lib` 布局；缺少开发头文件或 `libcudart.so` 时在编译前报告原因。

```bash
export LAB_CUDA_HOME=/path/to/cuda-12.8
LAB_GPU=1 ./lab doctor
CUDA_VISIBLE_DEVICES=2,5 LAB_GPU=5 ./lab doctor
```

第二个设备示例中的 `5` 是可见列表中的原始标识，不是缩窄后的逻辑序号。子进程继承同一设备。没有 CUDA_VISIBLE_DEVICES 和 LAB_GPU 时默认设备 0，不自动占用所有卡。

`nvidia-smi` 显示的 CUDA 上限描述驱动能力；`nvcc --version` 描述 Toolkit；`torch.version.cuda` 描述 PyTorch wheel。三者不能互相替代。DSL 和 CUDA C++ 是独立编译链，`doctor` 只检查可用性，真正的编译/调用须运行 `./lab smoke`。

## 安装后验收

按 [验证流程](validation.md) 运行 doctor、基础设施 pytest 和 GPU smoke。库加载错误优先核对选定环境及 `pip check`；编译错误核对 nvcc、CC/CXX 和编译日志；GPU 不可见核对驱动和调度器配置。不要通过切换全局工具链或取消设备隔离来掩盖问题。
