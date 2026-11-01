# 环境与依赖

课程统一使用独立 Conda 环境，默认名为 `flash-attention-lab`，可通过 `LAB_CONDA_ENV` 修改。`scripts/bootstrap.sh` 安装完整课程的声明依赖，包括 PyTorch、CuTe DSL、quack 与运行支持包；`./lab` 激活该环境。Python 包版本以 `pyproject.toml`、`requirements.txt` 和 `environment.yml` 为准。

```bash
git clone https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh
LAB_GPU=0 ./lab doctor
./lab python -m pytest -q
./lab smoke
```

已有 checkout 使用 `git submodule update --init -- third_party/cutlass third_party/flash-attention` 初始化固定版本。`third_party/cutlass` 与 `third_party/flash-attention` 的版本由父仓库 gitlinks 锁定。上游工作树保持干净；课程 FA4 rescale 补丁由公共设施应用到 `.cache/sources/` 下的编译副本，具体来源见 [参考实现](reference-implementations.md)。

学生入口使用 CUDA C++/PTX。s22/s31/s32 的答案使用 C++ CuTe 头文件，由 nvcc 编译；阶段 4 的官方答案使用 Python CuTe DSL 编译链。两条编译链所需的依赖统一安装。CUDA C++ 还需要兼容的 host C++ 编译器和 Ninja。

区分四个版本：驱动所支持的 CUDA 版本、`nvcc --version` 的 Toolkit、`torch.version.cuda` 的 PyTorch runtime，以及 CuTe DSL 的编译组件。`./lab doctor` 分别检查它们；查看驱动版本不能代替检查 Toolkit。

工具从 `LAB_CUDA_HOME`、`CUDA_HOME` 或 PATH 发现。实际检查选中 GPU 后再选择编译目标；B200 是 SM100，架构专用实验使用 `sm_100a`。消费级 SM120 的指令支持不同，本课程的 tcgen05 路径要求 SM100。

GPU 用 `LAB_GPU` 选择；已有 `CUDA_VISIBLE_DEVICES` 时选择必须在其分配范围内。默认运行一张 GPU，保持系统时钟、全局工具链和其他用户进程原状。缓存与诊断写入忽略目录。

第一次 CUDA 扩展或 DSL JIT 会较慢。先用小 case 完成构建与正确性检查，再测量已经准备好的设备工作。安装、基础设施 smoke 与学生正确性分别报告。
