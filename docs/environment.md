# 环境与依赖

课程统一使用独立 Conda 环境，默认名为 `flash-attention-lab`，可通过 `LAB_CONDA_ENV` 修改。`scripts/bootstrap.sh` 安装完整课程的声明依赖，包括 PyTorch 与 CUDA 扩展运行支持；`./lab` 激活该环境。Python 包版本以 `pyproject.toml`、`requirements.txt` 和 `environment.yml` 为准。

```bash
git clone https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh
LAB_GPU=0 ./lab doctor
./lab python -m pytest -q
./lab smoke
```

已有 checkout 使用 `git submodule update --init -- third_party/cutlass third_party/flash-attention` 初始化固定版本。`third_party/cutlass` 与 `third_party/flash-attention` 的版本由父仓库 gitlinks 锁定。上游源码用于生产实现与指令语义的对照阅读，具体来源见 [参考实现](reference-implementations.md)。

学生入口与参考答案统一使用 CUDA C++/PTX，由 nvcc 编译。Python 仅负责公共接口、输入输出、计时外准备和测试。绑定及构建使用固定版本的 `apache-tvm-ffi`，由 Ninja 调用 host C++ 编译器和 nvcc；不依赖 PyTorch C++ 扩展构建器。FFI host 绑定要求 C++20，设备源码保留 C++17，`bootstrap.sh --with-cuda` 提供的 GCC 13 支持两者。

TVM FFI 只负责跨语言调用与 DLPack 张量交换，不要求用 TVM 编译器生成设备代码。构建缓存位于 `.cache/build`，名称包含源码、传递包含的本地 header、FFI 版本和编译设置，避免加载旧共享库。正式测量指纹也包含 FFI 版本；迁移绑定或更新依赖后必须重新构建、验证和测量。

区分三个版本：驱动所支持的 CUDA 版本、`nvcc --version` 的 Toolkit，以及 `torch.version.cuda` 的 PyTorch runtime。`./lab doctor` 分别检查它们；查看驱动版本不能代替检查 Toolkit。

工具从 `LAB_CUDA_HOME`、`CUDA_HOME` 或 PATH 发现。实际检查选中 GPU 后再选择编译目标；B200 是 SM100，架构专用实验使用 `sm_100a`。消费级 SM120 的指令支持不同，本课程的 tcgen05 路径要求 SM100。

GPU 用 `LAB_GPU` 选择；已有 `CUDA_VISIBLE_DEVICES` 时选择必须在其分配范围内。默认运行一张 GPU，保持系统时钟、全局工具链和其他用户进程原状。缓存与诊断写入忽略目录。

第一次 CUDA 扩展编译会较慢。先用小 case 完成构建与正确性检查，再测量已经准备好的设备工作。安装、基础设施 smoke 与学生正确性分别报告。
