# 环境与依赖

使用独立 Conda 环境，默认 `flash-attention-lab`；通过 `LAB_CONDA_ENV` 修改名称。`scripts/bootstrap.sh` 安装声明依赖，`./lab` 负责激活。不要在 base 中安装项目依赖。Python/PyTorch 的固定依赖以 `pyproject.toml`、`requirements.txt` 和 `environment.yml` 为准。

主线学生与基础教学答案使用 CUDA C++，需要可用的 nvcc、兼容 host C++ 编译器、PyTorch 及 Ninja。PTX 内联汇编与 C++ CuTe 都由 CUDA Toolkit 编译；Python CuTe DSL 是另一条可选编译链。阶段 4 的官方完整答案使用 CuTe DSL，需要其专门依赖，安装入口及版本见 `references/README.md`。

分清四个版本：驱动所能支持的 CUDA 版本、`nvcc --version` 的 Toolkit、`torch.version.cuda` 的 PyTorch runtime，以及 Python CuTe DSL 的编译组件。它们不必相等，不能只看 `nvidia-smi` 就说已安装某 Toolkit。

工具从 `LAB_CUDA_HOME`、`CUDA_HOME` 或 PATH 发现，具体规则见 `./lab doctor`。只有实际检查选中 GPU 后才选择架构；B200 是 SM100，架构专用实验使用 `sm_100a`。消费级 SM120 不提供同样的 tcgen05 教学路径。

GPU 用 `LAB_GPU` 选择；已有 `CUDA_VISIBLE_DEVICES` 时必须尊重它提供的可见范围。默认仅运行一张 GPU，不终止他人进程、不改系统时钟或全局工具链。缓存与诊断日志写到忽略目录。安装成功、import 成功与实际 GPU kernel 正确是不同检查。

```bash
bash scripts/bootstrap.sh
LAB_GPU=0 ./lab doctor
./lab python -m pytest -q
./lab smoke
```

第一次 CUDA 扩展或 DSL JIT 会较慢；先用小 case 完成构建与正确性，随后才计时。公共文档记录安装方法，不记录某一台机器已经通过的结论。
