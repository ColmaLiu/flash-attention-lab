# flash-attention-lab

在 NVIDIA Blackwell SM100（例如 B200）上亲手实现 FlashAttention-1/2（CUDA C++）与受限 FlashAttention-4 forward（CuTe DSL）。阶段 0–4 共 27 个练习，包含任务、空学生入口、独立数学参考、测试和 benchmark。**核心 kernel 由学习者编写；未实现的入口明确报错。** 不独立实现 FA3，其异步执行与同步基础在阶段 3 补齐。

## 安装与检查

安装配置面向 **Linux x86_64（glibc ≥ 2.28）、SM100 GPU、Conda**。需要可用的 NVIDIA 驱动；CuTe DSL CUDA 12 系列建议驱动至少 575.51.03。其它 Blackwell compute capability 或 CPU 平台不在此配置的支持范围内。完整要求和来源见 [环境与安装](docs/environment.md)。

克隆或下载仓库，进入仓库根目录后执行：

```bash
# 已有 CUDA Toolkit 12.8 和兼容 C++ 编译器，且 nvcc 在 PATH 中：
bash scripts/bootstrap.sh
# 若缺少 Toolkit/编译器，改用下面这条，在专用 Conda 环境内安装：
# bash scripts/bootstrap.sh --with-cuda

./lab doctor                  # 检查选中设备、依赖和工具链
./lab python -m pytest -q      # 基础设施测试，不执行学生 kernel
./lab smoke --timeout 600      # 实际编译、绑定、GPU 调用和计时设施检查
```

安装脚本创建 `flash-attention-lab` Conda 环境，并从公开包源安装项目依赖。`./lab` 自动激活该环境，可在任意目录调用其路径；无需预先存在缓存、结果文件或 Git 元数据。首次编译/JIT 较慢，后续运行复用缓存。

默认使用一张 GPU：已有 `CUDA_VISIBLE_DEVICES` 时选其中第一张，否则选 GPU 0。`LAB_GPU` 可指定索引或 UUID；已有可见设备列表时，值必须是列表中的原始标识，不会越过调度器的分配。工具链发现顺序是 `LAB_CUDA_HOME`、`CUDA_HOME`、`PATH` 中的 `nvcc`。多版本 Toolkit 应显式选择 12.8。

```bash
LAB_GPU=1 ./lab doctor
# 若已有 CUDA_VISIBLE_DEVICES，则 LAB_GPU 必须选该列表中的设备。
# 非默认安装位置：export LAB_CUDA_HOME=/path/to/cuda-12.8
# 自定义环境名称：LAB_CONDA_ENV=my-fa-lab bash scripts/bootstrap.sh
# 后续命令也使用同一个 LAB_CONDA_ENV。
```

项目安装配置固定 PyTorch `2.10.0+cu128`、CuTe DSL `4.4.1` 和 CUDA Python `12.9.4`；项目依赖声明在 `pyproject.toml`，`requirements.txt` 指定 PyTorch 的 CUDA wheel 变体及官方源。驱动支持版本、nvcc Toolkit 与 PyTorch runtime 是不同概念。官方 FA4 是独立可选对照，不是安装前提。

## 从第一关开始

先读 [协作边界](AGENTS.md)，再打开 [s00_softmax](docs/tasks/s00_softmax.md)。[路线表](docs/roadmap.md) 列出所有任务、文件、依赖和验收命令；[验证流程](docs/validation.md) 说明检查结果的含义。

实现 `students/cuda/s00_softmax/kernel.cu` 中的 `student_launch` 及自己的设备模块。launch、线程映射、reduction 和 shared memory 均由你设计；需要全局 scratch 时填写同目录 `entry.py:workspace_spec`。实现前执行验收会报 `STUDENT_UNIMPLEMENTED`，退出码 2。完成后设置 `IMPLEMENTED=True`，运行：

```bash
./lab check s00_softmax --case 0
./lab check s00_softmax
./lab sanitize s00_softmax --tool memcheck --case 0
./lab sanitize s00_softmax --tool racecheck --case 0
./lab sanitize s00_softmax --tool synccheck --case 0
./lab bench s00_softmax --case 0
```

通关要求为全部声明案例通过、必要内存安全检查无错误，并在 [实验记录](experiments/s00_softmax.md) 中解释稳定性与线程 ownership。完成设计 review 后进入 `s01_reduce`；第一关没有性能门槛。缺少 sanitizer/profiler 时会说明原因，不能将其计为通过。

## 日常命令

上面的验收流程适用于所有关卡，替换关卡 ID 即可。ID 支持完整名称或短编号，例如 `./lab check s20` 等价于 `./lab check s20_fa2_grid`。其他常用入口：

| 命令 | 用途 |
|---|---|
| `./lab list --stage 3` | 查看该阶段的任务；doctor、smoke、status 也支持 `--stage` |
| `./lab task ID` | 查看 tensor 键名、学生文件和完整任务文档路径；`-v` 增加目的与案例列表 |
| `./lab status` | 查看各关状态和记录，默认不运行 kernel |
| `./lab profile ID --case 0` | 对指定案例运行 Nsight Compute |
| `./lab --help` | 查看全部命令；子命令加 `--help` 查看参数 |

每关独立运行 `./lab check ID` 或 `./lab bench ID`。status 根据实现标记和验收记录汇总各关状态；完整状态和退出码见 [状态语义](progress/README.md)。

测试从小规模定位延伸到多头 1K–8K attention、大矩阵 GEMM 与长流水线；`check ID` 默认验收本关的全部尺寸，`task ID -v` 列出 case 编号。各阶段的范围及性能比较方法见 [测试规模](docs/test_scales.md)。

命令默认使用英文：check 逐 case 显示 PASS/FAIL 和失败原因；bench 显示计时，不展开精度明细。`--verbose`（`-v`）展开细节及报告路径，`--json` 输出完整机器报告，例如 `./lab bench s00_softmax --json`。日志与结果保存在已忽略的 `artifacts/` 中。完整输出约定见 [终端输出说明](docs/cli.md)。

## 目录

```text
students/cuda/         11 个 CUDA C++ 练习，每关 kernel.cu + entry.py
students/cute/         16 个 CuTe DSL 练习，每关 entry.py
flash_attention_lab/  构建、绑定、host 契约、参考、检查和计时工具
tests/               基础设施测试与显式学生验收
benchmarks/           可选官方 FA4 性能对照
docs/                安装、路线、契约、算法说明和 27 关任务
experiments/          空白实验记录模板
progress/             验收记录和状态语义
scripts/              安装和基础设施验证入口
curriculum.json       关卡支持域、案例与学习依赖
```

各代实现保留独立入口。学习前置与代码复用依赖分别声明；缺少学生模块会报告具体名称，不调用参考或官方实现替代。

运行生成的 `artifacts/`、`.cache/` 和 `progress/results/` 均被 Git 忽略，缺失时按需创建。实验模板和源代码纳入版本管理；发布公共模板时不附带个人进度、测量结果或机器快照。

进一步阅读：[接口](docs/contracts.md) · [数值](docs/numerics.md) · [IO 推导](docs/io_model.md) · [Blackwell](docs/blackwell.md) · [调试](docs/debugging.md) · [计时](docs/benchmarking.md) · [来源版本](docs/sources.md)。
