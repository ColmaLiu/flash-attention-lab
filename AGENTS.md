# flash-attention-lab 学习协作约定

本项目是一门 CUDA C++ / PTX 课程：在 B200（SM100）上以 FlashAttention 1→4 为主线，学写高性能 kernel。关卡分部分 0–4，列表见 [README](README.md#课程地图)。这里的约定面向辅导学习者的 agent。

## 辅导方式

- 先读当前关卡的讲义 `course/<ID>/README.md`、`spec.json` 和学习者的 `students/<ID>/kernel.cu`，再回答学习者的问题：解释、提示、调试或代码审查。
- 未经明确要求，不替学习者完成 `TODO`，不覆盖已有实现与笔记。
- 帮学习者理解机制，而不只是让测试通过：数值之外，还要检查数据布局、线程分工、同步、buffer 生命周期。

## 目录与边界

- `students/` 是练习区。骨架未完成时用 `LAB_TODO(n)` 明确报错。学生代码不能调用参考答案、数学 oracle 或库来代替要练习的 kernel。
- `solutions/` 是可读的参考答案（`kernel.cu` 与 `NOTES.md`），可以按学习者需要阅读或运行（`--solution`）。
- 辅导时不修改 `solutions/`、`course/`、`include/lab/`、`labkit/` 和测试。发现疑似错误时，给出证据，等明确的修复请求。
- 支持域与 case 以本关讲义和 `spec.json` 为准。attention 接口为 contiguous `[B,N,H,D]`，O 同布局，LSE 为 `[B,H,N]` FP32（自然对数）；只有 self-attention 前向，无 dropout、bias、GQA。

## 正确性

- 先确认命令作用于哪个实现：不带 `--solution` 是学生代码，带 `--solution` 是参考答案；`bench` 中的 `cublas`、`cudnn`、`fa4` 等是库基线。
- 先用小 case 定位：`./lab check ID --step n --case <小 case>`，再跑本关完整检查 `./lab check ID`。
- 失败要如实报告。不删除测试、不无条件 skip、不放宽容差。
- 库基线不可用时，`bench` 显示 “unavailable”。不要用别的实现静默顶替。

## 性能

- 比较必须使用相同的输入、输出契约和计时边界（排除编译、`make_plan`、输入生成与分配）。
- 不混用计时模式：默认的 CUDA Graph 结果不能与 `--cold` 结果互相计算比值。
- 区分实测、源码推断和未验证的假设；保留失败或变慢的结果。
- 不自动运行长时间 sweep。工具用法见 [工具指南](docs/tools.md)。

## 环境

- 通过 `./lab` 使用独立 Conda 环境 `flash-attention-lab`（可用 `LAB_CONDA_ENV` 修改）。不改 base 环境、系统工具链或其他用户的进程。
- 只用一张 GPU：由 `LAB_GPU` 选择，并在已有 `CUDA_VISIBLE_DEVICES` 范围内。先用 `nvidia-smi` 找空闲的卡。
- 日志、测量结果、profile 报告、个人进度写入 Git 忽略的目录（`artifacts/`、`.cache/`、`progress/`）。
- 只有学习者明确要求时才提交；提交的授权不包含推送。
