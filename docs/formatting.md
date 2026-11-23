# 代码格式

本项目用三样工具统一格式：

| 工具 | 管什么 |
|---|---|
| clang-format | CUDA / C++（`.cu`、`.cuh`、`.h`、`.cpp` 等） |
| Ruff | Python 的格式与静态检查 |
| pre-commit | 在 `git commit` 前自动运行上面两个，以及空白、换行、大文件等检查 |

版本固定在 `requirements-lint.txt` 和 `.pre-commit-config.yaml`，升级时两处一起改。

## 安装与运行

在课程的 Conda 环境中安装并启用：

```bash
source scripts/activate.sh
python -m pip install -r requirements-lint.txt
python -m pre_commit install              # 以后每次 git commit 都会检查暂存的文件
python -m pre_commit run --all-files      # 检查全部已跟踪文件
```

第一次运行会安装隔离的 hook 环境，比较慢。`--all-files` 只检查已跟踪的文件；新文件先 `git add`，或用 `--files` 指定：

```bash
python -m pre_commit run --files solutions/32_ws_gemm/kernel.cu include/lab/tcgen05.cuh
```

如果 hook 修改了文件，先看 diff，确认无误后重新 `git add`，再提交。

## 风格

| 语言 | 风格 |
|---|---|
| CUDA / C++ | LLVM 风格，两空格缩进，80 列；不把短函数、短条件、短循环压成一行；保留 include 顺序 |
| Python | 四空格缩进，88 列；单行多条语句会被展开 |

inline PTX 字符串的内容不会被改动。

## 不会被格式化的目录

`.pre-commit-config.yaml` 排除了 `third_party/`、`artifacts/`、`.cache/`、`progress/`：

- `third_party/` 是固定版本的上游源码；
- 其余是 Git 忽略的输出目录。

`students/` 也在检查范围内：提交练习代码时，hook 会格式化暂存的 `kernel.cu`。注意 `run --all-files` 会直接改写工作区里所有已跟踪的文件，包括还没写完的练习；只想检查正在改的文件时，用 `--files`。

## 格式化之后

格式化通过不代表正确。改了公共代码（`include/lab/`、`labkit/`、`solutions/`）后，还要：

```bash
python -m pytest -q tests/test_labkit.py          # 基础设施测试
LAB_GPU=<n> ./lab check <ID> --solution           # 受影响的参考答案重新编译并检查
```
