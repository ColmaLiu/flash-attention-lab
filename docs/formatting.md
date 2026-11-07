# 代码格式

本项目使用 clang-format 整理 CUDA/C++，Ruff 整理并检查 Python，pre-commit
统一运行文件检查。

在课程的独立 Conda 环境中安装并启用：

```bash
./lab python -m pip install -r requirements-lint.txt
./lab python -m pre_commit install
./lab python -m pre_commit run --all-files
```

安装 hook 后，`git commit` 会检查暂存文件。若格式化工具修改文件，先检查 diff，
重新暂存，再提交。第一次运行会安装隔离的 hook 环境。
`--all-files` 只检查已跟踪文件；新文件先暂存，或用下面的 `--files` 显式指定。

只检查指定文件：

```bash
./lab python -m pre_commit run --files solutions/s32_specialize/kernel_impl.cuh
```

CUDA/C++ 采用 LLVM 风格、两空格缩进、80 列，不压缩短函数、条件或循环，保留
include 顺序。显式匹配 `.cu` 和 `.cuh`，避免只识别 C/C++ 时漏掉设备代码。
Python 使用四空格缩进、88 列，展开单行多语句。PTX 指令和约束仍保持原有语义。

工具默认排除 `students/`，以免批量改动学习者的实现和笔记；也排除第三方源码、
缓存和实验结果。学习者可以自行对正在编辑的文件运行 clang-format；如需对学生
Python 文件使用 Ruff，可显式运行
`./lab python -m ruff format --no-force-exclude students/具体文件.py`。

版本固定在 `requirements-lint.txt` 和 `.pre-commit-config.yaml`，升级时同步修改。

公共代码改动还需运行 `./lab python -m pytest -q`，CUDA 改动需重新编译并检查
受影响的参考实现。格式化通过不代表数值、同步或性能验收通过。
