# Code formatting

This project uses three tools to keep formatting consistent:

| Tool | What it covers |
|---|---|
| clang-format | CUDA / C++ (`.cu`, `.cuh`, `.h`, `.cpp` and so on) |
| Ruff | Python formatting and static checks |
| pre-commit | Runs the two tools above before `git commit`, plus checks for whitespace, line endings, large files and so on |

The versions are pinned in `requirements-lint.txt` and `.pre-commit-config.yaml`; when you upgrade, change both together.

## Install and run

Install and enable them in the course's Conda environment:

```bash
source scripts/activate.sh
python -m pip install -r requirements-lint.txt
python -m pre_commit install              # from now on every git commit checks the staged files
python -m pre_commit run --all-files      # check all tracked files
```

The first run installs isolated hook environments and is slow. `--all-files` checks only tracked files; `git add` new files first, or name them with `--files`:

```bash
python -m pre_commit run --files solutions/32_ws_gemm/kernel.cu include/lab/tcgen05.cuh
```

If a hook modifies files, read the diff first, `git add` again once it looks right, then commit.

## Style

| Language | Style |
|---|---|
| CUDA / C++ | LLVM style, two-space indent, 80 columns; short functions, short conditions and short loops are not squeezed onto one line; include order is kept |
| Python | Four-space indent, 88 columns; several statements on one line get split |

The contents of inline PTX strings are not changed.

## Directories that are not formatted

`.pre-commit-config.yaml` excludes `third_party/`, `artifacts/`, `.cache/` and `progress/`:

- `third_party/` holds the pinned upstream sources;
- the rest are Git-ignored output directories.

`students/` is checked too: when you commit student code, the hook formats the staged `kernel.cu`. Note that `run --all-files` rewrites every tracked file in the working tree, including exercises you have not finished; to check only the files you are editing, use `--files`.

## After formatting

Passing the formatter does not mean the code is correct. After you change shared code (`include/lab/`, `labkit/`, `solutions/`), also run:

```bash
python -m pytest -q tests/test_labkit.py          # infrastructure tests
LAB_GPU=<n> ./lab check <ID> --solution           # rebuild and check the affected reference solutions
```
