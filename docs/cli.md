# 终端输出

`./lab` 使用简洁英文输出。`--verbose` / `-v` 展开细节，`--json` 输出完整机器报告。终端标题使用少量颜色；管道、重定向或 `NO_COLOR=1` 时不输出颜色控制码。

| 命令 | 默认显示 | `--verbose` 增加 |
|---|---|---|
| `doctor` | GPU、驱动 CUDA 上限、Toolkit、PyTorch runtime、DSL 与工具状态 | 路径、编译器、失败诊断、Blackwell API 签名 |
| `list` | 按阶段分组的任务和案例数 | 前置关系和 backend |
| `task ID` | 前置、tensor 键/shape/dtype、学生文件、任务文档路径和验收命令 | 目的、全部案例与实验记录路径 |
| `status` | 各关状态和原因 | 完整原因和前置关系 |
| `check ID` | 每个 case 的 PASS/FAIL、输入、最大绝对误差；失败原因和复现命令 | 每个 stream、输出的误差、相对误差分母下限和容差 |
| `bench ID` | 输入、GPU、计时中位数/波动、采样数与计时范围；attention 的 FLOPs 口径 | 原始样本、配置、scratch 大小；不显示精度明细 |
| `smoke` | 导入、参考计算、绑定与 sentinel 的结果 | 空绑定名称和 sentinel 计时 |
| `sanitize ID` | 工具、case 和 PASS 或错误原因 | 日志和报告路径 |
| `profile ID` | case 和 NCU 报告路径，失败时给原因 | 日志和 JSON 报告路径 |

`task`、`check`、`bench`、`sanitize`、`profile` 的 ID 均支持短编号，例如 `s20` 对应 `s20_fa2_grid`。输出和报告仍使用完整 ID。只接受完整名称或完整短编号；`s2`、`s20_fa2` 等部分前缀会报错，未知编号可用 `./lab list` 查询。

`task` 不输出整篇教程；完整数学、设计验收与调试说明在其列出的任务 Markdown 中，保留中文。`status` 只查看实现标记和验收记录，不执行学生 kernel；代码变化会使旧记录失效，详见 [状态语义](../progress/README.md)。

## 正确性范围

`check` 的 PASS 要求该 case 的默认和非默认 stream 均通过，每种 stream 有两次独立调用。数值失败显示 FAIL 和原始错误原因，并继续检查其他案例；运行时错误显示 ERROR 并停止，剩余案例显示 NOT RUN，避免在损坏的 CUDA context 上继续执行。超时会终止本命令的子进程，不能据此推断未报告案例通过。

`--case N` 只检查指定案例，不能代替全关验收。数值通过不代表设计 review 或内存安全检查通过。smoke 的 sentinel 只验证基础设施。bench 仍在计时前后检查正确性；失败时不显示性能结果，精度明细保存在 JSON 中。输出格式不改变数值判据或容差。

## JSON 与日志

`--json` 可放在子命令前或后，优先于 `--verbose`。stdout 是单个对象，包含 `command`、`exit_code`；失败时有 `state`、`message` 和适用的复现信息。check 报告的 `cases` 保存每个案例的状态，`results` 保存已经执行的 stream 结果。

```bash
./lab check s00_softmax --case 0
./lab check s00_softmax --verbose
./lab bench s00_softmax --case 0
./lab status --stage 0 --json
```

check、bench、smoke、sanitize、profile 将完整报告写入已忽略的 `artifacts/runs/`，编译器和工具输出写入 `artifacts/logs/`。默认成功输出隐藏这些路径；失败或 `--verbose` 时显示。JSON 的 `artifact`、`log` 给出路径，NCU 成功时另有 `profile_report`。交互终端会显示一行运行提示，首次编译/JIT 可能耗时较长。

编译器与工具输出不会混入 JSON；工具非零退出不能被目标程序的数值通过覆盖。GPU stream 和 host batch 计时范围见 [benchmark 说明](benchmarking.md)，退出码见 [状态语义](../progress/README.md)。`./lab python ...` 原样转发 Python 输出和退出码；环境激活失败及 `--help` 使用原生输出。
