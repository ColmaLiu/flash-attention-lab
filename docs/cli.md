# 常用命令

`./lab --help` 和 `./lab COMMAND --help` 是当前参数的权威入口。

| 命令 | 意义 |
|---|---|
| `./lab list --stage 2` | 阶段课程 |
| `./lab task s22_fa2 -v` | 讲解位置、支持域与 case |
| `./lab reference s22_fa2 --case 0` | 独立快速库的正确性与计时 |
| `./lab solution s22_fa2 --case 0` | 完整参考答案的正确性与计时 |
| `./lab check s22_fa2 --solution` | 完整答案的全部声明 case 正确性 |
| `./lab check s22_fa2` | 学生全部声明 case 的正确性 |
| `./lab bench s22_fa2 --case 0` | 学生与快速库比较 |
| `./lab sanitize s22_fa2 --tool memcheck --case 0` | 内存安全检查 |
| `./lab profile s22_fa2 --case 0` | 指定案例 profiling |
| `./lab status` | 当前源码对应的学生状态 |

bench 不是全关通关记录；solution/reference 成功也不改变学生状态。JSON 报告和普通终端输出表达同一结论。日志/缓存/个人结果只写忽略目录。

`bench`、`solution` 和 `reference` 未提供 `--case` 时选择本课第一个 `perf` 案例；`check` 未提供 `--case` 时检查全部案例。`check --stage N --solution` 可验收该阶段所有答案。`sanitize` 和 `profile` 加 `--solution` 可检查答案。
