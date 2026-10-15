# 验证流程

仓库提供检查方法和验收契约，不随分发附带任何机器的通过记录。每个新环境均应自行执行下列检查。27 个学生 kernel 初始未实现，任何基础设施成功都不能将其标记为正确性通过。

## 基础设施

在仓库根目录运行：

```bash
./lab doctor
./lab python -m pip check
./lab python -m pytest -q
./lab python -m pytest tests/student --collect-only -q
./lab python scripts/verify_scaffolding.py
./lab smoke --timeout 600
./lab status
```

| 检查 | 完成标准 |
|---|---|
| doctor | 所选设备和所需 backend 可用；区分 driver、Toolkit、PyTorch runtime、DSL；默认显示摘要，`--verbose` 展开诊断，`--json` 输出完整报告 |
| pip check | 无依赖冲突 |
| 基础设施 pytest | 契约、独立参考、不正确输出拒绝、前置依赖、超时、运行配置测试通过 |
| 学生测试发现 | 发现 27 个学生验收入口；发现成功不代表执行成功 |
| verify_scaffolding | 全部学生模块导入；未实现关的 check/bench 均返回 2，错误包含关卡名；已实现关另行显式验收 |
| smoke | 选定阶段的输入/host 校验/FP64 参考有效；CUDA 空绑定明确报 TODO；独立 CUDA/CuTe 标量 sentinel 编译、调用、计时正常 |
| status | 列出所有关卡、验收状态和原因；新克隆无历史正确性结果 |

smoke 可以用 `--stage 0` 到 `--stage 4` 分阶段运行。首次编译时间较长，可调整 timeout。sentinel 只写一个标量，不含练习答案，不进入学生调用路径；它的计时只验证设施。

## 工具检查

工具通过 PATH 或配置变量选择，先确认安装版本和可用功能：

```bash
source scripts/activate.sh
lab_sanitizer="$(python -c 'from flash_attention_lab.runtime import require_tool; print(require_tool("compute-sanitizer", "LAB_SANITIZER"))')"
lab_ncu="$(python -c 'from flash_attention_lab.runtime import require_tool; print(require_tool("ncu", "LAB_NCU"))')"
"$lab_sanitizer" --version
"$lab_ncu" --list-sections
```

使用独立 sentinel 验证工具调用；首次运行先用 smoke 完成编译：

```bash
# 沿用上面激活的环境和发现的工具路径。
mkdir -p artifacts/profiles
timeout 120 "$lab_sanitizer" --tool memcheck --error-exitcode 3 python scripts/tool_probe.py
# 需要时分别改用 --tool racecheck 和 --tool synccheck。
timeout 180 "$lab_ncu" --nvtx --nvtx-include infra_tool_probe/ --section LaunchStats --section MemoryWorkloadAnalysis --section SpeedOfLight -o artifacts/profiles/infra-probe python scripts/tool_probe.py
```

缺少工具、计数器权限不足或工具不支持所需指令时，记录具体限制，不能当作通过。报告写到忽略的 `artifacts/`；再次采集时选择新的输出名。学生 profiler 会在启动前检查所需 NCU sections。详见 [调试文档](debugging.md)。

## 学生验收

```bash
./lab check s00_softmax --case 0
./lab check s00_softmax
./lab sanitize s00_softmax --tool memcheck --case 0
./lab bench s00_softmax --case 0
```

实现前，显式 check/bench 应返回 2 并报告 STUDENT_UNIMPLEMENTED。实现后，依据本关声明 cases、重复调用、非默认 stream 和必要的内存安全检查验收。数值通过、内存安全、论文机制实现和性能是不同结论，设计证据写入对应实验记录。官方 FA4 对照按 [独立 benchmark 说明](benchmarking.md) 另行安装和执行。
