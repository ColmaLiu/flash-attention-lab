# 编写关卡（维护者指南）

本文件规定一关由哪些文件组成、各自要满足什么条件。设计背景见 [design-v2.md](design-v2.md)；样板关是 [00_transpose](../course/00_transpose/README.md)，新关卡按它的结构和详略来写。

## 1. 文件

```
course/<ID>/README.md     讲义
course/<ID>/spec.json     case、step、性能目标
students/<ID>/kernel.cu   学生骨架
solutions/<ID>/kernel.cu  参考答案（可以 #include 同目录的 .cuh）
solutions/<ID>/NOTES.md   答案设计说明
include/lab/<name>.cuh    （仅引入新原语的关卡）公共原语的参考实现
```

`labkit/` 和 `include/lab/{lab,problem,register,check,scratch}.h` 是公共基础设施。只有关卡确实需要时才改，例如在 `labkit/kinds/<kind>.py` 中加一个输入 pattern 或 probe oracle；改动要小，并且保持向后兼容。

## 2. spec.json

```json
{
  "id": "00_transpose",
  "title": "访存与线程映射：矩阵转置",
  "part": 0,
  "kind": "transpose",
  "prerequisites": [],
  "steps": [
    {"step": 1, "title": "朴素转置", "cases": "all"},
    {"step": 2, "title": "单 tile", "cases": ["tiny", "one_tile"]},
    {"step": 3, "title": "只检查中间结果", "cases": ["tiny"], "probe": {"oracle": "scores"}}
  ],
  "cases": [
    {"name": "tiny", "shape": [7, 11], "dtype": "fp32", "pattern": "coordinate",
     "purpose": "这个 case 用来发现哪一类错误"},
    {"name": "big", "shape": [8192, 8192], "dtype": "fp32", "perf": true, "purpose": "..."}
  ],
  "perf_target": {"baseline": "torch", "ratio": 0.9},
  "profile_metrics": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
}
```

- **kind** 决定 Problem 结构体和 oracle：`transpose`、`softmax`、`gemm`、`attention`，定义在 `include/lab/problem.h` 与 `labkit/kinds/`。
- **shape**：transpose/softmax 为 `[rows, cols]`，gemm 为 `[M, N, K]`，attention 为 `[B, N, H, D]`。
- **其余 case 字段**原样进入 `case.options`，例如 attention 的 `causal`、`scale`，gemm 的 `c_dtype`。
- **pattern** 由 kind 的 `make_inputs` 解释。现有 pattern：
  - transpose：`random`、`coordinate`；
  - softmax：`random`、`large`、`shifted`、`constant`、`masked`；
  - gemm：`random`、`int`；
  - attention：`random`、`large_logits`、`uniform`、`ones_v`、`position`。
- **steps**：`"cases": "all"` 表示全部非 perf case。每个 step 运行时 `p.step` 等于该 step 的编号。不指定 `--step` 的 bench 和 check 都用最后一步，最后一步不能带 probe。
- **probe**：该 step 不检查 O 等最终输出，而是检查 `p.probe`（FP32 缓冲）。oracle 由 `labkit/kinds/<kind>.py` 中的 `probe_<oracle>(inputs, options, case)` 定义，返回 `(期望张量, (atol, rtol))`。
- **case 设计**：
  - 每个 case 写清 `purpose`，即它针对哪一类错误；
  - 必须覆盖尾部（非 tile 倍数）、B 与 H 均大于 1、causal/非 causal、各 dtype 与 D；
  - 每关 1–3 个 perf case，规模要大到能体现 HBM 或 Tensor Core 的瓶颈：输入超过 L2（126 MB），或 attention 达到 N ≥ 4096、B·H ≥ 16。
- **perf_target**：答案在全部 perf case 上相对该基线的最低比值（`基线耗时 / 答案耗时`），由 `./lab selftest --perf` 检查。中间关卡设一个能说明机制成立的值；每类算子的最终关卡设为 0.9，即 transpose `30_tma_mbarrier`、softmax `01_softmax`、GEMM `33_2cta_gemm`、attention `43_fa4_schedule`。

## 3. 学生骨架

- 能编译。每个 step 的缺失部分用 `LAB_TODO(n)` 标出，n 尽量等于 step 编号。设备代码中的 `LAB_TODO` 会让 kernel trap，harness 报告“TODO(n) 未实现”。
- 不能有死锁：mbarrier 等待、生产者/消费者循环等必须在到达 TODO 之前就 trap，或者由 TODO 所在的 host 端代码阻止 launch。
- 非本关重点的代码要写好：launch 配置、host 端描述符编码、本关之前学过的原语。
- 每个 TODO 前的注释写清：输入是什么、输出写到哪里、不变式、可以调用哪些 helper。不要写出答案的关键一行。
- `run()` 按 `p.step` 选择版本，让学生保留每一步的实现，可以互相对比性能。若各 step 是同一个 kernel 逐步补全，就不需要分支。
- 只 `#include` 本关已经开放的 `include/lab` 头文件（见设计稿 §5 的表格）。引入新原语的关卡，骨架里写出原语的函数签名和语义注释，由学生实现函数体。
- 文件头：一行说明加讲义路径，再写数据布局约定。

## 4. 参考答案

- 只用本仓库可读的 CUDA C++ 和内联 PTX，不用 CUTLASS/CuTe/库 wrapper 代替答案。
- 只实现本关引入的机制，不提前包含后续关卡的优化。可以复制前一关答案的代码作为起点。
- 每个 step 都能通过 `p.step` 运行，并通过该 step 的全部 case。
- 注释解释“为什么”：线程分工、数据在哪一级存储、每个同步保护什么。
- 正确性、无数据竞争、性能三项都要验证（见第 6 节）。
- `NOTES.md` 写：设计选择及理由、各 step 的性能变化及原因、已知局限。不写本机绝对时间，比例可以写。
- 引入新原语的关卡把原语放进 `include/lab/<name>.cuh`，答案 `#include` 它。
  - 每个函数注释 PTX 语义，并给出 PTX ISA 章节名；
  - 命名空间 `lab`，函数名用 snake_case；
  - 函数只封装单条指令或固定的指令组合，不包含算法逻辑。

## 5. 讲义（README.md）

用中文，按 00 的九节结构：

1. **本关目标**：做什么，做完能做到什么；
2. **动机**：上一关的瓶颈，对应 FA 论文哪一节；
3. **数学**：推导、不变式、手算小例；
4. **硬件与指令**：新指令的语义、操作数布局、最小代码片段、常见误用；
5. **调度设计**：线程/warp 分工、各级存储的预算、时序图；
6. **分步练习**：每个 step 的 TODO、命令、常见错误表；
7. **性能**：各 step 相对基线的实测比例、该看哪些 profile 指标；
8. **自测问题**：3–6 题，答案放在 `<details>` 中；
9. **延伸阅读**：论文章节、PTX ISA / CUDA 文档链接、上游源码位置（`third_party/` 中的文件路径）。

写作要求：

- 先定义再使用，一个句子只讲一件事，不堆砌限定语；
- 多用表格、ASCII 图和手算的小例子，数字要能和 `./lab` 的输出对上；
- 讲到 B200 硬件数据时给出具体数值（SM 数 148、每 SM 228 KB shared memory、TMEM 每 SM 128 行 × 512 列 × 32 bit、L2 126 MB、HBM 标称 8 TB/s 等），不确定的数值要先查证或实测；
- 不写维护者流程；讲义只面向学习者。

## 6. 验证

使用一张空闲 GPU：先用 `nvidia-smi` 确认，再设 `LAB_GPU=<n>`。

```bash
LAB_GPU=n ./lab check ID --solution                 # 全部 step、全部 case
LAB_GPU=n ./lab selftest ID                         # 答案通过 + 骨架在 TODO 处停下
LAB_GPU=n ./lab sanitize ID --solution --case <小 case> --tool memcheck
LAB_GPU=n ./lab sanitize ID --solution --case <小 case> --tool racecheck
LAB_GPU=n ./lab sanitize ID --solution --case <小 case> --tool synccheck
LAB_GPU=n ./lab bench ID --solution --step k        # 每个 step
LAB_GPU=n ./lab sass ID --solution                  # 寄存器、spill、指令
LAB_GPU=n ./lab selftest ID --perf                  # 有 perf_target 时
python -m pytest -q                                 # 在 scripts/activate.sh 环境中
```

另外手工确认：

- 把骨架的每个 TODO 换成答案代码后，对应 step 能通过。可以在临时副本中做，完成后恢复骨架。
- 用 `./lab check ID --step n` 逐步运行时，第 n 步只依赖前 n 个 TODO。
- 不要运行长时间的 sweep，不占用多张 GPU。
