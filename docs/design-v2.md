# flash-attention-lab v2 设计

状态：设计稿，跟踪 issue 见 GitHub `v2` 标签。本文件描述 v2 的目标、课程结构、仓库组织、接口与验收方式；实现过程中如有变更，先改本文件再改代码。

## 1. 目标

让**只有一点 CUDA 基础**（写过 elementwise kernel，知道 grid/block/thread 与 shared memory）的学习者，在 **B200（SM100）** 上以 **FlashAttention 1→4** 为主线，学会编写和优化高性能 kernel。

学完后学习者应能：

1. 从数学定义推导 attention 的 IO 复杂度，解释 FA1→FA4 每一代解决了什么瓶颈；
2. 独立写出 mma.sync、TMA、mbarrier、tcgen05/TMEM、warp specialization、2-CTA MMA 的正确代码，并说清每条同步的作用；
3. 用 SASS、Nsight Compute 和受控计时验证优化假设，而不是凭感觉调参；
4. 读懂上游 FA4（CuTe DSL）与 CUTLASS SM100 FMHA 源码的结构。

非目标：Hopper wgmma；反向传播；decode/split-KV；GQA/MQA、dropout、bias、变长序列。

## 2. 设计原则

1. **先学明白再写。** 每关的讲义覆盖本关全部新知识：数学、调度、指令。学生不需要先去读论文或 PTX 手册才能开始；讲义给出出处供深入。
2. **小步可验证。** 每关拆成 2–5 个 step，每个 step 对应骨架里的 `TODO(n)`，可用 `./lab check ID --step n` 单独验证。学生不会面对“整关一次过或一次挂”的黑盒。
3. **一关一个新机制。** 新硬件机制先在 GEMM（或更简单的算子）上单独练，再用进 attention。参考答案只包含本关引入的特性，不提前泄露后续关卡。
4. **前面写过的，后面提供。** 学生在某关亲手写过的原语（如 mbarrier 封装、TMA 描述符编码），从下一关起由公共头文件 `include/lab/` 提供参考实现，避免前面的小错误拖垮后面的实验。
5. **性能结论可复现。** 计时排除编译与准备；CUDA Graph 与 eager 不混算；每关以“相对库基线的比例”给出学生目标，仓库不提交本机实测数据；维护者验收与学生输出分开。
6. **简单的基础设施。** 学生只需要理解 `kernel.cu` 里的 `make_plan` / `run` 两个函数；Python 侧只负责造数据、调用、对答案、计时。

## 3. 课程地图

ID 采用 `<部分><序号>_<名字>`。部分 0 预备，1–4 分别对应 FA1–FA4。

| ID | 主题 | 新知识 | 练习（step 概要） | 对照基线 |
|---|---|---|---|---|
| **部分 0：预备** |||||
| `00_transpose` | 访存与线程映射 | 线性地址、coalescing、shared memory、bank conflict、padding/swizzle | 朴素转置 → smem 分块 → 消除 bank conflict | torch copy |
| `01_softmax` | 归约与数值稳定 | warp shuffle、block 归约、safe softmax、**online softmax** 推导 | 两遍 softmax → 单遍 online softmax | torch.softmax |
| **部分 1：FA1 — IO 感知的分块** |||||
| `10_simt_gemm` | 分块 GEMM | 数据复用、算术强度、smem tiling、寄存器分块、向量化 | 朴素 → smem tile → 寄存器 thread tile | cuBLAS（FP32） |
| `11_naive_attention` | 标准 attention 与 IO 分析 | attention 数学、mask/scale、HBM 流量计数 | 用提供的 GEMM 组合：S、带 mask 的 softmax、PV；测量并解释 N² 流量 | 数学 oracle |
| `12_fa1` | FlashAttention-1 | 分块 + online softmax 的正确性证明；论文 Algorithm 1（KV 外层、O/ℓ/m 回写 HBM）；LSE | 单块正确 → 完整 KV 循环 → causal | 11 关实现、cuDNN |
| **部分 2：FA2 — Tensor Core 与并行划分** |||||
| `20_mma_gemm` | warp 级 Tensor Core | `mma.sync.m16n8k16` fragment 布局、`ldmatrix`、XOR swizzle | 单 warp 16×8×16 → CTA tile → 无冲突 ldmatrix | cuBLAS（BF16） |
| `21_pipeline_gemm` | 异步拷贝流水 | `cp.async`、commit/wait_group、多级 buffer、延迟隐藏 | 双缓冲 → 多级流水 → 调参与分析 | cuBLAS |
| `22_fa2` | FlashAttention-2 | 循环交换（Q 外层）、延迟归一化、warp 按 Q 切分、P 留在寄存器作 A 操作数、causal 跳块 | QKᵀ fragment → online softmax in registers → PV → causal | cuDNN |
| **部分 3：FA3 的思想在 SM100 上 — 异步与分工** |||||
| `30_tma_mbarrier` | TMA 与 mbarrier | tensor map、swizzle 模式、`cp.async.bulk.tensor`、expect_tx、phase/parity、TMA store | 单 tile 搬运 → 多级环形 buffer（full/empty barrier） | torch copy |
| `31_tcgen05_gemm` | tcgen05 与 TMEM | TMEM 分配/释放、smem descriptor、instruction descriptor、`tcgen05.mma/commit/ld` | 单 tile → 完整 K 循环 → 多 CTA | cuBLAS |
| `32_ws_gemm` | warp specialization | 生产者/消费者、TMA warp + MMA warp + epilogue warps、TMEM 双缓冲、persistent | 分工流水 → 累加器双缓冲 → persistent 调度 | cuBLAS |
| `33_2cta_gemm` | 2-CTA MMA | cluster、distributed smem、`cta_group::2`、TMA multicast | 2-CTA 单 tile → 完整流水 | cuBLAS |
| `34_fa3` | FA3 式 attention | FA3 三个思想（异步生产者、GEMM/softmax 重叠、pingpong）在 tcgen05 上的对应；S/O 在 TMEM、P 经 smem | 单 KV 块 → 完整循环 → S 双缓冲重叠 QKᵀ 与 softmax | cuDNN |
| **部分 4：FA4 — Blackwell 上的完整优化** |||||
| `40_fa4_pipeline` | FA4 流水结构 | 2 个 Q tile ping-pong、P 别名写回 TMEM（TS-MMA）、独立 correction warpgroup、寄存器重分配 | 角色划分 → P in TMEM → correction WG | cuDNN、FA4 |
| `41_fa4_softmax` | softmax 吞吐 | MUFU 瓶颈分析、多项式 exp2 软件模拟、条件 rescale 及误差界 | 多项式 exp2 → 混合 MUFU/FMA → 条件 rescale | 40 关、FA4 |
| `42_fa4_2cta` | 2-CTA attention | 2-CTA 的 QKᵀ/PV 划分、cluster 同步 | 2-CTA QKᵀ → 2-CTA PV | FA4 |
| `43_fa4_schedule` | 调度与 causal | causal 的块分类与掩码开销、persistent、LPT 排序 | causal → persistent → LPT | cuDNN、FA4 |
| `44_fa4_fp8` | FP8 前向 | E4M3 量化、descale、P 的 FP8 量化误差、`kind::f8f6f4` | FP8 QKᵀ → FP8 PV → 误差分析 | FA4 FP8（若支持） |

支持域（除非某关另行声明）：

- 部分 0–1：FP32；
- attention：`[B,N,H,D]` contiguous BF16/FP16，`D ∈ {64,128}`，`scale > 0` 有限，causal/非 causal，O 同布局，LSE `[B,H,N]` FP32（自然对数）；
- FP8 关：E4M3 输入加 FP32 descale，O 为 BF16。

各关在 `spec.json` 中声明自己的 dtype 和形状约束；binding 与 Python 两侧都检查支持域，越界输入明确报错。

## 4. 每关的组成

```
course/<ID>/
  README.md     讲义（见下方模板）
  spec.json     case、step、支持域、性能 case
  figures/      讲义用图（SVG/mermaid 源）
students/<ID>/kernel.cu      学生骨架（含 TODO(n)），学生直接修改
solutions/<ID>/kernel.cu     参考答案（可拆多个 .cuh）
solutions/<ID>/NOTES.md      答案设计说明：为什么这样分工、如何验证、已知局限（不含本机实测数据）
```

讲义模板：

1. **本关目标**：一段话，加前置关卡；
2. **动机**：上一关的瓶颈是什么，对应 FA 论文哪一节；
3. **数学**：推导、不变式、手算小例；
4. **硬件与指令**：本关新指令的语义、操作数布局、最小代码片段、常见误用；
5. **调度设计**：线程/warp 分工、smem/TMEM/寄存器预算、时序图；
6. **分步练习**：每个 step 写清要实现哪个 TODO、运行哪条命令、常见错误与排查；
7. **性能**：测哪些 case、相对库基线的目标比例、用 SASS/NCU 看什么指标；
8. **自测问题**：3–6 个，答案在讲义末尾折叠；
9. **延伸阅读**：论文章节、PTX 手册章节、上游源码位置（固定版本）。

骨架规范：

- 骨架能编译，`make_plan` / `run`、launch 配置、host 侧描述符编码等“非本关重点”的部分已写好；
- 每个 `TODO(n)` 注释说明输入、输出、不变式和可调用的 helper；
- 未完成时运行到 TODO 处明确报错（`LAB_TODO(n)`），不静默输出垃圾。

## 5. C++ 接口

每种算子一个 Problem 结构体，定义在 `include/lab/problem.h`：

```cpp
namespace lab {
struct AttentionProblem {
  const void *q, *k, *v;  // [B,N,H,D]
  void *o;                // [B,N,H,D]
  float *lse;             // [B,H,N]
  int b, n, h, d;
  DType dtype;
  float scale;
  bool causal;
  float q_descale, k_descale, v_descale;  // FP8 only
  float *probe;           // step 调试输出，可为空
};
}  // namespace lab
```

学生的 `kernel.cu` 只实现两个函数和一个 Plan 类型：

```cpp
struct Plan { /* TMA 描述符、scratch、launch 参数 */ };
Plan make_plan(const lab::AttentionProblem& p);                          // 不计时：编码描述符、分配 scratch
void run(const Plan& plan, const lab::AttentionProblem& p, cudaStream_t s);  // 计时：只发射 kernel
LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
```

- `LAB_REGISTER` 生成类型擦除的入口，供 `labkit` 中每种 Problem 对应的 TVM FFI binding 调用；
- Plan 中的 scratch 用 `lab::Scratch`（RAII）分配，分配时填入垃圾值，强制学生自行初始化；
- 早期关卡的 Plan 为空结构体，骨架已写好，学生从第一关就接触“准备/执行分离”。

GEMM 统一为 `C[M,N] = A[M,K] · B[N,K]ᵀ`：A、B 都是 K-major，与 QKᵀ 的形状一致；PV 中 MN-major 的 B 操作数在 attention 关卡讲解。

公共头文件 `include/lab/`：

| 头文件 | 内容 | 首次提供给学生 |
|---|---|---|
| `problem.h`、`register.h`、`check.h` | Problem、注册宏、CUDA 错误检查、`LAB_TODO` | 00 |
| `scratch.h` | 带垃圾值初始化的 scratch 分配 | 00 |
| `mma_sync.cuh` | `ldmatrix`、`mma.sync` 封装、fragment 坐标函数 | 21 之后 |
| `cp_async.cuh` | `cp.async` 与 wait_group | 22 之后 |
| `tma.cuh`、`mbarrier.cuh` | tensor map 编码、TMA load/store、mbarrier | 31 之后 |
| `tcgen05.cuh` | TMEM alloc/ld/st、descriptor、mma/commit | 32 之后 |
| `cluster.cuh` | cluster、2-CTA 辅助 | 34 之后 |

“首次提供”之前的关卡要求学生自己写这些原语。参考实现放在 `include/lab/`，但这些关卡的骨架与讲义不引用它；讲义说明这一安排。

## 6. Python 基础设施 `labkit`

命令：

| 命令 | 作用 |
|---|---|
| `./lab list` / `./lab status` | 课程列表与各关、各 step 进度 |
| `./lab check ID [--step n] [--case name]` | 编译学生代码并对照 oracle；默认运行全部 correctness case |
| `./lab bench ID [--case name] [--cold]` | 计时：学生、答案、库基线，同一输入，CUDA Graph |
| `./lab solution ID ...` | 对答案运行 check/bench |
| `./lab sass ID` | 编译产物的寄存器、spill、smem 用量与 SASS 摘录 |
| `./lab profile ID --case name` | 用本关预设的 NCU 指标采集 |
| `./lab sanitize ID --tool memcheck/racecheck/synccheck` | Compute Sanitizer |
| `./lab doctor` | 环境检查 |
| `./lab selftest` | 维护者：全部答案通过、骨架可编译且明确报 TODO、基础设施单测 |

正确性检查：

- 数学 oracle：PyTorch FP64（GEMM/attention 分块计算以控制显存）；
- 容差按 dtype 与归约长度给出，定义在 `labkit/tolerance.py`，并在讲义中解释；
- 输出与 scratch 都先填 NaN 或垃圾值；检查输入未被修改；默认流与非默认流各跑一次；
- step 通过 `spec.json` 中的 case 子集、`probe` 输出及对应 oracle 定义。

计时：

- 默认 CUDA Graph 回放，可选 eager，两者不混算；
- `--cold` 在每次迭代之间清空 L2；
- 报告中位数与分布、TFLOP/s 或 GB/s、相对 B200 标称峰值的比例，以及对答案和库基线的比值；
- 库基线：cuBLAS（经 torch）、cuDNN SDPA（强制后端，失败即报错）、FA4（`flash_attn.cute`，可选安装，未安装则显示“不可用”）；
- 维护者的答案性能门槛只在 `selftest --perf` 中检查，不出现在学生输出里。

进度：`progress/results/<ID>.json` 记录每个 step 的通过状态与该关源码指纹。指纹只包含 `students/<ID>/` 与其 include 的 `include/lab/` 文件。

## 7. 质量保证

每个关卡合入 `v2` 前须满足：

1. 答案通过全部 case（含 `--cold`、非默认流），`compute-sanitizer` 的 memcheck、racecheck、synccheck 无报错；
2. 骨架可编译，每个 step 在未实现时报 `LAB_TODO(n)`；
3. 讲义包含模板全部九节，命令可直接运行，链接有效；
4. 答案性能达到 `spec.json` 中声明的目标比例（相对库基线）；每类算子的最终答案（transpose：`30_tma_mbarrier`，softmax：`01_softmax`，GEMM：`33_2cta_gemm`，attention：`43_fa4_schedule`）在本关**全部** perf case 上逐项达到库基线的 90%（同次 CUDA Graph 中位数），不平均、不筛选；实测报告只写入 Git 忽略的 `artifacts/`；
5. 有一次独立审查（未参与编写的 reviewer agent），结论记录在对应 issue 中。

## 8. 从 v1 迁移

- v1 的 `flash_attention_lab/`、`students/cuda/`、`solutions/s*`、`docs/tasks/`、`curriculum.json` 在 v2 中整体替换；
- v1 中已验证正确的 kernel 代码（如 s40/s43 的 TMA、tcgen05、softmax 实现）作为 v2 答案的起点，按“一关一机制”拆分并修复已知问题（scale ≤ 0 的 NaN、极端 scale 分支、重复的 PTX 封装）；
- `third_party/` submodule 保留，用于源码对照；
- v1 保留在 `main` 分支，v2 稳定后再讨论合并方式。

## 9. 里程碑

| 里程碑 | 内容 |
|---|---|
| M0 | 本设计稿、issue 拆分 |
| M1 | `labkit`、C++ 接口、`include/lab` 基础头文件；`00_transpose` 端到端打通 |
| M2 | 部分 0–1（00、01、10、11、12） |
| M3 | 部分 2（20、21、22） |
| M4 | 部分 3（30–34） |
| M5 | 部分 4（40–44） |
| M6 | 全局文档（入门、B200 架构概览、工具指南、术语表）、selftest、全面审查、清理 v1 |
