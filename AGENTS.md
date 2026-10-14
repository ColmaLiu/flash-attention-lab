# flash-attention-lab 协作规则

本仓库用于用户亲手实现 GPU kernel。以下规则约束所有后续 AI，适用于整个仓库。

- AI 可实现构建、依赖、host 参数校验/分配、绑定、独立数学参考、测试、benchmark、profiling、文档及不涉及练习目标的工具。
- `students/` 的设备计算、thread/warp/CTA 映射、reduction、online softmax、attention、寄存器/SMEM/TMEM layout、搬运、同步、流水线和 Tensor Core 协作必须由用户实现。属于练习的 launch/grid/block/shared-memory 配置也不能由 AI 预填。
- 不得在注释、测试、参考实现、隐藏文件、其他分支中放入练习答案；不得提供可直接翻译成完整 kernel 的详细伪代码。TODO 按有意义的模块设置，不拆成逐行填空。
- 不得用 PyTorch、SDPA、官方 FlashAttention 或其他 kernel 替代学生入口。参考、官方对照、trivial smoke 必须独立命名，不能接入学生执行路径。
- 核心入口未实现时明确报错、返回非零；基础设施通过不等于学生正确性通过。禁止空测试、无条件 skip/xfail 或放宽容差制造通过。
- 用户提交实现后，先复现和缩小失败案例；默认一次给一个针对性提示，按概念 → 定位 → 局部伪代码 → 局部代码递进。未经明确请求，不修复或重写学生 kernel；review 给原因和最小失败案例。基础设施 bug 可以修复并说明范围。
- 保留用户代码、笔记、实验与旧版本入口。补充脚手架前先读现有文件，禁止重跑生成脚本覆盖学生代码。
- 同时维护阶段 0–4 的可执行脚手架；学习可以逐关进行，不提前生成下一关核心答案。
- 仅 forward、contiguous self-attention、无 dropout/bias/GQA。Backward、varlen、paged attention 留作以后，不搭建无关框架。
- 默认只使用 `LAB_GPU` 指定的一张 GPU；不终止其他用户进程，不更改系统 GPU 配置或全局工具链。使用独立 Conda 环境 `flash-attention-lab`（可通过 `LAB_CONDA_ENV` 指定），不在 base 安装项目依赖。实际检查 GPU 后才选择编译目标。
- 仓库是可公开克隆的学习模板。提交内容只包含通用源码、依赖声明、来源版本、任务和验证方法；不得包含个人路径、机器环境快照、实际测试/性能结果、会话记录，或依赖未提交文件的入口。
- 区分驱动支持 CUDA、Toolkit、PyTorch runtime 和 CuTe 编译器版本。工具路径通过配置或 PATH 发现，尊重已有 CUDA_VISIBLE_DEVICES；运行生成的日志、报告、缓存和个人进度必须被 Git 忽略。检查结果向用户报告，不写成公共文档的既成结论。
- 性能必须先正确性验证，计时排除编译/输入生成/分配，说明布局转换和 host 开销。算法 IO 模型、实际 DRAM 计数和推测必须区分。不能把 B200 教学版结果说成历史官方 FA 性能。
- 机械正确性不能证明 FA1/FA2/FA4 的设计已经实现；必须另做文档中的源码/设计证据验收。状态记录必须绑定当前代码，变更后旧结果失效。
- 不自动提交 git、安装官方 kernel 或运行长 sweep。尊重用户后续明确授权。
