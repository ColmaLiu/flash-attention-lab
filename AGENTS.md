# flash-attention-lab 协作约定

本项目教学习者从零实现并优化 GPU attention，主要使用 CUDA C++ / PTX，面向 Blackwell SM100。课程有 0–4 五个阶段。

- `students/` 是练习区；初始化时核心为空，未实现明确报错。未经学习者要求，不替他们完成练习。保留学习者的代码与笔记。
- `solutions/` 提供完整、可读、独立运行的参考 kernel。允许 AI 实现与修复参考答案。逐关说明算法、实现边界、源码来源、性能局限；不能把调用库的 wrapper 称为完整 kernel 答案。
- 数学 oracle、教学 kernel、优化库基线与学生实现使用独立入口。学生路径不能回退到参考或库实现。库基线要明确实际 backend；不可静默降级到慢实现。
- attention 公共接口为 contiguous `[B,N,H,D]`，输出 O 同布局，LSE 为 `[B,H,N]` FP32。仅 self-attention forward，无 dropout/bias/GQA。
- 每关必须有概念讲解、前置联系、可读参考、按故障类型设计的 correctness cases、代表性性能 cases，以及源码/设计验收。数值通过不自动代表理解了设计。
- 性能报告先验证正确性。区分数学 oracle、教学实现与优化基线；报告相同输入/输出契约及计时边界，排除编译、生成输入和分配；不能混用 CUDA graph 与 eager 得出 speedup。
- 默认只用 `LAB_GPU` 选择的一张 GPU，尊重 `CUDA_VISIBLE_DEVICES`。使用独立 Conda 环境 `flash-attention-lab`（可配置 `LAB_CONDA_ENV`）；不改 base、系统工具链或其他用户进程。
- 源码与依赖固定版本，保留第三方许可。公开内容只包含通用代码、教学与验证方法。日志、机器快照、测量结果、个人进度及旧版本归档放进 Git 忽略目录。
- 改动后自检基础设施与相关参考实现。失败必须明确报告，不空测试、不无条件 skip、不靠放宽容差通过。
- 不自动 git commit/push，不自动运行长 sweep。外部 issue/PR 可在用户授权范围内使用。
