# flash-attention-lab 学习协作约定

本项目帮助学习者从零实现并优化 GPU attention，主要使用 CUDA C++ / PTX，面向
Blackwell SM100。课程分为 0–4 五个阶段。这里的约定面向辅导学习者的 agent。

- 先阅读当前关卡的任务说明、学习者代码与笔记，围绕学习者提出的问题提供解释、
  提示、调试或代码审查。未经明确要求，不替学习者完成练习，也不覆盖已有实现与笔记。
- `students/` 是练习区，初始核心未实现时应明确报错。学生入口不能回退到参考答案、
  数学 oracle 或库实现，不能用调用库的 wrapper 代替需要练习的 kernel。
- `solutions/` 是独立、可读的 CUDA C++ / PTX 参考答案，可按学习者需要阅读或运行。
  学习任务默认不修改参考答案、课程定义或测试基础设施；发现疑似错误时说明证据，
  按明确的修复请求处理。
- 数学 oracle、参考答案、优化库与学生实现各有独立入口。比较时确认实际 backend，
  不以静默 fallback 冒充优化库。接口和支持域以关卡说明及 `curriculum.json` 为准。
- attention 公共接口为 contiguous `[B,N,H,D]`，输出 O 同布局，LSE 为 `[B,H,N]`
  FP32。仅 self-attention forward，无 dropout、bias 或 GQA。
- 先用小案例定位问题，再运行本关完整 correctness 检查和代表性性能案例。失败要明确
  报告，不删除测试、不无条件 skip、不放宽容差来获得通过。数值正确之外，还要检查
  数据布局、线程分工、同步和缓冲生命周期，帮助学习者理解目标机制。
- 性能比较使用相同输入、输出契约与计时边界，排除编译、输入生成和分配；不要混用
  CUDA Graph 与 eager 计算 speedup。区分实测、源码推断和未验证假设，保留失败结果。
  参考答案的维护验收与学生学习验收不同，详见 [性能说明](docs/benchmarking.md)。
- 通过 `./lab` 使用独立 Conda 环境 `flash-attention-lab`（可配置 `LAB_CONDA_ENV`）。
  默认只用 `LAB_GPU` 选择的一张 GPU，尊重 `CUDA_VISIBLE_DEVICES`；不修改 base、
  系统工具链或其他用户进程。不自动运行长 sweep。
- 日志、机器快照、测量结果、个人进度与旧版本归档放入 Git 忽略目录。保留第三方
  许可和固定的依赖版本。公共代码格式见 [格式化说明](docs/formatting.md)，批量工具
  默认排除学生练习。
- 常用入口为 `./lab task ID -v`、`./lab check ID`、`./lab bench ID --case N`；
  `solution` 和 `reference` 分别运行参考答案与库基线。先确认命令作用于哪个实现。
- 只有学习者明确要求时才提交或推送代码；提交授权不包含推送授权。
