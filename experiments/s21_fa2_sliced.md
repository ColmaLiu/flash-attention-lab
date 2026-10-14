# s21_fa2_sliced 实验记录（由学生填写）

状态：未开始；数值验收和设计验收分开记录。

- 日期/代码 hash/命令/原始 artifact：
- GPU UUID/driver/Toolkit/PyTorch CUDA/DSL：
- shape/layout/dtype/scale/mask/seed：
- 算法 tile 与 CTA/warp/thread 映射（不要混用概念）：
- 各 tensor 的产生、保存、复用、搬运与生存期：
- launch/grid/block/SMEM/TMEM/register/stages 配置与理由：
- 不变量、同步 happens-before 图或 ownership 证明：
- 正确性最大绝对误差/带 floor 相对误差/首次失败案例：
- sanitizer 命令/结果/覆盖范围：
- benchmark warmup/repeats/iterations，median/min/max/stdev：
- 本次唯一改变的变量/未控制变量：
- profiler 原始报告与实际支持的 metric：
- 算法 IO 推导 vs 实测 DRAM/L2/SMEM：
- 本关问题的回答：固定 grid、dtype、shape 对比 s20；计数归一化操作、SMEM 读写和同步次数。不得仅凭更快就认定 sliced-Q 已实现。
- 源码/设计验收证据：画 CTA 内各 warp 的 Q/K/V/O 所有权；用源码/NCU 证据解释跨 warp 部分输出合并是否消除。s20+s21 才构成本仓库 FA2 算法结构版。
- 尚未解决的问题/下一次区分性实验：
- Review 结论/进入下一关的依据：
