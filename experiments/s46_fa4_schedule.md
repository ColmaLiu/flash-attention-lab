# s46_fa4_schedule 实验记录（由学生填写）

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
- 本关问题的回答：固定 shape/dtype/数学路径比较调度；记录是否受 CTA 尾部、register、SMEM 或 TMEM 容量限制。官方结果只作独立对照。
- 源码/设计验收证据：给出无遗漏/无重复 ownership 证明；源码说明调度策略，测量 tail 与资源占用，而非仅给总时间。
- 尚未解决的问题/下一次区分性实验：
- Review 结论/进入下一关的依据：
