# s38_async_gemm 实验记录（由学生填写）

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
- 本关问题的回答：固定 K，比较单 stage 与多 stage；记录占用、SMEM bytes 和 stall，用 profiler 判断瓶颈而非猜测。
- 源码/设计验收证据：提交每个 buffer 的生产/消费/释放图；说明至少两个 stage 的实际重叠证据，单看 launch 个数不够。
- 尚未解决的问题/下一次区分性实验：
- Review 结论/进入下一关的依据：
