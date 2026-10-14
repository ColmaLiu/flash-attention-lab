# s37_tcgen05 实验记录（由学生填写）

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
- 本关问题的回答：固定 A/B 比较 FP16/BF16，保存合法指令 shape、SMEM 对齐和 TMEM 生命周期证据。
- 源码/设计验收证据：使用 SM100 tcgen05 指令证据；说明为什么 Hopper wgmma 的寄存器协议不能直接搬来。此关不使用 2-CTA mode。
- 尚未解决的问题/下一次区分性实验：
- Review 结论/进入下一关的依据：
