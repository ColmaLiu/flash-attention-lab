# 从 IO 模型到测量

显式 attention 的 S/P 有 BHN² 个元素。仅消除这两个中间矩阵，就能改变内存占用的量级；这不意味着总 Q/K/V/O 访存也自动达到最小。分块大小、遍历次序、cache 行为及输出状态的回写都会改变实际流量。

先按源码列出每种 tensor 的读写次数与复用层次，给出随 B/N/H/D 和 tile 参数变化的算法模型。再通过 NCU 的实际 counters 检验；事务粒度、cache 命中、spill 与重放会使 counters 偏离简单元素计数。最后用 nsys/时间线判断这些操作是否位于关键路径。

提高算术强度可以帮助 GEMM，但 attention 还包含指数、归约、layout 交接和同步。Blackwell 的矩阵吞吐与其他资源不同比例增长，不能只用 Tensor Core 峰值预测完整 attention。比较图表分别标明模型、测量和尚未验证的解释。
