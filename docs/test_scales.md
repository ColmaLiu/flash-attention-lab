# 测试与工作负载设计

每个 case 都有稳定的 `id`、`purpose` 和 `suite`。`debug` 用于最小数据流，`edge` 覆盖边界/退化维度，`stress` 针对数值范围或多次资源复用，`perf` 选择可以讨论吞吐的规模。它们都参与全关正确性验收，分类不会豁免测试。

矩形转置检查地址方向、尾部及长宽比；softmax 同时包含单元素、短/宽行、全负值和整体平移；GEMM 包含非方阵、坐标模式、长 K，以及 QK/PV 两种比例。attention 包含 B/H 不相等的多头输入，单 token、非默认 scale、均匀 logits、较大 logits、causal/tail 和递增最大值。uniform attention 保持 Q/K=0、V 随机，以均值或 causal 前缀均值检验输出，不让全零 V 掩盖错误。

性能样本保持跨关可比：attention 少 heads `(1,1024,2,64)`、多 heads `(2,1024,16,128)`、长序列 `(1,4096,8,128)`；shape 为 BNHD。GEMM 使用方阵、短 K 大输出、长 K 窄输出。课程样本不代表某模型全部输入分布，不包含 decode、GQA 或 MLA。

`domain` 决定合法输入，而非这些测试尺寸。学习者应增加基于失败假设的案例，特别是在改变 tile、buffer 数或近似策略后；新增案例需要写出意图，不盲目扩大笛卡尔积。性能 sweep 由用户显式运行，基础设施默认不长期占用 GPU。
