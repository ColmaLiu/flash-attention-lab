# 数值验收

数学 oracle 在 PyTorch 中显式计算 FP64 分数、减最大值后的指数、分母、PV 和 LSE。每个 query 行始终一次性对所有 key 求归一化；大输入仅按独立的 batch/head/query 行分组，将单个 FP64 score 张量限制在 32 MiB 内，其他临时张量和输入输出仍需额外显存。causal mask 使用原始 query/key 的绝对位置。这里没有 key 分块状态或 online softmax，也不实现学生的 CUDA 算法。不依赖 SDPA 或官方 FA 的正确性。

准则为每个元素 `abs(actual-reference) <= atol + rtol*abs(reference)`，全部元素必须满足，同时拒绝 NaN/Inf。报告最大绝对误差、最大带 floor 的相对误差、失败元素数和首个失败坐标。诊断相对误差分母为 `max(abs(reference),max(atol,1e-12))`，不能把近零值放大的相对误差单独解释为失败。

| 输出/计算 | atol | rtol | 理由 |
|---|---:|---:|---|
| transpose | 0 | 0 | 无近似需求 |
| FP32 softmax、LSE | 2e-5 | 2e-5 | 声明输入模式下的 FP32 reduction 与指数误差预算 |
| FP32 attention O | 5e-5 | 5e-5 | 两次点积与归一化累计误差 |
| FP16 attention O | 3e-3 | 3e-3 | 包含 P 与 O 的低精度舍入预算 |
| BF16 attention O | 2e-2 | 2e-2 | BF16 尾数更短，单独预算；不是 FP32 测试阈值 |
| FP32/FP16/BF16 GEMM → FP32 | 2e-3 | 2e-4 | oracle 使用相同量化输入，预算主要对应 FP32 累加 |

这些是本实验受限输入规模下的预设工程预算，不是所有 shape/任意有限数值的误差定理。随机、零值、常量、大 logits、负数、坐标模式、抵消与递增最大值分别覆盖相应关卡。softmax 另检验非负和每行和；所有关检查输入未被改写、输出预填 NaN 后全部写入、重复启动以及非默认 stream。

大尺寸 case 使用相同容差。随机 GEMM 的输入为已量化的标准正态数，代表性长 K=2048；长 K 的归约次序可能改变误差，不能用增大容差代替检查。可独立比较相同输入的 FP32 数学运算与 FP64 oracle，区分 oracle/预算问题和学生实现问题，该对照不属于学生执行路径，也不证明学生正确性。

若出现数值问题：固定 case 与 seed，核对量化输入、scale、底数、mask 和首次失败坐标，再缩小。**不得靠调大 tolerance 通关**。若扩大输入数值范围或引入新的近似方法，应先给出独立误差实验与预算依据，再审阅契约变更。算法变更、输入精度变更、P 的转换和 Tensor Core 引入必须分别做消融。

注意 logits 全部偏移与 Q/K 放大不同：前者用于 stable softmax 的平移不变性，后者改变注意力分布并放大 QK 的舍入影响。causal 的有效行至少有对角线，但一个处理中间 tile 可以全部被屏蔽。
