# 数值验收

数学 oracle 在 PyTorch 中显式计算 FP64 分数、减最大值后的指数、分母、PV 和 LSE，保持独立的 full-matrix 定义；不实现学生应编写的 online/tiled CUDA 或 DSL 算法。不依赖 SDPA 或官方 FA 的正确性。

准则为每个元素 `abs(actual-reference) <= atol + rtol*abs(reference)`，全部元素必须满足，同时拒绝 NaN/Inf。报告最大绝对误差、最大带 floor 的相对误差、失败元素数和首个失败坐标。诊断相对误差分母为 `max(abs(reference),max(atol,1e-12))`，不能把近零值放大的相对误差单独解释为失败。

| 输出/计算 | atol | rtol | 理由 |
|---|---:|---:|---|
| copy/transpose/本仓库可精确表示的 affine、maximum | 0 | 0 | 无近似需求 |
| FP32 softmax/reduction/online、FP32 状态/LSE | 2e-5 | 2e-5 | 小规模有限输入的 FP32 reduction 与指数误差 |
| FP32 attention O | 5e-5 | 5e-5 | 两次点积与归一化累计误差 |
| FP16 attention O | 3e-3 | 3e-3 | 包含 P 与 O 的低精度舍入预算 |
| BF16 attention O | 2e-2 | 2e-2 | BF16 尾数更短，单独预算；不是 FP32 测试阈值 |
| 小规模 FP16/BF16 GEMM → FP32 | 2e-3 | 2e-4 | oracle 使用相同量化输入，预算主要对应 FP32 累加 |
| 指数小练习 FP32 | 1e-38 | 1e-4 | 独立的近似目标，避免低精度输出掩盖误差 |

这些是本实验受限输入规模下的预设工程预算，不是所有 shape/任意有限数值的误差定理。随机、零值、常量、大 logits、负数、坐标模式、抵消与递增最大值分别覆盖相应关卡。softmax 另检验非负和每行和；所有关检查输入未被改写、输出预填 NaN 后全部写入、重复启动以及非默认 stream。

若出现数值问题：固定 case 与 seed，核对量化输入、scale、底数、mask 和首次失败坐标，再缩小。**不得靠调大 tolerance 通关**。若未来扩大到更长序列或新的近似方法，应先给出独立误差实验与预算依据，再审阅契约变更。算法变更、输入精度变更、P 的转换和 Tensor Core 引入必须分别做消融。

注意 logits 全部偏移与 Q/K 放大不同：前者用于 stable softmax 的平移不变性，后者改变注意力分布并放大 QK 的舍入影响。causal 的有效行至少有对角线，但一个处理中间 tile 可以全部被屏蔽。
