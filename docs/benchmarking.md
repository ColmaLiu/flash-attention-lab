# 性能比较的含义

每课比较学生 kernel、完整 CUDA C++/PTX 参考答案与优化库。答案为学习者提供可以逐段阅读和继续优化的性能起点，FP64 oracle 单独判断数学正确性。transpose/softmax 使用 PyTorch 原生 CUDA 算子，GEMM 使用 cuBLAS，FP32 attention 使用显式矩阵乘与 native softmax；低精度 attention 强制 cuDNN fused 路径并返回 O+LSE。缺少所需后端时明确失败，不以数学 fallback 冒充快基线。参考答案也需要接受完整性能检查；运行库得到的快数字不能代替答案自身的速度。

`./lab reference ID --case N` 单独检查并测量库；`./lab solution ID --case N` 检查并测量完整答案；`./lab bench ID --case N` 先检查学生在该输入上的正确性，再自动运行同输入的双边比较。全关 `check` 仍是完整正确性依据。开始时选择 case 0，性能结论使用 task 表中 `suite=perf` 的案例。

计时排除编译、输入生成、oracle、workspace/output 预分配。需要的布局转换不能隐去：公共 attention 已是 BNHD；某参考内部若变换布局，应明确计入设备工作或单独列出准备成本。O 与 LSE 都属于本课程 attention 输出；只有 O 的参考不能当作等价工作量。

CUDA Graph 测量重复的预分配设备工作，减少 Python dispatch 间隙。eager 测量当前 stream 上逐次 host 发起的工作，可能包含调度间隙。两者都不等于包含输入生成和编译的应用端到端延迟。报告原始 samples、median/min/max、波动、warmup 和重复次数；交错双方测量以减少温度、频率与竞争造成的顺序偏差。运行期间的其他 GPU 工作会使一次比值难以解释，不应修改系统配置来制造结果。

统一使用 `reference_ms / candidate_ms` 表示相对参考的 speedup，大于 1 表示候选更快；用百分比时说明分母。不同 dtype、mask、输出集合、shape、graph/eager 不混算。GEMM 名义 FLOPs 为 `2MNK`；attention 的两次矩阵乘为非 causal `4BH N²D`，causal 有效元素计数为 `2BH N(N+1)D`。这些是有用计算量，不是实际硬件指令计数。

copy/transpose 可按读写有效字节计算有效带宽；它不能直接称为 DRAM 带宽，因为 cache 命中、重复访问与事务粒度不同。用 NCU 获取 counters，另列算法 IO 模型及假设。小 shape 可能由 launch 成本主导，长序列可能受资源与工作分配影响；一张最快数字不能概括实现。

参考答案的性能验收覆盖本课全部 `perf` cases。对比优化库之外，还应在共同支持域比较相邻阶段，确认本课引入的机制在哪些输入上有收益、哪些输入上有代价。明显落后的主要案例必须定位到访存、非矩阵工作、同步、并行度或资源用量，并据此改进；不能只展示最好的输入，也不能用“教学实现”免除优化工作。

性能门槛要对应具体课题与工作负载：SIMT FP32 GEMM 和 SM100 Tensor Core GEMM 不共享一个百分比要求。记录准确的 kernel 名称、源码版本、GPU、输入与 graph/eager 结果；没有测量数据时不宣称“足够快”或完成性能优化。学生验收还包含目标设计证据，不能仅凭数值通过或达到一个速度比例判断理解。
