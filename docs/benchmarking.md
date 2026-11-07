# 性能比较的含义

每课比较学生 kernel、完整 CUDA C++/PTX 参考答案与优化库。答案为学习者提供可以逐段阅读和继续优化的性能起点，FP64 oracle 单独判断数学正确性。transpose/softmax 使用 PyTorch 原生 CUDA 算子，GEMM 使用 cuBLAS，FP32 attention 使用显式矩阵乘与 native softmax；低精度 attention 强制 cuDNN fused 路径并返回 O+LSE。缺少所需后端时明确失败，不以数学 fallback 冒充快基线。参考答案也需要接受完整性能检查；运行库得到的快数字不能代替答案自身的速度。

`./lab reference ID --case N` 单独检查并测量库；`./lab solution ID --case N` 检查并测量完整答案；`./lab bench ID --case N` 先检查学生、答案与库在同输入上的正确性，再自动运行三方比较。全关 `check` 仍是完整正确性依据。开始时选择 case 0，性能结论使用 task 表中 `suite=perf` 的案例。

计时排除编译、输入生成、oracle、workspace/output 预分配。需要的布局转换不能隐去：公共 attention 已是 BNHD；某参考内部若变换布局，应明确计入设备工作或单独列出准备成本。O 与 LSE 都属于本课程 attention 输出；只有 O 的参考不能当作等价工作量。

CUDA Graph 测量重复的预分配设备工作，减少 Python dispatch 间隙。eager 测量当前 stream 上逐次 host 发起的工作，可能包含调度间隙。两者都不等于包含输入生成和编译的应用端到端延迟。报告原始 samples、median/min/max、波动、warmup 和重复次数；交错双方测量以减少温度、频率与竞争造成的顺序偏差。运行期间的其他 GPU 工作会使一次比值难以解释，不应修改系统配置来制造结果。

统一使用 `reference_ms / candidate_ms` 表示相对参考的 speedup，大于 1 表示候选更快；用百分比时说明分母。不同 dtype、mask、输出集合、shape、graph/eager 不混算。GEMM 名义 FLOPs 为 `2MNK`；attention 的两次矩阵乘为非 causal `4BH N²D`，causal 有效元素计数为 `2BH N(N+1)D`。这些是有用计算量，不是实际硬件指令计数。

copy/transpose 可按读写有效字节计算有效带宽；它不能直接称为 DRAM 带宽，因为 cache 命中、重复访问与事务粒度不同。用 NCU 获取 counters，另列算法 IO 模型及假设。小 shape 可能由 launch 成本主导，长序列可能受资源与工作分配影响；一张最快数字不能概括实现。

参考答案的性能验收覆盖本课全部 `perf` cases。对比优化库之外，还应在共同支持域比较相邻阶段，确认本课引入的机制在哪些输入上有收益、哪些输入上有代价。明显落后的主要案例必须定位到访存、非矩阵工作、同步、并行度或资源用量，并据此改进；不能只展示最好的输入，也不能用“教学实现”免除优化工作。

性能门槛要对应具体课题与工作负载：SIMT FP32 GEMM 和 SM100 Tensor Core GEMM 不共享一个百分比要求。记录准确的 kernel 名称、源码版本、GPU、输入与 graph/eager 结果；没有测量数据时不宣称“足够快”或完成性能优化。学生验收还包含目标设计证据，不能仅凭数值通过或达到一个速度比例判断理解。

## 最终参考答案的 90% 门槛

以下门槛用于维护者验收最终参考答案，不是学生通过关卡的要求。最终答案必须在以下入口的**全部既有 perf cases** 上逐项达到库性能的 90%。定义为同次交错采样中 `library_graph_median / solution_graph_median >= 0.9`，等价于答案耗时不超过库的 `1/0.9` 倍。边界只处理浮点除法舍入，不设置性能误差带。每个 case 独立判定，不能平均后宣称通过。

| 算子 | 最终验收入口 | perf cases |
|---|---|---|
| transpose | s30_tma | bandwidth、wide |
| softmax | s01_softmax | many_short、medium、wide |
| GEMM | s32_specialize | square、qk_shape、pv_shape |
| attention | s43_schedule | few_heads、many_heads、long_sequence、causal_perf、persistent_waves |

s30 同时承担 TMA 机制教学；把它固定为转置验收入口并不表示 TMA 总比 s00 的普通 shared-memory 转置更快。对照仍须包含 s00，不能在每个 case 上临时择优切换验收入口。其他中间阶段与非 perf 输入标记 `NOT_APPLICABLE`；它们仍需 correctness、性能比较和设计解释。

`solution` 与 `bench` 的 JSON 增加 `solution_performance`，终端显示本 case 的 `PASS` / `FAIL`；`bench` 判断其中独立参考答案的速度，不把学生速度当作答案速度。命令成功只表示测量完成，原退出码语义保持不变。单个 PASS 不代表最终答案全套通过。

把 13 个最终 perf case 各自的 `--json` 完整报告保存到独立的忽略目录（仅放这 13 份报告），随后执行离线验收：

```bash
./lab python -m flash_attention_lab.performance artifacts/final-perf/*.json > artifacts/final-perf-summary.json
```

该命令不运行 GPU。它检查 case 元数据、正确性状态、同一源码指纹及当前 checkout 指纹、答案使用当前默认配置、同为 CUDA Graph 的计时和相同捕获批量；逐项列出结果。全部通过返回 0，任一性能失败返回 3；没有已知性能失败但缺项、重复、源码不匹配或证据无效时返回 6（`INCOMPLETE`）。不要在报告目录混入旧版本、消融配置或汇总文件。完整 correctness、sanitizer、设计审查及稳定复测仍须分别完成。
