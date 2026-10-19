# 性能口径

`./lab bench ID --case 0` 默认只测一个小 case，先验证数学正确性、重复调用、输入不变和 nondefault stream。首次 prepare 包含 C++ build/DSL JIT；计时前完成。输出和 scratch 预分配，输入生成、布局转换、编译均在计时外。学生 launcher 不得内部进行这些操作。

默认 warmup=5、repeats=7、每组 iterations=10，CUDA Event 测当前 stream 的时间跨度，每组末等待完成，报告 per-launch median/min/max/stdev 和全部样本。Python dispatch 可能使 GPU stream 出现空隙，尤其对微小 kernel；因此命名为 `gpu_stream_ms`，不能声称它必然等于单 kernel 的设备执行时间。host_batch 是同一预分配调用组的 host 时间，含 event 操作和最后等待，**不是分配/转换全包含的端到端延迟**。若要端到端，另对 `api.run` 记录 wall-time 并说明是否包括输入创建与首次编译。

计时之后再次比较结果，防止重复执行后状态污染。英文终端输出显示输入、GPU、计时中位数/波动和采样数，不显示精度明细；`--verbose` 展开样本与配置，`--json` 输出包含精度结果的完整报告。保存的 JSON 包含 GPU/driver/Toolkit/PyTorch runtime/DSL、代码指纹、shape/dtype/mask/scale/seed、误差、scratch 字节、配置及完整样本。自定义 tile/stage 等参数应放入本关 entry.CONFIG（或程序调用的 config）并记录；修改 entry 中常量时也要写实验表。

attention FLOPs 只统计 QK 与 PV，FMA=2：非因果 `4 B H N² D`；因果有效工作 `4 B H [N(N+1)/2] D`，另给 dense-equivalent `4 B H N² D`。不包括指数、reduction、重缩放、padding 或 mask tile 中无效 MMA。不要把有效 TFLOP/s 当作实际执行指令 FLOPs。

GEMM FLOPs 为 `2*M*N*K`，FMA=2。`s22_mma` 与 `s38_async_gemm` 包含大方阵及 QK/PV 比例的长方阵；attention 包含多头、长序列及阶段允许的 causal/tail 组合。用 `./lab task ID -v` 查看全部 case 编号，按 [测试规模](test_scales.md) 做控制变量比较。`case 0` 用于快速定位与调用开销观察，不能代表大工作量下的吞吐；默认 `check ID` 验收全部尺寸，不需要额外选择测试组。

保持独立入口做控制变量：s10→s20 比 grid/循环与输出驻留；s20→s21 比 CTA 内分工/非矩阵计算；s21→s23 涉及 dtype 和 Tensor Core，必须另列；s40→s41 比流水线；s41→s43 比指数；s43→s44 比重缩放；s45→s46 比调度。只有支持域的交集可直接对比；不同 shape 的结果不能互相算 speedup。所有教学版本都运行在 B200，不代表历史官方 FA1/FA2 的性能。

可选官方对照不作为基础依赖：[sources](sources.md) 中固定版本的 FA4 需要 DSL >=4.6.2，与本项目固定的 4.4.1 不同。另建环境按 `docs/sources.md` 的固定 commit 安装官方包，在该环境激活后从仓库根目录显式运行（入口同样遵循 LAB_GPU/CUDA_VISIBLE_DEVICES）：

```bash
python -m benchmarks.official_fa4 \
  --exercise s40_fa4_base --case 0 --source-commit 1fc0221183e799363001394a12dd9e78e15f5dda
```

不要把官方包装进 student path。对照明确将 BHND 转为官方 BNHD，分别记录预转换后的官方 API span，以及包含输入输出布局转换的 API span。官方 API 内部可能分配输出，因此前者也不是严格的预分配 kernel 时间。该入口只验证官方返回的 O，不宣称验证了未返回的 LSE；未安装时明确非零退出。课程不要求安装官方对照。
