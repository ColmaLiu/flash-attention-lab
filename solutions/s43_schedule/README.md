# CUDA/PTX：按任务规模选择流水线与静态调度

本关使用独立的原生 CUDA C++/PTX attention 实现，支持 FP16/BF16、任意正 N、D64/128、causal/noncausal。输入与 O 为 contiguous `[B,N,H,D]`，自然对数 LSE 为 `[B,H,N]` FP32；只做 self-attention forward，无 dropout/bias/GQA。每次执行只发射一条本地 attention kernel。

从 [kernel.cu](kernel.cu) 阅读路径选择，再读 [attention_common.cuh](attention_common.cuh) 的 host tensor maps，以及 [attention_dense.cuh](attention_dense.cuh) 和 [attention_single.cuh](attention_single.cuh) 的设备代码；两条路径共用 [attention_exp.cuh](attention_exp.cuh) 的原生 packed 四阶指数。编译、输出分配及 tensor-map 编码在 prepare 阶段完成。

## 两条路径怎样选择

令 `paired_tasks = B*H*ceil(N/256)`，`SM` 为当前设备 SM 数量。causal 且 `paired_tasks < SM`，或 noncausal 且 `2*paired_tasks < SM` 时选择 single；其余选择 paired。它在较细 query 并行度与同一 KV 的双 query 复用之间取舍，条件来自输入规模与设备资源，不是 case ID 白名单。

| 属性 | paired：attention_dense.cuh | single：attention_single.cuh |
|---|---|---|
| 一个任务的 query | 相邻两个 128 行 tile | 一个 128 行 tile |
| CTA 分工 | 384 线程：两个 softmax WG、一个控制 WG | 256 线程：一个 softmax WG、一个控制 WG |
| TMA / MMA | thread 256 搬运；warp 9 选出一线程发 MMA | thread 128 搬运；warp 5 选出一线程发 MMA |
| KV 缓冲 | 两组完整 K/V pair，两个 query 共用 | 三组完整 K/V pair |
| TMEM 列 | S0/S1 从 0/128 开始，P0/P1 从 64/192 开始，O0/O1 从 256/384 开始 | S0/S1 从 0/128 开始，P0/P1 从 64/192 开始，O 从 256 开始 |
| 动态 shared 数据区，D64 / D128 | 96 / 192 KiB | 112 / 224 KiB |
| 寄存器分配 | softmax 每线程 224、控制 WG 56；任务边界恢复 168 | 静态分配，不调用 setmaxnreg |

这里 WG 是 128 个线程。两条路径都申请 512 列 TMEM，另有 shared barriers 和 TMEM 指针；表中 shared 数量只包含 Q/K/V 数据。grid 上限不等于实际驻留数，shared、寄存器及 TMEM 共同限制 occupancy。

## 布局、数学与输出

TMA 直接描述 BNHD，Q/K/V 的 shared tile 按 `[D/64,128,64]` 分 panel，每行 64 个半精度元素，占 128 bytes。`offset()` 的 XOR swizzle 对应 128-byte TMA swizzle。每个 softmax 线程拥有一行 query；其 warp 用 `tcgen05.ld/st.32x32b` 访问该行的 score、P 和 O，TMEM 地址的行部分由 WG 内 warp 编号决定。

QK 的 128 个 FP32 score 先全部读入寄存器，再把 FP16/BF16 P 两个元素打包到一个 32-bit TMEM cell，写进同一 score 槽的后 64 列。PV 以 TMEM P 为 A、shared V 为 B，直接累加到 FP32 TMEM O。P 写入覆盖已经读完的 score；下一次 QK 覆盖此槽前必须等 PV 结束。行 maximum 和 denominator 留在寄存器，跨 key tile 的 O 分子留在 TMEM。

默认 `polynomial_exp=True, delayed_rescale=True`。公共 API 要求 scale 可表示为正且有限的 FP32。令 `scale2=scale*log2(e)`；当 `isnormal(scale) && isnormal(scale2) && scale2<=1` 时，先归约原始 QK 的最大值，再缩放单个最大值；指数输入先用独立 packed 乘法舍入，再减去缩放后的参考值。score 和最大值必须采用相同的乘法舍入，不能合并成 FMA：`fma(score, scale2, -rounded_max)` 会留下乘法舍入残差，较大的有限 QK 可使本应为零的指数变成很大的正数或负数，进而让低精度 P 溢出或全零。求和与 O correction 也使用 packed FP32 运算。`scale2<=1` 避免有限的原始 QK 最大值在 log2 缩放时溢出。

其他合法 scale 走较慢的数值分支：行参考值保留为原始 QK，在 FP64 中计算 `(score-reference)*scale*log2(e)`、correction 与 delayed-rescale 判据，再把指数输入转为 FP32；LSE 的参考值乘 scale 也在 FP64 中完成。这样避免先形成 FP32 scale2 的溢出或次正规数被 flush-to-zero。QK Tensor Core 累加本身仍为 FP32，最终 LSE 也存为 FP32，二者必须处于可表示范围；这个分支不提供任意有限 Q/K/V 的全域精度保证，也不能补救已经溢出的 QK。

正常 scale 路径中，`polynomial_exp=True` 让每行 128 个 score 中的 48 个指数使用四阶 packed 多项式、其余 80 个使用 `ex2.approx.ftz.f32`：前 96 个元素每四个取第一对，最后 32 个全部使用 SFU。helper 用 round-down 加偏置求整数部分，在 `[0,1)` 上做四条 packed FMA，再重建指数位；最后显式 FTZ，使次正规输出归零，masked `-inf` 也严格得到正零。系数及范围归约改编自固定版本 FA4 `utils.py` 的 `POLY_EX2[4]`、`ex2_emulation_2`/`e2e_asm2`，完整 BSD-3-Clause 许可保留在 helper 中。

关闭 `polynomial_exp` 后所有 score 指数走 SFU；极端 scale 的 FP64 差值分支与每块 correction 始终使用 SFU。混合四阶与 delayed rescale 独立：后者在新最大值相对旧参考值增加不足 4 个 log2 单位时保留旧尺度，所以有效指数输入至多为 4，概率可大于 1；关闭后每块更新最大值，有效指数输入至多为 0。只有一个 warp 中至少一行 correction 不为 1 时，才集体读写该 warp 的 O 并修正；其余行乘 1。

`scaled_max_fp16/bf16` 及其 `_multi` 案例使用有限且精确的较大 QK、普通 scale 和 `V=1`，其解析 O 为 1，LSE 有限；少任务与多任务、双 key tile 分别覆盖两类调度输入。这个边界与 scale2 自身溢出、次正规 scale、QK 累加溢出不同。独立乘法与减法的性能成本必须保留在默认参考答案计时中。

混合指数减少 SFU 指令，也增加 packed FMA、整数范围归约和重建指令，是否加速必须逐例测量。多项式的局部近似误差会传播到分母、P 舍入及最终 O/LSE；CPU 区间扫描只能提供采样证据，不能替代完整 CUDA oracle 与近零 LSE、尾部/causal mask、参考值上升等边界验证。保持原有容差；本关四阶实现与 [s42](../s42_numerics/README.md) 的六阶教学消融是不同实现。

最终只计算一次分母倒数，读取 O、归一化并打包成 `int4`，通过已结束 Q 使用的 swizzled shared 区重排，随后合并写出 O；LSE 转回自然对数。尾部由 TMA 零填充、score 的 key/causal mask 和 O/LSE 的 query store guard 共同处理，集体操作的线程不能提前 return。noncausal 且完整 128-key tile 的模板省去逐元素 mask；causal 内部完整块也避开对角线的逐元素判断。

`descriptor()` 编码的 shared 起始地址和 leading offset 以 16 bytes 为单位。QK helper 先构造 descriptor 基址，再加常量偏移：16 个半精度元素对应 `+2`，跨到下一个 64-column panel 对应 `+1024`；PV 的 V 在 key 增加 16 时对应 `+128`。这是编码后的地址增量，不能直接加元素数。MMA 控制 warp 先执行一次 `elect.sync` 选出发射线程，保留统一的 descriptor 计算，避免每条 MMA 重建地址和选举控制流；仍需通过生成的 SASS 核对编译结果。

## 两种流水线的完成条件

paired 路径先发 QK0、QK1。对于 key tile t，MMA 线程依次等待各 WG 的 Pready，发 PV，提交 PV 完成与 KV-empty 事件，显式等待 PV 完成，再给该 WG 发下一块 QK。每个 softmax WG 独立等待自己的 score，更新 P/O，经过 `wait::st`、before-thread-sync fence、128-thread named barrier 和 after-thread-sync fence，才由一线程发布 Pready。下一块 score 的完成包含该 WG 前一块 PV 已完成这一事实；最后一块没有后续 score，consumer 额外等待 PV 后进入 epilogue。

single 路径先发 QK(0) 与存在时的 QK(1)，分别写两个 S 槽。MMA 对 tile t 等 Pready(t)，发 PV(t)，显式等完成、释放 KV(t)，再向同一 S 槽发 QK(t+2)。softmax 发布 Pready(t) 后可立即读另一槽的 score(t+1)，计算 max、exp、sum 和 P；**在修正 O(t+1) 前必须另等 PV(t)**。score(t+1) 是提前发出的，不能证明 PV(t) 已完成。行统计仍由同一 WG 顺序更新。

| barrier | paired 的 tile t | single 的 tile t |
|---|---|---|
| score / Pready / PV | 每个 query WG 独立对象，phase=`t&1` | 槽=`t%2`，phase=`(t/2)&1` |
| KV-full | 槽=`t%2`，phase=`(t/2)&1` | 槽=`t%3`，phase=`(t/3)&1` |
| KV-empty | 每代 count=2，两次 PV 各提交一次 | 每代 count=1，PV 完成后提交一次 |

每个任务的 Q 只加载一次，Q-barrier 等待 phase 0。producer 复用 KV 槽前等待上一代 empty phase，即 `(generation-1)&1`，首次使用不等待。循环内只使用各 WG 的 named barrier；公共 CTA barrier 位于初始化与任务边界。手工角色同步使用不带 `.aligned` 的 `barrier.sync`：WG barrier 的 count 为 128，paired 寄存器交接的公共 barrier count 为 384；每个参与 warp 的所有线程都必须到达对应实例。它不作 legacy `bar.sync` 所隐含的全 CTA 同指令参与承诺，参见 [PTX 8.7 §9.7.13.1](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)。`mbarrier.try_wait.parity` 是等待异步事件完成的循环，不是动态领取 query 任务的队列。

[PTX ISA 8.7 §9.7.16.5.2](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html) 的同线程 MMA 顺序保证要求匹配 accumulator、shape 等条件。PV 写 O、QK 写 S，并且后者可能覆盖前者仍在读取的 P，因此两条路径保留显式 PV completion wait。`commit` 跟踪同一发射线程此前的异步 tcgen 操作；跨线程 TMEM 交接还要结合完成等待与 before/after-thread-sync fence，参见同手册 §9.7.16.11.1、§9.7.16.5.4.4。

## Persistent 生命周期与控制实验

paired 的 `qtiles=ceil(N/256)`，single 的 `qtiles=ceil(N/128)`；`total_tasks=B*H*qtiles`。默认 grid 为 `min(total_tasks,2*SM)`，CTA 处理 `blockIdx.x+k*gridDim.x`。causal 时令 `batch_heads=B*H`，映射为 `query_tile=qtiles-1-task/batch_heads`、`bh=task%batch_heads`：较后的 query 块跨所有 head 优先安排。该静态策略不保证动态负载平衡。

TMEM allocation 由第一个 warp 在 CTA 开始时执行一次，跨任务保留，CTA 最后释放。每个任务重新初始化行统计与 barriers，首个 PV 初始化 O。paired 路径由唯一 TMA 线程 256 初始化全部 mbarrier，执行 async proxy fence，再经全 CTA barrier 发布给其他角色；初始化与 TMA expect-bytes 同属一个线程。single 路径仍由 thread 0 初始化并经全 CTA barrier 发布。

任务结束时，所有角色完成异步工作和 epilogue；每个访问 Q 的 consumer 执行 async proxy fence 并汇合，才允许下个任务 TMA 覆盖 Q。thread 0 按各 KV 槽最后一次实际使用的 generation **排空 KV-empty**，再 invalidate barriers，最后经全 CTA barrier 允许下一任务重新初始化。初始化与 invalidate 不必由同一线程执行，但整个生命周期必须有这些完成等待与 CTA 交接；观察 PV 完成不能替代等待稍后提交的 empty arrival。

paired 的动态寄存器分配与上述 TMEM 分配是两件事。控制 WG 先释放到 56，两个 softmax WG 增至 224；结束时 softmax 降到 168 并进入 CTA barrier，控制 WG 汇合并通过同一 barrier 后才恢复 168，避免它提前申请寄存器而持续重试。single 的 256 线程使用静态寄存器分配，不能沿用 paired 的入口寄存器数量假设。

`persistent_waves` 的 512 个 128-query tile 对应 **256 个 paired tasks**。它检查长 causal 循环及任务排序，但是否跨任务复用，必须比较所选路径的 `total_tasks` 与 `2*SM`；不能只看 query tile 数。`persistent_reuse` 有 512 个短任务，N128 只访问一个 KV 槽，适合检查跨任务初始化、TMEM 和 Q 复用。两槽跨任务控制实验可另外构造 causal `[1,256,H,64]`、`H>2*SM`（例如满足该条件时取 H=384），让 paired 的两个 KV 槽都被使用并跨任务复用；该手工控制例不属于 curriculum 的固定案例。

在相同 inputs/cfg 上可调用：

```python
from flash_attention_lab.api import prepare
plan = prepare("s43_schedule", inputs, config={**cfg, "persistent": False}, solution=True)
plan.launch()
```

此开关同时关闭 grid 限额与 causal 长任务优先映射；shape 驱动的 single/paired 选择保持相同。数值开关也可独立传入 `polynomial_exp=False`（全 SFU）或 `delayed_rescale=False`（每块更新尺度）。以上都是控制实验，不可用非默认配置替代默认答案验收。

## 验证与性能完成标准

```bash
./lab check s43_schedule --solution
./lab solution s43_schedule --case 9
./lab solution s43_schedule --case 10
./lab solution s43_schedule --case 11
./lab solution s43_schedule --case 13
./lab solution s43_schedule --case 15
./lab sanitize s43_schedule --solution --tool memcheck --case 12
./lab sanitize s43_schedule --solution --tool memcheck --case 16
./lab sanitize s43_schedule --solution --tool racecheck --case 16
./lab sanitize s43_schedule --solution --tool synccheck --case 16
```

先通过全部 correctness、stream、重复调用、所有数值开关组合与相关 sanitizer 检查，再测默认配置。全部五个 perf case 各自要求 `library_graph_median / solution_graph_median >= 0.90`，不以平均值抵消失败；两边须采用相同 CUDA Graph 捕获批量、BNHD O 与 BHN LSE 输出契约及计时边界。编译、输入生成、输出分配和 TMA 编码均排除在计时之外，实际库 backend 必须记录为 cuDNN。公共 noncausal BF16 输入还需与 s42 对照；多项架构同时变化时不能把全部收益归给 persistent 排序。

所有逐例测量、代码指纹、资源报告及剩余差距存入 Git 忽略目录。静态 grid、额外 shared 缓冲、控制 warp 与等待循环都有成本；改动后重新确认每个 perf case，不能把单个 shape 的结果推广到全部输入域。

设计阅读可对照固定版本 [FA4 forward](../../third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py) 的 query stages、softmax/MMA 分工及 [Blackwell helper](../../third_party/flash-attention/flash_attn/cute/blackwell_helpers.py) 的 descriptor/election 处理。它们用于源码与设计对照，本地可执行实现和同步协议以上述 CUDA/PTX 文件为准。版本、许可与论文入口见 [sources](../../docs/sources.md)。
