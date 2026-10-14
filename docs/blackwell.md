# B200 / CuTe DSL 4.4.1 接口边界

课程目标设备为 SM100（例如 B200）、compute capability 10.0；每次运行先检查所选设备。CUDA C++ 选择 `compute_100a/sm_100a`；需要架构专用特性的实验不可当作通用 sm_100 或消费级 sm_120 代码。CuTe 4.4.1 的 wheel 自带代码生成组件，与 nvcc C++ 扩展是两条不同编译链。

CUDA C++ 采用 Toolkit 12.8 与 PyTorch cu128，DSL 使用固定 4.4.1 的 CUDA 12 系列组件。驱动必须支持两条编译链的产物。安装要求见 [环境说明](environment.md)，验收方法见 [验证流程](validation.md)；当前 API 签名通过 `./lab doctor --stage 3` 检查，不能仅凭 import 认为异步 kernel 正确。

| 练习 | 需要核查的接口族 | 独立验收 |
|---|---|---|
| s30/s31 | `cute.make_layout`、Tensor、`from_dlpack`、`cute.compile` | 转置/softmax、坐标映射、nondefault stream |
| s32 | cp.async copy atom、完成组/等待、SMEM 可见性 | copy、racecheck、synccheck |
| s33 | `cpasync.CopyBulkTensorTileG2SOp`、`make_tiled_tma_atom`、mbarrier | descriptor 合法性、事务字节、phase |
| s34/s35 | `cutlass.pipeline` 的 producer/consumer 状态与 stage | 多次 wrap、短循环、drain |
| s36 | `utils.TmemAllocator`、`tcgen05.Ld32x32bOp/St32x32bOp` | roundtrip、分配释放与 collective 参与 |
| s37/s38 | `tcgen05.MmaF16BF16Op`、`CtaGroup`、`PipelineTmaUmma/PipelineUmmaAsync` | FP32 GEMM、跨引擎完成/复用协议 |

`./lab doctor --stage 3` 会读取当前安装包的签名；这是可用性检查，不会给出你的 kernel 的 layout 或 barrier 方案。源 API 以固定的 CUTLASS v4.4.1 为准；在线 latest 文档会漂移。某符号变化时先核对安装版本，不默认把 Hopper 示例搬来。

在固定 CUTLASS 4.4.1 API 中，`MmaF16BF16Op` 的 1-CTA mode 要求 M 为 64 或 128，N 在 8 到 256 之间且是 8 的倍数；TMEM allocator 要求分配列数为 32 的倍数且为 2 的幂，总量不超过架构上限。对应版本来源见 [sources](sources.md)。这些是合法域，不是预先选好的 tile/layout；构造合法 host 对象也不能证明 MMA 已在设备正确执行。

必须自行核查的硬件约束：TMA tensor-map 全局 base 对齐、stride 编码/粒度、box 尺寸、SMEM base 和 swizzle 一致性；mbarrier 存储对齐、参与计数、expected transaction bytes、phase；TMEM 分配粒度、容量、释放者与 collective ld/st 的参与线程；tcgen05 指令合法 M/N/K、输入 major、descriptor 对齐、累加类型、完成通知；寄存器/SMEM/TMEM 占用对驻留 CTA 的限制。将你最终采用的合法值和对应 NVIDIA 手册章节写入本关记录。

TMEM 是显式分配的 lane/column 组织，不是普通 global/shared 指针。Blackwell tcgen05 MMA 异步写 TMEM；Hopper wgmma 的寄存器输出与等待协议不能直接当作其替代。TMA 完成不表示 MMA 完成，MMA 发起不表示 SMEM 可以重写。所有阶段必须把跨引擎依赖写成数据可见性和缓冲生命周期，而不仅是 barrier 名称。

当前只使用 **1-CTA MMA mode**。论文 2-CTA MMA 和大幅减少 atomic 的收益主要涉及 backward，课程不实现；不把它误设为 forward 通关前提。FP8/FP4、cluster multicast、persistent scheduler 的全面工程化也不在必需范围。s46 可研究简化调度，无需实现完整官方调度器。

底层 smoke 使用一个标量 sentinel 的编译、调用、stream 和计时；不能证明尚未编写的 TMA、TMEM 或 pipeline 内核已经通过。实际机制验证由对应学生小练习承担。
