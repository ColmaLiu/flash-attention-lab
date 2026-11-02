# CUDA/PTX：显式 fragment 的完整 Tensor Core GEMM

入口 [kernel.cu](kernel.cu) 直接使用 `ldmatrix` 和 `mma.sync.aligned.m16n8k16.row.col`，输入 FP16/BF16，累加与输出 FP32。CTA 输出 tile 为 128×64，K 步长 32，四个 warp 排成 2×2；每个 warp 持有 64×32 输出，即 4×4 个 16×8 MMA fragment。一个 fragment 内 lane 的行是 `lane/4 + {0,8}`，列是 `2*(lane%4) + {0,1}`。

## 数据与同步

线程以 16 字节向量同步执行 global→register→shared 搬运，A/B 使用相同读写端可解释的 XOR swizzle。第一个 CTA barrier 发布完整输入 tile；`ldmatrix.x4` 装入 A，`ldmatrix.x2.trans` 装入 B，再执行两轮 K=16 的 MMA。第二个 barrier 保证所有消费者结束，下一轮才能覆盖 shared memory。warp 的 FP32 accumulator 一直保存在寄存器，最后以 float2 写回，warp 之间不合并部分和。

M/N/K 必须为正的 16 倍数，CTA 尾部可以不满。无效向量填零，最终写回按有效行列判断；16 倍数契约保证向量不会跨过合法尾部。坐标、长 K 和非方形案例分别检查 fragment 映射、重复累加和尾部。

## 性能实验

同 BF16/FP16 输入、FP32 输出的独立 cuBLAS 是性能对照。s10 仅接受 FP32；跨阶段可将相同低精度量化值升格到 FP32 输入 s10，但搬运字节数和算术路径随 dtype 改变，须单独标明，不能将差异全部归于 Tensor Core。CTA 复用减少重复输入读取，但同步搬运仍与计算串行；寄存器占用、短 K 的输出流量和小矩阵并行度也会限制速度。用全部 perf case 测量这些因素，不能从指令吞吐直接推断 GEMM 加速。

```bash
./lab check s20_mma --solution
./lab solution s20_mma --case 5
./lab sanitize s20_mma --solution --tool memcheck --case 4
```

算法对应标准分块 GEMM；指令与 lane 映射依据 [NVIDIA PTX MMA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)。s21 的双缓冲实现进一步讨论搬运与计算的交错，阶段间比较还需核对 tile、fragment 与 shared layout 是否相同。
