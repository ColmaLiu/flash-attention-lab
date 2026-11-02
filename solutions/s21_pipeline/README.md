# CUDA/PTX：与同步 GEMM 同布局的双缓冲

[kernel.cu](kernel.cu) 保留 s20 的 128×64×32 CTA tile、四个 warp 的 64×32 输出 ownership、XOR shared-memory swizzle、`ldmatrix`/`mma.sync` 和 FP32 epilogue。每个 operand 增加第二个 shared 缓冲，将同步 global→register→shared 搬运换为 16 字节 `cp.async`，在当前 tile 的矩阵运算之前预取下一 tile。输入仍为 FP16/BF16，输出 FP32，M/N/K 为正的 16 倍数。

## 两个不同的完成义务

prologue 装入 slot 0，执行 `wait_group 0` 后 CTA barrier 发布第一块数据。稳态循环在另一个空槽发起下一块 copy，再消费当前槽的 operands。循环末尾每个线程等待自己提交的 copy group，随后 CTA barrier 同时保证下一块输入可见、当前块所有读者结束；下一轮才能覆盖腾出的槽。

`wait_group` 不能替代 CTA barrier，因为它不等待其他线程的 copy。CTA barrier 也不能单独替代 `wait_group`，因为异步搬运不随发起线程到达 barrier 自动完成。最后一轮没有下一块 copy，仍执行相同完成协议。尾部向量用零字节源填零；无效源指针保持在输入分配内。

## 实验与验收

```bash
./lab check s21_pipeline --solution
./lab solution s21_pipeline --case 5
./lab solution s21_pipeline --case 6
./lab solution s21_pipeline --case 7
./lab sanitize s21_pipeline --solution --tool racecheck --case 4
./lab sanitize s21_pipeline --solution --tool synccheck --case 3
```

s20 为相同 tile、布局、fragment 和输出写回的同步对照；cuBLAS 为独立优化库对照。比较全部 perf case 的 graph/eager 延迟，并检查指令和时间线中的重叠。双缓冲增加 shared 存储，短 K 无法充分摊销 prologue；异步 copy 还改变了寄存器搬运工作，不能把全部变化简单归为一个等待指令的节省。

指令依据 [NVIDIA PTX cp.async 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)；fragment 解释见 [s20 README](../s20_mma/README.md)。
