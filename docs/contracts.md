# 输入与执行契约

Attention 的 Q/K/V/O 都是同一连续 `[B,N,H,D]`，与官方 `flash_attn_func` 保持一致。LSE 是 `[B,H,N]` FP32，为 `log(sum(exp(scale*QK)))` 的自然对数。仅支持等长 self-attention，相同 Q/K/V head 数；causal 包含对角线。默认 scale 是 `D**-0.5`，显式 scale 必须是实数标量，转换为 FP32 后仍有限且大于零。

其他算子：transpose 输入 `[R,C]` 输出 `[C,R]`；softmax 沿 `[R,C]` 的 C 轴；GEMM 元数据 shape 写作 `(M,N,K)`，输入 A `[M,K]`、B `[K,N]`，输出 C `[M,N]` FP32。名字与 shape 不由 tile 大小决定。

`curriculum.json` 中每课的 `domain` 定义保证支持域：`dtypes`、允许的 `causal` 值、`shape_multiple` 的逐轴整除约束，以及 attention 的 `head_dims`。`pointer_alignment` 若存在则要求每个输入基地址按该字节数对齐；连续 view 的非零 storage offset 不保证满足 TMA 或向量加载的对齐条件。所有维度必须是正整数。case 列表只是代表性采样，不再是 shape 白名单。输入要求有限、同 GPU、contiguous、无梯度；本课程不定义 NaN/Inf 输入语义。

C++ 输入键是 `q/k/v`、`a/b` 或 `x`；输出键是 `o/lse`、`c` 或 `y`。`TensorView` 提供 data/shape/dtype，`Options` 提供 scale/causal 及明确声明的配置。Python `workspace_spec` 声明 scratch 的 shape/dtype，由 host 预分配。学生 `student_launch` 必须在传入 stream 启动；不能默认 stream 0。

输出与 workspace 初始未定义，每次 launch 都应独立完成全部输出，不能依赖上次残留。输入不可修改。构建、描述符准备和输出分配放在 prepare 阶段；需要反复执行的设备工作属于 launch。基础设施不会为学生预先选 grid/block/shared memory 或计算正确答案。

学生、完整答案、快速库、数学 oracle 是独立入口。空学生必须失败；即使其他三者全部通过，也不能将学生记为完成。FP64 oracle 从已量化的同一输入计算，避免把输入量化误差混入 kernel 误差。

需要计时外的描述符准备时，可在学生 `entry.py` 中声明 `BACKEND = "python"`，提供 `prepare(inputs, outputs, workspace, config, stream)` 并返回无参数 launcher。该 launcher 仍必须调用自己的设备实现，不能转交数学参考或库基线。`build_cuda(ID)` 可编译学生目录内的 `kernel.cu` 与可选 `binding.cpp`；完整答案使用独立的 `solution=True`。`IMPLEMENTED` 检查仍在分配和准备之前。
