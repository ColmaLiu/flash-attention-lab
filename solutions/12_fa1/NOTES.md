# 12_fa1 参考答案说明

## 设计选择

**忠实于论文 Algorithm 1。** 外层循环遍历 K/V 块，内层遍历 Q 块；`O`、`l`、`m` 存在 HBM，
每个 `(i, j)` 读一次、写一次；`O` 在 HBM 中始终是已归一化的（第 12 行先乘 `l_i` 再除
`l_new`）。这些都是有意保留的低效之处：`22_fa2` 交换循环、把 `O` 留在寄存器、延迟归一化，
学习者可以用同一组 perf case 对比。

**一个 (batch, head) 一个 CTA**，即原始 FA1 kernel 的并行方式（FA2 论文 3.2 节）。
CTA 独占一个 head 的全部行，因此 HBM 中的 `O/l/m` 只在 CTA 内共享，`__syncthreads()`
足以保证可见性，不需要跨 CTA 同步，也不需要原子操作。`O` 直接用输出张量；`l`、`m` 用
`lab::Scratch`（`[B, H, N]` FP32 各一个），在 kernel 开头按第 2 行初始化。

**块大小 `Br = Bc = 64`，256 线程，每线程 4×4 的 S tile。** 不用论文的 `Bc = ⌈M/4d⌉`
（d = 64 时为 227）：S 放在寄存器里，`64×227` 会使每线程分数数量过多并溢出；整块占满
227 KB 时每 SM 只有 1 个 CTA；而且本 kernel 不受 HBM 带宽限制（见下）。`Br = Bc` 还让
causal 下的块只有三类（完全可见、对角、完全 mask），并保证真实 query 行永远不会在一个
处理到的块里被全部 mask。

**shared memory 布局。** `Kt[d][Bc+4]`（K 转置，float4 读 4 个 key）、`Vs[Bc][d]`、
`Qs[Br][d+4]`、`Ps[Br][Bc+4]`。`+4` 保持 16 字节对齐，并让同一 warp 中相隔 4 行的两组
float4 读落在不同 bank。K_j 的转置写入有 bank conflict，但每个 j 只做一次，被全部 Q 块摊薄。
d = 64 / 128 时分别用 67 KB / 116 KB，需要 `cudaFuncSetAttribute` 放开 48 KB 限制。

**行归约。** 一行的 16 个线程是同一 warp 中连续的 16 个 lane，用 xor 1/2/4/8 的 shuffle
完成 max/sum，16 个线程都得到结果。

**HBM 状态的读写顺序。** `load_state`（每个线程读自己 4 行的旧 `O/m/l`）放在第 8 行的位置，
即 `Q_i` 加载之后的 barrier 之前；`merge_state` 中只有 `tx == 0` 写回 `m/l`。中间至少隔着
两个 `__syncthreads()`，所以同一行其他 15 个线程一定先读到旧值。`O` 的每个元素始终由同一个
线程读写，不存在跨线程依赖。

**数值。** 用 `expf`（FP32 容差 `atol 2e-5, rtol 1e-4` 下足够）。全 mask 行用 0 作减数，
`m_new = −∞` 时不更新该行；在 `Br = Bc` 下这两个分支不会被真实行触发，但保证换块大小后
仍然正确。

**step 1** 是同一个 kernel 的 `kSingleBlock = true` 版本：不初始化、不读写 HBM 状态，直接写
`O = P~V/l~` 和 `LSE`。`make_plan` 拒绝 step 1 的 N > 64。step 2 与 step 3 是同一个 kernel
（step 2 的 case 都不是 causal）。

## 各 step 的性能（相对 `efficient` 基线，CUDA Graph）

| case | step | 比例 | 说明 |
|---|---|---|---|
| `one_block` | 1 | ≈ 1.2× | 只有 6 个 CTA，几乎全是 launch 开销，没有意义 |
| `perf` [4,4096,64,64] | 2 / 3 | ≈ 0.49× | 约 33% FP32 标称峰值；`--cold` 相同 |
| `perf_causal` | 3（不跳块） | ≈ 0.26× | 与非 causal 耗时相同 |
| `perf_causal` | 3 | ≈ 0.50× | 跳块后 tile 数 2080/4096，耗时约减半 |
| `few_heads` [1,4096,16,64] | 3 | ≈ 0.06× | 16 个 CTA，只有 16 个 SM 工作 |

`perf_target` 设为 0.4（相对 `efficient`），两个 perf case 都有约 20% 余量。

**Nsight Compute（`perf`）**：DRAM 读约为 Q+K+V 大小的 44 倍、写约为 O 大小的 67 倍，与
`Tc · (读 Q + 读 O + 写 O)` 的估算一致；折算到 bench 时间只占 HBM 标称带宽约 15%。
`sm__throughput` 约 50%，`sm__warps_active` 约 22%（d = 64 时寄存器 128 个，每 SM 2 个 CTA）。
即 SIMT 版本是计算/延迟受限，而不是 HBM 受限。

**与基线的差距来源**：PyTorch 的 FP32 memory-efficient attention 是 CUTLASS
`fmha_cutlassF_f32_aligned_64x64_rf_sm80`。按 `third_party/cutlass/examples/41_fused_multi_head_attention/gemm_kernel_utils.h`
的 `DefaultGemmType<ArchTag, float>`，SM80+ 的 FP32 使用 `OpClassTensorOp` 与
`OpMultiplyAddFastF32`（3xTF32）；Nsight 实测有 tensor pipe 指令。它按 64 行 Q 块切分 CTA
（`getBlocksGrid`），输出累加器留在寄存器。这些都是后续关卡的内容，本关答案不提前使用。

## 验证

- `./lab check 12 --solution`：全部 step、全部 case 通过。
- `compute-sanitizer` memcheck / racecheck / synccheck（`causal_tail` step 3，`tail` step 3，
  `rescale` step 2，`short_tail` step 1）：0 错误。
- 骨架：每个 TODO 用答案代码填入后对应 step 通过，且第 n 步只依赖前 n 个 TODO。

## 已知局限

- 只支持 FP32、d ∈ {64, 128}。
- `Br`、`Bc` 固定为 64，没有按 d 或 SRAM 预算调整；d = 128 时每 SM 只能驻留 1 个 CTA。
- 并行度为 B·H，B·H 远小于 148 时 GPU 大部分空闲（这是本关要展示的现象）。
- 每个 j 都重新加载并转置 K_j，Q_i 和 O_i 的加载没有与计算流水重叠（`cp.async` 在 21 关）。
