# 20_mma_gemm 答案说明

## 设计选择

- **支持域**：A、B 为 BF16，C 为 FP32，`K % 8 == 0`。K 是 8 的倍数时每行 16 字节对齐，一个 16 字节 chunk 要么整体在行内、要么整体在行外，`load_tile` 用整 chunk 补零即可。step 1 额外要求 `M % 16`、`N % 8`、`K % 16` 为 0：这一步只练 fragment 坐标，不处理尾部，`make_plan` 明确拒绝其他形状，spec 中 step 1 只运行对齐的 case。
- **step 1**：每个 warp 一个 16×8 块，每个寄存器从 global 读一个 `uint32_t`（两个相邻 BF16）。fragment 坐标函数直接对应 PTX 文档中的表格，`one_atom`（整数输入、一条 mma）能精确检查每一个坐标。
- **step 2/3**：128×128×32 的 CTA tile，8 个 warp 排成 2×4，warp tile 64×32（4×4 个 m16n8）。B tile 在 shared memory 中保持 `[n][k]`，这正是 `.col` 操作数的布局，所以 B 也用非转置的 `ldmatrix.x4`，一条覆盖两个 n8 块。`load_tile` 先把所有 global load 发出再统一写 shared，让每个线程有 2 个 16 字节 load 在路上。
- **swizzle**：64 字节的行（4 个 chunk）用 `chunk ^ ((row >> 1) & 3)`。8 个连续行的同一逻辑 chunk 落在 8 个不同的 16 字节槽，`ldmatrix` 每个 8×8 矩阵一个 wavefront；16 字节 store 同样无冲突。读写都经过同一个 `tile_offset`，正确性与是否 swizzle 无关。
- 写回直接从寄存器逐元素写 global（每元素判断边界）。这不是瓶颈（C 只写一次），向量化留给后续关卡。

## 各 step 的性能变化（相对 cuBLAS BF16→FP32，CUDA Graph）

| step | 4096³ | 8192×8192×4096 | 原因 |
|---|---|---|---|
| 1 | ≈ 0.02× | ≈ 0.01× | 每条 mma 6 次 4 字节 global 读，无复用 |
| 2 | ≈ 0.09× | ≈ 0.09× | shared memory 复用；但 `ldmatrix` 有 4 路 bank conflict |
| 3 | ≈ 0.17× | ≈ 0.17× | 消除冲突：shared load wavefront 正好降为 1/4（NCU：约 2.0 亿 → 0.50 亿，conflict 约 1.5 亿 → 1 万） |

参照：`mma.sync` 微基准上限约 560 TFLOP/s，约为 cuBLAS 的 0.36–0.39 倍；step 3 约为该上限的 45%。剩余差距主要是同步的 global→shared 搬运（下一关的内容）。`perf_target` 设为 0.12，答案在两个 perf case 上约 0.165。

## 验证

- `./lab check 20 --solution`：全部 step、全部 case 通过（整数 case 误差为 0）。
- compute-sanitizer memcheck / racecheck / synccheck：step 1 用 `atoms`，step 2、3 用 `tail`，均无报错。
- `./lab sass 20 --solution`：无 spill；`mma_cta` 约 125 个寄存器；SASS 中有 `HMMA`、`LDSM`。
- 骨架：逐步把 TODO 换成答案代码，第 n 步只依赖 TODO(1..n)，完成后与答案性能一致；骨架已恢复。

## 已知局限

- 只支持 BF16 输入和 FP32 输出（头文件中有 FP16 的 mma 封装，本关不使用）。
- 随机输入、大 K 时，Tensor Core 累加的误差约按 K^1.5 增长（cuBLAS 也一样），gemm 的容差为此对 16 位输入加了 `2e-8·K^1.5` 一项（见 `labkit/kinds/gemm.py`）。
