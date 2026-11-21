# 01_softmax 参考答案说明

讲义：[course/01_softmax/README.md](../../course/01_softmax/README.md)。本文件说明答案为什么这样写、怎样验证、还有哪些局限。比例均为同次 CUDA Graph 计时的中位数之比（`torch 耗时 / 本版本耗时`），不写绝对时间。

## 设计选择

### 所有 step 共用

- **一个 `exp_shifted(x, m) = ex2.approx.ftz((x - m) * log2 e)`。** 先减后乘：`x - m` 在 `x`、`m` 相近时精确（Sterbenz），乘法只引入一次舍入。曾试过 FlashAttention 源码里常见的 `fmaf(x, log2e, -m * log2e)`：它把 `m * log2e ≈ 7200` 舍入成 FP32，误差 `δ(m)` 最多 2⁻¹²，结果是 `e^(x-m) · 2^(-δ(m))`。步骤 1、2 整行只有一个平移量，这个因子在 `e / l` 中抵消，`shifted` 仍然通过；步骤 3、4 合并不同平移量 `m_t` 的部分状态，`e^(m_t - M)` 消不掉各自的 `2^(-δ(m_t))`，实测 `shifted` 相对误差约 2.5–4e-4，超过 1e-4 的容差。attention 的 logit 一般不大（`δ` 约 1e-6）且输出是 BF16，所以 FA 用 FMA 形式没有问题；本关是 FP32 输出，四个 step 统一用先减后乘。
- **直接写 `ex2.approx.ftz.f32`**，而不是 `exp2f`：`exp2f` 为了正确处理结果为次正规数的情况会多出 `FSETP` 和两条带谓词的 `FMUL`；`ftz` 把小于 2⁻¹²⁶ 的结果冲成 0，相对 1e-8 的绝对容差可以忽略。
- **`-inf` 的处理集中在两处**：`merge()` 中新最大值为 `-inf` 时返回空状态；步骤 4 中线程局部最大值为 `-inf` 时用 0 作平移。这两处都对应 FlashAttention 源码里的 `Check_inf` / `max == -INFINITY ? 0 : max`。

### 步骤 1–3：一个 block（256 线程）一行

- 线程 `t` 处理第 `t, t+256, …` 列，warp 内连续，访存合并。
- 步骤 1 的树形归约需要 `blockDim` 为 2 的幂；8 层，每层一次 `__syncthreads()`，最后再加一次保护 `smem` 复用。
- 步骤 2 先在 warp 内 butterfly，再由**每个 warp** 归约 8 个部分和，这样所有线程都直接拿到结果，不需要第二次经 shared memory 广播。
- 步骤 3 逐元素用 `s = merge(s, {x, 1})`，每个元素两次 `ex2`。曾用分支写法（`x > m` 时只算 `e^(m-x)`，否则只算 `e^(x-m)`）把 exp 次数减半，实测在 `rows4096` 上反而慢约 1.6–2.9 倍：分支把循环切成依赖链，编译器不再提前发射后面的 load，SASS 中每次只发出 1–2 个 `LDG`，而无分支写法把循环展开后连续发出 8 个 `LDG`。这个 kernel 受限于在途访存而不是 MUFU 吞吐，所以保留无分支写法。

### 步骤 4：整行在寄存器里

- `kThreads` 个线程负责一行，每线程 `kVecs` 个 chunk，每个 chunk 是 `V` 个连续 float（`V = 4` 用 `float4`）。chunk `k` 的起始列是 `(k * kThreads + t) * V`，一条 warp 指令覆盖连续的 `32 * V` 个 float。
- 先把全部 chunk 读进寄存器再计算，`kVecs` 个 load 同时在路上。
- 线程内先求局部 `(m, l)`，把 `v` 就地变成 `e^(x - m_local)`；一次 `row_merge` 得到整行的 `(M, L)`；输出时乘 `e^(m_local - M) / L`。这样每个元素只做一次 `ex2`，归约只有一次（而不是 max 和 sum 两次），并且与 FA 中对输出累加器的延迟 rescale 是同一个手法。
- 配置选择：每线程最多 32 个 float（不溢出到 local memory），短行（≤ 256 个 chunk）一个 warp 一行、每 block 4 行，长行一个 block 一行，线程数取 128–1024。超过 32768 列时退回步骤 3 的 online kernel。
- `__ldcs` / `__stcs`（evict-first）：x 和 y 都只碰一次。
- `cols % 4 != 0` 或指针未 16 字节对齐时用 `V = 1` 的标量版本，正确但较慢。

## 验证

```bash
./lab check 01 --solution
./lab selftest 01 --perf
./lab sanitize 01 --solution --step N --case tiny|masked_short|tail --tool memcheck|racecheck|synccheck   # N = 1..4
./lab sass 01 --solution          # 所有 kernel LOCAL:0
```

- `masked_short`、`tiny` 覆盖 warp 按行；`tail` 覆盖 block 按行（512 线程、标量路径）；三种 sanitizer 在四个 step 上都无报错。
- 骨架检查：依次把骨架的 TODO(1..n) 换成答案代码，`./lab check 01 --step n` 通过，`--step n+1` 报 `TODO(n+1)`；检查后恢复骨架。

## 各 step 的性能变化

| step | rows4096 `[16384, 4096]` | rows32768 `[4096, 32768]` | rows256 `[262144, 256]` |
|---|---|---|---|
| 1 三遍、树形归约 | ≈ 1.9× | ≈ 0.6–0.75× | ≈ 0.07–0.12× |
| 2 warp shuffle | ≈ 1.9× | ≈ 0.75× | ≈ 0.18× |
| 3 online | ≈ 2.0× | ≈ 1.0× | ≈ 0.15× |
| 4 寄存器 + float4 | ≈ 2.4×（HBM 标称 84%） | ≈ 1.2×（68%） | ≈ 1.08×（84%） |

（torch 自身在三个 case 上分别约为 HBM 标称峰值的 35%、56%、77%。`--cold` 下步骤 4 的比例基本不变：≈ 2.25×、1.22×、1.07×。）

原因（Nsight Compute，`dram__bytes_read.sum` 与 `lts__t_sectors_srcunit_tex_op_read.sum`，以 x 的大小为单位）：

| step | rows4096 DRAM 读 | rows4096 L2 读 | rows32768 DRAM 读 | rows32768 L2 读 |
|---|---|---|---|---|
| 1 | 1.0 | 2.7 | 3.0 | 3.0 |
| 3 | 1.0 | 1.8 | 2.0 | 2.0 |
| 4 | 1.0 | 1.0 | 1.0 | 1.0 |

- 16 KB 的行：同时驻留的行约 148 × 8 × 16 KB ≈ 19 MB，重读命中 L2，三遍与一遍的 DRAM 流量相同，差别只在 L2 带宽和指令数。
- 128 KB 的行：同时驻留约 148 × 8 × 128 KB ≈ 155 MB，超过 126 MB 的 L2，重读回到 HBM。这时 online（步骤 3）少一遍读，有明显收益。
- 256 列的短行：步骤 1–3 的 256 个线程每人只有 1 个元素，时间花在 block 调度和归约的 barrier 上；步骤 4 换成 warp 一行后没有 barrier。
- 宽行的步骤 4 只有 ≈ 68%：每线程 50 个寄存器 × 1024 线程，一个 SM 只能驻留 1 个 block（`sm__warps_active` ≈ 43%），load、归约、store 三段不能和别的 block 重叠。

## 已知局限

- `rows32768` 的步骤 4 受“每 SM 一行”限制。可能的改进：把一行拆给一个 cluster 中的 2 个 CTA，用 distributed shared memory 合并 `(m, l)`（第 33 关的机制），或者把半行放进 shared memory，让每 SM 驻留两行。这些属于后续关卡的机制，本关不做。
- 超过 32768 列时退回步骤 3（读两遍 x）；列数不是 4 的倍数时用标量访存。
- “vs torch”比值在慢版本上偶尔跳动：同一输入下 torch 的 `rows4096` 有时测到约 1.65 倍的时间（进程间可复现的双峰），会让步骤 1–3 的比值虚高。步骤 4 和它的 perf gate 结果多次运行稳定。
