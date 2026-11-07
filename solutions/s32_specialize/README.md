# CUDA/PTX：分工流水线与形状选择

本关在 [kernel_impl.cuh](kernel_impl.cuh) 中提供独立的原生实现，入口为 `entry.py → binding.cpp → kernel.cu`。核心计算使用 TMA、`tcgen05.mma` 与 TMEM；不调用 CUTLASS、CuTe 或 GEMM 库。s31 的串行教学基线保留在自己的目录。输入是连续 BF16 `A[M,K]`、`B[K,N]`，输出是连续 FP32 `C[M,N]`；合法域为正尺寸且 `M%64=N%8=K%16=0`，并满足公共接口的对齐要求。

## 从 s31 增加了哪些机制

s31 每个 K tile 串行完成 TMA、MMA 和等待。这里第五个 warp 推进 TMA，第一 warp 的一个线程推进 MMA，前四个 warp 最终读取 TMEM。每个 slot 有独立的 full/empty barrier，producer 可提前填充尚未被 MMA 使用的 slot。短 K、长 K、窄 N 使用不同资源预算，减少流水线等待或不足的 CTA 并行度。

| 输入特征 | 默认 CTA tile `BM×BN×BK` | 缓冲数 | 数据搬运 |
|---|---|---:|---|
| `K>=512, N<=128` | `64×32×256` | 4 | 2D TMA，实际 32 列 B 与 64B swizzle |
| `K>=512, N>128` | `128×64×128` | 4 | 条件满足时使用 rank-3 TMA |
| 其余输入 | `128×64/128×128/256` | 1 或 2 | 2D TMA；N tile 由尺寸选择 |

QK perf case 的默认 tile 是 `128×128×128`，只有一次 K tile，单缓冲即可。窄 N 路线直接产生完整输出，不需要 split-K；代表性的 PV 输入得到 128 个独立 CTA。宽 N 长 K 路线用四组 K128 缓冲，和两组 K256 缓冲消耗相同 shared memory，但更早地向 MMA 供应数据。具体选择始终以 host `prepare_gemm` 为准，这些规则覆盖尺寸范围而非 perf shape 白名单。

## 输入和输出 ownership

A 在 shared memory 中按 64 个 BF16 元素的 K 面板排列，每个面板是 `BM×64`。B 按最多 64 列的 N 面板排列，每个面板是 `BK×BWidth`；`BWidth=min(BN,64)`。64 元素面板使用 128B swizzle，32 元素面板使用 64B swizzle。MMA descriptor 的 leading/stride 和每个 K16 起点必须与这两个布局一致，不能仅改变分配大小。

对宽 N 长 K 路线，若 M、N、K 均为 64 的倍数且 BK 不超过 256，rank-3 tensor map 将一个 `64` 元素面板作为最内维。A 的 box 是 `[64,BM,BK/64]`，B 的 box 是 `[64,BK,BN/64]`。一次 A copy 和一次 B copy 就能覆盖完整 stage。其余合法尺寸使用 2D 面板拷贝，TMA 的越界填零处理 M/N/K 尾部；输出使用显式行列检查。

`tcgen05.mma` 每次处理 K16，BF16 乘法使用 FP32 accumulator。长 K 路线在整个 issuer 循环外进行 warp election。分工路径预先构造第零个 slot 的 A/B descriptor，每轮仅叠加 slot 和 K16 的地址偏移，复用 leading/stride/swizzle 字段；地址以 16 字节为单位。代码显式更新低 32 位并保留高位布局标记，合法 shared memory 资源上限保证地址不越过 14 位起点字段。完整 warp 必须进入 election，不能把 election 放在已经只剩单线程的分支内。

最终 `tcgen05.ld.16x256b.x4` 将 16×32 的结果块分配给一个 warp：每个 lane 得到两行、若干组相邻两列。代码用 `float2` 将这些元素写入连续输出。BM=64 时，每个 warp 拥有 16 个逻辑行；BM=128 时拥有 32 行。TMEM 的物理 row 地址与逻辑输出 row 分开计算，避免把 M64 当成截短的 M128。

## slot 的生命周期

- `full[s]` 跟踪 TMA。producer 先为完整 A/B stage 的字节数执行 `arrive.expect_tx`，再发出对应拷贝。MMA issuer 必须成功等待 full 后才能读取 operand。
- `empty[s]` 跟踪 MMA。issuer 为当前 stage 的全部 K16 指令执行 commit；producer 必须成功等待其完成，才能覆盖同一 stage。
- 迭代 i 使用 `slot=i%Stages` 和 `phase=(i/Stages)&1`。producer 首轮无需等待 empty，回绕时等待上一轮 phase；consumer 等待本轮 full phase。

这里没有逐 K tile 的 CTA barrier。最后一条 MMA 的完成通知由同一 issuer 等待，之后通过 CTA 同步和 `after_thread_sync` fence 把结果交给四个 epilogue warp。各 warp 完成 TMEM load 的 `wait::ld`，再用 before/after-thread-sync fence 与 CTA barrier 保证所有读取结束，最后释放 TMEM。完成等待与跨线程交接是不同的要求。

## 可控配置与计时

`entry.py` 接受 `tile_m`、`tile_n`、`tile_k`、`stages`、`split_k`、`single_issuer`；零值选择 host 默认。`dispatch` 只列出支持且可装入 shared memory 的组合，未支持的组合明确报错。显式 split-K 会为各 split 分配独立 FP32 partial，执行一次后续归约；它是供比较的可选路径。`single_issuer=1` 将 TMA 与 MMA 交给同一线程，保留 ring 的读写依赖，可与分工路径做消融。

prepare 完成编译、tensor-map 编码、输出与 partial 分配；计时只覆盖 launch。若使用 split-K，归约属于被测操作，不能从延迟中排除。s31→s32 的总体比较包含 tile 与流水线选择的共同影响；研究某项机制时，应固定 tile 和其他选项，只改变 stages 或 issuer 分工。

## 验收与局限

先检查全部 correctness cases、非默认 stream、重复污染输出和输入不变性，再运行 memcheck/racecheck/synccheck。`long_k` 检查多轮累加与 ring 回绕，`rectangular` 检查合法非整 tile 边界。数值通过后仍需说明每个 shared/TMEM 位置的所有者和完整 happens-before 链。

三项 perf case 必须分别比较 s31、s32 和 BF16→FP32 的 cuBLAS 基线。最终门槛是同组 CUDA Graph median 的 `library_ms / solution_ms >= 0.9`，逐 case 通过，不以平均值代替。更多 stage 增加 shared memory，窄 tile 重复加载输入，split-K 增加 launch 与输出流量；这些取舍需要实测，不能仅从占用率或 FLOP 数推断收益。原始测量、硬件快照与未采用候选保存在忽略的 artifacts 目录，不写入通用课程结果。

指令语义依据 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)。固定 CUTLASS 的 `cute/arch/mma_sm100_umma.hpp`、`mma_sm100_desc.hpp` 与 `copy_sm90_tma.hpp` 用于核对原生指令和 descriptor 定义；本地 kernel 独立实现，不依赖这些模板执行计算。依赖版本与许可见 [sources](../../docs/sources.md)。
