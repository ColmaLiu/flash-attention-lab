# 工具指南：SASS、Nsight Compute、Compute Sanitizer 与计时

`./lab check` 告诉你结果对不对，`./lab bench` 告诉你快不快。本文讲另外三个问题怎么回答：

| 问题 | 工具 |
|---|---|
| 编译器生成了什么指令？有没有 spill？ | `./lab sass`（第 1 节） |
| 时间花在哪里？瓶颈是访存、Tensor Core 还是指数？ | `./lab profile`（第 2 节） |
| 有没有越界、数据竞争、同步错误？ | `./lab sanitize`（第 3 节） |

第 5 节解释 `bench` 的计时方式和输出中的每一列。

所有命令都作用于 `students/<ID>/kernel.cu`；加 `--solution` 改为参考答案。输出文件都写在 Git 忽略的 `artifacts/` 下。

## 1. `./lab sass`：看编译器生成了什么

**SASS** 是 GPU 真正执行的机器指令。PTX 是一种虚拟指令集，`ptxas` 再把它翻译成 SASS。所以要看“到底发生了什么”，要看 SASS。

```bash
./lab sass 00              # 资源用量 + 指令统计
./lab sass 00 --show       # 另外打印完整 SASS
```

输出示例（00 关参考答案，节选）：

```
  (anonymous namespace)::transpose_naive(float const*, float*, int, int)
    REG:12 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:920 ...

Instruction counts (static, whole module):
  LDG            25  global load
  STG            25  global store
  LDS            24  shared load
  STS            24  shared store
  BAR             3  barrier

Full SASS: artifacts/sass/solution_00_transpose.sass
```

| 字段 | 含义 | 要注意什么 |
|---|---|---|
| `REG` | 每线程寄存器数 | 乘以每 block 线程数，判断一个 SM 能驻留几个 block（每 SM 65536 个） |
| `SHARED` | 静态 shared memory 字节数 | 动态 shared memory 不在这里 |
| `LOCAL`、`STACK` | 每线程 local memory | 大于 0 通常表示**寄存器 spill**，会变慢 |
| 指令统计 | 整个模块中每类指令出现的**静态**次数 | 不是执行次数。一条在循环里的指令只算一次 |

指令统计适合回答“有没有”：有没有 `LDSM`（用上了 `ldmatrix`）、有没有 `LDL`/`STL`（spill）、有没有 `MUFU.EX2`（指数走了 MUFU）。执行次数要用 `./lab profile`。

### 1.1 常见 SASS 指令

| SASS | 对应的 CUDA / PTX | 哪一关第一次出现 |
|---|---|---|
| `LDG` / `STG` | global memory 读 / 写（`ld.global` / `st.global`） | [00](../course/00_transpose/README.md) |
| `LDS` / `STS` | shared memory 读 / 写 | [00](../course/00_transpose/README.md) |
| `BAR.SYNC` | `__syncthreads()` | [00](../course/00_transpose/README.md) |
| `SHFL` | warp shuffle（`__shfl_xor_sync` 等） | [01](../course/01_softmax/README.md) |
| `MUFU.EX2` | `ex2.approx`，2 的幂。还有 `MUFU.RCP`（倒数）、`MUFU.LG2` 等 | [01](../course/01_softmax/README.md) |
| `FFMA` | FP32 乘加 | [10](../course/10_simt_gemm/README.md) |
| `LDL` / `STL` | local memory 读 / 写，通常意味着 spill | [01](../course/01_softmax/README.md) |
| `LDSM` | `ldmatrix`：从 shared memory 装 MMA fragment | [20](../course/20_mma_gemm/README.md) |
| `HMMA` | `mma.sync` 的 FP16/BF16 Tensor Core 指令 | [20](../course/20_mma_gemm/README.md) |
| `LDGSTS` | `cp.async`：global 直接写进 shared，不经过寄存器 | [21](../course/21_pipeline_gemm/README.md) |
| `DEPBAR` | `cp.async.wait_group` 等待 | [21](../course/21_pipeline_gemm/README.md) |
| `UTMALDG` / `UTMASTG` | TMA load / store（`cp.async.bulk.tensor`），后缀 `.2D`、`.4D` 是维数 | [30](../course/30_tma_mbarrier/README.md) |
| `SYNCS.*` | mbarrier 操作：`SYNCS.ARRIVE.TRANS64`（arrive / expect_tx）、`SYNCS.PHASECHK.TRANS64.TRYWAIT`（try_wait） | [30](../course/30_tma_mbarrier/README.md) |
| `ELECT` | `elect.sync`：在 warp 中选一个线程 | [30](../course/30_tma_mbarrier/README.md) |
| `FENCE.VIEW.ASYNC.*` | proxy fence（`fence.proxy.async` 等） | [30](../course/30_tma_mbarrier/README.md) |
| `UTCHMMA` / `UTCQMMA` | `tcgen05.mma`，FP16/BF16 / FP8 | [31](../course/31_tcgen05_gemm/README.md) |
| `UTCBAR` | `tcgen05.commit`：MMA 完成后到达 mbarrier | [31](../course/31_tcgen05_gemm/README.md) |
| `LDTM` / `STTM` | `tcgen05.ld` / `tcgen05.st`：TMEM 与寄存器之间 | [31](../course/31_tcgen05_gemm/README.md)、[34](../course/34_fa3/README.md) |

以 `U` 开头的指令（`UTMALDG`、`UTCHMMA`、`R2UR` 等）使用 **uniform datapath**：一个 warp 共享一份的标量寄存器。TMA 和 `tcgen05` 的描述符、地址都要放进 uniform 寄存器，所以 SASS 中会出现很多 `R2UR`（普通寄存器 → uniform 寄存器）。

### 1.2 读 SASS 的一个例子

00 关 §4.5 说：循环次数是编译期常数时，编译器会把多条 load 连续发出。验证方法：打开 `artifacts/sass/student_00_transpose.sass`，找到 `transpose_tiled` 的函数体（函数名经过 mangle，形如 `..._transpose_tiledILi1ELi8E...`，即 `<kPad=1, kThreadsY=8>`），数第一条 `STS` 之前有几条 `LDG`。

下面是参考答案步骤 3 的 SASS（`./lab sass 00 --solution`，只保留 LDG/STS/BAR）：

```
/*0200*/  @!P1 LDG.E.CONSTANT R16, desc[UR6][R22.64] ;
/*02c0*/  @!P3 LDG.E.CONSTANT R10, desc[UR6][R2.64] ;
/*02d0*/  @!P2 LDG.E.CONSTANT R4, desc[UR6][R4.64] ;
/*02e0*/  @!P0 LDG.E.CONSTANT R6, desc[UR6][R6.64] ;      ← 4 条 load 先全部发出
/*03c0*/  @!P1 STS [R9], R16 ;
/*0420*/  @!P3 STS [R9+0x420], R10 ;                      ← 0x420 = 1056 = 8 行 × 33 × 4 B
/*0430*/  @!P2 STS [R9+0x840], R4 ;
/*0440*/  @!P0 STS [R9+0xc60], R6 ;
/*0450*/        BAR.SYNC.DEFER_BLOCKING 0x0 ;             ← __syncthreads()
```

`@!P1` 是谓词：只有 `P1` 为假的线程执行这条指令，对应源码里的越界判断。`STS` 的偏移 `0x420` 说明 tile 每行 33 个 float（padding 生效了）。

## 2. `./lab profile`：Nsight Compute 指标

```bash
./lab profile 00 --step 2                 # 默认用本关第一个 perf case
./lab profile 34 --case perf --step 3
```

它用 Nsight Compute（`ncu`）采集一次 kernel 执行（先预热 3 次），打印两组指标：

- 每关都打印的基础指标：`gpu__time_duration`、`dram__throughput`、`sm__throughput`、`sm__warps_active`；
- 本关 `spec.json` 中 `profile_metrics` 列出的指标。

完整报告存在 `artifacts/profiles/*.ncu-rep`，可以用 `ncu-ui` 打开，或 `ncu --import <文件> --page details` 在终端查看。

**注意**：Nsight Compute 默认锁定时钟，`gpu__time_duration` 比 `bench` 的时间长。只用它比较指标，不要和 `bench` 的时间直接比。

### 2.1 按问题选指标

| 想知道 | 指标 | 怎么读 | 哪一关用 |
|---|---|---|---|
| global 访存是否合并 | `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum` ÷ `l1tex__t_requests_..._{ld,st}.sum` | 每个 warp 请求的 sector（32 B）数。32 个线程各读 4 B 连续数据时是 4；完全不合并时是 32 | [00](../course/00_transpose/README.md) |
| shared memory bank conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` | 冲突次数，理想为 0 或接近 0 | [00](../course/00_transpose/README.md)、[10](../course/10_simt_gemm/README.md)、[20](../course/20_mma_gemm/README.md)、[30](../course/30_tma_mbarrier/README.md) |
| HBM 实际读写多少字节 | `dram__bytes_read.sum`、`dram__bytes_write.sum` | 和你算的理论字节数比较 | [11](../course/11_naive_attention/README.md)、[12](../course/12_fa1/README.md) |
| HBM 带宽用了多少 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 访存型 kernel 应该接近上限 | [30](../course/30_tma_mbarrier/README.md) |
| L2 命中 | `lts__t_sector_hit_rate.pct`；SM 从 L2 读了多少：`lts__t_sectors_srcunit_tex_op_read.sum` | tile 顺序、multicast 改变的是这两个数 | [21](../course/21_pipeline_gemm/README.md)、[32](../course/32_ws_gemm/README.md)、[33](../course/33_2cta_gemm/README.md) |
| FP32 FMA 单元多忙 | `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | SIMT GEMM 的上限 | [10](../course/10_simt_gemm/README.md) |
| Tensor Core 多忙 | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 部分 2–4 最重要的一个数 | [20](../course/20_mma_gemm/README.md)–[34](../course/34_fa3/README.md) |
| `mma.sync` 执行了多少条 | `sm__inst_executed_pipe_tensor_subpipe_hmma.sum` | 和理论 MMA 条数比较 | [20](../course/20_mma_gemm/README.md)、[22](../course/22_fa2/README.md) |
| 指数（MUFU）执行了多少条 | `smsp__inst_executed_pipe_xu.sum` | XU 管线包含 MUFU。attention 中约为 B·H·N²/32（每个分数一次，按 warp 指令计） | [22](../course/22_fa2/README.md)、[34](../course/34_fa3/README.md) |
| warp 在等什么 | `smsp__average_warps_issue_stalled_{barrier,long_scoreboard}_per_issue_active.ratio` | `barrier`：等 `__syncthreads`；`long_scoreboard`：等 global/local memory | [01](../course/01_softmax/README.md) |
| 驻留了多少 warp | `sm__warps_active.avg.pct_of_peak_sustained_active`、`launch__registers_per_thread` | 占 64 个 warp 的比例 | [10](../course/10_simt_gemm/README.md)、[12](../course/12_fa1/README.md) |

指标的完整定义见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 3. `./lab sanitize`：Compute Sanitizer

```bash
./lab sanitize 21 --step 1 --case aligned --tool racecheck
```

sanitizer 会让 kernel 慢几十到几百倍，所以必须用 `--case` 选一个小 case。命令自动关掉 PyTorch 的缓存分配器（`PYTORCH_NO_CUDA_MEMORY_CACHING=1`），让每个张量单独分配，这样 memcheck 才能发现越界到相邻张量的访问。发现错误时退出码是 10。

| 工具 | 检查什么 | 典型发现 |
|---|---|---|
| `memcheck`（默认） | global / shared memory 越界与未对齐访问 | 尾部 tile 的越界判断写错 |
| `racecheck` | 同一 block 内线程之间对 shared memory 的读写冲突 | 少了 `__syncthreads()`；双缓冲时 buffer 被提前覆盖 |
| `synccheck` | 同步原语用错 | `__syncthreads()` 放在只有部分线程能到达的分支里 |
| `initcheck` | 读了从未写过的 global memory | 读未初始化的输出 |

### 3.1 sanitizer 看不到什么

sanitizer 报告 0 错误，**不等于**没有错误。部分 3、4 的 kernel 尤其如此：

| 盲区 | 后果 | 证据 |
|---|---|---|
| racecheck 不跟踪 async proxy（TMA、`tcgen05`）对 shared memory 的访问 | 删掉 `fence.proxy.async` 或删掉 producer 对 `empty` barrier 的等待，racecheck 仍报 0 hazard | [30 关 NOTES](../solutions/30_tma_mbarrier/NOTES.md) 的实验；[32 关 NOTES](../solutions/32_ws_gemm/NOTES.md) |
| TMEM 不是 sanitizer 检查的内存 | `tcgen05.ld` 读错 lane、TMEM 交接少了 fence，不会被报告 | 由此推断；按 [31 §4.7](../course/31_tcgen05_gemm/README.md) 的规则推理 |
| `cta_group::2` 的 TMEM 分配 | racecheck 可能报出与分配有关的误报 | [33 关 NOTES](../solutions/33_2cta_gemm/NOTES.md) |
| scratch 预先填了垃圾值 | initcheck 把它当作已初始化，看不出“没初始化就读 scratch” | 由此推断；`check` 的 NaN 检查负责这一类 |

所以异步代码的正确性要靠两样东西：**按 PTX 内存模型推理每个同步的作用**（讲义 §4、§5 的 barrier 地图），加上**能暴露问题的 case**（多级环形 buffer 绕多圈的长 K、尾部、非默认 stream）。

## 4. 直接使用 cuobjdump 与 nvdisasm

`./lab sass` 只是包装了 `cuobjdump`。编译产物在 `.cache/build/<student|solution>_<ID>/` 下的 `.so` 中。需要更多细节时可以直接用：

```bash
lib=$(ls -t .cache/build/student_00_transpose/*/*.so | head -1)
cuobjdump -res-usage "$lib"          # 每个 kernel 的寄存器、shared、local 用量
cuobjdump -sass "$lib"               # 全部 SASS
cuobjdump -ptx "$lib"                # 嵌入的 PTX（如果有）
cuobjdump -xelf all "$lib"           # 把 cubin 解出到当前目录
nvdisasm -g -c <file>.cubin          # 带源码行号（-g）的 SASS；课程用 -lineinfo 编译
nvdisasm -cfg <file>.cubin | dot -Tsvg > cfg.svg    # 控制流图（需要 graphviz）
```

参数说明见 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。

## 5. 计时：graph 与 cold

`./lab bench` 在同一次运行中，用同一份输入，计时你的 kernel、参考答案和库基线。三者轮流执行，减少频率漂移的影响。编译、`make_plan`、造输入、分配内存都**不计入**。

| 模式 | 命令 | 做法 | 测的是什么 |
|---|---|---|---|
| graph（默认） | `./lab bench ID` | 每个候选的 launch 录进一个 CUDA Graph（连续 20 次），回放 15 轮，用 CUDA event 计时，取中位数 | 去掉 launch 开销后的稳态时间。输入小于 L2 时，数据可能来自 L2 |
| cold | `./lab bench ID --cold` | 每次计时前写一个 2 倍 L2 大小的 buffer 来清空 L2；每次只 launch 一次 | 输入来自 HBM 时的时间，包含单次 launch 的开销 |

**两种模式的数字不能混在一起算比值。** 例如不能用 graph 下答案的时间除以 cold 下库的时间。

输入远大于 L2（126 MB）时两者差别不大；小输入时 cold 明显更慢。

### 5.1 读 `bench` 的输出

```
00_transpose  case big [8192, 8192] fp32  step 4  timing: graph
              median ms      GB/s  % peak    vs torch
  solution       0.0909    5907.7   73.8%       3.17x
  torch          0.2882    1862.6   23.3%       1.00x
  copy           0.0836    6422.0   80.3%       3.45x

% peak uses nominal B200 datasheet numbers; 'vs' > 1 means faster.
```

| 列 | 含义 |
|---|---|
| `median ms` | 中位数耗时 |
| `GB/s` 或 `TFLOP/s` | 访存型算子按“必须读写的字节数 / 时间”；计算型算子按“FLOP / 时间”。字节数和 FLOP 由算子本身决定，与实现无关 |
| `% peak` | 相对 B200 **标称**峰值：HBM 8 TB/s；FP32 75、FP16/BF16 2250、FP8 4500 TFLOP/s（稠密）。实际能达到的上限更低，见 [B200 架构概览](b200.md) |
| `vs <基线>` | **基线耗时 ÷ 本行耗时**。基线是除 student、solution 外的第一个库（表头写明）。大于 1 表示比基线快 |

各关的库基线：

| 基线 | 用于 | 说明 |
|---|---|---|
| `torch` / `copy` | transpose、softmax | PyTorch 原生算子；`copy` 是无转置的连续拷贝，作为带宽上限的参照 |
| `cublas` | GEMM | 经 PyTorch 调用；FP32 关闭 TF32 |
| `efficient` | FP32 attention（11、12 关） | PyTorch memory-efficient attention |
| `cudnn` | FP16/BF16 attention | 强制 cuDNN SDPA 后端，失败即报 unavailable |
| `fa4` | FP16/BF16 attention | FlashAttention-4（`flash_attn.cute`），需要 `bootstrap.sh --with-fa4` |

不可用的基线会显示为 `unavailable: <原因>`，不会被别的实现顶替。

### 5.2 让数字可信

- 先 `check` 再 `bench`：`bench` 只做一次快速的结果检查。
- 一次只改一处，在同一张卡上前后对比。
- 多跑几次，看比值的范围。库本身也有几个百分点的波动（例如 [33 关 §7](../course/33_2cta_gemm/README.md) 中 cuBLAS 的 `wide`）。
- 每次 `bench` 的完整结果（含源码指纹）存在 `artifacts/bench/*.json`。
- 区分三类结论：实测（有 `bench`/`profile` 输出）、源码推断（读 SASS 或代码得出）、未验证的假设。
