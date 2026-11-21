# Tools guide: SASS, Nsight Compute, Compute Sanitizer and timing

`./lab check` tells you whether the result is correct, and `./lab bench` tells you whether it is fast. This doc covers how to answer three more questions:

| Question | Tool |
|---|---|
| What instructions did the compiler generate? Is there any spill? | `./lab sass` (section 1) |
| Where does the time go? Is the bottleneck memory, the Tensor Core or the exponential? | `./lab profile` (section 2) |
| Are there out-of-bounds accesses, data races or synchronization errors? | `./lab sanitize` (section 3) |

Section 5 explains how `bench` times kernels and what each column of its output means.

Every command acts on `students/<ID>/kernel.cu`; add `--solution` to use the reference solution instead. All output files go under the Git-ignored `artifacts/`.

## 1. `./lab sass`: see what the compiler generated

**SASS** is the machine code the GPU actually executes. PTX is a virtual instruction set, and `ptxas` translates it into SASS. So to see "what really happens", look at the SASS.

```bash
./lab sass 00              # resource usage + instruction counts
./lab sass 00 --show       # also print the full SASS
```

Sample output (reference solution of lab 00, excerpt):

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

| Field | Meaning | What to watch for |
|---|---|---|
| `REG` | Registers per thread | Multiply by threads per block to judge how many blocks an SM can hold (65536 per SM) |
| `SHARED` | Static shared memory bytes | Dynamic shared memory is not included here |
| `LOCAL`, `STACK` | Local memory per thread | Greater than 0 usually means a **register spill**, which slows the kernel down |
| Instruction counts | **Static** count of each instruction class in the whole module | Not execution counts. An instruction inside a loop counts once |

Instruction counts are good for "is it there?" questions: is there an `LDSM` (`ldmatrix` is in use), is there an `LDL`/`STL` (spill), is there a `MUFU.EX2` (the exponential goes through MUFU). For execution counts, use `./lab profile`.

### 1.1 Common SASS instructions

| SASS | Matching CUDA / PTX | First appears in lab |
|---|---|---|
| `LDG` / `STG` | global memory read / write (`ld.global` / `st.global`) | [00](../course/00_transpose/README.md) |
| `LDS` / `STS` | shared memory read / write | [00](../course/00_transpose/README.md) |
| `BAR.SYNC` | `__syncthreads()` | [00](../course/00_transpose/README.md) |
| `SHFL` | warp shuffle (`__shfl_xor_sync` etc.) | [01](../course/01_softmax/README.md) |
| `MUFU.EX2` | `ex2.approx`, power of 2. There are also `MUFU.RCP` (reciprocal), `MUFU.LG2` and others | [01](../course/01_softmax/README.md) |
| `FFMA` | FP32 multiply-add | [10](../course/10_simt_gemm/README.md) |
| `LDL` / `STL` | local memory read / write, usually means a spill | [01](../course/01_softmax/README.md) |
| `LDSM` | `ldmatrix`: loads an MMA fragment from shared memory | [20](../course/20_mma_gemm/README.md) |
| `HMMA` | The FP16/BF16 Tensor Core instruction of `mma.sync` | [20](../course/20_mma_gemm/README.md) |
| `LDGSTS` | `cp.async`: global written directly into shared, without passing through registers | [21](../course/21_pipeline_gemm/README.md) |
| `DEPBAR` | `cp.async.wait_group` wait | [21](../course/21_pipeline_gemm/README.md) |
| `UTMALDG` / `UTMASTG` | TMA load / store (`cp.async.bulk.tensor`); the suffixes `.2D`, `.4D` give the rank | [30](../course/30_tma_mbarrier/README.md) |
| `SYNCS.*` | mbarrier operations: `SYNCS.ARRIVE.TRANS64` (arrive / expect_tx), `SYNCS.PHASECHK.TRANS64.TRYWAIT` (try_wait) | [30](../course/30_tma_mbarrier/README.md) |
| `ELECT` | `elect.sync`: picks one thread in the warp | [30](../course/30_tma_mbarrier/README.md) |
| `FENCE.VIEW.ASYNC.*` | proxy fence (`fence.proxy.async` etc.) | [30](../course/30_tma_mbarrier/README.md) |
| `UTCHMMA` / `UTCQMMA` | `tcgen05.mma`, FP16/BF16 / FP8 | [31](../course/31_tcgen05_gemm/README.md) |
| `UTCBAR` | `tcgen05.commit`: arrives on an mbarrier after the MMAs complete | [31](../course/31_tcgen05_gemm/README.md) |
| `LDTM` / `STTM` | `tcgen05.ld` / `tcgen05.st`: between TMEM and registers | [31](../course/31_tcgen05_gemm/README.md), [34](../course/34_fa3/README.md) |

Instructions starting with `U` (`UTMALDG`, `UTCHMMA`, `R2UR` and so on) use the **uniform datapath**: scalar registers that the whole warp shares as one copy. The descriptors and addresses for TMA and `tcgen05` must go into uniform registers, so SASS contains many `R2UR` (ordinary register → uniform register).

### 1.2 An example of reading SASS

Lab 00 §4.5 says: when the loop count is a compile-time constant, the compiler issues several loads back to back. To verify this, open `artifacts/sass/student_00_transpose.sass`, find the function body of `transpose_tiled` (the name is mangled, like `..._transpose_tiledILi1ELi8E...`, that is, `<kPad=1, kThreadsY=8>`), and count how many `LDG` come before the first `STS`.

Below is the SASS of step 3 of the reference solution (`./lab sass 00 --solution`, keeping only LDG/STS/BAR):

```
/*0200*/  @!P1 LDG.E.CONSTANT R16, desc[UR6][R22.64] ;
/*02c0*/  @!P3 LDG.E.CONSTANT R10, desc[UR6][R2.64] ;
/*02d0*/  @!P2 LDG.E.CONSTANT R4, desc[UR6][R4.64] ;
/*02e0*/  @!P0 LDG.E.CONSTANT R6, desc[UR6][R6.64] ;      ← all 4 loads are issued first
/*03c0*/  @!P1 STS [R9], R16 ;
/*0420*/  @!P3 STS [R9+0x420], R10 ;                      ← 0x420 = 1056 = 8 rows × 33 × 4 B
/*0430*/  @!P2 STS [R9+0x840], R4 ;
/*0440*/  @!P0 STS [R9+0xc60], R6 ;
/*0450*/        BAR.SYNC.DEFER_BLOCKING 0x0 ;             ← __syncthreads()
```

`@!P1` is a predicate: only threads where `P1` is false execute the instruction; it matches the bounds check in the source. The `STS` offset `0x420` shows that each tile row holds 33 floats (the padding took effect).

## 2. `./lab profile`: Nsight Compute metrics

```bash
./lab profile 00 --step 2                 # uses the lab's first perf case by default
./lab profile 34 --case perf --step 3
```

It uses Nsight Compute (`ncu`) to capture one kernel execution (after 3 warmup runs) and prints two groups of metrics:

- The base metrics printed for every lab: `gpu__time_duration`, `dram__throughput`, `sm__throughput`, `sm__warps_active`;
- The metrics listed in `profile_metrics` in the lab's `spec.json`.

The full report is saved in `artifacts/profiles/*.ncu-rep`. Open it with `ncu-ui`, or view it in the terminal with `ncu --import <file> --page details`.

**Note**: Nsight Compute locks clocks by default, so `gpu__time_duration` is longer than the `bench` time. Use it only to compare metrics; do not compare it directly with `bench` times.

### 2.1 Pick metrics by question

| You want to know | Metric | How to read it | Used in lab |
|---|---|---|---|
| Whether global accesses are coalesced | `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum` ÷ `l1tex__t_requests_..._{ld,st}.sum` | Sectors (32 B) per warp request. 4 when 32 threads each read 4 B of contiguous data; 32 when completely uncoalesced | [00](../course/00_transpose/README.md) |
| shared memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` | Number of conflicts; ideally 0 or close to 0 | [00](../course/00_transpose/README.md), [10](../course/10_simt_gemm/README.md), [20](../course/20_mma_gemm/README.md), [30](../course/30_tma_mbarrier/README.md) |
| How many bytes HBM actually reads and writes | `dram__bytes_read.sum`, `dram__bytes_write.sum` | Compare with the theoretical byte count you computed | [11](../course/11_naive_attention/README.md), [12](../course/12_fa1/README.md) |
| How much HBM bandwidth is used | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | A memory-bound kernel should be close to the limit | [30](../course/30_tma_mbarrier/README.md) |
| L2 hits | `lts__t_sector_hit_rate.pct`; how much the SMs read from L2: `lts__t_sectors_srcunit_tex_op_read.sum` | Tile order and multicast change these two numbers | [21](../course/21_pipeline_gemm/README.md), [32](../course/32_ws_gemm/README.md), [33](../course/33_2cta_gemm/README.md) |
| How busy the FP32 FMA units are | `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | The upper limit of a SIMT GEMM | [10](../course/10_simt_gemm/README.md) |
| How busy the Tensor Cores are | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | The single most important number in Parts 2–4 | [20](../course/20_mma_gemm/README.md)–[34](../course/34_fa3/README.md) |
| How many `mma.sync` executed | `sm__inst_executed_pipe_tensor_subpipe_hmma.sum` | Compare with the theoretical number of MMAs | [20](../course/20_mma_gemm/README.md), [22](../course/22_fa2/README.md) |
| How many exponentials (MUFU) executed | `smsp__inst_executed_pipe_xu.sum` | The XU pipe includes MUFU. In attention, about B·H·N²/32 (once per score, counted in warp instructions) | [22](../course/22_fa2/README.md), [34](../course/34_fa3/README.md) |
| What warps are waiting for | `smsp__average_warps_issue_stalled_{barrier,long_scoreboard}_per_issue_active.ratio` | `barrier`: waiting on `__syncthreads`; `long_scoreboard`: waiting on global/local memory | [01](../course/01_softmax/README.md) |
| How many warps are resident | `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__registers_per_thread` | Fraction of 64 warps | [10](../course/10_simt_gemm/README.md), [12](../course/12_fa1/README.md) |

Full metric definitions are in the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## 3. `./lab sanitize`: Compute Sanitizer

```bash
./lab sanitize 21 --step 1 --case aligned --tool racecheck
```

The sanitizer makes a kernel tens to hundreds of times slower, so you must pick a small case with `--case`. The command turns off PyTorch's caching allocator automatically (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) so that each tensor gets its own allocation; only then can memcheck catch accesses that run past one tensor into its neighbor. When it finds an error, the exit code is 10.

| Tool | What it checks | Typical finding |
|---|---|---|
| `memcheck` (default) | Out-of-bounds and misaligned global / shared memory accesses | Wrong bounds check for the tail tile |
| `racecheck` | Read/write conflicts on shared memory between threads in the same block | A missing `__syncthreads()`; with double buffering, a buffer overwritten too early |
| `synccheck` | Misused synchronization primitives | `__syncthreads()` inside a branch that only some threads reach |
| `initcheck` | Reads of global memory that was never written | Reading an uninitialized output |

### 3.1 What the sanitizer cannot see

A sanitizer report of 0 errors **does not mean** there are no errors. This is especially true for the kernels of Parts 3 and 4:

| Blind spot | Consequence | Evidence |
|---|---|---|
| racecheck does not track shared memory accesses from the async proxy (TMA, `tcgen05`) | Remove `fence.proxy.async`, or remove the producer's wait on the `empty` barrier, and racecheck still reports 0 hazards | The experiments in the [lab 30 NOTES](../solutions/30_tma_mbarrier/NOTES.md); [lab 32 NOTES](../solutions/32_ws_gemm/NOTES.md) |
| TMEM is not memory the sanitizer checks | A `tcgen05.ld` that reads the wrong lane, or a missing fence in a TMEM handoff, is not reported | Inferred; reasoned from the rules in [31 §4.7](../course/31_tcgen05_gemm/README.md) |
| TMEM allocation with `cta_group::2` | racecheck may report false positives related to the allocation | [lab 33 NOTES](../solutions/33_2cta_gemm/NOTES.md) |
| scratch prefilled with garbage values | initcheck treats it as initialized and cannot see "reading scratch before initializing it" | Inferred; the NaN check in `check` covers this class |

So the correctness of asynchronous code rests on two things: **reasoning about what each synchronization does under the PTX memory model** (the barrier maps in §4 and §5 of the lessons), plus **cases that can expose problems** (a long K that wraps a multi-stage ring buffer many times, tails, non-default streams).

## 4. Using cuobjdump and nvdisasm directly

`./lab sass` only wraps `cuobjdump`. The build output is in the `.so` under `.cache/build/<student|solution>_<ID>/`. When you need more detail, use the tools directly:

```bash
lib=$(ls -t .cache/build/student_00_transpose/*/*.so | head -1)
cuobjdump -res-usage "$lib"          # register, shared and local usage of each kernel
cuobjdump -sass "$lib"               # all SASS
cuobjdump -ptx "$lib"                # embedded PTX (if any)
cuobjdump -xelf all "$lib"           # extract the cubins into the current directory
nvdisasm -g -c <file>.cubin          # SASS with source line numbers (-g); the course compiles with -lineinfo
nvdisasm -cfg <file>.cubin | dot -Tsvg > cfg.svg    # control flow graph (needs graphviz)
```

The flags are described in [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).

## 5. Timing: graph and cold

In a single run, with the same input, `./lab bench` times your kernel, the reference solution and the library baselines. The three take turns executing, which reduces the effect of clock drift. Compilation, `make_plan`, input generation and memory allocation are **not included**.

| Mode | Command | Method | What it measures |
|---|---|---|---|
| graph (default) | `./lab bench ID` | Records each candidate's launches into a CUDA Graph (20 in a row), replays it 15 rounds, times with CUDA events, takes the median | Steady-state time without launch overhead. When the input is smaller than L2, the data may come from L2 |
| cold | `./lab bench ID --cold` | Before each timing, writes a buffer twice the size of L2 to flush L2; launches only once each time | Time when the input comes from HBM, including the overhead of a single launch |

**Do not compute ratios between numbers from the two modes.** For example, do not divide the solution's graph time by a library's cold time.

When the input is much larger than L2 (126 MB), the two differ little; for small inputs, cold is clearly slower.

### 5.1 Reading the `bench` output

```
00_transpose  case big [8192, 8192] fp32  step 4  timing: graph
              median ms      GB/s  % peak    vs torch
  solution       0.0909    5907.7   73.8%       3.17x
  torch          0.2882    1862.6   23.3%       1.00x
  copy           0.0836    6422.0   80.3%       3.45x

% peak uses nominal B200 datasheet numbers; 'vs' > 1 means faster.
```

| Column | Meaning |
|---|---|
| `median ms` | Median time |
| `GB/s` or `TFLOP/s` | Memory-bound operators use "bytes that must be read and written / time"; compute-bound operators use "FLOP / time". The bytes and FLOP are set by the operator itself, independent of the implementation |
| `% peak` | Relative to the B200 **nominal** peak: HBM 8 TB/s; FP32 75, FP16/BF16 2250, FP8 4500 TFLOP/s (dense). The upper bound you can actually reach is lower; see the [B200 architecture overview](b200.md) |
| `vs <baseline>` | **Baseline time ÷ this row's time**. The baseline is the first library other than student and solution (named in the header). Greater than 1 means faster than the baseline |

Library baselines per lab:

| Baseline | Used for | Notes |
|---|---|---|
| `torch` / `copy` | transpose, softmax | PyTorch native operators; `copy` is a contiguous copy without transpose, as a reference for the bandwidth upper bound |
| `cublas` | GEMM | Called through PyTorch; TF32 is off for FP32 |
| `efficient` | FP32 attention (labs 11, 12) | PyTorch memory-efficient attention |
| `cudnn` | FP16/BF16 attention | Forces the cuDNN SDPA backend; reports unavailable if that fails |
| `fa4` | FP16/BF16 attention | FlashAttention-4 (`flash_attn.cute`); needs `bootstrap.sh --with-fa4` |

An unavailable baseline shows as `unavailable: <reason>` and is not replaced by another implementation.

### 5.2 Making the numbers trustworthy

- Run `check` before `bench`: `bench` does only one quick result check.
- Change one thing at a time, and compare before and after on the same card.
- Run several times and look at the range of the ratio. The libraries themselves vary by a few percent (for example cuBLAS on `wide` in [lab 33 §7](../course/33_2cta_gemm/README.md)).
- The full results of every `bench` run (including a source fingerprint) are saved in `artifacts/bench/*.json`.
- Distinguish three kinds of conclusions: measured (there is `bench`/`profile` output), inferred from source (from reading SASS or code), and unverified assumptions.
