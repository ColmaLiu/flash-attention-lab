# flash-attention-lab

On an NVIDIA B200 (SM100), start from a matrix transpose and work step by step toward a FlashAttention that comes close to library performance.

This course writes every kernel in **CUDA C++ and inline PTX**, with FlashAttention 1→4 as the main thread: each FlashAttention generation removes one bottleneck, and each lab first practices one new hardware mechanism on a simple operator and then uses it in attention.

## Who this is for

| You need to know | You do not need to know |
|---|---|
| Basic C/C++, basic Python | Hardware details such as Tensor Core, TMA, TMEM (the lessons start from scratch) |
| Having written one elementwise CUDA kernel | The FlashAttention papers or the PTX manual |
| What grid / block / thread and shared memory are | CUTLASS, CuTe or Triton |
| The math definitions of matrix multiplication and softmax | |

If you have never written CUDA, read [Starting from zero](docs/starting-from-zero.md) first.

After the course you should be able to:

1. Derive the HBM traffic of attention from its math definition, and explain which bottleneck each generation FA1→FA4 removes;
2. Write correct code for `mma.sync`, TMA, mbarrier, `tcgen05`/TMEM, warp specialization and 2-CTA MMA on your own, and explain what each synchronization does;
3. Verify optimization hypotheses with SASS, Nsight Compute and controlled timing;
4. Read the source structure of upstream FA4 (CuTe DSL) and the CUTLASS SM100 FMHA.

## Requirements

| Item | Requirement |
|---|---|
| GPU | B200 or another GPU with compute capability 10.0 (SM100). Consumer Blackwell (SM120) has no `tcgen05` and is not supported |
| System | Linux x86_64, an NVIDIA driver that supports CUDA 12.8 |
| Software | Conda; CUDA Toolkit 12.8 (the install script can put it into the environment) |

## Installation

```bash
git clone -b v2 https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh            # optional --with-cuda, --with-fa4, see the table below
```

The install script creates a separate Conda environment `flash-attention-lab`, initializes the pinned upstream sources in `third_party/`, and installs PyTorch and the other Python dependencies. Two optional flags:

| Flag | What it does | When you need it |
|---|---|---|
| `--with-cuda` | Installs CUDA 12.8 and GCC 13 into the environment | The system has no CUDA 12.8 `nvcc` |
| `--with-fa4` | Installs FlashAttention-4 (pinned version) as a performance baseline | You want an FA4 column in the `bench` output of Parts 3 and 4 |

Details are in [Environment and dependencies](docs/environment.md).

## Quick start

```bash
./lab doctor                 # check GPU, nvcc, PyTorch and tools
./lab list                   # course list and your progress
```

Then open the lesson of the first lab, [course/00_transpose/README.md](course/00_transpose/README.md), complete `TODO(1)` in `students/00_transpose/kernel.cu`, and run:

```bash
./lab check 00 --step 1      # correctness check for step 1
./lab bench 00 --step 1      # compare speed with the reference solution and PyTorch
```

Before you write any code, `check` reports `TODO(1) is not implemented yet`. This is expected.

If the machine has several GPUs, pick an idle one with `LAB_GPU`, for example `LAB_GPU=3 ./lab check 00 --step 1`.

## Course map

The first digit of a lab ID is the "Part": Part 0 is the preliminaries, and Parts 1–4 correspond to FA1–FA4.

| ID | Topic | What this lab teaches |
|---|---|---|
| **Part 0 · Preliminaries** | | |
| [00_transpose](course/00_transpose/README.md) | Matrix transpose | Thread-to-address mapping, coalesced access, shared memory, bank conflicts, memory accesses in flight |
| [01_softmax](course/01_softmax/README.md) | Row softmax | warp shuffle, block reduction, numerical stability, online softmax |
| **Part 1 · FA1: IO-aware tiling** | | |
| [10_simt_gemm](course/10_simt_gemm/README.md) | FP32 tiled GEMM | Data reuse, arithmetic intensity, roofline, shared memory and register tiling |
| [11_naive_attention](course/11_naive_attention/README.md) | Standard attention | Attention math, causal mask, LSE, HBM traffic counting |
| [12_fa1](course/12_fa1/README.md) | FlashAttention-1 | Correctness proof of tiling + online softmax, Algorithm 1 of the paper |
| **Part 2 · FA2: Tensor Cores and work partitioning** | | |
| [20_mma_gemm](course/20_mma_gemm/README.md) | `mma.sync` GEMM | Warp-level Tensor Core, fragment layout, `ldmatrix`, XOR swizzle |
| [21_pipeline_gemm](course/21_pipeline_gemm/README.md) | Async copy pipeline | `cp.async`, commit/wait group, multi-stage buffers |
| [22_fa2](course/22_fa2/README.md) | FlashAttention-2 | Loop swap, P stays in registers, causal block skipping |
| **Part 3 · FA3 ideas on SM100: asynchrony and specialization** | | |
| [30_tma_mbarrier](course/30_tma_mbarrier/README.md) | TMA transpose | tensor map, TMA load/store, mbarrier phase, ring buffer |
| [31_tcgen05_gemm](course/31_tcgen05_gemm/README.md) | `tcgen05` GEMM | Tensor Memory (TMEM), smem/instruction descriptors, `tcgen05.mma/commit/ld` |
| [32_ws_gemm](course/32_ws_gemm/README.md) | warp specialization | Producer/consumer split, TMEM double buffering, persistent kernel, L2 rasterization |
| [33_2cta_gemm](course/33_2cta_gemm/README.md) | 2-CTA MMA | cluster, distributed shared memory, `cta_group::2`, TMA multicast |
| [34_fa3](course/34_fa3/README.md) | FA3-style attention | S/O in TMEM, overlapping softmax with MMA |
| **Part 4 · FA4: full optimization on Blackwell** | | |
| [40_fa4_pipeline](course/40_fa4_pipeline/README.md) | FA4 pipeline structure | Two Q tiles in ping-pong, P written back to TMEM, correction warpgroup |
| [41_fa4_softmax](course/41_fa4_softmax/README.md) | Softmax throughput | MUFU bottleneck, polynomial `exp2`, conditional rescale |
| [42_fa4_2cta](course/42_fa4_2cta/README.md) | 2-CTA attention | 2-CTA split of QKᵀ/PV, cluster synchronization |
| [43_fa4_schedule](course/43_fa4_schedule/README.md) | Scheduling and causal | Block classification, persistent, LPT ordering |
| [44_fa4_fp8](course/44_fa4_fp8/README.md) | FP8 forward | E4M3 quantization, descale, `kind::f8f6f4` |

Work through the labs in order. Each lesson lists its prerequisite labs at the top. The design of Part 4 is in [design doc §3](docs/design-v2.md#3-course-map).

## How to work through a lab

Every lab has the same structure: a lesson, a skeleton with `TODO`s, and a set of checks split by step.

```
read lesson ──► fill TODO(n) ──► ./lab check ID --step n ──► ./lab bench ID --step n ──► sass / profile / sanitize
 §1–§5           §6 exercises     correctness (small cases    speed vs solution and      explain why it is
                                  fail first)                 libraries                  fast or slow
                       ▲                                                                               │
                       └───────────────────────────────── next step ◄──────────────────────────────────┘
```

| Stage | Where | Notes |
|---|---|---|
| Lesson | `course/<ID>/README.md` | Nine sections: goals, motivation, math, hardware and instructions, schedule design, step-by-step exercises, performance, self-check, further reading |
| Skeleton | `students/<ID>/kernel.cu` | The only file you change. Parts outside the focus, such as the launch configuration, are already written; each `TODO(n)` states its inputs, outputs and invariants |
| Correctness | `./lab check ID --step n` | Compares against an FP64 math oracle. The output is first filled with NaN, so missing writes are caught; the kernel also runs a second time on a non-default stream |
| Performance | `./lab bench ID --step n` | With the same input and the same timing method, compares your kernel, the reference solution and libraries (cuBLAS, cuDNN, FA4) |
| Analysis | `./lab sass` / `profile` / `sanitize` | Registers and instructions, Nsight Compute metrics, Compute Sanitizer. See the [Tools guide](docs/tools.md) |
| Reference solution | `solutions/<ID>/` | A readable CUDA C++/PTX implementation and the design notes `NOTES.md`. Read it when you are stuck; add `--solution` to run it |

All commands:

```
./lab list                                     course list and progress
./lab check ID [--step N] [--case NAME]        correctness
./lab bench ID [--step N] [--case NAME] [--cold]   timing
./lab sass ID                                  registers, spills, instruction counts
./lab profile ID [--step N] [--case NAME]      Nsight Compute
./lab sanitize ID --case NAME [--tool memcheck|racecheck|synccheck|initcheck]
./lab doctor                                   environment check
./lab clean [--yes]                            delete build cache and bench/profile/sass output (keeps students/ and progress)
```

Add `--solution` after check, bench, sass, profile or sanitize to run the reference solution instead. An ID can be any unique prefix, for example `00` or `31_tc`.

## Repository layout

```
course/<ID>/          lesson README.md and spec.json (cases, steps, perf targets)
students/<ID>/        your exercise: kernel.cu
solutions/<ID>/       reference solution: kernel.cu (and .cuh) and NOTES.md
include/lab/          shared headers: interface, error checks, and primitives written in earlier labs (mma_sync, tma, tcgen05…)
labkit/               Python infrastructure: input generation, kernel calls, answer checks, timing
third_party/          pinned CUTLASS and FlashAttention sources, for side-by-side reading
docs/                 project-wide docs
scripts/              bootstrap.sh (install), activate.sh (activate environment)
tests/                infrastructure tests
artifacts/  .cache/  progress/              bench/profile results, build cache, progress (ignored by Git)
```

The primitives in `include/lab/` follow "what you wrote earlier is provided later": for example, you implement the `ldmatrix` and `mma.sync` wrappers yourself in lab 20, and from lab 21 on you `#include "lab/mma_sync.cuh"` directly.

## Docs

| Doc | Contents |
|---|---|
| [Starting from zero](docs/starting-from-zero.md) | Writing CUDA for the first time: threads, addresses, the memory hierarchy, and step 1 of the first lab |
| [B200 architecture overview](docs/b200.md) | Numbers for SM, registers, shared memory, L2, HBM, Tensor Core, TMEM, TMA, cluster, and which lab covers each |
| [Tools guide](docs/tools.md) | `./lab sass/profile/sanitize`, SASS instructions, timing modes and the `vs` column |
| [Glossary](docs/glossary.md) | The course glossary, with links to the lab that introduces each term |
| [Attention math](docs/attention-math.md) | Attention, online softmax, IO complexity, numerical checks |
| [Environment and dependencies](docs/environment.md) | Install flags, `LAB_GPU`, CUDA versions, the FA4 baseline |
| [Upstream sources](docs/sources.md) | Papers, pinned upstream sources |
| [Code formatting](docs/formatting.md) | clang-format, Ruff, pre-commit |
| [Design doc](docs/design-v2.md), [Writing labs](docs/authoring.md) | For maintainers |
