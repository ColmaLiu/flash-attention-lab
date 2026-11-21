# flash-attention-lab v2 design

Status: design doc; tracking issues carry the GitHub `v2` label. This file describes the goals, course structure, repository organization, interfaces and acceptance criteria of v2; if something changes during implementation, change this file first, then the code.

## 1. Goals

Let learners with **only a little CUDA background** (they have written an elementwise kernel and know grid/block/thread and shared memory) learn to write and optimize high-performance kernels on **B200 (SM100)**, with **FlashAttention 1→4** as the main thread.

After the course, learners should be able to:

1. Derive the IO complexity of attention from its math definition, and explain which bottleneck each generation FA1→FA4 removes;
2. Write correct code for mma.sync, TMA, mbarrier, tcgen05/TMEM, warp specialization and 2-CTA MMA on their own, and explain what each synchronization does;
3. Verify optimization hypotheses with SASS, Nsight Compute and controlled timing, rather than tuning by feel;
4. Read the source structure of upstream FA4 (CuTe DSL) and the CUTLASS SM100 FMHA.

Non-goals: Hopper wgmma; backward pass; decode/split-KV; GQA/MQA, dropout, bias, variable-length sequences.

## 2. Design principles

1. **Understand first, then write.** Each lesson covers all the new knowledge of its lab: math, scheduling, instructions. Learners do not need to read the papers or the PTX manual before they start; the lesson gives references for going deeper.
2. **Small, verifiable steps.** Each lab is split into 2–5 steps; each step matches a `TODO(n)` in the skeleton and can be verified on its own with `./lab check ID --step n`. Learners never face a black box where "the whole lab passes or fails at once".
3. **One new mechanism per lab.** A new hardware mechanism is first practiced on its own on a GEMM (or a simpler operator), then used in attention. The reference solution contains only the features this lab introduces and does not leak later labs early.
4. **What you wrote earlier is provided later.** Primitives that a learner wrote by hand in one lab (such as the mbarrier wrappers and TMA descriptor encoding) are provided as reference implementations in the shared headers `include/lab/` from the next lab on, so that small mistakes early on do not wreck later experiments.
5. **Reproducible performance conclusions.** Timing excludes compilation and preparation; CUDA Graph and eager results are never mixed; each lab gives the learner's goal as "a ratio relative to a library baseline", and the repository does not commit measurements from our machine; maintainer acceptance is kept separate from learner output.
6. **Simple infrastructure.** Learners only need to understand the two functions `make_plan` / `run` in `kernel.cu`; the Python side only generates data, calls the kernel, checks answers and times.

## 3. Course map

IDs follow `<part><index>_<name>`. Part 0 is preliminaries, and 1–4 correspond to FA1–FA4.

| ID | Topic | New knowledge | Exercises (step outline) | Baseline |
|---|---|---|---|---|
| **Part 0: Preliminaries** |||||
| `00_transpose` | Memory access and thread mapping | Linear addresses, coalescing, shared memory, bank conflicts, padding/swizzle | Naive transpose → smem tiling → remove bank conflicts → larger tiles for more memory accesses in flight | torch copy |
| `01_softmax` | Reductions and numerical stability | warp shuffle, block reduction, safe softmax, derivation of **online softmax** | Two-pass softmax → single-pass online softmax | torch.softmax |
| **Part 1: FA1 — IO-aware tiling** |||||
| `10_simt_gemm` | Tiled GEMM | Data reuse, arithmetic intensity, smem tiling, register tiling, vectorization | Naive → smem tile → register thread tile | cuBLAS (FP32) |
| `11_naive_attention` | Standard attention and IO analysis | Attention math, mask/scale, HBM traffic counting | Compose the provided GEMMs: S, softmax with mask, PV; measure and explain the N² traffic | Math oracle |
| `12_fa1` | FlashAttention-1 | Correctness proof of tiling + online softmax; Algorithm 1 of the paper (KV in the outer loop, O/ℓ/m written back to HBM); LSE | Single block correct → full KV loop → causal | Lab 11 implementation, cuDNN |
| **Part 2: FA2 — Tensor Cores and work partitioning** |||||
| `20_mma_gemm` | Warp-level Tensor Core | `mma.sync.m16n8k16` fragment layout, `ldmatrix`, XOR swizzle | Single warp 16×8×16 → CTA tile → conflict-free ldmatrix | cuBLAS (BF16) |
| `21_pipeline_gemm` | Async copy pipeline | `cp.async`, commit/wait_group, multi-stage buffers, latency hiding | Double buffering → multi-stage pipeline → tuning and analysis | cuBLAS |
| `22_fa2` | FlashAttention-2 | Loop interchange (Q outer), deferred normalization, warps split by Q, P stays in registers as the A operand, causal block skipping | QKᵀ fragment → online softmax in registers → PV → causal | cuDNN |
| **Part 3: FA3 ideas on SM100 — asynchrony and specialization** |||||
| `30_tma_mbarrier` | TMA and mbarrier | tensor map, swizzle modes, `cp.async.bulk.tensor`, expect_tx, phase/parity, TMA store | Single-tile copy → multi-stage ring buffer (full/empty barriers) | torch copy |
| `31_tcgen05_gemm` | tcgen05 and TMEM | TMEM alloc/free, smem descriptor, instruction descriptor, `tcgen05.mma/commit/ld` | Single tile → full K loop (already covering all tiles) → TMA store epilogue and tile shape | cuBLAS |
| `32_ws_gemm` | warp specialization | Producer/consumer, TMA warp + MMA warp + epilogue warps, TMEM double buffering, persistent | Specialized pipeline → accumulator double buffering → persistent scheduling | cuBLAS |
| `33_2cta_gemm` | 2-CTA MMA | cluster, distributed smem, `cta_group::2`, TMA multicast | 2-CTA single tile → full pipeline | cuBLAS |
| `34_fa3` | FA3-style attention | How the three FA3 ideas (asynchronous producers, GEMM/softmax overlap, pingpong) map onto tcgen05; S/O in TMEM, P through smem | Single KV block → full loop → S double buffering to overlap QKᵀ and softmax | cuDNN |
| **Part 4: FA4 — full optimization on Blackwell** |||||
| `40_fa4_pipeline` | FA4 pipeline structure | 2 Q tiles in ping-pong, P aliased and written back to TMEM (TS-MMA), separate correction warpgroup, register reallocation | Role split → P in TMEM → correction WG | cuDNN, FA4 |
| `41_fa4_softmax` | Softmax throughput | MUFU bottleneck analysis, software-emulated polynomial exp2, conditional rescale and its error bound | Polynomial exp2 → mixed MUFU/FMA → conditional rescale | Lab 40, FA4 |
| `42_fa4_2cta` | 2-CTA attention | 2-CTA split of QKᵀ/PV, cluster synchronization | 2-CTA QKᵀ → 2-CTA PV | FA4 |
| `43_fa4_schedule` | Scheduling and causal | Causal block classification and masking overhead, persistent, LPT ordering | causal → persistent → LPT | cuDNN, FA4 |
| `44_fa4_fp8` | FP8 forward | E4M3 quantization, descale, FP8 quantization error of P, `kind::f8f6f4` | FP8 QKᵀ → FP8 PV → error analysis | FA4 FP8 (if supported) |

Supported domain (unless a lab states otherwise):

- Parts 0–1: FP32;
- attention: `[B,N,H,D]` contiguous BF16/FP16, `D ∈ {64,128}`, `scale > 0` and finite, causal/non-causal, O in the same layout, LSE `[B,H,N]` FP32 (natural log);
- FP8 lab: E4M3 inputs plus FP32 descale, O in BF16.

Each lab declares its own dtype and shape constraints in `spec.json`; both the binding and the Python side check the supported domain, and out-of-domain inputs report an explicit error.

## 4. What each lab contains

```
course/<ID>/
  README.md     lesson (see the template below)
  spec.json     cases, steps, supported domain, perf cases
  figures/      figures for the lesson (SVG/mermaid sources)
students/<ID>/kernel.cu      student skeleton (with TODO(n)); the learner edits it directly
solutions/<ID>/kernel.cu     reference solution (may be split into several .cuh)
solutions/<ID>/NOTES.md      solution design notes: why the work is split this way, how it is verified, known limitations (no measurements from our machine)
```

Lesson template:

1. **Goals**: one paragraph, plus the prerequisite labs;
2. **Motivation**: what the bottleneck of the previous lab is, and which section of the FA paper it matches;
3. **Math**: derivations, invariants, small examples worked by hand;
4. **Hardware and instructions**: semantics of this lab's new instructions, operand layouts, minimal code snippets, common misuses;
5. **Schedule design**: thread/warp work split, smem/TMEM/register budget, timelines;
6. **Step-by-step exercises**: for each step, which TODO to implement, which command to run, common mistakes and how to debug them;
7. **Performance**: which cases to measure, the target ratio relative to the library baseline, which SASS/NCU metrics to look at;
8. **Self-check questions**: 3–6, with answers collapsed at the end of the lesson;
9. **Further reading**: paper sections, PTX manual sections, upstream source locations (pinned versions).

Skeleton rules:

- The skeleton compiles; the parts "outside this lab's focus", such as `make_plan` / `run`, the launch configuration and host-side descriptor encoding, are already written;
- The comment on each `TODO(n)` states the inputs, outputs, invariants and the helpers that may be called;
- When unfinished, running to a TODO reports an explicit error (`LAB_TODO(n)`) instead of silently producing garbage.

## 5. C++ interface

Each operator has a Problem struct, defined in `include/lab/problem.h`:

```cpp
namespace lab {
struct AttentionProblem {
  const void *q, *k, *v;  // [B,N,H,D]
  void *o;                // [B,N,H,D]
  float *lse;             // [B,H,N]
  int b, n, h, d;
  DType dtype;
  float scale;
  bool causal;
  float q_descale, k_descale, v_descale;  // FP8 only
  float *probe;           // step debug output, may be null
};
}  // namespace lab
```

The learner's `kernel.cu` implements only two functions and a Plan type:

```cpp
struct Plan { /* TMA descriptors, scratch, launch parameters */ };
Plan make_plan(const lab::AttentionProblem& p);                          // not timed: encode descriptors, allocate scratch
void run(const Plan& plan, const lab::AttentionProblem& p, cudaStream_t s);  // timed: only launches kernels
LAB_REGISTER(lab::AttentionProblem, Plan, make_plan, run);
```

- `LAB_REGISTER` generates a type-erased entry point, called by the TVM FFI binding for each Problem kind in `labkit`;
- The scratch in a Plan is allocated with `lab::Scratch` (RAII), which fills it with garbage on allocation, forcing learners to initialize it themselves;
- In early labs the Plan is an empty struct and the skeleton is already written, so learners meet the "prepare/execute split" from the first lab.

GEMM is unified as `C[M,N] = A[M,K] · B[N,K]ᵀ`: A and B are both K-major, matching the shape of QKᵀ; the MN-major B operand in PV is explained in the attention labs.

Shared headers in `include/lab/`:

| Header | Contents | Learner implements it in → provided from lab |
|---|---|---|
| `problem.h`, `register.h`, `check.h` | Problem, registration macro, CUDA error checks, `LAB_TODO` | 00 |
| `scratch.h` | scratch allocation with garbage initialization | 00 |
| `mma_sync.cuh` | `ldmatrix`, `mma.sync` wrappers, fragment coordinate functions | 20 → 21 |
| `cp_async.cuh` | `cp.async` and wait_group | 21 → 22 |
| `tma.cuh`, `mbarrier.cuh` | tensor map encoding, TMA load/store, mbarrier | 30 → 31 |
| `tcgen05.cuh` | TMEM alloc/dealloc/relinquish, `tcgen05.ld`/`st` (32x32b, including `_sync` versions that combine ld+wait) and wait, smem/instruction descriptors, mma/commit, tcgen05 fences | 31 → 32 |
| `cluster.cuh` | cluster and 2-CTA helpers | 33 → 34 |

For example `mma_sync.cuh`: in the lab 20 skeleton the learner implements these functions (the skeleton gives the signatures and semantic comments), and the lab 20 reference solution `#include`s it; from lab 21 on, the skeleton `#include`s it directly.

## 6. Python infrastructure `labkit`

Commands:

| Command | What it does |
|---|---|
| `./lab list` / `./lab status` | Course list and progress per lab and per step |
| `./lab check ID [--step n] [--case name]` | Compiles the student code and compares against the oracle; runs all correctness cases by default |
| `./lab bench ID [--case name] [--cold]` | Timing: student, solution and library baselines, same input, CUDA Graph |
| `--solution` on check/bench/sass/profile/sanitize | Runs the reference solution instead of the student code |
| `./lab sass ID` | Registers, spills, smem usage and a SASS excerpt of the build output |
| `./lab profile ID --case name` | Captures the lab's preset NCU metrics |
| `./lab sanitize ID --tool memcheck/racecheck/synccheck` | Compute Sanitizer |
| `./lab doctor` | Environment check |
| `./lab selftest` | Maintainers: all solutions pass, skeletons compile and report their TODOs explicitly, infrastructure unit tests |

Correctness checks:

- Math oracle: PyTorch FP64 (GEMM/attention computed in chunks to limit GPU memory);
- Tolerances are given per dtype and reduction length, defined by `tolerance()` in `labkit/kinds/<kind>.py`, and explained in the lessons;
- Outputs and scratch are first filled with NaN or garbage; inputs are checked for modification; the kernel runs once on the default stream and once on a non-default stream;
- Steps are defined by a subset of cases in `spec.json`, the `probe` output and its oracle.

Timing:

- CUDA Graph replay by default, eager optional; the two are never mixed;
- `--cold` flushes L2 between iterations;
- Reports the median and the distribution, TFLOP/s or GB/s, the ratio to the B200 nominal peak, and the ratios to the solution and the library baselines;
- Library baselines: cuBLAS (through torch), cuDNN SDPA (forced backend; reports an error if it fails), FA4 (`flash_attn.cute`, optional install; shows "unavailable" if not installed);
- The maintainers' perf gate for solutions is checked only in `selftest --perf` and never appears in learner output.

Progress: `progress/<ID>.json` records the pass status of each step and the source fingerprint of the lab. The fingerprint covers only `students/<ID>/` and the `include/lab/` files it includes.

## 7. Quality assurance

Before a lab merges into `v2`, it must satisfy:

1. The solution passes all cases (including `--cold` and the non-default stream), and the `compute-sanitizer` memcheck, racecheck and synccheck report no errors;
2. The skeleton compiles, and each step reports `LAB_TODO(n)` while unimplemented;
3. The lesson contains all nine sections of the template, its commands run as written, and its links are valid;
4. The solution's performance reaches the target ratio (relative to the library baseline) declared in `spec.json`; the final solution of each operator class (transpose: `30_tma_mbarrier`, softmax: `01_softmax`, GEMM: `33_2cta_gemm`, attention: `43_fa4_schedule`) reaches 90% of the library baseline on **every** perf case of its lab, case by case (medians from the same CUDA Graph run), with no averaging and no filtering; measurement reports are written only to the Git-ignored `artifacts/`;
5. There is one independent review (by a reviewer agent that did not take part in writing it), with the conclusion recorded in the matching issue.

## 8. Migration from v1

- v1's `flash_attention_lab/`, `students/cuda/`, `solutions/s*`, `docs/tasks/` and `curriculum.json` are replaced as a whole in v2;
- Kernel code from v1 that was verified correct (such as the TMA, tcgen05 and softmax implementations of s40/s43) is the starting point for the v2 solutions, split up by "one mechanism per lab", with known issues fixed (NaN for scale ≤ 0, extreme-scale branches, duplicated PTX wrappers);
- The `third_party/` submodules stay, for side-by-side source reading;
- v1 stays on the `main` branch; how to merge is discussed once v2 is stable.

## 9. Milestones

| Milestone | Contents |
|---|---|
| M0 | This design doc, splitting into issues |
| M1 | `labkit`, the C++ interface, the base headers in `include/lab`; `00_transpose` working end to end |
| M2 | Parts 0–1 (00, 01, 10, 11, 12) |
| M3 | Part 2 (20, 21, 22) |
| M4 | Part 3 (30–34) |
| M5 | Part 4 (40–44) |
| M6 | Project-wide docs (getting started, B200 architecture overview, tools guide, glossary), selftest, full review, v1 cleanup |
