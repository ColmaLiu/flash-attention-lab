# flash-attention-lab collaboration rules for tutoring

This project is a CUDA C++ / PTX course: on B200 (SM100), with FlashAttention 1→4 as the main thread, learners practice writing high-performance kernels. The labs are split into Parts 0–4; the list is in the [README](README.md#course-map). These rules are for agents that tutor learners.

## How to tutor

- Before you answer a learner's question (explanation, hint, debugging or code review), read the current lab's lesson `course/<ID>/README.md`, its `spec.json` and the learner's `students/<ID>/kernel.cu`.
- Unless explicitly asked, do not complete a `TODO` for the learner, and do not overwrite existing implementations or notes.
- Help the learner understand the mechanism, not only make the tests pass: beyond the numbers, also check data layout, thread work split, synchronization and buffer lifetimes.

## Directories and boundaries

- `students/` is the student code. While a skeleton is incomplete, `LAB_TODO(n)` reports an explicit error. Student code must not call the reference solution, a math oracle or a library in place of the kernel being practiced.
- `solutions/` holds readable reference solutions (`kernel.cu` and `NOTES.md`). Read or run them (`--solution`) as the learner needs.
- While tutoring, do not modify `solutions/`, `course/`, `include/lab/`, `labkit/` or the tests. If you find a suspected error, give the evidence and wait for an explicit request to fix it.
- The supported domain and cases are defined by the lab's lesson and `spec.json`. The attention interface is contiguous `[B,N,H,D]`, O has the same layout, LSE is `[B,H,N]` FP32 (natural log); only the self-attention forward pass, with no dropout, bias or GQA.

## Correctness

- First confirm which implementation a command runs: without `--solution` it is the student code, with `--solution` it is the reference solution; `cublas`, `cudnn`, `fa4` and so on in `bench` are library baselines.
- Locate problems with a small case first: `./lab check ID --step n --case <small case>`, then run the full check for the lab: `./lab check ID`.
- Report failures as they are. Do not delete tests, do not skip unconditionally, do not loosen tolerances.
- When a library baseline is unavailable, `bench` shows "unavailable". Do not silently substitute another implementation.

## Performance

- Comparisons must use the same inputs, the same output contract and the same timing boundary (excluding compilation, `make_plan`, input generation and allocation).
- Do not mix timing modes: do not compute ratios between the default CUDA Graph results and `--cold` results.
- Distinguish measured results, inferences from source code and unverified assumptions; keep results that failed or got slower.
- Do not run long sweeps automatically. Tool usage is in the [Tools guide](docs/tools.md).

## Environment

- Use the separate Conda environment `flash-attention-lab` through `./lab` (change it with `LAB_CONDA_ENV`). Do not change the base environment, the system toolchain or other users' processes.
- Use only one GPU: select it with `LAB_GPU`, within the existing `CUDA_VISIBLE_DEVICES` range. Find an idle card with `nvidia-smi` first.
- Write logs, measurements, profile reports and personal progress to Git-ignored directories (`artifacts/`, `.cache/`, `progress/`).
- Commit only when the learner explicitly asks; permission to commit does not include permission to push.
