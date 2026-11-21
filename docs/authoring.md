# Writing labs (maintainer guide)

This file specifies which files make up a lab and what each must satisfy. The design background is in [design-v2.md](design-v2.md); the model lab is [00_transpose](../course/00_transpose/README.md), and new labs follow its structure and level of detail.

## 1. Files

```
course/<ID>/README.md     lesson
course/<ID>/spec.json     cases, steps, perf targets
students/<ID>/kernel.cu   student skeleton
solutions/<ID>/kernel.cu  reference solution (may #include .cuh files in the same directory)
solutions/<ID>/NOTES.md   design notes for the solution
include/lab/<name>.cuh    (only labs that introduce a new primitive) reference implementation of the shared primitive
```

`labkit/` and `include/lab/{lab,problem,register,check,scratch}.h` are shared infrastructure. Change them only when a lab really needs it, for example to add an input pattern or a probe oracle in `labkit/kinds/<kind>.py`; keep changes small and backward compatible.

## 2. spec.json

```json
{
  "id": "00_transpose",
  "title": "Memory access and thread mapping: matrix transpose",
  "part": 0,
  "kind": "transpose",
  "prerequisites": [],
  "steps": [
    {"step": 1, "title": "Naive transpose", "cases": "all"},
    {"step": 2, "title": "Single tile", "cases": ["tiny", "one_tile"]},
    {"step": 3, "title": "Check only the intermediate result", "cases": ["tiny"], "probe": {"oracle": "scores"}}
  ],
  "cases": [
    {"name": "tiny", "shape": [7, 11], "dtype": "fp32", "pattern": "coordinate",
     "purpose": "which kind of bug this case is meant to catch"},
    {"name": "big", "shape": [8192, 8192], "dtype": "fp32", "perf": true, "purpose": "..."}
  ],
  "perf_target": {"baseline": "torch", "ratio": 0.9},
  "profile_metrics": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
}
```

- **kind** selects the Problem struct and the oracle: `transpose`, `softmax`, `gemm`, `attention`, defined in `include/lab/problem.h` and `labkit/kinds/`.
- **shape**: `[rows, cols]` for transpose/softmax, `[M, N, K]` for gemm, `[B, N, H, D]` for attention.
- **Other case fields** go unchanged into `case.options`, for example `causal` and `scale` for attention, `c_dtype` for gemm. Attention can also use `o_tol` / `lse_tol` (`[atol, rtol]`) to **tighten** the tolerance for structured inputs (never to loosen it).
- **pattern** is interpreted by the kind's `make_inputs`. Existing patterns:
  - transpose: `random`, `coordinate`;
  - softmax: `random`, `large`, `shifted`, `constant`, `masked`;
  - gemm: `random`, `int`;
  - attention: `random`, `large_logits`, `uniform`, `ones_v`, `position`, `growing_max`, `jump_max` (the last two make each row's maximum grow slowly or jump suddenly along the keys; see lab 41).
- **steps**: `"cases": "all"` means all non-perf cases. While a step runs, `p.step` equals that step's number. bench and check without `--step` both use the last step, and the last step must not have a probe.
- **probe**: the step does not check the final outputs such as O; it checks `p.probe` (an FP32 buffer) instead. The oracle is defined by `probe_<oracle>(inputs, options, case)` in `labkit/kinds/<kind>.py`, which returns `(expected tensor, (atol, rtol))`. The optional `"axes": ["b", "h", "i", "j"]` gives the dimension names used in error messages.
- **Case design**:
  - Every case states its `purpose`, that is, which kind of bug it targets;
  - Cases must cover the tail (sizes that are not a multiple of the tile), B and H both greater than 1, causal/non-causal, every dtype and every D;
  - Each lab has 1–3 perf cases (attention in Part 4 may have up to 5, covering non-causal, causal, D=64, long sequences and small B·H scheduling). They must be large enough to expose the HBM or Tensor Core bottleneck: inputs larger than L2 (126 MB), or for attention N ≥ 4096 and B·H ≥ 16.
- **perf_target**: the minimum ratio of the solution relative to this baseline over all perf cases (`baseline time / solution time`), checked by `./lab selftest --perf`. Intermediate labs set a value that shows the mechanism works; the final lab of each operator class sets 0.9, namely transpose `30_tma_mbarrier`, softmax `01_softmax`, GEMM `33_2cta_gemm`, attention `43_fa4_schedule`.

## 3. Student skeleton

- It compiles. The missing part of each step is marked with `LAB_TODO(n)`, with n equal to the step number where possible. A `LAB_TODO` in device code makes the kernel trap, and the harness reports "TODO(n) is not implemented yet".
- No deadlocks: mbarrier waits, producer/consumer loops and the like must trap before they reach a TODO, or the host-side code holding the TODO must prevent the launch.
- Code outside the lab's focus is already written: the launch configuration, host-side descriptor encoding, primitives learned in earlier labs.
- The comment before each TODO states: what the inputs are, where the output goes, the invariants, and which helpers may be called. Do not write out the key line of the answer.
- `run()` selects a version by `p.step`, so the learner keeps the implementation of every step and can compare their performance. If the steps complete the same kernel one piece at a time, no branching is needed.
- `#include` only the `include/lab` headers already opened up for this lab (see the table in design doc §5). In a lab that introduces a new primitive, the skeleton gives the primitive's function signature and a comment on its semantics, and the learner implements the function body.
- File header: one line of description plus the lesson path, then the data layout conventions.

## 4. Reference solution

- Use only readable CUDA C++ and inline PTX from this repository; do not substitute CUTLASS/CuTe/library wrappers for the solution.
- Implement only the mechanisms this lab introduces; do not include optimizations from later labs ahead of time. Copying code from the previous lab's solution as a starting point is fine.
- Every step runs through `p.step` and passes all cases of that step.
- Comments explain "why": the thread work split, which memory level holds the data, what each synchronization protects.
- Verify all three: correctness, no data races, performance (see section 6).
- `NOTES.md` covers: design choices and their reasons, the performance change at each step and its cause, known limitations. Do not write absolute times from your machine; ratios are fine.
- A lab that introduces a new primitive puts it in `include/lab/<name>.cuh`, and the solution `#include`s it.
  - Each function's comment states the PTX semantics and gives the PTX ISA section name;
  - Namespace `lab`, function names in snake_case;
  - A function wraps only a single instruction or a fixed group of instructions, with no algorithm logic.

## 5. Lesson (README.md)

In English, following the nine-section structure of lab 00:

1. **Goals**: what the lab does and what you can do after it;
2. **Motivation**: the bottleneck of the previous lab, and which section of the FA paper it matches;
3. **Math**: derivations, invariants, small examples worked by hand;
4. **Hardware and instructions**: semantics of the new instructions, operand layouts, minimal code snippets, common misuses;
5. **Schedule design**: thread/warp work split, the budget for each memory level, timelines;
6. **Step-by-step exercises**: the TODO, commands and common-mistakes table for each step;
7. **Performance**: measured ratios of each step relative to the baseline, which profile metrics to look at;
8. **Self-check questions**: 3–6 questions, with answers in `<details>`;
9. **Further reading**: paper sections, PTX ISA / CUDA doc links, upstream source locations (file paths in `third_party/`).

Writing rules:

- Define before use, one idea per sentence, no piles of qualifiers;
- Use plenty of tables, ASCII diagrams and small examples worked by hand; numbers must match the output of `./lab`;
- Diagrams and tables inside code fences use only characters one column wide (no CJK, no full-width punctuation), and they must line up by character count;
- When you cite B200 hardware data, give concrete values (148 SMs, 228 KB shared memory per SM, TMEM of 128 rows × 512 columns × 32 bit per SM, 126 MB L2, 8 TB/s nominal HBM, and so on); verify or measure any value you are unsure of first;
- Do not describe maintainer workflows; the lesson is for learners only.

## 6. Verification

Use one idle GPU: confirm with `nvidia-smi` first, then set `LAB_GPU=<n>`.

```bash
LAB_GPU=n ./lab check ID --solution                 # all steps, all cases
LAB_GPU=n ./lab selftest ID                         # solution passes + skeleton stops at the TODO
LAB_GPU=n ./lab sanitize ID --solution --case <small case> --tool memcheck
LAB_GPU=n ./lab sanitize ID --solution --case <small case> --tool racecheck
LAB_GPU=n ./lab sanitize ID --solution --case <small case> --tool synccheck
LAB_GPU=n ./lab bench ID --solution --step k        # every step
LAB_GPU=n ./lab sass ID --solution                  # registers, spills, instructions
LAB_GPU=n ./lab selftest ID --perf                  # when there is a perf_target
python -m pytest -q                                 # in the scripts/activate.sh environment
```

Also confirm by hand:

- After you replace each TODO in the skeleton with the solution code, the matching step passes. Do this in a temporary copy, and restore the skeleton afterwards.
- When you run step by step with `./lab check ID --step n`, step n depends only on the first n TODOs.
- Do not run long sweeps, and do not occupy several GPUs.
