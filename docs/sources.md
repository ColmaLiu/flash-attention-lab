# Upstream sources and pinned versions

All of the course's reference solutions are readable CUDA C++ / PTX in this repository; they do not call code from CUTLASS, CuTe or FlashAttention. The upstream sources are only for **side-by-side reading**: to see how production implementations handle the same problem.

## 1. Papers

| Paper | Pinned version | Matching labs |
|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135v2) | arXiv:2205.14135v2 | 11, 12: IO analysis, tiling, online softmax |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691v1) | arXiv:2307.08691v1 | 22: loop interchange, warp partitioning, fewer non-matmul operations |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608v2) | arXiv:2407.08608v2 | 34: asynchronous producers, GEMM/softmax overlap, pingpong |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451v1) | arXiv:2603.05451v1 | Part 4: Blackwell pipeline, exponentials, rescale, 2-CTA, scheduling |

## 2. Upstream sources (Git submodules)

The versions of the two submodules are locked by gitlinks in the parent repository. `bootstrap.sh` initializes them; you can also run this by hand:

```bash
git submodule update --init -- third_party/cutlass third_party/flash-attention
git ls-files --stage third_party/cutlass third_party/flash-attention   # show the locked commits
```

| Path | Version | License | Use |
|---|---|---|---|
| `third_party/cutlass` | v4.4.1 / `4370102f9dacab813282e1d67722fceb0b90a019` | BSD-3-Clause (`LICENSE.txt`) | SM100 instruction wrappers, descriptor bit fields, pipeline classes, Blackwell GEMM and FMHA examples |
| `third_party/flash-attention` | fa4-v4.0.0.beta35 / `1fc0221183e799363001394a12dd9e78e15f5dda` | BSD-3-Clause (`LICENSE`) | The SM100 forward implementation of FA4 (CuTe DSL); `--with-fa4` installs the performance baseline from here |

The "Further reading" section of each lesson names specific files. Common entry points:

| File | Contents | Matching labs |
|---|---|---|
| `third_party/cutlass/include/cute/arch/mma_sm100_desc.hpp` | Bit fields of the smem descriptor and instruction descriptor | 31 |
| `third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp` | Inline PTX of `tcgen05.mma` | 31, 33 |
| `third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp`, `copy_sm100_tma.hpp` | TMA instructions | 30, 33 |
| `third_party/cutlass/include/cutlass/arch/barrier.h` | mbarrier, cluster barrier | 30, 33 |
| `third_party/cutlass/examples/77_blackwell_fmha/` | CUTLASS SM100 FMHA example | 34, Part 4 |
| `third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py` | FA4 SM100 forward kernel | 34, Part 4 |
| `third_party/flash-attention/flash_attn/cute/interface.py` | FA4 Python interface (`flash_attn_func`) | The `fa4` baseline in `bench` |

Parts of this repository's solutions that are adapted directly from upstream cite their source in code comments; the licenses are kept with the submodules.

## 3. Other pinned sources

[sources.lock.json](sources.lock.json) records the paper versions and the commits and sha256 of upstream files in a machine-readable format. It also includes two historical versions that are not in the submodules:

| Source | Pinned version | Use |
|---|---|---|
| [FA1 implementation](https://github.com/Dao-AILab/flash-attention/tree/6d48e14a6c2f551db96f0badc658a6279a929df3) | v1.0.9 | The original CUDA implementation of FA1 |
| [FA2 forward kernel](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/csrc/flash_attn/src/flash_fwd_kernel.h) | v2.7.4.post1 | FA2's parallel partitioning, pipeline and numerics |

## 4. NVIDIA documentation

The online docs change over time. For instruction usage, the authority is PTX ISA 8.7, which matches the CUDA 12.8 the course uses; the wrappers in `include/lab/*.cuh` name the PTX ISA 8.7 sections.

| Doc | Use |
|---|---|
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | Instruction semantics, legal operands, memory model |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | Programming model, clusters, asynchronous copies, resource limits per compute capability |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Coalesced access, bank conflicts, instruction throughput tables |
| [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | SM100 resources and tuning advice |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Definitions of performance metrics |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | memcheck, racecheck, synccheck, initcheck |
| [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | `cuobjdump`, `nvdisasm` |
