# 完整参考实现与源码阅读

`solutions/` 提供可执行答案和实验配置，上游完整代码通过 `third_party/` 下的 Git submodule 引入。父仓库 gitlinks 固定版本，克隆与初始化方式如下：

```bash
git clone https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
git submodule update --init -- third_party/cutlass third_party/flash-attention
bash scripts/bootstrap.sh
```

| 上游 | 路径 | gitlink 对应 commit | 许可证 |
|---|---|---|---|
| NVIDIA CUTLASS v4.4.1 | `third_party/cutlass` | `4370102f9dacab813282e1d67722fceb0b90a019` | [BSD-3-Clause](../third_party/cutlass/LICENSE.txt) |
| Dao-AILab FlashAttention beta35 | `third_party/flash-attention` | `1fc0221183e799363001394a12dd9e78e15f5dda` | [BSD-3-Clause](../third_party/flash-attention/LICENSE) |

学生入口使用 CUDA C++/PTX。答案中的 s22/s31/s32 使用 C++ CuTe 表达布局与指令，阶段 4 使用上游 CuTe DSL 生产 kernel。`bootstrap` 安装包括 CuTe DSL、quack、TVM FFI、DLPack 支持在内的完整课程依赖；版本由项目依赖声明固定。

## 从执行入口到设备实现

| 课程 | 执行入口 | 完整设备实现与阅读重点 | 实验配置 |
|---|---|---|---|
| s22_fa2 | [kernel.cu](../solutions/s22_fa2/kernel.cu)、[fa2_optimized.cuh](../solutions/s22_fa2/fa2_optimized.cuh) | 本地 attention 实现及 CUTLASS CuTe layout/MMA/copy atoms；[WMMA 答案](../solutions/s22_fa2/wmma_readable.cuh) 展示各数学阶段 | CUDA C++ CuTe 与 WMMA 两种实现使用同一公共契约 |
| s31_tcgen | [kernel.cu](../solutions/s31_tcgen/kernel.cu)、[host 集成](../solutions/s31_tcgen/kernel_impl.cuh) | [SM100 collective](../third_party/cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp)、[tcgen05 atom](../third_party/cutlass/include/cute/arch/mma_sm100_umma.hpp)、[TMA 教程](../third_party/cutlass/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu) | 固定两个 K buffers |
| s32_specialize | [kernel.cu](../solutions/s32_specialize/kernel.cu) | 同一 SM100 collective、`include/cutlass/pipeline` 与 epilogue collective | 按 shared-memory 预算选择 K buffers，保持 tile/dtype 相同 |
| s40_attention | [entry.py](../solutions/s40_attention/entry.py)、[编译入口](../solutions/s40_attention/fa4_reference.py) | [flash_fwd_sm100.py](../third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py) 中 TMA、MMA、TMEM、softmax、correction 与 epilogue | q_stage=1；硬件 exp；rescale threshold=0；非 persistent |
| s41_overlap | [entry.py](../solutions/s41_overlap/entry.py) | forward 的双 query stage 与 [pipeline.py](../third_party/flash-attention/flash_attn/cute/pipeline.py) | q_stage=2，其余保持 s40 配置 |
| s42_numerics | [entry.py](../solutions/s42_numerics/entry.py) | [softmax.py](../third_party/flash-attention/flash_attn/cute/softmax.py)、[utils.py](../third_party/flash-attention/flash_attn/cute/utils.py) 与 forward 的 rescale 分支 | emulated exp 与 threshold=8，可分别做消融 |
| s43_schedule | [entry.py](../solutions/s43_schedule/entry.py) | [tile_scheduler.py](../third_party/flash-attention/flash_attn/cute/tile_scheduler.py) 与 forward 的选择逻辑 | static persistent；causal 采用源码中的 LPT 选择 |

s31/s32 都使用 producer/consumer warp 角色，可执行比较隔离缓冲数量。先通过 [01_mma_sm100.cu](../third_party/cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu) 和 02 教程理解各机制，再在生产 collective 中阅读其组合。s40 的单 query stage 也使用异步引擎；s41 增加跨 query stage 的交错。四个 attention 答案共享完整生产源码，并在配置中明确各课研究的变量。

## 可审阅的 FA4 补丁

[rescale-threshold.patch](../patches/flash-attention/rescale-threshold.patch) 把上游 16-bit softmax 路径中固定的 rescale threshold 暴露为构造参数，默认值 8.0 保持上游策略。课程以 0.0 研究 exact rescale，以 8.0 研究 conditional rescale。

公共设施从 gitlink 指定源码和补丁生成 `.cache/sources/` 下的编译副本，上游 submodule 保持干净。源码与补丁参与构建和验收指纹；修改后重新准备与验证。exp 与 query stage 使用已有对象配置。

## 执行与比较

```bash
./lab check s31_tcgen --solution
./lab solution s31_tcgen --case 5
./lab check s40_attention --solution --case 0 --timeout 300
./lab check s40_attention --solution --timeout 600
./lab solution s40_attention --case 8 --timeout 600
```

prepare 完成描述符、输出/workspace 分配和编译，随后在指定 stream 上运行设备工作。attention 输入/输出是 BNHD，LSE 是 BHN。公共 benchmark 对比相同输出集合、输入值和 dtype，分别报告 graph/eager 与库基线。完整正确性还包括全 case、重复调用、非默认 stream 和 sanitizer，方法见 [验证流程](validation.md)。
