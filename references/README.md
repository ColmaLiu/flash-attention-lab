# 完整参考源码与阅读顺序

这里保存可直接阅读的固定上游源码，配合 `solutions/` 的独立可执行入口。学生实现不调用这些答案。上游来源、commit、许可证和本地修改都列在下面；下载及安装不会在普通学生检查时隐式发生。

| 来源 | 固定 commit | 本地范围 | 许可证 |
|---|---|---|---|
| NVIDIA CUTLASS v4.4.1 | `4370102f9dacab813282e1d67722fceb0b90a019` | `cutlass-sm100/include` 完整设备实现、`tools/util/include`、Blackwell 教程与 collective builder 示例 | [BSD-3-Clause](cutlass-sm100/LICENSE.txt) |
| Dao-AILab FlashAttention FA4 beta35 | `1fc0221183e799363001394a12dd9e78e15f5dda` | `flash-attention/flash_attn/cute` 完整 Python/CuTe 实现及支持模块 | [BSD-3-Clause](flash-attention/LICENSE) |

本地文件清单与 SHA256 在 [manifest.json](manifest.json)。这是选定源码的固定快照，不追踪 upstream main。CUTLASS 与官方包不通过系统全局安装；C++ 扩展按需编译。本课程先前的可选 CUTLASS submodule 与这里的 SM100 答案快照互不替代。

## 从入口读到设备实现

| 课程 | 可执行入口 | 完整实现的主要阅读位置 | 配置差别 |
|---|---|---|---|
| s31_tcgen | [kernel.cu](../solutions/s31_tcgen/kernel.cu) / [host集成](../solutions/s31_tcgen/kernel_impl.cuh) | [SM100 collective](cutlass-sm100/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp)、[tcgen05 atom](cutlass-sm100/include/cute/arch/mma_sm100_umma.hpp)、[Blackwell教程](cutlass-sm100/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu) | 固定两个 K buffers，上游该 collective 的最小合法数量 |
| s32_specialize | [kernel.cu](../solutions/s32_specialize/kernel.cu) | 同一完整 SM100 collective、`include/cutlass/pipeline`、epilogue collective | 自动选择可容纳的 K buffers，其他 tile/dtype 不变 |
| s40_attention | [entry.py](../solutions/s40_attention/entry.py) / [直接编译入口](../solutions/s40_attention/fa4_reference.py) | [flash_fwd_sm100.py](flash-attention/flash_attn/cute/flash_fwd_sm100.py) 中 MMA、TMA、TMEM、softmax、correction 与 epilogue | q_stage=1；硬件 exp；rescale threshold=0；非 persistent |
| s41_overlap | [entry.py](../solutions/s41_overlap/entry.py) | 同一 forward 的双 query stage 与 pipeline、[pipeline.py](flash-attention/flash_attn/cute/pipeline.py) | q_stage=2；其余保持 s40 配置 |
| s42_numerics | [entry.py](../solutions/s42_numerics/entry.py) | [softmax.py](flash-attention/flash_attn/cute/softmax.py)、[utils.py](flash-attention/flash_attn/cute/utils.py)、forward 的指数与 correction | 启用 emulated exp 与 threshold=8；两者可在独立配置中分别消融 |
| s43_schedule | [entry.py](../solutions/s43_schedule/entry.py) | [tile_scheduler.py](flash-attention/flash_attn/cute/tile_scheduler.py) 与 forward 的选择逻辑 | 开启 static persistent；causal 按源码选择 LPT scheduler |

s31 的完整生产 collective 本身已经包含 warp 角色和异步执行，不能把 s31→s32 的速度变化全部归因于“第一次引入 warp specialization”。它们的可执行差别是缓冲数量，角色划分是 s32 要深入阅读的内容。s40 的 q_stage=1 也不是完全同步的设备 kernel。四个 attention 入口复用同一份完整生产源码，避免复制后漂移；它们不是四个空包装课程，也不代表新写了一份独立 CUDA FA4。

## FA4 答案的可选依赖

```bash
./lab python -m pip install -e '.[fa4]'
./lab check s40_attention --solution --case 0 --timeout 300
./lab check s40_attention --solution --timeout 600
./lab solution s40_attention --case 0
```

`pyproject.toml` 固定 CuTe DSL 4.6.2、quack-kernels 0.6.4、cuda-bindings/Python 12.9.4、einops、TVM FFI 与 DLPack 支持包。主线 CUDA/PTX 学生实验不要求这些依赖。编译直接指向本地固定源码，若进程已加载其他 flash_attn 包则明确报错；不会悄悄使用另一个安装版本。

host prepare 将 PyTorch tensor 包装为 CuTe 参数并完成编译。输出 O/LSE 预分配，传入当前 stream；计时只运行编译结果。初次编译和源码检查在计时之外。对应完整课程验收仍需全 case、非默认 stream、重复调用与 sanitizer，不能仅以成功 import 作为通过。

## 本地修改

CUTLASS 文件保持上游原样。FA4 的 `flash_fwd_sm100.py` 增加一个 constructor 参数 `rescale_threshold`（默认 8.0），保存到 `self.lab_rescale_threshold`，替代原来 16-bit 路径中硬编码的 8.0。除这一暴露既有策略的修改外，保留完整上游实现；详见 [PATCHES.md](flash-attention/PATCHES.md)。exp 开关与 query stage 使用原有对象配置，不动态改写源码。许可证保留原作者署名。
