# 测试规模与工作量

小 case 用于定位索引、数值和同步错误；大 case 用于检验完整问题的输出覆盖、长循环和多 CTA 扩展性。两者都放在同一个 `curriculum.json` 的 cases 列表中，`./lab check ID` 默认全部验收。旧 case 编号保持不变，完整输入与编号可用 `./lab task ID -v` 查看。benchmark、sanitizer 和 profiler 继续使用原有 `--case N` 参数。

## Attention

shape 为 `[B,H,N,D]`，仍限定 contiguous self-attention、相同 Q/K/V head 数、无 dropout/bias。scale 默认 `D**-0.5`；O 与输入同 dtype，LSE 和累加状态为 FP32。大尺寸不改变各阶段允许的 dtype 或 mask。

| 练习 | 主要规模与目的 |
|---|---|
| s02_naive | `(1,8,512,64)`、`(2,8,1024,64)`、`(1,8,2048,128)`；观察显式 S/P 的内存代价，控制朴素版调试成本 |
| s10 / s20 / s21 | 完全相同的 FP32 非因果形状：`(1,8,1024,64)`、`(2,8,2048,64)`、`(1,16,4096,128)`、`(1,16,8192,128)`；比较 FA1/FA2 结构时保持 shape/dtype 一致 |
| s11 | 同量级 FP32，包含 causal、N=2049/4097 的尾部；更长的归约与部分有效 tile |
| s23 | 上述四种非因果形状，各测试 BF16 和 FP16；Tensor Core 的收益与精度变化分别记录 |
| s24 | BF16/FP16，覆盖多头、8K 序列、causal 和 N=2049/4097 尾部 |
| s40 / s41 / s43 / s44 | 相同 BF16、D=128、非因果形状，N=1024/2048/4096/8192，B/H 变化；比较流水线、指数和重缩放优化 |
| s45 | 在 BF16/FP16 上覆盖 D=64/128、长序列、causal 与尾部 |
| s46 | 沿用阶段 4 的非因果形状，再加入 2K/4K causal；观察 grid 工作量和调度效果 |

这些是常见 dense MHA/prefill 工作量的受限样本，不代表某个模型的全部输入分布，也不包含 decode、GQA 或 MLA。[FA4 论文 v1 §5](https://arxiv.org/html/2603.05451v1#S5) 的评估使用更广的 1K–32K 序列、D=64/128 等设置，并固定总 token 数；本课程只取其中的规模量级，不声称复现其整套实验或性能。这里 B、H、N、D 独立变化，跨 case 比较必须记录完整 shape，不能只按 N 解读。

## GEMM

`s22_mma` 和 `s38_async_gemm` 均包含以下 BF16/FP16 输入、FP32 输出的完整 GEMM。shape 顺序为 `(M,N,K)`，实际 A `[M,K]`、B `[K,N]`，均 row-major。

| `(M,N,K)` | 要观察的问题 |
|---|---|
| `(256,256,256)` | 从指令级计算扩展到多个输出 tile |
| `(1024,1024,1024)` | 多 CTA 的输出覆盖和并行度 |
| `(2048,2048,2048)` | 大方阵、持续计算与复用 |
| `(4096,4096,128)` | QK 比例：输出面大、K 较短 |
| `(4096,128,4096)` | PV 比例：输出较窄、K 较长 |
| `(512,2048,256)` | 非方阵的 M/N 索引与覆盖 |

以上尺寸都是 16 的倍数；s22 暂不增加任意尾部要求。QK/PV 比例只指矩阵尺寸，GEMM 输入仍按本关独立随机模式生成，不是一次完整 attention 中产生的 P。小 MMA 的 fragment、算法 tile、CTA 和完整矩阵尺寸不能混为一谈。

```bash
./lab check s22_mma --timeout 300
./lab bench s22_mma --case 14  # BF16 2048^3
./lab bench s22_mma --case 15  # BF16 QK proportions
./lab bench s22_mma --case 16  # BF16 PV proportions
./lab profile s22_mma --case 14
```

## 基础与机制练习

- s01/s03：更多行、列宽 2048/8192、尾部宽度 4097，验证 reduction 与 online 状态能否跨越更长的数据范围。
- s30：1024×2048 与 1025×2049 transpose，覆盖更多工作块及尾部。
- s32/s33：1024×1024 和 2048×4096 copy，观察完整输出覆盖与搬运；后者单个 FP32 输入为 32 MiB，不能据此直接宣称已测到持续 HBM 带宽。
- s34/s35：T=64/257，s35 另有 T=1024；T 表示数据 tile 序列长度，用于检验循环和 phase wrap，不预设 buffer 数或 launch。
- s00/s31 保持固定 softmax 契约；s36 保持 TMEM 生存期小实验；s37 保持单次 tcgen05 MMA；s42 保持指数误差实验。完整矩阵的扩展与性能判断放在后续 GEMM/attention 中。

## 验收与测量

所有大 case 使用原有检查：固定种子、独立 FP64 oracle、默认/非默认 stream、重复调用、输出预填 NaN、输入不变。容差不因尺寸变大而放宽。全部 case 通过才记录完整数值通过；只跑单 case 或 benchmark 不能替代全关验收。

CPU 基础设施 pytest 检查所有 case 元数据，在小张量上验证数学函数及受限临时显存参考与 dense 定义的等价性。`./lab smoke` 在所选 GPU 上遍历全部 case 的输入生成、host 校验和 FP64 参考；它不执行学生计算，也不证明学生正确性。两者的覆盖范围见 [验证流程](validation.md)。

大 case 的参考和首次编译可能耗时更长；使用合理的 `--timeout`，超时需分辨编译/参考开销与学生死锁。不要无限延长超时。kernel 计时仍排除生成、分配和参考计算；attention 按有效 QK/PV FLOPs，GEMM 按 `2*M*N*K`，详见 [计时](benchmarking.md)。单个大矩阵也不保证已经达到稳态峰值，仍需查看缓存、occupancy 和计时波动。
