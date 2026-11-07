# CUDA/PTX：sliced-Q 的完整融合 attention

入口是 [kernel.cu](kernel.cu) → [fa2_optimized.cuh](fa2_optimized.cuh)。设备代码直接编写 `cp.async`、`ldmatrix`、`mma.sync` 与 warp shuffle，只包含 CUDA headers。输入/输出为 contiguous BNHD，LSE 为 BHN 的 FP32 自然对数；支持 FP16/BF16、D=64/128、任意正 N、causal/noncausal self-attention。

## 从坐标到 fragment

每个 CTA 有四个 warp。D64 使用 128×64 的 query/key tile，D128 使用 64×64；后者减少每线程长期驻留的输出与 score 寄存器。每个 warp 拥有独立 query 行，四个相邻 lane 共同处理一行的不同列。QK 的 16×8 accumulator 中，lane 的行是 `lane/4 + {0,8}`，列是 `2*(lane%4) + {0,1}`。沿着这些坐标施加 mask，再用四 lane shuffle 求行最大值。分母的局部和先留在线程中，最终再归约。

两个相邻 16×8 score fragment 经指数和 FP16/BF16 打包后，直接组成 PV 所需的 16×16 A fragment；无需将整个 P 写入 shared memory。O、行最大值和分母跨 key tile 保存在 FP32 寄存器。P 的低精度舍入仍是误差来源，应与指数、输出舍入分别分析。

## 搬运、掩码与写回

Q 一次搬入并保留。key tile 从后向前遍历：搬 V 时执行当前 QK；V 完成且 K 已被消费后，在 softmax/PV 期间搬入下一块 K。`wait_group 0` 仅等待当前线程发起的拷贝，随后 CTA barrier 才建立所有线程的可见性和复用边界。每轮 V 的覆盖也必须等到上一轮所有 PV 读者完成。

causal 的最早几轮可能整行无有效 key，代码显式处理 `maximum=-inf` 和零概率，避免 `-inf-(-inf)` 生成 NaN。尾部 copy 填零且使用分配范围内的源指针，fragment mask 排除无效 key，最终 O/LSE store 排除无效 query。结束后复用 Q 的 shared 空间，将分散 fragment 整理为连续 16 字节 O 写回。

## Scale 与表示范围

公共 scale 必须可表示为正且有限的 FP32。令 `scale2=scale*log2(e)`；仅当 scale、scale2 均为 normal、`scale2<=1` 且 `scale>=2^-119` 时选择普通 FP32 specialization。其余合法 scale 使用独立模板：最大值保留在原始 QK 单位，先把两个分数转为 FP64，再计算差值乘 scale 和 log2(e)；LSE 在自然对数单位计算 `raw_max*scale+log(denominator)` 后转回 FP32。课程默认 scale 仍走普通路径。

下限保护“两个有限 FP32 raw score 相减先溢出”的情形：scale 不小于 `2^-119` 时，这种差值对应的自然指数已小于约 -512，其概率必然在 FP32 下溢；更小 scale 则可能保留显著权重，必须先在 FP64 中相减。`scale_raw_gap` 用精确 BF16 幂次构造有限 raw QK，单独验证这一边界。该处理不改变 Tensor Core 的 FP32 累加、低精度 P 或 FP32 LSE 输出；QK 累加本身或最终 LSE 超出表示范围时，不能仅靠有限输入保证全域精度。

## 验证与性能

```bash
./lab check s22_fa2 --solution
./lab solution s22_fa2 --case 9
./lab solution s22_fa2 --case 10
./lab solution s22_fa2 --case 11
./lab sanitize s22_fa2 --solution --tool memcheck --case 6
./lab sanitize s22_fa2 --solution --tool racecheck --case 12
```

先用独立 FP64 oracle 验证，再与 cuDNN O+LSE 基线比较。与 s12 的比较还跨越了 FP32 SIMT 和低精度 Tensor Core，应另用相同 dtype/输出契约的控制版本区分各项收益。寄存器压力、query tile 大小、非矩阵指令和 `cp.async` 等待都是性能检查对象；答案的优化效果以全部 perf case 的同模式计时为准。

## 来源

分工和流式算法依据 [FA2 论文](https://arxiv.org/abs/2307.08691v1)。fragment 转换、反向 key 遍历及 staging 的设计对照为 [FlashAttention v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/tree/5231d95fe13733fb534c01895f7ea88c6a6c7793) 的 `kernel_traits.h`、`utils.h`、`softmax.h`、`flash_fwd_kernel.h`，commit `5231d95fe13733fb534c01895f7ea88c6a6c7793`。沿用改编工作的 [BSD-3-Clause 许可证](../../third_party/flash-attention/LICENSE) 和 [作者署名](../../third_party/flash-attention/AUTHORS)。本地入口、地址计算、PTX 指令及同步均可直接阅读；固定上游源码另用于比较生产实现。
