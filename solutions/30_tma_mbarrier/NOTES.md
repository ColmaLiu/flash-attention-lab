# 30_tma_mbarrier 答案说明

## 设计选择

- **支持域**：FP32，`rows % 4 == 0`、`cols % 4 == 0`。这是 TMA 的硬性要求：x 和 y 的行距（`cols·4`、`rows·4` 字节）都必须是 16 的倍数。尾部 tile 完全交给 TMA：load 越界填 0（且计入 tx 字节），store 越界跳过，kernel 中没有边界判断。没有扩展到 BF16：2 字节元素的转置需要在寄存器里重排半字（`prmt`），与本关的机制无关。
- **原语**：`include/lab/mbarrier.cuh`、`include/lab/tma.cuh`，答案只 `#include` 它们。地址转换函数命名为 `lab::shared_addr`，避免与 `mma_sync.cuh` 的 `lab::smem_u32` 在同一翻译单元中重定义。
- **step 1**：一个 CTA 一个 32×32 tile，4 个 warp。展示最小协议：init + `fence.mbarrier_init` + `__syncthreads`；`arrive.expect_tx` + TMA load；`try_wait.parity 0`；generic 写 `out` 后每个线程 `fence.proxy.async` 再 `__syncthreads`；TMA store + commit + `wait_group 0`。转置是标量的、有 32 路 bank conflict，故意留给 step 3。
- **step 2**：persistent。warp 0 中 `elect.sync` 选出的一个线程是 producer；`kRows/32` 个 consumer warp 各管 tile 的 32 行，各自存自己的 32×32 输出 box，用自己的 bulk group 管理两个 out buffer（`wait_group.read 1`）。因此初始化后整个 kernel 没有 CTA 级 barrier：`full[s]`（count 1 + tx）与 `empty[s]`（count = consumer warp 数，每 warp lane 0 在 `__syncwarp` 后 arrive 一次）就是全部同步。producer 首圈等 `empty` 的 parity 1，利用“前一个 phase 视为完成”，与 CUTLASS `PipelineTmaAsync` 相同。
- **step 3**：x、y 两个 map 都用 128B swizzle；每个 lane 处理两个 4×4 块 `(R, C) = (l%8, (l%8 + l/8 + 4t)%8)`，float4 读、寄存器转置、float4 写。用脚本逐一验证过：每 8 个 lane 一组时，读和写的物理 chunk 两两不同。配置 `Config<256, 2, true>`（box 32×256，2 级，9 warp，128 KB，每 SM 1 个 CTA）。
- **generic 读 → TMA 覆盖不加 proxy fence**：consumer 读 `in[s]` 后经 mbarrier arrive/wait 交给 producer 再发 TMA，与 CUTLASS 的 consumer_release 一致。generic 写 → TMA store 读则每个写线程都执行 `fence.proxy.async.shared::cta`。
- **退出前 `bulk_wait_group<0>`**（不是 `.read`）：保守写法，确保 store 全部完成；代价是每个 consumer warp 末尾多等一次写延迟，对 persistent kernel 可以忽略。31 关用 CUTLASS 的 `.read` 写法，讲义两处都有说明。

## 各 step 的性能（4 号卡，CUDA Graph，每个 step 各测 2–3 次）

| step | `big` GB/s | `wide` GB/s | `vs copy` | 说明 |
|---|---|---|---|---|
| 1 | ≈ 1950 | ≈ 1950 | ≈ 0.30 | bank conflict 限制（每条 LDS 约 31 次额外 wavefront） |
| 2 | ≈ 2100 | ≈ 2100 | ≈ 0.33 | 同上：流水只换来约 8%，转置本身是瓶颈 |
| 3（`<128,4>`，早先测量） | ≈ 5600 | ≈ 5600 | ≈ 0.87 | 只加 swizzle + 4×4 转置 |
| 3（答案 `<256,2>`） | ≈ 6190 | ≈ 6095 | ≈ 0.96 / 0.95 | |
| `copy`（`y.copy_(x)`） | ≈ 6430 | ≈ 6430 | 1 | 实际 HBM 上限 |
| `torch`（`y.copy_(x.t())`） | ≈ 1860–1905 | ≈ 1885–1905 | ≈ 0.29 | |

- 同机 00 关步骤 4 约 5910 GB/s（`vs copy` ≈ 0.92），所以 TMA 在带宽上只领先约 5%（`--cold` 约 2%）。讲义如实写了这一点，把收益表述为“1 个发起线程、9 个 warp 达到同样带宽”。
- `perf_target` 是 `{"baseline": "copy", "ratio": 0.85}`：`copy` 与答案都受 HBM 限制，比值稳定。`torch` 不适合当门槛：早先一次（3 号卡）测量中 step 1/2 只有 ≈ 1030/1060 GB/s，torch 在约 1.0 与 1.8 TB/s 之间跳动，而 step 3 与 copy 不变。假设（未验证）：那次 GPU 处于较低 SM 时钟，受 bank conflict / 延迟限制的 kernel（torch 转置、step 1/2）随 SM 时钟变慢，HBM 限制的 kernel 不受影响。本次测量时 SM 时钟为 1965 MHz（最大值），各次结果稳定。
- 调参（`big`）：`<32,4>` ≈ 5180，`<64,4>` ≈ 5650，`<128,2>` ≈ 5720，`<128,3>` ≈ 5720，`<128,4>` ≈ 5630，`<128,6>` ≈ 5600，`<192,3>` ≈ 5790，`<256,2>` ≈ 6200，`<256,3>` ≈ 5770，`<256,4>` ≈ 5750。更多在途字节并不更快；原因未查明。
- NCU（`big`）：step 1/2 shared load bank conflict ≈ 6.6e7（LDS ≈ 2.1e6 条）；step 3 load ≈ 7.3e5、store ≈ 1.0e6（LDS/STS 各 ≈ 5.2e5 条）。换一种 lane 分配后计数几乎不变，推测残余计数来自与 TMA 访问的仲裁，未证实。

## 验证

- `./lab check 30 --solution`：3 个 step × 7 个 case 全部精确通过。
- `selftest 30 --perf` 通过；骨架在 make_plan 处报 `TODO(1)`。
- compute-sanitizer：memcheck / racecheck / synccheck 在 `tail` 上（3 个 step）、racecheck 在 `ring` 上（step 2、3）均无报错。
- **sanitizer 的盲区**（实验）：删掉 consumer 的 `fence_proxy_async` 后，测试（`ring` × 3 次）和 racecheck 都通过；删掉 producer 的 `empty` 等待后 `ring` 结果错误，但 racecheck 仍报 0 hazard。racecheck 不跟踪 async proxy（TMA）的 shared memory 访问。讲义 §4.7、自测题 6 用了这个结论。
- `./lab sass 30 --solution`：无 spill；有 `UTMALDG`、`UTMASTG`、`SYNCS`。
- 骨架：在临时副本中按 TODO(1)、(2)、(3) 逐步填入，第 n 步只依赖前 n 组 TODO，填完与答案同速；骨架已恢复为原样。

## 已知局限

- 只支持 FP32、行列都是 4 的倍数。
- tile 顺序是简单的行优先（`t → (t / tiles_c, t % tiles_c)`），没有研究 DRAM 页局部性。
- 2 级优于 3–4 级的原因没有查明。
