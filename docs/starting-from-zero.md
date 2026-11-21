# 从零开始：读懂并运行第一个 GPU kernel

本章面向会写普通 C/C++ 和 Python、但还没怎么写过 CUDA 的读者。读完后你应该能看懂 [00_transpose](../course/00_transpose/README.md) 的骨架 `students/00_transpose/kernel.cu`，并完成它的第 1 步。

先不考虑 attention。把一个矩阵转置正确并做快，已经会用到后面所有关卡需要的四件事：地址、并行分工、存储层级、同步。

## 1. 一个函数为什么会执行很多次

CPU 上的普通函数调用只有一条执行流。CUDA 的 `__global__` 函数叫 **kernel**。host（CPU）发起一次 kernel，GPU 上会有很多线程同时运行这同一份函数体；每个线程根据自己的编号处理不同的数据。

```cpp
transpose_naive<<<grid, block, shared_bytes, stream>>>(x, y, rows, cols);
//               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  执行规模：多少线程、多少 shared memory、哪个 stream
//                                                      ^^^^^^^^^^^^^^^^^^  每个线程都收到同样的参数
```

| 名称 | 含义 | 由谁决定 |
|---|---|---|
| host | 发起 GPU 工作的 CPU 代码 | 你写的普通 C++ |
| kernel | 所有线程执行的设备函数 | `__global__` 函数 |
| grid | 一次 launch 的全部 block | `<<<grid, ...>>>` 指定各维 block 数 |
| block / CTA | 一组线程，共享一块 shared memory，可以用 `__syncthreads()` 同步 | `<<<..., block>>>` 指定各维线程数 |
| warp | 32 个线程一组，一起执行同一条指令 | 硬件按 block 内的线性编号划分 |
| lane | 线程在 warp 内的编号，0–31 | 线性编号 % 32 |
| SM | 执行 warp 的硬件单元；B200 有 148 个 | 硬件调度，程序不能指定 |

CTA（Cooperative Thread Array）是 block 在 PTX 里的名字，两者指同一个东西。

二维 block 中，线程的线性编号是 `t = threadIdx.x + blockDim.x * threadIdx.y`，所以 `warp = t / 32`，`lane = t % 32`。
00 关的 block 是 `(32, 8)`：共 256 个线程、8 个 warp；同一个 `threadIdx.y` 的 32 个线程恰好是一个 warp。

一个 SM 可以同时驻留多个 block，数量受寄存器、shared memory 和线程数限制。**block 编号不是 SM 编号，也不能假设 block 按编号顺序执行。**

```
host: make_plan + launch
        │
        ▼
      grid ──┬── block 0 ──┬── warp 0（lane 0..31）
             │   (shared)  └── warp 1 ...
             ├── block 1 ...
             └── ...         ← 硬件把 block 分配到空闲的 SM 上
```

## 2. 数组是一段线性存储

连续的（contiguous）行主序矩阵 `X[R,C]` 中，元素 `(r,c)` 在 `X[r*C + c]`。
例如 `X = [[0,1,2],[3,4,5]]` 在内存中是 `[0,1,2,3,4,5]`。
它的转置 `Y = Xᵀ` 形状是 `[3,2]`，内存中应为 `[0,3,1,4,2,5]`。

所以转置的地址等式是：

```
Y[c*R + r] = X[r*C + c]
```

只改 shape 不会完成转置：数据必须真的被搬到新位置。

后面的 attention 使用连续的 `[B,N,H,D]`。元素 `(b,n,h,d)` 的偏移是 `((b*N + n)*H + h)*D + d`。
固定 batch 和 head 时，相邻 token 相隔 `H*D` 个元素，而不是 `D` 个。
把它误当成 `[B,H,N,D]` 会读错数据，所以课程的 case 总是包含 `B>1`、`H>1`（见 [11 关 §3.3](../course/11_naive_attention/README.md)）。

大矩阵的元素数可能超过 2³¹，线性下标要用 `int64_t`。

## 3. 数据放在哪里，决定谁能复用它

| 存储 | 谁能直接访问 | 生命周期 | 00 关的用途 |
|---|---|---|---|
| global memory（HBM） | 所有线程 | 由 host 分配，直到释放 | 输入 x、输出 y |
| shared memory | 同一个 block 的线程 | block 执行期间 | 暂存一个 tile，改变读写顺序 |
| 寄存器 | 只有当前线程 | 线程执行期间 | 下标、临时值、累加器 |

源码里的标量变量通常放在寄存器。寄存器不够时，编译器会把一部分变量放进 local memory（实际在 HBM 上，经 L1/L2 缓存），这叫 **spill**，会变慢。`./lab sass` 可以看到有没有 spill（见 [工具指南](tools.md)）。

shared memory 也有容量限制（B200 每个 SM 228 KB，见 [B200 架构概览](b200.md)）。每个 block 用得越多，一个 SM 上能同时驻留的 block 越少。

## 4. 课程的 kernel 文件长什么样

每一关你只改一个文件：`students/<ID>/kernel.cu`。它固定包含三样东西：

```cpp
struct Plan {};                                          // 准备好的状态：00 关不需要，是空的

Plan make_plan(const lab::TransposeProblem &p);          // 准备：不计时
void run(const Plan &, const lab::TransposeProblem &p,   // 执行：计时；只在 stream 上发射 kernel
         cudaStream_t stream);

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);   // 把上面两个函数交给 Python 框架
```

| 部分 | 做什么 | 00 关 |
|---|---|---|
| `lab::TransposeProblem` | 框架传进来的问题：设备指针 `x`、`y`，尺寸 `rows`、`cols`，当前 step `p.step` | 定义在 [`include/lab/problem.h`](../include/lab/problem.h) |
| `make_plan` | 只调用一次，不计时。后面的关卡在这里编码 TMA 描述符、分配 scratch | 只检查 dtype 与尺寸 |
| `run` | 每次计时都调用。只负责发射 kernel，必须用传入的 `stream` | 按 `p.step` 选择版本 |
| `LAB_TODO(n)` | 未完成的地方。运行到这里时报告 “TODO(n) 未实现” | 每个 step 一个 |

`run` 按 `p.step` 选择版本，所以每一步的实现都保留在同一个文件里，可以互相比较速度。

输出 `y` 在每次检查前被填满 NaN。如果有元素没写到，检查会报告 “never written”。

## 5. 读懂第 1 步

打开 `students/00_transpose/kernel.cu`，看 `run` 中 `case 1`：

```cpp
dim3 block(32, 8);
dim3 grid((p.cols + 31) / 32, (p.rows + 7) / 8);
transpose_naive<<<grid, block, 0, stream>>>(x, y, p.rows, p.cols);
```

- 每个 block 覆盖 x 的 8 行 × 32 列，每个线程负责一个元素。
- `(cols + 31) / 32` 是向上取整：`cols = 11` 时也要 1 个 block。所以边缘 block 有一部分线程落在矩阵外面。

`transpose_naive` 里是 `LAB_TODO(1)`。骨架注释已经列出三件事：

1. 用 `blockIdx`、`blockDim`、`threadIdx` 算出本线程的 `(r, c)`；
2. `(r, c)` 在矩阵外就返回；
3. 把 `x[r][c]` 拷到 `y[c][r]`。

先在纸上回答：`threadIdx.x` 应该对应行还是列？提示：同一个 warp 的 32 个线程 `threadIdx.y` 相同、`threadIdx.x` 连续，而 `block` 的 x 方向是 32。

写完后运行：

```bash
./lab check 00 --step 1
```

还没写时，输出是这样的：

```
step 1: 朴素转置：一个线程一个元素
  TODO  tiny [7, 11] fp32
        purpose: 比一个 tile 还小：检查行列下标与越界判断。……

TODO: TODO(1) is not implemented yet (reached on the GPU).
```

第一个 case `tiny` 是 7×11，每个输入值等于它的线性下标。答错时报告会给出“在某坐标得到 v、期望 w”，`v` 就告诉你读到了 x 的哪个位置。每个 case 的 `purpose` 说明它针对哪类错误。

## 6. CPU 返回了，不代表 GPU 已经完成

kernel launch 只是把工作放进 **stream** 的队列，然后立刻返回 CPU。同一个 stream 里的工作按顺序执行；不同 stream 之间需要显式依赖。

因此，用 CPU 时间夹住一次 launch，测到的主要是提交开销，而不是 kernel 时间。`./lab bench` 用 CUDA event 在 stream 上计时，并默认用 CUDA Graph 一次回放多次 launch，去掉 Python 发起的间隙。编译、`make_plan`、造输入和分配内存都不计入。细节见 [工具指南 §5](tools.md#5-计时graph-与-cold)。

检查还会在非默认 stream 上运行你的 kernel。kernel 必须发射到 `run` 收到的 `stream` 上；写死默认 stream 可能在小例上碰巧正确，但会破坏依赖关系。

```bash
./lab bench 00 --step 1
```

输出中 `vs torch` 是 PyTorch 转置的耗时除以本行的耗时，大于 1 表示比 PyTorch 快。

## 7. 接下来

1. 读 00 关讲义 §4，弄清朴素转置的哪一侧访存不合并，再做第 2–4 步；
2. 每做完一步，先 `check` 再 `bench`，用 `./lab profile 00 --step n` 验证你对 bank conflict 的预测；
3. 完成后进入 [01_softmax](../course/01_softmax/README.md)，从“交换数据位置”进到“合并多个线程的数值结果”。

遇到不认识的词，查 [术语表](glossary.md)。
