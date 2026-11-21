# Starting from zero: read and run your first GPU kernel

This chapter is for readers who can write ordinary C/C++ and Python but have not written much CUDA. After reading it you should be able to understand the skeleton `students/00_transpose/kernel.cu` of [00_transpose](../course/00_transpose/README.md) and complete its step 1.

Set attention aside for now. Making a matrix transpose correct and fast already uses the four things every later lab needs: addresses, parallel work split, the memory hierarchy, synchronization.

## 1. Why one function runs many times

An ordinary function call on the CPU has a single flow of execution. A CUDA `__global__` function is called a **kernel**. When the host (CPU) launches a kernel once, many GPU threads run the same function body at the same time; each thread uses its own index to process different data.

```cpp
transpose_naive<<<grid, block, shared_bytes, stream>>>(x, y, rows, cols);
//               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  launch size: how many threads, how much shared memory, which stream
//                                                      ^^^^^^^^^^^^^^^^^^  every thread receives the same arguments
```

| Name | Meaning | Set by |
|---|---|---|
| host | The CPU code that launches GPU work | Ordinary C++ that you write |
| kernel | The device function that all threads execute | A `__global__` function |
| grid | All blocks of one launch | `<<<grid, ...>>>` gives the number of blocks in each dimension |
| block / CTA | A group of threads that share a region of shared memory and can synchronize with `__syncthreads()` | `<<<..., block>>>` gives the number of threads in each dimension |
| warp | A group of 32 threads that execute the same instruction together | The hardware splits by the linear index within the block |
| lane | A thread's index within its warp, 0–31 | Linear index % 32 |
| SM | The hardware unit that executes warps; B200 has 148 | Scheduled by hardware; the program cannot choose |

CTA (Cooperative Thread Array) is the PTX name for a block; the two mean the same thing.

In a 2D block, a thread's linear index is `t = threadIdx.x + blockDim.x * threadIdx.y`, so `warp = t / 32` and `lane = t % 32`.
The block in lab 00 is `(32, 8)`: 256 threads, 8 warps in total; the 32 threads with the same `threadIdx.y` form exactly one warp.

One SM can hold several blocks at once; the number is limited by registers, shared memory and thread count. **A block index is not an SM index, and you cannot assume blocks run in index order.**

```
host: make_plan + launch
        │
        ▼
      grid ──┬── block 0 ──┬── warp 0 (lane 0..31)
             │   (shared)  └── warp 1 ...
             ├── block 1 ...
             └── ...         ← the hardware assigns blocks to idle SMs
```

## 2. An array is a linear stretch of memory

In a contiguous row-major matrix `X[R,C]`, element `(r,c)` is at `X[r*C + c]`.
For example, `X = [[0,1,2],[3,4,5]]` is `[0,1,2,3,4,5]` in memory.
Its transpose `Y = Xᵀ` has shape `[3,2]` and should be `[0,3,1,4,2,5]` in memory.

So the address equation of the transpose is:

```
Y[c*R + r] = X[r*C + c]
```

Changing only the shape does not transpose anything: the data must actually move to new positions.

Attention later uses a contiguous `[B,N,H,D]`. The offset of element `(b,n,h,d)` is `((b*N + n)*H + h)*D + d`.
With batch and head fixed, adjacent tokens are `H*D` elements apart, not `D`.
Mistaking it for `[B,H,N,D]` reads the wrong data, so the course's cases always include `B>1` and `H>1` (see [lab 11 §3.3](../course/11_naive_attention/README.md)).

The element count of a large matrix can exceed 2³¹, so linear indices must use `int64_t`.

## 3. Where data lives decides who can reuse it

| Memory | Who can access it directly | Lifetime | Use in lab 00 |
|---|---|---|---|
| global memory (HBM) | All threads | Allocated by the host, until freed | Input x, output y |
| shared memory | Threads of the same block | While the block runs | Holds one tile temporarily to change the read/write order |
| Registers | Only the current thread | While the thread runs | Indices, temporaries, accumulators |

Scalar variables in source code usually live in registers. When registers run out, the compiler puts some variables in local memory (which actually lives in HBM, cached through L1/L2); this is called a **spill**, and it slows the kernel down. `./lab sass` shows whether there is a spill (see the [Tools guide](tools.md)).

shared memory also has a capacity limit (228 KB per SM on B200, see the [B200 architecture overview](b200.md)). The more each block uses, the fewer blocks one SM can hold at once.

## 4. What a course kernel file looks like

In each lab you change only one file: `students/<ID>/kernel.cu`. It always contains three things:

```cpp
struct Plan {};                                          // prepared state: lab 00 needs none, so it is empty

Plan make_plan(const lab::TransposeProblem &p);          // prepare: not timed
void run(const Plan &, const lab::TransposeProblem &p,   // execute: timed; only launches kernels on the stream
         cudaStream_t stream);

LAB_REGISTER(lab::TransposeProblem, Plan, make_plan, run);   // hands the two functions above to the Python framework
```

| Part | What it does | Lab 00 |
|---|---|---|
| `lab::TransposeProblem` | The problem the framework passes in: device pointers `x`, `y`, sizes `rows`, `cols`, the current step `p.step` | Defined in [`include/lab/problem.h`](../include/lab/problem.h) |
| `make_plan` | Called only once, not timed. Later labs encode TMA descriptors and allocate scratch here | Only checks the dtype and sizes |
| `run` | Called for every timed run. It only launches kernels, and it must use the `stream` passed in | Selects a version by `p.step` |
| `LAB_TODO(n)` | An unfinished spot. When execution reaches it, it reports "TODO(n) is not implemented yet" | One per step |

`run` selects a version by `p.step`, so the implementation of every step stays in the same file and you can compare their speeds.

The output `y` is filled with NaN before each check. If some element is never written, the check reports "never written".

## 5. Understanding step 1

Open `students/00_transpose/kernel.cu` and look at `case 1` in `run`:

```cpp
dim3 block(32, 8);
dim3 grid((p.cols + 31) / 32, (p.rows + 7) / 8);
transpose_naive<<<grid, block, 0, stream>>>(x, y, p.rows, p.cols);
```

- Each block covers 8 rows × 32 columns of x, and each thread handles one element.
- `(cols + 31) / 32` rounds up: even `cols = 11` needs 1 block. So some threads of the edge blocks fall outside the matrix.

`transpose_naive` contains `LAB_TODO(1)`. The skeleton comment already lists three things:

1. Compute this thread's `(r, c)` from `blockIdx`, `blockDim` and `threadIdx`;
2. Return if `(r, c)` is outside the matrix;
3. Copy `x[r][c]` to `y[c][r]`.

First answer on paper: should `threadIdx.x` map to the row or the column? Hint: the 32 threads of one warp share the same `threadIdx.y` and have consecutive `threadIdx.x`, and the x dimension of `block` is 32.

When you are done, run:

```bash
./lab check 00 --step 1
```

Before you write anything, the output looks like this:

```
step 1: Naive transpose: one thread per element
  TODO  tiny [7, 11] fp32
        purpose: Smaller than one tile: checks row/column indices and bounds checks. …

TODO: TODO(1) is not implemented yet (reached on the GPU).
```

The first case `tiny` is 7×11, and each input value equals its linear index. When the answer is wrong, the report says "got v at some coordinate, expected w", and `v` tells you which position of x was read. Each case's `purpose` states which kind of bug it targets.

## 6. The CPU returning does not mean the GPU is done

A kernel launch only puts work into the queue of a **stream** and returns to the CPU immediately. Work in the same stream runs in order; work in different streams needs explicit dependencies.

So if you wrap one launch with CPU timers, you mostly measure the submission overhead, not the kernel time. `./lab bench` times on the stream with CUDA events, and by default uses a CUDA Graph to replay many launches at once, removing the gaps from Python launching them. Compilation, `make_plan`, input generation and memory allocation are not included. Details are in the [Tools guide §5](tools.md#5-timing-graph-and-cold).

The check also runs your kernel on a non-default stream. The kernel must be launched on the `stream` that `run` receives; hard-coding the default stream may happen to work on small examples, but it breaks the dependencies.

```bash
./lab bench 00 --step 1
```

In the output, `vs torch` is the time of the PyTorch transpose divided by this row's time; greater than 1 means faster than PyTorch.

## 7. Next

1. Read §4 of the lab 00 lesson to find out which side of the naive transpose has uncoalesced memory access, then do steps 2–4;
2. After each step, run `check` first and then `bench`, and use `./lab profile 00 --step n` to verify your bank conflict predictions;
3. When done, move on to [01_softmax](../course/01_softmax/README.md), going from "swapping data positions" to "combining the numeric results of many threads".

When you meet an unfamiliar word, look it up in the [Glossary](glossary.md).
