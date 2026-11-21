# Environment and dependencies

## 1. What you need

| Item | Requirement | Notes |
|---|---|---|
| GPU | Compute capability 10.0 (SM100), for example B200 | `tcgen05` and TMEM in Parts 3 and 4 exist only on SM100. Consumer Blackwell (SM120) is not supported |
| System | Linux x86_64 | The install script supports only this platform |
| Driver | Supports CUDA 12.8 or newer | The top right of `nvidia-smi` shows the highest CUDA version the driver supports |
| CUDA Toolkit | 12.8 | Use the system one, or let the install script put it into the Conda environment |
| Conda | Any distribution | The course uses a separate environment and does not touch base |

Python 3.12, PyTorch 2.10.0 (CUDA 12.8 build), `apache-tvm-ffi` and so on are installed by the install script. The authoritative versions are in `environment.yml`, `requirements.txt` and `pyproject.toml`.

## 2. Installation

```bash
git clone -b v2 https://github.com/ColmaLiu/flash-attention-lab.git
cd flash-attention-lab
bash scripts/bootstrap.sh [--with-cuda] [--with-fa4]
```

The script does these things in order:

1. Creates the Conda environment `flash-attention-lab` (reuses it if it already exists);
2. Installs optional components according to the flags (see the table below);
3. Initializes two pinned submodules: `third_party/cutlass` and `third_party/flash-attention`;
4. Installs the Python dependencies in `requirements.txt`.

| Flag | What it does |
|---|---|
| `--with-cuda` | Installs CUDA Toolkit 12.8 and GCC 13 into the environment. Use it when the system has no suitable `nvcc` or host compiler |
| `--with-fa4` | Installs FlashAttention-4 from `third_party/flash-attention/flash_attn/cute`, as the performance baseline for the attention labs |

An existing checkout can run `bash scripts/bootstrap.sh --with-fa4` later at any time.

## 3. Usage

`./lab` activates the Conda environment automatically; you do not need to `conda activate` by hand. To run other commands (such as pytest) in the same environment:

```bash
source scripts/activate.sh
python -m pytest -q tests/test_labkit.py
```

### 3.1 Environment variables

| Variable | What it does | Default |
|---|---|---|
| `LAB_GPU` | Selects one GPU (index or UUID) | The first one in `CUDA_VISIBLE_DEVICES`; 0 if neither is set |
| `LAB_CONDA_ENV` | Conda environment name, or an absolute path | `flash-attention-lab` |
| `LAB_CONDA_EXE` | The conda executable | `$CONDA_EXE` or `conda` on PATH |
| `LAB_CUDA_HOME` | Which CUDA Toolkit to use | `$CUDA_HOME`, then `nvcc` on PATH |
| `LAB_NCU`, `LAB_SANITIZER` | Paths of `ncu` and `compute-sanitizer` | First the Toolkit directory, then PATH |
| `LAB_NVCC_FLAGS` | Extra nvcc flags, for example `-Xptxas -v` | Empty |

**Use only one GPU at a time**. `./lab` sets `CUDA_VISIBLE_DEVICES` to the selected one. If you have already set `CUDA_VISIBLE_DEVICES`, `LAB_GPU` must be one of those devices. When several people share the machine, find an idle card with `nvidia-smi` first:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
LAB_GPU=3 ./lab check 00
```

When someone else's kernels run on the same card as yours, the `bench` numbers cannot be trusted.

### 3.2 Checking the environment

```bash
./lab doctor
```

It checks, one by one: the selected GPU and its compute capability, the `nvcc` version, PyTorch and its CUDA version, `apache-tvm-ffi`, the FA4 baseline (optional), `ninja`, `compute-sanitizer`, `ncu` and `cuobjdump`.

Keep three "CUDA versions" apart:

| Version | How to see it | Meaning |
|---|---|---|
| Driver-supported version | `nvidia-smi` | The highest CUDA version the driver can run |
| Toolkit | `nvcc --version` | The compiler used to compile kernels |
| PyTorch runtime | `torch.version.cuda` | The CUDA runtime bundled with PyTorch |

The `nvcc` that `./lab doctor` shows is the one actually used to compile the course kernels.

## 4. Compilation and caching

- Kernels are compiled by `nvcc` with `-O3 -lineinfo -gencode=arch=compute_100a,code=sm_100a`, and exchange tensors with Python through [TVM FFI](https://github.com/apache/tvm-ffi). The PyTorch C++ extension build is not used.
- Host-side binding code uses C++20, and device code uses C++17. The GCC 13 provided by `--with-cuda` supports both.
- Build output is cached in `.cache/build/`, keyed by the source, the headers from this repository it `#include`s, the compile flags and the FFI version. Changing the code triggers a rebuild automatically.
- The first compile may take tens of seconds; later `check` and `bench` runs use the cache directly.

## 5. Where output goes

All of these directories are ignored by Git:

| Directory | Contents |
|---|---|
| `.cache/` | Build cache, CUDA JIT cache |
| `artifacts/bench/` | Full results of every `bench` run (JSON, with a source fingerprint) |
| `artifacts/profiles/` | Nsight Compute reports |
| `artifacts/sass/` | SASS exported by `./lab sass` |
| `progress/` | Your pass status per lab and per step (read by `./lab list`) |

The progress records include a source fingerprint. After you change the code, `./lab list` shows "code changed since last check".
