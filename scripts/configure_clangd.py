"""Generate local clangd commands; does not build or implement student kernels.

Run from the repository root after sourcing scripts/activate.sh:
    python scripts/configure_clangd.py
"""

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def clang_cuda_root(toolkit):
    """Clang needs include/cuda.h and nvvm/libdevice under its CUDA root.

    NVIDIA's Conda layout puts headers under targets/x86_64-linux and links
    nvvm there. nvcc uses the environment prefix, but Clang needs this target
    directory; an extra -isystem alone does not fix installation detection.
    """
    for root in (toolkit, toolkit / "targets/x86_64-linux"):
        if (root / "include/cuda.h").is_file() and any(
            (root / "nvvm/libdevice").glob("libdevice*.bc")
        ):
            return root
    raise RuntimeError(
        f"Clang cannot find CUDA headers and libdevice under {toolkit}; "
        "select a complete CUDA 12.8 Toolkit with LAB_CUDA_HOME"
    )


def clang_cuda_includes(cuda_includes):
    """Resolve the cuRAND header implicitly included by Clang's CUDA wrapper.

    A minimal Conda Toolkit may omit cuRAND; PyTorch's CUDA wheel supplies it
    separately. Missing this header can break preamble parsing and produce
    misleading errors on STL type aliases in otherwise valid CUDA sources.
    """
    includes = list(cuda_includes)
    if any((path / "curand_mtgp32_kernel.h").is_file() for path in includes):
        return includes
    try:
        curand = distribution("nvidia-curand-cu12").locate_file("nvidia/curand/include")
    except PackageNotFoundError:
        curand = None
    if curand is None or not (curand / "curand_mtgp32_kernel.h").is_file():
        raise RuntimeError(
            "Clang CUDA parsing requires curand_mtgp32_kernel.h; install the "
            "project's CUDA PyTorch dependencies or use a Toolkit with cuRAND headers"
        )
    return [*includes, curand]


def compilation_commands(root, toolkit, cuda_includes, ffi_include):
    cuda_root = clang_cuda_root(toolkit)
    common = [
        "clang++",
        "-fPIC",
        "-I" + str(root / "include"),
    ]
    for path in dict.fromkeys([*map(str, cuda_includes), str(ffi_include)]):
        common += ["-isystem", path]
    # Keep host-only FFI headers in C++20; device helpers still use C++17.
    host_flags = [
        "-std=c++20",
        "-x",
        "c++",
        '-DLAB_FFI_MODULE="lab_editor"',
        "-DLAB_KIND_ATTENTION",
    ]
    cuda_flags = [
        "-std=c++17",
        "-x",
        "cuda",
        "--cuda-path=" + str(cuda_root),
        "--cuda-gpu-arch=sm_100a",
    ]
    # Include explicit header entries so opening a header first cannot borrow
    # the other language mode.
    sources = sorted(
        p
        for folder in ("students", "solutions", "include", "labkit/csrc")
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".h", ".hpp")
    )
    entries = []
    for source in sources:
        cuda = source.suffix in (".cu", ".cuh") or (
            source.suffix == ".h" and source.parent.name == "lab"
        )
        entries.append(
            {
                "directory": str(root),
                "file": str(source),
                "arguments": common
                + (cuda_flags if cuda else host_flags)
                + ["-c", str(source)],
            }
        )
    return entries


def main():
    from labkit.env import cuda_paths

    # Editing only needs installed headers/toolchain, not an available GPU.
    toolkit, cuda_includes, _ = cuda_paths()
    cuda_includes = clang_cuda_includes(cuda_includes)
    # Importing tvm_ffi itself installs PyTorch hooks and probes CUDA. Resolve
    # the pinned wheel's headers from metadata without loading either runtime.
    ffi_include = distribution("apache-tvm-ffi").locate_file("tvm_ffi/include")
    if not (ffi_include / "tvm/ffi/function.h").is_file():
        raise RuntimeError(
            "TVM FFI development headers are missing; reinstall apache-tvm-ffi"
        )
    entries = compilation_commands(ROOT, toolkit, cuda_includes, ffi_include)
    output = ROOT / "compile_commands.json"
    output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {len(entries)} editor commands to {output}")
    print(f"CUDA Toolkit: {toolkit}; Clang CUDA root: {clang_cuda_root(toolkit)}")
    print("Host bindings: C++20 / TVM FFI; device code: C++17 / sm_100a")
    print(
        "Restart the clangd language server. This is not student correctness validation."
    )


if __name__ == "__main__":
    main()
