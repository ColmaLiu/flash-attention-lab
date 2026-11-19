"""Generate local clangd commands; does not build or implement student kernels.

Run from the repository root after sourcing scripts/activate.sh:
    python scripts/configure_clangd.py
"""

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
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
        "-I" + str(root / "flash_attention_lab"),
    ]
    for path in dict.fromkeys([*map(str, cuda_includes), str(ffi_include)]):
        common += ["-isystem", path]
    # Keep host-only FFI headers in C++20; device helpers still use C++17.
    host_flags = [
        "-std=c++20",
        "-x",
        "c++",
        '-DLAB_FFI_MODULE="falab_editor"',
    ]
    cuda_flags = [
        "-std=c++17",
        "-x",
        "cuda",
        "--cuda-path=" + str(cuda_root),
        "--cuda-gpu-arch=sm_100a",
    ]
    # Include explicit header entries so opening a header first cannot borrow
    # the other language mode. Also cover learners' custom bindings/helpers.
    sources = sorted(
        p
        for folder in ("students/cuda", "solutions", "flash_attention_lab")
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".h", ".hpp")
    )
    entries = []
    for source in sources:
        header_flags = []
        owner = source.with_name("kernel.cu")
        if source.suffix == ".cuh" and owner.is_file():
            # s31 selects its shared implementation with LAB_MULTISTAGE before
            # including it. Preserve that context when the header opens alone.
            prefix, included, _ = owner.read_text().partition(
                f'#include "{source.name}"'
            )
            if included:
                header_flags = [
                    f"-D{name}={value}"
                    for name, value in re.findall(
                        r"^#define\s+(LAB_\w+)\s+(\d+)[ \t]*$", prefix, re.MULTILINE
                    )
                ]
        entries.append(
            {
                "directory": str(root),
                "file": str(source),
                "arguments": common
                + (cuda_flags if source.suffix in (".cu", ".cuh") else host_flags)
                + header_flags
                + ["-c", str(source)],
            }
        )
    return entries


def main():
    from flash_attention_lab.environment import require_course_packages
    from flash_attention_lab.runtime import cuda_home, cuda_development_paths

    # Editing only needs installed headers/toolchain, not an available GPU.
    require_course_packages()
    toolkit = cuda_home()
    if toolkit is None:
        raise RuntimeError("CUDA Toolkit not found; set LAB_CUDA_HOME to CUDA 12.8")
    cuda_includes, _ = cuda_development_paths(toolkit)
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
