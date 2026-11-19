"""CUDA C++ extension build, selected after inspecting the active GPU."""

import hashlib
import importlib.metadata
import os
import re
import shlex
from functools import lru_cache
from pathlib import Path
from .registry import ROOT, student_dir


def build_cuda(exercise_id, *, solution=False):
    from .environment import require_environment
    from .runtime import cuda_home, cuda_development_paths
    from .registry import EXERCISES

    require_environment("cuda")
    toolkit = cuda_home()
    cuda_includes, cuda_library = cuda_development_paths(toolkit)
    source = (
        ROOT / "flash_attention_lab/infra_smoke.cu"
        if exercise_id == "infra_smoke"
        else (
            ROOT / "solutions" / exercise_id
            if solution
            else student_dir(EXERCISES[exercise_id])
        )
        / "kernel.cu"
    )
    includes = tuple(map(str, [ROOT / "flash_attention_lab", *cuda_includes]))
    flags = (
        "-O3",
        "-lineinfo",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        *shlex.split(os.environ.get("LAB_NVCC_FLAGS", "")),
    )
    binding = source.parent / "binding.cpp"
    if not binding.is_file():
        binding = ROOT / "flash_attention_lab/binding.cpp"
    # Give changed helpers and FFI/toolchain versions a distinct shared library
    # name; a loaded module must never conceal edits in the same process.
    digest = hashlib.sha256()
    sources = [binding, ROOT / "flash_attention_lab/cuda_api.h", source]
    sources += sorted(
        p for p in source.parent.rglob("*") if p.suffix in (".h", ".cuh", ".cu")
    )
    # Quoted helpers can live in another lesson (e.g. s32 shares s31's
    # implementation). Follow those includes so the loaded-module cache cannot
    # conceal edits to a transitive helper in the same Python process.
    pending = list(dict.fromkeys(sources))
    visited = set()
    while pending:
        path = pending.pop()
        path = path.resolve()
        if path in visited:
            continue
        visited.add(path)
        content = path.read_bytes()
        digest.update(str(path).encode())
        digest.update(content)
        for name in re.findall(
            rb'^\s*#\s*include\s*"([^"\n]+)"', content, re.MULTILINE
        ):
            for directory in (path.parent, *(Path(p) for p in includes)):
                helper = (directory / name.decode()).resolve()
                if helper.is_file():
                    if helper.is_relative_to(ROOT.resolve()):
                        pending.append(helper)
                    break
    configuration = (
        str(toolkit),
        str(cuda_library),
        includes,
        flags,
        os.environ.get("CXX", "c++"),
        os.environ.get("CC", ""),
        importlib.metadata.version("apache-tvm-ffi"),
        Path(__file__).read_bytes(),
    )
    digest.update(repr(configuration).encode())
    tag = digest.hexdigest()[:16]
    build_id = "solution_" + exercise_id if solution else exercise_id
    return _build(
        build_id,
        str(source),
        str(binding),
        str(toolkit),
        str(cuda_library),
        includes,
        flags,
        tag,
    )


@lru_cache(None)
def _build(exercise_id, source, binding, toolkit, cuda_library, includes, flags, tag):
    from tvm_ffi import cpp

    build_dir = ROOT / ".cache/build" / exercise_id / tag
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    name = "falab_" + exercise_id + "_" + tag
    selected = {
        "CUDA_HOME": toolkit,
        # TVM FFI otherwise probes the first physical GPU, ignoring our mask.
        "TVM_FFI_CUDA_ARCH_LIST": "10.0a",
    }
    previous = {key: os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        cuda_flags = list(flags)
        cuda_flags.remove("-gencode=arch=compute_100a,code=sm_100a")
        if os.environ.get("CC"):
            cuda_flags += ["-ccbin", shlex.quote(os.environ["CC"])]
        return cpp.load(
            name=name,
            sources=[binding, source],
            extra_include_paths=list(includes),
            extra_ldflags=["-L" + shlex.quote(cuda_library), "-lcuda"],
            extra_cflags=[
                "-O2",
                "-std=c++20",
                "-fvisibility=hidden",
                f'-DLAB_FFI_MODULE=\\"{name}\\"',
            ],
            # Device sources retain C++17; only the FFI host binding uses C++20.
            extra_cuda_cflags=cuda_flags,
            build_directory=str(build_dir),
            backend="cuda",
            keep_module_alive=True,
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
