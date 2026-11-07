"""CUDA C++ extension build, selected after inspecting the active GPU."""

import hashlib
import os
import re
import shlex
from functools import lru_cache
from pathlib import Path
import torch
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
    # Changes to local helpers invalidate the in-process
    # module too. Ninja alone is insufficient: cpp_extension also caches loaded modules.
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
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = toolkit
    build_dir = ROOT / ".cache/build" / exercise_id / tag
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    return cpp_extension.load(
        name="falab_" + exercise_id + "_" + tag,
        sources=[binding, source],
        extra_include_paths=list(includes),
        # Driver API is needed for host-side TMA tensor-map descriptors.
        extra_ldflags=["-L" + shlex.quote(cuda_library), "-lcuda"],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=list(flags),
        build_directory=str(build_dir),
        verbose=bool(os.getenv("LAB_BUILD_VERBOSE")),
    )


def descriptors(tensors):
    codes = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int32: 3,
        torch.uint8: 4,
        torch.int64: 5,
    }
    return {
        n: (t.data_ptr(), list(t.shape), codes[t.dtype]) for n, t in tensors.items()
    }
