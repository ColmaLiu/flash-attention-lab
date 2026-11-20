"""Compile a kernel.cu (student or solution) into a TVM FFI module."""

import hashlib
import os
import re
import shlex
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from . import ROOT, LabError
from .env import cuda_paths

INCLUDE = ROOT / "include"
BINDING = ROOT / "labkit" / "csrc" / "binding.cpp"
NVCC_FLAGS = (
    "-O3",
    "-std=c++17",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100a,code=sm_100a",
)


def source_dir(lab, solution):
    return lab.solution_dir() if solution else lab.student_dir()


def dependencies(path):
    """The file plus every repository file it #includes with quotes, recursively."""
    seen, pending = [], [Path(path).resolve()]
    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.append(p)
        text = p.read_text(errors="replace")
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.MULTILINE):
            for base in (p.parent, INCLUDE):
                q = (base / name).resolve()
                if q.is_file() and q.is_relative_to(ROOT):
                    pending.append(q)
                    break
    return sorted(seen)


def fingerprint(lab, solution=False):
    """Hash of the lab's own sources and the headers they include."""
    h = hashlib.sha256()
    for p in dependencies(source_dir(lab, solution) / "kernel.cu"):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build(lab, solution=False):
    kernel = source_dir(lab, solution) / "kernel.cu"
    if not kernel.is_file():
        raise LabError(f"missing {kernel.relative_to(ROOT)}")
    extra = tuple(shlex.split(os.environ.get("LAB_NVCC_FLAGS", "")))
    h = hashlib.sha256()
    for p in [BINDING, *dependencies(kernel)]:
        h.update(str(p).encode())
        h.update(p.read_bytes())
    h.update(repr((NVCC_FLAGS, extra, version("apache-tvm-ffi"), lab.kind)).encode())
    tag = h.hexdigest()[:16]
    who = "solution" if solution else "student"
    return _load(f"{who}_{lab.id}", str(kernel), lab.kind, extra, tag)


def build_dir(lab, solution=False):
    who = "solution" if solution else "student"
    return ROOT / ".cache" / "build" / f"{who}_{lab.id}"


@lru_cache(None)
def _load(name, kernel, kind, extra, tag):
    from tvm_ffi import cpp

    home, includes, libdir = cuda_paths()
    out = ROOT / ".cache" / "build" / name / tag
    out.mkdir(parents=True, exist_ok=True)
    module = f"lab_{name}_{tag}"
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
    cuda_flags = [f for f in NVCC_FLAGS if not f.startswith("-gencode")] + list(extra)
    try:
        return cpp.load(
            name=module,
            cpp_files=[str(BINDING)],
            cuda_files=[kernel],
            extra_include_paths=[str(INCLUDE), *map(str, includes)],
            extra_cflags=[
                "-O2",
                "-std=c++20",
                f"-DLAB_KIND_{kind.upper()}",
                f'-DLAB_FFI_MODULE=\\"{module}\\"',
            ],
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=[f"-L{libdir}", "-lcuda"],
            build_directory=str(out),
        )
    except RuntimeError as exc:
        raise LabError(_compile_error(str(exc), kernel)) from None


def _compile_error(text, kernel):
    """Keep the compiler diagnostics, drop the build-system noise."""
    lines = [
        line
        for line in text.splitlines()
        if ("error" in line or "warning" in line or line.startswith(" "))
        and "ninja" not in line
    ]
    body = "\n".join(lines[:40]) or text[-4000:]
    return f"compilation failed for {Path(kernel).relative_to(ROOT)}:\n{body}"
