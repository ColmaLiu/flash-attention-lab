"""Portable runtime configuration; no CUDA initialization or torch import."""

import os
from pathlib import Path
import re
import shutil

from .errors import EnvironmentUnsupported

ROOT = Path(__file__).resolve().parents[1]


def selected_device(environ):
    """Narrow an inherited scheduler/container mask; never expand it."""
    requested = environ.get("LAB_GPU")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if requested is not None and not re.fullmatch(
        r"(?:[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+)", requested
    ):
        raise EnvironmentUnsupported("LAB_GPU must be one GPU index or UUID")
    if visible is None:
        return requested or "0"
    allowed = [s.strip() for s in visible.split(",") if s.strip()]
    if requested is not None:
        if requested not in allowed:
            raise EnvironmentUnsupported(
                "LAB_GPU must name a device already in CUDA_VISIBLE_DEVICES"
            )
        return requested
    return allowed[0] if allowed else ""


def cuda_home():
    """Explicit overrides take precedence; an invalid override is not bypassed."""
    for key in ("LAB_CUDA_HOME", "CUDA_HOME"):
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser().resolve()
    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parent.parent if nvcc else None


def tool_path(name, override=None):
    if override and os.environ.get(override):
        return shutil.which(os.path.expanduser(os.environ[override]))
    found = shutil.which(name)
    if found:
        return found
    toolkit = cuda_home()
    return shutil.which(str(toolkit / "bin" / name)) if toolkit else None


def cuda_development_paths(toolkit):
    """Support both system Toolkit and NVIDIA Conda target-directory layouts."""
    roots = [toolkit, toolkit / "targets" / "x86_64-linux"]
    includes = [p / "include" for p in roots if (p / "include").is_dir()]
    libraries = [p / suffix for p in roots for suffix in ("lib64", "lib")]
    if not any((p / "cuda_runtime.h").is_file() for p in includes):
        raise EnvironmentUnsupported(
            "CUDA development headers missing; install Toolkit 12.8 or bootstrap --with-cuda"
        )
    library = next((p for p in libraries if (p / "libcudart.so").is_file()), None)
    if library is None:
        raise EnvironmentUnsupported(
            "CUDA development library libcudart.so missing; install Toolkit 12.8 or bootstrap --with-cuda"
        )
    return includes, library


def require_tool(name, override):
    found = tool_path(name, override)
    if not found:
        raise EnvironmentUnsupported(
            f"{name} not found; add it to PATH or set {override} to its executable"
        )
    return found


def configure_runtime():
    # Called before torch is imported by CLI, pytest, and standalone probes.
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device(os.environ)
    toolkit = cuda_home()
    if toolkit:
        os.environ["CUDA_HOME"] = str(toolkit)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(ROOT / ".cache" / "torch_extensions")
    )
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache" / "cuda"))
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *[p for p in paths if p and p != str(ROOT)]]
    )
