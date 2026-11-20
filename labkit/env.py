"""GPU selection and toolchain discovery. Must run before importing torch."""

import os
import re
import shutil
import subprocess
from pathlib import Path

from . import ROOT, EnvironmentError_


def selected_device(environ):
    """Pick one GPU: LAB_GPU narrows an inherited CUDA_VISIBLE_DEVICES mask."""
    requested = environ.get("LAB_GPU")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if requested is not None and not re.fullmatch(
        r"[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+", requested
    ):
        raise EnvironmentError_("LAB_GPU must be one GPU index or UUID")
    if visible is None:
        return requested or "0"
    allowed = [s.strip() for s in visible.split(",") if s.strip()]
    if requested is not None:
        if requested not in allowed:
            raise EnvironmentError_(
                "LAB_GPU must name a device already in CUDA_VISIBLE_DEVICES"
            )
        return requested
    if not allowed:
        raise EnvironmentError_("CUDA_VISIBLE_DEVICES hides every GPU")
    return allowed[0]


def cuda_home():
    for key in ("LAB_CUDA_HOME", "CUDA_HOME"):
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser().resolve()
    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parent.parent if nvcc else None


def tool(name, override=None):
    if override and os.environ.get(override):
        return shutil.which(os.path.expanduser(os.environ[override]))
    # Prefer the toolkit that compiles the kernels over other CUDA versions on PATH.
    home = cuda_home()
    if home and shutil.which(str(home / "bin" / name)):
        return shutil.which(str(home / "bin" / name))
    return shutil.which(name)


def require_tool(name, override=None):
    found = tool(name, override)
    if not found:
        hint = f" or set {override}" if override else ""
        raise EnvironmentError_(f"{name} not found; add it to PATH{hint}")
    return found


def cuda_paths():
    """(include dirs, library dir) for system and Conda CUDA layouts."""
    home = cuda_home()
    if home is None:
        raise EnvironmentError_("nvcc not found; run bash scripts/bootstrap.sh")
    roots = [home, home / "targets" / "x86_64-linux"]
    includes = [p / "include" for p in roots if (p / "include").is_dir()]
    if not any((p / "cuda_runtime.h").is_file() for p in includes):
        raise EnvironmentError_("CUDA headers missing (cuda_runtime.h)")
    for p in roots:
        for sub in ("lib64", "lib"):
            if (p / sub / "libcudart.so").is_file():
                return home, includes, p / sub
    raise EnvironmentError_("CUDA library libcudart.so missing")


def configure():
    """Select one GPU and set cache directories. Call before importing torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device(os.environ)
    home = cuda_home()
    if home:
        os.environ["CUDA_HOME"] = str(home)
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ.setdefault("TVM_FFI_CACHE_DIR", str(ROOT / ".cache" / "tvm_ffi"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache" / "cuda"))


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise EnvironmentError_(
            "PyTorch sees no GPU; check the driver and CUDA_VISIBLE_DEVICES/LAB_GPU"
        )
    cap = torch.cuda.get_device_capability()
    if cap != (10, 0):
        raise EnvironmentError_(
            f"this course targets SM100 (B200, capability 10.0); got {cap}"
        )


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"


def doctor():
    """Human-readable environment report; returns True when usable."""
    ok = True
    lines = []

    def item(name, value, good=True):
        nonlocal ok
        ok &= good
        lines.append(f"  {'ok ' if good else 'BAD'}  {name:<20} {value}")

    item("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    home = cuda_home()
    nvcc = _run([str(home / "bin/nvcc"), "--version"]) if home else ""
    m = re.search(r"release (\d+\.\d+)", nvcc)
    item("nvcc", m.group(1) if m else "not found", bool(m))
    try:
        import torch

        item("torch", f"{torch.__version__} (CUDA {torch.version.cuda})")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            good = (p.major, p.minor) == (10, 0)
            item(
                "GPU",
                f"{p.name} sm_{p.major}{p.minor}, {p.multi_processor_count} SMs",
                good,
            )
        else:
            item("GPU", "not visible to PyTorch", False)
    except ImportError as exc:
        item("torch", str(exc), False)
    try:
        import tvm_ffi  # noqa: F401
        from importlib.metadata import version

        item("apache-tvm-ffi", version("apache-tvm-ffi"))
    except ImportError:
        item("apache-tvm-ffi", "missing", False)
    try:
        import flash_attn.cute  # noqa: F401

        item("FA4 baseline", "available")
    except Exception:
        lines.append("  --   FA4 baseline        not installed (optional)")
    for name, override in (
        ("ninja", None),
        ("compute-sanitizer", "LAB_SANITIZER"),
        ("ncu", "LAB_NCU"),
        ("cuobjdump", None),
    ):
        found = tool(name, override)
        item(name, found or "not found", bool(found) or name in ("ncu",))
    print("\n".join(lines))
    return ok
