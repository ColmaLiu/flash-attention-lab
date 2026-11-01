"""Course source dependencies; the parent Git index is the version authority."""
from pathlib import Path
import subprocess
from .errors import EnvironmentUnsupported

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "cutlass": {
        "path": "third_party/cutlass",
        "required": ("include/cute/tensor.hpp", "include/cutlass/cutlass.h",
                     "tools/util/include/cutlass/util/packed_stride.hpp", "LICENSE.txt"),
    },
    "flash-attention": {
        "path": "third_party/flash-attention",
        "required": ("flash_attn/cute/interface.py", "flash_attn/cute/flash_fwd_sm100.py", "LICENSE"),
    },
}


def _git(*args):
    try:
        result = subprocess.run(["git", *map(str, args)], text=True, capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_info(name):
    """Inspect a source without raising, so doctor can report incomplete clones."""
    if name not in SOURCES:
        raise ValueError(f"unknown source dependency: {name}")
    spec = SOURCES[name]
    root = ROOT / spec["path"]
    lines = (_git("-C", ROOT, "ls-files", "--stage", "--", spec["path"]) or "").splitlines()
    entry = lines[0].split() if len(lines) == 1 else []
    expected = entry[1] if len(entry) == 4 and entry[0] == "160000" and entry[2] == "0" else None
    # Otherwise rev-parse could accidentally return the parent's HEAD.
    actual = _git("-C", root, "rev-parse", "--verify", "HEAD") if (root / ".git").exists() else None
    info = {"path": str(root), "expected_commit": expected, "commit": actual, "state": "READY"}
    if expected is None:
        reason = "parent gitlink missing or conflicted; use a complete Git clone"
    elif actual is None:
        reason = "submodule is not initialized"
    elif actual != expected:
        reason = "checkout differs from the parent gitlink"
    else:
        missing = [name for name in spec["required"] if not (root / name).is_file()]
        reason = "required source files missing: " + ", ".join(missing) if missing else None
    if reason:
        info.update(state="ENV_UNSUPPORTED", message=f"{name}: {reason}; run bash scripts/bootstrap.sh")
    return info


def source_snapshot():
    return {name: source_info(name) for name in SOURCES}


def require_source(name):
    info = source_info(name)
    if info["state"] != "READY":
        raise EnvironmentUnsupported(info["message"])
    return Path(info["path"])


def cpp_include_paths():
    root = require_source("cutlass")
    return [root / "include", root / "tools/util/include"]
