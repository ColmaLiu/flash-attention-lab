"""Optional C++ CUTLASS/CuTe headers; plain CUDA/PTX needs no submodule."""
import os
from pathlib import Path
import subprocess
from .errors import EnvironmentUnsupported

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_PATH = "third_party/cutlass"


def cutlass_root():
    """The optional configured tree, or the conventional repository location."""
    configured = os.environ.get("LAB_CUTLASS_ROOT")
    return Path(configured).expanduser().resolve() if configured else ROOT / CUTLASS_PATH


def _git(*args):
    try:
        result = subprocess.run(["git", *map(str, args)], text=True, capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cutlass_revision():
    if not os.environ.get("LAB_CUTLASS_ROOT"):
        return None
    root = cutlass_root()
    # Without this check git can accidentally report the parent checkout's HEAD.
    if not (root / ".git").exists():
        return None
    return _git("-C", root, "rev-parse", "--verify", "HEAD")


def cpp_include_paths():
    """Enable headers only when LAB_CUTLASS_ROOT explicitly names a tree.

    The bundled submodule retains its gitlink version lock. An external checkout
    is a learner-selected dependency; its actual revision is recorded at runtime.
    """
    if not os.environ.get("LAB_CUTLASS_ROOT"):
        return []
    root = cutlass_root()
    if (not (root / "include/cute/tensor.hpp").is_file()
            or not (root / "include/cutlass/cutlass.h").is_file()):
        raise EnvironmentUnsupported("LAB_CUTLASS_ROOT lacks CUTLASS/CuTe headers; select a complete checkout")
    if root.resolve() == (ROOT / CUTLASS_PATH).resolve():
        entry = (_git("-C", ROOT, "ls-files", "--stage", "--", CUTLASS_PATH) or "").split()
        if len(entry) != 4 or entry[0] != "160000" or entry[2] != "0":
            raise EnvironmentUnsupported("CUTLASS gitlink missing or conflicted; select a complete Git checkout")
        if cutlass_revision() != entry[1]:
            raise EnvironmentUnsupported("CUTLASS checkout differs from the project gitlink; run git submodule update --init --recursive -- third_party/cutlass")
    return [root / "include"]
