"""Project-wide C++ headers supplied by the CUTLASS Git submodule."""
from pathlib import Path
import subprocess
from .errors import EnvironmentUnsupported

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_PATH = "third_party/cutlass"


def cutlass_root():
    return ROOT / CUTLASS_PATH


def _git(*args):
    try:
        result = subprocess.run(["git", *map(str, args)], text=True, capture_output=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cutlass_revision():
    root = cutlass_root()
    # Without this check, git can accidentally report the parent checkout's HEAD.
    if not (root / ".git").exists():
        return None
    return _git("-C", root, "rev-parse", "--verify", "HEAD")


def cpp_include_paths():
    """Use the gitlink as the only dependency version lock, for every CUDA build."""
    root = cutlass_root()
    setup = "run bash scripts/bootstrap.sh or git submodule update --init --recursive"
    if (not (root / "include/cute/tensor.hpp").is_file()
            or not (root / "include/cutlass/cutlass.h").is_file()):
        raise EnvironmentUnsupported(f"CUTLASS/CuTe submodule not initialized; {setup}")
    entry = (_git("-C", ROOT, "ls-files", "--stage", "--", CUTLASS_PATH) or "").split()
    if len(entry) != 4 or entry[0] != "160000" or entry[2] != "0":
        raise EnvironmentUnsupported("CUTLASS gitlink missing or conflicted; use a Git checkout with its submodules")
    if cutlass_revision() != entry[1]:
        raise EnvironmentUnsupported(f"CUTLASS checkout differs from the project gitlink; {setup}")
    return [root / "include"]
