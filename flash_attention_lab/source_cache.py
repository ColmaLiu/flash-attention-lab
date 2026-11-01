"""Build a patched FA4 source tree without modifying its pinned submodule."""
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .cpp_dependencies import require_source
from .errors import EnvironmentUnsupported
from .registry import ROOT


def _manifest(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def _verify_cache(root):
    try:
        expected = json.loads((root / "manifest.json").read_text())
        valid = bool(expected) and _manifest(root) == expected
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise EnvironmentUnsupported(f"FA4 compiler cache is incomplete or modified: {root}; remove this cache directory and start a fresh lab process")


def prepare_flash_attention_source():
    upstream = require_source("flash-attention")
    package = upstream / "flash_attn/cute"
    patch = ROOT / "patches/flash-attention/rescale-threshold.patch"
    patch_bytes = patch.read_bytes()
    files = {path.relative_to(package): path.read_bytes()
             for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    digest = hashlib.sha256(b"fa-lab-fa4-cache-v2\0" + patch_bytes)
    for name, content in files.items():
        digest.update(str(name).encode() + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    parent = ROOT / ".cache/sources/flash-attention"
    target = parent / digest.hexdigest()
    if target.is_dir():
        _verify_cache(target)
        return target
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=parent) as temporary:
        staging = Path(temporary)
        # Hash and materialize the same immutable byte snapshot. Editing the
        # upstream tree or patch concurrently cannot change a cache key's meaning.
        for name, content in files.items():
            destination = staging / "flash_attn/cute" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        patch_input = staging / "input.patch"
        patch_input.write_bytes(patch_bytes)
        # git apply also works outside a repository. Prevent discovery of the
        # enclosing lab repository so all patch paths resolve within staging.
        environment = os.environ.copy()
        for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            environment.pop(variable, None)
        environment["GIT_CEILING_DIRECTORIES"] = str(parent.resolve())
        for flags in (("--check",), ("--whitespace=nowarn",)):
            result = subprocess.run(["git", "apply", *flags, str(patch_input.resolve())],
                                    cwd=staging, env=environment, capture_output=True, text=True)
            if result.returncode:
                raise EnvironmentUnsupported("FA4 source patch cannot be applied: " + result.stderr.strip())
        patch_input.unlink()
        (staging / "manifest.json").write_text(json.dumps(_manifest(staging), sort_keys=True) + "\n")
        try:
            staging.rename(target)
        except OSError as exc:
            # Concurrent preparations of the same content may finish together.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not target.is_dir():
                raise
    _verify_cache(target)
    return target
