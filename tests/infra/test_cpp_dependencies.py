"""Project headers must be initialized and match the parent Git gitlink."""
import subprocess
import pytest
from flash_attention_lab import cpp_dependencies as deps
from flash_attention_lab.errors import EnvironmentUnsupported


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
        text=True, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # Commits belong exclusively to disposable test repositories.
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    git(tmp_path, "init")
    root = deps.cutlass_root()
    root.mkdir(parents=True)
    git(root, "init")
    for name in ("cute/tensor.hpp", "cutlass/cutlass.h"):
        p = root / "include" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("// fixture header\n")
    git(root, "add", "include")
    git(root, "commit", "-m", "Fixture headers")
    revision = git(root, "rev-parse", "HEAD")
    git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{revision},{deps.CUTLASS_PATH}")
    return tmp_path, root, revision


def test_missing_submodule_reports_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    with pytest.raises(EnvironmentUnsupported, match="bootstrap.sh"):
        deps.cpp_include_paths()
    assert deps.cutlass_revision() is None


def test_initialized_submodule_uses_shared_include(checkout):
    _, root, revision = checkout
    assert deps.cpp_include_paths() == [root / "include"]
    assert deps.cutlass_revision() == revision


def test_checkout_must_match_gitlink(checkout):
    parent, _, _ = checkout
    git(parent, "update-index", "--cacheinfo", f"160000,{'1' * 40},{deps.CUTLASS_PATH}")
    with pytest.raises(EnvironmentUnsupported, match="differs from the project gitlink"):
        deps.cpp_include_paths()


def test_headers_without_gitlink_are_not_accepted(checkout):
    parent, _, _ = checkout
    git(parent, "update-index", "--force-remove", deps.CUTLASS_PATH)
    with pytest.raises(EnvironmentUnsupported, match="gitlink missing"):
        deps.cpp_include_paths()
