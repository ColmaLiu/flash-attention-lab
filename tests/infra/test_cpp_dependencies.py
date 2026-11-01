"""Optional headers do not burden plain CUDA; selected bundled headers stay pinned."""
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
    monkeypatch.setenv("LAB_CUTLASS_ROOT", str(tmp_path / deps.CUTLASS_PATH))
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


def test_plain_cuda_needs_no_submodule(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    monkeypatch.delenv("LAB_CUTLASS_ROOT", raising=False)
    assert deps.cpp_include_paths() == []
    assert deps.cutlass_revision() is None


def test_missing_selected_headers_report_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    monkeypatch.setenv("LAB_CUTLASS_ROOT", str(tmp_path / "missing"))
    with pytest.raises(EnvironmentUnsupported, match="LAB_CUTLASS_ROOT"):
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


def test_external_checkout_is_explicit_and_need_not_be_gitlink(tmp_path, monkeypatch):
    root = tmp_path / "external"
    monkeypatch.setenv("LAB_CUTLASS_ROOT", str(root))
    for name in ("cute/tensor.hpp", "cutlass/cutlass.h"):
        path = root / "include" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture header\n")
    assert deps.cpp_include_paths() == [root / "include"]
    assert deps.cutlass_revision() is None


def test_build_cache_separates_sources_helpers_flags_and_solution(tmp_path, monkeypatch):
    from flash_attention_lab import build, environment, registry, runtime
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "student_dir", lambda e: tmp_path / "students" / e["id"])
    monkeypatch.setattr(registry, "EXERCISES", {"lesson": {"id": "lesson"}})
    from types import SimpleNamespace
    monkeypatch.setattr(registry, "load_student", lambda e, **kwargs: SimpleNamespace(CPP_INCLUDE_DIRS=[]))
    monkeypatch.setattr(environment, "require_environment", lambda backend: None)
    monkeypatch.setattr(runtime, "cuda_home", lambda: tmp_path / "toolkit")
    monkeypatch.setattr(runtime, "cuda_development_paths", lambda toolkit: ([], tmp_path / "lib"))
    monkeypatch.delenv("LAB_CUTLASS_ROOT", raising=False)
    monkeypatch.delenv("LAB_NVCC_FLAGS", raising=False)
    files = ["flash_attention_lab/binding.cpp", "flash_attention_lab/cuda_api.h",
             "students/lesson/kernel.cu", "solutions/lesson/kernel.cu", "solutions/lesson/binding.cpp"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// " + name)
    monkeypatch.setattr(build, "_build", lambda *args: args)
    student = build.build_cuda("lesson")
    solution = build.build_cuda("lesson", solution=True)
    assert student[0] == "lesson" and solution[0] == "solution_lesson"
    assert student[-1] != solution[-1]
    assert solution[2] == str(tmp_path / "solutions/lesson/binding.cpp")
    helper = tmp_path / "students/lesson/helper.cuh"
    helper.write_text("// added helper")
    updated = build.build_cuda("lesson")
    assert updated[-1] != student[-1]
    monkeypatch.setenv("LAB_NVCC_FLAGS", "-Xptxas=-v")
    assert build.build_cuda("lesson")[-1] != updated[-1]


def test_shared_helper_edit_invalidates_solution_build(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from flash_attention_lab import build, environment, registry, runtime
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "EXERCISES", {"lesson": {"id": "lesson"}})
    monkeypatch.setattr(registry, "load_student", lambda e, **kwargs: SimpleNamespace(CPP_INCLUDE_DIRS=[]))
    monkeypatch.setattr(environment, "require_environment", lambda backend: None)
    monkeypatch.setattr(runtime, "cuda_home", lambda: tmp_path / "toolkit")
    monkeypatch.setattr(runtime, "cuda_development_paths", lambda toolkit: ([], tmp_path / "lib"))
    monkeypatch.delenv("LAB_CUTLASS_ROOT", raising=False)
    for name in ("flash_attention_lab/binding.cpp", "flash_attention_lab/cuda_api.h"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture")
    source = tmp_path / "solutions/lesson/kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text('#include "../shared/device.cuh"\n')
    helper = tmp_path / "solutions/shared/device.cuh"
    helper.parent.mkdir(parents=True)
    helper.write_text("// old helper")
    monkeypatch.setattr(build, "_build", lambda *args: args)
    before = build.build_cuda("lesson", solution=True)
    helper.write_text("// new helper")
    after = build.build_cuda("lesson", solution=True)
    assert before[-1] != after[-1]


def test_build_flags_change_result_fingerprint(monkeypatch):
    from flash_attention_lab.registry import fingerprint, CATALOG
    monkeypatch.delenv("LAB_CUTLASS_ROOT", raising=False)
    monkeypatch.delenv("LAB_NVCC_FLAGS", raising=False)
    before = fingerprint(CATALOG[0])
    monkeypatch.setenv("LAB_NVCC_FLAGS", "--use_fast_math")
    assert fingerprint(CATALOG[0]) != before
