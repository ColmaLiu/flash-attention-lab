"""Pinned upstream sources support reproducible design and provenance reading."""

import subprocess
import pytest
from flash_attention_lab import cpp_dependencies as deps
from flash_attention_lab.errors import EnvironmentUnsupported


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def make_source(root, name):
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init")
    for filename in deps.SOURCES[name]["required"]:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture source\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "Fixture source")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # Commits belong exclusively to disposable test repositories.
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    git(tmp_path, "init")
    sources = {}
    for name, spec in deps.SOURCES.items():
        root = tmp_path / spec["path"]
        revision = make_source(root, name)
        git(
            tmp_path,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{revision},{spec['path']}",
        )
        sources[name] = (root, revision)
    return tmp_path, sources


@pytest.mark.parametrize("name", deps.SOURCES)
def test_missing_source_is_not_ready(tmp_path, monkeypatch, name):
    monkeypatch.setattr(deps, "ROOT", tmp_path)
    with pytest.raises(EnvironmentUnsupported, match="gitlink missing.*bootstrap.sh"):
        deps.require_source(name)
    assert deps.source_info(name)["commit"] is None


def test_initialized_sources_use_index_versions_and_shared_headers(checkout):
    _, sources = checkout
    for name, (root, revision) in sources.items():
        assert deps.require_source(name) == root
        assert deps.source_info(name) == {
            "path": str(root),
            "expected_commit": revision,
            "commit": revision,
            "state": "READY",
        }
    cutlass = sources["cutlass"][0]
    assert deps.cpp_include_paths() == [
        cutlass / "include",
        cutlass / "tools/util/include",
    ]
    assert set(deps.source_snapshot()) == set(deps.SOURCES)


@pytest.mark.parametrize("name", deps.SOURCES)
def test_checkout_must_match_parent_gitlink(checkout, name):
    parent, _ = checkout
    path = deps.SOURCES[name]["path"]
    git(parent, "update-index", "--cacheinfo", f"160000,{'1' * 40},{path}")
    with pytest.raises(EnvironmentUnsupported, match="differs from the parent gitlink"):
        deps.require_source(name)


@pytest.mark.parametrize("name", deps.SOURCES)
def test_source_without_gitlink_is_not_accepted(checkout, name):
    parent, _ = checkout
    git(parent, "update-index", "--force-remove", deps.SOURCES[name]["path"])
    with pytest.raises(EnvironmentUnsupported, match="gitlink missing"):
        deps.require_source(name)


def test_missing_required_header_is_not_accepted(checkout):
    _, sources = checkout
    (sources["cutlass"][0] / "include/cute/tensor.hpp").unlink()
    with pytest.raises(EnvironmentUnsupported, match="required source files missing"):
        deps.require_source("cutlass")


def test_uninitialized_submodule_cannot_inherit_parent_revision(checkout):
    parent, sources = checkout
    # A subdirectory with source files is not an initialized submodule.
    import shutil

    shutil.rmtree(sources["cutlass"][0] / ".git")
    info = deps.source_info("cutlass")
    assert info["commit"] is None and info["state"] == "ENV_UNSUPPORTED"
    with pytest.raises(EnvironmentUnsupported, match="not initialized"):
        deps.require_source("cutlass")


@pytest.fixture
def build_fixture(tmp_path, monkeypatch):
    from flash_attention_lab import build, environment, registry, runtime

    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "student_dir", lambda e: tmp_path / "students" / e["id"])
    monkeypatch.setattr(registry, "EXERCISES", {"lesson": {"id": "lesson"}})
    monkeypatch.setattr(environment, "require_environment", lambda backend: None)
    monkeypatch.setattr(runtime, "cuda_home", lambda: tmp_path / "toolkit")
    monkeypatch.setattr(
        runtime, "cuda_development_paths", lambda toolkit: ([], tmp_path / "lib")
    )
    include = tmp_path / "third_party/cutlass/include"
    include.mkdir(parents=True)
    monkeypatch.setattr(deps, "cpp_include_paths", lambda: [include])
    monkeypatch.delenv("LAB_NVCC_FLAGS", raising=False)
    for name in (
        "flash_attention_lab/binding.cpp",
        "flash_attention_lab/cuda_api.h",
        "students/lesson/kernel.cu",
        "solutions/lesson/kernel.cu",
        "solutions/lesson/binding.cpp",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// " + name)
    monkeypatch.setattr(build, "_build", lambda *args: args)
    return tmp_path, build


def test_build_cache_separates_sources_helpers_flags_and_solution(
    build_fixture, monkeypatch
):
    root, build = build_fixture
    student = build.build_cuda("lesson")
    solution = build.build_cuda("lesson", solution=True)
    assert student[0] == "lesson" and solution[0] == "solution_lesson"
    assert student[-1] != solution[-1]
    assert not any("third_party" in path for path in student[5])
    assert (
        student[5] == solution[5]
    )  # Both use the same CUDA include directories, with no upstream kernel headers.
    assert solution[2] == str(root / "solutions/lesson/binding.cpp")
    helper = root / "students/lesson/helper.cuh"
    helper.write_text("// added helper")
    updated = build.build_cuda("lesson")
    assert updated[-1] != student[-1]
    monkeypatch.setenv("LAB_NVCC_FLAGS", "-Xptxas=-v")
    assert build.build_cuda("lesson")[-1] != updated[-1]


def test_shared_helper_edit_invalidates_solution_build(build_fixture):
    root, build = build_fixture
    (root / "solutions/lesson/kernel.cu").write_text(
        '#include "../shared/device.cuh"\n'
    )
    helper = root / "solutions/shared/device.cuh"
    helper.parent.mkdir(parents=True)
    helper.write_text("// old helper")
    before = build.build_cuda("lesson", solution=True)
    helper.write_text("// new helper")
    assert build.build_cuda("lesson", solution=True)[-1] != before[-1]


def test_build_flags_change_result_fingerprint(monkeypatch):
    from flash_attention_lab.registry import fingerprint, CATALOG

    monkeypatch.delenv("LAB_NVCC_FLAGS", raising=False)
    before = fingerprint(CATALOG[0])
    monkeypatch.setenv("LAB_NVCC_FLAGS", "--use_fast_math")
    assert fingerprint(CATALOG[0]) != before
