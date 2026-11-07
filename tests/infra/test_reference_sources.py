"""Public clones obtain source dependencies from their committed Git gitlinks."""

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
            "protocol.file.allow=always",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def test_source_submodules_match_parent_gitlinks_and_keep_licenses():
    for name, spec in deps.SOURCES.items():
        root = deps.require_source(name)
        info = deps.source_info(name)
        assert info["commit"] == info["expected_commit"]
        assert len(info["commit"]) == 40
        assert any("LICENSE" in filename for filename in spec["required"])
        assert all((root / filename).is_file() for filename in spec["required"])


def test_clean_clone_requires_and_initializes_only_declared_submodules(
    tmp_path, monkeypatch
):
    template = tmp_path / "template"
    template.mkdir()
    git(template, "init")
    for name, spec in deps.SOURCES.items():
        upstream = tmp_path / "upstream" / name
        upstream.mkdir(parents=True)
        git(upstream, "init")
        for filename in spec["required"]:
            path = upstream / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source\n")
        # Nested unrelated sources are intentionally not initialized by bootstrap.
        (upstream / ".gitmodules").write_text(
            '[submodule "unneeded"]\n\tpath = unneeded\n\turl = https://invalid.example/unneeded.git\n'
        )
        git(upstream, "add", ".")
        git(
            upstream,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{'1' * 40},unneeded",
        )
        git(upstream, "commit", "-m", "Fixture source")
        git(template, "submodule", "add", str(upstream), spec["path"])
    git(template, "commit", "-am", "Fixture source locks")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--no-recurse-submodules", str(template), str(clone))
    monkeypatch.setattr(deps, "ROOT", clone)
    for name in deps.SOURCES:
        with pytest.raises(EnvironmentUnsupported, match="not initialized"):
            deps.require_source(name)
    git(
        clone,
        "submodule",
        "update",
        "--init",
        "--",
        *(spec["path"] for spec in deps.SOURCES.values()),
    )
    for name in deps.SOURCES:
        root = deps.require_source(name)
        assert root.is_relative_to(clone)
        assert not (root / "unneeded/.git").exists()
