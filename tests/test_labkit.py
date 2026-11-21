"""Unit tests for labkit that do not need a GPU."""

import json

import pytest
import torch

from labkit import CorrectnessError, EnvironmentError_, LabError, ROOT, spec
from labkit.build import dependencies
from labkit.env import selected_device
from labkit.kinds import compare


def test_every_spec_parses_and_has_sources():
    labs = spec.labs()
    assert labs, "no labs found under course/"
    for lab in labs.values():
        assert (lab.dir / "README.md").is_file(), lab.id
        assert (lab.student_dir() / "kernel.cu").is_file(), lab.id
        assert (lab.solution_dir() / "kernel.cu").is_file(), lab.id
        for step in lab.steps:
            assert step.cases, f"{lab.id} step {step.step} has no cases"
        assert not lab.steps[-1].probe, f"{lab.id}: last step must check real outputs"
        for p in lab.prerequisites:
            assert p in labs, f"{lab.id}: unknown prerequisite {p}"


def test_lab_prefix_lookup():
    first = next(iter(spec.labs()))
    assert spec.get(first[:2]).id == first
    with pytest.raises(LabError):
        spec.get("does-not-exist")


def test_spec_rejects_bad_case(tmp_path, monkeypatch):
    bad = tmp_path / "course" / "99_bad"
    bad.mkdir(parents=True)
    (bad / "spec.json").write_text(
        json.dumps(
            {
                "id": "99_bad",
                "title": "t",
                "part": 0,
                "kind": "transpose",
                "steps": [{"step": 1, "title": "s", "cases": ["missing"]}],
                "cases": [
                    {"name": "a", "shape": [1, 1], "dtype": "fp32", "purpose": "p"}
                ],
            }
        )
    )
    monkeypatch.setattr(spec, "ROOT", tmp_path)
    with pytest.raises(LabError, match="unknown case"):
        spec._parse(bad / "spec.json")


def test_selected_device_respects_mask():
    assert selected_device({}) == "0"
    assert selected_device({"LAB_GPU": "3"}) == "3"
    assert selected_device({"CUDA_VISIBLE_DEVICES": "4,5"}) == "4"
    assert selected_device({"CUDA_VISIBLE_DEVICES": "4,5", "LAB_GPU": "5"}) == "5"
    with pytest.raises(EnvironmentError_):
        selected_device({"CUDA_VISIBLE_DEVICES": "4,5", "LAB_GPU": "1"})


def test_compare_reports_first_mismatch_and_nan():
    expected = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    got = expected.clone()
    got[1, 2] = float("nan")
    with pytest.raises(CorrectnessError) as info:
        compare("y", got, expected, 0.0, 0.0, ("r", "c"))
    text = str(info.value)
    assert "r=1, c=2" in text and "never written" in text
    assert compare("y", expected, expected, 0.0, 0.0)["max_abs_err"] == 0.0


def test_dependencies_follow_includes():
    deps = dependencies(ROOT / "solutions" / "00_transpose" / "kernel.cu")
    names = {p.name for p in deps}
    assert {"kernel.cu", "lab.h", "check.h", "register.h", "problem.h"} <= names


def test_attention_reference_matches_torch_sdpa():
    from labkit.kinds import attention

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for the reference helper")
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(2, 70, 3, 16, device="cuda", generator=gen) for _ in range(3)
    )
    for causal in (False, True):
        opts = {"scale": 0.3, "causal": float(causal)}
        o, lse = attention.reference({"q": q, "k": k, "v": v}, opts)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).double(),
            k.transpose(1, 2).double(),
            v.transpose(1, 2).double(),
            is_causal=causal,
            scale=0.3,
        ).transpose(1, 2)
        torch.testing.assert_close(o, ref)
        rows = torch.tensor([0, 37, 69], device="cuda")
        o_rows, lse_rows = attention.reference({"q": q, "k": k, "v": v}, opts, rows)
        torch.testing.assert_close(o_rows, o[:, rows])
        torch.testing.assert_close(lse_rows, lse[:, :, rows])


def test_relative_links_in_lessons_resolve():
    import re

    broken = []
    for md in sorted(
        [
            *ROOT.glob("course/*/README.md"),
            *ROOT.glob("solutions/*/NOTES.md"),
            *ROOT.glob("docs/*.md"),
            ROOT / "README.md",
            ROOT / "AGENTS.md",
        ]
    ):
        for target in re.findall(r"\]\(([^)\s]+)\)", md.read_text()):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not broken, "\n".join(broken)
