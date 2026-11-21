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


def _markdown_docs():
    return sorted(
        [
            *ROOT.glob("course/*/README.md"),
            *ROOT.glob("solutions/*/NOTES.md"),
            *ROOT.glob("docs/*.md"),
            ROOT / "README.md",
            ROOT / "AGENTS.md",
        ]
    )


def _anchors(md):
    """GitHub-style heading anchors of a Markdown file (code fences skipped)."""
    import re

    anchors, seen, fence = set(), {}, False
    for line in md.read_text().splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1)).lower()
        slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def test_relative_links_in_lessons_resolve():
    import re

    broken = []
    for md in _markdown_docs():
        for target in re.findall(r"\]\(([^)\s]+)\)", md.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            file, _, anchor = target.partition("#")
            path = (md.parent / file).resolve() if file else md
            if not path.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
            elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
                broken.append(f"{md.relative_to(ROOT)} -> {target} (no such heading)")
    assert not broken, "\n".join(broken)


def test_docs_and_specs_are_ascii_width():
    """No CJK or full-width text: it breaks code-block diagram alignment."""
    import re

    wide = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯]")
    files = [*_markdown_docs(), *ROOT.glob("course/*/spec.json")]
    bad = [
        f"{f.relative_to(ROOT)}:{n}"
        for f in files
        for n, line in enumerate(f.read_text().splitlines(), 1)
        if wide.search(line)
    ]
    assert not bad, "wide (CJK/full-width) characters at:\n" + "\n".join(bad)
