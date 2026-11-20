"""Course definition: one course/<ID>/spec.json per lab."""

import json
from dataclasses import dataclass, field
from functools import lru_cache

from . import ROOT, LabError

KINDS = ("transpose", "softmax", "gemm", "attention")
DTYPES = ("fp32", "fp16", "bf16", "e4m3")


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple
    dtype: str
    purpose: str
    pattern: str = "random"
    perf: bool = False
    options: dict = field(default_factory=dict)  # causal, scale, c_dtype, ...

    def label(self):
        opts = "".join(f" {k}={v}" for k, v in self.options.items())
        return f"{self.name} {list(self.shape)} {self.dtype}{opts}"


@dataclass(frozen=True)
class Step:
    step: int
    title: str
    cases: tuple  # case names
    probe: dict | None = None


@dataclass(frozen=True)
class Lab:
    id: str
    title: str
    part: int
    kind: str
    prerequisites: tuple
    steps: tuple
    cases: tuple
    perf_target: dict | None  # {"baseline": name, "ratio": float}
    profile_metrics: tuple = ()  # extra NCU metrics printed by ./lab profile

    @property
    def dir(self):
        return ROOT / "course" / self.id

    def student_dir(self):
        return ROOT / "students" / self.id

    def solution_dir(self):
        return ROOT / "solutions" / self.id

    def case(self, name):
        for c in self.cases:
            if c.name == name:
                return c
        names = ", ".join(c.name for c in self.cases)
        raise LabError(f"{self.id} has no case '{name}'. Cases: {names}")

    def step_info(self, n):
        for s in self.steps:
            if s.step == n:
                return s
        raise LabError(f"{self.id} has steps {[s.step for s in self.steps]}, not {n}")

    def perf_cases(self):
        return [c for c in self.cases if c.perf]


_CASE_KEYS = {"name", "shape", "dtype", "purpose", "pattern", "perf"}


def _parse(path):
    raw = json.loads(path.read_text())
    where = str(path.relative_to(ROOT))

    def need(cond, msg):
        if not cond:
            raise LabError(f"{where}: {msg}")

    need(raw.get("id") == path.parent.name, "id must equal the directory name")
    need(raw.get("kind") in KINDS, f"kind must be one of {KINDS}")
    cases = []
    for c in raw["cases"]:
        need(c.get("dtype") in DTYPES, f"case {c.get('name')}: bad dtype")
        need(c.get("purpose"), f"case {c.get('name')}: purpose required")
        options = {k: v for k, v in c.items() if k not in _CASE_KEYS}
        cases.append(
            Case(
                name=c["name"],
                shape=tuple(c["shape"]),
                dtype=c["dtype"],
                purpose=c["purpose"],
                pattern=c.get("pattern", "random"),
                perf=c.get("perf", False),
                options=options,
            )
        )
    names = [c.name for c in cases]
    need(len(set(names)) == len(names), "case names must be unique")
    steps = []
    for s in raw["steps"]:
        if s["cases"] == "all":
            selected = [c.name for c in cases if not c.perf]
        else:
            selected = s["cases"]
        for n in selected:
            need(n in names, f"step {s['step']}: unknown case {n}")
        steps.append(Step(s["step"], s["title"], tuple(selected), s.get("probe")))
    need(
        [s.step for s in steps] == list(range(1, len(steps) + 1)),
        "steps must be numbered 1..n",
    )
    return Lab(
        id=raw["id"],
        title=raw["title"],
        part=raw["part"],
        kind=raw["kind"],
        prerequisites=tuple(raw.get("prerequisites", [])),
        steps=tuple(steps),
        cases=tuple(cases),
        perf_target=raw.get("perf_target"),
        profile_metrics=tuple(raw.get("profile_metrics", [])),
    )


@lru_cache(None)
def labs():
    found = {}
    for path in sorted((ROOT / "course").glob("*/spec.json")):
        lab = _parse(path)
        found[lab.id] = lab
    return found


def get(lab_id):
    all_labs = labs()
    if lab_id in all_labs:
        return all_labs[lab_id]
    # Accept a unique prefix such as "00" or "00_trans".
    matches = [k for k in all_labs if k.startswith(lab_id)]
    if len(matches) == 1:
        return all_labs[matches[0]]
    raise LabError(
        f"unknown lab '{lab_id}'. Run ./lab list to see the labs."
        if not matches
        else f"'{lab_id}' is ambiguous: {', '.join(matches)}"
    )
