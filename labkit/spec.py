"""Course definition: one course/<ID>/spec.json per lab."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import ROOT, LabError

KINDS = ("transpose", "softmax", "gemm", "attention")
DTYPES = ("fp32", "fp16", "bf16", "e4m3")
# Number of shape dimensions and the input patterns each kind's make_inputs
# understands ("random" is the default).
RANK = {"transpose": 2, "softmax": 2, "gemm": 3, "attention": 4}
PATTERNS = {
    "transpose": ("random", "coordinate"),
    "softmax": ("random", "large", "shifted", "constant", "masked"),
    "gemm": ("random", "int"),
    "attention": (
        "random",
        "large_logits",
        "uniform",
        "ones_v",
        "position",
        "growing_max",
        "jump_max",
        "long_tail",
    ),
}
# The learner's own cases: students/<ID>/cases.json.
CUSTOM_FILE = "cases.json"


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

    def custom_path(self):
        return self.student_dir() / CUSTOM_FILE


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
        need(
            len(c.get("shape", ())) == RANK[raw["kind"]],
            f"case {c.get('name')}: shape needs {RANK[raw['kind']]} dimensions",
        )
        need(
            c.get("pattern", "random") in PATTERNS[raw["kind"]],
            f"case {c.get('name')}: unknown pattern {c.get('pattern')!r}",
        )
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


# ---------------------------------------------------------------------------
# The learner's own cases: students/<ID>/cases.json, or --shape on the
# command line. They use the same fields as spec.json; options are limited to
# the ones this lab's spec uses (no tolerance overrides), dtypes to the ones
# its kernel supports. A custom case runs on the last step unless it lists
# "steps".
# ---------------------------------------------------------------------------
_CUSTOM_KEYS = {"name", "shape", "dtype", "pattern", "purpose", "steps"}
_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class CustomCase(Case):
    steps: tuple = ()  # empty: the last step only


def _custom_options(lab):
    keys = {k for c in lab.cases for k in c.options}
    return sorted(k for k in keys if not k.endswith("_tol"))


def _value(text):
    """Command-line option value: JSON (true, 0.125, ...) or a plain string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def make_custom(lab, fields, where):
    def need(cond, msg):
        if not cond:
            raise LabError(f"{where}: {msg}")

    name = fields.get("name")
    need(isinstance(name, str) and _NAME.fullmatch(name), f"bad case name {name!r}")
    options = _custom_options(lab)
    unknown = sorted(set(fields) - _CUSTOM_KEYS - set(options))
    need(
        not unknown,
        f"case {name}: unknown field(s) {', '.join(unknown)}; this lab's options: "
        + (", ".join(options) or "none"),
    )
    shape = fields.get("shape")
    rank = RANK[lab.kind]
    need(
        isinstance(shape, (list, tuple))
        and len(shape) == rank
        and all(
            isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in shape
        ),
        f"case {name}: shape must be {rank} positive integers",
    )
    dtypes = sorted({c.dtype for c in lab.cases})
    dtype = fields.get("dtype", lab.cases[0].dtype)
    need(dtype in dtypes, f"case {name}: dtype {dtype!r} not in {dtypes}")
    pattern = fields.get("pattern", "random")
    need(
        pattern in PATTERNS[lab.kind],
        f"case {name}: pattern {pattern!r} not in {list(PATTERNS[lab.kind])}",
    )
    steps = fields.get("steps", [])
    valid = [s.step for s in lab.steps]
    need(
        isinstance(steps, list) and all(s in valid for s in steps),
        f"case {name}: steps must be a list drawn from {valid}",
    )
    return CustomCase(
        name=name,
        shape=tuple(shape),
        dtype=dtype,
        purpose=fields.get("purpose") or "custom case",
        pattern=pattern,
        options={k: v for k, v in fields.items() if k not in _CUSTOM_KEYS},
        steps=tuple(steps),
    )


def custom_cases(lab, path=None):
    """Cases from students/<ID>/cases.json; () when the file does not exist."""
    path = path or lab.custom_path()
    if not path.is_file():
        return ()
    try:
        where = path.relative_to(ROOT)
    except ValueError:
        where = path
    try:
        raw = json.loads(path.read_text())
    except ValueError as exc:
        raise LabError(f"{where}: not valid JSON ({exc})") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise LabError(f'{where}: expected {{"cases": [ ... ]}}')
    cases = [
        make_custom(lab, c if isinstance(c, dict) else {}, where) for c in raw["cases"]
    ]
    names = [c.name for c in cases]
    taken = sorted({n for n in names if names.count(n) > 1})
    taken += sorted({c.name for c in lab.cases} & set(names))
    if taken:
        raise LabError(
            f"{where}: case name(s) {', '.join(taken)} repeated or already in spec.json"
        )
    return tuple(cases)


def inline_case(lab, shape, dtype=None, pattern=None, opts=()):
    """The case described by --shape 300,500,72 [--dtype] [--pattern] [--opt k=v]."""
    try:
        dims = [int(x) for x in re.split(r"[,x]", shape.strip())]
    except ValueError:
        raise LabError(f"--shape {shape!r}: expected integers such as 300,500,72")
    fields = {"name": "custom", "shape": dims}
    if dtype:
        fields["dtype"] = dtype
    if pattern:
        fields["pattern"] = pattern
    for item in opts:
        key, eq, value = item.partition("=")
        if not eq:
            raise LabError(f"--opt {item!r}: expected key=value, e.g. causal=true")
        fields[key] = _value(value)
    return make_custom(lab, fields, "command line")


def find_case(lab, name):
    """A case of spec.json or of the learner's cases.json."""
    for c in (*lab.cases, *custom_cases(lab)):
        if c.name == name:
            return c
    names = ", ".join(c.name for c in lab.cases)
    custom = ", ".join(c.name for c in custom_cases(lab))
    raise LabError(
        f"{lab.id} has no case '{name}'. Cases: {names}"
        + (
            f"; yours ({lab.custom_path().relative_to(ROOT)}): {custom}"
            if custom
            else ""
        )
    )
