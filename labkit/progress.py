"""Per-lab progress of the student code, stored in progress/<ID>.json."""

import json
import time

from . import ROOT
from .build import fingerprint

DIR = ROOT / "progress"


def _path(lab):
    return DIR / f"{lab.id}.json"


def load(lab):
    try:
        return json.loads(_path(lab).read_text())
    except (OSError, ValueError):
        return {}


def record(lab, step, passed, detail=""):
    data = load(lab)
    fp = fingerprint(lab)
    if data.get("fingerprint") != fp:
        data = {"fingerprint": fp, "steps": {}}
    data["steps"][str(step)] = {
        "passed": passed,
        "detail": detail[:500],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    DIR.mkdir(parents=True, exist_ok=True)
    _path(lab).write_text(json.dumps(data, ensure_ascii=False, indent=1))


def summary(lab):
    """('done' | 'k/n' | '-' , stale flag)."""
    data = load(lab)
    steps = data.get("steps", {})
    passed = sum(1 for s in lab.steps if steps.get(str(s.step), {}).get("passed"))
    stale = bool(steps) and data.get("fingerprint") != fingerprint(lab)
    if not steps:
        return "-", False
    if passed == len(lab.steps):
        return "done", stale
    return f"{passed}/{len(lab.steps)}", stale
