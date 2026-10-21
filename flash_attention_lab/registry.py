import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "curriculum.json").read_text())
EXERCISES = {e["id"]: e for e in CATALOG}


def selected(stage=None):
    return [e for e in CATALOG if stage is None or e["stage"] == int(stage)]


def student_dir(e):
    return ROOT / "students" / e["backend"] / e["id"]


def load_student(e):
    path = student_dir(e) / "entry.py"
    spec = importlib.util.spec_from_file_location("student_" + e["id"], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(e):
    # Include all student sources (including helpers/dependencies) and infrastructure.
    h = hashlib.sha256()
    paths = [ROOT / "curriculum.json", ROOT / "docs/sources.lock.json"]
    if (ROOT / ".gitmodules").is_file():
        paths.append(ROOT / ".gitmodules")
    for folder in ("students", "flash_attention_lab"):
        paths += sorted(p for p in (ROOT / folder).rglob("*")
                        if p.suffix in (".py", ".cu", ".cuh", ".h", ".cpp"))
    # Header edits invalidate results as well as Ninja compilation dependencies.
    if e["backend"] == "cuda":
        from .cpp_dependencies import cutlass_root
        headers = cutlass_root() / "include"
        paths += sorted(p for p in headers.rglob("*") if p.is_file())
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()
