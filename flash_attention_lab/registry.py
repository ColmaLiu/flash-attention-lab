import hashlib
import importlib.util
import json
import os
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "curriculum.json").read_text())
EXERCISES = {e["id"]: e for e in CATALOG}


def selected(stage=None):
    return [e for e in CATALOG if stage is None or e["stage"] == int(stage)]


def student_dir(e):
    return ROOT / "students" / e["backend"] / e["id"]


def load_student(e, *, solution=False):
    path = (ROOT / "solutions" / e["id"] if solution else student_dir(e)) / "entry.py"
    spec = importlib.util.spec_from_file_location(
        ("solution_" if solution else "student_") + e["id"], path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(e):
    # Include all student sources (including helpers/dependencies) and infrastructure.
    h = hashlib.sha256()
    paths = [
        ROOT / name
        for name in (
            "curriculum.json",
            "docs/sources.lock.json",
            "pyproject.toml",
            "requirements.txt",
            "environment.yml",
        )
    ]
    settings = {
        key: os.environ.get(key)
        for key in (
            "LAB_NVCC_FLAGS",
            "CXX",
            "CC",
            "LAB_CUDA_HOME",
            "CUDA_HOME",
            "CUDA_VISIBLE_DEVICES",
            "LAB_GPU",
        )
    }
    packages = {}
    for name in ("torch", "ninja", "apache-tvm-ffi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    h.update(
        json.dumps(
            {"settings": settings, "packages": packages}, sort_keys=True
        ).encode()
    )
    if (ROOT / ".gitmodules").is_file():
        paths.append(ROOT / ".gitmodules")
    for folder in ("students", "solutions", "benchmarks", "flash_attention_lab"):
        paths += sorted(
            p
            for p in (ROOT / folder).rglob("*")
            if p.suffix in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".json")
        )
    from .cpp_dependencies import source_snapshot

    versions = {
        name: {key: info[key] for key in ("expected_commit", "commit", "state")}
        for name, info in source_snapshot().items()
    }
    h.update(json.dumps(versions, sort_keys=True).encode())
    for p in paths:
        h.update(str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()
