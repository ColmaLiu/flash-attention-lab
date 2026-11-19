import importlib.metadata as metadata
import os
import platform
import re
import subprocess
import sys
from .errors import EnvironmentUnsupported
from .runtime import cuda_home, cuda_development_paths, tool_path


def command_output(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return {
            "code": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"code": -1, "error": str(exc)}


def torch_module():
    try:
        import torch

        return torch
    except (ImportError, OSError) as exc:
        raise EnvironmentUnsupported(
            f"cannot load PyTorch: {exc}; run bash scripts/bootstrap.sh"
        ) from exc


def snapshot():
    packages = {}
    for name in ("torch", "pytest", "ninja", "apache-tvm-ffi"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    from .cpp_dependencies import source_snapshot

    toolkit = cuda_home()
    info = {
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "sources": source_snapshot(),
        "packages": packages,
        "python_prefix": sys.prefix,
        "conda_prefix": os.getenv("CONDA_PREFIX"),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "cuda_home": str(toolkit) if toolkit else None,
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvcc": command_output([str(toolkit / "bin/nvcc"), "--version"])
        if toolkit
        else {"error": "nvcc not found"},
        "cxx": command_output([os.getenv("CXX", "c++"), "--version"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,compute_cap,driver_version",
                "--format=csv",
            ]
        ),
        "driver_cuda_ceiling": command_output(["nvidia-smi"]),
        "tools": {
            name: tool_path(name, override)
            for name, override in (
                ("compute-sanitizer", "LAB_SANITIZER"),
                ("ncu", "LAB_NCU"),
                ("nsys", None),
            )
        },
    }
    try:
        torch = torch_module()
        info["torch_cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            info["selected_gpu"] = {
                "name": p.name,
                "compute_capability": [p.major, p.minor],
                "total_memory": p.total_memory,
                "sm_count": p.multi_processor_count,
            }
    except (EnvironmentUnsupported, RuntimeError) as exc:
        info["cuda_available"] = False
        info["torch_error"] = str(exc)
    return info


def require_environment(backend):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise EnvironmentUnsupported("this dependency recipe targets Linux x86_64")
    torch = torch_module()
    if not torch.cuda.is_available():
        raise EnvironmentUnsupported(
            "PyTorch CUDA unavailable; check driver, CUDA wheel and visible GPU mask"
        )
    cap = torch.cuda.get_device_capability()
    if cap != (10, 0):
        raise EnvironmentUnsupported(
            f"SM100 (compute capability 10.0, e.g. B200) required; selected capability is {cap}"
        )
    if backend == "cuda":
        toolkit = cuda_home()
        if not toolkit or not os.access(toolkit / "bin/nvcc", os.X_OK):
            raise EnvironmentUnsupported(
                "nvcc not found; set LAB_CUDA_HOME/CUDA_HOME, add nvcc to PATH, or bootstrap --with-cuda"
            )
        version = command_output([str(toolkit / "bin/nvcc"), "--version"])
        match = re.search(r"release (\d+)\.(\d+)", version.get("stdout", ""))
        if (
            version["code"] != 0
            or not match
            or tuple(map(int, match.groups())) != (12, 8)
        ):
            raise EnvironmentUnsupported(
                "this CUDA C++ recipe requires Toolkit 12.8; select it with LAB_CUDA_HOME or bootstrap --with-cuda"
            )
        if torch.version.cuda != "12.8":
            raise EnvironmentUnsupported(
                "install the pinned PyTorch cu128 wheel via bash scripts/bootstrap.sh"
            )
        cuda_development_paths(toolkit)
        if not tool_path("ninja"):
            raise EnvironmentUnsupported(
                "ninja is missing from PATH; run bash scripts/bootstrap.sh"
            )
        if not tool_path(os.getenv("CXX", "c++")):
            raise EnvironmentUnsupported(
                "C++ compiler missing; set CXX or use bootstrap --with-cuda"
            )
        if os.getenv("CC") and not tool_path(os.environ["CC"]):
            raise EnvironmentUnsupported("CC does not name an executable host compiler")
    else:
        raise ValueError(f"unknown backend: {backend}")
    from .cpp_dependencies import SOURCES, require_source

    for name in SOURCES:
        require_source(name)
    require_course_packages()


def require_course_packages():
    """Check the single package recipe used by bootstrap, without duplicate pins."""
    import tomllib
    from pathlib import Path

    recipe = Path(__file__).resolve().parents[1] / "pyproject.toml"
    requirements = tomllib.loads(recipe.read_text())["project"]["dependencies"]
    for requirement in requirements:
        name, expected = requirement.split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise EnvironmentUnsupported(
                f"course dependency {name} is missing; run bash scripts/bootstrap.sh"
            ) from exc
        # The recipe selects PyTorch's CUDA local version separately in requirements.txt.
        if actual.split("+", 1)[0] != expected:
            raise EnvironmentUnsupported(
                f"course dependency {name}=={expected} required, found {actual}; run bash scripts/bootstrap.sh"
            )


def require_ncu_sections(binary):
    result = command_output([binary, "--list-sections"])
    if result["code"] != 0:
        raise EnvironmentUnsupported(f"cannot query Nsight Compute sections: {result}")
    missing = [
        s
        for s in ("LaunchStats", "MemoryWorkloadAnalysis", "SpeedOfLight")
        if not re.search(r"\b" + s + r"\b", result.get("stdout", ""))
    ]
    if missing:
        raise EnvironmentUnsupported(
            "installed NCU lacks required sections: " + ", ".join(missing)
        )
