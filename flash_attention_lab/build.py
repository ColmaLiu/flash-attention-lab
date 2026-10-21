import os
import shlex
from functools import lru_cache
import torch
from .registry import ROOT, student_dir


@lru_cache(None)
def build_cuda(exercise_id):
    from .environment import require_environment
    from .runtime import cuda_home, cuda_development_paths
    from .registry import EXERCISES
    require_environment("cuda")
    from torch.utils import cpp_extension
    # cpp_extension caches discovery at import time; honor the selected toolkit even
    # when a caller imported it before configuring this project.
    cpp_extension.CUDA_HOME = str(cuda_home())
    cuda_includes, cuda_library = cuda_development_paths(cuda_home())
    from .cpp_dependencies import cpp_include_paths
    dependency_includes = cpp_include_paths()
    if exercise_id == "infra_smoke":
        source = ROOT / "flash_attention_lab" / "infra_smoke.cu"
    else:
        source = student_dir(EXERCISES[exercise_id]) / "kernel.cu"
    # Explicit target selected only after live device inspection; no global toolchain edits.
    build_dir = ROOT / ".cache" / "build" / exercise_id
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    return cpp_extension.load(name="falab_" + exercise_id,
                sources=[str(ROOT / "flash_attention_lab" / "binding.cpp"), str(source)],
                extra_include_paths=[str(ROOT / "flash_attention_lab"), *map(str, cuda_includes), *map(str, dependency_includes)],
                extra_ldflags=["-L" + shlex.quote(str(cuda_library))],
                extra_cflags=["-O2", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a"],
                build_directory=str(build_dir), verbose=bool(os.getenv("LAB_BUILD_VERBOSE")))


def descriptors(tensors):
    codes = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
             torch.int32: 3, torch.uint8: 4, torch.int64: 5}
    return {n: (t.data_ptr(), list(t.shape), codes[t.dtype]) for n, t in tensors.items()}
