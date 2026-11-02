"""Complete CUDA/PTX SM100 attention; host preparation excludes allocation and encoding."""
from flash_attention_lab.build import build_cuda
IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []
def workspace_spec(inputs, config): return {}
def prepare(inputs, outputs, workspace, config, stream):
    import torch
    lib = build_cuda("s40_attention", solution=True)
    q, k, v = (inputs[key] for key in ("q", "k", "v"))
    state = lib.prepare(q.data_ptr(), k.data_ptr(), v.data_ptr(), outputs["o"].data_ptr(),
        outputs["lse"].data_ptr(), *q.shape, q.dtype == torch.float16, config["scale"], config["causal"], stream)
    return lambda: lib.run(state)
