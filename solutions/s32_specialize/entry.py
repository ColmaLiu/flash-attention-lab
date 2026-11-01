"""Pinned CUTLASS SM100 GEMM; setup and workspace allocation occur before timing."""
from flash_attention_lab.build import build_cuda
IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []
def workspace_spec(inputs, config): return {}
def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s32_specialize", solution=True)
    a, b, out = inputs["a"], inputs["b"], outputs["c"]
    state = lib.prepare(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.shape[0], b.shape[1], a.shape[1], stream)
    def launch():
        lib.run(state)
    launch.extra_workspace_bytes = lib.workspace_bytes(state)
    return launch
