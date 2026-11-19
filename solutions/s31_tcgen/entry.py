"""Native CUDA/PTX SM100 GEMM; tensor-map preparation is outside timing."""

from flash_attention_lab.build import build_cuda
from flash_attention_lab.ffi import tensor_arguments

IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    return {}


def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s31_tcgen", solution=True)
    tensors = tensor_arguments({**inputs, **outputs}, stream)
    state = lib.prepare(
        tensors["a"],
        tensors["b"],
        tensors["c"],
        stream,
    )

    def launch():
        lib.run(state)

    launch.extra_workspace_bytes = lib.workspace_bytes(state)
    return launch
