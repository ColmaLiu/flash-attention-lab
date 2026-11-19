"""Complete CUDA/PTX SM100 attention; host preparation excludes allocation and encoding."""

from flash_attention_lab.build import build_cuda
from flash_attention_lab.ffi import tensor_arguments

IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    return {}


def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s41_overlap", solution=True)
    tensors = tensor_arguments({**inputs, **outputs}, stream)
    state = lib.prepare(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["o"],
        tensors["lse"],
        config["scale"],
        config["causal"],
        stream,
    )
    return lambda: lib.run(state)
