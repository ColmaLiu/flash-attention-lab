"""Prepare the TMA descriptor outside the timed launch path."""

import torch
from flash_attention_lab.build import build_cuda
from flash_attention_lab.ffi import launch_arguments

IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    return {"descriptor": ((128,), torch.uint8)}


def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s30_tma", solution=True)
    args = launch_arguments(inputs, outputs, workspace, config, stream)
    lib.prepare_descriptor(args[0]["x"], args[2]["descriptor"], stream)
    return lambda: lib.launch(*args)
