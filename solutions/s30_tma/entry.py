"""Prepare the TMA descriptor outside the timed launch path."""
import torch
from flash_attention_lab.build import build_cuda, descriptors
IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    return {"descriptor": ((128,), torch.uint8)}


def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s30_tma", solution=True)
    rows, cols = inputs["x"].shape
    lib.prepare_descriptor(inputs["x"].data_ptr(), workspace["descriptor"].data_ptr(), rows, cols, stream)
    args = (descriptors(inputs), descriptors(outputs), descriptors(workspace), config, stream)
    return lambda: lib.launch(*args)
