"""y[cols, rows] = x[rows, cols]^T"""

import torch

from . import TORCH_DTYPES, compare


def make_inputs(case, gen):
    rows, cols = case.shape
    dtype = TORCH_DTYPES[case.dtype]
    if case.pattern == "coordinate":
        # x[r, c] = r * cols + c, so a wrong value tells you where it came from.
        x = torch.arange(rows * cols, device="cuda", dtype=torch.float64)
        x = x.reshape(rows, cols).to(dtype)
    else:
        x = torch.randn(rows, cols, device="cuda", generator=gen).to(dtype)
    return {"x": x}


def make_outputs(inputs, case):
    rows, cols = inputs["x"].shape
    return {"y": torch.empty(cols, rows, device="cuda", dtype=inputs["x"].dtype)}


def options(case, inputs):
    return {}


def verify(inputs, outputs, options, case, full=True):
    return {"y": compare("y", outputs["y"], inputs["x"].t(), 0.0, 0.0, ("col", "row"))}


def work(case):
    rows, cols = case.shape
    size = TORCH_DTYPES[case.dtype].itemsize
    return {"flops": 0, "bytes": 2 * rows * cols * size}


def baselines():
    def torch_copy(inputs, options, case):
        x = inputs["x"]
        y = torch.empty(x.shape[1], x.shape[0], device="cuda", dtype=x.dtype)
        return (lambda: y.copy_(x.t())), {"y": y}

    def contiguous_copy(inputs, options, case):
        # Not a transpose: y.copy_(x) moves the same bytes with perfectly
        # sequential reads and writes, i.e. a practical HBM ceiling.
        x = inputs["x"]
        y = torch.empty_like(x)
        return (lambda: y.copy_(x)), {"y": y}

    return {"torch": torch_copy, "copy": contiguous_copy}
