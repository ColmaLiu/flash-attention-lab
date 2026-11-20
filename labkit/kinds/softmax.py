"""Row softmax: y[r, :] = softmax(x[r, :])"""

import torch

from . import compare


def make_inputs(case, gen):
    rows, cols = case.shape
    x = torch.randn(rows, cols, device="cuda", generator=gen, dtype=torch.float32)
    if case.pattern == "large":
        # exp(x) overflows FP32 without subtracting the row max.
        x = x * 1000.0
    elif case.pattern == "shifted":
        # Every row is offset by a large constant; softmax is shift-invariant.
        x = x + 5000.0
    elif case.pattern == "constant":
        x = torch.full_like(x, 3.0)
    elif case.pattern == "masked":
        # Some entries are -inf (as after a causal mask); at least one is finite.
        mask = torch.rand(rows, cols, device="cuda", generator=gen) < 0.5
        mask[:, 0] = False
        x = x.masked_fill(mask, float("-inf"))
    return {"x": x}


def make_outputs(inputs, case):
    return {"y": torch.empty_like(inputs["x"])}


def options(case, inputs):
    return {}


def verify(inputs, outputs, options, case, full=True):
    expected = torch.softmax(inputs["x"].to(torch.float64), dim=-1)
    return {"y": compare("y", outputs["y"], expected, 1e-8, 1e-4, ("row", "col"))}


def work(case):
    rows, cols = case.shape
    return {"flops": 0, "bytes": 2 * rows * cols * 4}


def baselines():
    def torch_softmax(inputs, options, case):
        x = inputs["x"]
        y = torch.empty_like(x)
        op = torch.ops.aten._softmax.out
        return (lambda: op(x, -1, False, out=y)), {"y": y}

    return {"torch": torch_softmax}
