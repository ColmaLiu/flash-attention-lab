"""Materialized attention keeps both quadratic intermediates visible for the IO experiment."""
import torch
IMPLEMENTED = True
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    batch, n, heads, _ = inputs["q"].shape
    return {name: ((batch, heads, n, n), torch.float32) for name in ("scores", "probabilities")}
