"""Operator kinds: inputs, mathematical reference, tolerances, baselines.

Each kind module defines:
  make_inputs(case, generator) -> dict of input tensors
  make_outputs(inputs, case)   -> dict of output tensors (contents irrelevant)
  options(case, inputs)        -> dict of scalar options for the binding
  verify(inputs, outputs, options, case, full) -> dict of error metrics,
                                  raising CorrectnessError on mismatch
  work(case)                   -> {"flops": int, "bytes": int}
  baselines()                  -> {name: factory(inputs, options) -> (launch, outputs)}
"""

import importlib

import torch

from .. import CorrectnessError

TORCH_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
}


def module(kind):
    return importlib.import_module(f"labkit.kinds.{kind}")


def _index(flat, shape):
    coords = []
    for size in reversed(shape):
        coords.append(flat % size)
        flat //= size
    return tuple(reversed(coords))


def compare(name, got, expected, atol, rtol, axes=None):
    """Elementwise |got - expected| <= atol + rtol * |expected|.

    `axes` names the dimensions for error messages, e.g. ("b", "n", "h", "d").
    """
    got64 = got.to(torch.float64)
    exp64 = expected.to(torch.float64)
    if got64.shape != exp64.shape:
        raise CorrectnessError(
            f"{name}: shape {tuple(got64.shape)} != expected {tuple(exp64.shape)}"
        )
    err = (got64 - exp64).abs()
    bad = ~(err <= atol + rtol * exp64.abs())  # NaN counts as bad
    metrics = {
        "max_abs_err": float(torch.nan_to_num(err, nan=float("inf")).max()),
        "atol": atol,
        "rtol": rtol,
    }
    count = int(bad.sum())
    if count:
        flat = int(bad.flatten().nonzero()[0])
        coord = _index(flat, tuple(got64.shape))
        where = ", ".join(
            f"{a}={c}"
            for a, c in zip(axes or [f"dim{i}" for i in range(len(coord))], coord)
        )
        nan = int(torch.isnan(got64).sum())
        lines = [
            f"{name}: {count} of {got64.numel()} elements wrong "
            f"(tolerance |err| <= {atol:g} + {rtol:g}*|expected|)",
            f"  first at [{where}]: got {float(got64[coord]):.6g}, "
            f"expected {float(exp64[coord]):.6g}",
            f"  max |err| = {metrics['max_abs_err']:.3g}",
        ]
        if nan:
            lines.append(
                f"  {nan} elements are NaN: outputs start as NaN, so these were "
                "probably never written"
            )
        raise CorrectnessError("\n".join(lines))
    return metrics
