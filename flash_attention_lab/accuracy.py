import torch
from .errors import CorrectnessFailure


def tolerance(e, name, input_dtype):
    if e["kind"] in ("copy", "transpose", "affine") or name == "maximum":
        return 0.0, 0.0
    if e["kind"] == "exp2":
        return 1e-38, 1e-4
    if name in ("lse", "m", "l") or e["kind"] in ("reduce", "softmax", "online"):
        return 2e-5, 2e-5
    if e["kind"] == "gemm":
        return 2e-3, 2e-4  # FP32 output; operands already quantized
    if input_dtype == torch.float16:
        return 3e-3, 3e-3
    if input_dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 5e-5, 5e-5


def compare(e, actual, expected, inputs):
    metrics = {}
    for name, ref in expected.items():
        out = actual[name].double()
        if out.shape != ref.shape or not torch.isfinite(out).all():
            raise CorrectnessFailure(f"{name}: invalid shape or NaN/Inf")
        atol, rtol = tolerance(e, name, next(iter(inputs.values())).dtype)
        error = (out - ref).abs()
        floor = max(atol, 1e-12)
        relative = error / ref.abs().clamp_min(floor)
        budget = atol + rtol * ref.abs()
        bad = error > budget
        metrics[name] = {
            "max_abs": error.max().item(),
            "max_rel_floored": relative.max().item(),
            "relative_floor": floor,
            "atol": atol,
            "rtol": rtol,
            "bad_elements": bad.sum().item(),
        }
        if bad.any():
            idx = tuple(bad.nonzero()[0].tolist())
            raise CorrectnessFailure(
                f"{name} first_bad={idx} actual={out[idx].item()} ref={ref[idx].item()} metrics={metrics[name]}"
            )
    if e["kind"] in ("softmax", "online"):
        y = actual["y"].double()
        if (y < 0).any() or (y.sum(-1) - 1).abs().max() > 5e-5:
            raise CorrectnessFailure(
                "softmax must be nonnegative and each row must sum to one"
            )
    return metrics
