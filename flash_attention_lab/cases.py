import math
import numbers
import struct
import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def make_inputs(e, case, device="cpu"):
    """CPU seeded generation then explicit transfer; excluded from timing."""
    g = torch.Generator(device="cpu").manual_seed(case.get("seed", 20261014))
    shape = tuple(case["shape"])
    dtype = DTYPES[case["dtype"]]
    pattern = case.get("pattern", "random")

    def sample(s):
        x = torch.randn(s, generator=g, dtype=torch.float32)
        if pattern == "zero": x.zero_()
        elif pattern == "constant": x.fill_(3.0)
        elif pattern == "large": x = x * 5
        elif pattern == "shifted": x = x + 10000
        elif pattern == "negative": x = -x.abs() - 1000
        elif pattern == "coordinate": x = (torch.arange(math.prod(s)).reshape(s) % 127 - 63).float()
        elif pattern == "cancellation":
            x = torch.where(torch.arange(math.prod(s)).reshape(s) % 2 == 0, 1.0, -1.0)
        return x.to(dtype=dtype, device=device).contiguous()

    cfg = {"causal": bool(case.get("causal", False)), "scale": case.get("scale", 1.0)}
    if e["kind"] == "attention":
        inputs = {name: sample(shape) for name in ("q", "k", "v")}
        cfg["scale"] = case.get("scale", 1 / math.sqrt(shape[-1]))
        if pattern == "max_ramp":
            # Analytic adversarial inputs: maxima increase late in the key sequence.
            inputs["q"].fill_(1)
            ramp = torch.linspace(-6, 6, shape[1], device=device, dtype=torch.float32)
            inputs["k"].copy_(ramp[None, :, None, None].expand(shape))
        if pattern == "uniform_attention":
            inputs["q"].zero_()
            inputs["k"].zero_()
        return inputs, cfg
    if e["kind"] == "gemm":
        m, n, k = shape
        return {"a": sample((m, k)), "b": sample((k, n))}, cfg
    inputs = {"x": sample(shape)}
    if e["kind"] == "exp2":
        lo, hi = (-80, 8) if pattern == "large" else (-16, 0)
        inputs["x"] = torch.linspace(lo, hi, math.prod(shape), device=device).reshape(shape)
        # Include exact integer boundaries and their adjacent FP32 values.
        integers = torch.arange(lo, hi + 1, device=device, dtype=torch.float32)
        neighbors = torch.cat((integers, torch.nextafter(integers, torch.full_like(integers, -torch.inf)),
                               torch.nextafter(integers, torch.full_like(integers, torch.inf))))
        neighbors = neighbors[(neighbors >= lo) & (neighbors <= hi)]
        inputs["x"].view(-1)[:neighbors.numel()] = neighbors
    return inputs, cfg


def output_spec(e, inputs):
    first = next(iter(inputs.values()))
    dtype = first.dtype
    if e["kind"] == "attention":
        b, n, h, _ = inputs["q"].shape
        return {"o": (inputs["q"].shape, dtype), "lse": ((b, h, n), torch.float32)}
    if e["kind"] == "gemm":
        return {"c": ((inputs["a"].shape[0], inputs["b"].shape[1]), torch.float32)}
    if e["kind"] == "reduce":
        return {k: (first.shape[:-1], torch.float32) for k in ("maximum", "sum")}
    shape = tuple(reversed(first.shape)) if e["kind"] == "transpose" else first.shape
    result = {"y": (shape, torch.float32 if e["kind"] in ("affine", "exp2") else dtype)}
    if e["kind"] == "online":
        result.update({k: (first.shape[:-1], torch.float32) for k in ("m", "l")})
    return result


def allocate_outputs(e, inputs):
    device = next(iter(inputs.values())).device
    return {n: torch.empty(s, dtype=d, device=device) for n, (s, d) in output_spec(e, inputs).items()}


def validate_inputs(e, inputs, config):
    kind = e["kind"]
    keys = {"q", "k", "v"} if kind == "attention" else {"a", "b"} if kind == "gemm" else {"x"}
    if set(inputs) != keys: raise ValueError(f"expected inputs {keys}")
    ts = list(inputs.values())
    t = ts[0]
    for a in ts:
        if not a.is_cuda or not a.is_contiguous(): raise ValueError("requires contiguous CUDA tensors")
        if a.dtype != t.dtype or a.device != t.device: raise ValueError("dtype/device mismatch")
        if a.requires_grad: raise ValueError("forward-only: requires_grad must be False")
    if kind == "attention":
        if any(a.shape != t.shape for a in ts) or t.ndim != 4: raise ValueError("expected equal [B,N,H,D]")
        shape = tuple(t.shape)
    elif kind == "gemm":
        a, b = inputs["a"], inputs["b"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]: raise ValueError("expected [M,K] @ [K,N]")
        shape = (a.shape[0], b.shape[1], a.shape[1])
    else: shape = tuple(t.shape)
    dtype = next((s for s, d in DTYPES.items() if d == t.dtype), None)
    domain = e["domain"]
    alignment = domain.get("pointer_alignment", 1)
    if any(a.data_ptr() % alignment for a in ts):
        raise ValueError(f"input pointer outside {e['id']} domain: requires {alignment}-byte alignment")
    multiples = domain["shape_multiple"]
    if len(shape) != len(multiples) or any(n <= 0 or n % m for n, m in zip(shape, multiples)):
        raise ValueError(f"shape outside {e['id']} domain: {shape}; multiples={multiples}")
    if dtype not in domain["dtypes"] or config["causal"] not in domain["causal"]:
        raise ValueError(f"dtype/mask outside {e['id']} domain")
    if kind == "attention" and shape[-1] not in domain["head_dims"]:
        raise ValueError(f"head dimension outside {e['id']} domain")
    scale = config["scale"]
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise ValueError("scale must be a real scalar representable as positive finite FP32")
    try:
        rounded_scale = struct.unpack("f", struct.pack("f", scale))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("scale must be representable as positive finite FP32") from exc
    if not math.isfinite(rounded_scale) or rounded_scale <= 0:
        raise ValueError("scale must be representable as positive finite FP32")


def tensor_contract(e):
    """Logical public shapes; displayed independently of prose formatting."""
    dtype = "/".join(e["domain"]["dtypes"])
    kind = e["kind"]
    if kind == "attention":
        return [["inputs", n, "[B,N,H,D]", dtype] for n in ("q", "k", "v")] + [
            ["outputs", "o", "[B,N,H,D]", dtype], ["outputs", "lse", "[B,H,N]", "fp32"]]
    if kind == "gemm":
        return [["inputs", "a", "[M,K]", dtype], ["inputs", "b", "[K,N]", dtype],
                ["outputs", "c", "[M,N]", "fp32"]]
    return [["inputs", "x", "[R,C]", dtype],
            ["outputs", "y", "[C,R]" if kind == "transpose" else "[R,C]", dtype]]
