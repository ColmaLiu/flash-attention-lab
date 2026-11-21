"""Self-attention forward.

q, k, v, o: [B, N, H, D]; lse: [B, H, N] FP32 (natural log).
O[b, i, h] = sum_j P[i, j] V[b, j, h],  P = softmax_j(scale * Q K^T + mask)
"""

import math

import torch

from .. import EnvironmentError_
from . import TORCH_DTYPES, compare

AXES = ("b", "n", "h", "d")


def _o_dtype(case):
    default = "bf16" if case.dtype == "e4m3" else case.dtype
    return case.options.get("o_dtype", default)


def make_inputs(case, gen):
    b, n, h, d = case.shape
    dtype = TORCH_DTYPES[case.dtype]

    def randn():
        return torch.randn(b, n, h, d, device="cuda", generator=gen)

    q, k, v = randn(), randn(), randn()
    if case.pattern == "large_logits":
        # Very peaked softmax: tests the running max and rescaling.
        q = q * 8.0
    elif case.pattern == "uniform":
        # Q = 0, so every key gets the same weight and O = mean of visible V.
        q = torch.zeros_like(q)
    elif case.pattern == "ones_v":
        # V = 1, so O must be exactly 1: tests the final 1/l normalization.
        v = torch.ones_like(v)
    elif case.pattern == "position":
        # V[j] = j / N: a wrong mask or K/V offset shows up as a wrong value.
        pos = torch.arange(n, device="cuda", dtype=torch.float32) / n
        v = pos.view(1, n, 1, 1).expand(b, n, h, d).clone()
    if case.dtype == "e4m3":
        # Real value = descale * stored value; store values in a good E4M3 range.
        q, k, v = (t * 4.0 for t in (q, k, v))
    return {"q": q.to(dtype), "k": k.to(dtype), "v": v.to(dtype)}


def make_outputs(inputs, case):
    b, n, h, d = inputs["q"].shape
    return {
        "o": torch.empty(b, n, h, d, device="cuda", dtype=TORCH_DTYPES[_o_dtype(case)]),
        "lse": torch.empty(b, h, n, device="cuda", dtype=torch.float32),
    }


def options(case, inputs):
    d = inputs["q"].shape[-1]
    opts = {
        "scale": case.options.get("scale", 1.0 / math.sqrt(d)),
        "causal": 1.0 if case.options.get("causal", False) else 0.0,
    }
    if case.dtype == "e4m3":
        opts.update(q_descale=0.25, k_descale=0.25, v_descale=0.25)
    return opts


def reference(inputs, options, rows=None):
    """FP64 O [B, R, H, D] and LSE [B, H, R] for query rows `rows` (default all)."""
    q, k, v = (inputs[x].to(torch.float64) for x in ("q", "k", "v"))
    q = q * options.get("q_descale", 1.0)
    k = k * options.get("k_descale", 1.0)
    v = v * options.get("v_descale", 1.0)
    n = q.shape[1]
    if rows is None:
        rows = torch.arange(n, device="cuda")
    q = q[:, rows]
    outs, lses = [], []
    for start in range(0, rows.numel(), 1024):
        r = rows[start : start + 1024]
        s = torch.einsum("bihd,bjhd->bhij", q[:, start : start + 1024], k)
        s = s * options["scale"]
        if options["causal"]:
            mask = torch.arange(n, device="cuda")[None, :] > r[:, None]
            s = s.masked_fill(mask, float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None])
        outs.append(torch.einsum("bhij,bjhd->bihd", p, v))
        lses.append(lse)
    return torch.cat(outs, dim=1), torch.cat(lses, dim=2)


def tolerance(case):
    """(atol, rtol) for O and for LSE."""
    if case.dtype == "fp32":
        return {"o": (2e-5, 1e-4), "lse": (1e-4, 1e-5)}
    o = {"fp16": (2e-3, 5e-3), "bf16": (1e-2, 2e-2)}[_o_dtype(case)]
    if case.dtype == "e4m3":
        o = (5e-2, 1e-1)  # P is quantized to E4M3 (about 2 significant bits)
    return {"o": o, "lse": (1e-3, 1e-4)}


def verify(inputs, outputs, options, case, full=True):
    n = inputs["q"].shape[1]
    rows = None
    if not full and n > 2048:
        # Large perf cases: check a deterministic sample of query rows,
        # always including the first and last row and both ends of a tile.
        rows = torch.unique(
            torch.cat(
                [
                    torch.arange(0, 130, device="cuda"),
                    torch.linspace(0, n - 1, 256, device="cuda").long(),
                    torch.arange(n - 130, n, device="cuda"),
                ]
            )
        )
    o_ref, lse_ref = reference(inputs, options, rows)
    o, lse = outputs["o"], outputs["lse"]
    if rows is not None:
        o, lse = o[:, rows], lse[:, :, rows]
    tol = tolerance(case)
    return {
        "o": compare("o", o, o_ref, *tol["o"], AXES),
        "lse": compare("lse", lse, lse_ref, *tol["lse"], ("b", "h", "n")),
    }


def work(case):
    b, n, h, d = case.shape
    pairs = n * (n + 1) // 2 if case.options.get("causal") else n * n
    size = TORCH_DTYPES[case.dtype].itemsize
    o_size = TORCH_DTYPES[_o_dtype(case)].itemsize
    return {
        "flops": 4 * b * h * d * pairs,
        "bytes": b * n * h * d * (3 * size + o_size) + b * h * n * 4,
    }


def _cudnn(inputs, options, case):
    """cuDNN fused attention through PyTorch (no fallback to other backends)."""
    if case.dtype not in ("fp16", "bf16"):
        raise EnvironmentError_("cuDNN baseline supports FP16/BF16 only")
    q, k, v = (inputs[x].transpose(1, 2) for x in ("q", "k", "v"))  # BHND views
    op = torch.ops.aten._scaled_dot_product_cudnn_attention.default
    o = torch.empty_like(inputs["q"])
    lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], device="cuda")

    def launch():
        out, l, *_ = op(
            q,
            k,
            v,
            None,
            True,
            0.0,
            bool(options["causal"]),
            False,
            scale=options["scale"],
        )
        # Copies into the BNHD output contract are part of the timed work.
        o.copy_(out.transpose(1, 2))
        lse.copy_(l.reshape(lse.shape))

    return launch, {"o": o, "lse": lse}


def _fa4(inputs, options, case):
    """FlashAttention-4 (CuTe DSL) from third_party/flash-attention, if installed."""
    try:
        from flash_attn.cute.interface import flash_attn_func
    except Exception as exc:
        raise EnvironmentError_(f"FA4 not installed: {exc}") from exc
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    extra = {}
    if case.dtype == "e4m3":
        extra = {
            "q_descale": torch.full(
                (q.shape[0], q.shape[2]), options["q_descale"], device="cuda"
            ),
            "k_descale": torch.full(
                (q.shape[0], q.shape[2]), options["k_descale"], device="cuda"
            ),
            "v_descale": torch.full(
                (q.shape[0], q.shape[2]), options["v_descale"], device="cuda"
            ),
        }
    result = {}

    def launch():
        out = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=options["scale"],
            causal=bool(options["causal"]),
            return_lse=True,
            **extra,
        )
        result["o"], result["lse"] = out[0], out[1]

    return launch, result


def _efficient(inputs, options, case):
    """PyTorch memory-efficient attention: the library baseline for FP32."""
    if case.dtype != "fp32":
        raise EnvironmentError_("used as the FP32 baseline only")
    q, k, v = (inputs[x].transpose(1, 2) for x in ("q", "k", "v"))  # BHND views
    op = torch.ops.aten._scaled_dot_product_efficient_attention.default
    o = torch.empty_like(inputs["q"])
    lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], device="cuda")

    def launch():
        out, l, *_ = op(
            q, k, v, None, True, 0.0, bool(options["causal"]), scale=options["scale"]
        )
        o.copy_(out.transpose(1, 2))
        lse.copy_(l[..., : lse.shape[-1]])

    return launch, {"o": o, "lse": lse}


def baselines(case=None):
    if case is not None and case.dtype == "fp32":
        return {"efficient": _efficient}
    return {"cudnn": _cudnn, "fa4": _fa4}


def _scores11(inputs, options):
    """FP64 S = scale * Q K^T as [B, H, N, N], without any mask."""
    q, k = (inputs[x].to(torch.float64) for x in ("q", "k"))
    return torch.einsum("bihd,bjhd->bhij", q, k) * options["scale"]


def probe_scores11(inputs, options, case):
    """Lab 11 step 1: the probe holds S = scale * Q K^T, [B, H, N, N].

    The causal mask is applied later (step 2), so S covers every (i, j).
    """
    return _scores11(inputs, options), (1e-4, 1e-4)


def probe_probs11(inputs, options, case):
    """Lab 11 step 2: the probe holds P = softmax_j(S + mask), [B, H, N, N].

    Masked entries (j > i when causal) must be 0.
    """
    s = _scores11(inputs, options)
    if options["causal"]:
        n = s.shape[-1]
        mask = torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1), (2e-6, 1e-4)


def probe_fa2_scores(inputs, options, case):
    """Lab 22 step 1: the probe holds S = scale * Q K^T, [B, H, N, N].

    Computed with mma.sync from FP16/BF16 inputs and FP32 accumulation; the
    causal mask is not applied (it comes in step 3), so S covers every (i, j).
    """
    return _scores11(inputs, options), (1e-3, 1e-3)
