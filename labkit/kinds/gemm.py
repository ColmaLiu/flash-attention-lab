"""c[M, N] = a_scale * b_scale * (a[M, K] @ b[N, K]^T)"""

import math

import torch

from . import TORCH_DTYPES, compare


def _dims(case):
    m, n, k = case.shape
    return m, n, k


def _c_dtype(case):
    return case.options.get("c_dtype", "fp32")


def make_inputs(case, gen):
    m, n, k = _dims(case)
    dtype = TORCH_DTYPES[case.dtype]
    if case.pattern == "int":
        # Small integers: every product and partial sum is exact, so any
        # difference from the reference is a bug, not rounding.
        def make(r, c):
            return torch.randint(-2, 3, (r, c), device="cuda", generator=gen).to(dtype)
    else:

        def make(r, c):
            return torch.randn(r, c, device="cuda", generator=gen).to(dtype)

    inputs = {"a": make(m, k), "b": make(n, k)}
    return inputs


def make_outputs(inputs, case):
    m, n, _ = _dims(case)
    return {"c": torch.empty(m, n, device="cuda", dtype=TORCH_DTYPES[_c_dtype(case)])}


def options(case, inputs):
    opts = {}
    if case.dtype == "e4m3":
        opts = {
            "a_scale": case.options.get("a_scale", 0.5),
            "b_scale": case.options.get("b_scale", 0.25),
        }
    return opts


def _reference(inputs, options, rows):
    a = inputs["a"][rows].to(torch.float64)
    b = inputs["b"].to(torch.float64)
    scale = options.get("a_scale", 1.0) * options.get("b_scale", 1.0)
    return (a @ b.t()) * scale


def tolerance(case):
    _, _, k = _dims(case)
    if case.pattern == "int" and _c_dtype(case) == "fp32" and case.dtype != "e4m3":
        return 0.0, 0.0
    # Rounding error of a length-K FP32 dot product grows ~ sqrt(K).
    atol = (2e-5 if case.dtype == "e4m3" else 1e-5) * math.sqrt(k)
    rtol = {"fp32": 1e-5, "bf16": 8e-3, "fp16": 1e-3}[_c_dtype(case)]
    if case.dtype == "e4m3":
        atol *= 50
    return atol, rtol


def verify(inputs, outputs, options, case, full=True):
    m = inputs["a"].shape[0]
    if full or m <= 1024:
        rows = slice(None)
    else:
        # Check a deterministic sample of rows for very large problems.
        rows = torch.linspace(0, m - 1, 512, device="cuda").long()
    expected = _reference(inputs, options, rows)
    atol, rtol = tolerance(case)
    scale = options.get("a_scale", 1.0) * options.get("b_scale", 1.0)
    return {
        "c": compare("c", outputs["c"][rows], expected, atol * scale, rtol, ("m", "n"))
    }


def work(case):
    m, n, k = _dims(case)
    size = TORCH_DTYPES[case.dtype].itemsize
    c_size = TORCH_DTYPES[_c_dtype(case)].itemsize
    return {"flops": 2 * m * n * k, "bytes": (m * k + n * k) * size + m * n * c_size}


def baselines():
    def cublas(inputs, options, case):
        a, b = inputs["a"], inputs["b"]
        m, n = a.shape[0], b.shape[0]
        c_dtype = TORCH_DTYPES[_c_dtype(case)]
        if a.dtype == torch.float8_e4m3fn:
            sa = torch.tensor(options["a_scale"], device="cuda")
            sb = torch.tensor(options["b_scale"], device="cuda")
            c = torch.empty(m, n, device="cuda", dtype=c_dtype)

            def launch():
                torch._scaled_mm(a, b.t(), sa, sb, out_dtype=c_dtype, out=c)

            return launch, {"c": c}
        c = torch.empty(m, n, device="cuda", dtype=c_dtype)
        if a.dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = False  # true FP32 SGEMM
        if c_dtype == a.dtype:
            return (lambda: torch.mm(a, b.t(), out=c)), {"c": c}
        return (lambda: torch.mm(a, b.t(), out_dtype=c_dtype, out=c)), {"c": c}

    return {"cublas": cublas}
