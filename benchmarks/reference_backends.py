"""Fast, independent library baselines. Never imported by the student API.

These are mathematical/library compositions, not example student kernels.
Attention uses the public BNHD input/output layout and BHN natural-log LSE.
The cuDNN entry is deliberately explicit: there is no math-backend fallback.
"""
from dataclasses import dataclass, field

import torch

from flash_attention_lab.errors import EnvironmentUnsupported


@dataclass
class ReferencePlan:
    launch: object
    outputs: dict
    provider: str
    scope: str
    workspace: dict = field(default_factory=dict)


def _attention(e, inputs, config):
    q, k, v = (inputs[name].permute(0, 2, 1, 3) for name in ("q", "k", "v"))
    # permute is a view: the baseline receives the original BNHD storage.
    b, h, n, d = q.shape
    outputs = {
        "o": torch.empty_like(inputs["q"]),
        "lse": torch.empty((b, h, n), device=q.device, dtype=torch.float32),
    }
    if q.dtype in (torch.float16, torch.bfloat16):
        try:
            operation = torch.ops.aten._scaled_dot_product_cudnn_attention.default
        except AttributeError as exc:
            raise EnvironmentUnsupported("the pinned PyTorch cuDNN O+LSE baseline is unavailable") from exc

        def launch():
            try:
                out, lse, *_ = operation(
                    q, k, v, None, True, 0.0, config["causal"], False,
                    scale=config["scale"],
                )
            except (RuntimeError, NotImplementedError) as exc:
                raise EnvironmentUnsupported(
                    "explicit cuDNN attention baseline rejected this case; no fallback: " + str(exc)
                ) from exc
            # Normalize to the complete student's output contract. These copies
            # are included in both eager and graph timing, not hidden in setup.
            outputs["o"].copy_(out.permute(0, 2, 1, 3))
            outputs["lse"].copy_(lse.squeeze(-1))

        return ReferencePlan(
            launch, outputs, "torch/cuDNN fused attention (O + LSE)",
            "BNHD storage; BHND input views only; full O+LSE contract; output copies included; "
            "eager includes cuDNN API output/workspace allocation, graph replay excludes allocation; "
            "explicit pinned ATen adapter, no backend fallback",
        )
    if q.dtype != torch.float32:
        raise EnvironmentUnsupported("attention baseline supports fp32/fp16/bf16")

    scores = torch.empty((b, h, n, n), device=q.device, dtype=q.dtype)
    probabilities = torch.empty_like(scores)
    temporary_output = torch.empty_like(q, memory_format=torch.contiguous_format)
    mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu_(1) if config["causal"] else None

    def launch():
        # This materializes the defining expression using vendor operators;
        # there is no online-softmax or tiled-attention implementation here.
        torch.matmul(q, k.transpose(-1, -2), out=scores)
        scores.mul_(config["scale"])
        if mask is not None:
            scores.masked_fill_(mask, -torch.inf)
        torch.logsumexp(scores, -1, out=outputs["lse"])
        torch.softmax(scores, -1, out=probabilities)
        torch.matmul(probabilities, v, out=temporary_output)
        outputs["o"].copy_(temporary_output.permute(0, 2, 1, 3))

    return ReferencePlan(
        launch, outputs, "torch/cuBLAS materialized FP32 attention",
        "full BNHD O+LSE contract; materialized scores/probabilities; FP32 vendor GEMMs; "
        "output layout copy included; eager may allocate internal operator temporaries; "
        "graph replay excludes allocation; not a fused FlashAttention baseline",
        {"scores": scores, "probabilities": probabilities, "temporary_output": temporary_output},
    )


def prepare_reference(e, inputs, config):
    """Prepare a production-library performance reference, independently of students.

    Caller owns validation and timing. Preparation allocates storage; launch does
    no compilation. Low-precision GEMM accumulates into an FP32 result through
    the pinned PyTorch dtype_out cuBLAS operator, without upcasting operands.
    """
    kind = e["kind"]
    if kind == "attention":
        return _attention(e, inputs, config)
    if kind == "gemm":
        a, b = inputs["a"], inputs["b"]
        outputs = {"c": torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)}

        def launch():
            if a.dtype == torch.float32:
                torch.mm(a, b, out=outputs["c"])
            else:
                torch.ops.aten.mm.dtype_out(a, b, torch.float32, out=outputs["c"])

        return ReferencePlan(launch, outputs, "torch/cuBLAS GEMM (FP32 output)",
                             "preallocated output; original operand dtype; FP32 accumulation/output")
    x = inputs.get("x")
    if kind == "softmax":
        outputs = {"y": torch.empty_like(x)}
        return ReferencePlan(lambda: torch.softmax(x, -1, out=outputs["y"]), outputs,
                             "torch native row softmax", "preallocated output; includes normalization")
    if kind == "transpose":
        outputs = {"y": torch.empty(tuple(reversed(x.shape)), dtype=x.dtype, device=x.device)}
        return ReferencePlan(lambda: outputs["y"].copy_(x.T), outputs,
                             "torch native contiguous transpose", "strided read and contiguous output write; preallocated output")
    raise ValueError(f"no fast reference for kind {kind!r}")
