"""GPU kernel lab. No reference fallback in the student-facing API."""


def flash_attn_func(q, k, v, *, exercise="s43_schedule", softmax_scale=None,
                    causal=False, return_lse=False):
    """Forward-only contiguous BNHD self-attention via the selected student kernel.

    O follows the official FlashAttention layout; optional LSE is FP32 BHN.
    This convenience entry allocates output/workspace. Benchmarks use prepare().
    """
    from .api import run
    config = {"causal": causal}
    if softmax_scale is not None:
        config["scale"] = softmax_scale
    outputs = run(exercise, {"q": q, "k": k, "v": v}, config)
    return (outputs["o"], outputs["lse"]) if return_lse else outputs["o"]
