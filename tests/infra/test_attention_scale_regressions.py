"""Public extreme-scale cases preserve finite analytic answers and real faults."""

import math
import struct

import pytest
import torch

from flash_attention_lab.accuracy import compare
from flash_attention_lab.cases import make_inputs
from flash_attention_lab.errors import CorrectnessFailure
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.registry import EXERCISES


STAGES = ("s22_fa2", "s40_attention", "s41_overlap", "s42_numerics", "s43_schedule")


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("case_id", ("scale_overflow_uniform", "scale_subnormal"))
def test_extreme_scale_cases_remain_in_the_public_domain_with_finite_oracles(
    stage, case_id
):
    exercise = EXERCISES[stage]
    case = next(case for case in exercise["cases"] if case["id"] == case_id)
    assert case["suite"] != "perf"
    domain = exercise["domain"]
    assert case["dtype"] in domain["dtypes"] and case["causal"] in domain["causal"]
    assert all(
        size % multiple == 0
        for size, multiple in zip(case["shape"], domain["shape_multiple"])
    )
    inputs, config = make_inputs(exercise, case)
    assert torch.count_nonzero(inputs["v"]) > 0
    fp32_scale = struct.unpack("f", struct.pack("f", config["scale"]))[0]
    assert math.isfinite(fp32_scale) and fp32_scale > 0
    if case_id == "scale_overflow_uniform":
        assert config["scale"] == 3e38
        assert torch.count_nonzero(inputs["q"]) == torch.count_nonzero(inputs["k"]) == 0
        assert torch.isinf(torch.tensor(fp32_scale) * torch.tensor(math.log2(math.e)))
    else:
        assert config["scale"] == fp32_scale == 2.0**-149
        assert (
            torch.count_nonzero(inputs["q"]) > 0
            and torch.count_nonzero(inputs["k"]) > 0
        )
    expected = mathematical_reference("attention", inputs, config)
    assert all(torch.isfinite(value).all() for value in expected.values())
    mean = inputs["v"].double().mean(1, keepdim=True).expand_as(expected["o"])
    torch.testing.assert_close(expected["o"], mean)
    torch.testing.assert_close(
        expected["lse"], torch.full_like(expected["lse"], math.log(case["shape"][1]))
    )


def test_wide_raw_scores_have_finite_exact_fp32_dots_but_expose_difference_overflow():
    exercise = EXERCISES["s22_fa2"]
    case = next(case for case in exercise["cases"] if case["id"] == "scale_raw_gap")
    inputs, config = make_inputs(exercise, case)
    q, k, v = (inputs[name].permute(0, 2, 1, 3).double() for name in ("q", "k", "v"))
    raw64 = q @ k.transpose(-1, -2)
    exact = torch.tensor([2.0**127, -(2.0**127)], dtype=torch.float64).expand_as(raw64)
    assert torch.equal(raw64, exact)
    assert torch.equal(raw64.float().double(), exact)
    assert torch.isfinite(raw64.float()).all()
    assert config["scale"] == torch.finfo(torch.float32).tiny
    # Every dot contains one exact product plus zeros; it is the later raw
    # subtraction, not a tensor-core QK accumulation, that exceeds FP32 range.
    raw32 = raw64.float()
    maximum = raw32.amax(-1, keepdim=True)
    difference = raw32 - maximum
    assert torch.isneginf(difference[..., 1]).all()
    scale2 = torch.tensor(config["scale"]) * torch.tensor(math.log2(math.e))
    assert torch.finfo(torch.float32).tiny <= scale2 <= 1.0
    weights = torch.exp2(difference * scale2)
    denominator = weights.sum(-1, keepdim=True)
    broken = {
        "o": ((weights / denominator) @ v.float())
        .permute(0, 2, 1, 3)
        .contiguous()
        .bfloat16(),
        "lse": (maximum * config["scale"] + denominator.log()).squeeze(-1),
    }
    expected = mathematical_reference("attention", inputs, config)
    torch.testing.assert_close(
        expected["o"], torch.full_like(expected["o"], 4.0 / (1.0 + math.exp(4.0)))
    )
    torch.testing.assert_close(
        expected["lse"],
        torch.full_like(expected["lse"], 2.0 + math.log1p(math.exp(-4.0))),
    )
    with pytest.raises(CorrectnessFailure):
        compare(exercise, broken, expected, inputs)


@pytest.mark.parametrize(
    "case_id",
    (
        "scaled_max_fp16",
        "scaled_max_fp16_multi",
        "scaled_max_bf16",
        "scaled_max_bf16_multi",
    ),
)
def test_scaled_max_cases_expose_fma_residual_with_finite_exact_dots(case_id):
    exercise = EXERCISES["s43_schedule"]
    case = next(case for case in exercise["cases"] if case["id"] == case_id)
    assert case["suite"] != "perf"
    inputs, config = make_inputs(exercise, case)
    value = inputs["q"][0, 0, 0, 0].double()
    assert torch.equal(inputs["q"], inputs["k"])
    assert torch.count_nonzero(inputs["q"][..., 1:]) == 0
    assert torch.all(inputs["q"][..., 0] == value)
    raw = value * value
    assert torch.isfinite(raw.float()) and raw.float().double() == raw
    assert config["scale"] == 0.125
    expected = mathematical_reference("attention", inputs, config)
    assert all(torch.isfinite(t).all() for t in expected.values())
    torch.testing.assert_close(expected["o"], torch.ones_like(expected["o"]))
    analytic_lse = raw * config["scale"] + math.log(case["shape"][1])
    torch.testing.assert_close(expected["lse"], analytic_lse.expand_as(expected["lse"]))

    # Two FP32 factors have an exact FP64 product here. One final FP32
    # rounding therefore reproduces fma(score, scale2, -rounded_max).
    scale2 = torch.tensor(config["scale"]) * torch.tensor(math.log2(math.e))
    rounded_max = raw.float() * scale2
    residual = (raw * scale2.double() - rounded_max.double()).float()
    probability = torch.exp2(residual)
    # With identical scores and V=1, the erroneous P cast alone predicts O.
    # No raw QK, mathematical LSE, or correct O overflows in these cases.
    broken_o = probability.to(inputs["q"].dtype).float() / probability
    assert not torch.isfinite(broken_o)
    broken = {"o": broken_o.expand_as(inputs["v"]), "lse": expected["lse"].float()}
    with pytest.raises(CorrectnessFailure):
        compare(exercise, broken, expected, inputs)
