"""Fast references obey BNHD/BHN contracts and never hide backend failure."""
import pytest
import torch

from benchmarks.reference_backends import prepare_reference
from flash_attention_lab.accuracy import compare
from flash_attention_lab.errors import CorrectnessFailure, EnvironmentUnsupported
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.runner import _assert_unchanged, _flops


@pytest.mark.parametrize("kind,shape", [("transpose", (7, 33)), ("softmax", (4, 257)), ("gemm", (7, 11, 13))])
def test_native_references_match_independent_definition(kind, shape):
    generator = torch.Generator().manual_seed(101)
    if kind == "gemm":
        m, n, k = shape
        inputs = {"a": torch.randn(m, k, generator=generator), "b": torch.randn(k, n, generator=generator)}
    else:
        inputs = {"x": torch.randn(shape, generator=generator)}
    saved = {name: tensor.clone() for name, tensor in inputs.items()}
    e = {"kind": kind}
    plan = prepare_reference(e, inputs, {})
    expected = mathematical_reference(kind, inputs, {})
    for _ in range(2):
        for tensor in plan.outputs.values():
            tensor.fill_(float("nan"))
        plan.launch()
        compare(e, plan.outputs, expected, inputs)
    _assert_unchanged(inputs, saved)
    assert all(value.is_contiguous() for value in plan.outputs.values())


@pytest.mark.parametrize("causal", [False, True])
def test_materialized_attention_uses_sequence_axis_and_writes_contiguous_bnhd(causal):
    # B, N and H all differ; accidental BHND interpretation cannot pass.
    generator = torch.Generator().manual_seed(37)
    inputs = {name: torch.randn(2, 7, 3, 8, generator=generator) for name in ("q", "k", "v")}
    config = {"scale": 0.37, "causal": causal}
    e = {"kind": "attention"}
    plan = prepare_reference(e, inputs, config)
    plan.launch()
    compare(e, plan.outputs, mathematical_reference("attention", inputs, config), inputs)
    assert plan.outputs["o"].shape == (2, 7, 3, 8)
    assert plan.outputs["lse"].shape == (2, 3, 7)
    assert all(value.is_contiguous() for value in plan.outputs.values())


def test_cudnn_adapter_normalizes_lse_and_preserves_original_bnhd_storage(monkeypatch):
    inputs = {name: torch.ones(2, 7, 3, 8, dtype=torch.float16) for name in ("q", "k", "v")}
    config = {"causal": True, "scale": 0.5}
    calls = []
    def operation(q, k, v, bias, need_lse, dropout, causal, debug, *, scale):
        calls.append((q.shape, q.stride(), q.data_ptr(), need_lse, dropout, causal, scale))
        return q + 2, torch.full((2, 3, 7, 1), 4.0), *([None] * 7)
    monkeypatch.setattr(torch.ops.aten._scaled_dot_product_cudnn_attention, "default", operation)
    plan = prepare_reference({"kind": "attention"}, inputs, config)
    plan.launch()
    assert calls == [(torch.Size([2, 3, 7, 8]), (168, 8, 24, 1), inputs["q"].data_ptr(), True, 0.0, True, 0.5)]
    assert torch.equal(plan.outputs["o"], torch.full_like(inputs["q"], 3))
    assert torch.equal(plan.outputs["lse"], torch.full((2, 3, 7), 4.0))


def test_cudnn_failure_is_explicit_and_does_not_fallback(monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported layout")
    monkeypatch.setattr(torch.ops.aten._scaled_dot_product_cudnn_attention, "default", unsupported)
    inputs = {name: torch.ones(1, 3, 2, 8, dtype=torch.bfloat16) for name in ("q", "k", "v")}
    plan = prepare_reference({"kind": "attention"}, inputs, {"causal": False, "scale": 1.0})
    with pytest.raises(EnvironmentUnsupported, match="no fallback.*unsupported layout"):
        plan.launch()


def test_hot_loop_input_corruption_is_rejected_against_fixed_snapshot():
    inputs = {"x": torch.ones(2, 3)}
    saved = {"x": inputs["x"].clone()}
    inputs["x"][0, 0] = 2
    with pytest.raises(CorrectnessFailure, match="input x modified"):
        _assert_unchanged(inputs, saved)


def test_attention_flops_use_bnhd_sequence_dimension():
    record = _flops({"kind": "attention"}, {"shape": [2, 7, 3, 8]}, {"causal": True}, {"gpu_stream_ms_median": 1})
    assert record["useful"] == 4 * 2 * 3 * 28 * 8
    assert record["dense_equivalent"] == 4 * 2 * 3 * 49 * 8
