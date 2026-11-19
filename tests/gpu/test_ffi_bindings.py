"""Run explicitly on the selected GPU: ownership, streams, and captured launches."""

import gc
import inspect
from types import SimpleNamespace

import pytest
import torch

from flash_attention_lab.accuracy import compare
from flash_attention_lab.api import prepare
from flash_attention_lab.build import build_cuda
from flash_attention_lab.cases import make_inputs
from flash_attention_lab.environment import require_environment
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.registry import EXERCISES, load_student

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def selected_gpu():
    require_environment("cuda")
    assert torch.cuda.device_count() == 1


@pytest.mark.parametrize(
    "name",
    [
        "s01_softmax",
        "s30_tma",
        "s31_tcgen",
        "s32_specialize",
        "s40_attention",
        "s41_overlap",
        "s42_numerics",
        "s43_schedule",
    ],
)
def test_launch_owns_tensors_and_uses_prepared_stream_through_graph_capture(name):
    exercise = EXERCISES[name]
    inputs, config = make_inputs(exercise, exercise["cases"][0], "cuda")
    config = {**load_student(exercise, solution=True).CONFIG, **config}
    expected = mathematical_reference(exercise["kind"], inputs, config)
    dtype = next(iter(inputs.values())).dtype
    shapes = [(tuple(t.shape), t.dtype) for t in inputs.values()]
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan = prepare(name, inputs, config, solution=True)
        launch, outputs = plan.launch, plan.outputs
        for tensor in outputs.values():
            tensor.fill_(float("nan"))
    stream.synchronize()
    del plan, inputs
    gc.collect()
    # Reuse would overwrite released inputs if the FFI arguments/state only
    # remembered addresses. The original Python input objects are now gone.
    with torch.cuda.stream(stream):
        churn = [
            torch.full(shape, float("nan"), dtype=kind, device="cuda")
            for shape, kind in shapes
        ]
    try:
        with torch.cuda.stream(other):
            torch.cuda._sleep(20_000_000)
            launch()
        stream.synchronize()
        compare(exercise, outputs, expected, {"dtype": SimpleNamespace(dtype=dtype)})
    finally:
        other.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    with torch.cuda.stream(stream):
        for tensor in outputs.values():
            tensor.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    compare(exercise, outputs, expected, {"dtype": SimpleNamespace(dtype=dtype)})
    assert all(torch.isnan(t).all() for t in churn)


def test_prepared_states_reject_none_and_other_modules():
    exercise = EXERCISES["s31_tcgen"]
    inputs, config = make_inputs(exercise, exercise["cases"][0], "cuda")
    plan = prepare(exercise["id"], inputs, config, solution=True)
    state = inspect.getclosurevars(plan.launch).nonlocals["state"]
    gemm = build_cuda("s31_tcgen", solution=True)
    attention = build_cuda("s43_schedule", solution=True)
    with pytest.raises(RuntimeError, match="must not be None"):
        gemm.run(None)
    with pytest.raises((TypeError, RuntimeError), match="Prepared|type"):
        attention.run(state)


def test_split_k_workspace_remains_accounted_and_owned():
    exercise = EXERCISES["s32_specialize"]
    case = {**exercise["cases"][0], "shape": [256, 64, 1024]}
    inputs, config = make_inputs(exercise, case, "cuda")
    config["split_k"] = 2
    expected = mathematical_reference("gemm", inputs, config)
    plan = prepare(exercise["id"], inputs, config, solution=True)
    assert plan.extra_workspace_bytes == 2 * 256 * 64 * 4
    for _ in range(2):
        plan.outputs["c"].fill_(float("nan"))
        plan.launch()
        torch.cuda.synchronize()
        compare(exercise, plan.outputs, expected, inputs)


def test_native_boundary_rejects_bad_tensor_metadata_before_launch():
    lib = build_cuda("s31_tcgen", solution=True)
    a = torch.ones(64, 32, dtype=torch.bfloat16, device="cuda")
    b = torch.ones(32, 64, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(64, 64, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(RuntimeError, match="BF16"):
        lib.prepare(a.float(), b, c, stream)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        lib.prepare(a, b, c[:32], stream)
    with pytest.raises(RuntimeError, match="contiguous"):
        lib.prepare(a, b.T, c, stream)
    with pytest.raises(RuntimeError, match="CUDA"):
        lib.prepare(a.cpu(), b, c, stream)
