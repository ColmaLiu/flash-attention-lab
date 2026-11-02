"""Exercise benchmark orchestration with real CPU tensors and simulated CUDA timing.

These tests verify data/configuration isolation and reporting. They do not certify
GPU kernels or substitute CPU work for a GPU performance measurement.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from benchmarks import reference_backends
from flash_attention_lab import runner, timing
from flash_attention_lab.errors import CorrectnessFailure


@pytest.fixture
def comparison(monkeypatch):
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    inputs = {"x": torch.tensor([[1., -2., 3.], [2., 4., -1.]])}
    config = {"scale": 1., "causal": False}
    exercise = {"id": "fixture", "kind": "softmax", "cases": [
        {"shape": [2, 3], "dtype": "fp32", "pattern": "random"}]}
    observed = {"prepared": [], "checks": [], "timed": [], "invalid": None}
    monkeypatch.setattr(runner, "make_inputs", lambda *args: (inputs, config))
    monkeypatch.setattr(runner, "load_student", lambda e, *, solution=False:
                        SimpleNamespace(CONFIG={"tile": 64 if solution else 32}))
    monkeypatch.setattr(runner, "fingerprint", lambda e: "fixture")
    monkeypatch.setattr(runner, "snapshot", lambda: {})

    def check(e, index, *, solution):
        observed["checks"].append(solution)
        return {"state": "CORRECTNESS_PASSED", "implementation": "solution" if solution else "student"}

    def prepare(name, values, cfg):
        observed["prepared"].append((name, values, dict(cfg)))
        outputs = {"y": torch.empty_like(values["x"])}

        def launch():
            outputs["y"].copy_(values["x"].softmax(-1))
            if observed["invalid"] == name:
                outputs["y"][0, 0] = 100.

        launch.name = name
        return SimpleNamespace(launch=launch, outputs=outputs, workspace={},
                               extra_workspace_bytes=0, provider="fixture " + name, scope="fixture")

    def measure(launch, *args):
        observed["timed"].append(("eager", launch.name))
        launch()
        return {"gpu_stream_ms_median": {"student": 30., "answer": 20., "library": 10.}[launch.name]}

    def graph_group(launches, *args):
        result = []
        for launch in launches:
            observed["timed"].append(("graph", launch.name))
            launch()
            result.append({"gpu_stream_ms_median": {"student": 3., "answer": 2., "library": 1.}[launch.name]})
        return result

    monkeypatch.setattr(runner, "check", check)
    monkeypatch.setattr(runner, "prepare", lambda identifier, values, cfg, *, solution:
                        prepare("answer" if solution else "student", values, cfg))
    monkeypatch.setattr(reference_backends, "prepare_reference", lambda e, values, cfg:
                        prepare("library", values, cfg))
    monkeypatch.setattr(runner, "measure", measure)
    monkeypatch.setattr(runner, "measure_graph_group", graph_group)
    return exercise, inputs, observed


def test_student_benchmark_compares_three_independent_configs_and_graph_times(comparison):
    exercise, inputs, observed = comparison
    result = runner.benchmark(exercise, 0, 1, 3, 2)
    assert observed["checks"] == [False, True]
    assert [name for name, _, _ in observed["prepared"]] == ["student", "answer", "library"]
    assert all(values is inputs for _, values, _ in observed["prepared"])
    assert [cfg.get("tile") for _, _, cfg in observed["prepared"]] == [32, 64, None]
    assert all(cfg["scale"] == 1. and cfg["causal"] is False for _, _, cfg in observed["prepared"])
    assert result["implementation"] == "student"
    assert result["comparison"]["student_over_reference"] == 3.
    assert result["comparison"]["student_over_solution"] == 1.5
    assert result["comparison"]["solution_over_reference"] == 2.
    assert result["solution"]["eager_timing"]["gpu_stream_ms_median"] == 20.
    assert result["solution"]["timing"]["gpu_stream_ms_median"] == 2.


def test_solution_benchmark_does_not_prepare_or_validate_a_student(comparison):
    exercise, _, observed = comparison
    result = runner.benchmark(exercise, 0, 1, 3, 2, solution=True)
    assert observed["checks"] == [True]
    assert [name for name, _, _ in observed["prepared"]] == ["answer", "library"]
    assert result["implementation"] == "solution" and "solution" not in result
    assert result["comparison"]["student_over_reference"] == 2.


@pytest.mark.parametrize("invalid", ["student", "answer", "library"])
def test_no_participant_is_timed_until_all_participants_pass(comparison, invalid):
    exercise, _, observed = comparison
    observed["invalid"] = invalid
    with pytest.raises(CorrectnessFailure):
        runner.benchmark(exercise, 0, 1, 3, 2)
    assert not observed["timed"]


def test_graph_samples_stay_with_their_candidate_when_order_rotates(monkeypatch):
    # Three candidates with distinct simulated durations expose accidental
    # cross-candidate sample assignment as well as per-operation scaling errors.
    clock = {"value": 0., "capturing": None}
    replays = []
    stream = SimpleNamespace(synchronize=lambda: None)

    class Graph:
        def replay(self):
            replays.append(self.name)
            clock["value"] += self.duration

    class Capture:
        def __init__(self, graph, *, stream):
            self.graph = graph

        def __enter__(self):
            self.graph.duration = 0.
            clock["capturing"] = self.graph

        def __exit__(self, *args):
            clock["capturing"] = None

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            self.time = clock["value"]

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.time - self.time

    def candidate(name, duration):
        def launch():
            graph = clock["capturing"]
            if graph is not None:
                graph.name = name
                graph.duration += duration
        return launch

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", Capture)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    results = timing.measure_graph_group([candidate("A", 2.), candidate("B", 3.), candidate("C", 5.)],
                                         warmup=1, repeats=3, iterations=4)
    assert replays == ["A", "B", "C", "A", "B", "C", "B", "C", "A", "C", "A", "B"]
    assert [r["gpu_stream_ms_samples"] for r in results] == [[2.] * 3, [3.] * 3, [5.] * 3]
    assert all(r["mode"] == "cuda_graph" and r["iterations"] == 4 for r in results)
