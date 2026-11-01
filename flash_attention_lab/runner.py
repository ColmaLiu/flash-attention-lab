import json
from datetime import datetime, timezone
import torch
from .accuracy import compare
from .api import prepare, require_implemented
from .cases import make_inputs
from .environment import require_environment, snapshot
from .errors import CorrectnessFailure
from .reference import mathematical_reference
from .registry import ROOT, fingerprint, load_student
from .timing import measure, measure_graph, measure_graph_pair
from .reporting import save_artifact  # retained for callers of the original entry point


def validate_case(e, case, *, nondefault=True, solution=False):
    inputs, cfg = make_inputs(e, case, "cuda")
    cfg = {**load_student(e, solution=solution).CONFIG, **cfg}
    saved = {n: t.clone() for n, t in inputs.items()}
    reference = mathematical_reference(e["kind"], inputs, cfg)
    stream = torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan = prepare(e["id"], inputs, cfg, solution=solution)
        # Poison all outputs before each independent launch. Workspace initialization is a student's responsibility.
        for repeat in range(2):
            for t in plan.outputs.values(): t.fill_(float("nan"))
            plan.launch()
            stream.synchronize()
            metrics = compare(e, plan.outputs, reference, inputs)
            for n, t in inputs.items():
                if not torch.equal(t, saved[n]): raise CorrectnessFailure(f"input {n} modified")
    torch.cuda.current_stream().wait_stream(stream)
    return metrics


def check(e, case_index=None, *, solution=False):
    if not solution:
        require_implemented(e)
    require_environment(e["backend"])
    indices = list(range(len(e["cases"]))) if case_index is None else [case_index]
    results = []
    cases = [{"case": i, "input": e["cases"][i], "state": "NOT_RUN"} for i in indices]
    report = {"exercise": e["id"], "state": "CORRECTNESS_PASSED", "full_contract": case_index is None,
              "case_count": len(indices), "cases": cases, "results": results, "fingerprint": fingerprint(e),
              "timestamp": datetime.now(timezone.utc).isoformat(), "design_review": "NOT_REVIEWED",
              "implementation": "solution" if solution else "student"}
    failures = []
    for item in cases:
        i = item["case"]
        case = e["cases"][i]
        item["state"] = "PASS"
        for nondefault in (False, True):
            record = {"case": i, "nondefault_stream": nondefault}
            try:
                metrics = validate_case(e, case, nondefault=nondefault, **({"solution": True} if solution else {}))
            except CorrectnessFailure as exc:
                # Numeric counterexamples are independent; still test the other cases.
                item["state"] = "FAIL"
                record.update(state="FAIL", message=str(exc))
                failures.append(f"case={i} nondefault_stream={nondefault}: {exc}")
            except Exception as exc:
                # A CUDA error can poison the context. Preserve partial results and stop.
                item["state"] = "ERROR"
                results.append({**record, "state": "ERROR", "message": str(exc)})
                report["state"] = "EXECUTION_ERROR"
                exc.check_report = report
                raise
            else:
                record.update(state="PASS", errors=metrics)
            results.append(record)
    if failures:
        report["state"] = "CORRECTNESS_FAILED"
        exc = CorrectnessFailure("\n".join(failures))
        exc.check_report = report
        raise exc
    return report


def save_result(e, result):
    path = ROOT / "progress" / "results"
    path.mkdir(parents=True, exist_ok=True)
    (path / (e["id"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")


def _assert_unchanged(inputs, saved):
    for name, tensor in inputs.items():
        if not torch.equal(tensor, saved[name]):
            raise CorrectnessFailure(f"input {name} modified during benchmark")


def _measure_plan(e, plan, inputs, saved, oracle, warmup, repeats, iterations, *, graph=True):
    # Validate before exposing either timing, and again after each warm loop.
    plan.launch()
    torch.cuda.current_stream().synchronize()
    errors = compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    eager = measure(plan.launch, warmup, repeats, iterations)
    compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    graph_timing = measure_graph(plan.launch, warmup, repeats, iterations) if graph else None
    post_errors = compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    return {"timing": graph_timing, "eager_timing": eager, "errors": errors,
            "post_timing_errors": post_errors}


def _flops(e, case, cfg, timing):
    if e["kind"] == "attention":
        b, n, h, d = case["shape"]
        dense = 4 * b * h * n * n * d
        pairs = n * (n + 1) // 2 if cfg["causal"] else n * n
        useful = 4 * b * h * pairs * d
        seconds = timing["gpu_stream_ms_median"] / 1e3
        return {"convention": "QK and PV only, FMA=2; excludes softmax, padding and rescaling",
                "useful": useful, "dense_equivalent": dense,
                "useful_tflops": useful / seconds / 1e12,
                "dense_equivalent_tflops": dense / seconds / 1e12}
    if e["kind"] == "gemm":
        m, n, k = case["shape"]
        operations = 2 * m * n * k
        seconds = timing["gpu_stream_ms_median"] / 1e3
        return {"convention": "GEMM: 2*M*N*K, FMA=2", "useful": operations,
                "useful_tflops": operations / seconds / 1e12}
    return None


def reference(e, case_index, warmup=5, repeats=7, iterations=10):
    """Validate/time a library baseline without loading any student module."""
    from benchmarks.reference_backends import prepare_reference
    require_environment(e["backend"])
    case = e["cases"][case_index]
    inputs, cfg = make_inputs(e, case, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        saved = {name: tensor.clone() for name, tensor in inputs.items()}
        oracle = mathematical_reference(e["kind"], inputs, cfg)
        plan = prepare_reference(e, inputs, cfg)
        result = _measure_plan(e, plan, inputs, saved, oracle, warmup, repeats, iterations)
    torch.cuda.current_stream().wait_stream(stream)
    result.update(exercise=e["id"], case_index=case_index, case=case, config=cfg,
                  provider=plan.provider, scope=plan.scope,
                  state="REFERENCE_VALIDATED", student_state="NOT_TESTED",
                  fingerprint=fingerprint(e), environment=snapshot())
    flops = _flops(e, case, cfg, result["timing"])
    if flops:
        result["flops"] = flops
    return result


def benchmark(e, case_index, warmup, repeats, iterations, *, solution=False):
    """Compare the validated student and fast library on identical inputs.

    CUDA graph replay is the comparison metric. Eager API spans are separately
    reported because the library may allocate internally and Python dispatch
    costs differ. Output layout copies are inside the library's timed launch.
    """
    from benchmarks.reference_backends import prepare_reference
    validation = check(e, case_index, solution=solution)
    case = e["cases"][case_index]
    inputs, cfg = make_inputs(e, case, "cuda")
    cfg = {**load_student(e, solution=solution).CONFIG, **cfg}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        saved = {name: tensor.clone() for name, tensor in inputs.items()}
        # Oracle is fixed before either implementation can touch the inputs.
        oracle = mathematical_reference(e["kind"], inputs, cfg)
        plan = prepare(e["id"], inputs, cfg, solution=solution)
        baseline = prepare_reference(e, inputs, cfg)
        measured = _measure_plan(e, plan, inputs, saved, oracle, warmup, repeats, iterations, graph=False)
        fast = _measure_plan(e, baseline, inputs, saved, oracle, warmup, repeats, iterations, graph=False)
        measured["timing"], fast["timing"] = measure_graph_pair(plan.launch, baseline.launch, warmup, repeats, iterations)
        measured["post_timing_errors"] = compare(e, plan.outputs, oracle, inputs)
        fast["post_timing_errors"] = compare(e, baseline.outputs, oracle, inputs)
        _assert_unchanged(inputs, saved)
    torch.cuda.current_stream().wait_stream(stream)
    student_ms = measured["timing"]["gpu_stream_ms_median"]
    reference_ms = fast["timing"]["gpu_stream_ms_median"]
    result = {
        "exercise": e["id"], "case_index": case_index, "case": case, "config": cfg,
        "fingerprint": fingerprint(e), "validation": validation, "environment": snapshot(),
        "implementation": "solution" if solution else "student",
        **measured,
        "reference": {**fast, "provider": baseline.provider, "scope": baseline.scope},
        "comparison": {
            "reference_over_student": reference_ms / student_ms,
            "student_over_reference": student_ms / reference_ms,
            "metric": "median CUDA graph replay time; same inputs, output contract and captured batch size",
            "interpretation": "reference_over_student > 1 means the candidate is faster for this case; not a learning-completion gate",
        },
        "output_and_workspace_bytes": sum(t.numel() * t.element_size()
                                          for t in list(plan.outputs.values()) + list(plan.workspace.values())) + plan.extra_workspace_bytes,
    }
    flops = _flops(e, case, cfg, result["timing"])
    if flops:
        result["flops"] = flops
    return result
