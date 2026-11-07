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
from .reporting import save_artifact as save_artifact  # Compatibility export.
from .timing import measure, measure_graph, measure_graph_group


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
            for t in plan.outputs.values():
                t.fill_(float("nan"))
            plan.launch()
            stream.synchronize()
            metrics = compare(e, plan.outputs, reference, inputs)
            for n, t in inputs.items():
                if not torch.equal(t, saved[n]):
                    raise CorrectnessFailure(f"input {n} modified")
    torch.cuda.current_stream().wait_stream(stream)
    return metrics


def check(e, case_index=None, *, solution=False):
    if not solution:
        require_implemented(e)
    require_environment(e["backend"])
    indices = list(range(len(e["cases"]))) if case_index is None else [case_index]
    results = []
    cases = [{"case": i, "input": e["cases"][i], "state": "NOT_RUN"} for i in indices]
    report = {
        "exercise": e["id"],
        "state": "CORRECTNESS_PASSED",
        "full_contract": case_index is None,
        "case_count": len(indices),
        "cases": cases,
        "results": results,
        "fingerprint": fingerprint(e),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "design_review": "NOT_REVIEWED",
        "implementation": "solution" if solution else "student",
    }
    failures = []
    for item in cases:
        i = item["case"]
        case = e["cases"][i]
        item["state"] = "PASS"
        for nondefault in (False, True):
            record = {"case": i, "nondefault_stream": nondefault}
            try:
                metrics = validate_case(
                    e,
                    case,
                    nondefault=nondefault,
                    **({"solution": True} if solution else {}),
                )
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


def _measure_plan(
    e, plan, inputs, saved, oracle, warmup, repeats, iterations, *, graph=True
):
    # Validate before exposing either timing, and again after each warm loop.
    plan.launch()
    torch.cuda.current_stream().synchronize()
    errors = compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    eager = measure(plan.launch, warmup, repeats, iterations)
    compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    graph_timing = (
        measure_graph(plan.launch, warmup, repeats, iterations) if graph else None
    )
    post_errors = compare(e, plan.outputs, oracle, inputs)
    _assert_unchanged(inputs, saved)
    return {
        "timing": graph_timing,
        "eager_timing": eager,
        "errors": errors,
        "post_timing_errors": post_errors,
    }


def _flops(e, case, cfg, timing):
    if e["kind"] == "attention":
        b, n, h, d = case["shape"]
        dense = 4 * b * h * n * n * d
        pairs = n * (n + 1) // 2 if cfg["causal"] else n * n
        useful = 4 * b * h * pairs * d
        seconds = timing["gpu_stream_ms_median"] / 1e3
        return {
            "convention": "QK and PV only, FMA=2; excludes softmax, padding and rescaling",
            "useful": useful,
            "dense_equivalent": dense,
            "useful_tflops": useful / seconds / 1e12,
            "dense_equivalent_tflops": dense / seconds / 1e12,
        }
    if e["kind"] == "gemm":
        m, n, k = case["shape"]
        operations = 2 * m * n * k
        seconds = timing["gpu_stream_ms_median"] / 1e3
        return {
            "convention": "GEMM: 2*M*N*K, FMA=2",
            "useful": operations,
            "useful_tflops": operations / seconds / 1e12,
        }
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
        result = _measure_plan(
            e, plan, inputs, saved, oracle, warmup, repeats, iterations
        )
    torch.cuda.current_stream().wait_stream(stream)
    result.update(
        exercise=e["id"],
        case_index=case_index,
        case=case,
        config=cfg,
        provider=plan.provider,
        scope=plan.scope,
        state="REFERENCE_VALIDATED",
        student_state="NOT_TESTED",
        fingerprint=fingerprint(e),
        environment=snapshot(),
    )
    flops = _flops(e, case, cfg, result["timing"])
    if flops:
        result["flops"] = flops
    return result


def benchmark(e, case_index, warmup, repeats, iterations, *, solution=False):
    """Compare validated CUDA/PTX kernels and the library on identical inputs.

    Student runs include the complete solution as a third candidate. Each kernel
    retains its own tuning CONFIG; only the case's mathematical options are shared.
    """
    from benchmarks.reference_backends import prepare_reference

    validation = check(e, case_index, solution=solution)
    solution_validation = None if solution else check(e, case_index, solution=True)
    case = e["cases"][case_index]
    inputs, math_cfg = make_inputs(e, case, "cuda")
    cfg = {**load_student(e, solution=solution).CONFIG, **math_cfg}
    answer_cfg = (
        {**load_student(e, solution=True).CONFIG, **math_cfg} if not solution else None
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        saved = {name: tensor.clone() for name, tensor in inputs.items()}
        # The oracle and original inputs are fixed before any candidate prepares.
        oracle = mathematical_reference(e["kind"], inputs, math_cfg)
        plan = prepare(e["id"], inputs, cfg, solution=solution)
        answer = (
            prepare(e["id"], inputs, answer_cfg, solution=True)
            if not solution
            else None
        )
        baseline = prepare_reference(e, inputs, math_cfg)
        plans = [plan, *([answer] if answer is not None else []), baseline]
        # Validate every participant before any timing is exposed, and check input
        # immutability after each so one candidate cannot alter another's workload.
        for participant in plans:
            for tensor in participant.outputs.values():
                tensor.fill_(float("nan"))
            participant.launch()
            stream.synchronize()
            compare(e, participant.outputs, oracle, inputs)
            _assert_unchanged(inputs, saved)
        records = [
            _measure_plan(
                e,
                participant,
                inputs,
                saved,
                oracle,
                warmup,
                repeats,
                iterations,
                graph=False,
            )
            for participant in plans
        ]
        graph_timings = measure_graph_group(
            [participant.launch for participant in plans], warmup, repeats, iterations
        )
        for participant, record, timing in zip(plans, records, graph_timings):
            record["timing"] = timing
            record["post_timing_errors"] = compare(
                e, participant.outputs, oracle, inputs
            )
        _assert_unchanged(inputs, saved)
    torch.cuda.current_stream().wait_stream(stream)
    measured, fast = records[0], records[-1]
    student_ms = measured["timing"]["gpu_stream_ms_median"]
    reference_ms = fast["timing"]["gpu_stream_ms_median"]
    result = {
        "exercise": e["id"],
        "case_index": case_index,
        "case": case,
        "config": cfg,
        "fingerprint": fingerprint(e),
        "validation": validation,
        "environment": snapshot(),
        "implementation": "solution" if solution else "student",
        **measured,
        "reference": {**fast, "provider": baseline.provider, "scope": baseline.scope},
        "comparison": {
            "reference_over_student": reference_ms / student_ms,
            "student_over_reference": student_ms / reference_ms,
            "metric": "median CUDA graph replay time; same inputs, output contract and captured batch size",
            "interpretation": "reference_over_student > 1 means the candidate is faster for this case; not a learning-completion gate",
        },
        "output_and_workspace_bytes": sum(
            t.numel() * t.element_size()
            for t in list(plan.outputs.values()) + list(plan.workspace.values())
        )
        + plan.extra_workspace_bytes,
    }
    if answer is not None:
        answer_record = records[1]
        answer_ms = answer_record["timing"]["gpu_stream_ms_median"]
        result["solution"] = {
            **answer_record,
            "provider": "cuda_ptx_solution",
            "config": answer_cfg,
            "validation": solution_validation,
            "output_and_workspace_bytes": sum(
                t.numel() * t.element_size()
                for t in list(answer.outputs.values()) + list(answer.workspace.values())
            )
            + answer.extra_workspace_bytes,
        }
        result["comparison"].update(
            solution_over_student=answer_ms / student_ms,
            student_over_solution=student_ms / answer_ms,
            solution_over_reference=answer_ms / reference_ms,
        )
    flops = _flops(e, case, cfg, result["timing"])
    if flops:
        result["flops"] = flops
    from .performance import assess_solution_performance

    result["solution_performance"] = assess_solution_performance(e, result)
    return result
