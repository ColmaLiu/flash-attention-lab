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
from .timing import measure
from .reporting import save_artifact  # retained for callers of the original entry point


def validate_case(e, case, *, nondefault=True):
    inputs, cfg = make_inputs(e, case, "cuda")
    cfg = {**load_student(e).CONFIG, **cfg}
    saved = {n: t.clone() for n, t in inputs.items()}
    reference = mathematical_reference(e["kind"], inputs, cfg)
    stream = torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan = prepare(e["id"], inputs, cfg)
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


def check(e, case_index=None):
    require_implemented(e)
    require_environment(e["backend"])
    indices = list(range(len(e["cases"]))) if case_index is None else [case_index]
    results = []
    cases = [{"case": i, "input": e["cases"][i], "state": "NOT_RUN"} for i in indices]
    report = {"exercise": e["id"], "state": "CORRECTNESS_PASSED", "full_contract": case_index is None,
              "case_count": len(indices), "cases": cases, "results": results, "fingerprint": fingerprint(e),
              "timestamp": datetime.now(timezone.utc).isoformat(), "design_review": "NOT_REVIEWED"}
    failures = []
    for item in cases:
        i = item["case"]
        case = e["cases"][i]
        item["state"] = "PASS"
        for nondefault in (False, True):
            record = {"case": i, "nondefault_stream": nondefault}
            try:
                metrics = validate_case(e, case, nondefault=nondefault)
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


def benchmark(e, case_index, warmup, repeats, iterations):
    validation = check(e, case_index)
    case = e["cases"][case_index]
    inputs, cfg = make_inputs(e, case, "cuda")
    cfg = {**load_student(e).CONFIG, **cfg}
    plan = prepare(e["id"], inputs, cfg)  # compilation/allocation excluded
    result = {"exercise": e["id"], "case_index": case_index, "case": case, "config": cfg,
              "fingerprint": fingerprint(e), "validation": validation,
              "environment": snapshot(), "timing": measure(plan.launch, warmup, repeats, iterations),
              "output_and_workspace_bytes": sum(t.numel() * t.element_size() for t in list(plan.outputs.values()) + list(plan.workspace.values()))}
    # Detect corruption that emerges only after repeated warm runs before showing timing.
    result["post_timing_errors"] = compare(e, plan.outputs, mathematical_reference(e["kind"], inputs, cfg), inputs)
    if e["kind"] == "attention":
        b, h, n, d = case["shape"]
        dense = 4 * b * h * n * n * d
        pairs = n * (n + 1) // 2 if cfg["causal"] else n * n
        useful = 4 * b * h * pairs * d
        seconds = result["timing"]["gpu_stream_ms_median"] / 1e3
        result["flops"] = {"convention": "QK and PV only, FMA=2; excludes softmax, padding and rescaling", "useful": useful, "dense_equivalent": dense,
                           "useful_tflops": useful / seconds / 1e12, "dense_equivalent_tflops": dense / seconds / 1e12}
    elif e["kind"] == "gemm":
        m, n, k = case["shape"]
        operations = 2 * m * n * k
        seconds = result["timing"]["gpu_stream_ms_median"] / 1e3
        result["flops"] = {"convention": "GEMM: 2*M*N*K, FMA=2", "useful": operations,
                           "useful_tflops": operations / seconds / 1e12}
    return result
