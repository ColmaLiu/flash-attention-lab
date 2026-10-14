import hashlib
import json
import time
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
    indices = range(len(e["cases"])) if case_index is None else [case_index]
    results = []
    for i in indices:
        case = e["cases"][i]
        for nondefault in (False, True):
            try: metrics = validate_case(e, case, nondefault=nondefault)
            except Exception as exc:
                raise type(exc)(f"{e['id']} case={i} nondefault_stream={nondefault} {case}: {exc}") from exc
            results.append({"case": i, "nondefault_stream": nondefault, "errors": metrics})
    return {"exercise": e["id"], "state": "CORRECTNESS_PASSED", "full_contract": case_index is None,
            "case_count": len(list(indices)), "results": results, "fingerprint": fingerprint(e),
            "timestamp": datetime.now(timezone.utc).isoformat(), "design_review": "NOT_REVIEWED"}


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
    return result


def save_artifact(prefix, result):
    directory = ROOT / "artifacts" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (prefix + "_" + str(time.time_ns()) + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
