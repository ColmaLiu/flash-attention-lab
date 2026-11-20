"""Run a kernel on one case: correctness checks and benchmarks."""

import json
import time
import zlib

import torch

from . import ROOT, CorrectnessError, EnvironmentError_, LabError, NotImplementedYet
from . import build as build_mod
from . import kinds, timing

# Nominal B200 datasheet peaks (dense), used only to put numbers in context.
PEAK = {
    "hbm_gbps": 8000.0,
    "tflops": {"fp32": 75.0, "fp16": 2250.0, "bf16": 2250.0, "e4m3": 4500.0},
}


def make_case(lab, case):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(zlib.crc32(f"{lab.id}/{case.name}".encode()))
    kind = kinds.module(lab.kind)
    inputs = kind.make_inputs(case, gen)
    return inputs, kind.options(case, inputs)


def _poison(t):
    if t.is_floating_point() and t.dtype != torch.float8_e4m3fn:
        t.fill_(float("nan"))
    else:
        t.view(torch.uint8).fill_(0xFF)


class Prepared:
    """A kernel bound to one set of tensors via make_plan."""

    def __init__(self, lab, case, inputs, options, step, solution):
        import tvm_ffi

        # step None means "the finished kernel", i.e. the last step.
        step = step or len(lab.steps)
        self.lab, self.case, self.step = lab, case, step
        self.lib = build_mod.build(lab, solution)
        self.kind = kinds.module(lab.kind)
        self.outputs = self.kind.make_outputs(inputs, case)
        tensors = {**inputs, **self.outputs}
        self.probe_expected = None
        probe = lab.step_info(step).probe
        if probe:
            expected, self.probe_tol = getattr(self.kind, "probe_" + probe["oracle"])(
                inputs, options, case
            )
            self.probe_expected = expected
            self.outputs["probe"] = torch.empty(
                expected.shape, device="cuda", dtype=torch.float32
            )
            tensors["probe"] = self.outputs["probe"]
        for t in self.outputs.values():
            _poison(t)
        converted = {k: tvm_ffi.from_dlpack(v) for k, v in tensors.items()}
        opts = {k: float(v) for k, v in {**options, "step": step}.items()}
        try:
            self.plan = self.lib.prepare(converted, opts)
        except Exception as exc:
            raise _translate(exc, self.lib, "make_plan") from None

    def launch(self, stream=None):
        stream = stream or torch.cuda.current_stream()
        self.lib.run(self.plan, stream.cuda_stream)

    def run_checked(self, stream):
        try:
            self.launch(stream)
            stream.synchronize()
        except Exception as exc:
            raise _translate(exc, self.lib, "run") from None


def _translate(exc, lib, where):
    text = str(exc)
    if "LAB_TODO(" in text:
        n = text.split("LAB_TODO(", 1)[1].split(")", 1)[0]
        return NotImplementedYet(f"TODO({n}) is not implemented yet ({where}).")
    try:
        todo = lib.todo_value()
    except Exception:
        todo = 0
    if todo:
        return NotImplementedYet(
            f"TODO({todo}) is not implemented yet (reached on the GPU)."
        )
    if "binding:" in text or "requirement failed" in text:
        return LabError(text.split("\n")[0])
    return LabError(
        f"{where} failed: {text.strip().splitlines()[-1] if text.strip() else exc!r}\n"
        "  A CUDA error leaves the GPU context unusable; try\n"
        "  ./lab sanitize <lab> --case <case>  to locate the bad access."
    )


def check_case(lab, case, step, solution=False, full=True):
    """Run once on a non-default stream and once on the default stream."""
    inputs, options = make_case(lab, case)
    saved = {k: v.clone() for k, v in inputs.items()}
    p = Prepared(lab, case, inputs, options, step, solution)
    metrics = None
    for stream in (torch.cuda.Stream(), torch.cuda.default_stream()):
        for t in p.outputs.values():
            _poison(t)
        p.run_checked(stream)
        if p.probe_expected is not None:
            from .kinds import compare

            metrics = {
                "probe": compare(
                    "probe", p.outputs["probe"], p.probe_expected, *p.probe_tol
                )
            }
        else:
            metrics = p.kind.verify(inputs, p.outputs, options, case, full)
        for k, v in inputs.items():
            if not torch.equal(v.view(torch.uint8), saved[k].view(torch.uint8)):
                raise CorrectnessError(
                    f"input '{k}' was modified; inputs are read-only"
                )
    return metrics


def bench(lab, case, step, solution=False, mode="graph", compare_solution=True):
    """Time the kernel, the reference solution and the library baselines."""
    inputs, options = make_case(lab, case)
    kind = kinds.module(lab.kind)
    who = "solution" if solution else "student"
    main = Prepared(lab, case, inputs, options, step, solution)
    main.run_checked(torch.cuda.current_stream())
    kind.verify(inputs, main.outputs, options, case, full=False)
    candidates = [(who, main.launch)]
    notes = {}
    if compare_solution and not solution:
        try:
            ref = Prepared(lab, case, inputs, options, None, True)
            ref.run_checked(torch.cuda.current_stream())
            candidates.append(("solution", ref.launch))
        except LabError as exc:
            notes["solution"] = str(exc).splitlines()[0]
    for name, factory in kind.baselines().items():
        try:
            launch, _ = factory(inputs, options, case)
            launch()
            torch.cuda.synchronize()
            candidates.append((name, launch))
        except (EnvironmentError_, RuntimeError, NotImplementedError) as exc:
            notes[name] = str(exc).splitlines()[0][:120]
    launches = [c[1] for c in candidates]
    results = timing.graph(launches) if mode == "graph" else timing.cold(launches)
    work = kind.work(case)
    rows = []
    for (name, _), r in zip(candidates, results):
        ms = r["median_ms"]
        row = {"name": name, **r}
        if work["flops"]:
            row["tflops"] = work["flops"] / ms / 1e9
            row["peak_fraction"] = row["tflops"] / PEAK["tflops"][case.dtype]
        else:
            row["gbps"] = work["bytes"] / ms / 1e6
            row["peak_fraction"] = row["gbps"] / PEAK["hbm_gbps"]
        rows.append(row)
    report = {
        "lab": lab.id,
        "case": case.name,
        "step": step,
        "mode": mode,
        "work": work,
        "results": rows,
        "unavailable": notes,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(),
        "fingerprint": build_mod.fingerprint(lab, solution),
    }
    out = ROOT / "artifacts" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}_{who}_{lab.id}_{case.name}.json").write_text(
        json.dumps(report, indent=1)
    )
    return report
