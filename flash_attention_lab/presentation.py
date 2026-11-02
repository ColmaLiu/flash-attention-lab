"""Concise English terminal views; JSON reports retain full diagnostics."""
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata

from .registry import ROOT

STAGES = {0: "Foundations", 1: "GEMM and streaming attention", 2: "Tensor Core and FlashAttention-2",
          3: "Blackwell asynchronous execution", 4: "Blackwell attention optimization"}
STATES = {
    "STUDENT_UNIMPLEMENTED": "NOT IMPLEMENTED", "IMPLEMENTED_UNVERIFIED": "UNVERIFIED",
    "CORRECTNESS_PASSED": "PASS", "CORRECTNESS_FAILED": "FAIL",
    "ENV_UNSUPPORTED": "UNSUPPORTED", "EXECUTION_ERROR": "ERROR",
    "TIMEOUT": "TIMEOUT", "INTERRUPTED": "INTERRUPTED", "TOOL_FAILED": "TOOL FAILED",
    "NOT_RUN": "NOT RUN",
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain(value):
    return "".join(c for c in ANSI.sub("", str(value)) if c in "\n\t" or unicodedata.category(c) != "Cc")


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 0 if unicodedata.combining(c) else 1
               for c in plain(text))


def wrap(text, limit):
    lines = []
    for line in plain(text).expandtabs(4).splitlines() or [""]:
        part, used = "", 0
        for char in line:
            size = width(char)
            if used + size > limit and part:
                lines.append(part); part, used = "", 0
            part += char; used += size
        lines.append(part)
    return lines


def path_label(value):
    try: return str(Path(value).relative_to(ROOT))
    except ValueError: return str(value)


def number(value):
    return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def milliseconds(value):
    return f"{value * 1000:.3f} us" if value < 1 else f"{value:.3f} ms"


def clipped(value, limit=54):
    lines = wrap(value, limit)
    return lines[0] + ("..." if len(lines) > 1 else "")


class Terminal:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.columns = max(32, min(120, shutil.get_terminal_size((100, 24)).columns))
        self.color = self.stream.isatty() and "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb"

    def print(self, text=""):
        print(text, file=self.stream)

    def heading(self, text):
        for line in wrap(text, self.columns):
            self.print(("\033[1;36m" + line + "\033[0m") if self.color else line)

    def line(self, label, value):
        prefix = "  " + label + "  "
        for i, part in enumerate(wrap(value, max(8, self.columns - width(prefix)))):
            self.print((prefix if i == 0 else " " * width(prefix)) + part)

    def note(self, text):
        for part in wrap(text, self.columns - 2): self.print("  " + part)

    def table(self, headers, rows):
        rows = [[plain(c) for c in row] for row in rows]
        sizes = [max([width(h), *(width(c[i]) for c in rows)]) for i, h in enumerate(headers)]
        budget = self.columns - 2 - 3 * (len(headers) - 1)
        minimum = [max(width(h), min(s, 12)) for h, s in zip(headers, sizes)]
        if sum(minimum) > budget:
            for row in rows:
                for label, value in zip(headers, row): self.line(label, value)
                self.print()
            return
        while sum(sizes) > budget:
            i = max(range(len(sizes)), key=lambda j: sizes[j] - minimum[j])
            sizes[i] -= 1
        def emit(row):
            cells = [wrap(value, size) for value, size in zip(row, sizes)]
            for i in range(max(map(len, cells))):
                parts = [cell[i] if i < len(cell) else "" for cell in cells]
                self.print("  " + "   ".join(s + " " * (size - width(s)) for s, size in zip(parts, sizes)).rstrip())
        emit(headers)
        self.print("  " + "   ".join("-" * size for size in sizes))
        for row in rows: emit(row)


def compact_version(command, pattern=None):
    text = command.get("stdout", "")
    if pattern:
        match = re.search(pattern, text)
        return match.group(1) if match else "unavailable"
    return text.splitlines()[0] if text else "unavailable"


def doctor(ui, data, verbose):
    ui.heading("Environment: " + ("READY" if data.get("state") == "ENVIRONMENT_OK" else "UNSUPPORTED"))
    gpu = data.get("selected_gpu", {})
    cap = ".".join(map(str, gpu.get("compute_capability", [])))
    ui.line("GPU", gpu.get("name", "unavailable") + (f" | CC {cap} | {gpu['total_memory'] / 2**30:.1f} GiB" if cap else ""))
    driver = data.get("driver_cuda_ceiling", {})
    ui.line("Driver", compact_version(driver, r"Driver Version:\s*([\d.]+)") + " | CUDA support <= " + compact_version(driver, r"CUDA Version:\s*([\d.]+)"))
    packages = data.get("packages", {})
    ui.line("CUDA", "Toolkit " + compact_version(data.get("nvcc", {}), r"release ([\d.]+)") + " | PyTorch runtime " + str(data.get("torch_cuda_runtime") or "unavailable"))
    ui.line("Packages", " | ".join(f"{label} {packages.get(name) or 'missing'}"
                                   for label, name in (("PyTorch", "torch"), ("Ninja", "ninja"), ("pytest", "pytest"))))
    ui.line("Python", data.get("python", "unknown").split()[0] + " | Conda " + str(data.get("conda_environment") or "inactive"))
    for backend, state in data.get("checks", {}).items():
        ui.line("CUDA C++" if backend == "cuda" else backend, "READY" if state == "ENVIRONMENT_OK" else state)
    tools = data.get("tools", {})
    ui.line("Tools", " | ".join(name + (": available" if tools.get(key) else ": missing") for name, key in [("Sanitizer", "compute-sanitizer"), ("NCU", "ncu")]))
    if verbose:
        for label, key in [("OS", "os"), ("Python path", "executable"), ("Toolkit path", "cuda_home"), ("Visible devices", "visible_devices")]:
            ui.line(label, data.get(key) or "unavailable")
        ui.line("C++", compact_version(data.get("cxx", {})))
        for name, value in tools.items(): ui.line(name, value or "missing")
        for key in ("nvcc", "cxx", "nvidia_smi", "driver_cuda_ceiling"):
            diagnostic = data.get(key, {})
            if diagnostic.get("code", 0) != 0: ui.line(key, diagnostic.get("error") or diagnostic.get("stderr") or "failed")
        if data.get("torch_error"): ui.line("PyTorch error", data["torch_error"])


def inventory(ui, data, verbose):
    status = data["command"] == "status"
    ui.heading("Exercise status" if status else "Exercises")
    for stage in sorted({e["stage"] for e in data["exercises"]}):
        ui.print(); ui.print(f"Stage {stage}: {STAGES[stage]}")
        exercises = [e for e in data["exercises"] if e["stage"] == stage]
        if status:
            rows = [[e["id"], STATES.get(e["state"], e["state"]),
                     e["reason"] if verbose else clipped(e["reason"])] for e in exercises]
            ui.table(["Exercise", "Status", "Reason"], rows)
        else:
            ui.table(["Exercise", "Task", "Cases"], [[e["id"], e["title"], e["case_count"]] for e in exercises])
        if verbose:
            for e in exercises: ui.line(e["id"], e["backend"] + " | prerequisites: " + (", ".join(e["prerequisites"]) or "none"))
    if status:
        ui.print()
        counts = Counter(e["state"] for e in data["exercises"])
        ui.note(" | ".join(f"{n} {STATES.get(s, s).lower()}" for s, n in counts.items()))
        for backend, state in data["environment"].items():
            if not state.startswith("ENVIRONMENT_OK"): ui.line(backend, state)


def case_label(case):
    parts = ["x".join(map(str, case.get("shape", []))), case.get("dtype", ""), case.get("pattern", "")]
    if len(case.get("shape", [])) == 4: parts.append("causal" if case.get("causal") else "noncausal")
    if "scale" in case: parts.append("scale=" + number(case["scale"]))
    return "  ".join(p for p in parts if p)


def task(ui, data, verbose):
    ui.heading(data["exercise"] + ": " + data["title"])
    if verbose: ui.note(data["purpose"])
    ui.line("Stage", f"{data['stage']} | {data['backend']} | {data['case_count']} cases")
    ui.line("Prerequisites", ", ".join(data["prerequisites"]) or "none")
    ui.print(); ui.table(["Argument", "Key", "Shape", "Dtype"], data["tensors"])
    ui.print()
    for path in data["student_files"]: ui.line("Edit", path)
    ui.line("Task", data["task_file"])
    ui.line("Check", f"./lab check {data['exercise']} --case 0")
    ui.line("Answer", f"./lab solution {data['exercise']} --case 0")
    ui.line("Library", f"./lab reference {data['exercise']}")
    if verbose:
        ui.line("Notes template", "experiments/README.md")
        ui.print(); ui.table(["Case", "ID / suite", "Input", "Purpose"], [[i, c["id"] + " / " + c["suite"], case_label(c), c["purpose"]] for i, c in enumerate(data["cases"])])


def check(ui, data, verbose):
    scope = "all cases" if data["full_contract"] else "selected case only"
    ui.heading(f"{data['exercise']}: check ({scope})")
    rows = []
    for item in data["cases"]:
        errors = [m["max_abs"] for r in data["results"] if r["case"] == item["case"] for m in r.get("errors", {}).values()]
        rows.append([item["case"], item["state"].replace("_", " "), case_label(item["input"]),
                     number(max(errors)) if errors and item["state"] == "PASS" else "-"])
    ui.table(["Case", "Result", "Input", "Max abs"], rows)
    for record in data["results"]:
        stream = "nondefault" if record["nondefault_stream"] else "default"
        if record.get("message"):
            ui.line(f"case {record['case']} / {stream}:", record["message"])
        elif verbose:
            for name, metrics in record["errors"].items():
                ui.line(f"case {record['case']} / {stream} / {name}:",
                        " | ".join(f"{k}={number(metrics[k])}" for k in ("max_abs", "max_rel_floored", "relative_floor", "atol", "rtol")))
    counts = Counter(c["state"] for c in data["cases"])
    ui.note(", ".join(f"{counts[state]} {label}" for state, label in
                      [("PASS", "passed"), ("FAIL", "failed"), ("ERROR", "errors"), ("NOT_RUN", "not run")]
                      if counts[state] or state in ("PASS", "FAIL")))
    if verbose: ui.note("Streams: default + nondefault; 2 launches each. Design review: not assessed.")


def stage_check(ui, data, verbose):
    ui.heading(f"Stage {data['stage']}: check summary")
    rows = []
    for report in data["exercises"]:
        passed = sum(c["state"] == "PASS" for c in report.get("cases", []))
        rows.append([report["exercise"], STATES.get(report["state"], report["state"]),
                     f"{passed}/{report['case_count']}"])
    ui.table(["Exercise", "Result", "Cases passed"], rows)
    passed = sum(r["state"] == "CORRECTNESS_PASSED" for r in data["exercises"])
    ui.note(f"{passed}/{len(rows)} exercises passed.")


def bench(ui, data, verbose):
    implementation = data.get("implementation", "library reference")
    ui.heading(f"{data['exercise']}: {implementation} | case {data['case_index']} | validated")
    ui.note(case_label(data["case"]))
    ui.line("GPU", data["environment"].get("selected_gpu", {}).get("name", "unknown"))
    rows = []
    records = [(implementation, data)]
    if "solution" in data:
        records.append(("CUDA/PTX answer", data["solution"]))
    if "reference" in data:
        records.append(("optimized library", data["reference"]))
    for label, record in records:
        timing = record["timing"]
        rows.append([label, *[milliseconds(timing["gpu_stream_ms_" + key]) for key in ("median", "min", "max")]])
    ui.table(["CUDA graph / operation", "Median", "Min", "Max"], rows)
    if "reference" in data:
        ui.line("Backend", data["reference"]["provider"])
        ui.line("Time / library", f"{data['comparison']['student_over_reference']:.3f}x (lower is faster)")
        if "solution" in data:
            ui.line("Time / answer", f"{data['comparison']['student_over_solution']:.3f}x (lower is faster)")
    else:
        ui.line("Backend", data.get("provider", "unknown"))
        ui.note("Student implementation: NOT TESTED")
    eager_rows = [[label, milliseconds(record["eager_timing"]["gpu_stream_ms_median"])]
                  for label, record in records if "eager_timing" in record]
    if eager_rows:
        ui.table(["Eager API GPU span", "Median"], eager_rows)
    if "flops" in data:
        ui.line("Useful TFLOP/s", number(data["flops"]["useful_tflops"]))
    ui.note("Graph replay excludes compilation, capture and allocation; includes output layout copies.")
    ui.note("Eager spans include host dispatch gaps and any allocation inside the API.")
    if verbose:
        ui.line("Timing scope", data["timing"]["scope"])
        if "scope" in data: ui.line("Reference scope", data["scope"])
        if "reference" in data: ui.line("Reference scope", data["reference"]["scope"])
        ui.line("Config", ", ".join(f"{k}={v}" for k, v in data["config"].items()))


def smoke(ui, data, verbose):
    ui.heading("Infrastructure: PASS")
    ui.line("Imports", f"{data['modules_imported']} modules | {data['input_reference_cases']} input/reference cases")
    if "cuda_stub_bindings_compiled" in data: ui.line("CUDA bindings", f"{len(data['cuda_stub_bindings_compiled'])} compiled")
    if "cuda_sentinel" in data:
        ui.line("CUDA sentinel", "PASS" + (" | " + milliseconds(data["cuda_sentinel"]["gpu_stream_ms_median"]) if verbose else ""))
    ui.line("Student kernels", "NOT TESTED")
    if verbose: ui.line("Stub bindings", ", ".join(data.get("cuda_stub_bindings_compiled", [])) or "none")


def render(data, *, verbose=False, stream=None):
    ui = Terminal(stream)
    command = data.get("command")
    failed = bool(data.get("exit_code", 0))
    if command == "check" and "exercises" in data:
        stage_check(ui, data, verbose)
    elif failed and data.get("cases"):
        check(ui, data, verbose)
        if data.get("hint"): ui.note(data["hint"])
    elif data.get("message") and failed:
        ui.heading(f"{data.get('exercise', command or 'lab')}: {data['state']}")
        ui.note(data["message"])
        if data.get("hint"): ui.note(data["hint"])
    elif command == "doctor": doctor(ui, data, verbose)
    elif command in ("list", "status"): inventory(ui, data, verbose)
    elif command == "task": task(ui, data, verbose)
    elif command == "check": check(ui, data, verbose)
    elif command in ("bench", "solution", "reference"): bench(ui, data, verbose)
    elif command == "smoke": smoke(ui, data, verbose)
    elif command in ("sanitize", "profile"):
        ui.heading(f"{data['exercise']}: {data.get('tool', 'ncu')} {'PASS' if command == 'sanitize' else 'collected'} | case {data.get('case_index', 0)}")
        if data.get("profile_report"): ui.line("Report", path_label(data["profile_report"]))
    if verbose or failed:
        for label, key in (("Report", "artifact"), ("Log", "log")):
            if data.get(key): ui.line(label, path_label(data[key]))
    if verbose and data.get("fingerprint"): ui.line("Fingerprint", data["fingerprint"])
