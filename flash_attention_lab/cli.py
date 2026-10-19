import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from .errors import CorrectnessFailure, EnvironmentUnsupported, StudentNotImplemented
from .registry import EXERCISES, ROOT, fingerprint, load_student, selected


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def positive_int(value):
    n = int(value)
    if n < 1: raise argparse.ArgumentTypeError("must be positive")
    return n


def exercise_id(value):
    """Resolve a complete ID or its exact stage/lesson code before dispatch."""
    if value in EXERCISES:
        return value
    matches = [name for name in EXERCISES if name.split("_", 1)[0] == value]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise argparse.ArgumentTypeError(f"ambiguous exercise {value!r}; use one of: {', '.join(matches)}")
    raise argparse.ArgumentTypeError(f"unknown exercise {value!r}; see './lab list'")


def parser():
    display = argparse.ArgumentParser(add_help=False)
    display.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="full JSON report")
    display.add_argument("--verbose", "-v", action="store_true", default=argparse.SUPPRESS, help="show diagnostics, details and report paths")
    p = Parser(prog="./lab", description="FlashAttention Lab: GPU kernel exercises", parents=[display])
    public = "doctor,list,task,status,check,bench,smoke,sanitize,profile,python"
    sub = p.add_subparsers(dest="command", required=True, metavar="{" + public + "}")
    descriptions = {"doctor": "check environment", "smoke": "check infrastructure", "status": "exercise status", "list": "list exercises", "task": "show task and tensor contract", "check": "validate student kernel", "bench": "validate, then benchmark", "sanitize": "run Compute Sanitizer", "profile": "run Nsight Compute", "python": "run Python in the project environment"}
    def add(command):
        kwargs = {"help": descriptions[command]} if command in descriptions else {}
        return sub.add_parser(command, parents=[display], **kwargs)
    for command in ("doctor", "smoke", "status", "list"):
        q = add(command)
        q.add_argument("--stage", type=int, choices=range(5), help="select a stage")
        if command == "smoke": q.add_argument("--timeout", type=float, default=600)
    q = add("task"); q.add_argument("exercise", type=exercise_id, metavar="ID", help="full or short exercise ID (e.g. s20)")
    for command in ("check", "bench", "sanitize", "profile", "_worker", "_profile"):
        q = add(command)
        if command == "check":
            scope = q.add_mutually_exclusive_group(required=True)
            scope.add_argument("exercise", nargs="?", type=exercise_id, metavar="ID", help="full or short exercise ID (e.g. s20)")
            scope.add_argument("--stage", type=int, choices=range(5), help="check every exercise in a stage")
        else:
            q.add_argument("exercise", type=exercise_id, metavar="ID", help="full or short exercise ID (e.g. s20)")
        q.add_argument("--case", type=int)
        q.add_argument("--timeout", type=float, default=180,
                       help="timeout in seconds (per exercise for --stage)" if command == "check" else "timeout in seconds")
        if command in ("bench", "_worker"):
            q.add_argument("--warmup", type=positive_int, default=5)
            q.add_argument("--repeats", type=positive_int, default=7)
            q.add_argument("--iterations", type=positive_int, default=10)
        if command == "sanitize": q.add_argument("--tool", choices=("memcheck", "racecheck", "synccheck", "initcheck"), default="memcheck")
        if command == "_worker": q.add_argument("--operation", choices=("check", "bench"), required=True)
        if command.startswith("_"): q.add_argument("--report-file", help=argparse.SUPPRESS)
    q = add("_smoke"); q.add_argument("--stage", type=int, choices=range(5)); q.add_argument("--report-file", help=argparse.SUPPRESS)
    q = add("python"); q.add_argument("args", nargs=argparse.REMAINDER)
    return p


def bounded(command, timeout, *, output=None):
    if timeout <= 0: raise ValueError("timeout must be positive")
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdout=output,
                             stderr=subprocess.STDOUT if output is not None else None)
    try: return child.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        if isinstance(exc, KeyboardInterrupt): raise
        return 5


def emit(result, args=None):
    if args is not None and getattr(args, "report_file", None):
        Path(args.report_file).write_text(json.dumps(result, ensure_ascii=False) + "\n")
    elif args is not None and getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from .presentation import render
        render(result, verbose=getattr(args, "verbose", False),
               stream=sys.stderr if result.get("message") and result.get("exit_code") else sys.stdout)


def worker_command(args, operation):
    cmd = [sys.executable, "-m", "flash_attention_lab", "_worker", args.exercise, "--operation", operation]
    if args.case is not None: cmd += ["--case", str(args.case)]
    if operation == "bench":
        cmd += ["--warmup", str(args.warmup), "--repeats", str(args.repeats), "--iterations", str(args.iterations)]
    return cmd


def managed(command, args, *, tool=False):
    """Separate structured worker results from compiler/tool stdout and stderr."""
    name = args.command
    directory = ROOT / "artifacts/logs"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (name + "_" + str(time.time_ns()) + ".log")
    if not getattr(args, "json", False) and (sys.stderr.isatty() or getattr(args, "verbose", False)):
        from .presentation import Terminal, path_label
        ui = Terminal(sys.stderr)
        ui.note(f"Running {name}" + (f" {args.exercise}" if hasattr(args, "exercise") else "") + "...")
        if getattr(args, "verbose", False): ui.line("Log", path_label(log))
    with tempfile.TemporaryDirectory(prefix="falab-report-") as folder:
        report = Path(folder) / "result.json"
        with log.open("w") as output:
            code = bounded([*command, "--report-file", str(report)], args.timeout, output=output)
        try:
            result = json.loads(report.read_text()) if report.exists() else {}
            if not isinstance(result, dict): raise ValueError("report must be an object")
        except (ValueError, OSError) as exc:
            code = 6
            result = {"state": "EXECUTION_ERROR", "exit_code": 6, "message": f"Cannot read worker report: {exc}"}
    if code == 0 and result.get("exit_code", 0) != 0:
        code = 6
        result = {"state": "EXECUTION_ERROR", "exit_code": 6,
                  "message": "Worker reported failure but exited with code 0.", "worker_report": result}
    if code < 0: code = 128 - code
    if code != 0 and result.get("exit_code", 0) == 0:
        state = "TIMEOUT" if code == 5 else "TOOL_FAILED" if tool else "EXECUTION_ERROR"
        message = f"Timed out after {args.timeout:g}s; stopped this command's processes." if code == 5 else "Worker or tool exited unsuccessfully."
        if code != 5:
            with log.open("rb") as handle:
                handle.seek(0, 2); handle.seek(max(0, handle.tell() - 6000))
                tail = handle.read().decode(errors="replace").strip()
            if tail: message += "\n" + tail
        result = {"state": state, "message": message}
    if code == 0 and not result:
        code = 6
        result = {"state": "EXECUTION_ERROR", "message": "Worker exited without a validation report."}
    result.update(command=name, exit_code=code, log=str(log))
    if hasattr(args, "exercise"): result["exercise"] = args.exercise
    if name == "sanitize" and code == 0:
        result = {"command": name, "exit_code": 0, "state": "SANITIZER_PASSED", "exercise": args.exercise,
                  "case_index": 0 if args.case is None else args.case, "tool": args.tool,
                  "validation": result, "log": str(log)}
    from .reporting import save_artifact
    result["artifact"] = str(save_artifact(name, result))
    return result


def check_stage(args):
    if args.case is not None:
        raise ValueError("--case cannot be combined with --stage; use check ID --case N")
    if not 0 < args.timeout < float("inf"):
        raise ValueError("timeout must be positive and finite")
    exercises = selected(args.stage)
    reports = [{"exercise": e["id"], "state": "NOT_RUN", "exit_code": None,
                "case_count": len(e["cases"])} for e in exercises]
    interrupted = False
    for i, e in enumerate(exercises):
        child_args = argparse.Namespace(**{**vars(args), "exercise": e["id"], "stage": None})
        try:
            report = managed(worker_command(child_args, "check"), child_args)
        except KeyboardInterrupt as exc:
            report = error_report(exc, child_args)
        except Exception as exc:
            report = error_report(exc, child_args)
        report.setdefault("case_count", len(e["cases"]))
        reports[i] = report
        if not getattr(args, "json", False):
            from .presentation import render
            render(report, verbose=getattr(args, "verbose", False))
            print(flush=True)
        if report["exit_code"] == 130:
            interrupted = True
            break
    code = 130 if interrupted else next((r["exit_code"] for r in reports if r["exit_code"]), 0)
    result = {"command": "check", "stage": args.stage, "exercises": reports, "exit_code": code,
              "state": "INTERRUPTED" if interrupted else "STAGE_FAILED" if code else "STAGE_PASSED"}
    from .reporting import save_artifact
    result["artifact"] = str(save_artifact("check_stage", result))
    return result


def brief(e):
    return {key: e[key] for key in ("id", "title", "stage", "backend", "prerequisites")} | {"case_count": len(e["cases"])}


def status_report(stage):
    from .environment import require_environment
    env = {}
    for backend in sorted({e["backend"] for e in selected(stage)}):
        try: require_environment(backend); env[backend] = "ENVIRONMENT_OK"
        except EnvironmentUnsupported as exc: env[backend] = f"ENV_UNSUPPORTED: {exc}"
    records, digest = [], None
    for e in selected(stage):
        module = load_student(e)
        state, reason = "STUDENT_UNIMPLEMENTED", "-"
        if module.IMPLEMENTED:
            missing = [d for d in module.RUNTIME_REQUIRES if d not in EXERCISES or not load_student(EXERCISES[d]).IMPLEMENTED]
            if missing: reason = "Missing modules: " + ", ".join(missing)
            elif env[e["backend"]].startswith("ENV_UNSUPPORTED"):
                state, reason = "ENV_UNSUPPORTED", "See environment diagnostics"
            else:
                state, reason = "IMPLEMENTED_UNVERIFIED", "No full check recorded"
                path = ROOT / "progress/results" / (e["id"] + ".json")
                if path.exists():
                    data = json.loads(path.read_text())
                    if digest is None: digest = fingerprint(e)
                    if data.get("fingerprint") == digest and data.get("full_contract"):
                        state, reason = data["state"], data.get("reason", "Current full check; design review separate")
                    else: reason = "Stale or partial check"
        records.append({**brief(e), "state": state, "reason": reason})
    return {"command": "status", "state": "INVENTORY", "environment": env, "exercises": records,
            "note": "inventory only; no correctness test implied", "exit_code": 0}


def error_report(exc, args):
    if isinstance(exc, StudentNotImplemented) or "STUDENT_TODO" in str(exc): state, code = "STUDENT_UNIMPLEMENTED", 2
    elif isinstance(exc, (EnvironmentUnsupported, ImportError)): state, code = "ENV_UNSUPPORTED", 4
    elif isinstance(exc, (CorrectnessFailure, AssertionError)): state, code = "CORRECTNESS_FAILED", 3
    elif isinstance(exc, KeyboardInterrupt): state, code = "INTERRUPTED", 130
    else: state, code = "EXECUTION_ERROR", 6
    command = getattr(args, "command", "lab")
    if command == "_worker": command = args.operation
    elif command == "_smoke": command = "smoke"
    elif command == "_profile": command = "profile"
    result = {**getattr(exc, "check_report", {}), "command": command, "state": state, "exit_code": code, "message": str(exc) or state}
    exercise = getattr(args, "exercise", None)
    if exercise:
        result["exercise"] = exercise
        case = re.search(r"\bcase=(\d+)", str(exc))
        case_index = int(case.group(1)) if case else getattr(args, "case", None)
        if case_index is None:
            case_index = next((c["case"] for c in result.get("cases", []) if c["state"] in ("FAIL", "ERROR")), None)
        if case_index is not None: result["case_index"] = case_index
        if state == "STUDENT_UNIMPLEMENTED": result["hint"] = f"Task: ./lab task {exercise}"
        elif state == "CORRECTNESS_FAILED" or case_index is not None and state == "EXECUTION_ERROR":
            result["hint"] = f"Reproduce: ./lab check {exercise} --case {case_index if case_index is not None else 0}"
    if state == "ENV_UNSUPPORTED": result["hint"] = "Diagnose: ./lab doctor --verbose"
    # Preserve failure bookkeeping: a counterexample invalidates an older success,
    # including failures found by a single-case check or benchmark validation.
    if getattr(args, "command", None) == "_worker" and code in (3, 6):
        from .runner import save_result
        e = EXERCISES[exercise]
        save_result(e, {"state": "CORRECTNESS_FAILED", "fingerprint": fingerprint(e), "full_contract": True, "reason": str(exc)})
    return result


def execute(args):
    if getattr(args, "exercise", None):
        e = EXERCISES[args.exercise]
        if getattr(args, "case", None) is not None and not 0 <= args.case < len(e["cases"]):
            raise ValueError(f"case must be in [0,{len(e['cases'])-1}]")
    name = args.command
    if name == "list": return {"command": name, "exercises": [brief(e) for e in selected(args.stage)]}
    if name == "task":
        files = [f"students/{e['backend']}/{e['id']}/entry.py"]
        if e["backend"] == "cuda": files.append(f"students/cuda/{e['id']}/kernel.cu")
        document = (ROOT / "docs/tasks" / (e["id"] + ".md")).read_text()
        tensors = re.findall(r'^\| `(inputs|outputs)` \| `"([a-z]+)"` \| `([^`]+)` \| ([^|]+) \|', document, re.M)
        return {**brief(e), "command": name, "exercise": e["id"], "purpose": e["purpose"],
                "dtypes": sorted({c["dtype"] for c in e["cases"]}), "student_files": files,
                "task_file": f"docs/tasks/{e['id']}.md", "tensors": [[v.strip() for v in row] for row in tensors], "cases": e["cases"]}
    if name == "doctor":
        from .environment import snapshot, require_environment, probe_blackwell_interfaces
        result = snapshot()
        result.update(command=name, checks={})
        for backend in sorted({e["backend"] for e in selected(args.stage)}):
            try: require_environment(backend); result["checks"][backend] = "ENVIRONMENT_OK"
            except EnvironmentUnsupported as exc: result["checks"][backend] = f"ENV_UNSUPPORTED: {exc}"
        if result["checks"].get("cute") == "ENVIRONMENT_OK": result["blackwell_api_symbols"] = probe_blackwell_interfaces()
        result["exit_code"] = 4 if any(v.startswith("ENV_UNSUPPORTED") for v in result["checks"].values()) else 0
        result["state"] = "ENV_UNSUPPORTED" if result["exit_code"] else "ENVIRONMENT_OK"
        return result
    if name == "status": return status_report(args.stage)
    if name == "smoke":
        cmd = [sys.executable, "-m", "flash_attention_lab", "_smoke"]
        if args.stage is not None: cmd += ["--stage", str(args.stage)]
        return managed(cmd, args)
    if name == "_smoke":
        from .smoke import infrastructure_smoke
        return {**infrastructure_smoke(args.stage), "command": "smoke"}
    if name == "check" and getattr(args, "stage", None) is not None:
        return check_stage(args)
    if name in ("check", "bench"): return managed(worker_command(args, name), args)
    if name == "_worker":
        from .runner import benchmark, check, save_result
        if args.operation == "check":
            result = check(e, args.case)
            if args.case is None: save_result(e, result)
        else: result = benchmark(e, 0 if args.case is None else args.case, args.warmup, args.repeats, args.iterations)
        return {**result, "command": args.operation}
    if name in ("sanitize", "profile"):
        from .api import require_implemented
        from .runtime import require_tool
        require_implemented(e)
        case = 0 if args.case is None else args.case
        if name == "sanitize":
            binary = require_tool("compute-sanitizer", "LAB_SANITIZER")
            cmd = [binary, "--tool", args.tool, "--error-exitcode", "3", sys.executable,
                   "-m", "flash_attention_lab", "_worker", args.exercise, "--operation", "check", "--case", str(case)]
        else:
            binary = require_tool("ncu", "LAB_NCU")
            from .environment import require_ncu_sections
            require_ncu_sections(binary)
            dest = ROOT / "artifacts/profiles"; dest.mkdir(parents=True, exist_ok=True)
            output = dest / (args.exercise + "_" + str(time.time_ns()))
            cmd = [binary, "--nvtx", "--nvtx-include", "lab_profile/", "--section", "LaunchStats",
                   "--section", "MemoryWorkloadAnalysis", "--section", "SpeedOfLight", "-o", str(output),
                   sys.executable, "-m", "flash_attention_lab", "_profile", args.exercise, "--case", str(case)]
        result = managed(cmd, args, tool=True)
        if name == "profile" and result["exit_code"] == 0:
            report = Path(str(output) + ".ncu-rep")
            if report.is_file() and report.stat().st_size:
                result["profile_report"] = str(report)
            else:
                result.update(state="TOOL_FAILED", exit_code=6,
                              message="NCU did not produce a report. See the tool log.")
            Path(result["artifact"]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    if name == "_profile":
        import torch
        from .runner import check
        from .cases import make_inputs
        from .api import prepare
        case = 0 if args.case is None else args.case
        validation = check(e, case)
        inputs, cfg = make_inputs(e, e["cases"][case], "cuda")
        plan = prepare(e["id"], inputs, cfg)
        for _ in range(3): plan.launch()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("lab_profile")
        plan.launch(); torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
        return {"command": "profile", "state": "PROFILE_COLLECTED", "exercise": e["id"], "case_index": case, "validation": validation}
    raise ValueError(f"unknown command: {name}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = argparse.Namespace(command="lab", json="--json" in argv, verbose="--verbose" in argv or "-v" in argv)
    try:
        # Python is a transparent passthrough, including its own flags and output.
        if argv and argv[0] == "python":
            from .runtime import configure_runtime
            configure_runtime()
            if len(argv) == 1: raise ValueError("usage: ./lab python -m MODULE [args] or ./lab python SCRIPT [args]")
            return subprocess.call([sys.executable, *argv[1:]], cwd=ROOT)
        args = parser().parse_args(argv)
        from .runtime import configure_runtime
        configure_runtime()
        result = execute(args)
        result.setdefault("exit_code", 0)
    except (StudentNotImplemented, EnvironmentUnsupported, ImportError, CorrectnessFailure,
            AssertionError, ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        result = error_report(exc, args)
    emit(result, args)
    return result["exit_code"]


if __name__ == "__main__": sys.exit(main())
