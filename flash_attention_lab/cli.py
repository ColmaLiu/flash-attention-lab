import argparse
import json
import os
import signal
import subprocess
import sys
from .errors import CorrectnessFailure, EnvironmentUnsupported, StudentNotImplemented
from .registry import CATALOG, EXERCISES, ROOT, fingerprint, load_student, selected


def positive_int(value):
    n = int(value)
    if n < 1: raise argparse.ArgumentTypeError("must be positive")
    return n


def parser():
    p = argparse.ArgumentParser(description="B200 FlashAttention learning infrastructure; no student fallback")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("doctor", "smoke", "status", "list"):
        q = sub.add_parser(command)
        q.add_argument("--stage", type=int, choices=range(5))
        if command == "smoke": q.add_argument("--timeout", type=float, default=600)
        if command == "status": q.add_argument("--check-enabled", action="store_true")
    for command in ("task", "enable"):
        q = sub.add_parser(command); q.add_argument("exercise", choices=EXERCISES)
    for command in ("check", "bench", "sanitize", "profile", "_worker", "_profile"):
        q = sub.add_parser(command)
        q.add_argument("exercise", choices=EXERCISES)
        q.add_argument("--case", type=int)
        q.add_argument("--timeout", type=float, default=180)
        if command in ("bench", "_worker"):
            q.add_argument("--warmup", type=positive_int, default=5)
            q.add_argument("--repeats", type=positive_int, default=7)
            q.add_argument("--iterations", type=positive_int, default=10)
        if command == "sanitize": q.add_argument("--tool", choices=("memcheck", "racecheck", "synccheck", "initcheck"), default="memcheck")
        if command == "_worker": q.add_argument("--operation", choices=("check", "bench"), required=True)
    q = sub.add_parser("_smoke"); q.add_argument("--stage", type=int, choices=range(5))
    q = sub.add_parser("python", help="run Python with the same environment/device configuration")
    q.add_argument("args", nargs=argparse.REMAINDER)
    return p


def bounded(command, timeout):
    if timeout <= 0: raise ValueError("timeout must be positive")
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try: return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        print(f"TIMEOUT: exceeded {timeout}s; only this command's process group was stopped", file=sys.stderr)
        return 5


def worker_command(args, operation):
    cmd = [sys.executable, "-m", "flash_attention_lab", "_worker", args.exercise, "--operation", operation]
    if args.case is not None: cmd += ["--case", str(args.case)]
    if operation == "bench":
        cmd += ["--warmup", str(args.warmup), "--repeats", str(args.repeats), "--iterations", str(args.iterations)]
    return cmd


def show_status(stage, check_enabled=False):
    from .environment import require_environment
    enabled_path = ROOT / "progress" / "enabled.json"
    enabled = json.loads((enabled_path if enabled_path.exists() else ROOT / "progress/defaults.json").read_text())
    env = {}
    for backend in {e["backend"] for e in selected(stage)}:
        try: require_environment(backend); env[backend] = "ENVIRONMENT_OK (run smoke to verify compilation)"
        except EnvironmentUnsupported as exc: env[backend] = f"ENV_UNSUPPORTED: {exc}"
    print(json.dumps({"environment": env, "note": "inventory only; no correctness test implied"}, ensure_ascii=False))
    rc = 0
    for e in selected(stage):
        module = load_student(e)
        reason = "entry flag is False"
        state = "STUDENT_UNIMPLEMENTED"
        if module.IMPLEMENTED:
            missing = [d for d in module.RUNTIME_REQUIRES if d not in EXERCISES or not load_student(EXERCISES[d]).IMPLEMENTED]
            if missing: reason = "missing runtime modules: " + ", ".join(missing)
            elif env[e["backend"]].startswith("ENV_UNSUPPORTED"):
                state, reason = "ENV_UNSUPPORTED", env[e["backend"]]
            else:
                state, reason = "IMPLEMENTED_UNVERIFIED", "no full-contract result for current source"
                path = ROOT / "progress" / "results" / (e["id"] + ".json")
                if path.exists():
                    data = json.loads(path.read_text())
                    if data.get("fingerprint") == fingerprint(e) and data.get("full_contract"):
                        state, reason = data["state"], data.get("reason", "numerical result only; design review separate")
                    else: reason = "previous result stale or only a single-case check"
        print(f"{e['id']:23} {'enabled' if e['id'] in enabled else 'disabled':8} {state:24} {reason}")
        if check_enabled and e["id"] in enabled:
            code = bounded([sys.executable, "-m", "flash_attention_lab", "_worker", e["id"], "--operation", "check"], 300)
            rc = max(rc, code)
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        from .runtime import configure_runtime
        configure_runtime()
        if argv and argv[0] == "python":
            if len(argv) == 1: raise ValueError("usage: ./lab python -m MODULE [args] or ./lab python SCRIPT [args]")
            return subprocess.call([sys.executable, *argv[1:]], cwd=ROOT)
        args = parser().parse_args(argv)
        if hasattr(args, "exercise"):
            e = EXERCISES[args.exercise]
            if getattr(args, "case", None) is not None and not 0 <= args.case < len(e["cases"]):
                raise ValueError(f"case must be in [0,{len(e['cases'])-1}]")
        if args.command == "list":
            for e in selected(args.stage): print(f"{e['id']:23} {e['title']} | prerequisites: {', '.join(e['prerequisites']) or '-'}")
        elif args.command == "task": print((ROOT / "docs/tasks" / (args.exercise + ".md")).read_text())
        elif args.command == "enable":
            p = ROOT / "progress/enabled.json"
            enabled = json.loads((p if p.exists() else ROOT / "progress/defaults.json").read_text())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(sorted(set(enabled + [args.exercise])), indent=2) + "\n")
            print("Enabled in default checks; student implementation flag is unchanged")
        elif args.command == "doctor":
            from .environment import snapshot, require_environment, probe_blackwell_interfaces
            result = snapshot()
            result["checks"] = {}
            for backend in sorted({e["backend"] for e in selected(args.stage)}):
                try:
                    require_environment(backend)
                    result["checks"][backend] = "ENVIRONMENT_OK"
                except EnvironmentUnsupported as exc:
                    result["checks"][backend] = f"ENV_UNSUPPORTED: {exc}"
            if result["checks"].get("cute") == "ENVIRONMENT_OK":
                result["blackwell_api_symbols"] = probe_blackwell_interfaces()
            print(json.dumps(result, indent=2))
            return 4 if any(v.startswith("ENV_UNSUPPORTED") for v in result["checks"].values()) else 0
        elif args.command == "status": return show_status(args.stage, args.check_enabled)
        elif args.command == "smoke":
            cmd = [sys.executable, "-m", "flash_attention_lab", "_smoke"]
            if args.stage is not None: cmd += ["--stage", str(args.stage)]
            return bounded(cmd, args.timeout)
        elif args.command == "_smoke":
            from .smoke import infrastructure_smoke
            from .runner import save_artifact
            result = infrastructure_smoke(args.stage)
            path = save_artifact("infra_smoke", result)
            print(json.dumps(result, indent=2)); print("ARTIFACT:", path)
        elif args.command in ("check", "bench"):
            return bounded(worker_command(args, args.command), args.timeout)
        elif args.command == "_worker":
            from .runner import benchmark, check, save_artifact, save_result
            if args.operation == "check":
                result = check(e, args.case)
                if args.case is None: save_result(e, result)
            else:
                result = benchmark(e, 0 if args.case is None else args.case, args.warmup, args.repeats, args.iterations)
            path = save_artifact(e["id"] + "_" + args.operation, result)
            print(json.dumps(result, indent=2)); print("ARTIFACT:", path)
        elif args.command in ("sanitize", "profile"):
            from .api import require_implemented
            from .runtime import require_tool
            require_implemented(e)  # explicit missing-module error even if tool is absent
            case = 0 if args.case is None else args.case
            if args.command == "sanitize":
                binary = require_tool("compute-sanitizer", "LAB_SANITIZER")
                cmd = [binary, "--tool", args.tool, "--error-exitcode", "3", sys.executable,
                       "-m", "flash_attention_lab", "_worker", args.exercise, "--operation", "check", "--case", str(case)]
            else:
                binary = require_tool("ncu", "LAB_NCU")
                from .environment import require_ncu_sections
                require_ncu_sections(binary)
                dest = ROOT / "artifacts/profiles"; dest.mkdir(parents=True, exist_ok=True)
                import time
                output = dest / (args.exercise + "_" + str(time.time_ns()))
                cmd = [binary, "--nvtx", "--nvtx-include", "lab_profile/", "--section", "LaunchStats",
                       "--section", "MemoryWorkloadAnalysis", "--section", "SpeedOfLight", "-o", str(output),
                       sys.executable, "-m", "flash_attention_lab", "_profile", args.exercise, "--case", str(case)]
            return bounded(cmd, args.timeout)
        elif args.command == "_profile":
            import torch
            from .runner import check
            from .cases import make_inputs
            from .api import prepare
            case = 0 if args.case is None else args.case
            check(e, case)
            inputs, cfg = make_inputs(e, e["cases"][case], "cuda")
            plan = prepare(e["id"], inputs, cfg)
            for _ in range(3): plan.launch()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("lab_profile")
            plan.launch()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        return 0
    except StudentNotImplemented as exc:
        print(f"STUDENT_UNIMPLEMENTED: {exc}", file=sys.stderr); return 2
    except (EnvironmentUnsupported, ImportError) as exc:
        print(f"ENV_UNSUPPORTED: {exc}", file=sys.stderr); return 4
    except (CorrectnessFailure, AssertionError) as exc:
        print(f"CORRECTNESS_FAILED: {exc}", file=sys.stderr)
        if "args" in locals() and args.command == "_worker":
            from .runner import save_result
            save_result(e, {"state": "CORRECTNESS_FAILED", "fingerprint": fingerprint(e), "full_contract": True, "reason": str(exc)})
        return 3
    except (ValueError, RuntimeError, OSError) as exc:
        text = str(exc)
        if "STUDENT_TODO" in text:
            print("STUDENT_UNIMPLEMENTED:", text, file=sys.stderr); return 2
        print(f"EXECUTION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "args" in locals() and args.command == "_worker":
            from .runner import save_result
            save_result(e, {"state": "CORRECTNESS_FAILED", "fingerprint": fingerprint(e), "full_contract": True, "reason": "execution error: " + text})
        return 6
