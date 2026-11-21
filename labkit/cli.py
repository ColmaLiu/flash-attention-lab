"""./lab command line."""

import argparse
import json
import os
import signal
import subprocess
import sys

from . import ROOT, LabError, NotImplementedYet, spec

USAGE = """\
./lab list                         labs and your progress
./lab check ID [--step N] [--case NAME]
                                   correctness of students/ID/kernel.cu
./lab bench ID [--case NAME] [--step N] [--cold]
                                   time your kernel vs. the solution and libraries
./lab sass ID                      registers, spills and instruction mix
./lab profile ID --case NAME       Nsight Compute report
./lab sanitize ID --case NAME [--tool memcheck|racecheck|synccheck|initcheck]
./lab doctor                       check the environment
./lab clean [--yes]                delete build caches, bench/profile/sass output
Add --solution to check/bench/sass/profile/sanitize to run the reference solution.
ID may be a unique prefix, e.g. 00 or 31_tc.
"""


def _parser():
    p = argparse.ArgumentParser(prog="./lab", usage=USAGE)
    sub = p.add_subparsers(dest="cmd", required=True)

    def lab_cmd(name, **kw):
        q = sub.add_parser(name, **kw)
        q.add_argument("lab")
        q.add_argument("--solution", action="store_true")
        q.add_argument("--step", type=int)
        q.add_argument("--case")
        q.add_argument("--timeout", type=int, default=300)
        return q

    sub.add_parser("list")
    sub.add_parser("status")
    sub.add_parser("doctor")
    c = sub.add_parser("clean", help="delete build caches and measurement output")
    c.add_argument("--yes", action="store_true", help="actually delete")
    lab_cmd("check")
    b = lab_cmd("bench")
    b.add_argument("--cold", action="store_true", help="flush L2 before every launch")
    b.add_argument(
        "--no-solution",
        action="store_true",
        help="do not include the reference solution in the comparison",
    )
    s = lab_cmd("sass")
    s.add_argument("--show", action="store_true", help="print the full SASS")
    lab_cmd("profile")
    z = lab_cmd("sanitize")
    z.add_argument(
        "--tool",
        default="memcheck",
        choices=["memcheck", "racecheck", "synccheck", "initcheck"],
    )
    t = sub.add_parser("selftest", help="maintainers: verify solutions and skeletons")
    t.add_argument("labs", nargs="*")
    t.add_argument("--perf", action="store_true")
    t.add_argument("--timeout", type=int, default=900)
    # Internal: run in a child process so hangs and CUDA faults are contained.
    for name in ("_check", "_bench", "_profile"):
        q = lab_cmd(name)
        q.add_argument("--cold", action="store_true")
        q.add_argument("--no-solution", action="store_true")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except LabError as exc:
        print(
            f"\n{type(exc).__name__ if not isinstance(exc, NotImplementedYet) else 'TODO'}: {exc}"
        )
        return exc.exit_code
    except KeyboardInterrupt:
        return 130


def _dispatch(args):
    from . import env

    env.configure()
    cmd = args.cmd
    if cmd in ("list", "status"):
        return _list()
    if cmd == "clean":
        return _clean(args.yes)
    if cmd == "doctor":
        return 0 if env.doctor() else 4
    if cmd == "selftest":
        return _selftest(args)
    lab = spec.get(args.lab)
    if cmd in ("check", "bench"):
        return _child(["_" + cmd, *_forward(args, lab)], args.timeout)
    if cmd == "sass":
        from .tools import sass

        return sass(lab, args.solution, args.show)
    if cmd == "profile":
        from .tools import profile_command

        case = args.case or _default_perf_case(lab)
        command, report = profile_command(lab, case, args.step, args.solution)
        code = _child_raw(command, args.timeout)
        if code == 0:
            from .tools import print_metrics

            print_metrics(lab, f"{report}.ncu-rep")
            print(
                f"\nReport: {report.relative_to(ROOT)}.ncu-rep  (open with ncu-ui, or"
                f" ncu --import {report.relative_to(ROOT)}.ncu-rep --page details)"
            )
        return code
    if cmd == "sanitize":
        from .tools import sanitize_command

        if not args.case:
            raise LabError("choose a small case with --case (sanitizers are slow)")
        # Without the caching allocator every tensor is its own cudaMalloc,
        # so memcheck can see out-of-bounds accesses between tensors.
        code = _child_raw(
            sanitize_command(lab, args.tool, args.case, args.step, args.solution),
            args.timeout,
            {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
        )
        if code == 10:
            print(f"\n{args.tool} reported errors (see above).")
        return code
    if cmd == "_check":
        return _do_check(lab, args)
    if cmd == "_bench":
        return _do_bench(lab, args)
    if cmd == "_profile":
        return _do_profile(lab, args)
    raise LabError(f"unknown command {cmd}")


def _forward(args, lab):
    out = [lab.id]
    for flag in ("solution", "cold", "no_solution"):
        if getattr(args, flag, False):
            out.append("--" + flag.replace("_", "-"))
    if args.step:
        out += ["--step", str(args.step)]
    if args.case:
        out += ["--case", args.case]
    return out


def _child_raw(command, timeout, extra_env=None):
    environ = {**os.environ, **(extra_env or {})}
    proc = subprocess.Popen(command, start_new_session=True, cwd=ROOT, env=environ)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        print(
            f"\nTIMEOUT after {timeout}s. The kernel probably hangs, e.g. a thread waits "
            "on an mbarrier phase that never completes, or a loop never ends.\n"
            "Try a small --case and check every wait has a matching arrive."
        )
        return 5
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGKILL)
        raise


def _child(argv, timeout):
    return _child_raw([sys.executable, "-m", "labkit", *argv], timeout)


def _default_perf_case(lab):
    perf = lab.perf_cases()
    if not perf:
        raise LabError(f"{lab.id} has no perf case; pass --case")
    return perf[0].name


def _list():
    from .progress import summary

    part_names = {
        0: "Preliminaries",
        1: "FA1",
        2: "FA2",
        3: "FA3 ideas on SM100",
        4: "FA4",
    }
    current = None
    for lab in spec.labs().values():
        if lab.part != current:
            current = lab.part
            print(f"\nPart {lab.part} · {part_names.get(lab.part, '')}")
        state, stale = summary(lab)
        flag = " (code changed since last check)" if stale else ""
        print(f"  {lab.id:<20} steps {state:<5}  {lab.title}{flag}")
    print(
        "\nStart with: ./lab check 00 --step 1   (lesson: course/00_transpose/README.md)"
    )
    return 0


def _do_check(lab, args):
    from . import progress
    from .runner import check_case
    import torch  # noqa: F401  (after env.configure)

    from .env import require_gpu

    require_gpu()
    who = "solution" if args.solution else "student"
    steps = [lab.step_info(args.step)] if args.step else list(lab.steps)
    if args.case and not args.step:
        # Only the steps that list this case (perf cases: the last step).
        lab.case(args.case)
        steps = [s for s in steps if args.case in s.cases] or [lab.steps[-1]]
    # Maintainer runs (selftest, tests) set LAB_NO_PROGRESS to keep progress clean.
    record = (
        not args.solution and not args.case and not os.environ.get("LAB_NO_PROGRESS")
    )
    print(f"{lab.id} · {lab.title}  [{who}]")
    for s in steps:
        print(f"\nstep {s.step}: {s.title}")
        cases = [lab.case(args.case)] if args.case else [lab.case(n) for n in s.cases]
        for c in cases:
            try:
                metrics = check_case(lab, c, s.step, args.solution, full=not c.perf)
            except LabError as exc:
                print(
                    f"  {'TODO' if isinstance(exc, NotImplementedYet) else 'FAIL'}  {c.label()}"
                )
                print(f"        purpose: {c.purpose}")
                if record:
                    progress.record(lab, s.step, False, str(exc))
                raise
            worst = max(m.get("max_abs_err", 0.0) for m in metrics.values())
            print(f"  ok    {c.label():<52} max|err| {worst:.2e}")
        if record:
            progress.record(lab, s.step, True)
    done = "all steps" if not args.step else f"step {args.step}"
    print(f"\nPASS: {done} of {lab.id}" + (f" (case {args.case})" if args.case else ""))
    if not args.solution and args.step and args.step < len(lab.steps):
        print(f"Next: step {args.step + 1} — {lab.step_info(args.step + 1).title}")
    return 0


def _do_bench(lab, args):
    from .env import require_gpu
    from .runner import bench

    require_gpu()
    case = lab.case(args.case or _default_perf_case(lab))
    report = bench(
        lab,
        case,
        args.step,
        args.solution,
        "cold" if args.cold else "graph",
        not args.no_solution,
    )
    if os.environ.get("LAB_BENCH_REPORT"):  # selftest --perf reads this file
        from pathlib import Path

        Path(os.environ["LAB_BENCH_REPORT"]).write_text(json.dumps(report))
    rows = report["results"]
    work = report["work"]
    unit = "TFLOP/s" if work["flops"] else "GB/s"
    base = next((r for r in rows if r["name"] not in ("student", "solution")), None)
    print(
        f"\n{lab.id}  case {case.label()}  step {report['step'] or len(lab.steps)}  "
        f"timing: {report['mode']}"
    )
    head = f"  {'':<10} {'median ms':>10} {unit:>9} {'% peak':>7}"
    if base:
        head += f" {'vs ' + base['name']:>11}"
    print(head)
    for r in rows:
        value = r.get("tflops", r.get("gbps"))
        line = f"  {r['name']:<10} {r['median_ms']:>10.4f} {value:>9.1f} {100 * r['peak_fraction']:>6.1f}%"
        if base:
            line += f" {base['median_ms'] / r['median_ms']:>10.2f}x"
        print(line)
    for name, why in report["unavailable"].items():
        print(f"  {name:<10} unavailable: {why}")
    print("\n% peak uses nominal B200 datasheet numbers; 'vs' > 1 means faster.")
    return 0


def _do_profile(lab, args):
    import torch

    from .runner import Prepared, make_case

    case = lab.case(args.case)
    inputs, options = make_case(lab, case)
    p = Prepared(lab, case, inputs, options, args.step, args.solution)
    for _ in range(3):
        p.run_checked(torch.cuda.current_stream())
    torch.cuda.nvtx.range_push("lab_profile")
    p.launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return 0


def _selftest(args):
    """Maintainer check: solutions pass, skeletons compile and stop at TODO."""
    os.environ["LAB_NO_PROGRESS"] = "1"
    labs = [spec.get(x) for x in args.labs] if args.labs else list(spec.labs().values())
    failures = []
    for lab in labs:
        print(f"\n=== {lab.id}: solution")
        if _child(["_check", lab.id, "--solution"], args.timeout) != 0:
            failures.append(f"{lab.id}: solution check")
        print(f"\n=== {lab.id}: student skeleton (expect TODO)")
        code = _child(["_check", lab.id, "--step", "1"], args.timeout)
        if code != NotImplementedYet.exit_code:
            failures.append(f"{lab.id}: skeleton exit code {code}, expected TODO")
        if args.perf:
            failures += _perf_gate(lab, args.timeout)
    print(
        "\n"
        + (
            "selftest PASSED"
            if not failures
            else "selftest FAILED:\n  " + "\n  ".join(failures)
        )
    )
    return 0 if not failures else 1


def _perf_gate(lab, timeout):
    target = lab.perf_target
    if not target:
        return []
    failures = []
    for case in lab.perf_cases():
        # The child writes its report to a path we choose, so concurrent
        # benches by other processes cannot be mistaken for this one.
        name = f"gate_{lab.id}_{case.name}_{os.getpid()}.json"
        report_path = ROOT / "artifacts" / "bench" / name
        command = [
            sys.executable,
            "-m",
            "labkit",
            "_bench",
            lab.id,
            "--solution",
            "--case",
            case.name,
        ]
        code = _child_raw(command, timeout, {"LAB_BENCH_REPORT": str(report_path)})
        if code != 0 or not report_path.is_file():
            failures.append(f"{lab.id}/{case.name}: bench failed")
            continue
        report = json.loads(report_path.read_text())
        times = {r["name"]: r["median_ms"] for r in report["results"]}
        if target["baseline"] not in times:
            failures.append(
                f"{lab.id}/{case.name}: baseline {target['baseline']} unavailable"
            )
            continue
        ratio = times[target["baseline"]] / times["solution"]
        verdict = "ok" if ratio >= target["ratio"] else "BELOW TARGET"
        print(
            f"  perf gate {case.name}: {ratio:.3f} (target {target['ratio']}) {verdict}"
        )
        if ratio < target["ratio"]:
            failures.append(f"{lab.id}/{case.name}: {ratio:.3f} < {target['ratio']}")
    return failures


def _clean(yes):
    """Remove regenerable output only; never students/ or progress/."""
    import shutil

    targets = [
        ROOT / ".cache" / "build",
        ROOT / ".cache" / "tvm_ffi",
        *(ROOT / "artifacts" / d for d in ("bench", "profiles", "sass")),
    ]
    total = 0
    for t in targets:
        if t.exists():
            size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            total += size
            print(f"  {t.relative_to(ROOT)}  {size / 2**20:.0f} MiB")
    if not yes:
        print(
            f"{total / 2**20:.0f} MiB would be deleted. Run ./lab clean --yes to delete."
        )
        return 0
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    print(f"Deleted {total / 2**20:.0f} MiB.")
    return 0
