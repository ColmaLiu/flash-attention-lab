"""SASS inspection, Nsight Compute profiling and Compute Sanitizer."""

import collections
import re
import subprocess
import sys
import time

from . import ROOT, LabError
from .build import build, build_dir
from .env import require_tool

# Instructions worth counting when learning what the compiler produced.
INTERESTING = [
    ("LDG", "global load"),
    ("STG", "global store"),
    ("LDS", "shared load"),
    ("STS", "shared store"),
    ("LDSM", "ldmatrix"),
    ("LDGSTS", "cp.async"),
    ("HMMA", "mma.sync (Tensor Core)"),
    ("UTMALDG", "TMA load"),
    ("UTMASTG", "TMA store"),
    ("UTCHMMA", "tcgen05.mma f16/bf16"),
    ("UTCQMMA", "tcgen05.mma fp8"),
    ("UTCBAR", "tcgen05.commit"),
    ("LDTM", "tcgen05.ld"),
    ("STTM", "tcgen05.st"),
    ("SYNCS", "mbarrier"),
    ("MUFU.EX2", "exp2 (SFU)"),
    ("SHFL", "warp shuffle"),
    ("BAR", "barrier"),
    ("LDL", "local load (spill?)"),
    ("STL", "local store (spill?)"),
]


def _library(lab, solution):
    build(lab, solution)
    libs = sorted(
        build_dir(lab, solution).glob("*/*.so"), key=lambda p: p.stat().st_mtime
    )
    if not libs:
        raise LabError("no compiled library found")
    return libs[-1]


def sass(lab, solution=False, show=False):
    cuobjdump = require_tool("cuobjdump")
    lib = _library(lab, solution)
    usage = subprocess.run(
        [cuobjdump, "-res-usage", str(lib)], capture_output=True, text=True
    ).stdout
    code = subprocess.run(
        [cuobjdump, "-sass", str(lib)], capture_output=True, text=True
    ).stdout
    out = ROOT / "artifacts" / "sass"
    out.mkdir(parents=True, exist_ok=True)
    who = "solution" if solution else "student"
    dump = out / f"{who}_{lab.id}.sass"
    dump.write_text(code)

    print(f"Resource usage ({who} {lab.id}):")
    for fn, res in re.findall(r"Function (\S+):\n\s*(REG:.*)", usage):
        print(f"  {_demangle(fn)[:90]}")
        print(f"    {res.strip()}")
        if re.search(r"LOCAL:[1-9]", res) or re.search(r"STACK:[1-9]", res):
            print("    note: LOCAL/STACK > 0 usually means register spills")
    counts = collections.Counter()
    for line in code.splitlines():
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            counts[m.group(1)] += 1
    print("\nInstruction counts (static, whole module):")
    for prefix, meaning in INTERESTING:
        n = sum(
            v for k, v in counts.items() if k == prefix or k.startswith(prefix + ".")
        )
        if n:
            print(f"  {prefix:<10} {n:>6}  {meaning}")
    print(f"\nFull SASS: {dump.relative_to(ROOT)}")
    if show:
        print(code)
    return 0


def _demangle(name):
    try:
        return subprocess.run(
            ["c++filt", name], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return name


SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "LaunchStats",
    "Occupancy",
    "WarpStateStats",
    "ComputeWorkloadAnalysis",
]
# Printed for every lab, followed by the lab's own profile_metrics.
BASE_METRICS = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def profile_command(lab, case, case_args, step, solution):
    ncu = require_tool("ncu", "LAB_NCU")
    out = ROOT / "artifacts" / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    who = "solution" if solution else "student"
    report = out / f"{who}_{lab.id}_{case}_{time.strftime('%Y%m%d-%H%M%S')}"
    cmd = [ncu, "--nvtx", "--nvtx-include", "lab_profile/"]
    for s in SECTIONS:
        cmd += ["--section", s]
    cmd += ["--metrics", ",".join(BASE_METRICS + list(lab.profile_metrics))]
    cmd += [
        "-o",
        str(report),
        sys.executable,
        "-m",
        "labkit",
        "_profile",
        lab.id,
        *case_args,
    ]
    if step:
        cmd += ["--step", str(step)]
    if solution:
        cmd.append("--solution")
    return cmd, report


def sanitize_command(lab, tool, case_args, step, solution):
    sanitizer = require_tool("compute-sanitizer", "LAB_SANITIZER")
    cmd = [
        sanitizer,
        "--tool",
        tool,
        "--error-exitcode",
        "10",
        sys.executable,
        "-m",
        "labkit",
        "_check",
        lab.id,
        *case_args,
    ]
    if step:
        cmd += ["--step", str(step)]
    if solution:
        cmd.append("--solution")
    return cmd


def print_metrics(lab, report):
    """Print the chosen metrics for every profiled kernel."""
    import csv
    import io

    ncu = require_tool("ncu", "LAB_NCU")
    wanted = BASE_METRICS + list(lab.profile_metrics)
    text = subprocess.run(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--metrics",
            ",".join(wanted),
        ],
        capture_output=True,
        text=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return
    header, units = rows[0], rows[1]
    for row in rows[2:]:
        record = dict(zip(header, row))
        print(f"\nkernel: {_demangle(record.get('Kernel Name', '?'))[:100]}")
        for m in wanted:
            if m in record:
                unit = units[header.index(m)]
                print(f"  {m:<62} {record[m]:>14} {unit}")
