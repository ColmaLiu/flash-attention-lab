"""Per-case acceptance for final CUDA/PTX answers; no GPU work is performed here."""

import argparse
import json
import math
import statistics
from pathlib import Path

from .registry import CATALOG, EXERCISES, fingerprint, load_student


FINAL_SOLUTIONS = {
    "transpose": "s30_tma",
    "softmax": "s01_softmax",
    "gemm": "s32_specialize",
    "attention": "s43_schedule",
}
MIN_LIBRARY_PERFORMANCE = 0.9


def assess_solution_performance(exercise, report):
    """Assess the answer, including when a student benchmark contains that answer.

    An individual PASS covers this perf case only. Full acceptance requires every
    declared perf case of all four final answers, checked by the summary below.
    """
    result = {
        "state": "NOT_APPLICABLE",
        "implementation": "solution",
        "minimum_library_performance": MIN_LIBRARY_PERFORMANCE,
        "metric": "library_graph_median / solution_graph_median",
        "scope": "selected perf case only; not full final-answer acceptance",
    }
    if exercise["id"] not in FINAL_SOLUTIONS.values():
        return {
            **result,
            "reason": "intermediate lesson; measure the optimization without enforcing the final-answer threshold",
        }
    if not isinstance(report, dict) or not isinstance(report.get("case"), dict):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "the report and its case metadata must be objects",
        }
    if report["case"].get("suite") != "perf":
        return {
            **result,
            "reason": "only declared perf cases participate in final-answer acceptance",
        }
    if report.get("implementation") == "solution":
        answer = report
    elif report.get("implementation") == "student":
        answer = report.get("solution", {})
    else:
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "a solution/library comparison is required",
        }
    library = report.get("reference", {})
    if not isinstance(answer, dict) or not isinstance(library, dict):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "the solution and library records must be objects",
        }
    if report.get("exit_code", 0) != 0 or not library.get("provider"):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "a successful measurement with an explicit library backend is required",
        }
    validation = answer.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("state") != "CORRECTNESS_PASSED"
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "the answer must pass correctness before performance acceptance",
        }
    source = report.get("fingerprint")
    validation_source = validation.get("fingerprint")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(validation_source, str)
        or validation_source != source
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "correctness and performance must have the same nonempty source fingerprint",
        }
    checked_cases = validation.get("cases")
    case_index = report.get("case_index")
    if (
        report.get("exercise") != exercise["id"]
        or validation.get("exercise") != exercise["id"]
        or validation.get("implementation") != "solution"
        or type(case_index) is not int
        or not 0 <= case_index < len(exercise["cases"])
        or report.get("case") != exercise["cases"][case_index]
        or not isinstance(checked_cases, list)
        or type(validation.get("case_count")) is not int
        or validation["case_count"] != len(checked_cases)
        or not all(isinstance(case, dict) for case in checked_cases)
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "answer validation must identify the same exercise and checked case as the measurement",
        }
    selected_checks = [
        case
        for case in checked_cases
        if type(case.get("case")) is int and case["case"] == case_index
    ]
    if (
        len(selected_checks) != 1
        or selected_checks[0].get("state") != "PASS"
        or selected_checks[0].get("input") != report.get("case")
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "the measured case must have exactly one passing answer validation with identical input metadata",
        }
    answer_timing, library_timing = answer.get("timing", {}), library.get("timing", {})
    if not isinstance(answer_timing, dict) or not isinstance(library_timing, dict):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "both timing records must be objects",
        }
    if (
        answer_timing.get("mode") != "cuda_graph"
        or library_timing.get("mode") != "cuda_graph"
        or type(answer_timing.get("iterations")) is not int
        or type(library_timing.get("iterations")) is not int
        or answer_timing["iterations"] <= 0
        or answer_timing["iterations"] != library_timing.get("iterations")
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "both timings must be CUDA graph medians with the same positive captured batch size",
        }
    candidate_count = 2 if report["implementation"] == "solution" else 3
    timings = (answer_timing, library_timing)
    if (
        any(
            timing.get("sampling_order") != "cyclic rotation of candidate indices"
            or type(timing.get("candidate_count")) is not int
            or timing["candidate_count"] != candidate_count
            or type(timing.get("repeats")) is not int
            or timing["repeats"] <= 0
            for timing in timings
        )
        or answer_timing["repeats"] != library_timing["repeats"]
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "timings must describe the same cyclic sampling group, candidate count and positive repeat count",
        }
    answer_ms = answer_timing.get("gpu_stream_ms_median")
    library_ms = library_timing.get("gpu_stream_ms_median")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in (answer_ms, library_ms)
    ):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "timing medians must be positive finite numbers",
        }
    for timing in timings:
        samples = timing.get("gpu_stream_ms_samples")
        if (
            not isinstance(samples, list)
            or len(samples) != timing["repeats"]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in samples
            )
            or statistics.median(samples) != timing["gpu_stream_ms_median"]
        ):
            return {
                **result,
                "state": "INVALID_MEASUREMENT",
                "reason": "each timing median must equal the median of its finite positive raw samples, with one sample per repeat",
            }
    # These fields check report consistency with measure_graph_group. JSON alone
    # cannot prove that the measurements actually shared one execution/session.
    ratio = library_ms / answer_ms
    if not math.isfinite(ratio):
        return {
            **result,
            "state": "INVALID_MEASUREMENT",
            "reason": "the measured performance ratio must be finite",
        }
    # One representable step below the boundary handles division roundoff only.
    # It is not a noise allowance or a performance tolerance.
    passed = ratio >= math.nextafter(MIN_LIBRARY_PERFORMANCE, -math.inf)
    return {
        **result,
        "state": "PASS" if passed else "FAIL",
        "library_performance": ratio,
        "solution_ms": answer_ms,
        "library_ms": library_ms,
        "provider": library.get("provider", "unknown"),
    }


def summarize_final_performance(reports, *, expected_fingerprint=None):
    """Require every final perf case individually; never average away a failure.

    Pass expected_fingerprint when claiming acceptance for the current checkout.
    The offline CLI supplies its current fingerprint automatically.
    """
    expected = {
        (name, index): case
        for name in FINAL_SOLUTIONS.values()
        for index, case in enumerate(EXERCISES[name]["cases"])
        if case["suite"] == "perf"
    }
    observed, problems, fingerprints = {}, [], set()
    configs = {
        name: load_student(EXERCISES[name], solution=True).CONFIG
        for name in FINAL_SOLUTIONS.values()
    }
    for report in reports:
        if not isinstance(report, dict):
            problems.append("each input must contain one benchmark report object")
            continue
        key = (report.get("exercise"), report.get("case_index"))
        if not isinstance(key[0], str) or type(key[1]) is not int:
            problems.append(
                "each report must identify an exercise string and an integer case index"
            )
            continue
        if key not in expected:
            problems.append(
                f"unexpected report: {key}; provide only final perf comparisons"
            )
            continue
        if key in observed:
            problems.append(
                f"duplicate report: {key}; select a single controlled run explicitly"
            )
            continue
        if report.get("case") != expected[key]:
            problems.append(f"case metadata differs from the curriculum: {key}")
        case = expected[key]
        scale = case.get(
            "scale",
            1.0 / math.sqrt(case["shape"][-1])
            if EXERCISES[key[0]]["kind"] == "attention"
            else 1.0,
        )
        config = {**configs[key[0]], "causal": case["causal"], "scale": scale}
        answer = (
            report
            if report.get("implementation") == "solution"
            else report.get("solution", {})
        )
        if not isinstance(answer, dict) or answer.get("config") != config:
            problems.append(
                f"answer configuration differs from the shipped final default: {key}"
            )
        value = report.get("fingerprint")
        if not isinstance(value, str) or not value:
            problems.append(f"missing source fingerprint: {key}")
        else:
            fingerprints.add(value)
        observed[key] = assess_solution_performance(EXERCISES[key[0]], report)
    if len(fingerprints) > 1:
        problems.append("reports have different source fingerprints")
    if (
        expected_fingerprint is not None
        and fingerprints
        and fingerprints != {expected_fingerprint}
    ):
        problems.append("reports do not match the expected current-source fingerprint")
    rows = []
    for key, case in expected.items():
        row = observed.get(
            key, {"state": "MISSING", "reason": "this declared perf case has no report"}
        )
        rows.append(
            {"exercise": key[0], "case_index": key[1], "case_id": case["id"], **row}
        )
    # An actual failed case remains a failure even when additional cases are
    # absent. Otherwise missing/invalid evidence prevents an overall PASS.
    state = (
        "FAIL"
        if any(row["state"] == "FAIL" for row in rows)
        else "INCOMPLETE"
        if problems or any(row["state"] != "PASS" for row in rows)
        else "PASS"
    )
    return {
        "state": state,
        "minimum_library_performance": MIN_LIBRARY_PERFORMANCE,
        "final_solutions": FINAL_SOLUTIONS,
        "required_cases": len(expected),
        "passed_cases": sum(row["state"] == "PASS" for row in rows),
        "fingerprint": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
        "expected_fingerprint": expected_fingerprint,
        "cases": rows,
        "problems": problems,
        "scope": "all declared perf cases of the selected final answers; every case must reach 90%; no averaging; correctness and design acceptance remain separate",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate saved final-answer performance reports without running GPU work."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="JSON reports from lab solution or lab bench, one per final perf case",
    )
    args = parser.parse_args(argv)
    reports = []
    for path in args.reports:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read {path}: {exc}")
        reports.append(report)
    result = summarize_final_performance(
        reports, expected_fingerprint=fingerprint(CATALOG[0])
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["state"] == "PASS" else 3 if result["state"] == "FAIL" else 6


if __name__ == "__main__":
    raise SystemExit(main())
