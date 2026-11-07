"""Final-answer acceptance rejects missing cases and hides no case behind an average."""

from copy import deepcopy
import io
import json
import math

import pytest

from flash_attention_lab import performance
from flash_attention_lab.presentation import render
from flash_attention_lab.registry import EXERCISES, load_student


def report(exercise="s01_softmax", index=8, *, ratio=1.0, source="current"):
    e = EXERCISES[exercise]
    case = e["cases"][index]
    scale = case.get(
        "scale", 1.0 / math.sqrt(case["shape"][-1]) if e["kind"] == "attention" else 1.0
    )
    config = {
        **load_student(e, solution=True).CONFIG,
        "causal": case["causal"],
        "scale": scale,
    }
    timing = {
        "mode": "cuda_graph",
        "iterations": 10,
        "repeats": 3,
        "sampling_order": "cyclic rotation of candidate indices",
        "candidate_count": 2,
        "gpu_stream_ms_median": 1.0,
        "gpu_stream_ms_samples": [1.0] * 3,
        "gpu_stream_ms_min": 1.0,
        "gpu_stream_ms_max": 1.0,
    }
    return {
        "exercise": exercise,
        "case_index": index,
        "case": deepcopy(e["cases"][index]),
        "implementation": "solution",
        "fingerprint": source,
        "environment": {},
        "config": config,
        "validation": {
            "state": "CORRECTNESS_PASSED",
            "fingerprint": source,
            "exercise": exercise,
            "implementation": "solution",
            "case_count": 1,
            "full_contract": False,
            "cases": [{"case": index, "input": deepcopy(case), "state": "PASS"}],
        },
        "timing": timing,
        "reference": {
            "provider": "test library",
            "timing": {
                **timing,
                "gpu_stream_ms_median": ratio,
                "gpu_stream_ms_samples": [ratio] * 3,
            },
        },
        "comparison": {"student_over_reference": 1.0 / ratio},
    }


def all_reports():
    return [
        report(name, index)
        for name in performance.FINAL_SOLUTIONS.values()
        for index, case in enumerate(EXERCISES[name]["cases"])
        if case["suite"] == "perf"
    ]


@pytest.mark.parametrize(
    "ratio,state", [(0.9, "PASS"), (0.899, "FAIL"), (0.9001, "PASS"), (2.0, "PASS")]
)
def test_inclusive_ninety_percent_boundary_without_performance_tolerance(ratio, state):
    r = report(ratio=ratio)
    result = performance.assess_solution_performance(EXERCISES[r["exercise"]], r)
    assert result["state"] == state
    assert result["library_performance"] == ratio


def test_roundoff_is_limited_to_one_representable_step():
    threshold = math.nextafter(0.9, -math.inf)
    for ratio, expected in (
        (threshold, "PASS"),
        (math.nextafter(threshold, -math.inf), "FAIL"),
    ):
        assert (
            performance.assess_solution_performance(
                EXERCISES["s01_softmax"], report(ratio=ratio)
            )["state"]
            == expected
        )


def test_intermediate_lessons_and_debug_cases_are_not_final_acceptance():
    for r in (report("s00_layout", 4, ratio=0.1), report("s01_softmax", 0, ratio=0.1)):
        assert (
            performance.assess_solution_performance(EXERCISES[r["exercise"]], r)[
                "state"
            ]
            == "NOT_APPLICABLE"
        )


def test_student_benchmark_assesses_the_answer_not_the_student():
    r = report(ratio=0.89)
    r["implementation"] = "student"
    r["solution"] = {
        "timing": deepcopy(r["timing"]),
        "validation": deepcopy(r["validation"]),
    }
    r["solution"]["timing"]["candidate_count"] = 3
    r["reference"]["timing"]["candidate_count"] = 3
    r["timing"]["gpu_stream_ms_median"] = (
        0.01  # A fast student cannot hide a slow answer.
    )
    result = performance.assess_solution_performance(EXERCISES["s01_softmax"], r)
    assert result["state"] == "FAIL" and result["library_performance"] == 0.89


@pytest.mark.parametrize("invalid", ["correctness", "mode", "batch", "zero", "nan"])
def test_invalid_comparisons_cannot_pass(invalid):
    r = report()
    if invalid == "correctness":
        r["validation"]["state"] = "CORRECTNESS_FAILED"
    elif invalid == "mode":
        r["reference"]["timing"]["mode"] = "eager"
    elif invalid == "batch":
        r["reference"]["timing"]["iterations"] = 20
    else:
        r["timing"]["gpu_stream_ms_median"] = 0.0 if invalid == "zero" else float("nan")
    assert (
        performance.assess_solution_performance(EXERCISES["s01_softmax"], r)["state"]
        == "INVALID_MEASUREMENT"
    )


@pytest.mark.parametrize("implementation", ["solution", "student"])
@pytest.mark.parametrize("source", [None, "", 17, "stale"])
def test_correctness_from_missing_or_different_source_cannot_accept_timings(
    implementation, source
):
    r = report()
    if source is None:
        del r["validation"]["fingerprint"]
    else:
        r["validation"]["fingerprint"] = source
    if implementation == "student":
        r["solution"] = {
            "timing": deepcopy(r["timing"]),
            "validation": deepcopy(r["validation"]),
        }
        r["implementation"] = implementation
        # A current student check must not authenticate a stale reference answer.
        r["validation"]["fingerprint"] = "current"
    result = performance.assess_solution_performance(EXERCISES["s01_softmax"], r)
    assert result["state"] == "INVALID_MEASUREMENT"
    assert "fingerprint" in result["reason"]


def test_matching_empty_fingerprints_do_not_authenticate_correctness():
    r = report(source="")
    assert (
        performance.assess_solution_performance(EXERCISES["s01_softmax"], r)["state"]
        == "INVALID_MEASUREMENT"
    )


@pytest.mark.parametrize("participant", ["solution", "library"])
def test_reported_median_cannot_replace_the_raw_sample_median(participant):
    r = report()
    timing = r["timing"] if participant == "solution" else r["reference"]["timing"]
    timing["gpu_stream_ms_samples"] = [0.5, 2.0, 3.0]
    result = performance.assess_solution_performance(EXERCISES[r["exercise"]], r)
    assert result["state"] == "INVALID_MEASUREMENT"
    assert "raw samples" in result["reason"]


@pytest.mark.parametrize(
    "samples",
    [
        None,
        [],
        "1,1,1",
        [1.0, 1.0],
        [1.0, float("nan"), 1.0],
        [1.0, float("inf"), 1.0],
        [1.0, True, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 0.0, 1.0],
    ],
)
def test_missing_or_invalid_raw_samples_cannot_accept_a_positive_median(samples):
    r = report()
    r["timing"]["gpu_stream_ms_samples"] = samples
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "INVALID_MEASUREMENT"
    )


def test_even_unsorted_raw_sample_median_is_accepted_without_rounding():
    r = report()
    for timing in (r["timing"], r["reference"]["timing"]):
        timing["repeats"] = 4
        timing["gpu_stream_ms_samples"] = [1.5, 0.75, 1.25, 0.5]
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "PASS"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampling_order", None),
        ("sampling_order", "independent runs"),
        ("candidate_count", None),
        ("candidate_count", 1),
        ("candidate_count", 3),
        ("candidate_count", 2.0),
        ("repeats", None),
        ("repeats", 0),
        ("repeats", True),
        ("repeats", 3.0),
        ("repeats", 4),
        ("iterations", True),
    ],
)
def test_group_fields_must_match_the_real_paired_runner(field, value):
    r = report()
    r["reference"]["timing"][field] = value
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "INVALID_MEASUREMENT"
    )


def test_agreeing_independent_timings_are_not_a_cyclic_sampling_group():
    r = report()
    for timing in (r["timing"], r["reference"]["timing"]):
        timing.pop("sampling_order")
        timing.pop("candidate_count")
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "INVALID_MEASUREMENT"
    )


@pytest.mark.parametrize("implementation", ["solution", "student"])
@pytest.mark.parametrize(
    "invalid",
    [
        "exercise",
        "implementation",
        "case_index",
        "case_input",
        "case_state",
        "missing_cases",
        "duplicate_case",
        "case_count",
    ],
)
def test_validation_must_certify_the_measured_answer_case(implementation, invalid):
    r = report()
    if implementation == "student":
        r["implementation"] = "student"
        r["solution"] = {
            "timing": deepcopy(r["timing"]),
            "validation": deepcopy(r["validation"]),
        }
        r["solution"]["timing"]["candidate_count"] = 3
        r["reference"]["timing"]["candidate_count"] = 3
        validation = r["solution"]["validation"]
    else:
        validation = r["validation"]
    if invalid == "exercise":
        validation["exercise"] = "s00_layout"
    elif invalid == "implementation":
        validation["implementation"] = "student"
    elif invalid == "case_index":
        validation["cases"][0]["case"] = 0
    elif invalid == "case_input":
        validation["cases"][0]["input"]["shape"] = [1, 1]
    elif invalid == "case_state":
        validation["cases"][0]["state"] = "FAIL"
    elif invalid == "missing_cases":
        del validation["cases"]
    elif invalid == "duplicate_case":
        validation["cases"].append(deepcopy(validation["cases"][0]))
        validation["case_count"] = 2
    else:
        validation["case_count"] = 2
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "INVALID_MEASUREMENT"
    )


def test_full_correctness_report_can_certify_its_selected_perf_case():
    r = report()
    validation = r["validation"]
    validation["full_contract"] = True
    validation["cases"] = [
        {"case": i, "input": deepcopy(case), "state": "PASS"}
        for i, case in enumerate(EXERCISES[r["exercise"]]["cases"])
    ]
    validation["case_count"] = len(validation["cases"])
    assert (
        performance.assess_solution_performance(EXERCISES[r["exercise"]], r)["state"]
        == "PASS"
    )


def test_summary_requires_all_thirteen_declared_final_perf_cases():
    reports = all_reports()
    summary = performance.summarize_final_performance(
        reports, expected_fingerprint="current"
    )
    assert summary["state"] == "PASS"
    assert summary["required_cases"] == summary["passed_cases"] == 13
    incomplete = performance.summarize_final_performance(
        reports[:-1], expected_fingerprint="current"
    )
    assert incomplete["state"] == "INCOMPLETE"
    assert incomplete["cases"][-1]["state"] == "MISSING"


def test_current_measurements_cannot_reuse_a_stale_correctness_pass():
    reports = all_reports()
    reports[-1]["validation"]["fingerprint"] = "old-correctness-source"
    summary = performance.summarize_final_performance(
        reports, expected_fingerprint="current"
    )
    assert summary["state"] == "INCOMPLETE"
    assert summary["passed_cases"] == 12
    assert summary["cases"][-1]["state"] == "INVALID_MEASUREMENT"
    assert "fingerprint" in summary["cases"][-1]["reason"]


def test_summary_accepts_the_case_generators_exact_math_configuration():
    from flash_attention_lab.cases import make_inputs

    reports = all_reports()
    for item in reports:
        if item["exercise"] == "s43_schedule":
            # Keep the generator's scale arithmetic, without allocating perf inputs.
            small = {**item["case"], "shape": [1, 1, 1, item["case"]["shape"][-1]]}
            _, math_config = make_inputs(EXERCISES[item["exercise"]], small)
            item["config"].update(math_config)
    assert (
        performance.summarize_final_performance(
            reports, expected_fingerprint="current"
        )["state"]
        == "PASS"
    )


def test_a_single_slow_case_fails_even_when_all_other_cases_are_faster():
    reports = all_reports()
    for r in reports:
        r["reference"]["timing"]["gpu_stream_ms_median"] = 4.0
        r["reference"]["timing"]["gpu_stream_ms_samples"] = [4.0] * 3
    reports[-1]["reference"]["timing"]["gpu_stream_ms_median"] = 0.89
    reports[-1]["reference"]["timing"]["gpu_stream_ms_samples"] = [0.89] * 3
    summary = performance.summarize_final_performance(
        reports, expected_fingerprint="current"
    )
    assert summary["state"] == "FAIL"
    assert summary["passed_cases"] == 12
    assert summary["cases"][-1]["state"] == "FAIL"


@pytest.mark.parametrize(
    "invalid",
    ["mixed_source", "stale_source", "duplicate", "changed_case", "extra", "ablation"],
)
def test_summary_cannot_accept_incoherent_or_selectively_repeated_evidence(invalid):
    reports = all_reports()
    if invalid == "mixed_source":
        reports[0]["fingerprint"] = "old"
    elif invalid == "stale_source":
        for r in reports:
            r["fingerprint"] = "old"
    elif invalid == "duplicate":
        reports.append(deepcopy(reports[0]))
    elif invalid == "changed_case":
        reports[0]["case"]["shape"] = [1, 1]
    elif invalid == "ablation":
        reports[-1]["config"]["persistent"] = False
    else:
        reports.append(report("s00_layout", 4))
    summary = performance.summarize_final_performance(
        reports, expected_fingerprint="current"
    )
    assert summary["state"] == "INCOMPLETE" and summary["problems"]


def test_offline_cli_reports_missing_cases_and_returns_nonzero(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "single.json"
    path.write_text(json.dumps(report()))
    monkeypatch.setattr(performance, "fingerprint", lambda e: "current")
    assert performance.main([str(path)]) == 6
    assert json.loads(capsys.readouterr().out)["state"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "slow_case,code,state", [(False, 0, "PASS"), (True, 3, "FAIL")]
)
def test_offline_cli_exit_code_matches_complete_evidence(
    tmp_path, monkeypatch, capsys, slow_case, code, state
):
    reports = all_reports()
    if slow_case:
        reports[-1]["reference"]["timing"]["gpu_stream_ms_median"] = 0.89
        reports[-1]["reference"]["timing"]["gpu_stream_ms_samples"] = [0.89] * 3
    paths = []
    for index, item in enumerate(reports):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(item))
        paths.append(str(path))
    monkeypatch.setattr(performance, "fingerprint", lambda e: "current")
    assert performance.main(paths) == code
    assert json.loads(capsys.readouterr().out)["state"] == state


@pytest.mark.parametrize(
    "field",
    [
        ("case",),
        ("reference",),
        ("validation",),
        ("timing",),
        ("reference", "timing"),
        ("solution",),
        ("solution", "validation"),
        ("solution", "timing"),
    ],
)
@pytest.mark.parametrize("invalid", [None, [], "broken record"])
def test_offline_cli_reports_malformed_nested_records_as_incomplete(
    tmp_path, monkeypatch, capsys, field, invalid
):
    reports = all_reports()
    bad = reports[0]
    if field[0] == "solution":
        bad["implementation"] = "student"
        bad["solution"] = {
            key: deepcopy(bad[key]) for key in ("timing", "validation", "config")
        }
        bad["solution"]["timing"]["candidate_count"] = 3
        bad["reference"]["timing"]["candidate_count"] = 3
    record = bad
    for key in field[:-1]:
        record = record[key]
    record[field[-1]] = invalid
    paths = []
    for index, item in enumerate(reports):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(item))
        paths.append(str(path))
    monkeypatch.setattr(performance, "fingerprint", lambda e: "current")
    assert performance.main(paths) == 6
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "INCOMPLETE" and result["passed_cases"] == 12
    assert result["cases"][0]["state"] == "INVALID_MEASUREMENT"


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "not a report",
        {"exercise": []},
        {"exercise": "s30_tma", "case_index": {}},
    ],
)
def test_offline_cli_rejects_malformed_report_identity_without_a_traceback(
    tmp_path, monkeypatch, capsys, invalid
):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid))
    monkeypatch.setattr(performance, "fingerprint", lambda e: "current")
    assert performance.main([str(path)]) == 6
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "INCOMPLETE" and result["problems"]


def test_report_schema_checks_do_not_hide_source_loading_errors(monkeypatch):
    reports = all_reports()

    def broken_source(*args, **kwargs):
        raise RuntimeError("solution module failed to load")

    monkeypatch.setattr(performance, "load_student", broken_source)
    with pytest.raises(RuntimeError, match="solution module failed to load"):
        performance.summarize_final_performance(reports)


def test_terminal_exposes_per_case_failure_without_claiming_full_acceptance():
    r = report(ratio=0.89)
    r["command"] = "solution"
    r["solution_performance"] = performance.assess_solution_performance(
        EXERCISES["s01_softmax"], r
    )
    output = io.StringIO()
    render(r, stream=output)
    text = output.getvalue()
    assert "Final answer target" in text and "FAIL" in text
    assert "89.00%" in text and "this case only" in text
