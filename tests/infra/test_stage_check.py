"""Stage orchestration tests use isolated reports, never substitute student kernels."""
import json
from pathlib import Path
import sys

import pytest

from flash_attention_lab import cli, reporting


@pytest.fixture
def local_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    return tmp_path


def result_for(e, state="CORRECTNESS_PASSED", code=0):
    result = {"command": "check", "exercise": e["id"], "state": state, "exit_code": code,
              "case_count": len(e["cases"]), "full_contract": True, "results": []}
    if state in ("CORRECTNESS_PASSED", "CORRECTNESS_FAILED", "EXECUTION_ERROR"):
        result["cases"] = [{"case": i, "input": case, "state": "PASS"}
                           for i, case in enumerate(e["cases"])]
        if code:
            result["cases"][0]["state"] = "ERROR" if code == 6 else "FAIL"
            result["results"] = [{"case": 0, "nondefault_stream": False,
                                  "state": result["cases"][0]["state"], "message": "fixture failure"}]
    if code: result["message"] = "fixture failure"
    return result


@pytest.mark.parametrize("arguments", [
    ["check"],
    ["check", "s20", "--stage", "2"],
    ["check", "--stage", "5"],
    ["check", "--stage", "2", "--case", "0"],
    ["check", "--stage", "2", "--timeout", "0"],
    ["check", "--stage", "2", "--timeout", "nan"],
    ["check", "--stage", "2", "--timeout", "inf"],
])
def test_invalid_stage_scope_never_launches_checks(arguments, monkeypatch, capsys):
    def unexpected(*args, **kwargs): pytest.fail("invalid scope launched a worker")
    monkeypatch.setattr(cli, "managed", unexpected)
    assert cli.main([*arguments, "--json"]) == 6
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["state"] == "EXECUTION_ERROR" and output.err == ""


def test_stage_checks_all_exercises_despite_mixed_failures(local_reports, monkeypatch, capsys):
    exercises = cli.selected(2)
    outcomes = [("CORRECTNESS_PASSED", 0), ("STUDENT_UNIMPLEMENTED", 2),
                ("CORRECTNESS_FAILED", 3), ("TIMEOUT", 5), ("EXECUTION_ERROR", 6)]
    calls = []
    def managed(command, args):
        e = exercises[len(calls)]
        assert args.exercise == e["id"] and args.case is None and args.timeout == 7
        assert command[-4:] == ["_worker", e["id"], "--operation", "check"]
        calls.append(args.exercise)
        return result_for(e, *outcomes[len(calls)-1])
    monkeypatch.setattr(cli, "managed", managed)
    assert cli.main(["check", "--stage", "2", "--timeout", "7", "--json"]) == 2
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""  # stdout must be exactly one JSON object.
    assert calls == [e["id"] for e in exercises]
    assert report["state"] == "STAGE_FAILED"
    assert [(r["state"], r["exit_code"]) for r in report["exercises"]] == outcomes
    saved = json.loads(Path(report["artifact"]).read_text())
    assert saved["exercises"] == report["exercises"]


def test_stage_success_prints_each_case_and_final_summary(local_reports, monkeypatch, capsys):
    monkeypatch.setattr(cli, "managed", lambda command, args: result_for(cli.EXERCISES[args.exercise]))
    assert cli.main(["check", "--stage", "1"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.count(": check (all cases)") == 2
    assert "s10_fa1" in output.out and "s11_fa1_edges" in output.out
    assert "Stage 1: check summary" in output.out
    assert "2/2 exercises passed" in output.out
    report = json.loads(next((local_reports / "artifacts/runs").glob("check_stage_*.json")).read_text())
    assert report["state"] == "STAGE_PASSED"


@pytest.mark.parametrize("state,code,label", [("STUDENT_UNIMPLEMENTED", 2, "NOT IMPLEMENTED"),
                                              ("ENV_UNSUPPORTED", 4, "UNSUPPORTED")])
def test_stage_missing_implementations_or_environment_are_not_passes(
        local_reports, monkeypatch, capsys, state, code, label):
    monkeypatch.setattr(cli, "managed", lambda command, args: result_for(cli.EXERCISES[args.exercise], state, code))
    assert cli.main(["check", "--stage", "1"]) == code
    text = capsys.readouterr().out
    assert "0/2 exercises passed" in text and label in text
    assert "fixture failure" in text


def test_interrupt_preserves_completed_results_and_stops_remaining_exercises(local_reports, monkeypatch, capsys):
    calls = []
    def managed(command, args):
        calls.append(args.exercise)
        if len(calls) == 2: raise KeyboardInterrupt()
        return result_for(cli.EXERCISES[args.exercise])
    monkeypatch.setattr(cli, "managed", managed)
    assert cli.main(["--json", "check", "--stage", "2"]) == 130
    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "INTERRUPTED"
    assert len(calls) == 2
    assert [r["state"] for r in report["exercises"]] == ["CORRECTNESS_PASSED", "INTERRUPTED", *["NOT_RUN"]*3]
    assert Path(report["artifact"]).exists()


def test_worker_launch_error_does_not_abort_the_stage(local_reports, monkeypatch, capsys):
    calls = []
    def managed(command, args):
        calls.append(args.exercise)
        if len(calls) == 1: raise OSError("cannot start fixture worker")
        return result_for(cli.EXERCISES[args.exercise])
    monkeypatch.setattr(cli, "managed", managed)
    assert cli.main(["check", "--stage", "1", "--json"]) == 6
    report = json.loads(capsys.readouterr().out)
    assert [r["state"] for r in report["exercises"]] == ["EXECUTION_ERROR", "CORRECTNESS_PASSED"]
    assert "cannot start" in report["exercises"][0]["message"]


def test_timed_out_child_is_isolated_and_next_child_runs(local_reports, monkeypatch, capsys):
    # Lightweight subprocess fixtures exercise the real timeout/log/report path.
    # They do not call, replace or mark any student implementation as correct.
    exercises = cli.selected(1)
    script = """import json,sys,time
from pathlib import Path
print('fixture child output', flush=True)
time.sleep(float(sys.argv[1]))
Path(sys.argv[-1]).write_text(sys.argv[2])
"""
    def command(args, operation):
        e = cli.EXERCISES[args.exercise]
        delay = "3" if e["id"] == exercises[0]["id"] else "0"
        return [sys.executable, "-c", script, delay, json.dumps(result_for(e))]
    monkeypatch.setattr(cli, "worker_command", command)
    assert cli.main(["check", "--stage", "1", "--timeout", "0.5", "--json"]) == 5
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert [r["state"] for r in report["exercises"]] == ["TIMEOUT", "CORRECTNESS_PASSED"]
    assert len({r["log"] for r in report["exercises"]}) == 2
    assert all(Path(r["log"]).exists() for r in report["exercises"])
    assert "fixture child output" in Path(report["exercises"][1]["log"]).read_text()
