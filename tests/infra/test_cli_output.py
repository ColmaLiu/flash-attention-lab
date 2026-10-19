"""Output contracts: truthful scope, usable JSON, and isolated child/tool logs."""
import argparse
import io
import json
from pathlib import Path
import sys

import pytest

from flash_attention_lab import cli, reporting
from flash_attention_lab.presentation import render, Terminal, width


@pytest.mark.parametrize('argv', [['list', '--json', '--stage', '0'], ['--json', 'list', '--stage', '0']])
def test_json_is_one_document_on_either_side_of_subcommand(argv, capsys):
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result['command'] == 'list' and len(result['exercises']) == 4
    assert captured.err == ''


def test_invalid_arguments_are_json_errors_with_nonzero_exit(capsys):
    assert cli.main(['bench', 's40_fa4_base', '--warmup', '0', '--json']) == 6
    result = json.loads(capsys.readouterr().out)
    assert result['state'] == 'EXECUTION_ERROR' and result['exit_code'] == 6
    assert 'positive' in result['message']


@pytest.mark.parametrize('command', ['task', 'check', 'bench', 'sanitize', 'profile'])
def test_short_ids_select_the_same_exercise_for_every_command(command):
    parser = cli.parser()
    for full_id in cli.EXERCISES:
        short_id = full_id.split('_', 1)[0]
        short = parser.parse_args([command, short_id])
        full = parser.parse_args([command, full_id])
        assert short == full
        assert short.exercise == full_id


def test_short_task_preserves_full_report_id_and_paths(capsys):
    assert cli.main(['task', 's20', '--json']) == 0
    short = json.loads(capsys.readouterr().out)
    assert cli.main(['task', 's20_fa2_grid', '--json']) == 0
    full = json.loads(capsys.readouterr().out)
    assert short == full
    assert short['exercise'] == 's20_fa2_grid'
    assert short['task_file'] == 'docs/tasks/s20_fa2_grid.md'


@pytest.mark.parametrize('value', ['s2', 's99', 's20_fa2', '20'])
def test_unknown_or_partial_ids_are_explicit_errors(value, capsys):
    assert cli.main(['check', value, '--json']) == 6
    result = json.loads(capsys.readouterr().out)
    assert result['state'] == 'EXECUTION_ERROR'
    assert f"unknown exercise {value!r}" in result['message']
    assert './lab list' in result['message']


def test_ambiguous_short_id_requires_full_name(monkeypatch):
    monkeypatch.setitem(cli.EXERCISES, 's20_alternative', {})
    parser = cli.parser()
    with pytest.raises(ValueError, match='ambiguous exercise') as caught:
        parser.parse_args(['check', 's20'])
    assert 's20_fa2_grid' in str(caught.value) and 's20_alternative' in str(caught.value)
    assert parser.parse_args(['check', 's20_fa2_grid']).exercise == 's20_fa2_grid'


def test_default_task_has_keys_and_verbose_lists_cases(capsys):
    assert cli.main(['task', 's00_softmax']) == 0
    summary = capsys.readouterr().out
    assert 'inputs' in summary and 'outputs' in summary and '[4,128]' in summary
    assert 'STUDENT_UNIMPLEMENTED' not in summary
    assert cli.main(['task', 's00_softmax', '--verbose']) == 0
    details = capsys.readouterr().out
    assert 'Case' in details and 'shifted' in details and 'Prerequisites' in details
    assert 'docs/tasks/s00_softmax.md' in details


def test_all_task_views_and_help_use_english(capsys):
    import re
    for exercise in cli.EXERCISES:
        assert cli.main(['task', exercise, '--verbose']) == 0
        output = capsys.readouterr().out
        assert 'inputs' in output and 'outputs' in output
        assert not re.search(r'[\u3400-\u9fff]', output)
    assert not re.search(r'[\u3400-\u9fff]', cli.parser().format_help())


def test_missing_student_has_human_state_and_action(monkeypatch, capsys):
    from flash_attention_lab import api
    from flash_attention_lab.errors import StudentNotImplemented
    def missing(e): raise StudentNotImplemented(e['id'] + ': missing student module')
    monkeypatch.setattr(api, 'require_implemented', missing)
    assert cli.main(['sanitize', 's03_online']) == 2
    output = capsys.readouterr().err
    assert 'STUDENT_UNIMPLEMENTED' in output and './lab task s03_online' in output


@pytest.fixture
def local_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(reporting, 'ROOT', tmp_path)
    return tmp_path


def arguments(**overrides):
    return argparse.Namespace(**{'command': 'check', 'exercise': 's00_softmax', 'case': 0,
                                 'timeout': 10, 'json': True, **overrides})


def test_native_and_python_child_noise_cannot_pollute_json(local_reports, capsys):
    script = '''import json,os,sys
from pathlib import Path
print('compiler stdout')
os.write(2,b'compiler stderr\\n')
Path(sys.argv[-1]).write_text(json.dumps({'state':'CORRECTNESS_PASSED','exit_code':0}))
'''
    args = arguments()
    result = cli.managed([sys.executable, '-c', script], args)
    cli.emit(result, args)
    output = capsys.readouterr()
    parsed = json.loads(output.out)
    assert output.err == '' and parsed['exit_code'] == 0
    log = Path(parsed['log']).read_text()
    assert 'compiler stdout' in log and 'compiler stderr' in log
    assert json.loads(Path(parsed['artifact']).read_text())['state'] == 'CORRECTNESS_PASSED'


def test_tool_failure_overrides_successful_target(local_reports):
    script = '''import json,sys
from pathlib import Path
Path(sys.argv[-1]).write_text(json.dumps({'state':'CORRECTNESS_PASSED','exit_code':0}))
print('sanitizer reported memory error')
sys.exit(3)
'''
    result = cli.managed([sys.executable, '-c', script], arguments(command='sanitize', tool='memcheck'), tool=True)
    assert result['exit_code'] == 3 and result['state'] == 'TOOL_FAILED'
    assert 'memory error' in result['message']


@pytest.mark.parametrize('script,timeout,code,state', [
    ('import time; time.sleep(3)', 0.05, 5, 'TIMEOUT'),
    ('pass', 10, 6, 'EXECUTION_ERROR'),
])
def test_timeout_or_missing_report_is_never_success(local_reports, script, timeout, code, state):
    result = cli.managed([sys.executable, '-c', script], arguments(timeout=timeout))
    assert result['exit_code'] == code and result['state'] == state


@pytest.mark.parametrize('report', ['{broken', '[]', '{"exit_code":3,"state":"CORRECTNESS_FAILED"}'])
def test_invalid_or_failed_report_with_zero_exit_is_not_success(local_reports, report):
    script = 'import sys; from pathlib import Path; Path(sys.argv[-1]).write_text(sys.argv[1])'
    result = cli.managed([sys.executable, '-c', script, report], arguments())
    assert result['exit_code'] == 6 and result['state'] == 'EXECUTION_ERROR'


def test_status_needs_no_progress_files_and_never_launches_checks(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from flash_attention_lab import environment
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(environment, 'require_environment', lambda backend: None)
    monkeypatch.setattr(cli, 'load_student', lambda e: SimpleNamespace(
        IMPLEMENTED=e['id'] == 's00_softmax', RUNTIME_REQUIRES=[]))
    def no_launch(*a, **k): pytest.fail('status must not launch a student check')
    monkeypatch.setattr(cli, 'managed', no_launch)
    assert cli.main(['status','--stage','0','--json']) == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    states = {e['id']: e['state'] for e in data['exercises']}
    assert len(states) == 4 and output.err == ''
    assert states['s00_softmax'] == 'IMPLEMENTED_UNVERIFIED'
    assert states['s03_online'] == 'STUDENT_UNIMPLEMENTED'
    assert not list(tmp_path.iterdir())


def test_profile_requires_an_actual_ncu_report(local_reports, monkeypatch):
    from flash_attention_lab import api, environment, runtime
    monkeypatch.setattr(api, 'require_implemented', lambda e: None)
    monkeypatch.setattr(runtime, 'require_tool', lambda *a: 'ncu')
    monkeypatch.setattr(environment, 'require_ncu_sections', lambda *a: None)
    artifact = local_reports / 'result.json'
    artifact.write_text('{}')
    monkeypatch.setattr(cli, 'managed', lambda *a, **k: {'command':'profile','state':'PROFILE_COLLECTED',
                                                     'exit_code':0,'artifact':str(artifact)})
    result = cli.execute(cli.parser().parse_args(['profile','s00_softmax','--case','0']))
    assert result['state'] == 'TOOL_FAILED' and result['exit_code'] == 6
    assert 'profile_report' not in result


def test_single_case_summary_does_not_claim_full_acceptance():
    result = {'command': 'check', 'exercise': 'example', 'full_contract': False, 'case_count': 1,
              'cases': [{'case': 0, 'state': 'PASS', 'input': {'shape': [4, 128], 'dtype': 'fp32', 'pattern': 'random'}}],
              'results': [{'case': 0, 'nondefault_stream': False, 'errors': {
                  'y': {'max_abs': 0.000012, 'max_rel_floored': 0.000034, 'atol': 2e-5, 'rtol': 2e-5}}}]}
    output = io.StringIO()
    render(result, stream=output)
    text = output.getvalue()
    assert 'selected case only' in text and 'PASS' in text
    assert '1.2e-05' in text and '1 passed, 0 failed' in text
    assert '\x1b' not in text


def test_sentinel_success_never_claims_student_success():
    output = io.StringIO()
    render({'command': 'smoke', 'modules_imported': 1, 'input_reference_cases': 2}, stream=output)
    assert 'Student kernels  NOT TESTED' in output.getvalue()


def test_chinese_tables_fit_narrow_terminals(monkeypatch):
    monkeypatch.setenv('COLUMNS', '48')
    output = io.StringIO()
    ui = Terminal(output)
    ui.table(['关卡', '状态', '说明'], [['s34_warp_specialize', '待验收', '旧记录已过期，需要重新运行完整验收']])
    assert all(width(line) <= 48 for line in output.getvalue().splitlines())
    assert '待验收' in output.getvalue()


@pytest.mark.parametrize('operation', ['check', 'bench'])
def test_failure_invalidates_prior_success_even_for_one_case(monkeypatch, operation):
    from flash_attention_lab import runner
    records = []
    monkeypatch.setattr(runner, 'save_result', lambda e, record: records.append(record))
    result = cli.error_report(RuntimeError('failed'), arguments(command='_worker', operation=operation))
    assert result['exit_code'] == 6
    assert records[0]['state'] == 'CORRECTNESS_FAILED' and records[0]['full_contract']


def test_benchmark_views_omit_accuracy_details_and_default_paths():
    data = {'command': 'bench', 'exercise': 'example', 'case_index': 0,
            'case': {'shape': [4, 128], 'dtype': 'fp32', 'pattern': 'random'},
            'environment': {}, 'config': {'causal': False}, 'output_and_workspace_bytes': 2048,
            'artifact': '/tmp/report.json', 'log': '/tmp/worker.log',
            'post_timing_errors': {'y': {'max_abs': 987.654321, 'max_rel_floored': 1.0}},
            'timing': {'gpu_stream_ms_median': 0.01, 'gpu_stream_ms_min': 0.009,
                       'gpu_stream_ms_max': 0.011, 'gpu_stream_ms_stdev': 0.001,
                       'gpu_stream_ms_samples': [0.009, 0.01, 0.011], 'host_batch_ms_median': 0.02,
                       'warmup': 2, 'repeats': 3, 'iterations': 5}}
    for verbose in (False, True):
        output = io.StringIO()
        render(data, stream=output, verbose=verbose)
        text = output.getvalue()
        assert 'Median' in text and 'validated' in text
        assert '987.654321' not in text and 'max_abs' not in text and 'atol' not in text
        assert ('/tmp/report.json' in text) == verbose
        assert ('/tmp/worker.log' in text) == verbose


@pytest.fixture
def simulated_check(monkeypatch):
    from flash_attention_lab import runner
    monkeypatch.setattr(runner, 'require_implemented', lambda e: None)
    monkeypatch.setattr(runner, 'require_environment', lambda backend: None)
    monkeypatch.setattr(runner, 'fingerprint', lambda e: 'test-code')
    monkeypatch.setattr(runner, 'save_result', lambda *a: None)
    # Only validation outcomes are simulated; no stand-in is installed in a student path.
    e = {'id': 's00_softmax', 'backend': 'cuda', 'cases': [
        {'shape': [4, 128], 'dtype': 'fp32', 'pattern': pattern} for pattern in ('random', 'large', 'zero')]}
    return runner, e


def test_check_preserves_all_case_results_on_numeric_failure(simulated_check, monkeypatch):
    from flash_attention_lab.errors import CorrectnessFailure
    runner, exercise = simulated_check
    calls = []
    def validate(e, case, *, nondefault):
        calls.append((case['pattern'], nondefault))
        if case['pattern'] == 'large':
            raise CorrectnessFailure('y first_bad=(0, 1) actual=0.0 ref=0.5')
        return {'y': {'max_abs': 0.0}}
    monkeypatch.setattr(runner, 'validate_case', validate)
    with pytest.raises(CorrectnessFailure) as caught:
        runner.check(exercise)
    report = cli.error_report(caught.value, arguments(command='_worker', operation='check', case=None))
    assert report['exit_code'] == 3
    assert [c['state'] for c in report['cases']] == ['PASS', 'FAIL', 'PASS']
    assert len(calls) == 6 and len(report['results']) == 6
    assert report['case_index'] == 1
    output = io.StringIO()
    render(report, stream=output)
    text = output.getvalue()
    assert '2 passed, 1 failed' in text
    assert 'first_bad=(0, 1)' in text and 'actual=0.0 ref=0.5' in text
    assert 'case 1 / default' in text and 'case 1 / nondefault' in text
    assert 'Reproduce: ./lab check s00_softmax --case 1' in text


def test_runtime_error_aborts_without_losing_prior_cases(simulated_check, monkeypatch):
    runner, exercise = simulated_check
    calls = []
    def validate(e, case, *, nondefault):
        calls.append((case['pattern'], nondefault))
        if case['pattern'] == 'large': raise RuntimeError('CUDA illegal memory access')
        return {'y': {'max_abs': 0.0}}
    monkeypatch.setattr(runner, 'validate_case', validate)
    with pytest.raises(RuntimeError) as caught:
        runner.check(exercise)
    report = cli.error_report(caught.value, arguments(command='_worker', operation='check', case=None))
    assert report['exit_code'] == 6 and len(calls) == 3
    assert [c['state'] for c in report['cases']] == ['PASS', 'ERROR', 'NOT_RUN']
    output = io.StringIO()
    render(report, stream=output)
    text = output.getvalue()
    assert 'NOT RUN' in text and 'CUDA illegal memory access' in text
    assert '1 passed, 0 failed, 1 errors, 1 not run' in text


def test_benchmark_still_stops_before_timing_failed_validation(simulated_check, monkeypatch):
    from flash_attention_lab.errors import CorrectnessFailure
    runner, exercise = simulated_check
    timed = []
    monkeypatch.setattr(runner, 'measure', lambda *a: timed.append(True))
    def fail(*a, **k): raise CorrectnessFailure('wrong output')
    monkeypatch.setattr(runner, 'validate_case', fail)
    with pytest.raises(CorrectnessFailure):
        runner.benchmark(exercise, 0, 2, 3, 5)
    assert timed == []
