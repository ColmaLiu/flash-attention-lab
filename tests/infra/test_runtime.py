"""Runtime regressions that matter when cloning on another machine."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from flash_attention_lab import runtime
from flash_attention_lab.errors import EnvironmentUnsupported


@pytest.mark.parametrize('environ,expected', [
    ({}, '0'),
    ({'LAB_GPU': '2'}, '2'),
    ({'CUDA_VISIBLE_DEVICES': '3,7'}, '3'),
    ({'CUDA_VISIBLE_DEVICES': '3,7', 'LAB_GPU': '7'}, '7'),
    ({'CUDA_VISIBLE_DEVICES': 'GPU-abc,GPU-def', 'LAB_GPU': 'GPU-def'}, 'GPU-def'),
    ({'CUDA_VISIBLE_DEVICES': ''}, ''),
    ({'CUDA_VISIBLE_DEVICES': '-1'}, '-1'),
])
def test_single_device_respects_inherited_mask(environ, expected):
    assert runtime.selected_device(environ) == expected
    inherited = {**environ, 'CUDA_VISIBLE_DEVICES': expected}
    assert runtime.selected_device(inherited) == expected  # child processes keep the same device


@pytest.mark.parametrize('environ', [
    {'LAB_GPU': '0,1'}, {'LAB_GPU': '-1'}, {'LAB_GPU': ''},
    {'CUDA_VISIBLE_DEVICES': '3,7', 'LAB_GPU': '0'},
    {'CUDA_VISIBLE_DEVICES': '', 'LAB_GPU': '0'},
    {'CUDA_VISIBLE_DEVICES': 'GPU-abc', 'LAB_GPU': 'GPU-def'},
])
def test_invalid_device_or_mask_expansion_is_rejected(environ):
    with pytest.raises(EnvironmentUnsupported):
        runtime.selected_device(environ)


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(0o755)
    return path


def test_toolkit_discovery_and_explicit_override(tmp_path, monkeypatch):
    toolkit = tmp_path / 'custom toolkit'
    executable(toolkit / 'bin/nvcc')
    monkeypatch.setenv('PATH', str(toolkit / 'bin'))
    monkeypatch.delenv('CUDA_HOME', raising=False)
    monkeypatch.delenv('LAB_CUDA_HOME', raising=False)
    assert runtime.cuda_home() == toolkit
    alternate = tmp_path / 'alternate'
    monkeypatch.setenv('CUDA_HOME', str(alternate))
    assert runtime.cuda_home() == alternate
    missing = tmp_path / 'missing'
    monkeypatch.setenv('LAB_CUDA_HOME', str(missing))
    assert runtime.cuda_home() == missing  # do not silently bypass an invalid explicit selection


def test_tool_discovery_accepts_names_paths_and_toolkit_fallback(tmp_path, monkeypatch):
    toolkit = tmp_path / 'toolkit'
    fallback = executable(toolkit / 'bin/ncu')
    preferred = executable(tmp_path / 'tools/ncu')
    monkeypatch.setenv('CUDA_HOME', str(toolkit))
    monkeypatch.delenv('LAB_CUDA_HOME', raising=False)
    monkeypatch.delenv('LAB_NCU', raising=False)
    monkeypatch.setenv('PATH', str(preferred.parent))
    assert runtime.require_tool('ncu', 'LAB_NCU') == str(preferred)
    monkeypatch.setenv('LAB_NCU', 'ncu')
    assert runtime.require_tool('ncu', 'LAB_NCU') == str(preferred)
    monkeypatch.setenv('LAB_NCU', str(fallback))
    assert runtime.require_tool('ncu', 'LAB_NCU') == str(fallback)
    monkeypatch.setenv('LAB_NCU', str(tmp_path / 'missing'))
    with pytest.raises(EnvironmentUnsupported, match='LAB_NCU'):
        runtime.require_tool('ncu', 'LAB_NCU')
    monkeypatch.delenv('LAB_NCU')
    monkeypatch.setenv('PATH', '')
    assert runtime.require_tool('ncu', 'LAB_NCU') == str(fallback)


@pytest.mark.parametrize('layout,libname', [('', 'lib64'), ('targets/x86_64-linux', 'lib')])
def test_cuda_development_paths_support_system_and_conda_layouts(tmp_path, layout, libname):
    root = tmp_path / layout
    (root / 'include').mkdir(parents=True)
    (root / 'include/cuda_runtime.h').touch()
    (root / libname).mkdir()
    (root / libname / 'libcudart.so').touch()
    includes, library = runtime.cuda_development_paths(tmp_path)
    assert root / 'include' in includes
    assert library == root / libname


def test_missing_cuda_development_files_are_reported(tmp_path):
    with pytest.raises(EnvironmentUnsupported, match='headers missing'):
        runtime.cuda_development_paths(tmp_path)
    (tmp_path / 'include').mkdir()
    (tmp_path / 'include/cuda_runtime.h').touch()
    with pytest.raises(EnvironmentUnsupported, match='libcudart.so missing'):
        runtime.cuda_development_paths(tmp_path)


def test_doctor_preserves_diagnostics_when_environment_is_missing(monkeypatch, capsys):
    from flash_attention_lab import cli, environment
    monkeypatch.setattr(environment, 'snapshot', lambda: {'cuda_available': False})
    def unavailable(backend):
        raise EnvironmentUnsupported('test: no GPU')
    monkeypatch.setattr(environment, 'require_environment', unavailable)
    assert cli.main(['doctor', '--stage', '0', '--json']) == 4
    report = json.loads(capsys.readouterr().out)
    assert report['cuda_available'] is False
    assert 'test: no GPU' in report['checks']['cuda']


def test_python_entrypoint_propagates_mask_and_exit_code():
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '3,7', 'LAB_GPU': '7'}
    process = subprocess.run([sys.executable, '-m', 'flash_attention_lab', 'python', '-c',
                              'import os,sys; print(os.environ["CUDA_VISIBLE_DEVICES"]); sys.exit(9)'],
                             env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 9
    assert process.stdout.strip() == '7'


def test_cuda_toolkit_mismatch_is_reported_before_build(tmp_path, monkeypatch):
    from flash_attention_lab import environment
    toolkit = tmp_path / 'toolkit'
    executable(toolkit / 'bin/nvcc')
    monkeypatch.setattr(environment, 'cuda_home', lambda: toolkit)
    monkeypatch.setattr(environment.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(environment.platform, 'machine', lambda: 'x86_64')
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                get_device_capability=lambda: (10, 0)), version=SimpleNamespace(cuda='12.8'))
    monkeypatch.setattr(environment, 'torch_module', lambda: fake_torch)
    monkeypatch.setattr(environment, 'command_output', lambda cmd: {'code': 0, 'stdout': 'release 13.0'})
    with pytest.raises(EnvironmentUnsupported, match='Toolkit 12.8'):
        environment.require_environment('cuda')


def test_profiler_sections_are_checked_against_selected_tool(monkeypatch):
    from flash_attention_lab import environment
    monkeypatch.setattr(environment, 'command_output', lambda cmd: {'code': 0, 'stdout': 'LaunchStats SpeedOfLight'})
    with pytest.raises(EnvironmentUnsupported, match='MemoryWorkloadAnalysis'):
        environment.require_ncu_sections('ncu')


def test_conda_activation_preserves_nounset_and_prefers_environment_bin(tmp_path):
    fake_conda = tmp_path / 'conda'
    fake_conda.write_text('''#!/bin/sh
if [ "$1" = shell.bash ] && [ "$2" = hook ]; then
    cat <<'HOOK'
conda() {
    export CONDA_PREFIX="$LAB_TEST_PREFIX"
    # Model a package hook that is valid without nounset.
    printf '%s' "$UNSET_PACKAGE_FLAG" >/dev/null
}
HOOK
else
    exit 1
fi
''')
    fake_conda.chmod(0o755)
    env = {**os.environ, 'LAB_CONDA_EXE': str(fake_conda), 'LAB_CONDA_ENV': 'test-env',
           'LAB_TEST_PREFIX': str(tmp_path / 'prefix')}
    env.pop('UNSET_PACKAGE_FLAG', None)
    script = runtime.ROOT / 'scripts/activate.sh'
    command = 'set -eu; before="$PWD"; source "$1"; [[ $- == *u* ]]; [[ "$before" == "$PWD" ]]; [[ "${PATH%%:*}" == "$CONDA_PREFIX/bin" ]]'
    result = subprocess.run(['bash', '-c', command, 'test', str(script)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_activation_reports_missing_environment_without_losing_shell_options(tmp_path):
    fake_conda = tmp_path / 'conda'
    fake_conda.write_text('#!/bin/sh\nprintf "conda() { return 1; }\\n"\n')
    fake_conda.chmod(0o755)
    env = {**os.environ, 'LAB_CONDA_EXE': str(fake_conda), 'LAB_CONDA_ENV': 'missing-env'}
    script = runtime.ROOT / 'scripts/activate.sh'
    command = 'set -u; source "$1"; code=$?; [[ $code == 4 ]]; [[ $- == *u* ]]'
    result = subprocess.run(['bash', '-c', command, 'test', str(script)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert 'ENV_UNSUPPORTED' in result.stderr
