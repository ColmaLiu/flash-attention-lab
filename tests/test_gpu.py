"""End-to-end tests of ./lab on lab 00 (needs the selected B200)."""

import os
import shutil
import subprocess
import sys

import pytest

from labkit import ROOT

pytestmark = pytest.mark.gpu


def lab(*args, timeout=600):
    env = {**os.environ, "LAB_NO_PROGRESS": "1"}
    p = subprocess.run(
        [sys.executable, "-m", "labkit", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout + p.stderr


@pytest.fixture(scope="module")
def scratch_lab(tmp_path_factory):
    """A throwaway copy of the 00 student dir we can break on purpose."""
    target = ROOT / "students" / "00_transpose"
    backup = tmp_path_factory.mktemp("backup") / "kernel.cu"
    shutil.copy(target / "kernel.cu", backup)
    yield target / "kernel.cu"
    shutil.copy(backup, target / "kernel.cu")


def test_solution_passes():
    code, out = lab("_check", "00_transpose", "--solution")
    assert code == 0, out
    assert "PASS" in out


def test_skeleton_reports_device_todo():
    code, out = lab("_check", "00_transpose", "--step", "1", "--case", "tiny")
    assert code == 2, out
    assert "TODO(1)" in out


def test_wrong_answer_and_unwritten_output(scratch_lab):
    text = scratch_lab.read_text()
    scratch_lab.write_text(
        text.replace(
            "  LAB_TODO(1);",
            "  int c = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  int r = blockIdx.y * blockDim.y + threadIdx.y;\n"
            "  if (r < rows && c < cols && c != 0) y[c * rows + r] = x[r * cols + c];",
            1,
        )
    )
    code, out = lab("_check", "00_transpose", "--step", "1", "--case", "tiny")
    assert code == 3, out
    assert "never written" in out and "col=0" in out


def test_custom_shape():
    code, out = lab("_check", "00_transpose", "--solution", "--shape", "37,1000")
    assert code == 0, out
    assert "custom [37, 1000]" in out and "step 4" in out


def test_bench_reports_baseline():
    code, out = lab("_bench", "00_transpose", "--solution")
    assert code == 0, out
    assert "torch" in out and "GB/s" in out
