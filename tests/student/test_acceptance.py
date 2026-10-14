import subprocess
from flash_attention_lab.registry import ROOT


def test_student_contract(student_exercise):
    # No skip/xfail. Explicit acceptance is an error until the student implements it.
    # CLI isolates CUDA faults/deadlocks in a bounded child process.
    p = subprocess.run([str(ROOT / "lab"), "check", student_exercise, "--timeout", "300"],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=320)
    assert p.returncode == 0, p.stdout
