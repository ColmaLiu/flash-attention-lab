from flash_attention_lab.runtime import configure_runtime
configure_runtime()

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--exercise", help="student exercise ID (explicit tests/student only)")


def pytest_generate_tests(metafunc):
    if "student_exercise" in metafunc.fixturenames:
        from flash_attention_lab.registry import EXERCISES
        chosen = metafunc.config.getoption("--exercise")
        if chosen and chosen not in EXERCISES: raise pytest.UsageError(f"unknown exercise {chosen}")
        ids = [chosen] if chosen else list(EXERCISES)
        metafunc.parametrize("student_exercise", ids)


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)
