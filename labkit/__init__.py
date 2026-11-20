"""labkit: build, check and time the kernels of flash-attention-lab."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LabError(Exception):
    """An error with a message meant for the learner (no traceback)."""

    exit_code = 1


class NotImplementedYet(LabError):
    """The student code reached LAB_TODO(n)."""

    exit_code = 2


class CorrectnessError(LabError):
    """Output differs from the mathematical reference."""

    exit_code = 3


class EnvironmentError_(LabError):
    """Missing tool, wrong GPU or toolchain."""

    exit_code = 4
