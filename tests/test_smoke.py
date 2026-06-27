"""Smoke tests for the project scaffolding.

These intentionally avoid any ML / runtime dependencies (torch is not
yet a project dependency). They exist so the toolchain (`just test`,
pytest) has something concrete to run while the experiment code is still
being built.
"""

import importlib
import sys


def test_exp_package_imports() -> None:
    """The flat `exp` package is importable from the repo root."""
    module = importlib.import_module("exp")
    assert module is not None


def test_running_supported_python() -> None:
    """The interpreter satisfies the project's requires-python (>=3.13)."""
    assert sys.version_info >= (3, 13)
