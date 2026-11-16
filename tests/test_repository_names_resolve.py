"""Every name the package reads is defined where it is read."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nominal_refactor_advisor

PACKAGE = Path(nominal_refactor_advisor.__file__).resolve().parent


def test_repository_has_no_undefined_names() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821", "--output-format", "concise", str(PACKAGE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
