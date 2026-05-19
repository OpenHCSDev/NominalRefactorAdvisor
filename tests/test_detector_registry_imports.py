"""Regression tests for detector registry discovery imports."""

from __future__ import annotations

import subprocess
import sys


def test_detector_registry_lazy_discovery_imports_private_collectors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from nominal_refactor_advisor.detectors import default_detectors; "
                "detectors = default_detectors(); "
                "assert detectors"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Failed to load registry module" not in result.stderr
