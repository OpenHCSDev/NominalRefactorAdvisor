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


def test_semantic_mirror_detector_role_is_inherited_by_mirror_families() -> None:
    from nominal_refactor_advisor.detectors import IssueDetector

    role_ids = IssueDetector.semantic_mirror_detector_ids()

    assert "semantic_mirror_issue" not in role_ids
    assert "per_module_semantic_mirror_issue" not in role_ids
    assert {
        "formal_boundary_literal_registry_mirror",
        "formal_boundary_external_string_registry_mirror",
        "generic_role_case_table",
        "local_role_case_logic",
        "runtime_authority_branch_semantics",
        "runtime_semantic_branch_chain",
        "semantic_mirror_without_descent",
    } <= role_ids
