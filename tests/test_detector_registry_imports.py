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


def test_authority_boundary_relations_are_inherited_by_detector_families() -> None:
    from nominal_refactor_advisor.detectors import (
        IssueDetector,
        SemanticMirrorIssueDetector,
        SsotAuthorityBoundaryDetector,
    )

    semantic_mirror_ids = IssueDetector.semantic_mirror_detector_ids()
    ssot_authority_ids = IssueDetector.ssot_authority_detector_ids()

    assert issubclass(SsotAuthorityBoundaryDetector, IssueDetector)
    assert issubclass(SemanticMirrorIssueDetector, SsotAuthorityBoundaryDetector)
    assert "semantic_mirror_issue" not in semantic_mirror_ids
    assert "per_module_semantic_mirror_issue" not in semantic_mirror_ids
    assert (
        {
            "formal_boundary_external_string_registry_mirror",
            "semantic_mirror_without_descent",
        }
        <= semantic_mirror_ids
        <= ssot_authority_ids
    )
    assert "repeated_builder_calls" in ssot_authority_ids
    assert "repeated_builder_calls" not in semantic_mirror_ids
    assert not hasattr(IssueDetector, "genericity")
    assert not hasattr(IssueDetector, "semantic_mirror_authority_evidence_indices")
