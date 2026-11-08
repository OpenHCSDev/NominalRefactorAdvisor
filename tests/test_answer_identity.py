"""Cached findings are keyed on detection code, never on recipe synthesis."""

from nominal_refactor_advisor.analysis_cache import AnalysisEngineSignature
from nominal_refactor_advisor.codemod_runtime import FindingRecipeEvaluator
from nominal_refactor_advisor.detectors._base import IssueDetector
from nominal_refactor_advisor.detectors._class_overlap import ClassOverlapDetectorABC
from nominal_refactor_advisor.implementation_identity import AnswerFreeDeclaration


def test_a_codemod_only_change_leaves_cached_findings_keyed_as_before() -> None:
    """Detectors inherit recipe synthesizers, and the key followed them into
    codemod.py, so the #201 split re-keyed every family (11-08). Synthesis runs
    on findings after they are cached and computes nothing the cache holds."""

    module_names = AnalysisEngineSignature.module_names()

    assert "nominal_refactor_advisor.codemod" not in module_names
    assert ClassOverlapDetectorABC.__module__ in module_names
    assert "nominal_refactor_advisor.detectors._helpers" in module_names


def test_a_detector_base_that_inherits_a_synthesizer_keeps_its_detection_code() -> None:
    assert issubclass(ClassOverlapDetectorABC, FindingRecipeEvaluator)
    assert AnswerFreeDeclaration.contributes_to(ClassOverlapDetectorABC)
    assert not AnswerFreeDeclaration.contributes_to(FindingRecipeEvaluator)
    assert AnswerFreeDeclaration.contributes_to(IssueDetector)
