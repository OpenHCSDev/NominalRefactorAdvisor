"""Cached findings are keyed on detection code, never on recipe synthesis."""

from types import CodeType

from nominal_refactor_advisor.analysis_cache import detector_implementation_module_names
from nominal_refactor_advisor.codemod_runtime import FindingRecipeEvaluator
from nominal_refactor_advisor.detectors._base import IssueDetector
from nominal_refactor_advisor.detectors._class_overlap import ClassOverlapDetectorABC
from nominal_refactor_advisor.implementation_identity import AnswerFreeDeclaration


def test_a_codemod_only_change_leaves_cached_findings_keyed_as_before() -> None:
    """Detectors inherit recipe synthesizers, and the key followed them into
    codemod.py, so the #201 split re-keyed every family (11-08). Synthesis runs
    on findings after they are cached and computes nothing the cache holds."""

    module_names = {
        name
        for detector in IssueDetector.registered_detector_types()
        for name in detector_implementation_module_names(detector)
    }

    assert "nominal_refactor_advisor.codemod" not in module_names
    assert ClassOverlapDetectorABC.__module__ in module_names
    assert "nominal_refactor_advisor.detectors._helpers" in module_names


def test_a_detector_base_that_inherits_a_synthesizer_keeps_its_detection_code() -> None:
    assert issubclass(ClassOverlapDetectorABC, FindingRecipeEvaluator)
    assert AnswerFreeDeclaration.contributes_to(ClassOverlapDetectorABC)
    assert not AnswerFreeDeclaration.contributes_to(FindingRecipeEvaluator)
    assert AnswerFreeDeclaration.contributes_to(IssueDetector)



def _loaded_names(member: object) -> set[str]:
    function = getattr(member, "__func__", member)
    pending = [function.__code__] if isinstance(getattr(function, "__code__", None), CodeType) else []
    names: set[str] = set()
    while pending:
        code = pending.pop()
        names.update(code.co_names)
        pending.extend(const for const in code.co_consts if isinstance(const, CodeType))
    return names


def test_no_detector_computes_its_findings_on_an_answer_free_owner() -> None:
    """The cache key skips answer-free owners, so a finding computed on one is keyed
    on nothing: FlatModulePackageDestinationDetector called its synthesizer's
    `destination_findings`, and a changed DestinationTable rule kept serving stale
    findings that moved Toad's thread_identity.py (11-08). A detector's findings
    entry points may name nothing that resolves to an answer-free owner."""

    IssueDetector.registered_detector_types()
    reached = sorted(
        (detector.__name__, owner.__name__, name)
        for detector in set(IssueDetector.__registry__.values())
        for entry in dir(detector)
        if entry.startswith(("_collect_findings", "_findings_from"))
        for name in _loaded_names(getattr(detector, entry))
        for owner in (next((owner for owner in detector.__mro__ if name in vars(owner)), None),)
        if owner is not None and not AnswerFreeDeclaration.contributes_to(owner)
    )

    assert reached == []
