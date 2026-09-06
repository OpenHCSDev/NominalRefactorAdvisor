"""Collector execution has the same field lookup for generated and authored types."""

from dataclasses import replace

import pytest

from nominal_refactor_advisor.descriptor_algebra import ClassAliasProperty
from nominal_refactor_advisor.detectors import IssueDetector
from nominal_refactor_advisor.detectors._base import (
    CandidateCollectorScope,
    DetectorDeclaration,
    DerivedCandidateCollectorMixin,
    SourceModuleCollectorCandidateDetector,
)


@pytest.fixture
def declaration() -> DetectorDeclaration:
    return next(
        declaration
        for detector in IssueDetector.registered_detector_types()
        if (declaration := detector.resolved_detector_declaration()) is not None
        and issubclass(detector, DerivedCandidateCollectorMixin)
    )


@pytest.mark.parametrize(
    "base",
    DerivedCandidateCollectorMixin.registered_collector_base_types(),
)
def test_declared_collector_projection_obeys_native_override(
    base: type, declaration: DetectorDeclaration
) -> None:
    def original(*args):
        return ("original",)

    def override(*args):
        return ("override",)

    declaration = replace(
        declaration,
        options=replace(
            declaration.options, detector_base=base, candidate_collector=original
        ),
    )
    generated = type("Projection", (base,), declaration.runtime_namespace())
    overridden = type(
        "Override", (generated,), {"candidate_collector": staticmethod(override)}
    )
    inputs = (
        [object()]
        if base.collector_scope is CandidateCollectorScope.FLATTENED_MODULE
        else object()
    )
    assert tuple(generated()._candidate_items(inputs, None)) == ("original",)
    assert tuple(overridden()._candidate_items(inputs, None)) == ("override",)
    assert isinstance(vars(generated)["candidate_collector"], ClassAliasProperty)
    assert generated.candidate_collector is declaration.options.candidate_collector
    assert generated().candidate_collector is declaration.options.candidate_collector


def test_source_collector_projection_obeys_native_override(
    declaration: DetectorDeclaration,
) -> None:
    calls = []

    def original(*args):
        calls.append("original")
        return None

    def override(*args):
        calls.append("override")
        return None

    declaration = replace(
        declaration,
        options=replace(declaration.options, source_candidate_collector=original),
    )
    generated = type(
        "SourceProjection",
        (SourceModuleCollectorCandidateDetector,),
        declaration.runtime_namespace(),
    )
    overridden = type(
        "SourceOverride",
        (generated,),
        {
            "candidate_collector": staticmethod(lambda module: ()),
            "source_candidate_collector": staticmethod(override),
        },
    )
    assert generated()._findings_for_source(None, None, None) is None
    assert overridden()._findings_for_source(None, None, None) is None
    assert calls == ["original", "override"]
    assert generated.source_candidate_collector is original
    assert generated().source_candidate_collector is original


def test_generated_collector_requires_a_nonempty_declaration(
    declaration: DetectorDeclaration,
) -> None:
    declaration = replace(
        declaration, options=replace(declaration.options, candidate_collector=None)
    )
    with pytest.raises(TypeError, match="must own its candidate_collector"):
        type(
            "MissingCollector",
            (declaration.options.detector_base,),
            declaration.runtime_namespace(),
        )
