"""Original-source demand must not truncate later complete or ambiguous queries."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.detectors import (
    ClosedParameterConveyorDetector,
    DeclaredCarrierExpansionDetector,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository


def module(name, source):
    return ParsedModule(
        path=Path(name + ".py"), module_name=name,
        module=ast.parse(source), source=source,
    )


def test_rejected_authority_defers_successors_but_later_queries_remain_complete():
    first = module(
        "first",
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Product:\n    left: int\n    right: int\n"
        "Product.left = 1\n",
    )
    last = module("last", "def late(): pass\n")
    repository = SourceProductFlowRepository.from_modules((first, last))

    assert repository.product_authorities_by_symbol == {}
    assert set(repository._source_projections) == {id(first)}
    assert repository.flow_context_for_symbol("last.late").owner_symbol == "last.late"
    assert set(repository._source_projections) == {id(first), id(last)}
    assert tuple(repository.iter_flow_contexts()) == repository.flow_contexts
    assert type(repository.product_projections) is tuple
    assert tuple(projection.module_name for projection in repository.product_projections) == (
        "first", "last",
    )


def test_later_duplicate_module_and_scope_keep_source_lookup_open():
    first = module("shared", "def same(): pass\n")
    last = module("shared", "def same(): return 1\n")
    repository = SourceProductFlowRepository.from_modules((first, last))

    assert repository.product_projection_for_module("shared") is None
    assert repository.flow_context_for_symbol("shared.same") is None
    assert set(repository._source_projections) == {id(first), id(last)}
    assert len(tuple(repository.iter_product_projections())) == 2
    assert repository.module_flow_context_for_name("shared") is None


@pytest.mark.parametrize(
    "detector_type", (ClosedParameterConveyorDetector, DeclaredCarrierExpansionDetector)
)
def test_source_carrier_preparation_retains_its_candidate_proof(
    monkeypatch, detector_type,
):
    calls = []
    original = detector_type._proven_components

    def observed(self, repository):
        calls.append(repository)
        return original(self, repository)

    monkeypatch.setattr(detector_type, "_proven_components", observed)
    prepared = detector_type().prepare_analysis(
        (module("empty", "def ordinary(): pass\n"),), DetectorConfig(),
    )
    assert len(calls) == 1
    assert type(calls[0]) is SourceProductFlowRepository
    assert prepared.findings() == []
    assert prepared.findings() == []
    assert len(calls) == 1
