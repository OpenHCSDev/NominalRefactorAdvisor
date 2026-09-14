"""Eligibility may finish early; complete rejection diagnostics remain available."""

import gc
import pickle
from weakref import ref

import pytest

from nominal_refactor_advisor.product_flow_authority import (
    CompactProductRuntimeFailure,
    CompactProductRuntimeViolation,
    ProductFlowRepository,
    ResolvedCompactClassTarget,
)
from test_empty_product_domain import DataclassEntryRepository
from test_product_flow_authority import _module, _repository
from test_product_call_resolution_reuse import SOURCE


def products():
    return _repository(
        _module(
            "example",
            "from dataclasses import dataclass\n"
            + "".join(
                f"@dataclass\nclass Product{index}:\n    left: object\n    right: object\n"
                for index in range(3)
            )
            + "unknown.first = 1\nunknown.second = 2\n",
        )
    )


@pytest.mark.parametrize("diagnostics_first", (False, True))
def test_early_rejection_preserves_the_complete_diagnostic_query(
    monkeypatch, diagnostics_first
):
    repository = products()
    visited = []
    original = ProductFlowRepository.iter_product_runtime_failures

    def observe(self):
        for failure in original(self):
            visited.append(failure)
            yield failure

    monkeypatch.setattr(ProductFlowRepository, "iter_product_runtime_failures", observe)
    if diagnostics_first:
        diagnostics = repository.product_runtime_failures_by_authority_symbol
        assert len(diagnostics.observations) >= 2
        visited.clear()
    result = repository.product_authorities_by_symbol
    assert not result
    assert len(visited) == 1
    first = visited[0]
    if not diagnostics_first:
        assert "product_runtime_failures_by_authority_symbol" not in vars(repository)
    diagnostics = repository.product_runtime_failures_by_authority_symbol
    assert len(diagnostics.observations) >= 2
    assert diagnostics.observations[0].source_event is first.source_event
    assert diagnostics.observations[0].context is first.context
    assert set(diagnostics) == set(repository.declared_product_authorities_by_symbol)
    assert result == {
        name: candidate
        for name, candidate in repository.declared_product_authorities_by_symbol.items()
        if name not in diagnostics
    }
    assert repository.product_authorities_by_symbol is result


def test_one_rejected_candidate_does_not_skip_other_candidates_or_late_rejections(
    monkeypatch,
):
    repository = products()
    candidates = tuple(repository.declared_product_authorities_by_symbol)
    context = repository.module_flow_contexts["example"]
    event = context.flow.mutations[-1]
    observed = []

    def selective_failures(self):
        for name in candidates[:2]:
            observed.append(name)
            yield CompactProductRuntimeFailure(
                context,
                event,
                ResolvedCompactClassTarget(self.class_index.class_for(name)),
                CompactProductRuntimeViolation.CLASS_REBINDING_OR_MEMBER_MUTATION,
            )

    monkeypatch.setattr(
        ProductFlowRepository, "iter_product_runtime_failures", selective_failures
    )
    assert tuple(repository.product_authorities_by_symbol) == candidates[2:]
    assert observed == list(candidates[:2])
    assert len(repository.product_runtime_failures_by_authority_symbol) == 2


def test_empty_candidate_query_does_not_create_a_failure_scan(monkeypatch):
    repository = _repository(_module("empty", "def call(value): return value\n"))

    def forbidden(self):
        raise AssertionError("No candidates require no runtime exclusion work")

    monkeypatch.setattr(
        ProductFlowRepository, "iter_product_runtime_failures", forbidden
    )
    assert not repository.product_authorities_by_symbol


def test_admitted_product_and_call_results_remain_available_in_either_query_order():
    for diagnostics_first in (False, True):
        repository = DataclassEntryRepository.from_modules(
            (_module("example", SOURCE),)
        )
        if diagnostics_first:
            assert not repository.product_runtime_failures_by_authority_symbol
        assert tuple(repository.product_authorities_by_symbol) == ("example.Payload",)
        assert len(repository.resolved_product_constructions) == 2
        assert not repository.product_runtime_failures_by_authority_symbol


def test_no_suspended_generator_is_retained_or_serialized_after_early_exit():
    repository = products()
    assert not repository.product_authorities_by_symbol
    restored = pickle.loads(pickle.dumps(repository))
    assert not restored.product_authorities_by_symbol
    assert len(restored.product_runtime_failures_by_authority_symbol.observations) >= 2
    weak = ref(repository)
    del repository
    gc.collect()
    assert weak() is None
