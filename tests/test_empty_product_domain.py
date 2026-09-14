"""Global product absence discharges product queries, not unrelated call queries."""

import builtins
from collections import Counter
import dataclasses

import pytest

from nominal_refactor_advisor.captured_reference import InitialNativeIsland
from nominal_refactor_advisor.carrier_collapse import CarrierCollapseBuilder
from nominal_refactor_advisor.carrier_expansion import DeclaredCarrierExpansionBuilder
from nominal_refactor_advisor.native_declarations import NativeLanguageFeature
from nominal_refactor_advisor.parameter_conveyor import (
    ClosedParameterConveyorComponentBuilder,
)
from nominal_refactor_advisor.product_flow import CompactCallableReferenceUse
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from test_product_call_resolution_reuse import SOURCE
from test_product_flow_authority import _module, _repository
from test_carrier_expansion import _closed_expansion_source
from test_parameter_conveyor import _base_source


class DataclassEntryRepository(SourceProductFlowRepository):
    """Supply the actual standard-library import premise, not a fabricated capture."""

    @staticmethod
    def source_entry(source):
        initial = InitialNativeIsland(
            (builtins, dataclasses, NativeLanguageFeature.module)
        )
        return ImportedSourceModuleEntryPremise.from_standard_source_loader(
            source, initial, initial.namespace_for_storage(vars(builtins))
        )


def track_target_queries(monkeypatch):
    observed = Counter()
    original = CompactCallableReferenceUse.resolve

    def resolve(use, resolver, context, **options):
        observed[id(context), id(use)] += 1
        return original(use, resolver, context, **options)

    monkeypatch.setattr(CompactCallableReferenceUse, "resolve", resolve)
    return observed


@pytest.mark.parametrize(
    "declaration",
    (
        "",
        "class Payload: pass\n",
        "from dataclasses import dataclass\n@dataclass\nclass Payload:\n"
        "    left: object\n    right: object\n"
        "    def __init__(self): pass\n",
    ),
)
def test_no_global_product_skips_target_queries_but_function_queries_still_work(
    monkeypatch, declaration
):
    repository = _repository(
        _module(
            "example",
            declaration + "def consume(value): return value\n"
            "def caller(value):\n    consume(value)\n    return consume(value)\n",
        )
    )
    assert not repository.declared_product_authorities_by_symbol
    observed = track_target_queries(monkeypatch)
    assert repository.resolved_product_constructions == ()
    assert not observed
    assert "function_call_resolutions" not in vars(repository)
    assert not repository.product_runtime_failures_by_authority_symbol
    assert len(repository.resolved_function_calls) == 2
    assert observed and set(observed.values()) == {1}
    assert repository.resolved_product_constructions == ()
    assert set(observed.values()) == {1}


@pytest.mark.parametrize("products_first", (False, True))
def test_available_global_product_retains_shared_original_target_resolution(
    monkeypatch, products_first
):
    repository = DataclassEntryRepository.from_modules((_module("example", SOURCE),))
    assert tuple(repository.product_authorities_by_symbol) == ("example.Payload",)
    observed = track_target_queries(monkeypatch)
    views = ("resolved_product_constructions", "function_call_resolutions")
    for name in views if products_first else reversed(views):
        getattr(repository, name)
    assert len(repository.resolved_product_constructions) == 2
    assert set(observed.values()) == {1}
    for product in repository.resolved_product_constructions:
        assert any(
            product.call is call.call and product.context is call.context
            for call in repository.function_call_resolutions
        )


def test_added_module_can_introduce_a_product_without_reusing_the_old_empty_result():
    first = _module("empty", "def call(value): return value\n")
    empty = DataclassEntryRepository.from_modules((first,))
    assert empty.resolved_product_constructions == ()
    expanded = DataclassEntryRepository.from_modules(
        (first, _module("example", SOURCE))
    )
    assert len(expanded.resolved_product_constructions) == 2
    assert empty.resolved_product_constructions == ()


def test_runtime_uncertainty_preserves_rejection_without_resolving_every_call(
    monkeypatch,
):
    # Compact observations supply no source activation. Their unbounded receivers
    # remain rejection evidence; this optimization must not manufacture admission.
    repository = _repository(_module("example", SOURCE))
    if not repository.product_runtime_failures_by_authority_symbol:
        pytest.skip("This interpreter defers the annotation writes")
    assert repository.declared_product_authorities_by_symbol
    assert not repository.product_authorities_by_symbol
    observed = track_target_queries(monkeypatch)
    assert repository.resolved_product_constructions == ()
    assert not observed


@pytest.mark.parametrize(
    "builder_type",
    (
        ClosedParameterConveyorComponentBuilder,
        DeclaredCarrierExpansionBuilder,
    ),
)
def test_empty_proven_query_is_inherited_without_forcing_call_resolution(
    monkeypatch,
    builder_type,
):
    modules = (
        _module(
            "empty",
            "def consume(value): return value\n"
            "def caller(value): return consume(value)\n",
        ),
    )
    builder = builder_type.from_modules(modules)
    assert type(builder) is builder_type
    assert builder_type.proven_components is CarrierCollapseBuilder.proven_components
    assert (
        builder_type.from_modules.__func__
        is CarrierCollapseBuilder.from_modules.__func__
    )
    assert (
        builder_type.from_projections.__func__
        is CarrierCollapseBuilder.from_projections.__func__
    )
    assert tuple(field.name for field in dataclasses.fields(builder)) == ("repository",)
    observed = track_target_queries(monkeypatch)
    assert builder.proven_components() == ()
    assert not observed
    assert len(builder.repository.resolved_function_calls) == 1
    assert observed and set(observed.values()) == {1}


@pytest.mark.parametrize(
    "builder_type,source_factory",
    (
        (ClosedParameterConveyorComponentBuilder, _base_source),
        (DeclaredCarrierExpansionBuilder, _closed_expansion_source),
    ),
)
def test_global_product_guard_retains_positive_proven_components(
    builder_type,
    source_factory,
):
    repository = DataclassEntryRepository.from_modules(
        (
            _module(
                "example",
                source_factory().replace("@dataclass(frozen=True)", "@dataclass"),
            ),
        )
    )
    builder = builder_type(repository)
    assert repository.product_authorities_by_symbol
    proven = builder.proven_components()
    assert proven
    assert proven == tuple(
        item for item in builder.assessed_components() if item.proof.is_proven
    )


def test_empty_proven_query_does_not_suppress_unproven_expansion_diagnostics():
    source = _closed_expansion_source().replace("@dataclass(frozen=True)\n", "")
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (_module("example", source),)
    )
    assert not builder.repository.product_authorities_by_symbol
    assert builder.proven_components() == ()
    assessed = builder.assessed_components()
    assert assessed
    assert all(not item.proof.is_proven for item in assessed)


def test_shared_builder_requires_a_concrete_assessment_family():
    with pytest.raises(TypeError, match="abstract"):
        CarrierCollapseBuilder(_repository(_module("empty", "")))
