"""Runtime product safety consumes captured writes, including open receivers."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import CompactMutationKind
from nominal_refactor_advisor.product_flow_authority import (
    CompactProductFlowRepository,
    CompactProductRuntimeFailure,
    CompactProductRuntimeViolation,
)


def _module(tail: str) -> ParsedModule:
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n"
        "class Other:\n"
        "    pass\n" + tail
    )
    return ParsedModule(
        path=Path("product_mutation.py"),
        module_name="product_mutation",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def _repository(module: ParsedModule) -> CompactProductFlowRepository:
    return CompactProductFlowRepository.from_modules((module,))


@pytest.mark.parametrize(
    "tail",
    (
        "(Product if condition else Other).extra = replacement\n",
        "get_product().extra = replacement\n",
        "unknown()[index] = replacement\n",
        "del get_product().extra\n",
        "def mutate(value):\n    value.extra = replacement\n",
    ),
    ids=("conditional", "computed", "item", "delete", "parameter"),
)
def test_unproved_receiver_write_cannot_leave_a_product_authorized(tail: str) -> None:
    repository = _repository(_module(tail))
    assert "product_mutation.Product" not in repository.product_authorities_by_symbol
    assert all(
        failure.violation is CompactProductRuntimeViolation.UNRESOLVED_MUTATION_RECEIVER
        for failure in repository.product_runtime_failures_by_authority_symbol[
            "product_mutation.Product"
        ]
    )


def test_confirmed_class_write_is_distinct_from_unresolved_receiver_identity() -> None:
    repository = _repository(_module("Product.extra = replacement\n"))
    (failure,) = repository.product_runtime_failures_by_authority_symbol[
        "product_mutation.Product"
    ]
    assert (
        failure.violation
        is CompactProductRuntimeViolation.CLASS_REBINDING_OR_MEMBER_MUTATION
    )


@pytest.mark.parametrize(
    "tail",
    (
        "Other.extra = replacement\n",
        "def callback():\n    pass\ncallback.extra = replacement\n",
        "def local():\n    Product = replacement\n",
    ),
    ids=("other-class", "known-function", "local-rebinding"),
)
def test_closed_nonproduct_target_does_not_invalidate_a_product(tail: str) -> None:
    repository = _repository(_module(tail))
    assert "product_mutation.Product" in repository.product_authorities_by_symbol


def test_augmented_receiver_resolution_uses_the_actual_captured_class_object() -> None:
    module = _module(
        "Product.counter = 1\n"
        "Other.counter = 10\n"
        "alias = Product\n"
        "alias.counter += (alias := Other).counter\n"
    )
    namespace = {}
    exec(module.source, namespace)
    assert namespace["Product"].counter == 11
    assert namespace["Other"].counter == 10

    repository = _repository(module)
    context = repository.module_flow_contexts[module.module_name]
    write = next(
        mutation
        for mutation in context.flow.mutations
        if mutation.kind is CompactMutationKind.AUGMENTED_ASSIGNMENT
    )
    assert write.resolve(repository, context).candidate_symbols_within(
        frozenset(("product_mutation.Product", "product_mutation.Other"))
    ) == frozenset(("product_mutation.Product",))


def test_one_unknown_write_is_shared_across_product_queries_without_class_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(
        "".join(
            f"@dataclass\nclass Product{index}:\n    left: object\n    right: object\n"
            for index in range(40)
        )
        + "unknown().extra = replacement\n"
    )
    repository = _repository(module)

    def forbid_deep_hash(_failure: CompactProductRuntimeFailure) -> int:
        raise AssertionError(
            "Shared source evidence must not require deep-flow hashing"
        )

    monkeypatch.setattr(CompactProductRuntimeFailure, "__hash__", forbid_deep_hash)
    index = repository.product_runtime_failures_by_authority_symbol
    assert (
        index.authority_candidates is repository.declared_product_authorities_by_symbol
    )
    assert len(index.observations) == 1
    assert len(index) == 41
    assert "product_mutation.Other" not in index
    (observation,) = index.observations
    assert all(index[symbol] == (observation,) for symbol in index)
    assert all(index[symbol][0] is observation for symbol in index)
    context = repository.module_flow_contexts[module.module_name]
    assert observation.source_event is context.flow.mutations[-1]
    assert observation.context is context

    def forbid_materializing_diagnostics(_index, _symbol):
        raise AssertionError("Membership must not materialize per-class diagnostics")

    monkeypatch.setattr(type(index), "__getitem__", forbid_materializing_diagnostics)
    assert "product_mutation.Product" in index
    assert len(index) == 41
    assert set(index) == set(repository.declared_product_authorities_by_symbol)
