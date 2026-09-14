"""Construction and invocation views share the original resolved call target."""

from collections import Counter
import gc
from weakref import ref

import pytest

from nominal_refactor_advisor.product_flow import CompactCallableReferenceUse
from test_product_flow_authority import _module, _source_repository

SOURCE = (
    "from dataclasses import dataclass\n"
    "@dataclass\nclass Payload:\n    value: object\n    other: object\n"
    "def consume(value): return value\n"
    "def caller(value):\n"
    "    first = Payload(value=value, other=value)\n"
    "    second = Payload(value=value, other=value)\n"
    "    return consume(first)\n"
)


@pytest.mark.parametrize("products_first", (False, True))
def test_both_views_resolve_each_original_target_once(monkeypatch, products_first):
    repository = _source_repository(_module("example", SOURCE))
    observed = Counter()
    original = CompactCallableReferenceUse.resolve

    def resolve(use, resolver, context, **options):
        observed[id(context), id(use)] += 1
        return original(use, resolver, context, **options)

    monkeypatch.setattr(CompactCallableReferenceUse, "resolve", resolve)
    views = ("resolved_product_constructions", "function_call_resolutions")
    for name in views if products_first else reversed(views):
        getattr(repository, name)
    products = repository.resolved_product_constructions
    assert len(products) == 2
    assert products[0].call is not products[1].call
    for resolution in repository.function_call_resolutions:
        assert observed[id(resolution.context), id(resolution.call.target_use)] == 1
    assert all(
        any(
            p.context is c.context and p.call is c.call
            for c in repository.function_call_resolutions
        )
        for p in products
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "def Payload(value): return value\n",
        "Payload = consume\n",
        "if uncertain:\n    Payload = consume\n",
    ),
)
def test_fresh_repository_edit_reclassifies_the_same_spelled_call(replacement):
    original = _source_repository(_module("example", SOURCE))
    assert len(original.resolved_product_constructions) == 2
    source = SOURCE.replace("def caller(value):", replacement + "def caller(value):")
    edited = _source_repository(_module("example", source))
    assert edited.resolved_product_constructions == ()
    assert edited.function_call_resolutions is not original.function_call_resolutions
    assert len(original.resolved_product_constructions) == 2


def test_shared_global_name_ambiguity_is_not_resolved_by_a_cached_local_view():
    first = _module("example", SOURCE)
    second = _module("example", SOURCE.replace("class Payload:", "class Other:"))
    alone = _source_repository(first)
    assert len(alone.resolved_product_constructions) == 2
    combined = _source_repository(first, second)
    assert combined.resolved_product_constructions == ()


def test_derived_call_views_do_not_outlive_the_repository():
    repository = _source_repository(_module("example", SOURCE))
    assert repository.resolved_product_constructions
    weak = ref(repository)
    del repository
    gc.collect()
    assert weak() is None


@pytest.mark.parametrize("keywords", (True, False))
def test_derived_and_direct_projections_preserve_original_call_and_authority(keywords):
    source = (
        SOURCE
        if keywords
        else SOURCE.replace("value=value, other=value", "value, value")
    )
    repository = _source_repository(_module("example", source))
    derived = repository.resolved_product_constructions
    direct = tuple(
        result
        for context in repository.flow_contexts
        for call in context.flow.calls
        if (result := repository.resolve_product_construction(context, call))
        is not None
    )
    assert len(direct) == len(derived) == (2 if keywords else 0)
    for original, projection in zip(direct, derived, strict=True):
        assert projection.context is original.context
        assert projection.call is original.call
        assert projection.authority is original.authority
        assert projection.construction == original.construction
