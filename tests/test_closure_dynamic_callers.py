"""A callee is closed only when every call that can reach it is known (#65).

A chained call (`f().m()`) is a caller of every method named `m`; an access on
an unproved receiver is bounded by its attribute name, except where a computed
fetch or a `__getattr__` can return a function under any name.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.product_flow import compact_product_flow_projection
from nominal_refactor_advisor.product_flow_authority import (
    CompactProductFlowRepository,
)


def _repository(source: str) -> CompactProductFlowRepository:
    path = Path("pkg", "calls.py")
    module = ParsedModule(
        path=path,
        module_name="pkg.calls",
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )
    return CompactProductFlowRepository(
        product_projections=(compact_product_flow_projection(module),),
        class_projections=CompactModuleClassProjectionFamily.collect_modules((module,)),
    )


def _closed(repository: CompactProductFlowRepository, symbol: str) -> bool:
    return repository.callable_component_authority_proof(
        {symbol: frozenset()}, frozenset()
    ).is_closed


CLASSES = (
    "class _Kind:\n"
    "    def _label(self):\n        return 'kind'\n\n"
    "    def _other(self):\n        return 'other'\n\n\n"
    "def _make():\n    return _Kind()\n\n\n"
)


def test_a_chained_call_is_a_caller_of_every_method_of_its_name():
    repository = _repository(CLASSES + "def _run():\n    return _make()._label()\n")
    proof = repository.callable_component_authority_proof(
        {"pkg.calls._Kind._label": frozenset()}, frozenset()
    )
    assert not proof.is_closed
    assert proof.unresolved_consumer_symbols == ("pkg.calls._Kind._label",)
    # The same call cannot reach a method of another name.
    assert _closed(repository, "pkg.calls._Kind._other")


def test_an_unproved_receiver_reaches_only_its_attribute_name():
    repository = _repository(CLASSES + "def _run(thing):\n    return thing._label()\n")
    assert not _closed(repository, "pkg.calls._Kind._label")
    assert _closed(repository, "pkg.calls._Kind._other")


def test_a_computed_getattr_on_an_unproved_receiver_keeps_every_method_open():
    repository = _repository(
        CLASSES + "def _run(thing, name):\n    return getattr(thing, name)()\n"
    )
    proof = repository.callable_component_authority_proof(
        {"pkg.calls._Kind._other": frozenset()}, frozenset()
    )
    assert proof.dynamically_fetched_symbols == ("pkg.calls._Kind._other",)
    assert not proof.is_closed


def test_a_literal_getattr_fetches_only_its_name():
    repository = _repository(
        CLASSES + "def _run(thing):\n    return getattr(thing, '_label')()\n"
    )
    assert not _closed(repository, "pkg.calls._Kind._label")
    assert _closed(repository, "pkg.calls._Kind._other")


def test_a_getattr_hook_in_the_scan_keeps_named_access_unbounded():
    repository = _repository(
        CLASSES
        + "class _Proxy:\n    def __getattr__(self, name):\n        return name\n\n\n"
        + "def _run(thing):\n    return thing._label()\n"
    )
    assert not _closed(repository, "pkg.calls._Kind._other")
