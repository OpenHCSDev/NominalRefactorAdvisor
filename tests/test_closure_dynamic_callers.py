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
    CompactFunctionCallIdentity,
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
    """Whether every call that can reach `symbol` is among its resolved calls."""

    return repository.callable_component_authority_proof(
        {symbol: frozenset()},
        frozenset(
            CompactFunctionCallIdentity.from_resolution(call)
            for call in repository.incoming_calls_for(symbol)
        ),
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


def test_a_nested_function_is_not_fetched_by_a_computed_getattr():
    repository = _repository(
        "def _outer():\n    def _inner():\n        return 1\n    return _inner()\n\n\n"
        "def _run(thing, name):\n    return getattr(thing, name)()\n"
    )
    assert _closed(repository, "pkg.calls._outer._inner")


def test_an_annotated_parameter_bounds_a_computed_getattr_to_its_class_family():
    repository = _repository(
        CLASSES
        + "class _Unrelated:\n    def _label(self):\n        return 'unrelated'\n\n\n"
        + "def _run(thing: _Unrelated | None, name):\n    return getattr(thing, name)()\n"
    )
    assert _closed(repository, "pkg.calls._Kind._other")
    assert not _closed(repository, "pkg.calls._Unrelated._label")


def test_a_classmethod_declared_to_return_self_types_its_call():
    repository = _repository(
        "from typing import Self\n\n\n"
        "class _Built:\n"
        "    @classmethod\n    def _make(cls) -> Self:\n        return cls()\n\n"
        "    def _label(self):\n        return 'built'\n\n\n"
        "class _Other:\n    def _label(self):\n        return 'other'\n\n\n"
        "def _run():\n    return _Built._make()._label()\n"
    )
    context = repository.flow_contexts_by_owner_symbol["pkg.calls._run"]
    resolved = repository.resolve_function_call(context, context.flow.calls[-1])
    assert resolved.resolved_call is not None
    assert resolved.resolved_call.callee.identity.symbol == "pkg.calls._Built._label"
    assert _closed(repository, "pkg.calls._Other._label")


def test_a_closed_world_project_closes_public_functions_and_records_its_premise(
    tmp_path,
):
    source = "def label():\n    return 'label'\n\n\ndef run():\n    return label()\n"
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "calls.py").write_text(source)
    module = ParsedModule(
        path=package / "calls.py",
        module_name="pkg.calls",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    repository = CompactProductFlowRepository(
        product_projections=(compact_product_flow_projection(module),),
        class_projections=CompactModuleClassProjectionFamily.collect_modules((module,)),
    )

    def proof():
        incoming = repository.incoming_calls_for("pkg.calls.label")
        return repository.callable_component_authority_proof(
            {"pkg.calls.label": frozenset()},
            frozenset(
                CompactFunctionCallIdentity.from_resolution(call) for call in incoming
            ),
        )

    assert proof().open_boundary_symbols == ("pkg.calls.label",)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.nominal-refactor-advisor]\nclosed-world = true\n"
    )
    closed = proof()
    assert closed.is_closed
    premise = closed.closed_world_premises["pkg.calls.label"]
    assert premise.pyproject == str(tmp_path / "pyproject.toml")
    assert len(premise.digest) == 64


def test_a_foreign_receiver_fetches_only_its_scanned_subclasses_methods():
    repository = _repository(
        "import argparse\n\n\n"
        + CLASSES
        + "class _Arguments(argparse.Namespace):\n"
        + "    def _label(self):\n        return 'arguments'\n\n\n"
        + "def _run(args: argparse.Namespace, name):\n    return vars(args)[name]()\n"
    )
    assert _closed(repository, "pkg.calls._Kind._other")
    assert not _closed(repository, "pkg.calls._Arguments._label")


def test_a_local_from_a_constructor_bounds_a_fetch_to_that_class_family():
    repository = _repository(
        CLASSES
        + "class _Unrelated:\n    def _label(self):\n        return 'unrelated'\n\n\n"
        + "def _run(name):\n    thing = _Unrelated()\n    return getattr(thing, name)()\n"
    )
    assert _closed(repository, "pkg.calls._Kind._other")
    assert not _closed(repository, "pkg.calls._Unrelated._label")


def test_an_annotated_member_of_self_bounds_a_fetch_to_its_class_family():
    repository = _repository(
        CLASSES
        + "class _Unrelated:\n    def _label(self):\n        return 'unrelated'\n\n\n"
        + "class _Holder:\n    held: _Unrelated\n\n"
        + "    def _run(self, name):\n        return getattr(self.held, name)()\n"
    )
    assert _closed(repository, "pkg.calls._Kind._other")
    assert not _closed(repository, "pkg.calls._Unrelated._label")
