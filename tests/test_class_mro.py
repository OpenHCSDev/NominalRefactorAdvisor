"""Source-backed MRO follows native Python without executing analyzed code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactClassFamilyIndexBuilder,
    MissingClassMember,
    OpenClassMember,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.class_mro import (
    ClassMroAuthority,
    CyclicHierarchy,
    DynamicClassDeclaration,
    InconsistentHierarchy,
    MissingDeclaration,
    UnresolvedBases,
    OpenClassMro,
)


def _index(source: str) -> CompactClassFamilyIndex:
    module = ParsedModule(
        path=Path("probe.py"),
        module_name="probe",
        module=ast.parse(source),
        source=source,
    )
    return CompactClassFamilyIndexBuilder(
        CompactModuleClassProjectionFamily.collect_modules((module,))
    ).build()


def _authority(source: str) -> ClassMroAuthority:
    return _index(source).mro_authority


@pytest.mark.parametrize(
    "source",
    (
        "class Left: pass\nclass Right: pass\nclass Leaf(Left, Right): pass\n",
        "class Left: pass\nclass Right: pass\nclass Leaf(Right, Left): pass\n",
        "class Root: pass\nclass Left(Root): pass\nclass Right(Root): pass\nclass Leaf(Left, Right): pass\n",
        "class Root: pass\nclass Left(Root): pass\nclass Right(Left): pass\nclass Leaf(Right, Root): pass\n",
        "class Leaf(object): pass\n",
        "import builtins as b\nclass Leaf(b.object): pass\n",
        "from abc import ABC as Abstract\nclass Root(Abstract): pass\nclass Leaf(Root): pass\n",
    ),
)
def test_static_mro_agrees_with_python(source: str) -> None:
    namespace = {"__name__": "probe"}
    exec(source, namespace)
    expected = tuple(
        f"probe.{owner.__name__}"
        for owner in namespace["Leaf"].__mro__
        if owner.__module__ == "probe"
    )
    authority = _authority(source)
    resolution = authority.resolve("probe.Leaf")
    assert resolution.mro_type is not None
    assert tuple(owner.symbol for owner in resolution.mro_type.declarations) == expected
    assert authority.resolve("probe.Leaf") is resolution


@pytest.mark.parametrize(
    "source,violation",
    (
        ("class Leaf(Unknown): pass\n", UnresolvedBases),
        (
            "object = factory()\nclass Leaf(object): pass\n",
            UnresolvedBases,
        ),
        (
            "from unknown import *\nclass Leaf(object): pass\n",
            UnresolvedBases,
        ),
        (
            "class Root: pass\nclass Leaf(Root[int]): pass\n",
            DynamicClassDeclaration,
        ),
        ("class Leaf(factory()): pass\n", DynamicClassDeclaration),
        (
            "class Leaf(metaclass=Meta): pass\n",
            DynamicClassDeclaration,
        ),
        ("@decorate\nclass Leaf: pass\n", DynamicClassDeclaration),
        (
            "class Root:\n def __init_subclass__(cls): raise RuntimeError\nclass Leaf(Root): pass\n",
            DynamicClassDeclaration,
        ),
        (
            "class Root: pass\nclass Leaf(Root, Root): pass\n",
            InconsistentHierarchy,
        ),
        (
            "class X: pass\nclass Y: pass\nclass A(X,Y): pass\nclass B(Y,X): pass\nclass Leaf(A,B): pass\n",
            InconsistentHierarchy,
        ),
        (
            "class Root(Leaf): pass\nclass Leaf(Root): pass\n",
            UnresolvedBases,
        ),
        (
            "class Root: pass\nclass Leaf(Root): pass\nclass Root: pass\n",
            UnresolvedBases,
        ),
    ),
)
def test_unproved_hierarchies_remain_open(
    source: str, violation: type[OpenClassMro]
) -> None:
    resolution = _authority(source).resolve("probe.Leaf")
    assert type(resolution) is violation
    assert resolution.mro_type is None


def test_missing_declaration_is_explicit() -> None:
    resolution = _authority("").resolve("probe.Missing")
    assert type(resolution) is MissingDeclaration


def test_cyclic_imported_hierarchy_is_explicit() -> None:
    resolution = _authority(
        "from probe import Leaf\n" "class Root(Leaf): pass\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert type(resolution) is CyclicHierarchy


@pytest.mark.parametrize("base", ("abc.ABC", "builtins.object"))
def test_standard_base_spelling_requires_an_actual_binding(base: str) -> None:
    resolution = _authority(f"class Leaf({base}): pass\n").resolve("probe.Leaf")
    assert type(resolution) is UnresolvedBases


def test_assigned_class_creation_hook_remains_open() -> None:
    resolution = _authority(
        "class Root:\n __init_subclass__ = hook\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert type(resolution) is DynamicClassDeclaration


def test_imported_base_does_not_resolve_to_overwritten_local_name() -> None:
    resolution = _authority(
        "class Root: pass\n" "from external import Root\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert type(resolution) is UnresolvedBases


def test_repository_bodies_are_not_executed() -> None:
    authority = _authority(
        "raise RuntimeError('must not execute module')\n"
        "class Root:\n raise RuntimeError('must not execute class')\n"
        "class Leaf(Root): pass\n"
    )
    assert authority.resolve("probe.Leaf").mro_type is not None


def test_changed_base_order_reproves_native_types() -> None:
    source = "class Left: pass\nclass Right: pass\nclass Leaf(Left, Right): pass\n"
    first = _authority(source).resolve("probe.Leaf").mro_type
    second = (
        _authority(source.replace("Leaf(Left, Right)", "Leaf(Right, Left)"))
        .resolve("probe.Leaf")
        .mro_type
    )
    assert first is not None and second is not None
    assert first is not second
    assert first.declarations[1].symbol == "probe.Left"
    assert second.declarations[1].symbol == "probe.Right"


TYPE_PARAMETER_SOURCES = (
    "from abc import ABC\nclass Base: pass\nclass Leaf[T](Base, ABC): pass\n",
    "class Leaf[T]: pass\n",
)


@pytest.mark.parametrize("source", TYPE_PARAMETER_SOURCES)
@pytest.mark.parametrize("policy", ("mro_authority", "declared_mro_authority"))
def test_type_parameters_end_the_order_as_creation_does(source: str, policy: str) -> None:
    """`class C[T]` ends its bases with `Generic`, unless it names `Protocol`."""

    namespace = {"__name__": "probe"}
    exec(source, namespace)
    expected = tuple(owner.__name__ for owner in namespace["Leaf"].__mro__)
    resolution = getattr(_index(source), policy).resolve("probe.Leaf")
    assert resolution.mro_type is not None
    assert tuple(entry.__name__.rpartition(".")[2] for entry in resolution.mro_type.__mro__) == expected


def test_a_protocol_with_type_parameters_gains_no_generic() -> None:
    """`Protocol` is generic itself, so creation appends nothing; its creation
    hook leaves the order unproved."""

    index = _index("from typing import Protocol\nclass Leaf[T](Protocol): pass\n")
    leaf = index.class_for("probe.Leaf")
    assert leaf is not None and leaf.implicit_base_names == ()
    assert type(index.mro_authority.resolve("probe.Leaf")) is UnresolvedBases


@pytest.mark.parametrize(
    "source,lookup",
    (
        (TYPE_PARAMETER_SOURCES[0], OpenClassMember),
        ("from abc import ABC\nclass Base: pass\nclass Leaf(Base, ABC): pass\n", MissingClassMember),
    ),
)
def test_a_name_generic_binds_is_read_at_generic(source: str, lookup: type) -> None:
    """No scanned class binds `__class_getitem__`; `Generic` does, so only a class
    with type parameters reads it, from a native whose code nothing declares."""

    assert type(_index(source).member_in_mro("probe.Leaf", "__class_getitem__")) is lookup
