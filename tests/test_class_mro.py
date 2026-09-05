"""Source-backed MRO follows native Python without executing analyzed code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndexBuilder,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.class_mro import (
    ClassMroAuthority,
    ClassMroViolation,
    OpenClassMro,
)


def _authority(source: str) -> ClassMroAuthority:
    module = ParsedModule(
        path=Path("probe.py"),
        module_name="probe",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    return (
        CompactClassFamilyIndexBuilder(
            CompactModuleClassProjectionFamily.collect_modules((module,))
        )
        .build()
        .mro_authority
    )


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
        ("class Leaf(Unknown): pass\n", ClassMroViolation.UNRESOLVED_BASES),
        (
            "object = factory()\nclass Leaf(object): pass\n",
            ClassMroViolation.UNRESOLVED_BASES,
        ),
        (
            "from unknown import *\nclass Leaf(object): pass\n",
            ClassMroViolation.UNRESOLVED_BASES,
        ),
        (
            "class Root: pass\nclass Leaf(Root[int]): pass\n",
            ClassMroViolation.DYNAMIC_CLASS_DECLARATION,
        ),
        ("class Leaf(factory()): pass\n", ClassMroViolation.DYNAMIC_CLASS_DECLARATION),
        (
            "class Leaf(metaclass=Meta): pass\n",
            ClassMroViolation.DYNAMIC_CLASS_DECLARATION,
        ),
        ("@decorate\nclass Leaf: pass\n", ClassMroViolation.DYNAMIC_CLASS_DECLARATION),
        (
            "class Root:\n def __init_subclass__(cls): raise RuntimeError\nclass Leaf(Root): pass\n",
            ClassMroViolation.DYNAMIC_CLASS_DECLARATION,
        ),
        (
            "class Root: pass\nclass Leaf(Root, Root): pass\n",
            ClassMroViolation.INCONSISTENT_HIERARCHY,
        ),
        (
            "class X: pass\nclass Y: pass\nclass A(X,Y): pass\nclass B(Y,X): pass\nclass Leaf(A,B): pass\n",
            ClassMroViolation.INCONSISTENT_HIERARCHY,
        ),
        (
            "class Root(Leaf): pass\nclass Leaf(Root): pass\n",
            ClassMroViolation.UNRESOLVED_BASES,
        ),
        (
            "class Root: pass\nclass Leaf(Root): pass\nclass Root: pass\n",
            ClassMroViolation.UNRESOLVED_BASES,
        ),
    ),
)
def test_unproved_hierarchies_remain_open(
    source: str, violation: ClassMroViolation
) -> None:
    resolution = _authority(source).resolve("probe.Leaf")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is violation
    assert resolution.mro_type is None


def test_missing_declaration_is_explicit() -> None:
    resolution = _authority("").resolve("probe.Missing")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is ClassMroViolation.MISSING_DECLARATION


def test_cyclic_imported_hierarchy_is_explicit() -> None:
    resolution = _authority(
        "from probe import Leaf\n" "class Root(Leaf): pass\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is ClassMroViolation.CYCLIC_HIERARCHY


@pytest.mark.parametrize("base", ("abc.ABC", "builtins.object"))
def test_standard_base_spelling_requires_an_actual_binding(base: str) -> None:
    resolution = _authority(f"class Leaf({base}): pass\n").resolve("probe.Leaf")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is ClassMroViolation.UNRESOLVED_BASES


def test_assigned_class_creation_hook_remains_open() -> None:
    resolution = _authority(
        "class Root:\n __init_subclass__ = hook\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is ClassMroViolation.DYNAMIC_CLASS_DECLARATION


def test_imported_base_does_not_resolve_to_overwritten_local_name() -> None:
    resolution = _authority(
        "class Root: pass\n" "from external import Root\n" "class Leaf(Root): pass\n"
    ).resolve("probe.Leaf")
    assert isinstance(resolution, OpenClassMro)
    assert resolution.violation is ClassMroViolation.UNRESOLVED_BASES


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
