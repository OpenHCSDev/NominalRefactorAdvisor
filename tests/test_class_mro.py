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


def test_a_subclass_hook_runs_after_the_order_exists() -> None:
    """`__init_subclass__` runs once Python has the order, so it closes; a hook
    NRA reads that binds nothing leaves a missing name missing."""

    index = _index("class Root:\n def __init_subclass__(cls): raise RuntimeError\nclass Leaf(Root): pass\n")
    assert index.mro_authority.resolve("probe.Leaf").mro_type is not None
    assert type(index.member_in_mro("probe.Leaf", "absent")) is MissingClassMember


def test_an_assigned_class_creation_hook_keeps_lookups_open() -> None:
    """A hook bound by assignment runs code NRA does not read: the order is
    still Python's, but any name it may bind is unknown."""

    index = _index("class Root:\n __init_subclass__ = hook\n" "class Leaf(Root): pass\n")
    assert index.mro_authority.resolve("probe.Leaf").mro_type is not None
    assert type(index.member_in_mro("probe.Leaf", "absent")) is OpenClassMember


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


DECLARED_FAMILY = '''
from abc import ABC, ABCMeta
from typing import ClassVar, cast

from metaclass_registry import AutoRegisterMeta, RegistryConfig


class FamilyMeta(AutoRegisterMeta, ABCMeta):
    def __new__(mcs, name, bases, namespace, *, affix=None, discovery_package=None, **class_options):
        roots = {base.family_root for base in bases if hasattr(base, "family_root")}
        root = next(iter(roots), None)
        is_root = affix is not None or not roots
        if is_root:
            namespace["_family_affix"] = affix if affix is not None else namespace.get("_family_affix", "")
            registry = namespace["__registry__"] = {}
            namespace["declared_name"] = None
            config = namespace["__registry_config__"] = RegistryConfig(
                registry_dict=registry, key_attribute="declared_name", skip_if_no_key=True
            )
        else:
            config = root.__registry_config__
            namespace["declared_name"] = class_options.pop("declared_name", namespace.get("declared_name", name))
        cls: type[DeclaredFamily] = super().__new__(mcs, name, bases, namespace, registry_config=config, **class_options)
        if is_root:
            cls.family_root = cls
        else:
            for family_root in roots:
                cls.family_root = family_root
        return cast(FamilyMeta, cls)


class FamilyMember(ABC):
    declared_name: ClassVar[str]
    family_root: ClassVar[type]


class DeclaredFamily(FamilyMember, metaclass=FamilyMeta):
    _family_affix: ClassVar[str]

    def __init_subclass__(cls, *, affix=None, declared_name=None, discovery_package=None):
        super().__init_subclass__()


class TurnOwner(DeclaredFamily, affix="Owner"):
    def thread_name(self) -> str:
        return ""


class AgentOwner(TurnOwner, declared_name="agent"):
    pass


class LeafOwner(AgentOwner):
    pass
'''


@pytest.mark.parametrize(
    "symbol,order",
    (
        ("probe.TurnOwner", ("probe.TurnOwner", "probe.DeclaredFamily", "probe.FamilyMember")),
        ("probe.AgentOwner", ("probe.AgentOwner", "probe.TurnOwner", "probe.DeclaredFamily", "probe.FamilyMember")),
        (
            "probe.LeafOwner",
            ("probe.LeafOwner", "probe.AgentOwner", "probe.TurnOwner", "probe.DeclaredFamily", "probe.FamilyMember"),
        ),
    ),
    ids=("affix", "declared-name", "two-levels-down"),
)
def test_a_declared_family_member_has_a_closed_order(symbol: str, order: tuple[str, ...]) -> None:
    """`FamilyMeta.__new__` takes `affix=` and `declared_name=`, writes namespace
    entries and sets `family_root`, but hands its own `bases` to
    `super().__new__` and returns what it built; `AutoRegisterMeta` and
    `ABCMeta` are modelled. So Python's order is C3's over the declared bases,
    and the detectors read it."""

    mro_type = _authority(DECLARED_FAMILY).resolve(symbol).mro_type
    assert mro_type is not None
    assert tuple(declaration.symbol for declaration in mro_type.declarations) == order


def test_a_declared_family_member_answers_lookups_creation_cannot_bind() -> None:
    """Creation binds only names the family declares, registry names and the
    configured key attribute, so a name nothing binds is missing and a
    declared one is found."""

    index = _index(DECLARED_FAMILY)
    assert type(index.member_in_mro("probe.LeafOwner", "describe")) is MissingClassMember
    assert type(index.member_in_mro("probe.LeafOwner", "thread_name")) is not OpenClassMember
    assert type(index.member_in_mro("probe.LeafOwner", "declared_name")) is not MissingClassMember


@pytest.mark.parametrize(
    "body",
    (
        "        bases = (*bases, object)\n        return super().__new__(mcs, name, bases, namespace)\n",
        "        return super().__new__(mcs, name, tuple(reversed(bases)), namespace)\n",
    ),
    ids=("rebinds-bases", "hands-on-other-bases"),
)
def test_a_metaclass_that_changes_its_bases_leaves_the_order_open(body: str) -> None:
    source = (
        "class Meta(type):\n    def __new__(mcs, name, bases, namespace):\n" + body
        + "class Root(metaclass=Meta): pass\nclass Leaf(Root): pass\n"
    )
    resolution = _authority(source).resolve("probe.Leaf")
    assert type(resolution) is DynamicClassDeclaration
    assert resolution.mro_type is None


@pytest.mark.parametrize(
    "store,probed",
    (
        ("        cls.extra = 1\n", "extra"),
        ("        setattr(cls, name.lower(), 1)\n", "anything"),
    ),
    ids=("undeclared-name", "computed-name"),
)
def test_a_metaclass_binding_an_undeclared_name_keeps_that_lookup_open(store: str, probed: str) -> None:
    """The order is closed, since the bases are handed on; the name creation
    binds is no body's, so a lookup of it learns nothing."""

    source = (
        "class Meta(type):\n    def __new__(mcs, name, bases, namespace):\n"
        "        cls = super().__new__(mcs, name, bases, namespace)\n" + store + "        return cls\n"
        "class Root(metaclass=Meta): pass\nclass Leaf(Root): pass\n"
    )
    index = _index(source)
    assert index.mro_authority.resolve("probe.Leaf").mro_type is not None
    assert type(index.member_in_mro("probe.Leaf", probed)) is OpenClassMember
