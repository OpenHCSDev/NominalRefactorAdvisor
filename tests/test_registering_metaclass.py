"""A class's metaclass registers by what it does, never by its name (#71)."""

from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjectionFamily,
    build_class_family_index,
    build_compact_class_family_index,
)

SOURCE = (
    "from abc import ABCMeta\n"
    "from metaclass_registry import AutoRegisterMeta as Registry\n\n\n"
    "class Aliased(metaclass=Registry):\n"
    "    __registry_key__ = 'name'\n\n\n"
    "class FamilyMeta(Registry):\n    pass\n\n\n"
    "class Family(metaclass=FamilyMeta):\n    pass\n\n\n"
    "class HandRolledMeta(type):\n"
    "    def __init__(cls, name, bases, namespace):\n"
    "        super().__init__(name, bases, namespace)\n"
    "        cls.__registry__[namespace.get('key')] = cls\n\n\n"
    "class HandRolled(metaclass=HandRolledMeta):\n    pass\n\n\n"
    "def _store(table, key, value):\n    table[key] = value\n\n\n"
    "class ThroughHelperMeta(type):\n"
    "    def __new__(mcs, name, bases, namespace):\n"
    "        created: type = super().__new__(mcs, name, bases, namespace)\n"
    "        mcs._register(created, name)\n"
    "        return created\n\n"
    "    @staticmethod\n"
    "    def _register(cls, key):\n"
    "        _store(TABLE, key, value=cls)\n\n\n"
    "class ThroughHelper(metaclass=ThroughHelperMeta):\n    pass\n\n\n"
    "class ReadsConfigMeta(type):\n"
    "    def __new__(mcs, name, bases, namespace):\n"
    "        created = super().__new__(mcs, name, bases, namespace)\n"
    "        TABLE[created.__registry_key__] = name\n"
    "        return created\n\n\n"
    "class ReadsConfig(metaclass=ReadsConfigMeta):\n    pass\n\n\n"
    "class NotAutoRegisterMeta(ABCMeta):\n    pass\n\n\n"
    "class NamedLikeIt(metaclass=NotAutoRegisterMeta):\n    pass\n\n\n"
    "class Plain:\n    pass\n"
)

EXPECTED = {
    "Aliased": True,
    "Family": True,
    "HandRolled": True,
    "ThroughHelper": True,
    "ReadsConfig": False,
    "NamedLikeIt": False,
    "Plain": False,
}


def _module() -> ParsedModule:
    path = Path("pkg/registry_family.py")
    return ParsedModule(
        path=path,
        module_name="pkg.registry_family",
        module=ast.parse(SOURCE, filename=str(path)),
        source=SOURCE,
    )


def _registering(index: CompactClassFamilyIndex) -> dict[str, bool]:
    return {
        name: index.classes_by_symbol[f"pkg.registry_family.{name}"].declares_autoregister_meta
        for name in EXPECTED
    }


def test_the_compact_index_resolves_each_metaclass_and_reads_it() -> None:
    projections = CompactModuleClassProjectionFamily.collect_modules((_module(),))
    index = build_compact_class_family_index(projections)

    assert _registering(index) == EXPECTED
    assert index.classes_by_symbol["pkg.registry_family.Family"].is_registration_authority


def test_a_subclass_of_a_registering_family_is_an_authority_by_inheritance() -> None:
    """Python creates a subclass with its base's metaclass; a name is no evidence (#139)."""

    source = (
        "from metaclass_registry import AutoRegisterMeta\n\n\n"
        "class Root(metaclass=AutoRegisterMeta):\n    pass\n\n\n"
        "class Member(Root):\n    pass\n\n\n"
        "class Grandchild(Member):\n    pass\n\n\n"
        "class StringRegistryConstant:\n    value: str\n"
    )
    path = Path("pkg/inherited.py")
    module = ParsedModule(
        path=path,
        module_name="pkg.inherited",
        module=ast.parse(source, filename=str(path)),
        source=source,
    )
    index = build_compact_class_family_index(
        CompactModuleClassProjectionFamily.collect_modules((module,))
    )
    classes = index.classes_by_symbol
    assert {
        name: classes[f"pkg.inherited.{name}"].is_registration_authority
        for name in ("Root", "Member", "Grandchild", "StringRegistryConstant")
    } == {"Root": True, "Member": True, "Grandchild": True, "StringRegistryConstant": False}
    # Only the root declares the metaclass; the subclasses inherit it.
    assert classes["pkg.inherited.Root"].declares_autoregister_meta
    assert not classes["pkg.inherited.Member"].declares_autoregister_meta


def test_a_keyed_family_key_type_is_read_from_a_registry_family_base_only() -> None:
    """`Family[Key]` names the key only when `Family` is a registry family, whatever its name."""

    source = (
        "from enum import Enum\n"
        "from typing import Generic, TypeVar\n\n"
        "K = TypeVar('K')\n\n\n"
        "class StoresMeta(type):\n"
        "    def __init__(cls, name, bases, namespace):\n"
        "        super().__init__(name, bases, namespace)\n"
        "        cls.__registry__[namespace.get('key')] = cls\n\n\n"
        "class Family(Generic[K], metaclass=StoresMeta):\n    pass\n\n\n"
        "class KeyedNominalFamily(Generic[K]):\n    pass\n\n\n"
        "class Mode(Enum):\n    ALPHA = 1\n\n\n"
        "class Keyed(Family[Mode]):\n    registry_key_attr = 'mode'\n\n\n"
        "class NamedLikeOne(KeyedNominalFamily[Mode]):\n    registry_key_attr = 'mode'\n"
    )
    path = Path("pkg/keyed.py")
    module = ParsedModule(
        path=path,
        module_name="pkg.keyed",
        module=ast.parse(source, filename=str(path)),
        source=source,
    )
    classes = build_compact_class_family_index(
        CompactModuleClassProjectionFamily.collect_modules((module,))
    ).classes_by_symbol
    assert {
        name: classes[f"pkg.keyed.{name}"].keyed_family_key_type_name
        for name in ("Keyed", "NamedLikeOne", "Family")
    } == {"Keyed": "Mode", "NamedLikeOne": None, "Family": None}


def test_the_source_index_gives_the_same_answer() -> None:
    index = build_class_family_index([_module()])

    assert _registering(index) == EXPECTED


def test_a_star_import_that_does_not_export_the_name_keeps_its_binding() -> None:
    """`detectors/_base.py` imports `AutoRegisterMeta`, then star-imports a module."""

    def module(name: str, source: str) -> ParsedModule:
        path = Path(*name.split(".")).with_suffix(".py")
        return ParsedModule(
            path=path,
            module_name=name,
            module=ast.parse(source, filename=str(path)),
            source=source,
        )

    support = module("pkg.support", "__all__ = ('helper',)\n\n\ndef helper():\n    return 1\n")
    family = module(
        "pkg.family",
        "from metaclass_registry import AutoRegisterMeta\n"
        "from pkg.support import *\n\n\n"
        "class Detector(metaclass=AutoRegisterMeta):\n    pass\n",
    )
    projections = CompactModuleClassProjectionFamily.collect_modules((support, family))

    assert build_compact_class_family_index(projections).classes_by_symbol[
        "pkg.family.Detector"
    ].declares_autoregister_meta
    assert build_class_family_index([support, family]).classes_by_symbol[
        "pkg.family.Detector"
    ].declares_autoregister_meta
