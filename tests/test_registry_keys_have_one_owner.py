"""A registry key names one class, in NRA's own families and in requested ones."""

import ast
import importlib
import pkgutil

import pytest
from metaclass_registry import AutoRegisterMeta

import nominal_refactor_advisor
from nominal_refactor_advisor.detectors._base import (
    ClassLineWitnessCandidate,
    LineWitnessCandidate,
    QualnameLineWitnessCandidate,
    WitnessNameAliasMixin,
)
from nominal_refactor_advisor.registry_identity import AutoRegisterClassAuthority


# The slots on type itself; a metaclass descriptor or __getattribute__ cannot
# intercept them.
_CLASS_STORAGE = type.__dict__["__dict__"]
_CLASS_MODULE = type.__dict__["__module__"]


def _registered_families() -> dict[type, dict[object, type]]:
    for module in pkgutil.walk_packages(
        nominal_refactor_advisor.__path__, f"{nominal_refactor_advisor.__name__}."
    ):
        importlib.import_module(module.name)
    families: dict[type, dict[object, type]] = {}
    seen: set[int] = set()
    pending: list[type] = [object]
    while pending:
        cls = pending.pop()
        # Other tests declare unhashable classes and metaclasses that refuse
        # attribute reads, so the walk keys on identity and reads native storage.
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        pending.extend(type.__subclasses__(cls))
        registry = _CLASS_STORAGE.__get__(cls).get("__registry__")
        if (
            issubclass(type(cls), AutoRegisterMeta)
            and registry is not None
            and _CLASS_MODULE.__get__(cls).startswith(nominal_refactor_advisor.__name__)
        ):
            families[cls] = dict(dict.items(registry))
    return families


def test_every_registered_class_holds_exactly_one_key_of_its_family():
    families = _registered_families()
    assert families
    for root, registry in families.items():
        members = tuple(registry.values())
        assert len(set(members)) == len(members), root


def test_witness_name_mixins_supply_a_name_and_register_nothing():
    assert not isinstance(WitnessNameAliasMixin, AutoRegisterMeta)
    registry = LineWitnessCandidate.__registry__
    keys = {
        witness: [key for key, member in registry.items() if member is witness]
        for witness in (ClassLineWitnessCandidate, QualnameLineWitnessCandidate)
    }
    for witness, witness_keys in keys.items():
        assert issubclass(witness, WitnessNameAliasMixin)
        assert len(witness_keys) == 1, witness
    assert keys[ClassLineWitnessCandidate] != keys[QualnameLineWitnessCandidate]


@pytest.mark.parametrize(
    "entries,accepted",
    (
        ((("A", "x"), ("B", "y")), True),
        ((("A", "x"), ("B", "x")), False),
        ((("A", "x"), ("A", "x")), False),
    ),
)
def test_requested_registration_refuses_a_key_claimed_twice(entries, accepted):
    source = (
        "class Root(metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = 'kind'\n"
        "    __registry__ = {}\n"
    )
    (node,) = ast.parse(source).body
    assert isinstance(node, ast.ClassDef)
    authority = AutoRegisterClassAuthority(node)
    if accepted:
        authority.require_requested_key_compatibility(entries)
        return
    with pytest.raises(ValueError, match="does not preserve source entries"):
        authority.require_requested_key_compatibility(entries)
