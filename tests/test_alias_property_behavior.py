"""Descriptor projections preserve live Python lookup and object identity."""

from dataclasses import dataclass

import pytest

from nominal_refactor_advisor.descriptor_algebra import (
    AliasProperty,
    ClassAliasProperty,
    CollectionAttributeProjection,
)


@dataclass
class Member:
    value: object


def test_alias_class_access_and_live_nested_path():
    class Owner:
        alias = AliasProperty[object]("member.value")

    instance = Owner()
    instance.member = Member(object())
    assert Owner.alias is vars(Owner)["alias"]
    assert instance.alias is instance.member.value
    instance.member.value = object()
    assert instance.alias is instance.member.value
    instance.member = Member(object())
    assert instance.alias is instance.member.value


def test_alias_inherits_and_remains_an_overridable_nondata_descriptor():
    class Base:
        value = object()
        alias = AliasProperty[object]("value")

    class Child(Base):
        value = object()

    instance = Child()
    assert Child.alias is Base.alias
    assert instance.alias is Child.value
    instance.alias = object()
    assert instance.alias is vars(instance)["alias"]
    assert Base().alias is Base.value
    del instance.alias
    assert instance.alias is Child.value


def test_alias_slots_properties_and_bound_methods_keep_native_lookup():
    class Owner:
        __slots__ = ("value", "reads")
        alias = AliasProperty[object]("observed")
        method_alias = AliasProperty[object]("method")

        @property
        def observed(self):
            self.reads += 1
            return self.value

        def method(self):
            return self.value

    instance = Owner()
    instance.value, instance.reads = object(), 0
    assert instance.alias is instance.value
    assert instance.alias is instance.value
    assert instance.reads == 2
    method = instance.method_alias
    assert method.__self__ is instance
    assert method.__func__ is Owner.method
    assert method() is instance.value
    with pytest.raises(AttributeError):
        instance.alias = object()


@pytest.mark.parametrize("path", ("missing", "member.missing"))
def test_alias_missing_attributes_raise_the_original_lookup_error(path):
    class Owner:
        member = Member(object())
        alias = AliasProperty[object](path)

    with pytest.raises(AttributeError, match="missing"):
        Owner().alias


def test_class_alias_uses_the_actual_inherited_owner_even_for_instance_access():
    class Base:
        value = object()
        alias = ClassAliasProperty[object]("value")

    class Child(Base):
        value = object()

    instance = Child()
    instance.value = object()
    assert Base.alias is Base.value
    assert Child.alias is Child.value
    assert instance.alias is Child.value
    Child.value = object()
    assert instance.alias is Child.value
    descriptor = vars(Base)["alias"]
    assert descriptor.__get__(instance, Base) is Base.value
    with pytest.raises(TypeError, match="requires an owner"):
        descriptor.__get__(instance)


def test_collection_projection_keeps_order_duplicates_identity_and_live_paths():
    class Owner:
        values = CollectionAttributeProjection[object]("holder.value", "value")

    first, second = Member(object()), Member(object())
    instance = Owner()
    instance.holder = Member([second, first, second])
    assert Owner.values is vars(Owner)["values"]
    values = instance.values
    assert values == (second.value, first.value, second.value)
    assert values[0] is values[2] is second.value
    first.value = object()
    assert instance.values[1] is first.value
    instance.holder.value = []
    assert instance.values == ()


def test_collection_projection_uses_member_properties_in_iteration_order():
    observed = []

    class Entry:
        def __init__(self, name):
            self.name = name

        @property
        def value(self):
            observed.append(self.name)
            return self.name

    class Owner:
        values = CollectionAttributeProjection[str]("members", "value")

    instance = Owner()
    instance.members = (Entry(name) for name in ("second", "first", "second"))
    assert instance.values == ("second", "first", "second")
    assert observed == ["second", "first", "second"]
    assert instance.values == ()


@pytest.mark.parametrize("fault", ("collection", "member", "noniterable"))
def test_collection_projection_preserves_missing_attribute_and_iteration_errors(fault):
    class Owner:
        values = CollectionAttributeProjection[object]("members", "value")

    instance = Owner()
    if fault == "member":
        instance.members = [object()]
    elif fault == "noniterable":
        instance.members = None
    with pytest.raises(TypeError if fault == "noniterable" else AttributeError):
        _ = instance.values
