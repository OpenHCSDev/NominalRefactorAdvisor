"""Native class inspection uses actual storage without replaying Python hooks."""

import traceback
from typing import Generic, TypeVar

import pytest

from nominal_refactor_advisor.class_mro import DeclarationMroType
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration


def original_mro(cls):
    return type.__dict__["__mro__"].__get__(cls, type(cls))


def projected_declarations(carrier):
    return tuple(
        (
            owner.declaration.declaration
            if isinstance(owner, DeclarationMroType)
            else owner
        )
        for owner in carrier.__mro__
    )


def assert_same_objects(actual, expected):
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected, strict=True))


def test_overridden_metaclass_attribute_lookup_is_not_an_inspection_authority():
    calls = []

    class Meta(type):
        def __getattribute__(cls, name):
            calls.append(name)
            raise AssertionError("native class attribute hook executed")

    class Leaf(metaclass=Meta):
        marker = 1

    calls.clear()
    native = NativeClassMroDeclaration(Leaf)
    assert "marker" in native.member_binding_names
    assert native.qualified_name.endswith(".Leaf")
    assert_same_objects(projected_declarations(native.mro_type), (Leaf, object))
    assert calls == []


@pytest.mark.parametrize(
    "attribute", ("__mro__", "__bases__", "__dict__", "__module__")
)
def test_metaclass_data_descriptor_cannot_replace_original_class_storage(attribute):
    calls = []

    def hostile_descriptor(cls):
        calls.append(attribute)
        raise AssertionError("metaclass data descriptor executed")

    meta = type("Meta", (type,), {attribute: property(hostile_descriptor)})
    leaf = meta("Leaf", (), {"__module__": __name__, "marker": 1})
    calls.clear()
    native = NativeClassMroDeclaration(leaf)
    assert native.qualified_name == __name__ + ".Leaf"
    assert "marker" in native.member_binding_names
    assert_same_objects(projected_declarations(native.mro_type), (leaf, object))
    assert calls == []


def test_generic_origin_refuses_custom_lookup_without_executing_it():
    calls = []
    item = TypeVar("item")
    sentinel = object()

    def substitute(argument):
        calls.append(argument)
        return sentinel

    class Meta(type):
        def __getattribute__(cls, name):
            calls.append(name)
            if name == "__class_getitem__":
                return substitute
            return type.__getattribute__(cls, name)

    class Root(Generic[item], metaclass=Meta):
        pass

    assert Root[int] is sentinel
    assert "__class_getitem__" in calls
    calls.clear()
    with pytest.raises(ValueError):
        NativeClassMroDeclaration(Root).require_generic_origin()
    assert calls == []


def test_generic_origin_refuses_metaclass_descriptor_substitution_without_execution():
    calls = []
    item = TypeVar("item")
    sentinel = object()

    def substitute(argument):
        calls.append(argument)
        return sentinel

    class Meta(type):
        @property
        def __class_getitem__(cls):
            calls.append("descriptor")
            return substitute

    class Root(Generic[item], metaclass=Meta):
        pass

    assert Root[int] is sentinel
    assert "descriptor" in calls
    calls.clear()
    with pytest.raises(ValueError):
        NativeClassMroDeclaration(Root).require_generic_origin()
    assert calls == []


def test_custom_subscription_is_refused_without_invoking_it():
    calls = []
    item = TypeVar("item")

    class Root(Generic[item]):
        def __class_getitem__(cls, argument):
            calls.append(argument)
            raise AssertionError("custom subscription executed")

    calls.clear()
    with pytest.raises(ValueError):
        NativeClassMroDeclaration(Root).require_generic_origin()
    assert calls == []


def test_metaclass_subscription_is_refused_without_invoking_it():
    calls = []
    item = TypeVar("item")

    class Meta(type):
        def __getitem__(cls, argument):
            calls.append(argument)
            raise AssertionError("metaclass subscription executed")

    class Root(Generic[item], metaclass=Meta):
        pass

    calls.clear()
    with pytest.raises(ValueError):
        NativeClassMroDeclaration(Root).require_generic_origin()
    assert calls == []


@pytest.mark.parametrize("consumer", ("namespace", "members", "generic"))
def test_nontext_colliding_namespace_key_is_rejected_before_lookup_or_hashing(consumer):
    calls = []

    class Key:
        def __hash__(self):
            calls.append("hash")
            return hash("__class_getitem__")

        def __eq__(self, other):
            calls.append("equality")
            return False

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("namespace key formatted")

    key = Key()
    leaf = type("Leaf", (), {key: 1})
    calls.clear()
    native = NativeClassMroDeclaration(leaf)
    with pytest.raises(ValueError):
        if consumer == "namespace":
            native.stored_namespace(leaf)
        elif consumer == "members":
            _ = native.member_binding_names
        else:
            native.require_generic_origin()
    assert calls == []


def test_text_subclass_namespace_key_is_not_accepted_as_inert_lookup_storage():
    calls = []

    class Key(str):
        def __hash__(self):
            calls.append("hash")
            return super().__hash__()

        def __eq__(self, other):
            calls.append("equality")
            return super().__eq__(other)

    leaf = type("Leaf", (), {Key("__class_getitem__"): object()})
    calls.clear()
    with pytest.raises(ValueError):
        NativeClassMroDeclaration(leaf).require_generic_origin()
    assert calls == []


@pytest.mark.parametrize("text_subclass", (False, True))
def test_module_metadata_is_validated_without_formatting_hostile_value(text_subclass):
    calls = []

    class Hostile:
        def __str__(self):
            calls.append("str")
            raise AssertionError("module metadata converted")

        def __format__(self, spec):
            calls.append("format")
            raise AssertionError("module metadata formatted")

    class HostileText(str):
        def __str__(self):
            calls.append("str")
            raise AssertionError("module text converted")

        def __format__(self, spec):
            calls.append("format")
            raise AssertionError("module text formatted")

    class Leaf:
        pass

    Leaf.__module__ = HostileText("native.module") if text_subclass else Hostile()
    calls.clear()
    with pytest.raises(ValueError):
        _ = NativeClassMroDeclaration(Leaf).qualified_name
    assert calls == []


def test_qualified_name_text_subclass_is_not_formatted_during_inspection():
    calls = []

    class HostileText(str):
        def __format__(self, spec):
            calls.append("format")
            raise AssertionError("qualified name text formatted")

    class Leaf:
        pass

    value = HostileText("native.Leaf")
    Leaf.__qualname__ = value
    assert type.__dict__["__qualname__"].__get__(Leaf, type(Leaf)) is value
    calls.clear()
    with pytest.raises(ValueError):
        _ = NativeClassMroDeclaration(Leaf).qualified_name
    assert calls == []


def test_custom_mro_matching_actual_c3_is_not_reexecuted():
    calls = []

    class Meta(type):
        def mro(cls):
            calls.append(cls)
            return type.mro(cls)

    class Leaf(metaclass=Meta):
        pass

    assert calls == [Leaf]
    calls.clear()
    native = NativeClassMroDeclaration(Leaf)
    projected = native.mro_type
    assert_same_objects(projected_declarations(projected), original_mro(Leaf))
    assert native.mro_type is projected
    assert calls == []


def test_native_non_c3_order_is_refused_without_reexecuting_custom_mro():
    calls = []

    class Left:
        pass

    class Right:
        pass

    class Meta(type):
        def mro(cls):
            calls.append(cls)
            return [cls, Right, Left, object]

    class Leaf(Left, Right, metaclass=Meta):
        pass

    assert_same_objects(original_mro(Leaf), (Leaf, Right, Left, object))
    calls.clear()
    with pytest.raises(ValueError):
        _ = NativeClassMroDeclaration(Leaf).mro_type
    assert calls == []


@pytest.mark.parametrize("mutate_ancestor", (False, True))
def test_mutable_bases_do_not_reuse_a_stale_cached_mro(mutate_ancestor):
    class Left:
        pass

    class Right:
        pass

    class Parent(Left):
        pass

    class Leaf(Parent):
        pass

    native = NativeClassMroDeclaration(Leaf)
    before = native.mro_type
    if mutate_ancestor:
        Parent.__bases__ = (Right,)
    else:
        Leaf.__bases__ = (Right,)
    after = native.mro_type
    assert after is not before
    assert_same_objects(projected_declarations(after), original_mro(Leaf))
    assert native.mro_type is after


def test_changed_custom_mro_with_unchanged_bases_invalidates_previous_validation():
    calls = []
    reversed_order = False

    class Left:
        pass

    class Right:
        pass

    class Meta(type):
        def mro(cls):
            calls.append(cls)
            if reversed_order:
                return [cls, Right, Left, object]
            return type.mro(cls)

    class Leaf(Left, Right, metaclass=Meta):
        pass

    native = NativeClassMroDeclaration(Leaf)
    before = native.mro_type
    assert_same_objects(projected_declarations(before), (Leaf, Left, Right, object))
    reversed_order = True
    Leaf.__bases__ = (Left, Right)
    assert_same_objects(original_mro(Leaf), (Leaf, Right, Left, object))
    calls.clear()
    with pytest.raises(ValueError):
        _ = native.mro_type
    assert calls == []


def test_same_named_classes_keep_distinct_carriers_without_hash_or_equality_hooks():
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            raise AssertionError("class hashed")

        def __eq__(cls, other):
            calls.append("equality")
            raise AssertionError("class compared")

    first = Meta("Leaf", (), {"__module__": "same"})
    second = Meta("Leaf", (), {"__module__": "same"})
    calls.clear()
    first_carrier = NativeClassMroDeclaration(first).mro_type
    second_carrier = NativeClassMroDeclaration(second).mro_type
    assert first_carrier is not second_carrier
    assert_same_objects(projected_declarations(first_carrier), (first, object))
    assert_same_objects(projected_declarations(second_carrier), (second, object))
    assert calls == []


def test_native_mro_omitting_actual_direct_base_is_explicitly_refused():
    calls = []

    class Base:
        pass

    class Meta(type):
        def mro(cls):
            calls.append(cls)
            return [cls, object]

    class Leaf(Base, metaclass=Meta):
        pass

    assert type.__dict__["__bases__"].__get__(Leaf, Meta) == (Base,)
    assert_same_objects(original_mro(Leaf), (Leaf, object))
    assert not issubclass(Leaf, Base)
    calls.clear()
    with pytest.raises(ValueError, match="ancestry|bases|C3"):
        _ = NativeClassMroDeclaration(Leaf).mro_type
    assert calls == []


def test_missing_base_diagnostic_formatting_does_not_invoke_native_class_repr():
    calls = []

    class BaseMeta(type):
        def __repr__(cls):
            calls.append("repr")
            return "hostile native class representation"

    class Base(metaclass=BaseMeta):
        pass

    class Meta(BaseMeta):
        def mro(cls):
            return [cls, object]

    class Leaf(Base, metaclass=Meta):
        pass

    calls.clear()
    with pytest.raises(ValueError) as caught:
        _ = NativeClassMroDeclaration(Leaf).mro_type
    assert calls == []
    rendered = "".join(traceback.format_exception(caught.value))
    assert "ValueError" in rendered
    assert calls == []


@pytest.mark.parametrize("depth", (2, 8))
def test_diamond_revalidates_each_actual_class_once_without_recreating_warm_carriers(
    depth, monkeypatch
):
    current = type("Root", (), {})
    for layer in range(depth):
        left = type(f"Left{layer}", (current,), {})
        right = type(f"Right{layer}", (current,), {})
        current = type(f"Join{layer}", (left, right), {})

    expected_classes = len(original_mro(current)) - 1
    calls = []
    original_getter = vars(NativeClassMroDeclaration)["native_bases"]

    def count_original_bases(declaration):
        calls.append(declaration)
        return original_getter(declaration)

    # Instrumentation delegates the actual native storage getter unchanged.
    # The classes and their native creation are outside this observation.
    monkeypatch.setattr(
        NativeClassMroDeclaration,
        "native_bases",
        staticmethod(count_original_bases),
    )
    native = NativeClassMroDeclaration(current)
    before = native._projected_mro.cache_info()
    first = native.mro_type
    fresh = native._projected_mro.cache_info()
    assert len(calls) == expected_classes
    assert len({id(declaration) for declaration in calls}) == expected_classes
    assert fresh.misses - before.misses == expected_classes
    assert_same_objects(projected_declarations(first), original_mro(current))

    calls.clear()
    second = native.mro_type
    warm = native._projected_mro.cache_info()
    assert second is first
    assert len(calls) == expected_classes
    assert len({id(declaration) for declaration in calls}) == expected_classes
    assert warm.misses == fresh.misses


@pytest.mark.parametrize("include_other_class", (False, True))
def test_native_base_mutation_omitting_root_is_refused_without_formatting_root(
    include_other_class,
):
    calls = []
    omit_root = False

    class Base:
        pass

    class Other:
        pass

    class Meta(type):
        def mro(cls):
            calls.append("mro")
            if omit_root:
                return [Other, object] if include_other_class else [object]
            return type.mro(cls)

        def __repr__(cls):
            calls.append("repr")
            return "hostile root representation"

    class Leaf(Base, metaclass=Meta):
        pass

    native = NativeClassMroDeclaration(Leaf)
    before = native.mro_type
    assert_same_objects(projected_declarations(before), (Leaf, Base, object))
    omit_root = True
    # CPython accepts this mutation even though fresh construction with that
    # MRO would fail later during __init_subclass__. No synthetic class state.
    Leaf.__bases__ = (Base,)
    expected = (Other, object) if include_other_class else (object,)
    assert_same_objects(original_mro(Leaf), expected)
    calls.clear()
    with pytest.raises(ValueError) as caught:
        _ = native.mro_type
    assert calls == []
    assert "ValueError" in "".join(traceback.format_exception(caught.value))
    assert calls == []
