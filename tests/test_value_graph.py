"""Native value semantics for declaration-owned immutable graph records."""

from dataclasses import dataclass, field, fields
import multiprocessing
import pickle
import weakref

import pytest

from nominal_refactor_advisor.value_graph import DataclassGraphValue


@dataclass(frozen=True, eq=False)
class GraphValue(DataclassGraphValue):
    value: object


@dataclass(frozen=True, eq=False)
class ExtendedGraphValue(GraphValue):
    label: str


@dataclass(frozen=True, eq=False)
class FlaggedValue(DataclassGraphValue):
    compared: object
    compared_not_hashed: object = field(hash=False)
    ignored: object = field(compare=False)


@dataclass(frozen=True)
class NativeValue:
    value: object


@dataclass(frozen=True, eq=False)
class IdentityLeaf:
    value: object

    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = object.__hash__


@dataclass(frozen=True, eq=False)
class CustomGraphValue(GraphValue):
    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return 37


@dataclass(frozen=True, eq=False)
class DelegatingGraphValue(GraphValue):
    def __eq__(self, other: object) -> bool:
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class TupleLeaf(tuple):
    pass


class AlwaysEqualTuple(tuple):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, AlwaysEqualTuple)

    def __hash__(self) -> int:
        return 19


class ReflectedEqual:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphValue)


class ContraryInequality:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContraryInequality)

    def __ne__(self, other: object) -> bool:
        return True


def _nested(depth: int) -> GraphValue:
    value = GraphValue(1)
    for _ in range(depth):
        value = GraphValue((value, value))
    return value


def _inspect_in_spawn(value: GraphValue) -> tuple[bool, int, bool]:
    return value == _nested(20), hash(value), value.value[0] is value.value[1]


def test_inherited_fields_are_compared_and_hashed() -> None:
    first = ExtendedGraphValue(4, "first")
    assert first == ExtendedGraphValue(4, "first")
    assert first != ExtendedGraphValue(5, "first")
    assert first != ExtendedGraphValue(4, "other")
    assert hash(first) == hash((4, "first"))
    assert tuple(declaration.name for declaration in fields(first)) == (
        "value",
        "label",
    )


def test_field_flags_remain_the_comparison_and_hash_authority() -> None:
    first = FlaggedValue(1, [2], object())
    equal = FlaggedValue(1, [2], object())
    unequal = FlaggedValue(1, [3], object())
    assert first == equal
    assert first != unequal
    assert hash(first) == hash(equal) == hash(unequal) == hash((1,))


def test_explicit_hash_true_is_not_replaced_by_compare_default() -> None:
    @dataclass(frozen=True, eq=False)
    class ExplicitFlags(DataclassGraphValue):
        first: int = field(compare=False, hash=True)
        second: int = field(compare=True, hash=False)

    # Native dataclasses allow this unusual policy. The declaration owns it.
    assert hash(ExplicitFlags(7, 8)) == hash((7,))


@pytest.mark.parametrize(
    "value",
    [None, 0, True, 1.25, "text", b"bytes", (), (1, (2, 3)), frozenset({2, 3})],
)
def test_hash_agrees_with_native_dataclass_field_tuple(value: object) -> None:
    assert hash(GraphValue(value)) == hash(NativeValue(value))


def test_tuple_subclass_preserves_native_hash_and_equality() -> None:
    ordinary = GraphValue((1, (2, 3)))
    subclassed = GraphValue(TupleLeaf((1, (2, 3))))
    assert ordinary == subclassed
    assert subclassed == ordinary
    assert hash(ordinary) == hash(subclassed)


def test_custom_tuple_and_graph_overrides_remain_opaque() -> None:
    assert GraphValue(AlwaysEqualTuple((1,))) == GraphValue(AlwaysEqualTuple((2,)))
    assert hash(GraphValue(AlwaysEqualTuple((1,)))) == hash((19,))
    first = GraphValue(CustomGraphValue("first"))
    other = GraphValue(CustomGraphValue("other"))
    assert first == other
    assert hash(first) == hash(other) == hash((37,))


def test_custom_dataclass_leaf_keeps_identity_semantics() -> None:
    shared = IdentityLeaf(1)
    assert GraphValue(shared) == GraphValue(shared)
    assert GraphValue(shared) != GraphValue(IdentityLeaf(1))
    assert hash(GraphValue(shared)) == hash((shared,))


def test_override_can_delegate_to_base_field_semantics() -> None:
    first = DelegatingGraphValue((GraphValue(1), GraphValue(2)))
    second = DelegatingGraphValue((GraphValue(1), GraphValue(2)))
    assert first == second
    assert first != DelegatingGraphValue((GraphValue(1), GraphValue(3)))
    assert hash(first) == hash(second) == hash((first.value,))
    assert GraphValue(first) == GraphValue(second)
    assert hash(GraphValue(first)) == hash(GraphValue(second))


def test_memo_retains_temporary_descriptor_values_during_traversal() -> None:
    projected = []

    class TemporaryField:
        def __get__(self, instance, owner):
            if instance is None:
                raise AttributeError
            assert all(reference() is not None for reference in projected)
            value = NativeValue(instance.__dict__["_value"])
            projected.append(weakref.ref(value))
            return value

        def __set__(self, instance, value):
            instance.__dict__["_value"] = value

    @dataclass(frozen=True, eq=False)
    class DescriptorValue(DataclassGraphValue):
        value: object = TemporaryField()

    first = GraphValue(tuple(DescriptorValue(index) for index in range(20)))
    second = GraphValue(tuple(DescriptorValue(index) for index in range(20)))
    assert first == second
    assert len(projected) == 40
    projected.clear()
    first_hash = hash(first)
    assert len(projected) == 20
    projected.clear()
    assert first_hash == hash(second)


def test_comparison_uses_leaf_equality_not_inequality_override() -> None:
    assert GraphValue(ContraryInequality()) == GraphValue(ContraryInequality())


def test_same_nan_leaf_matches_native_container_identity_shortcut() -> None:
    shared_nan = float("nan")
    assert GraphValue(shared_nan) == GraphValue(shared_nan)
    assert GraphValue(shared_nan) != GraphValue(float("nan"))
    assert hash(GraphValue(shared_nan)) == hash(NativeValue(shared_nan))


def test_nominal_type_and_reflected_comparison_match_dataclasses() -> None:
    assert GraphValue(1) != NativeValue(1)
    assert GraphValue(1) != ExtendedGraphValue(1, "label")
    assert DataclassGraphValue.__eq__(GraphValue(1), NativeValue(1)) is NotImplemented
    assert GraphValue(1) == ReflectedEqual()
    assert GraphValue(GraphValue(1)) == GraphValue(ReflectedEqual())


def test_unhashable_leaf_is_not_silently_omitted() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        hash(GraphValue([]))


def test_deep_shared_graph_is_compared_and_hashed_without_recursion() -> None:
    first, second = _nested(1500), _nested(1500)
    assert first == second
    assert hash(first) == hash(second)
    assert first != GraphValue((second, GraphValue(2)))


def test_physical_sharing_does_not_define_structural_equality() -> None:
    child = GraphValue((1, 2))
    shared = GraphValue((child, child))
    separate = GraphValue((GraphValue((1, 2)), GraphValue((1, 2))))
    assert shared == separate
    assert hash(shared) == hash(separate)


def test_shared_fields_are_read_once_per_comparison_and_hash(monkeypatch) -> None:
    first, second = _nested(50), _nested(50)
    native_getattribute = GraphValue.__getattribute__
    reads = 0

    def observed_getattribute(self, name):
        nonlocal reads
        if name == "value":
            reads += 1
        return native_getattribute(self, name)

    monkeypatch.setattr(GraphValue, "__getattribute__", observed_getattribute)
    assert first == second
    assert reads == 102
    reads = 0
    hash(first)
    assert reads == 51


def test_pickle_and_spawn_keep_values_without_transient_hash_caches() -> None:
    value = _nested(20)
    before = pickle.dumps(value)
    original_hash = hash(value)
    assert before == pickle.dumps(value)
    restored = pickle.loads(before)
    assert restored == value
    assert hash(restored) == original_hash
    assert restored.value[0] is restored.value[1]
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        same, child_hash, shared = pool.apply(_inspect_in_spawn, (value,))
    assert same and shared
    assert child_hash == original_hash  # Integer-only graph: no salted string hashes.


def test_encountered_cycles_fail_explicitly() -> None:
    first, second = GraphValue(None), GraphValue(None)
    object.__setattr__(first, "value", first)
    object.__setattr__(second, "value", second)
    with pytest.raises(ValueError, match="acyclic"):
        hash(first)
    with pytest.raises(ValueError, match="acyclic"):
        first == second
