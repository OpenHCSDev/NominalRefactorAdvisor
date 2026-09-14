"""Constant contents join an original store without manufacturing object identity."""

from copy import copy
from dataclasses import dataclass
import pickle

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeConstantValue,
    NativePythonCompilation,
)
from nominal_refactor_advisor.native_declarations import NativeConstantContentsABC
from nominal_refactor_advisor.source_execution import (
    SourceAssignmentStore,
    SourceTupleCapture,
)
from test_documentation_store import execution


@dataclass
class Contents(NativeConstantContentsABC):
    value: object

    def _native_constant_value(self):
        return self.value


@pytest.mark.parametrize(
    "value", (None, True, 4, "name", (), ("run",), (None, (1, False), ()), ..., (...,))
)
def test_exact_contents_compare_without_claiming_identity(value):
    assert Contents.supports_constant(value)
    Contents(value).require_same_native_constant(Contents(value))


@pytest.mark.parametrize(
    "left,right",
    ((True, 1), ((True,), (1,)), ((None,), ()), (("a",), ("b",)), (((1,),), (1,))),
)
def test_nested_types_arity_order_and_contents_are_checked(left, right):
    with pytest.raises(ValueError, match="Native constant"):
        Contents(left).require_same_native_constant(Contents(right))


@pytest.mark.parametrize(
    "value", (3.25, b"text", 2j, NotImplemented, [], {}, frozenset(), (1, []))
)
def test_values_without_an_admitted_constant_contract_stay_unproved(value):
    assert not Contents.supports_constant(value)
    with pytest.raises(ValueError, match="admitted exact contents"):
        Contents(value).require_same_native_constant(Contents(value))


def test_subclasses_and_unknown_objects_do_not_execute_protocols():
    class Poison:
        def __eq__(self, other):
            raise AssertionError("Unadmitted equality executed")

        def __iter__(self):
            raise AssertionError("Unadmitted iteration executed")

    class TuplePoison(tuple, Poison):
        __eq__ = Poison.__eq__
        __iter__ = Poison.__iter__

    class IntPoison(int, Poison):
        __eq__ = Poison.__eq__

    for value in (Poison(), TuplePoison((1,)), IntPoison(1)):
        assert not Contents.supports_constant(value)
        assert not Contents.supports_constant((value,))


@pytest.mark.parametrize("expression", ("()", "('run',)", "(None, (1, False), 'name')"))
@pytest.mark.parametrize("in_class", (False, True))
def test_folded_tuple_store_joins_original_source_and_actual_compiled_output(
    expression, in_class
):
    source = ("class Family:\n    " if in_class else "") + f"payload = {expression}\n"
    env = execution(source)
    context = (
        env.class_entry(env.module.module.body[0]).context
        if in_class
        else env.entry.context
    )
    store = SourceAssignmentStore(env, context.flow.mutations[-1])
    prefix = env.required_prefix(context, None)
    receipt = store.require_installation(prefix)
    assert type(receipt.value) is NativeConstantValue
    captured = store.native_value(receipt.value)
    assert isinstance(captured, SourceTupleCapture)
    assert captured is store.source_value
    assert store.return_continuation(prefix) is receipt.require_return()
    with pytest.raises(ValueError, match="original production"):
        store.native_value(copy(receipt.value))
    with pytest.raises(ValueError, match="scalar"):
        receipt.value.require_scalar_store_value()
    namespace = {}
    exec(env.module.native_compilation.compile(), namespace)  # Authored fixture only.
    result = vars(namespace["Family"])["payload"] if in_class else namespace["payload"]
    assert type(result) is tuple
    captured.require_same_native_constant(Contents(result))
    if in_class:
        env.require_class_creation(env.module.module.body[0])


def test_equal_folded_contents_do_not_prove_cross_evaluation_object_identity():
    env = execution("first = ('run',)\nsecond = ('run',)\n")
    stores = [
        SourceAssignmentStore(env, binding)
        for binding in env.entry.context.flow.mutations
    ]
    for store in stores:
        store.require_installation(env.required_prefix(env.entry.context, None))
    first, second = [store.source_value for store in stores]
    first.require_same_native_constant(second)
    assert first is not second
    assert not first.proves_same_object(second)


def test_deep_constant_comparison_is_iterative():
    left, right = 1, 1
    for _ in range(4096):
        left, right = (left,), (right,)
    Contents(left).require_same_native_constant(Contents(right))


def test_shared_constant_subgraphs_are_compared_once_per_original_pair():
    left, right = 1, 1
    for _ in range(128):
        left, right = (left, left), (right, right)
    Contents(left).require_same_native_constant(Contents(right))
    assert Contents.supports_constant(left)
    # Reusing the left value with another right value requires a separate check.
    with pytest.raises(ValueError, match="contents differ"):
        Contents((1, 1)).require_same_native_constant(Contents((1, 2)))


def test_warm_native_constants_keep_original_receipts_without_recompilation(
    monkeypatch,
):
    original = NativePythonCompilation("payload = ('run', (1, None))\n", "constant.py")
    stores = original._execution_outcome.value_stores
    restored = pickle.loads(pickle.dumps(original))

    def unexpected_compile(self):
        raise AssertionError("Warm constant receipts must not compile")

    monkeypatch.setattr(NativePythonCompilation, "compile", unexpected_compile)
    for stored in stores:
        receipt = restored.value_store_for(
            stored.production_span, stored.source_span, stored.binding.name
        )
        receipt.require_value(receipt.value)
        receipt.value.require_same_native_constant(stored.value)
