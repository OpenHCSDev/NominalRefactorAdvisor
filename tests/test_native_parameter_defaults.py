"""Live function defaults are distinct from compiled source default expressions."""

import ast
import dataclasses

import pytest

from nominal_refactor_advisor.call_binding import CompactFunctionSignature
from nominal_refactor_advisor.native_declarations import NativeParameterDefault
from nominal_refactor_advisor.scan_cache import ScanCache
from test_native_dataclass_factory import factory_environment


def default_map(function):
    return {
        default.parameter_name: default.value
        for default in NativeParameterDefault.from_function(function)
    }


def test_native_alignment_uses_actual_defaults_and_only_defaultable_parameters():
    def function(first, /, second=2, *items, option=3, **options):
        return first, second, option

    assert default_map(function) == {"second": 2, "option": 3}
    function.__defaults__ = (10, 20, 30)
    function.__kwdefaults__ = {"option": 40, "first": 999, "items": 999}
    assert default_map(function) == {"first": 20, "second": 30, "option": 40}
    assert function() == (20, 30, 40)
    function.__defaults__ = None
    function.__kwdefaults__ = None
    assert default_map(function) == {}


def test_default_values_keep_identity_without_equality_or_execution():
    class Hostile:
        def __eq__(self, other):
            raise AssertionError("Default equality was invoked")

        def __repr__(self):
            raise AssertionError("Default repr was invoked")

    value = Hostile()

    def function(argument=value):
        raise AssertionError("Target function was invoked")

    (default,) = NativeParameterDefault.from_function(function)
    assert default.value is value
    with pytest.raises(ValueError):
        default.require_constant_contents(None)


def test_default_storage_subclasses_cannot_run_introspection_callbacks():
    class Values(tuple):
        def __len__(self):
            raise AssertionError("Tuple len callback")

        def __iter__(self):
            raise AssertionError("Tuple iterator callback")

        def __getitem__(self, key):
            raise AssertionError("Tuple indexing callback")

    class Keywords(dict):
        def items(self):
            raise AssertionError("Dictionary items callback")

        def __iter__(self):
            raise AssertionError("Dictionary iterator callback")

        def __getitem__(self, key):
            raise AssertionError("Dictionary indexing callback")

    def function(value=None, *, option=None):
        raise AssertionError("Function invoked")

    function.__defaults__ = Values((12,))
    function.__kwdefaults__ = Keywords(option=13)
    assert default_map(function) == {"value": 12, "option": 13}


def test_non_string_keyword_default_key_is_rejected_without_comparison():
    class Key:
        def __hash__(self):
            return hash("option")

        def __eq__(self, other):
            raise AssertionError("Keyword key comparison")

    def function(*, option):
        pass

    function.__kwdefaults__ = {Key(): 12}
    with pytest.raises(ValueError, match="exact string"):
        NativeParameterDefault.from_function(function)


def test_signature_default_projection_replaces_source_defaults_without_mutation():
    arguments = ast.parse("def f(first, second=2, *, option=3): pass").body[0].args
    original = CompactFunctionSignature.from_arguments(arguments)
    projected = original.with_default_names(frozenset(("first",)))
    assert [p.name for p in original.parameters if p.has_default] == [
        "second",
        "option",
    ]
    assert [p.name for p in projected.parameters if p.has_default] == ["first"]
    assert [p.kind for p in projected.parameters] == [
        p.kind for p in original.parameters
    ]


@pytest.mark.parametrize("warm", (False, True))
@pytest.mark.parametrize("slot", ("__defaults__", "__kwdefaults__"))
def test_factory_cannot_omit_a_now_required_parameter(monkeypatch, warm, slot):
    environment, operation = factory_environment("")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    with ScanCache.scope():
        if warm:
            authority.require_closed()
        monkeypatch.setattr(dataclasses.dataclass, slot, None)
        with pytest.raises(ValueError, match="missing_required_argument"):
            authority.require_closed()
        with pytest.raises(TypeError):
            dataclasses.dataclass()


def test_factory_uses_actual_default_contents_even_when_signature_is_unchanged(
    monkeypatch,
):
    environment, operation = factory_environment("")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    with ScanCache.scope():
        authority.require_closed()
        monkeypatch.setattr(dataclasses.dataclass, "__defaults__", (object(),))
        with pytest.raises(ValueError):
            authority.require_closed()


def test_explicit_argument_does_not_consume_replaced_default(monkeypatch):
    environment, operation = factory_environment("None")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    monkeypatch.setattr(dataclasses.dataclass, "__defaults__", (object(),))
    authority.require_closed()


def test_native_default_inspection_requires_exact_function():
    with pytest.raises(ValueError, match="exact function"):
        NativeParameterDefault.from_function(len)


def test_binding_joins_current_factory_source_without_a_supplied_behavior_premise():
    environment, operation = factory_environment("frozen=True")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    assert authority.bound_arguments.is_exact
    authority.require_closed()
    assert not environment.entry.operation_conditions


@pytest.mark.parametrize("names", (("unknown",), ("items",), ("options",)))
def test_default_projection_cannot_add_unknown_or_variadic_parameters(names):
    arguments = ast.parse("def f(first, *items, **options): pass").body[0].args
    signature = CompactFunctionSignature.from_arguments(arguments)
    with pytest.raises(ValueError, match="nonvariadic"):
        signature.with_default_names(frozenset(names))


def test_binding_ignores_signature_and_wrapper_metadata(monkeypatch):
    environment, operation = factory_environment("")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    monkeypatch.setattr(dataclasses.dataclass, "__signature__", object(), raising=False)
    monkeypatch.setattr(dataclasses.dataclass, "__wrapped__", len, raising=False)
    authority.require_closed()


def test_current_default_observations_are_not_function_identity_cached():
    def function(value=None):
        pass

    with ScanCache.scope():
        (before,) = NativeParameterDefault.from_function(function)
        before.require_constant_contents(None)
        function.__defaults__ = (17,)
        (after,) = NativeParameterDefault.from_function(function)
        after.require_constant_contents(17)
        assert before is not after
        before.require_constant_contents(None)  # An observation, not current storage.
