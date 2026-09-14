"""Subscription production keeps operand provenance separate from source effects."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NativeTypePremise,
)
from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.native_declarations import NativeConstantContentsABC
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_source_function_result import execution


@pytest.mark.parametrize("expression", ("container[key]", "container[first][second]"))
def test_subscription_retains_original_inputs_in_its_native_return(expression):
    compilation = NativePythonCompilation(f"result = {expression}\n", "subscription.py")
    assignment = ast.parse(compilation.source).body[0]
    store = compilation.value_store_for(
        SourceByteSpan.require_node(assignment.value),
        SourceByteSpan.require_node(assignment.targets[0]),
        "result",
    )
    assert len(store.value.inputs) == 2
    store.require_return().require_value(store.value)
    for value in store.value.productions():
        store.require_return().require_value(value)
    with pytest.raises(ValueError):
        store.require_return().require_value(replace(store.value))


@pytest.mark.parametrize("expression", ("list[str]", "dict[str, tuple[int, ...]]"))
@pytest.mark.parametrize("annotation", (False, True))
def test_native_generic_alias_has_a_complete_original_class_storage_path(
    expression, annotation
):
    statement = f"field: {expression} = 1" if annotation else f"field = {expression}"
    source = f"class Owner:\n    {statement}\n"
    namespace = {}
    exec(source, namespace)  # Authored control only; the analyzer never runs it.
    env = execution(source)
    env.class_entry(env.module.module.body[0]).require_installed_result()


@pytest.mark.parametrize("expression", ("left + right", "left | right"))
def test_other_binary_operators_are_not_treated_as_subscription(expression):
    compilation = NativePythonCompilation(f"result = {expression}\n", "binary.py")
    assignment = ast.parse(compilation.source).body[0]
    with pytest.raises(ValueError):
        compilation.value_store_for(
            SourceByteSpan.require_node(assignment.value),
            SourceByteSpan.require_node(assignment.targets[0]),
            "result",
        )


def test_custom_subscription_effects_still_require_their_actual_protocol():
    source = (
        "class Custom:\n"
        "    def __class_getitem__(cls, key):\n"
        "        raise RuntimeError('must never execute analyzed source')\n"
        "class Owner:\n"
        "    field = Custom[str]\n"
    )
    env = execution(source)
    with pytest.raises(ValueError):
        env.class_entry(env.module.module.body[-1]).require_installed_result()


def test_ellipsis_is_constant_content_without_expanding_dictionary_key_admission():
    assert NativeConstantContentsABC.supports_constant((str.__name__, Ellipsis))
    assert not NativeConstantContentsABC.supports_scalar(Ellipsis)
    CapturedNativeObject(Ellipsis).require_constant_contents(Ellipsis)
    with pytest.raises(ValueError):
        NativeTypePremise(type(Ellipsis)).require_constant_contents(Ellipsis)
