"""Class-body evidence precedes decorator application and final installation."""

import ast
from dataclasses import replace

import pytest

from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import (
    SourceCreatedClassCapture,
    SourceCreatedFunctionCapture,
)
from test_source_function_result import execution, function, native


@pytest.mark.parametrize("nested", (False, True))
def test_raw_class_body_is_available_before_its_replacing_decorator(nested):
    source = (
        "def transform(raw):\n    global observed\n    observed = raw\n    return object\n"
        "@transform\nclass Owner:\n"
        "    def chosen(): pass\n" + ("    class Inner: pass\n" if nested else "")
    )
    native(
        source + "assert Owner is object\nassert observed.chosen.__name__ == 'chosen'\n"
    )
    environment = execution(source)
    owner_node = environment.module.module.body[1]
    entry = environment.class_entry(owner_node)
    node, context, binding = function(environment)
    result = environment.definition_result(context, binding)
    result.require_closed()
    assert result.node is node
    assert result.source_definition()[1] is binding
    assert entry.completed is None
    if nested:
        inner = owner_node.body[-1]
        environment.require_class_creation(inner)
    # A proved raw class body is not evidence for the installed decorated object.
    for require_installed in (
        lambda: environment.require_class_creation(owner_node),
        SourceCreatedClassCapture(entry).require_closed,
        lambda: environment.required_prefix(environment.entry.context, None),
    ):
        with pytest.raises(ValueError, match="unproved"):
            require_installed()


@pytest.mark.parametrize(
    "header",
    (
        "@missing\nclass Owner:",
        "@transform()\nclass Owner:",
        "@transform\nclass Owner(metaclass=type):",
    ),
)
def test_raw_body_still_requires_actual_header_and_builder_proof(header):
    source = "def transform(): return object\n" + header + "\n    def chosen(): pass\n"
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


def test_later_function_does_not_bypass_preceding_class_transformation():
    source = (
        "def transform(raw): return object\n"
        "@transform\nclass Owner: pass\n"
        "def chosen(): pass\n"
    )
    native(source)
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


def test_raw_class_body_keeps_original_native_creator_identity():
    source = (
        "def transform(raw): return object\n"
        "@transform\nclass Owner:\n    def chosen(): pass\n"
    )
    environment = execution(source)
    owner = next(
        node
        for node in environment.module.module.body
        if isinstance(node, ast.ClassDef)
    )
    entry = environment.class_entry(owner)
    original = entry.capture.body
    entry.require_native_creator(original)
    with pytest.raises(ValueError, match="different class body"):
        entry.require_native_creator(replace(original))


@pytest.mark.parametrize("decorator", ("missing", "transform()"))
def test_raw_function_also_requires_its_decorator_operand_to_be_evaluated(decorator):
    source = "def transform(): return object\n" f"@{decorator}\ndef chosen(): pass\n"
    environment = execution(source)
    node, _, _ = function(environment)
    with pytest.raises(ValueError, match="unproved|missing"):
        SourceCreatedFunctionCapture(environment, node).require_descriptor_argument()


def test_noncallable_decorator_allows_body_but_not_final_class_installation():
    source = "@42\nclass Owner:\n    global chosen\n    def chosen(): pass\n"
    native(
        "namespace = {}\n"
        f"try:\n    exec({source!r}, namespace)\n"
        "except TypeError:\n    pass\n"
        "else:\n    raise AssertionError('noncallable decorator accepted')\n"
        "assert namespace['chosen'].__name__ == 'chosen'\n"
        "assert 'Owner' not in namespace\n"
    )
    environment = execution(source)
    _, context, binding = function(environment)
    environment.definition_result(context, binding).require_closed()
    with pytest.raises(ValueError, match="unproved"):
        environment.require_class_creation(environment.module.module.body[0])


def test_deleted_decorator_cannot_resurrect_its_old_value_to_admit_a_body():
    source = (
        "decorator = property\ndel decorator\n"
        "@decorator\nclass Owner:\n    global chosen\n    def chosen(): pass\n"
    )
    native(
        "namespace = {}\n"
        f"try:\n    exec({source!r}, namespace)\n"
        "except NameError:\n    pass\n"
        "else:\n    raise AssertionError('missing decorator accepted')\n"
        "assert 'chosen' not in namespace\n"
    )
    environment = execution(source)
    _, context, binding = function(environment)
    with pytest.raises(ValueError, match="unproved"):
        environment.definition_result(context, binding)


def test_header_uses_original_decorator_capture_before_a_later_operand_rebinds_it():
    source = (
        "decorator = property\n@decorator\n@(decorator := 42)\n"
        "class Owner:\n    global chosen\n    def chosen(): pass\n"
    )
    native(
        "namespace = {}\n"
        f"try:\n    exec({source!r}, namespace)\n"
        "except TypeError:\n    pass\n"
        "else:\n    raise AssertionError('noncallable decorator accepted')\n"
        "assert namespace['chosen'].__name__ == 'chosen'\n"
        "assert namespace['decorator'] == 42\n"
    )
    environment = execution(source)
    _, context, binding = function(environment)
    environment.definition_result(context, binding).require_closed()
    entry = environment.class_entry(environment.module.module.body[-1])
    original = entry.definition.target.decorator_uses[0]
    environment.kernel._read_use(
        original, entry.parent_context, frozenset()
    ).require_native_identity(NativeDeclaration(property))
    with pytest.raises(ValueError, match="unproved"):
        entry.require_installed_result()
