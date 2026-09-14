"""Environment-level definition capture selects creation, not a later name read."""

import ast
from copy import deepcopy
from dataclasses import replace

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceKernel,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_reference import NativeReferenceEnvironment
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_native_subscription_admission import ReferenceFixtureEnvironment
from test_registry_original_values import authored_runtime, final_entry
from test_source_definition_result import execution


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
def test_capture_definition_keeps_creation_before_later_name_rebinding(declaration):
    source = declaration + "\nsaved = Original\nOriginal = None\n"
    environment = execution(source)
    node, alias, _ = environment.module.module.body
    operation = environment.source.mutation_operation(node)
    assert node not in environment.source.reference_reads_by_node
    assert node not in environment.source.value_reads_by_node

    value = environment.capture_definition(node)
    value.require_closed()
    context, binding = value.source_definition()
    assert binding is operation.event
    assert context is environment.context_for_owner(operation.owner)
    assert environment.source_operation(context, binding) is operation
    value.require_definition_identity(binding.target.owner)
    assert value.proves_same_object(environment.capture_value(alias.value))
    assert value.proves_same_object(environment.capture_definition(node))

    runtime = authored_runtime(source)
    assert runtime["Original"] is None
    assert runtime["saved"].__name__ == "Original"


@pytest.mark.parametrize(
    "declaration", ("class Original: pass", "def Original(): pass")
)
def test_generic_kernel_keeps_typed_open_instead_of_gaining_definition_proof(
    declaration,
):
    environment = execution(declaration + "\n")
    node = environment.module.module.body[0]
    operation = environment.source.mutation_operation(node)
    # Use actual source execution as effects, but the genuine conservative base
    # kernel. Successful source-prefix evidence does not add its missing role.
    generic = ReferenceFixtureEnvironment(
        environment.source,
        CapturedReferenceKernel(environment.initial, environment),
    )
    value = generic.capture_definition(node)
    assert isinstance(value, OpenCapturedReference)
    assert value.violation is CapturedReferenceViolation.UNPROVED_BINDING
    assert value.mutation is operation.event
    with pytest.raises(CapturedReferenceRejection) as caught:
        value.require_closed()
    assert caught.value.violation is CapturedReferenceViolation.UNPROVED_BINDING
    assert generic.definition_operation(node) is operation


def test_environment_methods_share_one_owner_and_dispatch_through_the_actual_kernel():
    environment = execution("class Original: pass\n")
    node = environment.module.module.body[0]
    shared = ReferenceFixtureEnvironment(environment.source, environment.kernel)
    assert (
        SourceModuleExecution.definition_result
        is NativeReferenceEnvironment.definition_result
    )
    assert (
        SourceModuleExecution.definition_operation
        is NativeReferenceEnvironment.definition_operation
    )
    assert shared.capture_definition(node).proves_same_object(
        environment.capture_definition(node)
    )


@pytest.mark.parametrize("foreign", ("copied", "reparsed"))
def test_capture_definition_rejects_coordinate_identical_foreign_nodes(foreign):
    environment = execution("class Original: pass\n")
    original = environment.module.module.body[0]
    node = (
        deepcopy(original)
        if foreign == "copied"
        else execution(environment.module.source).module.module.body[0]
    )
    assert node is not original
    assert ast.dump(node, include_attributes=True) == ast.dump(
        original, include_attributes=True
    )
    with pytest.raises(ValueError, match="unique actual operation"):
        environment.capture_definition(node)


@pytest.mark.parametrize("foreign", ("context", "binding", "other_execution_source"))
def test_inherited_result_query_authenticates_context_and_original_event(foreign):
    environment = execution("class Original: pass\n")
    operation = environment.source.mutation_operation(environment.module.module.body[0])
    context = environment.context_for_owner(operation.owner)
    binding = operation.event
    if foreign == "context":
        context = replace(context)
    elif foreign == "binding":
        binding = replace(binding)
    else:
        other = execution(environment.module.source)
        operation = other.source.mutation_operation(other.module.module.body[0])
        context = other.context_for_owner(operation.owner)
        binding = operation.event
    with pytest.raises(ValueError, match="canonical context|unique original operation"):
        environment.definition_result(context, binding)


def test_environment_cannot_pair_a_source_with_another_executions_kernel():
    original = execution("class Original: pass\n")
    foreign = execution(original.module.source)
    mixed = ReferenceFixtureEnvironment(foreign.source, original.kernel)
    with pytest.raises(ValueError, match="unique original operation"):
        mixed.capture_definition(foreign.module.module.body[0])


def test_same_canonical_source_in_two_executions_does_not_merge_created_objects():
    first = execution("class Original: pass\n")
    second = SourceModuleExecution.from_source(first.source)
    node = first.module.module.body[0]
    left = first.capture_definition(node)
    right = second.capture_definition(node)
    left.require_closed()
    right.require_closed()
    assert left.source_definition() == right.source_definition()
    assert first.source is second.source
    assert first.entry.frame is not second.entry.frame
    assert not left.proves_same_object(right)
    assert not right.proves_same_object(left)


def test_nondefinition_mutation_is_not_a_creation_result():
    environment = execution("value = object\n")
    node = environment.module.module.body[0].targets[0]
    operation = environment.source.mutation_operation(node)
    assert operation.node is node
    with pytest.raises(ValueError, match="actual definition binding"):
        environment.capture_definition(node)


@pytest.mark.parametrize(
    "source,definition_index,reason",
    (
        (
            "class Original(metaclass=type): pass\n",
            0,
            "Captured object is not the required native declaration",
        ),
        ("missing()\nclass Original: pass\n", 1, "unproved_execution_effects"),
        ("def Original(value=missing): pass\n", 0, "unproved"),
        ("def Original(*, value=missing): pass\n", 0, "unproved"),
        ("class Original:\n    def method(self, value=missing): pass\n", 0, "unproved"),
    ),
    ids=(
        "metaclass",
        "preceding-call",
        "positional-default",
        "keyword-default",
        "class-method-default",
    ),
)
def test_capture_definition_preserves_unproved_creation_and_header_operands(
    source, definition_index, reason
):
    environment = execution(source)
    node = environment.module.module.body[definition_index]
    with pytest.raises(ValueError, match=reason):
        environment.capture_definition(node).require_closed()
    if "missing" in source:
        with pytest.raises(NameError):
            authored_runtime(source)
    else:
        assert isinstance(authored_runtime(source)["Original"], type)


def test_function_capture_does_not_execute_or_admit_its_deferred_body():
    source = "def Original(): missing()\n"
    environment = execution(source)
    node = environment.module.module.body[0]
    value = environment.capture_definition(node)
    value.require_closed()
    operation = environment.definition_operation(node)
    value.require_definition_identity(operation.event.target.owner)
    with pytest.raises(NameError):
        authored_runtime(source)["Original"]()
    with pytest.raises(ValueError, match="unproved_execution_effects"):
        environment.required_prefix(value.context, value.context.flow.calls[0].position)


def test_registry_expected_creation_is_independent_of_a_wrong_original_operand():
    source = "class Alpha: pass\nclass Beta: pass\nsaved = Alpha\nAlpha = Beta\ntail = Alpha\n"
    environment = execution(source)
    entry = final_entry(environment)
    expected = environment.capture_definition(entry.class_node)
    actual = entry.captured_class(environment)
    expected.require_closed()
    actual.require_closed()
    assert not actual.proves_same_object(expected)
    # This unary check asks about the supplied class value, not the historical
    # registration operand. The original-operand check must independently fail.
    entry.require_class_value(expected, environment)
    with pytest.raises(ValueError):
        entry.require_class_value(actual, environment)
    with pytest.raises(ValueError):
        entry.require_original_value(environment)
    runtime = authored_runtime(source)
    assert runtime["tail"] is runtime["Beta"]
    assert runtime["tail"] is not runtime["saved"]
