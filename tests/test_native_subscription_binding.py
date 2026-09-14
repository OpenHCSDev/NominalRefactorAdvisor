"""Native argument safety is separate from successful subscription binding."""

import ast
from copy import deepcopy
from dataclasses import replace
from typing import ClassVar

import pytest

from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.captured_reference import CapturedNativeObject
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_subscription import (
    ClassVariableSubscription,
    NativeSubscriptionAuthority,
)
from nominal_refactor_advisor.product_flow import CompactSubscription
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_definition_header_prefix_admission import runtime
from test_native_subscription_admission import _subscription
from test_source_function_result import execution
from test_source_operation_completion_premise import supplied_entry


def original_subscription(environment):
    node = environment.module.module.body[1].value
    assert isinstance(node, ast.Subscript)
    operation = environment.source.node_operation(node, CompactSubscription)
    return node, operation


def require_endpoint(environment, node, endpoint):
    if endpoint == "invocation":
        environment.require_subscription(node)
    else:
        read = environment.source.value_reads_by_node[node]
        operation = environment.source.node_operation(node, CompactSubscription)
        assert operation.position.dominates(read.use.position)
        environment.required_prefix(read.context, read.use.position)


@pytest.mark.parametrize("endpoint", ("invocation", "post_invocation"))
@pytest.mark.parametrize(
    "argument,native_error",
    (
        ("()", TypeError),
        ("(int,)", TypeError),
        ("(int, str)", TypeError),
        ('"int["', SyntaxError),
    ),
)
def test_invalid_native_binding_cannot_close_the_original_invocation_or_result_cut(
    endpoint, argument, native_error
):
    text = f"from typing import ClassVar\nheld = ClassVar[{argument}]\n"
    with pytest.raises(native_error):
        runtime(text)
    environment = execution(text)
    node, _ = original_subscription(environment)
    with pytest.raises(ValueError):
        require_endpoint(environment, node, endpoint)
    assert not environment._pending


@pytest.mark.parametrize("endpoint", ("prefix", "later_read"))
@pytest.mark.parametrize("class_body", (False, True))
@pytest.mark.parametrize(
    "argument,native_error",
    (("(int, str)", TypeError), ('"int["', SyntaxError)),
)
def test_annotation_continuation_requires_binding_only_when_it_actually_executes(
    endpoint, class_body, argument, native_error
):
    annotation = f"held: ClassVar[{argument}]\n"
    text = "from typing import ClassVar\n"
    text += "class Holder:\n    " + annotation if class_body else annotation
    text += "observed = object\n"
    environment = execution(text)
    mode = ModuleAnnotationEvaluationMode.from_module(environment.module.module)

    def required_continuation():
        if endpoint == "prefix":
            environment.required_prefix(environment.source.module_context, None)
        else:
            environment.capture_value(
                environment.module.module.body[-1].value
            ).require_native_identity(NativeDeclaration(object))

    if mode.annotations_execute_at_declaration:
        with pytest.raises(native_error):
            runtime(text)
        with pytest.raises(ValueError):
            required_continuation()
    else:
        namespace = runtime(text)
        assert namespace["observed"] is object
        required_continuation()
        # Inspecting the deferred annotation still executes its original error.
        with pytest.raises(native_error):
            if class_body:
                _ = namespace["Holder"].__annotations__
            else:
                namespace["__annotate__"](1)
    assert not environment._pending


@pytest.mark.parametrize("argument", ("int", "None", '"not_yet_declared"', "list[int]"))
def test_supported_classvar_binding_needs_explicit_completion_for_prefix_admission(
    argument,
):
    text = f"from typing import ClassVar\nheld = ClassVar[{argument}]\n"
    namespace = runtime(text)
    assert namespace["held"].__origin__ is ClassVar
    environment = execution(text)
    node, operation = original_subscription(environment)
    authority = NativeSubscriptionAuthority.for_subscription(
        environment,
        environment.context_for_owner(operation.owner),
        operation.event,
    )
    assert type(authority) is ClassVariableSubscription
    assert authority.operation is operation
    NativeCreationBackend.current().require_classvar_binding(
        authority.inspected_argument
    )
    with pytest.raises(ValueError, match="explicit entry condition"):
        authority.require_closed()
    # The positive target execution above is the controlled case being supplied.
    environment = SourceModuleExecution(
        supplied_entry(environment, frozenset((operation,)))
    )
    require_endpoint(environment, node, "invocation")
    require_endpoint(environment, node, "post_invocation")


@pytest.mark.parametrize("native_receiver", (list, ClassVar))
def test_hostile_argument_hooks_are_not_run_to_manufacture_binding_evidence(
    native_receiver,
):
    events = []

    class Hostile:
        def __hash__(self):
            events.append("hash")
            raise RuntimeError("target hash executed")

        def __getattribute__(self, name):
            events.append(name)
            return object.__getattribute__(self, name)

        def __eq__(self, other):
            events.append("equality")
            raise RuntimeError("target equality executed")

        def __repr__(self):
            events.append("repr")
            return "hostile"

    value = Hostile()
    operation, environment = _subscription(
        CapturedNativeObject(native_receiver), CapturedNativeObject(value)
    )
    if native_receiver is ClassVar:
        with pytest.raises(ValueError):
            environment.require_subscription(operation.node)
    else:
        environment.require_subscription(operation.node)
        environment.capture_value(operation.node).require_closed()
    assert events == []
    # The actual object has active hooks; the analyzer simply never invokes them.
    with pytest.raises(RuntimeError, match="target hash executed"):
        ClassVar[value]
    assert events == ["hash"]


@pytest.mark.parametrize("foreign", ("node", "event", "argument"))
def test_binding_cannot_use_foreign_or_copied_original_inputs(foreign):
    environment = execution("from typing import ClassVar\nheld = ClassVar[int]\n")
    node, operation = original_subscription(environment)
    if foreign == "node":
        with pytest.raises(ValueError):
            environment.require_subscription(deepcopy(node))
    else:
        event = (
            replace(operation.event)
            if foreign == "event"
            else replace(
                operation.event,
                argument_use=replace(operation.event.argument_use),
            )
        )
        with pytest.raises(ValueError):
            ClassVariableSubscription(
                environment, replace(operation, event=event)
            ).require_closed()
    assert not environment._pending


@pytest.mark.parametrize("descriptor", (property, classmethod, staticmethod))
def test_native_descriptor_lambda_inspection_does_not_execute_source_body(descriptor):
    text = f"held = {descriptor.__name__}(lambda: body_must_remain_deferred)\n"
    namespace = runtime(text)
    assert type(namespace["held"]) is descriptor
    environment = execution(text)
    node = environment.module.module.body[0].value
    environment.require_call(node)
    environment.capture_value(node).require_closed()
    function = (
        namespace["held"].fget if descriptor is property else namespace["held"].__func__
    )
    with pytest.raises(NameError, match="body_must_remain_deferred"):
        function()
