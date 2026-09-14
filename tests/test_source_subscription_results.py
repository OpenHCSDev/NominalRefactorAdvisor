"""Builtin alias construction retains original operands without identity claims."""

import ast
from copy import deepcopy
from cProfile import Profile
from dataclasses import fields, replace
from types import GenericAlias

import pytest

from nominal_refactor_advisor.captured_reference import OpenCapturedReference
from nominal_refactor_advisor.native_compilation import (
    CPythonContainerConstruction,
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_subscription import (
    BuiltinGenericAliasCapture,
    BuiltinGenericAliasSubscription,
    NativeSubscriptionAuthority,
)
from nominal_refactor_advisor.product_flow import SubscriptionResultValue
from nominal_refactor_advisor.source_execution import (
    SourceModuleExecution,
    SourceTupleCapture,
)
from test_source_distinct_item_stores import execution
from test_source_tuple_capture import ConservativeResolver
from test_source_operation_completion_premise import supplied_entry


def subscription_node(environment):
    return next(
        node
        for node in ast.walk(environment.module.module)
        if isinstance(node, ast.Subscript)
    )


def subscription_read(environment, node=None):
    return environment.source.value_reads_by_node[
        subscription_node(environment) if node is None else node
    ]


def closed_alias(environment):
    result = environment.capture_value(subscription_node(environment))
    result.require_closed()
    assert isinstance(result, BuiltinGenericAliasCapture)
    return result


def authored_runtime(source):
    namespace = {}
    exec(compile(source, "<authored-subscription-control>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("native", BuiltinGenericAliasSubscription.native_declarations)
def test_builtin_origins_keep_the_actual_original_operands(native):
    source = f"held = {native.declaration.__name__}[int]\n"
    environment = execution(source)
    node = subscription_node(environment)
    read = subscription_read(environment)
    result = closed_alias(environment)
    authority = result.authority
    assert isinstance(authority, BuiltinGenericAliasSubscription)
    assert authority.invocation is read.use.value.invocation
    assert authority.node is node
    assert result.native_type is GenericAlias
    assert result.origin is authority.receiver
    assert result.argument is authority.argument
    result.origin.require_native_identity(native)
    result.argument.require_native_identity(NativeDeclaration(int))
    environment.require_subscription(node)
    runtime = authored_runtime(source)["held"]
    assert type(runtime) is GenericAlias
    assert runtime.__origin__ is native.declaration
    assert runtime.__args__ == (int,)


def test_generic_alias_stores_only_its_bound_authority_not_a_second_operand_inventory():
    assert tuple(field.name for field in fields(BuiltinGenericAliasCapture)) == (
        "authority",
    )


def test_builtin_subscription_family_projects_the_backend_owned_origin_declarations():
    assert BuiltinGenericAliasSubscription.native_declarations is (
        CPythonContainerConstruction.generic_alias_origins
    )
    assert len(BuiltinGenericAliasSubscription.native_declarations) == 7


def test_tuple_argument_retains_the_original_tuple_production_and_order():
    source = "held = tuple[int, str]\n"
    environment = execution(source)
    result = closed_alias(environment)
    argument = result.argument
    assert isinstance(argument, SourceTupleCapture)
    assert (
        argument.read.use
        is environment.source.value_reads_by_node[
            subscription_node(environment).slice
        ].use
    )
    for captured, native in zip(argument.elements, (int, str), strict=True):
        captured.require_native_identity(NativeDeclaration(native))
    assert authored_runtime(source)["held"].__args__ == (int, str)


def test_nested_alias_results_share_the_existing_original_value_query():
    source = "held = list[tuple[int, str]]\n"
    environment = execution(source)
    outer = closed_alias(environment)
    inner = environment.capture_value(subscription_node(environment).slice)
    inner.require_closed()
    assert isinstance(inner, BuiltinGenericAliasCapture)
    assert outer.argument is inner
    assert environment.capture_value(subscription_node(environment)) is outer
    inner.origin.require_native_identity(NativeDeclaration(tuple))
    runtime = authored_runtime(source)["held"]
    assert runtime.__origin__ is list
    assert runtime.__args__[0].__origin__ is tuple


def test_warmed_nested_alias_completion_does_not_repeat_native_construction_proofs():
    expression = "int"
    for _ in range(16):
        expression = f"list[{expression}]"
    environment = execution(f"held = {expression}\n")
    constructor = (
        CPythonContainerConstruction.require_generic_alias_construction.__code__
    )
    profile = Profile()
    with profile:
        result = closed_alias(environment)
    assert (
        sum(
            entry.callcount for entry in profile.getstats() if entry.code is constructor
        )
        > 0
    )
    profile.clear()
    with profile:
        for _ in range(100):
            result.require_closed()
    assert (
        sum(
            entry.callcount for entry in profile.getstats() if entry.code is constructor
        )
        == 0
    )


def test_failed_bound_completion_is_not_cached_as_success_or_as_permanent_failure(
    monkeypatch,
):
    environment = execution("held = list[int]\n")
    read = subscription_read(environment)
    authority = environment.subscription_authority(
        read.context, read.use.value.invocation
    )
    completion_name = NativeSubscriptionAuthority.completed.attrname
    assert completion_name not in vars(authority)
    constructor = NativeCreationBackend.require_generic_alias_construction.__code__
    profile = Profile()
    with monkeypatch.context() as altered:
        altered.setattr(
            NativeCreationBackend,
            "current",
            classmethod(lambda cls: SpanOnlyCreationBackend()),
        )
        with profile:
            for _ in range(3):
                with pytest.raises(ValueError):
                    authority.require_closed()
                assert completion_name not in vars(authority)
    assert (
        sum(
            entry.callcount for entry in profile.getstats() if entry.code is constructor
        )
        == 3
    )
    authority.require_closed()
    assert completion_name in vars(authority)


def test_source_created_argument_keeps_its_actual_definition_owner():
    source = "class Item: pass\nheld = list[Item]\n"
    environment = execution(source)
    result = closed_alias(environment)
    context, binding = result.argument.source_definition()
    assert (
        environment.source_operation(context, binding).node
        is environment.module.module.body[0]
    )
    runtime = authored_runtime(source)
    assert runtime["held"].__args__[0] is runtime["Item"]


def test_receiver_is_captured_before_argument_rebinding_not_recaptured_afterwards():
    source = "receiver = list\nheld = receiver[(receiver := tuple)]\n"
    environment = execution(source)
    result = closed_alias(environment)
    result.origin.require_native_identity(NativeDeclaration(list))
    result.argument.require_native_identity(NativeDeclaration(tuple))
    runtime = authored_runtime(source)
    assert runtime["receiver"] is tuple
    assert runtime["held"].__origin__ is list
    assert runtime["held"].__args__ == (tuple,)


def test_compact_result_does_not_grant_construction_to_a_generic_resolver():
    environment = execution("held = list[int]\n")
    value = subscription_read(environment).use.value
    assert isinstance(value, SubscriptionResultValue)
    unknown = object()
    assert value.resolve_value(ConservativeResolver(), unknown) is unknown


def test_canonical_earlier_invocation_cannot_produce_a_different_subscription_result():
    environment = execution("first = list[int]\nsecond = tuple[str]\n")
    first, second = (statement.value for statement in environment.module.module.body)
    first_read = subscription_read(environment, first)
    second_read = subscription_read(environment, second)
    first_invocation = first_read.use.value.invocation
    assert first_invocation.position.dominates(second_read.use.position)
    # Each operation remains original and individually authentic. The corrupt
    # association falsely claims the second result came from the first subscription.
    object.__setattr__(second_read.use.value, "invocation", first_invocation)
    with pytest.raises(ValueError):
        environment.capture_value(second).require_closed()


def test_direct_subscription_resolver_rejects_another_original_result_value():
    environment = execution("first = list[int]\nsecond = tuple[str]\n")
    first, second = (statement.value for statement in environment.module.module.body)
    first_read = subscription_read(environment, first)
    second_read = subscription_read(environment, second)
    with pytest.raises(ValueError):
        environment.kernel._subscription_result_value_resolution(
            first_read.use.value, (second_read, frozenset())
        ).require_closed()


@pytest.mark.parametrize(
    "defect",
    ("event_copy", "foreign_context", "foreign_graph"),
)
def test_subscription_binding_rejects_noncanonical_operation_ownership(defect):
    environment = execution("held = list[int]\n")
    read = subscription_read(environment)
    context, invocation = read.context, read.use.value.invocation
    other = execution(environment.module.source)
    if defect == "event_copy":
        invocation = replace(invocation)
    elif defect == "foreign_context":
        context = other.entry.context
    else:
        environment = other
    with pytest.raises(ValueError):
        NativeSubscriptionAuthority.for_subscription(
            environment, context, invocation
        ).require_closed()


def test_shared_source_projection_creates_separate_execution_bound_authorities():
    environment = execution("held = list[int]\n")
    other = SourceModuleExecution.from_source(environment.source)
    first, second = closed_alias(environment), closed_alias(other)
    assert first.authority.environment is environment
    assert second.authority.environment is other
    assert first.authority.invocation is second.authority.invocation
    assert first is not second
    assert not first.proves_same_object(second)


@pytest.mark.parametrize("query", ("capture_value", "require_subscription"))
def test_equal_but_copied_ast_cannot_authorize_subscription(query):
    environment = execution("held = list[int]\n")
    node = deepcopy(subscription_node(environment))
    with pytest.raises(ValueError):
        result = getattr(environment, query)(node)
        if result is not None:
            result.require_closed()


@pytest.mark.parametrize(
    "defect",
    (
        "copied_receiver",
        "copied_argument",
        "swapped_operands",
        "duplicated_receiver",
        "foreign_argument",
        "reversed_positions",
    ),
)
def test_original_operand_association_and_completion_order_are_required(defect):
    environment = execution("held = list[int]\n")
    read = subscription_read(environment)
    invocation = read.use.value.invocation
    receiver, argument = invocation.receiver_use, invocation.argument_use
    if defect == "copied_receiver":
        object.__setattr__(invocation, "receiver_use", replace(receiver))
    elif defect == "copied_argument":
        object.__setattr__(invocation, "argument_use", replace(argument))
    elif defect == "swapped_operands":
        object.__setattr__(invocation, "receiver_use", argument)
        object.__setattr__(invocation, "argument_use", receiver)
    elif defect == "duplicated_receiver":
        object.__setattr__(invocation, "argument_use", receiver)
    elif defect == "foreign_argument":
        other = execution(environment.module.source)
        object.__setattr__(
            invocation,
            "argument_use",
            subscription_read(other).use.value.invocation.argument_use,
        )
    else:
        object.__setattr__(receiver, "position", invocation.position)
    with pytest.raises(ValueError):
        NativeSubscriptionAuthority.for_subscription(
            environment, read.context, invocation
        ).require_closed()


@pytest.mark.parametrize(
    "source",
    (
        "if flag:\n    held = list[int]\n",
        "for item in items:\n    held = list[int]\n",
        "def deferred():\n    held = list[int]\n",
    ),
)
def test_unproved_conditional_repeated_and_deferred_activations_stay_open(source):
    environment = execution(source)
    result = environment.capture_value(subscription_node(environment))
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


@pytest.mark.parametrize(
    "expression", ("missing[int]", "list[missing]", "list[unknown()]")
)
def test_undefined_or_unproved_operands_cannot_gain_result_admission(expression):
    environment = execution(f"held = {expression}\n")
    result = environment.capture_value(subscription_node(environment))
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


def test_custom_subscription_is_not_executed_to_discover_its_result():
    source = (
        "class Custom:\n"
        "    def __class_getitem__(cls, item):\n"
        "        raise RuntimeError('custom subscription executed')\n"
        "held = Custom[int]\n"
    )
    environment = execution(source)
    result = environment.capture_value(subscription_node(environment))
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()
    with pytest.raises(RuntimeError, match="custom subscription executed"):
        authored_runtime(source)


def test_argument_hashing_is_not_executed_by_builtin_alias_construction():
    source = (
        "class Item:\n"
        "    def __hash__(self):\n"
        "        raise RuntimeError('hashing is a separate operation')\n"
        "item = Item()\nheld = list[item]\n"
    )
    environment = execution(source)
    original = environment.capture_value(environment.module.module.body[1].value)
    original.require_closed()
    result = closed_alias(environment)
    assert result.argument is original
    for _ in range(10):
        result.require_closed()
    with pytest.raises(ValueError):
        result.authority.require_inspected_result()
    runtime = authored_runtime(source)
    assert runtime["held"].__args__[0] is runtime["item"]
    with pytest.raises(RuntimeError, match="hashing is a separate operation"):
        hash(runtime["held"])


def test_classvar_execution_admission_does_not_imply_generic_alias_result():
    environment = execution("from typing import ClassVar\nheld = ClassVar[int]\n")
    node = subscription_node(environment)
    read = subscription_read(environment)
    operation = environment.source_operation(read.context, read.use.value.invocation)
    environment = SourceModuleExecution(
        supplied_entry(environment, frozenset((operation,)))
    )
    environment.require_subscription(node)
    read = subscription_read(environment)
    authority = environment.subscription_authority(
        read.context, read.use.value.invocation
    )
    with pytest.raises(ValueError):
        authority.require_inspected_result()
    result = environment.capture_value(node)
    result.require_closed()
    with pytest.raises(ValueError):
        _ = result.native_type
    with pytest.raises(ValueError):
        result.require_inspected_result()


def test_unadmitted_backend_cannot_inherit_builtin_alias_construction(monkeypatch):
    environment = execution("held = list[int]\n")
    monkeypatch.setattr(
        NativeCreationBackend,
        "current",
        classmethod(lambda cls: SpanOnlyCreationBackend()),
    )
    result = environment.capture_value(subscription_node(environment))
    assert isinstance(result, OpenCapturedReference)
    with pytest.raises(ValueError):
        result.require_closed()


def test_backend_requires_an_admitted_origin_not_just_a_native_type():
    with pytest.raises(ValueError):
        NativeCreationBackend.current().require_generic_alias_construction(
            NativeDeclaration(object)
        )


def test_closed_alias_does_not_grant_identity_release_or_plain_class_base():
    environment = execution("held = list[int]\n")
    result = closed_alias(environment)
    read = subscription_read(environment)
    assert not result.proves_same_object(result)
    assert not result.proves_same_object(result.origin)
    assert not result.proves_same_object(result.argument)
    with pytest.raises(ValueError):
        result.require_release()
    with pytest.raises(ValueError):
        result.require_plain_class_base(
            environment.kernel, read.context, read.use.position
        )
    with pytest.raises(ValueError):
        result.attribute_namespace(environment.initial, "__args__").require_closed()


def test_builtin_alias_as_class_base_remains_a_separate_unproved_operation():
    environment = execution("class Derived(list[int]): pass\n")
    closed_alias(environment)
    with pytest.raises(ValueError):
        environment.capture_definition(
            environment.module.module.body[0]
        ).require_closed()
