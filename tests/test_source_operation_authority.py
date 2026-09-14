"""Native calls and source construction share original operation admission."""

import ast
import cProfile
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from nominal_refactor_advisor import captured_reference
from nominal_refactor_advisor.captured_reference import CapturedReferenceKernel
from nominal_refactor_advisor.native_call import (
    CallAuthority,
    CopiedNativeNamespace,
    NativeCallAuthority,
    NativeDictCopyCall,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactFunctionCall
from nominal_refactor_advisor.source_execution import (
    SourceModuleExecution,
    SourceObjectConstruction,
)
from test_source_function_result import execution, native


def last_call(environment):
    return tuple(
        operation
        for operation in environment.source.operations
        if isinstance(operation.event, CompactFunctionCall)
    )[-1]


def call_count(profile, function):
    return sum(
        entry.callcount
        for entry in profile.getstats()
        if entry.code is function.__code__
    )


def test_shared_authority_owns_the_prelude_without_redeclared_call_state():
    owner = captured_reference.SourceOperationAuthority
    assert issubclass(CallAuthority, owner)
    assert issubclass(SourceObjectConstruction, owner)
    assert tuple(field.name for field in fields(owner)) == (
        "environment",
        "operation",
    )
    assert tuple(field.name for field in fields(CallAuthority)) == (
        "environment",
        "operation",
    )
    for member in ("context", "prefix", "__post_init__"):
        assert member in vars(owner)
        assert member not in vars(CallAuthority)
    assert "require_operation_kind" in owner.__abstractmethods__
    assert "require_operation_kind" in vars(CallAuthority)
    with pytest.raises(TypeError, match="abstract"):
        owner(None, None)


@pytest.mark.parametrize(
    "text",
    (
        "held = dict({}, alpha=object)\n",
        "class Owner:\n    held = dict({}, alpha=object)\n",
    ),
)
def test_actual_native_call_keeps_original_operation_context_and_prefix(text):
    native(text)
    environment = execution(text)
    operation = last_call(environment)
    copied = CopiedNativeNamespace.from_call(environment, operation)
    authority = copied
    assert type(authority) is NativeDictCopyCall
    assert isinstance(authority, captured_reference.SourceOperationAuthority)
    assert authority.environment is environment
    assert authority.operation is operation
    assert authority.node is operation.node
    assert authority.call is operation.event
    assert authority.context is environment.context_for_owner(operation.owner)
    assert authority.prefix is environment.kernel._admitted_prefix(
        authority.context, operation.position
    )
    assert environment.source_operation(authority.context, authority.call) is operation
    assert authority.declaration.declaration is dict
    copied.require_closed()
    copied.member("alpha").require_native_identity(NativeDeclaration(object))


def test_source_object_construction_keeps_the_same_shared_owner_and_opaque_identity():
    text = "class Payload: pass\nheld = Payload()\n"
    native(text + "assert type(held) is Payload\n")
    environment = execution(text)
    operation = last_call(environment)
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    assert type(authority) is SourceObjectConstruction
    assert isinstance(authority, captured_reference.SourceOperationAuthority)
    assert authority.operation is operation
    assert authority.result() is authority
    authority.require_closed()
    authority.require_class_installation()
    with pytest.raises(ValueError):
        authority.require_release()
    with pytest.raises(ValueError):
        authority.source_definition()


@pytest.mark.parametrize(
    "corruption",
    ("operation", "node", "event", "owner", "other_owner", "foreign_source"),
)
def test_equal_or_foreign_operation_components_cannot_supply_admission(corruption):
    text = "def deferred(): pass\nheld = dict(alpha=object)\n"
    environment = execution(text)
    operation = last_call(environment)
    if corruption == "operation":
        selected = replace(operation)
    elif corruption == "node":
        selected = replace(operation, node=deepcopy(operation.node))
    elif corruption == "event":
        selected = replace(operation, event=replace(operation.event))
    elif corruption == "owner":
        selected = replace(operation, owner=replace(operation.owner))
    elif corruption == "other_owner":
        definition = environment.definition_operation(environment.module.module.body[0])
        selected = replace(operation, owner=definition.event.target.owner)
    else:
        selected = last_call(execution(text))
    assert selected is not operation
    profile = cProfile.Profile()
    with pytest.raises(ValueError):
        profile.runcall(NativeDictCopyCall, environment, selected)
    assert call_count(profile, CapturedReferenceKernel._admitted_prefix) == 0


@pytest.mark.parametrize("context_kind", ("copied", "foreign"))
def test_factory_rejects_noncanonical_context_before_native_selection(context_kind):
    text = "held = dict(alpha=object)\n"
    environment = execution(text)
    operation = last_call(environment)
    original = environment.context_for_owner(operation.owner)
    selected = (
        replace(original)
        if context_kind == "copied"
        else execution(text).source.module_context
    )
    with pytest.raises(ValueError, match="canonical context"):
        NativeCallAuthority.for_call(environment, selected, operation.event)


@pytest.mark.parametrize("copied", (False, True))
def test_kind_refusal_precedes_an_unadmitted_prefix_but_follows_provenance(copied):
    environment = execution("unbound\nheld = object\n")
    operation = environment.source.mutation_operation(
        environment.module.module.body[-1].targets[0]
    )
    selected = replace(operation) if copied else operation
    profile = cProfile.Profile()
    if copied:
        with pytest.raises(ValueError, match="canonical source operation"):
            profile.runcall(NativeDictCopyCall, environment, selected)
    else:
        with pytest.raises(TypeError, match="actual invocation operation"):
            profile.runcall(NativeDictCopyCall, environment, selected)
    assert call_count(profile, CapturedReferenceKernel._admitted_prefix) == 0
    # This is a real failed prefix, not a permissive test effects provider.
    with pytest.raises(ValueError):
        environment.required_prefix(
            environment.context_for_owner(operation.owner), operation.position
        )


@pytest.mark.parametrize("corruption", ("duplicate", "foreign_graph", "owner"))
def test_recorded_call_still_requires_unique_original_graph_and_owner(corruption):
    environment = execution("held = dict(alpha=object)\n")
    source = environment.source
    operation = last_call(environment)
    if corruption == "duplicate":
        source = replace(source, operations=(*source.operations, operation))
    elif corruption == "foreign_graph":
        source = replace(source, compact=execution(source.module.source).source.compact)
    else:
        selected = replace(operation, owner=replace(operation.owner))
        source = replace(
            source,
            operations=tuple(
                selected if site is operation else site for site in source.operations
            ),
        )
        operation = selected
    environment = SourceModuleExecution.from_source(source)
    with pytest.raises(ValueError):
        NativeDictCopyCall(environment, operation)


def test_prefix_is_admitted_once_and_reused_by_the_bound_authority():
    environment = execution("held = dict(alpha=object)\n")
    operation = last_call(environment)
    initial_profile = cProfile.Profile()
    authority = initial_profile.runcall(NativeDictCopyCall, environment, operation)
    prefix_getter = captured_reference.SourceOperationAuthority.prefix.func
    assert call_count(initial_profile, prefix_getter) == 1
    prefix = authority.prefix
    profile = cProfile.Profile()

    def read_again():
        for _ in range(12):
            assert authority.prefix is prefix
            assert authority.context is environment.source.module_context

    profile.runcall(read_again)
    assert call_count(profile, CapturedReferenceKernel._admitted_prefix) == 0
    assert not environment._pending


def test_shared_source_operations_do_not_share_execution_prefixes():
    left = execution("class Payload: pass\nheld = Payload()\n")
    right = SourceModuleExecution.from_source(left.source)
    operation = last_call(left)
    first = SourceObjectConstruction(left, operation)
    second = SourceObjectConstruction(right, operation)
    first.require_closed()
    second.require_closed()
    assert first.operation is second.operation
    assert first.context is second.context
    assert first.prefix is not second.prefix
    assert first.prefix.endpoint.frame is not second.prefix.endpoint.frame
    assert first.environment.kernel is not second.environment.kernel
    assert not first.proves_same_object(second)


def test_actual_call_with_unclosed_prior_execution_does_not_gain_admission():
    environment = execution("unbound\nheld = dict(alpha=object)\n")
    operation = last_call(environment)
    assert isinstance(operation.node, ast.Call)
    with pytest.raises(ValueError):
        NativeDictCopyCall(environment, operation)
    assert not environment._pending
