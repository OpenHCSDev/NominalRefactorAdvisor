"""Completion and native behavior are distinct source-entry premises."""

import ast
from dataclasses import FrozenInstanceError, replace
import subprocess
import sys
from typing import ClassVar

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceEffectsABC,
    CapturedReferenceResolution,
    CompletedSourceOperation,
)
from nominal_refactor_advisor.native_call import (
    CallAuthority,
    DefaultObjectConstruction,
    NativeDataclassFactoryCall,
)
from nominal_refactor_advisor.codemod import (
    ConvertManualRegistryToAutoregisterOperation,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseProvenance,
    NativeUseRequirement,
)
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_subscription import (
    BuiltinGenericAliasSubscription,
    ClassVariableSubscription,
    NativeSubscriptionAuthority,
)
from nominal_refactor_advisor.product_flow import CompactSubscription
from nominal_refactor_advisor.source_entry import (
    DeclaredNativeOperationBehavior,
    DeclaredOperationCompletion,
    ImportedSourceModuleEntryPremise,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_function_result import execution

SOURCE = "from typing import ClassVar\nheld = ClassVar[int]\n"


def subscriptions(environment):
    return tuple(
        operation
        for operation in environment.source.operations
        if isinstance(operation.event, CompactSubscription)
    )


def supplied_entry(environment, operations):
    """Declare the controlled fixture's completion and callback noninterference."""
    return replace(
        environment.entry,
        bindings=dict(environment.entry.initial_entries),
        declared_operation_conditions=tuple(
            DeclaredNativeOperationBehavior(
                operation, protocol=ClassVariableSubscription
            )
            for operation in operations
        ),
    )


def authority_for(environment, operation):
    return NativeSubscriptionAuthority.for_subscription(
        environment,
        environment.context_for_owner(operation.owner),
        operation.event,
    )


def controlled_native_execution(source):
    """A disposable isolated interpreter, not a claim about an existing cache."""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", source],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("argument", ("int", "None", '"not_yet_declared"', "list[int]"))
def test_explicit_original_behavior_condition_supports_controlled_native_invocation(
    argument,
):
    text = f"from typing import ClassVar\nheld = ClassVar[{argument}]\n"
    controlled_native_execution(text + "assert held.__origin__ is ClassVar\n")
    original = execution(text)
    operation = subscriptions(original)[-1]
    environment = SourceModuleExecution(
        supplied_entry(original, frozenset((operation,)))
    )
    authority = authority_for(environment, operation)
    assert authority.operation is operation
    NativeCreationBackend.current().require_classvar_binding(
        authority.inspected_argument
    )
    authority.require_closed()
    read = environment.source.value_reads_by_node[operation.node]
    environment.required_prefix(read.context, read.use.position)
    assert not environment._pending


def test_default_entry_does_not_infer_cached_completion_from_successful_binding():
    environment = execution(SOURCE)
    operation = subscriptions(environment)[0]
    authority = authority_for(environment, operation)
    assert not environment.entry.operation_conditions
    NativeCreationBackend.current().require_classvar_binding(
        authority.inspected_argument
    )
    with pytest.raises(ValueError):
        authority.require_closed()
    assert not environment._pending


def test_generic_effect_provider_default_does_not_authorize_operation_conditions():
    environment = execution(SOURCE)
    authority = authority_for(environment, subscriptions(environment)[0])
    assert authority.prefix.endpoint.frame.globals is environment.entry
    with pytest.raises(ValueError):
        CapturedReferenceEffectsABC.require_operation_completion(environment, authority)
    with pytest.raises(ValueError):
        CapturedReferenceEffectsABC.require_native_behavior(environment, authority)


def test_fresh_standard_loader_does_not_copy_supplied_conditions_from_source():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    explicit = supplied_entry(original, frozenset((operation,)))
    fresh = ImportedSourceModuleEntryPremise.from_source(explicit.source)
    assert fresh.source is explicit.source
    assert not fresh.operation_conditions
    with pytest.raises(ValueError):
        authority_for(SourceModuleExecution(fresh), operation).require_closed()


def test_class_body_completion_belongs_to_original_module_entry_and_actual_local_frame():
    text = "from typing import ClassVar\nclass Holder:\n    held = ClassVar[int]\n"
    controlled_native_execution(text + "assert Holder.held.__origin__ is ClassVar\n")
    original = execution(text)
    operation = subscriptions(original)[0]
    entry = supplied_entry(original, frozenset((operation,)))
    environment = SourceModuleExecution(entry)
    authority = authority_for(environment, operation)
    assert authority.context is not environment.source.module_context
    assert authority.prefix.endpoint.frame.globals is entry
    assert authority.prefix.endpoint.frame.locals is not entry
    authority.require_closed()


def test_supplied_conditions_are_frozen_input_facts_not_a_live_mutable_set():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    supplied = [
        DeclaredNativeOperationBehavior(operation, protocol=ClassVariableSubscription)
    ]
    entry = replace(
        original.entry,
        bindings=dict(original.entry.initial_entries),
        declared_operation_conditions=supplied,
    )
    supplied.clear()
    assert tuple(entry.operation_conditions) == (operation,)
    condition = entry.operation_conditions[operation]
    assert condition.operation is operation
    assert condition.protocol is ClassVariableSubscription
    with pytest.raises(TypeError):
        entry.operation_conditions[operation] = DeclaredOperationCompletion(operation)
    with pytest.raises(FrozenInstanceError):
        entry.operation_conditions = {}
    authority_for(SourceModuleExecution(entry), operation).require_closed()


@pytest.mark.parametrize(
    "corruption", ("operation", "event", "node", "source", "revision")
)
def test_entry_rejects_copied_foreign_and_stale_operation_facts(corruption):
    environment = execution(SOURCE)
    original = subscriptions(environment)[0]
    if corruption == "operation":
        selected = replace(original)
    elif corruption == "event":
        selected = replace(original, event=replace(original.event))
    elif corruption == "node":
        selected = replace(original, node=ast.parse("ClassVar[int]", mode="eval").body)
    elif corruption == "source":
        selected = subscriptions(execution(SOURCE))[0]
    else:
        selected = subscriptions(execution(SOURCE.replace("[int]", "[str]")))[0]
    with pytest.raises(ValueError):
        supplied_entry(environment, frozenset((selected,)))


def test_two_execution_views_of_same_actual_entry_can_use_its_condition():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = supplied_entry(original, frozenset((operation,)))
    first = SourceModuleExecution(entry)
    second = SourceModuleExecution(entry)
    assert first is not second
    first_authority = authority_for(first, operation)
    second_authority = authority_for(second, operation)
    assert (
        first_authority.prefix.endpoint.frame is second_authority.prefix.endpoint.frame
    )
    entry.require_operation_completion(first_authority)
    entry.require_operation_completion(second_authority)
    first_authority.require_closed()
    second_authority.require_closed()


def test_another_entry_with_same_source_and_island_cannot_supply_this_activation():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    first = supplied_entry(original, frozenset((operation,)))
    second = supplied_entry(original, frozenset((operation,)))
    authority = authority_for(SourceModuleExecution(second), operation)
    assert first.source is second.source
    assert first.initial is second.initial
    assert first.frame is not second.frame
    with pytest.raises(ValueError):
        first.require_operation_completion(authority)


def test_foreign_source_authority_cannot_use_an_equal_text_completion_condition():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = supplied_entry(original, frozenset((operation,)))
    other = execution(SOURCE)
    other_operation = subscriptions(other)[0]
    authority = authority_for(other, other_operation)
    with pytest.raises(ValueError):
        entry.require_operation_completion(authority)


@pytest.mark.parametrize("corruption", ("operation", "context", "cut"))
def test_forged_authority_cannot_bypass_entry_canonical_cut_validation(corruption):
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = supplied_entry(original, frozenset((operation,)))
    environment = SourceModuleExecution(entry)
    authority = authority_for(environment, operation)
    if corruption == "operation":
        object.__setattr__(authority, "operation", replace(operation))
    elif corruption == "context":
        object.__setattr__(authority, "context", replace(authority.context))
    else:
        earlier = environment.required_prefix(
            authority.context, operation.event.receiver_use.position
        )
        assert earlier.endpoint.position != operation.position
        object.__setattr__(authority, "prefix", earlier)
    with pytest.raises(ValueError):
        entry.require_operation_completion(authority)
    with pytest.raises(ValueError):
        entry.require_native_behavior(authority)


@pytest.mark.parametrize("argument", ("(int, str)", '"int["'))
def test_known_invalid_binding_cannot_be_overridden_by_a_behavior_condition(argument):
    original = execution(f"from typing import ClassVar\nheld = ClassVar[{argument}]\n")
    operation = subscriptions(original)[0]
    environment = SourceModuleExecution(
        supplied_entry(original, frozenset((operation,)))
    )
    authority = authority_for(environment, operation)
    with pytest.raises(ValueError):
        authority.require_closed()
    assert not environment._pending


def test_unproved_prior_prefix_cannot_be_overridden_by_a_later_behavior_condition():
    original = execution(
        "from typing import ClassVar\nunknown()\nheld = ClassVar[int]\n"
    )
    operation = subscriptions(original)[0]
    environment = SourceModuleExecution(
        supplied_entry(original, frozenset((operation,)))
    )
    with pytest.raises(ValueError):
        authority_for(environment, operation).require_closed()
    assert not environment._pending


def test_native_behavior_is_not_returned_identity_or_automatic_codemod_acceptance():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    environment = SourceModuleExecution(
        supplied_entry(original, frozenset((operation,)))
    )
    authority = authority_for(environment, operation)
    authority.require_closed()
    result = authority.result()
    assert result is authority
    result.require_closed()
    with pytest.raises(ValueError):
        result.require_native_identity(NativeDeclaration(ClassVar))
    requirement = NativeUseRequirement(
        ConvertManualRegistryToAutoregisterOperation,
        operation.node.value,
        (NativeDeclaration(ClassVar),),
        environment,
    )
    resolution = DeclaredNativeUseInvariants().resolve((requirement,))[0]
    assert resolution.provenance is NativeUseProvenance.CAPTURED_IDENTITY
    assert not resolution.provenance.is_admitted


def test_completed_result_does_not_acquire_unknown_object_protocols():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    environment = SourceModuleExecution(supplied_entry(original, (operation,)))
    result = authority_for(environment, operation).result()
    result.require_closed()
    for unsupported in (
        result.require_release,
        result.require_class_installation,
        result.require_native_scalar,
        lambda: result.native_type,
        lambda: result.require_inspected_result(),
    ):
        with pytest.raises(ValueError):
            unsupported()


def test_invocation_families_inherit_one_opaque_result_owner():
    for declaration in (
        CallAuthority,
        NativeDataclassFactoryCall,
        NativeSubscriptionAuthority,
    ):
        assert declaration.result is CompletedSourceOperation.result
        assert declaration.captured_result is CompletedSourceOperation.captured_result
    assert CallAuthority.require_closed is CapturedReferenceResolution.require_closed
    for declaration in (DefaultObjectConstruction, NativeDataclassFactoryCall):
        assert "violation" not in vars(declaration)
        assert declaration.violation is CompletedSourceOperation.violation


def test_completed_invocations_do_not_claim_the_same_cached_native_result():
    original = execution(
        "from typing import ClassVar\nfirst = ClassVar[int]\nsecond = ClassVar[int]\n"
    )
    operations = subscriptions(original)
    environment = SourceModuleExecution(supplied_entry(original, operations))
    first, second = (
        authority_for(environment, operation).result() for operation in operations
    )
    assert first.operation is not second.operation
    assert not first.proves_same_object(second)
    assert not second.proves_same_object(first)


def test_each_distinct_canonical_invocation_needs_its_own_behavior_fact():
    text = "from typing import ClassVar\nheld = (ClassVar[int], ClassVar[str])\n"
    controlled_native_execution(text + "assert len(held) == 2\n")
    original = execution(text)
    first, second = subscriptions(original)
    partial = SourceModuleExecution(supplied_entry(original, frozenset((first,))))
    authority_for(partial, first).require_closed()
    with pytest.raises(ValueError):
        authority_for(partial, second).require_closed()
    complete = SourceModuleExecution(
        supplied_entry(original, frozenset((first, second)))
    )
    authority_for(complete, first).require_closed()
    authority_for(complete, second).require_closed()


def test_completion_only_does_not_close_cached_native_behavior_or_its_result_cut():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = replace(
        original.entry,
        bindings=dict(original.entry.initial_entries),
        declared_operation_conditions=(DeclaredOperationCompletion(operation),),
    )
    environment = SourceModuleExecution(entry)
    authority = authority_for(environment, operation)
    entry.require_operation_completion(authority)
    NativeCreationBackend.current().require_classvar_binding(
        authority.inspected_argument
    )
    with pytest.raises(ValueError):
        entry.require_native_behavior(authority)
    with pytest.raises(ValueError):
        authority.require_closed()
    read = environment.source.value_reads_by_node[operation.node]
    with pytest.raises(ValueError):
        environment.required_prefix(read.context, read.use.position)
    assert not environment._pending


def test_behavior_condition_requires_its_actual_nominal_protocol():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = replace(
        original.entry,
        bindings=dict(original.entry.initial_entries),
        declared_operation_conditions=(
            DeclaredNativeOperationBehavior(
                operation, protocol=BuiltinGenericAliasSubscription
            ),
        ),
    )
    authority = authority_for(SourceModuleExecution(entry), operation)
    entry.require_operation_completion(authority)
    with pytest.raises(ValueError):
        entry.require_native_behavior(authority)
    with pytest.raises(ValueError):
        authority.require_closed()


def test_behavior_query_derives_protocol_from_its_actual_authority():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    entry = supplied_entry(original, (operation,))
    authority = authority_for(SourceModuleExecution(entry), operation)
    assert type(authority) is ClassVariableSubscription
    assert entry.operation_conditions[operation].protocol is type(authority)
    entry.require_native_behavior(authority)


def test_same_source_foreign_entry_cannot_supply_behavior():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    first = supplied_entry(original, (operation,))
    second = supplied_entry(original, (operation,))
    authority = authority_for(SourceModuleExecution(second), operation)
    with pytest.raises(ValueError):
        first.require_native_behavior(authority)


def test_one_operation_cannot_carry_competing_condition_authorities():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    with pytest.raises(ValueError):
        replace(
            original.entry,
            bindings=dict(original.entry.initial_entries),
            declared_operation_conditions=(
                DeclaredOperationCompletion(operation),
                DeclaredNativeOperationBehavior(
                    operation, protocol=ClassVariableSubscription
                ),
            ),
        )


@pytest.mark.parametrize("protocol", (object, "ClassVariableSubscription"))
def test_behavior_condition_requires_a_nominal_native_protocol(protocol):
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    with pytest.raises(TypeError):
        DeclaredNativeOperationBehavior(operation, protocol=protocol)


def test_raw_operation_is_not_a_declared_condition():
    original = execution(SOURCE)
    operation = subscriptions(original)[0]
    with pytest.raises(TypeError):
        replace(
            original.entry,
            bindings=dict(original.entry.initial_entries),
            declared_operation_conditions=(operation,),
        )
