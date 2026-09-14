"""Original operations and definitions share activation, not completion state."""

import ast
from copy import deepcopy
from dataclasses import fields, replace
from functools import cached_property
from inspect import isabstract

import pytest

from nominal_refactor_advisor.captured_reference import (
    InitialNativeFrame,
    SourceActivationAuthorityABC,
    SourceOperationAuthority,
)
from nominal_refactor_advisor.native_declarations import (
    NativeDeclaration,
    NativeDeclarationFamily,
)
from nominal_refactor_advisor.native_subscription import ClassVariableSubscription
from nominal_refactor_advisor.source_entry import (
    DeclaredNativeOperationBehavior,
    DeclaredOperationCompletion,
)
from nominal_refactor_advisor.source_execution import (
    SourceClassEntry,
    SourceDefinitionEntry,
    SourceModuleExecution,
)
from test_documentation_store import execution

SOURCE = "class Root:\n    pass\nclass Leaf(Root):\n    pass\n"


class ConditionedClassEntry(SourceClassEntry, NativeDeclarationFamily):
    """Test-only ordinary class protocol requiring an explicit behavior premise."""

    native_declarations = (NativeDeclaration(type),)

    def require_preparation(self):
        super().require_preparation()
        self.execution.require_native_behavior(self)

    @cached_property
    def completed(self):
        _ = super().completed
        self.execution.require_native_behavior(self)


def conditioned_execution(text=SOURCE, *, omitted=(), completion_only=False):
    original = execution(text)
    nodes = tuple(
        node
        for node in ast.walk(original.module.module)
        if isinstance(node, ast.ClassDef)
    )
    operations = tuple(
        original.source.definition_operation(node)
        for node in nodes
        if node.name not in omitted
    )
    conditions = tuple(
        (
            DeclaredOperationCompletion(operation)
            if completion_only
            else DeclaredNativeOperationBehavior(
                operation, protocol=ConditionedClassEntry
            )
        )
        for operation in operations
    )
    environment = SourceModuleExecution(
        replace(
            original.entry,
            bindings=dict(original.entry.initial_entries),
            declared_operation_conditions=conditions,
        )
    )
    # Install the test-only family through the same canonical cache. Production
    # continues to select only SourceClassEntry; no native registration is admitted.
    for node in nodes:
        environment._class_entries[node] = ConditionedClassEntry(environment, node)
    return environment, tuple(environment.class_entry(node) for node in nodes)


def test_common_authority_has_no_parallel_operation_or_activation_storage():
    assert isabstract(SourceActivationAuthorityABC)
    assert issubclass(SourceOperationAuthority, SourceActivationAuthorityABC)
    assert issubclass(SourceDefinitionEntry, SourceActivationAuthorityABC)
    assert tuple(field.name for field in fields(SourceOperationAuthority)) == (
        "environment",
        "operation",
    )
    assert tuple(field.name for field in fields(SourceDefinitionEntry)) == (
        "execution",
        "node",
    )
    assert tuple(field.name for field in fields(ConditionedClassEntry)) == (
        "execution",
        "node",
    )
    for owner in (SourceOperationAuthority, SourceDefinitionEntry):
        assert "require_condition_activation" not in vars(owner)
        assert "activation_context" in vars(owner)
        assert "activation_position" in vars(owner)
        assert "activation_prefix" in vars(owner)


def test_root_and_implicit_leaf_reuse_one_condition_at_their_actual_definition():
    environment, entries = conditioned_execution()
    assert len(environment.entry.operation_conditions) == len(entries) == 2
    for entry in entries:
        environment.require_native_behavior(entry)
        assert entry.environment is environment
        assert entry.activation_context is entry.parent_context
        assert entry.activation_prefix is entry.parent_prefix
        assert entry.activation_position == entry.definition.target.header_position
        assert entry.activation_position != entry.operation.position
        assert entry.context is not entry.activation_context
        assert entry.result().entry is entry
        assert entry.completion_prefix.endpoint.context is entry.context
        assert entry.completion_prefix.endpoint.frame is entry.frame
        assert entry.completion_prefix.endpoint.position is None
        assert environment.class_entry(entry.node) is entry
        assert environment.entry.operation_conditions[entry.operation].protocol is type(
            entry
        )
    assert not environment._pending


@pytest.mark.parametrize("missing", ("Root", "Leaf"))
def test_each_definition_requires_its_own_condition(missing):
    environment, entries = conditioned_execution(omitted=(missing,))
    selected = next(entry for entry in entries if entry.node.name == missing)
    with pytest.raises(ValueError, match="explicit entry condition"):
        environment.require_native_behavior(selected)
    assert "completed" not in vars(selected)


def test_completion_premise_does_not_become_native_behavior():
    environment, (entry,) = conditioned_execution(
        "class Root: pass\n", completion_only=True
    )
    environment.require_operation_completion(entry)
    with pytest.raises(ValueError, match="Completion alone"):
        environment.require_native_behavior(entry)


def test_wrong_native_protocol_cannot_authorize_test_class_family():
    environment, (entry,) = conditioned_execution("class Root: pass\n")
    wrong = SourceModuleExecution(
        replace(
            environment.entry,
            bindings=dict(environment.entry.initial_entries),
            declared_operation_conditions=(
                DeclaredNativeOperationBehavior(
                    entry.operation, protocol=ClassVariableSubscription
                ),
            ),
        )
    )
    actual = ConditionedClassEntry(wrong, entry.node)
    with pytest.raises(ValueError, match="another protocol"):
        wrong.require_native_behavior(actual)


def test_condition_does_not_admit_actual_wrong_constructor_inputs():
    environment, (entry,) = conditioned_execution("class Root(metaclass=int): pass\n")
    # This authenticates the supplied claim's activation only. Actual native
    # construction still rejects its unsupported metaclass independently.
    environment.require_native_behavior(entry)
    with pytest.raises(ValueError, match="unproved construction hooks"):
        entry.result()


@pytest.mark.parametrize("text", ("", "# different revision\n"))
def test_equal_or_edited_source_cannot_reuse_another_activation_authority(text):
    _, (entry,) = conditioned_execution("class Root: pass\n")
    foreign, _ = conditioned_execution(text + "class Root: pass\n")
    with pytest.raises(ValueError, match="different source activation"):
        foreign.require_native_behavior(entry)


def test_same_graph_and_native_island_still_require_actual_entry_frame():
    environment, (entry,) = conditioned_execution("class Root: pass\n")
    foreign = SourceModuleExecution(
        replace(
            environment.entry,
            bindings=dict(environment.entry.initial_entries),
            declared_operation_conditions=tuple(
                environment.entry.operation_conditions.values()
            ),
        )
    )
    assert foreign.source is environment.source
    assert foreign.initial is environment.initial
    with pytest.raises(ValueError, match="original activation cut"):
        foreign.require_native_behavior(entry)
    # A separately supplied condition in the other entry can support its own
    # original activation; shared graph identity does not forbid that use.
    own = ConditionedClassEntry(foreign, entry.node)
    foreign.require_native_behavior(own)


@pytest.mark.parametrize(
    "corruption", ("copied-node", "other-node", "copied-operation")
)
def test_cached_operation_does_not_hide_forged_definition_node(corruption):
    environment, entries = conditioned_execution()
    entry, other = entries
    operation = entry.operation
    _ = entry.definition
    _ = entry.parent_context
    _ = entry.context
    _ = entry.parent_prefix
    if corruption == "copied-node":
        object.__setattr__(entry, "node", deepcopy(entry.node))
    elif corruption == "other-node":
        object.__setattr__(entry, "node", other.node)
    else:
        object.__setattr__(entry, "operation", replace(operation))
    with pytest.raises(ValueError):
        environment.require_native_behavior(entry)


def test_cached_definition_must_still_be_the_canonical_operations_event():
    environment, entries = conditioned_execution()
    entry, other = entries
    original = entry.operation
    object.__setattr__(entry, "definition", other.definition)
    # Without the definition-to-event join, these derived projections agree
    # with one another at the wrong header cut and could authenticate it.
    assert entry.activation_position == other.definition.target.header_position
    assert entry.operation is original
    with pytest.raises(ValueError, match="Definition requires its original"):
        environment.require_native_behavior(entry)


def test_subclass_protocol_cannot_reuse_parent_protocol_condition():
    class DifferentProtocol(ConditionedClassEntry):
        pass

    environment, (entry,) = conditioned_execution("class Root: pass\n")
    refined = DifferentProtocol(environment, entry.node)
    with pytest.raises(ValueError, match="another protocol"):
        environment.require_native_behavior(refined)


@pytest.mark.parametrize(
    "definition", ("class Root: pass", "def root(): pass", "async def root(): pass")
)
def test_definition_activation_projection_does_not_activate_its_body(definition):
    original = execution(definition + "\n")
    node = original.module.module.body[0]
    operation = original.source.definition_operation(node)
    environment = SourceModuleExecution(
        replace(
            original.entry,
            bindings=dict(original.entry.initial_entries),
            declared_operation_conditions=(DeclaredOperationCompletion(operation),),
        )
    )
    owner = SourceDefinitionEntry(environment, node)
    environment.require_operation_completion(owner)
    assert owner.activation_context is environment.entry.context
    assert owner.activation_position == operation.event.target.header_position
    assert owner.activation_prefix.endpoint.frame.globals is environment.entry
    assert not environment._class_entries
    assert not environment._pending


@pytest.mark.parametrize("swapped", (0, 1))
def test_class_and_function_cached_definitions_cannot_exchange_source_nodes(swapped):
    original = execution("class Root: pass\ndef root(): pass\n")
    nodes = original.module.module.body
    operations = tuple(original.source.definition_operation(node) for node in nodes)
    environment = SourceModuleExecution(
        replace(
            original.entry,
            bindings=dict(original.entry.initial_entries),
            declared_operation_conditions=tuple(
                DeclaredOperationCompletion(op) for op in operations
            ),
        )
    )
    owner = SourceDefinitionEntry(environment, nodes[swapped])
    _ = owner.operation
    _ = owner.definition
    _ = owner.context
    _ = owner.parent_context
    object.__setattr__(owner, "node", nodes[1 - swapped])
    with pytest.raises(ValueError, match="Definition requires its original"):
        environment.require_operation_completion(owner)


@pytest.mark.parametrize(
    "corruption", ("definition-cut", "completed-cut", "other-context", "other-globals")
)
def test_condition_requires_original_declared_cut_and_frame(corruption):
    environment, entries = conditioned_execution()
    entry, other = entries
    original = entry.parent_prefix.endpoint
    if corruption == "definition-cut":
        substituted = replace(original, position=entry.operation.position)
    elif corruption == "completed-cut":
        substituted = replace(original, position=None)
    elif corruption == "other-context":
        substituted = replace(original, context=other.context)
    else:
        foreign, _ = conditioned_execution("class Elsewhere: pass\n")
        substituted = replace(
            original,
            frame=InitialNativeFrame(
                original.frame.locals, foreign.entry, original.frame.builtins
            ),
        )
    object.__setattr__(entry, "native_frame_prefix", substituted)
    with pytest.raises(ValueError, match="original activation cut"):
        environment.require_native_behavior(entry)


def test_preparation_condition_does_not_close_unknown_body_or_cache_failure():
    environment, (entry,) = conditioned_execution(
        "class Root:\n    member = unknown()\n"
    )
    environment.require_native_behavior(entry)
    assert entry.frame.locals is entry
    assert "completion_prefix" not in vars(entry)
    for _ in range(2):
        with pytest.raises(ValueError):
            _ = entry.completed
        assert "completed" not in vars(entry)
    assert not environment._pending


def test_native_body_and_installed_result_remain_separate_with_class_decorator():
    environment, (entry,) = conditioned_execution("@object\nclass Root: pass\n")
    environment.require_native_behavior(entry)
    _ = entry.completed
    with pytest.raises(ValueError, match="decorator result"):
        entry.result()


@pytest.mark.parametrize(
    "text",
    (
        "class Root: pass\n",
        "class Root: pass\nclass Leaf(Root): pass\n",
        "if True:\n    class Root: pass\n",
        "class Outer:\n    class Inner: pass\n",
    ),
)
def test_existing_ordinary_classes_need_no_new_supplied_condition(text):
    environment = execution(text)
    assert not environment.entry.operation_conditions
    for node in ast.walk(environment.module.module):
        if isinstance(node, ast.ClassDef):
            entry = environment.class_entry(node)
            assert type(entry) is SourceClassEntry
            entry.result().require_closed()


@pytest.mark.parametrize(
    "text",
    (
        "if object:\n    class Root: pass\n",
        "for item in (1, 2):\n    class Root: pass\n",
        "def outer():\n    class Root: pass\n",
    ),
)
def test_condition_does_not_activate_unproved_conditional_repeated_or_function_body(
    text,
):
    environment, (entry,) = conditioned_execution(text)
    with pytest.raises(ValueError):
        entry.result()
