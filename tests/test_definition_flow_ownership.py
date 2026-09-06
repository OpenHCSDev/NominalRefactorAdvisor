"""Definition flow ownership retains source linkage, not runtime activation proof."""

import ast
import builtins
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    CompactClassDeclaration,
    CompactControlBranchKind,
    CompactDefinitionFlowOwner,
    CompactDefinitionTarget,
    CompactFlowOwnerKind,
    CompactFunctionDeclaration,
    CompactMutationKind,
    CompactNamespaceFlowOwner,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository


def _projection(source):
    return compact_product_flow_projection(
        ParsedModule(
            path=Path("definition_flow_ownership.py"),
            module_name="definition_flow_ownership",
            is_package_init=False,
            module=ast.parse(source),
            source=source,
        )
    )


def _definitions(projection):
    return tuple(
        (flow, mutation)
        for flow in projection.flows
        for mutation in flow.mutations
        if isinstance(mutation.target, CompactDefinitionTarget)
    )


def _require_shared_body_owners(projection):
    for _, mutation in _definitions(projection):
        assert mutation.kind is CompactMutationKind.DEFINITION
        target = mutation.target
        assert isinstance(target.owner, CompactDefinitionFlowOwner)
        (body,) = tuple(flow for flow in projection.flows if flow.owner is target.owner)
        assert body.owner.source_span == target.owner.source_span
        assert target.bound_name == target.owner.bound_name


def test_repeated_class_names_have_distinct_exact_source_owners():
    source = (
        "class Other: first = 1\n"
        "class Other: second = 2\n"
        "class Outer:\n"
        "    class Other: third = 3\n"
        "    class Other: fourth = 4\n"
    )
    projection = _projection(source)
    _require_shared_body_owners(projection)
    for qualname in ("Other", "Outer.Other"):
        first, second = (
            flow.owner for flow in projection.flows if flow.owner.qualname == qualname
        )
        assert first is not second
        assert first != second
        assert first.source_span != second.source_span
        assert isinstance(first, CompactClassDeclaration)
        assert first.kind is CompactFlowOwnerKind.CLASS_BODY
    expected = {
        SourceByteSpan.require_node(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
    }
    assert {
        mutation.target.owner.source_span for _, mutation in _definitions(projection)
    } == expected


def test_source_owner_index_distinguishes_same_qualname_and_reuses_contexts():
    projection = _projection("class Other: first = 1\nclass Other: second = 2\n")
    repository = CompactProductFlowRepository(
        product_projections=(projection,), class_projections=()
    )
    assert len(repository.flow_contexts) == len(projection.flow_contexts)
    assert all(
        actual is original
        for actual, original in zip(
            repository.flow_contexts, projection.flow_contexts, strict=True
        )
    )
    first, second = (
        context
        for context in projection.flow_contexts
        if context.flow.owner.qualname == "Other"
    )
    assert first.owner_symbol == second.owner_symbol
    assert projection.flow_contexts_by_owner[first.flow.owner] is first
    assert projection.flow_contexts_by_owner[second.flow.owner] is second
    assert first.owner_symbol not in repository.flow_contexts_by_owner_symbol
    module_context = projection.flow_contexts[0]
    assert repository.module_flow_contexts[projection.module_name] is module_context


def test_duplicate_actual_body_owner_is_not_selected_by_unique_owner_index():
    projection = _projection("class First: pass\nclass Second: pass\n")
    first, second = (
        flow for flow in projection.flows if flow.owner.kind.is_class_body_scope
    )
    duplicate = replace(projection, flows=(*projection.flows, first))
    assert first.owner not in duplicate.flow_contexts_by_owner
    selected = duplicate.flow_contexts_by_owner[second.owner]
    assert selected.flow is second
    assert any(selected is context for context in duplicate.flow_contexts)


def test_unified_definition_kind_dispatches_through_actual_owner_leaf(monkeypatch):
    projection = _projection("class Owner: pass\ndef callback(): pass\n")
    repository = CompactProductFlowRepository(
        product_projections=(projection,), class_projections=()
    )
    selections = []

    def select_class(self, symbol, binding):
        assert self is repository
        selections.append((CompactClassDeclaration, symbol, binding))
        return binding.target.owner

    def select_function(self, symbol, binding):
        assert self is repository
        selections.append((CompactFunctionDeclaration, symbol, binding))
        return binding.target.owner

    monkeypatch.setattr(
        CompactProductFlowRepository, "_selected_class_resolution", select_class
    )
    monkeypatch.setattr(
        CompactProductFlowRepository, "_selected_function_resolution", select_function
    )
    for _, mutation in _definitions(projection):
        owner = mutation.target.owner
        symbol = f"{projection.module_name}.{owner.qualname}"
        assert mutation.kind is CompactMutationKind.DEFINITION
        assert mutation.kind.resolve_definition(repository, symbol, mutation) is owner
        owner_type, selected_symbol, selected_binding = selections[-1]
        assert isinstance(owner, owner_type)
        assert selected_symbol == symbol
        assert selected_binding is mutation
    assert len(selections) == 2


def test_function_and_class_targets_share_existing_declaration_owners():
    source = (
        "@decorate\ndef outer(value=default()):\n"
        "    class Nested:\n"
        "        def method(self):\n"
        "            def local(): pass\n"
        "            return local\n"
        "    return Nested\n"
    )
    projection = _projection(source)
    _require_shared_body_owners(projection)
    owners = {
        flow.owner.qualname: flow.owner
        for flow in projection.flows
        if isinstance(flow.owner, CompactDefinitionFlowOwner)
    }
    assert isinstance(owners["outer"], CompactFunctionDeclaration)
    assert isinstance(owners["outer.Nested"], CompactClassDeclaration)
    assert isinstance(owners["outer.Nested.method.local"], CompactFunctionDeclaration)
    for owner in owners.values():
        if isinstance(owner, CompactFunctionDeclaration):
            assert owner.source_span is owner.execution.source_span


def test_definition_target_rejects_module_namespace_as_body_owner():
    ((_, mutation),) = _definitions(_projection("class Owner: pass\n"))
    module_owner = CompactNamespaceFlowOwner(CompactFlowOwnerKind.MODULE, "")
    with pytest.raises((TypeError, ValueError)):
        replace(mutation.target, owner=module_owner)


@pytest.mark.parametrize(
    "source",
    [
        "@decorate()\ndef Other(value=default()): pass\n",
        "@decorate()\nasync def Other(value=default()): pass\n",
        "@decorate()\nclass Other(base(), metaclass=meta(), option=keyword()): pass\n",
    ],
)
def test_explicit_header_evaluations_precede_retained_boundary_and_binding(source):
    projection = _projection(source)
    ((parent, mutation),) = _definitions(projection)
    target = mutation.target
    assert target.header_position.dominates(mutation.position)
    assert target.header_position != mutation.position
    for call in parent.calls:
        assert call.position.dominates(target.header_position)
    for decorator in target.decorator_uses:
        assert decorator.position.dominates(target.header_position)
    _require_shared_body_owners(projection)


def test_function_local_class_source_is_retained_without_eager_body_execution():
    source = (
        "def factory():\n"
        "    class Other: events.append('body')\n"
        "    return Other\n"
    )
    events = []
    namespace = {"events": events}
    exec(compile(source, "<trusted-local-class>", "exec"), namespace)
    assert events == []
    first = namespace["factory"]()
    second = namespace["factory"]()
    assert first is not second
    assert events == ["body", "body"]
    projection = _projection(source)
    definitions = _definitions(projection)
    ((parent, creation),) = (
        (flow, mutation)
        for flow, mutation in definitions
        if mutation.target.owner.qualname == "factory.Other"
    )
    assert parent.owner.qualname == "factory"
    assert all(
        mutation.target.owner.qualname != "factory.Other"
        for flow, mutation in definitions
        if flow.owner.kind.is_module_scope
    )
    _require_shared_body_owners(projection)


def test_loop_class_site_preserves_repetition_without_claiming_runtime_identity():
    source = (
        "results = []\n"
        "for index in range(2):\n"
        "    class Other: pass\n"
        "    results.append(Other)\n"
    )
    namespace = {}
    exec(compile(source, "<trusted-loop-class>", "exec"), namespace)
    assert namespace["results"][0] is not namespace["results"][1]
    projection = _projection(source)
    ((_, mutation),) = _definitions(projection)
    header = mutation.target.header_position
    assert header.branch_path == mutation.position.branch_path
    assert any(
        branch.kind is CompactControlBranchKind.LOOP_BODY
        for branch in header.branch_path
    )
    _require_shared_body_owners(projection)


def test_body_owner_sharing_survives_ast_free_pickle():
    projection = _projection(
        "def outer():\n"
        "    class Other:\n"
        "        def method(self): pass\n"
        "    class Other: pass\n"
    )
    restored = pickle.loads(pickle.dumps(projection))
    assert restored == projection
    _require_shared_body_owners(restored)
    pending = [restored]
    visited = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        assert not isinstance(value, ast.AST)
        if is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())


def test_native_builder_capture_precedes_base_rebinding_not_header_boundary():
    source = (
        "original = __builtins__['__build_class__']\n"
        "def captured(*args, **kwargs):\n"
        "    events.append('captured')\n"
        "    return original(*args, **kwargs)\n"
        "def replacement(*args, **kwargs):\n"
        "    events.append('replacement')\n"
        "    return int\n"
        "def base():\n"
        "    events.append('base')\n"
        "    __builtins__['__build_class__'] = replacement\n"
        "    return object\n"
        "__builtins__['__build_class__'] = captured\n"
        "class Other(base()): events.append('body')\n"
        "class Later: pass\n"
    )
    events = []
    namespace = {"events": events, "__builtins__": vars(builtins).copy()}
    exec(compile(source, "<trusted-builder-capture>", "exec"), namespace)
    assert events == ["base", "captured", "body", "replacement"]
    assert namespace["Other"] is not int
    assert namespace["Later"] is int
    projection = _projection(source)
    ((parent, mutation),) = (
        (flow, mutation)
        for flow, mutation in _definitions(projection)
        if mutation.target.owner.qualname == "Other"
    )
    (base_call,) = tuple(
        call for call in parent.calls if call.target.terminal_name == "base"
    )
    assert base_call.position.dominates(mutation.target.header_position)
    # This boundary follows argument evaluation; it is not LOAD_BUILD_CLASS.
    _require_shared_body_owners(projection)


def test_prepare_can_stop_before_retained_body_owner_ever_executes():
    source = (
        "class Meta(type):\n"
        "    @classmethod\n"
        "    def __prepare__(meta, name, bases):\n"
        "        events.append('prepare')\n"
        "        raise RuntimeError('no body entry')\n"
        "try:\n"
        "    class Other(metaclass=Meta): events.append('body')\n"
        "except RuntimeError:\n"
        "    events.append('caught')\n"
    )
    events = []
    namespace = {"events": events}
    exec(compile(source, "<trusted-preparation-stop>", "exec"), namespace)
    assert events == ["prepare", "caught"]
    assert "Other" not in namespace
    projection = _projection(source)
    assert any(
        mutation.target.owner.qualname == "Other"
        for _, mutation in _definitions(projection)
    )
    _require_shared_body_owners(projection)
