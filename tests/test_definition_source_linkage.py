"""Exact retained definition parent/body linkage is not runtime activation."""

import ast
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import pickle
from types import CodeType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.product_flow import (
    CompactControlBranchKind,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository


def _module(source):
    return ParsedModule(
        path=Path("source_linkage.py"),
        module_name="source_linkage",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def _projection(source):
    return compact_product_flow_projection(_module(source))


def _rows(projection):
    return tuple(
        (context, mutation)
        for context in projection.flow_contexts
        for mutation in context.flow.mutations
        if mutation.kind.is_definition_binding
    )


def _row(projection, qualname):
    (row,) = tuple(
        row for row in _rows(projection) if row[1].target.owner.qualname == qualname
    )
    return row


def _repository(*projections):
    return CompactProductFlowRepository(
        product_projections=projections, class_projections=()
    )


def _replace_flow(projection, old, new):
    return replace(
        projection,
        flows=tuple(new if flow is old else flow for flow in projection.flows),
    )


def _require_actual_link(projection, parent, binding):
    selected_parent, selected_binding = projection.definition_sources_by_owner[
        binding.target.owner
    ]
    assert selected_parent is parent
    assert selected_binding is binding
    body = projection.flow_contexts_by_owner[binding.target.owner]
    assert body.flow.owner is binding.target.owner
    assert _repository(projection)._definition_body_context(parent, binding) is body
    return body


@pytest.mark.parametrize(
    "source",
    (
        "class Selected: pass\n",
        "def selected(): pass\n",
        "async def selected(): pass\n",
    ),
)
def test_parsed_top_level_definition_has_actual_module_parent_and_body(source):
    projection = _projection(source)
    ((parent, binding),) = _rows(projection)
    assert parent.flow.owner.kind.is_module_scope
    _require_actual_link(projection, parent, binding)


@pytest.mark.parametrize("header", ("class Repeated:", "def repeated():"))
def test_repeated_qualnames_retain_distinct_source_sites(header):
    projection = _projection(f"{header} first = 1\n{header} second = 2\n")
    first, second = _rows(projection)
    assert first[1].target.owner.qualname == second[1].target.owner.qualname
    assert first[1].target.owner.source_span != second[1].target.owner.source_span
    assert len(projection.definition_sources_by_owner) == 2
    assert _require_actual_link(projection, *first) is not _require_actual_link(
        projection, *second
    )


def test_nested_class_source_parent_is_not_its_lexical_lookup_parent():
    projection = _projection("class Outer:\n    class Inner: pass\n")
    parent, binding = _row(projection, "Outer.Inner")
    body = _require_actual_link(projection, parent, binding)
    assert parent.flow.owner.qualname == "Outer"
    assert "Outer" not in body.flow.lexical_scope_qualnames


def test_nested_function_class_method_and_local_function_keep_immediate_parents():
    source = (
        "def factory():\n"
        "    class Local:\n"
        "        events.append('body')\n"
        "        def method(self):\n"
        "            def callback(): pass\n"
        "            return callback\n"
        "    return Local\n"
    )
    projection = _projection(source)
    expected = {
        "factory": "",
        "factory.Local": "factory",
        "factory.Local.method": "factory.Local",
        "factory.Local.method.callback": "factory.Local.method",
    }
    for parent, binding in _rows(projection):
        assert parent.flow.owner.qualname == expected[binding.target.owner.qualname]
        _require_actual_link(projection, parent, binding)
    events = []
    namespace = {"events": events}
    exec(compile(source, "<trusted-deferred-source-parent>", "exec"), namespace)
    assert events == []
    first, second = namespace["factory"](), namespace["factory"]()
    assert first is not second
    assert events == ["body", "body"]


def test_loop_preserves_one_source_site_without_asserting_one_runtime_class():
    source = (
        "results = []\nfor index in range(2):\n"
        "    class Local: pass\n    results.append(Local)\n"
    )
    projection = _projection(source)
    ((parent, binding),) = _rows(projection)
    _require_actual_link(projection, parent, binding)
    assert any(
        branch.kind is CompactControlBranchKind.LOOP_BODY
        for branch in binding.position.branch_path
    )
    namespace = {}
    exec(compile(source, "<trusted-repeated-source-site>", "exec"), namespace)
    assert namespace["results"][0] is not namespace["results"][1]


@pytest.mark.parametrize("copy_event", (False, True))
def test_duplicate_equal_definition_rows_are_not_deduplicated(copy_event):
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    duplicate = replace(binding) if copy_event else binding
    assert duplicate == binding
    altered = _replace_flow(
        projection,
        parent.flow,
        replace(parent.flow, mutations=(*parent.flow.mutations, duplicate)),
    )
    assert binding.target.owner not in altered.definition_sources_by_owner


@pytest.mark.parametrize("invalidate_competing_parent", (False, True))
def test_all_parent_rows_count_before_ambiguous_contexts_are_filtered(
    invalidate_competing_parent,
):
    projection = _projection("class Holder: pass\nclass Child: pass\n")
    _, binding = _row(projection, "Child")
    holder = next(flow for flow in projection.flows if flow.owner.qualname == "Holder")
    competing = replace(holder, mutations=(binding,))
    altered = _replace_flow(projection, holder, competing)
    if invalidate_competing_parent:
        # The competing context becomes ambiguous, but its definition row must
        # still prevent manufacture of a unique remaining module parent.
        altered = replace(altered, flows=(*altered.flows, holder))
        assert holder.owner not in altered.flow_contexts_by_owner
    assert binding.target.owner not in altered.definition_sources_by_owner


def test_missing_body_cannot_supply_complete_source_linkage():
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    altered = replace(projection, flows=(parent.flow,))
    assert binding.target.owner not in altered.definition_sources_by_owner
    altered_parent = altered.flow_contexts[0]
    assert (
        _repository(altered)._definition_body_context(altered_parent, binding) is None
    )


@pytest.mark.parametrize("copy_body", (False, True))
def test_duplicate_equal_body_owners_cannot_supply_complete_linkage(copy_body):
    projection = _projection("class Child: pass\n")
    _, binding = _row(projection, "Child")
    body = projection.flow_contexts_by_owner[binding.target.owner].flow
    duplicate = replace(body) if copy_body else body
    altered = replace(projection, flows=(*projection.flows, duplicate))
    assert binding.target.owner not in altered.flow_contexts_by_owner
    assert binding.target.owner not in altered.definition_sources_by_owner


def test_duplicate_parent_with_only_one_definition_row_stays_unproved():
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    empty_duplicate = replace(parent.flow, mutations=())
    altered = replace(projection, flows=(*projection.flows, empty_duplicate))
    assert sum(row[1] is binding for row in _rows(altered)) == 1
    assert parent.flow.owner not in altered.flow_contexts_by_owner
    assert binding.target.owner not in altered.definition_sources_by_owner


def test_orphan_body_is_not_linked_by_name_or_final_binding():
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    altered = _replace_flow(projection, parent.flow, replace(parent.flow, mutations=()))
    assert binding.target.owner in altered.flow_contexts_by_owner
    assert binding.target.owner not in altered.definition_sources_by_owner


def test_original_decorator_capture_survives_header_and_later_name_rebinding():
    source = (
        "@decorate\n"
        "def selected(value=(decorate := replacement)): pass\n"
        "selected = replacement\n"
    )
    projection = _projection(source)
    parent, binding = _row(projection, "selected")
    _require_actual_link(projection, parent, binding)
    (capture,) = binding.target.decorator_uses
    assert capture.lexical_reference.root_name == "decorate"
    (header_write,) = tuple(
        mutation
        for mutation in parent.flow.mutations
        if mutation.target.bound_name == "decorate"
    )
    assert capture.position.dominates(header_write.position)
    assert header_write.position.dominates(binding.target.header_position)
    assert binding.position.dominates(parent.flow.mutations[-1].position)
    observed = []

    def original(function):
        observed.append(function)
        return function

    replacement = object()
    namespace = {"decorate": original, "replacement": replacement}
    exec(compile(source, "<trusted-definition-capture>", "exec"), namespace)
    assert len(observed) == 1
    assert observed[0].__defaults__ == (replacement,)
    assert namespace["selected"] is replacement


@pytest.mark.parametrize("warm_caches", (False, True))
def test_pickle_keeps_actual_context_event_and_body_sharing(warm_caches):
    projection = _projection(
        "def factory():\n    class Child:\n        def method(self): pass\n"
    )
    if warm_caches:
        assert len(projection.definition_sources_by_owner) == 3
    restored = pickle.loads(pickle.dumps(projection))
    assert restored == projection
    for parent, binding in _rows(restored):
        _require_actual_link(restored, parent, binding)
    pending, visited = [restored], set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        assert not isinstance(value, (ast.AST, CodeType))
        if is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
            pending.extend(getattr(value, "__dict__", {}).values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())


@pytest.mark.parametrize("copy_context,copy_binding", ((True, False), (False, True)))
def test_repository_requires_actual_context_and_event_not_equal_copies(
    copy_context,
    copy_binding,
):
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    candidate_context = replace(parent) if copy_context else parent
    candidate_binding = replace(binding) if copy_binding else binding
    assert candidate_context == parent
    assert candidate_binding == binding
    assert (
        _repository(projection)._definition_body_context(
            candidate_context, candidate_binding
        )
        is None
    )


def test_repository_does_not_choose_between_duplicate_module_projections():
    projection = _projection("class Child: pass\n")
    parent, binding = _row(projection, "Child")
    repository = _repository(projection, projection)
    assert projection.module_name not in repository.product_projections_by_module_name
    assert repository._definition_body_context(parent, binding) is None


def test_actual_multi_attribute_consumer_uses_exact_definition_body(monkeypatch):
    source = (
        "class Other:\n"
        "    class Holder:\n"
        "        @staticmethod\n"
        "        def saved(): return 37\n"
        "result = Other.Holder.saved()\n"
    )
    parsed = _module(source)
    projection = compact_product_flow_projection(parsed)
    repository = CompactProductFlowRepository(
        product_projections=(projection,),
        class_projections=CompactModuleClassProjectionFamily.collect_modules((parsed,)),
    )
    original = CompactProductFlowRepository._definition_body_context
    observed = []

    def observe(self, context, binding):
        result = original(self, context, binding)
        observed.append((context, binding, result))
        return result

    monkeypatch.setattr(
        CompactProductFlowRepository, "_definition_body_context", observe
    )
    (resolution,) = repository.function_call_resolutions
    assert resolution.resolved_call is not None
    assert (
        resolution.resolved_call.callee.identity.symbol
        == "source_linkage.Other.Holder.saved"
    )
    parent, binding = _row(projection, "Other")
    body = projection.flow_contexts_by_owner[binding.target.owner]
    assert any(
        context is parent and event is binding and result is body
        for context, event, result in observed
    )
    namespace = {}
    exec(compile(source, "<trusted-multi-attribute-consumer>", "exec"), namespace)
    assert namespace["result"] == 37
