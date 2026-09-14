"""Flow queries and native proofs consume one snapshot's original observations."""

import ast
from dataclasses import fields, replace
import pickle

import pytest

from nominal_refactor_advisor import class_index, product_flow
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def snapshot(source="chosen = property\n"):
    return CodemodSourceSnapshot.from_source_mapping({"/repo/subject.py": source})


@pytest.mark.parametrize("native_first", (False, True))
def test_queries_share_actual_events_regardless_of_demand_order(native_first):
    state = snapshot()
    module = state.parsed_modules[0]
    bindings = state.module_binding_proof
    if native_first:
        environment = bindings.native_reference_environment(module)
    repository = state.product_flow_repository
    environment = bindings.native_reference_environment(module)
    projection = repository.product_projections[0]
    assert projection is environment.source.compact
    assert bindings.source_projection(module) is environment.source
    for context in projection.flow_contexts:
        assert environment.context_for_owner(context.flow.owner) is context
    for operation in environment.source.operations:
        context = projection.flow_contexts_by_owner[operation.owner]
        assert environment.source_operation(context, operation.event) is operation


def test_structurally_equal_parsed_owners_do_not_share_execution_or_observations():
    module = snapshot().parsed_modules[0]
    copied = replace(module)
    assert module == copied and module is not copied
    bindings = RepositoryModuleBindingProof((module, copied))
    first = bindings.native_reference_environment(module)
    second = bindings.native_reference_environment(copied)
    assert first is not second
    assert first.source is not second.source
    assert first.source.module is module
    assert second.source.module is copied
    assert first.entry.frame is not second.entry.frame


def test_snapshot_retains_supplied_parsed_owner_across_all_lookups():
    module = snapshot().parsed_modules[0]
    state = CodemodSourceSnapshot.from_modules((module,))
    assert state.parsed_modules[0] is module
    assert state.parsed_module_for_source_path(module.file_path) is module
    assert state.module_binding_proof.modules[0] is module
    native = state.module_binding_proof.native_reference_environment(module)
    assert state.product_flow_repository.product_projections[0] is native.source.compact


@pytest.mark.parametrize(
    "member",
    (
        "    def value(self): return 1\n",
        "    @classmethod\n    def value(cls): return 1\n",
    ),
)
def test_indexed_snapshot_targets_belong_to_its_actual_source_parse(member):
    original = snapshot("class Owner:\n" + member)
    state = CodemodSourceSnapshot.from_indexed_sources(
        original.source_index, original.sources_by_file_path
    )
    module = state.parsed_modules[0]
    nodes = set(ast.walk(module.module))
    assert all(node in nodes for node in state.ast_target_nodes_by_id.values())
    environment = state.module_binding_proof.native_reference_environment(module)
    owner = next(
        node
        for node in state.ast_target_nodes_by_id.values()
        if isinstance(node, ast.ClassDef)
    )
    environment.require_class_creation(owner)
    assert (
        state.product_flow_repository.product_projections[0]
        is environment.source.compact
    )


def test_flow_query_does_not_activate_a_native_entry(monkeypatch):
    state = snapshot("unknown()\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation must not imply execution admission")

    monkeypatch.setattr(SourceModuleExecution, "from_source", forbidden)
    projection = state.product_flow_repository.product_projections[0]
    assert projection.flow_contexts
    assert state.module_binding_proof._native_executions == {}


def test_snapshot_does_not_recollect_flows_for_its_second_consumer(monkeypatch):
    state = snapshot()
    calls = []
    original = product_flow._ProductFlowCollection

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(product_flow, "_ProductFlowCollection", counted)
    repository = state.product_flow_repository
    module = state.parsed_modules[0]
    environment = state.module_binding_proof.native_reference_environment(module)
    assert len(calls) == 1
    assert calls[0] is module
    assert repository.product_projections[0] is environment.source.compact


def test_shared_observations_do_not_share_activation_proofs():
    state = snapshot("class Owner: pass\n")
    module = state.parsed_modules[0]
    source = state.module_binding_proof.source_projection(module)
    first = SourceModuleExecution.from_source(source)
    second = SourceModuleExecution.from_source(source)
    first.require_class_creation(module.module.body[0])
    assert first.source is second.source
    assert first.entry is not second.entry
    assert not second._closed_intervals
    assert first._closed_intervals


def test_next_virtual_snapshot_has_new_flow_and_activation_owners():
    first = snapshot()
    second = first.with_virtual_sources({"/repo/subject.py": "chosen = object\n"})
    before = first.product_flow_repository.product_projections[0]
    after = second.product_flow_repository.product_projections[0]
    assert before is not after
    first_module, second_module = first.parsed_modules[0], second.parsed_modules[0]
    first_environment = first.module_binding_proof.native_reference_environment(
        first_module
    )
    second_environment = second.module_binding_proof.native_reference_environment(
        second_module
    )
    assert first_environment.source.compact is before
    assert second_environment.source.compact is after
    with pytest.raises(ValueError, match="unique original operation"):
        operation = first_environment.source.operations[0]
        second_environment.source_operation(
            before.flow_contexts_by_owner[operation.owner], operation.event
        )


def test_compact_payload_stays_independent_of_task_source(monkeypatch):
    state = snapshot()
    projection = state.product_flow_repository.product_projections[0]
    assert "module" not in {field.name for field in fields(projection)}
    restored = pickle.loads(pickle.dumps(projection))
    assert restored == projection

    def forbidden(*args, **kwargs):
        raise AssertionError("Compact lookup must not recreate source evidence")

    monkeypatch.setattr(class_index, "source_product_flow_projection", forbidden)
    assert restored.flow_contexts
