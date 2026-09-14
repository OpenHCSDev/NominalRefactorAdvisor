"""Completed proofs retain canonical source, cut, activation and cycle context."""

import ast
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceKernel,
    CapturedReferenceResolution,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import CompactBindingVisit
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(text):
    return SourceModuleExecution.from_module(
        ParsedModule(Path("memo.py"), "memo", False, ast.parse(text), text)
    )


def test_successful_original_value_query_is_not_reinterpreted(monkeypatch):
    environment = execution("answer = property\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    original = CapturedReferenceKernel._read_use
    calls = []

    def observed(self, use, context, pending):
        if use is read.use:
            calls.append(pending)
        return original(self, use, context, pending)

    monkeypatch.setattr(CapturedReferenceKernel, "_read_use", observed)
    first = environment.kernel.read(read)
    second = environment.kernel.read(read)
    assert first is second
    first.require_native_identity(NativeDeclaration(property))
    assert calls == [frozenset()]


def test_pending_binding_context_and_historical_cut_are_part_of_the_query():
    environment = execution("before = property\nproperty = object\nafter = property\n")
    first, assignment, last = environment.module.module.body
    before = environment.source.value_reads_by_node[first.value]
    after = environment.source.value_reads_by_node[last.value]
    mutation = environment.source.mutation_operation(assignment.targets[0]).event
    environment.kernel.read(after).require_native_identity(NativeDeclaration(object))
    rejected = environment.kernel._read_use(
        after.use,
        after.context,
        frozenset((CompactBindingVisit(after.context, mutation),)),
    )
    assert isinstance(rejected, OpenCapturedReference)
    environment.kernel.read(before).require_native_identity(NativeDeclaration(property))
    environment.kernel.read(after).require_native_identity(NativeDeclaration(object))


def test_cloned_equal_context_is_rejected_even_after_warm_value_query():
    environment = execution("answer = property\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    environment.kernel.read(read).require_closed()
    with pytest.raises(ValueError, match="canonical context"):
        environment.kernel.read(replace(read, context=replace(read.context)))


def test_foreign_event_registration_cannot_hit_original_success():
    environment = execution("answer = property\n")
    source = environment.source
    read = source.value_reads_by_node[environment.module.module.body[0].value]
    environment.kernel.read(read).require_closed()
    operation = source.source_operation(read.context, read.use)
    foreign = replace(read.use)
    corrupted = replace(
        source,
        operations=tuple(
            replace(site, event=foreign) if site is operation else site
            for site in source.operations
        ),
    )
    environment.entry.__dict__["source"] = corrupted
    with pytest.raises(ValueError):
        environment.kernel.read(replace(read, use=foreign))
    with pytest.raises(ValueError):
        environment.kernel.read(read)


def test_separate_source_activation_does_not_share_value_or_storage_cache():
    source = "registry={}\nregistry['key']='value'\ntail=property\n"
    first = execution(source)
    second = execution(source)
    first.capture(first.module.module.body[-1].value).require_closed()
    assert first.kernel._value_resolutions
    assert not second.kernel._value_resolutions
    assert first._closed_storage_operations
    assert not second._closed_storage_operations
    with pytest.raises(ValueError):
        second.kernel.read(next(iter(first.source.value_reads_by_node.values())))


def test_successful_storage_operation_reuses_existing_proof_owner(monkeypatch):
    environment = execution("registry={}\nregistry['key']='value'\n")
    target = environment.module.module.body[1].targets[0]
    original = CapturedReferenceResolution.require_item_write
    calls = []

    def observed(self, *arguments):
        calls.append(arguments)
        return original(self, *arguments)

    monkeypatch.setattr(CapturedReferenceResolution, "require_item_write", observed)
    environment.require_item_write(target)
    environment.require_item_write(target)
    assert len(calls) == 1


def test_tampered_storage_association_is_authenticated_before_cached_success():
    environment = execution("registry={}\nregistry['key']='value'\n")
    target = environment.module.module.body[1].targets[0]
    environment.require_item_write(target)
    operation = environment.source.mutation_operation(target)
    environment.source.operations_by_node[target] = (
        replace(operation, event=replace(operation.event)),
    )
    with pytest.raises(ValueError):
        environment.require_item_write(target)


def test_rejected_item_proof_is_never_recorded_as_completed_storage():
    environment = execution(
        "registry={}\nregistry['key']=[]\nregistry['key']='again'\n"
    )
    first, second = environment.module.module.body[1:]
    environment.require_item_write(first.targets[0])
    rejected = environment.source.mutation_operation(second.targets[0])
    for _ in range(2):
        with pytest.raises(ValueError):
            environment.require_item_write(second.targets[0])
        assert rejected not in environment._closed_storage_operations
        environment.require_item_write(first.targets[0])


def test_open_value_request_does_not_poison_later_complete_query(monkeypatch):
    environment = execution("answer=property\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[0].value
    ]
    original = CapturedReferenceKernel._read_use

    def unavailable(self, use, context, pending):
        return self._unproved_value_resolution((read, pending))

    monkeypatch.setattr(CapturedReferenceKernel, "_read_use", unavailable)
    assert isinstance(environment.kernel.read(read), OpenCapturedReference)
    assert not environment.kernel._value_resolutions
    monkeypatch.setattr(CapturedReferenceKernel, "_read_use", original)
    environment.kernel.read(read).require_native_identity(NativeDeclaration(property))


@pytest.mark.parametrize("warm", (False, True))
def test_forged_context_owner_index_cannot_authenticate_read_or_storage(warm):
    environment = execution("registry={}\nregistry['key']='value'\nanswer=property\n")
    read = environment.source.value_reads_by_node[
        environment.module.module.body[-1].value
    ]
    target = environment.module.module.body[1].targets[0]
    if warm:
        environment.kernel.read(read).require_closed()
        environment.require_item_write(target)
    original = read.context
    foreign = replace(original)
    environment.source.compact.flow_contexts_by_owner[original.flow.owner] = foreign
    with pytest.raises(ValueError):
        environment.kernel.read(replace(read, context=foreign))
    with pytest.raises(ValueError):
        environment.require_item_write(target)
    assert isinstance(
        environment.admit(foreign, read.use.position), OpenCapturedReference
    )


def test_warm_compact_context_indexes_rederive_after_pickle():
    environment = execution("class Owner: pass\nanswer=property\n")
    compact = environment.source.compact
    contexts = compact.flow_contexts
    assert all(
        compact.flow_contexts_by_identity[id(context)] is context
        for context in contexts
    )
    _ = compact.flow_contexts_by_owner
    _ = compact.value_captures_by_identity
    restored = pickle.loads(pickle.dumps(compact))
    assert "flow_contexts_by_identity" not in vars(restored)
    for context in restored.flow_contexts:
        assert restored.flow_contexts_by_identity[id(context)] is context
        assert restored.flow_contexts_by_owner[context.flow.owner] is context
