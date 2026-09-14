"""Execution premises belong to the canonical source graph, not each consumer."""

import ast
import builtins
import dataclasses
from dataclasses import replace
from functools import cached_property
from pathlib import Path
import typing

import metaclass_registry
import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    InitialNativeIsland,
)
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors import _base as collector_runtime
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration
from nominal_refactor_advisor.native_declarations import (
    NativeDeclaration,
    NativeLanguageFeature,
)
from nominal_refactor_advisor.native_subscription import NativeSubscriptionAuthority
from nominal_refactor_advisor.product_flow import (
    CompactSubscription,
    source_product_flow_projection,
)
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_native_mro import (
    NativeClassBaseSubstitution,
    SourceNativeClassMro,
)


def module_at(path: Path, source: str) -> ParsedModule:
    return ParsedModule(path, path.stem, False, ast.parse(source), source)


def collector_entry(source):
    initial = InitialNativeIsland(
        (builtins, typing, collector_runtime, metaclass_registry)
    )
    return ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, initial.namespace_for_storage(vars(builtins))
    )


class CollectorEntryRepository(SourceProductFlowRepository):
    source_entry = staticmethod(collector_entry)


class CollectorEntrySnapshot(CodemodSourceSnapshot):
    @cached_property
    def product_flow_repository(self):
        return CollectorEntryRepository.from_modules(self.parsed_modules)


def test_standard_factory_owns_unchanged_initial_associations(tmp_path):
    module = module_at(tmp_path / "standard_entry.py", "result = property\n")
    source = source_product_flow_projection(module)
    entry = ImportedSourceModuleEntryPremise.from_source(source)
    assert entry.source is source
    assert entry.native_island.modules == (
        builtins,
        typing,
        dataclasses,
        NativeLanguageFeature.module,
    )
    assert entry.frame.globals is entry.frame.locals is entry
    assert entry.frame.builtins is entry.builtins
    assert entry.member("result") is None
    execution = SourceModuleExecution.from_source(source)
    assert execution.source is source
    assert type(execution.entry) is ImportedSourceModuleEntryPremise
    assert execution.entry.native_island.modules == entry.native_island.modules
    assert execution.entry is not entry
    assert execution.entry.frame is not entry.frame
    execution.capture(module.module.body[0].value).require_native_identity(
        NativeDeclaration(property)
    )


def test_snapshot_and_multiple_mro_consumers_share_actual_execution(tmp_path):
    module = module_at(
        tmp_path / "shared.py", "class Base: pass\nclass Child(Base): pass\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,))
    first = SourceNativeClassMro(snapshot)
    second = SourceNativeClassMro(snapshot)
    bindings = snapshot.product_flow_repository
    assert (
        first.bindings is second.bindings is snapshot.module_binding_proof is bindings
    )
    source = bindings.source_projection(module)
    assert bindings.sources[0] is source
    execution = bindings.native_reference_environment(module)
    assert first.bindings.native_reference_environment(module) is execution
    assert second.bindings.native_reference_environment(module) is execution
    assert execution.source is source
    assert (
        execution.kernel is first.bindings.native_reference_environment(module).kernel
    )
    assert (
        execution.entry.frame
        is second.bindings.native_reference_environment(module).entry.frame
    )
    child = snapshot.required_class_family_index.classes_by_symbol["shared.Child"]
    first_projection = first.for_source_class(child)
    second_projection = second.for_source_class(child)
    assert first_projection is not second_projection
    assert first_projection.declaration is second_projection.declaration is child
    entry = execution.class_entry(child.node)
    assert entry is first.bindings.native_reference_environment(module).class_entry(
        child.node
    )
    assert entry.execution is execution
    assert entry.completion_prefix.endpoint.frame is entry.frame


def test_substitution_shares_execution_but_not_counterfactual_mro(tmp_path):
    module = module_at(
        tmp_path / "substitution.py",
        "class Base: pass\nclass Root(Base): pass\nclass Leaf(Root): pass\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,))
    classes = snapshot.required_class_family_index.classes_by_symbol
    base, root, leaf = (
        classes[f"substitution.{name}"] for name in ("Base", "Root", "Leaf")
    )
    first = SourceNativeClassMro(snapshot)
    substitution = NativeClassBaseSubstitution(
        root, root.node.bases[0], NativeClassMroDeclaration(object)
    )
    second = replace(first, substitution=substitution)
    assert second.bindings is first.bindings is snapshot.module_binding_proof
    execution = first.bindings.native_reference_environment(module)
    assert second.bindings.native_reference_environment(module) is execution
    original = first.for_source_class(leaf)
    changed = second.for_source_class(leaf)
    base_projection = first.for_source_class(base)
    assert original is not changed
    assert base_projection in original.__mro__
    assert base_projection not in changed.__mro__
    assert first.for_source_class(leaf) is original
    assert second.for_source_class(leaf) is changed
    assert execution.class_entry(root.node).node.bases[0] is substitution.base
    assert execution.source.module.source == module.source


@pytest.mark.parametrize("changed", (False, True))
def test_separate_snapshot_has_its_own_source_entry_kernel_and_frame(tmp_path, changed):
    module = module_at(tmp_path / "revision.py", "class Root: pass\n")
    original = CodemodSourceSnapshot.from_modules((module,))
    following = (
        original.with_virtual_sources(
            {module.file_path: "class Root:\n    marker = None\n"}
        )
        if changed
        else CodemodSourceSnapshot.from_modules((module,))
    )
    left = SourceNativeClassMro(original)
    right = replace(left, context=following)
    next_module = following.parsed_module_for_source_path(module.file_path)
    before = left.bindings.native_reference_environment(module)
    after = right.bindings.native_reference_environment(next_module)
    assert right.bindings is following.module_binding_proof
    assert left.bindings is not right.bindings
    assert before is not after
    assert before.source is not after.source
    assert before.entry is not after.entry
    assert before.kernel is not after.kernel
    assert before.entry.frame is not after.entry.frame
    assert before.source.module is module
    assert after.source.module is next_module
    if not changed:
        assert next_module is module


def test_entry_factory_runs_once_for_all_queries_in_one_source_owner(tmp_path):
    module = module_at(tmp_path / "counted_entry.py", "result = object\n")
    received = []

    class CountingRepository(SourceProductFlowRepository):
        @staticmethod
        def source_entry(source):
            received.append(source)
            return ImportedSourceModuleEntryPremise.from_source(source)

    repository = CountingRepository.from_modules((module,))
    source = repository.sources[0]
    first = repository.native_reference_environment(module)
    for _ in range(3):
        assert repository.native_reference_environment(module) is first
    assert len(received) == 1
    assert received[0] is source is first.entry.source


def test_explicit_real_import_premise_is_reused_by_snapshot_and_mro(tmp_path):
    module = module_at(
        tmp_path / "native_entry.py",
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "result = CrossModuleCandidateDetector\n",
    )
    ordinary = CodemodSourceSnapshot.from_modules((module,))
    default_execution = ordinary.module_binding_proof.native_reference_environment(
        module
    )
    with pytest.raises(CapturedReferenceRejection) as raised:
        default_execution.require_import(module.module.body[0])
    assert raised.value.violation is CapturedReferenceViolation.UNADMITTED_IMPORT

    snapshot = CollectorEntrySnapshot.from_modules((module,))
    hierarchy = SourceNativeClassMro(snapshot)
    assert hierarchy.bindings is snapshot.product_flow_repository
    assert type(hierarchy.bindings) is CollectorEntryRepository
    execution = hierarchy.bindings.native_reference_environment(module)
    assert execution is snapshot.module_binding_proof.native_reference_environment(
        module
    )
    assert execution.source is snapshot.product_flow_repository.sources[0]
    execution.require_import(module.module.body[0])
    execution.capture(module.module.body[-1].value).require_native_identity(
        NativeDeclaration(collector_runtime.CrossModuleCandidateDetector)
    )
    assert (
        execution.entry.native_island.module(collector_runtime.__name__).value
        is collector_runtime
    )
    with pytest.raises(CapturedReferenceRejection):
        default_execution.require_import(module.module.body[0])


@pytest.mark.parametrize("kind", ("generic", "metaclass"))
def test_import_premise_does_not_admit_generic_or_metaclass_activation(tmp_path, kind):
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "class Owner(CrossModuleCandidateDetector[int]): pass\n"
        if kind == "generic"
        else "from metaclass_registry import AutoRegisterMeta\n"
        "class Owner(metaclass=AutoRegisterMeta): pass\n"
    )
    module = module_at(tmp_path / "native_activation.py", source)
    snapshot = CollectorEntrySnapshot.from_modules((module,))
    hierarchy = SourceNativeClassMro(snapshot)
    execution = hierarchy.bindings.native_reference_environment(module)
    import_node, owner = module.module.body
    execution.require_import(import_node)
    if kind == "generic":
        base = owner.bases[0]
        execution.capture(base.value).require_native_identity(
            NativeDeclaration(collector_runtime.CrossModuleCandidateDetector)
        )
        operation = execution.source.node_operation(base, CompactSubscription)
        with pytest.raises(ValueError):
            NativeSubscriptionAuthority.for_subscription(
                execution,
                execution.context_for_owner(operation.owner),
                operation.event,
            )
    else:
        execution.capture(owner.keywords[0].value).require_native_identity(
            NativeDeclaration(metaclass_registry.AutoRegisterMeta)
        )
    with pytest.raises(ValueError):
        execution.require_class_creation(owner)


@pytest.mark.parametrize("foreign", (False, True))
def test_entry_factory_cannot_install_a_copied_or_foreign_source(tmp_path, foreign):
    module = module_at(tmp_path / "exact_entry.py", "result = object\n")
    other = module_at(tmp_path / "foreign_entry.py", "result = object\n")
    received = []

    class WrongSourceRepository(SourceProductFlowRepository):
        @staticmethod
        def source_entry(source):
            received.append(source)
            supplied = (
                source_product_flow_projection(other) if foreign else replace(source)
            )
            return ImportedSourceModuleEntryPremise.from_source(supplied)

    repository = WrongSourceRepository.from_modules((module,))
    source = repository.source_projection(module)
    for _ in range(2):
        with pytest.raises(ValueError, match="different canonical projection"):
            repository.native_reference_environment(module)
        assert not repository._native_executions
    assert len(received) == 2
    assert all(actual is source for actual in received)
    assert repository.source_projection(module) is source
