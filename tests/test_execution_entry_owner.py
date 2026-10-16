"""Supplied entry premises own execution provenance and initial associations."""

import ast
import builtins
import dataclasses
import inspect
from dataclasses import fields, replace
from pathlib import Path

import pytest

import nominal_refactor_advisor.source_execution as execution_module
import nominal_refactor_advisor.class_index as binding_module
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    InitialNativeIsland,
)
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_entry import (
    ImportedSourceModuleEntryPremise,
    SourceModuleEntryPremise,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def module(source="selected = property\n"):
    return ParsedModule(
        Path("entry_owner.py"), "entry_owner", False, ast.parse(source), source
    )


def supplied_entry(parsed):
    source = source_product_flow_projection(parsed)
    initial = InitialNativeIsland((builtins,))
    return SourceModuleEntryPremise(
        source,
        initial,
        {"property": CapturedNativeObject(object)},
        initial.namespace_for_storage(vars(builtins)),
    )


def test_supplied_entry_is_the_only_constructor_authority(monkeypatch):
    parsed = module()
    entry = supplied_entry(parsed)

    def no_recollection(*args, **kwargs):
        raise AssertionError("Supplied entry must not collect source again")

    monkeypatch.setattr(
        execution_module, "source_product_flow_projection", no_recollection
    )
    execution = SourceModuleExecution(entry)
    assert execution.entry is entry
    assert execution.module is parsed
    assert execution.source is entry.source
    assert execution.initial is entry.initial
    assert [field.name for field in fields(execution) if field.init] == ["entry"]
    assert not {"module", "source", "initial"}.intersection(execution.__dict__)


def test_supplied_globals_override_builtin_without_fabricated_frame():
    parsed = module()
    entry = supplied_entry(parsed)
    execution = SourceModuleExecution(entry)
    execution.capture(parsed.module.body[0].value).require_native_identity(
        NativeDeclaration(object)
    )
    assert execution.entry.frame.globals is entry
    assert execution.entry.frame.builtins is entry.builtins


def test_constructor_rejects_module_in_place_of_entry():
    with pytest.raises(TypeError, match="actual source entry premise"):
        SourceModuleExecution(module())


def test_entry_cannot_be_relabelled_by_an_independent_module_argument():
    parsed = module()
    entry = supplied_entry(parsed)
    forged = replace(parsed, source="selected = object  \n")
    assert forged.module is parsed.module
    with pytest.raises(TypeError):
        SourceModuleExecution(entry, module=forged)
    assert (
        "module"
        not in inspect.signature(
            ImportedSourceModuleEntryPremise.from_standard_source_loader
        ).parameters
    )


def test_standard_factory_collects_once_from_actual_module(monkeypatch):
    parsed = module()
    actual_collection = execution_module.source_product_flow_projection
    supplied = []

    def collect(actual):
        supplied.append(actual)
        return actual_collection(actual)

    monkeypatch.setattr(execution_module, "source_product_flow_projection", collect)
    execution = SourceModuleExecution.from_module(parsed)
    assert supplied == [parsed]
    assert execution.source is execution.entry.source
    assert execution.module is parsed
    assert execution.initial is execution.entry.initial
    execution.capture(parsed.module.body[0].value).require_native_identity(
        NativeDeclaration(property)
    )
    assert supplied == [parsed]


def test_binding_view_cache_reuses_one_entry_and_collection(monkeypatch):
    parsed = module()
    proof = RepositoryModuleBindingProof((parsed,))
    actual_collection = binding_module.source_product_flow_projection
    supplied = []

    def collect(actual):
        supplied.append(actual)
        return actual_collection(actual)

    monkeypatch.setattr(binding_module, "source_product_flow_projection", collect)
    first = proof.native_reference_environment(parsed)
    second = proof.native_reference_environment(parsed)
    assert first is second
    assert first.entry is second.entry
    assert supplied == [parsed]


def test_reparsed_source_gets_a_distinct_cached_entry():
    original = module()
    reparsed = module()
    proof = RepositoryModuleBindingProof((original,))
    first = proof.native_reference_environment(original)
    second = proof.native_reference_environment(reparsed)
    assert first is not second
    assert first.entry is not second.entry
    assert first.source.module is original
    assert second.source.module is reparsed
    foreign_context = second.entry.context
    with pytest.raises(ValueError, match="actual flow context"):
        first.entry.require_context(foreign_context)


def test_standard_factory_does_not_admit_additional_modules():
    execution = SourceModuleExecution.from_module(module("import ast\n"))
    with pytest.raises(ValueError, match="unadmitted_native_import"):
        execution.require_import(execution.module.module.body[0])


@pytest.mark.parametrize("loaded_module", (ast, dataclasses))
def test_explicit_cached_module_association_closes_import_not_arbitrary_execution(
    loaded_module,
):
    # These modules are already loaded by the test, not imported from source
    # names during analysis. The supplied premise restricts supported entry
    # associations; it is not proof of arbitrary module initialization/hooks.
    parsed = module(f"import {loaded_module.__name__}\nclass Plain: pass\n")
    source = source_product_flow_projection(parsed)
    initial = InitialNativeIsland((builtins, loaded_module))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, initial.namespace_for_storage(vars(builtins))
    )
    execution = SourceModuleExecution(entry)
    execution.require_import(parsed.module.body[0])
    execution.require_class_creation(parsed.module.body[-1])
    assert execution.initial is initial


def test_explicit_dataclasses_import_admits_the_proved_decorator_invocation():
    parsed = module(
        "from dataclasses import dataclass\n@dataclass\n"
        "class Product:\n    left: object\n    right: object\n"
    )
    source = source_product_flow_projection(parsed)
    initial = InitialNativeIsland((builtins, dataclasses))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        source, initial, initial.namespace_for_storage(vars(builtins))
    )
    execution = SourceModuleExecution(entry)
    execution.require_import(parsed.module.body[0])
    execution.require_class_creation(parsed.module.body[-1])


def test_explicit_module_admission_preserves_source_registration_collision_guard():
    parsed = replace(module("class Plain: pass\n"), module_name="ast")
    source = source_product_flow_projection(parsed)
    initial = InitialNativeIsland((builtins, ast))
    with pytest.raises(ValueError, match="replace an admitted native module"):
        ImportedSourceModuleEntryPremise.from_standard_source_loader(
            source, initial, initial.namespace_for_storage(vars(builtins))
        )
