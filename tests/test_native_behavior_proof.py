"""Real original operations, explicit environments, and fail-closed mutation controls."""

import ast
import builtins
import dataclasses
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import metaclass_registry

import nominal_refactor_advisor.registry_identity as registry_identity
from nominal_refactor_advisor.captured_reference import InitialNativeIsland
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    NativeDefinitionUseRequirement,
    NativeInvocationUseRequirement,
    NativeUseProvenance,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_class_mro import NativeClassMroDeclaration
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository
from nominal_refactor_advisor.source_entry import (
    ImportedSourceModuleEntryPremise,
    NoninterferingSourceModuleEntryPremise,
)


class NoninterferingImportedEntry(
    NoninterferingSourceModuleEntryPremise, ImportedSourceModuleEntryPremise
):
    """Explicit external noninterference, not assumed helper/native behavior."""

    @classmethod
    def from_source(cls, source):
        initial = InitialNativeIsland((builtins, registry_identity))
        return cls.from_standard_source_loader(
            source, initial, initial.namespace_for_storage(vars(builtins))
        )


class ExplicitEnvironmentRepository(SourceProductFlowRepository):
    source_entry = staticmethod(NoninterferingImportedEntry.from_source)


def snapshot_with_environment(tmp_path, source):
    path = (tmp_path / "behavior.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_source_mapping({path: source})
    repository = ExplicitEnvironmentRepository.from_modules(snapshot.parsed_modules)
    return path, snapshot._from_modules_with_indexes(
        snapshot.parsed_modules,
        snapshot.required_class_family_index,
        snapshot._source_index_build_artifacts,
        repository,
    )


def invocation_requirement(snapshot, path, declaration):
    module = snapshot.parsed_module_for_source_path(path)
    environment = snapshot.product_flow_repository.native_reference_environment(module)
    node = module.module.body[-1].value
    return NativeInvocationUseRequirement(
        PatchTargetOperation,
        node.func,
        (NativeDeclaration(declaration),),
        environment,
        source_state=snapshot.product_flow_repository,
    )


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("type(None)", type(None)),
        ("type(4)", int),
        ("type('text')", str),
        ("type(type)", type),
    ),
)
def test_exact_type_query_has_real_result_and_no_authored_native_invariant(
    tmp_path, expression, expected
):
    path, snapshot = snapshot_with_environment(tmp_path, f"result={expression}\n")
    requirement = invocation_requirement(snapshot, path, type)
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    environment = requirement.environment
    context, call = environment.source_call(environment.module.module.body[-1].value)
    environment.call_authority(context, call).result().require_native(
        (NativeDeclaration(expected),)
    )
    namespace = {}
    exec(environment.module.native_compilation.compile(), namespace)
    assert namespace["result"] is expected
    assert not environment.entry.operation_conditions


@pytest.mark.parametrize(
    "expression",
    (
        "type()",
        "type(None, None)",
        "type('New', (), {})",
        "type(value=None)",
        "type(*())",
        "type(unknown)",
    ),
)
def test_unsupported_type_binding_or_operand_stays_unresolved(tmp_path, expression):
    path, snapshot = snapshot_with_environment(tmp_path, f"result={expression}\n")
    inspected = invocation_requirement(snapshot, path, type).inspect()
    assert not inspected.provenance.is_admitted


def test_standalone_classmethod_definition_uses_actual_application_and_descriptor_contract(
    tmp_path,
):
    path, snapshot = snapshot_with_environment(
        tmp_path, "@classmethod\ndef method(cls):\n    return cls\n"
    )
    module = snapshot.parsed_module_for_source_path(path)
    environment = snapshot.product_flow_repository.native_reference_environment(module)
    definition = module.module.body[-1]
    requirement = NativeDefinitionUseRequirement(
        PatchTargetOperation,
        definition.decorator_list[0],
        (NativeDeclaration(classmethod),),
        environment,
        source_state=snapshot.product_flow_repository,
    )
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    environment.capture_definition(definition).require_class_installation()
    namespace = {}
    exec(module.native_compilation.compile(), namespace)
    assert type(namespace["method"]) is classmethod
    assert not environment.entry.operation_conditions


@pytest.mark.parametrize(
    "source",
    (
        "@unknown\n@classmethod\ndef method(cls): return cls\n",
        "classmethod=staticmethod\n@classmethod\ndef method(cls): return cls\n",
    ),
)
def test_other_applications_and_rebound_descriptor_operands_do_not_gain_proof(
    tmp_path, source
):
    path, snapshot = snapshot_with_environment(tmp_path, source)
    module = snapshot.parsed_module_for_source_path(path)
    environment = snapshot.product_flow_repository.native_reference_environment(module)
    node = module.module.body[-1].decorator_list[-1]
    requirement = NativeDefinitionUseRequirement(
        PatchTargetOperation, node, (NativeDeclaration(classmethod),), environment
    )
    try:
        resolution = requirement.inspect()
    except ValueError:
        return
    assert not resolution.provenance.is_admitted


def test_mro_lookup_derives_current_source_and_original_empty_mapping_result(tmp_path):
    source = "from nominal_refactor_advisor.registry_identity import mro_registry_value\nregistry={}\nresult=mro_registry_value(registry, type(None))\n"
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    namespace = {}
    exec(requirement.module.native_compilation.compile(), namespace)
    assert namespace["result"] is None
    assert not requirement.environment.entry.operation_conditions


def test_current_source_proof_is_not_a_function_identity_whitelist(
    tmp_path, monkeypatch
):
    source = "from nominal_refactor_advisor.registry_identity import mro_registry_value\nresult=mro_registry_value({}, type(None))\n"
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert requirement.inspect().provenance.is_admitted
    original = registry_identity.mro_registry_value

    def changed(registry, declaration_type):
        return "changed"

    monkeypatch.setattr(original, "__code__", changed.__code__)
    assert registry_identity.mro_registry_value is original
    assert original({}, type(None)) == "changed"
    assert not requirement.inspect().provenance.is_admitted


@pytest.mark.parametrize("before_entry", (False, True))
def test_rebound_helper_next_global_is_rejected_at_the_original_cut(
    tmp_path, monkeypatch, before_entry
):
    source = "from nominal_refactor_advisor.registry_identity import mro_registry_value\nresult=mro_registry_value({}, type(None))\n"
    if before_entry:
        monkeypatch.setattr(
            registry_identity, "next", lambda *args: "changed", raising=False
        )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    if not before_entry:
        assert requirement.inspect().provenance.is_admitted
        monkeypatch.setattr(
            registry_identity, "next", lambda *args: "changed", raising=False
        )
    assert not requirement.inspect().provenance.is_admitted


def test_dsl_projected_source_admits_real_type_operation_without_transporting_acceptance(
    tmp_path,
):
    path, snapshot = snapshot_with_environment(
        tmp_path, "padding=0\nresult=type(None)\n"
    )
    operation = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=path),
        replacements=(SourceTextReplacement("padding=0", "padding=1"),),
    )
    preview = CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)
    assert preview.is_clean
    requirement = invocation_requirement(preview.final_snapshot, path, type)
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    assert (
        requirement.environment.source.module
        is preview.final_snapshot.parsed_module_for_source_path(path)
    )
    assert snapshot.sources_by_file_path[path] == "padding=0\nresult=type(None)\n"


@pytest.mark.parametrize("spoof_qualification", (False, True))
@pytest.mark.parametrize(
    "registry_source,registry_operand",
    (("", "{}"), ("registry={}\nregistry[object]=property\n", "registry")),
)
def test_dsl_dependency_edit_invalidates_reused_local_native_execution(
    tmp_path, monkeypatch, spoof_qualification, registry_source, registry_operand
):
    path, initial = snapshot_with_environment(
        tmp_path,
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        "padding=0\n"
        + registry_source
        + f"result=mro_registry_value({registry_operand}, type(None))\n",
    )
    provider_path = registry_identity.__file__
    provider_source = Path(registry_identity.__file__).read_text()
    provider = ParsedModule(
        Path(provider_path),
        registry_identity.__name__,
        False,
        ast.parse(provider_source),
        provider_source,
    )
    modules = (*initial.parsed_modules, provider)
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    repository = ExplicitEnvironmentRepository.from_modules(snapshot.parsed_modules)
    snapshot = snapshot._from_modules_with_indexes(
        snapshot.parsed_modules,
        snapshot.required_class_family_index,
        snapshot._source_index_build_artifacts,
        repository,
    )
    original = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert original.inspect().provenance is NativeUseProvenance.PROVED
    unrelated = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=path),
        replacements=(SourceTextReplacement("padding=0", "padding=1"),),
    )
    first = (
        CodemodPlanSequence.from_operations((unrelated,))
        .simulate(snapshot)
        .final_snapshot
    )
    assert isinstance(first.product_flow_repository, ExplicitEnvironmentRepository)
    local = invocation_requirement(first, path, registry_identity.mro_registry_value)
    local.environment.capture(local.node).require_closed()
    assert local.inspect().provenance is NativeUseProvenance.PROVED
    dependency = PatchTargetOperation(
        target=SourceRewriteTarget(
            file_path=provider_path, qualname="mro_registry_value"
        ),
        replacements=(SourceTextReplacement("        None,", "        'changed',"),),
    )
    second = (
        CodemodPlanSequence.from_operations((dependency,))
        .simulate(first)
        .final_snapshot
    )
    changed = invocation_requirement(second, path, registry_identity.mro_registry_value)
    assert changed.environment is local.environment
    assert changed.node is local.node
    assert changed.source_state is not local.source_state
    if spoof_qualification:
        monkeypatch.setattr(
            registry_identity.mro_registry_value, "__module__", "external_spoof"
        )
    resolution = changed.inspect()
    assert not resolution.provenance.is_admitted
    assert "projected source state" in resolution.rationale
    # The earlier source state's own proof remains true; no global state or
    # previous native receipt is rewritten to impersonate the projected state.
    assert local.inspect().provenance is NativeUseProvenance.PROVED


def test_foreign_projected_source_state_is_not_an_execution_owner(tmp_path):
    path, snapshot = snapshot_with_environment(tmp_path, "result=type(None)\n")
    original = invocation_requirement(snapshot, path, type)
    independent = CodemodSourceSnapshot.from_source_mapping(
        {path: "result=type(None)\n"}
    )
    with pytest.raises(ValueError, match="different parsed owners"):
        replace(original, source_state=independent.product_flow_repository)


def test_native_qualification_does_not_run_mutated_metadata_hooks(monkeypatch):
    calls = []

    class Hook:
        def __str__(self):
            calls.append("str")
            raise AssertionError("qualification hook")

    monkeypatch.setattr(registry_identity.mro_registry_value, "__module__", Hook())
    with pytest.raises(ValueError, match="qualification requires exact text"):
        _ = NativeDeclaration(registry_identity.mro_registry_value).qualified_name
    assert calls == []


@pytest.mark.parametrize("mutation", ("new-scalar", "class-key", "replaced-value"))
def test_native_mapping_mutation_cannot_hide_behind_a_retained_namespace(
    tmp_path, monkeypatch, mutation
):
    fixture = ModuleType("native_mapping_fixture")
    fixture.DATA = {"control": 1}
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)

    class Entry(NoninterferingImportedEntry):
        @classmethod
        def from_source(cls, source):
            initial = InitialNativeIsland(
                (builtins, registry_identity, fixture), (fixture.DATA,)
            )
            return cls.from_standard_source_loader(
                source, initial, initial.namespace_for_storage(vars(builtins))
            )

    class Repository(ExplicitEnvironmentRepository):
        source_entry = staticmethod(Entry.from_source)

    path, original = snapshot_with_environment(
        tmp_path,
        "from native_mapping_fixture import DATA\nfrom nominal_refactor_advisor.registry_identity import mro_registry_value\nresult=mro_registry_value(DATA, type(None))\n",
    )
    repository = Repository.from_modules(original.parsed_modules)
    snapshot = original._from_modules_with_indexes(
        original.parsed_modules,
        original.required_class_family_index,
        original._source_index_build_artifacts,
        repository,
    )
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    if mutation == "new-scalar":
        fixture.DATA["new"] = 2
    elif mutation == "class-key":
        fixture.DATA[type(None)] = "hit"
        assert registry_identity.mro_registry_value(fixture.DATA, type(None)) == "hit"
    else:
        fixture.DATA["control"] = "changed"
    resolution = requirement.inspect()
    assert not resolution.provenance.is_admitted
    assert "dictionary" in resolution.rationale


def test_mapping_hooks_are_not_run_by_analysis_or_assigned_native_behavior(tmp_path):
    source = "from nominal_refactor_advisor.registry_identity import mro_registry_value\nclass Hooked(dict):\n    def __contains__(self, key):\n        raise RuntimeError('mapping hook')\nresult=mro_registry_value(Hooked(), type(None))\n"
    path, snapshot = snapshot_with_environment(tmp_path, source)
    assert (
        not invocation_requirement(snapshot, path, registry_identity.mro_registry_value)
        .inspect()
        .provenance.is_admitted
    )
    with pytest.raises(RuntimeError, match="mapping hook"):
        exec(compile(source, "<trusted-mapping-control>", "exec"), {})


def test_omitting_global_source_authority_does_not_turn_an_operation_into_local_proof(
    tmp_path,
):
    path, snapshot = snapshot_with_environment(tmp_path, "result=type(None)\n")
    requirement = invocation_requirement(snapshot, path, type)
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    omitted = replace(requirement, source_state=None).inspect()
    assert not omitted.provenance.is_admitted
    assert "complete source-state authority" in omitted.rationale


def test_changed_metaclass_prepare_is_rejected_without_running_the_hook(monkeypatch):
    from test_native_source_class_preparation import prepared_execution

    environment, (root, _) = prepared_execution(conditions=False)
    entry = environment.class_entry(root)
    entry.require_preparation()
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        return {}

    monkeypatch.setattr(
        metaclass_registry.AutoRegisterMeta, "__prepare__", classmethod(changed)
    )
    with pytest.raises(ValueError, match="exact type descriptor"):
        entry.require_preparation()
    assert calls == []
    assert not environment.entry.operation_conditions


def test_original_constructor_query_revalidates_implicit_class_binding(monkeypatch):
    from test_native_source_class_preparation import prepared_execution

    environment, (root,) = prepared_execution(
        "class Family(metaclass=Creator): pass\n", conditions=False
    )
    entry = environment.class_entry(root)
    with pytest.raises(ValueError, match="over prepared inputs remains unproved"):
        _ = entry.construction_admission
    function = NativeClassMroDeclaration(
        metaclass_registry.AutoRegisterMeta
    ).python_constructor()
    with monkeypatch.context() as mutation:
        mutation.setattr(function.__closure__[0], "cell_contents", type)
        with pytest.raises(ValueError, match="selected MRO owner"):
            _ = entry.construction_admission
    with pytest.raises(ValueError, match="over prepared inputs remains unproved"):
        _ = entry.construction_admission
    assert not environment.entry.operation_conditions
    assert "construction_admission" not in vars(entry)


@pytest.mark.parametrize(
    "expression",
    (
        "type(dataclass(frozen=Ephemeral()))",
        "type((dataclass(frozen=Ephemeral()),))",
    ),
)
def test_type_query_does_not_hide_temporary_destructor_effects(tmp_path, expression):
    class Entry(NoninterferingImportedEntry):
        @classmethod
        def from_source(cls, source):
            initial = InitialNativeIsland((builtins, dataclasses, registry_identity))
            return cls.from_standard_source_loader(
                source, initial, initial.namespace_for_storage(vars(builtins))
            )

    class Repository(ExplicitEnvironmentRepository):
        source_entry = staticmethod(Entry.from_source)

    source = (
        "from dataclasses import dataclass\n"
        "state={}\n"
        "class Ephemeral:\n"
        "    def __del__(self): state['destroyed']=True\n" + f"result={expression}\n"
    )
    path, initial = snapshot_with_environment(tmp_path, source)
    repository = Repository.from_modules(initial.parsed_modules)
    snapshot = initial._from_modules_with_indexes(
        initial.parsed_modules,
        initial.required_class_family_index,
        initial._source_index_build_artifacts,
        repository,
    )
    requirement = invocation_requirement(snapshot, path, type)
    resolution = requirement.inspect()
    assert not resolution.provenance.is_admitted
    assert "Release" in resolution.rationale or "lifetime" in resolution.rationale
    namespace = {}
    exec(requirement.module.native_compilation.compile(), namespace)
    assert namespace["state"] == {"destroyed": True}
    assert namespace["result"] in (type(lambda: None), tuple)
    assert not requirement.environment.entry.operation_conditions
