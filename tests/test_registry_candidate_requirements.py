"""Candidate native-use receipts do not certify registry motion or execution."""

import ast
import builtins
from copy import deepcopy
from dataclasses import replace

import metaclass_registry
import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    InitialNativeIsland,
    OpenCapturedReference,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseProvenance,
)
from nominal_refactor_advisor.codemod_source_edits import (
    CodemodSourceRevision,
    SourceTextReplacement,
)
from nominal_refactor_advisor.registry_identity import AutoRegisterClassAuthority
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.source_geometry import SourceByteSpan
from test_registry_policy_integrity import _PLAIN, _execute

SOURCES = (
    pytest.param(_PLAIN, id="plain"),
    pytest.param(
        "REGISTRY={}\nalias=REGISTRY\nobserved=dict(alias)\n"
        "class Alpha:pass\nclass Beta:pass\n"
        "REGISTRY['alpha']=Alpha\nREGISTRY['beta']=Beta\n",
        id="safe-copy-before-classes",
    ),
    pytest.param(
        "REGISTRY={}\nalias=REGISTRY\nclass Alpha:pass\n"
        "observed=dict(alias)\nclass Beta:pass\n"
        "REGISTRY['alpha']=Alpha\nREGISTRY['beta']=Beta\n",
        id="unsafe-copy-between-classes",
    ),
)


def conversion(tmp_path, source=_PLAIN, *, patch=False):
    path = tmp_path / "candidate.py"
    source = "padding=None\n" + source
    snapshot = CodemodSourceSnapshot.from_source_mapping({path.as_posix(): source})
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Alpha")
    )
    operations = (operation,)
    if patch:
        operations += (
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                replacements=(SourceTextReplacement("padding", "updated"),),
            ),
        )
    simulation = CodemodPlanDocument(
        recipes=(RefactorRecipe("candidate-requirements", operations=operations),)
    ).simulate(snapshot)
    return snapshot, operation, simulation


def authority_for(operation, simulation):
    targets = operation.required_targets(
        simulation.after_snapshot_projection.base_snapshot
    )
    _, _, node = simulation.required_after_snapshot.target_node_for_rewrite_target(
        SourceRewriteTarget(
            file_path=targets.file_path,
            qualname=targets.component.authority_name,
        )
    )
    assert isinstance(node, ast.ClassDef)
    return AutoRegisterClassAuthority(node)


@pytest.mark.parametrize("source", SOURCES)
def test_generated_requirement_owns_actual_candidate_node_environment_and_revision(
    tmp_path, source
):
    before, operation, simulation = conversion(tmp_path, source)
    assert simulation.simulation.parse_valid
    assert not simulation.is_clean
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    authority = authority_for(operation, simulation)
    after = simulation.required_after_snapshot
    module = after.parsed_module_for_source_path(requirement.module.file_path)
    environment = after.product_flow_repository.native_reference_environment(module)

    assert requirement.operation is type(operation)
    assert requirement.environment is environment
    assert requirement.node is authority.metaclass_operand
    assert requirement.node in environment.source.reference_reads_by_node
    assert requirement.declarations == (authority.native_metaclass,)
    assert authority.native_metaclass.declaration is metaclass_registry.AutoRegisterMeta
    original_module = before.parsed_module_for_source_path(module.file_path)
    original_source = before.product_flow_repository.source_projection(original_module)
    assert requirement.node not in original_source.reference_reads_by_node
    assert requirement.receipt.revision == CodemodSourceRevision(
        module.file_path, CodemodSourceRevision.hash_source(module.source)
    )
    assert requirement.receipt.span == SourceByteSpan.require_node(requirement.node)
    assert (
        requirement.receipt.revision.source_hash
        != CodemodSourceRevision.hash_source(original_module.source)
    )
    (again,) = operation.candidate_native_use_requirements(simulation)
    assert again.node is requirement.node
    assert again.environment is requirement.environment
    assert again.receipt == requirement.receipt


@pytest.mark.parametrize("source", SOURCES)
def test_explicit_candidate_acceptance_does_not_change_capture_or_motion(
    tmp_path, source
):
    before, operation, simulation = conversion(tmp_path, source)
    requirements = operation.candidate_native_use_requirements(simulation)
    (requirement,) = requirements
    unresolved = requirement.inspect()
    assert unresolved.provenance is NativeUseProvenance.UNRESOLVED
    assert not unresolved.provenance.is_admitted
    original_capture = requirement.environment.capture(requirement.node)
    assert isinstance(original_capture, OpenCapturedReference)
    with pytest.raises(CapturedReferenceRejection) as original_error:
        original_capture.require_closed()

    declared = DeclaredNativeUseInvariants.from_requirements(
        requirements,
        rationale="The practitioner explicitly supports the recorded candidate native use.",
    )
    (resolution,) = declared.resolve(requirements)
    assert resolution.provenance is NativeUseProvenance.DECLARED
    resolution.require_admitted()
    capture = requirement.environment.capture(requirement.node)
    with pytest.raises(CapturedReferenceRejection) as unchanged_error:
        capture.require_closed()
    assert unchanged_error.value.violation is original_error.value.violation
    assert requirement.inspect().provenance is NativeUseProvenance.UNRESOLVED
    authority = authority_for(operation, simulation)
    with pytest.raises(
        CapturedReferenceRejection,
        match="^Native object identity remains open: unproved_execution_effects$",
    ) as class_error:
        requirement.environment.require_class_creation(authority.node)
    assert class_error.value.violation is original_error.value.violation
    assert isinstance(class_error.value.__cause__, CapturedReferenceRejection)
    assert (
        class_error.value.__cause__.violation
        is CapturedReferenceViolation.UNADMITTED_IMPORT
    )

    # Only these tiny authored fixtures are executed; acceptance is never used
    # as execution evidence or as a claim that the observed mapping was preserved.
    before_runtime = _execute(before.sources_by_file_path[requirement.module.file_path])
    after_runtime = _execute(requirement.module.source)
    if "observed" in vars(before_runtime):
        assert tuple(before_runtime.observed) == ()
        expected = (
            ("alpha",)
            if source.index("class Alpha") < source.index("observed=")
            else ()
        )
        assert tuple(after_runtime.observed) == expected


def test_extra_actual_edit_invalidates_previously_accepted_candidate_receipt(tmp_path):
    _, operation, first = conversion(tmp_path)
    previous = operation.candidate_native_use_requirements(first)
    acceptance = DeclaredNativeUseInvariants.from_requirements(
        previous, rationale="Explicitly accepted for the original candidate only."
    )
    _, changed_operation, changed = conversion(tmp_path, patch=True)
    current = changed_operation.candidate_native_use_requirements(changed)
    assert previous[0].receipt.span == current[0].receipt.span
    assert previous[0].receipt.revision != current[0].receipt.revision
    with pytest.raises(ValueError, match="stale, foreign or no longer required"):
        acceptance.resolve(current)


@pytest.mark.parametrize("substitution", ("template", "coordinate-copy"))
def test_generated_template_or_coordinate_copy_is_not_a_candidate_read(
    tmp_path, substitution
):
    before, operation, simulation = conversion(tmp_path)
    (requirement,) = operation.candidate_native_use_requirements(simulation)
    if substitution == "template":
        authority = operation.required_targets(before).component.destination_authority
        node = authority.metaclass_operand
    else:
        node = deepcopy(requirement.node)
        assert ast.dump(node, include_attributes=True) == ast.dump(
            requirement.node, include_attributes=True
        )
    assert node is not requirement.node
    with pytest.raises(ValueError, match="canonical original read"):
        replace(requirement, node=node)


def test_real_cached_module_proves_candidate_identity_but_not_construction(tmp_path):
    _, operation, simulation = conversion(tmp_path)
    (original,) = operation.candidate_native_use_requirements(simulation)
    initial = InitialNativeIsland((builtins, metaclass_registry))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        original.environment.source,
        initial,
        initial.namespace_for_storage(vars(builtins)),
    )
    environment = SourceModuleExecution(entry)
    requirement = replace(original, environment=environment)
    authority = authority_for(operation, simulation)
    authority.require_native_metaclass(environment)
    identity = requirement.inspect()
    assert identity.provenance is NativeUseProvenance.CAPTURED_IDENTITY
    assert not identity.provenance.is_admitted
    assert not entry.operation_conditions
    with pytest.raises(
        ValueError,
        match="^Native operation needs an explicit entry condition$",
    ):
        environment.require_class_creation(authority.node)
    # The supplied premise does not mutate the document's default environment.
    assert original.environment is not environment
    assert original.inspect().provenance is NativeUseProvenance.UNRESOLVED
    with pytest.raises(CapturedReferenceRejection) as error:
        original.environment.capture(original.node).require_closed()
    assert error.value.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
