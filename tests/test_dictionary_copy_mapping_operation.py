"""An authored mapping obligation gates the complete actual rewrite document."""

import ast
from dataclasses import dataclass, replace
import json

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    CreateFileOperation,
    PatchTargetOperation,
    RefactorRecipe,
    RequireDictionaryCopyMappingOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ForbiddenCallArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.codemod_source_edits import SourceTextReplacement
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class MappingCase:
    name: str
    source: str
    replacements: tuple[SourceTextReplacement, ...]
    admitted: bool


STORE = "registry['alpha'] = object\n"
VALUES = "padding=None\nregistry={}\n" + STORE + "observed=dict(registry)\n"
STORES = "registry['alpha']=object\nregistry['beta']=property\n"
CASES = (
    *(
        MappingCase(
            f"store_motion_{operand}",
            "registry={}\nalias=registry\nother={}\n"
            + f"observed=dict({operand})\n"
            + STORE,
            (
                SourceTextReplacement(STORE, ""),
                SourceTextReplacement("observed=", STORE + "observed="),
            ),
            operand == "other",
        )
        for operand in ("other", "alias")
    ),
    MappingCase(
        "same_key_different_native_value",
        VALUES,
        (SourceTextReplacement("= object", "= property"),),
        False,
    ),
    MappingCase(
        "same_native_value",
        VALUES,
        (SourceTextReplacement("padding", "extra"),),
        True,
    ),
    MappingCase(
        "changed_keyword",
        "registry={}\nobserved=dict(registry, extra=object)\n",
        (SourceTextReplacement("extra=", "other="),),
        False,
    ),
    MappingCase(
        "changed_call",
        "registry={}\nobserved=dict(registry)\n",
        (SourceTextReplacement("dict(registry)", "dict(registry, added=object)"),),
        False,
    ),
    MappingCase(
        "source_created_identity_unproved",
        "padding=None\nclass Item: pass\nregistry={}\n"
        "registry['alpha']=Item\nobserved=dict(registry)\n",
        (SourceTextReplacement("padding", "extra"),),
        False,
    ),
    MappingCase(
        "unordered_mapping_only",
        "registry={}\n" + STORES + "observed=dict(registry)\n",
        (
            SourceTextReplacement(
                STORES,
                "registry['beta']=property\nregistry['alpha']=object\n",
            ),
        ),
        True,
    ),
)


def snapshot_for(path, source):
    path.write_text(source, encoding="utf-8", newline="")
    return CodemodSourceSnapshot.from_source_mapping({path.as_posix(): source})


def obligation(snapshot, path, *, qualname=None):
    module = snapshot.parsed_module_for_source_path(path.as_posix())
    (call,) = tuple(
        node for node in ast.walk(module.module) if isinstance(node, ast.Call)
    )
    span = SourceByteSpan.require_node(call)
    source = snapshot.product_flow_repository.source_projection(module)
    assert source.call_operation(span).node is call
    return RequireDictionaryCopyMappingOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname=qualname),
        call_spans=(span,),
    )


def document_for_operation(operation, patches=(), *, separate=False):
    if separate:
        recipes = (
            RefactorRecipe("required-mapping", operations=(operation,)),
            RefactorRecipe("authored-rewrite", operations=patches),
        )
    else:
        recipes = (
            RefactorRecipe("mapping-checked-rewrite", operations=(operation, *patches)),
        )
    return CodemodPlanDocument(recipes=recipes)


def patch_for(path, replacements):
    return PatchTargetOperation(
        target=SourceRewriteTarget(file_path=path.as_posix()),
        replacements=replacements,
    )


def mapping_reports(simulation):
    return tuple(
        report
        for report in simulation.preflight_report.reports
        if report.operation == RequireDictionaryCopyMappingOperation.operation_key()
    )


def native_mapping(source):
    namespace = {}
    exec(compile(source, "<authored-mapping-fixture>", "exec"), namespace)
    return namespace["observed"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_complete_document_mapping_relation_matches_eight_actual_rewrites(
    tmp_path, case
):
    path = tmp_path / "mapping.py"
    snapshot = snapshot_for(path, case.source)
    operation = obligation(snapshot, path)
    assert operation.source_edits(snapshot) == ()
    patches = (
        tuple(patch_for(path, (replacement,)) for replacement in case.replacements)
        if case.name.startswith("store_motion_")
        else (patch_for(path, case.replacements),)
    )
    simulation = document_for_operation(operation, patches).simulate(snapshot)
    assert simulation.is_clean is case.admitted
    reports = mapping_reports(simulation)
    assert reports
    assert all(report.status.is_passed for report in reports) is case.admitted
    assert simulation.simulation.parse_valid
    assert path.read_text() == case.source
    after = simulation.required_after_snapshot.sources_by_file_path[path.as_posix()]
    old_mapping, new_mapping = native_mapping(case.source), native_mapping(after)
    if case.admitted:
        assert old_mapping.keys() == new_mapping.keys()
        assert all(old_mapping[key] is new_mapping[key] for key in old_mapping)
    if case.name == "unordered_mapping_only":
        assert tuple(old_mapping) != tuple(new_mapping)
    if case.name == "store_motion_alias":
        assert old_mapping == {}
        assert new_mapping == {"alpha": object}
    if case.name == "same_key_different_native_value":
        assert old_mapping.keys() == new_mapping.keys()
        assert old_mapping["alpha"] is object
        assert new_mapping["alpha"] is property


@pytest.mark.parametrize("require_clean", (True, False))
def test_failed_mapping_blocks_application_even_without_optional_guards(
    tmp_path, require_clean
):
    path = tmp_path / "blocked.py"
    snapshot = snapshot_for(path, VALUES)
    before_bytes = path.read_bytes()
    operation = obligation(snapshot, path)
    patch = patch_for(path, (SourceTextReplacement("= object", "= property"),))
    simulation = document_for_operation(operation, (patch,)).simulate(snapshot)
    assert not simulation.is_clean
    with pytest.raises(ValueError):
        simulation.apply(require_clean=require_clean)
    assert path.read_bytes() == before_bytes


def test_positive_mapping_allows_checked_apply_to_original_file(tmp_path):
    path = tmp_path / "allowed.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    patch = patch_for(path, (SourceTextReplacement("padding", "extra"),))
    simulation = document_for_operation(operation, (patch,)).simulate(snapshot)
    assert simulation.is_clean
    assert simulation.apply() == (path.as_posix(),)
    assert path.read_text() == VALUES.replace("padding", "extra")


@pytest.mark.parametrize(
    "source",
    (
        "observed=object()\n",
        "dict=object\nregistry={}\nobserved=dict(registry)\n",
        "registry={}\nobserved=unknown(registry)\n",
    ),
)
def test_unsupported_or_shadowed_call_cannot_supply_a_mapping_proof(tmp_path, source):
    path = tmp_path / "unsupported.py"
    snapshot = snapshot_for(path, source)
    operation = obligation(snapshot, path)
    simulation = document_for_operation(operation).simulate(snapshot)
    assert not simulation.is_clean
    assert any(report.status.is_failed for report in mapping_reports(simulation))
    assert not simulation.edit_batch.edits
    with pytest.raises(ValueError):
        simulation.apply(require_clean=False)
    assert path.read_text() == source


def test_zero_edit_native_copy_document_can_pass_without_rewriting(tmp_path):
    path = tmp_path / "unchanged.py"
    source = "registry={}\nobserved=dict(registry)\n"
    snapshot = snapshot_for(path, source)
    operation = obligation(snapshot, path)
    simulation = document_for_operation(operation).simulate(snapshot)
    assert simulation.is_clean
    assert not simulation.edit_batch.edits
    assert simulation.apply() == ()
    assert path.read_text() == source


def test_call_span_outside_inherited_target_is_rejected_before_rewrite(tmp_path):
    path = tmp_path / "outside.py"
    source = "def selected(): pass\nregistry={}\nobserved=dict(registry)\n"
    snapshot = snapshot_for(path, source)
    operation = obligation(snapshot, path, qualname="selected")
    with pytest.raises(ValueError, match="outside.*target"):
        operation.source_edits(snapshot)
    with pytest.raises(ValueError, match="outside.*target"):
        document_for_operation(operation).simulate(snapshot)
    assert path.read_text() == source


def test_registered_operation_roundtrips_public_json_and_executes(tmp_path):
    path = tmp_path / "decoded.py"
    source = "registry={}\nobserved=dict(registry)\n"
    snapshot = snapshot_for(path, source)
    operation = obligation(snapshot, path)
    payload = json.loads(json.dumps(json_report_object(operation)))
    assert payload["operation"] == operation.operation_key()
    assert payload["call_spans"] == [
        json_report_object(span) for span in operation.call_spans
    ]
    decoded = RefactorRecipeOperation.from_json_value(payload)
    assert type(decoded) is RequireDictionaryCopyMappingOperation
    assert decoded == operation
    simulation = document_for_operation(decoded).simulate(snapshot)
    assert simulation.is_clean


@pytest.mark.parametrize("safe", (True, False))
def test_earlier_recipe_obligation_checks_later_recipe_physical_edits(tmp_path, safe):
    path = tmp_path / "recipes.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    replacement = (
        SourceTextReplacement("padding", "extra")
        if safe
        else SourceTextReplacement("= object", "= property")
    )
    document = document_for_operation(
        operation, (patch_for(path, (replacement,)),), separate=True
    )
    simulation = document.simulate(snapshot)
    assert len(simulation.document.recipes) == 2
    assert simulation.is_clean is safe
    assert (
        any(report.status.is_failed for report in mapping_reports(simulation))
        is not safe
    )
    assert path.read_text() == VALUES


def test_failed_stage_does_not_advance_to_later_file_creation(tmp_path):
    path = tmp_path / "first.py"
    later_path = tmp_path / "must_not_be_created.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    first = document_for_operation(
        operation,
        (patch_for(path, (SourceTextReplacement("= object", "= property"),)),),
    )
    second = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "later-stage",
                operations=(
                    CreateFileOperation(
                        target=SourceRewriteTarget(file_path=later_path.as_posix()),
                        source="created=True\n",
                    ),
                ),
            ),
        )
    )
    sequence = CodemodPlanSequence(documents=(first, second))
    first.preflight(snapshot).report.require_clean()
    try:
        simulation = sequence.simulate(snapshot)
    except CodemodOperationPreflightError as error:
        assert error.report.operation == operation.operation_key()
    else:
        assert not simulation.is_clean
        assert len(simulation.stage_reports) == 1
        assert (
            later_path.as_posix() not in simulation.final_snapshot.sources_by_file_path
        )
        with pytest.raises(ValueError):
            simulation.apply(require_clean=False)
    assert not later_path.exists()
    assert path.read_text() == VALUES


def test_sequence_preflight_retains_post_simulation_mapping_failure(tmp_path):
    path = tmp_path / "sequence_preflight.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    document = document_for_operation(
        operation,
        (patch_for(path, (SourceTextReplacement("= object", "= property"),)),),
    )
    document.preflight(snapshot).report.require_clean()
    report = CodemodPlanSequence(documents=(document,)).preflight_snapshot(snapshot)
    assert not report.is_clean
    assert any(
        item.operation == operation.operation_key() and item.status.is_failed
        for item in report.reports
    )
    assert path.read_text() == VALUES


def test_document_preflight_retains_post_simulation_mapping_failure(tmp_path):
    path = tmp_path / "document_preflight.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    document = document_for_operation(
        operation,
        (patch_for(path, (SourceTextReplacement("= object", "= property"),)),),
    )
    document.preflight(snapshot).report.require_clean()
    report = document.preflight_snapshot(snapshot)
    assert not report.is_clean
    assert any(
        item.operation == operation.operation_key() and item.status.is_failed
        for item in report.reports
    )
    assert path.read_text() == VALUES


def test_opaque_motion_patch_does_not_claim_retained_call_origin(tmp_path):
    case = CASES[0]
    path = tmp_path / "opaque_motion.py"
    snapshot = snapshot_for(path, case.source)
    operation = obligation(snapshot, path)
    document = document_for_operation(operation, (patch_for(path, case.replacements),))
    simulation = document.simulate(snapshot)
    assert not simulation.is_clean
    assert any(
        "unchanged-text correspondence" in report.message
        for report in mapping_reports(simulation)
    )
    assert (
        native_mapping(case.source)
        == native_mapping(
            simulation.required_after_snapshot.sources_by_file_path[path.as_posix()]
        )
        == {}
    )


@pytest.mark.parametrize("safe", (True, False))
def test_failed_optional_guard_cannot_hide_required_mapping_report(tmp_path, safe):
    path = tmp_path / "optional_guard.py"
    snapshot = snapshot_for(path, VALUES)
    operation = obligation(snapshot, path)
    replacement = (
        SourceTextReplacement("padding", "extra")
        if safe
        else SourceTextReplacement("= object", "= property")
    )
    document = document_for_operation(operation, (patch_for(path, (replacement,)),))
    document = replace(
        document,
        guard_suite=ArchitectureGuardSuite(
            (
                ArchitectureGuardRule(
                    rule_id="optional-no-dict-call",
                    constraints=(ForbiddenCallArchitectureGuardConstraint(("dict",)),),
                ),
            )
        ),
    )
    simulation = document.simulate(snapshot)
    assert not simulation.architecture_guard_report.is_clean
    assert not simulation.is_clean
    assert simulation.preflight_report.is_clean is safe
    assert mapping_reports(simulation)
    if safe:
        assert simulation.apply(require_clean=False) == (path.as_posix(),)
        assert path.read_text() == VALUES.replace("padding", "extra")
    else:
        with pytest.raises(ValueError):
            simulation.apply(require_clean=False)
        assert path.read_text() == VALUES
