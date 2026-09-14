"""Executed type-keyed descent under explicit, revision-bound native assumptions."""

import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from registry_test_sources import _type_keyed_behavior_projection_source

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DescendTypeKeyedBehaviorProjectionOperation,
    PatchTargetOperation,
    RefactorRecipe,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseProvenance,
    NativeUseResolution,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.json_reports import json_report_object


RATIONALE = (
    "Controlled fixture retains the declared native implementations at every use."
)
CANONICAL_IMPORT = (
    "from nominal_refactor_advisor.registry_identity import mro_registry_value"
)


def _subject(tmp_path, source=None):
    path = tmp_path / "subject.py"
    path.write_text(source or _type_keyed_behavior_projection_source())
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    )
    operation = DescendTypeKeyedBehaviorProjectionOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="EventProjection")
    )
    return path, snapshot, operation


def _accepted(operation, snapshot, requirements=None):
    return replace(
        operation,
        supported_execution=DeclaredNativeUseInvariants.from_requirements(
            operation.native_use_requirements(snapshot)
            if requirements is None
            else requirements,
            rationale=RATIONALE,
        ),
    )


def _simulate(operation, snapshot):
    return CodemodPlanSequence.from_operations((operation,)).simulate(snapshot)


def _outputs(source, monkeypatch, module_name):
    runtime = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, runtime)
    exec(compile(source, module_name, "exec"), runtime.__dict__)
    events = (runtime.Event(), runtime.NamedEvent(), runtime.CountedEvent())
    for event in events:
        event.value = "fallback"
    events[1].name = "named"
    events[2].count = 17
    return tuple(
        (runtime.render_event(event), runtime.render_event_locally(event))
        for event in events
    )


@pytest.mark.parametrize(
    "import_source,helper",
    (
        (CANONICAL_IMPORT, "mro_registry_value"),
        (CANONICAL_IMPORT + " as selected", "selected"),
        (
            "from nominal_refactor_advisor import registry_identity as helpers",
            "helpers.mro_registry_value",
        ),
    ),
)
def test_declared_contract_preserves_executed_dispatch(
    tmp_path, monkeypatch, import_source, helper
):
    source = (
        _type_keyed_behavior_projection_source()
        .replace(CANONICAL_IMPORT, import_source)
        .replace(
            "mro_registry_value(cls.__registry__, type(event))",
            f"{helper}(cls.__registry__, type(event))",
        )
    )
    path, snapshot, operation = _subject(tmp_path, source)
    assert len(operation.native_use_requirements(snapshot)) == 3
    result = _simulate(_accepted(operation, snapshot), snapshot)
    assert result.is_clean
    after = result.final_snapshot.parsed_module_for_source_path(str(path)).source
    assert after != source
    assert (
        _outputs(source, monkeypatch, "native_contract_before")
        == _outputs(after, monkeypatch, "native_contract_after")
        == (("fallback", "fallback"), ("named", "named"), ("17", "17"))
    )
    assert path.read_text() == source, "Simulation must not write fixture files"


@pytest.mark.parametrize("accepted_count", (0, 1, 2))
def test_unaccepted_required_uses_block_descent(tmp_path, accepted_count):
    _, snapshot, operation = _subject(tmp_path)
    if accepted_count:
        operation = _accepted(
            operation,
            snapshot,
            operation.native_use_requirements(snapshot)[:accepted_count],
        )
    reports = operation.preflight_reports(snapshot)
    assert sum(report.status.is_passed for report in reports) == accepted_count
    assert all(isinstance(report.detail, NativeUseResolution) for report in reports)
    with pytest.raises(
        CodemodOperationPreflightError, match="supported-execution invariant"
    ):
        _simulate(operation, snapshot)


def test_operation_and_plan_json_roundtrip_preserve_conditional_provenance(tmp_path):
    _, snapshot, operation = _subject(tmp_path)
    operation = _accepted(operation, snapshot)
    restored = RefactorRecipeOperation.from_json_value(
        json.loads(json.dumps(json_report_object(operation)))
    )
    assert restored == operation
    plan = CodemodPlanSequence.from_operations((restored,))
    replayed = CodemodPlanSequence.from_json_value(
        json.loads(json.dumps(json_report_object(plan)))
    )
    assert replayed == plan
    reports = restored.preflight_reports(snapshot)
    assert len(reports) == 3
    for report in reports:
        assert report.status.is_passed
        assert report.detail.provenance is NativeUseProvenance.DECLARED
        assert report.detail.rationale == RATIONALE
        assert report.detail.receipt in operation.supported_execution.requirements
        wire = json.loads(json.dumps(json_report_object(report)))
        assert wire["details"]["provenance"] == "declared"
    result = replayed.simulate(snapshot)
    assert result.is_clean
    stage = result.stage_reports[0].document_simulation
    assert result.preflight_report.reports
    assert len(result.preflight_report.reports) == len(stage.preflight_report.reports)
    assert all(
        aggregate is original
        for aggregate, original in zip(
            result.preflight_report.reports, stage.preflight_report.reports
        )
    )
    _assert_serialized_native_provenance(result)
    _assert_serialized_native_provenance(stage)


def _assert_serialized_native_provenance(result):
    reports = result.preflight_report.reports
    native_reports = tuple(
        report for report in reports if isinstance(report.detail, NativeUseResolution)
    )
    assert len(native_reports) == 3
    assert all(
        report.detail.provenance is NativeUseProvenance.DECLARED
        for report in native_reports
    )
    wire = json.loads(json.dumps(json_report_object(result)))
    native_wire = tuple(
        report
        for report in wire["preflight_report"]["reports"]
        if "provenance" in report["details"]
    )
    assert len(native_wire) == 3
    assert all(report["details"]["provenance"] == "declared" for report in native_wire)
    assert all(report["details"]["rationale"] == RATIONALE for report in native_wire)


def test_document_and_recipe_simulations_retain_native_contract(tmp_path):
    _, snapshot, operation = _subject(tmp_path)
    accepted = _accepted(operation, snapshot)
    plan = CodemodPlanSequence.from_operations((accepted,))
    preflight = plan.documents[0].preflight(snapshot)
    document_result = preflight.simulate()
    assert document_result.preflight_report is preflight.report
    _assert_serialized_native_provenance(document_result)
    recipe_result = RefactorRecipe(
        recipe_id="native-contract", operations=(accepted,)
    ).simulate(snapshot)
    _assert_serialized_native_provenance(recipe_result)


@pytest.mark.parametrize("require_clean", (True, False))
def test_failed_native_preflight_cannot_be_overridden_at_apply(tmp_path, require_clean):
    path, snapshot, operation = _subject(tmp_path)
    original = path.read_bytes()
    result = _simulate(_accepted(operation, snapshot), snapshot)
    failed = replace(
        result,
        preflight_report=CodemodPlanPreflightReport(
            operation.preflight_reports(snapshot)
        ),
    )
    assert not failed.is_clean
    with pytest.raises(CodemodOperationPreflightError):
        failed.apply(require_clean=require_clean)
    assert path.read_bytes() == original


def test_same_span_source_revision_invalidates_acceptance(tmp_path):
    path, snapshot, operation = _subject(tmp_path)
    accepted = _accepted(operation, snapshot)
    changed = path.read_text().replace("value: str", "value: int")
    _, current, _ = _subject(tmp_path, changed)
    old_requirements = operation.native_use_requirements(snapshot)
    new_requirements = operation.native_use_requirements(current)
    assert tuple(r.receipt.span for r in old_requirements) == tuple(
        r.receipt.span for r in new_requirements
    )
    with pytest.raises(CodemodOperationPreflightError, match="stale, foreign"):
        _simulate(accepted, current)


def test_earlier_sequential_edit_invalidates_accepted_revision(tmp_path):
    path, snapshot, operation = _subject(tmp_path)
    patch = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Event"),
        replacements=(
            SourceTextReplacement(old_source="value: str", new_source="value: int"),
        ),
    )
    assert _simulate(patch, snapshot).is_clean
    plan = CodemodPlanSequence.from_operations((patch, _accepted(operation, snapshot)))
    with pytest.raises(CodemodOperationPreflightError, match="stale, foreign"):
        plan.simulate(snapshot)
    assert "value: str" in path.read_text()


def test_predicted_stage_requirements_support_explicit_batched_descent(
    tmp_path, monkeypatch
):
    path, initial, descent = _subject(tmp_path)
    original = path.read_text()
    first_edit = ReplaceFunctionBodyOperation(
        target=SourceRewriteTarget(
            file_path=str(path), qualname="CountedEventProjection.render"
        ),
        body_source="return str(event.count + 1)",
    )
    prediction = _simulate(first_edit, initial)
    assert prediction.is_clean
    predicted = prediction.final_snapshot
    requirements = descent.native_use_requirements(predicted)
    assert len(requirements) == 3
    assert {requirement.receipt.revision for requirement in requirements}.isdisjoint(
        requirement.receipt.revision
        for requirement in descent.native_use_requirements(initial)
    )
    accepted = _accepted(descent, predicted, requirements)
    plan = CodemodPlanSequence.from_operations((first_edit, accepted))
    result = plan.simulate(initial)
    assert result.is_clean
    assert result.stage_count == 2
    second = result.stage_reports[1].document_simulation
    _assert_serialized_native_provenance(second)
    _assert_serialized_native_provenance(result)
    native_reports = tuple(
        report
        for report in second.preflight_report.reports
        if isinstance(report.detail, NativeUseResolution)
    )
    assert tuple(report.detail.receipt for report in native_reports) == (
        accepted.supported_execution.requirements
    )
    assert all(
        any(report is retained for retained in result.preflight_report.reports)
        for report in native_reports
    )
    predicted_source = predicted.parsed_module_for_source_path(str(path)).source
    final_source = result.final_snapshot.parsed_module_for_source_path(str(path)).source
    assert _outputs(original, monkeypatch, "batch_initial")[-1] == ("17", "17")
    assert (
        _outputs(predicted_source, monkeypatch, "batch_predicted")
        == _outputs(final_source, monkeypatch, "batch_final")
        == (("fallback", "fallback"), ("named", "named"), ("18", "18"))
    )
    assert path.read_text() == original


def test_source_only_operation_does_not_inherit_another_acceptance(tmp_path):
    _, snapshot, operation = _subject(tmp_path)
    accepted = _accepted(operation, snapshot)
    assert _simulate(accepted, snapshot).is_clean
    assert operation.supported_execution == DeclaredNativeUseInvariants()
    with pytest.raises(
        CodemodOperationPreflightError, match="supported-execution invariant"
    ):
        _simulate(operation, snapshot)


def test_acceptance_cannot_be_transferred_to_another_source_file(tmp_path):
    _, snapshot, operation = _subject(tmp_path)
    accepted = _accepted(operation, snapshot)
    other = tmp_path / "other"
    other.mkdir()
    _, other_snapshot, other_operation = _subject(other)
    transferred = replace(
        other_operation, supported_execution=accepted.supported_execution
    )
    with pytest.raises(CodemodOperationPreflightError, match="stale, foreign"):
        _simulate(transferred, other_snapshot)
