"""Actual invocation and conversion preserve the capture's original refusal."""

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
    RequireDictionaryCopyMappingOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_reproof import SourceReproofDiagnostic
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_geometry import SourceByteSpan


def environment_for(source):
    path = "/repo/subject.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping({path: source})
    module = snapshot.parsed_module_for_source_path(path)
    environment = snapshot.product_flow_repository.native_reference_environment(module)
    return snapshot, environment


@pytest.mark.parametrize(
    ("source", "violation"),
    (
        ("result=missing()\n", CapturedReferenceViolation.UNPROVED_BINDING),
        (
            "import deliberately_unadmitted_dependency\nresult=dict()\n",
            CapturedReferenceViolation.UNPROVED_EFFECTS,
        ),
    ),
)
def test_actual_open_callee_keeps_original_capture_evidence(source, violation):
    _, environment = environment_for(source)
    (operation,) = tuple(environment.source.call_operations_by_span.values())
    capture = environment.capture(operation.node.func)
    with pytest.raises(CapturedReferenceRejection) as rejected:
        capture.call_authority(
            environment, environment.context_for_owner(operation.owner), operation.event
        )
    assert rejected.value.evidence is capture
    assert rejected.value.violation is violation
    assert rejected.value.__cause__ is capture.cause
    with pytest.raises(CapturedReferenceRejection) as invocation:
        environment.call_authority(
            environment.context_for_owner(operation.owner), operation.event
        )
    assert invocation.value.violation is violation


def test_closed_source_function_still_has_no_proved_invocation():
    _, environment = environment_for("def factory():\n    pass\nresult=factory()\n")
    (operation,) = tuple(environment.source.call_operations_by_span.values())
    environment.capture(operation.node.func).require_closed()
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    # Selecting the original invocation supplies no function-state or body proof.
    with pytest.raises(ValueError, match="External source interference") as rejected:
        authority.require_closed()
    assert type(rejected.value) is ValueError
    assert rejected.value.__cause__ is None


@pytest.mark.parametrize("observe_before_classes", (True, False))
def test_generated_registry_mapping_report_preserves_native_entry_cause(
    observe_before_classes,
):
    declarations = "class Alpha: pass\nclass Beta: pass\n"
    observation = "observed=dict(alias)\n"
    source = (
        "REGISTRY={}\nalias=REGISTRY\n"
        + (
            observation + declarations
            if observe_before_classes
            else declarations + observation
        )
        + "REGISTRY['alpha']=Alpha\nREGISTRY['beta']=Beta\n"
    )
    snapshot, environment = environment_for(source)
    (operation,) = tuple(environment.source.call_operations_by_span.values())
    environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    ).require_closed()
    target = SourceRewriteTarget(file_path="/repo/subject.py")
    mapping = RequireDictionaryCopyMappingOperation(
        target=target,
        call_spans=(SourceByteSpan.require_node(operation.node),),
    )
    converter = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(file_path=target.file_path, qualname="Alpha")
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("convert-and-check", operations=(converter, mapping)),),
    )
    result = document.simulate(snapshot)
    assert not result.is_clean
    (report,) = tuple(
        report
        for report in result.preflight_report.reports
        if report.operation == mapping.operation_key()
    )
    assert not report.status.is_passed
    detail = report.detail
    assert isinstance(detail, SourceReproofDiagnostic)
    assert detail.causes[0].violation is CapturedReferenceViolation.UNPROVED_EFFECTS
    assert CapturedReferenceViolation.UNADMITTED_IMPORT in tuple(
        cause.violation for cause in detail.causes
    )
    if not observe_before_classes:
        assert any(
            cause.violation is CapturedReferenceViolation.UNPROVED_EFFECTS
            and cause.message
            == "Native object identity remains open: unproved_execution_effects"
            for cause in detail.causes
        )
    assert all(
        "Callable execution remains unproved" not in cause.message
        for cause in detail.causes
    )
    assert SourceReproofDiagnostic.from_json_value(json_report_object(detail)) == detail
