"""Exercise the simulated nominal rejection and report boundary, not messages."""

import ast
import json
from pathlib import Path
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    DeriveCandidateCollectorOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import SourceReproofDiagnostic
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor import cli
from nominal_refactor_advisor.detectors import _base as collector_runtime

TARGET = SourceRewriteTarget(file_path="subject.py", qualname="Example.collect")


def operation():
    return DeriveCandidateCollectorOperation(target=TARGET)


def throw(error):
    raise error


@pytest.mark.parametrize(
    "query",
    (
        lambda c: c.require_closed(),
        lambda c: c.require_native(()),
        lambda c: c.require_release(),
        lambda c: c.require_class_installation(),
        lambda c: c.call_authority(None, None, None),
        lambda c: c.require_plain_class_base(None, None, None),
        lambda c: c.require_attribute_write(None, "field", None, None),
    ),
)
def test_nominal_exception_retains_original_evidence_and_cause(query):
    cause = ValueError("earlier obligation")
    evidence = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=cause
    )
    errors = []
    for _ in range(2):
        with pytest.raises(CapturedReferenceRejection) as rejected:
            query(evidence)
        error = rejected.value
        assert error.evidence is evidence
        assert error.violation is evidence.violation
        assert error.__cause__ is cause
        assert "violation" not in vars(error)
        errors.append(error)
    assert errors[0] is not errors[1]


def test_preflight_preserves_nested_capture_reasons_without_runtime_graph():
    original = OpenCapturedReference(CapturedReferenceViolation.UNADMITTED_IMPORT)
    earlier = original.rejection("Import identity")
    latest = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=earlier
    )
    with pytest.raises(CodemodOperationPreflightError) as rejected:
        operation().required_reproof(latest.require_closed)
    report = rejected.value.report
    detail = report.detail
    assert isinstance(detail, SourceReproofDiagnostic)
    assert detail.target is TARGET
    assert tuple(c.violation for c in detail.causes) == (
        CapturedReferenceViolation.UNPROVED_EFFECTS,
        CapturedReferenceViolation.UNADMITTED_IMPORT,
    )
    assert rejected.value.__cause__.evidence is latest
    payload = json_report_object(report)
    encoded = json.dumps(payload)
    assert "evidence" not in encoded
    assert "traceback" not in encoded
    assert "mutation" not in encoded
    decoded = SourceReproofDiagnostic.from_json_value(payload["details"])
    assert decoded == detail
    with pytest.raises(CodemodOperationPreflightError) as reconstructed:
        CodemodPlanPreflightReport((report,)).require_clean()
    assert reconstructed.value.__cause__ is None
    assert reconstructed.value.report.detail == detail


@pytest.mark.parametrize("kind", (TypeError, ValueError))
def test_generic_exception_does_not_invent_semantic_classification(kind):
    # Deliberately identical to a real enum's text. Wording cannot confer a type.
    error = kind(CapturedReferenceViolation.UNADMITTED_IMPORT.value)
    with pytest.raises(CodemodOperationPreflightError) as rejected:
        operation().required_reproof(lambda: throw(error))
    assert rejected.value.report.detail.causes[0].violation is None
    assert rejected.value.report.detail.causes[0].message == str(error)


def test_message_change_does_not_change_typed_violation():
    error = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_BINDING
    ).rejection("Query")
    error.args = ("completely different explanatory prose",)
    detail = SourceReproofDiagnostic.from_error(TARGET, error)
    assert detail.causes[0].violation is CapturedReferenceViolation.UNPROVED_BINDING
    assert detail.causes[0].message == str(error)


def test_explicit_cause_cycles_terminate_without_structural_equality():
    first = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_BINDING
    ).rejection("One")
    second = ValueError("Two")
    first.__cause__ = second
    second.__cause__ = first
    detail = SourceReproofDiagnostic.from_error(TARGET, first)
    assert tuple(c.violation for c in detail.causes) == (
        CapturedReferenceViolation.UNPROVED_BINDING,
        None,
    )


def test_implicit_context_is_not_presented_as_explicit_proof_cause():
    error = ValueError("actual")
    error.__context__ = OpenCapturedReference(
        CapturedReferenceViolation.UNADMITTED_IMPORT
    ).rejection("unrelated")
    detail = SourceReproofDiagnostic.from_error(TARGET, error)
    assert len(detail.causes) == 1
    assert detail.causes[0].violation is None


def test_existing_structured_preflight_is_not_rewrapped():
    original = operation().failed_preflight(ValueError("already reported"))
    with pytest.raises(CodemodOperationPreflightError) as rejected:
        operation().required_reproof(lambda: throw(original))
    assert rejected.value is original


def test_actual_unadmitted_import_survives_source_reproof_boundary(capsys):
    source = "import deliberately_unadmitted_dependency\nselected = property\n"
    module = ParsedModule(
        Path("subject.py"), "subject", False, ast.parse(source), source
    )
    execution = SourceModuleExecution.from_module(module)
    read = module.module.body[-1].value
    with pytest.raises(CodemodOperationPreflightError) as rejected:
        operation().required_reproof(
            lambda: execution.capture_value(read).require_closed()
        )
    violations = tuple(c.violation for c in rejected.value.report.detail.causes)
    assert CapturedReferenceViolation.UNADMITTED_IMPORT in violations
    assert capsys.readouterr().out == ""


def test_non_reproof_errors_propagate_unchanged():
    original = RuntimeError("not a declared refusal")
    with pytest.raises(RuntimeError) as rejected:
        operation().required_reproof(lambda: throw(original))
    assert rejected.value is original


def test_cli_preserves_actual_collector_import_refusal(tmp_path, monkeypatch, capsys):
    source = (
        "from nominal_refactor_advisor.detectors._base import (\n"
        "    CandidateFindingDetector, DetectorConfig, ParsedModule,\n"
        ")\n"
        "class Candidate:\n    pass\n"
        "def _candidates(module, config):\n    return ()\n"
        "class AlphaDetector(CandidateFindingDetector[Candidate]):\n"
        "    detector_id = 'alpha'\n"
        "    def _candidate_items(self, module: ParsedModule, config: DetectorConfig):\n"
        "        return _candidates(module, config)\n"
        "    def _finding_for_candidate(self, candidate):\n"
        "        return candidate\n"
    )
    module_path = tmp_path / "collector_case.py"
    module_path.write_text(source)
    recipe = RefactorRecipe("diagnostic-control").with_operation(
        DeriveCandidateCollectorOperation(
            target=SourceRewriteTarget(
                file_path=str(module_path),
                qualname="AlphaDetector._candidate_items",
            )
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(json_report_object(CodemodPlanDocument(recipes=(recipe,))))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nra",
            str(module_path),
            collector_runtime.__file__,
            "--codemod-plan",
            str(plan_path),
            "--codemod-preflight",
            "--json",
        ],
    )
    assert cli.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    report = payload["reports"][0]
    detail = SourceReproofDiagnostic.from_json_value(report["details"])
    assert CapturedReferenceViolation.UNADMITTED_IMPORT in tuple(
        c.violation for c in detail.causes
    )
    assert module_path.read_text() == source
