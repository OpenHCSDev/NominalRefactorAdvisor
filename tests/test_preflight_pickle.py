"""Worker-pipe pickle round-trip regression tests for structured NRA exceptions."""

from __future__ import annotations

import pickle

from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
    CodemodPlanPreflightError,
    CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import SourceReproofDiagnostic
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


def test_needs_input_preserves_needs_across_pickle() -> None:
    original = NeedsInput("needs an owner name", NeededInput.OWNER_PICK, NeededInput.PROOF)
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, NeedsInput)
    assert str(restored) == "needs an owner name"
    assert restored.needs == (NeededInput.OWNER_PICK, NeededInput.PROOF)


def test_codemod_preflight_errors_round_trip_across_pickle() -> None:
    op_report = CodemodOperationPreflightReport(
        operation="extract_sibling_override_base",
        status=CodemodPreflightStatus.FAILED,
        message="missing owner pick",
        detail=SourceReproofDiagnostic(
            SourceRewriteTarget(file_path="pkg/a.py", qualname="Foo"), ()
        ),
        needs=(NeededInput.OWNER_PICK,),
    )
    op_error = CodemodOperationPreflightError(op_report)
    restored_op = pickle.loads(pickle.dumps(op_error))
    assert isinstance(restored_op, CodemodOperationPreflightError)
    assert restored_op.report == op_report
    assert restored_op.needs == (NeededInput.OWNER_PICK,)
    assert str(restored_op) == "missing owner pick"

    plan_report = CodemodPlanPreflightReport(reports=(op_report,))
    plan_error = CodemodPlanPreflightError(plan_report)
    restored_plan = pickle.loads(pickle.dumps(plan_error))
    assert isinstance(restored_plan, CodemodPlanPreflightError)
    assert restored_plan.plan == plan_report
    assert restored_plan.reports == (op_report,)
    assert restored_plan.needs == (NeededInput.OWNER_PICK,)
    assert str(restored_plan) == "missing owner pick"
