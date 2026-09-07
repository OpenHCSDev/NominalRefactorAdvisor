"""Generic rejected proposals retain evidence without becoming executable."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    PatchTargetOperation,
    RefactorRecipe,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodPlanPreflightReport,
)
from nominal_refactor_advisor.codemod_runtime import (
    ExecutableRecipeEvaluation,
    FindingRecipeSynthesisRecord,
    RejectedRecipeProposalEvaluation,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import AuthorityClaim


@pytest.fixture(params=(False, True), ids=("relative-source", "absolute-source"))
def invalid_proposal(request, tmp_path):
    source_path = (
        tmp_path / "simple.py" if request.param else Path("simple.py")
    ).as_posix()
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {source_path: "def chosen(): return 1\n"}
    )
    operation = PatchTargetOperation(
        target=SourceRewriteTarget(file_path=source_path, qualname="chosen"),
        replacements=(
            SourceTextReplacement(old_source="missing text", new_source="replacement"),
        ),
    )
    recipe = RefactorRecipe(recipe_id="terminal-generic", operations=(operation,))
    evaluation = ExecutableRecipeEvaluation(
        executable_recipe=recipe, evaluation_declaration_type=PatchTargetOperation
    )
    return snapshot, evaluation


def test_terminal_generic_failure_retains_original_report_instance(
    invalid_proposal, monkeypatch
):
    snapshot, evaluation = invalid_proposal
    observed = []
    native_init = CodemodOperationPreflightError.__init__

    def observe(error, report):
        observed.append(report)
        native_init(error, report)

    monkeypatch.setattr(CodemodOperationPreflightError, "__init__", observe)
    result = evaluation.terminal_evaluation(snapshot)
    assert isinstance(result, RejectedRecipeProposalEvaluation)
    assert result.proposal is evaluation.executable_recipe
    assert result.candidate_recipes == ()
    assert observed
    assert any(report is result.preflight.reports[0] for report in observed)


def test_no_context_gate_does_not_query_operation_preflight(
    invalid_proposal, monkeypatch
):
    _, evaluation = invalid_proposal

    def reject_query(self, context):
        pytest.fail("Operation preflight must not receive an absent source context")

    monkeypatch.setattr(PatchTargetOperation, "preflight_reports", reject_query)
    assert evaluation.gated_by_recipe_preflight(None) is evaluation


def test_failed_proposal_leaf_cannot_wrap_a_clean_report():
    with pytest.raises(ValueError, match="failed preflight"):
        RejectedRecipeProposalEvaluation(
            evaluation_declaration_type=PatchTargetOperation,
            proposed_recipe=RefactorRecipe(recipe_id="no-failure"),
            preflight=CodemodPlanPreflightReport(()),
        )


def test_shared_gate_retains_exact_aggregate_once(invalid_proposal, monkeypatch):
    snapshot, evaluation = invalid_proposal
    evaluation = replace(
        evaluation,
        executable_recipe=replace(
            evaluation.executable_recipe,
            authority_claims=(AuthorityClaim(claimed_symbol="AbsentAuthority"),),
        ),
    )
    observed = []
    native_preflight = RefactorRecipe.preflight_reports

    def observe(recipe, context):
        reports = native_preflight(recipe, context)
        observed.append(reports)
        return reports

    monkeypatch.setattr(RefactorRecipe, "preflight_reports", observe)
    result = evaluation.gated_by_recipe_preflight(snapshot)
    assert isinstance(result, RejectedRecipeProposalEvaluation)
    assert result.proposal is evaluation.executable_recipe
    assert len(observed) == 1
    assert result.preflight.reports is observed[0]
    assert result.recipe is None
    assert result.candidate_recipes == ()
    with pytest.raises(TypeError, match="executable"):
        _ = result.required_recipe


def test_rejected_proposal_json_preserves_recipe_without_selecting_it(invalid_proposal):
    snapshot, evaluation = invalid_proposal
    result = evaluation.terminal_evaluation(snapshot)
    record = FindingRecipeSynthesisRecord(
        finding=RefactorFinding(
            detector_id="example",
            pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
            title="Example",
            summary="Example",
            why="Example",
            capability_gap="Example",
            relation_context="Example",
        ),
        evaluation=result,
    )
    wire = json.loads(json.dumps(json_report_object(record)))
    assert wire["recipe"] is None
    assert record.candidate_recipes == ()
    assert wire["status"] == "rejected_by_safety_check"
    assert (
        RefactorRecipe.from_json_value(wire["proposal"]) == evaluation.executable_recipe
    )
    assert wire["proposal_preflight"]["reports"]
