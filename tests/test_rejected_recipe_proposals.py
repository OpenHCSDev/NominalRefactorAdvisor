"""Blocked proposals remain inspectable without entering executable planning."""

import json
from dataclasses import replace
from types import ModuleType
import sys

import pytest
from registry_test_sources import _type_keyed_behavior_projection_source

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    DescendTypeKeyedBehaviorProjectionOperation,
    MappingSemanticMirrorRecipeStrategy,
    RefactorRecipe,
    SemanticDescentRecipeEvaluation,
)
from nominal_refactor_advisor.codemod_authority_claims import (
    AuthorityClaimDeclarationPreflightDetail,
    AuthorityClaimResolutionPreflightDetail,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
    NativeUseProvenance,
    NativeUseResolution,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
)
from nominal_refactor_advisor.codemod_runtime import (
    ExecutableRecipeEvaluation,
    FindingRecipeSynthesisAttempt,
    FindingRecipeSynthesisStatus,
    RejectedRecipeProposalEvaluation,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.semantic_descent import AuthorityClaim


@pytest.fixture
def subject():
    source = _type_keyed_behavior_projection_source()
    snapshot = CodemodSourceSnapshot.from_source_mapping({"/repo/subject.py": source})
    findings = tuple(
        finding
        for finding in analyze_modules(snapshot.parsed_modules)
        if finding.detector_id == "type_keyed_behavior_projection"
    )
    assert len(findings) == 1
    plan = snapshot.plan_from_findings(
        findings, detector_ids=("type_keyed_behavior_projection",)
    )
    return source, snapshot, findings[0], plan


def _enrich(proposal, snapshot, count=None):
    (operation,) = proposal.operations
    requirements = operation.native_use_requirements(snapshot)
    accepted = replace(
        operation,
        supported_execution=DeclaredNativeUseInvariants.from_requirements(
            requirements if count is None else requirements[:count],
            rationale="Controlled fixture retains each selected native implementation at every use.",
        ),
    )
    return replace(proposal, operations=(accepted,))


def _native_output(source, monkeypatch, name):
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(
        compile(source, "<trusted-proposal-fixture>", "exec", dont_inherit=True),
        vars(module),
    )
    event = module.NamedEvent()
    event.name = "selected"
    event.value = "fallback"
    return module.render_event(event), module.render_event_locally(event)


def test_actual_synthesis_retains_proposal_and_all_typed_native_obligations(subject):
    _, snapshot, _, plan = subject
    (record,) = plan.records
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert isinstance(record.evaluation, RejectedRecipeProposalEvaluation)
    assert record.recipe is None
    assert record.candidate_recipes == ()
    assert plan.document.recipes == ()
    assert plan.report.candidate_count == 0
    with pytest.raises(TypeError, match="executable"):
        _ = record.evaluation.required_recipe
    assert record.proposal is record.evaluation.proposed_recipe
    assert record.proposal_preflight is record.evaluation.preflight
    assert not record.proposal_preflight.is_clean
    (operation,) = record.proposal.operations
    assert isinstance(operation, DescendTypeKeyedBehaviorProjectionOperation)
    assert operation.target.target_id
    assert operation.supported_execution == DeclaredNativeUseInvariants()
    requirements = operation.native_use_requirements(snapshot)
    native_reports = tuple(
        report
        for report in record.proposal_preflight.reports
        if isinstance(report.detail, NativeUseResolution)
    )
    assert len(native_reports) == len(requirements) == 3
    assert tuple(report.detail.receipt for report in native_reports) == tuple(
        requirement.receipt for requirement in requirements
    )
    assert all(
        report.detail.provenance is not NativeUseProvenance.DECLARED
        for report in native_reports
    )
    assert any(
        isinstance(report.detail, AuthorityClaimDeclarationPreflightDetail)
        for report in record.proposal_preflight.reports
    )


def test_json_proposal_can_be_deliberately_enriched_and_replayed(subject, monkeypatch):
    source, snapshot, _, plan = subject
    (record,) = plan.records
    wire = json.loads(json.dumps(json_report_object(record)))
    assert wire["recipe"] is None
    assert wire["status"] == "rejected_by_safety_check"
    restored = RefactorRecipe.from_json_value(wire["proposal"])
    assert restored == record.proposal
    assert len(wire["proposal_preflight"]["reports"]) == len(
        record.proposal_preflight.reports
    )
    with pytest.raises(CodemodOperationPreflightError):
        CodemodPlanDocument(recipes=(restored,)).simulate(snapshot)
    accepted = _enrich(restored, snapshot)
    result = CodemodPlanDocument(recipes=(accepted,)).simulate(snapshot)
    assert result.is_clean
    after = result.required_after_snapshot.parsed_module_for_source_path(
        "/repo/subject.py"
    ).source
    assert after != source
    assert (
        _native_output(source, monkeypatch, "proposal_before")
        == _native_output(after, monkeypatch, "proposal_after")
        == ("selected", "selected")
    )
    assert record.candidate_recipes == ()
    assert (
        record.proposal.operations[0].supported_execution
        == DeclaredNativeUseInvariants()
    )


def test_partial_enrichment_does_not_promote_blocked_proposal(subject):
    _, snapshot, _, plan = subject
    accepted = _enrich(plan.records[0].proposal, snapshot, count=1)
    with pytest.raises(CodemodOperationPreflightError):
        CodemodPlanDocument(recipes=(accepted,)).simulate(snapshot)


def test_changed_source_rejects_old_proposal_native_receipts(subject):
    source, snapshot, _, plan = subject
    accepted = _enrich(plan.records[0].proposal, snapshot)
    changed = snapshot.with_virtual_sources(
        {"/repo/subject.py": source.replace("value: str", "value: int")}
    )
    with pytest.raises(CodemodOperationPreflightError, match="stale|foreign"):
        CodemodPlanDocument(recipes=(accepted,)).simulate(changed)


def test_rejection_before_source_context_has_no_fabricated_proposal(subject):
    _, _, finding, _ = subject
    record = FindingRecipeSynthesisAttempt(finding, None).evaluate()
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert record.proposal is None
    assert record.proposal_preflight is None
    assert record.recipe is None
    assert record.candidate_recipes == ()


def test_shared_gate_retains_actual_failed_authority_report_once(subject, monkeypatch):
    _, snapshot, _, plan = subject
    proposal = replace(
        _enrich(plan.records[0].proposal, snapshot),
        authority_claims=(AuthorityClaim(claimed_symbol="AbsentAuthority"),),
    )
    observed = []
    native_preflight = RefactorRecipe.preflight_reports

    def observe(recipe, context):
        reports = native_preflight(recipe, context)
        observed.append(reports)
        return reports

    monkeypatch.setattr(RefactorRecipe, "preflight_reports", observe)
    result = ExecutableRecipeEvaluation(
        executable_recipe=proposal,
        evaluation_declaration_type=DescendTypeKeyedBehaviorProjectionOperation,
    ).gated_by_recipe_preflight(snapshot)
    assert isinstance(result, RejectedRecipeProposalEvaluation)
    assert result.proposal is proposal
    assert len(observed) == 1
    assert result.preflight.reports is observed[0]
    failed = tuple(
        report for report in result.preflight.reports if report.status.is_failed
    )
    assert len(failed) == 1
    assert isinstance(failed[0].detail, AuthorityClaimResolutionPreflightDetail)
    assert result.rejection_reason == failed[0].message


def test_semantic_descent_preserves_authority_and_explicit_candidate(subject):
    _, snapshot, finding, plan = subject
    proposal = _enrich(plan.records[0].proposal, snapshot)
    evaluation = SemanticDescentRecipeEvaluation(
        executable_recipe=proposal,
        evaluation_declaration_type=DescendTypeKeyedBehaviorProjectionOperation,
        strategy_type=MappingSemanticMirrorRecipeStrategy,
    )
    assert proposal.effective_authority_claims(snapshot)
    accepted = evaluation.gated_by_authority_claim(snapshot, finding)
    assert accepted is evaluation
    assert accepted.required_recipe is proposal
    assert accepted.candidate_recipes == (proposal,)
    assert accepted.strategy_type is MappingSemanticMirrorRecipeStrategy
