from __future__ import annotations

from dataclasses import replace

import pytest

from nominal_refactor_advisor.derivation_run import (
    AnalyzerProvenance,
    DerivationArtifact,
    DerivationArtifactKind,
    DerivationExclusion,
    DerivationRunBoundary,
    DerivationRunManifest,
    DerivationRunReceipt,
    DerivationRunStatus,
    DomainMeaning,
    EnumerationDecision,
    EnumerationDecisionKind,
    ExclusionEffect,
    IncompleteDerivationRunError,
    IndependentRoleDeclaration,
    RequiredQuestion,
    RequiredRelationRow,
    RevisionPin,
    StructuralCollision,
)
from nominal_refactor_advisor.json_reports import json_report_object


def boundary() -> DerivationRunBoundary:
    return DerivationRunBoundary(
        repository="example/repository",
        repository_revision="abc123",
        scope_label="billing",
        included_roots=("src/billing", "src/contracts"),
        dependency_revisions=(
            RevisionPin("schema", "v2"),
            RevisionPin("runtime", "def456"),
        ),
        generated_policy="exclude generated clients",
        vendored_policy="exclude",
        test_policy="include contract tests",
        migration_policy="admit import adapters only",
        language_runtime="Python 3.14",
    )


def manifest(*, reverse: bool = False) -> DerivationRunManifest:
    meanings = (
        DomainMeaning("invoice", "billing", "A payable invoice."),
        DomainMeaning("credit", "billing", "A credit against an invoice."),
    )
    questions = (
        RequiredQuestion(
            "display_label",
            "billing",
            "Which label must a consumer display?",
        ),
        RequiredQuestion(
            "posting_account",
            "billing",
            "Which account receives the posting?",
        ),
    )
    relation = (
        RequiredRelationRow("credit", "ledger"),
        RequiredRelationRow("invoice", "ledger"),
        RequiredRelationRow("invoice", "statement"),
    )
    return DerivationRunManifest(
        run_id="billing-v1",
        boundary=boundary(),
        domain_meanings=tuple(reversed(meanings)) if reverse else meanings,
        required_questions=tuple(reversed(questions)) if reverse else questions,
        structural_collisions=(
            StructuralCollision(
                "mapping with amount and account keys",
                ("invoice", "credit"),
                ("posting_account",),
            ),
        ),
        independent_roles=(
            IndependentRoleDeclaration(
                "credit",
                "Credits and invoices can change posting policy independently.",
            ),
        ),
        required_relation=tuple(reversed(relation)) if reverse else relation,
        exclusions=(
            DerivationExclusion(
                exclusion_id="plugin-keys",
                boundary="third-party plugins",
                reason="plugin keys are constructed at runtime",
                possible_effect=ExclusionEffect.ADD_OR_REMOVE_PAIRS,
                affected_requirements=("posting_account", "display_label"),
                admission_condition="capture the installed plugin manifest",
            ),
        ),
        enumeration_decisions=(
            EnumerationDecision(
                site="billing.EXPORTED_TYPES",
                kind=EnumerationDecisionKind.REPLICA,
                determining_authority="billing.Record",
                resolution="derive the export from nominal record declarations",
            ),
            EnumerationDecision(
                site="billing.migration_aliases",
                kind=EnumerationDecisionKind.NAMED_DIVERGENCE,
                determining_authority="billing.Record",
                resolution="retain the import-boundary alias role",
                named_divergence="historical wire names remain accepted",
            ),
            EnumerationDecision(
                site="billing.Record",
                kind=EnumerationDecisionKind.GENUINE_AUTHORITY,
                determining_authority=None,
                resolution="register the domain record family",
            ),
        ),
    )


def analyzer_for(value: DerivationRunManifest) -> AnalyzerProvenance:
    return AnalyzerProvenance(
        analyzer_name="exact-relation",
        analyzer_version="1.0.0",
        input_digest=value.input_digest,
        invocation=("exact-relation", "analyzer-input.json"),
    )


def complete_artifacts() -> tuple[DerivationArtifact, ...]:
    return tuple(
        DerivationArtifact(
            kind=kind,
            relative_path=f"{kind.value}.json",
            sha256=f"{index:064x}",
        )
        for index, kind in enumerate(DerivationArtifactKind, start=1)
    )


def test_manifest_canonicalizes_authored_sets_without_changing_relation_rows() -> None:
    left = manifest()
    right = manifest(reverse=True)

    assert left == right
    assert left.canonical_json == right.canonical_json
    assert left.input_digest == right.input_digest
    assert left.required_relation == (
        RequiredRelationRow("credit", "ledger"),
        RequiredRelationRow("invoice", "ledger"),
        RequiredRelationRow("invoice", "statement"),
    )
    payload = json_report_object(left)
    assert payload["required_relation"][0] == {
        "implementation_key": "credit",
        "consumer": "ledger",
    }


def test_manifest_digest_changes_with_one_required_pair() -> None:
    original = manifest()
    changed = replace(
        original,
        required_relation=(
            *original.required_relation,
            RequiredRelationRow("credit", "statement"),
        ),
    )

    assert changed.input_digest != original.input_digest


@pytest.mark.parametrize(
    "invalid",
    (
        lambda value: replace(value, domain_meanings=()),
        lambda value: replace(value, required_questions=()),
        lambda value: replace(
            value,
            domain_meanings=(value.domain_meanings[0], value.domain_meanings[0]),
        ),
        lambda value: replace(
            value,
            independent_roles=(
                IndependentRoleDeclaration("unknown", "independent future"),
            ),
        ),
        lambda value: replace(
            value,
            structural_collisions=(
                StructuralCollision(
                    "same shape",
                    ("invoice", "unknown"),
                    ("posting_account",),
                ),
            ),
        ),
        lambda value: replace(
            value,
            independent_roles=(
                IndependentRoleDeclaration("credit", "reason one"),
                IndependentRoleDeclaration("credit", "reason two"),
            ),
        ),
        lambda value: replace(
            value,
            exclusions=(
                value.exclusions[0],
                replace(value.exclusions[0], reason="another reason"),
            ),
        ),
        lambda value: replace(
            value,
            enumeration_decisions=(
                value.enumeration_decisions[0],
                replace(
                    value.enumeration_decisions[0],
                    resolution="another resolution",
                ),
            ),
        ),
    ),
)
def test_manifest_rejects_incomplete_or_foreign_domain_inputs(invalid) -> None:
    with pytest.raises(ValueError):
        invalid(manifest())


def test_required_relation_rejects_duplicate_pairs() -> None:
    value = manifest()
    with pytest.raises(
        ValueError, match="required_relation must not contain duplicates"
    ):
        replace(
            value,
            required_relation=(
                value.required_relation[0],
                value.required_relation[0],
            ),
        )


@pytest.mark.parametrize(
    ("kind", "authority", "divergence", "message"),
    (
        (
            EnumerationDecisionKind.GENUINE_AUTHORITY,
            "other.Owner",
            None,
            "cannot name another determining authority",
        ),
        (
            EnumerationDecisionKind.REPLICA,
            None,
            None,
            "requires a determining authority",
        ),
        (
            EnumerationDecisionKind.REPLICA,
            "Owner",
            "possible override",
            "replica cannot name a divergence",
        ),
        (
            EnumerationDecisionKind.NAMED_DIVERGENCE,
            "Owner",
            None,
            "requires a divergence receipt",
        ),
    ),
)
def test_enumeration_decisions_enforce_three_way_receipts(
    kind: EnumerationDecisionKind,
    authority: str | None,
    divergence: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EnumerationDecision(
            site="module.ROSTER",
            kind=kind,
            determining_authority=authority,
            resolution="record the decision",
            named_divergence=divergence,
        )


@pytest.mark.parametrize(
    "relative_path",
    ("/tmp/gap.json", "../gap.json", "proofs/../../gap.json", r"proofs\gap.json"),
)
def test_artifact_paths_cannot_escape_the_portable_run_directory(
    relative_path: str,
) -> None:
    with pytest.raises(ValueError, match="run directory|POSIX separators"):
        DerivationArtifact(
            DerivationArtifactKind.GAP_CERTIFICATE,
            relative_path,
            "1" * 64,
        )


def test_artifact_and_analyzer_digests_are_exact_sha256_values() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DerivationArtifact(
            DerivationArtifactKind.GAP_CERTIFICATE,
            "gap.json",
            "A" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        AnalyzerProvenance("analyzer", "1", "short", ("analyzer",))


def test_receipt_derives_complete_state_from_bound_provenance_and_artifacts() -> None:
    run = manifest()
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=analyzer_for(run),
        artifacts=tuple(reversed(complete_artifacts())),
    )

    assert receipt.status is DerivationRunStatus.COMPLETE
    assert receipt.missing_artifact_kinds == ()
    assert receipt.require_complete() is receipt
    payload = json_report_object(receipt)
    assert payload["manifest_digest"] == run.input_digest
    assert payload["status"] == "complete"
    assert payload["missing_artifact_kinds"] == ()
    assert tuple(item["kind"] for item in payload["artifacts"]) == tuple(
        sorted(kind.value for kind in DerivationArtifactKind)
    )


def test_receipt_retains_explicit_incomplete_state_and_fails_loud_on_demand() -> None:
    run = manifest()
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=None,
        artifacts=(complete_artifacts()[0],),
        blockers=("runtime plugin keys remain unresolved",),
    )

    assert receipt.status is DerivationRunStatus.INCOMPLETE
    assert DerivationArtifactKind.GAP_CERTIFICATE in receipt.missing_artifact_kinds
    with pytest.raises(
        IncompleteDerivationRunError,
        match="analyzer provenance is absent.*missing artifacts.*plugin keys",
    ):
        receipt.require_complete()


def test_receipt_rejects_analyzer_provenance_for_another_manifest() -> None:
    run = manifest()
    other = replace(
        run,
        required_relation=(
            *run.required_relation,
            RequiredRelationRow("credit", "statement"),
        ),
    )

    with pytest.raises(ValueError, match="does not match the run manifest"):
        DerivationRunReceipt(
            manifest=run,
            analyzer=analyzer_for(other),
        )


def test_receipt_rejects_duplicate_artifact_authorities() -> None:
    artifact = complete_artifacts()[0]
    with pytest.raises(ValueError, match="artifact kinds must not contain duplicates"):
        DerivationRunReceipt(
            manifest=manifest(),
            analyzer=None,
            artifacts=(artifact, replace(artifact, relative_path="copy.json")),
        )
