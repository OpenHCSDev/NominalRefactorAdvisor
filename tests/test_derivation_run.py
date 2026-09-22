from __future__ import annotations

from dataclasses import replace

import pytest

from nominal_refactor_advisor.derivation_run import (
    AdmittedCaseCoverageBinding,
    AdmittedRelationCase,
    AnalyzerConfiguration,
    AnalyzerProvenance,
    DerivationArtifact,
    DerivationArtifactKind,
    DerivationExclusion,
    DerivationRunBoundary,
    DerivationRunManifest,
    DerivationRunReceipt,
    DerivationRunStatus,
    DerivationSetting,
    DomainMeaning,
    DynamicBoundaryKind,
    EnumerationDecision,
    EnumerationDecisionKind,
    ExclusionEffect,
    IncompleteDerivationRunError,
    IndependentRoleDeclaration,
    KnownDynamicBoundary,
    RelationCoverageVerdict,
    RelationEvidenceKind,
    RelationEvidenceRecord,
    RequiredQuestion,
    RequiredRelationEvidenceBinding,
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
        language_runtime="Python 3.14",
        analyzer_configuration=AnalyzerConfiguration(
            analyzer_name="exact-relation",
            analyzer_version="1.0.0",
            analyzer_settings=(
                DerivationSetting("workers", "1"),
                DerivationSetting("collection_mode", "static_and_runtime"),
            ),
            solver_settings=(
                DerivationSetting("search", "bounded_exact"),
                DerivationSetting("tie_break", "lexicographic"),
            ),
        ),
        dynamic_boundaries=(
            KnownDynamicBoundary(
                "plugin-loading",
                DynamicBoundaryKind.DYNAMIC_LOADING,
                "billing.plugins",
                "freeze the installed plugin manifest at input capture",
            ),
            KnownDynamicBoundary(
                "registry-mutation",
                DynamicBoundaryKind.RUNTIME_MUTATION,
                "billing.registry",
                "reject mutation after input capture",
            ),
        ),
        dependency_revisions=(
            RevisionPin("schema", "v2"),
            RevisionPin("runtime", "def456"),
        ),
        generated_policy="exclude generated clients",
        vendored_policy="exclude",
        test_policy="include contract tests",
        migration_policy="admit import adapters only",
    )


def test_boundary_canonicalizes_typed_settings_and_dynamic_boundaries() -> None:
    left = boundary()
    right = replace(
        left,
        analyzer_configuration=replace(
            left.analyzer_configuration,
            analyzer_settings=tuple(
                reversed(left.analyzer_configuration.analyzer_settings)
            ),
            solver_settings=tuple(
                reversed(left.analyzer_configuration.solver_settings)
            ),
        ),
        dynamic_boundaries=tuple(reversed(left.dynamic_boundaries)),
    )

    assert left == right
    assert (
        left.analyzer_configuration.configuration_digest
        == right.analyzer_configuration.configuration_digest
    )
    payload = json_report_object(left)
    assert tuple(
        item["name"] for item in payload["analyzer_configuration"]["analyzer_settings"]
    ) == (
        "collection_mode",
        "workers",
    )
    assert {item["kind"] for item in payload["dynamic_boundaries"]} == {
        "dynamic_loading",
        "runtime_mutation",
    }


@pytest.mark.parametrize("field_name", ("analyzer_settings", "solver_settings"))
def test_boundary_rejects_duplicate_setting_names(field_name: str) -> None:
    value = boundary().analyzer_configuration
    duplicate_settings = (
        DerivationSetting("same", "first"),
        DerivationSetting("same", "second"),
    )

    with pytest.raises(ValueError, match=f"{field_name} names must not contain"):
        replace(value, **{field_name: duplicate_settings})


def test_boundary_rejects_duplicate_dynamic_boundary_identities() -> None:
    value = boundary()
    with pytest.raises(ValueError, match="dynamic boundary ids must not contain"):
        replace(
            value,
            dynamic_boundaries=(
                value.dynamic_boundaries[0],
                replace(
                    value.dynamic_boundaries[0],
                    policy="a conflicting policy",
                ),
            ),
        )


def manifest(*, reverse: bool = False) -> DerivationRunManifest:
    meanings = (
        DomainMeaning("invoice", "billing", "A payable invoice."),
        DomainMeaning("credit", "billing", "A credit against an invoice."),
        DomainMeaning(
            "legacy_wire_alias",
            "billing",
            "A compatibility-only historical wire identity.",
        ),
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
    cases = (
        AdmittedRelationCase(
            "ledger-current",
            "tests/test_ledger.py",
            "Current ledger access cases.",
        ),
        AdmittedRelationCase(
            "statement-current",
            "contracts/billing.md#statements",
            "Current statement contract cases.",
        ),
        AdmittedRelationCase(
            "zero-pair-case",
            "contracts/billing.md#no-posting",
            "A covered case that intentionally requires no relation row.",
        ),
    )
    evidence = (
        RelationEvidenceRecord(
            "contract-statement",
            RelationEvidenceKind.CITED_CONTRACT,
            "contracts/billing.md#statements",
            "manual contract review",
            "Statements must render every invoice.",
        ),
        RelationEvidenceRecord(
            "observed-credit",
            RelationEvidenceKind.OBSERVED_CODE,
            "src/billing/ledger.py:42",
            "static access-site collection",
            "The ledger dispatches credit records.",
        ),
        RelationEvidenceRecord(
            "observed-invoice",
            RelationEvidenceKind.OBSERVED_CODE,
            "src/billing/ledger.py:38",
            "static access-site collection",
            "The ledger dispatches invoice records.",
        ),
        RelationEvidenceRecord(
            "zero-pair-contract",
            RelationEvidenceKind.CITED_DOMAIN,
            "contracts/billing.md#no-posting",
            "manual domain review",
            "This admitted case intentionally requires no pair.",
        ),
    )
    relation_evidence_bindings = (
        RequiredRelationEvidenceBinding(
            relation[0].row_id,
            ("observed-credit",),
        ),
        RequiredRelationEvidenceBinding(
            relation[1].row_id,
            ("observed-invoice",),
        ),
        RequiredRelationEvidenceBinding(
            relation[2].row_id,
            ("contract-statement",),
        ),
    )
    case_coverage_bindings = (
        AdmittedCaseCoverageBinding(
            "ledger-current",
            (relation[0].row_id, relation[1].row_id),
            ("observed-credit", "observed-invoice"),
        ),
        AdmittedCaseCoverageBinding(
            "statement-current",
            (relation[2].row_id,),
            ("contract-statement",),
        ),
        AdmittedCaseCoverageBinding(
            "zero-pair-case",
            (),
            ("zero-pair-contract",),
        ),
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
        admitted_cases=tuple(reversed(cases)) if reverse else cases,
        relation_evidence=tuple(reversed(evidence)) if reverse else evidence,
        required_relation=tuple(reversed(relation)) if reverse else relation,
        relation_evidence_bindings=(
            tuple(reversed(relation_evidence_bindings))
            if reverse
            else relation_evidence_bindings
        ),
        admitted_case_coverage_bindings=(
            tuple(reversed(case_coverage_bindings))
            if reverse
            else case_coverage_bindings
        ),
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
                divergence_identity="legacy_wire_alias",
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
        configuration_digest=(
            value.boundary.analyzer_configuration.configuration_digest
        ),
        input_digest=value.input_digest,
        invocation=("exact-relation", "analyzer-input.json"),
    )


def complete_artifacts(
    run: DerivationRunManifest,
) -> tuple[DerivationArtifact, ...]:
    return tuple(
        DerivationArtifact(
            kind=kind,
            relative_path=f"{kind.value}.json",
            sha256=(
                run.required_relation_binding_receipt.content_digest
                if kind is DerivationArtifactKind.RELATION_BINDING
                else f"{index:064x}"
            ),
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
    assert left.required_relation[0].row_id.startswith("required-relation-row:")
    assert {row["kind"] for row in payload["relation_evidence"]} == {
        "cited_contract",
        "cited_domain",
        "observed_code",
    }


def test_relation_binding_receipt_references_authorities_and_derives_coverage() -> None:
    blocked_manifest = manifest(reverse=True)
    blocked = blocked_manifest.required_relation_binding_receipt
    covered = replace(
        blocked_manifest,
        exclusions=(),
    ).required_relation_binding_receipt

    assert blocked.coverage_verdict is RelationCoverageVerdict.BLOCKED
    assert blocked.unresolved_exclusion_ids == ("plugin-keys",)
    assert covered.coverage_verdict is RelationCoverageVerdict.COVERED
    assert covered.unresolved_exclusion_ids == ()
    assert (
        tuple(binding.row_id for binding in covered.relation_evidence_bindings)
        == covered.relation_row_ids
    )
    assert all(binding.evidence_ids for binding in covered.relation_evidence_bindings)
    assert (
        tuple(binding.case_id for binding in covered.admitted_case_coverage_bindings)
        == covered.admitted_case_ids
    )
    zero_pair_binding = next(
        binding
        for binding in covered.admitted_case_coverage_bindings
        if binding.case_id == "zero-pair-case"
    )
    assert zero_pair_binding.row_ids == ()
    assert zero_pair_binding.evidence_ids == ("zero-pair-contract",)
    payload = json_report_object(covered)
    assert payload["repository_revision"] == "abc123"
    assert payload["coverage_verdict"] == "covered"
    assert len(payload["boundary_digest"]) == 64
    assert "required_relation" not in payload
    assert "evidence_records" not in payload
    assert payload == json_report_object(
        replace(manifest(), exclusions=()).required_relation_binding_receipt
    )


def test_manifest_digest_changes_with_one_required_pair() -> None:
    original = manifest()
    added_pair = RequiredRelationRow("credit", "statement")
    changed = replace(
        original,
        required_relation=(*original.required_relation, added_pair),
        relation_evidence=(
            *original.relation_evidence,
            RelationEvidenceRecord(
                "contract-credit-statement",
                RelationEvidenceKind.CITED_CONTRACT,
                "contracts/billing.md#credits",
                "manual contract review",
                "Statements must render every credit.",
            ),
        ),
        relation_evidence_bindings=(
            *original.relation_evidence_bindings,
            RequiredRelationEvidenceBinding(
                added_pair.row_id,
                ("contract-credit-statement",),
            ),
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
        lambda value: replace(
            value,
            relation_evidence=(
                value.relation_evidence[0],
                replace(
                    value.relation_evidence[0],
                    description="different evidence with the same identity",
                ),
            ),
        ),
        lambda value: replace(
            value,
            relation_evidence=value.relation_evidence[:-1],
        ),
        lambda value: replace(
            value,
            relation_evidence_bindings=(
                replace(
                    value.relation_evidence_bindings[0],
                    row_id="required-relation-row:" + "0" * 64,
                ),
                *value.relation_evidence_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            admitted_cases=(
                *value.admitted_cases,
                AdmittedRelationCase(
                    "unbound-case",
                    "tests/test_unbound.py",
                    "A case with no relation binding.",
                ),
            ),
        ),
        lambda value: replace(
            value,
            enumeration_decisions=(
                replace(
                    value.enumeration_decisions[1],
                    divergence_identity="unknown-divergence",
                ),
            ),
        ),
    ),
)
def test_manifest_rejects_incomplete_or_foreign_domain_inputs(invalid) -> None:
    with pytest.raises(ValueError):
        invalid(manifest())


def test_row_evidence_binding_has_no_case_field_or_json() -> None:
    binding = manifest().relation_evidence_bindings[0]

    assert not hasattr(binding, "case_id")
    assert not hasattr(binding, "case_ids")
    assert not hasattr(binding, "admitted_case_ids")
    assert set(json_report_object(binding)) == {"row_id", "evidence_ids"}


def test_case_coverage_binding_requires_evidence_even_with_zero_rows() -> None:
    with pytest.raises(ValueError, match="evidence_ids must not be empty"):
        AdmittedCaseCoverageBinding("zero-pair-case", (), ())


@pytest.mark.parametrize(
    "invalid",
    (
        lambda value: replace(
            value,
            relation_evidence_bindings=(
                replace(
                    value.relation_evidence_bindings[0],
                    evidence_ids=("unknown-evidence",),
                ),
                *value.relation_evidence_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            admitted_case_coverage_bindings=(
                replace(
                    value.admitted_case_coverage_bindings[0],
                    case_id="unknown-case",
                ),
                *value.admitted_case_coverage_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            admitted_case_coverage_bindings=(
                replace(
                    value.admitted_case_coverage_bindings[0],
                    row_ids=("unknown-row",),
                ),
                *value.admitted_case_coverage_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            relation_evidence_bindings=(
                replace(
                    value.relation_evidence_bindings[0],
                    evidence_ids=("observed-credit", "observed-credit"),
                ),
                *value.relation_evidence_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            relation_evidence_bindings=(
                value.relation_evidence_bindings[0],
                replace(
                    value.relation_evidence_bindings[0],
                    evidence_ids=("observed-invoice",),
                ),
                *value.relation_evidence_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            admitted_case_coverage_bindings=(
                value.admitted_case_coverage_bindings[0],
                replace(
                    value.admitted_case_coverage_bindings[0],
                    evidence_ids=("contract-statement",),
                ),
                *value.admitted_case_coverage_bindings[1:],
            ),
        ),
        lambda value: replace(
            value,
            admitted_case_coverage_bindings=(
                *value.admitted_case_coverage_bindings[:-1],
            ),
        ),
        lambda value: replace(
            value,
            relation_evidence=(
                *value.relation_evidence,
                RelationEvidenceRecord(
                    "unused-evidence",
                    RelationEvidenceKind.CITED_TEST,
                    "tests/test_unused.py",
                    "test inspection",
                    "Evidence that no binding references.",
                ),
            ),
        ),
    ),
)
def test_manifest_rejects_unknown_duplicate_missing_or_unused_binding_refs(
    invalid,
) -> None:
    with pytest.raises(ValueError):
        invalid(manifest())


def test_manifest_requires_every_relation_pair_to_have_evidence() -> None:
    value = manifest()
    with pytest.raises(ValueError, match="incomplete coverage"):
        replace(
            value,
            relation_evidence_bindings=value.relation_evidence_bindings[:-1],
        )


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
            "requires a divergence identity",
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
            divergence_identity=divergence,
        )


def test_artifact_roster_uses_one_atomic_factorization_certificate() -> None:
    values = {kind.value for kind in DerivationArtifactKind}

    assert "relation_factorization_certificate" in values
    assert "gap_certificate" not in values
    assert "provider_cover" not in values
    assert "residual_gap" not in values


@pytest.mark.parametrize(
    "relative_path",
    ("/tmp/gap.json", "../gap.json", "proofs/../../gap.json", r"proofs\gap.json"),
)
def test_artifact_paths_cannot_escape_the_portable_run_directory(
    relative_path: str,
) -> None:
    with pytest.raises(ValueError, match="run directory|POSIX separators"):
        DerivationArtifact(
            DerivationArtifactKind.RELATION_FACTORIZATION_CERTIFICATE,
            relative_path,
            "1" * 64,
        )


def test_artifact_and_analyzer_digests_are_exact_sha256_values() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DerivationArtifact(
            DerivationArtifactKind.RELATION_FACTORIZATION_CERTIFICATE,
            "relation-factorization-certificate.json",
            "A" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        AnalyzerProvenance("short", "1" * 64, ("analyzer",))


def test_receipt_derives_complete_state_from_bound_provenance_and_artifacts() -> None:
    run = replace(manifest(), exclusions=())
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=analyzer_for(run),
        artifacts=tuple(reversed(complete_artifacts(run))),
    )

    assert receipt.status is DerivationRunStatus.COMPLETE
    assert receipt.missing_artifact_kinds == ()
    assert receipt.require_complete() is receipt
    payload = json_report_object(receipt)
    assert payload["manifest_digest"] == run.input_digest
    assert payload["status"] == "complete"
    assert payload["missing_artifact_kinds"] == ()
    assert payload["unresolved_exclusion_ids"] == ()
    assert tuple(item["kind"] for item in payload["artifacts"]) == tuple(
        sorted(kind.value for kind in DerivationArtifactKind)
    )


def test_receipt_rejects_arbitrary_relation_binding_artifact_digest() -> None:
    run = replace(manifest(), exclusions=())
    artifacts = tuple(
        replace(artifact, sha256="f" * 64)
        if artifact.kind is DerivationArtifactKind.RELATION_BINDING
        else artifact
        for artifact in complete_artifacts(run)
    )

    with pytest.raises(ValueError, match="does not match the typed receipt"):
        DerivationRunReceipt(
            manifest=run,
            analyzer=analyzer_for(run),
            artifacts=artifacts,
        )


def test_completion_requires_relation_binding_artifact_and_positive_verdict() -> None:
    run = replace(manifest(), exclusions=())
    artifacts_without_binding = tuple(
        artifact
        for artifact in complete_artifacts(run)
        if artifact.kind is not DerivationArtifactKind.RELATION_BINDING
    )
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=analyzer_for(run),
        artifacts=artifacts_without_binding,
    )

    assert (
        run.required_relation_binding_receipt.coverage_verdict
        is RelationCoverageVerdict.COVERED
    )
    assert receipt.status is DerivationRunStatus.INCOMPLETE
    assert receipt.missing_artifact_kinds == (DerivationArtifactKind.RELATION_BINDING,)


def test_unresolved_relation_exclusion_blocks_otherwise_complete_receipt() -> None:
    run = manifest()
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=analyzer_for(run),
        artifacts=complete_artifacts(run),
    )

    assert (
        run.required_relation_binding_receipt.coverage_verdict
        is RelationCoverageVerdict.BLOCKED
    )
    assert receipt.status is DerivationRunStatus.INCOMPLETE
    assert receipt.missing_artifact_kinds == ()
    assert receipt.unresolved_exclusion_ids == ("plugin-keys",)
    with pytest.raises(
        IncompleteDerivationRunError,
        match="unresolved exclusions: plugin-keys",
    ):
        receipt.require_complete()


def test_receipt_retains_explicit_incomplete_state_and_fails_loud_on_demand() -> None:
    run = manifest()
    receipt = DerivationRunReceipt(
        manifest=run,
        analyzer=None,
        artifacts=(complete_artifacts(run)[0],),
        blockers=("runtime plugin keys remain unresolved",),
    )

    assert receipt.status is DerivationRunStatus.INCOMPLETE
    assert (
        DerivationArtifactKind.RELATION_FACTORIZATION_CERTIFICATE
        in receipt.missing_artifact_kinds
    )
    assert receipt.unresolved_exclusion_ids == ("plugin-keys",)
    with pytest.raises(
        IncompleteDerivationRunError,
        match=(
            "analyzer provenance is absent.*missing artifacts.*"
            "unresolved exclusions: plugin-keys.*runtime plugin keys"
        ),
    ):
        receipt.require_complete()


def test_receipt_rejects_analyzer_configuration_digest_mismatch() -> None:
    run = replace(manifest(), exclusions=())
    wrong_provenance = replace(
        analyzer_for(run),
        configuration_digest="f" * 64,
    )

    with pytest.raises(ValueError, match="does not match the run boundary"):
        DerivationRunReceipt(
            manifest=run,
            analyzer=wrong_provenance,
            artifacts=complete_artifacts(run),
        )


def test_receipt_rejects_analyzer_provenance_for_another_manifest() -> None:
    run = manifest()
    other = replace(
        run,
        relation_evidence=(
            replace(
                run.relation_evidence[0],
                description="The contract was revised after this analyzer run.",
            ),
            *run.relation_evidence[1:],
        ),
    )

    with pytest.raises(ValueError, match="does not match the run manifest"):
        DerivationRunReceipt(
            manifest=run,
            analyzer=analyzer_for(other),
        )


def test_receipt_rejects_duplicate_artifact_authorities() -> None:
    run = manifest()
    artifact = complete_artifacts(run)[0]
    with pytest.raises(ValueError, match="artifact kinds must not contain duplicates"):
        DerivationRunReceipt(
            manifest=run,
            analyzer=None,
            artifacts=(artifact, replace(artifact, relative_path="copy.json")),
        )
