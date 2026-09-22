"""Typed run boundary and completion receipts for semantic derivation.

The records in this module preserve the authored inputs and reproducibility
receipts of one bounded derivation run.  They deliberately do not implement or
interpret factoring algorithms: an analyzer owns gap, cover, and residual
semantics, while this module records its identity and immutable output digests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar, Self, TypeVar

from .json_reports import (
    SemanticRecord,
    json_report_object,
    json_report_property,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RecordT = TypeVar("_RecordT")


def _require_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty normalized text")


def _require_sha256(value: str, field_name: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_unique(values: tuple[object, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _canonicalize_records(
    owner: object,
    field_name: str,
    values: tuple[_RecordT, ...],
    *,
    key: Callable[[_RecordT], object],
) -> None:
    _require_unique(values, field_name)
    object.__setattr__(owner, field_name, tuple(sorted(values, key=key)))


@dataclass(frozen=True, order=True)
class RevisionPin(SemanticRecord):
    """One named immutable repository or dependency revision."""

    name: str
    revision: str

    def __post_init__(self) -> None:
        _require_text(self.name, "RevisionPin.name")
        _require_text(self.revision, "RevisionPin.revision")


@dataclass(frozen=True, order=True)
class DerivationSetting(SemanticRecord):
    """One canonical analyzer or solver setting fixed for a run."""

    name: str
    value: str

    def __post_init__(self) -> None:
        _require_text(self.name, "DerivationSetting.name")
        _require_text(self.value, "DerivationSetting.value")


class DynamicBoundaryKind(StrEnum):
    """Known source of runtime behavior that static collection must bound."""

    DYNAMIC_LOADING = "dynamic_loading"
    RUNTIME_MUTATION = "runtime_mutation"


@dataclass(frozen=True, order=True)
class KnownDynamicBoundary(SemanticRecord):
    """Known loading or mutation boundary and its deterministic treatment."""

    boundary_id: str
    kind: DynamicBoundaryKind
    locator: str
    policy: str

    def __post_init__(self) -> None:
        _require_text(self.boundary_id, "KnownDynamicBoundary.boundary_id")
        _require_text(self.locator, "KnownDynamicBoundary.locator")
        _require_text(self.policy, "KnownDynamicBoundary.policy")


@dataclass(frozen=True)
class DerivationRunBoundary(SemanticRecord):
    """Fixed repository, scope, policy, and runtime boundary for one run."""

    repository: str
    repository_revision: str
    scope_label: str
    included_roots: tuple[str, ...]
    language_runtime: str
    analyzer_settings: tuple[DerivationSetting, ...]
    solver_settings: tuple[DerivationSetting, ...]
    dynamic_boundaries: tuple[KnownDynamicBoundary, ...]
    dependency_revisions: tuple[RevisionPin, ...] = ()
    generated_policy: str = "exclude"
    vendored_policy: str = "exclude"
    test_policy: str = "exclude"
    migration_policy: str = "boundary_only"

    def __post_init__(self) -> None:
        for field_name in (
            "repository",
            "repository_revision",
            "scope_label",
            "generated_policy",
            "vendored_policy",
            "test_policy",
            "migration_policy",
            "language_runtime",
        ):
            _require_text(
                getattr(self, field_name),
                f"DerivationRunBoundary.{field_name}",
            )
        if not self.included_roots:
            raise ValueError("DerivationRunBoundary.included_roots must not be empty")
        for root in self.included_roots:
            _require_text(root, "DerivationRunBoundary.included_roots item")
        _canonicalize_records(
            self,
            "included_roots",
            self.included_roots,
            key=lambda value: value,
        )
        _canonicalize_records(
            self,
            "dependency_revisions",
            self.dependency_revisions,
            key=lambda pin: (pin.name, pin.revision),
        )
        for field_name in ("analyzer_settings", "solver_settings"):
            settings = getattr(self, field_name)
            _canonicalize_records(
                self,
                field_name,
                settings,
                key=lambda setting: setting.name,
            )
            _require_unique(
                tuple(setting.name for setting in settings),
                f"DerivationRunBoundary {field_name} names",
            )
        _canonicalize_records(
            self,
            "dynamic_boundaries",
            self.dynamic_boundaries,
            key=lambda boundary: boundary.boundary_id,
        )
        _require_unique(
            tuple(boundary.boundary_id for boundary in self.dynamic_boundaries),
            "DerivationRunBoundary dynamic boundary ids",
        )
        dependency_names = tuple(pin.name for pin in self.dependency_revisions)
        _require_unique(dependency_names, "DerivationRunBoundary dependency names")


@dataclass(frozen=True, order=True)
class DomainMeaning(SemanticRecord):
    """Stable domain identity recovered before structural factoring."""

    meaning_id: str
    bounded_context: str
    description: str

    def __post_init__(self) -> None:
        _require_text(self.meaning_id, "DomainMeaning.meaning_id")
        _require_text(self.bounded_context, "DomainMeaning.bounded_context")
        _require_text(self.description, "DomainMeaning.description")


@dataclass(frozen=True, order=True)
class RequiredQuestion(SemanticRecord):
    """One in-scope question whose answers maintenance must preserve."""

    question_id: str
    bounded_context: str
    description: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, "RequiredQuestion.question_id")
        _require_text(self.bounded_context, "RequiredQuestion.bounded_context")
        _require_text(self.description, "RequiredQuestion.description")


@dataclass(frozen=True, order=True)
class StructuralCollision(SemanticRecord):
    """Named meanings collapsed by one current structural representation."""

    representation: str
    meaning_ids: tuple[str, ...]
    question_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.representation, "StructuralCollision.representation")
        if len(self.meaning_ids) < 2:
            raise ValueError("StructuralCollision requires at least two meanings")
        if not self.question_ids:
            raise ValueError("StructuralCollision requires at least one question")
        for field_name, values in (
            ("meaning_ids", self.meaning_ids),
            ("question_ids", self.question_ids),
        ):
            for value in values:
                _require_text(value, f"StructuralCollision.{field_name} item")
            _canonicalize_records(
                self,
                field_name,
                values,
                key=lambda value: value,
            )


@dataclass(frozen=True, order=True)
class RequiredRelationRow(SemanticRecord):
    """One exact two-column implementation/key-to-consumer pair."""

    implementation_key: str
    consumer: str

    def __post_init__(self) -> None:
        _require_text(
            self.implementation_key,
            "RequiredRelationRow.implementation_key",
        )
        _require_text(self.consumer, "RequiredRelationRow.consumer")

    @property
    def row_id(self) -> str:
        """Return a deterministic identity without adding a semantic column."""

        payload = json.dumps(
            (self.implementation_key, self.consumer),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"required-relation-row:{digest}"


class RelationEvidenceKind(StrEnum):
    """Permitted basis for admitting one pair to the required relation."""

    OBSERVED_CODE = "observed_code"
    CITED_CONTRACT = "cited_contract"
    CITED_TEST = "cited_test"
    CITED_DOMAIN = "cited_domain"


@dataclass(frozen=True, order=True)
class RelationEvidenceRecord(SemanticRecord):
    """Stable observation or authoritative citation referenced by bindings."""

    evidence_id: str
    kind: RelationEvidenceKind
    source: str
    collection_method: str
    description: str

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "source",
            "collection_method",
            "description",
        ):
            _require_text(
                getattr(self, field_name),
                f"RelationEvidenceRecord.{field_name}",
            )


@dataclass(frozen=True, order=True)
class AdmittedRelationCase(SemanticRecord):
    """One stable case admitted by the fixed investigation boundary."""

    case_id: str
    source: str
    description: str

    def __post_init__(self) -> None:
        _require_text(self.case_id, "AdmittedRelationCase.case_id")
        _require_text(self.source, "AdmittedRelationCase.source")
        _require_text(self.description, "AdmittedRelationCase.description")


@dataclass(frozen=True, order=True)
class RequiredRelationRowBinding(SemanticRecord):
    """References connecting one exact row to evidence and admitted cases."""

    row_id: str
    evidence_ids: tuple[str, ...]
    admitted_case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.row_id, "RequiredRelationRowBinding.row_id")
        if not self.evidence_ids:
            raise ValueError(
                "RequiredRelationRowBinding.evidence_ids must not be empty"
            )
        for field_name, values in (
            ("evidence_ids", self.evidence_ids),
            ("admitted_case_ids", self.admitted_case_ids),
        ):
            for value in values:
                _require_text(value, f"RequiredRelationRowBinding.{field_name} item")
            _canonicalize_records(
                self,
                field_name,
                values,
                key=lambda value: value,
            )


class RelationCoverageVerdict(StrEnum):
    """Derived boundary-coverage result for a relation binding artifact."""

    BLOCKED = "blocked"
    COVERED = "covered"

    @property
    def is_positive(self) -> bool:
        """Return whether the fixed boundary has complete relation coverage."""

        return self is RelationCoverageVerdict.COVERED


@dataclass(frozen=True)
class RequiredRelationBindingReceipt(SemanticRecord):
    """Reference-only deterministic contents of ``relation-binding.json``."""

    run_id: str
    boundary_digest: str
    repository_revision: str
    relation_row_ids: tuple[str, ...]
    evidence_record_ids: tuple[str, ...]
    admitted_case_ids: tuple[str, ...]
    bindings: tuple[RequiredRelationRowBinding, ...]
    unresolved_exclusion_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.run_id, "RequiredRelationBindingReceipt.run_id")
        _require_sha256(
            self.boundary_digest,
            "RequiredRelationBindingReceipt.boundary_digest",
        )
        _require_text(
            self.repository_revision,
            "RequiredRelationBindingReceipt.repository_revision",
        )
        for field_name in (
            "relation_row_ids",
            "evidence_record_ids",
            "admitted_case_ids",
            "unresolved_exclusion_ids",
        ):
            values = getattr(self, field_name)
            for value in values:
                _require_text(
                    value,
                    f"RequiredRelationBindingReceipt.{field_name} item",
                )
            _canonicalize_records(
                self,
                field_name,
                values,
                key=lambda value: value,
            )
        _canonicalize_records(
            self,
            "bindings",
            self.bindings,
            key=lambda binding: binding.row_id,
        )
        binding_row_ids = tuple(binding.row_id for binding in self.bindings)
        _require_unique(
            binding_row_ids,
            "RequiredRelationBindingReceipt binding row ids",
        )
        known_rows = frozenset(self.relation_row_ids)
        known_evidence = frozenset(self.evidence_record_ids)
        known_cases = frozenset(self.admitted_case_ids)
        bound_rows = frozenset(binding_row_ids)
        bound_evidence = frozenset(
            evidence_id
            for binding in self.bindings
            for evidence_id in binding.evidence_ids
        )
        bound_cases = frozenset(
            case_id
            for binding in self.bindings
            for case_id in binding.admitted_case_ids
        )
        unknown_rows = bound_rows - known_rows
        unknown_evidence = bound_evidence - known_evidence
        unknown_cases = bound_cases - known_cases
        if unknown_rows or unknown_evidence or unknown_cases:
            raise ValueError(
                "RequiredRelationBindingReceipt contains unknown references: "
                f"rows={tuple(sorted(unknown_rows))!r}, "
                f"evidence={tuple(sorted(unknown_evidence))!r}, "
                f"cases={tuple(sorted(unknown_cases))!r}"
            )
        uncovered_rows = known_rows - bound_rows
        orphan_evidence = known_evidence - bound_evidence
        uncovered_cases = known_cases - bound_cases
        if uncovered_rows or orphan_evidence or uncovered_cases:
            raise ValueError(
                "RequiredRelationBindingReceipt has incomplete coverage: "
                f"rows={tuple(sorted(uncovered_rows))!r}, "
                f"evidence={tuple(sorted(orphan_evidence))!r}, "
                f"cases={tuple(sorted(uncovered_cases))!r}"
            )

    @json_report_property()
    def coverage_verdict(self) -> RelationCoverageVerdict:
        return (
            RelationCoverageVerdict.BLOCKED
            if self.unresolved_exclusion_ids
            else RelationCoverageVerdict.COVERED
        )

    @property
    def canonical_json(self) -> str:
        """Return deterministic JSON for the machine-readable artifact."""

        return json.dumps(
            json_report_object(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def content_digest(self) -> str:
        """Return the SHA-256 identity of ``relation-binding.json``."""

        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class IndependentRoleDeclaration(SemanticRecord):
    """Domain evidence that one role retains an independent answer history."""

    role_id: str
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.role_id, "IndependentRoleDeclaration.role_id")
        _require_text(self.reason, "IndependentRoleDeclaration.reason")


class ExclusionEffect(StrEnum):
    """Direction of relation error an unresolved boundary can introduce."""

    ADD_PAIRS = "add_pairs"
    REMOVE_PAIRS = "remove_pairs"
    ADD_OR_REMOVE_PAIRS = "add_or_remove_pairs"


@dataclass(frozen=True, order=True)
class DerivationExclusion(SemanticRecord):
    """One unresolved source retained as an explicit relation obligation."""

    exclusion_id: str
    boundary: str
    reason: str
    possible_effect: ExclusionEffect
    affected_requirements: tuple[str, ...]
    admission_condition: str

    def __post_init__(self) -> None:
        for field_name in (
            "exclusion_id",
            "boundary",
            "reason",
            "admission_condition",
        ):
            _require_text(
                getattr(self, field_name),
                f"DerivationExclusion.{field_name}",
            )
        if not self.affected_requirements:
            raise ValueError(
                "DerivationExclusion.affected_requirements must not be empty"
            )
        for requirement in self.affected_requirements:
            _require_text(
                requirement,
                "DerivationExclusion.affected_requirements item",
            )
        _canonicalize_records(
            self,
            "affected_requirements",
            self.affected_requirements,
            key=lambda value: value,
        )


class EnumerationDecisionKind(StrEnum):
    """Result of the three-way derivation audit for one enumeration site."""

    GENUINE_AUTHORITY = "genuine_authority"
    REPLICA = "replica"
    NAMED_DIVERGENCE = "named_divergence"

    @property
    def requires_determining_authority(self) -> bool:
        """Return whether another authority determines this site's values."""

        return self is not EnumerationDecisionKind.GENUINE_AUTHORITY

    @property
    def requires_named_divergence(self) -> bool:
        """Return whether the site's difference needs an explicit domain name."""

        return self is EnumerationDecisionKind.NAMED_DIVERGENCE


@dataclass(frozen=True, order=True)
class EnumerationDecision(SemanticRecord):
    """Auditable authority, derivation, or divergence decision for one site."""

    site: str
    kind: EnumerationDecisionKind
    determining_authority: str | None
    resolution: str
    divergence_identity: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.site, "EnumerationDecision.site")
        _require_text(self.resolution, "EnumerationDecision.resolution")
        if self.determining_authority is not None:
            _require_text(
                self.determining_authority,
                "EnumerationDecision.determining_authority",
            )
        if self.divergence_identity is not None:
            _require_text(
                self.divergence_identity,
                "EnumerationDecision.divergence_identity",
            )
        if self.kind.requires_determining_authority:
            if self.determining_authority is None:
                raise ValueError(f"{self.kind.value} requires a determining authority")
        elif self.determining_authority is not None:
            raise ValueError(
                "a genuine authority cannot name another determining authority"
            )
        if self.kind.requires_named_divergence:
            if self.divergence_identity is None:
                raise ValueError("named_divergence requires a divergence identity")
        elif self.divergence_identity is not None:
            label = self.kind.value.replace("_", " ")
            raise ValueError(f"a {label} cannot name a divergence")


@dataclass(frozen=True)
class DerivationRunManifest(SemanticRecord):
    """Canonical authored inputs for one bounded semantic derivation run."""

    run_id: str
    boundary: DerivationRunBoundary
    domain_meanings: tuple[DomainMeaning, ...]
    required_questions: tuple[RequiredQuestion, ...]
    structural_collisions: tuple[StructuralCollision, ...] = ()
    independent_roles: tuple[IndependentRoleDeclaration, ...] = ()
    admitted_cases: tuple[AdmittedRelationCase, ...] = ()
    relation_evidence: tuple[RelationEvidenceRecord, ...] = ()
    required_relation: tuple[RequiredRelationRow, ...] = ()
    relation_bindings: tuple[RequiredRelationRowBinding, ...] = ()
    exclusions: tuple[DerivationExclusion, ...] = ()
    enumeration_decisions: tuple[EnumerationDecision, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.run_id, "DerivationRunManifest.run_id")
        orderings = (
            ("domain_meanings", self.domain_meanings, lambda row: row.meaning_id),
            (
                "required_questions",
                self.required_questions,
                lambda row: row.question_id,
            ),
            (
                "structural_collisions",
                self.structural_collisions,
                lambda row: (
                    row.representation,
                    row.meaning_ids,
                    row.question_ids,
                ),
            ),
            (
                "independent_roles",
                self.independent_roles,
                lambda row: row.role_id,
            ),
            (
                "admitted_cases",
                self.admitted_cases,
                lambda case: case.case_id,
            ),
            (
                "relation_evidence",
                self.relation_evidence,
                lambda row: row.evidence_id,
            ),
            (
                "required_relation",
                self.required_relation,
                lambda row: (row.implementation_key, row.consumer),
            ),
            (
                "relation_bindings",
                self.relation_bindings,
                lambda binding: binding.row_id,
            ),
            ("exclusions", self.exclusions, lambda row: row.exclusion_id),
            (
                "enumeration_decisions",
                self.enumeration_decisions,
                lambda row: row.site,
            ),
        )
        for field_name, values, key in orderings:
            _canonicalize_records(self, field_name, values, key=key)
        if not self.domain_meanings:
            raise ValueError("DerivationRunManifest.domain_meanings must not be empty")
        if not self.required_questions:
            raise ValueError(
                "DerivationRunManifest.required_questions must not be empty"
            )
        meaning_ids = tuple(row.meaning_id for row in self.domain_meanings)
        question_ids = tuple(row.question_id for row in self.required_questions)
        _require_unique(meaning_ids, "DerivationRunManifest meaning ids")
        _require_unique(question_ids, "DerivationRunManifest question ids")
        _require_unique(
            tuple(row.role_id for row in self.independent_roles),
            "DerivationRunManifest independent role ids",
        )
        case_ids = tuple(case.case_id for case in self.admitted_cases)
        _require_unique(case_ids, "DerivationRunManifest admitted case ids")
        evidence_ids = tuple(row.evidence_id for row in self.relation_evidence)
        _require_unique(evidence_ids, "DerivationRunManifest relation evidence ids")
        _require_unique(
            tuple(
                (row.implementation_key, row.consumer) for row in self.required_relation
            ),
            "DerivationRunManifest required relation pairs",
        )
        _require_unique(
            tuple(row.exclusion_id for row in self.exclusions),
            "DerivationRunManifest exclusion ids",
        )
        _require_unique(
            tuple(row.site for row in self.enumeration_decisions),
            "DerivationRunManifest enumeration sites",
        )
        known_meanings = frozenset(meaning_ids)
        known_questions = frozenset(question_ids)
        row_ids = tuple(row.row_id for row in self.required_relation)
        _require_unique(row_ids, "DerivationRunManifest required relation row ids")
        _ = self.required_relation_binding_receipt
        for collision in self.structural_collisions:
            unknown_meanings = frozenset(collision.meaning_ids) - known_meanings
            unknown_questions = frozenset(collision.question_ids) - known_questions
            if unknown_meanings or unknown_questions:
                raise ValueError(
                    "StructuralCollision references unknown identities: "
                    f"meanings={tuple(sorted(unknown_meanings))!r}, "
                    f"questions={tuple(sorted(unknown_questions))!r}"
                )
        unknown_divergences = (
            frozenset(
                decision.divergence_identity
                for decision in self.enumeration_decisions
                if decision.divergence_identity is not None
            )
            - known_meanings
        )
        if unknown_divergences:
            raise ValueError(
                "EnumerationDecision references unknown divergence identities: "
                f"{tuple(sorted(unknown_divergences))!r}"
            )
        unknown_roles = (
            frozenset(row.role_id for row in self.independent_roles) - known_meanings
        )
        if unknown_roles:
            raise ValueError(
                "IndependentRoleDeclaration references unknown meanings: "
                f"{tuple(sorted(unknown_roles))!r}"
            )

    @property
    def canonical_json(self) -> str:
        """Return deterministic UTF-8 JSON for analyzer input hashing."""

        return json.dumps(
            json_report_object(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def input_digest(self) -> str:
        """Return the SHA-256 identity of the canonical authored inputs."""

        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def required_relation_binding_receipt(self) -> RequiredRelationBindingReceipt:
        """Derive the standalone ``relation-binding.json`` authority."""

        boundary_json = json.dumps(
            json_report_object(self.boundary),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        boundary_digest = hashlib.sha256(boundary_json.encode("utf-8")).hexdigest()
        return RequiredRelationBindingReceipt(
            run_id=self.run_id,
            boundary_digest=boundary_digest,
            repository_revision=self.boundary.repository_revision,
            relation_row_ids=tuple(row.row_id for row in self.required_relation),
            evidence_record_ids=tuple(
                evidence.evidence_id for evidence in self.relation_evidence
            ),
            admitted_case_ids=tuple(case.case_id for case in self.admitted_cases),
            bindings=self.relation_bindings,
            unresolved_exclusion_ids=tuple(
                exclusion.exclusion_id for exclusion in self.exclusions
            ),
        )


@dataclass(frozen=True)
class AnalyzerProvenance(SemanticRecord):
    """Analyzer identity bound to the exact canonical manifest it consumed."""

    analyzer_name: str
    analyzer_version: str
    input_digest: str
    invocation: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.analyzer_name, "AnalyzerProvenance.analyzer_name")
        _require_text(self.analyzer_version, "AnalyzerProvenance.analyzer_version")
        _require_sha256(self.input_digest, "AnalyzerProvenance.input_digest")
        if not self.invocation:
            raise ValueError("AnalyzerProvenance.invocation must not be empty")
        for argument in self.invocation:
            _require_text(argument, "AnalyzerProvenance.invocation item")


class DerivationArtifactKind(StrEnum):
    """Closed receipt roster required by the general derivation protocol."""

    BOUNDARY = "boundary"
    EXAMPLE_TRANSFORMATIONS = "example_transformations"
    DOMAIN_GLOSSARY = "domain_glossary"
    REQUIRED_QUESTIONS = "required_questions"
    STRUCTURAL_COLLISIONS = "structural_collisions"
    INDEPENDENT_ROLES = "independent_roles"
    REQUIRED_RELATION = "required_relation"
    RELATION_BINDING = "relation_binding"
    EXCLUSIONS = "exclusions"
    ENUMERATION_DECISIONS = "enumeration_decisions"
    ANALYZER_INPUT = "analyzer_input"
    GAP_CERTIFICATE = "gap_certificate"
    PROVIDER_COVER = "provider_cover"
    RESIDUAL_GAP = "residual_gap"
    MIGRATION_MAP = "migration_map"
    VALIDATION = "validation"


@dataclass(frozen=True, order=True)
class DerivationArtifact(SemanticRecord):
    """Content-addressed portable artifact emitted by one derivation run."""

    kind: DerivationArtifactKind
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.relative_path, "DerivationArtifact.relative_path")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            raise ValueError(
                "DerivationArtifact.relative_path must stay within the run directory"
            )
        if "\\" in self.relative_path:
            raise ValueError(
                "DerivationArtifact.relative_path must use portable POSIX separators"
            )
        _require_sha256(self.sha256, "DerivationArtifact.sha256")


class DerivationRunStatus(StrEnum):
    """Derived completion state of a derivation run receipt."""

    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class IncompleteDerivationRunError(ValueError):
    """Raised when an incomplete receipt is required as a completed proof."""


@dataclass(frozen=True)
class DerivationRunReceipt(SemanticRecord):
    """Manifest-bound analyzer and artifact receipt with derived completion."""

    manifest: DerivationRunManifest
    analyzer: AnalyzerProvenance | None
    artifacts: tuple[DerivationArtifact, ...] = ()
    blockers: tuple[str, ...] = ()

    required_artifact_kinds: ClassVar[frozenset[DerivationArtifactKind]] = frozenset(
        DerivationArtifactKind
    )

    def __post_init__(self) -> None:
        for blocker in self.blockers:
            _require_text(blocker, "DerivationRunReceipt.blockers item")
        _canonicalize_records(
            self,
            "blockers",
            self.blockers,
            key=lambda value: value,
        )
        _canonicalize_records(
            self,
            "artifacts",
            self.artifacts,
            key=lambda artifact: (artifact.kind.value, artifact.relative_path),
        )
        artifact_kinds = tuple(artifact.kind for artifact in self.artifacts)
        artifact_paths = tuple(artifact.relative_path for artifact in self.artifacts)
        _require_unique(artifact_kinds, "DerivationRunReceipt artifact kinds")
        _require_unique(artifact_paths, "DerivationRunReceipt artifact paths")
        if (
            self.analyzer is not None
            and self.analyzer.input_digest != self.manifest.input_digest
        ):
            raise ValueError(
                "AnalyzerProvenance.input_digest does not match the run manifest"
            )
        relation_binding_artifact = next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.kind is DerivationArtifactKind.RELATION_BINDING
            ),
            None,
        )
        if (
            relation_binding_artifact is not None
            and relation_binding_artifact.sha256
            != self.manifest.required_relation_binding_receipt.content_digest
        ):
            raise ValueError(
                "RELATION_BINDING artifact digest does not match the typed receipt"
            )

    @json_report_property()
    def manifest_digest(self) -> str:
        return self.manifest.input_digest

    @json_report_property()
    def missing_artifact_kinds(self) -> tuple[DerivationArtifactKind, ...]:
        available = frozenset(artifact.kind for artifact in self.artifacts)
        return tuple(
            sorted(
                self.required_artifact_kinds - available, key=lambda kind: kind.value
            )
        )

    @json_report_property()
    def unresolved_exclusion_ids(self) -> tuple[str, ...]:
        return tuple(exclusion.exclusion_id for exclusion in self.manifest.exclusions)

    @json_report_property()
    def status(self) -> DerivationRunStatus:
        return (
            DerivationRunStatus.COMPLETE
            if self.analyzer is not None
            and not self.blockers
            and not self.missing_artifact_kinds
            and self.manifest.required_relation_binding_receipt.coverage_verdict.is_positive
            else DerivationRunStatus.INCOMPLETE
        )

    def require_complete(self) -> Self:
        """Return this receipt only when every declared proof receipt exists."""

        if self.status is DerivationRunStatus.COMPLETE:
            return self
        reasons: list[str] = []
        if self.analyzer is None:
            reasons.append("analyzer provenance is absent")
        if self.missing_artifact_kinds:
            reasons.append(
                "missing artifacts: "
                + ", ".join(kind.value for kind in self.missing_artifact_kinds)
            )
        if self.unresolved_exclusion_ids:
            reasons.append(
                "unresolved exclusions: " + ", ".join(self.unresolved_exclusion_ids)
            )
        reasons.extend(self.blockers)
        raise IncompleteDerivationRunError("; ".join(reasons))
