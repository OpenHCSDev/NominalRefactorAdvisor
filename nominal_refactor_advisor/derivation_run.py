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


@dataclass(frozen=True)
class DerivationRunBoundary(SemanticRecord):
    """Fixed repository, scope, policy, and runtime boundary for one run."""

    repository: str
    repository_revision: str
    scope_label: str
    included_roots: tuple[str, ...]
    language_runtime: str
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
    """One exact implementation/key-to-consumer requirement pair."""

    implementation_key: str
    consumer: str

    def __post_init__(self) -> None:
        _require_text(
            self.implementation_key,
            "RequiredRelationRow.implementation_key",
        )
        _require_text(self.consumer, "RequiredRelationRow.consumer")


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
    named_divergence: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.site, "EnumerationDecision.site")
        _require_text(self.resolution, "EnumerationDecision.resolution")
        if self.determining_authority is not None:
            _require_text(
                self.determining_authority,
                "EnumerationDecision.determining_authority",
            )
        if self.named_divergence is not None:
            _require_text(
                self.named_divergence,
                "EnumerationDecision.named_divergence",
            )
        if self.kind.requires_determining_authority:
            if self.determining_authority is None:
                raise ValueError(f"{self.kind.value} requires a determining authority")
        elif self.determining_authority is not None:
            raise ValueError(
                "a genuine authority cannot name another determining authority"
            )
        if self.kind.requires_named_divergence:
            if self.named_divergence is None:
                raise ValueError("named_divergence requires a divergence receipt")
        elif self.named_divergence is not None:
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
    required_relation: tuple[RequiredRelationRow, ...] = ()
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
                "required_relation",
                self.required_relation,
                lambda row: (row.implementation_key, row.consumer),
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
        for collision in self.structural_collisions:
            unknown_meanings = frozenset(collision.meaning_ids) - known_meanings
            unknown_questions = frozenset(collision.question_ids) - known_questions
            if unknown_meanings or unknown_questions:
                raise ValueError(
                    "StructuralCollision references unknown identities: "
                    f"meanings={tuple(sorted(unknown_meanings))!r}, "
                    f"questions={tuple(sorted(unknown_questions))!r}"
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
    def status(self) -> DerivationRunStatus:
        return (
            DerivationRunStatus.COMPLETE
            if self.analyzer is not None
            and not self.blockers
            and not self.missing_artifact_kinds
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
        reasons.extend(self.blockers)
        raise IncompleteDerivationRunError("; ".join(reasons))
