"""Authority-boundary proof gate for semantic refactor scans."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from .detectors import IssueDetector, SemanticMirrorWithoutDescentDetector
from .json_reports import (
    DataclassJsonReport,
    JsonReport,
    SemanticRecord,
    json_report_field,
    json_report_property,
)
from .models import RefactorFinding
from .models import SourceLocation
from .patterns import PatternId
from .semantic_descent import (
    AuthorityClaimResolution,
    AuthorityDiscoveryRequired,
    FindingDescentCertificateAuthority,
    PresentationProjectionKind,
    ResolvedDescentCertificate,
    SemanticAuthorityKind,
    SemanticDescentGraph,
    build_finding_backed_semantic_descent_graph,
)
from .taxonomy import CertificationLevel, ConfidenceLevel


UNRESOLVED_AUTHORITY_CLAIM_DETECTOR_ID = "unresolved_authority_claim"


def ssot_authority_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[RefactorFinding, ...]:
    """Findings that create authority-boundary proof obligations."""

    return tuple(
        finding
        for finding in findings
        if finding.detector_id in IssueDetector.ssot_authority_detector_ids()
    )


@dataclass(frozen=True)
class DescentCertificateFindingAuthority(FindingDescentCertificateAuthority):
    """Add semantic-boundary grouping to exact certificate selection."""

    def group_key_for_finding(
        self,
        finding: RefactorFinding,
    ) -> "SemanticRefactorFindingGroupKey":
        return self.group_key_for_certificate(
            self.resolved_certificate_for_finding(finding)
        )

    @staticmethod
    def group_key_for_certificate(
        resolved: ResolvedDescentCertificate,
    ) -> "SemanticRefactorFindingGroupKey":
        return SemanticRefactorFindingGroupKey(
            authority_label=resolved.authority.name,
            descent_path=resolved.certificate.missing_derivation_path,
        )

    def finding_groups(
        self,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[tuple[RefactorFinding, ...], ...]:
        groups: dict[SemanticRefactorFindingGroupKey, list[RefactorFinding]] = (
            defaultdict(list)
        )
        for finding in findings:
            groups[self.group_key_for_finding(finding)].append(finding)
        return tuple(tuple(group) for group in groups.values())


@dataclass(frozen=True)
class FindingCoverage(DataclassJsonReport):
    """Observed source-target and finding coverage for one evidence group."""

    target_count: int
    finding_count: int = json_report_field(field_name="covered_finding_count")


@dataclass(frozen=True)
class SemanticRefactorFindingGroupKey:
    """Graph-derived identity for one semantic boundary evidence group."""

    authority_label: str
    descent_path: str


@dataclass(frozen=True)
class SemanticRefactorBoundaryEvidence(SemanticRecord):
    """One graph-backed authority-boundary evidence group."""

    group_key: SemanticRefactorFindingGroupKey = json_report_field(included=False)
    label: str
    authority_candidates: tuple[str, ...]
    detector_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    finding_coverage: FindingCoverage = json_report_field(flattened=True)
    certificate_count: int
    matched_fact_count: int
    authority_kinds: tuple[SemanticAuthorityKind, ...]
    projection_kinds: tuple[PresentationProjectionKind, ...]
    authority_claims: tuple[AuthorityClaimResolution, ...]
    evidence_symbols: tuple[str, ...]
    evidence_locations: tuple[SourceLocation, ...] = json_report_field(
        included=False,
        default=(),
    )

    @classmethod
    def from_ssot_finding(
        cls,
        finding: RefactorFinding,
    ) -> "SemanticRefactorBoundaryEvidence":
        certificate_authority = DescentCertificateFindingAuthority(
            build_finding_backed_semantic_descent_graph((finding,))
        )
        return cls.from_ssot_finding_group(
            (finding,),
            certificate_authority=certificate_authority,
        )

    @classmethod
    def from_ssot_finding_group(
        cls,
        findings: tuple[RefactorFinding, ...],
        *,
        certificate_authority: DescentCertificateFindingAuthority,
    ) -> "SemanticRefactorBoundaryEvidence":
        first_finding = findings[0]
        certificate_selection = certificate_authority.resolved_selection_for_findings(
            findings
        )
        authority_candidates = certificate_selection.authority_names
        evidence_symbols = _unique_strings(
            location.symbol for finding in findings for location in finding.evidence
        )
        evidence_locations = tuple(
            dict.fromkeys(
                location for finding in findings for location in finding.evidence
            )
        )
        detector_ids = _unique_strings(finding.detector_id for finding in findings)
        group_key = certificate_authority.group_key_for_certificate(
            certificate_selection.certificates[0]
        )
        label = first_finding.title
        if len(findings) > 1:
            label = f"{label} ({len(findings)} raw signals)"
        if len(authority_candidates) == 1:
            label = f"{authority_candidates[0]} semantic descent boundary"
        return cls(
            group_key=group_key,
            label=label,
            authority_candidates=authority_candidates,
            detector_ids=detector_ids,
            finding_ids=tuple(finding.stable_id for finding in findings),
            finding_coverage=FindingCoverage(
                target_count=max(1, len(evidence_symbols)),
                finding_count=len(findings),
            ),
            certificate_count=len(certificate_selection.certificates),
            matched_fact_count=certificate_selection.matched_fact_count,
            authority_kinds=certificate_selection.authority_kinds,
            projection_kinds=certificate_selection.projection_kinds,
            authority_claims=certificate_selection.authority_claims,
            evidence_symbols=evidence_symbols,
            evidence_locations=evidence_locations,
        )

    @json_report_property()
    def stable_id(self) -> str:
        payload = "|".join(
            (
                self.primary_detector_id,
                self.group_key.authority_label,
                self.group_key.descent_path,
                *self.detector_ids,
                *self.finding_ids,
            )
        )
        return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()

    @json_report_property(field_name="detector_id")
    def primary_detector_id(self) -> str:
        detector_id = SemanticMirrorWithoutDescentDetector.effective_detector_id()
        if detector_id is None:
            raise TypeError("SemanticMirrorWithoutDescentDetector has no detector id")
        return detector_id

    @json_report_property(field_name="title")
    def primary_title(self) -> str:
        return SemanticMirrorWithoutDescentDetector.finding_spec.title

    @json_report_property()
    def summary(self) -> str:
        return (
            f"`{self.group_key.authority_label}` has "
            f"{self.finding_coverage.finding_count} "
            "raw mirror signal(s) from "
            f"{', '.join(self.detector_ids)}; missing derivation path: "
            f"{self.group_key.descent_path}."
        )

    @json_report_property(field_name="relation_context")
    def relation_context(self) -> str:
        return self.group_key.descent_path

    @json_report_property(field_name="authority_candidate")
    def authority_candidate(self) -> str:
        return self.group_key.authority_label

    @json_report_property(field_name="authority_discovery_required")
    def discovery_required(self) -> bool:
        return any(not claim.is_actionable for claim in self.authority_claims)


class AuthorityDiscoveryRequiredFindingProjection:
    """Project unresolved gate authority claims into hard advisor findings."""

    synthetic_file_path: str = "<semantic-refactor-gate>"

    @classmethod
    def findings_for_boundary_evidence(
        cls,
        boundary_evidence: tuple[SemanticRefactorBoundaryEvidence, ...],
    ) -> tuple[RefactorFinding, ...]:
        return tuple(
            cls.finding_for_resolution(item, resolution)
            for item in boundary_evidence
            for resolution in item.authority_claims
            if not resolution.is_actionable
        )

    @classmethod
    def finding_for_resolution(
        cls,
        item: SemanticRefactorBoundaryEvidence,
        resolution: AuthorityClaimResolution,
    ) -> RefactorFinding:
        discovery = cls.discovery_for_resolution(resolution)
        claim = resolution.claim
        return RefactorFinding(
            pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
            title="Authority discovery required",
            why=(
                "Unknown authority is acceptable, but fabricated authority is not. "
                "A refactor gate evidence group that names an authority must carry a "
                "unique proof path or explicitly ask for discovery."
            ),
            capability_gap=(
                "resolved AuthorityClaim with source/graph proof edge, or an "
                "explicit DeclareAuthority operation"
            ),
            relation_context=(
                "semantic refactor gate encountered an authority claim without a "
                "unique source-backed proof path"
            ),
            confidence=ConfidenceLevel.HIGH,
            certification=CertificationLevel.CERTIFIED,
            detector_id=UNRESOLVED_AUTHORITY_CLAIM_DETECTOR_ID,
            summary=(
                f"You claimed `{claim.claimed_symbol}`, but NRA found "
                f"{discovery.candidate_count} candidate authority proof path(s) after "
                f"searching {discovery.searched_symbols}: {discovery.reason}."
            ),
            evidence=cls.evidence_for_resolution(item, resolution),
        )

    @staticmethod
    def discovery_for_resolution(
        resolution: AuthorityClaimResolution,
    ) -> AuthorityDiscoveryRequired:
        if resolution.discovery_required is None:
            raise ValueError(
                "non-actionable authority claim resolution must carry "
                "AuthorityDiscoveryRequired evidence"
            )
        return resolution.discovery_required

    @classmethod
    def evidence_for_resolution(
        cls,
        item: SemanticRefactorBoundaryEvidence,
        resolution: AuthorityClaimResolution,
    ) -> tuple[SourceLocation, ...]:
        proof_evidence = tuple(
            SourceLocation(
                edge.file_path,
                edge.line,
                edge.symbol,
            )
            for edge in resolution.proof_edges
            if edge.file_path
        )
        if proof_evidence:
            return proof_evidence
        if item.evidence_locations:
            return item.evidence_locations[:6]
        return (
            SourceLocation(
                cls.synthetic_file_path,
                0,
                resolution.claim.claimed_symbol,
            ),
        )

@dataclass(frozen=True)
class SemanticRefactorGateReport(SemanticRecord):
    """Authority-boundary proof report for semantic refactor scans."""

    policy: ClassVar[str] = "authority_boundary_proof"
    raw_findings_default: ClassVar[str] = "suppressed_when_active"
    boundary_evidence: tuple[SemanticRefactorBoundaryEvidence, ...]
    authority_discovery_findings: tuple[RefactorFinding, ...]

    @json_report_property()
    def active(self) -> bool:
        return bool(self.boundary_evidence)

    @json_report_property()
    def ssot_authority_finding_count(self) -> int:
        return sum(
            item.finding_coverage.finding_count for item in self.boundary_evidence
        )

    @classmethod
    def from_findings(
        cls,
        findings: tuple[RefactorFinding, ...] = (),
    ) -> "SemanticRefactorGateReport":
        ssot_findings = ssot_authority_findings(findings)
        finding_descent_graph = build_finding_backed_semantic_descent_graph(
            ssot_findings,
        )
        boundary_evidence = cls._boundary_evidence(
            ssot_findings,
            finding_descent_graph,
        )
        return cls(
            boundary_evidence=boundary_evidence,
            authority_discovery_findings=(
                AuthorityDiscoveryRequiredFindingProjection.findings_for_boundary_evidence(
                    boundary_evidence
                )
            ),
        )

    @classmethod
    def inactive(cls) -> "SemanticRefactorGateReport":
        return cls(
            boundary_evidence=(),
            authority_discovery_findings=(),
        )

    @json_report_property(field_name="policy")
    def report_policy(self) -> str:
        return self.policy

    @json_report_property(field_name="raw_findings_default")
    def report_raw_findings_default(self) -> str:
        return self.raw_findings_default

    @staticmethod
    def _boundary_evidence(
        ssot_findings: tuple[RefactorFinding, ...],
        finding_descent_graph: SemanticDescentGraph,
    ) -> tuple[SemanticRefactorBoundaryEvidence, ...]:
        certificate_authority = DescentCertificateFindingAuthority(
            finding_descent_graph
        )
        items = tuple(
            SemanticRefactorBoundaryEvidence.from_ssot_finding_group(
                group,
                certificate_authority=certificate_authority,
            )
            for group in certificate_authority.finding_groups(ssot_findings)
        )
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.group_key.authority_label,
                    item.group_key.descent_path,
                    item.finding_ids,
                ),
            )
        )

    @property
    def finding_reports(self) -> tuple[JsonReport, ...]:
        """Return the typed findings presented when the gate is active."""

        return (*self.boundary_evidence, *self.authority_discovery_findings)

    @property
    def count_line(self) -> str:
        return (
            "   - Gate counts: "
            f"{len(self.boundary_evidence)} graph-backed boundary group(s); "
            f"{len(self.authority_discovery_findings)} requiring authority discovery."
        )

    def markdown(self) -> str:
        return "\n".join(self.markdown_lines())

    def markdown_lines(self) -> tuple[str, ...]:
        if not self.active:
            return ()
        return (
            *self._status_lines(),
            *self._boundary_evidence_lines(),
            *self._authority_discovery_lines(),
            *self._footer_lines(),
        )

    def _status_lines(self) -> tuple[str, ...]:
        return (
            "Semantic refactor gate:",
            "   - Status: ACTIVE. Raw findings are supporting evidence only.",
            (
                "   - Application gate: each touched projection requires a proved "
                "descent from its load-bearing authority."
            ),
            (
                "   - A boundary evidence group is not an application order or a "
                "trajectory recommendation."
            ),
            self.count_line,
            *self._evidence_context_lines(),
        )

    def _evidence_context_lines(self) -> tuple[str, ...]:
        lines = []
        if self.ssot_authority_finding_count:
            lines.append(
                f"   - SSOT-critical signals: {self.ssot_authority_finding_count}."
            )
        if self.ssot_authority_finding_count:
            lines.append(
                "   - Boundary groups come only from the semantic-descent graph; "
                "structural-overlap evidence does not order these groups."
            )
        return tuple(lines)

    def _boundary_evidence_lines(self) -> tuple[str, ...]:
        if not self.boundary_evidence:
            return ()
        lines = ["   - Boundary evidence groups:"]
        for item in self.boundary_evidence[:5]:
            lines.append(f"     - {item.label}")
            lines.append(
                f"        authority candidate: {item.group_key.authority_label}"
            )
            lines.append(f"        missing descent: {item.group_key.descent_path}")
            if item.certificate_count:
                lines.append(
                    "        descent certificates: "
                    f"{item.certificate_count}, matched facts: "
                    f"{item.matched_fact_count}"
                )
        return tuple(lines)

    def _authority_discovery_lines(self) -> tuple[str, ...]:
        if not self.authority_discovery_findings:
            return ()
        lines = [
            "   - Authority discovery required:",
        ]
        for index, finding in enumerate(
            self.authority_discovery_findings[:5],
            start=1,
        ):
            lines.append(f"     {index}. {finding.summary}")
        return tuple(lines)

    def _footer_lines(self) -> tuple[str, ...]:
        return (
            (
                "   - Raw findings: suppressed by default under this gate; use "
                "--raw-findings to inspect the evidence supporting these groups."
            ),
        )


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
