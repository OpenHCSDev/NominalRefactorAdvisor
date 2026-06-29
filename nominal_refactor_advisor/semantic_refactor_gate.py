"""Authority-boundary-first gate for semantic refactor scans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .codemod import (
    CodemodActionability,
    CodemodAutomationLevel,
    CodemodCandidate,
    JsonObject,
)
from .detectors import IssueDetector
from .impact_ranking import RefactorImpactRankingReport
from .models import RefactorFinding, SemanticRecord, SourceLocation

SEMANTIC_REFACTOR_GATE_DISABLED_MESSAGE = (
    "--no-impact-ranking disables the semantic refactor gate; pass "
    "--raw-findings to explicitly audit raw findings without authority "
    "boundary guidance."
)


class SemanticRefactorGateModeError(ValueError):
    """Raised when a CLI mode would bypass authority-boundary guidance."""


SSOT_AUTHORITY_BOUNDARY_TIER = "ssot_authority_boundary"
CLEANUP_FOLLOWUP_TIER = "cleanup_followup"
ORDINARY_SEMANTIC_TIER = "ordinary_semantic"
DEFERRED_CLEANUP_DETECTOR_IDS = frozenset(("trivial_forwarding_wrapper",))


def priority_tier_for_detector_ids(detector_ids: tuple[str, ...]) -> str:
    """Return the architectural priority tier for a detector family."""

    detector_id_set = frozenset(detector_ids)
    if detector_id_set & IssueDetector.ssot_authority_detector_ids():
        return SSOT_AUTHORITY_BOUNDARY_TIER
    if detector_id_set and detector_id_set <= DEFERRED_CLEANUP_DETECTOR_IDS:
        return CLEANUP_FOLLOWUP_TIER
    return ORDINARY_SEMANTIC_TIER


def detector_ids_have_semantic_mirror_role(detector_ids: tuple[str, ...]) -> bool:
    return bool(frozenset(detector_ids) & IssueDetector.semantic_mirror_detector_ids())


def priority_tier_has_ssot_authority_role(priority_tier: str) -> bool:
    return priority_tier == SSOT_AUTHORITY_BOUNDARY_TIER


def ssot_authority_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[RefactorFinding, ...]:
    """Findings that should drive authority-boundary work before cleanup."""

    return tuple(
        finding
        for finding in findings
        if finding.detector_id in IssueDetector.ssot_authority_detector_ids()
    )


def cleanup_followup_findings(
    findings: tuple[RefactorFinding, ...],
) -> tuple[RefactorFinding, ...]:
    """Findings that should not displace SSOT/authority-boundary work."""

    return tuple(
        finding
        for finding in findings
        if finding.detector_id in DEFERRED_CLEANUP_DETECTOR_IDS
    )


@dataclass(frozen=True)
class SemanticRefactorGateMode(SemanticRecord):
    """Advisor run mode policy for authority-boundary-first scans."""

    include_impact_ranking: bool
    semantic_refactor_gate: bool
    raw_findings: bool

    @classmethod
    def from_flags(
        cls,
        *,
        include_impact_ranking: bool,
        semantic_refactor_gate: bool,
        raw_findings: bool,
    ) -> "SemanticRefactorGateMode":
        return cls(
            include_impact_ranking=include_impact_ranking,
            semantic_refactor_gate=semantic_refactor_gate,
            raw_findings=raw_findings,
        )

    def require_authority_boundary_mode(self) -> None:
        if self.include_impact_ranking:
            return
        if self.semantic_refactor_gate:
            return
        if self.raw_findings:
            return
        raise SemanticRefactorGateModeError(SEMANTIC_REFACTOR_GATE_DISABLED_MESSAGE)


@dataclass(frozen=True)
class SemanticRefactorGateTrajectory(SemanticRecord):
    """First dynamic trajectory exposed by the semantic refactor gate."""

    sequence: tuple[str, ...]
    predicted_removed_finding_count: int
    residual_finding_count: int
    trajectory_score: int

    @classmethod
    def from_impact_ranking(
        cls,
        impact_ranking: RefactorImpactRankingReport | None,
    ) -> "SemanticRefactorGateTrajectory | None":
        if impact_ranking is None or not impact_ranking.trajectories:
            return None
        trajectory = impact_ranking.trajectories[0]
        return cls(
            sequence=tuple(f"{key.kind}:{key.label}" for key in trajectory.keys),
            predicted_removed_finding_count=(
                trajectory.predicted_removed_finding_count
            ),
            residual_finding_count=trajectory.residual_finding_count,
            trajectory_score=trajectory.trajectory_score,
        )

    def markdown_lines(self) -> tuple[str, ...]:
        return (
            (
                "   - First trajectory: "
                f"removes {self.predicted_removed_finding_count} finding(s), "
                f"residual {self.residual_finding_count}, "
                f"score {self.trajectory_score}"
            ),
            f"     sequence: {' -> '.join(self.sequence)}",
        )


@dataclass(frozen=True)
class SemanticRefactorAuthorityTarget(SemanticRecord):
    """One load-bearing authority target exposed by the semantic refactor gate."""

    opportunity_kind: str
    opportunity_label: str
    priority_tier: str
    detector_ids: tuple[str, ...]
    actionability: str
    target_count: int
    predicted_removed_finding_count: int
    strategy_id: str
    agent_action: str

    @classmethod
    def from_candidate(
        cls,
        candidate: CodemodCandidate,
    ) -> "SemanticRefactorAuthorityTarget":
        applicability = candidate.applicability
        return cls(
            opportunity_kind=candidate.opportunity_key.kind,
            opportunity_label=candidate.opportunity_key.label,
            priority_tier=priority_tier_for_detector_ids(
                candidate.opportunity.detector_ids
            ),
            detector_ids=candidate.opportunity.detector_ids,
            actionability=applicability.actionability.value,
            target_count=candidate.target_count,
            predicted_removed_finding_count=candidate.predicted_removed_finding_count,
            strategy_id=applicability.strategy.strategy_id,
            agent_action=applicability.agent_action,
        )

    def markdown_lines(self, index: int) -> tuple[str, ...]:
        return (
            (
                f"     {index}. {self.opportunity_kind} "
                f"`{self.opportunity_label}` -> "
                f"{self.predicted_removed_finding_count} finding(s), "
                f"{self.target_count} target(s), "
                f"{self.actionability}, priority {self.priority_tier}"
            ),
            f"        detectors: {', '.join(self.detector_ids)}",
            f"        agent action: {self.agent_action}",
        )


@dataclass(frozen=True)
class SemanticRefactorGateWorkItem(SemanticRecord):
    """One authority-boundary task that replaces raw finding iteration."""

    source: str
    priority_tier: str
    label: str
    authority_candidate: str
    authority_candidates: tuple[str, ...]
    missing_derivation_path: str
    detector_ids: tuple[str, ...]
    actionability: str
    finding_ids: tuple[str, ...]
    target_count: int
    predicted_removed_finding_count: int
    agent_action: str
    evidence_symbols: tuple[str, ...]

    @classmethod
    def from_authority_target(
        cls,
        target: SemanticRefactorAuthorityTarget,
    ) -> "SemanticRefactorGateWorkItem":
        return cls(
            source="impact_candidate",
            priority_tier=target.priority_tier,
            label=target.opportunity_label,
            authority_candidate=target.opportunity_label,
            authority_candidates=(target.opportunity_label,),
            missing_derivation_path=cls.missing_derivation_path_for_detectors(
                target.detector_ids
            ),
            detector_ids=target.detector_ids,
            actionability=target.actionability,
            finding_ids=(),
            target_count=target.target_count,
            predicted_removed_finding_count=target.predicted_removed_finding_count,
            agent_action=target.agent_action,
            evidence_symbols=(),
        )

    @classmethod
    def from_ssot_finding(
        cls,
        finding: RefactorFinding,
    ) -> "SemanticRefactorGateWorkItem":
        return cls.from_ssot_finding_group((finding,))

    @classmethod
    def from_ssot_finding_group(
        cls,
        findings: tuple[RefactorFinding, ...],
    ) -> "SemanticRefactorGateWorkItem":
        first_finding = findings[0]
        authority_candidates = _unique_strings(
            _authority_candidate_for_finding(finding) for finding in findings
        )
        evidence_symbols = _unique_strings(
            location.symbol for finding in findings for location in finding.evidence
        )
        detector_ids = _unique_strings(finding.detector_id for finding in findings)
        label = first_finding.title
        if len(findings) > 1:
            label = f"{label} ({len(findings)} authority candidates)"
        return cls(
            source="ssot_finding",
            priority_tier=priority_tier_for_detector_ids(detector_ids),
            label=label,
            authority_candidate=authority_candidates[0],
            authority_candidates=authority_candidates,
            missing_derivation_path=_missing_derivation_path_for_finding(first_finding),
            detector_ids=detector_ids,
            actionability="semantic_agent_refactor",
            finding_ids=tuple(finding.stable_id for finding in findings),
            target_count=max(1, len(evidence_symbols)),
            predicted_removed_finding_count=len(findings),
            agent_action=(
                "Design the nominal authority boundary named by this finding, "
                "then derive the mirrored surface from that authority before "
                "addressing lower-priority cleanup."
            ),
            evidence_symbols=evidence_symbols,
        )

    @property
    def priority_rank(self) -> tuple[int, int, str]:
        semantic_mirror_rank = (
            0 if detector_ids_have_semantic_mirror_role(self.detector_ids) else 1
        )
        ssot_rank = (
            0 if priority_tier_has_ssot_authority_role(self.priority_tier) else 1
        )
        return (
            semantic_mirror_rank,
            ssot_rank,
            self.label,
        )

    @staticmethod
    def missing_derivation_path_for_detectors(detector_ids: tuple[str, ...]) -> str:
        if detector_ids_have_semantic_mirror_role(detector_ids):
            return (
                "projection must be derived from the nominal authority registry, "
                "class family, enum, or schema owner"
            )
        return (
            "raw surfaces must collapse behind one nominal authority before "
            "lower-level finding cleanup"
        )


@dataclass(frozen=True)
class SemanticRefactorGateReport(SemanticRecord):
    """Authority-boundary-first report shape for semantic refactor scans."""

    active: bool
    policy: str
    raw_findings_default: str
    semantic_candidate_count: int
    semantic_agent_refactor_count: int
    semantic_uncertainty_review_count: int
    ssot_authority_finding_count: int
    cleanup_followup_finding_count: int
    first_trajectory: SemanticRefactorGateTrajectory | None
    authority_targets: tuple[SemanticRefactorAuthorityTarget, ...]
    work_queue: tuple[SemanticRefactorGateWorkItem, ...]

    @classmethod
    def from_scan(
        cls,
        candidates: tuple[CodemodCandidate, ...],
        *,
        impact_ranking: RefactorImpactRankingReport | None,
        findings: tuple[RefactorFinding, ...] = (),
    ) -> "SemanticRefactorGateReport":
        semantic_candidates = cls._semantic_candidates(candidates)
        ssot_findings = ssot_authority_findings(findings)
        cleanup_findings = cleanup_followup_findings(findings)
        authority_targets = tuple(
            SemanticRefactorAuthorityTarget.from_candidate(candidate)
            for candidate in cls._priority_sorted_candidates(semantic_candidates)[:10]
        )
        return cls(
            active=bool(semantic_candidates or ssot_findings),
            policy="authority_boundary_first",
            raw_findings_default="suppressed_when_active",
            semantic_candidate_count=len(semantic_candidates),
            semantic_agent_refactor_count=cls._actionability_count(
                semantic_candidates,
                CodemodActionability.SEMANTIC_AGENT_REFACTOR,
            ),
            semantic_uncertainty_review_count=cls._actionability_count(
                semantic_candidates,
                CodemodActionability.SEMANTIC_UNCERTAINTY_REVIEW,
            ),
            ssot_authority_finding_count=len(ssot_findings),
            cleanup_followup_finding_count=len(cleanup_findings),
            first_trajectory=SemanticRefactorGateTrajectory.from_impact_ranking(
                impact_ranking
            ),
            authority_targets=authority_targets,
            work_queue=cls._work_queue(authority_targets, ssot_findings),
        )

    @classmethod
    def from_optional_scan(
        cls,
        candidates: tuple[CodemodCandidate, ...] | None,
        *,
        impact_ranking: RefactorImpactRankingReport | None,
        findings: tuple[RefactorFinding, ...] = (),
    ) -> "SemanticRefactorGateReport":
        if candidates is None:
            return cls.from_scan((), impact_ranking=impact_ranking, findings=findings)
        return cls.from_scan(
            candidates,
            impact_ranking=impact_ranking,
            findings=findings,
        )

    @classmethod
    def inactive(cls) -> "SemanticRefactorGateReport":
        return cls(
            active=False,
            policy="authority_boundary_first",
            raw_findings_default="suppressed_when_active",
            semantic_candidate_count=0,
            semantic_agent_refactor_count=0,
            semantic_uncertainty_review_count=0,
            ssot_authority_finding_count=0,
            cleanup_followup_finding_count=0,
            first_trajectory=None,
            authority_targets=(),
            work_queue=(),
        )

    @staticmethod
    def _semantic_candidates(
        candidates: tuple[CodemodCandidate, ...],
    ) -> tuple[CodemodCandidate, ...]:
        return tuple(
            candidate
            for candidate in candidates
            if (
                candidate.applicability.strategy.automation_level
                is CodemodAutomationLevel.SEMANTIC_AGENT_REQUIRED
            )
        )

    @staticmethod
    def _priority_sorted_candidates(
        semantic_candidates: tuple[CodemodCandidate, ...],
    ) -> tuple[CodemodCandidate, ...]:
        return tuple(
            sorted(
                semantic_candidates,
                key=lambda candidate: (
                    priority_tier_for_detector_ids(candidate.opportunity.detector_ids)
                    != SSOT_AUTHORITY_BOUNDARY_TIER
                ),
            )
        )

    @staticmethod
    def _actionability_count(
        semantic_candidates: tuple[CodemodCandidate, ...],
        actionability: CodemodActionability,
    ) -> int:
        return sum(
            (candidate.applicability.actionability is actionability)
            for candidate in semantic_candidates
        )

    @staticmethod
    def _work_queue(
        authority_targets: tuple[SemanticRefactorAuthorityTarget, ...],
        ssot_findings: tuple[RefactorFinding, ...],
    ) -> tuple[SemanticRefactorGateWorkItem, ...]:
        items = (
            *(
                SemanticRefactorGateWorkItem.from_authority_target(target)
                for target in authority_targets
            ),
            *(
                SemanticRefactorGateWorkItem.from_ssot_finding_group(group)
                for group in SemanticRefactorGateReport._ssot_finding_groups(
                    ssot_findings
                )
            ),
        )
        return tuple(sorted(items, key=lambda item: item.priority_rank))

    @staticmethod
    def _ssot_finding_groups(
        ssot_findings: tuple[RefactorFinding, ...],
    ) -> tuple[tuple[RefactorFinding, ...], ...]:
        groups: dict[tuple[str, str], list[RefactorFinding]] = defaultdict(list)
        for finding in ssot_findings:
            groups[(finding.detector_id, finding.title)].append(finding)
        return tuple(tuple(group) for group in groups.values())

    def finding_payload(self) -> list[JsonObject]:
        """Return the JSON `findings` surface when the gate is active."""

        return [JsonObject(item.to_dict()) for item in self.work_queue]

    @property
    def count_line(self) -> str:
        return (
            "   - Gate counts: "
            f"{self.semantic_candidate_count} semantic candidate(s); "
            f"{self.semantic_agent_refactor_count} ready for agent refactor; "
            f"{self.semantic_uncertainty_review_count} needing boundary review."
        )

    def markdown(self) -> str:
        return "\n".join(self.markdown_lines())

    def markdown_lines(self) -> tuple[str, ...]:
        if not self.active:
            return ()
        return (
            *self._status_lines(),
            *self._trajectory_lines(),
            *self._target_lines(),
            *self._work_queue_lines(),
            *self._footer_lines(),
        )

    def _status_lines(self) -> tuple[str, ...]:
        return (
            "Semantic refactor gate:",
            "   - Status: ACTIVE. Raw findings are supporting evidence, not a work queue.",
            (
                "   - Required mode: choose one load-bearing authority boundary "
                "from the trajectory/candidate groups before editing."
            ),
            (
                "   - Forbidden mode: do not patch individual findings "
                "independently or rerun until the first finding disappears."
            ),
            (
                "   - Priority: SSOT/authority-boundary findings outrank "
                "cleanup-only wrapper findings."
            ),
            self.count_line,
            *self._priority_lines(),
        )

    def _priority_lines(self) -> tuple[str, ...]:
        lines = []
        if self.ssot_authority_finding_count:
            lines.append(
                "   - SSOT-critical signals: "
                f"{self.ssot_authority_finding_count}; collapse the single "
                "source of truth before cosmetic cleanup."
            )
        if self.cleanup_followup_finding_count:
            lines.append(
                "   - Cleanup-only signals: "
                f"{self.cleanup_followup_finding_count}; defer unless no "
                "SSOT-critical signal is present."
            )
        if self.ssot_authority_finding_count and not self.authority_targets:
            lines.append(
                "   - No impact-ranked target was generated; inspect the "
                "SSOT-critical raw finding evidence and design that authority "
                "boundary first."
            )
        return tuple(lines)

    def _trajectory_lines(self) -> tuple[str, ...]:
        if self.first_trajectory is None:
            return ()
        return self.first_trajectory.markdown_lines()

    def _target_lines(self) -> tuple[str, ...]:
        if not self.authority_targets:
            return ("   - Authority targets: none impact-ranked.",)
        lines: list[str] = ["   - Authority targets:"]
        for index, target in enumerate(self.authority_targets[:5], start=1):
            lines.extend(target.markdown_lines(index))
        return tuple(lines)

    def _work_queue_lines(self) -> tuple[str, ...]:
        if not self.work_queue:
            return ()
        lines = ["   - Primary work queue:"]
        for index, item in enumerate(self.work_queue[:5], start=1):
            lines.append(
                f"     {index}. {item.label} -> {item.actionability}, "
                f"priority {item.priority_tier}"
            )
            lines.append(f"        authority candidate: {item.authority_candidate}")
            lines.append(f"        missing descent: {item.missing_derivation_path}")
        return tuple(lines)

    def _footer_lines(self) -> tuple[str, ...]:
        if self.ssot_authority_finding_count and not self.authority_targets:
            raw_findings_instruction = (
                "--raw-findings to inspect SSOT-critical evidence before "
                "designing the authority boundary."
            )
        else:
            raw_findings_instruction = (
                "--raw-findings only after selecting the authority boundary."
            )
        return (
            (
                "   - Raw findings: suppressed by default under this gate; use "
                f"{raw_findings_instruction}"
            ),
        )


def _authority_candidate_for_finding(finding: RefactorFinding) -> str:
    if detector_ids_have_semantic_mirror_role((finding.detector_id,)):
        authority_evidence = _semantic_mirror_authority_evidence(finding)
        if authority_evidence is not None:
            return authority_evidence.symbol
        return finding.title
    if finding.evidence:
        return finding.evidence[0].symbol
    return finding.title


def _semantic_mirror_authority_evidence(
    finding: RefactorFinding,
) -> SourceLocation | None:
    detector_type = IssueDetector.registered_detector_type_for_id(finding.detector_id)
    if detector_type is None:
        return _first_evidence(finding)
    evidence_index = detector_type.semantic_mirror_authority_evidence_index
    if evidence_index is None:
        return _first_evidence(finding)
    if evidence_index >= len(finding.evidence):
        return _first_evidence(finding)
    return finding.evidence[evidence_index]


def _first_evidence(finding: RefactorFinding) -> SourceLocation | None:
    if not finding.evidence:
        return None
    return finding.evidence[0]


def _missing_derivation_path_for_finding(finding: RefactorFinding) -> str:
    if detector_ids_have_semantic_mirror_role((finding.detector_id,)):
        return finding.relation_context
    return SemanticRefactorGateWorkItem.missing_derivation_path_for_detectors(
        (finding.detector_id,)
    )


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
