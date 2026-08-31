"""Reusable closed-loop workflows for executable codemod DSL plans."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from .analysis import (
    AnalysisPathScope,
    EvidenceLocalFindingReuseAuthority,
    EvidenceLocalPartialDetectorSelection,
    SemanticDescentGraphAnalysisSource,
    SortedFindingsAuthority,
    analyze_detector_types,
    analyze_modules,
    default_detector_types_for_analysis,
)
from .ast_tools import ParsedModule, parse_python_module_roots
from .codemod import (
    ArchitectureGuardSuite,
    CodemodPlanDocument,
    CodemodPlanDocumentSimulation,
    CodemodJsonReport,
    CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    FindingRecipeClassPlan,
    FindingRecipeClassPlanReport,
    FindingRecipeSynthesisRecord,
    JsonObject,
    RefactorConcept,
    module_name_from_source_path,
)
from .detectors import DetectorConfig, IssueDetector, SemanticDescentGraphIssueDetector
from .models import RefactorFinding
from .source_index import SourceIndex


class CodemodWorkflowStopReason(StrEnum):
    """Terminal state for staged codemod workflows."""

    ACHIEVED = "achieved"
    NO_EXECUTABLE_RECIPES = "no_executable_recipes"
    EMPTY_REWRITE_BATCH = "empty_rewrite_batch"
    ARCHITECTURE_GUARD_FAILED = "architecture_guard_failed"
    NO_PROGRESS = "no_progress"
    MAX_STAGES = "max_stages"

    @property
    def completed(self) -> bool:
        return self is type(self).ACHIEVED


class CodemodProjectedScanMode(StrEnum):
    """Completeness contract for a projected post-codemod scan."""

    EXACT = "exact"
    EVIDENCE_LOCAL_PARTIAL = "evidence_local_partial"
    TARGET_DETECTOR_PARTIAL = "target_detector_partial"


class CodemodFindingClassStatus(StrEnum):
    """Projected status for one semantic class of advisor findings."""

    ELIMINATED = "eliminated"
    MOVED = "moved"
    PARTIALLY_ELIMINATED = "partially_eliminated"
    PERSISTED = "persisted"
    INTRODUCED = "introduced"
    UNCHANGED = "unchanged"

    @classmethod
    def counts(
        cls,
        changes: tuple["CodemodFindingClassChange", ...],
    ) -> JsonObject:
        return {
            status.value: sum(1 for change in changes if change.status is status)
            for status in cls
            if any(change.status is status for change in changes)
        }


@dataclass(frozen=True)
class CodemodFindingIdTransition:
    """Before/after id transition shared by finding and finding-class deltas."""

    before_ids: tuple[str, ...]
    after_ids: tuple[str, ...]

    @property
    def removed_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_ids)
        return tuple(
            finding_id for finding_id in self.before_ids if finding_id not in after_ids
        )

    @property
    def added_ids(self) -> tuple[str, ...]:
        before_ids = frozenset(self.before_ids)
        return tuple(
            finding_id for finding_id in self.after_ids if finding_id not in before_ids
        )

    @property
    def surviving_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_ids)
        return tuple(
            finding_id for finding_id in self.before_ids if finding_id in after_ids
        )

    @property
    def before_count(self) -> int:
        return len(self.before_ids)

    @property
    def after_count(self) -> int:
        return len(self.after_ids)

    @property
    def removed_count(self) -> int:
        return len(self.removed_ids)

    @property
    def added_count(self) -> int:
        return len(self.added_ids)

    def to_dict(self) -> JsonObject:
        return {
            "before_finding_ids": self.before_ids,
            "after_finding_ids": self.after_ids,
            "removed_finding_ids": self.removed_ids,
            "added_finding_ids": self.added_ids,
            "before_finding_count": self.before_count,
            "after_finding_count": self.after_count,
            "removed_finding_count": self.removed_count,
            "added_finding_count": self.added_count,
        }


@dataclass(frozen=True)
class CodemodFindingDelta:
    """Before/after finding transition for one codemod batch."""

    finding_ids: CodemodFindingIdTransition

    @classmethod
    def from_findings(
        cls,
        before_findings: tuple[RefactorFinding, ...],
        after_findings: tuple[RefactorFinding, ...],
    ) -> "CodemodFindingDelta":
        return cls(
            finding_ids=CodemodFindingIdTransition(
                before_ids=tuple(finding.stable_id for finding in before_findings),
                after_ids=tuple(finding.stable_id for finding in after_findings),
            ),
        )

    @property
    def before_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.before_ids

    @property
    def after_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.after_ids

    @property
    def removed_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.removed_ids

    @property
    def added_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.added_ids

    @property
    def before_finding_count(self) -> int:
        return self.finding_ids.before_count

    @property
    def after_finding_count(self) -> int:
        return self.finding_ids.after_count

    def confirmed_expected_removed_finding_ids(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        removed_ids = frozenset(self.removed_finding_ids)
        return tuple(
            finding_id
            for finding_id in expected_removed_finding_ids
            if finding_id in removed_ids
        )

    def surviving_expected_removed_finding_ids(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        after_ids = frozenset(self.after_finding_ids)
        return tuple(
            finding_id
            for finding_id in expected_removed_finding_ids
            if finding_id in after_ids
        )

    @property
    def removed_finding_count(self) -> int:
        return len(self.removed_finding_ids)

    @property
    def added_finding_count(self) -> int:
        return len(self.added_finding_ids)

    def confirmed_expected_removed_finding_count(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> int:
        return len(
            self.confirmed_expected_removed_finding_ids(expected_removed_finding_ids)
        )

    def surviving_expected_removed_finding_count(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> int:
        return len(
            self.surviving_expected_removed_finding_ids(expected_removed_finding_ids)
        )

    def fulfilled_expected_removals(
        self,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> bool:
        return (
            self.surviving_expected_removed_finding_count(expected_removed_finding_ids)
            == 0
        )

    def to_dict(
        self,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> JsonObject:
        return JsonObject(
            **self.finding_ids.to_dict(),
            expected_removed_finding_ids=expected_removed_finding_ids,
            confirmed_expected_removed_finding_ids=(
                self.confirmed_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            surviving_expected_removed_finding_ids=(
                self.surviving_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            confirmed_expected_removed_finding_count=(
                self.confirmed_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            surviving_expected_removed_finding_count=(
                self.surviving_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            fulfilled_expected_removals=self.fulfilled_expected_removals(
                expected_removed_finding_ids
            ),
        )


@dataclass(frozen=True)
class CodemodFindingClassSignature:
    """Detector-independent semantic identity for a class of equivalent findings."""

    detector_id: str
    pattern_id: int
    title: str
    capability_gap: str
    relation_context: str

    @classmethod
    def from_finding(cls, finding: RefactorFinding) -> "CodemodFindingClassSignature":
        return cls(
            detector_id=finding.detector_id,
            pattern_id=finding.pattern_id.value,
            title=finding.title,
            capability_gap=finding.capability_gap,
            relation_context=finding.relation_context,
        )

    @property
    def class_key(self) -> str:
        return "|".join(
            (
                self.detector_id,
                str(self.pattern_id),
                self.title,
                self.capability_gap,
                self.relation_context,
            )
        )

    def to_dict(self) -> JsonObject:
        return {
            "class_key": self.class_key,
            "detector_id": self.detector_id,
            "pattern_id": self.pattern_id,
            "title": self.title,
            "capability_gap": self.capability_gap,
            "relation_context": self.relation_context,
        }

    @classmethod
    def group_findings(
        cls,
        findings: tuple[RefactorFinding, ...],
    ) -> dict["CodemodFindingClassSignature", tuple[RefactorFinding, ...]]:
        signatures = tuple(
            dict.fromkeys(cls.from_finding(finding) for finding in findings)
        )
        return {
            signature: tuple(
                finding
                for finding in findings
                if cls.from_finding(finding) == signature
            )
            for signature in signatures
        }


@dataclass(frozen=True)
class CodemodFindingClassChange(CodemodFindingDelta):
    """Before/after membership for one semantic finding class."""

    signature: CodemodFindingClassSignature
    expected_removed_finding_ids: tuple[str, ...] = ()

    @property
    def status(self) -> CodemodFindingClassStatus:
        if not self.before_finding_ids and self.after_finding_ids:
            return CodemodFindingClassStatus.INTRODUCED
        if not self.after_finding_ids:
            return CodemodFindingClassStatus.ELIMINATED
        if self.expected_removed_finding_ids and self.added_finding_ids:
            return CodemodFindingClassStatus.MOVED
        if self.removed_finding_ids:
            return CodemodFindingClassStatus.PARTIALLY_ELIMINATED
        if self.expected_removed_finding_ids:
            return CodemodFindingClassStatus.PERSISTED
        return CodemodFindingClassStatus.UNCHANGED

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    def to_dict(self) -> JsonObject:
        return JsonObject(
            **self.finding_ids.to_dict(),
            signature=self.signature.to_dict(),
            status=self.status.value,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
            expected_removed_finding_count=self.expected_removed_finding_count,
        )


@dataclass(frozen=True)
class CodemodFindingClassDelta:
    """Class-level before/after projection for detecting moved smell classes."""

    changes: tuple[CodemodFindingClassChange, ...]

    @classmethod
    def from_findings(
        cls,
        before_findings: tuple[RefactorFinding, ...],
        after_findings: tuple[RefactorFinding, ...],
        *,
        expected_removed_finding_ids: tuple[str, ...] = (),
    ) -> "CodemodFindingClassDelta":
        expected_ids = frozenset(expected_removed_finding_ids)
        before_findings_by_signature = CodemodFindingClassSignature.group_findings(
            before_findings
        )
        after_findings_by_signature = CodemodFindingClassSignature.group_findings(
            after_findings
        )
        signatures = tuple(
            sorted(
                set(before_findings_by_signature) | set(after_findings_by_signature),
                key=lambda signature: signature.class_key,
            )
        )
        return cls(
            changes=tuple(
                CodemodFindingClassChange(
                    signature=signature,
                    finding_ids=CodemodFindingIdTransition(
                        before_ids=tuple(
                            finding.stable_id
                            for finding in before_findings_by_signature.get(
                                signature, ()
                            )
                        ),
                        after_ids=tuple(
                            finding.stable_id
                            for finding in after_findings_by_signature.get(
                                signature, ()
                            )
                        ),
                    ),
                    expected_removed_finding_ids=tuple(
                        finding.stable_id
                        for finding in before_findings_by_signature.get(signature, ())
                        if finding.stable_id in expected_ids
                    ),
                )
                for signature in signatures
            )
        )

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def moved_class_count(self) -> int:
        return self.count_status(CodemodFindingClassStatus.MOVED)

    @property
    def eliminated_class_count(self) -> int:
        return self.count_status(CodemodFindingClassStatus.ELIMINATED)

    def count_status(self, status: CodemodFindingClassStatus) -> int:
        return sum(1 for change in self.changes if change.status is status)

    def status_counts(self) -> JsonObject:
        return CodemodFindingClassStatus.counts(self.changes)

    def changes_for_before_ids(
        self,
        finding_ids: tuple[str, ...],
    ) -> tuple[CodemodFindingClassChange, ...]:
        selected_ids = frozenset(finding_ids)
        return tuple(
            change
            for change in self.changes
            if selected_ids.intersection(change.before_finding_ids)
        )

    def to_dict(self) -> JsonObject:
        return {
            "class_change_count": self.change_count,
            "moved_class_count": self.moved_class_count,
            "eliminated_class_count": self.eliminated_class_count,
            "status_counts": self.status_counts(),
            "changes": tuple(change.to_dict() for change in self.changes),
        }


@dataclass(frozen=True)
class CodemodRefactorGoalProgress:
    """Before/after target-finding progress for one goal stage."""

    finding_ids: CodemodFindingIdTransition

    @classmethod
    def from_findings(
        cls,
        migration_type: type[RefactorConcept],
        before_findings: Iterable[RefactorFinding],
        after_findings: Iterable[RefactorFinding],
        *,
        before_snapshot: CodemodSourceSnapshot,
        after_snapshot: CodemodSourceSnapshot,
    ) -> "CodemodRefactorGoalProgress":
        return cls(
            finding_ids=CodemodFindingIdTransition(
                before_ids=tuple(
                    finding.stable_id
                    for finding in migration_type.target_findings(
                        before_findings,
                        before_snapshot,
                    )
                ),
                after_ids=tuple(
                    finding.stable_id
                    for finding in migration_type.target_findings(
                        after_findings,
                        after_snapshot,
                    )
                ),
            )
        )

    @property
    def before_target_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.before_ids

    @property
    def after_target_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.after_ids

    @property
    def removed_target_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.removed_ids

    @property
    def surviving_target_finding_ids(self) -> tuple[str, ...]:
        return self.finding_ids.surviving_ids

    @property
    def removed_target_finding_count(self) -> int:
        return len(self.removed_target_finding_ids)

    @property
    def surviving_target_finding_count(self) -> int:
        return len(self.surviving_target_finding_ids)

    @property
    def achieved(self) -> bool:
        return not self.after_target_finding_ids

    @property
    def made_progress(self) -> bool:
        return self.removed_target_finding_count > 0

    def to_dict(self) -> JsonObject:
        return {
            "before_target_finding_ids": self.before_target_finding_ids,
            "after_target_finding_ids": self.after_target_finding_ids,
            "removed_target_finding_ids": self.removed_target_finding_ids,
            "surviving_target_finding_ids": self.surviving_target_finding_ids,
            "removed_target_finding_count": self.removed_target_finding_count,
            "surviving_target_finding_count": self.surviving_target_finding_count,
            "achieved": self.achieved,
            "made_progress": self.made_progress,
        }


@dataclass(frozen=True)
class CodemodRefactorGoalStage:
    """One simulated or applied staged plan toward a refactor goal."""

    class_plan_report: FindingRecipeClassPlanReport
    simulation: CodemodPlanDocumentSimulation
    progress: CodemodRefactorGoalProgress
    finding_delta: CodemodFindingDelta
    applied: bool = False

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return self.class_plan_report.finding_plan.expected_removed_finding_ids

    @property
    def rewrite_count(self) -> int:
        return self.simulation.simulation.applied_rewrite_count

    def to_dict(self) -> JsonObject:
        return {
            "applied": self.applied,
            "simulation": self.simulation.to_dict(),
            "progress": self.progress.to_dict(),
            "finding_delta": self.finding_delta.to_dict(
                self.expected_removed_finding_ids
            ),
            "class_plan_report": self.class_plan_report.to_dict(),
        }


@dataclass(frozen=True)
class CodemodRefactorGoalReport:
    """Machine-readable result of a goal-directed staged codemod run."""

    stop_reason: CodemodWorkflowStopReason
    final_finding_count: int
    migration_type: type[RefactorConcept]
    stages: tuple[CodemodRefactorGoalStage, ...]

    @property
    def final_target_finding_ids(self) -> tuple[str, ...]:
        if not self.stages:
            return ()
        return self.stages[-1].progress.after_target_finding_ids

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def total_rewrite_count(self) -> int:
        return sum(stage.rewrite_count for stage in self.stages)

    @property
    def replay_sequence(self) -> CodemodPlanSequence:
        return CodemodPlanSequence(
            documents=tuple(
                stage.simulation.document
                for stage in self.stages
                if stage.simulation.document.has_recipes
            )
        )

    def to_markdown(self) -> str:
        lines = [
            "Codemod refactor goal report:",
            f"   - Migration: {self.migration_type.concept_key()}",
            f"   - Stop reason: {self.stop_reason.value}",
            f"   - Stages: {self.stage_count}",
            f"   - Rewrites: {self.total_rewrite_count}",
            f"   - Final findings: {self.final_finding_count}",
            f"   - Remaining target findings: {len(self.final_target_finding_ids)}",
        ]
        for stage_number, stage in enumerate(self.stages, start=1):
            lines.append(
                "   - "
                f"Stage {stage_number}: "
                f"rewrites={stage.rewrite_count}, "
                f"removed_targets={stage.progress.removed_target_finding_count}, "
                f"surviving_targets={stage.progress.surviving_target_finding_count}, "
                f"applied={stage.applied}"
            )
        return "\n".join(lines)

    def to_dict(self) -> JsonObject:
        return {
            "migration": self.migration_type.concept_key(),
            "stop_reason": self.stop_reason.value,
            "stage_count": self.stage_count,
            "total_rewrite_count": self.total_rewrite_count,
            "final_finding_count": self.final_finding_count,
            "final_target_finding_ids": self.final_target_finding_ids,
            "replay_sequence": self.replay_sequence.to_dict(),
            "stages": tuple(stage.to_dict() for stage in self.stages),
        }


@dataclass(frozen=True)
class CodemodProjectedFindingReport:
    """Before/after advisor findings for one simulated codemod source state."""

    before_findings: tuple[RefactorFinding, ...]
    after_scan: "CodemodWorkflowScan"
    source_sequence: CodemodPlanSequence | None = None
    expected_removed_finding_ids: tuple[str, ...] = ()
    include_source_index: bool = False
    include_continuation: bool = False

    @property
    def scan_mode(self) -> CodemodProjectedScanMode:
        return self.after_scan.scan_mode

    @property
    def before_finding_count(self) -> int:
        return len(self.before_findings)

    @property
    def after_findings(self) -> tuple[RefactorFinding, ...]:
        return tuple(self.after_scan.findings)

    @property
    def after_finding_count(self) -> int:
        return len(self.after_findings)

    @property
    def projected_source_index(self) -> SourceIndex:
        return self.after_scan.source_index

    @cached_property
    def finding_delta(self) -> CodemodFindingDelta:
        return CodemodFindingDelta.from_findings(
            self.before_findings,
            self.after_findings,
        )

    @cached_property
    def finding_class_delta(self) -> CodemodFindingClassDelta:
        return CodemodFindingClassDelta.from_findings(
            self.before_findings,
            self.after_findings,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
        )

    @cached_property
    def continuation_report(self) -> CodemodPlanSequenceContinuationReport:
        projected_snapshot = self.after_scan.source_snapshot
        after_findings = self.after_findings
        return CodemodPlanSequenceContinuationReport(
            sequence=self.source_sequence or CodemodPlanSequence(),
            source_index=projected_snapshot.source_index,
            findings=after_findings,
            plan=projected_snapshot.plan_from_findings(after_findings),
        )

    def class_plan_delta_report(
        self,
        class_plan_report: FindingRecipeClassPlanReport,
    ) -> "CodemodClassPlanProjectedDeltaReport":
        return CodemodClassPlanProjectedDeltaReport(
            class_plan_report=class_plan_report,
            projected_finding_report=self,
        )

    def to_dict(self) -> JsonObject:
        after_findings = self.after_findings
        payload = {
            "scan_mode": self.scan_mode.value,
            "before_finding_count": self.before_finding_count,
            "after_finding_count": self.after_finding_count,
            "finding_delta": self.finding_delta.to_dict(),
            "finding_class_delta": self.finding_class_delta.to_dict(),
            "after_findings": tuple(finding.to_dict() for finding in after_findings),
        }
        if self.include_source_index:
            payload["projected_source_index"] = (
                self.after_scan.source_snapshot.source_index.to_dict()
            )
        if self.include_continuation:
            continuation_report = self.continuation_report
            payload.update(
                {
                    "projected_finding_recipe_plan": continuation_report.plan.to_dict(),
                    "projected_finding_continuation": continuation_report.to_dict(),
                }
            )
        return payload


@dataclass(frozen=True)
class CodemodClassPlanProjectedDelta:
    """Projected before/after finding-class result for one execution class plan."""

    class_plan: FindingRecipeClassPlan
    changes: tuple[CodemodFindingClassChange, ...]

    @classmethod
    def from_class_plan(
        cls,
        class_plan: FindingRecipeClassPlan,
        finding_class_delta: CodemodFindingClassDelta,
    ) -> "CodemodClassPlanProjectedDelta":
        return cls(
            class_plan=class_plan,
            changes=finding_class_delta.changes_for_before_ids(class_plan.finding_ids),
        )

    @property
    def status_counts(self) -> JsonObject:
        return CodemodFindingClassStatus.counts(self.changes)

    @property
    def site_deltas(self) -> tuple["CodemodClassPlanSiteProjectedDelta", ...]:
        return tuple(
            CodemodClassPlanSiteProjectedDelta.from_synthesis_record(
                synthesis_record,
                self.changes,
                expected_removed_finding_ids=(
                    self.class_plan.expected_removed_finding_ids
                ),
            )
            for synthesis_record in self.class_plan.synthesis_records
        )

    @property
    def fulfilled_expected_removals(self) -> bool:
        surviving_ids = {
            finding_id
            for change in self.changes
            for finding_id in change.after_finding_ids
        }
        return not any(
            finding_id in surviving_ids
            for finding_id in self.class_plan.expected_removed_finding_ids
        )

    def to_dict(self) -> JsonObject:
        return {
            "class_id": self.class_plan.execution_class.class_id,
            "fulfilled_expected_removals": self.fulfilled_expected_removals,
            "status_counts": self.status_counts,
            "changes": tuple(change.to_dict() for change in self.changes),
            "site_deltas": tuple(
                site_delta.to_dict() for site_delta in self.site_deltas
            ),
        }


@dataclass(frozen=True)
class CodemodClassPlanSiteProjectedDelta:
    """Projected finding-class status for one planned site inside a class plan."""

    synthesis_record: FindingRecipeSynthesisRecord
    changes: tuple[CodemodFindingClassChange, ...]
    expected_removed_finding_ids: tuple[str, ...] = ()

    @classmethod
    def from_synthesis_record(
        cls,
        synthesis_record: FindingRecipeSynthesisRecord,
        changes: tuple[CodemodFindingClassChange, ...],
        *,
        expected_removed_finding_ids: tuple[str, ...],
    ) -> "CodemodClassPlanSiteProjectedDelta":
        return cls(
            synthesis_record=synthesis_record,
            changes=tuple(
                change
                for change in changes
                if synthesis_record.finding_id in change.before_finding_ids
            ),
            expected_removed_finding_ids=tuple(
                finding_id
                for finding_id in expected_removed_finding_ids
                if finding_id == synthesis_record.finding_id
            ),
        )

    @property
    def status_counts(self) -> JsonObject:
        return CodemodFindingClassStatus.counts(self.changes)

    @property
    def fulfilled_expected_removal(self) -> bool:
        surviving_ids = {
            finding_id
            for change in self.changes
            for finding_id in change.after_finding_ids
        }
        return not (
            frozenset(self.expected_removed_finding_ids) & frozenset(surviving_ids)
        )

    def to_dict(self) -> JsonObject:
        return {
            "finding_id": self.synthesis_record.finding_id,
            "fulfilled_expected_removal": self.fulfilled_expected_removal,
            "status_counts": self.status_counts,
            "changes": tuple(change.to_dict() for change in self.changes),
        }


@dataclass(frozen=True)
class CodemodClassPlanProjectedDeltaReport(CodemodJsonReport):
    """Join simulated finding-class deltas back onto synthesized class plans."""

    class_plan_report: FindingRecipeClassPlanReport
    projected_finding_report: CodemodProjectedFindingReport

    @property
    def finding_class_delta(self) -> CodemodFindingClassDelta:
        return self.projected_finding_report.finding_class_delta

    @property
    def class_deltas(self) -> tuple[CodemodClassPlanProjectedDelta, ...]:
        return tuple(
            CodemodClassPlanProjectedDelta.from_class_plan(
                class_plan,
                self.finding_class_delta,
            )
            for class_plan in self.class_plan_report.classes
        )

    def to_dict(self) -> JsonObject:
        return {
            "classes": tuple(
                class_delta.to_dict() for class_delta in self.class_deltas
            ),
        }


@dataclass(frozen=True)
class CodemodWorkflowScan:
    """Parsed modules and findings for one migration state."""

    modules: list[ParsedModule]
    findings: list[RefactorFinding]
    scan_mode: CodemodProjectedScanMode = CodemodProjectedScanMode.EXACT

    @property
    def source_index(self) -> SourceIndex:
        return self.source_snapshot.source_index

    @property
    def sources_by_file_path(self) -> dict[str, str]:
        return dict(self.source_snapshot.sources_by_file_path)

    @cached_property
    def source_snapshot(self) -> CodemodSourceSnapshot:
        return CodemodSourceSnapshot.from_modules(self.modules, self.findings)


@dataclass(frozen=True)
class CodemodSimulationFindingProjection:
    """Analyze advisor findings after applying a simulation in memory."""

    modules: tuple[ParsedModule, ...]
    findings: tuple[RefactorFinding, ...]
    simulation: CodemodSimulationReport
    config: DetectorConfig
    roots: tuple[Path, ...] = ()
    report_roots: tuple[Path, ...] = ()
    analysis_workers: int = 1
    semantic_descent_source: SemanticDescentGraphAnalysisSource = field(
        default_factory=SemanticDescentGraphAnalysisSource
    )
    source_sequence: CodemodPlanSequence | None = None
    expected_removed_finding_ids: tuple[str, ...] = ()
    include_continuation: bool = False
    include_source_index: bool = False

    def scan(self) -> CodemodWorkflowScan:
        projected_modules = ProjectedScanModuleSet(
            modules=self.modules,
            simulation=self.simulation,
            roots=self.roots,
        ).modules_after_projection()
        if self.report_roots:
            return self.evidence_local_projected_scan(projected_modules)
        return CodemodWorkflowScan(
            modules=list(projected_modules),
            findings=analyze_modules(projected_modules, self.config),
            scan_mode=CodemodProjectedScanMode.EXACT,
        )

    def evidence_local_projected_scan(
        self,
        projected_modules: tuple[ParsedModule, ...],
    ) -> CodemodWorkflowScan:
        changed_paths = self.changed_paths
        if not changed_paths:
            return CodemodWorkflowScan(
                modules=list(projected_modules),
                findings=list(self.findings),
                scan_mode=CodemodProjectedScanMode.EVIDENCE_LOCAL_PARTIAL,
            )
        detector_types = default_detector_types_for_analysis()
        detector_selection = EvidenceLocalPartialDetectorSelection.from_detector_types(
            detector_types
        )
        rerun_detector_types = detector_selection.rerun_detector_family
        reuse_authority = EvidenceLocalFindingReuseAuthority(rerun_detector_types)
        changed_findings = self.changed_findings(
            projected_modules,
            changed_paths,
            detector_types=rerun_detector_types,
        )
        findings = self.report_scoped_findings(
            SortedFindingsAuthority.sort(
                [
                    *EvidenceLocalFindingReuseAuthority.unchanged_findings(
                        self.findings,
                        changed_paths,
                    ),
                    *reuse_authority.retained_changed_findings(
                        self.findings,
                        changed_paths,
                    ),
                    *reuse_authority.changed_findings(
                        changed_findings,
                        changed_paths,
                    ),
                ],
                detector_types=detector_types,
            )
        )
        return CodemodWorkflowScan(
            modules=list(projected_modules),
            findings=findings,
            scan_mode=CodemodProjectedScanMode.EVIDENCE_LOCAL_PARTIAL,
        )

    def changed_findings(
        self,
        projected_modules: tuple[ParsedModule, ...],
        changed_paths: frozenset[str],
        *,
        detector_types: tuple[type[IssueDetector], ...],
    ) -> list[RefactorFinding]:
        if not detector_types:
            return []
        changed_modules = self.changed_modules(projected_modules, changed_paths)
        if not changed_modules:
            return []
        return analyze_detector_types(
            changed_modules,
            self.config,
            detector_types=detector_types,
            semantic_descent_source=self.projected_semantic_descent_source(
                projected_modules,
                detector_types,
            ),
            analysis_workers=self.analysis_workers,
            detector_type_minimum_auto_work_items=4,
        )

    def projected_semantic_descent_source(
        self,
        projected_modules: tuple[ParsedModule, ...],
        detector_types: tuple[type[IssueDetector], ...],
    ) -> SemanticDescentGraphAnalysisSource:
        if not any(
            issubclass(detector_type, SemanticDescentGraphIssueDetector)
            for detector_type in detector_types
        ):
            return self.semantic_descent_source
        return SemanticDescentGraphAnalysisSource(
            cached_graph=self.semantic_descent_source.graph_for_modules(
                list(projected_modules)
            )
        )

    def changed_modules(
        self,
        projected_modules: tuple[ParsedModule, ...],
        changed_paths: frozenset[str],
    ) -> list[ParsedModule]:
        return [
            module
            for module in projected_modules
            if Path(module.path).resolve().as_posix() in changed_paths
        ]

    @property
    def changed_paths(self) -> frozenset[str]:
        return frozenset(
            Path(path).resolve().as_posix()
            for path in self.simulation.rewritten_sources
        )

    def report_scoped_findings(
        self,
        findings: list[RefactorFinding],
    ) -> list[RefactorFinding]:
        return AnalysisPathScope(
            analysis_roots=self.roots,
            report_roots=self.report_roots,
        ).filter_findings(findings)

    def report(self) -> CodemodProjectedFindingReport:
        after_scan = self.scan()
        return CodemodProjectedFindingReport(
            before_findings=self.findings,
            after_scan=after_scan,
            source_sequence=self.source_sequence,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
            include_source_index=self.include_source_index,
            include_continuation=self.include_continuation,
        )


@dataclass(frozen=True, kw_only=True)
class CodemodRefactorGoalRunner:
    """Derive and execute stages until one semantic migration resolves."""

    resolved_dir: Path | None = None
    enabled: bool = False
    roots: tuple[Path, ...]
    report_roots: tuple[Path, ...] = ()
    config: DetectorConfig
    parse_workers: int
    dry_run: bool
    guard_suite: ArchitectureGuardSuite
    initial_scan: CodemodWorkflowScan | None = None
    migration_type: type[RefactorConcept]
    max_stages: int = 8

    def starting_scan(self) -> CodemodWorkflowScan:
        if self.initial_scan is None:
            return self.fresh_scan()
        if not self.report_roots:
            return self.initial_scan
        return replace(
            self.initial_scan,
            findings=self.report_scoped_findings(self.initial_scan.findings),
        )

    def fresh_scan(self) -> CodemodWorkflowScan:
        modules = parse_python_module_roots(
            self.roots,
            cache_dir=self.resolved_dir,
            use_parse_cache=self.enabled,
            parse_workers=self.parse_workers,
        )
        return CodemodWorkflowScan(
            modules=modules,
            findings=self.report_scoped_findings(analyze_modules(modules, self.config)),
        )

    def projected_scan(
        self,
        scan: CodemodWorkflowScan,
        simulation: CodemodSimulationReport,
    ) -> CodemodWorkflowScan:
        return CodemodSimulationFindingProjection(
            modules=tuple(scan.modules),
            findings=tuple(scan.findings),
            simulation=simulation,
            config=self.config,
            roots=self.roots,
            report_roots=self.report_roots,
        ).scan()

    def projected_target_scan(
        self,
        scan: CodemodWorkflowScan,
        simulation: CodemodSimulationReport,
        target_findings: tuple[RefactorFinding, ...],
    ) -> CodemodWorkflowScan:
        """Rerun only active goal detector declarations between stages."""

        detector_ids = self.migration_type.detector_ids_for_findings(target_findings)
        detector_types = tuple(
            detector_type
            for detector_type in IssueDetector.registered_detector_types()
            if detector_type.effective_detector_id() in detector_ids
        )
        if len(detector_types) != len(detector_ids):
            return self.projected_scan(scan, simulation)
        projected_modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodWorkflowScan(
            modules=list(projected_modules),
            findings=self.report_scoped_findings(
                analyze_detector_types(
                    list(projected_modules),
                    self.config,
                    detector_types=detector_types,
                )
            ),
            scan_mode=CodemodProjectedScanMode.TARGET_DETECTOR_PARTIAL,
        )

    def exact_scan(self, scan: CodemodWorkflowScan) -> CodemodWorkflowScan:
        """Certify every detector over an in-memory migration state once."""

        if scan.scan_mode is CodemodProjectedScanMode.EXACT:
            return scan
        return CodemodWorkflowScan(
            modules=scan.modules,
            findings=self.report_scoped_findings(
                analyze_modules(scan.modules, self.config)
            ),
            scan_mode=CodemodProjectedScanMode.EXACT,
        )

    def report_scoped_findings(
        self,
        findings: Iterable[RefactorFinding],
    ) -> list[RefactorFinding]:
        return AnalysisPathScope(
            analysis_roots=self.roots,
            report_roots=self.report_roots,
        ).filter_findings(list(findings))

    def run(self) -> CodemodRefactorGoalReport:
        if self.max_stages < 1:
            raise ValueError("max_stages must be at least 1")
        stages: list[CodemodRefactorGoalStage] = []
        active_scan = self.starting_scan()
        if not self.target_findings(active_scan):
            return self.report(
                (),
                active_scan,
                CodemodWorkflowStopReason.ACHIEVED,
            )
        for _stage in range(self.max_stages):
            snapshot = active_scan.source_snapshot
            target_findings = self.target_findings(active_scan)
            plan = snapshot.plan_from_findings(target_findings)
            class_plan_report = FindingRecipeClassPlanReport.from_finding_plan(
                target_findings,
                root=self.class_plan_root(),
                finding_plan=plan,
            )
            document = CodemodPlanDocument(
                recipes=plan.document.recipes,
                guard_suite=self.guard_suite.merge(plan.document.guard_suite),
            )
            simulation = document.simulate_snapshot(snapshot)
            projected_scan = (
                self.projected_target_scan(
                    active_scan,
                    simulation.simulation,
                    target_findings,
                )
                if plan.document.has_recipes
                else active_scan
            )
            if not self.target_findings(projected_scan):
                projected_scan = self.exact_scan(projected_scan)
            stage = self.stage(
                active_scan,
                projected_scan,
                class_plan_report=class_plan_report,
                simulation=simulation,
            )
            if not plan.document.has_recipes:
                return self.report(
                    (*stages, stage),
                    active_scan,
                    CodemodWorkflowStopReason.NO_EXECUTABLE_RECIPES,
                )
            if stage.rewrite_count == 0:
                return self.report(
                    (*stages, stage),
                    active_scan,
                    CodemodWorkflowStopReason.EMPTY_REWRITE_BATCH,
                )
            if not stage.simulation.is_clean:
                return self.report(
                    (*stages, stage),
                    active_scan,
                    CodemodWorkflowStopReason.ARCHITECTURE_GUARD_FAILED,
                )
            if self.dry_run:
                next_scan = projected_scan
                recorded_stage = stage
            else:
                stage.simulation.apply()
                next_scan = self.fresh_scan()
                recorded_stage = replace(
                    self.stage(
                        active_scan,
                        next_scan,
                        class_plan_report=class_plan_report,
                        simulation=simulation,
                    ),
                    applied=True,
                )
            stages.append(recorded_stage)
            if recorded_stage.progress.achieved:
                return self.report(
                    tuple(stages),
                    next_scan,
                    CodemodWorkflowStopReason.ACHIEVED,
                )
            if not recorded_stage.progress.made_progress:
                return self.report(
                    tuple(stages),
                    next_scan,
                    CodemodWorkflowStopReason.NO_PROGRESS,
                )
            active_scan = next_scan
        return self.report(
            tuple(stages),
            self.exact_scan(active_scan),
            CodemodWorkflowStopReason.MAX_STAGES,
        )

    def target_findings(
        self,
        scan: CodemodWorkflowScan,
    ) -> tuple[RefactorFinding, ...]:
        return self.migration_type.target_findings(
            scan.findings,
            scan.source_snapshot,
        )

    def class_plan_root(self) -> Path:
        if self.roots:
            return self.roots[0]
        return Path.cwd()

    def stage(
        self,
        before_scan: CodemodWorkflowScan,
        after_scan: CodemodWorkflowScan,
        *,
        class_plan_report: FindingRecipeClassPlanReport,
        simulation: CodemodPlanDocumentSimulation,
    ) -> CodemodRefactorGoalStage:
        before_target_findings = self.target_findings(before_scan)
        after_target_findings = self.target_findings(after_scan)
        return CodemodRefactorGoalStage(
            class_plan_report=class_plan_report,
            simulation=simulation,
            progress=CodemodRefactorGoalProgress.from_findings(
                self.migration_type,
                before_scan.findings,
                after_scan.findings,
                before_snapshot=before_scan.source_snapshot,
                after_snapshot=after_scan.source_snapshot,
            ),
            finding_delta=CodemodFindingDelta.from_findings(
                before_target_findings,
                after_target_findings,
            ),
        )

    def report(
        self,
        stages: tuple[CodemodRefactorGoalStage, ...],
        scan: CodemodWorkflowScan,
        reason: CodemodWorkflowStopReason,
    ) -> CodemodRefactorGoalReport:
        verified_scan = self.exact_scan(scan)
        return CodemodRefactorGoalReport(
            migration_type=self.migration_type,
            stages=stages,
            stop_reason=reason,
            final_finding_count=len(verified_scan.findings),
        )


@dataclass(frozen=True)
class ProjectedScanModuleSet:
    """Parsed module set after a codemod simulation, including created files."""

    modules: tuple[ParsedModule, ...]
    simulation: CodemodSimulationReport
    roots: tuple[Path, ...] = ()

    def modules_after_projection(self) -> tuple[ParsedModule, ...]:
        return (
            *self.projected_existing_modules(),
            *self.created_modules(),
        )

    def projected_existing_modules(self) -> tuple[ParsedModule, ...]:
        return tuple(self.projected_module(module) for module in self.modules)

    def projected_module(self, module: ParsedModule) -> ParsedModule:
        projection = ProjectedModuleSource(
            module=module,
            simulation=self.simulation,
        )
        if not projection.has_rewrite:
            return module
        source = projection.source
        return ParsedModule(
            path=module.path,
            module_name=module.module_name,
            is_package_init=module.is_package_init,
            module=ast.parse(source, filename=module.file_path),
            source=source,
        )

    def created_modules(self) -> tuple[ParsedModule, ...]:
        known_paths = self.known_resolved_paths()
        return tuple(
            self.created_module(file_path, source)
            for file_path, source in sorted(self.simulation.rewritten_sources.items())
            if Path(file_path).resolve() not in known_paths
        )

    def known_resolved_paths(self) -> frozenset[Path]:
        return frozenset(module.path.resolve() for module in self.modules)

    def created_module(self, file_path: str, source: str) -> ParsedModule:
        path = Path(file_path)
        return ParsedModule(
            path=path,
            module_name=ProjectedModuleName(
                file_path=path,
                roots=self.roots,
            ).module_name(),
            is_package_init=path.name == "__init__.py",
            module=ast.parse(source, filename=file_path),
            source=source,
        )


@dataclass(frozen=True)
class ProjectedModuleName:
    """Resolve module names for simulated sources using known scan roots."""

    file_path: Path
    roots: tuple[Path, ...] = ()

    def module_name(self) -> str:
        relative_path = self.relative_path()
        return module_name_from_source_path(relative_path.as_posix())

    def relative_path(self) -> Path:
        resolved_file_path = self.file_path.resolve()
        for root in self.roots:
            resolved_root = root.resolve()
            if resolved_root.is_file():
                resolved_root = resolved_root.parent
            try:
                return resolved_file_path.relative_to(resolved_root)
            except ValueError:
                continue
        return self.file_path


@dataclass(frozen=True)
class ProjectedModuleSource:
    """Resolve one module's source in a simulated post-rewrite snapshot."""

    module: ParsedModule
    simulation: CodemodSimulationReport

    @property
    def module_path(self) -> str:
        return self.module.file_path

    @property
    def has_rewrite(self) -> bool:
        return self.module_path in self.simulation.rewritten_sources

    @property
    def source(self) -> str:
        if self.has_rewrite:
            return self.simulation.rewritten_sources[self.module_path]
        return self.module.source
