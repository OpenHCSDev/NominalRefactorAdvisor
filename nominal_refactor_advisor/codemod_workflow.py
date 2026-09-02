"""Reusable closed-loop workflows for executable codemod DSL plans."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar

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
    ArchitectureGuardReport,
    ArchitectureGuardSuite,
    CodemodPlanDocumentSimulation,
    CodemodJsonReport,
    CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    FindingRecipeClassPlan,
    FindingRecipeClassPlanReport,
    FindingRecipeSynthesisRecord,
    FindingRecipeFrontierBudget,
    FindingRecipeTrajectoryObstacle,
    JsonObject,
    RefactorConcept,
    module_name_from_source_path,
)
from .detectors import DetectorConfig, IssueDetector, SemanticDescentGraphIssueDetector
from .models import FindingObligationClass, RefactorFinding
from .source_index import SourceIndex

IdentityT = TypeVar("IdentityT", bound=Hashable)


class CodemodWorkflowStopReason(StrEnum):
    """Terminal state for staged codemod workflows."""

    def __new__(cls, value: str, completed: bool) -> "CodemodWorkflowStopReason":
        member = str.__new__(cls, value)
        member._value_ = value
        member._completed = completed
        return member

    ACHIEVED = ("achieved", True)
    ARCHITECTURE_GUARD_FAILED = ("architecture_guard_failed", False)
    APPLICATION_VERIFICATION_FAILED = ("application_verification_failed", False)
    UNPROVED_TRAJECTORY = ("unproved_trajectory", False)
    NO_PROVED_TRAJECTORY = ("no_proved_trajectory", False)

    @property
    def completed(self) -> bool:
        return self._completed


class CodemodRefactorTrajectoryStatus(StrEnum):
    """Proof status for exhaustive reachable-state exploration."""

    PROVED = (
        "proved",
        CodemodWorkflowStopReason.ACHIEVED,
        True,
        lambda proof: not proof.obstacles and len(proof.terminals) == 1,
    )
    NO_TERMINAL_STATE = (
        "no_terminal_state",
        CodemodWorkflowStopReason.NO_PROVED_TRAJECTORY,
        False,
        lambda proof: not proof.obstacles and not proof.terminals,
    )
    AMBIGUOUS_TERMINAL_STATES = (
        "ambiguous_terminal_states",
        CodemodWorkflowStopReason.UNPROVED_TRAJECTORY,
        False,
        lambda proof: not proof.obstacles and len(proof.terminals) > 1,
    )
    INCOMPLETE = (
        "incomplete",
        CodemodWorkflowStopReason.UNPROVED_TRAJECTORY,
        False,
        lambda proof: bool(proof.obstacles),
    )

    def __new__(
        cls,
        value: str,
        stop_reason: CodemodWorkflowStopReason,
        proved: bool,
        matcher: Callable[["CodemodRefactorTrajectoryProof"], bool],
    ) -> "CodemodRefactorTrajectoryStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._stop_reason = stop_reason
        member._proved = proved
        member._matcher = matcher
        return member

    @property
    def stop_reason(self) -> CodemodWorkflowStopReason:
        return self._stop_reason

    @property
    def proved(self) -> bool:
        return self._proved

    def matches(self, proof: "CodemodRefactorTrajectoryProof") -> bool:
        return self._matcher(proof)

    @classmethod
    def from_proof(
        cls,
        proof: "CodemodRefactorTrajectoryProof",
    ) -> "CodemodRefactorTrajectoryStatus":
        matching_statuses = frozenset(status for status in cls if status.matches(proof))
        if len(matching_statuses) != 1:
            raise TypeError(
                "trajectory proof must match exactly one status; matched "
                f"{tuple(sorted(status.value for status in matching_statuses))!r}"
            )
        return next(iter(matching_statuses))


class CodemodRefactorTrajectoryObstacleKind(StrEnum):
    """Typed source of incomplete reachable-state coverage."""

    RECIPE_FRONTIER = "recipe_frontier"
    DEPTH_BUDGET = "depth_budget"
    STATE_BUDGET = "state_budget"


@dataclass(frozen=True)
class CodemodRefactorTrajectoryBudget(CodemodJsonReport):
    """Single proof-search budget shared by frontier and graph exploration."""

    max_depth: int = 8
    max_states: int = 512
    recipe_frontier: FindingRecipeFrontierBudget = field(
        default_factory=FindingRecipeFrontierBudget
    )

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("trajectory depth budget must be at least 1")
        if self.max_states < 1:
            raise ValueError("trajectory state budget must be at least 1")

    def to_dict(self) -> JsonObject:
        return {
            "max_depth": self.max_depth,
            "max_states": self.max_states,
            "recipe_frontier": self.recipe_frontier.to_dict(),
        }


class CodemodProjectedScanMode(StrEnum):
    """Completeness contract for a projected post-codemod scan."""

    EXACT = ("exact", True)
    EVIDENCE_LOCAL_PARTIAL = ("evidence_local_partial", False)
    TARGET_DETECTOR_PARTIAL = ("target_detector_partial", False)

    def __new__(cls, value: str, exact: bool) -> "CodemodProjectedScanMode":
        member = str.__new__(cls, value)
        member._value_ = value
        member._exact = exact
        return member

    @property
    def exact(self) -> bool:
        return self._exact


class CodemodFindingClassStatus(StrEnum):
    """Projected status for one semantic class of advisor findings."""

    def __new__(
        cls,
        value: str,
        matcher: Callable[["CodemodFindingClassChange"], bool] | None,
    ) -> "CodemodFindingClassStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._matcher = matcher
        return member

    ELIMINATED = (
        "eliminated",
        lambda change: bool(change.before_finding_ids) and not change.after_finding_ids,
    )
    MOVED = (
        "moved",
        lambda change: bool(change.before_finding_ids)
        and bool(change.after_finding_ids)
        and change.before_finding_count == change.after_finding_count
        and bool(change.removed_finding_ids or change.added_finding_ids),
    )
    EXPANDED = (
        "expanded",
        lambda change: bool(change.before_finding_ids)
        and change.after_finding_count > change.before_finding_count,
    )
    PARTIALLY_ELIMINATED = (
        "partially_eliminated",
        lambda change: bool(change.after_finding_ids)
        and change.after_finding_count < change.before_finding_count,
    )
    PERSISTED = (
        "persisted",
        lambda change: bool(change.before_finding_ids)
        and change.before_finding_count == change.after_finding_count
        and not change.removed_finding_ids
        and bool(change.expected_removed_finding_ids),
    )
    INTRODUCED = (
        "introduced",
        lambda change: not change.before_finding_ids and bool(change.after_finding_ids),
    )
    UNCHANGED = ("unchanged", None)

    def matches(self, change: "CodemodFindingClassChange") -> bool:
        return self._matcher is not None and self._matcher(change)

    @classmethod
    def from_change(
        cls,
        change: "CodemodFindingClassChange",
    ) -> "CodemodFindingClassStatus":
        matching_statuses = frozenset(
            status for status in cls if status.matches(change)
        )
        if len(matching_statuses) > 1:
            raise TypeError(
                "finding-class transition matches multiple statuses: "
                f"{tuple(sorted(status.value for status in matching_statuses))!r}"
            )
        return next(iter(matching_statuses), cls.UNCHANGED)

    @classmethod
    def counts(
        cls,
        changes: tuple["CodemodFindingClassChange", ...],
    ) -> JsonObject:
        change_counts = Counter(change.status for change in changes)
        return {
            status.value: change_counts[status]
            for status in cls
            if change_counts[status]
        }


@dataclass(frozen=True)
class CodemodIdentityTransition(Generic[IdentityT]):
    """Before/after transition algebra shared by nominal identity roles."""

    before_ids: tuple[IdentityT, ...]
    after_ids: tuple[IdentityT, ...]

    def with_after_ids(
        self,
        after_ids: Iterable[IdentityT],
    ) -> Self:
        """Project the same before-state onto one newly observed after-state."""

        return replace(self, after_ids=tuple(after_ids))

    @property
    def removed_ids(self) -> tuple[IdentityT, ...]:
        after_ids = frozenset(self.after_ids)
        return tuple(item for item in self.before_ids if item not in after_ids)

    @property
    def added_ids(self) -> tuple[IdentityT, ...]:
        before_ids = frozenset(self.before_ids)
        return tuple(item for item in self.after_ids if item not in before_ids)

    @property
    def surviving_ids(self) -> tuple[IdentityT, ...]:
        after_ids = frozenset(self.after_ids)
        return tuple(item for item in self.before_ids if item in after_ids)

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


@dataclass(frozen=True)
class CodemodFindingIdTransition(CodemodIdentityTransition[str]):
    """Finding identity transition for codemod delta reports."""

    @classmethod
    def from_findings(
        cls,
        before_findings: Iterable[RefactorFinding],
        after_findings: Iterable[RefactorFinding],
    ) -> "CodemodFindingIdTransition":
        """Project finding declarations onto their stable identity transition."""

        return cls(
            before_ids=tuple(finding.stable_id for finding in before_findings),
            after_ids=tuple(finding.stable_id for finding in after_findings),
        )

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
class CodemodDetectorIdTransition(CodemodIdentityTransition[str]):
    """Detector provenance before and after one semantic obligation transition."""

    @classmethod
    def from_findings(
        cls,
        before_findings: Iterable[RefactorFinding],
        after_findings: Iterable[RefactorFinding],
    ) -> "CodemodDetectorIdTransition":
        return cls(
            before_ids=tuple(
                dict.fromkeys(finding.detector_id for finding in before_findings)
            ),
            after_ids=tuple(
                dict.fromkeys(finding.detector_id for finding in after_findings)
            ),
        )

    def to_dict(self) -> JsonObject:
        return {
            "before_detector_ids": self.before_ids,
            "after_detector_ids": self.after_ids,
            "removed_detector_ids": self.removed_ids,
            "added_detector_ids": self.added_ids,
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
            finding_ids=CodemodFindingIdTransition.from_findings(
                before_findings,
                after_findings,
            )
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
            expected_removed_finding_count=len(expected_removed_finding_ids),
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
class CodemodFindingClassChange(CodemodFindingDelta):
    """Before/after membership for one semantic finding class."""

    obligation_class: FindingObligationClass
    detector_ids: CodemodDetectorIdTransition
    expected_removed_finding_ids: tuple[str, ...] = ()

    @property
    def status(self) -> CodemodFindingClassStatus:
        return CodemodFindingClassStatus.from_change(self)

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    @property
    def finding_count_increase(self) -> int:
        """Return newly introduced obligations in this semantic class."""

        return max(self.after_finding_count - self.before_finding_count, 0)

    def to_dict(self) -> JsonObject:
        return JsonObject(
            **self.finding_ids.to_dict(),
            obligation_class=self.obligation_class.to_dict(),
            detector_transition=self.detector_ids.to_dict(),
            status=self.status.value,
            finding_count_increase=self.finding_count_increase,
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
        before_findings_by_class = cls.group_findings(before_findings)
        after_findings_by_class = cls.group_findings(after_findings)
        changes = []
        for obligation_class in sorted(
            set(before_findings_by_class) | set(after_findings_by_class)
        ):
            before_class_findings = before_findings_by_class.get(obligation_class, ())
            after_class_findings = after_findings_by_class.get(obligation_class, ())
            changes.append(
                CodemodFindingClassChange(
                    obligation_class=obligation_class,
                    detector_ids=CodemodDetectorIdTransition.from_findings(
                        before_class_findings,
                        after_class_findings,
                    ),
                    finding_ids=CodemodFindingIdTransition.from_findings(
                        before_class_findings,
                        after_class_findings,
                    ),
                    expected_removed_finding_ids=tuple(
                        finding.stable_id
                        for finding in before_class_findings
                        if finding.stable_id in expected_ids
                    ),
                )
            )
        return cls(changes=tuple(changes))

    @staticmethod
    def group_findings(
        findings: tuple[RefactorFinding, ...],
    ) -> dict[FindingObligationClass, tuple[RefactorFinding, ...]]:
        grouped_findings: dict[FindingObligationClass, list[RefactorFinding]] = {}
        for finding in findings:
            grouped_findings.setdefault(finding.obligation_class, []).append(finding)
        return {
            obligation_class: tuple(class_findings)
            for obligation_class, class_findings in grouped_findings.items()
        }

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def moved_class_count(self) -> int:
        return self.count_status(CodemodFindingClassStatus.MOVED)

    @property
    def eliminated_class_count(self) -> int:
        return self.count_status(CodemodFindingClassStatus.ELIMINATED)

    @property
    def increased_changes(self) -> tuple[CodemodFindingClassChange, ...]:
        """Return semantic classes whose finding obligations increased."""

        return tuple(
            change for change in self.changes if change.finding_count_increase > 0
        )

    @property
    def finding_count_increase(self) -> int:
        return sum(change.finding_count_increase for change in self.increased_changes)

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
            "finding_count_increase": self.finding_count_increase,
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
        before_findings: Iterable[RefactorFinding],
        after_findings: Iterable[RefactorFinding],
    ) -> "CodemodRefactorGoalProgress":
        return cls(
            finding_ids=CodemodFindingIdTransition.from_findings(
                before_findings,
                after_findings,
            )
        )

    @property
    def finding_delta(self) -> CodemodFindingDelta:
        """Project generic delta reporting from the single goal transition."""

        return CodemodFindingDelta(self.finding_ids)

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
    applied: bool = False

    @property
    def finding_delta(self) -> CodemodFindingDelta:
        return self.progress.finding_delta

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return self.class_plan_report.finding_plan.expected_removed_finding_ids

    @property
    def rewrite_count(self) -> int:
        return self.simulation.simulation.applied_rewrite_count

    def with_applied_target_findings(
        self,
        findings: Iterable[RefactorFinding],
    ) -> "CodemodRefactorGoalStage":
        """Record the exact post-commit target state on the final stage."""

        after_ids = tuple(finding.stable_id for finding in findings)
        return replace(
            self,
            progress=replace(
                self.progress,
                finding_ids=self.progress.finding_ids.with_after_ids(after_ids),
            ),
            applied=True,
        )

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
class CodemodRefactorTrajectoryObstacle(CodemodJsonReport, ABC):
    """Nominal proof obstacle for one unexhausted exact source state."""

    kind: ClassVar[CodemodRefactorTrajectoryObstacleKind]
    source_state_id: str
    depth: int

    @property
    @abstractmethod
    def reason(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evidence_payload(self) -> JsonObject:
        raise NotImplementedError

    def to_dict(self) -> JsonObject:
        return {
            "kind": self.kind.value,
            "source_state_id": self.source_state_id,
            "depth": self.depth,
            "reason": self.reason,
            **self.evidence_payload(),
        }

    def markdown_lines(self) -> tuple[str, ...]:
        return (
            "   - Trajectory obstacle: "
            f"{self.kind.value} at depth {self.depth}: {self.reason}",
        )


@dataclass(frozen=True)
class CodemodRefactorRecipeFrontierObstacle(CodemodRefactorTrajectoryObstacle):
    """Preserve typed current-state frontier evidence that failed closed."""

    kind = CodemodRefactorTrajectoryObstacleKind.RECIPE_FRONTIER
    recipe_obstacles: tuple[FindingRecipeTrajectoryObstacle, ...]

    @property
    def reason(self) -> str:
        return "the current-state recipe frontier is incomplete"

    def evidence_payload(self) -> JsonObject:
        return {
            "recipe_obstacles": tuple(
                obstacle.to_dict() for obstacle in self.recipe_obstacles
            )
        }

    def markdown_lines(self) -> tuple[str, ...]:
        return (
            *super().markdown_lines(),
            *(
                "     - "
                f"{obstacle.kind.value} [{', '.join(obstacle.finding_ids)}]: "
                f"{obstacle.reason}"
                for obstacle in self.recipe_obstacles
            ),
        )


@dataclass(frozen=True)
class CodemodRefactorDepthBudgetObstacle(CodemodRefactorTrajectoryObstacle):
    """Record a reachable state beyond the declared proof depth."""

    kind = CodemodRefactorTrajectoryObstacleKind.DEPTH_BUDGET
    max_depth: int

    @property
    def reason(self) -> str:
        return (
            f"reachable transitions exceed the declared depth limit of {self.max_depth}"
        )

    def evidence_payload(self) -> JsonObject:
        return {"max_depth": self.max_depth}


@dataclass(frozen=True)
class CodemodRefactorStateBudgetObstacle(CodemodRefactorTrajectoryObstacle):
    """Record a reachable state beyond the declared graph-size proof budget."""

    kind = CodemodRefactorTrajectoryObstacleKind.STATE_BUDGET
    max_states: int

    @property
    def reason(self) -> str:
        return f"reachable source states exceed the declared limit of {self.max_states}"

    def evidence_payload(self) -> JsonObject:
        return {"max_states": self.max_states}


@dataclass(frozen=True)
class CodemodRefactorTrajectoryState:
    """One exact in-memory source state and the path that first reached it."""

    scan: "CodemodWorkflowScan" = field(compare=False, repr=False)
    stages: tuple[CodemodRefactorGoalStage, ...] = ()

    @property
    def source_state_id(self) -> str:
        return self.scan.source_snapshot.source_state_id

    @property
    def depth(self) -> int:
        return len(self.stages)


@dataclass(frozen=True)
class CodemodRefactorTrajectoryTerminal:
    """One exact goal state reached by a completely explored trajectory graph."""

    state: CodemodRefactorTrajectoryState = field(compare=False, repr=False)

    @property
    def source_state_id(self) -> str:
        return self.state.source_state_id

    @property
    def stages(self) -> tuple[CodemodRefactorGoalStage, ...]:
        return self.state.stages

    def to_dict(self) -> JsonObject:
        return {
            "source_state_id": self.source_state_id,
            "stage_count": len(self.stages),
        }


@dataclass(frozen=True)
class CodemodRefactorGuardRejectedTerminal:
    """One target-free source state rejected by terminal architecture guards."""

    state: CodemodRefactorTrajectoryState = field(compare=False, repr=False)
    guard_report: ArchitectureGuardReport

    def to_dict(self) -> JsonObject:
        return {
            "source_state_id": self.state.source_state_id,
            "stage_count": len(self.state.stages),
            "guard_report": self.guard_report.to_dict(),
        }


@dataclass(frozen=True)
class CodemodRefactorUnjustifiedDebtTerminal:
    """Target-free state that introduced an unproved finding obligation."""

    state: CodemodRefactorTrajectoryState = field(compare=False, repr=False)
    finding_class_changes: tuple[CodemodFindingClassChange, ...]

    @property
    def finding_count_increase(self) -> int:
        return sum(
            change.finding_count_increase for change in self.finding_class_changes
        )

    def to_dict(self) -> JsonObject:
        return {
            "source_state_id": self.state.source_state_id,
            "stage_count": len(self.state.stages),
            "finding_count_increase": self.finding_count_increase,
            "finding_class_changes": tuple(
                change.to_dict() for change in self.finding_class_changes
            ),
        }


@dataclass(frozen=True)
class CodemodRefactorTrajectoryDeadEnd:
    """One fully explored non-goal state with no executable transition."""

    state: CodemodRefactorTrajectoryState = field(compare=False, repr=False)
    class_plan_report: FindingRecipeClassPlanReport = field(
        compare=False,
        repr=False,
    )

    def to_dict(self) -> JsonObject:
        return {
            "source_state_id": self.state.source_state_id,
            "depth": self.state.depth,
            "class_plan_report": self.class_plan_report.to_dict(),
        }


@dataclass(frozen=True)
class CodemodRefactorTrajectoryProof:
    """Complete reachable-state evidence with no local branch preference."""

    initial_source_state_id: str
    budget: CodemodRefactorTrajectoryBudget
    visited_state_count: int
    transition_count: int
    terminals: tuple[CodemodRefactorTrajectoryTerminal, ...] = ()
    guard_rejected_terminals: tuple[CodemodRefactorGuardRejectedTerminal, ...] = ()
    unjustified_debt_terminals: tuple[CodemodRefactorUnjustifiedDebtTerminal, ...] = ()
    dead_ends: tuple[CodemodRefactorTrajectoryDeadEnd, ...] = ()
    obstacles: tuple[CodemodRefactorTrajectoryObstacle, ...] = ()

    @property
    def status(self) -> CodemodRefactorTrajectoryStatus:
        return CodemodRefactorTrajectoryStatus.from_proof(self)

    @property
    def proved_terminal(self) -> CodemodRefactorTrajectoryTerminal:
        if not self.status.proved:
            raise TypeError("trajectory proof has no unique proved terminal state")
        return next(iter(self.terminals))

    def to_dict(self) -> JsonObject:
        return {
            "status": self.status.value,
            "initial_source_state_id": self.initial_source_state_id,
            "budget": self.budget.to_dict(),
            "visited_state_count": self.visited_state_count,
            "transition_count": self.transition_count,
            "terminal_count": len(self.terminals),
            "terminals": tuple(terminal.to_dict() for terminal in self.terminals),
            "guard_rejected_terminal_count": len(self.guard_rejected_terminals),
            "guard_rejected_terminals": tuple(
                terminal.to_dict() for terminal in self.guard_rejected_terminals
            ),
            "unjustified_debt_terminal_count": len(self.unjustified_debt_terminals),
            "unjustified_debt_terminals": tuple(
                terminal.to_dict() for terminal in self.unjustified_debt_terminals
            ),
            "dead_end_count": len(self.dead_ends),
            "dead_ends": tuple(dead_end.to_dict() for dead_end in self.dead_ends),
            "obstacles": tuple(obstacle.to_dict() for obstacle in self.obstacles),
        }


@dataclass(frozen=True)
class CodemodRefactorGoalReport:
    """Machine-readable result of a goal-directed staged codemod run."""

    stop_reason: CodemodWorkflowStopReason
    final_finding_count: int
    final_target_finding_ids: tuple[str, ...]
    migration_type: type[RefactorConcept]
    stages: tuple[CodemodRefactorGoalStage, ...]
    trajectory_proof: CodemodRefactorTrajectoryProof

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
            f"   - Trajectory proof: {self.trajectory_proof.status.value}",
            (
                "   - Reachable graph: "
                f"states={self.trajectory_proof.visited_state_count}, "
                f"transitions={self.trajectory_proof.transition_count}, "
                f"terminals={len(self.trajectory_proof.terminals)}, "
                "guard_rejected_terminals="
                f"{len(self.trajectory_proof.guard_rejected_terminals)}, "
                "unjustified_debt_terminals="
                f"{len(self.trajectory_proof.unjustified_debt_terminals)}, "
                f"dead_ends={len(self.trajectory_proof.dead_ends)}, "
                f"obstacles={len(self.trajectory_proof.obstacles)}"
            ),
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
        for obstacle in self.trajectory_proof.obstacles:
            lines.extend(obstacle.markdown_lines())
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
            "trajectory_proof": self.trajectory_proof.to_dict(),
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
            "finding_delta": self.finding_delta.to_dict(
                self.expected_removed_finding_ids
            ),
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
                ]
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
    """Prove virtual stages, then commit one revision-checked migration batch."""

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
    trajectory_budget: CodemodRefactorTrajectoryBudget = field(
        default_factory=CodemodRefactorTrajectoryBudget
    )

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

        if scan.scan_mode.exact:
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
        starting_scan = self.starting_scan()
        trajectory_proof = self.prove_trajectory(starting_scan)
        if not trajectory_proof.status.proved:
            return self.report(
                (),
                starting_scan,
                trajectory_proof.status.stop_reason,
                trajectory_proof,
            )
        terminal = trajectory_proof.proved_terminal
        if not terminal.stages:
            return self.report(
                (),
                terminal.state.scan,
                CodemodWorkflowStopReason.ACHIEVED,
                trajectory_proof,
            )
        return self.achieved_report(
            stages=terminal.stages,
            projected_scan=terminal.state.scan,
            starting_snapshot=starting_scan.source_snapshot,
            trajectory_proof=trajectory_proof,
        )

    def prove_trajectory(
        self,
        starting_scan: CodemodWorkflowScan,
    ) -> CodemodRefactorTrajectoryProof:
        """Exhaust exact reachable source states without ranking local moves."""

        initial_state = CodemodRefactorTrajectoryState(scan=starting_scan)
        pending = deque((initial_state,))
        visited_source_state_ids = {initial_state.source_state_id}
        terminals_by_source_state_id: dict[
            str,
            CodemodRefactorTrajectoryTerminal,
        ] = {}
        guard_rejected_terminals: list[CodemodRefactorGuardRejectedTerminal] = []
        unjustified_debt_terminals: list[CodemodRefactorUnjustifiedDebtTerminal] = []
        dead_ends: list[CodemodRefactorTrajectoryDeadEnd] = []
        obstacles: list[CodemodRefactorTrajectoryObstacle] = []
        transition_count = 0

        while pending:
            state = pending.popleft()
            active_scan = state.scan
            target_findings = self.target_findings(active_scan)
            if not target_findings:
                active_scan = self.exact_scan(active_scan)
                target_findings = self.target_findings(active_scan)
                state = replace(state, scan=active_scan)
                if not target_findings:
                    finding_class_delta = CodemodFindingClassDelta.from_findings(
                        tuple(starting_scan.findings),
                        tuple(active_scan.findings),
                    )
                    terminal_rejected = False
                    if finding_class_delta.increased_changes:
                        unjustified_debt_terminals.append(
                            CodemodRefactorUnjustifiedDebtTerminal(
                                state=state,
                                finding_class_changes=(
                                    finding_class_delta.increased_changes
                                ),
                            )
                        )
                        terminal_rejected = True
                    terminal_guard_report = (
                        self.guard_suite.clean_report()
                        if self.guard_suite.is_empty
                        else self.guard_suite.evaluate(
                            active_scan.source_index,
                            active_scan.sources_by_file_path,
                        )
                    )
                    if not terminal_guard_report.is_clean:
                        guard_rejected_terminals.append(
                            CodemodRefactorGuardRejectedTerminal(
                                state=state,
                                guard_report=terminal_guard_report,
                            )
                        )
                        terminal_rejected = True
                    if not terminal_rejected:
                        terminals_by_source_state_id[state.source_state_id] = (
                            CodemodRefactorTrajectoryTerminal(state)
                        )
                    continue

            snapshot = active_scan.source_snapshot
            plan = snapshot.plan_from_findings(
                target_findings,
                frontier_budget=self.trajectory_budget.recipe_frontier,
            )
            class_plan_report = FindingRecipeClassPlanReport.from_finding_plan(
                target_findings,
                root=self.class_plan_root(),
                finding_plan=plan,
            )
            frontier = plan.trajectory_frontier
            if not frontier.complete:
                obstacles.append(
                    CodemodRefactorRecipeFrontierObstacle(
                        source_state_id=state.source_state_id,
                        depth=state.depth,
                        recipe_obstacles=frontier.obstacles,
                    )
                )
                continue
            if not frontier.branches:
                dead_ends.append(
                    CodemodRefactorTrajectoryDeadEnd(
                        state=state,
                        class_plan_report=class_plan_report,
                    )
                )
                continue
            if state.depth >= self.trajectory_budget.max_depth:
                obstacles.append(
                    CodemodRefactorDepthBudgetObstacle(
                        source_state_id=state.source_state_id,
                        depth=state.depth,
                        max_depth=self.trajectory_budget.max_depth,
                    )
                )
                continue

            valid_transition_count = 0
            for branch in frontier.branches:
                simulation = branch.document_simulation
                if simulation.simulation.applied_rewrite_count == 0:
                    continue
                valid_transition_count += 1
                transition_count += 1
                projected_scan = self.projected_target_scan(
                    active_scan,
                    simulation.simulation,
                    target_findings,
                )
                if not self.target_findings(projected_scan):
                    projected_scan = self.exact_scan(projected_scan)
                stage = self.stage(
                    active_scan,
                    projected_scan,
                    class_plan_report=class_plan_report,
                    simulation=simulation,
                )
                next_state = CodemodRefactorTrajectoryState(
                    scan=projected_scan,
                    stages=(*state.stages, stage),
                )
                if next_state.source_state_id in visited_source_state_ids:
                    continue
                if len(visited_source_state_ids) >= self.trajectory_budget.max_states:
                    obstacles.append(
                        CodemodRefactorStateBudgetObstacle(
                            source_state_id=state.source_state_id,
                            depth=state.depth,
                            max_states=self.trajectory_budget.max_states,
                        )
                    )
                    continue
                visited_source_state_ids.add(next_state.source_state_id)
                pending.append(next_state)
            if valid_transition_count == 0:
                dead_ends.append(
                    CodemodRefactorTrajectoryDeadEnd(
                        state=state,
                        class_plan_report=class_plan_report,
                    )
                )

        return CodemodRefactorTrajectoryProof(
            initial_source_state_id=initial_state.source_state_id,
            budget=self.trajectory_budget,
            visited_state_count=len(visited_source_state_ids),
            transition_count=transition_count,
            terminals=tuple(
                terminals_by_source_state_id[source_state_id]
                for source_state_id in sorted(terminals_by_source_state_id)
            ),
            guard_rejected_terminals=tuple(guard_rejected_terminals),
            unjustified_debt_terminals=tuple(unjustified_debt_terminals),
            dead_ends=tuple(dead_ends),
            obstacles=tuple(obstacles),
        )

    def stages_with_terminal_guards(
        self,
        stages: tuple[CodemodRefactorGoalStage, ...],
        starting_snapshot: CodemodSourceSnapshot,
    ) -> tuple[CodemodRefactorGoalStage, ...] | None:
        """Attach caller guards only to the final state and replay the path."""

        if self.guard_suite.is_empty:
            return stages
        documents = tuple(stage.simulation.document for stage in stages)
        terminal_document = replace(
            documents[-1],
            guard_suite=documents[-1].guard_suite.merge(self.guard_suite),
        )
        sequence_simulation = CodemodPlanSequence(
            documents=(*documents[:-1], terminal_document),
        ).simulate_snapshot(starting_snapshot)
        if not sequence_simulation.is_clean:
            return None
        return tuple(
            replace(stage, simulation=stage_report.document_simulation)
            for stage, stage_report in zip(
                stages,
                sequence_simulation.stage_reports,
                strict=True,
            )
        )

    def achieved_report(
        self,
        *,
        stages: tuple[CodemodRefactorGoalStage, ...],
        projected_scan: CodemodWorkflowScan,
        starting_snapshot: CodemodSourceSnapshot,
        trajectory_proof: CodemodRefactorTrajectoryProof,
    ) -> CodemodRefactorGoalReport:
        """Commit a fully proved migration sequence once, or return its dry run."""

        guarded_stages = self.stages_with_terminal_guards(
            stages,
            starting_snapshot,
        )
        if guarded_stages is None:
            return self.report(
                (),
                projected_scan,
                CodemodWorkflowStopReason.ARCHITECTURE_GUARD_FAILED,
                trajectory_proof,
            )
        projected_report = self.report(
            guarded_stages,
            projected_scan,
            CodemodWorkflowStopReason.ACHIEVED,
            trajectory_proof,
        )
        if self.dry_run:
            return projected_report
        sequence_simulation = projected_report.replay_sequence.simulate_snapshot(
            starting_snapshot
        )
        if not sequence_simulation.is_clean:
            return replace(
                projected_report,
                stop_reason=CodemodWorkflowStopReason.ARCHITECTURE_GUARD_FAILED,
            )
        sequence_simulation.apply()
        committed_scan = self.fresh_scan()
        committed_target_findings = self.target_findings(committed_scan)
        applied_stages = (
            *(replace(stage, applied=True) for stage in guarded_stages[:-1]),
            guarded_stages[-1].with_applied_target_findings(committed_target_findings),
        )
        return self.report(
            applied_stages,
            committed_scan,
            (
                CodemodWorkflowStopReason.ACHIEVED
                if not committed_target_findings
                else CodemodWorkflowStopReason.APPLICATION_VERIFICATION_FAILED
            ),
            trajectory_proof,
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
                before_target_findings,
                after_target_findings,
            ),
        )

    def report(
        self,
        stages: tuple[CodemodRefactorGoalStage, ...],
        scan: CodemodWorkflowScan,
        reason: CodemodWorkflowStopReason,
        trajectory_proof: CodemodRefactorTrajectoryProof,
    ) -> CodemodRefactorGoalReport:
        verified_scan = self.exact_scan(scan)
        return CodemodRefactorGoalReport(
            migration_type=self.migration_type,
            stages=stages,
            stop_reason=reason,
            final_finding_count=len(verified_scan.findings),
            final_target_finding_ids=tuple(
                finding.stable_id for finding in self.target_findings(verified_scan)
            ),
            trajectory_proof=trajectory_proof,
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
