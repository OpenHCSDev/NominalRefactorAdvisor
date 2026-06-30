"""Reusable closed-loop workflows for executable codemod DSL plans."""

from __future__ import annotations

import ast
import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .analysis import analyze_detector_types, analyze_modules
from .ast_tools import ParsedModule, parse_python_module_roots
from .codemod import (
    ArchitectureGuardSuite,
    CodemodDslFieldKind,
    CodemodPlanDocument,
    CodemodPlanDocumentSimulation,
    CodemodPlanSequence,
    CodemodPlanSequenceContinuationReport,
    CodemodSimulationReport,
    CodemodSourceSnapshot,
    FindingRecipePlan,
    FindingRecipeSynthesisBoundary,
    FindingRecipeSynthesisReport,
    JsonObject,
    JsonValue,
    module_name_from_source_path,
)
from .detectors import DetectorConfig, IssueDetector
from .models import RefactorFinding
from .source_index import SourceIndex


class CodemodWorkflowStopReason(StrEnum):
    """Terminal state for staged codemod workflows."""

    ACHIEVED = "achieved"
    DRY_RUN = "dry_run"
    NO_TARGET_FINDINGS = "no_target_findings"
    NO_EXECUTABLE_RECIPES = "no_executable_recipes"
    EMPTY_REWRITE_BATCH = "empty_rewrite_batch"
    ARCHITECTURE_GUARD_FAILED = "architecture_guard_failed"
    NO_PROGRESS = "no_progress"
    MAX_ITERATIONS = "max_iterations"
    MAX_STAGES = "max_stages"


class CodemodRefactorGoalKind(StrEnum):
    """Supported high-level DSL refactor goals."""

    NOMINAL_BOUNDARY_EXTRACTION = "nominal_boundary_extraction"


class CodemodWorkflowPlanKind(StrEnum):
    """Top-level executable codemod workflow plans."""

    FIXPOINT = "fixpoint"
    REFACTOR_GOAL = "refactor_goal"


class CodemodFindingClassStatus(StrEnum):
    """Projected status for one semantic class of advisor findings."""

    ELIMINATED = "eliminated"
    MOVED = "moved"
    PARTIALLY_ELIMINATED = "partially_eliminated"
    PERSISTED = "persisted"
    INTRODUCED = "introduced"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class CodemodWorkflowPlanFieldManifest:
    """One JSON field accepted by an executable codemod workflow plan."""

    field_name: str
    value_kind: CodemodDslFieldKind
    required: bool
    description: str
    example_value: JsonValue

    def to_dict(self) -> JsonObject:
        return {
            "field_name": self.field_name,
            "value_kind": self.value_kind.value,
            "required": self.required,
            "description": self.description,
            "example_value": self.example_value,
        }


@dataclass(frozen=True)
class CodemodWorkflowPlanManifest:
    """Registry-derived schema and example for one workflow-plan DSL entry."""

    workflow: CodemodWorkflowPlanKind
    class_name: str
    description: str
    payload_fields: tuple[CodemodWorkflowPlanFieldManifest, ...]
    example_payload: JsonObject

    def to_dict(self) -> JsonObject:
        return {
            "workflow": self.workflow.value,
            "class_name": self.class_name,
            "description": self.description,
            "payload_fields": tuple(field.to_dict() for field in self.payload_fields),
            "example_payload": self.example_payload,
        }


@dataclass(frozen=True, kw_only=True)
class ParseCacheRequest:
    """Resolved parse-cache settings for reusable workflow scans."""

    resolved_dir: Path | None = None
    enabled: bool = False


@dataclass(frozen=True)
class CodemodFindingDelta:
    """Before/after finding ids for one simulated-and-applied codemod batch."""

    before_finding_ids: tuple[str, ...]
    after_finding_ids: tuple[str, ...]

    @classmethod
    def from_findings(
        cls,
        before_findings: tuple[RefactorFinding, ...],
        after_findings: tuple[RefactorFinding, ...],
    ) -> "CodemodFindingDelta":
        return cls(
            before_finding_ids=tuple(finding.stable_id for finding in before_findings),
            after_finding_ids=tuple(finding.stable_id for finding in after_findings),
        )

    @property
    def transition(self) -> "CodemodFindingIdTransition":
        return CodemodFindingIdTransition(
            before_ids=self.before_finding_ids,
            after_ids=self.after_finding_ids,
        )

    @property
    def removed_finding_ids(self) -> tuple[str, ...]:
        return self.transition.removed_ids

    @property
    def added_finding_ids(self) -> tuple[str, ...]:
        return self.transition.added_ids

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
        return {
            "before_finding_ids": self.before_finding_ids,
            "after_finding_ids": self.after_finding_ids,
            "expected_removed_finding_ids": expected_removed_finding_ids,
            "removed_finding_ids": self.removed_finding_ids,
            "added_finding_ids": self.added_finding_ids,
            "confirmed_expected_removed_finding_ids": (
                self.confirmed_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            "surviving_expected_removed_finding_ids": (
                self.surviving_expected_removed_finding_ids(
                    expected_removed_finding_ids
                )
            ),
            "removed_finding_count": self.removed_finding_count,
            "added_finding_count": self.added_finding_count,
            "confirmed_expected_removed_finding_count": (
                self.confirmed_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            "surviving_expected_removed_finding_count": (
                self.surviving_expected_removed_finding_count(
                    expected_removed_finding_ids
                )
            ),
            "fulfilled_expected_removals": self.fulfilled_expected_removals(
                expected_removed_finding_ids
            ),
        }


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


@dataclass(frozen=True)
class CodemodFindingIdTransition:
    """Before/after id transition shared by finding and finding-class deltas."""

    before_ids: tuple[str, ...]
    after_ids: tuple[str, ...]

    @property
    def removed_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_ids)
        return tuple(finding_id for finding_id in self.before_ids if finding_id not in after_ids)

    @property
    def added_ids(self) -> tuple[str, ...]:
        before_ids = frozenset(self.before_ids)
        return tuple(finding_id for finding_id in self.after_ids if finding_id not in before_ids)

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
class CodemodFindingClassChange:
    """Before/after membership for one semantic finding class."""

    signature: CodemodFindingClassSignature
    finding_ids: CodemodFindingIdTransition
    expected_removed_finding_ids: tuple[str, ...] = ()

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
    def before_finding_count(self) -> int:
        return self.finding_ids.before_count

    @property
    def after_finding_count(self) -> int:
        return self.finding_ids.after_count

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    @property
    def removed_finding_count(self) -> int:
        return self.finding_ids.removed_count

    @property
    def added_finding_count(self) -> int:
        return self.finding_ids.added_count

    def to_dict(self) -> JsonObject:
        return {
            "signature": self.signature.to_dict(),
            "status": self.status.value,
            "before_finding_ids": self.before_finding_ids,
            "after_finding_ids": self.after_finding_ids,
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "removed_finding_ids": self.removed_finding_ids,
            "added_finding_ids": self.added_finding_ids,
            "before_finding_count": self.before_finding_count,
            "after_finding_count": self.after_finding_count,
            "expected_removed_finding_count": self.expected_removed_finding_count,
            "removed_finding_count": self.removed_finding_count,
            "added_finding_count": self.added_finding_count,
        }


@dataclass(frozen=True)
class CodemodFindingClassBuckets:
    """Findings grouped by semantic class signature without fallback mutation."""

    findings_by_signature: Mapping[
        CodemodFindingClassSignature,
        tuple[RefactorFinding, ...],
    ]

    @classmethod
    def from_findings(
        cls,
        findings: tuple[RefactorFinding, ...],
    ) -> "CodemodFindingClassBuckets":
        signatures = tuple(
            dict.fromkeys(
                CodemodFindingClassSignature.from_finding(finding)
                for finding in findings
            )
        )
        return cls(
            findings_by_signature={
                signature: tuple(
                    finding
                    for finding in findings
                    if CodemodFindingClassSignature.from_finding(finding) == signature
                )
                for signature in signatures
            }
        )

    @property
    def signatures(self) -> tuple[CodemodFindingClassSignature, ...]:
        return tuple(
            sorted(
                self.findings_by_signature,
                key=lambda signature: signature.class_key,
            )
        )

    def findings_for(
        self,
        signature: CodemodFindingClassSignature,
    ) -> tuple[RefactorFinding, ...]:
        if signature not in self.findings_by_signature:
            return ()
        return self.findings_by_signature[signature]


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
        before_buckets = CodemodFindingClassBuckets.from_findings(before_findings)
        after_buckets = CodemodFindingClassBuckets.from_findings(after_findings)
        signatures = tuple(
            sorted(
                set(before_buckets.signatures) | set(after_buckets.signatures),
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
                            for finding in before_buckets.findings_for(signature)
                        ),
                        after_ids=tuple(
                            finding.stable_id
                            for finding in after_buckets.findings_for(signature)
                        ),
                    ),
                    expected_removed_finding_ids=tuple(
                        finding.stable_id
                        for finding in before_buckets.findings_for(signature)
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
        return {
            status.value: self.count_status(status)
            for status in CodemodFindingClassStatus
            if self.count_status(status)
        }

    def to_dict(self) -> JsonObject:
        return {
            "class_change_count": self.change_count,
            "moved_class_count": self.moved_class_count,
            "eliminated_class_count": self.eliminated_class_count,
            "status_counts": self.status_counts(),
            "changes": tuple(change.to_dict() for change in self.changes),
        }


@dataclass(frozen=True)
class CodemodFindingChangeProjection:
    """Expected and observed finding changes for one codemod workflow stage."""

    expected_removed_finding_ids: tuple[str, ...] = ()
    finding_delta: CodemodFindingDelta | None = None

    @property
    def expected_removed_finding_count(self) -> int:
        return len(self.expected_removed_finding_ids)

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
        }
        if self.finding_delta is not None:
            payload["finding_delta"] = self.finding_delta.to_dict(
                self.expected_removed_finding_ids
            )
        return payload


@dataclass(frozen=True)
class CodemodFindingChangeCarrier:
    """Mixin for workflow payloads that expose expected and observed changes."""

    finding_change: CodemodFindingChangeProjection

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return self.finding_change.expected_removed_finding_ids

    @property
    def expected_removed_finding_count(self) -> int:
        return self.finding_change.expected_removed_finding_count

    @property
    def finding_delta(self) -> CodemodFindingDelta | None:
        return self.finding_change.finding_delta


@dataclass(frozen=True)
class CodemodRefactorGoal:
    """Declarative target for staged semantic-fact extraction refactors."""

    goal_id: str
    kind: CodemodRefactorGoalKind = CodemodRefactorGoalKind.NOMINAL_BOUNDARY_EXTRACTION
    target_finding_ids: tuple[str, ...] = ()
    detector_ids: tuple[str, ...] = ()
    pattern_ids: tuple[int, ...] = ()
    max_stages: int = 8

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "CodemodRefactorGoal":
        """Parse a reusable goal declaration from codemod workflow-plan JSON."""

        payload = CodemodWorkflowPlanJsonParser.object_payload(
            value,
            "codemod refactor goal",
        )
        return cls(
            goal_id=CodemodWorkflowPlanJsonParser.required_string_field(
                payload,
                "goal_id",
            ),
            kind=CodemodRefactorGoalKind(
                CodemodWorkflowPlanJsonParser.required_string_field(
                    payload,
                    "kind",
                )
            ),
            target_finding_ids=CodemodWorkflowPlanJsonParser.string_tuple_field(
                payload,
                "target_finding_ids",
            ),
            detector_ids=CodemodWorkflowPlanJsonParser.string_tuple_field(
                payload,
                "detector_ids",
            ),
            pattern_ids=CodemodWorkflowPlanJsonParser.integer_tuple_field(
                payload,
                "pattern_ids",
            ),
            max_stages=CodemodWorkflowPlanJsonParser.required_integer_field(
                payload,
                "max_stages",
            ),
        )

    @property
    def has_explicit_targets(self) -> bool:
        return bool(self.target_finding_ids or self.detector_ids or self.pattern_ids)

    def to_dict(self) -> JsonObject:
        return {
            "goal_id": self.goal_id,
            "kind": self.kind.value,
            "target_finding_ids": self.target_finding_ids,
            "detector_ids": self.detector_ids,
            "pattern_ids": self.pattern_ids,
            "max_stages": self.max_stages,
        }


class CodemodRefactorGoalTargetPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered target-selection policy for one refactor goal kind."""

    __registry__: ClassVar[
        dict[CodemodRefactorGoalKind, type["CodemodRefactorGoalTargetPolicy"]]
    ] = {}
    __registry_key__ = "goal_kind"
    __skip_if_no_key__ = True

    goal_kind: ClassVar[CodemodRefactorGoalKind | None] = None

    @classmethod
    def policy_for(
        cls,
        goal_kind: CodemodRefactorGoalKind,
    ) -> "CodemodRefactorGoalTargetPolicy":
        try:
            return cls.__registry__[goal_kind]()
        except KeyError as error:
            raise ValueError(
                f"unsupported codemod refactor goal kind: {goal_kind}"
            ) from error

    def target_findings(
        self,
        goal: CodemodRefactorGoal,
        findings: Iterable[RefactorFinding],
    ) -> tuple[RefactorFinding, ...]:
        return tuple(
            finding for finding in findings if self.matches_finding(goal, finding)
        )

    def matches_finding(
        self,
        goal: CodemodRefactorGoal,
        finding: RefactorFinding,
    ) -> bool:
        if goal.has_explicit_targets:
            return self.explicit_target_matches_finding(goal, finding)
        return self.default_target_matches_finding(finding)

    @staticmethod
    def explicit_target_matches_finding(
        goal: CodemodRefactorGoal,
        finding: RefactorFinding,
    ) -> bool:
        if finding.stable_id in goal.target_finding_ids:
            return True
        if finding.detector_id in goal.detector_ids:
            return True
        return int(finding.pattern_id) in goal.pattern_ids

    @abstractmethod
    def default_target_matches_finding(self, finding: RefactorFinding) -> bool:
        raise NotImplementedError


class NominalBoundaryExtractionGoalTargetPolicy(CodemodRefactorGoalTargetPolicy):
    """Target SSOT authority-boundary findings for nominal-boundary goals."""

    goal_kind = CodemodRefactorGoalKind.NOMINAL_BOUNDARY_EXTRACTION

    @staticmethod
    def default_detector_ids() -> frozenset[str]:
        from .detectors import IssueDetector

        return IssueDetector.ssot_authority_detector_ids()

    def default_target_matches_finding(self, finding: RefactorFinding) -> bool:
        return finding.detector_id in self.default_detector_ids()


@dataclass(frozen=True)
class CodemodWorkflowPlan(ABC, metaclass=AutoRegisterMeta):
    """Reusable JSON/API plan for closed-loop codemod DSL execution."""

    __registry__: ClassVar[
        dict[CodemodWorkflowPlanKind, type["CodemodWorkflowPlan"]]
    ] = {}
    __registry_key__ = "workflow"
    __skip_if_no_key__ = True
    workflow: ClassVar[CodemodWorkflowPlanKind | None] = None

    plan_id: str

    @classmethod
    def workflow_key(cls) -> CodemodWorkflowPlanKind:
        if cls.workflow is None:
            raise ValueError(f"{cls.__name__} has no workflow key")
        return cls.workflow

    @classmethod
    def workflow_type(
        cls,
        workflow: CodemodWorkflowPlanKind,
    ) -> type["CodemodWorkflowPlan"]:
        try:
            return cls.__registry__[workflow]
        except KeyError as error:
            raise ValueError(
                f"unsupported codemod workflow plan kind: {workflow}"
            ) from error

    @classmethod
    def common_payload_field_manifests(
        cls,
    ) -> tuple[CodemodWorkflowPlanFieldManifest, ...]:
        return (
            CodemodWorkflowPlanFieldManifest(
                field_name="workflow",
                value_kind=CodemodDslFieldKind.STRING,
                required=True,
                description="Registered executable workflow-plan kind.",
                example_value=cls.workflow_key().value,
            ),
            CodemodWorkflowPlanFieldManifest(
                field_name="plan_id",
                value_kind=CodemodDslFieldKind.STRING,
                required=True,
                description="Stable caller-chosen identifier for this workflow run.",
                example_value=f"{cls.workflow_key().value}-plan",
            ),
        )

    @classmethod
    def workflow_manifest(cls) -> CodemodWorkflowPlanManifest:
        return CodemodWorkflowPlanManifest(
            workflow=cls.workflow_key(),
            class_name=cls.__name__,
            description=codemod_workflow_entry_description(cls),
            payload_fields=cls.payload_field_manifests(),
            example_payload=cls.example_payload(),
        )

    @classmethod
    @abstractmethod
    def payload_field_manifests(
        cls,
    ) -> tuple[CodemodWorkflowPlanFieldManifest, ...]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def example_payload(cls) -> JsonObject:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_payload(
        cls,
        payload: JsonObject,
        parser: "CodemodWorkflowPlanJsonParser",
    ) -> "CodemodWorkflowPlan":
        raise NotImplementedError

    @property
    def kind(self) -> CodemodWorkflowPlanKind:
        if self.workflow is None:
            raise ValueError(f"{type(self).__name__} has no workflow key")
        return self.workflow

    @abstractmethod
    def run(
        self,
        *,
        resolved_dir: Path | None,
        enabled: bool,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        parse_workers: int,
        dry_run: bool,
        guard_suite: ArchitectureGuardSuite,
        initial_scan: "CodemodFixpointScan | None" = None,
    ) -> "CodemodWorkflowReport":
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> JsonObject:
        raise NotImplementedError


@dataclass(frozen=True)
class CodemodFixpointWorkflowPlan(CodemodWorkflowPlan):
    """Workflow plan that repeatedly applies finding-backed recipes to a fixpoint."""

    workflow: ClassVar[CodemodWorkflowPlanKind] = CodemodWorkflowPlanKind.FIXPOINT
    max_iterations: int = 8

    @classmethod
    def payload_field_manifests(
        cls,
    ) -> tuple[CodemodWorkflowPlanFieldManifest, ...]:
        return (
            *cls.common_payload_field_manifests(),
            CodemodWorkflowPlanFieldManifest(
                field_name="max_iterations",
                value_kind=CodemodDslFieldKind.INTEGER,
                required=True,
                description="Maximum synthesize/apply/rescan cycles.",
                example_value=8,
            ),
        )

    @classmethod
    def example_payload(cls) -> JsonObject:
        return cls(
            plan_id="finding-backed-fixpoint",
            max_iterations=8,
        ).to_dict()

    @classmethod
    def from_payload(
        cls,
        payload: JsonObject,
        parser: "CodemodWorkflowPlanJsonParser",
    ) -> "CodemodFixpointWorkflowPlan":
        return cls(
            plan_id=parser.required_string_field(
                payload,
                parser.plan_id_field,
            ),
            max_iterations=parser.required_integer_field(
                payload,
                parser.max_iterations_field,
            ),
        )

    def run(
        self,
        *,
        resolved_dir: Path | None,
        enabled: bool,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        parse_workers: int,
        dry_run: bool,
        guard_suite: ArchitectureGuardSuite,
        initial_scan: "CodemodFixpointScan | None" = None,
    ) -> "CodemodWorkflowReport":
        return CodemodFixpointRunner(
            resolved_dir=resolved_dir,
            enabled=enabled,
            roots=roots,
            config=config,
            parse_workers=parse_workers,
            dry_run=dry_run,
            initial_scan=initial_scan,
            guard_suite=guard_suite,
            max_iterations=self.max_iterations,
        ).run()

    def to_dict(self) -> JsonObject:
        return {
            "workflow": self.kind.value,
            "plan_id": self.plan_id,
            "max_iterations": self.max_iterations,
        }


@dataclass(frozen=True)
class CodemodRefactorGoalWorkflowPlan(CodemodWorkflowPlan):
    """Workflow plan that drives a declarative refactor goal through staged recipes."""

    workflow: ClassVar[CodemodWorkflowPlanKind] = CodemodWorkflowPlanKind.REFACTOR_GOAL
    example_goal: ClassVar[CodemodRefactorGoal] = CodemodRefactorGoal(
        goal_id="nominal-boundary-goal"
    )
    goal: CodemodRefactorGoal

    @classmethod
    def payload_field_manifests(
        cls,
    ) -> tuple[CodemodWorkflowPlanFieldManifest, ...]:
        return (
            *cls.common_payload_field_manifests(),
            CodemodWorkflowPlanFieldManifest(
                field_name="goal",
                value_kind=CodemodDslFieldKind.OBJECT,
                required=True,
                description="Declarative semantic refactor goal to pursue.",
                example_value=cls.example_goal.to_dict(),
            ),
        )

    @classmethod
    def example_payload(cls) -> JsonObject:
        return cls(
            plan_id=cls.example_goal.goal_id,
            goal=cls.example_goal,
        ).to_dict()

    @classmethod
    def from_payload(
        cls,
        payload: JsonObject,
        parser: "CodemodWorkflowPlanJsonParser",
    ) -> "CodemodRefactorGoalWorkflowPlan":
        goal = CodemodRefactorGoal.from_json_value(
            parser.required_field(payload, parser.goal_field)
        )
        return cls(
            plan_id=parser.required_string_field(
                payload,
                parser.plan_id_field,
            ),
            goal=goal,
        )

    def run(
        self,
        *,
        resolved_dir: Path | None,
        enabled: bool,
        roots: tuple[Path, ...],
        config: DetectorConfig,
        parse_workers: int,
        dry_run: bool,
        guard_suite: ArchitectureGuardSuite,
        initial_scan: "CodemodFixpointScan | None" = None,
    ) -> "CodemodWorkflowReport":
        return CodemodRefactorGoalRunner(
            resolved_dir=resolved_dir,
            enabled=enabled,
            roots=roots,
            config=config,
            parse_workers=parse_workers,
            dry_run=dry_run,
            initial_scan=initial_scan,
            guard_suite=guard_suite,
            goal=self.goal,
        ).run()

    def to_dict(self) -> JsonObject:
        return {
            "workflow": self.kind.value,
            "plan_id": self.plan_id,
            "goal": self.goal.to_dict(),
        }


@dataclass(frozen=True)
class CodemodWorkflowPlanJsonParser:
    """Decode reusable workflow-plan JSON into nominal codemod workflow records."""

    workflow_field: str = "workflow"
    plan_id_field: str = "plan_id"
    max_iterations_field: str = "max_iterations"
    goal_field: str = "goal"

    def parse_plan(self, value: JsonValue) -> CodemodWorkflowPlan:
        payload = self.object_payload(value, "codemod workflow plan")
        kind = CodemodWorkflowPlanKind(
            self.required_string_field(
                payload,
                self.workflow_field,
            )
        )
        return CodemodWorkflowPlan.workflow_type(kind).from_payload(payload, self)

    @staticmethod
    def object_payload(value: JsonValue, label: str) -> JsonObject:
        if isinstance(value, dict):
            return JsonObject(value)
        raise ValueError(f"{label} must be a JSON object")

    @staticmethod
    def required_field(payload: JsonObject, field_name: str) -> JsonValue:
        if field_name in payload:
            return payload[field_name]
        raise ValueError(f"codemod workflow plan requires `{field_name}`")

    @classmethod
    def required_string_field(
        cls,
        payload: JsonObject,
        field_name: str,
    ) -> str:
        value = cls.required_field(payload, field_name)
        if isinstance(value, str):
            return value
        raise ValueError(f"`{field_name}` must be a string")

    @classmethod
    def required_integer_field(
        cls,
        payload: JsonObject,
        field_name: str,
    ) -> int:
        value = cls.required_field(payload, field_name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError(f"`{field_name}` must be an integer")

    @classmethod
    def string_tuple_field(
        cls,
        payload: JsonObject,
        field_name: str,
    ) -> tuple[str, ...]:
        return tuple(cls.array_items(payload, field_name, str))

    @classmethod
    def integer_tuple_field(
        cls,
        payload: JsonObject,
        field_name: str,
    ) -> tuple[int, ...]:
        values = []
        for value in cls.array_items(payload, field_name, int):
            if isinstance(value, bool):
                raise ValueError(f"`{field_name}` entries must be integers")
            values.append(value)
        return tuple(values)

    @staticmethod
    def array_items(
        payload: JsonObject,
        field_name: str,
        item_type: type[str] | type[int],
    ) -> tuple[str, ...] | tuple[int, ...]:
        if field_name not in payload:
            return ()
        value = payload[field_name]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"`{field_name}` must be an array")
        if not all(isinstance(item, item_type) for item in value):
            raise ValueError(f"`{field_name}` entries must be {item_type.__name__}")
        return tuple(value)


def codemod_workflow_entry_description(
    entry_type: type[CodemodWorkflowPlan],
) -> str:
    """Return a normalized semantic description for one workflow-plan entry."""

    description = inspect.getdoc(entry_type)
    if description is None:
        raise ValueError(f"{entry_type.__name__} must define a workflow description")
    return description


def codemod_workflow_plan_manifests() -> tuple[CodemodWorkflowPlanManifest, ...]:
    """Return registry-derived manifest rows for executable workflow plans."""

    return tuple(
        CodemodWorkflowPlan.workflow_type(kind).workflow_manifest()
        for kind in CodemodWorkflowPlanKind
    )


def codemod_workflow_plan_example_payloads() -> tuple[JsonObject, ...]:
    """Return parseable starter payloads for executable workflow plans."""

    return tuple(
        manifest.example_payload for manifest in codemod_workflow_plan_manifests()
    )


@dataclass(frozen=True)
class CodemodRefactorGoalProgress:
    """Before/after target-finding progress for one goal stage."""

    before_target_finding_ids: tuple[str, ...]
    after_target_finding_ids: tuple[str, ...]

    @classmethod
    def from_findings(
        cls,
        goal: CodemodRefactorGoal,
        before_findings: Iterable[RefactorFinding],
        after_findings: Iterable[RefactorFinding],
    ) -> "CodemodRefactorGoalProgress":
        target_policy = CodemodRefactorGoalTargetPolicy.policy_for(goal.kind)
        return cls(
            before_target_finding_ids=tuple(
                finding.stable_id
                for finding in target_policy.target_findings(goal, before_findings)
            ),
            after_target_finding_ids=tuple(
                finding.stable_id
                for finding in target_policy.target_findings(goal, after_findings)
            ),
        )

    @property
    def removed_target_finding_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_target_finding_ids)
        return tuple(
            finding_id
            for finding_id in self.before_target_finding_ids
            if finding_id not in after_ids
        )

    @property
    def surviving_target_finding_ids(self) -> tuple[str, ...]:
        after_ids = frozenset(self.after_target_finding_ids)
        return tuple(
            finding_id
            for finding_id in self.before_target_finding_ids
            if finding_id in after_ids
        )

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
class CodemodRefactorGoalStage(
    CodemodFindingChangeCarrier,
    FindingRecipeSynthesisBoundary,
):
    """One simulated or applied staged plan toward a refactor goal."""

    stage_index: int
    document: CodemodPlanDocument
    simulation: CodemodPlanDocumentSimulation
    progress: CodemodRefactorGoalProgress
    applied: bool = False

    @property
    def rewrite_count(self) -> int:
        return self.simulation.simulation.applied_rewrite_count

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return self.simulation.simulation.changed_file_paths

    def to_dict(self) -> JsonObject:
        return {
            "stage_index": self.stage_index,
            "applied": self.applied,
            "rewrite_count": self.rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "document": self.document.to_dict(),
            "simulation": self.simulation.to_dict(),
            "progress": self.progress.to_dict(),
            **self.finding_change.to_dict(),
            **self.synthesis_payload(),
        }


@dataclass(frozen=True)
class CodemodRefactorGoalStageAttempt(FindingRecipeSynthesisBoundary):
    """Synthesis attempt for one goal stage, including no-stage diagnostics."""

    stage_index: int
    target_finding_count: int
    stage: CodemodRefactorGoalStage | None = None
    projected_scan: "CodemodFixpointScan | None" = None

    @property
    def has_stage(self) -> bool:
        return self.stage is not None

    def to_dict(self) -> JsonObject:
        payload = {
            "stage_index": self.stage_index,
            "target_finding_count": self.target_finding_count,
            "has_stage": self.has_stage,
            **self.synthesis_payload(),
        }
        if self.stage is not None:
            payload["stage"] = self.stage.to_dict()
        return payload


@dataclass(frozen=True)
class CodemodWorkflowReport:
    """Shared terminal summary for staged codemod workflow reports."""

    completed: bool
    terminal_reason: CodemodWorkflowStopReason
    final_finding_count: int

    @property
    def stop_reason(self) -> CodemodWorkflowStopReason:
        return self.terminal_reason

    @property
    def replay_sequence(self) -> CodemodPlanSequence:
        return CodemodPlanSequence()

    def to_markdown(self) -> str:
        return "\n".join(
            (
                "Codemod workflow report:",
                f"   - Completed: {self.completed}",
                f"   - Stop reason: {self.stop_reason.value}",
                f"   - Final findings: {self.final_finding_count}",
            )
        )


@dataclass(frozen=True)
class CodemodRefactorGoalReport(CodemodWorkflowReport):
    """Machine-readable result of a goal-directed staged codemod run."""

    goal: CodemodRefactorGoal
    stages: tuple[CodemodRefactorGoalStage, ...]
    final_target_finding_ids: tuple[str, ...]
    terminal_synthesis_report: FindingRecipeSynthesisReport = field(
        default_factory=FindingRecipeSynthesisReport
    )

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def total_rewrite_count(self) -> int:
        return sum(stage.rewrite_count for stage in self.stages)

    @property
    def replay_sequence(self) -> CodemodPlanSequence:
        return CodemodPlanSequence(
            documents=tuple(stage.document for stage in self.stages)
        )

    @property
    def achieved(self) -> bool:
        return self.completed and not self.final_target_finding_ids

    def to_markdown(self) -> str:
        lines = [
            "Codemod refactor goal report:",
            f"   - Goal: {self.goal.goal_id} ({self.goal.kind.value})",
            f"   - Completed: {self.completed}",
            f"   - Achieved: {self.achieved}",
            f"   - Stop reason: {self.terminal_reason.value}",
            f"   - Stages: {self.stage_count}",
            f"   - Rewrites: {self.total_rewrite_count}",
            f"   - Final findings: {self.final_finding_count}",
            f"   - Remaining target findings: {len(self.final_target_finding_ids)}",
        ]
        if self.terminal_synthesis_report.records:
            status_counts = self.terminal_synthesis_report.to_dict()["status_counts"]
            lines.extend(
                (
                    "   - Terminal synthesis: "
                    f"planned={self.terminal_synthesis_report.planned_count}, "
                    f"rejected={self.terminal_synthesis_report.rejected_count}, "
                    f"unsupported={self.terminal_synthesis_report.unsupported_count}, "
                    f"statuses={status_counts}",
                )
            )
        for stage in self.stages:
            lines.append(
                "   - "
                f"Stage {stage.stage_index}: "
                f"rewrites={stage.rewrite_count}, "
                f"removed_targets={stage.progress.removed_target_finding_count}, "
                f"surviving_targets={stage.progress.surviving_target_finding_count}, "
                f"applied={stage.applied}"
            )
        return "\n".join(lines)

    def to_dict(self) -> JsonObject:
        return {
            "goal": self.goal.to_dict(),
            "completed": self.completed,
            "achieved": self.achieved,
            "terminal_reason": self.terminal_reason.value,
            "stage_count": self.stage_count,
            "total_rewrite_count": self.total_rewrite_count,
            "final_finding_count": self.final_finding_count,
            "final_target_finding_ids": self.final_target_finding_ids,
            "terminal_synthesis_report": self.terminal_synthesis_report.to_dict(),
            "replay_sequence": self.replay_sequence.to_dict(),
            "stages": tuple(stage.to_dict() for stage in self.stages),
        }


@dataclass(frozen=True)
class CodemodProjectedFindingReport:
    """Before/after advisor findings for one simulated codemod source state."""

    before_findings: tuple[RefactorFinding, ...]
    after_scan: "CodemodFixpointScan"
    source_sequence: CodemodPlanSequence | None = None
    expected_removed_finding_ids: tuple[str, ...] = ()

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

    @property
    def finding_delta(self) -> CodemodFindingDelta:
        return CodemodFindingDelta.from_findings(
            self.before_findings,
            self.after_findings,
        )

    @property
    def finding_class_delta(self) -> CodemodFindingClassDelta:
        return CodemodFindingClassDelta.from_findings(
            self.before_findings,
            self.after_findings,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
        )

    @property
    def continuation_report(self) -> CodemodPlanSequenceContinuationReport:
        projected_snapshot = self.after_scan.source_snapshot
        after_findings = self.after_findings
        return CodemodPlanSequenceContinuationReport(
            sequence=self.source_sequence or CodemodPlanSequence(),
            source_index=projected_snapshot.source_index,
            findings=after_findings,
            plan=projected_snapshot.plan_from_findings(after_findings),
        )

    def to_dict(self) -> JsonObject:
        after_findings = self.after_findings
        projected_snapshot = self.after_scan.source_snapshot
        continuation_report = self.continuation_report
        return {
            "before_finding_count": self.before_finding_count,
            "after_finding_count": self.after_finding_count,
            "finding_delta": self.finding_delta.to_dict(),
            "finding_class_delta": self.finding_class_delta.to_dict(),
            "after_findings": tuple(finding.to_dict() for finding in after_findings),
            "projected_source_index": projected_snapshot.source_index.to_dict(),
            "projected_finding_recipe_plan": continuation_report.plan.to_dict(),
            "projected_finding_continuation": continuation_report.to_dict(),
        }


@dataclass(frozen=True)
class CodemodFixpointScan:
    """Parsed source snapshot used by one fixpoint iteration."""

    modules: list[ParsedModule]
    findings: list[RefactorFinding]

    @property
    def source_index(self) -> SourceIndex:
        return self.source_snapshot.source_index

    @property
    def sources_by_file_path(self) -> dict[str, str]:
        return dict(self.source_snapshot.sources_by_file_path)

    @cached_property
    def source_snapshot(self) -> CodemodSourceSnapshot:
        return CodemodSourceSnapshot.from_modules(self.modules, self.findings)


@dataclass(frozen=True, kw_only=True)
class CodemodWorkflowScanRequest(ParseCacheRequest):
    """Shared scan/projection substrate for staged codemod workflows."""

    roots: tuple[Path, ...]
    config: DetectorConfig
    parse_workers: int
    dry_run: bool
    initial_scan: CodemodFixpointScan | None = None

    def scan(self, stage_index: int) -> CodemodFixpointScan:
        if stage_index == 0 and self.initial_scan is not None:
            return self.initial_scan
        modules = parse_python_module_roots(
            self.roots,
            cache_dir=self.resolved_dir,
            use_parse_cache=self.enabled,
            parse_workers=self.parse_workers,
        )
        return CodemodFixpointScan(
            modules=modules,
            findings=analyze_modules(modules, self.config),
        )

    def projected_scan(
        self,
        scan: CodemodFixpointScan,
        simulation: CodemodSimulationReport,
    ) -> CodemodFixpointScan:
        modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodFixpointScan(
            modules=list(modules),
            findings=analyze_modules(modules, self.config),
        )


@dataclass(frozen=True, kw_only=True)
class CodemodGuardedWorkflowRequest(CodemodWorkflowScanRequest):
    """Workflow request with architecture guards for synthesized codemod plans."""

    guard_suite: ArchitectureGuardSuite


@dataclass(frozen=True)
class CodemodSimulationFindingProjection:
    """Analyze advisor findings after applying a simulation in memory."""

    modules: tuple[ParsedModule, ...]
    findings: tuple[RefactorFinding, ...]
    simulation: CodemodSimulationReport
    config: DetectorConfig
    roots: tuple[Path, ...] = ()
    source_sequence: CodemodPlanSequence | None = None
    expected_removed_finding_ids: tuple[str, ...] = ()

    def scan(self) -> CodemodFixpointScan:
        projected_modules = ProjectedScanModuleSet(
            modules=self.modules,
            simulation=self.simulation,
            roots=self.roots,
        ).modules_after_projection()
        return CodemodFixpointScan(
            modules=list(projected_modules),
            findings=analyze_modules(projected_modules, self.config),
        )

    def report(self) -> CodemodProjectedFindingReport:
        after_scan = self.scan()
        return CodemodProjectedFindingReport(
            before_findings=self.findings,
            after_scan=after_scan,
            source_sequence=self.source_sequence,
            expected_removed_finding_ids=self.expected_removed_finding_ids,
        )


@dataclass(frozen=True)
class CodemodFixpointIteration(
    CodemodFindingChangeCarrier,
    FindingRecipeSynthesisBoundary,
):
    """One scan/simulate/apply/rescan step in the codemod fixpoint workflow."""

    iteration_index: int
    finding_count: int
    recipe_count: int
    document: CodemodPlanDocument | None = None
    simulation_result: CodemodPlanDocumentSimulation | None = None
    applied: bool = False
    stop_reason: CodemodWorkflowStopReason | None = None

    @property
    def applied_rewrite_count(self) -> int:
        if self.simulation_result is None or not self.applied:
            return 0
        return self.simulation_result.simulation.applied_rewrite_count

    @property
    def simulated_rewrite_count(self) -> int:
        if self.simulation_result is None:
            return 0
        return self.simulation_result.simulation.applied_rewrite_count

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        if self.simulation_result is None:
            return ()
        return self.simulation_result.simulation.changed_file_paths

    @property
    def is_clean(self) -> bool:
        if self.simulation_result is None:
            return True
        return self.simulation_result.is_clean

    @property
    def stop_label(self) -> str:
        if self.stop_reason is None:
            return "continue"
        return self.stop_reason.value

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "iteration_index": self.iteration_index,
            "finding_count": self.finding_count,
            "recipe_count": self.recipe_count,
            "expected_removed_finding_ids": self.expected_removed_finding_ids,
            "expected_removed_finding_count": self.expected_removed_finding_count,
            "applied": self.applied,
            "applied_rewrite_count": self.applied_rewrite_count,
            "simulated_rewrite_count": self.simulated_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "is_clean": self.is_clean,
            "stop_reason": (
                None if self.stop_reason is None else self.stop_reason.value
            ),
        }
        if self.document is not None:
            payload["document"] = self.document.to_dict()
        payload.update(self.synthesis_payload())
        if self.simulation_result is not None:
            payload["simulation"] = self.simulation_result.simulation.to_dict()
            payload["architecture_guard_report"] = (
                self.simulation_result.architecture_guard_report.to_dict()
            )
        if self.finding_delta is not None:
            payload["finding_delta"] = self.finding_delta.to_dict(
                self.expected_removed_finding_ids
            )
        return payload


@dataclass(frozen=True)
class CodemodFixpointReplayPlan:
    """Runnable staged DSL plan assembled from successful fixpoint iterations."""

    iterations: tuple[CodemodFixpointIteration, ...]

    @property
    def documents(self) -> tuple[CodemodPlanDocument, ...]:
        return tuple(
            iteration.document
            for iteration in self.iterations
            if iteration.stop_reason is None
            and iteration.document is not None
            and iteration.document.has_recipes
        )

    @property
    def sequence(self) -> CodemodPlanSequence:
        return CodemodPlanSequence(documents=self.documents)

    @property
    def stage_count(self) -> int:
        return len(self.documents)

    @property
    def has_stages(self) -> bool:
        return self.stage_count > 0

    def to_dict(self) -> JsonObject:
        return {
            "stage_count": self.stage_count,
            "has_stages": self.has_stages,
            "sequence": self.sequence.to_dict(),
        }


@dataclass(frozen=True)
class CodemodFixpointReport(CodemodWorkflowReport):
    """Machine-readable result of an iterative DSL codemod workflow."""

    iterations: tuple[CodemodFixpointIteration, ...]

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def applied(self) -> bool:
        return any(iteration.applied for iteration in self.iterations)

    @property
    def total_applied_rewrite_count(self) -> int:
        return sum(iteration.applied_rewrite_count for iteration in self.iterations)

    @property
    def total_simulated_rewrite_count(self) -> int:
        return sum(iteration.simulated_rewrite_count for iteration in self.iterations)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    file_path
                    for iteration in self.iterations
                    for file_path in iteration.changed_file_paths
                    if iteration.applied
                }
            )
        )

    @property
    def simulated_changed_file_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    file_path
                    for iteration in self.iterations
                    for file_path in iteration.changed_file_paths
                }
            )
        )

    @property
    def replay_plan(self) -> CodemodFixpointReplayPlan:
        return CodemodFixpointReplayPlan(iterations=self.iterations)

    @property
    def replay_sequence(self) -> CodemodPlanSequence:
        return self.replay_plan.sequence

    def to_markdown(self) -> str:
        lines = [
            "Codemod fixpoint report:",
            f"   - Completed: {self.completed}",
            f"   - Stop reason: {self.stop_reason.value}",
            f"   - Iterations: {self.iteration_count}",
            f"   - Applied rewrites: {self.total_applied_rewrite_count}",
            f"   - Simulated rewrites: {self.total_simulated_rewrite_count}",
            f"   - Changed files: {len(self.changed_file_paths)}",
            f"   - Simulated changed files: {len(self.simulated_changed_file_paths)}",
            f"   - Final findings: {self.final_finding_count}",
        ]
        for iteration in self.iterations:
            lines.append(
                "   - "
                f"Iteration {iteration.iteration_index}: "
                f"recipes={iteration.recipe_count}, "
                f"expected_removed={iteration.expected_removed_finding_count}, "
                f"rewrites={iteration.applied_rewrite_count}, "
                f"simulated={iteration.simulated_rewrite_count}, "
                f"applied={iteration.applied}, "
                f"stop={iteration.stop_label}"
            )
        return "\n".join(lines)

    def to_dict(self) -> JsonObject:
        return {
            "completed": self.completed,
            "applied": self.applied,
            "stop_reason": self.terminal_reason.value,
            "iteration_count": self.iteration_count,
            "total_applied_rewrite_count": self.total_applied_rewrite_count,
            "total_simulated_rewrite_count": self.total_simulated_rewrite_count,
            "changed_file_paths": self.changed_file_paths,
            "simulated_changed_file_paths": self.simulated_changed_file_paths,
            "final_finding_count": self.final_finding_count,
            "replay_plan": self.replay_plan.to_dict(),
            "iterations": tuple(iteration.to_dict() for iteration in self.iterations),
        }


@dataclass(frozen=True)
class CodemodFixpointStop:
    """Terminal decision for one fixpoint iteration."""

    completed: bool
    reason: CodemodWorkflowStopReason
    simulation: CodemodPlanDocumentSimulation | None = None

    @classmethod
    def no_executable_recipes(cls) -> "CodemodFixpointStop":
        return cls(
            completed=True,
            reason=CodemodWorkflowStopReason.NO_EXECUTABLE_RECIPES,
        )

    @classmethod
    def from_simulation(
        cls,
        simulation: CodemodPlanDocumentSimulation,
    ) -> "CodemodFixpointStop | None":
        if simulation.simulation.applied_rewrite_count == 0:
            return cls(
                completed=False,
                reason=CodemodWorkflowStopReason.EMPTY_REWRITE_BATCH,
                simulation=simulation,
            )
        if not simulation.is_clean:
            return cls(
                completed=False,
                reason=CodemodWorkflowStopReason.ARCHITECTURE_GUARD_FAILED,
                simulation=simulation,
            )
        return None


@dataclass(frozen=True)
class CodemodFixpointIterationIdentity:
    """Stable identity for one fixpoint iteration."""

    index: int


@dataclass(frozen=True)
class CodemodFixpointIterationBuilder:
    """Build iteration and terminal reports for one fixpoint scan/plan pair."""

    prior_iterations: tuple[CodemodFixpointIteration, ...]
    identity: CodemodFixpointIterationIdentity
    scan: CodemodFixpointScan
    plan: FindingRecipePlan

    @property
    def recipe_count(self) -> int:
        return len(self.plan.document.recipes)

    @property
    def expected_removed_finding_ids(self) -> tuple[str, ...]:
        return self.plan.expected_removed_finding_ids

    def stopped_report(
        self,
        stop: CodemodFixpointStop,
    ) -> CodemodFixpointReport:
        return CodemodFixpointReport(
            iterations=(
                *self.prior_iterations,
                self.iteration(
                    simulation=stop.simulation,
                    stop_reason=stop.reason,
                ),
            ),
            completed=stop.completed,
            terminal_reason=stop.reason,
            final_finding_count=len(self.scan.findings),
        )

    def applied_iteration(
        self,
        simulation: CodemodPlanDocumentSimulation,
        post_scan: CodemodFixpointScan,
        *,
        applied: bool,
    ) -> CodemodFixpointIteration:
        return self.iteration(
            simulation=simulation,
            finding_delta=CodemodFindingDelta.from_findings(
                tuple(self.scan.findings),
                tuple(post_scan.findings),
            ),
            applied=applied,
        )

    def iteration(
        self,
        *,
        simulation: CodemodPlanDocumentSimulation | None = None,
        finding_delta: CodemodFindingDelta | None = None,
        applied: bool = False,
        stop_reason: CodemodWorkflowStopReason | None = None,
    ) -> CodemodFixpointIteration:
        return CodemodFixpointIteration(
            iteration_index=self.identity.index,
            finding_count=len(self.scan.findings),
            recipe_count=self.recipe_count,
            finding_change=CodemodFindingChangeProjection(
                expected_removed_finding_ids=self.expected_removed_finding_ids,
                finding_delta=finding_delta,
            ),
            document=self.plan.document,
            report=self.plan.report,
            simulation_result=simulation,
            applied=applied,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True, kw_only=True)
class CodemodFixpointRunner(CodemodGuardedWorkflowRequest):
    """Iteratively apply finding-backed DSL recipes until reaching a fixpoint."""

    max_iterations: int

    def run(self) -> CodemodFixpointReport:
        if self.max_iterations < 1:
            raise ValueError("--codemod-fixpoint-max-iterations must be at least 1")
        iterations: list[CodemodFixpointIteration] = []
        next_scan: CodemodFixpointScan | None = None
        for iteration_index in range(self.max_iterations):
            scan = next_scan or self.scan(iteration_index)
            next_scan = None
            snapshot = scan.source_snapshot
            plan = snapshot.plan_from_findings(scan.findings)
            iteration_builder = CodemodFixpointIterationBuilder(
                prior_iterations=tuple(iterations),
                identity=CodemodFixpointIterationIdentity(index=iteration_index),
                scan=scan,
                plan=plan,
            )
            if not plan.document.has_recipes:
                return iteration_builder.stopped_report(
                    CodemodFixpointStop.no_executable_recipes()
                )
            guarded_document = CodemodPlanDocument(
                recipes=plan.document.recipes,
                guard_suite=self.guard_suite,
            )
            simulation = guarded_document.simulate_snapshot(snapshot)
            stop = CodemodFixpointStop.from_simulation(
                simulation,
            )
            if stop is not None:
                return iteration_builder.stopped_report(stop)
            if self.dry_run:
                next_scan = self.projected_scan(
                    scan,
                    simulation.simulation,
                )
            else:
                simulation.apply()
                next_scan = self.scan(iteration_index + 1)
            iterations.append(
                iteration_builder.applied_iteration(
                    simulation,
                    next_scan,
                    applied=not self.dry_run,
                )
            )
        final_scan = next_scan or self.scan(self.max_iterations)
        return CodemodFixpointReport(
            iterations=tuple(iterations),
            completed=False,
            terminal_reason=CodemodWorkflowStopReason.MAX_ITERATIONS,
            final_finding_count=len(final_scan.findings),
        )


@dataclass(frozen=True, kw_only=True)
class CodemodRefactorGoalRunner(CodemodGuardedWorkflowRequest):
    """Simulate or apply staged DSL recipes until a declared goal resolves."""

    goal: CodemodRefactorGoal

    @cached_property
    def target_policy(self) -> CodemodRefactorGoalTargetPolicy:
        return CodemodRefactorGoalTargetPolicy.policy_for(self.goal.kind)

    def run(self) -> CodemodRefactorGoalReport:
        if self.goal.max_stages < 1:
            raise ValueError("goal max_stages must be at least 1")
        stages: list[CodemodRefactorGoalStage] = []
        active_scan = self.scan(0)
        if not self.target_policy.target_findings(self.goal, active_scan.findings):
            return self._run_report_authority(
                (),
                active_scan,
                CodemodWorkflowStopReason.NO_TARGET_FINDINGS,
                True,
                None,
            )
        for stage_index in range(self.goal.max_stages):
            stage_attempt = self.stage_attempt(stage_index, active_scan)
            if stage_attempt.stage is None:
                return self._run_report_authority(
                    tuple(stages),
                    active_scan,
                    CodemodWorkflowStopReason.NO_EXECUTABLE_RECIPES,
                    False,
                    stage_attempt.report,
                )
            stage = stage_attempt.stage
            if stage.rewrite_count == 0:
                return self._run_report_authority(
                    (*stages, stage),
                    active_scan,
                    CodemodWorkflowStopReason.EMPTY_REWRITE_BATCH,
                    False,
                    None,
                )
            if not stage.simulation.is_clean:
                return self._run_report_authority(
                    (*stages, stage),
                    active_scan,
                    CodemodWorkflowStopReason.ARCHITECTURE_GUARD_FAILED,
                    False,
                    None,
                )
            next_scan = self.next_scan(
                active_scan,
                stage,
                projected_scan=stage_attempt.projected_scan,
            )
            recorded_stage = stage if self.dry_run else replace(stage, applied=True)
            stages.append(recorded_stage)
            if stage.progress.achieved:
                return self._run_report_authority(
                    tuple(stages),
                    next_scan,
                    CodemodWorkflowStopReason.ACHIEVED,
                    True,
                    None,
                )
            if not stage.progress.made_progress:
                return self._run_report_authority(
                    tuple(stages),
                    next_scan,
                    CodemodWorkflowStopReason.NO_PROGRESS,
                    False,
                    None,
                )
            active_scan = next_scan
        return self._run_report_authority(
            tuple(stages),
            active_scan,
            CodemodWorkflowStopReason.MAX_STAGES,
            False,
            None,
        )

    def stage_attempt(
        self,
        stage_index: int,
        scan: CodemodFixpointScan,
    ) -> CodemodRefactorGoalStageAttempt:
        target_findings = self.target_policy.target_findings(self.goal, scan.findings)
        if not target_findings:
            return CodemodRefactorGoalStageAttempt(
                stage_index=stage_index,
                target_finding_count=0,
            )
        snapshot = scan.source_snapshot
        plan = snapshot.plan_from_findings(
            target_findings,
            detector_ids=self.goal.detector_ids,
        )
        if not plan.document.has_recipes:
            return CodemodRefactorGoalStageAttempt(
                stage_index=stage_index,
                target_finding_count=len(target_findings),
                report=plan.report,
            )
        document = CodemodPlanDocument(
            recipes=plan.document.recipes,
            guard_suite=self.guard_suite.merge(plan.document.guard_suite),
        )
        simulation = document.simulate_snapshot(snapshot)
        projected_scan = self.projected_goal_scan(scan, simulation.simulation)
        progress = CodemodRefactorGoalProgress.from_findings(
            self.goal,
            scan.findings,
            projected_scan.findings,
        )
        return CodemodRefactorGoalStageAttempt(
            stage_index=stage_index,
            target_finding_count=len(target_findings),
            report=plan.report,
            projected_scan=projected_scan,
            stage=CodemodRefactorGoalStage(
                stage_index=stage_index,
                document=document,
                report=plan.report,
                simulation=simulation,
                progress=progress,
                finding_change=CodemodFindingChangeProjection(
                    expected_removed_finding_ids=plan.expected_removed_finding_ids,
                    finding_delta=CodemodFindingDelta.from_findings(
                        tuple(scan.findings),
                        tuple(projected_scan.findings),
                    ),
                ),
                applied=False,
            ),
        )

    def next_scan(
        self,
        scan: CodemodFixpointScan,
        stage: CodemodRefactorGoalStage,
        *,
        projected_scan: CodemodFixpointScan | None = None,
    ) -> CodemodFixpointScan:
        if self.dry_run:
            if projected_scan is not None:
                return projected_scan
            return self.projected_scan(scan, stage.simulation.simulation)
        stage.simulation.apply()
        return self.scan(stage.stage_index + 1)

    def projected_goal_scan(
        self,
        scan: CodemodFixpointScan,
        simulation: CodemodSimulationReport,
    ) -> CodemodFixpointScan:
        detector_types = self.goal_detector_types()
        if not detector_types:
            return self.projected_scan(scan, simulation)
        modules = ProjectedScanModuleSet(
            modules=tuple(scan.modules),
            simulation=simulation,
            roots=self.roots,
        ).modules_after_projection()
        target_findings = analyze_detector_types(
            list(modules),
            self.config,
            detector_types=detector_types,
        )
        preserved_findings = tuple(
            finding
            for finding in scan.findings
            if finding.detector_id not in self.goal.detector_ids
        )
        return CodemodFixpointScan(
            modules=list(modules),
            findings=[*preserved_findings, *target_findings],
        )

    def goal_detector_types(self) -> tuple[type[IssueDetector], ...]:
        if not self.goal.detector_ids:
            return ()
        detector_types = tuple(
            IssueDetector.registered_detector_type_for_id(detector_id)
            for detector_id in self.goal.detector_ids
        )
        if any(detector_type is None for detector_type in detector_types):
            return ()
        return tuple(
            detector_type
            for detector_type in detector_types
            if detector_type is not None
        )

    def _run_report_authority(
        self,
        stages: tuple[CodemodRefactorGoalStage, ...],
        scan: CodemodFixpointScan,
        reason: CodemodWorkflowStopReason,
        completed: bool,
        terminal_synthesis_report: FindingRecipeSynthesisReport | None = None,
    ) -> CodemodRefactorGoalReport:
        return self.report(
            stages=stages,
            scan=scan,
            reason=reason,
            completed=completed,
            terminal_synthesis_report=terminal_synthesis_report,
        )

    def report(
        self,
        *,
        stages: tuple[CodemodRefactorGoalStage, ...],
        scan: CodemodFixpointScan,
        reason: CodemodWorkflowStopReason,
        completed: bool,
        terminal_synthesis_report: FindingRecipeSynthesisReport | None = None,
    ) -> CodemodRefactorGoalReport:
        return CodemodRefactorGoalReport(
            goal=self.goal,
            stages=stages,
            completed=completed,
            terminal_reason=reason,
            final_finding_count=len(scan.findings),
            final_target_finding_ids=tuple(
                finding.stable_id
                for finding in self.target_policy.target_findings(
                    self.goal,
                    scan.findings,
                )
            ),
            terminal_synthesis_report=(
                terminal_synthesis_report or FindingRecipeSynthesisReport()
            ),
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
            module=ast.parse(source, filename=str(module.path)),
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
        return self.module.path.as_posix()

    @property
    def has_rewrite(self) -> bool:
        return self.module_path in self.simulation.rewritten_sources

    @property
    def source(self) -> str:
        if self.has_rewrite:
            return self.simulation.rewritten_sources[self.module_path]
        return self.module.source
