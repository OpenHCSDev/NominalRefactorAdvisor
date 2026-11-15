"""Closed semantic axes for codemod execution and proof."""

from collections.abc import (
    Callable,
    Iterable,
)
from enum import StrEnum
from typing import ClassVar

from nominal_refactor_advisor.fielded_enum import FieldedStrEnum
from nominal_refactor_advisor.native_compilation import NativePythonCompilation


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


class CodemodSourceDependencyScope(FieldedStrEnum):
    """Source coverage required to prove one operation's physical edits."""

    EXPLICIT_TARGETS = ("explicit_targets", True)
    REPOSITORY = ("repository", False)

    permits_fast_snapshot: bool
    """Return whether explicit targets are a complete proof source."""

    @classmethod
    def compose(
        cls,
        scopes: Iterable["CodemodSourceDependencyScope"],
    ) -> "CodemodSourceDependencyScope":
        """Return the first scope that forbids narrowing, if one exists."""

        return next(
            (scope for scope in scopes if not scope.permits_fast_snapshot),
            cls.EXPLICIT_TARGETS,
        )


def _validate_ast_span_source(source: str, file_path: str) -> None:
    NativePythonCompilation(source=source, file_path=file_path).compile()


class CodemodBackend(FieldedStrEnum):
    """Source backend carrying its simulated-source validation behavior."""

    AST_SPAN = ("ast_span", _validate_ast_span_source)

    _source_validator: Callable[[str, str], None]

    def validate_source(self, source: str, file_path: str) -> None:
        """Check syntax and Python compilation rules without executing source."""

        self._source_validator(source, file_path)


class FindingRecipeSynthesisDisposition(StrEnum):
    """Reporting disposition carried by each terminal synthesis status."""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNCOUNTED = "uncounted"


class FindingRecipePlanningHorizon(str):
    """MRO-ordered proof horizon for an executable recipe candidate."""

    wire_value: ClassVar[str] = "unproved"
    NONE: ClassVar["FindingRecipePlanningHorizon"]
    CURRENT_SNAPSHOT: ClassVar["FindingRecipePlanningHorizon"]
    UNPROVED: ClassVar["FindingRecipePlanningHorizon"]

    def __new__(cls) -> "FindingRecipePlanningHorizon":
        return str.__new__(cls, cls.wire_value)

    @classmethod
    def join(
        cls,
        horizons: Iterable["FindingRecipePlanningHorizon"],
    ) -> "FindingRecipePlanningHorizon":
        horizon_tuple = tuple(horizons)
        if not horizon_tuple:
            return cls.NONE
        for candidate in horizon_tuple:
            if all(
                isinstance(other_horizon, type(candidate))
                for other_horizon in horizon_tuple
            ):
                return candidate
        raise TypeError("planning horizons must form one nominal MRO chain")

    @property
    def requires_trajectory_proof(self) -> bool:
        return True

    @property
    def application_block_reason(self) -> str:
        return (
            "application requires a complete proof across reachable refactor "
            "trajectories"
        )


class CurrentSnapshotFindingRecipePlanningHorizon(
    FindingRecipePlanningHorizon
):
    """Candidate proved only on the current source snapshot."""

    wire_value = "current_snapshot"

    @property
    def application_block_reason(self) -> str:
        return "run the goal search to prove this across reachable trajectories, then apply"


class CompleteFindingRecipePlanningHorizon(
    CurrentSnapshotFindingRecipePlanningHorizon
):
    """Planning horizon requiring no additional trajectory proof."""

    wire_value = "none"

    @property
    def requires_trajectory_proof(self) -> bool:
        return False

    @property
    def application_block_reason(self) -> str:
        return ""


FindingRecipePlanningHorizon.NONE = CompleteFindingRecipePlanningHorizon()
FindingRecipePlanningHorizon.CURRENT_SNAPSHOT = (
    CurrentSnapshotFindingRecipePlanningHorizon()
)
FindingRecipePlanningHorizon.UNPROVED = FindingRecipePlanningHorizon()


class FindingRecipeSynthesisStatus(FieldedStrEnum):
    """Recipe-synthesis outcome for one advisor finding."""

    EXECUTABLE_CANDIDATE = (
        "executable_candidate",
        "",
        FindingRecipeSynthesisDisposition.CANDIDATE,
    )
    NO_EVALUATOR = (
        "no_evaluator",
        "author this fold from the registered operations: the finding names the "
        "restated fact and its owner, and each operation's preflight proves the move",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_ACTION_KEYS = (
        "no_action_keys",
        "select this recipe's sites by declaration (file_path and qualname) so the "
        "edits survive an earlier stage's rewrite",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    CONFLICTING_TRAJECTORY_BRANCHES = (
        "conflicting_trajectory_branches",
        "run the goal search to order these current-snapshot candidates into one "
        "proved trajectory",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    UNPROVED_RECIPE_PLAN = (
        "unproved_recipe_plan",
        "prove this plan by simulating the batch from the branch head",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_EFFECTIVE_REWRITES = (
        "no_effective_rewrites",
        "the current source already satisfies this finding; move to the next one",
        FindingRecipeSynthesisDisposition.REJECTED,
    )
    REJECTED_BY_SAFETY_CHECK = (
        "rejected_by_safety_check",
        "",
        FindingRecipeSynthesisDisposition.REJECTED,
    )
    COVERED_BY_CANDIDATE = (
        "covered_by_candidate",
        "",
        FindingRecipeSynthesisDisposition.UNCOUNTED,
    )

    default_reason: str
    _disposition: FindingRecipeSynthesisDisposition

    @property
    def candidate(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.CANDIDATE

    @property
    def rejected(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.REJECTED

    @property
    def unsupported(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.UNSUPPORTED


class CodemodPreflightStatus(FieldedStrEnum):
    """Machine-readable codemod preflight outcome."""

    PASSED = ("passed", True)
    FAILED = ("failed", False)

    is_passed: bool
    """Whether the checked codemod contract is satisfied."""

    @property
    def is_failed(self) -> bool:
        """Whether the checked codemod contract is unsatisfied."""

        return not self.is_passed
