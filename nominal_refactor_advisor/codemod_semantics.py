"""Closed semantic axes for codemod execution and proof."""

from collections.abc import (
    Callable,
    Iterable,
)
from enum import StrEnum
from typing import ClassVar

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.needed_input import NeededInput


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


class CodemodSourceDependencyScope(StrEnum):
    """Source coverage required to prove one operation's physical edits."""

    EXPLICIT_TARGETS = ("explicit_targets", True)
    REPOSITORY = ("repository", False)

    def __new__(
        cls,
        value: str,
        permits_fast_snapshot: bool,
    ) -> "CodemodSourceDependencyScope":
        member = str.__new__(cls, value)
        member._value_ = value
        member._permits_fast_snapshot = permits_fast_snapshot
        return member

    @property
    def permits_fast_snapshot(self) -> bool:
        """Return whether explicit targets are a complete proof source."""

        return self._permits_fast_snapshot

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
    NativePythonCompilation(source, file_path).compile()


class CodemodBackend(StrEnum):
    """Source backend carrying its simulated-source validation behavior."""

    AST_SPAN = ("ast_span", _validate_ast_span_source)

    def __new__(
        cls,
        value: str,
        source_validator: Callable[[str, str], None],
    ) -> "CodemodBackend":
        member = str.__new__(cls, value)
        member._value_ = value
        member._source_validator = source_validator
        return member

    def validate_source(self, source: str, file_path: str) -> None:
        """Check syntax and Python compilation rules without executing source."""

        self._source_validator(source, file_path)


class FindingRecipeSynthesisDisposition(StrEnum):
    """Reporting disposition carried by each terminal synthesis status: ready, needs
    an input the record names, needs NRA to grow, or counted by another record."""

    CANDIDATE = "candidate"
    REJECTED = "needs_input"
    UNSUPPORTED = "needs_nra"
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
    def requires_trajectory_proof(self) -> bool:
        return True

    @property
    def application_block_reason(self) -> str:
        return "application requires a proof across reachable refactor trajectories"


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


class FindingRecipeSynthesisStatus(StrEnum):
    """Recipe-synthesis outcome for one advisor finding, with the kinds of input
    an outcome of it needs before NRA can apply a fix."""

    EXECUTABLE_CANDIDATE = (
        "executable_candidate",
        "",
        FindingRecipeSynthesisDisposition.CANDIDATE,
        (),
    )
    NO_EVALUATOR = (
        "needs_operation",
        "no operation is written for this detector's findings yet",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
        (NeededInput.OPERATION,),
    )
    NO_ACTION_KEYS = (
        "no_action_keys",
        "the executable recipe needs stable source action keys",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
        (NeededInput.PROOF,),
    )
    CONFLICTING_TRAJECTORY_BRANCHES = (
        "conflicting_trajectory_branches",
        "conflicting current-snapshot candidates need trajectory exploration",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
        (NeededInput.PROOF,),
    )
    UNPROVED_RECIPE_PLAN = (
        "unproved_recipe_plan",
        "recipe compatibility or batch simulation needs a proof",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
        (NeededInput.PROOF,),
    )
    NO_EFFECTIVE_REWRITES = (
        "no_effective_rewrites",
        "the synthesizer's recipe produced no effective source rewrites",
        FindingRecipeSynthesisDisposition.REJECTED,
        (NeededInput.OPERATION,),
    )
    REJECTED_BY_SAFETY_CHECK = (
        "needs_input",
        "",
        FindingRecipeSynthesisDisposition.REJECTED,
        (NeededInput.UNTYPED,),
    )
    COVERED_BY_CANDIDATE = (
        "covered_by_candidate",
        "",
        FindingRecipeSynthesisDisposition.UNCOUNTED,
        (),
    )

    def __new__(
        cls,
        value: str,
        default_reason: str,
        disposition: FindingRecipeSynthesisDisposition,
        default_needs: tuple[NeededInput, ...],
    ) -> "FindingRecipeSynthesisStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._default_reason = default_reason
        member._disposition = disposition
        member.default_needs = default_needs
        return member

    @property
    def default_reason(self) -> str:
        return self._default_reason

    @property
    def candidate(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.CANDIDATE

    @property
    def rejected(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.REJECTED

    @property
    def unsupported(self) -> bool:
        return self._disposition is FindingRecipeSynthesisDisposition.UNSUPPORTED


class CodemodPreflightStatus(StrEnum):
    """Machine-readable codemod preflight outcome."""

    PASSED = ("passed", True)
    FAILED = ("failed", False)

    def __new__(
        cls,
        value: str,
        is_passed: bool,
    ) -> "CodemodPreflightStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_passed = is_passed
        return member

    @property
    def is_passed(self) -> bool:
        """Whether the checked codemod contract is satisfied."""

        return self._is_passed

    @property
    def is_failed(self) -> bool:
        """Whether the checked codemod contract is unsatisfied."""

        return not self.is_passed
