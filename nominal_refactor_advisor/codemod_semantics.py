"""Closed semantic axes for codemod execution and proof."""

import ast
from collections.abc import (
    Callable,
    Iterable,
)
from enum import StrEnum


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
    ast.parse(source, filename=file_path)


def _validate_libcst_source(source: str, file_path: str) -> None:
    del file_path
    import libcst as cst

    cst.parse_module(source)


class CodemodBackend(StrEnum):
    """Parser backend carrying its simulated-source validation behavior."""

    AST_SPAN = ("ast_span", _validate_ast_span_source)
    LIBCST = ("libcst", _validate_libcst_source)

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
        """Validate source through this backend's declared parser."""

        self._source_validator(source, file_path)


class FindingRecipeSynthesisDisposition(StrEnum):
    """Reporting disposition carried by each terminal synthesis status."""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNCOUNTED = "uncounted"


class FindingRecipePlanningHorizon(StrEnum):
    """Strongest horizon proved for an executable recipe candidate."""

    NONE = ("none", 0, "")
    CURRENT_SNAPSHOT = (
        "current_snapshot",
        1,
        "application requires a proof across reachable refactor trajectories",
    )
    UNPROVED = (
        "unproved",
        2,
        "application requires a complete proof across reachable refactor trajectories",
    )

    def __new__(
        cls,
        value: str,
        proof_rank: int,
        application_block_reason: str,
    ) -> "FindingRecipePlanningHorizon":
        member = str.__new__(cls, value)
        member._value_ = value
        member._proof_rank = proof_rank
        member._application_block_reason = application_block_reason
        return member

    @classmethod
    def join(
        cls,
        horizons: Iterable["FindingRecipePlanningHorizon"],
    ) -> "FindingRecipePlanningHorizon":
        return max(horizons, key=lambda horizon: horizon._proof_rank, default=cls.NONE)

    @property
    def requires_trajectory_proof(self) -> bool:
        return self is not type(self).NONE

    @property
    def application_block_reason(self) -> str:
        return self._application_block_reason


class FindingRecipeSynthesisStatus(StrEnum):
    """Recipe-synthesis outcome for one advisor finding."""

    EXECUTABLE_CANDIDATE = (
        "executable_candidate",
        "",
        FindingRecipeSynthesisDisposition.CANDIDATE,
    )
    NO_SYNTHESIZER = (
        "no_synthesizer",
        "detector declaration has no executable finding synthesis behavior",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_ACTION_KEYS = (
        "no_action_keys",
        "executable recipe has no stable source action keys",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    CONFLICTING_TRAJECTORY_BRANCHES = (
        "conflicting_trajectory_branches",
        "conflicting current-snapshot candidates require trajectory exploration",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    UNPROVED_RECIPE_PLAN = (
        "unproved_recipe_plan",
        "recipe compatibility or batch simulation is unproved",
        FindingRecipeSynthesisDisposition.UNSUPPORTED,
    )
    NO_EFFECTIVE_REWRITES = (
        "no_effective_rewrites",
        "synthesizer recipe produced no effective source rewrites",
        FindingRecipeSynthesisDisposition.REJECTED,
    )
    REJECTED_BY_SAFETY_CHECK = (
        "rejected_by_safety_check",
        "",
        FindingRecipeSynthesisDisposition.REJECTED,
    )

    def __new__(
        cls,
        value: str,
        default_reason: str,
        disposition: FindingRecipeSynthesisDisposition,
    ) -> "FindingRecipeSynthesisStatus":
        member = str.__new__(cls, value)
        member._value_ = value
        member._default_reason = default_reason
        member._disposition = disposition
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
