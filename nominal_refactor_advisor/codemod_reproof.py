"""Current-source reproof contracts for semantic codemod operations."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    ClassVar,
    TypeVar,
)

from .codemod_architecture_guards import ArchitectureGuardRule
from .codemod_operations import RefactorRecipeOperation
from .codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTargetPreflightDetail
from .codemod_semantics import (
    CodemodPreflightStatus,
    CodemodSourceDependencyScope,
)
from .codemod_source_edits import NominalSourceEdit
from .semantic_descent import AuthorityClaim

SourceReproofValueT = TypeVar("SourceReproofValueT")


@dataclass(frozen=True, kw_only=True)
class SourceReprovedOperation(RefactorRecipeOperation, ABC):
    """Operation whose physical edits must be re-derived from current source."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_reproof(
            lambda: self.source_edits_from_snapshot(context.execution_snapshot())
        )

    def required_reproof(
        self,
        derivation: Callable[[], SourceReproofValueT],
    ) -> SourceReproofValueT:
        """Evaluate one current-source derivation through the shared failure contract."""

        try:
            return derivation()
        except CodemodOperationPreflightError:
            raise
        except (TypeError, ValueError) as error:
            raise self.failed_preflight(str(error)) from error

    def declared_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return self.required_reproof(
            lambda: self.current_source_authority_claims(context)
        )

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        """Derive authority claims only from the current source snapshot."""

        return super().declared_authority_claims(context)

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return self.required_reproof(
            lambda: self.current_source_architecture_guard_rules(context)
        )

    def current_source_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Derive post-refactor invariants only from the current source snapshot."""

        return super().declared_architecture_guard_rules(context)

    def failed_preflight(self, message: str) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=message,
                detail=SourceRewriteTargetPreflightDetail(self.target),
            )
        )

    @abstractmethod
    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class RepositorySourceReprovedOperation(SourceReprovedOperation, ABC):
    """Source-reproved operation whose proof requires repository-wide context."""

    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
    )
