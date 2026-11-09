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

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceRejection,
    CapturedReferenceViolation,
)

from nominal_refactor_advisor.codemod_architecture_guards import ArchitectureGuardRule
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    OptionalStrEnumPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import (
    SourceRewriteTarget,
    SourceRewriteTargetPreflightDetail,
)
from nominal_refactor_advisor.codemod_semantics import (
    CodemodPreflightStatus,
    CodemodSourceDependencyScope,
)
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.semantic_descent import AuthorityClaim

SourceReproofValueT = TypeVar("SourceReproofValueT")


@dataclass(frozen=True)
class ReproofCauseDiagnostic(CodemodPayloadRecord):
    """Presentation of one actual cause; absent classification remains absent."""

    violation: CapturedReferenceViolation | None = codemod_payload_field(
        OptionalStrEnumPayloadValueCodec(CapturedReferenceViolation)
    )
    message: str = codemod_payload_field(RequiredStringPayloadValueCodec())


@dataclass(frozen=True)
class SourceReproofDiagnostic(SourceRewriteTargetPreflightDetail):
    """Source target and ordered refusal evidence, without runtime object graphs."""

    causes: tuple[ReproofCauseDiagnostic, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ReproofCauseDiagnostic)
    )

    @classmethod
    def from_error(
        cls, target: SourceRewriteTarget, error: BaseException
    ) -> SourceReproofDiagnostic:
        causes = []
        visited: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            causes.append(
                ReproofCauseDiagnostic(
                    (
                        current.violation
                        if isinstance(current, CapturedReferenceRejection)
                        else None
                    ),
                    str(current),
                )
            )
            current = current.__cause__
        return cls(target, tuple(causes))


@dataclass(frozen=True, kw_only=True)
class SourceReprovedOperation(RefactorRecipeOperation, ABC):
    """Operation whose physical edits must be re-derived from current source."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        snapshot = context.execution_snapshot()
        return snapshot.retained_operation_derivation(
            (self, SourceReprovedOperation.source_edits),
            lambda: self.required_reproof(lambda: self.source_edits_from_snapshot(snapshot)),
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
            raise self.failed_preflight(error) from error

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

    def failed_preflight(
        self, error: TypeError | ValueError
    ) -> CodemodOperationPreflightError:
        return CodemodOperationPreflightError(
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.FAILED,
                message=str(error),
                detail=SourceReproofDiagnostic.from_error(self.target, error),
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
