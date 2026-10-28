"""Executable recipes that delete a distinction the code then treats identically.

Each operation targets the scope that owns a branch (a function, a method or
the module) and the branch's line, and re-derives its edits from current
source: the branch must still be there, and its deleted tests must still be
free of effects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeVar, cast

from .codemod import FindingRecipeSynthesizer, PrimaryEvidenceActionKeysMixin
from .codemod_foreign_family_source import RewrittenLineFit
from .codemod_payload import RequiredIntegerPayloadValueCodec, codemod_payload_field
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from .codemod_target_selectors import SourceIndexTargetSelector
from .declared_root import DeclaredRoot
from .false_distinction_observations import (
    IdenticalArms,
    Scope,
    ScopeNode,
    UndistinguishedTypeArms,
)
from .models import RefactorFinding
from .refactor_concepts import FalseDistinctionErasureConcept
from .source_index import AstTargetNodeKind

FoundT = TypeVar("FoundT")


@dataclass(frozen=True, kw_only=True)
class ScopeBranchOperation(SourceReprovedOperation, ABC):
    """Rewrite the branch at `line` of the scope the target names."""

    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    @abstractmethod
    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """The edits for the one branch at `line`; a ValueError when it is gone or unsafe."""

        raise NotImplementedError

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        geometry = SourceTextGeometry(snapshot.sources_by_file_path[target.file_path])
        fit = RewrittenLineFit(
            geometry, DeclaredRoot.line_length_for(Path(target.file_path))
        )
        return geometry.physical_edits(
            file_path=target.file_path,
            replacements=self.replacements(
                Scope(cast(ScopeNode, node).body), geometry, fit
            ),
            rationale=self.rationale
            or f"Delete the distinction at line {self.line} that the code treats identically.",
        )

    def only(self, found: list[FoundT], what: str) -> FoundT:
        """The one `what` at `line`; current source must still hold exactly one."""

        if len(found) != 1:
            raise ValueError(f"{len(found)} {what} at line {self.line} of the target scope")
        return found[0]


@dataclass(frozen=True, kw_only=True)
class CollapseIdenticalArmsOperation(ScopeBranchOperation):
    """Replace the branch at `line`, whose arms are all the same code, with that code."""

    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        del fit
        found = [arms for arms in IdenticalArms.in_scope(scope) if arms.node.lineno == self.line]
        return (self.only(found, "branch(es) with identical arms").collapse(geometry),)


@dataclass(frozen=True, kw_only=True)
class MergeTypeTestArmsOperation(ScopeBranchOperation):
    """Merge `isinstance` rungs at `line` that run the same code into one test,
    or delete trailing rungs that run the `else` body."""

    def replacements(
        self, scope: Scope, geometry: SourceTextGeometry, fit: RewrittenLineFit
    ) -> tuple[SourceTextSpanReplacement, ...]:
        found = [
            arms
            for arms in UndistinguishedTypeArms.in_scope(scope)
            if arms.line == self.line
        ]
        return self.only(found, "undistinguished type test group(s)").collapse(
            geometry, fit
        )


class ScopeBranchFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    FalseDistinctionErasureConcept,
    ABC,
):
    """One operation on the branch the finding's primary witness names, in its scope."""

    operation_type: ClassVar[type[ScopeBranchOperation]]
    recipe_reason: ClassVar[str]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "deleting a false distinction requires source context and a primary witness"
            )
        selector = (
            SourceIndexTargetSelector(
                node_kinds=(AstTargetNodeKind.MODULE,), file_paths=(evidence.file_path,)
            )
            if evidence.symbol == "<module>"
            else SourceIndexTargetSelector.for_function_or_method(
                evidence.file_path, evidence.symbol
            )
        )
        target_ids = selector.target_ids(context)
        if len(target_ids) != 1:
            return self.rejected_evaluation(
                f"`{evidence.symbol}` is not one function, method or module"
            )
        operation = self.operation_type(
            target=SourceRewriteTarget(target_id=target_ids[0]), line=evidence.line
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=self.recipe_reason,
            ).with_operation(operation)
        )


class IdenticalArmsFindingRecipeSynthesizer(ScopeBranchFindingRecipeSynthesizer, ABC):
    operation_type = CollapseIdenticalArmsOperation
    recipe_reason = "Every arm runs the same code; the branch becomes that code."


class UndistinguishedTypeTestFindingRecipeSynthesizer(
    ScopeBranchFindingRecipeSynthesizer, ABC
):
    operation_type = MergeTypeTestArmsOperation
    recipe_reason = (
        "Type tests whose arms run the same code become one test; "
        "rungs that run the else body are deleted."
    )
