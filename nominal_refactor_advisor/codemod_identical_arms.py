"""Executable recipe that collapses a branch whose arms are all the same code."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import cast

from .codemod import FindingRecipeSynthesizer, PrimaryEvidenceActionKeysMixin
from .codemod_payload import RequiredIntegerPayloadValueCodec, codemod_payload_field
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import NominalSourceEdit, SourceTextGeometry
from .codemod_target_selectors import SourceIndexTargetSelector
from .false_distinction_observations import IdenticalArms, ScopeNode
from .models import RefactorFinding
from .refactor_concepts import FalseDistinctionErasureConcept
from .source_index import AstTargetNodeKind


@dataclass(frozen=True, kw_only=True)
class CollapseIdenticalArmsOperation(SourceReprovedOperation):
    """Replace the branch at `line` of one scope with its one arm.

    The target is the scope that owns the branch: a function, a method or the
    module. Current source must still hold a branch there whose arms are the
    same code and whose tests can be deleted without an effect.
    """

    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        scope = cast(ScopeNode, node)
        geometry = SourceTextGeometry(snapshot.sources_by_file_path[target.file_path])
        matching = [
            arms for arms in IdenticalArms.in_scope(scope.body) if arms.node.lineno == self.line
        ]
        if len(matching) != 1:
            raise ValueError(
                f"{target.qualname or target.file_path!r} holds "
                f"{len(matching)} branch(es) with identical arms at line {self.line}"
            )
        (arms,) = matching
        return geometry.physical_edits(
            file_path=target.file_path,
            replacements=(arms.collapse(geometry),),
            rationale=self.rationale
            or f"Collapse the branch at line {self.line} to its one arm.",
        )


class IdenticalArmsFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    FalseDistinctionErasureConcept,
    ABC,
):
    """Collapse the branch a finding's primary witness names, in the scope that owns it."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "collapsing identical arms requires source context and a primary witness"
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
        operation = CollapseIdenticalArmsOperation(
            target=SourceRewriteTarget(target_id=target_ids[0]), line=evidence.line
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-collapse-identical-arms",
                reason="Every arm runs the same code; the branch becomes that code.",
            ).with_operation(operation)
        )


__all__ = (
    "CollapseIdenticalArmsOperation",
    "IdenticalArmsFindingRecipeSynthesizer",
)
