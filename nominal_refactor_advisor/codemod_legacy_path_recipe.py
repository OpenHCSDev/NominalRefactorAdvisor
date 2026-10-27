"""Executable recipe that erases a path marked as the old one."""

from __future__ import annotations

from abc import ABC

from .codemod import (
    EraseDeadCompatibilityOperation,
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from .codemod_runtime import FindingRecipeEvaluation, RefactorRecipe
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_target_selectors import SourceIndexTargetSelector
from .models import RefactorFinding
from .refactor_concepts import LegacyPathErasureConcept
from .source_index import AstTargetNodeKind


class LegacyPathErasureFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin, FindingRecipeSynthesizer, LegacyPathErasureConcept, ABC
):
    """Erase the function or class a finding marks, and forbid residual calls to it.

    Only a declaration the finding's primary witness names exactly can be erased:
    a marked parameter, assignment, import or warning has no target of its own,
    so it is rejected rather than widened to its enclosing declaration. The
    operation's guard rejects the plan while any consumer still calls the path.
    """

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "legacy path erasure requires source context and a primary witness"
            )
        targets = [
            context.source_index.target_by_id[target_id]
            for target_id in SourceIndexTargetSelector(
                node_kinds=(
                    AstTargetNodeKind.FUNCTION,
                    AstTargetNodeKind.METHOD,
                    AstTargetNodeKind.CLASS,
                ),
                file_paths=(evidence.file_path,),
                qualnames=(evidence.symbol,),
            ).target_ids(context)
        ]
        targets = [target for target in targets if target.line == evidence.line]
        if len(targets) != 1:
            return self.rejected_evaluation(
                f"`{evidence.symbol}` at line {evidence.line} is not a function or "
                "class declaration, so only a reader can retire it"
            )
        (target,) = targets
        operation = EraseDeadCompatibilityOperation(
            target=SourceRewriteTarget(target_id=target.target_id),
            forbidden_call_names=(target.name,),
            rationale="Delete the path that declares itself legacy.",
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-erase-legacy-path",
                reason="Erase the legacy path once its consumers use the replacement.",
            ).with_operation(operation)
        )
