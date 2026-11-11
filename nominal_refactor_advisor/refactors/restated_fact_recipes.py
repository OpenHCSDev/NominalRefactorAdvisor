"""Executable recipes for restated-fact findings whose factoring an operation performs."""

from __future__ import annotations

from abc import ABC

from nominal_refactor_advisor.codemod_runtime import (
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.refactor_concepts import (
    DeadModuleDissolutionConcept,
    ForwarderInlineConcept,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.function_operations import InlineForwarderOperation
from nominal_refactor_advisor.refactors.module_move import DissolveModuleOperation


class ForwarderInlineFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ForwarderInlineConcept,
    ABC,
):
    """Inline every forward of a facade that proves, one operation each.

    The finding's evidence after the facade's own line names each forwarding
    member. A forward the operation refuses stays, and its refusal is the
    recipe's rejection only when no forward proves.
    """

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or len(finding.evidence) < 2:
            return self.rejected_evaluation(
                "inlining forwarders requires source context and the forwarding members"
            )
        proved: list[InlineForwarderOperation] = []
        refused: list[str] = []
        for member in finding.evidence[1:]:
            operation = InlineForwarderOperation(
                target=SourceRewriteTarget(file_path=member.file_path, qualname=member.symbol),
                rationale=f"Read the held object where {member.symbol} only forwarded to it.",
            )
            try:
                operation.source_edits(context)
            except ValueError as error:
                refused.append(f"{member.symbol}: {error}")
                continue
            proved.append(operation)
        if not proved:
            return self.rejected_evaluation("; ".join(refused))
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-inline-forwarders",
            reason="Consumers read the held object; the forwards are deleted.",
        )
        for operation in proved:
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)


class DeadModuleDissolutionFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    DeadModuleDissolutionConcept,
    ABC,
):
    """Dissolve the module the finding names; the operation refuses one that still
    declares something or that a consumer binds for its side effect."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "dissolving a dead module requires source context and its path"
            )
        operation = DissolveModuleOperation(
            target=SourceRewriteTarget(file_path=evidence.file_path),
            rationale="Nothing imports the module.",
        )
        try:
            operation.source_edits(context)
            operation.source_file_deletions(context)
        except ValueError as error:
            return self.rejected_evaluation(str(error))
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-dissolve-module",
                reason="Delete the module nothing imports.",
            ).with_operation(operation)
        )
