"""Move every hidden import of one module that closes no cycle to its top."""

from __future__ import annotations

from dataclasses import dataclass

from nominal_refactor_advisor.codemod_payload import StringArrayPayloadValueCodec, codemod_payload_field
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.hidden_imports import HiddenImportPlan
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.refactor_concepts import HiddenImportPromotionConcept
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)


@dataclass(frozen=True, kw_only=True)
class PromoteHiddenImportsOperation(RepositorySourceReprovedOperation):
    """Move each hidden import of the target module that the repository's plan
    accepts to the module's top: out of its `TYPE_CHECKING` guard, or out of
    its function. The plan is derived again on the state the operation applies
    to, so a stage sees every promotion an earlier stage made."""

    only: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec(), default=())
    """The identities of the imports to move, when a change freed some of the
    module's imports and leaves the rest as they were; empty moves every one."""

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        plan = snapshot.module_set.fact(HiddenImportPlan)
        only = frozenset(self.only) or None
        accepted = [hidden for hidden in plan.accepted_in(self.target.file_path) if only is None or hidden.identity in only]
        return plan.edits_for(
            self.target.file_path,
            self.rationale_text(
                f"Import {len(accepted)} hidden import(s) at the module's top: none closes a load-time cycle."
            ),
            only,
        )


class HiddenImportFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    HiddenImportPromotionConcept,
):
    """One promotion of every accepted hidden import in the finding's module."""

    action_key_child_subject = "hidden_imports"

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("promoting hidden imports needs source context and its module")
        operation = PromoteHiddenImportsOperation(target=SourceRewriteTarget(file_path=evidence.file_path))
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason="Each moved import closes no load-time cycle, even with every other accepted one in place.",
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = ("HiddenImportFindingRecipeSynthesizer", "PromoteHiddenImportsOperation")
