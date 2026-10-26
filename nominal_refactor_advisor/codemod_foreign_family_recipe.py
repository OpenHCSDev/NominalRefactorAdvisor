"""Executable recipes that move foreign-class sites onto declared families."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from .codemod import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ForeignAdapterBindChainConcept,
    ForeignFamilyRoutingConcept,
)
from .codemod_foreign_adapter_source import ForeignAdapterBindChainModuleFamily
from .codemod_foreign_family_source import ForeignFamilyFilterModuleFamily
from .codemod_function_operations import (
    FoldAdapterBindChainOperation,
    RouteForeignFamilyFilterOperation,
)
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import (
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .observation_families import ShapeFamily
from .observation_shapes import ForeignFamilySiteModule
from .codemod_target_selectors import SourceIndexTargetSelector
from .models import RefactorFinding


class ForeignFamilySiteFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin, FindingRecipeSynthesizer, ABC
):
    """One operation per evidenced function, on the family its finding names.

    The finding's sites already belong to a declared family; current source
    proves each operation before the recipe is offered.
    """

    operation_type: ClassVar[type[SourceReprovedOperation]]
    site_family: ClassVar[type[ShapeFamily[ForeignFamilySiteModule]]]
    recipe_id_suffix: ClassVar[str]
    recipe_reason: ClassVar[str]

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None:
            return self.rejected_evaluation(
                "foreign family routing requires source context"
            )
        snapshot = context.execution_snapshot()
        lines_by_file: dict[str, set[int]] = {}
        for evidence in finding.evidence:
            lines_by_file.setdefault(evidence.file_path, set()).add(evidence.line)
        operations: dict[tuple[str, str, str], SourceReprovedOperation] = {}
        refusals: list[str] = []
        for file_path, lines in sorted(lines_by_file.items()):
            parsed_module = snapshot.parsed_module_for_source_path(file_path)
            for projection in self.site_family.collect(parsed_module):
                for site in projection.sites:
                    key = (file_path, site.symbol, site.family)
                    if site.lineno not in lines or key in operations:
                        continue
                    try:
                        operations[key] = self.checked_operation(snapshot, *key)
                    except ValueError as error:
                        refusals.append(str(error))
        if not operations:
            return self.rejected_evaluation(
                "no evidenced function admits family routing"
                + (f" ({'; '.join(refusals)})" if refusals else "")
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        )
        for operation in operations.values():
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)

    @classmethod
    def checked_operation(
        cls,
        snapshot: CodemodSelectorContext,
        file_path: str,
        symbol: str,
        family: str,
    ) -> SourceReprovedOperation:
        """The operation for one function, proved against current source."""

        (target_id,) = SourceIndexTargetSelector.for_function_or_method(
            file_path, symbol
        ).target_ids(snapshot)
        operation = cls.operation_type(
            target=SourceRewriteTarget(target_id=target_id), family=family
        )
        operation.source_edits_from_snapshot(snapshot.execution_snapshot())
        return operation


class ForeignFamilyRoutableFilterFindingRecipeSynthesizer(
    ForeignFamilySiteFindingRecipeSynthesizer, ForeignFamilyRoutingConcept
):
    """Route each evidenced filter's function through the family its finding names."""

    operation_type = RouteForeignFamilyFilterOperation
    site_family = ForeignFamilyFilterModuleFamily
    recipe_id_suffix = "route-foreign-family-filter"
    recipe_reason = (
        "Filters testing exactly a declared family's classes iterate "
        "`Family.within(...)` instead of restating the type test."
    )


class ForeignAdapterBindChainFindingRecipeSynthesizer(
    ForeignFamilySiteFindingRecipeSynthesizer, ForeignAdapterBindChainConcept
):
    """Fold each evidenced bind chain's function into its adapter family."""

    operation_type = FoldAdapterBindChainOperation
    site_family = ForeignAdapterBindChainModuleFamily
    recipe_id_suffix = "fold-adapter-bind-chain"
    recipe_reason = (
        "Bindings read class by class from foreign nodes become property reads "
        "of the adapter family that already implements them."
    )
