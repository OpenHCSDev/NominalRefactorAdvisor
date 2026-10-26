"""Executable recipes for foreign type-test findings: filters onto declared families."""

from __future__ import annotations

from .codemod import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ForeignFamilyRoutingConcept,
)
from .codemod_function_operations import RouteForeignFamilyFilterOperation
from .codemod_runtime import (
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_target_selectors import SourceIndexTargetSelector
from .foreign_family import ForeignFamily
from .models import RefactorFinding
from .observation_families import ForeignTypeTestModuleFamily


class ForeignTypeSwitchFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ForeignFamilyRoutingConcept,
):
    """Route filters whose tested classes are exactly a declared family's.

    A function qualifies when one of the finding's own sites tests exactly the
    members of a declared family. Sites testing part of a family (possible
    missed cases) are not part of this recipe.
    """

    recipe_id_suffix = "route-foreign-family-filter"
    recipe_reason = (
        "Filters testing exactly a declared family's classes iterate "
        "`Family.within(...)` instead of restating the type test."
    )

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
        operations: dict[tuple[str, str, str], RouteForeignFamilyFilterOperation] = {}
        refusals: list[str] = []
        for file_path, lines in sorted(lines_by_file.items()):
            parsed_module = snapshot.parsed_module_for_source_path(file_path)
            sites = tuple(
                site
                for projection in ForeignTypeTestModuleFamily.collect(parsed_module)
                for site in projection.sites
                if site.lineno in lines
            )
            for site in sites:
                for family in ForeignFamily.concrete_families():
                    if not self.tests_exactly(family, site.type_names):
                        continue
                    key = (file_path, site.symbol, family_path(family))
                    if key in operations:
                        continue
                    try:
                        operation = self.checked_operation(snapshot, *key)
                    except ValueError as error:
                        refusals.append(str(error))
                        continue
                    operations[key] = operation
        if not operations:
            return self.rejected_evaluation(
                "no function holds a filter testing exactly a declared family"
                + (f" ({'; '.join(refusals)})" if refusals else "")
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-{self.recipe_id_suffix}",
            reason=self.recipe_reason,
        )
        for operation in operations.values():
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)

    @staticmethod
    def tests_exactly(family: type[ForeignFamily], type_names: tuple[str, ...]) -> bool:
        return {
            f"{member.__module__}.{member.__qualname__}" for member in family.members
        } == set(type_names)

    @staticmethod
    def checked_operation(
        snapshot: CodemodSelectorContext,
        file_path: str,
        symbol: str,
        family: str,
    ) -> RouteForeignFamilyFilterOperation:
        """The operation for one function, proved against current source."""

        (target_id,) = SourceIndexTargetSelector.for_function_or_method(
            file_path, symbol
        ).target_ids(snapshot)
        operation = RouteForeignFamilyFilterOperation(
            target=SourceRewriteTarget(target_id=target_id), family=family
        )
        operation.source_edits_from_snapshot(snapshot.execution_snapshot())
        return operation


def family_path(family: type[ForeignFamily]) -> str:
    return f"{family.__module__}.{family.__qualname__}"
