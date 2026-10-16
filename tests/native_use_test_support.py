"""Explicit supported-execution premises shared by authored codemod fixtures."""

from dataclasses import replace

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_native_requirements import (
    DeclaredNativeUseInvariants,
)
from nominal_refactor_advisor.codemod_runtime import CodemodPlanDocumentSimulation
from nominal_refactor_advisor.codemod_semantics import CodemodBackend


def with_rendered_registry_creator_support(
    operation: ConvertManualRegistryToAutoregisterOperation,
    preview: CodemodPlanDocumentSimulation,
) -> ConvertManualRegistryToAutoregisterOperation:
    """Accept the exact rendered creator use for a controlled runtime fixture."""

    return replace(
        operation,
        supported_execution=DeclaredNativeUseInvariants.from_requirements(
            operation.candidate_native_use_requirements(preview),
            rationale=(
                "The controlled fixture retains AutoRegisterMeta behavior at the "
                "rendered use."
            ),
        ),
    )


def simulate_with_rendered_registry_creator_support(
    operation: ConvertManualRegistryToAutoregisterOperation,
    snapshot: CodemodSourceSnapshot,
    *,
    recipe_id: str,
    backend: CodemodBackend | None = None,
) -> CodemodPlanDocumentSimulation:
    """Preview, bind the candidate receipt, then re-simulate the same fixture."""

    preview = CodemodPlanDocument(
        recipes=(RefactorRecipe(recipe_id, operations=(operation,)),)
    ).simulate(snapshot, backend=backend)
    supported = with_rendered_registry_creator_support(operation, preview)
    return CodemodPlanDocument(
        recipes=(RefactorRecipe(recipe_id, operations=(supported,)),)
    ).simulate(snapshot, backend=backend)
