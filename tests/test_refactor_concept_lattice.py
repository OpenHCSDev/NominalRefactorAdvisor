from __future__ import annotations

from pathlib import Path

import pytest

import nominal_refactor_advisor as advisor
from nominal_refactor_advisor import codemod
from nominal_refactor_advisor import codemod_workflow
from nominal_refactor_advisor import semantic_match
from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import parse_python_modules

EXPECTED_CONCEPT_DECLARATIONS = frozenset(
    {
        codemod.RefactorConcept,
        codemod.NominalBoundaryConcept,
        codemod.SemanticCarrierConcept,
        codemod.PrefixBundleCarrierConcept,
        codemod.DataclassInheritanceLiftConcept,
        codemod.ConstructorKwargCollapseConcept,
        codemod.ConstructorKwargCarrierProjectionConcept,
        codemod.DataclassContextCallProjectionConcept,
        codemod.TupleDictReturnNominalizationConcept,
        codemod.DataclassPayloadProjectionConcept,
        codemod.TupleDictReturnRecordConcept,
        codemod.DeadCompatibilityErasureConcept,
        codemod.AutoRegisterConcept,
        codemod.AutoRegisterClassRegistryConcept,
        codemod.AutoRegisterStrategyFamilyConcept,
        codemod.RoleCaseAuthorityConcept,
    }
)

EXPECTED_EXECUTABLE_CONCEPTS = {
    codemod.RuntimeProductRecordSchemaFindingRecipeSynthesizer: (
        codemod.TupleDictReturnRecordConcept
    ),
    codemod.FlattenedProjectionPropertyFindingRecipeSynthesizer: (
        codemod.DeadCompatibilityErasureConcept
    ),
    codemod.RepeatedFieldFamilyFindingRecipeSynthesizer: (
        codemod.DataclassInheritanceLiftConcept
    ),
    codemod.ExistingNominalAuthorityReuseFindingRecipeSynthesizer: (
        codemod.DataclassInheritanceLiftConcept
    ),
    codemod.PrefixedRoleBundleFindingRecipeSynthesizer: (
        codemod.PrefixBundleCarrierConcept
    ),
    codemod.ParallelPrimitiveCarrierFindingRecipeSynthesizer: (
        codemod.PrefixBundleCarrierConcept
    ),
    codemod.RepeatedBuilderSourceProjectionAuthorityMethod: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.ManualClassRegistrationFindingRecipeSynthesizer: (
        codemod.AutoRegisterClassRegistryConcept
    ),
    codemod.DataclassPayloadProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.SemanticDictBagReturnRecordMappingRecipeBuilder: (
        codemod.TupleDictReturnRecordConcept
    ),
    codemod.SemanticTupleReturnRecordMappingRecipeBuilder: (
        codemod.TupleDictReturnRecordConcept
    ),
    codemod.DataclassConstructorProjectionMappingRecipeBuilder: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.DataclassContextCallProjectionMappingRecipeBuilder: (
        codemod.DataclassContextCallProjectionConcept
    ),
    codemod.LocalRoleCaseLogicMappingRecipeBuilder: (codemod.RoleCaseAuthorityConcept),
    codemod.StringDispatchFindingRecipeSynthesizer: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
    codemod.NumericLiteralDispatchFindingRecipeSynthesizer: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
    codemod.InlineLiteralDispatchFindingRecipeSynthesizer: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
}

EXPECTED_INFERRED_MAPPING_DECLARATIONS = frozenset(
    {
        codemod.DataclassConstructorProjectionMappingRecipeBuilder,
        codemod.DataclassContextCallProjectionMappingRecipeBuilder,
        codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder,
        codemod.DataclassPayloadProjectionMappingRecipeBuilder,
    }
)

EXPECTED_DECLARED_MAPPING_BRIDGES = {
    codemod.SemanticDictBagFindingRecipeSynthesizer: (
        codemod.SemanticDictBagReturnRecordMappingRecipeBuilder
    ),
    codemod.SemanticTupleReturnRecordFindingRecipeSynthesizer: (
        codemod.SemanticTupleReturnRecordMappingRecipeBuilder
    ),
}

EXPECTED_MAPPING_DECLARATIONS = frozenset(
    {
        *EXPECTED_INFERRED_MAPPING_DECLARATIONS,
        *EXPECTED_DECLARED_MAPPING_BRIDGES.values(),
    }
)


def test_concept_taxonomy_is_derived_without_a_parallel_registry() -> None:
    assert frozenset(codemod.RefactorConcept.declaration_types()) == (
        EXPECTED_CONCEPT_DECLARATIONS
    )
    assert "__registry__" not in codemod.RefactorConcept.__dict__
    assert all(
        "__registry__" not in declaration_type.__dict__
        for declaration_type in EXPECTED_CONCEPT_DECLARATIONS
    )


def test_every_migrated_executable_declaration_has_one_intended_leaf() -> None:
    assert {
        declaration_type: codemod.RefactorConcept.leaf_concept_for_declaration(
            declaration_type
        )
        for declaration_type in EXPECTED_EXECUTABLE_CONCEPTS
    } == EXPECTED_EXECUTABLE_CONCEPTS


@pytest.mark.parametrize(
    ("parent_concept", "expected_leaves"),
    (
        (
            codemod.ConstructorKwargCollapseConcept,
            frozenset(
                {
                    codemod.ConstructorKwargCarrierProjectionConcept,
                    codemod.DataclassContextCallProjectionConcept,
                }
            ),
        ),
        (
            codemod.TupleDictReturnNominalizationConcept,
            frozenset(
                {
                    codemod.DataclassPayloadProjectionConcept,
                    codemod.TupleDictReturnRecordConcept,
                }
            ),
        ),
        (
            codemod.AutoRegisterConcept,
            frozenset(
                {
                    codemod.AutoRegisterClassRegistryConcept,
                    codemod.AutoRegisterStrategyFamilyConcept,
                }
            ),
        ),
    ),
)
def test_parent_concepts_match_descendants_by_mro(
    parent_concept: type[codemod.RefactorConcept],
    expected_leaves: frozenset[type[codemod.RefactorConcept]],
) -> None:
    matching_leaves = {
        leaf_concept
        for declaration_type, leaf_concept in EXPECTED_EXECUTABLE_CONCEPTS.items()
        if issubclass(declaration_type, parent_concept)
    }

    assert matching_leaves == expected_leaves


def test_unrelated_concepts_do_not_match() -> None:
    assert not issubclass(
        codemod.RuntimeProductRecordSchemaFindingRecipeSynthesizer,
        codemod.AutoRegisterConcept,
    )
    assert not issubclass(
        codemod.StringDispatchFindingRecipeSynthesizer,
        codemod.SemanticCarrierConcept,
    )
    assert not issubclass(
        codemod.DataclassPayloadProjectionMappingRecipeBuilder,
        codemod.RoleCaseAuthorityConcept,
    )


def test_mapping_builder_identity_is_nominal_or_bridge_owned() -> None:
    assert frozenset(
        codemod.InferredSemanticMirrorMappingRecipeBuilder.builder_types()
    ) == EXPECTED_INFERRED_MAPPING_DECLARATIONS
    assert {
        synthesizer_type: synthesizer_type.builder_type
        for synthesizer_type in EXPECTED_DECLARED_MAPPING_BRIDGES
    } == EXPECTED_DECLARED_MAPPING_BRIDGES
    assert "__registry__" not in codemod.MappingSemanticMirrorRecipeBuilder.__dict__
    assert all(
        not hasattr(builder_type, "mapping_name")
        for builder_type in EXPECTED_MAPPING_DECLARATIONS
    )
    assert all(
        codemod.RefactorConcept.leaf_concept_for_declaration(builder_type)
        is EXPECTED_EXECUTABLE_CONCEPTS[builder_type]
        for builder_type in EXPECTED_MAPPING_DECLARATIONS
    )


def test_semantic_mirror_strategy_identity_is_metric_type_derived() -> None:
    assert codemod.SemanticMirrorFindingRecipeStrategy.__registry__ == {
        codemod.RegistrationMetrics: codemod.RegistrationSemanticMirrorRecipeStrategy,
        codemod.MappingMetrics: codemod.MappingSemanticMirrorRecipeStrategy,
        codemod.BranchCountMetrics: codemod.BranchSemanticMirrorRecipeStrategy,
    }
    assert not hasattr(
        codemod.TypedMetricSemanticMirrorRecipeStrategy,
        "matches",
    )


def test_registered_mapping_and_unpack_cases_publish_no_numeric_precedence() -> None:
    mapping_declarations = (
        codemod.MappingSemanticMirrorRecipeBuilder,
        *EXPECTED_MAPPING_DECLARATIONS,
    )
    unpack_declarations = (
        codemod.TupleReturnUnpackValueMatcher,
        *codemod.TupleReturnUnpackValueMatcher.__registry__.values(),
    )
    contextual_registration_declarations = (
        codemod.ContextualSemanticMirrorRecipeBuilder,
        *codemod.ContextualSemanticMirrorRecipeBuilder.__registry__.values(),
    )

    assert all(
        not hasattr(declaration, "registration_order")
        for declaration in mapping_declarations
    )
    assert all(
        not hasattr(declaration, "registry_order")
        for declaration in unpack_declarations
    )
    assert all(
        not hasattr(declaration, "registry_order")
        for declaration in contextual_registration_declarations
    )
    assert not hasattr(
        codemod.RegistrationSemanticMirrorRecipeStrategy,
        "manual_registration_order",
    )


def test_semantic_match_families_publish_no_registry_or_numeric_precedence() -> None:
    assert not hasattr(semantic_match.EffectStep, "__registry__")
    assert not hasattr(semantic_match.EffectStep, "registration_order")
    assert not hasattr(semantic_match.AstPredicateRule, "__registry__")
    assert not hasattr(semantic_match.AstPredicateRule, "rule_order")


def _repeated_builder_declaration_for_source(
    tmp_path: Path,
    source: str,
) -> tuple[type[object], ...]:
    module_path = tmp_path / "pkg" / "mod.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(source, encoding="utf-8")
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == "repeated_builder_calls"
    )
    assert len(findings) == 1
    snapshot = codemod.CodemodSourceSnapshot.from_modules(modules, findings)
    synthesizer = codemod.FindingRecipeSynthesizer.for_finding(findings[0])
    assert synthesizer is not None
    evaluation = synthesizer.declared_evaluation_for_finding(findings[0], snapshot)
    return (evaluation.required_executable_declaration_type,)


def test_repeated_builder_dynamic_rule_preserves_the_exact_concept_leaf(
    tmp_path: Path,
) -> None:
    declarations = _repeated_builder_declaration_for_source(
        tmp_path,
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class RuntimePlan:\n"
        "    pose_id: str\n"
        "    score: float\n"
        "    theorem_handles: tuple[str, ...]\n\n\n"
        "def alpha(candidate):\n"
        "    return RuntimePlan(\n"
        "        pose_id=candidate.pose_id,\n"
        "        score=candidate.score,\n"
        "        theorem_handles=tuple(candidate.theorem_handles),\n"
        "    )\n\n\n"
        "def beta(entry):\n"
        "    return RuntimePlan(\n"
        "        pose_id=entry.pose_id,\n"
        "        score=entry.score,\n"
        "        theorem_handles=tuple(entry.theorem_handles),\n"
        "    )\n",
    )

    assert declarations == (codemod.RepeatedBuilderSourceProjectionAuthorityMethod,)
    assert (
        codemod.RefactorConcept.leaf_concept_for_declaration(declarations[0])
        is codemod.ConstructorKwargCarrierProjectionConcept
    )


def test_repeated_builder_nonconcept_rule_does_not_inherit_a_carrier_leaf(
    tmp_path: Path,
) -> None:
    declarations = _repeated_builder_declaration_for_source(
        tmp_path,
        "def main(builder):\n"
        '    builder.register("--json", action="store_true", help="JSON")\n'
        '    builder.register("--plans", action="store_true", help="Plans")\n'
        '    builder.register("--workers", type=int, default=3, help="Workers")\n'
        '    builder.register("--exclude", action="append", default=[], help="Exclude")\n'
        "    return builder\n",
    )

    assert declarations == (codemod.RepeatedBuilderCallFindingRecipeSynthesizer,)
    assert not issubclass(declarations[0], codemod.RefactorConcept)


def test_target_shape_and_selector_mirror_authorities_are_absent() -> None:
    assert not hasattr(advisor, "RefactorRecipeTargetShape")
    assert not hasattr(codemod, "RefactorRecipeTargetShape")
    assert "target_shape" not in codemod.RefactorRecipe.__dataclass_fields__
    assert not hasattr(advisor, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "SelectorBackedRefactorGoalTargetPolicy")
