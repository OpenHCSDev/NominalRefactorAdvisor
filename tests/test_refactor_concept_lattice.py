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
        codemod.CallMappingAuthorityConcept,
        codemod.ConstructorKwargCollapseConcept,
        codemod.ConstructorKwargCarrierProjectionConcept,
        codemod.TupleDictReturnNominalizationConcept,
        codemod.DataclassPayloadProjectionConcept,
        codemod.DerivedProjectionConcept,
        codemod.ClassFamilyAuthorityConcept,
        codemod.AutoRegisterConcept,
        codemod.AutoRegisterClassRegistryConcept,
        codemod.AutoRegisterStrategyFamilyConcept,
        codemod.RoleCaseAuthorityConcept,
    }
)

EXPECTED_EXECUTABLE_CONCEPTS = {
    codemod.RepeatedBuilderSourceProjectionAuthorityMethod: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.RepeatedBuilderAuthorityMethod: (codemod.ConstructorKwargCollapseConcept),
    codemod.RepeatedMethodCallAuthorityRecipeParts: (
        codemod.CallMappingAuthorityConcept
    ),
    codemod.ManualClassRegistrationFindingRecipeSynthesizer: (
        codemod.AutoRegisterClassRegistryConcept
    ),
    codemod.ClassFamilyCollectionSemanticMirrorRecipeBuilder: (
        codemod.ClassFamilyAuthorityConcept
    ),
    codemod.RepeatedMethodPromotionFindingRecipeSynthesizer: (
        codemod.ClassFamilyAuthorityConcept
    ),
    codemod.DataclassPayloadProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassConstructorProjectionMappingRecipeBuilder: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.LocalRoleCaseLogicMappingRecipeBuilder: (codemod.RoleCaseAuthorityConcept),
    codemod.NumericLiteralDispatchFindingRecipeSynthesizer: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
    codemod.InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer: (
        codemod.AutoRegisterConcept
    ),
    codemod.EnumSubsetSemanticMirrorRecipeBuilder: (codemod.DerivedProjectionConcept),
    codemod.AutoregisterInstanceViewRecipeBuilder: (
        codemod.AutoRegisterClassRegistryConcept
    ),
}

EXPECTED_INFERRED_MAPPING_DECLARATIONS = frozenset(
    {
        codemod.DataclassConstructorProjectionMappingRecipeBuilder,
        codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder,
        codemod.DataclassPayloadProjectionMappingRecipeBuilder,
    }
)

EXPECTED_MAPPING_DECLARATIONS = EXPECTED_INFERRED_MAPPING_DECLARATIONS


def test_concept_taxonomy_is_derived_without_a_parallel_registry() -> None:
    assert frozenset(codemod.RefactorConcept.declaration_types()) == (
        EXPECTED_CONCEPT_DECLARATIONS
    )
    assert "__registry__" not in codemod.RefactorConcept.__dict__
    assert all(
        "__registry__" not in declaration_type.__dict__
        for declaration_type in EXPECTED_CONCEPT_DECLARATIONS
    )
    assert not hasattr(codemod_workflow, "CodemodRefactorGoal")
    assert not hasattr(codemod_workflow, "CodemodRefactorGoalStageAttempt")
    assert "matches_finding" not in vars(codemod.NominalBoundaryConcept)
    assert (
        codemod.NominalBoundaryConcept.matches_finding.__func__
        is codemod.RefactorConcept.matches_finding.__func__
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
            codemod.NominalBoundaryConcept,
            frozenset(
                {
                    codemod.CallMappingAuthorityConcept,
                    codemod.ConstructorKwargCollapseConcept,
                    codemod.ConstructorKwargCarrierProjectionConcept,
                    codemod.DataclassPayloadProjectionConcept,
                    codemod.DerivedProjectionConcept,
                    codemod.ClassFamilyAuthorityConcept,
                    codemod.AutoRegisterConcept,
                    codemod.AutoRegisterClassRegistryConcept,
                    codemod.AutoRegisterStrategyFamilyConcept,
                    codemod.RoleCaseAuthorityConcept,
                }
            ),
        ),
        (
            codemod.ConstructorKwargCollapseConcept,
            frozenset(
                {
                    codemod.ConstructorKwargCollapseConcept,
                    codemod.ConstructorKwargCarrierProjectionConcept,
                }
            ),
        ),
        (
            codemod.CallMappingAuthorityConcept,
            frozenset(
                {
                    codemod.CallMappingAuthorityConcept,
                    codemod.ConstructorKwargCollapseConcept,
                    codemod.ConstructorKwargCarrierProjectionConcept,
                }
            ),
        ),
        (
            codemod.TupleDictReturnNominalizationConcept,
            frozenset(
                {
                    codemod.DataclassPayloadProjectionConcept,
                }
            ),
        ),
        (
            codemod.ClassFamilyAuthorityConcept,
            frozenset(
                {
                    codemod.ClassFamilyAuthorityConcept,
                    codemod.AutoRegisterConcept,
                    codemod.AutoRegisterClassRegistryConcept,
                    codemod.AutoRegisterStrategyFamilyConcept,
                }
            ),
        ),
        (
            codemod.AutoRegisterConcept,
            frozenset(
                {
                    codemod.AutoRegisterConcept,
                    codemod.AutoRegisterClassRegistryConcept,
                    codemod.AutoRegisterStrategyFamilyConcept,
                }
            ),
        ),
        (
            codemod.DerivedProjectionConcept,
            frozenset({codemod.DerivedProjectionConcept}),
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
        codemod.NumericLiteralDispatchFindingRecipeSynthesizer,
        codemod.SemanticCarrierConcept,
    )
    assert not issubclass(
        codemod.DataclassPayloadProjectionMappingRecipeBuilder,
        codemod.RoleCaseAuthorityConcept,
    )


def test_nominal_boundary_does_not_select_unexecutable_ssot_detectors() -> None:
    finding = advisor.RefactorFinding(
        detector_id="constant_property_hooks",
        pattern_id=advisor.PatternId.AUTHORITATIVE_SCHEMA,
        title="Constant property hooks",
        summary="An SSOT detector without an executable declaration.",
        why="Detector roles do not prove executable migration ownership.",
        capability_gap="A proof-backed executable declaration.",
        relation_context="An unsupported SSOT detector must not enter the lattice.",
    )
    snapshot = codemod.CodemodSourceSnapshot.from_modules((), (finding,))

    assert not codemod.NominalBoundaryConcept.matches_finding(finding, snapshot)


def test_mapping_builder_identity_is_nominally_owned() -> None:
    assert (
        frozenset(codemod.InferredSemanticMirrorMappingRecipeBuilder.builder_types())
        == EXPECTED_INFERRED_MAPPING_DECLARATIONS
    )
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
    assert all(
        builder_type.registry_key == registry_key
        for registry_key, builder_type in (
            codemod.ContextualSemanticMirrorRecipeBuilder.__registry__.items()
        )
    )


def test_semantic_mirror_strategy_identity_is_metric_type_derived() -> None:
    assert codemod.SemanticMirrorFindingRecipeStrategy.__registry__ == {
        codemod.RegistrationMetrics: codemod.RegistrationSemanticMirrorRecipeStrategy,
        codemod.MappingMetrics: codemod.MappingSemanticMirrorRecipeStrategy,
        codemod.BranchCountMetrics: codemod.BranchSemanticMirrorRecipeStrategy,
    }
    assert not hasattr(codemod.SemanticMirrorFindingRecipeStrategy, "matches")


def test_class_assignment_recipe_metadata_is_owned_by_its_synthesizer() -> None:
    synthesizer_type = (
        codemod.InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer
    )

    assert synthesizer_type.recipe_id_suffix == "delete-inherited-autoregister-config"
    assert "already inherited" in synthesizer_type.recipe_reason
    assert "action_keys" not in codemod.ClassAssignmentDeletionPlan.__dataclass_fields__
    assert not hasattr(codemod, "RecipeMetadataAuthority")
    assert not hasattr(codemod, "SharedRecipeIdSuffixRecipeReasonBase")


def test_class_base_operations_own_the_base_name_payload() -> None:
    for operation_type in (
        codemod.AddClassBaseOperation,
        codemod.RemoveClassBaseOperation,
    ):
        assert issubclass(operation_type, codemod.BaseNamePayloadOperation)
        assert "payload_value" not in operation_type.__dataclass_fields__
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == ("base_name",)


def test_authority_source_payload_is_owned_by_its_operation_family() -> None:
    assert "payload_value" not in codemod.AuthoritySourceOperation.__dataclass_fields__
    assert tuple(
        binding.field_name
        for binding in codemod.AuthoritySourceOperation.payload_bindings()
    ) == ("authority_source",)


def test_edit_payloads_are_owned_by_their_semantic_operations() -> None:
    for operation_type, field_name in (
        (codemod.EnsureImportOperation, "import_source"),
        (codemod.ReplaceFunctionSignatureOperation, "signature_source"),
        (codemod.ReplaceFunctionBodyOperation, "body_source"),
    ):
        assert "payload_value" not in operation_type.__dataclass_fields__
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == (field_name,)


def test_source_payload_operations_share_the_source_declaration() -> None:
    expected_bindings = {
        codemod.CreateFileOperation: ("source",),
        codemod.ReplaceModuleAssignmentOperation: ("assignment_name", "source"),
        codemod.InsertBeforeTargetOperation: ("source",),
        codemod.InsertAfterTargetOperation: ("source",),
        codemod.InsertAfterImportsOperation: ("source",),
    }

    assert not hasattr(codemod, "StringPayloadOperation")
    for operation_type, field_names in expected_bindings.items():
        assert issubclass(operation_type, codemod.SourcePayloadOperation)
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == field_names


def test_registered_mapping_cases_publish_no_numeric_precedence() -> None:
    mapping_declarations = (
        codemod.MappingSemanticMirrorRecipeBuilder,
        *EXPECTED_MAPPING_DECLARATIONS,
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


def _repeated_builder_evaluation_for_source(
    tmp_path: Path,
    source: str,
) -> codemod.FindingRecipeEvaluation:
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
    return synthesizer.evaluate_recipe_for_finding(findings[0], snapshot)


def test_repeated_builder_dynamic_rule_preserves_the_exact_concept_leaf(
    tmp_path: Path,
) -> None:
    evaluation = _repeated_builder_evaluation_for_source(
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

    declaration = evaluation.required_executable_declaration_type
    assert declaration is codemod.RepeatedBuilderSourceProjectionAuthorityMethod
    assert (
        codemod.RefactorConcept.leaf_concept_for_declaration(declaration)
        is codemod.ConstructorKwargCarrierProjectionConcept
    )


def test_repeated_builder_nonconcept_rule_does_not_inherit_a_carrier_leaf(
    tmp_path: Path,
) -> None:
    evaluation = _repeated_builder_evaluation_for_source(
        tmp_path,
        "class Builder:\n"
        "    def main(self):\n"
        '        self.register("--json", action="store_true", help="JSON")\n'
        '        self.register("--plans", action="store_true", help="Plans")\n'
        '        self.register("--workers", type=int, default=3, help="Workers")\n'
        '        self.register("--exclude", action="append", default=[], help="Exclude")\n'
        "        return self\n",
    )

    declaration = evaluation.required_executable_declaration_type
    assert declaration is codemod.RepeatedBuilderCallFindingRecipeSynthesizer
    assert not issubclass(declaration, codemod.RefactorConcept)


def test_repeated_owner_method_calls_publish_call_mapping_authority(
    tmp_path: Path,
) -> None:
    evaluation = _repeated_builder_evaluation_for_source(
        tmp_path,
        "class Renderer:\n"
        "    def emit(\n"
        "        self, name: str, enabled: bool = False, style: str = 'plain'\n"
        "    ) -> str:\n"
        "        return name if enabled else f'{style}:{name.lower()}'\n\n"
        "    def build(self):\n"
        "        first = self.emit(name='alpha')\n"
        "        second = self.emit(name='beta', enabled=True)\n"
        "        third = self.emit(name='gamma', style='compact')\n"
        "        fourth = self.emit(name='delta', enabled=True, style='wide')\n"
        "        return first, second, third, fourth\n",
    )

    declaration = evaluation.required_executable_declaration_type
    assert evaluation.status.planned
    assert declaration is codemod.RepeatedMethodCallAuthorityRecipeParts
    assert evaluation.refactor_concept_type is codemod.CallMappingAuthorityConcept


def test_target_shape_and_selector_mirror_authorities_are_absent() -> None:
    assert not hasattr(advisor, "RefactorRecipeTargetShape")
    assert not hasattr(codemod, "RefactorRecipeTargetShape")
    assert "target_shape" not in codemod.RefactorRecipe.__dataclass_fields__
    assert not hasattr(advisor, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "SelectorBackedRefactorGoalTargetPolicy")
