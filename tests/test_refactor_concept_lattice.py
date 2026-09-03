from __future__ import annotations

from pathlib import Path

import pytest

import nominal_refactor_advisor as advisor
from nominal_refactor_advisor import codemod
from nominal_refactor_advisor import codemod_workflow
from nominal_refactor_advisor import detectors
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
        codemod.AutoRegisterMroOrderingConcept,
    }
)

EXPECTED_EXECUTABLE_CONCEPTS = {
    codemod.RepeatedBuilderSourceProjectionAuthorityMethod: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.RepeatedBuilderAuthorityMethod: (codemod.ConstructorKwargCollapseConcept),
    codemod.ManualClassRegistrationFindingRecipeSynthesizer: (
        codemod.AutoRegisterClassRegistryConcept
    ),
    codemod.RegistrationSemanticMirrorRecipeStrategy: (
        codemod.AutoRegisterClassRegistryConcept
    ),
    codemod.ClassFamilyCollectionSemanticMirrorRecipeBuilder: (
        codemod.ClassFamilyAuthorityConcept
    ),
    codemod.ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer: (
        codemod.ClassFamilyAuthorityConcept
    ),
    codemod.DataclassPayloadProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassFieldNameCollectionProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder: (
        codemod.DataclassPayloadProjectionConcept
    ),
    codemod.DataclassConstructorProjectionMappingRecipeBuilder: (
        codemod.ConstructorKwargCarrierProjectionConcept
    ),
    codemod.NumericLiteralDispatchFindingRecipeSynthesizer: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
    codemod.InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer: (
        codemod.AutoRegisterConcept
    ),
    codemod.AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer: (
        codemod.AutoRegisterMroOrderingConcept
    ),
    detectors.ManualClassRegistrationDetector: (
        codemod.AutoRegisterClassRegistryConcept
    ),
    detectors.ExactLeafMethodAncestorPromotionDetector: (
        codemod.ClassFamilyAuthorityConcept
    ),
    detectors.NumericLiteralDispatchDetector: (
        codemod.AutoRegisterStrategyFamilyConcept
    ),
    detectors.InheritedAutoRegisterConfigBoilerplateDetector: (
        codemod.AutoRegisterConcept
    ),
    detectors.AutoRegisterExplicitPriorityOrderingDetector: (
        codemod.AutoRegisterMroOrderingConcept
    ),
    detectors.ClosedParameterConveyorDetector: codemod.SemanticCarrierConcept,
    detectors.DeclaredCarrierExpansionDetector: codemod.SemanticCarrierConcept,
    codemod.EnumSubsetSemanticMirrorRecipeBuilder: (codemod.DerivedProjectionConcept),
    codemod.AutoregisterInstanceViewRecipeBuilder: (
        codemod.AutoRegisterClassRegistryConcept
    ),
}

EXPECTED_INFERRED_MAPPING_DECLARATIONS = frozenset(
    {
        codemod.DataclassConstructorProjectionMappingRecipeBuilder,
        codemod.DataclassFieldNameCollectionProjectionMappingRecipeBuilder,
        codemod.DataclassKeyValueSequenceProjectionMappingRecipeBuilder,
        codemod.DataclassPayloadProjectionMappingRecipeBuilder,
        codemod.EnumSubsetSemanticMirrorRecipeBuilder,
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


def test_detector_declarations_own_executable_synthesis_through_mro() -> None:
    executable_bindings = (
        (
            detectors.AutoRegisterExplicitPriorityOrderingDetector,
            codemod.AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer,
        ),
        (
            detectors.AutoRegisterMetaUnderRentedDetector,
            codemod.AutoRegisterMetaUnderRentedFindingRecipeSynthesizer,
        ),
        (
            detectors.ClosedParameterConveyorDetector,
            detectors.ClosedParameterConveyorDetector,
        ),
        (
            detectors.DeclaredCarrierExpansionDetector,
            detectors.DeclaredCarrierExpansionDetector,
        ),
        (
            detectors.EnvironmentBooleanAuthorityDriftDetector,
            codemod.EnvironmentBooleanAuthorityDriftFindingRecipeSynthesizer,
        ),
        (
            detectors.ExactLeafMethodAncestorPromotionDetector,
            codemod.ExactLeafMethodAncestorPromotionFindingRecipeSynthesizer,
        ),
        (
            detectors.InheritedAutoRegisterConfigBoilerplateDetector,
            codemod.InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer,
        ),
        (
            detectors.ManualClassRegistrationDetector,
            codemod.ManualClassRegistrationFindingRecipeSynthesizer,
        ),
        (
            detectors.NumericLiteralDispatchDetector,
            codemod.NumericLiteralDispatchFindingRecipeSynthesizer,
        ),
        (
            detectors.RepeatedBuilderCallDetector,
            codemod.RepeatedBuilderCallFindingRecipeSynthesizer,
        ),
    )

    assert "__registry__" not in vars(codemod.FindingRecipeSynthesizer)
    for detector_type, synthesis_type in executable_bindings:
        detector_id = detector_type.effective_detector_id()
        assert detector_id is not None
        assert issubclass(detector_type, codemod.FindingRecipeSynthesizer)
        assert synthesis_type.detector_declaration_type() is detector_type
        finding = advisor.RefactorFinding(
            detector_id=detector_id,
            pattern_id=advisor.PatternId.NOMINAL_BOUNDARY,
            title="Executable detector declaration",
            summary="The detector declaration owns synthesis through its MRO.",
            why="A separate synthesizer registry would mirror detector identity.",
            capability_gap="one nominal executable detector declaration",
            relation_context="detector identity and synthesis share one leaf",
        )
        assert type(codemod.FindingRecipeSynthesizer.for_finding(finding)) is (
            detector_type
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
                        codemod.SemanticCarrierConcept,
                        codemod.ConstructorKwargCollapseConcept,
                    codemod.ConstructorKwargCarrierProjectionConcept,
                    codemod.DataclassPayloadProjectionConcept,
                    codemod.DerivedProjectionConcept,
                    codemod.ClassFamilyAuthorityConcept,
                    codemod.AutoRegisterConcept,
                    codemod.AutoRegisterClassRegistryConcept,
                    codemod.AutoRegisterStrategyFamilyConcept,
                    codemod.AutoRegisterMroOrderingConcept,
                }
                ),
            ),
            (
                codemod.SemanticCarrierConcept,
                frozenset(
                    {
                        codemod.SemanticCarrierConcept,
                        codemod.ConstructorKwargCollapseConcept,
                        codemod.ConstructorKwargCarrierProjectionConcept,
                        codemod.DataclassPayloadProjectionConcept,
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
                    codemod.AutoRegisterMroOrderingConcept,
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
                    codemod.AutoRegisterMroOrderingConcept,
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
        codemod.AutoRegisterStrategyFamilyConcept,
    )


def test_nominal_boundary_does_not_select_unexecutable_ssot_detectors() -> None:
    finding = advisor.RefactorFinding(
        detector_id="constant_property_default_bundle",
        pattern_id=advisor.PatternId.AUTHORITATIVE_SCHEMA,
        title="Constant property defaults",
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
    assert not hasattr(
        codemod.MappingSemanticMirrorRecipeStrategy,
        "enum_subset_builder_for_finding",
    )
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


def test_metric_dispatch_uses_nominal_mro_priority() -> None:
    class SpecializedMappingMetrics(codemod.MappingMetrics):
        def semantic_fact_names(self) -> tuple[str, ...]:
            return ("specialized",)

    specialized_metrics = SpecializedMappingMetrics.from_field_names(
        mapping_site_count=2,
        field_names=("value",),
    )

    assert codemod.SemanticMirrorFindingRecipeStrategy.__registry__ == {
        codemod.RegistrationMetrics: codemod.RegistrationSemanticMirrorRecipeStrategy,
        codemod.MappingMetrics: codemod.MappingSemanticMirrorRecipeStrategy,
    }
    assert isinstance(
        codemod.SemanticMirrorFindingRecipeStrategy.strategy_for(specialized_metrics),
        codemod.MappingSemanticMirrorRecipeStrategy,
    )
    assert specialized_metrics.semantic_authority_name_candidates() == (None, None)
    assert specialized_metrics.semantic_fact_names() == ("specialized",)

    class SpecializedMappingStrategy(codemod.MappingSemanticMirrorRecipeStrategy):
        metric_type = SpecializedMappingMetrics

    try:
        assert (
            type(
                codemod.SemanticMirrorFindingRecipeStrategy.strategy_for(
                    specialized_metrics
                )
            )
            is SpecializedMappingStrategy
        )
    finally:
        del codemod.SemanticMirrorFindingRecipeStrategy.__registry__[
            SpecializedMappingMetrics
        ]
    assert not hasattr(codemod.SemanticMirrorFindingRecipeStrategy, "matches")


def test_class_assignment_recipe_metadata_is_owned_by_its_synthesizer() -> None:
    synthesizer_type = (
        codemod.InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer
    )

    assert synthesizer_type.recipe_id_suffix == "delete-inherited-autoregister-config"
    assert "already inherited" in synthesizer_type.recipe_reason
    assert issubclass(
        codemod.DeleteInheritedAutoRegisterConfigurationOperation,
        codemod.SourceReprovedOperation,
    )
    assert not hasattr(codemod, "ClassAssignmentDeletionPlan")
    assert not hasattr(codemod, "ClassAssignmentDeletionFindingRecipeSynthesizer")
    assert not hasattr(codemod, "RecipeMetadataAuthority")
    assert not hasattr(codemod, "SharedRecipeIdSuffixRecipeReasonBase")


def test_source_derived_synthesized_operations_share_one_reproof_contract() -> None:
    operation_types = (
        codemod.CollapseClosedParameterConveyorOperation,
        codemod.CollapseDeclaredCarrierExpansionOperation,
        codemod.CollapseRedundantClassAuthorityOperation,
        codemod.ConvertManualRegistryToAutoregisterOperation,
        codemod.DeleteInheritedAutoRegisterConfigurationOperation,
        codemod.DeriveAutoregisterInstanceViewOperation,
        codemod.DeriveAutoRegisterMroOrderingOperation,
        codemod.DeriveClassFamilyCollectionOperation,
        codemod.DeriveDataclassConstructorProjectionOperation,
        codemod.DeriveDataclassFieldNameCollectionProjectionOperation,
        codemod.DeriveDataclassKeyValueSequenceProjectionOperation,
        codemod.DeriveDataclassPayloadProjectionOperation,
        codemod.DeriveEnumSubsetOperation,
        codemod.DeriveRepeatedBuilderAuthorityOperation,
        codemod.DispatchToPolymorphismOperation,
        codemod.FactorParallelMirroredLeafFamilyOperation,
        codemod.PromoteExactLeafMethodsToAncestorOperation,
    )

    assert all(
        issubclass(operation_type, codemod.SourceReprovedOperation)
        for operation_type in operation_types
    )


def test_class_base_operations_share_source_proof_and_own_leaf_mutation() -> None:
    for operation_type in (
        codemod.AddClassBaseOperation,
        codemod.RemoveClassBaseOperation,
    ):
        assert issubclass(operation_type, codemod.ClassBaseMutationOperationABC)
        assert issubclass(operation_type, codemod.SourceReprovedOperation)
        assert "source_edits_from_snapshot" not in operation_type.__dict__
        assert "replacement_header_lines" in operation_type.__dict__
        assert "payload_value" not in operation_type.__dataclass_fields__
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == ("target", "rationale", "base_name")
    assert not hasattr(codemod, "BaseNamePayloadOperation")


def test_legacy_mirrored_method_extraction_payload_is_absent() -> None:
    assert not hasattr(codemod, "ExtractMethodsToClassOperation")
    assert "extract_methods_to_class" not in codemod.RefactorRecipeOperation.__registry__


def test_authority_source_payload_is_owned_by_its_operation_family() -> None:
    assert "payload_value" not in codemod.AuthoritySourceOperation.__dataclass_fields__
    assert tuple(
        binding.field_name
        for binding in codemod.AuthoritySourceOperation.payload_bindings()
    ) == ("target", "rationale", "authority_kind", "authority_source")
    assert "authority_claim" not in codemod.AuthoritySourceOperation.__dataclass_fields__


def test_module_symbol_move_derives_its_source_reexport() -> None:
    for operation_type in (
        codemod.MoveSymbolsToModuleOperation,
        codemod.ExtractSymbolsToNewModuleOperation,
        codemod.ExtractSymbolClosureToNewModuleOperation,
    ):
        assert issubclass(operation_type, codemod.ModuleSymbolMoveOperation)
        assert issubclass(operation_type, codemod.RepositorySourceReprovedOperation)
        assert (
            codemod.RefactorRecipeOperation.__registry__[
                operation_type.operation_key()
            ]
            is operation_type
        )
    assert tuple(
        binding.field_name
        for binding in codemod.MoveSymbolsToModuleOperation.payload_bindings()
    ) == ("target", "rationale", "destination_path", "symbol_qualnames")
    assert tuple(
        binding.field_name
        for binding in codemod.ExtractSymbolsToNewModuleOperation.payload_bindings()
    ) == (
        "target",
        "rationale",
        "destination_path",
        "symbol_qualnames",
        "destination_source",
    )
    assert tuple(
        binding.field_name
        for binding in codemod.ExtractSymbolClosureToNewModuleOperation.payload_bindings()
    ) == (
        "target",
        "rationale",
        "destination_path",
        "root_symbol_qualnames",
        "destination_source",
    )
    assert (
        "source_edits_from_snapshot"
        in codemod.MoveSymbolsToModuleOperation.__dict__
    )
    assert (
        "source_edits_from_snapshot"
        in codemod.NewModuleSymbolMoveOperationABC.__dict__
    )
    assert (
        "move_symbol_qualnames"
        in codemod.ExplicitModuleSymbolSelectionOperationABC.__dict__
    )
    assert (
        "move_symbol_qualnames"
        in codemod.DependencyClosureModuleSymbolSelectionOperationABC.__dict__
    )
    assert "move_plan" in codemod.ModuleSymbolMoveOperation.__dict__
    assert "move_plan" not in codemod.MoveSymbolsToModuleOperation.__dict__
    assert "move_plan" not in codemod.ExtractSymbolsToNewModuleOperation.__dict__
    assert (
        "move_plan"
        not in codemod.ExtractSymbolClosureToNewModuleOperation.__dict__
    )
    assert not hasattr(codemod, "MovedSymbolImportPolicy")
    assert not hasattr(codemod, "ReplacementImportPayloadValueCodec")


def test_edit_payloads_are_owned_by_their_semantic_operations() -> None:
    for operation_type, field_name in (
        (codemod.EnsureImportOperation, "import_source"),
        (codemod.ReplaceFunctionSignatureOperation, "signature_suffix"),
        (codemod.ReplaceFunctionBodyOperation, "body_source"),
    ):
        assert "payload_value" not in operation_type.__dataclass_fields__
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == ("target", "rationale", field_name)


def test_function_mutations_share_typed_current_source_proof() -> None:
    for operation_type in (
        codemod.ReplaceFunctionSignatureOperation,
        codemod.ReplaceFunctionBodyOperation,
    ):
        assert issubclass(operation_type, codemod.FunctionMutationOperationABC)
        assert issubclass(operation_type, codemod.SourceReprovedOperation)
        assert "source_edits_from_snapshot" not in operation_type.__dict__
        assert "source_edits_for_function" in operation_type.__dict__


def test_source_payload_operations_share_the_source_declaration() -> None:
    operation_types = (
        codemod.CreateFileOperation,
        codemod.ReplaceModuleAssignmentOperation,
        codemod.InsertAfterImportsOperation,
    )

    assert not hasattr(codemod, "StringPayloadOperation")
    for operation_type in operation_types:
        assert issubclass(operation_type, codemod.SourcePayloadOperation)
        source_binding = next(
            binding
            for binding in operation_type.payload_bindings()
            if binding.constructor_argument_name == "source"
        )
        assert source_binding.field_name == "source"
    assert tuple(
        binding.field_name
        for binding in codemod.ReplaceModuleAssignmentOperation.payload_bindings()
    ) == ("target", "rationale", "source")
    assert "assignment_name" not in (
        codemod.ReplaceModuleAssignmentOperation.__dataclass_fields__
    )


def test_target_adjacent_insertions_share_source_proof_and_own_geometry() -> None:
    for operation_type in (
        codemod.InsertBeforeTargetOperation,
        codemod.InsertAfterTargetOperation,
    ):
        assert issubclass(
            operation_type,
            codemod.TargetAdjacentInsertionOperationABC,
        )
        assert issubclass(operation_type, codemod.SourceReprovedOperation)
        assert "source_edits_from_snapshot" not in operation_type.__dict__
        assert "insertion_line" in operation_type.__dict__
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == ("target", "rationale", "source")


def test_assignment_deletions_share_validated_source_reproof() -> None:
    for operation_type in (
        codemod.DeleteClassAssignmentsOperation,
        codemod.DeleteModuleAssignmentsOperation,
    ):
        assert issubclass(operation_type, codemod.AssignmentDeletionOperationABC)
        assert issubclass(operation_type, codemod.SourceReprovedOperation)
        assert tuple(
            binding.field_name for binding in operation_type.payload_bindings()
        ) == ("target", "rationale", "assignment_names")
    assert not hasattr(codemod, "AssignmentNamesPayloadOperation")
    assert not hasattr(codemod, "TargetNodeRecipeOperationMixin")


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
        "@dataclass(frozen=True)\n"
        "class PlanSource:\n"
        "    pose_id: str\n"
        "    score: float\n"
        "    theorem_handles: tuple[str, ...]\n\n\n"
        "def alpha(candidate: PlanSource):\n"
        "    return RuntimePlan(\n"
        "        pose_id=candidate.pose_id,\n"
        "        score=candidate.score,\n"
        "        theorem_handles=tuple(candidate.theorem_handles),\n"
        "    )\n\n\n"
        "def beta(entry: PlanSource):\n"
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


def test_target_shape_and_selector_mirror_authorities_are_absent() -> None:
    assert not hasattr(advisor, "RefactorRecipeTargetShape")
    assert not hasattr(codemod, "RefactorRecipeTargetShape")
    assert "target_shape" not in codemod.RefactorRecipe.__dataclass_fields__
    assert not hasattr(advisor, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "CodemodRefactorGoalFindingSelector")
    assert not hasattr(codemod_workflow, "SelectorBackedRefactorGoalTargetPolicy")
