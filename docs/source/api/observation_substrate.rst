Observation Substrate
=====================

AST And Registration Substrate
------------------------------

.. automodule:: nominal_refactor_advisor.ast_tools
   :members: ParsedModule, AutoRegisterMeta, ModuleShapeSpec, AutoRegisteredModuleShapeSpec, CollectedFamily, RegisteredSpecCollectedFamily, SingleSpecCollectedFamily, collect_family_items


Observation Families
--------------------

.. automodule:: nominal_refactor_advisor.observation_families
   :members: GeneratedFamilySpec, FamilyGeneratingSpec, ObservationFamily, ShapeFamily, TypedLiteralObservationFamily, family_for_item_type, family_for_literal_kind


Observation Graph
-----------------

.. automodule:: nominal_refactor_advisor.observation_graph
   :members: ObservationKind, StructuralExecutionLevel, StructuralObservation, StructuralObservationCarrier, ObservationFiber, NominalWitnessGroup, ObservationCohort, ObservationGraph, collect_structural_observations, build_observation_graph


Observation Shapes
------------------

.. automodule:: nominal_refactor_advisor.observation_shapes
   :members: LiteralKind, FieldOriginKind, StructuralObservationTemplate, FieldObservation, AttributeProbeObservation, LiteralDispatchObservation, ProjectionHelperShape, AccessorWrapperCandidate, ScopedShapeWrapperFunction, ScopedShapeWrapperSpec, ConfigDispatchObservation, ClassMarkerObservation, InterfaceGenerationObservation, SentinelTypeObservation, DynamicMethodInjectionObservation, RuntimeTypeGenerationObservation, LineageMappingObservation, DualAxisResolutionObservation, MethodShape, BuilderCallShape, ExportDictShape, RegistrationShape
