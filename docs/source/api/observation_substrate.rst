Observation Substrate
=====================

This page documents the internal observation and registration machinery that the
advisor uses to build structural evidence. Most users do not need this surface
to run the tool or consume emitted findings.

AST And Registration Substrate
------------------------------

``ParsedModule.structural_observations`` is the cached projection from one
parsed source authority through every registered collection family.  The graph
layer consumes that projection and does not import the AST registry during
collection.

.. automodule:: nominal_refactor_advisor.ast_tools
   :members: ParsedModule, AutoRegisterMeta, ModuleShapeSpec, AutoRegisteredModuleShapeSpec, CollectedFamily, RegisteredSpecCollectedFamily, SingleSpecCollectedFamily, collect_family_items


Observation Families
--------------------

.. automodule:: nominal_refactor_advisor.observation_families
   :members: GeneratedFamilySpec, FamilyGeneratingSpec, ObservationFamily, ShapeFamily, TypedLiteralObservationFamily


Export Policies
---------------

.. automodule:: nominal_refactor_advisor.export_tools
   :members: PublicExportPolicy, matches_public_export_policy, derive_public_exports


Observation Graph
-----------------

.. automodule:: nominal_refactor_advisor.observation_graph
   :members: ObservationKind, StructuralExecutionLevel, StructuralObservation, StructuralObservationCarrier, ObservationFiber, NominalWitnessGroup, ObservationCohort, ObservationGraph, collect_structural_observations, build_observation_graph


Observation Shapes
------------------

.. automodule:: nominal_refactor_advisor.observation_shapes
   :members: LiteralKind, FieldOriginKind, StructuralObservationTemplate, FieldObservation, LiteralDispatchObservation, ProjectionHelperShape, ScopedShapeWrapperFunction, ScopedShapeWrapperSpec, ConfigDispatchObservation, ClassMarkerObservation, SentinelTypeObservation, DynamicMethodInjectionObservation, BuilderCallShape, RegistrationShape
