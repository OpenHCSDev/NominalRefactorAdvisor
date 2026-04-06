Detector Catalog
================

This page is a maintenance-oriented catalog of the concrete detector families.
Unlike the substrate pages, this catalog is intentionally organized by refactoring
role rather than by source-file order.

The detector implementations remain an internal maintenance surface rather than a
promised downstream extension API, but this catalog is the right place to orient
future maintainers when a new smell family, detector regression, or self-hosting
gap appears.

.. currentmodule:: nominal_refactor_advisor.detectors

Registration And Derived Surface Detectors
------------------------------------------

These detectors focus on rate-1 authority for class-family registration, exported
surfaces, and derived module indexes.

.. autosummary::
   :nosignatures:

   ManualFamilyRosterDetector
   DeferredClassRegistrationDetector
   ManualClassRegistrationDetector
   BidirectionalRegistryDetector
   DeclarativeFamilyBoilerplateDetector
   TypeIndexedDefinitionBoilerplateDetector
   RegisteredUnionSurfaceDetector
   RegistryTraversalSubstrateDetector
   DerivedExportSurfaceDetector
   DerivedIndexedSurfaceDetector
   ManualPublicApiSurfaceDetector
   ExportPolicyPredicateDetector


Schema And Projection Detectors
-------------------------------

These detectors identify repeated record/projection mappings and other surfaces
that should derive from one authoritative schema.

.. autosummary::
   :nosignatures:

   RepeatedBuilderCallDetector
   RepeatedExportDictDetector
   StructuralObservationProjectionDetector
   AlternateConstructorFamilyDetector
   ExistingNominalAuthorityReuseDetector


Nominal Family And Witness Detectors
------------------------------------

These detectors recover semantic identity that is currently structural, duck-typed,
or distributed across repeated witness carriers.

.. autosummary::
   :nosignatures:

   StructuralConfusabilityDetector
   ExistingNominalAuthorityReuseDetector
   SemanticWitnessFamilyDetector
   MixinEnforcementDetector
   ManualFiberTagDetector
   DescriptorDerivedViewDetector


Algorithm And Template Detectors
--------------------------------

These detectors focus on repeated orchestration, template-method structure, and
 wrapper/helper families that should collapse into shared nominal substrates.

.. autosummary::
   :nosignatures:

   RepeatedPrivateMethodDetector
   InheritanceHierarchyCandidateDetector
   FindingAssemblyPipelineDetector
   GuardedDelegatorSpecDetector
   HelperBackedObservationSpecDetector
   RepeatedPropertyAliasHookDetector
   ConstantPropertyHookDetector
   OrchestrationHubDetector
   ParameterThreadFamilyDetector


Dispatch And Partial-View Detectors
-----------------------------------

These detectors cover closed-family dispatch, probing, and other partial-view
recoveries that should become nominal boundaries or authoritative dispatch tables.

.. autosummary::
   :nosignatures:

   EnumStrategyDispatchDetector
   InlineLiteralDispatchDetector
   StringDispatchDetector
   NumericLiteralDispatchDetector
   AttributeProbeDetector
   ReflectiveSelfAttributeEscapeDetector
   DynamicSelfFieldSelectionDetector


Dynamic Type And Interface Detectors
------------------------------------

These detectors cover runtime type generation, lineage, virtual membership, and
type-namespace mutation patterns.

.. autosummary::
   :nosignatures:

   GeneratedTypeLineageDetector
   DynamicInterfaceGenerationDetector
   ManualVirtualMembershipDetector
   SentinelTypeMarkerDetector
   DynamicMethodInjectionDetector
   DualAxisResolutionDetector


Maintenance Notes
-----------------

- Prefer adding a new detector only when an existing detector family cannot be
  widened semantically.
- Prefer semantic-role normalization and derived-surface reasoning over lexical
  pattern matching.
- When self-hosting finds a new generic miss, add detection first, rerun the tool,
  and only then refactor the advisor itself.
