Advisor Detection Edit Drafts
=============================

Purpose
-------

This file records concrete edit drafts for the detection-first work listed in
``advisor_self_audit_detection_plan.rst``. These are not yet applied code changes; they are proposed edit
shapes and insertion points.

Cross-Cutting Design Constraints
--------------------------------

Every draft below should be implemented in a way that keeps the advisor:

- deterministic rather than probabilistic
- generic over semantic roles rather than repository-local names where possible
- theory-accurate with respect to partial views, confusability, unit-rate coherence, provenance, and
  nominal identity
- biased toward reusing existing nominal authorities before inventing new ones

File Targets
------------

The main implementation surface is:

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/planner.py``
- ``nominal_refactor_advisor/observation_families.py``
- ``nominal_refactor_advisor/observation_shapes.py``
- ``tests/test_refactor_advisor.py``

Draft 0: Shared semantic substrate
----------------------------------

Target files
~~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/ast_tools.py``
- ``nominal_refactor_advisor/observation_shapes.py``

Proposed shared index and normalization layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class NominalAuthorityShape:
       file_path: str
       class_name: str
       line: int
       declared_base_names: tuple[str, ...]
       field_names: tuple[str, ...]
       field_type_map: tuple[tuple[str, str], ...]
       method_names: tuple[str, ...]
       is_abstract: bool
       is_dataclass_family: bool

   @dataclass(frozen=True)
   class SemanticRoleSignature:
       role_names: tuple[str, ...]
       field_names: tuple[str, ...]
       field_type_map: tuple[tuple[str, str], ...]

   class NominalAuthorityIndex:
       def all_shapes(self) -> tuple[NominalAuthorityShape, ...]: ...
       def compatible_bases_for(
           self,
           class_shape: NominalAuthorityShape,
           *,
           minimum_role_overlap: int = 2,
       ) -> tuple[NominalAuthorityShape, ...]: ...

Why this substrate comes first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- it gives later detectors one generic way to talk about existing bases, mixins, and carriers
- it avoids duplicating ad hoc class-shape discovery inside each new detector
- it supports the reuse-first rule the current detector set is missing

Draft 1: Manual family roster detection
---------------------------------------

Target file
~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class ManualFamilyRosterCandidate:
       file_path: str
       line: int
       owner_name: str
       roster_name: str
       family_base_name: str
       member_names: tuple[str, ...]
       constructor_style: str

   def _manual_family_roster_candidates(
       module: ParsedModule,
   ) -> tuple[ManualFamilyRosterCandidate, ...]:
       # Detect module-level helpers or constants that return/build
       # a tuple/list of sibling detector subclasses or detector instances.
       ...

   class ManualFamilyRosterDetector(PerModuleIssueDetector):
       detector_id = "manual_family_roster"
       finding_spec = FindingSpec(
           pattern_id=PatternId.AUTO_REGISTER_META,
           title="Manual subclass roster should become auto-registration",
           why=(
               "One helper manually enumerates a class family instead of deriving membership from class existence."
           ),
           capability_gap="zero-delay class-family discovery with declarative ordering",
           relation_context="family membership is maintained by a manual roster function or constant",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.CLASS_LEVEL_REGISTRATION,
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.MRO_ORDERING,
           ),
       )

Detection notes
~~~~~~~~~~~~~~~

- restrict to rosters whose members share a nominal base such as ``IssueDetector`` or
  ``PerModuleIssueDetector``
- accept either class names or zero-arg constructor calls
- treat ``default_detectors()`` in ``nominal_refactor_advisor/detectors.py`` as the motivating self-hit
- include priority metadata in the scaffold so the finding prescribes more than bare auto-registration

Suggested scaffold draft
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RegisteredDetector(IssueDetector):
       detector_priority: ClassVar[int] = 0

       def __init_subclass__(cls, **kwargs):
           super().__init_subclass__(**kwargs)
           if not inspect.isabstract(cls):
               DETECTOR_REGISTRY.register(cls, priority=cls.detector_priority)

Draft 2A: Existing nominal authority reuse detection
----------------------------------------------------

Target files
~~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/ast_tools.py``
- ``nominal_refactor_advisor/observation_shapes.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class ExistingNominalAuthorityReuseCandidate:
       file_path: str
       class_name: str
       line: int
       compatible_authority_name: str
       compatible_authority_line: int
       reuse_kind: str
       shared_role_names: tuple[str, ...]
       shared_field_names: tuple[str, ...]

   def _existing_nominal_authority_reuse_candidates(
       modules: tuple[ParsedModule, ...],
   ) -> tuple[ExistingNominalAuthorityReuseCandidate, ...]:
       # Build a project-wide authority index.
       # Compare concrete classes against existing abstract bases, reusable carriers,
       # and mixin-like authorities.
       # Emit only when semantic overlap is strong and the class does not already
       # inherit or compose the authority.
       ...

   class ExistingNominalAuthorityReuseDetector(IssueDetector):
       detector_id = "existing_nominal_authority_reuse"
       finding_spec = FindingSpec(
           pattern_id=PatternId.ABC_TEMPLATE_METHOD,
           title="Existing nominal authority should be reused",
           why=(
               "A compatible nominal authority already exists, but the class repeats the same semantic field family outside that hierarchy."
           ),
           capability_gap="reuse of an existing authoritative base or mixin instead of duplicating the family",
           relation_context="a concrete class repeats a semantic family already declared by an existing nominal authority",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
               CapabilityTag.MRO_ORDERING,
           ),
       )

Detection notes
~~~~~~~~~~~~~~~

- prefer semantic-role and annotation compatibility over raw name coincidence alone
- require that inheritance or mixin composition is semantically safe, not just structurally possible
- prefer direct base reuse when the authority owns the full family
- prefer mixin reuse when only one orthogonal slice overlaps

Suggested scaffold draft
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class ExistingAuthorityBase(ABC):
       file_path: str
       line: int

   @dataclass(frozen=True)
   class ConcreteCarrier(ExistingAuthorityBase):
       specific_payload: tuple[str, ...]

Draft 2: Fragmented family authority detection
----------------------------------------------

Target files
~~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/planner.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class FragmentedFamilyAuthorityCandidate:
       file_path: str
       mapping_names: tuple[str, ...]
       line_numbers: tuple[int, ...]
       key_family_name: str
       shared_keys: tuple[str, ...]
       role_names: tuple[str, ...]

   def _fragmented_family_authority_candidate(
       module: ParsedModule,
   ) -> FragmentedFamilyAuthorityCandidate | None:
       # Collect module-level dict literals keyed by PatternId.*
       # Group dicts with strong key overlap and semantically related names.
       ...

   class FragmentedFamilyAuthorityDetector(PerModuleIssueDetector):
       detector_id = "fragmented_family_authority"
       finding_spec = FindingSpec(
           pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
           title="Parallel key-family tables should become one authoritative record",
           why=(
               "Several key-aligned dicts together encode one planning record, but the authority is fragmented."
           ),
           capability_gap="single authoritative enum-keyed planning record",
           relation_context="one key family is split across parallel metadata tables",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.AUTHORITATIVE_MAPPING,
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.PROVENANCE,
           ),
       )

Suggested planner target shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class PatternPlanningSpec:
       pattern_id: PatternId
       priority: int
       step_builder: PatternPlanStepBuilder
       action_builder: PatternActionBuilder
       dependencies: tuple[PatternId, ...] = ()
       synergy_with: tuple[PatternId, ...] = ()

   PATTERN_PLANNING_SPECS = {
       PatternId.ABC_TEMPLATE_METHOD: PatternPlanningSpec(...),
       PatternId.AUTHORITATIVE_SCHEMA: PatternPlanningSpec(...),
   }

Draft 3: Finding assembly pipeline detection
--------------------------------------------

Target file
~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class FindingAssemblyPipelineCandidate:
       file_path: str
       class_name: str
       line: int
       method_name: str
       candidate_source_name: str
       summary_role: str
       evidence_role: str
       metrics_type_name: str | None
       scaffold_helper_name: str | None
       patch_helper_name: str | None

   def _finding_pipeline_candidates(
       module: ParsedModule,
   ) -> tuple[FindingAssemblyPipelineCandidate, ...]:
       # Restrict to detector classes.
       # Normalize methods named _findings_for_module.
       # Identify the semantic pipeline:
       # gather candidates -> loop -> build finding -> append/return.
       ...

   class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
       detector_id = "finding_assembly_pipeline"
       finding_spec = FindingSpec(
           pattern_id=PatternId.ABC_TEMPLATE_METHOD,
           title="Repeated finding-assembly pipeline should move into a detector base",
           why=(
               "Several detectors repeat the same candidate-to-finding pipeline with only orthogonal hooks varying."
           ),
           capability_gap="candidate-driven detector template with abstract hooks and mixins",
           relation_context="same finding assembly stages repeat across sibling detector classes",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.MRO_ORDERING,
           ),
       )

Suggested normalization hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalize these roles instead of exact AST text:

- candidate collector call
- loop variable and candidate family
- summary string construction
- evidence assembly shape
- scaffold helper call
- patch helper call
- metrics constructor
- return/append strategy
- detector capability tags and relation-context role

Suggested target substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CandidateFindingDetector(PerModuleIssueDetector, ABC):
       @abstractmethod
       def iter_candidates(self, module: ParsedModule, config: DetectorConfig) -> tuple[object, ...]:
           raise NotImplementedError

       @abstractmethod
       def build_finding(self, candidate: object) -> RefactorFinding:
           raise NotImplementedError

       def _findings_for_module(
           self, module: ParsedModule, config: DetectorConfig
       ) -> list[RefactorFinding]:
           return [
               self.build_finding(candidate)
               for candidate in self.iter_candidates(module, config)
           ]

Draft 4: Guarded delegator observation spec detection
-----------------------------------------------------

Target files
~~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/observation_families.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class GuardedDelegatorCandidate:
       file_path: str
       class_name: str
       line: int
       method_name: str
       guard_role: str
       delegate_name: str
       scope_role: str

   def _guarded_delegator_candidates(
       module: ParsedModule,
   ) -> tuple[GuardedDelegatorCandidate, ...]:
       # Detect tiny wrappers of the form:
       # if wrong scope: return None
       # return helper(...)
       ...

   class GuardedDelegatorSpecDetector(PerModuleIssueDetector):
       detector_id = "guarded_delegator_spec"
       finding_spec = FindingSpec(
           pattern_id=PatternId.ABC_TEMPLATE_METHOD,
           title="Repeated guarded spec wrappers should collapse into mixins",
           why=(
               "Several observation specs differ only by simple scope guards and one delegate helper call."
           ),
           capability_gap="shared wrapper substrate with orthogonal scope mixins",
           relation_context="guard-and-delegate wrapper logic repeats across sibling observation specs",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.MRO_ORDERING,
           ),
       )

Suggested target substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ScopeFilteredFunctionSpec(FunctionObservationSpec, ABC):
       @abstractmethod
       def accepts_scope(self, observation: ScopedAstObservation) -> bool:
           raise NotImplementedError

       @abstractmethod
       def delegate(self, parsed_module: ParsedModule, function: ast.FunctionDef) -> object | None:
           raise NotImplementedError

       def build_from_function(...):
           if not self.accepts_scope(observation):
               return None
           return self.delegate(parsed_module, function)

Draft 5: Structural observation projection detection
----------------------------------------------------

Target files
~~~~~~~~~~~~

- ``nominal_refactor_advisor/detectors.py``
- ``nominal_refactor_advisor/observation_shapes.py``

Proposed candidate and detector skeleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass(frozen=True)
   class StructuralObservationProjectionCandidate:
       file_path: str
       class_names: tuple[str, ...]
       line_numbers: tuple[int, ...]
       shared_role_names: tuple[str, ...]
       role_to_field_exprs: tuple[tuple[str, tuple[str, ...]], ...]

   def _structural_observation_projection_candidate(
       module: ParsedModule,
   ) -> StructuralObservationProjectionCandidate | None:
       # Inspect structural_observation properties that return StructuralObservation(...).
       # Normalize semantic roles instead of exact expressions.
       ...

   class StructuralObservationProjectionDetector(PerModuleIssueDetector):
       detector_id = "structural_observation_projection"
       finding_spec = FindingSpec(
           pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
           title="Repeated StructuralObservation builders should share one projection substrate",
           why=(
               "Several carriers repeat the same projection record assembly with only role hooks varying."
           ),
           capability_gap="single authoritative projection builder with role hooks",
           relation_context="same StructuralObservation schema is manually rebuilt across many carriers",
           confidence=HIGH_CONFIDENCE,
           certification=STRONG_HEURISTIC,
           capability_tags=(
               CapabilityTag.AUTHORITATIVE_MAPPING,
               CapabilityTag.NOMINAL_IDENTITY,
               CapabilityTag.PROVENANCE,
           ),
       )

Normalization notes
~~~~~~~~~~~~~~~~~~~

The detector should normalize projection roles such as:

- ``file_path``
- ``owner_symbol``
- ``nominal_witness``
- ``line`` / ``lineno``
- ``observation_kind``
- ``execution_level``
- ``observed_name``
- ``fiber_key``

so the grouping key is the semantic projection schema rather than the exact expression text.

Suggested target substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StructuralObservationTemplate(StructuralObservationCarrier, ABC):
       observation_kind: ClassVar[ObservationKind]
       execution_level: ClassVar[StructuralExecutionLevel]

       @property
       @abstractmethod
       def owner_symbol(self) -> str:
           raise NotImplementedError

       @property
       @abstractmethod
       def nominal_witness(self) -> str:
           raise NotImplementedError

       @property
       @abstractmethod
       def observed_name(self) -> str:
           raise NotImplementedError

       @property
       @abstractmethod
       def fiber_key(self) -> str:
           raise NotImplementedError

       @property
       def structural_observation(self) -> StructuralObservation:
           return StructuralObservation(
               file_path=self.file_path,
               owner_symbol=self.owner_symbol,
               nominal_witness=self.nominal_witness,
               line=self.lineno,
               observation_kind=type(self).observation_kind,
               execution_level=type(self).execution_level,
               observed_name=self.observed_name,
               fiber_key=self.fiber_key,
           )

Draft test additions
--------------------

Target file
~~~~~~~~~~~

- ``tests/test_refactor_advisor.py``

Suggested test names
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_detects_manual_family_roster_for_detector_registry(tmp_path: Path) -> None: ...
   def test_detects_existing_nominal_authority_reuse(tmp_path: Path) -> None: ...
   def test_detects_fragmented_pattern_planning_tables(tmp_path: Path) -> None: ...
   def test_detects_repeated_finding_assembly_pipeline(tmp_path: Path) -> None: ...
   def test_detects_guarded_delegator_spec_family(tmp_path: Path) -> None: ...
   def test_detects_repeated_structural_observation_projection(tmp_path: Path) -> None: ...

Suggested genericity checks
---------------------------

For every new detector, add at least one test that changes lexical names while preserving semantic roles.
The finding should still fire. This is the practical regression check that the detector is operating on
semantic duplication rather than brittle local spelling.

Suggested validation loop
-------------------------

After each detector is implemented:

1. run ``python -m pytest tests/test_refactor_advisor.py -q``
2. run ``python -m nominal_refactor_advisor nominal_refactor_advisor --include-plans``
3. confirm the new finding appears on the advisor itself
4. only then refactor the newly detected duplication
