Advisor Self-Audit Detection Plan
=================================

Purpose
-------

This file is the repository-specific execution ledger for the advisor's next detection-first iteration.
The goal is to make the tool detect its own higher-order structural duplication before those duplications
are manually collapsed.

This plan sits on top of the imported background guides in ``docs/source/development/`` and turns the
current chat conclusions into an implementation order for the standalone advisor.

Detector Quality Requirements
-----------------------------

The next iteration should not merely add more detectors. It should make the detector set more generic,
more theory-accurate, and more self-hosting.

Every new detector or widening pass in this plan should satisfy these rules:

- **Deterministic**: same repository plus same detector configuration must produce the same findings;
  no probabilistic ranking or opaque semantic scoring
- **Generic**: detection should be phrased in terms of structural roles, nominal family relations, and
  semantic normalization rather than advisor-local symbol names whenever possible
- **Theory-accurate**: representation, fiber, confusability, unit-rate coherence, provenance, and
  MRO-governed nominal identity should explain both the smell and the fix
- **Existing-authority first**: when a compatible nominal authority already exists, prefer reuse of that
  base, mixin, or schema before synthesizing a new one
- **Explicit certification**: keep the current rule that findings say whether they are ``CERTIFIED`` or
  ``STRONG_HEURISTIC`` rather than pretending every detector has the same theorem strength

Cross-Cutting Genericization Work
---------------------------------

Several current detectors are deterministic but still too lexical or too repository-shaped. The plan
therefore includes genericization work alongside the missing detectors.

Required substrate improvements:

- a project-wide nominal authority index for class definitions, abstractness, declared bases, field
  families, method families, and dataclass status
- semantic-role normalization for field families, projection builders, and finding-assembly pipelines so
  grouping is based on meaning rather than exact AST identity
- a split between generic detector cores and any repository-local schema adapters, so self-hosting logic
  remains reusable outside this repository
- a preference order in prescriptions: reuse existing nominal authority, else extract one new authority,
  else split orthogonal residue into mixins or multiple inheritance

Detection-First Rule
--------------------

Before refactoring the advisor's own structural duplication, the tool should either:

- already detect the duplication under an existing pattern family, or
- gain a new detector when no current detector is even attempting that class of smell.

This plan therefore separates each self-audit hotspot into one of two buckets:

- existing detection system should catch it but currently misses it
- no current detection system is even attempting it

Current Self-Audit Targets
--------------------------

The current manually observed hotspots are:

1. the manual detector roster in ``nominal_refactor_advisor/detectors.py``
2. the fragmented ``PatternId`` planning tables in ``nominal_refactor_advisor/planner.py``
3. the repeated detector-local ``_findings_for_module`` assembly pipelines in
   ``nominal_refactor_advisor/detectors.py``
4. the repeated guard-and-delegate observation spec wrappers in
   ``nominal_refactor_advisor/observation_families.py``
5. the repeated ``StructuralObservation(...)`` projection builders in
   ``nominal_refactor_advisor/observation_shapes.py``
6. concrete classes whose field family already matches an existing reusable ``ABC`` or base carrier, but
   the tool does not currently detect that inheritance or mixin reuse should happen

Expectation Matrix
------------------

Manual detector roster
~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 6 ``AutoRegisterMeta / class-level registration normal form``
- **Current status**: not detected
- **Should the current tool catch it?** no; no current detector even attempts this exact shape
- **Why it is in scope**: the roster is a class-family discovery mechanism detached from class existence,
  which is the same class-level smell already covered conceptually by Pattern 6
- **Current failure mode**: existing registration detectors only look for manual registry assignments,
  registration calls, or registration decorators; they never look for tuple/list rosters of sibling
  subclasses or subclass instances returned from one authority function
- **New detection required**: ``ManualFamilyRosterDetector``
- **Prescribed collapse**: ``__init_subclass__`` or metaclass-backed detector registration with declarative
  ordering metadata
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``tests/test_refactor_advisor.py``

Fragmented ``PatternId`` planning tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 14 ``authoritative constructor / projection schema``
- **Current status**: not detected
- **Should the current tool catch it?** no; no current detector even attempts fragmented multi-table
  authority over one enum-keyed family
- **Why it is in scope**: several dicts keyed by the same ``PatternId`` family collectively encode one
  semantic planning record, but the authority is split across parallel structures
- **Current failure mode**: existing mapping detectors see repeated builders and repeated export dicts at
  one call site, not a distributed family of key-aligned tables across module scope
- **New detection required**: ``FragmentedFamilyAuthorityDetector``
- **Prescribed collapse**: one authoritative ``PatternPlanningSpec`` dataclass keyed once by ``PatternId``
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``nominal_refactor_advisor/planner.py``
  - ``tests/test_refactor_advisor.py``

Repeated detector-local finding assembly pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 5 ``ABC template-method migration``
- **Current status**: not detected
- **Should the current tool catch it?** yes, at least partially
- **Existing detector families that should help**:
  - ``RepeatedPrivateMethodDetector``
  - ``InheritanceHierarchyCandidateDetector``
- **Current failure mode**:
  - method grouping is too lexical and too statement-count driven
  - the shared role is semantic pipeline structure, not exact AST identity
  - per-detector summary text, evidence shape, metrics construction, scaffold helpers, and patch helpers
    create surface variation that defeats the current fingerprinting
  - the hierarchy detector expects repeated method-role clusters across the same class family, while the
    strongest shared role here is the repeated ``_findings_for_module`` template
- **Detection change required**: add a semantic pipeline detector or widen Pattern 5 normalization so it
  understands finding-assembly pipelines instead of only repeated exact statement skeletons
- **Proposed detector name**: ``FindingAssemblyPipelineDetector``
- **Prescribed collapse**: one candidate-driven detector base plus mixins for evidence, metrics, and
  scaffold/codemod policies
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``tests/test_refactor_advisor.py``

Repeated guard-and-delegate observation spec wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 5 ``ABC template-method migration``
- **Current status**: not detected
- **Should the current tool catch it?** yes, partially
- **Existing detector families that should help**:
  - ``RepeatedPrivateMethodDetector``
  - ``InheritanceHierarchyCandidateDetector``
- **Current failure mode**:
  - the relevant methods are tiny wrappers and often fall below duplicate-statement thresholds
  - the current repeated-method logic does not normalize guard-then-delegate structure such as
    ``if class scope is wrong: return None`` followed by one helper call
  - the current hierarchy detector again expects larger repeated role families than this wrapper family
- **Detection change required**: add a dedicated guarded-delegator detector or widen Pattern 5
  normalization to understand wrapper-spec families
- **Proposed detector name**: ``GuardedDelegatorSpecDetector``
- **Prescribed collapse**: helper-backed spec substrate with mixins such as module-only, class-only,
  function-only, and assign-only behavior filters
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``nominal_refactor_advisor/observation_families.py``
  - ``tests/test_refactor_advisor.py``

Repeated ``StructuralObservation(...)`` projection builders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 14 ``authoritative constructor / projection schema``
- **Current status**: not detected
- **Should the current tool catch it?** yes, partially
- **Existing detector family that should help**: ``RepeatedBuilderCallDetector``
- **Current failure mode**:
  - grouping uses exact callee name, keyword list, and value fingerprint
  - semantic role overlap is high, but the current detector does not know that
    ``owner_symbol``, ``nominal_witness``, ``observed_name``, and ``fiber_key`` are role slots inside one
    projection family
  - small expression differences such as ``self.symbol`` versus ``self.function_name`` prevent grouping
- **Detection change required**: widen Pattern 14 normalization to recognize semantic projection-role
  families instead of exact literal builder equality
- **Proposed detector name**: ``StructuralObservationProjectionDetector``
- **Prescribed collapse**: one projection base or mixin substrate where carriers declare role hooks and the
  common ``StructuralObservation`` assembly lives once
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``nominal_refactor_advisor/observation_shapes.py``
  - ``tests/test_refactor_advisor.py``

Existing compatible ``ABC`` or base is not reused
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Primary pattern**: Pattern 5 ``ABC template-method migration`` with an existing-authority reuse check
- **Current status**: not detected as a first-class rule
- **Should the current tool catch it?** partially, but it currently does not
- **Existing detector families that should help**:
  - ``RepeatedFieldFamilyDetector``
  - ``SemanticWitnessFamilyDetector``
  - ``MixinEnforcementDetector``
- **Current failure mode**:
  - repeated-family detectors reason over sibling duplication cohorts, not over the space of already
    declared authorities
  - the tool can say that several classes should share a base, but it does not ask whether a compatible
    base already exists and should simply be inherited
  - there is no project-wide nominal authority index for comparing concrete classes against existing
    abstract bases, reusable dataclass carriers, or mixins
  - prescriptions therefore over-synthesize new bases and under-reuse existing nominal structure
- **Detection change required**: add an existing-authority reuse detector, or widen field-family and
  witness-family analysis so they first search for compatible declared authorities before proposing a new
  one
- **Proposed detector name**: ``ExistingNominalAuthorityReuseDetector``
- **Prescribed collapse**: inherit the existing ``ABC`` or base directly when semantics match; if only one
  orthogonal semantic slice matches, reuse or extract a mixin and compose through multiple inheritance
- **Primary files**:
  - ``nominal_refactor_advisor/detectors.py``
  - ``nominal_refactor_advisor/observation_families.py``
  - ``nominal_refactor_advisor/observation_shapes.py``
  - ``tests/test_refactor_advisor.py``

Priority Order
--------------

Phase 0: generic semantic substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. add a project-wide nominal authority index
2. add reusable semantic-role normalization helpers for fields, projections, and detector pipelines
3. add an existing-authority preference rule so prescriptions reuse compatible bases or mixins before
   synthesizing new ones

Phase 1: add missing detector families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4. ``ManualFamilyRosterDetector``
5. ``FragmentedFamilyAuthorityDetector``
6. ``ExistingNominalAuthorityReuseDetector``

These have the clearest gap: the current system is not even trying to detect them.

Phase 2: sharpen existing pattern families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7. ``FindingAssemblyPipelineDetector`` or equivalent Pattern 5 widening
8. ``GuardedDelegatorSpecDetector`` or equivalent Pattern 5 widening
9. ``StructuralObservationProjectionDetector`` or equivalent Pattern 14 widening

These are structurally in scope already, but the current normalization is too lexical.

Phase 3: genericization review of existing detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

10. audit current detectors for repository-local schema assumptions and push those assumptions into
    declarative adapters where possible
11. widen any remaining lexical-only grouping keys toward semantic role keys when the paper model supports
    it
12. keep deterministic output by expressing all widened logic as explicit normalization and threshold rules

Phase 4: self-hosting validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After each detector lands:

- add focused regression tests in ``tests/test_refactor_advisor.py``
- run ``python -m pytest tests/test_refactor_advisor.py -q``
- run ``python -m nominal_refactor_advisor nominal_refactor_advisor --include-plans``
- verify the new finding fires on the advisor itself before refactoring the underlying duplication

Implementation Checklist
------------------------

Shared substrate work
~~~~~~~~~~~~~~~~~~~~~

- add candidate dataclasses for each new detection family in ``nominal_refactor_advisor/detectors.py``
- add a nominal authority index that can answer: which reusable bases, abstract carriers, or mixins already
  exist for this semantic family?
- prefer semantic-role normalization over lexical-only field comparison
- prefer compatible existing authorities before proposing a synthetic new one
- keep generic detector cores separate from any repository-local schema catalogs
- keep each new detector prescriptive: finding text must say which nominal collapse is required
- add scaffold and codemod draft text for each detector, not just a complaint

Tests to add
~~~~~~~~~~~~

Add one focused regression test per family:

- manual subclass roster returned from a helper
- two or more ``PatternId``-keyed dicts that clearly split one planning record
- a concrete class whose fields and types match an existing reusable ``ABC`` or base carrier
- several detector classes with repeated ``_findings_for_module`` candidate pipelines
- several observation spec classes with repeated guard-and-delegate wrappers
- several carriers with repeated ``StructuralObservation(...)`` projection roles

Each test should assert all of the following when possible:

- the expected detector id fires
- the summary names the semantic family, not just raw syntax
- the scaffold points to the correct nominal collapse
- the codemod patch mentions the right base, mixin, metaclass, or authoritative record target

Non-Goals for This Iteration
----------------------------

The repetitive family wrappers in ``nominal_refactor_advisor/observation_families.py`` that are already
near-declarative normal form are not the first target. They may still be simplified later, but the higher
priority is closing the detection gaps listed above.

Likewise, this plan does not aim to remove all self-hosting special cases immediately. The nearer-term goal
is to move those cases behind generic semantic detectors and explicit adapter tables rather than leaving the
reasoning baked into lexical one-offs.

Success Criteria
----------------

This plan is complete when the advisor can run on itself and emit findings for each of the five hotspot
families before any of those hotspot families are manually collapsed.
