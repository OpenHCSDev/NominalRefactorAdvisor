Public API
==========

This page documents the import surface that downstream tooling should prefer.
It does not restate every internal helper module.

Package Surface
---------------

.. automodule:: nominal_refactor_advisor
   :members:


CLI Entry Points
----------------

.. automodule:: nominal_refactor_advisor.cli
   :members: analyze_path, analyze_paths, plan_path, plan_paths, main


Structural Hypothesis Surface
-----------------------------

.. automodule:: nominal_refactor_advisor.planner
   :members: build_refactor_plans


Codemod Candidate Surface
-------------------------

The codemod surface models source-anchored candidate rewrites and simulations.
A clean current-snapshot simulation is not an application recommendation;
export and application require a proof across reachable refactor trajectories.
Its cancelable-composition signal is generic: it treats pack, unpack, and
field-forwarding wrappers as factorable product morphisms when they preserve
common fields and do not own an invariant.
Codemod operations, selectors, and payload records declare wire semantics with
``codemod_payload_field`` on their dataclass fields.
``DataclassPayloadProjection`` derives each binding catalogue and JSON
projection from those declarations.  ``FlattenedPayloadRecordValueCodec`` lets
a nested nominal record, such as a source-rewrite target, own a flattened wire
projection without an envelope exception.  ``CodemodPayloadRecord`` inherits
object decoding and unknown-field rejection from the same nominal declaration;
``DiscriminatedPayloadRecord`` adds inherited discriminator decoding and
projection while each polymorphic family resolves its own nominal registry.
``PayloadRecordValueCodec`` and ``PayloadRecordArrayValueCodec`` reuse those
declarations for nested records, including proof-carrying ``AuthorityClaim``
values.  No parallel schema, payload carrier, or boundary-role catalogue is
maintained.
``CodemodPlanRoot`` owns the document-or-sequence input sum.  Exact document
and sequence declarations reject each other's fields; each variant provides
its own execution-sequence projection without making the sequence parser a
compatibility fallback.
Authority proof is structural rather than textual.  Recipe preflight validates
declared ``AuthorityClaim`` values but never infers obligations from rationale
prose.  ``SemanticDescentRecipeEvaluation`` owns the stronger semantic-descent
contract: each recipe builder must carry a claim derived from its exact resolved
source target, while a source-reproved operation that establishes a new boundary
declares that claim from its current source proof.  The evaluation gate validates
both forms without guessing authority names from finding text or copying
participant rosters into the recipe.
Caller-authored authority source operations likewise fail closed unless their
typed claim names the destination and the supplied source contains exactly that
one top-level class declaration.
Method-promotion and method-extraction operations derive the new base or peer
authority claim from their validated current-source targets.
Candidate-collector forwarding findings compile to a one-target operation that
re-derives the collector, traversal scope, candidate type, configuration use,
old base, and replacement base from the same current-source witness.
``AuthorityClaim`` carries ``SemanticAuthorityKind`` directly and matches exact
target identity, name, and any declared location coordinates.  A declaration
missing a coordinate required by the claim does not prove it.
``AuthorityClaimStatus`` owns actionability and the admissible number of proved
authority identities, while ``AuthorityClaimResolution`` derives resolved or
ambiguous outcomes from proof edges.  Exact-plan source snapshots include claim
locations as source dependencies; unlocated claims and repository-wide guards
select the complete scan-backed snapshot instead.

.. automodule:: nominal_refactor_advisor.codemod_payload
   :members: DataclassPayloadProjection, CodemodPayloadRecord, DiscriminatedPayloadRecord, codemod_payload_field, FlattenedPayloadRecordValueCodec, PayloadRecordValueCodec, PayloadRecordArrayValueCodec, PayloadValueCodec, PayloadBindingSet

Rejected ``FindingRecipeSynthesisRecord`` values expose ``proof_obstacles``.
Each obstacle identifies the nominal executable declaration that failed to
prove a recipe and carries that declaration's diagnostic.  The record's
``reason`` is a concise summary; consumers that need diagnostic proof detail
should inspect the structured obstacles instead of parsing that summary.
For detector-owned synthesis, ``executable_declaration`` names the concrete
detector declaration whose MRO supplied the recipe behaviour.  Inferred
metric-driven synthesis names the strategy or builder declaration selected by
the finding evidence.

Exact repeated methods without a proved owner remain evidence-only findings for
automatic synthesis because source structure cannot determine the new
authority's semantic name.  An authored ``FactorExactMethodRoleOperation``
persists one evidence-method target and that explicit name, then re-proves the
complete class cohort and method set from the current source before factoring
the role.  It does not persist a mirrored class or method roster.
The ``exact_leaf_method_ancestor_promotion`` detector emits a codemod candidate
only when one existing direct authority is unique, every direct child
participates and is a leaf, the method source is exact and promotion-safe, all
receiver requirements belong to the authority contract, no competing ancestor
binds the promoted names, no decorator or class-creation hook can observe the
ownership move, and the complete method batch pays compression rent.  The
codemod preflight reconstructs the same proof from the current full AST.

The ``closed_parameter_conveyor`` detector exposes a recipe only for a complete
private call component that transports every field of one existing dataclass
authority.  ``CollapseClosedParameterConveyorOperation`` serialises only that
authority's source target.  Execution re-derives the product and call families
from the current source snapshot, requires the component proof again, and then
rewrites all participating signatures, field loads, and calls as one operation.
Exact byte spans distinguish multiple calls on the same line.  Source forms
that cannot be edited without losing comments or lexical-scope semantics remain
rejected; field mappings are never copied into the recipe payload.

Class-family, enum, and exhaustive dataclass projection recipes follow the same
source-derived contract.  Dataclass return dictionaries, field-name
collections, returned key/value sequences, and constructor calls mediated by an
existing authority method each persist only the exact authority and
containing-function targets.  Constructor mediation additionally proves that
both calls resolve to the same nominal class and that the direct authority
method exhaustively preserves its field and parameter relation.
``DeriveClassFamilyCollectionOperation``, ``DeriveEnumSubsetOperation``, and
the corresponding ``DeriveDataclass*ProjectionOperation`` declarations
re-resolve those targets on every simulation or application, prove one
unambiguous projection against the current declarations, derive the required
imports and replacement source, and reject shadowed builtins, name collisions,
or changed relations.
Collection members, dataclass fields, assignment names, generated source, and
finding metrics are not copied into the persisted plan.

.. automodule:: nominal_refactor_advisor.codemod
   :members: CodemodPlanRoot, CodemodPlanDocument, CodemodPlanSequence, PlannedSourceRewrite, RefactorRecipeOperation, ReplaceTargetOperation, FactorExactMethodRoleOperation, PromoteExactLeafMethodsToAncestorOperation, CollapseClosedParameterConveyorOperation, DeriveClassFamilyCollectionOperation, DeriveEnumSubsetOperation, DeriveDataclassPayloadProjectionOperation, DeriveDataclassFieldNameCollectionProjectionOperation, DeriveDataclassKeyValueSequenceProjectionOperation, DeriveDataclassConstructorProjectionOperation, FindingRecipeProofObstacle, FindingRecipeSynthesisRecord, FindingRecipeFrontierBudget, FindingRecipeTrajectoryFrontier, CodemodSimulationReport, format_codemod_unified_diff, apply_codemod_simulation, simulate_planned_rewrites, CancelableCompositionSignal, detect_cancelable_composition_signals

Goal Trajectory Surface
-----------------------

The goal runner reports complete reachable-state evidence separately from the
single replay sequence.  ``PROVED`` means that exhaustive exploration found one
unique terminal source state.  Ambiguity or any frontier, depth, or state-budget
obstacle keeps ``stages`` empty and prevents application.
Target-free states that increase any complete-scan finding class relative to
the starting state are recorded as unjustified-debt terminals rather than
accepted as successful refactors.  The search may pass through intermediate
states with additional obligations, but a proved terminal must discharge them.
``FindingObligationClass`` derives that identity from the finding's pattern,
capability gap, and required-relation context.  Detector provenance is reported
separately, so a relation observed by a different detector is not mistaken for
a newly introduced obligation.
Caller-supplied architecture guards are evaluated at terminal states and stored
on the final replay document; recipe-owned guards remain transition-local.

.. automodule:: nominal_refactor_advisor.codemod_workflow
   :members: CodemodRefactorGoalRunner, CodemodRefactorGoalReport, CodemodRefactorTrajectoryBudget, CodemodRefactorTrajectoryProof, CodemodRefactorTrajectoryStatus, CodemodRefactorTrajectoryObstacle, CodemodRefactorUnjustifiedDebtTerminal


Result Records And Taxonomy
---------------------------

See :doc:`theory_and_results` for the frozen result dataclasses, taxonomy values,
and pattern metadata referenced by the public entrypoints.
