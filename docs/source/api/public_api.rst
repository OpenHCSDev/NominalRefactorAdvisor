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
Each operation declaration also owns its ``CodemodSourceDependencyScope``.
Operations whose proof is complete over their explicit targets permit a narrow
snapshot.  ``RepositorySourceReprovedOperation`` marks operations that derive
participants, inheritance, imports, or other obligations from the repository;
the plan then selects the complete scan-backed snapshot without maintaining an
operation-name catalogue in the CLI.
Caller-authored authority source operations carry only the independent typed
authority kind.  Their full claim is derived from the operation target and the
single top-level class declaration in the supplied source, so serialized plans
cannot mirror or disagree with those nominal coordinates.
Exact method and dataclass-field factoring operations re-prove their complete
cohorts from one current-source witness and an explicit authority name.  Their
generated claims and physical edits therefore do not serialize participant or
member rosters.
Module-symbol moves likewise derive canonical source-module re-exports from the
destination module and moved declarations.  They reject import cycles rather
than accepting caller-authored import text that can disagree with the move.
Module-assignment replacements derive the selected assignment name from their
single replacement declaration instead of serializing that name twice.
Function signature and body mutations share one current-source proof of their
typed function target.
Function-signature replacements carry only their parameter and return suffix;
the targeted declaration supplies its function name and sync or async kind.
Whole-target replacements parse exactly one class or function declaration and
re-prove its concrete declaration kind and name against the current indexed
target before producing a physical edit.
Direct class-base mutations share one current-source proof and edit shell; the
nominal add and remove operation leaves own only their respective header
transformation.  Semantic no-ops preserve the original header source.
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
Repeated dataclass field prefixes follow the same current-source proof.  An
authored ``FactorExactDataclassFieldAuthorityOperation`` supplies the semantic
name when no owner exists.  When one behavior-free participant already owns
exactly the repeated prefix,
``PromoteExactDataclassFieldsToExistingAuthorityOperation`` instead derives the
other participants, moves that owner before its new descendants when source
order requires it, and refuses relocation across a use of the authority name.
Ordered plan stages can therefore factor nested field-prefix lattices without
persisting class or field rosters between stages.
The ``exact_leaf_method_ancestor_promotion`` detector emits a codemod candidate
only when one existing direct authority is unique, every direct child
participates and is a leaf, the method source is exact and promotion-safe, all
receiver requirements belong to the authority contract, no competing ancestor
binds the promoted names, no decorator or class-creation hook can observe the
ownership move, and the complete method batch pays compression rent.  The
codemod preflight reconstructs the same proof from the current full AST.
``ReplaceDirectClassBaseOperation`` persists only the displaced and replacement
class targets.  It derives the complete direct-child cohort, source spelling of
aliased bases, canonical imports, import-cycle safety, and replacement-relative
MRO conflicts from the current repository before rewriting any class header.
The operation rejects incomplete nominal base graphs rather than serialising a
caller-maintained child roster.
``CollapseRedundantClassAuthorityOperation`` strengthens that contract for a
closed repository-local authority.  It additionally proves that the displaced
and replacement classes are standalone, that their complete method syntax and
global bindings are behaviorally equivalent, and that every reference to the
displaced authority is one of the rewritten direct-base edges.  One operation
then redirects the source-derived child cohort, removes imports used only by the
deleted declaration, and deletes the redundant class.  Class-creation policy,
resolved non-base or exact string references, imported exposure, star-import
boundaries, and open inheritance remain explicit proof failures rather than
inferred cleanup permissions.
The operation is authored with both class targets because source equivalence
cannot decide which semantic name is canonical; no detector guesses that
direction.

The ``closed_parameter_conveyor`` detector exposes a recipe only for a complete
private call component that transports every field of one existing dataclass
authority.  ``CollapseClosedParameterConveyorOperation`` serialises only that
authority's source target.  Execution re-derives the product and call families
from the current source snapshot, requires the component proof again, and then
rewrites all participating signatures, field loads, and calls as one operation.
The ``declared_carrier_expansion`` detector exposes
``CollapseDeclaredCarrierExpansionOperation`` through the same atomic rewrite
contract when a value already proved to have the carrier's declared return type
is expanded into its fields at a call boundary.  It follows the complete
downstream forwarding graph, re-proves every participating callable and product
relation, and derives any cycle-safe carrier imports from the repository.
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
Manual carrier-collapse recipes likewise identify the affected class through
their exact source target.  Each unavoidable field mapping is a nominal
``CarrierFieldProjection`` payload record rather than an encoded string pair.
Constructor rewrites resolve that target's nominal class identity from the
current source.  Attribute projections resolve stable nominal parameter types.
Neither constructor nor attribute-owner spellings are persisted in the plan.

.. automodule:: nominal_refactor_advisor.codemod
   :members: CodemodPlanRoot, CodemodPlanDocument, CodemodPlanSequence, CodemodSourceDependencyScope, PlannedSourceRewrite, RefactorRecipeOperation, ReplaceTargetOperation, ReplaceDirectClassBaseOperation, CollapseRedundantClassAuthorityOperation, CarrierFieldProjection, ReplaceFieldsWithCarrierOperation, FactorExactDataclassFieldAuthorityOperation, PromoteExactDataclassFieldsToExistingAuthorityOperation, FactorExactMethodRoleOperation, PromoteExactLeafMethodsToAncestorOperation, CollapseClosedParameterConveyorOperation, CollapseDeclaredCarrierExpansionOperation, DeriveClassFamilyCollectionOperation, DeriveEnumSubsetOperation, DeriveDataclassPayloadProjectionOperation, DeriveDataclassFieldNameCollectionProjectionOperation, DeriveDataclassKeyValueSequenceProjectionOperation, DeriveDataclassConstructorProjectionOperation, FindingRecipeProofObstacle, FindingRecipeSynthesisRecord, FindingRecipeFrontierBudget, FindingRecipeTrajectoryFrontier, CodemodSimulationReport, format_codemod_unified_diff, apply_codemod_simulation, simulate_planned_rewrites, CancelableCompositionSignal, detect_cancelable_composition_signals

.. automodule:: nominal_refactor_advisor.class_authority_collapse
   :members: ClassMethodBehaviorAuthority, RedundantClassAuthorityCollapseProof

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
