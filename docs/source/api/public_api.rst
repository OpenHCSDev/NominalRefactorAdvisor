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

Inherited call selection uses ``CompactClassFamilyIndex.mro_authority``.
``ClassMroAuthority`` projects source declarations into inert types and delegates
C3 linearisation to Python. The projection does not execute analysed modules or
class bodies. Results are cached within that class-index snapshot and shared by
method and annotated-member lookup.

``ResolvedClassMro`` establishes declaration order, not successful execution of
the original class body or equivalence of an authored replacement expression.
``OpenClassMro`` carries the unresolved obligation and the class where it arose.
Unknown or ambiguous bases, dynamic base expressions, unsupported class-creation
hooks, cyclic graphs and inconsistent C3 orders remain open. Imported base
resolution preserves qualified paths across analysis-root boundaries; an
unrelated class with the same terminal name does not establish a binding.

``CompactFunctionFlow.binding_resolution_for`` owns source-write selection for
lexical and class-method lookup. Explicit use positions select the preceding
write; deferred module and class namespace lookup uses the final write.
Deferred closure lookup with multiple writes remains unresolved. Conditional
writes also remain unresolved.

``lexical_bindings`` owns ``ScopeBindingCollector``,
``LexicalScopeBindingAuthority`` and import-name/origin projection.
``FunctionBindingProjection`` uses the same collector when deriving function
locals. Declaration headers are visited in their enclosing scope; function and
class bodies are separate scopes. Comprehension loop targets remain internal,
while assignment-expression targets retain their enclosing ownership. Lambda
defaults are visited without entering the lambda body. These binding rules also
supply class-member collision checks.

``lexical_scopes.ClassNamespaceScope`` owns ordered class-name availability
for declaration dependency resolution. Assignment values, defaults, decorators
and dictionary entries are traversed in evaluation order. ``if`` branches and
short-circuit expressions join their possible namespace states.
``LexicalNameResolution`` distinguishes internal, external and unproved reads;
``ModuleNameReferenceSurface`` retains that result with the source reference.
``LexicalScopeABC`` supplies the lookup contract shared by class scopes and
``FunctionBindingProjection``. The traversal walks enclosing scopes without
dispatching on their concrete types. Function scopes retain compile-time locals;
class namespaces participate only in immediate lookup. ``TypeParameterScope``
retains the enclosing class visibility required by PEP 695 annotation scopes.
``ScopeBindingProjection`` supplies compile-time declarations to both function
and class lookup. For a class-local name, a read before assignment or after
deletion falls back to the module, rather than an enclosing function's local.
A path-dependent class/module lookup cannot authorise a rename. Loop,
exception, context-manager and pattern-matching regions retain affected bindings
as unproved until their execution paths can be established. Unrelated exact
references remain available. An augmented assignment that reads a module name
and writes a class name also requires a separate ownership proof.
The recorded :download:`scope ownership refactor
<../../examples/lexical_scope_refactor.py>` moves ``FunctionBindingProjection``
and its helper into this module, deriving consumer imports and the original
module's identity-preserving re-export through the DSL.
The :download:`argument binding consolidation
<../../examples/argument_binding_refactor.py>` then replaces both callers of the
duplicated parameter-name helper, removes its import and deletes the helper in
six projected stages.

``LexicalScopeContext`` owns the lexical frame stack. Class provenance is a
read-only projection of each frame's ``class_declaration``, rather than a second
stack. ``ClassNamespaceScope`` retains its source node and derives compile-time
bindings from it. Scope entry restores the prior frame stack on normal exit and
exceptions. The :download:`scope context refactor
<../../examples/lexical_scope_context_refactor.py>` promotes the existing lookup
members to this ancestor before moving the ancestor into ``lexical_scopes``.
Its reusable three-stage ``OWNERSHIP_PLAN`` composes those declaration moves;
the complete plan also changes the visitor's scope entry and decorator source.

Type-keyed behaviour descent uses ``ClassMethodPromotionSafetyProfile`` before
changing a method's class owner. The shared profile retains dependencies on
private-name mangling, ``__class__``, ``super()``, evaluated defaults and
class-local annotations. A dependency without a preservation proof rejects the
descent. Registry-family closure separately rejects descendants outside the
proved type bindings, including a subclass that inherits and overwrites a
parent's registry key. The recorded :download:`behaviour descent safety refactor
<../../examples/behavior_descent_safety_refactor.py>` adds the shared proof and
replaces repeated parameter enumeration with the lexical binding authority.

``python_module_identity`` owns importable module names derived from source
paths. The former ``ast_tools`` exports refer to the same declaration objects;
production consumers import from the owning modules.

The ``call_binding`` module owns Python signature and argument-binding
declarations. ``value_expression`` owns the shared exact-reference and opaque
value model. Neither depends on product-flow collection. ``product_flow``
re-exports their public declarations as the same objects; repository consumers
import from the owning modules.

Method selection checks the selected binding against its source declaration.
Exact aliases retain the captured declaration; other reassignments and deletions
cannot authorise edits to an older method. A later method definition can
supersede an earlier assignment. Annotation-only
statements do not replace module or class values, while function-local
annotations retain their lexical binding effect.

Import mutations retain the origin selected at their source position, derived
by ``ImportBoundNameProjection``. The class index consumes the same projection.
Imported call selection follows exporting namespace bindings and re-export
chains, including relative imports and class aliases. Rebound exports cannot
authorise edits to an older declaration; cyclic re-exports carry
``CYCLIC_BINDING`` rather than selecting an arbitrary declaration.

``CompactExactValueAlias`` records the source reference and its evaluation
position before assignment targets are written. Callable lookup follows those
facts through module and local aliases and same-class method aliases, including
inherited static and class methods. Later rebinding of the source name does not
change the captured declaration. Alias cycles are tracked by source binding
events, allowing repeated assignments to the same name without treating them as
cycles. Conditional aliases remain unresolved and retain their possible callee
identities for codemod selection checks.

Descriptor transfers across classes or through attribute access remain open
where receiver binding has not been established. Callable identity alone does
not prove that the original call signature applies to such transfers.

``ResolvedCompactFunctionTarget`` retains ``CompactDescriptorAccess`` alongside
the declaration. Direct descriptor access, class lookup and instance lookup
select their implicit-argument rules from ``CompactFunctionBindingKind``.
``CompactResolvedFunctionCall`` retains that resolved target and derives its
callee, call signature and argument binding from it. Call-edit and carrier
refactoring consumers use the resolved call rather than reconstructing binding
from the callee alone.

An instance method accessed through its class requires an explicit receiver;
instance lookup supplies it. Class methods bind their receiver through either
class or instance lookup, while static methods receive no implicit argument.
A raw classmethod descriptor retains declaration identity but reports
``INVALID_DESCRIPTOR_ACCESS`` when used as a call target. Declaration-only
``bind_call`` defaults to instance lookup; its ``access`` argument selects a
different explicit lookup form.

The codemod surface models source-anchored candidate rewrites and simulations.
``DeclarationDecoratorsSourceAuthority`` owns the decorator region independently
of a class or function's header and suite. ``FunctionDecoratorsSourceAuthority``
is its function-only refinement, sharing the same rendering implementation.
``SourceTextGeometry`` resolves decorator
markers from tokens, including parenthesized multiline expressions whose AST
positions begin after ``@``. Statement moves and deletions use that same source
geometry; moved declaration text is derived from its source rather than stored
as an independent copy.

``codemod_declaration_operations`` owns ``DeclarationMutationOperationABC``,
``DeclarationDecoratorsPayload`` and ``ReplaceDeclarationDecoratorsOperation``.
The function-specific operation combines the same payload with the function
mutation contract. Both derive their wire fields from the shared payload
declaration and retain distinct operation identities.

``InsertClassMemberOperation`` in ``codemod_class_operations`` derives a
``ClassMemberSource`` from one authored declaration and emits the existing
``ClassMemberInsertion`` edit. Member identity comes from lexical binding
projection, while indentation and insertion position come from
``ClassBodySourceAuthority``. The payload can contain a method, nested class,
assignment or annotation-only field binding one name. Imports and multi-member
blocks are not member declarations for this operation.

Coalescing retains the supplied member order within each destination class.
Identical insertions for one name merge; conflicting sources or collisions with
existing direct members fail. Inherited members may be overridden. The operation
does not establish behavioural equivalence of a new or overriding member.
``EnumKeyedQueryMemberInsertion`` owns canonical name ordering for the separate,
source-proved generated enum-query family, allowing independent query recipes
to compose to identical source. Authored insertions do not use that policy.

``codemod_assignment_operations`` owns module and named-scope assignment
replacement. Its operations share ``AssignmentReplacementOperationABC`` and
the statement geometry in ``codemod_statement_source``. ``codemod`` exports
the same operation classes.

``ReplaceModuleAssignmentOperation`` derives the selected name from its
replacement source. ``ReplaceScopeAssignmentOperation`` selects a direct
class or function assignment by ``assignment_name``; the replacement may use a
new name. Both support annotated fields without initialisers. The replacement
must be one direct-name assignment. Ambiguous selection, partial selection of
a multi-name assignment, and removal of embedded comments are rejected.
Neighbouring statements, suffix comments and file-ending bytes are retained;
authored multiline literals retain their contents. Changes to names, types,
initialisation and dependent references remain author-selected semantic changes.

Simulation applies planned replacement source verbatim, including its final
newline. Operation-specific renderers supply any required separators before
handing source to simulation; exact offset edits retain their requested bytes
through physical edit conversion.
A clean current-snapshot simulation is not an application recommendation;
export and application require a proof across reachable refactor trajectories.
Closed execution axes are declared in ``codemod_semantics``.  Their enum
members own validation, composition, ranking, and presentation behavior, while
``codemod`` explicitly re-exports the same objects as its public facade.  The
facade is an import surface, not a second semantic authority.
The refactor concept lattice is declared independently in
``refactor_concepts``.  Recipe synthesis projects findings onto that lattice
through executable declaration MROs, so concept declarations do not depend
back on the planner that inherits them.
Canonical import parsing and rendering records are declared in
``codemod_imports``.  Module import mutations consume those records through the
same ``codemod`` facade, so import syntax has one declaration owner while
existing public imports retain object identity.
Its cancelable-composition signal is generic: it treats pack, unpack, and
field-forwarding wrappers as factorable product morphisms when they preserve
common fields and do not own an invariant.
Codemod operations, selectors, and payload records declare wire semantics with
``codemod_payload_field`` on their dataclass fields.
``DataclassJsonReport`` derives a shallow JSON object from a concrete
dataclass's declared fields.  Report instances do not expose ``to_dict`` and
therefore cannot erase their nominal type inside semantic code.
``json_report_object`` is the single explicit object-erasure entry point; it delegates
through the report type's MRO-owned ``project_json_object`` contract.
``DataclassPayloadProjection`` derives each binding catalogue and JSON
projection from explicit field codecs when decoding or value conversion is
required.  ``FlattenedPayloadRecordValueCodec`` lets
a nested nominal record, such as a source-rewrite target, own a flattened wire
projection without an envelope exception.  ``CodemodPayloadRecord`` inherits
object decoding and unknown-field rejection from the same nominal declaration;
``DiscriminatedPayloadRecord`` adds inherited discriminator decoding and
projection while each polymorphic family resolves its own nominal registry.
``PayloadRecordValueCodec`` and ``PayloadRecordArrayValueCodec`` reuse those
declarations for nested records, including proof-carrying ``AuthorityClaim``
values.  ``RecipeCallReplacement`` composes the target-reference and exact
source-transformation declarations through nominal inheritance, so its flat
payload does not require a second old/new source schema.  No parallel schema,
payload carrier, or boundary-role catalogue is maintained.
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
``PromoteClassMembersToAncestorOperation`` is the authored counterpart for a
known ownership decision.  It accepts only the source class, its existing
ancestor, and the selected member identities.  Current source supplies the
member syntax, declaration order, insertion and deletion geometry, ancestry,
and ownership-sensitive promotion checks.  The operation rejects cross-module
moves, destination collisions, class-local field dependencies, name mangling,
and hazardous method forms rather than asking the plan to carry derived source.
Module-symbol moves likewise derive canonical source-module re-exports from the
destination module and moved declarations.  They reject import cycles rather
than accepting caller-authored import text that can disagree with the move.
Destination dependency imports precede the declarations that consume them,
including when the destination is a newly created empty module.
Module-assignment replacements derive the selected assignment name from their
single replacement declaration instead of serializing that name twice.
Function signature and body mutations share one current-source proof of their
typed function target.
Function-signature replacements carry only their parameter and return suffix;
the targeted declaration supplies its function name and sync or async kind.
Whole-target replacements parse exactly one class or function declaration and
re-prove its concrete declaration kind and name against the current indexed
target before producing a physical edit.
``SourceTextPatch`` owns a non-empty ordered sequence of exact old/new source
transformations and applies each transformation to the preceding result.
``PatchTargetOperation`` composes that declaration with the source-reproved
target operation axis and compiles the result to one revision-checked physical
rewrite.  A missing or ambiguous intermediate match fails preflight without
applying a partial patch.
Direct class-base mutations share one current-source proof and edit shell; the
nominal add and remove operation leaves own only their respective header
transformation.  Semantic no-ops preserve the original header source.
Target-adjacent insertions likewise share one current-source proof and physical
edit shell; the before and after leaves own only their insertion geometry.
Class and module assignment deletions share one exact statement-selection
authority.  A chained assignment is deleted only when every name it binds is
selected, so a compressed plan cannot remove an unrequested binding.
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

.. automodule:: nominal_refactor_advisor.json_reports
   :members: JsonReport, DataclassJsonReport, SemanticRecord, json_report_object

.. automodule:: nominal_refactor_advisor.codemod_payload
   :members: DataclassPayloadProjection, CodemodPayloadRecord, DiscriminatedPayloadRecord, codemod_payload_field, FlattenedPayloadRecordValueCodec, PayloadRecordValueCodec, PayloadRecordArrayValueCodec, PayloadValueCodec, PayloadBindingSet

Rejected ``FindingRecipeSynthesisRecord`` values expose ``proof_obstacles``.
Each obstacle identifies the nominal executable declaration that failed to
prove a recipe and carries that declaration's diagnostic.  The record's
``reason`` is a concise summary; consumers that need diagnostic proof detail
should inspect the structured obstacles instead of parsing that summary.
``evaluation_declaration`` names the concrete detector, strategy, or builder
whose MRO assessed the finding. An executable candidate's evaluation
declaration is also the nominal owner that supplied its recipe behavior;
evaluator-only declarations remain explicit without being misreported as
executable.

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
``CollapseIntermediateClassAuthorityOperation`` is the related but distinct
leaf for an intermediary whose members have already moved to its direct
ancestor in an earlier plan stage.  It requires the intermediary to be free of
state, behavior, decorators, class keywords, non-neutral external bases, and
non-base references.  It then derives the entire child cohort, replaces the
base edge, removes obsolete imports, and deletes the empty declaration.  A
``CodemodPlanSequence`` can therefore express member promotion followed by
family collapse while each stage is re-proved against the previous stage's
projected source.
``RenameTopLevelDeclarationAuthorityOperation`` renames one exact class,
function, or async-function declaration and derives its direct imports,
transitive same-name re-exports, preserved aliases, explicit public exports,
and lexically resolved direct, qualified, or forward-annotation references
across the repository.  Unrelated and shadowed bindings remain unchanged.
Nested imports, rebinding, affected star-import boundaries, unresolved export
policy, reflective strings, comments, explicit global/nonlocal declarations,
and binding collisions fail preflight.  Ordered plan stages can chain these
renames while every stage re-proves its edits against the preceding source
state.

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
When a document creates a module, its declared initial source becomes part of
the document's rewrite snapshot before any operation is compiled.  Imports,
symbol moves, preflight proofs, and parse validation therefore resolve against
the same source that will be written; an empty placeholder is never a second
authority for the new module.
``MoveSymbolsToModuleOperation`` and ``ExtractSymbolsToNewModuleOperation``
validate an explicitly complete symbol set.  ``MoveSymbolClosureToModuleOperation``
and ``ExtractSymbolClosureToNewModuleOperation`` instead accept semantic root
declarations and derive their transitive movable source-local dependencies.
The ``Move*`` operations target an indexed module; the ``Extract*`` operations
create their destination atomically.  All four derive dependency imports,
source re-exports, and insertion geometry from the current source snapshot.
Closure moves retain their requested declaration objects alongside the full
derived selection.  ``ModuleMoveDependencyReport`` projects requested, moved,
and derived symbol names and counts from that one selection authority, making
closure growth visible without persisting a second name roster.  Every closure
operation declares a ``maximum_moved_symbol_count``; preflight rejects a
source-derived closure beyond that explicit review bound.
The extraction variants also derive the new-file revision contract.
Annotated declarations require matching source and destination annotation
evaluation modes.  A mismatch appears in the dependency report and fails
preflight before source edits are compiled.
``DeclarationDependencyProjection`` resolves dependencies within their lexical
scope, partitions annotation uses from executable uses, and respects the
sequential bindings of a class body.  A binding inside one nested scope cannot
hide a dependency evaluated by an enclosing declaration.
An import binding used only by moved declarations is removed from the source
module; bindings with any remaining static name or string reference are
retained.  A source-local dependency that is not one unambiguous movable class
or function leaves the closure unproved and fails preflight.
Dependencies introduced by a star import are transferred only when the imported
module's source proves both public exposure and one canonical underlying
binding.  Dynamic export surfaces and multiple possible origins fail closed;
the planner never invents a binding from an export policy alone.
``ModuleMoveDependencyReport.import_dependencies`` is the authoritative import
transfer record.  Each ``ModuleMoveImportDependency`` carries one parsed
binding, its canonical module identity, its execution scope, its destination
requirement, and its source-removal decision.  Source spelling is a derived
presentation and removal surface; destination spelling is rendered from the
identity at the destination package depth.  A same-named destination binding
with a different identity is an explicit proof failure.  The flat name and
source properties in the report remain derived presentation views of those
rows.
Repository-local explicit ``from`` imports are redirected to the declaration
owner while mixed imports retain their unmoved aliases.  Generated imports
preserve future, absolute, and relative source groups; the source-module
reexports preserve names on the module's declared public surface.  Private
closure helpers are imported back only when retained source still references
them, and are not published as compatibility aliases.

.. automodule:: nominal_refactor_advisor.codemod_semantics
   :members: RewriteOperation, CodemodSourceDependencyScope, CodemodBackend, FindingRecipePlanningHorizon, FindingRecipeSynthesisStatus, CodemodPreflightStatus

.. automodule:: nominal_refactor_advisor.codemod_preflight
   :members: CodemodOperationPreflightReport, CodemodOperationPreflightError, CodemodPlanPreflightReport

.. automodule:: nominal_refactor_advisor.codemod_import_scopes
   :members: ModuleImportScope, TypeCheckingGuardReference, TypeCheckingGuardProjection

.. automodule:: nominal_refactor_advisor.codemod_import_bindings
   :members: ModuleImportBinding, ModuleImportBindingIdentity, DirectModuleImportBindingIdentity, FromModuleImportBindingIdentity

.. automodule:: nominal_refactor_advisor.codemod_imports
   :members: ImportSourceGroup, TypeCheckingGuardImportInsertionPoint, ImportAliasRequirement, RequestedImportStatement, RequestedImportBlock, ImportFromModuleName, ImportFromSource, ModuleImportInsertionPoint, ImportNameRemoval, ImportBoundNameRemoval, ModuleImportMutation

.. automodule:: nominal_refactor_advisor.codemod_import_graph
   :members: SourceModuleImportGraph

.. automodule:: nominal_refactor_advisor.codemod_module_declarations
   :members: ModuleSymbolTable, SourceTopLevelDeclaration, NamedSourceTopLevelDeclaration, AssignedSourceTopLevelDeclaration, SourceTopLevelDeclarationIndex, MovedTopLevelDeclarationSource

.. automodule:: nominal_refactor_advisor.codemod_module_move_reports
   :members: ModuleMoveImportDependency, ModuleMoveObstacleKind, ModuleMoveObstacle, ModuleMoveDependencyReport

.. automodule:: nominal_refactor_advisor.declaration_dependencies
   :members: DeclarationDependencyUse, DeclarationDependencyProjection, FunctionBindingProjection

.. automodule:: nominal_refactor_advisor.codemod_source_edits
   :members: SourceNodeDecoratorPolicy, ReplacementSource, SourceEditOrigin, SourceRewriteContributor, NominalSourceEdit, PhysicalSourceEdit, PhysicalSourceEditConflictError, SourceSpanEdit, SourceSpanReplacement, SourceSpanDeletion, SourceInsertion, SourceFileCreation, SourceTextSpanReplacement, SourceTextSpan, SourceTextReplacement, SourceTextPatch, SourceNodeSpan, SourceTextGeometry, SourceTargetEditor, SourceLineSpan, CodemodSourceRevision, CodemodSourceRevisionError

.. automodule:: nominal_refactor_advisor.codemod_declaration_source
   :members: PythonExpressionSourceFormatter, ClassHeaderSpanSourceAuthority, ClassSourceAuthority, ClassBodySourceAuthority, FunctionSignatureSourceAuthority

.. automodule:: nominal_refactor_advisor.codemod_paths
   :members: ExactSourcePathResolution, NormalizedSourcePathResolution, ResolvedSourcePathResolution, RelativeSuffixSourcePathResolution, SourcePathCandidateSet, SourcePathCandidateAuthority, SourcePathResolutionAuthority, SourceCreationPathAuthority

.. automodule:: nominal_refactor_advisor.codemod_architecture_guards
   :members: ArchitectureGuardConstraint, ForbiddenCallArchitectureGuardConstraint, ForbiddenAttributeArchitectureGuardConstraint, ForbiddenDispatchArchitectureGuardConstraint, ArchitectureGuardTargetScope, ResolvedArchitectureGuardTargetScope, ArchitectureGuardRule, ArchitectureGuardRuleResolution, ArchitectureGuardViolationTarget, ArchitectureGuardViolation, ArchitectureGuardReport, ArchitectureGuardSuite, ArchitectureGuardSuitePayloadValueCodec, evaluate_architecture_guards

.. automodule:: nominal_refactor_advisor.cancelable_composition
   :members: CancelableCompositionKind, CancelableCompositionSignal, detect_cancelable_composition_signals

.. automodule:: nominal_refactor_advisor.refactor_concepts
   :members:

.. automodule:: nominal_refactor_advisor.codemod_operations
   :members: RefactorRecipeOperation, SourcePayloadOperation

.. automodule:: nominal_refactor_advisor.codemod_call_declarations
   :members: ModuleCallDeclaration, ModuleCallDeclarationSelector, DeleteModuleCallDeclarationsOperation

.. automodule:: nominal_refactor_advisor.codemod_runtime
   :members: CodemodSourceSnapshot, RefactorRecipeOperationCompiler, RefactorRecipe, CodemodPlanRoot, CodemodPlanDocument, CodemodPlanSequence, CodemodPlanDocumentSimulation, CodemodPlanSequenceSimulation, CodemodSimulationReport, FindingRecipeProofObstacle, FindingRecipeSynthesisRecord, FindingRecipeFrontierBudget, FindingRecipeTrajectoryFrontier, format_codemod_unified_diff, apply_codemod_simulation, simulate_planned_rewrites, codemod_plan_from_findings

.. automodule:: nominal_refactor_advisor.codemod
   :members: CreateFileOperation, ModuleImportBinding, ModuleMoveImportDependency, ModuleMoveDependencyReport, MoveSymbolsToModuleOperation, MoveSymbolClosureToModuleOperation, ExtractSymbolsToNewModuleOperation, ExtractSymbolClosureToNewModuleOperation, ReplaceTargetOperation, PatchTargetOperation, InsertBeforeTargetOperation, InsertAfterTargetOperation, DeleteTargetOperation, EraseDeadCompatibilityOperation, DeleteModuleCallDeclarationsOperation, RenameTopLevelBindingAuthorityOperation, RenameTopLevelDeclarationAuthorityOperation, ReplaceDirectClassBaseOperation, CollapseRedundantClassAuthorityOperation, CollapseIntermediateClassAuthorityOperation, PromoteClassMembersToAncestorOperation, CarrierFieldProjection, ReplaceFieldsWithCarrierOperation, FactorExactDataclassFieldAuthorityOperation, PromoteExactDataclassFieldsToExistingAuthorityOperation, FactorExactMethodRoleOperation, PromoteExactLeafMethodsToAncestorOperation, CollapseClosedParameterConveyorOperation, CollapseDeclaredCarrierExpansionOperation, DeriveClassFamilyCollectionOperation, DeriveEnumSubsetOperation, DeriveDataclassPayloadProjectionOperation, DeriveDataclassFieldNameCollectionProjectionOperation, DeriveDataclassKeyValueSequenceProjectionOperation, DeriveDataclassConstructorProjectionOperation

.. automodule:: nominal_refactor_advisor.class_authority_collapse
   :members: ClassMethodBehaviorAuthority, ClassAuthorityCollapseProofContext, RedundantClassAuthorityCollapseProof, IntermediateClassAuthorityCollapseProof

.. automodule:: nominal_refactor_advisor.declaration_authority_rename
   :members: TopLevelBindingRenameTarget, DeclarationAuthorityImportReference, DeclarationAuthorityRenameBindingClosure, DeclarationAuthorityModuleReferenceProof, DeclarationAuthorityModuleRenameProof, DeclarationAuthorityRenameProof

Goal Trajectory Surface
-----------------------

The goal runner reports complete reachable-state evidence separately from the
single replay sequence.  ``PROVED`` means that exhaustive exploration found one
unique terminal source state.  Ambiguity or any frontier, depth, or state-budget
obstacle keeps ``stages`` empty and prevents application.
Finding deltas and goal progress inherit one ``CodemodFindingIdTransition``
algebra over ``before_ids`` and ``after_ids``.  Finding-specific and
target-specific names exist only in their JSON presentations.  Expected
removals belong to ``CodemodFindingDelta``; a goal stage derives that delta from
its progress and its class plan instead of retaining a second transition
carrier.  Class-plan and site-plan projections likewise inherit
``CodemodFindingClassDelta``; the shared delta owns change aggregation and each
leaf derives its expected-removal scope from its nominal plan declaration.
Target-free states that increase any complete-scan finding class relative to
the starting state are recorded as unjustified-debt terminals rather than
accepted as successful refactors.  The search may pass through intermediate
states with additional obligations, but a proved terminal must discharge them.
``FindingObligationClass`` derives that identity from the detector class in the
MRO that physically declares the executed ``finding_spec``.  Capability-gap and
relation-context prose remain presentation only.  A relation observed by a
different detector is therefore treated as newly introduced unless both
detectors genuinely inherit the same nominal spec owner.
Caller-supplied architecture guards are evaluated at terminal states and stored
on the final replay document; recipe-owned guards remain transition-local.
``ArchitectureGuardRule.constraints`` is a registered discriminated family.
Each constraint declaration owns its JSON key, payload fields, source
observation, violation kind, and diagnostic.  The supported constraint keys are
``forbidden_calls``, ``forbidden_attributes``, and ``forbidden_dispatch``.
``ArchitectureGuardTargetScope`` keeps a module-relative path and optional
nominal qualname together, so target-specific rules do not rely on parallel
path and target arrays and remain portable across checkouts.
Dispatch subjects are parsed Python expressions.  The dispatch constraint
matches literal and enum-member comparisons, ``isinstance`` and ``type`` case
recovery, ``match`` statements, and inline mapping dispatch.  An optionality
check against ``None`` is not classified as semantic case dispatch.
``DispatchToPolymorphismOperation`` derives this target-scoped dispatch guard
from the same current-source proof that derives its strategy family.  Simulation
materialises the rule into the recipe and staged replay carries it forward, so
a later operation cannot reintroduce the removed branch axis silently.

.. code-block:: json

   {
     "architecture_guards": [{
       "rule_id": "declaration-owned-status",
       "constraints": [{
         "constraint": "forbidden_dispatch",
         "subjects": ["status.phase", "declaration"]
       }],
       "scopes": [{
         "file_path": "presenter.py",
         "target_qualname": "StatusPresenter.present"
       }],
       "reason": "semantic cases execute on their nominal leaves"
     }]
   }

.. automodule:: nominal_refactor_advisor.codemod_workflow
   :members: CodemodRefactorGoalRunner, CodemodRefactorGoalReport, CodemodRefactorTrajectoryBudget, CodemodRefactorTrajectoryProof, CodemodRefactorTrajectoryStatus, CodemodRefactorTrajectoryObstacle, CodemodRefactorUnjustifiedDebtTerminal


Result Records And Taxonomy
---------------------------

See :doc:`theory_and_results` for the frozen result dataclasses, taxonomy values,
and pattern metadata referenced by the public entrypoints.
