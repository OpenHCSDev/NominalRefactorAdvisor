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

``ClassDeclarationIndex`` owns a class-declaration map and derives identity,
child, ancestor and descendant views from it. ``ClassFamilyIndex`` specialises
the record type to ``IndexedClass``; ``CompactClassFamilyIndex`` specialises it
to ``CompactIndexedClass``. Their constructors take only ``classes_by_symbol``.
The former constructor arguments for derived maps are no longer independent
inputs. Both builders use the shared unique-identity gate before resolving
bases, leaving colliding class symbols unproved.

Individual ancestor and descendant queries cache only the requested root's
reachability. The bulk map properties materialise their complete derived views
on demand. ``DirectedGraph`` owns ordered adjacency, reversal and cycle-safe
breadth-first traversal; it uses a deque and retains declaration order among
neighbours. Reachability order is distinct from Python's C3 MRO. Method lookup
continues to use ``ClassMroAuthority`` below.

``ReplaceDeclaredCallTargetOperation`` selects calls by their current declaring
function or method and replaces only their callable expression. ``target`` names
the caller scope, ``callee`` names the existing declaration,
``expression_source`` supplies the new callable expression, and
``selection_count`` constrains the number of selected calls. Argument bytes are
retained. Callable-region comments, unresolved selections and invalid expression
syntax are rejected. The replacement's binding and behaviour remain authored
decisions, as with ``ReplaceDeclaredCallOperation``. Both operations use the same
selection, expression parsing, source geometry and payload contracts.

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

``CompactFunctionFlow.binding_resolution_for`` owns entry-binding and source-write
selection for lexical and class-method lookup. Explicit use positions select
the preceding write or the parameter's entry binding; deferred module and class
namespace lookup uses the final write when branch uncertainty is absent.
Deferred closure lookup with multiple
possible bindings remains unresolved, including an entry value followed by a
write. At an explicit read position, a dominating write supersedes earlier
conditional writes when the flow ordering proves they cannot occur after it.
Intervening or repeating writes remain unresolved; proved future writes do not
obscure an earlier value. Header bindings are included even when the write itself
has no child-suite branch path. This source-selection rule does not establish call
activation or native-object integrity. The :download:`authored overwrite fix
<../../examples/binding_overwrite_order.json>` applies it through the DSL.
Bound call-result queries use the same write selection. For an attribute result
such as ``owner.child.result``, replacing
``owner`` or ``owner.child`` invalidates that result's provenance. Receiver
writes also leave attribute results unproved: different lexical receiver names
do not establish different objects, and attribute or item setters may affect
other slots. Local-name result bindings remain independent of these receiver
writes. An unresolved binding cannot supply the declared
return type used by carrier-expansion refactor proofs.

``CompactCallTargetReference.resolve`` selects the target's lookup contract
through its declared MRO. Bare and qualified lexical targets share
``LexicalCallTargetReference``; current-class member targets supply their
member-lookup behaviour. ``CompactProductFlowRepository`` implements
``CompactCallTargetResolverABC`` with its concrete context and resolution types.
Unknown and ambiguous targets retain their nominal unresolved results.

``CompactMutationKind`` distinguishes function and class definitions. Each
definition member selects its resolution behaviour; import and definition
identity flags are derived from those declarations. ``CompactCallTargetResolution``
represents the resulting callable target. Its ``declaration`` property is the
function projection; a ``ResolvedCompactClassTarget`` instead supplies a
construction projection through ``resolve_construction``.

Constructor lookup uses the same lexical scopes, reaching bindings, aliases
and imports as function lookup. Enclosing parameters and local definitions
therefore take precedence over a same-named module class. Function-local classes
absent from the class index remain unresolved under their own qualified name.
A resolved class definition does not establish the type or unchanged lookup
behaviour of an instance returned by its constructor. Plain-product schema and
runtime checks remain required for product-construction proofs.
Class decorators can also replace the class object bound to the declared name.
The current declaration lookup does not prove that runtime identity relation;
it remains an open obligation alongside native namespace-slot integrity.

``CompactFunctionFlow.callable_reference_uses`` retains lexical reads outside
call-target positions, including names whose callable identity is unknown.
Attribute reads retain their lexical subexpressions in evaluation order, so
``function.__call__`` preserves the use of ``function`` itself. Call targets belong to
``calls``; assignments and deletions belong to ``mutations``. The repository
derives ``callable_escapes`` from all retained non-call uses, including unresolved
targets. Collection does not filter reads against a separate inventory of
function, method or import names.

``CompactFunctionFlow.evaluated_results`` retains immediate value dispositions
for assignments, assignment expressions, expression statements and returns.
``CompactEvaluatedResult`` owns the captured value use, destination, disposition
position and exact statement span. A bare return has no value use; an explicit
``return None`` retains its expression. A directly evaluated call shares the
same ``CompactValueDestination`` object as its enclosing result. Nested calls
retain their own destinations. Assignment destination selection belongs to
``CompactValueDestination.for_assignment``. Exact alias recording derives its
lexical reference from the captured result instead of projecting the RHS again.

An immediately evaluated call also retains its actual ``CompactFunctionCall``
in ``CallResultValue.invocation``. Arguments and computed mutation receivers or
indices share those captured call objects. An enclosing container, operator or
``await`` expression retains its own opaque value rather than adopting a nested
call's result. Capturing an invocation does not establish its returned object or
successful completion.

Captured call records use ``DataclassGraphValue`` for structural equality and
hashing. Participating immutable dataclasses inherit this owner with ``eq=False``;
their native field declarations, including inherited fields and ``compare`` and
``hash`` options, select the compared and hashed values. Shared subgraphs are
visited once within each operation, independently of their physical sharing
topology. Non-participating values and custom comparison or hashing overrides
retain their native semantics. Traversal state is temporary and is not pickled;
cycles reached by the traversal raise ``ValueError``. The
:download:`captured-call graph refactor <../../examples/captured_call_graph.json>`
applies 22 authored DSL stages against ``22afe9f``.

``CompactFlowPosition.may_precede`` excludes proved future events within one
flow. ``CompactControlBranchKind`` owns repeating-suite and try-stage ordering.
Shared loop bodies and repeated header evaluations remain conservative across
iterations. These receipts describe captured values, not completed function
paths: a later ``finally`` return can replace an earlier return, and unresolved
calls or definition execution still need their own proof. The
:download:`evaluated-result refactor <../../examples/evaluated_flow_results.json>`
applies 22 authored DSL stages against ``ebb6476``.

``ParsedModule.native_compilation`` lazily provides native compiler evidence over
the original module source. ``NativePythonCompilation.compile()`` returns a
transient code object without executing the module. ``execution_for`` accepts a
``SourceByteSpan`` and returns a ``NativeFunctionExecution`` receipt. Exact
receipts retain compiler flags; ``NativeFunctionExecutionMode`` derives ordinary,
generator, coroutine or async-generator execution from those flags. The evidence
describes raw function code, not an object subsequently returned by a decorator.

Rejected compilation, missing debug ranges, absent emitted code and ambiguous
source spans produce explicit open receipts. Definitions are not recovered by
name or first line. The cached index contains compact receipts and shared source
and interpreter provenance, not executable code; queried modules remain
pickleable. AST-span validation uses the same compilation owner and preserves
the compiler's syntax error. Exact source signatures are owned by
``source_identity.python_source_cache_signature``; the existing ``ast_tools``
import reexports the same function. The
:download:`native-compilation refactor <../../examples/native_compilation.json>`
applies 11 authored DSL stages against ``22afe9f``.

``CompactFunctionDeclaration.execution`` retains the native receipt selected by
the original function's full source span. Its ``line`` and ``end_line`` properties
derive from that receipt's ``SourceByteSpan``; they are not constructor fields.
The declaration collector holds the existing ``ParsedModule`` and shares its
lazy compilation owner. Collecting a module without function declarations does
not request native compilation. Ordinary, coroutine and generator code remain
distinguishable even when their captured bodies otherwise agree. Unavailable
compiler evidence remains attached to the declaration as an open receipt.
The :download:`declaration-execution refactor <../../examples/declaration_execution.json>`
applies seven authored DSL stages against ``4515378``.

Native indexing skips disassembly when a code object's constant tuple contains
no child code objects. Child constants that do exist still require emitted
instruction positions before they contribute evidence. The CPython 3.14 backend
resolves annotation-helper ambiguity through native function creation and
annotation attachment. A unique attached target identifies the raw declaration
body even when its helper has identical names, flags and source geometry.
Compiler support is declared on the backend; other interpreters retain the
span-uniqueness rule.

Creation evidence follows contiguous admitted native operations. Jump and
exception entries end an existing creation region; an independent code load can
start another. Each load site remains distinct, including repeated loads of the
same code object in ``finally``. Multiple candidate targets and unannotated
generic wrappers remain open. Transient instruction and code objects are
discarded after the compact index is built. The
:download:`native-creation refactor <../../examples/native_creation.json>`
applies eight authored DSL stages against ``b79684a``.

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

``DeclarationModuleBindingTransfer`` checks a moved declaration's external
references in its source and destination environments. Each
``DeclarationModuleBindingEnvironment`` derives local declaration ownership
from ``SourceTopLevelDeclarationIndex`` and lookup snapshots from
``ModuleNominalBindingAuthority``. Equal names or values do not establish the
same authority. Rebound declarations remain unproved without evidence for the
particular definition. Builtin shadowing, missing bindings and annotation
representation changes reject the transfer.

``DeclarationDependencyUse`` owns evaluation-phase selection: method-body
references use their lexical execution phase, eager annotations use their
declaration environment, and deferred annotations use final module bindings.
``ModuleNominalBindingAuthority.snapshots_before`` accepts ``None`` for the
final module snapshot and batches it with requested source lines in one pass.
The recorded :download:`module binding proof integration
<../../examples/behavior_module_binding_refactor.py>` replaces the former
name-presence check and batches snapshot traversal through the DSL. A plan may
establish an exact destination import before requesting descent; the later
stage re-proves the reference against that projected source.

``ModuleLexicalDependencyProjection.name_surfaces`` retains both original-source
references and names parsed from quoted annotations. The direct-source view and
dependency-name sets derive from this collection. ``DeclarationDependencyUse``
declares which references have editable source positions, so parsed annotation
names cannot be mistaken for source tokens. The :download:`lexical reference
projection refactor <../../examples/lexical_reference_projection_refactor.py>`
removes the collector's separate name table and derives these views through an
eight-stage DSL plan. Its prerequisites are the declared direct-source policy
on dependency uses and the collector-provided annotation-count field.

``python_module_identity`` owns importable module names derived from source
paths. The former ``ast_tools`` exports refer to the same declaration objects;
production consumers import from the owning modules.

The ``call_binding`` module owns Python signature and argument-binding
declarations. ``value_expression`` owns the shared exact-reference and opaque
value model. Neither depends on product-flow collection. ``product_flow``
re-exports their public declarations as the same objects; repository consumers
import from the owning modules.

Call arguments and binding results preserve a shared value type through
``CallValueT``. ``CompactCallArguments.from_call(node, project_value)`` requires
an explicit expression projector and invokes it once per argument, in Python's
positional-then-keyword evaluation order. Signature binding retains those
objects unchanged. Authored call edits supply ``CompactValueExpression.project``;
other projections can carry richer source facts through the same binder.
Unpacked arguments retain the existing explicit binding limit. The
:download:`value-polymorphic binding refactor
<../../examples/call_value_polymorphism.py>` applies this contract through the DSL.

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

``CompactExactValueAlias.source_use`` retains the already-collected nominal
source read. Its lexical reference and evaluation position derive from that
object, before assignment targets are written. Callable lookup follows those
facts through module and local aliases and same-class method aliases, including
inherited static and class methods. Later rebinding of the source name does not
change the captured declaration. Alias cycles are tracked by source binding
events, allowing repeated assignments to the same name without treating them as
cycles. Conditional aliases remain unresolved and retain their possible callee
identities and candidate bounds for codemod selection checks. Capturing
``self.method`` retains its receiver lookup contract rather than converting it
into an ordinary lexical path. Attribute suffixes on captured non-lexical
targets remain unbounded until their lookup behaviour is established. The
:download:`alias source capture refactor <../../examples/alias_source_capture.py>`
applies this contract after the receiver-binding refactor.

Descriptor transfers across classes or through attribute access remain open
where receiver binding has not been established. Callable identity alone does
not prove that the original call signature applies to such transfers.

``CompactFunctionCall.target_use`` retains a ``CompactCallableReferenceUse``
at the callable's evaluation event, before argument evaluation.
Each reference use retains its exact ``SourceByteSpan``. Its one-based ``line``
is derived from that span, as it is for a call. Separate reads on the same line
remain distinguishable, including UTF-8 names and multiline expressions. An
exact alias retains the original reference-use object. The
:download:`read-site geometry refactor <../../examples/read_site_geometry.py>`
applies this representation change through five authored DSL stages.
``CompactFunctionCall.position`` identifies invocation after the arguments.
The reference-use contract resolves both call targets and non-call references
at their captured positions. Rebinding a callable name inside an argument does
not change the selected target; an unbound local at target evaluation remains
unresolved. The :download:`call target capture refactor
<../../examples/call_target_capture.py>` applies this distinction through the DSL.
This timing fact does not establish descriptor side effects or receiver lifetime.

Attribute call targets retain each lexical receiver read before target capture:
``owner.child.execute()`` records ``owner`` and ``owner.child``. The terminal
``execute`` lookup belongs to the call, rather than an additional non-call use.
Consequently ``function.__call__()`` retains the function-object use required
by callable-escape checks. Bare calls do not create such an escape. The
:download:`receiver capture refactor <../../examples/call_receiver_capture.py>`
applies this collector rule through the DSL.

Non-call attribute reads use the same nominal target projection as calls,
including method values reached through ``type(self).member``. Unresolved
attribute reads retain an explicit dynamic target instead of being discarded
by a separate syntax filter. ``CompactFunctionFlow.loaded_value_root_names``
is a cached view of retained call targets and value reads, not an independently
supplied flow field. The :download:`reference ownership refactor
<../../examples/reference_fact_ownership.py>` removes the collector's parallel
loaded-name state.

``CompactCallableEscape`` retains the complete ``target_resolution`` for each
non-call use. ``callable_escapes_for(symbol)`` includes both exact and possible
references to that symbol. The shared callable-component proof rejects a
signature rewrite when a participant appears among those possible targets;
an unresolved declaration does not erase the escape. The
:download:`escape-evidence refactor <../../examples/escape_resolution_evidence.py>`
applies the declaration rename and its consumers through the DSL.

``CompactFlowOwner`` is the nominal scope-owner contract.
``CompactFunctionDeclaration`` implements it directly: a function flow owns
that declaration object, and its qualified name derives from the declaration's
identity. ``CompactNamespaceFlowOwner`` represents module and class-body scopes
and rejects function scope construction without a declaration.
``CompactProductFlowModuleProjection.function_declarations`` and
``CompactFlowContext.declaration`` derive from flow owners. Repeated
function names therefore retain their individual signatures and source sites;
repository ambiguity checks remain separate. The :download:`flow ownership
refactor <../../examples/flow_declaration_ownership.py>` removes the former
name-based declaration join through the DSL.

``CompactBindingSource`` separates value-origin evidence from callable lookup.
Its ``resolve_binding`` method dispatches to ``CompactBindingResolverABC``
through the selected source declaration. Exact sources pass their retained
mutation; unresolved sources share the inherited possible-binding projection.
``target_lookup_violation`` belongs to unresolved sources only. This dispatch
preserves the existing source-selection and callable-resolution rules; it does
not yet establish captured-object or native namespace integrity. The
:download:`binding-source dispatch refactor
<../../examples/binding_source_dispatch.json>` applies the extraction and
diagnostic cleanup in 18 authored stages.

``CompactFlowContext`` in ``product_flow`` owns the module/flow join formerly
declared in the product repository. ``CompactBindingResolverABC[ResultT]`` uses
that context for shared exact-source cycle handling, alias selection and
mutation-operation dispatch. ``CompactMutationKind.binding_operation`` owns
value, import and definition behaviour, including pending-visit rules and import
origin validation. Callable projections remain in the product repository;
``CompactDefinitionResolverABC`` supplies source-definition selection without
requiring the callable interface. ``introduces_nominal_binding`` identifies a
syntactic binding category, not post-decorator or metaclass object identity.

Both exact and alternative alias paths use the captured-reference projection.
The current callable projection retains the previous suffix-at-capture rule;
it does not yet prove a later slot access unchanged. Imported-name projection
also does not retain import activation evidence. The
:download:`shared binding interpretation refactor
<../../examples/shared_binding_interpretation.json>` moves and factors these
contracts in 21 stages, followed by two corrective stages that use Python's
native relative-name resolver. Imports beyond the known package boundary remain
unresolved instead of producing an invented module name.

``ImportDeclarationABC`` in ``lexical_bindings`` owns an AST-free import request.
``ModuleImportDeclaration`` and ``FromImportDeclaration`` derive binding names,
requested module names and canonical source from their aliases. The existing
``ImportAliasRequirement`` and ``ImportFromModuleName`` declarations live at this
lower boundary; codemod exports retain the same objects. Higher-level import
editing retains its formatting and insertion policies.

``ImportedNameOrigin`` retains the declaration, selected alias position and
source module identity. Its alias, bound name, requested module and qualified
catalogue name are derived. Relative requests remain available when their
absolute module cannot be resolved. ``CompactImportTarget`` retains that origin;
``CompactMutation.imported_origin`` derives from its target rather than storing
a separate string. Import and ordinary assignment targets share
``CompactLexicalBindingTargetABC`` behaviour. The imported-source resolver
receives the actual mutation, context and accessed reference.

These declarations distinguish a module request from import-from member capture,
including explicitly aliased dotted imports. They retain source binding evidence;
qualified catalogue names do not prove runtime object identity or import
activation. The :download:`nominal import evidence refactor
<../../examples/nominal_import_evidence.json>` applies 31 authored DSL stages
against ``2c1fa56``.

``FunctionParameterSource.from_arguments`` in ``lexical_bindings`` retains each
actual ``ast.arg``, its ``CompactParameterKind`` and its default expression in
signature order. Missing defaults remain ``None``; an explicit ``=None`` retains
its ``ast.Constant`` node. Positional default-tail alignment and keyword-only
default pairing belong to this projection. ``CompactFunctionParameter.from_source``
derives the persisted, AST-free parameter used by ``CompactFunctionSignature``.
``call_binding.CompactParameterKind`` re-exports the same enum declaration.

``FunctionDefaultVisitor`` supplies shared default-expression traversal to eager
name reads, declaration dependencies and compact flow collection. Creating a
lambda visits its defaults without entering its body. Declaration dependency
collection additionally retains body references in their deferred lexical scope.
Signature order describes parameter binding; native annotation evaluation order
and captured default values require separate execution evidence. The
:download:`parameter-source refactor <../../examples/parameter_source_defaults.json>`
applies 19 authored DSL stages against ``2ae5a55``.

``CompactDefinitionTarget`` retains the exact ``CompactDefinitionFlowOwner``
shared with its separate body flow. Its lexical binding name derives from that
owner. ``CompactClassDeclaration`` retains the class's full source span;
``CompactFunctionDeclaration.source_span`` derives from its existing execution
receipt. Repeated same-name definitions therefore retain distinct source owners.
The single ``CompactMutationKind.DEFINITION`` delegates source-selection dispatch
to the retained declaration rather than storing a second function/class tag.

The target's ``decorator_uses`` are captured by the enclosing flow. Each use
retains its evaluation position and value expression; decorator factories retain the same
``CallResultValue`` invocation as the flow's call record. Function, asynchronous
function and class definitions supply this payload, including an empty tuple
for an undecorated definition. ``header_position`` follows the header expressions
collected in the enclosing flow and precedes the final binding event. It records
neither the earlier native builder capture nor entry into the class body.
Builder, metaclass and decorator effects require separate execution evidence.
``CompactMutation`` validates the definition-kind and target relation once; its
generic target type carries that contract into definition resolution.

``CompactProductFlowModuleProjection.flow_contexts`` owns the module/flow joins
consumed by the repository. Its derived ``flow_contexts_by_owner`` index preserves
distinct positioned owners and excludes duplicate owner handles. Definition
targets and body contexts retain the same owner object after serialisation.
These receipts describe potentially repeated source sites, not unique runtime
objects or successful creation. The :download:`definition-owner refactor
<../../examples/definition_flow_ownership.json>` applies authored DSL stages
against ``72879a6``.

``ResolvedCompactFunctionTarget.for_object_mutation`` projects decorated or
class-owned function declarations to an unbounded object target. A decorator
or class namespace can install a different object; recognised decorator spelling
does not exempt that obligation. Raw free and local function declarations retain
their existing distinct-object bound. The original source declaration remains
available independently of this mutation projection. Class-result identity and
runtime-call admission require additional evidence. The
:download:`definition-capture refactor <../../examples/definition_captures.json>`
applies 17 authored DSL stages against ``25aacf1``.

``InitialCompactParameterBinding`` retains the exact parameter object from the
owning signature and has no mutation event. Its value origin is that entry
parameter; ``target_lookup_violation`` remains ``DYNAMIC_BINDING`` because an
entry value does not identify a callable. Selected writes and unresolved writes
implement the same value-origin contract through their own leaves. Parameters
reassigned to known functions therefore use the ordinary write resolver.
Entry bindings are constructed only when positioned write selection needs them.
The :download:`initial binding refactor <../../examples/initial_binding_sources.py>`
applies the shared source contract and removes the repository's separate
parameter-name check.

Current-class method and annotated-member lookup requires the receiver's value
origin to remain its entry parameter at the captured read position. Self-aliases
preserve that origin; reassignment to another object or class leaves the target
unresolved. Argument evaluation after target capture does not change that
earlier lookup. The :download:`receiver-binding refactor
<../../examples/receiver_binding_proof.py>` adds this shared obligation.

``CompactCallTargetResolution.candidate_symbols_within(symbols)`` projects the
participants a target can reference. ``UnboundedCompactFunctionTarget`` cannot
exclude any supplied participant; its ``possible_symbols`` are diagnostic
names, not a complete bound. ``AlternativeCompactFunctionTargets`` retains
binding alternatives and unions their candidate queries without flattening
unbounded evidence into names. Escape checks, callable-component proofs and
declared-call edits use this contract. The :download:`alias candidate bounds
refactor <../../examples/alias_candidate_bounds.py>` preserves it across
conditional bindings.

These checks establish receiver-root provenance, not a complete instance
lifetime proof. Member reassignment, descriptor effects and constructor result
types require additional evidence. An unbounded receiver can conservatively
prevent an otherwise unrelated closed-signature edit within the analysed scope.

Collected arguments carry ``CompactValueUse`` through signature binding.
Each use owns its expression and evaluation position; ``origin_in(flow)``
resolves at that position. Opaque expressions return an explicit
``OPAQUE_EXPRESSION`` origin result. Value-origin lookup selects writes through
the shared binding-event resolver and tracks alias cycles by selected write,
rather than rejecting every repeated name.

``CompactMutation`` retains a typed assignment target separately from the write
event. ``CompactBindingTarget`` represents a lexical name binding;
``CompactAttributeTarget`` and ``CompactItemTarget`` retain the evaluated
receiver, and item targets also retain the evaluated index. Their
``CompactValueUse`` records precede the later write, including when index or
right-hand-side evaluation rebinds the receiver's original name. Computed
receivers retain explicit unresolved value evidence rather than being omitted.
``mutations_by_root_name`` derives only lexical binding writes;
``mutated_roots_within`` derives possible affected roots from the target family.
Rebinding an alias does not mutate the object it previously referenced.
The :download:`captured assignment-target migration
<../../examples/captured_assignment_targets.json>` is an authored staged plan
against the repository at ``5a3d16d``. Its changes are corrections checked by
native execution and replay, not automatically established equivalences.

``CompactMutationResolverABC`` separates lexical rebinding from writes through
captured receivers. Targets select their resolution contract; repository
consumers do not dispatch on target types. Unresolved receiver identity remains
unbounded for product-runtime safety, even when callable lookup has diagnostic
candidate names. An annotation alone does not prove that a receiver is a
non-class object.

``declared_product_authorities_by_symbol`` supplies the declaration-proved
product candidates before runtime obligations. ``CompactProductRuntimeFailure``
retains its context, source event and target resolution; owner and line are
derived views. ``CompactProductRuntimeFailureIndex`` provides lazy mapping
queries over shared observations. Membership checks do not construct
per-class diagnostics, and requested diagnostics retain those same observation
objects. ``UNRESOLVED_MUTATION_RECEIVER`` distinguishes unknown identity from a
confirmed class write. Runtime uncertainty can currently prevent a candidate
from reaching conveyor assessment; its source evidence remains queryable here.

Calls in attribute and item assignment and deletion targets retain their
evaluation events. Augmented assignments evaluate and read the target before
their right-hand side, then record the write. An attribute or item annotation
without an assigned value evaluates its target without recording a write. These calls
participate in declaration-resolved call edits. The :download:`assignment-target
evaluation correction <../../examples/assignment_target_evaluation.py>` applies
the collector change through the DSL. Receiver capture and namespace-slot
integrity remain separate proof obligations.

``CompactResolvedFunctionCall.bound_value_uses`` and
``CompactProductConstruction.field_values`` derive their parameter and field
views from these same captured objects. Constructor and forwarding consumers
retain the uses until origin resolution. The :download:`argument value capture
refactor <../../examples/argument_value_capture.py>` applies the change through
the DSL, including removal of the former builder-owned argument indexes.

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

``ProjectFunctionParameterOperation`` and ``ProjectFunctionLocalOperation``
select reads owned by their lexical binding declaration. Their optional
``attribute_path`` narrows the selection to an exact access prefix: for example,
``parameter_name="context", attribute_path=("old_field",),
projection_source="context.new_field"`` rewrites those field reads without
replacing their enclosing statements. The empty path selects whole binding
reads as before. Longer suffixes remain intact, and shadowed roots are excluded.

``FunctionBindingProjectionSourceAuthority`` consumes ``FunctionBindingABC``;
the existing parameter and local binding implementations own lexical scope
resolution. The shared rewriter checks replacement-root capture and rejects
direct writes, deletes and comments inside the selected access span. An access
prefix used to reach a deeper write is a read: replacing ``context.old`` in
``context.old.value = 3`` retains the assignment to ``value``. The operation
proves source selection and lexical ownership, not equivalence of the old and
new attribute values, descriptors or their effects. Signature, caller and
initializer changes remain explicit operations in the enclosing plan.

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
The :doc:`codemod_catalog` lists every registered operation with its current
constructor and declaration documentation.
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
Header capture checks include bindings in both the source and destination
classes. ``ResolvedClassTarget.bound_names`` derives those bindings from the
current declaration; the move context carries no separately supplied copy.
Quoted field annotations use the shared annotation-syntax parser, which keeps
type references distinct from ``Literal`` values and ``Annotated`` metadata.
Ordinary method-body globals remain separate from class-header lookup.
It also checks member lookup throughout the destination's descendant cohort.
The projected lookup must retain each existing owner, except where the selected
member moves from the source to the destination. Native C3 supplies precedence;
an earlier competing branch or a diamond descendant can therefore prevent a
move even when the source class itself would retain the intended behaviour.
Annotation-only names are not treated as installed class members.

``ClassMemberLookupProof`` supplies the shared lookup check for member promotion
and collector migration. Namespace changes project added and removed bindings
over their original declarations; they do not become replacement declaration
identities. Native declarations with identical qualified names remain distinct
when they refer to different Python objects.

``SourceNativeClassMro`` derives reachable hierarchies lazily within one fixed
source context and optional base substitution. Shared ancestors reuse their
closed namespaces and inert C3 types. Replacing the context or substitution
creates a fresh projection cache; unproved class construction is not stored as
a successful resolution.

Source namespace closure uses use-point lexical references and the module's
annotation evaluation mode. Native generic aliases store their arguments;
``typing.ClassVar`` can inspect or hash them. Explicit native descriptor calls
likewise require evidence for argument metadata access. Unknown argument effects,
custom creation hooks and unavailable base declarations leave the move unproved.

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
target before producing a physical edit. ``ReplaceTargetOperation`` replaces
the header and body, preserving the existing decorator block. Its default
``decorator_policy`` excludes decorators, so a replacement payload containing
decorators fails preflight. ``ReplaceDeclarationDecoratorsOperation`` edits the
decorator block independently; both operations can compose on one snapshot.
A nominal operation refinement can select the inclusive decorator policy.
The same policy then validates the payload and selects the complete decorated
source span, including multiline decorator markers.

Replacement indentation derives from
``NamedDeclarationSourceAuthority.declaration_indentation``. The payload can be
unindented or already indented; its class or function remains in the selected
enclosing scope. ``PythonBlockSource`` owns parsing and structural relocation
for declaration replacements, class-member insertion, function bodies,
decorator scaffolds and assignment rendering. It preserves comments, line
endings and multiline literal contents while adjusting code indentation.
An indented payload cannot introduce statements outside its initial suite.
Validation and rendering share the same parsed block; no separately dedented
copy supplies declaration identity. ``SourceTextGeometry.iter_tokens`` permits
prefix inspection without tokenising the complete payload first.
The :download:`declaration replacement layout refactor
<../../examples/declaration_replacement_layout.py>` records this change through
the DSL, including the explicitly authored indentation-getter transfer.

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
ownership move. The
codemod preflight reconstructs the same proof from the current full AST.
Exact-method components retain their source-size cost estimate as descriptive
metadata. That estimate does not gate proof or execution for either promotion
to an existing authority or an explicitly named new role. A two-leaf family is
eligible when its binding obligations are proved, including a one-line method
or property getter. The recorded :download:`method proof and cost separation
<../../examples/method_proof_cost_separation_refactor.py>` removes the former
heuristic veto from both shared component builders. An authored plan can factor
a role and then extract it into a new module; the extraction stage resolves the
new declaration and derives its imports from the preceding stage's source.
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
Qualified annotation references use ``DeclarationDependencyUse`` to select
their module binding snapshot. Eager annotations retain the alias at declaration
time; postponed and lazy annotations use the final module alias. The recorded
:download:`annotation binding phase refactor
<../../examples/annotation_binding_phase_refactor.py>` updates that lookup with
one declaration-resolved call-argument operation, without replacing its enclosing
method.
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

.. automodule:: nominal_refactor_advisor.declaration_binding_transfer
   :members: DeclarationModuleBindingEnvironment, DeclarationModuleBindingTransfer, ClassBodyReferenceCapture

.. automodule:: nominal_refactor_advisor.positional_forwarding
   :members: PositionalForwardingCall

``PositionalForwardingCall`` projects a function's complete callable expression,
required positional-or-keyword parameters and forwarded parameter names. It
accepts a return call with an optional preceding deletion of unused parameters.
Keywords, unpacking, defaults and additional execution remain outside this
projection. Source-backed native functions use the same projection as analysed
source declarations.

``ClassBodyReferenceCapture`` compares a bare module reference's runtime binding
with its binding at class creation. It rejects class-local shadowing, unresolved
or rebound module declarations, and references unavailable at class creation.
Qualified and computed callable expressions remain intact in observations but
are unproved for capture; their attribute and descriptor behaviour requires
additional evidence. Imported bare aliases can retain their original spelling.

Collector migration derives its direct-forwarding relation from the registered
collector implementations. Flattening is not a direct forwarding relation, even
when its inherited method signature matches. Parameter spelling participates in
the relation because callers may supply keyword arguments.

The candidate's ``collector_descriptor_type`` owns the generated descriptor.
Its constant property retains the native type without exposing the type's own
``__isabstractmethod__`` descriptor to the candidate's ABC machinery. Rendering
and validation derive from this same value.
``ModuleNominalBindingView.require_native_type_in_class`` checks the generated
name against that declaration at class creation. Module or class shadowing and
unresolved wildcard exposure prevent the rewrite. Explicit builtin imports and
wildcard imports whose export declarations exclude the name can establish the
required binding.

Native binding witnesses do not yet establish namespace-slot integrity.
Source-visible replacement of a native module attribute can leave the qualified
name unchanged while changing the object it selects. Such mutation remains an
open proof gap for automatic class-member promotion. A saved object and a later
attribute lookup through a saved module require distinct positioned evidence;
neither qualified-name equality nor descriptor recognition proves that relation.

.. automodule:: nominal_refactor_advisor.native_declarations
   :members: QualifiedDeclaration, ClassNamespaceDeclaration, NativeDeclaration

``NativeDeclaration`` derives a qualified name from a Python declaration and
lazily inspects its source. Source matching compares declaration ASTs without
location attributes. Builtin identity is available independently of source;
source-dependent operations reject declarations without inspectable source.

Wrappers of the same loaded declaration share the inspected AST. Cache identity
uses the declaration object, independently of metaclass value equality or
hashability. A newly loaded declaration receives a separate projection even when
its qualified name matches an earlier declaration.

``require_source_matches`` compares every proposed AST with the captured native
projection; its result is not cached. Editing source does not reload a native
declaration, and a changed source declaration remains unproved against its older
projection. The cache retains inspected declarations for the process lifetime.
This is a source-declaration contract, not a proof of arbitrary live monkeypatches.

Collector migration resolves the original and replacement bases by their native
qualified identities and requires their source declarations to match. The
original base reference must resolve at class creation. A same-named
class in another module or an altered declaration under the canonical name does
not establish native authority.

``SourceNativeClassMro`` projects reachable source bases and authenticated native
ancestors into the shared ``DeclarationMroType`` carrier. Python derives the C3
order; a topological traversal only schedules carrier construction. Migration
requires the first inherited binding of the removed method to belong to the
replacement collector's native implementation. This accounts for indirect bases
and earlier competing branches while admitting independent branches. Source and
native classes expose their member names through ``ClassNamespaceDeclaration``.
Custom native MRO implementations, unresolved bases and unproved source class
creation prevent the rewrite. Native generic applications require the inherited
``typing.Generic`` subscription implementation.

``ClassNamespaceExecutionEvidence`` derives final member bindings and creation
effects from the existing ordered lexical traversal. Deleted methods and
annotation-only names do not become inherited method bindings. Native decorator
references require an external binding at their evaluation point; deleting a
shadowing name later does not establish that proof. Calls, operators, imports,
iteration and other executable forms require creation-effect evidence. Native
method descriptors, literal values and sequence construction have dedicated
conditions; arbitrary custom decorators and constructors remain unproved.
Annotation effects follow the module's annotation evaluation mode, including
deferred annotations on Python 3.14. Deferred function and generator bodies are
separate from their immediately evaluated defaults and outer iterables.

These checks establish declaration identity, forwarding and capture conditions.
They do not establish arbitrary multiple-inheritance replacement equivalence,
class-creation hook equivalence or unchanged annotation introspection. The
recorded plans in ``docs/examples/collector_base_authority_refactor.py`` and
``docs/examples/native_declaration_consumers_refactor.py`` migrate consumers
after the shared native-source proof declarations have been authored.
``docs/examples/mro_declaration_carrier_refactor.py`` records the shared carrier
and binding-consumer migration. ``docs/examples/collector_mro_proof_refactor.py``
records the operation's inherited-method gate. These plans consume the authored
namespace and native/source MRO modules. The four-stage
``docs/examples/class_namespace_effect_projection_refactor.py`` records the
separation of node-effect selection from ordered scope traversal.

Generated detector classes expose collector options through ``ClassAliasProperty``
descriptors pointing to their retained ``DetectorDeclaration.options``. The option
names derive from the declaration's existing class-shell field projection; the
generated class does not copy their values. Collector execution reads
``type(self).candidate_collector`` or ``type(self).source_candidate_collector``.
Authored subclass overrides follow native attribute lookup and MRO, including
when the parent class was generated. There is no separate helper selecting
between a declaration's options and a class field at execution time.

``docs/examples/collector_attribute_projection_refactor.py`` records the runtime
projection and call-site migration as an executable DSL plan.

.. automodule:: nominal_refactor_advisor.codemod_source_edits
   :members: SourceNodeDecoratorPolicy, ReplacementSource, SourceEditOrigin, SourceRewriteContributor, NominalSourceEdit, PhysicalSourceEdit, PhysicalSourceEditConflictError, SourceSpanEdit, SourceSpanReplacement, SourceSpanDeletion, SourceInsertion, SourceFileCreation, SourceTextSpanReplacement, SourceTextSpan, SourceTextReplacement, SourceTextPatch, SourceNodeSpan, SourceTextGeometry, SourceTargetEditor, SourceLineSpan, CodemodSourceRevision, CodemodSourceRevisionError

.. automodule:: nominal_refactor_advisor.codemod_declaration_source
   :members: PythonExpressionSourceFormatter, NamedDeclarationSourceAuthority, ClassHeaderSpanSourceAuthority, ClassSourceAuthority, ClassBodySourceAuthority, FunctionSignatureSourceAuthority

``NamedDeclarationSourceAuthority.declaration_line_span`` includes the complete
class, function or async-function declaration, starting at its first decorator's
``@`` token when present. Token geometry includes parenthesised decorators and
comments between decorators. Comments preceding the first decorator are outside
this span. Adjacent insertion operations use this span for placement and retain
the declaration header's indentation; source-index header positions remain
navigation locations rather than complete declaration boundaries.

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

``CodemodPlanSequenceContinuationReport`` retains the projected ``source_index``
as typed provenance. Its default JSON projection includes continuation plans
and finding evidence without serialising that index. In the CLI,
``--codemod-project-source-index`` includes the index once, under
``projected_findings.projected_source_index``. Requesting a continuation plan
does not implicitly request source-index output.

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
