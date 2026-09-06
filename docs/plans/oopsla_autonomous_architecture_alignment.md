# OOPSLA Brainstorming Note: Autonomous Architectural Alignment

Status: aligned with the current OOPSLA paper; empirical evaluation remains open

## Observation

During several days of autonomous OpenHCS work, the agent remained closely
aligned with an architecture that differs substantially from common patterns in
its general training distribution. The user needed to make only a few minor,
highly specific corrections. This is notable because the work crossed repository
boundaries, long-running context, UI and runtime state, declaration systems,
tests, packaging, documentation, and CI.

The motivating reaction was that autonomous model-driven development may be
able to preserve a demanding architecture, rather than merely produce working
local patches.

## Verbatim Message

> The main reason is that your standards became an explicit local reasoning system, not merely a coding preference I had to imitate.
>
> I did not retrain on OpenHCS. What happened was:
>
> - Your architecture is unusually coherent. Once the core rule is understood — semantics belong to their nominal declaration owner, while consumers derive or project them — many design decisions become logically constrained. I do not need to memorize every acceptable pattern.
>
> - The repository contains many working examples of that rule. Those examples outweigh generic training-data habits when I inspect analogous OpenHCS code before designing something new.
> - You clearly identified the dangerous failure modes: mirrored state, parallel registries, caller-side dispatch, stringly typed semantics, fake abstraction layers, structural protocols, and local fallbacks. That gave me a strong negative checklist as well as a positive design model.
>
> - Maintained architecture documents, implementation, tests, and CI provide independent evidence. When they disagree, current declarations and execution paths win. That prevents architectural prose from becoming another detached authority.
>
> - NRA acts as a second-pass critic. It does not design the architecture for me, but it makes me explicitly investigate ownership, semantic duplication, suspicious dispatch, and fake boundaries before accepting a change.
> - The long-running goal kept the architectural model warm. I was not repeatedly reconstructing OpenHCS from scratch, and the persistent project memory preserved the important invariants across context compaction.
> - Your corrections were very specific. “Put behavior on the enum member,” “no Protocols,” or “this is a mirrored authority” teaches a much sharper constraint than broad feedback like “make it cleaner.” I incorporated those as general audit rules and revisited surrounding changes, rather than patching only the line you noticed.
>
> The difference from my training distribution is real. A generic model prior often favors locally understandable service layers, adapters, mapping dictionaries, compatibility fallbacks, and small wrappers. Those patterns are common and sometimes reasonable, but in OpenHCS they can quietly manufacture competing semantic authorities. The remaining minor drift generally happened when that generic local-convenience prior fired before I had traced the full ownership path.
>
> What makes sustained alignment possible is that the OpenHCS architecture is not just aesthetic taste. It is a compact set of composable invariants with abundant evidence in the codebase. Once I treat those invariants as hard constraints and require proof of ownership before editing, the search space narrows dramatically. Your architecture is different from the average codebase, but it is internally legible enough to reason from rather than merely imitate.
>
> So, candidly: it is partly your steering, partly continuity and memory, partly the repository’s own consistency, and partly the verification machinery. The strongest factor is that you have made the architecture sufficiently principled that correct extensions can usually be derived from first principles.

## Working Explanation

The architectural standards became an explicit local reasoning system rather
than a coding style for the agent to imitate.

- OpenHCS has a coherent central invariant: semantics belong to their nominal
  declaration owner, while consumers derive or project those semantics.
- The repository contains many working examples from which new decisions can be
  derived.
- The user identified both the positive model and dangerous failure modes:
  mirrored state, parallel registries, caller-side dispatch, stringly typed
  semantics, fake abstraction layers, structural protocols, and local
  fallbacks.
- Maintained architecture documents, implementation, tests, and CI provide an
  evidence hierarchy. Current declarations and execution paths outrank stale
  prose.
- Nominal Refactor Advisor acts as a second-pass critic. It does not choose the
  design, but it forces explicit investigation of ownership, semantic
  duplication, suspicious dispatch, and false boundaries.
- A persistent goal and project memory preserve the model across long-running
  work and context compaction.
- Specific user corrections become general audit rules. For example, "put the
  behavior on the enum member", "no Protocols", and "this is a mirrored
  authority" constrain later work beyond the originally observed line.

Generic model priors often favor locally understandable services, adapters,
mapping dictionaries, compatibility fallbacks, and small wrapper functions.
Those patterns may be reasonable elsewhere, but in OpenHCS they can create
competing semantic authorities. The minor observed drift occurred when this
local-convenience prior took effect before the complete ownership path had been
traced.

## Candidate Hypothesis

Autonomous architectural alignment becomes practical when a repository supplies
a compact, composable set of invariants that can be used to derive decisions,
plus enough executable evidence and corrective machinery to reject locally
plausible violations.

The important mechanism may not be imitation of repository style. It may be the
construction of a project-local decision procedure that narrows the agent's
search space before code is written.

## Paper Alignment

The paper sharpens the working explanation into a source-fidelity obligation.
For a fixed maintenance task, the implementation must represent every required
answer in the relation over questions and cases, derive those answers from one
logical authority, and retain enough evidence to recover the source judgment
that execution selected. Agreement among duplicated answers is not source
recovery, and an unproved answer remains an assurance gap.

Applied to NRA:

- ``PatternId`` supplies a descriptive pattern vocabulary, while each detector's
  nominal required-relation declaration owns the executable finding semantics.
- ``IssueDetector.required_relation_source`` recovers the selected declaration's
  source rather than inferring ownership from a matching label.
- ``DetectorRefactorCapabilityReport`` derives the loaded detector family's
  contribution contracts, exact C3 resolution paths, and contract-method
  implementation sources. It is static capability evidence, not proof that a
  finding has a valid executable refactor.
- Finding-recipe evaluation, authority claims, source-index preflight,
  simulation, architecture guards, and before/after obligation projection carry
  the repository-specific proof. A missing or ambiguous authority remains an
  explicit failed-closed result.
- The refactoring DSL should remove deterministic syntax work only after its
  selectors and operations retain the source evidence needed to re-prove every
  affected relation in the projected state.

The autonomous-maintenance hypothesis is therefore stronger than style
imitation: a compact local decision procedure is useful only when its internal
representation is semantically complete for the decisions being delegated and
its outputs retain their derivation source.

## Operational Decision Procedure

### Active refactoring objective (6 September 2026)

Make the cost of expressing a refactor track its semantic decisions rather
than its affected lines or files. Correct maintenance requires complete
derivation of the task's answers and recovery of their selected source.
Behavioural agreement alone does not discharge those obligations.

Factor repeated implementation into its declaration owners and compose
overlapping capabilities through multiple inheritance and native MRO. Shared
algorithms belong upstream; concrete implementations supply the irreducible
hooks. Generic consumers use the ABC contract without rediscovering concrete
implementations. Classes can still mirror one another: class count and line
count are not optimisation targets. A class named after another class is an
ownership-audit trigger, not an automatic diagnosis.

The DSL should make existing declarations selectable operands, derive syntax
and import changes, and expose further moves in the projected state so an
agent or practitioner can chain decisions before applying a coherent batch.
Evaluate progress by eliminated independent semantics, preserved source
evidence, and reduced mechanical authoring. Preserve explicit unresolved
obligations; neither a clean detector report nor simulation alone proves
complete correctness. Full autonomous semantic discovery is not a prerequisite
for a useful refactoring language.

The first resumed increment removes mirrored binding-projection variants and
selects owned attribute reads without copying enclosing statements. The
22-stage plan in `docs/examples/access_path_projection.py` reconstructs the
change from its pre-edit source. The earlier member-move plan now uses the new
selector and reproduces the same two module ASTs from its original baseline.

The attempt to use `ReplaceDeclaredCallArgumentsOperation` exposed an unresolved
receiver boundary: a method invoked through a local constructed instance was
not recovered as the selected declaration. The plan uses an explicit body edit
for that caller; it does not claim that the narrower operation succeeded.
Next, trace existing product-flow construction and method-resolution evidence,
factor caller-side concrete dispatch into the nominal contracts, and make that
call edit source-selected without adding a parallel receiver registry. The
source-derived transfer of members between unrelated owners remains another
gap; introducing temporary inheritance merely to enable a move is not a repair.

Validation for this increment: 57 focused tests; 13 new access-path cases also
run under an ASCII locale; full suites of 2,341 passed and 15 skipped on Python
3.11, and 2,356 passed on Python 3.14, using eight workers per suite. All 81
detectors completed with no findings in the touched-source/context audit.
Both saved-plan replays matched the expected module ASTs. The Sphinx build
retained its two existing duplicate-object warnings.

### Review questions

Before introducing a semantic change, ask:

1. Which declaration already owns this fact?
2. Is the proposed representation an authority or a derived projection?
3. Is stored state necessary, or can it be derived from the existing owner?
4. Is a caller interpreting a subtype instead of invoking nominal polymorphism?
5. Is generic machinery being placed in a domain package?
6. Does a fallback conceal a violated contract?
7. Would one semantic addition require coordinated edits across multiple
   consumers?
8. Which source, test, runtime observation, or CI gate proves the ownership
   claim?

## Research Questions

- How much alignment comes from written invariants, repository examples, direct
  feedback, persistent context, and automated critique respectively?
- Does converting corrections into general invariants reduce the rate and
  severity of later architectural drift?
- How well does alignment survive context compaction or a fresh agent session?
- Can ownership and projection claims be represented as proof obligations that
  an advisor can check without becoming another semantic authority?
- Which NRA findings predict genuine architectural drift, and which merely
  detect legitimate fan-out from one declaration?
- How should false positives be handled so the critic prompts investigation
  without encouraging mechanical refactoring?
- Does a nominal, declaration-owned architecture make autonomous maintenance
  measurably easier than an equivalently functional but conventionally layered
  architecture?
- Can edit fan-out, reviewer corrections, reverted designs, and time-to-green be
  used as practical alignment metrics?

## Possible Evaluation Design

Compare autonomous maintenance under several conditions:

1. Repository source and tests only.
2. Source, tests, and maintained architectural documentation.
3. The above plus explicit anti-patterns and evidence hierarchy.
4. The above plus persistent goal and memory.
5. The complete system plus NRA as an adversarial second pass.

Candidate outcomes include behavioral correctness, architectural violations,
semantic edit fan-out, number and severity of human corrections, time to a
coherent accepted patch, and survival of the intended ownership model after
follow-up changes.

## Evidence to Preserve

- User corrections and the broader invariant derived from each correction.
- Before-and-after diffs for mirrored state, caller dispatch, enum behavior,
  fallbacks, and generic package ownership.
- NRA reports, including true findings and justified false positives.
- Commit and CI history for autonomous batches.
- Context-compaction handoffs and the decisions that remained stable afterward.
- Cases where behavior passed tests but architectural review still rejected the
  solution.

## Next Step

Use ``nominal-refactor-advisor --detector-capabilities`` together with
finding-specific synthesis and projected-state reports to classify the current
detector inventory. For each detector, establish whether it only observes a
required relation, recovers authority evidence, rejects an under-proved recipe,
or reaches a clean source-reproved refactor. That evidence can distinguish
useful assurance-gap detectors from machinery that does not improve correct
maintenance, then support a controlled empirical study or detailed experience
report.

## Open Dependency Evidence: Derived Registry Keys

Observed on 2026-09-05 with `metaclass-registry` 0.1.4: a concrete subclass
inherits its parent's generated key before the configured class-name extractor
is consulted. Registration then replaces the parent's entry. This is a source
identity problem, not a reason to prohibit concrete inheritance.

```python
from metaclass_registry import AutoRegisterMeta

class Family(metaclass=AutoRegisterMeta):
    __registry_key__ = "key"
    __key_extractor__ = staticmethod(lambda name, cls: name)
    __skip_if_no_key__ = True

class Parent(Family):
    pass

class Child(Parent):
    pass

# Observed: Parent.key == Child.key == "Parent"
# Observed: Family.__registry__["Parent"] is Child
```

NRA's concrete-descendant catalog test caught this during decorator-operation
development. The decorator operations now combine one shared payload trait with
their respective target contracts, following the existing function-body payload
model. That factoring does not resolve the dependency's general inheritance
behaviour. The remaining audit must distinguish declared keys from materialised
derived keys, check the dependency's supported policy surface, and verify the
advisor's static registry model against native registration. The fix belongs
with the key-selection authority; per-child copied keys would introduce another
maintenance obligation.

The native follow-up verified that type-keyed descent already rejects an
inherited-key overwrite through its closed-family check: the replacement child
is outside the explicit type bindings. This is now covered by a runtime-backed
regression, rather than inferred from the guard's source. It does not fix the
dependency's generated-key behaviour.

The same audit found an independent source-fidelity failure in method descent:
moving `event.__secret` from `NamedEventProjection` to `NamedEvent` changed its
native lookup from `_NamedEventProjection__secret` to `_NamedEvent__secret`.
The recipe had accepted the change. Descent now consumes the existing
`ClassMethodPromotionSafetyProfile`; private identifiers, class cells, `super()`
and the profile's other ownership dependencies share the promotion proof.
`docs/examples/behavior_descent_safety_refactor.py` records the applied DSL
sequence. Native CLI tests cover successful descent and refusal without source
writes, with LF and CRLF input.

### Cross-Module Method Globals: Binding Proof Integration

A native two-module probe also confirmed that
`_TypeKeyedBehaviorMethodDescent._require_target_module_bindings` accepted a
same-spelled destination global without proving its origin. With `label =
"source"` in the projection module and `label = "target"` in the target module,
descending `return event.name + label` changed `event:source` to `event:target`.
The class-ownership profile does not establish module-global equivalence.
The name-presence check now delegates to `DeclarationModuleBindingTransfer`,
which uses the lexical dependency collector, source declaration index and
module binding authority. `DeclarationDependencyUse` owns evaluation-phase
selection, including eager and deferred annotations. Equal spellings or equal
current values cannot prove shared ownership. Rebound declarations require
definition-position evidence and remain unproved by this transfer boundary.

Native CLI regressions cover same-name globals, different imports under one
alias, builtin shadowing, matching imports, and a two-stage plan that first
establishes the destination import and then descends the methods. Comprehension
locals are resolved lexically rather than mistaken for module dependencies.
The applied six-stage plan is
`docs/examples/behavior_module_binding_refactor.py`; it also replaces separate
declaration/final snapshot traversals with one batched authority traversal.

Final review exposed a missing relation in the first transfer implementation:
quoted annotation names were present in the collector's separate name sets but
excluded from its direct-source reference list. Native `typing.get_type_hints`
tests proved that both `'Result'` and `list['Result']` could change meaning
undetected. The collector now retains one complete reference collection;
`DeclarationDependencyUse` owns both evaluation phase and direct-source status.
Name sets and the editable direct-source view are derived from that collection.
`docs/examples/lexical_reference_projection_refactor.py` records the applied
eight-stage consolidation, including removal of an unused visitor forwarder.

### Qualified Annotation Rename: Shared Phase Policy

The reference-consumer audit found that qualified annotation renaming still
used the source-position snapshot for a module alias under postponed
annotations. Rebinding that alias after a function declaration caused the
rename either to change an unrelated attribute or to miss the intended one.
Both cases broke native `typing.get_type_hints` after an accepted CLI rewrite.
The lookup now consumes `DeclarationDependencyUse.binding_phase`, the same
policy used by declaration transfer, instead of introducing a renamer policy.

`docs/examples/annotation_binding_phase_refactor.py` records the applied
single-operation DSL plan. It selects the caller and the called declaration,
requires exactly one resolved call, and supplies only the replacement arguments.
Native before/after CLI tests cover function, async-function, class and module
annotations, both alias orders, and explicit postponed versus interpreter-default
evaluation on Python 3.11 and 3.14. This demonstrates a targeted authored change;
the DSL does not infer that the evaluation policy is the desired one.

### Capability Inventory: Native Contract Fulfilment

The current inventory contains 81 required-relation observers, 21 recipe
evaluators and 15 recipe synthesis declarations. Those counts describe
contracts, not successful source transformations. Inspection of their evidence
producer found two deviations from native derivation: it omitted inherited
abstract slots and skipped abstract overrides when looking for a concrete
implementation. The first left all 15 synthesis contributions without their
inherited evaluator-slot evidence. The second could report fulfilment for a
class that Python refused to instantiate.

The proof now uses the ABC's native obligation set and first-owner MRO lookup,
with explicit rejection of unrelated or merely virtual contract membership.
Native regression cases distinguish abstract-first and concrete-first multiple
inheritance, re-abstraction, unrelated methods and virtual registration. Existing
catalogue tests now check native obligations instead of reproducing the old
filtered lookup. `docs/examples/native_contract_evidence_refactor.py` records
the applied nine-stage DSL plan. Exact annotation and guard patches remain
authored source fragments; this plan does not demonstrate semantic selectors
for arbitrary expression edits.

### Exact Method Proof Versus Source-Size Heuristics

The property-hook audit found overlap with exact leaf-method promotion for
larger closed families, but also a distinct observation-only case when the
receiver's ownership is unknown. The property observer remains until those
unknown relations have a shared replacement; dropping it now would erase useful
evidence rather than consolidate it.

The audit exposed a separate obstruction to authored factoring. A complete
two-leaf family with a shared property and a declared receiver dependency had
no promotion component solely because its line-count cost estimate did not
clear a heuristic margin. The same filter prevented a practitioner from
factoring two exact one-line methods into an explicitly named role. These
filters conflated an estimate of source size with binding proof.

The component builders now retain all binding-proved exact method families;
their cost estimates remain descriptive metadata. Native CLI regressions cover
two-leaf property promotion and a two-stage plan that creates an authored role
then extracts it into a new module, including derived source imports. Both LF
and CRLF inputs are exercised. `docs/examples/method_proof_cost_separation_refactor.py`
records the applied three-stage DSL change. Missing ownership still blocks
automatic role synthesis, and the existing receiver, closure, MRO and
source-ownership checks remain authoritative.

### Regex Observation: Imported Identity and Undecided Ownership

The regex-bundle observer treated a parameter named `re` as the standard library
and missed module aliases, imported function aliases, captured aliases and
keyword pattern arguments. It also counted calls whose arguments were invalid.
Twelve focused regressions failed against the old recogniser. The new projection
uses the shared lexical dependency model, top-level declaration index and module
binding snapshot. The operation enum refers to actual standard-library function
declarations; qualified identities and signature binding derive from them.

This remains observation evidence. Repeated literals do not prove that the
sites must change together or that a new typed grammar is their correct owner.
The finding no longer prescribes that architecture. Its threshold selects
substantial repeated syntax, not correctness or rewrite eligibility. Local or
ambiguous bindings remain unproved rather than receiving a spelling fallback.

The path now reuses `named_function_nodes`, removing `SurfaceFunctionIndex`, its
private aliases and the old positional tuple-based grouping. Imported-root filtering
avoids the expensive lexical projection for unrelated modules. Native tests
cover all nine declared regex operations; source tests cover shadowing, alias
capture, signature errors and the lazy analysis path.
`docs/examples/regex_observation_cleanup.py` records the ten applied cleanup
stages, after the new detector declaration was authored.

A direct-collector benchmark across 123 NRA modules exposed repeated projection
construction on warm scans (about 0.27 seconds). A bounded source-keyed cache
reduced warm runs to about 0.01 seconds in that probe. These modules produced no
regex-bundle candidates, so this is not an end-to-end or regex-heavy workload
claim. A weak-reference regression verifies that analysis memory release drops
the parsed module; another assertion changes the threshold while retaining the
same source projection, ruling out cached configuration decisions.

### Dataclass Recipe Validation: Collapse the Forwarding Contract

A self-scan found 21 exact method orbits and six assessed promotion components,
with no automatically proved placement. One component contained all four
dataclass mapping recipe builders. Each supplied the same forwarding method for
an abstract hook, calling the exhaustive-schema validator already owned by their
base. The hook introduced a second variation point without a distinct policy.

The authored refactor redirects the shared caller to that validator and removes
the abstract hook and its four implementations. Exact leaf promotion was not the
appropriate operation: its existing-member rejection correctly prevented
overwriting the abstract slot without an explicit contract decision.
`docs/examples/dataclass_recipe_validation_refactor.py` records the six-stage
plan applied through the real CLI.

This exposed a useful syntax-level gap. Whole-call replacement required copying
the argument expressions merely to change the callable. The new
`ReplaceDeclaredCallTargetOperation` shares the existing declaration selection
and expression machinery, replacing only the callable span. Native CLI tests
retain argument evaluation order, Unicode, comments and LF/CRLF bytes. Nested
selected calls can now be redirected in one stage because their callable spans
do not overlap. MRO, ambiguous selection and comment-protection checks reuse the
existing authority path. This is an authored edit, not a claim of inferred
destination binding or behavioural equivalence.

A replay against the preceding commit applied all six stages and produced the
same recipe-builder ASTs as the working checkout. Before and after the replay,
the real CLI generated identical recipe documents and diffs for direct payload
and key/value-sequence projections. All four concrete builders retain the same
native base-validator identity. Validation passed 2,148 full-suite tests (13
skipped), 60 focused Python 3.14 tests, and 16 ASCII-locale tests. The touched-file
audit ran all 81 detectors without omissions or findings. Sphinx built with the
two existing duplicate-description warnings for dataclass promotion operations.

### Class Indexes: One Declaration Map, Derived Graph Views

The self-scan's duplicated class-index lookups led to the underlying ownership
problem: both compact and syntax-backed indexes accepted five independently
supplied maps derived from their declarations. Their builders also repeated
child, ancestor and descendant traversal. Four regressions exposed duplicated
implementation ownership and eagerly stored graph views; a fifth test established
matching graph results before changing either implementation.

`ClassDeclarationIndex[ClassDeclarationT]` now owns the common lookup surface and
derives all views from `classes_by_symbol`. Its two specialisations retain their
record types and specialised capabilities. A generic `DirectedGraph` supplies
ordered adjacency reversal and cycle-safe reachability. Per-root queries are
lazy, and deque traversal replaces list-front removal. Tests cover diamonds,
imported bases, ambiguous simple names, nested classes, unresolved bases, cycles,
duplicate edges, dangling vertices and native-MRO distinctions.

Colliding qualified identities exposed another difference: the full builder
overwrote a prior class record, while the compact builder excluded the ambiguous
handle. The full builder now uses the same existing `UniqueIdentityIndexAuthority`
gate before base resolution. The new collision regression failed before that
change. No spelling fallback or second collision policy was added.

`docs/examples/class_declaration_index_refactor.py` records the 24-stage DSL
refactor, after the new shared declarations were authored. It adds the generic
bases, deletes duplicate fields and methods, rewrites the two builder bodies and
removes their obsolete traversal helpers. The body replacements are authored
source, not an inferred proof of arbitrary builder equivalence.

The full suite exposed a consumer which also wrote the derived path lookup:
semantic-graph checkout rebasing. That operation now replaces only declaration
paths. Its regression primes the old lookup and verifies that the new index
derives target-checkout paths without changing the original cache.

A 1,500-leaf synthetic graph retained all descendants. Warm compact-index build
time changed from about 0.018 to 0.015 seconds; the syntax-backed builder remained
about 2.35 seconds. This is a targeted construction probe, not an end-to-end
speed claim. The remaining full-builder cost warrants a separate audit of its
repeated module-binding snapshots rather than further graph micro-optimisation.

Replay preparation exposed a DSL boundary defect: `InsertBeforeTargetOperation`
used the class/function header line rather than the first decorator line. Adding
the new type parameter before a decorated class was rejected by source
validation because it separated the decorator from its declaration. The replay
used the existing after-previous-declaration operation. The declaration-boundary
batch below addresses this defect with native decorator-ownership checks.

The completed replay applied all 24 stages to the previous source. Its class
index and rebasing declarations match the working ASTs, and both native index
forms returned identical name, location, child, ancestor and descendant views
before and after on the diamond fixture. Validation passed 2,159 full-suite tests
(13 skipped), 43 focused Python 3.14 tests and 12 ASCII-locale tests. The final
touched-file audit ran all 81 detectors with no omissions or findings. The docs
build retains the existing duplicate-description warnings noted above.

### Declaration Boundaries: Preserve Decorator Ownership

An adjacent insertion can compile successfully while transferring an existing
decorator to the new declaration. Parenthesised decorators also begin before
their expression's AST line. `NamedDeclarationSourceAuthority` now derives the
complete declaration span through the existing token-aware `SourceTextGeometry`.
Adjacent insertion, generated dispatch-family placement and candidate-collector
migration consume this shared span instead of reconstructing edit boundaries
from header or expression positions.

`docs/examples/decorated_declaration_boundary_refactor.py` records the 11-stage
consumer migration after authoring the span property. The plan contains authored
signatures, bodies and scoped replacements; it does not infer their equivalence.
Native tests compare decorator evaluation and ownership across class, function,
async-function and nested-method insertions, including stacked and parenthesised
decorators, LF/CRLF source, dataclass constructors and generated dispatch families.

The collector audit also exposed a missing prerequisite: a decorated forwarding
method can change the result independently of its body. Its candidate declaration
now requires a plain method before offering collector migration. Regression
fixtures demonstrate this behavioural difference and verify refusal without
source changes. This guard establishes one necessary condition, not a complete
audit of collector binding and signature preservation.

The recorded 11-stage plan replayed against `e9dafe0` with matching implementation
ASTs. The preceding 24-stage class-index refactor also replayed after inserting
its shared authority before the decorated index class; both generated index
classes retained native dataclass construction. The final full suite passed
2,199 tests (13 skipped). Python 3.14 validation passed 48
focused tests, and the ASCII-locale run passed 41. The touched-file audit ran all
81 detectors without omissions or findings. Diataxis keeps the precise span
contract in the API reference and the authored migration in its runnable example.

Windows CI for the preceding index batch found a test comparing native Windows
path spelling against normalised index paths. The cache-preservation assertion
now checks path containment with `Path.is_relative_to`; production rebasing is
unchanged. Cross-platform CI must verify this correction.

The next collector audit should establish qualified callee identity, binding
phase, keyword forwarding and signature preservation. Current terminal-name
recognition is an observation to investigate, not proof that a callable can move
from a method body into a class-level strategy declaration.

### Collector Forwarding: Signature and Capture Evidence

Seven native probes demonstrated incorrect migrations: lost keywords and
unpacking, truncated qualified callees, early capture of a later binding,
capture before a function exists, removed defaults and deletion of a live
parameter. A subsequent probe demonstrated that renaming `settings` to `config`
breaks keyword callers. Class-namespace shadowing is also covered. These cases
now retain native behaviour or refuse without writing.

`PositionalForwardingCall` owns the shared function projection, retaining the
complete callable expression. Candidate records derive their callable display
and configuration usage from this record. `ClassBodyReferenceCapture` extends
the existing module-binding transfer proof to compare different execution phases.
Reference identity uses source spans, allowing independently parsed snapshots
of the same source to participate in the proof.

Collector scopes no longer repeat parameter names. Their direct-forwarding
relations come from the actual registered leaf implementations, projected by
the same function model used on candidate source. This distinguishes flattening
from forwarding without a separate scope-dispatch table. Native implementation
projections are cached per scope. Qualified/computed capture remains unproved;
the full expression is retained rather than shortened to a terminal name.

`docs/examples/collector_forwarding_refactor.py` consolidates the authored
consumer migration and import management into 17 stages, after authoring the
shared projection and capture-proof declarations. Native positive CLI tests
exercise configured and unconfigured imported collector aliases. The prior
positive synthesis fixture now declares its callables instead of treating
undefined names as executable proof.

The 17-stage CLI replay against `e3181ea` produced matching consumer declaration
ASTs. The full suite passed 2,210 tests (13 skipped); Python 3.14 passed 58 focused
checks, and the ASCII-locale run passed 51. The touched-file audit ran all 81
detectors without omissions or findings.
The documentation build retains the two previously recorded duplicate-description
warnings. Windows CI for `e3181ea` also passed, verifying its path-assertion fix.

Further collector work must establish the complete base-replacement relation
and generated descriptor binding, beyond the forwarding and capture conditions
checked here. This batch does not establish arbitrary class-replacement or
annotation-introspection equivalence.

### Generated Descriptor Binding

Native counterexamples confirmed that module-level or class-local shadowing of
`staticmethod` changes the generated collector behaviour. The candidate now
owns its descriptor type through the existing `ConstantProperty`; both source
rendering and validation derive from that value. A direct attribute containing
the native `staticmethod` type exposed its `__isabstractmethod__` descriptor to
ABC machinery, so a constant descriptor is necessary rather than a second type
table or a special metaclass exception.

`ModuleNominalBindingView.require_native_type_in_class` reuses its builtin and
named-reference witnesses. Repository export evidence must also close wildcard
imports. A positive legacy fixture now imports its base explicitly, while native
CLI tests cover explicit builtin imports and wildcard imports with declared
exports. Unknown wildcard exposure remains unproved.

`docs/examples/collector_descriptor_refactor.py` records the six-stage migration
after authoring the shared binding-view method. The API reference records the
binding boundary separately from this runnable plan.

The six-stage CLI replay against `226e13f` produced matching consumer declaration
ASTs. The full suite passed 2,216 tests (13 skipped), Python 3.14 passed 64 focused
checks and the ASCII-locale run passed 57. The touched-file audit ran all 81
detectors without omissions or findings. The docs build retains the two existing
duplicate-description warnings.

A native base-selection probe confirmed the remaining relation gap. A source
class named `CrossModuleCollectorCandidateDetector` retained the collection
result but changed an inherited `marker` from `original` to `replacement`.
Current selection accepted it. The next batch must prove the base-replacement
relation rather than infer full class behaviour from a name and forwarding
shape. The probe used an in-memory source snapshot and native execution before
and after simulation; it did not modify repository source.

### Native Collector Declaration Authority

The migration now resolves original and replacement bases by native qualified
identity and compares their declaration ASTs with inspected source. Original
base binding is checked at class creation. Candidate records retain the native
replacement type; emitted names derive from it. `NativeDeclaration` owns native
identity and inspected source for the base proof, forwarding projection and
generated-descriptor binding consumers.

Native regression cases cover unrelated same-name bases that change inherited
attributes, methods or constructors, and altered source under the canonical
qualified name. An additional native MI counterexample demonstrated a
one-argument collector being called with two arguments after replacement.
Competing registered collector bases are rejected by their resolved identity,
including imported aliases. This preserves the previous competing-base check
without its short-name matching. Positive fixtures now include the actual
registered declarations, including batch coalescing and finding-backed synthesis.

`docs/examples/collector_base_authority_refactor.py` and
`docs/examples/native_declaration_consumers_refactor.py` each record six stages.
A real CLI replay against `15fedaa`, after two prerequisite declaration/import
stages and adding the authored native-source module, applied all 14 stages and
produced matching consumer declaration ASTs. The ASCII-locale run passed 70
tests; a Python 3.14 collector and CLI run passed 100 tests.

This establishes native declaration identity and the checked forwarding and
capture conditions, not arbitrary base-replacement equivalence. Full sibling
MRO effects, class-creation hooks and annotation-introspection equivalence remain
open. The API reference records these boundaries separately from the plans.

### Deferred Annotation Metadata During Dependency Inspection

CI runs `33995924217` and `33995325471` failed on Python 3.14 with a live class
dictionary changing during implementation-dependency traversal. The downstream
CLI JSON failures resulted from that exception, rather than malformed reporting.
A native Python 3.14 reproduction showed method-annotation evaluation
materialising metadata on its owning class while that class was being scanned.

The traversal now snapshots each owner's declared values before recursively
visiting dependencies. It still follows the native MRO and includes dependencies
revealed by deferred annotations. Two native regressions failed before the change;
the post-change run covering them, the affected cache paths and CLI checks passed
59 tests. `docs/examples/implementation_namespace_snapshot_refactor.py` records
the authored edit; its real CLI replay produced an identical module AST.

The next performance audit should reuse native source projections across batch
operations without accepting stale source. A local probe under concurrent test
load took 1.87 seconds for 20 fresh projections of one registered collector class,
while reusing one projection avoided repeated inspection. This is a diagnostic
comparison, not a representative throughput benchmark.

Final combined validation passed 2,229 tests on Python 3.11 (15 skipped) and all
2,244 tests on Python 3.14 with coverage. The touched-file audit ran all 81
detectors with no omissions or findings. The documentation build retains only
the two previously recorded duplicate-description warnings. Cross-platform CI
must confirm the metadata-traversal correction on Windows and macOS.

### Native Projection Reuse Without Cached Proof Results

`NativeDeclaration` now keys inspected source by the loaded declaration's object
identity. Its equality and hash do not delegate to the declaration's metaclass:
distinct classes may compare equal, and a class may be unhashable. Repeated
wrappers reuse inspection without a parallel type-name registry. The cache keeps
loaded declarations and their projections alive for the process lifetime.

Source comparison remains per request. A real source-edit/reload test confirms
that changing source does not change the old loaded declaration, the changed AST
is rejected against its captured projection, and reloading creates a distinct
projection even under the same qualified name. This does not prove arbitrary
live monkeypatch equivalence.

The 20-collector batch regression inspected native bases 40 times against
`588e622`, and at most once per base after the change. It also executes the
original and rewritten batch in native subprocesses and compares their outputs.
The archived baseline test overrides pytest's repository `pythonpath` explicitly;
without that override it imports the working checkout and is not baseline evidence.

`docs/examples/native_projection_identity_refactor.py` expresses the complete
production edit in six stages, including import and decorator changes. Its real
CLI replay against `588e622` produced an identical module AST. No source-match
result is cached; current proposals are checked against the reused projection.

The full suites passed 2,237 tests on Python 3.11 (15 skipped) and 2,252 on
Python 3.14. The focused ASCII-locale run passed 78 tests, and the touched-file
audit ran all 81 detectors without omissions or findings. Documentation retains
the two existing duplicate-description warnings.

Windows CI for `588e622` exposed a fixture provenance error: `inspect.getsource`
normalised the native module's CRLF text before constructing a source snapshot.
The exact revision guard correctly rejected that text. The fixture now uses
the existing `read_source_text` authority. In an archived CRLF checkout, all five
affected cases failed before the correction and passed afterwards; the broader
78-test native/collector suite then passed on both Python 3.11 and Python 3.14.
The production revision guard is unchanged. Both macOS jobs and both Linux jobs
for `588e622` passed, confirming the preceding lazy-metadata traversal correction
on those platforms; a new Windows run must verify the fixture correction.

A subsequent native probe confirms the next inheritance gap: `Other` inherits
`ConfiguredCrossModuleCollectorCandidateDetector[int]`, and `Owner` inherits
`Other` plus `CrossModuleCandidateDetector[int]` while declaring a one-argument
collector forwarder. Migration currently accepts it, then inherited configured
dispatch calls the collector with two arguments. The original executes and the
rewrite raises `TypeError`. The next relation proof must account for inherited
member ownership through indirect bases, rather than extend direct-name checks.
This probe used an in-memory source snapshot and separate native subprocesses;
it did not write source files or change the repository.

### Collector Execution Through One Native Attribute Relation

The inherited-dispatch investigation exposed a separate runtime authority split:
`required_candidate_collector` selected retained declaration options before
considering the receiver's field. An explicit subclass collector override was
ignored for all six registered collector base families. The source-collector
fast path used the same competing lookup. Seven native override regressions
failed against the preceding implementation; the missing-collector validation
case already passed.

Generated classes now expose their existing optional class-shell fields through
`ClassAliasProperty`, pointing into the retained declaration's options. Values
remain owned by the declaration. Execution reads the collector attribute through
ordinary Python lookup; both collector-selection helpers were removed. Subclass
overrides therefore have the same meaning for generated and authored parents.
The class-creation check validates the projected field without inspecting a
second representation. This also removes the helper-override collision from
the pending collector migration counterexamples.

`docs/examples/collector_attribute_projection_refactor.py` expresses the whole
production change in 11 DSL stages. Replaying it against `1931322` produced an
identical module AST. In that isolated replay, the full suites passed 2,245 tests
on Python 3.11 (15 skipped) and 2,260 on Python 3.14. Three old cache-test assertions
that prohibited a collector attribute were removed; their native/compact result
comparisons remain. Repository architecture tests now verify projected descriptor
identity and value provenance centrally. The focused ASCII run passed 82 tests,
all 81 architecture detectors ran without omissions or findings, and the docs
build retained only the two existing duplicate-description warnings.

A diagnostic empty-collector microbenchmark under concurrent test load measured
median times of 1.50 seconds before and 0.61 seconds after for 300,000 calls,
across five repetitions. This measures the lookup path, not analysis throughput.

The source/native C3 carrier work remains uncommitted and unintegrated into
collector migration. Its 48 focused tests pass on Python 3.14. Two native
migration regressions remain: indirect configured ancestry and an earlier MRO
branch overriding `_candidate_items`. The migration must prove the selected
method after replacement and removal; the runtime cleanup does not claim to
establish that proof. The user-owned `uv.lock` remains untouched.

### Collector Migration Through Native C3 and Closed Namespace Evidence

Collector migration now proves the inherited method selected after replacing the
forwarding base and removing the forwarding method. `QualifiedDeclaration` owns
the representation-independent name contract, and `ClassNamespaceDeclaration`
provides the member-binding contract. Source and authenticated native declarations
share `DeclarationMroType`; Python derives C3 order. The source traversal schedules
construction without inventing precedence or rerunning repository class hooks.
Indirect configured ancestors and earlier overriding branches are accounted for;
independent branches and later overridden branches remain admissible.

Class namespace evidence comes from the existing ordered lexical scope traversal.
An effect selector observes each visited node without traversing its children or
maintaining another binding state. The scope traversal owns evaluation order,
branch joins, deletion, annotation phases and deferred bodies. Native references
capture their use-point resolution rather than consult the final class namespace.
Source creation effects require explicit evidence, including operators, hashing,
iteration, decorators and constructors. Unknown executable forms remain unproved.
Native descriptor declarations also supply the existing promotion-name projection.

Native-process regressions cover inherited dispatch, class-body `exec`, a decorator
shadowed then deleted, condition and iteration hooks, operator/hash effects in
defaults, deleted methods, annotation-only names, native generic annotations and
deferred function/generator execution. Thirteen of the 26 cases fail against
`bfb8c53`; all pass with this change. Python 3.14 tests also exercise the difference
between eager and deferred custom annotation subscriptions. Two decorated-anchor
tests now retain a native-decorator success case and explicitly reject the custom
decorator whose creation effects were previously assumed safe. General adjacent
insertion still tests custom decorators and their native evaluation order.

The 26-stage `mro_declaration_carrier_refactor.py` and three-stage
`collector_mro_proof_refactor.py` replay against `bfb8c53` with the three new proof
modules supplied as authored prerequisites. All four edited module ASTs match the
working implementation. The four-stage
`class_namespace_effect_projection_refactor.py` was also replayed against the
intermediate implementation and reproduces its complete module AST. It replaces
repeated forwarding visitors with one traversal hook and an effect selector.
An initial replay selected the named-scope assignment operation for a module
assignment; using the existing `ReplaceModuleAssignmentOperation` resolved that
authoring error without changing the DSL runtime.

Validation: 2,284 passed and 15 skipped on Python 3.11; 2,299 passed on Python 3.14.
The focused ASCII-locale suite passed 105 tests. All 81 architecture detectors ran
without omissions or findings. Sphinx retains only its two existing duplicate
description warnings. These results establish the supported relation proofs,
not arbitrary Python equivalence or equivalence under live monkeypatching and
class introspection. The user-owned `uv.lock` is unchanged by this batch.

The next leverage target remains expressing recurring semantic refactors with
fewer authored replacement bodies, using these shared proof boundaries rather
than repeating checks in each operation. The current DSL plans demonstrate
execution and replay; their length still exposes that semantic-vocabulary gap.

### Shared Member Lookup Proof and Native Argument Effects

General member promotion now shares the source/native C3 proof used by collector
migration. It checks lookup across the destination's complete indexed descendant
cohort, including diamonds. A moved member must retain the same selected owner,
with the authored source-to-destination transfer as the only owner substitution.
An earlier competing branch now prevents a move; a later branch remains valid.
Annotation-only declarations do not manufacture installed class bindings.

`ClassMemberLookupProof` owns lookup over C3 and projected namespace changes.
`ClassNamespaceDelta` is an edit over an existing declaration, not another
declaration identity. Native owner equality retains Python object identity even
when two declarations have the same qualified name. Raw native terminal bases
participate in lookup. Collector migration uses this shared proof instead of
maintaining its own MRO-member loop.

Closing source namespaces exposed a distinction between native generic aliases,
which store arguments, and `typing.ClassVar`, which can hash or inspect them.
Explicit `property`, `staticmethod` and `classmethod` calls can also execute
argument metadata hooks. Native-process probes reproduce these hooks mutating
the class namespace. Use-point `ScopedNativeReference` and `NativeArgumentEvidence`
carry the evidence into declaration-owned subscription families and the argument
inspection visitor. The existing lexical traversal still owns evaluation order
and eager versus deferred annotation behaviour. Unknown computed references
produce an explicit unproved result rather than an incidental lookup exception.

The two-stage `native_argument_evidence_refactor.py` was used on the working
source for the argument-evidence rename and move. A separate real CLI replay
against the preserved intermediate files applied nine physical rewrites; the
moved declaration AST matches that snapshot after the rename. Fresh imports
confirm that the retained source binding refers to the same declaration object.
Later authored argument-proof behaviour is outside that replay's comparison.

The nine-stage `referenced_namespace_effect_refactor.py` records the subsequent
shared-base factor. Its automated replay reconstructs the unfactored fields and
getters from current source, simulates through the CLI, applies through the DSL,
and compares the complete module AST and fresh-process effect observations.
This baseline is reconstructed, not a preserved pre-edit checkout. The plan uses
import, insertion, base, decorator, assignment and deletion operations without
whole-method replacements. It was recorded after the structural edit: future
supported edits should use the DSL before mutation so projected continuation
analysis can inform the next semantic decision.

The API reference now describes cohort lookup and argument-effect boundaries.
Two stale observation-family names were removed from autodoc after a clean
documentation build exposed their absence from current source.

Validation: 2,304 passed and 15 skipped on Python 3.11; 2,319 passed on Python
3.14. The focused ASCII-locale run passed 43 tests. All 81 architecture detectors
completed with no omissions or findings on the touched production paths using
the whole package as context. Ruff and the whitespace check passed. A clean
Sphinx rebuild retained the two existing duplicate-description warnings.
The user-owned `uv.lock` is excluded from the batch.

### Scenario-Scoped Source C3 Reuse Through the DSL

The member-promotion cohort traversed shared ancestry independently for every
class. A 51-class probe performed 891 namespace closure checks. The source MRO
authority now owns a lazy cache for one fixed source context, native-root set
and optional substitution. Shared ancestors are closed and constructed once.
Changing the context or substitution creates a new authority with an empty
derived cache, using the existing `dataclasses.replace` semantics for non-init
fields. Python still computes C3 precedence.

`source_mro_scenario_reuse.py` expresses this as seven DSL stages. The plan was
simulated against the untouched checkout and its projected source was rescanned
before application. The exact scan reported zero findings before and after,
and produced an empty continuation plan. It therefore supplied no further
executable semantic suggestion for this edit. Preflight did reject an ambiguous
text replacement; the authored plan was narrowed to exact accesses before the
successful simulation. The inspected plan then applied through the CLI as seven
physical rewrites in one production module.

The saved plan was then strengthened with the existing
`ProjectFunctionParameterOperation` and `ReplaceDeclaredCallOperation`. These
replace exact-text substitutions with lexical-binding projection and a
declaration-resolved caller edit. Replaying the final seven-stage plan against
`fd52e41` produces the complete working module AST, with no text-patch operations
or whole-method replacements. The source-context and cache-lifetime decisions
remain authored semantics; the DSL derives the owned reference edits.

The same probe now performs 51 namespace checks and repeated lookups return the
same inert class objects. Focused tests also exercise independent substitution
and snapshot projections, retrying unproved construction, native base identity,
collector migration and member promotion across branches. All 40 pass. Diagnostic
probe times were approximately 46 ms before and 7 ms after; these timings cover
the small hierarchy projection, not whole-repository analysis throughput.

Initial full-suite runs hit the `/tmp` user quota while creating fixtures and
cache entries. An inactive 667 MB relocated NRA cache was removed after checking
for open handles and recent writes, freeing about 45,000 inodes. Replay sources,
diagnostic logs and user files were retained. Those quota-affected runs are not
accepted as validation gates.

Four further inactive analysis caches (497 MB) were removed as inode use grew
during the retry. The completed retry passed 2,307 tests with 15 skipped on
Python 3.11 and 2,322 tests on Python 3.14. The 40-test focused ASCII run passed.
All 81 architecture detectors completed with no findings or omissions, and
Ruff and the whitespace check passed. Sphinx retained its two existing duplicate
description warnings. `uv.lock` remains user-owned and excluded.

### Operation Catalogue and Portable Reference Generation

The existing documentation generator now renders the complete registered
`RefactorRecipeOperation` family. Each entry includes its operation key,
canonical declaration path, source dependency scope, native constructor and
declaration documentation. Sphinx derives the local contents and constructor
parameters. There is no separately maintained operation roster or input schema.
The generated reference currently contains 71 operations, including lexical
parameter/local projection and declaration-resolved call edits that were easy
to overlook in the broad facade module. Diataxis keeps this surface as reference
material linked to the existing execution guide.

The touched generator also uses `NativeDeclaration` for detector implementation
paths and the shared exact-text reader for unchanged-file comparison. Its writer
uses explicit UTF-8 and preserves supplied newlines. Tests reproduce the prior
ASCII encoding failure and the unnecessary rewrite of unchanged CRLF text, and
exercise automatic operation registration/removal without catalogue edits.

`codemod_catalog_generation.py` records the seven-stage transformation. The DSL
simulated the initial change before application; final refinements also used DSL
operations. A replay from the committed baseline reproduces the entire working
generator AST. The projected finding report explicitly used an evidence-local
partial scan and offered no executable continuation. It is not an all-detector
completion claim.

The multi-root probe also exposed a remaining CLI path-boundary issue: relative
live module paths reach the checkout-relative cache codec with multiple roots,
where their origin is ambiguous. Explicit absolute roots and target paths allow
the same scan to proceed. A future correction belongs at the live-path owner;
the relocatable cache codec should retain its fail-closed origin contract.

### Windows Replay Fixture Correction

CI for `fd52e41` passed Linux, macOS and the documentation/wheel job. Both Windows
jobs failed because the new namespace replay fixture removed its shared base
with AST spans but reconstructed consumers using LF-only text patterns against
CRLF source. An explicit CRLF case reproduced the same `NameError` locally.
Commit `81a11d9` reconstructs in one spelling, writes the requested physical
newline and compares unchanged source through the exact-text reader. LF and
CRLF now both pass on Python 3.11 and 3.14, including the ASCII-locale probe.

The combined batch passed 2,313 tests with 15 skipped on Python 3.11 and 2,328
tests on Python 3.14. The rendered catalogue contains all 71 registered entries
with visible declaration paths and typed constructor inputs. A fresh Sphinx
build retained its two existing duplicate-description warnings. The first
uncached audit exceeded its 60-second budget; the completed retry used the
compact report with structural-overlap rendering disabled and ran all 81
detectors without omissions or findings. Ruff and the whitespace check passed.

### Live Root Binding Before Cache Projection

The relative multi-root CLI failure is reproduced by a real subprocess scan.
Live path requests were entering the relocatable cache codec without their
working-directory origin. `AnalysisPathScope.from_requested_roots` now binds
both requested and explicit context roots through the existing lexical absolute
path operation before deriving analysis and reporting scopes. The codec still
rejects ambiguous root-relative paths; no cache-side fallback was added.

The neighbouring file-context resolver previously dereferenced symlinks while
the cache contract preserves the admitted path spelling. It now derives the
parent through the same lexical path operation. Regression cases cover file
and directory requests through a symlink, roots captured before a working
directory change, explicit two-root scans, and a verified cold miss followed
by a warm hit.

`docs/examples/live_analysis_root_binding.py` expresses the three-stage change.
The initial two-stage simulation preceded all production edits. After the
symlink regression exposed the adjoining inconsistency, the third operation
was appended and the complete sequence was simulated before application. Both
projected scans were explicitly evidence-local partial and offered no automatic
continuation. All three production rewrites were applied through the CLI DSL.

The focused path/cache suite passed all 146 cases. The original relative
two-root repository scan now completes: all 81 detectors, no omissions or
findings, with caches disabled. Its measured wall time was approximately
10 seconds. This is a completed analysis probe, not a throughput benchmark.

The forecast also exposed a separate agent-interface cost: requesting a
continuation serialises the complete source index inside the continuation
report, even without `--codemod-project-source-index`. This small three-stage
forecast emitted about 5 MB despite having no continuation candidates. The
typed continuation must retain its index for proof and planning; its default
external projection need not duplicate the opt-in source-index surface.
This is the next concrete reporting boundary to audit.

The completed full runs passed 2,317 tests with 15 skipped on Python 3.11 and
2,332 tests on Python 3.14. The four path regressions also pass with Python UTF-8
mode and locale coercion disabled. Ruff and the whitespace check passed; a
fresh Sphinx build retained its two existing duplicate-description warnings.
The unrelated `uv.lock` changes remain excluded.

### Continuation Projection Without Implicit Index Export

`CodemodPlanSequenceContinuationReport.source_index` retains its typed reference
to the projected source index, but now declares `included=False` through the
existing JSON field policy. The explicit CLI source-index flag remains the
single opt-in output surface. No extra serialiser, projection wrapper or
caller-side removal pass was introduced.

Three failing regressions established the previous behaviour: direct report
projection and CLI continuation output with the index flag both off and on.
The declaration change was simulated and applied through
`docs/examples/continuation_report_projection.py`. Its projected scan was
evidence-local partial, with no findings or automatic continuation. A repeated
repository forecast emitted 90,279 bytes after the change, compared with
4,944,228 bytes before it. The later forecast is an idempotent application of
the same assignment operation, so its edit evidence also differs; the omitted
index accounts for the large reduction. This measures payload size, not a
runtime speedup.

The direct continuation test still verifies index identity. It now also
simulates the extended two-stage sequence, applies the original file creation
and discovered registry conversion together, and executes both resulting
handlers in a fresh Python process. CLI tests verify that the opt-in index
contains the newly projected file, remains absent from the nested continuation
report, and leaves the emitted continuation plan loadable and executable.
Five focused cases and seven JSON/continuation cases under an ASCII locale pass.

Both the touched runtime-module audit and a fresh whole-package audit ran all
81 detectors without omissions or findings. The whole-package report exposes
raw findings and has caches disabled. It supplies no next-edit candidates;
that observation is not evidence that the broader architectural goal is
complete. Manual boundary review and detector contribution coverage remain
necessary. The public API reference now distinguishes typed provenance from
opt-in JSON output, separately from the executable example.

Full validation passed 2,318 tests with 15 skipped on Python 3.11 and 2,333
tests on Python 3.14. Ruff and the whitespace check passed. Sphinx retained its
two existing duplicate-description warnings.

The refreshed capability inventory reports 81 required-relation observers,
21 recipe evaluators and 15 recipe synthesis providers, with overlapping
authority-boundary and semantic-mirror roles. These are native contract
memberships, not empirical successful-refactor counts. A clean whole-package
scan therefore still needs comparison with concrete manual factoring decisions
to distinguish absent debt from discovery or synthesis gaps.

### Destination Binding Proof Before Extending Member Transfer

Manual review found a one-use `FindingDetectorCountsAuthority` aggregation
wrapper. Its only repository caller is `FindingSummary.from_findings`; no
separate lifecycle or independent state was identified. Moving its behaviour
onto a count-record declaration would currently require re-authoring the
method source or adding source-derived member transfer between unrelated
owners. The existing ancestor-promotion operation is not permission to invent
a temporary inheritance relation merely to route that edit through the DSL.
The count refactor and unrelated-owner transfer remain unfinished.

Review of the reusable movement machinery exposed a load-bearing proof gap
first. Promotion checked source-class capture and destination member-name
collisions, but not capture of the moved header by destination bindings. Two
native subprocess probes simulated as clean: moving a method into a class
with `int = 3` changed its annotation from the builtin type to `3`; moving a
`@staticmethod` into a class with `staticmethod = 3` made module execution fail
with `TypeError`. Further failing regressions cover ordinary and quoted field
annotations. Earlier probes using class-level type aliases were already
rejected by creation-effect checks; literal shadowing isolated the missing
destination relation without weakening those checks.

`ResolvedClassTarget.bound_names` now derives and caches lexical bindings from
its retained class AST. The move context derives the union of source and
destination bindings and no longer accepts a separately supplied source-name
set. Existing method-header checks consume this union. Field checks also reuse
`StringizedAnnotationSurface` for deferred type names, leaving `Literal` value
strings and `Annotated` metadata separate from type references. Destination
collision checks reuse the same resolved-class projection.

`docs/examples/member_move_scope_binding.py` expresses all ten production
rewrites. The initial eight-stage plan was simulated before mutation, then
extended for the quoted-field counterexample and simulated again before
application. Both projected finding reports were evidence-local partial and
offered no automatic continuation. The final production edits were applied
through the DSL; only formatting followed outside it.

The 30 focused cases exercise capture rejection, unchanged source on failure,
safe method-body globals, unrelated quoted type names, quoted `Literal` and
`Annotated` values, C3 lookup and native execution after promotion. Eager
`Literal`/`Annotated` subscriptions remain unproved in the Python 3.11
creation-effect gate. The same syntax is deferred on Python 3.14 and can move
without that eager obligation. The tests distinguish these native evaluation
modes and execute the accepted result rather than imposing the older runtime's
rejection on both interpreters. Final focused checks also pass under the ASCII
locale. The touched-source audit ran all 81 detectors without omissions or
findings. The reference describes the strengthened header boundary separately
from this implementation and verification record.

Full validation passed 2,328 tests with 15 skipped on Python 3.11 and 2,343
tests on Python 3.14. The Python 3.11 full run preceded the final test-only
annotation-mode clarification; the subsequent 30-case ASCII run validates that
final fixture on 3.11, and the 3.14 full run includes it. Replaying the saved
ten-stage plan from the committed baseline reproduces both complete working
module ASTs. Ruff and the whitespace check passed. The rendered reference
contains the new header-boundary description; Sphinx retained its two existing
duplicate-description warnings. The preceding commit `9bacb9a` completed CI on
all platforms. `uv.lock` remains unrelated and excluded.

## Shared reaching-write evidence for bound call results (2026-09-06)

Tracing the remaining constructed-instance call-selection gap exposed a second
binding algorithm in `CompactFunctionFlow.bound_call_result_for`. Unlike the
existing lexical binding resolver, it ignored conditional writes between a
factory call and its consumer. It also tracked only an attribute's exact path,
so replacing `owner` or `owner.child` could leave `owner.child.result` falsely
associated with an earlier call. That association feeds the declared return
type used by carrier-expansion proofs.

Both queries now use the same flow-owned mutation selection. Bound-result
lookup selects the reaching write first and then joins it to its originating
call. Access-prefix semantics belong to `LexicalValueReference.is_prefix_of`.
Parent replacements invalidate the result; sibling writes and writes inside
the result do not replace its identity. Conditional bindings retain the
existing unresolved outcome. No new registry, state cache or variant class was
introduced.

`docs/examples/bound_result_binding.py` applies the change as four dependent DSL
stages. Replaying it from `7bbf17f` reproduces both complete edited module ASTs.
This is an authored semantic change, not an equivalence claim: the old lookup
returned an unjustified factory origin for the regression cases. The DSL still
requires authored method bodies for this extraction; selecting and moving an
existing method region is a separate capability gap.

The focused 144-test run includes conditional assignments, iteration/context/
pattern bindings, deletion, parent-path replacement, retained sibling paths,
and downstream refusal to derive a carrier-expansion proof from a conditionally
rebound value. A 1,000-assignment native-flow probe, comparing the committed
method with the new method on the same facts, measured median lookup times of
197.48 ms and 0.60 ms across five repetitions. This measures removal of the
per-call mutation rescan, not whole-repository scan throughput. All 81 detectors
completed the touched-source audit with zero findings. Sphinx built the API
reference with its two existing duplicate-description warnings.

The preceding Windows 3.11 CI failures passed on an unchanged rerun of
`34012234564`. The original run crossed the 20-second CLI deadline in two tiny
synthesis tests; local reproductions took about 2.4 seconds each. The deadline
has not been increased, and the successful rerun does not establish a root
cause for the intermittent timing failure.

Next: establish constructed-receiver identity through existing class and flow
evidence before broadening declaration-selected call edits. A class-name match
alone does not rule out constructor replacement, custom lookup or intervening
mutation. Do not substitute the stricter dataclass field-schema proof for this
different required relation.

Full validation passed 2,354 tests with 15 skipped on Python 3.11 and 2,369
tests on Python 3.14, using eight pytest workers. The latter run retained 96
warnings about forking a multi-threaded process; the analysis-pool lifecycle
needs a separate audit rather than hiding the warnings. Logs are
`/tmp/nra-binding-full.log` and `/tmp/nra-binding-full-314.log`; the focused,
architecture, benchmark and documentation receipts share the
`/tmp/nra-binding-*` or `/tmp/nra-bound-result-*` prefix. Ruff and the whitespace
check passed. The unrelated `uv.lock` change remains excluded.

## Call-target-owned dispatch (2026-09-06)

Before adding constructed-receiver lookup, removed the repository's concrete
`CurrentClassMemberMethodReference` branch and lexical-presence dispatch.
`CompactCallTargetReference.resolve` now selects its lookup through native MRO.
Bare and qualified targets share the lexical refinement; current-class member
targets supply the distinct member lookup. The repository consumes that
declaration instead of rediscovering its syntax family.

The resolver ABC declares the three repository obligations. Its context and
result parameters keep the syntax module independent of the concrete
repository records without a runtime import cycle, local imports or an
untyped context. The existing resolution implementations satisfy the ABC;
there is no delegate wrapper, secondary variant registry or numeric priority.

An actual repository regression composes lexical lookup ahead of current-class
member syntax using MI. The committed dispatch failed to find its free-function
target because its concrete-class check overrode the declared MRO. The new
dispatch resolves the function. A direct comparison against the committed
method confirms the difference; the 145-case focused suite retains the
existing descriptor, binding and unresolved-result behaviour.

The eleven-stage `docs/examples/call_target_dispatch.py` sequence applies the
production refactor. Replaying from `ba36274` reproduces both complete module
ASTs. Shared method-body extraction remains authored in this plan; the DSL
handles insertion, imports, base substitution and dependent-stage replay.
All 81 detectors completed the touched-source audit with zero findings.

This is the dispatch prerequisite, not constructed-instance resolution itself.
That lookup still needs the class identity and mutation evidence described in
the preceding entry.

The full suites passed 2,355 tests with 15 skipped on Python 3.11 and 2,370
tests on Python 3.14, with eight workers. Python 3.14 retained the same 96
fork/thread warnings tracked above. Sphinx built the rendered API contract with
the two existing duplicate-description warnings; Ruff and the whitespace check
passed. Validation receipts use `/tmp/nra-target-dispatch-*`. CI for the
preceding binding-resolution commit was still running at this handoff.

## Retained reference-use evidence (2026-09-06)

The constructed-receiver investigation found that flow collection discarded
reads unless a separate inventory recognised their names as possibly callable.
A locally constructed result stored in a tuple, returned, or passed to an
unknown function could disappear from the reference-use facts. The repository
could not use those missing facts to establish receiver lifetime or escape.

Collection now retains lexical reads independently of callable identity. The
existing repository resolver determines which reads resolve to callable
escapes. Four function/method/import name inventories, their duplicate import
collector and the untyped constructor-argument dictionary are removed. The
production change removes 70 net lines without adding a parallel fact family.

Attribute reads retain their lexical subexpressions in evaluation order.
An extra probe after the first full-suite run caught that retaining only the
outer path would lose the function escape in `function.__call__`. That first
implementation was corrected through the DSL before committing; regressions
now cover both direct and nested function-attribute reads. Actual call targets
remain owned by call facts, and stores/deletions remain mutations.

`docs/examples/reference_use_collection.py` replays eleven dependent stages
from `6e8b9a5` and reproduces the complete edited module AST. It uses the existing
declaration-selected call-argument operation after changing the callee
signature. Constructor setup and visitor-body edits are still authored:
instance-attribute assignment deletion and constructed-instance call selection
remain concrete DSL gaps, rather than being presented as solved by this plan.

The focused suite passes 155 cases. The four-module source-collection probe
retains 13,164 candidate reads instead of 2,600, with pickled projections of
2.83 MB rather than 1.90 MB. Under concurrent full-suite load, seven-repetition
median collection times were 724 ms and 569 ms respectively. This adds source
evidence and costs collection time and space; it is not a performance win.
The probe is not a whole-repository throughput measurement. Future optimisation
must preserve the retained evidence rather than reintroduce name filtering.

All 81 detectors complete the touched-source audit with zero findings. The
preceding `ba36274` and `6e8b9a5` commits both completed CI successfully.
Validation receipts use `/tmp/nra-reference-uses-*` and
`/tmp/nra-reference-prefix-*`. The unrelated `uv.lock` remains excluded.

Final full validation passes 2,365 tests with 15 skipped on Python 3.11 and
2,380 tests on Python 3.14, using eight workers per suite. Python 3.14 retains
the same 96 fork/thread lifecycle warnings. Sphinx builds the updated reference
with its two existing duplicate-description warnings. Ruff and the whitespace
check pass. Constructed-instance method resolution remains the next task;
this batch supplies source-use evidence, not that resolution proof.

## Shared constructor and function binding lookup (2026-09-06)

Four runtime-backed regressions exposed an unsound constructor proof. A
closure-local parameter, assignment, function or class named `Payload` could
replace the module dataclass at runtime, while constructor lookup still
reported that module dataclass as the product authority. The separate
`_has_dynamic_local_binding` algorithm checked the immediate function and
module but skipped the enclosing lexical scopes.

Constructor lookup now consumes the ordinary target resolution. Function and
class definition mutations select their own leaf resolution behaviour through
`CompactMutationKind`; the import/definition identity flags are derived from
those declarations rather than repeated Boolean columns. The common result
family is `CompactCallTargetResolution`. Its resolved-class refinement supplies
the product-construction projection, while its function refinement retains the
existing descriptor and argument-binding behaviour. Shared no-op projections
live on the base. The constructor-only binding algorithm is deleted. There is
no new receiver-state cache or separately maintained constructor namespace.

The four original failures now pass. Function-local imports also resolve to
their declared class. Classes declared inside functions are currently absent
from the class index: their qualified local identity stays unresolved rather
than being replaced with a same-named module class. A class that overwrites a
method is now represented as a class target, not as an unresolved function.

The sixteen-stage `docs/examples/constructor_binding_lookup.py` plan replays
from `013af1d` and reproduces both complete edited module ASTs. The actual
`FunctionBindingProjectionSourceAuthority` constructor in NRA now resolves to
its class declaration. This establishes the constructor's selected source
definition, not the class or lookup behaviour of the returned instance. Custom
construction, receiver escape and later method replacement remain the next
proof obligations before enabling instance-selected DSL calls.

The first authored replacement accidentally included the existing dataclass
decorator. `ReplaceTargetOperation` preserves that surrounding decorator, so
the payload duplicated it and the resulting module failed to import. The plan
was corrected to omit it and the local import seam repaired before testing;
the final replay independently verifies the complete source. The DSL should
reject that ambiguous decorated payload during preflight rather than report a
clean source simulation. This is a concrete follow-up validation gap.

On 1,000 unchanged constructor calls over the same collected facts, seven
alternating repetitions measured median lookup times of 40.97 ms for the old
constructor algorithm and 9.58 ms for shared lookup. This measures removal of
the repeated mutation rescan, not end-to-end scan throughput. All 81 detectors
completed the touched-source audit with zero findings. The preceding `013af1d`
commit completed CI successfully.

Full suites passed 2,372 tests with 15 skipped on Python 3.11 and 2,387 tests on
Python 3.14. The latter retained the existing 96 fork/thread warnings. A final
docstring-only clarification of the non-function call projection was followed
by the 162-case focused suite, 142 cache tests and complete AST replay. The
docstring edit uses the existing exact, target-owned text-patch operation.
Sphinx built the reference
with its two existing duplicate-description warnings. Receipts are under
`/tmp/nra-constructor-bindings-*`; the unrelated `uv.lock` remains excluded.

## Decorator policy governs validation and geometry (2026-09-06)

Closed the preflight gap found while bootstrapping constructor lookup.
`ReplaceTargetOperation` previously accepted a decorated replacement while
editing only the original header and body. Existing decorators stayed in place,
so a replacement could duplicate `@dataclass(frozen=True)` and produce a module
that failed to import despite a clean simulation.

The operation now declares its `SourceNodeDecoratorPolicy` once. That policy
validates authored decorators and supplies the source span through the existing
`SourceTextGeometry`/`SourceNodeSpan` machinery. The default excludes decorators:
the original decorator block is preserved and decorated payloads fail preflight.
A nominal refinement selecting the inclusive policy owns the complete decorated
span, including a multiline decorator's opening `@`. No separate validation
flag, mirrored span rule or additional dispatch registry was introduced.

The nine regressions cover classes, synchronous and asynchronous functions,
decorated and undecorated originals, actual frozen-dataclass execution, a policy
provided through MI, and simultaneous header/body and decorator edits on one
snapshot. Seven cases failed before the change. The 34-case focused run passes.
The CLI also rejects the original frozen-dataclass payload with exit status 1
and `applied: false`; the fixture SHA-256 is unchanged.

`docs/examples/replacement_decorator_policy.py` performs the production change
in four DSL stages. Replaying from `0e53046` reproduces both complete module
ASTs. All 81 detectors complete the touched-code audit with zero findings.
Sphinx builds the API reference with its two existing duplicate-description
warnings. Receipts use `/tmp/nra-decorator-policy-*`; the CLI fixture is in
`/tmp/nra-decorator-cli.u1x4tU`. The unrelated `uv.lock` remains excluded.

Full validation passes 2,381 tests with 15 skipped on Python 3.11 and 2,396
tests on Python 3.14, with eight workers per suite. Python 3.14 retains the
existing 96 fork/thread lifecycle warnings. Ruff and the whitespace check pass.
Receiver-lifetime proof remains the next implementation task.

## Callable capture precedes argument evaluation (2026-09-06)

Tracing receiver lifetime exposed a timing gap in the shared flow facts.
Python captures the callable before evaluating arguments. NRA retained only
the later invocation position, so `selected((selected := replacement))`
discarded a known original target. Runtime comparison reproduced this for
positional, keyword, unpacked and nested-call arguments. The previous answer
was unresolved, not a falsely exact replacement target.

Each call now owns a `CompactCallableReferenceUse` captured before arguments.
Its `target` view derives from that fact through `AliasProperty`; invocation
keeps its existing later position. The reference-use contract owns positioned
resolution for calls, constructors and callable escapes. The collector shares
one fact factory and does not record direct call targets as non-call escapes.
No second target identity, lookup timestamp rule or dispatch registry is kept.

The nine regressions compare against executing Python, retain use-before-local-
binding as unresolved, distinguish nested-call ordering and reject reuse of
a captured callable as the later local binding. The constructor case uses a
local import: an ordinary class-object alias remains excluded by the existing
class-escape gate. Selecting that class's source declaration alone does not
discharge its runtime product-authority requirements.

The ten-stage `docs/examples/call_target_capture.py` plan was simulated and
applied through the real CLI. Replaying it from `33f71fa` reproduces both
complete edited module ASTs. All 81 detectors completed the touched-source
audit with zero findings. Sphinx built the API reference with its two existing
duplicate-description warnings. Receipts use `/tmp/nra-call-capture-*`.

A four-module collector comparison retained the same 3,497 calls. Seven-run
median collection times were 0.780 seconds before and 0.835 seconds after;
serialized facts grew from 2,835,886 to 3,052,134 bytes. These were sequential
measurements under concurrent test load, not an isolated throughput benchmark.
The extra captured event costs space and collection work; this change is a
correctness prerequisite, not a performance improvement.

Receiver lifetime still requires source evidence for construction hooks,
attribute/descriptor evaluation, intervening effects and escapes. In
particular, the real bootstrap caller evaluates `authority.geometry` before
the nested `authority.replacements_for(...)` call. The collector currently
does not retain getter effects for a lexical attribute-chain callee. Individual
argument origins also still use invocation timing. Neither gap is proved away
by the new target-capture fact.

A read-only runtime probe confirmed the corresponding argument gap:
`consume(selected, (selected := None))` retains the original first argument,
while the current argument-origin projection reports an ambiguous binding.
This is another conservative loss of evidence, not an exact wrong origin.

Full validation passed 2,390 tests with 15 skipped on Python 3.11 and 2,405
tests on Python 3.14, using eight workers per suite. Python 3.14 retains the
existing 96 fork/thread warnings. Both suites completed in about 217 seconds.
Ruff and the whitespace check pass; the unrelated `uv.lock` remains excluded.

## Isolate nominal probe registration in the decorator test (2026-09-06)

The Python 3.14 Ubuntu job in CI run `34018639002` reported 70 registered
operations against 71 concrete production descendants. The new decorator-policy
probe inherited `replace_target`, temporarily replaced the production entry,
and then deleted it in cleanup. The failure depended on which tests shared a
worker. Running the probe followed by the catalogue check in one Python 3.14
process reproduced the exact failure; the same ordered Python 3.11 run passed.

The probe now runs against a copied registry installed through pytest's scoped
monkeypatch, and explicitly verifies restoration of the original registry and
`ReplaceTargetOperation` entry. Production key inheritance is unchanged. The
catalogue assertion compares the complete declaration-derived mapping, so a
future failure identifies the missing or incorrect entry instead of only counts.

The complete decorator file followed by the catalogue checks passes all 23
cases in one process on Python 3.11 and Python 3.14 after isolation.

## Preserve value type through the shared signature binder (2026-09-06)

The argument-capture investigation found an unnecessary boundary restriction:
the binder annotated every transported value as `CompactValueExpression`,
although it only assigns objects to parameters. Captured source uses need to
retain their evaluation events through that same assignment, not through a
second parameter-to-value table or a separate binding implementation.

`CallValueT` now connects positional/keyword arguments, bound arguments and
exact/open binding results. The existing signature and descriptor-binding
algorithms retain these supplied objects unchanged. `CompactCallArguments`
uses an explicit expression projector, shared by source collection and authored
call edits. Its classmethod still returns the selected nominal refinement.
No new binder, runtime value-type dispatcher or fallback projection was added.

Three regressions check arbitrary typed source tokens against Python's native
signature binding, exact object identity through variadic parameters, explicit
unpacking limits and subclass-preserving projection. Existing call and
descriptor tests continue using the expression projector. The focused run
passed 176 tests before adding the final subclass case.

The 21-stage `docs/examples/call_value_polymorphism.py` plan was simulated and
applied through the CLI. Replay from `92079cb` reproduces all four complete
production module ASTs. All 81 detectors complete the touched-source audit
with zero findings. Sphinx builds the updated reference with its two existing
duplicate-description warnings. The seven-run alternating 10,000-binding
probe measured medians of 0.2354 seconds before and 0.2448 seconds after under
concurrent test load; this is not an isolated performance comparison.

Full production-batch validation passed 2,393 tests with 15 skipped on Python
3.11 and 2,408 tests on Python 3.14, using eight workers per suite. Python 3.14
retains the existing 96 fork/thread warnings. These runs preceded the
test-only registry isolation correction, which then passed the explicit
23-case same-process checks on both versions. Ruff and whitespace checks pass.
Receipts use `/tmp/nra-call-values-*` and `/tmp/nra-decorator-registry-*`.

Per-argument capture and its origin consumers are next. This batch removes
their typed-boundary obstacle; it does not yet change argument-origin timing
or establish receiver lifetime. The source collector can now supply one
positioned value per argument through its projector and eliminate its separate
argument traversal. The constructor and forwarding consumers must keep those
positioned values until origin resolution rather than strip them back to names.

## Captured argument values and shared origin selection (2026-09-06)

Closed the argument-timing gap across the collector, call binding, constructor
fields and both conveyor/expansion consumers. Each `CompactValueUse` owns its
expression and evaluation event. The existing argument projector now visits
and captures each argument in one pass. The same captured objects pass through
signature binding; no invocation-time reconstruction or second argument table
is used to recover their origins.

The investigation also found that `_value_origin_for` rejected every name with
multiple writes, independently of its supplied use position. It now selects
the write through `binding_resolution_for`, shared with callable and bound-result
lookup. Alias recursion tracks selected mutations rather than root names, so
`selected = original; selected = selected` retains its actual origin. Ambiguous
control flow and opaque assignments remain unresolved. Opaque expressions now
produce an explicit `OPAQUE_EXPRESSION` result instead of disappearing from the
argument-origin sequence.

`CompactResolvedFunctionCall.bound_value_uses` projects single supplied values
through the existing binding result. `CompactProductConstruction.field_values`
owns the field view and derives its field names. Their consumers retain these
uses until origin resolution, including declared carrier class lookup at each
argument's evaluation event. Five builder-side helpers/indexes were removed;
the four production modules are 86 lines smaller overall.

Five regressions failed before the change. The seven-case addition compares
positional, keyword and nested-call rebinding with real Python execution,
checks repeated-alias write events and later opaque assignments, and verifies
constructor-field and bound-argument object identity. The 229-case focused run
passes. The remaining raw `value_origin_for` call in the conveyor builder is
for a mutation event, not an argument, and correctly retains mutation timing.

The 34-stage `docs/examples/argument_value_capture.py` plan was applied through
the CLI, with an explicit final import-cleanup stage. Replaying from `f31c5d5`
reproduces all four complete production module ASTs. All 81 detectors complete
the touched-source audit with zero findings. Sphinx builds the API reference
with its two existing duplicate-description warnings. Ruff and whitespace
checks pass. Receipts use `/tmp/nra-argument-capture-*`.

Full suites pass 2,400 tests with 15 skipped on Python 3.11 and 2,415 tests on
Python 3.14, with eight workers per suite. Python 3.14 retains the existing 96
fork/thread warnings. The four-module alternating collector probe retained
3,500 calls: seven-run median collection times were 0.738 seconds before and
0.782 seconds after; serialized facts grew from 3,056,153 to 3,350,436 bytes.
Those timings include concurrent test load and do not measure the eliminated
downstream indexes or end-to-end analysis throughput.

Receiver-lifetime proof remains open. Captured argument identity does not prove
construction hooks, descriptor effects, intervening mutations or escape safety.
The real bootstrap caller's `authority.geometry` getter still needs source
effect evidence before the instance-selected `replacements_for` call can be
authorised. The origin diagnostic taxonomy also retains an unused
`CONTROL_FLOW_JOIN` member after moving selection to the shared resolver; it
can be consolidated with the next binding-proof audit.

## Call receiver evidence before instance-lifetime proof (2026-09-06)

The receiver audit reproduced a broken automatic rewrite. Adding an ordinary
`_build.__call__(left, right)` caller to a closed-conveyor fixture still produced
a proven component. The resulting signature collapse left that caller unchanged:
real Python execution returned `(1, 2)` before the rewrite and raised `TypeError`
afterwards. No callable escape was recorded for `_build`.

The shared collector skipped evaluation of every lexical callee chain, dropping
its receiver reads along with the terminal call target. It now visits the
receiver of an attribute call and lets the existing call fact own the terminal
lookup. Bare names remain direct calls without additional non-call escapes;
dynamic receiver expressions retain their normal evaluation. No callable-name
inventory, consumer-specific exception or second escape registry was added.

Seven new cases cover receiver-prefix order, bare and dynamic targets, real
property-before-argument execution, and signature-collapse rejection for direct
and chained `__call__` access. Five failed before the collector correction.
The 130-case focused run passes. The actual NRA bootstrap caller now retains
`authority` and `authority.geometry` reads before selecting `physical_edits`,
followed by the receiver read for `replacements_for` inside its arguments.

The two-stage `docs/examples/call_receiver_capture.py` plan was simulated and
applied through the CLI. Full-module AST replay from `40bf222` matches the
current implementation. Its second stage removes the unused `CONTROL_FLOW_JOIN`
origin result left by the preceding shared-binding refactor. Receipts use
`/tmp/nra-receiver-capture-*`.

Full suites pass 2,407 tests with 15 skipped on Python 3.11 and 2,422 tests on
Python 3.14, with eight workers each (215/216 seconds). Python 3.14 retains the
96 existing fork/thread warnings. All 81 detectors complete the touched-module
audit with zero findings. Ruff, whitespace checks and Sphinx pass; Sphinx retains
its two existing duplicate-description warnings.

A four-module probe retains the same 1,032 calls and adds 512 receiver reads
(4,446 to 4,958). Serialised projection size grows from 1,056,911 to 1,100,675
bytes. Ten alternating collector runs during the full suites have medians
0.156/0.207 seconds, with minima 0.117/0.121 seconds. This is added evidence,
not a demonstrated throughput improvement; concurrent-load timing is insufficient
to isolate its cost. The existing collector still projects lexical call targets
more than once, a potential subsequent factoring/performance investigation.

Receiver reads establish evaluation evidence, not getter purity or receiver
lifetime. The next proof boundary remains constructor results, native-MRO
lookup hooks, intervening effects and escapes. A source declaration or return
annotation alone cannot authorise an instance-selected rewrite.

## Nominal reference facts own read summaries (2026-09-06)

The follow-up audit found another dropped callable escape. Returning
`type(self).renderer.render` produced a working bound method in real Python,
but no escape for `Renderer.render`. The attribute visitor required a lexical
path before invoking the nominal target projection, excluding the runtime-class
member syntax that the projection already supports. Direct calls used that
projection without this restriction.

Non-call attribute loads now retain the shared target projection, including
explicit dynamic results for unresolved syntax. The mutation path still uses
lexical assignment targets; this change does not invent mutation or lifetime
proof for dynamic receivers. Seven new cases exercise native-MRO inherited
member values, opaque reads, and derived loaded-name summaries. Six failed
before the change; the 243-case focused suite passes.

`loaded_value_root_names` is now derived from retained calls and reference uses.
The collector no longer owns a parallel set or updates it in three visitors,
and no longer reparses call targets or loaded attributes merely to populate
that set. Replacing a flow's read facts derives a new summary without carrying
a stale constructor field or cached value across `dataclasses.replace`.

An all-production comparison covers 131 modules and 8,184 flow scopes with
zero loaded-name differences. The new collector retains 681 additional
attribute reads. Raw serialised projection size falls from 25,412,782 to
25,170,365 bytes before any summary property is accessed. One paired collection
pass takes 1.86/1.99 seconds; this is not a throughput improvement claim.

The eight-stage `docs/examples/reference_fact_ownership.py` plan was simulated
and applied through the CLI. Receipts use `/tmp/nra-reference-ownership-*`.
Receiver lifetime and constructor/descriptor effect evidence remain open;
retaining the reference is a prerequisite to those proofs, not a substitute.

The next audit has a concrete reproducer: `Owner.run` assigns `self = Other()`
and then calls `self.method()`. Runtime selects `Other.method`; the repository
still resolves `Owner.method`. Current-class local lookup discards the use
position even though the call fact retains it, and member lookup also needs
receiver-binding evidence. The fix must put that obligation on the shared
current-class target family and reuse binding selection, rather than add
separate rebinding checks to individual refactoring consumers.

That next batch must also review unresolved escape consumption:
`resolve_callable_escape` currently drops a resolution without a declaration,
including its possible symbols. Making receiver lookup unresolved must not
silently remove a relevant method-value use from the signature boundary proof.
Call and non-call reference evidence need consistent fail-closed treatment.

Full suites pass 2,414 tests with 15 skipped on Python 3.11 and 2,429 tests on
Python 3.14, with eight workers each (217/219 seconds). Python 3.14 retains its
96 existing warnings. All 81 detectors complete the touched-module audit with
zero findings. Eight-stage full-module AST replay from `98de5c5`, Ruff and
whitespace checks pass. Sphinx builds with its two existing duplicate-description
warnings; Diataxis keeps the API contract separate from this investigation log.

## Unresolved callable escapes retain authority evidence (2026-09-06)

The unresolved-escape audit reproduced another invalid automatic signature
rewrite. A function conditionally imports `_build` and returns it. The resolver
already identifies `_build` among the possible targets, but the old escape
projection discarded the whole result because its declaration was unresolved.
The conveyor was consequently classified as closed. Runtime execution returned
`(1, 2)` before rewriting and raised `TypeError` afterwards when invoking that
returned callable with its original arguments.

The existing escape carrier is now `CompactCallableEscape` and retains the
typed target-resolution object instead of extracting only a declaration.
Every non-call use retains its result. Symbol queries include possible targets,
and the shared component proof intersects participant symbols with those targets
in one traversal. This removes the per-participant scan without introducing a
second registry or classification hierarchy. Both new regressions failed before
the change; all 245 focused cases pass, including opaque unresolved reads.

The seven-stage `docs/examples/escape_resolution_evidence.py` plan uses the
repository-aware declaration rename and applies the consumer edits through the
CLI. Receipts use `/tmp/nra-escape-evidence-*`. This change precedes the receiver
rebinding guard deliberately: returning an open receiver lookup must not discard
a method-value use that remains relevant to a signature boundary.

The receiver family still needs initial-binding and use-position evidence.
Its selected class and possible alternatives must survive uncertainty; replacing
an incorrect exact lookup with an empty candidate set would not close the proof
gap. Initial receiver aliases and calls captured before a later reassignment
also require shared binding semantics, not blanket rejection of any written
receiver name.

Full suites pass 2,416 tests with 15 skipped on Python 3.11 and 2,431 tests on
Python 3.14, with eight workers each (215/219 seconds). Python 3.14 retains
its 96 existing warnings. Seven-stage full-module AST replay from `d035daf`,
Ruff and whitespace checks pass. All 81 detectors complete the touched-module
audit with zero findings. Sphinx retains its two existing duplicate-description
warnings; the API guarantee is separate from these Diataxis investigation notes.

An eight-run alternating aggregation probe uses 500 participants and 2,000 read
results, including 1,000 exact function references. Both algorithms select the
same participants. Median aggregation time falls from 0.193 seconds for the old
per-participant scan to 0.000298 seconds for the shared intersection. This
measures only the boundary aggregation, not collection, resolution or end-to-end
analysis. Retaining unresolved results increases the cached escape evidence.

For the subsequent receiver work, initial parameter bindings must be distinguished
from writes. The current mutation sequence describes assignments and other
body events, while the existing function declaration already owns the signature.
Pretending parameters are ordinary mutations would force unrelated consumers
to filter those fake writes; initial-binding evidence should instead derive from
the declaration and join the shared selection contract explicitly.

## Function declarations own their flow scopes (2026-09-06)

The initial-binding investigation found that flows did not retain their function
declaration. Instead they stored a copied scope kind and qualified name. The
repository then rebuilt a declaration lookup by that name. Two definitions of
`run`, with parameters `left` and `right`, consequently attached the second
signature to both flow contexts. Both new ownership regressions failed before
the refactor.

`CompactFunctionDeclaration` now implements the `CompactFlowOwner` ABC and is
the actual owner object supplied to its flow. Qualified names derive from the
declaration identity; function scope kind belongs to the declaration class.
`CompactNamespaceFlowOwner` represents module/class-body scopes and rejects
function scope construction. The module's declaration sequence derives from
its flows, and the repository context projects its declaration through the
owner contract. The former name-indexed join and independently supplied context
declaration are removed. Duplicate-name ambiguity is still reported by the
existing declaration-multiplicity authority.

The 13-stage `docs/examples/flow_declaration_ownership.py` plan simulates and
applies through the CLI. Class-member insertion expects one declaration per
operation, so the two class attributes and declaration property are separate
stages in that batch. Tests preserve owner identity across pickle round-trips
and show that replacing module flow facts derives a new declaration view.
The focused suite passes 247 cases; the additional namespace-construction
contract case passes separately. Receipts use `/tmp/nra-flow-owner-*`.

This establishes declaration-owned signature access for initial-binding proof.
It does not yet change binding selection or fix receiver reassignment; those
remain the next obligations, now without needing a second parameter table or
treating parameters as ordinary mutation events.

The all-production comparison covers 131 modules, 8,190 flows and 6,082
function declarations. Declaration payloads and scope identities are unchanged,
and every function declaration is its flow's exact owner object. Raw serialised
projections decrease from 25,172,678 to 19,520,779 bytes before derived views
are accessed. A paired collection pass takes 4.07/5.85 seconds during concurrent
full suites; this is not evidence of a collection speedup.

Full suites pass 2,419 tests with 15 skipped on Python 3.11 and 2,434 tests on
Python 3.14, with eight workers each (218/221 seconds). Python 3.14 retains
96 existing warnings. The 13-stage plan reproduces both complete production
module ASTs from `81596d9`. All 81 detectors complete the touched-source audit
with zero findings. Ruff and whitespace checks pass; Sphinx retains its two
existing duplicate-description warnings. Diataxis keeps the ownership contract
in the API reference and this investigation/measurement record here.
