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
