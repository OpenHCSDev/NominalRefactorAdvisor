---
name: nra-refactoring
description: "Use NominalRefactorAdvisor (NRA), domain-driven ownership reasoning, and source-checked OpenHCS cases to discover underowned Python semantics, adjudicate a nominal contract, and compose proof-gated multistage refactoring DSL plans. Applies to NRA-assisted architectural refactoring and interpreting its scan or proof results."
---

# NRA Refactoring

**The agent does the ownership investigation and proposes the refactor**, then
lets NRA manage deterministic source transformations and intermediate states.
Do not hand the practitioner an unfilled checklist as the result. Optimize for
collapsing semantic surface area: independent authorities, repeated decisions,
forwarding layers and duplicated implementation. Stage count and fewer lines
are useful observations, not the objective or proof of correct factoring.

## Architectural objective: polymorphism maximalism

For behavior-bearing domain families, **prefer an ABC and concrete subclasses**
over enums plus case switches, handler tables or detached configuration maps.
Co-locate each case's related declarations, data, invariants and behavior on its
class; put common algorithms on the public parent and leave only irreducible
hooks on leaves. Compose overlapping nominal capabilities through inheritance
and meaningful MI instead of copying implementation or externalizing selection.
Derive discovery/lookup from the declared family where needed. A consumer should
invoke the public contract, not repeatedly recover which concrete case it has.

An enum needs a specific value-only/boundary justification; it is not an equally
preferred endpoint when other code still interprets its members. Maximize
polymorphic ownership and shared derivation, **not class count**. The agent owns
this architectural judgment under the user's task intent; NRA must separately
check the chosen migration's binding, MRO, effects and behavioral obligations.
Do not turn an unresolved proof into a reason to praise the existing mirror:
record the blocker and work toward proving or refining the nominal design.

## Start with worked transformations

Use the [anti-pattern → owned-structure cookbook](references/pattern-cookbook.md)
for concrete declaration, ABC, shared-algorithm, registration, MI and state moves.
For a complete executable trajectory, read the
[action-dispatch batching example](references/action-batch.md): it uses the existing
DSL, includes a new-case maintenance experiment and tests changed-source rejection.
The example is an authored exact-fixture migration, not automatic async extraction.

## Establish the actual source and contracts

Resolve the requested checkout, branch, dirty work and Python import location
before using examples or cached results. Multiple NRA worktrees and installed
versions can coexist. Do not silently analyze one and apply to another.

Use the current NRA checkout as the API authority:

- `docs/source/api/getting_started.rst`: CLI, ordered plans and insertion rules.
- `docs/source/api/codemod_catalog.rst`: declaration-generated operation catalog.
- `docs/source/api/public_api.rst`: shared proof and execution contracts.
- `docs/source/development/nominal_architecture_playbook.rst`: architectural reasoning.
- `nominal_refactor_advisor/codemod_runtime.py`: recipes, plans and ordered stages.
- `nominal_refactor_advisor/codemod_architecture_guards.py`: guards.
- `nominal_refactor_advisor/codemod.py`: public Python exports.

Inspect `python -m nominal_refactor_advisor --help` in the intended environment.
Discover operations through their registered declarations and generated catalog;
selected source/registrations outrank stale catalog prose. Do not maintain a
second operation inventory in this skill. Read the applicable
operation's constructor, preflight and proof scope before adapting a recipe.

## Reason globally, choose ownership explicitly

First bound a **source corpus** (revision, import roots, selected production
modules), not a presumed domain noun. NRA observations and cheap exact-string,
substring or helper-prefix searches may help select/rank that corpus; false
positives are acceptable as retrieval. Within the selected boundary, census **every original `ClassDef`** through
NRA's `ModuleSyntaxIndex`, then join eligible direct declarations to its
canonical class-family projection for actual bases, methods, ABC/Enum/dataclass
roles. Conditional/function-local/ambiguous classes remain original syntax
rows marked unprojected OPEN; do not invent family members. Only then overlay
residual rosters, string-key accesses, case comparisons, dispatch, forwarded
parameters and delegated state. Keep unaligned classes as alternative-owner
and counterevidence rows. Then the agent names bounded domain questions from the joined
source, rather than requiring the user to supply them. Reuse NRA's lexical/product-flow
owners rather than a second parser, roster or call resolver. Keep original
positions, nested executable ownership, unmatched/OPEN cases, aliases, rebinding,
ordered guards, final fallback and alternate callers. A literal or substring match, matching helper name or nominally resolved
callee is a useful lead, not live binding, domain identity or behavioral proof.
The agent—not the user—clusters source-backed leads into candidate maintenance
questions, keeping nonmatches and unsupported sites visible.

Start with the bounded class-first corpus; expand to a complete scan of the
relevant package and dependency context **when a proposed ownership claim
depends on that context**. Use `--context-root` for explicit global context
while limiting reported findings to selected paths. Tests are excluded by
default; do not exclude production dependencies merely to obtain a smaller or
cleaner scan. Inspect `scan_status` and analyzed/omitted detector counts.
A `focused_local_partial` loop result is useful feedback, not a global
ownership audit.

For each proposed change, **the agent must trace** the declaration that
determines each answer, its implementations, consumers and dependent projections.
Follow [the owner-finding procedure](references/finding-owners.md) and supply
an actual ranked, source-backed ownership proposal with counterevidence;
do not ask the user to perform the trace. Repetition identifies a maintenance object;
it does not alone establish its correct owner. Look for an existing richer
authority before introducing another carrier, wrapper or registry.

For this project's nominal architecture:

- Where an admitted domain contract supports it, put shared implementation on
  its meaningful public ancestor and irreducible behavior on substitutable
  children. MI requires positive MRO, constructor and method-lookup evidence;
  priority checks are not automatically replaceable by inheritance.
- Where an admitted closed-family contract supports it, put leaf behavior on
  its owner and derive consumers. An enum plus another handwritten roster or
  a service that retains the same dispatch does not remove an authority.
- Derive views from declarations or original proof objects. A typed class can
  still mirror another authority; introducing classes is not sufficient.
- Trust guaranteed fields and ABC contracts directly. Fix the violated boundary
  instead of adding `getattr` defaults, string-key fallbacks or Protocol substitutes.

When a lead suggests Enum behavior, an ABC, multiple inheritance or
class-definition registration, read [nominal mechanism choices](references/nominal-mechanisms.md).
It contrasts the user's status/dict/flag examples with exact #38/#44/#58/#60/#69
primary-source transitions and falsifiers; no mechanism is an automatic fix.
Preserve real external formats and genuinely optional contracts. These are
ownership rules, not a blanket ban on strings, branches or dictionaries.

Sketch the destination ownership and migration closure before editing: declaration,
bases/MRO, moved members, fields, signatures, callers, imports and obsolete uses.
Consider what the projected change will expose next; do not optimize a single
finding into a local minimum that leaves competing authorities intact.

## Produce a bounded decision receipt before prescribing a refactor

Use [the reusable transformations and case contrasts](references/architecture-decisions.md)
for the task-relative required-answer model and exact #44/#58/#60 positive/negative
controls. **Extract and fill**: bounded context, domain noun and required questions; the
existing declarations and executable consumers with source revision and original
positions; each independently writable authority versus derived view; proposed
required/forbidden implementation–consumer/class pairs (`R*`), independent
provider roles, alternative UI/transport/schema ownership; and every OPEN
binding, alias, priority, unknown/fallback, dynamic or alternate-caller row.
A missing subclass is required only if the admitted relation demands it. The
Agent must propose and justify a relation (or a specific alternative) from the
admitted task intent and source; its `R*` remains **provisional** until the
bounded context admits its pairs and exclusions under explicit task decision
authority. First inspect more code, tests, history and counterexamples to settle
uncertainty yourself. Ask the user only the **minimal set of precise questions**
when genuinely unavailable domain intent prevents an authorized choice; do not
outsource routine source tracing or pattern recognition. Neither an agent nor human assertion
proves equivalence.

This separates the leverage layers: cheap heuristics and NRA observations
find where to look; the agent chooses a bounded domain ownership decision;
NRA's existing selectors, staged planner, preflights and guards then mechanize
and check a chosen trajectory. For a **prescriptive** plan, first admit the
required/forbidden relation and determining owner(s) under that decision authority. Before admission, label any
candidate DSL sequence exploratory/unproved and never present it as the repair.
There may be one determining authority *per admitted fact family*, not one
class/aggregate for independent domain, UI, transport and setup roles.

## Compose a trajectory, not isolated edits

Read [references/batching.md](references/batching.md) and
[ownership-to-DSL gates](references/ownership-to-dsl.md) when authoring or
extending an ordered plan. Prefer declaration-selected movement, promotion, projection,
rename and call-migration operations over copying implementations into authored
replacement bodies. Use exact target patches when needed, but label the semantic
decision and unsupported DSL gap honestly.

Simulate dependent stages against projected source, inspect projected findings
and selectors, then extend the same trajectory. NRA's global ownership analysis
helps choose the trajectory; operation preflight alone does not choose it.
Do not bypass an unproved planning horizon to export or apply a synthesized plan.

Review one combined diff and apply through the revision-checked NRA transaction
when authorized. Application rechecks supplied analysis-only source files too.
If source changed since simulation, rebuild and simulate rather than forcing a
stale write set. Rescan after creating modules or changing dependency boundaries.

## Keep evidence distinct

Report separately:

- **Coverage:** analyzed source/dependency context, detector omissions and scan mode.
- **Ownership:** admitted or provisional `R*`, determining authority per fact,
  independent provider/adapter roles, and applicable claims/guards.
- **Replay:** exact stages, preflight requirements, final source and unchanged input.
- **Behavior:** the specific native proof obligations or executed tests that passed.

`is_clean` means the configured operation preflights and guards passed. Inspect
which guards were present; an empty suite does not certify global architecture.
Valid Python, resolved callees or signature binding do not prove an authored
body/expression equivalent. Class simulation does not execute metaclasses or
class bodies. Preserve fail-closed unsupported native behavior; do not fabricate
creator frames, suppress failures or reinterpret observations as invocation proofs.

Unchanged parsed modules can retain local evidence through an authenticated
transition. Changed dependencies still require fresh global resolution, including
absence and ambiguity. Cross-process cached summaries are not completed native
execution proofs. Consult the current proof-reuse contracts for cache work.

Validate unproved behavior, integrations and the refactoring tool itself. Batch
tests after a coherent sequence when its intermediate states need not run, but
do not replace behavioral validation with syntax replay. Use bounded parallel
tests within the available resource budget; this user's usual bounds are 60
seconds per shard, or 165 for larger shards. Preserve timed-out/failed evidence;
split slow shards without weakening inputs or assertions. Check CLI budget flag
scope: a shell timeout bounds the command when its internal flag does not.

Claim completion only when the admitted relation is represented with every
intended consumer deriving from its authority **and** scoped behavior/equivalence
gates pass. Otherwise stop with named OPEN obligations, not a completion claim.
Keep a concise checkpoint with exact source revision, recipe, commands, results,
proof limits and unfinished work. Publish only within the user's authorization,
and verify CI for the exact pushed revision. Keep task caches in an owned root
and clean them after workers exit, retaining recipes and validation evidence.
