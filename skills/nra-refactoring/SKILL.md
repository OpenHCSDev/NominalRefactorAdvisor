---
name: nra-refactoring
description: "Use NominalRefactorAdvisor (NRA), domain-driven ownership reasoning, and source-checked OpenHCS cases to discover underowned Python semantics, adjudicate a nominal contract, and compose proof-gated multistage refactoring DSL plans. Applies to NRA-assisted architectural refactoring and interpreting its scan or proof results."
---

# NRA Refactoring

Help the practitioner decide where semantics belong, then let NRA manage the
deterministic source transformations and intermediate states. Optimize for
collapsing semantic surface area: independent authorities, repeated decisions,
forwarding layers and duplicated implementation. Stage count and fewer lines
are useful observations, not the objective or proof of correct factoring.

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

Bound one domain question first. Inventory **actual classes, ABCs, enums,
dataclasses, ancestry and methods** using NRA's existing source/class indexes;
then overlay residual rosters, string-key accesses, case comparisons, dispatch,
forwarded parameters and delegated state. Reuse NRA's lexical/product-flow
owners rather than a second parser, roster or call resolver. Keep original
positions, nested executable ownership, unmatched/OPEN cases, aliases, rebinding,
ordered guards, final fallback and alternate callers. A literal match, matching
helper name or nominally resolved callee is a search question, not live binding,
domain identity or behavioral proof.

Start with the bounded class-first source question; expand to a complete scan
of the relevant package and dependency context **when the proposed ownership
claim depends on that context**. Use `--context-root` for explicit global context
while limiting reported findings to selected paths. Tests are excluded by
default; do not exclude production dependencies merely to obtain a smaller or
cleaner scan. Inspect `scan_status` and analyzed/omitted detector counts.
A `focused_local_partial` loop result is useful feedback, not a global
ownership audit.

For each proposed change, trace the declaration owner, its implementations,
consumers and dependent projections. Repetition identifies a maintenance object;
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

Preserve real external formats and genuinely optional contracts. These are
ownership rules, not a blanket ban on strings, branches or dictionaries.

Sketch the destination ownership and migration closure before editing: declaration,
bases/MRO, moved members, fields, signatures, callers, imports and obsolete uses.
Consider what the projected change will expose next; do not optimize a single
finding into a local minimum that leaves competing authorities intact.

## Produce a bounded decision receipt before prescribing a refactor

Read [architecture decisions and case contrasts](references/architecture-decisions.md)
for the task-relative required-answer model and exact #44/#58/#60 positive/negative
controls. Record: bounded context, domain noun and required questions; the
existing declarations and executable consumers with source revision and original
positions; each independently writable authority versus derived view; proposed
required/forbidden implementation–consumer/class pairs (`R*`), independent
provider roles, alternative UI/transport/schema ownership; and every OPEN
binding, alias, priority, unknown/fallback, dynamic or alternate-caller row.
A missing subclass is required only if the admitted relation demands it. The
LLM may propose and justify a relation or abstain, but its `R*` is **provisional**
until the bounded context admits its pairs and exclusions under explicit task
decision authority. Escalate disputed domain meaning; neither a model answer
nor a human assertion proves equivalence.

For a **prescriptive** plan, first admit the required/forbidden relation and
determining owner(s) under that decision authority. Before admission, label any
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
