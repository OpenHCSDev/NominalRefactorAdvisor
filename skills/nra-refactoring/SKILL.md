---
name: nra-refactoring
description: "Use NominalRefactorAdvisor (NRA) to audit Python semantic ownership and compose declaration-targeted, multistage refactoring DSL plans. Applies to NRA-assisted refactoring, bootstrapping NRA itself, and interpreting its scan or proof results."
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
- `nominal_refactor_advisor/codemod.py`: public Python exports.

Inspect `python -m nominal_refactor_advisor --help` in the intended environment.
Discover operations through their registered declarations and generated catalog;
do not maintain a second operation inventory in this skill. Read the applicable
operation's constructor, preflight and proof scope before adapting a recipe.

## Reason globally, choose ownership explicitly

Start with a complete scan of the relevant package and dependency context.
Use `--context-root` for explicit global context while limiting reported findings
to selected paths. Tests are excluded by default; do not exclude production
dependencies merely to obtain a smaller or cleaner scan. Inspect `scan_status`
and analyzed/omitted detector counts. A `focused_local_partial` loop result is
useful feedback, not a global ownership audit.

For each proposed change, trace the declaration owner, its implementations,
consumers and dependent projections. Repetition identifies a maintenance object;
it does not alone establish its correct owner. Look for an existing richer
authority before introducing another carrier, wrapper or registry.

For this project's nominal architecture:

- Put shared implementation on the owning ABC/ancestor; keep concrete classes
  as small behavior hooks. Use MI and declared MRO where independent nominal
  capabilities compose, rather than recreating dispatch or priority tables.
- Put closed-family leaf behavior on its existing declaration/member. Consumers
  should use the nominal contract, not branch on strings or concrete types.
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

## Compose a trajectory, not isolated edits

Read [references/batching.md](references/batching.md) when authoring or extending
an ordered plan. Prefer declaration-selected movement, promotion, projection,
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
- **Ownership:** the chosen nominal authority and applicable global claims/guards.
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

Keep a concise checkpoint with exact source revision, recipe, commands, results,
proof limits and unfinished work. Publish only within the user's authorization,
and verify CI for the exact pushed revision. Keep task caches in an owned root
and clean them after workers exit, retaining recipes and validation evidence.
