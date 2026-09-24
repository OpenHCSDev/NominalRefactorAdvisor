# From an admitted ownership decision to NRA's existing DSL

Read the selected checkout's `../../../docs/source/api/getting_started.rst`, `../../../docs/source/api/codemod_catalog.rst`, `../../../docs/source/api/public_api.rst`, `nominal_refactor_advisor/codemod_runtime.py`, `nominal_refactor_advisor/codemod_architecture_guards.py` and applicable registered operation declarations. The selected source/registrations outrank generated prose if stale. The maintained [batching guide](batching.md) explains `CodemodPlanDocument` versus `CodemodPlanSequence` and the same-snapshot class-member/alias insertion trap; this guide adds **adjudication and proof gates**, not an operation inventory.

## Preconditions and minimal trajectory

A prescriptive plan needs the bounded domain's **admitted** required and forbidden implementation–consumer/class pairs, determining owner per fact family, independently varying roles, known adapter boundaries and unknown/excluded cells. An LLM can propose these from evidence, but a provisional worksheet authorizes only an exploratory/unproved preview. Link every operation to one admitted decision, source witness and reject-on-missing-evidence condition. Do not equate one file or one class with one semantic authority for all roles.

Choose only the stages needed for the admitted migration:

1. Establish/import an authority and a meaningful nominal contract without leaving a second live handwritten roster.
2. Where native MRO, constructor and method-lookup obligations allow it, promote/move existing members and fields through declaration-selected operations; an authored body is a separate behavioral debt.
3. Derive each admitted fact's registry/schema/UI/transport projection from its determining authority. Preserve genuinely independent UI or transport adapter ownership and explicit external translations.
4. Change signatures/callers with exact resolved cardinality; check bound/unbound receiver form, argument evaluation and keywords, defaults, decorators, properties, overrides and alternate callers. Declared-call selection/binding cannot prove invocation effects.
5. Delete obsolete rosters, assignments, forwarding layers and imports only after residual-use and evaluation checks. Deletion of an assignment also deletes its initializer effects. Add a guard that prevents a retired independently writable surface from returning.
6. Simulate the composed trajectory; inspect each projected source/index/finding and the combined diff. Expand one missing proof boundary, not the apparent regex or clone count. If a guard fails, do not weaken it for a green report.

Existing `RefactorRecipe` holds source-selected operations, architecture guards and authority claims. `CodemodPlanDocument` groups operations against one snapshot; `CodemodPlanSequence.from_operations`/`.compose` reindexes later stages against earlier projected output. A parse-clean simulation does not execute a class body or metaclass. Current-snapshot synthesized candidates may lack a complete trajectory: do not bypass an unproved planning horizon. An authored expression or replacement body needs its own native/behavioral evidence, even when selected source and generated syntax are exact.

Preview only, adjusting CLI options to the selected checkout's `--help`:

```bash
python -m nominal_refactor_advisor path/to/package \
  --context-root path/to/context \
  --codemod-plan plan.json --codemod-simulate \
  --codemod-project-findings --codemod-project-source-index --json
```

## Two independent validation ledgers

**Architecture:** Are all admitted `R*` pairs retained? Which source declaration supplies each answer? Are required/forbidden edges and independent providers preserved? Do derived views trace back to their appropriate authority, while domain, UI and transport own different obligations when needed? What aliases, setup-only/external cases, unmatched source rows, unknown defaults or residual writable mirrors remain? Guard scope and the source boundary matter; zero findings or an empty guard suite prove nothing.

**Behavior:** Which native before/after and targeted/integration tests check actual MRO, constructors, descriptors, decorators/defaults, import/call binding, property timing, evaluation order, priority, unknown inputs and external formats? Is source under exclusive quiescence and still the selected revision? Source hashes are not a concurrent ABA guarantee or atomic multi-file snapshot. Signature binding, syntax simulation and finite passing tests are separate evidence, not a proof of arbitrary Python equivalence. For unsupported behavior, stop with named OPEN obligations or redesign the bounded task; never fabricate a safe rewrite.

Apply a reviewed plan only when authorized through NRA's revision-checked transaction, then rescan changed dependencies and validate the exact written revision. Handoff: source/revisions and scan omissions, adjudicated or provisional receipt, plan stages and preflight/diff, architecture ledger, behavior ledger, and residual OPENs. Prefer one coherent bounded sequence over repeated full-repo scans after every speculative idea.
