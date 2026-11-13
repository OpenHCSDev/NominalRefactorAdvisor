# From an ownership decision you've made to NRA's existing DSL

Read the selected checkout's `../../../docs/source/api/getting_started.rst`, `../../../docs/source/api/codemod_catalog.rst`, `../../../docs/source/api/public_api.rst`, `nominal_refactor_advisor/codemod_runtime.py`, `nominal_refactor_advisor/codemod_architecture_guards.py` and the registered operation declarations that apply. If generated docs disagree with the selected source and registrations, the source wins. The [batching guide](batching.md) covers `CodemodPlanDocument` versus `CodemodPlanSequence` and the same-snapshot class-member/alias insertion trap. This guide covers the **decision and proof gates**, not a list of operations.

## What you need first, and the shortest path

A real plan needs these for the scope you're looking at: the required and forbidden implementation–consumer/class pairs you've accepted, the owner for each fact family, the roles that vary independently, the known adapter boundaries, and the cells that are unknown or excluded. An LLM can propose these from evidence, and can derive pairs that follow from the rules it was given or from what the task says to do. It can't turn source similarity into business meaning. Write down who or what authorized each accepted pair. A candidate worksheet only supports an exploratory preview with no proof. Tie every operation to one accepted decision, one source witness, and a condition that rejects the plan if evidence is missing. One file or one class is not automatically one authority for every role.

Pick only the stages the migration needs:

1. Set up or import an owner and a real nominal contract, without leaving a second live hand-written roster.
2. Where native MRO, constructor and method-lookup rules allow it, promote or move existing members and fields with declaration-selected operations. A hand-written body is separate behavioral debt.
3. Derive each accepted fact's registry/schema/UI/transport projection from the owner that determines it. Keep genuinely independent UI or transport adapters and explicit external translations.
4. Change signatures and callers with exact resolved cardinality. Check bound vs. unbound receiver form, argument evaluation and keywords, defaults, decorators, properties, overrides and other callers. Selecting and binding a declared call doesn't prove what the invocation does.
5. Delete dead rosters, assignments, forwarding layers and imports only after checking what still uses them and what evaluating them does. Deleting an assignment deletes its initializer effects too. Add a guard that keeps a retired, independently writable surface from coming back.
6. Simulate the whole trajectory, and look at each projected source, index and finding plus the combined diff. Widen one missing proof boundary at a time, not whatever the regex or clone count suggests. If a guard fails, don't weaken it to get a green report.

Existing `RefactorRecipe` holds source-selected operations, architecture guards and authority claims. `CodemodPlanDocument` groups operations against one snapshot. `CodemodPlanSequence.from_operations`/`.compose` reindexes later stages against earlier projected output. A parse-clean simulation doesn't execute a class body or metaclass. Candidates synthesized from the current snapshot may lack a complete trajectory, so don't skip past a planning horizon that isn't proved. A hand-written expression or replacement body needs its own native and behavioral evidence, even when the selected source and generated syntax are exact.

Preview only, adjusting the CLI options to the selected checkout's `--help`:

```bash
python -m nominal_refactor_advisor path/to/package \
  --context-root path/to/context \
  --codemod-plan plan.json --codemod-simulate \
  --codemod-project-findings --codemod-project-source-index --json
```

## Two separate checklists

**Architecture:** Did you keep every accepted `R*` pair? Which source declaration supplies each answer? Are required and forbidden edges and independent providers preserved? Do derived views trace back to the right owner, while domain, UI and transport own different obligations where they should? What aliases, setup-only or external cases, unmatched source rows, unknown defaults or leftover writable mirrors are still around? Guard scope and the source boundary matter. Zero findings or an empty guard suite proves nothing.

**Behavior:** Which native before/after and targeted/integration tests check the real MRO, constructors, descriptors, decorators/defaults, import/call binding, property timing, evaluation order, priority, unknown inputs and external formats? Is the source quiet, with nobody else editing it, and still the revision you selected? Source hashes don't guard against concurrent ABA and don't give you an atomic multi-file snapshot. Signature binding, syntax simulation and a finite set of passing tests are separate evidence, and none of them proves arbitrary Python equivalence. For behavior the tooling doesn't support, stop and name the OPEN obligations, or redesign the task to a smaller scope. Never invent a rewrite and call it safe.

Apply a reviewed plan only when you're authorized to, through NRA's revision-checked transaction. Then rescan the changed dependencies and validate the exact revision that got written. Hand off: source/revisions and scan omissions, whether the decision is settled or still a candidate, plan stages with preflight/diff, the architecture checklist, the behavior checklist, and the remaining OPENs. Prefer one coherent sequence over rerunning a full-repo scan after every half-baked idea.
