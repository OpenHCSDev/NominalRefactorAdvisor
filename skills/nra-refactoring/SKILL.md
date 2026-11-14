---
name: nra-refactoring
description: "Use NominalRefactorAdvisor (NRA) to find slop, trace its owner, and compose proved codemod plans that fold it. Applies to NRA-assisted refactoring and interpreting its scan results."
---

# NRA Refactoring

**You investigate, propose the fix, and NRA applies it deterministically.**
Don't hand someone a checklist. Collapse the slop: repeated decisions,
forwarding layers, duplicated implementation. The goal is fewer facts
maintained by hand, not fewer lines or more stages.

## Goal: polymorphism over switches

For families with behavior, **use an ABC and concrete subclasses** —
not enums with case switches, handler tables or config maps.
Put each case's data and behavior on its class. Put shared algorithms on
the parent. Leave only what differs on the leaves. Compose overlapping
capabilities through MI instead of copying code or routing around it.
A consumer calls the method, never figures out which case it has.

An enum earns its place only when values cross a boundary and nothing
interprets them. Maximize what inheritance derives, **not class count**.
You decide the architecture from the code and the task. Source patterns
are leads, not business rules. NRA checks the chosen migration's MRO,
bindings and effects — that's its job. If a proof doesn't go through,
record the blocker and keep working, don't declare the existing code
correct because the tool couldn't finish.

## The front door: count, declare the family, let NRA fold it

**A consumer that knows which subclass it has is under-abstracting.**
Every `isinstance`, `match` class pattern or member comparison outside
the family's own decoder restates a case decision the family should
answer once. m consumers × n cases → m + n.

1. **Count first.** `python -m nominal_refactor_advisor PKG
   --count-member-awareness [--json]` shows every consumer that tests
   two or more members on the same subject, grouped by the set of
   members it checks. Largest bypassing group first. Then run the full
   scan (`--json`, budget 600, `complete`, zero omitted): each restated-
   fact detector reports what's restated, the owner that should hold it,
   and the fix. A finding whose recipe names an operation is fixable now.
   The rest are the codemod backlog. They're findings, not "leads" — a
   restated fact is slop by definition. Only the fix can be unproved.

2. **Pick the owner shape** for the top group:
   - consumers only ask *which nodes are in the set* → `ForeignFamily`
   - consumers ask the members *different questions* → `ForeignAdapter`
     family: one adapter per foreign class, `decoded(node)` is the one
     type test, siblings keep only what differs
   - own enum read as cases outside it → class family on the enum's owner

3. **Fold through the goal search.** Each fold ships as detector +
   operation + recipe + concept:
   `python -m nominal_refactor_advisor PKG --codemod-refactor-goal
   CONCEPT --codemod-goal-detector DETECTOR --codemod-apply --json`.
   The fold proves each site and refuses anything it can't prove.
   Refusals are the next thing to widen, not a reason to hand-edit.

4. **CI proves behavior.** Commit the applied diff with the command and
   wall time. Re-run the census: the group must shrink.

5. **Every refactor also improves NRA.** A script you needed becomes a
   command, a dodge an agent used becomes a detector, a refusal that
   blocked a real site becomes a widening. A refactor that leaves the
   tool unchanged left the next one just as expensive.

Check subagent work from the diff and by re-running the command; a
report is a claim.

## Don't wait on synthesized recipes

The recipe synthesizer hedges: one unresolved decision and it gives you
nothing (`no_evaluator`). Not a partial plan with the gap named —
nothing. NRA on itself (2026-11-13): 4,919 findings, zero recipes.
The synthesizer flinches where the agent should decide.

**Author plans from existing operations. That is the primary path.**
The synthesizer should produce candidate recipes with choices marked.
Until it does, every finding is a finding the agent composes a plan for,
using the 110 registered operations directly. The operations have
preflights that prove each site. What the synthesizer refused to decide,
the agent decides and the preflight checks.

When a finding names the restated fact and its owner, the plan writes
itself: `patch_target` for exact text, `insert_class_member` for a dict
or property, `compose_existing_class` for MI, `create_mixin` for a new
shared parent, `factor_exact_method_role` for repeated methods,
`dispatch_to_polymorphism` for string dispatch. Every operation's
preflight is a proved check. An authored plan with proved operations is
stronger than a synthesized recipe that hedged its way to silence.

## Authored plan gotchas (learned 2026-11-13)

- **`ensure_import` takes `import_source`** (the full import line as a
  string), not `import_module`/`import_names`.
- **`patch_target` does substring matching** within the target class.
  `"line: int"` matches `"from_namespace_line: int"`. Use the full
  indented source line including leading whitespace to disambiguate.
- **`add_class_base` changes field order.** Base fields come first in
  the MRO. Any positional constructor call breaks. After applying, grep
  every changed class for positional construction and convert to kwargs.
  Script: `ast.walk` for `ast.Call` with `node.args` and func name in
  the changed set.
- **`patch_target` replacement `""` (empty) removes the line.** But it
  leaves a blank line. The codemod pipeline handles this — parse
  validation still passes.

## Running a sweep: lessons from agent-comms (2026-11-11)

- **Fill the plan, simulate once, apply once.** Compose with
  `--codemod-compose-sequence A.json B.json …`, simulate the whole
  sequence from the branch head. One simulate per slice is the
  incremental loop. A refusal inside is the only evidence an operation
  is missing.
- **`no_evaluator` does not mean no operation.** Grep the registered
  operations before calling something a gap. `CreateMixinOperation` +
  `ComposeExistingClassOperation` was the extract-new-base everyone kept
  asking for, and `KeywordizeConstructionOperation` existed a whole
  night with no synthesis branch building it. Sort rejections by their
  stated reason first. The reason names the real gap.
- **Authored plans are never blocked.** Only synthesized one-shot export
  is. Author with existing operations (`patch_target`,
  `insert_before_target`, `compose_existing_class`) and let each
  preflight prove its site. A hand edit is worse than an authored op.
- **Select by declaration, not by id.** Synthesized recipes use
  `target_id` from the scan, and those go stale once an earlier plan
  rewrites the file. Use `file_path` + `target_qualname`, or rescan.
- **Deleting a class's only member needs `compose_existing`**, which
  writes `pass` on its own. `patch_target` leaves an empty body and
  parse validation refuses it.
- **Scan with every consumer.** `dead_module` and `dissolve_module` only
  see the scan. Pass `--context-root` for sibling repos that import
  the package before deleting a module.
- **Check which NRA you're running.** Installing the target into the NRA
  venv can replace editable NRA with a pinned wheel. After any
  `uv pip install`, print `nominal_refactor_advisor.__file__`.
- **Results travel; caches don't.** Finding ids hash detector, pattern,
  summary and evidence, not the build. Plans posted as JSON match on any
  agent's scan of the same source.

## Goal search: the automatic trajectory finder

The most powerful move: NRA explores every proved refactoring trajectory
and emits a multi-stage plan. Use it when a concept + detector pair exists.

```bash
# Find and apply all forwarder inlines
nominal-refactor-advisor PKG \
  --codemod-refactor-goal forwarder_inline \
  --codemod-goal-detector forwarding_facade \
  --codemod-apply

# Just explore without applying
nominal-refactor-advisor PKG \
  --codemod-refactor-goal CONCEPT \
  --codemod-goal-detector DETECTOR
```

**Do NOT pass `--codemod-simulate` with goal search** — it rejects the
combination. The goal search simulates internally at each stage.

Registered concept keys (from `refactor_concepts.py`):
`auto_register`, `auto_register_class_registry`,
`auto_register_mro_ordering`, `auto_register_strategy_family`,
`call_mapping_authority`, `class_body_composition`,
`class_family_authority`, `constructor_kwarg_carrier_projection`,
`constructor_kwarg_collapse`, `dataclass_payload_projection`,
`dead_module_dissolution`, `declaration_reference`,
`derived_projection`, `false_distinction_erasure`,
`foreign_adapter_bind_chain`, `foreign_family_routing`,
`forwarder_inline`, `legacy_path_erasure`, `nominal_boundary`,
`optional_ast_field_decoder`, `package_move`,
`semantic_carrier`, `static_projection_adapter`,
`tuple_dict_return_nominalization`.

Each concept key pairs with one or more detector IDs
(`--codemod-goal-detector`). The detector provides findings; the concept
provides the recipe synthesizer.

**Goal search often finds "Nothing to do?!"** — it found targets but
the synthesizer produced zero recipes. This means the closed-world proof
failed for every candidate (method names too common, string references,
etc.). When this happens, author the plan with `patch_target` and
`add_class_base` instead. The goal search is fast for target
identification (~1s for 23 targets) but the synthesizer hedges on
anything it can't fully prove. Don't wait for it to prove what you
can author.

## Worked examples

Read [theory → evidence → migration](references/theory-to-workflow.md)
for the Paper 1 anchor. The guide includes a worked #60 migration.

Use the [pattern cookbook](references/pattern-cookbook.md) for concrete
ABC, shared-algorithm, registration, MI and state moves.

The [action-dispatch example](references/action-batch.md) uses the DSL
end-to-end with a new-case experiment and changed-source rejection.

[String dispatch → whole plan](references/source-backed-dispatch-planning.md)
shows a source-backed dispatch lead through full migration closure.

[Heuristic leads → trajectory](references/heuristic-to-trajectory.md)
turns clustered findings into one multistage plan.

## Source and contracts

Resolve the checkout, branch and Python location before using examples
or cached results. Multiple NRA worktrees can coexist. Don't silently
analyze one and apply to another.

API authority is the current NRA checkout:

- `docs/source/api/getting_started.rst`: CLI, plans, insertion rules
- `docs/source/api/codemod_catalog.rst`: the operation catalog
- `docs/source/api/public_api.rst`: proof and execution contracts
- `docs/source/development/nominal_architecture_playbook.rst`: reasoning
- `codemod_runtime.py`: recipes, plans and ordered stages
- `codemod_architecture_guards.py`: guards
- `codemod.py`: public Python exports

Run `python -m nominal_refactor_advisor --help` in the target
environment. Discover operations through their registrations and the
generated catalog. Don't keep a second operation inventory here.

## How to investigate ownership

Record the task rules and the code structure first — don't replace them
with a pattern inferred from one file. Start from the classes:

1. Census every `ClassDef` through NRA's `ModuleSyntaxIndex`
2. Join to the class-family projection for bases, methods, ABC/Enum roles
3. Overlay the restated facts: rosters, string keys, dispatch, forwarded
   params, delegated state
4. Cluster the leads into candidate questions

**The agent traces the owners, not the user.** Follow
[finding-owners.md](references/finding-owners.md): trace the declaration
that determines each answer, its implementations, consumers and
projections. Repetition identifies a maintenance object; it doesn't
alone establish who should own it. Look for an existing richer owner
before adding another wrapper or registry.

For each proposed change, trace every occurrence of the shared behavior:
inputs, effects, ordering, role, the declaration that determines it.
Factor shared algorithms onto the meaningful parent. Use MI for
overlapping capabilities. Keep each class's irreducible behavior.

Architecture rules for this project:
- Shared implementation on the public ancestor, irreducible behavior on
  substitutable children. MI needs positive MRO and constructor evidence.
- Leaf behavior on its owner, consumers derive. An enum plus a
  handwritten roster doesn't remove the dispatch.
- Derive views from declarations. A typed class can still mirror another
  owner; introducing classes isn't sufficient.
- Trust guaranteed fields and ABC contracts directly. Fix the violated
  boundary instead of adding `getattr` defaults or Protocol substitutes.

Read [nominal mechanisms](references/nominal-mechanisms.md) when a lead
suggests an Enum, ABC, MI or registration.

## Before prescribing a refactor

Use [architecture-decisions.md](references/architecture-decisions.md)
for the required-answer model and the #44/#58/#60 controls.

Fill in: what the code does now (declarations, consumers, with source
revision and lines); which facts are independently maintained vs derived;
the proposed required/forbidden pairs (what should inherit what, what
should not); and every unresolved question. Include raw record shapes
from the full scan (`--json --raw-findings --json-payload full`).

The agent decides routine source questions. Ask Tristan only for
genuinely missing domain intent. Neither an agent's claim nor a human's
assertion proves behavioral equivalence — the tests prove it.

Sketch the destination before editing: declaration, bases/MRO, moved
members, fields, signatures, callers, imports, deleted paths. Think
about what the change exposes next; don't optimize one finding into a
local minimum that leaves competing owners intact.

## Compose a trajectory, not isolated edits

Start from real consumer friction. Trace the shared behavior. Fix the
owner. Delete the replaced paths. Mechanize what NRA supports. Run the
tests. Measure the actual work removed.

Keep the complete scan's findings and source manifest. Assemble the
largest supported ordered codemod batch. Simulate dependent stages on
projected source. Apply with revision checks. Validate the consumer.
Keep unsupported obligations as an explicit OPEN list.

Read [batching.md](references/batching.md) and
[ownership-to-dsl.md](references/ownership-to-dsl.md) when composing
plans. Prefer declaration-selected operations (move, promote, project,
rename, call-migration) over copying implementations into authored
replacement bodies. Use `patch_target` when needed, but say what the
semantic decision is and what DSL gap remains.

Simulate, inspect projected findings, extend the trajectory. NRA's
global analysis helps choose the trajectory; preflight alone doesn't.

Apply through the revision-checked transaction. If source changed since
simulation, resimulate. Rescan after creating modules or changing
dependency boundaries.

## Keep evidence distinct

Report separately:
- **Coverage:** what was analyzed, what was omitted, scan mode
- **Ownership:** which pairs are proposed, which are proven, which are open
- **Replay:** exact stages, preflight results, final source
- **Behavior:** which tests passed, which proof obligations remain

`is_clean` means the configured preflights and guards passed. Check
which guards were present — an empty suite certifies nothing.

For a refused operation, read the full causal chain. Don't report the
target as the problem when the refusal names a different declaration.

Don't claim done until the proposed pairs are represented and behavior
tests pass. Otherwise stop with named OPEN obligations, not a
completion claim. Keep a checkpoint: source revision, recipe, commands,
results, proof limits, unfinished work.

## Engine gaps (2026-11-11)

1. **Narrowed composition moves (852ca705).** `candidates_touching`
   rebuilt three repo-wide indexes per virtual snapshot. Now
   ClassUseIndex resolves lazily. 93.9 → 84.0 s, identical report.
2. **All refusals in one run (b29c899b).** `edit_batch_for_recipes`
   collects every refusal instead of stopping at the first.
3. **Stored-row refusal (752f143a).** Deleting a class a stored record's
   field can hold now refuses, naming the field.
4. **Module-level alias binding (9c4559a6).** `sb = source_module`
   followed transitively is now a binding.
5. **Silent no-op refusal (eb339dbc).** `remove_import_names` on a
   TYPE_CHECKING name now refuses instead of producing zero edits.
6. **Guard blank lines (abe5641d).** Emptied guard blocks absorb their
   separator lines.
7. **Package-init cycle edges (0ce90a1d).** The import graph now includes
   ancestor package edges.
8. **TypeCheckerSession-per-method (99cbdf5d).** One session per run
   instead of per method. 10 GB / 8 min → 31 s.
9. **Trajectory search collision (874c54b3, f76a3b8e).** Five bare
   `ValueError`s escaped the typed-error catch block.

## Efficiency detectors and their codemod paths (2026-11-13)

Five detectors in `_efficiency.py` find runtime work that restates a
fact the structure already holds. None has a synthesized recipe — author
plans using existing operations.

| Detector | PatternId (stable) | What it finds | Codemod path |
|---|---|---|---|
| `callback_threading` | 51 | param forwarded unchanged through 2+ signatures | `project_function_parameter`, `replace_function_signature`, `patch_target` |
| `loop_relookup` | 52 | loop body re-looks up an element the iterator already yielded | `patch_target` on iterator; or `insert_class_member` for a dict |
| `linear_scan_by_name` | 53 | `next(x for x in SEQ if x.attr == key)` | `insert_class_member` for a dict, `patch_target` on the scan |
| `unused_method_parameter` | 54 | method param unused in the body | `factor_exact_method_role` for cross-class proof, `replace_function_signature` |
| `phantom_family` | 55 | same name in 2+ modules, no shared ancestor | `create_mixin` + `add_class_base` + `fold_shared_declarations_into_base` |

**Self-scan (2026-11-13, NRA on NRA, tests included):** 4,919 findings,
37 detectors. Top: `own_type_test` (1,167), `phantom_family` (858),
`unused_method_parameter` (632), `foreign_absence_probe` (514). Read the
`phantom_family` and `unused_method_parameter` counts with the "known
wrong" notes under Performance: neither checks what its title says.
`_base.py` at 7,779 lines with 200 classes is AGENT-4 textbook.

**The `_base.py` god file (7,779 lines, 201 classes):** 97 candidate
dataclasses and 41 concrete detectors live in the "base" module. The
base should hold only the base classes (`IssueDetector`,
`PerModuleIssueDetector`, `CrossModuleCandidateDetector`). The 138
non-base declarations should move to their own modules via
`extract_symbol_closure_to_new_module`. This is a `package_move`
concept target, the single highest-leverage structural change in NRA.

**Cross-class analysis for detector #4:** `factor_exact_method_role` and
`ExactMethodRoleComponent` already group exact methods across class
cohorts. The unused-parameter check is a special case of that traversal.
Extending it avoids building a second cross-class walker.

## Performance: measured numbers and the two rules that keep them

Cold CLI scan of NRA's own package (347 files, fresh `--cache-dir`,
4 cores, `--json --json-payload loop`), 2026-11-14, one run each:

| Code | Wall | Findings | Scan mode |
|---|---|---|---|
| main (`1832c42`) | 74.8s | 1,164 | `exact_compact_global` |
| this branch, one detector keeping syntax trees | 84.1s | 2,434 | full |
| this branch, that detector on a compact family | 73.2s | 2,413 | `exact_compact_global` |

A warm scan is under a second. These are single runs; differences under
a few seconds are noise.

**Rule 1: a detector that needs other modules reads a compact family.**
`DetectorTypePartition(default_detector_types_for_analysis()).compact_global_complete`
must stay True. One context detector that keeps syntax trees (a
`CrossModuleCollectorCandidateDetector`) switches every scan off the
compact path: the JSON payload loses `scan_status`, three CLI tests
fail, and the scan slows down. Collect per module with a
`CollectedFamily` (a `RestatingSiteFamily` when the items are sites)
and build the detector on `CompactModuleProjectionDetectorMixin`.
`PhantomFamilyDetector` and `PublicDefinitionSiteFamily` in
`detectors/_efficiency.py` are the worked example.

**Rule 2: something every family needs from a parse is a
`ParsedModuleFact`, or a view on `ModuleSyntaxIndex`.** A module-level
generator like `bound_classes(parsed_module)` runs again in every family
that calls it. `parsed_module.fact(Kind)` derives it once per parse.
Before writing a tree walk, look at what the index already holds:
`scopes` (every class and def with its dotted name),
`declarations(type)`, `lexical_symbol`, `evaluating_symbol` (who
evaluates a node: a def's header belongs to the scope it stands in),
`indexed_nodes_of_type`, `parent_by_node`. Existing facts:
`ModuleScopes`, `ModuleDefinitions`, `ModuleClassReadings`,
`ClassLineBindingSnapshots`, `ModuleImportBindings`.

**Site families come in families.** When two `RestatingSiteFamily`
members share their loop, the loop belongs to a parent and each member
is one declaration: `FalseDistinctionSiteFamily` (`distinction = ...`),
`MemberAnswerSiteFamily` (`answer = ...`).

**A function whose first argument is a class node is a method of
`ClassStatement`.** `module.fact(ModuleClassStatements)` holds one record
per class statement for the whole parse: `statements` (all, `ast.walk`
order), `top_level`, `by_name`, `top_level_by_name`. The record answers
`methods`, `method_named`, `direct_assignments`, `annotated_names`,
`spelled_base_names`, `base_terminal_names`, `spells_abstract`,
`dataclass_decorator`, `spells_dataclass` and `dataclass_flag(name)`,
each read once. `IndexedClass` inherits it; a class the index does not
name is an `UnindexedClassStatement`. Iterate the records, never
`module.module.body` filtered by `isinstance(node, ast.ClassDef)`, and
pass the record, never the node. The `spells_*` answers read names; the
compact index's `is_abstract` and `dataclass_declaration` are the
binding-checked ones (they agree on all 3,938 indexed classes of NRA's
own tree, 2026-11-14).

**Deleting what nothing reaches.** `erase_dead_compatibility` deletes a
target and guards against its return; `remove_import_bindings` drops
imports a module no longer reads. Neither proves the target is
unreached. The certificate used on 2026-11-14 (1,027 lines out of
`detectors/`): the name is written nowhere outside declarations that are
themselves unreached, counting tests, docs, string literals and every
sibling repository, and no coverage run of the scans or the suite
executes the body. Classes with a registering metaclass or an
`__init_subclass__` hook above them, dunders and visitor hooks are never
candidates. Drop the unread imports and search again: an import nothing
reads keeps its source declaration looking used.

**A permitted overlap verdict is a lead, not a merge.** Before composing
a pair, read where the copying class is built. If a block field is
filled with something other than the source's own value, the two
classes use one name for two facts and stay apart (paper1 §2: identical
fields today do not justify merging declarations). Two permitted
verdicts on `detectors/_base.py` failed that read on 2026-11-14:
`DataclassNamespaceCliMirrorCandidate.field_names` holds only the
mirrored fields (`_systemic.py:759`), and
`ResidualClosedAxisBranchingCandidate.case_names` holds only the cases
shared with a family (`_base.py:4277`). One passed and was composed:
`ManualKeyedRecordTableClassCandidate` copied all three fields from its
`ManualRecordRegistrationShape`.

**Known wrong, not yet fixed:**
- `phantom_family` says "no shared declaring ancestor" and checks no
  ancestry. It reports every public name defined in two modules: 765
  findings on NRA's package. A real check joins the sites with
  `MemberFamilies.family` in `detectors/_relocations.py`.
- `unused_method_parameter` is titled "ignored in every override" and
  looks at one method in one module.
- Legacy-path erasure does not count a class or function referenced by
  bare name without a call.
- `ClassOverlapVerdicts.existing_owner` (`class_overlap.py`) takes any
  class with subclasses whose declarations *include* the block as its
  owner, even when it declares more. For three records sharing
  `file_path, lineno, symbol` it named one sibling the owner and told
  the others to compose it, extra fields and all, so `create_mixin` was
  never on offer ("no extract-mixin NewMixinOwner"). Composing an owner
  that declares more than the block supplies members the composer does
  not have, which is unsound ancestry (paper1 §4: ancestry may supply
  only required pairs). Until that is fixed, author the mixin in two
  stages: `insert_before_target` a class holding exactly the shared
  fields, then `compose_existing_class` each record onto it; add methods
  to the parent in a later stage. A parent with one extra member is no
  longer contained in the records, and the compose verdict disappears.
