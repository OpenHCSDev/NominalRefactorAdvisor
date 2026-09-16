# Composing high-leverage NRA plans

## Choose the appropriate stage boundary

`CodemodPlanSequence.from_operations(...)` makes each operation a separate
projected stage. Later operations resolve targets against earlier output.
Use it when a later operation needs a newly moved declaration, altered signature,
new base or changed binding.

Use `CodemodPlanDocument` for a cohesive group that must resolve against the same
snapshot. Compose documents and sequences with `CodemodPlanSequence.compose(...)`.
Do not assume two kinds of batching have identical declaration-time effects.

A concrete trap: inserting a method and then an alias with two separately
reindexed `InsertClassMemberOperation` stages can put the alias above the method.
Put the dependent insertions in one document, method first; or anchor the alias
after an existing method. A parse-clean simulation cannot detect every incorrect
class-body evaluation. The maintained example is
`docs/examples/cohesive_class_members.py` and its explanation in
`docs/source/api/getting_started.rst`.

## Author the migration closure

A useful ownership trajectory often has this shape, adapted to the actual source:

1. Reuse or declare the destination authority and introduce needed imports.
2. Establish inheritance and move existing members without copying their bodies.
3. Retain the richer context/evidence object and project old reads through it.
4. Change signatures and declaration-resolved callers with explicit cardinality.
5. Remove obsolete assignments, forwarding hooks and imports after reviewing
   their remaining uses and evaluation effects.
6. Inspect projected global findings and extend the plan where the new state
   exposes a further ownership collapse.

Do not add all six phases when the change only needs two. Moving a field does
not automatically migrate its users. Changing a signature does not automatically
change callers. Deleting an assignment removes its evaluation too. Bound and
unbound method calls have different receiver requirements; inspect actual call
lookup, not just the common callee name.

Use these maintained examples as starting points, after checking their baseline
and targets; do not run historical recipes blindly on current source:

- `docs/examples/renderer_refactor.py`: ancestor extraction, parameter projection,
  declared-call migration and sequence composition.
- `docs/examples/lexical_binding_authority_refactor.py`: moving a dependency closure
  and separating identities across modules.
- `docs/examples/retain_resolution_refactor.py`: retaining an authority rather
  than mirroring one of its fields.
- `docs/examples/assignment_projection_refactor.py`: inheritance plus field collapse.

## Iterate in projected source without writing every stage

With the current CLI and an authored JSON plan, preview the sequence and its
next-state evidence:

```bash
python -m nominal_refactor_advisor path/to/package \
  --context-root path/to/context \
  --codemod-plan plan.json --codemod-simulate \
  --codemod-project-findings --codemod-project-source-index --json
```

For Python recipes, the core in-memory loop is:

```python
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

original = CodemodSourceSnapshot.from_source_mapping(sources)
current = original
for builder in builders:
    result = builder(current).simulate(current)
    if not result.is_clean:
        raise RuntimeError(f"Rejected projected plan: {builder.__name__}")
    current = result.final_snapshot
```

Here `sources` must cover the supplied proof context, not just edited files;
`builders` are authored functions returning current public plan objects. Preserve
the recipes and stage boundaries as a composed plan for final simulation and
transactional application, rather than manually copying `current` onto disk.

Inspect the rejected report to identify missing proof or ambiguous selection.
Narrow a selector only when actual source evidence supports it. Do not increase
cardinality, weaken guards or insert fallback behavior to make a plan pass.

Use projected findings as opportunities, not orders. Some findings have no recipe
because source establishes repetition but not ownership. Synthesized continuations
still require the supported planning horizon and the practitioner's semantic
review. Exhaustive exploration is bounded by the selected goals, source scope,
detector set and state/branch limits; it is not completeness over arbitrary futures.

## What the bootstrapping evidence actually demonstrates

The September 2026 NRA work composed eight builders into a 40-stage replay
across six production modules. The replay preserved its original snapshot and
matched the final formatted production sources. Its value was batching dependent
transformations with intermediate source validation and one combined destination.
Some stages contained authored bodies: this was not an automatic equivalence
proof or an automatically chosen globally correct trajectory.

The detailed historical record is
`docs/plans/native_behavior_proof_pause_20260915.md`, with the five recipes linked
by `docs/plans/native_behavior_proof_progress_20260915.md`. Treat those documents
as a dated case study, not the current API or current CI status.

Do not infer a token/tool-call savings ratio from stage count or final diff size.
Manual edits can also be batched. If measuring leverage, record actual authoring,
application, test cycles and cumulative work against a comparable baseline.
