# Composing high-leverage NRA plans

## Pick the right stage boundary

`CodemodPlanSequence.from_operations(...)` makes each operation its own
projected stage. Later operations resolve their targets against earlier
output. Use it when a later operation needs a freshly moved declaration, a
changed signature, a new base or a changed binding.

Use `CodemodPlanDocument` for a group that belongs together and has to resolve
against the same snapshot. Combine documents and sequences with
`CodemodPlanSequence.compose(...)`. The two kinds of batching don't behave the
same at declaration time, so don't treat them as interchangeable.

A real trap: insert a method and then an alias as two separately reindexed
`InsertClassMemberOperation` stages, and the alias can land above the method.
Put the dependent insertions in one document, method first, or anchor the alias
after a method that already exists. A parse-clean simulation won't catch every
class body that evaluates wrong. The maintained example is
`docs/examples/cohesive_class_members.py`, explained in
`docs/source/api/getting_started.rst`.

## Write out the whole migration

A good ownership trajectory usually looks like this, adjusted to the source in
front of you:

1. Reuse or declare the destination owner and add the imports it needs.
2. Set up inheritance and move existing members without copying their bodies.
3. Keep the richer context/evidence object and project the old reads through it.
4. Change signatures and declaration-resolved callers, with explicit cardinality.
5. Remove the dead assignments, forwarding hooks and imports, after checking
   what still uses them and what their evaluation does.
6. Look at the projected global findings, and extend the plan if the new state
   shows another ownership collapse.

Don't add all six phases when the change needs two. Moving a field doesn't
migrate its users. Changing a signature doesn't change its callers. Deleting
an assignment deletes whatever running it did, too. Bound and unbound method
calls need different receivers, so check how the call is really looked up, not
just that the callee has the same name.

Start from these maintained examples, but check their baseline and targets
first. Don't run old recipes blindly against current source:

- `docs/examples/renderer_refactor.py`: ancestor extraction, parameter projection,
  declared-call migration and sequence composition.
- `docs/examples/lexical_binding_authority_refactor.py`: moving a dependency closure
  and separating identities across modules.
- `docs/examples/retain_resolution_refactor.py`: keeping an authority instead of
  mirroring one of its fields.
- `docs/examples/assignment_projection_refactor.py`: inheritance plus field collapse.

## Iterate in projected source without writing every stage

With the current CLI and a hand-written JSON plan, preview the sequence and
the evidence for the next state:

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

`sources` has to cover all the context you're relying on, not just the files
you edit. `builders` are your own functions that return current public plan
objects. Keep the recipes and stage boundaries as one composed plan for the
final simulation and transactional apply. Don't copy `current` onto disk by
hand.

When a plan gets rejected, read the report to see what proof is missing or
which selection is ambiguous. Narrow a selector only when the source backs it
up. Don't raise the cardinality, weaken a guard or bolt on fallback behavior
just to make the plan pass.

Projected findings are leads, not orders. Some findings have no recipe because
the source shows repetition but not who should own it. Synthesized
continuations still need the supported planning horizon and your own review of
what the code means. Exhaustive exploration stops at the goals you picked, the
source scope, the detector set and the state/branch limits. It does not cover
arbitrary futures.

## What the bootstrapping run really shows

The September 2026 NRA work composed eight builders into a 40-stage replay
across six production modules. The replay kept its original snapshot and
matched the final formatted production sources. What you get from it is
dependent transformations batched together, with source checked between
stages, landing in one combined result. Some stages had hand-written bodies.
It is not an automatic equivalence proof, and the tool did not pick a
globally correct trajectory by itself.

The detailed history is in `docs/plans/native_behavior_proof_pause_20260915.md`,
and `docs/plans/native_behavior_proof_progress_20260915.md` links the five
recipes. Treat both as a dated case study, not the current API or current CI
status.

Don't derive a token or tool-call savings ratio from stage count or final diff
size. You can batch manual edits too. To measure leverage, record real
authoring time, application, test cycles and total work against a baseline you
can compare to.
