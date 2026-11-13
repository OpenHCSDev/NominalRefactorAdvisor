# Worked migration: action branches → request-owned behavior

This is a teaching migration I wrote by hand against a frozen fixture. It is
not a patch pulled from a live service, and nothing here calls a live goal
service. It shows how to batch a nominal design you've already picked, using
the existing NRA DSL.

Read [the fixture and plan](../examples/action_batch.py) and
[its behavior and rejection tests](../examples/test_action_batch.py).

## 1. Find the answers and pick who owns them

`Rpc.handle` originally reads `action`, picks `set_goal` or `clear_goal`,
validates that branch's payload, calls the agent, shapes the result, dumps
JSON plus a newline, writes bytes, and awaits drain. That's several different
facts jammed into one method.

| Required answer | Before | Who owns it now / who derives from it |
|---|---|---|
| Which command does the external action name mean? | String comparisons in `Rpc.handle` | Explicit subclass `action` declaration → lookup derived at `Request.decode` |
| Which inputs does this command accept? | Checks inside each branch | `SetGoal` constructor/`from_payload`; `ClearGoal.from_payload` |
| Which domain operation runs, and what shape comes back? | Branch-local call and envelope contents | The concrete request's `execute` |
| How is a successful result framed and sent? | `json.dumps`, newline, encode, write, drain, copy-pasted | One inherited `Request.respond` algorithm |
| What happens on an unknown action? | The original `ValueError` fallback | The boundary decoder keeps its message for the fixture's inputs |

The public `Request` ABC ties the concrete requests together where they
actually get consumed. It does **not** swallow the agent/domain service, which
stays independent. Here, response shaping belongs to the RPC request family.
If the same domain were reused over other transports, it would need its own
adapter boundary. Don't walk away thinking domain classes should always know
about JSON or streams.

The new `Rpc.handle`:

```python
async def handle(self, session_id, request, writer):
    command = Request.decode(request)
    await command.respond(self.agent, session_id, writer)
```

`respond` awaits `execute`, then serializes, writes and drains. `SetGoal` owns
text validation and the `set_goal` call; `ClearGoal` owns the `clear_goal`
call. Valid text is checked with `strip()` but kept **untrimmed**, same as
before. Input validation that has to happen still happens. The repeated
internal case-switching goes away.

## 2. Batch the whole migration

The example uses five projected NRA stages:

1. Ensure the ABC imports.
2. Ensure the existing `AutoRegisterMeta` import.
3. Insert the hand-written Request/SetGoal/ClearGoal declarations before `Rpc`.
4. Exact-patch the old `Rpc.handle` to call the new public contract.
5. Run a terminal architecture guard scoped to `Rpc.handle`.

The registry comes from the class declarations, not from a second hand-kept
list. The fixture rejects duplicate action keys, and action keys inherited
without being redeclared, at subclass creation. That's this fixture's policy,
not a universal registration rule. Discovery, abstract intermediate classes
and runtime registry mutation each need their own design. In real code, reuse
an existing family or registration mechanism before inventing another.

The shared response algorithm and the concrete classes are **hand-written
source** fed to the existing insertion and patch operations.
`DispatchToPolymorphismOperation` doesn't support this async method, and a
test checks that it rejects the target. Five clean stages don't change that.

## 3. Tie the rehearsal to its real source, not just an old method

`build_plan` returns a private `ActionRehearsal`, **not** an exportable
`CodemodPlanRoot`. Every `simulate(snapshot)` checks the exact one-file scope
and the full `CodemodSourceRevision` before handing off to the normal NRA
sequence. You need that check because an untouched method can sit next to a
new module statement `Request = None` and break the generated consumer.
Matching the old method wasn't enough, and neither was checking a `source`
argument only when building the plan: both let the plan replay against the
changed module. The test rejects even an added comment. This is a reviewed
fixture, not a generic migration engine.

The underlying `_authored_sequence`, and the sequence inside the returned NRA
result, are **unbound DSL**. Don't export or replay them as source-pinned
plans. NRA's normal `result.apply()` freshness checks compare the simulated
original sources to the files on disk. That's a different check from this
fixture's source gate. The example makes no claim about concurrency, ABA or
atomicity.

The terminal guard only looks for the expressions `action`,
`request.get("action")` and `request["action"]` in the selected method. It
doesn't detect every equivalent way of dispatching, and it doesn't run class
bodies or metaclasses.

## 4. Run the migration and the maintenance experiment

From the NRA checkout, in an environment that has this checkout and its
declared dependencies:

```bash
PYTHONPATH="$PWD" python -m unittest discover \
  -s skills/nra-refactoring/examples -p 'test_action_batch.py' -v
```

The suite simulates without rewriting the fixture source, and runs before and
after with stub agents and streams. Its 12 tests include:

- 90 combinations of decoded ordinary JSON action/text values plus missing fields;
- exact protocol bytes, untrimmed valid text, errors, cancellation and effect order;
- rejection of changed method or module baselines, extra source context and stale apply;
- duplicate and inherited registration-key rejection, without overwriting the registry;
- a new `Ping` subclass running through **unchanged** `Rpc.handle` and
  `Request.respond`, with no separate case roster to edit.

The new-case experiment is the maintenance evidence. A case declaration owns
its action, decoding and execution, and framing and selection derive from the
family. This isn't a line-count win: the tiny after-fixture is longer. What
you save is independently maintained case decisions and repeated shared
behavior.

The tested claim stops short of: custom Python mappings/equality, registry
mutation, discovery/import order, other callers, reflective class inspection,
arbitrary input equivalence, traceback identity, and the real service's domain
contracts. Moving this to real code means a fresh source and contract
investigation.

## 5. Copy the workflow, not this fixture's source strings

For a real case, start from the original declarations and use NRA's existing
lexical and product-flow tools to trace each arm's reads, calls, exceptions
and effects. Reuse any request/domain family that already exists. Pick the
ownership relation, then build the full migration against that source:
declarations, imports, callers, removal of the old mirrors. Prefer supported
declaration-preserving operations over copying bodies, and say plainly which
moves you wrote by hand. Test both that behavior is preserved and that one
new-case change works. See [batching](batching.md) and
[ownership-to-DSL](ownership-to-dsl.md).
