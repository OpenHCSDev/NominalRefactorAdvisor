# Worked migration: action branches → request-owned behavior

This is an **authored, executable teaching migration** of a frozen fixture, not
an extracted patch for an inspected live service. It demonstrates how to batch a
chosen nominal design using the existing NRA DSL. No live goal service is called.

Read [the fixture and plan](../examples/action_batch.py) and
[its behavioral and rejection tests](../examples/test_action_batch.py).

## 1. Identify the answers and choose their owners

`Rpc.handle` originally reads `action`, selects `set_goal` or `clear_goal`,
validates the selected payload, calls the agent, shapes the result, serializes
JSON plus newline, writes bytes, and awaits drain. These are not all one fact.

| Required answer | Before | Chosen determining owner / derived consumer |
|---|---|---|
| Which command does the external action name denote? | String comparisons in `Rpc.handle` | Explicit subclass `action` declaration → derived lookup at `Request.decode` |
| Which inputs does this command accept? | Branch-local checks | `SetGoal` constructor/`from_payload`; `ClearGoal.from_payload` |
| Which domain operation and result shape apply? | Branch-local invocation and envelope contents | Concrete request's `execute` |
| How is a successful result framed and sent? | Repeated `json.dumps`, newline, encode, write, drain | One inherited `Request.respond` algorithm |
| What happens for unknown actions? | Original `ValueError` fallback | Boundary decoder preserves its message for the fixture input domain |

The public `Request` ABC joins those concrete requests at their actual consumer
contract. It does **not** absorb the independent agent/domain service. Here request
response shaping belongs to the RPC request family; a domain reused across other
transports would need its own adapter boundary. Do not infer that domain classes
should universally know JSON or streams.

The destination `Rpc.handle` is:

```python
async def handle(self, session_id, request, writer):
    command = Request.decode(request)
    await command.respond(self.agent, session_id, writer)
```

`respond` awaits `execute`, then serializes, writes and drains. `SetGoal` owns text
validation and `set_goal` invocation; `ClearGoal` owns `clear_goal` invocation.
Valid text is checked with `strip()` but retained **untrimmed**, as before.
Necessary input validation remains; repeated internal case interpretation goes.

## 2. Batch the entire chosen migration

The example uses five projected NRA stages:

1. Ensure the ABC imports.
2. Ensure the existing `AutoRegisterMeta` import.
3. Insert the authored Request/SetGoal/ClearGoal declarations before `Rpc`.
4. Exact-patch the old `Rpc.handle` to invoke the new public contract.
5. Evaluate a terminal architecture guard scoped to `Rpc.handle`.

The registry is derived from class declarations rather than a second hand-written
list. The fixture explicitly rejects duplicate and inherited-without-redeclaration
action keys during subclass creation. This is a fixture extension policy, not a
universal registration policy: discovery, abstract intermediate classes and dynamic
registry mutation require their own design. Reuse an existing family/registration
mechanism in real code before introducing another.

The shared response algorithm and concrete classes are **authored source** supplied
to existing insertion/patch operations. `DispatchToPolymorphismOperation` does not
support this async method or multi-statement effectful arms; a test checks that it
rejects the target. Five successful stages do not upgrade that proof scope.

## 3. Bind this rehearsal to its actual source, not just an old method

`build_plan` returns a private `ActionRehearsal`, **not** an exportable
`CodemodPlanRoot`. Every `simulate(snapshot)` checks the exact one-file scope and
its complete `CodemodSourceRevision` before delegating to the normal NRA sequence.
This receipt check is needed because an unchanged method can coexist with a new
module statement `Request = None`, breaking the generated consumer. Matching the
old method, or checking a `source` argument only when building the plan, did not
prevent replay against that changed module. The test rejects even an added comment:
this is intentionally an exact reviewed fixture, not a generic migration engine.

The underlying `_authored_sequence` and the returned NRA result's sequence are
**unbound DSL**. Do not export/replay them as source-pinned plans. NRA's normal
`result.apply()` freshness checks compare the simulated original sources to the
physical files; that is a different check from this authored-fixture admission.
No concurrency/ABA/atomicity proof is claimed by the example.

The terminal guard checks only the expressions `action`, `request.get("action")`
and `request["action"]` in the selected method. It is not semantic detection of
all equivalent dispatch, and it does not execute class bodies or metaclasses.

## 4. Run the migration and the maintenance experiment

From the NRA checkout, using an environment with this checkout and its declared
dependencies available:

```bash
PYTHONPATH="$PWD" python -m unittest discover \
  -s skills/nra-refactoring/examples -p 'test_action_batch.py' -v
```

The suite simulates without rewriting the fixture source and executes before and
after with stub agents/streams. Its 12 tests include:

- 90 combinations of decoded ordinary JSON action/text values plus missing fields;
- exact protocol bytes, untrimmed valid text, errors, cancellation and effect order;
- rejection of changed method/module baselines, extra source context and stale apply;
- duplicate/inherited registration-key rejection without overwriting the registry;
- a new `Ping` subclass executing through **unchanged** `Rpc.handle` and
  `Request.respond`, without editing a separate case roster.

The new-case experiment is the maintenance evidence: a case declaration owns its
own action, decoding and execution, while framing and selection consumers derive
from the family. This is not a line-count win; the tiny after fixture is longer.
It removes independently maintained case decisions and repeated common behavior.

Custom Python mappings/equality, registry mutation, discovery/import order,
alternate callers, reflective class inspection, arbitrary input equivalence,
traceback identity and the real service's domain contracts remain outside the
fixture's tested claim. Transferring it requires a new source/contract investigation.

## 5. Transfer the workflow, not this fixture's source strings

For a real case, use original declarations and NRA's existing lexical/product-flow
authorities to trace each arm's reads, calls, exceptions and effects. Reuse any
existing request/domain family. Choose the ownership relation, then compose the
complete declarations/imports/callers/mirror-removal migration against that source.
Prefer supported declaration-preserving operations over copying bodies; explicitly
label remaining authored moves. Test both preservation and one new-case change.
See [batching](batching.md) and [ownership-to-DSL](ownership-to-dsl.md).
