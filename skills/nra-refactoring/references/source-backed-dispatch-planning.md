# String dispatch leads → whole-migration plan, not a one-arm rewrite

`StringLiteralDispatchOwnershipLeadDetector` is a **heuristic source lead**. It
uses NRA's existing `ModuleSyntaxIndex` and literal-case matcher. It keeps the
original source-spelled `if`/`elif` order, every `else` statement in the chain,
and trailing sibling rows. Unmatched guards (a root guard included) and nested
`else: if` show up as OPEN. It detects async methods but has **no
`FindingRecipeEvaluator`**: nothing certifies the domain relation or an async
body move. `else: if` is not quietly treated as a source `elif`.

`GuardedStringCaseOwnershipLeadDetector` is separate. It surfaces **selected
sites** with exactly one direct string-equality conjunct under an `and` guard
in one lexical source owner, possibly spread across different `if` roots. It
keeps original case positions, root lines and full source tests, including the
order of extra conjuncts. It does **not** list `kind in {...}`, disjunctions,
nested or unsupported conditions, or every same-axis test. Short-circuit
reachability, priority, effects and binding stay OPEN, so scan the original
source before you claim you've covered everything.

Both leads use the distinct `SOURCE_BACKED_DISPATCH_LEAD` pattern. Its rendered
required relation says that membership, behavior and migration remain OPEN. It
is **not** the certified `CLOSED_FAMILY_DISPATCH` relation. Both are heuristic,
have no recipe, and can't show that their cases form one domain family. Match
patterns and indirect case recovery still need investigation.

Example: a selected read-only `agent_comms.runtime.RuntimeServer.handle` source
snapshot has simple action comparisons on both sides of an
`action in {'goal_snapshot', 'edit_goal', 'update_goal'}` guard. A compact
all-equality check throws out the whole ladder at that guard. The new lead
keeps the unmatched guard and the later `set_goal` branch as source evidence.
That evidence does **not** show those actions belong to one ABC in the task
rules, share framing and error behavior, or can be swapped out without
changing order. Pin the real source revision and recheck all callers before
using this lead. Don't replay the teaching fixture against that service.

A separate read-only ACP source snapshot (the working code later committed on
a separate branch at `55268d7`, not a merged PR) has
`reply_targets and kind == 'chunk'`,
`reply_targets and kind == 'committed_progress'`, then later `kind == 'tool_end'`
and `kind == 'done'` decisions. The guarded lead lists the exact source roots.
That these are event kinds, who should own each behavior, publication order and
reply obligations all come from the task contract and more source
investigation, not from matching the word `kind`. It's a candidate for
behavior-bearing event subclasses under a public ABC with the common stream
logic inherited. It is **not** permission to edit the still-moving ACP worktree.

## Fill in the closure before writing a plan

| Ownership/behavior question | What the lead gives you | What you have to establish |
|---|---|---|
| Case vocabulary and priority | Original positions of simple Eq arms, unmatched guards, chain-local else and trailing siblings | The task rule that allows or forbids each case–consumer pair; how duplicates, unknowns, overlapping membership, aliases and priority behave |
| Case implementation | Source location and the lexical class/function that owns it | Actual reads, Calls and receiver binding from NRA's lexical/product-flow tools; validation, effects, exceptions and async cancellation/order |
| Common algorithm | Repeated branch-local work is a place to look | Which framing/serialization/write/drain algorithm is really the same everywhere, and which adapters or case-specific responses stay independently owned |
| Derived lookup | Current case spellings | Registration/discovery/import order, collision policy, unknown-value behavior and any public API/wire compatibility |
| End state | No automatic recipe for this finding | Declarations, bases/MRO, imports, every migrated caller, derived projections, the old writable roster/branches removed, guards and tests |

For a **family you've decided is behavior-bearing**, the end state is concrete
case-owned subclasses of a public behavior ABC, with shared implementation
inherited. Use multiple inheritance only for crossing capabilities that are
independently justified. Source syntax alone never settles who owns what. Use
an existing domain family where it already owns the behavior. An enum can stay
only where there's a real, separate value-only boundary role. Don't turn an
unsupported guard into an implicit default, and don't drop unmatched branches
to get past a plan preflight.

Compare against the reviewed primary sources instead of counting every string
or enum as a defect:

- Toad PR #47 already declares `PreparationWork(ABC)` with inherited
  `ThreadWork`/`RendererWork` execution, `ContentAddressedWork` identity, and a
  `RenderPreparation(ContentAddressedWork, RendererWork)` that composes
  independent policies. Its `WorkLane` is a value for admission, not a central
  behavior switch.
- Toad PR #58's package-owned MCP decisions and its UI/PTY/ACP projections have
  different owners. A shared screen-level switch doesn't mean one subclass can
  issue approvals.
- Agent-comms PR #17's sealed wake, response obligation and claim records have
  independent receipt roles, and no supported common behavior ABC.
- ACP's message-event dispatch is a *different*, behavior-bearing candidate.
  Separate it from those value/receipt boundaries and from the unrelated
  tool/goal/usage event roles before you propose its class family.

## Executable, exact-fixture planning exercise

[The action example](../examples/action_batch.py) and
[its tests](../examples/test_action_batch.py) connect this source lead to a
whole migration written separately. The example is **not** the live
agent-comms source and is not an automatically extracted rewrite.

1. `SourceModule` parses the exact reviewed BEFORE fixture. The heuristic
   string-ladder detector reports two source-spelled arms, their order and the
   chain-local else. The dedicated
   [`tests/test_string_dispatch_ownership_lead.py`](../../../tests/test_string_dispatch_ownership_lead.py)
   separately records ACP-shaped conjunctions and distinct decision roots,
   without pretending to resolve their effects or meaning. The tests assert
   `FindingRecipeEvaluator.for_finding(lead) is None`.
2. A separately written, source-bound `ActionRehearsal` runs the existing NRA
   `CodemodPlanSequence`: ensure imports (two stages), insert public
   Request/SetGoal/ClearGoal declarations, exact-patch `Rpc.handle`, and run a
   terminal scoped forbidden-dispatch guard (five stages total).
3. Look at each existing `stage_report.document_simulation`, the source indexes
   before and after, the final snapshot and the guard report. A clean stage
   means its configured preflights and guards passed. It does **not** mean
   behavior is equivalent or that source and context were fully analyzed. The
   rehearsal rejects a changed whole-fixture snapshot before it reaches the
   underlying DSL. Exporting its raw sequence would lose that private
   source-bound check.
4. Run before/after protocol and failure-order tests, reject duplicate keys,
   test stale source, and add a `Ping` subclass without editing dispatch or a
   second roster. The new-case edit sites are the maintenance experiment.

For a real multi-module migration, use `CodemodPlanSequence` over the full
source context:

- Select or import the existing owner, move members or add a hand-written
  declaration as needed.
- Migrate *every consumer you've accepted* with supported target/call operations.
- Derive the legitimate projections.
- Delete dead assignments and imports **after** checking residual uses and
  initializer effects.

Record the source revision, the rule that backs each pair, the original
unmatched rows, each stage's target/cardinality, preflight, before/after
binding, architecture guard scope, and behavioral tests. Current NRA
operations don't make arbitrary async body movement or source-only ownership
choices automatic.

Keep four gates separate: finding the source, deciding the domain rule,
replaying the plan, and proving equivalence. If a required witness is still
OPEN, publish an exploratory trajectory and say exactly what's blocking it.
Don't call it a certified repair.
