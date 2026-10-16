# String dispatch leads → whole-migration plan, not a one-arm rewrite

`StringLiteralDispatchOwnershipLeadDetector` is a **heuristic source lead**. It
uses NRA's existing `ModuleSyntaxIndex` and literal-case matcher, retains the
original source-spelled `if`/`elif` order, every chain-local `else` statement,
and trailing sibling rows. It lists unmatched guards—including a root guard—and
nested `else: if` as OPEN. It detects async methods but
has **no `FindingRecipeEvaluator`**: neither the domain relation nor an async
body move is certified. `else: if` is not silently treated as source `elif`.
`GuardedStringCaseOwnershipLeadDetector` separately surfaces **selected sites**
with exactly one direct string-equality conjunct under an `and` guard in one
lexical source owner, potentially across distinct `if` roots. It preserves
original case positions, root lines and full source tests, including the order
of extra conjuncts. It does **not** inventory `kind in {...}`, disjunctions,
nested/unsupported conditions or every same-axis test. Short-circuit
reachability, priority, effects and binding stay OPEN; scan original source
before claiming closure. Both leads use the distinct
`SOURCE_BACKED_DISPATCH_LEAD` pattern, whose rendered required relation states
that membership, behavior and migration remain OPEN—**not** the certified
`CLOSED_FAMILY_DISPATCH` relation. They are heuristic, have no recipe, and
cannot prove their cases constitute one domain family. Match patterns and indirect
case recovery still need investigation.

For example, a selected read-only `agent_comms.runtime.RuntimeServer.handle`
source snapshot has simple action comparisons surrounding an
`action in {'goal_snapshot', 'edit_goal', 'update_goal'}` guard. A compact
all-equality observation rejects the whole ladder at this guard; the new lead
retains the unmatched guard and subsequent `set_goal` branch as source evidence. The
latter is **not** proof that those actions belong to one admitted domain ABC,
share framing/error behavior, or can be replaced without changing order. Pin
the actual source revision and re-examine all callers before using this lead.
Do not replay the teaching fixture against that service. An independent
read-only ACP source snapshot (the operative code later committed on the
separate branch at `55268d7`, not a merged PR) contains `reply_targets and kind == 'chunk'`,
`reply_targets and kind == 'committed_progress'`, later `kind == 'tool_end'`
and `kind == 'done'` decisions. The guarded lead lists the exact source roots;
the fact that these are event kinds, their right behavior owners, publication
order and reply obligations come from the task contract and further source
investigation—not from matching the word `kind`. This is a candidate for
behavior-bearing event subclasses under a public ABC with common stream logic
inherited, **not** a license to edit the still-stabilizing ACP worktree.

## Fill the closure before authoring a plan

| Ownership/behavior question | What the lead supplies | What the agent must establish |
|---|---|---|
| Case vocabulary and priority | Original positions of simple Eq arms, unmatched guards, chain-local else and trailing siblings | Supplied rule admitting each required/forbidden case–consumer pair; duplicate, unknown, overlapping membership, alias and priority behavior |
| Case implementation | Source location and owning lexical class/function | Actual reads, Calls and receiver binding from existing NRA lexical/product-flow authorities; validation, effects, exceptions and async cancellation/order |
| Common algorithm | Repeated branch-local work is a search lead | Which framing/serialization/write/drain algorithm is actually invariant, and which adapters or case-specific responses remain independently owned |
| Derived lookup | Current case spellings | Registration/discovery/import order, collision policy, unknown value behavior and any public API/wire compatibility |
| End state | No automatic recipe for this finding | Declarations, bases/MRO, imports, all migrated callers, derived projections, old writable roster/branches removed, guards and tests |

For an **admitted behavior-bearing family**, the endpoint requires concrete
case-owned subclasses of a public behavior ABC and inherited shared
implementation (with multiple inheritance only for independently admitted
crossing capabilities). Source syntax alone never adjudicates that ownership.
Use an existing domain family where it owns the behavior; an enum may remain
only for a justified separate value-only boundary role. Do not convert an
unsupported guard into an implicit default or drop unmatched branches to
satisfy a plan preflight.

Contrast the reviewed primary sources, rather than counting every string or
enum as a defect. Toad PR #47 already declares `PreparationWork(ABC)` with
inherited `ThreadWork`/`RendererWork` execution, `ContentAddressedWork` identity
and a `RenderPreparation(ContentAddressedWork, RendererWork)` composition of
independent policies; its `WorkLane` is a value for admission, not a central
behavioral branch. Toad PR #58's package-owned MCP decisions and UI/PTY/ACP
projections have different authorities; a shared screen-level switch is not
proof that one subclass can issue approvals. Agent-comms PR #17's sealed wake,
response obligation and claim records have independent receipt roles and no
supported common behavioral ABC. ACP's message-event dispatch is a *different*
behavior-bearing candidate: distinguish it from those value/receipt boundaries
and from unrelated tool/goal/usage event roles before proposing its class family.

## Executable, exact-fixture planning exercise

[The action example](../examples/action_batch.py) and
[its tests](../examples/test_action_batch.py) connect this new source lead to
an independently authored whole migration. The example is **not** the live
agent-comms source and is not an automatically extracted rewrite.

1. `SourceModule` parses the exact reviewed BEFORE fixture. The heuristic
   string-ladder detector reports two source-spelled arms, their order and
   chain-local else. The dedicated
   [`tests/test_string_dispatch_ownership_lead.py`](../../../tests/test_string_dispatch_ownership_lead.py)
   separately records ACP-shaped conjunctions and distinct decision roots
   without pretending to resolve their effects or semantics. The tests assert
   `FindingRecipeEvaluator.for_finding(lead) is None`.
2. A separately authored, source-bound `ActionRehearsal` invokes the existing
   NRA `CodemodPlanSequence`: ensure imports (two stages), insert public
   Request/SetGoal/ClearGoal declarations, exact-patch `Rpc.handle`, and run a
   terminal scoped forbidden-dispatch guard (five stages total).
3. Examine each existing `stage_report.document_simulation`, the source
   indexes before/after, final snapshot and guard report. A clean stage means
   its configured preflights/guards passed, **not** behavioral equivalence or
   complete source/context analysis. The rehearsal rejects a changed whole
   fixture snapshot before delegating to the underlying DSL; exporting its raw
   sequence would lose that private source-bound admission.
4. Run before/after protocol and failure-order tests, reject duplicate keys,
   test stale source, and add a `Ping` subclass without editing dispatch or a
   second roster. The new-case edit sites are the maintenance experiment.

For an actual multi-module migration, use `CodemodPlanSequence` over the full
source context. Select/import the existing authority, move members or introduce
an authored declaration as needed, migrate *every admitted consumer* with
supported target/call operations, derive legitimate projections, then delete
obsolete assignments/imports **after** residual-use and initializer-effect
checks. Record source revision, rule/authority for each pair, original unmatched
rows, stage target/cardinality, preflight, before/after binding, architecture
guard scope, and behavioral tests. Current NRA operations do not make arbitrary
async body movement or source-only ownership choice automatic.

Treat source discovery, domain admission, plan replay and equivalence as four
separate gates. If any required witness remains OPEN, publish an exploratory
trajectory and its precise blocker rather than a certified repair.
