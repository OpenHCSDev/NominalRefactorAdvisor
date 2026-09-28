# Round2 R1 implementation receipt

Source scope: agent-comms [PR227](https://github.com/OpenHCSDev/agent-comms/pull/227),
`docs/refactor/round2/R0-R1-stop-the-inflow.md` (R1.a/b/c), following
00-RULES, 01-INDEX, 02-SHARED-ABSTRACTIONS, 03-COORDINATION and the NRA skill.
Implementation: [draft NRA PR9](https://github.com/OpenHCSDev/NominalRefactorAdvisor/pull/9).
Owner: refactor-r1 / Nietzsche. Worktree: `~/wt/nra-refactor2-r1-20260928`.
Base: NRA `1119ca6`; tested implementation: `8c71ebd`.

## Scope and closure

| Part | Owner and consumers | Evidence |
|---|---|---|
| R1.a | `semantic_descent.py`: function/subject mapping-read projection, existing class/schema join and SemanticMirrorWithoutDescentDetector | Raw consumers detected; named/positional constructors, imported aliases and classmethod `cls` decode sites descend. Swapped fields, unrelated constructors and shadowed constructor names do not suppress reads. |
| R1.b | UnmodeledRecordShapeDetector on the same compact semantic projection and class-index context | At least three distinct keys, grouped by exact key set; known schema matches, smaller/dynamic reads and distinct subjects are excluded. |
| R1.c | RedundantTypeCheckDetector; `record_checks.py` retains lexical check evidence in the existing semantic module projection | Annotated parameters, forward references, self, inherited fields and cross-module owners; negative cases include Any/unions, untyped/rebound subjects, unknown attributes and shadowed builtins. |
| Skill | Repository-managed `skills/nra-refactoring/SKILL.md` decision receipt | Requests bypassed schemas, unmodeled shapes and source/declaration check pairs, including the raw full-JSON command. Installed skill unchanged. |

The initial AST-only R1.c path was **deleted** after it broke compact coverage.
All three detector consumers now use the existing semantic projection/class join.
The only CompactSemanticModuleProjection producer and its focused-context
projection were updated; its new type-check evidence is required, not optional.
There is no separate detector registry, old detector implementation, alias,
compatibility reader, or alternate analysis store. IDs derive from class names.

Derived NRA graph/cache data is the only persisted format affected. Its schema
version advances and implementation dependencies include `record_checks.py`;
old cache entries are invalidated. No durable history, Comms runtime stores,
main checkout, installed package or provider was changed by this implementation.

Relative to origin/main: **production −18 / +657 lines**, **tests −0 / +230 lines
(5 behavioral/guard tests)**, skill +9 lines. Net growth is required because R1
adds three detection capabilities; the existing compact join/cache contract was
hoisted into one shared base instead of copied.

PR8 owns dispatch leads in `_runtime.py`, `patterns.py` and its skill planning
reference. Those source files are untouched. The skill overlap was coordinated
in [PR8's comment](https://github.com/OpenHCSDev/NominalRefactorAdvisor/pull/8#issuecomment-5872423314);
the changes occupy separate sections. Both old NRA wire owners were verified STOPPED.

## Local verification

**185 tests passed**, sequentially, with pytest automatic parallelism disabled:

- 48: new R1 fixtures, cold/warm scan coverage, exact cache coverage and parallel
  semantic preparation; 55.48 seconds, command bounded at 60 seconds.
- 137: semantic descent, repository architecture and detector registry imports;
  87.76 seconds, command bounded at 165 seconds.

The earlier cache test attempt failed 8 cases; the implementation was corrected
and all 48 cases passed. No assertion was weakened. `git diff --check` is clean.
CI was not a gate. The complete CLI scan below is actual package analysis,
not a mocked runtime result; no live installation was requested.

## Complete Comms calibration

Snapshot: Comms `a7346153c56d693ecf7f0b8106cd274022ca0fba`, current remote main when
captured; **192 production Python files**. Same snapshot, full context, one parser
and analysis worker, separate cold caches, sequential runs. Full CLI wall time:
**44.11s baseline → 50.06s candidate (+13.49%, within the 25% bound)**.
Internal parse/analysis/index time: 30.021s → 33.484s. Peak candidate RSS: 679,776 KiB.

| Measurement | Plan prototype | Current baseline | R1 |
|---|---:|---:|---:|
| Raw findings | 6 | 12 | 180 |
| Semantic mirror findings | not separated | 11 | 25 (14 additional mapping-read relationships) |
| Unmodeled key-set groups | not separated | 0 | 93 |
| Declared attribute checks | not separated | 0 | 61 |
| Existing repeated builder finding | not separated | 1 | 1 |
| Mapping-read function/subject projections | not measured | 0 | 188 |

The prototype's **321 SQLite column reads in 32 modules** are individual access
sites on a narrower surface. The current package contains 748 literal subscript
loads plus 355 literal `.get` calls, grouped before reporting; 93 is a count of
unmodeled **key sets**, not accesses. The prototype's six findings had no preserved
machine receipt and used the older audited Comms head; this measurement uses the
current head and complete default roster. We do not invent an exact attribution
for that historical six-to-twelve difference.

Final coverage: **81/81 registered detectors, zero omissions**. The warm CLI
receipt reports `complete: true`, `mode: exact_cache`, authenticated against the
same complete roster. The initial CLI attempt hit the default 20-second budget;
complete scans explicitly used 150 seconds and exited successfully.

Reproduce from the respective NRA checkout after exporting the Comms revision:

```sh
XDG_CACHE_HOME="$PWD/.owned-cold-cache" python -m nominal_refactor_advisor \
  /path/to/comms/src/agent_comms --json --raw-findings --json-payload full \
  --parse-workers 1 --analysis-workers 1 --scan-budget-seconds 150
```

Receipts: `evidence/r1/calibration.json`, compressed original full CLI payloads,
`complete-status.json`, and `local-tests.log`. Disposable source snapshots,
pytest roots and cold caches were removed after the processes finished.

## Interpretation limits

These findings identify source shapes and declared contracts, not business
meaning or proof that deleting a check preserves runtime behavior. Schema
matching uses the existing standard-dataclass authorities. Exact `type()` tests
may deliberately reject subclasses; Python does not enforce annotations.
Indirect materializers, dynamic unpacking and alias/dataflow-heavy decode paths
can remain investigation candidates. Imported aliases and direct named,
positional and `cls` constructors are covered explicitly. No automatic rewrite,
whole-repository zero-debt claim, merge or live installation is asserted.

## Precise changed implementation files

- `nominal_refactor_advisor/semantic_descent.py`
- `nominal_refactor_advisor/record_checks.py`
- `nominal_refactor_advisor/detectors/_semantic_descent.py`
- `nominal_refactor_advisor/detectors/_record_checks.py`
- `nominal_refactor_advisor/detectors/_implementations.py`
- `tests/test_raw_record_detectors.py`
- `skills/nra-refactoring/SKILL.md`
