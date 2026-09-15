# Native-behavior proof continuation

Date: 2026-09-15

## Scope and reconciliation

Work starts from `76fda5db5dccb60bcfbdb062a19336ff5a4b569b` on
`checkpoint/native-proof-integration-20260914`. The preceding native-proof and
performance integration and its 209 historical failures are complete. This
increment addresses only the three assumptions in
`native_behavior_proof_handoff_20260915.md`; it does not reopen that integration
or begin OpenHCS extraction or manuscript work.

## Ownership and proof boundaries

- `NativeUseRequirement` currently authenticates the original read and captured
  declaration, but identity does not prove an invocation or its behavior.
- Actual invocations belong to `CallAuthority`; original definition applications
  and construction belong to the existing source-definition owners. New native
  use admission must consume their execution evidence, not another name table.
- Narrow `type` queries and descriptor wrapping can use supported native
  primitive laws with independently authenticated operands and original cuts.
- `mro_registry_value` must validate its current Python implementation and the
  bound `next`, MRO lookup, mapping operations, and relevant mutation boundaries.
  Arbitrary `Mapping` implementations do not acquire native dictionary behavior.
- `AutoRegisterMeta` uses mutable Python source, ABC initialization, inherited
  configuration, and reachable logging/discovery callbacks. The inert
  compatibility fixture is not construction evidence. Unsupported effects remain
  explicit obligations.
- Environmental noninterference and successful source entry remain distinct
  premises. Default entry loaders will not infer them.
- Source edits retain unchanged local proof owners, not dependency-sensitive
  answers. Native source and mutable dependencies must revalidate when queried.

## Status

The callable-metadata ownership prerequisite is committed and pushed as
`ec541bd2d22d3e3d42097df962a793741b9ddb5f` to
`checkpoint/native-proof-integration-20260914`; the remote SHA matches. Its
exact-commit hosted run
[35005841288](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35005841288)
is in progress as of 18:11 UTC, not yet a passing gate. Its complete local
Python 3.11/3.14 suites, docs, isolated package
builds, installed-wheel mutation/DSL controls, installed public-API self-scan,
and complete production CLI self-scan pass. Complete-package cold/warm/edit
gates pass. Construction/registration remains unproved; see the current metadata
section for exact scope and current-source evidence.

An audit after the registry-hit checkpoint found an unproved type-query operand
cleanup that was incorrectly admitted. The committed correction reuses
original-frame lifetime evidence and passes 142 focused regressions per Python
version, both complete local suites, docs, distribution builds, and an isolated
installed-wheel API self-scan, and complete-package cold/warm/one-edit scans.
See the cleanup audit below. Prior CI
success does not discharge that newly identified obligation.

Preceding operand-cleanup code checkpoint:
`997d85f95466d5f902679d3f221259b15f23748b`, committed and pushed to
`checkpoint/native-proof-integration-20260914`. The complete corrected-source
local gates are recorded in the final checkpoint section below. Its exact-SHA
hosted integration run is
[35002821051](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35002821051),
complete with all seven jobs passing, verified at 18:08 UTC. This gate belongs
to that code checkpoint; it does not validate the later metadata refactor.
The persistent goal remains
active. Generated construction/registration and source-created class keys are
still unproved and remain the next implementation work.

Preceding registry-hit code checkpoint:
`c425cbb20a38031a6cb01bc54a4fb6aa26997ce9`, committed and pushed to
`checkpoint/native-proof-integration-20260914`. Complete local gates for its
registry-hit increment are recorded below. Its hosted integration run is
[34999888194](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/34999888194),
complete with all seven jobs passing, verified at 17:57 UTC. This validates that
checkpoint, not later local changes or completion of generated registration.
The persistent goal remains active;
generated construction/registration and source-created class keys are still
unproved. Work continues on those original-source joins.

Checkpoint `308375890ef8927aafeb95b4eed0a27f50ff95b6` is committed and pushed to
`checkpoint/native-proof-integration-20260914`. Its scoped implementation passes
the frozen-source local gates below. Hosted cross-platform validation passes
all seven jobs (both Python versions on Ubuntu/macOS/Windows, plus docs/wheel):
[Integration Tests 34996343607](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/34996343607).
The following table describes that committed checkpoint, not completion of all
three targets. Registry-hit work continues separately below; generated
registration remains fail-closed.

| Target | Automatically derived evidence now available | Remaining obligation |
| --- | --- | --- |
| MRO registry lookup | Current implementation and exact `next` binding; actual static MRO; exact original scalar-key dictionary contents and mutations; no-hit result | Type-key hits, subscription results, and source-created class operands remain unresolved. |
| `type` / `classmethod` | Exact one-operand type queries and results; original descriptor applications, wrapping, and installation over supported operands | Dynamic three-operand construction, unknown operand metadata, and additional decorator effects remain unresolved. |
| `AutoRegisterMeta` | Exact original metaclass/header; actual selected `type.__prepare__`; fresh prepared namespace; original prepared values/installations; current selected Python constructor source | ABC construction/result identity, configuration inheritance, registration writes, key/hook effects, and their returned entries remain unresolved. This is not generated registration admission. |

The native-use owner now consumes the existing call or definition execution
authority. Captured identity is still a failed obligation unless actual behavior
is proved or an explicit authored invariant is supplied. Environmental entry and
noninterference premises remain explicit in positive controls.

Dependency-sensitive requirements retain their complete source-state authority.
A sequential DSL control exposed that compiler conversion discarded the retained
proof repository; a DSL-applied refactor now uses its existing exact-owner
constructor. A later provider edit blocks admission even when the consumer's
local execution remains retained. Omitting the projected source authority or
supplying foreign parsed owners is rejected.

Native dependencies are matched by the current function's compiler source path,
not its mutable `__module__` label. An adversarial qualification edit cannot hide
a changed provider. Qualification itself requires exact text before formatting,
so mutated metadata cannot execute a string-conversion hook. Source-less native
type dependencies require the compiler's existing static-type immutability law;
heap metaclass construction does not acquire a source-less exemption.

The cached native Python source owner no longer retains mutable definition AST.
Each syntax query uses the existing fresh, exact-code correspondence authority.
An isolation control checks that mutating a returned AST cannot poison later
queries on that same cached owner.

Python 3.14 exposed unrelated sibling code with native slice constants that
marshal format 2 cannot represent. Function-source correspondence now narrows
emissions by the original native first line before exact contents comparison.
That condition is necessary, not sufficient: the full byte contents, constant
alias graph, unique source span, and capture receipt still determine admission.
Unsupported constants in the selected body remain unproved.

Latest focused source/native/descriptor/DSL surface: 246 passed, 3 skipped on
Python 3.14 in 13.89 seconds, eight workers and a 60-second bound. Broad validation
was rerun from frozen source after implementation changes correctly
invalidated in-flight cache and source-revision controls. A prior 165-second
Python 3.14 shard hit its bound after reporting its tests; it is not a green gate.
The test fixtures now check exact retained node/proof owners rather than cache
container identity and supply every source file present in their index.

### Frozen-source full suites

| Runtime / file shard | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 advisor | 795 | 0 | 104.67 |
| Python 3.11 non-advisor `0::2` | 3315 | 28 | 96.25 |
| Python 3.11 non-advisor `1::2` | 3056 | 46 | 120.87 |
| **Python 3.11 total** | **7166** | **74** | |
| Python 3.14 advisor | 795 | 0 | 67.83 |
| Python 3.14 non-advisor `0::4` | 1922 | 3 | 103.03 |
| Python 3.14 non-advisor `1::4` | 1542 | 7 | 71.22 |
| Python 3.14 non-advisor `2::4` | 1415 | 3 | 55.35 |
| Python 3.14 non-advisor `3::8` | 712 | 25 | 57.06 |
| Python 3.14 non-advisor `7::8` | 816 | 0 | 131.54 |
| **Python 3.14 total** | **7202** | **38** | |

All rows pass with eight workers and 165-second bounds. Non-advisor file lists
are sorted `tests/test_*.py`, excluding `test_refactor_advisor.py`; the indicated
Python slice selects a complete disjoint shard. Python 3.11.11 uses installed
metaclass-registry 0.1.4; Python 3.14.6 uses 0.2.1. Proofs follow each actual
dependency implementation; these versions do not share an assumed policy.

```sh
timeout 165 "$PYTHON" -m pytest tests/test_refactor_advisor.py \
  -q -n 8 -p no:cacheprovider --tb=short
timeout 165 "$PYTHON" -c 'from pathlib import Path; import pytest; files=sorted(path.as_posix() for path in Path("tests").glob("test_*.py") if path.name != "test_refactor_advisor.py"); raise SystemExit(pytest.main([*files[0::2],"-q","-n","8","-p","no:cacheprovider","--tb=short"]))'
```

Set `PYTHON` to the interpreter above and change only the documented file slice
for the remaining rows. Use a task-owned `TMPDIR` so cache/fixture cleanup stays
isolated. Python 3.14 reports existing multiprocessing fork and non-string
class-dictionary-key warnings in their respective controls.

Docs and sdist/wheel builds pass. Sphinx reports two existing duplicate-object
warnings for the exact dataclass field factoring/promotion APIs. A production NRA
self-scan completes all 79 detectors with zero omissions and zero findings
(16.772 scan seconds on the final frozen source). The final built wheel also passes a fresh, isolated
Python 3.11 install and production self-analysis outside the repository, with
zero findings. The smoke confirms the installed compiler API and fresh-source
definition property, rather than importing the editable checkout.

### Complete package performance

The final input is a disposable copy of current OpenHCS production Python and
eight external libraries, excluding tests: 1057 discovered files. This differs
from the preceding 1015-file baseline, so the timings are current gates rather
than a same-input speedup claim. No competing test or smoke process was running
for these accepted measurements.

| Scan | Scan seconds | Wall seconds | Cache |
| --- | ---: | ---: | --- |
| Cold | 48.698 | 51.40 | miss |
| Warm | 1.166 | 2.50 | hit |
| One edit | 5.191 | 8.12 | partial |

Every scan covers all 79 detectors with zero omissions and 215 raw findings.
The one edit appends a novel comment to the copied `openhcs/__init__.py`; it does
not alter executable source or original node positions, and is restored after
measurement. The cold scan is bounded at 165 seconds; warm/edit scans at 60.

```sh
timeout 165 env NRA_CACHE_HOME="$TASK_CACHE" "$PYTHON" \
  -m nominal_refactor_advisor "$SOURCE_COPY" \
  --context-root "$SOURCE_COPY" --no-auto-context-root \
  --json --json-payload agent --parse-workers 16 --analysis-workers 16 \
  --scan-budget-seconds 165
```

Use a fresh task-owned cache for cold, then the same cache for warm and the
copied-source edit, changing only both bounds to 60. All three semantic report
projections have SHA-256
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100` after
`jq -S 'del(.timing,.payload_timing) | .scan_status |= del(.mode,.reason)'`.
Completeness and detector/omission counts remain in that projection. Report
equality checks cache consistency, not detector recall or general correctness.

An earlier cold attempt overlapped isolated-wheel validation and was cancelled;
it is not an accepted measurement. Interrupted wheel-smoke attempts likewise
do not count as passing gates; the fresh-cache isolated smoke above is the
accepted result.

### Next proof obligations

1. Derive non-scalar dictionary-key operation evidence through the existing
   namespace/key authority. Do not relax scalar-only namespace admission or run
   arbitrary hash/equality callbacks to obtain a registry hit.
2. Join actual first-hit membership/subscription and returned value identity to
   the existing MRO/source/call authorities, including later key or value
   mutation and projected-provider edits.
3. Follow actual selected metaclass construction, ABC filtering, configuration,
   inherited registry, and registration writes. Existing preparation and
   compatibility checks are prerequisites, not construction/effect proof.

The persistent continuation goal remains active after this checkpoint.

## Validated registry-hit increment

The existing compiler backend now owns an explicit dictionary-key operation
contract. Exact scalar values retain their unchanged value contract; immutable
non-heap ordinary static type identities acquire supported dictionary-key
hash/equality and temporary-release evidence. This does not admit heap classes,
custom metaclasses, compound keys, unknown mapping protocols, or arbitrary
hash/equality callbacks.

Namespace admission, original member inventory, item-write queries, and copies
consume that shared key evidence. Supported actual MRO hits return the original
installed value through the existing call authority; actual MRO order controls
selection independently of dictionary insertion order. A stored `None` is a
value, not absence. Returned scalar contents do not acquire unsupported object
identity evidence.

An authored four-stage DSL plan simulated cleanly and generated the namespace
annotation refinement; its generated patch was applied without re-rendering
unrelated source. Scalar producers and the explicit scalar-value query keep
their narrow annotations. See
`docs/examples/native_dictionary_key_annotations.py`.

Current-source adversarial controls also reject lexical parameter/generator
shadowing: authenticating a module builtin is insufficient when native code
selects a local or closure binding. The selected current function code and
original source roles retain those independent lookup obligations.

### Registry-hit frozen-source validation

The final explicit class-lookup contract also rejects non-class operands;
dictionary-key admission alone cannot supply the class/MRO operation law.

| Runtime / file shard | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 advisor | 795 | 0 | 128.02 |
| Python 3.11 non-advisor `0::4` | 1790 | 12 | 90.56 |
| Python 3.11 non-advisor `1::4` | 1601 | 9 | 46.62 |
| Python 3.11 non-advisor `2::4` | 1574 | 15 | 77.14 |
| Python 3.11 non-advisor `3::4` | 1435 | 38 | 31.10 |
| **Python 3.11 total** | **7195** | **74** | |
| Python 3.14 advisor | 795 | 0 | 86.07 |
| Python 3.14 non-advisor `0::8` | 775 | 3 | 155.72 |
| Python 3.14 non-advisor `4::8` | 1022 | 2 | 69.59 |
| Python 3.14 non-advisor `1::4` | 1606 | 4 | 40.82 |
| Python 3.14 non-advisor `2::4` | 1583 | 6 | 60.28 |
| Python 3.14 non-advisor `3::8` | 659 | 23 | 34.47 |
| Python 3.14 non-advisor `7::8` | 791 | 0 | 17.62 |
| **Python 3.14 total** | **7231** | **38** | |

All rows exit successfully with eight workers and 165-second bounds. File-slice
selection and interpreters are exactly as documented for the first checkpoint;
the table replaces its validation for this increment. A final Python 3.11
focused key/source/effect surface reports 166 passed in 6.22 seconds with a
60-second bound. No authored native-use invariant supplies registry-hit
acceptance.

Final docs and sdist/wheel builds pass, with the same two existing Sphinx
duplicate-API warnings. The final wheel is installed outside the checkout in
`/home/ts/nra-native-behavior-validation-G481Dc/wheel-venv`; installed-package
path checks, static-key/scalar-separation checks, unsupported-key rejection, and
a fresh production self-analysis (zero findings) pass. Its self-analysis uses
eight parse/analysis workers and a 165-second bound. The earlier 60-second
wheel self-analysis timed out and is not a passing gate. It was retried with a
new cache rather than inheriting an interrupted publisher lease. Finished
self/wheel caches (approximately 320 MB) have been removed; reports and built
artifacts are retained.

The complete-package check uses the same 1057-file OpenHCS/eight-library copy
as the preceding checkpoint, excluding tests, with no competing test or smoke
process. It covers all 79 detectors, zero omissions, and 215 raw findings:

| Scan | Scan seconds | Wall seconds | Cache |
| --- | ---: | ---: | --- |
| Cold | 50.751 | 53.55 | miss |
| Warm | 1.086 | 2.38 | hit |
| One edit | 4.444 | 7.27 | partial |

Bounds, worker counts, and commands match the earlier complete-package check.
The copied-source one-edit comment was restored afterward. All three complete
semantic projections match each other and the preceding checkpoint's hash
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`.
This checks report/cache consistency, not detector recall or general proof
correctness. Small timing differences do not establish a performance change.

Hosted results for `3083758` concern only the preceding checkpoint, not these
subsequent edits.

The final production NRA CLI self-scan uses the production package as both
target and explicit context: 79 detectors, zero omissions/findings, 16.730 scan
seconds. An accidental whole-repository context attempt reached its 60-second
bound and is not a passing gate. The accepted command is:

```sh
timeout 60 env NRA_CACHE_HOME="$FRESH_SELF_CACHE" "$PYTHON" \
  -m nominal_refactor_advisor nominal_refactor_advisor \
  --context-root nominal_refactor_advisor --no-auto-context-root \
  --json --json-payload agent --parse-workers 8 --analysis-workers 8 \
  --scan-budget-seconds 60
```

Broad testing exposed a layer error in the first key implementation: initial
primitive-key admission incorrectly required a supported compiler-construction
backend. The shared base authority now composes the existing primitive-content
or native static-MRO evidence. Storage/invocation effects still have separate
fail-closed methods. The four affected unsupported-backend controls are
unchanged and pass; final focused regressions report 168 passed on both Python
3.11 (5.33 seconds) and 3.14 (7.25 seconds), before the final explicit non-class
lookup controls. A timed-out broad shard and earlier red shards do not count as
green gates.

### Generated-registration source map

The inspected dependencies are metaclass-registry 0.1.4 (local Python 3.11) and
0.2.1 (local Python 3.14). They have materially different construction paths:
0.2.1 derives declared/inherited registry configurations and a registry family
before ABC construction, whereas 0.1.4 starts with ABC construction and then
auto-configuration. Native proofs must follow actual source, not a version
label or one copied policy.

Remaining joins, in execution order:

1. Authenticate the original metaclass/base/header and selected construction
   hooks. Current selection alone does not prove native `type.__call__`,
   `__init__`, or the selected constructor's implicit `super()` closure binding.
2. Bind actual prepared inputs into current constructor source. Native/source
   dependencies must include reachable inherited helpers and closure/global
   bindings, not just the top-level `__new__` code identity.
3. Join ABCMeta's current Python forwarding source, native type construction,
   and `_abc_init` to original members/bases and the resulting class namespace.
   Inert member installation does not prove inherited abstractness or ABC writes.
4. Follow actual configuration/key selection and registry writes. Explicit
   dictionary storage does not by itself prove configuration construction,
   inherited policy, extractor irrelevance, or attribute-write effects.
5. Prove irrelevant or account for secondary/discovery and logging callbacks.
   `_auto_configure_registry` itself can log even if final registration logging
   is disabled. Existing authored subprocess controls demonstrate reachable
   handlers that can alter results; never silently assume logging is inert.

Source-created class keys also need symbolic creation-identity evidence joined
to the existing dictionary slot model. The static-key increment is deliberately
narrow; it does not turn current heap classes or projected classes into static
native objects. Generated construction and registration remain unproved while
these joins are missing.

## Constructor class-cell and type-cleanup checkpoint

`NativeClassMroDeclaration.python_constructor` now validates the selected
constructor's actual implicit class cell against its selected MRO owner. An
inherited constructor belongs to that ancestor, not necessarily the invoked
metaclass. The same owner revalidates the current cell on every query; function
identity and code contents do not freeze mutable closure contents.

The supported captured closure has exactly the compiler's implicit `__class__`
role. Other closure roles remain unresolved. Empty or foreign cells are rejected
without invoking target equality, representation, constructors, or callbacks.
This provides initial operand provenance, not a claim that `super`, ABC
construction, native initialization, or registry writes are proved. Generated
registration still fails closed.

Six new controls cover actual inherited/diamond selection, same-function cell
mutation/restoration, empty cells, foreign hooks, unsupported closure roles, and
the original source class-construction query with no authored native-operation
conditions. Focused gates: 72 passed on Python 3.11 in 4.55 seconds and 72 passed
on Python 3.14 in 5.84 seconds, eight workers and a 60-second bound. The initial
positive fixture accidentally captured a callback; it correctly failed the
narrow contract and was replaced by a class-cell-only fixture, without weakening
the production proof. The first broader matrix was stopped after the cleanup
audit below; its partial passes and timed-out shard do not validate the corrected
source.

### Native type-query cleanup audit and correction

An authored original-source control exposed a fail-open obligation:
`type(dataclass(frozen=Ephemeral()))` returned the expected function type but
destroyed the temporary closure and ran `Ephemeral.__del__`. Before correction,
the native-use requirement reported `PROVED` even though that cleanup changed
the control's state. Wrapping the closure in a temporary tuple has the same
effect. Correct result metadata alone did not prove the complete operation.

`NativeTypeQueryCall` now consumes the original activation frame's existing
`require_release_in` lifetime/independent-retention evidence. The returned type
retains its class, not the input object. Unproved temporary release therefore
stays unresolved; native Py_TYPE lookup is not treated as inert cleanup. No
parallel release model or authored invariant supplies acceptance. The authored
execution is a regression control, not automatic proof evidence.

The combined class-cell/type-cleanup/key/native-use regression surface passes
142 tests on Python 3.11 in 20.79 seconds and 142 on Python 3.14 in 22.11 seconds,
eight workers and 60-second bounds. Broader/package/performance gates for this
correction pass locally as recorded below. Earlier checkpoints and their CI do not validate this
newly identified cleanup obligation. Generated registration remains unproved.

The corrected-source Python 3.11 suite passes all 7203 tests, with 74 skips,
using disjoint advisor/`0::4`/`1::4`/`2::4`/`3::4` file shards and eight workers.
Python 3.14's completed shards pass 6921 tests with 38 skips; `0::16` reached
its 165-second bound and is not counted as a passing gate. A verbose rerun
with 30-second faulthandler snapshots identified active work in the 200-class
source-history regression and the historical member-insertion DSL bootstrap,
not a failed assertion. That rerun passes 318 tests in 132.75 seconds (the
200-class case takes 125.73 seconds), completing the corrected-source Python
3.14 suite: 7239 passed, 38 skipped. No tests, input sizes, proof conditions,
or timeout bounds were weakened. The timed-out attempt is retained as a failed
validation attempt, separate from the successful complete rerun.

The corrected-source docs and sdist/wheel builds pass; Sphinx retains the same
two duplicate-object warnings. A fresh install of this wheel outside the
checkout passes actual constructor class-cell mutation/restoration controls
and asserts that its imports come from the installed environment. Both temporary
destructor regression cases pass against that installed wheel. Its public
`analyze_path` API self-scan passes with zero findings from outside the checkout;
the earlier incorrectly named `analyse` smoke failed to import and is not counted
as a passing gate. All 141 production Python files in the wheel match the frozen
checkout byte-for-byte.

The corrected-source production CLI self-scan completes all 79 detectors,
with zero omissions and zero findings, in 16.217 scan seconds. Final full-suite
details (eight workers and 165-second process bounds):

| Runtime / file shard | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 advisor | 795 | 0 | 101.40 |
| Python 3.11 `0::4` | 1790 | 12 | 77.53 |
| Python 3.11 `1::4` | 1606 | 9 | 31.91 |
| Python 3.11 `2::4` | 1577 | 15 | 77.53 |
| Python 3.11 `3::4` | 1435 | 38 | 48.65 |
| **Python 3.11 total** | **7203** | **74** | |
| Python 3.14 advisor | 795 | 0 | 101.22 |
| Python 3.14 `0::16` diagnostic rerun | 318 | 0 | 132.75 |
| Python 3.14 `8::16` | 457 | 3 | 10.63 |
| Python 3.14 `4::8` | 1022 | 2 | 64.53 |
| Python 3.14 `1::4` | 1611 | 4 | 48.09 |
| Python 3.14 `2::4` | 1586 | 6 | 85.53 |
| Python 3.14 `3::8` | 659 | 23 | 38.76 |
| Python 3.14 `7::8` | 791 | 0 | 18.66 |
| **Python 3.14 total** | **7239** | **38** | |

The same sorted non-advisor file list and exclusion described above select these
complete disjoint shards. `0::16` and `8::16` together replace `0::8`; they do
not omit tests. The diagnostic rerun changes verbosity, adds duration reporting
and `-o faulthandler_timeout=30`, and retains the same source, eight workers,
165-second bound, and assertions. Its log is
`/home/ts/nra-native-behavior-validation-G481Dc/native-cleanup-314-0of16-diagnostic.txt`.

### Corrected-source complete-package performance

The same disposable 1057-file OpenHCS-plus-eight-libraries source copy is used,
with 16 parse/analysis workers, a fresh task-owned cache, a 165-second cold bound,
and 60-second warm/edit bounds. These runs do not overlap tests, wheel smoke, or
other advisor scans. The copied EOF-comment edit is restored after timing.

| Mode | Scan seconds | Process wall seconds |
| --- | ---: | ---: |
| Cold | 49.811 | 52.55 |
| Warm | 1.086 | 2.33 |
| One source edit | 4.676 | 7.39 |

All modes complete 79 detectors with zero omissions, 180 active findings and
215 raw findings. The complete semantic projection hash matches each other and
the prior checkpoint:
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`.
Only `.timing`, `.payload_timing`, and `.scan_status.mode/reason` are removed;
detector coverage, completion status, counts, and the full report remain.
These small timing differences do not establish a performance improvement.

Final validation artifacts are retained under
`/home/ts/nra-native-behavior-validation-G481Dc/`: `native-cleanup-*` logs,
JSON reports and wall files, plus `native-cleanup-dist/`. The canonical timing
invocation follows the command above, using that directory's `source`,
`--context-root` equal to `source`, `--no-auto-context-root`, `--json-payload agent`,
and the recorded worker/time bounds. The isolated installed-wheel regressions
use `runpy.run_path` on the test file from outside the checkout and assert
package import paths below the fresh wheel environment's `sys.prefix`.

### Next proof work

Generated construction/registration is still unproved. The class-cell increment
establishes a real current operand prerequisite, not constructor behavior.
Continue the original-source activation, `super`/ABC construction, returned
class identity/lifetime, and actual configuration/registration joins listed
above. A captured Python function must use its genuine initial identity and
current source/default/global/cell evidence; it must not masquerade as an
original source-created function. Existing `SourceFunctionActivationABC`
currently assumes the latter and must be factored at that ownership boundary
before admitting metaclass-body execution. The persistent goal remains active.

After all local validation processes finished, the four exact task-owned roots
for the corrected-source self cache, installed-wheel cache, timing cache, and
pytest fixtures were removed (approximately 732 MiB). They are disposable and
regenerable, not retained proof evidence. Reports, logs, distributions, source
copy, documentation output, and the installed-wheel environment are retained.
The original `/home/ts/code/projects/nominal-refactor-advisor` checkout is
unchanged: its inherited `uv.lock`, `PAUSED_NRA_GOAL.md`, and `cufile.log` remain.

## Current callable metadata ownership prerequisite

Status at 17:58 UTC: applied locally, not yet committed or pushed. Generated
construction and registration still raise the same unproved obligation.

`NativePythonFunctionSource` now owns current signature and default observations
alongside its existing current-code/source correspondence. `NativeCallAuthority`
projects those properties through `AliasProperty`, and the native source-class
entry selects its constructor source through the same owner. The existing
`CompactFunctionSignature.with_default_names` contract is reused. Every default
query rejoins the current function code; cached source ownership is not cached
mutation validation. No callable body is executed to derive metadata, and no
source-created activation or generated-class result is manufactured.

The actual `AutoRegisterMeta.__new__` declarations in both dependency versions
have a fifth, optional `registry_config=None` parameter. Earlier expectations
that the constructor had only four required parameters were incorrect.
Changing that actual default association is observed by the same warmed owner;
the prior signature observation does not become proof of current state.

The refactor was previewed as one six-stage DSL plan over the pre-edit production
snapshot. The example is `docs/examples/native_callable_metadata_owner.py`.
It inserts source-owned defaults/signature and the consumer projection, replaces
three former properties with aliases, adds the constructor source query, and
updates the still-rejecting construction query. Replacement geometry is derived
through `FunctionSourceAuthority.declaration_line_span`, not copied old method
bodies. The combined simulation is clean and emits one two-file diff; its
portable regression also checks the unchanged input snapshot and the remaining
explicit rejection. This is an authored syntax plan, not a behavioral-equivalence
proof. Replacement Python and target selectors remain manually authored, and
the plan is still verbose. No elapsed-time saving is claimed.

Two first-pass regressions had incorrect fixture expectations: current default
observations are intentionally distinct `eq=False` objects, and the actual
constructor already has its `registry_config` default. The controls were
corrected to compare parameter names plus retained value identity and to mutate
the actual default association. The production proof contracts were not relaxed.

Focused gates, eight workers and 60-second bounds, now pass:

| Runtime | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 | 190 | 1 | 13.10 |
| Python 3.14 | 188 | 3 | 17.99 |

The selected files are `test_native_function_source`, `test_native_behavior_proof`,
`test_native_source_class_preparation`, `test_native_class_mro`,
`test_native_definition_applications`, `test_native_definition_operand_chain`,
`test_authored_native_use_invariants`, `test_type_keyed_native_contract`, and
`test_registry_candidate_requirements` under `tests/`, with `.py` suffixes.
Logs: `/home/ts/nra-native-behavior-validation-G481Dc/callable-metadata-{311,314}-focused.txt`.
At 17:59 UTC, the exact new source passes the standalone Python 3.14 `0::16`
full-suite shard: 318 passed in 132.47 seconds, eight workers, 165-second bound.
The remaining complete disjoint shards are running two at a time with eight
workers each, under the same bounds and the file-slice schedule of the preceding
checkpoint. Logs use the `callable-metadata-full-*` prefix in the validation
directory. Full-suite totals are not yet established.

Fresh Sphinx `-E -j 8` passes with the same two duplicate API-object warnings.
Fresh isolated sdist/wheel builds pass. An initial `--no-isolation` build could
not import the active environment's missing `setuptools.build_meta`; it is not
a passing gate. The normal isolated build uses the project's declared backend
dependencies and passes without changing the active interpreter. Logs are
`callable-metadata-docs.txt`, `callable-metadata-build.txt` (failed non-isolated
attempt), and `callable-metadata-build-isolated.txt` (successful declared build).
Installed-wheel and complete cold/warm/edit gates are still pending.

The preceding `997d85f` hosted run now passes five of seven jobs: docs/wheel,
both macOS jobs, and Python 3.14 on Ubuntu/Windows. Python 3.11 on Ubuntu/Windows
remain in progress, verified at 17:59 UTC. That CI run does not validate the
uncommitted metadata refactor.

### Frozen-source metadata checkpoint validation

All full-suite shards pass with eight workers and 165-second bounds:

| Runtime / file shard | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 advisor | 795 | 0 | 102.36 |
| Python 3.11 `0::4` | 1790 | 12 | 91.80 |
| Python 3.11 `1::4` | 1606 | 9 | 58.17 |
| Python 3.11 `2::4` | 1584 | 15 | 71.29 |
| Python 3.11 `3::4` | 1435 | 38 | 42.39 |
| **Python 3.11 total** | **7210** | **74** | |
| Python 3.14 advisor | 795 | 0 | 102.16 |
| Python 3.14 `0::16` | 318 | 0 | 132.47 |
| Python 3.14 `8::16` | 457 | 3 | 12.80 |
| Python 3.14 `4::8` | 1022 | 2 | 73.77 |
| Python 3.14 `1::4` | 1611 | 4 | 64.82 |
| Python 3.14 `2::4` | 1593 | 6 | 79.20 |
| Python 3.14 `3::8` | 659 | 23 | 36.76 |
| Python 3.14 `7::8` | 791 | 0 | 27.37 |
| **Python 3.14 total** | **7246** | **38** | |

The same documented sorted non-advisor file lists and disjoint slices are used;
`0::16` runs alone first, then the remaining shards two at a time. This avoids
contention in the existing 200-class history control without changing its
inputs, assertions, or bounds. The complete runner exits zero. Python versions
and metaclass-registry dependencies are unchanged from the preceding checkpoint.

The freshly built wheel is installed into the existing task-owned isolated
environment and all 141 production Python files match the current checkout
byte-for-byte. Outside the checkout, with package imports asserted below that
environment's `sys.prefix`, its new constructor-default, current call-metadata,
six-stage DSL, and warmed constructor-code mutation controls pass. The installed
public `analyze_path` API also returns zero production findings with eight parse
and analysis workers. Adding only the integration `tests/` directory to the
control interpreter does not expose an editable package import.

The production CLI self-scan completes all 79 detectors with zero omissions and
zero findings (16.910 scan seconds). It and the independent installed-API gate
ran concurrently, so this is a completion gate, not a performance comparison.
Logs/reports: `callable-metadata-wheel-controls.txt`,
`callable-metadata-wheel-api.txt`, and `callable-metadata-self.json` under the
validation directory. Complete-package scans run after all local tests and
API/self-scan processes have completed.

Complete package gates use the same restored 1057-file source copy as the
preceding checkpoint, excluding tests and including the eight external
libraries. All local test/build/self/API processes are terminal before timing.

| Mode | Scan seconds | Wall seconds | Cache |
| --- | ---: | ---: | --- |
| Cold | 47.176 | 49.83 | miss |
| Warm | 1.021 | 2.25 | hit |
| One edit | 4.335 | 7.03 | partial |

Each mode completes all 79 detectors with zero omissions, 180 active findings,
and 215 raw findings. The complete semantic projection hash remains
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100` for
all modes and the preceding checkpoint, normalizing only the same timing and
scan mode/reason fields. No claim of improved detector recall, behavioral proof,
or causal performance improvement follows from these equivalent reports.
Commands use the documented complete-package scope, sixteen parse/analysis
workers, a fresh task-owned cold cache, and 165/60/60-second bounds. The edit is
one novel EOF comment in the copied `openhcs/__init__.py`, restored after the
successful edit scan. Logs, JSON, and wall files use the `callable-metadata-`
prefix in the validation directory.

The preceding `997d85f` hosted integration run completes all seven jobs
successfully, verified at 18:08 UTC. The new metadata source has complete local
gates, but no exact-SHA hosted gate until it is committed and dispatched.

After every local validation process completed, the three exact new task-owned
self/wheel/performance caches and pytest fixture root were removed, recovering
approximately 729 MiB. They can be regenerated; retained logs, reports,
distributions, documentation, installed-wheel environment, and source copy
are unchanged. The original checkout's inherited changes are still untouched.
