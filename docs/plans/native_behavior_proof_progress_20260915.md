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

Current work at 19:43 UTC: the body-cache validation correction and its shared
source-geometry factoring are committed and pushed as
`b3749510cb309b723234a6968b10ec9523e31a32`. Four adversarial
controls reproduce failures on the published source and pass after the
correction. The first broader suite exposed repeated full-module syntax parsing;
the resulting shared-geometry correction passes 210 tests on Python 3.11 and 208
on Python 3.14. The complete suites pass 7,227 tests with 74 skips on Python 3.11
and 7,263 tests with 38 skips on Python 3.14.
One Python 3.11 worker crashed while its timed traceback diagnostic was active;
the actual test passes directly and with canonical pytest diagnostic settings.
The complete affected Python 3.11 shard also passes without that added
diagnostic, with all assertions/input sizes preserved. The installed public-API
gate now passes against a fresh owned cache. Complete-package cold/warm/edit
gates pass with unchanged complete semantic reports. Exact-SHA hosted run
[35012319015](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35012319015)
is terminal: five jobs pass and both macOS runtime jobs fail the same
collected-family fixture assertion. The new current-C3 parent-operand contract
and derived exact cache-size fixture correction are committed and pushed as
`98ca65bf4a104e990bd3ffe2cf10aac97fd991b0` on the same checkpoint branch.
Their complete local suites pass 7,237/74 and 7,273/38, respectively, along
with fresh docs/package, installed-wheel/API and production self-scan gates.
Complete cold/warm/edit gates also pass with unchanged semantic reports.
Exact-SHA hosted run
[35015301107](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35015301107)
matches that code commit. All seven jobs are queued or running at this check;
hosted verification is not yet complete.
Generated
construction/registration still rejects the same unproved obligations.

The callable-metadata ownership prerequisite is committed and pushed as
`ec541bd2d22d3e3d42097df962a793741b9ddb5f` to
`checkpoint/native-proof-integration-20260914`; the remote SHA matches. Its
exact-commit hosted run
[35005841288](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35005841288)
is complete with all seven jobs passing, verified at 18:42 UTC. Its complete local
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

## Captured-body cache validation correction

The constructor activation audit exposed two related current-code gaps in the
existing source owners: `NativePythonFunctionSource.flow` returned a cached
compact projection without rejoining current code, and
`NativeReturnedClosureFactorySource` retained a mutable definition AST. Reused
dataclass application owners also cached parameter and selector observations.
They are direct prerequisites for captured-function execution, not permission
to admit generated metaclass behavior.

Four new controls fail on the unchanged published implementation: warmed-flow
code mutation, a swap to another valid body in the same defining file, warmed
closure-factory code mutation, and exposed definition-AST mutation. The last two
still accepted or returned stale syntax; the first two returned stale flow.
Exact source correspondence alone cannot reject another valid source span.
The pre-fix gate reports four failures in 2.00 seconds; the same controls pass
afterwards, four in 3.07 seconds. Logs: `current-body-cache-before.txt` and
`current-body-after-controls.txt` in the validation directory.

The compact projection remains retained as immutable evidence. Its public flow
query rejoins actual current code and rejects a changed source span before
returning that retained projection. A new source owner can represent the new
valid body; the old warmed owner cannot silently keep using its prior body.
Closure-factory syntax now derives from its existing `NativePythonFunctionSource`
owner, not a copied AST. Parameter and selector queries are live properties;
one local observation tuple is used within an individual binding check so its
identity comparisons remain coherent. Neither body activation nor arbitrary
side effects acquire automatic admission from these changes.

The production ownership edit is an eleven-stage DSL preview in
`docs/examples/current_native_body_owner.py`, clean and applied as its combined
diff. A declared-call rewrite correctly failed because the synthesized
dataclass constructor call authority is unresolved. The replacement uses the
DSL's explicit syntax patch, with the call span derived through
`SourceTextGeometry`, without inventing a call proof. Repeated property-read
replacement uses declaration-owned current source, not copied old function
bodies. No unrelated DSL capability or native whitelist was added.

The initial broader focused runs hit their 60-second bounds near completion,
on both versions; they are not green gates. A separate diagnostic run shows
repeated full-module native emission work outside `ScanCache.scope()`.
The existing bounded, nested-sharing cache scope now encloses the closure and
dataclass proof queries. This caches immutable correspondence inside an
invocation, not current-code/default/binding validation. A three-stage DSL
preview wraps those queries while preserving the function docstring and
original body. The scoped focused suites pass with eight workers:

| Runtime | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 | 158 | 1 | 20.50 |
| Python 3.14 | 159 | 0 | 22.23 |

The seven selected files are `test_native_function_source`,
`test_native_dataclass_factory`, `test_definition_application_activation`,
`test_native_behavior_proof`, `test_authored_native_use_invariants`,
`test_type_keyed_native_contract`, and `test_registry_candidate_requirements`,
under `tests/` with `.py` suffixes. Logs use `current-body-scoped-*-focused.txt`.
Further controls query a reused end-to-end dataclass application after factory
or processor mutation, then restore its original code, and replay both DSL
plans on a portable source snapshot. The final focused gates pass, 161 passed
and one skip on Python 3.11 in 21.61 seconds; 162 passed on Python 3.14 in
22.98 seconds. Logs: `current-body-final-{311,314}-focused.txt`. Neither target
function body is invoked by the mutation controls. Broad validation now follows;
full-suite/package/installed-wheel/complete-package gates are pending before
this additional correction is committed.

### Shared current-code geometry and fresh body windows

The first full correction suite did not pass: Python 3.11 shard `0::4` reached
its 165-second bound. The remaining owned matrix processes were stopped before
further source edits. A separate eight-worker result-identity diagnostic also
reached 60 seconds. The stacks showed repeated parsing of the complete stdlib
`dataclasses` source while revisiting current factory/processor evidence.
Neither timeout is recorded as a passing gate.

`NativePythonCompilation.function_source_span` now owns the current exact
code/source correspondence without exposing syntax. Compact flow and initial
current-source capture consume that immutable geometry directly. Consumers
requesting syntax still receive a new AST, but only the selected function window
is reparsed. The first source-layout query derives the original declaration
boundary; its bounded cache retains an integer, not AST or a mutable-code
validation answer. Every subsequent query first rejoins actual current code.

An initial window used `co_firstlineno` as its beginning. The multiline
parenthesized-decorator control correctly rejected that approach: the native
first line can identify the decorator expression after its `@` marker.
The existing exact token-marker semantics were factored to
`SourceLineSegmentAuthority` instead. `SourceTextGeometry` inherits token
production and projects its decorator policy through that owner.
`NativePythonCompilation` inherits the same source/geometry contract and no
longer redeclares `source`. Fresh function windows preserve decorators, UTF-8
columns and original AST positions; no target function is invoked.

These edits were previewed and applied through existing authored syntax DSL
operations: a five-stage current-span plan, one fresh-window stage, and a
thirteen-stage shared-geometry plan, in addition to the preceding eleven plus
three body-cache stages. Examples are `current_native_body_owner.py` and
`shared_declaration_geometry_owner.py` under `docs/examples/`. The new portable
chain checks inherited field ownership, shared token methods and unchanged
input source. A clean syntax simulation is not a behavior-equivalence proof.

Four fresh-window controls failed before windowing; the warmed compact-flow
geometry control already passed after the span-only step. The span-only
result-identity diagnostic still exceeded 60 seconds, so that intermediate
source was not published. After the shared-geometry correction, the complete
result-identity/native-source/generic-geometry/class-header selected surface
passes 97 tests with one skip in 7.72 seconds under eight workers.

The wider final affected surface passes on both runtimes:

| Runtime | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| Python 3.11 | 210 | 1 | 14.73 |
| Python 3.14 | 208 | 3 | 15.02 |

Selected files are `test_native_function_source`, `test_source_geometry`,
`test_definition_result_identity`, `test_native_dataclass_factory`,
`test_native_definition_applications`, `test_native_source_class_preparation`,
`test_authored_native_use_invariants`, `test_type_keyed_native_contract`, and
`test_registry_candidate_requirements`. Logs use
`current-body-{311,314}-final-selected.txt`. All runs retain the existing
fail-closed construction/registration obligations. Final full-suite, docs,
package, installed-wheel and complete-package cold/warm/edit gates remain
pending on this frozen source; earlier package/self-scan passes belong to the
intermediate correction, not this later geometry factoring.

The complete authored trajectory was also replayed from the committed four-file
source mapping: all `[11, 3, 5, 1, 13]` stages are clean, total 33. After normal
formatting, its final source matches every current production file exactly, and
the original snapshot retains its initial source. The aggregate four-file diff
and result are retained in `current-body-composed-33-replay.txt`. This validates
composition and replay, not behavioral equivalence or elapsed-time saving.

Frozen-source complete validation started with the large Python 3.14 `0::16`
shard alone: 318 tests pass in 126.53 seconds, exit zero under its 165-second
bound. The remaining disjoint Python 3.11/3.14 matrix is running with two pools
of eight workers and individual 165-second bounds. Final logs use
`current-body-geometry-full-*`; the previous timed-out `current-body-full-*`
logs are not reused as gates.

Final-source fresh Sphinx and isolated wheel/sdist builds pass. The newly
installed wheel matches all 141 production Python modules byte-for-byte against
the wheel archive and the checkout. Mutation, narrow-syntax, shared-geometry,
and composed DSL controls pass from outside the checkout with production
imports verified under the installed environment prefix. Logs use
`current-body-geometry-{docs,build,wheel-controls}.txt`.

The fresh CLI production self-scan passes with 79 detectors, zero omissions,
complete status and zero findings (`current-body-geometry-self.json`). Its
30.837 seconds were measured while tests were running, not as a performance
benchmark. The first installed public-API scan reached its 60-second process
bound while the complete matrix was active; it is not a pass. That terminal
run is now repeated with the allowed 165-second bound and the same owned cache,
logged to `current-body-geometry-wheel-api-retry165.txt`. The full matrix and
public-API retry remain live at 18:56 UTC. Complete-package cold/warm/edit scans,
the scoped commit/push, and exact-SHA hosted validation still follow.

### Complete-suite diagnostic crash audit

The final Python 3.14 disjoint suite is terminal and passes: 7,263 tests with
38 skips. All its shards return zero. Python 3.11 `0::4` now passes 1,790 tests
with 12 skips in 97.44 seconds, below the unchanged 165-second bound. This
replaces the earlier correction's timed-out shard; it does not hide it.

Python 3.11 `2::4` reports a worker crash in
`test_one_unknown_write_is_shared_across_product_queries_without_class_fanout`.
A separate eight-worker rerun with `faulthandler_timeout=15` reproduces that
crash; the timed native stack dump contains invalid frame/line observations.
The identical 41-product test passes as a direct diagnostic and in pytest with
its default traceback settings, eight workers, and the same assertions/input:
one test passes in 31.71 seconds. These observations implicate the added timed
traceback diagnostic, but are not a complete explanation of the CPython crash.
No test or production code was changed to bypass the failure.

The complete Python 3.11 `2::4` shard is now being rerun using the normal CI
settings, without the added timed dump and under the same 165-second process
bound. Its log is `current-body-geometry-full-311-2of4-canonical.txt`. The crash
logs and direct diagnostic remain retained. The installed public-API retry also
reached its 165-second process bound while the earlier matrix was active; it is
not a pass. The unchanged request is repeated after that matrix is terminal in
`current-body-geometry-wheel-api-postmatrix.txt`; this functional API gate has
not yet passed. Complete-package cold/warm/edit scans still await terminal
heavy processes. The source remains frozen and uncommitted.

### Canonical complete suite and installed-API diagnosis

The complete affected Python 3.11 `2::4` shard passes under canonical CI
settings: 1,597 tests, 15 skips, 57.98 seconds, exit zero. The complete disjoint
suite therefore passes 7,227 tests with 74 skips. The timed diagnostic crash
remains separately recorded above; its failing run is not counted as a pass.

| Runtime | Shard | Passed | Skipped | Seconds |
| --- | --- | ---: | ---: | ---: |
| Python 3.11 | Advisor | 795 | 0 | 109.16 |
| Python 3.11 | `0::4` | 1,790 | 12 | 97.44 |
| Python 3.11 | `1::4` | 1,606 | 9 | 48.39 |
| Python 3.11 | `2::4`, canonical diagnostics | 1,597 | 15 | 57.98 |
| Python 3.11 | `3::4` | 1,439 | 38 | 36.49 |
| Python 3.14 | Advisor | 795 | 0 | 110.84 |
| Python 3.14 | `0::16` | 318 | 0 | 126.53 |
| Python 3.14 | `8::16` | 457 | 3 | 11.29 |
| Python 3.14 | `4::8` | 1,022 | 2 | 80.43 |
| Python 3.14 | `1::4` | 1,611 | 4 | 50.45 |
| Python 3.14 | `2::4` | 1,606 | 6 | 66.68 |
| Python 3.14 | `3::8` | 663 | 23 | 34.75 |
| Python 3.14 | `7::8` | 791 | 0 | 17.35 |

The installed public-API request timed out again after the matrix, and is not
green. Its cache contains an exclusive rebuild lock recording PID `2703355`,
the original timed-out process, with modification time 18:54:38 UTC. That PID
is absent at inspection. `AnalysisCacheRebuildLockAuthority.lease` polls an
existing lock, and its current stale policy is age-only with a default of 600
seconds. This explains why retries against the same interrupted cache can wait
instead of performing analysis; it does not explain the initial timeout.
No cache lock was removed to make the gate pass. The identical installed-API
request is now running against a new owned cache with the unchanged eight
parse/analysis workers and 165-second bound. Its log is
`current-body-geometry-wheel-api-fresh.txt`. The complete CLI self-scan is an
independent passing gate, not a substitute for this API gate.

The fresh installed public-API request is terminal at 19:10 UTC, exit zero. It
returns a list with zero production findings using the installed wheel, eight
parse/analysis workers, and the unchanged 165-second bound. No rebuild lock
remains in that fresh cache. The original interrupted-cache lock and the failed
retry logs are retained for the documented age-only lock-policy concern. These
observations distinguish interrupted-cache waiting from this successful fresh
analysis; they are not a fix to the existing stale-lock policy.

All full-suite, build, installed controls, API, and CLI self-scan processes are
terminal before the complete-package timing run. Cold timing now uses the
restored 1,057-file production source copy and fresh owned cache
`current-body-geometry-perf-cache-amxtcE`, with sixteen parse/analysis workers,
the same complete-package scope, and the unchanged 165-second bound. Reports
use `current-body-geometry-{cold,warm,edit}` prefixes; warm and one-edit timings
follow the terminal cold run.

### Final current-body/shared-geometry complete-package gates

All three final-source timing runs are terminal, exit zero. The one novel EOF
comment was added only to the disposable copied `openhcs/__init__.py`, and
removed after the one-edit run completed. No reference, live OpenHCS, or
original NRA checkout was changed.

| Mode | Scan seconds | Wall seconds | Cache |
| --- | ---: | ---: | --- |
| Cold | 45.984 | 48.63 | miss |
| Warm | 1.039 | 2.29 | hit |
| One edit | 4.277 | 6.82 | partial |

Each run completes all 79 detectors, with zero omissions, 180 retained active
findings and 215 raw findings. The full semantic projection hash is
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`
in every mode and matches the preceding checkpoint. Normalization removes only
`.timing`, `.payload_timing`, and `.scan_status.mode`/`.scan_status.reason`.
This is report consistency, not proof of complete detector recall or behavioral
equivalence. Single-run timings do not establish a causal speedup over the
preceding 47.176/1.021/4.335-second checkpoint.

The command shape remains the documented complete-package gate:

```sh
timeout 165s env \
  NRA_CACHE_HOME=/home/ts/nra-native-behavior-validation-G481Dc/current-body-geometry-perf-cache-amxtcE \
  TMPDIR=/home/ts/nra-native-behavior-validation-G481Dc \
  /home/ts/code/projects/nominal-refactor-advisor/.venv/bin/python \
  -m nominal_refactor_advisor \
  /home/ts/nra-native-behavior-validation-G481Dc/source \
  --context-root /home/ts/nra-native-behavior-validation-G481Dc/source \
  --no-auto-context-root --json --json-payload agent \
  --parse-workers 16 --analysis-workers 16 --scan-budget-seconds 165
```

Warm and edit use 60 seconds for both bounds, the same scope and cache, and
the recorded EOF edit between those two terminal runs. Reports, stderr and
wall records are `current-body-geometry-{cold,warm,edit}.{json,stderr,wall}`.
The cache is task-owned and disposable; reproducing a cold run requires a new
empty cache, not a deleted path assumed to contain prior evidence.

The current correction adds 17 test cases without weakening existing inputs,
assertions, skips or automatic proof obligations. All 33 authored DSL stages
replay to the exact four final production files. Supported actual MRO lookup
and narrow native calls remain admitted; generated AutoRegisterMeta
construction/registration remains explicitly unproved. This checkpoint is a
validated ownership/cache correction, not completion of target three.

After every local gate was terminal, the resolved task-owned intermediate,
self-scan, API, performance caches and pytest fixture directory were removed
(roughly 950 MiB). This includes the documented interrupted-cache lock; its
removal was cleanup after the successful independent API gate, not a fix used
to pass that gate. Logs, reports, source copy, environments, docs and build
artifacts remain. The removed generated caches/fixtures can be regenerated.
No shared cache, other checkout, user input handoff or unrelated process was
changed.

### Current correction publication

Code commit `b3749510cb309b723234a6968b10ec9523e31a32` is pushed to
`checkpoint/native-proof-integration-20260914`; the remote SHA matches. Main
remains unchanged at `76fda5db5dccb60bcfbdb062a19336ff5a4b569b`. The user's input
handoff remains untracked and was not staged. All ten committed paths are this
correction's source, tests, authored replay examples and progress record.

Hosted integration run
[35012319015](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35012319015)
was dispatched on the checkpoint branch and reports this exact code SHA. It is
queued at 19:13 UTC, so local gates are passing but the hosted gate is pending.
No completed hosted result from the previous metadata commit is transferred to
this correction. A later documentation-only status commit does not change
which code SHA the run validates. The long-running goal remains active because
generated construction/registration evidence is not complete.

## Current-C3 parent operand and hosted fixture correction

Status at 19:31 UTC: local and uncommitted. The generated-class construction
and registration obligations still reject; no new automatic status was asserted.

### Declaration-owned parent selection

`NativeClassMroDeclaration.member_owner` now accepts an optional exact
`start_after` owner. It reads the current native C3 order on every query, finds
the starting owner by identity, and inspects subsequent stored namespaces using
the existing key-validation contract. A foreign or missing starting owner
rejects without invoking equality or representation hooks. Query names must
be exact native strings before hashing. The old method accepted a `str`
subclass and invoked its active hash in a new adversarial control.

`python_constructor` projects the same scoped lookup, retaining its existing
ordinary-metameta, exact-staticmethod, exact-function and class-cell validation.
For the real AutoRegisterMeta, the selected parent is the actual ABCMeta
function, with its class cell bound to ABCMeta. In a diamond, traversal after
the left branch selects the right sibling before the common ancestor. Neither
member selection nor source metadata proves descriptor execution, original
`super()` activation, parent construction, ABC initialization or registry writes.

Six controls fail on the pre-edit source; after the implementation and fixture
correction, both runtime selections pass 143 tests. Three additional controls
revalidate the selected sibling cell, current descriptor and portable DSL
signature/rewrite composition. The first MRO mutation fixture used pytest's
class-attribute patching, which treats the inherited `__bases__` descriptor as
absent and attempts an invalid deletion during cleanup. Its corrected fixture
restores the actual native base tuple in `finally`; the mutation assertions are
unchanged. Logs are `current-super-lookup-{311,314}-final-selected.txt`.

The authored four-stage plan is `docs/examples/current_super_mro_lookup.py`.
It uses the existing signature and body operations and source-derived call
geometry. Its clean combined simulation reproduces the formatted production
file exactly and preserves the committed input snapshot
(`current-super-lookup-replay.txt`). This is syntax composition, not a
behavioral-equivalence or elapsed-time proof.

### Actual frozen parent boundary

The genuine parent function has code filename `<frozen abc>` in both runtimes.
Python 3.11's current source loader rejects it with `Python implementation has
no inspectable source`. Python 3.14's current loader succeeds and the current
source signature contains four positional-only parameters plus variadic
keyword parameters. Logs are `current-super-parent-source-{311,314}.txt`.
The next source-entry work must supply genuine source provenance for the
frozen function and rejoin its exact current code. Injecting linecache text,
trusting a module name, or treating a filename as behavioral evidence would
not discharge that requirement.

The existing `SourceFunctionActivationABC` requires a source-created callee;
its default binding and `SourceFunctionEntry` creator frame likewise come from
that source creation. Actual imported constructor activation needs the shared
body/entry contracts to consume its existing current native source owner,
actual defining globals/builtins and validated cells. It must not invent a
source creation event. Subsequent parent construction and original receiver
effect transport remain required before a generated registry entry can pass.

### Hosted failure and exact serialized fixture boundary

Exact code-SHA run 35012319015 reports the same single failure on macOS/Python
3.11 and 3.14: `test_collected_family_can_opt_into_a_larger_bounded_cache_payload`
retains zero files where its fixed 10,000-byte allowance expects one. The
respective jobs otherwise pass 7,226 and 7,262 tests, with 74 and 38 skips.
The Ubuntu/Python 3.14 and docs/wheel jobs pass; the remaining jobs are live.
The complete job logs are `current-body-macos{311,314}-ci.txt`.

The retained log does not report the macOS payload's actual byte count. A real
Linux serialization probe records 8,532 bytes under the ordinary fixture path.
The same committed test with a longer real path serializes 11,792 bytes and
reproduces its assertion failure. The new derived-boundary fixture passes on
that real path with a 11,794-byte receipt. Implementation-source paths are part
of the serialized payload even though they are excluded from its repr/equality.
This corroborates the fixed-size fixture problem; it does not substitute a
measured macOS byte count. Logs are `current-ci-payload-{probe,long-path}.txt`.

The test now measures the real unchanged serializer output under the default
64-byte test limit and verifies that it is not retained. It then opts into the
measured exact boundary through the same collection/publication path and
requires a file of precisely that size. An additional case sets a limit one
byte smaller and requires rejection. The production schema and family policy
are unchanged. Both runtime selections pass three tests
(`current-ci-cache-{311,314}-final-selected.txt`); hosted confirmation follows
the eventual scoped code checkpoint, not a rerun of unchanged failed code.

### Broader validation in progress

The final current-source file inventory matches the full pytest collection.
There are nine new MRO cases plus one additional cache-boundary case; expected
full totals are 7,237/74 on Python 3.11 and 7,273/38 on Python 3.14. Python 3.14
`0::16` is terminal and passes 318 tests in 133.58 seconds. The remaining
file-disjoint shards run with two concurrent jobs, eight workers each, canonical
diagnostic settings and 165-second bounds. The temporary driver is
`current_super_suite.py`; worklist and logs use `current-super-full-*` under
the validation root. The live supervisor session is `36825` at this checkpoint.
No full-suite pass is claimed until every required shard is terminal.

Fresh docs and isolated wheel/sdist builds pass with the two existing docs
warnings (`current-super-{docs,build}.txt`). Installed-wheel controls/API,
production self-scan, and complete cold/warm/edit gates follow on this frozen
source. The earlier passing gates belong to the pushed body correction, not
this subsequent parent-operand contract. Own live-run caches and fixtures must
remain until their handles are terminal; clean them after retaining evidence.

The installed wheel matches all 141 production module bytes against both the
archive and checkout. All nine new MRO/DSL controls and both actual serialized
cache boundaries pass outside the checkout, with every imported production
module verified under the installed prefix (`current-super-wheel-controls.txt`).
Fresh installed public-API and production self-scan requests are live in
sessions `14894` and `12358`; their results are not yet passes.

A further Python 3.11 probe verifies that the actual ABCMeta parent function's
globals are the actual `abc` module dictionary. Reading that module's physical
source, preserving the function's frozen code filename for compilation, and
using the existing full current-code matcher succeeds. It also derives four
positional-only parameters plus keyword variadic parameters. No target function
or constructor was invoked and no linecache content was injected
(`current-super-frozen-physical-source-311.txt`). Thus the automatic loader gap
is source acquisition, not absence of matching source in this environment.
This explicit-input probe is not an automatic source-loader implementation or
a constructor-effect proof; authenticating actual source candidates still
belongs in the existing source owner before imported activation can use them.

### Terminal broad gates for the parent-operand correction

All 13 file-disjoint suite shards are terminal with exit zero. The aggregate
counts match the full collected scope: 7,237 passed/74 skipped on Python 3.11
and 7,273 passed/38 skipped on Python 3.14. The extra one-byte rejection case
increases the final full collection to 7,311; the collected file inventory is
unchanged. All source/test inputs remain frozen from the final fixture edit.

| Runtime | Shard | Passed | Skipped | Seconds |
| --- | --- | ---: | ---: | ---: |
| Python 3.11 | Advisor | 795 | 0 | 106.49 |
| Python 3.11 | `0::4` | 1,791 | 12 | 95.53 |
| Python 3.11 | `1::4` | 1,615 | 9 | 46.42 |
| Python 3.11 | `2::4` | 1,597 | 15 | 73.82 |
| Python 3.11 | `3::4` | 1,439 | 38 | 62.20 |
| Python 3.14 | Advisor | 795 | 0 | 108.65 |
| Python 3.14 | `0::16` | 318 | 0 | 133.58 |
| Python 3.14 | `8::16` | 457 | 3 | 15.05 |
| Python 3.14 | `4::8` | 1,023 | 2 | 71.21 |
| Python 3.14 | `1::4` | 1,620 | 4 | 41.27 |
| Python 3.14 | `2::4` | 1,606 | 6 | 95.16 |
| Python 3.14 | `3::8` | 663 | 23 | 38.27 |
| Python 3.14 | `7::8` | 791 | 0 | 18.55 |

The installed public API is terminal, returns a list with zero production
findings, and verifies production imports under the installed environment.
The separate complete CLI production self-scan is terminal, with 79 detectors,
zero omissions, complete status and zero findings. Its 40.337 scan seconds
overlapped the matrix, so are not a performance benchmark. Both gates retain
fresh isolated caches and the existing eight-worker configuration.

Earlier hosted run 35012319015 is terminal with five passing jobs and two
macOS fixture failures. That red result remains recorded. It is not a passing
gate for this later source. After every local heavy process was terminal, the
complete-package cold run started with the restored 1,057-file source copy and
fresh cache `current-super-perf-cache-ScMPQI`, sixteen parse/analysis workers,
and the same 165-second bound. Warm and one-edit use 60-second bounds and follow
terminal preceding runs. Logs use `current-super-{cold,warm,edit}`.

### Final parent-operand complete-package performance gate

| Run | Scan seconds | Command wall seconds | Cache result |
| --- | ---: | ---: | --- |
| Empty-cache cold | 48.913 | 51.67 | MISS |
| Unchanged warm | 1.080 | 2.30 | HIT |
| Novel one-file edit | 4.538 | 7.33 | PARTIAL |

All three complete-package runs include the same 1,057-file production source
copy and all 79 detectors, with zero omissions, 180 retained findings and 215
supporting raw findings. Removing only timing fields and scan cache-mode/reason
fields gives the same complete semantic report hash as the preceding checkpoint:
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`.
This checks report consistency, not detector recall, behavioral equivalence or
a causal performance improvement from one measurement. The copied-source EOF
edit was restored after its process completed. Both final full collections
record 7,311 tests, matching the disjoint-suite pass/skip totals above.

After authoritative process inspection confirmed that every task-owned test,
collection, installed-wheel, self-scan and performance process was terminal,
the seven exact validated disposable paths were removed: the parent-operand
self/API/API-analysis/performance caches, installed-wheel fixtures, pytest
fixtures, and the long-path payload probe fixtures. This freed approximately
741 MiB. These caches and fixture files are regenerable; retained logs, reports,
source copy, docs, distributions, helper scripts and environments were not
removed. No other checkout or shared cache was cleaned.

### Parent-operand checkpoint publication

The five owned implementation/test/example/progress files are committed as
`98ca65bf4a104e990bd3ffe2cf10aac97fd991b0` and pushed to
`checkpoint/native-proof-integration-20260914`; `git ls-remote` confirms that
exact remote SHA. Workflow dispatch produced run
[35015301107](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35015301107)
with the same head SHA, independently verified through `gh run view`. At
19:43 UTC its seven jobs are queued or running. The preceding two macOS
fixture failures remain failures of their older commit, not passing evidence
for this correction. This status-only follow-up may have a different HEAD
from the code SHA under test. The supplied handoff remains untracked and
unstaged; main, other worktrees, release tags and publications are untouched.
Generated construction/registration remains the active unfinished objective.
