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

Checkpoint `308375890ef8927aafeb95b4eed0a27f50ff95b6` is committed and pushed to
`checkpoint/native-proof-integration-20260914`. Its scoped implementation passes
the frozen-source local gates below. Hosted cross-platform validation is running:
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
