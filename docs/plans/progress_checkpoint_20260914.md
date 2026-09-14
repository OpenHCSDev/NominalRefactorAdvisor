# Native-proof and scan-performance checkpoint, 14 September 2026

This records the completed native-proof and scan-performance increment that
began as an explicitly unfinished development checkpoint. The integration
branch preserves the complete source, test, proof-reuse, and performance work
together. Base commit: `5f96c76358fe7c3be53af8d325bc513d693168e4`.

## Integration update

The native-admission and dependency-aware source-proof reuse branches are
combined on `checkpoint/native-proof-integration-20260914`. The historical 209
failures classified below are resolved. The complete Python 3.11 and 3.14
suites pass under the prescribed eight-worker splits:

| Runtime and shard | Passed | Failed | Skipped | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Python 3.11 `test_refactor_advisor.py` | 795 | 0 | 0 | 93.10 |
| Python 3.11 remaining 1/2 | 3,085 | 0 | 63 | 110.26 |
| Python 3.11 remaining 2/2 | 3,256 | 0 | 8 | 69.90 |
| **Python 3.11 total** | **7,136** | **0** | **71** | |
| Python 3.14 `test_refactor_advisor.py` | 795 | 0 | 0 | 94.19 |
| Python 3.14 remaining 1/4 | 1,624 | 0 | 4 | 84.62 |
| Python 3.14 remaining 2/4 | 1,488 | 0 | 32 | 39.06 |
| Python 3.14 remaining 3/4 | 1,482 | 0 | 0 | 45.50 |
| Python 3.14 remaining 4/4 | 1,782 | 0 | 0 | 154.03 |
| **Python 3.14 total** | **7,171** | **0** | **36** | |

Final cross-platform hardening makes context-free module identity derive from
the source path's lexical anchor rather than the process working directory.
This preserves root-relative Windows source identities without coupling them to
the checkout path. The project module-path authority now recognizes that same
lexical anchor before selecting its declared, analysis, or containing import
root, so Windows root-relative overlays follow the same module boundary as
POSIX rooted overlays. Exact-source test fixtures now write exact bytes and
consume the same slash-normalized path projection as the source index. After
this change, the complete local Python 3.11 suite passes 7,136 tests with 73
skips; the affected Python 3.11 and 3.14 surfaces each pass 35 tests with two
platform-specific skips.

The integrated proof retains exact native definition-application evidence for
the supported dataclass transformation, activation-specific returned parameter
identity, conclusive partial non-injectivity evidence, and exact unchanged
module-local proof owners across virtual source edits. Global dependency queries
are recomputed from the complete projected module set.

Python 3.14 generic classes now retain the exact compiler-generated wrapper,
its immediate activation, and the original synthetic type-parameter closure
binding without equating that wrapper to a source frame. The declaration-owned
binding admits only its exact tuple production. Arbitrary closure reads and the
implicit Generic base protocol remain unproved; the mutation controls continue
to reject them.

One attempted speedup cached validation-bearing class-body receipts and allowed
a warm query to hide later source mutation. The existing negative controls
caught that regression; those receipts again revalidate their original source
on every access. Compiler-owned annotation writes instead use their exact
compact target and declared key to avoid replaying unrelated completed class
bodies. Rebound `__annotations__` values retain the ordinary unresolved item
write path.

The fresh full OpenHCS plus eight-external-library production scan covers 1,015
Python files with tests excluded and all 79 detectors included:

| Run | Parse seconds | Analysis seconds | Scan seconds | Command wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Empty-cache cold | 40.175 | 6.582 | 46.757 | 49.72 |
| Unchanged exact cache | 0.000 | 1.197 | 1.197 | 2.47 |
| Novel one-file edit | 2.270 | 2.257 | 4.527 | 7.35 |

Every run completed 79 of 79 detectors with zero omissions and emitted the
same 180 active findings from 215 supporting raw findings. The complete
semantic report projection has SHA-256
`4191f7d7c49191222549f74d03ad17dd924d0c3f4d08cc9acc17e92a487d8743`
for all three reports. The one-line edit was made only in a fresh disposable
source copy and was restored. A separate empty-cache scan of NRA's production
package after the final cross-platform hardening completed all 79 detectors
with zero omissions and zero findings in 16.265 scan seconds (17.79 seconds
command wall time).

The older validation sections below remain as historical checkpoint evidence
rather than current status.

## Included work

- Parallel semantic preparation and scan-scoped caching/lifetime improvements.
- Original compiler observation, operand, store, return and source-activation
  ownership; Python 3.11 and 3.14 coverage, including fused fast-local loads.
- Original-source capture, binding, namespace, construction, installation and
  operation-condition proof boundaries. Unknown obligations remain explicit.
- Declaration-owned call binding and result retention. A completed operation
  can retain an opaque result without proving its type, identity or protocols.
  Calls and subscriptions share this behavior through inheritance.
- Source-call native entry continuity. This is only a prerequisite: source
  function body execution and cleanup were still explicitly unproved at the
  base checkpoint.
- Associated DSL examples, regression tests, and historical work notes.

## Historical base-checkpoint validation state

Latest completed Python 3.11 broad runs, eight workers and 165-second bounds:

| Suite | Passed | Failed | Skipped | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Tests excluding test_refactor_advisor.py | 6114 | 174 | 69 | 130.38 |
| test_refactor_advisor.py | 760 | 35 | 0 | 60.21 |
| Total | 6874 | 209 | 69 | |

The latest shared-result change removed four failing subscription tests and one
outdated expected-error assertion. The subscription fixtures now reach the
hashing boundary instead of failing during unrelated custom-class construction;
they retain independent real-runtime controls and assert that analysis does not
invoke the hostile hash callback. New negative controls retain rejection of
unknown type, identity, destruction, metadata and cache-identity claims.

Focused shared-result coverage: 172 passed on Python 3.11 (3.56s) and Python 3.14
(4.06s), eight workers with 60-second bounds. The 209 failures have now been
reproduced across their 23 affected files and classified at the shared-boundary
level in
[native_proof_failure_clusters_20260914.md](native_proof_failure_clusters_20260914.md).
The classification is an implementation map, not evidence that a failure is
benign. The exact outstanding test identifiers remain in
[checkpoint_failures_20260914.txt](checkpoint_failures_20260914.txt).

The next source-function activation increment now joins one original invocation
to fresh activation-local storage, exact explicit/default parameter values, its
native entry-to-return receipt, local assignments, and its returned source
value. Its exact final local inventory now proves frame cleanup: the returned
value is retained across frame exit, while every other live local must satisfy
its ordinary release contract. An opaque discarded argument therefore stays
open rather than acquiring an inert lifetime. Distinct and nested calls retain
distinct kernels and locals. Suspended functions, variadic activation
containers, bare returns, and effects outside the activation-local namespace
remain explicitly open. The source/native regression surface passes 2,577 tests
with 50 skips on Python 3.11 in 47.16 seconds using eight workers. The earlier
high-risk activation and assignment cases also pass on Python 3.14. Rerunning
the affected files before the cleanup increment still produced exactly 209
failures, 1328 passes, and 1 skip. This confirms that activation is a
prerequisite, while the next shared blocker remains the declaration-derived
admission of supported native operations.

The first declaration-derived native-operation increment proves the current
source shape and native closure layout of a returned-closure factory without
inventing an operation condition or executing the target function. The exact
``parameter is None`` branch must return one fresh undecorated closure, every
non-selector parameter must be retained by that closure, and the fallback must
apply the same closure to the selector. This admits the standard dataclass
decorator factory while leaving its later decorator application independent.
Negative controls reject preceding effects, released parameters, and executable
closure headers. The original 209-node failure inventory now yields 56 passes
and 153 failures in 60.39 seconds. The focused source/native surface passes 86
tests on Python 3.11 in 6.93 seconds and 85 tests on Python 3.14 in 7.84 seconds,
using eight workers and 60-second bounds.

Reproduce the broad tests from an environment with the dev dependencies installed:

```sh
timeout 165 python -m pytest tests --ignore=tests/test_refactor_advisor.py -q -n 8
timeout 165 python -m pytest tests/test_refactor_advisor.py -q -n 8
```

## Historical pre-integration full-global performance

OpenHCS plus eight external production-library roots, 1014 Python files, tests
excluded, 79 detectors, 16 workers, 165-second bounds:

| Stage | Scan seconds | Command wall seconds |
| --- | ---: | ---: |
| Cold | 47.409 | 50.38 |
| Unchanged cache | 1.243 | 2.54 |
| Novel representative edit | 8.482 | 11.23 |

Original baseline: 56.512 seconds cold and 18.551-19.547 seconds after a novel
edit. All 79 detectors completed with zero omissions and exact findings equality
across the last validated cold/warm/edit reports and the preceding checkpoint.
The source roots were derived from OpenHCS's .gitmodules, not a reduced scan.
Identical findings establish benchmark consistency, not complete detector recall.

These measurements preceded the final shared opaque-result change and are
superseded by the completed integration measurements above.

## Completed requirements

1. The 209 failures were classified and resolved by shared
   proof/consumer/fixture cause without suppressing failures or replacing
   runtime evidence with compact declarations.
2. The source-function activation, parameter/default/variadic transport,
   effects, returned-value, and cleanup proofs are complete while preserving
   creator globals, original evaluation cuts, and distinct invocation locals.
3. Declaration-derived dependency-aware reuse of unaffected completed
   proofs after edits, preserving global reasoning, invalidation, ambiguity and
   cycle correctness, is complete.
4. Isolated cold/warm/novel-edit benchmarks were refreshed after the final
   coherent changes with exact semantic-report equality.
5. The full dependency closure and applicable tests passed. The planned OpenHCS
   domain extraction was not started as part of this checkpoint.

Temporary scripts, caches, and raw logs are outside this commit and are cleaned
after final verification. The historical cross-project pause note and unrelated
original-worktree `uv.lock` edit remain untouched. Dependency declarations are
already present in `pyproject.toml`.
