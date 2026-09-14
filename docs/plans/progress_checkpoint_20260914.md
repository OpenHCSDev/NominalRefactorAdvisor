# Native-proof and scan-performance checkpoint, 14 September 2026

This is an explicitly unfinished development checkpoint, requested by the user
after discussing the remaining failures. It is not a release or a declaration
that the full goal is complete. The checkpoint branch preserves the complete
source/test foundation together; it must not be merged as a passing release.
Base commit: `5f96c76358fe7c3be53af8d325bc513d693168e4`.

## Integration update

The native-admission and dependency-aware source-proof reuse branches are now
combined on `checkpoint/native-proof-integration-20260914`. The historical 209
failures classified below are resolved on that integration branch. The complete
Python 3.11 suite passes under the prescribed eight-worker split:

| Suite | Passed | Failed | Skipped | Seconds |
| --- | ---: | ---: | ---: | ---: |
| `test_refactor_advisor.py` | 795 | 0 | 0 | 62.32 |
| Remaining tests | 6,337 | 0 | 70 | 150.43 |
| **Total** | **7,132** | **0** | **70** | |

The integrated proof retains exact native definition-application evidence for
the supported dataclass transformation, activation-specific returned parameter
identity, conclusive partial non-injectivity evidence, and exact unchanged
module-local proof owners across virtual source edits. Global dependency queries
are recomputed from the complete projected module set.

One attempted speedup cached validation-bearing class-body receipts and allowed
a warm query to hide later source mutation. The existing negative controls
caught that regression; those receipts again revalidate their original source
on every access. Compiler-owned annotation writes instead use their exact
compact target and declared key to avoid replaying unrelated completed class
bodies. Rebound `__annotations__` values retain the ordinary unresolved item
write path.

The remaining completion work is fresh cold, unchanged-warm, and novel-edit
measurement of the full OpenHCS plus external-production-library scan, exact
finding comparison, final full-diff review, and the applicable publication
gates. The older validation sections below remain as historical checkpoint
evidence rather than current status.

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

## Validation state

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

## Last validated full-global performance

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

These performance measurements precede the final shared opaque-result change.
Its full-global benchmark remains outstanding. Do not report these timings as
fresh measurements of the exact checkpoint revision.

## Remaining completion requirements

1. Classify and resolve the 209 failures by shared proof/consumer/fixture cause.
   Do not silence failures or replace runtime evidence with compact declarations.
2. Finish the needed source-function activation, parameter/default/variadic
   transport, effects, returned-value and cleanup proofs. Preserve creator
   globals, original evaluation cuts, and distinct invocation locals.
3. Complete declaration-derived dependency-aware reuse of unaffected completed
   proofs after edits, preserving global reasoning, invalidation, ambiguity and
   cycle correctness.
4. Refresh profiling and run isolated cold/warm/novel-edit benchmarks after the
   final coherent changes; preserve exact findings and bounded memory/lifetimes.
5. Review the full dependency closure, pass the applicable tests and publication
   gates, then merge/publish a completed increment. Do not start the planned
   OpenHCS domain extraction as part of this checkpoint.

Local scratch scripts, caches, raw logs, the historical cross-project pause note,
and the unrelated user uv.lock edit are intentionally outside this commit.
They remain on disk; no user work was discarded. Dependency declarations are
already present in pyproject.toml; lockfile reconciliation remains for publication.
