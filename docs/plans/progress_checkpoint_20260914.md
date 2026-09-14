# Native-proof and scan-performance checkpoint, 14 September 2026

This is an explicitly unfinished development checkpoint, requested by the user
after discussing the remaining failures. It is not a release or a declaration
that the full goal is complete. The checkpoint branch preserves the complete
source/test foundation together; it must not be merged as a passing release.
Base commit: `5f96c76358fe7c3be53af8d325bc513d693168e4`.

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
  function body execution and cleanup are still explicitly unproved.
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
value. Distinct and nested calls retain distinct kernels and locals. Suspended
functions, variadic activation containers, bare returns, and effects outside the
activation-local namespace remain explicitly open. Surrounding source/native
proof coverage passes 544 tests with 5 skips on Python 3.11; the 60 highest-risk
activation and assignment cases pass on Python 3.14. Rerunning the affected
files still produces exactly 209 failures, 1328 passes, and 1 skip. This confirms
that activation is a prerequisite, while the next shared blocker remains the
declaration-derived admission of supported native operations.

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
