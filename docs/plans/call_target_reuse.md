# Shared call-target projection

Reference record, 13 September 2026. This change is part of the uncommitted
performance and native-proof batch based on `5f96c76`.

## Ownership

`ProductFlowRepository.function_call_resolutions` retains each original call,
context and resolved target. `resolved_product_constructions` now derives its
construction view from those records. The target declaration's existing
`resolve_construction` method still owns classification and argument handling.
Previously the two aggregate views independently resolved the same call targets.

The change adds no cache, resolver or source-signature policy. Recursive target
resolution retains its existing pending-binding, attribute-suffix, ambiguity and
cycle handling. Point queries through `resolve_product_construction` remain
available. The reuse belongs to one repository snapshot; dependency-aware reuse
of completed proofs across source edits remains unfinished.

The applied DSL batch is `.codex-temp/product_call_projection_reuse.py`.

## Measurements

The workload is the full OpenHCS plus eight-library production scope: 1,014 files
and 79 detectors, with tests excluded. All six comparison runs complete without
detector omissions and produce identical findings.

| Run | Baseline scan seconds | Derived scan seconds | Baseline wall seconds | Derived wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Cold projection build | 52.853 | 52.839 | 56.29 | 56.41 |
| First new edit, derived first | 19.774 | 17.934 | 23.36 | 21.55 |
| Second new edit, baseline first | 18.976 | 17.072 | 22.56 | 20.67 |

The paired edit reduction averages 1.872 scan seconds, approximately 9.7%, and
1.85 command-wall seconds. Cold performance is effectively unchanged in this
comparison. The second edit's analysis phase decreases from 15.089 to 13.159
seconds while preparation remains similar, 3.887 versus 3.913 seconds. Peak RSS
stays near 1,327,000 KiB for both edited variants. End-to-end time remains above
20 seconds.

The unchanged edited-input repeat takes 1.375 scan seconds and 2.45 command-wall
seconds, with 136,404 KiB peak RSS, identical findings and complete exact-cache
coverage (`target-read-optimised-warm.json` and `.stderr`).

Before the refactor, the refreshed edit profile records 116,085 calls to
`resolve_product_construction`, taking 9.828 cumulative seconds under
instrumentation. The complete instrumented scan takes 41.126 seconds, or 51.30
command-wall seconds. Instrumented durations are diagnostic costs, not baseline
timings for the uninstrumented comparison above.

## Comparison boundaries

Both variants use the same disposable source copy under
`/home/ts/nra-global-scan-CXVD7T/source`. The diagnostic driver
`.codex-temp/benchmark_product_call_reuse.py` reinstates the exact preceding
aggregate only when invoked with `--baseline`. Its implementation is in the
importable `.codex-temp/product_call_baseline.py`, allowing source-signature
inspection to identify it. Production files are not rewritten for baseline runs.
An initial diagnostic using a function from `__main__` was rejected during
implementation-identity construction, before a scan completed.

The baseline and derived variants have independent caches:
`cache-target-read-baseline-Rw8w7Y` and `cache-target-read-optimised-e9afxM` under
the report directory. Both completed their initial projection builds before the
copy's `ObjectState` `needs_navigation` predicate changed. Its unchanged first
term is `bool(self.changed_paths)`; the second term follows this sequence:

1. Initial input: `0 < (len(self.meta_changed_keys) + 0)`.
2. First new input: `(len(self.meta_changed_keys) + 0) > 0`.
3. Second new input: `0 != (len(self.meta_changed_keys) + 0)`.

Each variant first encounters each changed input in its own seeded cache. Live
OpenHCS is unchanged. Benchmarks and tests run sequentially. Scan workers remain
16 and the absolute timeout remains 165 seconds.

Reports are `target-read-{baseline,optimised}-{cold,edit,edit2}.json` with `.stderr`
timing records under `/home/ts/nra-global-scan-CXVD7T`. The refreshed diagnostic
profile is `target-read-edit-current.pstats`, with its JSON and timing records.

## Validation

Focused product-flow, cycle, carrier and conveyor checks report 191 passed on
Python 3.11 and 3.14 in 5.75 and 6.70 seconds. After adding direct-versus-derived
comparison coverage, all nine new cases pass separately on both versions in
1.97 and 2.19 seconds. They cover original call identity, either query order,
fresh-repository edits, duplicate-module ambiguity, lifetime, unchanged direct
query behaviour and positional-construction rejection. No production admission
rule was relaxed to make a fixture pass.

The full support suite reports **5,375 passed, 148 failed and 68 skipped** in
119.25 seconds. Core reports **760 passed and 35 failed** in 67.19 seconds.
Combined coverage is **6,135 passed, 183 failed and 68 skipped**. Exact failed-node
sets match the preceding decorator-input checkpoint. Reports are
`target-read-support-311.txt` and `target-read-core-311.txt`. Native-proof
completion is still required before publication.
