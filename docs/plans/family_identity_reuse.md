# Scan-scoped family identity reuse

Reference record, 13 September 2026. These changes are part of the unfinished
performance/native-proof batch based on `5f96c76`; nothing has been committed
or pushed.

## Measured cost

The fresh empty-cache profile completes the 1,014-file OpenHCS plus eight-library
production scan with all 79 detectors and unchanged findings. It records 115.347
instrumented scan seconds and 122.67 command-wall seconds. Profiling adds overhead;
these durations are not ordinary scan benchmarks.

Repeated collected-family cache-token computation accounts for 38,532 calls and
5.559 cumulative seconds. Most callers generate payload and content-signature
paths. Explicit garbage collection accounts for three calls and 0.299 seconds,
so this profile does not justify changing collection policy. Global product-call
and carrier/conveyor analysis remain substantial consumers of source resolution.

The profile is `constant-current-cold.pstats`, with
`constant-current-cold-profile.json` and its `.stderr` timing record, under
`/home/ts/nra-global-scan-CXVD7T`.

## Ownership and lifetime

`CollectedFamilySchemaIdentity.from_family` and
`CollectedFamilyCacheContext.identity` use the existing `ScanCache` invocation
lifetime. Equal immutable source/family inputs reuse the same identity within
that lifetime; a later invocation derives fresh identities. Calls outside a
scope remain uncached. This is the same lifetime already used for family item
schema and implementation signatures.

`CollectedFamilyCacheIdentity.cache_token` is derived once per immutable identity.
Its representation and hash algorithm are unchanged. The identity inherits
`StoredDataclassState`, so serialisation retains declared fields and omits the
derived token. `dataclasses.replace` likewise creates an identity without an old
token. Source signatures, module/path identity, family schema, implementation
identity, Python version and focused demand remain part of the key.

The authored DSL batches `.codex-temp/family_identity_lifetime.py` and
`.codex-temp/family_identity_transport.py` applied the change. An initial
method-targeted decorator replacement failed simulation without writes; the
successful plan used the enclosing declaration and exact method header.

This reuses cache-key computation, not completed semantic proofs across edits.
The latter still requires dependency evidence and remains unfinished.

## Isolated cold comparison

The diagnostic runner `.codex-temp/benchmark_family_identity.py` restores the
original uncached identity methods and token property for the baseline process.
The production source, analysed inputs and key contents are identical between
variants. Each cold run uses a separate empty cache.

| Variant | Preparation | Analysis | Scan | Command wall | Peak RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncached identity baseline | 34.444 s | 18.167 s | 52.611 s | 56.18 s | 1,324,308 |
| Scan-scoped identity reuse | 34.439 s | 16.482 s | 50.921 s | 54.53 s | 1,328,100 |
| Unchanged optimised run | 0 s | 1.437 s | 1.437 s | 2.48 s | 135,400 |

Cold scan time decreases by 1.690 seconds, about 3.2%, in this paired observation.
All 79 detectors complete without omissions and findings match exactly. The
unchanged run is an exact cache hit. Reports use the
`family-identity-baseline-cold`, `family-identity-optimised-cold` and
`family-identity-optimised-warm` prefixes, with `.json` and `.stderr` files in
the report directory above.

The earlier constant-join implementation refresh took 69.86 command-wall seconds.
The fresh uncached baseline here is 56.18 seconds; that earlier wall observation
alone did not establish a regression attributable to constant comparison.

## New-edit comparison

The disposable source copy under `/home/ts/nra-global-scan-CXVD7T/source` was
scanned before editing, then its completed cache was copied into separate A/B
branches. The sole edited file is
`external/ObjectState/src/objectstate/object_state_registry.py` in that copy.
Its `needs_navigation` predicate changed from `0 < len(self.meta_changed_keys)`
to `len(self.meta_changed_keys) > 0`, then to `0 != len(self.meta_changed_keys)`.
Both variants saw each new input for the first time. The second comparison ran
the optimised variant first to reverse the initial run order. The live OpenHCS
checkout was not modified.

| New input | Uncached scan | Optimised scan | Uncached wall | Optimised wall |
| --- | ---: | ---: | ---: | ---: |
| First edit | 19.786 s | 19.325 s | 23.28 s | 22.96 s |
| Second edit, reversed run order | 19.531 s | 18.787 s | 23.08 s | 22.52 s |

Each run completes all 79 detectors with exact compact-global coverage. Findings
match between variants and the seed result. The mean scan-time saving across
these two observations is 0.603 seconds, about 3.1%. Peak RSS remains close:
1,319,388/1,323,732 KiB for the first baseline/optimised pair and
1,319,336/1,323,608 KiB for the second. These small gains are not evidence of
completed-proof reuse; global analysis still runs where its dependencies changed.
Wall times remain above 20 seconds.

Reports are `family-identity-edit-seed.json`,
`family-identity-baseline-edit.json`, `family-identity-optimised-edit.json`,
`family-identity-baseline-edit2.json` and `family-identity-optimised-edit2.json`,
with their `.stderr` timing records.

## Validation

Focused suites pass **40 tests** on each of Python 3.11 and 3.14 in 9.85 and
9.34 seconds respectively. Checks include scope nesting and cleanup, fresh
unscoped and next-invocation identities, declaration changes, source/demand/path
invalidation, unchanged token format, replacement and pickle behaviour, and the
existing publication and analysis-invocation contracts. Reports are
`family-identity-final-focused-311.txt` and
`family-identity-final-focused-314.txt`.

The sequential bounded Python 3.11 regression suites report **6,048 passed,
186 failed and 68 skipped**. Support reports 5,288 passed, 151 failed and
68 skipped in 117.15 seconds; core reports 760 passed and 35 failed in
66.74 seconds. Exact failed-node sets match the constant-content checkpoint
in both suites. The eight additional passing tests exercise identity lifetime;
they do not close a native-proof obligation.

Reports are `family-identity-support-311.txt` and
`family-identity-core-311.txt`. Both completed within their 165-second bounds
using eight workers. The publication batch remains unfinished.
