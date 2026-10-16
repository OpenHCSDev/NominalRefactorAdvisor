# Requested detector roster ownership

## Scope

Follow-up to PR #1 (`fix/exact-cache-detector-subset-isolation`). The PR supplies
the actual detector roster to aggregate cache identities. This follow-up moves
the duplicated omitted/full versus explicit-roster decision onto the existing
`DetectorRegistrySignature.current` factory. Identity constructors delegate to
that owner; no additional registry, coverage field or cache authority is added.

`None` resolves the current full registry. An explicit empty tuple remains an
empty roster. Resolution happens before the cached `from_detector_types` call,
so a registry change inside a live `ScanCache.scope` changes the default request
instead of reusing a signature cached under `None`.

## DSL and validation

- Authored recipe: `docs/examples/requested_detector_roster_owner.py`.
- Baseline: `6ae22999a14f5792aa364297506954115829266c`.
- Four clean stages over a 602-file supplied source snapshot, immutable original
  input, and one production file in the revision-checked write set. Signature
  editing and scoped target patches automate the chosen migration; the authored
  body is not independently proved behaviorally equivalent by syntax preflight.
- Black formats only the changed production line ranges; the recipe and test
  file are formatted normally.
- Four new direct factory controls fail before the change with an unexpected
  keyword argument. Five new controls cover full/empty/subset requests and an
  actual registry mutation inside a scan-cache scope.
- Python 3.11: 191 cache/CLI/scan-cache tests pass with eight workers in 23.06 s.
- Python 3.14: the same 191 tests pass with eight workers in 21.63 s; 98 existing
  multiprocessing `fork()` deprecation warnings remain.
- Before/after repository scans both cover 79 detectors with zero omissions in
  `exact_compact_global` mode. The same three existing documentation-recipe mirror
  findings remain; this is coverage/consistency evidence, not whole-body equivalence.

Run the affected suites from the checkout in an environment containing the
project and `pytest-xdist`:

```bash
timeout 60 python -m pytest tests/test_exact_cache_coverage.py \
  tests/test_analysis_cache.py tests/test_cached_scan_status.py \
  tests/test_scan_cache.py -n 8
```

The complete local test suite was not rerun for this bounded follow-up. Merge
requires the updated exact-head hosted matrix, including all six Python/OS test
jobs and docs/wheel smoke. Earlier green checks belong to the preceding PR head.

The separate skill-publication packaging repair explicitly discovers only the
NRA Python package. Its isolated sdist/wheel build and editable install pass;
the wheel contains all 141 production Python modules byte-for-byte and excludes
the non-Python `skills` namespace. That main revision is brought into the PR
before hosted validation.

Logs and scan/replay evidence for this run are retained under
`/home/ts/nra-roster-validation-w5pDzO`. The unrelated native-behavior goal and
the original dirty worktree remain untouched.
