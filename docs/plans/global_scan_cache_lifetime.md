# Global scan cache-lifetime measurements

Reference record, 12 September 2026. This optimisation batch is validated.

The user's revised working targets are **60 seconds cold** and **at most 20
seconds for a new single-file edit**, retaining the full scope and detector
roster. These supersede the initial approximate 75-second cold target below.
The retained batch completes a cold scan in 56.512 seconds and an unchanged
repeat in 1.376 seconds. A new OpenHCS edit took 18.551 seconds; a new edit in a
larger bundled ObjectState module took 19.547 seconds. Both targets are met in
these representative scan measurements. They do not establish a worst-case
bound for arbitrary edits. Total command times, including startup and output,
were 60.09 seconds cold, 2.46 seconds unchanged, and 22.54/23.57 seconds after
the respective edits.

## Workload

The scan covers OpenHCS production sources and all eight bundled libraries'
`src` trees, discovered from `.gitmodules`. The existing source-discovery policy
excludes tests. There are 1,014 Python files and 79 registered detectors.
Installed third-party dependencies outside these checkouts are not included.

The first optimisation's benchmark uses the disposable copy at
`/home/ts/nra-global-scan-CXVD7T/source`. The live OpenHCS checkout is unchanged.
All timed scans use 16 parse workers, 16 analysis workers, and the `agent` JSON
payload. Times below are the CLI's reported scan duration, excluding interpreter
startup and final payload construction.

| Measurement | Seconds | Cache and completeness |
| --- | ---: | --- |
| Previous cold baseline, live checkout | 152.527 | Empty cache; all 79 detectors |
| Scan-scoped identities, copied checkout | 102.350 | Empty cache; all 79 detectors |
| Unchanged repeat of copied checkout | 1.376 | Exact cache; all 79 detectors |
| One function-body edit, copied checkout | 35.747 | Partial cache reuse; all 79 detectors |
| Engine-signature reuse, original checkout | 94.351 | Empty cache; all 79 detectors |
| Engine-signature reuse, unchanged original checkout | 1.427 | Exact cache; all 79 detectors |
| Engine-signature reuse, new function-body edit on copy | 29.229 | Partial cache reuse; all 79 detectors |
| Return to a previously scanned source state on copy | 3.692 | Partial cache reuse; all 79 detectors |
| On-demand callable proofs, original checkout | 88.034 | Empty cache; all 79 detectors |
| On-demand callable proofs, new function-body edit on copy | 20.429 | Partial cache reuse; all 79 detectors |
| On-demand callable proofs, second new function-body edit on copy | 20.168 | Partial cache reuse; all 79 detectors |
| Initial cache-backed worker handoff, original checkout | 59.700 | Empty cache; all 79 detectors; receipt-index regression subsequently corrected |
| Corrected publication receipts, original checkout | 58.358 | Empty cache; all 79 detectors |
| Receipts and bounded workers, original checkout | 58.710 | Empty cache; all 79 detectors |
| Receipts and bounded workers, unchanged original checkout | 1.423 | Exact cache; all 79 detectors |
| Receipts and bounded workers, new OpenHCS function edit | 19.926 | Partial cache reuse; all 79 detectors |
| Receipts and bounded workers, new ObjectState function edit | 20.912 | Partial cache reuse; all 79 detectors |
| Identity-based alias visits, original checkout | 56.512 | Empty cache; all 79 detectors |
| Identity-based alias visits, unchanged original checkout | 1.376 | Exact cache; all 79 detectors |
| Identity-based alias visits, new OpenHCS function edit | 18.551 | Partial cache reuse; all 79 detectors |
| Identity-based alias visits, new ObjectState function edit | 19.547 | Partial cache reuse; all 79 detectors |

The first cold improvement is 32.9%; the second is 38.1%; on-demand proofs bring
the improvement to 42.3%. The latest implementation improves cold scanning
by 62.9% relative to the original baseline.
No before-change **combined-scope after-edit** timing was
collected, so 35.747 seconds is not an incremental speedup claim. The older
26.490-second edited-source measurement covered OpenHCS alone.

The edit changes `ComponentSet.__bool__` from `bool(self.components)` to
`len(self.components) > 0`. Cold, warm and edited-copy findings match exactly.
Comparison with the original checkout matches after normalising the checkout
prefix and removing path-derived finding, authority and evidence identifiers.
The subsequent original-checkout run matches the baseline finding payload
exactly, including all identifiers, without normalisation.

The latest new edit changes the restored original body to
`return len(self.components) != 0`, which was absent from this cache. Its finding
payload again exactly matches the copied-checkout baseline. The 3.692-second
measurement instead restores a previously scanned body and is not an estimate
for an arbitrary new edit. Revalidating this copied checkout after the engine
implementation changes took 4.354 seconds with partial reuse; that was not an
empty-cache measurement.

After on-demand proof derivation, a new `return not not self.components` body
took 20.429 seconds (3.429 preparation and 17.000 analysis). Its findings match
the copied-checkout cold baseline exactly. This is 30.1% below the preceding
29.229-second new-edit measurement, but remains above the 20-second target.
Refreshing the copy for the changed NRA implementation took 28.373 seconds;
that separate refresh is not the new-edit measurement.
Another new body, `return len(self.components) >= 1`, took 20.168 seconds
(3.384 preparation and 16.784 analysis), again with exact findings equivalence
and all 79 detectors. Both observed new-edit runs remain above 20 seconds;
these two edits do not establish a worst-case bound for arbitrary file edits.

## Cache ownership

`ScanCache` owns memoisation of declaration dependencies, implementation source
signatures, family item schemas and family implementation identities for one
invocation. Nested calls reuse that invocation. Each projection worker starts
its own cache lifetime; standalone shard calls also establish a bounded scope.
Outside a scope, these functions recompute their answers.

AST-bound caches retain their existing per-module cleanup. The cleanup no longer
discards declaration memoisation because that storage belongs to the invocation,
not to process-global LRU wrappers. A subsequent invocation recomputes identities,
including implementation source content whose size and timestamp are unchanged.

The applied 13-stage DSL batch is
[`scan_cache_lifetime.py`](../examples/scan_cache_lifetime.py). It creates the
cache owner, derives imports, replaces declaration decorators and initialises the
projection pool's cache scope. Tests and removal of the unused `lru_cache` import
were edited separately.

`AnalysisEngineSignature.current` and its implementation-file hashing helpers
now use the same lifetime. The further four-stage batch is
[`engine_signature_lifetime.py`](../examples/engine_signature_lifetime.py),
applied in two two-stage sequences. This removes per-source reconstruction of
an invariant engine identity. The edited-source profile before this second
change recorded 1,121 calls to `AnalysisEngineSignature.current` and 178,249
calls to `_module_source_signature` (15.111 profiled seconds in the latter).

## Worker profile

Three real OpenHCS files were profiled with the full 11-family projection roster
and the declared per-module detector set. Before the change, implementation
dependency traversal used 3.351 of 4.290 profiled seconds. The traversal was
repeated after each shard's cleanup.

After the change, traversal calls fell from 36 to 12 across those three shards.
The first shard took 1.574 seconds; the next two took 0.231 and 0.250 seconds.
These are instrumented worker measurements, not full-scan timings.

## Validation and remaining work

- Cache-lifetime, shard-publication and implementation-identity checks:
  15 passed, 2 skipped (Python 3.14-only cases).
- Broader initial cache/CLI/parser run: 174 passed, 7 failed, 2 skipped.
  Three failures match the pre-existing proof-coverage failures. Four tests
  replaced cache producer declarations with test stubs, legitimately changing
  their newly refreshed implementation identities.
- Those four tests now observe native call events without replacing producers;
  all four pass, preserving the original cache-hit and skipped-work assertions.
- After engine-signature reuse, one test also required distinguishing persisted
  input-source signatures from implementation validity checks. It still rejects
  any read of unchanged input files and requires only the edited input to be
  reread; a subsequent invocation may revalidate NRA's own source files.
- Final broader cache/CLI/parser/lifetime run: **180 passed, 3 failed, 2 skipped**
  in 93.31 seconds with eight workers. The three failures are the existing
  public-export, keyed-registry-axis and registry-projection proof gaps.
- Python 3.14 lifetime/publication/implementation checks: **19 passed** in
  4.20 seconds with four workers, including deferred annotations and process
  initialisation under fork, spawn and forkserver.
- Latest edited-source findings and detector coverage match exactly. Further
  optimisation towards the cold-scan target remains active.
- The current NRA production package also completes a fresh global audit in
  22.183 seconds with all 79 detectors and zero findings. This checks scan
  execution; zero findings does not establish that the architecture is debt-free.

## Remaining measured work

The latest cold scan spends 38.388 seconds preparing module facts and
18.124 seconds analysing them. An earlier edited-source profile completed
with the full detector roster; its instrumented timing is not a benchmark.
It identifies repository-wide callable and alias resolution as a major
remaining cost: 616,991 positioned-reference resolutions and 539,814 lexical
target resolutions. Callable-boundary checks also repeatedly traverse retained
escape observations across participant sets.

An exact-query memoisation probe retained 533,697 distinct keys. Within the
carrier/conveyor subsystem it reduced instrumented analysis from 19.879 to
16.497 seconds. This diagnostic was not adopted as a production cache.

Any resolution reuse must retain the original context, source position,
attribute suffix and pending binding-cycle evidence, and must not substitute
local context for the global declaration/ambiguity indices. This has been
inspected as a candidate, not implemented.
Source-backed runtime capture must not be silently memoised as though it were
only a lexical query; the ownership boundary needs to be established first.

### On-demand proof obligations

`CompactCallableComponentAuthorityProof` now owns the repository and immutable
component inputs. Its existing symbol-set observations are derived and retained
on first access. Carrier and conveyor boolean checks short-circuit through their
existing violation declarations; requesting `violations` still evaluates every
obligation. Rejection ordering and complete diagnostic values are unchanged.

The source-checked four-stage DSL batch is recorded in
`/home/ts/nra-global-scan-CXVD7T/lazy-proof-plan-apply.txt`. Focused regression
tests cover input snapshots, skipped work after a decisive rejection, and full
diagnostics requested afterwards. The original-path cold scan matches the
pre-optimisation baseline findings exactly, including identifiers.

The expanded parallel regression group finished with **402 passed, 2 skipped,
3 failed in 94.02 seconds**. All three failures are the previously recorded
public-export, keyed-registry-axis and registry-projection native-proof failures
in `test_analysis_cache.py`; none is a new failure in this test set. The changed
production files and proof tests pass Ruff and `git diff --check`.

The diagnostic probe emitted matching complete proof records and findings for
all 2,028 projections in the two-family carrier/conveyor context. It reported
8.875 seconds for analysis before diagnostic materialisation, versus 19.879
seconds before this change. The instrumented process subsequently reached its
60-second timeout during teardown; this is a diagnostic observation, not a
completed end-to-end benchmark. The production cold run completed normally.

### Cache-backed worker handoff

The complete parent profile (`cold-parent.pstats`) finished with all 79
detectors. Instrumented scan time was 149.478 seconds. Its main-thread profile
recorded 81.259 seconds waiting on worker results, 14.327 seconds in three
explicit garbage collections, and 15.044 seconds deriving source semantic
hashes. Instrumented times identify work; they are not uninstrumented speed
claims.

A diagnostic omitted complete, already-persisted fact bundles from worker
results. The resulting cold scan took 60.144 seconds with exact baseline
findings. The initial production version took 59.700 scan seconds and 63.24
command-wall seconds. Its new-edit run took 20.013 scan seconds and 23.55
command-wall seconds, again with exact findings and all detectors.

The expanded regression run exposed a missing publication receipt: omitting
the whole batch forced incremental scans to reread content signatures instead
of reusing their consolidated index. The corrected handoff retains a
`CompactFamilyProjectionReceipt` containing family and content signature.
`CompactFamilyProjectionBatch` extends that receipt with in-memory facts.
Both register through the same polymorphic operation; signature recording is
owned by one manifest method. Complete cache bundles transfer only receipts;
disabled, oversized or incomplete publications retain the full fact batches.

The corrected source-checked DSL batch is recorded in
`projection-receipts-plan-apply.txt`. All 23 focused publication, compact-scan
and consolidated-signature tests pass. The corrected cold scan completed in
58.358 seconds with exact baseline findings and all 79 detectors.

### Worker budget and intermediate validation

`DetectorAnalysisWorkerPlan` now caps an explicit process budget by available
work items. A single-file rebuild no longer launches 16 workers for one job.
Automatic-worker policy is unchanged. The DSL record is
`bounded-worker-plan-apply.txt`; a regression test forbids pool creation during
an actual one-file incremental rebuild requesting 16 workers, then compares
its findings with uncached analysis. Larger source sets still use 16 workers.

| Workload at this stage | Scan seconds | Complete command wall seconds |
| --- | ---: | ---: |
| Cold, original paths | 58.710 | 62.31 |
| Unchanged warm, original paths | 1.423 | 2.50 |
| New OpenHCS `ComponentSet.__bool__` body | 19.926 | 23.38 |
| New ObjectState `TimeTravelScopeChange.needs_navigation` body | 20.912 | 24.47 |

The OpenHCS edit changes the copied body to `bool(len(self.components))`.
The following single-file edit changes the ObjectState expression from
`bool(self.changed_paths or self.meta_changed_keys)` to
`bool(self.changed_paths) or bool(self.meta_changed_keys)`, in a 2,261-line
module. Both edits were new to the cache. All four findings payloads match
their same-path baselines exactly, including identifiers; all 79 detectors
remain included. The live OpenHCS checkout is clean and unchanged.

The final expanded Python 3.11 regression run reports **442 passed, 2 skipped,
3 pre-existing failures in 76.61 seconds**. Its sole newly exposed failure from
the initial handoff experiment is fixed without weakening the test. A parallel
Python 3.14 run of cache, publication, implementation-identity and callable-proof
tests reports **204 passed in 11.13 seconds**. Ruff and `git diff --check` pass
for the affected production handoff and focused publication tests.

Serial semantic tokenisation remains a measured cold-start opportunity; no
token semantics were changed. At this stage the larger-file edit exceeded 20 seconds.
### Binding-selection experiment

A diagnostic over the same 2,028 carrier/conveyor projections observed 540,004
binding-selection calls and 303,468 repeated queries. Diagnostic subsystem time
changed from 8.592 to 7.806 seconds with query reuse. This did not translate
into sufficient end-to-end benefit: the following new ObjectState edit measured
20.741 scan seconds and 24.53 command-wall seconds, versus 20.912 and 24.47 in
the preceding experiment. Findings were identical and all 79 detectors were
included. Maximum resident size increased from 1,305,616 to 1,357,856 KiB.

The per-flow query cache was removed after this measurement. The retained
change moves the existing empty-mutation check ahead of empty generator walks;
it preserves the same declaration-owned initial-binding result. All 246 focused
product-flow, authority, carrier, conveyor and mutation tests pass in 6.30
seconds after cache removal. The subsequent value-origin benchmarks include
this small retained change. The rejected cache's 20.741-second result is not a
retained speedup.

Apply/removal records are `binding-selection-plan-apply.txt` and
`remove-selection-cache-apply.txt`. Future optimisation must preserve source
positions, branch ambiguity, pending cycles and runtime capture. The next
investigation should identify a larger cost than repeated selection before
adding cache state.

### Value-origin cycle evidence

A full-scope diagnostic attributed 25,326 deep value-graph hashes to alias-cycle
membership checks and another 2,563 to extending their visited set. These checks
need the original binding event and flow, not structural equality of the event's
entire evaluated value graph. The existing exact-alias index already identifies
bindings by their retained event identity.

Value-origin traversal now reuses `CompactBindingVisit`, parameterised by its
context type. Callable binding traversal retains its `CompactFlowContext` type;
value-origin traversal uses `CompactFunctionFlow`. Both retain the actual owners
for the duration of traversal. The graph-value equality and hashing contracts
are unchanged, and no additional query cache is introduced.

The four-module DSL application is recorded in `value-origin-identity-apply.txt`.
A regression that forbids graph hashing during alias-origin lookup failed before
the change and passes afterwards. The initial focused suite reports 239 passed
in 6.37 seconds. A further test confirms that an actual repeated event in the
same flow remains cyclic, while copied events or flows do not create false cycles.

The first empty-cache scan with this change took **56.512 scan seconds**
(38.388 preparation, 18.124 analysis), or **60.09 command-wall seconds**.
All 79 detectors completed and the original-path findings match the initial
baseline exactly. Reports are `identity-cold-live.json` and its `.stderr` file.

After refreshing the disposable copy for the changed implementation (49.451
seconds, not a cold or new-edit measurement), a previously unscanned change to
`TimeTravelScopeChange.needs_navigation` in the 2,261-line ObjectState module
took **19.547 scan seconds** (3.944 preparation, 15.603 analysis), or 23.57
command-wall seconds. The following new `ComponentSet.__bool__` body took
**18.551 scan seconds** (3.028 preparation, 15.523 analysis), or 22.54 command-wall
seconds. Each run changed exactly one file relative to the preceding scan.
Both findings payloads match the copied-checkout baseline exactly, with all
79 detectors included. The unchanged original checkout took **1.376 scan
seconds**, or 2.46 command-wall seconds, using the complete exact cache.

Reports are `identity-refresh-copy.json`, `identity-external-edit-copy.json`,
`identity-openhcs-edit-copy.json`, and `identity-warm-live.json`, each with its
corresponding `.stderr` timing. These measurements meet the revised scan
targets, not a 20-second whole-command target or a bound for arbitrary changes.

Remaining opportunities are serial semantic tokenisation in source preparation
and repository-wide callable/alias proof traversal after edits. The earlier
instrumented cold profile attributed 15.044 seconds to semantic source hashing;
that is profiling evidence, not an additive estimate of achievable wall-time
savings. Further work must preserve source-signature semantics and the full
context, position and cycle evidence of each proof query. The rejected query
cache is not part of the retained implementation.

Final expanded validation reports **574 passed, 2 skipped, and the same three
pre-existing failures in 98.17 seconds**, using eight workers on Python 3.11.
The failures remain the keyed-registry-axis, registry-projection and explicit
public-export proof tests in `test_analysis_cache.py`. Python 3.14 reports
**266 passed in 8.62 seconds** using four workers. Reports are
`identity-regressions.txt` and `identity-python314.txt`.

Ruff passes on the cycle-identity production modules and regression file, and
`git diff --check` passes. The broader touched-module lint run reports 42
existing `ast_tools.py` issues; linting its HEAD version also reports those
42 issues. This batch does not remove its existing public re-exports.
The live OpenHCS checkout remains unchanged. The changes are not committed
separately because they depend on the unfinished native-proof worktree.

Raw reports are under `/home/ts/nra-global-scan-CXVD7T`. The earlier baseline is
`/home/ts/nra-openhcs-external-gM1tsz/cold.json`. Worker profiles are
`/home/ts/nra-shard-profile-a7f8e9nu/shards.pstats` (before) and
`/home/ts/nra-shard-profile-3ela9k9l/shards.pstats` (after).

The existing native-proof worktree remains unfinished and unpublished. These
performance results do not establish completion of that separate batch.
