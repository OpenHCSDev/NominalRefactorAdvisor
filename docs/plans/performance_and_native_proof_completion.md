# Performance and native-proof completion

Historical progress notes from 13 September 2026. For the current unfinished
development checkpoint, validation counts and remaining obligations, see
[the 14 September checkpoint](progress_checkpoint_20260914.md). Measurements
below describe their individual increments, not the latest combined revision.

The [keyword-call checkpoint](native_proof_completion_status.md#original-keyword-call-capture)
adds shared original keyword metadata and operand correspondence on Python 3.11
and 3.14, including class headers. Broad checks report **6,525 passed, 153 failed
and 69 skipped**, with unchanged failed nodes. Full-global cold/unchanged/novel-edit
commands retain exact findings in 55.76/2.47/20.36 seconds. Native construction,
returned-class identity and registry effects remain unproved. The preceding
[abstract-member checkpoint](native_proof_completion_status.md#native-abstract-member-inspection)
records the shared static-MRO validation and fresh-function member inspection.

The [immutable family representation checkpoint](native_proof_completion_status.md#immutable-family-representation-reuse)
removes repeated rendering of the scan-owned family schema without changing its
text or cache keys. The edit profile reduces representation work from 3.764 to
0.599 seconds. Full-global cold/unchanged/novel-edit commands take
54.10/2.48/20.37 seconds, with exact findings and all 79 detectors retained.
The 20-second edited command-wall target, cross-edit completed-proof reuse and
the native-proof completion batch remain open.
Broad validation reports **6,482 passed, 153 failed and 69 skipped**, with the
same failed-node sets as the preceding checkpoint.

The [item-storage checkpoint](native_proof_completion_status.md#item-storage-completion-and-overwrite-queries)
adds original item-store completion through the existing mutation visitor and
shares consuming native transfers with discard completion. Slot release uses
the actual previous value's existing release contract. Latest-value lookup
avoids recursively replaying superseded stores; a 32-write fixture resolves one
matching value, with original-cut, interference and cycle checks retained.
The shared prefix hoist and traversal change were applied using the DSL.
Final broad validation reports **6,475 passed, 153 failed and 69 skipped**, with
unchanged failed-node sets. The native-proof batch remains unfinished.
Every detector retains exact findings across cold/unchanged/novel-edit runs in
52.479/1.401/17.956 scan seconds (56.10/2.50/21.60 command-wall seconds).
The local lookup improvement does not establish a whole-scan speedup.

The [source completion checkpoint](native_proof_completion_status.md#source-completion-boundaries)
separates completed source operations from installed bindings. Discarded
expressions now join original native returns without a fabricated store;
expression completion and import cleanup share native discard lookup. Focused
checks pass on Python 3.11 and 3.14. Completed broad split runs report 6,432
passed, 153 failed and 69 skipped, removing three failed cases from the previous
checkpoint and adding none. A 16-worker combined run timed out at 165 seconds
without a summary; it does not replace the completed split-run evidence.
Cold/unchanged/novel-edit scans retain exact findings with every detector in
51.667/1.442/17.679 scan seconds (55.25/2.51/21.31 command-wall seconds).
The larger native-proof batch and cross-edit proof reuse remain open. Nothing
has been committed or pushed, and the live OpenHCS source is unchanged.

The [ordered sequence checkpoint](native_proof_completion_status.md#ordered-sequence-construction)
shares tuple/list operand capture and joins literal lists, including compiler
constant expansion, through original native/source receipts. Focused checks pass
on Python 3.11 and 3.14. Final broad checks report 6,407 passed, 156 failed and 69
skipped: the list-valued metadata failure is closed, with no new failed nodes.
The full native-proof batch remains unfinished.
All-detector cold/unchanged/novel-edit scans retain exact findings in
53.443/1.384/18.043 scan seconds (57.12/2.46/21.72 command-wall seconds).
These verify performance after the proof extension; they do not establish a new
optimisation or complete cross-edit proof reuse.

The [source import checkpoint](native_proof_completion_status.md#source-import-installation-and-cleanup)
joins original requests and source bindings to native stores and module cleanup.
It reuses import-declaration dispatch, the existing importer/source admission
and the prefix's preserved module associations. Focused checks pass on Python
3.11 and 3.14. Runtime relative imports now use the original global package read,
not the static catalogue path; the regression control also executes the failing
Python fixture. Final broad checks report 6,364 passed, 157 failed and 69 skipped:
two existing imported-member completion failures are closed, with no new failed
nodes. The larger native-proof batch remains unfinished.
All-detector scans retain exact findings in 53.342/1.403/18.238 scan seconds
for cold/unchanged/novel-edit input (57.06/2.50/21.97 command-wall seconds).
These are proof-coverage regression measurements; the edited command-wall
target and cross-edit completed-proof reuse remain unfinished.

The [shared operand segments checkpoint](native_proof_completion_status.md#shared-operand-segments)
replaces per-store value walks and a separate continuing operand walk with one
original segment authority. Native import transfers now retain their module
operand across member stores. Source import installation and module release
were still unfinished at that checkpoint; the source import checkpoint records
their subsequent implementation. Focused checks pass on Python 3.11 and
3.14. Final broad checks report 6,324 passed, 159 failed and 69 skipped, with
unchanged failed-node sets. The reference records the coverage and remaining
boundaries; the batch remains unpublished.
All-detector cold/warm/novel-edit scans retain exact findings in
52.409/1.422/17.851 scan seconds (55.98/2.52/21.53 command-wall seconds).
The small changes are treated as broadly neutral performance, not an attributed
whole-scan optimisation. The edit command-wall target remains unmet.

The [class-result installation checkpoint](native_proof_completion_status.md#class-result-installation)
joins native builder/body operands to the stored class result and its return.
The class creation site derives from the existing body receipt, and builder
identity reuses the source entry's builtins-only lookup. Nested class tails now
use the common definition-result installation contract. Broad checks report
6,306 passed, 159 failed and 69 skipped: seven earlier failures are closed,
with no new failed nodes. The reference records the strengthened nested-class
control and the retained-stack/release obligations still needed for imports.
Cold/warm/novel-edit scans complete every detector with exact findings in
52.572/1.410/18.116 scan seconds (56.18/2.49/21.89 command-wall seconds).
These are coverage-regression measurements, not a new optimisation claim.

The [decorated-result installation checkpoint](native_proof_completion_status.md#decorated-result-installation)
joins original callees, implicit arguments and final wrapper stores. Shared
definition availability is factored separately from raw object identity, and
keyword-only defaults retain their compiler dictionary inputs. Final broad
validation reports 6,280 passed, 166 failed and 68 skipped: 16 earlier support
failures are closed, and the core failed-node set is unchanged. Five tests for
the older blanket map rejection now verify the finer transfer-versus-source-proof
boundary. The larger native-proof batch and cross-edit proof reuse remain open.
Final cold/warm/novel-edit scans complete every detector and retain exact findings
in 52.853/1.387/18.044 scan seconds (56.51/2.44/21.74 command-wall seconds).
These measure the added proof coverage; they are not a new speedup claim.

The [function-operand integration](native_proof_completion_status.md#function-creation-operands)
retains original function inputs, decorator-call operands and the final store in
the shared native walk. Snapshot transport preserves long operand chains and
shared identities. Broad checks report 6,229 passed, 182 failed and 68 skipped,
with unchanged failed-node sets. This earlier checkpoint preceded decorated-result
source installation; cross-edit completed-proof reuse remains unfinished.
Cold, warm and novel-edit global scans retain exact findings and all 79 detectors
at 53.057, 1.382 and 17.824 scan seconds respectively (56.59, 2.42 and 21.47
command-wall seconds). The cold/edit measurements are slower than the preceding
checkpoint; they are regression checks, not a claimed optimisation.

The [native store convergence checkpoint](native_proof_completion_status.md#converging-native-store-observations)
joins compatible creation/operand observations through the existing store owner.
Broad checks report 6,210 passed, 182 failed and 68 skipped, with unchanged
failed-node sets. The empty-cache full scan retains all 79 detectors and exact
findings in 50.284 scan seconds / 53.90 command-wall seconds. This prerequisite
does not yet complete function-creation operand or decorator installation proof.
Warm and novel-edit checks retain exact findings and complete detector coverage
at 1.355 / 17.380 scan seconds, or 2.42 / 20.97 command-wall seconds.

The [ordinary attribute checkpoint](native_proof_completion_status.md#ordinary-attribute-operands)
closes the native operand/store gap for admitted attribute reads using the
existing source lookup authority. Final broad checks report 6,202 passed,
182 failed and 68 skipped: one existing class-completion failure is removed and
no new failed tests are introduced. The full cold scan retains exact findings
and all 79 detectors in 50.974 scan seconds / 54.57 command-wall seconds.
Warm and novel-edit runs retain the same findings: 1.404 / 17.442 scan seconds,
or 2.46 / 21.06 command-wall seconds. The reference records full measurements.

The [implicit native call checkpoint](native_implicit_call_slots.md) records the
preceding operand-transfer and DAG-hashing changes. Broad checks report 6,186 passed,
183 failed and 68 skipped, with unchanged failed-node sets. Whole-package scans
retain all 79 detectors and exact findings: 51.635 seconds cold, 1.431 seconds
warm and 17.486 seconds after a new edit. Corresponding command-wall times are
55.23, 2.49 and 21.04 seconds. This is not yet a passing publication batch.

The [scalar dictionary slot checkpoint](scalar_dictionary_slots.md) records the
preceding native-proof boundary change: complete storage views use the existing
scalar contract, and registry comparison consumes that same authority. Broad
validation reports 6,175 passed, 183 failed and 68 skipped; the remaining
failed-node sets are unchanged.

The [family identity measurements](family_identity_reuse.md) record the latest
full-scan profile and isolated cold comparison for scan-scoped cache-key reuse.

The [shared call-target checkpoint](call_target_reuse.md) reports 6,135 passing
tests and 183 outstanding failures across completed, sequential bounded runs.
Two paired novel-edit comparisons reduce scan time by an average of 1.872 seconds
without a new cache. The latest edited run takes 17.072 scan seconds and 20.67
command-wall seconds, with all 79 detectors and unchanged findings. Cold
performance is neutral in that paired comparison.
Ordinary class completion now consumes the final native namespace; remaining
source/native operation boundaries and explicit native-metaclass construction
still prevent publication. Earlier measurements below describe their named
implementation stages, not a passing current batch.

## Acceptance boundary

The user requested investigation of parallel source preparation, dependency-aware
proof reuse after edits, and graph lifetime/cleanup, followed by commit and push.
The subsequent instruction explicitly includes completion of the large unfinished
native-proof batch before publication. Merely including that batch's dependencies
does not satisfy the request.

The [previous performance batch](global_scan_cache_lifetime.md) establishes the
56.512-second cold and 18.551/19.547-second new-edit baselines. The
[native-proof status](native_proof_completion_status.md) describes its existing
boundaries and outstanding obligations. Completion of that work remains required;
performance improvement alone does not discharge those obligations.

## Refreshed profile

The full 1,014-file, 79-detector OpenHCS plus eight-library production scan
completed under instrumentation with unchanged findings. Instrumented durations
are diagnostic costs, not ordinary command timings.

- Semantic hashing: 14.856 seconds across 1,014 sources.
- Explicit garbage collection: 0.303 seconds across three calls, versus 14.327
  seconds in the older profile. Further collection-policy changes are not
  justified by this refreshed measurement.
- Carrier/conveyor component analysis remains a substantial consumer of
  repository-wide callable and alias resolution.

The profile is `opportunities-cold.pstats`; its JSON report is
`opportunities-cold-profile.json` under `/home/ts/nra-global-scan-CXVD7T`.

## Parallel semantic preparation

`CachedSourceFileSignature.with_semantic_hash` derives lexical identity against
the record's exact source hash. `SourceFileSignatureCache` identifies missing
work and validates completed records before publishing them. Workers receive
immutable records, not independently writable cache copies. A changed source
or mismatched publication authority is rejected.

`DetectorAnalysisWorkerPlan.map` owns the shared process execution for semantic
preparation and projection construction. Worker counts remain bounded by actual
pending tasks; one changed source does not start idle workers. Exact cache hits
return before preparation, and cached lexical identities are not recomputed.
The existing tokenisation and source-signature definitions are unchanged.

| Workload | Scan seconds | Command-wall seconds |
| --- | ---: | ---: |
| Empty-cache original checkout | 51.615 | 55.23 |
| Unchanged original checkout | 1.433 | 2.49 |
| Implementation refresh on disposable copy | 4.735 | 7.27 |
| New ObjectState function-body edit | 19.500 | 23.54 |

All 79 detectors completed; cold and edited findings match their respective
same-path baselines exactly. The cold result is 8.7% below the previous 56.512
seconds. The implementation refresh reuses existing facts and is not a cold
or novel-edit timing. The new edit changes `needs_navigation` in the 2,261-line
ObjectState module from `len(self.meta_changed_keys) != 0` to
`0 < len(self.meta_changed_keys)`, with the rest of the function unchanged.
Only the disposable copy was edited; the live OpenHCS checkout is unchanged.

The DSL applications are recorded in `parallel-semantic-apply.txt` and
`semantic-signature-owner-apply.txt`. Initial validation reports 36 passed in
4.16 seconds, covering actual sequential/process execution, parent-owned cache
publication, changed-source rejection, warm reuse and bounded worker budgets.
The focused checks also include the existing cache-lifetime and projection
publication regressions. Broader and cross-version validation remains in progress.

The shared local-source selection now also drives preparation and per-module
finding inclusion, removing the repeated report-scope predicate. The two-stage
DSL application is `source-report-selection-apply.txt`. Python 3.14 preparation,
cache-lifetime, publication and scope checks report **38 passed in 6.95 seconds**
(`parallel-semantic-python314.txt`). Broader proof validation remains in progress.

## Native-proof work and remaining performance work

### Direct API invocation lifetime

The timed-out test runs exposed an omission in the previous scan-cache change:
CLI scans had an invocation scope, but direct analysis APIs repeatedly rebuilt
implementation-dependency identities outside that scope. A representative
`analyze_path` test took 19.55 seconds without profiling and exceeded 60 seconds
under profiling. A diagnostic that supplied the existing scope passed in 2.97
profiled seconds. Its timeout stack showed repeated annotation-dependency
traversal through `AnalysisEngineSignature.current`.

The public analysis and cache-loading entry points now declare the existing
`ScanCache.scope` boundary. Nested calls share it; subsequent invocations start
fresh. Both detector worker initialisers establish worker-owned lifetimes.
No additional cache has been introduced. The DSL application is
`analysis-api-lifetime-apply.txt`.

The same direct API test now passes in **1.12 seconds**, without profiling.
Focused Python 3.11 validation reports **28 passed in 11.64 seconds**; Python
3.14 reports **27 passed in 10.66 seconds**. Tests check one engine computation
per nominal signature declaration per invocation, shared nested scopes,
next-invocation freshness, exception cleanup and worker isolation. The two
signature subclasses correctly have distinct cached identities; they are not
collapsed into one engine record.

The full core suite now completes in **60.69 seconds**, with **760 passed and
35 failed**. Its failed-test set exactly matches the earlier native-proof core
baseline. Reports are `native-core-scoped.txt`, `analysis-api-lifetime-tests-after.txt`
and `analysis-api-lifetime-python314.txt`. The support suite completes with
**5,038 passed, 105 failed and 66 skipped in 105.95 seconds**
(`native-support-scoped.txt`). Its failed-test set also exactly matches the
earlier native-proof support baseline. Combined current coverage is **5,798
passed, 140 failed and 66 skipped**, across the two bounded runs. There are no
new failed tests in that comparison; the 140 proof failures remain unresolved.

## Native continuation integration

The [native-proof reference](native_proof_completion_status.md#native-store-continuations)
records the new original-store-to-native-return boundary. Primitive return
behaviour remains on its enum declaration and uses the existing operand
interpreter. Each uninterrupted suffix has one walk and one shared receipt;
there is no per-store suffix interpretation or target-code execution.

The isolated empty-cache production scan after this addition completed in
**51.804 seconds scan time / 55.36 seconds command-wall time**, with 1,308,880 KiB
maximum resident memory. Preparation took 33.837 seconds and analysis 17.967.
All 79 detectors completed, zero were omitted, and the grouped findings exactly
match `parallel-semantic-cold-live.json`. Reports are
`native-return-cold-live.json` and `native-return-cold-live.stderr` under
`/home/ts/nra-global-scan-CXVD7T`. This verifies global coverage and preserves the
earlier cold improvement; it is not evidence that the unfinished native-proof
batch or cross-edit proof reuse is complete.

The cold run preceded the final separation of receipt membership validation
from transfer-view materialisation. The final source and native focused suites
pass on Python 3.11 and 3.14; the native-proof reference records their exact
coverage and the unchanged broad-suite failure sets. No commit or push has
been made for this incomplete publication batch.

Reusing that cache after the final implementation change caused a full cache
miss, not a warm run: **51.515 seconds scan / 55.77 seconds command-wall**, with
the same findings and complete detector coverage. It verifies the final
implementation while demonstrating conservative implementation-signature
invalidation. This is distinct from changing an analysed source file.
The report is `native-return-refresh-live.json`.

The subsequent unchanged fresh-process scan reports an exact cache hit:
**1.413 seconds scan / 2.47 seconds command-wall**, all 79 detectors and the
same findings (`native-return-warm-live.json`).

## Shared function-store continuation scan

After factoring the continuation observer above scalar and function stores and
restoring native receipt snapshot equality through `DataclassGraphValue`, a new
empty-cache scan completed in **51.610 seconds scan / 55.21 seconds command-wall**.
Preparation took 33.273 seconds and analysis 18.337; maximum resident memory was
1,308,812 KiB. All 79 detectors completed with zero omissions and exactly the
same findings payload, representing 215 raw findings in 180 boundary-evidence
groups (`shared-return-cold-live.json` and `.stderr`). This
run preceded the final cache-to-source owner check; it is not a benchmark of
that later boundary change. All reports remain in
`/home/ts/nra-global-scan-CXVD7T`.

After the terminal source/native join and shared original-production inventory,
another isolated empty-cache scan completed in **52.099 seconds scan / 55.66
seconds command-wall**: 33.726 seconds preparation and 18.373 seconds analysis,
with maximum resident memory of 1,308,928 KiB. All 79 detectors completed, with
zero omissions and an identical findings payload (`prepared-tail-cold-live.json`
and `.stderr`). This is a small increase from the preceding measurement, not a
performance improvement claim.

The unchanged fresh-process repeat was **1.430 seconds scan / 2.54 seconds
command-wall**, with validated exact-cache coverage and the identical payload
(`prepared-tail-warm-live.json` and `.stderr`). These runs cover the current
cache-to-source owner check as well as the native-proof additions. No live
OpenHCS source files were changed. Novel-edit timings have not been rerun after
these additions.

## Native namespace interpretation cache refresh

After sharing entry/tail interpretation and joining compiler cells, the existing
cache was reused to check implementation invalidation. The scan correctly
reported an analysis-cache miss and rebuilt the full 79-detector result in
**30.060 seconds scan / 33.82 seconds command-wall**. Preparation took 11.675
seconds and analysis 18.385; maximum resident memory was 1,311,120 KiB. Its
findings payload exactly matches the preceding scan. This is an implementation
refresh using retained source artefacts, not an empty-cache or novel-source-edit
benchmark (`native-namespace-refresh-live.json` and `.stderr`).

The unchanged fresh-process repeat reports an exact cache hit in **1.388 seconds
scan / 2.43 seconds command-wall**, with all 79 detectors and identical findings
(`native-namespace-warm-live.json` and `.stderr`). These runs used
`cache-prepared-tail-live`; the live OpenHCS worktree remained clean. Cold and
novel-edit measurements after this factoring remain pending.

After declaration-selected store and event-free body continuations, the same
cache again reported an implementation miss: **51.796 seconds scan / 56.48
seconds command-wall**, with 33.396 seconds preparation and 18.400 seconds
analysis. Maximum resident memory was 1,319,772 KiB. All 79 detectors completed,
zero were omitted and the findings payload exactly matches the earlier namespace
scan (`entry-tail-refresh-live.json`, `.stderr`). This is an implementation
refresh, not an isolated empty-cache or source-edit measurement.

The unchanged fresh-process repeat reports an exact cache hit in **1.402 seconds
scan / 2.48 seconds command-wall**, with identical findings and complete detector
coverage (`entry-tail-warm-live.json`, `.stderr`). The current proof batch has
not been committed or pushed. Cold and novel-edit measurements remain pending.

The general adjacent-store and original assignment-read join retains full scan
coverage. Its implementation refresh reports **52.277 seconds scan / 57.41
seconds command-wall**, with 33.830 seconds preparation, 18.447 seconds analysis
and 1,327,984 KiB maximum resident memory. All 79 detectors completed with zero
omissions and the exact preceding findings payload
(`assignment-join-refresh-live.json`, `.stderr`). This reused the existing cache
after implementation invalidation; it is not an empty-cache or novel-edit run.

The unchanged fresh-process repeat reports **1.413 seconds scan / 2.50 seconds
command-wall**, with validated exact-cache coverage and identical findings
(`assignment-join-warm-live.json`, `.stderr`). These measurements verify
compatibility of the proof extension, not an additional speedup claim.

## Construction integration scan

After ordinary construction began consuming the native namespace and global
function installation gained its existing shared return continuation, the full
production scan completed in **52.681 seconds scan / 59.14 seconds command-wall**.
Preparation took 34.272 seconds and analysis 18.409; maximum resident memory was
1,339,272 KiB. All 79 detectors completed, with no omissions and the exact same
findings payload as `assignment-join-refresh-live.json`.

This reused `cache-prepared-tail-live` after implementation invalidation. It is
an implementation-refresh measurement, not a new cold or novel-edit result.
The unchanged fresh-process repeat took **1.427 seconds scan / 2.53 seconds
command-wall**, with an exact cache hit, complete coverage and identical findings.
Reports are `native-construction-refresh-live.json` and
`native-construction-warm-live.json`, with corresponding `.stderr` files, under
`/home/ts/nra-global-scan-CXVD7T`. OpenHCS remained unmodified.

## Source-symbol point queries

The complete NRA production snapshot contains 141 source modules. A benchmark
adds one virtual two-function module and simulates an argument edit to its call,
with all 142 files retained in scope. The benchmark leaves repository files
unchanged and does not execute the virtual target.

Before this change, the simulation took **16.665 seconds**, and the whole command
took **24.89 seconds**, with 592,240 KiB maximum resident memory. Its profile
attributes 24.218 instrumented seconds to collecting all 142 source-flow graphs
and another 10.405 to building class projections for the import-origin query.
The profile durations include instrumentation overhead.

`ProductFlowRepository` now separates point queries from complete enumeration.
The compact implementation reuses its existing global indexes. The source
implementation visits all possible module owners of the requested symbol,
including duplicate names and overlapping module/class prefixes, then checks the
original declaration rows. Source queries reuse `ModuleNominalBindingView`'s
canonical projection cache; no new completed-proof cache is introduced.

Module-entry queries use `SourceProductFlowProjection.module_context`. Import
origins use `RepositoryModuleBindingProof.star_import_origins_for` through MRO;
that owner now returns no selected origins for an ambiguous module name.
Transitive exposure remains on the existing complete repository index. The
unused eager `sources_by_module_name` view has been removed.

The initial after-measurement reports **0.043 seconds** for simulation and
**6.93 seconds** for the command, with 221,628 KiB maximum resident memory.
Snapshot preparation took 4.411 seconds, versus 4.332 before. This measures a
point-query DSL workflow, not full-detector scan speed or cross-edit proof reuse.
Reports are `repository-call-edit-before.txt`,
`repository-call-edit-before.pstats`, and `repository-call-edit-after.txt` under
`/home/ts/nra-global-scan-CXVD7T`. The initial script also printed an observation
count from the parent snapshot; replay uses a derived snapshot, so that field
does not measure the modules visited and has been removed from the benchmark.

Focused checks report **85 passed on Python 3.11** and **85 passed on Python
3.14** (`lazy-source-query-final-311.txt`, `lazy-source-query-final-314.txt`).
They cover original-object identity, duplicate module rows and declarations,
module/class name overlaps, post-edit ambiguity, imports and source-preserving
DSL application. The support run before the final module-entry delegation
reports **5,182 passed, 158 failed and 68 skipped in 112.21 seconds**; its exact
failed-node set is unchanged (`lazy-source-symbol-support-311.txt`).

After delegating module entry to its existing source owner, the repeated
single-stage measurement reports **0.046 seconds simulation / 6.97 seconds
command-wall**, with 221,572 KiB maximum resident memory. A three-stage sequence
reports **0.122 seconds simulation / 6.89 seconds command-wall**, with 222,236 KiB
maximum resident memory. Each stage re-proves its edit against the preceding
stage's virtual output. Reports are `repository-call-edit-final.txt` and
`repository-call-edit-three-stages.txt`. The final core suite reports **760
passed and 35 failed in 65.42 seconds**, with its exact preceding failure set
(`lazy-source-query-core-311.txt`).

The full OpenHCS plus eight-library implementation refresh reports **32.826
seconds scan / 38.47 seconds command-wall**, with 12.511 seconds preparation,
20.315 seconds analysis and 1,340,544 KiB maximum resident memory. All 79 detectors
completed, with zero omissions and the exact preceding findings payload.
This reused retained source artefacts after implementation invalidation, so the
lower total does not establish a cold-scan speedup. The unchanged repeat reports
**1.404 seconds scan / 2.49 seconds command-wall**, with an exact cache hit and
identical complete findings. Reports are `lazy-source-query-refresh-live.json`
and `lazy-source-query-warm-live.json`, with corresponding `.stderr` files.
Final cold and novel-source-edit measurements remain outstanding.

### Earlier incomplete validation attempts

Two simultaneous eight-worker full-suite partitions reached their respective
165-second limits without final summaries. They are incomplete attempts, not
passing runs or a new definitive failure count. Reports are
`native-refresh-core.txt` and `native-refresh-support.txt`. Subsequent runs use
sequential bounded partitions to establish the current failure set.
A subsequent unpartitioned core-only attempt also reached its 165-second limit
without a summary (`native-current-core.txt`). The next grouping derives four
disjoint partitions from pytest's collected 795 core node identifiers; it retains
every parametrisation. Partition outputs are `native-core-part0.txt` onwards.
The first partition also timed out; a subsequent 60-second diagnostic enabled
verbose node reporting and timeout stacks. These attempts led to the API
lifetime investigation above. They are not counted as completed coverage.

### Remaining obligations

Cross-edit proof reuse must account for positive dependencies and negative
queries, including new consumers, ambiguous declarations and escaping references.
The existing proof obligations query those global facts. Source-path-only
invalidation is insufficient. Native-proof completion and the dependency
contracts it establishes precede claiming safe cross-edit reuse.

Publication requires the coherent source/test/documentation batch, completed
regression checks, review of generated artefacts and remote commit verification.
Scratch diagnostics and unrelated user artefacts are excluded. Discussion of
the planned OpenHCS refactoring follows this work; implementation of that
refactoring has not been started as part of this goal.
