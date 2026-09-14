# Full-package scan performance evidence

Reference record, 11 September 2026. The workload is the OpenHCS production
package at `e867013a8eb188edcc63b5b0cdfd06f42a99b409`, analysed by the current
working NRA checkout based on `5f96c76358fe7c3be53af8d325bc513d693168e4`.
The checkout already contains an unfinished native-proof refactoring batch.

## Scope and results

The source discovery policy admits 674 Python modules. It excludes
`processing/backends/analysis/test_simple_implementation.py`; tests and submodule
checkouts are not part of this package-root workload. Every successful scan
below reports all 79 registered detectors, zero omitted detectors, and 188
findings. This is completeness for the declared detector roster, not a claim
that detectors recognise every possible architectural issue.

| Run | Cache condition | Workers | Reported scan seconds | Result |
| --- | --- | ---: | ---: | --- |
| Before scope correction | Existing cache, explicit equal report/context roots | 8 | 150.000 | Deadline; incomplete |
| Scope correction | Analysis miss, existing family facts | 8 | 109.557 | Complete |
| Immediate repeat | Exact analysis-cache hit | 8 | 0.888 | Complete |
| Scope correction, isolated cache | Fresh cache | 8 | 153.166 | Complete |
| Bounded collection correction | Analysis/family invalidation after source edits | 8 | 146.130 | Complete |
| Final isolated run | Fresh cache, completed performance batch | 16 | 131.383 | Complete |
| Final immediate repeat | Exact analysis-cache hit | 16 | 0.894 | Complete |

These are CLI `timing.total_seconds`, not total interpreter startup wall time.
The absolute CLI deadline covered each process run. Diagnostic runs overlapped
with profiling or regression checks; they establish completion, not controlled
speedup ratios. The final cold run had no competing profiler or regression suite.
Its findings and finding counts exactly match both its warm repeat and the
earlier isolated 8-worker cold run.

## Owning changes

`AnalysisPathScope` normalises a report boundary covering all analysis roots to
an unrestricted report. Previously, explicitly passing the same root twice
selected the focused prepass, serially materialising package-wide facts before
the global join. Scope normalisation now determines both execution and cache
identity. Genuinely narrower reports retain their full analysis context.

`BoundedCompactProjectionManifest.findings_by_detector` reuses
`suspend_cyclic_gc`, the existing parsing-batch collection control. The bounded
join suspends allocation-triggered cyclic collection while keeping reference
counting and its existing explicit collection boundaries. Caller policy is
restored on success and exceptions. No detector or proof check is removed.

An isolated load of the complete cached product-flow family measured:

| Automatic cyclic collection | Load seconds | Seconds inside GC callbacks | Modules | Cache misses |
| --- | ---: | ---: | ---: | ---: |
| Enabled | 26.940 | 24.255 | 674 | 0 |
| Suspended | 2.651 | 0.000 | 674 | 0 |

`parse_python_module_roots` deduplicates admitted paths before parsing, retaining
the first parser's module identity and lexical path. The previous implementation
parsed overlapping roots repeatedly and discarded duplicate results afterwards.
The reproduction `(pipeline_directory, pipeline_directory, compiler_file)`
performed 23 parses for 11 unique files. The new regressions cover overlapping
roots, symlinks and first-root identity.

## Regression evidence

- New scope and bounded-collection regressions plus live-root checks: 17 passed.
- New overlapping-root regressions: 3 passed; existing selected parser checks:
  3 passed.
- Selected compact/focused cache checks: 7 passed.
- Final combined scope, collection-lifetime, overlapping-root and live-root
  regression run: 20 passed in 6.90 seconds with four test workers.
- Broader cache and scan-status run: 153 passed, 3 failed, stopped after the
  third failure. All three failures also reproduce with this performance batch's
  scope normalisation, bounded-join decorator and pre-parse deduplication disabled.

The existing failing contracts in `tests/test_analysis_cache.py` are:

- `test_compact_module_public_export_contract_is_exact_and_fails_closed`
- `test_compact_keyed_registry_axis_facts_preserve_axis_semantics`
- `test_compact_registry_projection_candidates_preserve_projection_semantics`

They concern declaration/export proof coverage, not scan completeness or the new
collection lifetime. No expected proof result was relaxed to make them pass.

## Reproduction and raw evidence

The final scan command uses a dedicated cache directory. Its first invocation
is cold only when that directory has not previously been used:

```sh
NRA_CACHE_HOME=/home/ts/nra-scan-perf-eN5nyu/final-cache \
  .venv/bin/nominal-refactor-advisor \
  /home/ts/code/projects/openhcs/openhcs \
  --context-root /home/ts/code/projects/openhcs/openhcs \
  --json --json-payload agent \
  --parse-workers 16 --analysis-workers 16 --scan-budget-seconds 165
```

Raw measurements are under `/home/ts/nra-scan-perf-eN5nyu/`. Initial deadline
reports are under `/var/tmp/openhcs-domain-audit-kAEAjH/`; profiler statistics
are `/var/tmp/nra-full-scan-profile.pstats` and
`/var/tmp/nra-full-fixed-profile.pstats`. Scratch reproduction scripts are in
`.codex-temp/`. Timing evidence does not imply the unfinished checkout is ready
to publish.

## Implications for batched refactoring

Full-package context now fits the agreed broad-scan budget, and exact unchanged
context can be reused cheaply. The existing projected-state DSL can therefore
start from package-wide findings rather than substituting a local scan. No DSL
transformation or proof contract changed in this performance batch.

Remaining measured optimisation candidates include repeated scan-engine
signature construction and deep value hashing during binding resolution.
Any reuse must be owned by a scan or immutable snapshot; process-global
memoisation must not hide source changes. The advisor's ability to recognise
runtime-generated enum families remains a separate semantic coverage question.

## Follow-up: cold and edited-source scans

A second bounded performance batch used a disposable copy of the same 674-module
package at `/home/ts/nra-edit-latency-Y2OuOK/openhcs`. The live OpenHCS source was
not edited. Both baselines and improvements used 16 workers, the full detector
roster, and the `agent` JSON payload.

| Workload | Before, seconds | After, seconds |
| --- | ---: | ---: |
| Empty-cache full scan | 130.591 | 106.844 |
| One function-body edit with existing cache | 40.265 | 26.490 |

The final unchanged-source scan took 0.868 seconds with exact cache coverage.

The edited-source comparison uses the same change to `ComponentSet.__bool__`:
`return bool(self.components)` becomes `return len(self.components) > 0`.
An intermediate rescan restoring the original body took 26.496 seconds. The
improved cold run used another equivalent body, `return self.components != ()`.
All completed runs reported 79 covered detectors, zero omitted detectors, and
the same 188 findings. Exact finding payloads were compared, not just counts.
The edited-source status is complete; cache status `partial` describes reuse,
not partial detector coverage.

The changes in this batch are:

- `CompactFunctionFlow.exact_alias_for` owns alias lookup by the retained binding
  event's identity. Consumers no longer structurally hash the event's full value
  graph. Equal but distinct event snapshots cannot acquire another event's alias.
  The existing stored-dataclass serialization contract rebuilds the derived
  index against restored graph identities.
- `build_compact_projection_shard` reuses the full-family collector's published
  content signature. It no longer validates, serializes and signs the same full
  family a second time. Demanded and source-native projections retain their own
  publication paths; absent signature metadata remains recoverable from facts.
- Bounded cyclic-collection control also covers direct family materialisation,
  including semantic-graph consumers outside the detector join.

The separate cold-shard audit confirmed 11 publications for 11 requested
families, all from the owning collector. Binding/capture tests passed (84), as
did the broader product-flow, carrier, conveyor and alias-codemod tests (278).
New tests cover event identity, lookup without graph hashing, pickle restoration
and exactly one full-family cache publication.
The final combined regression run passed 384 tests in 8.34 seconds using eight
workers. The selected projection/cache checks also passed (6 tests).

Raw before/after reports and regression output are in
`/home/ts/nra-edit-latency-Y2OuOK/`; the improved runs use its `optimised-cache`
directory. The CLI command otherwise matches the earlier command, with the
copied package as both target and context. Cold scans used a fresh directory and
a 165-second budget; edited-source scans used a 60-second budget. No competing
test suite or profiler ran during the final cold and edited-source measurements.

An attempted incremental run under cProfile and repeated faulthandler sampling
exited 139 before writing statistics. It supplied no valid timing evidence;
the cause remains undiagnosed. Ordinary CLI runs on the same copied package
completed successfully.

Further cold optimisation would benefit from explicit cache lifetimes.
`release_module_analysis_memory` currently clears declaration-level family
identity caches along with AST-bound caches. A representative two-shard probe
recomputed the implementation-module traversal for all 11 families on each
shard. Retaining those caches requires a per-scan invalidation boundary that
still observes implementation changes between scans. This batch does not add
method-name exemptions or disable AST cleanup.
