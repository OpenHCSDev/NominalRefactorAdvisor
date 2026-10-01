# Bounded complete FULL audit

Status: repair in progress. No production repair or complete audit is claimed.
Tracking: [issue 11](https://github.com/OpenHCSDev/NominalRefactorAdvisor/issues/11).
Base: `0844525ecaba93e090a064a4ae4914466b2dae60` (current remote main,
29 September 2026). Worktree: `fix/bounded-full-audit-20260929`.

## Required behavior

Retain the complete declared production and dependency source corpus, every
registered detector, authenticated source identities and existing native proof
admission. `--json --raw-findings --json-payload full` must retain source index,
observation graph and fibers, semantic-descent graph and refactor gate,
finding-recipe plan, and raw-record evidence. A compact `agent` or `summary`
payload is not a substitute. Increasing timeouts is not the repair.

The motivating workload is OpenHCS plus its eight bundled libraries' production
source roots. Tests follow the existing discovery policy. Scientific images,
held-out annotations and notebook reference answers are not inputs to this task.

## Source evidence at the base

- `cli.py:754`, `JsonPayloadSections.compact_analysis_compatible`, excludes
  source-index, observations/fibers and finding-recipe demand.
- `cli.py:3911`, the compact analysis branch, therefore does not serve FULL.
- `cli.py:3960`, the FULL branch, eagerly obtains all parsed modules before
  `analyze_modules_with_cache`.
- `ast_tools.py:2738`, `parse_python_module_roots`, retains every parsed result
  in a list. `PythonModuleRootParser.parsed_source_paths` also returns a list.
- `cli.py:1197`, `JsonPayloadBuilder`, consumes parsed modules for observations,
  semantic descent and `CodemodSourceSnapshot.from_modules` for source-index and
  recipe exports. These obligations cannot simply be removed from eligibility.
- `analysis.py:273`, `DetectorAnalysisWorkerPlan`, uses no process pool with one
  requested analysis worker. The baseline requests one parse and analysis worker.

The shared checkout at `52fe8b4` is preserved but is not the repair base: it
predates merged R1 detectors and subsequent native-proof changes. Historical
79-detector receipts do not establish current main's detector coverage.
The current source's `default_detector_types_for_analysis()` returns 85
registered detectors. Import location was verified against the isolated worktree.

## Current baseline and resource contract

One FULL scan ran against the isolated current source and the same nine
analysis/context roots. Its separate receipt is
`scan-recovery-full-2048-cli-restart-current-main.*`; it does not overwrite the
earlier held-prelaunch receipt. Native scan budget is 160 seconds, guarded shell
shard 165 seconds, sampled scanner RSS ceiling 2 GiB. Start requires 11 GiB
available and memory PSI full at most 1 percent for ten seconds. Stop on less
than 8 GiB available, sampled RSS over the ceiling, or sustained pressure.
This is a diagnostic envelope, not a new production default or a proof of
constant-space behavior.

Actual result: the guard stopped the scanner at 94 seconds when sampled RSS
reached 2,105,892 KiB, above the 2,097,152 KiB ceiling. Exit 143 reflects that
guard termination, not a native deadline. The FULL JSON is empty and the owned
scanner is absent after cleanup. Thus no complete coverage or full/raw R1 result
is available. This is a reproduced resource failure on current main with the
original source context, not just the earlier held-prelaunch condition.

An attempted 20-second `py-spy` attach to this owned scanner was denied by the
operating system. It exited 1 and produced no profile. No privilege or kernel
policy was changed. The scan result is not presented as controlled timing.

## Ownership and validation work

Main owns integration, the baseline and public tracking. Darwin reviews existing
source/fact/snapshot lifetime owners; Nash independently reviews full-export,
cache-invalidation and proof tests. Both initial reviews completed read-only
against the same production base. Neither produced a production patch. Nash's
test worktree is clean; no test patch or test run was made. Both workers are now
closed to reduce the fleet after the desktop resource check warned about disk
headroom and swap. The two owned worktrees were moved intact under `/home/ts/wt`.

The user explicitly directed that NRA must not block OpenHCS work. Use the
lightweight census/overlay scripts and bounded canonical Python queries for
local ownership investigation, followed by actual native regression and MCP
checks. Keep the FULL repair independent, without relabeling local evidence as
global audit completion. The next OpenHCS defect is the already reproduced
PolyStore POINT archive codec issue, tracked in PolyStore issue 12 / draft PR13.

The census and overlay completed against current NRA production sources with
no parse warning. A bounded Python query through `parse_python_modules` and
`ModuleSyntaxIndex` inspected all 24 original classes in `roi.py` and 29 in
`roi_converters.py`, exposing the existing ImageJ converter family. It exited 0
in 0.94 seconds at 64,300 KiB maximum RSS. This is original-syntax inspection,
not resolved native binding/MRO, invocation, codec or biological proof.

The architecture review identified the decisive FULL lifetime problem: cached
`CodemodSourceContext` still carries AST-bearing `ClassFamilyIndex`, and recipe
compilation/simulation retains complete parsed snapshots. Extending bounded
projection/source-index/observation owners is insufficient unless native recipe
and proof materialization also gains a bounded original-identity lifetime.
Do not merely enable compact eligibility or defer the same eager snapshot.
The independent test review identified missing combined FULL cold/warm export,
complete-status, dependency-invalidation and native-proof rejection regressions.

Prefer one source owner with bounded native materialization and derived exports
over another source roster, parser, proof cache or metadata mirror. Exact target
ownership remains under investigation. Preserve fail-closed absent, ambiguous,
changed and unsupported source/proof results. Tooling meets the same ownership
standard as product code (refactor-audit AGENT-8); a partially migrated second
route is not a completed repair (AGENT-2).

Acceptance requires real all-detector/global completion, native FULL export
behavior, raw-record evidence, authenticated context invalidation, and bounded
wall time/RSS on this workload. Focused tests, cache hits or parse-only success
must be reported separately, not promoted to full audit or biological proof.
