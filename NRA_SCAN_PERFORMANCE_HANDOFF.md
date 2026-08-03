# NRA scan-performance handoff

## 2026-08-03 large-repository update

The normal file-focused edit loop is now bounded and explicitly partial on a
cold auto-context scan.  For ``--json --json-payload loop`` with inferred
package context, NRA parses requested files one at a time, runs the 182
per-module detectors, releases AST-bound caches at each module boundary, and
reports the 70 omitted context-dependent detectors in ``scan_status``.  Full,
agent, and explicit ``--context-root`` scans retain exact contextual behavior.

The surviving five-file DQDock reproduction now completes with the same 158
local findings:

| State | Wall time | Peak RSS | Result |
|---|---:|---:|---|
| Cold focused loop before bounded lane | >20 s | about 1.1 GB | Deadline |
| First bounded local lane | 12.88 s | 158 MB | Complete partial payload |
| Module streaming + detector short-circuit | 10.48 s | 134 MB | Complete partial payload |

The remaining exact-mode floor is representation-level: parsing DQDock's 842
production modules without running any detector takes 14.48 seconds and peaks
at about 955 MB.  Reducing that path further requires compact contextual
projections or streamed indexes rather than another cache lookup optimization.

Persistent cache retention is also bounded to 128 recent roots, 4 GiB total,
2 GiB per active root, and four recent exact semantic-graph generations.  The
first live maintenance pass removed 8,112 abandoned derived-cache roots and
7.02 GB of payloads, reducing the default cache home from 12 GB to 3.4 GB.

Relevant pushed commits are ``0b867fa`` (symlink identities and deadline
process exit), ``acbaa44`` (bounded focused loop), ``591f4f8`` (cache
retention), ``3822abb`` (companion detector short-circuit), and ``d90a165``
(module-boundary AST cache release).

Verification collected 936 tests.  The broad run reported 934 passes and two
fixture-root failures because its pytest base directory was intentionally
nested inside the checkout; both failures passed when rerun from an external
temporary root, which restores the repository-root assumption those detectors
exercise.

## Publication update

This document preserves the original slow-scan reproduction and measurements.
The audited publication set now adds an end-to-end scan deadline,
checkout-relocatable cache identities, scope-bound focused aggregate reuse,
semantic AST/signature memoization, detector preparation reuse, and focused
report scheduling.  The measurements below remain historical and must not be
read as a completed before/after benchmark.

Publication verification collected 924 tests: 920 passed and four failed.  The
same four failures reproduce from the pre-change `origin/main` baseline
(`test_module_cli_synthesizes_authoring_selectors`,
`test_module_cli_synthesizes_and_preflights_finding_backed_plan`,
`test_module_cli_codemod_fixpoint_dry_run_does_not_apply`, and
`test_builds_composed_subsystem_plan`).  The two dirty-worktree regressions in
focused cache reuse and semantic-carrier selector ordering were repaired and
pass their focused gates.

The remaining-work list later in this document is retained as the original
diagnostic handoff, not as a claim that every item remains unimplemented.

## Scope and current state

This handoff is for optimizing the focused Nominal Refactor Advisor (NRA) scan
used while repairing DQDock.  The scan is still too slow for the edit/inspect
loop.  One safe inventory fix is present in the dirty NRA worktree, but the
dominant contextual-signature latency and scan-budget overrun remain unresolved.

Do not assume that every dirty NRA file belongs to this task.  The worktree
already contained a large, weeks-long set of uncommitted performance changes
before this lane began.  Preserve all of them and inspect ownership before
editing or reverting anything.

## Exact reproduction

Set `NRA_REPO`, `NRA_PYTHON`, and `DQDOCK_REPO` to the local NRA checkout,
Python interpreter, and DQDock receipt checkout, respectively, then run:

```sh
PYTHONPATH="$NRA_REPO" \
"$NRA_PYTHON" -m nominal_refactor_advisor \
  --json \
  --json-payload loop \
  --context-root "$DQDOCK_REPO" \
  --scan-budget-seconds 45 \
  --parse-workers 1 \
  --analysis-workers 1 \
  "$DQDOCK_REPO/dq_dock_engine/docking/autodock426_mixed_row_action_provenance.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/autodock426_mixed_row_action_exclusion_receipt.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/definitive_refinement_sparse_projection.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/definitive_refinement_certified_sparse_runtime.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/definitive_refinement.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/definitive_refinement_local_stages.py" \
  "$DQDOCK_REPO/dq_dock_engine/docking/definitive_refinement_local_refinement.py"
```

Use an external wall-clock/RSS monitor.  Do not trust an orchestration-tool
yield as process completion: the initial command appeared to return after about
28 seconds while its child PID remained alive and CPU-bound.

## Measurements

| State | Wall time | RSS | Result |
|---|---:|---:|---|
| Before hidden-descendant pruning, early sample | 25–30 s | 1.71 GB | Still running |
| Before hidden-descendant pruning, terminal observation | >107 s | about 1.9 GB | No JSON; process terminated |
| After hidden-descendant pruning, observed sample | >68 s | 1.31 GB at 68 s | Still running; no JSON at observation |

The post-fix result is censored: no completed JSON or trustworthy completion
time was obtained.  Do not convert it into a claimed speedup.  It does show a
lower observed RSS at the comparable still-running state, while also proving
that the 45-second budget is not enforced end-to-end.

## Root cause found

An interrupt stack placed the live process in
`PrivateReferenceModuleIndex.from_module` while computing a contextual detector
signature.

The context-root inventory also incorrectly treated immutable provenance
snapshots as live Python context:

- Before: 1,184 Python files, 58.34 MB.
- Hidden `.dqdock-provenance/generations/...` descendants alone: 207 Python
  files, 23,912,157 bytes.
- After generic hidden-descendant pruning: 977 files, 34.43 MB.
- Reduction: 17.5% of files and 41.0% of source bytes.

The inventory bug amplified every context-wide signature/index pass, but it was
not the sole bottleneck.

## Implemented change

Known task-owned edits are:

- `nominal_refactor_advisor/ast_tools.py`
  - `PythonSourcePathPolicy.allows_directory_name` now rejects hidden
    descendant directories generically.
  - An explicitly supplied hidden directory remains scannable as a root; only
    descent into hidden children of a broader root is pruned.
- `tests/test_refactor_advisor.py`
  - The production-tree inventory test covers hidden source-history exclusion.
  - A regression test proves that an explicitly requested hidden root remains
    scannable.

The worker reported six focused tests passing for this change.  The two
inventory-specific tests were independently re-run after this document was
written: `2 passed, 783 deselected in 0.84s`.

All other currently modified files may predate this lane:

```text
nominal_refactor_advisor/analysis.py
nominal_refactor_advisor/analysis_cache.py
nominal_refactor_advisor/ast_tools.py
nominal_refactor_advisor/cli.py
nominal_refactor_advisor/detectors/_abstraction_reuse.py
nominal_refactor_advisor/detectors/_base.py
nominal_refactor_advisor/detectors/_role_surface_drift.py
nominal_refactor_advisor/detectors/_runtime.py
nominal_refactor_advisor/detectors/_semantic_descent.py
nominal_refactor_advisor/detectors/_surface.py
nominal_refactor_advisor/semantic_descent.py
tests/test_analysis_cache.py
tests/test_refactor_advisor.py
tests/test_semantic_descent.py
tests/test_context_projection_reuse.py (untracked)
```

Inspect `git diff` before modifying overlapping code.  Do not reset, checkout,
delete, or rewrite the dirty worktree.

## Remaining work, in priority order

1. Make `--scan-budget-seconds` an actual end-to-end deadline.  The current
   focused command remained CPU-bound past both 45 and 68 seconds.  Budget
   checks must cover contextual signature/index construction, not only detector
   execution after setup.
2. Profile and cache/deduplicate
   `PrivateReferenceModuleIndex.from_module` and its contextual signature
   facets.  The interrupt stack identifies this as the current hot boundary.
3. Ensure each semantic AST/source signature facet is computed once per module
   per source version and shared by every contextual detector that consumes it.
4. Separate analysis scope from report scope: the seven requested files need
   repository context, but local/per-module detectors should run only for report
   modules.  Existing dirty-worktree changes appear to be moving in this
   direction; audit rather than duplicate them.
5. Add process-lifecycle regression coverage: JSON mode must not let the parent
   report completion while a scan child remains alive.
6. Repeat the exact reproduction with stage timings and peak RSS.  Require a
   completed JSON result before claiming a final before/after speedup.

## Suggested next commands

```sh
cd "$NRA_REPO"
git status --short
git diff --stat
git diff -- nominal_refactor_advisor/ast_tools.py tests/test_refactor_advisor.py
```

```sh
cd "$NRA_REPO"
"$NRA_PYTHON" -m pytest -q \
  tests/test_refactor_advisor.py \
  -k 'parse_python_modules_can_skip_test_trees or parse_python_modules_can_explicitly_scan_hidden_root'
```

For the next timed reproduction, wrap the exact command above with
`/usr/bin/time -v` and also monitor the spawned PID.  Impose an external hard
timeout until NRA's own budget is proven reliable.

## DQDock coordination

Do not run the expensive NRA reproduction while a DQDock benchmark is active.
NRA work must remain in the NRA checkout; it must not edit the authoritative
DQDock receipt checkout.
