# Handoff: NRA strengthening, 2026-11-17/18

NRA (nominal refactor advisor) is the product. agent-comms, toad and textual are benches for finding where it fails. Every failure becomes NRA work, never a hand refactor of a bench (Tristan, 2026-11-17; rules/slop.md "NRA is the product").

## State right now

- NRA main: 0dae25a (#478). All of today's work is merged except the items under "Open work".
- Bench checkouts: `/home/user/agent-comms` and `/home/user/toad` were reset to their mains (106007daa, 579f1092) on branch `ccr-604fcf75-vmgoks`, and pushed. Textual is at f5ec9e7, its main.
- **Bench rule, learned the hard way:** fetching doesn't move a checkout, and a leftover `origin/HEAD` ref can point at an old commit. Every scan runs on `git worktree add --detach X origin/main`, created right after `git fetch`. Today's toad results came partly from code 48 commits old, and one "bug" (toad `composer_surround.py` `describe`) had already been deleted by toad #2063.
- **Host rule:** heavy jobs (scans, searches, full suites) run one at a time:
  - scans: `flock $SCRATCH/nra_scan_slot2.lock bash -c 'ulimit -v 9000000; …'`
  - test runs: `flock $SCRATCH/nra_light.lock bash -c 'ulimit -v 6000000; …'`

  Two heavy jobs at once ran the 16 GB host out of memory twice.

## Merged today (NRA)

| PR | What |
|---|---|
| #461 | moves rewrite downstream checkouts' imports (consumer roots from pyproject) |
| #462 | goal search frees explored branches |
| #463 | class references follow re-exports; move_member carries module imports; unannotated subjects typed |
| #464 | a name a module imports is owned by its source module (`import m`, `import m as x`, `from m import Y as Z`). Cuts went from 17 freeing 21 edges to 24 freeing 35; hub self-reads from 412 to 1 |
| #465 | cut read-back refusals typed by shape (dragged / sole reader kept / one reader / hub fan) |
| #466, #467 | threaded_absence precision: 106 → 90 findings |
| #468 | per-snapshot decoder caches live as long as their snapshot |
| #469 | each concept's goal search runs in a forked worker. The full search on agent-comms now finishes, at 2,881 s, where it used to die of memory |
| #470 | `CentralClassReadsBack`: a fan reading back through more than 10 modules, 262 edges over 37 hubs |
| #471 | detectors read the creation proof: FamilyMeta/DeclaredFamily classes get a closed MRO. agent-comms dynamic classes 4,067 → 1,294; toad + agent-comms 1,198 → 404 |
| #472 | type-only retype step 1: an annotation-only guarded import is retyped to a base below, with a mypy proof |
| #473 | typed refusals `TypeCheckerReExport` and `AliasNotClass` |
| #474 | class order has one owner (DeclaredC3 reads the creation proof) |
| #475 | `absent_attribute_probe`: an attribute probe no reachable class can satisfy |
| #476 | retype targets a contract base only (ABC or registry family), never an implementation mixin (`Todo` → `StoreRows` erased identity) |
| #477 | named_attribute precision: 73 → 51 |
| #478 | search speed phase 1: reproofs skip the 5 semantic-mirror detectors (they decide nothing in acceptance; the settle still runs them); the probe's mypy reads types only in files holding the asked receivers, 14.4 s → 9.9 s per state, same answers |

## Open work, in priority order

1. **Search speed (Tristan: "2881s is insane", needs 10x, i.e. 288 s or less; don't rerun the full search to measure).** Phase 1 is merged (#478) and saves only seconds per state; the 10x comes from the delta scan below, which is not built. The bench for it is on the unmerged branch `s-nra/reproof-delta-wip` (f883fdf), see "Fork state".
   - **Split of the 2,881 s:**

     | Bucket | Seconds | Share |
     |---|---|---|
     | Whole-tree exact scans, about 28 × 49 s | ~1,350 | 47% |
     | Goal reproofs, about 24 × 24.5 s | ~590 | 20% |
     | Targets and plan | ~300 | 10% |
     | Outside the goal runs (census score and parse) | 618 | 21% |

   - **One cold exact scan on the saved 11-file move:**

     | Part | Seconds |
     |---|---|
     | Whole tree | 50.2 |
     | Whole-tree detectors | 45.8 |
     | — ForeignAbsenceProbe (mypy) | 16.1 |
     | — NameOnlySubclass | 4.6 |
     | — UndistinguishedEnumMembers | 4.2 |
     | — 68 others | ~21 |
     | Per-module detectors | 4.4 |
     | Base census score | 29.6 (includes its own mypy build) |

   - **mypy can't be made per-move here:**
     - mypy 2.4.1's fine-grained update crashes, and its native parser ignores the file-system cache.
     - Cached incremental mode rechecks 825 of 853 modules, because agent-comms is one ~700-module import cycle.
     - Carrying answers through a relocation's name map is exact only for pure relocations, and the search plays almost none.
   - **Next, as one owner:** a "delta scan". A state's findings are its parent's findings, minus those in the files the move wrote, plus detectors rerun on those files and on whatever depends on them. Reproof, settle, `projected_scan` fallback and census all read it. Steps:
     1. find out from the code why 11 runs fell back to `projected_scan`;
     2. make the whole-tree detectors (45.8 s) update by delta, starting with the 4–5 s single detectors that look like per-call index rebuilds;
     3. census by delta;
     4. build planning's index once per base.

     Measure each step on the saved move (one reproof), never with a full search. Run one final full search at the end, on a fresh detached agent-comms main.

     Step 1 detail: the fallback is where `len(detector_types) != len(detector_ids)` sends a reproof to `projected_scan` (codemod_workflow.py, right after the detector-id computation). Find which concepts hit it and why the id isn't registered; a wrong fallback costs up to about 11 × 50 s. Step 2 identity: prove the delta equals the cold exact scan on the saved move plus 4 more moves with non-empty findings. Step 3 options: share the probe's per-state build (same source set) or answer from declared typing.

     The arithmetic, so nobody optimises a detector inside a phase that can't fit: 28 whole-tree scans at even 10 s each is already 280 s. The scan count has to fall (delta), not just the per-scan cost.
2. **Type-only retype applies on agent-comms.** Owner: fork M. Contract-base rounds rerun on agent-comms 106007daa with NRA ff6fb73. Every surviving retype goes into one NRA-generated agent-comms PR on `ccr-604fcf75-vmgoks`. State: see M's line below.
3. **Remaining NRA inventory** (no recipe yet, on current mains): own_type_test (shapes measured, largest is already-typed receivers, 63), string_kind_dispatch + string_literal_dispatch_ownership_lead (about 90, unmeasured), long_condition_factoring_lead (45, unmeasured), foreign_absence_probe (159 findings; has a recipe, but its detector is the slowest). Measure the shapes first, and build only where one mechanical shape reaches about 30 (paper1 Cor 6.2).

## Measured and not built (each under the bar of about 30)

- Absent-member repair (IMPL-9) for threaded_absence: 0 of 106 fit.
- Decode-once: at most 4.
- Move-onto-owner: about 17.
- Extended cut (i): 6 of 155.
- Contract inversion (ii): 19 edges; 0 of 5 central hubs are clean.
- Step-2 declared contract: 3 clean member-reading edges.
- Annotate-from-inferred-type: 10 edges.
- Comms readers taking services directly: frees 4 edges. Thread goal/process members leaving: frees 0.
- Creation gates reading the proof: 29 refusals lifted, 0 new moves.
- named_attribute "declare and dereference": about 25.

**Real limit:** most remaining cycle edges (125 of 169 on agent-comms) are blocked because the central classes' member signatures name types inside the same cycle (`Thread`, `Comms`, `TurnSession`, `RegistrySnapshot`, `TurnRunner`). That's agent-comms layering, not a missing NRA operation.

## Lessons for the next session

- Measure, report, then build. Every fork brief this session that said "measure only" saved a wasted build.
- Rebase without force-push: cherry-pick onto fresh main as a new branch (`…-r`). Never merge main into a branch: #462, #468 and #464's branch slipped.
- CI is post hoc. Merge on a green local targeted run plus `pytest tests -k test_repository_`.
- A fork's claim is a claim until its proof is reread. Three were corrected today: hub-fan counts measured through the ownership bug, a "runner swallows exceptions" claim that was really the probe script, and a 107 "type-only" count that overstated.
- Don't hand Tristan decisions you can make, and check main before calling something a bug.

## Fork state at handoff

- **M: finished; everything is merged** (#470, #472, #473, #476). The agent-comms retype rounds under #476 were stopped in round 1, with 0 applied. Branch `ccr-604fcf75-vmgoks` is untouched. To resume:
  - Rerun the rounds: `docs/handoffs/2026-11-18/retype_rounds.sh NRA_CHECKOUT AGENT_COMMS_TREE OUT_DIR`, next to M's `retype_apply.py` (one retype per process, exit 0 applied / 3 none). AGENT_COMMS_TREE is a fresh `git worktree add --detach … origin/main`. `retype_apply.py TREE --census` prints the offers and the reads that need a declared contract, without applying anything.
  - Expected: `AutoRetryEnd` → `OpenPayload` (family root) and `NativeAttestation` → `SnapshotAttestation` (ABC) apply, and `Todo` → `StoreRows` is refused.
  - Land the survivors as one agent-comms PR on `ccr-604fcf75-vmgoks`, with the touched-module tests run in `/home/user/agent-comms/.venv` and `PYTHONPATH` set to the worktree's `src`.
- **P: phase 1 merged (#478, merge 0dae25a, branch `s-nra/fine-grained-probe` aa6eba4).** Goal-search tests: 1514 pass, 2 fail (the same 2 fail on main; not yet named, find them first). Unmerged WIP: `s-nra/reproof-delta-wip` (f883fdf), holding `tools/performance/reproof_bench/`: the saved move `bench_move.json` (the 11-file class_family_authority move: sources it writes and its 26 reproof detector ids), `bench_exact.py`, `bench_a.py`, `fallback_ids.py`, and the dropped fine-grained/incremental mypy prototypes. It's a bench, not a PR: lift the scripts into the delta-scan PR or delete them there.
  - Run: `PYTHONPATH=<nra checkout> python bench_exact.py src bench_move.json` from an agent-comms checkout. About 80 s of base parse and scan, then seconds per phase.
  - The saved base was agent-comms a38fe730e (32 behind main 106007daa). Rebuild the base from a fresh `git worktree add --detach … origin/main`. The move's file sources may then need regenerating from that base, so check it still applies before timing.
  - The split of the 2,881 s: run counts are measured; per-phase seconds inside runs are inferred from same-concept runs, because the run's log was deleted.
- **Older unmerged work, preserved:** `s-nra/pr280-stage-proof` (64861d9, "Let a concept declare how its stages are proved", from an earlier session; not reviewed this session).
