# Native-behavior proof checkpoint and pause handoff

Date: 2026-09-15

## User direction

The user requested finishing, committing and pushing the current uncommitted
work after the main merge, then stopping NRA work so the computer can be used
for OpenHCS. Do not restart scans, test matrices or new refactoring work from
this handoff until the user resumes NRA. This pause does not complete the
three-target native-behavior objective.

## Worktree and publication

- Worktree: `/home/ts/nra-native-proof-integration-20260914`.
- Remote: `https://github.com/OpenHCSDev/NominalRefactorAdvisor.git`.
- Active branch: `main`.
- Main was fast-forwarded from `76fda5db5dccb60bcfbdb062a19336ff5a4b569b`
  to `93ad35f007379fc156ea6fd24fad057100916d5a` after exact-SHA integration
  run [35017535185](https://github.com/OpenHCSDev/NominalRefactorAdvisor/actions/runs/35017535185)
  completed successfully with all seven jobs passing.
- That source checkpoint is merged, not merely proposed. Local and remote main
  were verified at the same SHA. The follow-up containing this handoff adds the
  canonical namespace admission fix, nine controls, its authored DSL recipe,
  the completed edit-location timing report and updated progress notes.
- Resolve the follow-up's exact immutable SHA with
  `git log -1 --format=%H -- docs/plans/native_behavior_proof_pause_20260915.md`.
  Verify hosted CI for that SHA separately; the preceding green run does not
  validate the new follow-up.

## Follow-up ownership fix

`SourceClassBodyEntryABC.require_admitted` previously checked only the native
island. A copied or reconstructed entry could claim a second original namespace
for the same execution and class node. Eight controls reproduce this before
the fix, covering plain/native-metaclass preparation, copy/reconstruction and
cold/warm state. The declaration owner now validates its original operation
and requires its canonical entry from `execution.class_entry(node)`.

Original-operation diagnostics are revalidated on rejected noncanonical paths,
preserving stale-node rejection without repeating unrelated validation on every
successful frame query. Actual operation consumers keep their existing checks.
This does not construct a native class, invoke a metaclass,
execute logging or registration callbacks, or fabricate an original creator
frame. Unsupported construction/results remain fail-closed.

## Validation record

Final focused suites use eight workers and 60-second bounds:

- Python 3.11 closeout: 141 passed, two skipped, 8.67 seconds.
- Python 3.14 closeout: 143 passed, 11.27 seconds.
- The complete authored replay: 40 clean stages, six exact formatted production
  files and immutable original input.
- Package, broad-suite and scan closeout results are recorded below before
  publication. Retain initial failed and timed-out logs; they are not green
  gates. The Python 3.14 isolated-subprocess import problem was corrected by
  installing this checkout in the runtime, not by changing tests.

Disjoint broad-shard coverage across the follow-up and corrections covers all
7,330 collected items per runtime: Python 3.11 totals 7,256 passes and 74 skips;
Python 3.14 totals 7,292 passes and 38 skips. The slow class-history shard passes
339 tests in 138.90 seconds after the hot-path correction, with its original
200-class input and assertions intact. The earlier 165-second terminations and
failed diagnostic/import controls remain recorded. These aggregate gates span
corrective source revisions; they are not a claim that one frozen final source
ran every test locally. Exact final source is checked by the closeout affected
suites, installed-wheel controls, and the separate post-push hosted matrix.

Fresh docs and isolated sdist/wheel builds pass; docs retain two pre-existing
duplicate API-description warnings. Exact closeout installed-wheel checks
compare all 141 production module byte strings to the archive and checkout,
then exercise all eight owner-identity controls and the portable DSL control
without importing production from the checkout. Installed public-API analysis
returns a list with zero findings. The final wheel CLI self-scan passes all 79
detectors, zero omissions, zero findings, in 8.759 scan seconds; its report is
`canonical-namespace-closeout-wheel-self.json`.

Complete-package cold/warm/one-edit gates pass with all 79 detectors, zero
omissions and 180 retained/215 raw findings. Scan times are 77.024/2.992/6.248
seconds and wall times 80.96/6.54/8.87 seconds. Cold was bounded at 165 seconds;
warm/edit at 60. All normalized report hashes equal
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`,
also matching the preceding physical-source checkpoint. These scans overlap
bounded test work and precede the final hot-path correction; they are complete
semantic-consistency gates, not isolated timing comparisons or equivalence
proofs. The final installed-source scan supplies closeout validation separately.
The source-copy EOF edit is restored.

The replay driver is 51 lines; its five recipe files total 636 lines, including
authored Python replacement bodies. Its leverage is ordered batching and
intermediate-state checks. Final before/after diff size is not cumulative edit
volume, and no manual tool-call or token-savings ratio was measured. Raw body
replacement does not automatically establish the body's correct factoring or
behavioral equivalence. The advisor's global ownership audit remains distinct.

## Remaining native-behavior objective

Resume from the original assignment in `native_behavior_proof_handoff_20260915.md`
and the exact owner/operation evidence in `native_behavior_proof_progress_20260915.md`.
Do not redo the historical 209 failures or the completed performance integration.

1. Retain and audit supported actual current MRO registry lookup and narrow
   native `type`/`classmethod` proofs. Do not treat captured identity, source
   correspondence or authored native-use premises as automatic execution proof.
2. Finish actual generated `AutoRegisterMeta` construction/registration evidence.
   `NativeSourceClassEntryABC.construction_admission` and
   `AutoRegisterClassEntry._created_result` remain unresolved. Prepared inputs
   and current constructor source/signature are prerequisites, not that proof.
3. The existing source-function activation authority assumes a source-created
   callee and its original creator frame; imported current functions need their
   actual native globals/builtins/closure and exact invocation binding, without
   relabeling them as source-created captures. Frozen ABC `**kwargs` needs its
   real supported storage/binding law, not fabricated source defaults/frames.
4. Follow the actual installed metaclass version's inheritance, abstract-member
   filtering, key selection, registry mutation and reachable hooks. In particular,
   `_auto_configure_registry` calls `logger.debug` unconditionally;
   `log_registration=False` does not discharge that callback. Lazy/secondary
   registration and unsupported effects need real exclusion or unresolved proof.
5. Preserve declaration-owned semantics, ABC/MI/MRO, original source cuts,
   canonical owner identities and dependency-aware revalidation. Use the DSL
   where supported. Complete broad bounded parallel Python 3.11/3.14 validation,
   docs/package/installed-wheel/API gates, complete cold/warm/edit scans and
   exact-SHA hosted CI for substantive future proof changes.

## Environment and retained artifacts

- Python 3.11:
  `/home/ts/code/projects/nominal-refactor-advisor/.venv/bin/python`.
- Python 3.14:
  `/home/ts/.cache/nra-uv/archive-v0/JPEtYotW8LyXrtWG/bin/python`.
  This is a cache-backed environment: verify it still exists and its editable
  NRA import resolves the integration worktree before reuse.
- Reports, logs, disjoint-shard runners and replay helpers:
  `/home/ts/nra-native-behavior-validation-G481Dc`.
- Full scan source copy: that directory's `source` subtree; 1,057 Python files,
  OpenHCS plus eight external packages, tests removed. Its edits must be restored
  and disposable caches removed before handing the computer back.
- Preserve the unrelated original NRA worktree, all other worktrees and the
  user-provided untracked `native_behavior_proof_handoff_20260915.md`.

At closeout, the owned test/scan/package and old CI-monitor handles are terminal,
and the process check finds no owned NRA worker. Removed 66 explicit task-owned
cache/fixture directories totaling 799,784 KiB (about 781 MiB). Source copies,
reports, logs, package builds, environments and replay helpers are retained.
Hosted CI after the final push is a separate remote gate; inspect its exact SHA
on resumption. The full objective is unfinished, not marked complete.
