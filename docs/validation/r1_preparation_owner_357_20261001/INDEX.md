# Source checkpoint and retained evidence

Current authorized follow-through:
[same-node cooperative receipt](same-node-followthrough/README.md), source19f21b6.
The original checkpoint/results below remain historical evidence, not overwritten
or retrospectively qualified by the follow-through.

Source owner Schrodinger/Codex; integration and original R0/R1/installed
qualification parent. Draft https://github.com/OpenHCSDev/NominalRefactorAdvisor/pull/16.
Production source checkpoint `2ae2309967904e0195f0d8e75b9b9988c736a52a`;
baseline `673c062fc656e9c74f1eddcab30f036c9befbc1f` remains unchanged/clean.
No native/UI/science, installations, shared locks, providers or downloads.

## Common execution envelope

Read-only interpreter:
`/home/ts/wt/openhcs-generated-inputs-installed-parent-20261001/.venv/bin/python`.
Python3.12.3, `-I -B`, source import explicitly pinned before importing NRA.
Each process: `systemd-run --user --scope --quiet -p MemoryMax=512M
-p MemorySwapMax=0 -p CPUQuota=100%`, `/usr/bin/time -v`,
`timeout --signal=TERM --kill-after=3s 60s`, `taskset -c 0`.
Serial processes only. `PYTHONDONTWRITEBYTECODE=1`; OMP/OPENBLAS/MKL/NUMBA
thread counts1. XDG_CACHE_HOME, NRA_CACHE_HOME and TMPDIR all under the new
owned scratch root, never the terminal parent diagnostic address.
Tests disable plugin autoload, `-o addopts=`, `-p no:cacheprovider` and have
explicit separate owned basetemps. The command line in every log identifies
the exact tests/arguments. Source diagnostic drivers import original NRA APIs;
none copies a detector or implements another parser/proof engine.

Tests/diagnostics cwd is the owned NRA WT except baseline proof test08 (pinned
baseline WT) and original-context profiles (original OpenHCS UI workflow WT).
Original-context profiles retain consumer SHA256
`4c6282f52e7c188068dd9d63399f5522a7cb1d222504d0c0b809be10303884ca`,
the original791650087 ->3d58912 revisions, all3roots/eightexactdeps and
`--budget-seconds160`. The SIGPROF20CPU diagnostic stop is separate from the
original SIGALRM deadline. Git packed window/limit8m/128m match admission.

## Results (not original complete qualification)

| Receipt | Outcome | Wall seconds | Peak KiB |
| --- | --- | ---: | ---: |
| cooperative-tests-01.log | rejected pytest parallel defaults; exit4 | 2.67 | 83596 |
| cooperative-tests-02.log | 6passed/1failed; capability event lost | 2.77 | 85216 |
| cooperative-tests-03.log | 20passed | 5.79 | 88868 |
| proof-tests-04.log | INCOMPLETE exit124; partial dots not counted | 60.01 | 90540 |
| cache-tests-05.log | 17passed/4CLI tests deferred to09/10 | 30.23 | 89540 |
| owner-tests-06.log | 32passed | 7.55 | 87268 |
| graph-tests-07.log | 20passed/1native proof failure | 14.84 | 453892 |
| predecessor-test-08.log | same native proof failure on673c062f | 2.18 | 86920 |
| cli-tests-09.log | 12passed (includes10 repeated focused cases) | 17.91 | 89560 |
| cli-tests-10.log | 2passed | 18.57 | 88644 |
| baseline-profile.log +baseline.pstats | INCOMPLETE exit75,20CPU | 23.37 | 122328 |
| candidate-profile.log +candidate.pstats | INCOMPLETE exit75,20CPU | 23.34 | 121804 |

Accepted source shards cover83distinct passing cases plus one retained failing
native-capture case; do not sum repeated cases or partial dots. The failing
`test_completion_does_not_repeat_whole_module_syntax_walk` raises
`CapturedReferenceRejection: Object capture remains open: unproved_execution_effects`
on both candidate and baseline under the same frozen environment. Its admission
is not weakened, skipped or converted to success. No blanket green-suite claim.

The initial new-case failure02 exposed another capability being hidden by
presentation early returns. It was fixed, not waived: projected assignment,
annotated assignment and return now suppress only presentation descendants,
while continuing shared traversal. Lambda check exclusion, scope/shadowing,
exception unwind, one-event-per-call and declaration-only extension are tested.

Original-context partial profile: semantic family collection3.231 ->2.853s;
`DeclaredTypeCheckModule.collect` second traversal is absent in candidate,
`from_collector` called8times. Generic AST visits534623 ->411402. Both samples
have8parse calls but stop at different syntax events; these are NOT full-context
throughput estimates or R1 results. AST retention checks0.829 ->0.957s are
preserved, not declared dominant or weakened.

`family-parity-{baseline,candidate}.log` is the initial source-component
comparison. `canonical-parity-{baseline,candidate}.log` improves its provenance:
the existing PythonModuleRootParser supplies actual package/module identities.
Both canonical runs retain the same four baseline source files/rawSHA256s and
identical complete content signatures, 458presentations/36checks/292supplements.
Collection total0.747386 ->0.612257s in this single sample (~18.1%); process
wall2.99 ->3.73s. **No cold-process or end-to-end speed improvement claim.**

## Ownership and architecture review

The original ClassFunctionStackNodeVisitor ABC remains traversal/stack owner.
ParsedModuleClassFunctionStackNodeVisitor centralizes the one parsed source
context formerly assigned separately by both capabilities. The concrete
composition declaration owns C3 order: presentation ->declared check ->shared
source/scope ancestor ->ast.NodeVisitor. The shared record constructor belongs
to DeclaredTypeCheckModule. No copied extraction body or fallback facade.

Required pairs: presentation observations ->semantic family; declared checks
->semantic family; class/function scope ->both capabilities; parsed-module
identity ->both capabilities. Lambda checks remain forbidden while another
capability's lambda events remain admitted. Previously projected descendants
remain forbidden for the presentation capability, not for declared checks.
This preserves distinct independently varying questions, not a merged record.

Catalog: IMPL-12/13 duplicate traversal removed; IMPL-4 completed cooperative
participation; IMPL-1/2/3/5 reviewed, no new domain string/enum/type dispatch;
MEMB-1/2 no roster introduced; IDEN-5 one source context; TIME-1/3 no duplicate
cutover/compatibility path. Python AST's external vocabulary remains honored.
The new-case CallCensus test adds only its declaration/hooks and a composition
declaration: no generic consumer edits. It proves the admitted event contract,
not every possible future AST policy. No ornamental MI or mirrored schema/store.

`owner-census-*.log`: original ModuleSyntaxIndex census and compact class-family
projection; canonical variants also join through original class-family index.
Baseline212ClassDefs,211index-admitted,1OPEN function-nested Visitor; candidate
214ClassDefs,213index-admitted, same1OPEN. Alternative owners are retained, not
heuristically invented. Census is a source-review receipt, not global proof.
This cooperative fusion is manually authored; no unsupported DSL/native
equivalence certificate or global ancestry-minimum certificate is claimed.

## Remaining limits / cleanup

Parent runs original160NRA/165wall two-revision qualification and original R0,
then any paired installed acceptance. Global85detector/FULL is separately
unqualified. No baseline/main/backing package or other owner's source modified.
The original archived failure/60s diagnostic remain immutable predecessor
evidence, and their old scratch address was never reused or cleaned by us.

All retained receipts/profiles have SHA256 in SHA256SUMS. Source file digests
and exact predecessor references are in freeze-manifest.json. Only the owned
`/home/ts/.cache/agent-scratch/nra-r1-preparation-owner-357-20261001` is disposable;
cleanup disposition is recorded after process/open-handle proof below.

Cleanup complete: lsof +D exact owned root returned1/no handles; scoped process
search found no remaining worker. Every retained receipt/profile verified with
sha256sum -c before removal. realpath matched the explicit owned root. Removed
only that19MiB root with `rm -r --` and verified absence; disposable synthetic
test/cache artifacts are removed permanently, while all logs/profiles/recipes
are retained here. No parent cache, original archive, input, installed package,
baseline WT or other owner's work was removed. All tool PTY sessions terminal.
