# Publication/preparation profile after the original R1 failure

Source owner Schrodinger/Codex; parent retains integration and complete
qualification. Same [PR16](https://github.com/OpenHCSDev/NominalRefactorAdvisor/pull/16),
reviewed evidence head18910a5, production checkpoint19f21b6. Production source,
test declarations and the original consumer are **unchanged by this checkpoint**.

## Original full qualification is terminal, not a pass

Parent's exact791650087 ->3d58912 case, immutable4c6282f5 consumer,
all3roots/eightexactdeps, inner160/outer165, CPU0/512MiB reached native
`ScanDeadlineExceeded` at160s. Receipt: exit1,164.42s,187.41MiB. The stack was
`CompactFamilyProjectionBatch.__post_init__` -> accumulator `_retains_ast` ->
the original `retains_python_ast`. This is a failure before the base scan
returned, not installed NRA/globalFULL qualification.

The log and command JSON copied here retain exact bytes and SHA256 of the
parent's immutable originals under
`/home/ts/wt/openhcs-issue-batch-20260929/producer-installed-20261001/`:
`r1-357-newcase-parent-original-context-20261001.{log,command.json}`.
No original run, script, archive or parent scratch address was reused/changed.

## One bounded useful profile, not another complete attempt

Used the existing `tools/profile_r1_preparation.py` and original consumer with
the unchanged revisions/context/inner160 budget. A new owned root and SIGPROF45
CPU diagnostic stop remain inside the fixed outer60s/oneCPU/512MiB source bounds.
This is not an increased original deadline. Interpreter remains the read-only
frozen installed Python3.12.3, `-I -B`; systemd scope MemoryMax512M/no swap,
CPUQuota100%, tasksetCPU0, thread pools1, original Git packed-window/limit8m/128m.
All exact command arguments appear in the retained log. No narrower scan, cached
continuation, context pruning, detector copy or provider/native/UI/science run.

Profile: intentional INCOMPLETE exit75,48.37s wall,122540KiB RSS (119.67MiB),
96,830,231 calls /45.582s instrumented work. It stops in AST line-number
canonicalization before the first base scan result. 33 parse calls,64 family
collection calls and54 runtime-batch construction calls are **invocation counts**,
not a certificate of distinct/complete modules or full-context coverage.

| Original owner/function | Calls | Inclusive profile seconds |
| --- | ---: | ---: |
| build_compact_projection_shard | 28 | 31.900 |
| collect_family_batch | 64 | 27.193 |
| compact class family collection | 32 | 12.589 |
| compact semantic family collection | 32 | 9.947 |
| _parse_source_module | 33 | 7.361 |
| _compact_class_syntax_facets | 32 | 7.239 |
| ModuleSyntaxIndex.build | 32 | 5.654 |
| cache store_items | 64 | 4.147 |
| retains_python_ast | 118 roots /731740 total | 2.752 |
| runtime batch __post_init__ | 54 | 1.240 |

These times include children and overlap; they must not be added. AST retention
is6.04% of this partial sample, not demonstrated to dominate the complete run.
Caller attribution is64 cache-admission roots/1.515s and54 batch-admission
roots/1.237s. A deadline landing inside one guard does not establish its overall
cost. No guard is weakened, bypassed or memoized on that observation.

## Concrete ownership review and unresolved duplication

Reviewed the actual combined production diff against authoritative current
refactor-audit catalogue (IMPL-4/12/13, IMPL-1/2/3/5, MEMB-1/2, IDEN-5,
TIME-1/3). Shared traversal still belongs to ClassFunctionStackNodeVisitor;
source context belongs to ParsedModuleClassFunctionStackNodeVisitor. Independent
presentation/check algorithms remain on their existing capabilities, composing
through real C3 and matching `super().visit_*` hooks. Projection suppression is
owned by that capability and does not perform traversal. The replaced bypassing
helper and second full check walk remain deleted. No new consumer switch,
registry, duplicate procedure, cache authority or compatibility path appears.

Retained same-node evidence remains applicable because production hashes are
unchanged: StatementCensus supplies only declarations/hooks, before/after
composition observes each assignment/annotated assignment/return and call once,
both presentation modes pass, and downstream controlled exceptions unwind scope
and suppression. Seven new cases are included in the earlier47 distinct current
source passes. This receipt does not pretend to have rerun that suite.

The profile and source reveal a **specific unresolved IMPL-12/13 preparation
duplication**: within the composed function event, declared-check prelude and
presentation postlude independently call the existing lexical binding authority
on the same `node.body`. The caller trace reports4393 calls/1.026s from
DeclaredAttributeCheckCollector.visit_FunctionDef and4393 calls/1.040s from
_ProjectionVisitor.visit_FunctionDef. Class syntax facets also request function
bindings (4425 body/module calls/1.069s). The algorithm itself already belongs
to LexicalScopeBindingAuthority; its determining function scope is not yet shared
by the preparation consumers. This is separate from successful same-node hook
delivery and is not hidden by it.

Next owner-level source work is to share that exact lexical determining answer
through the existing function/source-scope owners, with prelude/postlude lifetime,
shadowing, nested scope and exception behavior demonstrated. It must not add a
parallel binding store/roster, cross-boundary AST-free verdict cache, new type
taxonomy or copied collector. This diagnostic checkpoint does not implement or
certify that migration. ModuleSyntaxIndex's one traversal is measured too; it is
not claimed to be duplicated merely because its cost is large.

## Bounds, remaining limits and cleanup

Admission helper reported RAM20.9GiB/home8.2GiB/swap11.0GiB; advisory warning,
not a passed helper or hard-limit breach. Only serial bounded source processes
were started. Caller/callee reports read the original cProfile artifact with
stdlib pstats under the same resource envelope; no algorithm/detector was copied.
No production edit or new behavioral/throughput pass is claimed. Original
R1 remains FAILED; original R0/installed acceptance/global85detector/FULL are not
qualified by this work. Parent's serial lock/native slot are untouched.

Owned disposable root:
`/home/ts/.cache/agent-scratch/nra-r1-publication-357-20261001` (288KiB).
All logs/profile/parent witness copies are retained here with SHA256SUMS.
Cleanup complete: all six retained checksums passed; exact-root `lsof +D`
returned1/no handles; scoped process search found no remaining worker. `realpath`
matched the explicit owned address. Removed only that288KiB root with `rm -r --`
and verified absence. Disposable staging/cache artifacts are removed permanently;
all diagnostic logs/profile/recipes and parent witness copies are retained here.
All tool PTY sessions terminal. No parent scratch, original archive, frozen input,
baseline, installed package or other owner's source was removed.
