# Same-node cooperative follow-through

Source owner Schrodinger/Codex; parent retains integration and installed/full
qualification. Same [draft PR16](https://github.com/OpenHCSDev/NominalRefactorAdvisor/pull/16),
not a competing implementation. Source checkpoint
`19f21b6cb660826fa7c593af1d59101009b46019`, predecessor evidence HEAD
`c6ec13f8bf2b6c07b8fae8db8533fcd6fb5d6d2b`. The preceding source freeze,
profiles, timeout and native-proof failure remain retained unchanged.

## What the experiment establishes

At c6ec13f, `_ProjectionVisitor.visit_Assign`, `visit_AnnAssign` and
`visit_Return` called `_traverse_projection_children`, which delegated directly
to `super().generic_visit`. Child events could reach another capability, but
same-node hooks downstream of this capability in C3 could not.

The independent `StatementCensus` declares a cooperative constructor and small
Assign/AnnAssign/Return/Call hooks. Two composition declarations put it before
and after the existing compact visitor. The new-case test compares exact node
identity, source scope and event multiplicity, projected/unprojected and
valueless statements, nested scopes, descendant calls, both presentation modes,
unchanged presentation/check/supplement facts, and one shared ancestor in MRO.
No shared consumer or taxonomy registration is edited to admit the experiment.

The first experiment retained an additional test-constructor mistake: the
before-order constructor did not forward projection kwargs. After correcting
that independent test hook, the before-order cases passed and both after-order
cases failed with **zero of seven required same-node observations**. Thus the
bypass is demonstrated, while no production regression or installed failure is
claimed. The current test with `-k same_node_capability` supplies the same
experiment against the predecessor production owners.

## Ownership and replacement

`ClassFunctionStackNodeVisitor` remains the single shared traversal owner. It
now supplies native AST traversal endpoints for the three additional events,
using the original `ast.NodeVisitor.generic_visit`, not a copied algorithm.
The projection capability delegates through each matching `super().visit_*`.
Its one context manager owns only its suppression state and exception restoration;
it does not perform traversal or dispatch. Suppression spans the downstream
cooperative chain and its children; independent observations remain enabled.
The bypassing helper is removed, not retained as a facade or fallback.

Three additional tests raise at a downstream same-node hook after presentation
emission and verify suppression, both capability stacks and the shared scope
unwind. The full new-case extension needs only its declaration/hooks and a
composition declaration. The generic consumer is unchanged. This is a contract
for the admitted events, not a claim that every unrelated AST override is now
cooperative.

Catalog review: IMPL-4 incomplete cooperative participation repaired; IMPL-12/13
the original shared traversal reused rather than duplicated; IMPL-1/2/3/5 no
new consumer dispatch; MEMB-1/2 no roster or hook table; IDEN-5 no mirrored
source/suppression store; TIME-1/3 no legacy helper or compatibility facade.
Existing cache identity, strict AST-free admission and fail-closed proof code
are unchanged. This remains manually authored, not a native-equivalence or
global architecture certificate.

## Bounded source receipts

| Receipt | Outcome | Wall seconds | Peak KiB |
| --- | --- | ---: | ---: |
| same-node-before.log | 4 failed; includes constructor error | 2.72 | 86036 |
| same-node-before-02.log | 2 passed / 2 failed; downstream bypass isolated | 2.77 | 86036 |
| same-node-after.log | 36 passed, including 7 new cases | 6.33 | 90072 |
| same-node-consumers.log | 11 existing consumer/cache cases passed | 6.57 | 105996 |
| same-node-parity.log | complete four-file component signatures unchanged | 3.51 | 85300 |

47 distinct current-checkpoint cases pass, not a rerun of the earlier entire
83-case coverage. Maximum observed RSS is 103.51MiB. The component comparison
retains identical original paths, raw source SHA256s, content signatures and
458 presentations / 36 checks / 292 supplements from the predecessor receipts.
No end-to-end speed, completed original R1 or global audit claim follows.

Each command is retained in its log: exact read-only frozen Python3.12.3,
`-I -B`, one CPU/thread pools, systemd scope512MiB/no swap/100% CPU, outer60s,
explicit source import, isolated pytest without xdist/cacheprovider. All source
checks are serial and provider-free. No native/UI/MCP/science, installation,
shared lock, original consumer, parent run or baseline change. Baseline remains
clean at673c062f and the immutable consumer retains its recorded SHA256.

Resource admission reported RAM18.5GiB/home9.6GiB and swap12.9GiB; advisory
warning, not a hard-limit failure. Owned disposable root:
`/home/ts/.cache/agent-scratch/nra-r1-same-node-357-20261001` (816KiB).
Logs are copied here and checksummed before removal. The first copy command was
rejected before execution because its new destination cwd did not yet exist;
the successful copy used the existing owned worktree cwd.

Parent retains complete original R0/R1 160/165 qualification and installed
acceptance. The original native-proof failure and broader timeout are not
waived or requalified by these tests; global85detector/FULL remains unqualified.
The parent's terminal scratch address is neither reused nor removed.

Cleanup complete: all five retained log checksums passed; `lsof +D` exact root
returned1/no handles and scoped process search found no worker. `realpath`
matched the explicit owned address. Removed only that816KiB root with `rm -r --`
and verified absence. Synthetic cache/test artifacts are permanently removed;
all logs and command recipes are retained here. All tool PTY sessions terminal.
No parent scratch, installed package, frozen input or other-owner source removed.
