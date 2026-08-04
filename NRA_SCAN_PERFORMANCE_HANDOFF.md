# NRA scan-performance handoff

## 2026-08-03 exact-global checkpoint

Exact/global optimization is now a separate active workstream from the bounded
focused loop.  The first checkpoint preserves every detector family while
removing two avoidable costs: cyclic garbage collection is suspended while
acyclic repository ASTs are materialized, equal AST line-number integers are
shared across modules, and completed module-local/contextual detector caches
are released at their correctness boundary.

On the current DQDock checkout (919 production modules), a controlled cold
parse without detector execution changed from 22.44 seconds / 986.2 MB to
16.59 seconds / 889.1 MB.  This is a same-checkout comparison; it supersedes
neither the earlier 842-module observation nor its different inventory.  A
45-second exact focused-context probe reached a 1,170,584-KB high-water mark
before its enforced deadline, below the historical terminal observation near
1.9 GB but still too large for the intended workflow.

The remaining work is representation-level.  Exact global scans still retain
the full repository AST, so the next stage is to persist compact per-module
context projections and migrate context-dependent detectors onto those
projections without changing findings.  The focused partial lane remains the
recommended interactive command until that exact projection path is complete.

Verification for this checkpoint passed all 944 tests in 332.97 seconds.

## 2026-08-03 compact-global projection checkpoint

The exact/global path now has an explicit bounded representation contract.
Context-dependent detectors can declare a persisted, per-module compact fact
family; a streaming accumulator parses one source module, validates that the
facts retain neither ``ParsedModule`` nor ``ast.AST``, and releases the module
before advancing.  Findings are then reconstructed from the complete
repository fact set, so this remains cross-module reasoning rather than a
focused/local approximation.

Four detector families have been migrated to this contract:

- formal-boundary external string-registry mirrors;
- generated-boundary semantic-constant mirrors;
- export-policy predicates; and
- registry-traversal substrates.

One detector incorrectly classified as global, the callable-method-axis
registry detector, was also proved module-local and moved to the per-module
lane.  The default partition is now 183 per-module detectors, four compact
global detectors, and 65 context-dependent detectors that still require the
all-at-once AST representation.

A cold, cache-disabled stream across the current 919-module DQDock production
inventory completed projection and finding reconstruction for the four
migrated families in 21.96 seconds, found 61 issues, and peaked at 151,772 KB.
That is 82.9% below the 889,100-KB peak of the controlled all-at-once parse
checkpoint, although the measurements cover different amounts of detector
work and therefore are a representation-memory comparison rather than an
end-to-end speedup claim.  An isolated persistent-cache run took 26.07 seconds
cold and 14.13 seconds warm, with a combined-process high-water mark of
162,644 KB and the same 61 findings.

The streaming authority is not wired into the normal exact CLI yet.  Doing so
before the remaining 65 detector families are migrated would still require
materializing the repository AST for those families and would not reduce the
exact command's memory floor.  The next stage is to migrate shared contextual
indexes and the remaining detector families, with full-AST equivalence tests,
then switch exact orchestration once no context-dependent family retains ASTs.

Checkpoint verification passed all eight focused detector/projection tests.
The full run passed 946 tests and had one order-sensitive failure in the
unchanged local-role semantic-descent detector; that test passed immediately in
isolation, after the complete analysis-cache file, and after its 15 preceding
semantic-descent tests.

## 2026-08-03 shared compact-family checkpoint

The compact path now covers ten context-dependent detector families.  In
addition to the first four, it streams normalized method hierarchies, repeated
private methods, repeated builder calls, declared-field extraction sites,
repeated export dictionaries, and manual class-registration shapes.  Shared
families are retained once even when several detectors consume them.  The
remaining partition is 183 per-module detectors, ten compact-global detectors,
and 59 context-dependent detectors that still retain repository ASTs.

Persistent compact facts can now be loaded from an exact source signature,
module identity, family type, Python version, and schema without first
deserializing the module AST.  A regression test makes AST loading fail on the
second pass and proves that a complete warm family-cache hit bypasses it.  Any
missing or oversized family payload falls back to parsing that module, so the
optimization does not weaken cache invalidation or correctness.

On the same 919-module DQDock production inventory, an isolated persistent-cache
run across all ten migrated families took 55.19 seconds cold and 18.19 seconds
warm, reconstructed 311 findings from 63,067 unique compact facts, and reached a
combined-process high-water mark of 340,968 KB.  The migrated set now performs
substantially more detector work than the four-family checkpoint, so timings
are not directly comparable.  Peak memory remains 61.7% below the controlled
889,100-KB all-at-once parse peak.

## 2026-08-03 compact class-index checkpoint

The compact path now reconstructs the repository inheritance graph from
AST-free class declarations, import aliases, base-reference parts, direct
class assignments, metaclass names, and selected registry-ordering calls.  Its
resolved bases, children, ancestors, and descendants match the full AST index
across local inheritance, aliased and qualified imports, generic/subscripted
bases, unique unqualified names, and the existing unique-suffix rule.

Two AutoRegister detector families consume the shared index: inherited
registry-configuration boilerplate and explicit priority-like registry
ordering.  The default partition is now 183 per-module detectors, 12 compact
global detectors, and 57 context-dependent detectors that still retain the
repository AST.

On the same 919-module DQDock production inventory, an isolated persistent-cache
run across the 12 migrated families took 64.67 seconds cold and 21.54 seconds
warm.  Both passes reconstructed 326 findings from 65,571 unique compact facts,
and the combined process reached a 363,280-KB high-water mark.  This is 59.1%
below the controlled 889,100-KB all-at-once parse peak while covering two more
global detector families than the previous benchmark.  Projection/cache,
class-graph equivalence, and direct detector verification passed 48 plus four
focused tests.

## 2026-08-03 compact private-reference checkpoint

The compact path now covers 15 detector families.  A shared private-reference
projection retains only valid private-identifier counts and AST-free function
facts, then reconstructs dead embedded-payload, unreferenced private-function,
and dangling private-method findings from the repository-wide aggregate.  It
does not retain embedded payload strings or function ASTs.  Compact findings
are checked directly against the preserved legacy AST candidate algorithms,
including cross-module call witnesses.

The first DQDock measurement exposed a classmethod-level ``lru_cache`` that the
module-boundary cleanup did not discover.  It retained every streamed surface
function AST and drove the private-reference-only stream to an 882,332-KB peak.
Cache cleanup now discovers cached class/static methods once, reuses that
clearer registry at later module boundaries, and leaves the surface-function
cache empty.  The same private-reference-only 919-module stream then peaked at
121,872 KB.  Caching the clearer discovery avoids rescanning all class
descriptors at every boundary.

The combined isolated persistent-cache benchmark across all 15 migrated
families took 71.45 seconds cold and 23.16 seconds warm.  Both passes produced
335 findings from 66,490 retained top-level projection items, and the process
peaked at 386,916 KB.  This remains 56.5% below the controlled 889,100-KB
all-at-once parse peak.  The default partition is now 183 per-module detectors,
15 compact global detectors, and 54 context-dependent detectors that still
retain repository ASTs.

After that measured checkpoint, support-prelude module-family detection moved
onto a compact per-module import/class fact.  The current partition is 183
per-module detectors, 16 compact global detectors, and 53 AST-retaining
context-dependent detectors; the 15-family numbers above remain the latest
like-for-like DQDock benchmark rather than being relabeled after the change.

Environment-boolean authority drift subsequently moved to compact parser-site,
declared-authority, and fixed-key wrapper selector-chain facts.  Wrapper calls
still resolve against the complete repository authority set, so this preserves
global matching rather than reducing the detector to module-local behavior.
The current partition is 183 per-module, 17 compact global, and 52
AST-retaining context-dependent detectors.

Public bare support-function analysis now also uses compact eligible-definition
and filtered reference-site facts.  Its candidate output is regression-checked
against the legacy repository reference index.  The current partition is 183
per-module, 18 compact global, and 51 AST-retaining context-dependent
detectors.

The 18-family isolated persistent-cache DQDock benchmark took 89.13 seconds
cold and 22.78 seconds warm, with 339 identical findings from 68,328 top-level
projection items and a 370,776-KB process high-water mark.  This is 58.3% below
the controlled 889,100-KB all-at-once parse peak.  An initial per-site support
reference representation peaked at 461,952 KB; aggregating the 70,923 eligible
per-module symbol/site groups into exact counts removed 91 MB without changing
findings and is the accepted checkpoint representation.

The shared compact class projection now also carries exact keyed-family type
arguments, enum-keyed module-table summaries, and per-function closed-axis
branch aggregates.  Four additional global detectors consume those facts:
parallel keyed families, parallel keyed tables, keyed table/family overlap, and
residual downstream branching over an already-owned closed axis.  Their
family/table/branch candidates are checked directly against the legacy AST
collectors.  The current partition is 183 per-module detectors, 22 compact
global detectors, and 47 AST-retaining context-dependent detectors.

DQDock was being edited during the first measurement, so a production-source
snapshot was taken under ``/tmp`` for the accepted cold/warm comparison.  It
contains the same 919 production Python modules and the same 116 small external
authority files used by the formal-boundary detector.  The isolated 22-family
run took 98.71 seconds cold and 25.64 seconds warm, retained 68,465 top-level
projection items, produced 340 identical findings, and reached a 369,568-KB
combined-process high-water mark.  The shared class facts include 16,911 class
records, two keyed tables, 8,606 branch-bearing functions, and 19,718
aggregated function/axis rows.  Peak memory is 58.4% below the controlled
889,100-KB all-at-once parse baseline and slightly below the 18-family
checkpoint despite the added global work.

The keyed-axis projection now also covers cross-module manual-selector shadow
families, completing that related detector cluster.  Compact top-level
definition facts migrate private-helper shadow analysis, and a sparse
dataclass/namespace/CLI projection migrates the configuration-surface mirror
detector.  The latter retains no module record at all when a module contributes
neither a relevant dataclass nor a CLI tuple.  This also exposed and fixed a
latent ``NamedValueBinding.name`` mismatch in the legacy CLI collector.  The
current partition is 183 per-module detectors, 25 compact-global detectors,
and 44 AST-retaining context-dependent detectors.

The stable-snapshot 25-family run retained 68,466 top-level projection items,
produced 349 identical cold/warm findings, and took 99.35 seconds cold and
28.16 seconds warm.  Peak RSS was 370,264 KB, still 58.4% below the controlled
all-at-once parse baseline and only 696 KB above the 22-family checkpoint.  The
shared class projection carries 20,965 aggregated top-level definition rows;
DQDock contributed no manual-selector-axis rows and only one sparse
dataclass/CLI module projection.

Exact-type fail-loud boundary analysis now also reconstructs from the shared
class projection.  Per-module guard facts retain normalized subject/type text,
reference parts, polarity, lexical ``type``-shadowing decisions, and certified
failure-branch shape; imported and aliased type references are resolved later
against the complete compact inheritance graph.  The current partition is 183
per-module detectors, 26 compact-global detectors, and 43 AST-retaining
context-dependent detectors.

The stable-snapshot 26-family run retained the same 68,466 top-level items plus
2,253 nested exact-type guard facts.  It produced 398 identical cold/warm
findings in 104.86 seconds cold and 30.01 seconds warm, with a 380,992-KB peak.
This remains 57.1% below the controlled 889,100-KB all-at-once parse baseline
while adding 49 exact global inheritance-boundary findings.

Semantic-inheritance membership SSOT analysis now also reconstructs from the
shared compact class graph.  The projection retains direct method and abstract
method names, dataclass/abstract status, registration-authority predicates, and
source extents without retaining class ASTs.  Direct candidate-equivalence
tests cover the compact and legacy collectors, while detector calibration still
covers inherited/external registration authorities, enum roots, imported key
bases, and dataclass product families.  The current partition is 183 per-module
detectors, 27 compact-global detectors, and 42 AST-retaining context-dependent
detectors.

The stable-snapshot 27-family run retained the same 68,466 top-level projection
items and produced 491 identical cold/warm findings, including 93 semantic
inheritance findings.  It took 107.44 seconds cold and 30.41 seconds warm.
Process-local high-water RSS was 337,992 KB cold and 395,684 KB warm; reporting
the larger value leaves the checkpoint 55.5% below the controlled 889,100-KB
all-at-once parse baseline.

AutoRegisterMeta rent analysis now shares the same compact global class graph.
Registry keys and extractors, registry-reading methods, dynamic factory facts,
and receiver/method consumer edges are normalized while each module AST is
live.  The consumer graph preserves nested-function attribution and all 150,566
DQDock receiver/attribute edges, but interns its names and stores the edges in
1,422,592 encoded bytes rather than retaining Python tuple/object graphs.  A
single relevant-key consumer index replaces repeated scans for every family.
Direct candidate equivalence covers dynamic factories, external consumers,
registry projections, and abstract hooks.  The current partition is 183
per-module detectors, 28 compact-global detectors, and 41 AST-retaining
context-dependent detectors.

The stable-snapshot 28-family run retained the same 68,466 top-level projection
items and produced 520 identical cold/warm findings, including 29 AutoRegister
rent findings.  It took 113.25 seconds cold and 35.41 seconds warm.  Process
high-water RSS was 346,388 KB cold and 393,480 KB warm; the larger value is
55.7% below the controlled 889,100-KB all-at-once parse baseline and is slightly
lower than the 27-family warm peak despite the added global reasoning.

Keyed-registry proof analysis now reconstructs from compact class facts for the
premature-infrastructure, non-injective, and mature-injective registry
detectors.  The shared proof preserves exact key/type mappings, missing keyed
types, duplicate keys and types, reverse lookup names, maturity signals, and
external consumer fanout.  The three detectors build that proof context once
per accumulator run instead of repeating the class/consumer graph traversal.
Repeated ``AutoRegisterByClassVar`` family roots also project their lookup
style, error type, key attribute, and abstract hooks while each module AST is
live.  Direct compact/legacy equivalence gates cover both the full injectivity
fact and repeated-family candidates.  The current partition is 183 per-module
detectors, 32 compact-global detectors, and 37 AST-retaining context-dependent
detectors.

The stable-snapshot 32-family run retained the same 68,466 top-level projection
items and produced 520 identical cold/warm findings.  DQDock contributes no
findings from the four newly migrated registry detectors, but their exact
analysis now runs without repository AST retention.  The paired run took
110.42 seconds cold and 35.59 seconds warm.  Process high-water RSS was 346,700
KB cold and 405,444 KB warm; the larger value remains 54.4% below the controlled
889,100-KB all-at-once parse baseline.

Predicate-selected concrete-family and parallel mirrored-leaf-family analysis
now reconstruct from the same compact inheritance projection.  Predicate
selection retains only locally certified selector method coordinates (line,
selector, predicate, and context parameter); mirrored-family analysis needs no
new syntax facts.  The two detectors share one compact inheritance graph per
accumulator run.  Direct candidate-equivalence tests cover both positive
shapes, and the full suite passes with the partition at 183 per-module
detectors, 34 compact-global detectors, and 35 AST-retaining context-dependent
detectors.

The earlier frozen source snapshot had been removed, so the 34-family benchmark
uses a new frozen copy of the current 919-file production inventory (32,144,215
source bytes).  It retained 68,481 top-level projections and produced 459
identical cold/warm findings; the finding-count difference from the 32-family
checkpoint reflects the changed DQDock source snapshot, not detector drift.
Neither newly migrated detector fires on this snapshot.  The paired run took
113.40 seconds cold and 35.22 seconds warm.  Process high-water RSS was 320,372
KB cold and 374,492 KB warm; the larger value is 57.9% below the controlled
889,100-KB all-at-once baseline and 7.6% below the prior 32-family warm peak.

Manual concrete-subclass roster and latent implementation-roster analysis now
also reconstruct from the shared compact class projection.  Manual roots retain
only normalized ``__init_subclass__`` registration guards, registry consumer
locations, and exact descendant-filter flags.  Collection rosters retain only
member names, projection role, source coordinates, and line count; class facts
add constant-string and non-``None`` assignment summaries for exact key/guard
matching.  All four concrete-family detectors share one reconstructed
inheritance graph.  Compact/legacy candidate-equivalence tests and the existing
shape calibration suite pass.  The collected-family cache schema is bumped to
version 3 so older cached class records cannot deserialize without the new
fields.  The full suite passes 966 tests, and the partition is now 183
per-module detectors, 36 compact-global detectors, and 33 AST-retaining
context-dependent detectors.

On the same frozen 919-file snapshot, the 36-family run retained the same
68,481 top-level projections and produced the same 459 cold/warm findings.  The
new facts comprise zero qualifying manual roots, 858 normalized collection
rosters, and 6,981 constant-string class assignments; neither added detector
produces a DQDock finding.  The paired run took 110.13 seconds cold and 35.51
seconds warm.  Process high-water RSS was 324,564 KB cold and 356,080 KB warm;
the larger value is 60.0% below the controlled 889,100-KB all-at-once baseline
and 4.9% below the 34-family warm peak.

Registry projection-surface and repeated subset-policy authority analysis now
also reconstruct from the shared compact keyed-registry proof.  Top-level
tuple, list, and dict references are normalized while each module AST is live;
constant strings remain distinct from imported aliases, and mapping direction
is retained so key-to-type and type-to-key projections remain exact.  All five
keyed-registry detectors build the shared proof context once.  Direct
compact/legacy equivalence covers cross-module aliases, surface roles, subset
policies, and the legacy reassignment behavior where one name can contribute
both sequence and mapping surfaces.  The collected-family cache schema is now
version 4.  The full suite passes 967 tests, and the partition is 183
per-module detectors, 38 compact-global detectors, and 31 AST-retaining
context-dependent detectors.

On the same frozen 919-file snapshot, the 38-family run retained the same
68,481 top-level projections plus 223 nested named projection surfaces and
produced the same 459 cold/warm findings.  Neither newly migrated detector
fires on this snapshot.  The paired run took 111.11 seconds cold and 35.67
seconds warm.  Process high-water RSS was 325,596 KB cold and 355,908 KB warm;
the larger value remains 60.0% below the controlled 889,100-KB all-at-once
baseline and is 172 KB below the 36-family warm peak despite the added exact
global analysis.

Manual family-roster analysis now also reconstructs from the shared compact
concrete-family context.  It retains only top-level local class rosters and
preserves the legacy detector's simple-name ancestry behavior, including
qualified bases, function-local shadow classes, and first-evidence ordering.
All five related concrete-family detectors build one compact inheritance graph.
Candidate and complete-finding equivalence cover both ordinary constructor
rosters and the function-local shadow edge.  The collected-family schema is
version 5, the full suite passes 967 tests, and the partition is now 183
per-module detectors, 39 compact-global detectors, and 30 AST-retaining
context-dependent detectors.

The same frozen snapshot contributes zero manual family rosters, one sparse
first-location override, and one extra simple-base row.  The 39-family run
retained the same 68,481 top-level projections and produced the same 459
cold/warm findings; the new detector does not fire.  It took 114.60 seconds
cold and 35.69 seconds warm, with 326,048-KB and 366,512-KB high-water RSS.
The larger value is 58.8% below the controlled 889,100-KB all-at-once baseline.
A trial slotted value-object representation reduced cold RSS to 317,688 KB and
the family cache by about 1.7 MB, but repeated warm peaks rose to about 369.8
MB; that trial was rejected in favor of the lower warm-memory representation.

Existing nominal-authority reuse and dataclass implementation-retreat analysis
now also reconstruct from the shared compact class projection.  The projection
retains full legacy simple-name ancestry but stores typed fields only for the
4,866 classes with at least two fields, totaling 22,546 field rows.  It also
preserves ``ast.walk`` class order and function-local classes.  A reusable
authority index keyed by the authority's first typed field replaces the legacy
quadratic compatibility scan without changing candidate ordering or findings.
Candidate and complete-finding equivalence cover both detectors, which share
one reconstructed nominal-authority context.  The collected-family schema is
version 7, the final full suite passes 969 tests, and the partition is now 183
per-module detectors, 41 compact-global detectors, and 28 AST-retaining
context-dependent detectors.

The accepted 41-family run on the same frozen snapshot retained the same
68,481 top-level projections and produced 475 identical cold/warm findings.
The two newly migrated detectors each contribute eight findings.  Cold took
114.13 seconds at 330,888 KB; warm took 35.24 seconds at 377,300 KB.  Runtime
is effectively flat versus the 39-family checkpoint while exact coverage grew,
and the larger RSS remains 57.6% below the controlled 889,100-KB all-at-once
baseline.  Two rejected layouts informed the sparse representation: a separate
duplicated authority family took 202.46 seconds cold, while putting typed fields
on every compact class took 220.03 seconds and 386,280 KB cold because it
crossed a per-object dictionary capacity boundary across roughly 17,000 class
records.  Neither rejected layout is present in the accepted code.

Duplicate nominal-authority surfaces and pass-through nominal wrappers now
also reconstruct from the shared compact class projection.  The projection
adds only surfaces that have at least two typed fields plus public self-field
flow, reusable wrapper authorities, and locally proven forwarding shells.  It
preserves repository ``ast.walk`` order, including function-local authority
classes, before joining wrapper members to the first reusable authority.  The
duplicate-surface component pass now unions exact axis-equality buckets instead
of materializing the equivalent quadratic confusability-graph cliques; an
oracle test proves identical transitive connected components.  Candidate
tuples match the legacy AST algorithms both in focused fixtures and across all
919 frozen DQDock modules.  That full-AST oracle used 1,001,260 KB, illustrating
the representation being removed.  The collected-family schema is version 8,
the final suite passes 971 tests, and the partition is 183 per-module, 43
compact-global, and 26 AST-retaining context-dependent detectors.

DQDock contributes 2,271 duplicate-surface facts, 1,295 reusable wrapper
authorities, and six locally proven wrapper facts.  Duplicate-surface analysis
adds ten findings; pass-through wrappers add none, producing 485 identical
cold/warm findings from the same 68,481 top-level projections.  The accepted
fresh-cache pair took 116.59 seconds cold at 332,432 KB and 38.38 seconds warm
at 385,004 KB; an equivalent-cache repeat took 35.80 seconds at 385,444 KB,
showing that the warm timing variance is in projection loading while finding
reconstruction remains about 8.6 seconds.  The larger observed RSS is 56.6%
below the controlled 889,100-KB all-at-once parse baseline.  Schema 8 increases
the complete persistent cache by only 255,691 bytes relative to schema 7.

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

The checked benchmark command is now ``nominal-refactor-benchmark``.  With a
15-second cold, 5-second warm, and 180-MB ceiling, an isolated-cache DQDock run
measured 12.46 seconds / 135.1 MB cold and 1.65 seconds / 120.4 MB warm.  Warm
detector analysis itself fell from 8.67 seconds to 0.38 seconds after the
bounded lane began reusing canonical per-module finding shards.  Both passes
reported 158 findings, ``focused_local_partial`` status, and clean process
exit.  The same isolated-cache command caught and now covers the remaining
source-signature symlink canonicalization path.

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

Final verification collected 942 tests in one external-root run: all 942
passed in 314.56 seconds.  Keeping the pytest base outside the checkout
preserves the repository-root assumption exercised by formal-boundary source
discovery while avoiding the shared ``/tmp`` quota.

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
