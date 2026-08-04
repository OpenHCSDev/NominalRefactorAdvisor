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

Five related structural ABC/inheritance optimizers now reconstruct from one
shared compact context instead of each retaining the repository ASTs and
rebuilding the same inheritance and method plans.  The projection keeps a
lightweight row for every relevant direct method so that a short sibling still
rejects an otherwise matching long-method family exactly.  Eligible long
methods carry reversible, zlib-compressed normalized skeleton and semantic
coordinate tuples; class declarations carry only the inheritable assignment
metadata required by the class-level optimizer.  Coordinate payloads are
decoded only after statement counts and compressed skeletons match.  A naive
expanded layout was rejected after its prototype serialized 53,718,261 bytes;
the accepted representation increases the complete cache by 3,149,364 bytes.

Focused fixtures compare both candidate tuples and complete findings against
the legacy algorithms, including the short-method poisoning boundary, and
prove that all five detectors reuse one compact context.  A separate oracle
over all 919 frozen DQDock modules compared compact and legacy method plans,
family plans, and all five candidate families exactly.  Every comparison was
equal, including 57 class-level candidates; that oracle peaked at 1,004,896 KB
while holding the legacy AST representation.  The collected-family schema is
version 9, the final suite passes 973 tests, and the partition is now 183
per-module detectors, 48 compact-global detectors, and 21 AST-retaining
context-dependent detectors.

DQDock contributes 9,238 optimizer method rows, including the exact short
method guards, and 8,561 compact declaration rows.  The five migrated
detectors add 57 findings, producing 542 identical cold/warm findings from the
same 68,481 top-level projections.  The accepted fresh-cache cold run took
117.19 seconds at 344,308 KB.  Warm runs took 43.90 seconds at 367,808 KB and
42.63 seconds at 367,708 KB.  Cold runtime is effectively flat versus schema 8
while exact compact coverage grows by five detectors; the larger production
RSS remains 58.7% below the controlled 889,100-KB all-at-once parse baseline.

Available carrier reuse, carrier-composition retreat, and parallel primitive
carrier detection now share one compact carrier context.  Schema 10 stores
8,361 sparse carrier class rows and 9,021 simple-name inheritance edges on the
existing module-class projection.  The rows preserve exact direct annotations,
constructor-derived fields, dataclass state, bases, module imports, constructor
aliases, and source order without retaining AST nodes.  Nominal and carrier
field maps are extracted in one class-body pass, so the new detector coverage
does not duplicate the existing nominal-surface traversal.

Focused gates compare candidates and complete findings against all three
legacy AST collectors, cover aliased forward references, and prove one shared
compact context.  Across all 919 frozen DQDock modules, compact and legacy
projections matched exactly for 3,205 carrier surfaces and 76 primitive
bundles.  Final candidate tuples also matched exactly: 31 available-carrier,
233 composition-retreat, and nine parallel-carrier candidates.  The indexed
available-carrier join separately matched exhaustive authority pairing.  That
full-AST oracle peaked at 1,018,468 KB.  The final suite passes 975 tests, and
the partition is now 183 per-module, 51 compact-global, and 18 AST-retaining
context-dependent detectors.

An initial exact implementation retained the sparse memory result but used an
all-pairs join across 3,205 surfaces and 684 carrier authorities.  It took
150.59 seconds cold, including 30.11 seconds of finding reconstruction, and
was rejected.  The accepted join indexes authorities by the minimum three
shared semantic roles, then runs the unchanged exact predicate on only the
eligible pairs.  Finding reconstruction fell to 11.90 seconds.  The final
fresh-cache run produced 815 findings in 119.39 seconds at 352,348 KB, with
projection itself at 107.50 seconds versus schema 9's 107.42 seconds.  The
matching warm run took 42.17 seconds at 371,028 KB.  The complete persistent
cache grows by only 555,956 bytes, and the larger production RSS remains 58.3%
below the controlled 889,100-KB all-at-once parse baseline.

Private-helper semantic clustering now reuses the compact private-reference
projection instead of retaining repository ASTs.  Schema 11 adds exact
parameter, public-callee, return-kind, constructed-result-type, and global
private-caller summaries to that existing family.  The production projection
contains 3,032 private-function rows and 5,126 per-module private caller-name
rows.  Candidate reconstruction still joins callers across every module, so
the migration preserves repository-wide reasoning while allowing each module
AST to be released after projection.

A full oracle over all 919 frozen DQDock modules compared compact and legacy
candidate tuples and complete rendered findings exactly.  Both paths produced
the same 12 semantic-cluster candidates.  The comparison peaked at 1,039,100
KB because it deliberately retained the legacy full-AST context alongside the
compact facts.  The final suite passes 976 tests, and the partition is now 183
per-module detectors, 52 compact-global detectors, and 17 AST-retaining
context-dependent detectors.

The isolated schema-11 cold run produced 827 findings in 118.70 seconds at
355,044 KB, including 106.83 seconds of projection and 11.87 seconds of finding
reconstruction.  The matching warm run took 42.69 seconds at 382,752 KB,
including 30.79 seconds of projection and 11.90 seconds of finding
reconstruction.  The projection count remains 68,481 because the migrated
detector shares an already-retained family, and the complete persistent cache
grows by 601,382 bytes versus schema 10.

Role-guarded surface access now uses an AST-free role-surface projection.
Schema 12 stores declared class members under both simple and module-qualified
type names, plus local ``isinstance`` guard/access events.  The global join
unions declarations across the repository and intersects each guarded access
with the declared role surface.  DQDock contributes 29,280 per-module role
surface rows and 1,186 local access events.

The full 919-module oracle matched all 32 candidate tuples and rendered
findings exactly, including source order and negative guarded accesses to
members absent from the declared role.  The comparison peaked at 1,001,628 KB
while intentionally holding the compact and legacy full-AST representations
together.  The final suite passes 977 tests, and the partition is now 183
per-module detectors, 53 compact-global detectors, and 16 AST-retaining
context-dependent detectors.

The space-controlled schema-12 cache produced 859 findings from 69,400
top-level projections.  Cold took 124.01 seconds at 359,880 KB, including
112.05 seconds of projection and 11.96 seconds of finding reconstruction.
Warm took 43.16 seconds at 387,000 KB, including 31.07 seconds of projection
and 12.08 seconds of reconstruction.  The complete persistent cache grows by
4,352,145 bytes versus schema 11.  A prior 81.27-second nominally warm sample
was rejected after its cache audit found 7,004 zero-byte entries while
``/tmp`` was under pressure; the accepted pair had 4.1 GB free and zero
zero-byte entries.

Escaped non-nominal private-helper analysis now joins multiple reusable
compact families.  The accumulator's schema-13 multi-family contract lets one
detector consume the existing private-reference and compact class-family
projections without copying either family or introducing another per-module
projection.  The private-reference family adds exact helper call-argument
summaries and statement counts.  It retains all 30,824 caller qualnames to
preserve legacy last-definition-wins behavior, while storing call payloads for
only the 3,105 caller indexes that actually invoke private helpers.

The full 919-module oracle matched all 390 candidate tuples, placement plans,
residue plans, and rendered findings exactly.  The placement distribution is
176 module nominal authorities, 101 boundary strategies, 98 new family
mixins/ABCs, and 15 existing inheritance roots.  The oracle peaked at
1,143,628 KB because it deliberately retained both compact families and the
legacy full-AST context.  The final suite passes 978 tests, and the partition
is now 183 per-module detectors, 54 compact-global detectors, and 15
AST-retaining context-dependent detectors.

The accepted schema-13 cache produced 1,249 findings while the top-level
projection count remained 69,400.  Cold took 124.01 seconds at 369,068 KB,
including 111.32 seconds of projection and 12.70 seconds of finding
reconstruction.  Warm took 44.24 seconds at 397,884 KB, including 31.27
seconds of projection and 12.97 seconds of reconstruction.  The complete
persistent cache grows by 2,616,959 bytes versus schema 12, and both cache
integrity audits found zero zero-byte entries.

Distributed-boundary fanout and local-wrapper-collapse analysis now share one
AST-free per-module boundary projection.  Schema 14 retains compact declaration
facts, class-base rows, and a conservative superset of eligible keyword and
attribute-use facts.  The repository join first resolves inherited class-field
contracts and repeated declaring classes globally, then filters the use
superset and renders both dependent rules from the same exact fanout graph.
This preserves uses in modules that do not themselves declare the field.

Across the frozen 919-module DQDock inventory, the projection contains 26,674
declarations, 16,912 class-base rows, and 83,858 possible uses.  Compact and
legacy output matched all 1,841 fanout candidate tuples and all 17 wrapper
candidate tuples exactly, including declaration subclasses, evidence order,
context tokens, and rendered findings.  The compact join took 0.16 seconds
after projection in the combined oracle.  The final suite passes 980 tests,
and the partition is now 183 per-module detectors, 56 compact-global
detectors, and 13 AST-retaining context-dependent detectors.

The family explicitly opts into a 1-MB per-module cache ceiling while the
generic family limit remains 100 KB.  This is sufficient for all 919 boundary
payloads and avoids rebuilding the 15 largest modules on a boundary-only warm
scan.  The isolated migrated pair produced 1,858 findings from 919 projections.
Its final fresh-cache pass took 33.91 seconds at 159,824 KB, including 33.65
seconds of projection and 0.26 seconds of reconstruction; warm took 2.41
seconds at 144,100 KB, including 2.10 seconds of projection and 0.31 seconds of
reconstruction.

The complete 56-detector schema-14 run produced 3,107 findings from 70,319
top-level projections.  Cold took 138.54 seconds at 425,916 KB (125.32
projection, 13.22 reconstruction); warm took 46.03 seconds at 455,712 KB
(32.87 projection, 13.16 reconstruction).  The cache occupies 191,029,774
payload bytes, 13,038,588 more than schema 13, and both integrity audits found
zero zero-byte entries.  The full run now performs 1,858 more finding
reconstructions than schema 13, so its headline time and RSS are not a
like-for-like workload comparison.

Role-surface drift and generic role-case-table analysis now share one compact
role-surface projection.  Schema 15 stores declared field-role tokens, a
conservative superset of eligible structural attribute uses, and local
case-table sites collected with a configuration-independent one-case floor.
The global joins apply the active detector thresholds later.  Role drift joins
the existing compact class family so inherited-field uses are still suppressed
through the complete repository inheritance graph; generic case tables need
only the shared role projection.

The frozen 919-module DQDock oracle contains 24,027 role declarations, 45,239
possible structural uses, and 657 raw case-table sites.  Its compact payloads
serialize to 8,213,787 bytes, with a 314,643-byte largest module.  Compact and
legacy output matched all 982 role-drift candidates and all 137 generic
case-table candidates exactly.  The combined equivalence process peaked at
1,102,948 KB because it deliberately retained the AST indexes and both compact
families together.  Configuration-override gates also compare strict role-use
and case-count thresholds.

Profiling exposed a legacy quadratic owner lookup: each of 12,915 field groups
rescanned every indexed repository class.  One exact
``(file_path, simple_name) -> class symbols`` index reduces compact role finding
construction from about 18 seconds to 0.50 seconds.  A DQDock gate compared all
12,915 indexed results with the former full scan.  The final suite passes 981
tests, and the partition is now 183 per-module detectors, 58 compact-global
detectors, and 11 AST-retaining context-dependent detectors.

The accepted schema-15 cache produced 4,226 findings from 71,238 top-level
projections.  Cold took 147.95 seconds at 460,140 KB, including 133.17 seconds
of projection and 14.77 seconds of finding reconstruction.  Warm took 47.89
seconds at 493,572 KB, including 33.00 seconds of projection and 14.89 seconds
of reconstruction.  The cache occupies 199,855,570 payload bytes, 8,825,796
more than schema 14, and both integrity audits found zero zero-byte entries.
Compared with schema 14, the warm total grows by only 1.86 seconds while
performing 1,119 additional exact finding reconstructions.

## 2026-08-04 nominal-bypass projection checkpoint

ABC-polymorphism bypass and algebraic variant-method-family analysis now share
one AST-free per-module projection.  It retains concrete-dispatch scatters,
cross-class method templates, wrapper chains, cancelable product-composition
signals, and variant-method surfaces; the bypass detector joins those facts to
the existing complete compact inheritance graph.  Composition target IDs are
projected from the live module AST and the same stable source-index geometry,
so the compact path preserves related evidence without reparsing source text.

The frozen 919-module DQDock oracle matched both legacy bypass findings and its
zero variant-family findings object-for-object.  It also matched all 16 raw
``isinstance`` scatter candidates.  A relevance prepass reduced scatter
projection from 9.57 to 2.38 seconds while preserving the legacy repeated-walk
attribution and ordering for every function that can contribute.  The complete
new family takes about 11.2 seconds when all repository ASTs are already live.

In isolation, the migrated pair completed from a bounded stream at 193,880 KB
peak, versus 1,888,896 KB for the retained-AST legacy equivalence run, an 89.7%
reduction.  The isolated wall times are not comparable because the compact
pair includes module streaming and the shared class projection that the full
compact scan pays once.  The partition is now 183 per-module detectors, 60
compact-global detectors, and nine AST-retaining context-dependent detectors.

The accepted schema-16 cache produced 4,228 findings from 72,157 top-level
projections.  Cold took 162.13 seconds at 460,364 KB, including 146.05 seconds
of projection and 16.07 seconds of finding reconstruction.  Warm took 51.02
seconds at 491,604 KB, including 35.49 seconds of projection and 15.53 seconds
of reconstruction.  The cache contains 18,118 files and 201,188,173 payload
bytes with zero zero-byte entries.  The new family contributes 919 payloads,
1,332,603 bytes total, and a 17,476-byte largest module, so the generic 100-KB
per-family ceiling covers the entire corpus.
Checkpoint verification passes all 983 tests in 340.19 seconds.

## 2026-08-04 semantic-descent projection checkpoint

Semantic-mirror-without-descent analysis now builds its repository graph from
one deferred semantic module family plus the existing compact class family.
Per-module facts retain presentation tokens, construction shapes, unresolved
class-reference parts, and sparse enum/dataclass/materializer supplements.
Class references are resolved only after the complete compact import and
inheritance graph exists.  The final graph does not retain a class AST index.

The frozen 919-module DQDock oracle matches the retained-AST graph exactly at
every layer: 7,236 authorities, 52,151 facts, 20,652 presentation projections,
and all 5,246 mirror edges and certificates.  Compact graph reconstruction took
5.88 seconds in the combined oracle versus 13.17 seconds for the legacy graph.
The combined process intentionally retained both representations and is not a
bounded-memory measurement.

The isolated compact detector produced all 5,246 findings at a 348,472-KB cold
peak and 351,824-KB warm peak.  Cold took 63.41 seconds (60.55 projection, 2.86
finding reconstruction); warm took 22.47 seconds (19.13 projection, 3.33
reconstruction).  The old isolated retained-AST graph reached 1,090,768 KB in
the direct oracle, while the earlier detector-order profile reached about 1.58
GB.  The compact comparison therefore removes at least 68% of the measured
semantic detector's isolated high-water mark.

Schema 17 moves the partition to 183 per-module detectors, 61 compact-global
detectors, and eight AST-retaining context-dependent detectors.  The complete
compact run produced 9,474 findings from 73,076 top-level projections.  Cold
took 174.39 seconds at 570,696 KB, including 155.57 seconds of projection and
18.82 seconds of reconstruction.  Warm took 54.72 seconds at 604,292 KB,
including 35.32 seconds of projection and 19.40 seconds of reconstruction.
The cache contains 19,037 files and 219,514,657 payload bytes with zero
zero-byte entries.  The semantic family contributes 919 payloads and
18,326,484 bytes; its largest module is 265,448 bytes, below the explicit 1-MB
family ceiling.

The semantic detector no longer materializes the legacy semantic-graph cache
during ordinary exact analysis.  Evidence-local partial scans consequently
omit changed semantic-mirror findings like other compact-global detectors;
the next exact scan recomputes them from source-validated compact family
payloads.  Regression coverage now states that boundary explicitly while
retaining separate tests for the legacy overlay API used by explicit graph
consumers.
Checkpoint verification passes all 984 tests in 349.69 seconds.

## 2026-08-04 available-abstraction projection checkpoint

Available-abstraction reuse now projects focused authority and local
implementation capability signatures per module, then performs the existing
availability, overlap, coverage, and best-authority join over those immutable
facts.  The legacy retained-AST collector and the compact join share the same
candidate authority, so this migration does not fork detector semantics.

The frozen 919-module DQDock oracle produced the same 79 candidates and 79
findings on both paths; canonical candidate and finding digests match exactly.
The isolated retained-AST run took 22.28 seconds and peaked at 1,009,284 KB,
including a 9.88-second global join.  The uncached bounded run took 27.83
seconds and peaked at 210,840 KB, including a 4.11-second global join.  The
79.1% peak-memory reduction is authoritative; the cold end-to-end wall times
are not directly comparable because only the compact path streams and releases
module ASTs.

The family contains 1,720 authority signatures and 8,998 local signatures.
Its 919 cache payloads occupy 16,699,920 bytes; the largest is 839,140 bytes,
so an explicit 1-MB family ceiling covers the whole corpus.  Isolated cold
projection plus finding reconstruction took 28.96 seconds at 212,608 KB.  With
all family payloads cached, warm took 3.19 seconds at 192,048 KB.

Schema 18 moves the partition to 183 per-module detectors, 62 compact-global
detectors, and seven AST-retaining context-dependent detectors.  The complete
compact run produced 9,553 findings from 73,995 top-level projections.  Cold
took 187.28 seconds at 668,352 KB, including 165.64 seconds of projection and
21.64 seconds of reconstruction.  Warm took 58.87 seconds at 720,856 KB,
including 37.68 seconds of projection and 21.18 seconds of reconstruction.
The cache contains 19,956 files and 236,213,431 payload bytes with zero
zero-byte entries.
Checkpoint verification passes all 985 tests in 352.12 seconds.

## 2026-08-04 public/private delegate projection checkpoint

Public-API/private-delegate shell and family analysis now share one per-module
projection containing forwarding-wrapper facts, top-level delegate locations,
and import-resolved call targets.  The two compact detectors build one shared
global context, so external-call matching and delegate-family grouping run once
per exact scan.  The legacy AST entry points and compact path share the same
fact-to-candidate authorities.

The frozen 919-module DQDock oracle has zero shell and family candidates on
both paths with matching canonical candidate and finding digests; a separate
non-empty three-module oracle matches two shell candidates and one family
candidate object-for-object.  The isolated retained-AST DQDock run took 22.77
seconds at 964,412 KB, including a 9.78-second global join.  The uncached
bounded pair took 26.57 seconds at 155,396 KB, including a 0.12-second shared
join.  That is an 83.9% peak-memory reduction; as in the abstraction checkpoint,
cold wall time includes streaming and release work that the legacy run omits.

The projection contains 95 forwarding wrappers and 17,456 per-module resolved
call targets.  Its 919 cache payloads occupy 8,362,395 bytes; the largest is
248,868 bytes and fits the explicit 1-MB ceiling.  With all payloads cached,
the isolated detector pair takes 0.95 seconds at 125,840 KB.

Schema 19 moves the partition to 183 per-module detectors, 64 compact-global
detectors, and five AST-retaining context-dependent detectors.  The complete
compact run produced 9,553 findings from 74,914 top-level projections.  Cold
took 188.39 seconds at 698,900 KB, including 167.04 seconds of projection and
21.34 seconds of reconstruction.  Warm took 58.68 seconds at 749,588 KB,
including 37.29 seconds of projection and 21.39 seconds of reconstruction.
The cache contains 20,875 files and 244,574,920 payload bytes with zero
zero-byte entries.
Checkpoint verification passes all 986 tests in 322.26 seconds.

## 2026-08-04 spec-axis and shape-guard projection checkpoint

Cross-module spec-axis authority and repeated validate-shape guard analysis now
persist their existing per-module semantic records.  Exact scans flatten seven
spec families and 35 normalized validate-method records, then run the unchanged
pairwise axis comparison and maximal guard-clique grouping algorithms.

The frozen 919-module DQDock oracle matches exactly: both paths have zero
spec-axis candidates and the same two validate-shape candidates and findings,
with matching canonical digests.  The retained-AST pair took 14.71 seconds at
930,864 KB, including a 2.22-second join.  The uncached bounded pair took 19.52
seconds at 133,340 KB, an 85.7% peak-memory reduction.  Its post-projection
grouping took less than one millisecond.

The two families produce 1,838 payloads totaling 1,374,838 bytes; the largest
is 2,766 bytes, well below the generic 100-KB ceiling.  Isolated cold took
23.48 seconds at 134,020 KB, while warm took 0.61 seconds at 78,016 KB.

Schema 20 moves the partition to 183 per-module detectors, 66 compact-global
detectors, and three AST-retaining context-dependent detectors.  The complete
compact run produced 9,555 findings from 76,752 top-level projections.  Cold
took 199.22 seconds at 695,136 KB, including 176.58 seconds of projection and
22.64 seconds of reconstruction.  Warm took 60.69 seconds at 752,704 KB,
including 39.26 seconds of projection and 21.43 seconds of reconstruction.
The cache contains 22,713 files and 245,949,440 payload bytes with zero
zero-byte entries.
Checkpoint verification passes all 986 tests in 375.78 seconds.

## 2026-08-04 zero retained-AST global checkpoint

The final three context-wide detectors now share one systemic module projection
and the existing compact class graph.  It retains unresolved concrete-type
checks for post-graph class resolution, concrete mixin cast/access summaries,
and infrastructure declarations plus per-module reference counts and owner
symbols.  Reference sites are reduced to the exact sufficient statistics used
by rent analysis; no line-level locations or repository ASTs survive the
module stream.

The frozen 919-module DQDock oracle matches all three legacy collectors and
renderers exactly: zero repeated concrete-type candidates, zero implicit-self
mixin candidates, and nine under-amortized infrastructure candidates/findings,
with matching canonical digests.  The retained-AST triple took 26.85 seconds at
1,272,516 KB, including a 13.65-second global join.  The final uncached bounded
triple took 63.46 seconds at 281,728 KB, including 61.78 seconds of projection
and 1.68 seconds of finding reconstruction.  This is a 77.9% peak-memory
reduction; cold wall time includes streaming, release, and construction of the
shared compact class graph that the legacy detector-order measurement retains.

The systemic family projects 755 possible concrete-type functions and 174,964
module-local reference-symbol summaries on DQDock.  Its 919 payloads occupy
30,435,447 bytes; the largest is 727,629 bytes, below the explicit 1-MB ceiling.
With both the systemic and class families cached, the isolated triple takes
20.10 seconds at 298,780 KB.

Schema 21 moves the partition to 183 per-module detectors, 69 compact-global
detectors, and zero AST-retaining context-dependent detectors.  The complete
compact run produced 9,564 findings from 77,671 top-level projections.  Cold
took 206.68 seconds at 775,948 KB, including 183.04 seconds of projection and
23.64 seconds of reconstruction.  Warm took 63.27 seconds at 845,152 KB,
including 39.02 seconds of projection and 24.25 seconds of reconstruction.
The cache contains 23,632 files and 276,385,400 payload bytes with zero
zero-byte entries.
Checkpoint verification passes all 986 tests in 367.58 seconds.

## 2026-08-04 exact CLI compact-orchestration checkpoint

The lightweight exact CLI now uses the compact engine in production.  It
streams one source module at a time, reuses or computes the 183 module-local
detector shards, accumulates the 69 exact global projection families, releases
the module AST, and only then performs repository-wide joins.  A source-derived
per-module cache identity is exactly compatible with the parsed-module identity,
so complete local cache hits and complete projection-family hits bypass AST
deserialization.  The final aggregate is published under the same exact cache
identity used by the pre-parse fast path.

On the frozen 919-module DQDock snapshot, the first integrated whole-repository
CLI pass completed all 252 detectors and produced 12,570 findings in 250.14
seconds at 881,776 KB.  Its reported split was 62.26 seconds of streamed
preparation and 181.90 seconds of local plus global analysis.  This pass reused
the schema-21 family cache but populated the new local and aggregate finding
shards.  The next identical CLI invocation returned the same 12,570 findings in
0.99 seconds at 72,724 KB; the exact aggregate lookup itself took 0.082 seconds.

A file-focused exact scan of `definitive_refinement.py`, with all 919 modules as
reasoning context, completed all 252 detectors and returned 333 in-scope
findings in 79.00 seconds at 835,988 KB.  It spent 45.76 seconds loading and
validating compact facts and 31.49 seconds reconstructing findings.  The path is
now bounded and complete, but this cold/changed-context latency and peak remain
above the interactive target.  The next representation step is family-at-a-time
loading and joining so every compact family need not coexist in memory.
Checkpoint verification passes all 987 tests in 336.99 seconds.

## 2026-08-04 family-at-a-time exact-join checkpoint

Exact compact orchestration no longer retains all 25 projection families at
once.  Detectors are grouped by their declared family tuple; the compact class
graph stays live as the shared anchor for its five multi-family joins, while
each companion and single-family projection is loaded, joined, and released in
turn.  Shared contexts still span every detector consuming the same live family,
and every loaded item is revalidated against the no-AST contract.

Each module now has one source- and family-set-specific completeness marker.
After the marker exists, streamed preparation performs one marker check rather
than 25 cache-entry checks.  Lazy payload loading still validates every pickle
and reparses just that source/family if an entry is absent or corrupt.  The
compact class, inheritance-method, and repeated-builder families now have an
explicit 3-MB per-module ceiling so large but AST-free DQ modules persist rather
than being reparsed on every exact run.  On the frozen inventory all three have
919 payloads; their observed maxima are 687,144, 2,003,228, and 472,469 bytes.

Before bundle markers, the same 333-finding `definitive_refinement.py` exact
scan fell from 79.00 seconds / 835,988 KB to 66.93 seconds / 449,552 KB.  Once
all 919 module bundles were complete, a fresh exact scan of
`definitive_refinement_sparse_projection.py` completed all 252 detectors with
12 in-scope findings in 34.38 seconds at 406,464 KB.  Its reported preparation
time was 0.56 seconds and analysis time was 32.65 seconds.  Relative to the
first integrated focused exact run, this is 56.5% less wall time and 51.4% less
peak memory while preserving complete repository context.
Checkpoint verification passes all 988 tests in 373.52 seconds.

## 2026-08-04 report-independent global-result checkpoint

Focused exact aggregate identities remain report-path-specific, but the 69
compact-global detector outputs are now also published once under an unfiltered
repository-context identity.  Moving to another focused file with unchanged
context loads that exact global family, runs or reuses only the requested
module-local shard, filters the combined findings, and publishes the new focused
aggregate.  Global reasoning is not approximated: the reusable payload was
produced from all 919 module projections and is invalidated by source, config,
detector-registry, Python, and engine identity.

On the frozen DQ inventory, the first publisher scan of
`definitive_refinement_sparse_projection.py` completed all 252 detectors with
12 in-scope findings in 37.91 seconds at 415,288 KB.  A subsequent exact scan of
`definitive_refinement.py` reused the same complete global result and returned
333 in-scope findings in 9.69 seconds at 173,972 KB.  Its reported preparation
and analysis times were 3.99 and 4.45 seconds.  A synthetic two-target parity
gate forces the global join to fail on the second scan and proves that the
report-independent cache still matches independently filtered full-AST output.
Checkpoint verification passes all 989 tests in 434.58 seconds.

## 2026-08-04 shared compact class-repository checkpoint

The 34 detectors that consume the compact module-class family now share one
scan-scoped repository context.  That context constructs the exact 919-module
inheritance graph once and lazily memoizes detector-group derivatives such as
ABC optimizer plans, carrier-reuse candidates, concrete-family rosters, keyed
axis specifications, and keyed-registry facts.  Exact-type guards,
AutoRegister rent, semantic-family SSOT, inherited registry configuration, and
axis-authority joins all consume the same graph instead of reconstructing it.
This preserves global reasoning: the shared index still contains every compact
class declaration and every resolved cross-module ancestor/descendant edge.

On the frozen DQDock inventory, the isolated 34-detector class-family join
returned the same 538 unfiltered findings while falling from 11.23 seconds to
4.06 seconds, a 63.9% reduction.  Loading the 919 cached projections took 1.88
seconds and the isolated process peaked at 203,292 KB.  A regression mixes
detectors from the structural, abstraction-reuse, runtime, systemic, and
surface modules, makes every detector-local class-index builder fail, and
asserts that the repository builder runs exactly once.  Checkpoint verification
passes all 990 tests in 343.83 seconds.

The first complete global-result publisher after this change returned the same
12 focused findings in 31.50 seconds at 404,608 KB, including 0.77 seconds of
preparation and 30.69 seconds of analysis.  Relative to the preceding
report-independent publisher checkpoint, shared class reasoning removed 6.41
seconds (16.9%) and 10,680 KB from the end-to-end first pass.

## 2026-08-04 indexed abstraction-availability checkpoint

Available-abstraction reuse no longer compares every local implementation with
every repository authority.  The exact availability predicate already requires
either an imported authority name or a shared-path authority in the same
top-level package, so the join now indexes authorities by those coordinates and
only then evaluates the unchanged five-capability-atom overlap rule.  This is a
lossless join bound, not a heuristic: an exhaustive synthetic parity gate uses
the original Cartesian predicate and must return the same best candidates.

DQDock contains 1,720 abstraction authorities and 8,998 local implementation
signatures.  The former join considered 15,476,560 pairs to return 79
candidates; the indexed join returns the same 79 in 0.14 seconds instead of
about 2.07 seconds.  A fresh complete global-result publisher now returns the
same 12 focused findings in 28.41 seconds at 403,004 KB, with 0.76 seconds of
preparation and 27.61 seconds of analysis.  That is 25.1% less wall time and
3.0% less peak memory than the 37.91-second / 415,288-KB publisher before the
shared-class and indexed-availability checkpoints.
Checkpoint verification passes all 990 tests in 359.48 seconds.

## 2026-08-04 shared multi-family context checkpoint

Compact shared contexts now support multi-family joins as well as single-family
joins.  The three remaining-systemic detectors consume both the 919-module
systemic projection and the 919-module class projection; they previously built
the same exact inheritance graph independently.  They now share one graph while
retaining both complete projection families and their unchanged candidate
logic.  A regression replaces their common group-context builder with a counter
and requires one construction across all three detectors.

On the frozen DQDock inventory, the isolated three-detector join preserved the
same result counts (zero repeated-concrete-type findings, zero implicit-mixin
findings, and nine under-amortized-infrastructure findings) while completing in
0.42 seconds.  A fresh complete global-result publisher returned the same 12
focused findings in 27.94 seconds at 406,708 KB, including 0.76 seconds of
preparation and 27.14 seconds of analysis.  This is 26.3% below the
37.91-second publisher before the shared-context and exact-index checkpoints.
Checkpoint verification passes all 991 tests in 344.86 seconds; one unrelated
CLI subprocess test failed once with empty output in the preceding run and then
passed both in isolation and in the clean full-suite rerun.

## 2026-08-04 store-validated projection checkpoint

The compact class anchor is now shared across every class-dependent global
join, not only detectors with the same family tuple.  The single-family class
repository context seeds the raw class index used by semantic descent, nominal
bypass, role-surface drift, private-helper, and remaining-systemic joins.  The
bounded manifest carries that context across family-at-a-time joins, reducing
the exact scan from six repository class-index builds to one.  A regression
executes all five distinct multi-family shapes plus a single-family class
detector and makes any detector-local rebuild fail.

Group timing then exposed a larger warm-cache cost.  On the frozen 919-module
snapshot, loading 25 compact families consumed 9.40 seconds, the detector joins
consumed 10.19 seconds, and explicit module-memory release consumed only 0.23
seconds.  Recursive ``ast.AST`` retention checks were rewalking every nested
field of all 77,671 already persisted projections on each exact analysis-cache
miss.  Compact payloads are now checked before the family cache writes them;
uncached fallbacks and repairs are checked at their insertion boundary.  Warm
typed payload loads therefore preserve the syntax-free representation contract
without repeating the full object-graph traversal.

With a fresh analysis-output cache and a certified complete compact-family
cache, the focused publisher reproduction changed from 27.97 seconds / 430,908
KB to 18.77 seconds / 427,608 KB.  Both runs reconstructed the same 77,671
projections and returned the same 12 findings.  That is a 9.20-second (32.9%)
runtime reduction with effectively unchanged peak memory.  Existing payloads
without the certificate are checked and upgraded once; that compatibility pass
completed in 26.23 seconds with the same output.  This remains exact
repository-global reasoning; no family was sampled, omitted, or reduced to a
local approximation.  Checkpoint verification passes all 994 tests in 369.28
seconds; the 11 warnings are the existing concurrent pytest cleanup race.

## 2026-08-04 bounded global-result shard checkpoint

The focused exact path no longer retains every repository-global finding until
one aggregate cache write.  Phase-level RSS sampling showed 9,564 findings
alive together at that boundary; the semantic-mirror detector alone contributed
5,246 intermediate findings and roughly 95 MB of live object growth.  The
bounded manifest now sends each completed detector result to a consumer, which
persists one report-independent ``GlobalDetectorAnalysisCacheIdentity`` shard,
selects any evidence relevant to the current report scope, and releases the
complete repository result before loading the next projection family.  A
regression requires this non-retaining consumer mode, one shard per detector,
and no replacement family aggregate.

The class repository context now also releases single-family derived indexes
before class-dependent multi-family joins, then releases the raw class context
after the final such join.  Later independent families no longer inherit the
class-analysis memory floor.  The class graph itself remains shared while it is
needed, so semantic descent, role drift, nominal bypass, private-helper, and
systemic joins retain the same resolved repository lineage.

On the frozen 919-module DQDock snapshot, a fresh exact publisher miss now
returns the same 12 focused findings from the same 77,671 projections in 17.95
seconds at 371,472 KB.  Compared with the preceding certified-cache checkpoint,
that is 56,136 KB (13.1%) less peak RSS and 0.82 seconds faster.  Compared with
the directly instrumented 27.97-second / 430,908-KB pre-validation run, it is
35.8% faster and 13.8% lower in peak memory.  A second report target reused all
global detector shards without loading projections: it returned 333 exact
focused findings in 7.85 seconds at 129,320 KB with ``partial`` cache status.
No detector, authority, or context-dependent match is sampled or omitted.
Checkpoint verification passes all 995 tests in 359.38 seconds; the 11 warnings
are the existing concurrent pytest cleanup race.

## 2026-08-04 semantic and method-key compaction checkpoint

The exact join now consumes each detector result immediately instead of
retaining all results from the same projection-family group.  Semantic descent
also reuses persisted presentation projections and key/value pairs when class
resolution does not change them, and computes a fact's normalized aliases on
demand instead of retaining an extra tuple on each of 52,151 facts.  On the
frozen inventory, 16,820 of 20,652 semantic projections have no direct class
reference to rewrite.  The isolated semantic build peak fell from 354,724 KB
to 343,812 KB while stable analysis time remained approximately 5.9 seconds.

The later inheritance-method join was the remaining whole-process high-water.
It contains 26,756 method shapes, and the old grouping path copied each often
large normalized AST fingerprint into a concatenated string fiber key.  The
compact path now groups by a tuple that references the persisted fingerprint,
then derives cohorts directly from those method carriers without constructing
``StructuralObservation`` or ``ObservationGraph`` mirrors.  In isolation,
fiber grouping rose only 3,332 KB above the loaded-family baseline instead of
43,132 KB, and took 0.028 seconds instead of 0.054 seconds.  Compact/full-AST
parity covers overlapping three-class cohorts and the cross-domain public
method guard.

On the frozen 919-module DQDock snapshot, a fresh exact publisher miss returns
the unchanged 12 focused findings from 77,671 projections in 16.96 seconds at
361,448 KB.  Against the preceding pushed checkpoint this is 10,024 KB (2.7%)
less peak RSS and 0.99 seconds (5.5%) faster.  Against the directly
instrumented 27.97-second / 430,908-KB pre-validation run, exact analysis is
39.4% faster and 16.1% lower in peak memory.  Full verification passes all 995
tests in 381.54 seconds; the 11 warnings remain the existing concurrent pytest
cleanup race.

### Fact-token index follow-up

The semantic resolver previously allocated one ``_FactTokenReference`` plus
globally sorted pair tuples for every normalized fact alias.  The exact index
now groups references to the already-live ``SemanticFact`` carriers in one
pass, transfers each list into its final tuple without retaining both complete
representations, and preserves deterministic authority/fact ordering.  A
regression requires the token index to return the original fact objects even
when its input arrives out of order.  The name-normalization working sets are
also capped at 8,192 variant entries and 4,096 token-set entries, bounding
their growth within a single large-repository join.

The isolated semantic peak fell by a further 8,628 KB, from 339,364 KB before
the token-index change to 330,736 KB with the bounded cache, while graph build
time remained approximately 2.1 seconds.  A fresh frozen-DQDock exact miss now
returns the same 12 focused findings and 77,671 projections in 16.61 seconds
at 353,688 KB.  Relative to the immediately preceding pushed checkpoint that
is another 7,760 KB (2.1%) and 0.35 seconds (2.1%) reduction.  Relative to the
27.97-second / 430,908-KB pre-validation run, exact analysis is now 40.6%
faster and 17.9% lower in peak memory.  Full verification passes all 996 tests
in 350.03 seconds; the 11 warnings remain the existing concurrent pytest
cleanup race.

### Semantic lineage and edge-reference follow-up

Mirror resolution no longer caches a ``ProjectionOwnerSymbol`` wrapper on all
20,652 projections merely to split its qualified owner name.  Class ancestor
sets are also resolved on demand: DQDock needs 5,670 of the 7,236 possible
sets.  Finally, mirror edges share one lightweight ``SemanticFactReference``
per matched fact instead of constructing one per edge occurrence.  The frozen
inventory retains 7,530 shared references for 18,912 edge occurrences while
preserving the serialized graph shape.  Regressions require compact/full-AST
graph parity, absence of cached projection-owner wrappers, and identity reuse
for repeated fact references.

These changes reduce the isolated mirror-edge peak from 330,968 KB to 324,240
KB while edge resolution remains approximately 1.39 seconds.  A fresh exact
DQDock miss returns the unchanged 12 focused findings and 77,671 projections
in 16.68 seconds at 346,492 KB.  That is another 7,196 KB (2.0%) below the
preceding pushed checkpoint, with the 0.07-second wall-time difference inside
run-to-run noise.  Relative to the 27.97-second / 430,908-KB pre-validation
run, exact analysis is 40.4% faster and 19.6% lower in peak memory.  Full
verification passes all 996 tests in 349.87 seconds; the 11 warnings remain the
existing concurrent pytest cleanup race.

### Lazy semantic specificity and anchor reclamation follow-up

The dataclass mirror policy previously built a map from every fact-role name to
the complete set of owning authority ids, although its only question is whether
the role occurs under more than one authority.  Exact resolution now builds an
8,213-key reused-role set and leaves the legacy full map unmaterialized.  The
projection-to-dataclass descent index is demand-driven as well: only 936 of
20,652 projection lineage rows are required on frozen DQDock.  Regressions
require both lazy indexes to remain unmaterialized until queried while
preserving compact/full-AST graph parity.

After the final class-dependent multi-family join, the bounded manifest now
performs one real cycle collection as it clears the shared class/semantic
contexts.  This matters for subsequent large independent families: the
available-abstraction join fell from an approximately 349-MB process high-water
to approximately 311 MB in the instrumented phase run.  A boundary regression
requires this collection at anchor teardown; per-family joins still avoid
repeated collections.

Isolated semantic edge resolution fell from 324,240 KB / about 1.39 seconds to
310,336 KB / about 1.14 seconds with the same 5,246 edges.  Two fresh exact
DQDock misses stabilized at 333,472 and 333,524 KB; the repeat returned the
unchanged 12 focused findings and 77,671 projections in 16.68 seconds.  Against
the preceding pushed checkpoint this is 12,968 KB (3.7%) less peak RSS with
unchanged wall time.  Against the 27.97-second / 430,908-KB pre-validation run,
exact analysis is 40.4% faster and 22.6% lower in peak memory.  Full
verification passes all 996 tests in 357.26 seconds; the 11 warnings remain the
existing concurrent pytest cleanup race.

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
