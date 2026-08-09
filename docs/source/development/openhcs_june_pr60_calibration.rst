OpenHCS June And PR #60 Calibration
===================================

This page calibrates NRA against OpenHCS history from 1 June 2026 onward, with
the final cleanup of PR #60 as the strongest labeled case.  It extends the
older OpenHCS survival and diff studies with exact full-package scans and an
explicit false-positive audit.

The purpose is not to reward a detector because it emits many findings.  The
question is whether its findings concentrate on code that later corrective
work removed or replaced, and whether the proposed remediation matches the
architecture that survived.

Corpus And Ground Truth
-----------------------

The OpenHCS main history from 1 June through 6 August 2026 contains 558 commits:

.. list-table:: Commit volume
   :header-rows: 1

   * - Month
     - Commits
   * - June
     - 83
   * - July
     - 414
   * - August through the current head
     - 61

Git ``numstat`` reports 791,342 added and 446,663 deleted lines over 6,777 file
touches.  Generated references and benchmark artifacts contribute to that
volume, so raw churn is inventory evidence rather than a quality score.

The strongest positive labels come from commits whose implementation and
deletion gates agree about the intended authority move:

- ``24c24eb28`` centralizes runtime source provenance instead of deriving it
  from filenames and local component heuristics.
- ``9110b1047`` splits artifact binding, source candidates, measurement
  materialization, output recording, and plane projection into explicit
  authorities.
- ``63f0ede15`` moves CellProfiler semantics from parallel catalogs to module
  declarations.
- ``d90f5de10`` and ``b7e3bf001`` compile runtime ownership and callable ABI
  once, leaving execution as a mechanical consumer.
- ``153662382`` and ``6647a3925`` derive MCP tools and renderer contracts from
  capability declarations instead of maintaining transport-side mirrors.
- ``ccfef5f6d`` completes the runtime artifact owner cutover and removes
  superseded compatibility layers.
- ``5e8812ee8`` removes the parallel CellProfiler compiler, compatibility,
  runtime, and benchmark-library lattices.
- ``fe13c573e`` deletes a hard-coded PNG normalization table and path-format
  inference in favor of the generic payload intensity authority.
- ``ed6d7eb6b`` deletes OpenHCS's parallel performance-monitor configuration
  and delegates to the pyqt-reactive owner.
- ``c8abd9ed7`` replaces closed strings and generated display-config mirrors
  with enum/dataclass declarations and existing registry strategies.

The PR #60 branch also provides a direct negative chain.  Earlier commits
explicitly describe an LLM-absorbed CellProfiler library, LLM-inferred runtime
categories, generated mappings, and successive corrections to those inferred
categories.  The final PR removes that copied library and its parallel
compiler/runtime lattice.  The negative label is therefore the parallel,
inferred authority structure, not the mere fact that an LLM contributed code.

Repeated Durable Fix Patterns
-----------------------------

The successful changes converge on a small set of architectural moves.

.. list-table:: Durable pattern taxonomy
   :header-rows: 1

   * - Pressure
     - Durable normal form
     - Rejected drift shape
   * - Domain cases evolve together
     - One enum, declaration family, or registered nominal owner with derived
       views
     - copied lists, module-name tables, string dispatch, and consumer-owned
       aliases
   * - Compile and runtime disagree
     - compile declaration-owned contracts once; execute the compiled plan
       mechanically
     - runtime signature inspection, hidden rebinding, and per-module recovery
   * - Artifacts lose identity across stages
     - typed artifact, source, producer, axis, and provenance carriers shared
       across planning, execution, materialization, and viewing
     - filename inference, incidental dictionary keys, and reconstructed paths
   * - Transport needs domain data
     - domain DTO/service authority projected into MCP, ZMQ, Qt, or generated
       source
     - transport-local schemas and duplicated coercion or accepted-value lists
   * - Similar consumers repeat policy
     - policy lives on the declaration/owner and consumers derive projections
     - parallel defaults, exclusion sets, special-case tables, and prompt-side
       type rosters
   * - Concrete family behavior is recovered by callers
     - polymorphic hook or registered strategy on the family
     - scattered ``isinstance`` checks, role guards, and substring classifiers
   * - Missing or stale state is hidden
     - fail-loud typed result, explicit lifecycle state, or bounded retry owned
       at the boundary
     - ``getattr``/``hasattr`` probing, ``or ''`` chains, swallowed exceptions,
       and fabricated fallback defaults
   * - Infrastructure exists in several packages
     - one extracted owner with typed dependency projection
     - repository-local compatibility copy or a second framework
   * - A helper layer only renames operations
     - direct use of the owning public API, or one facade with real invariants
     - pass-through wrappers, dangling helper piles, and private/public shadows

Exact Snapshot Experiment
-------------------------

All scans below analyze only the production ``openhcs`` package, exclude tests,
run all 252 detectors, use exact compact-global mode, and omit caches.  This
keeps submodule and benchmark-data changes from confounding the comparison.

.. list-table:: Exact package snapshots
   :header-rows: 1

   * - Snapshot
     - Meaning
     - Raw findings
   * - ``4bc91c242``
     - exact parent of the runtime-owner cutover
     - 7,558
   * - ``ccfef5f6d``
     - runtime artifact owner cutover
     - 7,142
   * - ``1398c8662``
     - exact parent of the compiler cleanup
     - 7,115
   * - ``5e8812ee8``
     - parallel compiler/runtime cleanup
     - 7,247
   * - current OpenHCS package
     - post-PR evolution through ``5d124139a``
     - 7,691

The runtime cutover reduces the aggregate by 416, but the compiler cleanup
*increases* it by 132.  The latter commit is a known architectural improvement,
so aggregate raw count is not a valid quality objective.  New declaration-owned
code can trigger more broad syntactic detectors even while obsolete authorities
are deleted.

Deleted-Surface Enrichment
--------------------------

For each cleanup, NRA was run on the exact parent with full package context but
reporting restricted to production Python files that the cleanup deleted.

.. list-table:: Findings on deliberately deleted production files
   :header-rows: 1

   * - Cleanup
     - Deleted files / lines
     - Share of package LOC
     - Findings touching deleted files
     - Share of package findings
     - Enrichment over LOC
   * - ``ccfef5f6d`` compatibility/owner cutover
     - 38 / 25,029
     - 6.27%
     - 672
     - 8.89%
     - 1.42x
   * - ``5e8812ee8`` compiler/runtime lattice deletion
     - 68 / 16,120
     - 4.18%
     - 384
     - 5.40%
     - 1.29x

NRA is directionally enriched on the known-bad surfaces, but 1.29--1.42x is
far too weak for a high-confidence work queue.  Detector-specific enrichment
is much more useful than the aggregate.

First Precision Correction
--------------------------

The first calibration pass made two changes supported by source inventory and
the cleanup labels:

- ``carrier_composition_retreat`` was removed, including its compact and AST
  collection substrate.  A carrier-valued field proves a has-a relationship,
  not the is-a relationship required to recommend inheritance.
- ``direct_reflective_builtin_call`` now excludes only
  ``object.__setattr__(self, "static_field", value)``.  It continues to report
  dynamic field names and writes to foreign receivers.  Of the 297 current
  OpenHCS ``object.__setattr__`` sites, 285 have that static-self form; 289 are
  in dataclass contexts, 277 are in ``__post_init__``, and only seven target a
  foreign receiver.

The exact snapshots were then replayed with the remaining 251 detectors:

.. list-table:: Snapshot replay after the first precision correction
   :header-rows: 1

   * - Snapshot
     - Before
     - After
     - Change
   * - runtime-owner parent ``4bc91c242``
     - 7,558
     - 7,287
     - -271
   * - runtime-owner cutover ``ccfef5f6d``
     - 7,142
     - 6,888
     - -254
   * - compiler-cleanup parent ``1398c8662``
     - 7,115
     - 6,868
     - -247
   * - compiler cleanup ``5e8812ee8``
     - 7,247
     - 7,021
     - -226
   * - current OpenHCS package
     - 7,691
     - 7,443
     - -248

The cleanup deltas remain directionally inconsistent: the runtime cutover is
``-399`` after calibration, while the known-good compiler cleanup is ``+153``.
This confirms that raw count cannot become the optimization target merely by
removing two noisy rules.

These are production CLI ``exact_compact_global`` results.  A legacy AST API
replay produced 7,487 findings on the current package, 44 more than the exact
compact path.  That result was rejected rather than mixed into this table.  AST
and compact projection parity is a separate correctness gate; calibration must
name one analysis authority and keep it fixed across snapshots.

Using the same primary-location attribution as the initial deleted-surface
experiment, the runtime cutover's deleted files retain 622 findings and the
compiler cleanup's deleted files retain 351.  Their aggregate enrichment falls
to 1.35x and 1.21x respectively because the removed noise was itself
concentrated on dataclass-heavy compatibility code.  This is a precision win,
not an enrichment win: the compiler deletion loses all eight unsound
composition findings and 25 of 29 reflection findings while preserving the
``available_carrier_reuse``, ``isinstance_family_scatter``, private-authority,
and wrapper-lattice signals.

Initial Detector Labels
-----------------------

These labels are provisional until more June corrective chains and matched
retained controls are scored.

High-signal candidates
~~~~~~~~~~~~~~~~~~~~~~

- ``abc_polymorphism_bypassed_by_concrete_dispatch`` is strongly concentrated
  in the first owner cleanup.  Its examples identify concrete recovery over a
  shared runtime-value or label-domain base.
- ``role_guarded_surface_access`` identifies callers that inspect a concrete
  role and then pull role-owned fields.  PR #60 deletes several such paths.
- ``opaque_object_annotation`` is heavily concentrated in the first cleanup
  and moderately concentrated in the second.  It corresponds to a real
  type-safety theme, but serializer and plugin boundaries still need exemptions.
- ``distributed_boundary_fanout`` is enriched in both cleanups and matches the
  history's repeated replacement of threaded primitive identity with typed
  carriers.  It needs boundary/projection awareness before being high precision.
- ``available_carrier_reuse``, ``private_helper_shadow``, and
  ``trivial_forwarding_wrapper`` have useful examples in the compiler lattice,
  especially when two independent signals point to the same owner.

Confirmed or likely noise
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``direct_reflective_builtin_call`` treats ``object.__setattr__`` as generic
  reflection.  Current OpenHCS has 297 such source sites, overwhelmingly used
  to validate, normalize, or cache fields on frozen nominal dataclasses.  This
  is a type-safe immutable-carrier implementation pattern, not duck typing.
- ``carrier_composition_retreat`` assumes that a request/result containing a
  typed carrier should inherit that carrier.  Examples such as
  ``ArtifactProducer.spec: ArtifactSpec`` and
  ``AgentCapabilitySearchResult.query: AgentCapabilitySearchRequest`` are
  ordinary has-a relationships.  Recommending inheritance changes semantics
  and is unsound without an independently proved is-a relation.
- ``unclassified_runtime_fallback`` counts ordinary keyword defaults and
  conditional numerical defaults.  Its deleted-surface concentration is below
  baseline in the first cleanup, and current examples include routine CLI and
  dataclass defaults.
- ``semantic_dict_bag`` flags display glyph tables, AST icon tables, process
  environment mappings, and launcher template substitutions.  A dictionary is
  not a nominal record merely because its literal keys are stable.
- ``typing_protocol_contract`` assumes every ``Protocol`` should be an ABC.
  Structural protocols can be the correct typed dependency boundary; nominal
  replacement requires identity, lifecycle, or shared implementation evidence.
- ``semantic_mirror_without_descent`` contains real mirrors, but it is
  under-represented in both PR #60 deleted surfaces and also produces unrelated
  cross-family matches.  For example, ``AVAILABLE_MEMORY_TYPES`` correctly
  matches ``MemoryType`` but also matches an unrelated runtime-testing base.
- broad helper, branch-count, role-token, and identical-small-method detectors
  often describe size or lexical similarity without proving a shared semantic
  authority.  They should be supporting evidence, not independent work items.

Current Noise Shape
-------------------

After the first precision correction, the current package's six largest
detector families account for 4,206 of 7,443 raw findings (56.5%):

- opaque object annotations: 1,742
- semantic mirrors: 872
- distributed boundary fanout: 502
- non-nominal private helpers: 407
- unclassified runtime fallbacks: 371
- role-surface drift: 312

Direct reflective builtin findings fall from 299 to 184, and the 133 current
composition-retreat findings disappear with the invalid detector authority.

This explains why the 11,263-finding whole-repository result is not a usable
review list.  Raw findings are an evidence substrate.  A useful work queue must
require stronger correlations, exclude disproved premises, and rank against
historically validated authority moves.

Next Calibration Gates
----------------------

Before declaring the study complete:

1. Score more June/July corrective commits, including declaration-owned MCP,
   source provenance, enum/config ownership, and extracted-owner changes.
2. Add matched retained controls so deletion enrichment is not the only label.
3. Measure whether the revised work queue ranks the historically deleted
   surfaces ahead of retained production code; do not optimize raw count.
4. Promote only detector combinations whose survivor set is small enough for a
   maintainer to inspect and whose remediation agrees with the surviving
   architecture.
