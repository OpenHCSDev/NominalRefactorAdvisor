Semantic Derivation Protocol
============================

A repeatable method for taking an undisciplined codebase to a nominally correct,
minimally factored correct-maintenance architecture.

This protocol is intentionally general.  NRA can help collect evidence, and the
OpenHCS changes below are worked examples, but neither Python nor NRA defines
the method.  The method applies when the target language and architecture can
represent every semantic distinction required by the maintenance problem.

The organising claim is:

  Every manually maintained enumeration of members of a set that already
  determines them is a required answer being maintained outside the mechanism
  that derives it.

Everything below either discovers that condition, proves it, repairs it, or
records enough evidence to audit the repair.


What ``Semantically Complete`` Means Here
-----------------------------------------

Semantic completeness is relative to a domain and its required questions.  Let:

- ``D`` be the domain meanings and reachable states;
- ``Q`` be the questions the system must answer correctly;
- ``A(q, d)`` be the required answer for question ``q`` in state ``d``; and
- ``R(d)`` be the representation available to the implementation.

The chosen mechanisms are semantically complete for the task when their native
derivation preserves every distinction needed to supply ``A(q, d)``.  A wider
system may still be correct by completing missing answers in code or tooling
while the restricted mechanism remains incomplete.  If two domain states
collapse to the same representation while some required question gives them
different answers, no helper operating only on that representation can recover
the answer.  The architecture must retain a distinguishing name, enrich the
representation, or make one of the states unreachable by an enforced
precondition.

Correct maintenance is also stronger than one correct snapshot: the program
must continue to satisfy its declared specification on every reachable state as
requirements change.  This protocol additionally minimizes avoidable external
semantic obligations; it does not redefine correctness as mere derivation.

This is not Turing completeness.  A language can compute arbitrary functions
and still encourage an architecture that erases the identity, provenance,
phase, ownership, or variation needed for correct maintenance.  Relevant
language mechanisms can include nominal types, algebraic data types, traits,
classes, enums, modules, schemas, type classes, metaclasses, code generation,
or proof-carrying declarations.  Inheritance is one possible derivation
mechanism, not the definition of semantic completeness.

The protocol therefore has two distinct parts:

1. **domain extraction** makes supplied domain rules and task-authorized
   specifications/declarations explicit, traces their source representations,
   and proposes unresolved meanings, required answers, and independent roles;
2. **semantic implementation** encodes those facts once and makes all other
   surfaces derivations.

A factoring over anonymous shapes is not correct merely because it is small.
Domain-driven design comes first; minimisation follows from the recovered
domain relation.


Inputs And Fixed Boundary
-------------------------

Before beginning, freeze a reproducible investigation boundary:

- repository revision and dependency revisions;
- included packages or bounded contexts;
- generated, vendored, test, migration, and compatibility policy;
- language/runtime version;
- analyzer version and solver settings; and
- known dynamic loading or mutation boundaries.

The protocol expects four inputs:

1. intended domain rules and authoritative declarations supplied by the task
   owner, including existing specifications or tests the task authorizes as
   normative;
2. the code and its current observable behavior;
3. worked architectural changes whose before and after states can be inspected;
4. an exact analyzer for the required relation, including ancestry-gap and
   rectangle-cover certificates when those are the chosen factoring model.

The agent performs the source investigation and proposes concrete ownership
moves; it may derive all pairs entailed by already authorized rules without
asking the human to enumerate them.  Genuinely missing or contested domain
meaning remains OPEN until adjudicated.  Source, historical outcomes and an
LLM's confidence do not independently supply normative business rules.  NRA
checks supported observations, provenance, uncertainty and transformations
against the admitted inputs; it does not certify the truth of those rules.

Do not silently broaden or narrow the boundary during the run.  A newly found
external source, dynamic key, or runtime mutation is an exclusion until it is
explicitly admitted and the analysis is rerun.


Phase 0 — Mine Transformations, Not Narratives
----------------------------------------------

Study the worked changes before touching the new code.  Inspect their actual
before and after trees, not only their descriptions.  For each transformation,
produce rows with this schema:

.. list-table:: Example-mining record
   :header-rows: 1

   * - Before surface form
     - Set or relation that already determined it
     - Derivation introduced
     - Failure exposed without it
   * - literal field-name list
     - annotated fields on the declaration
     - field introspection at the owner
     - list drift omitted a valid field
   * - repeated type-tag branches
     - closed nominal implementation family
     - polymorphic hook or type-keyed strategy
     - a new case required edits at scattered consumers

Produce rows, not PR summaries.  The output is a catalog of surface forms that
can seed a mechanical search in a new repository.

Expected surface forms include:

- literal names equal to class fields, enum members, registry keys, or exports;
- ``if``/``elif`` dispatch over a closed tag;
- dict literals mapping a known key family to handlers;
- index-aligned names, labels, defaults, validators, or serializers;
- repeated ``isinstance``/``hasattr`` branches;
- keys authored independently by producers and consumers;
- hand-written exports, form fields, serialization fields, or migration lists;
- per-member validation copied across a declared family;
- counts and lengths of declared sets; and
- comments or documentation that enumerate executable cases.

These syntaxes are not automatically defects.  They become one defect when the
same required answer is already determined elsewhere and both surfaces must
agree.


Empirical Pattern From Epoch-Defining Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary OpenHCS PR-history snapshot inspected for this protocol contained
94 PRs, of which 71 were merged.  The important evidence was not raw churn or
old case-study prose, but the recurring transformation visible in the merged
trees:

- PR #38 replaced a hand-maintained conversion grid with derivation over a
  closed memory-type axis.
- PR #44 introduced nominal get/set capability contracts and services, but
  retained concrete Qt dispatch for other operations; its reset fallback also
  changed.  Service extraction alone is not dispatch elimination or parity.
- PR #58 moved model state, snapshots, provenance, and dirty tracking out of a
  widget manager into ``ObjectState``; UI became a projection.
- PR #60 integrated the typed declaration/compiler/runtime cutover: authored
  declarations lower once into typed plans consumed mechanically by runtime,
  UI, MCP, serialization, and parity evidence.
- PR #69 replaced string-selected materialization handlers with typed writer
  and option ownership.
- PRs #87 and #92 deleted configuration mirrors after nominal declarations
  became authoritative across consumers.
- PRs #111, #112, and #115 were different review lenses over one integrated
  convergence event, not three independent rewrites.  Together they tightened
  value evidence, source geometry, and artifact/callable/materialization
  semantics.

Across these different surfaces, the latent algorithm is stable:

1. recover a domain distinction hidden behind incidental structure;
2. give the distinction a nominal owner;
3. recover the complete consumer relation;
4. compile or derive consumer-specific views from the owner;
5. move stable derivation to the owner lifetime;
6. delete parallel authorities and fallback recovery paths; and
7. retain executable gates that prevent the old surface from returning.

Large additions can be legitimate infrastructure investment and large
subtractions can be mere relocation.  The architectural measure is whether a
required answer gained one owner and its other appearances became derivations.


Phase 1 — Name Before Factoring
-------------------------------

This is the domain-driven extraction phase.  Start from the supplied rules and
identify which existing specifications/declarations have task-authorized domain
authority.  Trace code, tests and history to recover their implementation and
expose contradictions or missing decisions; do not infer authority merely from
which implementation survives a historical PR.

Structure cannot distinguish meanings that happen to look alike.  Before
proposing helpers, bases, registries, schemas, or services:

1. identify the bounded context;
2. inventory the domain meanings used by that context;
3. assign one stable name to each meaning;
4. identify lifecycle or observation phases in which the same value has a
   different meaning;
5. record provenance distinctions that consumers must recover; and
6. create a collision record for meanings that can share the same structural
   representation.

Use this rule:

  Erase a distinction only when every merged case requires the same answer for
  every in-scope question.

Otherwise retain a name, migrate references to a richer carrier, or enforce a
precondition that makes the conflicting history unreachable.  Returning a
different answer is a contract change, not a refactor.

The phase produces:

- a domain glossary with stable identifiers;
- a bounded-context map;
- a list of required questions;
- a structural-collision table; and
- a declaration of independently changeable roles.

Do not infer independent variation from current equality.  Two roles can return
the same answer in every existing test while retaining independent future
histories.  Conversely, ``we may override it later`` is not evidence of
independence until the divergent meaning or future is named.


Phase 2 — Recover The Required Relation, Nothing Else
-----------------------------------------------------

Now inspect every access site and record the relation the code currently
implements.

The canonical table has exactly two semantic columns::

   implementation/key, consumer

Use stable identifiers from Phase 1.  Emit one row per required pair.  Keep
source locations, confidence, and collection details as evidence metadata; do
not add proposed classes or groupings to the relation itself.

For each implementation or key, separately record whether its answer history
can change independently.  This declaration cannot be recovered from the
pair table after meanings have been merged, so it is a mandatory input to
minimisation.

Reconcile the observed table with the Phase 1 contract before certifying it as
the required relation.  Source and runtime extraction recover observable pairs;
they do not infer undocumented intent.  An intended-but-absent pair needs a
cited task-authorized contract, normative test or domain decision; once that
requirement and its absence are established, record the current semantic bug.
Unknown or external cases remain in the exclusion log rather than becoming
zero-valued absences; until they are resolved, relation completeness is unknown.
Current behavior is evidence, not automatically the correctness oracle.  Preserve
provenance for every included pair, exclusion, and added requirement.

Rules:

- record what the code does, not the architecture you want;
- trace aliases and generated surfaces back to their effective access sites;
- distinguish reads, writes, dispatch, validation, serialization, and display
  if they ask different questions;
- preserve phase and provenance when consumers require them; and
- do not silently discard uncertain rows.

Every unresolved source goes into an exclusion log with:

- stable exclusion identifier;
- source location or boundary;
- reason it cannot be resolved;
- requirements it may contain;
- whether it can add or remove pairs; and
- the condition for admitting it later.

A dropped requirement creates an unsigned error: it can make the computed gap
either too large or too small.  Completeness of the relation is therefore a
proof obligation, not a data-cleaning preference.


Phase 3 — Audit Every Enumeration
---------------------------------

For every enumeration found by the Phase 0 catalog, apply one three-way test
against the admitted question and authority.  If those are not established,
retain OPEN rather than forcing the site into a branch.

Branch 1 — Genuine authority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No existing set or declaration determines the enumeration.

Name it, give it a nominal owner, and stop treating it as duplication.  Its
consumers must derive from this new authority.

Branch 2 — Replica
~~~~~~~~~~~~~~~~~~

An existing authority determines the enumeration and the two must always
agree.

Replace the enumeration with a derivation.  Deleting it removes no domain
requirement; it removes an independent drift site.

Branch 3 — Named divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An existing set currently determines the values, but the two roles may
legitimately diverge.

Name the divergent required answer or independently changeable future.  Give
that role its own identity and include it in the Phase 1 model.  Classify the
site as Branch 2 only when the admitted question and authority require equality.
If the role or intended future is unknown, retain OPEN; inability to name a
divergence does not establish an equality obligation.

Record one decision per site, or OPEN with the missing rule/evidence during
investigation.  Completion still requires those in-scope cases to be resolved.
The example below assumes an **admitted public API rule** that ``__all__`` must
contain exactly this declaration family's public names.  Without that rule,
inspect whether exports are an independent API policy; do not classify them as
a replica merely because current names match.

.. list-table:: Enumeration decision (under the stated API rule)
   :header-rows: 1

   * - Site
     - Determining authority
     - Branch
     - Named divergence or derivation
   * - ``module.__all__``
     - public declaration family
     - 2
     - derive exports from declarations

Branch 3 always requires a receipt.  Speculative flexibility without a named
future is not an architectural requirement.


Phase 4 — Compute The Factoring
-------------------------------

Feed the Phase 2 relation and the independent-role declarations to the exact
analyzer.  Write the required relation as :math:`R \subseteq I \times C`, where
:math:`(i, c)` means that consumer :math:`c` requires the answer owned by
implementation :math:`i`.  In the declared finite, deletion-only single-parent
model, an admissible retained relation :math:`L \subseteq R` has laminar
implementation scopes: every pair of scopes is nested or disjoint.  Under the
admitted construction, latent provider nodes and consumer leaves are permitted;
a scope-containment forest over the nonempty retained implementation scopes
witnesses realizability.  These forest edges are not necessarily direct consumer-
inheritance edges in the target program.  Require the analyzer to return:

- a maximum-cardinality admissible retained relation :math:`L^*`;
- the exact residual relation :math:`R \setminus L^*` and ancestry gap
  :math:`|R \setminus L^*|`;
- an arrangement attaining that optimum;
- a checkable certificate; and
- the tied-optimum count and a declared deterministic selection policy.

Then compute an exact cover of the residual relation.  A provider rectangle is a
nonempty complete subrelation :math:`A \times B \subseteq R \setminus L^*`:
every consumer in :math:`B` requires every implementation in :math:`A`, so the
rectangle contains no forbidden pair.  A provider cover has union exactly equal
to the residual.  It is a cover, not a partition; rectangles may overlap.
Interpret each selected rectangle as one latent provider role.

With the relation, independence declarations, laminar repair model, and rectangle
provider model fixed, an exact solver that completes within its declared bounds
can certify the minimum cover cardinality for that admitted representation.  It
is not a universal implementation-cost claim: ordering, state, precedence, and
other carrier constraints require separate certificates.  A complete certificate
must prove the selected union equals the residual.  Exceeding a declared exact
search limit yields ``unknown`` and no certificate, never a heuristic or partial
optimum.

The result is a blueprint derived from the required relation.  It is not a
class diagram proposed by taste.

Guardrails:

- split independently changeable roles before minimisation; splitting them
  cannot legitimately be avoided by exploiting current coincidental equality;
- merge roles only when their full answer histories agree, not merely sampled
  outputs;
- preserve domain names across structurally equal rows;
- do not force every provider to become a class or parent; select the target
  language mechanism after the provider relation is known;
- report all equal minima unless a stable policy chooses among them; and
- treat analyzer infeasibility or an unchecked certificate as ``unknown``, not
  permission for a heuristic architecture.

Relative determinism comes from fixing the relation, independence declarations,
admissible mechanism family, solver, and tie-break.  The protocol does not claim
that incomplete domain evidence can produce a unique architecture.


Phase 5 — Derive Choke Points And Invalidation Surface
------------------------------------------------------

For each proposed owner, record its retained or derived consumers and compute
its exact residual required pairs outside ancestry or derivation closure.

- A carrier with nonempty required or derived consumer coverage and zero residual
  gap is a true choke point.  A cached or derived value placed there reaches every
  required consumer through declared structure.
- A zero-degree carrier is vacuous and is not an owner or choke point.
- An owner with residual gap is not a universal choke point.  Every residual
  pair is an explicit invalidation or adapter site.

This turns invalidation from an unbounded search into an enumerable surface.
Record:

- owner identifier;
- derived consumers;
- residual pairs;
- invalidation mechanism;
- lifecycle at which derivation is stable; and
- provenance needed to audit the result.

Move computation to the longest correct lifetime, not simply the earliest
possible point.  Compile-time facts belong in compiled plans; declaration facts
belong on declarations; request-local facts must not be cached globally.


Phase 6 — Implement The Semantic Model
--------------------------------------

Implement in dependency order:

1. introduce stable names and nominal identities;
2. introduce authoritative declarations for the provider rectangles;
3. enforce invariants at construction, declaration, or compilation boundaries;
4. derive typed plans, registries, schemas, adapters, or views from those
   declarations;
5. migrate consumers to the derived projections;
6. confine compatibility translation to explicit boundaries;
7. delete replica enumerations, structural recovery, and fallback paths; and
8. add deletion gates that make the retired architecture unrepresentable or
   mechanically detectable.

Prefer compile-before-execute when declarations have consequences across many
consumers.  Runtime should consume a proved plan rather than rediscover names,
source identity, configuration, ownership, or output policy.

Preserve provenance with the value whenever correctness or auditability asks
``which owner supplied this answer?``.  Equal values do not prove equal origins.

Fail loudly when a required relation is absent, ambiguous, stale, or
conflicting.  A fallback that fabricates an answer converts a visible proof gap
into silent semantic drift.

A safe migration may temporarily contain old and new surfaces, but only one may
be authoritative.  Mark the other as an adapter derived from the owner, attach
a deletion condition, and do not permit bidirectional synchronization.


Phase 7 — Validate In The Right Order
-------------------------------------

Validation has two independent jobs.

Structural validation
~~~~~~~~~~~~~~~~~~~~~

First prove the maintenance architecture:

- every required pair is preserved or explicitly changed;
- every enumeration decision has an authority, derivation, or named divergence;
- every provider traces to a rectangle in the certified cover;
- every residual pair appears in the invalidation list;
- every derived surface traces to one nominal owner;
- no retired mirror remains writable;
- independent roles remain separately identifiable; and
- provenance survives every boundary where it is a required answer.

Behavioral validation
~~~~~~~~~~~~~~~~~~~~~

Only then use unit, integration, parity, benchmark, and acceptance suites to
show that the derived architecture did not unintentionally change behavior.

Behavior is not the factoring oracle.  Two implementations can pass identical
runs while one derives the relation and the other maintains it manually in many
places.  A suite exercises points; independently variable roles define a space
of possible histories.  Passing samples alone cannot establish that two meanings
have one future; finite-suite completeness needs a separate coverage proof.

Add negative architecture tests as well as positive behavior tests:

- adding a declaration automatically updates every intended projection;
- a conflicting duplicate fails at the owner boundary;
- a missing required relation cannot fall back silently;
- a retired manual roster or dispatch form is rejected; and
- stale generated or cached views invalidate by declared identity.


Deterministic Run Artifacts
---------------------------

Store one run directory per bounded context.  A fresh agent should be able to
replay the reasoning from these artifacts alone:

.. code-block:: text

   derivation-run/
     boundary.md
     domain-rules.md
     example-transformations.csv
     domain-glossary.md
     required-questions.csv
     structural-collisions.csv
     independent-roles.csv
     required-relation.csv
     relation-binding.json
     exclusions.csv
     enumeration-decisions.csv
     analyzer-input.json
     gap-certificate.json
     provider-cover.csv
     residual-gap.csv
     migration-map.csv
     validation.md

``domain-rules.md`` records supplied rules, task-authorized specifications and
admission decisions, including the rule/decision licensing each required or
forbidden pair and the disposition of unknown cases.  These are provenance
metadata, not a third semantic relation column.  The agent derives entailed
pairs and records the rationale; the task owner need not enumerate each edge.
This is a workflow decision record, not a newly implemented NRA wire schema or
a claim that the current source-binding checker validates business meaning.

``relation-binding.json`` is a deterministic machine-readable receipt, not a
second prose authority.  It binds each exact two-column relation pair to one or
more evidence-record identifiers and records the boundary identity, source
revision, covered case identifiers, exclusions, and coverage verdict.  Pair,
evidence-record, and case identifiers must be stable and unique within the run.
A positive verdict requires every pair to reference evidence, every admitted
case to be bound, and no unresolved exclusion that could add or remove a pair.
The receipt points to the required relation and evidence; it does not restate or
invent either one.  Its source-binding verdict does not itself admit domain
requirements: review it alongside ``domain-rules.md``, including
intended-but-absent requirements and explicit exclusions.

Use stable identifiers and deterministic ordering.  Record tool versions,
command lines, source revisions, and hashes of analyzer inputs.  Never overwrite
a failed or superseded run; link it to the succeeding run so changes in domain
evidence remain visible.


Iteration Strategy
------------------

Run the protocol on one bounded context at a time:

1. choose a region with a coherent vocabulary and consumer relation;
2. complete all receipts for that region;
3. implement owners before projections;
4. rerun extraction after deletion because new authorities may expose the next
   relation; and
5. join adjacent contexts only when their boundary relation is explicit.

Prioritize by proof leverage rather than raw finding count:

- authorities whose derivation removes many drift sites;
- boundaries where lost identity creates wrong answers;
- compile/runtime disagreements;
- view or transport layers that currently own domain facts; and
- regions whose residual gap blocks several later collapses.

Do not optimize line count, class count, detector count, or immediate diff size.
An epoch-defining refactor may first add a semantic owner and only later delete
the old lattice.  The invariant is decreasing independent semantic authority.


What Does Not Work As An Oracle
-------------------------------

Do not derive the architecture from:

- behavior or benchmark parity alone;
- lexical similarity or clone counts;
- line, class, or finding counts;
- a framework preference such as inheritance versus composition;
- names inferred only from current code shape;
- generated documentation or historical PR prose without source verification;
- an LLM-proposed class diagram; or
- the largest immediate compression step.

These can locate evidence or validate consequences.  They cannot decide the
required relation or whether two roles have one semantic future.


Completion Receipts
-------------------

A region is complete only when all of these exist:

- the fixed investigation boundary and supplied domain-rule/authority decisions;
- the example-derived surface catalog;
- the domain vocabulary and collision table;
- the Phase 2 relation, pair-level provenance, and exclusion log;
- a binding and coverage verdict showing why that relation still describes the
  declared boundary at the recorded revision;
- an independence declaration for every merge candidate;
- one three-way decision per enumeration site;
- the analyzer's minimum gap and checkable certificate;
- the provider cover, with every provider traceable to a rectangle;
- the residual-gap/invalidation list;
- a named divergence for every Branch 3 decision;
- an owner-to-projection migration map;
- deletion gates for retired authorities; and
- structural and behavioral validation results.

Any later change faces one question:

  Does it preserve the required relation and every independently changeable
  role, or does it change them?

If it changes them, the change must name the required answer or independent
future it removes.  That review does not route through sampled behavior, which
is why the architecture remains maintainable after the original refactorers
leave.
