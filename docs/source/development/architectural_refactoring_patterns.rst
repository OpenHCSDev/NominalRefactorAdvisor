Architectural Refactoring Patterns
==================================

Reusable high-level patterns the advisor is designed to recommend.

These are not the generated shipped pattern records from
:doc:`../api/pattern_catalog`. They are the broader architectural moves that
show up repeatedly across Python codebases.

Pattern 1: Nominal Family Over Structural Probing
-------------------------------------------------

Use an ``ABC`` plus concrete subclasses when behavior depends on semantic role.

Typical trigger:

- repeated ``hasattr`` or ``isinstance`` recovery
- strings or enums selecting classes indirectly
- parallel method families across sibling implementations

Preferred collapse:

- one nominal root
- concrete subclasses for semantic cases
- shared orchestration in the base class
- orthogonal residue in mixins when ordering matters

Pattern 2: Authoritative Record Plus Derived Views
--------------------------------------------------

Use one frozen record type or one authoritative row family when several surfaces
repeat the same facts.

Typical trigger:

- multiple dicts keyed by the same axis
- helper functions rebuilding the same keyword bags
- repeated projection helpers
- export lists or indexes maintained by hand

Preferred collapse:

- one authoritative dataclass row or tuple of rows
- derived indexes, exports, summaries, and builders

Pattern 3: Class-Time Registration Instead Of Manual Rosters
------------------------------------------------------------

Use class-time registration when the code needs enumeration of a behavioral
family.

Typical trigger:

- handwritten subclass tuples
- registry mutation in several files
- repeated discovery helpers or union builders

Preferred collapse:

- one registry-backed root
- declarative key extraction
- derived lookup and traversal surfaces

Pattern 4: Staged Orchestration Instead Of Hub Functions
--------------------------------------------------------

Split oversized orchestrators into explicit phases when one function owns too
many transitions at once.

Typical trigger:

- long branch-heavy execution functions
- repeated result assembly pipelines
- mixed discovery, validation, execution, and formatting in one body

Preferred collapse:

- one orchestration surface
- named stage boundaries
- helper layers only where they remove true duplication

Pattern 5: Derived Surface, Not Shadow Authority
------------------------------------------------

Whenever a second surface exists only so another subsystem can consume it, make
that surface derived.

Typical trigger:

- manual ``__all__`` lists beside a canonical family
- registry views maintained in parallel with the registry
- wrapper specs that restate metadata already declared elsewhere

Preferred collapse:

- one authority
- one materializer for every secondary surface

Pattern 6: Fail-Loud Contracts
------------------------------

Do not preserve invalid states with defensive fallbacks when the architecture
already guarantees the contract.

Typical trigger:

- ``getattr(..., default)`` on guaranteed fields
- ``hasattr`` around abstract methods
- defaulting away missing provenance or role identity

Preferred collapse:

- direct access
- explicit validation at the real boundary
- immediate failure for broken contracts
