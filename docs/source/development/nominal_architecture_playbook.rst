Nominal Architecture Playbook
=============================

Canonical guide for agents and maintainers who need to reason about nominal
architecture before touching a structurally significant part of the codebase.

This document consolidates the main architectural lessons that follow from the two paper models and from
the existing development guides. It is meant to answer one question clearly:

What should an agent believe about identity, typing, provenance, duplication, and refactoring before
editing code?


Who Should Read This
--------------------

Read this first if you are about to:

- refactor a subsystem
- introduce or remove an ``ABC``
- replace a dispatch mechanism
- introduce metaclass, decorator, registry, or type-generation logic
- simplify or remove duplicated machinery
- decide whether duck typing is acceptable in a region

If you only read three docs before editing architecture, read them in this order:

1. ``nominal_architecture_playbook.rst``
2. ``agent_refactoring_crash_course.rst``
3. ``nominal_identity_case_studies.rst``

Then read ``nominal_refactor_advisor.rst`` for the repository-specific direction of travel and the
current self-audit.


Executive Summary
-----------------

The central architectural claim is:

- structural similarity is only a partial view
- partial views collapse semantically distinct cases into the same fiber
- if the system later needs to distinguish those cases exactly, it must carry an explicit nominal handle
- if the same fact is writable in multiple places, the system is above the coherence boundary
- if provenance is not observable, exact correctness claims are not verifiable

In Python terms:

- use ``ABC`` and subclasses for semantic role families
- use enums for closed, lightweight variant axes
- use nominal sentinel types or objects when exact marker identity is required
- use metaclasses when the architecture is fundamentally class-level
- use authoritative dataclasses for fact ownership
- derive exports, views, caches, and summaries from the authoritative source

Do not treat ``Protocol``, ``hasattr``, sentinel attributes, or repeated field probing as substitutes for
semantic identity when the code needs enumeration, provenance, ordering, registration, or exact dispatch.


The Core Translation from the Papers
------------------------------------

Representation
~~~~~~~~~~~~~~

A representation is whatever information the current code is using to make a decision.

Examples:

- a method set
- a bag of attributes
- a string mode
- a sentinel field
- a dict shape
- a subset of fields from a larger object

Every representation hides some information and exposes some information.


Fiber
~~~~~

A fiber is the set of semantically distinct objects or states that collapse to the same representation.

Examples:

- two classes that expose the same methods but mean different things
- two config objects with the same shape but different provenance
- a base type and its lazy companion when viewed only through shared fields

If code later needs to recover distinctions inside the fiber, the original contract was too weak.


Confusability
~~~~~~~~~~~~~

Two cases are confusable if the current representation does not separate them but the architecture needs
to treat them differently.

Code smell translation:

- if downstream code needs repeated probing to discover what object it actually has, the upstream type
  boundary is confusable
- if multiple semantic cases hit the same fallback branch, the branch is collapsing a real distinction
- if a dispatch layer depends on a sequence of ``hasattr`` checks, the contract is carrying too little
  information


Nominal Handle
~~~~~~~~~~~~~~

A nominal handle is the explicit identity carrier that separates cases the structural view cannot.

Python examples:

- a concrete subclass
- an ``ABC`` membership relation
- an enum member
- a runtime-generated type
- a unique sentinel object created for identity, not for structure


Unit Rate / Single Source of Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coherence exists only when one source is authoritative and every other representation is derived.

Software translation:

- one authoritative owner for each fact family
- all exports and secondary forms derived from it
- no manually synchronized duplicates that can drift independently


Provenance Observability
~~~~~~~~~~~~~~~~~~~~~~~~

Exact claims are only verifiable when the code can say where a value came from.

Software translation:

- a result should be able to carry not just ``value`` but also ``who supplied it``
- constructors, class hierarchies, and registries should make ownership obvious
- hidden fallback chains reduce verifiability


Orthogonal Core
~~~~~~~~~~~~~~~

If one piece of machinery is derivable from another, it is redundant and should not be independently
maintained.

Software translation:

- repeated setup across subclasses belongs in the base class
- repeated field mapping belongs in one constructor or factory
- repeated export schemas belong in one authoritative row model
- repeated wrappers that simply transport the same facts should be collapsed


The Main Architectural Rule
---------------------------

Always ask these questions in order:

1. What semantic distinction does this region need to preserve?
2. What representation is it currently using?
3. Which distinct cases collapse into the same fiber under that representation?
4. What capability does the code need that the partial view cannot recover?
5. What nominal handle or authoritative source should carry that distinction instead?

If you cannot answer those five questions, do not refactor yet.


Capability Matrix
-----------------

This is the shortest practical decision table.

.. list-table:: Required capability vs required architectural tool
   :header-rows: 1

   * - Capability needed
     - Structural / duck-typed view sufficient?
     - Required tool
     - Why
   * - Local behavioral call on one object
     - Sometimes
     - Direct method call or small helper
     - No global identity reasoning is required
   * - Exhaustive enumeration of all variants
     - No
     - ``ABC`` subclass family or explicit nominal registry
     - The system needs a closed, discoverable family
   * - Import-time enforcement of interface obligations
     - No
     - ``ABC`` / abstract methods / metaclass checks
     - Structure alone does not fail loudly when declarations are wrong
   * - Ordered conflict resolution across variants
     - No
     - Inheritance + MRO or explicit precedence model
     - Ordering must be attached to declared identity
   * - Provenance: ``which type supplied this value?``
     - No
     - Nominal type identity in return values or registries
     - Structural equality erases origin
   * - O(1) dispatch by family or backend
     - Usually no
     - Enum key, type key, or nominal registry
     - Structural probing becomes linear search or parallel indexing
   * - Auto-registration at class definition time
     - No
     - Metaclass or decorator + class identity
     - The architecture is operating on classes, not just instances
   * - Dynamic interface membership
     - No
     - ABC or generated nominal interface
     - Explicit role claims require nominal identity
   * - Shared behavior for all current and future instances of a family
     - No
     - Type namespace / class-level injection
     - The type itself is the shared mutable namespace
   * - Bidirectional exact lookup between companion types
     - No
     - Type-keyed registry with bijection checks
     - Structural similarity is not a stable key
   * - One authoritative fact source with many derived views
     - No
     - Authoritative dataclass + derived constructors
     - Multiple writable copies violate coherence


Decision Procedure
------------------

Step 1: Identify the Semantic Role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

State explicitly what the object is, not just what methods it has.

Examples:

- a certified returned-pose plan
- a broad-contract ambiguity plan
- a lazy config companion
- a microscope handler class
- a streaming backend config

If you can only describe it as ``something with methods X and Y``, you are probably still at the
representation layer.


Step 2: Decide Whether Identity Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identity matters if the code needs any of the following:

- exhaustive enumeration
- provenance
- conflict ordering
- class-level registration
- distinction between structurally equal but semantically different classes
- stable keys for registries
- fail-loud validation at definition time

If identity matters, use nominal tools immediately.


Step 3: Decide Whether the Concern Is Instance-Level or Class-Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instance-level concerns:

- authoritative fact payloads
- variant-specific runtime behavior
- value-level transformations

Class-level concerns:

- registration at import time
- generated interfaces
- ``isinstance`` customization
- method injection into all members of a family
- MRO ordering and lineage

If the concern is class-level, duck typing is almost always the wrong lens.


Step 4: Decide Whether the Variant Axis Is Closed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an enum when:

- the family is closed
- the behavior is lightweight
- the main need is discrete selection, not inheritance semantics

Use an ``ABC`` when:

- variants share substantial behavior or invariants
- subclasses differ by real semantic refinements
- provenance and declared family membership matter

Use a metaclass when:

- registration or validation must happen at class definition time
- class objects themselves are the architectural subject

Use a nominal sentinel object or runtime-generated type when:

- you need a unique marker identity independent of structure
- the marker must remain robust under refactoring
- strings or fields would be too weak or too coupled to implementation details


Step 5: Decide Where the Fact Lives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every fact family, identify:

- authoritative source
- derived views
- export forms
- caches
- summaries
- UI projections

If more than one of these is independently writable, the subsystem is above the coherence boundary.


Step 6: Remove Redundant Machinery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Move derivable machinery upward:

- shared validation into the ``ABC``
- shared field mapping into classmethods or factories
- shared export logic into one authoritative record model
- shared control flow into template methods

Only irreducible differences should remain in subclasses.


When Duck Typing Is Actually Acceptable
---------------------------------------

Duck typing is acceptable only when all of the following are true:

- the code is performing a local behavioral operation
- there is no need to enumerate all matching implementations
- there is no need to report provenance
- there is no need for import-time enforcement
- there is no conflict ordering problem
- the system is not using the object as a registry key or identity marker
- the role question is not architectural, only operational

Good example:

- a tiny helper that accepts any object with ``write(text)`` and immediately writes text once

Bad examples:

- discovering all parameter-info variants
- determining which config family supplied a field
- deciding which handler class owns a microscope type
- maintaining lazy/base type bijections


What ``Protocol`` Means in This Codebase
----------------------------------------

Treat ``Protocol`` as a structural view only.

It does not provide:

- a discoverable variant family
- import-time enforcement of semantic role identity
- provenance
- MRO ordering
- metaclass integration
- class-level registration

Therefore:

- do not use ``Protocol`` as the primary boundary for semantic role families
- do not replace ABC-based architecture with ``Protocol`` to "reduce boilerplate"
- do not treat structural compatibility as proof of same-family semantics


Sentinel Attributes vs Sentinel Types
-------------------------------------

This distinction is critical.

Sentinel attribute
~~~~~~~~~~~~~~~~~~

Example:

.. code-block:: python

   class Handler:
       sigma = "imagexpress"

Problems:

- not enforced by the type system
- not naturally enumerable
- cannot explain provenance
- has no MRO of its own

This only simulates identity by convention.


Sentinel type or sentinel object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example:

.. code-block:: python

   _FRAMEWORK_CONFIG = type("_FrameworkConfigSentinel", (), {})()

Properties:

- unique nominal identity
- usable as a dictionary key
- decoupled from attribute names
- robust under refactoring

This is a real nominal handle.


ABC vs Enum vs Metaclass vs Sentinel Object vs Dataclass
--------------------------------------------------------

Use an ``ABC`` when:

- there is a semantic family of implementations
- shared behavior should live in the base class
- subclasses are refinements, not just tags
- MRO or class membership matters

Use mixins / multiple inheritance when:

- a concern is orthogonal but still belongs inside the nominal hierarchy
- MRO precedence is part of the semantics
- the concern should be reusable without being externalized into a composition wrapper
- you need class-level or method-resolution participation from more than one reusable concern

Use an enum when:

- the family is closed and small
- behavior differences can remain lightweight
- you need a stable key for dispatch but not a full class hierarchy

Use a metaclass when:

- the architecture must act while classes are being defined
- registration, validation, or interface generation is class-level

Use a sentinel object or generated nominal type when:

- you need exact marker identity independent of field structure
- there may be no methods or payload at all

Use a dataclass when:

- the main problem is authoritative fact ownership
- the object is a record of facts, not a behavior family
- exports and summaries should be derived from the same payload


How ABCs Must Be Used
---------------------

An ``ABC`` is justified only if it removes duplication and clarifies identity.

Put in the base class:

- shared invariants
- shared validation
- shared prelude / postlude logic
- shared record building
- shared export helpers
- shared classmethod constructors

Leave abstract only:

- irreducible variant-specific behavior
- irreducible variant-specific payload additions
- irreducible theorem/backend hooks

Bad ABC:

- only renames an interface
- subclasses still repeat the same setup and row-building code

Good ABC:

- centralizes shared machinery
- exposes a single template method or authoritative constructor
- leaves subclasses small and semantically meaningful


How Authoritative Dataclasses Must Be Used
------------------------------------------

Use authoritative dataclasses to solve fact drift.

Good pattern:

- one dataclass owns the fact family
- ``from_*`` constructors declare repeated mappings once
- JSON, CSV, summary rows, and UI projections derive from the same source

Bad pattern:

- several parallel dicts or records store overlapping writable facts
- builders repeat the same field copying in multiple places
- updates require manual synchronization across representations


Typical Failure Modes and Their Fixes
-------------------------------------

Failure: Repeated ``hasattr`` dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- the code is rediscovering semantic role from a partial view

Fix:

- introduce an ABC or explicit nominal variant family


Failure: Giant overlapping parameter bags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- the real domain object has not been declared

Fix:

- create one authoritative request or context dataclass


Failure: Repeated field assignment blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- the mapping is known but not declared once

Fix:

- create authoritative constructors or shared builders


Failure: String-based registries for semantic roles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- the system wants nominal identity but is simulating it with weak keys

Fix:

- key the registry by enum or type identity


Failure: Structurally equal companion types losing lineage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- a lazy/base or generated/original distinction is semantically real

Fix:

- preserve nominal type lineage in registries and provenance returns


Failure: Manually synchronized dual registries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meaning:

- the system lacks one authoritative bijection mechanism

Fix:

- centralize registration with bijection checks on nominal keys


How to Read the Codebase with This Lens
---------------------------------------

When opening a module, ask:

1. What are the semantic families here?
2. What are the authoritative fact owners?
3. What uses nominal identity already?
4. Where is the code still trying to recover identity from structure?
5. Where are exports or summaries stored as parallel writable forms?
6. Where is provenance currently thrown away?

Red flags:

- ``Protocol`` at semantic boundaries
- sentinel attributes standing in for class identity
- string-based role dispatch where type or enum identity exists
- repeated ``hasattr`` / ``getattr`` chains
- repeated builders that unpack the same source object field by field
- separate dicts that mirror the same facts
- structurally identical generated types treated as interchangeable despite different lineage


What Agents Must Preserve
-------------------------

- declared semantic role families
- nominal identity where capabilities depend on it
- provenance-carrying data flow
- single-source fact ownership
- fail-loud contract enforcement
- class-level architecture when the design is inherently class-level


What Agents Must Remove
-----------------------

- fake flexibility based on partial structural views
- duplicated machinery that belongs in an ``ABC``
- string registries where nominal keys are the real invariant
- repeated field-copy constructors
- independently writable derived views
- sentinel attributes pretending to be full type identity


Minimal Pre-Edit Checklist
--------------------------

Before touching code, write down:

1. semantic role(s)
2. current representation(s)
3. fiber collisions under those representations
4. required capability that structural reasoning cannot recover
5. chosen nominal tool
6. authoritative fact owner
7. derived views to remove or centralize

If you cannot fill in all seven items, you are not ready to refactor safely.


Stop Conditions for Architectural Refactoring
---------------------------------------------

Stop only when all of the following are true:

- semantic families are explicit
- identity-sensitive behavior is carried by nominal tools
- authoritative fact ownership is obvious
- derived views are derived, not manually synchronized
- provenance is visible or intentionally preserved
- repeated probing has been replaced with declared contracts
- repeated builders and field-copy patterns have been centralized
- remaining duplication is genuinely orthogonal or trivial


Final Commandment
-----------------

Do not ask only:

- does this object have the needed methods?

Also ask:

- what semantic identity is being hidden by the current representation?
- what capability does the architecture need that structure cannot recover?
- what nominal handle should carry that distinction?
- what fact should be authoritative here?

If you answer those questions first, the correct refactor shape usually becomes obvious.
