Systematic Refactoring Framework
================================

Background framework for large structural refactors in the advisor and in other
Python codebases the tool analyzes.

This page is intentionally generic. It does not define the advisor's shipped
pattern taxonomy; it records the maintenance procedure that keeps structural
refactors disciplined.

Core Goal
---------

The goal of refactoring is not to make the code merely shorter. The goal is to
reduce writable authority:

- one semantic axis should have one authoritative owner
- duplicated orchestration should collapse into one reusable form
- derived surfaces should be generated from their source rather than maintained
  by hand
- error signaling should become clearer, not more defensive

Operating Principles
--------------------

Declare Before You Derive
~~~~~~~~~~~~~~~~~~~~~~~~~

If a subsystem has a real semantic family, declare it explicitly with the right
nominal tool:

- ``ABC`` and subclasses for behavioral families
- enums for closed lightweight axes
- frozen dataclasses for authoritative fact rows
- registries for class-level discovery when enumeration is required

Do not try to recover architecture later from strings, sentinel fields, or
ad hoc branching if the semantic family is already present.

Prefer Directness Over Transport Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unnecessary routing code is a smell:

- wrappers that only forward a single call
- adapter records that only restate existing fields
- secondary tables keyed by the same closed axis
- guard code that probes for attributes the architecture already guarantees

When a layer exists only to transport information without changing authority, it
should usually be collapsed.

Preserve Nominal Identity Where Capabilities Depend On It
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the code needs enumeration, provenance, ordering, or exact dispatch, use a
nominal boundary rather than a structural approximation.

Typical examples:

- use an ``ABC`` family instead of repeated ``hasattr`` checks
- use type registration instead of handwritten subclass rosters
- use a dataclass row authority instead of parallel dicts keyed by the same enum

Refactoring Loop
----------------

1. Identify the semantic axis or duplicated authority.
2. Decide which representation should become authoritative.
3. Detect the current drift mechanically where possible.
4. Collapse the duplicated or parallel surfaces into one authority.
5. Rebuild derived views from that authority.
6. Verify that the system is simpler in both structure and explanation.

What To Look For
----------------

Common structural smells:

- repeated ``if/elif`` dispatch over strings, enums, or classes
- manual registries, rosters, and export lists
- sibling dataclasses with repeated lifecycle helpers
- parallel dicts keyed by one enum or type family
- repeated builder calls or projection helpers
- defensive attribute probes around architecturally-guaranteed fields
- runtime shells that merely copy fields into a second record

Decision Table
--------------

When choosing a collapse target:

- behavior varies, identity matters, dispatch is primary:
  use a nominal class family
- immutable facts vary, behavior is minimal:
  use one frozen dataclass row type plus authoritative instances
- both data and behavior vary:
  choose the true source of authority, then derive the other surface

Validation Checklist
--------------------

After refactoring, confirm:

- one authoritative surface owns the fact family
- secondary forms are derivable instead of writable
- control flow became more explicit rather than more abstract
- provenance is easier to explain
- tests and documentation describe the new shape directly

The framework is doing its job when the final explanation becomes shorter than
the original implementation.
