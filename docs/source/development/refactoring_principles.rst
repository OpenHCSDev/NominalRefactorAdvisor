Refactoring Principles
======================

Compact maintenance heuristics for simplifying code without losing semantic
clarity.

This page complements :doc:`systematic_refactoring_framework` by focusing on the
small local moves that usually precede a larger architectural collapse.

Local Simplification Rules
--------------------------

Factor Shared Shape, Not Just Shared Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If two regions repeat the same logical structure with only parameter variation,
extract the common shape.

.. code-block:: python

   # Before
   if use_primary:
       result = build_plan(primary_source, primary_budget)
   else:
       result = build_plan(fallback_source, fallback_budget)

   # After
   result = build_plan(
       primary_source if use_primary else fallback_source,
       primary_budget if use_primary else fallback_budget,
   )

Inline Single-Use Transport Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a helper only forwards one call site and contributes no independent
authority, inline it.

Keep the helper only when it improves the public surface, hides an important
boundary, or centralizes a real invariant.

Prefer Comprehensions And Tables Over Repeated Ceremony
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fieldwise loops, registry rows, and literal mappings are usually clearer when
written as:

- comprehensions for one-pass data shaping
- small declarative tables for closed data families
- one shared constructor for repeated record assembly

Do not introduce a table if the real abstraction is behavioral dispatch; in that
case a nominal family is usually the better authority.

Remove Defensive Noise Around Guaranteed Contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the architecture guarantees a field or method, access it directly.

Bad:

.. code-block:: python

   value = getattr(record, "field", None)
   if hasattr(step, "process"):
       step.process(context)

Better:

.. code-block:: python

   value = record.field
   step.process(context)

Fail loudly when the contract is violated instead of preserving a broken state
with fallback behavior.

Escalation Rules
----------------

Escalate from local cleanup to structural refactor when you see:

- the same axis represented in multiple writable places
- repeated wrappers around one downstream authority
- a closed family encoded by conditionals instead of nominal identity
- repeated record lifecycle helpers across sibling classes
- repeated projection or builder code that is clearly derivable

At that point, stop doing syntax cleanup and choose an authoritative surface.

Quick Checklist
---------------

Before keeping a helper, table, or layer, ask:

1. Does it own new information?
2. Does it enforce a real invariant?
3. Does it improve the public surface?
4. Would deleting it make the system harder to explain?

If the answer is "no" across the board, it is probably transport noise.
