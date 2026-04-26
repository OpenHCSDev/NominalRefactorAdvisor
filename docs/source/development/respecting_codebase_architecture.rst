Respecting Codebase Architecture
================================

Background note on fail-loud design and architectural contract respect.

The core rule is simple: if the architecture guarantees something, code should
use that guarantee directly instead of pretending the system is always half
broken.

What Architectural Respect Means
--------------------------------

Respecting architecture means:

- trusting constructor and ``ABC`` contracts
- keeping validation at the real boundary
- failing loudly when invariants are broken
- avoiding fallback code that hides structural bugs

Disrespect usually looks like:

- ``getattr(obj, "field", default)`` for guaranteed fields
- ``hasattr`` around abstract methods
- catching ``AttributeError`` only to fabricate a fallback value
- repeated calls to recover information that is already available in scope

Example
-------

Bad:

.. code-block:: python

   step_name = getattr(step, "name", "N/A") if hasattr(step, "name") else "N/A"

Better:

.. code-block:: python

   step_name = step.name

If ``step.name`` is missing, that is an architectural violation. The right
outcome is a direct failure, not a disguised fallback.

Information Reuse
-----------------

Architectural respect also means reusing information once it has already been
retrieved.

Bad:

.. code-block:: python

   config = builder.effective_config()
   initialize(context, builder)
   config = builder.effective_config()
   context.visualizer = config.visualizer

Better:

.. code-block:: python

   config = builder.effective_config()
   initialize(context, builder, config=config)
   context.visualizer = config.visualizer

When the data is already present, re-fetching it signals weak ownership and
makes provenance harder to follow.

Checklist
---------

Before adding defensive code, ask:

1. Is this failure expected as part of normal operation?
2. Is the missing field or method actually optional by contract?
3. Am I preserving a useful recovery path, or just masking a bug?
4. Could the invariant be enforced earlier instead?

If the failure is not expected, the usual answer is to remove the fallback and
let Python fail naturally.
