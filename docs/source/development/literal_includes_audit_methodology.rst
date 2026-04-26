Python Object Reference Documentation Methodology
=================================================

Methodology for keeping the advisor's documentation aligned with the codebase
without manually synchronizing long symbol inventories.

Core Principle
--------------

Prefer validated object references and generated catalogs over pasted code or
hand-maintained symbol lists.

In this repository that means:

- generated pattern docs come from ``PATTERN_SPECS``
- generated detector docs come from the registered ``IssueDetector`` family
- API pages use ``automodule`` and object references for stable roots
- development pages explain rationale and maintenance procedure rather than
  restating runtime inventories

Why This Works
--------------

Sphinx already gives the right failure mode:

- invalid references fail during docs build
- renamed symbols surface immediately
- generated pages stay aligned with runtime authority

That lets the code remain the source of truth while the docs remain readable.

Audit Process
-------------

1. Identify hand-maintained symbol inventories.
2. Decide whether the source of truth already exists in code.
3. Replace the inventory with either:

   - ``automodule`` or object references for stable public surfaces
   - generated pages for authoritative registries and tables
   - short prose that explains the rationale instead of enumerating symbols

4. Build the docs with warnings treated seriously.

What To Avoid
-------------

Avoid:

- long pasted API lists that drift from the code
- code examples that pretend to be authoritative when the code is changing
- development pages that restate generated runtime facts
- local terminology that contradicts the public API pages

What To Keep
------------

Keep:

- short conceptual prose before generated or autodoc material
- references to stable public roots
- generated catalogs where the runtime already has a registry or authoritative
  table
- narrowly chosen code blocks that explain a pattern better than prose alone

Advisor-Specific Rule
---------------------

If a development page needs to describe the current detector set or pattern set,
it should link to the generated API pages instead of reproducing the inventory by
hand.

Validation Command
------------------

.. code-block:: bash

   python -m sphinx -E -b html docs/source docs/_build/html
