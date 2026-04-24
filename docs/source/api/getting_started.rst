Getting Started
===============

The advisor is an AST-driven tool for finding structural refactors that collapse
duplicate authority, replace duck-typed branching with nominal boundaries, and
derive repeated surfaces from one canonical source.

The documentation model intentionally keeps authority in code:

- pattern docs are generated from ``PATTERN_SPECS``
- detector docs are generated from the registered ``IssueDetector`` family
- public API pages document only stable roots, not internal helper sprawl
- development pages hold theory, case studies, and design notes that are useful
  but not themselves authoritative runtime metadata

Quickstart
----------

Analyze a package from the CLI:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package

Emit JSON instead of Markdown:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package --json

Include composed subsystem plans:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package --include-plans

Programmatic Surface
--------------------

The stable package roots are:

- ``analyze_path`` for finding generation
- ``plan_path`` and ``build_refactor_plans`` for composed plan synthesis
- ``AnalysisReport``, ``RefactorFinding``, and ``RefactorPlan`` for results
- ``PATTERN_SPECS`` for canonical pattern metadata

Architecture Map
----------------

The main authorities are:

- ``nominal_refactor_advisor.cli``: CLI entrypoints and output formatting
- ``nominal_refactor_advisor.detectors``: registered detector family and finding synthesis
- ``nominal_refactor_advisor.patterns``: canonical pattern metadata shared by findings, plans, and docs
- ``nominal_refactor_advisor.models``: result records and finding metrics
- ``nominal_refactor_advisor.observation_*``: structural observation substrate
- ``nominal_refactor_advisor.planner``: subsystem-level plan synthesis over findings

Building Docs
-------------

.. code-block:: bash

   pip install -e .[docs]
   python -m sphinx -b html docs/source docs/_build/html
