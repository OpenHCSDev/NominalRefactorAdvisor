Getting Started
===============

The advisor is an AST-driven tool for finding structural refactors that collapse
duplicate authority, replace duck-typed branching with nominal boundaries, and
derive repeated surfaces from one canonical source.

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

What Stays Stable
-----------------

For downstream use, treat these as the main supported surfaces:

- ``analyze_path`` for finding generation
- ``plan_path`` and ``build_refactor_plans`` for composed plan synthesis
- ``AnalysisReport``, ``RefactorFinding``, and ``RefactorPlan`` for results
- ``PATTERN_SPECS`` for canonical pattern metadata

How To Read The Rest Of The Docs
--------------------------------

- Use :doc:`public_api` for importable entrypoints.
- Use :doc:`theory_and_results` for result dataclasses, taxonomy, and pattern metadata.
- Use :doc:`pattern_catalog` and :doc:`detector_catalog` for the current shipped behavior.
- Use :doc:`../development/index` for rationale, case studies, and maintenance workflow.

Architecture Map
----------------

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
