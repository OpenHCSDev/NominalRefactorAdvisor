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

Include graph-clustered structural hypotheses:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package --include-plans

Focused Edit Loops
------------------

For a bounded file-focused check during development, use the compact loop
payload:

.. code-block:: bash

   nominal-refactor-advisor changed_module.py --json --json-payload loop

When this command infers a larger package context and no reusable context cache
exists, it analyzes the requested files with per-module detectors only.  The
JSON ``scan_status`` then reports ``focused_local_partial`` together with the
analyzed and omitted detector counts.  This is an intentionally incomplete
edit-loop result, not proof that context-dependent findings are absent.

Use the full payload or provide ``--context-root`` when an exact contextual
scan is required:

.. code-block:: bash

   nominal-refactor-advisor changed_module.py --context-root path/to/package \
     --json --json-payload full

Inspecting Bounded Codemod Candidates
-------------------------------------

To inspect a declaration-owned recipe against the current source snapshot,
request a simulation:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-synthesize-plan --codemod-simulate --json

This proves only that the candidate is coherent in the current snapshot.  It
does not establish that the candidate belongs to a globally complete refactor
trajectory.  NRA therefore blocks one-shot plan export and source application
while the planning horizon is ``current_snapshot`` or ``unproved``.

Some findings intentionally have no recipe.  Repeated source proves that one
maintenance object exists, but it does not necessarily prove where that object
belongs.  NRA keeps such findings as evidence instead of inventing an authority.

To extract a declaration and its movable source-local dependency closure into a
new module, provide only the semantic roots:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-plan - --codemod-simulate <<'JSON'
   {
     "recipes": [{
       "recipe_id": "extract-source-edit-algebra",
       "operations": [{
         "operation": "extract_symbol_closure_to_new_module",
         "file_path": "package/monolith.py",
         "root_symbol_qualnames": ["NominalSourceEdit"],
         "destination_path": "package/source_edits.py"
       }]
     }]
   }
   JSON

Review the simulated diff, then replace ``--codemod-simulate`` with
``--codemod-apply``.  NRA derives transitive movable declarations, imports,
source re-exports, repository-local consumer imports, and the new-file revision
contract.  A non-movable or unresolved dependency fails preflight without
writing either module.

Runtime imports remain runtime imports.  Imports declared under a recognised
``typing.TYPE_CHECKING`` guard retain that scope when the dependency is used
only by deferred annotations.  NRA also rewrites guarded imports in repository
consumers and fails preflight when eager annotation evaluation would make a
guarded dependency unavailable.

Relative imports are resolved to their canonical module identity before a
declaration moves.  NRA then renders the dependency from the destination, so a
move into a deeper or shallower package does not silently retarget the import.
An existing destination binding satisfies the dependency only when its resolved
authority matches; a same-named local declaration or different import fails
preflight.

Source and destination modules must use the same annotation evaluation policy
when moved declarations contain annotations.  Align their
``from __future__ import annotations`` usage before applying the move; NRA
fails preflight instead of changing annotation semantics.

To prove a goal across reachable source states, run:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-refactor-goal semantic_carrier --json

The goal runner enumerates every clean compatible recipe batch at each state,
explores the resulting exact source-state graph, and deduplicates cycles by
source identity.  It emits an applicable replay sequence only when complete
exploration reaches one unique terminal source state.  Local compression scores
and candidate order do not select a branch.

Guards supplied through ``--codemod-plan`` are terminal invariants.  The search
may cross an intermediate state that violates them when a later stage repairs
that state.  The terminal source must satisfy every supplied guard, and the
exported replay attaches those guards to its final stage.  Guards owned by an
individual recipe continue to validate that recipe's immediate result.

``--codemod-goal-max-stages``, ``--codemod-goal-max-states``, and
``--codemod-goal-max-branches`` bound the proof search.  Reaching any bound
produces an ``incomplete`` trajectory proof and leaves the checkout unchanged.
Distinct terminal states produce ``ambiguous_terminal_states``; a completely
explored graph with no terminal produces ``no_terminal_state``.

The default persistent cache is maintained at most once per hour.  Retention
keeps at most 128 recently used analysis roots, 4 GiB across those roots, 2 GiB
within one active root, and four recent exact semantic-graph generations.
Explicit ``--cache-dir`` locations are caller-managed and are not pruned by
this policy.

Reproducible Performance Gate
-----------------------------

``nominal-refactor-benchmark`` runs the compact focused scan twice against one
isolated cache, samples the complete process-tree RSS, and emits cold/warm JSON
measurements.  Optional ceilings turn it into a regression gate:

.. code-block:: bash

   nominal-refactor-benchmark \
     --max-cold-seconds 15 --max-cold-rss-mb 180 \
     --max-warm-seconds 5 --max-warm-rss-mb 180 \
     changed_module.py another_changed_module.py

The command fails when either subprocess times out, leaks a nonzero exit,
emits invalid JSON, changes finding counts between cold and warm runs, omits
the partial-scan contract, or exceeds a supplied time/RSS ceiling.

What Stays Stable
-----------------

For downstream use, treat these as the main supported surfaces:

- ``analyze_path`` for finding generation
- ``plan_path`` and ``build_refactor_plans`` for non-actionable structural hypotheses
- ``AnalysisReport``, ``RefactorFinding``, and ``RefactorPlan`` for results
- ``PatternId`` for canonical pattern identity and metadata

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
- ``nominal_refactor_advisor.patterns``: pattern metadata shared by findings, hypotheses, and docs
- ``nominal_refactor_advisor.models``: result records and finding metrics
- ``nominal_refactor_advisor.observation_*``: structural observation substrate
- ``nominal_refactor_advisor.planner``: subsystem-level evidence clustering over findings

Building Docs
-------------

.. code-block:: bash

   pip install -e .[docs]
   python -m sphinx -b html docs/source docs/_build/html
