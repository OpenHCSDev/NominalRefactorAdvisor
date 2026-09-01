Public API
==========

This page documents the import surface that downstream tooling should prefer.
It does not restate every internal helper module.

Package Surface
---------------

.. automodule:: nominal_refactor_advisor
   :members:


CLI Entry Points
----------------

.. automodule:: nominal_refactor_advisor.cli
   :members: analyze_path, analyze_paths, plan_path, plan_paths, main


Planning Surface
----------------

.. automodule:: nominal_refactor_advisor.planner
   :members: build_refactor_plans


Codemod Planning Surface
------------------------

The codemod surface models planned rewrites and simulations without applying
edits. Its cancelable-composition signal is generic: it treats pack, unpack, and
field-forwarding wrappers as factorable product morphisms when they preserve
common fields and do not own an invariant.

Rejected ``FindingRecipeSynthesisRecord`` values expose ``proof_obstacles``.
Each obstacle identifies the nominal executable declaration that failed to
prove a recipe and carries that declaration's diagnostic.  The record's
``reason`` is a concise summary; consumers that need actionable proof detail
should inspect the structured obstacles instead of parsing that summary.

.. automodule:: nominal_refactor_advisor.codemod
   :members: PlannedSourceRewrite, RefactorRecipeOperation, ReplaceTargetOperation, FindingRecipeProofObstacle, FindingRecipeSynthesisRecord, CodemodActionability, CodemodStrategy, CodemodApplicability, CodemodCandidate, CodemodSimulationReport, codemod_candidates_from_impact_ranking, simulate_codemod_candidates, format_codemod_unified_diff, apply_codemod_simulation, simulate_planned_rewrites, CancelableCompositionSignal, detect_cancelable_composition_signals


Result Records And Taxonomy
---------------------------

See :doc:`theory_and_results` for the frozen result dataclasses, taxonomy values,
and pattern metadata referenced by the public entrypoints.
