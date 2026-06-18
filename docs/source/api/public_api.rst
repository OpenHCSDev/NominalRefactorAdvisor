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
   :members: analyze_modules, analyze_path, plan_path, main


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

.. automodule:: nominal_refactor_advisor.codemod
   :members: PlannedSourceRewrite, CodemodStrategy, CodemodStrategyRegistry, CodemodRewriteBuilder, SortedTupleWrapperCodemodBuilder, SourceLocationEvidencePropertyCodemodBuilder, ZippedSourceLocationEvidencePropertyCodemodBuilder, CodemodApplicability, CodemodCandidate, CodemodSimulationReport, codemod_candidates_from_impact_ranking, codemod_candidates_with_automated_rewrites, simulate_planned_rewrites, CancelableCompositionSignal, detect_cancelable_composition_signals


Result Records And Taxonomy
---------------------------

See :doc:`theory_and_results` for the frozen result dataclasses, taxonomy values,
and pattern metadata referenced by the public entrypoints.
