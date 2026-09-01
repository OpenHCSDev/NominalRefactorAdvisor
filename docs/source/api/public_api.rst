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


Structural Hypothesis Surface
-----------------------------

.. automodule:: nominal_refactor_advisor.planner
   :members: build_refactor_plans


Codemod Candidate Surface
-------------------------

The codemod surface models source-anchored candidate rewrites and simulations.
A clean current-snapshot simulation is not an application recommendation;
export and application require a proof across reachable refactor trajectories.
Its cancelable-composition signal is generic: it treats pack, unpack, and
field-forwarding wrappers as factorable product morphisms when they preserve
common fields and do not own an invariant.
``RefactorRecipeOperation`` subclasses declare wire semantics with
``codemod_payload_field`` on their dataclass fields.  The inherited operation
codec catalog is derived from those declarations, so adding a field does not
require a parallel serialization method.

Rejected ``FindingRecipeSynthesisRecord`` values expose ``proof_obstacles``.
Each obstacle identifies the nominal executable declaration that failed to
prove a recipe and carries that declaration's diagnostic.  The record's
``reason`` is a concise summary; consumers that need diagnostic proof detail
should inspect the structured obstacles instead of parsing that summary.
For detector-owned synthesis, ``executable_declaration`` names the concrete
detector declaration whose MRO supplied the recipe behaviour.  Inferred
metric-driven synthesis names the strategy or builder declaration selected by
the finding evidence.

Exact repeated methods without a proved owner remain evidence-only findings.
The ``exact_leaf_method_ancestor_promotion`` detector emits a codemod candidate
only when one existing direct authority is unique, every direct child
participates and is a leaf, the method source is exact and promotion-safe, all
receiver requirements belong to the authority contract, no competing ancestor
binds the promoted names, no decorator or class-creation hook can observe the
ownership move, and the complete method batch pays compression rent.  The
codemod preflight reconstructs the same proof from the current full AST.

.. automodule:: nominal_refactor_advisor.codemod
   :members: PlannedSourceRewrite, RefactorRecipeOperation, codemod_payload_field, ReplaceTargetOperation, PromoteExactLeafMethodsToAncestorOperation, FindingRecipeProofObstacle, FindingRecipeSynthesisRecord, FindingRecipeFrontierBudget, FindingRecipeTrajectoryFrontier, CodemodSimulationReport, format_codemod_unified_diff, apply_codemod_simulation, simulate_planned_rewrites, CancelableCompositionSignal, detect_cancelable_composition_signals

Goal Trajectory Surface
-----------------------

The goal runner reports complete reachable-state evidence separately from the
single replay sequence.  ``PROVED`` means that exhaustive exploration found one
unique terminal source state.  Ambiguity or any frontier, depth, or state-budget
obstacle keeps ``stages`` empty and prevents application.
Target-free states that increase any complete-scan finding class relative to
the starting state are recorded as unjustified-debt terminals rather than
accepted as successful refactors.  The search may pass through intermediate
states with additional obligations, but a proved terminal must discharge them.
``FindingObligationClass`` derives that identity from the finding's pattern,
capability gap, and required-relation context.  Detector provenance is reported
separately, so a relation observed by a different detector is not mistaken for
a newly introduced obligation.
Caller-supplied architecture guards are evaluated at terminal states and stored
on the final replay document; recipe-owned guards remain transition-local.

.. automodule:: nominal_refactor_advisor.codemod_workflow
   :members: CodemodRefactorGoalRunner, CodemodRefactorGoalReport, CodemodRefactorTrajectoryBudget, CodemodRefactorTrajectoryProof, CodemodRefactorTrajectoryStatus, CodemodRefactorTrajectoryObstacle, CodemodRefactorUnjustifiedDebtTerminal


Result Records And Taxonomy
---------------------------

See :doc:`theory_and_results` for the frozen result dataclasses, taxonomy values,
and pattern metadata referenced by the public entrypoints.
