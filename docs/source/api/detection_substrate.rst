Detection Substrate
===================

This page documents the internal detector framework used to extend the advisor.
It is reference material for maintainers, not the primary end-user API.

Detector Base Classes
---------------------

.. automodule:: nominal_refactor_advisor.detectors
   :members: DetectorConfig, IssueDetector, PerModuleIssueDetector, CandidateFindingDetector, EvidenceOnlyPerModuleDetector, StaticModulePatternDetector, default_detectors


Planning Substrate
------------------

Planning metadata is carried by ``PATTERN_SPECS`` in
:mod:`nominal_refactor_advisor.patterns`. The planner module provides the
subsystem-level composition surface that consumes that metadata.

For maintainers, the important split is:

- ``PATTERN_SPECS`` defines dependencies, synergy, and builder selection
- ``build_refactor_plans`` clusters findings and materializes plan output
- the planner module's internal builder families turn pattern metadata into
  concrete plan steps and action records

The public ``build_refactor_plans`` entrypoint is documented in
:doc:`public_api`.
