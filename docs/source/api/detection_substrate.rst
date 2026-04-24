Detection Substrate
===================

Detector Base Classes
---------------------

.. automodule:: nominal_refactor_advisor.detectors
   :members: DetectorConfig, IssueDetector, PerModuleIssueDetector, CandidateFindingDetector, EvidenceOnlyPerModuleDetector, StaticModulePatternDetector, default_detectors


Planning Substrate
------------------

Planning metadata is carried by ``PATTERN_SPECS`` in
:mod:`nominal_refactor_advisor.patterns`. The planner module provides the
subsystem-level composition surface that consumes that metadata.

.. automodule:: nominal_refactor_advisor.planner
   :members: build_refactor_plans
