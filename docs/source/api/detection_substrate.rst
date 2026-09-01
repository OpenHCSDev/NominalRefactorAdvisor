Detection Substrate
===================

This page documents the internal detector framework used to extend the advisor.
It is reference material for maintainers, not the primary end-user API.

Detector Base Classes
---------------------

.. automodule:: nominal_refactor_advisor.detectors
   :members: DetectorConfig, IssueDetector, PerModuleIssueDetector, CandidateFindingDetector, EvidenceOnlyPerModuleDetector, StaticModulePatternDetector, default_detectors


Structural Hypothesis Substrate
-------------------------------

Structural pattern metadata is carried by ``PatternId`` members in
:mod:`nominal_refactor_advisor.patterns`. The planner module clusters findings
through shared source evidence without selecting an application order.

For maintainers, the important split is:

- each ``PatternId`` member owns its identity, required relation, and witness capabilities
- ``build_refactor_plans`` preserves every observed pattern in stable identity order
- pattern evidence states the missing relation without prescribing a normal form
- graph execution classes expose structural evidence only; they do not rank or
  schedule refactors

The public ``build_refactor_plans`` entrypoint is documented in
:doc:`public_api`.
