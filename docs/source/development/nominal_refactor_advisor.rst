Nominal Refactor Advisor
========================

This document explains the advisor's repository-specific architecture and
maintenance rules. It does not restate the generated runtime catalogs.

For the current shipped surfaces, use:

- :doc:`../api/pattern_catalog` for the canonical pattern taxonomy
- :doc:`../api/detector_catalog` for the registered detector set
- :doc:`../api/public_api` for stable entrypoints and result surfaces

Purpose
-------

The tool is not a generic clone detector. It is a structural issue classifier
that maps AST observations onto a nominal-architecture model and then prescribes
one canonical refactoring shape.

The repository-specific aim is:

- deterministic, AST-only analysis
- generic detectors rather than repo-local symbol matching where possible
- shared pattern metadata across findings, plans, and docs
- self-hosting discipline: detect a structural smell before manually refactoring it


Canonical Authorities
---------------------

The main authorities are:

- ``nominal_refactor_advisor.patterns.PATTERN_SPECS`` for canonical pattern metadata
- ``nominal_refactor_advisor.detectors.IssueDetector`` for the detector registry
- ``nominal_refactor_advisor.models`` for findings, plans, and metrics
- ``nominal_refactor_advisor.planner`` for subsystem-level plan synthesis
- ``nominal_refactor_advisor.observation_*`` for the structural observation substrate

The docs follow the same rule. Pattern and detector catalogs are generated from
those code authorities rather than restated by hand in development pages.


Current Design Rules
--------------------

The advisor currently enforces these repository-level choices:

- detector bases centralize orchestration so concrete detectors keep only matching logic
- findings and plans use authoritative dataclasses rather than loose dict bags
- class-time registration is normalized through ``metaclass-registry`` rather than local manual rosters
- docs prefer generated catalogs and code-validated object references over hand-maintained inventories
- semantic-role normalization is allowed where lexical equality is too weak to recover true shared authority


Extension Workflow
------------------

When extending the tool:

1. decide whether the smell fits an existing pattern in :doc:`../api/pattern_catalog`
2. widen an existing generic detector before adding a new detector family
3. keep any new detector generic over semantic roles rather than advisor-local names
4. add or update regression coverage in ``tests/test_refactor_advisor.py``
5. rerun the advisor on itself before refactoring the newly detected smell

This keeps the tool self-hosting and limits detector proliferation.


How To Read The Rest Of The Docs
--------------------------------

- Read :doc:`nominal_architecture_playbook` for the core conceptual model.
- Read :doc:`agent_refactoring_crash_course` for the operational refactoring procedure.
- Read :doc:`nominal_identity_case_studies` for worked examples of when nominal identity matters.
- Read :doc:`advisor_self_audit_detection_plan` and :doc:`advisor_detection_edit_drafts` as historical implementation notes rather than current runtime reference.


What This Page Deliberately Does Not Do
---------------------------------------

This page does not duplicate:

- the shipped detector list
- the shipped pattern list
- the public API symbol inventory

Those are already maintained as code-derived references elsewhere. This page is
only for repository-specific architecture and maintenance policy.
