Development
===========

These pages are design notes, background theory, and historical implementation
records. They are not the canonical runtime reference surface.

For shipped behavior, use the generated API docs:

- :doc:`../api/pattern_catalog` for the current pattern taxonomy
- :doc:`../api/detector_catalog` for the current detector set
- :doc:`../api/public_api` for stable programmatic entrypoints

This directory combines:

- repository-specific advisor notes
- imported background material copied from ``dq-dock-extracted``
- operational notes that are useful during maintenance but do not define the tool

Repository-Specific Notes
-------------------------

.. toctree::
   :maxdepth: 1

   nominal_refactor_advisor
   advisor_self_audit_detection_plan
   advisor_detection_edit_drafts

Conceptual Background
---------------------

.. toctree::
   :maxdepth: 1

   nominal_architecture_playbook
   agent_refactoring_crash_course
   nominal_identity_case_studies

Imported Background Heuristics
------------------------------

.. toctree::
   :maxdepth: 1

   systematic_refactoring_framework
   refactoring_principles
   respecting_codebase_architecture
   architectural_refactoring_patterns
   literal_includes_audit_methodology

Operational Notes
-----------------

.. toctree::
   :maxdepth: 1

   compositional_commit_message_generation
   git_worktree_testing

Notes
-----

Most imported background pages still use OpenHCS examples. They remain useful as
theory or heuristics, but they should not be treated as the canonical statement
of the advisor's current shipped patterns, detectors, or public API.

The following development documents were intentionally not copied because they
remain subsystem-specific rather than reusable for the standalone advisor
repository:

- ``dq_dock_refactor_advisor.rst``
- ``dq_dock_agent_handoff_guide.rst``
- ``dq_dock_architecture_debugging_guide.rst``
- ``dq_dock_failure_forensics_playbook.rst``
- ``docking_pipeline_audit_1hk4.rst``
- ``lean_jax_python_bridge_framework.rst``
- ``openhcs_architecture_lineage.rst``
