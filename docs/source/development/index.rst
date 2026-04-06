Development
===========

This directory combines repository-specific advisor notes with reusable architecture and refactoring
documents copied from ``dq-dock-extracted``.

The imported documents were copied because their guidance is general-purpose even when some examples still
use OpenHCS names or broader repository examples.

Core Advisor Docs
-----------------

.. toctree::
   :maxdepth: 1

   nominal_refactor_advisor
   advisor_self_audit_detection_plan
   advisor_detection_edit_drafts

Reusable Architecture Docs
--------------------------

.. toctree::
   :maxdepth: 1

   systematic_refactoring_framework
   architectural_refactoring_patterns
   refactoring_principles
   respecting_codebase_architecture
   nominal_architecture_playbook
   agent_refactoring_crash_course
   nominal_identity_case_studies
   literal_includes_audit_methodology
   compositional_commit_message_generation
   git_worktree_testing

Notes
-----

The following development documents were intentionally not copied because they remain subsystem-specific
rather than reusable for the standalone advisor repository:

- ``dq_dock_refactor_advisor.rst``
- ``dq_dock_agent_handoff_guide.rst``
- ``dq_dock_architecture_debugging_guide.rst``
- ``dq_dock_failure_forensics_playbook.rst``
- ``docking_pipeline_audit_1hk4.rst``
- ``lean_jax_python_bridge_framework.rst``
- ``openhcs_architecture_lineage.rst``
