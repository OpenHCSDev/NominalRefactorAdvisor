Development
===========

These pages explain why the advisor is shaped the way it is and how to evolve it
without regressing its architecture.

For the current shipped surface, use the generated API docs:

- :doc:`../api/pattern_catalog` for the current pattern taxonomy
- :doc:`../api/detector_catalog` for the current detector set
- :doc:`../api/public_api` for stable programmatic entrypoints

If you are changing the advisor itself, read these pages in roughly this order:

1. :doc:`nominal_refactor_advisor`
2. :doc:`nominal_architecture_playbook`
3. :doc:`agent_refactoring_crash_course`

This directory then branches into three kinds of material:

- repository-specific advisor notes
- conceptual background that explains the tool's architectural stance
- maintenance playbooks and operational procedures used while evolving the repo

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
   openhcs_merged_pr_case_studies
   openhcs_high_significance_pr_deep_dives
   openhcs_focused_pr_notes
   openhcs_diff_evolution_case_studies
   openhcs_detour_case_studies
   dna_case_study
   systematic_refactoring_framework
   refactoring_principles
   respecting_codebase_architecture
   architectural_refactoring_patterns

Documentation And Operations
----------------------------

.. toctree::
   :maxdepth: 1

   literal_includes_audit_methodology
   compositional_commit_message_generation
   git_worktree_testing
