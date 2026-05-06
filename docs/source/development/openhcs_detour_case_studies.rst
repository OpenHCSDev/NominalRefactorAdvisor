OpenHCS Detour Case Studies
===========================

This page is the negative companion to the OpenHCS merged-PR case studies.  The
first pass over the corpus emphasized successful compression.  That missed an
important signal: several large PRs introduced abstractions that were later
deleted, externalized, or superseded.  Those wrong turns are more useful to the
advisor than success stories because they identify recommendations NRA must not
make blindly.

Method
------

The analysis used three evidence streams:

- merged PR file statistics and merge commits
- current ``main`` survival of files added or heavily changed by those PRs
- closed unmerged PRs, especially large draft branches and branches whose own
  description names instability

Survival is not treated as a moral score.  A file can disappear because an idea
was wrong, because it was successfully externalized into a dependency, because
the repository was cleaned, or because the same authority moved to a better
nominal object.  The useful question is narrower: what earlier signal could
have distinguished a durable abstraction from a detour?

Merged PR Survival Ledger
-------------------------

The largest later-absent surfaces were not random.  They concentrate around a
few authority questions: UI state ownership, config resolution, registry
machinery, remote runtime boundaries, and development-artifact sprawl.

.. list-table:: Merged PRs with high later-absence
   :header-rows: 1

   * - PR
     - Later-absent touched files
     - Added files later absent
     - Reading
   * - #44 UI anti-duck-typing
     - ``48``
     - ``34``
     - Large ABC/service extraction included useful direction, but many
       concrete UI abstractions were transitional.  ``abstract_manager_widget``
       and ``widget_creation_config`` later moved out through pyqt-formgen
       extraction.
   * - #58 ObjectState extraction
     - ``43``
     - ``13``
     - Mostly successful compression/externalization rather than a detour:
       config framework and auto-registration machinery moved toward
       ``ObjectState`` and external packages.
   * - #14 napari streaming
     - ``41``
     - ``18``
     - Bridge PR.  It introduced ``dual_axis_resolver_recursive`` and
       composition/lazy-config machinery that later proved insufficient or was
       consolidated.
   * - #20 generic config
     - ``37``
     - ``14``
     - Mixed result.  It named the right config-resolution pressure, but
       ``context_manager`` and ``lazy_factory`` later moved to ObjectState.
   * - #45 lazy registry
     - ``34``
     - ``21``
     - The registry idea survived, but the in-repo custom metaclass machinery
       did not.  ``auto_register_meta.py`` was later replaced by the external
       ``metaclass-registry`` package.
   * - #49 artifact cleanup
     - ``32``
     - ``0``
     - Pure negative signal: repo-local plans, logs, and analysis artifacts had
       accumulated until a cleanup PR deleted them.
   * - #51 GUI performance
     - ``23``
     - ``7``
     - Mixed infrastructure.  Log/chat/performance UI work later moved through
       extraction and ObjectState simplification rather than remaining as-is.
   * - #38 memory conversion
     - ``23``
     - ``6``
     - Successful deletion of a bad matrix abstraction; the absent files are
       evidence that the old conversion matrix was the wrong normal form.
   * - #23 OMERO/ZMQ
     - ``21``
     - ``10``
     - Boundary was real, but the first in-repo ZMQ/OMERO runtime shape was
       later externalized or pruned.

Detour Types
------------

Wrong Authority
~~~~~~~~~~~~~~~

Several detours placed state or resolution authority inside UI manager
machinery.  The later ObjectState work shows the better boundary: the UI should
project state, not own the model.

Examples:

- ``parameter_form_manager.py`` repeatedly appears as a large changed/removed
  surface across #4, #9, #14, #17, #20, #44, #51, and #58.
- ``abstract_manager_widget.py`` and ``widget_creation_config.py`` were added
  by #44, then later migrated during pyqt-formgen extraction.
- closed PRs #53, #54, #55, and #56 repeatedly try to add visual feedback,
  flash state, scope coloring, dirty tracking, and context stacks inside GUI
  widget surfaces.

Advisor lesson:

- When state/dirty/flash/preview semantics repeatedly change the same widget
  manager, do not recommend another widget-level service.  First ask whether
  a model authority is missing.

Premature Registry
~~~~~~~~~~~~~~~~~~

The registry story is not "registry bad."  The detour is custom registry
machinery before the domain has stabilized enough to justify it.

Evidence:

- #45 added ``openhcs/core/auto_register_meta.py``.
- Later commits include "ULTIMATE ZERO BOILERPLATE: Eliminate ALL custom
  metaclasses" and "Migrate to external metaclass-registry package."
- #56 introduced ``context_stack_registry.py`` in a WIP branch.
- #57 proposed ``ResolvedValueRegistry`` but remained a plan/branch rather than
  a merged authority.

Advisor lesson:

- A registry recommendation should require a stable key axis, repeated
  discovery lifecycle, and multiple consumers.  A registry for one turbulent UI
  workflow is likely a detour.

Feature Bundle Overload
~~~~~~~~~~~~~~~~~~~~~~~

Some PRs failed because they bundled too many orthogonal concerns.  The clearest
case is #53, whose own title and body warn that the branch is a mess.  It
attempted visual feedback, flash animations, performance optimization, window
close behavior, unsaved-change indicators, and inheritance/placeholder fixes in
one branch.

The follow-up branches show the attempted recovery:

- #54 tries a cleaner visual-feedback subset.
- #55 expands again into scope hierarchy and cache infrastructure.
- #56 adds context-stack registry and reactive dirty tracking.
- #57 narrows to a resolved-value registry.

Advisor lesson:

- A detector should treat a PR/branch that changes presentation, state
  resolution, cache invalidation, and performance together as an authority
  split signal.  The recommendation should be decomposition of the change plan,
  not a larger abstraction.

Externalization Success
~~~~~~~~~~~~~~~~~~~~~~~

Several disappeared files are not failures.  They show successful extraction
into libraries or submodules.

Examples:

- ``auto_register_meta.py`` moves out to ``metaclass-registry``.
- ZMQ runtime files move toward ``zmqruntime``.
- parameter-form UI machinery moves toward ``pyqt-formgen``.
- config framework pieces move toward ``ObjectState``.
- python introspection code moves toward ``python-introspect``.

Advisor lesson:

- Survival analysis needs an "externalized" fate, not just "deleted."  A file
  disappearing after the same named authority becomes a dependency is evidence
  of successful boundary sharpening.

Development Artifact Accretion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#49 and later cleanup commits show that many plans, logs, reviews, and analysis
documents lived inside the production repository until they were deleted.

Advisor lesson:

- NRA should distinguish durable documentation from transient planning residue.
  A repository with many one-off plan/review/log artifacts needs an artifact
  boundary, not another code abstraction.

Closed Unmerged PRs
-------------------

The unmerged corpus confirms the detour story more sharply than merged history.

.. list-table:: High-effort abandoned PRs
   :header-rows: 1

   * - PR
     - Diff shape
     - Status signal
     - Detour reading
   * - #62 pyqt-formgen extraction plans
     - ``16254+ / 33275-`` across ``284`` files
     - draft
     - A serious extraction branch.  Not simply abandoned work; it captures the
       migration from in-repo UI framework to package boundary.
   * - #55 scope visual feedback and performance
     - ``15852+ / 961-`` across ``62`` files
     - draft; PR body says 60 FPS target not met
     - Feature bundle overload: scope colors, flash, cache controls, hierarchy,
       performance, and bug fixes in one branch.
   * - #53 scope visual feedback "MESS"
     - ``9099+ / 308-`` across ``35`` files
     - closed; title/body explicitly warn about mess
     - Negative gold standard.  The PR itself admits chronological chaos and
       mixed concerns.
   * - #54 clean visual feedback subset
     - ``3939+ / 142-`` across ``21`` files
     - closed
     - Narrower than #53 but still presentation-heavy and widget-encoded.
   * - #56 scope feedback plus context stack registry
     - ``3264+ / 233-`` across ``34`` files
     - draft; WIP/debugging
     - Premature registry under unstable UI state semantics.
   * - #50 dynamic function registration
     - ``2310+ / 0-`` across ``9`` files
     - draft
     - Mostly additive infrastructure.  Needs fanout/rent proof and separation
       from the included critical bug fix.
   * - #57 resolved value registry
     - ``627+ / 128-`` across ``8`` files
     - draft; checklist says implementation not complete
     - A likely correct pressure, but not yet a proven authority.

NRA On Abandoned Heads
----------------------

Bounded NRA scans were run on closed PR heads where GitHub still exposed the
``pull/<n>/head`` refs.

.. list-table:: Abandoned PR bounded scans
   :header-rows: 1

   * - PR
     - Scan result
     - Interpretation
   * - #53
     - ``pyqt_gui/widgets`` timed out at ``18s``; ``config_framework`` had
       ``29`` findings
     - The branch was too wide and widget-heavy for cheap analysis.
   * - #54
     - ``pyqt_gui/widgets`` timed out at ``18s``
     - Even the "clean" subset still concentrated complexity in widgets.
   * - #55
     - ``pyqt_gui/widgets`` timed out at ``18s``; ``config_framework`` had
       ``41`` findings
     - Performance/feedback branch grew both presentation and config surfaces.
   * - #56
     - ``pyqt_gui/widgets`` timed out at ``18s``; ``config_framework`` had
       ``25`` findings
     - WIP registry branch retained unresolved config/UI smells.
   * - #57
     - ``config_framework`` had ``25`` findings; widget services had ``42``
       findings including ``under_amortized_infrastructure``
     - Resolved-value service pressure was real, but service extraction was not
       yet rent-proven.
   * - #50
     - ``custom_functions`` had ``7`` findings
     - Additive infrastructure was smaller and more contained, but still needed
       public-surface and product-record cleanup.
   * - #62
     - ``pyqt_gui/widgets`` completed in ``3.49s`` with ``111`` findings;
       ``config_framework`` had ``0`` findings
     - Extraction improved scan partitioning and config cleanliness, but left
       widget-side residue.

Objective Wrong Turns
---------------------

The strongest objective wrong turns are not "this code was later deleted."
They are cases where the evidence says the branch created an authority before
the real owner was known.

1. **Widget-owned state authority**: flash, dirty, preview, resolved value, and
   scope semantics were repeatedly pushed into widget managers.  ObjectState
   later proves that the model should own those facts.
2. **Custom metaclass registry**: in-repo metaclass machinery was replaced by a
   dedicated external package.  The idea was right; the repo-local mechanism
   was the wrong ownership boundary.
3. **Recursive dual-axis resolver**: ``dual_axis_resolver_recursive.py`` was an
   attempt to name a real two-axis resolution problem, but it was later
   consolidated.  The detector lesson is to require an explicit algebraic
   contract for axes, not merely recursive mechanics.
4. **Feature-plus-infrastructure PRs**: #53/#55 mixed visual features,
   performance targets, config hierarchy, cache controls, and bug fixes.  The
   branch shape itself was incoherent.
5. **Planning artifacts as repo surface**: #49 shows that temporary reasoning
   material became repository payload.  That is a process boundary failure.

Detector Implications
---------------------

This analysis suggests new generic advisor capabilities:

- **Survival ledger**: for a refactor branch, compute which new files/classes
  survive after later commits, which are deleted, and which are externalized.
- **Abandoned-work miner**: closed unmerged PRs should be first-class
  calibration cases.  They contain failed abstraction attempts before history
  normalizes them.
- **Authority instability score**: repeated large edits to the same manager
  file across PRs indicate that the manager is probably hosting the wrong
  authority.
- **Feature bundle detector**: a branch touching UI presentation, state
  resolution, cache invalidation, and performance at once should be flagged as
  multiple orthogonal concerns.
- **Externalization classifier**: deletion followed by dependency/submodule
  introduction should be classified as boundary improvement, not failure.
- **Registry maturity guard**: recommend registries only when the key axis,
  discovery lifecycle, and consumer fanout are all stable.
