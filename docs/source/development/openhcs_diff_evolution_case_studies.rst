OpenHCS Diff Evolution Case Studies
===================================

This page cross-references the OpenHCS merged-PR descriptions with file-level
diff evidence and bounded NRA snapshot scans.  The goal is to separate three
different phenomena that are easy to conflate:

- semantic compression, where a named authority makes local code derivable
- infrastructure investment, where lines increase to create a future authority
- relocation churn, where code moves but the semantic model is not simpler

Corpus And Method
-----------------

The corpus is the same 32 merged pull requests classified in
:doc:`openhcs_merged_pr_case_studies`.  For this pass, each PR was inspected
through GitHub file statistics and its merge commit when available.  The most
architecturally significant commits were also scanned with NRA on bounded
subsystems rather than with one full-tree scan.

The bounded scan policy is intentional.  A full current ``openhcs`` package
scan exceeded 100 seconds before producing output.  That is not acceptable for
routine repository archaeology.  Subsystem scans were therefore used as the
repeatable unit:

- ``openhcs/config_framework``
- ``openhcs/core``
- ``openhcs/io``
- ``openhcs/processing/materialization``
- ``openhcs/pyqt_gui/widgets``
- ``openhcs/pyqt_gui/services``

This is itself an advisor lesson: repository evolution analysis needs a
partitioned scan model with cached AST products and subsystem scheduling.  A
single monolithic scan hides both architectural signal and performance cost.

Largest Diff Events
-------------------

The following PRs dominate the codebase evolution by additions plus deletions.
Net deletion is not automatically good, and net addition is not automatically
bad; the nominal question is whether declarations are replaced by derivations.

.. list-table:: Highest-volume merged PRs
   :header-rows: 1

   * - PR
     - Diff shape
     - Main touched authority
     - Nominal reading
   * - #58
     - ``22245+ / 54828-``; net ``-32583``
     - ``ObjectState`` extraction from ``ParameterFormManager``
     - Real semantic compression.  View-owned state and duplicated framework
       machinery were deleted in favor of an explicit state authority.
   * - #44
     - ``22609+ / 7182-``; net ``+15427``
     - ABC/service UI architecture
     - Infrastructure investment.  The PR creates nominal UI boundaries, but
       the diff is growth-heavy and must be judged by later deletion or
       detector reduction.
   * - #51
     - ``20151+ / 2510-``; net ``+17641``
     - GUI performance and cross-window synchronization
     - Infrastructure and feature growth.  It creates performance authorities,
       but the raw diff does not prove compression.
   * - #49
     - ``17+ / 19555-``; net ``-19538``
     - Repository artifact boundary
     - Hygiene compression.  Important for repo signal, but not a reusable
       runtime abstraction.
   * - #14
     - ``12454+ / 3332-``; net ``+9122``
     - Napari streaming through materialization-aware filtering
     - Feature/infrastructure bridge.  It exposes staged transport semantics
       but is not primarily a deletion event.
   * - #23
     - ``12031+ / 1561-``; net ``+10470``
     - OMERO virtual backend and ZMQ execution
     - Backend boundary creation.  The value is nominal isolation of remote IO,
       not immediate LOC collapse.
   * - #20
     - ``5937+ / 6833-``; net ``-896``
     - Generic configuration framework
     - Mixed but positive compression.  A new framework deletes specific lazy
       resolver and typed-widget machinery.
   * - #9
     - ``7000+ / 5163-``; net ``+1837``
     - PyQt parameter form system
     - Consolidation with remaining growth.  It stabilizes UI semantics but is
       not yet the final compressed normal form.
   * - #45
     - ``9100+ / 521-``; net ``+8579``
     - Lazy auto-discovery registry framework
     - Registry infrastructure.  The PR creates a derivation mechanism, but its
       rent must be measured against future deletion.
   * - #38
     - ``2359+ / 5256-``; net ``-2897``
     - Enum-driven memory conversion
     - Strong local compression.  Closed memory-type axes replace conversion
       matrix boilerplate.

Compression Events
------------------

PR #58 is the strongest corpus example of semantic compression.  The largest
deleted files are not random cleanup:

- ``openhcs/config_framework/lazy_factory.py``: ``0+ / 1309-``
- ``openhcs/config_framework/context_manager.py``: ``0+ / 1177-``
- ``openhcs/core/auto_register_meta.py``: ``0+ / 719-``
- ``openhcs/core/memory/framework_config.py``: ``0+ / 470-``

The important feature is not only deletion volume.  The PR extracted the model
from the parameter form manager into ``ObjectState``.  That converts
GUI-local coordination into an explicit state object.  In NRA language, the
view stopped being the semantic owner of state history, serialization identity,
and cross-window synchronization.

PR #69 is smaller but cleaner as a materialization case:

- ``openhcs/processing/materialization/core.py``: ``432+ / 886-``
- analysis writers such as cell counting and SKAN modules delete repeated
  output logic
- new ``options.py``, ``presets.py``, ``constants.py``, and ``utils.py`` make
  writer selection and output options explicit

This is the pattern NRA should prefer: concrete backend functions keep the
domain-specific computation, while materialization ownership moves into a
writer/options authority.

PR #38 is the best closed-axis example:

- ``openhcs/core/memory/conversion_functions.py``: ``0+ / 1566-``
- ``openhcs/core/memory/wrapper.py``: ``0+ / 401-``
- ``openhcs/core/memory/decorators.py``: ``221+ / 1792-``

The deleted material is exactly the kind of matrix boilerplate a detector
should flag.  The replacement axis is an enum-driven conversion model rather
than many pairwise conversion declarations.

A bounded scan of the parent of PR #38 now finds this shape directly in
``openhcs/core/memory/conversion_functions.py`` as a
``closed_axis_conversion_matrix`` finding over functions such as
``_numpy_to_cupy`` and ``_pyclesperanto_to_numpy``.

Infrastructure Events
---------------------

PR #44 and PR #45 are important because they explain why LOC alone is not a
sufficient signal.

PR #44 deletes large widget bodies:

- ``parameter_form_manager.py``: ``706+ / 3341-``
- ``plate_manager.py``: ``464+ / 2013-``
- ``cross_window_preview_mixin.py``: ``61+ / 472-``

But it also adds new ABC/service infrastructure:

- ``abstract_manager_widget.py``: ``1293+ / 0-``
- ``widget_creation_config.py``: ``516+ / 0-``
- service and architecture documentation

That is a legitimate nominal move only if the new ABC owns real template
methods and lets implementation classes shrink into hooks.  A future detector
should therefore distinguish "ABC as semantic compressor" from "ABC as class
overexpansion."

PR #45 adds the lazy registry framework.  Its immediate diff is growth-heavy,
but it introduces a metaprogramming authority that can make later manual
registration derivable.  NRA should judge this pattern with a rent condition:
there must be a repeated discovery, registration, or cache lifecycle that the
registry removes or prevents.

Detector Lessons To Carry Forward
---------------------------------

The diff evolution suggests these generic advisor improvements:

- Add a rent-aware infrastructure detector.  A new ABC, registry, or service is
  justified only when it deletes repeated lifecycle mechanics or makes future
  declarations derivable.  The infrastructure detector now applies this
  single-consumer fanout rule to generic ABC/base/mixin/registry/service
  surfaces, not only to effect-step modules.
- Add a snapshot trend report.  A refactor should be judged against bounded
  finding deltas in the touched subsystems, not only against one post-change
  scan.
- Add a closed-axis matrix detector for enum-driven conversions.  PR #38 shows
  the canonical shape: many pairwise conversions are a product of two closed
  axes and should collapse into one algebraic dispatcher.  NRA now detects
  conversion-domain modules that spell source/target pairs as many
  ``*_to_*``/``*_from_*`` functions.
- Add an option-record quotient detector.  PR #69's materialization options
  are semantically good, but many small frozen records may be derivable from a
  format catalog when they contain only fields and defaults.  NRA now groups
  field-only ``*Options``/``*Config``/``*Settings`` records and recommends one
  typed schema catalog when the family is large enough to pay rent.
- Add a scan partitioner.  Full-tree scan latency over 100 seconds is a
  usability failure; subsystem-level scans give useful signal in seconds.
