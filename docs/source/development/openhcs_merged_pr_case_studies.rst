OpenHCS Merged PR Case Studies
==============================

This document classifies the merged OpenHCS pull-request descriptions available
from ``OpenHCSDev/openhcs`` as of 2026-05-06.  The purpose is not to mirror
release notes.  It is to preserve architectural examples that sharpen the
advisor's detector vocabulary.

Source corpus:

- repository: ``OpenHCSDev/openhcs``
- query: merged pull requests
- count: 32 merged PRs
- range: PR #1 through PR #69

Use :doc:`openhcs_high_significance_pr_deep_dives` for proportional treatment
of the largest architectural PRs, :doc:`openhcs_focused_pr_notes` for medium
and small PRs, :doc:`openhcs_diff_evolution_case_studies` for file-level diff
evidence plus bounded NRA snapshot scans, and
:doc:`openhcs_detour_case_studies` for abstractions that were later deleted,
externalized, or abandoned.  This page remains the complete corpus index.

Significance Tiers
------------------

The corpus is intentionally not documented uniformly.  Large architecture PRs
with explicit semantic boundaries receive deep treatment; one-bug PRs receive
shorter notes unless they expose a general detector rule.

.. list-table:: Documentation depth by PR
   :header-rows: 1

   * - Tier
     - PRs
     - Why this depth is appropriate
   * - A
     - #4, #20, #30, #44, #58, #69
     - These introduce new authorities from which many behaviors become
       derivable.
   * - B
     - #9, #12, #23, #38, #45, #51
     - These isolate important closed axes, registries, runtime boundaries, or
       cross-window state mechanisms.
   * - C
     - #1, #14, #17, #35, #36, #39, #41, #43
     - These are subsystem-level changes with clear detector lessons but a
       narrower architectural surface.
   * - D
     - #2, #10, #19, #22, #24, #25, #28, #33, #37, #42, #48, #49
     - These are bug fixes, hygiene changes, or small feature surfaces.  They
       are documented as focused detector examples rather than full deep dives.

Classification Axes
-------------------

Each case study is classified along three axes:

- **Primary nominal lesson**: the architectural identity move the PR made.
- **Advisor pattern family**: the closest NRA pattern family.
- **Detector lesson**: what future detectors should learn to see.

The same PR can support several patterns.  The primary classification is the
most useful one for detector design.

Corpus Matrix
-------------

.. list-table:: Merged OpenHCS PR classification
   :header-rows: 1

   * - PR
     - Title
     - Primary classification
     - Advisor pattern family
   * - #1
     - Feature/input source system
     - Replace decorator sentinel with declared enum strategy
     - Nominal strategy family; authoritative context
   * - #2
     - Critical fail-loud bug fixes
     - Convert silent structural failure into explicit integrity checks
     - Config contracts; authoritative schema
   * - #4
     - Registry, metadata, configuration, materialization systems
     - Multi-subsystem authority extraction
     - Auto-register meta; authoritative schema; staged orchestration
   * - #9
     - PyQt6 parameter form system
     - UI architecture consolidation around lazy config and services
     - Authoritative context; descriptor-derived view
   * - #10
     - Dynamic step parameter editor resize
     - View behavior derived from content geometry
     - Descriptor-derived view
   * - #12
     - Variable component refactor
     - Separate closed semantic axes and normalize grouping identity
     - Dual-axis resolution; nominal boundary
   * - #14
     - Automatic napari streaming
     - Materialization-aware streaming route
     - Staged orchestration; authoritative context
   * - #17
     - Dual-axis configuration resolution
     - Explicit resolution of frame and config axes
     - Dual-axis resolution; authoritative context
   * - #19
     - Menu bar cleanup
     - Remove presentation duplication
     - Local value authority
   * - #20
     - Generic configuration framework
     - Type-driven generic config discovery and lazy resolution
     - Type lineage; authoritative context
   * - #22
     - Non-technical user guide
     - Documentation surface for user mental model
     - Staged orchestration documentation
   * - #23
     - OMERO integration with virtual backend and ZMQ execution
     - Virtual backend boundary and transport staging
     - Nominal interface witness; staged orchestration
   * - #24
     - Group-by selection saving fix
     - Preserve explicit selected enum state
     - Local value authority; config contracts
   * - #25
     - Description field for steps
     - Attach declared provenance to pipeline steps
     - Authoritative context
   * - #28
     - Global window bounds filter
     - Centralize window placement policy
     - Authoritative context; staged orchestration
   * - #30
     - Virtual workspace backend
     - Metadata-owned workspace initialization
     - Nominal interface witness; authoritative schema
   * - #33
     - Windows virtual workspace separators
     - Normalize platform path representation
     - Local value authority
   * - #35
     - GUI performance, metadata, microscope handlers
     - Shared service and metadata authorities
     - Staged orchestration; authoritative schema
   * - #36
     - macOS GUI backend and metadata filtering
     - Platform-specific fail-loud stabilization
     - Config contracts; local value authority
   * - #37
     - Enabled parameter for registered functions
     - Declare a shared execution gate on function records
     - Authoritative schema; descriptor-derived view
   * - #38
     - Enum-driven memory conversion
     - Replace conversion matrix boilerplate with closed enum dispatch
     - Closed-family dispatch; nominal strategy family
   * - #39
     - Fiji streaming race and virtual workspace loading
     - Synchronize staged runtime transport state
     - Staged orchestration; fail-loud contracts
   * - #41
     - Transport mode fix
     - Make transport selection explicit and coherent
     - Config contracts; closed-family dispatch
   * - #42
     - Optional enabled field on configs
     - Shared config gating with UI derivation
     - Descriptor-derived view; authoritative schema
   * - #43
     - Sequential component processing
     - Memory-efficient staged component execution
     - Staged orchestration; authoritative context
   * - #44
     - UI anti-duck-typing refactor
     - Replace structural UI probing with ABC/service boundaries
     - Nominal interface witness; ABC template method
   * - #45
     - Lazy auto-discovery registry framework
     - Registry discovery through typed families
     - Auto-register meta; type lineage
   * - #48
     - Windows path unicode escape fix
     - Centralize escaping policy
     - Local value authority
   * - #49
     - Remove development artifacts
     - Repository hygiene and artifact boundary
     - Authoritative context
   * - #51
     - GUI performance, cross-window sync, analysis fixes
     - Cross-window state authority and performance collapse
     - Authoritative context; descriptor-derived view
   * - #58
     - Extract ObjectState from ParameterFormManager
     - MVC separation with state registry and time-travel
     - Authoritative context; staged orchestration
   * - #69
     - Writer-based materialization architecture
     - Type-safe writer dispatch over output-format options
     - Nominal strategy family; authoritative schema

Batch 1: Early Pipeline Authority And Fail-Loud Conversion
----------------------------------------------------------

PR #1: InputSource Replaces ``chain_breaker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR replaces a decorator marker with an explicit ``InputSource`` enum carried
by ``FunctionStep``.  The old model encoded input routing as hidden function
metadata.  The new model makes routing a declared part of the step contract.

Nominal reading:

- the semantic axis is "where does this step read from?"
- ``PIPELINE_START`` and ``PREVIOUS_STEP`` are closed cases of that axis
- the path planner no longer infers behavior by probing a decorated function

Advisor lesson:

- Flag decorators that are only carrying closed-family routing identity.
- Prefer a typed enum or strategy family when downstream planning needs the
  value as part of a contract.
- Treat "removed introspection logic and replaced it with a declared field" as
  a canonical authoritative-context win.

PR #2: Fail-Loud Bug Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes zarr path portability, pattern-generation failures, and GUI help
constructor ordering.  The important theme is not the individual bugs; it is the
conversion of warnings and silent fallbacks into errors with actionable context.

Nominal reading:

- a warning that lets execution continue is not an integrity witness
- no-pattern and no-well states are real semantic states, not incidental empty
  collections
- GUI constructor roles must be named by the correct parameter boundary

Advisor lesson:

- Flag warning-only paths around required pipeline structure.
- Flag positional UI constructor call sites when argument roles have different
  semantic families.
- Treat portability bugs as authority leaks when serialized identifiers contain
  machine-local context that should be projected away.

PR #4: Registry, Metadata, Configuration, And Materialization Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the first large OpenHCS architecture consolidation.  It introduces
shared registry machinery, a dedicated metadata generator, atomic metadata
writing, generic lazy configuration, and materialization planning.

Nominal reading:

- function discovery moved toward one registry authority
- OpenHCS metadata became a typed record instead of ad hoc dictionary assembly
- metadata writes became a serialized authority with locking and migration
- lazy configuration became a generic dataclass mechanism instead of a
  pipeline-specific special case
- pipeline compilation was moved out of orchestration

Advisor lesson:

- Flag duplicated registry implementations when they differ mostly by exclusion
  lists, metadata shape, or discovery roots.
- Flag function classes that assemble large metadata dictionaries inline.
- Flag mixed lazy/concrete configuration logic that is manually repeated across
  config types.
- A multi-axis refactor can be valid when every extracted authority has a clear
  single-writer boundary.

Batch 2: UI State, Lazy Configuration, And Axis Resolution
----------------------------------------------------------

PR #9: PyQt6 Parameter Form System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This PR consolidates parameter forms around service-layer extraction, lazy
dataclass preservation, automatic lazy config generation, and fail-loud
configuration resolution.  It also repairs multiprocessing step identity by
moving from object addresses to stable positional indexes.

Nominal reading:

- parameter editing has a true state model, not just widget-local values
- lazy fields require explicit inheritance semantics; ``None`` is a declared
  state, not absence
- subprocess step identity must be serializable and stable across address spaces

Advisor lesson:

- Flag UI code that keeps domain state in widget trees.
- Flag memory-address identity used across serialization boundaries.
- Flag manually named lazy config classes when field annotations can derive the
  lazy family.
- Treat ``None`` semantics as a nominal state when it means inheritance.

PR #10: Dynamic Step Parameter Editor Resize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR makes the step parameter editor resize according to content.  This is a
small UI case, but it represents a general descriptor-derived-view pattern:
layout should derive from the contained parameter surface rather than from a
separate fixed declaration.

Advisor lesson:

- Flag fixed-size UI policy when the authoritative size can be derived from
  parameter content and constraints.
- Small PRs can still be case studies when they collapse a presentation copy of
  model structure.

PR #12: Variable Component Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This PR separates ``AllComponents`` from ``VariableComponents``, aligns
``GroupBy`` semantics, standardizes a multiprocessing axis, and hardens
migration.  The core move is disentangling closed semantic axes that were
previously conflated by naming and migration behavior.

Nominal reading:

- "all components" and "variable components" are distinct type-level meanings
- grouping identity belongs to a closed enum family
- multiprocessing configuration is an execution axis that must be named once

Advisor lesson:

- Flag names that encode two axes in one object.
- Flag migrations that infer old state through ambiguous component labels.
- Detect enum alignment work as a sign that a nominal boundary was missing.

PR #14: Automatic Napari Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR adds automatic napari streaming with materialization-aware filtering.  It
connects streaming to materialization decisions so the viewer sees the correct
subset without duplicating selection logic.

Nominal reading:

- streaming is a staged transport behavior, not a widget side effect
- materialization filtering is the authority for what can be streamed
- napari is a backend identity with its own execution contract

Advisor lesson:

- Flag viewer integrations that duplicate materialization filters.
- Treat viewer-specific streaming as a backend strategy when transport behavior
  varies by viewer.

PR #17: Dual-Axis Configuration Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR implements dual-axis resolution with frame injection.  The important
architectural move is making the two axes explicit rather than searching ambient
state with partial heuristics.

Nominal reading:

- the resolution target depends on both field/config identity and caller frame
- frame injection is an authority channel and must be explicit
- local test dataclasses require the same resolution model as production types

Advisor lesson:

- Flag config resolution code that climbs frames or type graphs without naming
  the resolution axes.
- Prefer a resolver object or type-indexed registry when both source scope and
  target field influence the result.

PR #19: Menu Bar Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~

The PR is a small presentation cleanup.  The case-study value is that UI command
surfaces should not carry duplicated or stale menu structure.

Advisor lesson:

- Flag repeated command labels and menu entries when they project from one
  command registry.
- Use local value authority for small UI cleanup rather than introducing a large
  new framework.

PR #20: Generic Configuration Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This PR genericizes the configuration framework and fixes lazy config
resolution.  The recurring theme is replacing hardcoded config knowledge with
type inspection, generic dataclass traversal, and a coherent lazy resolution
contract.

Nominal reading:

- dataclass type identity is the config family handle
- field paths are derivable from annotations
- lazy wrappers should be generated from base config identity, not listed by
  hand

Advisor lesson:

- Flag hardcoded config class name maps when dataclass annotations provide the
  same information.
- Flag duplicate lazy wrapper declarations.
- Treat generic dataclass discovery as a type-lineage refactor.

PR #22: Non-Technical User Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR introduces an incomplete user guide.  Even incomplete documentation is a
case study when it captures a missing user-facing semantic model.

Advisor lesson:

- Documentation PRs can expose missing domain vocabulary.
- If the code requires a user guide to explain a concept, detectors should look
  for the corresponding declared identity in code.

Batch 3: Virtual Backends, Workspace Identity, And Platform Boundaries
----------------------------------------------------------------------

PR #23: OMERO Integration With Virtual Backend And ZMQ Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR adds an OMERO integration using a virtual backend and ZMQ execution.  It
introduces a backend boundary for remote image state and stages execution across
transport.

Nominal reading:

- OMERO is not just a path source; it is a backend identity
- ZMQ execution is a transport stage with its own failure modes
- virtual backend objects mediate between external state and OpenHCS workspace
  contracts

Advisor lesson:

- Flag remote integrations that masquerade as local file paths.
- Require nominal backend witnesses for external systems that own state,
  credentials, or transport semantics.

PR #24: Group-By Selection Saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes group-by selections not saving.  Architecturally, the selected
grouping value is an explicit config state and must be preserved as such.

Advisor lesson:

- Flag UI save paths that serialize resolved values but drop explicitly selected
  enum values.
- Distinguish "inherit group behavior" from "explicit no grouping".

PR #25: Description Field On ``AbstractStep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR adds a description field to pipeline steps.  This is provenance work:
step objects become able to carry human-readable intent beside executable
function identity.

Advisor lesson:

- Flag external side tables for step descriptions when the step contract can
  own them directly.
- Documentation-bearing fields should live on the semantic object they describe.

PR #28: Global Window Bounds Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR centralizes a global window bounds filter so windows do not open off
screen.  The window placement policy becomes application-level authority rather
than being reimplemented by each window.

Advisor lesson:

- Flag per-window geometry clamps when an application-global placement policy
  exists or should exist.
- Window geometry is a staged UI service, not widget-local business logic.

PR #30: Virtual Workspace Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This large PR implements metadata-based workspace initialization through a
virtual backend.  It treats workspace construction as a metadata authority
problem rather than a direct filesystem walk.

Nominal reading:

- virtual workspace identity is distinct from disk workspace identity
- metadata is the source of truth for workspace initialization
- backend initialization is a staged protocol with validation

Advisor lesson:

- Flag workspace initialization code that branches on path strings instead of
  backend identity.
- Require one metadata schema authority when workspace state is reconstructed
  from stored facts.
- Detect virtual-backend introduction as a nominal interface witness.

PR #33: Windows Virtual Workspace Path Separators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes path separators for Windows virtual workspaces.  The lesson is that
platform normalization belongs in one path utility, not at call sites.

Advisor lesson:

- Flag repeated ``replace("\\\\", "/")``-style path normalization.
- Treat path separator policy as a local value authority.

PR #35: GUI Performance, Metadata, And Microscope Handler Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR combines GUI performance work, metadata refactoring, and microscope
handler improvements.  The common thread is moving repeated runtime decisions
into service boundaries and typed metadata authorities.

Advisor lesson:

- Flag GUI refresh paths that recompute derived metadata per widget.
- Flag microscope handler conditionals when device/backend identity can own the
  behavior.
- Prefer a service boundary when performance bugs come from repeated local
  recomputation.

PR #36: macOS GUI Backend And Metadata Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes macOS GUI backend issues and metadata file filtering.  It is a
platform-boundary case: OS-specific backend behavior and metadata selection must
be explicit, fail-loud, and centralized.

Advisor lesson:

- Flag platform checks scattered across GUI backend code.
- Flag metadata filters duplicated between IO and UI layers.

Batch 4: Execution Gates, Conversion Families, And Runtime Synchronization
--------------------------------------------------------------------------

PR #37: Enabled Parameter On Registered Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR adds an ``enabled`` parameter to all registered functions.  The execution
gate becomes a shared field on the function registry surface rather than an
ad hoc filter elsewhere.

Advisor lesson:

- Flag repeated "enabled" checks outside the registered function record.
- Shared execution gates belong to the authoritative schema for the executable
  family.

PR #38: Enum-Driven Memory Type Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR reports a 93 percent code reduction by replacing conversion boilerplate
with enum-driven memory type conversion.  It is a clean closed-family dispatch
case.

Nominal reading:

- memory type is a closed enum axis
- conversion behavior belongs to an enum-keyed converter family
- all pairwise conversion code is derivable from that family

Advisor lesson:

- Flag rectangular conversion matrices when the axes are closed enums.
- Look for symmetric boilerplate of the form ``A_to_B``, ``B_to_A``, and
  repeated backend method families.
- Prefer enum-keyed O(1) dispatch or generated converter classes.

PR #39: Fiji Streaming Race Conditions And Virtual Workspace Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes Fiji streaming races and virtual workspace image loading.  It
documents a staged runtime state problem: transport, viewer readiness, and
workspace state must synchronize through explicit boundaries.

Advisor lesson:

- Flag viewer streaming code that assumes backend readiness without an explicit
  state transition.
- Runtime race fixes often indicate a missing staged orchestration object.

PR #41: Transport Mode Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR fixes transport mode behavior.  Transport mode is a closed execution
axis; it should be selected coherently, not inferred by scattered conditionals.

Advisor lesson:

- Flag transport string conditionals repeated across execution sites.
- Promote transport mode to a declared config or strategy key.

PR #42: Optional Enabled Field On Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR adds an optional enabled field to configs with UI styling and performance
optimizations.  It extends the execution-gate idea from registered functions to
configuration objects and derives UI presentation from that state.

Advisor lesson:

- Flag separate UI disabled styling when it mirrors a config field.
- Treat optional config gates as descriptor-derived views over one authoritative
  config field.

PR #43: Sequential Component Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR introduces memory-efficient sequential component processing for large
datasets.  The processing mode becomes an explicit stage strategy rather than an
implicit consequence of memory pressure.

Advisor lesson:

- Flag loops that encode staged processing policy directly in backend logic.
- Expose memory-sensitive execution as a strategy or pipeline stage.

Batch 5: Anti-Duck-Typing, Registry Discovery, And Repository Hygiene
---------------------------------------------------------------------

PR #44: UI Anti-Duck-Typing Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR explicitly refactors UI code away from duck typing into ABC-based
architecture and service-layer extraction.  It is one of the clearest OpenHCS
matches for the advisor's nominal identity theory.

Nominal reading:

- UI components that share structure are not necessarily the same semantic role
- services own behavior that widgets should not reimplement
- ABCs provide fail-loud contracts and enumerability for UI families

Advisor lesson:

- Flag UI branches that probe for methods or attributes instead of depending on
  a nominal interface.
- Flag widget classes that combine view rendering with service orchestration.
- Prefer ABC template methods when widgets share lifecycle algorithms.

PR #45: Lazy Auto-Discovery Registry Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR implements a lazy auto-discovery registry framework.  It moves registry
population away from manual lists and toward discoverable typed families.

Advisor lesson:

- Flag manual registries whose entries can be derived from importable class
  families.
- Require clear discovery roots and cache invalidation boundaries.
- Auto-discovery is safe when the nominal family and inclusion rules are
  explicit.

PR #48: Windows Path Unicode Escape Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR centralizes Windows path unicode escaping in a utility function.

Advisor lesson:

- Flag repeated platform escaping snippets.
- A path escaping utility should be the single writer for representation policy.

PR #49: Remove Development Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR removes development artifacts from the repository.  The architectural
lesson is boundary hygiene: generated, local, and transient artifacts should not
share the same authority surface as source code.

Advisor lesson:

- Detectors should distinguish production source from generated and temporary
  artifacts.
- Repository hygiene fixes can justify generated-file exclusion policies in
  payoff accounting.

Batch 6: Cross-Window State, MVC Separation, And Materialization Writers
------------------------------------------------------------------------

PR #51: GUI Performance, Cross-Window Synchronization, Analysis Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR improves GUI performance, cross-window synchronization, and analysis
pipeline behavior.  Its main case-study value is the shift from widget-local
refresh behavior to shared state and invalidation authority.

Advisor lesson:

- Flag per-window caches that do not subscribe to a shared state authority.
- Flag refresh work repeated across widgets when one invalidation graph can
  derive the affected views.
- Performance PRs should be examined for structural duplication, not only
  micro-optimizations.

PR #58: ObjectState Extraction From ParameterFormManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This PR extracts the model from ``ParameterFormManager`` into ``ObjectState`` and
``ObjectStateRegistry``, adds DAG time travel, and reduces the form manager to a
view layer.  It is the strongest case study for semantic compression through
state authority extraction.

Nominal reading:

- model state is a first-class object, not a byproduct of form widgets
- live and saved state are different semantic timelines
- snapshots and branches are nominal history objects
- flash/dirty UI behavior derives from state transitions

Advisor lesson:

- Flag manager classes that mix model storage, view rendering, and controller
  orchestration.
- Look for "before large class, after model object plus view object" as an MVC
  extraction detector family.
- Time-travel and dirty tracking require an authoritative state graph, not
  distributed widget fields.

PR #69: Writer-Based Materialization Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PR replaces handler-registered string materializers with typed writer
options and format writer functions.  It shifts the abstraction boundary from
handler names to output format identities.

Nominal reading:

- output format is a typed strategy axis
- writer options are immutable configuration records
- writers are pure serialization functions selected by options type
- presets provide ergonomic views over the same typed contract

Advisor lesson:

- Flag string-keyed handlers when a typed options dataclass can select behavior.
- Flag materialization specs whose option payload does not prove compatibility
  with the selected writer.
- Prefer metaprogrammed writer registries when adding a new format is
  structurally repetitive.

Cross-Case Detector Requirements
--------------------------------

The PR corpus suggests detector families that are worth keeping or deepening:

- **Decorator sentinel replacement**: detect decorators used only as semantic
  routing markers and recommend declared enum or strategy fields.
- **Silent failure hardening**: detect warnings or empty-result fallbacks around
  required pipeline structure.
- **Registry duplication**: detect sibling registries with the same discovery,
  exclusion, caching, or metadata mechanics.
- **Dataclass-derived configuration**: detect hardcoded lazy config maps and
  derive field paths from dataclass annotations.
- **Dual-axis resolution**: detect resolver functions whose behavior depends on
  both scope and target type but only one axis is explicit.
- **Virtual backend identity**: detect path-based handling of remote or virtual
  systems that should have backend witnesses.
- **Platform normalization authority**: detect repeated path separator and
  escaping logic.
- **Manager overexpansion**: detect classes that combine model, view, controller,
  cache, and synchronization roles.
- **Typed writer dispatch**: detect string handler registries where options
  types can select pure writer functions.

What The Corpus Teaches About Overengineering
---------------------------------------------

The OpenHCS PRs are useful because they separate good abstraction from merely
larger abstraction.

Good abstractions in this corpus have these properties:

- they create a single writer for state or metadata
- they make a hidden semantic axis explicit
- they replace structural probing with nominal contracts
- they improve serialization or multiprocessing behavior
- they delete or centralize repeated logic across many call sites
- they give UI views a model authority to observe

Weak abstractions would fail one of those tests.  Adding a class does not pay
rent unless it owns a real semantic boundary.  The strongest cases here are PR
#4, #20, #44, #58, and #69 because their new objects become authorities from
which other behavior is derived.
