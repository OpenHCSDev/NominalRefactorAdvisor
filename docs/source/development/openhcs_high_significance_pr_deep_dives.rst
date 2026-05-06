OpenHCS High-Significance PR Deep Dives
=======================================

This page gives proportional treatment to the OpenHCS merged PRs whose
descriptions contain enough architectural content to function as serious
Nominal Refactor Advisor case studies.

The triage rule used here is deliberately conservative.  A PR is high
significance when it satisfies at least two of these conditions:

- it moves a semantic authority into a named object, registry, ABC, enum, or
  typed record
- it removes a structural probe, string dispatch surface, or hidden sentinel
- it collapses a repeated algorithm or mapping into a reusable mechanism
- it changes serialization, multiprocessing, backend, or cross-window state
  semantics
- it reports measurable deletion, performance improvement, or reduced
  recomputation caused by the new abstraction

The sections below use the same structure:

- **Original pressure**: what architectural stress the PR description reports.
- **False normal form**: the tempting but incomplete shape.
- **Nominal move**: the semantic object introduced or clarified.
- **Compression mechanism**: what becomes derivable afterward.
- **Detector implication**: what NRA should detect automatically.
- **Overfit guard**: how to avoid recommending an abstraction that does not pay
  rent.

Tier-A PRs
----------

The strongest case studies are PR #4, #20, #30, #44, #58, and #69.  They create
new semantic authorities from which many local behaviors become derivable.

Tier-B PRs
----------

PR #9, #12, #23, #38, #45, and #51 are also high-signal.  They either prepare a
later Tier-A refactor or isolate an important closed axis.

PR #4: Registry, Metadata, Configuration, And Materialization
-------------------------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR description reports several forms of structural duplication:

- separate library registries with similar discovery and exclusion mechanics
- ``FunctionStep`` owning OpenHCS metadata construction inline
- metadata dictionaries assembled without a declared typed record
- concurrent metadata writes without a single atomic writer boundary
- lazy configuration behavior tied to pipeline-specific code
- path planning and materialization behavior repeated around step execution

The key observation is that these were not independent bugs.  They were symptoms
of authority being distributed across places that only had partial views of the
same facts.

False Normal Form
~~~~~~~~~~~~~~~~~

A weaker refactor would have introduced local helpers:

- one helper for function registry filtering
- one helper for metadata dictionaries
- one helper for materialized paths
- one helper for lazy defaults

That would reduce some lines but would preserve the wrong ownership model.  The
system would still have several places writing equivalent facts.

Nominal Move
~~~~~~~~~~~~

The PR creates or strengthens several named authorities:

- ``LibraryRegistryBase`` for library discovery policy
- ``OpenHCSMetadataGenerator`` for metadata construction
- ``OpenHCSMetadata`` as a typed record for metadata shape
- ``AtomicMetadataWriter`` for concurrent metadata updates
- generic lazy dataclass machinery for config inheritance
- compiler/orchestrator separation for pipeline construction

Each one names a semantic boundary that was previously represented by repeated
logic.  This is the difference between helper extraction and nominal
compression.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

Once these authorities exist:

- registry metadata can be cached uniformly
- function metadata becomes a typed projection rather than a hand-built dict
- metadata migration and atomic writes are routed through one writer
- lazy configuration can be derived from dataclass type and field path
- materialization paths can be planned through one collision-aware route

The payoff is not just deletion.  The PR also makes consistency checks possible
because there is one object to inspect.

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- sibling registries with repeated discovery, exclusion, or cache invalidation
  code
- large step classes that construct metadata payloads inline
- repeated dictionary literals whose keys form a stable domain record
- metadata write paths that perform read-modify-write without one writer object
- lazy config code that switches on specific config classes instead of
  inspecting dataclass fields

Overfit Guard
~~~~~~~~~~~~~

The detector should not recommend a registry base just because two registries
exist.  It should require shared lifecycle mechanics: discovery, filtering,
version/cache behavior, ordering, or emitted metadata shape.  Without at least
one shared lifecycle, a shared base would be decorative.

PR #9: PyQt6 Parameter Form System
----------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

This PR addresses a configuration UI whose behavior depended on multiple
unstable views:

- subprocesses used memory-address step identity
- lazy dataclass structures lost ``None`` and user edits
- parameter form creation had inconsistent pathways
- UI utilities and conditional dispatch were duplicated
- defensive programming hid real configuration errors

The high-significance part is the interaction among state preservation,
multiprocessing, and UI editing.  A local widget fix cannot solve this class of
problem.

False Normal Form
~~~~~~~~~~~~~~~~~

The tempting fix is to add more guards to each widget:

- if a field is ``None``, keep it
- if a process cannot find a step object, search again
- if one widget creates parameters differently, special-case that widget

That keeps every widget responsible for global semantics it cannot actually own.

Nominal Move
~~~~~~~~~~~~

The PR points toward these semantic objects:

- positional ``step_index`` as the subprocess-stable step handle
- lazy dataclass preservation methods as the authority for reconstructing
  inherited state
- automatic lazy config generation from dataclass fields
- ``ParameterFormService`` and utility services as UI-independent business
  logic

The most important move is replacing object identity with an explicit
serializable handle.  Memory address is not a valid nominal identity across a
process boundary.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After the nominal handles exist:

- subprocess lookup is integer based and stable
- lazy config reconstruction can be tested once
- parameter extraction can route through one service path
- generated lazy config classes follow field annotations

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- object identity or ``id(...)`` semantics crossing serialization boundaries
- UI widgets that reconstruct domain dataclasses directly
- manually maintained lazy wrapper rosters
- repeated parameter extraction paths that only differ by widget framework

Overfit Guard
~~~~~~~~~~~~~

Not every UI helper deserves a service layer.  The signal is a business rule
that must remain true across widgets, frameworks, or subprocesses.  Pure
presentation helpers can stay local.

PR #12: Variable Components, GroupBy, And Multiprocessing Axis
--------------------------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR separates ``AllComponents`` from ``VariableComponents``, aligns
``GroupBy`` semantics, standardizes a multiprocessing axis, and introduces
metaprogrammed component interfaces.

The description names the problem directly: component names such as well, site,
and channel were hardcoded through parsers, validators, and processors.  Adding
a new component required editing several parallel structures.

False Normal Form
~~~~~~~~~~~~~~~~~

A weaker solution would add another component list or config file.  That only
creates another writable source and leaves parsers and processors as consumers
of string names.

Nominal Move
~~~~~~~~~~~~

The PR uses metaprogramming to generate interfaces from the component axis:

- filename parser interfaces are derived from component declarations
- component processor interfaces are derived from the same family
- grouping behavior is aligned with ``GroupBy`` identity
- multiprocessing behavior is named as its own axis

This is a good example of "less declaration, more derivation."  The declared
object is the component axis; parser and processor surfaces are projections.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

Once the component family is the authority:

- parsers can derive expected component fields
- validators can derive legal component sets
- processors can derive per-component methods
- migration code can reason about old names through the new axis

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- repeated lists of the same domain names across parser, validator, and
  processing modules
- method families whose names are mechanical projections of component names
- migrations that map strings across versions without a typed component axis
- enum and parser surfaces that evolve together but are declared separately

Overfit Guard
~~~~~~~~~~~~~

Metaprogramming pays rent only when there is a stable grammar.  A detector
should require repeated mechanical projections across at least two surfaces,
such as parser plus validator or enum plus processor.  One dynamic method family
alone is not enough.

PR #20: Generic Configuration Framework
---------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

This PR redesigns the configuration framework so it is generic rather than
OpenHCS-specific.  The pressure points are especially instructive:

- context setup needed raw ``None`` values
- compiler serialization needed resolved values
- lazy dataclass resolution occurred at the wrong time
- context hierarchy was duplicated between UI and compiler
- legacy modules encoded application-specific assumptions

This is a semantic timing problem.  The same field value has different meanings
at different phases.

False Normal Form
~~~~~~~~~~~~~~~~~

The false normal form is a single "resolve config" function used everywhere.
That collapses two distinct operations:

- preserve raw inheritance markers
- materialize concrete values for serialization

The PR demonstrates that unifying those operations too early causes bugs.

Nominal Move
~~~~~~~~~~~~

The PR separates the phases:

- framework-level config machinery under ``openhcs/config_framework``
- explicit base config type registration
- raw access through ``object.__getattribute__`` during context construction
- resolved access through ``getattr`` inside active context
- compiler-specific resolution for serialization

The key identity is not just "config."  It is config-at-phase:
construction, UI live resolution, saved baseline, and compiler serialization.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After the phase distinction is named:

- field inheritance can preserve ``None``
- compiler context can resolve values only when pickling requires it
- nested contexts can mirror UI behavior
- the same framework can support other applications

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- functions that call ``getattr`` on lazy objects during context construction
- duplicated context hierarchy setup in UI and compiler code
- app-specific config discovery logic inside a supposedly generic framework
- repeated "raw vs resolved" bug fixes as evidence for a missing phase model

Overfit Guard
~~~~~~~~~~~~~

A generic framework is justified only when the code has at least two consumers
with the same lifecycle but different application types.  If there is only one
config family and no external reuse pressure, a smaller authoritative context
record may be enough.

PR #23: OMERO Virtual Backend And ZMQ Execution
-----------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR adds OMERO integration, a virtual backend, and ZMQ execution.  The body
describes remote state, multiprocessing-safe connections, code-based pipeline
serialization, and dual ZMQ channels.

The pressure is that OMERO cannot be safely treated as "just files."  It owns
remote identity, connection lifecycle, transport behavior, and workspace
projection.

False Normal Form
~~~~~~~~~~~~~~~~~

The false normal form is to make path handlers smarter:

- detect OMERO-looking paths
- special-case connection setup in pipeline code
- pass remote handles through existing local IO assumptions

That hides backend identity behind strings and makes transport failures look
like filesystem failures.

Nominal Move
~~~~~~~~~~~~

The PR introduces:

- ``VirtualBackend`` as the backend witness
- ``OMEROLocalBackend`` for OMERO-specific projection
- ZMQ server/client ABCs for transport
- execution server/client roles
- ``OMEROHandler`` and instance manager components

These are not arbitrary classes.  They separate backend identity, execution
transport, and microscope-specific behavior.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After the backend and transport roles are explicit:

- compiler code can auto-detect backend needs
- pipeline objects can be serialized as code for remote reconstruction
- control and data channels can be reasoned about separately
- UI integrations can target backend roles instead of path patterns

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- remote or virtual resources represented only by path strings
- transport setup mixed into backend IO code
- execution protocols that pass opaque runtime objects across process or network
  boundaries without a serialization authority
- backend conditionals repeated across compiler, IO, and UI modules

Overfit Guard
~~~~~~~~~~~~~

Do not introduce a virtual backend for a local path format that only needs one
normalization function.  The backend witness is justified by remote state,
credentials, connection lifecycle, or non-filesystem workspace projection.

PR #30: Virtual Workspace Backend
---------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

This PR removes physical workspace preparation for microscope formats with
nested folder structures.  Instead of creating symlinks or file copies, it
initializes a workspace from metadata and uses a virtual backend mapping.

The problem is an authority mismatch.  Physical workspace layout was being used
as the source of truth even when microscope metadata already knew the real
mapping.

False Normal Form
~~~~~~~~~~~~~~~~~

A weaker solution would optimize symlink or copy creation, or add more path
rewrites for each microscope.  That improves the old mechanism while retaining
the wrong authority.

Nominal Move
~~~~~~~~~~~~

The PR introduces:

- a virtual workspace backend enum case
- metadata structure for workspace mappings
- ``VirtualWorkspaceBackend``
- microscope handler integration via ``_build_virtual_mapping``
- backend selection during workspace initialization

It also corrects an interface-segregation issue: handlers that do not use the
base initialization flow should not be forced to implement virtual mapping.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

With metadata as authority:

- initialization becomes zero-copy
- flattened processing paths can map to nested microscope paths
- ImageXpress and Opera Phenix can share parameterized flattening logic
- zarr conversion can use the same plate structure instead of separate plate
  folders

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- physical filesystem preparation used only to simulate a logical workspace
- duplicated ``*_virtual`` methods beside non-virtual flattening methods
- abstract methods required by subclasses that do not participate in the
  workflow
- backend selection logic driven by path structure instead of backend identity

Overfit Guard
~~~~~~~~~~~~~

Virtualization pays rent when logical and physical structure differ
systematically.  If the physical layout is already the logical layout, a virtual
backend adds indirection without compression.

PR #38: Enum-Driven Memory Type Conversion
------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR reports a 93 percent code reduction through enum-driven memory type
conversion.  The typical smell is a conversion matrix: many functions or methods
whose names encode source and target memory types.

False Normal Form
~~~~~~~~~~~~~~~~~

The false normal form is to keep adding conversion methods:

- ``numpy_to_cupy``
- ``cupy_to_numpy``
- ``numpy_to_zarr``
- ``zarr_to_numpy``

This treats every pair as unrelated, even though the source and target axes are
closed families.

Nominal Move
~~~~~~~~~~~~

The PR makes memory type the closed enum axis.  Conversion behavior becomes
derivable from enum identity and a converter family rather than from a manually
written rectangular grid.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

Once source and target memory types are closed:

- lookup can be enum-keyed
- method naming becomes mechanical
- unsupported conversions can fail loudly through the same matrix authority
- new memory types require one family extension, not scattered function edits

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- function names matching ``*_to_*`` over a repeated closed vocabulary
- source/target parameter pairs that branch over the same enum values
- manually maintained conversion tables with symmetric entries

Overfit Guard
~~~~~~~~~~~~~

Do not collapse conversions whose algorithms are genuinely unrelated and whose
axes are open-ended.  The detector should require a closed vocabulary and
repeated source/target Cartesian structure.

PR #44: UI Anti-Duck-Typing Refactor
------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR directly names the smell: UI code relied on ``hasattr`` checks,
``getattr`` fallbacks, and attribute-based dispatch tables.  Business logic was
tightly coupled to PyQt6 widgets, and manager widgets duplicated large lifecycle
surfaces.

False Normal Form
~~~~~~~~~~~~~~~~~

The weak fix is to standardize attribute names while keeping duck typing:

- every widget should have ``refresh``
- every widget should maybe have ``get_state``
- every manager should call those names through ``hasattr``

That still lacks enforcement, enumeration, and provenance.

Nominal Move
~~~~~~~~~~~~

The PR introduces:

- ABC-based widget protocols
- framework-agnostic service classes
- ``AbstractManagerWidget`` for shared manager lifecycle
- ``FieldChangeDispatcher`` for centralized field handling
- live context registry for cross-window updates
- parametric widget creation

This is exactly the advisor's nominal interface witness pattern: if callers need
to rely on a capability, the capability needs a declared interface.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After the ABC/service split:

- widget roles are explicit
- shared manager lifecycle can live once
- field-change behavior routes through one dispatcher
- business logic can be tested outside PyQt6
- cross-window preview behavior can be simplified

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- repeated ``hasattr``/``getattr`` probes over the same capability names
- UI managers with parallel methods that differ only by widget role
- framework widgets containing business logic that can be service methods
- callback dispatch tables whose keys are structural attribute names

Overfit Guard
~~~~~~~~~~~~~

ABCs pay rent when consumers require the contract.  A one-off optional method
does not require an interface hierarchy.  The detector should require repeated
capability probes across multiple consumers or classes.

PR #45: Lazy Auto-Discovery Registry Framework
----------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The PR implements lazy auto-discovery for registries.  It addresses the common
failure mode where registries are manually maintained and become a second source
of truth beside the class family.

False Normal Form
~~~~~~~~~~~~~~~~~

The weak fix is to build a better list:

- sort entries
- validate duplicates
- add a helper to append entries

That still requires humans to keep the list in sync with the declared classes.

Nominal Move
~~~~~~~~~~~~

The PR makes the typed family discoverable.  Registry entries are derived from
class identity and discovery roots.  The registry becomes a view over the class
family rather than an independent declaration.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

Once registry population is derived:

- adding a class can be enough to make it discoverable
- import/discovery boundaries become explicit
- cache invalidation can be attached to the discovery mechanism
- duplicate keys can fail at registration time

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- module-level registry lists of classes that share a base type
- import-time registration functions whose arguments repeat class attributes
- lazy discovery code duplicated for several plugin families

Overfit Guard
~~~~~~~~~~~~~

Auto-discovery is risky when import side effects are uncontrolled.  The detector
should recommend it only when there is a clear base type, discovery root, and
stable key extraction rule.

PR #51: GUI Performance, Cross-Window Synchronization, Analysis Fixes
---------------------------------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

This PR combines GUI performance, cross-window synchronization, and pipeline
fixes.  The architecture section describes token-based cache invalidation, live
context resolution, scope-based routing, MRO filtering, debounce, background
threads, async highlighting, and dtype/config propagation.

The common pressure is uncontrolled recomputation.  Many views were resolving
or refreshing overlapping state without one invalidation authority.

False Normal Form
~~~~~~~~~~~~~~~~~

The false normal form is local caching:

- add a cache to each widget
- debounce each callback independently
- thread only the slowest operation

That can improve performance while making coherence worse.

Nominal Move
~~~~~~~~~~~~

The PR introduces or strengthens:

- global token-based cache invalidation
- live context resolver as a pure service
- hierarchical scope identifiers
- type-based inheritance filtering
- reusable preview update patterns
- deterministic orchestrator cleanup
- dtype config inheritance and injection

These are all authority moves.  Performance improves because the invalidation
graph becomes explicit.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

Once scope and invalidation are named:

- views can refresh only affected items
- unrelated sibling configs can be skipped through MRO filtering
- resets can propagate consistently through ``None`` semantics
- registered function parameters can be injected centrally
- cleanup guarantees can live in the orchestrator boundary

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- repeated placeholder refresh logic across widgets
- caches without a shared invalidation token
- full-list refreshes triggered by scoped changes
- config inheritance checks that ignore type lineage
- registered function calls that manually receive the same injected parameters

Overfit Guard
~~~~~~~~~~~~~

Do not introduce a global invalidation system for isolated widgets.  The signal
is cross-window or cross-scope coherence plus measurable repeated recomputation.

PR #58: ObjectState Extraction From ParameterFormManager
--------------------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

This PR extracts model state from ``ParameterFormManager`` into ``ObjectState``
and ``ObjectStateRegistry``.  It also adds DAG time travel, per-field styling,
state callbacks, scoped visual feedback, singleton windows, and separate saved
and live global config contexts.

The PR description reports that ``ParameterFormManager`` dropped from 1209 to
621 lines.  More important than the line count is the semantic split: model
state became a first-class object and the form manager became a view.

False Normal Form
~~~~~~~~~~~~~~~~~

The false normal form is a thinner form manager with helper functions.  Helpers
would still leave the form manager owning:

- stored parameters
- live resolved values
- saved resolved values
- provenance
- dirty state
- flash behavior
- history
- widget rendering

That is not MVC separation; it is a large class with helper files.

Nominal Move
~~~~~~~~~~~~

The PR introduces:

- ``ObjectState`` as the per-object state authority
- ``ObjectStateRegistry`` as the cross-window registry
- immutable snapshot records
- branch/timeline records
- callbacks for resolved, saved, parameter, and dirty changes
- ``WindowManager`` for singleton-per-scope windows
- dual saved/live global config contexts

This is a strong semantic compression case because the new objects own concepts
that were previously implicit in widget behavior.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After extraction:

- widgets observe state instead of storing it
- time travel operates on immutable state snapshots
- dirty markers derive from live vs saved comparison
- flash behavior derives from state transitions
- window uniqueness derives from scope keys
- compiler reads saved config while UI reads live config

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- classes that mix model storage, widget construction, dirty tracking, and
  cross-window synchronization
- repeated live/saved comparisons outside one state object
- UI fields that act as the source of truth for domain state
- history or undo behavior stored as widget-local snapshots
- close/reopen bugs caused by missing state registry ownership

Overfit Guard
~~~~~~~~~~~~~

MVC extraction pays rent when there are multiple views, saved/live distinction,
history, or cross-window synchronization.  A simple form with no shared state
does not need ``ObjectState``.

PR #69: Writer-Based Materialization Architecture
-------------------------------------------------

Original Pressure
~~~~~~~~~~~~~~~~~

The old materialization system used handler-registered string dispatch.  The PR
description identifies four failures:

- analysis backends knew handler names
- runtime registration increased complexity
- options were not type-safe
- every analysis function needed verbose configuration

False Normal Form
~~~~~~~~~~~~~~~~~

The weak fix is a better string registry:

- validate handler names
- define constants for ``"csv"`` and ``"json"``
- add helper functions for common materializers

That still leaves options and handlers loosely coupled.

Nominal Move
~~~~~~~~~~~~

The PR shifts the axis from handler name to output format and options type:

- ``CsvOptions``, ``JsonOptions``, and related dataclasses carry writer config
- ``MaterializationSpec`` is the contract between analysis and framework
- writer functions serialize data from typed options
- presets provide ergonomic derived views
- ObjectState rebuild hooks integrate config serialization

The selected writer is now proved by the options object.  This is type-driven
dispatch rather than string lookup.

Compression Mechanism
~~~~~~~~~~~~~~~~~~~~~

After this refactor:

- IDE/type checker can see available options
- adding a format is a repeatable declaration pattern
- writers are pure functions and easier to test
- presets collapse common multi-output declarations
- analysis functions describe what to output, not how to find a handler

Detector Implication
~~~~~~~~~~~~~~~~~~~~

NRA should detect:

- string handler registries where payload dataclasses already imply the handler
- specs that pair a string key with an options object
- repeated materializer setup in analysis functions
- writer functions that differ only by format/options type

Overfit Guard
~~~~~~~~~~~~~

Typed writer dispatch is appropriate when options are format-specific and the
format family is finite enough to register.  If output behavior is arbitrary
user code, forcing it into a closed writer family can reduce flexibility.
