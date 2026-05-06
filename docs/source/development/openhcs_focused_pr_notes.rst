OpenHCS Focused PR Notes
========================

This page documents the medium and small OpenHCS merged PRs proportionally.
These PRs do not all justify full deep dives, but each contributes at least one
detector lesson.

Tier-C PRs: Subsystem-Level Architectural Moves
-----------------------------------------------

PR #1: InputSource System
~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium-high.  The change is narrow, but the
refactor is exemplary: replace hidden decorator identity with an explicit
closed strategy axis.

Important facts from the PR description:

- ``@chain_breaker`` was removed from three position-generation functions.
- ``InputSource`` introduced ``PREVIOUS_STEP`` and ``PIPELINE_START``.
- path planning stopped performing function introspection and instead read a
  declared step attribute.
- backend consistency for pipeline-start reads stayed centralized.

Detector rule:

- Find decorators whose only durable effect is to attach routing identity to a
  function.
- If downstream planning consumes that identity, recommend a declared enum or
  strategy field on the step object.

Bad recommendation to avoid:

- Do not recommend a new enum for decorators that wrap real behavior.  This
  case is about marker decorators whose semantic payload is pure identity.

PR #14: Napari Streaming With Materialization-Aware Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium.  The PR touches runtime streaming,
materialization filtering, dual-axis config resolution, and UI placeholder
behavior.  It is a bridge PR, not a final normal form.

Important facts from the PR description:

- napari viewer creation is automatic for steps with streaming configs
- process-based viewer execution avoids Qt threading conflicts
- compiler detects lazy napari streaming configs during compilation
- materialization filtering governs what is streamed
- placeholder synchronization depends on context hierarchy and inheritance

Detector rule:

- Flag viewer integrations that perform their own output selection instead of
  reading materialization authority.
- Flag streaming behavior constructed as a widget side effect rather than a
  declared step/runtime config.
- Flag UI placeholder updates that ignore whether a field is user-set.

Bad recommendation to avoid:

- Do not collapse streaming into materialization.  Streaming is a transport
  stage that depends on materialization filtering, not the same authority.

PR #17: Dual-Axis Configuration Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium-high.  This PR names an important
mathematical structure: resolution depends on context hierarchy and inheritance
chain simultaneously.

Important facts from the PR description:

- X-axis: step context, orchestrator context, global context, static defaults
- Y-axis: MRO inheritance traversal with field-specific blocking
- resolution exhausts one context before moving up the hierarchy
- frame injection keeps user APIs clean while isolating complexity
- placeholder consistency depends on the same resolution model

Detector rule:

- Flag resolver code that mixes scope search and MRO search without naming both
  axes.
- Flag class-level override checks when the PR describes field-specific
  blocking.
- Flag placeholder or default systems whose view resolution differs from
  compiler resolution.

Bad recommendation to avoid:

- Do not recommend frame injection as a default pattern.  It is justified only
  when the API must stay clean and the frame boundary is isolated in a service.

PR #35: GUI Performance, Metadata, Microscope Handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium.  The PR combines performance and handler
cleanup.  The strongest lesson is lazy import as derived namespace access.

Important facts from the PR description:

- GUI startup improved from more than eight seconds to less than one second.
- processing backends are lazily imported through ``__getattr__`` and
  ``__all__``.
- backend import paths are derived from module structure.
- storage registries and GPU utilities are initialized lazily.
- metadata and microscope handler improvements share service-boundary themes.

Detector rule:

- Flag top-level imports of heavyweight optional backends when a module export
  list can derive lazy imports.
- Flag import registries that repeat module names already derivable from
  package structure.
- Flag GUI startup paths that initialize execution-only resources.

Bad recommendation to avoid:

- Lazy imports are not automatically better.  Require measurable startup cost,
  optional dependency weight, or backend-specific initialization side effects.

PR #36: macOS GUI Backend And Metadata Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium.  It shows platform authority and metadata
filter authority.

Important facts from the PR description:

- napari ZMQ mode incorrectly forced Linux ``xcb`` on macOS
- macOS ``._*`` resource fork files broke filename parsing
- shared memory names exceeded platform length limits
- Fiji/ImageJ initialization needed macOS event-loop handling

Detector rule:

- Flag hardcoded platform backend values inside runtime launchers.
- Flag metadata parsers that do not route through a shared file-filter policy.
- Flag shared-memory or IPC identifiers built from unbounded strings.

Bad recommendation to avoid:

- Do not hide platform behavior behind generic exception suppression.  The fix
  should name the platform axis and fail loudly for unsupported modes.

PR #39: Fiji Streaming Race Conditions And Virtual Workspace Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium-high.  This PR clarifies runtime handshake
semantics.

Important facts from the PR description:

- PUSH/PULL plus sleep was unreliable for Fiji streaming
- REQ/REP creates acknowledgement before workers release shared memory
- Fiji copies shared memory before acknowledging
- virtual workspace backend registration must be plate-specific and lazy
- image browsing must use the orchestrator's file manager

Detector rule:

- Flag timing sleeps used as synchronization between workers and consumers.
- Flag global registration of backends whose constructors require scoped
  runtime context.
- Flag duplicate metadata image-list methods when an ABC already defines the
  precedence between workspace mapping and image list.

Bad recommendation to avoid:

- Do not replace every queue with REQ/REP.  Require a resource lifetime hazard
  where producer cleanup can invalidate consumer reads.

PR #41: Transport Mode Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium.  This PR names transport as a config axis
and lifts shared streaming fields to a base class.

Important facts from the PR description:

- ``TransportMode`` distinguishes IPC and TCP
- IPC uses Unix domain sockets or Windows named pipes
- TCP supports remote execution but can trigger firewall prompts
- ``host``, ``port``, and ``transport_mode`` move to ``StreamingConfig``
- streaming ports are discovered from ``StreamingConfig`` subclasses
- IPC socket path construction is centralized

Detector rule:

- Flag hardcoded streaming port ranges per viewer type.
- Flag subclass-specific field names like ``napari_port`` and ``fiji_host``
  when callers want polymorphic ``config.port`` and ``config.host``.
- Flag repeated socket path construction.

Bad recommendation to avoid:

- Do not force all streamer-specific settings into the base class.  Only fields
  consumed polymorphically belong there.

PR #43: Sequential Component Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: medium-high.  This PR converts memory pressure into
an explicit execution strategy.

Important facts from the PR description:

- example memory drops from 5,400 images to 27 images per pipeline iteration
- sequential processing is configured at pipeline level
- component combinations are precomputed at compile time
- each component combination flows through the whole pipeline before the next
- intermediate memory can be freed between combinations

Detector rule:

- Flag runtime loops that imply a processing mode but do not expose it as
  pipeline configuration.
- Flag step-level memory controls that actually affect whole-pipeline
  scheduling.
- Flag repeated Cartesian-product generation outside compiler/preplanning.

Bad recommendation to avoid:

- Sequentialization should not be recommended only because a loop exists.  It is
  justified by memory pressure, component axes, and whole-pipeline scheduling.

Tier-D PRs: Focused Bug, UI, And Hygiene Lessons
------------------------------------------------

PR #2: Critical Bug Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused.  The PR is small but useful because it
turns silent warnings into fail-loud errors.

Detector rule:

- Flag warning-only paths around required pattern detection.
- Flag serialized zarr paths that include machine-local parent directories when
  only filenames should be portable.
- Flag GUI constructor calls where positional arguments cross semantic roles.

PR #10: Step Parameter Editor Resize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused UI view correction.

Detector rule:

- Flag fixed widget sizing when size should derive from content layout.
- Flag parallel tab implementations where one tab has a correct layout model
  and another manually diverges.

PR #19: Menu Bar Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: small but valid presentation authority cleanup.

Detector rule:

- Flag nonfunctional menu commands and stale widget references.
- Flag duplicated documentation/help entry points when one direct documentation
  link is the intended authority.

PR #22: Incomplete User Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: documentation-only.

Detector rule:

- Treat incomplete user documentation as evidence of missing domain vocabulary,
  not as a code smell by itself.

PR #24: Group-By Selection Save Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused config-state preservation.

Detector rule:

- Flag UI save paths that reconstruct an object but drop enum-selected state.
- Distinguish explicit ``GroupBy.NONE`` from inherited ``None`` when the domain
  has both meanings.

PR #25: Step Description Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused provenance addition.

Detector rule:

- Flag sidecar documentation maps when the described object can own its
  description field.
- Prefer attaching human intent to the executable step contract.

PR #28: Global Window Bounds Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused UI policy centralization.

Detector rule:

- Flag per-window off-screen clamping logic.
- Recommend one application-level event filter when all subwindows require the
  same placement invariant.

PR #33: Windows Virtual Workspace Path Separators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused platform normalization.

Detector rule:

- Flag repeated path-key normalization and prefer ``Path.as_posix()`` for
  cross-platform virtual workspace mapping keys.
- Flag atomic file writes that use operations with weaker cross-platform
  replacement guarantees.

PR #37: Enabled Parameter On Registered Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused execution-plan authority.

Detector rule:

- Flag runtime checks for ``enabled`` when disabled functions can be removed at
  compile time.
- Flag config toggles passed into function signatures when they are planner
  metadata, not function parameters.

PR #42: Optional Enabled Field On Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused config-view derivation.

Detector rule:

- Flag UI dimming logic duplicated by widget type when it derives from one
  ``enabled`` field.
- Flag reset operations that trigger expensive cross-window recomputation per
  field instead of batching.

PR #48: Windows Path Unicode Escape Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: focused local value authority.

Detector rule:

- Flag subprocess Python-code f-strings that interpolate Windows paths directly.
- Prefer one formatter utility for escaping generated code.

PR #49: Remove Development Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Architectural significance: repository hygiene.

Detector rule:

- Keep generated, temporary, log, and planning artifacts out of production
  source accounting.
- Use this PR as justification for excluding development artifacts from
  advisor payoff calculations unless the task explicitly targets docs or plans.
