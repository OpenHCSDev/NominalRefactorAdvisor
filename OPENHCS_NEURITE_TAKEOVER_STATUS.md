# Blind neurite takeover — 2026-09-20/21

Continuation of `OPENHCS_NEURITE_ANALYSIS_HANDOFF.md`. This is operational and
diagnostic progress, not acceptance of an analysis candidate.

## Verified state

- Read the handoff and `/home/ts/.config/opencode/skills/openhcs-neurite-visual-qa/SKILL.md`.
  Loaded current MCP first-use, UI workflow, image-analysis, viewer-review and
  custom-function guidance.
- Current checkout: `/home/ts/code/projects/openhcs`, version 0.8.5. The embedded
  Codex MCP serves a different checkout; use this checkout's persistent
  `python -m openhcs.mcp.dev_client shell --no-prompt` client.
- UI bridge: `ui-56abbbfc-c916-4317-8b8b-766c2ca5d54e`, PID 1867158, IPC port 7888.
  Viewer: existing Napari IPC 5555. No new UI or viewer was launched.
- Selected source is still `blind-full-A018/P001_openhcs` on the external drive.
  A035 declaration explicitly selects nuclear channel index 0 and calcein index 1.
- A035 snapshot: `44e269a5-94c0-4f64-93d3-a9e95e66900c`. All earlier outputs remain.
- A14 has already been used for tuning. Do not call it an untouched held-out
  field. Audit exposure of the other four wells before claiming held-out scoring.

## Infrastructure failure repaired

`openhcs/mcp/server.py::_await_with_declared_progress` caught an operation's own
`TimeoutError` as a heartbeat timeout. It looped on the already-failed task,
emitting hundreds of thousands of false progress messages and suppressing the
actual error. Replaced exception-based polling with `asyncio.wait` and terminal
`task.result()`. Added six regression cases for immediate/delayed timeout,
ordinary error, and cancellation. No scientific algorithm was changed.

Validation: 314 passed across `tests/unit/agent/test_mcp_server.py` and
`tests/unit/test_mcp_dev_client_persistent_session.py`, four workers, per-test
60-second timeout. Changes are local and uncommitted; inherited dirty changes
elsewhere were preserved.

The surfaced error was a 5-second function-catalog control timeout. Old execution
server PID 1607531 logged repeated cache invalidations because its loaded
scikit-image was 0.25.2 while the installed/prepared environment was 0.26.0.
Its log also proves the previous transient diagnostic registered at 20:50:11.
Verified zero running/queued jobs before refreshing it. Lifecycle graceful
shutdown only killed workers; forced lifecycle shutdown did not terminate it.
Sent SIGTERM to that exact idle PID; its log confirms normal cleanup and stop.
The next MCP function search launched fresh execution PID 1242653 on 7777 and
completed. UI/session/viewer were preserved. Historical transient registration
did not survive the server restart.

Fresh server log:
`/home/ts/.local/share/openhcs/logs/openhcs_zmq_server_port_7777_1789953448100870.log`.

## Visual observations and provenance

MCP snapshots in `/tmp/openhcs-neurite-takeover-20260921/`:

- `20260921T010911535954Z_napari_5555_OpenHCS_Napari_Visualization.png`: raw DAPI,
  p0.5–99.9 = [656, 17158].
- `20260921T010939026257Z_napari_5555_OpenHCS_Napari_Visualization.png`: raw calcein,
  p0.5–99.9 = [145, 53786.36].
- `20260921T010958679469Z_napari_5555_OpenHCS_Napari_Visualization.png`: raw calcein,
  p0.5–97.5 = [145, 2380]. Processes are much more visible at this window.
- `20260921T011404606024Z_napari_5555_OpenHCS_Napari_Visualization.png`: raw DAPI
  with A035 nuclear ROIs. Central compact bright cluster is absent from the
  overlay; an elongated peripheral structure is included. These screenshots
  are orientation evidence, not completed native-pixel biological validation.

The central cluster's interpretation needs care: brightness alone does not
prove every fragment is a valid nucleus or neuron. Do not force its admission
merely to maximize counts. Native-coordinate morphology, calcein colocalization
and intermediate masks remain the decision evidence.

## New diagnostic attempts

Root: `/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/`.

### A036 — failed wrapper ABI, no acceptance

- Source/config retained under `qc/takeover-A036/`.
- Registered `openhcs:nuclear_admission_stage_diagnostics`, persisted by MCP.
- UI validation and compilation passed; execution
  `6df2a3a5-be37-4b71-991e-2ab5def29243` failed with
  `Runtime callable trailing return count ... 3 != 1`.
- Cause: the wrapper returned named image artifacts as trailing values.
  Image artifacts occupy the canonical image return slot, bundled when there
  are multiple; only the labels output belongs in the trailing slot.
- Intended output root was `blind-a018-diagnostic-A036/P001_openhcs_openhcs`.
  Retained evidence is source/config, UI history, and server error logs; do not
  assume a failed execution created scientific output files.
- This is an authoring error in the new diagnostic, not evidence of a
  segmentation failure. A fixed-size return-annotation/contract consistency
  check or synthetic ABI test could catch this case before plate execution.
- The failed custom function remains registered; do not use it. Remove it via
  the owning custom-function management surface once provenance is secured.

### A037 — corrected named-image diagnostic

- Source/config under `qc/takeover-A037/`.
- Function `openhcs:nuclear_admission_image_bundle` returns the existing
  `ImageOutputBundle`, with contexts derived from the exact image declarations,
  followed by accepted labels. No copied image-name lookup.
- Artifacts: `nuclear_local_response`, `nuclear_threshold_candidate`,
  `nuclear_accepted_labels`, each with declared TIFF materialization.
- A14 only; nuclear widths 5–30 um, response 200; pixel size injected from plate
  metadata. Uses the existing background-response and segmentation kernel.
- Planned output `blind-a018-diagnostic-A037/P001_openhcs_openhcs`.
- Registration, reflected contract, UI source validation and compilation passed.
  Execution `da2bc875-c935-4e50-b0cb-876cf302abd6` failed: the named image
  outputs consumed a runtime stack without a declared source-context relation.
  This is another diagnostic authoring-contract failure, not detector evidence.

### A038 — source-plane-aware diagnostic

- Source/config under `qc/takeover-A038/`.
- Function `openhcs:nuclear_source_plane_diagnostics` uses
  `MainFlowStackOutputSpec`, `SelectedPlaneImageOutput`, and the accepted-label
  lineage relation to the response. These existing types declare exactly which
  original channel supplies each diagnostic; no source identity is guessed.
- Settings still match A035. Output target is
  `blind-a018-diagnostic-A038/P001_openhcs_openhcs`.
- Registered and UI-validated, but compilation rejected the labels-to-response
  group-scope relation because that relation must reference an invocation input,
  not a sibling output. Compile request `cf59dbbe-272e-41a5-a702-80476cd4d9b3`.
  No execution was attempted after that rejection.

### A039 — remove invalid sibling group-scope relation

- Source/config under `qc/takeover-A039/`; function
  `openhcs:inspect_nuclear_admission`. Keeps the explicit source-plane image
  projections and uses the existing kernel's ordinary labels output contract,
  without the invalid sibling-output group-scope relation.
- Output target `blind-a018-diagnostic-A039/P001_openhcs_openhcs`.
- Registration and compilation passed; execution
  `7030ca49-8e16-4e4a-b28c-77a28bd63b77` failed because the labels were a one-plane
  stack while the labels contract retained the two-plane input axis.
  The output directory was not created.
- These versions are retained as diagnostic authoring attempts, not biological
  biological parameter sweeps. Nuclear detection settings are unchanged.
- Environment caveat: new execution uses the currently installed dependencies.
  Compare A039 labels to A035 before assuming byte-equivalence across the server
  refresh; an environment refresh is not proof of invariant segmentation.

### A040 — align labels with the existing neurite kernel convention

- Source/config under `qc/takeover-A040/`; registry function
  `openhcs:nuclear_admission_qc`.
- The production neurite function creates an integer label stack matching the
  whole input shape, then populates only the nuclear-channel plane. This
  diagnostic now does the same. Named response/candidate images still use
  explicit source-plane projection. Nuclear parameters remain unchanged.
- Output target `blind-a018-diagnostic-A040/P001_openhcs_openhcs`.
- Registration/reflection, UI source validation and compilation passed.
  Execution `7f24bcf1-513d-4f32-a4eb-045145d9fd2d` failed at source projection:
  `Source metadata channel='1' conflicts with canonical channel='2'.`
- Current UI selection is this failed A040 diagnostic, not a completed new
  neurite analysis. No nuclear/body/outgrowth parameter has been retuned.
- Next technical task: trace the complete traceback in the server log through
  named-image source projection/materialization. Establish which output claims
  channel 2 before altering declarations again. Do not discard provenance,
  suppress the validation, or interpret a partial diagnostic as accepted.
- Stop adding variant functions as a substitute for fixing the projection
  contract. Earlier custom prototypes remain registered and need owner-managed
  retirement after retaining their sources and failure records.

## Continuation control

The current goal is active: correctly set up the blind analysis and strengthen
the discoverability and content of the QA skill/operating guidance. No new
subagents were started during this takeover.

## A041 — diagnostic execution now succeeds

The A040 traceback identified the failure in checkpoint metadata generation,
not the nuclear detector. Both named images selected DAPI, but positional input
alignment assigned the second image a channel-2 storage filename while its
semantic metadata still said channel 1. This happens when a selected output
retains stitched-image contributor provenance.

- Added a failing regression to the existing explicit-payload identity test,
  retaining contributor provenance. Fixed
  `FunctionOutputIdentityAuthority._input_aligned_payload_identity` so explicit
  semantic component values also own the storage filename. Positional alignment
  still resolves unspecified coordinates. No channel-specific branch or weakened
  consistency check was added.
- 138 output/projection tests passed; final combined verification of those and
  QA/context/knowledge tests: **184 passed in 18.85s**, four workers, 60-second
  per-test timeout. Source-identity fix remains uncommitted.
- Verified runtime idle, stopped exact execution PID 1242653 with SIGTERM, and
  allowed the normal catalogue owner to start current-code PID 1501179.
  Existing UI and viewer were preserved.
- Applied `qc/takeover-A041/orchestrator.py` through the revision-checked UI
  document. Only the output destination changed from A040; same custom function,
  same nuclear settings. No new function variant was registered.
- Execution `deea07e5-bcaa-4bd7-9b09-edeecb3c7d82` completed in 8.8 seconds
  (function step 6.002 seconds including finalisation).
- Log: `/home/ts/.local/share/openhcs/logs/openhcs_zmq_server_port_7777_1789957546145439.log`.
- Output: `blind-a018-diagnostic-A041/P001_openhcs_openhcs` under the existing
  external-drive analysis root. Both named checkpoint images now have `w1`:
  `analysis_inputs/A14_s001_w1_z001_t001_nuclear_local_response.tif` and
  `analysis_inputs/A14_s001_w1_z001_t001_nuclear_threshold_candidate.tif`.
  Accepted-label TIFFs exist under `analysis_inputs_results/`.

### Evidence and remaining inspection friction

- MCP inventory resolves both checkpoint images. Viewer payload inspection
  confirms response float32, candidate uint8, shape 2859x2859, well A14,
  channel 1, scale 1.3556. Candidate foreground count is 241,332 pixels; this is
  NOT an accepted-nucleus or cell count.
- Native candidate tile y=[1410,1418), x=[1350,1358) has 44/64 foreground pixels.
  Do not call this the missed-cluster proof yet: coordinates were exploratory,
  not a completed raw/calcein/accepted-label correspondence ledger.
- Raw DAPI snapshot `20260921T023043611305Z_...png` and response snapshot
  `20260921T023155175922Z_...png` in `/tmp/openhcs-neurite-takeover-20260921/`
  show the bright central cluster. Response display p0.5-p99.9=[1,16458].
- Manual image streaming reuses the generic `selected_images` route and replaces
  matching coordinates. It returns a settlement error although payloads arrive.
  Candidate payload has 2.95% foreground, but the p0-p100 [0,1] screenshot
  `20260921T023249341560Z_...png` is entirely white. This rendering is invalid
  evidence; investigate the mounted-image refresh rather than interpreting it.
- Explicitly streaming the saved accepted-label TIFF failed because its filename
  lacks source-plane component metadata. The generated plate inventory currently
  returns only the two checkpoint images. Inspect the typed artifact persistence
  path; do not invent component identities from filenames.
- Next useful managed diagnostic: use the existing step's declared Napari
  streaming to publish each named output with its own producer identity and
  labels with their typed source context. Preserve A041. Do not compare a stale
  A035 ROI against A041 as though it proves current-label correspondence.
- Extra unqualified image TIFF/summary also exist from named-image artifact
  materialisation. Check for colliding materialisation destinations before
  treating those unqualified files as authoritative; qualified checkpoints are
  distinguishable and source-backed.

### Guidance changes

Added raw-first assay interpretation, earliest-failed-stage diagnostics, and
development/validation exposure rules to the existing `ImageQaPrecondition`
declarations, which generate the MCP guide. The packaged `use-openhcs` skill
explicitly routes neurite/segmentation QA to both canonical MCP contexts.
Condensed repeated workflow prose rather than increasing the 16,000-character
guide bound. QA and progressive-context tests pass; skill validation passes.
The skill-creator and Diataxis guidance informed this narrow how-to update.
No local OpenCode skill was overwritten; broader consolidation remains open.

For a fresh dev-client, explicitly configure the task-scoped external-drive
read/write roots through `OPENHCS_AGENT_READ_ROOTS` and
`OPENHCS_AGENT_WRITE_ROOTS` (colon-separated). Otherwise plate-file operations
correctly refuse the external drive. Source edits invalidate the MCP generation;
finish an edit batch, then reconnect before issuing viewer operations.

Scientific status remains incomplete: successful stage instrumentation is not
segmentation acceptance. No biological parameter has changed since A035.

Final current dev-client PTY session: `14937`, MCP PID `1524132`, checkout
`/home/ts/code/projects/openhcs`. Restreaming both original A14 source channels
completed successfully; the old A035 ROI is hidden. Reapply DAPI p0.5-p99.9
before reviewing raw morphology. UI remains on completed A041.

## Next scientific decision

Compare response, candidate and accepted labels against native A14 raw data.
If signal is present in candidate but absent in accepted labels, expose the
watershed/pre-filter labels through the shared kernel owner to distinguish
splitting from short-axis filtering. Do not sweep body/tracing settings first.

## A042 and A043 continuation

A042 completed through managed execution `782e39ad-38b6-4ce8-adcb-57c20d238bab`
with unchanged A035 nuclear settings and explicit per-artifact viewer streams.
Unlike the generic manual route, these streamed masks rendered correctly.
The central cluster is present in the threshold candidate but absent from the
accepted-label image. Native tile y=[1410,1418), x=[1350,1358) has candidate
foreground 44/64 and accepted foreground 0/64. This localizes the loss to
splitting/filtering, but does not establish the biological validity of every
bright spot. Screenshots `20260921T023811230438Z_...` (candidate) and
`20260921T023822780697Z_...` (accepted) are in the takeover screenshot folder.

The generic manual image-route settlement error has a concrete cause: the
`selected_images` route retains old channel 2 entries while a new single-channel
request declares domain [1]. Do not suppress that validation or interpret the
stale native display as the new payload. No route fix was made in this turn.

The shared round-object detector now exposes prefilter labels, measured axes,
and the existing keep mask in `RoundObjectSegmentationStages`. The normal
detector delegates to the same implementation; no segmentation control changed.
New builtin `inspect_metaxpress_round_objects` emits typed prefilter/accepted
label artifacts and an object-keyed width/rejection table. Synthetic correspondence
and existing count/neurite tests: **96 passed in 11.47s**, four workers, 60-second
test timeouts. This diagnostic change remains uncommitted.

A043 configuration is `qc/takeover-A043/orchestrator.py`, same A14 settings,
new output `blind-a018-diagnostic-A043`. Its first validation exposed a process
alignment problem: the old UI PID 1867158 was imported from
`/tmp/opencode/openhcs-compile-perf`, not the intended checkout. Its relaunch
loaded that worktree's version 0.8.6; current diagnostic checkout is 0.8.5.
Do not equate version labels with source alignment or call this the latest release.

Used the managed restart action to capture all three plates and 17 history
snapshots; retained a recovery copy at `qc/takeover-session-20260921` on the
external drive (also `/tmp/openhcs-session-recovery-FvWPae/captured`). Exited
the intermediate UI through its version-matched MCP client, then launched
the intended checkout with a separate restore copy. New UI PID 1600678 has
cwd `/home/ts/code/projects/openhcs`; A042 source hash and 17 snapshots restored.
Existing Napari 5555 and all analysis outputs were preserved.

A043 validated and applied through the UI, selected plate reinitialized, and
compile `727868d6-7d92-4e51-87a6-762b20288d12` completed. Execution is next.
Current persistent MCP PTY is 18111 (PID 1566761). UI launch PTY is 29758.
UI log: `openhcs_unified_20260920_225047.log`. Execution-server restart is
recorded in `openhcs_zmq_server_port_7777_1789959052789027.log`.

### A043 runtime correction and workspace containment

A043 execution `6e22c440-1e5d-4c65-8d51-605d929d3c68` failed before output
creation: diagnostic trailing return count 3 versus contract 2. A first label
declaration participates in main flow; placing it first incorrectly consumed
the unchanged image return. Corrected the diagnostic to declare/return the
measurement artifact first, followed by both labels (the same convention as
existing counting functions). Extended its synthetic test through the real
`RuntimeReturnedOutputMatcher`. Count, neurite and output-matcher suites now
report **109 passed in 11.76s**. No detector settings or kernel changed.

Restarted the UI through its managed session action; verified idle execution
server 1601307 and sent SIGTERM to reload current diagnostic code. New UI PID
1618930 is on the intended checkout. Fresh persistent MCP PTY **4666** replaces
18111. Function catalogue reflects the corrected return contract. New config
`qc/takeover-A044/orchestrator.py` is ready with a separate A044 output root;
it has NOT yet been applied/compiled/run. Current UI declaration remains A043.

At the user's request, moved all seven matching windows (main UI, startup/update/
restart dialogs and three OpenHCS Napari viewers) to the existing **OpenHCS**
AwesomeWM tag. Added `/home/ts/.config/awesome/openhcs_workspace.lua`, loaded
at the end of `rc.lua`, and activated its rules live. It scopes matching to
OpenHCS classes/titled viewers and their transient dialogs, clears sticky state,
and rejects background activation while the OpenHCS tag is not selected. The
user's current tag remained selected; a real activation-request check confirmed
focus unchanged. `luac -p` and `awesome -k` pass. Unrelated browser/terminal
windows were not moved. This is OS workspace management, not image-analysis
automation or a substitute for MCP viewer control.

After the user flagged lingering dialogs, used MCP `windows`/`widget-tree`
and `invoke-widget-action` to acknowledge **OpenHCS Restart Complete** and
cancel **OpenHCS Update Available**. No installation was started. Rechecked
`windows`: no visible extra top-level dialogs remain. Make this an explicit
post-restart cleanup step; do not leave completion/update prompts accumulating.

A044 has now completed: execution `3f6b2753-103f-4aaf-938f-cc530f04777e`,
snapshot `a3cff166-21b6-4804-a421-4ec1157ca063` (main@18). Output files are
under `blind-a018-diagnostic-A044/P001_openhcs_openhcs/results`.
Live measurement surface reports 17,189 pre-filter rows (only 50 previewed).
MCP exact ROI payload identifies the central candidate as label **8044**,
area **956 px**, bbox **[1400,1342,1436,1390]**, centroid
**[1419.7092050209205,1364.1642259414225]**. Its width row is not yet retrieved.
Result-file inventory incorrectly searches the default results directory and
misses this explicitly configured results path; no workaround or fix yet.

Raw A044 DAPI snapshot `20260921T030032431240Z_...png` and aggressive calcein
snapshot `20260921T030358794015Z_...png` show the same field. The central bright
cluster has compact calcein-positive lobes but no clear rooted processes at this
scale, unlike surrounding star-shaped somata with obvious processes. Its
biological interpretation remains **ambiguous**, not a confirmed set of missed
neurons. Do not tune solely to admit it. Inspect an enlarged moderate-window
view and individually plausible surrounding cells before choosing an intervention.
Current viewer is raw calcein, p0.5-p99.9=[145,53786.36], all ROIs hidden.

### Native crop C1 and baseline replay A045

Enlarged the ambiguous central cluster at source (1419,1364), zoom 3. Raw
DAPI shows compact/lobed spots; calcein at both moderate and aggressive windows
does not establish broad somata with rooted processes. This is not a confirmed
missed-neuron positive control. Earlier A035 rejection language about that
cluster must not be treated as ground truth.

Nearby crop C1 is centered on source (1460,1620), zoom 2. Exact MCP raw tile
y[1465,1545), x[1675,1755) contains a compact DAPI anchor near (1476,1740)
and a broader calcein-positive soma with thin processes. A044 accepted nucleus
657 has centroid (1476.317,1740.667); A035 body 382 has centroid
(1474.204,1739.199), area 668 px. This clear cell is not a missing nucleus/body.
The next diagnostic question is process coverage, not higher nuclear sensitivity.

Manual streaming of A035 body and graph archives yields valid native payloads
and matching 1.3556 um scale, but no visible ROIs on raw channel 2. The manual
routes have empty stack axes and singleton channel domain, unlike the raw
two-channel stack. Suspected cross-route slice mismatch is not yet proven or
fixed. Do not infer absent analysis from this overlay attempt.

Restored historical A035 snapshot to recover exact declarations, then returned
to A044 head and applied new A045 replay through revision-checked code mode.
Scientific parameters unchanged; only output root and pipeline-owned Napari
streaming differ. Config: `qc/takeover-A045/orchestrator.py`; new output root:
`blind-a018-replay-A045`. Historical outputs and snapshots remain preserved.
Compilation completed. Managed execution has been started; completion and
same-coordinate visual assessment remain pending.

A045 completed. Its compiled body/graph overlays render correctly at C1.
Screenshot `20260921T031959526075Z_...png` was actually opened and inspected:
the lower soma has a clearly visible upward calcein process with little/no
overlaid graph, while another direction is traced. This is a concrete process
coverage failure, not a reason to loosen nuclear admission. Current code yields
910 somata versus historical A035's 951 despite unchanged parameters; do not
claim historical byte equivalence. The checkout includes an inherited soma
radius/overlap/final-area correction, now active in the refreshed execution server.

Registered `openhcs:neurite_candidate_diagnostics` through MCP. It calls the
existing `_identify_neurites_cellprofiler` kernel and projects its mask, medial
axis and local response as typed, source-plane-aware main-flow images. It does
not reimplement thresholding or change scientific defaults. Registered source
is retained in `qc/takeover-A046/neurite_candidate_diagnostics.py`.

A046 code-mode document adds that diagnostic after the unchanged analysis step;
both use PIPELINE_START and identical candidate controls. Config validation and
compilation passed. Managed execution started, with a new output destination
`blind-a018-diagnostic-A046`; inspection remains pending. No accepted candidate
or held-out validation is claimed.

A046 completed successfully, execution `b9bb9f66-3205-4b1d-8de3-0f3f1a80561d`,
94.4 seconds including full analysis, artifact persistence/streaming and diagnostic.
All three diagnostic images preserve channel 2 identity, 2859x2859 source shape
and physical scale 1.3556. Source-backed native inspection of the C1 upward
process gives a concrete failure stage:

- MCP tile y[1450,1514), x[1610,1658), channel 2; exact raw, response,
  candidate-mask and skeleton arrays retrieved independently.
- Within branch bbox y[1470,1514), x[1618,1633), raw range 182..5175,
  response max4913, 345 pixels with response>=30, **zero candidate pixels and
  zero candidate-skeleton pixels**.
- Response ridge peaks follow x1629→1621 with y1470→1510. This is not loss
  caused solely by the local-response cutoff, soma rooting, or final ownership.
- Opened/inspected candidate skeleton screenshot `20260921T032713870103Z_...`
  and response screenshot `20260921T032734121345Z_...`. The upward process is
  present in raw/local response but absent before tracing.

Next controlled hypothesis is the optional candidate hysteresis seed filter.
A047 is diagnostic-only (no cell or neurite measurements), with exactly the same
candidate controls except seed correction 0.85→None. Config validated, at
`qc/takeover-A047/orchestrator.py`. This ablation is not a promoted pipeline.

A047 completed. Disabling only seed hysteresis recovers **222 candidate pixels
and 71 skeleton pixels** in the identical branch bbox (both were zero). Exact
MCP samples retained in `qc/takeover-A046/C1_native_pixel_evidence.json` and
`qc/takeover-A047/C1_native_pixel_evidence.json`; the latter records the ablation.
Screenshot `20260921T033134950725Z_...png` was opened: the upward candidate
branch is now visible. Whole-field candidate skeleton increased 104,787→176,849
pixels, so false-positive and rooted-acceptance checks are still essential.

A048 full analysis with the seed filter disabled is applied and compiled.
Other settings identical to A045; output `blind-a018-candidate-A048`. This is a
candidate, not accepted. First run request was refused because the MCP became
stale after the QA-policy edit; it did not start execution. Reconnected only the
MCP stdio client, health current, and dispatched the managed run. UI and viewer
were not restarted. Current persistent MCP PTY **22665**, MCP PID **1769719**.

Canonical `ImageQaPrecondition.EARLIEST_FAILED_DEPENDENCY` now explains seed
hysteresis ablation and its non-acceptance boundary; local QA skill points there.
QA/knowledge suites: **36 passed in 14.80s** (-n4, timeout60). Local skill
validator passed. No production defaults changed.

### Sol continuation checkpoint — A048 completed, broader QA next

User is switching to Sol to conserve usage. Do not redo the A035–A048 investigation.
A048 execution `01709a24-b7f0-4731-80c3-63b37c7a4eba` completed in 154.1s.
Snapshot `50fb19fd-76f0-4efb-8e08-c78f54b1dbec`, main@22. MCP remains live
in PTY22665. UI and viewer preserved; no analysis currently running.

Actually opened screenshot `20260921T033608759013Z_...png`: the C1 upward
branch is now traced continuously in the lower soma's colour. Exact MCP accepted
outgrowth-label sample y[1450,1546), x[1610,1658), channel index1 confirms
**71 accepted pixels, all label380**, inside the frozen branch bbox
y[1470,1514), x[1618,1633). Stored raw MCP reply in
`qc/takeover-A048/C1_accepted_labels.json`. Same soma count910 as A045/A046;
only the optional candidate seed filter changed. This proves local recovery,
not full-field specificity or biological acceptance.

Guidance follow-up: progressive-authoring-context suite10passed14.41s. Fresh MCP
`image_analysis_workflow` response has15,903characters and contains the new
seed-filter rule within its default budget. Total guidance checks this turn46passed.
Skill-creator and Diataxis were used for the narrow practical guidance addition.

NEXT: broaden raw-first A14 QA before another parameter change. Viewer currently
shows raw calcein alone at p0.5–97.5=[145,2380], camera source centre(715,715),
world centre[0,969.254,969.254], zoom0.6. New wide raw snapshot
`20260921T033727137116Z_napari_5555_OpenHCS_Napari_Visualization.png` is saved
but NOT YET OPENED/INSPECTED. Open it, inspect moderate calcein and matching
DAPI, then reveal A048 body/graph overlays at identical coordinates. Select and
freeze additional sparse/dense/dim/crossing/background crops from raw, not from
the result. Candidate sensitivity increased substantially; inspect unsupported
short branches and any false bridges rather than accepting aggregate growth.

Current A048 route suffix:
`_step_0_scope__run_media_ts_0BA20E780BA20E78_blind-neurite-analysis-20260916_blind-full-A018_P001_openhcs::pipeline::submission_7fbda1521cc0::step_0_name_TargetedNeuriteOutgrowth`.
Raw prefix `origin_pipeline_kind_main_projection_main`; ROI prefixes
`origin_pipeline_kind_artifact_projection_cell_bodies` and
`origin_pipeline_kind_artifact_projection_neurite_morphology`, each with `_shapes`
after suffix. Accepted trace image prefix
`origin_pipeline_kind_artifact_projection_neurite_outgrowth`, suffix then
`_scalar_stack`. Raw/image-stack channel index1 is FITC; singleton ROI route's
local channel index0 corresponds to semantic channel2. Do not confuse those.

After development QA: identify genuinely untouched wells from exposure records,
freeze settings, run/inspect those, then plan the complete blind run within disk
budget (external drive ~13–14GB free, four full label stacks per well are costly).
Do not call original exposed five wells held-out. Preserve all prior outputs.
No candidate is yet accepted; full goal remains active.

### A048 raw-first A14 crop panel completed

The A14 development-well QA panel is now recorded in the authoritative ledger:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A048/A048_raw_first_qa_ledger.md`

Six predeclared raw-first regions were inspected at matching coordinates with
FITC, DAPI and A048 overlays:

- C1: the confirmed soma-rooted branch missed by A045 is restored by A048
  (71 accepted pixels, all label 380).
- C2: four adjacent DAPI-anchored somata remain distinct in a dense overlap;
  their traced paths remain signal-aligned.
- C3: the central DAPI-anchored soma and its connected dim left branches are
  traced. A faint right-edge segment is untraced but visibly disconnected from
  that soma in the crop, so it cannot be classified as a missed rooted branch.
- C4: a continuous sparse process is traced without promoting its small bright,
  DAPI-negative endpoint to a soma; its root may lie beyond the crop.
- C5: a bright ring-shaped FITC object has no DAPI anchor and is not promoted to
  an independent soma. A path from a nearby DAPI-anchored soma reaches it along
  contiguous FITC, so biological interpretation remains cautious.
- C6: the interior of the empty-background crop remains free of A048 overlays;
  the only edge graph fragment coincides with a real raw fragment.

The live viewer was returned to an A14 stitched-field overview showing raw FITC,
cell bodies, neurite morphology and nuclei together. The opened screenshot is:

`/tmp/openhcs-neurite-takeover-20260921/20260921T035136708712Z_napari_5555_OpenHCS_Napari_Visualization.png`

This completes development-well crop QA and finds no new false positive in the
panel, but it is not held-out validation. A14 informed parameter development.
The confirmed exposed A018 wells are currently A02, A14, A25, A29 and A37.
Finish the source-provenance audit before freezing genuinely untouched wells;
do not infer exposure from similarly named outputs belonging to another plate.

## A049 — frozen held-out validation completed and failed upstream

A048 settings were frozen before opening a deterministic held-out set. Prior
raw/result exposure excluded `A02, A08, A14, A16, A18, A25, A29, A37, A50,
A53`; SHA-256 ordering selected `A60, A45, A04, A13, A07`. The freeze record is
`qc/takeover-A049/A049_VALIDATION_FREEZE.md` and the exact raw-first evidence is
in `qc/takeover-A049/A049_RAW_FIRST_VALIDATION_LEDGER.md`.

A049 execution `6c7a82be-0efe-4729-a166-d62b3a98b976` completed for all five
wells. One well at a time was loaded into a fresh Napari process to avoid the
large retained ROI memory footprint. A04 passed its reviewed crop. A07 exposed
a compact group of three clear DAPI nuclei without nucleus overlays. A13, A45,
and A60 exposed touching or bilobed DAPI nuclei represented as single nucleus
instances. The frozen set therefore failed at the common earliest dependency:
nuclear admission/instance separation. No setting was changed after any A049
well was opened.

Do not repair A049 against these held-out wells or call the analysis validated.
The next development cycle should instrument and correct nuclear seed/instance
separation, then freeze a new validation set before judging downstream cell
bodies or neurite tracing.

## A050-A053 — nuclear instance-separation development

The A049 failure was traced to a width/area mismatch in the original
declumping gate: ordinary touching nuclei never reached watershed unless their
combined area approached the maximum-width area. A050-A052 progressively
changed split eligibility and shape-marker generation. The exact broad-neck
doublet remained one object in A052 despite two distinct DAPI intensity
centers, proving that distance geometry alone was insufficient.

A053 adds intensity-assisted markers to the shape watershed, with guards for
hollow objects and implausibly small fragments. The exact A60 development
target now splits into two instances. Four unrelated crops and a field overview
showed no obvious split of an ordinary single nucleus. Its first full-field
implementation raised kernel time from 3.692 s to 22.139 s. A054 moved marker
work to each candidate component, preserved the target split, and reduced
kernel time to 5.995 s. A054 produced 2,558 accepted objects versus A053's
2,557. Do not freeze the kernel yet: A055 is the complete-pipeline development
run on the already exposed A07, A13, A45 and A60 wells.

Authoritative ledger:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A053/A053_INTENSITY_DECLUMPING_LEDGER.md`

Affected unit suites currently pass: **99 tests** across
`test_count_cells_simple.py` and `test_neurite_outgrowth.py`.

The A049 held-out verdict remains failed and unchanged.

## A055 — four-well nuclear development review

A055 completed all four scientific computations for already exposed wells
`A07`, `A13`, `A45`, and `A60` and persisted the outputs. Its managed execution
was cancelled only during unbounded Napari settlement after the viewer exceeded
8 GB RSS, so A055 itself is not recorded as terminally completed. The outputs
were reopened one well at a time through MCP for bounded QA.

At the exact historical A049 coordinates, the A054 kernel repairs the clear
multi-center failures in A07, A13, and A60. A less-saturated close view confirms
four independently supported DAPI centers in the A07 group and four matching
instances. The A45 target has one strong compact center plus a faint elongated
tail; it is now treated as biologically ambiguous rather than a required split.
No threshold was changed to force it apart.

The full evidence ledger is:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A055/A055_DEVELOPMENT_QA_LEDGER.md`

The canonical QA policy and local neurite QA skill now state that lobed outline
shape alone does not justify nuclear splitting; a less-saturated raw view must
support multiple intensity centres. A055 remains development evidence. Next:
freeze a genuinely untouched validation set before downstream acceptance.

## A056 — raw-unseen reserve completed and rejected upstream

The unchanged A055/A054 candidate was frozen for `A03`, `A20`, `A40`, and
`A59` before their raw images or current outputs were opened in this repair
cycle. Because historical summary values exist for every A018 well, this is
recorded as a **raw-unseen validation reserve**, not a pristine statistical
holdout. The authoritative freeze and evidence ledger is:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A056/A056_VALIDATION_FREEZE.md`

Running-UI snapshot `88f66157-4d34-475c-a0a2-2ec306eaff3a` compiled and managed
execution `b2976f36-ee77-4531-b58a-65f35ebf71ea` completed. The output plate
contains 74 durable files: 18 per well plus two metadata files.

Raw-first review stopped at the first well, A03. At native center
`[0, 1937, 1937]`, zoom `2.5`, and p5-p99 DAPI limits `[670, 3127]`, one dim
object has a single diffuse intensity centre but is represented by a larger
green nucleus plus an adjacent thin blue instance. The blue instance has no
independently supported raw nuclear centre. Exact matching screenshots are:

- raw: `qc/takeover-A056/20260921T073716582804Z_napari_5555_OpenHCS_Napari_Visualization.png`
- overlay: `qc/takeover-A056/20260921T073741031707Z_napari_5555_OpenHCS_Napari_Visualization.png`

A056 is therefore rejected at nuclear instance separation. Downstream soma and
neurite acceptance was not attempted. A03 is now exposed development evidence;
`A20`, `A40`, and `A59` remain unopened reserves. Repair must use already
exposed development wells before another declaration is frozen.

## A057-A061 — source-lineage diagnosis and bounded satellite repair

The A03 failure was investigated only on exposed A03 and A07 development wells.
The typed round-object diagnostic now retains per-object source-component
lineage, shape and response statistics, weak-core candidates, and the narrower
adjacent-satellite candidate set. These are projections of the production
kernel, not a second segmentation implementation.

A058 proved that the unsupported A03 satellite and supported main nucleus came
from different threshold-stage source components, each producing one output.
Watershed did not create the satellite. A059 measured the satellite as a 17 px,
5.199 um minor-axis object with peak local response 282 beside a 114 px,
15.793 um minor-axis nucleus with peak response 720. A060 showed why a global
weak-core filter was unsafe: 63 A03 objects met that diagnostic condition,
including visually plausible large faint nuclei.

A061 therefore uses a conjunctive production rule derived from existing scales:
reject only a width-accepted weak-core object smaller than the pair-area scale
and within the seed-spacing distance of a larger width-accepted strong-core
object. Isolated faint objects remain accepted. Managed execution
`3cd98607-1b85-42e0-b104-ed77b5f6a4e3` completed in 19.6 seconds and retained
all typed diagnostic outputs for A03 and A07.

At the original A03 failure coordinate, the 17 px satellite is rejected and the
main nucleus remains accepted; only three A03 objects satisfy the full rule. At
the A07 four-centre positive-control coordinate, all four independently
supported DAPI centres remain distinct. The only two A07 rejections were
visually confirmed tiny weak lobes attached to larger supported nuclei. The
affected round-object and neurite suites pass **102 tests in 7.81 seconds**.

Authoritative ledger:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A061/A061_DEVELOPMENT_QA_LEDGER.md`

A061 is a development candidate, not held-out acceptance. A20, A40, and A59
remain unopened reserves. Before opening any of them, freeze the exact code,
pipeline document, environment and deterministic reserve selection in a new
validation record.

## A062 — frozen raw-unseen nuclear validation in progress

The A061 scientific candidate was frozen before opening A20, A40 or A59. Only
the well filter and output destination changed. Managed execution
`718ec62e-0c4b-4dd1-8427-848c8b722b60` completed without error in 33.35 seconds
and reported all three wells. The freeze and accumulating validation ledger are:

- `/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A062/A062_VALIDATION_FREEZE.md`
- `/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A062/A062_RAW_FIRST_VALIDATION_LEDGER.md`

A20 passes the frozen nuclear checkpoint. Raw DAPI was reviewed before result
layers at full, moderate and dim windows. A deterministic 3 x 3 native-coordinate
grid showed no obvious isolated raw nucleus without an accepted object and no
unsupported single-centre split. All nine captures have distinct hashes and the
live final viewport matched the declared bottom-right grid coordinate.

All six A20 adjacent-satellite candidates were then reviewed at matching raw
coordinates. Each is a small weak fragment beside a larger supported nucleus.
The visually least obvious 20 px fragment was reconciled using typed lineage:
it shares source component 15018 with a 208 px accepted bright nucleus, with
peak responses 382 versus 17,947. Its rejection is supported.

A40 also passes the frozen nuclear checkpoint. Raw DAPI was opened before the
results and reviewed at full `[0, 65534]`, moderate `[623, 17946.837]` and
dim-signal `[623, 847]` windows computed over the complete stitched image. The
accepted layer contains 2,422 objects. A deterministic 3 x 3 native-coordinate
grid contains no obvious isolated nucleus without an accepted ROI and no
unsupported single-centre split.

The A40 satellite layer contains 11 shape members representing 10 unique typed
candidates. Every unique candidate was reviewed at dim and moderate windows and
is supported as a weak peripheral fragment without an independent raw core. One
duplicated member produced a malformed large selected-overlay polygon; that
render was excluded from biological interpretation and retained as provenance.
The underlying candidate was adjudicated from separate raw-only captures.

A59 also passes the frozen nuclear checkpoint. Raw DAPI was opened first and
reviewed at full `[0, 65534]`, moderate `[634, 18315]` and dim-signal
`[634, 857]` windows computed over the complete stitched image. The accepted
layer contains 2,371 objects. A deterministic 3 x 3 native-coordinate grid
showed no obvious isolated compact nucleus without an accepted ROI and no
unsupported single-centre split; all nine captures have distinct hashes and
the live final viewport matched the declared bottom-right coordinate.

All seven A59 adjacent-satellite candidates were reviewed at their typed
centroids under dim and moderate windows. Each is a small fragment on diffuse,
merged or peripheral signal without an independent compact core. Candidate 1
also produced a malformed selected-overlay polygon despite a typed area of 16
pixels; the overlay was excluded, and raw-only captures supported rejection.

The frozen A062 nuclear checkpoint therefore passes across A20, A40 and A59
without a post-exposure scientific change. This authorizes progression to
blinded soma and neurite-stage validation. Soma assignment and neurite tracing
remain unvalidated and are the next scientific gate.

## A063 — downstream execution complete; validation instrumentation incomplete

The exact downstream candidate frozen in
`qc/takeover-A063/A063_VALIDATION_FREEZE.md` executed successfully as managed
execution `68d92678-f0b7-42e9-be91-8964254da471` for A20, A40 and A59. The
output plate is
`blind-a018-downstream-validation-A063/P001_openhcs_openhcs` and contains raw
inputs, nuclei, cell bodies, rooted neurites, unified neurons, graph ROI/SWC and
measurement artifacts for all three wells.

A20 raw-first review used whole-mosaic moderate and faint-process windows,
followed by a deterministic 3 x 3 grid. A conspicuous calcein-bright object
without a neuron overlay was adjudicated at a matching DAPI viewport and has no
plausible nuclear anchor, supporting its exclusion. No obvious unsupported soma
assignment or crossing-derived ownership error was found in the first overview.

The review nevertheless found a proof gap before any pass claim: the frozen
function does not persist the pre-rooting candidate mask, secondary owner
regions, or the unrooted candidate residual required by the predeclared review
protocol. Final artifacts cannot exactly reconstruct those stages. A063 is
therefore not called passed. A20, A40 and A59 are now downstream-exposed. The
next candidate must project those stages from the production computation and be
frozen on a different raw-unseen reserve.

Authoritative accumulating ledger:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A063/A063_DOWNSTREAM_VALIDATION_LEDGER.md`

## A064 — checkpoint-complete reserve ready for raw-first review

The A064 candidate was frozen on raw-unseen wells A49, A48 and A22 before any
of their pixels or result values were opened. The production neurite computation
now retains candidate foreground, unrooted residual and secondary-ownership
checkpoints in addition to the existing raw, label, ROI, graph and measurement
artifacts.

The first execution (`6896faed-9b6c-4fbe-becf-b020fbc317aa`) exposed a stale
execution-server declaration: the server retained the former seven-output ABI
while spawned workers imported the current ten-output implementation. It failed
with `10 != 7` before retaining an A064 output root. Replacing that server through
the managed UI lifecycle corrected the operational source drift without changing
the frozen scientific candidate.

Current-source execution `ae9d0431-7498-4528-827a-c266bbe0eb95` then completed
all three wells in 270.257 seconds with no error. UI state independently reports
the plate terminally complete. The output holds 64 non-lock files: six copied
raw inputs, nine QA checkpoints, twelve label TIFFs, fifteen ROI ZIPs, three SWC
graphs, six CSV tables, twelve segmentation summaries and one metadata record.

Authoritative pre-exposure ledger:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A064/A064_RAW_FIRST_VALIDATION_LEDGER.md`

Execution/artifact completeness passes. Scientific review remains pending and
must proceed A49, A48, A22, raw first, stopping at the first failed dependency
stage. No reserve pixel had been opened when the ledger was created.

## A065 — runtime image-artifact lineage validated

A065 reran the frozen A064 scientific declaration on the same three exposed
validation wells, changing only the output destination. Managed execution
`205f3be1-f8e0-4108-98ab-21426153b3fe` completed A22, A48 and A49 without
error. Its output contains the same 65 filesystem entries as A064, including
the lock file. Relative artifact paths match. Pixel TIFFs, CSV measurements and
SWC graphs are byte-identical to A064; differences are confined to generated
ROI ZIP containers and path-bearing/time-bearing text summaries.

The output metadata now retains every QA checkpoint as a typed
`source_artifact` projection with its exact virtual path, well/site/channel/z/time
address, source alias, artifact kind, provenance and image payload metadata.
Fresh-process MCP inventory found all three `neurite_candidate_mask` images from
those projections, and the A49 candidate mask streamed successfully into the
existing managed Napari viewer on port 5555. This closes the former gap where
the checkpoint TIFF existed on disk but was absent from the typed workspace
projection and therefore could not be opened through the supported MCP route.

The viewer is currently isolated to A49 raw calcein plus the candidate mask.
The binary checkpoint required a `[0, 1]` display window because its persisted
uint8 storage-domain scale is `[0, 255]`; this is a display-metadata issue, not
a scientific-output difference. The whole-mosaic foreground overview is open,
but the A49 candidate stage is not yet called passed: bounded raw/mask and
residual review remains before progressing to rooted ownership and graph QA.

## A066 — aggregate label lineage and direct MCP streaming validated

A066 changed only the A065 output destination and completed A22, A48 and A49 as
managed execution `1b46b91c-71f0-463d-a007-5ebff2f41e9d` in approximately
280.10 seconds. The current-source execution server was launched by the running
UI after the earlier idle server was retired; both runtime status and UI state
report terminal completion with no execution error.

The real A066 metadata retains all twelve multi-plane label TIFFs as typed
`source_artifact` projections. Each has `artifact_kind=object_labels`, no false
scalar address, its exact well execution scope, the `runtime_slice` plane axis,
and two ordered source provenance planes. The initial direct MCP stream exposed
a consumer error: route metadata redundantly required scalar channel even
though channel 1/2 belonged to the payload-local plane axis.

The viewer metadata projector now derives scalar route fields by excluding
components declared in `plane_component_values`; the same nominal derivation is
used by manual streaming and materialization. No channel was invented and no
valid aggregate was rejected. A fresh MCP call then streamed and settled the
A49 neurite-outgrowth label TIFF on the existing Napari viewer with shape
`(2, 2858, 2859)`, aggregate channel domain `[1, 2]`, and scalar route identity
`A49/z1/t1`. Focused suites pass 98 tests; the broader affected boundary,
including the real synthetic neurite integration, passes 188 tests.

Authoritative record:

`/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A066/A066_ARTIFACT_LINEAGE_VALIDATION.md`

The current viewer is isolated to the A49 aggregate neurite-label image stack.
It is transport/lineage evidence, not a biological acceptance view: the current
receiver presents integer labels as a grayscale image stack rather than a
native colored Napari Labels layer. Scientific continuation remains A49
raw-first review of rooted ownership, residual, secondary ownership, unified
neurons and graph correspondence. The pathological giant ROI archive remains
out of scope for direct viewer expansion.

## A066 frozen-candidate result — 2026-09-21

A066 completed on A49, A48 and A22 through the managed UI/runtime and produced
the complete typed artifact set. The aggregate-label projection repair was
verified by 98 focused tests, 188 broader affected tests, and live MCP streaming
of the persisted two-plane label artifacts. This closes the operational and
lineage failure; it does not by itself accept the biology.

The frozen A49 candidate-foreground gate passed across the original 3 x 3 raw
grid. Review then advanced to rooted neurites and the unrooted residual at the
same coordinates. The rooted result no longer has the earlier quadrant-specific
sensitivity artifact, but it still under-traces supported signal. In the
middle-right sector, a long bright process visible in raw calcein is absent from
the rooted raster and remains in the residual. Other sectors retain coherent
branched residual components. A49 has 113,194 residual pixels out of 753,005
candidate pixels (about 15.0%); this number is descriptive, while the matched
raw-to-result view is the acceptance evidence.

The frozen candidate therefore fails at rooted-neurite completeness. Per the
predeclared protocol, later-stage acceptance and A48/A22 scoring stop here.
The next scientific change must address ownership/rooting of valid candidate
components on development data, then create a new frozen candidate before any
validation wells are scored again. Do not tune on A49, A48 or A22.

Authoritative evidence:

- A064 review ledger:
  `/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A064/A064_RAW_FIRST_VALIDATION_LEDGER.md`;
- A066 lineage and streaming report:
  `/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/takeover-A066/A066_ARTIFACT_LINEAGE_VALIDATION.md`;
- rooted grid: `takeover-A066/grid-rooted-neurites/`;
- residual grid: `takeover-A066/grid-unrooted-residual/`.

Keep rejected attempts and do not promote a candidate from aggregate counts.

## A067 development replay and viewer handoff — 2026-09-21

A067 replayed the unchanged A066 scientific pipeline on development well A14
through the managed UI compile/run actions. Execution
`c7c99259-c38e-4048-9c0c-aae0cf7a7b35` reached terminal `complete` and wrote
the full raw, checkpoint, object-label, ROI, SWC and measurement artifact set
under:

`/home/ts/code/projects/openhcs/mcp_outputs/analysis-runs/blind-a018-rooting-development-A067/P001_openhcs_openhcs`

The existing managed Napari viewer on port 5555 now contains both the frozen
A49 evidence layers and the A14 development layers. Its active view is A14 raw
FITC with a 2--99.8 percentile intensity window plus the neurite-outgrowth ROI
overlay (71,150 shapes). The A14 candidate mask, unrooted residual and secondary
ownership checkpoints are loaded as separate hidden layers for direct toggling.
The current screenshot is:

`/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/a067-a14/20260921T163053984846Z_napari_5555_OpenHCS_Napari_Visualization.png`

This is the development baseline, not an accepted repair. The raw/overlay view
still exposes faint processes that require topology-aware ownership/rooting
work before another frozen validation candidate is created.

## A075-A078 final-topology loss attribution — 2026-09-21

A075 isolated a remaining A14 development failure after secondary-owner
adoption and signal-supported soma repair. Of 141,212 candidate trace pixels,
128,674 were represented in the final rooted candidate result. The final graph
conversion removed 5,553 pixels that already carried an owner and introduced
137 pixels. Exact sparse-mask coordinates from the viewer API localized a long
dropped segment beside owner 828's soma; matched raw FITC showed continuous
faint neurite signal at those coordinates. This is a topology-stage loss, not
background admitted by a permissive detector threshold.

A077 reran the unchanged A075 scientific declaration on a freshly restarted
execution server after adding one attribution measure. It completed through the
managed UI and reproduced every prior count. Only 250 of the 5,553 dropped
pixels (4.50%) overlapped resolved crossing support, ruling out the post-repair
crossing overlay as a direct source of most removed pixels. This does not yet
exclude a small crossing mismatch from disconnecting a longer downstream path.

A078 completed through the managed UI and partitioned the dropped set into
5,275 pixels represented by graph paths but rejected as unrooted and 278 pixels
absent from the path graph. The two values sum exactly to the 5,553-pixel total.
This localizes the dominant loss to logical soma rooting after graph formation,
not isolated candidate pixels.

A079 completed through the managed UI and partitioned the 5,275 rejected graph
path pixels by an independent, pixel-domain connectivity test. Of those pixels,
4,689 (88.9%) were already in a same-owner connected component that reached the
owner's soma neighbourhood; only 586 were physically soma-detached. The graph
therefore discarded predominantly valid, physically rooted owner paths. A
minimal same-owner plus-shaped trace reproduced the defect: geometric crossing
pairing split one nominal owner's connected trace into two path groups and kept
only the group containing the root.

The A080 repair makes geometric crossing pairing provisional until nominal path
ownership is known. Final topology is analysed independently for each nominal
owner, with the shared crossing cores supplied as context. This prevents a
global union/majority vote from deleting one owner's valid trace or merging
different owners at a crossing.

A080 completed through the managed UI on exposed development well A14 as
execution `095847e3-c45a-4565-830e-f0dc26c16a1c` in 114.1 seconds. Relative to
A079, the final-topology rejection fell from 5,553 to 1,129 pixels. The
independent physical-root partition changed from 4,689 rooted / 586 unrooted to
352 rooted / 495 unrooted, while candidate trace yield rose from 0.9112115 to
0.9357632 and traced pixels rose from 128,674 to 132,141. Residual pixels fell
from 60,546 to 51,143 and secondary-owned residual fell from 41,084 to 31,659.
Cell count remained 1,124; branches changed from 6,438 to 6,572 and processes
from 8,728 to 8,632, so the recovered trace was not accompanied by a branch or
process explosion. Candidate mask and candidate trace inputs remained unchanged
at 446,830 and 141,212 pixels.

Matched raw and label views at the exact previously dropped coordinate show the
faint soma-connected process retained in A080:

- raw: `/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/a080-a14/known-dropped-coordinate/raw-axis0/20260921T223524075614Z_napari_5555_OpenHCS_Napari_Visualization.png`
- labels: `/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/a080-a14/known-dropped-coordinate/labels-axis1/20260921T223643122598Z_napari_5555_OpenHCS_Napari_Visualization.png`

The scalar raw route initially appeared blank because the shared viewer channel
slider remained at index 1 after the aggregate label stack; restoring it to
index 0 exposed the scalar raw image. This was presentation state, not missing
data.

A080 exposed one summary-semantic defect: `resolved_crossovers` was read from
the owner-separated final topology, where cross-owner shared nodes correctly no
longer exist, and therefore reported zero. The corrected declaration now counts
connected physical crossing-core components whose adjacent rooted support has
at least two positive owners before owner-separated topology analysis. Focused
and affected validation passes: **133 tests passed in 16.14 seconds**. A081 is
prepared as an otherwise identical A14 rerun with output root
`blind-a018-rooting-development-A081`; it has not yet been applied or executed.

A080 remains development evidence, not a frozen candidate. The untouched
A49/A48/A22 reserve must not be reopened until A081 confirms the corrected
crossover summary and matched raw/label review confirms no cross-owner mergers.

## A081 freeze, reserve runtime failure, and A082 robustness repair — 2026-09-21/22

A081 completed on development well A14 as execution
`b36ac95b-2bb2-4fa7-a8e9-14c6ca8201e5`. Its scientific values reproduce A080,
and the corrected physical crossing summary reports 33 resolved crossovers.
The candidate declaration, implementation hash, environment, metrics, and
reserve rule were frozen before reserve execution in:

`/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/A081_CANDIDATE_FREEZE.md`

Frozen reserve execution `becf4927-d644-47bf-abf8-b06dcf6877e6` compiled and
started the ordered wells A22, A48, A49, but failed on A22 before producing an
output root. A48 and A49 never ran. The exact runtime error originated when a
three-pixel endpoint-free cycle reached Skan 0.13.1, whose path-CSR construction
underallocates the pointer buffer for that graph (`index pointer size 1 should
be 2`). This graph exists only after segmentation, ownership and owner-local
cropping, so generic compilation cannot prove it absent. The correct prevention
boundary is `_analyze_topology`, which owns valid neurite-path admission.

The implementation now removes only connected three-pixel components whose
three pixels each have degree two under the same full-neighbourhood graph.
Ordinary lines and larger cycles are preserved. New regressions cover an
isolated triangle, a triangle beside a valid path, and a triangle owned by a
separate cell. The exact focused family passes 9 tests; the complete neurite
unit module passes **81 tests** with eight workers.

A22 is now a spent runtime-robustness case rather than held-out biological
evidence. A48 and A49 remain unopened. The otherwise unchanged A082 declaration
selects only A22 and writes to a new development root:

`/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/a082_runtime_robustness_a22_orchestrator_config.py`

After a managed execution-server replacement, A082 completed as execution
`6393063b-0021-41f5-9da6-86a524f48f3d` in 172.3 seconds and produced all 24
expected non-lock artifacts. A22 contains 1,378 cells, 256,784.05 source-length
units of outgrowth, 10,677 processes, 8,446 branches, and 54 resolved physical
crossovers. The former three-pixel-cycle failure is therefore closed at its
topology-owner boundary.

A083 then replayed development well A14 with the corrected implementation as
execution `2645152d-f580-40a7-9d49-d59f7654f40b`. Relative to A081, the guard
removed only 12 published trace pixels and four tiny invalid process fragments;
cell count, branch count, candidate mask, crossover count, cell-body metrics,
and upstream segmentation are unchanged. The corrected implementation and
declaration are frozen in:

`/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/A083_CANDIDATE_FREEZE.md`

Current-candidate validation wells A48 and A49 are running without parameter
changes as execution `0a75b787-a87f-4662-844d-b66e87953d91`, writing to
`blind-a018-rooting-current-candidate-validation-A083`. These wells appeared in
earlier rejected candidates, so they are current-candidate validation rather
than never-seen biological holdouts.

The execution completed successfully with 47 non-lock files. A48 reported
1,263 cells, 286,777.62 outgrowth units, and rooted-candidate trace yield
0.927861; A49 reported 1,215 cells, 309,838.14 outgrowth units, and yield
0.931903. The current-source viewer was restarted through MCP and raw FITC plus
colored morphology paths were reviewed across four fixed quadrants per well at
the same 0.1/99.8 percentile window and zoom. Coverage follows the visible
neurite fields without a quadrant-dependent sensitivity boundary or recurrence
of the cross-owner topology failure. The evidence and bounded interpretation
are recorded in:

`/home/ts/code/projects/openhcs/mcp_outputs/current-neurite-review/A083_CURRENT_CANDIDATE_VALIDATION.md`

## Viewer and agent-guidance validation — 2026-09-22

The viewer's route-local/global channel alignment defect has an architectural
repair in the current worktree: per-route component domains remain exact,
viewer-wide coordinates are derived as their union, and already-mounted
shape-compatible routes are reprojected when the union expands. Deleted and
replaced routes purge their domains before new routing. The affected viewer
boundary passes **191 tests**, including real model proof that scalar channel 2
and stack channel 2 resolve to the same global coordinate. The currently open
viewer predates this code and must be replaced, not duplicated, before a live
alignment proof.

The packaged `use-openhcs` skill validates successfully. QA and progressive
authoring-context tests pass **14 tests**. Fresh MCP retrieval proves that
`first_use` routes image-analysis work to `image_analysis_workflow` and
`viewer_review`; the live contexts expose shared SITE-stack percentile fitting,
fixed-coordinate raw/result review, native-resolution samples, residual ranking,
development/held-out separation, one-semantic-change attempts, custom-function
registration, and escalation on missed signal. This is live discoverability
evidence, not merely source-text inspection.

The oversized Napari interface is a separate presentation defect. The X server
reports a 1600x900 screen as 248x140 mm (164x163 DPI), while monitor EDID reports
approximately 550x310 mm. Napari's normal 9-point stylesheet therefore renders
near 20 pixels. No `QT_SCALE_FACTOR` or `QT_FONT_DPI` override is present. Any
OpenHCS correction must be viewer-scoped and configurable rather than imposing
a hard-coded global DPI on legitimate HiDPI systems.

A direct PyQt5 launch probe confirms the mechanism: the inherited session gives
a 9-point label a 28-pixel font height and 70x28 size hint, while a viewer-local
`QT_FONT_DPI=96` gives the same declared font a 17-pixel height and 42x17 size
hint. `QT_SCALE_FACTOR=0.585` does not correct the point-font metrics. The
correct control is therefore the detached viewer's pre-application font-DPI
policy, and an already running viewer must be restarted for it to take effect.
