# OpenHCS Blind Neurite Analysis Handoff

## Purpose

Continue and finish the blind DAPI/calcein neurite analysis through the managed OpenHCS UI and viewer. Do not promote a configuration until native-coordinate visual QA passes on the frozen A14 diagnostic field and then on the five held-out wells.

This handoff records the state through candidate A035 on 2026-09-20/21.

Current continuation: [OPENHCS_NEURITE_TAKEOVER_STATUS.md](OPENHCS_NEURITE_TAKEOVER_STATUS.md)
records the verified takeover, MCP timeout repair, execution-server refresh,
and diagnostic attempts after A035. Read it before acting on the historical
"current" state below.

## Non-Negotiable Workflow Rules

- Use OpenHCS MCP capabilities for pipeline authoring, execution, artifact inspection, and viewer control.
- Do not replace managed QA with standalone scripts, direct callable invocation, shell-driven GUI actions, keyboard/mouse automation, or `xdotool`.
- Start a resumed session with `openhcs_health_check`.
- Load the MCP `image_analysis_workflow` and `viewer_review` authoring contexts before interpreting new results.
- Load the OpenCode skill `openhcs-neurite-visual-qa`.
- Inspect raw data before overlays: DAPI, moderate-window calcein, neurite-window calcein, composite, then nuclei, bodies, and traces.
- Stop at the earliest failed biological dependency. Downstream metrics cannot rescue invalid nuclear anchors or somata.
- Change one biologically interpretable control at a time and preserve each rejected output directory.
- Do not call segmented nuclei a cell count. A cell requires a plausible DAPI nuclear anchor colocalized with a broader calcein-positive soma.
- Keep ingestion and biological interpretation separate. The selected source plate is a valid OpenHCS plate and must retain its native handler.

## Biological Contract

- Channel 1 / source alias `DAPI`: nuclear anchors only. Compact nucleus-scale morphology is required; DAPI alone does not define a soma.
- Channel 2 / source alias `FITC`: calcein-positive neuronal cytoplasm. A soma is broader calcein-positive cytoplasm surrounding or adjoining a plausible DAPI nucleus.
- A neurite is a thin, continuous calcein-positive extension rooted in a valid soma.
- Tiny calcein islands, puncta, isolated flecks, arbitrary response patches, saturated debris, and unsupported paths are not somata or neurites.

## Source Plate and Frozen QA Field

Selected source plate:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-full-A018/P001_openhcs
```

Frozen diagnostic well: `A14`

Raw images:

```text
mosaics/A14_s001_w1_z001_t001.tif  # DAPI
mosaics/A14_s001_w2_z001_t001.tif  # FITC/calcein
```

Both images:

- Shape: `2859 x 2859`
- Physical scale: `1.3556 um/pixel` in Y and X
- Site `1`, Z `1`, timepoint `1`, well `A14`

Frozen viewer presentation:

```text
center = [0.0, 1937.1524, 1937.1524]
zoom = 0.65
DAPI display percentile window = p0.5-p99.9
DAPI resolved contrast limits = [656.0, 17158.0]
```

Do not move this viewport when comparing nuclear candidates.

## Held-Out Set

The five managed held-out wells are:

```text
A02, A14, A25, A29, A37
```

Frozen selection record:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/heldout-raw-selection/selection.json
```

Earlier body-response evidence:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/qc/heldout-body-response-experiment/
```

Body response `100` was preferred over `200` and `300` in that managed comparison.

## Baseline Analysis Controls

Controls held constant unless explicitly stated for a candidate:

```text
pixel size: 1.3556 um
cell-body intensity above local background: 100
cell-body minimum area: 200 um2
cell-body approximate maximum width: 30 um
outgrowth intensity above local background: 30
candidate threshold correction factor: 0.02
candidate hysteresis seed correction factor: 0.85
nuclear approximate maximum width: 30 um
```

Pipeline function:

```text
openhcs.processing.backends.analysis.neurite_outgrowth.neurite_outgrowth_metaxpress
```

Registry ID:

```text
openhcs:analysis_neurite_outgrowth_neurite_outgrowth_metaxpress
```

The source-backed signature confirms these nuclear controls:

```text
MetaXpressNuclearSettings(
    channel_index=1,
    approx_min_width=5.0,
    approx_max_width=30.0,
    intensity_above_local_background=100.0,
)
```

## Candidate History

### A032: Five-Well Candidate, Rejected

Output:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-a018-targeted-A032/P001_openhcs_openhcs
```

Configuration difference:

```text
nuclear approx_min_width = 10.0 um
nuclear intensity_above_local_background = 100.0
```

Execution completed with 80 result artifacts across the five held-out wells.

Aggregate values looked numerically plausible but were not acceptance evidence:

- 759-1,028 somata per well
- Mean soma area 511-555 um2
- Mean outgrowth 157-201 um/cell
- 78-88% significant-growth cells

Reason for rejection at the frozen A14 field:

- Several compact bright DAPI nuclei had no nucleus ROI.
- Corresponding calcein-positive somata and connected branches were absent downstream.
- Some admitted soma masks occurred on weak or ambiguous calcein patches.
- Some traces followed real neurites, but obvious soma-connected branches were absent or fragmentary.
- Earliest failed dependency: nuclear-anchor under-admission.

### A033: Width 5 um Diagnostic, Rejected

Output:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-a018-targeted-A033/P001_openhcs_openhcs
```

Snapshot:

```text
4576c0ce-55c0-4631-b31b-c7adfcf16b85
A14 nuclear minimum width 5um diagnostic A033
```

Configuration difference from A032:

```text
nuclear approx_min_width = 5.0 um
```

Managed execution completed successfully with 16 A14 artifacts.

Summary values:

```text
nuclear segmentation summary: 17,780 cells
cell-body segmentation summary: 1,548 cells
mean outgrowth: 54.738 um/cell
mean soma area: 661.127 um2
significant growth: 61.176%
```

Reason for rejection:

- The candidate recovered some previously missed bright nuclei.
- It also admitted abundant tiny DAPI fragments and background specks as nuclear anchors.
- This was clear at the identical A14 viewport and DAPI intensity window.
- Lowering width alone changed under-admission into severe over-admission.

Evidence:

```text
/tmp/opencode/20260921T003510141684Z_napari_5555_OpenHCS_Napari_Visualization.png
```

### A034: Width 7.5 um Midpoint, Rejected

Output:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-a018-targeted-A034/P001_openhcs_openhcs
```

Snapshot:

```text
4111fa37-7cb8-49fc-87cb-1d8f33fbff71
A14 nuclear minimum width 7.5um diagnostic A034
```

Configuration:

```text
nuclear approx_min_width = 7.5 um
nuclear intensity_above_local_background = 100.0
```

Managed execution completed successfully.

Summary values:

```text
nuclear segmentation summary: 7,570 cells
cell-body segmentation summary: 1,249 cells
mean outgrowth: 68.210 um/cell
mean soma area: 774.268 um2
significant growth: 63.251%
```

Reason for rejection:

- It still missed compact bright DAPI nuclei in the frozen center cluster.
- It retained multiple small irregular low-signal objects.
- Width interpolation could not separate plausible nuclei from fragments.

Evidence:

```text
/tmp/opencode/20260921T004315057045Z_napari_5555_OpenHCS_Napari_Visualization.png
```

### A035: Width 5 um, Nuclear Local-Background 200, Rejected

Current output:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-a018-targeted-A035/P001_openhcs_openhcs
```

Current snapshot:

```text
44e269a5-94c0-4f64-93d3-a9e95e66900c
A14 nuclear width 5um local-background 200 diagnostic A035
```

Configuration:

```text
nuclear approx_min_width = 5.0 um
nuclear intensity_above_local_background = 200.0
```

Managed execution completed successfully.

Summary values:

```text
nuclear segmentation summary: 1,486 cells
cell-body segmentation summary: 951 cells
```

Reason for rejection:

- Raising local-background response from `100` to `200` strongly reduced admissions, proving the control is active.
- The same obvious compact bright center cluster remained unsegmented.
- An unsupported elongated DAPI fragment remained admitted.
- The final mask therefore still combines false negatives and false positives.
- Do not tune body or neurite controls while this nuclear dependency remains invalid.

Evidence:

```text
/tmp/opencode/20260921T004816649109Z_napari_5555_OpenHCS_Napari_Visualization.png
```

## Current Scientific Conclusion

No candidate is promotable.

- A032 / width `10`: under-admission.
- A033 / width `5`, response `100`: severe over-admission.
- A034 / width `7.5`, response `100`: simultaneous obvious misses and fragment admissions.
- A035 / width `5`, response `200`: fewer objects but still simultaneous obvious misses and an unsupported elongated admission.

The available final nuclei mask cannot distinguish whether the missed bright cluster failed at:

1. Local-background response generation.
2. Candidate thresholding.
3. Watershed splitting.
4. Final short-axis size filtering.

Another final-mask parameter sweep would not be falsifiable. Stage diagnostics are required before further tuning.

## Implementation-Backed Nuclear Segmentation Facts

Relevant source:

```text
/home/ts/code/projects/openhcs/openhcs/processing/backends/analysis/count_cells_simple.py
```

Function:

```text
segment_metaxpress_round_objects()
```

Current algorithm:

1. Converts minimum and maximum widths from micrometers to pixels.
2. Computes `local_background_response()` using maximum width as object scale.
3. Thresholds response against `intensity_above_local_background`.
4. Labels and watersheds large connected components.
5. Computes major and minor axis lengths.
6. Keeps objects using only the minor-axis interval.

Relevant lines observed:

```python
binary = intensity_above_background >= settings.intensity_above_local_background
...
_, minor_axis_lengths = _axis_lengths_by_label(labeled)
keep_mask = (minor_axis_lengths >= min_width_px) & (
    minor_axis_lengths <= max_width_px
)
```

This explains why width-only tuning cannot directly reject elongated objects: the acceptance filter currently uses minor-axis width, not eccentricity or major-axis extent. Do not change this behavior without tests and a clear API/biological decision.

## Diagnostic Function Attempt

A transient managed custom function was authored to expose four image channels:

1. Raw DAPI.
2. Exact local-background response.
3. Thresholded candidate mask.
4. Final accepted mask.

Intended function name:

```text
nuclear_admission_diagnostics
```

The `openhcs_register_custom_function` request timed out after five seconds. The immediately following `openhcs_search_functions` call was aborted, so registration status is unknown.

On resume:

1. Run `openhcs_health_check`.
2. Search the registry for `nuclear_admission_diagnostics` before registering again.
3. Do not create a duplicate if it already exists.
4. If absent, retry registration with `persist=false`.
5. Describe the returned function contract before adding it to a managed pipeline.

The attempted source was:

```python
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_inputs
from openhcs.processing.backends.analysis.count_cells_simple import (
    MetaXpressWavelengthSettings,
    segment_metaxpress_round_objects,
)
from openhcs.processing.backends.analysis.metaxpress_utils import (
    HiddenPixelSize,
    local_background_response,
)
import numpy as np


@numpy
@artifact_inputs("pixel_size")
def nuclear_admission_diagnostics(
    image,
    channel_index: int = 0,
    approx_min_width: float = 5.0,
    approx_max_width: float = 30.0,
    intensity_above_local_background: float = 200.0,
    pixel_size: HiddenPixelSize = HiddenPixelSize(1.0),
):
    """Expose raw, response, candidate, and accepted nuclear-admission planes."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a channel stack (C, Y, X), got {array.shape}")
    if not 0 <= channel_index < array.shape[0]:
        raise ValueError("channel_index is outside the input stack")
    pixel_size_um = float(pixel_size)
    settings = MetaXpressWavelengthSettings(
        channel_index=channel_index,
        approx_min_width=approx_min_width,
        approx_max_width=approx_max_width,
        intensity_above_local_background=intensity_above_local_background,
    )
    settings.validate("nuclear")
    source = array[channel_index]
    response = local_background_response(
        source,
        object_width_px=approx_max_width / pixel_size_um,
        bright_objects=True,
    )
    candidate = response >= intensity_above_local_background
    accepted = segment_metaxpress_round_objects(source, settings, pixel_size_um) > 0
    return np.stack(
        (
            source.astype(np.float32, copy=False),
            response.astype(np.float32, copy=False),
            candidate.astype(np.float32, copy=False),
            accepted.astype(np.float32, copy=False),
        )
    )
```

This diagnostic must run through a managed OpenHCS pipeline. Do not invoke it directly.

## Current UI and Viewer State

OpenHCS UI selected plate:

```text
/run/media/ts/0BA20E780BA20E78/blind-neurite-analysis-20260916/blind-full-A018/P001_openhcs
```

Current PlateManager declaration is A035 and points to:

```text
global_output_folder = path_root / 'blind-a018-targeted-A035'
well_filter = ['A14']
```

Current viewer connection:

```text
host = localhost
port = 5555
transport_mode = ipc
viewer = OpenHCS Napari Visualization
```

Current viewer contains only:

- Raw A14 DAPI/FITC image route.
- A035 nuclei ROI route.

Current route keys:

```text
origin_manual_kind_manual_projection_selected_images
origin_manual_kind_manual_projection_analysis_inputs_results_A14_s001_w1_z001_t001_nuclei_step0_rois.roi.zip_shapes
```

Current viewport and DAPI contrast already match the frozen values above.

The A035 nuclei ROI layer reported 1,604 feature rows while the segmentation summary reported 1,486 cells. Similar summary/shape-count discrepancies occurred for earlier candidates. Do not equate Napari feature rows with biological object counts; investigate projection/member semantics if exact reconciliation becomes necessary.

Viewer streaming config key:

```text
napari_streaming_config
```

Using `viewer_config_key='napari'` fails because that key is not registered.

Large multi-ROI stream calls can exceed the five-second MCP response window while Napari continues loading. After a timeout, wait and probe port `5555` before retrying to avoid duplicate layers.

## Viewer Streaming Fixes Already Implemented

Changes were made in the OpenHCS repository at:

```text
/home/ts/code/projects/openhcs
```

Modified areas:

```text
openhcs/core/viewer_streaming_service.py
openhcs/agent/services/plate_streaming_service.py
tests/unit/test_streaming_service.py
tests/unit/agent/test_plate_streaming_service.py
```

Behavior fixed:

- Each ROI artifact now gets a distinct producer identity instead of collapsing into `selected_rois`.
- ROI Shapes receive authoritative source voxel spacing and match the image scale `1.3556`.
- Empty ROI archives preserve producer-index alignment.

Verification already completed:

- 53 focused viewer/plate streaming tests passed.
- Ruff import checks passed.
- Managed viewer validation passed with distinguishable routes, nonzero payloads, no pending updates, no missing payload coordinates, and no duplicate payload coordinates.

Other neurite fixes were also present in:

```text
openhcs/processing/backends/analysis/neurite_outgrowth.py
tests/unit/test_neurite_outgrowth.py
```

An earlier focused neurite test run had 73 passing tests.

The OpenHCS worktree may contain unrelated user/agent changes. Inspect `git status` and diffs before editing. Do not revert, overwrite, or stage unrelated changes.

## Recommended Continuation

1. Run `openhcs_health_check` and confirm the MCP process is current.
2. Load `image_analysis_workflow`, `viewer_review`, and the `openhcs-neurite-visual-qa` skill.
3. Search for `nuclear_admission_diagnostics` in the registry.
4. Register it only if absent, then describe its reflected contract.
5. Create a one-well A14 diagnostic pipeline in a new output directory such as `blind-a018-targeted-A036-diagnostics`.
6. Run the diagnostic through the selected-plate UI workflow, not direct invocation.
7. Stream its raw/response/candidate/accepted channels into a fresh managed viewer.
8. Freeze the exact viewport at center `[0, 1937.1524, 1937.1524]`, zoom `0.65`.
9. Inspect the missed bright center cluster and the elongated false admission in each diagnostic channel.
10. If the bright cluster is absent from the candidate channel, debug `local_background_response` scale/threshold semantics.
11. If present in the candidate channel but absent from accepted labels, debug watershed splitting and short-axis filtering.
12. Add persisted typed diagnostic artifacts to the production implementation if the transient diagnostic proves necessary for repeatable QA.
13. Add focused unit tests for the identified failure before changing segmentation behavior.
14. Re-run only A14 first with a separately named candidate.
15. Promote to the five held-out wells only after nuclei pass the same-coordinate gate with no obvious new false positives.
16. Then review calcein soma admission, candidate neurites, rooted connectivity, ownership, and trace completeness in dependency order.
17. Do not promote product defaults or report completion until all five held-out wells pass native-coordinate QA.

## Historical Evidence Images

Raw-first A14 evidence from the earlier review:

```text
/tmp/opencode/20260921T001853092951Z_napari_5555_OpenHCS_Napari_Visualization.png  # raw DAPI
/tmp/opencode/20260921T001934531092Z_napari_5555_OpenHCS_Napari_Visualization.png  # moderate calcein
/tmp/opencode/20260921T001958832182Z_napari_5555_OpenHCS_Napari_Visualization.png  # neurite-window calcein
/tmp/opencode/20260921T002041652079Z_napari_5555_OpenHCS_Napari_Visualization.png  # rejected A032 soma overlay
/tmp/opencode/20260921T002120936327Z_napari_5555_OpenHCS_Napari_Visualization.png  # rejected A032 trace overlay
/tmp/opencode/20260921T002205851613Z_napari_5555_OpenHCS_Napari_Visualization.png  # A032 nuclei under-admission
```

Candidate nuclear evidence:

```text
/tmp/opencode/20260921T003510141684Z_napari_5555_OpenHCS_Napari_Visualization.png  # A033
/tmp/opencode/20260921T004315057045Z_napari_5555_OpenHCS_Napari_Visualization.png  # A034
/tmp/opencode/20260921T004816649109Z_napari_5555_OpenHCS_Napari_Visualization.png  # A035
```

## Release and PR Context

OpenHCS `0.8.6` was published and verified on GitHub, PyPI, the MCP Registry, and the publication run.

PR #125:

```text
https://github.com/OpenHCSDev/openhcs/pull/125
```

That PR must remain limited to the restored stitched gallery overview:

```text
website/assets/gallery/stitched-neurite-overview.webp
website/assets/gallery/release-media-record.json
website/assets/gallery/source-capture-evidence.json
```

Do not add the analysis or viewer-streaming fixes to PR #125.

## Storage Constraint

The external volume is nearly full. Avoid duplicate analysis output directories and duplicate streams. Always use a new clearly named output only for a scientifically justified candidate or diagnostic, and preserve existing rejected evidence.

## Completion Definition

The task is complete only when all of the following are true:

- The nuclear stage passes at frozen native coordinates without obvious bright misses or fragment admissions.
- Calcein soma masks correspond to broader DAPI-anchored cytoplasm.
- Traces follow continuous soma-connected raw calcein neurites.
- Unsupported traces, background bridges, and obvious missed branches are absent at reviewed fields.
- Route identity and raw/result transforms validate.
- A promoted candidate passes the untouched five-well held-out set.
- Provenance, exact settings, output paths, viewer evidence, and summary measurements are recorded.

Until then, the correct status is: **analysis incomplete; no candidate promoted**.
