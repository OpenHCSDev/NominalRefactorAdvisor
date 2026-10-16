# OpenHCS Paper and Figures Handoff

Updated 2026-10-14 (Toronto; build timestamps below are UTC).

## Current Request

Tristan asked for this handoff so another agent can take over. Stop manuscript/figure work here and continue from the state below. The active project is **OpenHCS**, not OpenMFD and not NRA. This file is in NRA only because Tristan requested that location.

His latest substantive request has two parts, explicitly confirmed as **both in parallel**:

1. Improve and rerun the actual neurite-outgrowth image analysis overnight. He finds the current analysis quality unsatisfactory. Do not treat a prettier plot as satisfying this request.
2. Redraw the paper's figures from the existing results while that analysis work proceeds. Figure 6 is oversized, sparse and hard to read. Bring the useful coverage and throughput information into a tighter main-results presentation, with individual workflow points, larger text and smaller, cleaner panels assembled together.

**Neither new scientific analysis nor figure edits have started.** Only source inspection and one read-only neurite investigator were launched before the handoff request. All manuscript edits described below are already committed, pushed and rendered.

## Current Files and Delivery

- Main checkout: `/home/ts/code/projects/openhcs`, branch `main`, remote `openhcsdev` (`https://github.com/OpenHCSDev/openhcs.git`).
- Latest committed and pushed source: `8f75047c7`, `Balance abstract and explain OpenHCS purpose in plain language`.
- Main manuscript source: `paper/manuscript.md`.
- Supplement source: `paper/supplementary/README.md`.
- Build entrypoint: `paper/build_paper.py`.
- `paper/current` points to `build/run-20261015T013847-8354c9c8`.
- Current manuscript PDF/DOCX: `paper/current/openhcs_manuscript.pdf` and `openhcs_manuscript.docx`.
- Current supplement PDF/DOCX: `paper/current/openhcs_supplement.pdf` and `openhcs_supplement.docx`.
- Current rendered lengths: manuscript 26 pages; supplement 44 pages.
- OpenHCS had no tracked modifications when the figure request arrived. There are many unrelated untracked assets; preserve them and do not bulk-add.
- NRA already has a modified `uv.lock`; it is unrelated and must be left alone.

## Completed Writing Work

The paper targets the SLAS Technology special issue on laboratory automation discussed earlier with Tristan. Do not narrow the contribution to self-driving laboratories or neuroscience merely because those are applications.

The agreed main argument is:

> Researchers know the biological question but may lack the expertise to develop reliable image analysis. Fragmented tools add integration work. A scientist describes what needs measuring; an agent develops the analysis; the scientist can inspect and correct the same executable method.

The abstract, introduction, Discussion and closing paragraph now follow that argument. Self-driving laboratories remain one application. The introduction explicitly states the purpose, rather than making the reader infer it from an architecture inventory.

Recent commits, newest first:

- `8f75047c7`: reduced the abstract from 241 to 219 source words; retained representative nuclear, neurite, output-agreement and execution results; explained technical points in plain context; moved CellProfiler dependency versions from the introduction into performance Methods.
- `de066c506`: carried the access-to-analysis framing through abstract, introduction, Discussion and conclusion; moved agent findings earlier in Discussion.
- `bfb0da48f`: changed the abstract's main problem from self-driving laboratories to access to expertise and fragmented tooling.
- `7b6695c68`: foregrounded delegated analysis and autonomous results in the abstract.
- `ccf695930`: earlier broad prose cleanup, shorter manuscript and performance-focused presentation.
- `58db20945`: earlier comparison corrections and fixed-workload scaling presentation.

Earlier cleanup removed roughly 1,175 words and reduced the manuscript from 29 to 26 pages. Detailed CI material was moved to the supplement. Three author-declaration placeholders remain; do not invent contributions, funding or conflicts. Check the current source before relying on the earlier cleanup counts.

No novice-user study has been performed. The demonstrated result is that agents developed analyses from biological briefs. Keep the broader accessibility purpose explicit without claiming measured novice usability or universal expert-equivalent analysis quality.

Tristan's writing preferences are important: plain English, direct facts, no repeated obvious explanations, no lawyer-like qualifications, no generic defensive warnings about claims the paper never makes. Use **fragmented**, not fragile. Explain what a technical detail lets the researcher do, or what integration work it removes. Preserve the software contribution rather than turning this into a biological-discovery paper.

## Latest Figure Request

Tristan supplied these references:

- Current Figure 6 screenshot: `/tmp/codex-clipboard-tVbs2H.png`. It shows the single-core bar chart and worker-scaling bar chart split across two pages, with large empty areas and small plot text. This clipboard file is volatile; the current PDF is the durable reference.
- Historical coverage image: `/home/ts/code/projects/openhcs/benchmark/results/labmeeting_20260513/official30_well_throughput/figures/06_module_coverage_by_abstraction.png`.
- Historical multicore image: `/home/ts/code/projects/openhcs/benchmark/results/labmeeting_20260513/official30_well_throughput/figures/05_speedup_summary_by_core_and_wells_per_core_log.png`.

His complaints and requested changes:

- The current enormous mean bars convey very little for their size.
- Show individual workflow points, especially in the throughput comparisons.
- Include useful module-coverage and multicore-throughput information in the main results rather than burying it.
- Make text larger and plots physically smaller; assemble related panels cleanly together rather than giving each sparse panel most of a page.
- Improve neurite analysis itself and redraw its panels in parallel.

Do not copy the historical numbers or old coverage categories into the current paper. Those images are references for the information he wants to see, not permission to substitute old results. The current sweep distinguishes measured CellProfiler 1/8-assignment batches from projected 12/16-assignment references. Preserve that distinction visibly.

### Existing Figure Owners

- `paper/figures/build_slas_benchmark.py`: reads qualified saved benchmark records, generates claims and measured figures through `benchmark.reports.cppipe_figures`. It does not rerun scientific benchmarks.
- `benchmark/reports/cppipe_figures.py`: owns benchmark metric/source types and shared plotting (`FIGURE_STYLE`, `generate_average_point_figures`, and related methods). Read the declarations before changing plot behavior.
- `paper/figures/build_slas_benchmark_reference.py`: rebuilds reference chart forms from the qualified sweep; `build_reference_figures()` combines seven available configurations of all 30 workflows. It labels measured/projected native baselines. `build_module_test_coverage()` owns the current evidence-graded coverage rendering.
- `paper/figures/build_slas_supplement.py`: `benchmark_publication()` currently wraps existing raster charts into separate `submission_benchmark_speed`, `submission_benchmark_schedule` and `submission_benchmark_coverage` assets. This extra composition contributes to small text and wasted space. Read the shared figure owner before adding another layout layer.
- `paper/figures/build_slas_visual_story.py`: `FigureSheet` owns composition/provenance; `submission_neurite_results()` assembles current Figure 5.
- `paper/figures/build_slas_neurite_views.py`: frozen microscopy views and display transforms.
- `paper/figures/build_slas_neurite_effects.py`: existing neurite treatment-effect plotting.
- `paper/figures/compare_personal_neurite.py`: existing well-level comparisons and aggregation.

### Current Benchmark Sources

- `paper/figures/slas/benchmark-publication/benchmark_claims.json`: canonical generated numerical claims. The manuscript resolves `[pending]{.benchmark-claim key=...}` spans from this file. Do not hardcode or independently recalculate a competing claim table.
- `paper/figures/slas/benchmark-publication/measured_benchmark_publication_log.png`: current single-sample execution/compile-plus-run artwork.
- `paper/figures/slas/benchmark-publication/fixed12/execution/scaling/derived_workflow_metrics.csv`: current fixed-workload execution-scaling rows.
- `paper/figures/slas/benchmark-publication/reference-layout/reference_pipeline_metrics.csv`: combined qualified comparison rows.
- `paper/figures/slas/benchmark-publication/reference-layout/reference_module_coverage.csv`: current module evidence tiers.
- Corresponding `figure2_provenance.json` / retained coverage records verify source and output hashes. Do not silently replace them with historical inputs.

Current main Figure 6A is single-core CellProfiler-relative execution and compile-plus-run speedup. Current 6B is measured OpenHCS scaling with **twelve samples held fixed**, comparing 1/2/3/4 workers. The supplement has the mixed-assignment CellProfiler comparisons and current module-coverage figure. Coverage is graded evidence, not proof that every declared module has algorithm-specific validation.

## Neurite Analysis State

Current Figure 5 combines three different sources:

1. Public NeuronCyto II field: raw image, **initial** OpenHCS output, and published algorithm output. The published output is not manual ground truth.
2. Earlier nine-field autonomous laboratory P001 analysis: matched raw, soma/path and combined views. This is separate from the later treatment cohort.
3. Three independent blind authors' treatment responses and MetaXpress comparisons, normalized to each drug curve's own DMSO controls.

Important paths:

- `paper/figures/slas/submission_neurite_results_provenance.json`: exact source files, image transforms, cohorts and output hashes for Figure 5.
- `paper/figures/slas/frozen_neurite_views.json`: frozen views.
- `paper/supplementary/personal_neurite_autonomous/cohort.json`: three treatment authors' original pipelines, reports, model/skill/software identities, freeze times and original run locations.
- `paper/supplementary/personal_neurite_autonomous/author114/`, `author115/`, `author116/`: retained pipelines, joined wells, effects and source evidence.
- `paper/supplementary/personal_neurite_repaired_morphometry/`: earlier scientist-assisted comparison, not autonomous success.
- Original three-author run root recorded by the cohort: `/home/ts/wt/openhcs-issue-batch-20260929/next-personal-neurite-graphical-three-20261008/`. Verify availability and active ownership before using it; do not edit another owner's checkout.

The current cohort analysed 20 coded wells, nine sites per well, DAPI/FITC, 1024 x 1024 uint16 images, XY scale 1.3556 micrometers. All three recovered both positive drug responses, but outgrowth/branching response magnitudes were smaller than MetaXpress and anatomical recovery is not established by correlation.

The actual cohort dispositions are stronger cautions than a success-only summary would suggest: Blind 1 was conditional/descriptive; Blind 2 abstained from validated census and complete anatomical arbors; Blind 3 was rejected for complete biological specification while exploratory algorithm measurements were retained. Read their original QA/reports and inspect matched raw-plus-overlay views before calling the biological analysis adequate.

Preserve all frozen trials. A repaired pipeline is a **development/assisted result**, not a retroactive improvement to a blind trial. Improve the canonical packaged skill from observed failures, then run fresh independent authors with only brief, MCP and packaged skill if claiming new autonomous success. Do not tune against MetaXpress responses or show treatments/reference answers to blind authors before freeze. Preserve curve-specific zero-dose DMSO controls and the existing unweighted mean of nine field-level values; do not silently change aggregation.

Standing project instructions call for at least three independent SOL 6.1 analysis authors concurrently when measured resources permit, plus a separate cleanup/monitoring owner. Read `/home/ts/.codex/skills/use-openhcs/SKILL.md` and call the actual OpenHCS MCP health check before operating live analysis. No new live health check or rerun has been performed in this task.

NRA also contains `OPENHCS_NEURITE_ANALYSIS_HANDOFF.md`, but it describes older September A035-era diagnostic work and links to a later takeover status. It is **not** the current October three-author cohort. Do not mistake its old "current" state for this paper's current results.

## Investigator and Resource State

A read-only investigator was launched immediately before Tristan requested the handoff:

- Nickname: Ohm.
- Agent ID: `01a13d3f-28d7-7df2-8bff-b9438a46d602`.
- Assignment: identify visible neurite defects, current panel provenance and existing run/launch owners; no edits, paid agent launches or scientific reruns.
- On the handoff request, it was interrupted and told to return current findings and stop. It must not continue launching work independently.

The investigator has now completed. Its findings below come from retained reviews and source declarations; it **did not independently open the microscopy bitmaps** before interruption. The receiving agent still needs to inspect the actual raw/result/combined images.

### Investigator Findings and Launch Paths

- Public neurite review: `paper/supplementary/task_only_analysis/h004-fresh20-qualified-completion.rst`. Faint soma-connected shafts were lost at the candidate gate. Relaxing it recovered paths but added uncertain twigs and changed crossing ownership; branches rose from 12 to 234 with eight somata unchanged. The last chronological `repair03` was not accepted as a shaft result; the author's conservative preference was `repair01`. Figure 5 shows FIRST, not either repair.
- Laboratory nine-field review: `figure-collection-20261004/P001-FRESH13-NINE-FIELD-REVIEW.rst`. It documents missing fine branches, unresolved partitions in broad lobed bodies and uncertain crossing ownership. Acquisition overlap prevents treating the summed fields as a deduplicated neuronal census.
- Treatment-author reports: `paper/supplementary/personal_neurite_autonomous/author114/author-report.txt`, and corresponding files under `author115` and `author116`. Reported defects include missing dim extensions/gaps, punctate or body-margin spurs, crowded-body ambiguity, short texture-associated paths, tiny shaft spurs and soma contours absorbing proximal shafts.
- Public original control root: `/home/ts/wt/openhcs-issue-batch-20260929/next-h004-retina-fresh20-95-96-20261005/H004_FRESH20_95/author-workspace/output`.
- Public original payload root: `/run/media/ts/hdd/openhcs-science/next-h004-retina-fresh20-95-96-20261005/H004_FRESH20_95`. Selected input is `paired_input/field_w1.tif`; selected outputs are under `first/paired_input_openhcs/results/`. Original `REPORT.rst`, `REPRODUCE.rst` and `FINAL-ATTEMPT.json` identify the attempts.
- Laboratory original control root: `/home/ts/wt/openhcs-issue-batch-20260929/next-bbbc01388-p00196-fresh13-20261005/P001_FRESH13_96/author-workspace`.
- Laboratory original payload root: `/run/media/ts/hdd/openhcs-science/next-bbbc01388-p00196-fresh13-20261005/P001_FRESH13_96`. Selected raw is `staged/P001/A01_s001_w2_z001_t001.tif`; final outputs are under `analysis/final-attempt/P001_openhcs/results/`. `FINAL-MANIFEST.json` records the final pipeline; the investigator had not resolved its exact filename.
- Treatment run declaration: `/home/ts/wt/openhcs-issue-batch-20260929/next-personal-neurite-graphical-three-20261008/program.json`.
- Treatment payload root: `/run/media/ts/hdd/openhcs-science/personal-neurite-graphical-three-20261008`, with one directory per author.
- Existing launcher family: `scripts/blind_analysis/operations/` in OpenHCS. `project-program.sh` prepares/publishes successor runs; `slot-env.sh` resolves declared members; `launch-author.sh` launches authors; `recorded-mcp.sh` retains the client; `resource-check.sh` observes resources. Read these owners rather than inventing a new overnight launcher.

Still unresolved: exact prospective brief and input-root paths; the currently authorized run configuration and membership; whether the installed launch/runtime family is ready; and whether another agent already owns the repair. A targeted ownership lookup was interrupted, so absence of an owner is **not established**. Old viewer/native processes still reference `/home/ts/wt/openhcs-neurite-repair-20261007`; do not delete or edit that checkout or assume those processes prove active repair. Resolve ownership before overlapping work.

Latest `/home/ts/bin/agent-resource-check --assert-headroom` returned a **warning**, not a critical stop: available RAM 13.7 GiB, swap used 15.6 GiB, `/home` free 8.5 GiB. Right-size any overnight run from measured memory and disk requirements; do not assume there is space for several full retained-output cohorts. No analysis has started, so there is no original rerun process to resume or replay.

## Building and Checking

From `/home/ts/code/projects/openhcs`:

```sh
/home/ts/code/projects/papers/.venv/bin/python paper/build_paper.py build
```

- The build validates both manuscript and supplement and publishes `paper/current` only when the pair succeeds and source inputs are unchanged.
- Finish source edits **before** building. A provisional export was rejected because I corrected wording during its run; the subsequent finished-source build succeeded. Do not repeat that wasted build.
- The build environment has `pypdf`; system Python has lxml/UNO. Figure dependencies are available in the OpenHCS `.venv`.
- Use the existing figure-generation entrypoints and their provenance contracts. Inspect their command-line options and build integration before running; do not regenerate every scientific figure unnecessarily.
- For the completed prose edits, `git diff --check`, section/citation preservation checks, paired build and PDF text checks passed. Pages 1, 2, 3 and 10 were visually inspected; no clipping/overlap was found. The numerical speed claim rendered as 4.484-fold.
- No scientific measurements were rerun for the prose changes. No new figure acceptance has happened.
- New builds can take around one to two minutes. Follow the original process handle; do not relaunch merely because observation timed out.

## Immediate Next Work

Take over the figure request and the overnight analysis as separate parallel work, with one owner for each. Inspect the current figures, their saved metric rows and the retained biological overlays before editing. Make Figure 6 compact and readable while preserving measured/projected distinctions; draw individual workflow points rather than relying on giant aggregate bars. Redraw Figure 5 from existing results without relabeling initial or assisted outputs as new blind results. Carry biological repairs through actual execution and raw-plus-overlay review; evaluate fresh autonomous authors separately. Rebuild the paired documents only after figure sources, captions and provenance are consistent.

## Earlier OpenMFD Work

This conversation also covered a different manuscript. It is background, not authorization to resume that project instead of the OpenHCS figure request:

- Checkout `/home/ts/code/projects/mfd`; latest commit is `757f853`, `Trim repeated OpenMFD summaries while preserving frame discussion`.
- OpenMFD V9-to-V10 source workflow: `papers/mfd_platform/revisions/build_v10.py` and `v10_notes.json`; generated clean/tracked Word documents and PDFs are under `papers/mfd_platform/build/`.
- Tristan accepted a biology-led opening that leads to a general fabrication platform: neuroscience is an example application, not the boundary of the contribution. The platform adapts different device layouts to multiwell plate layouts with fine channels and standard pipetting/imaging.
- He requested a generic Word-document comparison workflow, localized tracked edits and plain comments only on changes. Earlier work repaired the shared comparison helper in `/home/ts/code/projects/papers`; do not replace it with whole-paragraph rewrite tracking.
- He explicitly retained the frame discussion and accepted the other proposed prose cuts. Abstract mold-production numbers and end-of-abstract assay detail had been removed. Do not reinstate them while working on OpenHCS.

Tristan does not need another plan/report loop. This file is the requested handoff; proceed with the actual work from here.
