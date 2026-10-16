# Different-file incremental scan timing

Measured 2026-09-15 on code checkpoint
`93ad35f007379fc156ea6fd24fad057100916d5a`.

## Result

On this corpus, changing a function body costs more than changing a trailing
comment. Comment-only edits in different files show a smaller spread. Four
files edited together do not incur four separate one-edit scan costs.

| Edit | Scan seconds, trial 1 / 2 | Command wall seconds, trial 1 / 2 |
| --- | --- | --- |
| UI help window, trailing comment | 5.322 / 5.361 | 7.78 / 7.84 |
| OpenHCS core config, trailing comment | 5.502 / 5.455 | 7.96 / 7.88 |
| Metaclass registry core, trailing comment | 5.424 / 5.402 | 7.85 / 7.89 |
| ZMQ config, trailing comment | 5.043 / 4.983 | 7.48 / 7.44 |
| ZMQ config, function-body edit | 6.853 / 6.825 | 9.31 / 9.23 |
| All four files, trailing comments | 6.811 / 5.895 | 9.26 / 8.42 |

The fresh cold control takes 48.834 scan seconds / 51.52 wall seconds.
Exact warmed controls generally take approximately one scan second. Three
restoration controls return a cache hit while using the complete compact
global path rather than the exact summary fast path; they do not indicate
an incomplete scan.

The ZMQ comment edit takes approximately 2.0 seconds in parsing and 3.0 in
analysis; its function-body edit takes approximately 2.1 and 4.8 respectively.
The measured additional cost is in analysis. Two trials per case are a small
sample, not a general latency guarantee or a causal benchmark of dependency
fan-out. The four-file case has visibly more variability.

## Scope and controls

The retained full source copy contains 1,057 Python files: OpenHCS plus eight
external libraries, with tests excluded. Every measured report completes
all 79 detectors with zero omissions, 180 retained findings and 215 raw
findings. All twelve edit reports and the cold/warm controls have the same
normalized report SHA-256:
`b081a09b6acdd41cc3f7b4d57af3383a9d32bc1061c210a21d554ed9aab89100`.
Normalization removes timing, payload timing and scan mode/reason only.
Report equality is not behavioral equivalence or detector-recall proof.

Only the disposable source copy was edited; production code and live OpenHCS
processes were not changed. Each trial restores the original text and scans
the baseline before the next edit. A distinct comment for each trial prevents
an exact previous-edit cache hit. Each edit returns `partial`, rather than
`hit`, for the analysis cache. The original single-file source-copy control
is also removed.

Relative edited paths:

- `openhcs/pyqt_gui/windows/help_window.py`
- `openhcs/core/config.py`
- `external/metaclass-registry/src/metaclass_registry/core.py`
- `external/zmqruntime/src/zmqruntime/config.py`

The body edit expands `TransportMode.optional_from_text` from
`return None if value is None else cls(value)` into an explicit `if` returning
`None`, followed by `return cls(value)`. It changes the body AST while keeping
the same branch choice and calls for these operands. The trial-specific
comment is inside the edited body. This is an experiment, not a proposed
production refactor or an execution-equivalence proof.

## Reproduction and retained artifacts

Worktree: `/home/ts/nra-native-proof-integration-20260914`.
Source and artifact root: `/home/ts/nra-native-behavior-validation-G481Dc`.
The measurement command is:

```sh
/usr/bin/time -f '%e' -o "$ARTIFACT_ROOT/edit-location-$STAGE.wall" \
  timeout 60 env NRA_CACHE_HOME="$TASK_CACHE" TMPDIR="$ARTIFACT_ROOT" \
  /home/ts/code/projects/nominal-refactor-advisor/.venv/bin/python \
  -m nominal_refactor_advisor "$ARTIFACT_ROOT/source" \
  --context-root "$ARTIFACT_ROOT/source" --no-auto-context-root \
  --json --json-payload agent --parse-workers 16 --analysis-workers 16 \
  --scan-budget-seconds 60 \
  > "$ARTIFACT_ROOT/edit-location-$STAGE.json" \
  2> "$ARTIFACT_ROOT/edit-location-$STAGE.stderr"
```

Use a newly created task-owned cache for the cold control, whose timeout and
scan budget are 165 seconds. Do not recreate an edit cache hit when repeating
a trial; use a new trial comment. Stages are `cold`, `baseline-warm`,
`ui-leaf-{1,2}`, `domain-config-{1,2}`, `metaclass-core-{1,2}`,
`zmq-config-{1,2}`, `zmq-body-{1,2}` and `four-files-{1,2}`. Restoration and
extra priming reports have `-restored` and `-prime` suffixes respectively.

All scan handles are terminal. Temporary edit comments and the body change
were removed, and the final normalized restoration report matches the cold
control. The task-owned cache is disposable; reports, stderr and wall-time
files are retained for review.
