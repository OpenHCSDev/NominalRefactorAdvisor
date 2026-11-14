"""Dump every finding of one tree from a cold cache: stable id, detector, summary, evidence.

Usage: python tools/proof/stable_ids.py ROOT OUT.json

Take the before and the after dump the same way: an absolute ROOT, the package imported
natively, a fresh cache.  Two dumps of one tree by the same code are byte-identical, so a
difference between a before and an after dump is a difference in what the code finds.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nominal_refactor_advisor
from nominal_refactor_advisor.analysis import analyze_path

out = Path(sys.argv[2])
# the analysis cache is written beside the cache directory, so both live in one parent
parent = Path(tempfile.mkdtemp(prefix="nra-cold-", dir=out.parent))
started = time.perf_counter()
try:
    findings = analyze_path(Path(sys.argv[1]), cache_dir=parent / "cache")
finally:
    shutil.rmtree(parent, ignore_errors=True)
rows = sorted(
    [
        finding.stable_id,
        finding.detector_id,
        finding.summary,
        [[e.file_path.rsplit("/", 1)[-1], e.line, e.symbol] for e in finding.evidence],
    ]
    for finding in findings
)
out.write_text(json.dumps(rows, indent=0))
print(
    "code:", Path(nominal_refactor_advisor.__file__).parent.parent.name,
    "| findings:", len(rows),
    "| distinct stable ids:", len({row[0] for row in rows}),
    "| detectors:", len({row[1] for row in rows}),
    f"| {time.perf_counter() - started:.0f}s",
)
