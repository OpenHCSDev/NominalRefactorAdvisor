"""Fresh probe build on the saved move: build time vs extracting every expression vs only the asked spans' files."""
import json, sys, time, importlib
from pathlib import Path
from collections import defaultdict
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_facts import ForeignAbsenceProbeDetector, ProbeOwnerQuery
from nominal_refactor_advisor.type_checker_session import SourceTexts, BuiltTypeChecker, TypeCheckerSession
import nominal_refactor_advisor.type_checker_session as tcs
root = Path(sys.argv[1]).resolve(); bench = json.loads(Path(sys.argv[2]).read_text())
config = DetectorConfig()
scan = CodemodWorkflowScan.parsed((root,), config, cache_dir=None, use_parse_cache=True)
after = scan.source_snapshot.with_virtual_sources(bench["sources"]).module_set
groups = ForeignAbsenceProbeDetector.compact_module_projection_groups(after)
t = time.perf_counter(); query = ProbeOwnerQuery.of(groups); print(f"query {time.perf_counter()-t:.2f}s spans={len(query.spans)} files={len(query.file_paths)}", flush=True)
sources = SourceTexts(tuple(m for m in after if m.file_path in query.file_paths))
captured = {}
original_of = BuiltTypeChecker.of
def timed_of(result, options, paths, shared):
    captured["result"] = (result, options, paths, shared)
    t = time.perf_counter(); out = original_of(result, options, paths, shared); captured["all"] = time.perf_counter() - t
    return out
BuiltTypeChecker.of = staticmethod(timed_of)
for run in range(2):
    t = time.perf_counter(); session = TypeCheckerSession.started(sources); total = time.perf_counter() - t
    result, options, paths, shared = captured["result"]
    span_files = frozenset(s.file_path for s in query.spans)
    t = time.perf_counter(); only = original_of(result, options, span_files & paths, shared); spans_s = time.perf_counter() - t
    same = all(session.answer(s) == only.answer(s) for s in query.spans)
    print(f"run {run}: started total {total:.2f}s = build {total-captured['all']:.2f}s + extract all {captured['all']:.2f}s; extract only {len(span_files)} span files {spans_s:.3f}s; answers same={same}", flush=True)
