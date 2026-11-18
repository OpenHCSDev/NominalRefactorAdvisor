"""Bench: one saved 7-file move on the saved base. Times the projection, every detector on the projected state
(cold for that state, then again warm), per-module detectors on the changed files only, and the census score."""
import json, sys, time, collections
from pathlib import Path
from nominal_refactor_advisor.analysis import default_detector_types_for_analysis
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.ast_tools import ParsedModuleSet
from nominal_refactor_advisor.detectors._base import DetectorConfig, DetectorCacheGranularity
from nominal_refactor_advisor.refactor_search import RefactorSearch
def rss():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith("VmRSS"))
root = Path(sys.argv[1]).resolve(); bench = json.loads(Path(sys.argv[2]).read_text())
config = DetectorConfig()
t = time.perf_counter(); scan = CodemodWorkflowScan.parsed((root,), config, cache_dir=None, use_parse_cache=True)
print(f"base parse+scan {time.perf_counter()-t:.1f}s findings={len(scan.findings)} rss={rss()}MB", flush=True)
base = scan.source_snapshot
t = time.perf_counter(); after = base.with_virtual_sources(bench["sources"]); modules = after.module_set
print(f"projection {time.perf_counter()-t:.2f}s changed={len(bench['sources'])}", flush=True)
types = default_detector_types_for_analysis()
per_module = set(DetectorCacheGranularity.PER_MODULE.select_detector_types(types))
for label in ("cold", "warm"):
    times = {}
    for d in types:
        t = time.perf_counter(); d().detect(modules, config); times[d] = time.perf_counter() - t
    pm = sum(v for d, v in times.items() if d in per_module); wt = sum(v for d, v in times.items() if d not in per_module)
    print(f"== {label}: all detectors {sum(times.values()):.1f}s (per-module {pm:.1f}s over {len(per_module)}, whole-tree {wt:.1f}s over {len(types)-len(per_module)})", flush=True)
    for d, v in sorted(times.items(), key=lambda kv: -kv[1])[:25]:
        print(f"   {v:6.2f}s {d.cache_granularity.name if hasattr(d.cache_granularity,'name') else d.cache_granularity} {d.__name__}", flush=True)
changed = ParsedModuleSet.of(m for m in modules if m.file_path in bench["sources"])
t = time.perf_counter()
for d in per_module:
    d().detect(changed, config)
print(f"per-module detectors on the {len(changed)} changed files {time.perf_counter()-t:.2f}s", flush=True)
search = RefactorSearch((root,), config)
for label in ("base", "after"):
    target = scan if label == "base" else CodemodWorkflowScan(modules=modules, findings=list(scan.findings))
    t = time.perf_counter(); search.score(target); print(f"census score {label} {time.perf_counter()-t:.1f}s", flush=True)
print("rss", rss(), "MB", flush=True)
