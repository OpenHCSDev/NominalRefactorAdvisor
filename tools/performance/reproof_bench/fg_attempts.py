"""Two update orders for the fine-grained build on the saved move, then mypy's cached incremental mode on it."""
import json, os, sys, time, shutil, tempfile, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fg_cursor
from fg_cursor import Cursor, unique, options
from mypy import build as mypy_build
from mypy.modulefinder import BuildSource
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.detectors._base import DetectorConfig
root = Path(sys.argv[1]).resolve(); bench = json.loads(Path(sys.argv[2]).read_text())
config = DetectorConfig()
scan = CodemodWorkflowScan.parsed((root,), config, cache_dir=None, use_parse_cache=True)
base_modules = tuple(scan.modules)
after = tuple(scan.source_snapshot.with_virtual_sources(bench["sources"]).module_set)
target = unique(after)
def attempt(label, batches_of):
    cursor = Cursor(base_modules)
    current = cursor.current
    changed = [(n, p) for p, (n, text) in target.items() if current.get(p) != (n, text)]
    for p in [p for _, p in changed]:
        cursor.fscache.overlay[p] = target[p][1].encode()
    t = time.perf_counter()
    try:
        for batch in batches_of(changed):
            cursor.fscache.flush()
            cursor.fgm.update(batch, [])
        print(f"{label}: ok in {time.perf_counter()-t:.2f}s, messages={len(cursor.fgm.previous_messages)}", flush=True)
    except BaseException as error:
        print(f"{label}: {type(error).__name__} after {time.perf_counter()-t:.2f}s: {str(error)[:200]}", flush=True)
# mypy's ordinary cached incremental mode
cache = tempfile.mkdtemp(prefix="nra-mypy-cache-")
def incremental_build(modules):
    o = options(); o.fine_grained_incremental = False; o.cache_dir = cache; o.incremental = True; o.local_partial_types = False
    fs = fg_cursor.OverlayFileSystemCache()
    texts = unique(modules)
    for p, (n, text) in texts.items():
        fs.overlay[p] = text.encode()
    t = time.perf_counter()
    result = mypy_build.build([BuildSource(p, n) for p, (n, _) in texts.items()], o, fscache=fs)
    rechecked = sum(1 for s in result.graph.values() if s.tree is not None and s.xpath in texts)
    print("   stale SCC sizes:", sorted((len(scc) for scc in []), reverse=True), flush=True)
    return time.perf_counter() - t, rechecked, len(texts)
print("incremental base build %.1fs rechecked %d of %d" % incremental_build(base_modules), flush=True)
print("incremental moved build %.1fs rechecked %d of %d" % incremental_build(after), flush=True)
shutil.rmtree(cache)
