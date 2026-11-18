"""Prototype: one fine-grained mypy build, moved to any state by updating the files that differ."""
import os, stat, sys, time
from collections import defaultdict
from mypy import build as mypy_build
from mypy.fscache import FileSystemCache
from mypy.modulefinder import BuildSource
from mypy.options import Options
from mypy.server.update import FineGrainedBuildManager
from mypy.server.subexpr import get_subexpressions
from mypy.util import hash_digest
from nominal_refactor_advisor.probe_owner import FailedReceiverType, InferredReceiverType, ReceiverSpan
from nominal_refactor_advisor import type_checker_session as tcs
from mypy.messages import format_type_bare

class OverlayFileSystemCache(FileSystemCache):
    def __init__(self):
        super().__init__()
        self.overlay = {}
    def stat_or_none(self, path):
        if path in self.overlay:
            data = self.overlay[path]
            stamp = int(hash_digest(data)[:8], 16)
            return os.stat_result((stat.S_IFREG | 0o644, 0, 0, 1, 0, 0, len(data), stamp, stamp, stamp))
        return super().stat_or_none(path)
    def read(self, path):
        if path in self.overlay:
            data = self.overlay[path]
            self.hash_cache[path] = hash_digest(data)
            return data
        return super().read(path)

def options():
    o = Options()
    o.export_types = True; o.preserve_asts = True; o.strict_equality = True
    o.check_untyped_defs = True; o.follow_imports = "skip"; o.ignore_missing_imports = True
    o.python_version = sys.version_info[:2]
    o.incremental = True; o.fine_grained_incremental = True; o.cache_dir = os.devnull
    o.local_partial_types = True; o.num_workers = 0; o.raise_exceptions = True; o.show_traceback = True; o.native_parser = False
    return o

def unique(modules):
    by_name = defaultdict(list)
    for m in modules:
        by_name[m.module_name].append(m)
    return {ms[0].file_path: (name, ms[0].source) for name, ms in by_name.items() if len(ms) == 1}

class Cursor:
    def __init__(self, modules):
        self.options = options()
        self.fscache = OverlayFileSystemCache()
        self.current = unique(modules)
        for path, (name, text) in self.current.items():
            self.fscache.overlay[path] = text.encode()
        t = time.perf_counter()
        result = mypy_build.build([BuildSource(p, n) for p, (n, _) in self.current.items()], self.options, fscache=self.fscache)
        self.fgm = FineGrainedBuildManager(result)
        self.build_seconds = time.perf_counter() - t
    def move_to(self, modules):
        target = unique(modules)
        changed = [(n, p) for p, (n, text) in target.items() if self.current.get(p) != (n, text)]
        removed = [(n, p) for p, (n, _) in self.current.items() if p not in target or target[p][0] != n]
        for n, p in removed:
            self.fscache.overlay.pop(p, None)
        for n, p in changed:
            self.fscache.overlay[p] = target[p][1].encode()
        t = time.perf_counter()
        if changed or removed:
            self.fscache.flush()
            self.fgm.update(changed, removed)
        self.current = target
        return len(changed), len(removed), time.perf_counter() - t
    def answers(self, spans):
        by_file = defaultdict(list)
        for s in spans:
            by_file[s.file_path].append(s)
        out = {}
        types = self.fgm.manager.all_types
        for path, wanted in by_file.items():
            if path not in self.current:
                for s in wanted: out[s] = FailedReceiverType("not in build")
                continue
            state = self.fgm.graph.get(self.current[path][0])
            index = {}
            for e in get_subexpressions(state.tree):
                found = types.get(e)
                if found is not None:
                    index[ReceiverSpan(path, e.line, e.column, e.end_line, e.end_column)] = found
            for s in wanted:
                found = index.get(s)
                out[s] = (InferredReceiverType(format_type_bare(found, self.options, verbosity=2), tcs._classes_of(found), tcs._instances_only(found), tcs._families_of(found))
                          if found is not None else FailedReceiverType(f"the build exported no type at {s.location}"))
        return out
