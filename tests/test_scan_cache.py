"""Scan lifetime owns declaration reuse; module lifetime still releases ASTs."""

import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import multiprocessing
import os
from types import ModuleType
import sys
import weakref

import pytest

from nominal_refactor_advisor.analysis import release_module_analysis_memory
from nominal_refactor_advisor.analysis_cache import (
    AnalysisEngineSignature,
    detector_module_source_hash,
)
from nominal_refactor_advisor.ast_tools import (
    BuilderCallShapeFamily,
    module_syntax_index,
)
from nominal_refactor_advisor.implementation_identity import (
    ImplementationSource,
    declaration_implementation_module_names,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def test_no_scope_never_retains_results():
    @ScanCache.cached
    def construct(value):
        return [value]

    assert construct(1) == construct(1)
    assert construct(1) is not construct(1)


def test_nested_scope_shares_until_outer_exit():
    @ScanCache.cached
    def construct(value):
        return [value]

    with ScanCache.scope():
        first = construct(1)
        with ScanCache.scope():
            assert construct(1) is first
            assert construct(2) is not first
        assert construct(1) is first
    with ScanCache.scope():
        assert construct(1) is not first


def test_exception_releases_retained_results():
    class Value:
        pass

    @ScanCache.cached
    def construct():
        return Value()

    with pytest.raises(RuntimeError):
        with ScanCache.scope():
            retained = weakref.ref(construct())
            assert retained() is not None
            raise RuntimeError("leave scope")
    gc.collect()
    assert retained() is None


def test_module_cleanup_keeps_declarations_but_releases_ast():
    with ScanCache.scope():
        original = BuilderCallShapeFamily.implementation_identity()
        tree = ast.parse("def action(): return 1")
        tree_ref = weakref.ref(tree)
        module_syntax_index(tree)
        release_module_analysis_memory()
        assert BuilderCallShapeFamily.implementation_identity() is original
        assert module_syntax_index.cache_info().currsize == 0
        del tree
        gc.collect()
        assert tree_ref() is None
    with ScanCache.scope():
        assert BuilderCallShapeFamily.implementation_identity() is not original


def test_source_content_refreshes_between_scans_even_with_same_stat(
    tmp_path, monkeypatch
):
    path = tmp_path / "implementation.py"
    path.write_text("value = 1\n")
    module = ModuleType("scan_cache_fixture")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with ScanCache.scope():
        first = ImplementationSource.from_module_name(module.__name__)
        assert ImplementationSource.from_module_name(module.__name__) == first
    stamp = path.stat()
    path.write_text("value = 2\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with ScanCache.scope():
        second = ImplementationSource.from_module_name(module.__name__)
    assert first.source_signature != second.source_signature


def test_changed_declaration_dependencies_refresh_between_scans():
    module = ModuleType("scan_cache_fixture")
    exec(
        "class Payload: pass\nclass Root:\n    def action(self): return Payload()",
        vars(module),
    )
    first_payload = module.Payload
    first_payload.__module__ = "first_dependency"
    with ScanCache.scope():
        first = declaration_implementation_module_names((module.Root,))
        assert "first_dependency" in first
    exec("class Payload: pass", vars(module))
    module.Payload.__module__ = "second_dependency"
    with ScanCache.scope():
        second = declaration_implementation_module_names((module.Root,))
    assert "second_dependency" in second
    assert "first_dependency" not in second


def test_engine_identity_reused_per_owner_and_refreshed_between_scans(
    tmp_path, monkeypatch
):
    path = tmp_path / "engine.py"
    path.write_text("value = 1\n")
    module = ModuleType("scan_engine_fixture")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Engine(AnalysisEngineSignature):
        @staticmethod
        def module_names():
            return (module.__name__,)

    class OtherEngine(Engine):
        pass

    with ScanCache.scope():
        original = Engine.current()
        release_module_analysis_memory()
        assert Engine.current() is original
        assert type(OtherEngine.current()) is OtherEngine
    stat = path.stat()
    path.write_text("value = 2\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with ScanCache.scope():
        current = Engine.current()
    assert current != original
    assert current.source_files[0].source_hash != original.source_files[0].source_hash


def test_detector_implementation_hash_refreshes_between_scans(tmp_path, monkeypatch):
    path = tmp_path / "detector.py"
    path.write_text("value = 1\n")
    stat = path.stat()
    module = ModuleType("detector_hash_fixture")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    declaration = type("Detector", (), {"__module__": module.__name__})
    with ScanCache.scope():
        original = detector_module_source_hash(declaration)
        assert detector_module_source_hash(declaration) == original
    path.write_text("value = 2\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with ScanCache.scope():
        assert detector_module_source_hash(declaration) != original


def test_threads_have_independent_scopes():
    @ScanCache.cached
    def construct():
        return object()

    def job():
        with ScanCache.scope():
            result = construct()
            assert construct() is result
            return result

    with ScanCache.scope():
        parent = construct()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = tuple(pool.map(lambda _: job(), range(4)))
        assert all(result is not parent for result in results)
        assert len({id(result) for result in results}) == len(results)


@ScanCache.cached
def cached_pid():
    return os.getpid()


def worker_identity(_):
    return cached_pid(), os.getpid()


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_worker_scope_does_not_reuse_parent_cache(method):
    with ScanCache.scope():
        parent = cached_pid()
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context(method),
            initializer=ScanCache.initialize_worker,
        ) as pool:
            results = tuple(pool.map(worker_identity, range(4)))
        assert all(cached == actual and actual != parent for cached, actual in results)
        assert cached_pid() == parent
