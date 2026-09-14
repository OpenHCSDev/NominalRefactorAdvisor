"""Public analysis invocations own declaration reuse and release it on return."""
import inspect
import sys

import pytest

from nominal_refactor_advisor import analysis
from nominal_refactor_advisor.analysis_cache import AnalysisEngineSignature
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.scan_cache import ScanCache


@pytest.mark.parametrize("api", ("analyze_path", "analyze_paths", "analyze_modules_with_cache"))
def test_direct_api_computes_engine_identity_once_per_invocation(tmp_path, api):
    root = tmp_path / "pkg"
    root.mkdir()
    path = root / "source.py"
    path.write_text("VALUE = 1\n")
    modules = [SourceModule(path=path, module_name="pkg.source", source="VALUE = 1\n").parse()]
    invocations = {
        "analyze_path": lambda: analysis.analyze_path(root, cache_dir=tmp_path / "parse"),
        "analyze_paths": lambda: analysis.analyze_paths((root,), cache_dir=tmp_path / "parse"),
        "analyze_modules_with_cache": lambda: analysis.analyze_modules_with_cache(
            (root,), modules, analysis_cache_dir=tmp_path / "analysis",
        ),
    }
    code = inspect.unwrap(AnalysisEngineSignature.current).__code__
    scopes = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is code:
            scopes.append((ScanCache._active.get(), frame.f_locals["cls"]))

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        invocation_owners = []
        for _ in range(2):
            boundary = len(scopes)
            invocations[api]()
            current = scopes[boundary:]
            assert current
            owner = current[0][0]
            assert owner is not None
            assert all(scope is owner for scope, _ in current)
            declarations = [declaration for _, declaration in current]
            assert len(declarations) == len(set(declarations))
            invocation_owners.append(owner)
            assert ScanCache._active.get() is None
    finally:
        sys.setprofile(previous)
    assert invocation_owners[0] is not invocation_owners[1]


def test_direct_detector_analysis_borrows_an_existing_invocation(tmp_path):
    source = SourceModule(path=tmp_path / "source.py", module_name="source", source="VALUE = 1\n").parse()
    with ScanCache.scope():
        owner = ScanCache._active.get()
        assert analysis.analyze_detector_types([source], DetectorConfig(), detector_types=()) == []
        assert ScanCache._active.get() is owner
    assert ScanCache._active.get() is None


def test_failed_path_analysis_releases_invocation(tmp_path):
    path = tmp_path / "broken.py"
    path.write_bytes(b"def broken:\n    pass\n")
    with pytest.raises(SyntaxError):
        analysis.analyze_path(path, cache_dir=tmp_path / "cache")
    assert ScanCache._active.get() is None


@pytest.mark.parametrize("initialize,state_type", (
    (analysis.initialize_detector_analysis_worker, analysis.DetectorAnalysisWorkerState),
    (analysis.initialize_per_module_detector_shard_worker, analysis.PerModuleDetectorShardWorkerState),
))
def test_worker_initialisation_replaces_inherited_invocation(initialize, state_type, monkeypatch):
    monkeypatch.setattr(analysis, "detector_analysis_worker_state", None)
    monkeypatch.setattr(analysis, "per_module_detector_shard_worker_state", None)
    original = ScanCache._active.get()
    try:
        with ScanCache.scope():
            inherited = ScanCache._active.get()
            initialize(state_type((), DetectorConfig()))
            assert ScanCache._active.get() is not inherited
            assert ScanCache._active.get() is not None
    finally:
        ScanCache._active.set(original)
