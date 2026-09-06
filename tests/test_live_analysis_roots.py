"""Live CLI roots bind to their working directory before cache projection."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.analysis import AnalysisPathScope
from nominal_refactor_advisor.analysis_cache import AnalysisCacheIdentity
from nominal_refactor_advisor.ast_tools import parse_python_module_roots
from nominal_refactor_advisor.cache_checkout import (
    CacheCheckoutPathError,
    checkout_relative_path,
)
from nominal_refactor_advisor.detectors import DetectorConfig


def _sources(root: Path) -> None:
    for name in ("first", "second"):
        package = root / name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "source.py").write_text("class Owner: pass\n", encoding="utf-8")


def test_live_scope_binds_relative_roots_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sources(tmp_path)
    monkeypatch.chdir(tmp_path)
    roots = (Path("first"), Path("second"))
    report = (Path("first/source.py"),)
    scope = AnalysisPathScope.from_requested_roots(report, roots)
    assert scope.analysis_roots == tuple(tmp_path / root for root in roots)
    assert scope.report_roots == (tmp_path / report[0],)
    modules = tuple(parse_python_module_roots(scope.analysis_roots))
    identity = AnalysisCacheIdentity.from_modules(
        scope.analysis_roots,
        modules,
        DetectorConfig(),
        report_roots=scope.report_roots,
    )
    assert identity.report_filter_roots == ("0:source.py",)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert scope.analysis_roots == tuple(tmp_path / root for root in roots)
    assert scope.includes_report_file_path(str(tmp_path / report[0]))
    with pytest.raises(CacheCheckoutPathError, match="ambiguous"):
        checkout_relative_path("source.py", scope.analysis_roots)


def test_relative_multi_root_cli_can_reuse_its_cache(tmp_path: Path) -> None:
    _sources(tmp_path)
    command = [
        sys.executable,
        "-m",
        "nominal_refactor_advisor",
        "first/source.py",
        "--context-root",
        "first",
        "--context-root",
        "second",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--json",
        "--json-payload",
        "summary",
    ]
    for cache_status in ("miss", "hit"):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["findings"] == []
        assert payload["timing"]["analysis_cache_status"] == cache_status


@pytest.mark.parametrize("file_request", (False, True), ids=("directory", "file"))
def test_live_root_binding_preserves_a_source_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_request: bool
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "source.py").write_text("class Owner: pass\n", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Source symlink creation unavailable: {error}")
    monkeypatch.chdir(tmp_path)
    request = Path("alias/source.py") if file_request else Path("alias")
    scope = AnalysisPathScope.from_requested_roots((request,))
    assert scope.analysis_roots == (alias,)
    assert scope.analysis_roots != (target,)
    identity = AnalysisCacheIdentity.from_modules(
        scope.analysis_roots,
        tuple(parse_python_module_roots(scope.analysis_roots)),
        DetectorConfig(),
        report_roots=scope.report_roots,
    )
    assert identity.report_filter_roots == (("0:source.py",) if file_request else ())
