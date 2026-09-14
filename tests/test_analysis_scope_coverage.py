"""Equivalent full-report scopes share global execution and cache identity."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import AnalysisPathScope
from nominal_refactor_advisor.analysis_cache import AnalysisCacheIdentity
from nominal_refactor_advisor.ast_tools import parse_python_module_roots
from nominal_refactor_advisor.detectors import DetectorConfig


@pytest.mark.parametrize("report_kind", ("same", "parent", "reordered", "redundant"))
def test_full_coverage_has_no_focused_report_boundary(
    tmp_path: Path, report_kind: str
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        root.mkdir()
        (root / "source.py").write_text("class Owner: pass\n")
    reports = {
        "same": roots,
        "parent": (tmp_path,),
        "reordered": roots[::-1],
        "redundant": (*roots, roots[0] / "source.py"),
    }[report_kind]
    explicit = AnalysisPathScope.from_requested_roots(reports, roots)
    direct = AnalysisPathScope(roots, reports)
    assert explicit == direct == AnalysisPathScope(roots)
    assert not direct.has_report_filter
    assert direct.resolved_report_roots == ()
    modules = tuple(parse_python_module_roots(roots, use_parse_cache=False))
    config = DetectorConfig()
    assert AnalysisCacheIdentity.from_modules(
        roots, modules, config, report_roots=explicit.report_roots
    ) == AnalysisCacheIdentity.from_modules(roots, modules, config)


def test_partial_coverage_still_requires_full_context(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        root.mkdir()
    scope = AnalysisPathScope(roots, (roots[0],))
    assert scope.has_report_filter
    assert scope.analysis_roots == roots
    assert scope.includes_report_path(roots[0] / "source.py")
    assert not scope.includes_report_path(roots[1] / "source.py")


def test_file_report_does_not_cover_its_package(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("class Owner: pass\n")
    scope = AnalysisPathScope((tmp_path,), (source,))
    assert scope.has_report_filter
    assert scope.includes_report_path(source)
    assert not scope.includes_report_path(tmp_path / "other.py")


def test_file_context_can_be_fully_reported(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("class Owner: pass\n")
    assert not AnalysisPathScope((source,), (source,)).has_report_filter


def test_symlink_report_coverage_preserves_analysis_spelling(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    scope = AnalysisPathScope((alias,), (target,))
    assert scope.analysis_roots == (alias,)
    assert not scope.has_report_filter
