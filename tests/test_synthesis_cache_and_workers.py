"""Regression tests for synthesis disk-cache reuse and memory-bounded workers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import (
    DetectorAnalysisWorkerPlan,
    analyze_path,
    default_detector_types_for_analysis,
)
from nominal_refactor_advisor.ast_tools import ParsedModuleSet, parse_python_modules
from nominal_refactor_advisor.class_identity_observations import DeclaredIdentityFamily
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    ModuleClassOverlapProjection,
)
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipePlanBuilder,
)
from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
from nominal_refactor_advisor.detectors._base import DetectorCacheGranularity
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_cached_family_batch_loads_from_disk_and_overlap_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pkg"
    cache_dir = tmp_path / "cache"
    _write(
        root,
        {
            "__init__.py": "",
            "models.py": (
                "class Base:\n"
                "    def value(self) -> int:\n"
                "        return 1\n\n"
                "class Child(Base):\n"
                "    pass\n"
            ),
        },
    )
    # Cold module has no cached batch yet and does not compute on cached_family_batch.
    cold = tuple(parse_python_modules(root, cache_dir=cache_dir))
    assert all(
        module.cached_family_batch(CompactModuleClassProjectionFamily) is None
        for module in cold
    )
    # Populate disk cache via normal family_batch.
    for module in cold:
        module.family_batch(CompactModuleClassProjectionFamily)

    # Freshly parsed modules should hit disk via cached_family_batch without running overlap facet derivation.
    fresh = tuple(parse_python_modules(root, cache_dir=cache_dir))
    assert all(
        not module.has_family_batch(CompactModuleClassProjectionFamily)
        for module in fresh
    )

    def _fail_overlap_derive(cls, parsed_module):
        del cls, parsed_module
        raise AssertionError("disk-cached module should not re-derive overlap projection")

    monkeypatch.setattr(
        ModuleClassOverlapProjection, "derive", classmethod(_fail_overlap_derive)
    )
    projections = CompactModuleClassProjectionFamily.collect_overlap_modules(fresh)
    assert len(projections) == len(fresh)


def test_declared_identities_uses_disk_family_cache(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    cache_dir = tmp_path / "cache"
    _write(
        root,
        {
            "__init__.py": "",
            "kinds.py": (
                "class Kind:\n"
                "    pass\n\n"
                "class Alpha(Kind):\n"
                "    pass\n"
            ),
        },
    )
    with ScanCache.scope():
        first_modules = ParsedModuleSet.of(parse_python_modules(root, cache_dir=cache_dir))
        first_identities = CodemodSourceSnapshot.from_modules(first_modules).declared_identities

    family_cache_dir = cache_dir / "collected-family"
    before_entries = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in family_cache_dir.glob("*.pickle")
    }
    assert len(before_entries) == 2

    with ScanCache.scope():
        second_modules = ParsedModuleSet.of(parse_python_modules(root, cache_dir=cache_dir))
        assert all(
            not module.has_family_batch(DeclaredIdentityFamily)
            for module in second_modules
        )
        second_identities = CodemodSourceSnapshot.from_modules(second_modules).declared_identities
        assert all(
            module.has_family_batch(DeclaredIdentityFamily)
            for module in second_modules
        )
    after_entries = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in family_cache_dir.glob("*.pickle")
    }
    assert after_entries == before_entries
    assert second_identities == first_identities


def test_partition_unchanged_cross_detectors_skips_disk_cached_unchanged_families(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pkg"
    cache_dir = tmp_path / "cache"
    _write(
        root,
        {
            "__init__.py": "",
            "a.py": "class Alpha:\n    def f(self) -> int:\n        return 1\n",
            "b.py": "class Beta:\n    def g(self) -> int:\n        return 2\n",
        },
    )
    # Warm disk cache via analyze_path.
    analyze_path(root, cache_dir=cache_dir)

    all_detectors = default_detector_types_for_analysis()
    per_mod = set(DetectorCacheGranularity.PER_MODULE.select_detector_types(all_detectors))
    cross = tuple(det for det in all_detectors if det not in per_mod)

    with ScanCache.scope():
        baseline_modules = tuple(parse_python_modules(root, cache_dir=cache_dir))
        before_by_path = {module.file_path: module for module in baseline_modules}
        target = next(m for m in baseline_modules if m.file_path.endswith("a.py"))
        # Comment-only edit keeps all AST/semantic family batches identical.
        edited = target.with_source(target.source + "# comment\n", persist_cache=False)
        dirty, skipped = CodemodRefactorGoalRunner._partition_unchanged_cross_detectors(
            before_by_path=before_by_path,
            removed_paths=frozenset(),
            changed_or_created=(edited,),
            cross_detectors=cross,
            analysis_workers=1,
        )
    assert len(skipped) > 0
    assert "restated_base" in skipped
    assert len(dirty) < len(cross)


def test_parallel_synthesis_respects_memory_worker_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pkg"
    _write(
        root,
        {
            "__init__.py": "",
            "dups.py": (
                "class A1:\n    def x(self) -> int:\n        return 1\n    def y(self) -> int:\n        return 2\n\n"
                "class A2:\n    def x(self) -> int:\n        return 1\n    def y(self) -> int:\n        return 2\n\n"
                "class B1:\n    def p(self) -> int:\n        return 3\n    def q(self) -> int:\n        return 4\n\n"
                "class B2:\n    def p(self) -> int:\n        return 3\n    def q(self) -> int:\n        return 4\n\n"
                "class C1:\n    def u(self) -> int:\n        return 5\n    def v(self) -> int:\n        return 6\n\n"
                "class C2:\n    def u(self) -> int:\n        return 5\n    def v(self) -> int:\n        return 6\n\n"
                "class D1:\n    def m(self) -> int:\n        return 7\n    def n(self) -> int:\n        return 8\n\n"
                "class D2:\n    def m(self) -> int:\n        return 7\n    def n(self) -> int:\n        return 8\n"
            ),
        },
    )
    findings = [
        f for f in analyze_path(root, use_parse_cache=False) if f.detector_id == "duplicate_class"
    ]
    assert len(findings) >= 4

    def _fail_fork() -> int:
        raise AssertionError("memory_worker_ceiling=1 should not fork")

    monkeypatch.setattr(DetectorAnalysisWorkerPlan, "memory_worker_ceiling", 1)
    monkeypatch.setattr(os, "fork", _fail_fork)
    with ScanCache.scope():
        modules = ParsedModuleSet.of(parse_python_modules(root, use_parse_cache=False))
        snapshot = CodemodSourceSnapshot.from_modules(modules)
        plan = FindingRecipePlanBuilder(findings=findings).plan(selector_context=snapshot)
    assert len(plan.report.records) == len(findings)
