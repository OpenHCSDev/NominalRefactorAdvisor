"""Bulk fact joins retain proofs without allocation-driven heap rescans."""

import gc
from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import (
    BoundedCompactProjectionManifest,
    analyze_compact_roots_with_cache,
)
from nominal_refactor_advisor.ast_tools import suspend_cyclic_gc
from nominal_refactor_advisor.detectors._runtime import (
    GeneratedBoundarySemanticConstantMirrorDetector,
)


@pytest.mark.parametrize("initially_enabled", (False, True))
@pytest.mark.parametrize("raises", (False, True))
def test_collection_scope_restores_caller_policy(
    initially_enabled: bool, raises: bool
) -> None:
    original = gc.isenabled()
    try:
        (gc.enable if initially_enabled else gc.disable)()
        try:
            with suspend_cyclic_gc():
                assert not gc.isenabled()
                with suspend_cyclic_gc():
                    assert not gc.isenabled()
                    if raises:
                        raise ValueError("batch failed")
                assert not gc.isenabled()
        except ValueError as error:
            assert raises and str(error) == "batch failed"
        assert gc.isenabled() == initially_enabled
    finally:
        (gc.enable if original else gc.disable)()


def test_cached_global_join_preserves_findings_with_bounded_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "generated.py").write_text("# generated file\nPOLICY_ID = 'shared'\n")
    (package / "runtime.py").write_text("POLICY_ID = 'shared'\n")
    kwargs = dict(
        cache_dir=tmp_path / "ast",
        detector_types=(GeneratedBoundarySemanticConstantMirrorDetector,),
    )
    first = analyze_compact_roots_with_cache(
        (package,), analysis_cache_dir=tmp_path / "first", **kwargs
    )
    original = BoundedCompactProjectionManifest._source_family_items
    calls = []

    def checked_load(self, source, family, *, derive_content_identity):
        assert not gc.isenabled()
        calls.append(family)
        return original(
            self, source, family, derive_content_identity=derive_content_identity
        )

    monkeypatch.setattr(
        BoundedCompactProjectionManifest, "_source_family_items", checked_load
    )
    original_policy = gc.isenabled()
    second = analyze_compact_roots_with_cache(
        (package,), analysis_cache_dir=tmp_path / "second", **kwargs
    )
    assert calls
    assert gc.isenabled() == original_policy
    assert second.findings == first.findings
    assert second.projection_count == first.projection_count
