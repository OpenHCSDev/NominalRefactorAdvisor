"""A cold shard reuses the full-family collector's single cache publication."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import (
    BoundedCompactProjectionManifest,
    CompactFamilyProjectionReceipt,
    CompactProjectionBuildRequest,
    CompactProjectionCacheSource,
    build_compact_projection_shard,
)
from nominal_refactor_advisor.ast_tools import (
    CollectedFamilyCacheContext,
    PythonModuleRootParser,
    PythonSourcePathPolicy,
    collect_family_batch,
    collect_family_items,
    collected_family_items_content_signature,
)
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.product_flow import (
    CompactProductFlowModuleProjectionFamily,
)
from nominal_refactor_advisor.source_identity import python_source_cache_signature


@pytest.mark.parametrize("use_cache", (False, True))
@pytest.mark.parametrize("explicit_identity", (False, True))
@pytest.mark.parametrize("publication_fits", (False, True))
def test_full_family_is_published_once_per_shard(
    tmp_path: Path, monkeypatch, use_cache, explicit_identity, publication_fits
):
    path = tmp_path / "source.py"
    source = "def target(): pass\nalias = target\nalias()\n"
    path.write_bytes(source.encode())
    parser = PythonModuleRootParser.for_root(
        tmp_path, cache_dir=tmp_path / "cache", use_parse_cache=use_cache
    )
    request_source = CompactProjectionCacheSource(
        path=path,
        module_name="application.source" if explicit_identity else "source",
        source_signature=python_source_cache_signature(source),
        family_cache_dir=(
            tmp_path / "requested-family-cache"
            if explicit_identity and use_cache
            else parser.collected_family_cache_dir
        ),
        scan_root=tmp_path,
        cache_dir=parser.parse_cache_dir,
        use_parse_cache=use_cache,
        source_policy=PythonSourcePathPolicy(),
    )
    original = CollectedFamilyCacheContext.store_items
    publications = []

    def observed_store(self, family, items, demand_signature=""):
        publications.append((self, family, items))
        return original(self, family, items, demand_signature)

    def unexpected_signature_read(self, family, demand_signature=""):
        raise AssertionError("Fresh collection must retain its publication receipt")

    monkeypatch.setattr(CollectedFamilyCacheContext, "store_items", observed_store)
    monkeypatch.setattr(
        CollectedFamilyCacheContext, "load_content_signature", unexpected_signature_read
    )
    family = CompactProductFlowModuleProjectionFamily
    if not publication_fits:
        monkeypatch.setattr(family, "cache_payload_max_bytes", 0)
    result = build_compact_projection_shard(
        CompactProjectionBuildRequest(
            source=request_source,
            missing_families=(family,),
            config=DetectorConfig(),
            bundle_families=(family,),
        )
    )
    ((publication_source, publication_family, items),) = publications
    assert publication_family is family
    assert publication_source.identity(family) == request_source.identity(family)
    assert publication_source.family_cache_dir == request_source.family_cache_dir
    assert items
    assert result.cache_bundle_complete is (use_cache and publication_fits)
    if result.cache_bundle_complete:
        (receipt,) = result.projection_batches
        assert type(receipt) is CompactFamilyProjectionReceipt
        assert receipt.family is family
        assert receipt.content_signature == collected_family_items_content_signature(
            items
        )
        assert request_source.load_items(family) == items
        manifest = BoundedCompactProjectionManifest(())
        receipt.add_to(manifest, request_source)
        assert not manifest.runtime_projections
        assert (
            manifest._source_projection_signatures[
                family, request_source.resolved_path_text
            ]
            == receipt.content_signature
        )
    else:
        (batch,) = result.projection_batches
        assert batch.items == items
        assert batch.content_signature is None


def test_list_collection_and_batch_share_the_same_publication(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"def target(): pass\nalias = target\nalias()\n")
    module = PythonModuleRootParser.for_root(
        tmp_path, cache_dir=tmp_path / "cache"
    ).parsed_source_path(path)
    family = CompactProductFlowModuleProjectionFamily
    items = collect_family_items(module, family)

    def unexpected_store(self, family, items, demand_signature=""):
        raise AssertionError("List and batch consumers must share the cached receipt")

    monkeypatch.setattr(CollectedFamilyCacheContext, "store_items", unexpected_store)
    batch = collect_family_batch(module, family)
    assert batch.items == tuple(items)
    assert batch.content_signature == collected_family_items_content_signature(
        batch.items
    )


def test_incomplete_publication_keeps_all_facts_in_memory(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    source = "class Example: pass\ndef target(): return Example()\n"
    path.write_bytes(source.encode())
    parser = PythonModuleRootParser.for_root(tmp_path, cache_dir=tmp_path / "cache")
    projection_source = CompactProjectionCacheSource(
        path=path,
        module_name="source",
        source_signature=python_source_cache_signature(source),
        family_cache_dir=parser.collected_family_cache_dir,
        scan_root=tmp_path,
        cache_dir=parser.parse_cache_dir,
        use_parse_cache=True,
        source_policy=PythonSourcePathPolicy(),
    )
    families = (
        CompactModuleClassProjectionFamily,
        CompactProductFlowModuleProjectionFamily,
    )
    monkeypatch.setattr(
        CompactProductFlowModuleProjectionFamily, "cache_payload_max_bytes", 0
    )
    result = build_compact_projection_shard(
        CompactProjectionBuildRequest(
            source=projection_source,
            missing_families=families,
            config=DetectorConfig(),
            bundle_families=families,
        )
    )
    assert not result.cache_bundle_complete
    assert tuple(batch.family for batch in result.projection_batches) == families
    stored, retained = result.projection_batches
    assert stored.items == projection_source.load_items(stored.family)
    assert stored.content_signature is not None
    assert retained.items
    assert retained.content_signature is None
    assert projection_source.load_items(retained.family) is None
