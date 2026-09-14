"""Global family demand retains original source/version and complete-query semantics."""

import pytest

from nominal_refactor_advisor.analysis import (
    BoundedCompactProjectionManifest,
    CompactProjectionCacheSource,
)
from nominal_refactor_advisor.ast_tools import (
    CollectedFamilyCacheContext,
    PythonSourcePathPolicy,
    SourceModule,
    _collected_family_cache_path,
)
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.collection_algebra import DeferredBatchSequence
from nominal_refactor_advisor.product_flow import (
    CompactProductFlowModuleProjectionFamily as Family,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository
from nominal_refactor_advisor.source_identity import python_source_cache_signature


def manifest_for(tmp_path, texts, *, persist=True, module_names=None):
    manifest = BoundedCompactProjectionManifest(())
    modules = []
    for index, text in enumerate(texts):
        path = tmp_path / f"source{index}.py"
        path.write_text(text)
        name = f"source{index}" if module_names is None else module_names[index]
        source = CompactProjectionCacheSource(
            path=path,
            module_name=name,
            source_signature=python_source_cache_signature(text),
            family_cache_dir=tmp_path / "family" if persist else None,
            scan_root=tmp_path,
            cache_dir=None,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(),
        )
        manifest.add_source(source)
        module = SourceModule(path=path, module_name=name, source=text).parse()
        modules.append(module)
        if persist:
            assert source.store_items(
                Family, (compact_product_flow_projection(module),)
            )
    return manifest, tuple(modules)


def observe_loads(monkeypatch):
    loads = []
    original = CollectedFamilyCacheContext.load_items

    def observe(self, family, *args):
        loads.append((self.path, family))
        return original(self, family, *args)

    monkeypatch.setattr(CollectedFamilyCacheContext, "load_items", observe)
    return loads


def test_complete_content_signature_does_not_require_loading_all_graphs(
    tmp_path, monkeypatch
):
    loads = observe_loads(monkeypatch)
    manifest, _ = manifest_for(tmp_path, ("def first(): pass\n", "def last(): pass\n"))
    loads.clear()
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=True
    )
    signature = manifest.projection_signature(Family)
    assert signature
    assert loads == []
    first = next(iter(sequence))
    assert first.module_name == "source0"
    assert loads == [(manifest.sources[0].path, Family)]
    assert next(iter(sequence)) is first
    assert manifest.projection_count == 1
    assert tuple(sequence)[-1].module_name == "source1"
    assert manifest.projection_count == 2
    assert manifest.projection_signature(Family) == signature
    assert len(loads) == 2


def test_missing_signature_requires_complete_original_content_evidence(tmp_path):
    manifest, _ = manifest_for(
        tmp_path, ("def first(): pass\n", "def last(): pass\n"), persist=False
    )
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=True
    )
    assert sequence.materialized_item_count == 2
    assert manifest.projection_signature(Family)
    assert len(tuple(sequence)) == 2


def test_materialized_public_api_retains_its_tuple_contract(tmp_path):
    manifest, _ = manifest_for(tmp_path, ("def first(): pass\n",))
    assert type(manifest.projections_for_family(Family)) is tuple


def test_count_derives_unique_source_materializations_without_rescanning_batches(
    tmp_path, monkeypatch
):
    manifest, _ = manifest_for(
        tmp_path, ("def first(): pass\n", "def last(): pass\n"), persist=False
    )

    def forbid_recount(self):
        raise AssertionError("Loading another source must not recount earlier batches")

    monkeypatch.setattr(
        DeferredBatchSequence, "materialized_item_count", property(forbid_recount)
    )
    first_view = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    assert next(iter(first_view)).module_name == "source0"
    assert manifest.projection_count == 1
    # A different consumer can request the final source independently. The total
    # counts unique source rows, not the largest individual consumer's prefix.
    assert (
        manifest._source_family_items(
            manifest.sources[1], Family, derive_content_identity=False
        )[0].module_name
        == "source1"
    )
    assert manifest.projection_count == 2
    repeated = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    assert next(iter(repeated)).module_name == "source0"
    assert manifest.projection_count == 2
    assert len(tuple(first_view)) == 2
    assert manifest.projection_count == 2


def test_corrupt_cache_repairs_only_the_requested_original_source(tmp_path):
    manifest, _ = manifest_for(tmp_path, ("def first(): pass\n", "def last(): pass\n"))
    source = manifest.sources[0]
    _collected_family_cache_path(
        source.family_cache_dir, source.identity(Family)
    ).write_bytes(b"corrupt")
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    first = next(iter(sequence))
    assert first.module_name == "source0"
    assert sequence.materialized_item_count == 1
    assert any(
        declaration.identity.qualname == "first"
        for declaration in first.function_declarations
    )


def test_repair_rejects_a_changed_source_and_can_retry_the_original(tmp_path):
    text = "def original(): pass\n"
    manifest, _ = manifest_for(tmp_path, (text,), persist=False)
    source = manifest.sources[0]
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    source.path.write_text("def replacement(): pass\n")
    with pytest.raises(ValueError, match="Source changed"):
        tuple(sequence)
    assert sequence.materialized_item_count == 0
    source.path.write_text(text)
    assert tuple(sequence)[0].function_declarations[0].identity.qualname == "original"


def test_global_lookup_still_reads_later_ambiguous_modules(tmp_path):
    manifest, modules = manifest_for(
        tmp_path,
        ("def first(): pass\n", "def other(): pass\n"),
        module_names=("shared", "shared"),
    )
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    repository = CompactProductFlowRepository(
        sequence, CompactModuleClassProjectionFamily.collect_modules(modules)
    )
    assert repository.product_projection_for_module("shared") is None
    assert sequence.materialized_item_count == 2
    assert repository.module_flow_context_for_name("shared") is None


def test_eligibility_short_circuit_does_not_truncate_later_queries(tmp_path):
    manifest, modules = manifest_for(
        tmp_path,
        (
            "from dataclasses import dataclass\n@dataclass\nclass Product:\n    first: int\n    last: int\nProduct.first = 1\n",
            "def late(): pass\n",
        ),
    )
    sequence = manifest.deferred_projections_for_family(
        Family, derive_content_identity=False
    )
    classes = CompactModuleClassProjectionFamily.collect_modules(modules)
    repository = CompactProductFlowRepository(sequence, classes)
    eager = CompactProductFlowRepository(
        tuple(compact_product_flow_projection(module) for module in modules), classes
    )
    assert (
        repository.product_authorities_by_symbol
        == eager.product_authorities_by_symbol
        == {}
    )
    assert sequence.materialized_item_count == 1
    assert any(
        context.owner_symbol == "source1.late" for context in repository.flow_contexts
    )
    assert sequence.materialized_item_count == 2
    assert tuple(repository.iter_flow_contexts()) == repository.flow_contexts
