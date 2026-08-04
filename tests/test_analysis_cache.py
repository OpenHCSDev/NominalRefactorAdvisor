from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import sys
from time import sleep

import pytest

from nominal_refactor_advisor.analysis import (
    CachedPathAnalysisRequest,
    ChangedPathRootAssignment,
    DetectorTypePartition,
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphCacheContext,
    SemanticDescentGraphAnalysisSource,
    analyze_modules,
    analyze_modules_with_cache,
    analyze_module_detector_types_with_cache,
    analyze_path,
    accumulate_compact_global_projections_for_roots,
    default_detector_types_for_analysis,
    release_module_analysis_memory,
)
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisFindingCache,
    DetectorRegistrySignature,
    SourceFileSignatureCache,
)
from nominal_refactor_advisor.ast_tools import (
    ExportDictShapeFamily,
    collect_family_items,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor import ast_tools as ast_tools_module
from nominal_refactor_advisor import semantic_descent as semantic_descent_module
from nominal_refactor_advisor.cache_paths import (
    AdvisorCacheRetention,
    AdvisorCacheRetentionPolicy,
    analysis_cache_sibling,
    default_parse_cache_dir,
    semantic_descent_cache_sibling,
)
from nominal_refactor_advisor.cache_checkout import (
    CacheCheckoutPathError,
    absolute_checkout_path,
    checkout_relative_path,
    rebase_checkout_path,
)
from nominal_refactor_advisor.detectors import (
    CrossModuleCandidateDetector,
    DetectorConfig,
    IssueDetector,
    PerModuleIssueDetector,
    SemanticDescentGraphIssueDetector,
)
from nominal_refactor_advisor.detectors import _environment as environment_detectors
from nominal_refactor_advisor.detectors import (
    _abstraction_reuse as abstraction_reuse_detectors,
)
from nominal_refactor_advisor.detectors import (
    _nominal_authority_surface as nominal_surface_detectors,
)
from nominal_refactor_advisor.detectors import _runtime as runtime_detectors
from nominal_refactor_advisor.detectors import (
    _semantic_descent as semantic_descent_detectors,
)
from nominal_refactor_advisor.detectors import (
    _role_surface_drift as role_surface_detectors,
)
from nominal_refactor_advisor.detectors import _surface as surface_detectors
from nominal_refactor_advisor.detectors import _structural as structural_detectors
from nominal_refactor_advisor.detectors import _systemic as systemic_detectors
from nominal_refactor_advisor.models import FindingSpec, RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticModuleProjectionFamily,
    SemanticAuthority,
    SemanticAuthorityKind,
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheIdentity,
    build_semantic_descent_graph,
    build_compact_semantic_descent_graph,
)
from nominal_refactor_advisor.semantic_algebra import FiniteAxisSystem


def _empty_semantic_descent_graph(authority_name: str = "") -> SemanticDescentGraph:
    authorities = (
        (
            SemanticAuthority(
                authority_id=f"authority:{authority_name}",
                kind=SemanticAuthorityKind.CLASS_FAMILY,
                name=authority_name,
                location=SourceLocation("fixture.py", 1, authority_name),
                fact_ids=(),
            ),
        )
        if authority_name
        else ()
    )
    return SemanticDescentGraph(
        authorities=authorities,
        facts=(),
        projections=(),
        mirror_edges=(),
        certificates=(),
    )


def test_cache_retention_evicts_old_roots_and_protects_active_root(
    tmp_path: Path,
) -> None:
    cache_home = tmp_path / "cache-home"
    active_root = cache_home / "active"
    recent_root = cache_home / "recent"
    old_root = cache_home / "old"
    for index, root in enumerate((old_root, recent_root, active_root), start=1):
        root.mkdir(parents=True)
        (root / "entry.pickle").write_bytes(b"cache")
        os.utime(root, (index, index))
    retention = AdvisorCacheRetention(
        cache_home,
        AdvisorCacheRetentionPolicy(
            max_root_count=2,
            max_total_bytes=1024,
            max_root_bytes=1024,
            maintenance_interval_seconds=0.0,
        ),
    )

    report = retention.maintain(active_root)

    assert active_root.is_dir()
    assert recent_root.is_dir()
    assert not old_root.exists()
    assert report.removed_root_count == 1
    assert report.removed_file_count == 1
    assert report.removed_bytes == len(b"cache")


def test_cache_retention_prunes_oldest_files_to_per_root_byte_bound(
    tmp_path: Path,
) -> None:
    cache_home = tmp_path / "cache-home"
    active_root = cache_home / "active"
    active_root.mkdir(parents=True)
    cache_files = tuple(active_root / f"entry-{index}.pickle" for index in range(3))
    for index, cache_file in enumerate(cache_files, start=1):
        cache_file.write_bytes(b"0123456789")
        os.utime(cache_file, (index, index))
    retention = AdvisorCacheRetention(
        cache_home,
        AdvisorCacheRetentionPolicy(
            max_root_count=2,
            max_total_bytes=1024,
            max_root_bytes=15,
            maintenance_interval_seconds=0.0,
        ),
    )

    report = retention.maintain(active_root)

    assert not cache_files[0].exists()
    assert not cache_files[1].exists()
    assert cache_files[2].is_file()
    assert report.removed_file_count == 2
    assert report.removed_bytes == 20


def test_cache_retention_throttles_repeated_tree_walks(tmp_path: Path) -> None:
    cache_home = tmp_path / "cache-home"
    active_root = cache_home / "active"
    retention = AdvisorCacheRetention(
        cache_home,
        AdvisorCacheRetentionPolicy(
            max_root_count=1,
            maintenance_interval_seconds=3600.0,
        ),
    )
    retention.maintain(active_root)
    late_root = cache_home / "late"
    late_root.mkdir()

    report = retention.maintain(active_root)

    assert report.skipped
    assert late_root.is_dir()


def test_module_analysis_memory_release_clears_ast_bound_lru_caches() -> None:
    module = ast_tools_module.ast.parse("value = source + 1\n")
    ast_tools_module._walk_nodes(module)
    runtime_detectors.SurfaceFunctionIndex.from_module(module)

    cleared_cache_count = release_module_analysis_memory()

    assert cleared_cache_count > 0
    assert ast_tools_module._walk_nodes.cache_info().currsize == 0
    assert runtime_detectors.SurfaceFunctionIndex.from_module.cache_info().currsize == 0


def test_module_detector_shard_cache_reuses_exact_focused_findings(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    source_path = package_root / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    module = parse_python_modules(package_root)[0]

    class FocusedShardDetector(PerModuleIssueDetector):
        call_count = 0

        def _findings_for_module(self, module, config):
            del module, config
            type(self).call_count += 1
            return []

    arguments = {
        "detector_types": (FocusedShardDetector,),
        "presentation_roots": (package_root,),
        "analysis_cache_dir": tmp_path / "analysis-cache",
    }

    try:
        cold = analyze_module_detector_types_with_cache(
            module,
            DetectorConfig(),
            **arguments,
        )
        warm = analyze_module_detector_types_with_cache(
            module,
            DetectorConfig(),
            **arguments,
        )
    finally:
        for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
            if detector_type is FocusedShardDetector:
                del IssueDetector.__registry__[registry_key]

    assert cold.cache_status is AnalysisCacheStatus.MISS
    assert warm.cache_status is AnalysisCacheStatus.HIT
    assert FocusedShardDetector.call_count == 1


def test_semantic_graph_cache_treats_truncated_payload_as_miss(
    tmp_path: Path,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    identity = SemanticDescentGraphCacheIdentity.from_roots((tmp_path,))
    cache._entry_path(identity).write_bytes(b"\x80\x05")

    assert cache.load(identity).graph is None


def test_semantic_graph_cache_interrupted_store_preserves_published_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    identity = SemanticDescentGraphCacheIdentity.from_roots((tmp_path,))
    published_graph = _empty_semantic_descent_graph("Published")
    cache.store(identity, published_graph)
    original_dump = semantic_descent_module.pickle.dump

    def interrupted_dump(payload, handle, *, protocol):
        handle.write(b"partial")
        raise OSError("simulated interrupted cache publication")

    monkeypatch.setattr(semantic_descent_module.pickle, "dump", interrupted_dump)
    cache.store(identity, _empty_semantic_descent_graph("Interrupted"))
    monkeypatch.setattr(semantic_descent_module.pickle, "dump", original_dump)

    assert cache.load(identity).graph == published_graph
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_semantic_graph_cache_concurrent_publications_remain_readable(
    tmp_path: Path,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    identity = SemanticDescentGraphCacheIdentity.from_roots((tmp_path,))
    graphs = tuple(
        _empty_semantic_descent_graph(f"Writer{index}") for index in range(8)
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(lambda graph: cache.store(identity, graph), graphs))

    assert cache.load(identity).graph in graphs
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_semantic_graph_latest_cache_is_lightweight_identity_pointer(
    tmp_path: Path,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    identity = SemanticDescentGraphCacheIdentity.from_roots((tmp_path,))
    graph = _empty_semantic_descent_graph("Published")

    cache.store(identity, graph)
    family_identity = (
        semantic_descent_module.SemanticDescentGraphCacheFamilyIdentity.from_identity(
            identity
        )
    )
    exact_path = cache._entry_path(identity)
    latest_path = cache._latest_path(family_identity)

    assert cache.load_latest(family_identity).graph == graph
    assert latest_path.stat().st_size < exact_path.stat().st_size


def test_semantic_graph_cache_retains_only_recent_exact_generations(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "module.py"
    cache_root = tmp_path / "cache"
    cache = SemanticDescentGraphCache(cache_root, max_exact_entry_count=2)
    identities = []
    for index in range(4):
        source_path.write_text(f"VALUE = {index}\n", encoding="utf-8")
        identity = SemanticDescentGraphCacheIdentity.from_roots((source_root,))
        identities.append(identity)
        cache.store(identity, _empty_semantic_descent_graph(f"Graph{index}"))

    exact_paths = tuple(
        path
        for path in cache_root.glob("*.pickle")
        if not path.name.startswith("latest-")
    )
    family_identity = (
        semantic_descent_module.SemanticDescentGraphCacheFamilyIdentity.from_identity(
            identities[-1]
        )
    )

    assert len(exact_paths) == 2
    assert cache._entry_path(identities[-1]).is_file()
    assert cache.load_latest(family_identity).graph == _empty_semantic_descent_graph(
        "Graph3"
    )


def test_semantic_graph_cache_context_reuses_latest_graph_for_exact_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    roots = (tmp_path,)
    identity = SemanticDescentGraphCacheIdentity.from_roots(roots)
    graph = _empty_semantic_descent_graph("Published")
    cache.store(identity, graph)
    context = SemanticDescentGraphCacheContext(
        storage_root=tmp_path,
        roots=roots,
    )
    load_calls = 0
    original_load = semantic_descent_module.SemanticDescentGraphCache.load

    def counted_load(self, requested_identity):
        nonlocal load_calls
        load_calls += 1
        return original_load(self, requested_identity)

    monkeypatch.setattr(
        semantic_descent_module.SemanticDescentGraphCache,
        "load",
        counted_load,
    )

    latest_graph = context.latest_graph()
    exact_graph = context.cached_graph()

    assert latest_graph is graph or latest_graph == graph
    assert exact_graph is latest_graph
    assert load_calls == 1


def test_equivalent_checkouts_reuse_graph_and_detector_caches_with_rebased_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    checkout_a.mkdir()
    checkout_b.mkdir()
    source_text = "class StableAuthority:\n    pass\n"
    source_a = checkout_a / "module.py"
    source_b = checkout_b / "module.py"
    source_a.write_text(source_text, encoding="utf-8")
    source_b.write_text(source_text, encoding="utf-8")

    graph_cache = SemanticDescentGraphCache(tmp_path / "shared-graph-cache")
    graph_identity_a = SemanticDescentGraphCacheIdentity.from_roots((checkout_a,))
    graph_identity_b = SemanticDescentGraphCacheIdentity.from_roots((checkout_b,))
    analysis_identity_a = AnalysisCacheIdentity.from_roots(
        (checkout_a,),
        DetectorConfig(),
    )
    analysis_identity_b = AnalysisCacheIdentity.from_roots(
        (checkout_b,),
        DetectorConfig(),
    )
    graph = SemanticDescentGraph(
        authorities=(
            SemanticAuthority(
                authority_id="stable-authority",
                kind=SemanticAuthorityKind.CLASS_FAMILY,
                name="StableAuthority",
                location=SourceLocation(str(source_a), 1, "StableAuthority"),
                fact_ids=(),
            ),
        ),
        facts=(),
        projections=(),
        mirror_edges=(),
        certificates=(),
    )
    graph_cache.store(graph_identity_a, graph)

    assert graph_identity_a.cache_token == graph_identity_b.cache_token
    assert analysis_identity_a.cache_token == analysis_identity_b.cache_token
    assert str(checkout_a) not in repr(graph_identity_a)
    assert str(checkout_a) not in repr(analysis_identity_a)
    relocated_graph = graph_cache.load(graph_identity_b).graph
    assert relocated_graph is not None
    assert relocated_graph.authorities[0].location.file_path == str(source_b)

    detector_calls = 0
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Relocatable cache",
        why="relocatable cache",
        capability_gap="relocatable cache",
        relation_context="relocatable cache",
    )

    class RelocatableCacheDetector(IssueDetector):
        detector_id = "relocatable_cache_detector"

        def _collect_findings(
            self,
            modules: list,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            nonlocal detector_calls
            del config
            detector_calls += 1
            module = modules[0]
            return [
                finding_spec.build(
                    self.detector_id,
                    "relocatable finding",
                    (SourceLocation(str(module.path), 1, "StableAuthority"),),
                )
            ]

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (RelocatableCacheDetector,),
    )
    analysis_cache_dir = tmp_path / "shared-analysis-cache"
    try:
        first = analyze_modules_with_cache(
            (checkout_a,),
            parse_python_module_roots((checkout_a,)),
            DetectorConfig(),
            analysis_cache_dir=analysis_cache_dir,
        )
        second = analyze_modules_with_cache(
            (checkout_b,),
            parse_python_module_roots((checkout_b,)),
            DetectorConfig(),
            analysis_cache_dir=analysis_cache_dir,
        )
    finally:
        for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
            if detector_type is RelocatableCacheDetector:
                del IssueDetector.__registry__[registry_key]

    assert first.cache_status is AnalysisCacheStatus.MISS
    assert second.cache_status is AnalysisCacheStatus.HIT
    assert detector_calls == 1
    assert second.findings[0].evidence[0].file_path == str(source_b)

    source_b.write_text("VALUE = 'foreign content'\n", encoding="utf-8")
    foreign_identity = SemanticDescentGraphCacheIdentity.from_roots((checkout_b,))
    assert foreign_identity.cache_token != graph_identity_a.cache_token
    assert graph_cache.load(foreign_identity).graph is None


def test_relocatable_caches_reject_path_escape_and_ambiguous_roots(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    nested_root = checkout / "nested"
    nested_root.mkdir(parents=True)
    source_path = nested_root / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(CacheCheckoutPathError, match="unsafe"):
        checkout_relative_path("../escape.py", (checkout,))
    with pytest.raises(CacheCheckoutPathError, match="multiple roots"):
        checkout_relative_path(source_path, (checkout, nested_root))

    identity = AnalysisCacheIdentity.from_roots((checkout,), DetectorConfig())
    cache = AnalysisFindingCache(tmp_path / "analysis-cache")
    escaped_finding = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Escaped",
        why="escaped",
        capability_gap="escaped",
        relation_context="escaped",
    ).build(
        "escaped_cache_detector",
        "escaped finding",
        (SourceLocation(str(tmp_path / "outside.py"), 1, "outside"),),
    )
    cache.store(identity, [escaped_finding])

    assert cache.load(identity).status is AnalysisCacheStatus.MISS


def test_checkout_cache_identity_preserves_in_root_source_symlink(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    generated = checkout / "pkg" / "generated"
    generated.mkdir(parents=True)
    provenance = tmp_path / ".provenance" / "generated.py"
    provenance.parent.mkdir()
    provenance.write_text("VALUE = 1\n")
    source_link = generated / "generated.py"
    source_link.symlink_to(provenance)

    logical_path = checkout_relative_path(source_link, (checkout,))

    assert logical_path == "0:pkg/generated/generated.py"
    assert absolute_checkout_path(logical_path, (checkout,)) == str(source_link)


def test_analysis_identity_preserves_in_root_source_symlink(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    generated = checkout / "pkg" / "generated"
    generated.mkdir(parents=True)
    provenance = tmp_path / ".provenance" / "generated.py"
    provenance.parent.mkdir()
    provenance.write_text("VALUE = 1\n")
    source_link = generated / "generated.py"
    source_link.symlink_to(provenance)

    identity = AnalysisCacheIdentity.from_roots((checkout,), DetectorConfig())

    assert tuple(source_file.path for source_file in identity.source_files) == (
        "0:pkg/generated/generated.py",
    )


def test_source_signature_cache_keys_in_root_symlink_lexically(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    provenance = tmp_path / ".provenance" / "generated.py"
    provenance.parent.mkdir()
    provenance.write_text("VALUE = 1\n")
    source_link = checkout / "generated.py"
    source_link.symlink_to(provenance)
    storage = AnalysisFindingCache(tmp_path / "cache").storage()
    assert storage is not None
    signature_cache = SourceFileSignatureCache(storage)

    first = signature_cache.source_file_signatures((source_link,))[0]
    second = SourceFileSignatureCache(storage).source_file_signatures((source_link,))[0]

    assert first == second
    assert first.path == str(source_link)


def test_relocated_cache_identity_rebases_in_root_source_symlink_logically(
    tmp_path: Path,
) -> None:
    source_checkout = tmp_path / "source"
    target_checkout = tmp_path / "target"
    source_link = source_checkout / "pkg" / "generated.py"
    target_link = target_checkout / "pkg" / "generated.py"
    for checkout, source_path in (
        (source_checkout, source_link),
        (target_checkout, target_link),
    ):
        (checkout / "pkg").mkdir(parents=True)
        provenance = tmp_path / f".{checkout.name}-provenance.py"
        provenance.write_text("VALUE = 1\n")
        source_path.symlink_to(provenance)

    assert rebase_checkout_path(
        source_link,
        (source_checkout,),
        (target_checkout,),
    ) == str(target_link)


class CountingSemanticCacheDetector(IssueDetector):
    call_count = 0

    def _collect_findings(
        self, modules: list, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del modules, config
        type(self).call_count += 1
        return []


def _load_dynamic_detector(detector_module_path: Path, module_name: str) -> type:
    spec = importlib.util.spec_from_file_location(module_name, detector_module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.DynamicDetector


def test_custom_cache_dir_uses_non_colliding_sibling_paths(tmp_path: Path) -> None:
    default_parse_cache = tmp_path / ".nra-cache" / "ast"
    custom_parse_cache = tmp_path / "run-cache"

    assert analysis_cache_sibling(default_parse_cache) == (
        tmp_path / ".nra-cache" / "analysis"
    )
    assert semantic_descent_cache_sibling(default_parse_cache) == (
        tmp_path / ".nra-cache" / "semantic_descent"
    )
    assert analysis_cache_sibling(custom_parse_cache) == tmp_path / "run-cache-analysis"
    assert semantic_descent_cache_sibling(custom_parse_cache) == (
        tmp_path / "run-cache-semantic_descent"
    )


def test_default_parse_cache_uses_cache_home_root_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    cache_home = tmp_path / "cache-home"

    monkeypatch.setenv("NRA_CACHE_HOME", cache_home.as_posix())

    cache_dir = default_parse_cache_dir(package_root)

    assert cache_dir.parent.parent == cache_home
    assert cache_dir.parent.name.startswith("pkg-")
    assert cache_dir.name == "ast"


def test_detector_registry_signature_tracks_detector_module_source(
    tmp_path: Path,
) -> None:
    module_name = "dynamic_signature_detector_module"
    module_path = tmp_path / f"{module_name}.py"

    def write_detector(helper_value: int) -> None:
        module_path.write_text(
            "from nominal_refactor_advisor.detectors import IssueDetector\n"
            "\n"
            "class DynamicDetector(IssueDetector):\n"
            "    detector_id = 'dynamic_signature_detector'\n"
            f"    helper_value = {helper_value}\n"
            "\n"
            "    def _collect_findings(self, modules, config):\n"
            "        del modules, config\n"
            "        return []\n",
            encoding="utf-8",
        )

    try:
        write_detector(1)
        first_detector = _load_dynamic_detector(module_path, module_name)
        first_signature = DetectorRegistrySignature.from_detector_types(
            (first_detector,)
        )

        sleep(0.01)
        write_detector(2)
        second_detector = _load_dynamic_detector(module_path, module_name)
        second_signature = DetectorRegistrySignature.from_detector_types(
            (second_detector,)
        )
    finally:
        sys.modules.pop(module_name, None)
        IssueDetector.__registry__.pop("dynamic_signature_detector", None)

    assert first_signature != second_signature
    assert (
        first_signature.detector_types[0].implementation_source_hash
        != second_signature.detector_types[0].implementation_source_hash
    )


def test_analysis_cache_reuses_semantic_identity_after_comment_only_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "analysis"
    CountingSemanticCacheDetector.call_count = 0

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingSemanticCacheDetector,),
    )

    first_result = analyze_modules_with_cache(
        (package_root,),
        parse_python_module_roots((package_root,)),
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    assert first_result.cache_status is AnalysisCacheStatus.MISS
    assert CountingSemanticCacheDetector.call_count == 1

    module_path.write_text("VALUE = 1\n# trailing comment\n", encoding="utf-8")
    second_result = analyze_modules_with_cache(
        (package_root,),
        parse_python_module_roots((package_root,)),
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )

    assert second_result.cache_status is AnalysisCacheStatus.HIT
    assert CountingSemanticCacheDetector.call_count == 1


def test_analysis_cache_rebuild_lease_waits_for_exact_cache_entry(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    identity = AnalysisCacheIdentity.from_roots((package_root,), DetectorConfig())
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")

    def wait_for_cached_lease() -> tuple[bool, AnalysisCacheStatus | None]:
        with cache.rebuild_lease(identity, poll_interval_seconds=0.01) as lease:
            cached_status = (
                None if lease.cached_lookup is None else lease.cached_lookup.status
            )
            return lease.owns_rebuild, cached_status

    with cache.rebuild_lease(identity) as first_lease:
        assert first_lease.owns_rebuild
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wait_for_cached_lease)
            sleep(0.05)
            assert not future.done()
            cache.store(identity, [])

    assert future.result(timeout=1.0) == (False, AnalysisCacheStatus.HIT)


def test_analysis_cache_rejects_malformed_finding_evidence(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    identity = AnalysisCacheIdentity.from_roots((package_root,), DetectorConfig())
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")
    storage = cache.storage()
    assert storage is not None
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Malformed evidence",
        why="malformed evidence",
        capability_gap="malformed evidence",
        relation_context="malformed evidence",
    )
    finding = finding_spec.build(
        "malformed_evidence",
        "malformed evidence",
        (SourceLocation((package_root / "mod.py").as_posix(), 1, "VALUE"),),
    )
    object.__setattr__(finding, "evidence", (finding.evidence,))
    storage.store_payload_atomic(
        storage.entry_path(identity),
        {"identity": identity, "findings": [finding]},
    )

    assert cache.load(identity).status is AnalysisCacheStatus.MISS
    with pytest.raises(TypeError, match="non-SourceLocation evidence"):
        cache.store(identity, [finding])


def test_cross_module_candidate_detector_reuses_contextual_global_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "a.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
    (package_root / "b.py").write_text("class Beta:\n    pass\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "analysis"
    candidate_calls = 0
    finding_calls = 0
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Contextual cache",
        why="contextual cache",
        capability_gap="contextual cache",
        relation_context="contextual cache",
    )

    class CountingCrossModuleDetector(CrossModuleCandidateDetector[str]):
        detector_id = "counting_cross_module"

        def _candidate_items(
            self,
            modules: list,
            config: DetectorConfig,
        ) -> tuple[str, ...]:
            nonlocal candidate_calls, finding_calls
            del config
            candidate_calls += 1
            return tuple(module.path.name for module in modules)

        def _finding_for_candidate(self, candidate: str) -> RefactorFinding:
            nonlocal finding_calls
            finding_calls += 1
            return finding_spec.build(
                self.detector_id,
                f"candidate {candidate}",
                (SourceLocation(str(package_root / candidate), 1, candidate),),
            )

    for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
        if detector_type is CountingCrossModuleDetector:
            del IssueDetector.__registry__[registry_key]
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CountingCrossModuleDetector,),
    )

    first_result = analyze_modules_with_cache(
        (package_root,),
        parse_python_module_roots((package_root,)),
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )
    (package_root / "b.py").write_text(
        "class Beta:\n    pass\n\nclass Changed:\n    pass\n",
        encoding="utf-8",
    )
    second_result = analyze_modules_with_cache(
        (package_root,),
        parse_python_module_roots((package_root,)),
        DetectorConfig(),
        analysis_cache_dir=cache_dir,
    )

    assert first_result.cache_status is AnalysisCacheStatus.MISS
    assert second_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert candidate_calls == 2
    assert finding_calls == 2


def test_contextual_global_graph_detectors_share_semantic_descent_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "analysis"
    graph_cache_finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Graph cache",
        why="graph cache",
        capability_gap="graph cache",
        relation_context="graph cache",
    )
    graph_build_count = 0
    graph_ids: list[int] = []
    registered_test_detectors: list[type[IssueDetector]] = []

    def counting_graph_builder(
        modules: list,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> SemanticDescentGraph:
        nonlocal graph_build_count
        del cache_dir, use_cache
        graph_build_count += 1
        return build_semantic_descent_graph(modules)

    class FirstGraphDetector(SemanticDescentGraphIssueDetector, IssueDetector):
        detector_id = "first_graph_cache_detector"
        finding_spec = graph_cache_finding_spec

        @classmethod
        def context_signature(
            cls,
            modules: tuple,
            config: DetectorConfig,
        ) -> str:
            del cls, modules, config
            return "shared-graph-context"

        def _collect_findings(
            self, modules: list, config: DetectorConfig
        ) -> list[RefactorFinding]:
            del modules, config
            raise AssertionError("graph-backed detector should receive graph evidence")

        def _collect_findings_from_graph(
            self,
            graph: SemanticDescentGraph,
            modules: list,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del modules, config
            graph_ids.append(id(graph))
            return []

    class SecondGraphDetector(FirstGraphDetector):
        detector_id = "second_graph_cache_detector"

    registered_test_detectors.extend((FirstGraphDetector, SecondGraphDetector))

    def unregister_test_detectors() -> None:
        for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
            if detector_type in registered_test_detectors:
                del IssueDetector.__registry__[registry_key]

    try:
        monkeypatch.setattr(
            "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
            lambda: (FirstGraphDetector, SecondGraphDetector),
        )
        monkeypatch.setattr(
            "nominal_refactor_advisor.analysis.build_semantic_descent_graph",
            counting_graph_builder,
        )
        result = analyze_modules_with_cache(
            (package_root,),
            parse_python_module_roots((package_root,)),
            DetectorConfig(),
            analysis_cache_dir=cache_dir,
        )
    finally:
        unregister_test_detectors()

    assert result.cache_status is AnalysisCacheStatus.MISS
    assert graph_build_count == 1
    assert len(graph_ids) == 2
    assert len(set(graph_ids)) == 1


def test_graph_detector_uses_cached_repo_graph_for_changed_module_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "a.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
    (package_root / "b.py").write_text("class Beta:\n    pass\n", encoding="utf-8")
    graph_cache_dir = tmp_path / ".nra-cache" / "semantic_descent"
    cached_graph = SemanticDescentGraph(
        authorities=(
            SemanticAuthority(
                authority_id="repo-authority",
                kind=SemanticAuthorityKind.CLASS_FAMILY,
                name="RepoAuthority",
                location=SourceLocation(str(package_root / "a.py"), 1, "RepoAuthority"),
                fact_ids=(),
            ),
        ),
        facts=(),
        projections=(),
        mirror_edges=(),
        certificates=(),
    )
    SemanticDescentGraphCache(graph_cache_dir).store(
        SemanticDescentGraphCacheIdentity.from_roots((package_root,)),
        cached_graph,
    )
    observed_authority_names: list[tuple[str, ...]] = []
    graph_cache_finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Cached repo graph",
        why="cached repo graph",
        capability_gap="cached repo graph",
        relation_context="cached repo graph",
    )

    class CachedRepoGraphDetector(SemanticDescentGraphIssueDetector, IssueDetector):
        detector_id = "cached_repo_graph_detector"
        finding_spec = graph_cache_finding_spec

        @classmethod
        def context_signature(
            cls,
            modules: tuple,
            config: DetectorConfig,
        ) -> str:
            del cls, modules, config
            return "cached-repo-graph"

        def _collect_findings(
            self, modules: list, config: DetectorConfig
        ) -> list[RefactorFinding]:
            del modules, config
            raise AssertionError("graph-backed detector should receive cached graph")

        def _collect_findings_from_graph(
            self,
            graph: SemanticDescentGraph,
            modules: list,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del modules, config
            observed_authority_names.append(
                tuple(authority.name for authority in graph.authorities)
            )
            return []

    for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
        if detector_type is CachedRepoGraphDetector:
            del IssueDetector.__registry__[registry_key]

    def fail_narrow_graph_build(
        modules: list,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> SemanticDescentGraph:
        del modules, cache_dir, use_cache
        raise AssertionError("changed-module analysis rebuilt a narrow graph")

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (CachedRepoGraphDetector,),
    )
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.build_semantic_descent_graph",
        fail_narrow_graph_build,
    )

    changed_module = parse_python_module_roots((package_root / "b.py",))[0]
    result = analyze_modules(
        [changed_module],
        DetectorConfig(),
        semantic_descent_source=SemanticDescentGraphAnalysisSource(
            cache_context=SemanticDescentGraphCacheContext(
                storage_root=graph_cache_dir,
                roots=(package_root,),
            ),
        ),
    )

    assert result == []
    assert observed_authority_names == [("RepoAuthority",)]


def test_uncached_analysis_preserves_cached_repo_graph_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "a.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
    graph_cache_dir = tmp_path / ".nra-cache" / "semantic_descent"
    cached_graph = SemanticDescentGraph(
        authorities=(
            SemanticAuthority(
                authority_id="repo-authority",
                kind=SemanticAuthorityKind.CLASS_FAMILY,
                name="RepoAuthority",
                location=SourceLocation(str(package_root / "a.py"), 1, "RepoAuthority"),
                fact_ids=(),
            ),
        ),
        facts=(),
        projections=(),
        mirror_edges=(),
        certificates=(),
    )
    SemanticDescentGraphCache(graph_cache_dir).store(
        SemanticDescentGraphCacheIdentity.from_roots((package_root,)),
        cached_graph,
    )
    observed_authority_names: list[tuple[str, ...]] = []
    graph_cache_finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Uncached graph source",
        why="uncached graph source",
        capability_gap="uncached graph source",
        relation_context="uncached graph source",
    )

    class UncachedRepoGraphDetector(SemanticDescentGraphIssueDetector, IssueDetector):
        detector_id = "uncached_repo_graph_detector"
        finding_spec = graph_cache_finding_spec

        @classmethod
        def context_signature(
            cls,
            modules: tuple,
            config: DetectorConfig,
        ) -> str:
            del cls, modules, config
            return "uncached-repo-graph"

        def _collect_findings(
            self, modules: list, config: DetectorConfig
        ) -> list[RefactorFinding]:
            del modules, config
            raise AssertionError("graph-backed detector should receive cached graph")

        def _collect_findings_from_graph(
            self,
            graph: SemanticDescentGraph,
            modules: list,
            config: DetectorConfig,
        ) -> list[RefactorFinding]:
            del modules, config
            observed_authority_names.append(
                tuple(authority.name for authority in graph.authorities)
            )
            return []

    for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
        if detector_type is UncachedRepoGraphDetector:
            del IssueDetector.__registry__[registry_key]

    def fail_narrow_graph_build(
        modules: list,
        *,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> SemanticDescentGraph:
        del modules, cache_dir, use_cache
        raise AssertionError("uncached analysis rebuilt a narrow graph")

    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
        lambda: (UncachedRepoGraphDetector,),
    )
    monkeypatch.setattr(
        "nominal_refactor_advisor.analysis.build_semantic_descent_graph",
        fail_narrow_graph_build,
    )

    result = analyze_modules_with_cache(
        (package_root,),
        parse_python_module_roots((package_root,)),
        DetectorConfig(),
        analysis_cache_dir=None,
        semantic_descent_source=SemanticDescentGraphAnalysisSource(
            cache_context=SemanticDescentGraphCacheContext(
                storage_root=graph_cache_dir,
                roots=(package_root,),
            ),
        ),
    )

    assert result.cache_status is AnalysisCacheStatus.DISABLED
    assert result.findings == []
    assert observed_authority_names == [("RepoAuthority",)]


def test_global_detector_shard_survives_detector_registry_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    module_path.write_text("class Alpha:\n    pass\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "analysis"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Global cache",
        why="global cache",
        capability_gap="global cache",
        relation_context="global cache",
    )
    global_calls = 0
    local_calls = 0
    registered_test_detectors: list[type[IssueDetector]] = []

    class StableGlobalDetector(IssueDetector):
        detector_id = "stable_global_cache"

        def _collect_findings(
            self, modules: list, config: DetectorConfig
        ) -> list[RefactorFinding]:
            nonlocal global_calls
            del config
            global_calls += 1
            return [
                finding_spec.build(
                    self.detector_id,
                    "stable global",
                    (SourceLocation(str(modules[0].path), 1, "global"),),
                )
            ]

    registered_test_detectors.append(StableGlobalDetector)

    def unregister_test_detectors() -> None:
        for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
            if detector_type in registered_test_detectors:
                del IssueDetector.__registry__[registry_key]

    try:
        monkeypatch.setattr(
            "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
            lambda: (StableGlobalDetector,),
        )
        first_result = analyze_modules_with_cache(
            (package_root,),
            parse_python_module_roots((package_root,)),
            DetectorConfig(),
            analysis_cache_dir=cache_dir,
        )

        class AddedPerModuleDetector(PerModuleIssueDetector):
            detector_id = "added_per_module_cache"

            def _findings_for_module(
                self, module, config: DetectorConfig
            ) -> list[RefactorFinding]:
                nonlocal local_calls
                del module, config
                local_calls += 1
                return []

        registered_test_detectors.append(AddedPerModuleDetector)
        monkeypatch.setattr(
            "nominal_refactor_advisor.analysis.default_detector_types_for_analysis",
            lambda: (StableGlobalDetector, AddedPerModuleDetector),
        )
        second_result = analyze_modules_with_cache(
            (package_root,),
            parse_python_module_roots((package_root,)),
            DetectorConfig(),
            analysis_cache_dir=cache_dir,
        )
    finally:
        unregister_test_detectors()

    assert first_result.cache_status is AnalysisCacheStatus.MISS
    assert second_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert global_calls == 1
    assert local_calls == 1
    assert [finding.summary for finding in second_result.findings] == ["stable global"]


def test_collected_family_items_are_persisted_beside_parse_cache(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    module_path.write_text(
        "\n"
        "def export(item):\n"
        "    return {\n"
        "        'name': item.name,\n"
        "        'score': item.score,\n"
        "        'label': item.label,\n"
        "    }\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    first_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    first_items = collect_family_items(first_module, ExportDictShapeFamily)
    family_cache_dir = cache_dir / "collected-family"

    assert first_items
    assert tuple(family_cache_dir.glob("*.pickle"))

    second_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    second_items = collect_family_items(second_module, ExportDictShapeFamily)

    assert [item.key_names for item in second_items] == [
        item.key_names for item in first_items
    ]


def test_collected_family_can_opt_into_a_larger_bounded_cache_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "def export(item):\n"
        "    return {'name': item.name, 'score': item.score, 'label': item.label}\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"
    schema = ast_tools_module.CollectedFamilyCacheSchema(
        version=10_001,
        max_payload_bytes=64,
    )
    monkeypatch.setattr(
        ast_tools_module,
        "collected_family_cache_schema",
        schema,
    )
    monkeypatch.setattr(
        ExportDictShapeFamily,
        "cache_payload_max_bytes",
        10_000,
    )

    module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    assert collect_family_items(module, ExportDictShapeFamily)
    payload_paths = tuple((cache_dir / "collected-family").glob("*.pickle"))

    assert len(payload_paths) == 1
    assert payload_paths[0].stat().st_size > schema.max_payload_bytes
    assert payload_paths[0].stat().st_size <= 10_000


def test_generated_boundary_global_projection_reuses_compact_module_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated_catalog.py").write_text(
        "# generated file\nSEMANTIC_MODE = 'canonical'\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"
    first_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    first_sites = collect_family_items(
        first_module,
        runtime_detectors.GeneratedBoundarySemanticConstantSiteFamily,
    )
    release_module_analysis_memory()

    def unexpected_collection(
        cls: type,
        module,
    ) -> tuple:
        del cls, module
        raise AssertionError("compact global projection cache was not reused")

    monkeypatch.setattr(
        runtime_detectors.GeneratedBoundarySemanticConstantAuthority,
        "module_sites",
        classmethod(unexpected_collection),
    )
    second_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    second_sites = collect_family_items(
        second_module,
        runtime_detectors.GeneratedBoundarySemanticConstantSiteFamily,
    )

    assert second_sites == first_sites
    assert second_sites[0].file_path == str(package_root / "generated_catalog.py")


def test_warm_compact_projection_stream_skips_ast_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated_catalog.py").write_text(
        "# generated file\nSEMANTIC_MODE = 'canonical'\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"
    detector_types = (
        runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector,
    )
    first = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        cache_dir=cache_dir,
    )
    assert first.projection_count == 1
    release_module_analysis_memory()

    def unexpected_ast_load(self, paths):
        del self, paths
        raise AssertionError("warm compact projection stream deserialized an AST")

    monkeypatch.setattr(
        ast_tools_module.PythonModuleRootParser,
        "parsed_source_paths",
        unexpected_ast_load,
    )
    second = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        cache_dir=cache_dir,
    )

    assert second.projection_count == first.projection_count


def test_compact_global_projection_accumulator_matches_full_ast_detection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    generated_path = package_root / "generated_policy.py"
    runtime_path = package_root / "runtime.py"
    generated_path.write_text(
        "# generated from policy schema\nPOLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    runtime_path.write_text(
        "POLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    detector_type = runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (detector_type,),
        use_parse_cache=False,
    )

    projected_findings = accumulator.findings_by_detector(DetectorConfig())[
        detector_type
    ]
    modules = parse_python_modules(package_root, use_parse_cache=False)
    full_ast_findings = detector_type().detect(modules, DetectorConfig())

    assert [finding.to_dict() for finding in projected_findings] == [
        finding.to_dict() for finding in full_ast_findings
    ]


def test_compact_hierarchy_projection_matches_full_ast_detection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "alpha.py").write_text(
        "class Alpha:\n"
        "    def prepare(self, value):\n"
        "        ready = self.normalize(value)\n"
        "        return self.finish(ready)\n"
        "\n"
        "    def score(self, value):\n"
        "        scored = self.compute(value)\n"
        "        return self.finish(scored)\n",
        encoding="utf-8",
    )
    (package_root / "beta.py").write_text(
        "class Beta:\n"
        "    def build(self, value):\n"
        "        ready = self.normalize(value)\n"
        "        return self.finish(ready)\n"
        "\n"
        "    def evaluate(self, value):\n"
        "        scored = self.compute(value)\n"
        "        return self.finish(scored)\n",
        encoding="utf-8",
    )
    detector_types = (
        systemic_detectors.InheritanceHierarchyCandidateDetector,
        systemic_detectors.RepeatedPrivateMethodDetector,
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    projected_findings = accumulator.findings_by_detector(DetectorConfig())
    modules = parse_python_modules(package_root, use_parse_cache=False)

    for detector_type in detector_types:
        full_ast_findings = detector_type().detect(modules, DetectorConfig())
        assert [finding.to_dict() for finding in projected_findings[detector_type]] == [
            finding.to_dict() for finding in full_ast_findings
        ]


def test_compact_private_reference_detectors_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "def _stale_export(rows):\n"
        "    normalized = [str(row).strip() for row in rows]\n"
        "    if not normalized:\n"
        "        return []\n"
        "    return [value.upper() for value in normalized if value]\n"
        "\n"
        "class Publisher:\n"
        "    def _stale_method(self, rows):\n"
        "        normalized = [str(row).strip() for row in rows]\n"
        "        if not normalized:\n"
        "            return []\n"
        "        return [value.upper() for value in normalized if value]\n"
        "\n"
        "    def _write_static_shell(self, dest):\n"
        "        payload = '''<section class=\"report\">\n"
        "<header><h1>Release</h1></header>\n"
        "<main><article>Generated view</article></main>\n"
        "</section>'''\n"
        "        (dest / 'index.html').write_text(payload, encoding='utf-8')\n",
        encoding="utf-8",
    )
    detector_types = (
        runtime_detectors.DeadEmbeddedStaticPayloadDetector,
        runtime_detectors.UnreferencedPrivateFunctionDetector,
        runtime_detectors.DanglingPrivateMethodDetector,
    )
    config = DetectorConfig(
        min_unreferenced_private_function_lines=4,
        min_static_payload_function_lines=4,
        min_static_payload_literal_lines=4,
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    legacy_context = runtime_detectors.PrivateReferenceDetectorContext(modules)
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    projected_findings = accumulator.findings_by_detector(config)

    for detector_type in detector_types:
        detector = detector_type()
        legacy_findings = [
            detector._finding_for_candidate(candidate)
            for module in modules
            for candidate in detector._candidate_items_for_private_reference_context(
                module,
                legacy_context,
                config,
            )
        ]
        assert [finding.to_dict() for finding in projected_findings[detector_type]] == [
            finding.to_dict() for finding in legacy_findings
        ]


def test_compact_public_support_projection_matches_legacy_reference_index(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "_helpers.py").write_text(
        "def parameter_names(function):\n"
        "    return tuple(function.args)\n"
        "\n"
        "def enum_member_ref(node):\n"
        "    return node.name, node.value\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text(
        "from pkg._helpers import parameter_names\n"
        "\n"
        "def consume(function):\n"
        "    return parameter_names(function)\n",
        encoding="utf-8",
    )
    detector_type = systemic_detectors.PublicBareSupportFunctionDetector
    config = DetectorConfig()
    modules = parse_python_modules(package_root, use_parse_cache=False)
    detector = detector_type()
    legacy_findings = detector._findings_for_candidates(
        systemic_detectors._public_bare_support_function_candidates(modules),
        config,
    )
    projected_findings = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (detector_type,),
        use_parse_cache=False,
    ).findings_by_detector(config)[detector_type]

    assert [finding.to_dict() for finding in projected_findings] == [
        finding.to_dict() for finding in legacy_findings
    ]


def test_compact_flattened_candidate_projections_match_full_ast_detection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "builders.py").write_text(
        "def build_left(source):\n"
        "    return Target(alpha=source.alpha, beta=source.beta)\n"
        "\n"
        "def build_right(source):\n"
        "    return Target(alpha=source.alpha, beta=source.beta)\n"
        "\n"
        "def build_pose(source):\n"
        "    return Target(**declared_values_by_type(PoseCarrier, source))\n"
        "\n"
        "def build_repair(source):\n"
        "    return Target(**declared_values_by_type(RepairCarrier, source))\n"
        "\n"
        "def export_left(source):\n"
        "    return {'alpha': source.alpha, 'beta': source.beta, 'gamma': source.gamma}\n"
        "\n"
        "def export_right(source):\n"
        "    return {'alpha': source.alpha, 'beta': source.beta, 'gamma': source.gamma}\n"
        "\n"
        "REGISTRY = {}\n"
        "REGISTRY['alpha'] = Alpha\n"
        "REGISTRY['beta'] = Beta\n"
        "REGISTRY['gamma'] = Gamma\n",
        encoding="utf-8",
    )
    detector_types = (
        runtime_detectors.RepeatedBuilderCallDetector,
        runtime_detectors.DeclaredFieldExtractionFanoutDetector,
        runtime_detectors.RepeatedExportDictDetector,
        runtime_detectors.ManualClassRegistrationDetector,
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    projected_findings = accumulator.findings_by_detector(DetectorConfig())
    modules = parse_python_modules(package_root, use_parse_cache=False)

    for detector_type in detector_types:
        full_ast_findings = detector_type().detect(modules, DetectorConfig())
        assert [finding.to_dict() for finding in projected_findings[detector_type]] == [
            finding.to_dict() for finding in full_ast_findings
        ]


def test_compact_class_index_detectors_match_full_ast_detection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "base.py").write_text(
        "class RegisteredStrategy:\n"
        "    __registry_key__ = 'method'\n"
        "    __skip_if_no_key__ = True\n",
        encoding="utf-8",
    )
    (package_root / "implementation.py").write_text(
        "from .base import RegisteredStrategy\n"
        "\n"
        "class ConcreteStrategy(RegisteredStrategy, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = 'method'\n"
        "    __skip_if_no_key__ = True\n"
        "    priority = 10\n"
        "\n"
        "def ordered():\n"
        "    return sorted(\n"
        "        ConcreteStrategy.__registry__.values(),\n"
        "        key=lambda strategy: strategy.priority,\n"
        "    )\n",
        encoding="utf-8",
    )
    detector_types = (
        systemic_detectors.InheritedAutoRegisterConfigBoilerplateDetector,
        systemic_detectors.AutoRegisterExplicitPriorityOrderingDetector,
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    projected_findings = accumulator.findings_by_detector(DetectorConfig())
    modules = parse_python_modules(package_root, use_parse_cache=False)

    for detector_type in detector_types:
        full_ast_findings = detector_type().detect(modules, DetectorConfig())
        assert [finding.to_dict() for finding in projected_findings[detector_type]] == [
            finding.to_dict() for finding in full_ast_findings
        ]


def test_compact_keyed_axis_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "specs.py").write_text(
        "class ModeConfig:\n"
        "    pass\n"
        "\n"
        "class ModeSpecPolicy(KeyedNominalFamily[Mode]):\n"
        "    registry_key_attr = 'mode'\n"
        "    family_label = 'mode case'\n"
        "\n"
        "class AlphaModeSpec(ModeSpecPolicy):\n"
        "    mode = Mode.ALPHA\n"
        "\n"
        "class BetaModeSpec(ModeSpecPolicy):\n"
        "    mode = Mode.BETA\n"
        "\n"
        "MODE_CONFIGS = {\n"
        "    Mode.ALPHA: ModeConfig(),\n"
        "    Mode.BETA: ModeConfig(),\n"
        "}\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text(
        "from .specs import Mode, KeyedNominalFamily\n"
        "\n"
        "class ModeRuntimePolicy(KeyedNominalFamily[Mode]):\n"
        "    registry_key_attr = 'mode'\n"
        "    family_label = 'mode case'\n"
        "\n"
        "class AlphaModeRuntime(ModeRuntimePolicy):\n"
        "    mode = Mode.ALPHA\n"
        "\n"
        "class BetaModeRuntime(ModeRuntimePolicy):\n"
        "    mode = Mode.BETA\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "from .specs import Mode\n"
        "\n"
        "MODE_HANDLERS = {Mode.ALPHA: alpha, Mode.BETA: beta}\n"
        "\n"
        "class ModeResolver:\n"
        "    @classmethod\n"
        "    def for_mode(cls, mode):\n"
        "        return MODE_HANDLERS[mode]\n"
        "\n"
        "def resolve(mode):\n"
        "    if mode == Mode.ALPHA:\n"
        "        return 'alpha'\n"
        "    return 'beta'\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    projections = (
        systemic_detectors.ParallelKeyedAxisFamilyDetector.compact_module_projections(
            modules
        )
    )

    legacy_family_specs = (
        systemic_detectors.DISPATCH_ALGEBRA_AUTHORITY.keyed_family_axis_specs(modules)
    )
    compact_family_specs = systemic_detectors._compact_keyed_family_axis_specs(
        projections
    )
    legacy_table_specs = tuple(
        table_spec
        for module in modules
        for table_spec in systemic_detectors.DISPATCH_ALGEBRA_AUTHORITY.module_keyed_table_axis_specs(
            module
        )
    )
    compact_table_specs = systemic_detectors._compact_keyed_table_axis_specs(
        projections
    )
    legacy_manual_selector_specs = systemic_detectors._manual_selector_axis_specs(
        modules
    )
    compact_manual_selector_specs = (
        systemic_detectors._compact_manual_selector_axis_specs(projections)
    )

    assert compact_family_specs == legacy_family_specs
    assert compact_table_specs == legacy_table_specs
    assert compact_manual_selector_specs == legacy_manual_selector_specs
    assert systemic_detectors._parallel_keyed_axis_family_candidates_from_specs(
        compact_family_specs
    ) == systemic_detectors._parallel_keyed_axis_family_candidates(modules)
    assert systemic_detectors._parallel_keyed_table_and_family_candidates_from_specs(
        compact_family_specs,
        compact_table_specs,
    ) == systemic_detectors._parallel_keyed_table_and_family_candidates(modules)
    assert systemic_detectors._parallel_keyed_table_axis_candidates_from_specs(
        compact_table_specs
    ) == systemic_detectors._parallel_keyed_table_axis_candidates(modules)
    assert systemic_detectors._residual_closed_axis_branching_candidates_from_compact_projections(
        projections
    ) == systemic_detectors._residual_closed_axis_branching_candidates(
        modules
    )
    assert systemic_detectors._cross_module_axis_shadow_family_candidates_from_specs(
        compact_family_specs,
        compact_manual_selector_specs,
    ) == systemic_detectors._cross_module_axis_shadow_family_candidates(modules)


def test_compact_top_level_definitions_match_private_helper_legacy_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "authority.py").write_text(
        "def normalize(value):\n"
        "    return value\n"
        "\n"
        "class Catalog:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "def _normalize(value):\n"
        "    return value\n"
        "\n"
        "class _Catalog:\n"
        "    pass\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    projections = (
        systemic_detectors.PrivateHelperShadowDetector.compact_module_projections(
            modules
        )
    )

    compact_candidates = (
        systemic_detectors._private_helper_shadow_candidates_from_definition_facts(
            tuple(
                (projection.file_path, projection.top_level_definitions)
                for projection in projections
            )
        )
    )

    assert compact_candidates == systemic_detectors._private_helper_shadow_candidates(
        modules
    )


def test_compact_dataclass_cli_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "config.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class RunConfig:\n"
        "    alpha: str\n"
        "    beta: int\n"
        "    gamma: bool\n"
        "    delta: float\n"
        "\n"
        "    @classmethod\n"
        "    def from_namespace(cls, namespace):\n"
        "        return cls(\n"
        "            alpha=namespace.alpha,\n"
        "            beta=namespace.beta,\n"
        "            gamma=namespace.gamma,\n"
        "            delta=namespace.delta,\n"
        "        )\n",
        encoding="utf-8",
    )
    (package_root / "cli.py").write_text(
        "RUN_ARGUMENTS = (\n"
        "    ArgumentSpec(flags=('--alpha',)),\n"
        "    ArgumentSpec(flags=('--beta',)),\n"
        "    ArgumentSpec(flags=('--gamma',)),\n"
        "    ArgumentSpec(flags=('--delta',)),\n"
        ")\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    projections = systemic_detectors.DataclassNamespaceCliMirrorDetector.compact_module_projections(
        modules
    )

    assert (
        systemic_detectors._dataclass_namespace_cli_mirror_candidates_from_projections(
            projections
        )
        == systemic_detectors._dataclass_namespace_cli_mirror_candidates(modules)
    )


def test_compact_exact_type_guard_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "class Boundary:\n"
        "    pass\n"
        "\n"
        "class ConcreteBoundary(Boundary):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "from .family import Boundary as ImportedBoundary\n"
        "\n"
        "def require_boundary(value):\n"
        "    if type(value) is not ImportedBoundary:\n"
        "        raise TypeError('boundary required')\n"
        "\n"
        "def assert_boundary(value):\n"
        "    assert type(value) is ImportedBoundary\n"
        "\n"
        "def shadowed_type(type, value):\n"
        "    assert type(value) is ImportedBoundary\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = runtime_detectors.ExactTypeGuardInheritanceRetreatDetector()
    legacy_findings = detector._findings_for_candidates(
        runtime_detectors.ExactTypeGuardBoundaryCollector.collect(modules),
        DetectorConfig(),
    )
    projections = detector.compact_module_projections(modules)
    compact_findings = detector._findings_for_candidates(
        runtime_detectors._exact_type_guard_candidates_from_compact_projections(
            projections
        ),
        DetectorConfig(),
    )

    assert [finding.to_dict() for finding in compact_findings] == [
        finding.to_dict() for finding in legacy_findings
    ]


def test_compact_semantic_inheritance_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class Exporter(ABC):\n"
        "    @abstractmethod\n"
        "    def emit(self, rows): ...\n"
        "\n"
        "class CsvExporter(Exporter):\n"
        "    format = 'csv'\n"
        "    def emit(self, rows): return rows\n"
        "\n"
        "class JsonExporter(Exporter):\n"
        "    format = 'json'\n"
        "    def emit(self, rows): return rows\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = runtime_detectors.SemanticInheritanceFamilySSOTDetector.compact_module_projections(
        modules
    )

    assert runtime_detectors._compact_semantic_inheritance_family_ssot_candidates(
        projections,
        config,
    ) == runtime_detectors._semantic_inheritance_family_ssot_candidates(
        list(modules),
        config,
    )


def test_compact_autoregister_rent_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "class Exporter(ABC, metaclass=AutoRegisterMeta):\n"
        "    @classmethod\n"
        "    def for_format(cls, name): return cls.__registry__[name]\n"
        "    @abstractmethod\n"
        "    def emit(self, rows): ...\n"
        "\n"
        "class CsvExporter(Exporter):\n"
        "    def emit(self, rows): return rows\n"
        "\n"
        "class JsonExporter(Exporter):\n"
        "    def emit(self, rows): return rows\n"
        "\n"
        "def materialize_exporters(specs):\n"
        "    return [AutoRegisterMeta(name, (Exporter,), body) for name, body in specs]\n"
        "\n"
        "def select_exporter(name):\n"
        "    return Exporter.for_format(name)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = runtime_detectors.AutoRegisterMetaUnderRentedDetector.compact_module_projections(
        modules
    )

    assert runtime_detectors._compact_autoregister_meta_rent_candidates(
        projections,
        config,
    ) == runtime_detectors._autoregister_meta_rent_candidates(
        list(modules),
        config,
    )


def test_compact_keyed_registry_axis_facts_match_legacy_ast_facts(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC\n"
        "from enum import Enum\n"
        "\n"
        "class Kind(Enum):\n"
        "    ALPHA = 'alpha'\n"
        "    BETA = 'beta'\n"
        "\n"
        "class Handler(KeyedNominalFamily[Kind], ABC):\n"
        "    registry_key_attr = 'kind'\n"
        "    @classmethod\n"
        "    def for_kind(cls, kind): return cls._registry[kind]\n"
        "    @classmethod\n"
        "    def type_for_kind(cls, kind): return cls._registry[kind]\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    kind = Kind.ALPHA\n"
        "\n"
        "class AliasAlphaHandler(Handler):\n"
        "    kind = Kind.ALPHA\n"
        "\n"
        "class MissingKeyHandler(Handler):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "consumers.py").write_text(
        "from .family import Handler\n"
        "\n"
        "def first(kind): return Handler.for_kind(kind)\n"
        "def second(kind): return Handler.type_for_kind(kind)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = (
        systemic_detectors.NonInjectiveTypeRegistryDetector.compact_module_projections(
            modules
        )
    )

    assert systemic_detectors._compact_keyed_registry_axis_facts(
        projections,
        config,
    ) == systemic_detectors.DISPATCH_ALGEBRA_AUTHORITY.keyed_registry_axis_fact_records(
        list(modules),
        config,
    )


def test_compact_registry_projection_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "core.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "from enum import Enum, auto\n"
        "from typing import Generic, TypeVar\n"
        "KeyT = TypeVar('KeyT')\n"
        "class AutoRegisterByClassVar: pass\n"
        "class KeyedNominalFamily(AutoRegisterByClassVar, Generic[KeyT]): pass\n"
        "class Mode(Enum):\n"
        "    ALPHA = auto()\n"
        "    BETA = auto()\n"
        "    GAMMA = auto()\n"
        "class ModeRunner(KeyedNominalFamily[Mode], ABC):\n"
        "    registry_key_attr = 'mode'\n"
        "    _registry = {}\n"
        "    @classmethod\n"
        "    def for_mode(cls, mode): return cls._registry[mode]\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
        "class AlphaModeRunner(ModeRunner):\n"
        "    mode = Mode.ALPHA\n"
        "    def run(self): return 'alpha'\n"
        "class BetaModeRunner(ModeRunner):\n"
        "    mode = Mode.BETA\n"
        "    def run(self): return 'beta'\n"
        "class GammaModeRunner(ModeRunner):\n"
        "    mode = Mode.GAMMA\n"
        "    def run(self): return 'gamma'\n"
        "def run_alpha(): return ModeRunner.for_mode(Mode.ALPHA).run()\n"
        "def run_beta(): return ModeRunner.for_mode(Mode.BETA).run()\n",
        encoding="utf-8",
    )
    (package_root / "config.py").write_text(
        "from pkg.core import (\n"
        "    AlphaModeRunner as Alpha, BetaModeRunner as Beta, Mode as ModeKey,\n"
        ")\n"
        "PUBLIC_MODE_CHOICES = (ModeKey.ALPHA, ModeKey.BETA)\n"
        "PUBLIC_MODE_TYPES = {ModeKey.ALPHA: Alpha, ModeKey.BETA: Beta}\n"
        "DUAL_MODE_SURFACE = (ModeKey.ALPHA, ModeKey.BETA)\n"
        "DUAL_MODE_SURFACE = {ModeKey.ALPHA: Alpha, ModeKey.BETA: Beta}\n"
        "ALIAS_NAME_STRINGS = ('Alpha', 'Beta')\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = (
        systemic_detectors.RegistryProjectionSurfaceDetector.compact_module_projections(
            modules
        )
    )
    facts = systemic_detectors._compact_keyed_registry_axis_facts(projections, config)

    assert (
        systemic_detectors._compact_registry_projection_surface_candidates_from_facts(
            projections, facts
        )
        == systemic_detectors._REGISTRY_PROJECTION_SURFACE_ANALYZER.surface_candidates(
            list(modules), config
        )
    )
    assert systemic_detectors._compact_registry_projection_policy_authority_candidates_from_facts(
        projections, facts
    ) == systemic_detectors._REGISTRY_PROJECTION_SURFACE_ANALYZER.policy_authority_candidates(
        list(modules), config
    )


def test_keyed_registry_detectors_share_one_compact_fact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC\n"
        "\n"
        "class Handler(KeyedNominalFamily[str], ABC):\n"
        "    registry_key_attr = 'kind'\n"
        "    @classmethod\n"
        "    def for_kind(cls, kind): return cls._registry[kind]\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    kind = 'alpha'\n"
        "\n"
        "class BetaHandler(Handler):\n"
        "    kind = 'beta'\n",
        encoding="utf-8",
    )
    detector_types = (
        systemic_detectors.NonInjectiveTypeRegistryDetector,
        systemic_detectors.InjectiveTypeRegistryDetector,
        systemic_detectors.PrematureRegistryInfrastructureDetector,
        systemic_detectors.RegistryProjectionSurfaceDetector,
        systemic_detectors.RegistryProjectionPolicyAuthorityDetector,
    )
    calls = 0
    original_builder = systemic_detectors._compact_keyed_registry_axis_facts

    def counting_builder(projections, config):
        nonlocal calls
        calls += 1
        return original_builder(projections, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_builder),
        )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def test_compact_repeated_keyed_family_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    for module_name, class_name, key_name in (
        ("alpha", "AlphaPolicy", "alpha"),
        ("beta", "BetaPolicy", "beta"),
        ("gamma", "GammaPolicy", "gamma"),
    ):
        (package_root / f"{module_name}.py").write_text(
            "from abc import ABC, abstractmethod\n"
            "\n"
            f"class {class_name}(AutoRegisterByClassVar, ABC):\n"
            f"    registry_key_attr = '{key_name}'\n"
            "    _registry = {}\n"
            "    @classmethod\n"
            f"    def for_{key_name}(cls, key):\n"
            "        try:\n"
            "            return cls._registry[key]\n"
            "        except KeyError as error:\n"
            "            raise ValueError(key) from error\n"
            "    @abstractmethod\n"
            "    def run(self): ...\n",
            encoding="utf-8",
        )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = (
        systemic_detectors.RepeatedKeyedFamilyDetector.compact_module_projections(
            modules
        )
    )

    assert systemic_detectors._compact_repeated_keyed_family_candidates(
        projections,
        config,
    ) == systemic_detectors._repeated_keyed_family_candidates(
        list(modules),
        config,
    )


def test_compact_concrete_family_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "families.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class RenderRule(ABC):\n"
        "    _registered_types = []\n"
        "    @classmethod\n"
        "    def registered_types(cls): return tuple(cls._registered_types)\n"
        "    @classmethod\n"
        "    def resolve(cls, artifact):\n"
        "        matches = [candidate for candidate in cls.registered_types() "
        "if candidate.matches_context(artifact)]\n"
        "        if not matches: raise ValueError(artifact)\n"
        "        if len(matches) > 1: raise TypeError(artifact)\n"
        "        return matches[0]()\n"
        "\n"
        "class InvoiceFieldEmitter(ABC):\n"
        "    _registered_types = []\n"
        "    @abstractmethod\n"
        "    def emit(self, artifact): ...\n"
        "\n"
        "class ReceiptFieldEmitter(ABC):\n"
        "    _registered_types = []\n"
        "    @abstractmethod\n"
        "    def emit(self, artifact): ...\n"
        "\n"
        "class AlphaRenderRule(RenderRule): pass\n"
        "class BetaRenderRule(RenderRule): pass\n"
        "class InvoiceAlphaEmitter(InvoiceFieldEmitter): pass\n"
        "class InvoiceBetaEmitter(InvoiceFieldEmitter): pass\n"
        "class InvoiceGammaEmitter(InvoiceFieldEmitter): pass\n"
        "class ReceiptAlphaEmitter(ReceiptFieldEmitter): pass\n"
        "class ReceiptBetaEmitter(ReceiptFieldEmitter): pass\n"
        "class ReceiptGammaEmitter(ReceiptFieldEmitter): pass\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = runtime_detectors.PredicateSelectedConcreteFamilyDetector.compact_module_projections(
        modules
    )
    context = runtime_detectors._compact_concrete_family_context(projections, config)

    assert runtime_detectors._compact_predicate_selected_concrete_family_candidates(
        context, config
    ) == runtime_detectors._predicate_selected_concrete_family_candidates(
        list(modules), config
    )
    assert runtime_detectors._compact_parallel_mirrored_leaf_family_candidates(
        context, config
    ) == runtime_detectors._parallel_mirrored_leaf_family_candidates(
        list(modules), config
    )


def test_compact_roster_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "base.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class RoutedRequest(ABC):\n"
        "    route_name = None\n"
        "    _registered_types = []\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        if cls.__dict__.get('route_name') is not None:\n"
        "            cls._registered_types.append(cls)\n"
        "    @classmethod\n"
        "    def concrete_types(cls): return tuple(cls._registered_types)\n"
        "\n"
        "class Exporter(ABC):\n"
        "    @abstractmethod\n"
        "    def emit(self, rows): ...\n"
        "\n"
        "EXPORT_FORMATS = ('csv', 'json')\n",
        encoding="utf-8",
    )
    (package_root / "implementations.py").write_text(
        "from .base import Exporter, RoutedRequest\n"
        "\n"
        "class DirectRequest(RoutedRequest): route_name = 'direct'\n"
        "class GuidedRequest(RoutedRequest): route_name = 'guided'\n"
        "class CsvExporter(Exporter):\n"
        "    format = 'csv'\n"
        "    def emit(self, rows): return rows\n"
        "class JsonExporter(Exporter):\n"
        "    format = 'json'\n"
        "    def emit(self, rows): return rows\n"
        "\n"
        "DEFAULT_EXPORTERS = (CsvExporter(), JsonExporter())\n",
        encoding="utf-8",
    )
    (package_root / "a_shadow.py").write_text(
        "def hidden_duplicate_names():\n"
        "    class CsvExporter(Local): pass\n"
        "    class JsonExporter(Local): pass\n"
        "    return CsvExporter, JsonExporter\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = runtime_detectors.ManualConcreteSubclassRosterDetector.compact_module_projections(
        modules
    )
    context = runtime_detectors._compact_concrete_family_context(projections, config)

    assert runtime_detectors._compact_manual_concrete_subclass_roster_candidates(
        projections, context, config
    ) == runtime_detectors._manual_concrete_subclass_roster_candidates(
        list(modules), config
    )
    assert runtime_detectors._compact_latent_implementation_roster_candidates(
        context, config
    ) == runtime_detectors._latent_implementation_roster_candidates(
        list(modules), config
    )
    legacy_index = surface_detectors.NominalAuthorityIndex(modules)
    assert surface_detectors._compact_manual_family_roster_candidates(
        projections, context
    ) == tuple(
        candidate
        for module in modules
        for candidate in surface_detectors._manual_family_roster_candidates(
            module, legacy_index
        )
    )
    manual_family_detector = surface_detectors.ManualFamilyRosterDetector()
    assert manual_family_detector._findings_from_compact_context(
        projections, context, config
    ) == manual_family_detector._collect_findings(list(modules), config)


def test_concrete_family_detectors_share_one_compact_graph_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("class Root: pass\n", encoding="utf-8")
    detector_types = (
        runtime_detectors.ManualConcreteSubclassRosterDetector,
        runtime_detectors.LatentImplementationRosterDetector,
        runtime_detectors.PredicateSelectedConcreteFamilyDetector,
        runtime_detectors.ParallelMirroredLeafFamilyDetector,
        surface_detectors.ManualFamilyRosterDetector,
    )
    calls = 0
    original_builder = runtime_detectors._compact_concrete_family_context

    def counting_builder(projections, config):
        nonlocal calls
        calls += 1
        return original_builder(projections, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_builder),
        )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,), detector_types, use_parse_cache=False
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def test_compact_nominal_authority_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "authority.py").write_text(
        "from abc import ABC\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class JobSpecBase(ABC):\n"
        "    name: str\n"
        "    priority: int\n"
        "    def start(self, value): return value\n"
        "    def stop(self, value): return value\n",
        encoding="utf-8",
    )
    (package_root / "duplicate.py").write_text(
        "from dataclasses import dataclass\n"
        "from .authority import JobSpecBase\n"
        "@dataclass\n"
        "class JobSpecCopy:\n"
        "    name: str\n"
        "    priority: int\n"
        "class JobSpecWrapper:\n"
        "    delegate: JobSpecBase\n"
        "    def start(self, value): return self.delegate.start(value)\n"
        "    def stop(self, value): return self.delegate.stop(value)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    detector = surface_detectors.ExistingNominalAuthorityReuseDetector()
    projections = detector.compact_module_projections(modules)
    compact_index = surface_detectors._compact_nominal_authority_index(
        projections, config
    )
    legacy_index = surface_detectors.NominalAuthorityIndex(modules)

    assert surface_detectors._existing_nominal_authority_reuse_candidates_from_index(
        compact_index
    ) == surface_detectors._existing_nominal_authority_reuse_candidates_from_index(
        legacy_index
    )
    assert surface_detectors._nominal_authority_implementation_retreat_candidates_from_index(
        compact_index
    ) == surface_detectors._nominal_authority_implementation_retreat_candidates_from_index(
        legacy_index
    )
    for detector_type, collector in (
        (
            surface_detectors.ExistingNominalAuthorityReuseDetector,
            surface_detectors._existing_nominal_authority_reuse_candidates_from_index,
        ),
        (
            surface_detectors.NominalAuthorityImplementationRetreatDetector,
            surface_detectors._nominal_authority_implementation_retreat_candidates_from_index,
        ),
    ):
        instance = detector_type()
        assert instance._findings_from_compact_context(
            projections, compact_index, config
        ) == instance._findings_for_candidates(collector(legacy_index), config)
    compact_wrapper_candidates = (
        surface_detectors._compact_pass_through_nominal_wrapper_candidates(projections)
    )
    legacy_wrapper_candidates = (
        surface_detectors._pass_through_nominal_wrapper_candidates(modules)
    )
    assert compact_wrapper_candidates == legacy_wrapper_candidates
    wrapper_detector = surface_detectors.PassThroughNominalWrapperDetector()
    assert wrapper_detector._findings_from_compact_projections(projections, config) == [
        wrapper_detector._finding_for_candidate(candidate)
        for candidate in legacy_wrapper_candidates
    ]


def test_compact_duplicate_nominal_surface_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "authority.py").write_text(
        "class JobAuthority:\n"
        "    job_name: str\n"
        "    job_path: str\n"
        "    def run(self):\n"
        "        return self.job_name, self.job_path\n",
        encoding="utf-8",
    )
    (package_root / "shell.py").write_text(
        "from .authority import JobAuthority\n"
        "class JobShell:\n"
        "    job_name: str\n"
        "    job_path: str\n"
        "    def run(self):\n"
        "        return self.job_name, self.job_path\n"
        "    def build(self):\n"
        "        return JobAuthority(self.job_name, self.job_path)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    detector = surface_detectors.DuplicateNominalAuthoritySurfaceDetector()
    projections = detector.compact_module_projections(modules)
    compact_candidates = (
        surface_detectors._compact_duplicate_nominal_authority_surface_candidates(
            projections
        )
    )
    legacy_candidates = (
        surface_detectors._duplicate_nominal_authority_surface_candidates(modules)
    )

    assert compact_candidates == legacy_candidates
    assert detector._findings_from_compact_projections(
        projections, config
    ) == detector._findings_for_candidates(legacy_candidates, config)


def test_nominal_surface_indexed_components_match_axis_graph() -> None:
    def surface_node(
        class_name: str,
        public_method_names: tuple[str, ...],
        method_flow_roles: tuple[tuple[str, tuple[str, ...]], ...],
    ):
        return nominal_surface_detectors._NominalAuthoritySurfaceNode(
            shape=surface_detectors.NominalAuthorityShape(
                file_path="fixture.py",
                class_name=class_name,
                line=len(class_name),
                declared_base_names=(),
                ancestor_names=(),
                field_names=("job_name", "job_path"),
                field_type_map=(("job_name", "str"), ("job_path", "str")),
                method_names=public_method_names,
                is_abstract=False,
                is_dataclass_family=False,
            ),
            field_roles=("job",),
            public_method_names=public_method_names,
            method_flow_roles=method_flow_roles,
            constructed_delegate_names=(),
        )

    nodes = (
        surface_node("Alpha", ("extra", "run"), (("run", ("name",)),)),
        surface_node("Beta", ("extra", "run"), (("run", ("path",)),)),
        surface_node("Gamma", ("other", "run"), (("run", ("path",)),)),
    )
    axis_system = FiniteAxisSystem.from_rows(
        (
            (
                node,
                {
                    "field_roles": node.field_roles,
                    "method_names": node.public_method_names,
                    "method_flow_roles": node.method_flow_roles,
                },
            )
            for node in nodes
        )
    )
    expected = axis_system.confusability_graph(
        (
            ("field_roles", "method_names"),
            ("field_roles", "method_flow_roles"),
        )
    ).connected_components

    assert expected == (nodes,)
    assert (
        nominal_surface_detectors._surface_confusability_components(nodes) == expected
    )


def test_nominal_authority_detectors_share_one_compact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("class Root: pass\n", encoding="utf-8")
    detector_types = (
        surface_detectors.ExistingNominalAuthorityReuseDetector,
        surface_detectors.NominalAuthorityImplementationRetreatDetector,
    )
    calls = 0
    original_builder = surface_detectors._compact_nominal_authority_index

    def counting_builder(projections, config):
        nonlocal calls
        calls += 1
        return original_builder(projections, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_builder),
        )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,), detector_types, use_parse_cache=False
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def _write_compact_abc_optimizer_fixture(package_root: Path) -> None:
    package_root.mkdir()
    (package_root / "workers.py").write_text(
        "from abc import ABC\n"
        "class Worker(ABC):\n"
        "    pass\n"
        "class CsvWorker(Worker):\n"
        "    FORMAT_VERSION = 1\n"
        "    SHARED_MODE = 'batch'\n"
        "    def emit(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = encode_csv(clean)\n"
        "        self.write(value, suffix='.csv')\n"
        "        return value\n"
        "    def validate(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = validate_csv(clean)\n"
        "        self.write(value, suffix='.csv')\n"
        "        return value\n"
        "    def audit(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = audit_tabular(clean)\n"
        "        self.write(value, suffix='.csv')\n"
        "        return value\n"
        "    def poison(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = poison_csv(clean)\n"
        "        return value\n"
        "class JsonWorker(Worker):\n"
        "    FORMAT_VERSION = 1\n"
        "    SHARED_MODE = 'batch'\n"
        "    def emit(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = encode_json(clean)\n"
        "        self.write(value, suffix='.json')\n"
        "        return value\n"
        "    def validate(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = validate_json(clean)\n"
        "        self.write(value, suffix='.json')\n"
        "        return value\n"
        "    def audit(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = audit_tabular(clean)\n"
        "        self.write(value, suffix='.json')\n"
        "        return value\n"
        "    def cache(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = cache_payload(clean)\n"
        "        self.write(value, suffix='.json')\n"
        "        return value\n"
        "    def poison(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = poison_json(clean)\n"
        "        return value\n"
        "class XmlWorker(Worker):\n"
        "    FORMAT_VERSION = 1\n"
        "    SHARED_MODE = 'batch'\n"
        "    def emit(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = encode_xml(clean)\n"
        "        self.write(value, suffix='.xml')\n"
        "        return value\n"
        "    def validate(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = validate_xml(clean)\n"
        "        self.write(value, suffix='.xml')\n"
        "        return value\n"
        "    def cache(self, rows):\n"
        "        clean = self.normalize(rows)\n"
        "        value = cache_payload(clean)\n"
        "        self.write(value, suffix='.xml')\n"
        "        return value\n"
        "    def poison(self, rows):\n"
        "        return rows\n",
        encoding="utf-8",
    )


def test_compact_abc_optimizer_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_abc_optimizer_fixture(package_root)
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = structural_detectors.ClassLevelInheritanceOptimizationDetector.compact_module_projections(
        modules
    )
    context = structural_detectors._compact_abc_optimizer_context(projections, config)
    detector_legacy_pairs = (
        (
            structural_detectors.ClassLevelInheritanceOptimizationDetector,
            context.class_level_candidates,
            structural_detectors._class_level_inheritance_optimization_candidates_from_modules(
                modules
            ),
        ),
        (
            structural_detectors.SemanticOverlapAbcOptimizationDetector,
            context.method_candidates,
            structural_detectors._semantic_overlap_abc_optimization_candidates_from_modules(
                modules
            ),
        ),
        (
            structural_detectors.SemanticOverlapAbcFamilyOptimizationDetector,
            context.family_candidates,
            structural_detectors._semantic_overlap_abc_family_optimization_candidates(
                modules
            ),
        ),
        (
            structural_detectors.GlobalInheritanceOptimizationDetector,
            context.global_candidates,
            structural_detectors._semantic_overlap_global_inheritance_candidates(
                modules
            ),
        ),
        (
            structural_detectors.SemanticOverlapAbcResidueAxisCatalogDetector,
            context.residue_axis_candidates,
            structural_detectors._semantic_overlap_abc_residue_axis_catalog_candidates(
                modules
            ),
        ),
    )

    for detector_type, compact_candidates, legacy_candidates in detector_legacy_pairs:
        assert compact_candidates == legacy_candidates
        detector = detector_type()
        assert detector._findings_from_compact_context(
            projections, context, config
        ) == detector._findings_for_candidates(legacy_candidates, config)
    assert context.class_level_candidates
    assert context.method_candidates
    assert context.family_candidates
    assert context.global_candidates
    assert context.residue_axis_candidates
    assert all(
        candidate.method_name != "poison" for candidate in context.method_candidates
    )


def test_abc_optimizer_detectors_share_one_compact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_abc_optimizer_fixture(package_root)
    detector_types = (
        structural_detectors.ClassLevelInheritanceOptimizationDetector,
        structural_detectors.SemanticOverlapAbcOptimizationDetector,
        structural_detectors.SemanticOverlapAbcFamilyOptimizationDetector,
        structural_detectors.GlobalInheritanceOptimizationDetector,
        structural_detectors.SemanticOverlapAbcResidueAxisCatalogDetector,
    )
    calls = 0
    original_builder = structural_detectors._compact_abc_optimizer_context

    def counting_builder(projections, config):
        nonlocal calls
        calls += 1
        return original_builder(projections, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_builder),
        )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,), detector_types, use_parse_cache=False
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def _write_compact_carrier_reuse_fixture(package_root: Path) -> None:
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "models.py").write_text(
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "@dataclass(frozen=True)\n"
        "class RequestCarrier:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n",
        encoding="utf-8",
    )
    (package_root / "local.py").write_text(
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "from .models import RequestCarrier as RC\n"
        "@dataclass(frozen=True)\n"
        "class LocalEnvelope:\n"
        "    request_id: str\n"
        "    source_path: Path\n"
        "    workspace_root: Path\n"
        "@dataclass(frozen=True)\n"
        "class ComposedRequest:\n"
        "    carrier: 'RC'\n",
        encoding="utf-8",
    )


def test_compact_carrier_reuse_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_carrier_reuse_fixture(package_root)
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = abstraction_reuse_detectors.AvailableCarrierReuseDetector.compact_module_projections(
        modules
    )
    context = abstraction_reuse_detectors._compact_carrier_reuse_context(
        projections, config
    )
    carrier_surfaces = abstraction_reuse_detectors._carrier_surfaces_with_ancestors(
        abstraction_reuse_detectors._compact_carrier_surfaces(projections)
    )
    assert abstraction_reuse_detectors._available_carrier_reuse_candidates_from_surfaces(
        carrier_surfaces
    ) == abstraction_reuse_detectors._exhaustive_available_carrier_reuse_candidates_from_surfaces(
        carrier_surfaces
    )
    detector_legacy_pairs = (
        (
            abstraction_reuse_detectors.AvailableCarrierReuseDetector,
            context.available_candidates,
            abstraction_reuse_detectors._available_carrier_reuse_candidates(modules),
        ),
        (
            abstraction_reuse_detectors.CarrierCompositionRetreatDetector,
            context.composition_candidates,
            abstraction_reuse_detectors._carrier_composition_retreat_candidates(
                modules
            ),
        ),
        (
            abstraction_reuse_detectors.ParallelPrimitiveCarrierDetector,
            context.parallel_candidates,
            abstraction_reuse_detectors._parallel_primitive_carrier_candidates(
                list(modules)
            ),
        ),
    )

    for detector_type, compact_candidates, legacy_candidates in detector_legacy_pairs:
        assert compact_candidates == legacy_candidates
        detector = detector_type()
        assert detector._findings_from_compact_context(
            projections, context, config
        ) == detector._findings_for_candidates(legacy_candidates, config)
        assert compact_candidates


def test_compact_available_abstraction_reuse_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    shared_root = package_root / "shared"
    shared_root.mkdir(parents=True)
    (shared_root / "button_panel.py").write_text(
        "class ButtonPanel:\n"
        "    def __init__(self, button_configs, on_action, style_generator=None, parent=None):\n"
        "        layout = QGridLayout(self)\n"
        "        layout.setContentsMargins(5, 5, 5, 5)\n"
        "        layout.setSpacing(5)\n"
        "        self.buttons = {}\n"
        "        for index, (label, action_id, tooltip) in enumerate(button_configs):\n"
        "            button = QPushButton(label)\n"
        "            button.setToolTip(tooltip)\n"
        "            if style_generator:\n"
        "                button.setStyleSheet(style_generator.generate_button_style())\n"
        "            button.clicked.connect(lambda checked, a=action_id: on_action(a))\n"
        "            self.buttons[action_id] = button\n"
        "            layout.addWidget(button, 0, index)\n",
        encoding="utf-8",
    )
    (package_root / "debug_toolbar.py").write_text(
        "class DebugToolbarWidget:\n"
        "    BUTTONS = (('Run', 'run', 'Run'), ('Stop', 'stop', 'Stop'))\n\n"
        "    def __init__(self, style_generator=None):\n"
        "        layout = QVBoxLayout(self)\n"
        "        layout.setContentsMargins(0, 0, 0, 0)\n"
        "        layout.setSpacing(0)\n"
        "        self.buttons = {}\n"
        "        for label, action_id, tooltip in self.BUTTONS:\n"
        "            button = QPushButton(label)\n"
        "            button.setToolTip(tooltip)\n"
        "            if style_generator:\n"
        "                button.setStyleSheet(style_generator.generate_button_style())\n"
        "            button.clicked.connect(lambda checked, a=action_id: self.emit(a))\n"
        "            self.buttons[action_id] = button\n"
        "            layout.addWidget(button, 0, index)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(tmp_path, use_parse_cache=False))
    detector = abstraction_reuse_detectors.AvailableAbstractionReuseDetector()
    projections = type(detector).compact_module_projections(modules)
    legacy_candidates = (
        abstraction_reuse_detectors._available_abstraction_reuse_candidates(modules)
    )
    compact_candidates = (
        abstraction_reuse_detectors._compact_available_abstraction_reuse_candidates(
            projections
        )
    )

    assert compact_candidates == legacy_candidates
    assert compact_candidates
    config = DetectorConfig()
    assert detector._findings_from_compact_projections(
        projections, config
    ) == detector._findings_for_candidates(legacy_candidates, config)
    accumulator = accumulate_compact_global_projections_for_roots(
        (tmp_path,),
        (type(detector),),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == len(modules)
    assert accumulator.findings_by_detector(config)[type(detector)] == (
        detector._findings_for_candidates(legacy_candidates, config)
    )


def test_compact_public_private_delegate_context_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "scoring.py").write_text(
        "class _Router:\n"
        "    @classmethod\n"
        "    def for_engine(cls, engine):\n"
        "        return cls()\n\n"
        "    def score(self, payload):\n"
        "        return payload['value']\n\n"
        "    def requires_electrostatics(self):\n"
        "        return True\n\n"
        "def route_scoring(engine, **payload):\n"
        "    return _Router.for_engine(engine).score(payload)\n\n"
        "def scoring_engine_requires_electrostatics(engine):\n"
        "    return _Router.for_engine(engine).requires_electrostatics()\n",
        encoding="utf-8",
    )
    consumer_source = (
        "from pkg.scoring import route_scoring, scoring_engine_requires_electrostatics\n\n"
        "def score_request():\n"
        "    if scoring_engine_requires_electrostatics('fast'):\n"
        "        return route_scoring('fast', value=1.0)\n"
        "    return 0.0\n"
    )
    (package_root / "pipeline.py").write_text(consumer_source, encoding="utf-8")
    (package_root / "api.py").write_text(consumer_source, encoding="utf-8")
    modules = tuple(parse_python_modules(tmp_path, use_parse_cache=False))
    config = DetectorConfig()
    shell_detector = runtime_detectors.PublicApiPrivateDelegateShellDetector()
    family_detector = runtime_detectors.PublicApiPrivateDelegateFamilyDetector()
    projections = type(shell_detector).compact_module_projections(modules)
    context = runtime_detectors._compact_public_api_private_delegate_context(
        projections, config
    )
    legacy_shell_candidates = (
        runtime_detectors._public_api_private_delegate_shell_candidates(modules, config)
    )
    legacy_family_candidates = (
        runtime_detectors._public_api_private_delegate_family_candidates(
            modules, config
        )
    )

    assert context.shell_candidates == legacy_shell_candidates
    assert context.family_candidates == legacy_family_candidates
    assert len(context.shell_candidates) == 2
    assert len(context.family_candidates) == 1
    assert shell_detector._findings_from_compact_context(
        projections, context, config
    ) == shell_detector._findings_for_candidates(legacy_shell_candidates, config)
    assert family_detector._findings_from_compact_context(
        projections, context, config
    ) == family_detector._findings_for_candidates(legacy_family_candidates, config)
    detector_types = (type(shell_detector), type(family_detector))
    accumulator = accumulate_compact_global_projections_for_roots(
        (tmp_path,), detector_types, use_parse_cache=False
    )
    findings = accumulator.findings_by_detector(config)
    assert accumulator.projection_count == len(modules)
    assert len(findings[type(shell_detector)]) == 2
    assert len(findings[type(family_detector)]) == 1


def test_carrier_reuse_detectors_share_one_compact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_carrier_reuse_fixture(package_root)
    detector_types = (
        abstraction_reuse_detectors.AvailableCarrierReuseDetector,
        abstraction_reuse_detectors.CarrierCompositionRetreatDetector,
        abstraction_reuse_detectors.ParallelPrimitiveCarrierDetector,
    )
    calls = 0
    original_builder = abstraction_reuse_detectors._compact_carrier_reuse_context

    def counting_builder(projections, config):
        nonlocal calls
        calls += 1
        return original_builder(projections, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_builder),
        )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,), detector_types, use_parse_cache=False
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def _write_compact_private_helper_cluster_fixture(package_root: Path) -> None:
    package_root.mkdir()
    (package_root / "helpers.py").write_text(
        "def _class_field_names(node):\n"
        "    names = []\n"
        "    for item in node.body:\n"
        "        if isinstance(item, AnnAssign):\n"
        "            names.append(item.target)\n"
        "    return tuple(names)\n\n"
        "def _class_method_names(node):\n"
        "    names = []\n"
        "    for item in node.body:\n"
        "        if isinstance(item, FunctionDef):\n"
        "            names.append(item.name)\n"
        "    return tuple(names)\n\n"
        "def _class_base_names(node):\n"
        "    names = []\n"
        "    for item in node.bases:\n"
        "        if isinstance(item, Name):\n"
        "            names.append(item.id)\n"
        "    return tuple(names)\n\n"
        "def _class_decorator_names(node):\n"
        "    names = []\n"
        "    for item in node.decorator_list:\n"
        "        if isinstance(item, Name):\n"
        "            names.append(item.id)\n"
        "    return tuple(names)\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "def inspect_fields(node):\n" "    return _class_field_names(node)\n",
        encoding="utf-8",
    )


def test_compact_role_guarded_surface_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "roles.py").write_text(
        "class AvoidWidgetsWindow:\n"
        "    def position_avoid_widgets(self):\n"
        "        raise NotImplementedError\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "def place_window(window):\n"
        "    if isinstance(window, AvoidWidgetsWindow):\n"
        "        return tuple(window.position_avoid_widgets())\n"
        "    return ()\n\n"
        "def inspect_window(window):\n"
        "    if isinstance(window, AvoidWidgetsWindow):\n"
        "        return window.windowTitle()\n"
        "    return None\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = runtime_detectors.RoleGuardedSurfaceAccessDetector()
    projections = type(detector).compact_module_projections(modules)
    compact_candidates = (
        runtime_detectors._compact_role_guarded_surface_access_candidates(projections)
    )
    role_surfaces = runtime_detectors._role_surface_members_by_type_name(modules)
    legacy_candidates = tuple(
        candidate
        for module in modules
        for candidate in runtime_detectors._role_guarded_surface_access_candidates_for_module(
            module,
            role_surfaces,
        )
    )

    assert compact_candidates == legacy_candidates
    assert len(compact_candidates) == 1
    assert compact_candidates[0].accessed_members == ("position_avoid_widgets",)
    assert detector._findings_from_compact_projections(
        projections,
        DetectorConfig(),
    ) == [detector._finding_for_candidate(candidate) for candidate in legacy_candidates]


def test_compact_non_nominal_private_helper_matches_legacy_multi_family_join(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "helpers.py").write_text(
        "def _build_plan(value, mode):\n"
        "    first = normalize(value)\n"
        "    second = prepare(first, mode)\n"
        "    third = validate(second)\n"
        "    fourth = transform(third)\n"
        "    fifth = finalize(fourth)\n"
        "    sixth = audit(fifth)\n"
        "    seventh = publish(sixth)\n"
        "    return seventh\n\n"
        "class BaseRunner:\n"
        "    pass\n\n"
        "class AlphaRunner(BaseRunner):\n"
        "    def run(self, value):\n"
        "        return _build_plan(value, 'alpha')\n\n"
        "class BetaRunner(BaseRunner):\n"
        "    def run(self, value):\n"
        "        return _build_plan(value, self.mode)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = runtime_detectors.NonNominalPrivateHelperDetector()
    projection_groups = type(detector).compact_module_projection_groups(modules)
    private_projections = projection_groups[
        runtime_detectors.CompactPrivateReferenceModuleProjectionFamily
    ]
    class_projections = projection_groups[
        runtime_detectors.CompactModuleClassProjectionFamily
    ]
    config = DetectorConfig()
    compact_candidates = (
        runtime_detectors._compact_non_nominal_private_helper_candidates(
            private_projections,
            class_projections,
            config,
        )
    )
    legacy_context = runtime_detectors.PrivateReferenceDetectorContext(modules)
    legacy_candidates = tuple(
        candidate
        for module in modules
        for candidate in runtime_detectors._non_nominal_private_helper_candidates(
            module,
            config,
            reference_modules=modules,
            derived_candidate_collector_contract_names=(
                legacy_context.derived_candidate_collector_contract_names
            ),
            private_helper_call_graph=legacy_context.private_helper_call_graph,
            class_index=legacy_context.class_index,
        )
    )

    assert compact_candidates == legacy_candidates
    assert len(compact_candidates) == 1
    placement = compact_candidates[0].placement_plan
    assert placement.placement_kind == "existing_inheritance_root"
    assert placement.insertion_owner_name == "BaseRunner"
    assert placement.residue_plan.transported_parameter_names == ("value",)
    assert placement.residue_plan.callsite_axis_count == 1
    assert detector._findings_from_compact_projection_groups(
        projection_groups,
        config,
    ) == [detector._finding_for_candidate(candidate) for candidate in legacy_candidates]
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (
            runtime_detectors.NonNominalPrivateHelperDetector,
            runtime_detectors.PrivateHelperSemanticClusterDetector,
            runtime_detectors.ManualConcreteSubclassRosterDetector,
        ),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == 2
    assert accumulator.findings_by_detector(config)[
        runtime_detectors.NonNominalPrivateHelperDetector
    ] == [detector._finding_for_candidate(candidate) for candidate in legacy_candidates]


def test_compact_private_helper_cluster_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_private_helper_cluster_fixture(package_root)
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    detector = runtime_detectors.PrivateHelperSemanticClusterDetector()
    projections = type(detector).compact_module_projections(modules)
    compact_candidates = (
        runtime_detectors._compact_private_helper_semantic_cluster_candidates(
            projections,
            config,
        )
    )
    legacy_context = runtime_detectors.PrivateReferenceDetectorContext(modules)
    legacy_candidates = tuple(
        candidate
        for module in modules
        for candidate in runtime_detectors._private_helper_semantic_cluster_candidates(
            module,
            config,
            reference_modules=modules,
            derived_candidate_collector_contract_names=(
                legacy_context.derived_candidate_collector_contract_names
            ),
            private_helper_call_graph=legacy_context.private_helper_call_graph,
        )
    )

    assert compact_candidates == legacy_candidates
    assert compact_candidates
    assert "inspect_fields" in compact_candidates[0].consumer_symbols
    assert detector._findings_from_compact_projections(
        projections,
        config,
    ) == [detector._finding_for_candidate(candidate) for candidate in legacy_candidates]


def test_compact_distributed_boundary_graph_matches_legacy_global_join(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "models_a.py").write_text(
        "class RuntimeAdapter:\n"
        "    axis_id: str\n"
        "    axis_scope: object\n\n"
        "class ArtifactQuery:\n"
        "    axis_scope: object\n",
        encoding="utf-8",
    )
    (package_root / "models_b.py").write_text(
        "class PlaneResolution:\n"
        "    axis_id: str\n\n"
        "class CacheKey:\n"
        "    axis_scope: object\n",
        encoding="utf-8",
    )
    (package_root / "uses.py").write_text(
        "def resolve(adapter):\n"
        "    return PlaneResolution(axis_id=adapter.axis_id)\n\n"
        "def rebuild(adapter):\n"
        "    return RuntimeAdapter(axis_id=adapter.axis_id)\n\n"
        "def project(adapter):\n"
        "    axis_key = adapter.axis_id\n"
        "    return axis_key\n\n"
        "def artifact(adapter):\n"
        "    return ArtifactQuery(axis_scope=adapter.axis_scope)\n\n"
        "def cache(adapter):\n"
        "    return CacheKey(axis_scope=adapter.axis_scope)\n\n"
        "def project_scope(query):\n"
        "    runtime_scope = query.axis_scope\n"
        "    return runtime_scope\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    fanout_detector = surface_detectors.DistributedBoundaryFanoutDetector()
    wrapper_detector = surface_detectors.BoundaryLocalWrapperCollapseDetector()
    projections = fanout_detector.compact_module_projections(modules)
    compact_fanout = surface_detectors._compact_distributed_boundary_fanout_candidates(
        projections,
        config,
    )
    legacy_fanout = surface_detectors._distributed_boundary_fanout_candidates(
        modules,
        config,
    )
    compact_wrappers = (
        surface_detectors._compact_boundary_local_wrapper_collapse_candidates(
            projections,
            config,
        )
    )
    wrappers_from_shared_fanout = surface_detectors._boundary_local_wrapper_pairs(
        compact_fanout,
        config,
    )
    legacy_wrappers = surface_detectors._boundary_local_wrapper_collapse_candidates(
        modules,
        config,
    )

    assert compact_fanout == legacy_fanout
    assert {candidate.field_name for candidate in compact_fanout} == {
        "axis_id",
        "axis_scope",
    }
    assert compact_wrappers == legacy_wrappers
    assert wrappers_from_shared_fanout == legacy_wrappers
    assert compact_wrappers
    assert fanout_detector._findings_from_compact_projections(
        projections,
        config,
    ) == fanout_detector._findings_for_candidates(legacy_fanout, config)
    assert wrapper_detector._findings_from_compact_projections(
        projections,
        config,
    ) == wrapper_detector._findings_for_candidates(legacy_wrappers, config)

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (
            surface_detectors.DistributedBoundaryFanoutDetector,
            surface_detectors.BoundaryLocalWrapperCollapseDetector,
        ),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == len(modules)
    findings_by_detector = accumulator.findings_by_detector(config)
    assert findings_by_detector[
        surface_detectors.DistributedBoundaryFanoutDetector
    ] == fanout_detector._findings_for_candidates(legacy_fanout, config)
    assert findings_by_detector[
        surface_detectors.BoundaryLocalWrapperCollapseDetector
    ] == wrapper_detector._findings_for_candidates(legacy_wrappers, config)


def test_compact_role_surface_projection_matches_both_legacy_global_joins(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "provenance.py").write_text(
        "class SourceProvenance:\n"
        "    channel_source_component_metadata: tuple[object, ...]\n",
        encoding="utf-8",
    )
    (package_root / "closure.py").write_text(
        "from .provenance import SourceProvenance\n\n"
        "class SourceClosure(SourceProvenance):\n"
        "    def inherited_projection(self, plane_index):\n"
        "        plane_metadata = self.channel_source_component_metadata[plane_index]\n"
        "        return plane_metadata\n",
        encoding="utf-8",
    )
    (package_root / "consumers.py").write_text(
        "def stream_plane(provenance, plane_index):\n"
        "    plane_metadata = provenance.channel_source_component_metadata[plane_index]\n"
        "    return plane_metadata\n\n"
        "def display_plane(provenance, plane_index):\n"
        "    return Target(plane_metadata=provenance.channel_source_component_metadata[plane_index])\n",
        encoding="utf-8",
    )
    for module_name, class_name in (
        ("display", "FieldDisplayPolicy"),
        ("labels", "WidgetFieldLabelAuthority"),
        ("report", "ReportFieldLabelPresenter"),
    ):
        (package_root / f"{module_name}.py").write_text(
            f"class {class_name}:\n"
            "    LABELS = {\n"
            "        'alpha': 'A', 'beta': 'B', 'gamma': 'G',\n"
            "        'delta': 'D', 'epsilon': 'E',\n"
            "    }\n\n"
            "    def field_label(self, field, value):\n"
            "        return self.LABELS.get(field, value)\n",
            encoding="utf-8",
        )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    role_detector = role_surface_detectors.RoleSurfaceDriftDetector()
    case_detector = role_surface_detectors.GenericRoleCaseTableDetector()
    role_projections = role_detector.compact_module_projection_groups(modules)[
        role_surface_detectors.CompactRoleSurfaceModuleProjectionFamily
    ]
    class_projections = role_detector.compact_module_projection_groups(modules)[
        runtime_detectors.CompactModuleClassProjectionFamily
    ]
    compact_role_candidates = (
        role_surface_detectors._compact_role_surface_drift_candidates(
            role_projections,
            class_projections,
            config,
        )
    )
    legacy_role_candidates = role_surface_detectors._role_surface_drift_candidates(
        modules,
        config,
    )
    compact_case_candidates = (
        role_surface_detectors._compact_generic_role_case_table_candidates(
            role_projections,
            config,
        )
    )
    legacy_case_candidates = role_surface_detectors._generic_role_case_table_candidates(
        modules, config
    )

    assert compact_role_candidates == legacy_role_candidates
    assert len(compact_role_candidates) == 1
    assert len(compact_role_candidates[0].use_sites) == 2
    assert compact_case_candidates == legacy_case_candidates
    assert compact_case_candidates
    strict_role_config = DetectorConfig(min_role_drift_use_sites=3)
    assert role_surface_detectors._compact_role_surface_drift_candidates(
        role_projections,
        class_projections,
        strict_role_config,
    ) == role_surface_detectors._role_surface_drift_candidates(
        modules,
        strict_role_config,
    )
    strict_case_config = DetectorConfig(min_generic_role_case_table_cases=6)
    assert role_surface_detectors._compact_generic_role_case_table_candidates(
        role_projections,
        strict_case_config,
    ) == role_surface_detectors._generic_role_case_table_candidates(
        modules,
        strict_case_config,
    )
    assert role_detector._findings_from_compact_projection_groups(
        {
            role_surface_detectors.CompactRoleSurfaceModuleProjectionFamily: (
                role_projections
            ),
            runtime_detectors.CompactModuleClassProjectionFamily: class_projections,
        },
        config,
    ) == role_detector._findings_for_candidates(legacy_role_candidates, config)
    assert case_detector._findings_from_compact_projections(
        role_projections,
        config,
    ) == case_detector._findings_for_candidates(legacy_case_candidates, config)

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (
            role_surface_detectors.RoleSurfaceDriftDetector,
            role_surface_detectors.GenericRoleCaseTableDetector,
        ),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == len(modules) * 2
    findings_by_detector = accumulator.findings_by_detector(config)
    assert findings_by_detector[
        role_surface_detectors.RoleSurfaceDriftDetector
    ] == role_detector._findings_for_candidates(legacy_role_candidates, config)
    assert findings_by_detector[
        role_surface_detectors.GenericRoleCaseTableDetector
    ] == case_detector._findings_for_candidates(legacy_case_candidates, config)


def test_compact_nominal_bypass_and_variant_candidates_match_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "from abc import ABC\n\n"
        "class Payload(ABC):\n"
        "    pass\n\n"
        "class AlphaPayload(Payload):\n"
        "    def render_payload(self, request):\n"
        "        return PayloadResult(request.left, request.right)\n\n"
        "class BetaPayload(Payload):\n"
        "    def render_payload(self, request):\n"
        "        return PayloadResult(request.left, request.right)\n\n"
        "def render_payload(value, request):\n"
        "    if isinstance(value, AlphaPayload):\n"
        "        return value.render_payload(request)\n"
        "    if isinstance(value, BetaPayload):\n"
        "        return value.render_payload(request)\n"
        "    return None\n\n"
        "class PayloadBuilder:\n"
        "    def build_alpha_payload(self, request):\n"
        "        return PayloadResult(request.left, request.right)\n\n"
        "    def build_beta_payload(self, request):\n"
        "        return PayloadResult(request.left, request.right)\n\n"
        "def payload_forward(request):\n"
        "    return PayloadResult(request.left, request.right)\n\n"
        "def payload_outer(request):\n"
        "    return payload_inner(request)\n\n"
        "def payload_inner(request):\n"
        "    return request.payload()\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    bypass_detector = (
        runtime_detectors.ABCPolymorphismBypassedByConcreteDispatchDetector()
    )
    variant_detector = runtime_detectors.AlgebraicVariantMethodFamilyDetector()
    groups = type(bypass_detector).compact_module_projection_groups(modules)
    nominal_projections = groups[
        runtime_detectors.CompactNominalBypassModuleProjectionFamily
    ]
    class_projections = groups[runtime_detectors.CompactModuleClassProjectionFamily]

    legacy_bypass = runtime_detectors._nominal_authority_bypass_candidates(
        list(modules)
    )
    compact_bypass = (
        runtime_detectors._nominal_authority_bypass_candidates_from_compact_projections(
            nominal_projections,
            class_projections,
        )
    )
    legacy_variants = runtime_detectors._variant_method_family_candidates(list(modules))
    compact_variants = (
        runtime_detectors._variant_method_family_candidates_from_compact_projections(
            nominal_projections
        )
    )

    assert len(compact_bypass) == len(legacy_bypass) == 1
    assert len(compact_variants) == len(legacy_variants) == 1
    assert compact_bypass[0].composition_signals == (
        legacy_bypass[0].composition_signals
    )
    assert compact_variants[0].composition_signals == (
        legacy_variants[0].composition_signals
    )
    assert bypass_detector._findings_from_compact_projection_groups(
        groups,
        config,
    ) == [
        bypass_detector._finding_for_candidate(candidate) for candidate in legacy_bypass
    ]
    assert variant_detector._findings_from_compact_projections(
        nominal_projections,
        config,
    ) == [
        variant_detector._finding_for_candidate(candidate)
        for candidate in legacy_variants
    ]

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (type(bypass_detector), type(variant_detector)),
        use_parse_cache=False,
    )
    findings = accumulator.findings_by_detector(config)
    assert accumulator.projection_count == 2
    assert len(findings[type(bypass_detector)]) == 1
    assert len(findings[type(variant_detector)]) == 1


def test_compact_isinstance_scatter_preserves_nested_function_attribution(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "class AlphaPayload:\n"
        "    pass\n\n"
        "class BetaPayload:\n"
        "    pass\n\n"
        "def outer(value):\n"
        "    def inner():\n"
        "        if isinstance(value, AlphaPayload):\n"
        "            return 'alpha'\n"
        "        if isinstance(value, BetaPayload):\n"
        "            return 'beta'\n"
        "        return None\n"
        "    return inner()\n",
        encoding="utf-8",
    )
    module = parse_python_modules(package_root, use_parse_cache=False)[0]

    legacy = runtime_detectors._isinstance_family_scatter_candidates(module)
    compact = runtime_detectors._compact_isinstance_family_scatter_candidates(module)

    assert compact == legacy
    assert {candidate.qualname for candidate in compact} == {"inner", "outer"}


def test_compact_semantic_descent_graph_matches_legacy_ast_graph(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "from dataclasses import dataclass\n"
        "from enum import Enum\n\n"
        "class Step:\n"
        "    pass\n\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n\n"
        "class Field(Enum):\n"
        "    TITLE = 'title'\n"
        "    STATUS = 'status'\n\n"
        "FIELDS = ('title', 'status')\n\n"
        "@dataclass\n"
        "class Request:\n"
        "    title: str\n"
        "    status: str\n\n"
        "REQUEST_FIELDS = ('title', 'status')\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = semantic_descent_detectors.SemanticMirrorWithoutDescentDetector()
    groups = type(detector).compact_module_projection_groups(modules)
    semantic_projections = groups[CompactSemanticModuleProjectionFamily]
    class_projections = groups[runtime_detectors.CompactModuleClassProjectionFamily]

    legacy_graph = build_semantic_descent_graph(list(modules), use_cache=False)
    compact_graph = build_compact_semantic_descent_graph(
        semantic_projections,
        class_projections,
    )

    assert compact_graph.authorities == legacy_graph.authorities
    assert compact_graph.facts == legacy_graph.facts
    assert compact_graph.projections == legacy_graph.projections
    assert compact_graph.mirror_edges == legacy_graph.mirror_edges
    assert compact_graph.certificates == legacy_graph.certificates
    config = DetectorConfig()
    assert detector._findings_from_compact_projection_groups(groups, config) == (
        detector._collect_findings_from_graph(legacy_graph, list(modules), config)
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (type(detector),),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == 2
    assert accumulator.findings_by_detector(config)[type(detector)] == (
        detector._collect_findings_from_graph(legacy_graph, list(modules), config)
    )


def test_global_projection_partition_tracks_migrated_detector_boundary() -> None:
    partition = DetectorTypePartition.from_detector_types(
        default_detector_types_for_analysis()
    )

    assert (
        runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector
        in partition.compact_global_detector_types
    )
    assert (
        systemic_detectors.InheritanceHierarchyCandidateDetector
        in partition.compact_global_detector_types
    )
    assert runtime_detectors.RepeatedBuilderCallDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.DeclaredFieldExtractionFanoutDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.RepeatedExportDictDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ManualClassRegistrationDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RepeatedPrivateMethodDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.InheritedAutoRegisterConfigBoilerplateDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.AutoRegisterExplicitPriorityOrderingDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.DeadEmbeddedStaticPayloadDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.UnreferencedPrivateFunctionDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.DanglingPrivateMethodDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.SupportPreludeModuleFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert environment_detectors.EnvironmentBooleanAuthorityDriftDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.PublicBareSupportFunctionDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.ParallelKeyedAxisFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.ParallelKeyedTableAndFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.ParallelKeyedTableAxisDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.ResidualClosedAxisBranchingDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.CrossModuleAxisShadowFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.PrivateHelperShadowDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.DataclassNamespaceCliMirrorDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ExactTypeGuardInheritanceRetreatDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.SemanticInheritanceFamilySSOTDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.AutoRegisterMetaUnderRentedDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.NonInjectiveTypeRegistryDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.InjectiveTypeRegistryDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.PrematureRegistryInfrastructureDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RepeatedKeyedFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.PredicateSelectedConcreteFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ParallelMirroredLeafFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ManualConcreteSubclassRosterDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.LatentImplementationRosterDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RegistryProjectionSurfaceDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RegistryProjectionPolicyAuthorityDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.ManualFamilyRosterDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.ExistingNominalAuthorityReuseDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.NominalAuthorityImplementationRetreatDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.DuplicateNominalAuthoritySurfaceDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.PassThroughNominalWrapperDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.ClassLevelInheritanceOptimizationDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.SemanticOverlapAbcOptimizationDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.SemanticOverlapAbcFamilyOptimizationDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.GlobalInheritanceOptimizationDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.SemanticOverlapAbcResidueAxisCatalogDetector in (
        partition.compact_global_detector_types
    )
    assert abstraction_reuse_detectors.AvailableCarrierReuseDetector in (
        partition.compact_global_detector_types
    )
    assert abstraction_reuse_detectors.CarrierCompositionRetreatDetector in (
        partition.compact_global_detector_types
    )
    assert abstraction_reuse_detectors.ParallelPrimitiveCarrierDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.PrivateHelperSemanticClusterDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.RoleGuardedSurfaceAccessDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.NonNominalPrivateHelperDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.DistributedBoundaryFanoutDetector in (
        partition.compact_global_detector_types
    )
    assert surface_detectors.BoundaryLocalWrapperCollapseDetector in (
        partition.compact_global_detector_types
    )
    assert role_surface_detectors.RoleSurfaceDriftDetector in (
        partition.compact_global_detector_types
    )
    assert role_surface_detectors.GenericRoleCaseTableDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ABCPolymorphismBypassedByConcreteDispatchDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.AlgebraicVariantMethodFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert semantic_descent_detectors.SemanticMirrorWithoutDescentDetector in (
        partition.compact_global_detector_types
    )
    assert abstraction_reuse_detectors.AvailableAbstractionReuseDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.PublicApiPrivateDelegateShellDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.PublicApiPrivateDelegateFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.CrossModuleSpecAxisAuthorityDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RepeatedValidateShapeGuardFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert len(partition.compact_global_detector_types) == 66
    assert len(partition.ast_retaining_context_detector_types) == 3
    assert len(partition.per_module_detector_types) == 183


def test_parse_cache_persists_semantic_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "ast"
    original_hash = ast_tools_module.semantic_python_source_hash
    hash_calls = 0

    def counted_hash(source: str) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(source)

    monkeypatch.setattr(
        ast_tools_module,
        "semantic_python_source_hash",
        counted_hash,
    )

    first = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    second = parse_python_modules(package_root, cache_dir=cache_dir)[0]

    assert first.semantic_hash == second.semantic_hash
    assert hash_calls == 1


def test_analysis_identity_reuses_cached_source_hashes_for_unchanged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    first_path = package_root / "a.py"
    second_path = package_root / "b.py"
    first_path.write_text("VALUE = 1\n", encoding="utf-8")
    second_path.write_text("VALUE = 2\n", encoding="utf-8")
    cache = AnalysisFindingCache(tmp_path / ".nra-cache" / "analysis")
    source_signature_cache = cache.source_signature_cache()
    assert source_signature_cache is not None
    original_read_bytes = Path.read_bytes

    first_identity = AnalysisCacheIdentity.from_roots(
        (package_root,),
        DetectorConfig(),
        source_signature_cache=source_signature_cache,
    )

    def fail_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"unexpected source reread for {path}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    cached_source_signature_cache = cache.source_signature_cache()
    assert cached_source_signature_cache is not None
    second_identity = AnalysisCacheIdentity.from_roots(
        (package_root,),
        DetectorConfig(),
        source_signature_cache=cached_source_signature_cache,
    )

    assert second_identity == first_identity

    read_paths: list[Path] = []

    def count_read_bytes(path: Path) -> bytes:
        read_paths.append(path.resolve())
        return original_read_bytes(path)

    second_path.write_text("VALUE = 200\n", encoding="utf-8")
    monkeypatch.setattr(Path, "read_bytes", count_read_bytes)
    invalidated_source_signature_cache = cache.source_signature_cache()
    assert invalidated_source_signature_cache is not None
    changed_identity = AnalysisCacheIdentity.from_roots(
        (package_root,),
        DetectorConfig(),
        source_signature_cache=invalidated_source_signature_cache,
    )

    assert changed_identity != first_identity
    assert read_paths == [second_path.resolve()]


def test_changed_path_root_assignment_returns_absolute_owner_for_relative_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    changed_file = package_root / "mod.py"
    changed_file.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    paths_by_root = ChangedPathRootAssignment(
        roots=(Path("pkg"),),
        changed_paths=frozenset((str(changed_file.resolve()),)),
    ).paths_by_root()

    assert tuple(paths_by_root) == (package_root.resolve(),)
    assert paths_by_root[package_root.resolve()] == (changed_file.resolve(),)


def test_incremental_cache_reruns_global_detectors_for_repo_context(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "authority.py").write_text(
        "class Step:\n" "    pass\n",
        encoding="utf-8",
    )
    (package_root / "members.py").write_text(
        "class LoadStep(Step):\n" "    step_id = 'load'\n",
        encoding="utf-8",
    )
    (package_root / "registry.py").write_text(
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in initial_findings
    )

    (package_root / "members.py").write_text(
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n",
        encoding="utf-8",
    )

    updated_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )

    assert any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in updated_findings
    )


def test_partial_cache_omits_changed_compact_global_semantic_findings(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    authority_path = package_root / "authority.py"
    members_path = package_root / "members.py"
    registry_path = package_root / "registry.py"
    authority_path.write_text("class Step:\n    pass\n", encoding="utf-8")
    members_path.write_text(
        "class LoadStep(Step):\n" "    step_id = 'load'\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in initial_findings
    )

    members_path.write_text(
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n",
        encoding="utf-8",
    )

    partial_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(package_root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=0,
            analysis_workers=0,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()

    assert partial_result is not None
    assert partial_result.cache_status is AnalysisCacheStatus.PARTIAL
    mirror_findings = tuple(
        finding
        for finding in partial_result.findings
        if finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
    )
    assert not mirror_findings

    exact_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        and any(
            evidence.file_path == str(members_path) for evidence in finding.evidence
        )
        for finding in exact_findings
    )


def test_compact_semantic_detector_does_not_materialize_legacy_graph_cache(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    members_path = package_root / "members.py"
    (package_root / "authority.py").write_text(
        "class Step:\n    pass\n", encoding="utf-8"
    )
    members_path.write_text(
        "class LoadStep(Step):\n" "    step_id = 'load'\n",
        encoding="utf-8",
    )
    (package_root / "registry.py").write_text(
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in initial_findings
    )

    graph_cache_context = SemanticDescentGraphCacheContext.from_parse_cache(
        (package_root,),
        cache_dir,
        True,
        None,
    )
    assert graph_cache_context.latest_graph() is None

    members_path.write_text(
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n",
        encoding="utf-8",
    )

    partial_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(package_root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=0,
            analysis_workers=0,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()

    assert partial_result is not None
    assert partial_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in partial_result.findings
    )


def test_partial_cache_omits_changed_compact_semantic_projection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    authority_path = package_root / "authority.py"
    members_path = package_root / "members.py"
    registry_path = package_root / "registry.py"
    authority_path.write_text("class Step:\n    pass\n", encoding="utf-8")
    members_path.write_text(
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n",
        encoding="utf-8",
    )
    registry_path.write_text("NO_REGISTRY = None\n", encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert not any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in initial_findings
    )

    registry_path.write_text(
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )

    partial_result = FastCachedPathAnalysisAuthority(
        CachedPathAnalysisRequest(
            roots=(package_root,),
            config=DetectorConfig(),
            parse_cache_dir=cache_dir,
            use_parse_cache=True,
            parse_workers=0,
            analysis_workers=0,
            source_policy=None,
            reuse_policy=FastCacheReusePolicy.EVIDENCE_LOCAL_PARTIAL,
        )
    ).result()

    assert partial_result is not None
    assert partial_result.cache_status is AnalysisCacheStatus.PARTIAL
    mirror_findings = tuple(
        finding
        for finding in partial_result.findings
        if finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
    )
    assert not mirror_findings

    exact_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )
    assert any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        and any(
            evidence.file_path == str(registry_path) for evidence in finding.evidence
        )
        for finding in exact_findings
    )


def test_contextual_module_cache_invalidates_when_repo_context_changes(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "roles.py").write_text(
        "class AvoidWidgetsWindow:\n" "    pass\n",
        encoding="utf-8",
    )
    (package_root / "consumer.py").write_text(
        "from pkg.roles import AvoidWidgetsWindow\n"
        "\n"
        "\n"
        "def place_window(window):\n"
        "    if isinstance(window, AvoidWidgetsWindow):\n"
        "        return tuple(window.position_avoid_widgets())\n"
        "    return ()\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )

    assert not any(
        finding.detector_id == "role_guarded_surface_access"
        for finding in initial_findings
    )

    (package_root / "roles.py").write_text(
        "class AvoidWidgetsWindow:\n"
        "    def position_avoid_widgets(self):\n"
        "        raise NotImplementedError\n",
        encoding="utf-8",
    )

    updated_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )

    assert any(
        finding.detector_id == "role_guarded_surface_access"
        and "position_avoid_widgets" in finding.summary
        for finding in updated_findings
    )


def test_private_reference_contextual_cache_invalidates_when_reference_edge_changes(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    helpers_path = package_root / "helpers.py"
    helpers_path.write_text(
        "def _build_plan(value):\n"
        "    first = value + 1\n"
        "    second = first * 2\n"
        "    third = second - value\n"
        "    fourth = third + first\n"
        "    fifth = fourth + second\n"
        "    sixth = fifth - third\n"
        "    seventh = sixth + fourth\n"
        "    return seventh\n",
        encoding="utf-8",
    )
    consumer_path = package_root / "consumer.py"
    consumer_path.write_text(
        "def use(value):\n" "    return value\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    initial_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )

    assert any(
        finding.detector_id == "unreferenced_private_function"
        and finding.evidence[0].file_path == str(helpers_path)
        and "`_build_plan`" in finding.summary
        for finding in initial_findings
    )

    consumer_path.write_text(
        "from pkg.helpers import _build_plan\n"
        "\n"
        "\n"
        "def use(value):\n"
        "    return _build_plan(value)\n",
        encoding="utf-8",
    )

    updated_findings = analyze_path(
        package_root,
        cache_dir=cache_dir,
        parse_workers=0,
        analysis_workers=0,
    )

    assert not any(
        finding.detector_id == "unreferenced_private_function"
        and finding.evidence[0].file_path == str(helpers_path)
        and "`_build_plan`" in finding.summary
        for finding in updated_findings
    )
