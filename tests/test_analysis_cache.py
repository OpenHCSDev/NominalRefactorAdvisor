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
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphCacheContext,
    SemanticDescentGraphAnalysisSource,
    analyze_modules,
    analyze_modules_with_cache,
    analyze_module_detector_types_with_cache,
    analyze_path,
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
from nominal_refactor_advisor.models import FindingSpec, RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import (
    SemanticAuthority,
    SemanticAuthorityKind,
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheIdentity,
    build_semantic_descent_graph,
)


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

    cleared_cache_count = release_module_analysis_memory()

    assert cleared_cache_count > 0
    assert ast_tools_module._walk_nodes.cache_info().currsize == 0


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


def test_partial_cache_overlays_changed_modules_for_semantic_graph_findings(
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

    graph_cache_context = SemanticDescentGraphCacheContext.from_parse_cache(
        (package_root,),
        cache_dir,
        True,
        None,
    )
    cached_graph = graph_cache_context.latest_graph()
    assert cached_graph is not None
    assert cached_graph.class_index is not None

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
            semantic_descent_source=SemanticDescentGraphAnalysisSource(
                cached_graph=cached_graph,
                cache_context=graph_cache_context,
            ),
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
    assert mirror_findings
    assert any(
        evidence.file_path == str(members_path)
        for finding in mirror_findings
        for evidence in finding.evidence
    )


def test_partial_cache_uses_latest_repo_graph_for_changed_module_scan(
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
    assert graph_cache_context.latest_graph() is not None

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
            semantic_descent_source=SemanticDescentGraphAnalysisSource(
                cache_context=graph_cache_context,
            ),
        )
    ).result()

    assert partial_result is not None
    assert partial_result.cache_status is AnalysisCacheStatus.PARTIAL
    assert any(
        finding.detector_id == "semantic_mirror_without_descent"
        and "`STEP_TABLE` mirrors `Step`" in finding.title
        for finding in partial_result.findings
    )


def test_partial_cache_overlays_changed_projection_for_cached_authority_graph(
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

    graph_cache_context = SemanticDescentGraphCacheContext.from_parse_cache(
        (package_root,),
        cache_dir,
        True,
        None,
    )
    cached_graph = graph_cache_context.latest_graph()
    assert cached_graph is not None
    assert cached_graph.class_index is not None

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
            semantic_descent_source=SemanticDescentGraphAnalysisSource(
                cached_graph=cached_graph,
                cache_context=graph_cache_context,
            ),
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
    assert mirror_findings
    assert any(
        evidence.file_path == str(registry_path)
        for finding in mirror_findings
        for evidence in finding.evidence
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
