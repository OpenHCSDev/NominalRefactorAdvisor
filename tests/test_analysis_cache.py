from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import pickle
import sys
from time import sleep
import weakref

import pytest

from nominal_refactor_advisor.analysis import (
    AnalysisPathScope,
    CachedPathAnalysisRequest,
    ChangedPathRootAssignment,
    DetectorTypePartition,
    FastCacheReusePolicy,
    FastCachedPathAnalysisAuthority,
    SemanticDescentGraphCacheContext,
    SemanticDescentGraphAnalysisSource,
    analyze_compact_roots_with_cache,
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
    AnalysisCacheStorage,
    AnalysisFindingCacheChunkStreamHeader,
    AnalysisFindingCacheEntryPayload,
    AnalysisFindingCache,
    DetectorRegistrySignature,
    GlobalDetectorAnalysisCacheIdentity,
    GlobalDetectorFamilyAnalysisCacheIdentity,
    SourceFileSignatureCache,
)
from nominal_refactor_advisor.ast_tools import (
    BuilderCallShapeFamily,
    RegistrationShapeFamily,
    SourceModule,
    collect_family_items,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex
from nominal_refactor_advisor import ast_tools as ast_tools_module
from nominal_refactor_advisor import analysis as analysis_module
from nominal_refactor_advisor import analysis_cache as analysis_cache_module
from nominal_refactor_advisor import class_index as class_index_module
from nominal_refactor_advisor import native_syntax as native_syntax_module
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
from nominal_refactor_advisor.detectors import _reflection as reflection_detectors
from nominal_refactor_advisor.detectors import _base as base_detectors
from nominal_refactor_advisor.detectors import _helpers as helper_detectors
from nominal_refactor_advisor.detectors import _runtime as runtime_detectors
from nominal_refactor_advisor.detectors import (
    _semantic_descent as semantic_descent_detectors,
)
from nominal_refactor_advisor.detectors import _surface as surface_detectors
from nominal_refactor_advisor.detectors import _structural as structural_detectors
from nominal_refactor_advisor.detectors import _systemic as systemic_detectors
from nominal_refactor_advisor.models import FindingSpec, RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import (
    CompactSemanticModuleProjectionFamily,
    CompactSemanticProjectionDemand,
    SemanticAuthority,
    SemanticAuthorityKind,
    SemanticDescentGraph,
    SemanticDescentGraphCache,
    SemanticDescentGraphCacheIdentity,
    build_semantic_descent_graph,
    build_compact_semantic_descent_graph,
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
        relations=(),
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
    module = ast_tools_module.ast.parse("def project():\n    return source + 1\n")
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    ast_tools_module._walk_nodes(module)
    ast_tools_module.walk_function_body_nodes(function)
    ast_tools_module.named_function_nodes(module)
    ast_tools_module.module_syntax_index(module)
    runtime_detectors.SurfaceFunctionIndex.from_module(module)

    cleared_cache_count = release_module_analysis_memory()

    assert cleared_cache_count > 0
    assert ast_tools_module._walk_nodes.cache_info().currsize == 0
    assert ast_tools_module.walk_function_body_nodes.cache_info().currsize == 0
    assert ast_tools_module.named_function_nodes.cache_info().currsize == 0
    assert ast_tools_module.module_syntax_index.cache_info().currsize == 0
    assert runtime_detectors.SurfaceFunctionIndex.from_module.cache_info().currsize == 0


def test_module_analysis_memory_release_preserves_compiled_native_queries() -> None:
    query_source = "(class_definition) @class"
    syntax_index = NativePythonSyntaxIndex.from_source("class Role: pass\n")
    expected_query = native_syntax_module._python_query(query_source)
    assert syntax_index.captures(query_source)["class"]

    release_module_analysis_memory()

    assert native_syntax_module._python_query(query_source) is expected_query


def test_native_syntax_index_shares_frozen_captures_between_families() -> None:
    syntax_index = NativePythonSyntaxIndex.from_source(
        "class Role:\n    def build(self): return Role()\n"
    )
    query_source = "(class_definition) @class\n(call) @call"

    first = syntax_index.captures(query_source)
    second = syntax_index.captures(query_source)

    assert second is first
    assert len(first["class"]) == 1
    assert len(first["call"]) == 1


def test_function_body_consumers_share_bounded_projection_authority(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "environment.py").write_text(
        "def trace_environment_enabled(value):\n"
        '    """Interpret one environment flag."""\n'
        "    normalized = value.strip().lower()\n"
        "    return normalized in {'1', 'true'}\n",
        encoding="utf-8",
    )
    module = parse_python_modules(package_root, use_parse_cache=False)[0]
    scope = environment_detectors._function_scopes(module)[0]

    environment_nodes = scope.nodes()
    authority_nodes = ast_tools_module.walk_function_body_nodes(scope.node)

    assert environment_nodes is scope.nodes()
    assert environment_nodes is authority_nodes
    assert isinstance(environment_nodes[0], ast.Assign)
    assert not any(
        isinstance(node, ast.Constant)
        and node.value == "Interpret one environment flag."
        for node in environment_nodes
    )


def test_class_and_detector_collectors_share_named_function_projection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "projection.py").write_text(
        "class Outer:\n"
        "    class Inner:\n"
        "        def build(self):\n"
        "            return Result(value=1)\n",
        encoding="utf-8",
    )
    module = parse_python_modules(package_root, use_parse_cache=False)[0]

    class_functions = class_index_module._named_functions(module.module)
    detector_functions = base_detectors._iter_named_functions(module)
    base_detectors._module_builder_call_shapes(module)

    assert class_functions is detector_functions
    assert tuple(name for name, _ in class_functions) == ("Outer.Inner.build",)
    function = class_functions[0][1]
    assert ast_tools_module.walk_function_body_nodes(function) is (
        ast_tools_module.walk_function_body_nodes(function)
    )


def test_module_syntax_index_projects_nodes_from_its_owned_traversal() -> None:
    module = ast.parse(
        "class Outer:\n"
        "    class Inner:\n"
        "        def build(self):\n"
        "            return Result()\n"
    )
    syntax_index = ast_tools_module.module_syntax_index(module)

    indexed_classes = syntax_index.indexed_nodes_of_type(ast.ClassDef)
    indexed_calls = syntax_index.indexed_nodes_of_type(ast.Call)

    assert tuple(node.name for _index, node in indexed_classes) == ("Outer", "Inner")
    assert tuple(ast.unparse(node) for _index, node in indexed_calls) == ("Result()",)
    assert all(
        syntax_index.depth_first_nodes[index] is node
        for index, node in (*indexed_classes, *indexed_calls)
    )


def test_context_semantic_supplements_use_indexed_call_projection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "module.py").write_text(
        "class Presenter:\n" "    def build(self):\n" "        return Result()\n",
        encoding="utf-8",
    )
    module = parse_python_modules(package_root, use_parse_cache=False)[0]

    projection = CompactSemanticModuleProjectionFamily.collect_demanded(
        module,
        CompactSemanticProjectionDemand(include_presentations=False),
    )

    assert projection is not None
    assert len(projection) == 1
    assert tuple(
        (supplement.class_symbol, supplement.constructed_type_names)
        for supplement in projection[0].class_supplements
    ) == (("module.Presenter", ("Result",)),)


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


def test_module_detector_cache_reuses_unchanged_implementation_bundle(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    module = parse_python_modules(package_root, use_parse_cache=False)[0]
    calls = {"stable": 0, "first": 0, "second": 0}
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Bundle cache",
        why="Bundle cache",
        capability_gap="Bundle cache",
        relation_context="Bundle cache",
    )

    def finding(detector_id: str, module) -> RefactorFinding:
        return finding_spec.build(
            detector_id,
            detector_id,
            (SourceLocation(str(module.path), 1, detector_id),),
        )

    class StableBundleDetector(PerModuleIssueDetector):
        detector_id = "stable_bundle_detector"

        def _findings_for_module(self, module, config):
            del config
            calls["stable"] += 1
            return [finding(self.detector_id, module)]

    class FirstChangingBundleDetector(PerModuleIssueDetector):
        detector_id = "first_changing_bundle_detector"

        def _findings_for_module(self, module, config):
            del config
            calls["first"] += 1
            return [finding(self.detector_id, module)]

    class SecondChangingBundleDetector(PerModuleIssueDetector):
        detector_id = "second_changing_bundle_detector"

        def _findings_for_module(self, module, config):
            del config
            calls["second"] += 1
            return [finding(self.detector_id, module)]

    StableBundleDetector.__module__ = "test_stable_detector_bundle"
    FirstChangingBundleDetector.__module__ = "test_changing_detector_bundle_v1"
    SecondChangingBundleDetector.__module__ = "test_changing_detector_bundle_v2"
    registered_test_detectors = (
        StableBundleDetector,
        FirstChangingBundleDetector,
        SecondChangingBundleDetector,
    )
    arguments = {
        "module": module,
        "config": DetectorConfig(),
        "presentation_roots": (package_root,),
        "analysis_cache_dir": tmp_path / "analysis-cache",
    }

    try:
        cold = analyze_module_detector_types_with_cache(
            detector_types=(StableBundleDetector, FirstChangingBundleDetector),
            **arguments,
        )
        partial = analyze_module_detector_types_with_cache(
            detector_types=(StableBundleDetector, SecondChangingBundleDetector),
            **arguments,
        )
    finally:
        for registry_key, detector_type in tuple(IssueDetector.__registry__.items()):
            if detector_type in registered_test_detectors:
                del IssueDetector.__registry__[registry_key]

    assert cold.cache_status is AnalysisCacheStatus.MISS
    assert partial.cache_status is AnalysisCacheStatus.PARTIAL
    assert calls == {"stable": 1, "first": 1, "second": 1}
    assert {item.detector_id for item in cold.findings} == {
        "stable_bundle_detector",
        "first_changing_bundle_detector",
    }
    assert {item.detector_id for item in partial.findings} == {
        "stable_bundle_detector",
        "second_changing_bundle_detector",
    }


def test_detector_shard_cache_identity_ignores_orchestration_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    module = parse_python_modules(tmp_path, use_parse_cache=False)[0]
    identity_type = analysis_cache_module.PerModuleAnalysisCacheFamilyIdentity
    baseline = identity_type.from_module(module, DetectorConfig(), (tmp_path,))
    aggregate_baseline = AnalysisCacheIdentity.from_modules(
        (tmp_path,),
        (module,),
        DetectorConfig(),
    )
    detector_type = default_detector_types_for_analysis()[0]
    global_baseline = GlobalDetectorAnalysisCacheIdentity.from_global_context(
        DetectorConfig(),
        detector_type,
        aggregate_baseline.source_context_token,
        (tmp_path,),
    )
    original_signature = analysis_cache_module._module_source_signature

    def changed_signature(module_name: str):
        signature = original_signature(module_name)
        if module_name == "nominal_refactor_advisor.analysis":
            return analysis_cache_module.SourceFileSignature(
                signature.path,
                "changed-orchestration",
            )
        return signature

    monkeypatch.setattr(
        analysis_cache_module,
        "_module_source_signature",
        changed_signature,
    )
    after_orchestration_change = identity_type.from_module(
        module,
        DetectorConfig(),
        (tmp_path,),
    )
    aggregate_after_orchestration_change = AnalysisCacheIdentity.from_modules(
        (tmp_path,),
        (module,),
        DetectorConfig(),
    )
    global_after_orchestration_change = (
        GlobalDetectorAnalysisCacheIdentity.from_global_context(
            DetectorConfig(),
            detector_type,
            aggregate_after_orchestration_change.source_context_token,
            (tmp_path,),
        )
    )

    def changed_semantic_signature(module_name: str):
        signature = changed_signature(module_name)
        if module_name == "nominal_refactor_advisor.detectors._base":
            return analysis_cache_module.SourceFileSignature(
                signature.path,
                "changed-detector-semantics",
            )
        return signature

    monkeypatch.setattr(
        analysis_cache_module,
        "_module_source_signature",
        changed_semantic_signature,
    )
    after_semantic_change = identity_type.from_module(
        module,
        DetectorConfig(),
        (tmp_path,),
    )
    global_after_semantic_change = (
        GlobalDetectorAnalysisCacheIdentity.from_global_context(
            DetectorConfig(),
            detector_type,
            aggregate_after_orchestration_change.source_context_token,
            (tmp_path,),
        )
    )

    assert after_orchestration_change == baseline
    assert after_semantic_change != baseline
    assert aggregate_after_orchestration_change != aggregate_baseline
    assert (
        aggregate_after_orchestration_change.source_context_token
        == aggregate_baseline.source_context_token
    )
    assert global_after_orchestration_change == global_baseline
    assert global_after_semantic_change != global_baseline


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


def test_semantic_graph_cache_publishes_when_directory_sync_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SemanticDescentGraphCache(tmp_path)
    identity = SemanticDescentGraphCacheIdentity.from_roots((tmp_path,))
    graph = _empty_semantic_descent_graph("Published")
    original_open = semantic_descent_module.os.open

    def open_without_directory_support(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise PermissionError("directory handles are unsupported")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        semantic_descent_module.os, "open", open_without_directory_support
    )

    cache.store(identity, graph)
    family_identity = (
        semantic_descent_module.SemanticDescentGraphCacheFamilyIdentity.from_identity(
            identity
        )
    )

    assert cache.load(identity).graph == graph
    assert cache.load_latest(family_identity).graph == graph


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
        relations=(),
    )
    graph_cache.store(graph_identity_a, graph)

    assert graph_identity_a.cache_token == graph_identity_b.cache_token
    assert analysis_identity_a.cache_token == analysis_identity_b.cache_token
    assert str(checkout_a) not in repr(graph_identity_a)
    assert str(checkout_a) not in repr(analysis_identity_a)
    relocated_graph = graph_cache.load(graph_identity_b).graph
    assert relocated_graph is not None
    assert relocated_graph.authorities[0].location.file_path == source_b.as_posix()

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
    assert second.findings[0].evidence[0].file_path == source_b.as_posix()

    source_b.write_text("VALUE = 'foreign content'\n", encoding="utf-8")
    foreign_identity = SemanticDescentGraphCacheIdentity.from_roots((checkout_b,))
    assert foreign_identity.cache_token != graph_identity_a.cache_token
    assert graph_cache.load(foreign_identity).graph is None


def test_same_checkout_finding_rebase_reuses_validated_objects(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    finding = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Stable",
        why="stable",
        capability_gap="stable",
        relation_context="stable",
    ).build(
        "stable_cache_detector",
        "stable finding",
        (SourceLocation(str(source_path), 1, "VALUE"),),
    )

    rebased = analysis_cache_module._rebase_findings(
        (finding,),
        (str(tmp_path),),
        (str(tmp_path),),
    )

    assert rebased[0] is finding
    assert rebased[0].evidence[0] is finding.evidence[0]


def test_exact_finding_cache_chunks_pickles_and_loads_legacy_payloads(
    tmp_path: Path,
) -> None:
    assert AnalysisCacheStorage(tmp_path / "default-cache").finding_chunk_size == 64
    source_path = tmp_path / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    identity = AnalysisCacheIdentity.from_roots((tmp_path,), DetectorConfig())
    findings = [
        FindingSpec(
            pattern_id=PatternId.NOMINAL_BOUNDARY,
            title="Chunked",
            why="chunked",
            capability_gap="chunked",
            relation_context="chunked",
        ).build(
            "chunked_cache_detector",
            f"chunked finding {index}",
            (SourceLocation(str(source_path), index + 1, f"VALUE:{index}"),),
        )
        for index in range(5)
    ]
    payload = AnalysisFindingCacheEntryPayload.from_findings(identity, findings)
    storage = AnalysisCacheStorage(tmp_path / "cache", finding_chunk_size=2)
    cache_path = storage.entry_path(identity)

    storage.store_finding_payload_atomic(cache_path, payload)

    with cache_path.open("rb") as handle:
        header = pickle.load(handle)
        chunks = (pickle.load(handle), pickle.load(handle), pickle.load(handle))
        with pytest.raises(EOFError):
            pickle.load(handle)
    assert isinstance(header, AnalysisFindingCacheChunkStreamHeader)
    assert header.finding_count == 5
    assert tuple(len(chunk) for chunk in chunks) == (2, 2, 1)
    assert storage.load_finding_payload(cache_path, identity) == payload

    streamed_path = storage.cache_file_path("streamed.pickle")
    storage.store_finding_chunks_atomic(
        streamed_path,
        identity,
        len(findings),
        (tuple(findings[:2]), tuple(findings[2:4]), tuple(findings[4:])),
    )
    assert storage.load_finding_payload(streamed_path, identity) == payload

    incomplete_path = storage.cache_file_path("incomplete-stream.pickle")
    with pytest.raises(ValueError, match="did not reach"):
        storage.store_finding_chunks_atomic(
            incomplete_path,
            identity,
            len(findings) + 1,
            (tuple(findings[:2]), tuple(findings[2:4]), tuple(findings[4:])),
        )
    assert not incomplete_path.exists()

    legacy_path = storage.cache_file_path("legacy.pickle")
    with legacy_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    assert storage.load_finding_payload(legacy_path, identity) == payload

    truncated_path = storage.cache_file_path("truncated.pickle")
    with truncated_path.open("wb") as handle:
        pickle.dump(header, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(chunks[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    assert storage.load_finding_payload(truncated_path, identity) is None


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
        relations=(),
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
        relations=(),
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
        "class Payload: pass\n"
        "def build(item):\n"
        "    return Payload(name=item.name, score=item.score, label=item.label)\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"

    first_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    first_items = collect_family_items(first_module, BuilderCallShapeFamily)
    family_cache_dir = cache_dir / "collected-family"

    assert first_items
    assert tuple(family_cache_dir.glob("*.pickle"))

    second_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    second_items = collect_family_items(second_module, BuilderCallShapeFamily)

    assert [item.field_names for item in second_items] == [
        item.field_names for item in first_items
    ]


def test_legacy_family_cache_payload_is_ast_checked_and_certified(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "class Payload: pass\n"
        "def build(item):\n"
        "    return Payload(name=item.name, score=item.score, label=item.label)\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"
    module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    expected_items = collect_family_items(module, BuilderCallShapeFamily)
    payload_path = next((cache_dir / "collected-family").glob("*.pickle"))
    payload = pickle.loads(payload_path.read_bytes())
    payload_path.write_bytes(
        pickle.dumps(
            ast_tools_module.CollectedFamilyCachePayload(
                identity=payload.identity,
                items=payload.items,
                ast_free=False,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    release_module_analysis_memory()

    reloaded_module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    actual_items = collect_family_items(reloaded_module, BuilderCallShapeFamily)
    certified_payload = pickle.loads(payload_path.read_bytes())

    assert actual_items == expected_items
    assert certified_payload.ast_free is True


def test_collected_family_can_opt_into_a_larger_bounded_cache_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "class Payload: pass\n"
        "def build(item):\n"
        "    return Payload(name=item.name, score=item.score, label=item.label)\n",
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
        BuilderCallShapeFamily,
        "cache_payload_max_bytes",
        10_000,
    )

    module = parse_python_modules(package_root, cache_dir=cache_dir)[0]
    assert collect_family_items(module, BuilderCallShapeFamily)
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
    assert (
        second_sites[0].file_path == (package_root / "generated_catalog.py").as_posix()
    )


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


def test_warm_bounded_projection_load_skips_revalidating_ast_free_cache(
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
    first = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=tmp_path / ".nra-cache" / "analysis-first",
        detector_types=detector_types,
    )

    def unexpected_revalidation(cls, value, seen_ids=None):
        del cls, value, seen_ids
        raise AssertionError("store-validated compact cache was recursively rescanned")

    monkeypatch.setattr(
        analysis_module.CompactGlobalProjectionAccumulator,
        "_retains_ast",
        classmethod(unexpected_revalidation),
    )
    second = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=tmp_path / ".nra-cache" / "analysis-second",
        detector_types=detector_types,
    )

    assert second.findings == first.findings
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


def test_parallel_compact_root_analysis_returns_uncached_projection_fallbacks(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated_policy.py").write_text(
        "# generated from policy schema\nPOLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text(
        "POLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    detector_type = runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector
    modules = parse_python_modules(package_root, use_parse_cache=False)
    expected = detector_type().detect(modules, DetectorConfig())

    result = analyze_compact_roots_with_cache(
        (package_root,),
        use_parse_cache=False,
        parse_workers=2,
        analysis_cache_dir=tmp_path / "analysis",
        detector_types=(detector_type,),
    )

    assert [finding.to_dict() for finding in result.findings] == [
        finding.to_dict() for finding in expected
    ]
    assert result.projection_count == 2


def test_native_registration_projection_matches_registered_ast_specs(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "registrations.py"
    module_path.write_text(
        "class Registry:\n"
        "    def register(self, cls, key): return cls\n"
        "    def auto_register(self, registry, key): return lambda cls: cls\n"
        "\n"
        "registry = Registry()\n"
        "REGISTRY = {}\n"
        "\n"
        "@registry.auto_register(REGISTRY, 'alpha')\n"
        "class Alpha:\n"
        "    pass\n"
        "\n"
        "class Beta:\n"
        "    pass\n"
        "\n"
        "REGISTRY['alpha'] = Alpha\n"
        "registry.register(Beta, 'beta')\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )

    native = RegistrationShapeFamily.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, RegistrationShapeFamily)


def test_native_builder_projection_matches_canonical_ast_family(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "builders.py"
    module_path.write_text(
        "class Request:\n"
        "    @classmethod\n"
        "    def from_value(cls, value):\n"
        "        return cls(name=value.name, score=value.score, "
        "label=value.label)\n"
        "\n"
        "def build(value, enabled):\n"
        "    return (\n"
        "        Request\n"
        "        .from_value(value)\n"
        "        if enabled\n"
        "        else Request(name=value.name, score=value.score, "
        "label=value.label)\n"
        "    )\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    family = runtime_detectors.RepeatedBuilderCallShapeProjectionFamily

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)
    full_items = tuple(family.collect(parsed_module))
    target_items = tuple(
        item for item in full_items if item.callee_name == "from_value"
    )
    demand = family.report_demand(target_items, DetectorConfig())

    assert demand is not None
    assert tuple(family.collect_demanded(parsed_module, demand) or ()) == (
        family.project_cached_demand(full_items, demand)
    )


@pytest.mark.parametrize(
    "family, source",
    (
        (
            runtime_detectors.FormalBoundaryPythonStringConstantFamily,
            "REQUEST_PROFILE_ID = 'selection_replay_request'\n"
            "REUSE_PROFILE_ID: str = 'selection_replay_reuse'\n"
            "FINAL_PROFILE_ID = 'selection_replay_final'\n"
            "def build_profile():\n"
            "    return LeanRuntimePolicy.profile(REQUEST_PROFILE_ID)\n",
        ),
        (
            runtime_detectors.GeneratedBoundarySemanticConstantSiteFamily,
            "# generated from policy schema\n"
            "POLICY_PROFILE_ID: str = 'axis_policy_profile'\n"
            "OTHER_PROFILE_ID = MIRRORED_PROFILE_ID = 'shared_profile'\n"
            "lower_value = 'ignored'\n",
        ),
    ),
)
def test_native_constant_projection_matches_ast_family(
    tmp_path: Path,
    family: type[CollectedFamily],
    source: str,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "generated_policy.py"
    module_path.write_text(source, encoding="utf-8")
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)


def test_native_subclass_traversal_projection_matches_ast_family(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "registry.py"
    module_path.write_text(
        "class PluginBase: pass\n"
        "def all_plugins():\n"
        "    ordered = []\n"
        "    queue = list(PluginBase.__subclasses__())\n"
        "    while queue:\n"
        "        current = queue.pop(0)\n"
        "        queue.extend(current.__subclasses__())\n"
        "        if not current.__dict__.get('plugin_name'):\n"
        "            continue\n"
        "        ordered.append(current)\n"
        "    return tuple(ordered)\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    family = helper_detectors.SubclassTraversalSiteFamily

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)
    assert native is not None
    assert (
        native[0].materialization_kind
        is base_detectors.SubclassMaterializationKind.TYPE
    )


def test_native_export_policy_projection_matches_ast_family(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "exports.py"
    module_path.write_text(
        "class Root: pass\n"
        "def _is_public_export(name, value):\n"
        "    if name.startswith('_'): return False\n"
        "    if not isinstance(value, type): return False\n"
        "    return issubclass(value, Root)\n"
        "__all__ = sorted(\n"
        "    name for name, value in globals().items()\n"
        "    if _is_public_export(name, value)\n"
        ")\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    family = structural_detectors.ExportPolicyPredicateCandidateFamily

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)


def test_native_support_prelude_projection_matches_ast_family(
    tmp_path: Path,
) -> None:
    assert not hasattr(
        structural_detectors,
        "_support_prelude_module_family_candidates",
    )
    assert not hasattr(
        structural_detectors,
        "_support_prelude_module_family_candidates_from_facts",
    )
    assert not hasattr(structural_detectors, "_native_support_prelude_module_facts")

    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "support.py").write_text(
        "from pathlib import Path\n",
        encoding="utf-8",
    )
    module_path = package_root / "alpha.py"
    module_path.write_text(
        "from .support import *\n@decorator\nclass AlphaMixin: pass\n",
        encoding="utf-8",
    )
    parsed_module = next(
        module
        for module in parse_python_modules(package_root, use_parse_cache=False)
        if module.path == module_path
    )
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    family = structural_detectors.SupportPreludeModuleFactFamily

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)


@pytest.mark.parametrize(
    "family, source",
    (
        (
            systemic_detectors.CompactValidateShapeModuleProjectionFamily,
            "class Payload:\n"
            "    def validate(self):\n"
            "        if self.values.ndim != 2:\n"
            "            raise ValueError('ndim')\n"
            "        if self.values.shape[0] != self.count:\n"
            "            raise ValueError('shape')\n",
        ),
        (
            systemic_detectors._DataclassNamespaceCliModuleProjectionFamily,
            "@dataclass(frozen=True)\n"
            "class Config:\n"
            "    alpha: str\n"
            "    beta: str\n"
            "    gamma: str\n"
            "    delta: str\n"
            "    @classmethod\n"
            "    def from_namespace(cls, namespace):\n"
            "        return cls(alpha=namespace.alpha, beta=namespace.beta, "
            "gamma=namespace.gamma, delta=namespace.delta)\n"
            "CLI_ARGUMENTS = (\n"
            "    CliArgumentSpec(flags=('--alpha',)),\n"
            "    CliArgumentSpec(flags=('--beta',)),\n"
            "    CliArgumentSpec(flags=('--gamma',)),\n"
            "    CliArgumentSpec(flags=('--delta',)),\n"
            ")\n",
        ),
        (
            systemic_detectors.CompactSpecAxisModuleProjectionFamily,
            "ALPHA_SPEC = CaseSpec(stage=AlphaStage, handler=run_alpha)\n"
            "BETA_SPEC = CaseSpec(stage=BetaStage, handler=run_beta)\n",
        ),
    ),
)
def test_native_sparse_systemic_projection_matches_ast_family(
    tmp_path: Path,
    family: type[CollectedFamily],
    source: str,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "systemic.py"
    module_path.write_text(source, encoding="utf-8")
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)


def test_native_grouped_report_demands_match_ast_views(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "groups.py"
    module_path.write_text(
        "def selected(source):\n"
        "    payload = Payload(alpha=source.alpha, beta=source.beta, gamma=source.gamma)\n"
        "    return {'alpha': source.alpha, 'beta': source.beta, 'gamma': source.gamma}\n"
        "def ignored(source):\n"
        "    payload = Payload(other_a=source.alpha, other_b=source.beta, other_c=source.gamma)\n"
        "    return {'other_a': source.alpha, 'other_b': source.beta, 'other_c': source.gamma}\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    syntax_index = NativePythonSyntaxIndex.from_source(source_module.source)
    config = DetectorConfig()
    for family in (runtime_detectors.RepeatedBuilderCallShapeProjectionFamily,):
        full_items = tuple(collect_family_items(parsed_module, family))
        selected_items = tuple(
            item for item in full_items if item.function_name == "selected"
        )
        demand = family.report_demand(selected_items, config)

        native = family.collect_demanded_source(
            source_module,
            syntax_index,
            demand,
        )
        ast_view = family.collect_demanded(parsed_module, demand)

        assert native == ast_view
        assert native == list(selected_items)


def test_native_environment_projection_matches_ast_family(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "environment.py"
    module_path.write_text(
        "import os\n"
        "DISABLED_VALUES = ('0', 'false', 'no', 'off')\n"
        "def declared_environment_flag_decision(name: str) -> bool:\n"
        "    value = os.environ.get(name)\n"
        "    if value is None: raise ValueError(name)\n"
        "    return value.lower() not in DISABLED_VALUES\n"
        "def trace_enabled() -> bool:\n"
        "    return os.getenv('TRACE', '0').lower() not in DISABLED_VALUES\n"
        "class FeatureEnvironmentAuthority:\n"
        "    FEATURE_ENV = 'FEATURE_FLAG'\n"
        "    @staticmethod\n"
        "    def enabled() -> bool:\n"
        "        return declared_environment_flag_decision(\n"
        "            FeatureEnvironmentAuthority.FEATURE_ENV\n"
        "        )\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    source_module = SourceModule(
        path=parsed_module.path,
        module_name=parsed_module.module_name,
        source=parsed_module.source,
    )
    family = environment_detectors._EnvironmentBooleanModuleProjectionFamily

    native = family.collect_source(
        source_module,
        NativePythonSyntaxIndex.from_source(source_module.source),
    )

    assert native == collect_family_items(parsed_module, family)


def test_source_native_projection_shard_skips_python_ast_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "registrations.py"
    source = (
        "REGISTRY = {}\n"
        "class Alpha: pass\n"
        "class Beta: pass\n"
        "class Projection: pass\n"
        "REGISTRY['alpha'] = Alpha\n"
        "REGISTRY['beta'] = Beta\n"
        "def export(item):\n"
        "    projected = Projection(name=item.name, score=item.score, "
        "label=item.label)\n"
        "    return {'name': item.name, 'score': item.score, "
        "'label': item.label}\n"
    )
    module_path.write_text(source, encoding="utf-8")
    projection_source = analysis_module.CompactProjectionCacheSource(
        path=module_path,
        module_name="registrations",
        source_signature=ast_tools_module.python_source_cache_signature(source),
        family_cache_dir=None,
        scan_root=package_root,
        cache_dir=None,
        use_parse_cache=False,
        source_policy=ast_tools_module.PythonSourcePathPolicy(),
    )

    def unexpected_ast_parse(self, paths):
        del self, paths
        raise AssertionError("source-native family should bypass Python AST parsing")

    monkeypatch.setattr(
        ast_tools_module.PythonModuleRootParser,
        "parsed_source_paths",
        unexpected_ast_parse,
    )
    result = analysis_module.build_compact_projection_shard(
        analysis_module.CompactProjectionBuildRequest(
            source=projection_source,
            missing_families=(
                RegistrationShapeFamily,
                runtime_detectors.RepeatedBuilderCallShapeProjectionFamily,
                environment_detectors._EnvironmentBooleanModuleProjectionFamily,
                runtime_detectors.CompactAlgebraicVariantModuleProjectionFamily,
            ),
            config=DetectorConfig(),
        )
    )

    assert [
        (batch.family, len(batch.items)) for batch in result.projection_batches
    ] == [
        (RegistrationShapeFamily, 2),
        (runtime_detectors.RepeatedBuilderCallShapeProjectionFamily, 1),
        (environment_detectors._EnvironmentBooleanModuleProjectionFamily, 1),
        (runtime_detectors.CompactAlgebraicVariantModuleProjectionFamily, 1),
    ]


def test_compact_family_projection_batch_rejects_ast_payloads() -> None:
    with pytest.raises(TypeError, match="RegistrationShapeFamily projection"):
        analysis_module.CompactFamilyProjectionBatch(
            family=RegistrationShapeFamily,
            items=(ast.parse("VALUE = 1\n"),),
        )


def test_uncached_compact_analysis_skips_persistent_content_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "class Handler:\n" "    pass\n" "\n" "class Alpha(Handler):\n" "    pass\n",
        encoding="utf-8",
    )

    def unexpected_content_identity(items: tuple[object, ...]) -> str:
        del items
        raise AssertionError("disabled finding cache cannot consume this identity")

    monkeypatch.setattr(
        analysis_module,
        "collected_family_items_content_signature",
        unexpected_content_identity,
    )

    result = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=None,
        analysis_cache_dir=None,
        use_parse_cache=False,
        detector_types=(systemic_detectors.RepeatedConcreteTypeCaseAnalysisDetector,),
    )

    assert result.cache_status is AnalysisCacheStatus.DISABLED


def test_source_local_detector_requests_ast_fallback_for_lexical_binding(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "reflection.py"
    source = "def capture(value):\n    return locals()\n"
    module_path.write_text(source, encoding="utf-8")
    result = analysis_module.build_compact_projection_shard(
        analysis_module.CompactProjectionBuildRequest(
            source=analysis_module.CompactProjectionCacheSource(
                path=module_path,
                module_name="reflection",
                source_signature=ast_tools_module.python_source_cache_signature(source),
                family_cache_dir=None,
                scan_root=package_root,
                cache_dir=None,
                use_parse_cache=False,
                source_policy=ast_tools_module.PythonSourcePathPolicy(),
            ),
            missing_families=(),
            config=DetectorConfig(),
            local_detector_types=(reflection_detectors.BuiltinLocalsCallDetector,),
        )
    )

    assert [finding.detector_id for finding in result.local_findings] == [
        "builtin_locals_call"
    ]


def test_uncached_compact_ast_fallback_skips_semantic_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "reflection.py").write_text(
        "def capture(value):\n    return locals()\n",
        encoding="utf-8",
    )
    hash_call_count = 0
    semantic_hash = ast_tools_module.semantic_python_source_hash

    def counted_semantic_hash(source: str) -> str:
        nonlocal hash_call_count
        hash_call_count += 1
        return semantic_hash(source)

    monkeypatch.setattr(
        ast_tools_module,
        "semantic_python_source_hash",
        counted_semantic_hash,
    )
    monkeypatch.setattr(
        analysis_module,
        "semantic_python_source_hash",
        counted_semantic_hash,
    )

    result = analyze_compact_roots_with_cache(
        (package_root,),
        use_parse_cache=False,
        parse_workers=1,
        detector_types=(reflection_detectors.BuiltinLocalsCallDetector,),
    )

    assert hash_call_count == 0
    assert [finding.detector_id for finding in result.findings] == [
        "builtin_locals_call"
    ]


def test_source_local_detector_does_not_switch_mixed_families_to_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mixed.py"
    source = "REGISTRY = {}\nclass Alpha: pass\nREGISTRY['alpha'] = Alpha\n"
    module_path.write_text(source, encoding="utf-8")

    def unexpected_source_family(cls, source_module, syntax_index):
        del cls, source_module, syntax_index
        raise AssertionError("mixed projection families must retain their AST path")

    monkeypatch.setattr(
        RegistrationShapeFamily,
        "collect_source",
        classmethod(unexpected_source_family),
    )
    result = analysis_module.build_compact_projection_shard(
        analysis_module.CompactProjectionBuildRequest(
            source=analysis_module.CompactProjectionCacheSource(
                path=module_path,
                module_name="mixed",
                source_signature=ast_tools_module.python_source_cache_signature(source),
                family_cache_dir=None,
                scan_root=package_root,
                cache_dir=None,
                use_parse_cache=False,
                source_policy=ast_tools_module.PythonSourcePathPolicy(),
            ),
            missing_families=(
                RegistrationShapeFamily,
                systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
            ),
            config=DetectorConfig(),
            local_detector_types=(reflection_detectors.BuiltinLocalsCallDetector,),
        )
    )

    assert [batch.family for batch in result.projection_batches] == [
        RegistrationShapeFamily,
        systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
    ]
    assert result.local_findings == ()


def test_report_presence_demand_skips_context_only_single_family_facts(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "context.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    family = runtime_detectors.GeneratedBoundarySemanticConstantSiteFamily

    empty_demand = family.report_demand((), DetectorConfig())
    present_demand = family.report_demand((object(),), DetectorConfig())

    assert isinstance(empty_demand, ast_tools_module.CollectedFamilyPresenceDemand)
    assert empty_demand.include_context is False
    assert family.collect_demanded(parsed_module, empty_demand) == []
    assert family.project_cached_demand((object(),), empty_demand) == ()
    assert isinstance(present_demand, ast_tools_module.CollectedFamilyPresenceDemand)
    assert present_demand.include_context is True
    assert family.collect_demanded(parsed_module, present_demand) is None


def test_report_context_witness_skips_detector_without_target_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    target_path = package_root / "target.py"
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    (package_root / "generated.py").write_text(
        "# generated from policy schema\nPOLICY_ID = 'shared'\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text(
        "POLICY_ID = 'shared'\n",
        encoding="utf-8",
    )
    family = runtime_detectors.GeneratedBoundarySemanticConstantSiteFamily
    original_collect = family.collect.__func__
    collected_paths: list[Path] = []

    def observed_collect(cls, parsed_module):
        collected_paths.append(parsed_module.path.resolve())
        return original_collect(cls, parsed_module)

    monkeypatch.setattr(family, "collect", classmethod(observed_collect))
    result = analyze_compact_roots_with_cache(
        (package_root,),
        use_parse_cache=False,
        analysis_cache_dir=None,
        parse_workers=1,
        report_scope=AnalysisPathScope(
            analysis_roots=(package_root,),
            report_roots=(target_path,),
        ),
        detector_types=(
            runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector,
        ),
    )

    assert result.findings == []
    assert collected_paths == [target_path.resolve()]


def test_report_context_witness_retains_context_promotion_for_target_projection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    target_path = package_root / "target.py"
    target_path.write_text("POLICY_ID = 'shared'\n", encoding="utf-8")
    (package_root / "generated.py").write_text(
        "# generated from policy schema\nPOLICY_ID = 'shared'\n",
        encoding="utf-8",
    )
    result = analyze_compact_roots_with_cache(
        (package_root,),
        use_parse_cache=False,
        analysis_cache_dir=None,
        parse_workers=1,
        report_scope=AnalysisPathScope(
            analysis_roots=(package_root,),
            report_roots=(target_path,),
        ),
        detector_types=(
            runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector,
        ),
    )

    assert {finding.detector_id for finding in result.findings} == {
        "generated_boundary_semantic_constant_mirror"
    }
    assert any(
        evidence.file_path == target_path.as_posix()
        for finding in result.findings
        for evidence in finding.evidence
    )


def test_class_candidate_anchor_witnesses_follow_reported_seed_locations() -> None:
    family = class_index_module.CompactModuleClassProjectionFamily
    empty_projection = class_index_module.CompactModuleClassProjection(
        module_name="pkg.target",
        file_path="/repo/pkg/target.py",
        import_aliases=(),
        classes=(),
    )
    base_class = class_index_module.CompactIndexedClass(
        symbol="pkg.target.Root",
        module_name="pkg.target",
        qualname="Root",
        simple_name="Root",
        file_path="/repo/pkg/target.py",
        line=1,
        declared_base_names=(),
        base_reference_parts=(),
    )
    autoregister_projection = replace(
        empty_projection,
        classes=(replace(base_class, declares_autoregister_meta=True),),
    )
    predicate_projection = replace(
        empty_projection,
        classes=(
            replace(
                base_class,
                direct_assignment_expressions=(("_registered_types", "[]"),),
                predicate_selected_methods=((2, "select", "matches", "context"),),
            ),
        ),
    )
    keyed_registry_projection = replace(
        empty_projection,
        classes=(
            replace(
                base_class,
                direct_assignment_expressions=(("registry_key_attr", "'kind'"),),
                keyed_family_key_type_name="Kind",
            ),
        ),
    )
    detector_projection_pairs = (
        (
            runtime_detectors.ManualConcreteSubclassRosterDetector,
            replace(empty_projection, manual_subclass_roster_roots=(object(),)),
        ),
        (
            runtime_detectors.LatentImplementationRosterDetector,
            replace(empty_projection, latent_rosters=(object(),)),
        ),
        (
            runtime_detectors.AutoRegisterMetaUnderRentedDetector,
            autoregister_projection,
        ),
        (
            runtime_detectors.PredicateSelectedConcreteFamilyDetector,
            predicate_projection,
        ),
        (
            surface_detectors.ManualFamilyRosterDetector,
            replace(empty_projection, manual_family_rosters=(object(),)),
        ),
        (
            systemic_detectors.RepeatedKeyedFamilyDetector,
            replace(empty_projection, repeated_keyed_family_roots=(object(),)),
        ),
        (
            systemic_detectors.CrossModuleAxisShadowFamilyDetector,
            replace(empty_projection, manual_selector_axes=(object(),)),
        ),
        (
            systemic_detectors.ResidualClosedAxisBranchingDetector,
            replace(empty_projection, closed_axis_branch_functions=(object(),)),
        ),
        (
            systemic_detectors.ParallelKeyedAxisFamilyDetector,
            keyed_registry_projection,
        ),
        (
            systemic_detectors.ParallelKeyedTableAxisDetector,
            replace(empty_projection, keyed_table_axes=(object(),)),
        ),
        (
            systemic_detectors.ParallelKeyedTableAndFamilyDetector,
            replace(empty_projection, keyed_table_axes=(object(),)),
        ),
        (
            systemic_detectors.NonInjectiveTypeRegistryDetector,
            keyed_registry_projection,
        ),
        (
            systemic_detectors.InjectiveTypeRegistryDetector,
            keyed_registry_projection,
        ),
        (
            systemic_detectors.PrematureRegistryInfrastructureDetector,
            keyed_registry_projection,
        ),
        (
            systemic_detectors.RegistryProjectionSurfaceDetector,
            replace(empty_projection, named_projection_surfaces=(object(),)),
        ),
        (
            systemic_detectors.RegistryProjectionPolicyAuthorityDetector,
            replace(empty_projection, named_projection_surfaces=(object(),)),
        ),
    )
    config = DetectorConfig()

    for detector_type, anchored_projection in detector_projection_pairs:
        assert not detector_type.compact_report_context_can_promote(
            {family: (empty_projection,)},
            config,
        )
        assert detector_type.compact_report_context_can_promote(
            {family: (anchored_projection,)},
            config,
        )


def test_class_demand_omits_unreportable_autoregister_reference_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    context_path = package_root / "context.py"
    context_path.write_text(
        "class ContextRegistry(metaclass=AutoRegisterMeta):\n"
        "    pass\n\n"
        "def consume():\n"
        "    return ContextRegistry.__registry__\n",
        encoding="utf-8",
    )
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    family = class_index_module.CompactModuleClassProjectionFamily
    empty_target = class_index_module.CompactModuleClassProjection(
        module_name="pkg.target",
        file_path=str(package_root / "target.py"),
        import_aliases=(),
        classes=(),
    )
    demand = family.report_demand((empty_target,), DetectorConfig())

    assert isinstance(demand, class_index_module.CompactClassProjectionDemand)
    assert demand.include_autoregister_references is False

    collect_autoregister_values: list[bool] = []
    original_collector = class_index_module._compact_class_syntax_facets

    def recording_syntax_facets(
        selected_module,
        *,
        collect_autoregister: bool = True,
    ):
        collect_autoregister_values.append(collect_autoregister)
        return original_collector(
            selected_module,
            collect_autoregister=collect_autoregister,
        )

    monkeypatch.setattr(
        class_index_module,
        "_compact_class_syntax_facets",
        recording_syntax_facets,
    )
    demanded = family.collect_demanded(parsed_module, demand)

    assert demanded is not None
    assert demanded[0].autoregister_function_references == ()
    assert demanded[0].autoregister_reference_index is None
    assert collect_autoregister_values == [False]

    target_root = class_index_module.CompactIndexedClass(
        symbol="pkg.target.TargetRegistry",
        module_name="pkg.target",
        qualname="TargetRegistry",
        simple_name="TargetRegistry",
        file_path=str(package_root / "target.py"),
        line=1,
        declared_base_names=(),
        base_reference_parts=(),
        declares_autoregister_meta=True,
    )
    positive_demand = family.report_demand(
        (replace(empty_target, classes=(target_root,)),),
        DetectorConfig(),
    )
    assert isinstance(positive_demand, class_index_module.CompactClassProjectionDemand)
    assert positive_demand.include_autoregister_references is True


def test_native_definition_headers_preserve_decorators_lines_and_full_span() -> None:
    source = (
        "class Outer:\n"
        "    @dataclass(frozen=True)\n"
        "    class Inner(Base, metaclass=RegistryMeta):\n"
        "        @classmethod\n"
        "        @abstractmethod\n"
        "        async def choose(cls, value: int) -> str:\n"
        "            return str(value)\n"
        "        # The canonical AST span excludes trailing comments.\n"
        "    # The enclosing class span excludes them too.\n"
    )
    syntax_index = NativePythonSyntaxIndex.from_source(source)
    classes = syntax_index.common_captures()["class"]
    functions = syntax_index.common_captures()["function"]
    inner = next(
        node for node in classes if syntax_index.declared_name(node) == "Inner"
    )
    choose = next(
        node for node in functions if syntax_index.declared_name(node) == "choose"
    )

    class_header = syntax_index.class_header_for(inner)
    function_header = syntax_index.function_header_for(choose)

    assert class_header.lineno == 3
    assert class_header.end_lineno == 7
    assert ast.unparse(class_header.decorator_list[0]) == "dataclass(frozen=True)"
    assert [ast.unparse(base) for base in class_header.bases] == ["Base"]
    assert ast.unparse(class_header.keywords[0].value) == "RegistryMeta"
    assert function_header.lineno == 6
    assert function_header.end_lineno == 7
    assert [ast.unparse(item) for item in function_header.decorator_list] == [
        "classmethod",
        "abstractmethod",
    ]
    assert function_header.args.args[-1].annotation is not None
    assert isinstance(function_header, ast.AsyncFunctionDef)


def test_native_class_header_core_matches_cached_minimal_projection(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    source_path = package_root / "mod.py"
    source = (
        "from __future__ import annotations\n"
        "from support import Parent as ImportedParent\n"
        "\n"
        "@final\n"
        "class Child(ImportedParent):\n"
        "    registry_key = 'child'\n"
        "\n"
        "    @classmethod\n"
        "    def select(cls, value: int) -> str:\n"
        "        cls.__registry__[value] = cls\n"
        "        return str(value)\n"
        "\n"
        "    class Nested(ImportedParent):\n"
        "        pass\n"
    )
    source_path.write_text(source, encoding="utf-8")
    parsed_module = parse_python_modules(package_root, use_parse_cache=False)[0]
    family = class_index_module.CompactModuleClassProjectionFamily
    demand = class_index_module.CompactClassProjectionDemand(
        abc_method_names=frozenset(),
        header_core_only=True,
    )
    full_items = tuple(family.collect(parsed_module))
    expected = family.project_cached_demand(full_items, demand)
    actual = family.collect_demanded_source(
        SourceModule(source_path, "mod", source),
        NativePythonSyntaxIndex.from_source(source),
        demand,
    )

    assert actual is not None
    assert tuple(actual) == expected
    assert [item.qualname for item in actual[0].classes] == ["Child", "Child.Nested"]
    child = actual[0].classes[0]
    assert child.declared_base_names == ("ImportedParent",)
    assert child.is_final is True
    assert child.direct_assignment_expressions == ()
    assert child.method_names == ()
    assert dict(actual[0].import_aliases)["ImportedParent"] == "support.Parent"
    assert class_index_module.CompactIndexedClass.__mro__[:3] == (
        class_index_module.CompactIndexedClass,
        class_index_module.CompactClassHeader,
        class_index_module.ClassDeclaration,
    )
    assert class_index_module.CompactModuleClassProjection.__mro__[:3] == (
        class_index_module.CompactModuleClassProjection,
        class_index_module.CompactClassSyntaxFacets,
        class_index_module.CompactModuleClassHeader,
    )
    assert not hasattr(class_index_module, "_CompactClassSyntaxFacets")
    assert not hasattr(
        class_index_module,
        "_compact_autoregister_function_references",
    )
    assert not hasattr(class_index_module, "_compact_closed_axis_branch_functions")
    assert not hasattr(class_index_module, "_compact_exact_type_guards")
    assert not hasattr(class_index_module, "_CLASS_HEADER_CORE_CLASS_DEFAULTS")
    assert not hasattr(class_index_module, "_CLASS_HEADER_CORE_MODULE_DEFAULTS")
    assert not hasattr(
        class_index_module.CompactModuleClassProjection,
        "nominal_class_first_line_overrides",
    )
    assert not hasattr(
        class_index_module.CompactModuleClassProjection,
        "extra_nominal_class_bases",
    )
    assert not hasattr(class_index_module, "_compact_nominal_class_scope_facts")


def test_grouped_report_demands_preserve_target_findings_and_drop_other_groups(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    target_path = package_root / "target.py"
    context_path = package_root / "context.py"
    irrelevant_path = package_root / "irrelevant.py"

    def module_source(class_name: str, field_prefix: str = "") -> str:
        return (
            "class Payload:\n"
            "    pass\n"
            "\n"
            f"class {class_name}:\n"
            "    def _shared(self, value):\n"
            "        first = value + 1\n"
            "        second = first * 2\n"
            "        return second\n"
            "    def _other(self, value):\n"
            "        first = value - 1\n"
            "        second = first / 2\n"
            "        return second\n"
            "    def build(self, source):\n"
            f"        return Payload({field_prefix}alpha=source.alpha, "
            f"{field_prefix}beta=source.beta, {field_prefix}gamma=source.gamma)\n"
            "    def rebuild(self, source):\n"
            f"        return Payload({field_prefix}alpha=source.alpha, "
            f"{field_prefix}beta=source.beta, {field_prefix}gamma=source.gamma)\n"
            "    def export(self, source):\n"
            f"        return {{'{field_prefix}alpha': source.alpha, "
            f"'{field_prefix}beta': source.beta, "
            f"'{field_prefix}gamma': source.gamma}}\n"
        )

    target_path.write_text(module_source("Target"), encoding="utf-8")
    context_path.write_text(module_source("Context"), encoding="utf-8")
    irrelevant_path.write_text(
        module_source("Irrelevant", "other_"),
        encoding="utf-8",
    )
    modules = parse_python_modules(package_root, use_parse_cache=False)
    scope = AnalysisPathScope(
        analysis_roots=(package_root,),
        report_roots=(target_path,),
    )
    config = DetectorConfig()
    family_detector_pairs = (
        (
            runtime_detectors.RepeatedBuilderCallShapeProjectionFamily,
            runtime_detectors.RepeatedBuilderCallDetector(),
        ),
    )
    for family, detector in family_detector_pairs:
        full_items = tuple(
            item for module in modules for item in collect_family_items(module, family)
        )
        target_items = tuple(
            item
            for item in full_items
            if scope.includes_report_file_path(item.file_path)
        )
        context_items = tuple(
            item
            for item in full_items
            if not scope.includes_report_file_path(item.file_path)
        )
        demand = family.report_demand(target_items, config)
        demanded_items = target_items + family.project_cached_demand(
            context_items,
            demand,
        )
        full_findings = scope.filter_findings(
            detector._findings_from_compact_projections(full_items, config)
        )
        demanded_findings = scope.filter_findings(
            detector._findings_from_compact_projections(demanded_items, config)
        )

        assert full_findings
        assert [finding.to_dict() for finding in demanded_findings] == [
            finding.to_dict() for finding in full_findings
        ]
        assert len(demanded_items) < len(full_items)


def test_cold_focused_semantic_scan_omits_only_context_presentations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    context_path = package_root / "context.py"
    target_path = package_root / "target.py"
    context_path.write_text(
        "class Step:\n"
        "    pass\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n",
        encoding="utf-8",
    )
    target_path.write_text(
        "from .context import LoadStep, SaveStep\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )
    report_scope = AnalysisPathScope(
        analysis_roots=(package_root,),
        report_roots=(target_path,),
    )
    detector_types = (semantic_descent_detectors.SemanticMirrorWithoutDescentDetector,)
    eager = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=tmp_path / "eager-parse-cache",
        analysis_cache_dir=tmp_path / "eager-analysis-cache",
        use_parse_cache=True,
        parse_workers=1,
        report_scope=report_scope,
        detector_types=detector_types,
    )
    family = CompactSemanticModuleProjectionFamily
    original_collector = family.collect_demanded
    demanded_paths: list[Path] = []

    def observed_collect_demanded(cls, parsed_module, demand):
        demanded_paths.append(parsed_module.path.resolve())
        items = original_collector(parsed_module, demand)
        assert items is not None
        assert all(not item.projections for item in items)
        return items

    monkeypatch.setattr(
        family,
        "collect_demanded",
        classmethod(observed_collect_demanded),
    )
    demanded = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=None,
        analysis_cache_dir=None,
        use_parse_cache=False,
        parse_workers=1,
        report_scope=report_scope,
        detector_types=detector_types,
    )

    assert [finding.to_dict() for finding in demanded.findings] == [
        finding.to_dict() for finding in eager.findings
    ]
    assert {finding.detector_id for finding in demanded.findings} == {
        "semantic_mirror_without_descent"
    }
    assert set(demanded_paths) == {
        (package_root / "__init__.py").resolve(),
        context_path.resolve(),
    }


def test_mixed_projection_shard_uses_only_python_ast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mixed.py"
    source = (
        "REGISTRY = {}\n"
        "class Alpha: pass\n"
        "REGISTRY['alpha'] = Alpha\n"
        "def export(item):\n"
        "    return {'name': item.name, 'score': item.score}\n"
    )
    module_path.write_text(source, encoding="utf-8")
    family_cache_dir = tmp_path / "family-cache"
    projection_source = analysis_module.CompactProjectionCacheSource(
        path=module_path,
        module_name="mixed",
        source_signature=ast_tools_module.python_source_cache_signature(source),
        family_cache_dir=family_cache_dir,
        scan_root=package_root,
        cache_dir=None,
        use_parse_cache=False,
        source_policy=ast_tools_module.PythonSourcePathPolicy(),
    )

    def unexpected_native_parse(cls, source_text):
        del cls, source_text
        raise AssertionError("mixed shard should not build a second syntax tree")

    monkeypatch.setattr(
        NativePythonSyntaxIndex,
        "from_source",
        classmethod(unexpected_native_parse),
    )
    result = analysis_module.build_compact_projection_shard(
        analysis_module.CompactProjectionBuildRequest(
            source=projection_source,
            missing_families=(
                RegistrationShapeFamily,
                systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
            ),
            config=DetectorConfig(),
            bundle_families=(
                RegistrationShapeFamily,
                systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
            ),
        )
    )

    assert [batch.family for batch in result.projection_batches] == [
        RegistrationShapeFamily,
        systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
    ]
    assert result.cache_bundle_complete
    assert {
        batch.family: batch.content_signature for batch in result.projection_batches
    } == {
        batch.family: ast_tools_module.collected_family_items_content_signature(
            batch.items
        )
        for batch in result.projection_batches
    }
    assert (
        projection_source.load_content_signature(
            systemic_detectors.CompactRemainingSystemicModuleProjectionFamily,
        )
        is not None
    )


def test_compact_root_analysis_matches_full_ast_and_reuses_aggregate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated_policy.py").write_text(
        "# generated from policy schema\nPOLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text(
        "POLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".nra-cache" / "ast"
    analysis_cache_dir = tmp_path / ".nra-cache" / "analysis"
    modules = parse_python_modules(package_root, use_parse_cache=False)
    expected = analyze_modules(modules, DetectorConfig(), analysis_workers=1)

    cold = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        parse_workers=2,
    )

    assert [finding.to_dict() for finding in cold.findings] == [
        finding.to_dict() for finding in expected
    ]
    assert cold.cache_status is AnalysisCacheStatus.MISS
    assert cold.projection_count > 0

    def unexpected_parser(*args, **kwargs):
        del args, kwargs
        raise AssertionError("aggregate hit should bypass compact projection parsing")

    monkeypatch.setattr(
        ast_tools_module.PythonModuleRootParser,
        "for_root",
        unexpected_parser,
    )
    warm = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
    )

    assert warm.cache_status is AnalysisCacheStatus.HIT
    assert warm.findings == cold.findings
    assert warm.projection_count == 0


def test_compact_incremental_analysis_reuses_consolidated_family_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated_policy.py").write_text(
        "# generated from policy schema\nPOLICY_PROFILE_ID = 'axis_policy_profile'\n",
        encoding="utf-8",
    )
    runtime_path = package_root / "runtime.py"
    runtime_source = "POLICY_PROFILE_ID = 'axis_policy_profile'\n"
    runtime_path.write_text(runtime_source, encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "ast"
    analysis_cache_dir = tmp_path / ".nra-cache" / "analysis"

    cold = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
    )
    signature_index_path = (
        cache_dir / "collected-family" / "content-signature-index-v1.pickle"
    )
    assert signature_index_path.is_file()

    runtime_path.write_text(
        f"{runtime_source}# comment-only edit\n",
        encoding="utf-8",
    )

    def unexpected_individual_signature_load(self, family, demand_signature=""):
        del self, family, demand_signature
        raise AssertionError(
            "consolidated index should cover unchanged source families"
        )

    monkeypatch.setattr(
        ast_tools_module.CollectedFamilyCacheContext,
        "load_content_signature",
        unexpected_individual_signature_load,
    )
    incremental = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
    )

    assert incremental.cache_status is AnalysisCacheStatus.PARTIAL
    assert incremental.findings == cold.findings


def test_collected_family_content_signature_index_rejects_stale_source(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "collected-family"
    index = ast_tools_module.CollectedFamilyContentSignatureIndex.load(cache_dir)
    source_v1 = ast_tools_module.CollectedFamilyCacheContext(
        path=Path("/checkout/pkg/mod.py"),
        module_name="pkg.mod",
        source_signature="source-v1",
        family_cache_dir=cache_dir,
    )
    index.record(
        source_v1.identity(RegistrationShapeFamily),
        content_signature="content-v1",
    )
    index.store_if_dirty()

    reloaded = ast_tools_module.CollectedFamilyContentSignatureIndex.load(cache_dir)
    assert reloaded.lookup(source_v1.identity(RegistrationShapeFamily)) == "content-v1"
    source_v2 = ast_tools_module.CollectedFamilyCacheContext(
        path=source_v1.path,
        module_name=source_v1.module_name,
        source_signature="source-v2",
        family_cache_dir=source_v1.family_cache_dir,
    )
    assert reloaded.lookup(source_v2.identity(RegistrationShapeFamily)) is None


def test_compact_family_bundle_marker_skips_per_family_cache_stat_fanout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    source = "class Example:\n    pass\n"
    module_path.write_text(source, encoding="utf-8")
    cache_dir = tmp_path / ".nra-cache" / "ast"
    analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=tmp_path / ".nra-cache" / "analysis",
    )
    partition = DetectorTypePartition.from_detector_types(
        default_detector_types_for_analysis()
    )
    families = tuple(
        dict.fromkeys(
            family
            for detector_type in partition.compact_global_detector_types
            for family in detector_type.compact_projection_families()
        )
    )
    parser = ast_tools_module.PythonModuleRootParser.for_root(
        package_root,
        cache_dir=cache_dir,
    )
    module_identity = ast_tools_module.PythonModulePathIdentity.from_path(
        module_path,
        parser.analysis_root,
    )
    family_cache = ast_tools_module.CollectedFamilyCacheContext(
        path=module_path,
        module_name=module_identity.import_name,
        source_signature=ast_tools_module.python_source_cache_signature(source),
        family_cache_dir=parser.collected_family_cache_dir,
    )
    assert family_cache.bundle_is_complete(families)
    family_cache_dir = parser.collected_family_cache_dir
    assert family_cache_dir is not None
    family_entries = tuple(
        ast_tools_module.CollectedFamilyProjectionIdentity.from_identity(
            family_cache.identity(family)
        )
        for family in families
    )
    marker_path = family_cache._bundle_marker_path(family_entries)
    marker_path.write_bytes(b"complete\n")
    assert family_cache.bundle_is_complete(families)
    assert marker_path.read_bytes() == b"complete-v4\n"

    def unexpected_family_stat(self, family, demand_signature=""):
        del self, family, demand_signature
        raise AssertionError("complete bundle marker should bypass family stat fan-out")

    monkeypatch.setattr(
        ast_tools_module.CollectedFamilyCacheContext,
        "entry_exists",
        unexpected_family_stat,
    )
    assert family_cache.bundle_is_complete(families)


def test_demanded_family_bundle_marker_skips_per_family_cache_stat_fanout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "context.py"
    source = "VALUE = 1\n"
    source_path.write_text(source, encoding="utf-8")
    family_cache_dir = tmp_path / "collected-family"
    family = runtime_detectors.RepeatedBuilderCallShapeProjectionFamily
    demand = runtime_detectors.RepeatedBuilderCallProjectionDemand(
        exact_mapping_keys=frozenset(),
        owner_family_keys=frozenset(),
    )
    demand_signature = ast_tools_module.collected_family_demand_cache_signature(demand)
    source_signature = ast_tools_module.python_source_cache_signature(source)
    family_cache = ast_tools_module.CollectedFamilyCacheContext(
        path=source_path,
        module_name="context",
        source_signature=source_signature,
        family_cache_dir=family_cache_dir,
    )
    family_cache.store_items(family, (), demand_signature)
    assert family_cache.bundle_is_complete(
        (family,),
        ((family, demand_signature),),
    )

    def unexpected_family_stat(self, family, demand_signature=""):
        del self, family, demand_signature
        raise AssertionError("complete demand bundle should bypass family stat fan-out")

    monkeypatch.setattr(
        ast_tools_module.CollectedFamilyCacheContext,
        "entry_exists",
        unexpected_family_stat,
    )
    assert family_cache.bundle_is_complete(
        (family,),
        ((family, demand_signature),),
    )


def test_compact_family_cache_rejects_zero_byte_failed_write(tmp_path: Path) -> None:
    source_path = tmp_path / "mod.py"
    source = "VALUE = 1\n"
    source_path.write_text(source, encoding="utf-8")
    family_cache_dir = tmp_path / "collected-family"
    family_cache_dir.mkdir()
    family = BuilderCallShapeFamily
    family_cache = ast_tools_module.CollectedFamilyCacheContext(
        path=source_path,
        module_name="mod",
        source_signature=ast_tools_module.python_source_cache_signature(source),
        family_cache_dir=family_cache_dir,
    )
    identity = family_cache.identity(family)
    ast_tools_module._collected_family_cache_path(
        family_cache_dir, identity
    ).write_bytes(b"")

    assert not family_cache.entry_exists(family)


def test_compact_family_cache_identity_derives_item_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class StringItem:
        value: str

    @dataclass(frozen=True)
    class IntegerItem:
        value: int

    StringItem.__module__ = IntegerItem.__module__ = "fixture"
    StringItem.__qualname__ = IntegerItem.__qualname__ = "SchemaItem"
    source_path = tmp_path / "mod.py"
    family_cache = ast_tools_module.CollectedFamilyCacheContext(
        path=source_path,
        module_name="mod",
        source_signature="source",
        family_cache_dir=None,
    )

    monkeypatch.setattr(BuilderCallShapeFamily, "item_type", StringItem)
    BuilderCallShapeFamily.item_schema_signature.cache_clear()
    string_identity = family_cache.identity(BuilderCallShapeFamily)
    monkeypatch.setattr(BuilderCallShapeFamily, "item_type", IntegerItem)
    BuilderCallShapeFamily.item_schema_signature.cache_clear()
    integer_identity = family_cache.identity(BuilderCallShapeFamily)

    assert (
        string_identity.family_schema.item_type_module
        == integer_identity.family_schema.item_type_module
    )
    assert (
        string_identity.family_schema.item_type_qualname
        == integer_identity.family_schema.item_type_qualname
    )
    assert (
        string_identity.family_schema.item_schema_signature
        != integer_identity.family_schema.item_schema_signature
    )
    assert string_identity.cache_token != integer_identity.cache_token


def test_compact_global_detector_shards_reuse_across_report_targets(
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
    cache_dir = tmp_path / ".nra-cache" / "ast"
    analysis_cache_dir = tmp_path / ".nra-cache" / "analysis"
    first_scope = AnalysisPathScope(
        analysis_roots=(package_root,),
        report_roots=(generated_path,),
    )
    second_scope = AnalysisPathScope(
        analysis_roots=(package_root,),
        report_roots=(runtime_path,),
    )
    modules = parse_python_modules(package_root, use_parse_cache=False)
    all_findings = analyze_modules(modules, DetectorConfig(), analysis_workers=1)

    first = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        report_scope=first_scope,
    )
    assert first.findings == first_scope.filter_findings(all_findings)

    second = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        report_scope=second_scope,
    )

    assert second.cache_status is AnalysisCacheStatus.PARTIAL
    assert second.findings == second_scope.filter_findings(all_findings)
    assert first.projection_count > second.projection_count


def test_compact_root_analysis_consumes_global_detector_shards_without_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "generated.py").write_text(
        "# generated file\nSEMANTIC_MODE = 'canonical'\n",
        encoding="utf-8",
    )
    detector_types = (
        runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector,
        structural_detectors.ExportPolicyPredicateDetector,
    )
    observed_retain_findings: list[bool] = []
    observed_consumer: list[object] = []
    observed_inner_retention: list[tuple[bool, bool]] = []
    stored_identities: list[object] = []
    original_join = (
        analysis_module.BoundedCompactProjectionManifest.findings_by_detector
    )
    original_store = AnalysisFindingCache.store
    original_compact_join = analysis_module._compact_findings_by_detector

    def observing_join(self, config, **kwargs):
        observed_retain_findings.append(kwargs["retain_findings"])
        observed_consumer.append(kwargs["finding_consumer"])
        return original_join(self, config, **kwargs)

    def observing_store(self, identity, findings, *args, **kwargs):
        stored_identities.append(identity)
        return original_store(self, identity, findings, *args, **kwargs)

    def observing_compact_join(*args, **kwargs):
        observed_inner_retention.append(
            (
                kwargs["retain_findings"],
                callable(kwargs["finding_consumer"]),
            )
        )
        return original_compact_join(*args, **kwargs)

    monkeypatch.setattr(
        analysis_module.BoundedCompactProjectionManifest,
        "findings_by_detector",
        observing_join,
    )
    monkeypatch.setattr(AnalysisFindingCache, "store", observing_store)
    monkeypatch.setattr(
        analysis_module,
        "_compact_findings_by_detector",
        observing_compact_join,
    )

    analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=tmp_path / ".nra-cache" / "ast",
        analysis_cache_dir=tmp_path / ".nra-cache" / "analysis",
        detector_types=detector_types,
    )

    assert observed_retain_findings == [False]
    assert len(observed_consumer) == 1
    assert callable(observed_consumer[0])
    assert observed_inner_retention
    assert set(observed_inner_retention) == {(False, True)}
    assert sum(
        isinstance(identity, GlobalDetectorAnalysisCacheIdentity)
        for identity in stored_identities
    ) == len(detector_types)
    assert not any(
        isinstance(identity, GlobalDetectorFamilyAnalysisCacheIdentity)
        for identity in stored_identities
    )


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


def test_compact_keyed_axis_projection_is_the_only_global_candidate_authority(
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
        "    return 'beta'\n"
        "\n"
        "def resolve_nested(mode):\n"
        "    def nested(candidate):\n"
        "        if candidate == Mode.ALPHA:\n"
        "            return 'alpha'\n"
        "        if candidate == Mode.BETA:\n"
        "            return 'beta'\n"
        "    return nested(mode)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    projections = (
        systemic_detectors.ParallelKeyedAxisFamilyDetector.compact_module_projections(
            modules
        )
    )
    config = DetectorConfig()
    context = systemic_detectors.CompactClassRepositoryContext.from_projections(
        projections,
        config,
    )
    family_specs = systemic_detectors._compact_keyed_family_axis_specs_from_context(
        context
    )
    table_specs = systemic_detectors._compact_keyed_table_axis_specs(projections)
    manual_selector_specs = systemic_detectors._compact_manual_selector_axis_specs(
        projections
    )
    detectors_and_candidates = (
        (
            systemic_detectors.ParallelKeyedAxisFamilyDetector(),
            systemic_detectors._parallel_keyed_axis_family_candidates_from_specs(
                family_specs
            ),
        ),
        (
            systemic_detectors.ParallelKeyedTableAndFamilyDetector(),
            systemic_detectors._parallel_keyed_table_and_family_candidates_from_specs(
                family_specs,
                table_specs,
            ),
        ),
        (
            systemic_detectors.ParallelKeyedTableAxisDetector(),
            systemic_detectors._parallel_keyed_table_axis_candidates_from_specs(
                table_specs
            ),
        ),
        (
            systemic_detectors.ResidualClosedAxisBranchingDetector(),
            systemic_detectors._residual_closed_axis_branching_candidates_from_compact_specs(
                projections,
                family_specs,
            ),
        ),
        (
            systemic_detectors.CrossModuleAxisShadowFamilyDetector(),
            systemic_detectors._cross_module_axis_shadow_family_candidates_from_specs(
                family_specs,
                manual_selector_specs,
            ),
        ),
    )

    assert family_specs
    assert table_specs
    assert manual_selector_specs
    assert detectors_and_candidates[0][1]
    for detector, candidates in detectors_and_candidates:
        assert detector._candidate_items(list(modules), config) == candidates
        assert "candidate_collector" not in type(detector).__dict__
    for removed_name in (
        "_compact_keyed_family_axis_specs",
        "_parallel_keyed_axis_family_candidates",
        "_parallel_keyed_table_and_family_candidates",
        "_parallel_keyed_table_axis_candidates",
        "_residual_closed_axis_branching_candidates",
        "_residual_closed_axis_branching_candidates_from_compact_projections",
        "_cross_module_axis_shadow_family_candidates",
        "_manual_selector_axis_specs",
    ):
        assert not hasattr(systemic_detectors, removed_name)
    assert not hasattr(
        systemic_detectors.DispatchAlgebraAuthority,
        "keyed_family_axis_specs",
    )
    assert not hasattr(
        systemic_detectors.DispatchAlgebraAuthority,
        "module_keyed_table_axis_specs",
    )


def test_compact_dataclass_cli_projection_preserves_semantics_without_ast_shadow(
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

    candidates = (
        systemic_detectors._dataclass_namespace_cli_mirror_candidates_from_projections(
            projections
        )
    )
    assert len(candidates) == 1
    assert candidates[0].class_name == "RunConfig"
    assert candidates[0].field_names == ("alpha", "beta", "gamma", "delta")
    assert candidates[0].cli_field_names == ("alpha", "beta", "gamma", "delta")
    assert (
        systemic_detectors.DataclassNamespaceCliMirrorDetector()._candidate_items(
            modules,
            DetectorConfig(),
        )
        == candidates
    )
    assert not hasattr(
        systemic_detectors,
        "_dataclass_namespace_cli_mirror_candidates",
    )
    assert "candidate_collector" not in (
        systemic_detectors.DataclassNamespaceCliMirrorDetector.__dict__
    )


def test_compact_exact_type_guard_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    assert not hasattr(runtime_detectors, "ExactTypeGuardPredicate")
    assert not hasattr(runtime_detectors, "ExactTypeComparisonAuthority")
    assert not hasattr(runtime_detectors, "EXACT_TYPE_COMPARISON_AUTHORITY")
    assert not hasattr(runtime_detectors, "ExactTypeGuardBoundaryCollector")
    assert not hasattr(runtime_detectors, "FailLoudBlockAuthority")
    assert not hasattr(runtime_detectors, "FAIL_LOUD_BLOCK_AUTHORITY")

    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "class Boundary:\n    pass\n\nclass ConcreteBoundary(Boundary):\n    pass\n",
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
        "    assert type(value) is ImportedBoundary\n"
        "\n"
        "def require_nested(value):\n"
        "    def nested(candidate):\n"
        "        assert type(candidate) is ImportedBoundary\n"
        "    return nested(value)\n"
        "\n"
        "def shadowed_outer(type, value):\n"
        "    def nested(candidate):\n"
        "        assert type(candidate) is ImportedBoundary\n"
        "    return nested(value)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = runtime_detectors.ExactTypeGuardInheritanceRetreatDetector()
    projections = detector.compact_module_projections(modules)
    candidates = (
        runtime_detectors._exact_type_guard_candidates_from_compact_projections(
            projections
        )
    )

    assert tuple(candidate.guard.qualname for candidate in candidates) == (
        "require_boundary",
        "assert_boundary",
        "require_nested.nested",
    )
    assert all(
        candidate.base_class.simple_name == "Boundary" for candidate in candidates
    )
    assert all(
        tuple(descendant.simple_name for descendant in candidate.descendant_classes)
        == ("ConcreteBoundary",)
        for candidate in candidates
    )
    assert detector._candidate_items(list(modules), DetectorConfig()) == candidates
    assert "_candidate_items" not in type(detector).__dict__
    assert "_findings_from_compact_projections" not in type(detector).__dict__
    assert "_findings_from_compact_context" not in type(detector).__dict__


def test_compact_autoregister_rent_projection_matches_legacy_ast_candidates(
    tmp_path: Path,
) -> None:
    assert not hasattr(helper_detectors, "_autoregister_meta_rent_candidates")
    assert not hasattr(helper_detectors, "AutoRegisterFunctionReference")
    assert not hasattr(helper_detectors, "_autoregister_function_references")
    assert not hasattr(helper_detectors, "_autoregister_dynamic_factory_symbols")

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
        "    return Exporter.for_format(name)\n"
        "\n"
        "def materialize_nested(spec):\n"
        "    def build(name, body):\n"
        "        Exporter.for_format(name)\n"
        "        return AutoRegisterMeta(name, (Exporter,), body)\n"
        "    return build(*spec)\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = runtime_detectors.AutoRegisterMetaUnderRentedDetector.compact_module_projections(
        modules
    )

    candidates = runtime_detectors._compact_autoregister_meta_rent_candidates(
        projections,
        config,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.class_name == "Exporter"
    assert candidate.concrete_class_names == ("CsvExporter", "JsonExporter")
    assert candidate.behavior_method_names == ("emit",)
    assert candidate.abstract_method_names == ("emit",)
    detector = runtime_detectors.AutoRegisterMetaUnderRentedDetector()
    assert detector._candidate_items(list(modules), config) == candidates
    assert "_candidate_items" not in type(detector).__dict__
    assert "_findings_from_compact_projections" not in type(detector).__dict__
    assert "_findings_from_compact_context" not in type(detector).__dict__
    assert candidate.registry_projection_names == ("for_format",)
    assert candidate.missing_rent_signals == ("stable_key_axis",)


def test_compact_keyed_registry_axis_facts_preserve_axis_semantics(
    tmp_path: Path,
) -> None:
    assert not hasattr(
        systemic_detectors.DISPATCH_ALGEBRA_AUTHORITY,
        "keyed_registry_axis_fact_records",
    )
    assert not hasattr(base_detectors, "_keyed_type_registry_injectivity_proof")

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

    facts = systemic_detectors._compact_keyed_registry_axis_facts(
        projections,
        config,
    )

    assert len(facts) == 1
    fact = facts[0]
    assert fact.class_name == "Handler"
    assert fact.key_type_name == "Kind"
    assert fact.registry_key_attr_name == "kind"
    assert fact.lookup_method_names == ("for_kind", "type_for_kind")
    assert fact.registered_case_names == ("Kind.ALPHA",)
    assert fact.consumer_symbols == ("first", "second")
    assert fact.missing_maturity_signals == ("registered_case_axis",)
    assert fact.injectivity_proof.duplicate_key_names == ("Kind.ALPHA",)
    assert fact.injectivity_proof.missing_type_names == ("MissingKeyHandler",)


def test_compact_registry_projection_candidates_preserve_projection_semantics(
    tmp_path: Path,
) -> None:
    assert not hasattr(
        systemic_detectors._REGISTRY_PROJECTION_SURFACE_ANALYZER,
        "surface_candidates",
    )
    assert not hasattr(
        systemic_detectors._REGISTRY_PROJECTION_SURFACE_ANALYZER,
        "policy_authority_candidates",
    )

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
        "from core import (\n"
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

    candidates = (
        systemic_detectors._compact_registry_projection_surface_candidates_from_facts(
            projections,
            facts,
        )
    )
    policies = systemic_detectors._compact_registry_projection_policy_authority_candidates_from_facts(
        projections,
        facts,
    )

    assert {candidate.surface_name for candidate in candidates} == {
        "DUAL_MODE_SURFACE",
        "PUBLIC_MODE_CHOICES",
        "PUBLIC_MODE_TYPES",
    }
    assert {candidate.projection_role for candidate in candidates} == {"config_choices"}
    assert len(policies) == 1
    assert policies[0].policy_hint == "public"
    assert policies[0].surface_names == (
        "PUBLIC_MODE_CHOICES",
        "PUBLIC_MODE_TYPES",
    )


def test_registry_projection_role_ambiguity_fails_closed() -> None:
    evidence = systemic_detectors.RegistryProjectionSurfaceEvidence(
        surface_name="CLI_CONFIG_MODE_CHOICES",
        shared_key_names=("ALPHA", "BETA"),
        shared_type_names=(),
        has_key_to_type_pairs=False,
        has_type_to_key_pairs=False,
    )

    assert (
        systemic_detectors.RegistryProjectionRole.for_surface(
            evidence,
            file_path="pkg/options.py",
            default=systemic_detectors.RegistryProjectionRole.OPTION_ROSTER,
        )
        is None
    )


def test_keyed_registry_detectors_share_one_compact_repository_context(
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

    def counting_builder(projections, config, *, class_index=None):
        nonlocal calls
        calls += 1
        return original_builder(
            projections,
            config,
            class_index=class_index,
        )

    monkeypatch.setattr(
        systemic_detectors,
        "_compact_keyed_registry_axis_facts",
        counting_builder,
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def test_compact_class_detectors_share_one_repository_inheritance_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class Handler(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    def run(self): return 'alpha'\n"
        "\n"
        "class BetaHandler(Handler):\n"
        "    def run(self): return 'beta'\n",
        encoding="utf-8",
    )
    detector_types = (
        runtime_detectors.ManualConcreteSubclassRosterDetector,
        runtime_detectors.LatentImplementationRosterDetector,
        runtime_detectors.AutoRegisterMetaUnderRentedDetector,
        runtime_detectors.ExactTypeGuardInheritanceRetreatDetector,
        systemic_detectors.CrossModuleAxisShadowFamilyDetector,
        systemic_detectors.PrematureRegistryInfrastructureDetector,
        systemic_detectors.InheritedAutoRegisterConfigBoilerplateDetector,
        systemic_detectors.AutoRegisterExplicitPriorityOrderingDetector,
        surface_detectors.ManualFamilyRosterDetector,
    )
    calls = 0
    original_builder = base_detectors.build_compact_class_family_index

    def counting_builder(projections):
        nonlocal calls
        calls += 1
        return original_builder(projections)

    def forbidden_builder(_projections):
        raise AssertionError("detector rebuilt the shared compact class graph")

    monkeypatch.setattr(
        base_detectors,
        "build_compact_class_family_index",
        counting_builder,
    )
    for module in (
        helper_detectors,
        runtime_detectors,
        systemic_detectors,
    ):
        monkeypatch.setattr(
            module,
            "build_compact_class_family_index",
            forbidden_builder,
        )

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1


def test_multi_family_systemic_detectors_share_one_compact_class_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class Handler(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    def run(self): return 'alpha'\n"
        "\n"
        "class BetaHandler(Handler):\n"
        "    def run(self): return 'beta'\n",
        encoding="utf-8",
    )
    detector_types = (
        systemic_detectors.RepeatedConcreteTypeCaseAnalysisDetector,
        systemic_detectors.ImplicitSelfContractMixinDetector,
    )
    calls = 0
    original_builder = systemic_detectors.compact_class_index_from_projection_groups

    def counting_builder(projections_by_family, config):
        nonlocal calls
        calls += 1
        return original_builder(projections_by_family, config)

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_group_context_builder",
            staticmethod(counting_builder),
        )

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        detector_types,
        use_parse_cache=False,
    )
    findings = accumulator.findings_by_detector(DetectorConfig())

    assert calls == 1
    assert set(findings) == set(detector_types)


def test_bounded_multi_family_joins_reuse_the_single_class_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "family.py").write_text(
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class Handler(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    def run(self): return 'alpha'\n",
        encoding="utf-8",
    )
    detector_types = (
        runtime_detectors.ExactTypeGuardInheritanceRetreatDetector,
        systemic_detectors.RepeatedConcreteTypeCaseAnalysisDetector,
        semantic_descent_detectors.SemanticMirrorWithoutDescentDetector,
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    manifest = analysis_module.BoundedCompactProjectionManifest(detector_types)
    projections_by_family = {
        family: tuple(
            projection
            for module in modules
            for projection in collect_family_items(module, family)
        )
        for family in manifest.projection_families
    }
    monkeypatch.setattr(
        manifest,
        "projections_for_family",
        lambda family: projections_by_family[family],
    )

    calls = 0
    collect_cycles_calls: list[bool] = []
    original_builder = base_detectors.build_compact_class_family_index
    original_release = analysis_module.release_module_analysis_memory

    def counting_builder(projections):
        nonlocal calls
        calls += 1
        return original_builder(projections)

    def forbidden_builder(_projections):
        raise AssertionError("multi-family detector rebuilt the compact class graph")

    def observing_release(*, collect_cycles=True):
        collect_cycles_calls.append(collect_cycles)
        return original_release(collect_cycles=collect_cycles)

    monkeypatch.setattr(
        base_detectors,
        "build_compact_class_family_index",
        counting_builder,
    )
    for module in (
        runtime_detectors,
        semantic_descent_module,
        systemic_detectors,
    ):
        monkeypatch.setattr(
            module,
            "build_compact_class_family_index",
            forbidden_builder,
        )
    monkeypatch.setattr(
        analysis_module,
        "release_module_analysis_memory",
        observing_release,
    )

    findings = manifest.findings_by_detector(DetectorConfig())

    assert calls == 1
    assert set(findings) == set(detector_types)
    assert collect_cycles_calls[-1] is True


def test_compact_repeated_keyed_family_preserves_grouping_semantics(
    tmp_path: Path,
) -> None:
    assert not hasattr(base_detectors, "KeyedFamilyRootCandidate")
    assert not hasattr(base_detectors, "_repeated_keyed_family_candidates")

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

    candidates = systemic_detectors.RepeatedKeyedFamilyDetector._candidates_from_compact_projections(
        projections,
        config,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.family_base_name == "AutoRegisterByClassVar"
    assert candidate.lookup_style is class_index_module.RegistryLookupStyle.TRY_EXCEPT
    assert tuple(root.class_name for root in candidate.roots) == (
        "AlphaPolicy",
        "BetaPolicy",
        "GammaPolicy",
    )
    assert tuple(root.registry_key_attr_name for root in candidate.roots) == (
        "alpha",
        "beta",
        "gamma",
    )
    assert tuple(root.lookup_method_name for root in candidate.roots) == (
        "for_alpha",
        "for_beta",
        "for_gamma",
    )


def test_compact_concrete_family_candidates_preserve_semantics(
    tmp_path: Path,
) -> None:
    for deleted_shadow in (
        "_registered_type_match_assignment_shape",
        "_registered_type_list_assignment",
        "_registered_type_list_generator",
        "_registered_type_predicate_shape",
        "_is_selected_match_subscript",
        "_predicate_selected_concrete_family_candidates",
        "_mirrored_leaf_family_map",
        "_parallel_mirrored_leaf_family_candidates",
    ):
        assert not hasattr(helper_detectors, deleted_shadow)

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

    predicate_candidates = (
        runtime_detectors._compact_predicate_selected_concrete_family_candidates(
            context, config
        )
    )
    assert len(predicate_candidates) == 1
    predicate_candidate = predicate_candidates[0]
    assert predicate_candidate.class_name == "RenderRule"
    assert predicate_candidate.selector_method_name == "resolve"
    assert predicate_candidate.predicate_method_name == "matches_context"
    assert predicate_candidate.context_param_name == "artifact"
    assert predicate_candidate.concrete_class_names == (
        "AlphaRenderRule",
        "BetaRenderRule",
    )

    mirrored_candidates = (
        runtime_detectors._compact_parallel_mirrored_leaf_family_candidates(
            context, config
        )
    )
    assert len(mirrored_candidates) == 1
    mirrored_candidate = mirrored_candidates[0]
    assert mirrored_candidate.left.root_name == "InvoiceFieldEmitter"
    assert mirrored_candidate.right.root_name == "ReceiptFieldEmitter"
    assert mirrored_candidate.contract_method_names == ("emit",)
    assert mirrored_candidate.shared_leaf_family_names == (
        "alpha emitter",
        "beta emitter",
        "gamma emitter",
    )


def test_compact_roster_candidates_preserve_semantics(
    tmp_path: Path,
) -> None:
    assert not hasattr(class_index_module, "CompactLatentRosterObservation")
    assert not hasattr(helper_detectors, "_LatentRosterObservation")
    assert not hasattr(helper_detectors, "LatentRosterProjectionAuthority")
    assert not hasattr(helper_detectors, "LATENT_ROSTER_PROJECTION_AUTHORITY")
    assert not hasattr(base_detectors, "_ManualSubclassRegistrationSite")
    for deleted_shadow in (
        "_class_list_registry_names",
        "_registration_append_registry_name",
        "_looks_like_cls_registration_value",
        "_class_dict_get_attr_name",
        "_guarded_defined_attr_name",
        "_guard_requires_concrete_subclass",
        "_manual_subclass_registration_sites",
        "_uses_named_registry",
        "_registry_consumer_locations",
        "_registered_descendant_classes",
        "_manual_concrete_subclass_roster_candidates",
        "_family_roster_member",
        "_extract_family_roster_members",
        "_best_shared_family_base_name",
        "_manual_family_roster_candidates",
    ):
        assert not hasattr(helper_detectors, deleted_shadow)

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

    manual_candidates = (
        runtime_detectors._compact_manual_concrete_subclass_roster_candidates(
            context, config
        )
    )
    assert len(manual_candidates) == 1
    manual_candidate = manual_candidates[0]
    assert manual_candidate.class_name == "RoutedRequest"
    assert manual_candidate.registry_name == "_registered_types"
    assert manual_candidate.guard_summary == (
        "cls.__dict__.get('route_name') is not None"
    )
    assert manual_candidate.registration_site.selector_attr_name == "route_name"
    assert not manual_candidate.registration_site.requires_concrete_subclass
    assert manual_candidate.consumer_names == ("RoutedRequest.concrete_types",)
    assert manual_candidate.concrete_class_names == (
        "DirectRequest",
        "GuidedRequest",
    )
    latent_candidates = (
        runtime_detectors._compact_latent_implementation_roster_candidates(
            context, config
        )
    )
    assert {
        (
            candidate.class_name,
            candidate.roster.roster_name,
            candidate.key_attr_name,
            candidate.match.coverage_ratio,
        )
        for candidate in latent_candidates
    } == {
        ("Exporter", "EXPORT_FORMATS", "format", 1.0),
        ("Exporter", "DEFAULT_EXPORTERS", None, 1.0),
    }
    manual_family_candidates = (
        surface_detectors._compact_manual_family_roster_candidates(context)
    )
    assert len(manual_family_candidates) == 1
    manual_family_candidate = manual_family_candidates[0]
    assert manual_family_candidate.owner_name == "DEFAULT_EXPORTERS"
    assert manual_family_candidate.member_names == ("CsvExporter", "JsonExporter")
    assert manual_family_candidate.family_base_name == "Exporter"
    assert manual_family_candidate.constructor_style == "constructor_call"
    assert tuple(
        (Path(location.file_path).name, location.symbol)
        for location in manual_family_candidate.member_locations
    ) == (
        ("implementations.py", "CsvExporter"),
        ("implementations.py", "JsonExporter"),
    )
    assert (
        surface_detectors.ManualFamilyRosterDetector()._candidate_items(
            list(modules), config
        )
        == manual_family_candidates
    )


def test_concrete_family_detectors_share_one_compact_graph_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not hasattr(base_detectors, "compact_class_repository_context")
    assert not hasattr(base_detectors, "require_compact_class_repository_context")
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text("class Root: pass\n", encoding="utf-8")
    detector_types = (
        runtime_detectors.ManualConcreteSubclassRosterDetector,
        runtime_detectors.LatentImplementationRosterDetector,
        runtime_detectors.PredicateSelectedConcreteFamilyDetector,
        runtime_detectors.ParallelMirroredLeafFamilyDetector,
        surface_detectors.ManualFamilyRosterDetector,
        runtime_detectors.AutoRegisterMetaUnderRentedDetector,
        runtime_detectors.ExactTypeGuardInheritanceRetreatDetector,
    )
    assert {
        detector_type.compact_shared_context_builder for detector_type in detector_types
    } == {runtime_detectors.CompactClassRepositoryContext.from_projections}

    repository_calls = 0
    concrete_context_calls = 0
    original_repository_builder = (
        runtime_detectors.CompactClassRepositoryContext.from_projections
    )
    original_concrete_context_builder = (
        runtime_detectors._compact_concrete_family_context
    )

    def counting_repository_builder(projections, config):
        nonlocal repository_calls
        repository_calls += 1
        return original_repository_builder(projections, config)

    def counting_concrete_context_builder(projections, config, *, class_index=None):
        nonlocal concrete_context_calls
        concrete_context_calls += 1
        return original_concrete_context_builder(
            projections,
            config,
            class_index=class_index,
        )

    for detector_type in detector_types:
        monkeypatch.setattr(
            detector_type,
            "compact_shared_context_builder",
            staticmethod(counting_repository_builder),
        )
    monkeypatch.setattr(
        runtime_detectors,
        "_compact_concrete_family_context",
        counting_concrete_context_builder,
    )
    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,), detector_types, use_parse_cache=False
    )

    accumulator.findings_by_detector(DetectorConfig())

    assert repository_calls == 1
    assert concrete_context_calls == 1


def test_compact_pass_through_nominal_wrapper_preserves_semantics(
    tmp_path: Path,
) -> None:
    for deleted_shadow in (
        "_normalized_authority_name",
        "_is_self_delegate_attribute",
        "_forwarded_delegate_property_name",
        "_forwarded_delegate_call",
        "_call_forwards_parameters",
        "_forwarded_delegate_member_name",
        "_pass_through_nominal_wrapper_candidates_for_class",
        "_pass_through_nominal_wrapper_candidates",
    ):
        assert not hasattr(helper_detectors, deleted_shadow)

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
    projections = (
        surface_detectors.PassThroughNominalWrapperDetector.compact_module_projections(
            modules
        )
    )
    compact_wrapper_candidates = (
        surface_detectors._compact_pass_through_nominal_wrapper_candidates(projections)
    )
    assert len(compact_wrapper_candidates) == 1
    wrapper_candidate = compact_wrapper_candidates[0]
    assert wrapper_candidate.class_name == "JobSpecWrapper"
    assert wrapper_candidate.delegate_field_name == "delegate"
    assert wrapper_candidate.delegate_authority_name == "JobSpecBase"
    assert wrapper_candidate.forwarded_member_names == ("start", "stop")
    wrapper_detector = surface_detectors.PassThroughNominalWrapperDetector()
    assert wrapper_detector._findings_from_compact_projections(projections, config) == [
        wrapper_detector._finding_for_candidate(candidate)
        for candidate in compact_wrapper_candidates
    ]


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


def test_compact_abc_optimizer_candidates_preserve_semantics_without_ast_shadow(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_abc_optimizer_fixture(package_root)
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    config = DetectorConfig()
    projections = structural_detectors.SemanticOverlapAbcOptimizationDetector.compact_module_projections(
        modules
    )
    context = structural_detectors.CompactABCOptimizerContext.from_projections(
        projections
    )
    detector_candidate_pairs = (
        (
            structural_detectors.SemanticOverlapAbcOptimizationDetector,
            context.method_candidates,
        ),
        (
            structural_detectors.SemanticOverlapAbcFamilyOptimizationDetector,
            context.family_candidates,
        ),
        (
            structural_detectors.GlobalInheritanceOptimizationDetector,
            context.global_candidates,
        ),
        (
            structural_detectors.SemanticOverlapAbcResidueAxisCatalogDetector,
            context.residue_axis_candidates,
        ),
    )

    for detector_type, compact_candidates in detector_candidate_pairs:
        detector = detector_type()
        assert detector._candidate_items(list(modules), config) == compact_candidates
        assert detector._findings_from_compact_context(
            projections, context, config
        ) == detector._findings_for_candidates(compact_candidates, config)
        assert "candidate_collector" not in detector_type.__dict__
    assert tuple(candidate.method_name for candidate in context.method_candidates) == (
        "emit",
        "validate",
        "audit",
        "cache",
    )
    assert context.family_candidates[0].method_names == ("emit", "validate")
    assert context.global_candidates[0].method_names == (
        "audit",
        "cache",
        "emit",
        "validate",
    )
    assert context.residue_axis_candidates[0].residue_kind_names == (
        "call",
        "constant",
    )
    for removed_name in (
        "_semantic_overlap_abc_optimization_candidates",
        "_semantic_overlap_abc_optimization_candidates_from_modules",
        "_semantic_overlap_abc_family_optimization_candidates",
        "_semantic_overlap_global_inheritance_candidates",
        "_semantic_overlap_abc_residue_axis_catalog_candidates",
        "_abc_optimizer_specific_method_plans",
        "_abc_optimizer_candidates_from_family_plans",
        "_compact_abc_optimizer_context",
        "ABCOptimizerAuthority",
        "ABC_OPTIMIZER_AUTHORITY",
        "_ABCSemanticSkeletonNormalizer",
        "_ABCOptimizerFamilyCandidateOrder",
        "_abc_optimizer_statement_skeleton",
        "_semantic_overlap_coordinates",
    ):
        assert not hasattr(helper_detectors, removed_name)
    assert not hasattr(
        structural_detectors._CompactABCOptimizerDetectorBase,
        "compact_candidate_attribute",
    )
    assert not hasattr(class_index_module, "_COMPACT_ABC_OPTIMIZER_IGNORED_BASE_NAMES")
    assert not hasattr(helper_detectors, "_ABC_OPTIMIZER_IGNORED_BASE_NAMES")
    assert not hasattr(class_index_module.CompactABCOptimizerMethod, "skeleton_blob")
    assert not hasattr(class_index_module.CompactABCOptimizerMethod, "coordinates_blob")


def test_compact_abc_optimizer_profiles_are_derived_after_the_family_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_abc_optimizer_fixture(package_root)
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    projection_family = class_index_module.CompactModuleClassProjectionFamily
    projections = tuple(
        projection
        for module in modules
        for projection in projection_family.collect(module)
    )
    profile_derivations: list[tuple[str, ...]] = []
    original_derivation = (
        class_index_module._compact_abc_optimizer_statements_from_sources
    )

    def recording_derivation(statement_sources):
        profile_derivations.append(statement_sources)
        return original_derivation(statement_sources)

    monkeypatch.setattr(
        class_index_module,
        "_compact_abc_optimizer_statements_from_sources",
        recording_derivation,
    )

    assert profile_derivations == []
    context = structural_detectors.CompactABCOptimizerContext.from_projections(
        projections
    )

    assert tuple(candidate.method_name for candidate in context.method_candidates) == (
        "emit",
        "validate",
        "audit",
        "cache",
    )
    assert len(profile_derivations) == 10
    assert all(len(statement_sources) == 4 for statement_sources in profile_derivations)
    assert class_index_module.ClassSymbolResolutionAuthority.establishes_nominal_family(
        "domain.Worker"
    )
    assert not class_index_module.ClassSymbolResolutionAuthority.establishes_nominal_family(
        "abc.ABC"
    )


def test_abc_optimizer_detectors_share_one_compact_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "pkg"
    _write_compact_abc_optimizer_fixture(package_root)
    detector_types = (
        structural_detectors.SemanticOverlapAbcOptimizationDetector,
        structural_detectors.SemanticOverlapAbcFamilyOptimizationDetector,
        structural_detectors.GlobalInheritanceOptimizationDetector,
        structural_detectors.SemanticOverlapAbcResidueAxisCatalogDetector,
    )
    calls = 0
    original_builder = structural_detectors.CompactABCOptimizerContext.from_projections

    def counting_builder(projections, config):
        nonlocal calls
        del config
        calls += 1
        return original_builder(projections)

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


def test_compact_algebraic_variant_candidates_own_global_analysis(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
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
    variant_detector = runtime_detectors.AlgebraicVariantMethodFamilyDetector()
    projections = variant_detector.compact_module_projections(modules)
    source_module = SourceModule(
        path=modules[0].path,
        module_name=modules[0].module_name,
        source=modules[0].source,
    )
    native_projections = (
        runtime_detectors.CompactAlgebraicVariantModuleProjectionFamily.collect_source(
            source_module,
            NativePythonSyntaxIndex.from_source(source_module.source),
        )
    )
    assert tuple(native_projections or ()) == projections

    compact_variants = variant_detector._candidates_from_compact_projections(
        projections,
        config,
    )

    assert len(compact_variants) == 1
    assert variant_detector._candidate_items(list(modules), config) == compact_variants
    assert "candidate_collector" not in type(variant_detector).__dict__
    for removed_name in (
        "ABCPolymorphismBypassedByConcreteDispatchDetector",
        "CompactNominalBypassProjectionDemand",
        "_isinstance_family_scatter_candidates",
        "_cross_class_small_method_template_candidates",
        "CancelableCompositionSignalQuery",
        "_nominal_authority_bypass_candidates",
        "_variant_method_family_candidates",
    ):
        assert not hasattr(runtime_detectors, removed_name)
    assert variant_detector._findings_from_compact_projections(
        projections,
        config,
    ) == [
        variant_detector._finding_for_candidate(candidate)
        for candidate in compact_variants
    ]

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (type(variant_detector),),
        use_parse_cache=False,
    )
    findings = accumulator.findings_by_detector(config)
    assert accumulator.projection_count == 1
    assert len(findings[type(variant_detector)]) == 1


def test_compact_semantic_descent_graph_matches_legacy_ast_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        "    def to_dict(self):\n"
        "        return {'title': self.title, 'status': self.status}\n\n"
        "REQUEST_FIELDS = ('title', 'status')\n"
        "REQUEST_COLUMNS = ('title', 'status')\n",
        encoding="utf-8",
    )
    modules = tuple(parse_python_modules(package_root, use_parse_cache=False))
    detector = semantic_descent_detectors.SemanticMirrorWithoutDescentDetector()
    groups = type(detector).compact_module_projection_groups(modules)
    semantic_projections = groups[CompactSemanticModuleProjectionFamily]
    class_projections = groups[runtime_detectors.CompactModuleClassProjectionFamily]
    legacy_supplements = tuple(
        supplement
        for module in modules
        for qualname, node in semantic_descent_module._semantic_indexed_class_nodes(
            list(module.module.body)
        )
        if (
            supplement := semantic_descent_module._semantic_class_supplement(
                f"{module.module_name}.{qualname}",
                node,
            )
        )
        is not None
    )

    assert (
        tuple(
            supplement
            for projection in semantic_projections
            for supplement in projection.class_supplements
        )
        == legacy_supplements
    )

    legacy_graph = build_semantic_descent_graph(list(modules), use_cache=False)
    compact_graph = build_compact_semantic_descent_graph(
        semantic_projections,
        class_projections,
    )

    original_projection_objects = {
        id(projection)
        for module_projection in semantic_projections
        for projection in module_projection.projections
    }
    assert any(
        id(projection) in original_projection_objects
        for projection in compact_graph.projections
    )
    assert compact_graph.facts
    assert compact_graph.facts[0].normalized_aliases
    assert "normalized_aliases" not in vars(compact_graph.facts[0])
    assert all(
        "owner" not in vars(projection) for projection in compact_graph.projections
    )
    fact_reference_by_id: dict[str, object] = {}
    reused_fact_reference = False
    for edge in compact_graph.missing_descent_relations:
        for fact_reference in edge.match.fact_refs:
            previous = fact_reference_by_id.get(fact_reference.fact_id)
            if previous is None:
                fact_reference_by_id[fact_reference.fact_id] = fact_reference
                continue
            reused_fact_reference = True
            assert previous is fact_reference
    assert reused_fact_reference
    assert compact_graph.authorities == legacy_graph.authorities
    assert compact_graph.facts == legacy_graph.facts
    assert compact_graph.projections == legacy_graph.projections
    assert compact_graph.relations == legacy_graph.relations
    assert compact_graph.certificates == legacy_graph.certificates
    assert len(compact_graph.certificates) > len(
        compact_graph.missing_descent_certificates
    )
    config = DetectorConfig()
    original_resolution = (
        semantic_descent_detectors.build_compact_semantic_mirror_resolution
    )
    released_edge_refs: list[weakref.ReferenceType[object]] = []

    def tracked_resolution(*args, **kwargs):
        graph_space, resolution = original_resolution(*args, **kwargs)
        released_edge_refs.append(
            weakref.ref(resolution.relations[0].missing_descent_relations()[0])
        )
        return graph_space, resolution

    monkeypatch.setattr(
        semantic_descent_detectors,
        "build_compact_semantic_mirror_resolution",
        tracked_resolution,
    )
    expected_findings = detector._collect_findings_from_graph(
        legacy_graph,
        list(modules),
        config,
    )
    assert (
        detector._findings_from_compact_projection_groups(
            groups,
            config,
        )
        == expected_findings
    )
    assert released_edge_refs
    assert all(edge_ref() is None for edge_ref in released_edge_refs)

    class_index = base_detectors.compact_class_index_from_projection_groups(
        groups,
        config,
    )
    finding_stream = detector._stream_findings_from_compact_projection_groups_context(
        groups,
        class_index,
        config,
    )
    chunks = tuple(finding_stream.chunks)
    assert finding_stream.finding_count == len(expected_findings)
    assert all(
        0 < len(chunk) <= detector.compact_finding_chunk_size for chunk in chunks
    )
    assert [finding for chunk in chunks for finding in chunk] == expected_findings
    assert all(edge_ref() is None for edge_ref in released_edge_refs)

    accumulator = accumulate_compact_global_projections_for_roots(
        (package_root,),
        (type(detector),),
        use_parse_cache=False,
    )
    assert accumulator.projection_count == 2
    assert accumulator.findings_by_detector(config)[type(detector)] == expected_findings

    def unexpected_retained_findings(*args, **kwargs):
        del args, kwargs
        raise AssertionError("eager exact consumption must use the finding stream")

    monkeypatch.setattr(
        semantic_descent_detectors.SemanticMirrorWithoutDescentDetector,
        "_findings_from_compact_projection_groups_context",
        unexpected_retained_findings,
    )
    consumed_streams: list[object] = []
    analysis_module._compact_findings_by_detector(
        (type(detector),),
        groups,
        config,
        finding_consumer=lambda _detector_type, findings: consumed_streams.append(
            findings
        ),
        retain_findings=False,
    )
    assert len(consumed_streams) == 1
    assert isinstance(consumed_streams[0], base_detectors.CompactFindingStream)
    assert list(consumed_streams[0]) == expected_findings


def test_compact_analysis_returns_semantic_graph_on_cold_and_aggregate_hits(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "mod.py").write_text(
        "class Handler:\n"
        "    pass\n\n"
        "class AlphaHandler(Handler):\n"
        "    handler_id = 'alpha'\n\n"
        "class BetaHandler(Handler):\n"
        "    handler_id = 'beta'\n\n"
        "HANDLERS = {'alpha': AlphaHandler, 'beta': BetaHandler}\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    analysis_cache_dir = tmp_path / "analysis"
    detector_types = (semantic_descent_detectors.SemanticMirrorWithoutDescentDetector,)

    cold = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=detector_types,
        include_semantic_descent_graph=True,
    )
    warm = analyze_compact_roots_with_cache(
        (package_root,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=detector_types,
        include_semantic_descent_graph=True,
    )

    assert cold.semantic_descent_graph is not None
    assert cold.semantic_descent_graph.certificates
    assert warm.cache_status is AnalysisCacheStatus.HIT
    assert warm.semantic_descent_graph == cold.semantic_descent_graph
    assert warm.findings == cold.findings


def test_global_projection_partition_tracks_migrated_detector_boundary() -> None:
    partition = DetectorTypePartition.from_detector_types(
        default_detector_types_for_analysis()
    )

    assert (
        runtime_detectors.GeneratedBoundarySemanticConstantMirrorDetector
        in partition.compact_global_detector_types
    )
    assert runtime_detectors.RepeatedBuilderCallDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ManualClassRegistrationDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.InheritedAutoRegisterConfigBoilerplateDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.AutoRegisterExplicitPriorityOrderingDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.NominalInstanceExplicitOrderingDetector in (
        partition.compact_global_detector_types
    )
    assert structural_detectors.SupportPreludeModuleFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert environment_detectors.EnvironmentBooleanAuthorityDriftDetector in (
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
    assert systemic_detectors.DataclassNamespaceCliMirrorDetector in (
        partition.compact_global_detector_types
    )
    assert runtime_detectors.ExactTypeGuardInheritanceRetreatDetector in (
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
    assert surface_detectors.PassThroughNominalWrapperDetector in (
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
    assert runtime_detectors.AlgebraicVariantMethodFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert semantic_descent_detectors.SemanticMirrorWithoutDescentDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.CrossModuleSpecAxisAuthorityDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RepeatedValidateShapeGuardFamilyDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.RepeatedConcreteTypeCaseAnalysisDetector in (
        partition.compact_global_detector_types
    )
    assert systemic_detectors.ImplicitSelfContractMixinDetector in (
        partition.compact_global_detector_types
    )
    assert len(partition.ast_retaining_context_detector_types) == 0
    assert all(
        detector_type.detector_id
        not in {"simple_property_alias_class", "simple_property_alias_method"}
        for detector_type in partition.per_module_detector_types
    )


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


def test_single_source_parser_rejects_stale_semantic_hash_identity(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module_path = package_root / "mod.py"
    original_source = "VALUE = 1\n"
    changed_source = "VALUE = 2\n"
    module_path.write_text(original_source, encoding="utf-8")
    source_semantic_hash = ast_tools_module.PythonSourceSemanticHash(
        ast_tools_module.python_source_cache_signature(original_source),
        ast_tools_module.semantic_python_source_hash(original_source),
    )
    module_path.write_text(changed_source, encoding="utf-8")
    parser = ast_tools_module.PythonModuleRootParser.for_root(
        package_root,
        use_parse_cache=False,
    )

    parsed_module = parser.parsed_source_path(
        module_path,
        source_semantic_hash=source_semantic_hash,
    )

    assert parsed_module.semantic_hash == (
        ast_tools_module.semantic_python_source_hash(changed_source)
    )
    assert parsed_module.semantic_hash != source_semantic_hash.semantic_hash


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


def test_source_signature_cache_reuses_lazy_semantic_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "mod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    cache = AnalysisFindingCache(tmp_path / "analysis")
    source_cache = cache.source_signature_cache()
    assert source_cache is not None
    source_cache.source_file_signatures((module_path,))
    expected_hash = source_cache.semantic_source_hash(module_path)
    source_cache.store_if_dirty()

    def unexpected_semantic_hash(_source: str) -> str:
        raise AssertionError("unchanged semantic source should not be rehashed")

    monkeypatch.setattr(
        analysis_cache_module,
        "semantic_python_source_hash",
        unexpected_semantic_hash,
    )
    warm_source_cache = cache.source_signature_cache()
    assert warm_source_cache is not None

    assert warm_source_cache.semantic_source_hash(module_path) == expected_hash


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
        "class Step:\n    pass\n",
        encoding="utf-8",
    )
    (package_root / "members.py").write_text(
        "class LoadStep(Step):\n    step_id = 'load'\n",
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
        "class LoadStep(Step):\n    step_id = 'load'\n",
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
            evidence.file_path == members_path.as_posix()
            for evidence in finding.evidence
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
        "class LoadStep(Step):\n    step_id = 'load'\n",
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
            evidence.file_path == registry_path.as_posix()
            for evidence in finding.evidence
        )
        for finding in exact_findings
    )
