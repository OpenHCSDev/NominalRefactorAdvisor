from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    analyze_path,
)
from nominal_refactor_advisor.analysis_cache import (
    AnalysisCacheIdentity,
    AnalysisCacheStatus,
    AnalysisFindingCache,
)
from nominal_refactor_advisor.ast_tools import (
    ExportDictShapeFamily,
    collect_family_items,
    parse_python_module_roots,
    parse_python_modules,
)
from nominal_refactor_advisor.cache_paths import (
    analysis_cache_sibling,
    default_parse_cache_dir,
    semantic_descent_cache_sibling,
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


class CountingSemanticCacheDetector(IssueDetector):
    call_count = 0

    def _collect_findings(
        self, modules: list, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del modules, config
        type(self).call_count += 1
        return []


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
    assert candidate_calls == 3
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


def test_collected_family_items_are_persisted_beside_parse_cache(tmp_path: Path) -> None:
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
        "class AvoidWidgetsWindow:\n"
        "    pass\n",
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
        "def use(value):\n"
        "    return value\n",
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
