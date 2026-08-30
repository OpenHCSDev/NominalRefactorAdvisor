from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nominal_refactor_advisor.semantic_descent as semantic_descent_module
from nominal_refactor_advisor.detectors import (
    _semantic_descent as semantic_descent_detectors,
)
from nominal_refactor_advisor.analysis import analyze_modules, analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceContext,
    CodemodSourceSnapshot,
    NominalBoundaryConcept,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.detectors import (
    DetectorCacheGranularity,
    DetectorConfig,
    DerivedMetricCountBoilerplateDetector,
    InheritedAutoRegisterConfigBoilerplateDetector,
    IssueDetector,
    SemanticMirrorWithoutDescentDetector,
)
from nominal_refactor_advisor.models import (
    MappingMetrics,
    RefactorFinding,
    SourceLocation,
)
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import (
    AuthorityProofEdgeKind,
    ConstructionAuthorityResolver,
    PresentationAuthorityConstruction,
    PresentationTokenRole,
    SemanticAuthorityKind,
    SemanticDerivationCertificate,
    SemanticMirrorResolver,
    PresentationProjectionKind,
    SemanticAuthorityMirrorPolicy,
    SemanticDescentGraphModuleOverlay,
    SemanticDescentGraphCacheIdentity,
    build_finding_backed_semantic_descent_graph,
    build_semantic_descent_graph,
    rebase_semantic_descent_graph,
    semantic_descent_finding_projection_id,
)
from nominal_refactor_advisor.semantic_refactor_gate import SemanticRefactorGateWorkItem


def _write_module(root: Path, source: str) -> Path:
    path = root / "pkg" / "mod.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def _goal_policy_finding(detector_id: str) -> RefactorFinding:
    return RefactorFinding(
        detector_id=detector_id,
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Finding",
        summary=f"{detector_id} summary",
        why="semantic fact is mirrored outside its nominal authority",
        capability_gap="derive the projection from the authority instead",
        relation_context="projection lacks a semantic-descent certificate",
        evidence=(SourceLocation("pkg/mod.py", 1, detector_id),),
    )


def test_class_name_algebra_uses_tokens_for_common_suffixes() -> None:
    assert (
        CLASS_NAME_ALGEBRA.longest_common_suffix(("AlphaResult", "BetaResult"))
        == "Result"
    )


def test_semantic_authority_mirror_policy_registry_covers_authority_kinds() -> None:
    assert SemanticAuthorityMirrorPolicy.registered_authority_kinds() == frozenset(
        SemanticAuthorityKind
    )


def test_semantic_authority_selection_is_one_mro_owned_fallback_chain() -> None:
    provider_types = (
        semantic_descent_module.SemanticAuthorityProvider,
        semantic_descent_module.DataclassSemanticAuthorityProvider,
        semantic_descent_module.ClassFamilySemanticAuthorityProvider,
    )

    assert (
        semantic_descent_module.SemanticAuthorityProvider.__mro__[:3] == provider_types
    )
    assert not hasattr(
        semantic_descent_module,
        "CompactSemanticAuthorityBuilder",
    )
    assert all(
        not hasattr(provider_type, "provider_order")
        and not hasattr(provider_type, "__registry__")
        for provider_type in provider_types
    )
    assert (
        semantic_descent_module.SemanticAuthorityBuildContext.__abstractmethods__
        == {
            "aliases_for",
            "supplement_for",
        }
    )
    assert issubclass(
        semantic_descent_module.AstSemanticAuthorityBuildContext,
        semantic_descent_module.SemanticAuthorityBuildContext,
    )
    assert issubclass(
        semantic_descent_module.CompactSemanticAuthorityBuildContext,
        semantic_descent_module.SemanticAuthorityBuildContext,
    )


def test_class_key_resolution_is_one_mro_owned_fallback_chain() -> None:
    resolver_type = semantic_descent_detectors.SemanticMirrorClassKeySourceResolver

    assert resolver_type.__mro__[:2] == (
        resolver_type,
        semantic_descent_detectors.AliasOverlapClassKeySourceResolver,
    )
    assert not hasattr(
        semantic_descent_detectors,
        "DictProjectionClassKeySourceResolver",
    )
    assert not hasattr(resolver_type, "resolver_order")
    assert not hasattr(resolver_type, "ordered_resolvers")
    assert not hasattr(resolver_type, "__registry__")


def test_fact_token_index_reuses_facts_in_deterministic_reference_order() -> None:
    later_fact = semantic_descent_module.SemanticFact(
        fact_id="beta:item",
        authority_id="beta",
        kind=semantic_descent_module.SemanticFactKind.DATACLASS_FIELD,
        name="item",
        aliases=("shared_item",),
        location=SourceLocation("pkg/beta.py", 2, "Beta.item"),
    )
    earlier_fact = semantic_descent_module.SemanticFact(
        fact_id="alpha:item",
        authority_id="alpha",
        kind=semantic_descent_module.SemanticFactKind.DATACLASS_FIELD,
        name="item",
        aliases=("shared_item",),
        location=SourceLocation("pkg/alpha.py", 2, "Alpha.item"),
    )

    refs = semantic_descent_module.SemanticFactTokenIndex(
        (later_fact, earlier_fact)
    ).refs_for_token("shared_item")

    assert refs == (earlier_fact, later_fact)
    assert refs[0] is earlier_fact
    assert refs[1] is later_fact
    specificity = semantic_descent_module.SemanticFactSpecificityIndex(
        (later_fact, earlier_fact)
    )
    assert specificity.matched_facts_are_reused_roles((earlier_fact, later_fact))
    assert "authority_ids_by_fact_name" not in vars(specificity)


def test_construction_authority_materialization_indexes_each_method_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class AlphaState:\n"
        "    value: int\n"
        "    count: int\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BetaState:\n"
        "    value: int\n"
        "    count: int\n"
        "\n"
        "class StateFactory:\n"
        "    def state_build(self):\n"
        "        return AlphaState(1, 2)\n"
        "\n"
        "class NoiseFactory:\n"
        "    def noise_build(self):\n"
        "        return BetaState(1, 2)\n",
    )
    graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path),
        use_cache=False,
    )
    assert graph.class_index is not None
    alpha_authority = next(
        authority for authority in graph.authorities if authority.name == "AlphaState"
    )
    beta_authority = next(
        authority for authority in graph.authorities if authority.name == "BetaState"
    )
    walked_method_names: list[str] = []
    original_walk = semantic_descent_module.ast.walk

    def counted_walk(node):
        if isinstance(node, semantic_descent_module.ast.FunctionDef):
            walked_method_names.append(node.name)
        return original_walk(node)

    monkeypatch.setattr(semantic_descent_module.ast, "walk", counted_walk)
    resolver = ConstructionAuthorityResolver(
        class_index=graph.class_index,
        authorities=graph.authorities,
        projection_construction_type_names=frozenset({"StateFactory"}),
    )
    construction = PresentationAuthorityConstruction("StateFactory", ())

    for _ in range(5):
        assert resolver.construction_type_materializes_authority(
            construction,
            alpha_authority,
        )
        assert not resolver.construction_type_materializes_authority(
            construction,
            beta_authority,
        )

    assert walked_method_names == ["state_build"]
    missing_constructed_ids = resolver.authority_ids_for_constructed_type_name(
        "MissingState"
    )
    assert not missing_constructed_ids
    assert (
        resolver.authority_ids_for_constructed_type_name("MissingState")
        is missing_constructed_ids
    )
    assert "MissingState" not in resolver.authority_ids_by_constructed_type_name
    assert "MissingState" in resolver.resolved_constructed_type_names

    missing_materialized_ids = (
        resolver.materialized_authority_ids_for_construction_type("MissingFactory")
    )
    assert not missing_materialized_ids
    assert (
        resolver.materialized_authority_ids_for_construction_type("MissingFactory")
        is missing_materialized_ids
    )
    assert (
        "MissingFactory" not in resolver.materialized_authority_ids_by_construction_type
    )
    assert "MissingFactory" in resolver.resolved_additional_materialization_type_names
    assert "compact_authority_ids_by_name" in vars(resolver)
    assert "authority_ids_by_name" not in vars(resolver)


def test_constructed_dataclass_inverse_index_matches_pairwise_resolution(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class AlphaState:\n"
        "    value: int\n"
        "    count: int\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BetaState:\n"
        "    name: str\n"
        "    enabled: bool\n"
        "\n"
        "class AlphaFactory:\n"
        "    def build(self):\n"
        "        return AlphaState(value=1, count=2)\n"
        "\n"
        "ROWS = (\n"
        "    AlphaState(value=1, count=2),\n"
        "    AlphaFactory(value=1, count=2),\n"
        "    BetaState(name='beta', enabled=True),\n"
        ")\n",
    )
    graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path),
        use_cache=False,
    )
    assert graph.class_index is not None
    resolver = SemanticMirrorResolver(
        authorities=graph.authorities,
        facts=graph.facts,
        projections=graph.projections,
        class_index=graph.class_index,
    )
    descent = resolver.dataclass_descent
    assert descent.projection_descent_authority_ids == {}
    assert descent.dataclass_fact_tokens_by_authority_id == {}
    first_dataclass_authority = descent.dataclass_authorities[0]
    expected_first_tokens = frozenset(
        variant
        for fact in resolver.fact_authority_index.facts_for_authority(
            first_dataclass_authority.authority_id
        )
        for variant in semantic_descent_module.normalized_name_variants(fact.name)
    )
    assert (
        descent.fact_tokens_for_authority(first_dataclass_authority.authority_id)
        == expected_first_tokens
    )
    assert tuple(descent.dataclass_fact_tokens_by_authority_id) == (
        first_dataclass_authority.authority_id,
    )
    first_projection = resolver.projections[0]
    descent.descent_authority_ids_for_projection(first_projection)
    assert tuple(descent.projection_descent_authority_ids) == (
        first_projection.projection_id,
    )

    for projection in resolver.projections:
        for construction in projection.owner_constructions:
            pairwise_ids = frozenset(
                authority.authority_id
                for authority in resolver.authorities
                if (
                    resolver.construction_resolver.construction_type_is_authority_class(
                        construction, authority
                    )
                    or resolver.construction_resolver.construction_type_materializes_authority(
                        construction, authority
                    )
                )
            )
            assert (
                resolver.construction_resolver.descended_authority_ids_for_construction_type(
                    construction.type_name
                )
                == pairwise_ids
            )

        pairwise_constructed_ids = frozenset(
            authority.authority_id
            for authority in resolver.dataclass_descent.dataclass_authorities
            if resolver.dataclass_descent.projection_owner_constructs_dataclass_authority(
                projection,
                authority,
                resolver.fact_authority_index.facts_for_authority(
                    authority.authority_id
                ),
            )
        )
        assert (
            resolver.dataclass_descent.constructed_dataclass_authority_ids(projection)
            == pairwise_constructed_ids
        )

    qualified_token = semantic_descent_module.PresentationToken(
        value="value",
        kind=semantic_descent_module.PresentationTokenKind.QUALIFIED_ATTRIBUTE,
        role=PresentationTokenRole.DICT_VALUE,
        qualifier="external",
    )
    resolver._candidate_refs_for_token(qualified_token)
    assert tuple(resolver.candidate_refs_by_token) == (qualified_token,)
    assert next(iter(resolver.candidate_refs_by_token)) is qualified_token

    resolver.resolve()
    retained_token_ids = {
        id(token) for projection in resolver.projections for token in projection.tokens
    } | {id(qualified_token)}
    assert all(
        id(token) in retained_token_ids for token in resolver.candidate_refs_by_token
    )
    assert "candidate_refs_by_token_signature" not in vars(resolver)


def test_semantic_descent_graph_flags_manual_class_family_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC\n"
        "\n"
        "class Handler(ABC):\n"
        "    pass\n"
        "\n"
        "class AlphaHandler(Handler):\n"
        "    handler_id = 'alpha'\n"
        "\n"
        "class BetaHandler(Handler):\n"
        "    handler_id = 'beta'\n"
        "\n"
        "HANDLERS = {'alpha': AlphaHandler, 'beta': BetaHandler}\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    handler_authority = next(
        authority for authority in graph.authorities if authority.name == "Handler"
    )
    assert handler_authority.kind is SemanticAuthorityKind.CLASS_FAMILY

    certificate = next(
        item
        for item in graph.missing_descent_certificates
        if item.edge.authority_id == handler_authority.authority_id
    )
    projection = graph.projection_catalog.projection_for_edge(certificate.edge)

    assert projection.label == "HANDLERS"
    assert set(certificate.edge.match.tokens) >= {"alpha", "beta"}
    assert {
        fact.name
        for fact in graph.fact_authority_index.facts_for_edge(certificate.edge)
    } == {"AlphaHandler", "BetaHandler"}
    assert any(
        token.value == "alpha" and token.role is PresentationTokenRole.DICT_KEY
        for token in projection.tokens
    )
    assert any(
        token.value == "alpha_handler"
        and token.role is PresentationTokenRole.DICT_VALUE
        for token in projection.tokens
    )
    assert (
        "authority registry or subclass family" in certificate.missing_derivation_path
    )


def test_semantic_graph_overlay_remaps_line_shifted_projection_identity(
    tmp_path: Path,
) -> None:
    source = (
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n"
    )
    module_path = _write_module(tmp_path, source)
    base_graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    base_projection = next(
        projection
        for projection in base_graph.projections
        if projection.label == "STEP_TABLE"
    )

    module_path.write_text(f"\n{source}", encoding="utf-8")
    overlay = SemanticDescentGraphModuleOverlay(
        base_graph,
        tuple(parse_python_modules(tmp_path)),
    )
    updated_graph = overlay.graph()
    updated_projection = next(
        projection
        for projection in updated_graph.projections
        if projection.label == "STEP_TABLE"
    )
    certificate = next(
        item
        for item in updated_graph.missing_descent_certificates
        if item.edge.projection_id == updated_projection.projection_id
    )

    assert updated_projection.projection_id != base_projection.projection_id
    assert updated_projection.location.line == base_projection.location.line + 1
    assert certificate.edge.projection_id == updated_projection.projection_id


def test_semantic_graph_overlay_recomputes_positive_proof_locations(
    tmp_path: Path,
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'backend': self.backend,\n"
        "            'parse_valid': self.parse_valid,\n"
        "        }\n"
    )
    module_path = _write_module(tmp_path, source)
    base_graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    base_proof = next(
        certificate.proof_edges[0]
        for certificate in base_graph.certificates
        if isinstance(certificate, SemanticDerivationCertificate)
    )

    module_path.write_text(f"\n{source}", encoding="utf-8")
    updated_graph = base_graph.overlay_modules(
        tuple(parse_python_modules(tmp_path, use_parse_cache=False))
    )
    updated_proof = next(
        certificate.proof_edges[0]
        for certificate in updated_graph.certificates
        if isinstance(certificate, SemanticDerivationCertificate)
    )

    assert updated_proof.file_path == base_proof.file_path
    assert updated_proof.line == base_proof.line + 1


def test_semantic_descent_graph_flags_returned_declaration_table(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class FieldName(StrEnum):\n"
        "    TITLE = 'title'\n"
        "    SUMMARY = 'summary'\n"
        "    STATUS = 'status'\n"
        "\n"
        "class Manifest:\n"
        "    def to_dict(self) -> dict[str, tuple[FieldName, ...]]:\n"
        "        return {'fields': ('title', 'summary', 'status')}\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    field_authority = next(
        authority for authority in graph.authorities if authority.name == "FieldName"
    )
    certificate = next(
        item
        for item in graph.missing_descent_certificates
        if item.edge.authority_id == field_authority.authority_id
    )
    projection = graph.projection_catalog.projection_for_edge(certificate.edge)

    assert projection.kind is PresentationProjectionKind.MAPPING_LITERAL
    assert projection.label.startswith("Manifest.to_dict:return@")
    assert set(certificate.edge.match.tokens) >= {"title", "summary", "status"}
    assert (
        "repeats members from enum `FieldName`" in certificate.missing_derivation_path
    )
    assert "derive it by iterating enum members" in certificate.missing_derivation_path


def test_semantic_mirror_detector_reports_authority_and_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )

    findings = SemanticMirrorWithoutDescentDetector().detect(
        parse_python_modules(tmp_path),
        DetectorConfig(),
    )

    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
    )
    assert "`STEP_TABLE` mirrors `Step`" in finding.title
    assert "without a descent path" in finding.summary
    assert "derived class-family registry" in finding.capability_gap
    assert finding.metrics.plan_registry_name == "STEP_TABLE"
    assert finding.metrics.plan_class_names == ("LoadStep", "SaveStep")
    assert finding.metrics.plan_class_key_pairs == (
        "LoadStep='load'",
        "SaveStep='save'",
    )


def test_class_family_mirror_requires_affinity_for_reused_crosscutting_facts(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class SerializerAdapter:\n"
        "    pass\n"
        "\n"
        "class RenderAdapter:\n"
        "    pass\n"
        "\n"
        "class CsvAdapter(SerializerAdapter, RenderAdapter):\n"
        "    adapter_id = 'csv'\n"
        "\n"
        "class JsonAdapter(SerializerAdapter, RenderAdapter):\n"
        "    adapter_id = 'json'\n"
        "\n"
        "SERIALIZER_ADAPTER_TABLE = {'csv': CsvAdapter, 'json': JsonAdapter}\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )
    mirror_titles = {
        finding.title
        for finding in findings
        if finding.metrics.plan_registry_name == "SERIALIZER_ADAPTER_TABLE"
    }

    assert mirror_titles == {"`SERIALIZER_ADAPTER_TABLE` mirrors `SerializerAdapter`"}


def test_semantic_mirror_focused_collection_filters_before_rendering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    target_path = package_root / "alpha.py"
    target_path.write_text(
        "class AlphaStep:\n"
        "    pass\n"
        "\n"
        "class LoadAlphaStep(AlphaStep):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveAlphaStep(AlphaStep):\n"
        "    step_id = 'save'\n"
        "\n"
        "ALPHA_STEPS = {'load': LoadAlphaStep, 'save': SaveAlphaStep}\n",
        encoding="utf-8",
    )
    (package_root / "beta.py").write_text(
        "class BetaStep:\n"
        "    pass\n"
        "\n"
        "class LoadBetaStep(BetaStep):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveBetaStep(BetaStep):\n"
        "    step_id = 'save'\n"
        "\n"
        "BETA_STEPS = {'load': LoadBetaStep, 'save': SaveBetaStep}\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    graph = build_semantic_descent_graph(modules, use_cache=False)
    detector = SemanticMirrorWithoutDescentDetector()
    full_findings = detector._collect_findings_from_graph(
        graph,
        modules,
        DetectorConfig(),
    )
    expected_findings = [
        finding
        for finding in full_findings
        if any(
            Path(evidence.file_path).resolve() == target_path.resolve()
            for evidence in finding.evidence
        )
    ]
    rendered_certificates = []
    original_renderer = detector._finding_for_certificate

    def counted_renderer(graph, certificate):
        rendered_certificates.append(certificate)
        return original_renderer(graph, certificate)

    monkeypatch.setattr(detector, "_finding_for_certificate", counted_renderer)
    focused_findings = detector._collect_focused_findings_from_graph(
        graph,
        modules,
        DetectorConfig(),
        includes_path=lambda path: path.resolve() == target_path.resolve(),
    )

    assert [item.stable_id for item in focused_findings] == [
        item.stable_id for item in expected_findings
    ]
    assert len(rendered_certificates) == len(focused_findings)
    assert len(rendered_certificates) < len(graph.missing_descent_certificates)


def test_semantic_mirror_detector_uses_semantic_descent_context_signature(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    pass\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep}\n",
    )
    modules = tuple(parse_python_modules(tmp_path))

    assert (
        SemanticMirrorWithoutDescentDetector.cache_granularity
        is DetectorCacheGranularity.CONTEXTUAL_GLOBAL
    )
    assert (
        SemanticMirrorWithoutDescentDetector.context_signature(
            modules,
            DetectorConfig(),
        )
        == SemanticDescentGraphCacheIdentity.from_modules(modules).cache_token
    )


def test_semantic_mirror_finding_projects_to_descent_graph(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
        if item.detector_id == "semantic_mirror_without_descent"
    )

    graph = build_finding_backed_semantic_descent_graph(
        (finding,),
        semantic_mirror_detector_ids=frozenset({finding.detector_id}),
        authority_evidence_index_by_detector_id={finding.detector_id: 1},
    )

    authority = graph.authorities[0]
    projection = graph.projections[0]
    certificate = graph.missing_descent_certificates[0]

    assert authority.name == "Step"
    assert authority.kind is SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY
    assert projection.kind is PresentationProjectionKind.DETECTOR_FINDING
    assert projection.projection_id == semantic_descent_finding_projection_id(finding)
    assert projection.source_text == finding.stable_id
    assert certificate.edge.authority_id == authority.authority_id
    assert certificate.edge.projection_id == projection.projection_id
    assert certificate.missing_derivation_path == finding.relation_context
    assert {fact.name for fact in graph.facts} == {"LoadStep", "SaveStep"}


def test_finding_backed_graph_projects_non_mirror_metrics_authority() -> None:
    finding = RefactorFinding(
        detector_id="authority_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Role cases should descend to an authority",
        summary="local case table mirrors an axis authority",
        why="case literals repeat an axis-owned semantic fact family",
        capability_gap="one authority-owned role-case projection",
        relation_context="case table lacks an authority-derived projection",
        evidence=(SourceLocation("pkg/mod.py", 7, "local_cases"),),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=3,
            mapping_name="authority_projection_test",
            source_name="AxisRoleAuthority",
            field_names=("alpha", "beta"),
            identity_field_names=("role",),
        ),
    )

    graph = build_finding_backed_semantic_descent_graph(
        (finding,),
        semantic_mirror_detector_ids=frozenset(),
        authority_evidence_index_by_detector_id={},
    )
    authority = graph.authorities[0]
    certificate = graph.missing_descent_certificates[0]

    assert authority.name == "AxisRoleAuthority"
    assert authority.kind is SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY
    assert {fact.name for fact in graph.facts} == {"alpha", "beta"}
    assert certificate.missing_derivation_path == finding.relation_context


def test_finding_backed_graph_falls_back_to_evidence_owner_for_generic_metric_authority() -> (
    None
):
    finding = RefactorFinding(
        detector_id="authority_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Role cases should descend to an authority",
        summary="local case table mirrors an axis authority",
        why="case literals repeat an axis-owned semantic fact family",
        capability_gap="one authority-owned role-case projection",
        relation_context="case table lacks an authority-derived projection",
        evidence=(SourceLocation("pkg/mod.py", 7, "local_cases"),),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=3,
            mapping_name="authority_projection_test",
            source_name="authority,authority",
            field_names=("alpha", "beta"),
            identity_field_names=("role",),
        ),
    )

    graph = build_finding_backed_semantic_descent_graph(
        (finding,),
        semantic_mirror_detector_ids=frozenset(),
        authority_evidence_index_by_detector_id={},
    )

    assert graph.authorities[0].name == "local_cases"


def test_finding_backed_graph_uses_common_evidence_owner_before_detector_mapping_name() -> (
    None
):
    first = RefactorFinding(
        detector_id="cross_module_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Concrete role-case tables should move behind one generic axis authority",
        summary="codemod plan command cases mirror a shared authority",
        why="case literals repeat a semantic fact family",
        capability_gap="one authority-owned role-case projection",
        relation_context="case table lacks an authority-derived projection",
        evidence=(
            SourceLocation(
                "pkg/cli.py",
                10,
                "CodemodSynthesizePlanCliCommand:role_cases:applied,diff",
            ),
        ),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            mapping_name="cross_module_projection_test",
            source_name="codemod,command,plan",
            field_names=("applied", "diff"),
        ),
    )
    second = RefactorFinding(
        detector_id="cross_module_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title=first.title,
        summary="codemod class plan command cases mirror a shared authority",
        why=first.why,
        capability_gap=first.capability_gap,
        relation_context=first.relation_context,
        evidence=(
            SourceLocation(
                "pkg/cli.py",
                20,
                "CodemodSynthesizeClassPlanCliCommand:role_cases:applied,diff",
            ),
        ),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            mapping_name="cross_module_projection_test",
            source_name="codemod,command,plan",
            field_names=("applied", "diff"),
        ),
    )

    graph = build_finding_backed_semantic_descent_graph(
        (first, second),
        semantic_mirror_detector_ids=frozenset(),
        authority_evidence_index_by_detector_id={},
    )

    assert {authority.name for authority in graph.authorities} == {
        "CodemodSynthesizePlanCliCommand",
        "CodemodSynthesizeClassPlanCliCommand",
    }


def test_gate_uses_finding_backed_graph_for_non_mirror_authority() -> None:
    finding = RefactorFinding(
        detector_id="authority_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Role cases should descend to an authority",
        summary="local case table mirrors an axis authority",
        why="case literals repeat an axis-owned semantic fact family",
        capability_gap="one authority-owned role-case projection",
        relation_context="case table lacks an authority-derived projection",
        evidence=(SourceLocation("pkg/mod.py", 7, "local_cases"),),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=3,
            mapping_name="authority_projection_test",
            source_name="AxisRoleAuthority",
            field_names=("alpha", "beta"),
            identity_field_names=("role",),
        ),
    )
    graph = build_finding_backed_semantic_descent_graph(
        (finding,),
        semantic_mirror_detector_ids=frozenset(),
        authority_evidence_index_by_detector_id={},
    )

    work_item = SemanticRefactorGateWorkItem.from_ssot_finding_group(
        (finding,),
        finding_descent_graph=graph,
    )

    assert work_item.group_key.authority_label == "AxisRoleAuthority"
    assert work_item.group_key.descent_path == finding.relation_context
    assert work_item.evidence_symbols == ("local_cases",)
    assert work_item.certificate_count == 1
    assert work_item.matched_fact_count == 2
    assert work_item.authority_kinds == ("finding_declared_authority",)
    assert work_item.projection_kinds == ("detector_finding",)


def test_nominal_boundary_goal_targets_all_ssot_authority_findings_by_default() -> None:
    mirror_finding = _goal_policy_finding("semantic_mirror_without_descent")
    non_mirror_ssot_finding = _goal_policy_finding("repeated_builder_calls")
    ordinary_finding = _goal_policy_finding("typing_protocol_contract")
    findings = (mirror_finding, non_mirror_ssot_finding, ordinary_finding)
    assert (
        non_mirror_ssot_finding.detector_id
        in IssueDetector.ssot_authority_detector_ids()
    )
    assert (
        non_mirror_ssot_finding.detector_id
        not in IssueDetector.semantic_mirror_detector_ids()
    )
    assert tuple(
        finding.detector_id
        for finding in NominalBoundaryConcept.target_findings(findings)
    ) == ("semantic_mirror_without_descent", "repeated_builder_calls")


def test_dataclass_template_materializer_certifies_projection_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Action:\n"
        "    kind: str\n"
        "    description: str\n"
        "    target: str\n"
        "    confidence: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionTemplate:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "\n"
        "    def to_action(self, target: str) -> Action:\n"
        "        return Action(\n"
        "            kind=self.kind,\n"
        "            description=self.description,\n"
        "            target=target,\n"
        "            confidence=self.confidence,\n"
        "        )\n"
        "\n"
        "TEMPLATES = (\n"
        "    ActionTemplate(kind='create', description='Create', confidence='high'),\n"
        "    ActionTemplate(kind='delete', description='Delete', confidence='medium'),\n"
        ")\n",
    )

    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )

    assert not any(
        finding.metrics.plan_mapping_name == "TEMPLATES"
        and finding.metrics.plan_source_name == "Action"
        for finding in findings
    )


def test_dataclass_template_construction_does_not_mirror_sibling_schema(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Action:\n"
        "    kind: str\n"
        "    description: str\n"
        "    target: str\n"
        "    confidence: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionTemplate:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "\n"
        "TEMPLATES = (\n"
        "    ActionTemplate(kind='create', description='Create', confidence='high'),\n"
        "    ActionTemplate(kind='delete', description='Delete', confidence='medium'),\n"
        ")\n",
    )

    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )

    assert not any(
        finding.metrics.plan_mapping_name == "TEMPLATES"
        and finding.metrics.plan_source_name == "Action"
        for finding in findings
    )


def test_semantic_descent_ignores_suppression_vocabularies(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ProductRecordOption:\n"
        "    bases: str\n"
        "    defaults: str\n"
        "    doc: str\n"
        "\n"
        "OPTIONS = ('bases', 'defaults', 'doc')\n"
        "OPTION_STOP_TOKENS = frozenset(('bases', 'defaults', 'doc'))\n"
        "OPTION_STOPWORDS = frozenset(('bases', 'defaults', 'doc'))\n"
        "EXCLUDED_OPTION_NAMES = frozenset(('bases', 'defaults', 'doc'))\n"
        "OPAQUE_OPTION_ANNOTATION_NAMES = frozenset(('bases', 'defaults', 'doc'))\n",
    )

    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )

    assert any(
        finding.metrics.plan_mapping_name == "OPTIONS"
        and finding.metrics.plan_source_name == "ProductRecordOption"
        for finding in findings
    )
    assert not any(
        finding.metrics.plan_mapping_name
        in {
            "OPTION_STOP_TOKENS",
            "OPTION_STOPWORDS",
            "EXCLUDED_OPTION_NAMES",
            "OPAQUE_OPTION_ANNOTATION_NAMES",
        }
        and finding.metrics.plan_source_name == "ProductRecordOption"
        for finding in findings
    )


def test_semantic_mirror_registry_finding_synthesizes_autoregister_recipe(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    operation = plan.document.to_dict()["recipes"][0]["operations"][0]
    record = plan.records[0]
    repair_plan = record.semantic_repair_plan

    assert plan.expected_removed_finding_count == 1
    assert record.detector_id == "semantic_mirror_without_descent"
    assert record.status.value == "planned"
    assert (
        record.executable_declaration_name
        == "ManualClassRegistrationFindingRecipeSynthesizer"
    )
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert operation["registry_name"] == "STEP_TABLE"
    assert operation["class_key_pairs"] == ("LoadStep='load'", "SaveStep='save'")
    assert repair_plan is not None
    assert repair_plan.repair_kind == "registration"
    assert repair_plan.action_keys == record.action_keys
    assert repair_plan.operation_kinds == ("convert_manual_registry_to_autoregister",)
    repaired_finding = next(
        finding for finding in findings if finding.stable_id == repair_plan.finding_id
    )
    assert repair_plan.missing_derivation_path == repaired_finding.relation_context
    assert (
        repair_plan.to_dict()["missing_derivation_path"]
        == repaired_finding.relation_context
    )
    assert simulation.is_clean is True
    assert simulation.simulation.applied_rewrite_count == 1


def test_codemod_source_context_hydrates_selected_finding_files_only(
    tmp_path: Path,
) -> None:
    alpha_path = tmp_path / "alpha.py"
    beta_path = tmp_path / "beta.py"
    alpha_path.write_text("import beta\n\nclass Alpha:\n    pass\n", encoding="utf-8")
    beta_path.write_text("class Beta:\n    pass\n", encoding="utf-8")
    modules = parse_python_modules(tmp_path, use_parse_cache=False, parse_workers=1)
    alpha_finding = RefactorFinding(
        detector_id="semantic_mirror_without_descent",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Alpha mirror",
        summary="alpha mirrors beta",
        why="projection mirrors an authority",
        capability_gap="derive the projection",
        relation_context="missing derivation path",
        evidence=(SourceLocation(alpha_path.as_posix(), 3, "Alpha"),),
    )
    beta_finding = replace(
        alpha_finding,
        title="Beta mirror",
        evidence=(SourceLocation(beta_path.as_posix(), 1, "Beta"),),
    )

    context = CodemodSourceContext.from_modules(
        modules,
        (alpha_finding, beta_finding),
    )
    snapshot = context.snapshot_for_findings((alpha_finding,))

    assert tuple(snapshot.module_node_cache or {}) == (alpha_path.as_posix(),)
    assert {
        snapshot.source_index.target_by_id[target_id].file_path
        for target_id in snapshot.ast_target_node_cache or {}
    } == {alpha_path.as_posix()}
    assert snapshot.module_import_graph.import_would_create_cycle(
        importing_file_path=beta_path.as_posix(),
        imported_file_path=alpha_path.as_posix(),
    )


def test_inherited_autoregister_config_synthesizes_assignment_deletions(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC, abstractmethod\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "DEFAULT_REGISTRY_KEY_ATTRIBUTE = '__registry_key__'\n"
        "\n"
        "def class_name_registry_key(name, cls):\n"
        "    return name.lower()\n"
        "\n"
        "class RegisteredEvidenceProperty(ABC):\n"
        "    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE\n"
        "    __key_extractor__ = class_name_registry_key\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "class SourceLocationZipEvidenceProperty(\n"
        "    RegisteredEvidenceProperty, ABC, metaclass=AutoRegisterMeta\n"
        "):\n"
        "    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE\n"
        "    __key_extractor__ = class_name_registry_key\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "    @abstractmethod\n"
        "    def _source_locations(self, instance):\n"
        "        raise NotImplementedError\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in InheritedAutoRegisterConfigBoilerplateDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.detector_id == "inherited_autoregister_config_boilerplate"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    operations = tuple(
        operation.to_dict() for operation in plan.document.recipes[0].operations
    )
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name
        == "InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer"
    )
    assert {operation["attribute_name"] for operation in operations} == {
        "__key_extractor__",
        "__registry_key__",
        "__skip_if_no_key__",
    }
    assert all(
        operation["operation"] == "delete_class_assignment" for operation in operations
    )
    assert rewritten.count("    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE") == 1
    assert rewritten.count("    __key_extractor__ = class_name_registry_key") == 1
    assert rewritten.count("    __skip_if_no_key__ = True") == 1
    assert simulation.is_clean is True


def test_derived_metric_count_synthesizes_constructor_rewrite(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "def build_metrics(field_names):\n"
        "    return MappingMetrics(\n"
        "        mapping_site_count=1,\n"
        "        field_count=len(field_names),\n"
        "        mapping_name='case_table',\n"
        "        field_names=field_names,\n"
        "    )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in DerivedMetricCountBoilerplateDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.detector_id == "derived_metric_count_boilerplate"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    operation = plan.document.recipes[0].operations[0].to_dict()
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name
        == "DerivedMetricCountBoilerplateFindingRecipeSynthesizer"
    )
    assert operation["operation"] == "replace_text"
    assert "MappingMetrics.from_field_names(" in rewritten
    assert "field_count=len(field_names)" not in rewritten
    assert "field_names=field_names" in rewritten
    assert simulation.is_clean is True


def test_identity_keyword_forwarding_shell_synthesizes_inline_delete_recipe(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SupportItem:\n"
        "    name: str\n"
        "    value: str\n"
        "\n"
        "class SupportBuilder:\n"
        "    def first(self, name, value):\n"
        "        return (self.item(name, value),)\n"
        "\n"
        "    def second(self, *, name, value):\n"
        "        return self.item(value=value, name=name)\n"
        "\n"
        "    @staticmethod\n"
        "    def item(name, value):\n"
        "        return SupportItem(name=name, value=value)\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "identity_keyword_forwarding_shell"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    recipe = plan.document.to_dict()["recipes"][0]
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name
        == "IdentityKeywordForwardingShellFindingRecipeSynthesizer"
    )
    assert [operation["operation"] for operation in recipe["operations"]] == [
        "replace_target",
    ]
    assert "def item" not in rewritten
    assert "return (SupportItem(name=name, value=value),)" in rewritten
    assert "return SupportItem(name=name, value=value)" in rewritten
    assert simulation.is_clean is True


def test_identity_keyword_forwarding_shell_coalesces_nested_function_call_site(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SupportItem:\n"
        "    name: str\n"
        "    value: str\n"
        "\n"
        "class SupportBuilder:\n"
        "    def first(self, name, value):\n"
        "        def nested():\n"
        "            return self.second(value=value, name=name)\n"
        "\n"
        "        return nested()\n"
        "\n"
        "    def second(self, *, name, value):\n"
        "        return SupportItem(name=name, value=value)\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "identity_keyword_forwarding_shell"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert "def second" not in rewritten
    assert "return SupportItem(name=name, value=value)" in rewritten
    assert rewritten.count("SupportItem(name=name, value=value)") == 1
    assert simulation.is_clean is True


def test_repeated_builder_call_synthesizes_constructor_authority_recipe(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BranchItem:\n"
        "    axis_name: str\n"
        "    expected_source: str\n"
        "    result_source: str\n"
        "\n"
        "    def render(self):\n"
        "        return self.axis_name\n"
        "\n"
        "class BranchBuilder:\n"
        "    def first(self, source_axis, expected_source, result_source):\n"
        "        return BranchItem(\n"
        "            axis_name=source_axis,\n"
        "            expected_source=expected_source,\n"
        "            result_source=result_source,\n"
        "        )\n"
        "\n"
        "    def second(self, source_axis, expected_source, result_source):\n"
        "        return BranchItem(\n"
        "            axis_name=source_axis,\n"
        "            expected_source=expected_source,\n"
        "            result_source=result_source,\n"
        "        )\n"
        "\n"
        "    def third(self, source_axis, expected_source, result_source):\n"
        "        return BranchItem(\n"
        "            axis_name=source_axis,\n"
        "            expected_source=expected_source,\n"
        "            result_source=result_source,\n"
        "        )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "repeated_builder_calls"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    recipe = plan.document.to_dict()["recipes"][0]
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert (
        plan.records[0].executable_declaration_name == "RepeatedBuilderAuthorityMethod"
    )
    assert [operation["operation"] for operation in recipe["operations"]] == [
        "replace_target",
        "replace_target",
        "replace_target",
        "replace_target",
    ]
    assert "def from_sources(" in rewritten
    assert rewritten.count("BranchItem.from_sources(") == 3
    assert "return cls(" in rewritten
    assert simulation.is_clean is True


def test_repeated_builder_call_keeps_repeated_local_values_as_parameters(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "from enum import Enum\n"
        "\n"
        "class NodeKind(Enum):\n"
        "    FUNCTION = 'function'\n"
        "    METHOD = 'method'\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class TargetSelector:\n"
        "    node_kinds: tuple[NodeKind, ...] = ()\n"
        "    file_paths: tuple[str, ...] = ()\n"
        "    qualnames: tuple[str, ...] = ()\n"
        "\n"
        "    def select(self):\n"
        "        return self.qualnames\n"
        "\n"
        "class SelectorBuilder:\n"
        "    def first(self, source_path, wrapper_qualname):\n"
        "        return TargetSelector(\n"
        "            node_kinds=(NodeKind.FUNCTION, NodeKind.METHOD),\n"
        "            file_paths=(source_path,),\n"
        "            qualnames=(wrapper_qualname,),\n"
        "        )\n"
        "\n"
        "    def second(self, source_path, function_qualname):\n"
        "        return TargetSelector(\n"
        "            node_kinds=(NodeKind.FUNCTION, NodeKind.METHOD),\n"
        "            file_paths=(source_path,),\n"
        "            qualnames=(function_qualname,),\n"
        "        )\n"
        "\n"
        "    def third(self, source_path, target_qualname):\n"
        "        return TargetSelector(\n"
        "            node_kinds=(NodeKind.FUNCTION, NodeKind.METHOD),\n"
        "            file_paths=(source_path,),\n"
        "            qualnames=(target_qualname,),\n"
        "        )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "repeated_builder_calls"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert "def for_function_or_method(" in rewritten
    assert "file_path: str" in rewritten
    assert "qualname: str" in rewritten
    assert "file_paths=(file_path,)" in rewritten
    assert "file_paths=(source_path,)" not in rewritten
    assert rewritten.count("TargetSelector.for_function_or_method(") == 3
    assert simulation.is_clean is True


def test_repeated_builder_call_synthesizes_single_owner_method_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    stages: tuple[str, ...]\n"
        "    scan: str\n"
        "    reason: str\n"
        "    completed: bool\n"
        "    terminal_synthesis_report: str | None = None\n"
        "\n"
        "class Runner:\n"
        "    def run(self, active_scan, stages, stage, report):\n"
        "        if not active_scan:\n"
        "            return self.report(\n"
        "                stages=(),\n"
        "                scan=active_scan,\n"
        "                reason='no_targets',\n"
        "                completed=True,\n"
        "            )\n"
        "        if report is None:\n"
        "            return self.report(\n"
        "                stages=tuple(stages),\n"
        "                scan=active_scan,\n"
        "                reason='no_recipe',\n"
        "                completed=False,\n"
        "                terminal_synthesis_report=report,\n"
        "            )\n"
        "        if stage:\n"
        "            return self.report(\n"
        "                stages=(*stages, stage),\n"
        "                scan=active_scan,\n"
        "                reason='staged',\n"
        "                completed=False,\n"
        "            )\n"
        "        return self.report(\n"
        "            stages=tuple(stages),\n"
        "            scan=active_scan,\n"
        "            reason='done',\n"
        "            completed=True,\n"
        "            terminal_synthesis_report=report,\n"
        "        )\n"
        "\n"
        "    def report(\n"
        "        self,\n"
        "        *,\n"
        "        stages: tuple[str, ...],\n"
        "        scan: str,\n"
        "        reason: str,\n"
        "        completed: bool,\n"
        "        terminal_synthesis_report: str | None = None,\n"
        "    ) -> Report:\n"
        "        return Report(\n"
        "            stages=stages,\n"
        "            scan=scan,\n"
        "            reason=reason,\n"
        "            completed=completed,\n"
        "            terminal_synthesis_report=terminal_synthesis_report,\n"
        "        )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_path(tmp_path)
        if item.detector_id == "repeated_builder_calls"
        and item.relation_context
        == "one owner repeats a builder call family with varying declarative payload"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))
    projected_findings = analyze_modules(
        snapshot.with_virtual_sources(
            simulation.simulation.rewritten_sources
        ).parsed_modules,
        DetectorConfig(),
    )

    assert plan.records[0].status.value == "planned"
    assert "def _run_report_authority(" in rewritten
    assert "return self.report(" in rewritten
    assert rewritten.count("self._run_report_authority(") == 4
    assert rewritten.count("return self.report(") == 1
    assert "terminal_synthesis_report: str | None = None" in rewritten
    assert not any(
        item.detector_id == "repeated_builder_calls"
        and item.relation_context
        == "one owner repeats a builder call family with varying declarative payload"
        for item in projected_findings
    )
    assert simulation.is_clean is True


def test_finding_recipe_synthesis_detector_scope_excludes_unselected_findings(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.detector_id == "semantic_mirror_without_descent"
    )
    unrelated_finding = replace(finding, detector_id="prefixed_role_field_bundle")

    plan = codemod_plan_from_findings(
        (finding, unrelated_finding),
        detector_ids=("semantic_mirror_without_descent",),
    )

    assert tuple(record.detector_id for record in plan.records) == (
        "semantic_mirror_without_descent",
    )


def test_semantic_mirror_registry_recipe_resolves_absolute_finding_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module_path = _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    monkeypatch.chdir(tmp_path)
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            "pkg/mod.py": module_path.read_text(encoding="utf-8"),
        }
    )
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            snapshot.parsed_modules,
            DetectorConfig(),
        )
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
    )
    absolute_finding = replace(
        finding,
        evidence=tuple(
            replace(
                location,
                file_path=(tmp_path / location.file_path).resolve().as_posix(),
            )
            for location in finding.evidence
        ),
    )

    plan = codemod_plan_from_findings(
        (absolute_finding,),
        selector_context=snapshot,
    )
    simulation = plan.simulate_snapshot(snapshot)
    operation = plan.document.recipes[0].operations[0].to_dict()

    assert plan.records[0].status.value == "planned"
    assert plan.expected_removed_finding_count == 1
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert operation["registry_name"] == "STEP_TABLE"
    assert simulation.is_clean is True


def test_semantic_mirror_autoregister_instance_view_synthesizes_recipe(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC, abstractmethod\n"
        "from enum import StrEnum\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "class StepId(StrEnum):\n"
        "    LOAD = 'load'\n"
        "    SAVE = 'save'\n"
        "\n"
        "class Step(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = 'registry_key'\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "    @abstractmethod\n"
        "    def build(self):\n"
        "        raise NotImplementedError\n"
        "\n"
        "class LoadStep(Step):\n"
        "    def build(self):\n"
        "        return 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    def build(self):\n"
        "        return 'save'\n"
        "\n"
        "STEP_TABLE = {StepId.LOAD: LoadStep(), StepId.SAVE: SaveStep()}\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_registry_name == "STEP_TABLE"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    operation = plan.document.recipes[0].operations[0].to_dict()
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "planned"
    assert operation["operation"] == "derive_autoregister_instance_view"
    assert operation["assignment_name"] == "STEP_TABLE"
    assert "__registry__ = {}" in rewritten
    assert "registry_key = StepId.LOAD" in rewritten
    assert "registry_key = StepId.SAVE" in rewritten
    assert "def instances_by_registry_key" in rewritten
    assert "key_attribute = cls.__registry_key__" in rewritten
    assert "registered_type.__dict__[key_attribute]: registered_type()" in rewritten
    assert "if isinstance(registered_type.__dict__[key_attribute], StepId)" in rewritten
    assert "for key, registered_type in cls.__registry__.items()" not in rewritten
    assert "STEP_TABLE = Step.instances_by_registry_key()" in rewritten
    assert simulation.is_clean is True


def test_semantic_descent_resolves_bound_constructor_values_to_class_family(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "class ActionBuilder(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry__ = {}\n"
        "    __registry_key__ = 'builder_id'\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "class GenericActionBuilder(ActionBuilder):\n"
        "    builder_id = 'generic'\n"
        "\n"
        "class TemplateActionBuilder(ActionBuilder):\n"
        "    builder_id = 'template'\n"
        "\n"
        "class AbcActionBuilder(ActionBuilder):\n"
        "    builder_id = 'abc'\n"
        "\n"
        "TEMPLATE_BUILDER = TemplateActionBuilder()\n"
        "ACTION_BUILDERS = {\n"
        "    'abc': AbcActionBuilder(),\n"
        "    'template': TEMPLATE_BUILDER,\n"
        "}\n",
    )

    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )

    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_registry_name == "ACTION_BUILDERS"
    )
    assert "`ACTION_BUILDERS` mirrors `ActionBuilder`" in finding.title
    assert finding.metrics.plan_class_names == (
        "AbcActionBuilder",
        "TemplateActionBuilder",
    )
    assert finding.metrics.plan_class_key_pairs == (
        "AbcActionBuilder='abc'",
        "TemplateActionBuilder='template'",
    )


def test_partial_class_constructor_composition_is_not_a_family_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class ValueCodec(ABC):\n"
        "    @abstractmethod\n"
        "    def read(self):\n"
        "        raise NotImplementedError\n"
        "\n"
        "class StringCodec(ValueCodec):\n"
        "    def read(self):\n"
        "        return 'string'\n"
        "\n"
        "class IntegerCodec(ValueCodec):\n"
        "    def read(self):\n"
        "        return 'integer'\n"
        "\n"
        "class BooleanCodec(ValueCodec):\n"
        "    def read(self):\n"
        "        return 'boolean'\n"
        "\n"
        "class ObjectCodec(ValueCodec):\n"
        "    def read(self):\n"
        "        return 'object'\n"
        "\n"
        "class BindingSet:\n"
        "    @classmethod\n"
        "    def from_specs(cls, *specs):\n"
        "        return cls()\n"
        "\n"
        "class CodecCatalog:\n"
        "    def __init__(self, codecs):\n"
        "        self.codecs = codecs\n"
        "\n"
        "PLAN_BINDINGS = BindingSet.from_specs(\n"
        "    ('name', StringCodec()),\n"
        "    ('count', IntegerCodec()),\n"
        "    ('enabled', BooleanCodec()),\n"
        ")\n"
        "ALL_CODECS = CodecCatalog((\n"
        "    StringCodec(), IntegerCodec(), BooleanCodec(), ObjectCodec(),\n"
        "))\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    codec_authority = next(
        authority for authority in graph.authorities if authority.name == "ValueCodec"
    )
    mirrored_projection_labels = {
        graph.projection_catalog.projection_for_edge(certificate.edge).label
        for certificate in graph.missing_descent_certificates
        if certificate.edge.authority_id == codec_authority.authority_id
    }

    assert "PLAN_BINDINGS" not in mirrored_projection_labels
    assert "ALL_CODECS" in mirrored_projection_labels


def test_semantic_mirror_omits_ambiguous_mapping_class_key_fallback(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "class ActionBuilder(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = 'builder_id'\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "class TemplateActionBuilder(ActionBuilder):\n"
        "    builder_id = 'template'\n"
        "\n"
        "class AbcActionBuilder(ActionBuilder):\n"
        "    builder_id = 'abc'\n"
        "\n"
        "FAST_TEMPLATE = TemplateActionBuilder()\n"
        "SAFE_TEMPLATE = TemplateActionBuilder()\n"
        "ACTION_BUILDERS = {\n"
        "    'abc': AbcActionBuilder(),\n"
        "    'fast': FAST_TEMPLATE,\n"
        "    'safe': SAFE_TEMPLATE,\n"
        "}\n",
    )

    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )

    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_registry_name == "ACTION_BUILDERS"
    )
    assert finding.metrics.plan_class_names == (
        "AbcActionBuilder",
        "TemplateActionBuilder",
    )
    assert finding.metrics.plan_class_key_pairs == ("AbcActionBuilder='abc'",)
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))
    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    assert plan.records[0].status.value == "rejected_by_safety_check"
    assert "class/key pairs are incomplete" in plan.records[0].reason


def test_dataclass_template_keywords_do_not_invent_dsl_action(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "    create_symbol: str | None = None\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionTemplate:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "\n"
        "ACTION_TEMPLATES = (\n"
        "    ActionTemplate(\n"
        "        kind='create_base', description='Create base', confidence='high'\n"
        "    ),\n"
        "    ActionTemplate(\n"
        "        kind='move_fields', description='Move fields', confidence='medium'\n"
        "    ),\n"
        ")\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    assert not any(
        finding.metrics.plan_source_name == "RefactorAction" for finding in findings
    )


def test_semantic_mirror_return_dict_synthesizes_dataclass_payload_recipe(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionReport:\n"
        "    action: RefactorAction\n"
        "    emitted: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'kind': self.action.kind,\n"
        "            'description': self.action.description,\n"
        "            'confidence': self.action.confidence,\n"
        "            'emitted': self.emitted,\n"
        "        }\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_source_name == "RefactorAction"
        and item.metrics.plan_mapping_name == "ActionReport.to_dict:return@15"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate_snapshot(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[str(module_path)]
    recipe = plan.document.recipes[0]

    assert record.status.value == "planned"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert record.refactor_concept == "dataclass_payload_projection"
    assert (
        record.executable_declaration_name
        == "DataclassPayloadProjectionMappingRecipeBuilder"
    )
    assert record.semantic_repair_plan is not None
    assert record.semantic_repair_plan.repair_kind == "mapping"
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "replace_text",
        "replace_target",
    )
    assert "def payload_from_field_values(cls, **values)" in rewritten_source
    assert "**RefactorAction.payload_from_field_values(" in rewritten_source
    assert "kind=self.action.kind" in rewritten_source
    assert "description=self.action.description" in rewritten_source
    assert "confidence=self.action.confidence" in rewritten_source
    assert "'emitted': self.emitted" in rewritten_source


def test_semantic_mirror_key_value_sequence_synthesizes_dataclass_payload_recipe(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class WorkflowModel:\n"
        "    workflow_id: str\n"
        "    command_action_ids: tuple[str, ...]\n"
        "    default_next_action_id: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class WorkflowDefinition:\n"
        "    workflow_id: str\n"
        "    description: str\n"
        "    command_action_ids: tuple[str, ...]\n"
        "    default_next_action_id: str\n"
        "\n"
        "    def definition_items(self):\n"
        "        return (\n"
        "            ('workflow_id', self.workflow_id),\n"
        "            ('description', self.description),\n"
        "            ('command_action_ids', self.command_action_ids),\n"
        "            ('default_next_action_id', self.default_next_action_id),\n"
        "        )\n",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_source_name == "WorkflowModel"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate_snapshot(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[str(module_path)]

    assert record.status.value == "planned"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert record.refactor_concept == "dataclass_payload_projection"
    assert "def payload_items_from_field_values(cls, **values)" in rewritten_source
    assert "*WorkflowModel.payload_items_from_field_values(" in rewritten_source
    assert "workflow_id=self.workflow_id" in rewritten_source
    assert "command_action_ids=self.command_action_ids" in rewritten_source
    assert "default_next_action_id=self.default_next_action_id" in rewritten_source
    assert (
        rewritten_source.index("workflow_id=self.workflow_id")
        < rewritten_source.index("('description', self.description)")
        < rewritten_source.index("command_action_ids=self.command_action_ids")
    )
    module_path.write_text(rewritten_source, encoding="utf-8")
    migrated_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path),
        use_cache=False,
    )
    workflow_authority = next(
        authority
        for authority in migrated_graph.authorities
        if authority.name == "WorkflowModel"
    )
    migrated_derivation = next(
        certificate
        for certificate in migrated_graph.certificates
        if certificate.status
        is semantic_descent_module.DescentStatus.DESCENDS_TO_AUTHORITY
        and certificate.edge.authority_id == workflow_authority.authority_id
    )

    assert all(
        certificate.edge.authority_id != workflow_authority.authority_id
        for certificate in migrated_graph.missing_descent_certificates
    )
    assert migrated_derivation.status.value == "descends_to_authority"
    assert migrated_derivation.proof_edges[0].edge_kind is (
        AuthorityProofEdgeKind.PROVIDES_QUERY_METHOD
    )


def test_semantic_mirror_cross_file_return_dict_synthesizes_dataclass_payload_recipe(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n",
        encoding="utf-8",
    )
    (package_dir / "report.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionReport:\n"
        "    action: object\n"
        "    emitted: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'kind': self.action.kind,\n"
        "            'description': self.action.description,\n"
        "            'confidence': self.action.confidence,\n"
        "            'emitted': self.emitted,\n"
        "        }\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_source_name == "RefactorAction"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate_snapshot(snapshot)
    recipe = plan.document.recipes[0]
    rewritten_model = simulation.simulation.rewritten_sources[
        (package_dir / "model.py").as_posix()
    ]
    rewritten_report = simulation.simulation.rewritten_sources[
        (package_dir / "report.py").as_posix()
    ]

    assert record.status.value == "planned"
    assert record.refactor_concept == "dataclass_payload_projection"
    assert simulation.is_clean is True
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "ensure_import",
        "replace_text",
        "replace_target",
    )
    assert "def payload_from_field_values(cls, **values)" in rewritten_model
    assert "from .model import RefactorAction" in rewritten_report
    assert "**RefactorAction.payload_from_field_values(" in rewritten_report
    assert "'emitted': self.emitted" in rewritten_report


def test_semantic_mirror_cross_file_payload_recipe_rejects_import_cycle(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.py").write_text(
        "from dataclasses import dataclass\n"
        "from .report import ActionReport\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n",
        encoding="utf-8",
    )
    (package_dir / "report.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionReport:\n"
        "    action: object\n"
        "    emitted: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'kind': self.action.kind,\n"
        "            'description': self.action.description,\n"
        "            'confidence': self.action.confidence,\n"
        "            'emitted': self.emitted,\n"
        "        }\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_source_name == "RefactorAction"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]

    assert record.status.value == "rejected_by_safety_check"
    assert "module cycle" in record.reason


def test_semantic_mirror_constructor_projection_uses_dataclass_method(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLineReplacement:\n"
        "    file_path: str\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "    replacement_lines: tuple[str, ...]\n"
        "    rationale: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLineSpan:\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "\n"
        "    def line_replacement(self, *, file_path, replacement_lines, rationale):\n"
        "        return SourceLineReplacement(\n"
        "            file_path=file_path,\n"
        "            start_line=self.start_line,\n"
        "            end_line=self.end_line,\n"
        "            replacement_lines=replacement_lines,\n"
        "            rationale=rationale,\n"
        "        )\n"
        "\n"
        "def build_replacement(source_path, insertion_line, import_lines, reason):\n"
        "    return (\n"
        "        SourceLineReplacement(\n"
        "            file_path=source_path,\n"
        "            start_line=insertion_line,\n"
        "            end_line=insertion_line - 1,\n"
        "            replacement_lines=import_lines,\n"
        "            rationale=reason,\n"
        "        ),\n"
        "    )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        detector_id="semantic_mirror_without_descent",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Semantic mirror without descent",
        summary=(
            "Constructor projection repeats SourceLineSpan fields without "
            "using the authority method."
        ),
        why="semantic fact is mirrored outside its nominal authority",
        capability_gap="derive the projection from the authority instead",
        relation_context="projection lacks a semantic-descent certificate",
        evidence=(
            SourceLocation(str(module_path), 26, "build_replacement:return@26"),
            SourceLocation(str(module_path), 12, "SourceLineSpan"),
        ),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=("end_line", "start_line"),
            mapping_name="build_replacement:return@26",
            source_name="SourceLineSpan",
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate_snapshot(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[str(module_path)]
    recipe = plan.document.recipes[0]

    assert record.status.value == "planned"
    assert record.refactor_concept == "constructor_kwarg_carrier_projection"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "replace_text",
    )
    assert "SourceLineSpan(" in rewritten_source
    assert ".line_replacement(" in rewritten_source
    assert "start_line=insertion_line" in rewritten_source
    assert "end_line=insertion_line - 1" in rewritten_source
    assert "replacement_lines=import_lines" in rewritten_source


def test_nested_factory_keyword_names_do_not_prove_dataclass_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class FindingBuildContext:\n"
        "    scaffold: str | None = None\n"
        "    codemod_patch: str | None = None\n"
        "    metrics: object | None = None\n"
        "\n"
        "class Detector:\n"
        "    def build_finding(self, summary, evidence, context=None, **overrides):\n"
        "        return (summary, evidence, context, overrides)\n"
        "\n"
        "    def collect(self, evidence, metric):\n"
        "        return [\n"
        "            self.build_finding(\n"
        "                'summary',\n"
        "                evidence,\n"
        "                scaffold='scaffold',\n"
        "                codemod_patch='patch',\n"
        "                metrics=metric,\n"
        "            )\n"
        "        ]\n",
    )
    findings = SemanticMirrorWithoutDescentDetector().detect(
        parse_python_modules(tmp_path),
        DetectorConfig(),
    )

    assert not any(
        finding.metrics.plan_source_name == "FindingBuildContext"
        for finding in findings
    )


def test_semantic_mirror_enum_subset_synthesizes_authority_method_recipe(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "taxonomy.py").write_text(
        "from enum import StrEnum\n"
        "\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n"
        "    LOW = 'low'\n",
        encoding="utf-8",
    )
    (package_dir / "codemod.py").write_text(
        "import pkg.taxonomy\n"
        "\n"
        "_ACTIONABLE_CONFIDENCE_LEVELS: frozenset[pkg.taxonomy.ConfidenceLevel] = "
        "frozenset(('high', 'medium'))\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_mapping_name == "_ACTIONABLE_CONFIDENCE_LEVELS"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    recipe_payload = plan.document.to_dict()["recipes"][0]
    operations = recipe_payload["operations"]

    assert plan.records[0].status.value == "planned"
    assert simulation.is_clean is True
    assert (
        "def actionable_confidence_levels(cls) -> frozenset[str]"
        in operations[0]["replacement_source"]
    )
    assert [operation["operation"] for operation in operations] == [
        "replace_target",
        "ensure_import",
        "replace_module_assignment",
    ]
    assert (
        operations[1]["import_source"] == "from pkg.taxonomy import ConfidenceLevel\n"
    )
    assert operations[2]["source"] == (
        "_ACTIONABLE_CONFIDENCE_LEVELS = ConfidenceLevel.actionable_confidence_levels()"
    )


def test_semantic_mirror_class_collection_synthesizes_authority_query_recipe(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "taxonomy.py").write_text(
        "from enum import StrEnum\n"
        "\n"
        "class LabeledMode(StrEnum):\n"
        "    def label(self) -> str:\n"
        "        return self.value\n"
        "\n"
        "class CapabilityMode(LabeledMode):\n"
        "    FAST = 'fast'\n"
        "\n"
        "class ObservationMode(LabeledMode):\n"
        "    SLOW = 'slow'\n",
        encoding="utf-8",
    )
    (package_dir / "detector.py").write_text(
        "from .taxonomy import CapabilityMode, ObservationMode\n"
        "\n"
        "ModeEnum = type[CapabilityMode] | type[ObservationMode]\n"
        "MODE_ENUMS: tuple[ModeEnum, ...] = (CapabilityMode, ObservationMode)\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_registry_name == "MODE_ENUMS"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    recipe_payload = plan.document.to_dict()["recipes"][0]
    recipe = plan.document.recipes[0]
    operations = recipe_payload["operations"]
    rewritten = next(
        source
        for path, source in simulation.simulation.rewritten_sources.items()
        if path.endswith("detector.py")
    )

    assert plan.records[0].status.value == "planned"
    assert len(recipe.authority_claims) == 1
    claim = recipe.authority_claims[0]
    assert claim.claimed_symbol == "LabeledMode"
    assert claim.file_path == (package_dir / "taxonomy.py").as_posix()
    assert claim.qualname == "LabeledMode"
    assert claim.authority_kind == ""
    assert claim.claimed_symbol not in {"CapabilityMode", "ObservationMode"}
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert [operation["operation"] for operation in operations] == [
        "ensure_import",
        "replace_module_assignment",
    ]
    assert operations[0]["import_source"] == "from .taxonomy import LabeledMode\n"
    assert operations[1]["source"] == (
        "MODE_ENUMS: tuple[ModeEnum, ...] = tuple(LabeledMode.__subclasses__())"
    )
    assert (
        "from .taxonomy import (\n"
        "    CapabilityMode,\n"
        "    ObservationMode,\n"
        "    LabeledMode,\n"
        ")\n"
    ) in rewritten


def test_semantic_mirror_class_name_collection_synthesizes_authority_query_recipe(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "taxonomy.py").write_text(
        "from enum import StrEnum\n"
        "\n"
        "class LabeledMode(StrEnum):\n"
        "    def label(self) -> str:\n"
        "        return self.value\n"
        "\n"
        "class CapabilityMode(LabeledMode):\n"
        "    FAST = 'fast'\n"
        "\n"
        "class ObservationMode(LabeledMode):\n"
        "    SLOW = 'slow'\n",
        encoding="utf-8",
    )
    (package_dir / "detector.py").write_text(
        "from .taxonomy import CapabilityMode, ObservationMode\n"
        "\n"
        "OWNER_NAMES = frozenset({'ObservationMode', 'CapabilityMode'})\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(modules, DetectorConfig())
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_registry_name == "OWNER_NAMES"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate_snapshot(snapshot)
    operation = plan.document.to_dict()["recipes"][0]["operations"][1]
    rewritten = next(
        source
        for path, source in simulation.simulation.rewritten_sources.items()
        if path.endswith("detector.py")
    )

    assert plan.records[0].status.value == "planned"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert operation["source"] == (
        "OWNER_NAMES = frozenset(member_type.__name__ for member_type in "
        "LabeledMode.__subclasses__())"
    )
    assert (
        "from .taxonomy import (\n"
        "    CapabilityMode,\n"
        "    ObservationMode,\n"
        "    LabeledMode,\n"
        ")\n"
    ) in rewritten


def test_semantic_descent_ignores_qualified_enum_member_references(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class LabeledMode(StrEnum):\n"
        "    def label(self) -> str:\n"
        "        return self.value\n"
        "\n"
        "class CapabilityMode(LabeledMode):\n"
        "    FAST = 'fast'\n"
        "\n"
        "class ObservationMode(LabeledMode):\n"
        "    SLOW = 'slow'\n"
        "\n"
        "SPEC = make_spec(\n"
        "    capability_tags=(CapabilityMode.FAST,),\n"
        "    observation_tags=(ObservationMode.SLOW,),\n"
        ")\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_registry_name == "SPEC"
        and finding.metrics.plan_source_name == "LabeledMode"
        for finding in findings
    )


def test_two_member_enum_overlap_without_authority_affinity_is_not_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class SourceNodeDecoratorPolicy(StrEnum):\n"
        "    EXCLUDE = 'exclude'\n"
        "    INCLUDE = 'include'\n"
        "\n"
        "def selector_payload_bindings(specs):\n"
        "    return specs\n"
        "\n"
        "SELECTOR_BINDINGS = selector_payload_bindings((\n"
        "    ('include', 'include'),\n"
        "    ('require', 'require'),\n"
        "    ('exclude', 'exclude'),\n"
        "))\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "SourceNodeDecoratorPolicy"
        for finding in findings
    )


def test_two_member_class_family_branch_without_authority_affinity_is_not_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class RegistryMembershipStep:\n"
        "    pass\n"
        "\n"
        "class InMembershipStep(RegistryMembershipStep):\n"
        "    step_id = 'in'\n"
        "\n"
        "class NotInMembershipStep(RegistryMembershipStep):\n"
        "    step_id = 'not_in'\n"
        "\n"
        "def operator_compares_to_literal(operator):\n"
        "    if operator in ('in', 'not_in'):\n"
        "        return True\n"
        "    return False\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "RegistryMembershipStep"
        for finding in findings
    )


def test_enum_string_literal_branch_still_reports_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class Mode(StrEnum):\n"
        "    FAST = 'fast'\n"
        "    SLOW = 'slow'\n"
        "    SAFE = 'safe'\n"
        "\n"
        "def classify(mode: Mode):\n"
        "    if mode in ('fast', 'slow', 'safe'):\n"
        "        return True\n"
        "    return False\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert any(
        finding.metrics.plan_source_name == "Mode"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_untyped_enum_string_literal_branch_is_not_authority_proof(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class Mode(StrEnum):\n"
        "    FAST = 'fast'\n"
        "    SLOW = 'slow'\n"
        "    SAFE = 'safe'\n"
        "\n"
        "def classify(mode):\n"
        "    if mode in ('fast', 'slow', 'safe'):\n"
        "        return True\n"
        "    return False\n",
    )

    findings = SemanticMirrorWithoutDescentDetector().detect(
        parse_python_modules(tmp_path),
        DetectorConfig(),
    )

    assert not any(finding.metrics.plan_source_name == "Mode" for finding in findings)


def test_builtin_calls_do_not_prove_enum_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class BuiltinCallName(StrEnum):\n"
        "    MIN = 'min'\n"
        "    STR = 'str'\n"
        "    SUM = 'sum'\n"
        "    TUPLE = 'tuple'\n"
        "\n"
        "def collect(values):\n"
        "    return (\n"
        "        build(label='minimum', value=min(values)),\n"
        "        build(label='text', value=str(values)),\n"
        "        build(label='total', value=sum(values)),\n"
        "        build(label='items', value=tuple(values)),\n"
        "    )\n",
    )

    findings = SemanticMirrorWithoutDescentDetector().detect(
        parse_python_modules(tmp_path),
        DetectorConfig(),
    )

    assert not any(
        finding.metrics.plan_source_name == "BuiltinCallName" for finding in findings
    )


def test_enum_branch_over_structural_type_names_is_not_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "import ast\n"
        "from enum import StrEnum\n"
        "\n"
        "class BuiltinCallName(StrEnum):\n"
        "    DICT = 'dict'\n"
        "    LIST = 'list'\n"
        "    SET = 'set'\n"
        "\n"
        "def structural_kind(node):\n"
        "    if isinstance(node, (ast.Dict, ast.List, ast.Set)):\n"
        "        return 'collection'\n"
        "    return 'other'\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "BuiltinCallName"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_semantic_descent_treats_empty_enum_base_as_class_family_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class LabeledMode(StrEnum):\n"
        "    def label(self):\n"
        "        return self.value\n"
        "\n"
        "class CapabilityMode(LabeledMode):\n"
        "    FAST = 'fast'\n"
        "\n"
        "class ObservationMode(LabeledMode):\n"
        "    SLOW = 'slow'\n"
        "\n"
        "MODE_ENUMS = (CapabilityMode, ObservationMode)\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    authority = next(item for item in graph.authorities if item.name == "LabeledMode")
    assert authority.kind is SemanticAuthorityKind.CLASS_FAMILY
    assert any(
        graph.projection_catalog.projection_for_edge(certificate.edge).label
        == "MODE_ENUMS"
        and certificate.edge.authority_id == authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_enum_branch_with_only_weak_authority_name_affinity_is_not_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class BuiltinCallName(StrEnum):\n"
        "    LEN = 'len'\n"
        "    ISINSTANCE = 'isinstance'\n"
        "    TUPLE = 'tuple'\n"
        "\n"
        "def returned_sequence_name(value):\n"
        "    if isinstance(value, tuple) and len(value) == 1:\n"
        "        return 'sequence'\n"
        "    return None\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "BuiltinCallName"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_dataclass_branch_over_local_variable_names_is_not_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Edge:\n"
        "    left: int\n"
        "    right: int\n"
        "\n"
        "def compare(left, right):\n"
        "    if left is None or right is None:\n"
        "        return False\n"
        "    return left == right\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "Edge"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_dataclass_branch_over_field_name_literals_still_reports_mirror(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Edge:\n"
        "    left: int\n"
        "    right: int\n"
        "\n"
        "def compare(field_name):\n"
        "    if field_name in ('left', 'right'):\n"
        "        return True\n"
        "    return False\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert any(
        finding.metrics.plan_source_name == "Edge"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_dataclass_branch_using_qualified_authority_constructor_is_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Edge:\n"
        "    left: int\n"
        "    right: int\n"
        "\n"
        "    @classmethod\n"
        "    def from_indices(cls, left, right):\n"
        "        return cls(left, right)\n"
        "\n"
        "def check(edges, component):\n"
        "    for left, right in component:\n"
        "        if Edge.from_indices(left, right) not in edges:\n"
        "            return False\n"
        "    return True\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "Edge"
        and finding.metrics.plan_mapping_name.startswith("if@")
        for finding in findings
    )


def test_semantic_descent_ignores_function_local_variable_tuple_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RentAxes:\n"
        "    behavior_method_names: tuple[str, ...]\n"
        "    abstract_method_names: tuple[str, ...]\n"
        "    registry_projection_names: tuple[str, ...]\n"
        "    consumer_symbols: tuple[str, ...]\n"
        "\n"
        "def missing_axes(\n"
        "    behavior_method_names,\n"
        "    abstract_method_names,\n"
        "    registry_projection_names,\n"
        "    consumer_symbols,\n"
        "):\n"
        "    projection_rent_axes = (\n"
        "        behavior_method_names,\n"
        "        abstract_method_names,\n"
        "        registry_projection_names,\n"
        "        consumer_symbols,\n"
        "    )\n"
        "    return any(projection_rent_axes)\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert not any(
        finding.metrics.plan_source_name == "RentAxes"
        and finding.metrics.plan_mapping_name == "projection_rent_axes"
        for finding in findings
    )


def test_semantic_mirror_enum_subset_recipe_resolves_absolute_finding_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "taxonomy.py").write_text(
        "from enum import StrEnum\n"
        "\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n"
        "    LOW = 'low'\n",
        encoding="utf-8",
    )
    (package_dir / "codemod.py").write_text(
        "import pkg.taxonomy\n"
        "\n"
        "_ACTIONABLE_CONFIDENCE_LEVELS: frozenset[pkg.taxonomy.ConfidenceLevel] = "
        "frozenset(('high', 'medium'))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            "pkg/taxonomy.py": (package_dir / "taxonomy.py").read_text(
                encoding="utf-8"
            ),
            "pkg/codemod.py": (package_dir / "codemod.py").read_text(encoding="utf-8"),
        }
    )
    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            snapshot.parsed_modules,
            DetectorConfig(),
        )
    )
    finding = next(
        item
        for item in findings
        if item.detector_id == "semantic_mirror_without_descent"
        and item.metrics.plan_mapping_name == "_ACTIONABLE_CONFIDENCE_LEVELS"
    )
    absolute_finding = replace(
        finding,
        evidence=tuple(
            replace(
                location,
                file_path=(tmp_path / location.file_path).resolve().as_posix(),
            )
            for location in finding.evidence
        ),
    )

    plan = codemod_plan_from_findings(
        (absolute_finding,),
        selector_context=snapshot,
    )

    assert plan.records[0].status.value == "planned"
    import_operation = next(
        operation
        for operation in plan.document.recipes[0].operations
        if operation.operation_key() == "ensure_import"
    )
    assert import_operation.to_dict()["import_source"] == (
        "from pkg.taxonomy import ConfidenceLevel\n"
    )


def test_semantic_descent_treats_module_constructor_assignment_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Decoder:\n"
        "    type_label: str\n"
        "    projection: str\n"
        "    validation_error: str\n"
        "\n"
        "MANIFEST_STRING = Decoder[str](\n"
        "    type_label='string',\n"
        "    projection='text',\n"
        "    validation_error='bad',\n"
        ")\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    decoder_authority = next(
        authority for authority in graph.authorities if authority.name == "Decoder"
    )
    assert all(
        certificate.edge.authority_id != decoder_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    decoder_derivation = next(
        certificate
        for certificate in graph.certificates
        if certificate.status
        is semantic_descent_module.DescentStatus.DESCENDS_TO_AUTHORITY
        and certificate.edge.authority_id == decoder_authority.authority_id
    )
    assert isinstance(decoder_derivation, SemanticDerivationCertificate)
    assert decoder_derivation.proof_edges[0].edge_kind is (
        AuthorityProofEdgeKind.OWNS_FIELD_SET
    )
    decoder_projection = graph.projection_catalog.projection_for_edge(
        decoder_derivation.edge
    )
    assert decoder_projection.projection_constructions == (
        PresentationAuthorityConstruction(
            "Decoder",
            ("projection", "type_label", "validation_error"),
            ("Decoder",),
        ),
    )


def test_semantic_descent_treats_constructor_collection_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    action: str\n"
        "    rationale: str\n"
        "    target: str\n"
        "\n"
        "ACTION_TEMPLATES = (\n"
        "    RefactorAction(\n"
        "        action='pull_up', rationale='shared behavior', target='base'\n"
        "    ),\n"
        "    RefactorAction(\n"
        "        action='split', rationale='separate role', target='leaf'\n"
        "    ),\n"
        ")\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    action_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "RefactorAction"
    )
    assert all(
        certificate.edge.authority_id != action_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_class_family_construction_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC\n"
        "\n"
        "class PhysicalSourceEdit(ABC):\n"
        "    pass\n"
        "\n"
        "class SourceInsertion(PhysicalSourceEdit):\n"
        "    def __init__(self, file_path): self.file_path = file_path\n"
        "\n"
        "class SourceSpanReplacement(PhysicalSourceEdit):\n"
        "    def __init__(self, file_path): self.file_path = file_path\n"
        "\n"
        "def source_edits(file_path):\n"
        "    return (\n"
        "        SourceInsertion(file_path=file_path),\n"
        "        SourceSpanReplacement(file_path=file_path),\n"
        "    )\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "PhysicalSourceEdit"
    )

    assert not any(
        certificate.edge.authority_id == authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    assert any(
        certificate.edge.authority_id == authority.authority_id
        and certificate.proof_edges[0].edge_kind is AuthorityProofEdgeKind.INHERITS_FROM
        for certificate in graph.certificates
        if isinstance(certificate, SemanticDerivationCertificate)
    )


def test_semantic_descent_treats_declared_materializer_class_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Finding:\n"
        "    summary: str\n"
        "    evidence: tuple[str, ...]\n"
        "    scaffold: str | None = None\n"
        "    metrics: object | None = None\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class FindingRenderer:\n"
        "    target_finding_type: ClassVar[type[Finding]] = Finding\n"
        "    summary: object\n"
        "    evidence: object\n"
        "    scaffold: object | None = None\n"
        "    metrics: object | None = None\n"
        "\n"
        "finding_renderer = FindingRenderer(\n"
        "    summary=lambda candidate: candidate.summary,\n"
        "    evidence=lambda candidate: candidate.evidence,\n"
        "    scaffold=lambda candidate: candidate.scaffold,\n"
        "    metrics=lambda candidate: candidate.metrics,\n"
        ")\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    finding_authority = next(
        authority for authority in graph.authorities if authority.name == "Finding"
    )
    assert not any(
        certificate.edge.authority_id == finding_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_descriptor_literal_fields_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLineReference:\n"
        "    file_path: str\n"
        "    line: int\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLocation(SourceLineReference):\n"
        "    symbol: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class CandidateOrder:\n"
        "    file_path: str\n"
        "    line: int\n"
        "\n"
        "def project(attribute_name, instance):\n"
        "    return None\n"
        "\n"
        "class SourceLocationEvidenceProperty:\n"
        "    def __init__(self, file_attribute_name, line_attribute_name, symbol_attribute_name):\n"
        "        self.file_attribute_name = file_attribute_name\n"
        "        self.line_attribute_name = line_attribute_name\n"
        "        self.symbol_attribute_name = symbol_attribute_name\n"
        "\n"
        "    def __get__(self, instance, owner=None):\n"
        "        return SourceLocation(\n"
        "            project(self.file_attribute_name, instance),\n"
        "            project(self.line_attribute_name, instance),\n"
        "            project(self.symbol_attribute_name, instance),\n"
        "        )\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Candidate:\n"
        "    file_path: str\n"
        "    line: int\n"
        "    qualname: str\n"
        "\n"
        "    evidence = SourceLocationEvidenceProperty(\n"
        "        'file_path', 'line', 'qualname'\n"
        "    )\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    authority_ids = {
        authority.name: authority.authority_id
        for authority in graph.authorities
        if authority.name in {"SourceLineReference", "CandidateOrder"}
    }
    certificate_authority_ids = {
        certificate.edge.authority_id
        for certificate in graph.missing_descent_certificates
    }
    assert authority_ids["SourceLineReference"] not in certificate_authority_ids
    assert authority_ids["CandidateOrder"] not in certificate_authority_ids


def test_semantic_descent_ignores_local_constructor_assignment_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceTargetSpan:\n"
        "    qualname: str\n"
        "    line: int\n"
        "    end_line: int\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class AstTargetGeometryKey:\n"
        "    qualname: str\n"
        "    line: int\n"
        "    end_line: int\n"
        "\n"
        "def build_geometry(target):\n"
        "    geometry = AstTargetGeometryKey(\n"
        "        qualname=target.qualname,\n"
        "        line=target.line,\n"
        "        end_line=target.end_line,\n"
        "    )\n"
        "    return geometry\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    span_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "SourceTargetSpan"
    )
    assert all(
        certificate.edge.authority_id != span_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_requires_specific_affinity_for_small_dataclass_overlap(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class EnumSubsetSemanticMirrorRecipeParts:\n"
        "    projection: str\n"
        "    authority: str\n"
        "    selection: str\n"
        "    authority_source: str\n"
        "\n"
        "_RUNTIME_SEMANTIC_BRANCH_AXIS_TOKENS = frozenset(\n"
        "    ('projection', 'selection')\n"
        ")\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    parts_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "EnumSubsetSemanticMirrorRecipeParts"
    )
    assert all(
        certificate.edge.authority_id != parts_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_ignores_enum_member_keyed_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class Mode(StrEnum):\n"
        "    FAST = 'fast'\n"
        "    SLOW = 'slow'\n"
        "\n"
        "MODE_COST = {Mode.FAST: 1, Mode.SLOW: 2}\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    mode_authority = next(
        authority for authority in graph.authorities if authority.name == "Mode"
    )
    assert mode_authority.kind is SemanticAuthorityKind.ENUM
    assert not any(
        certificate.edge.authority_id == mode_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_instance_field_tuple_as_schema_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Observation:\n"
        "    observation_kind: str\n"
        "    execution_level: str\n"
        "    fiber_key: str\n"
        "\n"
        "def group_observations(observations):\n"
        "    grouped = {}\n"
        "    for observation in observations:\n"
        "        key = (\n"
        "            observation.observation_kind,\n"
        "            observation.execution_level,\n"
        "            observation.fiber_key,\n"
        "        )\n"
        "        grouped.setdefault(key, []).append(observation)\n"
        "    return grouped\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    observation_authority = next(
        authority for authority in graph.authorities if authority.name == "Observation"
    )
    assert observation_authority.kind is SemanticAuthorityKind.DATACLASS_SCHEMA
    assert not any(
        certificate.edge.authority_id == observation_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_dataclass_owned_payload_as_schema_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        payload = {\n"
        "            'backend': self.backend,\n"
        "            'parse_valid': self.parse_valid,\n"
        "        }\n"
        "        return payload\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    report_authority = next(
        authority for authority in graph.authorities if authority.name == "Report"
    )
    assert report_authority.kind is SemanticAuthorityKind.DATACLASS_SCHEMA
    assert not any(
        certificate.edge.authority_id == report_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    report_derivation = next(
        certificate
        for certificate in graph.certificates
        if certificate.status
        is semantic_descent_module.DescentStatus.DESCENDS_TO_AUTHORITY
        and certificate.edge.authority_id == report_authority.authority_id
    )
    assert isinstance(report_derivation, SemanticDerivationCertificate)
    assert report_derivation.proof_edges[0].edge_kind is (
        AuthorityProofEdgeKind.OWNS_FIELD_SET
    )


def test_semantic_descent_treats_dataclass_subclass_payload_as_schema_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "class DetailedReport(Report):\n"
        "    def to_dict(self):\n"
        "        payload = {\n"
        "            'backend': self.backend,\n"
        "            'parse_valid': self.parse_valid,\n"
        "        }\n"
        "        return payload\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    report_authority = next(
        authority for authority in graph.authorities if authority.name == "Report"
    )
    assert report_authority.kind is SemanticAuthorityKind.DATACLASS_SCHEMA
    assert not any(
        certificate.edge.authority_id == report_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    report_derivation = next(
        certificate
        for certificate in graph.certificates
        if certificate.status
        is semantic_descent_module.DescentStatus.DESCENDS_TO_AUTHORITY
        and certificate.edge.authority_id == report_authority.authority_id
    )
    assert isinstance(report_derivation, SemanticDerivationCertificate)
    assert report_derivation.proof_edges[0].edge_kind is (
        AuthorityProofEdgeKind.INHERITS_FROM
    )


def test_semantic_descent_reports_external_dataclass_payload_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "def payload(report):\n"
        "    row = {\n"
        "        'backend': report.backend,\n"
        "        'parse_valid': report.parse_valid,\n"
        "    }\n"
        "    return row\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    report_authority = next(
        authority for authority in graph.authorities if authority.name == "Report"
    )
    assert any(
        certificate.edge.authority_id == report_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_reports_constructor_catalog_schema_projection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "class ConstructorVariantCatalog:\n"
        "    def __init__(self, variants):\n"
        "        self.variants = variants\n"
        "\n"
        "class ConstructorVariantSpec:\n"
        "    def __init__(self, name, parameters, *, parameter_fields=(), derived_fields=(), constants=()):\n"
        "        pass\n"
        "\n"
        "class ConstructorDerivedField:\n"
        "    def __init__(self, name, callback):\n"
        "        pass\n"
        "\n"
        "class ConstructorConstant:\n"
        "    def __init__(self, name, value):\n"
        "        pass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RegistrationShape:\n"
        "    file_path: str\n"
        "    lineno: int\n"
        "    registry_name: str\n"
        "    registered_class: str\n"
        "    key_fingerprint: str\n"
        "    key_expression: str\n"
        "    registration_style: str\n"
        "\n"
        "REGISTRATION_SHAPE_CONSTRUCTORS = ConstructorVariantCatalog((\n"
        "    ConstructorVariantSpec(\n"
        "        'from_registration_call',\n"
        "        ('parsed_module', 'node', 'registry_name', 'registered_class', 'key_fingerprint'),\n"
        "        parameter_fields=('registry_name', 'registered_class', 'key_fingerprint'),\n"
        "        derived_fields=(\n"
        "            ConstructorDerivedField('file_path', lambda bound: bound['parsed_module'].path),\n"
        "            ConstructorDerivedField('lineno', lambda bound: bound['node'].lineno),\n"
        "            ConstructorDerivedField('key_expression', lambda bound: bound['node'].name),\n"
        "        ),\n"
        "        constants=(ConstructorConstant('registration_style', 'registration_call'),),\n"
        "    ),\n"
        "))\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    registration_shape_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "RegistrationShape"
    )
    certificate = next(
        item
        for item in graph.missing_descent_certificates
        if item.edge.authority_id == registration_shape_authority.authority_id
    )
    projection = graph.projection_catalog.projection_for_edge(certificate.edge)

    assert projection.label == "REGISTRATION_SHAPE_CONSTRUCTORS"
    assert set(certificate.edge.match.tokens) >= {
        "file_path",
        "lineno",
        "registry_name",
        "registered_class",
        "key_fingerprint",
        "key_expression",
        "registration_style",
    }


def test_semantic_descent_treats_constructor_guard_as_dataclass_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class FieldAccess:\n"
        "    source_expression: str\n"
        "    key_name: str\n"
        "    line: int\n"
        "\n"
        "def parse(node):\n"
        "    source_expression = extract_source(node)\n"
        "    key_name = extract_key(node)\n"
        "    if source_expression is None or key_name is None:\n"
        "        return None\n"
        "    return FieldAccess(\n"
        "        source_expression=source_expression,\n"
        "        key_name=key_name,\n"
        "        line=node.lineno,\n"
        "    )\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    field_access_authority = next(
        authority for authority in graph.authorities if authority.name == "FieldAccess"
    )
    assert not any(
        certificate.edge.authority_id == field_access_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_direct_dataclass_construction_as_sibling_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ReplacementSpan:\n"
        "    file_path: str\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "    replacement_lines: tuple[str, ...]\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BlankLineRun:\n"
        "    file_path: str\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "    blank_line_count: int\n"
        "\n"
        "def replacement(span):\n"
        "    return (\n"
        "        ReplacementSpan(\n"
        "            file_path=span.file_path,\n"
        "            start_line=span.start_line,\n"
        "            end_line=span.end_line,\n"
        "            replacement_lines=(),\n"
        "        ),\n"
        "    )\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    blank_line_authority = next(
        authority for authority in graph.authorities if authority.name == "BlankLineRun"
    )
    assert not any(
        certificate.edge.authority_id == blank_line_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_nominal_record_construction_as_complete_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "class RuntimeHandler:\n"
        "    @classmethod\n"
        "    def category_prefixes(cls):\n"
        "        return ()\n"
        "\n"
        "class CropHandler(RuntimeHandler):\n"
        "    pass\n"
        "\n"
        "class ImageHandler(RuntimeHandler):\n"
        "    pass\n"
        "\n"
        "class ObjectHandler(RuntimeHandler):\n"
        "    pass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RuntimeDialect:\n"
        "    category_prefixes_provider: object\n"
        "    feature_part_aliases: object\n"
        "    feature_part_aliases_provider: object\n"
        "    qualifier_tokens: frozenset[str]\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RuntimeLookupDialect:\n"
        "    category_prefixes_provider: object\n"
        "    feature_part_aliases: object\n"
        "    feature_part_aliases_provider: object\n"
        "\n"
        "RUNTIME_LOOKUP_DIALECT = RuntimeLookupDialect(\n"
        "    category_prefixes_provider=RuntimeHandler.category_prefixes,\n"
        "    feature_part_aliases={},\n"
        "    feature_part_aliases_provider=RuntimeHandler.category_prefixes,\n"
        ")\n"
        "RUNTIME_DIALECT = RuntimeDialect(\n"
        "    category_prefixes_provider=RuntimeHandler.category_prefixes,\n"
        "    feature_part_aliases={},\n"
        "    feature_part_aliases_provider=RuntimeHandler.category_prefixes,\n"
        "    qualifier_tokens=frozenset({'crop', 'image', 'object'}),\n"
        ")\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    nominal_record_projection_labels = {
        "RUNTIME_DIALECT",
        "RUNTIME_LOOKUP_DIALECT",
    }
    assert not any(
        graph.projection_catalog.projection_for_edge(certificate.edge).label
        in nominal_record_projection_labels
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_subclass_construction_as_schema_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BoundarySurface:\n"
        "    file_path: str\n"
        "    line: int\n"
        "    field_name: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class BoundaryUse(BoundarySurface):\n"
        "    symbol: str\n"
        "    use_kind: str\n"
        "    context_tokens: tuple[str, ...]\n"
        "\n"
        "class BoundaryUseVisitor:\n"
        "    def record(self, line, field_name, use_kind, context_tokens):\n"
        "        tokens = tuple(token for token in context_tokens if token != field_name)\n"
        "        key = (field_name, line, self.qualname, use_kind, tokens)\n"
        "        self.uses.append(\n"
        "            BoundaryUse(\n"
        "                file_path=self.file_path,\n"
        "                line=line,\n"
        "                field_name=field_name,\n"
        "                symbol=self.qualname,\n"
        "                use_kind=use_kind,\n"
        "                context_tokens=tokens,\n"
        "            )\n"
        "        )\n"
        "        return key\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    boundary_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "BoundarySurface"
    )
    assert not any(
        certificate.edge.authority_id == boundary_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_ignores_unrelated_partial_dataclass_vocabulary(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class AnalysisCacheEntryContext:\n"
        "    config: str\n"
        "    detector_registry: str\n"
        "    python_version: tuple[int, int]\n"
        "    schema: str\n"
        "\n"
        "class RegistryProjectionSurfaceAnalyzer:\n"
        "    role_terms = (\n"
        "        ('config_choices', ('config', 'schema')),\n"
        "        ('cli_choices', ('cli', 'option')),\n"
        "    )\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    context_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "AnalysisCacheEntryContext"
    )

    assert not any(
        certificate.edge.authority_id == context_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_treats_shared_dataclass_base_as_descent(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ZipDescriptorShape:\n"
        "    line_numbers_attribute_name: str\n"
        "    symbol_names_attribute_name: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RuntimeZipDescriptor(ZipDescriptorShape):\n"
        "    line_numbers_attribute_name: str\n"
        "    symbol_names_attribute_name: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ParsedZipDescriptor(ZipDescriptorShape):\n"
        "    file_attribute_name: str\n"
        "\n"
        "    @classmethod\n"
        "    def from_parts(cls, line_numbers_attribute_name, symbol_names_attribute_name):\n"
        "        if line_numbers_attribute_name is None or symbol_names_attribute_name is None:\n"
        "            return None\n"
        "        return cls('file_path', line_numbers_attribute_name, symbol_names_attribute_name)\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))
    shape_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "ZipDescriptorShape"
    )
    sibling_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "RuntimeZipDescriptor"
    )
    assert graph.class_index is not None
    resolver = SemanticMirrorResolver(
        authorities=graph.authorities,
        facts=graph.facts,
        projections=graph.projections,
        class_index=graph.class_index,
    )
    sibling_candidate_projection = next(
        projection
        for projection in graph.projections
        if sibling_authority.authority_id in resolver._matches_by_authority(projection)
    )

    assert not any(
        certificate.edge.authority_id == shape_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    assert not any(
        certificate.edge.authority_id == sibling_authority.authority_id
        for certificate in graph.certificates
    )
    assert not any(
        relation.authority_id == sibling_authority.authority_id
        and relation.projection_id == sibling_candidate_projection.projection_id
        for relation in graph.relations
    )


def test_semantic_descent_ignores_partial_schema_overlap_for_owned_payload(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RewriteDelta:\n"
        "    replacement_source: str\n"
        "    operation: str\n"
        "    rationale: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class OperationExample:\n"
        "    operation: str\n"
        "    rationale: str\n"
        "\n"
        "    def to_dict(self):\n"
        "        payload = {\n"
        "            'operation': self.operation,\n"
        "            'rationale': self.rationale,\n"
        "        }\n"
        "        return payload\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    rewrite_delta_authority = next(
        authority for authority in graph.authorities if authority.name == "RewriteDelta"
    )
    assert not any(
        certificate.edge.authority_id == rewrite_delta_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_ignores_low_specificity_full_schema_overlap(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RewriteDelta:\n"
        "    operation: str\n"
        "    rationale: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class OperationManifest:\n"
        "    operation: str\n"
        "    description: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class PlanItem:\n"
        "    target: str\n"
        "    rationale: str\n"
        "\n"
        "class ManifestRenderer:\n"
        "    def example_payload(self):\n"
        "        payload = {\n"
        "            'operation': self.operation,\n"
        "            'rationale': 'explain the edit',\n"
        "        }\n"
        "        return payload\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    rewrite_delta_authority = next(
        authority for authority in graph.authorities if authority.name == "RewriteDelta"
    )
    assert not any(
        certificate.edge.authority_id == rewrite_delta_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_ignores_prose_payload_literals(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class Mode(StrEnum):\n"
        "    FAST = 'fast'\n"
        "    SLOW = 'slow'\n"
        "\n"
        "MODE_DOCS = {\n"
        "    'title': 'Fast Mode And Slow Mode',\n"
        "    'why': 'Use fast mode when the slow mode branch is too expensive.',\n"
        "}\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    mode_authority = next(
        authority for authority in graph.authorities if authority.name == "Mode"
    )
    assert not any(
        certificate.edge.authority_id == mode_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_keeps_uppercase_enum_member_identity(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from enum import StrEnum\n"
        "\n"
        "class PatternId(StrEnum):\n"
        "    ABC_FAMILY = 'abc_family'\n"
        "    CONFIG_DISPATCH = 'config_dispatch'\n"
        "\n"
        "class ObservationKind(StrEnum):\n"
        "    LITERAL_DISPATCH = 'literal_dispatch'\n"
        "    CLASS_MARKER = 'class_marker'\n"
        "\n"
        "class ActionBuilderId(StrEnum):\n"
        "    ABC_FAMILY = 'abc_family'\n"
        "    CONFIG_DISPATCH = 'config_dispatch'\n"
        "\n"
        "PATTERN_SPECS = {\n"
        "    PatternId.ABC_FAMILY: object(),\n"
        "    PatternId.CONFIG_DISPATCH: object(),\n"
        "}\n",
    )

    graph = build_semantic_descent_graph(parse_python_modules(tmp_path))

    observation_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "ObservationKind"
    )
    assert not any(
        certificate.edge.authority_id == observation_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )
    action_authority = next(
        authority
        for authority in graph.authorities
        if authority.name == "ActionBuilderId"
    )
    assert not any(
        certificate.edge.authority_id == action_authority.authority_id
        for certificate in graph.missing_descent_certificates
    )


def test_semantic_descent_graph_cache_invalidates_on_source_change(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
    )
    parse_cache_dir = tmp_path / ".nra-cache" / "ast"
    graph_cache_dir = tmp_path / ".nra-cache" / "semantic_descent"

    first_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path, cache_dir=parse_cache_dir),
        cache_dir=graph_cache_dir,
    )
    assert not first_graph.missing_descent_certificates
    assert tuple(graph_cache_dir.glob("*.pickle"))

    module_path.write_text(
        "class Step:\n"
        "    pass\n"
        "\n"
        "class LoadStep(Step):\n"
        "    step_id = 'load'\n"
        "\n"
        "class SaveStep(Step):\n"
        "    step_id = 'save'\n"
        "\n"
        "STEP_TABLE = {'load': LoadStep, 'save': SaveStep}\n",
        encoding="utf-8",
    )

    second_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path, cache_dir=parse_cache_dir),
        cache_dir=graph_cache_dir,
    )

    assert any(
        second_graph.projection_catalog.projection_for_edge(certificate.edge).label
        == "STEP_TABLE"
        and "authority registry or subclass family"
        in certificate.missing_derivation_path
        for certificate in second_graph.missing_descent_certificates
    )


def test_semantic_descent_graph_cache_preserves_positive_proof_relations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'backend': self.backend,\n"
        "            'parse_valid': self.parse_valid,\n"
        "        }\n",
    )
    graph_cache_dir = tmp_path / ".nra-cache" / "semantic_descent"
    first_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path, use_parse_cache=False),
        cache_dir=graph_cache_dir,
    )
    positive_certificate = next(
        certificate
        for certificate in first_graph.certificates
        if certificate.status
        is semantic_descent_module.DescentStatus.DESCENDS_TO_AUTHORITY
    )
    assert isinstance(positive_certificate, SemanticDerivationCertificate)

    def unexpected_graph_rebuild(*args, **kwargs):
        del args, kwargs
        raise AssertionError("positive relations must load from the graph cache")

    monkeypatch.setattr(
        semantic_descent_module,
        "_build_semantic_descent_graph_cached",
        unexpected_graph_rebuild,
    )
    cached_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path, use_parse_cache=False),
        cache_dir=graph_cache_dir,
    )

    assert cached_graph.relations == first_graph.relations
    assert cached_graph.certificates == first_graph.certificates
    assert cached_graph.certificates[0].to_dict()["status"] == ("descends_to_authority")


def test_semantic_descent_graph_rebase_moves_positive_proof_paths(
    tmp_path: Path,
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Report:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "    def to_dict(self):\n"
        "        return {\n"
        "            'backend': self.backend,\n"
        "            'parse_valid': self.parse_valid,\n"
        "        }\n"
    )
    source_root = tmp_path / "checkout-a"
    target_root = tmp_path / "checkout-b"
    _write_module(source_root, source)
    target_module = _write_module(target_root, source)
    graph = build_semantic_descent_graph(
        parse_python_modules(source_root, use_parse_cache=False)
    )

    rebased_graph = rebase_semantic_descent_graph(
        graph,
        (str(source_root),),
        (str(target_root),),
    )
    proof = next(
        certificate.proof_edges[0]
        for certificate in rebased_graph.certificates
        if isinstance(certificate, SemanticDerivationCertificate)
    )

    assert proof.file_path == str(target_module)
