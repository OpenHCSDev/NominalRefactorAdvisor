from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import nominal_refactor_advisor.semantic_descent as semantic_descent_module
import nominal_refactor_advisor.semantic_refactor_gate as semantic_refactor_gate_module
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.detectors import (
    _semantic_descent as semantic_descent_detectors,
)
from nominal_refactor_advisor.analysis import analyze_modules, analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodPlanDocument,
    CodemodPreflightStatus,
    CodemodSourceContext,
    CodemodSourceSnapshot,
    DeriveClassFamilyCollectionOperation,
    DeriveDataclassConstructorProjectionOperation,
    DeriveDataclassFieldNameCollectionProjectionOperation,
    DeriveDataclassKeyValueSequenceProjectionOperation,
    DeriveDataclassPayloadProjectionOperation,
    DeriveEnumSubsetOperation,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.detectors import (
    DetectorCacheGranularity,
    DetectorConfig,
    InheritedAutoRegisterConfigBoilerplateDetector,
    SemanticMirrorWithoutDescentDetector,
)
from nominal_refactor_advisor.models import (
    BranchCountMetrics,
    ConstructorOwnedMappingMetrics,
    EmptyFindingMetrics,
    MappingMetrics,
    ParameterThreadMetrics,
    ProbeCountMetrics,
    RefactorFinding,
    RegistrationMetrics,
    RepeatedMethodMetrics,
    SemanticMirrorMetricRelation,
    SourceLocation,
    WitnessCarrierMetrics,
)
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    AuthorityClaimResolution,
    AuthorityClaimStatus,
    AuthorityClaimProvenance,
    AuthorityDiscoveryRequired,
    AuthorityProofEdge,
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
from nominal_refactor_advisor.semantic_refactor_gate import (
    DescentCertificateFindingAuthority,
    SemanticRefactorBoundaryEvidence,
)


def _write_module(root: Path, source: str) -> Path:
    path = root / "pkg" / "mod.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_class_name_algebra_uses_tokens_for_common_suffixes() -> None:
    assert (
        CLASS_NAME_ALGEBRA.longest_common_suffix(("AlphaResult", "BetaResult"))
        == "Result"
    )


def test_presentation_kinds_own_their_projection_policies() -> None:
    assert PresentationProjectionKind.CALL_LITERAL.class_family_call_policy_applies
    assert (
        not PresentationProjectionKind.COLLECTION_LITERAL.admits_function_local_tokens(
            has_string_literal=False
        )
    )
    assert PresentationProjectionKind.COLLECTION_LITERAL.admits_function_local_tokens(
        has_string_literal=True
    )
    assert PresentationProjectionKind.MAPPING_LITERAL.has_structured_key_source(
        has_key_value_pairs=True
    )
    assert (
        PresentationProjectionKind.BRANCH_LITERAL.low_coverage_authority_affinity_required
    )
    assert PresentationProjectionKind.BRANCH_LITERAL.is_branch_like
    assert PresentationProjectionKind.MATCH_LITERAL.is_branch_like
    assert not (
        PresentationProjectionKind.MATCH_LITERAL.low_coverage_authority_affinity_required
    )
    assert (
        semantic_descent_module.PresentationTokenKind.STRING_LITERAL.is_string_literal
    )
    assert (
        semantic_descent_module.PresentationTokenKind.QUALIFIED_ATTRIBUTE.is_qualified_attribute
    )


def test_semantic_authority_mirror_policy_registry_covers_authority_kinds() -> None:
    assert SemanticAuthorityMirrorPolicy.registered_authority_kinds() == frozenset(
        SemanticAuthorityKind
    )


def test_semantic_authority_mirror_policy_mro_owns_metric_projection() -> None:
    relation = SemanticMirrorMetricRelation(
        fact_names=("Alpha", "Beta"),
        projection_name="HANDLERS",
        authority_name="Handler",
        identity_field_names=("alpha", "beta"),
        class_key_pairs=("Alpha='alpha'", "Beta='beta'"),
    )
    metric_types_by_kind = {
        authority_kind: type(policy_type().semantic_mirror_metrics(relation))
        for authority_kind, policy_type in SemanticAuthorityMirrorPolicy.__registry__.items()
    }

    assert metric_types_by_kind == {
        SemanticAuthorityKind.CLASS_FAMILY: RegistrationMetrics,
        SemanticAuthorityKind.AUTOREGISTER_FAMILY: RegistrationMetrics,
        SemanticAuthorityKind.DATACLASS_SCHEMA: MappingMetrics,
        SemanticAuthorityKind.ENUM: MappingMetrics,
        SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY: MappingMetrics,
    }
    assert not hasattr(SemanticAuthorityKind.CLASS_FAMILY, "uses_registration_metrics")


def test_metric_declarations_own_finding_backed_fact_projection() -> None:
    assert EmptyFindingMetrics().semantic_authority_name_candidates() == ()
    assert EmptyFindingMetrics().semantic_fact_names() == ()
    assert RepeatedMethodMetrics(
        duplicate_site_count=2,
        statement_count=3,
        class_count=2,
        method_symbols=("Alpha.run", "Beta.run"),
    ).semantic_fact_names() == ("Alpha", "Beta")
    assert WitnessCarrierMetrics(
        class_count=2,
        shared_role_count=2,
        class_names=("Alpha", "Beta"),
        shared_role_names=("encode", "decode"),
    ).semantic_fact_names() == ("encode", "decode")
    assert MappingMetrics.from_field_names(
        mapping_site_count=2,
        field_names=(),
        identity_field_names=("mode",),
    ).semantic_fact_names() == ("mode",)
    assert RegistrationMetrics.from_class_names(
        registration_site_count=2,
        class_names=("Alpha", "Beta"),
    ).semantic_fact_names() == ("Alpha", "Beta")
    assert BranchCountMetrics(
        branch_site_count=2,
        literal_cases=("alpha", "beta"),
    ).semantic_fact_names() == ("alpha", "beta")
    assert ProbeCountMetrics(probe_site_count=2).semantic_fact_names() == ()
    assert ParameterThreadMetrics(
        function_count=2,
        shared_parameter_count=2,
        shared_parameter_names=("context", "config"),
    ).semantic_fact_names() == ("context", "config")


def test_authority_claim_status_owns_actionability_and_resolution_shape() -> None:
    assert AuthorityClaimStatus.RESOLVED.is_actionable is True
    assert AuthorityClaimStatus.DECLARED.is_actionable is True
    assert AuthorityClaimStatus.AMBIGUOUS.requires_discovery is True
    assert AuthorityClaimStatus.UNRESOLVED.requires_discovery is True

    claim = AuthorityClaim(claimed_symbol="AlphaAuthority")
    proof_edges = tuple(
        AuthorityProofEdge(
            edge_kind=edge_kind,
            authority_id="alpha-authority",
            authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            file_path="pkg/mod.py",
            line=line,
            symbol="AlphaAuthority",
        )
        for line, edge_kind in enumerate(
            (
                AuthorityProofEdgeKind.SOURCE_INDEX_TARGET,
                AuthorityProofEdgeKind.SEMANTIC_DESCENT_GRAPH,
            ),
            start=1,
        )
    )

    resolution = AuthorityClaimResolution.from_proof_edges(
        claim,
        proof_edges,
        searched_symbols=claim.searched_symbols,
        ambiguity_reason="not used for a unique authority",
    )

    assert resolution.status is AuthorityClaimStatus.RESOLVED
    assert resolution.proof_edges == proof_edges
    assert resolution.discovery_required is None

    with pytest.raises(ValueError, match="ambiguous authority resolution"):
        AuthorityClaimResolution(
            claim=claim,
            status=AuthorityClaimStatus.AMBIGUOUS,
            proof_edges=proof_edges,
            discovery_required=AuthorityDiscoveryRequired(
                claimed_symbol=claim.claimed_symbol,
                searched_symbols=claim.searched_symbols,
                candidate_count=1,
                reason="one identity cannot be ambiguous",
            ),
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


def test_dataclass_semantic_authority_uses_the_nominal_class_declaration(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass as dc\n"
        "\n"
        "def dataclass(cls):\n"
        "    return cls\n"
        "\n"
        "@dataclass\n"
        "class FalseProduct:\n"
        "    left: object\n"
        "    right: object\n"
        "\n"
        "@dc\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n",
    )

    graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path),
        use_cache=False,
    )

    assert not any(authority.name == "FalseProduct" for authority in graph.authorities)
    product = next(
        authority for authority in graph.authorities if authority.name == "Product"
    )
    assert product.kind is SemanticAuthorityKind.DATACLASS_SCHEMA


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
        "REPORT = Report(backend='cpython', parse_valid=True)\n"
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
    assert projection.label == "Manifest.to_dict:return"
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
    original_renderer = detector._finding_for_resolved_certificate

    def counted_renderer(resolved):
        rendered_certificates.append(resolved)
        return original_renderer(resolved)

    monkeypatch.setattr(
        detector,
        "_finding_for_resolved_certificate",
        counted_renderer,
    )
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

    graph = build_finding_backed_semantic_descent_graph((finding,))

    authority = graph.authorities[0]
    projection = graph.projections[0]
    certificate = graph.missing_descent_certificates[0]
    authority_evidence = (
        semantic_descent_module.FindingBackedAuthorityProjection.authority_evidence(
            finding
        )
    )

    assert authority.name == "Step"
    assert finding.authority_evidence == finding.evidence[1]
    assert isinstance(
        authority_evidence,
        semantic_descent_module.DetectorSourceFindingAuthorityEvidence,
    )
    assert authority.kind is SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY
    assert (
        authority.claim_provenance is AuthorityClaimProvenance.DETECTOR_SOURCE_EVIDENCE
    )
    assert projection.kind is PresentationProjectionKind.DETECTOR_FINDING
    assert projection.projection_id == semantic_descent_finding_projection_id(finding)
    assert projection.source_text == finding.stable_id
    assert certificate.edge.authority_id == authority.authority_id
    assert certificate.edge.projection_id == projection.projection_id
    assert certificate.missing_derivation_path == finding.relation_context
    assert {fact.name for fact in graph.facts} == {"LoadStep", "SaveStep"}


def test_finding_authority_evidence_must_belong_to_its_evidence() -> None:
    with pytest.raises(ValueError, match="must also occur in evidence"):
        RefactorFinding(
            detector_id="authority_projection_test",
            pattern_id=PatternId.NOMINAL_BOUNDARY,
            title="Authority evidence",
            summary="authority evidence is detached from the finding",
            why="detached evidence cannot prove the finding's authority",
            capability_gap="one self-contained authority witness",
            relation_context="authority provenance must travel with the finding",
            evidence=(SourceLocation("pkg/mod.py", 7, "local_cases"),),
            authority_evidence=SourceLocation("pkg/model.py", 3, "AxisRole"),
        )


def test_finding_certificate_authority_requires_exact_graph_membership() -> None:
    finding = RefactorFinding(
        detector_id="authority_projection_test",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Authority evidence",
        summary="finding is absent from the supplied graph",
        why="a different graph cannot certify this finding",
        capability_gap="one exact finding-backed certificate",
        relation_context="finding and graph membership must agree",
    )
    certificate_authority = DescentCertificateFindingAuthority(
        build_finding_backed_semantic_descent_graph(())
    )

    with pytest.raises(ValueError, match="requires exactly one"):
        certificate_authority.resolved_certificate_for_finding(finding)
    assert not hasattr(
        semantic_refactor_gate_module,
        "SemanticRefactorFindingGroupAuthority",
    )


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

    graph = build_finding_backed_semantic_descent_graph((finding,))
    authority = graph.authorities[0]
    certificate = graph.missing_descent_certificates[0]
    authority_evidence = (
        semantic_descent_module.FindingBackedAuthorityProjection.authority_evidence(
            finding
        )
    )

    assert authority.name == "AxisRoleAuthority"
    assert isinstance(
        authority_evidence,
        semantic_descent_module.InferredProjectionFindingAuthorityEvidence,
    )
    assert authority.kind is SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY
    assert (
        authority.claim_provenance is AuthorityClaimProvenance.INFERRED_FROM_PROJECTION
    )
    assert {fact.name for fact in graph.facts} == {"alpha", "beta"}
    assert certificate.missing_derivation_path == finding.relation_context


def test_constructor_mapping_metrics_own_the_constructor_authority() -> None:
    finding = RefactorFinding(
        detector_id="repeated_builder_calls",
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated construction",
        summary="consumer methods repeat one nominal constructor",
        why="the constructor owns its field schema",
        capability_gap="one constructor-owned builder",
        relation_context="constructor calls bypass their nominal authority",
        evidence=(SourceLocation("pkg/mod.py", 7, "Builder.first:Result"),),
        metrics=ConstructorOwnedMappingMetrics.from_field_names(
            mapping_site_count=3,
            mapping_name="Result",
            source_name="self",
            field_names=("reason", "owner_type"),
        ),
    )

    graph = build_finding_backed_semantic_descent_graph((finding,))

    assert graph.authorities[0].name == "Result"
    assert {fact.name for fact in graph.facts} == {"reason", "owner_type"}
    assert not hasattr(
        semantic_descent_module,
        "FindingMetricsSemanticProjection",
    )


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

    graph = build_finding_backed_semantic_descent_graph((finding,))

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

    graph = build_finding_backed_semantic_descent_graph((first, second))

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
    graph = build_finding_backed_semantic_descent_graph((finding,))

    certificate_authority = DescentCertificateFindingAuthority(graph)
    boundary = SemanticRefactorBoundaryEvidence.from_ssot_finding_group(
        (finding,),
        certificate_authority=certificate_authority,
    )

    assert boundary.group_key.authority_label == "AxisRoleAuthority"
    assert boundary.group_key.descent_path == finding.relation_context
    assert boundary.evidence_symbols == ("local_cases",)
    assert boundary.certificate_count == 1
    assert boundary.matched_fact_count == 2
    assert boundary.authority_kinds == (
        SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY,
    )
    assert boundary.projection_kinds == (PresentationProjectionKind.DETECTOR_FINDING,)
    certificate_selection = certificate_authority.resolved_selection_for_findings(
        (finding, finding)
    )
    assert len(certificate_selection.certificates) == 1
    assert len(certificate_selection.authority_claims) == 1


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
        "PRODUCT_RECORD_OPTION_FIELDS = ('bases', 'defaults', 'doc')\n"
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
        finding.metrics.plan_mapping_name == "PRODUCT_RECORD_OPTION_FIELDS"
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
    simulation = plan.simulate(snapshot)
    operation = json_report_object(plan.document)["recipes"][0]["operations"][0]
    record = plan.records[0]

    assert plan.expected_removed_finding_count == 1
    assert record.detector_id == "semantic_mirror_without_descent"
    assert record.status.value == "executable_candidate"
    assert (
        record.evaluation_declaration_name == "RegistrationSemanticMirrorRecipeStrategy"
    )
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert set(operation) == {"operation", "target_id", "rationale"}
    assert operation["target_id"]
    recipe = plan.document.recipes[0]
    assert recipe.authority_claims == ()
    declared_claims = recipe.declared_authority_claims(snapshot)
    assert len(declared_claims) == 1
    assert declared_claims[0].claimed_symbol == "Step"
    assert declared_claims[0].authority_kind is (
        SemanticAuthorityKind.AUTOREGISTER_FAMILY
    )
    assert declared_claims[0].authority_id
    assert record.action_keys
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
    snapshot = context.snapshot_for_findings((alpha_finding,), parse_workers=4)
    parallel_snapshot = context.snapshot_for_findings(
        (beta_finding, alpha_finding),
        parse_workers=4,
    )

    assert tuple(snapshot.module_node_cache or {}) == (alpha_path.as_posix(),)
    assert tuple(parallel_snapshot.module_node_cache or {}) == (
        alpha_path.as_posix(),
        beta_path.as_posix(),
    )
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
    simulation = plan.simulate(snapshot)
    operations = tuple(
        json_report_object(operation) for operation in plan.document.recipes[0].operations
    )
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "executable_candidate"
    assert (
        plan.records[0].evaluation_declaration_name
        == "InheritedAutoRegisterConfigBoilerplateDetector"
    )
    assert plan.records[0].refactor_concept == "auto_register"
    assert len(operations) == 1
    assert set(operations[0]) == {"operation", "target_id", "rationale"}
    assert operations[0]["operation"] == (
        "delete_inherited_auto_register_configuration"
    )
    assert operations[0]["target_id"]
    assert rewritten.count("    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE") == 1
    assert rewritten.count("    __key_extractor__ = class_name_registry_key") == 1
    assert rewritten.count("    __skip_if_no_key__ = True") == 1
    assert simulation.is_clean is True


def test_inherited_autoregister_config_replay_reproves_ancestor_values(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "pkg"
    package_path.mkdir()
    base_path = package_path / "base.py"
    base_path.write_text(
        "class RegisteredStrategy:\n"
        "    __registry_key__ = 'kind'\n"
        "    __skip_if_no_key__ = True\n",
        encoding="utf-8",
    )
    strategy_path = package_path / "strategy.py"
    strategy_path.write_text(
        "from metaclass_registry import AutoRegisterMeta\n"
        "from .base import RegisteredStrategy\n"
        "\n"
        "class Strategy(RegisteredStrategy, metaclass=AutoRegisterMeta):\n"
        "    __registry_key__ = 'kind'\n"
        "    __skip_if_no_key__ = True\n"
        "\n"
        "    def run(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    initial_modules = parse_python_modules(tmp_path)
    finding = next(
        finding
        for finding in InheritedAutoRegisterConfigBoilerplateDetector().detect(
            initial_modules,
            DetectorConfig(),
        )
        if finding.detector_id == "inherited_autoregister_config_boilerplate"
    )
    initial_snapshot = CodemodSourceSnapshot.from_modules(
        initial_modules,
        (finding,),
    )
    document_payload = json_report_object(
        initial_snapshot.plan_from_findings((finding,)).document
    )

    base_path.write_text(
        "class RegisteredStrategy:\n"
        "    __registry_key__ = 'strategy_kind'\n"
        "    __skip_if_no_key__ = False\n",
        encoding="utf-8",
    )
    current_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="no AutoRegister configuration repeated from an ancestor",
    ):
        CodemodPlanDocument.from_json_value(document_payload).simulate(
            current_snapshot
        )


def test_autoregister_priority_ordering_synthesizes_one_proven_mro_batch(
    tmp_path: Path,
) -> None:
    source = (
        "from abc import ABC, abstractmethod\n"
        "from enum import Enum\n"
        "from typing import ClassVar\n"
        "from metaclass_registry import AutoRegisterMeta\n"
        "\n"
        "class PhaseKey(str, Enum):\n"
        "    FIRST = 'first'\n"
        "    SECOND = 'second'\n"
        "    FINAL = 'final'\n"
        "\n"
        "\n"
        "class Phase(ABC, metaclass=AutoRegisterMeta):\n"
        "    __registry__: ClassVar[dict[PhaseKey, type['Phase']]] = {}\n"
        "    __registry_key__ = 'phase'\n"
        "    __skip_if_no_key__ = True\n"
        "    phase: ClassVar[PhaseKey | None] = None\n"
        "    priority: ClassVar[int]\n"
        "\n"
        "    @classmethod\n"
        "    def ordered_types(cls) -> tuple[type['Phase'], ...]:\n"
        "        return tuple(\n"
        "            sorted(\n"
        "                cls.__registry__.values(),\n"
        "                key=lambda candidate: candidate.priority,\n"
        "            )\n"
        "        )\n"
        "\n"
        "    @abstractmethod\n"
        "    def matches(self, value: str) -> bool:\n"
        "        raise NotImplementedError\n"
        "\n"
        "\n"
        "class FirstPhase(Phase):\n"
        "    phase = PhaseKey.FIRST\n"
        "    priority = 10\n"
        "\n"
        "    def matches(self, value: str) -> bool:\n"
        "        return value == self.phase\n"
        "\n"
        "\n"
        "class TerminalPhase(Phase):\n"
        "    pass\n"
        "\n"
        "\n"
        "class SecondPhase(TerminalPhase):\n"
        "    phase = PhaseKey.SECOND\n"
        "    priority = 20\n"
        "\n"
        "    def matches(self, value: str) -> bool:\n"
        "        return value == self.phase\n"
        "\n"
        "\n"
        "class FinalPhase(TerminalPhase):\n"
        "    phase = PhaseKey.FINAL\n"
        "    priority = 100\n"
        "\n"
        "    def matches(self, value: str) -> bool:\n"
        "        return value == self.phase\n"
    )
    module_path = _write_module(tmp_path, source)
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in analyze_modules(modules)
        if item.detector_id == "autoregister_explicit_priority_ordering"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate(snapshot)
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))
    operations = tuple(
        json_report_object(operation) for operation in plan.document.recipes[0].operations
    )
    projected_findings = analyze_modules(
        snapshot.with_virtual_sources(
            simulation.simulation.rewritten_sources
        ).parsed_modules,
        DetectorConfig(),
    )
    namespace: dict[str, object] = {}
    exec(rewritten, namespace)
    ordered_types = namespace["Phase"].ordered_types()

    assert plan.records[0].status.value == "executable_candidate"
    assert (
        plan.records[0].evaluation_declaration_name
        == "AutoRegisterExplicitPriorityOrderingDetector"
    )
    assert plan.records[0].refactor_concept == "auto_register_mro_ordering"
    assert [operation["operation"] for operation in operations] == [
        "derive_auto_register_mro_ordering"
    ]
    assert set(operations[0]) == {
        "operation",
        "target_id",
        "rationale",
    }
    assert not {
        "priority_field_name",
        "sorted_call_source",
        "resolution_class_name",
        "resolution_class_source",
    }.intersection(operations[0])
    assert len(plan.records[0].action_keys) == 4
    recipe = plan.document.recipes[0]
    assert recipe.authority_claims == ()
    declared_claims = recipe.declared_authority_claims(snapshot)
    assert len(declared_claims) == 1
    assert declared_claims[0].claimed_symbol == "_PhaseResolutionMro"
    assert declared_claims[0].authority_kind is SemanticAuthorityKind.CLASS_FAMILY
    assert declared_claims[0].file_path == module_path.as_posix()
    assert declared_claims[0].qualname == "_PhaseResolutionMro"
    authority_report = recipe.authority_claim_preflight_report(snapshot)
    assert authority_report is not None
    assert authority_report.status is CodemodPreflightStatus.PASSED
    assert authority_report.detail.resolutions[0].status.value == "declared"
    assert type(RefactorRecipeOperation.from_json_value(operations[0])).__name__ == (
        "DeriveAutoRegisterMroOrderingOperation"
    )
    assert "priority" not in rewritten
    assert (
        "class _PhaseResolutionMro(\n"
        "    FirstPhase,\n"
        "    SecondPhase,\n"
        "    FinalPhase,\n"
        "):" in rewritten
    )
    assert (
        "        return tuple(\n"
        "            _PhaseResolutionMro.registered_types()\n"
        "        )" in rewritten
    )
    assert [item.__name__ for item in ordered_types] == [
        "FirstPhase",
        "SecondPhase",
        "FinalPhase",
    ]
    assert not any(
        item.detector_id == "autoregister_explicit_priority_ordering"
        for item in projected_findings
    )
    assert simulation.is_clean is True
    replay = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).simulate(snapshot)
    assert (
        replay.simulation.rewritten_sources == simulation.simulation.rewritten_sources
    )

    module_path.write_text(
        source.replace("    priority = 10\n", "    priority = 30\n").replace(
            "    priority = 20\n",
            "    priority = 10\n",
        ),
        encoding="utf-8",
    )
    reprioritized_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )
    reprioritized = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).simulate(reprioritized_snapshot)
    reprioritized_source = next(
        iter(reprioritized.simulation.rewritten_sources.values())
    )
    assert (
        "class _PhaseResolutionMro(\n"
        "    SecondPhase,\n"
        "    FirstPhase,\n"
        "    FinalPhase,\n"
        "):" in reprioritized_source
    )
    reprioritized_namespace: dict[str, object] = {}
    exec(reprioritized_source, reprioritized_namespace)
    assert [
        item.__name__ for item in reprioritized_namespace["Phase"].ordered_types()
    ] == ["SecondPhase", "FirstPhase", "FinalPhase"]

    module_path.write_text(
        source.replace("    priority = 20\n", "    priority = 10\n"),
        encoding="utf-8",
    )
    duplicate_priority_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )
    duplicate_priority_preflight = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).preflight_snapshot(duplicate_priority_snapshot)
    assert duplicate_priority_preflight.preflight_failed is True
    assert duplicate_priority_preflight.reports[-1].operation == (
        "derive_auto_register_mro_ordering"
    )

    def plan_for_source(source_root: Path, source_text: str):
        _write_module(source_root, source_text)
        source_modules = parse_python_modules(source_root)
        source_finding = next(
            item
            for item in analyze_modules(source_modules)
            if item.detector_id == "autoregister_explicit_priority_ordering"
        )
        source_snapshot = CodemodSourceSnapshot.from_modules(
            source_modules,
            (source_finding,),
        )
        return codemod_plan_from_findings(
            (source_finding,),
            selector_context=source_snapshot,
        )

    registered_ancestor_root = tmp_path / "registered_ancestor"
    unsafe_plan = plan_for_source(
        registered_ancestor_root,
        source.replace(
            "class TerminalPhase(Phase):\n    pass\n",
            "class TerminalPhase(Phase):\n"
            "    phase = PhaseKey.FIRST\n"
            "    priority = 15\n"
            "\n"
            "    def matches(self, value: str) -> bool:\n"
            "        return value == self.phase\n",
        ),
    )

    assert unsafe_plan.records[0].status.value == "rejected_by_safety_check"
    assert "incomparable single-inheritance leaves" in unsafe_plan.records[0].reason

    open_key_plan = plan_for_source(
        tmp_path / "open_key",
        source.replace("PhaseKey | None", "str | None")
        .replace("PhaseKey.FIRST", "'first'")
        .replace("PhaseKey.SECOND", "'second'")
        .replace("PhaseKey.FINAL", "'final'"),
    )

    assert open_key_plan.records[0].status.value == "rejected_by_safety_check"
    assert "exhaust one local enum key" in open_key_plan.records[0].reason

    custom_key_plan = plan_for_source(
        tmp_path / "custom_key",
        source.replace(
            "    __skip_if_no_key__ = True\n",
            "    __skip_if_no_key__ = True\n"
            "    __key_extractor__ = staticmethod(lambda name, cls: name)\n",
            1,
        ),
    )

    assert custom_key_plan.records[0].status.value == "rejected_by_safety_check"
    assert "without a custom key extractor" in custom_key_plan.records[0].reason


def test_repeated_builder_call_rejects_identity_constructor_wrapper(
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
    assert isinstance(finding.metrics, ConstructorOwnedMappingMetrics)
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    assert plan.records[0].status.value == "rejected_by_safety_check"
    assert "requires a source projection or invariant selector axis" in (
        plan.records[0].reason
    )
    assert plan.document.recipes == ()


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
        "    @property\n"
        "    def selected_qualnames(self):\n"
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
    simulation = plan.simulate(snapshot)
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "executable_candidate"
    assert "def for_function_or_method(" in rewritten
    assert "file_path: str" in rewritten
    assert "qualname: str" in rewritten
    assert "file_paths=(file_path,)" in rewritten
    assert "file_paths=(source_path,)" not in rewritten
    assert "    @classmethod\n    def for_function_or_method(" in rewritten
    assert "    @property\n    @classmethod" not in rewritten
    assert rewritten.count("TargetSelector.for_function_or_method(") == 3
    assert simulation.is_clean is True


def test_repeated_builder_call_ignores_varying_owned_method_calls(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class Renderer:\n"
        "    def emit(\n"
        "        self, name: str, enabled: bool = False, style: str = 'plain'\n"
        "    ) -> str:\n"
        "        return name if enabled else f'{style}:{name.lower()}'\n"
        "\n"
        "    def build(self):\n"
        "        first = self.emit(name='alpha')\n"
        "        second = self.emit(name='beta', enabled=True)\n"
        "        third = self.emit(name='gamma', style='compact')\n"
        "        fourth = self.emit(name='delta', enabled=True, style='wide')\n"
        "        return first, second, third, fourth\n",
    )

    assert not any(
        item.detector_id == "repeated_builder_calls" for item in analyze_path(tmp_path)
    )


def test_repeated_builder_call_ignores_existing_owned_method_authority(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "class RuntimeCache:\n"
        "    def request_key(self, *, plan, files, backend):\n"
        "        return plan, files, backend\n"
        "\n"
        "    def cached_context(self, plan, files, backend):\n"
        "        return self.request_key(\n"
        "            plan=plan, files=files, backend=backend\n"
        "        )\n"
        "\n"
        "    def cached_state(self, plan, files, backend):\n"
        "        return self.request_key(\n"
        "            plan=plan, files=files, backend=backend\n"
        "        )\n"
        "\n"
        "    def cached_metadata(self, plan, files, backend):\n"
        "        return self.request_key(\n"
        "            plan=plan, files=files, backend=backend\n"
        "        )\n",
    )

    assert not any(
        item.detector_id == "repeated_builder_calls" for item in analyze_path(tmp_path)
    )


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
    unrelated_finding = replace(finding, detector_id="numeric_literal_dispatch")

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
    absolute_finding = finding.map_evidence(
        lambda location: replace(
            location,
            file_path=(tmp_path / location.file_path).resolve().as_posix(),
        )
    )

    plan = codemod_plan_from_findings(
        (absolute_finding,),
        selector_context=snapshot,
    )
    simulation = plan.simulate(snapshot)
    operation = json_report_object(plan.document.recipes[0].operations[0])

    assert plan.records[0].status.value == "executable_candidate"
    assert plan.expected_removed_finding_count == 1
    assert operation["operation"] == "convert_manual_registry_to_autoregister"
    assert set(operation) == {"operation", "target_id", "rationale"}
    assert operation["target_id"]
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
    simulation = plan.simulate(snapshot)
    recipe = plan.document.recipes[0]
    operation = json_report_object(recipe.operations[0])
    rewritten = next(iter(simulation.simulation.rewritten_sources.values()))

    assert plan.records[0].status.value == "executable_candidate"
    assert plan.records[0].refactor_concept == "auto_register_class_registry"
    assert len(recipe.authority_claims) == 1
    claim = recipe.authority_claims[0]
    assert claim.claimed_symbol == "Step"
    assert claim.authority_kind is SemanticAuthorityKind.AUTOREGISTER_FAMILY
    assert claim.file_path.endswith("pkg/mod.py")
    assert claim.qualname == "Step"
    assert claim.authority_id
    assert operation["operation"] == "derive_autoregister_instance_view"
    assert set(operation) == {"operation", "target_id", "rationale"}
    assert operation["target_id"] == claim.authority_id
    assert RefactorRecipeOperation.from_json_value(operation) == recipe.operations[0]
    assert "__registry__ = {}" in rewritten
    assert "registry_key = StepId.LOAD" in rewritten
    assert "registry_key = StepId.SAVE" in rewritten
    assert "def instances_by_registry_key" in rewritten
    assert "key_attribute = cls.__registry_key__" in rewritten
    assert "registered_type.__dict__[key_attribute]: registered_type()" in rewritten
    assert "for key, registered_type in cls.__registry__.items()" not in rewritten
    assert "STEP_TABLE = Step.instances_by_registry_key()" in rewritten
    assert simulation.is_clean is True
    namespace: dict[str, object] = {}
    exec(compile(rewritten, "pkg/mod.py", "exec"), namespace)
    step_id = namespace["StepId"]
    step_table = namespace["STEP_TABLE"]
    assert set(step_table) == {step_id.LOAD, step_id.SAVE}
    assert type(step_table[step_id.LOAD]) is namespace["LoadStep"]
    assert type(step_table[step_id.SAVE]) is namespace["SaveStep"]


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

    record = plan.records[0]
    obstacles_by_declaration = {
        obstacle.executable_declaration_name: obstacle.reason
        for obstacle in record.proof_obstacles
    }

    assert record.status.value == "rejected_by_safety_check"
    assert record.reason == (
        "no class-family recipe declaration proved an executable exact derivation"
    )
    assert (
        "source does not prove one complete zero-argument constructor view"
        in obstacles_by_declaration["AutoregisterInstanceViewRecipeBuilder"]
    )
    assert json_report_object(record)["proof_obstacles"] == tuple(
        json_report_object(obstacle) for obstacle in record.proof_obstacles
    )


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


def test_reused_dataclass_fields_do_not_invent_foreign_authority_mirrors(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Conversation:\n"
        "    id: str\n"
        "    title: str\n"
        "    created_at: str\n"
        "    updated_at: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactoringCheckpoint:\n"
        "    id: str\n"
        "    title: str\n"
        "    created_at: str\n"
        "    updated_at: str\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactoringStep:\n"
        "    id: str\n"
        "    title: str\n"
        "    created_at: str\n"
        "    updated_at: str\n"
        "\n"
        "class ConversationContext:\n"
        "    def __init__(self, conversation):\n"
        "        self.conversation = conversation\n"
        "\n"
        "    def get_conversation_summary(self):\n"
        "        return {\n"
        "            'id': self.conversation.id,\n"
        "            'title': self.conversation.title,\n"
        "            'created_at': self.conversation.created_at.upper(),\n"
        "            'updated_at': self.conversation.updated_at.upper(),\n"
        "        }\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )
    source_names = {
        finding.metrics.plan_source_name
        for finding in findings
        if finding.metrics.plan_mapping_name
        == "ConversationContext.get_conversation_summary:return"
    }

    assert source_names == set()


def test_other_authority_kinds_do_not_make_dataclass_fields_look_reused(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "from enum import Enum\n"
        "\n"
        "class RequestField(Enum):\n"
        "    TITLE = 'title'\n"
        "    STATUS = 'status'\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Request:\n"
        "    title: str\n"
        "    status: str\n"
        "\n"
        "REQUEST_FIELDS = ('title', 'status')\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )

    assert any(
        finding.metrics.plan_source_name == "Request"
        and finding.metrics.plan_mapping_name == "REQUEST_FIELDS"
        for finding in findings
    )


def test_typed_axis_selects_one_of_two_dataclasses_with_the_same_fields(
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
        "@dataclass(frozen=True)\n"
        "class Snapshot:\n"
        "    backend: str\n"
        "    parse_valid: bool\n"
        "\n"
        "def payload(report: Report):\n"
        "    return {\n"
        "        'backend': report.backend,\n"
        "        'parse_valid': report.parse_valid,\n"
        "    }\n",
    )

    findings = tuple(
        SemanticMirrorWithoutDescentDetector().detect(
            parse_python_modules(tmp_path),
            DetectorConfig(),
        )
    )
    source_names = {
        finding.metrics.plan_source_name
        for finding in findings
        if finding.metrics.plan_mapping_name == "payload:return"
    }

    assert source_names == {"Report"}


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
        and item.metrics.plan_mapping_name == "ActionReport.to_dict:return"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[module_path.as_posix()]
    recipe = plan.document.recipes[0]

    assert record.status.value == "executable_candidate"
    assert len(recipe.authority_claims) == 1
    claim = recipe.authority_claims[0]
    assert claim.claimed_symbol == "RefactorAction"
    assert claim.authority_kind is SemanticAuthorityKind.DATACLASS_SCHEMA
    assert claim.file_path == module_path.as_posix()
    assert claim.qualname == "RefactorAction"
    assert claim.authority_id
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert record.refactor_concept == "dataclass_payload_projection"
    assert (
        record.evaluation_declaration_name
        == "DataclassPayloadProjectionMappingRecipeBuilder"
    )
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "derive_dataclass_payload_projection",
    )
    operation = recipe.operations[0]
    assert isinstance(operation, DeriveDataclassPayloadProjectionOperation)
    operation_payload = json_report_object(operation)
    assert set(operation_payload) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    assert RefactorRecipeOperation.from_json_value(operation_payload) == operation
    with pytest.raises(
        ValueError,
        match="Unsupported DeriveDataclassPayloadProjectionOperation payload field",
    ):
        RefactorRecipeOperation.from_json_value(
            {
                **operation_payload,
                "field_names": ["kind", "description", "confidence"],
                "old_source": "copied projection source",
                "new_source": "copied generated source",
                "import_source": "import dataclasses",
            }
        )
    assert "import dataclasses" in rewritten_source
    assert "for field in dataclasses.fields(" in rewritten_source
    assert "RefactorAction\n" in rewritten_source
    assert "getattr(\n                    self.action," in rewritten_source
    assert "payload_from_field_values" not in rewritten_source
    assert "'kind': self.action.kind" not in rewritten_source
    assert "'emitted': self.emitted" in rewritten_source
    namespace: dict[str, object] = {}
    exec(rewritten_source, namespace)
    action = namespace["RefactorAction"]("run", "Run it", "high")
    report = namespace["ActionReport"](action, True)
    assert report.to_dict() == {
        "kind": "run",
        "description": "Run it",
        "confidence": "high",
        "emitted": True,
    }


def _dataclass_payload_projection_operation(
    snapshot: CodemodSourceSnapshot,
    authority_qualname: str,
    projection_qualname: str,
) -> DeriveDataclassPayloadProjectionOperation:
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == authority_qualname
    )
    projection_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == projection_qualname
    )
    return DeriveDataclassPayloadProjectionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id),
        projection_target=SourceRewriteTarget(target_id=projection_target.target_id),
    )


def test_dataclass_payload_operation_rederives_current_source(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n\n"
        "def payload(action: RefactorAction):\n"
        "    return {\n"
        "        'kind': action.kind,\n"
        "        'description': action.description,\n"
        "    }\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _dataclass_payload_projection_operation(
        snapshot,
        "RefactorAction",
        "payload",
    )
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    changed_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "action",
        "record",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )

    simulation = (
        RefactorRecipe("derive-current-dataclass-payload")
        .with_operation(replayed)
        .simulate(changed_snapshot)
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert "getattr(" in rewritten
    assert "record," in rewritten
    assert "action.kind" not in rewritten
    assert "action.description" not in rewritten


def test_dataclass_payload_operation_rejects_authority_schema_drift(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n\n"
        "def payload(action: RefactorAction):\n"
        "    return {\n"
        "        'kind': action.kind,\n"
        "        'description': action.description,\n"
        "    }\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _dataclass_payload_projection_operation(
        snapshot,
        "RefactorAction",
        "payload",
    )
    changed_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "    description: str",
        "    summary: str",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one exhaustive return-dict",
    ):
        operation.source_edits(changed_snapshot)


def test_dataclass_payload_operation_rejects_ambiguous_return_projections(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n\n"
        "def payload(action: RefactorAction, alternate: bool):\n"
        "    if alternate:\n"
        "        return {\n"
        "            'kind': action.kind,\n"
        "            'description': action.description,\n"
        "        }\n"
        "    return {\n"
        "        'kind': action.kind,\n"
        "        'description': action.description,\n"
        "    }\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _dataclass_payload_projection_operation(
        snapshot,
        "RefactorAction",
        "payload",
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exhaustive return-dict.*found 2",
    ):
        operation.source_edits(snapshot)


def test_dataclass_payload_operation_ignores_nested_function_returns(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n\n"
        "def payload(action: RefactorAction):\n"
        "    def nested():\n"
        "        return {\n"
        "            'kind': action.kind,\n"
        "            'description': action.description,\n"
        "        }\n"
        "    return {}\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _dataclass_payload_projection_operation(
        snapshot,
        "RefactorAction",
        "payload",
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exhaustive return-dict.*found 0",
    ):
        operation.source_edits(snapshot)


def test_semantic_mirror_synthesizes_dataclass_field_name_collection_recipe(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class PhaseRecord:\n"
        "    run_id: str\n"
        "    seconds: float\n"
        "\n"
        "def field_names():\n"
        "    phase_record_fields = ('run_id', 'seconds')\n"
        "    return phase_record_fields\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "PhaseRecord"
        and item.metrics.plan_mapping_name == "phase_record_fields"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[module_path.as_posix()]
    recipe = plan.document.recipes[0]

    assert plan.records[0].status.value == "executable_candidate"
    assert (
        plan.records[0].evaluation_declaration_name
        == "DataclassFieldNameCollectionProjectionMappingRecipeBuilder"
    )
    assert plan.records[0].refactor_concept == "dataclass_payload_projection"
    assert simulation.is_clean is True
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "derive_dataclass_field_name_collection_projection",
    )
    operation = recipe.operations[0]
    assert isinstance(
        operation,
        DeriveDataclassFieldNameCollectionProjectionOperation,
    )
    assert set(json_report_object(operation)) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    changed_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "('run_id', 'seconds')",
        "['run_id', 'seconds']",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )
    replay_simulation = (
        RefactorRecipe("derive-current-field-name-collection")
        .with_operation(replayed)
        .simulate(changed_snapshot)
    )
    replayed_source = replay_simulation.simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert "[field.name for field in dataclasses.fields(PhaseRecord)]" in (
        replayed_source
    )
    commented_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "('run_id', 'seconds')",
        "(\n        'run_id',\n        # Preserve this explanation.\n"
        "        'seconds',\n    )",
    )
    commented_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): commented_source},
    )
    with pytest.raises(
        CodemodOperationPreflightError,
        match="absent from current source",
    ):
        replayed.source_edits(commented_snapshot)
    assert "import dataclasses" in rewritten_source
    assert (
        "tuple(field.name for field in dataclasses.fields(PhaseRecord))"
        in rewritten_source
    )
    namespace: dict[str, object] = {}
    exec(rewritten_source, namespace)
    assert namespace["field_names"]() == ("run_id", "seconds")


def test_semantic_mirror_rejects_reordered_dataclass_field_name_collection(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class PhaseRecord:\n"
        "    run_id: str\n"
        "    seconds: float\n"
        "\n"
        "def field_names():\n"
        "    phase_record_fields = ('seconds', 'run_id')\n"
        "    return phase_record_fields\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "PhaseRecord"
        and item.metrics.plan_mapping_name == "phase_record_fields"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    record = plan.records[0]
    assert record.status.value == "rejected_by_safety_check"
    assert tuple(
        obstacle.executable_declaration_name for obstacle in record.proof_obstacles
    ) == ("DataclassFieldNameCollectionProjectionMappingRecipeBuilder",)
    assert plan.document.recipes == ()


def test_semantic_mirror_field_name_collection_rejects_new_runtime_import(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "pkg"
    package_path.mkdir(parents=True)
    (package_path / "model.py").write_text(
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class PhaseRecord:\n"
        "    run_id: str\n"
        "    seconds: float\n",
        encoding="utf-8",
    )
    (package_path / "report.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from .model import PhaseRecord\n"
        "\n"
        "def field_names():\n"
        "    phase_record_fields = ('run_id', 'seconds')\n"
        "    return phase_record_fields\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "PhaseRecord"
        and item.metrics.plan_mapping_name == "phase_record_fields"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    record = plan.records[0]
    assert record.status.value == "rejected_by_safety_check"
    assert tuple(
        obstacle.executable_declaration_name for obstacle in record.proof_obstacles
    ) == ("DataclassFieldNameCollectionProjectionMappingRecipeBuilder",)
    assert plan.document.recipes == ()


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
        "    description: str\n"
        "    model: WorkflowModel\n"
        "\n"
        "    def definition_items(self):\n"
        "        return (\n"
        "            ('description', self.description),\n"
        "            ('workflow_id', self.model.workflow_id),\n"
        "            ('command_action_ids', self.model.command_action_ids),\n"
        "            ('default_next_action_id', self.model.default_next_action_id)\n"
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
    simulation = plan.simulate(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[module_path.as_posix()]
    recipe = plan.document.recipes[0]

    assert record.status.value == "executable_candidate"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert record.refactor_concept == "dataclass_payload_projection"
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "derive_dataclass_key_value_sequence_projection",
    )
    operation = recipe.operations[0]
    assert isinstance(operation, DeriveDataclassKeyValueSequenceProjectionOperation)
    assert set(json_report_object(operation)) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    changed_source = (
        snapshot.sources_by_file_path[module_path.as_posix()]
        .replace(
            "    model: WorkflowModel",
            "    record: WorkflowModel",
        )
        .replace("self.model", "self.record")
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )
    replay_simulation = (
        RefactorRecipe("derive-current-key-value-sequence")
        .with_operation(replayed)
        .simulate(changed_snapshot)
    )
    replayed_source = replay_simulation.simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert "self.record" in replayed_source
    assert "self.model" not in replayed_source
    assert "import dataclasses" in rewritten_source
    assert "for field in dataclasses.fields(" in rewritten_source
    assert "getattr(\n                        self.model," in rewritten_source
    assert "payload_items_from_field_values" not in rewritten_source
    assert "('description', self.description)" in rewritten_source
    assert "('workflow_id', self.workflow_id)" not in rewritten_source
    namespace: dict[str, object] = {}
    exec(rewritten_source, namespace)
    model = namespace["WorkflowModel"](
        "workflow",
        ("prepare", "run"),
        "run",
    )
    definition = namespace["WorkflowDefinition"]("Workflow", model)
    assert definition.definition_items() == (
        ("description", "Workflow"),
        ("workflow_id", "workflow"),
        ("command_action_ids", ("prepare", "run")),
        ("default_next_action_id", "run"),
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
    assert all(
        certificate.edge.authority_id != workflow_authority.authority_id
        for certificate in migrated_graph.missing_descent_certificates
    )
    assert all(
        "definition_items" not in projection.label
        for projection in migrated_graph.projections
    )


def test_key_value_sequence_recipe_rejects_interleaving_and_comments(
    tmp_path: Path,
) -> None:
    source = (
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
        "    description: str\n"
        "    model: WorkflowModel\n"
        "\n"
        "    def definition_items(self):\n"
        "        return (\n"
        "            ('description', self.description),\n"
        "            ('workflow_id', self.model.workflow_id),\n"
        "            ('command_action_ids', self.model.command_action_ids),\n"
        "            ('default_next_action_id', self.model.default_next_action_id),\n"
        "        )\n"
    )
    unsafe_sources = {
        "interleaved": source.replace(
            "            ('description', self.description),\n"
            "            ('workflow_id', self.model.workflow_id),\n",
            "            ('workflow_id', self.model.workflow_id),\n"
            "            ('description', self.description),\n",
        ),
        "commented": source.replace(
            "            ('command_action_ids', self.model.command_action_ids),\n",
            "            # This explanation must survive any migration.\n"
            "            ('command_action_ids', self.model.command_action_ids),\n",
        ),
    }

    for case_name, unsafe_source in unsafe_sources.items():
        case_root = tmp_path / case_name
        _write_module(case_root, unsafe_source)
        modules = parse_python_modules(case_root)
        finding = next(
            item
            for item in SemanticMirrorWithoutDescentDetector().detect(
                modules,
                DetectorConfig(),
            )
            if item.metrics.plan_source_name == "WorkflowModel"
        )
        snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))
        plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

        record = plan.records[0]
        assert record.status.value == "rejected_by_safety_check"
        assert not record.recipe_id
        assert tuple(
            obstacle.executable_declaration_name for obstacle in record.proof_obstacles
        ) == ("DataclassKeyValueSequenceProjectionMappingRecipeBuilder",)


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
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class ActionReport:\n"
        '    action: "RefactorAction"\n'
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
    simulation = plan.simulate(snapshot)
    recipe = plan.document.recipes[0]
    rewritten_report = simulation.simulation.rewritten_sources[
        (package_dir / "report.py").as_posix()
    ]

    assert record.status.value == "executable_candidate"
    assert record.refactor_concept == "dataclass_payload_projection"
    assert simulation.is_clean is True
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "derive_dataclass_payload_projection",
    )
    assert "from .model import RefactorAction" in rewritten_report
    assert "import dataclasses" in rewritten_report
    assert "for field in dataclasses.fields(" in rewritten_report
    assert "getattr(\n                    self.action," in rewritten_report
    assert (package_dir / "model.py").as_posix() not in (
        simulation.simulation.rewritten_sources
    )
    assert "'emitted': self.emitted" in rewritten_report

    report_path = package_dir / "report.py"
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\nfrom .model import RefactorAction\n",
        ),
        encoding="utf-8",
    )
    imported_modules = parse_python_modules(tmp_path)
    imported_finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            imported_modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "RefactorAction"
    )
    imported_snapshot = CodemodSourceSnapshot.from_modules(
        imported_modules,
        (imported_finding,),
    )
    imported_plan = codemod_plan_from_findings(
        (imported_finding,),
        selector_context=imported_snapshot,
    )

    assert tuple(
        operation.operation_key()
        for operation in imported_plan.document.recipes[0].operations
    ) == ("derive_dataclass_payload_projection",)
    imported_simulation = imported_plan.simulate(imported_snapshot)
    imported_rewritten_report = imported_simulation.simulation.rewritten_sources[
        report_path.as_posix()
    ]
    assert imported_rewritten_report.count("from .model import RefactorAction") == 1
    assert "import dataclasses" in imported_rewritten_report

    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "from .model import RefactorAction\n",
            "from .model import RefactorAction\nRefactorAction = object\n",
        ),
        encoding="utf-8",
    )
    shadowed_modules = parse_python_modules(tmp_path)
    shadowed_finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            shadowed_modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "RefactorAction"
    )
    shadowed_snapshot = CodemodSourceSnapshot.from_modules(
        shadowed_modules,
        (shadowed_finding,),
    )
    shadowed_plan = codemod_plan_from_findings(
        (shadowed_finding,),
        selector_context=shadowed_snapshot,
    )

    assert shadowed_plan.records[0].status.value == "rejected_by_safety_check"
    assert not shadowed_plan.records[0].recipe_id


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
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import dataclass\n"
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from .model import RefactorAction\n"
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

    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "RefactorAction"
    )
    projection_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == "ActionReport.to_dict"
    )
    operation = DeriveDataclassPayloadProjectionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id),
        projection_target=SourceRewriteTarget(target_id=projection_target.target_id),
    )
    with pytest.raises(CodemodOperationPreflightError, match="module cycle"):
        operation.source_edits(snapshot)


def test_dataclass_payload_recipe_requires_one_exhaustive_direct_field_run(
    tmp_path: Path,
) -> None:
    source = (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class RefactorAction:\n"
        "    kind: str\n"
        "    description: str\n"
        "    confidence: str\n"
        "\n"
        "\n"
        "def project(action: RefactorAction):\n"
        "    return {\n"
        "        'kind': action.kind,\n"
        "        'description': action.description,\n"
        "        'confidence': action.confidence,\n"
        "    }\n"
    )
    unsafe_sources = {
        "subset": source.replace(
            "    confidence: str\n",
            "    confidence: str\n    category: str\n",
            1,
        ),
        "transformed": source.replace(
            "action.confidence,",
            "action.confidence.upper(),",
        ),
        "reordered": source.replace(
            "        'kind': action.kind,\n"
            "        'description': action.description,\n",
            "        'description': action.description,\n"
            "        'kind': action.kind,\n",
        ),
        "interleaved": source.replace(
            "        'description': action.description,\n",
            "        'unrelated': 1,\n        'description': action.description,\n",
        ),
        "commented": source.replace(
            "        'description': action.description,\n",
            "        # This explanation must survive any migration.\n"
            "        'description': action.description,\n",
        ),
        "shadowed_dataclasses": source.replace(
            "def project(action: RefactorAction):",
            "def project(action: RefactorAction, dataclasses):",
        ),
        "structural_owner": source.replace(
            "def project(action: RefactorAction):",
            "def refactor_action_project(action: object):",
        ),
        "nested_authority_annotation": source.replace(
            "def project(action: RefactorAction):",
            "def project(action: list[RefactorAction]):",
        ),
        "shadowed_constructor": source.replace(
            "def project(action: RefactorAction):\n",
            "def refactor_action_project():\n"
            "    RefactorAction = lambda: object()\n"
            "    action = RefactorAction()\n",
        ),
        "inherited": source.replace(
            "@dataclass(frozen=True)\nclass RefactorAction:\n",
            "@dataclass(frozen=True)\n"
            "class RefactorActionBase:\n"
            "    inherited: str\n"
            "\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class RefactorAction(RefactorActionBase):\n",
        ),
    }

    for case_name, unsafe_source in unsafe_sources.items():
        case_root = tmp_path / case_name
        _write_module(case_root, unsafe_source)
        modules = parse_python_modules(case_root)
        expected_mapping_name = (
            "refactor_action_project:return"
            if case_name in {"shadowed_constructor", "structural_owner"}
            else "project:return"
        )
        finding = next(
            item
            for item in SemanticMirrorWithoutDescentDetector().detect(
                modules,
                DetectorConfig(),
            )
            if item.metrics.plan_source_name == "RefactorAction"
            and item.metrics.plan_mapping_name == expected_mapping_name
        )
        snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))
        plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

        record = plan.records[0]
        obstacles_by_declaration = {
            obstacle.executable_declaration_name: obstacle.reason
            for obstacle in record.proof_obstacles
        }

        assert record.status.value == "rejected_by_safety_check"
        assert not record.recipe_id
        assert record.reason == (
            "no inferred mapping recipe builder proved an executable exact derivation"
        )
        assert (
            "dataclass payload projection requires one contiguous, exhaustive"
            in obstacles_by_declaration[
                "DataclassPayloadProjectionMappingRecipeBuilder"
            ]
        )
        assert tuple(obstacles_by_declaration) == (
            "DataclassPayloadProjectionMappingRecipeBuilder",
        )


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
        "    return SourceLineReplacement(\n"
        "        file_path=source_path,\n"
        "        start_line=insertion_line,\n"
        "        end_line=insertion_line - 1,\n"
        "        replacement_lines=import_lines,\n"
        "        rationale=reason,\n"
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
            SourceLocation(str(module_path), 26, "build_replacement:return"),
            SourceLocation(str(module_path), 4, "SourceLineReplacement"),
            SourceLocation(str(module_path), 12, "SourceLineSpan"),
        ),
        projection_evidence=SourceLocation(
            str(module_path),
            26,
            "build_replacement:return",
        ),
        authority_evidence=SourceLocation(
            str(module_path),
            12,
            "SourceLineSpan",
        ),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=("end_line", "start_line"),
            mapping_name="build_replacement:return",
            source_name="SourceLineSpan",
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    record = plan.records[0]
    simulation = plan.simulate(snapshot)
    rewritten_source = simulation.simulation.rewritten_sources[module_path.as_posix()]
    recipe = plan.document.recipes[0]

    assert record.status.value == "executable_candidate"
    assert record.refactor_concept == "constructor_kwarg_carrier_projection"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert tuple(operation.operation_key() for operation in recipe.operations) == (
        "derive_dataclass_constructor_projection",
    )
    operation = recipe.operations[0]
    assert isinstance(operation, DeriveDataclassConstructorProjectionOperation)
    assert set(json_report_object(operation)) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    assert replayed == operation
    changed_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "def line_replacement(",
        "def as_replacement(",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )
    replay_simulation = (
        RefactorRecipe("derive-current-constructor-projection")
        .with_operation(replayed)
        .simulate(changed_snapshot)
    )
    replayed_source = replay_simulation.simulation.rewritten_sources[
        module_path.as_posix()
    ]
    assert ".as_replacement(" in replayed_source
    assert ".line_replacement(" not in replayed_source
    assert "SourceLineSpan(" in rewritten_source
    assert ".line_replacement(" in rewritten_source
    assert "start_line=insertion_line" in rewritten_source
    assert "end_line=insertion_line - 1" in rewritten_source
    assert "replacement_lines=import_lines" in rewritten_source
    namespace: dict[str, object] = {}
    exec(rewritten_source, namespace)
    result = namespace["build_replacement"](
        "module.py",
        7,
        ("import x",),
        "reason",
    )
    assert result == namespace["SourceLineReplacement"](
        "module.py",
        7,
        6,
        ("import x",),
        "reason",
    )


def test_constructor_projection_rejection_reports_only_constructor_builder(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLineReplacement:\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class SourceLineSpan:\n"
        "    start_line: int\n"
        "    end_line: int\n"
        "\n"
        "def build_replacement(start_line, end_line):\n"
        "    return SourceLineReplacement(\n"
        "        start_line=start_line,\n"
        "        end_line=end_line,\n"
        "    )\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = RefactorFinding(
        detector_id="semantic_mirror_without_descent",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Semantic mirror without descent",
        summary="Constructor projection repeats SourceLineSpan fields.",
        why="semantic fact is mirrored outside its nominal authority",
        capability_gap="derive the projection from the authority instead",
        relation_context="projection lacks a semantic-descent certificate",
        evidence=(
            SourceLocation(str(module_path), 14, "build_replacement:return"),
            SourceLocation(str(module_path), 9, "SourceLineSpan"),
        ),
        projection_evidence=SourceLocation(
            str(module_path),
            14,
            "build_replacement:return",
        ),
        authority_evidence=SourceLocation(
            str(module_path),
            9,
            "SourceLineSpan",
        ),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=("end_line", "start_line"),
            mapping_name="build_replacement:return",
            source_name="SourceLineSpan",
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    record = codemod_plan_from_findings(
        (finding,),
        selector_context=snapshot,
    ).records[0]

    assert record.status.value == "rejected_by_safety_check"
    assert tuple(
        obstacle.executable_declaration_name for obstacle in record.proof_obstacles
    ) == ("DataclassConstructorProjectionMappingRecipeBuilder",)


def test_constructor_projection_requires_same_nominal_constructor(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "pkg"
    package_path.mkdir()
    (package_path / "__init__.py").write_text("", encoding="utf-8")
    for module_name in ("left", "right"):
        (package_path / f"{module_name}.py").write_text(
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True)\n"
            "class Replacement:\n"
            "    start_line: int\n"
            "    end_line: int\n",
            encoding="utf-8",
        )
    model_path = package_path / "model.py"
    model_path.write_text(
        "from dataclasses import dataclass\n"
        "from .left import Replacement\n\n"
        "@dataclass(frozen=True)\n"
        "class Span:\n"
        "    start_line: int\n"
        "    end_line: int\n\n"
        "    def replacement(self):\n"
        "        return Replacement(\n"
        "            start_line=self.start_line,\n"
        "            end_line=self.end_line,\n"
        "        )\n",
        encoding="utf-8",
    )
    report_path = package_path / "report.py"
    report_path.write_text(
        "from .model import Span\n"
        "from .right import Replacement\n\n"
        "def build(start_line, end_line):\n"
        "    return Replacement(\n"
        "        start_line=start_line,\n"
        "        end_line=end_line,\n"
        "    )\n",
        encoding="utf-8",
    )
    finding = RefactorFinding(
        detector_id="semantic_mirror_without_descent",
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Semantic mirror without descent",
        summary="Constructor projection repeats Span fields.",
        why="semantic fact is mirrored outside its nominal authority",
        capability_gap="derive the projection from the authority instead",
        relation_context="projection lacks a semantic-descent certificate",
        evidence=(
            SourceLocation(str(report_path), 5, "build:return"),
            SourceLocation(str(model_path), 5, "Span"),
        ),
        projection_evidence=SourceLocation(str(report_path), 5, "build:return"),
        authority_evidence=SourceLocation(str(model_path), 5, "Span"),
        metrics=MappingMetrics.from_field_names(
            mapping_site_count=2,
            field_names=("start_line", "end_line"),
            mapping_name="build:return",
            source_name="Span",
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path),
        (finding,),
    )

    record = codemod_plan_from_findings(
        (finding,),
        selector_context=snapshot,
    ).records[0]

    assert record.status.value == "rejected_by_safety_check"
    assert tuple(
        obstacle.executable_declaration_name for obstacle in record.proof_obstacles
    ) == ("DataclassConstructorProjectionMappingRecipeBuilder",)


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
    simulation = plan.simulate(snapshot)
    recipe = plan.document.recipes[0]
    recipe_payload = json_report_object(plan.document)["recipes"][0]
    operations = recipe_payload["operations"]

    assert plan.records[0].status.value == "executable_candidate"
    assert plan.records[0].refactor_concept == "derived_projection"
    assert len(recipe.authority_claims) == 1
    claim = recipe.authority_claims[0]
    assert claim.claimed_symbol == "ConfidenceLevel"
    assert claim.authority_kind is SemanticAuthorityKind.ENUM
    assert claim.file_path == (package_dir / "taxonomy.py").as_posix()
    assert claim.qualname == "ConfidenceLevel"
    assert claim.authority_id
    assert simulation.is_clean is True
    assert [operation["operation"] for operation in operations] == [
        "derive_enum_subset",
    ]
    operation = operations[0]
    assert set(operation) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    assert RefactorRecipeOperation.from_json_value(operation) == recipe.operations[0]
    with pytest.raises(
        ValueError,
        match="Unsupported DeriveEnumSubsetOperation payload field",
    ):
        RefactorRecipeOperation.from_json_value(
            {
                **operation,
                "mapping_name": "_ACTIONABLE_CONFIDENCE_LEVELS",
                "selected_names": ["HIGH", "MEDIUM"],
                "class_source": "copied authority source",
                "source": "copied projection source",
            }
        )
    rewritten_taxonomy = simulation.simulation.rewritten_sources[
        (package_dir / "taxonomy.py").as_posix()
    ]
    rewritten_codemod = simulation.simulation.rewritten_sources[
        (package_dir / "codemod.py").as_posix()
    ]
    assert (
        "def actionable_confidence_levels(cls) -> frozenset[str]" in rewritten_taxonomy
    )
    assert "cls.HIGH.value" in rewritten_taxonomy
    assert "cls.MEDIUM.value" in rewritten_taxonomy
    assert "from pkg.taxonomy import ConfidenceLevel\n" in rewritten_codemod
    assert (
        "_ACTIONABLE_CONFIDENCE_LEVELS: "
        "frozenset[pkg.taxonomy.ConfidenceLevel] = "
        "ConfidenceLevel.actionable_confidence_levels()" in rewritten_codemod
    )


def _enum_subset_operation(
    snapshot: CodemodSourceSnapshot,
    authority_qualname: str,
    projection_path: Path,
) -> DeriveEnumSubsetOperation:
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == authority_qualname
    )
    projection_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.is_module and target.file_path == projection_path.as_posix()
    )
    return DeriveEnumSubsetOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id),
        projection_target=SourceRewriteTarget(target_id=projection_target.target_id),
    )


def test_enum_subset_operation_rederives_current_source(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    taxonomy_path = package_dir / "taxonomy.py"
    projection_path = package_dir / "projection.py"
    taxonomy_source = (
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n"
        "    LOW = 'low'\n"
    )
    projection_source = (
        "from .taxonomy import ConfidenceLevel\n\n"
        "ACTIONABLE = frozenset(('high', 'medium'))\n"
    )
    taxonomy_path.write_text(taxonomy_source, encoding="utf-8")
    projection_path.write_text(projection_source, encoding="utf-8")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _enum_subset_operation(snapshot, "ConfidenceLevel", projection_path)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    changed_projection_source = projection_source.replace("medium", "low")
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {
            taxonomy_path.as_posix(): taxonomy_source,
            projection_path.as_posix(): changed_projection_source,
        },
    )

    edits = replayed.source_edits(changed_snapshot)
    rendered_edits = "\n".join(
        "".join(edit.inserted_lines)
        if hasattr(edit, "inserted_lines")
        else "".join(edit.replacement_lines)
        for edit in edits
    )

    assert "cls.HIGH.value" in rendered_edits
    assert "cls.LOW.value" in rendered_edits
    assert "cls.MEDIUM.value" not in rendered_edits


def test_enum_subset_operation_rejects_authority_value_drift(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n\n"
        "ACTIONABLE = frozenset(('high', 'medium'))\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _enum_subset_operation(snapshot, "ConfidenceLevel", module_path)
    changed_source = snapshot.sources_by_file_path[module_path.as_posix()].replace(
        "MEDIUM = 'medium'",
        "MEDIUM = 'moderate'",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {module_path.as_posix(): changed_source},
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one literal frozenset subset",
    ):
        operation.source_edits(changed_snapshot)


def test_enum_subset_operation_rejects_ambiguous_projections(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n\n"
        "FIRST = frozenset(('high', 'medium'))\n"
        "SECOND = frozenset(('medium',))\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _enum_subset_operation(snapshot, "ConfidenceLevel", module_path)

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one literal frozenset subset; found 2",
    ):
        operation.source_edits(snapshot)


@pytest.mark.parametrize("shadow_module", ("authority", "projection"))
def test_enum_subset_operation_rejects_shadowed_frozenset(
    tmp_path: Path,
    shadow_module: str,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    taxonomy_path = package_dir / "taxonomy.py"
    projection_path = package_dir / "projection.py"
    shadow_source = "frozenset = lambda values: tuple(values)\n"
    taxonomy_path.write_text(
        (shadow_source if shadow_module == "authority" else "")
        + "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n",
        encoding="utf-8",
    )
    projection_path.write_text(
        "from .taxonomy import ConfidenceLevel\n"
        + (shadow_source if shadow_module == "projection" else "")
        + "\nACTIONABLE = frozenset(('high', 'medium'))\n",
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _enum_subset_operation(snapshot, "ConfidenceLevel", projection_path)

    with pytest.raises(CodemodOperationPreflightError):
        operation.source_edits(snapshot)


def test_enum_subset_operation_rejects_aliases_and_accessor_collisions(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'shared'\n"
        "    MEDIUM = 'shared'\n\n"
        "ACTIONABLE = frozenset(('shared',))\n",
    )
    aliased_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )
    aliased_operation = _enum_subset_operation(
        aliased_snapshot,
        "ConfidenceLevel",
        module_path,
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="aliased string values",
    ):
        aliased_operation.source_edits(aliased_snapshot)

    module_path.write_text(
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n"
        "    actionable = None\n\n"
        "ACTIONABLE = frozenset(('high', 'medium'))\n",
        encoding="utf-8",
    )
    collision_snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path)
    )
    collision_operation = _enum_subset_operation(
        collision_snapshot,
        "ConfidenceLevel",
        module_path,
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="already binds 'actionable'",
    ):
        collision_operation.source_edits(collision_snapshot)


def test_enum_subset_operation_executes_source_derived_view(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        "from enum import StrEnum\n\n"
        "class ConfidenceLevel(StrEnum):\n"
        '    """Confidence semantics remain declaration owned."""\n'
        "    HIGH = 'high'\n"
        "    MEDIUM = 'medium'\n"
        "    LOW = 'low'\n\n"
        "ACTIONABLE = frozenset(('high', 'medium'))\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _enum_subset_operation(snapshot, "ConfidenceLevel", module_path)

    simulation = (
        RefactorRecipe("derive-enum-subset")
        .with_operation(operation)
        .simulate(snapshot)
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)

    confidence_level = namespace["ConfidenceLevel"]
    assert confidence_level.__doc__ == "Confidence semantics remain declaration owned."
    assert namespace["ACTIONABLE"] == frozenset(("high", "medium"))
    assert confidence_level.actionable() == namespace["ACTIONABLE"]


def test_semantic_mirror_enum_rejection_reports_only_enum_builder(
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
        "    LOW = 'low'\n"
        "\n"
        "    @classmethod\n"
        "    def actionable_confidence_levels(cls):\n"
        "        return frozenset((cls.HIGH.value,))\n",
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
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.metrics.plan_mapping_name == "_ACTIONABLE_CONFIDENCE_LEVELS"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    record = codemod_plan_from_findings(
        (finding,),
        selector_context=snapshot,
    ).records[0]

    assert record.status.value == "rejected_by_safety_check"
    assert tuple(
        obstacle.executable_declaration_name for obstacle in record.proof_obstacles
    ) == ("EnumSubsetSemanticMirrorRecipeBuilder",)


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
    simulation = plan.simulate(snapshot)
    recipe_payload = json_report_object(plan.document)["recipes"][0]
    recipe = plan.document.recipes[0]
    operations = recipe_payload["operations"]
    rewritten = next(
        source
        for path, source in simulation.simulation.rewritten_sources.items()
        if path.endswith("detector.py")
    )

    assert plan.records[0].status.value == "executable_candidate"
    assert len(recipe.authority_claims) == 1
    claim = recipe.authority_claims[0]
    assert claim.claimed_symbol == "LabeledMode"
    assert claim.file_path == (package_dir / "taxonomy.py").as_posix()
    assert claim.qualname == "LabeledMode"
    assert claim.authority_kind is SemanticAuthorityKind.CLASS_FAMILY
    assert claim.authority_id
    assert claim.claimed_symbol not in {"CapabilityMode", "ObservationMode"}
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert [operation["operation"] for operation in operations] == [
        "derive_class_family_collection",
    ]
    operation = operations[0]
    assert set(operation) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    assert "source" not in operation
    assert "assignment_name" not in operation
    assert RefactorRecipeOperation.from_json_value(operation) == recipe.operations[0]
    with pytest.raises(
        ValueError,
        match="Unsupported DeriveClassFamilyCollectionOperation payload field",
    ):
        RefactorRecipeOperation.from_json_value(
            {
                **operation,
                "assignment_name": "MODE_ENUMS",
                "class_names": ["CapabilityMode", "ObservationMode"],
                "source": ("MODE_ENUMS = tuple(LabeledMode.__subclasses__())"),
            }
        )
    assert (
        "MODE_ENUMS: tuple[ModeEnum, ...] = tuple(LabeledMode.__subclasses__())"
    ) in rewritten
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
    simulation = plan.simulate(snapshot)
    operation = json_report_object(plan.document)["recipes"][0]["operations"][0]
    rewritten = next(
        source
        for path, source in simulation.simulation.rewritten_sources.items()
        if path.endswith("detector.py")
    )

    assert plan.records[0].status.value == "executable_candidate"
    assert plan.expected_removed_finding_count == 1
    assert simulation.is_clean is True
    assert operation["operation"] == "derive_class_family_collection"
    assert "source" not in operation
    assert (
        "OWNER_NAMES = frozenset(member_type.__name__ for member_type in "
        "LabeledMode.__subclasses__())"
    ) in rewritten
    assert (
        "from .taxonomy import (\n"
        "    CapabilityMode,\n"
        "    ObservationMode,\n"
        "    LabeledMode,\n"
        ")\n"
    ) in rewritten


def _class_family_collection_operation(
    snapshot: CodemodSourceSnapshot,
    authority_qualname: str,
    projection_path: Path,
) -> DeriveClassFamilyCollectionOperation:
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.qualname == authority_qualname
    )
    projection_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if target.is_module and target.file_path == projection_path.as_posix()
    )
    return DeriveClassFamilyCollectionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id),
        projection_target=SourceRewriteTarget(target_id=projection_target.target_id),
    )


def test_class_family_collection_operation_rederives_current_source(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    taxonomy_path = package_dir / "taxonomy.py"
    projection_path = package_dir / "projection.py"
    taxonomy_source = (
        "class Root:\n"
        "    pass\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n"
        "class Beta(Root):\n"
        "    pass\n"
    )
    projection_source = "from .taxonomy import Alpha, Beta\n\nMEMBERS = (Alpha, Beta)\n"
    taxonomy_path.write_text(taxonomy_source, encoding="utf-8")
    projection_path.write_text(projection_source, encoding="utf-8")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", projection_path)
    payload = json_report_object(operation)
    replayed = RefactorRecipeOperation.from_json_value(payload)
    changed_projection_source = projection_source.replace(
        "(Alpha, Beta)",
        "(Beta, Alpha)",
    )
    changed_snapshot = CodemodSourceSnapshot.from_indexed_sources(
        snapshot.source_index,
        {
            taxonomy_path.as_posix(): taxonomy_source,
            projection_path.as_posix(): changed_projection_source,
        },
    )

    assert set(payload) == {
        "operation",
        "target_id",
        "projection_target",
        "rationale",
    }
    assert replayed == operation
    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one complete literal collection",
    ):
        replayed.source_edits(changed_snapshot)


def test_class_family_collection_operation_rejects_ambiguous_projections(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "class Root:\n"
        "    pass\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n"
        "class Beta(Root):\n"
        "    pass\n\n"
        "FIRST = (Alpha, Beta)\n"
        "SECOND = (Alpha, Beta)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", module_path)

    with pytest.raises(
        CodemodOperationPreflightError,
        match="exactly one complete literal collection; found 2",
    ):
        operation.source_edits(snapshot)


def test_class_family_collection_operation_executes_source_derived_view(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "class Root:\n"
        "    pass\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n"
        "class Beta(Root):\n"
        "    pass\n\n"
        "MEMBERS: tuple[type[Root], ...] = (Alpha, Beta)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", module_path)

    direct_edits = operation.source_edits(snapshot)
    simulation = (
        RefactorRecipe("derive-members")
        .with_operation(operation)
        .simulate(snapshot)
    )
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    namespace: dict[str, object] = {}
    exec(compile(rewritten, module_path.as_posix(), "exec"), namespace)

    assert direct_edits
    assert "MEMBERS: tuple[type[Root], ...] = tuple(Root.__subclasses__())" in rewritten
    assert namespace["MEMBERS"] == (namespace["Alpha"], namespace["Beta"])


@pytest.mark.parametrize(
    ("collection_source", "is_executable"),
    (
        ("(Alpha, Beta)", False),
        ("frozenset({Alpha, Beta})", True),
    ),
)
def test_class_family_collection_operation_requires_provable_runtime_order(
    tmp_path: Path,
    collection_source: str,
    is_executable: bool,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "root.py").write_text("class Root:\n    pass\n", encoding="utf-8")
    (package_dir / "alpha.py").write_text(
        "from .root import Root\n\nclass Alpha(Root):\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "beta.py").write_text(
        "from .root import Root\n\nclass Beta(Root):\n    pass\n",
        encoding="utf-8",
    )
    projection_path = package_dir / "projection.py"
    projection_path.write_text(
        "from .alpha import Alpha\n"
        "from .beta import Beta\n\n"
        f"MEMBERS = {collection_source}\n",
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", projection_path)

    if not is_executable:
        with pytest.raises(
            ValueError,
            match="exactly one complete literal collection",
        ):
            operation.source_edits(snapshot)
        return

    edits = operation.source_edits(snapshot)

    assert edits
    assert any(
        "frozenset(Root.__subclasses__())" in "".join(edit.replacement_lines)
        for edit in edits
        if hasattr(edit, "replacement_lines")
    )


def test_class_family_collection_operation_rejects_shadowed_factory(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "class Root:\n"
        "    pass\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n"
        "class Beta(Root):\n"
        "    pass\n\n"
        "frozenset = lambda values: tuple(values)\n"
        "MEMBERS = frozenset({Alpha, Beta})\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", module_path)

    with pytest.raises(ValueError, match="exactly one complete literal collection"):
        operation.source_edits(snapshot)


def test_class_family_collection_operation_rejects_authority_name_collision(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    taxonomy_path = package_dir / "taxonomy.py"
    projection_path = package_dir / "projection.py"
    taxonomy_path.write_text(
        "class Root:\n"
        "    pass\n\n"
        "class Alpha(Root):\n"
        "    pass\n\n"
        "class Beta(Root):\n"
        "    pass\n",
        encoding="utf-8",
    )
    projection_path.write_text(
        "from .taxonomy import Alpha, Beta\n\nRoot = object\nMEMBERS = (Alpha, Beta)\n",
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _class_family_collection_operation(snapshot, "Root", projection_path)

    with pytest.raises(ValueError, match="authority name 'Root' is rebound"):
        operation.source_edits(snapshot)


def test_semantic_mirror_preserves_public_export_contract(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class ToolAdapter(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
        "\n"
        "class AlphaAdapter(ToolAdapter):\n"
        "    def run(self):\n"
        "        return 'alpha'\n"
        "\n"
        "class BetaAdapter(ToolAdapter):\n"
        "    def run(self):\n"
        "        return 'beta'\n"
        "\n"
        "__all__ = ('AlphaAdapter', 'BetaAdapter')\n",
    )

    findings = SemanticMirrorWithoutDescentDetector().detect(
        parse_python_modules(tmp_path),
        DetectorConfig(),
    )

    assert not any(
        finding.metrics.plan_mapping_name == "__all__" for finding in findings
    )


def test_semantic_mirror_deep_class_collection_requires_complete_runtime_query(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "family.py"
    module_path.write_text(
        "class Root:\n"
        "    pass\n"
        "\n"
        "class Intermediate(Root):\n"
        "    pass\n"
        "\n"
        "class Alpha(Intermediate):\n"
        "    pass\n"
        "\n"
        "class Beta(Intermediate):\n"
        "    pass\n"
        "\n"
        "ALL_MEMBERS = (Intermediate, Alpha, Beta)\n",
        encoding="utf-8",
    )
    modules = parse_python_modules(tmp_path)
    finding = next(
        finding
        for finding in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if finding.metrics.plan_registry_name == "ALL_MEMBERS"
        and finding.evidence[1].symbol == "Root"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)

    record = plan.records[0]
    obstacles_by_declaration = {
        obstacle.executable_declaration_name: obstacle.reason
        for obstacle in record.proof_obstacles
    }

    assert record.status.value == "rejected_by_safety_check"
    assert (
        "exactly one complete literal collection"
        in obstacles_by_declaration["ClassFamilyCollectionSemanticMirrorRecipeBuilder"]
    )
    assert plan.document.recipes == ()


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
        and finding.metrics.plan_mapping_name == "classify:if"
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
        and finding.metrics.plan_mapping_name == "structural_kind:if"
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
        and finding.metrics.plan_mapping_name == "returned_sequence_name:if"
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
        and finding.metrics.plan_mapping_name == "compare:if"
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
        "def compare_edge_fields(field_name):\n"
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
        and finding.metrics.plan_mapping_name == "compare_edge_fields:if"
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
        and finding.metrics.plan_mapping_name == "check:if"
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
    absolute_finding = finding.map_evidence(
        lambda location: replace(
            location,
            file_path=(tmp_path / location.file_path).resolve().as_posix(),
        )
    )

    plan = codemod_plan_from_findings(
        (absolute_finding,),
        selector_context=snapshot,
    )

    assert plan.records[0].status.value == "executable_candidate"
    operation = plan.document.recipes[0].operations[0]
    assert isinstance(operation, DeriveEnumSubsetOperation)
    assert json_report_object(operation)["projection_target"]["target_id"] == next(
        target.target_id
        for target in snapshot.source_index.ast_targets
        if target.is_module and target.file_path == "pkg/codemod.py"
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


def test_semantic_mirror_owned_dataclass_payload_synthesizes_derivation_recipe(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
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
    modules = parse_python_modules(tmp_path)
    finding = next(
        item
        for item in SemanticMirrorWithoutDescentDetector().detect(
            modules,
            DetectorConfig(),
        )
        if item.metrics.plan_source_name == "Report"
        and item.metrics.plan_mapping_name == "Report.to_dict:return"
    )
    snapshot = CodemodSourceSnapshot.from_modules(modules, (finding,))

    plan = codemod_plan_from_findings((finding,), selector_context=snapshot)
    simulation = plan.simulate(snapshot)
    operation = plan.document.recipes[0].operations[0]
    rewritten_source = simulation.simulation.rewritten_sources[
        module_path.as_posix()
    ]

    assert plan.records[0].status.value == "executable_candidate"
    assert isinstance(operation, DeriveDataclassPayloadProjectionOperation)
    assert simulation.is_clean is True
    assert "for field in dataclasses.fields(" in rewritten_source
    assert "                    Report\n" in rewritten_source
    assert "'backend': self.backend" not in rewritten_source
    namespace: dict[str, object] = {}
    exec(rewritten_source, namespace)
    report = namespace["Report"]("cpython", True)
    assert report.to_dict() == {
        "backend": "cpython",
        "parse_valid": True,
    }


def test_semantic_descent_reports_dataclass_subclass_payload_mirror(
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
    assert any(
        certificate.edge.authority_id == report_authority.authority_id
        for certificate in graph.missing_descent_certificates
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
        "def payload(report: Report):\n"
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
        "REPORT = Report(backend='cpython', parse_valid=True)\n",
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
        semantic_descent_module.SemanticDescentGraphBuildRequest,
        "graph",
        unexpected_graph_rebuild,
    )
    cached_graph = build_semantic_descent_graph(
        parse_python_modules(tmp_path, use_parse_cache=False),
        cache_dir=graph_cache_dir,
    )

    assert cached_graph.relations == first_graph.relations
    assert cached_graph.certificates == first_graph.certificates
    assert json_report_object(cached_graph.certificates[0])["status"] == (
        "descends_to_authority"
    )


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
        "REPORT = Report(backend='cpython', parse_valid=True)\n"
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
