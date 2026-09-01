"""Detectors backed by the semantic-descent graph."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, cast

from ._base import (
    CompactFindingStream,
    CompactMultiModuleProjectionDetectorMixin,
    ContextualGlobalCacheContract,
    DetectorConfig,
    SemanticMirrorIssueDetector,
    compact_class_index_from_projection_groups,
    high_confidence_certified_spec,
)
from ..ast_tools import CollectedFamily, ParsedModule
from ..class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..models import (
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    SourceLocation,
)
from ..patterns import PatternId
from ..semantic_descent import (
    CompactSemanticModuleProjection,
    CompactSemanticModuleProjectionFamily,
    DescentCertificate,
    MirrorEdge,
    PresentationProjectionKind,
    PresentationProjection,
    SemanticAuthority,
    SemanticDescentGraph,
    SemanticDescentGraphCacheIdentity,
    SemanticDescentGraphSpace,
    SemanticFact,
    build_compact_semantic_mirror_resolution,
    normalized_name_variants,
)
from ..taxonomy import CapabilityTag, ObservationTag


class AliasOverlapClassKeySourceResolver:
    """Use fact aliases when no structured projection key owns the key source."""

    def key_source_for(
        self,
        fact: SemanticFact,
        projection: PresentationProjection,
        matched_token_set: frozenset[str],
    ) -> str | None:
        if (
            projection.kind is PresentationProjectionKind.MAPPING_LITERAL
            and projection.key_value_pairs
        ):
            return None
        non_class_aliases = tuple(alias for alias in fact.aliases if alias != fact.name)
        for alias in non_class_aliases:
            if matched_token_set & frozenset(normalized_name_variants(alias)):
                return repr(alias)
        for alias in fact.aliases:
            if matched_token_set & frozenset(normalized_name_variants(alias)):
                return repr(alias)
        return None


class SemanticMirrorClassKeySourceResolver(AliasOverlapClassKeySourceResolver):
    """Prefer a unique structured mapping key, then use alias overlap."""

    def key_source_for(
        self,
        fact: SemanticFact,
        projection: PresentationProjection,
        matched_token_set: frozenset[str],
    ) -> str | None:
        fact_aliases = frozenset(fact.normalized_aliases)
        matches = tuple(
            pair.key_source
            for pair in projection.key_value_pairs
            if fact_aliases & frozenset(pair.value_tokens)
        )
        if len(matches) == 1:
            return matches[0]
        return super().key_source_for(fact, projection, matched_token_set)


class SemanticMirrorWithoutDescentDetector(
    CompactMultiModuleProjectionDetectorMixin,
    ContextualGlobalCacheContract,
    SemanticMirrorIssueDetector,
):
    """Report presentation projections that mirror a nominal semantic authority."""

    compact_finding_chunk_size = 64
    semantic_mirror_authority_evidence_index = 1
    module_projection_families = (
        CompactSemanticModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        compact_class_index_from_projection_groups
    )
    finding_spec = high_confidence_certified_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Semantic mirror should descend to its nominal authority",
        "A raw syntax surface enumerates facts that already have a nominal owner. "
        "The surface is a presentation-level mirror unless it is derived from the "
        "authority's registry, class family, enum, or schema declaration.",
        "one descent path from the nominal authority to every presentation view",
        "presentation-level syntax mirrors a semantic authority without derivation",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.ENUMERATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
            ObservationTag.PROJECTION_DICT,
        ),
    )

    @classmethod
    def context_signature(
        cls,
        modules: tuple[ParsedModule, ...],
        config: DetectorConfig,
    ) -> str:
        del cls, config
        return SemanticDescentGraphCacheIdentity.from_modules(modules).cache_token

    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        semantic_projections = cast(
            tuple[CompactSemanticModuleProjection, ...],
            projections_by_family[CompactSemanticModuleProjectionFamily],
        )
        class_projections = cast(
            tuple[CompactModuleClassProjection, ...],
            projections_by_family[CompactModuleClassProjectionFamily],
        )
        return self._findings_from_compact_resolver(
            semantic_projections,
            class_projections,
        )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        if not isinstance(context, CompactClassFamilyIndex):
            raise TypeError("shared compact class index is unavailable")
        semantic_projections = cast(
            tuple[CompactSemanticModuleProjection, ...],
            projections_by_family[CompactSemanticModuleProjectionFamily],
        )
        class_projections = cast(
            tuple[CompactModuleClassProjection, ...],
            projections_by_family[CompactModuleClassProjectionFamily],
        )
        return self._findings_from_compact_resolver(
            semantic_projections,
            class_projections,
            class_index=context,
        )

    def _stream_findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
        context: object | None,
        config: DetectorConfig,
    ) -> CompactFindingStream:
        del config
        if not isinstance(context, CompactClassFamilyIndex):
            raise TypeError("shared compact class index is unavailable")
        return self._finding_stream_from_compact_resolver(
            cast(
                tuple[CompactSemanticModuleProjection, ...],
                projections_by_family[CompactSemanticModuleProjectionFamily],
            ),
            cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
            class_index=context,
        )

    def _findings_from_compact_resolver(
        self,
        semantic_projections: tuple[CompactSemanticModuleProjection, ...],
        class_projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> list[RefactorFinding]:
        return list(
            self._finding_stream_from_compact_resolver(
                semantic_projections,
                class_projections,
                class_index=class_index,
            )
        )

    def _finding_stream_from_compact_resolver(
        self,
        semantic_projections: tuple[CompactSemanticModuleProjection, ...],
        class_projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> CompactFindingStream:
        graph_space, resolution = build_compact_semantic_mirror_resolution(
            semantic_projections,
            class_projections,
            class_index=class_index,
        )
        edge_queue: list[MirrorEdge | None] = [
            edge
            for relation in resolution.relations
            for edge in relation.missing_descent_relations()
        ]
        del resolution

        def finding_chunks() -> Iterator[tuple[RefactorFinding, ...]]:
            chunk: list[RefactorFinding] = []
            for index in range(len(edge_queue)):
                edge = edge_queue[index]
                if edge is None:
                    raise TypeError("semantic mirror edge queue contains a non-edge")
                chunk.append(
                    self._finding_for_certificate(
                        graph_space,
                        edge.certificate(graph_space),
                    )
                )
                edge_queue[index] = None
                if len(chunk) == self.compact_finding_chunk_size:
                    yield tuple(chunk)
                    chunk = []
            if chunk:
                yield tuple(chunk)

        return CompactFindingStream(len(edge_queue), finding_chunks())

    def _collect_findings_from_graph(
        self,
        graph: SemanticDescentGraph,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del modules, config
        return [
            self._finding_for_certificate(graph, certificate)
            for certificate in graph.missing_descent_certificates
        ]

    def _collect_focused_findings_from_graph(
        self,
        graph: SemanticDescentGraph,
        modules: list[ParsedModule],
        config: DetectorConfig,
        *,
        includes_path: Callable[[Path], bool],
    ) -> list[RefactorFinding]:
        del modules, config
        return [
            self._finding_for_certificate(graph, certificate)
            for certificate in graph.missing_descent_certificates
            if any(
                includes_path(Path(evidence.file_path))
                for evidence in self._certificate_evidence(graph, certificate)
            )
        ]

    def _finding_for_certificate(
        self,
        graph: SemanticDescentGraphSpace,
        certificate: DescentCertificate,
    ) -> RefactorFinding:
        edge = certificate.edge
        authority = graph.authority_catalog.authority_for_edge(edge)
        projection = graph.projection_catalog.projection_for_edge(edge)
        matched_facts = graph.fact_authority_index.facts_for_edge(edge)
        matched_names = tuple(fact.name for fact in matched_facts)
        summary = (
            f"`{projection.label}` mirrors {len(matched_facts)} member(s) of "
            f"`{authority.name}` without a descent path"
        )
        return self.build_finding(
            summary,
            self._certificate_evidence(graph, certificate),
            title=f"`{projection.label}` mirrors `{authority.name}`",
            why=(
                f"The {projection.kind.value.replace('_', ' ')} at "
                f"{projection.location.file_path}:{projection.location.line} repeats "
                f"{', '.join(matched_names[:6])} from the `{authority.name}` "
                "semantic authority. A later class, enum member, or schema field can "
                "diverge from this hand-maintained view."
            ),
            capability_gap=self._capability_gap(authority),
            relation_context=(
                f"{projection.kind.value} has semantic overlap "
                f"{edge.match.tokens} with {authority.kind.value} "
                f"`{authority.name}`; {certificate.missing_derivation_path}"
            ),
            scaffold=self._scaffold(authority),
            codemod_patch=self._codemod_patch(authority, matched_facts),
            metrics=self._metrics(
                authority, projection, matched_facts, edge.match.tokens
            ),
        )

    @classmethod
    def _certificate_evidence(
        cls,
        graph: SemanticDescentGraphSpace,
        certificate: DescentCertificate,
    ) -> tuple[SourceLocation, ...]:
        edge = certificate.edge
        authority = graph.authority_catalog.authority_for_edge(edge)
        projection = graph.projection_catalog.projection_for_edge(edge)
        return cls._evidence(
            authority,
            projection_location=projection.location,
            matched_facts=graph.fact_authority_index.facts_for_edge(edge),
        )

    @staticmethod
    def _evidence(
        authority: SemanticAuthority,
        *,
        projection_location: SourceLocation,
        matched_facts: tuple[SemanticFact, ...],
    ) -> tuple[SourceLocation, ...]:
        return (
            projection_location,
            authority.location,
            *(fact.location for fact in matched_facts),
        )

    @staticmethod
    def _capability_gap(authority: SemanticAuthority) -> str:
        return authority.kind.reporting_capability_gap

    @staticmethod
    def _scaffold(authority: SemanticAuthority) -> str:
        return authority.kind.reporting_scaffold_template.format(
            authority_name=authority.name,
        )

    @staticmethod
    def _codemod_patch(
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
    ) -> str:
        matched_names = ", ".join(fact.name for fact in matched_facts[:8])
        return authority.kind.reporting_codemod_patch_template.format(
            authority_name=authority.name,
            matched_names=matched_names,
        )

    @staticmethod
    def _metrics(
        authority: SemanticAuthority,
        projection: PresentationProjection,
        matched_facts: tuple[SemanticFact, ...],
        matched_tokens: tuple[str, ...],
    ) -> MappingMetrics | RegistrationMetrics:
        names = tuple(fact.name for fact in matched_facts)
        if authority.kind.uses_registration_metrics:
            return RegistrationMetrics.from_class_names(
                registration_site_count=len(matched_facts),
                class_names=names,
                registry_name=projection.label,
                class_key_pairs=SemanticMirrorWithoutDescentDetector._class_key_pairs(
                    matched_facts,
                    projection,
                    matched_tokens,
                ),
            )
        return MappingMetrics.from_field_names(
            mapping_site_count=max(2, len(matched_facts)),
            field_names=names,
            mapping_name=projection.label,
            source_name=authority.name,
            identity_field_names=matched_tokens,
        )

    @staticmethod
    def _class_key_pairs(
        matched_facts: tuple[SemanticFact, ...],
        projection: PresentationProjection,
        matched_tokens: tuple[str, ...],
    ) -> tuple[str, ...]:
        matched_token_set = frozenset(matched_tokens)
        pairs: list[str] = []
        for fact in matched_facts:
            key_source = SemanticMirrorWithoutDescentDetector._class_key_source(
                fact,
                projection,
                matched_token_set,
            )
            if key_source is None:
                continue
            pairs.append(f"{fact.name}={key_source}")
        return tuple(pairs)

    @staticmethod
    def _class_key_source(
        fact: SemanticFact,
        projection: PresentationProjection,
        matched_token_set: frozenset[str],
    ) -> str | None:
        return SemanticMirrorClassKeySourceResolver().key_source_for(
            fact,
            projection,
            matched_token_set,
        )
