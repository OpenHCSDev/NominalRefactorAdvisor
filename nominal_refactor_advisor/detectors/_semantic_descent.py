"""Detectors backed by the semantic-descent graph."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, ClassVar, Iterator

from ..codemod import SemanticMirrorFindingRecipeEvaluator
from ._base import (
    CompactClassIndexMultiProjectionDetector,
    CompactFindingStream,
    CompactProjectionGroups,
    ContextualGlobalCacheContract,
    DetectorConfig,
    SemanticMirrorIssueDetector,
    high_confidence_certified_spec,
)
from ..ast_tools import ParsedModule
from ..class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassProjectionFamily,
)
from ..models import (
    RefactorFinding,
    SemanticMirrorMetricRelation,
)
from ..patterns import PatternId
from ..semantic_descent import (
    CompactSemanticDescentRepository,
    CompactSemanticModuleProjectionFamily,
    MirrorEdge,
    PresentationProjection,
    ResolvedDescentCertificate,
    SemanticAuthorityMirrorPolicy,
    SemanticDescentGraph,
    SemanticDescentGraphCacheIdentity,
    SemanticFact,
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
        if projection.has_structured_key_source:
            return None
        non_class_aliases = tuple(alias for alias in fact.aliases if alias != fact.name)
        for alias in non_class_aliases:
            if matched_token_set & frozenset(normalized_name_variants(alias)):
                return repr(alias)
        for alias in fact.aliases:
            if matched_token_set & frozenset(normalized_name_variants(alias)):
                return repr(alias)
        return None

    def key_pairs_for(
        self,
        facts: tuple[SemanticFact, ...],
        projection: PresentationProjection,
        matched_tokens: tuple[str, ...],
    ) -> tuple[str, ...]:
        matched_token_set = frozenset(matched_tokens)
        return tuple(
            f"{fact.name}={key_source}"
            for fact in facts
            if (
                key_source := self.key_source_for(
                    fact,
                    projection,
                    matched_token_set,
                )
            )
            is not None
        )


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
    CompactClassIndexMultiProjectionDetector,
    ContextualGlobalCacheContract,
    SemanticMirrorIssueDetector,
    SemanticMirrorFindingRecipeEvaluator,
):
    """Report presentation projections that mirror a nominal semantic authority."""

    compact_finding_chunk_size = 64
    class_key_source_resolver: ClassVar[SemanticMirrorClassKeySourceResolver] = (
        SemanticMirrorClassKeySourceResolver()
    )
    module_projection_families = (
        CompactSemanticModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
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

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return list(
            self._stream_findings_from_compact_projection_groups_context(
                projections_by_family,
                context,
                config,
            )
        )

    def _stream_findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> CompactFindingStream:
        del config
        return self._finding_stream_from_repository(
            CompactSemanticDescentRepository.from_projection_groups(
                projections_by_family,
                class_index=CompactClassFamilyIndex.require(context),
            )
        )

    def _finding_stream_from_repository(
        self,
        repository: CompactSemanticDescentRepository,
    ) -> CompactFindingStream:
        compact_resolution = repository.resolve()
        graph_space = compact_resolution.graph_space
        edge_queue: list[MirrorEdge | None] = [
            edge
            for relation in compact_resolution.relation_resolution.relations
            for edge in relation.missing_descent_relations()
        ]
        del compact_resolution

        def finding_chunks() -> Iterator[tuple[RefactorFinding, ...]]:
            chunk: list[RefactorFinding] = []
            for index in range(len(edge_queue)):
                edge = edge_queue[index]
                if edge is None:
                    raise TypeError("semantic mirror edge queue contains a non-edge")
                chunk.append(
                    self._finding_for_resolved_certificate(
                        ResolvedDescentCertificate.from_graph(
                            graph_space,
                            edge.certificate(graph_space),
                        )
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
            self._finding_for_resolved_certificate(
                ResolvedDescentCertificate.from_graph(graph, certificate)
            )
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
        findings: list[RefactorFinding] = []
        for certificate in graph.missing_descent_certificates:
            resolved = ResolvedDescentCertificate.from_graph(
                graph,
                certificate,
            )
            if not any(
                includes_path(Path(evidence.file_path))
                for evidence in resolved.evidence
            ):
                continue
            findings.append(self._finding_for_resolved_certificate(resolved))
        return findings

    def _finding_for_resolved_certificate(
        self,
        resolved: ResolvedDescentCertificate,
    ) -> RefactorFinding:
        certificate = resolved.certificate
        edge = certificate.edge
        authority = resolved.authority
        projection = resolved.projection
        matched_names = resolved.matched_names
        summary = (
            f"`{projection.label}` mirrors {len(resolved.matched_facts)} member(s) of "
            f"`{authority.name}` without a descent path"
        )
        return self.build_finding(
            summary,
            resolved.evidence,
            projection_evidence=projection.location,
            authority_evidence=authority.location,
            title=f"`{projection.label}` mirrors `{authority.name}`",
            why=(
                f"The {projection.kind.value.replace('_', ' ')} at "
                f"{projection.location.file_path}:{projection.location.line} repeats "
                f"{', '.join(matched_names[:6])} from the `{authority.name}` "
                "semantic authority. A later class, enum member, or schema field can "
                "diverge from this hand-maintained view."
            ),
            capability_gap=authority.kind.reporting_capability_gap,
            relation_context=(
                f"{projection.kind.value} has semantic overlap "
                f"{edge.match.tokens} with {authority.kind.value} "
                f"`{authority.name}`; {certificate.missing_derivation_path}"
            ),
            metrics=SemanticAuthorityMirrorPolicy.for_authority(
                authority
            ).semantic_mirror_metrics(
                SemanticMirrorMetricRelation(
                    fact_names=matched_names,
                    projection_name=projection.label,
                    authority_name=authority.name,
                    identity_field_names=edge.match.tokens,
                    class_key_pairs=self.class_key_source_resolver.key_pairs_for(
                        resolved.matched_facts,
                        projection,
                        edge.match.tokens,
                    ),
                )
            ),
        )
