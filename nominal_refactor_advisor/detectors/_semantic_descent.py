"""Detectors backed by the semantic-descent graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from ._base import (
    DetectorConfig,
    SemanticDescentGraphIssueDetector,
    SemanticMirrorIssueDetector,
    high_confidence_certified_spec,
)
from ..ast_tools import ParsedModule
from ..models import (
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    SourceLocation,
)
from ..patterns import PatternId
from ..semantic_descent import (
    DescentCertificate,
    PresentationProjectionKind,
    PresentationProjection,
    SemanticAuthority,
    SemanticDescentGraph,
    SemanticDescentGraphCacheIdentity,
    SemanticFact,
    build_semantic_descent_graph,
    normalized_name_variants,
)
from ..registry_identity import class_name_registry_key
from ..taxonomy import CapabilityTag, ObservationTag


class SemanticMirrorClassKeySourceResolver(ABC, metaclass=AutoRegisterMeta):
    """Resolve a registration key source for a mirrored class-family projection."""

    __registry__: ClassVar[dict[str, type["SemanticMirrorClassKeySourceResolver"]]] = {}
    __registry_key__ = "resolver_id"
    __key_extractor__ = staticmethod(class_name_registry_key)
    __skip_if_no_key__ = True
    resolver_order: ClassVar[int] = 100

    @classmethod
    def ordered_resolvers(cls) -> tuple["SemanticMirrorClassKeySourceResolver", ...]:
        return tuple(
            resolver_type()
            for resolver_type in sorted(
                cls.__registry__.values(),
                key=lambda item: (item.resolver_order, item.__name__),
            )
        )

    @abstractmethod
    def key_source_for(
        self,
        fact: SemanticFact,
        projection: PresentationProjection,
        matched_token_set: frozenset[str],
    ) -> str | None:
        raise NotImplementedError


class DictProjectionClassKeySourceResolver(SemanticMirrorClassKeySourceResolver):
    """Use source dict keys when dict values identify the mirrored class."""

    resolver_order = 10

    def key_source_for(
        self,
        fact: SemanticFact,
        projection: PresentationProjection,
        matched_token_set: frozenset[str],
    ) -> str | None:
        del matched_token_set
        fact_aliases = frozenset(fact.normalized_aliases)
        matches = tuple(
            pair.key_source
            for pair in projection.key_value_pairs
            if fact_aliases & frozenset(pair.value_tokens)
        )
        if len(matches) != 1:
            return None
        return matches[0]


class AliasOverlapClassKeySourceResolver(SemanticMirrorClassKeySourceResolver):
    """Use fact aliases when no structured projection key owns the key source."""

    resolver_order = 100

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


class SemanticMirrorWithoutDescentDetector(
    SemanticDescentGraphIssueDetector,
    SemanticMirrorIssueDetector,
):
    """Report presentation projections that mirror a nominal semantic authority."""

    detector_priority = -100
    semantic_mirror_authority_evidence_index = 1
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

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        graph = build_semantic_descent_graph(modules)
        return self._collect_findings_from_graph(graph, modules, config)

    def _collect_findings_from_graph(
        self,
        graph: SemanticDescentGraph,
        modules: list[ParsedModule],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del modules, config
        return [
            self._finding_for_certificate(graph, certificate)
            for certificate in graph.certificates
        ]

    def _finding_for_certificate(
        self,
        graph: SemanticDescentGraph,
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
            self._evidence(
                authority,
                projection_location=projection.location,
                matched_facts=matched_facts,
            ),
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
        for resolver in SemanticMirrorClassKeySourceResolver.ordered_resolvers():
            key_source = resolver.key_source_for(fact, projection, matched_token_set)
            if key_source is not None:
                return key_source
        return None
