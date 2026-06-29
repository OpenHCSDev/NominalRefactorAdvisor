"""Semantic-descent graph for nominal authority and mirror detection.

The descent graph separates three concerns that older detectors often mixed
together: the nominal owner of a semantic fact, presentation-level projections
of those facts, and the certificate explaining whether a projection is derived
or mirrored.  Detectors can then report descent failures without hardcoding one
surface form at a time.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import pickle
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, lru_cache
from itertools import groupby
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import SingleAssignmentAndValueNameProjection
from .ast_tools import ClassFunctionStackNodeVisitor, ParsedModule
from . import class_index as class_index_module
from .cache_paths import default_semantic_descent_cache_dir
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleClassReferenceResolver,
    build_class_family_index,
)
from .collection_algebra import sorted_tuple
from .models import (
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    RegistrationMetrics,
    SourceLocation,
)
from .registry_identity import AutoRegisterClassAuthority, class_name_registry_key
from .semantic_identity import SemanticRoleIdentityToken

_NAME_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+|[0-9]+")
_SEMANTIC_STRING_LITERAL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_CLASS_SUFFIXES = (
    "Detector",
    "Candidate",
    "Handler",
    "Adapter",
    "Renderer",
    "Operation",
    "Strategy",
    "Authority",
    "Report",
    "Config",
    "Payload",
    "Record",
    "Spec",
)
_ENUM_BASE_NAMES = frozenset(("Enum", "IntEnum", "StrEnum"))


class SemanticAuthorityKind(StrEnum):
    """Nominal owner categories that can anchor semantic facts."""

    CLASS_FAMILY = (
        "class_family",
        True,
        "derived class-family registry or polymorphic dispatch",
        True,
        "# Derive presentation views from `{authority_name}` instead of "
        "maintaining a parallel list/dict.\n"
        "for key, member_type in {authority_name}.__registry__.items():\n"
        "    ...",
        "Replace the mirrored projection with a registry-derived view or "
        "polymorphic method on `{authority_name}`. Matched members: "
        "{matched_names}.",
    )
    AUTOREGISTER_FAMILY = (
        "autoregister_family",
        True,
        "derived class-family registry or polymorphic dispatch",
        True,
        "# Derive presentation views from `{authority_name}` instead of "
        "maintaining a parallel list/dict.\n"
        "for key, member_type in {authority_name}.__registry__.items():\n"
        "    ...",
        "Replace the mirrored projection with a registry-derived view or "
        "polymorphic method on `{authority_name}`. Matched members: "
        "{matched_names}.",
    )
    DATACLASS_SCHEMA = (
        "dataclass_schema",
        False,
        "dataclass-schema-derived projection",
        False,
        "# Derive the projection from `{authority_name}` dataclass fields or "
        "move the schema-owned behavior onto the record.",
        "Move the repeated field projection behind `{authority_name}` or derive "
        "it from dataclass fields. Matched fields: {matched_names}.",
    )
    ENUM = (
        "enum",
        False,
        "enum-derived case table or enum-owned behavior",
        False,
        "# Use `{authority_name}` members as the authority and derive secondary "
        "views from the enum.",
        "Move the case table behind `{authority_name}` or derive it by iterating "
        "enum members. Matched members: {matched_names}.",
    )
    FINDING_DECLARED_AUTHORITY = (
        "finding_declared_authority",
        False,
        "detector-declared semantic mirror authority",
        False,
        "# Replace the detector-observed mirror `{authority_name}` with a "
        "projection derived from its nominal authority.",
        "Derive `{authority_name}` from the nominal authority instead of "
        "maintaining the detector-observed mirror. Matched facts: {matched_names}.",
    )

    def __new__(
        cls,
        value: str,
        is_class_family_like: bool,
        reporting_capability_gap: str,
        uses_registration_metrics: bool,
        reporting_scaffold_template: str,
        reporting_codemod_patch_template: str,
    ) -> "SemanticAuthorityKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_class_family_like = is_class_family_like
        member._reporting_capability_gap = reporting_capability_gap
        member._uses_registration_metrics = uses_registration_metrics
        member._reporting_scaffold_template = reporting_scaffold_template
        member._reporting_codemod_patch_template = reporting_codemod_patch_template
        return member

    @property
    def is_class_family_like(self) -> bool:
        return self._is_class_family_like

    @property
    def reporting_capability_gap(self) -> str:
        return self._reporting_capability_gap

    @property
    def uses_registration_metrics(self) -> bool:
        return self._uses_registration_metrics

    @property
    def reporting_scaffold_template(self) -> str:
        return self._reporting_scaffold_template

    @property
    def reporting_codemod_patch_template(self) -> str:
        return self._reporting_codemod_patch_template


class SemanticFactKind(StrEnum):
    """Facts owned by a semantic authority."""

    CLASS_MEMBER = "class_member"
    DATACLASS_FIELD = "dataclass_field"
    ENUM_MEMBER = "enum_member"
    FINDING_EVIDENCE = "finding_evidence"


class PresentationProjectionKind(StrEnum):
    """Raw presentation shapes that may mirror a semantic authority."""

    CALL_LITERAL = "call_literal"
    COLLECTION_LITERAL = "collection_literal"
    DETECTOR_FINDING = "detector_finding"
    MAPPING_LITERAL = "mapping_literal"
    BRANCH_LITERAL = "branch_literal"
    MATCH_LITERAL = "match_literal"


class PresentationTokenKind(StrEnum):
    """Source syntax category for one normalized presentation token."""

    STRING_LITERAL = "string_literal"
    NAME_REFERENCE = "name_reference"
    QUALIFIED_ATTRIBUTE = "qualified_attribute"


class PresentationTokenRole(StrEnum):
    """Structural role of one token within its presentation surface."""

    CALL_ARGUMENT = "call_argument"
    CALL_KEYWORD = "call_keyword"
    CALL_TARGET = "call_target"
    COLLECTION_ITEM = "collection_item"
    DICT_KEY = "dict_key"
    DICT_VALUE = "dict_value"
    CONDITION = "condition"
    MATCH_CASE = "match_case"


class DescentStatus(StrEnum):
    """Whether a presentation descends to its semantic authority."""

    MIRRORED_WITHOUT_DESCENT = "mirrored_without_descent"


class SemanticDescentGraphCacheReadError(RuntimeError):
    """Raised when an existing semantic-descent graph cache entry is invalid."""


@dataclass(frozen=True)
class SemanticDescentGraphCacheSchema:
    """Nominal schema identity for persisted semantic-descent graph entries."""

    version: int = 5
    digest_size: int = 16


semantic_descent_graph_cache_schema = SemanticDescentGraphCacheSchema()


@dataclass(frozen=True)
class SemanticDescentImplementationSignature:
    """Implementation identity for graph semantics that affect cache validity."""

    source_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def current(cls) -> "SemanticDescentImplementationSignature":
        return cls(
            source_hashes=tuple(
                sorted(
                    (
                        (path.name, _source_file_hash(path))
                        for path in _semantic_descent_implementation_paths()
                    ),
                    key=lambda item: item[0],
                )
            )
        )


@dataclass(frozen=True)
class SemanticDescentModuleSignature:
    """Parsed module identity used to invalidate semantic-descent graphs."""

    path: str
    parsed_import_name: str
    is_package_init: bool
    source_hash: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> "SemanticDescentModuleSignature":
        return cls(
            path=str(module.path.resolve()),
            parsed_import_name=module.module_name,
            is_package_init=module.is_package_init,
            source_hash=_text_hash(module.source),
        )


@dataclass(frozen=True)
class SemanticDescentGraphCacheIdentity:
    """Complete invalidation identity for one semantic-descent graph."""

    schema: SemanticDescentGraphCacheSchema
    implementation: SemanticDescentImplementationSignature
    python_version: tuple[int, int]
    modules: tuple[SemanticDescentModuleSignature, ...]

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> "SemanticDescentGraphCacheIdentity":
        return cls(
            schema=semantic_descent_graph_cache_schema,
            implementation=SemanticDescentImplementationSignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            modules=tuple(
                sorted(
                    (
                        SemanticDescentModuleSignature.from_module(module)
                        for module in modules
                    ),
                    key=lambda item: item.path,
                )
            ),
        )

    @property
    def cache_token(self) -> str:
        return hashlib.blake2s(
            repr(self).encode("utf-8"),
            digest_size=self.schema.digest_size,
        ).hexdigest()


@dataclass(frozen=True)
class SemanticAuthorityReference:
    """Reference to one nominal semantic authority."""

    authority_id: str


@dataclass(frozen=True)
class SemanticFactReference(SemanticAuthorityReference):
    """Reference to one semantic fact under a nominal authority."""

    fact_id: str


@dataclass(frozen=True)
class SemanticProjectionReference:
    """Reference to one presentation projection."""

    projection_id: str


@dataclass(frozen=True)
class SemanticAuthorityProjectionReference(
    SemanticAuthorityReference,
    SemanticProjectionReference,
):
    """Reference binding a nominal authority to a presentation projection."""


@dataclass(frozen=True)
class SemanticFact(SemanticFactReference):
    """One semantic member owned by a nominal authority."""

    kind: SemanticFactKind
    name: str
    aliases: tuple[str, ...]
    location: SourceLocation

    @cached_property
    def normalized_aliases(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                variant
                for alias in self.aliases
                for variant in normalized_name_variants(alias)
            }
        )


@dataclass(frozen=True)
class SemanticAuthority(SemanticAuthorityReference):
    """Nominal source of truth for a semantic fact family."""

    kind: SemanticAuthorityKind
    name: str
    location: SourceLocation
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class PresentationToken:
    """One token observed inside a presentation-level syntax surface."""

    value: str
    kind: PresentationTokenKind
    role: PresentationTokenRole
    qualifier: str | None = None


@dataclass(frozen=True)
class PresentationAuthorityConstruction:
    """Nominal construction observed in the owner that contains a projection."""

    type_name: str
    field_tokens: tuple[str, ...]


ConstructionAuthorityPredicate: TypeAlias = Callable[
    [PresentationAuthorityConstruction, SemanticAuthority],
    bool,
]


@dataclass(frozen=True)
class PresentationKeyValuePair:
    """One source-level key/value binding inside a presentation projection."""

    key_source: str
    value_source: str
    value_tokens: tuple[str, ...]
    value_class_symbols: tuple[str, ...] = ()

    @classmethod
    def from_nodes(
        cls,
        *,
        key: ast.AST,
        value: ast.AST,
        class_reference_resolver: ModuleClassReferenceResolver,
    ) -> "PresentationKeyValuePair":
        value_class_symbols = class_reference_resolver.symbols_for_node(value)
        return cls(
            key_source=ast.unparse(key),
            value_source=ast.unparse(value),
            value_tokens=sorted_tuple(
                {
                    token.value
                    for token in PresentationTokenProjection.tokens_for_node(
                        value,
                        PresentationTokenRole.DICT_VALUE,
                    )
                }
                | _class_reference_normalized_tokens(
                    class_reference_resolver.class_index,
                    value_class_symbols,
                )
            ),
            value_class_symbols=value_class_symbols,
        )


@dataclass(frozen=True)
class PresentationProjection(SemanticProjectionReference):
    """Raw syntax projection that may duplicate a semantic fact family."""

    kind: PresentationProjectionKind
    label: str
    owner_symbol: str
    location: SourceLocation
    tokens: tuple[PresentationToken, ...]
    source_text: str
    owner_constructions: tuple[PresentationAuthorityConstruction, ...] = ()
    key_value_pairs: tuple[PresentationKeyValuePair, ...] = ()
    class_symbols: tuple[str, ...] = ()

    @cached_property
    def normalized_tokens(self) -> tuple[str, ...]:
        return sorted_tuple({token.value for token in self.tokens})


class ProjectionSuppressionIntent(StrEnum):
    """Assignment-label intent tokens that mark lexical suppression vocabularies."""

    EXCLUDE = "exclude"
    EXCLUDED = "excluded"
    GENERIC = "generic"
    OPAQUE = "opaque"
    STOP = "stop"
    STOPWORD = "stopword"
    STOPWORDS = "stopwords"
    WEAK = "weak"


@dataclass(frozen=True)
class ProjectionSuppressionPolicy:
    """Classify projections whose labels declare non-authoritative suppression sets."""

    label: str

    @cached_property
    def label_tokens(self) -> frozenset[str]:
        return NormalizeNameProjection.token_set(self.label)

    def suppresses_semantic_projection(self) -> bool:
        return any(
            intent.value in self.label_tokens for intent in ProjectionSuppressionIntent
        )


@dataclass(frozen=True)
class SemanticAuthorityAffinityPolicy:
    """Compare authority/projection names without generic role-token affinity."""

    authority_name: str
    projection_label: str
    projection_owner_symbol: str
    projection_location_symbol: str

    @cached_property
    def authority_tokens(self) -> frozenset[str]:
        return self._specific_tokens(self.authority_name)

    @cached_property
    def projection_tokens(self) -> frozenset[str]:
        return self._specific_tokens(
            " ".join(
                (
                    self.projection_label,
                    self.projection_owner_symbol,
                    self.projection_location_symbol,
                )
            )
        )

    def has_authority_affinity(self) -> bool:
        return bool(self.authority_tokens & self.projection_tokens)

    @staticmethod
    def _specific_tokens(raw_name: str) -> frozenset[str]:
        weak_tokens = SemanticRoleIdentityToken.authority_affinity_weak_values()
        return frozenset(
            token
            for token in NormalizeNameProjection.token_set(raw_name)
            if token not in weak_tokens
        )


@dataclass(frozen=True)
class MirrorEdge(SemanticAuthorityProjectionReference):
    """Candidate relation between a raw projection and a nominal authority."""

    matched_fact_ids: tuple[str, ...]
    matched_tokens: tuple[str, ...]
    coverage_ratio: float


@dataclass(frozen=True)
class SemanticMirrorEdgeCandidate:
    """Resolved projection/fact overlap before policy admissibility filtering."""

    projection: PresentationProjection
    authority: SemanticAuthority
    facts: tuple[SemanticFact, ...]
    matched_fact_ids: tuple[str, ...]
    matched_tokens: tuple[str, ...]
    coverage_ratio: float

    @cached_property
    def matched_facts(self) -> tuple[SemanticFact, ...]:
        matched_fact_ids = frozenset(self.matched_fact_ids)
        return tuple(fact for fact in self.facts if fact.fact_id in matched_fact_ids)

    @cached_property
    def branch_like_projection(self) -> bool:
        return self.projection.kind in (
            PresentationProjectionKind.BRANCH_LITERAL,
            PresentationProjectionKind.MATCH_LITERAL,
        )


class SemanticAuthorityMirrorPolicy(ABC, metaclass=AutoRegisterMeta):
    """Authority-kind-specific mirror admissibility and descent rules."""

    __registry__: ClassVar[
        dict[SemanticAuthorityKind, type["SemanticAuthorityMirrorPolicy"]]
    ] = {}
    __registry_key__ = "authority_kind"
    __skip_if_no_key__ = True

    authority_kind: ClassVar[SemanticAuthorityKind | None] = None
    authority_qualified_token_reference_admitted: ClassVar[bool] = True
    foreign_qualified_attribute_token_reference_admitted: ClassVar[bool] = False
    dataclass_authority_selected: ClassVar[bool] = False

    @classmethod
    def for_authority(
        cls,
        authority: SemanticAuthority,
    ) -> "SemanticAuthorityMirrorPolicy":
        return cls.__registry__[authority.kind]()

    @classmethod
    def registered_authority_kinds(cls) -> frozenset[SemanticAuthorityKind]:
        return frozenset(cls.__registry__)

    def edge_is_admissible(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        del resolver, candidate
        return True

    def projection_descends_to_authority(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        del resolver, candidate
        return False


class ClassFamilyLikeMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Shared policy for class-family authorities and AutoRegister families."""

    foreign_qualified_attribute_token_reference_admitted = True

    def edge_is_admissible(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        return not (
            candidate.projection.kind is PresentationProjectionKind.BRANCH_LITERAL
            and len(candidate.matched_fact_ids) <= 2
            and not resolver.projection_has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
        )


class ClassFamilyMirrorPolicy(ClassFamilyLikeMirrorPolicy):
    """Mirror policy for conventional class-family authorities."""

    authority_kind = SemanticAuthorityKind.CLASS_FAMILY


class AutoRegisterFamilyMirrorPolicy(ClassFamilyLikeMirrorPolicy):
    """Mirror policy for AutoRegisterMeta-backed class-family authorities."""

    authority_kind = SemanticAuthorityKind.AUTOREGISTER_FAMILY


class DataclassSchemaMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Mirror policy for dataclass schema authorities."""

    authority_kind = SemanticAuthorityKind.DATACLASS_SCHEMA
    dataclass_authority_selected = True

    def edge_is_admissible(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        if (
            candidate.coverage_ratio < 1.0
            and len(candidate.matched_fact_ids) <= 2
            and not resolver.projection_has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        if (
            candidate.branch_like_projection
            and not resolver.dataclass_branch_projection_has_field_syntax(
                candidate.projection,
                frozenset(candidate.matched_tokens),
            )
            and not resolver.projection_has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        if (
            resolver.projection_materializes_any_dataclass_authority(
                candidate.projection,
            )
            and not resolver.projection_has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
            and not resolver.projection_has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        return not (
            candidate.coverage_ratio < 1.0
            and (
                resolver.projection_descends_to_any_dataclass_authority(
                    candidate.projection,
                )
                or resolver.projection_materializes_any_dataclass_authority(
                    candidate.projection,
                )
            )
        )

    def projection_descends_to_authority(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        return (
            resolver.dataclass_projection_descends_to_authority(
                candidate.projection,
                candidate.authority,
            )
            or resolver.projection_owner_constructs_dataclass_authority(
                candidate.projection,
                candidate.authority,
                candidate.matched_facts,
            )
            or resolver.projection_shares_dataclass_base_with_authority(
                candidate.projection,
                candidate.authority,
            )
        )


class EnumMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Mirror policy for enum authorities."""

    authority_kind = SemanticAuthorityKind.ENUM
    authority_qualified_token_reference_admitted = False

    def edge_is_admissible(
        self,
        resolver: "SemanticMirrorResolver",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        if (
            candidate.branch_like_projection
            and not resolver.enum_branch_projection_has_case_syntax(
                candidate.projection,
                frozenset(candidate.matched_tokens),
            )
            and not resolver.projection_has_enum_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
            and not resolver.projection_has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        return not (
            len(candidate.matched_fact_ids) <= 2
            and not resolver.projection_has_enum_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
            and not resolver.projection_has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        )


class FindingDeclaredAuthorityMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Mirror policy for detector findings projected into the descent graph."""

    authority_kind = SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY


@dataclass(frozen=True)
class _FactTokenReference(SemanticFactReference):
    """Indexed lookup row from one normalized token to one semantic fact."""


FactsByAuthorityId: TypeAlias = dict[str, tuple[SemanticFact, ...]]
AuthorityIdsByName: TypeAlias = dict[str, tuple[str, ...]]
FactRefsByToken: TypeAlias = dict[str, tuple[_FactTokenReference, ...]]
FactMatchesByAuthority: TypeAlias = dict[str, dict[str, set[str]]]
ConstructionAuthorityCacheKey: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class SemanticFactAuthorityIndex:
    """Authority-owned facts indexed without runtime default fallbacks."""

    facts: tuple[SemanticFact, ...]

    @cached_property
    def by_id(self) -> dict[str, SemanticFact]:
        return {fact.fact_id: fact for fact in self.facts}

    @cached_property
    def by_authority_id(self) -> FactsByAuthorityId:
        ordered_facts = sorted_tuple(
            self.facts,
            key=lambda fact: (fact.authority_id, fact.name, fact.fact_id),
        )
        return {
            authority_id: tuple(facts)
            for authority_id, facts in groupby(
                ordered_facts,
                key=lambda fact: fact.authority_id,
            )
        }

    def facts_for_authority(self, authority_id: str) -> tuple[SemanticFact, ...]:
        return self.by_authority_id[authority_id]


@dataclass(frozen=True)
class SemanticAuthorityNameIndex:
    """Authority ids grouped by nominal source name."""

    authorities: tuple[SemanticAuthority, ...]

    @cached_property
    def by_name(self) -> AuthorityIdsByName:
        ordered_authorities = sorted_tuple(
            self.authorities,
            key=lambda authority: (authority.name, authority.authority_id),
        )
        return {
            name: tuple(authority.authority_id for authority in authorities)
            for name, authorities in groupby(
                ordered_authorities,
                key=lambda authority: authority.name,
            )
        }

    def contains_name(self, name: str) -> bool:
        return name in self.by_name

    def authority_ids_for_name(self, name: str) -> tuple[str, ...]:
        return self.by_name[name]


@dataclass(frozen=True)
class SemanticFactTokenIndex:
    """Fact references grouped by normalized presentation token."""

    facts: tuple[SemanticFact, ...]

    @cached_property
    def by_token(self) -> FactRefsByToken:
        ref_tokens = tuple(
            (alias, _FactTokenReference(fact.authority_id, fact.fact_id))
            for fact in self.facts
            for alias in fact.normalized_aliases
        )
        ordered_ref_tokens = sorted_tuple(
            ref_tokens,
            key=lambda item: (item[0], item[1].authority_id, item[1].fact_id),
        )
        return {
            token: tuple(ref for _, ref in token_refs)
            for token, token_refs in groupby(
                ordered_ref_tokens,
                key=lambda item: item[0],
            )
        }

    def contains_token(self, token: str) -> bool:
        return token in self.by_token

    def refs_for_token(self, token: str) -> tuple[_FactTokenReference, ...]:
        return self.by_token[token]


@dataclass(frozen=True)
class SemanticFactTokenMatch(SemanticFactReference):
    """One presentation-token match against one authority-owned fact."""

    token_value: str


@dataclass(frozen=True)
class SemanticFactMatchIndex:
    """Projection matches grouped by authority and fact."""

    matches: tuple[SemanticFactTokenMatch, ...]

    @cached_property
    def by_authority(self) -> FactMatchesByAuthority:
        ordered_matches = sorted_tuple(
            self.matches,
            key=lambda match: (match.authority_id, match.fact_id, match.token_value),
        )
        return {
            authority_id: self._fact_matches(tuple(authority_matches))
            for authority_id, authority_matches in groupby(
                ordered_matches,
                key=lambda match: match.authority_id,
            )
        }

    @staticmethod
    def _fact_matches(
        matches: tuple[SemanticFactTokenMatch, ...],
    ) -> dict[str, set[str]]:
        return {
            fact_id: {match.token_value for match in fact_matches}
            for fact_id, fact_matches in groupby(
                matches,
                key=lambda match: match.fact_id,
            )
        }


@dataclass(frozen=True)
class ProjectionClassSymbolFactMatcher:
    """Match resolved projection class references to class-family facts."""

    projection: PresentationProjection
    class_index: ClassFamilyIndex
    authority_by_id: dict[str, SemanticAuthority]
    fact_authority_index: SemanticFactAuthorityIndex

    def matches(self) -> tuple[SemanticFactTokenMatch, ...]:
        matches: list[SemanticFactTokenMatch] = []
        for class_symbol in self.projection.class_symbols:
            indexed_class = self.class_index.class_for(class_symbol)
            if indexed_class is None:
                continue
            matches.extend(self._matches_for_indexed_class(indexed_class))
        return tuple(matches)

    def _matches_for_indexed_class(
        self,
        indexed_class: IndexedClass,
    ) -> tuple[SemanticFactTokenMatch, ...]:
        matches: list[SemanticFactTokenMatch] = []
        for authority_id in self.class_index.ancestor_symbols(indexed_class.symbol):
            authority = self.authority_by_id.get(authority_id)
            if authority is None or not authority.kind.is_class_family_like:
                continue
            fact_id = f"{authority_id}:{indexed_class.symbol}"
            if fact_id not in self.fact_authority_index.by_id:
                continue
            matches.extend(
                SemanticFactTokenMatch(
                    authority_id=authority_id,
                    fact_id=fact_id,
                    token_value=token_value,
                )
                for token_value in normalized_name_variants(indexed_class.simple_name)
            )
        return tuple(matches)


@dataclass(frozen=True)
class DescentCertificate:
    """Concrete proof object for one semantic-descent classification."""

    status: DescentStatus
    edge: MirrorEdge
    missing_derivation_path: str


@dataclass(frozen=True)
class SemanticDescentGraphSpace:
    authorities: tuple[SemanticAuthority, ...]
    facts: tuple[SemanticFact, ...]
    projections: tuple[PresentationProjection, ...]

    @cached_property
    def fact_authority_index(self) -> SemanticFactAuthorityIndex:
        return SemanticFactAuthorityIndex(self.facts)

    @cached_property
    def authority_name_index(self) -> SemanticAuthorityNameIndex:
        return SemanticAuthorityNameIndex(self.authorities)

    @cached_property
    def fact_token_index(self) -> SemanticFactTokenIndex:
        return SemanticFactTokenIndex(self.facts)

    @cached_property
    def facts_by_authority_id(self) -> FactsByAuthorityId:
        return self.fact_authority_index.by_authority_id

    @cached_property
    def authority_by_id(self) -> dict[str, SemanticAuthority]:
        return {authority.authority_id: authority for authority in self.authorities}

    @cached_property
    def projection_by_id(self) -> dict[str, PresentationProjection]:
        return {projection.projection_id: projection for projection in self.projections}


@dataclass(frozen=True)
class SemanticDescentGraph(SemanticDescentGraphSpace):
    """Repository-level graph of authorities, projections, and descent failures."""

    mirror_edges: tuple[MirrorEdge, ...]
    certificates: tuple[DescentCertificate, ...]


def semantic_descent_finding_authority_id(finding: RefactorFinding) -> str:
    return f"finding:{finding.stable_id}:authority"


def semantic_descent_finding_projection_id(finding: RefactorFinding) -> str:
    return f"finding:{finding.stable_id}:projection"


@dataclass(frozen=True)
class FindingBackedSemanticDescentGraphRequest:
    """Graph request for detector findings already classified as semantic mirrors."""

    findings: tuple[RefactorFinding, ...]
    semantic_mirror_detector_ids: frozenset[str]
    authority_evidence_indices: tuple[tuple[str, int | None], ...] = ()

    @classmethod
    def from_inputs(
        cls,
        findings: tuple[RefactorFinding, ...],
        *,
        semantic_mirror_detector_ids: frozenset[str],
        authority_evidence_index_by_detector_id: Mapping[str, int | None],
    ) -> "FindingBackedSemanticDescentGraphRequest":
        return cls(
            findings=tuple(
                finding
                for finding in findings
                if finding.detector_id in semantic_mirror_detector_ids
            ),
            semantic_mirror_detector_ids=semantic_mirror_detector_ids,
            authority_evidence_indices=tuple(
                sorted(authority_evidence_index_by_detector_id.items())
            ),
        )

    @cached_property
    def authority_evidence_index_by_detector_id(self) -> dict[str, int | None]:
        return dict(self.authority_evidence_indices)

    def build_graph(self) -> SemanticDescentGraph:
        authorities: list[SemanticAuthority] = []
        facts: list[SemanticFact] = []
        projections: list[PresentationProjection] = []
        edges: list[MirrorEdge] = []
        certificates: list[DescentCertificate] = []
        for finding in self.findings:
            authority = FindingBackedAuthorityProjection.authority(
                finding,
                self.authority_evidence_index_by_detector_id,
            )
            finding_facts = FindingBackedFactProjection.facts(finding, authority)
            projection = FindingBackedPresentationProjection.projection(finding)
            edge = FindingBackedMirrorEdgeProjection.edge(
                authority,
                finding_facts,
                projection,
            )
            authorities.append(authority)
            facts.extend(finding_facts)
            projections.append(projection)
            edges.append(edge)
            certificates.append(FindingBackedCertificateProjection.certificate(finding, edge))
        return SemanticDescentGraph(
            authorities=sorted_tuple(authorities, key=lambda item: item.authority_id),
            facts=sorted_tuple(facts, key=lambda item: item.fact_id),
            projections=sorted_tuple(projections, key=lambda item: item.projection_id),
            mirror_edges=sorted_tuple(edges, key=lambda item: item.projection_id),
            certificates=sorted_tuple(
                certificates,
                key=lambda item: item.edge.projection_id,
            ),
        )


class FindingBackedAuthorityProjection:
    """Project detector finding evidence onto a nominal semantic authority."""

    @classmethod
    def authority(
        cls,
        finding: RefactorFinding,
        authority_evidence_index_by_detector_id: Mapping[str, int | None],
    ) -> SemanticAuthority:
        authority_location = cls.authority_location(
            finding,
            authority_evidence_index_by_detector_id,
        )
        authority_id = semantic_descent_finding_authority_id(finding)
        return SemanticAuthority(
            authority_id=authority_id,
            kind=SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY,
            name=authority_location.symbol,
            location=authority_location,
            fact_ids=tuple(
                FindingBackedFactProjection.fact_id(authority_id, index)
                for index, _fact_name in enumerate(
                    FindingBackedFactProjection.fact_names(finding)
                )
            ),
        )

    @classmethod
    def authority_location(
        cls,
        finding: RefactorFinding,
        authority_evidence_index_by_detector_id: Mapping[str, int | None],
    ) -> SourceLocation:
        evidence_index = authority_evidence_index_by_detector_id.get(
            finding.detector_id
        )
        if evidence_index is not None and evidence_index < len(finding.evidence):
            return finding.evidence[evidence_index]
        return FindingBackedPresentationProjection.projection_location(finding)


class FindingBackedPresentationProjection:
    """Project detector finding evidence onto a presentation projection."""

    @classmethod
    def projection(cls, finding: RefactorFinding) -> PresentationProjection:
        fact_names = FindingBackedFactProjection.fact_names(finding)
        return PresentationProjection(
            projection_id=semantic_descent_finding_projection_id(finding),
            kind=PresentationProjectionKind.DETECTOR_FINDING,
            label=finding.title,
            owner_symbol=finding.detector_id,
            location=cls.projection_location(finding),
            tokens=tuple(
                PresentationToken(
                    value=fact_name,
                    kind=PresentationTokenKind.STRING_LITERAL,
                    role=PresentationTokenRole.COLLECTION_ITEM,
                )
                for fact_name in fact_names
            ),
            source_text=finding.stable_id,
        )

    @staticmethod
    def projection_location(finding: RefactorFinding) -> SourceLocation:
        if finding.evidence:
            return finding.evidence[0]
        return SourceLocation("", 0, finding.title)


class FindingBackedFactProjection:
    """Project detector finding metrics and evidence into semantic facts."""

    @classmethod
    def facts(
        cls,
        finding: RefactorFinding,
        authority: SemanticAuthority,
    ) -> tuple[SemanticFact, ...]:
        fact_names = cls.fact_names(finding)
        fact_location = FindingBackedPresentationProjection.projection_location(finding)
        return tuple(
            SemanticFact(
                authority_id=authority.authority_id,
                fact_id=cls.fact_id(authority.authority_id, index),
                kind=SemanticFactKind.FINDING_EVIDENCE,
                name=fact_name,
                aliases=(fact_name,),
                location=fact_location,
            )
            for index, fact_name in enumerate(fact_names)
        )

    @staticmethod
    def fact_id(authority_id: str, index: int) -> str:
        return f"{authority_id}:fact:{index}"

    @staticmethod
    def fact_names(finding: RefactorFinding) -> tuple[str, ...]:
        metric_names = FindingMetricsFactExtractor.fact_names_for(finding.metrics)
        if metric_names:
            return metric_names
        evidence_names = sorted_tuple(location.symbol for location in finding.evidence)
        if evidence_names:
            return evidence_names
        return (finding.title,)


class FindingBackedMirrorEdgeProjection:
    """Project finding-backed authorities and facts into mirror edges."""

    @staticmethod
    def edge(
        authority: SemanticAuthority,
        facts: tuple[SemanticFact, ...],
        projection: PresentationProjection,
    ) -> MirrorEdge:
        return MirrorEdge(
            authority_id=authority.authority_id,
            projection_id=projection.projection_id,
            matched_fact_ids=tuple(fact.fact_id for fact in facts),
            matched_tokens=sorted_tuple(
                {
                    variant
                    for fact in facts
                    for variant in normalized_name_variants(fact.name)
                }
            ),
            coverage_ratio=1.0,
        )


class FindingBackedCertificateProjection:
    """Project detector finding relation context into descent certificates."""

    @staticmethod
    def certificate(
        finding: RefactorFinding,
        edge: MirrorEdge,
    ) -> DescentCertificate:
        return DescentCertificate(
            status=DescentStatus.MIRRORED_WITHOUT_DESCENT,
            edge=edge,
            missing_derivation_path=(
                finding.relation_context
                or "detector finding reports a mirror without a derivation path"
            ),
        )


def build_finding_backed_semantic_descent_graph(
    findings: tuple[RefactorFinding, ...],
    *,
    semantic_mirror_detector_ids: frozenset[str],
    authority_evidence_index_by_detector_id: Mapping[str, int | None],
) -> SemanticDescentGraph:
    """Project semantic-mirror detector findings into descent graph certificates."""

    request = FindingBackedSemanticDescentGraphRequest.from_inputs(
        findings,
        semantic_mirror_detector_ids=semantic_mirror_detector_ids,
        authority_evidence_index_by_detector_id=authority_evidence_index_by_detector_id,
    )
    return _build_finding_backed_semantic_descent_graph_cached(request)


@lru_cache(maxsize=16)
def _build_finding_backed_semantic_descent_graph_cached(
    request: FindingBackedSemanticDescentGraphRequest,
) -> SemanticDescentGraph:
    return request.build_graph()


class FindingMetricsFactExtractor(ABC, metaclass=AutoRegisterMeta):
    """Registered projection from finding metrics to semantic fact names."""

    __registry__: ClassVar[
        dict[type[FindingMetrics], type["FindingMetricsFactExtractor"]]
    ] = {}
    __registry_key__ = "metrics_type"
    __skip_if_no_key__ = True
    metrics_type: ClassVar[type[FindingMetrics]]

    @abstractmethod
    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def fact_names_for(cls, metrics: FindingMetrics) -> tuple[str, ...]:
        for extractor_type in cls.__registry__.values():
            if isinstance(metrics, extractor_type.metrics_type):
                return extractor_type().fact_names(metrics)
        return ()


class MappingMetricsFactExtractor(FindingMetricsFactExtractor):
    """Use mapping field names as semantic facts."""

    metrics_type: ClassVar[type[FindingMetrics]] = MappingMetrics

    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        if not isinstance(metrics, MappingMetrics):
            return ()
        return metrics.field_names or metrics.identity_field_names


class RegistrationMetricsFactExtractor(FindingMetricsFactExtractor):
    """Use registered class names as semantic facts."""

    metrics_type: ClassVar[type[FindingMetrics]] = RegistrationMetrics

    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        if not isinstance(metrics, RegistrationMetrics):
            return ()
        return metrics.class_names or tuple(
            class_key_pair.split("=", 1)[0]
            for class_key_pair in metrics.class_key_pairs
        )


class SemanticDescentGraphCacheLookup:
    """Base result of loading one semantic-descent graph cache entry."""

    graph: SemanticDescentGraph | None = None


@dataclass(frozen=True)
class SemanticDescentGraphCacheDisabled(SemanticDescentGraphCacheLookup):
    """Graph cache lookup result when persistence is disabled."""


@dataclass(frozen=True)
class SemanticDescentGraphCacheMiss(SemanticDescentGraphCacheLookup):
    """Graph cache lookup result when no matching entry exists."""


@dataclass(frozen=True)
class SemanticDescentGraphCacheHit(SemanticDescentGraphCacheLookup):
    """Graph cache lookup result with a valid graph payload."""

    graph: SemanticDescentGraph


@dataclass(frozen=True)
class SemanticDescentGraphCache:
    """Persistent graph cache for repo-wide semantic descent context."""

    storage_root: Path | None

    def load(
        self,
        identity: SemanticDescentGraphCacheIdentity,
    ) -> SemanticDescentGraphCacheLookup:
        if self.storage_root is None:
            return SemanticDescentGraphCacheDisabled()
        try:
            with self._entry_path(identity).open("rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError:
            return SemanticDescentGraphCacheMiss()
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise SemanticDescentGraphCacheReadError(
                f"Could not read semantic-descent graph cache entry {identity.cache_token}"
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticDescentGraphCacheReadError(
                f"Semantic-descent graph cache entry {identity.cache_token} is not a mapping"
            )
        if payload.get("identity") != identity:
            return SemanticDescentGraphCacheMiss()
        graph = payload.get("graph")
        if not isinstance(graph, SemanticDescentGraph):
            raise SemanticDescentGraphCacheReadError(
                f"Semantic-descent graph cache entry {identity.cache_token} has invalid graph payload"
            )
        return SemanticDescentGraphCacheHit(graph)

    def store(
        self,
        identity: SemanticDescentGraphCacheIdentity,
        graph: SemanticDescentGraph,
    ) -> None:
        if self.storage_root is None:
            return
        try:
            self.storage_root.mkdir(parents=True, exist_ok=True)
            with self._entry_path(identity).open("wb") as handle:
                pickle.dump(
                    {"identity": identity, "graph": graph},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except OSError:
            return

    def _entry_path(self, identity: SemanticDescentGraphCacheIdentity) -> Path:
        if self.storage_root is None:
            raise ValueError("semantic descent graph cache directory is disabled")
        return self.storage_root / f"{identity.cache_token}.pickle"


def build_semantic_descent_graph(
    modules: list[ParsedModule],
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> SemanticDescentGraph:
    """Build the cached semantic-descent graph for parsed modules."""

    module_tuple = tuple(modules)
    resolved_cache_dir = (
        cache_dir
        if cache_dir is not None
        else SemanticDescentGraphCacheDirAuthority(module_tuple).cache_dir()
    )
    if use_cache and resolved_cache_dir is not None:
        identity = SemanticDescentGraphCacheIdentity.from_modules(module_tuple)
        cache = SemanticDescentGraphCache(resolved_cache_dir)
        cache_lookup = cache.load(identity)
        if cache_lookup.graph is not None:
            return cache_lookup.graph
        graph = _build_semantic_descent_graph_cached(module_tuple)
        cache.store(identity, graph)
        return graph
    return _build_semantic_descent_graph_cached(module_tuple)


@dataclass(frozen=True)
class SemanticDescentGraphCacheDirAuthority:
    """Resolve the default persistent graph-cache directory for parsed modules."""

    modules: tuple[ParsedModule, ...]

    def cache_dir(self) -> Path | None:
        if not self.modules:
            return None
        common_root = Path(
            os.path.commonpath(
                tuple(str(module.path.resolve().parent) for module in self.modules)
            )
        )
        return default_semantic_descent_cache_dir(common_root)


@lru_cache(maxsize=None)
def _build_semantic_descent_graph_cached(
    modules: tuple[ParsedModule, ...],
) -> SemanticDescentGraph:
    class_index = build_class_family_index(list(modules))
    authority_builder = SemanticAuthorityBuilder(tuple(modules), class_index)
    authorities, facts = authority_builder.build()
    projections = SemanticProjectionCollector(tuple(modules), class_index).collect()
    mirror_edges = SemanticMirrorResolver(
        authorities,
        facts,
        projections,
        class_index,
    ).edges()
    certificates = tuple(
        DescentCertificate(
            status=DescentStatus.MIRRORED_WITHOUT_DESCENT,
            edge=edge,
            missing_derivation_path=(
                "projection enumerates nominal facts directly instead of deriving "
                "from the authority registry, class family, enum, or schema owner"
            ),
        )
        for edge in mirror_edges
    )
    return SemanticDescentGraph(
        authorities=authorities,
        facts=facts,
        projections=projections,
        mirror_edges=mirror_edges,
        certificates=certificates,
    )


@dataclass(frozen=True)
class SemanticAuthorityBuilder:
    """Build nominal authority and fact records from repo-level class data."""

    modules: tuple[ParsedModule, ...]
    class_index: ClassFamilyIndex

    def build(self) -> tuple[tuple[SemanticAuthority, ...], tuple[SemanticFact, ...]]:
        authorities: list[SemanticAuthority] = []
        facts: list[SemanticFact] = []
        for indexed_class in sorted_tuple(
            self.class_index.classes_by_symbol.values(),
            key=lambda item: item.symbol,
        ):
            provider_result = SemanticAuthorityProvider.result_for_class(
                indexed_class,
                SemanticAuthorityBuildContext(self.class_index),
            )
            if provider_result is None:
                continue
            authorities.append(
                SemanticAuthority(
                    authority_id=indexed_class.symbol,
                    kind=provider_result.kind,
                    name=indexed_class.simple_name,
                    location=SourceLocation(
                        indexed_class.file_path,
                        indexed_class.line,
                        indexed_class.qualname,
                    ),
                    fact_ids=tuple(fact.fact_id for fact in provider_result.facts),
                )
            )
            facts.extend(provider_result.facts)
        return (
            sorted_tuple(authorities, key=lambda item: item.authority_id),
            sorted_tuple(facts, key=lambda item: item.fact_id),
        )


@dataclass(frozen=True)
class SemanticAuthorityProviderResult:
    """Authority kind and facts selected by one authority provider."""

    kind: SemanticAuthorityKind
    facts: tuple[SemanticFact, ...]


@dataclass(frozen=True)
class SemanticAuthorityBuildContext:
    """Shared construction authority for semantic authority providers."""

    class_index: ClassFamilyIndex

    def class_member_fact(
        self,
        authority_symbol: str,
        descendant_symbol: str,
    ) -> SemanticFact:
        descendant = self.class_index.classes_by_symbol[descendant_symbol]
        aliases = (
            descendant.simple_name,
            *self._string_class_assignments(descendant.node),
        )
        return SemanticFact(
            fact_id=f"{authority_symbol}:{descendant.symbol}",
            authority_id=authority_symbol,
            kind=SemanticFactKind.CLASS_MEMBER,
            name=descendant.simple_name,
            aliases=sorted_tuple(aliases),
            location=SourceLocation(
                descendant.file_path,
                descendant.line,
                descendant.qualname,
            ),
        )

    @staticmethod
    def _string_class_assignments(node: ast.ClassDef) -> tuple[str, ...]:
        values: list[str] = []
        for statement in node.body:
            assignment = SingleAssignmentAndValueNameProjection(statement).pair
            if assignment is None:
                continue
            name, value = assignment
            if name.startswith("__"):
                continue
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and PresentationTokenProjection.looks_like_semantic_literal(value.value)
            ):
                values.append(value.value)
        return sorted_tuple(values)


class SemanticAuthorityProvider(ABC, metaclass=AutoRegisterMeta):
    """Registered authority-kind provider for indexed class declarations."""

    __registry__: ClassVar[dict[str, type["SemanticAuthorityProvider"]]] = {}
    __registry_key__ = "provider_id"
    __key_extractor__ = staticmethod(class_name_registry_key)
    __skip_if_no_key__ = True
    provider_order: ClassVar[int] = 100

    @classmethod
    def ordered_providers(cls) -> tuple["SemanticAuthorityProvider", ...]:
        return tuple(
            provider_type()
            for provider_type in sorted(
                cls.__registry__.values(),
                key=lambda item: (item.provider_order, item.__name__),
            )
        )

    @classmethod
    def result_for_class(
        cls,
        indexed_class: IndexedClass,
        context: SemanticAuthorityBuildContext,
    ) -> SemanticAuthorityProviderResult | None:
        for provider in cls.ordered_providers():
            result = provider.provide(indexed_class, context)
            if result is not None:
                return result
        return None

    @abstractmethod
    def provide(
        self,
        indexed_class: IndexedClass,
        context: SemanticAuthorityBuildContext,
    ) -> SemanticAuthorityProviderResult | None:
        raise NotImplementedError


class EnumSemanticAuthorityProvider(SemanticAuthorityProvider):
    """Provide enum member authorities from enum subclasses."""

    provider_order = 10

    def provide(
        self,
        indexed_class: IndexedClass,
        context: SemanticAuthorityBuildContext,
    ) -> SemanticAuthorityProviderResult | None:
        del context
        if not self._is_enum(indexed_class):
            return None
        facts = self._enum_facts(indexed_class)
        if not facts:
            return None
        return SemanticAuthorityProviderResult(SemanticAuthorityKind.ENUM, facts)

    @staticmethod
    def _is_enum(indexed_class: IndexedClass) -> bool:
        return any(
            base_name.rsplit(".", 1)[-1] in _ENUM_BASE_NAMES
            for base_name in indexed_class.declared_base_names
        )

    @staticmethod
    def _enum_facts(indexed_class: IndexedClass) -> tuple[SemanticFact, ...]:
        facts: list[SemanticFact] = []
        for statement in indexed_class.node.body:
            assignment = SingleAssignmentAndValueNameProjection(statement).pair
            if assignment is None:
                continue
            name, value = assignment
            if not isinstance(value, ast.Constant):
                continue
            aliases = (name,)
            if isinstance(value.value, str):
                aliases = (name, value.value)
            facts.append(
                SemanticFact(
                    fact_id=f"{indexed_class.symbol}:{name}",
                    authority_id=indexed_class.symbol,
                    kind=SemanticFactKind.ENUM_MEMBER,
                    name=name,
                    aliases=sorted_tuple(aliases),
                    location=SourceLocation(
                        indexed_class.file_path,
                        statement.lineno,
                        f"{indexed_class.qualname}.{name}",
                    ),
                )
            )
        return tuple(facts) if len(facts) >= 2 else ()


class DataclassSemanticAuthorityProvider(SemanticAuthorityProvider):
    """Provide dataclass field-schema authorities."""

    provider_order = 20

    def provide(
        self,
        indexed_class: IndexedClass,
        context: SemanticAuthorityBuildContext,
    ) -> SemanticAuthorityProviderResult | None:
        del context
        if not self._is_dataclass(indexed_class.node):
            return None
        facts = self._dataclass_facts(indexed_class)
        if not facts:
            return None
        return SemanticAuthorityProviderResult(
            SemanticAuthorityKind.DATACLASS_SCHEMA,
            facts,
        )

    @staticmethod
    def _is_dataclass(node: ast.ClassDef) -> bool:
        return any(
            AttributeChainAuthority.decorator_terminal_name(decorator) == "dataclass"
            for decorator in node.decorator_list
        )

    @staticmethod
    def _dataclass_facts(indexed_class: IndexedClass) -> tuple[SemanticFact, ...]:
        facts: list[SemanticFact] = []
        for statement in indexed_class.node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            name = statement.target.id
            facts.append(
                SemanticFact(
                    fact_id=f"{indexed_class.symbol}:{name}",
                    authority_id=indexed_class.symbol,
                    kind=SemanticFactKind.DATACLASS_FIELD,
                    name=name,
                    aliases=(name,),
                    location=SourceLocation(
                        indexed_class.file_path,
                        statement.lineno,
                        f"{indexed_class.qualname}.{name}",
                    ),
                )
            )
        return tuple(facts) if len(facts) >= 2 else ()


class ClassFamilySemanticAuthorityProvider(SemanticAuthorityProvider):
    """Provide conventional and AutoRegister class-family authorities."""

    provider_order = 30

    def provide(
        self,
        indexed_class: IndexedClass,
        context: SemanticAuthorityBuildContext,
    ) -> SemanticAuthorityProviderResult | None:
        descendants = context.class_index.descendant_symbols(indexed_class.symbol)
        if len(descendants) < 2:
            return None
        facts = tuple(
            context.class_member_fact(indexed_class.symbol, descendant_symbol)
            for descendant_symbol in descendants
            if context.class_index.class_for(descendant_symbol) is not None
        )
        if not facts:
            return None
        return SemanticAuthorityProviderResult(
            self._authority_kind(indexed_class),
            facts,
        )

    @staticmethod
    def _authority_kind(indexed_class: IndexedClass) -> SemanticAuthorityKind:
        if AutoRegisterClassAuthority(indexed_class.node).semantic_authority_shape:
            return SemanticAuthorityKind.AUTOREGISTER_FAMILY
        return SemanticAuthorityKind.CLASS_FAMILY


@dataclass(frozen=True)
class SemanticProjectionCollector:
    """Collect presentation-level projections from parsed modules."""

    modules: tuple[ParsedModule, ...]
    class_index: ClassFamilyIndex

    def collect(self) -> tuple[PresentationProjection, ...]:
        projections: list[PresentationProjection] = []
        for parsed_module in self.modules:
            visitor = _ProjectionVisitor(parsed_module, self.class_index)
            visitor.visit(parsed_module.module)
            projections.extend(visitor.projections)
        return sorted_tuple(
            projections,
            key=lambda item: (item.location.file_path, item.location.line, item.label),
        )


class _ProjectionVisitor(ClassFunctionStackNodeVisitor):
    def __init__(
        self,
        parsed_module: ParsedModule,
        class_index: ClassFamilyIndex,
    ) -> None:
        super().__init__()
        self.parsed_module = parsed_module
        self.class_reference_resolver = ModuleClassReferenceResolver(
            parsed_module,
            class_index,
        )
        self.projections: list[PresentationProjection] = []
        self.owner_construction_stack: list[
            tuple[PresentationAuthorityConstruction, ...]
        ] = []

    @property
    def current_owner_constructions(
        self,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        if not self.owner_construction_stack:
            return ()
        return self.owner_construction_stack[-1]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.owner_construction_stack.append(
            PresentationAuthorityConstructionCollector.constructions_for_function(node)
        )
        try:
            super().visit_FunctionDef(node)
        finally:
            self.owner_construction_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._collect_assignment_projection(node, node.value):
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._collect_assignment_projection(
            node, node.value
        ):
            return
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        tokens = tuple(
            PresentationTokenProjection.tokens_for_node(
                node.test,
                PresentationTokenRole.CONDITION,
            )
        )
        if len({token.value for token in tokens}) >= 2:
            self._append_projection(
                node,
                PresentationProjectionKind.BRANCH_LITERAL,
                f"if@{node.lineno}",
                tokens,
            )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        tokens: list[PresentationToken] = []
        for case in node.cases:
            tokens.extend(
                PresentationTokenProjection.tokens_for_node(
                    case.pattern,
                    PresentationTokenRole.MATCH_CASE,
                )
            )
        if len({token.value for token in tokens}) >= 2:
            self._append_projection(
                node,
                PresentationProjectionKind.MATCH_LITERAL,
                f"match@{node.lineno}",
                tuple(tokens),
            )
        self.generic_visit(node)

    def _collect_assignment_projection(self, node: ast.stmt, value: ast.AST) -> bool:
        label = _assignment_label(node)
        if ProjectionSuppressionPolicy(label).suppresses_semantic_projection():
            return False
        if SingleAssignmentAndValueNameProjection(node).pair is None:
            return False
        if isinstance(value, ast.Dict):
            projection_kind = PresentationProjectionKind.MAPPING_LITERAL
            key_value_pairs = self._projection_key_value_pairs(value)
        elif isinstance(value, ast.List | ast.Tuple | ast.Set):
            projection_kind = PresentationProjectionKind.COLLECTION_LITERAL
            key_value_pairs = ()
        elif isinstance(value, ast.Call):
            if self.current_function_name is not None:
                return False
            projection_kind = PresentationProjectionKind.CALL_LITERAL
            key_value_pairs = ()
        else:
            return False
        projection_constructions = self._projection_constructions(value)
        class_symbols = self.class_reference_resolver.symbols_for_node(value)
        tokens = tuple(
            PresentationTokenProjection.tokens_for_node(
                value,
                PresentationTokenRole.COLLECTION_ITEM,
            )
        )
        if (
            self.current_function_name is not None
            and projection_kind is PresentationProjectionKind.COLLECTION_LITERAL
            and not any(
                token.kind is PresentationTokenKind.STRING_LITERAL for token in tokens
            )
        ):
            return False
        if len({token.value for token in tokens}) < 2:
            return False
        self._append_projection(
            node,
            projection_kind,
            label,
            tokens,
            projection_constructions,
            key_value_pairs,
            class_symbols,
        )
        return True

    @staticmethod
    def _projection_constructions(
        value: ast.AST,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        return PresentationAuthorityConstructionCollector.constructions_for_node(value)

    def _projection_key_value_pairs(
        self,
        value: ast.Dict,
    ) -> tuple[PresentationKeyValuePair, ...]:
        pairs = tuple(
            PresentationKeyValuePair.from_nodes(
                key=key,
                value=item_value,
                class_reference_resolver=self.class_reference_resolver,
            )
            for key, item_value in zip(value.keys, value.values, strict=True)
            if key is not None
        )
        return sorted_tuple(
            pairs,
            key=lambda item: (item.key_source, item.value_source),
        )

    def _append_projection(
        self,
        node: ast.stmt,
        kind: PresentationProjectionKind,
        label: str,
        tokens: tuple[PresentationToken, ...],
        projection_constructions: tuple[PresentationAuthorityConstruction, ...] = (),
        key_value_pairs: tuple[PresentationKeyValuePair, ...] = (),
        class_symbols: tuple[str, ...] = (),
    ) -> None:
        line = node.lineno
        projection_id = (
            f"{self.parsed_module.path}:{line}:{self.qualname}:{kind.value}:{label}"
        )
        self.projections.append(
            PresentationProjection(
                projection_id=projection_id,
                kind=kind,
                label=label,
                owner_symbol=self.qualname,
                location=SourceLocation(str(self.parsed_module.path), line, label),
                tokens=sorted_tuple(
                    frozenset(tokens),
                    key=lambda item: (
                        item.value,
                        item.kind,
                        item.role,
                        item.qualifier or "",
                    ),
                ),
                source_text="",
                owner_constructions=sorted_tuple(
                    frozenset(
                        (*self.current_owner_constructions, *projection_constructions)
                    ),
                    key=lambda item: (item.type_name, item.field_tokens),
                ),
                key_value_pairs=key_value_pairs,
                class_symbols=class_symbols,
            )
        )


@dataclass(frozen=True)
class SemanticMirrorResolver(SemanticDescentGraphSpace):
    """Resolve graph edges where a projection mirrors an authority."""

    class_index: ClassFamilyIndex

    @cached_property
    def dataclass_authorities(self) -> tuple[SemanticAuthority, ...]:
        return tuple(
            authority
            for authority in self.authorities
            if self._policy_for_authority(authority).dataclass_authority_selected
        )

    @cached_property
    def dataclass_authority_ids(self) -> frozenset[str]:
        return frozenset(
            authority.authority_id for authority in self.dataclass_authorities
        )

    @cached_property
    def ancestor_symbol_sets(self) -> dict[str, frozenset[str]]:
        return {
            symbol: frozenset(self.class_index.ancestor_symbols(symbol))
            for symbol in self.class_index.classes_by_symbol
        }

    @cached_property
    def projection_owner_class_symbols(self) -> dict[str, str | None]:
        return {
            projection.projection_id: self._resolve_projection_owner_class_symbol(
                projection
            )
            for projection in self.projections
        }

    @cached_property
    def dataclass_projection_descent_authority_ids(self) -> dict[str, frozenset[str]]:
        return {
            projection.projection_id: self._dataclass_projection_descent_authority_ids(
                projection
            )
            for projection in self.projections
        }

    @cached_property
    def construction_authority_class_cache(
        self,
    ) -> dict[ConstructionAuthorityCacheKey, bool]:
        return {}

    @cached_property
    def construction_materializes_authority_cache(
        self,
    ) -> dict[ConstructionAuthorityCacheKey, bool]:
        return {}

    @cached_property
    def projection_materializes_any_dataclass_authority_cache(self) -> dict[str, bool]:
        return {}

    def edges(self) -> tuple[MirrorEdge, ...]:
        edges: list[MirrorEdge] = []
        for projection in self.projections:
            for authority_id, matches in self._matches_by_authority(
                projection,
            ).items():
                edge = self._edge_for(
                    projection,
                    self.authority_by_id[authority_id],
                    self.fact_authority_index.facts_for_authority(authority_id),
                    matches,
                )
                if edge is not None:
                    edges.append(edge)
        return sorted_tuple(
            edges,
            key=lambda item: (
                -len(item.matched_fact_ids),
                item.authority_id,
                item.projection_id,
            ),
        )

    def _matches_by_authority(
        self,
        projection: PresentationProjection,
    ) -> FactMatchesByAuthority:
        matches = [
            SemanticFactTokenMatch(
                authority_id=ref.authority_id,
                fact_id=ref.fact_id,
                token_value=token.value,
            )
            for token in projection.tokens
            for ref in self._candidate_refs_for_token(token)
        ]
        matches.extend(
            ProjectionClassSymbolFactMatcher(
                projection,
                self.class_index,
                self.authority_by_id,
                self.fact_authority_index,
            ).matches()
        )
        return SemanticFactMatchIndex(matches).by_authority

    def _candidate_refs_for_token(
        self,
        token: PresentationToken,
    ) -> tuple[_FactTokenReference, ...]:
        if not self.fact_token_index.contains_token(token.value):
            return ()
        refs = self.fact_token_index.refs_for_token(token.value)
        if token.kind is not PresentationTokenKind.QUALIFIED_ATTRIBUTE:
            return refs
        qualifier = token.qualifier
        if qualifier is not None and self.authority_name_index.contains_name(qualifier):
            allowed_authority_ids = frozenset(
                authority_id
                for authority_id in self.authority_name_index.authority_ids_for_name(
                    qualifier
                )
                if self._policy_for_authority_id(
                    authority_id
                ).authority_qualified_token_reference_admitted
            )
            return tuple(
                ref for ref in refs if ref.authority_id in allowed_authority_ids
            )
        return tuple(
            ref
            for ref in refs
            if self._policy_for_authority_id(
                ref.authority_id
            ).foreign_qualified_attribute_token_reference_admitted
        )

    def _edge_for(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        facts: tuple[SemanticFact, ...],
        matches_by_fact_id: dict[str, set[str]],
    ) -> MirrorEdge | None:
        if len(facts) < 2:
            return None
        matched_fact_ids = [
            fact.fact_id for fact in facts if fact.fact_id in matches_by_fact_id
        ]
        if len(matched_fact_ids) < 2:
            return None
        matched_tokens: set[str] = set()
        for fact_id in matched_fact_ids:
            matched_tokens.update(matches_by_fact_id[fact_id])
        coverage_ratio = len(matched_fact_ids) / len(facts)
        if coverage_ratio < 0.5 and len(matched_fact_ids) < 3:
            return None
        candidate = SemanticMirrorEdgeCandidate(
            projection=projection,
            authority=authority,
            facts=facts,
            matched_fact_ids=tuple(matched_fact_ids),
            matched_tokens=sorted_tuple(matched_tokens),
            coverage_ratio=coverage_ratio,
        )
        policy = self._policy_for_authority(authority)
        if not policy.edge_is_admissible(self, candidate):
            return None
        if policy.projection_descends_to_authority(self, candidate):
            return None
        return MirrorEdge(
            authority_id=authority.authority_id,
            projection_id=projection.projection_id,
            matched_fact_ids=candidate.matched_fact_ids,
            matched_tokens=candidate.matched_tokens,
            coverage_ratio=coverage_ratio,
        )

    def _policy_for_authority_id(
        self,
        authority_id: str,
    ) -> SemanticAuthorityMirrorPolicy:
        return self._policy_for_authority(self.authority_by_id[authority_id])

    @staticmethod
    def _policy_for_authority(
        authority: SemanticAuthority,
    ) -> SemanticAuthorityMirrorPolicy:
        return SemanticAuthorityMirrorPolicy.for_authority(authority)

    def dataclass_projection_descends_to_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return (
            authority.authority_id
            in self.dataclass_projection_descent_authority_ids[projection.projection_id]
        )

    def _dataclass_projection_descent_authority_ids(
        self,
        projection: PresentationProjection,
    ) -> frozenset[str]:
        owner_class_symbol = self._projection_owner_class_symbol(projection)
        if owner_class_symbol is None:
            return frozenset()
        owner_ancestor_symbols = self.ancestor_symbol_sets[owner_class_symbol]
        return frozenset(
            authority_id
            for authority_id in self.dataclass_authority_ids
            if owner_class_symbol == authority_id
            or authority_id in owner_ancestor_symbols
        )

    def projection_owner_constructs_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
    ) -> bool:
        return self._projection_owner_derives_dataclass_authority(
            projection,
            authority,
            matched_facts,
            self._construction_type_descends_to_authority,
        )

    def _construction_type_descends_to_authority(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        if self._construction_type_is_authority_class(construction, authority):
            return True
        return self._construction_type_materializes_authority(construction, authority)

    def _construction_type_is_authority_class(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        return self._construction_authority_cache_result(
            self.construction_authority_class_cache,
            construction,
            authority,
            self._construction_type_is_authority_class_uncached,
        )

    def _construction_type_is_authority_class_uncached(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        if construction.type_name == authority.name:
            return True
        for class_symbol in self.class_index.symbols_by_simple_name.get(
            construction.type_name, ()
        ):
            if authority.authority_id in self.class_index.ancestor_symbols(
                class_symbol
            ):
                return True
        return False

    def _construction_type_materializes_authority(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        return self._construction_authority_cache_result(
            self.construction_materializes_authority_cache,
            construction,
            authority,
            self._construction_type_materializes_authority_uncached,
        )

    def _construction_authority_cache_result(
        self,
        cache: dict[ConstructionAuthorityCacheKey, bool],
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
        compute: ConstructionAuthorityPredicate,
    ) -> bool:
        cache_key = (construction.type_name, authority.authority_id)
        if cache_key not in cache:
            cache[cache_key] = compute(
                construction,
                authority,
            )
        return cache[cache_key]

    def _construction_type_materializes_authority_uncached(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            (indexed_class := self.class_index.class_for(class_symbol)) is not None
            and self._class_materializes_authority(indexed_class, authority)
            for class_symbol in self.class_index.symbols_by_simple_name.get(
                construction.type_name, ()
            )
        )

    def _class_materializes_authority(
        self,
        indexed_class: IndexedClass,
        authority: SemanticAuthority,
    ) -> bool:
        return self._class_declares_materialized_authority(
            indexed_class,
            authority,
        ) or any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and self._function_materializes_authority(statement, authority)
            for statement in indexed_class.node.body
        )

    @staticmethod
    def _class_declares_materialized_authority(
        indexed_class: IndexedClass,
        authority: SemanticAuthority,
    ) -> bool:
        for statement in indexed_class.node.body:
            assignment = SingleAssignmentAndValueNameProjection(statement).pair
            if assignment is None:
                continue
            _, value = assignment
            if AttributeChainAuthority.terminal_name(value) == authority.name:
                return True
        return False

    def _function_materializes_authority(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            self._call_constructs_authority(child, authority)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        )

    def _call_constructs_authority(
        self,
        node: ast.Call,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            self._construction_type_is_authority_class(
                PresentationAuthorityConstruction(type_name, ()),
                authority,
            )
            for type_name in PresentationAuthorityConstructionCollector.construction_type_names(
                node
            )
        )

    def projection_descends_to_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        return any(
            self.dataclass_projection_descends_to_authority(projection, authority)
            for authority in self.dataclass_authorities
        )

    def projection_materializes_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        cache = self.projection_materializes_any_dataclass_authority_cache
        if projection.projection_id not in cache:
            cache[
                projection.projection_id
            ] = self._projection_materializes_any_dataclass_authority_uncached(
                projection,
            )
        return cache[projection.projection_id]

    def _projection_materializes_any_dataclass_authority_uncached(
        self,
        projection: PresentationProjection,
    ) -> bool:
        return any(
            self.projection_owner_materializes_dataclass_authority(
                projection,
                authority,
                self.fact_authority_index.facts_for_authority(authority.authority_id),
            )
            for authority in self.dataclass_authorities
        )

    def projection_owner_materializes_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
    ) -> bool:
        return self._projection_owner_derives_dataclass_authority(
            projection,
            authority,
            matched_facts,
            self._construction_type_materializes_authority,
        )

    def _projection_owner_derives_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
        accepts_construction: ConstructionAuthorityPredicate,
    ) -> bool:
        if not matched_facts:
            return False
        matched_tokens = frozenset(
            variant
            for fact in matched_facts
            for variant in normalized_name_variants(fact.name)
        )
        if not matched_tokens:
            return False
        descended_field_tokens: set[str] = set()
        for construction in projection.owner_constructions:
            if not accepts_construction(construction, authority):
                continue
            descended_field_tokens.update(construction.field_tokens)
            if matched_tokens <= frozenset(construction.field_tokens):
                return True
        return matched_tokens <= frozenset(descended_field_tokens)

    def projection_shares_dataclass_base_with_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        owner_class_symbol = self._projection_owner_class_symbol(projection)
        if owner_class_symbol is None:
            return False
        shared_ancestors = (
            self.ancestor_symbol_sets[owner_class_symbol]
            & self.ancestor_symbol_sets[authority.authority_id]
            & self.dataclass_authority_ids
        )
        return bool(shared_ancestors)

    def _projection_owner_class_symbol(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        return self.projection_owner_class_symbols[projection.projection_id]

    def _resolve_projection_owner_class_symbol(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        if projection.owner_symbol == "<module>":
            return None
        parts = projection.owner_symbol.split(".")
        for end_index in range(len(parts), 0, -1):
            owner_qualname = ".".join(parts[:end_index])
            symbol = self.class_index.symbol_for(
                file_path=projection.location.file_path,
                qualname=owner_qualname,
            )
            if symbol is not None:
                return symbol
        return None

    @staticmethod
    def projection_has_authority_affinity(
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        authority_tokens = NormalizeNameProjection.token_set(authority.name)
        projection_tokens = NormalizeNameProjection.token_set(
            f"{projection.label} {projection.owner_symbol} {projection.location.symbol}"
        )
        return len(authority_tokens & projection_tokens) >= 2

    @staticmethod
    def projection_has_enum_authority_affinity(
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return SemanticAuthorityAffinityPolicy(
            authority_name=authority.name,
            projection_label=projection.label,
            projection_owner_symbol=projection.owner_symbol,
            projection_location_symbol=projection.location.symbol,
        ).has_authority_affinity()

    @staticmethod
    def projection_has_qualified_authority_reference(
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            token.kind is PresentationTokenKind.QUALIFIED_ATTRIBUTE
            and token.qualifier == authority.name
            for token in projection.tokens
        )

    @staticmethod
    def enum_branch_projection_has_case_syntax(
        projection: PresentationProjection,
        matched_tokens: frozenset[str],
    ) -> bool:
        return (
            len(
                {
                    token.value
                    for token in projection.tokens
                    if token.kind is PresentationTokenKind.STRING_LITERAL
                    and token.value in matched_tokens
                }
            )
            >= 2
        )

    @staticmethod
    def dataclass_branch_projection_has_field_syntax(
        projection: PresentationProjection,
        matched_tokens: frozenset[str],
    ) -> bool:
        return any(
            token.kind is PresentationTokenKind.STRING_LITERAL
            and token.value in matched_tokens
            for token in projection.tokens
        )


class NormalizeNameProjection:
    """Normalize source names and literal keys into semantic comparison tokens."""

    @classmethod
    def variants(cls, raw_name: str) -> tuple[str, ...]:
        normalized = cls.normalize(raw_name)
        variants = {normalized} if normalized else set()
        for suffix in _CLASS_SUFFIXES:
            if raw_name.endswith(suffix) and len(raw_name) > len(suffix):
                suffix_trimmed = cls.normalize(raw_name[: -len(suffix)])
                if suffix_trimmed:
                    variants.add(suffix_trimmed)
        return sorted_tuple(variants)

    @classmethod
    def token_set(cls, raw_name: str) -> frozenset[str]:
        return frozenset(
            token
            for variant in cls.variants(raw_name)
            for token in variant.split("_")
            if token
        )

    @staticmethod
    def normalize(raw_name: str) -> str:
        if not raw_name:
            return ""
        parts: list[str] = []
        for segment in re.split(r"[_\-.:]+", raw_name):
            if segment:
                parts.extend(_NAME_TOKEN_PATTERN.findall(segment))
        return "_".join(part.lower() for part in parts if part)


def normalized_name_variants(raw_name: str) -> tuple[str, ...]:
    """Return conservative normalized variants for names and semantic keys."""

    return NormalizeNameProjection.variants(raw_name)


def _class_reference_normalized_tokens(
    class_index: ClassFamilyIndex,
    class_symbols: tuple[str, ...],
) -> frozenset[str]:
    tokens: set[str] = set()
    for class_symbol in class_symbols:
        indexed_class = class_index.class_for(class_symbol)
        if indexed_class is None:
            continue
        tokens.update(normalized_name_variants(indexed_class.simple_name))
    return frozenset(tokens)


class PresentationTokenNodeProjector(ABC, metaclass=AutoRegisterMeta):
    """Registered projection rule for one AST node family."""

    __registry_key__ = "projector_id"
    __key_extractor__ = staticmethod(class_name_registry_key)
    __skip_if_no_key__ = True

    node_type: ClassVar[type[ast.AST] | tuple[type[ast.AST], ...]]

    @classmethod
    def registered_projector_types(
        cls,
    ) -> tuple[type["PresentationTokenNodeProjector"], ...]:
        return sorted_tuple(
            cls.__registry__.values(),
            key=lambda projector_type: projector_type.__name__,
        )

    @classmethod
    def projector_for_node(
        cls,
        node: ast.AST,
    ) -> type["PresentationTokenNodeProjector"] | None:
        for projector_type in cls.registered_projector_types():
            if isinstance(node, projector_type.node_type):
                return projector_type
        return None

    @classmethod
    @abstractmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        """Project one supported AST node into normalized semantic tokens."""


class PresentationTokenProjection:
    """Project AST syntax into normalized presentation tokens."""

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        projector_type = PresentationTokenNodeProjector.projector_for_node(node)
        if projector_type is None:
            return ()
        return projector_type.tokens_for_node(node, role)

    @staticmethod
    def looks_like_semantic_literal(value: str) -> bool:
        stripped = value.strip()
        if stripped != value:
            return False
        return bool(_SEMANTIC_STRING_LITERAL_PATTERN.fullmatch(value))


class IterChildPresentationTokenProjectorMixin(ABC):
    """Project all child AST nodes with the inherited token role."""

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        if not isinstance(node, cls.node_type):
            raise TypeError(f"Expected {cls.node_type!r}, got {type(node)!r}")
        tokens: list[PresentationToken] = []
        for child in ast.iter_child_nodes(node):
            tokens.extend(PresentationTokenProjection.tokens_for_node(child, role))
        return tuple(tokens)


class ConstantPresentationTokenProjector(PresentationTokenNodeProjector):
    """Project semantic string constants."""

    node_type = ast.Constant

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.Constant):
            raise TypeError(f"Expected ast.Constant, got {type(node)!r}")
        if not isinstance(node.value, str):
            return ()
        if not PresentationTokenProjection.looks_like_semantic_literal(node.value):
            return ()
        return tuple(
            PresentationToken(value, PresentationTokenKind.STRING_LITERAL, role)
            for value in normalized_name_variants(node.value)
        )


class NamePresentationTokenProjector(PresentationTokenNodeProjector):
    """Project identifier references."""

    node_type = ast.Name

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.Name):
            raise TypeError(f"Expected ast.Name, got {type(node)!r}")
        return tuple(
            PresentationToken(value, PresentationTokenKind.NAME_REFERENCE, role)
            for value in normalized_name_variants(node.id)
        )


class AttributePresentationTokenProjector(PresentationTokenNodeProjector):
    """Project attribute terminals and their immediate qualifier."""

    node_type = ast.Attribute

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.Attribute):
            raise TypeError(f"Expected ast.Attribute, got {type(node)!r}")
        chain = AttributeChainAuthority.chain(node)
        if len(chain) >= 2:
            qualifier = chain[-2]
            return tuple(
                PresentationToken(
                    value,
                    PresentationTokenKind.QUALIFIED_ATTRIBUTE,
                    role,
                    qualifier=qualifier,
                )
                for value in normalized_name_variants(chain[-1])
            )
        return tuple(
            PresentationToken(value, PresentationTokenKind.NAME_REFERENCE, role)
            for value in normalized_name_variants(node.attr)
        )


class DictPresentationTokenProjector(PresentationTokenNodeProjector):
    """Project dictionary keys and values into distinct token roles."""

    node_type = (ast.Dict,)

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls, role
        if not isinstance(node, ast.Dict):
            raise TypeError(f"Expected ast.Dict, got {type(node)!r}")
        tokens: list[PresentationToken] = []
        for key in node.keys:
            if key is not None:
                tokens.extend(
                    PresentationTokenProjection.tokens_for_node(
                        key, PresentationTokenRole.DICT_KEY
                    )
                )
        for value in node.values:
            tokens.extend(
                PresentationTokenProjection.tokens_for_node(
                    value, PresentationTokenRole.DICT_VALUE
                )
            )
        return tuple(tokens)


class SequencePresentationTokenProjector(PresentationTokenNodeProjector):
    """Project list, tuple, and set members."""

    node_type = (ast.List, ast.Tuple, ast.Set)

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.List | ast.Tuple | ast.Set):
            raise TypeError(f"Expected sequence literal, got {type(node)!r}")
        tokens: list[PresentationToken] = []
        for item in node.elts:
            tokens.extend(PresentationTokenProjection.tokens_for_node(item, role))
        return tuple(tokens)


class CallPresentationTokenProjector(PresentationTokenNodeProjector):
    """Project call targets, positional arguments, and keyword arguments."""

    node_type = ast.Call

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls, role
        if not isinstance(node, ast.Call):
            raise TypeError(f"Expected ast.Call, got {type(node)!r}")
        tokens: list[PresentationToken] = []
        tokens.extend(
            PresentationTokenProjection.tokens_for_node(
                node.func, PresentationTokenRole.CALL_TARGET
            )
        )
        for arg in node.args:
            tokens.extend(
                PresentationTokenProjection.tokens_for_node(
                    arg, PresentationTokenRole.CALL_ARGUMENT
                )
            )
        for keyword in node.keywords:
            if keyword.arg is not None:
                tokens.extend(
                    PresentationToken(
                        value,
                        PresentationTokenKind.STRING_LITERAL,
                        PresentationTokenRole.CALL_KEYWORD,
                    )
                    for value in normalized_name_variants(keyword.arg)
                )
            tokens.extend(
                PresentationTokenProjection.tokens_for_node(
                    keyword.value, PresentationTokenRole.CALL_KEYWORD
                )
            )
        return tuple(tokens)


class ComparePresentationTokenProjector(
    IterChildPresentationTokenProjectorMixin,
    PresentationTokenNodeProjector,
):
    """Project comparison operands."""

    node_type = ast.Compare


class BoolOpPresentationTokenProjector(
    IterChildPresentationTokenProjectorMixin,
    PresentationTokenNodeProjector,
):
    """Project boolean operands."""

    node_type = ast.BoolOp


class MatchValuePresentationTokenProjector(PresentationTokenNodeProjector):
    """Project value-pattern payloads."""

    node_type = ast.MatchValue

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.MatchValue):
            raise TypeError(f"Expected ast.MatchValue, got {type(node)!r}")
        return PresentationTokenProjection.tokens_for_node(node.value, role)


class MatchSequencePresentationTokenProjector(
    IterChildPresentationTokenProjectorMixin,
    PresentationTokenNodeProjector,
):
    """Project sequence-pattern payloads."""

    node_type = ast.MatchSequence


class MatchOrPresentationTokenProjector(
    IterChildPresentationTokenProjectorMixin,
    PresentationTokenNodeProjector,
):
    """Project alternative-pattern payloads."""

    node_type = ast.MatchOr


class PresentationAuthorityConstructionCollector:
    """Collect nominal authority construction evidence inside one owner function."""

    @classmethod
    def constructions_for_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        return cls.constructions_for_node(node)

    @classmethod
    def constructions_for_node(
        cls,
        node: ast.AST,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        constructions: list[PresentationAuthorityConstruction] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            constructions.extend(cls.constructions_for_call(child))
        return sorted_tuple(
            frozenset(constructions),
            key=lambda item: (item.type_name, item.field_tokens),
        )

    @classmethod
    def constructions_for_call(
        cls,
        node: ast.Call,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        field_tokens = cls.constructor_field_tokens(node)
        if len(field_tokens) < 2:
            return ()
        type_names = cls.construction_type_names(node)
        return tuple(
            PresentationAuthorityConstruction(
                type_name=type_name,
                field_tokens=field_tokens,
            )
            for type_name in type_names
        )

    @classmethod
    def construction_type_names(cls, node: ast.Call) -> tuple[str, ...]:
        chain = AttributeChainAuthority.chain(node.func)
        if not chain:
            return ()
        type_names = {chain[-1]}
        if len(chain) >= 2:
            type_names.add(chain[-2])
        return sorted_tuple(type_names)

    @classmethod
    def constructor_field_tokens(cls, node: ast.Call) -> tuple[str, ...]:
        field_names = [
            keyword.arg for keyword in node.keywords if keyword.arg is not None
        ]
        field_names.extend(cls.argument_field_name(argument) for argument in node.args)
        return sorted_tuple(
            variant
            for field_name in field_names
            if field_name is not None
            for variant in normalized_name_variants(field_name)
        )

    @staticmethod
    def argument_field_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and PresentationTokenProjection.looks_like_semantic_literal(node.value)
        ):
            return node.value
        return None


class AttributeChainAuthority:
    """Own AST attribute-chain parsing for semantic projection logic."""

    @classmethod
    def chain(cls, node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            parent = cls.chain(node.value)
            if parent:
                return (*parent, node.attr)
        if isinstance(node, ast.Subscript):
            return cls.chain(node.value)
        return ()

    @classmethod
    def terminal_name(cls, node: ast.AST) -> str | None:
        chain = cls.chain(node)
        if chain:
            return chain[-1]
        return None

    @classmethod
    def decorator_terminal_name(cls, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            return cls.terminal_name(node.func)
        return cls.terminal_name(node)


def _assignment_label(node: ast.stmt) -> str:
    name = SingleAssignmentAndValueNameProjection(node).name
    if name is not None:
        return name
    if isinstance(node, ast.Assign) and node.targets:
        return ast.unparse(node.targets[0])
    return f"assignment@{node.lineno}"


def _semantic_descent_implementation_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(class_index_module.__file__).resolve(),
    )


def _source_file_hash(path: Path) -> str:
    return hashlib.blake2s(
        path.read_bytes(),
        digest_size=semantic_descent_graph_cache_schema.digest_size,
    ).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.blake2s(
        value.encode("utf-8"),
        digest_size=semantic_descent_graph_cache_schema.digest_size,
    ).hexdigest()
