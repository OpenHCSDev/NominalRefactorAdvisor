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
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, lru_cache
from itertools import groupby
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from .assignment_projection import SingleAssignmentAndValueNameProjection
from .ast_tools import (
    ClassFunctionStackNodeVisitor,
    ParsedModule,
    PythonModulePathIdentity,
    PythonSourcePathPolicy,
    python_module_path_identities_for_roots,
)
from . import class_index as class_index_module
from .cache_paths import default_semantic_descent_cache_dir
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleClassReferenceResolver,
    build_class_family_index,
    overlay_class_family_index,
)
from .collection_algebra import sorted_tuple
from .models import (
    FindingMetrics,
    MappingMetrics,
    RefactorFinding,
    SemanticRecord,
    RegistrationMetrics,
    SourceLocation,
)
from .name_algebra import CLASS_NAME_ALGEBRA
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
        "members",
        "derive it by iterating the authority registry or subclass family instead "
        "of maintaining a parallel presentation surface",
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
        "members",
        "derive it by iterating the AutoRegisterMeta registry instead of "
        "maintaining a parallel presentation surface",
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
        "fields",
        "derive it from dataclass fields or move the projection onto the record",
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
        "members",
        "derive it by iterating enum members or move behavior onto the enum cases",
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
        "facts",
        "replace the detector-observed mirror with a graph-certified derivation path",
    )

    def __new__(
        cls,
        value: str,
        is_class_family_like: bool,
        reporting_capability_gap: str,
        uses_registration_metrics: bool,
        reporting_scaffold_template: str,
        reporting_codemod_patch_template: str,
        mirrored_fact_label: str,
        missing_derivation_instruction: str,
    ) -> "SemanticAuthorityKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_class_family_like = is_class_family_like
        member._reporting_capability_gap = reporting_capability_gap
        member._uses_registration_metrics = uses_registration_metrics
        member._reporting_scaffold_template = reporting_scaffold_template
        member._reporting_codemod_patch_template = reporting_codemod_patch_template
        member._mirrored_fact_label = mirrored_fact_label
        member._missing_derivation_instruction = missing_derivation_instruction
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

    @property
    def mirrored_fact_label(self) -> str:
        return self._mirrored_fact_label

    @property
    def missing_derivation_instruction(self) -> str:
        return self._missing_derivation_instruction


class SemanticFactKind(StrEnum):
    """Facts owned by a semantic authority."""

    CLASS_MEMBER = "class_member"
    DATACLASS_FIELD = "dataclass_field"
    ENUM_MEMBER = "enum_member"
    FINDING_EVIDENCE = "finding_evidence"


class PresentationProjectionKind(StrEnum):
    """Raw presentation shapes that may mirror a semantic authority."""

    CALL_LITERAL = ("call_literal", "call projection")
    COLLECTION_LITERAL = ("collection_literal", "collection literal")
    DETECTOR_FINDING = ("detector_finding", "detector finding")
    MAPPING_LITERAL = ("mapping_literal", "mapping literal")
    BRANCH_LITERAL = ("branch_literal", "branch literal")
    MATCH_LITERAL = ("match_literal", "match literal")

    def __new__(cls, value: str, surface_label: str) -> "PresentationProjectionKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._surface_label = surface_label
        return member

    @property
    def is_branch_like(self) -> bool:
        return self in (
            type(self).BRANCH_LITERAL,
            type(self).MATCH_LITERAL,
        )

    @property
    def surface_label(self) -> str:
        return self._surface_label


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

    version: int = 7
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

    @classmethod
    def from_path_identity(
        cls,
        identity: PythonModulePathIdentity,
    ) -> "SemanticDescentModuleSignature":
        return cls(
            path=str(identity.path.resolve()),
            parsed_import_name=identity.import_name,
            is_package_init=identity.is_package_init,
            source_hash=_source_file_hash(identity.path),
        )


@dataclass(frozen=True)
class SemanticDescentModuleFamilySignature:
    """Source-set member identity for latest semantic-descent graph reuse."""

    path: str
    parsed_import_name: str
    is_package_init: bool

    @classmethod
    def from_module_signature(
        cls,
        signature: SemanticDescentModuleSignature,
    ) -> "SemanticDescentModuleFamilySignature":
        return cls(
            path=signature.path,
            parsed_import_name=signature.parsed_import_name,
            is_package_init=signature.is_package_init,
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

    @classmethod
    def from_path_identities(
        cls,
        identities: tuple[PythonModulePathIdentity, ...],
    ) -> "SemanticDescentGraphCacheIdentity":
        return cls(
            schema=semantic_descent_graph_cache_schema,
            implementation=SemanticDescentImplementationSignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            modules=tuple(
                sorted(
                    (
                        SemanticDescentModuleSignature.from_path_identity(identity)
                        for identity in identities
                    ),
                    key=lambda item: item.path,
                )
            ),
        )

    @classmethod
    def from_roots(
        cls,
        roots: tuple[Path, ...],
        *,
        source_policy: PythonSourcePathPolicy | None = None,
    ) -> "SemanticDescentGraphCacheIdentity":
        return cls.from_path_identities(
            python_module_path_identities_for_roots(
                roots,
                source_policy=source_policy,
            )
        )

    @property
    def cache_token(self) -> str:
        return hashlib.blake2s(
            repr(self).encode("utf-8"),
            digest_size=self.schema.digest_size,
        ).hexdigest()


@dataclass(frozen=True)
class SemanticDescentGraphCacheFamilyIdentity:
    """Source-set family identity for latest semantic-descent graph reuse."""

    schema: SemanticDescentGraphCacheSchema
    implementation: SemanticDescentImplementationSignature
    python_version: tuple[int, int]
    modules: tuple[SemanticDescentModuleFamilySignature, ...]

    @classmethod
    def from_identity(
        cls,
        identity: SemanticDescentGraphCacheIdentity,
    ) -> "SemanticDescentGraphCacheFamilyIdentity":
        return cls(
            schema=identity.schema,
            implementation=identity.implementation,
            python_version=identity.python_version,
            modules=tuple(
                SemanticDescentModuleFamilySignature.from_module_signature(module)
                for module in identity.modules
            ),
        )

    @classmethod
    def from_path_identities(
        cls,
        identities: tuple[PythonModulePathIdentity, ...],
    ) -> "SemanticDescentGraphCacheFamilyIdentity":
        return cls.from_identity(
            SemanticDescentGraphCacheIdentity.from_path_identities(identities)
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


@dataclass(frozen=True)
class ProjectionOwnerSymbol:
    """Nominal projection-owner symbol, including module-level ownership."""

    module_owner_value: ClassVar[str] = "<module>"
    value: str

    @property
    def module_level(self) -> bool:
        return self.value == self.module_owner_value

    @property
    def qualname_parts(self) -> tuple[str, ...]:
        if self.module_level:
            return ()
        return tuple(self.value.split("."))


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

    @cached_property
    def owner(self) -> ProjectionOwnerSymbol:
        return ProjectionOwnerSymbol(self.owner_symbol)


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
class SemanticMirrorMatch:
    """Fact/token overlap carried by one semantic mirror edge."""

    fact_refs: tuple[SemanticFactReference, ...]
    tokens: tuple[str, ...]
    coverage_ratio: float

    @classmethod
    def from_facts(cls, facts: tuple[SemanticFact, ...]) -> "SemanticMirrorMatch":
        return cls(
            fact_refs=tuple(
                SemanticFactReference(fact.authority_id, fact.fact_id) for fact in facts
            ),
            tokens=sorted_tuple(
                {
                    variant
                    for fact in facts
                    for variant in normalized_name_variants(fact.name)
                }
            ),
            coverage_ratio=1.0,
        )

    @classmethod
    def from_authority_matches(
        cls,
        facts: tuple[SemanticFact, ...],
        matches_by_fact_id: dict[str, set[str]],
    ) -> "SemanticMirrorMatch | None":
        if len(facts) < 2:
            return None
        fact_refs = tuple(
            SemanticFactReference(fact.authority_id, fact.fact_id)
            for fact in facts
            if fact.fact_id in matches_by_fact_id
        )
        if len(fact_refs) < 2:
            return None
        tokens: set[str] = set()
        for fact_ref in fact_refs:
            tokens.update(matches_by_fact_id[fact_ref.fact_id])
        coverage_ratio = len(fact_refs) / len(facts)
        if coverage_ratio < 0.5 and len(fact_refs) < 3:
            return None
        return cls(
            fact_refs=fact_refs,
            tokens=sorted_tuple(tokens),
            coverage_ratio=coverage_ratio,
        )

    @property
    def fact_count(self) -> int:
        return len(self.fact_refs)


@dataclass(frozen=True)
class MirrorEdge(SemanticAuthorityProjectionReference):
    """Candidate relation between a raw projection and a nominal authority."""

    match: SemanticMirrorMatch


@dataclass(frozen=True)
class SemanticMirrorEdgeCandidate:
    """Resolved projection/fact overlap before policy admissibility filtering."""

    projection: PresentationProjection
    authority: SemanticAuthority
    facts: tuple[SemanticFact, ...]
    match: SemanticMirrorMatch

    @cached_property
    def matched_facts(self) -> tuple[SemanticFact, ...]:
        fact_ids = frozenset(ref.fact_id for ref in self.match.fact_refs)
        return tuple(fact for fact in self.facts if fact.fact_id in fact_ids)

    @cached_property
    def branch_like_projection(self) -> bool:
        return self.projection.kind.is_branch_like

    @cached_property
    def authority_affinity(self) -> SemanticAuthorityAffinityPolicy:
        return SemanticAuthorityAffinityPolicy(
            authority_name=self.authority.name,
            projection_label=self.projection.label,
            projection_owner_symbol=self.projection.owner_symbol,
            projection_location_symbol=self.projection.location.symbol,
        )

    @cached_property
    def missing_derivation_path(self) -> str:
        return (
            f"{self.projection.kind.surface_label} `{self.projection.label}` "
            f"repeats {self.authority.kind.mirrored_fact_label} from "
            f"{self.authority.kind.value} `{self.authority.name}`; "
            f"{self.authority.kind.missing_derivation_instruction}"
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
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        del context, candidate
        return True

    def projection_descends_to_authority(
        self,
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        del context, candidate
        return False


class ClassFamilyLikeMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Shared policy for class-family authorities and AutoRegister families."""

    foreign_qualified_attribute_token_reference_admitted = True

    def edge_is_admissible(
        self,
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        return not (
            candidate.projection.kind is PresentationProjectionKind.BRANCH_LITERAL
            and candidate.match.fact_count <= 2
            and not context.projection_semantics.has_authority_affinity(
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
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        if (
            candidate.match.fact_count <= 2
            and not context.projection_semantics.has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
            and not context.projection_semantics.has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
            and context.fact_specificity.matched_facts_are_reused_roles(
                candidate.matched_facts
            )
        ):
            return False
        if (
            candidate.match.coverage_ratio < 1.0
            and candidate.match.fact_count <= 2
            and not context.projection_semantics.has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        if (
            candidate.branch_like_projection
            and not context.projection_semantics.dataclass_branch_has_field_syntax(
                candidate.projection,
                frozenset(candidate.match.tokens),
            )
            and not context.projection_semantics.has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        if (
            (
                context.dataclass_descent.projection_constructs_unrelated_dataclass_authority(
                    candidate.projection,
                    candidate.authority,
                )
                or context.dataclass_descent.projection_materializes_any_dataclass_authority(
                    candidate.projection,
                )
            )
            and not context.projection_semantics.has_authority_affinity(
                candidate.projection,
                candidate.authority,
            )
            and not context.projection_semantics.has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        return not (
            candidate.match.coverage_ratio < 1.0
            and (
                context.dataclass_descent.projection_descends_to_any_dataclass_authority(
                    candidate.projection,
                )
                or context.dataclass_descent.projection_materializes_any_dataclass_authority(
                    candidate.projection,
                )
                or context.dataclass_descent.projection_constructs_unrelated_dataclass_authority(
                    candidate.projection,
                    candidate.authority,
                )
            )
        )

    def projection_descends_to_authority(
        self,
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        return (
            context.dataclass_descent.projection_descends_to_authority(
                candidate.projection,
                candidate.authority,
            )
            or context.dataclass_descent.projection_owner_constructs_dataclass_authority(
                candidate.projection,
                candidate.authority,
                candidate.matched_facts,
            )
            or context.dataclass_descent.projection_shares_dataclass_base_with_authority(
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
        context: "SemanticMirrorResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        if (
            candidate.branch_like_projection
            and not context.projection_semantics.enum_branch_has_case_syntax(
                candidate.projection,
                frozenset(candidate.match.tokens),
            )
            and not candidate.authority_affinity.has_authority_affinity()
            and not context.projection_semantics.has_qualified_authority_reference(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        return not (
            candidate.match.fact_count <= 2
            and not candidate.authority_affinity.has_authority_affinity()
            and not context.projection_semantics.has_qualified_authority_reference(
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
AuthorityIdsByFactName: TypeAlias = dict[tuple[SemanticFactKind, str], frozenset[str]]
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

    def fact(self, fact_id: str) -> SemanticFact:
        return self.by_id[fact_id]

    def facts_for_edge(self, edge: MirrorEdge) -> tuple[SemanticFact, ...]:
        return tuple(self.fact(fact_ref.fact_id) for fact_ref in edge.match.fact_refs)


@dataclass(frozen=True)
class SemanticFactSpecificityIndex:
    """Score whether matched facts identify a specific authority or generic roles."""

    facts: tuple[SemanticFact, ...]

    @cached_property
    def authority_ids_by_fact_name(self) -> AuthorityIdsByFactName:
        authority_ids: dict[tuple[SemanticFactKind, str], set[str]] = {}
        for fact in self.facts:
            authority_ids.setdefault((fact.kind, fact.name), set()).add(
                fact.authority_id
            )
        return {
            key: frozenset(value)
            for key, value in sorted(
                authority_ids.items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
        }

    def fact_is_reused_role(self, fact: SemanticFact) -> bool:
        return len(self.authority_ids_by_fact_name[(fact.kind, fact.name)]) > 1

    def matched_facts_are_reused_roles(
        self,
        matched_facts: tuple[SemanticFact, ...],
    ) -> bool:
        return bool(matched_facts) and all(
            self.fact_is_reused_role(fact) for fact in matched_facts
        )


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
class SemanticAuthorityCatalog:
    """Nominal lookup catalog for semantic authorities."""

    authorities: tuple[SemanticAuthority, ...]

    @cached_property
    def by_id(self) -> dict[str, SemanticAuthority]:
        return {authority.authority_id: authority for authority in self.authorities}

    def authority(self, authority_id: str) -> SemanticAuthority:
        return self.by_id[authority_id]

    def authority_for_edge(self, edge: MirrorEdge) -> SemanticAuthority:
        return self.authority(edge.authority_id)


@dataclass(frozen=True)
class PresentationProjectionCatalog:
    """Nominal lookup catalog for presentation projections."""

    projections: tuple[PresentationProjection, ...]

    @cached_property
    def by_id(self) -> dict[str, PresentationProjection]:
        return {projection.projection_id: projection for projection in self.projections}

    def projection(self, projection_id: str) -> PresentationProjection:
        return self.by_id[projection_id]

    def projection_for_edge(self, edge: MirrorEdge) -> PresentationProjection:
        return self.projection(edge.projection_id)


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
    authority_catalog: SemanticAuthorityCatalog
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
            if authority_id not in self.authority_catalog.by_id:
                continue
            authority = self.authority_catalog.authority(authority_id)
            if not authority.kind.is_class_family_like:
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

    @classmethod
    def mirrored_without_descent(
        cls,
        edge: MirrorEdge,
        path_description: str,
    ) -> "DescentCertificate":
        return cls(
            DescentStatus.MIRRORED_WITHOUT_DESCENT,
            edge,
            path_description,
        )

    @classmethod
    def from_mirror_candidate(
        cls,
        edge: MirrorEdge,
        candidate: SemanticMirrorEdgeCandidate,
    ) -> "DescentCertificate":
        return cls.mirrored_without_descent(edge, candidate.missing_derivation_path)


@dataclass(frozen=True)
class SemanticDescentCertificateBuilder:
    """Build cached descent certificates from resolved mirror edges."""

    graph_space: "SemanticDescentGraphSpace"

    def certificates_for_edges(
        self,
        edges: tuple[MirrorEdge, ...],
    ) -> tuple[DescentCertificate, ...]:
        return tuple(self.certificate_for_edge(edge) for edge in edges)

    def certificate_for_edge(self, edge: MirrorEdge) -> DescentCertificate:
        authority = self.graph_space.authority_catalog.authority_for_edge(edge)
        projection = self.graph_space.projection_catalog.projection_for_edge(edge)
        candidate = SemanticMirrorEdgeCandidate(
            projection=projection,
            authority=authority,
            facts=self.graph_space.fact_authority_index.facts_for_edge(edge),
            match=edge.match,
        )
        return DescentCertificate.from_mirror_candidate(edge, candidate)


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
    def authority_catalog(self) -> SemanticAuthorityCatalog:
        return SemanticAuthorityCatalog(self.authorities)

    @cached_property
    def projection_catalog(self) -> PresentationProjectionCatalog:
        return PresentationProjectionCatalog(self.projections)

    @cached_property
    def fact_token_index(self) -> SemanticFactTokenIndex:
        return SemanticFactTokenIndex(self.facts)

    @cached_property
    def fact_specificity_index(self) -> SemanticFactSpecificityIndex:
        return SemanticFactSpecificityIndex(self.facts)

    @cached_property
    def facts_by_authority_id(self) -> FactsByAuthorityId:
        return self.fact_authority_index.by_authority_id


@dataclass(frozen=True)
class SemanticDescentGraph(SemanticDescentGraphSpace):
    """Repository-level graph of authorities, projections, and descent failures."""

    mirror_edges: tuple[MirrorEdge, ...]
    certificates: tuple[DescentCertificate, ...]
    class_index: ClassFamilyIndex | None = None

    def overlay_modules(
        self,
        changed_modules: tuple[ParsedModule, ...],
    ) -> "SemanticDescentGraph":
        if not changed_modules or self.class_index is None:
            return self
        return SemanticDescentGraphModuleOverlay(
            base_graph=self,
            changed_modules=changed_modules,
        ).graph()


@dataclass(frozen=True)
class SemanticDescentGraphModuleOverlay:
    """Refresh changed modules inside a cached repository semantic graph."""

    base_graph: SemanticDescentGraph
    changed_modules: tuple[ParsedModule, ...]

    def graph(self) -> SemanticDescentGraph:
        class_index = self.merged_class_index()
        authorities, facts = self.merged_authorities_and_facts(class_index)
        projections = self.merged_projections(class_index)
        graph_space = SemanticDescentGraphSpace(
            authorities,
            facts,
            projections,
        )
        mirror_edges = self.merged_mirror_edges(graph_space, class_index)
        certificates = SemanticDescentCertificateBuilder(
            graph_space
        ).certificates_for_edges(mirror_edges)
        return SemanticDescentGraph(
            authorities=authorities,
            facts=facts,
            projections=projections,
            mirror_edges=mirror_edges,
            certificates=certificates,
            class_index=class_index,
        )

    def merged_authorities_and_facts(
        self,
        class_index: ClassFamilyIndex,
    ) -> tuple[tuple[SemanticAuthority, ...], tuple[SemanticFact, ...]]:
        return SemanticAuthorityBuilder(
            self.changed_modules,
            class_index,
        ).build()

    def merged_mirror_edges(
        self,
        graph_space: SemanticDescentGraphSpace,
        class_index: ClassFamilyIndex,
    ) -> tuple[MirrorEdge, ...]:
        changed_projection_ids = self.changed_projection_ids(graph_space.projections)
        changed_authority_ids = self.changed_authority_ids(
            graph_space.authorities,
            graph_space.facts,
        )
        current_projection_ids = frozenset(
            projection.projection_id for projection in graph_space.projections
        )
        current_authority_ids = frozenset(
            authority.authority_id for authority in graph_space.authorities
        )
        retained_edges = tuple(
            edge
            for edge in self.base_graph.mirror_edges
            if edge.projection_id in current_projection_ids
            and edge.authority_id in current_authority_ids
            and edge.projection_id not in changed_projection_ids
            and edge.authority_id not in changed_authority_ids
        )
        resolver = SemanticMirrorResolver(
            graph_space.authorities,
            graph_space.facts,
            graph_space.projections,
            class_index,
        )
        recomputed_edges = (
            *resolver.edges_for_projection_ids(changed_projection_ids),
            *resolver.edges_for_authority_ids(changed_authority_ids),
        )
        return self._deduplicate_edges((*retained_edges, *recomputed_edges))

    def changed_projection_ids(
        self,
        projections: tuple[PresentationProjection, ...],
    ) -> frozenset[str]:
        base_projections = self.base_graph.projection_catalog.by_id
        return frozenset(
            projection.projection_id
            for projection in projections
            if self.resolved_path_text(projection.location.file_path)
            in self.changed_path_texts
            or base_projections.get(projection.projection_id) != projection
        )

    def changed_authority_ids(
        self,
        authorities: tuple[SemanticAuthority, ...],
        facts: tuple[SemanticFact, ...],
    ) -> frozenset[str]:
        base_authorities = self.base_graph.authority_catalog.by_id
        base_facts = self.base_graph.facts_by_authority_id
        current_facts = SemanticFactAuthorityIndex(facts).by_authority_id
        current_authorities = {
            authority.authority_id: authority for authority in authorities
        }
        changed_ids = {
            authority.authority_id
            for authority in authorities
            if base_authorities.get(authority.authority_id) != authority
            or base_facts.get(authority.authority_id, ())
            != current_facts.get(authority.authority_id, ())
        }
        changed_ids.update(
            authority_id
            for authority_id in base_authorities
            if authority_id not in current_authorities
        )
        return frozenset(changed_ids)

    @staticmethod
    def _deduplicate_edges(edges: tuple[MirrorEdge, ...]) -> tuple[MirrorEdge, ...]:
        by_reference = {
            (edge.authority_id, edge.projection_id): edge for edge in edges
        }
        return sorted_tuple(
            by_reference.values(),
            key=lambda item: (
                -item.match.fact_count,
                item.authority_id,
                item.projection_id,
            ),
        )

    def merged_class_index(self) -> ClassFamilyIndex:
        if self.base_graph.class_index is None:
            raise ValueError("semantic graph overlay requires a cached class index")
        return overlay_class_family_index(
            self.base_graph.class_index,
            self.changed_modules,
        )

    def merged_projections(
        self,
        class_index: ClassFamilyIndex,
    ) -> tuple[PresentationProjection, ...]:
        changed_path_texts = self.changed_path_texts
        unchanged_projections = tuple(
            projection
            for projection in self.base_graph.projections
            if self.resolved_path_text(projection.location.file_path)
            not in changed_path_texts
        )
        changed_projections = SemanticProjectionCollector(
            self.changed_modules,
            class_index,
        ).collect()
        return sorted_tuple(
            (*unchanged_projections, *changed_projections),
            key=lambda item: (item.location.file_path, item.location.line, item.label),
        )

    @cached_property
    def changed_path_texts(self) -> frozenset[str]:
        return frozenset(
            self.resolved_path_text(str(module.path)) for module in self.changed_modules
        )

    @staticmethod
    def resolved_path_text(file_path: str) -> str:
        return str(Path(file_path).resolve())


@dataclass(frozen=True)
class SemanticDescentAuthorityKindCount(SemanticRecord):
    """Count of graph authorities for one nominal authority kind."""

    authority_kind: str
    count: int


@dataclass(frozen=True)
class SemanticDescentProjectionKindCount(SemanticRecord):
    """Count of graph projections for one presentation projection kind."""

    projection_kind: str
    count: int


@dataclass(frozen=True)
class SemanticDescentCertificateSummary(SemanticRecord):
    """Compact report row for one missing semantic-descent certificate."""

    authority_name: str
    authority_kind: str
    projection_label: str
    projection_kind: str
    projection_owner_symbol: str
    file_path: str
    line: int
    matched_fact_count: int
    coverage_ratio: float
    matched_tokens: tuple[str, ...]
    missing_derivation_path: str

    @classmethod
    def from_graph(
        cls,
        graph: SemanticDescentGraph,
        certificate: DescentCertificate,
    ) -> "SemanticDescentCertificateSummary":
        edge = certificate.edge
        authority = graph.authority_catalog.authority_for_edge(edge)
        projection = graph.projection_catalog.projection_for_edge(edge)
        return cls(
            authority_name=authority.name,
            authority_kind=authority.kind.value,
            projection_label=projection.label,
            projection_kind=projection.kind.value,
            projection_owner_symbol=projection.owner_symbol,
            file_path=projection.location.file_path,
            line=projection.location.line,
            matched_fact_count=edge.match.fact_count,
            coverage_ratio=edge.match.coverage_ratio,
            matched_tokens=edge.match.tokens,
            missing_derivation_path=certificate.missing_derivation_path,
        )


@dataclass(frozen=True)
class SemanticDescentGraphReport(SemanticRecord):
    """Compact acceptance report for cached semantic-descent graph objects."""

    authority_count: int
    fact_count: int
    projection_count: int
    mirror_edge_count: int
    certificate_count: int
    authorities_by_kind: tuple[SemanticDescentAuthorityKindCount, ...]
    projections_by_kind: tuple[SemanticDescentProjectionKindCount, ...]
    top_certificates: tuple[SemanticDescentCertificateSummary, ...]

    @classmethod
    def from_graph(
        cls,
        graph: SemanticDescentGraph,
        *,
        certificate_limit: int = 10,
    ) -> "SemanticDescentGraphReport":
        return cls(
            authority_count=len(graph.authorities),
            fact_count=len(graph.facts),
            projection_count=len(graph.projections),
            mirror_edge_count=len(graph.mirror_edges),
            certificate_count=len(graph.certificates),
            authorities_by_kind=tuple(
                SemanticDescentAuthorityKindCount(authority_kind, count)
                for authority_kind, count in sorted(
                    Counter(
                        authority.kind.value for authority in graph.authorities
                    ).items()
                )
            ),
            projections_by_kind=tuple(
                SemanticDescentProjectionKindCount(projection_kind, count)
                for projection_kind, count in sorted(
                    Counter(
                        projection.kind.value for projection in graph.projections
                    ).items()
                )
            ),
            top_certificates=tuple(
                SemanticDescentCertificateSummary.from_graph(graph, certificate)
                for certificate in sorted_tuple(
                    graph.certificates,
                    key=lambda item: (
                        -item.edge.match.fact_count,
                        graph.authority_catalog.authority_for_edge(item.edge).name,
                        graph.projection_catalog.projection_for_edge(item.edge).label,
                    ),
                )[:certificate_limit]
            ),
        )


@dataclass(frozen=True)
class SemanticDescentGraphPayloadReport(SemanticRecord):
    """JSON-facing report that separates repository and finding-backed graphs."""

    active_graph_source: str
    repository_graph: SemanticDescentGraphReport
    finding_backed_graph: SemanticDescentGraphReport | None = None

    @classmethod
    def from_graphs(
        cls,
        repository_graph: SemanticDescentGraph,
        *,
        finding_backed_graph: SemanticDescentGraph | None = None,
        certificate_limit: int = 10,
    ) -> "SemanticDescentGraphPayloadReport":
        repository_report = SemanticDescentGraphReport.from_graph(
            repository_graph,
            certificate_limit=certificate_limit,
        )
        finding_backed_report = (
            None
            if finding_backed_graph is None
            else SemanticDescentGraphReport.from_graph(
                finding_backed_graph,
                certificate_limit=certificate_limit,
            )
        )
        active_graph_source = "repository"
        if (
            not repository_report.certificate_count
            and finding_backed_report is not None
            and finding_backed_report.certificate_count
        ):
            active_graph_source = "finding_backed"
        return cls(
            active_graph_source=active_graph_source,
            repository_graph=repository_report,
            finding_backed_graph=finding_backed_report,
        )


def semantic_descent_finding_authority_id(finding: RefactorFinding) -> str:
    return f"finding:{finding.stable_id}:authority"


def semantic_descent_finding_projection_id(finding: RefactorFinding) -> str:
    return f"finding:{finding.stable_id}:projection"


@dataclass(frozen=True)
class FindingBackedSemanticDescentGraphRequest:
    """Graph request for findings projected into semantic-descent certificates."""

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
            findings=tuple(findings),
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
            certificates.append(
                FindingBackedCertificateProjection.certificate(finding, edge)
            )
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
        authority_name = FindingBackedAuthorityNameProjection.authority_name(
            finding,
            authority_location,
            prefer_metric_authority=(
                authority_evidence_index_by_detector_id.get(finding.detector_id) is None
            ),
        )
        return SemanticAuthority(
            authority_id=authority_id,
            kind=SemanticAuthorityKind.FINDING_DECLARED_AUTHORITY,
            name=authority_name,
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


class FindingBackedAuthorityNameProjection:
    """Project finding evidence symbols onto the nominal owner they imply."""

    @classmethod
    def authority_name(
        cls,
        finding: RefactorFinding,
        location: SourceLocation,
        *,
        prefer_metric_authority: bool,
    ) -> str:
        metric_candidates = (
            cls._metric_authority_candidates(finding.metrics)
            if prefer_metric_authority
            else ()
        )
        evidence_candidates = cls._evidence_owner_candidates(finding)
        location_candidates = cls._authority_name_candidates(location.symbol)
        authority_candidates = (
            (
                *metric_candidates[:1],
                *evidence_candidates,
                *location_candidates,
                *metric_candidates[1:],
            )
            if prefer_metric_authority
            else (
                *location_candidates,
                *evidence_candidates,
            )
        )
        return (
            FindingAuthorityNamePolicy.first_specific_name(*authority_candidates)
            or location.symbol
        )

    @staticmethod
    def _authority_name_candidates(symbol: str) -> tuple[str, ...]:
        if "." not in symbol:
            return (symbol,)
        owner, _member = symbol.split(".", 1)
        return (owner, symbol)

    @staticmethod
    def _metric_authority_candidates(
        metrics: FindingMetrics,
    ) -> tuple[str | None, ...]:
        projection = FindingMetricsSemanticProjection.projection_for(metrics)
        if projection is None:
            return ()
        return projection.authority_name_candidate_names(metrics)

    @classmethod
    def _evidence_owner_candidates(
        cls,
        finding: RefactorFinding,
    ) -> tuple[str, ...]:
        owner_names = tuple(
            cls._symbol_owner_name(location.symbol) for location in finding.evidence
        )
        owner_names = tuple(name for name in owner_names if name)
        common_prefix = CLASS_NAME_ALGEBRA.public_name_from_tokens(
            CLASS_NAME_ALGEBRA.longest_common_token_prefix(owner_names)
        )
        common_suffix = CLASS_NAME_ALGEBRA.public_name_from_tokens(
            CLASS_NAME_ALGEBRA.longest_common_token_suffix(owner_names)
        )
        multi_owner_candidates = tuple(
            dict.fromkeys(
                (
                    common_prefix,
                    common_suffix,
                    *owner_names,
                )
            )
        )
        return owner_names if len(owner_names) <= 1 else multi_owner_candidates

    @classmethod
    def _symbol_owner_name(cls, symbol: str) -> str:
        owner = symbol.split(":", 1)[0]
        return cls._authority_name_candidates(owner)[0]


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
        metric_names = FindingMetricsSemanticProjection.fact_names_for(finding.metrics)
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
            match=SemanticMirrorMatch.from_facts(facts),
        )


class FindingBackedCertificateProjection:
    """Project detector finding relation context into descent certificates."""

    @staticmethod
    def certificate(
        finding: RefactorFinding,
        edge: MirrorEdge,
    ) -> DescentCertificate:
        return DescentCertificate.mirrored_without_descent(
            edge,
            (
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


class FindingMetricsSemanticProjection(ABC, metaclass=AutoRegisterMeta):
    """Registered projection from finding metrics into descent-graph semantics."""

    __registry__: ClassVar[
        dict[type[FindingMetrics], type["FindingMetricsSemanticProjection"]]
    ] = {}
    __registry_key__ = "metrics_type"
    __skip_if_no_key__ = True
    metrics_type: ClassVar[type[FindingMetrics]]

    def authority_name_candidate_names(
        self,
        metrics: FindingMetrics,
    ) -> tuple[str | None, ...]:
        del metrics
        return ()

    def authority_name(self, metrics: FindingMetrics) -> str | None:
        return FindingAuthorityNamePolicy.first_specific_name(
            *self.authority_name_candidate_names(metrics)
        )

    @abstractmethod
    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def projection_for(
        cls,
        metrics: FindingMetrics,
    ) -> "FindingMetricsSemanticProjection | None":
        for projection_type in cls.__registry__.values():
            if isinstance(metrics, projection_type.metrics_type):
                return projection_type()
        return None

    @classmethod
    def authority_name_for(cls, metrics: FindingMetrics) -> str | None:
        projection = cls.projection_for(metrics)
        if projection is None:
            return None
        return projection.authority_name(metrics)

    @classmethod
    def fact_names_for(cls, metrics: FindingMetrics) -> tuple[str, ...]:
        projection = cls.projection_for(metrics)
        if projection is None:
            return ()
        return projection.fact_names(metrics)


class FindingAuthorityNamePolicy:
    """Select metric-derived authority names only when they carry identity."""

    bag_delimiters: ClassVar[frozenset[str]] = frozenset((",", "/", "|"))
    generic_tokens: ClassVar[
        frozenset[str]
    ] = SemanticRoleIdentityToken.identity_axis_values() | frozenset(
        (
            "authority",
            "candidate",
            "generic",
            "level",
            "local",
            "logic",
            "mapping",
            "projection",
            "semantic",
            "unknown",
        )
    )

    @classmethod
    def first_specific_name(cls, *names: str | None) -> str | None:
        for name in names:
            if name is not None and cls.is_specific_name(name):
                return name
        return None

    @classmethod
    def is_specific_name(cls, name: str) -> bool:
        tokens = NormalizeNameProjection.token_set(name)
        return bool(
            name
            and not any(delimiter in name for delimiter in cls.bag_delimiters)
            and tokens
            and tokens - cls.generic_tokens
        )


class MappingMetricsSemanticProjection(FindingMetricsSemanticProjection):
    """Use mapping metrics as source-authority and projected field facts."""

    metrics_type: ClassVar[type[FindingMetrics]] = MappingMetrics

    def authority_name_candidate_names(
        self,
        metrics: FindingMetrics,
    ) -> tuple[str | None, ...]:
        if not isinstance(metrics, MappingMetrics):
            return ()
        return (metrics.source_name, metrics.mapping_name)

    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        if not isinstance(metrics, MappingMetrics):
            return ()
        return metrics.field_names or metrics.identity_field_names


class RegistrationMetricsSemanticProjection(FindingMetricsSemanticProjection):
    """Use registration metrics as registry-authority and registered facts."""

    metrics_type: ClassVar[type[FindingMetrics]] = RegistrationMetrics

    def authority_name_candidate_names(
        self,
        metrics: FindingMetrics,
    ) -> tuple[str | None, ...]:
        if not isinstance(metrics, RegistrationMetrics):
            return ()
        return (metrics.registry_name,)

    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        if not isinstance(metrics, RegistrationMetrics):
            return ()
        return metrics.class_names or tuple(
            class_key_pair.split("=", 1)[0]
            for class_key_pair in metrics.class_key_pairs
        )


class FallbackMetricsSemanticProjection(FindingMetricsSemanticProjection):
    """Use generic plan fields when no more specific metrics projection exists."""

    metrics_type: ClassVar[type[FindingMetrics]] = FindingMetrics

    def authority_name_candidate_names(
        self,
        metrics: FindingMetrics,
    ) -> tuple[str | None, ...]:
        return (
            metrics.plan_source_name,
            metrics.plan_mapping_name,
            metrics.plan_registry_name,
        )

    def fact_names(self, metrics: FindingMetrics) -> tuple[str, ...]:
        return (
            metrics.plan_field_names
            or metrics.plan_identity_field_names
            or metrics.plan_class_names
            or metrics.plan_literal_cases
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

    def _load_payload(
        self,
        cache_path: Path,
    ) -> (
        dict[
            str,
            SemanticDescentGraph
            | SemanticDescentGraphCacheIdentity
            | SemanticDescentGraphCacheFamilyIdentity,
        ]
        | None
    ):
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise SemanticDescentGraphCacheReadError(
                f"Could not read semantic-descent graph cache entry {cache_path.name}"
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticDescentGraphCacheReadError(
                f"Semantic-descent graph cache entry {cache_path.name} is not a mapping"
            )
        return payload

    def load(
        self,
        identity: SemanticDescentGraphCacheIdentity,
    ) -> SemanticDescentGraphCacheLookup:
        if self.storage_root is None:
            return SemanticDescentGraphCacheDisabled()
        payload = self._load_payload(self._entry_path(identity))
        if payload is None:
            return SemanticDescentGraphCacheMiss()
        if payload.get("identity") != identity:
            return SemanticDescentGraphCacheMiss()
        graph = payload.get("graph")
        if not isinstance(graph, SemanticDescentGraph):
            raise SemanticDescentGraphCacheReadError(
                f"Semantic-descent graph cache entry {identity.cache_token} has invalid graph payload"
            )
        return SemanticDescentGraphCacheHit(graph)

    def load_latest(
        self,
        family_identity: SemanticDescentGraphCacheFamilyIdentity,
    ) -> SemanticDescentGraphCacheLookup:
        if self.storage_root is None:
            return SemanticDescentGraphCacheDisabled()
        payload = self._load_payload(self._latest_path(family_identity))
        if payload is None:
            return SemanticDescentGraphCacheMiss()
        if payload.get("family_identity") != family_identity:
            return SemanticDescentGraphCacheMiss()
        graph = payload.get("graph")
        if not isinstance(graph, SemanticDescentGraph):
            raise SemanticDescentGraphCacheReadError(
                f"Semantic-descent graph latest cache entry {family_identity.cache_token} has invalid graph payload"
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
            family_identity = SemanticDescentGraphCacheFamilyIdentity.from_identity(
                identity
            )
            with self._latest_path(family_identity).open("wb") as handle:
                pickle.dump(
                    {
                        "family_identity": family_identity,
                        "identity": identity,
                        "graph": graph,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except OSError:
            return

    def _entry_path(self, identity: SemanticDescentGraphCacheIdentity) -> Path:
        if self.storage_root is None:
            raise ValueError("semantic descent graph cache directory is disabled")
        return self.storage_root / f"{identity.cache_token}.pickle"

    def _latest_path(
        self,
        family_identity: SemanticDescentGraphCacheFamilyIdentity,
    ) -> Path:
        if self.storage_root is None:
            raise ValueError("semantic descent graph cache directory is disabled")
        return self.storage_root / f"latest-{family_identity.cache_token}.pickle"


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


def load_cached_semantic_descent_graph_for_roots(
    roots: tuple[Path, ...],
    *,
    cache_dir: Path | None,
    source_policy: PythonSourcePathPolicy | None = None,
) -> SemanticDescentGraph | None:
    """Load a semantic-descent graph cache entry addressable before AST parsing."""

    identities = python_module_path_identities_for_roots(
        roots,
        source_policy=source_policy,
    )
    identity = SemanticDescentGraphCacheIdentity.from_path_identities(identities)
    return SemanticDescentGraphCache(cache_dir).load(identity).graph


def load_latest_semantic_descent_graph_for_roots(
    roots: tuple[Path, ...],
    *,
    cache_dir: Path | None,
    source_policy: PythonSourcePathPolicy | None = None,
) -> SemanticDescentGraph | None:
    """Load the latest graph for a source-set family before AST parsing."""

    identities = python_module_path_identities_for_roots(
        roots,
        source_policy=source_policy,
    )
    family_identity = SemanticDescentGraphCacheFamilyIdentity.from_path_identities(
        identities
    )
    return SemanticDescentGraphCache(cache_dir).load_latest(family_identity).graph


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
    graph_space = SemanticDescentGraphSpace(
        authorities,
        facts,
        projections,
    )
    certificate_builder = SemanticDescentCertificateBuilder(graph_space)
    certificates = certificate_builder.certificates_for_edges(mirror_edges)
    return SemanticDescentGraph(
        authorities=authorities,
        facts=facts,
        projections=projections,
        mirror_edges=mirror_edges,
        certificates=certificates,
        class_index=class_index,
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
        for name, value in AutoRegisterClassAuthority(node).assignment_pairs:
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
        for name, value in AutoRegisterClassAuthority(
            indexed_class.node
        ).assignment_pairs:
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
                        value.lineno,
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

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and self._collect_return_projection(node, node.value):
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
        return self._collect_value_projection(
            node,
            value,
            label=label,
            allow_call_projection=self.current_function_name is None,
        )

    def _collect_return_projection(self, node: ast.Return, value: ast.AST) -> bool:
        return self._collect_value_projection(
            node,
            value,
            label=f"{self.qualname}:return@{node.lineno}",
            allow_call_projection=False,
        )

    def _collect_value_projection(
        self,
        node: ast.stmt,
        value: ast.AST,
        *,
        label: str,
        allow_call_projection: bool,
    ) -> bool:
        projection_kind = self._projection_kind(value, allow_call_projection)
        if projection_kind is None:
            return False
        key_value_pairs = (
            self._projection_key_value_pairs(value)
            if isinstance(value, ast.Dict)
            else ()
        )
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
    def _projection_kind(
        value: ast.AST,
        allow_call_projection: bool,
    ) -> PresentationProjectionKind | None:
        if isinstance(value, ast.Dict):
            return PresentationProjectionKind.MAPPING_LITERAL
        if isinstance(value, ast.List | ast.Tuple | ast.Set):
            return PresentationProjectionKind.COLLECTION_LITERAL
        if isinstance(value, ast.Call) and allow_call_projection:
            return PresentationProjectionKind.CALL_LITERAL
        return None

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
class SemanticMirrorPolicyCatalog:
    """Nominal policy lookup for semantic mirror authority kinds."""

    authority_catalog: SemanticAuthorityCatalog

    def policy_for_authority(
        self,
        authority: SemanticAuthority,
    ) -> SemanticAuthorityMirrorPolicy:
        return SemanticAuthorityMirrorPolicy.for_authority(authority)

    def policy_for_authority_id(
        self,
        authority_id: str,
    ) -> SemanticAuthorityMirrorPolicy:
        return self.policy_for_authority(self.authority_catalog.authority(authority_id))


@dataclass(frozen=True)
class ProjectionSemanticAuthority:
    """Projection-level syntax and affinity predicates for mirror policies."""

    @staticmethod
    def has_authority_affinity(
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        authority_tokens = NormalizeNameProjection.token_set(authority.name)
        projection_tokens = NormalizeNameProjection.token_set(
            f"{projection.label} {projection.owner_symbol} {projection.location.symbol}"
        )
        return len(authority_tokens & projection_tokens) >= 2

    @staticmethod
    def has_qualified_authority_reference(
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            token.kind is PresentationTokenKind.QUALIFIED_ATTRIBUTE
            and token.qualifier == authority.name
            for token in projection.tokens
        )

    @staticmethod
    def enum_branch_has_case_syntax(
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
    def dataclass_branch_has_field_syntax(
        projection: PresentationProjection,
        matched_tokens: frozenset[str],
    ) -> bool:
        return any(
            token.kind is PresentationTokenKind.STRING_LITERAL
            and token.value in matched_tokens
            for token in projection.tokens
        )


@dataclass(frozen=True)
class ProjectionClassSymbolLineageIndex:
    """Resolve presentation projections into indexed class lineage."""

    class_index: ClassFamilyIndex
    projections: tuple[PresentationProjection, ...]

    @cached_property
    def class_symbols_by_projection_id(self) -> dict[str, str | None]:
        return {
            projection.projection_id: self._resolve_class_symbol(projection)
            for projection in self.projections
        }

    @cached_property
    def ancestor_symbols_by_class_symbol(self) -> dict[str, frozenset[str]]:
        return {
            symbol: frozenset(self.class_index.ancestor_symbols(symbol))
            for symbol in self.class_index.classes_by_symbol
        }

    def class_symbol_for_projection(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        return self.class_symbols_by_projection_id[projection.projection_id]

    def _resolve_class_symbol(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        for end_index in range(len(projection.owner.qualname_parts), 0, -1):
            owner_qualname = ".".join(projection.owner.qualname_parts[:end_index])
            symbol = self.class_index.symbol_for(
                file_path=projection.location.file_path,
                qualname=owner_qualname,
            )
            if symbol is not None:
                return symbol
        return None


@dataclass(frozen=True)
class DataclassAuthorityNameAffinity:
    """Conservative name affinity between dataclass schema authorities."""

    left: SemanticAuthority
    right: SemanticAuthority

    @cached_property
    def left_tokens(self) -> frozenset[str]:
        return self.specific_tokens(self.left.name)

    @cached_property
    def right_tokens(self) -> frozenset[str]:
        return self.specific_tokens(self.right.name)

    def has_affinity(self) -> bool:
        return bool(self.left_tokens & self.right_tokens)

    @classmethod
    def specific_tokens(cls, raw_name: str) -> frozenset[str]:
        return frozenset(
            token
            for token in NormalizeNameProjection.token_set(raw_name)
            if token not in cls.weak_tokens()
        )

    @staticmethod
    def weak_tokens() -> frozenset[str]:
        return SemanticRoleIdentityToken.authority_affinity_weak_values() | frozenset(
            (
                "candidate",
                "count",
                "file",
                "line",
                "path",
                "range",
                "replacement",
                "run",
                "source",
                "span",
            )
        )


@dataclass(frozen=True)
class ConstructionAuthorityResolver:
    """Resolve owner construction sites that descend to semantic authorities."""

    class_index: ClassFamilyIndex

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

    def construction_type_descends_to_authority(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        if self.construction_type_is_authority_class(construction, authority):
            return True
        return self.construction_type_materializes_authority(construction, authority)

    def construction_type_is_authority_class(
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

    def construction_type_materializes_authority(
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
            cache[cache_key] = compute(construction, authority)
        return cache[cache_key]

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
        for _, value in AutoRegisterClassAuthority(indexed_class.node).assignment_pairs:
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
            self.construction_type_is_authority_class(
                PresentationAuthorityConstruction(type_name, ()),
                authority,
            )
            for type_name in PresentationAuthorityConstructionCollector.construction_type_names(
                node
            )
        )


@dataclass(frozen=True)
class DataclassProjectionDescentAuthority:
    """Dataclass-schema descent semantics for presentation projections."""

    authorities: tuple[SemanticAuthority, ...]
    projections: tuple[PresentationProjection, ...]
    fact_authority_index: SemanticFactAuthorityIndex
    policy_catalog: SemanticMirrorPolicyCatalog
    projection_class_symbol_lineage: ProjectionClassSymbolLineageIndex
    construction_resolver: ConstructionAuthorityResolver

    @cached_property
    def dataclass_authorities(self) -> tuple[SemanticAuthority, ...]:
        return tuple(
            authority
            for authority in self.authorities
            if self.policy_catalog.policy_for_authority(
                authority
            ).dataclass_authority_selected
        )

    @cached_property
    def dataclass_authority_ids(self) -> frozenset[str]:
        return frozenset(
            authority.authority_id for authority in self.dataclass_authorities
        )

    @cached_property
    def projection_descent_authority_ids(self) -> dict[str, frozenset[str]]:
        return {
            projection.projection_id: self._projection_descent_authority_ids(projection)
            for projection in self.projections
        }

    @cached_property
    def projection_materializes_any_dataclass_authority_cache(self) -> dict[str, bool]:
        return {}

    @cached_property
    def constructed_dataclass_authorities_by_projection_id(
        self,
    ) -> dict[str, tuple[SemanticAuthority, ...]]:
        return {}

    def projection_descends_to_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return (
            authority.authority_id
            in self.projection_descent_authority_ids[projection.projection_id]
        )

    def projection_descends_to_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        return any(
            self.projection_descends_to_authority(projection, authority)
            for authority in self.dataclass_authorities
        )

    def projection_materializes_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        cache = self.projection_materializes_any_dataclass_authority_cache
        if projection.projection_id not in cache:
            cache[projection.projection_id] = (
                self._projection_materializes_any_dataclass_authority_uncached(
                    projection,
                )
            )
        return cache[projection.projection_id]

    def projection_constructs_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        return bool(self.constructed_dataclass_authorities(projection))

    def projection_constructs_unrelated_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        return any(
            constructed_authority.authority_id == authority.authority_id
            or not DataclassAuthorityNameAffinity(
                constructed_authority,
                authority,
            ).has_affinity()
            for constructed_authority in self.constructed_dataclass_authorities(
                projection,
            )
        )

    def constructed_dataclass_authorities(
        self,
        projection: PresentationProjection,
    ) -> tuple[SemanticAuthority, ...]:
        cache = self.constructed_dataclass_authorities_by_projection_id
        if projection.projection_id not in cache:
            cache[projection.projection_id] = tuple(
                authority
                for authority in self.dataclass_authorities
                if self.projection_owner_constructs_dataclass_authority(
                    projection,
                    authority,
                    self.fact_authority_index.facts_for_authority(authority.authority_id),
                )
            )
        return cache[projection.projection_id]

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
            self.construction_resolver.construction_type_descends_to_authority,
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
            self.construction_resolver.construction_type_materializes_authority,
        )

    def projection_shares_dataclass_base_with_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        projection_class_symbol = (
            self.projection_class_symbol_lineage.class_symbol_for_projection(projection)
        )
        if projection_class_symbol is None:
            return False
        shared_ancestors = (
            self.projection_class_symbol_lineage.ancestor_symbols_by_class_symbol[
                projection_class_symbol
            ]
            & self.projection_class_symbol_lineage.ancestor_symbols_by_class_symbol[
                authority.authority_id
            ]
            & self.dataclass_authority_ids
        )
        return bool(shared_ancestors)

    def _projection_descent_authority_ids(
        self,
        projection: PresentationProjection,
    ) -> frozenset[str]:
        projection_class_symbol = (
            self.projection_class_symbol_lineage.class_symbol_for_projection(projection)
        )
        if projection_class_symbol is None:
            return frozenset()
        projection_ancestor_symbols = (
            self.projection_class_symbol_lineage.ancestor_symbols_by_class_symbol[
                projection_class_symbol
            ]
        )
        return frozenset(
            authority_id
            for authority_id in self.dataclass_authority_ids
            if projection_class_symbol == authority_id
            or authority_id in projection_ancestor_symbols
        )

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


@dataclass(frozen=True)
class SemanticMirrorResolutionContext:
    """Composed policy context for deciding mirror admissibility and descent."""

    projection_semantics: ProjectionSemanticAuthority
    dataclass_descent: DataclassProjectionDescentAuthority
    fact_specificity: SemanticFactSpecificityIndex


@dataclass(frozen=True)
class SemanticMirrorResolver(SemanticDescentGraphSpace):
    """Resolve graph edges where a projection mirrors an authority."""

    class_index: ClassFamilyIndex

    @cached_property
    def policy_catalog(self) -> SemanticMirrorPolicyCatalog:
        return SemanticMirrorPolicyCatalog(self.authority_catalog)

    @cached_property
    def projection_semantics(self) -> ProjectionSemanticAuthority:
        return ProjectionSemanticAuthority()

    @cached_property
    def projection_class_symbol_lineage(self) -> ProjectionClassSymbolLineageIndex:
        return ProjectionClassSymbolLineageIndex(self.class_index, self.projections)

    @cached_property
    def construction_resolver(self) -> ConstructionAuthorityResolver:
        return ConstructionAuthorityResolver(self.class_index)

    @cached_property
    def dataclass_descent(self) -> DataclassProjectionDescentAuthority:
        return DataclassProjectionDescentAuthority(
            authorities=self.authorities,
            projections=self.projections,
            fact_authority_index=self.fact_authority_index,
            policy_catalog=self.policy_catalog,
            projection_class_symbol_lineage=self.projection_class_symbol_lineage,
            construction_resolver=self.construction_resolver,
        )

    @cached_property
    def resolution_context(self) -> SemanticMirrorResolutionContext:
        return SemanticMirrorResolutionContext(
            projection_semantics=self.projection_semantics,
            dataclass_descent=self.dataclass_descent,
            fact_specificity=self.fact_specificity_index,
        )

    def edges(self) -> tuple[MirrorEdge, ...]:
        return self._edges_for(self.projections, None)

    def edges_for_projection_ids(
        self,
        projection_ids: frozenset[str],
    ) -> tuple[MirrorEdge, ...]:
        if not projection_ids:
            return ()
        return self._edges_for(
            tuple(
                projection
                for projection in self.projections
                if projection.projection_id in projection_ids
            ),
            None,
        )

    def edges_for_authority_ids(
        self,
        authority_ids: frozenset[str],
    ) -> tuple[MirrorEdge, ...]:
        if not authority_ids:
            return ()
        return self._edges_for(self.projections, authority_ids)

    def _edges_for(
        self,
        projections: tuple[PresentationProjection, ...],
        authority_ids: frozenset[str] | None,
    ) -> tuple[MirrorEdge, ...]:
        edges: list[MirrorEdge] = []
        for projection in projections:
            for authority_id, matches in self._matches_by_authority(
                projection,
            ).items():
                if authority_ids is not None and authority_id not in authority_ids:
                    continue
                edge = self._edge_for(
                    projection,
                    self.authority_catalog.authority(authority_id),
                    self.fact_authority_index.facts_for_authority(authority_id),
                    matches,
                )
                if edge is not None:
                    edges.append(edge)
        return sorted_tuple(
            edges,
            key=lambda item: (
                -item.match.fact_count,
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
                self.authority_catalog,
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
                if self.policy_catalog.policy_for_authority_id(
                    authority_id
                ).authority_qualified_token_reference_admitted
            )
            return tuple(
                ref for ref in refs if ref.authority_id in allowed_authority_ids
            )
        return tuple(
            ref
            for ref in refs
            if self.policy_catalog.policy_for_authority_id(
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
        match = SemanticMirrorMatch.from_authority_matches(facts, matches_by_fact_id)
        if match is None:
            return None
        candidate = SemanticMirrorEdgeCandidate(
            projection=projection,
            authority=authority,
            facts=facts,
            match=match,
        )
        policy = self.policy_catalog.policy_for_authority(authority)
        if not policy.edge_is_admissible(self.resolution_context, candidate):
            return None
        if policy.projection_descends_to_authority(self.resolution_context, candidate):
            return None
        return MirrorEdge(
            authority_id=authority.authority_id,
            projection_id=projection.projection_id,
            match=candidate.match,
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
