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
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property, lru_cache
from itertools import groupby
from typing import ClassVar, Generic, TypeAlias, TypeVar

from metaclass_registry import AutoRegisterMeta
from .assignment_projection import SingleAssignmentAndValueNameProjection
from .ast_tools import (
    ClassFunctionStackNodeVisitor,
    CollectedFamily,
    CompactModuleIdentity,
    ParsedModule,
    PythonModulePathIdentity,
    PythonSourcePathPolicy,
    module_syntax_index,
    python_module_path_identities_for_roots,
)
from . import class_index as class_index_module
from .cache_paths import default_semantic_descent_cache_dir
from .cache_checkout import (
    CacheCheckoutPathError,
    checkout_relative_path,
    inferred_checkout_roots,
    presentation_root_texts,
    rebase_checkout_path,
)
from .class_index import (
    ClassDeclaration,
    ClassFamilyIndex,
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
    IndexedClass,
    ModuleClassReferenceResolver,
    build_class_family_index,
    build_compact_class_family_index,
    overlay_class_family_index,
)
from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .deadline import scan_deadline_checkpoint
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

SemanticClassFamilyIndex: TypeAlias = ClassFamilyIndex | CompactClassFamilyIndex
IndexedClassDeclarationT = TypeVar(
    "IndexedClassDeclarationT",
    bound=ClassDeclaration,
)


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


class AuthorityClaimStatus(StrEnum):
    """Resolution state for an agent- or detector-authored authority claim."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    DECLARED = "declared"


class AuthorityProofEdgeKind(StrEnum):
    """Proof edge categories that make an authority claim non-prose."""

    SOURCE_INDEX_TARGET = "source_index_target"
    SEMANTIC_DESCENT_GRAPH = "semantic_descent_graph"
    EXPLICIT_DECLARATION = "explicit_declaration"
    INHERITS_FROM = "inherits_from"
    REGISTERED_BY = "registered_by"
    OWNS_FIELD_SET = "owns_field_set"
    DECLARES_ENUM_MEMBERS = "declares_enum_members"
    PROVIDES_QUERY_METHOD = "provides_query_method"


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
    DESCENDS_TO_AUTHORITY = "descends_to_authority"


class SemanticDescentGraphCacheReadError(RuntimeError):
    """Raised when an existing semantic-descent graph cache entry is invalid."""


@dataclass(frozen=True)
class SemanticDescentGraphCacheSchema:
    """Nominal schema identity for persisted semantic-descent graph entries."""

    version: int = 9
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
    def from_module(
        cls,
        module: ParsedModule,
        roots: tuple[Path | str, ...],
    ) -> "SemanticDescentModuleSignature":
        return cls(
            path=checkout_relative_path(module.path, roots),
            parsed_import_name=module.module_name,
            is_package_init=module.is_package_init,
            source_hash=_text_hash(module.source),
        )

    @classmethod
    def from_path_identity(
        cls,
        identity: PythonModulePathIdentity,
        roots: tuple[Path | str, ...],
    ) -> "SemanticDescentModuleSignature":
        return cls(
            path=checkout_relative_path(identity.path, roots),
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
    presentation_roots: tuple[str, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
        *,
        roots: tuple[Path | str, ...] | None = None,
    ) -> "SemanticDescentGraphCacheIdentity":
        effective_roots = (
            inferred_checkout_roots(tuple(module.path for module in modules))
            if roots is None
            else roots
        )
        return cls(
            schema=semantic_descent_graph_cache_schema,
            implementation=SemanticDescentImplementationSignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            modules=tuple(
                sorted(
                    (
                        SemanticDescentModuleSignature.from_module(
                            module,
                            effective_roots,
                        )
                        for module in modules
                    ),
                    key=lambda item: item.path,
                )
            ),
            presentation_roots=presentation_root_texts(effective_roots),
        )

    @classmethod
    def from_path_identities(
        cls,
        identities: tuple[PythonModulePathIdentity, ...],
        *,
        roots: tuple[Path | str, ...] | None = None,
    ) -> "SemanticDescentGraphCacheIdentity":
        effective_roots = (
            inferred_checkout_roots(tuple(identity.path for identity in identities))
            if roots is None
            else roots
        )
        return cls(
            schema=semantic_descent_graph_cache_schema,
            implementation=SemanticDescentImplementationSignature.current(),
            python_version=(sys.version_info.major, sys.version_info.minor),
            modules=tuple(
                sorted(
                    (
                        SemanticDescentModuleSignature.from_path_identity(
                            identity,
                            effective_roots,
                        )
                        for identity in identities
                    ),
                    key=lambda item: item.path,
                )
            ),
            presentation_roots=presentation_root_texts(effective_roots),
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
            ),
            roots=roots,
        )

    def relocated_to(
        self,
        roots: tuple[Path | str, ...],
    ) -> "SemanticDescentGraphCacheIdentity":
        """Bind checkout-independent semantics to a current presentation root."""

        return replace(self, presentation_roots=presentation_root_texts(roots))

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
    presentation_roots: tuple[str, ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

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
            presentation_roots=identity.presentation_roots,
        )

    @classmethod
    def from_path_identities(
        cls,
        identities: tuple[PythonModulePathIdentity, ...],
        *,
        roots: tuple[Path | str, ...] | None = None,
    ) -> "SemanticDescentGraphCacheFamilyIdentity":
        return cls.from_identity(
            SemanticDescentGraphCacheIdentity.from_path_identities(
                identities,
                roots=roots,
            )
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

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        return sorted_tuple(
            {
                variant
                for alias in self.aliases
                for variant in normalized_name_variants(alias)
            }
        )

    @classmethod
    def class_member(
        cls,
        authority_symbol: str,
        indexed_class: ClassDeclaration,
        aliases: Iterable[str],
    ) -> "SemanticFact":
        return cls(
            fact_id=f"{authority_symbol}:{indexed_class.symbol}",
            authority_id=authority_symbol,
            kind=SemanticFactKind.CLASS_MEMBER,
            name=indexed_class.simple_name,
            aliases=sorted_tuple((indexed_class.simple_name, *aliases)),
            location=SourceLocation(
                indexed_class.file_path,
                indexed_class.line,
                indexed_class.qualname,
            ),
        )

    @classmethod
    def dataclass_field(
        cls,
        indexed_class: ClassDeclaration,
        name: str,
        line: int,
    ) -> "SemanticFact":
        return cls(
            fact_id=f"{indexed_class.symbol}:{name}",
            authority_id=indexed_class.symbol,
            kind=SemanticFactKind.DATACLASS_FIELD,
            name=name,
            aliases=(name,),
            location=SourceLocation(
                indexed_class.file_path,
                line,
                f"{indexed_class.qualname}.{name}",
            ),
        )

    @classmethod
    def enum_member(
        cls,
        indexed_class: ClassDeclaration,
        name: str,
        line: int,
        string_value: str | None,
    ) -> "SemanticFact":
        return cls(
            fact_id=f"{indexed_class.symbol}:{name}",
            authority_id=indexed_class.symbol,
            kind=SemanticFactKind.ENUM_MEMBER,
            name=name,
            aliases=(
                (name,)
                if string_value is None
                else sorted_tuple((name, string_value))
            ),
            location=SourceLocation(
                indexed_class.file_path,
                line,
                f"{indexed_class.qualname}.{name}",
            ),
        )


@dataclass(frozen=True)
class SemanticAuthority(SemanticAuthorityReference):
    """Nominal source of truth for a semantic fact family."""

    kind: SemanticAuthorityKind
    name: str
    location: SourceLocation
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class AuthorityClaim(SemanticRecord):
    """Structured claim that a named authority exists or is being declared."""

    claimed_symbol: str
    authority_kind: str = ""
    file_path: str = ""
    qualname: str = ""
    authority_id: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "AuthorityClaim":
        return cls(
            claimed_symbol=cls.required_string(payload, "claimed_symbol"),
            authority_kind=cls.optional_string(payload, "authority_kind"),
            file_path=cls.optional_string(payload, "file_path"),
            qualname=cls.optional_string(payload, "qualname"),
            authority_id=cls.optional_string(payload, "authority_id"),
        )

    @classmethod
    def from_authority(cls, authority: SemanticAuthority) -> "AuthorityClaim":
        return cls(
            claimed_symbol=authority.name,
            authority_kind=authority.kind.value,
            file_path=authority.location.file_path,
            qualname=authority.location.symbol,
            authority_id=authority.authority_id,
        )

    @property
    def searched_symbols(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                symbol
                for symbol in (
                    self.claimed_symbol,
                    self.qualname,
                    self.qualname.rsplit(".", maxsplit=1)[-1],
                )
                if symbol
            )
        )

    def matches_declared_claim(self, declared_claim: "AuthorityClaim") -> bool:
        return (
            declared_claim.claimed_symbol == self.claimed_symbol
            and self.compatible_authority_kind(declared_claim)
            and self.compatible_location(declared_claim)
        )

    def matches_authority(self, authority: SemanticAuthority) -> bool:
        return self.matches_authority_kind(authority) and self.matches_file_qualname(
            authority.location.file_path,
            authority.location.symbol,
        )

    def matches_authority_kind(self, authority: SemanticAuthority) -> bool:
        return not self.authority_kind or authority.kind.value == self.authority_kind

    def compatible_authority_kind(self, declared_claim: "AuthorityClaim") -> bool:
        return (
            not self.authority_kind
            or not declared_claim.authority_kind
            or declared_claim.authority_kind == self.authority_kind
        )

    def compatible_location(self, declared_claim: "AuthorityClaim") -> bool:
        return self.matches_file_qualname(
            declared_claim.file_path,
            declared_claim.qualname,
            allow_empty_candidate=True,
        )

    def matches_file_qualname(
        self,
        file_path: str,
        qualname: str,
        *,
        allow_empty_candidate: bool = False,
    ) -> bool:
        return (
            not self.file_path
            or (allow_empty_candidate and not file_path)
            or file_path == self.file_path
        ) and (
            not self.qualname
            or (allow_empty_candidate and not qualname)
            or qualname == self.qualname
        )

    @staticmethod
    def required_string(payload: Mapping[str, object], field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} is required")
        return value

    @staticmethod
    def optional_string(payload: Mapping[str, object], field_name: str) -> str:
        value = payload.get(field_name)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        return value


@dataclass(frozen=True, kw_only=True)
class AuthorityClaimCarrier(SemanticRecord):
    """Shared carrier for records that own one authority claim."""

    authority_claim: AuthorityClaim


@dataclass(frozen=True)
class AuthorityProofEdge(SemanticRecord):
    """Concrete graph/source edge proving an authority claim."""

    edge_kind: AuthorityProofEdgeKind
    authority_id: str
    authority_kind: str
    file_path: str
    line: int
    symbol: str
    detail: str = ""

    @classmethod
    def from_authority(
        cls,
        authority: SemanticAuthority,
        edge_kind: AuthorityProofEdgeKind,
        *,
        detail: str = "",
    ) -> "AuthorityProofEdge":
        return cls(
            edge_kind=edge_kind,
            authority_id=authority.authority_id,
            authority_kind=authority.kind.value,
            file_path=authority.location.file_path,
            line=authority.location.line,
            symbol=authority.location.symbol,
            detail=detail,
        )


@dataclass(frozen=True)
class AuthorityDiscoveryRequired(SemanticRecord):
    """First-class unknown result when an authority claim cannot be proved."""

    claimed_symbol: str
    searched_symbols: tuple[str, ...]
    candidate_count: int
    reason: str


@dataclass(frozen=True)
class AuthorityClaimResolution(SemanticRecord):
    """Resolved, ambiguous, declared, or unresolved authority claim."""

    claim: AuthorityClaim
    status: AuthorityClaimStatus
    proof_edges: tuple[AuthorityProofEdge, ...] = ()
    discovery_required: AuthorityDiscoveryRequired | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status is AuthorityClaimStatus.RESOLVED

    @property
    def is_declared(self) -> bool:
        return self.status is AuthorityClaimStatus.DECLARED

    @property
    def is_actionable(self) -> bool:
        return self.is_resolved or self.is_declared

    @classmethod
    def declared(
        cls,
        claim: AuthorityClaim,
        *,
        detail: str = "recipe declares this authority boundary",
    ) -> "AuthorityClaimResolution":
        proof = AuthorityProofEdge(
            edge_kind=AuthorityProofEdgeKind.EXPLICIT_DECLARATION,
            authority_id=claim.authority_id or claim.claimed_symbol,
            authority_kind=claim.authority_kind,
            file_path=claim.file_path,
            line=0,
            symbol=claim.qualname or claim.claimed_symbol,
            detail=detail,
        )
        return cls(
            claim=claim,
            status=AuthorityClaimStatus.DECLARED,
            proof_edges=(proof,),
        )

    @classmethod
    def unresolved(
        cls,
        claim: AuthorityClaim,
        *,
        searched_symbols: tuple[str, ...],
        candidate_count: int = 0,
        reason: str = "no source-backed authority matched the claim",
    ) -> "AuthorityClaimResolution":
        return cls(
            claim=claim,
            status=AuthorityClaimStatus.UNRESOLVED,
            discovery_required=AuthorityDiscoveryRequired(
                claimed_symbol=claim.claimed_symbol,
                searched_symbols=searched_symbols,
                candidate_count=candidate_count,
                reason=reason,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class AuthorityClaimResolver:
    """Resolve authority claims against a semantic-descent graph catalog."""

    graph: "SemanticDescentGraph"

    def resolve(self, claim: AuthorityClaim) -> AuthorityClaimResolution:
        candidates = self._candidate_authorities(claim)
        searched_symbols = claim.searched_symbols
        if not candidates:
            return AuthorityClaimResolution.unresolved(
                claim,
                searched_symbols=searched_symbols,
            )
        if len(candidates) > 1:
            return AuthorityClaimResolution(
                claim=claim,
                status=AuthorityClaimStatus.AMBIGUOUS,
                proof_edges=tuple(
                    AuthorityProofEdge.from_authority(
                        authority,
                        AuthorityProofEdgeKind.SEMANTIC_DESCENT_GRAPH,
                        detail="multiple graph authorities match this claim",
                    )
                    for authority in candidates
                ),
                discovery_required=AuthorityDiscoveryRequired(
                    claimed_symbol=claim.claimed_symbol,
                    searched_symbols=searched_symbols,
                    candidate_count=len(candidates),
                    reason="multiple source-backed authorities match the claim",
                ),
            )
        authority = candidates[0]
        return AuthorityClaimResolution(
            claim=claim,
            status=AuthorityClaimStatus.RESOLVED,
            proof_edges=(
                AuthorityProofEdge.from_authority(
                    authority,
                    AuthorityProofEdgeKind.SEMANTIC_DESCENT_GRAPH,
                    detail="claim matched semantic-descent authority catalog",
                ),
            ),
        )

    def _candidate_authorities(
        self,
        claim: AuthorityClaim,
    ) -> tuple[SemanticAuthority, ...]:
        if claim.authority_id:
            authority = self.graph.authority_catalog.by_id.get(claim.authority_id)
            if authority is None:
                return ()
            return self._filter_authorities((authority,), claim)
        authority_ids = tuple(
            dict.fromkeys(
                authority_id
                for name in claim.searched_symbols
                for authority_id in self.graph.authority_name_index.by_name.get(
                    name, ()
                )
            )
        )
        return self._filter_authorities(
            tuple(
                self.graph.authority_catalog.authority(item) for item in authority_ids
            ),
            claim,
        )

    def _filter_authorities(
        self,
        authorities: tuple[SemanticAuthority, ...],
        claim: AuthorityClaim,
    ) -> tuple[SemanticAuthority, ...]:
        return tuple(
            authority for authority in authorities if claim.matches_authority(authority)
        )


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
    call_target_parts: tuple[str, ...] = ()

    def queries_authority(self, authority: SemanticAuthority) -> bool:
        return authority.name in self.call_target_parts[:-1]


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
    value_class_reference_parts: tuple[tuple[str, ...], ...] = ()

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
    projection_constructions: tuple[PresentationAuthorityConstruction, ...] = ()
    key_value_pairs: tuple[PresentationKeyValuePair, ...] = ()
    class_symbols: tuple[str, ...] = ()
    class_reference_parts: tuple[tuple[str, ...], ...] = ()

    @cached_property
    def normalized_tokens(self) -> tuple[str, ...]:
        return sorted_tuple({token.value for token in self.tokens})

    @cached_property
    def owner(self) -> ProjectionOwnerSymbol:
        return ProjectionOwnerSymbol(self.owner_symbol)


@dataclass(frozen=True)
class SemanticClassSupplement:
    """AST-free class facts shared by every semantic authority source."""

    class_symbol: str
    constant_assignments: tuple[tuple[str, int, str | None], ...]
    annotated_fields: tuple[tuple[str, int], ...]
    declared_type_names: tuple[str, ...]
    constructed_type_names: tuple[str, ...]
    is_dataclass: bool
    autoregister_authority_shape: bool


@dataclass(frozen=True)
class CompactSemanticModuleProjection(CompactModuleIdentity):
    """One module's deferred semantic projections and class supplements."""

    projections: tuple[PresentationProjection, ...]
    class_supplements: tuple[SemanticClassSupplement, ...]


@dataclass(frozen=True)
class CompactSemanticProjectionDemand:
    """Focused-scan view that retains authorities but no context reports."""

    include_presentations: bool


def _semantic_report_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactSemanticProjectionDemand:
    del target_items, config
    return CompactSemanticProjectionDemand(include_presentations=False)


def _cached_semantic_demand_projection(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, CompactSemanticProjectionDemand):
        raise TypeError("semantic projection demand has the wrong authority type")
    return tuple(
        replace(
            item,
            projections=(item.projections if demand.include_presentations else ()),
        )
        for item in items
        if isinstance(item, CompactSemanticModuleProjection)
    )


def _class_reference_parts(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Subscript):
        return _class_reference_parts(node.value)
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _class_reference_parts(node.value)
        return None if parent is None else (*parent, node.attr)
    return None


@dataclass(frozen=True)
class DeferredModuleClassReferenceCollector:
    """Collect resolvable reference parts before the global class index exists."""

    parsed_module: ParsedModule

    @cached_property
    def constructor_assignment_reference_parts(
        self,
    ) -> dict[str, tuple[str, ...]]:
        assignments: dict[str, tuple[str, ...]] = {}
        for statement in self.parsed_module.module.body:
            pair = SingleAssignmentAndValueNameProjection(statement).pair
            if pair is None:
                continue
            target_name, value = pair
            if not isinstance(value, ast.Call):
                continue
            reference_parts = _class_reference_parts(value.func)
            if reference_parts is not None:
                assignments[target_name] = reference_parts
        return assignments

    def reference_parts_for_node(
        self,
        node: ast.AST,
    ) -> tuple[tuple[str, ...], ...]:
        collector = _DeferredClassReferencePartsVisitor(
            self.constructor_assignment_reference_parts
        )
        collector.visit(node)
        return sorted_tuple(collector.reference_parts)


class _DeferredClassReferencePartsVisitor(ast.NodeVisitor):
    def __init__(
        self,
        constructor_assignment_reference_parts: dict[str, tuple[str, ...]],
    ) -> None:
        self.constructor_assignment_reference_parts = (
            constructor_assignment_reference_parts
        )
        self.reference_parts: set[tuple[str, ...]] = set()

    def visit_Call(self, node: ast.Call) -> None:
        self._add(_class_reference_parts(node.func))
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add(_class_reference_parts(node))

    def visit_Name(self, node: ast.Name) -> None:
        self._add(
            self.constructor_assignment_reference_parts.get(
                node.id,
                (node.id,),
            )
        )

    def _add(self, reference_parts: tuple[str, ...] | None) -> None:
        if reference_parts is not None:
            self.reference_parts.add(reference_parts)


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
class SemanticAuthorityMatch:
    """Fact/token overlap carried by one authority-projection relation."""

    fact_refs: tuple[SemanticFactReference, ...]
    tokens: tuple[str, ...]
    coverage_ratio: float

    @classmethod
    def from_facts(cls, facts: tuple[SemanticFact, ...]) -> "SemanticAuthorityMatch":
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
        fact_references_by_id: dict[str, SemanticFactReference] | None = None,
    ) -> "SemanticAuthorityMatch | None":
        if len(facts) < 2:
            return None
        active_fact_references = (
            {} if fact_references_by_id is None else fact_references_by_id
        )
        fact_refs: list[SemanticFactReference] = []
        for fact in facts:
            if fact.fact_id not in matches_by_fact_id:
                continue
            fact_ref = active_fact_references.get(fact.fact_id)
            if fact_ref is None:
                fact_ref = SemanticFactReference(fact.authority_id, fact.fact_id)
                active_fact_references[fact.fact_id] = fact_ref
            fact_refs.append(fact_ref)
        if len(fact_refs) < 2:
            return None
        tokens: set[str] = set()
        for fact_ref in fact_refs:
            tokens.update(matches_by_fact_id[fact_ref.fact_id])
        coverage_ratio = len(fact_refs) / len(facts)
        if coverage_ratio < 0.5 and len(fact_refs) < 3:
            return None
        return cls(
            fact_refs=tuple(fact_refs),
            tokens=sorted_tuple(tokens),
            coverage_ratio=coverage_ratio,
        )

    @property
    def fact_count(self) -> int:
        return len(self.fact_refs)


@dataclass(frozen=True)
class SemanticAuthorityProjectionRelation(
    SemanticAuthorityProjectionReference,
    ABC,
):
    """Nominal classification of one authority-projection relationship."""

    match: SemanticAuthorityMatch

    @property
    def identity(self) -> tuple[str, str]:
        """Return the unique authority-projection endpoint identity."""
        return (self.authority_id, self.projection_id)

    @abstractmethod
    def certificate(
        self,
        graph_space: "SemanticDescentGraphSpace",
    ) -> "SemanticDescentCertificate":
        """Build the certificate owned by this relation leaf."""

    @abstractmethod
    def missing_descent_relations(self) -> tuple["MirrorEdge", ...]:
        """Project this relation into the detector's failure-only view."""

    @abstractmethod
    def rebase_proof_paths(
        self,
        source_roots: tuple[str, ...],
        target_roots: tuple[str, ...],
    ) -> "SemanticAuthorityProjectionRelation":
        """Rebase source-bearing proof paths owned by this relation."""

    @abstractmethod
    def proof_file_paths(self) -> tuple[str, ...]:
        """Return source paths carried only by positive proof edges."""


@dataclass(frozen=True)
class MirrorEdge(SemanticAuthorityProjectionRelation):
    """Projection that repeats authority facts without a derivation path."""

    missing_derivation_path: str = ""

    def certificate(
        self,
        graph_space: "SemanticDescentGraphSpace",
    ) -> "DescentCertificate":
        if self.missing_derivation_path:
            return DescentCertificate.mirrored_without_descent(
                self,
                self.missing_derivation_path,
            )
        authority = graph_space.authority_catalog.authority_for_edge(self)
        projection = graph_space.projection_catalog.projection_for_edge(self)
        return DescentCertificate.from_mirror_candidate(
            self,
            SemanticMirrorEdgeCandidate(
                projection=projection,
                authority=authority,
                facts=(),
                match=self.match,
            ),
        )

    def missing_descent_relations(self) -> tuple["MirrorEdge", ...]:
        return (self,)

    def rebase_proof_paths(
        self,
        source_roots: tuple[str, ...],
        target_roots: tuple[str, ...],
    ) -> "MirrorEdge":
        del source_roots, target_roots
        return self

    def proof_file_paths(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class SemanticDerivationEdge(SemanticAuthorityProjectionRelation):
    """Positive source-backed derivation from a projection to its authority."""

    proof_edges: tuple[AuthorityProofEdge, ...]

    def certificate(
        self,
        graph_space: "SemanticDescentGraphSpace",
    ) -> "SemanticDerivationCertificate":
        del graph_space
        return SemanticDerivationCertificate(self)

    def missing_descent_relations(self) -> tuple[MirrorEdge, ...]:
        return ()

    def rebase_proof_paths(
        self,
        source_roots: tuple[str, ...],
        target_roots: tuple[str, ...],
    ) -> "SemanticDerivationEdge":
        return replace(
            self,
            proof_edges=tuple(
                replace(
                    proof,
                    file_path=rebase_checkout_path(
                        proof.file_path,
                        source_roots,
                        target_roots,
                    ),
                )
                for proof in self.proof_edges
            ),
        )

    def proof_file_paths(self) -> tuple[str, ...]:
        return tuple(proof.file_path for proof in self.proof_edges)


@dataclass(frozen=True)
class SemanticAuthorityProjectionResolution:
    """Complete typed classification of authority-projection candidates."""

    relations: tuple[SemanticAuthorityProjectionRelation, ...] = ()

    @classmethod
    def suppressed(cls) -> "SemanticAuthorityProjectionResolution":
        return cls()

    @classmethod
    def mirrored(
        cls,
        candidate: "SemanticMirrorEdgeCandidate",
    ) -> "SemanticAuthorityProjectionResolution":
        return cls(
            relations=(
                MirrorEdge(
                    authority_id=candidate.authority.authority_id,
                    projection_id=candidate.projection.projection_id,
                    match=candidate.match,
                ),
            ),
        )

    @classmethod
    def derived(
        cls,
        candidate: "SemanticMirrorEdgeCandidate",
        proof_edges: tuple[AuthorityProofEdge, ...],
    ) -> "SemanticAuthorityProjectionResolution":
        if not proof_edges:
            raise ValueError("semantic derivation requires at least one proof edge")
        return cls(
            relations=(
                SemanticDerivationEdge(
                    authority_id=candidate.authority.authority_id,
                    projection_id=candidate.projection.projection_id,
                    match=candidate.match,
                    proof_edges=proof_edges,
                ),
            ),
        )

    @classmethod
    def combine(
        cls,
        resolutions: Iterable["SemanticAuthorityProjectionResolution"],
    ) -> "SemanticAuthorityProjectionResolution":
        relation_index = UniqueIdentityIndexAuthority[
            tuple[str, str],
            SemanticAuthorityProjectionRelation,
            SemanticAuthorityProjectionRelation,
        ]()
        for resolution in resolutions:
            for relation in resolution.relations:
                relation_index.add(relation.identity, relation, relation)
        return cls(
            relations=sorted_tuple(
                relation_index.values_by_handle().values(),
                key=lambda item: (
                    -item.match.fact_count,
                    item.authority_id,
                    item.projection_id,
                ),
            ),
        )


@dataclass(frozen=True)
class SemanticMirrorEdgeCandidate:
    """Resolved projection/fact overlap before policy admissibility filtering."""

    projection: PresentationProjection
    authority: SemanticAuthority
    facts: tuple[SemanticFact, ...]
    match: SemanticAuthorityMatch

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
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        del context, candidate
        return True

    def classify(
        self,
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> SemanticAuthorityProjectionResolution:
        if not self.edge_is_admissible(context, candidate):
            return SemanticAuthorityProjectionResolution.suppressed()
        return self.classify_admissible(context, candidate)

    def classify_admissible(
        self,
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> SemanticAuthorityProjectionResolution:
        del context
        return SemanticAuthorityProjectionResolution.mirrored(candidate)


class ClassFamilyLikeMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Shared policy for class-family authorities and AutoRegister families."""

    foreign_qualified_attribute_token_reference_admitted = True

    @staticmethod
    def call_projection_is_inadmissible(
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        matched_tokens = frozenset(candidate.match.tokens)
        matched_token_roles = frozenset(
            token.role
            for token in candidate.projection.tokens
            if token.value in matched_tokens
        )
        has_matched_class_reference = (
            context.projection_semantics.has_matched_class_reference(
                candidate.projection,
                candidate.matched_facts,
            )
        )
        return (
            candidate.projection.kind is PresentationProjectionKind.CALL_LITERAL
            and (
                (
                    context.dataclass_descent.projection_constructs_any_dataclass_authority(
                        candidate.projection,
                    )
                    and not has_matched_class_reference
                )
                or (
                    candidate.match.coverage_ratio < 1.0
                    and matched_token_roles
                    == frozenset((PresentationTokenRole.CALL_TARGET,))
                    and has_matched_class_reference
                )
            )
        )

    def edge_is_admissible(
        self,
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> bool:
        if self.call_projection_is_inadmissible(context, candidate):
            return False
        if (
            context.fact_specificity.matched_facts_are_reused_roles(
                candidate.matched_facts
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
        context: "SemanticAuthorityProjectionResolutionContext",
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
        if candidate.projection.kind is PresentationProjectionKind.CALL_LITERAL and (
            context.dataclass_descent.projection_constructs_distinct_dataclass_authority(
                candidate.projection,
                candidate.authority,
            )
        ):
            return False
        if (
            context.dataclass_descent.projection_materializes_any_dataclass_authority(
                candidate.projection,
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
                or (
                    candidate.projection.kind is PresentationProjectionKind.CALL_LITERAL
                    and context.dataclass_descent.projection_constructs_distinct_dataclass_authority(
                        candidate.projection,
                        candidate.authority,
                    )
                )
                or (
                    candidate.projection.kind
                    is not PresentationProjectionKind.CALL_LITERAL
                    and context.dataclass_descent.projection_constructs_name_unrelated_dataclass_authority(
                        candidate.projection,
                        candidate.authority,
                    )
                )
            )
        )

    def classify_admissible(
        self,
        context: "SemanticAuthorityProjectionResolutionContext",
        candidate: SemanticMirrorEdgeCandidate,
    ) -> SemanticAuthorityProjectionResolution:
        proof_edges = context.dataclass_descent.derivation_proof_edges(
            candidate.projection,
            candidate.authority,
            candidate.matched_facts,
        )
        if proof_edges:
            return SemanticAuthorityProjectionResolution.derived(
                candidate,
                proof_edges,
            )
        if context.dataclass_descent.projection_owner_constructs_dataclass_authority(
            candidate.projection,
            candidate.authority,
            candidate.matched_facts,
        ) or context.dataclass_descent.projection_shares_dataclass_base_with_authority(
            candidate.projection,
            candidate.authority,
        ):
            return SemanticAuthorityProjectionResolution.suppressed()
        return SemanticAuthorityProjectionResolution.mirrored(candidate)


class EnumMirrorPolicy(SemanticAuthorityMirrorPolicy):
    """Mirror policy for enum authorities."""

    authority_kind = SemanticAuthorityKind.ENUM
    authority_qualified_token_reference_admitted = False

    def edge_is_admissible(
        self,
        context: "SemanticAuthorityProjectionResolutionContext",
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


FactsByAuthorityId: TypeAlias = dict[str, tuple[SemanticFact, ...]]
AuthorityIdsByName: TypeAlias = dict[str, tuple[str, ...]]
AuthorityIdsByFactName: TypeAlias = dict[tuple[SemanticFactKind, str], frozenset[str]]
FactRefsByToken: TypeAlias = dict[str, tuple[SemanticFact, ...]]
FactMatchesByAuthority: TypeAlias = dict[str, dict[str, set[str]]]
ConstructionAuthorityCacheKey: TypeAlias = tuple[str, str]
CompactAuthorityIdsByNameValue: TypeAlias = str | tuple[str, ...]
_EMPTY_STRING_FROZENSET: frozenset[str] = frozenset()


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

    def facts_for_edge(
        self,
        edge: SemanticAuthorityProjectionRelation,
    ) -> tuple[SemanticFact, ...]:
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

    @cached_property
    def reused_fact_names(self) -> frozenset[tuple[SemanticFactKind, str]]:
        authority_id_by_fact_name: dict[tuple[SemanticFactKind, str], str] = {}
        reused_fact_names: set[tuple[SemanticFactKind, str]] = set()
        for fact in self.facts:
            key = (fact.kind, fact.name)
            authority_id = authority_id_by_fact_name.get(key)
            if authority_id is None:
                authority_id_by_fact_name[key] = fact.authority_id
            elif authority_id != fact.authority_id:
                reused_fact_names.add(key)
        return frozenset(reused_fact_names)

    def fact_is_reused_role(self, fact: SemanticFact) -> bool:
        return (fact.kind, fact.name) in self.reused_fact_names

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

    def authority_for_edge(
        self,
        edge: SemanticAuthorityProjectionRelation,
    ) -> SemanticAuthority:
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

    def projection_for_edge(
        self,
        edge: SemanticAuthorityProjectionRelation,
    ) -> PresentationProjection:
        return self.projection(edge.projection_id)


@dataclass(frozen=True)
class SemanticFactTokenIndex:
    """Fact references grouped by normalized presentation token."""

    facts: tuple[SemanticFact, ...]

    @cached_property
    def by_token(self) -> FactRefsByToken:
        refs_by_token: dict[str, list[SemanticFact]] = {}
        for fact in sorted(
            self.facts,
            key=lambda item: (item.authority_id, item.fact_id),
        ):
            for alias in fact.normalized_aliases:
                refs_by_token.setdefault(alias, []).append(fact)
        ordered_refs: FactRefsByToken = {}
        for token in sorted(refs_by_token):
            ordered_refs[token] = tuple(refs_by_token.pop(token))
        return ordered_refs

    def contains_token(self, token: str) -> bool:
        return token in self.by_token

    def refs_for_token(self, token: str) -> tuple[SemanticFact, ...]:
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
class SemanticDescentCertificate(SemanticRecord, ABC):
    """Nominal certificate emitted by one authority-projection relation leaf."""

    edge: SemanticAuthorityProjectionRelation
    status: ClassVar[DescentStatus]

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class DescentCertificate(SemanticDescentCertificate):
    """Certificate for a mirror relation that lacks semantic descent."""

    status = DescentStatus.MIRRORED_WITHOUT_DESCENT
    edge: MirrorEdge
    missing_derivation_path: str

    @classmethod
    def mirrored_without_descent(
        cls,
        edge: MirrorEdge,
        path_description: str,
    ) -> "DescentCertificate":
        return cls(edge, path_description)

    @classmethod
    def from_mirror_candidate(
        cls,
        edge: MirrorEdge,
        candidate: SemanticMirrorEdgeCandidate,
    ) -> "DescentCertificate":
        return cls.mirrored_without_descent(edge, candidate.missing_derivation_path)


@dataclass(frozen=True)
class SemanticDerivationCertificate(SemanticDescentCertificate):
    """Positive certificate preserving the source edges that prove descent."""

    status = DescentStatus.DESCENDS_TO_AUTHORITY
    edge: SemanticDerivationEdge

    @property
    def proof_edges(self) -> tuple[AuthorityProofEdge, ...]:
        return self.edge.proof_edges


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
    """Repository graph with one nominal authority-projection relation set."""

    relations: tuple[SemanticAuthorityProjectionRelation, ...]
    class_index: ClassFamilyIndex | None = None

    @property
    def certificates(self) -> tuple[SemanticDescentCertificate, ...]:
        return tuple(relation.certificate(self) for relation in self.relations)

    @property
    def missing_descent_relations(self) -> tuple[MirrorEdge, ...]:
        return tuple(
            missing_relation
            for relation in self.relations
            for missing_relation in relation.missing_descent_relations()
        )

    @property
    def missing_descent_certificates(self) -> tuple[DescentCertificate, ...]:
        return tuple(
            relation.certificate(self) for relation in self.missing_descent_relations
        )

    @classmethod
    def from_resolution(
        cls,
        graph_space: SemanticDescentGraphSpace,
        resolution: SemanticAuthorityProjectionResolution,
        *,
        class_index: ClassFamilyIndex | None = None,
    ) -> "SemanticDescentGraph":
        return cls(
            authorities=graph_space.authorities,
            facts=graph_space.facts,
            projections=graph_space.projections,
            relations=resolution.relations,
            class_index=class_index,
        )

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
        authority_catalog = SemanticAuthorityBuilder.from_class_index(
            class_index
        ).build()
        projections = self.merged_projections(class_index)
        graph_space = SemanticDescentGraphSpace(
            authority_catalog.authorities,
            authority_catalog.facts,
            projections,
        )
        resolution = self.merged_resolution(graph_space, class_index)
        return SemanticDescentGraph.from_resolution(
            graph_space,
            resolution,
            class_index=class_index,
        )

    def merged_resolution(
        self,
        graph_space: SemanticDescentGraphSpace,
        class_index: ClassFamilyIndex,
    ) -> SemanticAuthorityProjectionResolution:
        return SemanticMirrorResolver(
            graph_space.authorities,
            graph_space.facts,
            graph_space.projections,
            class_index,
        ).resolve()

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
    relation_count: int
    missing_descent_count: int
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
            relation_count=len(graph.relations),
            missing_descent_count=len(graph.missing_descent_relations),
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
                    graph.missing_descent_certificates,
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
            not repository_report.missing_descent_count
            and finding_backed_report is not None
            and finding_backed_report.missing_descent_count
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
                finding,
            )
            authorities.append(authority)
            facts.extend(finding_facts)
            projections.append(projection)
            edges.append(edge)
        graph_space = SemanticDescentGraphSpace(
            authorities=sorted_tuple(authorities, key=lambda item: item.authority_id),
            facts=sorted_tuple(facts, key=lambda item: item.fact_id),
            projections=sorted_tuple(projections, key=lambda item: item.projection_id),
        )
        graph = SemanticDescentGraph.from_resolution(
            graph_space,
            SemanticAuthorityProjectionResolution(
                relations=sorted_tuple(
                    edges,
                    key=lambda item: item.projection_id,
                ),
            ),
        )
        return graph


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
        finding: RefactorFinding,
    ) -> MirrorEdge:
        return MirrorEdge(
            authority_id=authority.authority_id,
            projection_id=projection.projection_id,
            match=SemanticAuthorityMatch.from_facts(facts),
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
    identity: SemanticDescentGraphCacheIdentity | None = None


def _rebase_source_location(
    location: SourceLocation,
    source_roots: tuple[str, ...],
    target_roots: tuple[str, ...],
) -> SourceLocation:
    if not location.file_path:
        return location
    return replace(
        location,
        file_path=rebase_checkout_path(
            location.file_path,
            source_roots,
            target_roots,
        ),
    )


def _rebase_class_family_index(
    class_index: ClassFamilyIndex | None,
    source_roots: tuple[str, ...],
    target_roots: tuple[str, ...],
) -> ClassFamilyIndex | None:
    if class_index is None:
        return None
    classes_by_symbol = {
        symbol: replace(
            indexed_class,
            file_path=rebase_checkout_path(
                indexed_class.file_path,
                source_roots,
                target_roots,
            ),
        )
        for symbol, indexed_class in class_index.classes_by_symbol.items()
    }
    return replace(
        class_index,
        classes_by_symbol=classes_by_symbol,
        symbols_by_file_and_qualname={
            (
                rebase_checkout_path(
                    file_path,
                    source_roots,
                    target_roots,
                ),
                qualname,
            ): symbol
            for (file_path, qualname), symbol in (
                class_index.symbols_by_file_and_qualname.items()
            )
        },
    )


def rebase_semantic_descent_graph(
    graph: SemanticDescentGraph,
    source_roots: tuple[str, ...],
    target_roots: tuple[str, ...],
) -> SemanticDescentGraph:
    """Validate and rebase every source-bearing graph record to target roots."""

    if not source_roots or not target_roots:
        raise CacheCheckoutPathError(
            "semantic graph cache has no admitted presentation roots"
        )
    if source_roots == target_roots:
        # Exact-cache publication and lookup do not relocate the graph.  Keep
        # path admission validation, but perform it once per distinct path and
        # retain the already immutable graph instead of rebuilding tens of
        # thousands of duplicate source-bearing records.
        file_paths = {
            location.file_path
            for location in (
                *(authority.location for authority in graph.authorities),
                *(fact.location for fact in graph.facts),
                *(projection.location for projection in graph.projections),
            )
            if location.file_path
        }
        file_paths.update(
            file_path
            for relation in graph.relations
            for file_path in relation.proof_file_paths()
            if file_path
        )
        if graph.class_index is not None:
            file_paths.update(
                indexed_class.file_path
                for indexed_class in graph.class_index.classes_by_symbol.values()
                if indexed_class.file_path
            )
            file_paths.update(
                file_path
                for file_path, _qualname in (
                    graph.class_index.symbols_by_file_and_qualname
                )
                if file_path
            )
        for file_path in file_paths:
            checkout_relative_path(file_path, source_roots)
        return graph
    return replace(
        graph,
        authorities=tuple(
            replace(
                authority,
                location=_rebase_source_location(
                    authority.location,
                    source_roots,
                    target_roots,
                ),
            )
            for authority in graph.authorities
        ),
        facts=tuple(
            replace(
                fact,
                location=_rebase_source_location(
                    fact.location,
                    source_roots,
                    target_roots,
                ),
            )
            for fact in graph.facts
        ),
        projections=tuple(
            replace(
                projection,
                location=_rebase_source_location(
                    projection.location,
                    source_roots,
                    target_roots,
                ),
            )
            for projection in graph.projections
        ),
        relations=tuple(
            relation.rebase_proof_paths(source_roots, target_roots)
            for relation in graph.relations
        ),
        class_index=_rebase_class_family_index(
            graph.class_index,
            source_roots,
            target_roots,
        ),
    )


@dataclass(frozen=True)
class SemanticDescentGraphCache:
    """Persistent graph cache for repo-wide semantic descent context."""

    storage_root: Path | None
    max_exact_entry_count: int = 4

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
        scan_deadline_checkpoint("semantic_descent_cache_load")
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError:
            return None
        except (
            OSError,
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            TypeError,
            ValueError,
        ):
            # Cache entries are derived artifacts.  A process may have been
            # interrupted while publishing an entry created by an older,
            # non-atomic writer, or a Python/schema upgrade may make the
            # pickle unreadable.  Neither condition is a source-analysis
            # failure: verify it as a miss and rebuild from authoritative
            # source instead.
            return None
        if not isinstance(payload, dict):
            return None
        scan_deadline_checkpoint("semantic_descent_cache_validation")
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
        stored_identity = payload.get("identity")
        if not isinstance(stored_identity, SemanticDescentGraphCacheIdentity):
            return SemanticDescentGraphCacheMiss()
        if stored_identity != identity:
            return SemanticDescentGraphCacheMiss()
        graph = payload.get("graph")
        if not isinstance(graph, SemanticDescentGraph):
            return SemanticDescentGraphCacheMiss()
        try:
            graph = rebase_semantic_descent_graph(
                graph,
                stored_identity.presentation_roots,
                identity.presentation_roots,
            )
        except CacheCheckoutPathError:
            return SemanticDescentGraphCacheMiss()
        return SemanticDescentGraphCacheHit(graph, identity)

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
        identity = payload.get("identity")
        if not isinstance(identity, SemanticDescentGraphCacheIdentity):
            return SemanticDescentGraphCacheMiss()
        if (
            SemanticDescentGraphCacheFamilyIdentity.from_identity(identity)
            != family_identity
        ):
            return SemanticDescentGraphCacheMiss()
        target_identity = identity.relocated_to(family_identity.presentation_roots)
        identity_lookup = self.load(target_identity)
        if identity_lookup.graph is not None:
            return identity_lookup
        # Read compatibility for graph-bearing latest entries produced before
        # latest pointers became lightweight.  New stores always publish the
        # exact entry first and the latest identity pointer second.
        graph = payload.get("graph")
        if not isinstance(graph, SemanticDescentGraph):
            return SemanticDescentGraphCacheMiss()
        try:
            graph = rebase_semantic_descent_graph(
                graph,
                identity.presentation_roots,
                target_identity.presentation_roots,
            )
        except CacheCheckoutPathError:
            return SemanticDescentGraphCacheMiss()
        return SemanticDescentGraphCacheHit(graph, target_identity)

    @staticmethod
    def _store_payload_atomic(cache_path: Path, payload: object) -> None:
        """Publish one pickle only after its complete bytes are durable."""

        file_descriptor, temporary_path_text = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_path_text)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, cache_path)
            directory_descriptor = os.open(cache_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def store(
        self,
        identity: SemanticDescentGraphCacheIdentity,
        graph: SemanticDescentGraph,
    ) -> None:
        if self.storage_root is None:
            return
        try:
            validated_graph = rebase_semantic_descent_graph(
                graph,
                identity.presentation_roots,
                identity.presentation_roots,
            )
            self.storage_root.mkdir(parents=True, exist_ok=True)
            self._store_payload_atomic(
                self._entry_path(identity),
                {"identity": identity, "graph": validated_graph},
            )
            family_identity = SemanticDescentGraphCacheFamilyIdentity.from_identity(
                identity
            )
            self._store_payload_atomic(
                self._latest_path(family_identity),
                {
                    "family_identity": family_identity,
                    "identity": identity,
                },
            )
            self._prune_exact_entries(self._entry_path(identity))
        except (OSError, CacheCheckoutPathError):
            return

    def _prune_exact_entries(self, protected_path: Path) -> None:
        if self.storage_root is None:
            return
        exact_paths = [
            path
            for path in self.storage_root.glob("*.pickle")
            if not path.name.startswith("latest-")
        ]
        exact_paths.sort(
            key=lambda path: (
                path == protected_path,
                path.stat().st_mtime,
            ),
            reverse=True,
        )
        for stale_path in exact_paths[max(1, self.max_exact_entry_count) :]:
            try:
                stale_path.unlink()
            except OSError:
                continue

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
    identity = SemanticDescentGraphCacheIdentity.from_path_identities(
        identities,
        roots=roots,
    )
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
        identities,
        roots=roots,
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
    scan_deadline_checkpoint("semantic_descent_class_index")
    class_index = build_class_family_index(list(modules))
    scan_deadline_checkpoint("semantic_descent_authorities")
    authority_catalog = SemanticAuthorityBuilder.from_class_index(class_index).build()
    scan_deadline_checkpoint("semantic_descent_projections")
    projections = SemanticProjectionCollector(tuple(modules), class_index).collect()
    scan_deadline_checkpoint("semantic_descent_mirror_edges")
    resolution = SemanticMirrorResolver(
        authority_catalog.authorities,
        authority_catalog.facts,
        projections,
        class_index,
    ).resolve()
    graph_space = SemanticDescentGraphSpace(
        authority_catalog.authorities,
        authority_catalog.facts,
        projections,
    )
    scan_deadline_checkpoint("semantic_descent_certificates")
    return SemanticDescentGraph.from_resolution(
        graph_space,
        resolution,
        class_index=class_index,
    )


@dataclass(frozen=True)
class SemanticAuthorityInventory:
    """Authorities and their facts as one construction result."""

    authorities: tuple[SemanticAuthority, ...]
    facts: tuple[SemanticFact, ...]


@dataclass(frozen=True)
class SemanticAuthorityBuilder:
    """Build authorities once from source-form-independent declarations."""

    declarations: tuple["SemanticAuthorityDeclaration", ...]

    @classmethod
    def from_class_index(
        cls,
        class_index: ClassFamilyIndex,
    ) -> "SemanticAuthorityBuilder":
        context = AstSemanticAuthorityBuildContext(class_index)
        return cls(
            tuple(
                context.declaration_for(indexed_class)
                for indexed_class in sorted_tuple(
                    class_index.classes_by_symbol.values(),
                    key=lambda item: item.symbol,
                )
            )
        )

    @classmethod
    def from_compact_class_index(
        cls,
        class_index: CompactClassFamilyIndex,
        supplements_by_symbol: Mapping[str, SemanticClassSupplement],
    ) -> "SemanticAuthorityBuilder":
        context = CompactSemanticAuthorityBuildContext(
            class_index,
            supplements_by_symbol,
        )
        return cls(
            tuple(
                context.declaration_for(indexed_class)
                for indexed_class in sorted_tuple(
                    class_index.classes_by_symbol.values(),
                    key=lambda item: item.symbol,
                )
            )
        )

    def build(self) -> SemanticAuthorityInventory:
        authorities: list[SemanticAuthority] = []
        facts: list[SemanticFact] = []
        for declaration in self.declarations:
            provider_result = SemanticAuthorityProvider().provide(declaration)
            if provider_result is None:
                continue
            indexed_class = declaration.indexed_class
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
        return SemanticAuthorityInventory(
            authorities=sorted_tuple(
                authorities,
                key=lambda item: item.authority_id,
            ),
            facts=sorted_tuple(facts, key=lambda item: item.fact_id),
        )


@dataclass(frozen=True)
class SemanticAuthorityProviderResult:
    """Authority kind and facts selected by one authority provider."""

    kind: SemanticAuthorityKind
    facts: tuple[SemanticFact, ...]


class SemanticAuthorityBuildContext(ABC, Generic[IndexedClassDeclarationT]):
    """Project one indexed source form into shared authority declarations."""

    class_index: SemanticClassFamilyIndex

    def declaration_for(
        self,
        indexed_class: IndexedClassDeclarationT,
    ) -> "SemanticAuthorityDeclaration":
        return SemanticAuthorityDeclaration(
            indexed_class=indexed_class,
            supplement=self.supplement_for(indexed_class),
            class_facts=self.class_facts_for(indexed_class.symbol),
        )

    def class_facts_for(
        self,
        authority_symbol: str,
    ) -> tuple[SemanticFact, ...]:
        descendant_symbols = self.class_index.descendant_symbols(authority_symbol)
        if len(descendant_symbols) < 2:
            return ()
        return tuple(
            SemanticFact.class_member(
                authority_symbol,
                descendant,
                self.aliases_for(descendant),
            )
            for descendant_symbol in descendant_symbols
            if (descendant := self.class_index.class_for(descendant_symbol)) is not None
        )

    @abstractmethod
    def supplement_for(
        self,
        indexed_class: IndexedClassDeclarationT,
    ) -> SemanticClassSupplement | None:
        raise NotImplementedError

    @abstractmethod
    def aliases_for(
        self,
        indexed_class: IndexedClassDeclarationT,
    ) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class AstSemanticAuthorityBuildContext(
    SemanticAuthorityBuildContext[IndexedClass]
):
    """Project AST-backed class declarations into provider input."""

    class_index: ClassFamilyIndex

    def supplement_for(
        self,
        indexed_class: IndexedClass,
    ) -> SemanticClassSupplement | None:
        return _semantic_class_supplement(
            indexed_class.symbol,
            indexed_class.node,
            constructed_type_names=(),
        )

    def aliases_for(self, indexed_class: IndexedClass) -> tuple[str, ...]:
        values: list[str] = []
        for name, value in AutoRegisterClassAuthority(
            indexed_class.node
        ).assignment_pairs:
            if name.startswith("__"):
                continue
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and PresentationTokenProjection.looks_like_semantic_literal(value.value)
            ):
                values.append(value.value)
        return sorted_tuple(values)


@dataclass(frozen=True)
class CompactSemanticAuthorityBuildContext(
    SemanticAuthorityBuildContext[CompactIndexedClass]
):
    """Project compact class declarations into the same provider input."""

    class_index: CompactClassFamilyIndex
    supplements_by_symbol: Mapping[str, SemanticClassSupplement]

    def supplement_for(
        self,
        indexed_class: CompactIndexedClass,
    ) -> SemanticClassSupplement | None:
        return self.supplements_by_symbol.get(indexed_class.symbol)

    def aliases_for(
        self,
        indexed_class: CompactIndexedClass,
    ) -> tuple[str, ...]:
        return tuple(
            value
            for name, value in indexed_class.direct_constant_string_assignments
            if not name.startswith("__")
            if PresentationTokenProjection.looks_like_semantic_literal(value)
        )


@dataclass(frozen=True)
class SemanticAuthorityDeclaration:
    """One source-form-independent class declaration for authority selection."""

    indexed_class: ClassDeclaration
    supplement: SemanticClassSupplement | None
    class_facts: tuple[SemanticFact, ...]

    @property
    def is_enum(self) -> bool:
        return any(
            base_name.rsplit(".", 1)[-1] in _ENUM_BASE_NAMES
            for base_name in self.indexed_class.declared_base_names
        )


class ClassFamilySemanticAuthorityProvider:
    """Terminal fallback for conventional and AutoRegister class families."""

    def provide(
        self,
        declaration: SemanticAuthorityDeclaration,
    ) -> SemanticAuthorityProviderResult | None:
        if not declaration.class_facts:
            return None
        supplement = declaration.supplement
        return SemanticAuthorityProviderResult(
            (
                SemanticAuthorityKind.AUTOREGISTER_FAMILY
                if supplement is not None
                and supplement.autoregister_authority_shape
                else SemanticAuthorityKind.CLASS_FAMILY
            ),
            declaration.class_facts,
        )


class DataclassSemanticAuthorityProvider(ClassFamilySemanticAuthorityProvider):
    """Prefer dataclass field schemas, then continue to class-family facts."""

    def provide(
        self,
        declaration: SemanticAuthorityDeclaration,
    ) -> SemanticAuthorityProviderResult | None:
        indexed_class = declaration.indexed_class
        supplement = declaration.supplement
        if supplement is not None and supplement.is_dataclass:
            facts = tuple(
                SemanticFact.dataclass_field(indexed_class, name, line)
                for name, line in supplement.annotated_fields
            )
            if len(facts) >= 2:
                return SemanticAuthorityProviderResult(
                    SemanticAuthorityKind.DATACLASS_SCHEMA,
                    facts,
                )
        return super().provide(declaration)


class SemanticAuthorityProvider(DataclassSemanticAuthorityProvider):
    """Complete MRO-owned authority selection: enum, dataclass, then family."""

    def provide(
        self,
        declaration: SemanticAuthorityDeclaration,
    ) -> SemanticAuthorityProviderResult | None:
        indexed_class = declaration.indexed_class
        supplement = declaration.supplement
        if declaration.is_enum and supplement is not None:
            facts = tuple(
                SemanticFact.enum_member(
                    indexed_class,
                    name,
                    line,
                    string_value,
                )
                for name, line, string_value in supplement.constant_assignments
            )
            if len(facts) >= 2:
                return SemanticAuthorityProviderResult(
                    SemanticAuthorityKind.ENUM,
                    facts,
                )
        return super().provide(declaration)


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


def _semantic_indexed_class_nodes(
    statements: list[ast.stmt],
    *,
    parent_qualname: str | None = None,
) -> tuple[tuple[str, ast.ClassDef], ...]:
    classes: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if not isinstance(statement, ast.ClassDef):
            continue
        qualname = (
            statement.name
            if parent_qualname is None
            else f"{parent_qualname}.{statement.name}"
        )
        classes.append((qualname, statement))
        classes.extend(
            _semantic_indexed_class_nodes(
                list(statement.body),
                parent_qualname=qualname,
            )
        )
    return tuple(classes)


def _semantic_class_supplement(
    class_symbol: str,
    node: ast.ClassDef,
    *,
    constructed_type_names: Iterable[str] | None = None,
) -> SemanticClassSupplement | None:
    authority = AutoRegisterClassAuthority(node)
    is_dataclass = any(
        AttributeChainAuthority.decorator_terminal_name(decorator) == "dataclass"
        for decorator in node.decorator_list
    )
    constant_assignments = tuple(
        (
            name,
            value.lineno,
            value.value if isinstance(value.value, str) else None,
        )
        for name, value in authority.assignment_pairs
        if isinstance(value, ast.Constant)
    )
    annotated_fields = tuple(
        (statement.target.id, statement.lineno)
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        if isinstance(statement.target, ast.Name)
    )
    declared_type_names = sorted_tuple(
        {
            terminal_name
            for _, value in authority.assignment_pairs
            if (terminal_name := authority.terminal_name(value)) is not None
        }
    )
    if constructed_type_names is None:
        collected_constructed_type_names: set[str] = set()
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(statement):
                if isinstance(child, ast.Call):
                    collected_constructed_type_names.update(
                        PresentationAuthorityConstructionCollector.construction_type_names(
                            child
                        )
                    )
    else:
        collected_constructed_type_names = set(constructed_type_names)
    if not (
        constant_assignments
        or annotated_fields
        or declared_type_names
        or collected_constructed_type_names
        or is_dataclass
        or authority.semantic_authority_shape
    ):
        return None
    return SemanticClassSupplement(
        class_symbol=class_symbol,
        constant_assignments=constant_assignments,
        annotated_fields=annotated_fields,
        declared_type_names=declared_type_names,
        constructed_type_names=sorted_tuple(collected_constructed_type_names),
        is_dataclass=is_dataclass,
        autoregister_authority_shape=authority.semantic_authority_shape,
    )


def _compact_semantic_class_supplements_from_syntax_index(
    parsed_module: ParsedModule,
) -> tuple[SemanticClassSupplement, ...]:
    """Derive context-only supplements from the shared module event index."""

    indexed_class_nodes = _semantic_indexed_class_nodes(list(parsed_module.module.body))
    if not indexed_class_nodes:
        return ()
    class_ids_by_direct_method_id: dict[int, tuple[int, ...]] = {}
    for _qualname, class_node in indexed_class_nodes:
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            class_ids_by_direct_method_id[id(statement)] = (
                *class_ids_by_direct_method_id.get(id(statement), ()),
                id(class_node),
            )

    constructed_type_names_by_class_id: dict[int, set[str]] = {}
    syntax_index = module_syntax_index(parsed_module.module)
    for node_index in syntax_index.node_indices_by_type.get(ast.Call, ()):
        call = syntax_index.depth_first_nodes[node_index]
        if not isinstance(call, ast.Call):
            continue
        type_names = PresentationAuthorityConstructionCollector.construction_type_names(
            call
        )
        if not type_names:
            continue
        scope = syntax_index.scopes[syntax_index.scope_ids[node_index]]
        for function_node_index in scope.function_node_indices:
            function = syntax_index.depth_first_nodes[function_node_index]
            for class_id in class_ids_by_direct_method_id.get(id(function), ()):
                constructed_type_names_by_class_id.setdefault(class_id, set()).update(
                    type_names
                )

    return tuple(
        supplement
        for qualname, class_node in indexed_class_nodes
        if (
            supplement := _semantic_class_supplement(
                f"{parsed_module.module_name}.{qualname}",
                class_node,
                constructed_type_names=constructed_type_names_by_class_id.get(
                    id(class_node), ()
                ),
            )
        )
        is not None
    )


class CompactSemanticModuleProjectionFamily(
    CollectedFamily[CompactSemanticModuleProjection]
):
    """Persist deferred semantic-descent facts without repository ASTs."""

    item_type = CompactSemanticModuleProjection
    cache_payload_max_bytes = 1_000_000
    report_demand_builder = staticmethod(_semantic_report_demand)
    cached_demand_projector = staticmethod(_cached_semantic_demand_projection)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactSemanticModuleProjection]:
        return cls._collect(parsed_module, include_presentations=True)

    @classmethod
    def collect_demanded(
        cls,
        parsed_module: ParsedModule,
        demand: object,
    ) -> list[CompactSemanticModuleProjection] | None:
        if not isinstance(demand, CompactSemanticProjectionDemand):
            raise TypeError("semantic projection demand has the wrong authority type")
        return cls._collect(
            parsed_module,
            include_presentations=demand.include_presentations,
        )

    @classmethod
    def _collect(
        cls,
        parsed_module: ParsedModule,
        *,
        include_presentations: bool,
    ) -> list[CompactSemanticModuleProjection]:
        del cls
        visitor = None
        if include_presentations:
            visitor = _ProjectionVisitor(parsed_module, None)
            visitor.visit(parsed_module.module)
        return [
            CompactSemanticModuleProjection(
                module_name=parsed_module.module_name,
                file_path=str(parsed_module.path),
                projections=sorted_tuple(
                    (() if visitor is None else visitor.projections),
                    key=lambda item: (
                        item.location.file_path,
                        item.location.line,
                        item.label,
                    ),
                ),
                class_supplements=(
                    _compact_semantic_class_supplements_from_syntax_index(parsed_module)
                    if visitor is None
                    else visitor.class_supplements
                ),
            )
        ]


def _resolved_compact_class_symbols(
    resolver: CompactClassReferenceResolver,
    *,
    module_name: str,
    reference_parts: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            symbol
            for parts in reference_parts
            if (
                symbol := resolver.symbol_for(
                    module_name=module_name,
                    reference_parts=parts,
                    allow_unique_unqualified=False,
                )
            )
            is not None
        }
    )


def _resolved_compact_semantic_projections(
    semantic_projections: tuple[CompactSemanticModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    class_index: CompactClassFamilyIndex,
) -> tuple[PresentationProjection, ...]:
    resolver = CompactClassReferenceResolver.from_index(
        class_projections,
        class_index,
    )
    resolved: list[PresentationProjection] = []
    for module_projection in semantic_projections:
        for projection in module_projection.projections:
            class_symbols = _resolved_compact_class_symbols(
                resolver,
                module_name=module_projection.module_name,
                reference_parts=projection.class_reference_parts,
            )
            key_value_pairs: list[PresentationKeyValuePair] = []
            for pair in projection.key_value_pairs:
                value_class_symbols = _resolved_compact_class_symbols(
                    resolver,
                    module_name=module_projection.module_name,
                    reference_parts=pair.value_class_reference_parts,
                )
                value_tokens = sorted_tuple(
                    set(pair.value_tokens)
                    | _class_reference_normalized_tokens(
                        class_index,
                        value_class_symbols,
                    )
                )
                if (
                    pair.value_tokens == value_tokens
                    and pair.value_class_symbols == value_class_symbols
                    and not pair.value_class_reference_parts
                ):
                    key_value_pairs.append(pair)
                    continue
                key_value_pairs.append(
                    replace(
                        pair,
                        value_tokens=value_tokens,
                        value_class_symbols=value_class_symbols,
                        value_class_reference_parts=(),
                    )
                )
            resolved_key_value_pairs = tuple(key_value_pairs)
            if (
                projection.key_value_pairs == resolved_key_value_pairs
                and projection.class_symbols == class_symbols
                and not projection.class_reference_parts
            ):
                resolved.append(projection)
                continue
            resolved.append(
                replace(
                    projection,
                    key_value_pairs=resolved_key_value_pairs,
                    class_symbols=class_symbols,
                    class_reference_parts=(),
                )
            )
    return sorted_tuple(
        resolved,
        key=lambda item: (item.location.file_path, item.location.line, item.label),
    )


def build_compact_semantic_mirror_resolver(
    semantic_projections: tuple[CompactSemanticModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> SemanticMirrorResolver:
    """Build one reusable exact resolver from AST-free repository projections."""

    if class_index is None:
        class_index = build_compact_class_family_index(class_projections)
    supplements = tuple(
        supplement
        for projection in semantic_projections
        for supplement in projection.class_supplements
    )
    authority_catalog = SemanticAuthorityBuilder.from_compact_class_index(
        class_index,
        {supplement.class_symbol: supplement for supplement in supplements},
    ).build()
    projections = _resolved_compact_semantic_projections(
        semantic_projections,
        class_projections,
        class_index,
    )
    return SemanticMirrorResolver(
        authority_catalog.authorities,
        authority_catalog.facts,
        projections,
        class_index,
        supplements,
    )


def build_compact_semantic_mirror_resolution(
    semantic_projections: tuple[CompactSemanticModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[SemanticDescentGraphSpace, SemanticAuthorityProjectionResolution]:
    """Resolve edges, then release matching-only caches before publication."""

    resolver = build_compact_semantic_mirror_resolver(
        semantic_projections,
        class_projections,
        class_index=class_index,
    )
    resolution = resolver.resolve()
    graph_space = SemanticDescentGraphSpace(
        resolver.authorities,
        resolver.facts,
        resolver.projections,
    )
    return graph_space, resolution


def build_compact_semantic_descent_graph(
    semantic_projections: tuple[CompactSemanticModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> SemanticDescentGraph:
    """Build the exact semantic graph from AST-free repository projections."""

    graph_space, resolution = build_compact_semantic_mirror_resolution(
        semantic_projections,
        class_projections,
        class_index=class_index,
    )
    return SemanticDescentGraph.from_resolution(graph_space, resolution)


@dataclass
class ProjectionOwnerConstructionFrame:
    """Single-pass function construction state for direct owner projections."""

    constructions: set[PresentationAuthorityConstruction]
    projection_indices: list[int]


@dataclass
class CompactSemanticClassSupplementFrame:
    """Accumulate class construction facts during the projection traversal."""

    node: ast.ClassDef
    qualname: str
    direct_method_ids: frozenset[int]
    supplement_index: int
    constructed_type_names: set[str] = field(default_factory=set)


class _ProjectionVisitor(ClassFunctionStackNodeVisitor):
    def __init__(
        self,
        parsed_module: ParsedModule,
        class_index: ClassFamilyIndex | None,
        *,
        include_presentations: bool = True,
    ) -> None:
        super().__init__()
        self.parsed_module = parsed_module
        self.include_presentations = include_presentations
        self.class_reference_resolver = (
            None
            if class_index is None or not include_presentations
            else ModuleClassReferenceResolver(parsed_module, class_index)
        )
        self.deferred_class_reference_collector = (
            DeferredModuleClassReferenceCollector(parsed_module)
            if class_index is None and include_presentations
            else None
        )
        self.projections: list[PresentationProjection] = []
        self.owner_construction_stack: list[ProjectionOwnerConstructionFrame] = []
        self.class_supplement_stack: list[CompactSemanticClassSupplementFrame] = []
        self.active_class_method_frames: list[CompactSemanticClassSupplementFrame] = []
        self._class_supplements: list[SemanticClassSupplement | None] = []

    @property
    def class_supplements(self) -> tuple[SemanticClassSupplement, ...]:
        return tuple(
            supplement
            for supplement in self._class_supplements
            if supplement is not None
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        frame: CompactSemanticClassSupplementFrame | None = None
        if not self.function_stack:
            frame = CompactSemanticClassSupplementFrame(
                node=node,
                qualname=".".join((*self.class_stack, node.name)),
                direct_method_ids=frozenset(
                    id(statement)
                    for statement in node.body
                    if isinstance(
                        statement,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                ),
                supplement_index=len(self._class_supplements),
            )
            self._class_supplements.append(None)
            self.class_supplement_stack.append(frame)
        try:
            super().visit_ClassDef(node)
        finally:
            if frame is not None:
                self.class_supplement_stack.pop()
                self._class_supplements[frame.supplement_index] = (
                    _semantic_class_supplement(
                        f"{self.parsed_module.module_name}.{frame.qualname}",
                        node,
                        constructed_type_names=frame.constructed_type_names,
                    )
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        direct_class_frames = tuple(
            frame
            for frame in self.class_supplement_stack
            if id(node) in frame.direct_method_ids
        )
        self.active_class_method_frames.extend(direct_class_frames)
        frame = (
            ProjectionOwnerConstructionFrame(
                constructions=set(),
                projection_indices=[],
            )
            if self.include_presentations
            else None
        )
        if frame is not None:
            self.owner_construction_stack.append(frame)
        try:
            super().visit_FunctionDef(node)
        finally:
            if frame is not None:
                self.owner_construction_stack.pop()
                owner_constructions = frozenset(frame.constructions)
                for projection_index in frame.projection_indices:
                    projection = self.projections[projection_index]
                    self.projections[projection_index] = replace(
                        projection,
                        owner_constructions=sorted_tuple(
                            frozenset(
                                (*projection.owner_constructions, *owner_constructions)
                            ),
                            key=lambda item: (item.type_name, item.field_tokens),
                        ),
                    )
            if direct_class_frames:
                del self.active_class_method_frames[-len(direct_class_frames) :]

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self.active_class_method_frames:
            self._record_class_construction_type_names(
                PresentationAuthorityConstructionCollector.construction_type_names(node)
            )
        if self.include_presentations:
            self._record_owner_constructions(
                PresentationAuthorityConstructionCollector.constructions_for_call(node)
            )
        self.generic_visit(node)

    def _record_class_construction_type_names(
        self,
        type_names: Iterable[str],
    ) -> None:
        type_name_tuple = tuple(type_names)
        if not type_name_tuple:
            return
        for frame in self.active_class_method_frames:
            frame.constructed_type_names.update(type_name_tuple)

    def _record_class_construction_type_names_for_node(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._record_class_construction_type_names(
                    PresentationAuthorityConstructionCollector.construction_type_names(
                        child
                    )
                )

    def _record_owner_constructions(
        self,
        constructions: tuple[PresentationAuthorityConstruction, ...],
    ) -> None:
        if not constructions:
            return
        for frame in self.owner_construction_stack:
            frame.constructions.update(constructions)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        if self._collect_assignment_projection(node, node.value):
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        if node.value is not None and self._collect_assignment_projection(
            node, node.value
        ):
            return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        if node.value is not None and self._collect_return_projection(node, node.value):
            return
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
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
        if not self.include_presentations:
            self.generic_visit(node)
            return
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
        self._record_owner_constructions(projection_constructions)
        class_symbols = (
            self.class_reference_resolver.symbols_for_node(value)
            if self.class_reference_resolver is not None
            else ()
        )
        class_reference_parts = (
            self.deferred_class_reference_collector.reference_parts_for_node(value)
            if self.deferred_class_reference_collector is not None
            else ()
        )
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
        self._record_class_construction_type_names_for_node(value)
        self._append_projection(
            node,
            projection_kind,
            label,
            tokens,
            projection_constructions,
            key_value_pairs,
            class_symbols,
            class_reference_parts,
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
        return PresentationAuthorityConstructionCollector.constructions_for_projection(
            value
        )

    def _projection_key_value_pairs(
        self,
        value: ast.Dict,
    ) -> tuple[PresentationKeyValuePair, ...]:
        pairs: list[PresentationKeyValuePair] = []
        for key, item_value in zip(value.keys, value.values, strict=True):
            if key is None:
                continue
            if self.class_reference_resolver is not None:
                pairs.append(
                    PresentationKeyValuePair.from_nodes(
                        key=key,
                        value=item_value,
                        class_reference_resolver=self.class_reference_resolver,
                    )
                )
                continue
            if self.deferred_class_reference_collector is None:
                raise RuntimeError("deferred class-reference collector disappeared")
            pairs.append(
                PresentationKeyValuePair(
                    key_source=ast.unparse(key),
                    value_source=ast.unparse(item_value),
                    value_tokens=sorted_tuple(
                        {
                            token.value
                            for token in PresentationTokenProjection.tokens_for_node(
                                item_value,
                                PresentationTokenRole.DICT_VALUE,
                            )
                        }
                    ),
                    value_class_reference_parts=(
                        self.deferred_class_reference_collector.reference_parts_for_node(
                            item_value
                        )
                    ),
                )
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
        class_reference_parts: tuple[tuple[str, ...], ...] = (),
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
                    frozenset(projection_constructions),
                    key=lambda item: (item.type_name, item.field_tokens),
                ),
                projection_constructions=sorted_tuple(
                    frozenset(projection_constructions),
                    key=lambda item: (item.type_name, item.field_tokens),
                ),
                key_value_pairs=key_value_pairs,
                class_symbols=class_symbols,
                class_reference_parts=class_reference_parts,
            )
        )
        if self.owner_construction_stack:
            self.owner_construction_stack[-1].projection_indices.append(
                len(self.projections) - 1
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
    def has_matched_class_reference(
        projection: PresentationProjection,
        matched_facts: tuple[SemanticFact, ...],
    ) -> bool:
        """Return whether a projection names one of the matched family leaves."""

        referenced_class_names = frozenset(
            symbol.rpartition(".")[2] for symbol in projection.class_symbols
        )
        return any(
            fact.location.symbol.rpartition(".")[2] in referenced_class_names
            for fact in matched_facts
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

    class_index: SemanticClassFamilyIndex
    projections: tuple[PresentationProjection, ...]

    @cached_property
    def class_symbols_by_projection_id(self) -> dict[str, str | None]:
        return {}

    @cached_property
    def ancestor_symbols_by_class_symbol(self) -> dict[str, frozenset[str]]:
        return {}

    def class_symbol_for_projection(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        cache = self.class_symbols_by_projection_id
        if projection.projection_id not in cache:
            cache[projection.projection_id] = self._resolve_class_symbol(projection)
        return cache[projection.projection_id]

    def ancestor_symbols_for_class(self, class_symbol: str) -> frozenset[str]:
        cache = self.ancestor_symbols_by_class_symbol
        if class_symbol not in cache:
            cache[class_symbol] = frozenset(
                self.class_index.ancestor_symbols(class_symbol)
            )
        return cache[class_symbol]

    def _resolve_class_symbol(
        self,
        projection: PresentationProjection,
    ) -> str | None:
        if projection.owner_symbol == ProjectionOwnerSymbol.module_owner_value:
            return None
        owner_qualname_parts = tuple(projection.owner_symbol.split("."))
        for end_index in range(len(owner_qualname_parts), 0, -1):
            owner_qualname = ".".join(owner_qualname_parts[:end_index])
            symbol = self.class_index.symbol_for(
                file_path=projection.location.file_path,
                qualname=owner_qualname,
            )
            if symbol is not None:
                return symbol
        return None


@dataclass(frozen=True)
class DataclassAuthorityNameAffinity:
    """Specific shared role identity between two dataclass authorities."""

    left: SemanticAuthority
    right: SemanticAuthority

    def has_affinity(self) -> bool:
        weak_tokens = SemanticRoleIdentityToken.authority_affinity_weak_values()
        left_tokens = NormalizeNameProjection.token_set(self.left.name) - weak_tokens
        right_tokens = NormalizeNameProjection.token_set(self.right.name) - weak_tokens
        return bool(left_tokens & right_tokens)


@dataclass(frozen=True)
class ConstructionAuthorityResolver:
    """Resolve owner construction sites that descend to semantic authorities."""

    class_index: SemanticClassFamilyIndex
    authorities: tuple[SemanticAuthority, ...]
    projection_construction_type_names: frozenset[str] = frozenset()
    compact_class_supplements_by_symbol: Mapping[
        str, SemanticClassSupplement
    ] = field(default_factory=dict)

    @cached_property
    def construction_authority_class_cache(
        self,
    ) -> dict[ConstructionAuthorityCacheKey, bool]:
        return {}

    def construction_type_descends_to_authority(
        self,
        construction: PresentationAuthorityConstruction,
        authority: SemanticAuthority,
    ) -> bool:
        return authority.authority_id in (
            self.descended_authority_ids_for_construction_type(construction.type_name)
        )

    @cached_property
    def descended_authority_ids_by_construction_type(
        self,
    ) -> dict[str, frozenset[str]]:
        """Memoize the complete descent target set for each constructed type.

        A construction's type relationship is independent of the projection and
        of the candidate edge.  Keeping the relationship as an authority-id set
        turns later projection resolution into set membership instead of one
        class-lineage/materializer query per projection-authority pair.
        """

        return {}

    def descended_authority_ids_for_construction_type(
        self,
        construction_type: str,
    ) -> frozenset[str]:
        descended = self.descended_authority_ids_by_construction_type
        if construction_type not in descended:
            descended[construction_type] = frozenset(
                self.authority_ids_for_constructed_type_name(construction_type)
                | self.materialized_authority_ids_for_construction_type(
                    construction_type
                )
            )
        return descended[construction_type]

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
        return authority.authority_id in (
            self.materialized_authority_ids_for_construction_type(
                construction.type_name
            )
        )

    def materialized_authority_ids_for_construction_type(
        self,
        construction_type: str,
    ) -> frozenset[str]:
        materialized = self.materialized_authority_ids_by_construction_type
        if (
            construction_type not in materialized
            and construction_type not in self.projection_construction_type_names
            and construction_type
            not in self.resolved_additional_materialization_type_names
        ):
            authority_ids = self._materialized_authority_ids_for_construction_type(
                construction_type
            )
            if authority_ids:
                materialized[construction_type] = authority_ids
            self.resolved_additional_materialization_type_names.add(construction_type)
        return materialized.get(construction_type, _EMPTY_STRING_FROZENSET)

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

    @cached_property
    def authority_ids_by_name(self) -> dict[str, frozenset[str]]:
        """Materialize the legacy set-valued name index only on direct demand."""

        return {
            name: frozenset(
                (authority_ids,) if isinstance(authority_ids, str) else authority_ids
            )
            for name, authority_ids in self.compact_authority_ids_by_name.items()
        }

    @cached_property
    def compact_authority_ids_by_name(
        self,
    ) -> dict[str, CompactAuthorityIdsByNameValue]:
        authority_ids: dict[str, CompactAuthorityIdsByNameValue] = {}
        for authority in self.authorities:
            existing = authority_ids.get(authority.name)
            if existing is None:
                authority_ids[authority.name] = authority.authority_id
            elif isinstance(existing, str):
                authority_ids[authority.name] = (existing, authority.authority_id)
            else:
                authority_ids[authority.name] = (*existing, authority.authority_id)
        return authority_ids

    @cached_property
    def known_authority_ids(self) -> frozenset[str]:
        return frozenset(authority.authority_id for authority in self.authorities)

    @cached_property
    def authority_ids_by_constructed_type_name(
        self,
    ) -> dict[str, frozenset[str]]:
        """Memoize only constructed types reached from relevant materializers."""

        return {}

    @cached_property
    def resolved_constructed_type_names(self) -> set[str]:
        return set()

    def authority_ids_for_constructed_type_name(
        self,
        type_name: str,
    ) -> frozenset[str]:
        authority_ids_by_type = self.authority_ids_by_constructed_type_name
        if type_name in self.resolved_constructed_type_names:
            return authority_ids_by_type.get(type_name, _EMPTY_STRING_FROZENSET)
        direct_authority_ids = self.compact_authority_ids_by_name.get(type_name)
        if direct_authority_ids is None:
            authority_ids: set[str] = set()
        elif isinstance(direct_authority_ids, str):
            authority_ids = {direct_authority_ids}
        else:
            authority_ids = set(direct_authority_ids)
        for class_symbol in self.class_index.symbols_by_simple_name.get(
            type_name,
            (),
        ):
            if class_symbol in self.known_authority_ids:
                authority_ids.add(class_symbol)
            authority_ids.update(
                self.known_authority_ids.intersection(
                    self.class_index.ancestor_symbols(class_symbol)
                )
            )
        resolved_ids = (
            frozenset(authority_ids) if authority_ids else _EMPTY_STRING_FROZENSET
        )
        if resolved_ids:
            authority_ids_by_type[type_name] = resolved_ids
        self.resolved_constructed_type_names.add(type_name)
        return resolved_ids

    def _class_materialization_inputs(
        self,
        indexed_class: IndexedClass | CompactIndexedClass,
    ) -> tuple[frozenset[str], frozenset[str]]:
        if isinstance(indexed_class, CompactIndexedClass):
            supplement = self.compact_class_supplements_by_symbol.get(
                indexed_class.symbol
            )
            if supplement is None:
                return _EMPTY_STRING_FROZENSET, _EMPTY_STRING_FROZENSET
            return (
                frozenset(supplement.declared_type_names),
                frozenset(supplement.constructed_type_names),
            )
        declared_type_names = frozenset(
            terminal_name
            for _, value in AutoRegisterClassAuthority(
                indexed_class.node
            ).assignment_pairs
            if (terminal_name := AttributeChainAuthority.terminal_name(value))
            is not None
        )
        constructed_type_names: set[str] = set()
        for statement in indexed_class.node.body:
            if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for child in ast.walk(statement):
                if isinstance(child, ast.Call):
                    constructed_type_names.update(
                        PresentationAuthorityConstructionCollector.construction_type_names(
                            child
                        )
                    )
        return declared_type_names, frozenset(constructed_type_names)

    @cached_property
    def materialized_authority_ids_by_construction_type(
        self,
    ) -> dict[str, frozenset[str]]:
        """Preindex only materializers named by projection construction evidence."""

        materialized: dict[str, frozenset[str]] = {}
        for construction_type in self.projection_construction_type_names:
            authority_ids = self._materialized_authority_ids_for_construction_type(
                construction_type
            )
            if authority_ids:
                materialized[construction_type] = authority_ids
        return materialized

    @cached_property
    def resolved_additional_materialization_type_names(self) -> set[str]:
        return set()

    def _materialized_authority_ids_for_construction_type(
        self,
        construction_type: str,
    ) -> frozenset[str]:
        authority_ids: set[str] = set()
        for class_symbol in self.class_index.symbols_by_simple_name.get(
            construction_type,
            (),
        ):
            indexed_class = self.class_index.class_for(class_symbol)
            if indexed_class is None:
                continue
            declared_type_names, constructed_type_names = (
                self._class_materialization_inputs(indexed_class)
            )
            for type_name in declared_type_names:
                declared_authority_ids = self.compact_authority_ids_by_name.get(
                    type_name
                )
                if isinstance(declared_authority_ids, str):
                    authority_ids.add(declared_authority_ids)
                elif declared_authority_ids is not None:
                    authority_ids.update(declared_authority_ids)
            for type_name in constructed_type_names:
                authority_ids.update(
                    self.authority_ids_for_constructed_type_name(type_name)
                )
        return frozenset(authority_ids) if authority_ids else _EMPTY_STRING_FROZENSET


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
    def dataclass_authorities_by_id(self) -> dict[str, SemanticAuthority]:
        return {
            authority.authority_id: authority
            for authority in self.dataclass_authorities
        }

    @cached_property
    def dataclass_fact_tokens_by_authority_id(
        self,
    ) -> dict[str, frozenset[str]]:
        return {}

    def fact_tokens_for_authority(self, authority_id: str) -> frozenset[str]:
        cache = self.dataclass_fact_tokens_by_authority_id
        if authority_id not in cache:
            cache[authority_id] = frozenset(
                variant
                for fact in self.fact_authority_index.facts_for_authority(authority_id)
                for variant in normalized_name_variants(fact.name)
            )
        return cache[authority_id]

    @cached_property
    def projection_descent_authority_ids(self) -> dict[str, frozenset[str]]:
        return {}

    def descent_authority_ids_for_projection(
        self,
        projection: PresentationProjection,
    ) -> frozenset[str]:
        cache = self.projection_descent_authority_ids
        if projection.projection_id not in cache:
            cache[projection.projection_id] = self._projection_descent_authority_ids(
                projection
            )
        return cache[projection.projection_id]

    @cached_property
    def projection_materializes_any_dataclass_authority_cache(self) -> dict[str, bool]:
        return {}

    @cached_property
    def constructed_dataclass_authorities_by_projection_id(
        self,
    ) -> dict[str, tuple[SemanticAuthority, ...]]:
        return {}

    @cached_property
    def constructed_dataclass_authority_ids_by_projection_id(
        self,
    ) -> dict[str, frozenset[str]]:
        return {}

    def projection_descends_to_any_dataclass_authority(
        self,
        projection: PresentationProjection,
    ) -> bool:
        return bool(self.descent_authority_ids_for_projection(projection))

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

    def projection_constructs_distinct_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        """Return whether a nominal construction targets another schema.

        A direct dataclass construction already has a nominal output authority.
        Similar names do not turn its keyword arguments into a projection of a
        sibling schema; any such relationship belongs on the declarations, not
        on the constructor call.
        """

        return bool(
            self.constructed_dataclass_authority_ids(projection)
            - {authority.authority_id}
        )

    def projection_constructs_name_unrelated_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> bool:
        """Return whether a collection constructs a different semantic role."""

        return any(
            constructed_authority.authority_id != authority.authority_id
            and not DataclassAuthorityNameAffinity(
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
                self.dataclass_authorities_by_id[authority_id]
                for authority_id in sorted(
                    self.constructed_dataclass_authority_ids(projection)
                )
            )
        return cache[projection.projection_id]

    def constructed_dataclass_authority_ids(
        self,
        projection: PresentationProjection,
    ) -> frozenset[str]:
        cache = self.constructed_dataclass_authority_ids_by_projection_id
        if projection.projection_id not in cache:
            cache[projection.projection_id] = (
                self._constructed_dataclass_authority_ids_uncached(projection)
            )
        return cache[projection.projection_id]

    def _constructed_dataclass_authority_ids_uncached(
        self,
        projection: PresentationProjection,
    ) -> frozenset[str]:
        """Resolve nominal constructor targets independent of omitted defaults."""

        return frozenset().union(
            *(
                self.construction_resolver.authority_ids_for_constructed_type_name(
                    construction.type_name
                )
                & self.dataclass_authority_ids
                for construction in projection.projection_constructions
            )
        )

    def derivation_proof_edges(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
    ) -> tuple[AuthorityProofEdge, ...]:
        class_proof = self._class_derivation_proof(projection, authority)
        if class_proof:
            return class_proof
        direct_construction_proof = self._construction_derivation_proof(
            projection,
            authority,
            matched_facts,
            projection.projection_constructions,
            self.construction_resolver.construction_type_descends_to_authority,
            AuthorityProofEdgeKind.OWNS_FIELD_SET,
            "projection constructs the authority-owned field set",
        )
        if direct_construction_proof:
            return direct_construction_proof
        return self._construction_derivation_proof(
            projection,
            authority,
            matched_facts,
            projection.projection_constructions,
            self.construction_resolver.construction_type_materializes_authority,
            AuthorityProofEdgeKind.PROVIDES_QUERY_METHOD,
            "projection uses a declared materializer for the authority-owned field set",
        )

    def _class_derivation_proof(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
    ) -> tuple[AuthorityProofEdge, ...]:
        projection_class_symbol = (
            self.projection_class_symbol_lineage.class_symbol_for_projection(projection)
        )
        if projection_class_symbol is None:
            return ()
        class_authority_ids = {
            projection_class_symbol,
            *self.projection_class_symbol_lineage.ancestor_symbols_for_class(
                projection_class_symbol
            ),
        }
        if authority.authority_id not in class_authority_ids:
            return ()
        indexed_class = self.projection_class_symbol_lineage.class_index.class_for(
            projection_class_symbol
        )
        if indexed_class is None:
            return ()
        owns_authority = projection_class_symbol == authority.authority_id
        return (
            AuthorityProofEdge(
                edge_kind=(
                    AuthorityProofEdgeKind.OWNS_FIELD_SET
                    if owns_authority
                    else AuthorityProofEdgeKind.INHERITS_FROM
                ),
                authority_id=authority.authority_id,
                authority_kind=authority.kind.value,
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                symbol=indexed_class.symbol,
                detail=(
                    "projection is owned by the dataclass authority"
                    if owns_authority
                    else "projection owner inherits the dataclass authority"
                ),
            ),
        )

    def _construction_derivation_proof(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
        constructions: tuple[PresentationAuthorityConstruction, ...],
        accepts_construction: ConstructionAuthorityPredicate,
        edge_kind: AuthorityProofEdgeKind,
        detail: str,
    ) -> tuple[AuthorityProofEdge, ...]:
        if not self._constructions_derive_dataclass_authority(
            constructions,
            authority,
            matched_facts,
            accepts_construction,
        ):
            return ()
        if any(
            construction.queries_authority(authority)
            and accepts_construction(construction, authority)
            for construction in constructions
        ):
            edge_kind = AuthorityProofEdgeKind.PROVIDES_QUERY_METHOD
            detail = "projection calls an authority-owned derivation method"
        return (
            AuthorityProofEdge(
                edge_kind=edge_kind,
                authority_id=authority.authority_id,
                authority_kind=authority.kind.value,
                file_path=projection.location.file_path,
                line=projection.location.line,
                symbol=projection.location.symbol,
                detail=detail,
            ),
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
            self.projection_class_symbol_lineage.ancestor_symbols_for_class(
                projection_class_symbol
            )
            & self.projection_class_symbol_lineage.ancestor_symbols_for_class(
                authority.authority_id
            )
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
            return _EMPTY_STRING_FROZENSET
        projection_ancestor_symbols = (
            self.projection_class_symbol_lineage.ancestor_symbols_for_class(
                projection_class_symbol
            )
        )
        authority_ids = {
            projection_class_symbol,
            *projection_ancestor_symbols,
        }.intersection(self.dataclass_authority_ids)
        return frozenset(authority_ids) if authority_ids else _EMPTY_STRING_FROZENSET

    def _projection_materializes_any_dataclass_authority_uncached(
        self,
        projection: PresentationProjection,
    ) -> bool:
        candidate_authority_ids = frozenset(
            authority_id
            for construction in projection.owner_constructions
            for authority_id in (
                self.construction_resolver.materialized_authority_ids_for_construction_type(
                    construction.type_name
                )
            )
            if authority_id in self.dataclass_authority_ids
        )
        return any(
            self._constructions_derive_dataclass_authority_from_tokens(
                projection.owner_constructions,
                self.dataclass_authorities_by_id[authority_id],
                self.fact_tokens_for_authority(authority_id),
                self.construction_resolver.construction_type_materializes_authority,
            )
            for authority_id in candidate_authority_ids
        )

    def _projection_owner_derives_dataclass_authority(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
        accepts_construction: ConstructionAuthorityPredicate,
    ) -> bool:
        return self._constructions_derive_dataclass_authority(
            projection.owner_constructions,
            authority,
            matched_facts,
            accepts_construction,
        )

    @classmethod
    def _constructions_derive_dataclass_authority(
        cls,
        constructions: tuple[PresentationAuthorityConstruction, ...],
        authority: SemanticAuthority,
        matched_facts: tuple[SemanticFact, ...],
        accepts_construction: ConstructionAuthorityPredicate,
    ) -> bool:
        if not matched_facts:
            return False
        return cls._constructions_derive_dataclass_authority_from_tokens(
            constructions,
            authority,
            frozenset(
                variant
                for fact in matched_facts
                for variant in normalized_name_variants(fact.name)
            ),
            accepts_construction,
        )

    @staticmethod
    def _constructions_derive_dataclass_authority_from_tokens(
        constructions: tuple[PresentationAuthorityConstruction, ...],
        authority: SemanticAuthority,
        matched_tokens: frozenset[str],
        accepts_construction: ConstructionAuthorityPredicate,
    ) -> bool:
        if not matched_tokens:
            return False
        descended_field_tokens: set[str] = set()
        for construction in constructions:
            if not accepts_construction(construction, authority):
                continue
            descended_field_tokens.update(construction.field_tokens)
            if matched_tokens <= frozenset(construction.field_tokens):
                return True
        return matched_tokens <= frozenset(descended_field_tokens)


@dataclass(frozen=True)
class SemanticAuthorityProjectionResolutionContext:
    """Composed policy context for deciding mirror admissibility and descent."""

    projection_semantics: ProjectionSemanticAuthority
    dataclass_descent: DataclassProjectionDescentAuthority
    fact_specificity: SemanticFactSpecificityIndex


@dataclass(frozen=True)
class SemanticMirrorResolver(SemanticDescentGraphSpace):
    """Resolve graph edges where a projection mirrors an authority."""

    class_index: SemanticClassFamilyIndex
    compact_class_supplements: tuple[SemanticClassSupplement, ...] = ()

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
        return ConstructionAuthorityResolver(
            class_index=self.class_index,
            authorities=self.authorities,
            projection_construction_type_names=frozenset(
                construction.type_name
                for projection in self.projections
                for construction in projection.owner_constructions
            ),
            compact_class_supplements_by_symbol={
                supplement.class_symbol: supplement
                for supplement in self.compact_class_supplements
            },
        )

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
    def resolution_context(self) -> SemanticAuthorityProjectionResolutionContext:
        return SemanticAuthorityProjectionResolutionContext(
            projection_semantics=self.projection_semantics,
            dataclass_descent=self.dataclass_descent,
            fact_specificity=self.fact_specificity_index,
        )

    def resolve(self) -> SemanticAuthorityProjectionResolution:
        classifications: list[SemanticAuthorityProjectionResolution] = []
        for projection in self.projections:
            for authority_id, matches in self._matches_by_authority(
                projection,
            ).items():
                classifications.append(
                    self._resolution_for(
                        projection,
                        self.authority_catalog.authority(authority_id),
                        self.fact_authority_index.facts_for_authority(authority_id),
                        matches,
                    )
                )
        return SemanticAuthorityProjectionResolution.combine(classifications)

    def _matches_by_authority(
        self,
        projection: PresentationProjection,
    ) -> FactMatchesByAuthority:
        matches_by_authority: FactMatchesByAuthority = {}
        for token in projection.tokens:
            for ref in self._candidate_refs_for_token(token):
                matches_by_authority.setdefault(ref.authority_id, {}).setdefault(
                    ref.fact_id,
                    set(),
                ).add(token.value)
        for match in ProjectionClassSymbolFactMatcher(
            projection,
            self.class_index,
            self.authority_catalog,
            self.fact_authority_index,
        ).matches():
            matches_by_authority.setdefault(match.authority_id, {}).setdefault(
                match.fact_id,
                set(),
            ).add(match.token_value)
        return {
            authority_id: matches_by_fact_id
            for authority_id, matches_by_fact_id in matches_by_authority.items()
            if self._matches_can_form_mirror(
                authority_id,
                matches_by_fact_id,
            )
        }

    def _matches_can_form_mirror(
        self,
        authority_id: str,
        matches_by_fact_id: dict[str, set[str]],
    ) -> bool:
        matched_fact_count = len(matches_by_fact_id)
        if matched_fact_count < 2:
            return False
        authority_fact_count = len(
            self.fact_authority_index.facts_for_authority(authority_id)
        )
        return matched_fact_count >= 3 or (
            matched_fact_count / authority_fact_count >= 0.5
        )

    @cached_property
    def candidate_refs_by_token_signature(
        self,
    ) -> dict[
        tuple[str, PresentationTokenKind, str | None],
        tuple[SemanticFact, ...],
    ]:
        return {}

    @cached_property
    def candidate_refs_by_token(
        self,
    ) -> dict[PresentationToken, tuple[SemanticFact, ...]]:
        return {}

    @cached_property
    def fact_references_by_id(self) -> dict[str, SemanticFactReference]:
        return {}

    def _candidate_refs_for_token(
        self,
        token: PresentationToken,
    ) -> tuple[SemanticFact, ...]:
        refs = self.fact_token_index.by_token.get(token.value, ())
        if not refs or token.kind is not PresentationTokenKind.QUALIFIED_ATTRIBUTE:
            return refs
        cache = self.candidate_refs_by_token
        if token not in cache:
            cache[token] = self._candidate_refs_for_token_uncached(token)
        return cache[token]

    def _candidate_refs_for_token_uncached(
        self,
        token: PresentationToken,
    ) -> tuple[SemanticFact, ...]:
        refs = self.fact_token_index.by_token.get(token.value, ())
        if not refs or token.kind is not PresentationTokenKind.QUALIFIED_ATTRIBUTE:
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

    def _resolution_for(
        self,
        projection: PresentationProjection,
        authority: SemanticAuthority,
        facts: tuple[SemanticFact, ...],
        matches_by_fact_id: dict[str, set[str]],
    ) -> SemanticAuthorityProjectionResolution:
        match = SemanticAuthorityMatch.from_authority_matches(
            facts,
            matches_by_fact_id,
            self.fact_references_by_id,
        )
        if match is None:
            return SemanticAuthorityProjectionResolution.suppressed()
        candidate = SemanticMirrorEdgeCandidate(
            projection=projection,
            authority=authority,
            facts=facts,
            match=match,
        )
        policy = self.policy_catalog.policy_for_authority(authority)
        return policy.classify(self.resolution_context, candidate)


class NormalizeNameProjection:
    """Normalize source names and literal keys into semantic comparison tokens."""

    @classmethod
    @lru_cache(maxsize=8_192)
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
    @lru_cache(maxsize=4_096)
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
    class_index: SemanticClassFamilyIndex,
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
        return cls.projector_for_node_type(type(node))

    @classmethod
    @lru_cache(maxsize=None)
    def projector_for_node_type(
        cls,
        node_type: type[ast.AST],
    ) -> type["PresentationTokenNodeProjector"] | None:
        for projector_type in cls.registered_projector_types():
            if issubclass(node_type, projector_type.node_type):
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


class StarredPresentationTokenProjector(PresentationTokenNodeProjector):
    """Preserve semantic tokens exposed through iterable unpacking."""

    node_type = ast.Starred

    @classmethod
    def tokens_for_node(
        cls,
        node: ast.AST,
        role: PresentationTokenRole,
    ) -> tuple[PresentationToken, ...]:
        del cls
        if not isinstance(node, ast.Starred):
            raise TypeError(f"Expected ast.Starred, got {type(node)!r}")
        return PresentationTokenProjection.tokens_for_node(node.value, role)


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
        return cls._constructions_for_node(node, minimum_field_count=2)

    @classmethod
    def constructions_for_projection(
        cls,
        node: ast.AST,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        return cls._constructions_for_node(node, minimum_field_count=1)

    @classmethod
    def _constructions_for_node(
        cls,
        node: ast.AST,
        *,
        minimum_field_count: int,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        constructions: list[PresentationAuthorityConstruction] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            constructions.extend(
                cls._constructions_for_call(
                    child,
                    minimum_field_count=minimum_field_count,
                )
            )
        return sorted_tuple(
            frozenset(constructions),
            key=lambda item: (item.type_name, item.field_tokens),
        )

    @classmethod
    def constructions_for_call(
        cls,
        node: ast.Call,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        return cls._constructions_for_call(node, minimum_field_count=2)

    @classmethod
    def _constructions_for_call(
        cls,
        node: ast.Call,
        *,
        minimum_field_count: int,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        field_tokens = cls.constructor_field_tokens(node)
        if len(field_tokens) < minimum_field_count:
            return ()
        type_names = cls.construction_type_names(node)
        return tuple(
            PresentationAuthorityConstruction(
                type_name=type_name,
                field_tokens=field_tokens,
                call_target_parts=AttributeChainAuthority.chain(node.func),
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
