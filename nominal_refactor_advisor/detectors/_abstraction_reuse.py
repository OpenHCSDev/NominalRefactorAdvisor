"""Generic detection for reuse of available nominal carriers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Sequence
from ..class_index import (
    CompactCarrierClassFact,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..collection_algebra import sorted_tuple
from ..models import MappingMetrics
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    DetectorConfig,
    RefactorFinding,
    SourceLocation,
    high_confidence_spec,
    CompactProjectionCandidateDetector,
)
from ._helpers import _semantic_role_names_for_fields
from ._substrate_support import (
    _IGNORED_ANCESTOR_NAMES,
    _class_ancestor_name_map,
)

_MIN_CARRIER_REUSE_FIELDS = 3
_MIN_CARRIER_REUSE_ROLES = 3
_MIN_CARRIER_SHARED_FIELD_MATCHES = 2
_MIN_CARRIER_ROLE_OVERLAP = 3
_MIN_CARRIER_AUTHORITY_COVERAGE = 0.50
_MIN_CARRIER_LOCAL_COVERAGE = 0.50

_CARRIER_NAME_SUFFIXES = (
    "Boundary",
    "Carrier",
    "Context",
    "Domain",
    "Fields",
    "Metadata",
    "Payload",
    "Provenance",
    "Record",
    "Request",
    "Semantics",
    "Spec",
    "State",
    "Value",
)


@dataclass(frozen=True, slots=True)
class FilePathLineModuleNameBase:
    file_path: str
    line: int
    module_name: str


@dataclass(frozen=True, slots=True)
class SharedFieldsBase(FilePathLineModuleNameBase):
    class_name: str


@dataclass(frozen=True, slots=True)
class CarrierBase(SharedFieldsBase):
    base_names: tuple[str, ...]
    nominal_ancestor_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CarrierSurface(CarrierBase):
    field_names: tuple[str, ...]
    field_type_map: tuple[tuple[str, str], ...]
    role_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AvailableCarrierReuseCandidate:
    local: CarrierSurface
    authority: CarrierSurface
    shared_roles: tuple[str, ...]
    shared_field_names: tuple[str, ...]


def _looks_like_reusable_carrier_name(name: str) -> bool:
    return name.endswith(_CARRIER_NAME_SUFFIXES)


def _public_name(name: str) -> bool:
    return bool(name and not name.startswith("_"))


def _top_level_package(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _compact_carrier_surface(
    projection: CompactModuleClassProjection,
    fact: CompactCarrierClassFact,
) -> CarrierSurface | None:
    if not _public_name(fact.class_name):
        return None
    if len(fact.field_type_map) < _MIN_CARRIER_REUSE_FIELDS:
        return None
    field_names = tuple(name for name, _ in fact.field_type_map)
    role_names = _semantic_role_names_for_fields(field_names)
    if len(role_names) < _MIN_CARRIER_REUSE_ROLES:
        return None
    return CarrierSurface(
        file_path=projection.file_path,
        module_name=projection.module_name,
        line=fact.line,
        class_name=fact.class_name,
        field_names=field_names,
        field_type_map=fact.field_type_map,
        role_names=role_names,
        base_names=fact.base_names,
        nominal_ancestor_names=(),
    )


def _compact_carrier_surfaces(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[CarrierSurface, ...]:
    return tuple(
        surface
        for projection in projections
        for surface in _compact_module_carrier_surfaces(projection)
    )


def _compact_module_carrier_surfaces(
    projection: CompactModuleClassProjection,
) -> tuple[CarrierSurface, ...]:
    surfaces: list[CarrierSurface] = []
    for fact in projection.carrier_class_facts:
        surface = _compact_carrier_surface(projection, fact)
        if surface is not None:
            surfaces.append(surface)
    return sorted_tuple(
        surfaces,
        key=lambda item: (item.file_path, item.line, item.class_name),
    )


def _carrier_authority_surfaces(
    surfaces: tuple[CarrierSurface, ...],
) -> tuple[CarrierSurface, ...]:
    return tuple(
        surface
        for surface in surfaces
        if _looks_like_reusable_carrier_name(surface.class_name)
    )


@lru_cache(maxsize=None)
def _package_root_name_for_path(file_path: str) -> str | None:
    path = Path(file_path)
    package_dirs: list[Path] = []
    current = path.parent
    while (current / "__init__.py").exists():
        package_dirs.append(current)
        current = current.parent
    if package_dirs:
        return package_dirs[-1].name
    if not path.is_absolute() and path.parts:
        return path.parts[0]
    return None


def _carrier_surfaces_share_package(
    left: CarrierSurface,
    right: CarrierSurface,
) -> bool:
    if _top_level_package(left.module_name) == _top_level_package(right.module_name):
        return True
    left_path_package = _package_root_name_for_path(left.file_path)
    right_path_package = _package_root_name_for_path(right.file_path)
    return left_path_package is not None and left_path_package == right_path_package


def _carrier_surface_related(left: CarrierSurface, right: CarrierSurface) -> bool:
    return (
        left.class_name == right.class_name
        or left.class_name in right.base_names
        or right.class_name in left.base_names
    )


def _carrier_surfaces_with_ancestors(
    surfaces: tuple[CarrierSurface, ...],
) -> tuple[CarrierSurface, ...]:
    base_lookup: dict[str, set[str]] = defaultdict(set)
    for surface in surfaces:
        base_lookup[surface.class_name].update(surface.base_names)
    ancestor_names_by_class = _class_ancestor_name_map(base_lookup)
    return tuple(
        sorted(
            (
                CarrierSurface(
                    file_path=surface.file_path,
                    module_name=surface.module_name,
                    line=surface.line,
                    class_name=surface.class_name,
                    field_names=surface.field_names,
                    field_type_map=surface.field_type_map,
                    role_names=surface.role_names,
                    base_names=surface.base_names,
                    nominal_ancestor_names=ancestor_names_by_class[surface.class_name],
                )
                for surface in surfaces
            ),
            key=lambda surface: (surface.file_path, surface.line, surface.class_name),
        )
    )


def _carrier_surfaces_share_nominal_ancestor(
    local: CarrierSurface,
    authority: CarrierSurface,
) -> bool:
    return bool(
        (
            set(local.nominal_ancestor_names)
            & set(authority.nominal_ancestor_names) - _IGNORED_ANCESTOR_NAMES
        )
    )


def _annotation_type_names(annotation_text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in annotation_text.replace(".", " ")
        .replace("[", " ")
        .replace("]", " ")
        .split()
        if token.isidentifier()
    )


def _carrier_uses_authority(local: CarrierSurface, authority: CarrierSurface) -> bool:
    if authority.class_name in local.base_names:
        return True
    return any(
        authority.class_name in _annotation_type_names(annotation_text)
        for _, annotation_text in local.field_type_map
    )


def _shared_carrier_field_names(
    local: CarrierSurface,
    authority: CarrierSurface,
) -> tuple[str, ...]:
    authority_field_types = dict(authority.field_type_map)
    local_field_types = dict(local.field_type_map)
    return tuple(
        field_name
        for field_name in local.field_names
        if field_name in authority_field_types
        and local_field_types.get(field_name) == authority_field_types[field_name]
    )


def _carrier_authority_rank(authority: CarrierSurface) -> tuple[object, ...]:
    module_parts = tuple(part.lower() for part in authority.module_name.split("."))
    path_parts = tuple(part.lower() for part in Path(authority.file_path).parts)
    location_parts = (*module_parts, *path_parts)
    shared_module = bool(
        set(location_parts)
        & {
            "common",
            "core",
            "model",
            "models",
            "schema",
            "schemas",
            "semantic",
            "semantics",
            "shared",
        }
    )
    return (
        not shared_module,
        -len(authority.role_names),
        authority.file_path,
        authority.line,
        authority.class_name,
    )


def _carrier_reuse_candidate(
    local: CarrierSurface,
    authority: CarrierSurface,
) -> AvailableCarrierReuseCandidate | None:
    if local.file_path == authority.file_path:
        return None
    if not _carrier_surfaces_share_package(local, authority):
        return None
    if _carrier_surface_related(local, authority):
        return None
    if _carrier_uses_authority(local, authority):
        return None
    if _carrier_surfaces_share_nominal_ancestor(local, authority):
        return None
    if _looks_like_reusable_carrier_name(local.class_name) and (
        _carrier_authority_rank(local) <= _carrier_authority_rank(authority)
    ):
        return None

    shared_roles = sorted_tuple(set(local.role_names) & set(authority.role_names))
    if len(shared_roles) < _MIN_CARRIER_ROLE_OVERLAP:
        return None
    authority_coverage = len(shared_roles) / max(len(authority.role_names), 1)
    if authority_coverage < _MIN_CARRIER_AUTHORITY_COVERAGE:
        return None
    local_coverage = len(shared_roles) / max(len(local.role_names), 1)
    if local_coverage < _MIN_CARRIER_LOCAL_COVERAGE:
        return None
    shared_field_names = _shared_carrier_field_names(local, authority)
    if len(shared_field_names) < _MIN_CARRIER_SHARED_FIELD_MATCHES:
        return None
    return AvailableCarrierReuseCandidate(
        local=local,
        authority=authority,
        shared_roles=shared_roles,
        shared_field_names=shared_field_names,
    )


def _available_carrier_reuse_candidates_from_surfaces(
    surfaces: tuple[CarrierSurface, ...],
) -> tuple[AvailableCarrierReuseCandidate, ...]:
    authorities = _carrier_authority_surfaces(surfaces)
    if not authorities:
        return ()

    authority_indexes_by_role: dict[str, set[int]] = defaultdict(set)
    for authority_index, authority in enumerate(authorities):
        for role_name in set(authority.role_names):
            authority_indexes_by_role[role_name].add(authority_index)

    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableCarrierReuseCandidate]
    ] = defaultdict(list)
    for local in surfaces:
        candidate_authority_indexes: set[int] = set()
        for shared_role_floor in combinations(
            sorted(set(local.role_names)), _MIN_CARRIER_ROLE_OVERLAP
        ):
            indexed_authorities = [
                authority_indexes_by_role[role_name]
                for role_name in shared_role_floor
                if role_name in authority_indexes_by_role
            ]
            if len(indexed_authorities) != _MIN_CARRIER_ROLE_OVERLAP:
                continue
            candidate_authority_indexes.update(set.intersection(*indexed_authorities))
        for authority_index in sorted(candidate_authority_indexes):
            authority = authorities[authority_index]
            candidate = _carrier_reuse_candidate(local, authority)
            if candidate is not None:
                candidates_by_local[
                    (local.file_path, local.line, local.class_name)
                ].append(candidate)

    return _selected_available_carrier_reuse_candidates(candidates_by_local)


def _selected_available_carrier_reuse_candidates(
    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableCarrierReuseCandidate]
    ],
) -> tuple[AvailableCarrierReuseCandidate, ...]:

    selected = []
    for candidates in candidates_by_local.values():
        selected.append(
            sorted(
                candidates,
                key=lambda candidate: (
                    -len(candidate.shared_roles),
                    -len(candidate.shared_field_names),
                    len(candidate.authority.role_names) - len(candidate.shared_roles),
                    _carrier_authority_rank(candidate.authority),
                ),
            )[0]
        )
    return sorted_tuple(
        selected,
        key=lambda candidate: (
            candidate.local.file_path,
            candidate.local.line,
            candidate.local.class_name,
            candidate.authority.class_name,
        ),
    )


def _compact_available_carrier_reuse_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[AvailableCarrierReuseCandidate, ...]:
    return _available_carrier_reuse_candidates_from_surfaces(
        _carrier_surfaces_with_ancestors(_compact_carrier_surfaces(projections))
    )


class AvailableCarrierReuseDetector(
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        AvailableCarrierReuseCandidate,
    ]
):
    module_projection_family = CompactModuleClassProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Local carrier should reuse an available nominal carrier",
        "A record or context class repeats the field-role surface of an existing carrier in the same package. The docs prefer reusing the existing nominal carrier, or extending it through inheritance/composition, before adding another parallel class.",
        "reuse of an existing nominal carrier instead of a parallel field surface",
        "class field-role overlap with an available carrier authority",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[AvailableCarrierReuseCandidate]:
        del config
        return _compact_available_carrier_reuse_candidates(projections)

    def _findings_for_candidates(
        self,
        candidates: Sequence[AvailableCarrierReuseCandidate],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in candidates:
            role_summary = ", ".join(candidate.shared_roles)
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.local.class_name}` repeats carrier roles "
                        f"({role_summary}) already represented by "
                        f"`{candidate.authority.class_name}`."
                    ),
                    (
                        SourceLocation(
                            candidate.local.file_path,
                            candidate.local.line,
                            candidate.local.class_name,
                        ),
                        SourceLocation(
                            candidate.authority.file_path,
                            candidate.authority.line,
                            candidate.authority.class_name,
                        ),
                    ),
                    scaffold=(
                        f"# Reuse `{candidate.authority.class_name}` for roles: "
                        f"{role_summary}.\n"
                        "# Keep only fields that are genuinely local residue on "
                        f"`{candidate.local.class_name}`."
                    ),
                    codemod_patch=(
                        f"# Replace overlapping fields on `{candidate.local.class_name}` "
                        f"with `{candidate.authority.class_name}` through inheritance or "
                        "a single carrier field.\n"
                        "# Do not duplicate the shared nominal surface across modules."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=2,
                        mapping_name="available_carrier_reuse",
                        field_names=candidate.shared_roles,
                        source_name=candidate.authority.class_name,
                        identity_field_names=tuple(
                            candidate.shared_field_names or candidate.shared_roles
                        ),
                    ),
                )
            )
        return findings


__all__ = tuple(name for name in globals() if not name.startswith("_"))
