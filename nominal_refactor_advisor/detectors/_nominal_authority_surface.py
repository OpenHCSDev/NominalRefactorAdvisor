"""Nominal authority surface graph detection helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from ..class_index import CompactModuleClassProjection
from ..semantic_algebra import FiniteAxisSystem
from ._base import (
    DuplicateNominalAuthoritySurfaceCandidate,
    NominalAuthorityShape,
)
from ._helpers import _semantic_role_names_for_fields


@dataclass(frozen=True)
class _NominalAuthoritySurfaceNode:
    shape: NominalAuthorityShape
    field_roles: tuple[str, ...]
    public_method_names: tuple[str, ...]
    method_flow_roles: tuple[tuple[str, tuple[str, ...]], ...]
    constructed_delegate_names: tuple[str, ...]

    @property
    def class_name(self) -> str:
        return self.shape.class_name

    @property
    def file_path(self) -> str:
        return self.shape.file_path

    @property
    def line(self) -> int:
        return self.shape.line


def _surface_nodes_with_ancestors(
    nodes: tuple[_NominalAuthoritySurfaceNode, ...],
) -> tuple[_NominalAuthoritySurfaceNode, ...]:
    base_lookup: defaultdict[str, set[str]] = defaultdict(set)
    for surface_node in nodes:
        base_lookup[surface_node.class_name].update(
            surface_node.shape.declared_base_names
        )

    def ancestors_for(class_name: str) -> tuple[str, ...]:
        if class_name in base_lookup:
            stack = list(base_lookup[class_name])
        else:
            stack = []
        seen: set[str] = set()
        while stack:
            base_name = stack.pop()
            if base_name in seen or base_name == class_name:
                continue
            seen.add(base_name)
            if base_name in base_lookup:
                stack.extend(sorted(base_lookup[base_name] - seen))
        return tuple(sorted(seen))

    return tuple(
        _NominalAuthoritySurfaceNode(
            shape=NominalAuthorityShape(
                file_path=surface_node.file_path,
                class_name=surface_node.class_name,
                line=surface_node.line,
                declared_base_names=surface_node.shape.declared_base_names,
                ancestor_names=ancestors_for(surface_node.class_name),
                field_names=surface_node.shape.field_names,
                field_type_map=surface_node.shape.field_type_map,
                method_names=surface_node.shape.method_names,
                is_abstract=surface_node.shape.is_abstract,
                is_dataclass_family=surface_node.shape.is_dataclass_family,
            ),
            field_roles=surface_node.field_roles,
            public_method_names=surface_node.public_method_names,
            method_flow_roles=surface_node.method_flow_roles,
            constructed_delegate_names=surface_node.constructed_delegate_names,
        )
        for surface_node in nodes
    )


def _compact_nominal_authority_surface_nodes(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[_NominalAuthoritySurfaceNode, ...]:
    shapes_by_location = {
        (shape.file_path, shape.line, shape.class_name): shape
        for projection in projections
        for shape in projection.nominal_authority_shapes
    }
    known_class_names = frozenset(
        (
            *(
                indexed_class.simple_name
                for projection in projections
                for indexed_class in projection.classes
            ),
            *(
                class_name
                for projection in projections
                for class_name, _ in projection.extra_nominal_class_bases
            ),
        )
    )
    nodes: list[_NominalAuthoritySurfaceNode] = []
    for projection in projections:
        for fact in projection.nominal_surface_facts:
            shape = shapes_by_location[(fact.file_path, fact.line, fact.class_name)]
            field_names = tuple(name for name, _ in shape.field_type_map)
            nodes.append(
                _NominalAuthoritySurfaceNode(
                    shape=NominalAuthorityShape(
                        file_path=shape.file_path,
                        class_name=shape.class_name,
                        line=shape.line,
                        declared_base_names=shape.declared_base_names,
                        ancestor_names=(),
                        field_names=field_names,
                        field_type_map=shape.field_type_map,
                        method_names=fact.public_method_names,
                        is_abstract=shape.is_abstract,
                        is_dataclass_family=shape.is_dataclass,
                    ),
                    field_roles=_semantic_role_names_for_fields(field_names),
                    public_method_names=fact.public_method_names,
                    method_flow_roles=tuple(
                        sorted(
                            (
                                method_name,
                                _semantic_role_names_for_fields(field_names),
                            )
                            for method_name, field_names in fact.method_flow_field_names
                        )
                    ),
                    constructed_delegate_names=tuple(
                        name
                        for name in fact.constructed_delegate_candidate_names
                        if name in known_class_names
                    ),
                )
            )
    return _surface_nodes_with_ancestors(tuple(nodes))


class SurfaceNodesRelatedAuthority:
    def related(
        self,
        left: _NominalAuthoritySurfaceNode,
        right: _NominalAuthoritySurfaceNode,
    ) -> bool:
        return (
            left.class_name == right.class_name
            or left.class_name in set(right.shape.ancestor_names)
            or right.class_name in set(left.shape.ancestor_names)
        )


SURFACE_NODES_RELATED_AUTHORITY = SurfaceNodesRelatedAuthority()


def _shared_surface_roles(
    left: _NominalAuthoritySurfaceNode,
    right: _NominalAuthoritySurfaceNode,
) -> tuple[str, ...]:
    return tuple(sorted(set(left.field_roles) & set(right.field_roles)))


def _shared_surface_methods(
    left: _NominalAuthoritySurfaceNode,
    right: _NominalAuthoritySurfaceNode,
) -> tuple[str, ...]:
    return tuple(sorted(set(left.public_method_names) & set(right.public_method_names)))


def _direct_duplicate_nominal_authority_surface_candidates(
    nodes: tuple[_NominalAuthoritySurfaceNode, ...],
) -> tuple[DuplicateNominalAuthoritySurfaceCandidate, ...]:
    nodes_by_name: defaultdict[str, list[_NominalAuthoritySurfaceNode]] = defaultdict(
        list
    )
    for node in nodes:
        nodes_by_name[node.class_name].append(node)

    candidates: list[DuplicateNominalAuthoritySurfaceCandidate] = []
    for shell in nodes:
        if shell.shape.is_abstract:
            continue
        for delegate_name in shell.constructed_delegate_names:
            if delegate_name not in nodes_by_name:
                continue
            for authority in nodes_by_name[delegate_name]:
                if SURFACE_NODES_RELATED_AUTHORITY.related(shell, authority):
                    continue
                shared_roles = _shared_surface_roles(shell, authority)
                shared_methods = _shared_surface_methods(shell, authority)
                if len(shared_roles) < 2 or not shared_methods:
                    continue
                candidates.append(
                    DuplicateNominalAuthoritySurfaceCandidate(
                        file_path=shell.file_path,
                        line=shell.line,
                        subject_name=shell.class_name,
                        name_family=shared_roles,
                        authority_file_path=authority.file_path,
                        authority_name=authority.class_name,
                        authority_line=authority.line,
                        duplicate_class_names=(shell.class_name,),
                        duplicate_line_numbers=(shell.line,),
                        shared_method_names=shared_methods,
                        detection_kind="delegate_construction",
                    )
                )
    return tuple(candidates)


def _preferred_surface_authority(
    component: tuple[_NominalAuthoritySurfaceNode, ...],
) -> _NominalAuthoritySurfaceNode:
    return sorted(
        component,
        key=lambda node: (
            bool(node.constructed_delegate_names),
            node.shape.is_abstract,
            -len(node.public_method_names),
            -len(node.field_roles),
            node.class_name,
        ),
    )[0]


def _surface_confusability_components(
    nodes: tuple[_NominalAuthoritySurfaceNode, ...],
) -> tuple[tuple[_NominalAuthoritySurfaceNode, ...], ...]:
    """Return the exact axis-equality graph components without clique edges."""

    axis_system = FiniteAxisSystem.from_rows(
        (
            (
                node,
                {
                    "field_roles": node.field_roles,
                    "method_names": node.public_method_names,
                    "method_flow_roles": node.method_flow_roles,
                },
            )
            for node in nodes
        )
    )
    return axis_system.confusability_components(
        (
            ("field_roles", "method_names"),
            ("field_roles", "method_flow_roles"),
        )
    )


def _component_duplicate_nominal_authority_surface_candidates(
    nodes: tuple[_NominalAuthoritySurfaceNode, ...],
) -> tuple[DuplicateNominalAuthoritySurfaceCandidate, ...]:
    if len(nodes) < 3:
        return ()

    candidates: list[DuplicateNominalAuthoritySurfaceCandidate] = []
    for component in _surface_confusability_components(nodes):
        if len(component) < 3:
            continue
        if any(
            SURFACE_NODES_RELATED_AUTHORITY.related(left, right)
            for left, right in combinations(component, 2)
        ):
            continue
        shared_roles = tuple(
            sorted(set.intersection(*(set(node.field_roles) for node in component)))
        )
        shared_methods = tuple(
            sorted(
                set.intersection(*(set(node.public_method_names) for node in component))
            )
        )
        if len(shared_roles) < 2 or not shared_methods:
            continue
        authority = _preferred_surface_authority(component)
        duplicates = tuple(node for node in component if node is not authority)
        if len(duplicates) < 2:
            continue
        candidates.append(
            DuplicateNominalAuthoritySurfaceCandidate(
                file_path=authority.file_path,
                line=authority.line,
                subject_name=authority.class_name,
                name_family=shared_roles,
                authority_file_path=authority.file_path,
                authority_name=authority.class_name,
                authority_line=authority.line,
                duplicate_class_names=tuple(node.class_name for node in duplicates),
                duplicate_line_numbers=tuple(node.line for node in duplicates),
                shared_method_names=shared_methods,
                detection_kind="field_flow_confusability_component",
            )
        )
    return tuple(candidates)


def _compact_duplicate_nominal_authority_surface_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
) -> tuple[DuplicateNominalAuthoritySurfaceCandidate, ...]:
    return _duplicate_nominal_authority_surface_candidates_from_nodes(
        _compact_nominal_authority_surface_nodes(projections)
    )


def _duplicate_nominal_authority_surface_candidates_from_nodes(
    nodes: tuple[_NominalAuthoritySurfaceNode, ...],
) -> tuple[DuplicateNominalAuthoritySurfaceCandidate, ...]:
    candidates = (
        *_direct_duplicate_nominal_authority_surface_candidates(nodes),
        *_component_duplicate_nominal_authority_surface_candidates(nodes),
    )
    deduped: dict[
        tuple[str, str, tuple[str, ...], tuple[str, ...]],
        DuplicateNominalAuthoritySurfaceCandidate,
    ] = {}
    for candidate in candidates:
        key = (
            candidate.authority_name,
            candidate.detection_kind,
            candidate.duplicate_class_names,
            candidate.name_family,
        )
        if key not in deduped:
            deduped[key] = candidate
    return tuple(
        sorted(
            deduped.values(),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.authority_name,
                candidate.duplicate_class_names,
            ),
        )
    )
