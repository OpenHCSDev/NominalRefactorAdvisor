"""Nominal authority surface graph detection helpers."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence, TypeAlias

from ..class_index import CompactModuleClassProjection
from ._base import (
    DuplicateNominalAuthoritySurfaceCandidate,
    NominalAuthorityShape,
    ParsedModule,
)
from ._helpers import (
    CLASS_NODE_AUTHORITY,
    HELPER_SYNTAX_PROJECTION_AUTHORITY,
    _is_dataclass_class,
    _semantic_role_names_for_fields,
    _walk_nodes,
    name_id,
)

SurfaceMethodNodes: TypeAlias = tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]


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


def _public_surface_methods(node: ast.ClassDef) -> SurfaceMethodNodes:
    return tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not statement.name.startswith("_")
    )


def _self_attribute_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                current.attr
                for current in _walk_nodes(node)
                if isinstance(current, ast.Attribute)
                and isinstance(current.value, ast.Name)
                and current.value.id == "self"
            }
        )
    )


def _method_flow_roles(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...]]:
    return (
        method.name,
        _semantic_role_names_for_fields(_self_attribute_names(method)),
    )


def _call_self_attribute_names(call: ast.Call) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                current.attr
                for argument in (
                    *call.args,
                    *(keyword.value for keyword in call.keywords),
                )
                for current in _walk_nodes(argument)
                if isinstance(current, ast.Attribute)
                and isinstance(current.value, ast.Name)
                and current.value.id == "self"
            }
        )
    )


def _constructed_delegate_names(
    methods: SurfaceMethodNodes,
    known_class_names: frozenset[str],
) -> tuple[str, ...]:
    delegate_names: set[str] = set()
    for method in methods:
        for call in _walk_nodes(method):
            if not isinstance(call, ast.Call):
                continue
            callee_name = name_id(call.func)
            if callee_name is None or callee_name not in known_class_names:
                continue
            if len(_call_self_attribute_names(call)) < 2:
                continue
            delegate_names.add(callee_name)
    return tuple(sorted(delegate_names))


def _nominal_authority_surface_nodes(
    modules: Sequence[ParsedModule],
) -> tuple[_NominalAuthoritySurfaceNode, ...]:
    class_records: list[tuple[ParsedModule, ast.ClassDef]] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if isinstance(node, ast.ClassDef):
                class_records.append((module, node))
    known_class_names = frozenset(node.name for _, node in class_records)

    nodes: list[_NominalAuthoritySurfaceNode] = []
    for module, node in class_records:
        typed_fields = HELPER_SYNTAX_PROJECTION_AUTHORITY.typed_field_map(node)
        field_names = tuple(name for name, _ in typed_fields)
        if len(field_names) < 2:
            continue
        methods = _public_surface_methods(node)
        public_method_names = tuple(sorted(method.name for method in methods))
        if not public_method_names:
            continue
        method_flow_roles = tuple(
            sorted(
                flow for method in methods if (flow := _method_flow_roles(method))[1]
            )
        )
        if not method_flow_roles:
            continue
        nodes.append(
            _NominalAuthoritySurfaceNode(
                shape=NominalAuthorityShape(
                    file_path=str(module.path),
                    class_name=node.name,
                    line=node.lineno,
                    declared_base_names=CLASS_NODE_AUTHORITY.declared_base_names(node),
                    ancestor_names=(),
                    field_names=field_names,
                    field_type_map=typed_fields,
                    method_names=public_method_names,
                    is_abstract=CLASS_NODE_AUTHORITY.is_abstract(node),
                    is_dataclass_family=_is_dataclass_class(node),
                ),
                field_roles=_semantic_role_names_for_fields(field_names),
                public_method_names=public_method_names,
                method_flow_roles=method_flow_roles,
                constructed_delegate_names=_constructed_delegate_names(
                    methods, known_class_names
                ),
            )
        )

    return _surface_nodes_with_ancestors(tuple(nodes))


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

    parents = list(range(len(nodes)))
    ranks = [0] * len(nodes)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1

    first_index_by_view: dict[tuple[object, ...], int] = {}
    for index, node in enumerate(nodes):
        for view_key in (
            ("methods", node.field_roles, node.public_method_names),
            ("flow", node.field_roles, node.method_flow_roles),
        ):
            first_index = first_index_by_view.setdefault(view_key, index)
            union(first_index, index)

    indices_by_root: dict[int, list[int]] = {}
    for index in range(len(nodes)):
        indices_by_root.setdefault(find(index), []).append(index)
    ordered_indices = sorted(indices_by_root.values(), key=lambda indices: indices[0])
    return tuple(
        tuple(nodes[index] for index in component_indices)
        for component_indices in ordered_indices
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


def _duplicate_nominal_authority_surface_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[DuplicateNominalAuthoritySurfaceCandidate, ...]:
    return _duplicate_nominal_authority_surface_candidates_from_nodes(
        _nominal_authority_surface_nodes(modules)
    )


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
