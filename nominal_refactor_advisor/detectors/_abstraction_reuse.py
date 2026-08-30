"""Generic detection for local reimplementation of available abstractions."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from tree_sitter import Node

from ..ast_tools import CollectedFamily, SourceModule
from ..class_index import (
    CompactCarrierClassFact,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from ..collection_algebra import sorted_tuple
from ..models import MappingMetrics
from ..native_syntax import NativePythonSyntaxIndex
from ..patterns import PatternId
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    DetectorConfig,
    ParsedModule,
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

_MIN_AUTHORITY_ATOMS = 7
_MIN_LOCAL_ATOMS = 6
_MIN_OVERLAP_ATOMS = 5
_MIN_OVERLAP_SCORE = 9
_MIN_AUTHORITY_COVERAGE = 0.40
_MIN_LOCAL_COVERAGE = 0.35
_MAX_FOCUSED_AUTHORITY_ATOMS = 32

_AUTHORITY_PATH_PARTS = frozenset(
    {
        "common",
        "component",
        "components",
        "factory",
        "factories",
        "scaffold",
        "scaffolds",
        "shared",
        "support",
        "utils",
    }
)
_AUTHORITY_NAME_SUFFIXES = (
    "Adapter",
    "Authority",
    "Base",
    "Builder",
    "Catalog",
    "Factory",
    "Formatter",
    "Manager",
    "Mixin",
    "Panel",
    "Parser",
    "Renderer",
    "Resolver",
    "Scaffold",
    "Strategy",
)
_HIGH_SIGNAL_ATOM_PREFIXES = ("construct:", "method:", "signal:", "store:", "control:")
_STRUCTURAL_ATOM_PREFIXES = ("construct:", "method:", "signal:", "store:", "control:")
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


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    atoms: frozenset[str]
    call_names: frozenset[str]

    @property
    def high_signal_atoms(self) -> frozenset[str]:
        return frozenset(
            atom for atom in self.atoms if atom.startswith(_HIGH_SIGNAL_ATOM_PREFIXES)
        )


def _looks_like_reusable_carrier_name(name: str) -> bool:
    return name.endswith(_CARRIER_NAME_SUFFIXES)


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


@dataclass(frozen=True, slots=True)
class SignatureBase(FilePathLineModuleNameBase):
    signature: CapabilitySignature
    symbol: str


@dataclass(frozen=True, slots=True)
class AbstractionAuthoritySignature(SignatureBase):
    name: str
    shared_path_authority: bool


@dataclass(frozen=True, slots=True)
class LocalImplementationSignature(SignatureBase):
    imported_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class AvailableAbstractionReuseCandidate:
    local: LocalImplementationSignature
    authority: AbstractionAuthoritySignature
    overlap_atoms: tuple[str, ...]
    overlap_score: int


@dataclass(frozen=True, slots=True)
class CompactAvailableAbstractionReuseModuleProjection:
    authorities: tuple[AbstractionAuthoritySignature, ...]
    locals: tuple[LocalImplementationSignature, ...]


class _CapabilityAtomVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.atoms: set[str] = set()
        self.call_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            self.atoms.add(f"param:{argument.arg}")
        if node.args.vararg is not None:
            self.atoms.add(f"param:{node.args.vararg.arg}")
        if node.args.kwarg is not None:
            self.atoms.add(f"param:{node.args.kwarg.arg}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_terminal_name(node.func)
        if call_name is not None:
            self.call_names.update(_call_reference_names(node.func))
            self.atoms.add(f"call:{call_name}")
            if _looks_like_constructor_name(call_name):
                self.atoms.add(f"construct:{call_name}")
        if isinstance(node.func, ast.Attribute):
            self.atoms.add(f"method:{node.func.attr}")
            if node.func.attr == "connect":
                signal_name = _terminal_name(node.func.value)
                if signal_name is not None:
                    self.atoms.add(f"signal:{signal_name}.connect")
        for keyword in node.keywords:
            if keyword.arg is not None:
                self.atoms.add(f"keyword:{keyword.arg}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_store_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_store_target(node.target)
        self.generic_visit(node)

    visit_AugAssign = visit_AnnAssign

    def visit_For(self, node: ast.For) -> None:
        self.atoms.add("control:for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_If(self, node: ast.If) -> None:
        self.atoms.add("control:if")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.atoms.add("control:try")
        self.generic_visit(node)

    def _record_store_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.atoms.add(f"store:{target.id}")
            return
        if isinstance(target, ast.Attribute):
            self.atoms.add(f"store:{target.attr}")
            return
        if isinstance(target, ast.Subscript):
            target_name = _terminal_name(target.value)
            if target_name is not None:
                self.atoms.add(f"store:{target_name}")
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._record_store_target(element)


class _LocalSignatureCollector(ast.NodeVisitor):
    def __init__(self, module: ParsedModule) -> None:
        self.module = module
        self.class_stack: list[str] = []
        self.locals: list[LocalImplementationSignature] = []
        self.imported_names = frozenset(_imported_local_names(module))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol = ".".join((*self.class_stack, node.name))
        signature = _signature_for_node(node)
        if len(signature.high_signal_atoms) >= _MIN_LOCAL_ATOMS:
            self.locals.append(
                LocalImplementationSignature(
                    file_path=str(self.module.path),
                    module_name=self.module.module_name,
                    line=node.lineno,
                    symbol=symbol,
                    signature=signature,
                    imported_names=self.imported_names,
                )
            )


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return None


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _call_terminal_name(node: ast.AST) -> str | None:
    return _terminal_name(node)


def _call_reference_names(node: ast.AST) -> frozenset[str]:
    chain = _attribute_chain(node)
    if chain is None:
        terminal = _call_terminal_name(node)
        return frozenset(() if terminal is None else (terminal,))
    names: set[str] = set(chain)
    for start in range(len(chain)):
        suffix = chain[start:]
        if len(suffix) > 1:
            names.add(".".join(suffix))
    return frozenset(names)


def _looks_like_constructor_name(name: str) -> bool:
    return bool(name) and name[:1].isupper()


@lru_cache(maxsize=32768)
def _signature_for_node(node: ast.AST) -> CapabilitySignature:
    visitor = _CapabilityAtomVisitor()
    visitor.visit(node)
    return CapabilitySignature(
        atoms=frozenset(visitor.atoms),
        call_names=frozenset(visitor.call_names),
    )


def _module_path_parts(module: ParsedModule) -> frozenset[str]:
    return frozenset(part.lower() for part in Path(module.path).with_suffix("").parts)


def _is_shared_authority_location(module: ParsedModule) -> bool:
    return bool(_module_path_parts(module) & _AUTHORITY_PATH_PARTS)


def _looks_like_reusable_authority_name(name: str) -> bool:
    return name.endswith(_AUTHORITY_NAME_SUFFIXES)


def _public_name(name: str) -> bool:
    return not name.startswith("_")


def _imported_local_names(module: ParsedModule) -> tuple[str, ...]:
    names: list[str] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Import):
            names.extend(
                alias.asname or alias.name.split(".", 1)[0] for alias in statement.names
            )
        elif isinstance(statement, ast.ImportFrom):
            names.extend(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name != "*"
            )
    return sorted_tuple(set(names))


def _class_method_nodes(
    node: ast.ClassDef,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _combined_class_signature(node: ast.ClassDef) -> CapabilitySignature:
    atoms: set[str] = set()
    call_names: set[str] = set()
    for method in _class_method_nodes(node):
        signature = _signature_for_node(method)
        atoms.update(signature.atoms)
        call_names.update(signature.call_names)
    return CapabilitySignature(frozenset(atoms), frozenset(call_names))


def _module_authorities(
    module: ParsedModule,
) -> tuple[AbstractionAuthoritySignature, ...]:
    shared_path_authority = _is_shared_authority_location(module)
    authorities: list[AbstractionAuthoritySignature] = []
    for statement in module.module.body:
        if isinstance(statement, ast.ClassDef):
            if not _public_name(statement.name):
                continue
            signature = _combined_class_signature(statement)
            if len(signature.high_signal_atoms) < _MIN_AUTHORITY_ATOMS:
                continue
            if len(signature.high_signal_atoms) > _MAX_FOCUSED_AUTHORITY_ATOMS:
                continue
            if not _looks_like_reusable_authority_name(statement.name):
                continue
            authorities.append(
                AbstractionAuthoritySignature(
                    file_path=str(module.path),
                    module_name=module.module_name,
                    line=statement.lineno,
                    name=statement.name,
                    symbol=statement.name,
                    signature=signature,
                    shared_path_authority=shared_path_authority,
                )
            )
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _public_name(statement.name):
                continue
            signature = _signature_for_node(statement)
            if len(signature.high_signal_atoms) < _MIN_AUTHORITY_ATOMS:
                continue
            if len(signature.high_signal_atoms) > _MAX_FOCUSED_AUTHORITY_ATOMS:
                continue
            if not _looks_like_reusable_authority_name(statement.name):
                continue
            authorities.append(
                AbstractionAuthoritySignature(
                    file_path=str(module.path),
                    module_name=module.module_name,
                    line=statement.lineno,
                    name=statement.name,
                    symbol=statement.name,
                    signature=signature,
                    shared_path_authority=shared_path_authority,
                )
            )
    return sorted_tuple(
        authorities,
        key=lambda authority: (authority.file_path, authority.line, authority.name),
    )


def _module_locals(module: ParsedModule) -> tuple[LocalImplementationSignature, ...]:
    collector = _LocalSignatureCollector(module)
    collector.visit(module.module)
    return sorted_tuple(
        collector.locals,
        key=lambda local: (local.file_path, local.line, local.symbol),
    )


class CompactAvailableAbstractionReuseModuleProjectionFamily(
    CollectedFamily[CompactAvailableAbstractionReuseModuleProjection]
):
    item_type = CompactAvailableAbstractionReuseModuleProjection
    cache_payload_max_bytes = 1_000_000

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactAvailableAbstractionReuseModuleProjection]:
        del cls
        return [
            CompactAvailableAbstractionReuseModuleProjection(
                authorities=_module_authorities(parsed_module),
                locals=_module_locals(parsed_module),
            )
        ]


@dataclass(frozen=True)
class CompactAvailableAbstractionReuseProjectionDemand:
    """Target signatures that can participate in one report-scoped candidate."""

    authorities: tuple[AbstractionAuthoritySignature, ...]
    locals: tuple[LocalImplementationSignature, ...]


def _available_abstraction_reuse_projection_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactAvailableAbstractionReuseProjectionDemand:
    del config
    projections = tuple(
        item
        for item in target_items
        if isinstance(item, CompactAvailableAbstractionReuseModuleProjection)
    )
    return CompactAvailableAbstractionReuseProjectionDemand(
        authorities=tuple(
            authority
            for projection in projections
            for authority in projection.authorities
        ),
        locals=tuple(
            local for projection in projections for local in projection.locals
        ),
    )


def _project_available_abstraction_reuse_demand(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, CompactAvailableAbstractionReuseProjectionDemand):
        return items
    projected: list[CompactAvailableAbstractionReuseModuleProjection] = []
    for item in items:
        if not isinstance(item, CompactAvailableAbstractionReuseModuleProjection):
            continue
        authorities = tuple(
            authority
            for authority in item.authorities
            if any(
                _reimplements_authority(local, authority) is not None
                for local in demand.locals
            )
        )
        locals = tuple(
            local
            for local in item.locals
            if any(
                _reimplements_authority(local, authority) is not None
                for authority in demand.authorities
            )
        )
        if authorities or locals:
            projected.append(
                CompactAvailableAbstractionReuseModuleProjection(
                    authorities=authorities,
                    locals=locals,
                )
            )
    return tuple(projected)


def _collect_available_abstraction_reuse_ast_demand(
    parsed_module: ParsedModule,
    demand: object,
) -> list[object]:
    return list(
        _project_available_abstraction_reuse_demand(
            tuple(
                CompactAvailableAbstractionReuseModuleProjectionFamily.collect(
                    parsed_module
                )
            ),
            demand,
        )
    )


def _native_capability_terminal_name(
    syntax_index: NativePythonSyntaxIndex,
    node: Node | None,
) -> str | None:
    if node is None:
        return None
    if node.type == "identifier":
        return syntax_index.source_for(node).decode("utf-8")
    if node.type == "attribute":
        attribute = node.child_by_field_name("attribute")
        return (
            None
            if attribute is None
            else syntax_index.source_for(attribute).decode("utf-8")
        )
    if node.type == "subscript":
        return _native_capability_terminal_name(
            syntax_index,
            node.child_by_field_name("value"),
        )
    return None


def _native_capability_store_atoms(
    syntax_index: NativePythonSyntaxIndex,
    node: Node | None,
) -> set[str]:
    if node is None:
        return set()
    if node.type in {"identifier", "attribute", "subscript"}:
        name = _native_capability_terminal_name(syntax_index, node)
        return set() if name is None else {f"store:{name}"}
    if node.type in {
        "tuple",
        "list",
        "pattern_list",
        "tuple_pattern",
        "list_pattern",
    }:
        return set().union(
            *(
                _native_capability_store_atoms(syntax_index, child)
                for child in node.named_children
            )
        )
    return set()


def _native_capability_high_signal_atoms(
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
) -> frozenset[str]:
    """Project the exact high-signal subset used by the overlap gate."""

    atoms: set[str] = set()
    stack = [function_node]
    while stack:
        node = stack.pop()
        stack.extend(node.named_children)
        if node.type == "call":
            function = node.child_by_field_name("function")
            name = _native_capability_terminal_name(syntax_index, function)
            if name and name[:1].isupper():
                atoms.add(f"construct:{name}")
            if function is not None and function.type == "attribute":
                atoms.add(f"method:{name}")
                if name == "connect":
                    signal_name = _native_capability_terminal_name(
                        syntax_index,
                        function.child_by_field_name("object"),
                    )
                    if signal_name:
                        atoms.add(f"signal:{signal_name}.connect")
            continue
        if node.type in {"assignment", "augmented_assignment"}:
            atoms.update(
                _native_capability_store_atoms(
                    syntax_index,
                    node.child_by_field_name("left"),
                )
            )
        elif node.type == "for_statement":
            atoms.add("control:for")
        elif node.type == "if_statement":
            atoms.add("control:if")
        elif node.type == "try_statement":
            atoms.add("control:try")
    return frozenset(atoms)


def _native_available_abstraction_imported_names(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> frozenset[str]:
    statements = [
        syntax_index.statement_for(node)
        for node in syntax_index.tree.root_node.named_children
        if node.type in {"import_statement", "import_from_statement"}
    ]
    module = source_module.parsed_module(
        ast.Module(body=statements, type_ignores=[]),
    )
    return frozenset(_imported_local_names(module))


def _collect_available_abstraction_reuse_source_demand(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    if not isinstance(demand, CompactAvailableAbstractionReuseProjectionDemand):
        raise TypeError("available-abstraction demand has the wrong authority type")
    if not syntax_index.is_complete:
        return None
    imported_names = _native_available_abstraction_imported_names(
        source_module,
        syntax_index,
    )
    module_stub = source_module.parsed_module(
        ast.Module(body=[], type_ignores=[]),
    )
    shared_path_authority = _is_shared_authority_location(module_stub)
    package_name = source_module.module_name.split(".", 1)[0]
    available_target_authorities = tuple(
        authority
        for authority in demand.authorities
        if authority.name in imported_names
        or (
            authority.shared_path_authority
            and authority.module_name.split(".", 1)[0] == package_name
        )
    )
    target_local_imports = frozenset(
        name for local in demand.locals for name in local.imported_names
    )
    shared_context_authority = shared_path_authority and any(
        local.module_name.split(".", 1)[0] == package_name
        for local in demand.locals
    )
    functions = tuple(
        sorted(
            syntax_index.common_captures().get("function", ()),
            key=lambda node: (node.start_byte, -node.end_byte),
        )
    )
    parsed_functions: dict[Node, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def parsed_function(node: Node) -> ast.FunctionDef | ast.AsyncFunctionDef:
        function = parsed_functions.get(node)
        if function is None:
            function = syntax_index.function_for(node)
            parsed_functions[node] = function
        return function

    locals_: list[LocalImplementationSignature] = []
    authorities: list[AbstractionAuthoritySignature] = []
    for function_node in functions:
        scopes = syntax_index.named_scope_nodes(function_node)
        if any(scope.type == "function_definition" for scope in scopes):
            continue
        direct_class = syntax_index.direct_enclosing_class(function_node)
        top_level = function_node.parent == syntax_index.tree.root_node or (
            function_node.parent is not None
            and function_node.parent.type == "decorated_definition"
            and function_node.parent.parent == syntax_index.tree.root_node
        )
        if not top_level and direct_class is None:
            continue
        declared_name = syntax_index.declared_name(function_node)
        possible_authority = (
            top_level
            and _public_name(declared_name)
            and _looks_like_reusable_authority_name(declared_name)
            and (shared_context_authority or declared_name in target_local_imports)
        )
        if not available_target_authorities and not possible_authority:
            continue
        coarse_atoms = _native_capability_high_signal_atoms(
            syntax_index,
            function_node,
        )
        class_names = tuple(
            syntax_index.declared_name(scope)
            for scope in scopes
            if scope.type == "class_definition"
        )
        symbol = ".".join((*class_names, declared_name))
        coarse_local = LocalImplementationSignature(
            file_path=str(source_module.path),
            line=function_node.start_point.row + 1,
            module_name=source_module.module_name,
            signature=CapabilitySignature(coarse_atoms, frozenset()),
            symbol=symbol,
            imported_names=imported_names,
        )
        can_be_local = len(coarse_atoms) >= _MIN_LOCAL_ATOMS and any(
            _reimplements_authority_from_atoms(
                coarse_local,
                authority,
                coarse_atoms,
                authority.signature.high_signal_atoms,
            )
            is not None
            for authority in available_target_authorities
        )
        coarse_authority = AbstractionAuthoritySignature(
            file_path=str(source_module.path),
            line=function_node.start_point.row + 1,
            module_name=source_module.module_name,
            signature=CapabilitySignature(coarse_atoms, frozenset()),
            symbol=declared_name,
            name=declared_name,
            shared_path_authority=shared_path_authority,
        )
        can_be_authority = (
            possible_authority
            and _MIN_AUTHORITY_ATOMS
            <= len(coarse_atoms)
            <= _MAX_FOCUSED_AUTHORITY_ATOMS
            and any(
                _reimplements_authority_from_atoms(
                    local,
                    coarse_authority,
                    local.signature.high_signal_atoms,
                    coarse_atoms,
                )
                is not None
                for local in demand.locals
            )
        )
        if not can_be_local and not can_be_authority:
            continue
        function = parsed_function(function_node)
        signature = _signature_for_node(function)
        if can_be_local and len(signature.high_signal_atoms) >= _MIN_LOCAL_ATOMS:
            locals_.append(
                LocalImplementationSignature(
                    file_path=str(source_module.path),
                    line=function.lineno,
                    module_name=source_module.module_name,
                    signature=signature,
                    symbol=symbol,
                    imported_names=imported_names,
                )
            )
        if (
            can_be_authority
            and _MIN_AUTHORITY_ATOMS
            <= len(signature.high_signal_atoms)
            <= _MAX_FOCUSED_AUTHORITY_ATOMS
        ):
            authorities.append(
                AbstractionAuthoritySignature(
                    file_path=str(source_module.path),
                    line=function.lineno,
                    module_name=source_module.module_name,
                    signature=signature,
                    symbol=declared_name,
                    name=declared_name,
                    shared_path_authority=shared_path_authority,
                )
            )
    for class_node in syntax_index.top_level_declarations("class"):
        name = syntax_index.declared_name(class_node)
        if (
            not _public_name(name)
            or not _looks_like_reusable_authority_name(name)
            or (not shared_context_authority and name not in target_local_imports)
        ):
            continue
        method_nodes = tuple(
            function
            for function in functions
            if syntax_index.direct_enclosing_class(function) == class_node
        )
        coarse_atoms = frozenset().union(
            *(
                _native_capability_high_signal_atoms(syntax_index, method)
                for method in method_nodes
            )
        )
        coarse_authority = AbstractionAuthoritySignature(
            file_path=str(source_module.path),
            line=class_node.start_point.row + 1,
            module_name=source_module.module_name,
            signature=CapabilitySignature(coarse_atoms, frozenset()),
            symbol=name,
            name=name,
            shared_path_authority=shared_path_authority,
        )
        if not (
            _MIN_AUTHORITY_ATOMS
            <= len(coarse_atoms)
            <= _MAX_FOCUSED_AUTHORITY_ATOMS
            and any(
                _reimplements_authority_from_atoms(
                    local,
                    coarse_authority,
                    local.signature.high_signal_atoms,
                    coarse_atoms,
                )
                is not None
                for local in demand.locals
            )
        ):
            continue
        atoms: set[str] = set()
        call_names: set[str] = set()
        for method in method_nodes:
            signature = _signature_for_node(parsed_function(method))
            atoms.update(signature.atoms)
            call_names.update(signature.call_names)
        signature = CapabilitySignature(frozenset(atoms), frozenset(call_names))
        if not (
            _MIN_AUTHORITY_ATOMS
            <= len(signature.high_signal_atoms)
            <= _MAX_FOCUSED_AUTHORITY_ATOMS
        ):
            continue
        authorities.append(
            AbstractionAuthoritySignature(
                file_path=str(source_module.path),
                line=class_node.start_point.row + 1,
                module_name=source_module.module_name,
                signature=signature,
                symbol=name,
                name=name,
                shared_path_authority=shared_path_authority,
            )
        )
    projected = _project_available_abstraction_reuse_demand(
        (
            CompactAvailableAbstractionReuseModuleProjection(
                authorities=sorted_tuple(
                    authorities,
                    key=lambda authority: (
                        authority.file_path,
                        authority.line,
                        authority.name,
                    ),
                ),
                locals=sorted_tuple(
                    locals_,
                    key=lambda local: (
                        local.file_path,
                        local.line,
                        local.symbol,
                    ),
                ),
            ),
        ),
        demand,
    )
    return list(projected)


CompactAvailableAbstractionReuseModuleProjectionFamily.report_demand_builder = (
    staticmethod(_available_abstraction_reuse_projection_demand)
)
CompactAvailableAbstractionReuseModuleProjectionFamily.ast_demand_collector = (
    staticmethod(_collect_available_abstraction_reuse_ast_demand)
)
CompactAvailableAbstractionReuseModuleProjectionFamily.source_demand_collector = (
    staticmethod(_collect_available_abstraction_reuse_source_demand)
)
CompactAvailableAbstractionReuseModuleProjectionFamily.cached_demand_projector = (
    staticmethod(_project_available_abstraction_reuse_demand)
)


def _top_level_package(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _authority_available_to_local(
    authority: AbstractionAuthoritySignature, local: LocalImplementationSignature
) -> bool:
    if authority.name in local.imported_names:
        return True
    if not authority.shared_path_authority:
        return False
    return _top_level_package(authority.module_name) == _top_level_package(
        local.module_name
    )


def _structural_overlap(atoms: Iterable[str]) -> tuple[str, ...]:
    return sorted_tuple(
        atom for atom in atoms if atom.startswith(_STRUCTURAL_ATOM_PREFIXES)
    )


def _overlap_score(atoms: Sequence[str]) -> int:
    score = 0
    for atom in atoms:
        if atom.startswith("construct:"):
            score += 3
        elif atom.startswith(("method:", "signal:", "store:")):
            score += 2
        elif atom.startswith("control:"):
            score += 1
        else:
            score += 1
    return score


def _local_declares_authority_name(
    local: LocalImplementationSignature, authority: AbstractionAuthoritySignature
) -> bool:
    return local.symbol == authority.name or local.symbol.startswith(
        f"{authority.name}."
    )


def _reimplements_authority(
    local: LocalImplementationSignature, authority: AbstractionAuthoritySignature
) -> AvailableAbstractionReuseCandidate | None:
    return _reimplements_authority_from_atoms(
        local,
        authority,
        local.signature.high_signal_atoms,
        authority.signature.high_signal_atoms,
    )


def _reimplements_authority_from_atoms(
    local: LocalImplementationSignature,
    authority: AbstractionAuthoritySignature,
    local_atoms: frozenset[str],
    authority_atoms: frozenset[str],
) -> AvailableAbstractionReuseCandidate | None:
    if local.file_path == authority.file_path:
        return None
    if not _authority_available_to_local(authority, local):
        return None
    if (
        authority.name in local.signature.call_names
        and not _local_declares_authority_name(local, authority)
    ):
        return None
    overlap = local_atoms & authority_atoms
    if len(overlap) < _MIN_OVERLAP_ATOMS:
        return None
    authority_coverage = len(overlap) / max(len(authority_atoms), 1)
    if authority_coverage < _MIN_AUTHORITY_COVERAGE:
        return None
    local_coverage = len(overlap) / max(len(local_atoms), 1)
    if local_coverage < _MIN_LOCAL_COVERAGE:
        return None
    structural_overlap = _structural_overlap(overlap)
    if len(structural_overlap) < _MIN_OVERLAP_ATOMS:
        return None
    if not any(atom.startswith("construct:") for atom in structural_overlap):
        if (
            len(
                tuple(
                    atom
                    for atom in structural_overlap
                    if atom.startswith(("method:", "signal:", "store:"))
                )
            )
            < 4
        ):
            return None
    score = _overlap_score(structural_overlap)
    if score < _MIN_OVERLAP_SCORE:
        return None
    return AvailableAbstractionReuseCandidate(
        local=local,
        authority=authority,
        overlap_atoms=structural_overlap,
        overlap_score=score,
    )


def _available_abstraction_reuse_candidates_from_signatures(
    authorities: Sequence[AbstractionAuthoritySignature],
    local_signatures: Sequence[LocalImplementationSignature],
) -> tuple[AvailableAbstractionReuseCandidate, ...]:
    if not authorities:
        return ()
    authority_tuple = tuple(authorities)
    authority_atoms = tuple(
        authority.signature.high_signal_atoms for authority in authority_tuple
    )
    authority_indexes_by_name: dict[str, list[int]] = defaultdict(list)
    shared_authority_indexes_by_package: dict[str, list[int]] = defaultdict(list)
    for authority_index, authority in enumerate(authority_tuple):
        authority_indexes_by_name[authority.name].append(authority_index)
        if authority.shared_path_authority:
            shared_authority_indexes_by_package[
                _top_level_package(authority.module_name)
            ].append(authority_index)
    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableAbstractionReuseCandidate]
    ] = defaultdict(list)
    for local in local_signatures:
        local_atoms = local.signature.high_signal_atoms
        available_authority_indexes = {
            authority_index
            for imported_name in local.imported_names
            for authority_index in authority_indexes_by_name.get(imported_name, ())
        }
        available_authority_indexes.update(
            shared_authority_indexes_by_package.get(
                _top_level_package(local.module_name),
                (),
            )
        )
        for authority_index in sorted(available_authority_indexes):
            if len(local_atoms & authority_atoms[authority_index]) < _MIN_OVERLAP_ATOMS:
                continue
            candidate = _reimplements_authority_from_atoms(
                local,
                authority_tuple[authority_index],
                local_atoms,
                authority_atoms[authority_index],
            )
            if candidate is not None:
                candidates_by_local[(local.file_path, local.line, local.symbol)].append(
                    candidate
                )
    best_candidates = [
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.overlap_score,
                candidate.authority.file_path,
                candidate.authority.line,
                candidate.authority.name,
            ),
        )[0]
        for candidates in candidates_by_local.values()
    ]
    return sorted_tuple(
        best_candidates,
        key=lambda candidate: (
            candidate.local.file_path,
            candidate.local.line,
            candidate.local.symbol,
            candidate.authority.name,
        ),
    )


def _compact_available_abstraction_reuse_candidates(
    projections: Sequence[CompactAvailableAbstractionReuseModuleProjection],
) -> tuple[AvailableAbstractionReuseCandidate, ...]:
    return _available_abstraction_reuse_candidates_from_signatures(
        tuple(
            authority
            for projection in projections
            for authority in projection.authorities
        ),
        tuple(local for projection in projections for local in projection.locals),
    )


class AvailableAbstractionReuseDetector(
    CompactProjectionCandidateDetector[
        CompactAvailableAbstractionReuseModuleProjection,
        AvailableAbstractionReuseCandidate,
    ],
):
    module_projection_family = CompactAvailableAbstractionReuseModuleProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Local implementation should reuse the available abstraction authority",
        "A local function or method rebuilds the construction/orchestration shape already owned by an available abstraction. The docs prefer routing through the existing authority instead of recreating its internal mechanics at the call site.",
        "reuse of the available abstraction authority instead of local reconstruction",
        "local code and an available abstraction share the same high-signal capability atoms",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
        (ObservationTag.NORMALIZED_AST, ObservationTag.METHOD_ROLE),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactAvailableAbstractionReuseModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[AvailableAbstractionReuseCandidate]:
        del config
        return _compact_available_abstraction_reuse_candidates(projections)

    def _findings_for_candidates(
        self,
        candidates: Sequence[AvailableAbstractionReuseCandidate],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in candidates:
            overlap_preview = ", ".join(candidate.overlap_atoms[:8])
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.local.symbol}` locally rebuilds `{candidate.authority.name}` "
                        f"with shared capability atoms {overlap_preview}."
                    ),
                    (
                        SourceLocation(
                            candidate.local.file_path,
                            candidate.local.line,
                            candidate.local.symbol,
                        ),
                        SourceLocation(
                            candidate.authority.file_path,
                            candidate.authority.line,
                            candidate.authority.symbol,
                        ),
                    ),
                    scaffold=(
                        f"# Replace local reconstruction in `{candidate.local.symbol}` with `{candidate.authority.name}`.\n"
                        f"{candidate.authority.name}(...)"
                    ),
                    codemod_patch=(
                        f"# Import and call `{candidate.authority.name}` instead of rebuilding its internals.\n"
                        "# Keep local residue as configuration, callback, or adapter arguments passed into the authority."
                    ),
                )
            )
        return findings


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
