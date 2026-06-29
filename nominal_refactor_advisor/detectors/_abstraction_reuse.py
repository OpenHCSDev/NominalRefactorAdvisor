"""Generic detection for local reimplementation of available abstractions."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from ..collection_algebra import sorted_tuple
from ..constructor_algebra import ConstructorParameterField
from ..models import MappingMetrics
from ..patterns import PatternId
from ..semantic_identity import SemanticRoleIdentityToken
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import (
    DetectorConfig,
    IssueDetector,
    ParsedModule,
    RefactorFinding,
    SourceLocation,
    high_confidence_spec,
)
from ._helpers import (
    HELPER_SYNTAX_PROJECTION_AUTHORITY,
    _semantic_role_names_for_fields,
)
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
_IDENTITY_FIELD_TERMINALS = frozenset(
    (
        *SemanticRoleIdentityToken.pluralized_string_identifier_values(),
        "path",
        "paths",
        "root",
        "roots",
    )
)
_MIN_PARALLEL_PRIMITIVE_FIELDS = 3
_MIN_PARALLEL_PRIMITIVE_RECORDS = 2
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
class ParallelPrimitiveFieldBundle(SharedFieldsBase):
    field_names: tuple[str, ...]
    semantic_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParallelPrimitiveCarrierCandidate:
    semantic_roles: tuple[str, ...]
    bundles: tuple[ParallelPrimitiveFieldBundle, ...]


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
class CarrierCompositionRetreatCandidate(CarrierBase):
    field_name: str
    carrier_type_name: str


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    atoms: frozenset[str]
    call_names: frozenset[str]

    @property
    def high_signal_atoms(self) -> frozenset[str]:
        return frozenset(
            atom for atom in self.atoms if atom.startswith(_HIGH_SIGNAL_ATOM_PREFIXES)
        )


def _snake_tokens(name: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for chunk in name.replace("-", "_").split("_"):
        if chunk:
            tokens.append(chunk.lower())
    return tuple(tokens)


def _semantic_role_for_identity_field(name: str) -> str | None:
    tokens = _snake_tokens(name)
    if len(tokens) < 2 or tokens[-1] not in _IDENTITY_FIELD_TERMINALS:
        return None
    role_tokens = tokens[:-1]
    if not role_tokens:
        return None
    return "_".join(role_tokens)


def _annotation_is_primitive_carrier(node: ast.AST | None) -> bool:
    if node is None:
        return True
    text = ast.unparse(node)
    return any(
        token in text
        for token in (
            "str",
            "Path",
            "Any",
            "Optional",
            "None",
            "Union",
        )
    )


def _class_identity_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    names: list[str] = []
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        field_name = statement.target.id
        if _semantic_role_for_identity_field(field_name) is None:
            continue
        if not _annotation_is_primitive_carrier(statement.annotation):
            continue
        names.append(field_name)
    return tuple(names)


def _module_parallel_primitive_bundles(
    module: ParsedModule,
) -> tuple[ParallelPrimitiveFieldBundle, ...]:
    bundles: list[ParallelPrimitiveFieldBundle] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        field_names = _class_identity_field_names(node)
        semantic_roles = tuple(
            role
            for field_name in field_names
            if (role := _semantic_role_for_identity_field(field_name)) is not None
        )
        if len(semantic_roles) < _MIN_PARALLEL_PRIMITIVE_FIELDS:
            continue
        bundles.append(
            ParallelPrimitiveFieldBundle(
                file_path=str(module.path),
                module_name=module.module_name,
                line=node.lineno,
                class_name=node.name,
                field_names=field_names,
                semantic_roles=semantic_roles,
            )
        )
    return tuple(bundles)


def _literal_string_tuple(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            value.value
            for value in node.elts
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    return ()


def _class_slot_names(node: ast.ClassDef) -> tuple[str, ...]:
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in statement.targets
        ):
            continue
        return _literal_string_tuple(statement.value)
    return ()


def _parameter_annotation_map(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        if argument.arg in {"self", "cls"} or argument.annotation is None:
            continue
        annotations[argument.arg] = ast.unparse(argument.annotation)
    return annotations


def _assigned_self_attribute(
    statement: ast.stmt,
) -> tuple[str, ast.AST | None] | None:
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
        value = statement.value
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    return target.attr, value


def _expression_name_references(node: ast.AST | None) -> frozenset[str]:
    if node is None:
        return frozenset()
    return frozenset(
        current.id for current in ast.walk(node) if isinstance(current, ast.Name)
    )


def _constructor_field_type_map(node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    slots = set(_class_slot_names(node))
    typed_fields: dict[str, str] = {}
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name != "__init__":
            continue
        annotations = _parameter_annotation_map(statement)
        parameter_names = tuple(annotations)
        for inner in statement.body:
            assignment = _assigned_self_attribute(inner)
            if assignment is None:
                continue
            field_name, value = assignment
            if slots and field_name not in slots:
                continue
            constructor_field = ConstructorParameterField.from_assignment(
                field_name=field_name,
                parameter_names=parameter_names,
                value_references=_expression_name_references(value),
            )
            if constructor_field is None:
                continue
            typed_fields.setdefault(
                constructor_field.field_name,
                annotations[constructor_field.source_name],
            )
    return sorted_tuple(typed_fields.items())


def _carrier_field_type_map(node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    field_type_map = dict(HELPER_SYNTAX_PROJECTION_AUTHORITY.typed_field_map(node))
    for field_name, annotation_text in _constructor_field_type_map(node):
        field_type_map.setdefault(field_name, annotation_text)
    return sorted_tuple(field_type_map.items())


def _looks_like_reusable_carrier_name(name: str) -> bool:
    return name.endswith(_CARRIER_NAME_SUFFIXES)


def _module_carrier_surfaces(module: ParsedModule) -> tuple[CarrierSurface, ...]:
    surfaces: list[CarrierSurface] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _public_name(node.name):
            continue
        field_type_map = _carrier_field_type_map(node)
        if len(field_type_map) < _MIN_CARRIER_REUSE_FIELDS:
            continue
        field_names = tuple(name for name, _ in field_type_map)
        role_names = _semantic_role_names_for_fields(field_names)
        if len(role_names) < _MIN_CARRIER_REUSE_ROLES:
            continue
        surfaces.append(
            CarrierSurface(
                file_path=str(module.path),
                module_name=module.module_name,
                line=node.lineno,
                class_name=node.name,
                field_names=field_names,
                field_type_map=field_type_map,
                role_names=role_names,
                base_names=HELPER_SYNTAX_PROJECTION_AUTHORITY.class_base_names(node),
                nominal_ancestor_names=(),
            )
        )
    return sorted_tuple(
        surfaces,
        key=lambda surface: (surface.file_path, surface.line, surface.class_name),
    )


def _carrier_authority_surfaces(
    surfaces: tuple[CarrierSurface, ...],
) -> tuple[CarrierSurface, ...]:
    return tuple(
        surface
        for surface in surfaces
        if _looks_like_reusable_carrier_name(surface.class_name)
    )


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


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
            return True
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "dataclass":
                return True
    return False


def _semantic_carrier_type_names(annotation_text: str) -> tuple[str, ...]:
    if "[" in annotation_text:
        return ()
    return sorted_tuple(
        name
        for name in _annotation_type_names(annotation_text)
        if name.endswith(_CARRIER_NAME_SUFFIXES)
    )


def _carrier_composition_retreat_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[CarrierCompositionRetreatCandidate, ...]:
    base_lookup: dict[str, set[str]] = defaultdict(set)
    class_nodes: list[tuple[ParsedModule, ast.ClassDef]] = []
    for module in modules:
        for node in ast.walk(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            class_nodes.append((module, node))
            base_lookup[node.name].update(
                HELPER_SYNTAX_PROJECTION_AUTHORITY.class_base_names(node)
            )
    ancestor_names_by_class = _class_ancestor_name_map(base_lookup)

    candidates: list[CarrierCompositionRetreatCandidate] = []
    for module, node in class_nodes:
        if not _public_name(node.name) or not _is_dataclass_class(node):
            continue
        base_names = tuple(sorted(base_lookup[node.name]))
        inherited_names = set(base_names) | set(ancestor_names_by_class[node.name])
        for field_name, annotation_text in _carrier_field_type_map(node):
            carrier_type_names = _semantic_carrier_type_names(annotation_text)
            if not carrier_type_names:
                continue
            for carrier_type_name in carrier_type_names:
                if (
                    carrier_type_name == node.name
                    or carrier_type_name in inherited_names
                ):
                    continue
                candidates.append(
                    CarrierCompositionRetreatCandidate(
                        file_path=str(module.path),
                        module_name=module.module_name,
                        line=node.lineno,
                        class_name=node.name,
                        field_name=field_name,
                        carrier_type_name=carrier_type_name,
                        base_names=base_names,
                        nominal_ancestor_names=(ancestor_names_by_class[node.name]),
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.class_name,
            candidate.field_name,
            candidate.carrier_type_name,
        ),
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


def _available_carrier_reuse_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[AvailableCarrierReuseCandidate, ...]:
    surfaces = _carrier_surfaces_with_ancestors(
        tuple(
            surface
            for module in modules
            for surface in _module_carrier_surfaces(module)
        )
    )
    authorities = _carrier_authority_surfaces(surfaces)
    if not authorities:
        return ()

    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableCarrierReuseCandidate]
    ] = defaultdict(list)
    for local in surfaces:
        for authority in authorities:
            candidate = _carrier_reuse_candidate(local, authority)
            if candidate is not None:
                candidates_by_local[
                    (local.file_path, local.line, local.class_name)
                ].append(candidate)

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


def _parallel_primitive_carrier_candidates(
    modules: list[ParsedModule],
) -> tuple[ParallelPrimitiveCarrierCandidate, ...]:
    grouped: dict[tuple[str, ...], list[ParallelPrimitiveFieldBundle]] = defaultdict(
        list
    )
    for module in modules:
        for bundle in _module_parallel_primitive_bundles(module):
            for role_count in range(
                _MIN_PARALLEL_PRIMITIVE_FIELDS, len(bundle.semantic_roles) + 1
            ):
                for semantic_roles in combinations(bundle.semantic_roles, role_count):
                    grouped[semantic_roles].append(bundle)
    candidates: list[ParallelPrimitiveCarrierCandidate] = []
    for semantic_roles, bundles in grouped.items():
        ordered = sorted_tuple(
            bundles, key=lambda item: (item.file_path, item.line, item.class_name)
        )
        if len(ordered) < _MIN_PARALLEL_PRIMITIVE_RECORDS:
            continue
        candidates.append(
            ParallelPrimitiveCarrierCandidate(
                semantic_roles=semantic_roles,
                bundles=ordered,
            )
        )
    ordered_candidates = sorted_tuple(
        candidates,
        key=lambda item: (
            -len(item.bundles),
            -len(item.semantic_roles),
            item.semantic_roles,
            item.bundles[0].file_path,
        ),
    )
    selected: list[ParallelPrimitiveCarrierCandidate] = []
    selected_bundle_sets: list[frozenset[tuple[str, int, str]]] = []
    for candidate in ordered_candidates:
        candidate_bundle_set = frozenset(
            (bundle.file_path, bundle.line, bundle.class_name)
            for bundle in candidate.bundles
        )
        if any(
            candidate_bundle_set <= selected_bundle_set
            for selected_bundle_set in selected_bundle_sets
        ):
            continue
        selected.append(candidate)
        selected_bundle_sets.append(candidate_bundle_set)
    return tuple(selected)


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
                    imported_names=frozenset(_imported_local_names(self.module)),
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
    if local.file_path == authority.file_path:
        return None
    if not _authority_available_to_local(authority, local):
        return None
    if (
        authority.name in local.signature.call_names
        and not _local_declares_authority_name(local, authority)
    ):
        return None
    overlap = local.signature.high_signal_atoms & authority.signature.high_signal_atoms
    if len(overlap) < _MIN_OVERLAP_ATOMS:
        return None
    authority_coverage = len(overlap) / max(
        len(authority.signature.high_signal_atoms), 1
    )
    if authority_coverage < _MIN_AUTHORITY_COVERAGE:
        return None
    local_coverage = len(overlap) / max(len(local.signature.high_signal_atoms), 1)
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


def _available_abstraction_reuse_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[AvailableAbstractionReuseCandidate, ...]:
    authorities = tuple(
        authority for module in modules for authority in _module_authorities(module)
    )
    if not authorities:
        return ()
    candidates_by_local: dict[
        tuple[str, int, str], list[AvailableAbstractionReuseCandidate]
    ] = defaultdict(list)
    for module in modules:
        for local in _module_locals(module):
            for authority in authorities:
                candidate = _reimplements_authority(local, authority)
                if candidate is not None:
                    candidates_by_local[
                        (local.file_path, local.line, local.symbol)
                    ].append(candidate)
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


class AvailableAbstractionReuseDetector(IssueDetector):
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

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _available_abstraction_reuse_candidates(modules):
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


class AvailableCarrierReuseDetector(IssueDetector):
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

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _available_carrier_reuse_candidates(modules):
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


class CarrierCompositionRetreatDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Carrier-valued dataclass field masks semantic inheritance",
        "A dataclass stores a nominal carrier/boundary/state/context as a regular field while not inheriting it. That is usually a retreat from load-bearing nominal identity: downstream code must unpack a composed object instead of relying on the type lattice.",
        "inheritance-compatible nominal boundary instead of carrier composition",
        "dataclass field annotation references a semantic carrier outside the inheritance closure",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.KEYWORD_MAPPING,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _carrier_composition_retreat_candidates(modules):
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.class_name}.{candidate.field_name}` stores "
                        f"`{candidate.carrier_type_name}` as a field instead of "
                        "inheriting it."
                    ),
                    (
                        SourceLocation(
                            candidate.file_path,
                            candidate.line,
                            f"{candidate.class_name}.{candidate.field_name}",
                        ),
                    ),
                    scaffold=(
                        f"class {candidate.class_name}({candidate.carrier_type_name}, ...):\n"
                        "    # keep only genuine local residue here\n"
                        "    ..."
                    ),
                    codemod_patch=(
                        f"# Replace `{candidate.field_name}: {candidate.carrier_type_name}` "
                        f"on `{candidate.class_name}` with direct inheritance from "
                        f"`{candidate.carrier_type_name}`.\n"
                        "# If dataclass frozen/mutable settings block inheritance, normalize "
                        "the carrier family configuration instead of hiding the carrier behind "
                        "a composed field."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=1,
                        mapping_name="carrier_composition_retreat",
                        field_names=(candidate.field_name,),
                        source_name=candidate.carrier_type_name,
                        identity_field_names=(candidate.field_name,),
                    ),
                )
            )
        return findings


class ParallelPrimitiveCarrierDetector(IssueDetector):
    ssot_authority_boundary = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel primitive fields should become a nominal carrier",
        "Several record/request classes carry the same correlated primitive identity fields. The docs prefer one nominal carrier with local invariants over repeatedly threading adjacent strings or paths that must describe one semantic object.",
        "single nominal carrier for correlated identity/path roles",
        "same primitive identity role bundle is repeated across record classes",
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

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _parallel_primitive_carrier_candidates(modules):
            bundles = candidate.bundles
            role_summary = ", ".join(candidate.semantic_roles)
            class_summary = ", ".join(bundle.class_name for bundle in bundles[:5])
            field_summary = ", ".join(bundles[0].field_names)
            findings.append(
                self.build_finding(
                    (
                        f"Primitive identity roles ({role_summary}) are repeated "
                        f"across records {class_summary}."
                    ),
                    tuple(
                        SourceLocation(
                            bundle.file_path,
                            bundle.line,
                            bundle.class_name,
                        )
                        for bundle in bundles[:6]
                    ),
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        "class NominalIdentityCarrier:\n"
                        f"    # roles: {role_summary}\n"
                        "    ...\n\n"
                        f"# Replace parallel primitive fields ({field_summary}) "
                        "with one nominal carrier and project it at the transport boundary."
                    ),
                    codemod_patch=(
                        "# Introduce one dataclass/record for the repeated role bundle.\n"
                        "# Store invariants on that record; pass the carrier internally; "
                        "serialize primitive fields only at external protocol boundaries."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(bundles),
                        mapping_name="parallel_primitive_carrier",
                        field_names=bundles[0].field_names,
                        identity_field_names=candidate.semantic_roles,
                    ),
                )
            )
        return findings


__all__ = tuple(name for name in globals() if not name.startswith("_"))
