"""What each method of a class reads, writes and requires of its receiver."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
import hashlib
from collections import (
    defaultdict,
    deque,
)
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import (
    Callable,
    TypeAlias,
)

from nominal_refactor_advisor.ast_projection import nodes_of
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_namespace import PROMOTABLE_METHOD_DECORATOR_IDENTITIES
from nominal_refactor_advisor.class_symbols import (
    ClassSymbolResolutionAuthority,
    _terminal_reference_name,
)
from nominal_refactor_advisor.class_syntax_facets import _trim_leading_docstring
from nominal_refactor_advisor.declaration_dependencies import ClassScopeDependency
from nominal_refactor_advisor.fielded_enum import FieldedStrEnum
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY

class DecoratorBindings(ABC):
    """What the bindings at a class statement's line say each decorator names. The
    module's binding snapshot is one; this module asks only this question of it."""

    @abstractmethod
    def decorators_bound_to(
        self, definition: ast.FunctionDef | ast.AsyncFunctionDef, identities: frozenset[str]
    ) -> tuple[bool, ...]:
        """For each decorator of the definition, in order, whether it is a name path the
        bindings resolve to one of the identities."""
        raise NotImplementedError


class MethodFamilyResidueRole(StrEnum):
    """Declaration role required to absorb one varying method coordinate."""

    CLASS_VARIABLE = "class_variable"
    PROPERTY_HOOK = "property_hook"
    BEHAVIOR_HOOK = "behavior_hook"


MethodFamilyResidueNameBuilder: TypeAlias = Callable[[str, int], str]


class CompactMethodSemanticCoordinateKind(FieldedStrEnum):
    """Closed coordinate grammar with declaration-owned residue semantics."""

    CONSTANT = (
        "constant",
        MethodFamilyResidueRole.CLASS_VARIABLE,
        lambda method_name, index: f"{method_name}_constant_{index}".upper(),
    )
    NAME = (
        "name",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_value_{index}",
    )
    SELF_ATTRIBUTE = (
        "self_attr",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_property_{index}",
    )
    ATTRIBUTE = (
        "attribute",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_value_{index}",
    )
    CALL = (
        "call",
        MethodFamilyResidueRole.BEHAVIOR_HOOK,
        lambda method_name, index: f"_{method_name}_operation_{index}",
    )

    residue_role: MethodFamilyResidueRole
    _residue_name_builder: MethodFamilyResidueNameBuilder

    def residue_name(self, method_name: str, index: int) -> str:
        return self._residue_name_builder(method_name, index)


@dataclass(frozen=True)
class CompactMethodSemanticCoordinate:
    """One typed varying value in a normalized method body."""

    path: tuple[str, ...]
    kind: CompactMethodSemanticCoordinateKind
    value: str


@dataclass(frozen=True)
class CompactClassMethodSemanticProfile:
    """Semantic body profile derived lazily from one compact class method."""

    skeleton: tuple[str, ...]
    coordinates: tuple[CompactMethodSemanticCoordinate, ...]


@dataclass(frozen=True)
class MethodPromotionInspection:
    """Current method syntax and lexical bindings relevant to promotion safety."""

    method: ast.FunctionDef | ast.AsyncFunctionDef
    binding_snapshot: DecoratorBindings
    """The module's bindings where the method's class statement runs, which is when its
    decorators and annotations are evaluated."""
    class_bound_names: frozenset[str]
    source_lines: tuple[str, ...]


@dataclass(frozen=True)
class ClassMethodReceiverRequirements:
    """Receiver members required directly or through local aliases."""

    member_names: frozenset[str]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ClassMethodReceiverRequirements:
        receiver_name = cls.receiver_name(method)
        if receiver_name is None:
            return cls(frozenset())
        aliases = cls.receiver_aliases(method, receiver_name)
        return cls(
            frozenset(
                node.attr
                for node in nodes_of(ast.Attribute, method)
                if isinstance(node.value, ast.Name)
                and node.value.id in aliases
            )
        )

    @staticmethod
    def receiver_aliases(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        receiver_name: str,
    ) -> frozenset[str]:
        alias_targets_by_source: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                alias_targets_by_source[node.value.id].update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif (
                isinstance(node, ast.AnnAssign | ast.NamedExpr)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                alias_targets_by_source[node.value.id].add(node.target.id)

        aliases = {receiver_name}
        pending = deque((receiver_name,))
        while pending:
            source_name = pending.popleft()
            for target_name in alias_targets_by_source[source_name]:
                if target_name in aliases:
                    continue
                aliases.add(target_name)
                pending.append(target_name)
        return frozenset(aliases)

    @staticmethod
    def receiver_name(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str | None:
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in method.decorator_list
        ):
            return None
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        return positional_parameters[0].arg if positional_parameters else None


MethodPromotionHazardPredicate: TypeAlias = Callable[[MethodPromotionInspection], bool]


def _decorator_head(decorator: ast.expr) -> str | None:
    """The name a decorator path starts from, when the decorator is a name path."""

    while isinstance(decorator, ast.Attribute):
        decorator = decorator.value
    return decorator.id if isinstance(decorator, ast.Name) else None


def _has_custom_method_decorator(inspection: MethodPromotionInspection) -> bool:
    """A decorator promotes only when it is a name path the class statement's bindings
    resolve to a promotable declaration; a name the class body binds first is the class's."""

    bound = inspection.binding_snapshot.decorators_bound_to(
        inspection.method, PROMOTABLE_METHOD_DECORATOR_IDENTITIES
    )
    return any(
        (head := _decorator_head(decorator)) is None
        or head in inspection.class_bound_names
        or not promotable
        for decorator, promotable in zip(inspection.method.decorator_list, bound, strict=True)
    )


def _is_direct_namespace_sensitive(inspection: MethodPromotionInspection) -> bool:
    name = inspection.method.name
    return name.startswith("__") and name.endswith("__")


def _has_evaluated_default(inspection: MethodPromotionInspection) -> bool:
    return bool(
        inspection.method.args.defaults
        or any(default is not None for default in inspection.method.args.kw_defaults)
    )


def _method_annotation_nodes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    arguments = (
        *method.args.posonlyargs,
        *method.args.args,
        *method.args.kwonlyargs,
        *((method.args.vararg,) if method.args.vararg is not None else ()),
        *((method.args.kwarg,) if method.args.kwarg is not None else ()),
    )
    return (
        *(
            argument.annotation
            for argument in arguments
            if argument.annotation is not None
        ),
        *((method.returns,) if method.returns is not None else ()),
    )


def _has_class_local_annotation_reference(
    inspection: MethodPromotionInspection,
) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in inspection.class_bound_names
        for annotation in _method_annotation_nodes(inspection.method)
        for node in ast.walk(annotation)
    )


def _has_attached_leading_comment(inspection: MethodPromotionInspection) -> bool:
    decorator_lines = tuple(
        decorator.lineno
        for decorator in inspection.method.decorator_list
        if decorator.lineno
    )
    start_line = min((*decorator_lines, inspection.method.lineno))
    if start_line <= 1 or start_line > len(inspection.source_lines):
        return False
    method_line = inspection.source_lines[start_line - 1]
    preceding_line = inspection.source_lines[start_line - 2]
    method_indent = method_line[: len(method_line) - len(method_line.lstrip())]
    preceding_indent = preceding_line[
        : len(preceding_line) - len(preceding_line.lstrip())
    ]
    return preceding_indent == method_indent and preceding_line.lstrip().startswith("#")


class MethodPromotionHazard(FieldedStrEnum):
    """One promotion hazard carrying its own syntax recognition behavior."""

    CUSTOM_METHOD_DECORATOR = (
        "custom_method_decorator",
        _has_custom_method_decorator,
    )
    DIRECT_NAMESPACE_SENSITIVE_METHOD = (
        "direct_namespace_sensitive_method",
        _is_direct_namespace_sensitive,
    )
    EVALUATED_DEFAULT = ("evaluated_default", _has_evaluated_default)
    CLASS_LOCAL_ANNOTATION_REFERENCE = (
        "class_local_annotation_reference",
        _has_class_local_annotation_reference,
    )
    ATTACHED_LEADING_COMMENT = (
        "attached_leading_comment",
        _has_attached_leading_comment,
    )

    predicate: MethodPromotionHazardPredicate

    def is_present(self, inspection: MethodPromotionInspection) -> bool:
        return self.predicate(inspection)


@dataclass(frozen=True)
class ClassMethodPromotionSafetyProfile:
    """Complete promotion hazards derived from one current method declaration."""

    hazards: tuple[ClassScopeDependency | MethodPromotionHazard, ...]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        binding_snapshot: DecoratorBindings,
        class_bound_names: frozenset[str] = frozenset(),
        *,
        source_lines: tuple[str, ...],
    ) -> ClassMethodPromotionSafetyProfile:
        inspection = MethodPromotionInspection(
            method,
            binding_snapshot,
            class_bound_names,
            source_lines,
        )
        return cls(
            ClassScopeDependency.from_node(method)
            + tuple(
                hazard
                for hazard in MethodPromotionHazard
                if hazard.is_present(inspection)
            )
        )


@dataclass(frozen=True)
class CompactClassMethod:
    """AST-free method fact shared by exact promotion and anti-unification."""

    class_symbol: str
    method_name: str
    line: int
    line_count: int
    body_statement_count: int
    statement_sources: tuple[str, ...]
    exact_source_digest: str | None
    promotion_hazards: tuple[ClassScopeDependency | MethodPromotionHazard, ...]
    receiver_member_names: frozenset[str]

    @property
    def statement_count(self) -> int:
        return self.body_statement_count

    @property
    def exact_promotion_source_digest(self) -> str | None:
        """Return the source identity only when exact promotion is safe."""

        return None if self.promotion_hazards else self.exact_source_digest

    @cached_property
    def semantic_profile(self) -> CompactClassMethodSemanticProfile:
        statements = _compact_class_method_statements_from_sources(
            self.statement_sources
        )
        coordinates = tuple(
            coordinate
            for statement_index, statement in enumerate(statements)
            for coordinate in self._compact_class_method_coordinates(
                statement,
                ("body", statement_index),
            )
        )
        return CompactClassMethodSemanticProfile(
            skeleton=tuple(
                self._compact_class_method_statement_skeleton(statement)
                for statement in statements
            ),
            coordinates=coordinates,
        )

    @staticmethod
    def _compact_class_method_coordinates(
        node: ast.AST,
        path: tuple[object, ...] = (),
    ) -> tuple[CompactMethodSemanticCoordinate, ...]:
        coordinates: list[CompactMethodSemanticCoordinate] = []
        coordinate_path = tuple(str(item) for item in path)
        if isinstance(node, ast.Constant):
            coordinates.append(
                CompactMethodSemanticCoordinate(
                    coordinate_path,
                    CompactMethodSemanticCoordinateKind.CONSTANT,
                    repr(node.value),
                )
            )
        elif isinstance(node, ast.Name):
            coordinates.append(
                CompactMethodSemanticCoordinate(
                    coordinate_path,
                    CompactMethodSemanticCoordinateKind.NAME,
                    node.id,
                )
            )
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                coordinates.append(
                    CompactMethodSemanticCoordinate(
                        coordinate_path,
                        CompactMethodSemanticCoordinateKind.SELF_ATTRIBUTE,
                        node.attr,
                    )
                )
            else:
                coordinates.append(
                    CompactMethodSemanticCoordinate(
                        coordinate_path,
                        CompactMethodSemanticCoordinateKind.ATTRIBUTE,
                        ast.unparse(node),
                    )
                )
        elif isinstance(node, ast.Call):
            coordinates.append(
                CompactMethodSemanticCoordinate(
                    tuple(str(item) for item in (*path, "func")),
                    CompactMethodSemanticCoordinateKind.CALL,
                    ast.unparse(node.func),
                )
            )
        skipped_fields = {"func"} if isinstance(node, ast.Call) else set()
        for field_name, value in ast.iter_fields(node):
            if field_name in skipped_fields:
                continue
            if isinstance(value, ast.AST):
                coordinates.extend(
                    CompactClassMethod._compact_class_method_coordinates(value, (*path, field_name))
                )
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        coordinates.extend(
                            CompactClassMethod._compact_class_method_coordinates(
                                item,
                                (*path, field_name, index),
                            )
                        )
        return tuple(coordinates)

    @staticmethod
    def _compact_class_method_statement_skeleton(statement: ast.stmt) -> str:
        normalized = _CompactClassMethodSemanticSkeletonNormalizer().visit(statement)
        ast.fix_missing_locations(normalized)
        return ast.dump(normalized, include_attributes=False)


class _CompactClassMethodSemanticSkeletonNormalizer(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        del node
        return ast.arg(arg="ARG", annotation=None, type_comment=None)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr="ATTR",
                ctx=node.ctx,
            ),
            node,
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value="CONST"), node)


def _compact_class_method_statements_from_sources(
    statement_sources: tuple[str, ...],
) -> tuple[ast.stmt, ...]:
    statements = tuple(ast.parse("\n".join(statement_sources)).body)
    if len(statements) != len(statement_sources):
        raise ValueError("Method-family statement source lost its AST boundary")
    return statements


def _compact_class_method(
    class_symbol: str,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: tuple[str, ...],
    binding_snapshot: DecoratorBindings,
    class_bound_names: frozenset[str],
    *,
    include_family_semantics: bool,
) -> CompactClassMethod:
    body = _trim_leading_docstring(list(method.body))
    decorator_lines = tuple(
        decorator.lineno for decorator in method.decorator_list if decorator.lineno
    )
    start_line = min((*decorator_lines, method.lineno))
    is_tiny_role = len(body) <= 2
    safety_profile = (
        ClassMethodPromotionSafetyProfile.from_method(
            method,
            binding_snapshot,
            class_bound_names,
            source_lines=source_lines,
        )
        if is_tiny_role
        else None
    )
    return CompactClassMethod(
        class_symbol=class_symbol,
        method_name=method.name,
        line=start_line,
        line_count=max(1, (method.end_lineno or method.lineno) - start_line + 1),
        body_statement_count=len(body),
        statement_sources=(
            tuple(ast.unparse(statement) for statement in body)
            if include_family_semantics and len(body) >= 3
            else ()
        ),
        exact_source_digest=(
            hashlib.blake2s(
                "".join(
                    source_lines[start_line - 1 : (method.end_lineno or method.lineno)]
                ).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            if is_tiny_role
            else None
        ),
        promotion_hazards=() if safety_profile is None else safety_profile.hazards,
        receiver_member_names=(
            ClassMethodReceiverRequirements.from_method(method).member_names
            if is_tiny_role
            else frozenset()
        ),
    )


def _compact_class_methods(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    binding_snapshots: Mapping[int | None, DecoratorBindings],
    *,
    method_names: frozenset[str] | None = None,
) -> tuple[CompactClassMethod, ...]:
    methods: list[CompactClassMethod] = []
    source_lines = tuple(parsed_module.source.splitlines(keepends=True))
    for qualname, node in indexed_class_nodes:
        class_symbol = f"{parsed_module.module_name}.{qualname}"
        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        establishes_nominal_family = any(
            ClassSymbolResolutionAuthority.establishes_nominal_family(declared_name)
            for base in node.bases
            if (
                declared_name := ClassSymbolResolutionAuthority.declared_base_name(base)
            )
            is not None
        )
        for statement in FunctionDefinition.within(node.body):
            if method_names is not None and statement.name not in method_names:
                continue
            method = _compact_class_method(
                class_symbol,
                statement,
                source_lines,
                binding_snapshots[node.lineno],
                class_bound_names,
                include_family_semantics=establishes_nominal_family,
            )
            if establishes_nominal_family or method.statement_count <= 2:
                methods.append(method)
    return tuple(methods)


def _is_classmethod(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _terminal_reference_name(decorator) == "classmethod"
        for decorator in method.decorator_list
    )
