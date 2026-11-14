"""Shared detector helper functions.

This module contains private analysis helpers that support detector families
across the split implementation modules.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import (
    ClassVar,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
)

from nominal_refactor_advisor.ast_field_decoders import (
    ParameterAnnotations,
    variadic_parameters,
)
from nominal_refactor_advisor.class_method_facets import (
    CompactClassMethod,
    CompactClassMethodSemanticProfile,
    CompactMethodSemanticCoordinate,
    CompactMethodSemanticCoordinateKind,
    MethodFamilyResidueRole,
)
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.value_expression import CompactValueExpression
from nominal_refactor_advisor.annotation_semantics import (
    CLASSVAR_ANNOTATION_AUTHORITY,
    TypeWrapper,
)
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_projection import (
    CallKeywords,
    StatementValue,
    StatementValues,
)
from nominal_refactor_advisor.ast_tools import (
    BuiltinCallName,
    SourceModule,
    walk_function_body_nodes,
)
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactLeafMethodAncestorPromotionComponentBuilder,
    ExactMethodRoleComponent,
    ExactMethodRoleComponentBuilder,
)
from nominal_refactor_advisor.factorization import (
    FactorizationRow,
    ResidueHookNamesCarrier,
    factorization_axis_catalog_certificate,
)
from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex
from nominal_refactor_advisor.semantic_algebra import (
    ObjectFamilyShape,
)
from nominal_refactor_advisor.semantic_description_length import (
    ClassFamilyCompressionProfile,
    CompressionCertificate,
)
import ast
from typing import Iterable
from nominal_refactor_advisor.ast_tools import (
    is_docstring_statement,
    ParsedModule,
    statements_without_docstring,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.semantic_match import (
    as_ast,
    constant_value,
    Maybe,
    name_id,
    named_call_assignment,
    named_value_binding,
    NamedCallAssignment,
    return_call,
    single_assign_target,
    single_item,
)
from nominal_refactor_advisor.detectors._base import (
    _PIPELINE_ASSIGN_STAGE,
    _PIPELINE_RETURN_STAGE,
    _self_attr_name,
    CANDIDATE_COLLECTION_AUTHORITY,
    ClassFamilyWitnessCarrier,
    ClassMethodLineWitnessCandidate,
    DetectorConfig,
    NamedFunctionNode,
    OverlappingInheritanceFamiliesCandidate,
    PipelineAssemblyStage,
    SemanticOverlapMethodCandidate,
    SemanticOverlapMethodFamilyCandidate,
    SemanticOverlapResidueAxisCandidate,
    StaticModulePatternDetector,
    TupleIndexSemanticOpacityCandidate,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassStatement,
)
from nominal_refactor_advisor.detectors._substrate_support import (
    ModuleNamedValues,
)
from nominal_refactor_advisor.ast_projection import nodes_of

BaseBundleClassGroups: TypeAlias = dict[tuple[str, ...], list[ast.ClassDef]]
NamedStringSequenceSpec: TypeAlias = tuple[str, int, tuple[str, ...]]
NamedStringSequenceSpecs: TypeAlias = tuple[NamedStringSequenceSpec, ...]
MutableNamedStringSequenceSpecs: TypeAlias = list[NamedStringSequenceSpec]
NamedStringSequenceSpecsByName: TypeAlias = dict[str, MutableNamedStringSequenceSpecs]
DerivedQuerySignature: TypeAlias = tuple[str | None, str, str, str]
DerivedQuerySpecsBySignature: TypeAlias = dict[
    DerivedQuerySignature, MutableNamedStringSequenceSpecs
]

MethodFamilyCandidateItemT = TypeVar("MethodFamilyCandidateItemT", covariant=True)


_IMPLICIT_METHOD_PARAMETER_NAMES = frozenset({"self", "cls"})
_SEQUENCE_WRAPPER_CALL_NAMES = BuiltinCallName.sequence_wrapper_names()


class HelperSupportProjectionAuthority:
    def expression_root_names(self, node: ast.AST) -> set[str]:
        roots: set[str] = set()

        class Visitor(ast.NodeVisitor):
            def visit_Attribute(self, node: ast.Attribute) -> None:
                current: ast.AST = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    roots.add(current.id)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                roots.add(node.id)

        Visitor().visit(node)
        return roots

    def if_returns_none_only(self, node: ast.If) -> bool:
        return bool(
            len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
            and StatementValue.of(node.body[0]).compact_value().value_is_none_literal
            and (not node.orelse)
        )

HELPER_SUPPORT_PROJECTION_AUTHORITY = HelperSupportProjectionAuthority()


def _function_parameter_annotation_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for arg in (
        ParameterAnnotations.of(
            tuple(node.args.posonlyargs)
            + tuple(node.args.args)
            + tuple(node.args.kwonlyargs)
        ).annotated
    ):
        annotations[arg.arg] = ast.unparse(arg.annotation)
    return annotations


def _call_display_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _name_mentions(node: ast.AST, name: str) -> bool:
    return any(
        (
            isinstance(current, ast.Name) and current.id == name
            for current in ast.walk(node)
        )
    )


def _raised_exception_name(
    statement: ast.stmt,
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return None
    exc = statement.exc
    if isinstance(exc, ast.Call):
        exc_name = ExpressionReference.decoded(exc.func).terminal_name
        referenced_names = sorted_tuple(
            {
                current.id
                for current in ast.walk(exc)
                if isinstance(current, ast.Name)
            }
        )
        if exc_name is not None:
            return (exc_name, referenced_names)
    exc_name = ExpressionReference.decoded(exc).terminal_name
    if exc_name is not None:
        return (exc_name, ())
    return None


def _linear_query_loop(body: list[ast.stmt]) -> ast.For | None:
    if len(body) < 2:
        return None
    loop = next(
        (statement for statement in body if isinstance(statement, ast.For)), None
    )
    if loop is None or not isinstance(loop.target, ast.Name):
        return None
    return loop


def _linear_query_return_expr(loop: ast.For, result_name: str) -> ast.AST | None:
    return_exprs = StatementValues.of(nodes_of(ast.Return, loop)).values
    if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
        return None
    return return_exprs[0]


def _linear_query_raised_exception(
    body: list[ast.stmt],
) -> tuple[str, tuple[str, ...]] | None:
    raised = next(
        (
            _raised_exception_name(statement)
            for statement in body
            if _raised_exception_name(statement) is not None
        ),
        None,
    )
    if raised is None:
        return None
    exception_name, exception_names = raised
    if exception_name not in {"KeyError", "LookupError", "ValueError"}:
        return None
    return exception_name, exception_names


def _linear_query_key_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    loop: ast.For,
    return_expr: ast.AST,
    exception_names: tuple[str, ...],
) -> tuple[str, ...]:
    parameter_names = tuple(
        (
            arg.arg
            for arg in tuple(node.args.posonlyargs)
            + tuple(node.args.args)
            + tuple(node.args.kwonlyargs)
            if arg.arg not in {"self", "cls"}
        )
    )
    query_key_names = sorted_tuple(
        (
            name
            for name in parameter_names
            if _name_mentions(return_expr, name)
            or name in exception_names
            or any(
                (
                    isinstance(current, ast.If) and _name_mentions(current.test, name)
                    for current in ast.walk(loop)
                )
            )
        )
    )
    return query_key_names


class MethodProjection:
    def aliases_to_self_attrs(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for subnode in ast.walk(method):
            binding = named_value_binding(subnode)
            if binding is None or binding.value is None:
                continue
            attr_name = None
            if (value_attr_name := _self_attr_name(binding.value)) is not None:
                attr_name = value_attr_name
            elif (value_name := name_id(binding.value)) is not None:
                attr_name = aliases.get(value_name)
            if attr_name is None:
                aliases.pop(binding.name, None)
                continue
            aliases[binding.name] = attr_name
        return aliases


METHOD_PROJECTION = MethodProjection()


def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
    return value_fingerprint.startswith(("Name(", "Attribute(", "IfExp(", "Constant("))


def _guarded_wrapper_node_types(node: ast.If) -> tuple[str, ...] | None:
    test = node.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    isinstance_arguments = BuiltinCallName.isinstance_arguments(test.operand)
    if isinstance_arguments is None:
        return None
    _subject, type_node = isinstance_arguments
    if isinstance(type_node, ast.Tuple):
        node_types = tuple(ast.unparse(item) for item in type_node.elts)
    else:
        node_types = (ast.unparse(type_node),)
    return tuple(item for item in node_types if item)


def _guarded_wrapper_function_candidates(
    module: ParsedModule,
) -> NamedStringSequenceSpecs:
    candidates: MutableNamedStringSequenceSpecs = []
    guarded_returns: list[tuple[tuple[ast.FunctionDef, ast.If], ast.Return]] = []
    for statement in module.module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        body = statements_without_docstring(statement.body)
        while (
            body
            and isinstance(body[0], ast.Assign)
            and (len(body[0].targets) == 1)
            and isinstance(body[0].targets[0], ast.Name)
        ):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(
            guard, ast.If
        ) or not HELPER_SUPPORT_PROJECTION_AUTHORITY.if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return):
            continue
        guarded_returns.append(((statement, guard), return_stmt))
    # A wrapper whose return is bare wraps nothing.
    for (statement, guard), _returned in StatementValues.keyed(guarded_returns).supplied:
        node_types = _guarded_wrapper_node_types(guard)
        if not node_types:
            continue
        candidates.append((statement.name, statement.lineno, node_types))
    return tuple(candidates)


def _param_backed_name(expr: ast.AST, parameter_names: set[str]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in parameter_names:
        return expr.id
    if isinstance(expr, ast.IfExp):
        body_name = _param_backed_name(expr.body, parameter_names)
        orelse_name = _param_backed_name(expr.orelse, parameter_names)
        if body_name is not None and orelse_name is None:
            return body_name
        if orelse_name is not None and body_name is None:
            return orelse_name
        if body_name == orelse_name:
            return body_name
    if isinstance(expr, ast.BoolOp):
        names = {
            name
            for value in expr.values
            for name in (_param_backed_name(value, parameter_names),)
            if name is not None
        }
        if len(names) == 1:
            return next(iter(names))
    return None


_ABC_BASE_NAME = "ABC"
_COMPOSABLE_BASE_NAME_SUFFIXES = (
    _ABC_BASE_NAME,
    "Base",
    "Carrier",
    "Contract",
    "Mixin",
    "Template",
)


def _is_composable_base_name(base_name: str) -> bool:
    return base_name == _ABC_BASE_NAME or base_name.endswith(
        _COMPOSABLE_BASE_NAME_SUFFIXES
    )


def _is_contiguous_subtuple(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


@dataclass(frozen=True)
class _RegisteredUnionSurfaceSource:
    owner_name: str
    value: ast.AST
    line: int

    @classmethod
    def from_node(cls, node: ast.AST) -> _RegisteredUnionSurfaceSource | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in statements_without_docstring(node.body):
                if isinstance(statement, ast.For):
                    return cls(node.name, statement.iter, statement.lineno)
                if isinstance(statement, ast.Assign):
                    return cls(node.name, statement.value, statement.lineno)
            return None
        if not isinstance(node, ast.Assign):
            return None
        target_name = name_id(single_item(node.targets))
        return (
            None if target_name is None else cls(target_name, node.value, node.lineno)
        )


_CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES = (
    "build",
    "create",
    "from_",
    "make",
)


def _annotation_union_members(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            *_annotation_union_members(node.left),
            *_annotation_union_members(node.right),
        )
    if isinstance(node, ast.Subscript):
        root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.value)
        if root_name != "Union":
            return (node,)
        if isinstance(node.slice, ast.Tuple):
            return tuple(node.slice.elts)
        if isinstance(node.slice, ast.List):
            return tuple(node.slice.elts)
    return (node,)


def _class_object_type_argument_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.value)
    if not TypeWrapper.CLASS_OBJECT.spelled_by(root_name):
        return None
    return HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.slice)


def _class_def_declared_and_ancestor_names(
    classes_by_name: dict[str, ClassStatement],
    ancestor_names_by_class: dict[str, tuple[str, ...]],
    class_name: str,
) -> set[str]:
    return set(classes_by_name[class_name].base_terminal_names) | set(
        ancestor_names_by_class[class_name]
    )


def _base_defines_contract_attributes(
    classes_by_name: dict[str, ClassStatement],
    base_name: str,
    observed_attribute_names: tuple[str, ...],
) -> bool:
    base_class = classes_by_name.get(base_name)
    if base_class is None:
        return False
    return all(
        (
            base_class.method_named(attribute_name) is not None
            for attribute_name in observed_attribute_names
        )
    )


def _shared_edge_tokens(member_type_names: tuple[str, ...]) -> tuple[str, ...]:
    token_rows = tuple(
        (
            CLASS_NAME_ALGEBRA.ordered_tokens(member_type_name)
            for member_type_name in member_type_names
        )
    )
    if not token_rows or not all(token_rows):
        return ()
    prefix_tokens: list[str] = []
    for column in zip(*token_rows, strict=False):
        if len(set(column)) != 1:
            break
        prefix_tokens.append(column[0])
    suffix_tokens: list[str] = []
    reversed_rows = tuple(tuple(reversed(row)) for row in token_rows)
    for column in zip(*reversed_rows, strict=False):
        if len(set(column)) != 1:
            break
        suffix_tokens.append(column[0])
    return (*prefix_tokens, *tuple(reversed(suffix_tokens)))


_MethodFamilyKey: TypeAlias = tuple[str, tuple[str, ...]]
_MethodFamilyAxisSpecsByFamily: TypeAlias = dict[_MethodFamilyKey, set[str]]
_MethodFamilyAxisRow: TypeAlias = tuple[str, tuple[str, ...]]
_MethodFamilyAxisRowsByFamily: TypeAlias = dict[
    _MethodFamilyKey, set[_MethodFamilyAxisRow]
]
_MethodFamilyLatticeEdgesByFamily: TypeAlias = dict[
    _MethodFamilyKey, set[tuple[tuple[str, ...], tuple[str, ...]]]
]
_MethodFamilyMethodPlanKey: TypeAlias = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class _MethodFamilyMethodGroupProfile(ResidueHookNamesCarrier):
    shared_statement_count: int
    varying_coordinates: tuple[CompactMethodSemanticCoordinate, ...]
    compression_certificate: CompressionCertificate

    @staticmethod
    def _varying_coordinates(
        profiles: tuple[CompactClassMethodSemanticProfile, ...],
    ) -> tuple[CompactMethodSemanticCoordinate, ...]:
        grouped: dict[
            tuple[tuple[str, ...], CompactMethodSemanticCoordinateKind],
            set[str],
        ] = defaultdict(set)
        representatives: dict[
            tuple[tuple[str, ...], CompactMethodSemanticCoordinateKind],
            CompactMethodSemanticCoordinate,
        ] = {}
        for profile in profiles:
            for coordinate in profile.coordinates:
                key = (coordinate.path, coordinate.kind)
                grouped[key].add(coordinate.value)
                representatives.setdefault(key, coordinate)
        return tuple(
            representatives[key]
            for key, values in sorted(grouped.items(), key=lambda item: item[0])
            if len(values) >= 2
        )

    @staticmethod
    def _residue_names(
        method_name: str,
        varying_coordinates: tuple[CompactMethodSemanticCoordinate, ...],
    ) -> ResidueHookNamesCarrier:
        names_by_role: dict[MethodFamilyResidueRole, list[str]] = defaultdict(list)
        for index, coordinate in enumerate(varying_coordinates, start=1):
            names_by_role[coordinate.kind.residue_role].append(
                coordinate.kind.residue_name(method_name, index)
            )
        return ResidueHookNamesCarrier(
            classvar_names=tuple(
                dict.fromkeys(
                    names_by_role[MethodFamilyResidueRole.CLASS_VARIABLE]
                )
            ),
            property_hook_names=tuple(
                dict.fromkeys(names_by_role[MethodFamilyResidueRole.PROPERTY_HOOK])
            ),
            behavior_hook_names=tuple(
                dict.fromkeys(names_by_role[MethodFamilyResidueRole.BEHAVIOR_HOOK])
            ),
        )

    @classmethod
    def from_class_methods(
        cls,
        method_name: str,
        class_methods: tuple[tuple[CompactIndexedClass, CompactClassMethod], ...],
    ) -> _MethodFamilyMethodGroupProfile | None:
        if len(class_methods) < 2:
            return None
        methods = tuple(method for _, method in class_methods)
        statement_counts = {method.statement_count for method in methods}
        if len(statement_counts) != 1:
            return None
        shared_statement_count = next(iter(statement_counts))
        if shared_statement_count < 3:
            return None
        profiles = tuple(method.semantic_profile for method in methods)
        if len({profile.skeleton for profile in profiles}) != 1:
            return None
        varying_coordinates = cls._varying_coordinates(profiles)
        if not varying_coordinates or len(varying_coordinates) > max(
            4,
            shared_statement_count * 2,
        ):
            return None
        residue_names = cls._residue_names(
            method_name,
            varying_coordinates,
        )
        certificate = ClassFamilyCompressionProfile.from_repeated_method_family(
            class_count=len(
                {indexed_class.symbol for indexed_class, _ in class_methods}
            ),
            shared_statement_count=shared_statement_count,
            hook_count=len(residue_names.property_hook_names)
            + len(residue_names.behavior_hook_names),
            classvar_count=len(residue_names.classvar_names),
        ).compression_certificate
        if not certificate.pays_rent:
            return None
        return cls(
            shared_statement_count=shared_statement_count,
            varying_coordinates=varying_coordinates,
            classvar_names=residue_names.classvar_names,
            property_hook_names=residue_names.property_hook_names,
            behavior_hook_names=residue_names.behavior_hook_names,
            compression_certificate=certificate,
        )


@dataclass(frozen=True)
class _MethodFamilyMethodPlan(ClassFamilyWitnessCarrier):
    base_symbol: str
    method_name: str
    profile: _MethodFamilyMethodGroupProfile
    line_numbers: tuple[int, ...]
    line_count: int

    @classmethod
    def from_class_methods(
        cls,
        base_symbol: str,
        base_name: str,
        method_name: str,
        class_methods: tuple[tuple[CompactIndexedClass, CompactClassMethod], ...],
    ) -> _MethodFamilyMethodPlan | None:
        profile = _MethodFamilyMethodGroupProfile.from_class_methods(
            method_name,
            class_methods,
        )
        if profile is None:
            return None
        methods = tuple(method for _, method in class_methods)
        return cls(
            base_symbol=base_symbol,
            base_name=base_name,
            method_name=method_name,
            profile=profile,
            class_names=tuple(
                indexed_class.simple_name for indexed_class, _ in class_methods
            ),
            file_paths=tuple(
                indexed_class.file_path for indexed_class, _ in class_methods
            ),
            line_numbers=tuple(method.line for method in methods),
            line_count=sum(method.line_count for method in methods),
        )

    def is_more_specific_than(
        self,
        incumbent: _MethodFamilyMethodPlan,
        class_index: CompactClassFamilyIndex,
    ) -> bool:
        return incumbent.base_symbol in class_index.ancestor_symbols(self.base_symbol)

    @classmethod
    def most_specific(
        cls,
        method_plans: Iterable[_MethodFamilyMethodPlan],
        class_index: CompactClassFamilyIndex,
    ) -> tuple[_MethodFamilyMethodPlan, ...]:
        plans_by_key: dict[
            _MethodFamilyMethodPlanKey,
            list[_MethodFamilyMethodPlan],
        ] = defaultdict(list)
        for method_plan in method_plans:
            key = (method_plan.method_name, method_plan.class_names)
            incumbents = plans_by_key[key]
            if any(
                incumbent.base_symbol == method_plan.base_symbol
                or incumbent.is_more_specific_than(method_plan, class_index)
                for incumbent in incumbents
            ):
                continue
            plans_by_key[key] = [
                incumbent
                for incumbent in incumbents
                if not method_plan.is_more_specific_than(incumbent, class_index)
            ]
            plans_by_key[key].append(method_plan)
        return tuple(
            method_plan
            for method_plan_group in plans_by_key.values()
            for method_plan in method_plan_group
        )

    @staticmethod
    def _classes_by_base(
        class_index: CompactClassFamilyIndex,
    ) -> dict[tuple[str, str], list[CompactIndexedClass]]:
        classes_by_base = {
            (base_symbol, base.simple_name): [
                indexed_class
                for descendant_symbol in descendant_symbols
                if (
                    indexed_class := class_index.class_for(descendant_symbol)
                )
                is not None
            ]
            for base_symbol, descendant_symbols in class_index.descendants_by_symbol.items()
            if (base := class_index.class_for(base_symbol)) is not None
            and ClassSymbolResolutionAuthority.establishes_nominal_family(
                base.simple_name
            )
        }
        for indexed_class in class_index.classes_by_symbol.values():
            resolved_base_names = {
                base.simple_name
                for base_symbol in indexed_class.resolved_base_symbols
                if (base := class_index.class_for(base_symbol)) is not None
            }
            for base_reference in indexed_class.base_references:
                if (
                    ClassSymbolResolutionAuthority.establishes_nominal_family(
                        base_reference.qualified_name
                    )
                    and base_reference.simple_name not in resolved_base_names
                ):
                    classes_by_base.setdefault(
                        (
                            base_reference.qualified_name,
                            base_reference.simple_name,
                        ),
                        [],
                    ).append(indexed_class)
        return classes_by_base

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        class_index: CompactClassFamilyIndex,
    ) -> tuple[_MethodFamilyMethodPlan, ...]:
        methods_by_class: dict[str, dict[str, CompactClassMethod]] = defaultdict(dict)
        for projection in projections:
            for method in projection.class_methods:
                methods_by_class[method.class_symbol][method.method_name] = method
        method_plans: list[_MethodFamilyMethodPlan] = []
        for (base_symbol, base_name), indexed_classes in cls._classes_by_base(
            class_index
        ).items():
            if len(indexed_classes) < 2:
                continue
            # One pass over the family, in its order: each method name gets
            # the (class, method) pairs that declare it, without rescanning
            # every class once per name.
            class_methods_by_name: dict[
                str, list[tuple[CompactIndexedClass, CompactClassMethod]]
            ] = defaultdict(list)
            for indexed_class in indexed_classes:
                for method_name, method in methods_by_class.get(
                    indexed_class.symbol, {}
                ).items():
                    class_methods_by_name[method_name].append((indexed_class, method))
            for method_name in sorted(class_methods_by_name):
                method_plan = cls.from_class_methods(
                    base_symbol,
                    base_name,
                    method_name,
                    tuple(class_methods_by_name[method_name]),
                )
                if method_plan is not None:
                    method_plans.append(method_plan)
        return cls.most_specific(method_plans, class_index)

    @property
    def residue_kinds(self) -> tuple[CompactMethodSemanticCoordinateKind, ...]:
        return tuple(
            coordinate.kind for coordinate in self.profile.varying_coordinates
        )

    def semantic_overlap_candidate(
        self,
        family: _MethodFamilyEvidence,
    ) -> SemanticOverlapMethodCandidate:
        return SemanticOverlapMethodCandidate(
            file_path=self.file_paths[0],
            line=min(self.line_numbers),
            base_name=self.base_name,
            method_name=self.method_name,
            class_names=self.class_names,
            file_paths=self.file_paths,
            line_numbers=self.line_numbers,
            shared_statement_count=self.profile.shared_statement_count,
            varying_coordinate_count=len(self.profile.varying_coordinates),
            classvar_names=self.profile.classvar_names,
            property_hook_names=self.profile.property_hook_names,
            behavior_hook_names=self.profile.behavior_hook_names,
            family_method_names=family.method_names,
            leaf_residue_names=family.leaf_residue_names,
            residue_declaration_count=family.residue_declaration_count,
            shared_to_residue_ratio=family.shared_to_residue_ratio,
            strict_subset_family_specs=family.strict_subset_family_specs,
            partial_overlap_family_specs=family.partial_overlap_family_specs,
            lattice_node_count=family.lattice_node_count,
            lattice_edge_count=family.lattice_edge_count,
            line_count=self.line_count,
            compression_certificate=self.profile.compression_certificate,
        )


class _MethodPlanEvidenceABC(ABC):
    """Shared projections derived from one or more method-family plans."""

    @property
    @abstractmethod
    def method_plans(self) -> tuple[_MethodFamilyMethodPlan, ...]:
        raise NotImplementedError

    @cached_property
    def method_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            method_plan.method_name for method_plan in self.method_plans
        )

    @cached_property
    def classvar_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            name
            for method_plan in self.method_plans
            for name in method_plan.profile.classvar_names
        )

    @cached_property
    def property_hook_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            name
            for method_plan in self.method_plans
            for name in method_plan.profile.property_hook_names
        )

    @cached_property
    def behavior_hook_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            name
            for method_plan in self.method_plans
            for name in method_plan.profile.behavior_hook_names
        )

    @cached_property
    def leaf_residue_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            (
                *self.classvar_names,
                *self.property_hook_names,
                *self.behavior_hook_names,
            )
        )

    @cached_property
    def residue_declaration_count(self) -> int:
        return sum(
            len(method_plan.class_names)
            * (
                len(method_plan.profile.classvar_names)
                + len(method_plan.profile.property_hook_names)
                + len(method_plan.profile.behavior_hook_names)
            )
            for method_plan in self.method_plans
        )

    @cached_property
    def shared_statement_count(self) -> int:
        return sum(
            method_plan.profile.shared_statement_count
            for method_plan in self.method_plans
        )

    @cached_property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(
            file_path
            for method_plan in self.method_plans
            for file_path in method_plan.file_paths
        )

    @cached_property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(
            line_number
            for method_plan in self.method_plans
            for line_number in method_plan.line_numbers
        )

    @cached_property
    def method_symbols(self) -> tuple[str, ...]:
        return tuple(
            f"{class_name}.{method_plan.method_name}"
            for method_plan in self.method_plans
            for class_name in method_plan.class_names
        )

    @cached_property
    def line_count(self) -> int:
        return sum(method_plan.line_count for method_plan in self.method_plans)


@dataclass(frozen=True)
class _MethodFamilyEvidence(_MethodPlanEvidenceABC):
    """One exact class-family relation and the method plans proving it."""

    _method_plans: tuple[_MethodFamilyMethodPlan, ...]
    strict_subset_family_specs: tuple[str, ...]
    partial_overlap_family_specs: tuple[str, ...]
    lattice_node_count: int
    lattice_edge_count: int

    @property
    def method_plans(self) -> tuple[_MethodFamilyMethodPlan, ...]:
        return self._method_plans

    @property
    def base_symbol(self) -> str:
        return self.method_plans[0].base_symbol

    @property
    def base_name(self) -> str:
        return self.method_plans[0].base_name

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.method_plans[0].class_names

    @cached_property
    def shared_to_residue_ratio(self) -> float:
        return self.shared_statement_count / max(self.residue_declaration_count, 1)

    def semantic_overlap_candidates(
        self,
    ) -> tuple[SemanticOverlapMethodCandidate, ...]:
        return tuple(
            method_plan.semantic_overlap_candidate(self)
            for method_plan in self.method_plans
        )

    @property
    def group_spec(self) -> str:
        return f"{'+'.join(self.method_names)}[{','.join(self.class_names)}]"

    def compression_certificate(self) -> CompressionCertificate | None:
        residue_names = self.leaf_residue_names
        certificate = CompressionCertificate.from_object_family(
            manual_object_count=sum(
                len(self.class_names) * method_plan.profile.shared_statement_count
                for method_plan in self.method_plans
            ),
            replacement_shape=ObjectFamilyShape(
                shared_objects=("abc_base", "family_template"),
                per_axis_objects=("residue_declaration",),
            ),
            semantic_axes=(*self.class_names, *residue_names),
            residual_object_count=len(self.class_names) * len(residue_names),
            provenance_object_count=1,
            independent_source_count=len(set(self.method_plans[0].file_paths)),
        )
        return certificate if certificate.pays_rent else None

    def family_candidate(self) -> SemanticOverlapMethodFamilyCandidate | None:
        if len(self.method_plans) < 2:
            return None
        certificate = self.compression_certificate()
        if certificate is None:
            return None
        return SemanticOverlapMethodFamilyCandidate(
            file_path=self.file_paths[0],
            line=min(self.line_numbers),
            base_name=self.base_name,
            class_names=self.class_names,
            method_names=self.method_names,
            file_paths=self.file_paths,
            line_numbers=self.line_numbers,
            method_symbols=self.method_symbols,
            shared_statement_count=self.shared_statement_count,
            residue_declaration_count=self.residue_declaration_count,
            classvar_names=self.classvar_names,
            property_hook_names=self.property_hook_names,
            behavior_hook_names=self.behavior_hook_names,
            leaf_residue_names=self.leaf_residue_names,
            shared_to_residue_ratio=self.shared_to_residue_ratio,
            strict_subset_family_specs=self.strict_subset_family_specs,
            partial_overlap_family_specs=self.partial_overlap_family_specs,
            lattice_node_count=self.lattice_node_count,
            lattice_edge_count=self.lattice_edge_count,
            line_count=self.line_count,
            compression_certificate=certificate,
        )

    def residue_axis_candidate(self) -> SemanticOverlapResidueAxisCandidate | None:
        if len(self.method_plans) < 2:
            return None
        residue_kind_signatures = {
            method_plan.residue_kinds for method_plan in self.method_plans
        }
        if len(residue_kind_signatures) != 1:
            return None
        residue_kinds = next(iter(residue_kind_signatures))
        if not residue_kinds:
            return None
        axis_names = tuple(
            f"{index}:{residue_kind}"
            for index, residue_kind in enumerate(residue_kinds)
        )
        certificate = factorization_axis_catalog_certificate(
            (
                FactorizationRow.from_mapping(
                    method_plan.method_name,
                    {
                        axis_name: residue_kind
                        for axis_name, residue_kind in zip(
                            axis_names,
                            residue_kinds,
                        )
                    },
                    source_name="|".join(
                        sorted_tuple(frozenset(method_plan.file_paths))
                    ),
                )
                for method_plan in self.method_plans
            ),
            shared_objects=("residue_axis_catalog",),
            per_axis_objects=("residue_axis_row",),
        )
        if not certificate.pays_rent:
            return None
        return SemanticOverlapResidueAxisCandidate(
            file_path=self.file_paths[0],
            line=min(self.line_numbers),
            base_name=self.base_name,
            class_names=self.class_names,
            method_names=self.method_names,
            residue_kinds=residue_kinds,
            file_paths=self.file_paths,
            line_numbers=self.line_numbers,
            method_symbols=self.method_symbols,
            residue_site_count=len(self.method_plans) * len(residue_kinds),
            line_count=self.line_count,
            compression_certificate=certificate,
        )

    @staticmethod
    def _member_spec(method_name: str, class_names: tuple[str, ...]) -> str:
        return f"{method_name}[{','.join(class_names)}]"

    @classmethod
    def from_method_plans(
        cls,
        method_plans: tuple[_MethodFamilyMethodPlan, ...],
    ) -> tuple[_MethodFamilyEvidence, ...]:
        exact_groups: dict[
            _MethodFamilyKey,
            list[_MethodFamilyMethodPlan],
        ] = defaultdict(list)
        method_plans_by_base: dict[
            str,
            list[_MethodFamilyMethodPlan],
        ] = defaultdict(list)
        subset_rows_by_family: _MethodFamilyAxisRowsByFamily = defaultdict(set)
        overlap_specs_by_family: _MethodFamilyAxisSpecsByFamily = defaultdict(set)
        lattice_nodes_by_family: dict[
            _MethodFamilyKey,
            set[tuple[str, ...]],
        ] = defaultdict(set)
        lattice_edges_by_family: _MethodFamilyLatticeEdgesByFamily = defaultdict(set)
        for method_plan in method_plans:
            family_key = (method_plan.base_symbol, method_plan.class_names)
            family_classes = set(method_plan.class_names)
            family_class_names = sorted_tuple(family_classes)
            lattice_nodes_by_family[family_key].add(family_class_names)
            exact_groups[family_key].append(method_plan)
            method_plans_by_base[method_plan.base_symbol].append(method_plan)
        for method_plan in method_plans:
            family_key = (method_plan.base_symbol, method_plan.class_names)
            family_classes = set(method_plan.class_names)
            family_class_names = sorted_tuple(family_classes)
            for other_plan in method_plans_by_base[method_plan.base_symbol]:
                other_classes = set(other_plan.class_names)
                if other_classes == family_classes:
                    continue
                class_intersection = family_classes & other_classes
                if not class_intersection:
                    continue
                other_class_names = sorted_tuple(other_classes)
                intersection_class_names = sorted_tuple(class_intersection)
                lattice_nodes_by_family[family_key].update(
                    (other_class_names, intersection_class_names)
                )
                if intersection_class_names != family_class_names:
                    lattice_edges_by_family[family_key].add(
                        (intersection_class_names, family_class_names)
                    )
                if intersection_class_names != other_class_names:
                    lattice_edges_by_family[family_key].add(
                        (intersection_class_names, other_class_names)
                    )
                if other_classes < family_classes:
                    subset_rows_by_family[family_key].add(
                        (other_plan.method_name, other_class_names)
                    )
                elif not family_classes < other_classes:
                    overlap_specs_by_family[family_key].add(
                        cls._member_spec(
                            other_plan.method_name,
                            other_class_names,
                        )
                    )
        return tuple(
            cls(
                _method_plans=tuple(group),
                strict_subset_family_specs=subset_specs,
                partial_overlap_family_specs=overlap_specs,
                lattice_node_count=len(lattice_nodes_by_family[family_key]),
                lattice_edge_count=len(lattice_edges_by_family[family_key]),
            )
            for family_key, group in exact_groups.items()
            for subset_specs, overlap_specs in (
                cls._partitioned_axis_specs(
                    subset_rows_by_family[family_key],
                    overlap_specs_by_family[family_key],
                ),
            )
        )

    @classmethod
    def _partitioned_axis_specs(
        cls,
        subset_rows: set[_MethodFamilyAxisRow],
        overlap_axis_specs: set[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        nonorthogonal_subset_rows: set[_MethodFamilyAxisRow] = set()
        subset_row_tuple = tuple(subset_rows)
        for left_index, left_row in enumerate(subset_row_tuple):
            left_classes = set(left_row[1])
            for right_row in subset_row_tuple[left_index + 1 :]:
                right_classes = set(right_row[1])
                if not (left_classes & right_classes):
                    continue
                if left_classes < right_classes or right_classes < left_classes:
                    continue
                nonorthogonal_subset_rows.update((left_row, right_row))

        clean_subset_rows = subset_rows - nonorthogonal_subset_rows
        clean_subset_specs = {
            cls._member_spec(method_name, class_names)
            for method_name, class_names in clean_subset_rows
        }
        nonorthogonal_specs = {
            cls._member_spec(method_name, class_names)
            for method_name, class_names in nonorthogonal_subset_rows
        }
        return (
            sorted_tuple(clean_subset_specs),
            sorted_tuple(overlap_axis_specs | nonorthogonal_specs),
        )


@dataclass(frozen=True)
class _MethodFamilyBaseEvidence(_MethodPlanEvidenceABC):
    """All proved method families descending from one resolved base symbol."""

    base_symbol: str
    families: tuple[_MethodFamilyEvidence, ...]

    @classmethod
    def from_families(
        cls,
        families: tuple[_MethodFamilyEvidence, ...],
    ) -> tuple[_MethodFamilyBaseEvidence, ...]:
        families_by_base: dict[str, list[_MethodFamilyEvidence]] = defaultdict(list)
        for family in families:
            families_by_base[family.base_symbol].append(family)
        return tuple(
            cls(base_symbol, tuple(base_families))
            for base_symbol, base_families in families_by_base.items()
        )

    @cached_property
    def method_plans(self) -> tuple[_MethodFamilyMethodPlan, ...]:
        return tuple(
            method_plan
            for family in self.families
            for method_plan in family.method_plans
        )

    @property
    def base_name(self) -> str:
        return self.families[0].base_name

    @cached_property
    def class_sets(self) -> tuple[frozenset[str], ...]:
        return tuple(frozenset(family.class_names) for family in self.families)

    @cached_property
    def has_global_structure(self) -> bool:
        for left_index, left_classes in enumerate(self.class_sets):
            for right_classes in self.class_sets[left_index + 1 :]:
                if (left_classes & right_classes) and left_classes != right_classes:
                    return True
        return False

    @cached_property
    def lattice_edge_count(self) -> int:
        edges: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for left_index, left_classes in enumerate(self.class_sets):
            for right_classes in self.class_sets[left_index + 1 :]:
                intersection = left_classes & right_classes
                if not intersection:
                    continue
                intersection_names = sorted_tuple(intersection)
                for class_set in (left_classes, right_classes):
                    class_names = sorted_tuple(class_set)
                    if intersection_names != class_names:
                        edges.add((intersection_names, class_names))
        return len(edges)

    @cached_property
    def certificate(self) -> CompressionCertificate:
        return CompressionCertificate.from_object_family(
            manual_object_count=sum(
                len(method_plan.class_names)
                * method_plan.profile.shared_statement_count
                for method_plan in self.method_plans
            ),
            replacement_shape=ObjectFamilyShape(
                shared_objects=("inheritance_lattice", "abc_base"),
                per_axis_objects=("residue_declaration",),
            ),
            semantic_axes=(
                *(method_plan.method_name for method_plan in self.method_plans),
                *(family.group_spec for family in self.families),
            ),
            residual_object_count=self.residue_declaration_count,
            provenance_object_count=len(self.families),
            independent_source_count=len(
                {
                    class_name
                    for method_plan in self.method_plans
                    for class_name in method_plan.class_names
                }
            ),
        )

    def candidate(self) -> OverlappingInheritanceFamiliesCandidate | None:
        if len(self.families) < 2 or not self.has_global_structure:
            return None
        certificate = self.certificate
        if not certificate.pays_rent:
            return None
        class_names = sorted_tuple(
            {
                class_name
                for method_plan in self.method_plans
                for class_name in method_plan.class_names
            }
        )
        return OverlappingInheritanceFamiliesCandidate(
            file_path=self.file_paths[0],
            line=min(self.line_numbers),
            base_name=self.base_name,
            class_names=class_names,
            method_names=self.method_names,
            family_specs=tuple(family.group_spec for family in self.families),
            strict_subset_family_specs=sorted_tuple(
                axis_spec
                for family in self.families
                for axis_spec in family.strict_subset_family_specs
            ),
            partial_overlap_family_specs=sorted_tuple(
                axis_spec
                for family in self.families
                for axis_spec in family.partial_overlap_family_specs
            ),
            file_paths=self.file_paths,
            line_numbers=self.line_numbers,
            method_symbols=self.method_symbols,
            shared_statement_count=self.shared_statement_count,
            residue_declaration_count=self.residue_declaration_count,
            leaf_residue_names=self.leaf_residue_names,
            lattice_node_count=len(
                {family.class_names for family in self.families}
                | {
                    sorted_tuple(set(left.class_names) & set(right.class_names))
                    for left in self.families
                    for right in self.families
                    if set(left.class_names) & set(right.class_names)
                }
            ),
            lattice_edge_count=self.lattice_edge_count,
            line_count=self.line_count,
            compression_certificate=certificate,
        )


@dataclass(frozen=True)
class CompactMethodFamilyCandidates(Generic[MethodFamilyCandidateItemT]):
    """One runtime candidate family carrying its nominal element type."""

    candidate_type: type[MethodFamilyCandidateItemT]
    candidates: tuple[MethodFamilyCandidateItemT, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(candidate, self.candidate_type) for candidate in self.candidates
        ):
            raise TypeError(
                f"Candidate family violates its {self.candidate_type.__qualname__} "
                "declaration"
            )


CompactMethodFamilyCandidate: TypeAlias = (
    ExactMethodRoleComponent
    | ExactLeafMethodAncestorPromotionComponent
    | SemanticOverlapMethodCandidate
    | SemanticOverlapMethodFamilyCandidate
    | OverlappingInheritanceFamiliesCandidate
    | SemanticOverlapResidueAxisCandidate
)


@dataclass(frozen=True)
class CompactMethodFamilyContext:
    candidate_families: tuple[
        CompactMethodFamilyCandidates[CompactMethodFamilyCandidate], ...
    ]

    def __post_init__(self) -> None:
        candidate_types = tuple(
            family.candidate_type for family in self.candidate_families
        )
        if len(candidate_types) != len(frozenset(candidate_types)):
            raise TypeError(
                f"{type(self).__name__} requires one family per candidate type"
            )

    def candidates_for(
        self,
        candidate_type: type[MethodFamilyCandidateItemT],
    ) -> tuple[MethodFamilyCandidateItemT, ...]:
        """Project the unique family carrying the requested nominal type."""

        matching_families = tuple(
            family
            for family in self.candidate_families
            if family.candidate_type is candidate_type
        )
        if len(matching_families) != 1:
            raise TypeError(
                f"{type(self).__name__} declares {len(matching_families)} candidate "
                "families "
                f"for {candidate_type.__module__}.{candidate_type.__qualname__}"
            )
        return cast(
            tuple[MethodFamilyCandidateItemT, ...],
            matching_families[0].candidates,
        )

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> CompactMethodFamilyContext:
        if class_index is None:
            class_index = build_compact_class_family_index(projections)
        method_plans = _MethodFamilyMethodPlan.from_projections(
            projections,
            class_index,
        )
        families = _MethodFamilyEvidence.from_method_plans(method_plans)
        method_candidates_by_key: dict[
            tuple[str, str, tuple[str, ...]],
            SemanticOverlapMethodCandidate,
        ] = {}
        for family in families:
            for candidate in family.semantic_overlap_candidates():
                method_candidates_by_key.setdefault(
                    (
                        family.base_symbol,
                        candidate.method_name,
                        candidate.class_names,
                    ),
                    candidate,
                )
        exact_promotion_builder = (
            ExactLeafMethodAncestorPromotionComponentBuilder.from_projections(
                projections,
                class_index=class_index,
            )
        )
        exact_method_candidates = ExactMethodRoleComponentBuilder(
            exact_promotion_builder
        ).proven_components
        method_candidates = sorted_tuple(
            method_candidates_by_key.values(),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.base_name,
                candidate.method_name,
            ),
        )
        family_candidates = sorted_tuple(
            (
                candidate
                for family in families
                if (candidate := family.family_candidate()) is not None
            ),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.base_name,
                candidate.method_names,
            ),
        )
        global_candidates = sorted_tuple(
            (
                candidate
                for base_evidence in _MethodFamilyBaseEvidence.from_families(families)
                if (candidate := base_evidence.candidate()) is not None
            ),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.base_name,
            ),
        )
        residue_axis_candidates = sorted_tuple(
            (
                candidate
                for family in families
                if (candidate := family.residue_axis_candidate()) is not None
            ),
            key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.base_name,
                candidate.method_names,
            ),
        )
        return cls(
            candidate_families=(
                CompactMethodFamilyCandidates(
                    ExactMethodRoleComponent,
                    exact_method_candidates,
                ),
                CompactMethodFamilyCandidates(
                    ExactLeafMethodAncestorPromotionComponent,
                    exact_promotion_builder.proven_components,
                ),
                CompactMethodFamilyCandidates(
                    SemanticOverlapMethodCandidate,
                    method_candidates,
                ),
                CompactMethodFamilyCandidates(
                    SemanticOverlapMethodFamilyCandidate,
                    family_candidates,
                ),
                CompactMethodFamilyCandidates(
                    OverlappingInheritanceFamiliesCandidate,
                    global_candidates,
                ),
                CompactMethodFamilyCandidates(
                    SemanticOverlapResidueAxisCandidate,
                    residue_axis_candidates,
                ),
            )
        )


def _pipeline_body_stages(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[PipelineAssemblyStage, ...] | None:
    body = list(function.body)
    if body and is_docstring_statement(body[0]):
        body = body[1:]
    if len(body) < 2:
        return None
    stages: list[PipelineAssemblyStage] = []
    for statement in body:
        stage = _pipeline_stage(statement)
        if stage is None:
            return None
        stages.append(stage)
    if not stages or stages[-1].kind != _PIPELINE_RETURN_STAGE:
        return None
    return tuple(stages)


@dataclass(frozen=True)
class _PipelineStageSource:
    kind: str
    call: ast.Call
    output_arity: int


def _pipeline_stage_source(statement: ast.stmt) -> _PipelineStageSource | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is not None:
        call = as_ast(assignment.value, ast.Call)
        output_arity = _assignment_target_arity(single_assign_target(assignment))
        if call is None or output_arity is None:
            return None
        return _PipelineStageSource(_PIPELINE_ASSIGN_STAGE, call, output_arity)
    call = return_call(statement)
    if call is None:
        return None
    return _PipelineStageSource(_PIPELINE_RETURN_STAGE, call, 0)


def _pipeline_stage(statement: ast.stmt) -> PipelineAssemblyStage | None:
    source = _pipeline_stage_source(statement)
    callee_name = (
        ExpressionReference.decoded(source.call.func).terminal_name
        if source is not None
        else None
    )
    if source is None or callee_name is None:
        return None
    keyword_names = CallKeywords.of(source.call.keywords).names
    return PipelineAssemblyStage(
        kind=source.kind,
        callee_name=callee_name,
        output_arity=source.output_arity,
        arg_count=len(source.call.args) + len(keyword_names),
        keyword_names=keyword_names,
    )


_STATIC_OBSERVATION_DETECTOR_BASE_NAMES = frozenset({StaticModulePatternDetector.__name__})
_STATIC_OBSERVATION_CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
)


def _single_payload_parameter_name(method: ast.FunctionDef) -> str | None:
    positional = (*method.args.posonlyargs, *method.args.args)
    payload_names = tuple(
        (argument.arg for argument in positional if argument.arg not in {"self", "cls"})
    )
    if variadic_parameters(method.args):
        return None
    return single_item(payload_names) if len(payload_names) == 1 else None


def _parameter_name_is_reused(
    statements: Sequence[ast.stmt], parameter_name: str
) -> bool:
    return any(
        (
            isinstance(node, ast.Name) and node.id == parameter_name
            for statement in statements
            for node in ast.walk(statement)
        )
    )


def _first_named_call_assignment(
    statements: Sequence[ast.stmt],
) -> NamedCallAssignment | None:
    first_statement = single_item(statements[:1])
    assignment = as_ast(first_statement, ast.Assign)
    return named_call_assignment(assignment) if assignment is not None else None


def _call_is_cast_of_parameter(call: ast.Call, parameter_name: str) -> bool:
    return (
        ExpressionReference.decoded(call.func).terminal_name == "cast"
        and len(call.args) == 2
        and (name_id(call.args[1]) == parameter_name)
    )


def _self_finding_spec_build_call(node: ast.AST) -> ast.Call | None:
    call = as_ast(node, ast.Call)
    if call is None or len(call.args) < 1:
        return None
    if not _is_self_finding_spec_build_func(call.func):
        return None
    if not _is_self_detector_id_attribute(call.args[0]):
        return None
    return call


def _is_self_finding_spec_build_func(node: ast.AST) -> bool:
    build_attr = as_ast(node, ast.Attribute)
    spec_attr = as_ast(build_attr.value if build_attr else None, ast.Attribute)
    return (
        build_attr is not None
        and build_attr.attr == "build"
        and (spec_attr is not None)
        and (spec_attr.attr == "finding_spec")
        and (name_id(spec_attr.value) == "self")
    )


def _is_self_detector_id_attribute(node: ast.AST) -> bool:
    detector_id_arg = as_ast(node, ast.Attribute)
    return (
        detector_id_arg is not None
        and detector_id_arg.attr == "detector_id"
        and (name_id(detector_id_arg.value) == "self")
    )


def _class_declares_detector_id(node: ast.ClassDef) -> bool:
    return any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "detector_id" for target in statement.targets))
            for statement in node.body
        )
    )


def _finding_spec_build_boilerplate_class_candidates(
    module: ParsedModule, node: ast.ClassDef
) -> tuple[ClassMethodLineWitnessCandidate, ...]:
    if not _class_declares_detector_id(node):
        return ()
    return tuple(
        (
            ClassMethodLineWitnessCandidate(
                file_path=module.file_path,
                line=child.lineno,
                class_name=node.name,
                method_name=statement.name,
            )
            for statement in node.body
            if isinstance(statement, ast.FunctionDef)
            for child in ast.walk(statement)
            if _self_finding_spec_build_call(child) is not None
        )
    )


def _finding_spec_build_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[ClassMethodLineWitnessCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        ast.ClassDef,
        _finding_spec_build_boilerplate_class_candidates,
    )


def _source_segment(module: ParsedModule, node: ast.expr) -> str:
    return module.source_segments.segment_for_node(node) or ast.unparse(node)


class HelperSyntaxProjectionAuthority:
    def annotation_root_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self.annotation_root_name(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.annotation_root_name(node.left) or self.annotation_root_name(
                node.right
            )
        return None

    def typed_field_map(self, node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
        typed_fields: dict[str, str] = {}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                if CLASSVAR_ANNOTATION_AUTHORITY.matches(statement.annotation):
                    continue
                typed_fields.setdefault(
                    statement.target.id, ast.unparse(statement.annotation)
                )
                continue
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if statement.name != "__init__":
                continue
            parameter_annotations = _function_parameter_annotation_map(statement)
            for inner in statement.body:
                target: ast.AST | None = None
                value: ast.AST | None = None
                if isinstance(inner, ast.Assign) and len(inner.targets) == 1:
                    target = inner.targets[0]
                    value = inner.value
                elif isinstance(inner, ast.AnnAssign):
                    target = inner.target
                    value = inner.value
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and (target.value.id == "self")
                    and isinstance(value, ast.Name)
                    and (value.id in parameter_annotations)
                ):
                    continue
                typed_fields.setdefault(target.attr, parameter_annotations[value.id])
        return sorted_tuple(typed_fields.items())

HELPER_SYNTAX_PROJECTION_AUTHORITY = HelperSyntaxProjectionAuthority()


class TupleIndexSemanticOpacityCandidateCollector:
    """Collect tuple-index opacity under one named detector grammar."""

    carrier_calls: ClassVar[frozenset[str]] = frozenset(
        {"Maybe.of", "project", "map", "filter", "with_projection", "bind_all"}
    )

    @classmethod
    def collect(
        cls,
        module: ParsedModule,
    ) -> tuple[TupleIndexSemanticOpacityCandidate, ...]:
        return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
            module,
            cls.for_function,
            sort_key=lambda candidate: (
                candidate.file_path,
                candidate.line,
                candidate.function_name,
            ),
        )

    @classmethod
    def collect_source(
        cls,
        module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
        config: DetectorConfig,
    ) -> tuple[TupleIndexSemanticOpacityCandidate, ...] | None:
        del syntax_index, config
        may_contain_candidate = any(
            token in module.source for token in cls.carrier_calls
        ) and re.search(r"\[[^\]]*\d[^\]]*\]", module.source) is not None
        return None if may_contain_candidate else ()

    @classmethod
    def for_function(
        cls,
        module: ParsedModule,
        qualname: str,
        function: NamedFunctionNode,
    ) -> tuple[TupleIndexSemanticOpacityCandidate, ...]:
        body_nodes = walk_function_body_nodes(function.body)
        carrier_call_names = sorted_tuple(
            {
                call_name
                for node in body_nodes
                if isinstance(node, ast.Call)
                for call_name in (ExpressionReference.decoded(node.func).terminal_name,)
                if call_name in cls.carrier_calls
            }
        )
        index_paths = sorted_tuple(
            {
                f"{root}[{']['.join(str(index) for index in indexes)}]"
                for node in body_nodes
                if isinstance(node, ast.Subscript)
                for path in (cls.numeric_subscript_path(node),)
                if path is not None
                for root, indexes in (path,)
                if root not in {"args", "body", "items"}
                if len(indexes) >= 2 or root in {"pair", "pairs", "call_pair"}
            }
        )
        if not carrier_call_names or not index_paths:
            return ()
        return (
            TupleIndexSemanticOpacityCandidate(
                file_path=module.file_path,
                line=function.lineno,
                function_name=qualname,
                index_expressions=index_paths,
                nested_index_count=len(index_paths),
                carrier_call_names=carrier_call_names,
            ),
        )

    @staticmethod
    def numeric_subscript_path(
        node: ast.AST,
    ) -> tuple[str, tuple[int, ...]] | None:
        indexes: list[int] = []
        current = node
        while isinstance(current, ast.Subscript):
            if not isinstance(current.slice, ast.Constant) or not isinstance(
                current.slice.value, int
            ):
                return None
            indexes.append(current.slice.value)
            current = current.value
        root_name = name_id(current)
        if root_name is None:
            return None
        return root_name, tuple(reversed(indexes))


def _dataclass_config_field_names(declared: ClassStatement) -> tuple[str, ...]:
    return declared.annotated_names if declared.is_dataclass else ()


def _from_namespace_keyword_names(
    node: ast.ClassDef,
) -> tuple[int, tuple[str, ...]] | None:
    for statement in node.body:
        if (
            not isinstance(statement, ast.FunctionDef)
            or statement.name != "from_namespace"
        ):
            continue
        for call in ast.walk(statement):
            if isinstance(call, ast.Call) and name_id(call.func) == "cls":
                keyword_names = CallKeywords.of(call.keywords).names
                if keyword_names:
                    return (statement.lineno, keyword_names)
    return None


def _argument_spec_field_name(node: ast.AST) -> str | None:
    return (
        Maybe.of(as_ast(node, ast.Call))
        .filter(lambda call: (name_id(call.func) or "").endswith("ArgumentSpec"))
        .map(lambda call: {keyword.arg: keyword.value for keyword in call.keywords})
        .combine(
            lambda keywords: as_ast(keywords.get("flags"), ast.Tuple),
            lambda keywords, flags: (keywords, flags) if flags.elts else None,
        )
        .combine(
            lambda context: constant_value(context[1].elts[0]),
            lambda context, first_flag: (
                (
                    dest_name
                    if isinstance(
                        dest_name := constant_value(context[0].get("dest")), str
                    )
                    else first_flag.removeprefix("--").replace("-", "_")
                )
                if isinstance(first_flag, str) and first_flag.startswith("--")
                else None
            ),
        )
        .unwrap_or_none()
    )


def _cli_argument_spec_fields(
    module: ParsedModule,
) -> NamedStringSequenceSpecs:
    specs: MutableNamedStringSequenceSpecs = []
    for binding in module.fact(ModuleNamedValues).bindings:
        if not isinstance(binding.value, ast.Tuple):
            continue
        field_names = tuple(
            (
                field_name
                for field_name in (
                    _argument_spec_field_name(element) for element in binding.value.elts
                )
                if field_name is not None
            )
        )
        if field_names:
            specs.append((binding.name, binding.line, field_names))
    return tuple(specs)


def _assignment_target_arity(target: ast.AST) -> int | None:
    if isinstance(target, ast.Name):
        return 1
    if isinstance(target, (ast.Tuple, ast.List)):
        if not target.elts or not all(
            (isinstance(item, ast.Name) for item in target.elts)
        ):
            return None
        return len(target.elts)
    return None


_SCHEMA_ACCESSOR_FETCH_METHOD_NAMES = frozenset({"required", "optional"})
_SCHEMA_ACCESSOR_MIN_METHODS = 4
_SCHEMA_ACCESSOR_COPY_CALL_NAMES = BuiltinCallName.schema_accessor_copy_call_names()
_SCHEMA_ACCESSOR_SELF_NAME = "self"


def _schema_accessor_self_fetch_func(call: ast.Call) -> ast.Attribute | None:
    return (
        Maybe.of(as_ast(call.func, ast.Attribute))
        .filter(
            lambda func: (
                isinstance(func.value, ast.Name)
                and func.value.id == _SCHEMA_ACCESSOR_SELF_NAME
            )
        )
        .unwrap_or_none()
    )


def _schema_accessor_fetch_call(value: ast.AST) -> tuple[str, str, str] | None:
    return (
        Maybe.of(as_ast(value, ast.Call))
        .filter(lambda call: len(call.args) == 1 and not call.keywords)
        .combine(
            _schema_accessor_self_fetch_func,
            lambda call, func: (call, func),
        )
        .filter(lambda context: context[1].attr in _SCHEMA_ACCESSOR_FETCH_METHOD_NAMES)
        .combine(
            lambda context: ExpressionReference.decoded(
                context[0].args[0]
            ).attribute_chain,
            lambda context, field_chain: (context[1], field_chain),
        )
        .filter(lambda context: len(context[1]) >= 2)
        .map(
            lambda context: (
                context[0].attr,
                ".".join(context[1][:-1]),
                context[1][-1],
            )
        )
        .unwrap_or_none()
    )


def _node_mentions_name(node: ast.AST, local_name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == local_name
        for item in ast.walk(node)
    )


def _compare_mentions_none(node: ast.Compare, local_name: str) -> bool:
    operands = (node.left, *node.comparators)
    return _node_mentions_name(node, local_name) and any(
        CompactValueExpression.project(operand).value_is_none_literal
        for operand in operands
    )


def _schema_accessor_coercion_kinds(
    function: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> tuple[str, ...]:
    kinds: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Compare) and _compare_mentions_none(node, local_name):
            kinds.add("none_guard")
        if not isinstance(node, ast.Call):
            continue
        call_name = ExpressionReference.decoded(node.func).terminal_name
        if (
            BuiltinCallName.is_isinstance_name(call_name)
            and node.args
            and _node_mentions_name(node.args[0], local_name)
        ):
            kinds.add("runtime_type_guard")
        elif call_name in _SCHEMA_ACCESSOR_COPY_CALL_NAMES and any(
            _node_mentions_name(argument, local_name) for argument in node.args
        ):
            kinds.add(f"{call_name}_coercion")
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("from_")
            and any(_node_mentions_name(argument, local_name) for argument in node.args)
        ):
            kinds.add(node.func.attr)
    return tuple(sorted(kinds))


def _schema_accessor_returns_local(
    function: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> bool:
    return any(
        _node_mentions_name(returned, local_name)
        for returned in StatementValues.of(nodes_of(ast.Return, function)).values
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
