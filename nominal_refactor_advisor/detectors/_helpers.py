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
    Callable,
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
from nominal_refactor_advisor.class_syntax_facets import annotation_type_names
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
    single_name_targets,
)
from nominal_refactor_advisor.ast_tools import (
    BuiltinCall,
    BuiltinCallName,
    Getattr,
    NamespaceCall,
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
    FiniteAxisSystem,
    ObjectFamilyShape,
)
from nominal_refactor_advisor.semantic_description_length import (
    ClassFamilyCompressionProfile,
    CompressionCertificate,
)
import ast
from typing import Iterable
from nominal_refactor_advisor.ast_tools import (
    _walk_nodes,
    is_docstring_statement,
    ParsedModule,
    ScopedAstObservation,
    statements_without_docstring,
)
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.models import SourceLocation
from nominal_refactor_advisor.name_algebra import CLASS_NAME_ALGEBRA
from nominal_refactor_advisor.observation_shapes import BuilderCallShape
from nominal_refactor_advisor.semantic_match import (
    as_ast,
    attribute_call_match,
    constant_value,
    Maybe,
    name_id,
    named_call_assignment,
    named_value_binding,
    NamedCallAssignment,
    return_call,
    single_assign_target,
    single_item,
    single_return_value,
)
from nominal_refactor_advisor.detectors._base import (
    _module_builder_call_shapes,
    _PIPELINE_ASSIGN_STAGE,
    _PIPELINE_RETURN_STAGE,
    _self_attr_name,
    CANDIDATE_COLLECTION_AUTHORITY,
    ClassFamilyWitnessCarrier,
    ClassMethodLineWitnessCandidate,
    DetectorConfig,
    GuardedWrapperSpecPair,
    IssueDetector,
    NamedFunctionNode,
    OverlappingInheritanceFamiliesCandidate,
    PipelineAssemblyStage,
    PropertyAliasHookGroup,
    ResultAssemblyPipelineFunction,
    SchemaAccessorFamilyCandidate,
    SemanticOverlapMethodCandidate,
    SemanticOverlapMethodFamilyCandidate,
    SemanticOverlapResidueAxisCandidate,
    StaticModulePatternDetector,
    StaticTypedObservationDetectorCandidate,
    StructuralObservationPropertyCandidate,
    TupleIndexSemanticOpacityCandidate,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassStatement,
    ModuleClassStatements,
)
from nominal_refactor_advisor.detectors._substrate_support import (
    ModuleNamedValues,
    _constant_string,
    _is_private_symbol_name,
)

BaseBundleClassGroups: TypeAlias = dict[tuple[str, ...], list[ast.ClassDef]]
NamedStringSequenceSpec: TypeAlias = tuple[str, int, tuple[str, ...]]
NamedStringSequenceSpecs: TypeAlias = tuple[NamedStringSequenceSpec, ...]
MutableNamedStringSequenceSpecs: TypeAlias = list[NamedStringSequenceSpec]
NamedStringSequenceSpecsByName: TypeAlias = dict[str, MutableNamedStringSequenceSpecs]
DerivedQuerySignature: TypeAlias = tuple[str | None, str, str, str]
DerivedQuerySpecsBySignature: TypeAlias = dict[
    DerivedQuerySignature, MutableNamedStringSequenceSpecs
]

_APPEND_METHOD_NAME = "append"


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


def _class_direct_constant_string_assignments(
    declared: ClassStatement,
) -> dict[str, str]:
    return {
        name: string_value
        for name, value in declared.direct_assignments.items()
        if (string_value := _constant_string(value)) is not None
    }


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


def _enum_key_family(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    return (node.value.id, node.attr)


def _is_detectorish_class(declared: ClassStatement) -> bool:
    if declared.node.name.endswith("Detector"):
        return True
    return bool(
        IssueDetector.detector_family_base_names()
        & set(declared.base_terminal_names)
    )


def _finding_build_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for node in _walk_nodes(method):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "build":
            continue
        value = node.func.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "finding_spec"
            and isinstance(value.value, ast.Name)
            and (value.value.id == "self")
        ):
            continue
        return node
    return None


def _call_display_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _build_call_keyword_helper_name(
    build_call: ast.Call, keyword_name: str
) -> str | None:
    for keyword in build_call.keywords:
        if keyword.arg != keyword_name:
            continue
        if isinstance(keyword.value, ast.Call):
            return _call_display_name(keyword.value)
    return None


def _candidate_source_name_from_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    assigned_calls: dict[str, str] = {}
    for statement in statements_without_docstring(method.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and isinstance(statement.value, ast.Call):
                call_name = _call_display_name(statement.value)
                if call_name is not None:
                    assigned_calls[target.id] = call_name
        if isinstance(statement, ast.For):
            iterator = statement.iter
            if isinstance(iterator, ast.Call):
                return _call_display_name(iterator)
            if isinstance(iterator, ast.Name):
                return assigned_calls.get(iterator.id)
    return None


def _is_observation_spec_class(declared: ClassStatement) -> bool:
    if declared.node.name.endswith("ObservationSpec"):
        return True
    return bool(
        {
            "ObservationShapeSpec",
            "FunctionObservationSpec",
            "AssignObservationSpec",
            "ContextForwardingShapeSpec",
        }
        & set(declared.base_terminal_names)
    )


def _delegate_name_from_return(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        outer_name = _call_display_name(node)
        if outer_name in _SEQUENCE_WRAPPER_CALL_NAMES and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call):
                return _call_display_name(inner)
        return outer_name
    return None


def _guard_role_name(node: ast.AST) -> str:
    return ScopedAstObservation.guard_role_name_from_text(ast.unparse(node))


def _scope_role_name(node: ast.AST) -> str:
    return ScopedAstObservation.scope_role_name_from_text(ast.unparse(node))


def _name_mentions(node: ast.AST, name: str) -> bool:
    return any(
        (
            isinstance(current, ast.Name) and current.id == name
            for current in _walk_nodes(node)
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
                for current in _walk_nodes(exc)
                if isinstance(current, ast.Name)
            }
        )
        if exc_name is not None:
            return (exc_name, referenced_names)
    exc_name = ExpressionReference.decoded(exc).terminal_name
    if exc_name is not None:
        return (exc_name, ())
    return None


def _linear_query_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...], str, str] | None:
    body = statements_without_docstring(node.body)
    return (
        Maybe.of(_linear_query_loop(body))
        .combine(
            lambda loop: _linear_query_return_expr(
                loop, cast(ast.Name, loop.target).id
            ),
            lambda loop, return_expr: (loop, return_expr),
        )
        .combine(
            lambda _context: _linear_query_raised_exception(body),
            lambda context, raised: (context[0], context[1], raised),
        )
        .combine(
            lambda context: _linear_query_key_names(
                node, context[0], context[1], context[2][1]
            ),
            lambda context, query_key_names: (
                (
                    ast.unparse(context[0].iter),
                    query_key_names,
                    ast.unparse(context[1]),
                    context[2][0],
                )
                if query_key_names
                else None
            ),
        )
        .unwrap_or_none()
    )


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
    return_exprs = StatementValues.of(
        current for current in _walk_nodes(loop) if isinstance(current, ast.Return)
    ).values
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
                    for current in _walk_nodes(loop)
                )
            )
        )
    )
    return query_key_names


def _linear_query_source_owner(
    qualname: str,
    source_expression: str,
) -> str | None:
    expression = ast.parse(source_expression, mode="eval").body
    if not (
        HELPER_SUPPORT_PROJECTION_AUTHORITY.expression_root_names(expression)
        & _IMPLICIT_METHOD_PARAMETER_NAMES
    ):
        return None
    owner_name, separator, _method_name = qualname.rpartition(".")
    return owner_name if separator else qualname


def _structural_observation_property_candidates_for_class(
    module: ParsedModule, declared: ClassStatement
) -> Iterable[StructuralObservationPropertyCandidate]:
    node = declared.node
    for method in declared.methods:
        if not method.spells("property"):
            continue
        statement = method.node
        body = statements_without_docstring(statement.body)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            continue
        constructor_name = ExpressionReference.decoded(returned.func).terminal_name
        if constructor_name is None:
            continue
        keyword_names = sorted_tuple(CallKeywords.of(returned.keywords).names)
        if len(keyword_names) < 6:
            continue
        yield StructuralObservationPropertyCandidate(
            file_path=module.file_path,
            line=statement.lineno,
            subject_name=node.name,
            name_family=keyword_names,
            property_name=statement.name,
            constructor_name=constructor_name,
        )


def _structural_observation_property_candidates(
    module: ParsedModule,
) -> tuple[StructuralObservationPropertyCandidate, ...]:
    return tuple(
        candidate
        for declared in module.fact(ModuleClassStatements).statements
        for candidate in _structural_observation_property_candidates_for_class(
            module, declared
        )
    )


class MethodProjection:
    def aliases_to_self_attrs(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for subnode in _walk_nodes(method):
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


def _projection_builder_groups(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[tuple[BuilderCallShape, ...], ...]:
    grouped: dict[tuple[str, tuple[str, ...]], list[BuilderCallShape]] = defaultdict(
        list
    )
    for builder in _module_builder_call_shapes(module):
        if builder.is_nominal_owner_projection:
            continue
        if len(builder.field_names) < max(config.min_builder_keywords, 6):
            continue
        if not all(
            (
                _is_projection_like_builder_value(value)
                for value in builder.value_fingerprint
            )
        ):
            continue
        grouped[(builder.callee_name, builder.field_names)].append(builder)
    candidates: list[Sequence[BuilderCallShape]] = []
    for builders in grouped.values():
        if len(builders) < 3:
            continue
        source_arities = tuple(builder.source_arity for builder in builders)
        if min(source_arities) != 1 or max(source_arities) > 2:
            continue
        if len({builder.value_fingerprint for builder in builders}) < 2:
            continue
        if len({builder.symbol for builder in builders}) < 2:
            continue
        candidates.append(
            sorted_tuple(builders, key=lambda item: (item.file_path, item.lineno))
        )
    return sorted_tuple(
        candidates,
        key=lambda group: (group[0].file_path, group[0].lineno, group[0].callee_name),
    )


def _property_alias_hook_groups(
    module: ParsedModule,
) -> tuple[PropertyAliasHookGroup, ...]:
    grouped: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for declared in module.fact(ModuleClassStatements).statements:
        node = declared.node
        base_names = declared.family_base_names
        if not base_names:
            continue
        for method in declared.methods:
            if not method.spells("property"):
                continue
            statement = method.node
            if len(statement.args.args) != 1:
                continue
            body = statements_without_docstring(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Name)
                and (returned.value.id == "self")
            ):
                continue
            for base_name in base_names:
                grouped[base_name, statement.name, returned.attr].append(
                    (node.name, statement.lineno)
                )
    return tuple(
        (
            PropertyAliasHookGroup(
                file_path=module.file_path,
                base_name=base_name,
                property_name=property_name,
                returned_attribute=returned_attribute,
                class_names=tuple((class_name for class_name, _ in ordered)),
                line_numbers=tuple((line for _, line in ordered)),
            )
            for (base_name, property_name, returned_attribute), items in sorted(
                grouped.items()
            )
            if len(items) >= 2
            for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
        )
    )


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


def _guarded_wrapper_spec_pairs(
    module: ParsedModule,
) -> tuple[GuardedWrapperSpecPair, ...]:
    wrapper_functions = {
        function_name: (lineno, node_types)
        for function_name, lineno, node_types in _guarded_wrapper_function_candidates(
            module
        )
    }
    pairs: list[GuardedWrapperSpecPair] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            lineno = statement.lineno
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
            lineno = statement.lineno
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        constructor_name = ExpressionReference.decoded(value.func).terminal_name
        if constructor_name is None:
            continue
        referenced_functions = [
            keyword.value.id
            for keyword in CallKeywords.of(value.keywords).named
            if isinstance(keyword.value, ast.Name)
            and keyword.value.id in wrapper_functions
        ]
        if len(referenced_functions) != 1:
            continue
        node_types_node = next(
            (
                keyword.value
                for keyword in value.keywords
                if keyword.arg == "node_types"
            ),
            None,
        )
        if node_types_node is None:
            continue
        if isinstance(node_types_node, ast.Tuple):
            node_types = tuple(ast.unparse(item) for item in node_types_node.elts)
        else:
            node_types = (ast.unparse(node_types_node),)
        function_name = referenced_functions[0]
        function_line, function_node_types = wrapper_functions[function_name]
        if tuple(node_types) != function_node_types:
            continue
        pairs.append(
            GuardedWrapperSpecPair(
                file_path=module.file_path,
                spec_name=target.id,
                spec_line=lineno,
                function_name=function_name,
                function_line=function_line,
                constructor_name=constructor_name,
                node_types=function_node_types,
            )
        )
    return tuple(pairs)


def _reflective_lookup_shape(
    node: ast.AST,
) -> tuple[str, str, ast.AST] | None:
    if isinstance(node, ast.Call):
        match BuiltinCall.decoded(node):
            case Getattr(name_argument=selector_node) as call if (
                _constant_string(selector_node) is None
            ):
                return (call.builtin, ast.unparse(call.object_argument), selector_node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 1
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "__dict__"
        ):
            selector_node = node.args[0]
            if _constant_string(selector_node) is None:
                return ("dict.get", ast.unparse(node.func.value.value), selector_node)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(namespace := BuiltinCall.decoded(node.value), NamespaceCall)
        and (_constant_string(node.slice) is None)
    ):
        return (f"{namespace.builtin}[]", f"{namespace.builtin}()", node.slice)
    return None


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


def _class_init_concrete_param_backed_attrs(
    declared: ClassStatement,
) -> dict[str, str]:
    init_statement = declared.method_named("__init__")
    if init_statement is None:
        return {}
    init_method = init_statement.node
    parameter_type_names = {
        argument.arg: annotation_type_names(
            argument.annotation
        )
        for argument in ParameterAnnotations.of(
            tuple(init_method.args.posonlyargs)
            + tuple(init_method.args.args)
            + tuple(init_method.args.kwonlyargs)
        ).annotated
    }
    parameter_names = set(parameter_type_names)
    attr_type_names: dict[str, str] = {}
    for subnode in _walk_nodes(init_method):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
        attr_name = None if target is None else _self_attr_name(target)
        if attr_name is None or value is None:
            continue
        param_name = _param_backed_name(value, parameter_names)
        if param_name is None:
            continue
        type_names = parameter_type_names.get(param_name, ())
        if len(type_names) != 1:
            continue
        attr_type_names.setdefault(attr_name, type_names[0])
    return attr_type_names


def _receiver_self_attr_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Attribute):
        return _self_attr_name(node)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


_DECLARATIVE_FAMILY_DEFINITION_BASE_NAMES = frozenset(
    {
        "SingleShapeFamilyDefinition",
        "RegisteredShapeFamilyDefinition",
        "RegisteredObservationFamilyDefinition",
        "TypedLiteralObservationFamilyDefinition",
    }
)


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


def _contiguous_base_bundles(
    base_names: tuple[str, ...],
    *,
    minimum_width: int = 3,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            base_names[start:end]
            for start in range(len(base_names))
            for end in range(start + minimum_width, len(base_names) + 1)
            if all(_is_composable_base_name(name) for name in base_names[start:end])
        )
    )


def _is_contiguous_subtuple(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _maximal_repeated_base_bundle_items(
    grouped: BaseBundleClassGroups,
) -> tuple[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]], ...]:
    qualified = tuple(
        (
            (bundle, sorted_tuple(nodes, key=lambda node: node.lineno))
            for bundle, nodes in grouped.items()
            if len(nodes) >= 3
        )
    )
    maximal: list[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]]] = []
    for bundle, nodes in sorted(
        qualified,
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    ):
        class_names = {node.name for node in nodes}
        if any(
            (
                class_names <= {node.name for node in existing_nodes}
                and _is_contiguous_subtuple(bundle, existing_bundle)
                for existing_bundle, existing_nodes in maximal
            )
        ):
            continue
        maximal.append((bundle, nodes))
    return tuple(maximal)


def _registered_surface_roots(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    calls: list[ast.Call] = []

    def collect_calls(current: ast.AST) -> bool:
        binary = as_ast(current, ast.BinOp)
        if binary is not None and isinstance(binary.op, ast.Add):
            return collect_calls(binary.left) and collect_calls(binary.right)
        call = as_ast(current, ast.Call)
        if call is None:
            return False
        calls.append(call)
        return True

    if not collect_calls(node) or len(calls) < 2:
        return None
    accessors = tuple(
        (
            accessor
            for call in calls
            if (
                accessor := attribute_call_match(
                    call, owner_type=ast.Name, argument_count=0, allow_keywords=False
                )
            )
            is not None
        )
    )
    accessor_names = {accessor.attribute.attr for accessor in accessors}
    if len(accessor_names) != 1:
        return None
    accessor_name = next(iter(accessor_names))
    root_names = sorted_tuple((accessor.owner.id for accessor in accessors))
    return (accessor_name, root_names)


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


def _concrete_class_object_union_member_names(
    annotation: ast.AST | None,
) -> tuple[str, ...]:
    if annotation is None:
        return ()
    union_members = _annotation_union_members(annotation)
    member_names = tuple(
        (
            member_name
            for member in union_members
            if (member_name := _class_object_type_argument_name(member)) is not None
        )
    )
    if len(member_names) != len(union_members):
        return ()
    return tuple(dict.fromkeys(member_names))


def _parameter_class_call_attribute_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_name: str,
) -> tuple[str, ...]:
    attribute_names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == parameter_name
            ):
                attribute_names.add(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(
        ast.Module(body=statements_without_docstring(function.body), type_ignores=[])
    )
    return sorted_tuple(attribute_names)


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


def _common_contract_base_names(
    classes_by_name: dict[str, ClassStatement],
    ancestor_names_by_class: dict[str, tuple[str, ...]],
    member_type_names: tuple[str, ...],
    observed_attribute_names: tuple[str, ...],
) -> tuple[str, ...]:
    base_sets = tuple(
        (
            _class_def_declared_and_ancestor_names(
                classes_by_name, ancestor_names_by_class, member_type_name
            )
            for member_type_name in member_type_names
        )
    )
    common_base_names = {
        name
        for name in set.intersection(*base_sets)
        if ClassSymbolResolutionAuthority.establishes_nominal_family(name)
    }
    return sorted_tuple(
        (
            base_name
            for base_name in common_base_names
            if _base_defines_contract_attributes(
                classes_by_name, base_name, observed_attribute_names
            )
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


def _concrete_union_contract_name(
    member_type_names: tuple[str, ...],
    observed_attribute_names: tuple[str, ...],
) -> str:
    edge_tokens = _shared_edge_tokens(member_type_names)
    base_name = (
        "".join((token.title() for token in edge_tokens))
        if edge_tokens
        else f"{member_type_names[0]}Family"
    )
    role_suffix = (
        "Factory"
        if any(
            (
                attribute_name.startswith(_CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES)
                for attribute_name in observed_attribute_names
            )
        )
        else "Contract"
    )
    return base_name if base_name.endswith(role_suffix) else f"{base_name}{role_suffix}"


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


def _typed_observation_collection_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    if any(
        (
            isinstance(child, _STATIC_OBSERVATION_CONTROL_FLOW_NODES)
            for child in _walk_nodes(method)
        )
    ):
        return None
    collection_calls: list[tuple[str, str]] = []
    for child in _walk_nodes(method):
        if not isinstance(child, ast.Call):
            continue
        if name_id(child.func) != "_collect_typed_family_items":
            continue
        if len(child.args) < 3 or name_id(child.args[0]) != "module":
            continue
        family_name = name_id(child.args[1])
        observation_type_name = name_id(child.args[2])
        if family_name is not None and observation_type_name is not None:
            collection_calls.append((family_name, observation_type_name))
    return single_item(tuple(collection_calls))


def _source_location_from_line_symbol_call(
    node: ast.AST,
) -> bool:
    if not isinstance(node, ast.Call) or name_id(node.func) != SourceLocation.__name__:
        return False
    if len(node.args) < 3:
        return False
    return all(
        (
            isinstance(argument, ast.Attribute) and argument.attr == expected_attribute
            for argument, expected_attribute in zip(
                node.args[:3], ("file_path", "line", "symbol"), strict=True
            )
        )
    )


def _static_minimum_evidence_count(declared: ClassStatement) -> int | None:
    minimum_method = declared.method_named("_minimum_evidence")
    if minimum_method is None:
        return 1
    returns = tuple(
        child for child in _walk_nodes(minimum_method.node) if isinstance(child, ast.Return)
    )
    returned = single_item(returns)
    if returned is None:
        return None
    value = constant_value(returned.value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _static_summary_expression(
    module: ParsedModule, declared: ClassStatement
) -> str | None:
    return (
        Maybe.of(declared.method_named("_summary"))
        .map(
            lambda summary_method: tuple(
                child
                for child in _walk_nodes(summary_method.node)
                if isinstance(child, ast.Return)
            )
        )
        .project(single_item)
        .project(lambda returned: returned.value)
        .project(lambda value: _source_segment(module, value))
        .unwrap_or_none()
    )


def _static_typed_observation_detector_candidates(
    module: ParsedModule,
) -> tuple[StaticTypedObservationDetectorCandidate, ...]:
    candidates: list[StaticTypedObservationDetectorCandidate] = []
    for declared in module.fact(ModuleClassStatements).top_level:
        node = declared.node
        if node.end_lineno is None:
            continue
        if not (
            _STATIC_OBSERVATION_DETECTOR_BASE_NAMES
            & set(declared.base_terminal_names)
        ):
            continue
        evidence_statement = declared.method_named("_module_evidence")
        if evidence_statement is None:
            continue
        evidence_method = evidence_statement.node
        collection = _typed_observation_collection_call(evidence_method)
        if collection is None:
            continue
        if not any(
            _source_location_from_line_symbol_call(child)
            for child in _walk_nodes(evidence_method)
        ):
            continue
        minimum_evidence = _static_minimum_evidence_count(declared)
        summary_expression = _static_summary_expression(module, declared)
        if minimum_evidence is None or summary_expression is None:
            continue
        family_name, observation_type_name = collection
        candidates.append(
            StaticTypedObservationDetectorCandidate(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                observation_family_name=family_name,
                observation_type_name=observation_type_name,
                minimum_evidence_count=minimum_evidence,
                summary_expression=summary_expression,
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(candidates)


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
            for node in _walk_nodes(statement)
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


def _typed_candidate_cast_assignment(
    method: ast.FunctionDef,
) -> tuple[str, str, str] | None:
    parameter_name = _single_payload_parameter_name(method)
    body = statements_without_docstring(method.body)
    call_assignment = _first_named_call_assignment(body)
    if parameter_name is None or call_assignment is None:
        return None
    cast_call = call_assignment.call
    if not _call_is_cast_of_parameter(cast_call, parameter_name):
        return None
    if _parameter_name_is_reused(body[1:], parameter_name):
        return None
    return (parameter_name, call_assignment.target_name, ast.unparse(cast_call.args[0]))


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
            for child in _walk_nodes(statement)
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


ClassShapeT = TypeVar("ClassShapeT")
BuiltCandidateT = TypeVar("BuiltCandidateT")
ClassShapeProjector = Callable[[ast.ClassDef], ClassShapeT | None]
ClassShapeCandidateFactory = Callable[
    [ParsedModule, ast.ClassDef, ClassShapeT], BuiltCandidateT
]


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


def _enum_metadata_table_cases(module: ParsedModule) -> dict[str, tuple[str, int]]:
    tables: dict[str, tuple[str, int]] = {}
    for binding in module.fact(ModuleNamedValues).bindings:
        value = as_ast(binding.value, ast.Dict)
        if value is None:
            continue
        enum_key_names = tuple(
            (
                key.value.id
                for key in value.keys
                if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
            )
        )
        enum_names = set(enum_key_names)
        if len(enum_key_names) == len(value.keys) and len(enum_names) == 1:
            tables[binding.name] = (next(iter(enum_names)), len(value.keys))
    return tables


def _enum_metadata_property_table(method: ast.FunctionDef) -> str | None:
    returned = single_return_value(statements_without_docstring(method.body))
    lookup_source = returned.value if isinstance(returned, ast.Attribute) else returned
    lookup = as_ast(lookup_source, ast.Subscript)
    table_name = name_id(None if lookup is None else lookup.value)
    property_method = any(
        (name_id(decorator) == "property" for decorator in method.decorator_list)
    )
    return (
        table_name
        if property_method and lookup is not None and (name_id(lookup.slice) == "self")
        else None
    )


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
        for call in _walk_nodes(statement):
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


def _result_assembly_pipeline_functions(
    module: ParsedModule,
) -> tuple[ResultAssemblyPipelineFunction, ...]:
    functions: list[ResultAssemblyPipelineFunction] = []
    for qualname, function in module.syntax_index.named_functions:
        stages = _pipeline_body_stages(function)
        if stages is None:
            continue
        functions.append(
            ResultAssemblyPipelineFunction(
                file_path=module.file_path,
                qualname=qualname,
                lineno=function.lineno,
                stages=stages,
            )
        )
    return sorted_tuple(functions, key=lambda item: (item.lineno, item.qualname))


def _shared_pipeline_tail(
    left: ResultAssemblyPipelineFunction,
    right: ResultAssemblyPipelineFunction,
) -> tuple[PipelineAssemblyStage, ...]:
    shared: list[PipelineAssemblyStage] = []
    left_index = len(left.stages) - 1
    right_index = len(right.stages) - 1
    while left_index >= 0 and right_index >= 0:
        left_stage = left.stages[left_index]
        right_stage = right.stages[right_index]
        if left_stage.shape_key != right_stage.shape_key:
            break
        shared.append(left_stage)
        left_index -= 1
        right_index -= 1
    return tuple(reversed(shared))


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
        for item in _walk_nodes(node)
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
    for node in _walk_nodes(function):
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
        for returned in StatementValues.of(
            node for node in _walk_nodes(function) if isinstance(node, ast.Return)
        ).values
    )


def _schema_accessor_method_row(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str, str, str, int, int] | None:
    if method.name in _SCHEMA_ACCESSOR_FETCH_METHOD_NAMES:
        return None
    if _is_private_symbol_name(method.name):
        return None
    body = statements_without_docstring(list(method.body))
    fetches = tuple(
        (target.id, *fetch)
        for target, assigned in StatementValues.keyed(
            (target, statement)
            for statement in body
            if isinstance(statement, ast.Assign | ast.AnnAssign)
            for target in single_name_targets(statement)
        ).supplied
        if (fetch := _schema_accessor_fetch_call(assigned.value)) is not None
    )
    if len(fetches) != 1:
        return None
    local_name, fetch_mode, enum_name, field_name = fetches[0]
    coercion_kinds = _schema_accessor_coercion_kinds(method, local_name)
    if not coercion_kinds or not _schema_accessor_returns_local(method, local_name):
        return None
    line_count = max(1, (method.end_lineno or method.lineno) - method.lineno + 1)
    return (
        method.name,
        enum_name,
        field_name,
        fetch_mode,
        "+".join(coercion_kinds),
        method.lineno,
        line_count,
    )


def _schema_accessor_axis_system(
    rows: tuple[tuple[str, str, str, str, str, int, int], ...],
) -> FiniteAxisSystem[str, str]:
    return FiniteAxisSystem.from_rows(
        (
            (
                method_name,
                {
                    "method": method_name,
                    "enum": enum_name,
                    "field": field_name,
                    "fetch_mode": fetch_mode,
                    "coercion": coercion_kind,
                },
            )
            for (
                method_name,
                enum_name,
                field_name,
                fetch_mode,
                coercion_kind,
                _line,
                _line_count,
            ) in rows
        )
    )


def _schema_accessor_family_certificate(
    *,
    method_count: int,
    line_count: int,
    field_axes: tuple[str, ...],
    coercion_kind_count: int,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(line_count, method_count * max(coercion_kind_count, 1)),
        replacement_shape=ObjectFamilyShape.from_roles(
            ("payload_projection_schema", "typed_payload_projector"),
            axis=("projection_row",),
        ),
        semantic_axes=field_axes,
        residual_object_count=coercion_kind_count,
    )


def _schema_accessor_family_candidates(
    module: ParsedModule,
) -> tuple[SchemaAccessorFamilyCandidate, ...]:
    candidates: list[SchemaAccessorFamilyCandidate] = []
    for declared in module.fact(ModuleClassStatements).statements:
        class_node = declared.node
        grouped_rows: dict[str, list[tuple[str, str, str, str, str, int, int]]] = (
            defaultdict(list)
        )
        for method in declared.methods:
            row = _schema_accessor_method_row(method.node)
            if row is not None:
                grouped_rows[row[1]].append(row)
        for enum_name, rows in grouped_rows.items():
            if len(rows) < _SCHEMA_ACCESSOR_MIN_METHODS:
                continue
            ordered_rows = tuple(sorted(rows, key=lambda item: (item[5], item[0])))
            field_names = tuple((row[2] for row in ordered_rows))
            if len(frozenset(field_names)) < _SCHEMA_ACCESSOR_MIN_METHODS:
                continue
            axis_system = _schema_accessor_axis_system(ordered_rows)
            if not axis_system.determines(("field",), "method"):
                continue
            line_count = sum((row[6] for row in ordered_rows))
            coercion_kinds = tuple((row[4] for row in ordered_rows))
            field_axes = tuple(
                (f"{enum_name}.{field_name}" for field_name in field_names)
            )
            certificate = _schema_accessor_family_certificate(
                method_count=len(ordered_rows),
                line_count=line_count,
                field_axes=field_axes,
                coercion_kind_count=len(frozenset(coercion_kinds)),
            )
            if not certificate.pays_rent:
                continue
            candidates.append(
                SchemaAccessorFamilyCandidate(
                    file_path=module.file_path,
                    line=class_node.lineno,
                    class_name=class_node.name,
                    enum_name=enum_name,
                    method_names=tuple((row[0] for row in ordered_rows)),
                    field_names=field_names,
                    requirement_modes=tuple((row[3] for row in ordered_rows)),
                    coercion_kinds=coercion_kinds,
                    line_numbers=tuple((row[5] for row in ordered_rows)),
                    line_count=line_count,
                    compression_certificate=certificate,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.enum_name,
            ),
        )
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
