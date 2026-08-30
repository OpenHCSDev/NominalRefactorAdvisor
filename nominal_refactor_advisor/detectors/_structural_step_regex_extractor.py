"""Step/regex/extractor structural detector cohort."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from ..ast_tools import ParsedModule, _walk_nodes
from ..collection_algebra import sorted_tuple
from ..semantic_match import (
    Maybe,
    as_ast,
    ast_sequence,
    attribute_call_match,
    constant_value,
    name_id,
    named_call_assignment,
    single_assign_target,
    single_item,
)
from ._base import ClassMethodFamilyCandidate
from ._substrate_support import _trim_docstring_body




@dataclass(frozen=True)
class AccumulatorFoldFamilyCandidate(ClassMethodFamilyCandidate):
    accumulator_type_name: str
    result_method_name: str
    source_parameter_names: tuple[str, ...]
    step_method_names: tuple[str, ...]


@dataclass(frozen=True)
class RegexGroupExtractorFamilyCandidate(ClassMethodFamilyCandidate):
    pattern_attribute_names: tuple[str, ...]
    matcher_names: tuple[str, ...]
    group_index: int

@dataclass(frozen=True)
class _AccumulatorFoldStatements:
    assign: ast.stmt
    loop: ast.For
    returned: ast.Return


@dataclass(frozen=True)
class _AccumulatorFoldContext:
    statements: _AccumulatorFoldStatements
    accumulator_name: str
    accumulator_type_name: str
    step_call: ast.Call



@dataclass(frozen=True)
class _AccumulatorFoldMethod:
    method_name: str
    line: int
    source_parameter_name: str
    accumulator_type_name: str
    step_method_name: str
    result_method_name: str

    @property
    def shape_key(self) -> tuple[str, str]:
        return (self.accumulator_type_name, self.result_method_name)




_ParsedFamilyMethod = TypeVar("_ParsedFamilyMethod")
_ShapeKey = TypeVar("_ShapeKey", bound=Hashable)


@dataclass(frozen=True)
class ClassMethodGroupsShapeProjector(Generic[_ParsedFamilyMethod, _ShapeKey]):
    method_parser: Callable[[ast.FunctionDef], _ParsedFamilyMethod | None]
    shape_key: Callable[[_ParsedFamilyMethod], _ShapeKey]

    def project(
        self, module: ParsedModule
    ) -> tuple[tuple[ast.ClassDef, tuple[_ParsedFamilyMethod, ...]], ...]:
        groups: list[tuple[ast.ClassDef, tuple[_ParsedFamilyMethod, ...]]] = []
        for class_node in (
            node
            for node in _walk_nodes(module.module)
            if isinstance(node, ast.ClassDef)
        ):
            grouped: dict[_ShapeKey, list[_ParsedFamilyMethod]] = defaultdict(list)
            for statement in class_node.body:
                if not isinstance(statement, ast.FunctionDef):
                    continue
                method = self.method_parser(statement)
                if method is not None:
                    grouped[self.shape_key(method)].append(method)
            for methods in grouped.values():
                if len(methods) < 2:
                    continue
                groups.append(
                    (
                        class_node,
                        sorted_tuple(
                            methods, key=lambda item: (item.line, item.method_name)
                        ),
                    )
                )
        return tuple(groups)




def _accumulator_fold_method(
    method: ast.FunctionDef,
) -> _AccumulatorFoldMethod | None:
    body = _trim_docstring_body(method.body)
    fold_shape = _accumulator_fold_shape(body)
    if fold_shape is None:
        return None
    accumulator_name, accumulator_type_name, loop, step_call, result_call = fold_shape
    args = method.args.args
    offset = 1 if args and args[0].arg in {"self", "cls"} else 0
    if len(args) <= offset:
        return None
    source_parameter = args[offset].arg
    if not (isinstance(loop.iter, ast.Name) and loop.iter.id == source_parameter):
        return None
    return _AccumulatorFoldMethod(
        method_name=method.name,
        line=method.lineno,
        source_parameter_name=source_parameter,
        accumulator_type_name=accumulator_type_name,
        step_method_name=step_call.func.attr,
        result_method_name=result_call.func.attr,
    )


def _accumulator_fold_shape(
    body: list[ast.stmt],
) -> tuple[str, str, ast.For, ast.Call, ast.Call] | None:
    return (
        Maybe.of(tuple(body) if len(body) == 3 else None)
        .project(
            lambda statements: (
                _AccumulatorFoldStatements(
                    assign=statements[0],
                    loop=statements[1],
                    returned=statements[2],
                )
                if isinstance(statements[1], ast.For)
                and isinstance(statements[2], ast.Return)
                else None
            )
        )
        .combine(
            lambda statements: _accumulator_initializer(statements.assign),
            lambda statements, accumulator: (
                _AccumulatorFoldContext(
                    statements=statements,
                    accumulator_name=accumulator[0],
                    accumulator_type_name=accumulator[1],
                    step_call=_accumulator_step_call(statements.loop, accumulator[0]),
                )
                if _accumulator_step_call(statements.loop, accumulator[0]) is not None
                else None
            ),
        )
        .combine(
            lambda context: _accumulator_result_call(
                context.statements.returned,
                context.accumulator_name,
            ),
            lambda context, result_call: (
                context.accumulator_name,
                context.accumulator_type_name,
                context.statements.loop,
                context.step_call,
                result_call,
            ),
        )
        .unwrap_or_none()
    )


def _accumulator_initializer(statement: ast.stmt) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Assign):
        return None
    target = as_ast(single_assign_target(statement), ast.Name)
    call = as_ast(statement.value, ast.Call)
    if target is None or call is None or call.args or call.keywords:
        return None
    return target.id, ast.unparse(call.func)


def _accumulator_step_call(loop: ast.For, accumulator_name: str) -> ast.Call | None:
    target = as_ast(loop.target, ast.Name)
    expression = as_ast(single_item(loop.body), ast.Expr)
    call = as_ast(expression.value if expression is not None else None, ast.Call)
    arg = single_item(call.args) if call is not None else None
    if not (
        target is not None
        and call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == accumulator_name)
        and (not call.keywords)
        and isinstance(arg, ast.Name)
        and (arg.id == target.id)
    ):
        return None
    return call


def _accumulator_result_call(
    returned: ast.Return, accumulator_name: str
) -> ast.Call | None:
    call = as_ast(returned.value, ast.Call)
    if not (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == accumulator_name)
        and (not call.args)
        and (not call.keywords)
    ):
        return None
    return call


def _accumulator_fold_family_candidates(
    module: ParsedModule,
) -> tuple[AccumulatorFoldFamilyCandidate, ...]:
    candidates: list[AccumulatorFoldFamilyCandidate] = []
    projector = ClassMethodGroupsShapeProjector(
        _accumulator_fold_method, lambda method: method.shape_key
    )
    for class_node, ordered in projector.project(module):
        if len({method.step_method_name for method in ordered}) < 2:
            continue
        candidates.append(
            AccumulatorFoldFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                accumulator_type_name=ordered[0].accumulator_type_name,
                result_method_name=ordered[0].result_method_name,
                method_names=tuple((method.method_name for method in ordered)),
                line_numbers=tuple((method.line for method in ordered)),
                source_parameter_names=tuple(
                    (method.source_parameter_name for method in ordered)
                ),
                step_method_names=tuple(
                    (method.step_method_name for method in ordered)
                ),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.class_name),
    )


@dataclass(frozen=True)
class _RegexGroupExtractorMethod:
    method_name: str
    line: int
    pattern_attribute_name: str
    matcher_name: str
    group_index: int
    supported_matcher_names: ClassVar[frozenset[str]] = frozenset(
        {"search", "match", "fullmatch"}
    )

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef,
    ) -> "_RegexGroupExtractorMethod | None":
        statements = ast_sequence(
            _trim_docstring_body(method.body), ast.Assign, ast.Return
        )
        if statements is None:
            return None
        assign, returned = statements
        assignment = named_call_assignment(assign)
        if assignment is None:
            return None
        matcher = attribute_call_match(
            assignment.call,
            method_names=cls.supported_matcher_names,
            owner_type=ast.Attribute,
            owner_name="self",
            single_argument_required=True,
        )
        if matcher is None:
            return None
        conditional = as_ast(returned.value, ast.IfExp)
        if conditional is None or name_id(conditional.test) != assignment.target_name:
            return None
        none_orelse = as_ast(conditional.orelse, ast.Constant)
        group_call = as_ast(conditional.body, ast.Call)
        if none_orelse is None or none_orelse.value is not None or group_call is None:
            return None
        group = attribute_call_match(
            group_call,
            method_name="group",
            owner_type=ast.Name,
            owner_name=assignment.target_name,
            single_argument_required=True,
        )
        group_index = constant_value(group.single_argument) if group else None
        if not isinstance(group_index, int):
            return None
        return cls(
            method_name=method.name,
            line=method.lineno,
            pattern_attribute_name=matcher.owner.attr,
            matcher_name=matcher.attribute.attr,
            group_index=group_index,
        )


def _regex_group_extractor_family_candidates(
    module: ParsedModule,
) -> tuple[RegexGroupExtractorFamilyCandidate, ...]:
    candidates: list[RegexGroupExtractorFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        methods = tuple(
            (
                extractor
                for statement in class_node.body
                if isinstance(statement, ast.FunctionDef)
                for extractor in (_RegexGroupExtractorMethod.from_method(statement),)
                if extractor is not None
            )
        )
        grouped: dict[int, list[_RegexGroupExtractorMethod]] = defaultdict(list)
        for method in methods:
            grouped[method.group_index].append(method)
        for group_index, grouped_methods in grouped.items():
            if len(grouped_methods) < 2:
                continue
            ordered = sorted_tuple(
                grouped_methods, key=lambda item: (item.line, item.method_name)
            )
            candidates.append(
                RegexGroupExtractorFamilyCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    method_names=tuple((method.method_name for method in ordered)),
                    line_numbers=tuple((method.line for method in ordered)),
                    pattern_attribute_names=tuple(
                        (method.pattern_attribute_name for method in ordered)
                    ),
                    matcher_names=tuple((method.matcher_name for method in ordered)),
                    group_index=group_index,
                )
            )
    return tuple(candidates)




__all__ = tuple(name for name in globals() if not name.startswith("__"))
