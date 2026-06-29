"""Step/regex/extractor structural detector cohort."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from ..ast_tools import ParsedModule, _walk_nodes
from ..collection_algebra import sorted_tuple
from ..semantic_match import (
    AstTypedEffectStep,
    GuardedEffectStep,
    Maybe,
    RegisteredEffectStep,
    as_ast,
    ast_sequence,
    attribute_call_match,
    constant_value,
    name_id,
    named_call_assignment,
    registered_effect_steps,
    single_assign_target,
    single_item,
)
from ._base import (
    ClassMethodFamilyCandidate,
    KeywordMethodFamilyCandidate,
    _dataclass_field_names,
    _is_classmethod,
)
from ._helpers import HELPER_SUPPORT_PROJECTION_AUTHORITY
from ._substrate_support import (
    _ast_terminal_name,
    _is_dataclass_class,
    _trim_docstring_body,
)


@dataclass(frozen=True)
class ConstructorVariantFamilyCandidate(ClassMethodFamilyCandidate):
    callee_name: str
    coordinate_count: int
    varying_coordinate_names: tuple[str, ...]


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
class _ConstructorCallContext:
    call: ast.Call
    callee_name: str


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
class SparseConstructorVariantFamilyCandidate(KeywordMethodFamilyCandidate):
    pass


@dataclass(frozen=True)
class _ConstructorVariantMethod:
    method_name: str
    line: int
    callee_name: str
    positional_count: int
    keyword_names: tuple[str, ...]
    coordinate_fingerprints: tuple[str, ...]

    @property
    def shape_key(self) -> tuple[object, ...]:
        return (self.callee_name, self.positional_count, self.keyword_names)


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


def _simple_classmethod_return_call(
    node: ast.FunctionDef,
) -> ast.Call | None:
    if not _is_classmethod(node):
        return None
    body = _trim_docstring_body(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    returned = body[0].value
    if not isinstance(returned, ast.Call):
        return None
    return returned


def _classmethod_constructor_callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id == "cls":
        return "cls"
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and (call.func.value.id == "cls")
    ):
        return f"cls.{call.func.attr}"
    return None


def _call_coordinate_fingerprints(call: ast.Call) -> tuple[str, ...] | None:
    if any((keyword.arg is None for keyword in call.keywords)):
        return None
    positional = tuple(ast.dump(arg, annotate_fields=False) for arg in call.args)
    keywords = tuple(
        (
            ast.dump(keyword.value, annotate_fields=False)
            for keyword in sorted(call.keywords, key=lambda item: item.arg or "")
        )
    )
    return positional + keywords


def _constructor_variant_method(
    method: ast.FunctionDef,
) -> _ConstructorVariantMethod | None:
    return (
        Maybe.of(_simple_classmethod_return_call(method))
        .combine(
            _classmethod_constructor_callee_name,
            lambda call, callee_name: _ConstructorCallContext(
                call=call,
                callee_name=callee_name,
            ),
        )
        .combine(
            lambda context: _call_coordinate_fingerprints(context.call),
            lambda context, coordinate_fingerprints: _ConstructorVariantMethod(
                method_name=method.name,
                line=method.lineno,
                callee_name=context.callee_name,
                positional_count=len(context.call.args),
                keyword_names=tuple(
                    (
                        keyword.arg or ""
                        for keyword in sorted(
                            context.call.keywords, key=lambda item: item.arg or ""
                        )
                    )
                ),
                coordinate_fingerprints=coordinate_fingerprints,
            ),
        )
        .unwrap_or_none()
    )


def _varying_coordinate_names(
    methods: tuple[_ConstructorVariantMethod, ...],
) -> tuple[str, ...]:
    coordinate_count = len(methods[0].coordinate_fingerprints)
    varying: list[str] = []
    positional_count = methods[0].positional_count
    keyword_names = methods[0].keyword_names
    for index in range(coordinate_count):
        values = {method.coordinate_fingerprints[index] for method in methods}
        if len(values) < 2:
            continue
        if index < positional_count:
            varying.append(f"arg{index}")
        else:
            varying.append(keyword_names[index - positional_count])
    return tuple(varying)


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


def _constructor_variant_family_candidates(
    module: ParsedModule,
) -> tuple[ConstructorVariantFamilyCandidate, ...]:
    candidates: list[ConstructorVariantFamilyCandidate] = []
    projector = ClassMethodGroupsShapeProjector(
        _constructor_variant_method, lambda method: method.shape_key
    )
    for class_node, ordered in projector.project(module):
        varying_coordinates = _varying_coordinate_names(ordered)
        if not varying_coordinates:
            continue
        candidates.append(
            ConstructorVariantFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                callee_name=ordered[0].callee_name,
                method_names=tuple((method.method_name for method in ordered)),
                line_numbers=tuple((method.line for method in ordered)),
                coordinate_count=len(ordered[0].coordinate_fingerprints),
                varying_coordinate_names=varying_coordinates,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line_numbers, item.class_name),
    )


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


_REGEX_MATCHER_NAMES = frozenset({"search", "match", "fullmatch"})


@dataclass(frozen=True)
class _RegexExtractorBody:
    method: ast.FunctionDef
    assign: ast.Assign
    returned: ast.Return


@dataclass(frozen=True)
class _RegexExtractorMethodContext:
    method: ast.FunctionDef
    match_name: str


@dataclass(frozen=True)
class _RegexExtractorReturnedContext(_RegexExtractorMethodContext):
    returned: ast.Return


@dataclass(frozen=True)
class _RegexExtractorAssignment(_RegexExtractorReturnedContext):
    call: ast.Call


@dataclass(frozen=True)
class _RegexExtractorMatcherCall(_RegexExtractorReturnedContext):
    pattern_attribute_name: str
    matcher_name: str


@dataclass(frozen=True)
class _RegexExtractorConditionalReturn(_RegexExtractorMatcherCall):
    group_call: ast.Call


class _RegexGroupExtractorStep(RegisteredEffectStep):
    pass


class _RegexExtractorBodyStep(
    _RegexGroupExtractorStep,
    AstTypedEffectStep[ast.FunctionDef, _RegexExtractorBody],
):
    step_id = "regex_extractor_body"
    registration_order = 10
    node_type = ast.FunctionDef

    def project_ast(self, value: ast.FunctionDef) -> _RegexExtractorBody | None:
        statements = ast_sequence(
            _trim_docstring_body(value.body), ast.Assign, ast.Return
        )
        if statements is None:
            return None
        assign, returned = statements
        return _RegexExtractorBody(value, assign, returned)


class _RegexExtractorAssignmentStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorBody, _RegexExtractorAssignment],
):
    step_id = "regex_extractor_assignment"
    registration_order = 20

    def project(self, value: _RegexExtractorBody) -> _RegexExtractorAssignment | None:
        assignment = named_call_assignment(value.assign)
        if assignment is None:
            return None
        return _RegexExtractorAssignment(
            method=value.method,
            match_name=assignment.target_name,
            returned=value.returned,
            call=assignment.call,
        )


class _RegexExtractorMatcherCallStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorAssignment, _RegexExtractorMatcherCall],
):
    step_id = "regex_extractor_matcher_call"
    registration_order = 30

    def project(
        self, value: _RegexExtractorAssignment
    ) -> _RegexExtractorMatcherCall | None:
        match = attribute_call_match(
            value.call,
            method_names=_REGEX_MATCHER_NAMES,
            owner_type=ast.Attribute,
            owner_name="self",
            single_argument_required=True,
        )
        if match is None:
            return None
        return _RegexExtractorMatcherCall(
            method=value.method,
            match_name=value.match_name,
            returned=value.returned,
            pattern_attribute_name=match.owner.attr,
            matcher_name=match.attribute.attr,
        )


def _regex_conditional_group_call(
    value: _RegexExtractorMatcherCall,
) -> ast.Call | None:
    ifexp = as_ast(value.returned.value, ast.IfExp)
    none_orelse = as_ast(ifexp.orelse if ifexp else None, ast.Constant)
    group_call = as_ast(ifexp.body if ifexp else None, ast.Call)
    if (
        ifexp is None
        or name_id(ifexp.test) != value.match_name
        or none_orelse is None
        or (none_orelse.value is not None)
        or (group_call is None)
    ):
        return None
    return group_call


class _RegexExtractorConditionalReturnStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorMatcherCall, _RegexExtractorConditionalReturn],
):
    step_id = "regex_extractor_conditional_return"
    registration_order = 40

    def project(
        self, value: _RegexExtractorMatcherCall
    ) -> _RegexExtractorConditionalReturn | None:
        group_call = _regex_conditional_group_call(value)
        if group_call is None:
            return None
        return _RegexExtractorConditionalReturn(
            method=value.method,
            match_name=value.match_name,
            returned=value.returned,
            pattern_attribute_name=value.pattern_attribute_name,
            matcher_name=value.matcher_name,
            group_call=group_call,
        )


class _RegexExtractorGroupCallStep(
    _RegexGroupExtractorStep,
    GuardedEffectStep[_RegexExtractorConditionalReturn, _RegexGroupExtractorMethod],
):
    step_id = "regex_extractor_group_call"
    registration_order = 50

    def project(
        self, value: _RegexExtractorConditionalReturn
    ) -> _RegexGroupExtractorMethod | None:
        match = attribute_call_match(
            value.group_call,
            method_name="group",
            owner_type=ast.Name,
            owner_name=value.match_name,
            single_argument_required=True,
        )
        group_index = constant_value(match.single_argument) if match else None
        if not isinstance(group_index, int):
            return None
        return _RegexGroupExtractorMethod(
            method_name=value.method.name,
            line=value.method.lineno,
            pattern_attribute_name=value.pattern_attribute_name,
            matcher_name=value.matcher_name,
            group_index=group_index,
        )


def _regex_group_extractor_method(
    method: ast.FunctionDef,
) -> _RegexGroupExtractorMethod | None:
    return cast(
        _RegexGroupExtractorMethod | None,
        Maybe.of(method)
        .bind_all(registered_effect_steps(_RegexGroupExtractorStep))
        .unwrap_or_none(),
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
                for extractor in (_regex_group_extractor_method(statement),)
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


def _class_has_constructor_variant_mixin(node: ast.ClassDef) -> bool:
    return any(
        (_ast_terminal_name(base) == "ConstructorVariantMixin" for base in node.bases)
    )


def _sparse_constructor_variant_family_candidates(
    module: ParsedModule,
) -> tuple[SparseConstructorVariantFamilyCandidate, ...]:
    candidates: list[SparseConstructorVariantFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        if not _is_dataclass_class(class_node) or _class_has_constructor_variant_mixin(
            class_node
        ):
            continue
        field_names = set(_dataclass_field_names(class_node))
        methods: list[tuple[ast.FunctionDef, tuple[str, ...]]] = []
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef) or not _is_classmethod(
                statement
            ):
                continue
            call = HELPER_SUPPORT_PROJECTION_AUTHORITY.constructor_return_call(
                statement
            )
            if call is None or call.args:
                continue
            keyword_names = tuple(
                (
                    keyword.arg or ""
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
            )
            if not keyword_names or not set(keyword_names) <= field_names:
                continue
            methods.append((statement, keyword_names))
        if len(methods) < 2:
            continue
        union_keywords = sorted_tuple({name for _, names in methods for name in names})
        if not union_keywords:
            continue
        ordered = sorted_tuple(methods, key=lambda item: (item[0].lineno, item[0].name))
        candidates.append(
            SparseConstructorVariantFamilyCandidate(
                file_path=str(module.path),
                class_name=class_node.name,
                method_names=tuple((method.name for method, _ in ordered)),
                line_numbers=tuple((method.lineno for method, _ in ordered)),
                keyword_names=union_keywords,
            )
        )
    return tuple(candidates)


__all__ = tuple(name for name in globals() if not name.startswith("__"))
