"""Nominal traversal semantics for detector candidate collection."""

from __future__ import annotations

import ast
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from metaclass_registry import AutoRegisterMeta

from .semantic_match import Maybe, as_ast, name_id


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


@dataclass(frozen=True)
class NamedFunctionLoopComponents:
    module_parameter_name: str
    qualname_parameter_name: str
    function_parameter_name: str
    loop: ast.For


@dataclass(frozen=True)
class NamedFunctionLoopTargetNames:
    qualname_parameter_name: str
    function_parameter_name: str


def _named_function_loop_target_names(
    target: ast.expr,
) -> NamedFunctionLoopTargetNames | None:
    return (
        Maybe.of(as_ast(target, ast.Tuple))
        .filter(lambda tuple_target: len(tuple_target.elts) == 2)
        .combine(
            lambda tuple_target: name_id(tuple_target.elts[0]),
            lambda tuple_target, qualname_name: (
                tuple_target,
                qualname_name,
            ),
        )
        .combine(
            lambda tuple_and_qualname: name_id(tuple_and_qualname[0].elts[1]),
            lambda tuple_and_qualname, function_name: NamedFunctionLoopTargetNames(
                qualname_parameter_name=tuple_and_qualname[1],
                function_parameter_name=function_name,
            ),
        )
        .unwrap_or_none()
    )


TraversalSubjectT = TypeVar("TraversalSubjectT")
TraversalMatchT = TypeVar("TraversalMatchT")


class CandidateCollectorTraversal(
    Generic[TraversalSubjectT, TraversalMatchT],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered traversal declaration that owns import-time family membership."""

    __registry__: ClassVar[dict[str, type["CandidateCollectorTraversal"]]] = {}
    __registry_key__ = "call_name"
    __skip_if_no_key__ = True

    call_name: ClassVar[str]

    @classmethod
    def first_match(cls, subject: TraversalSubjectT) -> TraversalMatchT | None:
        for traversal_type in cls.__registry__.values():
            if issubclass(traversal_type, cls):
                match = traversal_type.match(subject)
                if match is not None:
                    return match
        return None

    @classmethod
    def match(cls, subject: TraversalSubjectT) -> TraversalMatchT | None:
        raise NotImplementedError


class NamedFunctionLoopTraversal(
    CandidateCollectorTraversal[ast.stmt, NamedFunctionLoopComponents],
    ABC,
):
    """Nominal declaration for named-function traversal loop syntax."""

    @classmethod
    def match(
        cls, statement: ast.stmt
    ) -> NamedFunctionLoopComponents | None:
        return (
            Maybe.of(as_ast(statement, ast.For))
            .combine(
                lambda loop: _named_function_loop_target_names(loop.target),
                lambda loop, target_names: (loop, target_names),
            )
            .combine(
                lambda loop_and_names: as_ast(loop_and_names[0].iter, ast.Call),
                lambda loop_and_names, call: (
                    loop_and_names[0],
                    loop_and_names[1],
                    call,
                ),
            )
            .filter(
                lambda loop_names_call: (
                    _call_name(loop_names_call[2].func) == cls.call_name
                    and len(loop_names_call[2].args) == 1
                    and not loop_names_call[2].keywords
                )
            )
            .combine(
                lambda loop_names_call: name_id(loop_names_call[2].args[0]),
                lambda loop_names_call, module_name: NamedFunctionLoopComponents(
                    module_parameter_name=module_name,
                    qualname_parameter_name=(
                        loop_names_call[1].qualname_parameter_name
                    ),
                    function_parameter_name=(
                        loop_names_call[1].function_parameter_name
                    ),
                    loop=loop_names_call[0],
                ),
            )
            .unwrap_or_none()
        )


class IterNamedFunctionsTraversal(NamedFunctionLoopTraversal):
    call_name = "_iter_named_functions"


@dataclass(frozen=True)
class AstStreamTraversalMatch:
    root_expression: ast.expr
    traversal_expression: ast.expr
    traversal_type: type["AstStreamTraversal"]

    @property
    def call_name(self) -> str:
        return self.traversal_type.call_name


class AstStreamTraversal(
    CandidateCollectorTraversal[ast.Call, AstStreamTraversalMatch],
    ABC,
):
    """Nominal declaration for supported AST stream traversal syntax."""

    emits_default_traversal: ClassVar[bool] = False

    @classmethod
    def match(cls, call: ast.Call) -> AstStreamTraversalMatch | None:
        return (
            Maybe.of(call)
            .filter(
                lambda candidate_call: (
                    _call_name(candidate_call.func) == cls.call_name
                    and len(candidate_call.args) == 1
                    and not candidate_call.keywords
                )
            )
            .map(
                lambda candidate_call: AstStreamTraversalMatch(
                    root_expression=candidate_call.args[0],
                    traversal_expression=candidate_call.func,
                    traversal_type=cls,
                )
            )
            .unwrap_or_none()
        )


class DefaultWalkNodesTraversal(AstStreamTraversal):
    call_name = "_walk_nodes"
    emits_default_traversal = True


class AstWalkTraversal(AstStreamTraversal):
    call_name = "ast.walk"


@dataclass(frozen=True)
class AstStreamLoopComponents:
    node_parameter_name: str
    traversal_match: AstStreamTraversalMatch
    loop: ast.For


def named_function_loop_components(
    statement: ast.stmt,
) -> NamedFunctionLoopComponents | None:
    return NamedFunctionLoopTraversal.first_match(statement)


def ast_stream_loop_components(
    statement: ast.stmt,
) -> AstStreamLoopComponents | None:
    return (
        Maybe.of(as_ast(statement, ast.For))
        .combine(
            lambda loop: name_id(loop.target),
            lambda loop, node_name: (loop, node_name),
        )
        .combine(
            lambda loop_and_name: as_ast(loop_and_name[0].iter, ast.Call),
            lambda loop_and_name, iter_call: (
                loop_and_name[0],
                loop_and_name[1],
                iter_call,
            ),
        )
        .combine(
            lambda loop_name_call: AstStreamTraversal.first_match(loop_name_call[2]),
            lambda loop_name_call, traversal_match: AstStreamLoopComponents(
                node_parameter_name=loop_name_call[1],
                traversal_match=traversal_match,
                loop=loop_name_call[0],
            ),
        )
        .unwrap_or_none()
    )
