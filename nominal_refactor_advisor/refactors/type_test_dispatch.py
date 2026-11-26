"""Push a type test on a scanned class family onto a polymorphic method of that family.

When a consumer branches on `isinstance(x, Sub)`, `issubclass(cls, Sub)`,
`type(x) is Sub`, `cls is Sub`, or a class pattern `case Sub():` for classes the
scan declares, and every runtime value of `x` is proved to belong to a closed
scanned family rooted at `Base`, moving each arm's body onto its tested class as
an override of one `Base` method deletes the type test at the consumer.

Whole-function dispatches reuse the enclosing function's name automatically when
it is unbound on the family; embedded dispatches inside larger functions ask for
a method name and one-line docstring via `--codemod-owner-picks` (`OWNER_PICK`).
"""

from __future__ import annotations

import ast
import builtins
import copy
import keyword as keyword_module
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

from nominal_refactor_advisor.ast_field_decoders import (
    declared_annotations,
    optional_values,
)
from nominal_refactor_advisor.ast_tools import AstParentIndex, ParsedModule
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
    PayloadRecordValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import InferredReceiverType, ReceiverSpan
from nominal_refactor_advisor.refactor_concepts import TypeTestDispatchConcept
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod
from nominal_refactor_advisor.refactors.class_overlap import obsolete_import_edits
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class DispatchKind(ABC):
    """Whether the tested value is an instance (`self`) or a class (`cls`)."""

    receiver: ClassVar[str]
    decorators: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def unwrap_annotation(self, annotation: ast.expr) -> ast.expr | None:
        raise NotImplementedError

    @abstractmethod
    def require_inferred_compatible(
        self, answer: InferredReceiverType, spelled_subject: str
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class InstanceDispatchKind(DispatchKind):
    """Instance dispatch: plain method with `self`."""

    receiver: ClassVar[str] = "self"

    def unwrap_annotation(self, annotation: ast.expr) -> ast.expr | None:
        match annotation:
            case ast.Subscript(value=ast.Name(id="type")):
                return None
            case ast.Subscript(value=inner):
                return inner
        return annotation

    def require_inferred_compatible(
        self, answer: InferredReceiverType, spelled_subject: str
    ) -> None:
        if re.search(r"\bNone\b", answer.spelled) or not answer.instances:
            raise NeedsInput(
                f"the type checker infers `{answer.spelled}` for `{spelled_subject}`, "
                "whose values are not all instances with methods: decode it into the "
                "family where it enters",
                NeededInput.SOURCE_CHANGE,
            )


@dataclass(frozen=True)
class ClassDispatchKind(DispatchKind):
    """Class dispatch: `@classmethod` with `cls`."""

    receiver: ClassVar[str] = "cls"
    decorators: ClassVar[tuple[str, ...]] = ("@classmethod",)

    def unwrap_annotation(self, annotation: ast.expr) -> ast.expr | None:
        match annotation:
            case ast.Subscript(value=ast.Name(id="type"), slice=inner):
                return inner
        return None

    def require_inferred_compatible(
        self, answer: InferredReceiverType, spelled_subject: str
    ) -> None:
        if re.search(r"\bNone\b", answer.spelled) or answer.instances:
            raise NeedsInput(
                f"the type checker infers `{answer.spelled}` for `{spelled_subject}`, "
                "which is not a class type (`type[Base]`)",
                NeededInput.SOURCE_CHANGE,
            )


@dataclass(frozen=True)
class ParsedTypeTest:
    """One parsed type test expression: `isinstance`, `issubclass`, `type(x) is`, `x is`."""

    subject: ast.expr
    tested_exprs: tuple[ast.expr, ...]
    kind: DispatchKind
    negated: bool
    exact: bool
    test_node: ast.expr


def _is_pure_subject(expr: ast.expr) -> bool:
    """Whether `expr` is a side-effect-free `Name` or attribute chain (`a.b.c`)."""

    current = expr
    while isinstance(current, ast.Attribute):
        current = current.value
    return isinstance(current, ast.Name)


def _unpack_type_targets(expr: ast.expr) -> tuple[ast.expr, ...] | None:
    match expr:
        case ast.Tuple(elts=elts) if elts:
            return tuple(elts)
        case ast.BinOp(op=ast.BitOr(), left=left, right=right):
            left_items = _unpack_type_targets(left)
            right_items = _unpack_type_targets(right)
            if left_items is not None and right_items is not None:
                return (*left_items, *right_items)
            return None
        case ast.Name() | ast.Attribute():
            return (expr,)
    return None


def _parse_type_test(expr: ast.expr) -> ParsedTypeTest | None:
    negated = False
    inner = expr
    match expr:
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            negated = True
            inner = operand
    match inner:
        case ast.Call(
            func=ast.Name(id="isinstance" | "issubclass" as func_name),
            args=[subject, target],
            keywords=[],
        ) if _is_pure_subject(subject):
            targets = _unpack_type_targets(target)
            if targets is None:
                return None
            kind: DispatchKind = (
                ClassDispatchKind()
                if func_name == "issubclass"
                else InstanceDispatchKind()
            )
            return ParsedTypeTest(subject, targets, kind, negated, False, expr)
        case ast.Compare(
            left=ast.Call(func=ast.Name(id="type"), args=[subject], keywords=[]),
            ops=[ast.Is() | ast.IsNot() as op],
            comparators=[ast.Name() | ast.Attribute() as target],
        ) if _is_pure_subject(subject):
            is_negated = negated ^ isinstance(op, ast.IsNot)
            return ParsedTypeTest(
                subject, (target,), InstanceDispatchKind(), is_negated, True, expr
            )
        case ast.Compare(
            left=subject,
            ops=[ast.Is() | ast.IsNot() as op],
            comparators=[ast.Name() | ast.Attribute() as target],
        ) if _is_pure_subject(subject):
            is_negated = negated ^ isinstance(op, ast.IsNot)
            return ParsedTypeTest(
                subject, (target,), ClassDispatchKind(), is_negated, True, expr
            )
    return None


@dataclass(frozen=True)
class TypeTestArm:
    """One arm of a type-test dispatch targeting one or more classes."""

    tested_exprs: tuple[ast.expr, ...]
    body: tuple[ast.stmt, ...]
    exact: bool
    alias: str | None = None


@dataclass(frozen=True)
class DispatchSite(ABC):
    """One consumer-side type-test site normalized into arms plus a base default."""

    subject: ast.expr
    kind: DispatchKind
    arms: tuple[TypeTestArm, ...]
    default_body: tuple[ast.stmt, ...]
    is_whole_function: bool
    test_nodes: tuple[ast.AST, ...]

    @abstractmethod
    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        raise NotImplementedError

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    @property
    def checks_local_store_escapes(self) -> bool:
        return False

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return ()

    def tested_expr_nodes(self) -> tuple[ast.AST, ...]:
        return tuple(
            expr for arm in self.arms for expr in arm.tested_exprs
        )


@dataclass(frozen=True)
class TernaryDispatchSite(DispatchSite):
    """`a if isinstance(x, Sub) else b` replaced inline by `x.method(...)`."""

    node: ast.IfExp

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)


@dataclass(frozen=True)
class BooleanPredicateDispatchSite(DispatchSite):
    """`return isinstance(x, Sub)` replaced by `return x.method()`."""

    predicate_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.predicate_expr)


@dataclass(frozen=True)
class BoolOpSuffixDispatchSite(BooleanPredicateDispatchSite):
    """Suffix `isinstance(x, Sub) and/or REST` inside a longer `BoolOp`."""

    end_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.predicate_expr)
        _, end = geometry.required_node_offsets(self.end_expr)
        return start, end


@dataclass(frozen=True)
class StatementDispatchSite(DispatchSite):
    """`if/elif/else`, early-return `if`, or `match` replaced by a statement call."""

    replaced_nodes: tuple[ast.stmt, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    @property
    def checks_local_store_escapes(self) -> bool:
        return not self.is_whole_function

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes


@dataclass(frozen=True)
class ReturningStatementDispatchSite(StatementDispatchSite):
    """Returning `if/elif/else`, early-return `if`, or `match` replaced by `return x.method(...)`."""

    @property
    def checks_local_store_escapes(self) -> bool:
        return False

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"return {call_expr}"


@dataclass(frozen=True)
class PrefixGuardedStatementDispatchSite(StatementDispatchSite):
    """Void `if PREFIX and isinstance(x, Sub): body` replaced by `if PREFIX:\\n    x.method(...)`."""

    guard_expr: ast.expr

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"if {ast.unparse(self.guard_expr)}:\n{line_indent}    {call_expr}"


def _render_indented_statements(
    statements: Sequence[ast.stmt], body_indent: str
) -> list[str]:
    return [
        f"{body_indent}{raw}" if raw else ""
        for stmt in statements
        for raw in ast.unparse(ast.fix_missing_locations(stmt)).splitlines()
    ]


@dataclass(frozen=True)
class PrefixGuardedOrDispatchSite(StatementDispatchSite):
    """`if PREFIX or not isinstance(x, Sub): early_exit` + tail/else replaced by
    `if PREFIX:\\n    <early_exit>\\n[return ]x.method(...)`."""

    guard_expr: ast.expr
    guard_body: tuple[ast.stmt, ...]
    returns_value: bool

    @property
    def checks_local_store_escapes(self) -> bool:
        return not self.returns_value

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        body_lines = _render_indented_statements(self.guard_body, f"{line_indent}    ")
        call_line = (
            f"{line_indent}return {call_expr}"
            if self.returns_value
            else f"{line_indent}{call_expr}"
        )
        return "\n".join(
            (f"if {ast.unparse(self.guard_expr)}:", *body_lines, call_line)
        )


@dataclass(frozen=True)
class OrElseGuardedStatementDispatchSite(StatementDispatchSite):
    """Void `if PREFIX or TEST: body` replaced by `if PREFIX:\\n    body\\nelse:\\n    x.method(...)`."""

    guard_expr: ast.expr
    guard_body: tuple[ast.stmt, ...]

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        body_indent = f"{line_indent}    "
        body_lines = _render_indented_statements(self.guard_body, body_indent)
        return "\n".join(
            (
                f"if {ast.unparse(self.guard_expr)}:",
                *body_lines,
                f"{line_indent}else:",
                f"{body_indent}{call_expr}",
            )
        )


def _is_docstring_stmt(statement: ast.stmt) -> bool:
    match statement:
        case ast.Expr(value=ast.Constant(value=str())):
            return True
    return False


def _body_without_docstring(function: FunctionNode) -> list[ast.stmt]:
    if function.body and _is_docstring_stmt(function.body[0]):
        return list(function.body[1:])
    return list(function.body)


def _always_terminates(statements: Sequence[ast.stmt]) -> bool:
    """Whether every control-flow path through `statements` returns or raises."""

    if not statements:
        return False
    last = statements[-1]
    match last:
        case ast.Return() | ast.Raise():
            return True
        case ast.If(body=body, orelse=orelse) if orelse:
            return _always_terminates(body) and _always_terminates(orelse)
        case ast.Try(body=body, handlers=handlers, orelse=[], finalbody=[]) if handlers:
            return _always_terminates(body) and all(
                _always_terminates(handler.body) for handler in handlers
            )
    return False


def _has_any_return(statements: Sequence[ast.stmt]) -> bool:
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Return):
                return True
    return False


def _has_value_return(statements: Sequence[ast.stmt]) -> bool:
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Return) and optional_values(node):
                return True
    return False


def _always_void_terminates(statements: Sequence[ast.stmt]) -> bool:
    """Whether every path exits via bare `return` or `raise`, never returning a value."""

    return _always_terminates(statements) and not _has_value_return(statements)


def _strip_trailing_bare_return(statements: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    if (
        statements
        and isinstance(statements[-1], ast.Return)
        and not optional_values(statements[-1])
    ):
        remaining = statements[:-1]
        return tuple(remaining) if remaining else (ast.Pass(),)
    return tuple(statements)


def _has_loop_escape(statements: Sequence[ast.stmt]) -> bool:
    """Whether `statements` contains `return`, `break`, or `continue` escaping an enclosing loop."""

    stack: list[ast.AST] = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Return, ast.Break, ast.Continue)):
            return True
        if isinstance(
            node,
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _has_non_continue_loop_escape(statements: Sequence[ast.stmt]) -> bool:
    """Whether `statements` contains a `break` escaping the current loop or any `return`."""

    if _has_any_return(statements):
        return True
    stack: list[ast.AST] = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(
            node,
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


class _LoopContinueToReturnRewriter(ast.NodeTransformer):
    """Replace `continue` targeting the current loop with bare `return`."""

    def visit_Continue(self, node: ast.Continue) -> ast.AST:
        return ast.copy_location(ast.Return(value=None), node)

    def visit_For(self, node: ast.For) -> ast.AST:
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


def _rewrite_loop_continues_to_returns(
    statements: Sequence[ast.stmt],
) -> tuple[ast.stmt, ...]:
    rewriter = _LoopContinueToReturnRewriter()
    return tuple(
        cast(ast.stmt, rewriter.visit(copy.deepcopy(stmt)))
        for stmt in statements
    )


def _names_stored_in(nodes: Sequence[ast.AST]) -> frozenset[str]:
    stored: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            match node:
                case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                    stored.add(name)
                case ast.ExceptHandler(name=str() as name):
                    stored.add(name)
    return frozenset(stored)


def _names_referenced_outside(
    function: FunctionNode, excluded_nodes: Sequence[ast.AST]
) -> frozenset[str]:
    excluded_ids = {id(node) for node in excluded_nodes}
    referenced: set[str] = set()

    def walk(node: ast.AST) -> None:
        if id(node) in excluded_ids:
            return
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        for child in ast.iter_child_nodes(node):
            walk(child)

    for stmt in function.body:
        walk(stmt)
    return frozenset(referenced)


def _ends_with_continue_only(statements: Sequence[ast.stmt]) -> bool:
    """Whether `statements` ends with `continue` and has no other loop-escaping jumps."""

    return (
        bool(statements)
        and isinstance(statements[-1], ast.Continue)
        and not _has_loop_escape(statements[:-1])
    )


def _strip_trailing_continue(statements: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    remaining = statements[:-1]
    return tuple(remaining) if remaining else (ast.Pass(),)


def _statement_list_children(
    block: list[ast.stmt],
) -> list[tuple[list[ast.stmt], int, ast.stmt, bool]]:
    """Every `(containing_list, index, stmt, in_loop_body)` in `block` (excluding nested defs/classes)."""

    found: list[tuple[list[ast.stmt], int, ast.stmt, bool]] = []
    stack: list[tuple[list[ast.stmt], bool]] = [(block, False)]
    while stack:
        current, in_loop_body = stack.pop()
        for idx, stmt in enumerate(current):
            found.append((current, idx, stmt, in_loop_body))
            match stmt:
                case ast.If(body=b, orelse=o):
                    stack.append((o, False))
                    stack.append((b, False))
                case ast.For(body=b, orelse=o) | ast.AsyncFor(body=b, orelse=o) | ast.While(body=b, orelse=o):
                    stack.append((o, False))
                    stack.append((b, True))
                case ast.With(body=b) | ast.AsyncWith(body=b):
                    stack.append((b, False))
                case ast.Try(body=b, handlers=hs, orelse=o, finalbody=f):
                    stack.append((f, False))
                    stack.append((o, False))
                    for h in hs:
                        stack.append((h.body, False))
                    stack.append((b, False))
                case ast.Match(cases=cases):
                    for case in cases:
                        stack.append((case.body, False))
    return found


def _extract_compound_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    test_line: int,
    in_loop_body: bool = False,
) -> DispatchSite | None:
    """Extract `if [PREFIX and] isinstance(x, Sub) [and REST]:` or `if not isinstance(x, Sub) or REST:`."""

    if not isinstance(stmt.test, ast.BoolOp) or len(stmt.test.values) < 2:
        return None
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        return None
    first_test = _parse_type_test(stmt.test.values[0])
    if first_test is None:
        if isinstance(stmt.test.op, ast.Or):
            match_index: int | None = None
            matched_test: ParsedTypeTest | None = None
            for idx, value in enumerate(stmt.test.values[1:], start=1):
                parsed = _parse_type_test(value)
                if parsed is not None and parsed.negated and value.lineno == test_line:
                    match_index = idx
                    matched_test = parsed
                    break
            if match_index is None or matched_test is None:
                return None
            prefix_values = stmt.test.values[:match_index]
            guard_expr: ast.expr = (
                prefix_values[0]
                if len(prefix_values) == 1
                else ast.BoolOp(op=ast.Or(), values=prefix_values)
            )
            suffix_values = stmt.test.values[match_index + 1 :]
            suffix_expr: ast.expr | None = (
                None
                if not suffix_values
                else (
                    suffix_values[0]
                    if len(suffix_values) == 1
                    else ast.BoolOp(op=ast.Or(), values=suffix_values)
                )
            )
            if stmt.orelse:
                replaced = (stmt,)
                if _has_any_return((stmt,)):
                    if not (
                        _always_terminates(stmt.body) and _always_terminates(stmt.orelse)
                    ):
                        return None
                    arm_body: tuple[ast.stmt, ...] = (
                        (
                            ast.If(
                                test=suffix_expr,
                                body=list(stmt.body),
                                orelse=list(stmt.orelse),
                            ),
                        )
                        if suffix_expr is not None
                        else tuple(stmt.orelse)
                    )
                    return PrefixGuardedOrDispatchSite(
                        subject=matched_test.subject,
                        kind=matched_test.kind,
                        arms=(
                            TypeTestArm(
                                matched_test.tested_exprs, arm_body, matched_test.exact
                            ),
                        ),
                        default_body=tuple(stmt.body),
                        is_whole_function=False,
                        test_nodes=(matched_test.test_node,),
                        replaced_nodes=replaced,
                        guard_expr=guard_expr,
                        guard_body=tuple(stmt.body),
                        returns_value=True,
                    )
                return None
            tail = container[index + 1 :]
            if tail and _always_terminates(stmt.body) and _always_terminates(tail):
                replaced = tuple(container[index:])
                arm_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=tuple(stmt.body),
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=replaced,
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=_has_any_return(replaced),
                )
            if (
                container is whole_fn_body
                and tail
                and _always_void_terminates(stmt.body)
                and not _has_value_return(tail)
            ):
                stripped = _strip_trailing_bare_return(stmt.body)
                arm_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=stripped,
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=tuple(container[index:]),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=False,
                )
            if (
                in_loop_body
                and tail
                and _ends_with_continue_only(stmt.body)
                and not _has_non_continue_loop_escape(tail)
            ):
                stripped = _strip_trailing_continue(stmt.body)
                rewritten_tail = _rewrite_loop_continues_to_returns(tail)
                arm_body = (
                    (
                        ast.If(
                            test=suffix_expr,
                            body=[*stmt.body[:-1], ast.Return(value=None)],
                            orelse=[],
                        ),
                        *rewritten_tail,
                    )
                    if suffix_expr is not None
                    else rewritten_tail
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=stripped,
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=tuple(container[index:]),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=False,
                )
            if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
                arm_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
                    if suffix_expr is not None
                    else (ast.Pass(),)
                )
                return OrElseGuardedStatementDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=tuple(stmt.body),
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=(stmt,),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                )
            return None
        # Void `if PREFIX and isinstance(x, Sub) [and SUFFIX]: body` preserves PREFIX short-circuiting.
        if (
            not isinstance(stmt.test.op, ast.And)
            or stmt.orelse
            or _has_any_return(stmt.body)
        ):
            return None
        match_index: int | None = None
        matched_test: ParsedTypeTest | None = None
        for idx, value in enumerate(stmt.test.values[1:], start=1):
            parsed = _parse_type_test(value)
            if parsed is not None and not parsed.negated and value.lineno == test_line:
                match_index = idx
                matched_test = parsed
                break
        if match_index is None or matched_test is None:
            return None
        prefix_values = stmt.test.values[:match_index]
        guard_expr: ast.expr = (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=ast.And(), values=prefix_values)
        )
        suffix_values = stmt.test.values[match_index + 1 :]
        if suffix_values:
            suffix_expr: ast.expr = (
                suffix_values[0]
                if len(suffix_values) == 1
                else ast.BoolOp(op=ast.And(), values=suffix_values)
            )
            arm_body_guarded: tuple[ast.stmt, ...] = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
            )
        else:
            arm_body_guarded = tuple(stmt.body)
        return PrefixGuardedStatementDispatchSite(
            subject=matched_test.subject,
            kind=matched_test.kind,
            arms=(
                TypeTestArm(
                    matched_test.tested_exprs, arm_body_guarded, matched_test.exact
                ),
            ),
            default_body=(ast.Pass(),),
            is_whole_function=False,
            test_nodes=(matched_test.test_node,),
            replaced_nodes=(stmt,),
            guard_expr=guard_expr,
        )
    rest_values = stmt.test.values[1:]
    rest_expr: ast.expr = (
        rest_values[0]
        if len(rest_values) == 1
        else ast.BoolOp(op=stmt.test.op, values=rest_values)
    )
    if isinstance(stmt.test.op, ast.And) and not first_test.negated:
        # `if isinstance(x, Sub) and REST: body [else: orelse]`
        if stmt.orelse:
            arm_stmt: ast.stmt = ast.If(
                test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, (arm_stmt,), first_test.exact),),
                tuple(stmt.orelse),
                (stmt,),
                is_whole,
                (first_test.test_node,),
            )
        tail = container[index + 1 :]
        if _always_terminates(stmt.body) and tail and _always_terminates(tail):
            arm_body: tuple[ast.stmt, ...] = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            is_whole = container is whole_fn_body and index == 0
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                tuple(tail),
                tuple(container[index:]),
                is_whole,
                (first_test.test_node,),
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            arm_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=tuple(tail),
                is_whole_function=index == 0,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            arm_body = (
                ast.If(
                    test=rest_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=rewritten_tail,
                is_whole_function=False,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body):
            arm_stmt = ast.If(test=rest_expr, body=list(stmt.body), orelse=[])
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, (arm_stmt,), first_test.exact),),
                default_body=(ast.Pass(),),
                is_whole_function=is_whole,
                test_nodes=(first_test.test_node,),
                replaced_nodes=(stmt,),
            )
        return None
    if isinstance(stmt.test.op, ast.Or) and first_test.negated:
        # `if not isinstance(x, Sub) or REST: default [else: orelse]`
        if stmt.orelse:
            arm_stmt = ast.If(
                test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, (arm_stmt,), first_test.exact),),
                tuple(stmt.body),
                (stmt,),
                is_whole,
                (first_test.test_node,),
            )
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise):
            arm_stmt = ast.If(test=rest_expr, body=list(stmt.body), orelse=[])
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, (arm_stmt,), first_test.exact),),
                default_body=tuple(stmt.body),
                is_whole_function=is_whole,
                test_nodes=(first_test.test_node,),
                replaced_nodes=(stmt,),
            )
        tail = container[index + 1 :]
        if _always_terminates(stmt.body) and tail and _always_terminates(tail):
            arm_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            is_whole = container is whole_fn_body and index == 0
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                tuple(stmt.body),
                tuple(container[index:]),
                is_whole,
                (first_test.test_node,),
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            arm_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=_strip_trailing_bare_return(stmt.body),
                is_whole_function=index == 0,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            arm_body = (
                ast.If(
                    test=rest_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=_strip_trailing_continue(stmt.body),
                is_whole_function=False,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        return None
    return None


def _extract_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    test_line: int,
    in_loop_body: bool = False,
) -> DispatchSite | None:
    first_test = _parse_type_test(stmt.test)
    if first_test is None:
        return _extract_compound_if_site(
            container, index, stmt, whole_fn_body, test_line, in_loop_body=in_loop_body
        )
    subject_dump = ast.dump(first_test.subject)

    # Negated single `if not test:` with early return/raise or else.
    replaced: tuple[ast.stmt, ...]
    if first_test.negated:
        if stmt.orelse:
            # `if not isinstance(x, Sub): default else: arm`
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                return None
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            arms = (TypeTestArm(first_test.tested_exprs, tuple(stmt.orelse), first_test.exact),)
            default_body = tuple(stmt.body)
            return _build_statement_site(
                first_test, arms, default_body, replaced, is_whole, (first_test.test_node,)
            )
        # Guard without else: either `if not test: raise ...` (void) or early return followed by tail.
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise):
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            arms = (TypeTestArm(first_test.tested_exprs, (ast.Pass(),), first_test.exact),)
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=arms,
                default_body=tuple(stmt.body),
                is_whole_function=is_whole,
                test_nodes=(first_test.test_node,),
                replaced_nodes=replaced,
            )
        tail = container[index + 1 :]
        if tail and _always_terminates(stmt.body) and _always_terminates(tail):
            replaced = tuple(container[index:])
            is_whole = container is whole_fn_body and index == 0
            arms = (TypeTestArm(first_test.tested_exprs, tuple(tail), first_test.exact),)
            return _build_statement_site(
                first_test,
                arms,
                tuple(stmt.body),
                replaced,
                is_whole,
                (first_test.test_node,),
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, tuple(tail), first_test.exact),),
                default_body=_strip_trailing_bare_return(stmt.body),
                is_whole_function=index == 0,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, rewritten_tail, first_test.exact),),
                default_body=_strip_trailing_continue(stmt.body),
                is_whole_function=False,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        return None

    # Positive `if test:` / `elif test:` chain.
    arms_list: list[TypeTestArm] = [
        TypeTestArm(first_test.tested_exprs, tuple(stmt.body), first_test.exact)
    ]
    test_nodes: list[ast.AST] = [first_test.test_node]
    current_orelse = stmt.orelse
    while len(current_orelse) == 1 and isinstance(current_orelse[0], ast.If):
        branch = current_orelse[0]
        branch_test = _parse_type_test(branch.test)
        if branch_test is None or branch_test.negated:
            return None
        same_subject = ast.dump(branch_test.subject) == subject_dump
        same_kind = type(branch_test.kind) is type(first_test.kind)
        if not (same_subject and same_kind):
            return None
        arms_list.append(
            TypeTestArm(branch_test.tested_exprs, tuple(branch.body), branch_test.exact)
        )
        test_nodes.append(branch_test.test_node)
        current_orelse = branch.orelse

    if current_orelse:
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
        return _build_statement_site(
            first_test,
            tuple(arms_list),
            tuple(current_orelse),
            replaced,
            is_whole,
            tuple(test_nodes),
        )

    # No explicit `else:`: either early-return chain followed by terminating tail, void early-return, loop continue, or void `if`.
    tail = container[index + 1 :]
    if all(_always_terminates(arm.body) for arm in arms_list) and tail and _always_terminates(tail):
        replaced = tuple(container[index:])
        is_whole = container is whole_fn_body and index == 0
        return _build_statement_site(
            first_test,
            tuple(arms_list),
            tuple(tail),
            replaced,
            is_whole,
            tuple(test_nodes),
        )
    if (
        container is whole_fn_body
        and tail
        and all(_always_void_terminates(arm.body) for arm in arms_list)
        and not _has_value_return(tail)
    ):
        stripped_arms = tuple(
            TypeTestArm(
                arm.tested_exprs,
                _strip_trailing_bare_return(arm.body),
                arm.exact,
                alias=arm.alias,
            )
            for arm in arms_list
        )
        return StatementDispatchSite(
            subject=first_test.subject,
            kind=first_test.kind,
            arms=stripped_arms,
            default_body=tuple(tail),
            is_whole_function=index == 0,
            test_nodes=tuple(test_nodes),
            replaced_nodes=tuple(container[index:]),
        )
    if (
        in_loop_body
        and tail
        and all(_ends_with_continue_only(arm.body) for arm in arms_list)
        and not _has_non_continue_loop_escape(tail)
    ):
        rewritten_tail = _rewrite_loop_continues_to_returns(tail)
        stripped_arms = tuple(
            TypeTestArm(
                arm.tested_exprs,
                _strip_trailing_continue(arm.body),
                arm.exact,
                alias=arm.alias,
            )
            for arm in arms_list
        )
        return StatementDispatchSite(
            subject=first_test.subject,
            kind=first_test.kind,
            arms=stripped_arms,
            default_body=rewritten_tail,
            is_whole_function=False,
            test_nodes=tuple(test_nodes),
            replaced_nodes=tuple(container[index:]),
        )
    if not any(_has_any_return(arm.body) for arm in arms_list):
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
        return StatementDispatchSite(
            subject=first_test.subject,
            kind=first_test.kind,
            arms=tuple(arms_list),
            default_body=(ast.Pass(),),
            is_whole_function=is_whole,
            test_nodes=tuple(test_nodes),
            replaced_nodes=replaced,
        )
    return None


def _build_statement_site(
    first_test: ParsedTypeTest,
    arms: tuple[TypeTestArm, ...],
    default_body: tuple[ast.stmt, ...],
    replaced_nodes: tuple[ast.stmt, ...],
    is_whole_function: bool,
    test_nodes: tuple[ast.AST, ...],
) -> DispatchSite | None:
    all_bodies = (*(arm.body for arm in arms), default_body)
    any_returns = any(_has_any_return(body) for body in all_bodies)
    if any_returns:
        if not all(_always_terminates(body) for body in all_bodies):
            return None
        return ReturningStatementDispatchSite(
            subject=first_test.subject,
            kind=first_test.kind,
            arms=arms,
            default_body=default_body,
            is_whole_function=is_whole_function,
            test_nodes=test_nodes,
            replaced_nodes=replaced_nodes,
        )
    return StatementDispatchSite(
        subject=first_test.subject,
        kind=first_test.kind,
        arms=arms,
        default_body=default_body,
        is_whole_function=is_whole_function,
        test_nodes=test_nodes,
        replaced_nodes=replaced_nodes,
    )


def _extract_match_pattern_classes(
    pattern: ast.pattern,
) -> tuple[tuple[ast.expr, ...], str | None, tuple[tuple[str, str], ...]] | None:
    match pattern:
        case ast.MatchClass(
            cls=cls_expr, patterns=[], kwd_attrs=kwd_attrs, kwd_patterns=kwd_patterns
        ):
            bindings: list[tuple[str, str]] = []
            for attr, subpattern in zip(kwd_attrs, kwd_patterns, strict=True):
                match subpattern:
                    case ast.MatchAs(pattern=None, name=str() as bound_name):
                        bindings.append((attr, bound_name))
                    case _:
                        return None
            return ((cls_expr,), None, tuple(bindings))
        case ast.MatchAs(
            pattern=ast.MatchClass(
                cls=cls_expr, patterns=[], kwd_attrs=kwd_attrs, kwd_patterns=kwd_patterns
            ),
            name=str() as alias,
        ):
            bindings = []
            for attr, subpattern in zip(kwd_attrs, kwd_patterns, strict=True):
                match subpattern:
                    case ast.MatchAs(pattern=None, name=str() as bound_name):
                        bindings.append((attr, bound_name))
                    case _:
                        return None
            return ((cls_expr,), alias, tuple(bindings))
        case ast.MatchOr(patterns=subpatterns) if subpatterns:
            classes: list[ast.expr] = []
            for sub in subpatterns:
                extracted = _extract_match_pattern_classes(sub)
                if extracted is None or extracted[1] is not None or extracted[2]:
                    return None
                classes.extend(extracted[0])
            return (tuple(classes), None, ())
    return None


def _extract_match_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.Match,
    whole_fn_body: list[ast.stmt],
) -> DispatchSite | None:
    if not _is_pure_subject(stmt.subject) or not stmt.cases:
        return None
    arms: list[TypeTestArm] = []
    default_body: tuple[ast.stmt, ...] | None = None
    for case_idx, case in enumerate(stmt.cases):
        match case:
            case ast.match_case(guard=ast.expr()):
                return None
        match case.pattern:
            case ast.MatchAs(pattern=None, name=None):
                if case_idx != len(stmt.cases) - 1:
                    return None
                default_body = tuple(case.body)
                continue
        extracted = _extract_match_pattern_classes(case.pattern)
        if extracted is None:
            return None
        cls_exprs, alias, attr_bindings = extracted
        binding_stmts: tuple[ast.stmt, ...] = tuple(
            ast.Assign(
                targets=[ast.Name(id=bound_name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=copy.deepcopy(stmt.subject), attr=attr, ctx=ast.Load()
                ),
            )
            for attr, bound_name in attr_bindings
        )
        arms.append(
            TypeTestArm(cls_exprs, (*binding_stmts, *case.body), False, alias=alias)
        )
    if not arms:
        return None
    if default_body is None:
        tail = container[index + 1 :]
        if all(_always_terminates(arm.body) for arm in arms) and tail and _always_terminates(tail):
            replaced = tuple(container[index:])
            is_whole = container is whole_fn_body and index == 0
            default_body = tuple(tail)
        elif (
            container is whole_fn_body
            and tail
            and all(_always_void_terminates(arm.body) for arm in arms)
            and not _has_value_return(tail)
        ):
            arms = [
                TypeTestArm(
                    arm.tested_exprs,
                    _strip_trailing_bare_return(arm.body),
                    arm.exact,
                    alias=arm.alias,
                )
                for arm in arms
            ]
            replaced = tuple(container[index:])
            is_whole = index == 0
            default_body = tuple(tail)
            return StatementDispatchSite(
                subject=stmt.subject,
                kind=InstanceDispatchKind(),
                arms=tuple(arms),
                default_body=default_body,
                is_whole_function=is_whole,
                test_nodes=tuple(expr for arm in arms for expr in arm.tested_exprs),
                replaced_nodes=replaced,
            )
        elif not any(_has_any_return(arm.body) for arm in arms):
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            default_body = (ast.Pass(),)
        else:
            return None
    else:
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
    synthetic_test = ParsedTypeTest(
        stmt.subject, arms[0].tested_exprs, InstanceDispatchKind(), False, False, stmt.subject
    )
    return _build_statement_site(
        synthetic_test,
        tuple(arms),
        default_body,
        replaced,
        is_whole,
        tuple(expr for arm in arms for expr in arm.tested_exprs),
    )


def _extract_ternary_site(
    node: ast.IfExp,
    is_whole_return: bool,
) -> TernaryDispatchSite | None:
    parsed = _parse_type_test(node.test)
    if parsed is None:
        return None
    if parsed.negated:
        arm_body = (ast.Return(value=node.orelse),)
        default_body = (ast.Return(value=node.body),)
    else:
        arm_body = (ast.Return(value=node.body),)
        default_body = (ast.Return(value=node.orelse),)
    return TernaryDispatchSite(
        subject=parsed.subject,
        kind=parsed.kind,
        arms=(TypeTestArm(parsed.tested_exprs, arm_body, parsed.exact),),
        default_body=default_body,
        is_whole_function=is_whole_return,
        test_nodes=(parsed.test_node,),
        node=node,
    )


def _extract_boolean_expr_site(
    expr: ast.expr,
    is_whole_return: bool,
    test_line: int | None = None,
) -> BooleanPredicateDispatchSite | None:
    """Extract bare `isinstance(x, Sub)` or compound `[PREFIX and/or] isinstance(x, Sub) [and/or REST]`."""

    parsed = _parse_type_test(expr)
    if parsed is not None:
        true_val = not parsed.negated
        return BooleanPredicateDispatchSite(
            subject=parsed.subject,
            kind=parsed.kind,
            arms=(
                TypeTestArm(
                    parsed.tested_exprs,
                    (ast.Return(value=ast.Constant(value=true_val)),),
                    parsed.exact,
                ),
            ),
            default_body=(ast.Return(value=ast.Constant(value=not true_val)),),
            is_whole_function=is_whole_return,
            test_nodes=(parsed.test_node,),
            predicate_expr=expr,
        )
    if not isinstance(expr, ast.BoolOp) or len(expr.values) < 2:
        return None
    match_idx: int | None = None
    matched: ParsedTypeTest | None = None
    for idx, value in enumerate(expr.values):
        if test_line is not None and value.lineno != test_line:
            continue
        candidate = _parse_type_test(value)
        if candidate is not None:
            match_idx = idx
            matched = candidate
            break
    if match_idx is None or matched is None:
        return None
    rest_values = expr.values[match_idx + 1 :]
    if not rest_values:
        true_val = not matched.negated
        return BooleanPredicateDispatchSite(
            subject=matched.subject,
            kind=matched.kind,
            arms=(
                TypeTestArm(
                    matched.tested_exprs,
                    (ast.Return(value=ast.Constant(value=true_val)),),
                    matched.exact,
                ),
            ),
            default_body=(ast.Return(value=ast.Constant(value=not true_val)),),
            is_whole_function=False,
            test_nodes=(matched.test_node,),
            predicate_expr=expr.values[match_idx],
        )
    rest_expr: ast.expr = (
        rest_values[0]
        if len(rest_values) == 1
        else ast.BoolOp(op=expr.op, values=rest_values)
    )
    if isinstance(expr.op, ast.And):
        if matched.negated:
            arm_body: tuple[ast.stmt, ...] = (ast.Return(value=ast.Constant(value=False)),)
            default_body: tuple[ast.stmt, ...] = (ast.Return(value=rest_expr),)
        else:
            arm_body = (ast.Return(value=rest_expr),)
            default_body = (ast.Return(value=ast.Constant(value=False)),)
    else:
        if matched.negated:
            arm_body = (ast.Return(value=rest_expr),)
            default_body = (ast.Return(value=ast.Constant(value=True)),)
        else:
            arm_body = (ast.Return(value=ast.Constant(value=True)),)
            default_body = (ast.Return(value=rest_expr),)
    if match_idx == 0:
        return BooleanPredicateDispatchSite(
            subject=matched.subject,
            kind=matched.kind,
            arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
            default_body=default_body,
            is_whole_function=is_whole_return,
            test_nodes=(matched.test_node,),
            predicate_expr=expr,
        )
    return BoolOpSuffixDispatchSite(
        subject=matched.subject,
        kind=matched.kind,
        arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
        default_body=default_body,
        is_whole_function=False,
        test_nodes=(matched.test_node,),
        predicate_expr=expr.values[match_idx],
        end_expr=expr.values[-1],
    )


def find_dispatch_site(function: FunctionNode, test_line: int) -> DispatchSite | None:
    """Locate the normalized type-test dispatch in `function` whose test sits on `test_line`."""

    body = _body_without_docstring(function)
    for container, index, stmt, in_loop_body in _statement_list_children(body):
        if isinstance(stmt, ast.If) and (
            stmt.test.lineno == test_line
            or (
                isinstance(stmt.test, ast.BoolOp)
                and any(value.lineno == test_line for value in stmt.test.values)
            )
        ):
            extracted = _extract_if_site(
                container, index, stmt, body, test_line, in_loop_body=in_loop_body
            )
            if extracted is not None:
                return extracted
        if isinstance(stmt, ast.Match) and (
            stmt.lineno == test_line
            or any(case.pattern.lineno == test_line for case in stmt.cases)
        ):
            return _extract_match_site(container, index, stmt, body)
        if isinstance(stmt, ast.Return):
            match stmt:
                case ast.Return(value=ast.IfExp() as ifexp) if ifexp.test.lineno == test_line:
                    is_whole = container is body and len(body) == 1
                    return _extract_ternary_site(ifexp, is_whole)
                case ast.Return(value=ast.expr() as returned) if (
                    returned.lineno == test_line
                    or (
                        isinstance(returned, ast.BoolOp)
                        and any(value.lineno == test_line for value in returned.values)
                    )
                ):
                    is_whole = container is body and len(body) == 1
                    extracted = _extract_boolean_expr_site(returned, is_whole, test_line)
                    if extracted is not None:
                        return extracted
    # Fallback: ternary `IfExp` or non-branching compound `BoolOp` anywhere inside the function on `test_line`.
    branch_tests = {
        id(node.test)
        for node in ast.walk(function)
        if isinstance(node, (ast.If, ast.IfExp))
    }
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp) and node.test.lineno == test_line:
            return _extract_ternary_site(node, is_whole_return=False)
        if (
            isinstance(node, ast.BoolOp)
            and id(node) not in branch_tests
            and any(value.lineno == test_line for value in node.values)
        ):
            extracted = _extract_boolean_expr_site(
                node, is_whole_return=False, test_line=test_line
            )
            if extracted is not None:
                return extracted
    return None


def _function_local_names(function: FunctionNode) -> frozenset[str]:
    names: set[str] = {
        arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)
    }
    for stmt in function.body:
        for node in ast.walk(stmt):
            match node:
                case ast.Name(id=name, ctx=ast.Store()):
                    names.add(name)
                case ast.ExceptHandler(name=str() as name):
                    names.add(name)
                case ast.MatchAs(name=str() as name) | ast.MatchStar(name=str() as name):
                    names.add(name)
    return frozenset(names)


@dataclass(frozen=True)
class CapturedArgument:
    """One outer local forwarded into the pushed method, renaming on receiver collision."""

    call_expr: str
    param_name: str


def _outer_receiver_alias(
    function: FunctionNode,
    site: DispatchSite,
    raw_bodies: Sequence[tuple[ast.stmt, ...]],
) -> str | None:
    """Fresh parameter name when the outer function binds `receiver` independently of `site.subject`."""

    receiver = site.kind.receiver
    if isinstance(site.subject, ast.Name) and site.subject.id == receiver:
        return None
    fn_locals = _function_local_names(function)
    if receiver not in fn_locals:
        return None
    subject_dump = ast.dump(site.subject)
    uses_outer_receiver = False

    def scan(node: ast.AST) -> None:
        nonlocal uses_outer_receiver
        if uses_outer_receiver:
            return
        if isinstance(node, ast.expr) and ast.dump(node) == subject_dump:
            return
        if isinstance(node, ast.Name) and node.id == receiver:
            uses_outer_receiver = True
            return
        for child in ast.iter_child_nodes(node):
            scan(child)

    for body in raw_bodies:
        for stmt in body:
            scan(stmt)
    if not uses_outer_receiver:
        return None
    used_names = set(fn_locals)
    for body in raw_bodies:
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
    for candidate in ("owner", "caller", "other"):
        if candidate not in used_names:
            return candidate
    suffix = 1
    while f"outer_{receiver}_{suffix}" in used_names:
        suffix += 1
    return f"outer_{receiver}_{suffix}"


def _rewrite_subject_to_receiver(
    statements: tuple[ast.stmt, ...],
    subject: ast.expr,
    receiver: str,
    alias: str | None = None,
    outer_receiver_param: str | None = None,
) -> tuple[ast.stmt, ...]:
    subject_dump = ast.dump(subject)

    class _Rewriter(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if isinstance(node, ast.expr) and ast.dump(node) == subject_dump:
                return ast.Name(id=receiver, ctx=ast.Load())
            match node:
                case ast.Name(id=name, ctx=ast.Load()) if name == alias:
                    return ast.Name(id=receiver, ctx=ast.Load())
                case ast.Name(id=name, ctx=ast.Load()) if (
                    outer_receiver_param is not None and name == receiver
                ):
                    return ast.Name(id=outer_receiver_param, ctx=ast.Load())
            return super().visit(node)

    rewriter = _Rewriter()
    return tuple(cast(ast.stmt, rewriter.visit(copy.deepcopy(stmt))) for stmt in statements)


def _analyze_arm_bodies(
    function: FunctionNode,
    site: DispatchSite,
    rewritten_bodies: Sequence[tuple[ast.stmt, ...]],
    outer_receiver_param: str | None = None,
) -> tuple[tuple[CapturedArgument, ...], bool]:
    """Validate purity of rewritten bodies and return `(captured_args, is_async)`."""

    fn_locals = _function_local_names(function)
    receiver = site.kind.receiver
    captured: list[CapturedArgument] = []
    seen_captured: set[str] = set()
    uses_await = False

    for body in rewritten_bodies:
        local_stores: set[str] = set(_body_bound_names(body))
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Break, ast.Continue)):
                    raise NeedsInput(
                        f"`{function.name}` uses `{type(node).__name__}` inside a type-test "
                        "arm, which cannot be moved into a separate method",
                        NeededInput.SOURCE_CHANGE,
                    )
                if isinstance(node, (ast.Global, ast.Nonlocal)):
                    raise NeedsInput(
                        f"`{function.name}` uses `global`/`nonlocal` inside a type-test arm",
                        NeededInput.SOURCE_CHANGE,
                    )
                if isinstance(node, ast.Await):
                    uses_await = True
                match node:
                    case ast.Call(func=ast.Name(id="super")):
                        raise NeedsInput(
                            f"`{function.name}` calls `super()` inside a type-test arm",
                            NeededInput.SOURCE_CHANGE,
                        )
                    case ast.Name(id=name, ctx=ast.Del()):
                        local_stores.add(name)
        for stmt in body:
            for node in ast.walk(stmt):
                match node:
                    case ast.Name(id=name, ctx=ast.Load()) if (
                        outer_receiver_param is not None
                        and name == outer_receiver_param
                        and name not in local_stores
                        and name not in seen_captured
                    ):
                        seen_captured.add(name)
                        captured.append(CapturedArgument(call_expr=receiver, param_name=name))
                    case ast.Name(id=name, ctx=ast.Load()) if (
                        name != receiver
                        and name in fn_locals
                        and name not in local_stores
                        and name not in seen_captured
                    ):
                        seen_captured.add(name)
                        captured.append(CapturedArgument(call_expr=name, param_name=name))
                    case ast.Name(id=name, ctx=ast.Store() | ast.Del()) if name == receiver:
                        raise NeedsInput(
                            f"`{function.name}` rebinds the tested subject `{ast.unparse(site.subject)}`",
                            NeededInput.SOURCE_CHANGE,
                        )

    # Reject if an arm mutates or assigns a local variable read outside the replaced span.
    if not site.is_whole_function:
        for body in rewritten_bodies:
            for stmt in body:
                for node in ast.walk(stmt):
                    match node:
                        case ast.AugAssign(target=ast.Name(id=name)) if name in fn_locals:
                            raise NeedsInput(
                                f"`{function.name}` mutates outer local `{name}` inside a type-test arm",
                                NeededInput.SOURCE_CHANGE,
                            )
    if site.checks_local_store_escapes:
        replaced_nodes = site.replaced_ast_nodes()
        if replaced_nodes:
            stored = _names_stored_in(
                tuple(stmt for body in rewritten_bodies for stmt in body)
            )
            used_outside = _names_referenced_outside(function, replaced_nodes)
            escaping = sorted(stored & used_outside)
            if escaping:
                raise NeedsInput(
                    f"`{function.name}` assigns outer local(s) {', '.join(escaping)} inside a dispatch arm",
                    NeededInput.SOURCE_CHANGE,
                )

    if uses_await and not isinstance(function, ast.AsyncFunctionDef):
        raise NeedsInput(
            f"`{function.name}` uses `await` outside an async function",
            NeededInput.PROOF,
        )
    return tuple(captured), uses_await


def _unpack_union_annotations(expr: ast.expr) -> tuple[ast.expr, ...]:
    match expr:
        case ast.BinOp(op=ast.BitOr(), left=left, right=right):
            return (*_unpack_union_annotations(left), *_unpack_union_annotations(right))
        case ast.Subscript(
            value=ast.Name(id="Union") | ast.Attribute(attr="Union"),
            slice=ast.Tuple(elts=elts),
        ):
            return tuple(elts)
    return (expr,)


class FunctionEnclosure(ABC):
    """Whether a consumer function is module-level or a method of a scanned class."""

    @abstractmethod
    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        raise NotImplementedError

    def receiver_symbol(self, snapshot: CodemodSourceSnapshot) -> str | None:
        del snapshot
        return None

    def field_annotation(self, field_name: str) -> ast.expr | None:
        del field_name
        return None


@dataclass(frozen=True)
class ModuleEnclosure(FunctionEnclosure):
    """Top-level function declared directly in a module."""

    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        return module.body


@dataclass(frozen=True)
class ClassEnclosure(FunctionEnclosure):
    """Method declared directly on a scanned class."""

    owner: ResolvedClassTarget

    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        del module
        return self.owner.node.body

    def receiver_symbol(self, snapshot: CodemodSourceSnapshot) -> str | None:
        return self.owner.required_symbol(snapshot)

    def field_annotation(self, field_name: str) -> ast.expr | None:
        for statement in self.owner.node.body:
            match statement:
                case ast.AnnAssign(
                    target=ast.Name(id=target_name), annotation=ann
                ) if target_name == field_name:
                    return ann
        return None


def _resolve_subject_roots(
    snapshot: CodemodSourceSnapshot,
    resolver: AnsweredMethod,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
    site: DispatchSite,
    parents: AstParentIndex,
    source_path: str,
) -> tuple[tuple[str, ...], ...]:
    """Union branches of conjunctive scanned root classes every value of `site.subject` inhabits."""

    positional = (*function.args.posonlyargs, *function.args.args)
    first_param = positional[0].arg if positional else None
    annotation: ast.expr | None = None
    branches: list[tuple[str, ...]] = []

    match site.subject:
        case ast.Name(id=name) if name == first_param:
            owner_symbol = enclosure.receiver_symbol(snapshot)
            if owner_symbol is not None:
                branches.append((owner_symbol,))
            else:
                param = positional[0]
                ann_tuple = declared_annotations(param)
                annotation = ann_tuple[0] if ann_tuple else None
        case ast.Name(id=name):
            matched_param = next(
                (
                    arg
                    for arg in (*positional, *function.args.kwonlyargs)
                    if arg.arg == name
                ),
                None,
            )
            if matched_param is not None:
                ann_tuple = declared_annotations(matched_param)
                annotation = ann_tuple[0] if ann_tuple else None
        case ast.Attribute(value=ast.Name(id=name), attr=field_name) if (
            name == first_param
        ):
            annotation = enclosure.field_annotation(field_name)

    if not branches and annotation is not None:
        resolved_all = True
        for branch in _unpack_union_annotations(annotation):
            unwrapped = site.kind.unwrap_annotation(branch)
            symbol = (
                None
                if unwrapped is None
                else resolver.class_symbol(
                    source_path, unwrapped, cast(dict[ast.AST, ast.AST], parents)
                )
            )
            if symbol is None:
                resolved_all = False
                break
            branches.append((symbol,))
        if not resolved_all:
            branches.clear()

    if not branches:
        # Check that a bare local is not rebound before trusting mypy's inferred type.
        if isinstance(site.subject, ast.Name):
            parsed = snapshot.parsed_module_for_source_path(source_path)
            if not parsed.fact(ModuleNodeIndex).bound_once(site.subject):
                raise NeedsInput(
                    f"`{site.subject.id}` is bound more than once in `{function.name}`, "
                    "so no single inferred type holds every value it reaches",
                    NeededInput.PROOF,
                )
        span = ReceiverSpan.of(source_path, site.subject)
        answer = snapshot.type_checker.receiver_types((span,)).get(span)
        if not isinstance(answer, InferredReceiverType) or not answer.classes:
            raise NeedsInput(
                f"`{ast.unparse(site.subject)}` has no declared or inferred scanned class family",
                NeededInput.SOURCE_CHANGE,
            )
        classes_by_symbol = snapshot.required_class_family_index.classes_by_symbol
        inferred_branches = answer.branch_conjunctions()
        scanned_branches: list[tuple[str, ...]] = []
        for conjunction in inferred_branches:
            scanned = tuple(
                symbol for symbol in conjunction if symbol in classes_by_symbol
            )
            if not scanned:
                raise NeedsInput(
                    f"the type checker infers `{answer.spelled}` for `{ast.unparse(site.subject)}`, "
                    "which includes classes outside the scanned repository",
                    NeededInput.SOURCE_CHANGE,
                )
            scanned_branches.append(scanned)
        site.kind.require_inferred_compatible(answer, ast.unparse(site.subject))
        branches.extend(scanned_branches)

    return tuple(dict.fromkeys(branches))


def _nearest_shared_ancestor(
    snapshot: CodemodSourceSnapshot, branches: Sequence[tuple[str, ...]]
) -> str | None:
    index = snapshot.required_class_family_index
    if not branches:
        return None
    covered_per_branch = [
        frozenset().union(
            *(
                frozenset((sym, *index.ancestor_symbols(sym)))
                for sym in branch
                if sym in index.classes_by_symbol
            )
        )
        for branch in branches
    ]
    if any(not covered for covered in covered_per_branch):
        return None
    common = frozenset.intersection(*covered_per_branch)
    scanned = [sym for sym in common if sym in index.classes_by_symbol]
    if not scanned:
        return None
    return max(scanned, key=lambda sym: (len(index.ancestor_symbols(sym)), sym))


def _body_bound_names(body: Sequence[ast.stmt]) -> frozenset[str]:
    """Every lexical name bound inside `body` (stores, comprehensions, lambdas, defs, handlers, aliases)."""

    bound: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            match node:
                case ast.Name(id=name, ctx=ast.Store()):
                    bound.add(name)
                case ast.arg(arg=name):
                    bound.add(name)
                case ast.ExceptHandler(name=str() as name):
                    bound.add(name)
                case ast.MatchAs(name=str() as name) | ast.MatchStar(name=str() as name):
                    bound.add(name)
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                    bound.add(name)
                case ast.alias(name=raw_name, asname=asname):
                    bound.add(asname or raw_name.split(".", 1)[0])
    return frozenset(bound)


def _cross_module_imports_for_body(
    snapshot: CodemodSourceSnapshot,
    source_module: ParsedModule,
    source_line: int,
    destination: ResolvedClassTarget,
    body: tuple[ast.stmt, ...],
    local_exclusions: frozenset[str],
    rationale: str,
) -> tuple[ModuleImportMutation, ...]:
    """Import free module-level names referenced by `body` into `destination.file_path`."""

    if source_module.file_path == destination.file_path:
        return ()
    bindings = snapshot.module_binding_proof
    dest_module = snapshot.parsed_module_for_source_path(destination.file_path)
    excluded = local_exclusions | _body_bound_names(body)
    imports: list[ModuleImportMutation] = []
    names = sorted(
        {
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in excluded
        }
    )
    for name in names:
        reference = ast.Name(id=name, ctx=ast.Load())
        expected = bindings.reference_or_builtin_witness_at(
            source_module, reference, line=source_line
        )
        if expected is None:
            if name in builtins.__dict__:
                continue
            raise NeedsInput(
                f"free name `{name}` in {source_module.file_path} has no proved "
                f"cross-module import for {destination.file_path}",
                NeededInput.SOURCE_CHANGE,
            )
        actual = bindings.reference_or_builtin_witness_at(
            dest_module, reference, line=destination.node.lineno
        )
        if actual is not None:
            if actual.qualified_name != expected.qualified_name:
                raise NeedsInput(
                    f"`{name}` is `{actual.qualified_name}` in {destination.file_path}, "
                    f"not `{expected.qualified_name}`",
                    NeededInput.SOURCE_CHANGE,
                )
            continue
        targets = snapshot.source_index.targets_matching_repository_symbol(
            expected.qualified_name
        )
        if len(targets) != 1:
            raise NeedsInput(
                f"`{expected.qualified_name}` has no single scanned declaration to import",
                NeededInput.PROOF,
            )
        imports.append(
            ModuleImportMutation.from_source(
                file_path=destination.file_path,
                import_source=snapshot.module_import_graph.required_import_source(
                    importing_file_path=destination.file_path,
                    imported_file_path=targets[0].file_path,
                    imported_name=name,
                ),
                rationale=rationale,
            )
        )
    return tuple(imports)


def _render_method(
    kind: DispatchKind,
    method_name: str,
    params: tuple[str, ...],
    body: tuple[ast.stmt, ...],
    *,
    is_async: bool,
    docstring: str | None = None,
) -> str:
    prefix = "async def" if is_async else "def"
    sig_params = ", ".join((kind.receiver, *params))
    lines: list[str] = [*kind.decorators, f"{prefix} {method_name}({sig_params}):"]
    if docstring:
        lines.append(f'    """{docstring}"""')
    for stmt in body:
        for raw_line in ast.unparse(ast.fix_missing_locations(stmt)).splitlines():
            lines.append(f"    {raw_line}" if raw_line else "")
    return "\n".join(lines)


def _class_append_span(
    geometry: SourceTextGeometry, node: ast.ClassDef, member_source: str
) -> tuple[int, int, str]:
    """Append `member_source` onto `node`, replacing a bare `pass` body cleanly."""

    last = node.body[-1]
    indent = " " * last.col_offset
    if len(node.body) == 1 and isinstance(last, ast.Pass):
        start, end = geometry.required_node_offsets(last)
        rendered = "\n".join(
            f"{indent}{line}" if idx > 0 and line else line
            for idx, line in enumerate(member_source.splitlines())
        )
        return start, end, rendered
    if (
        len(node.body) == 2
        and _is_docstring_stmt(node.body[0])
        and isinstance(last, ast.Pass)
    ):
        start, end = geometry.required_node_offsets(last)
        rendered = "\n".join(
            f"{indent}{line}" if idx > 0 and line else line
            for idx, line in enumerate(member_source.splitlines())
        )
        return start, end, rendered
    _start, end = geometry.required_node_offsets(last)
    text = "".join(
        f"\n{indent}{line}" if line else "\n"
        for line in ("", *member_source.splitlines())
    )
    return end, end, text


@dataclass(frozen=True)
class ResolvedTargetArm:
    """One arm resolved to its scanned target classes and rewritten method body."""

    targets: tuple[ResolvedClassTarget, ...]
    symbols: tuple[str, ...]
    body: tuple[ast.stmt, ...]
    exact: bool


class TargetBaseSelection(ABC):
    """How the base class receiving the pushed dispatch method is chosen."""

    @abstractmethod
    def resolver_class(self, snapshot: CodemodSourceSnapshot) -> ResolvedClassTarget:
        raise NotImplementedError

    @abstractmethod
    def select_base(
        self,
        snapshot: CodemodSourceSnapshot,
        subject_roots: tuple[tuple[str, ...], ...],
        tested_symbols: Sequence[str],
        spelled_subject: str,
    ) -> tuple[ResolvedClassTarget, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class ExplicitBaseSelection(TargetBaseSelection):
    """Base class explicitly named on `PushTypeTestDispatchOperation`."""

    target: SourceRewriteTarget

    def resolver_class(self, snapshot: CodemodSourceSnapshot) -> ResolvedClassTarget:
        return ResolvedClassTarget.from_rewrite_target(snapshot, self.target)

    def select_base(
        self,
        snapshot: CodemodSourceSnapshot,
        subject_roots: tuple[tuple[str, ...], ...],
        tested_symbols: Sequence[str],
        spelled_subject: str,
    ) -> tuple[ResolvedClassTarget, str]:
        del subject_roots, tested_symbols, spelled_subject
        resolved = self.resolver_class(snapshot)
        return resolved, resolved.required_symbol(snapshot)


@dataclass(frozen=True)
class SharedAncestorBaseSelection(TargetBaseSelection):
    """Nearest scanned ancestor shared by the subject's family and every tested class."""

    def resolver_class(self, snapshot: CodemodSourceSnapshot) -> ResolvedClassTarget:
        classes_by_symbol = snapshot.required_class_family_index.classes_by_symbol
        first_symbol = next(iter(classes_by_symbol), None)
        if first_symbol is None:
            raise NeedsInput("scan has no declared classes", NeededInput.SOURCE_CHANGE)
        decl = classes_by_symbol[first_symbol]
        return ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname),
        )

    def select_base(
        self,
        snapshot: CodemodSourceSnapshot,
        subject_roots: tuple[tuple[str, ...], ...],
        tested_symbols: Sequence[str],
        spelled_subject: str,
    ) -> tuple[ResolvedClassTarget, str]:
        shared_base = _nearest_shared_ancestor(
            snapshot, (*subject_roots, *((symbol,) for symbol in tested_symbols))
        )
        if shared_base is None:
            # Multiple-inheritance mixin dispatch: subject belongs to a scanned base and
            # each tested class is a scanned mixin composed into subclasses of that base.
            # `_require_dispatch_mro_safe` proves C3 MRO dispatch equivalence for every member.
            subject_base = _nearest_shared_ancestor(snapshot, subject_roots)
            if subject_base is not None:
                index = snapshot.required_class_family_index
                family = frozenset(
                    (subject_base, *index.descendant_symbols(subject_base))
                )
                if tested_symbols and all(
                    any(
                        tested in (member, *index.ancestor_symbols(member))
                        for member in family
                    )
                    for tested in tested_symbols
                ):
                    shared_base = subject_base
        if shared_base is None:
            raise NeedsInput(
                f"`{spelled_subject}` and its tested classes share no scanned base class",
                NeededInput.SOURCE_CHANGE,
            )
        decl = snapshot.required_class_family_index.classes_by_symbol[shared_base]
        return (
            ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname),
            ),
            shared_base,
        )


@dataclass(frozen=True)
class TypeTestDispatchProof:
    """A type test proved safe to push onto a closed scanned class family."""

    source_path: str
    test_line: int
    function: FunctionNode
    site: DispatchSite
    base: ResolvedClassTarget
    base_symbol: str
    arms: tuple[ResolvedTargetArm, ...]
    default_body: tuple[ast.stmt, ...]
    captured_args: tuple[CapturedArgument, ...]
    is_async: bool
    method_name: str
    docstring: str | None

    @classmethod
    def locate_function(
        cls, snapshot: CodemodSourceSnapshot, target: SourceRewriteTarget
    ) -> tuple[str, FunctionEnclosure, FunctionNode]:
        source_path = target.file_path
        if not source_path or source_path not in snapshot.module_nodes_by_file_path:
            raise NeedsInput("target module is not in the source snapshot", NeededInput.SOURCE_CHANGE)
        qualname = target.qualname or ""
        class_name, _, function_name = qualname.rpartition(".")
        module = snapshot.module_nodes_by_file_path[source_path]
        if class_name:
            try:
                owner = ResolvedClassTarget.from_rewrite_target(
                    snapshot,
                    SourceRewriteTarget(file_path=source_path, qualname=class_name),
                )
            except ValueError as error:
                raise NeedsInput(
                    f"`{qualname}` is nested inside non-class `{class_name}`",
                    NeededInput.SOURCE_CHANGE,
                ) from error
            enclosure: FunctionEnclosure = ClassEnclosure(owner)
        else:
            enclosure = ModuleEnclosure()
        body = enclosure.body_statements(module)
        function = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ),
            None,
        )
        if function is None:
            raise NeedsInput(
                f"`{qualname}` is not a top-level function or method in `{source_path}`",
                NeededInput.SOURCE_CHANGE,
            )
        return source_path, enclosure, function

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        test_line: int,
        base_selection: TargetBaseSelection,
        method_name: str,
        docstring: str | None = None,
    ) -> TypeTestDispatchProof:
        if (
            not method_name.isidentifier()
            or keyword_module.iskeyword(method_name)
            or (method_name.startswith("__") and method_name.endswith("__"))
        ):
            raise NeedsInput(
                f"`{method_name}` is not a valid non-dunder method name",
                NeededInput.OWNER_PICK,
            )
        source_path, enclosure, function = cls.locate_function(snapshot, target)
        site = find_dispatch_site(function, test_line)
        if site is None:
            raise NeedsInput(
                f"type test at {source_path}:{test_line} is not a pushable dispatch shape",
                NeededInput.SOURCE_CHANGE,
            )
        parsed_module = snapshot.parsed_module_for_source_path(source_path)
        parents = parsed_module.fact(ModuleNodeIndex).parents

        index = snapshot.required_class_family_index
        classes_by_symbol = index.classes_by_symbol
        resolver = AnsweredMethod(snapshot, base_selection.resolver_class(snapshot), function)

        resolved_arm_symbols: list[tuple[str, ...]] = []
        all_tested_symbols: list[str] = []
        for arm in site.arms:
            symbols_for_arm: list[str] = []
            for expr in arm.tested_exprs:
                symbol = resolver.class_symbol(
                    source_path, expr, cast(dict[ast.AST, ast.AST], parents)
                )
                if symbol is None or symbol not in classes_by_symbol:
                    raise NeedsInput(
                        f"`{ast.unparse(expr)}` is not a class declared in the scan",
                        NeededInput.SOURCE_CHANGE,
                    )
                symbols_for_arm.append(symbol)
                all_tested_symbols.append(symbol)
            resolved_arm_symbols.append(tuple(symbols_for_arm))

        subject_roots = _resolve_subject_roots(
            snapshot, resolver, enclosure, function, site, parents, source_path
        )
        subject_family = frozenset().union(
            *(
                frozenset.intersection(
                    *(
                        frozenset((symbol, *index.descendant_symbols(symbol)))
                        for symbol in branch
                    )
                )
                for branch in subject_roots
            )
        )

        base, base_symbol = base_selection.select_base(
            snapshot, subject_roots, all_tested_symbols, ast.unparse(site.subject)
        )

        cls._require_dispatch_mro_safe(
            snapshot,
            subject_family,
            base,
            base_symbol,
            site.arms,
            resolved_arm_symbols,
            method_name,
        )

        receiver = site.kind.receiver
        outer_receiver_param = _outer_receiver_alias(
            function, site, (*(arm.body for arm in site.arms), site.default_body)
        )
        rewritten_arms: list[ResolvedTargetArm] = []
        for arm, arm_symbols in zip(site.arms, resolved_arm_symbols):
            targets = tuple(
                ResolvedClassTarget.from_rewrite_target(
                    snapshot,
                    SourceRewriteTarget(
                        file_path=classes_by_symbol[sym].file_path,
                        qualname=classes_by_symbol[sym].qualname,
                    ),
                )
                for sym in arm_symbols
            )
            rewritten_arms.append(
                ResolvedTargetArm(
                    targets=targets,
                    symbols=arm_symbols,
                    body=_rewrite_subject_to_receiver(
                        arm.body,
                        site.subject,
                        receiver,
                        alias=arm.alias,
                        outer_receiver_param=outer_receiver_param,
                    ),
                    exact=arm.exact,
                )
            )
        rewritten_default = _rewrite_subject_to_receiver(
            site.default_body,
            site.subject,
            receiver,
            outer_receiver_param=outer_receiver_param,
        )

        captured_args, is_async = _analyze_arm_bodies(
            function,
            site,
            (*(arm.body for arm in rewritten_arms), rewritten_default),
            outer_receiver_param=outer_receiver_param,
        )

        # Verify cross-module imports are cycle-free and unambiguous during proof.
        local_exclusions = frozenset(
            (receiver, *(arg.param_name for arg in captured_args))
        )
        rationale_probe = f"Push `{method_name}` onto `{base.qualname}`."
        _cross_module_imports_for_body(
            snapshot,
            parsed_module,
            test_line,
            base,
            rewritten_default,
            local_exclusions,
            rationale_probe,
        )
        for resolved_arm in rewritten_arms:
            for target_cls in resolved_arm.targets:
                _cross_module_imports_for_body(
                    snapshot,
                    parsed_module,
                    test_line,
                    target_cls,
                    resolved_arm.body,
                    local_exclusions,
                    rationale_probe,
                )

        return cls(
            source_path=source_path,
            test_line=test_line,
            function=function,
            site=site,
            base=base,
            base_symbol=base_symbol,
            arms=tuple(rewritten_arms),
            default_body=rewritten_default,
            captured_args=captured_args,
            is_async=is_async,
            method_name=method_name,
            docstring=docstring.strip() if docstring else None,
        )

    @staticmethod
    def _require_dispatch_mro_safe(
        snapshot: CodemodSourceSnapshot,
        subject_family: frozenset[str],
        base: ResolvedClassTarget,
        base_symbol: str,
        arms: tuple[TypeTestArm, ...],
        resolved_arm_symbols: list[tuple[str, ...]],
        method_name: str,
    ) -> None:
        classes_by_symbol = snapshot.required_class_family_index.classes_by_symbol
        if method_name in base.bound_names:
            raise NeedsInput(
                f"`{base.qualname}` already binds `{method_name}`",
                NeededInput.OWNER_PICK,
            )
        owner_arm: dict[str, int | None] = {base_symbol: None}
        for arm_idx, symbols_for_arm in enumerate(resolved_arm_symbols):
            for sym in symbols_for_arm:
                if sym in owner_arm:
                    raise NeedsInput(
                        f"`{classes_by_symbol[sym].qualname}` is tested in conflicting arms",
                        NeededInput.PROOF,
                    )
                target_cls = ResolvedClassTarget.from_rewrite_target(
                    snapshot,
                    SourceRewriteTarget(
                        file_path=classes_by_symbol[sym].file_path,
                        qualname=classes_by_symbol[sym].qualname,
                    ),
                )
                if method_name in target_cls.bound_names:
                    raise NeedsInput(
                        f"`{target_cls.qualname}` already binds `{method_name}`",
                        NeededInput.OWNER_PICK,
                    )
                # When a tested class is a mixin outside `base`'s subtree, verify no
                # ancestor or descendant of the mixin already binds `method_name`.
                compact_index = snapshot.compact_class_index
                compact_classes = compact_index.classes_by_symbol
                if base_symbol not in (sym, *compact_index.ancestor_symbols(sym)):
                    for related in (
                        *compact_index.ancestor_symbols(sym),
                        *compact_index.descendant_symbols(sym),
                    ):
                        related_decl = compact_classes.get(related)
                        if (
                            related_decl is not None
                            and method_name in related_decl.member_binding_names
                        ):
                            raise NeedsInput(
                                f"`{related_decl.qualname}` already declares `{method_name}` "
                                f"along mixin `{target_cls.qualname}`'s hierarchy",
                                NeededInput.OWNER_PICK,
                            )
                owner_arm[sym] = arm_idx

        authority = snapshot.compact_class_index.declared_mro_authority
        for symbol in sorted(subject_family):
            mro = authority.resolve(symbol).mro_type
            if mro is None:
                raise NeedsInput(
                    f"`{symbol}` has an open MRO, so dispatch of `{method_name}` cannot be proved",
                    NeededInput.PROOF,
                )
            order = tuple(decl.symbol for decl in mro.declarations)
            # Determine which arm `symbol` matched in original code.
            expected_arm: int | None = None
            for arm_idx, (arm, arm_syms) in enumerate(zip(arms, resolved_arm_symbols)):
                matched = (
                    symbol in arm_syms
                    if arm.exact
                    else any(tested in order for tested in arm_syms)
                )
                if matched:
                    expected_arm = arm_idx
                    break
            # Determine which owner answers `method_name` along `order`.
            winner_symbol: str | None = None
            for decl in mro.declarations:
                if decl.symbol in owner_arm:
                    winner_symbol = decl.symbol
                    break
                if method_name in decl.member_binding_names:
                    raise NeedsInput(
                        f"`{decl.symbol}` already declares `{method_name}` ahead of "
                        f"`{base.qualname}` for `{symbol}`",
                        NeededInput.OWNER_PICK,
                    )
            if winner_symbol is None:
                raise NeedsInput(
                    f"`{symbol}` reaches neither `{base.qualname}` nor any tested class",
                    NeededInput.PROOF,
                )
            if owner_arm[winner_symbol] != expected_arm:
                raise NeedsInput(
                    f"`{symbol}` would dispatch `{method_name}` via `{winner_symbol}`, "
                    "which disagrees with the original type-test branch order",
                    NeededInput.PROOF,
                )

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        geometries: dict[str, SourceTextGeometry] = {}
        replacements_by_file: dict[str, list[tuple[int, int, str]]] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        def add_span(file_path: str, start: int, end: int, text: str) -> None:
            replacements_by_file.setdefault(file_path, []).append((start, end, text))

        # 1. Consumer call replacement.
        call_args = ", ".join(arg.call_expr for arg in self.captured_args)
        param_names = tuple(arg.param_name for arg in self.captured_args)
        call = f"{ast.unparse(self.site.subject)}.{self.method_name}({call_args})"
        if self.is_async:
            call = f"await {call}"
        source_geometry = geometry(self.source_path)
        start, end = self.site.replaced_offsets(source_geometry)
        line_start = source_geometry.source.rfind("\n", 0, start) + 1
        line_indent = source_geometry.source[line_start:start]
        if not line_indent.isspace():
            line_indent = ""
        add_span(
            self.source_path,
            start,
            end,
            self.site.render_call_replacement(call, line_indent),
        )

        # 2. Base method and per-arm overrides.
        base_method = _render_method(
            self.site.kind,
            self.method_name,
            param_names,
            self.default_body,
            is_async=self.is_async,
            docstring=self.docstring,
        )
        add_span(
            self.base.file_path,
            *_class_append_span(geometry(self.base.file_path), self.base.node, base_method),
        )
        for resolved_arm in self.arms:
            arm_method = _render_method(
                self.site.kind,
                self.method_name,
                param_names,
                resolved_arm.body,
                is_async=self.is_async,
            )
            for target_cls in resolved_arm.targets:
                add_span(
                    target_cls.file_path,
                    *_class_append_span(
                        geometry(target_cls.file_path), target_cls.node, arm_method
                    ),
                )

        # 3. Cross-module imports for moved bodies and obsolete-import cleanup at consumer.
        parsed_source = snapshot.parsed_module_for_source_path(self.source_path)
        local_exclusions = frozenset((self.site.kind.receiver, *param_names))
        extra_edits: list[NominalSourceEdit] = [
            *_cross_module_imports_for_body(
                snapshot,
                parsed_source,
                self.test_line,
                self.base,
                self.default_body,
                local_exclusions,
                rationale,
            ),
        ]
        for resolved_arm in self.arms:
            for target_cls in resolved_arm.targets:
                extra_edits.extend(
                    _cross_module_imports_for_body(
                        snapshot,
                        parsed_source,
                        self.test_line,
                        target_cls,
                        resolved_arm.body,
                        local_exclusions,
                        rationale,
                    )
                )
        extra_edits.extend(
            obsolete_import_edits(
                snapshot, self.source_path, self.site.tested_expr_nodes(), rationale
            )
        )

        span_edits = tuple(
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=s, end_offset=e, replacement_source=text
                    )
                    for s, e, text in spans
                ),
                rationale=rationale,
            )
            for file_path, spans in replacements_by_file.items()
        )
        return (*span_edits, *extra_edits)

    def rationale(self) -> str:
        targets = ", ".join(
            f"`{cls.qualname}`"
            for arm in self.arms
            for cls in arm.targets
        )
        return (
            f"Pushes the type test in `{self.function.name}` onto "
            f"`{self.base.qualname}.{self.method_name}` with overrides on {targets}."
        )


@dataclass(frozen=True, kw_only=True)
class PushTypeTestDispatchOperation(RepositorySourceReprovedOperation):
    """Push one consumer type test on a scanned class family onto a method of that family."""

    test_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    base: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )

    def __post_init__(self) -> None:
        if self.test_line < 1:
            raise ValueError("test_line must be a positive line number")
        if not self.method_name.isidentifier() or keyword_module.iskeyword(
            self.method_name
        ):
            raise ValueError("method_name must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        proof = TypeTestDispatchProof.proved(
            snapshot,
            self.target,
            self.test_line,
            ExplicitBaseSelection(self.base),
            self.method_name,
            self.docstring,
        )
        return proof.edits(snapshot, self.rationale_text(proof.rationale()))


class OwnTypeTestFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
    TypeTestDispatchConcept,
    ABC,
):
    """Push an `OWN_TYPE_TEST` site onto its tested class family when proved safe."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not finding.evidence:
            return self.rejected_evaluation(
                "pushing a type test onto its class family needs source context and evidence"
            )
        snapshot = context.execution_snapshot()
        primary = finding.evidence[0]
        target = SourceRewriteTarget(
            file_path=primary.file_path, qualname=primary.symbol
        )
        pick = finding.metrics.picked_base
        try:
            _path, _owner, function = TypeTestDispatchProof.locate_function(
                snapshot, target
            )
        except ValueError as error:
            return self.stopped_evaluation(error)
        site = find_dispatch_site(function, primary.line)
        if site is None:
            return self.stopped_evaluation(
                NeedsInput(
                    f"type test at {primary.file_path}:{primary.line} is not a pushable dispatch shape",
                    NeededInput.SOURCE_CHANGE,
                )
            )
        if pick is not None:
            method_name = pick.name
            docstring: str | None = pick.judgement
        elif site.is_whole_function and not (
            function.name.startswith("__") and function.name.endswith("__")
        ):
            method_name = function.name
            docstring = ast.get_docstring(function)
        else:
            # Verify the site is structurally proved before asking for a method name.
            probe_name = "__nra_probe_method"
            try:
                proof = TypeTestDispatchProof.proved(
                    snapshot,
                    target,
                    primary.line,
                    SharedAncestorBaseSelection(),
                    probe_name,
                    None,
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            return self.stopped_evaluation(
                NeedsInput(
                    f"embedded type test in `{primary.symbol}` can push onto "
                    f"`{proof.base.qualname}`; supply `owner` `{primary.symbol}` with a "
                    "method name and one-line judgement in `--codemod-owner-picks`",
                    NeededInput.OWNER_PICK,
                )
            )

        try:
            proof = TypeTestDispatchProof.proved(
                snapshot,
                target,
                primary.line,
                SharedAncestorBaseSelection(),
                method_name,
                docstring,
            )
        except ValueError as error:
            return self.stopped_evaluation(error)

        operation = PushTypeTestDispatchOperation(
            target=target,
            test_line=primary.line,
            base=SourceRewriteTarget(
                file_path=proof.base.file_path, qualname=proof.base.qualname
            ),
            method_name=method_name,
            docstring=docstring,
        )
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=proof.rationale(),
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "BoolOpSuffixDispatchSite",
    "BooleanPredicateDispatchSite",
    "ClassDispatchKind",
    "DispatchKind",
    "DispatchSite",
    "InstanceDispatchKind",
    "OwnTypeTestFindingRecipeSynthesizer",
    "ParsedTypeTest",
    "PushTypeTestDispatchOperation",
    "ResolvedTargetArm",
    "ReturningStatementDispatchSite",
    "StatementDispatchSite",
    "TernaryDispatchSite",
    "TypeTestArm",
    "TypeTestDispatchProof",
    "find_dispatch_site",
)
