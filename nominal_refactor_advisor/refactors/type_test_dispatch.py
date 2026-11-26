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
import io
import keyword as keyword_module
import re
import tokenize
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, cast

from nominal_refactor_advisor.ast_field_decoders import (
    declared_annotations,
    exception_types,
    match_guards,
    match_patterns,
    optional_values,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_tools import AstParentIndex, ParsedModule
from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.class_mro import ClassMroAuthority, DeclarationMroType
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
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
    PhysicalSourceEdit,
    SourceInsertion,
    SourceLineSpan,
    SourceSpanDeletion,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_spacing import (
    SourceInsertionBoundary,
    SourceInsertionLayer,
)
from nominal_refactor_advisor.declared_receivers import (
    DeclaredClassObject,
    DeclaredInstance,
)
from nominal_refactor_advisor.models import OwnTypeTestMetrics, RefactorFinding
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import (
    InferredReceiverType,
    ProbeOwnerResolver,
    ReceiverSpan,
)
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

    @abstractmethod
    def matches_method_decorators(self, decorators: Sequence[ast.expr]) -> bool:
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

    def matches_method_decorators(self, decorators: Sequence[ast.expr]) -> bool:
        return not decorators


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

    def matches_method_decorators(self, decorators: Sequence[ast.expr]) -> bool:
        return (
            len(decorators) == 1
            and isinstance(decorators[0], ast.Name)
            and decorators[0].id == "classmethod"
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
    embeds_default: bool = False
    falls_through_to_default: bool = False


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

    @property
    def propagates_function_return_type(self) -> bool:
        return self.is_whole_function

    @property
    def is_void_method(self) -> bool:
        return False

    @property
    def returns_constant_bool(self) -> bool:
        return False

    @property
    def reassigned_target(self) -> str | None:
        """Outer local explicitly passed in, rebound/augmented, and returned by the pushed method."""
        return None


@dataclass(frozen=True)
class TernaryDispatchSite(DispatchSite):
    """`a if isinstance(x, Sub) else b` replaced inline by `x.method(...)`."""

    node: ast.IfExp

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)


@dataclass(frozen=True)
class PrefixGuardedAndTernaryDispatchSite(TernaryDispatchSite):
    """`A if PREFIX and [not] TEST else B` replaced by `x.method(...) if PREFIX else B`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return (
            f"{call_expr} if {ast.unparse(self.guard_expr)} "
            f"else {ast.unparse(self.short_circuit_expr)}"
        )


@dataclass(frozen=True)
class PrefixGuardedOrTernaryDispatchSite(TernaryDispatchSite):
    """`A if PREFIX or [not] TEST else B` replaced by `A if PREFIX else x.method(...)`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return (
            f"{ast.unparse(self.short_circuit_expr)} if {ast.unparse(self.guard_expr)} "
            f"else {call_expr}"
        )


@dataclass(frozen=True)
class BooleanPredicateDispatchSite(DispatchSite):
    """`return isinstance(x, Sub)` replaced by `return x.method()`."""

    predicate_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.predicate_expr)

    @property
    def returns_constant_bool(self) -> bool:
        return all(
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, bool)
            for body in (*(arm.body for arm in self.arms), self.default_body)
        )


@dataclass(frozen=True)
class BoolOpSuffixDispatchSite(BooleanPredicateDispatchSite):
    """Suffix `isinstance(x, Sub) and/or REST` inside a longer `BoolOp`."""

    end_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.predicate_expr)
        _, end = geometry.required_node_offsets(self.end_expr)
        tail = geometry.source[start:]
        rel_end = end - start
        line_offsets = [0]
        for line in tail.splitlines(keepends=True):
            line_offsets.append(line_offsets[-1] + len(line))
        depth = 0
        for tok in tokenize.generate_tokens(io.StringIO(tail).readline):
            tok_end = line_offsets[tok.end[0] - 1] + tok.end[1]
            if tok.type == tokenize.OP:
                if tok.string in ("(", "[", "{"):
                    depth += 1
                elif tok.string in (")", "]", "}"):
                    if depth == 0:
                        break
                    depth -= 1
                    if tok_end > rel_end and depth == 0:
                        return start, start + tok_end
            if tok_end >= rel_end and depth == 0:
                break
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

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def is_void_method(self) -> bool:
        return True


@dataclass(frozen=True)
class ReturningStatementDispatchSite(StatementDispatchSite):
    """Returning `if/elif/else`, early-return `if`, or `match` replaced by `return x.method(...)`."""

    @property
    def checks_local_store_escapes(self) -> bool:
        return False

    @property
    def propagates_function_return_type(self) -> bool:
        return True

    @property
    def is_void_method(self) -> bool:
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

    @property
    def propagates_function_return_type(self) -> bool:
        return self.returns_value

    @property
    def is_void_method(self) -> bool:
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
        case ast.Try(body=body, handlers=handlers, orelse=[]):
            if handlers:
                return _always_terminates(body) and all(
                    _always_terminates(handler.body) for handler in handlers
                )
            return bool(last.finalbody) and _always_terminates(body)
        case ast.Match(cases=cases) if (
            cases
            and not match_guards(cases[-1])
            and isinstance(cases[-1].pattern, ast.MatchAs)
            and not match_patterns(cases[-1].pattern)
        ):
            return all(_always_terminates(case.body) for case in cases)
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


def _has_break_or_continue(statements: Sequence[ast.stmt]) -> bool:
    """Whether `statements` contains `break` or `continue` targeting an enclosing loop."""

    stack: list[ast.AST] = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Break, ast.Continue)):
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


def _names_loaded_in(nodes: Sequence[ast.AST]) -> frozenset[str]:
    loaded: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
    return frozenset(loaded)


def _names_loaded_in_source(source: str) -> frozenset[str]:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError:
            return frozenset()
    return _names_loaded_in((parsed,))


def _super_fallback_return(
    method_name: str, param_names: Sequence[str], *, is_async: bool
) -> ast.Return:
    call: ast.expr = ast.Call(
        func=ast.Attribute(
            value=ast.Call(func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[]),
            attr=method_name,
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=name, ctx=ast.Load()) for name in param_names],
        keywords=[],
    )
    if is_async:
        call = ast.Await(value=call)
    return ast.Return(value=call)


def _target_store_names(node: ast.AST) -> frozenset[str]:
    """Names bound by an assignment or comprehension target expression."""

    return frozenset(
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    )


def _walk_comprehension_walrus_stores(node: ast.AST, stored: set[str]) -> None:
    """Collect PEP 572 `NamedExpr` targets inside a comprehension without entering nested defs/lambdas."""

    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef() | ast.Lambda():
            return
        case ast.NamedExpr(target=ast.Name(id=name), value=value):
            stored.add(name)
            _walk_comprehension_walrus_stores(value, stored)
            return
    for child in ast.iter_child_nodes(node):
        _walk_comprehension_walrus_stores(child, stored)


def _names_stored_in(nodes: Sequence[ast.AST]) -> frozenset[str]:
    """Names bound or deleted in the enclosing function scope by `nodes`."""

    stored: set[str] = set()

    def walk(node: ast.AST) -> None:
        match node:
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                stored.add(name)
                return
            case ast.ExceptHandler(name=name, body=body):
                if isinstance(name, str):
                    stored.add(name)
                for exc_type in exception_types(node):
                    walk(exc_type)
                for stmt in body:
                    walk(stmt)
                return
            case ast.MatchAs(name=name):
                if isinstance(name, str):
                    stored.add(name)
                for pattern in match_patterns(node):
                    walk(pattern)
                return
            case ast.MatchStar(name=name):
                if isinstance(name, str):
                    stored.add(name)
                return
            case ast.MatchMapping(keys=keys, patterns=patterns, rest=rest):
                if isinstance(rest, str):
                    stored.add(rest)
                for item in (*keys, *patterns):
                    walk(item)
                return
            case ast.FunctionDef(
                name=name,
                decorator_list=decorators,
                args=args,
            ) | ast.AsyncFunctionDef(
                name=name,
                decorator_list=decorators,
                args=args,
            ):
                stored.add(name)
                for expr in (
                    *decorators,
                    *args.defaults,
                    *(d for d in args.kw_defaults if d is not None),
                    *declared_annotations(node),
                ):
                    walk(expr)
                return
            case ast.ClassDef(
                name=name,
                decorator_list=decorators,
                bases=bases,
                keywords=keywords,
            ):
                stored.add(name)
                for item in (*decorators, *bases, *keywords):
                    walk(item)
                return
            case ast.Lambda(args=args):
                for expr in (
                    *args.defaults,
                    *(d for d in args.kw_defaults if d is not None),
                ):
                    walk(expr)
                return
            case ast.ListComp(elt=elt, generators=generators) | ast.SetComp(
                elt=elt, generators=generators
            ) | ast.GeneratorExp(elt=elt, generators=generators):
                if generators:
                    walk(generators[0].iter)
                    for gen in generators[1:]:
                        _walk_comprehension_walrus_stores(gen.iter, stored)
                    for gen in generators:
                        for cond in gen.ifs:
                            _walk_comprehension_walrus_stores(cond, stored)
                _walk_comprehension_walrus_stores(elt, stored)
                return
            case ast.DictComp(key=key, value=value, generators=generators):
                if generators:
                    walk(generators[0].iter)
                    for gen in generators[1:]:
                        _walk_comprehension_walrus_stores(gen.iter, stored)
                    for gen in generators:
                        for cond in gen.ifs:
                            _walk_comprehension_walrus_stores(cond, stored)
                _walk_comprehension_walrus_stores(key, stored)
                _walk_comprehension_walrus_stores(value, stored)
                return
        for child in ast.iter_child_nodes(node):
            walk(child)

    for root in nodes:
        walk(root)
    return frozenset(stored)


def _walrus_bound_before_loads(expr: ast.expr) -> set[str]:
    """Names unconditionally assigned by `:=` before any read inside a comprehension `if` clause."""

    bound: set[str] = set()
    read_before: set[str] = set()

    def walk(node: ast.AST) -> None:
        match node:
            case ast.NamedExpr(target=ast.Name(id=name), value=value):
                walk(value)
                if name not in read_before:
                    bound.add(name)
                return
            case ast.Name(id=name, ctx=ast.Load()):
                if name not in bound:
                    read_before.add(name)
                return
            case ast.BoolOp(op=ast.And(), values=values):
                for value in values:
                    walk(value)
                return
            case ast.BoolOp(op=ast.Or(), values=values):
                if values:
                    walk(values[0])
                return
            case ast.Compare(left=left, comparators=comparators):
                walk(left)
                for comp in comparators:
                    walk(comp)
                return
            case ast.UnaryOp(operand=operand):
                walk(operand)
                return
            case ast.IfExp(test=test, body=body, orelse=orelse):
                walk(test)
                for branch in (body, orelse):
                    for inner in ast.walk(branch):
                        if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                            if inner.id not in bound:
                                read_before.add(inner.id)
                return
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef() | ast.Lambda():
                return
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(expr)
    return bound


def _names_referenced_outside(
    function: FunctionNode, excluded_nodes: Sequence[ast.AST]
) -> frozenset[str]:
    """Enclosing-scope `Load` references outside `excluded_nodes`, respecting inner scopes."""

    excluded_ids = {id(node) for node in excluded_nodes}
    referenced: set[str] = set()

    def walk(node: ast.AST, shadowed: frozenset[str]) -> None:
        if id(node) in excluded_ids:
            return
        match node:
            case ast.Name(id=name, ctx=ast.Load()):
                if name not in shadowed:
                    referenced.add(name)
                return
            case ast.Name():
                return
            case ast.ListComp(elt=elt, generators=generators) | ast.SetComp(
                elt=elt, generators=generators
            ) | ast.GeneratorExp(elt=elt, generators=generators):
                comp_shadow = shadowed
                for idx, gen in enumerate(generators):
                    walk(gen.iter, shadowed if idx == 0 else comp_shadow)
                    comp_shadow = comp_shadow | _target_store_names(gen.target)
                    for cond in gen.ifs:
                        # Any name bound by `:=` before its own read in `cond` shadows
                        # outer bindings for `cond`'s trailing uses, later `ifs`, and `elt`.
                        walrus = _walrus_bound_before_loads(cond)
                        walk(cond, comp_shadow | walrus)
                        comp_shadow = comp_shadow | walrus
                walk(elt, comp_shadow)
                return
            case ast.DictComp(key=key, value=value, generators=generators):
                comp_shadow = shadowed
                for idx, gen in enumerate(generators):
                    walk(gen.iter, shadowed if idx == 0 else comp_shadow)
                    comp_shadow = comp_shadow | _target_store_names(gen.target)
                    for cond in gen.ifs:
                        walrus = _walrus_bound_before_loads(cond)
                        walk(cond, comp_shadow | walrus)
                        comp_shadow = comp_shadow | walrus
                walk(key, comp_shadow)
                walk(value, comp_shadow)
                return
            case ast.Lambda(args=args, body=body):
                for expr in (*args.defaults, *(d for d in args.kw_defaults if d is not None)):
                    walk(expr, shadowed)
                param_names = frozenset(
                    arg.arg
                    for arg in (
                        *args.posonlyargs,
                        *args.args,
                        *args.kwonlyargs,
                        *variadic_parameters(args),
                    )
                )
                walk(body, shadowed | param_names)
                return
            case ast.FunctionDef(
                decorator_list=decorators, args=args, body=body
            ) | ast.AsyncFunctionDef(
                decorator_list=decorators, args=args, body=body
            ):
                for expr in (
                    *decorators,
                    *args.defaults,
                    *(d for d in args.kw_defaults if d is not None),
                    *declared_annotations(node),
                ):
                    walk(expr, shadowed)
                param_names = frozenset(
                    arg.arg
                    for arg in (
                        *args.posonlyargs,
                        *args.args,
                        *args.kwonlyargs,
                        *variadic_parameters(args),
                    )
                )
                nonlocals = frozenset(
                    name
                    for stmt in body
                    for inner in ast.walk(stmt)
                    if isinstance(inner, ast.Nonlocal)
                    for name in inner.names
                )
                inner_shadow = shadowed | param_names | (_names_stored_in(body) - nonlocals)
                for stmt in body:
                    walk(stmt, inner_shadow)
                return
            case ast.ClassDef(
                decorator_list=decorators, bases=bases, keywords=keywords, body=body
            ):
                for item in (*decorators, *bases, *keywords):
                    walk(item, shadowed)
                inner_shadow = shadowed | _names_stored_in(body)
                for stmt in body:
                    walk(stmt, inner_shadow)
                return
        for child in ast.iter_child_nodes(node):
            walk(child, shadowed)

    for stmt in function.body:
        walk(stmt, frozenset())
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


def _polarity_bodies(
    negated: bool,
    hit_body: tuple[ast.stmt, ...],
    miss_body: tuple[ast.stmt, ...],
) -> tuple[tuple[ast.stmt, ...], tuple[ast.stmt, ...]]:
    """Order `(arm_body, default_body)` for a positive or negated type test."""

    if negated:
        return miss_body, hit_body
    return hit_body, miss_body


def _extract_compound_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    test_line: int,
    in_loop_body: bool = False,
) -> DispatchSite | None:
    """Extract `if [PREFIX and/or] [not] TEST [and/or REST]:`."""

    if not isinstance(stmt.test, ast.BoolOp) or len(stmt.test.values) < 2:
        return None
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        return None
    first_value = stmt.test.values[0]
    first_test = (
        _parse_type_test(first_value)
        if first_value.lineno <= test_line <= (first_value.end_lineno or first_value.lineno)
        else None
    )
    if first_test is None:
        if isinstance(stmt.test.op, ast.Or):
            match_index: int | None = None
            matched_test: ParsedTypeTest | None = None
            for idx, value in enumerate(stmt.test.values[1:], start=1):
                parsed = _parse_type_test(value)
                if parsed is not None and value.lineno <= test_line <= (
                    value.end_lineno or value.lineno
                ):
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
                    fallback_body: tuple[ast.stmt, ...] = (
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
                    arm_body, default_body = _polarity_bodies(
                        matched_test.negated, tuple(stmt.body), fallback_body
                    )
                    return PrefixGuardedOrDispatchSite(
                        subject=matched_test.subject,
                        kind=matched_test.kind,
                        arms=(
                            TypeTestArm(
                                matched_test.tested_exprs, arm_body, matched_test.exact
                            ),
                        ),
                        default_body=default_body,
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
                fallback_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                arm_body, default_body = _polarity_bodies(
                    matched_test.negated, tuple(stmt.body), fallback_body
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=default_body,
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
                fallback_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                arm_body, default_body = _polarity_bodies(
                    matched_test.negated, stripped, fallback_body
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=default_body,
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
                fallback_body = (
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
                arm_body, default_body = _polarity_bodies(
                    matched_test.negated, stripped, fallback_body
                )
                return PrefixGuardedOrDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=default_body,
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=tuple(container[index:]),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=False,
                )
            if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
                fallback_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
                    if suffix_expr is not None
                    else (ast.Pass(),)
                )
                arm_body, default_body = _polarity_bodies(
                    matched_test.negated, tuple(stmt.body), fallback_body
                )
                return OrElseGuardedStatementDispatchSite(
                    subject=matched_test.subject,
                    kind=matched_test.kind,
                    arms=(
                        TypeTestArm(
                            matched_test.tested_exprs, arm_body, matched_test.exact
                        ),
                    ),
                    default_body=default_body,
                    is_whole_function=False,
                    test_nodes=(matched_test.test_node,),
                    replaced_nodes=(stmt,),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                )
            return None
        # Void `if PREFIX and [not] TEST [and SUFFIX]: body` preserves PREFIX short-circuiting.
        if (
            not isinstance(stmt.test.op, ast.And)
            or stmt.orelse
            or _has_any_return(stmt.body)
            or _has_loop_escape(stmt.body)
        ):
            return None
        match_index = None
        matched_test = None
        for idx, value in enumerate(stmt.test.values[1:], start=1):
            parsed = _parse_type_test(value)
            if parsed is not None and value.lineno <= test_line <= (
                value.end_lineno or value.lineno
            ):
                match_index = idx
                matched_test = parsed
                break
        if match_index is None or matched_test is None:
            return None
        prefix_values = stmt.test.values[:match_index]
        guard_expr = (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=ast.And(), values=prefix_values)
        )
        suffix_values = stmt.test.values[match_index + 1 :]
        if suffix_values:
            suffix_expr = (
                suffix_values[0]
                if len(suffix_values) == 1
                else ast.BoolOp(op=ast.And(), values=suffix_values)
            )
            hit_body: tuple[ast.stmt, ...] = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
            )
        else:
            hit_body = tuple(stmt.body)
        arm_body, default_body = _polarity_bodies(
            matched_test.negated, hit_body, (ast.Pass(),)
        )
        return PrefixGuardedStatementDispatchSite(
            subject=matched_test.subject,
            kind=matched_test.kind,
            arms=(
                TypeTestArm(
                    matched_test.tested_exprs, arm_body, matched_test.exact
                ),
            ),
            default_body=default_body,
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
    if isinstance(stmt.test.op, ast.And):
        # `if [not] TEST and REST: body [else: orelse]`
        tail = container[index + 1 :]
        if stmt.orelse:
            hit_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)),
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, hit_body, tuple(stmt.orelse)
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body,
                (stmt,),
                is_whole,
                (first_test.test_node,),
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
            )
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
            and not _has_break_or_continue(stmt.body)
            and not _has_break_or_continue(tail)
        ):
            tail_tuple = tuple(tail)
            if not first_test.negated and _names_stored_in(stmt.body).isdisjoint(
                _names_loaded_in(tail_tuple)
            ):
                arm_only: tuple[ast.stmt, ...] = (
                    ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                )
                return ReturningStatementDispatchSite(
                    subject=first_test.subject,
                    kind=first_test.kind,
                    arms=(
                        TypeTestArm(
                            first_test.tested_exprs,
                            arm_only,
                            first_test.exact,
                            falls_through_to_default=True,
                        ),
                    ),
                    default_body=tail_tuple,
                    is_whole_function=container is whole_fn_body and index == 0,
                    test_nodes=(first_test.test_node,),
                    replaced_nodes=tuple(container[index:]),
                )
            hit_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, hit_body, tail_tuple
            )
            is_whole = container is whole_fn_body and index == 0
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body,
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
            hit_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, hit_body, tuple(tail)
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
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
            hit_body = (
                ast.If(
                    test=rest_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, hit_body, rewritten_tail
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
                is_whole_function=False,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            hit_body = (ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),)
            arm_body, default_body = _polarity_bodies(
                first_test.negated, hit_body, (ast.Pass(),)
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
                is_whole_function=is_whole,
                test_nodes=(first_test.test_node,),
                replaced_nodes=(stmt,),
            )
        return None
    if isinstance(stmt.test.op, ast.Or):
        # `if [not] TEST or REST: body [else: orelse]`
        tail = container[index + 1 :]
        if stmt.orelse:
            miss_body: tuple[ast.stmt, ...] = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)),
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, tuple(stmt.body), miss_body
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body,
                (stmt,),
                is_whole,
                (first_test.test_node,),
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
            )
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
        ):
            miss_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, tuple(stmt.body), miss_body
            )
            is_whole = container is whole_fn_body and index == 0
            return _build_statement_site(
                first_test,
                (TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body,
                tuple(container[index:]),
                is_whole,
                (first_test.test_node,),
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            miss_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, _strip_trailing_bare_return(stmt.body), miss_body
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
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
            miss_body = (
                ast.If(
                    test=rest_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            arm_body, default_body = _polarity_bodies(
                first_test.negated, _strip_trailing_continue(stmt.body), miss_body
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
                is_whole_function=False,
                test_nodes=(first_test.test_node,),
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            miss_body = (ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),)
            arm_body, default_body = _polarity_bodies(
                first_test.negated, tuple(stmt.body), miss_body
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=(TypeTestArm(first_test.tested_exprs, arm_body, first_test.exact),),
                default_body=default_body,
                is_whole_function=is_whole,
                test_nodes=(first_test.test_node,),
                replaced_nodes=(stmt,),
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
                first_test,
                arms,
                default_body,
                replaced,
                is_whole,
                (first_test.test_node,),
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
            )
        # Guard without else: early return followed by tail, loop continue, or void/raising `if not test:`.
        tail = container[index + 1 :]
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
        ):
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
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
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
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
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
            container=container,
            index=index,
            whole_fn_body=whole_fn_body,
        )

    # No explicit `else:`: absorb consecutive terminating `if` chains on the same subject
    # (e.g. `if isinstance(x, A): return ...; if isinstance(x, B): return ...; <tail>`),
    # then handle early-return tail, void early-return, loop continue, or void `if`.
    merged_arms = list(arms_list)
    merged_test_nodes = list(test_nodes)
    end_index = index + 1
    if all(_always_terminates(arm.body) for arm in merged_arms):
        while end_index < len(container):
            nxt = container[end_index]
            if not isinstance(nxt, ast.If):
                break
            nxt_test = _parse_type_test(nxt.test)
            if (
                nxt_test is None
                or nxt_test.negated
                or ast.dump(nxt_test.subject) != subject_dump
                or type(nxt_test.kind) is not type(first_test.kind)
            ):
                break
            nxt_arms: list[TypeTestArm] = [
                TypeTestArm(nxt_test.tested_exprs, tuple(nxt.body), nxt_test.exact)
            ]
            nxt_test_nodes: list[ast.AST] = [nxt_test.test_node]
            nxt_orelse = nxt.orelse
            valid_chain = True
            while len(nxt_orelse) == 1 and isinstance(nxt_orelse[0], ast.If):
                branch = nxt_orelse[0]
                branch_test = _parse_type_test(branch.test)
                if (
                    branch_test is None
                    or branch_test.negated
                    or ast.dump(branch_test.subject) != subject_dump
                    or type(branch_test.kind) is not type(first_test.kind)
                ):
                    valid_chain = False
                    break
                nxt_arms.append(
                    TypeTestArm(
                        branch_test.tested_exprs, tuple(branch.body), branch_test.exact
                    )
                )
                nxt_test_nodes.append(branch_test.test_node)
                nxt_orelse = branch.orelse
            if (
                not valid_chain
                or nxt_orelse
                or not all(_always_terminates(arm.body) for arm in nxt_arms)
            ):
                break
            merged_arms.extend(nxt_arms)
            merged_test_nodes.extend(nxt_test_nodes)
            end_index += 1

    if end_index > index + 1:
        merged_tail = container[end_index:]
        if merged_tail and _always_terminates(merged_tail):
            return _build_statement_site(
                first_test,
                tuple(merged_arms),
                tuple(merged_tail),
                tuple(container[index:]),
                container is whole_fn_body and index == 0,
                tuple(merged_test_nodes),
                container=container,
                index=index,
                whole_fn_body=whole_fn_body,
            )
        if (
            container is whole_fn_body
            and merged_tail
            and all(_always_void_terminates(arm.body) for arm in merged_arms)
            and not _has_value_return(merged_tail)
        ):
            stripped_merged = tuple(
                TypeTestArm(
                    arm.tested_exprs,
                    _strip_trailing_bare_return(arm.body),
                    arm.exact,
                    alias=arm.alias,
                )
                for arm in merged_arms
            )
            return StatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=stripped_merged,
                default_body=tuple(merged_tail),
                is_whole_function=index == 0,
                test_nodes=tuple(merged_test_nodes),
                replaced_nodes=tuple(container[index:]),
            )

    tail = container[index + 1 :]
    if (
        tail
        and _always_terminates(tail)
        and (
            all(_always_terminates(arm.body) for arm in arms_list)
            or any(_has_any_return(arm.body) for arm in arms_list)
        )
    ):
        replaced = tuple(container[index:])
        is_whole = container is whole_fn_body and index == 0
        return _build_statement_site(
            first_test,
            tuple(arms_list),
            tuple(tail),
            replaced,
            is_whole,
            tuple(test_nodes),
            container=container,
            index=index,
            whole_fn_body=whole_fn_body,
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
    *,
    container: list[ast.stmt] | None = None,
    index: int | None = None,
    whole_fn_body: list[ast.stmt] | None = None,
) -> DispatchSite | None:
    all_bodies = (*(arm.body for arm in arms), default_body)
    any_returns = any(_has_any_return(body) for body in all_bodies)
    if any_returns:
        if all(_always_terminates(body) for body in all_bodies):
            return ReturningStatementDispatchSite(
                subject=first_test.subject,
                kind=first_test.kind,
                arms=arms,
                default_body=default_body,
                is_whole_function=is_whole_function,
                test_nodes=test_nodes,
                replaced_nodes=replaced_nodes,
            )
        if container is None or index is None or whole_fn_body is None:
            return None
        tail = tuple(container[index + 1 :])
        if not tail or not _always_terminates(tail) or _has_break_or_continue(tail):
            return None
        if any(
            not _always_terminates(body) and _has_break_or_continue(body)
            for body in all_bodies
        ):
            return None
        extended_default = (
            default_body
            if _always_terminates(default_body)
            else (*default_body, *tail)
        )
        default_is_tail = extended_default == tail
        tail_loads = _names_loaded_in(tail) if default_is_tail else frozenset()
        extended_arms = tuple(
            arm
            if _always_terminates(arm.body)
            else (
                TypeTestArm(
                    arm.tested_exprs,
                    arm.body,
                    arm.exact,
                    alias=arm.alias,
                    embeds_default=arm.embeds_default,
                    falls_through_to_default=True,
                )
                if default_is_tail
                and _names_stored_in(arm.body).isdisjoint(tail_loads)
                else TypeTestArm(
                    arm.tested_exprs,
                    (*arm.body, *tail),
                    arm.exact,
                    alias=arm.alias,
                    embeds_default=arm.embeds_default,
                )
            )
            for arm in arms
        )
        return ReturningStatementDispatchSite(
            subject=first_test.subject,
            kind=first_test.kind,
            arms=extended_arms,
            default_body=extended_default,
            is_whole_function=container is whole_fn_body and index == 0,
            test_nodes=test_nodes,
            replaced_nodes=tuple(container[index:]),
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


@dataclass(frozen=True)
class _ExtractedMatchClassCase:
    """One class-pattern `match` case with its tested class(es), optional alias, and keyword patterns."""

    tested_exprs: tuple[ast.expr, ...]
    alias: str | None
    kwd_attrs: tuple[str, ...]
    kwd_patterns: tuple[ast.pattern, ...]

    @property
    def has_subpatterns(self) -> bool:
        return any(
            not (
                isinstance(pat, ast.MatchAs)
                and not match_patterns(pat)
                and isinstance(pat.name, str)
            )
            for pat in self.kwd_patterns
        )


def _extract_match_pattern_classes(
    pattern: ast.pattern,
) -> _ExtractedMatchClassCase | None:
    match pattern:
        case ast.MatchClass(
            cls=cls_expr, patterns=[], kwd_attrs=kwd_attrs, kwd_patterns=kwd_patterns
        ):
            return _ExtractedMatchClassCase(
                tested_exprs=(cls_expr,),
                alias=None,
                kwd_attrs=tuple(kwd_attrs),
                kwd_patterns=tuple(kwd_patterns),
            )
        case ast.MatchAs(
            pattern=ast.MatchClass(
                cls=cls_expr, patterns=[], kwd_attrs=kwd_attrs, kwd_patterns=kwd_patterns
            ),
            name=str() as alias,
        ):
            return _ExtractedMatchClassCase(
                tested_exprs=(cls_expr,),
                alias=alias,
                kwd_attrs=tuple(kwd_attrs),
                kwd_patterns=tuple(kwd_patterns),
            )
        case ast.MatchOr(patterns=subpatterns) if subpatterns:
            classes: list[ast.expr] = []
            for sub in subpatterns:
                extracted = _extract_match_pattern_classes(sub)
                if extracted is None or extracted.alias is not None or extracted.kwd_attrs:
                    return None
                classes.extend(extracted.tested_exprs)
            return _ExtractedMatchClassCase(
                tested_exprs=tuple(classes),
                alias=None,
                kwd_attrs=(),
                kwd_patterns=(),
            )
    return None


def _is_pass_only(statements: Sequence[ast.stmt]) -> bool:
    return len(statements) == 1 and isinstance(statements[0], ast.Pass)


def _is_dispatchable_match_subject(expr: ast.expr) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Lambda)):
            return False
    return True


def _expr_occurs_in_nodes(expr_dump: str, nodes: Sequence[ast.AST]) -> bool:
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.expr) and ast.dump(node) == expr_dump:
                return True
    return False


def _build_match_arm_body(
    subject: ast.expr,
    extracted: _ExtractedMatchClassCase,
    guard: ast.expr | None,
    case_body: tuple[ast.stmt, ...],
    fallback_body: tuple[ast.stmt, ...],
) -> tuple[ast.stmt, ...]:
    if not extracted.has_subpatterns:
        binding_stmts: tuple[ast.stmt, ...] = tuple(
            ast.Assign(
                targets=[ast.Name(id=cast(str, cast(ast.MatchAs, pat).name), ctx=ast.Store())],
                value=ast.Attribute(
                    value=copy.deepcopy(subject), attr=attr, ctx=ast.Load()
                ),
            )
            for attr, pat in zip(extracted.kwd_attrs, extracted.kwd_patterns, strict=True)
        )
        if guard is None:
            return (*binding_stmts, *case_body)
        orelse = [] if _is_pass_only(fallback_body) else list(fallback_body)
        return (
            *binding_stmts,
            ast.If(test=copy.deepcopy(guard), body=list(case_body), orelse=orelse),
        )

    attr_exprs: list[ast.expr] = [
        ast.Attribute(value=copy.deepcopy(subject), attr=attr, ctx=ast.Load())
        for attr in extracted.kwd_attrs
    ]
    if len(attr_exprs) == 1:
        inner_subject: ast.expr = attr_exprs[0]
        inner_pattern: ast.pattern = copy.deepcopy(extracted.kwd_patterns[0])
    else:
        inner_subject = ast.Tuple(elts=attr_exprs, ctx=ast.Load())
        inner_pattern = ast.MatchSequence(
            patterns=[copy.deepcopy(pat) for pat in extracted.kwd_patterns]
        )
    inner_cases = [
        ast.match_case(
            pattern=inner_pattern,
            guard=None if guard is None else copy.deepcopy(guard),
            body=list(case_body),
        )
    ]
    if not _is_pass_only(fallback_body):
        inner_cases.append(
            ast.match_case(
                pattern=ast.MatchAs(pattern=None, name=None),
                guard=None,
                body=list(fallback_body),
            )
        )
    return (ast.Match(subject=inner_subject, cases=inner_cases),)


def _extract_match_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.Match,
    whole_fn_body: list[ast.stmt],
    in_loop_body: bool = False,
) -> DispatchSite | None:
    if not stmt.cases or not _is_dispatchable_match_subject(stmt.subject):
        return None
    raw_arms: list[tuple[_ExtractedMatchClassCase, ast.expr | None, tuple[ast.stmt, ...]]] = []
    explicit_default: tuple[ast.stmt, ...] | None = None
    for case_idx, case in enumerate(stmt.cases):
        match case.pattern:
            case ast.MatchAs(pattern=None, name=None) if (not match_guards(case)):
                if case_idx != len(stmt.cases) - 1:
                    return None
                explicit_default = tuple(case.body)
                continue
        extracted = _extract_match_pattern_classes(case.pattern)
        if extracted is None:
            return None
        raw_arms.append((extracted, case.guard, tuple(case.body)))
    if not raw_arms:
        return None
    # A case with a guard or nested subpattern may fail at runtime and fall through;
    # allow it only on the last class arm so failure always reaches `default_body`.
    for extracted, guard, _body in raw_arms[:-1]:
        if guard is not None or extracted.has_subpatterns:
            return None

    # For impure subjects (e.g. `match BuiltinCall.decoded(node):`), ensure no arm or
    # default body re-evaluates the subject expression directly.
    impure_subject = not _is_pure_subject(stmt.subject)
    subject_dump = ast.dump(stmt.subject) if impure_subject else ""
    if impure_subject:
        check_nodes: list[ast.AST] = [
            node
            for _ex, guard, body in raw_arms
            for node in (*body, *((guard,) if guard is not None else ()))
        ]
        if explicit_default is not None:
            check_nodes.extend(explicit_default)
        if _expr_occurs_in_nodes(subject_dump, check_nodes):
            return None

    has_last_fallthrough = raw_arms[-1][1] is not None or raw_arms[-1][0].has_subpatterns

    def finalize(
        default_body: tuple[ast.stmt, ...],
        replaced: tuple[ast.stmt, ...],
        is_whole: bool,
        *,
        strip_bare_return: bool = False,
        strip_continue: bool = False,
        include_container: bool = False,
    ) -> DispatchSite | None:
        if impure_subject and explicit_default is None and _expr_occurs_in_nodes(
            subject_dump, default_body
        ):
            return None
        arms: list[TypeTestArm] = []
        for idx, (extracted, guard, raw_body) in enumerate(raw_arms):
            case_body = raw_body
            if strip_bare_return:
                case_body = _strip_trailing_bare_return(case_body)
            elif strip_continue:
                case_body = _strip_trailing_continue(case_body)
            is_conditional = idx == len(raw_arms) - 1 and has_last_fallthrough
            built_body = _build_match_arm_body(
                stmt.subject,
                extracted,
                guard,
                case_body,
                default_body if is_conditional else (ast.Pass(),),
            )
            arms.append(
                TypeTestArm(
                    extracted.tested_exprs,
                    built_body,
                    False,
                    alias=extracted.alias,
                    embeds_default=is_conditional,
                )
            )
        test_nodes = tuple(expr for arm in arms for expr in arm.tested_exprs)
        if strip_bare_return or strip_continue:
            return StatementDispatchSite(
                subject=stmt.subject,
                kind=InstanceDispatchKind(),
                arms=tuple(arms),
                default_body=default_body,
                is_whole_function=is_whole,
                test_nodes=test_nodes,
                replaced_nodes=replaced,
            )
        synthetic_test = ParsedTypeTest(
            stmt.subject,
            arms[0].tested_exprs,
            InstanceDispatchKind(),
            False,
            False,
            stmt.subject,
        )
        return _build_statement_site(
            synthetic_test,
            tuple(arms),
            default_body,
            replaced,
            is_whole,
            test_nodes,
            container=container if include_container else None,
            index=index if include_container else None,
            whole_fn_body=whole_fn_body if include_container else None,
        )

    if explicit_default is not None:
        return finalize(
            explicit_default,
            (stmt,),
            container is whole_fn_body and len(whole_fn_body) == 1,
            include_container=True,
        )

    tail = container[index + 1 :]
    if (
        tail
        and _always_terminates(tail)
        and (
            all(_always_terminates(body) for _ex, _g, body in raw_arms)
            or any(_has_any_return(body) for _ex, _g, body in raw_arms)
        )
    ):
        return finalize(
            tuple(tail),
            tuple(container[index:]),
            container is whole_fn_body and index == 0,
            include_container=True,
        )
    if (
        container is whole_fn_body
        and tail
        and all(_always_void_terminates(body) for _ex, _g, body in raw_arms)
        and not _has_value_return(tail)
    ):
        return finalize(
            tuple(tail),
            tuple(container[index:]),
            index == 0,
            strip_bare_return=True,
        )
    if (
        in_loop_body
        and tail
        and all(_ends_with_continue_only(body) for _ex, _g, body in raw_arms)
        and not _has_non_continue_loop_escape(tail)
    ):
        return finalize(
            _rewrite_loop_continues_to_returns(tail),
            tuple(container[index:]),
            False,
            strip_continue=True,
        )
    if not any(_has_any_return(body) for _ex, _g, body in raw_arms):
        return finalize(
            (ast.Pass(),),
            (stmt,),
            container is whole_fn_body and len(whole_fn_body) == 1,
        )
    return None


def _is_pure_short_circuit_expr(expr: ast.expr) -> bool:
    """Whether `expr` is a side-effect-free literal or pure name/attribute chain safe to retain at an `IfExp` guard."""

    match expr:
        case (
            ast.Constant()
            | ast.Tuple(elts=[])
            | ast.List(elts=[])
            | ast.Set(elts=[])
            | ast.Dict(keys=[])
        ):
            return True
    return _is_pure_subject(expr)


def _extract_ternary_site(
    node: ast.IfExp,
    is_whole_return: bool,
    test_line: int | None = None,
) -> TernaryDispatchSite | None:
    parsed = _parse_type_test(node.test)
    if parsed is not None:
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
    if not isinstance(node.test, ast.BoolOp) or len(node.test.values) < 2:
        return None
    match_idx: int | None = None
    matched: ParsedTypeTest | None = None
    for idx, value in enumerate(node.test.values):
        if test_line is not None and not (
            value.lineno <= test_line <= (value.end_lineno or value.lineno)
        ):
            continue
        candidate = _parse_type_test(value)
        if candidate is not None:
            match_idx = idx
            matched = candidate
            break
    if match_idx is None or matched is None:
        return None
    prefix_values = node.test.values[:match_idx]
    suffix_values = node.test.values[match_idx + 1 :]
    suffix_expr: ast.expr | None = (
        None
        if not suffix_values
        else (
            suffix_values[0]
            if len(suffix_values) == 1
            else ast.BoolOp(op=node.test.op, values=suffix_values)
        )
    )
    if isinstance(node.test.op, ast.And):
        hit_expr: ast.expr = (
            ast.IfExp(test=suffix_expr, body=node.body, orelse=node.orelse)
            if suffix_expr is not None
            else node.body
        )
        miss_expr: ast.expr = node.orelse
        arm_body, default_body = _polarity_bodies(
            matched.negated, (ast.Return(value=hit_expr),), (ast.Return(value=miss_expr),)
        )
        if not prefix_values:
            return TernaryDispatchSite(
                subject=matched.subject,
                kind=matched.kind,
                arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
                default_body=default_body,
                is_whole_function=is_whole_return,
                test_nodes=(matched.test_node,),
                node=node,
            )
        if not _is_pure_short_circuit_expr(node.orelse):
            return None
        guard_expr: ast.expr = (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=ast.And(), values=prefix_values)
        )
        return PrefixGuardedAndTernaryDispatchSite(
            subject=matched.subject,
            kind=matched.kind,
            arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
            default_body=default_body,
            is_whole_function=False,
            test_nodes=(matched.test_node,),
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.orelse,
        )
    if isinstance(node.test.op, ast.Or):
        hit_expr = node.body
        miss_expr = (
            ast.IfExp(test=suffix_expr, body=node.body, orelse=node.orelse)
            if suffix_expr is not None
            else node.orelse
        )
        arm_body, default_body = _polarity_bodies(
            matched.negated, (ast.Return(value=hit_expr),), (ast.Return(value=miss_expr),)
        )
        if not prefix_values:
            return TernaryDispatchSite(
                subject=matched.subject,
                kind=matched.kind,
                arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
                default_body=default_body,
                is_whole_function=is_whole_return,
                test_nodes=(matched.test_node,),
                node=node,
            )
        if not _is_pure_short_circuit_expr(node.body):
            return None
        guard_expr = (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=ast.Or(), values=prefix_values)
        )
        return PrefixGuardedOrTernaryDispatchSite(
            subject=matched.subject,
            kind=matched.kind,
            arms=(TypeTestArm(matched.tested_exprs, arm_body, matched.exact),),
            default_body=default_body,
            is_whole_function=False,
            test_nodes=(matched.test_node,),
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.body,
        )
    return None


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
        if test_line is not None and not (
            value.lineno <= test_line <= (value.end_lineno or value.lineno)
        ):
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
            return _extract_match_site(
                container, index, stmt, body, in_loop_body=in_loop_body
            )
        if isinstance(stmt, ast.Return):
            match stmt:
                case ast.Return(value=ast.IfExp() as ifexp) if (
                    ifexp.test.lineno <= test_line <= (ifexp.test.end_lineno or ifexp.test.lineno)
                ):
                    is_whole = container is body and len(body) == 1
                    extracted = _extract_ternary_site(ifexp, is_whole, test_line=test_line)
                    if extracted is not None:
                        return extracted
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
    # Fallback: ternary `IfExp`, non-branching compound `BoolOp`, or bare boolean type-test
    # expression (e.g. inside `any(...)`, `all(...)`, assignments, comprehensions) on `test_line`.
    branch_tests: set[int] = set()
    excluded_bare_tests: set[int] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.If, ast.IfExp)):
            branch_tests.add(id(node.test))
            excluded_bare_tests.add(id(node.test))
            if isinstance(node.test, ast.BoolOp):
                for val in node.test.values:
                    excluded_bare_tests.add(id(val))
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and _parse_type_test(node) is not None
        ):
            excluded_bare_tests.add(id(node.operand))
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp) and (
            node.test.lineno <= test_line <= (node.test.end_lineno or node.test.lineno)
        ):
            extracted = _extract_ternary_site(
                node, is_whole_return=False, test_line=test_line
            )
            if extracted is not None:
                return extracted
    for node in ast.walk(function):
        if (
            isinstance(node, ast.BoolOp)
            and id(node) not in branch_tests
            and any(
                value.lineno <= test_line <= (value.end_lineno or value.lineno)
                for value in node.values
            )
        ):
            extracted = _extract_boolean_expr_site(
                node, is_whole_return=False, test_line=test_line
            )
            if extracted is not None:
                return extracted
    for node in ast.walk(function):
        if (
            isinstance(node, ast.expr)
            and id(node) not in excluded_bare_tests
            and node.lineno <= test_line <= (node.end_lineno or node.lineno)
            and _parse_type_test(node) is not None
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


@dataclass(frozen=True)
class OuterReceiverAlias:
    """One enclosing `self`/`cls` receiver renamed when forwarded into a pushed method."""

    source_name: str
    param_name: str


def _outer_receiver_aliases(
    function: FunctionNode,
    site: DispatchSite,
    raw_bodies: Sequence[tuple[ast.stmt, ...]],
    enclosing_locals: frozenset[str] = frozenset(),
) -> tuple[OuterReceiverAlias, ...]:
    """Fresh parameter names for outer `self`/`cls` locals referenced independently of `site.subject`."""

    receiver = site.kind.receiver
    subject_name = site.subject.id if isinstance(site.subject, ast.Name) else None
    fn_locals = _function_local_names(function) | enclosing_locals
    opposite = "cls" if receiver == "self" else "self"
    candidates = tuple(
        src
        for src in (receiver, opposite)
        if src in fn_locals and src != subject_name
    )
    if not candidates:
        return ()
    subject_dump = ast.dump(site.subject)
    referenced: set[str] = set()

    def scan(node: ast.AST) -> None:
        if isinstance(node, ast.expr) and ast.dump(node) == subject_dump:
            return
        if isinstance(node, ast.Name) and node.id in candidates:
            referenced.add(node.id)
            return
        for child in ast.iter_child_nodes(node):
            scan(child)

    for body in raw_bodies:
        for stmt in body:
            scan(stmt)
    if not referenced:
        return ()
    used_names = set(fn_locals)
    for body in raw_bodies:
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
    aliases: list[OuterReceiverAlias] = []
    for src in candidates:
        if src not in referenced:
            continue
        chosen: str | None = None
        for candidate in ("owner", "caller", "other"):
            if candidate not in used_names:
                chosen = candidate
                break
        if chosen is None:
            suffix = 1
            while f"outer_{src}_{suffix}" in used_names:
                suffix += 1
            chosen = f"outer_{src}_{suffix}"
        used_names.add(chosen)
        aliases.append(OuterReceiverAlias(source_name=src, param_name=chosen))
    return tuple(aliases)


def _rewrite_subject_to_receiver(
    statements: tuple[ast.stmt, ...],
    subject: ast.expr,
    receiver: str,
    alias: str | None = None,
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = (),
) -> tuple[ast.stmt, ...]:
    subject_dump = ast.dump(subject)
    outer_map = {item.source_name: item.param_name for item in outer_receiver_aliases}

    class _Rewriter(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if isinstance(node, ast.expr) and ast.dump(node) == subject_dump:
                return ast.Name(id=receiver, ctx=ast.Load())
            match node:
                case ast.Name(id=name, ctx=ast.Load()) if name == alias:
                    return ast.Name(id=receiver, ctx=ast.Load())
                case ast.Name(id=name, ctx=ast.Load()) if name in outer_map:
                    return ast.Name(id=outer_map[name], ctx=ast.Load())
            return super().visit(node)

    rewriter = _Rewriter()
    return tuple(cast(ast.stmt, rewriter.visit(copy.deepcopy(stmt))) for stmt in statements)


def _analyze_arm_bodies(
    function: FunctionNode,
    site: DispatchSite,
    rewritten_bodies: Sequence[tuple[ast.stmt, ...]],
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = (),
    enclosing_locals: frozenset[str] = frozenset(),
) -> tuple[tuple[CapturedArgument, ...], bool]:
    """Validate purity of rewritten bodies and return `(captured_args, is_async)`."""

    fn_locals = _function_local_names(function) | enclosing_locals
    receiver = site.kind.receiver
    alias_to_source = {
        item.param_name: item.source_name for item in outer_receiver_aliases
    }
    captured: list[CapturedArgument] = []
    seen_captured: set[str] = set()
    uses_await = False

    reassigned = site.reassigned_target
    for body in rewritten_bodies:
        local_stores: set[str] = set(_body_bound_names(body))
        if reassigned is not None:
            local_stores.discard(reassigned)
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
                        name in alias_to_source
                        and name not in local_stores
                        and name not in seen_captured
                    ):
                        seen_captured.add(name)
                        captured.append(
                            CapturedArgument(
                                call_expr=alias_to_source[name], param_name=name
                            )
                        )
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
                        case ast.AugAssign(target=ast.Name(id=name)) if (
                            name in fn_locals and name != reassigned
                        ):
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
    """Whether a consumer function is module-level, a class method, or nested in a function."""

    @abstractmethod
    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        raise NotImplementedError

    def receiver_symbol(self, snapshot: CodemodSourceSnapshot) -> str | None:
        del snapshot
        return None

    def field_annotation(self, field_name: str) -> ast.expr | None:
        del field_name
        return None

    def enclosing_local_names(self) -> frozenset[str]:
        return frozenset()

    def enclosing_function_chain(self) -> tuple[FunctionNode, ...]:
        return ()

    def enclosing_receiver_annotation(
        self, function: FunctionNode, receiver: str
    ) -> ast.expr | None:
        del function, receiver
        return None

    def enclosing_type_param_names(self) -> frozenset[str]:
        return frozenset()

    def class_receiver_symbol(self, snapshot):
        """Return enclosing class symbol for ClassEnclosure or None."""
        return None

    def dispatch_kind_for_receiver(self, subject, first_param, function, source_path, snapshot):
        """Return dispatch kind if subject is this class enclosure's receiver."""
        parsed = snapshot.parsed_module_for_source_path(source_path)
        declared = ProbeOwnerResolver.of(parsed).builder.path_of(subject).declared(snapshot.declared_receiver_typing)
        if isinstance(declared, DeclaredClassObject):
            return ClassDispatchKind()
        if isinstance(declared, DeclaredInstance):
            return InstanceDispatchKind()
        span = ReceiverSpan.of(source_path, subject)
        inferred = snapshot.type_checker.receiver_types((span,)).get(span)
        if isinstance(inferred, InferredReceiverType):
            if inferred.instances:
                return InstanceDispatchKind()
            if inferred.families:
                return ClassDispatchKind()
        raise NeedsInput(f'cannot determine whether `{ast.unparse(subject)}` at {source_path}:{subject.lineno} is a class object (`type[...]`) or an instance', NeededInput.PROOF)


@dataclass(frozen=True)
class ModuleEnclosure(FunctionEnclosure):
    """Top-level function declared directly in a module."""

    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        return module.body


def _is_staticmethod(function: FunctionNode) -> bool:
    return any(
        isinstance(dec, ast.Name) and dec.id == "staticmethod"
        for dec in function.decorator_list
    )


@dataclass(frozen=True)
class StaticMethodEnclosure(FunctionEnclosure):
    """`@staticmethod` declared directly on a scanned class (no `self`/`cls` receiver)."""

    owner: ResolvedClassTarget

    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        del module
        return self.owner.node.body

    def enclosing_type_param_names(self) -> frozenset[str]:
        return frozenset(
            tp.name for tp in getattr(self.owner.node, "type_params", ())
        )


@dataclass(frozen=True)
class ClassEnclosure(StaticMethodEnclosure):
    """Instance or class method declared directly on a scanned class."""

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

    def class_receiver_symbol(self, snapshot):
        return self.receiver_symbol(snapshot)

    def enclosing_receiver_annotation(
        self, function: FunctionNode, receiver: str
    ) -> ast.expr | None:
        positional = (*function.args.posonlyargs, *function.args.args)
        if not positional or positional[0].arg != receiver:
            return None
        owner_expr = _qualname_expr(self.owner.qualname)
        is_classmethod = any(
            isinstance(dec, ast.Name) and dec.id == "classmethod"
            for dec in function.decorator_list
        )
        if is_classmethod:
            return ast.Subscript(
                value=ast.Name(id="type", ctx=ast.Load()),
                slice=owner_expr,
                ctx=ast.Load(),
            )
        return owner_expr

    def dispatch_kind_for_receiver(self, subject, first_param, function, source_path, snapshot):
        if isinstance(subject, ast.Name) and subject.id == first_param:
            is_cm = any((isinstance(dec, ast.Name) and dec.id == 'classmethod' for dec in function.decorator_list))
            return ClassDispatchKind() if is_cm else InstanceDispatchKind()
        return super().dispatch_kind_for_receiver(subject, first_param, function, source_path, snapshot)


@dataclass(frozen=True)
class NestedFunctionEnclosure(FunctionEnclosure):
    """Function nested inside one or more enclosing functions (optionally inside a class)."""

    enclosing_functions: tuple[FunctionNode, ...]
    enclosing_class: StaticMethodEnclosure | None = None

    def body_statements(self, module: ast.Module) -> list[ast.stmt]:
        del module
        return self.enclosing_functions[-1].body

    def enclosing_local_names(self) -> frozenset[str]:
        return frozenset().union(
            *(_function_local_names(fn) for fn in self.enclosing_functions)
        )

    def enclosing_function_chain(self) -> tuple[FunctionNode, ...]:
        return self.enclosing_functions

    def enclosing_receiver_annotation(
        self, function: FunctionNode, receiver: str
    ) -> ast.expr | None:
        del function
        if self.enclosing_class is None or not self.enclosing_functions:
            return None
        return self.enclosing_class.enclosing_receiver_annotation(
            self.enclosing_functions[0], receiver
        )

    def enclosing_type_param_names(self) -> frozenset[str]:
        names = set(
            self.enclosing_class.enclosing_type_param_names()
            if self.enclosing_class is not None
            else ()
        )
        for fn in self.enclosing_functions:
            names.update(tp.name for tp in getattr(fn, "type_params", ()))
        return frozenset(names)


def _find_child_function(
    statements: list[ast.stmt],
    name: str,
    *,
    test_line: int | None = None,
    search_nested_blocks: bool = False,
) -> FunctionNode | None:
    """Locate `def name` in `statements` (optionally inside compound blocks, never crossing inner defs/classes)."""

    candidates: list[FunctionNode] = []
    if search_nested_blocks:
        for _container, _idx, stmt, _in_loop in _statement_list_children(statements):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                candidates.append(stmt)
    else:
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                candidates.append(stmt)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if test_line is not None:
        containing = [
            fn
            for fn in candidates
            if fn.lineno <= test_line <= (fn.end_lineno or fn.lineno)
        ]
        if len(containing) == 1:
            return containing[0]
    return None


def _format_receiver_call(subject: ast.expr, method_name: str, call_args: str) -> str:
    subject_source = ast.unparse(subject)
    if isinstance(subject, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return f"{subject_source}.{method_name}({call_args})"
    return f"({subject_source}).{method_name}({call_args})"


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


def _qualname_expr(qualname: str) -> ast.expr:
    parts = [part for part in qualname.split(".") if part]
    expr: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        expr = ast.Attribute(value=expr, attr=part, ctx=ast.Load())
    return expr


def _has_future_annotations(module: ast.Module) -> bool:
    for stmt in module.body:
        match stmt:
            case ast.ImportFrom(module="__future__", names=aliases):
                if any(alias.name == "annotations" for alias in aliases):
                    return True
    return False


def _is_bound_before_line(module: ast.Module, name: str, line: int) -> bool:
    """Whether `name` is unconditionally bound at top-level of `module` strictly before `line`."""

    for stmt in module.body:
        if stmt.lineno >= line:
            break
        match stmt:
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                for alias in aliases:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    if bound == name:
                        return True
            case (
                ast.ClassDef(name=decl_name)
                | ast.FunctionDef(name=decl_name)
                | ast.AsyncFunctionDef(name=decl_name)
            ) if decl_name == name:
                return True
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return True
            case ast.AnnAssign(target=ast.Name(id=target_name)) if target_name == name:
                return True
            case ast.TypeAlias(name=ast.Name(id=target_name)) if target_name == name:
                return True
    return False


def _module_typevar_names(module: ast.Module) -> frozenset[str]:
    """Top-level names bound to `TypeVar`, `ParamSpec`, or `TypeVarTuple`."""

    typevar_ctors = frozenset(("TypeVar", "ParamSpec", "TypeVarTuple"))
    names: set[str] = set()
    for stmt in module.body:
        target_name: str | None = None
        value: ast.expr | None = None
        match stmt:
            case ast.Assign(targets=[ast.Name(id=bound)], value=rhs):
                target_name = bound
                value = rhs
            case ast.AnnAssign(target=ast.Name(id=bound), value=rhs) if rhs is not None:
                target_name = bound
                value = rhs
        if target_name is None or not isinstance(value, ast.Call):
            continue
        match value.func:
            case ast.Name(id=ctor) | ast.Attribute(attr=ctor) if ctor in typevar_ctors:
                names.add(target_name)
    return frozenset(names)


def _contains_forward_ref_string(expr: ast.expr) -> bool:
    """Whether `expr` contains a string literal outside `Literal[...]`."""

    match expr:
        case ast.Constant(value=str()):
            return True
        case ast.Subscript(
            value=ast.Name(id="Literal") | ast.Attribute(attr="Literal")
        ):
            return False
    return any(
        _contains_forward_ref_string(child)
        for child in ast.iter_child_nodes(expr)
        if isinstance(child, ast.expr)
    )


def _function_has_yield(function: FunctionNode) -> bool:
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _external_import_source_for_name(
    module: ast.Module,
    name: str,
    scanned_module_names: frozenset[str],
) -> str | None:
    """Return an import statement for `name` from `module` when it targets a non-scanned module."""

    for stmt in module.body:
        match stmt:
            case ast.Import(names=aliases):
                for alias in aliases:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    if bound != name:
                        continue
                    top_mod = alias.name.split(".", 1)[0]
                    if (
                        alias.name in scanned_module_names
                        or top_mod in scanned_module_names
                    ):
                        return None
                    if alias.asname:
                        return f"import {alias.name} as {alias.asname}"
                    return f"import {alias.name}"
            case ast.ImportFrom(module=mod_name, names=aliases, level=0) if (
                mod_name is not None and mod_name != "__future__"
            ):
                for alias in aliases:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    if bound != name:
                        continue
                    top_mod = mod_name.split(".", 1)[0]
                    if (
                        mod_name in scanned_module_names
                        or top_mod in scanned_module_names
                    ):
                        return None
                    if alias.asname:
                        return f"from {mod_name} import {alias.name} as {alias.asname}"
                    return f"from {mod_name} import {alias.name}"
    return None


def _parameter_annotation_in_scope(
    function: FunctionNode,
    enclosing_functions: tuple[FunctionNode, ...],
    name: str,
) -> ast.expr | None:
    """Find the declared annotation for parameter or annotated local `name` in lexical scope."""

    for fn in (function, *reversed(enclosing_functions)):
        for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
            if arg.arg == name:
                return arg.annotation
        if any(
            (
                variadic_parameter.arg == name
                for variadic_parameter in variadic_parameters(fn.args)
            )
        ):
            return None
        ann_candidates: list[ast.expr] = []
        assigned_without_ann = False
        for _container, _idx, stmt, _in_loop in _statement_list_children(fn.body):
            match stmt:
                case ast.AnnAssign(target=ast.Name(id=target_name), annotation=ann) if (
                    target_name == name
                ):
                    ann_candidates.append(ann)
                case ast.Assign(targets=targets):
                    for target in targets:
                        if any(
                            isinstance(node, ast.Name) and node.id == name
                            for node in ast.walk(target)
                        ):
                            assigned_without_ann = True
        if ann_candidates:
            dumps = {ast.dump(ann) for ann in ann_candidates}
            return ann_candidates[0] if len(dumps) == 1 else None
        if assigned_without_ann:
            return None
    return None


@dataclass(frozen=True)
class ResolvedMethodAnnotations:
    """Sound parameter and return annotations resolved across every destination class."""

    param_annotations: tuple[tuple[str, ast.expr], ...] = ()
    return_annotation: ast.expr | None = None
    import_edits: tuple[ModuleImportMutation, ...] = ()

    @property
    def param_map(self) -> dict[str, ast.expr]:
        return dict(self.param_annotations)

    def annotation_nodes(self) -> tuple[ast.expr, ...]:
        nodes = [ann for _name, ann in self.param_annotations]
        if self.return_annotation is not None:
            nodes.append(self.return_annotation)
        return tuple(nodes)

    @classmethod
    def for_push(
        cls,
        snapshot: CodemodSourceSnapshot,
        source_module: ParsedModule,
        source_line: int,
        enclosure: FunctionEnclosure,
        function: FunctionNode,
        outer_receiver_aliases: tuple[OuterReceiverAlias, ...],
        captured_args: tuple[CapturedArgument, ...],
        destinations: Sequence[ResolvedClassTarget],
        *,
        propagates_function_return_type: bool,
        is_void_method: bool,
        constant_bool_return: bool = False,
        rationale: str,
    ) -> ResolvedMethodAnnotations:
        bindings = snapshot.module_binding_proof
        import_graph = snapshot.module_import_graph
        scanned_module_names = frozenset(import_graph.source_files_by_module_name)
        enclosing_chain = enclosure.enclosing_function_chain()
        forbidden_names = (
            frozenset(tp.name for tp in getattr(function, "type_params", ()))
            | enclosure.enclosing_type_param_names()
            | _module_typevar_names(source_module.module)
        )

        def resolve_expr(
            expr: ast.expr,
        ) -> tuple[ast.expr, dict[tuple[str, str], ModuleImportMutation]] | None:
            if _contains_forward_ref_string(expr):
                return None
            names = sorted(
                {
                    node.id
                    for node in ast.walk(expr)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
            )
            if any(name in forbidden_names for name in names):
                return None
            needed_imports: dict[tuple[str, str], ModuleImportMutation] = {}
            for name in names:
                reference = ast.Name(id=name, ctx=ast.Load())
                expected = bindings.reference_or_builtin_witness_at(
                    source_module, reference, line=source_line
                )
                if expected is None:
                    if name in builtins.__dict__:
                        continue
                    return None
                if expected.qualified_name in ("typing.Self", "typing_extensions.Self"):
                    return None
                for dest in destinations:
                    dest_module = snapshot.parsed_module_for_source_path(dest.file_path)
                    future_ann = _has_future_annotations(dest_module.module)
                    lookup_line = (
                        dest_module.module.body[-1].end_lineno
                        or dest_module.module.body[-1].lineno
                        if future_ann and dest_module.module.body
                        else dest.node.lineno
                    )
                    actual = bindings.reference_or_builtin_witness_at(
                        dest_module, reference, line=lookup_line
                    )
                    if actual is not None:
                        if actual.qualified_name != expected.qualified_name:
                            return None
                        if not future_ann and name not in builtins.__dict__:
                            if not _is_bound_before_line(
                                dest_module.module, name, dest.node.lineno
                            ):
                                return None
                        continue
                    if dest.file_path == source_module.file_path:
                        return None
                    targets = snapshot.source_index.targets_matching_repository_symbol(
                        expected.qualified_name
                    )
                    import_source: str | None = None
                    if len(targets) == 1:
                        try:
                            import_source = import_graph.required_import_source(
                                importing_file_path=dest.file_path,
                                imported_file_path=targets[0].file_path,
                                imported_name=name,
                            )
                        except ValueError:
                            return None
                    else:
                        import_source = _external_import_source_for_name(
                            source_module.module, name, scanned_module_names
                        )
                    if import_source is None:
                        return None
                    key = (dest.file_path, import_source)
                    if key not in needed_imports:
                        needed_imports[key] = ModuleImportMutation.from_source(
                            file_path=dest.file_path,
                            import_source=import_source,
                            rationale=rationale,
                        )
            return expr, needed_imports

        all_imports: dict[tuple[str, str], ModuleImportMutation] = {}
        alias_by_param = {
            item.param_name: item.source_name for item in outer_receiver_aliases
        }
        resolved_params: list[tuple[str, ast.expr]] = []
        for arg in captured_args:
            outer_source = alias_by_param.get(arg.param_name)
            if outer_source is not None:
                candidate = _parameter_annotation_in_scope(
                    function, enclosing_chain, outer_source
                )
                if candidate is None:
                    candidate = enclosure.enclosing_receiver_annotation(
                        function, outer_source
                    )
            else:
                candidate = _parameter_annotation_in_scope(
                    function, enclosing_chain, arg.call_expr
                )
            if candidate is None:
                continue
            resolved = resolve_expr(candidate)
            if resolved is None:
                continue
            ann, imports = resolved
            resolved_params.append((arg.param_name, ann))
            for key, mut in imports.items():
                all_imports.setdefault(key, mut)

        return_candidate: ast.expr | None = None
        if is_void_method:
            return_candidate = ast.Constant(value=None)
        elif (
            propagates_function_return_type
            and function.returns is not None
            and not _function_has_yield(function)
        ):
            return_candidate = function.returns
        elif constant_bool_return:
            return_candidate = ast.Name(id="bool", ctx=ast.Load())

        resolved_return: ast.expr | None = None
        if return_candidate is not None:
            resolved = resolve_expr(return_candidate)
            if resolved is not None:
                resolved_return, imports = resolved
                for key, mut in imports.items():
                    all_imports.setdefault(key, mut)

        return cls(
            param_annotations=tuple(resolved_params),
            return_annotation=resolved_return,
            import_edits=tuple(all_imports.values()),
        )


def _render_method(
    kind: DispatchKind,
    method_name: str,
    params: tuple[str, ...],
    body: tuple[ast.stmt, ...],
    *,
    is_async: bool,
    docstring: str | None = None,
    param_annotations: Mapping[str, ast.expr] = {},
    return_annotation: ast.expr | None = None,
) -> str:
    prefix = "async def" if is_async else "def"
    rendered_params = [kind.receiver]
    for name in params:
        ann = param_annotations.get(name)
        rendered_params.append(
            name if ann is None else f"{name}: {ast.unparse(ann)}"
        )
    sig_params = ", ".join(rendered_params)
    ret_suffix = (
        "" if return_annotation is None else f" -> {ast.unparse(return_annotation)}"
    )
    lines: list[str] = [
        *kind.decorators,
        f"{prefix} {method_name}({sig_params}){ret_suffix}:",
    ]
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


def _class_append_edits(
    file_path: str,
    geometry: SourceTextGeometry,
    node: ast.ClassDef,
    member_source: str,
    *,
    member_names: tuple[str, ...],
    rationale: str,
) -> tuple[PhysicalSourceEdit, ...]:
    """Emit commutative class-member insertion edits for appending `member_source` onto `node`."""

    last = node.body[-1]
    _start, end = geometry.required_node_offsets(last)
    last_end_line = geometry.line_number_for_offset(max(0, end - 1))
    logical_lines = geometry.logical_lines
    if (
        last.lineno == node.lineno
        or last_end_line < 1
        or last_end_line > len(logical_lines)
        or not logical_lines[last_end_line - 1].endswith("\n")
    ):
        start, stop, text = _class_append_span(geometry, node, member_source)
        return (
            geometry.nominal_edit(
                file_path=file_path,
                replacements=(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=stop,
                        replacement_source=text,
                    ),
                ),
                rationale=rationale,
            ),
        )

    indent = " " * last.col_offset
    body_lines = tuple(
        f"{indent}{line}\n" if line else "\n"
        for line in member_source.splitlines()
    )
    is_placeholder_pass = isinstance(last, ast.Pass) and (
        len(node.body) == 1
        or (len(node.body) == 2 and _is_docstring_stmt(node.body[0]))
    )
    if is_placeholder_pass:
        pass_line = last.lineno
        return (
            SourceSpanDeletion(
                file_path=file_path,
                span=SourceLineSpan(pass_line, pass_line),
                rationale=rationale,
            ),
            SourceInsertion(
                file_path=file_path,
                insertion_line=pass_line + 1,
                inserted_lines=body_lines,
                leading_boundary=SourceInsertionBoundary.ONE_BLANK_LINE,
                layer=SourceInsertionLayer.CLASS_MEMBER,
                declared_member_names=member_names,
                rationale=rationale,
            ),
        )
    return (
        SourceInsertion(
            file_path=file_path,
            insertion_line=last_end_line + 1,
            inserted_lines=("\n", *body_lines),
            leading_boundary=SourceInsertionBoundary.ONE_BLANK_LINE,
            layer=SourceInsertionLayer.CLASS_MEMBER,
            declared_member_names=member_names,
            rationale=rationale,
        ),
    )


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
class BaseReorderPlan:
    """One minimal conflicting class whose direct base list reorders cleanly with zero lookup change."""

    target: ResolvedClassTarget
    symbol: str
    new_base_order: tuple[int, ...]


@dataclass(frozen=True)
class PrecedenceOverridePlan:
    """One minimal conflicting class resolving MI lookup precedence via `super(<winner>, self/cls)`."""

    target: ResolvedClassTarget
    symbol: str
    winner_target: ResolvedClassTarget
    winner_symbol: str
    expected_arm: int
    import_source: str | None = None


@dataclass(frozen=True)
class _MroDispatchPlan:
    super_safe_arms: frozenset[int]
    base_reorders: tuple[BaseReorderPlan, ...] = ()
    precedence_overrides: tuple[PrecedenceOverridePlan, ...] = ()


def _has_nonempty_slots(
    snapshot: CodemodSourceSnapshot, decl: CompactIndexedClass
) -> bool:
    if "__slots__" not in decl.member_binding_names:
        return False
    if decl.file_path not in snapshot.sources_by_file_path:
        return True
    target = ResolvedClassTarget.from_rewrite_target(
        snapshot, SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname)
    )
    for stmt in target.node.body:
        value: ast.expr | None = None
        match stmt:
            case ast.Assign(targets=targets, value=rhs) if any(
                isinstance(t, ast.Name) and t.id == "__slots__" for t in targets
            ):
                value = rhs
            case ast.AnnAssign(target=ast.Name(id="__slots__"), value=rhs):
                value = rhs
        if value is None:
            continue
        if isinstance(value, (ast.Tuple, ast.List)) and not value.elts:
            continue
        return True
    return False


def _has_annotated_instance_fields(
    snapshot: CodemodSourceSnapshot, decl: CompactIndexedClass
) -> bool:
    if decl.file_path not in snapshot.sources_by_file_path:
        return True
    target = ResolvedClassTarget.from_rewrite_target(
        snapshot, SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname)
    )
    for stmt in target.node.body:
        match stmt:
            case ast.AnnAssign(target=ast.Name(), annotation=ann):
                match ann:
                    case ast.Subscript(
                        value=ast.Name(id="ClassVar") | ast.Attribute(attr="ClassVar")
                    ):
                        continue
                    case ast.Name(id="ClassVar") | ast.Attribute(attr="ClassVar"):
                        continue
                return True
    return False


def _super_precedence_statement(
    winner_name: str,
    receiver: str,
    method_name: str,
    param_names: Sequence[str],
    *,
    is_async: bool,
    is_void: bool,
) -> ast.stmt:
    call: ast.expr = ast.Call(
        func=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="super", ctx=ast.Load()),
                args=[
                    ast.Name(id=winner_name, ctx=ast.Load()),
                    ast.Name(id=receiver, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            attr=method_name,
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=name, ctx=ast.Load()) for name in param_names],
        keywords=[],
    )
    if is_async:
        call = ast.Await(value=call)
    return ast.Expr(value=call) if is_void else ast.Return(value=call)


class _AlphaRenameParameters(ast.NodeTransformer):
    """Alpha-rename free parameter references without entering nested scopes."""

    def __init__(self, rename_map: dict[str, str]) -> None:
        self.rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        mapped = self.rename_map.get(node.id)
        if mapped is None:
            return node
        return ast.copy_location(ast.Name(id=mapped, ctx=node.ctx), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


def _statements_without_docstring(body: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    if (
        len(body) > 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return tuple(body[1:])
    return tuple(body)


def _body_shadows_parameter(body: Sequence[ast.stmt], param_names: frozenset[str]) -> bool:
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.comprehension):
                for target_node in ast.walk(node.target):
                    if isinstance(target_node, ast.Name) and target_node.id in param_names:
                        return True
    return False


def _normalized_body_dumps(
    body: Sequence[ast.stmt], param_names: Sequence[str]
) -> tuple[str, ...] | None:
    stripped = _statements_without_docstring(body)
    if _body_shadows_parameter(stripped, frozenset(param_names)):
        return None
    rename_map = {name: f"__p{index}" for index, name in enumerate(param_names)}
    transformer = _AlphaRenameParameters(rename_map)
    return tuple(
        ast.dump(transformer.visit(copy.deepcopy(stmt)), include_attributes=False)
        for stmt in stripped
    )


def _free_module_bindings_match(
    snapshot: CodemodSourceSnapshot,
    source_module: ParsedModule,
    source_line: int,
    dest_module: ParsedModule,
    dest_line: int,
    expected_body: tuple[ast.stmt, ...],
    local_exclusions: frozenset[str],
) -> bool:
    """Whether every free module-level name in `expected_body` resolves identically in `dest_module`."""

    if source_module.file_path == dest_module.file_path:
        return True
    bindings = snapshot.module_binding_proof
    excluded = local_exclusions | _body_bound_names(expected_body)
    names = {
        node.id
        for stmt in expected_body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in excluded
    }
    for name in names:
        reference = ast.Name(id=name, ctx=ast.Load())
        expected = bindings.reference_or_builtin_witness_at(
            source_module, reference, line=source_line
        )
        if expected is None:
            if name in builtins.__dict__:
                actual = bindings.reference_or_builtin_witness_at(
                    dest_module, reference, line=dest_line
                )
                if actual is not None and not actual.qualified_name.startswith("builtins."):
                    return False
                continue
            return False
        actual = bindings.reference_or_builtin_witness_at(
            dest_module, reference, line=dest_line
        )
        if actual is None or actual.qualified_name != expected.qualified_name:
            return False
    return True


def _class_method_def(
    snapshot: CodemodSourceSnapshot,
    decl: CompactIndexedClass,
    method_name: str,
) -> tuple[ParsedModule, FunctionNode] | None:
    if decl.file_path not in snapshot.sources_by_file_path:
        return None
    target = ResolvedClassTarget.from_rewrite_target(
        snapshot, SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname)
    )
    matches = [
        stmt
        for stmt in target.node.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and stmt.name == method_name
    ]
    if len(matches) != 1:
        return None
    return snapshot.parsed_module_for_source_path(decl.file_path), matches[0]


def _matching_existing_dispatch_methods(
    snapshot: CodemodSourceSnapshot,
    source_module: ParsedModule,
    test_line: int,
    subject_family: frozenset[str],
    base: ResolvedClassTarget,
    base_symbol: str,
    checked_arms: tuple[TypeTestArm, ...],
    checked_arm_symbols: list[tuple[str, ...]],
    kind: DispatchKind,
    rewritten_default: tuple[ast.stmt, ...],
    rewritten_arm_bodies: Sequence[tuple[ast.stmt, ...]],
    captured_args: tuple[CapturedArgument, ...],
    is_async: bool,
) -> tuple[str, ...]:
    """Find non-dunder methods already declared on `base` whose family-wide C3 dispatch
    matches the exact generated bodies for every member of `subject_family`."""

    receiver = kind.receiver
    param_names = tuple(arg.param_name for arg in captured_args)
    generated_params = (receiver, *param_names)
    local_exclusions = frozenset(generated_params)
    default_dumps = _normalized_body_dumps(rewritten_default, generated_params)
    if default_dumps is None:
        return ()

    arm_full_bodies: list[tuple[ast.stmt, ...]] = []
    arm_full_dumps: list[tuple[str, ...] | None] = []
    for arm, body in zip(checked_arms, rewritten_arm_bodies, strict=True):
        full_body = (*body, *rewritten_default) if arm.falls_through_to_default else body
        arm_full_bodies.append(full_body)
        arm_full_dumps.append(_normalized_body_dumps(full_body, generated_params))
    if any(dumps is None for dumps in arm_full_dumps):
        return ()

    candidate_names: list[str] = []
    for stmt in base.node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if stmt.name.startswith("__") and stmt.name.endswith("__"):
            continue
        if isinstance(stmt, ast.AsyncFunctionDef) != is_async:
            continue
        if not kind.matches_method_decorators(stmt.decorator_list):
            continue
        args = stmt.args
        if (
            args.posonlyargs
            or args.kwonlyargs
            or variadic_parameters(args)
            or args.defaults
            or len(args.args) != len(generated_params)
        ):
            continue
        candidate_names.append(stmt.name)
    if not candidate_names:
        return ()

    compact_index = snapshot.compact_class_index
    authority = compact_index.declared_mro_authority
    family_mros: list[tuple[str, tuple[CompactIndexedClass, ...], int | None]] = []
    for symbol in sorted(subject_family):
        mro = authority.resolve(symbol).mro_type
        if mro is None:
            return ()
        order = tuple(decl.symbol for decl in mro.declarations)
        expected_arm: int | None = None
        for arm_idx, (arm, arm_syms) in enumerate(
            zip(checked_arms, checked_arm_symbols, strict=True)
        ):
            matched = (
                symbol in arm_syms
                if arm.exact
                else any(tested in order for tested in arm_syms)
            )
            if matched:
                expected_arm = arm_idx
                break
        family_mros.append((symbol, mro.declarations, expected_arm))

    method_cache: dict[tuple[str, str], tuple[ParsedModule, FunctionNode] | None] = {}
    matched_names: list[str] = []
    for cand in candidate_names:
        ok = True
        for _symbol, declarations, expected_arm in family_mros:
            winner_idx: int | None = None
            for idx, decl in enumerate(declarations):
                if cand in decl.member_binding_names:
                    winner_idx = idx
                    break
            if winner_idx is None:
                ok = False
                break
            winner_decl = declarations[winner_idx]
            cache_key = (winner_decl.symbol, cand)
            if cache_key not in method_cache:
                method_cache[cache_key] = _class_method_def(snapshot, winner_decl, cand)
            located = method_cache[cache_key]
            if located is None:
                ok = False
                break
            dest_module, existing_fn = located
            if isinstance(existing_fn, ast.AsyncFunctionDef) != is_async:
                ok = False
                break
            if not kind.matches_method_decorators(existing_fn.decorator_list):
                ok = False
                break
            args = existing_fn.args
            if (
                args.posonlyargs
                or args.kwonlyargs
                or variadic_parameters(args)
                or args.defaults
                or len(args.args) != len(generated_params)
            ):
                ok = False
                break
            existing_dumps = _normalized_body_dumps(
                existing_fn.body, tuple(arg.arg for arg in args.args)
            )
            if existing_dumps is None:
                ok = False
                break
            if expected_arm is None:
                expected_body = rewritten_default
                expected_options = (default_dumps,)
            else:
                expected_body = arm_full_bodies[expected_arm]
                arm_dump = arm_full_dumps[expected_arm]
                assert arm_dump is not None
                if checked_arms[expected_arm].falls_through_to_default:
                    next_decl_sym: str | None = None
                    for later in declarations[winner_idx + 1 :]:
                        if cand in later.member_binding_names:
                            next_decl_sym = later.symbol
                            break
                    if next_decl_sym == base_symbol:
                        super_body = (
                            *rewritten_arm_bodies[expected_arm],
                            _super_fallback_return(cand, param_names, is_async=is_async),
                        )
                        super_dump = _normalized_body_dumps(super_body, generated_params)
                        expected_options = (
                            (arm_dump, super_dump)
                            if super_dump is not None
                            else (arm_dump,)
                        )
                    else:
                        expected_options = (arm_dump,)
                else:
                    expected_options = (arm_dump,)
            if existing_dumps not in expected_options:
                ok = False
                break
            if not _free_module_bindings_match(
                snapshot,
                source_module,
                test_line,
                dest_module,
                existing_fn.lineno,
                expected_body,
                local_exclusions,
            ):
                ok = False
                break
        if ok:
            matched_names.append(cand)
    return tuple(matched_names)


@dataclass(frozen=True)
class DispatchMethodTarget(ABC):
    """Whether a type-test push adds a new polymorphic method family or reuses an exact existing one."""

    method_name: str

    @property
    def docstring(self) -> str | None:
        return None

    @abstractmethod
    def member_and_import_edits(
        self,
        proof: TypeTestDispatchProof,
        snapshot: CodemodSourceSnapshot,
        geometry: object,
        param_names: tuple[str, ...],
        rationale: str,
    ) -> tuple[tuple[NominalSourceEdit, ...], frozenset[str]]:
        """Return `(extra_edits, retained_names_in_source)` beyond the consumer call replacement."""
        raise NotImplementedError

    @abstractmethod
    def rationale(self, proof: TypeTestDispatchProof) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ExistingDispatchMethod(DispatchMethodTarget):
    """Reuse an already-declared polymorphic method family whose C3 dispatch matches identically."""

    def member_and_import_edits(
        self,
        proof: TypeTestDispatchProof,
        snapshot: CodemodSourceSnapshot,
        geometry: object,
        param_names: tuple[str, ...],
        rationale: str,
    ) -> tuple[tuple[NominalSourceEdit, ...], frozenset[str]]:
        del proof, snapshot, geometry, param_names, rationale
        return (), frozenset()

    def rationale(self, proof: TypeTestDispatchProof) -> str:
        return (
            f"Reuses existing `{proof.base.qualname}.{self.method_name}` for the "
            f"type test in `{proof.function.name}`."
        )


@dataclass(frozen=True)
class NewDispatchMethod(DispatchMethodTarget):
    """Append a newly factored polymorphic method onto the base and tested classes."""

    _docstring: str | None = None

    @property
    def docstring(self) -> str | None:
        return self._docstring

    def member_and_import_edits(
        self,
        proof: TypeTestDispatchProof,
        snapshot: CodemodSourceSnapshot,
        geometry: object,
        param_names: tuple[str, ...],
        rationale: str,
    ) -> tuple[tuple[NominalSourceEdit, ...], frozenset[str]]:
        parsed_source = snapshot.parsed_module_for_source_path(proof.source_path)
        all_destinations = (
            proof.base,
            *(target_cls for arm in proof.arms for target_cls in arm.targets),
            *(po.target for po in proof.precedence_overrides),
        )
        method_annotations = ResolvedMethodAnnotations.for_push(
            snapshot,
            parsed_source,
            proof.test_line,
            proof.enclosure,
            proof.function,
            proof.outer_receiver_aliases,
            proof.captured_args,
            all_destinations,
            propagates_function_return_type=proof.site.propagates_function_return_type,
            is_void_method=proof.site.is_void_method,
            constant_bool_return=proof.site.returns_constant_bool,
            rationale=rationale,
        )
        param_annotations = method_annotations.param_map
        base_method = _render_method(
            proof.site.kind,
            self.method_name,
            param_names,
            proof.default_body,
            is_async=proof.is_async,
            docstring=self._docstring,
            param_annotations=param_annotations,
            return_annotation=method_annotations.return_annotation,
        )
        member_edits: list[NominalSourceEdit] = [
            *_class_append_edits(
                proof.base.file_path,
                geometry(proof.base.file_path),  # type: ignore[operator]
                proof.base.node,
                base_method,
                member_names=(self.method_name,),
                rationale=rationale,
            )
        ]
        for resolved_arm in proof.arms:
            arm_method = _render_method(
                proof.site.kind,
                self.method_name,
                param_names,
                resolved_arm.body,
                is_async=proof.is_async,
                param_annotations=param_annotations,
                return_annotation=method_annotations.return_annotation,
            )
            for target_cls in resolved_arm.targets:
                member_edits.extend(
                    _class_append_edits(
                        target_cls.file_path,
                        geometry(target_cls.file_path),  # type: ignore[operator]
                        target_cls.node,
                        arm_method,
                        member_names=(self.method_name,),
                        rationale=rationale,
                    )
                )
        for reorder in proof.base_reorders:
            header = ClassHeaderSpanSourceAuthority(
                node=reorder.target.node,
                source=snapshot.sources_by_file_path[reorder.target.file_path],
            )
            reordered_items = tuple(
                header.base_items[i] for i in reorder.new_base_order
            )
            member_edits.extend(
                header.source_edits(
                    header.with_base_items(reordered_items),
                    file_path=reorder.target.file_path,
                    rationale=rationale,
                )
            )
        precedence_imports: list[NominalSourceEdit] = []
        for po in proof.precedence_overrides:
            po_stmt = _super_precedence_statement(
                po.winner_target.qualname,
                proof.site.kind.receiver,
                self.method_name,
                param_names,
                is_async=proof.is_async,
                is_void=proof.site.is_void_method,
            )
            po_method = _render_method(
                proof.site.kind,
                self.method_name,
                param_names,
                (po_stmt,),
                is_async=proof.is_async,
                param_annotations=param_annotations,
                return_annotation=method_annotations.return_annotation,
            )
            member_edits.extend(
                _class_append_edits(
                    po.target.file_path,
                    geometry(po.target.file_path),  # type: ignore[operator]
                    po.target.node,
                    po_method,
                    member_names=(self.method_name,),
                    rationale=rationale,
                )
            )
            if po.import_source is not None:
                precedence_imports.append(
                    ModuleImportMutation.from_source(
                        file_path=po.target.file_path,
                        import_source=po.import_source,
                        rationale=rationale,
                    )
                )

        local_exclusions = frozenset((proof.site.kind.receiver, *param_names))
        extra_edits: list[NominalSourceEdit] = [
            *member_edits,
            *precedence_imports,
            *method_annotations.import_edits,
            *_cross_module_imports_for_body(
                snapshot,
                parsed_source,
                proof.test_line,
                proof.base,
                proof.default_body,
                local_exclusions,
                rationale,
            ),
        ]
        for resolved_arm in proof.arms:
            for target_cls in resolved_arm.targets:
                extra_edits.extend(
                    _cross_module_imports_for_body(
                        snapshot,
                        parsed_source,
                        proof.test_line,
                        target_cls,
                        resolved_arm.body,
                        local_exclusions,
                        rationale,
                    )
                )
        retained_in_source: set[str] = set()
        if any(dest.file_path == proof.source_path for dest in all_destinations):
            retained_in_source.update(
                _names_loaded_in(method_annotations.annotation_nodes())
            )
        for po in proof.precedence_overrides:
            if po.target.file_path == proof.source_path:
                retained_in_source.add(po.winner_target.qualname)
        if proof.base.file_path == proof.source_path:
            retained_in_source.update(_names_loaded_in(proof.default_body))
        for resolved_arm in proof.arms:
            if any(
                target_cls.file_path == proof.source_path
                for target_cls in resolved_arm.targets
            ):
                retained_in_source.update(_names_loaded_in(resolved_arm.body))
        return tuple(extra_edits), frozenset(retained_in_source)

    def rationale(self, proof: TypeTestDispatchProof) -> str:
        targets = ", ".join(
            f"`{cls.qualname}`"
            for arm in proof.arms
            for cls in arm.targets
        )
        if not targets:
            return (
                f"Pushes the type test in `{proof.function.name}` onto "
                f"`{proof.base.qualname}.{self.method_name}`."
            )
        return (
            f"Pushes the type test in `{proof.function.name}` onto "
            f"`{proof.base.qualname}.{self.method_name}` with overrides on {targets}."
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
    method_target: DispatchMethodTarget
    matching_existing_methods: tuple[str, ...] = ()
    enclosure: FunctionEnclosure = ModuleEnclosure()
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = ()
    base_reorders: tuple[BaseReorderPlan, ...] = ()
    precedence_overrides: tuple[PrecedenceOverridePlan, ...] = ()

    @property
    def method_name(self) -> str:
        return self.method_target.method_name

    @property
    def docstring(self) -> str | None:
        return self.method_target.docstring

    @classmethod
    def locate_function(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        *,
        test_line: int | None = None,
    ) -> tuple[str, FunctionEnclosure, FunctionNode]:
        source_path = target.file_path
        if not source_path or source_path not in snapshot.module_nodes_by_file_path:
            raise NeedsInput("target module is not in the source snapshot", NeededInput.SOURCE_CHANGE)
        qualname = target.qualname or ""
        parts = [part for part in qualname.split(".") if part and part != "<locals>"]
        if not parts:
            raise NeedsInput(
                f"`{qualname}` is not a top-level function or method in `{source_path}`",
                NeededInput.SOURCE_CHANGE,
            )
        function_name = parts[-1]
        owner_parts = parts[:-1]
        module = snapshot.module_nodes_by_file_path[source_path]

        enclosing_class: ClassEnclosure | None = None
        enclosing_functions: list[FunctionNode] = []
        current_body: list[ast.stmt] = module.body
        class_prefix: list[str] = []

        for part in owner_parts:
            if not enclosing_functions:
                candidate_class_name = ".".join((*class_prefix, part))
                try:
                    owner = ResolvedClassTarget.from_rewrite_target(
                        snapshot,
                        SourceRewriteTarget(
                            file_path=source_path, qualname=candidate_class_name
                        ),
                    )
                except ValueError:
                    owner = None
                if owner is not None:
                    class_prefix.append(part)
                    enclosing_class = ClassEnclosure(owner)
                    current_body = owner.node.body
                    continue
            outer_fn = _find_child_function(
                current_body,
                part,
                test_line=test_line,
                search_nested_blocks=bool(enclosing_functions),
            )
            if outer_fn is None:
                raise NeedsInput(
                    f"`{qualname}` is nested inside non-class `{'.'.join(owner_parts)}`",
                    NeededInput.SOURCE_CHANGE,
                )
            enclosing_functions.append(outer_fn)
            current_body = outer_fn.body

        function = _find_child_function(
            current_body,
            function_name,
            test_line=test_line,
            search_nested_blocks=bool(enclosing_functions),
        )
        if function is None:
            raise NeedsInput(
                f"`{qualname}` is not a top-level function or method in `{source_path}`",
                NeededInput.SOURCE_CHANGE,
            )
        if enclosing_functions:
            nested_class: StaticMethodEnclosure | None = (
                StaticMethodEnclosure(enclosing_class.owner)
                if enclosing_class is not None and _is_staticmethod(enclosing_functions[0])
                else enclosing_class
            )
            enclosure: FunctionEnclosure = NestedFunctionEnclosure(
                enclosing_functions=tuple(enclosing_functions),
                enclosing_class=nested_class,
            )
        elif enclosing_class is not None:
            enclosure = (
                StaticMethodEnclosure(enclosing_class.owner)
                if _is_staticmethod(function)
                else enclosing_class
            )
        else:
            enclosure = ModuleEnclosure()
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
        *,
        required_tested_symbols: Sequence[str] = (),
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
        source_path, enclosure, function = cls.locate_function(
            snapshot, target, test_line=test_line
        )
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

        if required_tested_symbols and not frozenset(required_tested_symbols).issubset(
            all_tested_symbols
        ):
            raise NeedsInput(
                f"type test at {source_path}:{test_line} is not a pushable dispatch shape",
                NeededInput.SOURCE_CHANGE,
            )

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

        # When the sole non-exact arm tests `base_symbol` itself AND already embeds
        # `default_body` (e.g. a conditional `match` arm with `case _: <default>`),
        # every member of `subject_family` enters that arm; push it directly onto `base`.
        folds_onto_base = (
            len(site.arms) == 1
            and resolved_arm_symbols == [(base_symbol,)]
            and not site.arms[0].exact
            and site.arms[0].embeds_default
        )
        checked_arms = () if folds_onto_base else site.arms
        checked_arm_symbols: list[tuple[str, ...]] = (
            [] if folds_onto_base else resolved_arm_symbols
        )

        receiver = site.kind.receiver
        enclosing_locals = enclosure.enclosing_local_names()
        raw_bodies: tuple[tuple[ast.stmt, ...], ...] = (
            (site.arms[0].body,)
            if folds_onto_base
            else (*(arm.body for arm in site.arms), site.default_body)
        )
        outer_receiver_aliases = _outer_receiver_aliases(
            function, site, raw_bodies, enclosing_locals=enclosing_locals
        )
        rewritten_arms: list[ResolvedTargetArm] = []
        if folds_onto_base:
            rewritten_default = _rewrite_subject_to_receiver(
                site.arms[0].body,
                site.subject,
                receiver,
                alias=site.arms[0].alias,
                outer_receiver_aliases=outer_receiver_aliases,
            )
        else:
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
                            outer_receiver_aliases=outer_receiver_aliases,
                        ),
                        exact=arm.exact,
                    )
                )
            rewritten_default = _rewrite_subject_to_receiver(
                site.default_body,
                site.subject,
                receiver,
                outer_receiver_aliases=outer_receiver_aliases,
            )

        captured_args, is_async = _analyze_arm_bodies(
            function,
            site,
            (*(arm.body for arm in rewritten_arms), rewritten_default),
            outer_receiver_aliases=outer_receiver_aliases,
            enclosing_locals=enclosing_locals,
        )

        matching_existing = _matching_existing_dispatch_methods(
            snapshot,
            parsed_module,
            test_line,
            subject_family,
            base,
            base_symbol,
            checked_arms,
            checked_arm_symbols,
            site.kind,
            rewritten_default,
            tuple(arm.body for arm in rewritten_arms),
            captured_args,
            is_async,
        )
        if method_name in matching_existing:
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
                method_target=ExistingDispatchMethod(method_name=method_name),
                matching_existing_methods=matching_existing,
                enclosure=enclosure,
                outer_receiver_aliases=outer_receiver_aliases,
            )

        mro_plan = cls._require_dispatch_mro_safe(
            snapshot,
            subject_family,
            base,
            base_symbol,
            checked_arms,
            checked_arm_symbols,
            method_name,
        )
        super_safe_arms = mro_plan.super_safe_arms

        if not folds_onto_base:
            param_names = tuple(arg.param_name for arg in captured_args)
            finalized_arms: list[ResolvedTargetArm] = []
            for arm_idx, (arm, resolved_arm) in enumerate(
                zip(site.arms, rewritten_arms, strict=True)
            ):
                if not arm.falls_through_to_default:
                    finalized_arms.append(resolved_arm)
                    continue
                if arm_idx in super_safe_arms:
                    fallback_stmts: tuple[ast.stmt, ...] = (
                        _super_fallback_return(
                            method_name, param_names, is_async=is_async
                        ),
                    )
                else:
                    fallback_stmts = rewritten_default
                finalized_arms.append(
                    ResolvedTargetArm(
                        targets=resolved_arm.targets,
                        symbols=resolved_arm.symbols,
                        body=(*resolved_arm.body, *fallback_stmts),
                        exact=resolved_arm.exact,
                    )
                )
            rewritten_arms = finalized_arms

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
            method_target=NewDispatchMethod(
                method_name=method_name,
                _docstring=docstring.strip() if docstring else None,
            ),
            matching_existing_methods=matching_existing,
            enclosure=enclosure,
            outer_receiver_aliases=outer_receiver_aliases,
            base_reorders=mro_plan.base_reorders,
            precedence_overrides=mro_plan.precedence_overrides,
        )

    @staticmethod
    def _reordered_base_permutation(
        snapshot: CodemodSourceSnapshot,
        authority: ClassMroAuthority,
        compact_classes: Mapping[str, CompactIndexedClass],
        decl: CompactIndexedClass,
        winner_symbol: str,
        arm_symbols: tuple[str, ...],
    ) -> tuple[ResolvedClassTarget, tuple[int, ...]] | None:
        if (
            not decl.base_references_are_complete
            or not decl.class_header_is_reconstructible
            or not authority.bases.creation_is_static(decl)
            or decl.file_path not in snapshot.sources_by_file_path
        ):
            return None
        target_cls = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(file_path=decl.file_path, qualname=decl.qualname),
        )
        if len(target_cls.node.bases) != len(decl.base_references):
            return None
        header = ClassHeaderSpanSourceAuthority(
            node=target_cls.node,
            source=snapshot.sources_by_file_path[target_cls.file_path],
        )
        if not header.can_rewrite:
            return None
        win_idx: int | None = None
        arm_idx: int | None = None
        for idx, ref in enumerate(decl.base_references):
            b_sym = ref.qualified_name
            if b_sym not in compact_classes:
                continue
            b_mro = authority.resolve(b_sym).mro_type
            if b_mro is None:
                return None
            b_order = frozenset(d.symbol for d in b_mro.declarations)
            if win_idx is None and winner_symbol in b_order:
                win_idx = idx
            if arm_idx is None and any(sym in b_order for sym in arm_symbols):
                arm_idx = idx
        if win_idx is None or arm_idx is None or arm_idx <= win_idx:
            return None
        perm = (
            *range(0, win_idx),
            arm_idx,
            *range(win_idx, arm_idx),
            *range(arm_idx + 1, len(decl.base_references)),
        )
        return target_cls, perm

    @staticmethod
    def _verify_reorder_lookup_neutral(
        snapshot: CodemodSourceSnapshot,
        old_authority: ClassMroAuthority,
        new_authority: ClassMroAuthority,
        compact_classes: Mapping[str, CompactIndexedClass],
        affected_symbols: Sequence[str],
    ) -> bool:
        nonempty_slots_cache: dict[str, bool] = {}
        annotated_fields_cache: dict[str, bool] = {}

        def nonempty_slots(decl: CompactIndexedClass) -> bool:
            cached = nonempty_slots_cache.get(decl.symbol)
            if cached is None:
                cached = _has_nonempty_slots(snapshot, decl)
                nonempty_slots_cache[decl.symbol] = cached
            return cached

        def annotated_fields(decl: CompactIndexedClass) -> bool:
            cached = annotated_fields_cache.get(decl.symbol)
            if cached is None:
                cached = _has_annotated_instance_fields(snapshot, decl)
                annotated_fields_cache[decl.symbol] = cached
            return cached

        for sym in affected_symbols:
            old_mro = old_authority.resolve(sym).mro_type
            new_mro = new_authority.resolve(sym).mro_type
            if old_mro is None or new_mro is None:
                return False
            old_native = tuple(
                entry
                for entry in old_mro.__mro__
                if not isinstance(entry, DeclarationMroType)
            )
            new_native = tuple(
                entry
                for entry in new_mro.__mro__
                if not isinstance(entry, DeclarationMroType)
            )
            if old_native != new_native:
                return False
            old_decls = old_mro.declarations
            new_decls = new_mro.declarations
            all_names: set[str] = set()
            for decl in old_decls:
                all_names.update(decl.member_binding_names)
            for name in all_names:
                if name == "__slots__":
                    old_seq = tuple(
                        d.symbol for d in old_decls if nonempty_slots(d)
                    )
                    new_seq = tuple(
                        d.symbol for d in new_decls if nonempty_slots(d)
                    )
                else:
                    old_seq = tuple(
                        d.symbol for d in old_decls if name in d.member_binding_names
                    )
                    new_seq = tuple(
                        d.symbol for d in new_decls if name in d.member_binding_names
                    )
                if old_seq != new_seq:
                    return False
            decl_obj = compact_classes.get(sym)
            if (
                decl_obj is not None
                and decl_obj.dataclass_declaration.generates_construction
            ):
                old_dc = tuple(d.symbol for d in old_decls if annotated_fields(d))
                new_dc = tuple(d.symbol for d in new_decls if annotated_fields(d))
                if old_dc != new_dc:
                    return False
        return True

    @classmethod
    def _require_dispatch_mro_safe(
        cls,
        snapshot: CodemodSourceSnapshot,
        subject_family: frozenset[str],
        base: ResolvedClassTarget,
        base_symbol: str,
        arms: tuple[TypeTestArm, ...],
        resolved_arm_symbols: list[tuple[str, ...]],
        method_name: str,
    ) -> _MroDispatchPlan:
        classes_by_symbol = snapshot.required_class_family_index.classes_by_symbol
        owner_arm: dict[str, int | None] = {base_symbol: None}
        arm_targets: list[tuple[str, ResolvedClassTarget]] = []
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
                arm_targets.append((sym, target_cls))
                owner_arm[sym] = arm_idx

        compact_index = snapshot.compact_class_index
        compact_classes = compact_index.classes_by_symbol
        authority = compact_index.declared_mro_authority

        def expected_arm_for(symbol: str, order: tuple[str, ...]) -> int | None:
            for arm_idx, (arm, arm_syms) in enumerate(zip(arms, resolved_arm_symbols)):
                matched = (
                    symbol in arm_syms
                    if arm.exact
                    else any(tested in order for tested in arm_syms)
                )
                if matched:
                    return arm_idx
            return None

        def evaluate_family(
            active_authority: ClassMroAuthority,
        ) -> tuple[
            list[tuple[str, tuple[CompactIndexedClass, ...], str, int | None]],
            dict[str, tuple[str, int | None, tuple[CompactIndexedClass, ...]]],
        ]:
            evaluated: list[
                tuple[str, tuple[CompactIndexedClass, ...], str, int | None]
            ] = []
            mismatches: dict[
                str, tuple[str, int | None, tuple[CompactIndexedClass, ...]]
            ] = {}
            for symbol in sorted(subject_family):
                mro = active_authority.resolve(symbol).mro_type
                if mro is None:
                    raise NeedsInput(
                        f"`{symbol}` has an open MRO, so dispatch of `{method_name}` cannot be proved",
                        NeededInput.PROOF,
                    )
                order = tuple(decl.symbol for decl in mro.declarations)
                expected_arm = expected_arm_for(symbol, order)
                winner_symbol: str | None = None
                for decl in mro.declarations:
                    if decl.symbol in owner_arm:
                        winner_symbol = decl.symbol
                        break
                if winner_symbol is None:
                    raise NeedsInput(
                        f"`{symbol}` reaches neither `{base.qualname}` nor any tested class",
                        NeededInput.PROOF,
                    )
                if owner_arm[winner_symbol] != expected_arm:
                    mismatches[symbol] = (winner_symbol, expected_arm, mro.declarations)
                evaluated.append((symbol, mro.declarations, winner_symbol, expected_arm))
            return evaluated, mismatches

        def raise_branch_order_mismatch(symbol: str, winner_symbol: str) -> None:
            raise NeedsInput(
                f"`{symbol}` would dispatch `{method_name}` via `{winner_symbol}`, "
                "which disagrees with the original type-test branch order",
                NeededInput.PROOF,
            )

        # Check name-independent MRO dispatch soundness (PROOF) before name-dependent
        # binding collisions (OWNER_PICK), so unfixable MRO/order mismatches never ask
        # the owner to pick a method name that still fails dispatch proof.
        evaluated, conflicts = evaluate_family(authority)
        base_reorders: list[BaseReorderPlan] = []
        precedence_overrides: list[PrecedenceOverridePlan] = []
        active_authority = authority

        if conflicts:
            # Only resolve genuine MI branch-order conflicts where a non-exact arm class
            # sits behind an incomparable winning class in C3 MRO.
            for sym, (winner_symbol, expected_arm, _decls) in conflicts.items():
                if (
                    expected_arm is None
                    or arms[expected_arm].exact
                    or winner_symbol == sym
                ):
                    raise_branch_order_mismatch(sym, winner_symbol)
                arm_syms = resolved_arm_symbols[expected_arm]
                winner_ancestors = compact_index.ancestor_symbols(winner_symbol)
                if any(t in winner_ancestors for t in arm_syms):
                    raise_branch_order_mismatch(sym, winner_symbol)

            conflict_expected = {
                sym: exp for sym, (_win, exp, _decls) in conflicts.items()
            }
            minimal_conflicts = [
                sym
                for sym in sorted(conflicts)
                if not any(
                    conflict_expected.get(d.symbol) == conflict_expected[sym]
                    for d in conflicts[sym][2][1:]
                )
            ]

            # Tier 1: lookup-neutral direct base reordering on minimal conflicting classes.
            accepted_patches: dict[str, tuple[BaseReorderPlan, CompactIndexedClass]] = {}
            for sym in minimal_conflicts:
                winner_symbol, expected_arm, _decls = conflicts[sym]
                assert expected_arm is not None
                decl = compact_classes.get(sym)
                if decl is None:
                    continue
                candidate = cls._reordered_base_permutation(
                    snapshot,
                    authority,
                    compact_classes,
                    decl,
                    winner_symbol,
                    resolved_arm_symbols[expected_arm],
                )
                if candidate is None:
                    continue
                target_cls, perm = candidate
                new_refs = tuple(decl.base_references[i] for i in perm)
                new_resolved_syms = tuple(
                    ref.qualified_name
                    for ref in new_refs
                    if ref.qualified_name in compact_classes
                )
                patched_decl = replace(
                    decl,
                    base_references=new_refs,
                    resolved_base_symbols=new_resolved_syms,
                )
                trial_authority = ClassMroAuthority(
                    {**compact_classes, sym: patched_decl},
                    bases=authority.bases,
                )
                affected = (sym, *compact_index.descendant_symbols(sym))
                if cls._verify_reorder_lookup_neutral(
                    snapshot, authority, trial_authority, compact_classes, affected
                ):
                    accepted_patches[sym] = (
                        BaseReorderPlan(
                            target=target_cls, symbol=sym, new_base_order=perm
                        ),
                        patched_decl,
                    )

            if accepted_patches:
                joint_classes = {
                    **compact_classes,
                    **{sym: pdecl for sym, (_plan, pdecl) in accepted_patches.items()},
                }
                joint_authority = ClassMroAuthority(joint_classes, bases=authority.bases)
                all_affected = sorted(
                    frozenset().union(
                        *(
                            (sym, *compact_index.descendant_symbols(sym))
                            for sym in accepted_patches
                        )
                    )
                )
                if cls._verify_reorder_lookup_neutral(
                    snapshot, authority, joint_authority, compact_classes, all_affected
                ):
                    base_reorders = [
                        plan for _sym, (plan, _pdecl) in sorted(accepted_patches.items())
                    ]
                    active_authority = joint_authority
                    evaluated, conflicts = evaluate_family(active_authority)

            # Tier 2: C3-proved `super(<winner>, self/cls)` precedence binding on remaining
            # minimal conflicting classes where existing member overlaps prevent reordering.
            if conflicts:
                rem_expected = {
                    sym: exp for sym, (_win, exp, _decls) in conflicts.items()
                }
                tier2_minimal = [
                    sym
                    for sym in sorted(conflicts)
                    if not any(
                        rem_expected.get(d.symbol) == rem_expected[sym]
                        for d in conflicts[sym][2][1:]
                    )
                ]
                tier2_set = frozenset(tier2_minimal)
                bindings = snapshot.module_binding_proof
                import_graph = snapshot.module_import_graph
                tier2_plans: dict[str, PrecedenceOverridePlan] = {}

                for sym in tier2_minimal:
                    winner_symbol, expected_arm, decls = conflicts[sym]
                    if expected_arm is None or arms[expected_arm].exact:
                        raise_branch_order_mismatch(sym, winner_symbol)
                    decl = compact_classes.get(sym)
                    winner_decl = compact_classes.get(winner_symbol)
                    if (
                        decl is None
                        or winner_decl is None
                        or "." in winner_decl.qualname
                        or decl.file_path not in snapshot.sources_by_file_path
                        or winner_decl.file_path not in snapshot.sources_by_file_path
                    ):
                        raise_branch_order_mismatch(sym, winner_symbol)
                    target_cls = ResolvedClassTarget.from_rewrite_target(
                        snapshot,
                        SourceRewriteTarget(
                            file_path=decl.file_path, qualname=decl.qualname
                        ),
                    )
                    winner_target = ResolvedClassTarget.from_rewrite_target(
                        snapshot,
                        SourceRewriteTarget(
                            file_path=winner_decl.file_path,
                            qualname=winner_decl.qualname,
                        ),
                    )
                    dest_module = snapshot.parsed_module_for_source_path(
                        target_cls.file_path
                    )
                    actual = bindings.reference_or_builtin_witness_at(
                        dest_module,
                        ast.Name(id=winner_decl.qualname, ctx=ast.Load()),
                        line=target_cls.node.lineno,
                    )
                    import_source: str | None = None
                    if actual is not None:
                        if actual.qualified_name != winner_symbol:
                            raise_branch_order_mismatch(sym, winner_symbol)
                    elif winner_target.file_path == target_cls.file_path:
                        raise_branch_order_mismatch(sym, winner_symbol)
                    else:
                        try:
                            import_source = import_graph.required_import_source(
                                importing_file_path=target_cls.file_path,
                                imported_file_path=winner_target.file_path,
                                imported_name=winner_decl.qualname,
                            )
                        except ValueError:
                            raise_branch_order_mismatch(sym, winner_symbol)

                    tier2_plans[sym] = PrecedenceOverridePlan(
                        target=target_cls,
                        symbol=sym,
                        winner_target=winner_target,
                        winner_symbol=winner_symbol,
                        expected_arm=expected_arm,
                        import_source=import_source,
                    )

                # Prove every member of `subject_family` dispatches to its exact expected arm
                # once `tier2_plans` are installed, and record effective executing owners.
                updated_evaluated: list[
                    tuple[str, tuple[CompactIndexedClass, ...], str, int | None]
                ] = []
                for symbol, decls, winner_symbol, expected_arm in evaluated:
                    first_hook: str | None = None
                    for decl in decls:
                        if decl.symbol in tier2_set or decl.symbol in owner_arm:
                            first_hook = decl.symbol
                            break
                    if first_hook in tier2_plans:
                        plan = tier2_plans[first_hook]
                        seen_winner = False
                        delegated_owner: str | None = None
                        for decl in decls:
                            if not seen_winner:
                                if decl.symbol == plan.winner_symbol:
                                    seen_winner = True
                                continue
                            if decl.symbol in owner_arm:
                                delegated_owner = decl.symbol
                                break
                        if (
                            delegated_owner is None
                            or owner_arm[delegated_owner] != expected_arm
                        ):
                            raise_branch_order_mismatch(symbol, winner_symbol)
                        updated_evaluated.append(
                            (symbol, decls, delegated_owner, expected_arm)
                        )
                    else:
                        if owner_arm[winner_symbol] != expected_arm:
                            raise_branch_order_mismatch(symbol, winner_symbol)
                        updated_evaluated.append(
                            (symbol, decls, winner_symbol, expected_arm)
                        )
                evaluated = updated_evaluated
                precedence_overrides = [
                    tier2_plans[sym] for sym in sorted(tier2_plans)
                ]

        resolved_mros: list[tuple[str, tuple[CompactIndexedClass, ...], str]] = [
            (symbol, decls, winner_symbol)
            for symbol, decls, winner_symbol, _expected_arm in evaluated
        ]

        if method_name in base.bound_names:
            raise NeedsInput(
                f"`{base.qualname}` already binds `{method_name}`",
                NeededInput.OWNER_PICK,
            )
        for sym, target_cls in arm_targets:
            if method_name in target_cls.bound_names:
                raise NeedsInput(
                    f"`{target_cls.qualname}` already binds `{method_name}`",
                    NeededInput.OWNER_PICK,
                )
            # When a tested class is a mixin outside `base`'s subtree, verify no
            # ancestor or descendant of the mixin already binds `method_name`.
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
        for po in precedence_overrides:
            if method_name in po.target.bound_names:
                raise NeedsInput(
                    f"`{po.target.qualname}` already binds `{method_name}`",
                    NeededInput.OWNER_PICK,
                )

        for symbol, declarations, winner_symbol in resolved_mros:
            for decl in declarations:
                if decl.symbol == winner_symbol:
                    break
                if method_name in decl.member_binding_names:
                    raise NeedsInput(
                        f"`{decl.symbol}` already declares `{method_name}` ahead of "
                        f"`{base.qualname}` for `{symbol}`",
                        NeededInput.OWNER_PICK,
                    )

        # Determine which arms can delegate fallthrough to `base_symbol` via `super()`.
        # An arm is super-safe iff for every member in `subject_family` dispatched to that
        # arm, the very next `owner_arm` entry after `winner_symbol` in C3 MRO is `base_symbol`
        # and no intervening class declares `method_name`.
        arm_has_member: set[int] = set()
        super_unsafe_arms: set[int] = set()
        for _symbol, declarations, winner_symbol in resolved_mros:
            winning_arm = owner_arm.get(winner_symbol)
            if winning_arm is None:
                continue
            arm_has_member.add(winning_arm)
            after_winner = False
            next_owner: str | None = None
            intervening_conflict = False
            for decl in declarations:
                if not after_winner:
                    if decl.symbol == winner_symbol:
                        after_winner = True
                    continue
                if decl.symbol in owner_arm:
                    next_owner = decl.symbol
                    break
                if method_name in decl.member_binding_names:
                    intervening_conflict = True
                    break
            if next_owner != base_symbol or intervening_conflict:
                super_unsafe_arms.add(winning_arm)
        return _MroDispatchPlan(
            super_safe_arms=frozenset(arm_has_member - super_unsafe_arms),
            base_reorders=tuple(base_reorders),
            precedence_overrides=tuple(precedence_overrides),
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
        call = _format_receiver_call(self.site.subject, self.method_name, call_args)
        if self.is_async:
            call = f"await {call}"
        source_geometry = geometry(self.source_path)
        start, end = self.site.replaced_offsets(source_geometry)
        line_start = source_geometry.source.rfind("\n", 0, start) + 1
        line_indent = source_geometry.source[line_start:start]
        if not line_indent.isspace():
            line_indent = ""
        replacement_text = self.site.render_call_replacement(call, line_indent)
        add_span(
            self.source_path,
            start,
            end,
            replacement_text,
        )

        # 2. Target-method edits (new polymorphic family or empty for exact existing-method reuse).
        target_edits, target_retained = self.method_target.member_and_import_edits(
            self, snapshot, geometry, param_names, rationale
        )
        retained_in_source = set(_names_loaded_in_source(replacement_text)) | set(
            target_retained
        )
        obsolete_nodes = (
            *self.site.tested_expr_nodes(),
            *self.site.replaced_ast_nodes(),
        )
        extra_edits: list[NominalSourceEdit] = [
            *target_edits,
            *obsolete_import_edits(
                snapshot,
                self.source_path,
                obsolete_nodes,
                rationale,
                retained_names=frozenset(retained_in_source),
            ),
        ]

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
        return self.method_target.rationale(self)


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
                snapshot, target, test_line=primary.line
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
        required_tested_symbols = (
            finding.metrics.tested_symbols
            if isinstance(finding.metrics, OwnTypeTestMetrics)
            else ()
        )
        if pick is not None:
            method_name = pick.name
            docstring: str | None = pick.judgement
        else:
            # Verify the site is structurally proved and detect an exact existing family method first.
            probe_name = "__nra_probe_method"
            try:
                probe_proof = TypeTestDispatchProof.proved(
                    snapshot,
                    target,
                    primary.line,
                    SharedAncestorBaseSelection(),
                    probe_name,
                    None,
                    required_tested_symbols=required_tested_symbols,
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            if len(probe_proof.matching_existing_methods) == 1:
                method_name = probe_proof.matching_existing_methods[0]
                docstring = None
            elif site.is_whole_function and not (
                function.name.startswith("__") and function.name.endswith("__")
            ):
                method_name = function.name
                docstring = ast.get_docstring(function)
            else:
                return self.stopped_evaluation(
                    NeedsInput(
                        f"embedded type test in `{primary.symbol}` can push onto "
                        f"`{probe_proof.base.qualname}`; supply `owner` `{primary.symbol}` with a "
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
                required_tested_symbols=required_tested_symbols,
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
