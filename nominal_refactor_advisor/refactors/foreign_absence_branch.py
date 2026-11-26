"""Push absence-probing consumer branches onto the scanned class that owns the probed field.

When a consumer branches on `other.attr is None` / `other.attr is not None` and
`other` is an instance of a scanned concrete class, moving the branch onto that
class as a method eliminates the foreign absence probe without exporting a
derived boolean predicate back to callers.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

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
from nominal_refactor_advisor.declared_receivers import (
    DeclaredClassObject,
    DeclaredInstance,
    DeclaredReceiverTyping,
)
from nominal_refactor_advisor.models import (
    ForeignAbsenceProbeMetrics,
    RefactorFinding,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import (
    InferredReceiverType,
    ProbeOwnerResolver,
    ReceiverSpan,
)
from nominal_refactor_advisor.refactors.class_overlap import source_rewrite_target
from nominal_refactor_advisor.refactors.finding_recipes import PrewarmTypeCheckerMixin
from nominal_refactor_advisor.refactors.optional_ast_field import (
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    CapturedArgument,
    DispatchSite,
    FunctionNode,
    InstanceDispatchKind,
    TypeTestDispatchProof,
    _always_terminates,
    _always_void_terminates,
    _analyze_arm_bodies,
    _body_without_docstring,
    _class_append_span,
    _cross_module_imports_for_body,
    _ends_with_continue_only,
    _has_any_return,
    _has_loop_escape,
    _has_non_continue_loop_escape,
    _has_value_return,
    _is_pure_subject,
    _names_referenced_outside,
    _names_stored_in,
    _outer_receiver_alias,
    _render_indented_statements,
    _render_method,
    _rewrite_loop_continues_to_returns,
    _rewrite_subject_to_receiver,
    _statement_list_children,
)

NON_CONCRETE_OWNER_BASES = frozenset(
    (
        "typing.Protocol",
        "typing_extensions.Protocol",
        "typing.TypedDict",
        "typing_extensions.TypedDict",
    )
)


@dataclass(frozen=True)
class ParsedAbsenceProbe:
    """One `subject.attr is None` or `subject.attr is not None` comparison."""

    subject: ast.expr
    attr_name: str
    node: ast.Compare


def _parse_absence_probe(node: ast.AST) -> ParsedAbsenceProbe | None:
    match node:
        case ast.Compare(
            left=ast.Attribute(value=subject, attr=attr_name),
            ops=[ast.Is() | ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ) if _is_pure_subject(subject) and not (
            isinstance(subject, ast.Name) and subject.id in ("self", "cls")
        ):
            return ParsedAbsenceProbe(subject=subject, attr_name=attr_name, node=node)
    return None


def _absence_probes_in(
    node: ast.AST, target_lines: frozenset[int]
) -> tuple[ParsedAbsenceProbe, ...]:
    found: list[ParsedAbsenceProbe] = []
    for child in ast.walk(node):
        if getattr(child, "lineno", None) not in target_lines:
            continue
        parsed = _parse_absence_probe(child)
        if parsed is not None:
            found.append(parsed)
    return tuple(found)


def _leading_unconditional_probe(expr: ast.expr) -> ParsedAbsenceProbe | None:
    """The absence probe guaranteed to execute first when `expr` is evaluated, or None.

    A short-circuiting `if current is None or current.attr is None:` or
    `if guard and current.attr is not None:` does NOT unconditionally evaluate
    `current.attr` on entry, so pushing `current.method(...)` would dereference
    `current` before the short-circuit guard ran.
    """

    parsed = _parse_absence_probe(expr)
    if parsed is not None:
        return parsed
    match expr:
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return _leading_unconditional_probe(operand)
        case ast.BoolOp(values=[first, *_rest]):
            return _leading_unconditional_probe(first)
    return None


def _unconditional_absence_probes_in(
    test: ast.expr, target_lines: frozenset[int], subject_source: str | None = None
) -> tuple[ParsedAbsenceProbe, ...]:
    """Return probes on `target_lines` inside `test` only when `test` unconditionally
    evaluates a probe on that exact subject before any short-circuiting guard."""

    leading = _leading_unconditional_probe(test)
    if leading is None:
        return ()
    leading_dump = ast.dump(leading.subject)
    if subject_source is not None and ast.unparse(leading.subject) != subject_source:
        return ()
    probes = _absence_probes_in(test, target_lines)
    if subject_source is not None:
        probes = tuple(
            probe for probe in probes if ast.unparse(probe.subject) == subject_source
        )
    if not probes or any(ast.dump(probe.subject) != leading_dump for probe in probes):
        return ()
    return probes
def _is_bool_constant(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and isinstance(expr.value, bool)


def _is_zero_one_tuple_pair(first: ast.expr, second: ast.expr) -> bool:
    """Whether `(first, second)` returns `()` on one arm and `(x,)` on the other (ZERO_ONE_TUPLE)."""

    def _tuple_len(expr: ast.expr) -> int | None:
        return len(expr.elts) if isinstance(expr, ast.Tuple) else None

    return {_tuple_len(first), _tuple_len(second)} == {0, 1}


def _is_trivial_constant_expr(expr: ast.expr) -> bool:
    """Whether `expr` is a tiny literal safe to retain at a short-circuiting `IfExp` guard."""

    match expr:
        case ast.Constant():
            return True
        case ast.Tuple(elts=[]) | ast.List(elts=[]) | ast.Set(elts=[]) | ast.Dict(keys=[]):
            return True
    return False


def _replace_trailing_continue_with_return(
    statements: Sequence[ast.stmt],
) -> tuple[ast.stmt, ...]:
    return (*statements[:-1], ast.Return(value=None))


def _line_is_elif(source_text: str, lineno: int) -> bool:
    lines = source_text.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].lstrip().startswith("elif ")
    return False


@dataclass(frozen=True)
class AbsenceBranchSite(ABC):
    """One consumer-side absence-probe branch normalized for pushing onto the owner class."""

    subject: ast.expr
    probe_nodes: tuple[ast.Compare, ...]
    method_body: tuple[ast.stmt, ...]
    is_whole_function: bool

    @abstractmethod
    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        raise NotImplementedError

    @abstractmethod
    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        raise NotImplementedError

    @property
    def checks_local_store_escapes(self) -> bool:
        return False

    def covers_lines(self, lines: Sequence[int]) -> bool:
        nodes = self.replaced_ast_nodes()
        if not nodes:
            return False
        start = min(getattr(node, "lineno", 0) for node in nodes)
        end = max(getattr(node, "end_lineno", getattr(node, "lineno", 0)) for node in nodes)
        return all(start <= line <= end for line in lines)

    def as_dispatch_adapter(self) -> DispatchSite:
        """Minimal `DispatchSite` view for reusing `_outer_receiver_alias` / `_analyze_arm_bodies`."""

        return _AbsenceDispatchAdapter(
            subject=self.subject,
            kind=InstanceDispatchKind(),
            arms=(),
            default_body=self.method_body,
            is_whole_function=self.is_whole_function,
            test_nodes=self.probe_nodes,
        )


@dataclass(frozen=True)
class _AbsenceDispatchAdapter(DispatchSite):
    """Internal adapter so `type_test_dispatch` body-analysis helpers operate on an absence site."""

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class ReturningAbsenceBranchSite(AbsenceBranchSite):
    """Returning `if` / `if-else` / early-return tail replaced by `return subject.method(...)`."""

    replaced_nodes: tuple[ast.stmt, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"return {call_expr}"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes


@dataclass(frozen=True)
class VoidAbsenceBranchSite(AbsenceBranchSite):
    """Void or raising `if` / void early-return tail / loop-continue tail replaced by `subject.method(...)`."""

    replaced_nodes: tuple[ast.stmt, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def checks_local_store_escapes(self) -> bool:
        return not self.is_whole_function


@dataclass(frozen=True)
class PrefixGuardedOrAbsenceBranchSite(AbsenceBranchSite):
    """`if PREFIX or subject.attr is [not] None: early_exit` replaced by
    `if PREFIX:\\n    <early_exit>\\n[return ]subject.method(...)`."""

    replaced_nodes: tuple[ast.stmt, ...]
    guard_expr: ast.expr
    guard_body: tuple[ast.stmt, ...]
    returns_value: bool

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

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

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def checks_local_store_escapes(self) -> bool:
        return not self.returns_value


@dataclass(frozen=True)
class OrElseGuardedAbsenceBranchSite(PrefixGuardedOrAbsenceBranchSite):
    """Void `if PREFIX or subject.attr is [not] None: body` replaced by
    `if PREFIX:\\n    body\\nelse:\\n    subject.method(...)`."""

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


@dataclass(frozen=True)
class PrefixGuardedAndAbsenceBranchSite(AbsenceBranchSite):
    """Void `if PREFIX and subject.attr is [not] None: body` replaced by
    `if PREFIX:\\n    subject.method(...)`."""

    replaced_nodes: tuple[ast.stmt, ...]
    guard_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"if {ast.unparse(self.guard_expr)}:\n{line_indent}    {call_expr}"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def checks_local_store_escapes(self) -> bool:
        return True


@dataclass(frozen=True)
class TernaryAbsenceBranchSite(AbsenceBranchSite):
    """Non-boolean `a if subject.attr is [not] None else b` replaced inline by `subject.method(...)`."""

    node: ast.IfExp

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return (self.node,)


@dataclass(frozen=True)
class PrefixGuardedOrTernaryAbsenceBranchSite(TernaryAbsenceBranchSite):
    """`A if PREFIX or subject.attr is [not] None else B` replaced by `A if PREFIX else subject.method(...)`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{ast.unparse(self.short_circuit_expr)} if {ast.unparse(self.guard_expr)} else {call_expr}"


@dataclass(frozen=True)
class PrefixGuardedAndTernaryAbsenceBranchSite(TernaryAbsenceBranchSite):
    """`A if PREFIX and subject.attr is [not] None else B` replaced by `subject.method(...) if PREFIX else B`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{call_expr} if {ast.unparse(self.guard_expr)} else {ast.unparse(self.short_circuit_expr)}"


def _extract_absence_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    probes: tuple[ParsedAbsenceProbe, ...],
    *,
    in_loop_body: bool,
) -> AbsenceBranchSite | None:
    subject = probes[0].subject
    probe_nodes = tuple(probe.node for probe in probes)

    if stmt.orelse:
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
        any_returns = _has_any_return(replaced)
        if any_returns:
            if not (_always_terminates(stmt.body) and _always_terminates(stmt.orelse)):
                return None
            return ReturningAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=replaced,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        if _has_loop_escape(replaced):
            return None
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=replaced,
            is_whole_function=is_whole,
            replaced_nodes=replaced,
        )

    # No explicit `else:`: terminating tail, void early-return tail, loop continue tail, or void `if`.
    tail = container[index + 1 :]
    if tail and _always_terminates(stmt.body) and _always_terminates(tail):
        replaced = tuple(container[index:])
        is_whole = container is whole_fn_body and index == 0
        if _has_any_return(replaced):
            return ReturningAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=replaced,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=replaced,
            is_whole_function=is_whole,
            replaced_nodes=replaced,
        )
    if (
        container is whole_fn_body
        and tail
        and _always_void_terminates(stmt.body)
        and not _has_value_return(tail)
    ):
        replaced = tuple(container[index:])
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=replaced,
            is_whole_function=index == 0,
            replaced_nodes=replaced,
        )
    if (
        in_loop_body
        and tail
        and _ends_with_continue_only(stmt.body)
        and not _has_non_continue_loop_escape(tail)
    ):
        rewritten_if = ast.If(
            test=stmt.test,
            body=list(_replace_trailing_continue_with_return(stmt.body)),
            orelse=[],
        )
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=(rewritten_if, *_rewrite_loop_continues_to_returns(tail)),
            is_whole_function=False,
            replaced_nodes=tuple(container[index:]),
        )
    if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=replaced,
            is_whole_function=is_whole,
            replaced_nodes=replaced,
        )
    return None


def _split_prefix_guarded_boolop(
    test: ast.expr,
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
) -> tuple[ast.boolop, ast.expr, ast.expr, tuple[ParsedAbsenceProbe, ...]] | None:
    """Split `PREFIX <op> subject.attr is [not] None [<op> SUFFIX]` when all target probes
    sit unconditionally at or after the first matching operand."""

    if not isinstance(test, ast.BoolOp) or len(test.values) < 2:
        return None
    all_probes = _absence_probes_in(test, target_lines)
    if subject_source is not None:
        all_probes = tuple(
            probe for probe in all_probes if ast.unparse(probe.subject) == subject_source
        )
    if not all_probes:
        return None
    match_index: int | None = None
    leading: ParsedAbsenceProbe | None = None
    for idx, value in enumerate(test.values[1:], start=1):
        candidate = _leading_unconditional_probe(value)
        if candidate is None:
            continue
        if subject_source is not None and ast.unparse(candidate.subject) != subject_source:
            continue
        match_index = idx
        leading = candidate
        break
    if match_index is None or leading is None:
        return None
    leading_dump = ast.dump(leading.subject)
    if any(ast.dump(probe.subject) != leading_dump for probe in all_probes):
        return None
    prefix_values = test.values[:match_index]
    for prefix_expr in prefix_values:
        if any(
            ast.dump(probe.subject) == leading_dump
            for probe in _absence_probes_in(prefix_expr, target_lines)
        ):
            return None
    remaining_values = test.values[match_index:]
    remaining_probes = tuple(
        probe
        for rem in remaining_values
        for probe in _absence_probes_in(rem, target_lines)
        if ast.dump(probe.subject) == leading_dump
    )
    if len(remaining_probes) != len(all_probes):
        return None
    guard_expr: ast.expr = (
        prefix_values[0]
        if len(prefix_values) == 1
        else ast.BoolOp(op=test.op, values=prefix_values)
    )
    remaining_test: ast.expr = (
        remaining_values[0]
        if len(remaining_values) == 1
        else ast.BoolOp(op=test.op, values=remaining_values)
    )
    return (test.op, guard_expr, remaining_test, remaining_probes)


def _extract_prefix_guarded_absence_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
    in_loop_body: bool,
) -> AbsenceBranchSite | None:
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        return None
    split = _split_prefix_guarded_boolop(
        stmt.test, target_lines, subject_source=subject_source
    )
    if split is None:
        return None
    op, guard_expr, remaining_test, probes = split
    subject = probes[0].subject
    probe_nodes = tuple(probe.node for probe in probes)

    if isinstance(op, ast.Or):
        if stmt.orelse:
            replaced = (stmt,)
            if _has_any_return(replaced):
                if not (_always_terminates(stmt.body) and _always_terminates(stmt.orelse)):
                    return None
                return PrefixGuardedOrAbsenceBranchSite(
                    subject=subject,
                    probe_nodes=probe_nodes,
                    method_body=(
                        ast.If(
                            test=remaining_test,
                            body=list(stmt.body),
                            orelse=list(stmt.orelse),
                        ),
                    ),
                    is_whole_function=False,
                    replaced_nodes=replaced,
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=True,
                )
            return None
        tail = container[index + 1 :]
        if tail and _always_terminates(stmt.body) and _always_terminates(tail):
            replaced = tuple(container[index:])
            return PrefixGuardedOrAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
                    *tail,
                ),
                is_whole_function=False,
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
            return PrefixGuardedOrAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
                    *tail,
                ),
                is_whole_function=False,
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
            return PrefixGuardedOrAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(
                        test=remaining_test,
                        body=list(_replace_trailing_continue_with_return(stmt.body)),
                        orelse=[],
                    ),
                    *_rewrite_loop_continues_to_returns(tail),
                ),
                is_whole_function=False,
                replaced_nodes=tuple(container[index:]),
                guard_expr=guard_expr,
                guard_body=tuple(stmt.body),
                returns_value=False,
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            site_cls = (
                PrefixGuardedOrAbsenceBranchSite
                if _always_terminates(stmt.body)
                else OrElseGuardedAbsenceBranchSite
            )
            return site_cls(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
                ),
                is_whole_function=False,
                replaced_nodes=(stmt,),
                guard_expr=guard_expr,
                guard_body=tuple(stmt.body),
                returns_value=False,
            )
        return None

    if (
        isinstance(op, ast.And)
        and not stmt.orelse
        and not _has_any_return(stmt.body)
        and not _has_loop_escape(stmt.body)
    ):
        return PrefixGuardedAndAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=(
                ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
            ),
            is_whole_function=False,
            replaced_nodes=(stmt,),
            guard_expr=guard_expr,
        )
    return None


def _extract_absence_ternary_site(
    node: ast.IfExp,
    probes: tuple[ParsedAbsenceProbe, ...],
    *,
    is_whole_return: bool,
) -> TernaryAbsenceBranchSite | None:
    if (
        (_is_bool_constant(node.body) and _is_bool_constant(node.orelse))
        or _is_zero_one_tuple_pair(node.body, node.orelse)
    ):
        return None
    return TernaryAbsenceBranchSite(
        subject=probes[0].subject,
        probe_nodes=tuple(probe.node for probe in probes),
        method_body=(ast.Return(value=node),),
        is_whole_function=is_whole_return,
        node=node,
    )


def _extract_prefix_guarded_absence_ternary_site(
    node: ast.IfExp,
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
) -> TernaryAbsenceBranchSite | None:
    if (
        (_is_bool_constant(node.body) and _is_bool_constant(node.orelse))
        or _is_zero_one_tuple_pair(node.body, node.orelse)
    ):
        return None
    split = _split_prefix_guarded_boolop(
        node.test, target_lines, subject_source=subject_source
    )
    if split is None:
        return None
    op, guard_expr, remaining_test, probes = split
    inner_ifexp = ast.IfExp(test=remaining_test, body=node.body, orelse=node.orelse)
    if isinstance(op, ast.Or) and _is_trivial_constant_expr(node.body):
        return PrefixGuardedOrTernaryAbsenceBranchSite(
            subject=probes[0].subject,
            probe_nodes=tuple(probe.node for probe in probes),
            method_body=(ast.Return(value=inner_ifexp),),
            is_whole_function=False,
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.body,
        )
    if isinstance(op, ast.And) and _is_trivial_constant_expr(node.orelse):
        return PrefixGuardedAndTernaryAbsenceBranchSite(
            subject=probes[0].subject,
            probe_nodes=tuple(probe.node for probe in probes),
            method_body=(ast.Return(value=inner_ifexp),),
            is_whole_function=False,
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.orelse,
        )
    return None


def find_absence_branch_site(
    function: FunctionNode,
    source_text: str,
    test_line: int,
    *,
    subject_source: str | None = None,
) -> AbsenceBranchSite | None:
    """Locate a pushable `if` or non-boolean `IfExp` probing `subject.attr is [not] None` on `test_line`."""

    target_lines = frozenset((test_line,))
    body = _body_without_docstring(function)
    for container, index, stmt, in_loop_body in _statement_list_children(body):
        if isinstance(stmt, ast.If) and not _line_is_elif(source_text, stmt.lineno):
            probes = _unconditional_absence_probes_in(
                stmt.test, target_lines, subject_source=subject_source
            )
            if probes:
                extracted = _extract_absence_if_site(
                    container, index, stmt, body, probes, in_loop_body=in_loop_body
                )
                if extracted is not None:
                    return extracted
            extracted = _extract_prefix_guarded_absence_if_site(
                container,
                index,
                stmt,
                body,
                target_lines,
                subject_source=subject_source,
                in_loop_body=in_loop_body,
            )
            if extracted is not None:
                return extracted
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.IfExp):
            ifexp = stmt.value
            probes = _unconditional_absence_probes_in(
                ifexp.test, target_lines, subject_source=subject_source
            )
            if probes:
                is_whole = container is body and len(body) == 1
                extracted = _extract_absence_ternary_site(
                    ifexp, probes, is_whole_return=is_whole
                )
                if extracted is not None:
                    return extracted
            extracted = _extract_prefix_guarded_absence_ternary_site(
                ifexp, target_lines, subject_source=subject_source
            )
            if extracted is not None:
                return extracted
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp):
            probes = _unconditional_absence_probes_in(
                node.test, target_lines, subject_source=subject_source
            )
            if probes:
                extracted = _extract_absence_ternary_site(
                    node, probes, is_whole_return=False
                )
                if extracted is not None:
                    return extracted
            extracted = _extract_prefix_guarded_absence_ternary_site(
                node, target_lines, subject_source=subject_source
            )
            if extracted is not None:
                return extracted
    return None


def _resolve_scanned_instance_owner(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    subject: ast.expr,
) -> ResolvedClassTarget:
    """Resolve `subject` to a single scanned concrete class instance owner."""

    parsed = snapshot.parsed_module_for_source_path(source_path)
    index = snapshot.compact_class_index
    classes = index.classes_by_symbol
    typing = DeclaredReceiverTyping.from_projections(
        snapshot.compact_class_projections, index
    )
    path_owner = ProbeOwnerResolver.of(parsed).owner(subject)
    declared = path_owner.path.declared(typing)
    if isinstance(declared, DeclaredClassObject):
        raise NeedsInput(
            f"`{ast.unparse(subject)}` is a class object, not an instance",
            NeededInput.SOURCE_CHANGE,
        )
    owner_symbol: str | None = None
    if isinstance(declared, DeclaredInstance):
        symbols = declared.class_symbols()
        if len(symbols) == 1 and symbols[0] in classes:
            owner_symbol = symbols[0]
    if owner_symbol is None:
        span = ReceiverSpan.of(source_path, subject)
        answer = snapshot.type_checker.receiver_types((span,)).get(span)
        if not isinstance(answer, InferredReceiverType) or not answer.instances:
            raise NeedsInput(
                f"`{ast.unparse(subject)}` has no single scanned instance owner class",
                NeededInput.SOURCE_CHANGE,
            )
        if (
            re.search(r"\bNone\b", answer.spelled)
            or len(answer.classes) != 1
            or answer.classes[0] not in classes
        ):
            raise NeedsInput(
                f"the type checker infers `{answer.spelled}` for `{ast.unparse(subject)}`, "
                "which is not a single non-None scanned class",
                NeededInput.SOURCE_CHANGE,
            )
        owner_symbol = answer.classes[0]
    owner_indexed = classes[owner_symbol]
    ancestors = frozenset((owner_symbol, *index.ancestor_symbols(owner_symbol)))
    if ancestors & NON_CONCRETE_OWNER_BASES or any(
        ".".join(ref.resolved_parts) in NON_CONCRETE_OWNER_BASES
        for sym in ancestors
        if (decl := classes.get(sym)) is not None
        for ref in decl.base_references
    ):
        raise NeedsInput(
            f"`{owner_indexed.qualname}` is a Protocol or TypedDict and cannot own concrete methods",
            NeededInput.SOURCE_CHANGE,
        )
    return ResolvedClassTarget.from_rewrite_target(snapshot, source_rewrite_target(owner_indexed))


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


def _body_without_docstring(body: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
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
    stripped = _body_without_docstring(body)
    if _body_shadows_parameter(stripped, frozenset(param_names)):
        return None
    rename_map = {name: f"__p{index}" for index, name in enumerate(param_names)}
    transformer = _AlphaRenameParameters(rename_map)
    return tuple(
        ast.dump(transformer.visit(copy.deepcopy(stmt)), include_attributes=False)
        for stmt in stripped
    )


def _matching_existing_method_names(
    owner_node: ast.ClassDef,
    index: object,
    owner_symbol: str,
    rewritten_body: tuple[ast.stmt, ...],
    captured_args: tuple[CapturedArgument, ...],
    is_async: bool,
    import_edits: tuple[NominalSourceEdit, ...],
) -> tuple[str, ...]:
    """Find plain methods on `owner_node` with the exact generated signature and body."""

    if import_edits:
        return ()
    generated_params = ("self", *(arg.param_name for arg in captured_args))
    generated_dumps = _normalized_body_dumps(rewritten_body, generated_params)
    if generated_dumps is None:
        return ()
    classes = index.classes_by_symbol  # type: ignore[attr-defined]
    descendant_names: set[str] = set()
    for desc_symbol in index.descendant_symbols(owner_symbol):  # type: ignore[attr-defined]
        desc = classes.get(desc_symbol)
        if desc is not None:
            descendant_names.update(desc.member_binding_names)

    matches: list[str] = []
    for stmt in owner_node.body:
        if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if stmt.decorator_list:
            continue
        if isinstance(stmt, ast.AsyncFunctionDef) != is_async:
            continue
        if stmt.name.startswith("__") and stmt.name.endswith("__"):
            continue
        if stmt.name in descendant_names:
            continue
        args = stmt.args
        if (
            args.posonlyargs
            or args.kwonlyargs
            or args.vararg is not None
            or args.kwarg is not None
            or args.defaults
            or len(args.args) != len(generated_params)
        ):
            continue
        existing_dumps = _normalized_body_dumps(
            stmt.body, tuple(arg.arg for arg in args.args)
        )
        if existing_dumps == generated_dumps:
            matches.append(stmt.name)
    return tuple(matches)


@dataclass(frozen=True)
class OwnerAbsenceMethodTarget(ABC):
    """Whether an absence-branch push adds a new method or reuses an exact existing one."""

    method_name: str

    @abstractmethod
    def owner_replacements(
        self,
        proof: ForeignAbsenceBranchProof,
        owner_geometry: SourceTextGeometry,
    ) -> tuple[tuple[int, int, str], ...]:
        raise NotImplementedError

    @abstractmethod
    def import_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError

    @abstractmethod
    def rationale(self, proof: ForeignAbsenceBranchProof) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ExistingOwnerAbsenceMethod(OwnerAbsenceMethodTarget):
    """Reuse an already-declared owner method whose body matches identically."""

    def owner_replacements(
        self,
        proof: ForeignAbsenceBranchProof,
        owner_geometry: SourceTextGeometry,
    ) -> tuple[tuple[int, int, str], ...]:
        del proof, owner_geometry
        return ()

    def import_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        del proof, snapshot, rationale
        return ()

    def rationale(self, proof: ForeignAbsenceBranchProof) -> str:
        return (
            f"Reuse existing `{proof.owner.qualname}.{self.method_name}` for foreign "
            f"absence branch on `{ast.unparse(proof.site.subject)}` in "
            f"`{proof.function.name}`."
        )


@dataclass(frozen=True)
class NewOwnerAbsenceMethod(OwnerAbsenceMethodTarget):
    """Append a newly factored method onto the scanned owner class."""

    docstring: str | None = None

    def owner_replacements(
        self,
        proof: ForeignAbsenceBranchProof,
        owner_geometry: SourceTextGeometry,
    ) -> tuple[tuple[int, int, str], ...]:
        params = tuple(arg.param_name for arg in proof.captured_args)
        method_source = _render_method(
            InstanceDispatchKind(),
            self.method_name,
            params,
            proof.method_body,
            is_async=proof.is_async,
            docstring=self.docstring,
        )
        return (_class_append_span(owner_geometry, proof.owner.node, method_source),)

    def import_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        source_module = snapshot.parsed_module_for_source_path(proof.source_path)
        local_exclusions = frozenset(
            (
                "self",
                *(arg.param_name for arg in proof.captured_args),
                *_names_stored_in(proof.method_body),
            )
        )
        return _cross_module_imports_for_body(
            snapshot,
            source_module,
            proof.test_line,
            proof.owner,
            proof.method_body,
            local_exclusions,
            rationale=rationale,
        )

    def rationale(self, proof: ForeignAbsenceBranchProof) -> str:
        return (
            f"Push foreign absence branch on `{ast.unparse(proof.site.subject)}` in "
            f"`{proof.function.name}` onto `{proof.owner.qualname}.{self.method_name}`."
        )


@dataclass(frozen=True)
class ForeignAbsenceBranchProof:
    """A foreign absence probe proved safe to push onto the scanned owner class."""

    source_path: str
    test_line: int
    function: FunctionNode
    site: AbsenceBranchSite
    owner: ResolvedClassTarget
    owner_symbol: str
    method_body: tuple[ast.stmt, ...]
    captured_args: tuple[CapturedArgument, ...]
    is_async: bool
    method_target: OwnerAbsenceMethodTarget
    matching_existing_methods: tuple[str, ...] = ()

    @property
    def method_name(self) -> str:
        return self.method_target.method_name

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        test_line: int,
        owner_target: SourceRewriteTarget | None,
        method_name: str,
        docstring: str | None = None,
        *,
        subject_source: str | None = None,
    ) -> ForeignAbsenceBranchProof:
        if (
            not method_name.isidentifier()
            or keyword_module.iskeyword(method_name)
            or (method_name.startswith("__") and method_name.endswith("__"))
        ):
            raise NeedsInput(
                f"`{method_name}` is not a valid non-dunder method name",
                NeededInput.OWNER_PICK,
            )
        source_path, _enclosure, function = TypeTestDispatchProof.locate_function(
            snapshot, target
        )
        source_text = snapshot.sources_by_file_path[source_path]
        site = find_absence_branch_site(
            function, source_text, test_line, subject_source=subject_source
        )
        if site is None:
            raise NeedsInput(
                f"absence probe at {source_path}:{test_line} is not a pushable branch shape",
                NeededInput.SOURCE_CHANGE,
            )

        resolved_owner = _resolve_scanned_instance_owner(snapshot, source_path, site.subject)
        resolved_symbol = resolved_owner.required_symbol(snapshot)
        if owner_target is not None:
            explicit = ResolvedClassTarget.from_rewrite_target(snapshot, owner_target)
            if explicit.required_symbol(snapshot) != resolved_symbol:
                raise NeedsInput(
                    f"explicit owner `{explicit.qualname}` does not match proved owner `{resolved_owner.qualname}`",
                    NeededInput.PROOF,
                )

        if site.checks_local_store_escapes:
            stored = _names_stored_in(site.method_body)
            used_outside = _names_referenced_outside(function, site.replaced_ast_nodes())
            escaping = sorted(stored & used_outside)
            if escaping:
                raise NeedsInput(
                    f"`{function.name}` assigns outer local(s) {', '.join(escaping)} inside the absence branch",
                    NeededInput.SOURCE_CHANGE,
                )

        adapter = site.as_dispatch_adapter()
        outer_receiver_param = _outer_receiver_alias(
            function, adapter, (site.method_body,)
        )
        rewritten_body = _rewrite_subject_to_receiver(
            site.method_body,
            site.subject,
            "self",
            outer_receiver_param=outer_receiver_param,
        )
        captured_args, is_async = _analyze_arm_bodies(
            function, adapter, (rewritten_body,), outer_receiver_param=outer_receiver_param
        )

        # Verify cross-module imports are cycle-free and resolvable.
        source_module = snapshot.parsed_module_for_source_path(source_path)
        local_exclusions = frozenset(
            ("self", *(arg.param_name for arg in captured_args), *_names_stored_in(rewritten_body))
        )
        try:
            import_edits = _cross_module_imports_for_body(
                snapshot,
                source_module,
                test_line,
                resolved_owner,
                rewritten_body,
                local_exclusions,
                rationale=f"Import symbols needed by `{resolved_owner.qualname}.{method_name}`.",
            )
        except NeedsInput:
            raise
        except ValueError as error:
            raise NeedsInput(str(error), NeededInput.SOURCE_CHANGE) from error

        index = snapshot.compact_class_index
        classes = index.classes_by_symbol
        matching_existing = _matching_existing_method_names(
            resolved_owner.node,
            index,
            resolved_symbol,
            rewritten_body,
            captured_args,
            is_async,
            import_edits,
        )
        method_target: OwnerAbsenceMethodTarget
        if method_name in matching_existing:
            method_target = ExistingOwnerAbsenceMethod(method_name=method_name)
        else:
            family_symbols = (
                resolved_symbol,
                *index.ancestor_symbols(resolved_symbol),
                *index.descendant_symbols(resolved_symbol),
            )
            for symbol in family_symbols:
                decl = classes.get(symbol)
                if decl is not None and method_name in decl.member_binding_names:
                    raise NeedsInput(
                        f"`{decl.qualname}` already declares `{method_name}`",
                        NeededInput.OWNER_PICK,
                    )
            method_target = NewOwnerAbsenceMethod(
                method_name=method_name, docstring=docstring
            )

        return cls(
            source_path=source_path,
            test_line=test_line,
            function=function,
            site=site,
            owner=resolved_owner,
            owner_symbol=resolved_symbol,
            method_body=rewritten_body,
            captured_args=captured_args,
            is_async=is_async,
            method_target=method_target,
            matching_existing_methods=matching_existing,
        )

    def rationale(self) -> str:
        return self.method_target.rationale(self)

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        import_edits = self.method_target.import_edits(self, snapshot, rationale)
        owner_geometry = SourceTextGeometry(
            snapshot.sources_by_file_path[self.owner.file_path]
        )
        owner_replacements = self.method_target.owner_replacements(self, owner_geometry)

        subject_source = ast.unparse(self.site.subject)
        call_args = ", ".join(arg.call_expr for arg in self.captured_args)
        raw_call = f"{subject_source}.{self.method_name}({call_args})"
        call_expr = f"await {raw_call}" if self.is_async else raw_call
        consumer_geometry = SourceTextGeometry(
            snapshot.sources_by_file_path[self.source_path]
        )
        rep_start, rep_end = self.site.replaced_offsets(consumer_geometry)
        line_start = consumer_geometry.source.rfind("\n", 0, rep_start) + 1
        line_indent = consumer_geometry.source[line_start:rep_start]
        if not line_indent.isspace():
            line_indent = ""
        replacement = self.site.render_call_replacement(call_expr, line_indent)

        replacements_by_file: dict[str, list[tuple[int, int, str]]] = {}
        for owner_span in owner_replacements:
            replacements_by_file.setdefault(self.owner.file_path, []).append(owner_span)
        replacements_by_file.setdefault(self.source_path, []).append(
            (rep_start, rep_end, replacement)
        )
        geometries = {
            self.owner.file_path: owner_geometry,
            self.source_path: consumer_geometry,
        }
        span_edits = tuple(
            geometries[file_path].nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=text,
                    )
                    for start, end, text in spans
                ),
                rationale=rationale,
            )
            for file_path, spans in replacements_by_file.items()
        )
        return (*span_edits, *import_edits)


@dataclass(frozen=True, kw_only=True)
class PushForeignAbsenceBranchOperation(RepositorySourceReprovedOperation):
    """Push a foreign absence-probe branch onto the scanned class that owns the probed field."""

    test_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    owner: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )
    subject_source: str | None = codemod_payload_field(
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
        proof = ForeignAbsenceBranchProof.proved(
            snapshot,
            self.target,
            self.test_line,
            self.owner,
            self.method_name,
            self.docstring,
            subject_source=self.subject_source,
        )
        return proof.edits(snapshot, self.rationale_text(proof.rationale()))


class ForeignAbsenceProbeFindingRecipeSynthesizer(
    OptionalAstFieldDecoderFindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
    ABC,
):
    """Route optional AST field probes through boundary decoders, or push scanned-class absence branches onto their owner."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        ast_evaluation = super().evaluate_recipe_for_finding(finding, context)
        if not ast_evaluation.rejection_reason:
            return ast_evaluation
        if context is None or not finding.evidence:
            return ast_evaluation
        snapshot = context.execution_snapshot()
        metrics = (
            finding.metrics
            if isinstance(finding.metrics, ForeignAbsenceProbeMetrics)
            else None
        )
        owner_symbol = None if metrics is None else metrics.owner_symbol
        if owner_symbol is None:
            return ast_evaluation
        classes = snapshot.compact_class_index.classes_by_symbol
        if owner_symbol not in classes:
            return ast_evaluation

        primary = finding.evidence[0]
        target = SourceRewriteTarget(
            file_path=primary.file_path, qualname=primary.symbol
        )
        subject_source = None if metrics is None else metrics.subject_source
        probe_lines = (
            metrics.probe_lines
            if metrics is not None and metrics.probe_lines
            else tuple(sorted({ev.line for ev in finding.evidence}))
        )
        try:
            _path, _enclosure, function = TypeTestDispatchProof.locate_function(
                snapshot, target
            )
        except ValueError as error:
            return self.stopped_evaluation(error)
        source_text = snapshot.sources_by_file_path[primary.file_path]
        site = find_absence_branch_site(
            function, source_text, probe_lines[0], subject_source=subject_source
        )
        if site is None or not site.covers_lines(probe_lines):
            return self.stopped_evaluation(
                NeedsInput(
                    f"absence probe at {primary.file_path}:{probe_lines[0]} is not a pushable branch shape",
                    NeededInput.SOURCE_CHANGE,
                )
            )

        pick = finding.metrics.picked_base
        owner_decl = classes[owner_symbol]
        owner_target = source_rewrite_target(owner_decl)
        is_dunder = function.name.startswith("__") and function.name.endswith("__")
        if pick is not None:
            method_name = pick.name
            docstring: str | None = pick.judgement
        else:
            probe_name = "__nra_probe_absence_method"
            try:
                probe_proof = ForeignAbsenceBranchProof.proved(
                    snapshot,
                    target,
                    probe_lines[0],
                    owner_target,
                    probe_name,
                    None,
                    subject_source=subject_source,
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            if len(probe_proof.matching_existing_methods) == 1:
                method_name = probe_proof.matching_existing_methods[0]
                docstring = None
            elif site.is_whole_function and not is_dunder:
                method_name = function.name
                docstring = ast.get_docstring(function)
            else:
                return self.stopped_evaluation(
                    NeedsInput(
                        f"embedded absence probe in `{primary.symbol}` can push onto "
                        f"`{probe_proof.owner.qualname}`; supply `owner` `{primary.symbol}` with a "
                        "method name and one-line judgement in `--codemod-owner-picks`",
                        NeededInput.OWNER_PICK,
                    )
                )

        try:
            proof = ForeignAbsenceBranchProof.proved(
                snapshot,
                target,
                probe_lines[0],
                owner_target,
                method_name,
                docstring,
                subject_source=subject_source,
            )
        except ValueError as error:
            return self.stopped_evaluation(error)

        operation = PushForeignAbsenceBranchOperation(
            target=target,
            test_line=probe_lines[0],
            owner=SourceRewriteTarget(
                file_path=proof.owner.file_path, qualname=proof.owner.qualname
            ),
            method_name=method_name,
            docstring=docstring,
            subject_source=subject_source,
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
    "AbsenceBranchSite",
    "ForeignAbsenceBranchProof",
    "ForeignAbsenceProbeFindingRecipeSynthesizer",
    "ParsedAbsenceProbe",
    "PushForeignAbsenceBranchOperation",
    "ReturningAbsenceBranchSite",
    "TernaryAbsenceBranchSite",
    "VoidAbsenceBranchSite",
    "find_absence_branch_site",
)
