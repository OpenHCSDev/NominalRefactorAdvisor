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

from nominal_refactor_advisor.ast_field_decoders import (
    optional_values,
    variadic_parameters,
)
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
from nominal_refactor_advisor.relocation_observations import PURE_BUILTINS
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    CapturedArgument,
    DispatchSite,
    FunctionEnclosure,
    FunctionNode,
    InstanceDispatchKind,
    ModuleEnclosure,
    OuterReceiverAlias,
    ResolvedMethodAnnotations,
    TypeTestDispatchProof,
    _all_returns_proved_non_none,
    _always_terminates,
    _always_void_terminates,
    _analyze_arm_bodies,
    _body_without_docstring,
    _class_append_edits,
    _cross_module_imports_for_body,
    _definitely_assigns_in_block,
    _ends_with_continue_only,
    _expr_proved_non_none,
    _has_any_return,
    _has_break_or_continue,
    _has_loop_escape,
    _has_non_continue_loop_escape,
    _has_value_return,
    _is_definitely_bound_before,
    _is_dispatchable_match_subject,
    _is_not_none_guard_dumps,
    _is_pure_subject,
    _names_referenced_outside,
    _names_stored_in,
    _normalized_body_dumps,
    _outer_receiver_aliases,
    _reads_before_assign_in_block,
    _render_indented_statements,
    _render_method,
    _rewrite_loop_continues_to_returns,
    _rewrite_subject_to_receiver,
    _statement_list_children,
    _trivial_pure_return,
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


def _node_covers_line(node: ast.AST, line: int) -> bool:
    start = getattr(node, "lineno", None)
    if start is None:
        return False
    end = getattr(node, "end_lineno", start) or start
    return start <= line <= end


def _is_pure_slice(expr: ast.expr) -> bool:
    match expr:
        case ast.Constant():
            return True
        case ast.Name() | ast.Attribute():
            return _is_pure_subject(expr)
        case ast.Tuple(elts=elts):
            return all(_is_pure_slice(elt) for elt in elts)
    return False


def _is_pure_subscript(expr: ast.expr) -> bool:
    match expr:
        case ast.Subscript(value=value, slice=sl):
            return (
                _is_pure_subject(value) or _is_pure_subscript(value)
            ) and _is_pure_slice(sl)
        case ast.Attribute(value=value):
            return _is_pure_subscript(value)
    return False


def _subject_occurs_outside_probes(
    subject_dump: str,
    nodes: Sequence[ast.AST],
    excluded_probe_subject_ids: frozenset[int],
) -> bool:
    for root in nodes:
        for child in ast.walk(root):
            if (
                isinstance(child, ast.expr)
                and id(child) not in excluded_probe_subject_ids
                and ast.dump(child) == subject_dump
            ):
                return True
    return False


def _subject_allowed_for_branch(
    subject: ast.expr,
    probe_nodes: tuple[ast.Compare, ...],
    method_body: tuple[ast.stmt, ...],
    extra_exprs: Sequence[ast.expr] = (),
) -> bool:
    if _is_pure_subject(subject):
        return True
    if not _is_dispatchable_match_subject(subject):
        return False
    probe_subject_ids = frozenset(
        id(probe.left.value)
        for probe in probe_nodes
        if isinstance(probe.left, ast.Attribute)
    )
    subject_dump = ast.dump(subject)
    extra_roots: list[ast.AST] = [*method_body, *extra_exprs]
    if not _subject_occurs_outside_probes(subject_dump, extra_roots, probe_subject_ids):
        return len(probe_nodes) == 1
    if _is_pure_subscript(subject):
        subject_names = frozenset(
            child.id
            for child in ast.walk(subject)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        )
        if subject_names & _names_stored_in(method_body):
            return False
        for stmt in method_body:
            for child in ast.walk(stmt):
                match child:
                    case ast.Subscript(value=ast.Name(id=nm), ctx=ast.Store() | ast.Del()) if (
                        nm in subject_names
                    ):
                        return False
        return True
    return False


def _parse_absence_probe(node: ast.AST) -> ParsedAbsenceProbe | None:
    match node:
        case ast.Compare(
            left=ast.Attribute(value=subject, attr=attr_name),
            ops=[ast.Is() | ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ) if _is_dispatchable_match_subject(subject) and not (
            isinstance(subject, ast.Name) and subject.id in ("self", "cls")
        ):
            return ParsedAbsenceProbe(subject=subject, attr_name=attr_name, node=node)
    return None


def _absence_probes_in(
    node: ast.AST, target_lines: frozenset[int]
) -> tuple[ParsedAbsenceProbe, ...]:
    found: list[ParsedAbsenceProbe] = []
    for child in ast.walk(node):
        if not any(_node_covers_line(child, line) for line in target_lines):
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

    @property
    def propagates_function_return_type(self) -> bool:
        return self.is_whole_function

    @property
    def is_void_method(self) -> bool:
        return False

    @property
    def returns_bool(self) -> bool:
        return False

    @property
    def reassigned_target(self) -> str | None:
        return None

    def with_assigning_target(
        self,
        target_name: str,
        method_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> AbsenceBranchSite | None:
        """Promote this void branch to return and assign `target_name` when supported."""

        del target_name, method_body, reassigned_target
        return None

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return ()

    def subject_allowed(self) -> bool:
        return _subject_allowed_for_branch(
            self.subject, self.probe_nodes, self.method_body, self.guard_exprs()
        )

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
            _reassigned_target=self.reassigned_target,
        )


@dataclass(frozen=True)
class _AbsenceDispatchAdapter(DispatchSite):
    """Internal adapter so `type_test_dispatch` body-analysis helpers operate on an absence site."""

    _reassigned_target: str | None = None

    @property
    def reassigned_target(self) -> str | None:
        return self._reassigned_target

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

    @property
    def propagates_function_return_type(self) -> bool:
        return True


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

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def is_void_method(self) -> bool:
        return True

    def with_assigning_target(
        self,
        target_name: str,
        method_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> AbsenceBranchSite | None:
        return AssigningAbsenceBranchSite(
            subject=self.subject,
            probe_nodes=self.probe_nodes,
            method_body=method_body,
            is_whole_function=False,
            replaced_nodes=self.replaced_nodes,
            target_name=target_name,
            _reassigned_target=reassigned_target,
        )


@dataclass(frozen=True)
class AssigningAbsenceBranchSite(AbsenceBranchSite):
    """Absence branch computing or updating one local, replaced by `target = subject.method(...)`."""

    replaced_nodes: tuple[ast.stmt, ...]
    target_name: str
    _reassigned_target: str | None = None

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{self.target_name} = {call_expr}"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def reassigned_target(self) -> str | None:
        return self._reassigned_target


@dataclass(frozen=True)
class PrefixGuardedAndAssigningAbsenceBranchSite(AssigningAbsenceBranchSite):
    """`if PREFIX and subject.attr is [not] None: target = ...` replaced by
    `if PREFIX:\\n    target = subject.method(...)`."""

    guard_expr: ast.expr = ast.Constant(value=True)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return (
            f"if {ast.unparse(self.guard_expr)}:\n"
            f"{line_indent}    {self.target_name} = {call_expr}"
        )


@dataclass(frozen=True)
class ContinuingAbsenceBranchSite(VoidAbsenceBranchSite):
    """Nested-block loop-`continue` tail replaced by `subject.method(...)\\ncontinue`."""

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"{call_expr}\n{line_indent}continue"

    def with_assigning_target(
        self,
        target_name: str,
        method_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> AbsenceBranchSite | None:
        del target_name, method_body, reassigned_target
        return None


@dataclass(frozen=True)
class ElifAbsenceBranchSite(AbsenceBranchSite):
    """`elif` branch rewritten to `else:\\n    <inner replacement>`."""

    inner: AbsenceBranchSite

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return self.inner.replaced_offsets(geometry)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        body_indent = f"{line_indent}    "
        inner_rendered = self.inner.render_call_replacement(call_expr, body_indent)
        return f"else:\n{body_indent}{inner_rendered}"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.inner.replaced_ast_nodes()

    def subject_allowed(self) -> bool:
        return self.inner.subject_allowed()

    @property
    def checks_local_store_escapes(self) -> bool:
        return self.inner.checks_local_store_escapes

    @property
    def propagates_function_return_type(self) -> bool:
        return self.inner.propagates_function_return_type

    @property
    def is_void_method(self) -> bool:
        return self.inner.is_void_method

    def with_assigning_target(
        self,
        target_name: str,
        method_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> AbsenceBranchSite | None:
        promoted = self.inner.with_assigning_target(
            target_name, method_body, reassigned_target=reassigned_target
        )
        if promoted is None:
            return None
        return ElifAbsenceBranchSite(
            subject=self.subject,
            probe_nodes=self.probe_nodes,
            method_body=promoted.method_body,
            is_whole_function=False,
            inner=promoted,
        )


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

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    @property
    def checks_local_store_escapes(self) -> bool:
        return not self.returns_value

    @property
    def propagates_function_return_type(self) -> bool:
        return self.returns_value

    @property
    def is_void_method(self) -> bool:
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

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    @property
    def checks_local_store_escapes(self) -> bool:
        return True

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def is_void_method(self) -> bool:
        return True

    def with_assigning_target(
        self,
        target_name: str,
        method_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> AbsenceBranchSite | None:
        return PrefixGuardedAndAssigningAbsenceBranchSite(
            subject=self.subject,
            probe_nodes=self.probe_nodes,
            method_body=method_body,
            is_whole_function=False,
            replaced_nodes=self.replaced_nodes,
            target_name=target_name,
            _reassigned_target=reassigned_target,
            guard_expr=self.guard_expr,
        )


@dataclass(frozen=True)
class PrefixGuardedAndReturningAbsenceBranchSite(AbsenceBranchSite):
    """Returning `if PREFIX and subject.attr is [not] None: return ...` with a trivial pure
    fallback return replaced by `if PREFIX:\\n    return subject.method(...)\\n<fallback>`."""

    replaced_nodes: tuple[ast.stmt, ...]
    guard_expr: ast.expr
    fallback_return: ast.Return

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return (
            f"if {ast.unparse(self.guard_expr)}:\n"
            f"{line_indent}    return {call_expr}\n"
            f"{line_indent}{ast.unparse(self.fallback_return)}"
        )

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    @property
    def propagates_function_return_type(self) -> bool:
        return True


@dataclass(frozen=True)
class CompositeTernaryAbsenceBranchSite(AbsenceBranchSite):
    """Composite expression containing multiple non-boolean `IfExp`s on one subject replaced by `subject.method(...)`."""

    node: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return (self.node,)

    def subject_allowed(self) -> bool:
        return _is_pure_subject(self.subject) or _is_pure_subscript(self.subject)


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

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{ast.unparse(self.short_circuit_expr)} if {ast.unparse(self.guard_expr)} else {call_expr}"


@dataclass(frozen=True)
class PrefixGuardedAndTernaryAbsenceBranchSite(TernaryAbsenceBranchSite):
    """`A if PREFIX and subject.attr is [not] None else B` replaced by `subject.method(...) if PREFIX else B`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{call_expr} if {ast.unparse(self.guard_expr)} else {ast.unparse(self.short_circuit_expr)}"


@dataclass(frozen=True)
class NonNoneReturnAbsenceBranchSite(AbsenceBranchSite):
    """Absence branch returning a proved non-`None` value (or falling through as `None`),
    replaced by `if (_found := subject.method(...)) is not None:\\n    return _found`."""

    replaced_nodes: tuple[ast.stmt, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"if (_found := {call_expr}) is not None:\n{line_indent}    return _found"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def checks_local_store_escapes(self) -> bool:
        return True

    @property
    def propagates_function_return_type(self) -> bool:
        return False


@dataclass(frozen=True)
class PrefixGuardedAndNonNoneReturnAbsenceBranchSite(NonNoneReturnAbsenceBranchSite):
    """`if PREFIX and subject.attr is [not] None: return <non-None>` replaced by
    `if PREFIX and (_found := subject.method(...)) is not None:\\n    return _found`."""

    guard_expr: ast.expr = ast.Constant(value=True)

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return (
            f"if {ast.unparse(self.guard_expr)} and (_found := {call_expr}) is not None:\n"
            f"{line_indent}    return _found"
        )


@dataclass(frozen=True)
class VoidReturnAbsenceBranchSite(AbsenceBranchSite):
    """Nested-block bare-`return` early-exit tail replaced by `if subject.method(...):\\n    return`."""

    replaced_nodes: tuple[ast.stmt, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"if {call_expr}:\n{line_indent}    return"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    @property
    def checks_local_store_escapes(self) -> bool:
        return True

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def returns_bool(self) -> bool:
        return True


@dataclass(frozen=True)
class PrefixGuardedAndLoopContinueAbsenceBranchSite(AbsenceBranchSite):
    """Loop `if PREFIX and subject.attr is [not] None: ... continue` replaced by
    `if PREFIX and subject.method(...):\\n    continue`."""

    replaced_nodes: tuple[ast.stmt, ...]
    guard_expr: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        start, _ = geometry.required_node_offsets(self.replaced_nodes[0])
        _, end = geometry.required_node_offsets(self.replaced_nodes[-1])
        return start, end

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        return f"if {ast.unparse(self.guard_expr)} and {call_expr}:\n{line_indent}    continue"

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return self.replaced_nodes

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    @property
    def checks_local_store_escapes(self) -> bool:
        return True

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def returns_bool(self) -> bool:
        return True


@dataclass(frozen=True)
class PrefixGuardedAndNonNoneTernaryAbsenceBranchSite(TernaryAbsenceBranchSite):
    """`A if PREFIX and subject.attr is [not] None else B` (proved non-`None` `A`) replaced by
    `(_found if PREFIX and (_found := subject.method(...)) is not None else B)`."""

    guard_expr: ast.expr
    short_circuit_expr: ast.expr

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return (
            f"(_found if {ast.unparse(self.guard_expr)} and (_found := {call_expr}) is not None "
            f"else {ast.unparse(self.short_circuit_expr)})"
        )


def _has_break_or_yield(statements: Sequence[ast.stmt]) -> bool:
    """True when `statements` contain loop `break`, `yield`, or `yield from` outside nested defs/loops."""
    stack: list[ast.AST] = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, ast.Break):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # `break` inside a nested loop stays inside that loop, while `yield` still escapes.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Yield, ast.YieldFrom)):
                    return True
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _uses_local_name(function: FunctionNode, target: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == target
        for node in ast.walk(function)
    )


def _is_vacuous_attr_return(stmt: ast.If, subject_dump: str) -> bool:
    """True when `stmt.body` is solely `return subject.<attr>` with no domain work."""
    if len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Return):
        return False
    values = optional_values(stmt.body[0])
    if not values:
        return False
    return _is_bare_subject_attr(values[0], subject_dump)


def _is_bare_subject_attr(expr: ast.expr, subject_dump: str) -> bool:
    return isinstance(expr, ast.Attribute) and ast.dump(expr.value) == subject_dump


def _rewrite_bare_returns_to_true(statements: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    """Deep-copy `statements`, replacing bare `return` with `return True` outside nested defs."""

    class _Rewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            return node

        def visit_Return(self, node: ast.Return) -> ast.AST:
            return ast.Return(value=ast.Constant(value=True))

    rewriter = _Rewriter()
    return tuple(rewriter.visit(copy.deepcopy(stmt)) for stmt in statements)


def _rewrite_loop_continues_to_true(statements: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    """Deep-copy `statements`, replacing loop-level `continue` with `return True`."""

    class _Rewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            return node

        def visit_For(self, node: ast.For) -> ast.AST:
            return node

        def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
            return node

        def visit_While(self, node: ast.While) -> ast.AST:
            return node

        def visit_Continue(self, node: ast.Continue) -> ast.AST:
            del node
            return ast.Return(value=ast.Constant(value=True))

    rewriter = _Rewriter()
    return tuple(rewriter.visit(copy.deepcopy(stmt)) for stmt in statements)


def _all_subject_probes_in(
    nodes: Sequence[ast.AST], subject_dump: str
) -> tuple[ast.Compare, ...]:
    found: list[ast.Compare] = []
    for root in nodes:
        for node in ast.walk(root):
            probe = _parse_absence_probe(node)
            if probe is not None and ast.dump(probe.subject) == subject_dump:
                found.append(probe.node)
    return tuple(found)


def _non_none_callee_names(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
) -> frozenset[str]:
    """Names in `source_path` resolving to scanned classes with closed MROs and no custom `__new__`."""
    index = snapshot.compact_class_index
    classes = index.classes_by_symbol
    authority = index.declared_mro_authority
    safe: set[str] = set()

    def class_never_returns_none(symbol: str) -> bool:
        if symbol not in classes:
            return False
        mro = authority.resolve(symbol).mro_type
        if mro is None:
            return False
        return not any("__new__" in decl.member_binding_names for decl in mro.declarations)

    parsed = snapshot.parsed_module_for_source_path(source_path)
    module_identity = parsed.module_path_identity
    for decl in classes.values():
        if decl.file_path == source_path and "." not in decl.qualname:
            if class_never_returns_none(decl.symbol):
                safe.add(decl.qualname)
    for node in parsed.module.body:
        match node:
            case ast.ImportFrom(module=mod, names=aliases, level=level):
                resolved_mod = module_identity.resolve_import_from_module(
                    imported_module=mod, level=level
                )
                if resolved_mod is None:
                    continue
                for alias in aliases:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    if class_never_returns_none(f"{resolved_mod}.{alias.name}"):
                        safe.add(bound)
    enclosing_sym = enclosure.class_receiver_symbol(snapshot)
    if (
        enclosing_sym is not None
        and class_never_returns_none(enclosing_sym)
        and any(
            isinstance(dec, ast.Name) and dec.id == "classmethod"
            for dec in function.decorator_list
        )
        and function.args.args
    ):
        safe.add(function.args.args[0].arg)
    return frozenset(safe)


@dataclass(frozen=True)
class BoolOpAbsenceBranchSite(AbsenceBranchSite):
    """Compound `subject.attr is [not] None and/or <domain>` expression replaced inline by `subject.method(...)`."""

    node: ast.BoolOp
    is_bool_expr: bool = False

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return (self.node,)

    @property
    def returns_bool(self) -> bool:
        return self.is_bool_expr


@dataclass(frozen=True)
class PrefixGuardedBoolOpAbsenceBranchSite(BoolOpAbsenceBranchSite):
    """`PREFIX and/or subject.attr is [not] None and/or <domain>` replaced by `PREFIX and/or subject.method(...)`."""

    guard_expr: ast.expr = ast.Constant(value=True)
    op_keyword: str = "and"

    def guard_exprs(self) -> tuple[ast.expr, ...]:
        return (self.guard_expr,)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return f"{ast.unparse(self.guard_expr)} {self.op_keyword} {call_expr}"


def _extract_absence_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    probes: tuple[ParsedAbsenceProbe, ...],
    *,
    in_loop_body: bool,
    allow_found_walrus: bool = False,
    non_none_callees: frozenset[str] = frozenset(),
) -> AbsenceBranchSite | None:
    subject = probes[0].subject
    subject_dump = ast.dump(subject)
    probe_nodes = tuple(probe.node for probe in probes)

    if stmt.orelse:
        replaced = (stmt,)
        is_whole = container is whole_fn_body and len(whole_fn_body) == 1
        any_returns = _has_any_return(replaced)
        if any_returns:
            if _always_terminates(stmt.body) and _always_terminates(stmt.orelse):
                return ReturningAbsenceBranchSite(
                    subject=subject,
                    probe_nodes=probe_nodes,
                    method_body=replaced,
                    is_whole_function=is_whole,
                    replaced_nodes=replaced,
                )
            tail = container[index + 1 :]
            if (
                tail
                and _always_terminates(tail)
                and not _has_break_or_continue((stmt, *tail))
            ):
                extended = tuple(container[index:])
                return ReturningAbsenceBranchSite(
                    subject=subject,
                    probe_nodes=probe_nodes,
                    method_body=extended,
                    is_whole_function=container is whole_fn_body and index == 0,
                    replaced_nodes=extended,
                )
            return None
        if _has_loop_escape(replaced):
            return None
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=replaced,
            is_whole_function=is_whole,
            replaced_nodes=replaced,
        )

    # No explicit `else:`: terminating/partial-return tail, void early-return tail, loop continue tail,
    # proved non-`None` early-return, or void/multi-probe sibling slice.
    tail = container[index + 1 :]
    if (
        tail
        and _always_terminates(tail)
        and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
        and not _has_break_or_continue((stmt, *tail))
    ):
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
        container is not whole_fn_body
        and tail
        and _always_void_terminates(stmt.body)
        and _has_any_return(stmt.body)
        and not _has_value_return(tail)
        and not _has_loop_escape(tail)
    ):
        replaced = tuple(container[index:])
        return VoidReturnAbsenceBranchSite(
            subject=subject,
            probe_nodes=_all_subject_probes_in(replaced, subject_dump) or probe_nodes,
            method_body=(
                *_rewrite_bare_returns_to_true(replaced),
                ast.Return(value=ast.Constant(value=False)),
            ),
            is_whole_function=False,
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
    if (
        not in_loop_body
        and len(tail) >= 2
        and isinstance(tail[-1], ast.Continue)
        and _ends_with_continue_only(stmt.body)
        and not _has_non_continue_loop_escape(tail)
    ):
        rewritten_if = ast.If(
            test=stmt.test,
            body=list(_replace_trailing_continue_with_return(stmt.body)),
            orelse=[],
        )
        return ContinuingAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=(rewritten_if, *_rewrite_loop_continues_to_returns(tail[:-1])),
            is_whole_function=False,
            replaced_nodes=tuple(container[index:]),
        )
    if allow_found_walrus and in_loop_body:
        loop_slice = tuple(container[index:])
        if (
            _has_value_return(loop_slice)
            and not _has_break_or_yield(loop_slice)
            and _all_returns_proved_non_none(
                loop_slice, subject_dump, non_none_callees
            )
            and (len(loop_slice) > 1 or not _is_vacuous_attr_return(stmt, subject_dump))
        ):
            rewritten_slice = _rewrite_loop_continues_to_returns(loop_slice)
            return NonNoneReturnAbsenceBranchSite(
                subject=subject,
                probe_nodes=_all_subject_probes_in(loop_slice, subject_dump)
                or probe_nodes,
                method_body=(
                    *rewritten_slice,
                    ast.Return(value=ast.Constant(value=None)),
                ),
                is_whole_function=False,
                replaced_nodes=loop_slice,
            )
    if (
        allow_found_walrus
        and _always_terminates(stmt.body)
        and _has_value_return(stmt.body)
        and not _has_break_or_continue(stmt.body)
        and not _has_break_or_yield(stmt.body)
        and _all_returns_proved_non_none((stmt,), subject_dump, non_none_callees)
        and not _is_vacuous_attr_return(stmt, subject_dump)
    ):
        return NonNoneReturnAbsenceBranchSite(
            subject=subject,
            probe_nodes=probe_nodes,
            method_body=(stmt, ast.Return(value=ast.Constant(value=None))),
            is_whole_function=False,
            replaced_nodes=(stmt,),
        )
    if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
        # Coalesce contiguous sibling void statements probing the same subject (e.g. two
        # sequential `if call.a is not None: ...; if call.b is not None: ...` checks).
        subject_name = subject.id if isinstance(subject, ast.Name) else None
        end_idx = index
        for candidate_idx in range(index + 1, len(container)):
            sibling = container[candidate_idx]
            if _has_any_return((sibling,)) or _has_loop_escape((sibling,)):
                break
            if subject_name is not None and subject_name in _names_stored_in((sibling,)):
                break
            if _all_subject_probes_in((sibling,), subject_dump):
                end_idx = candidate_idx
        replaced = tuple(container[index : end_idx + 1])
        is_whole = (
            container is whole_fn_body
            and index == 0
            and end_idx == len(whole_fn_body) - 1
        )
        return VoidAbsenceBranchSite(
            subject=subject,
            probe_nodes=_all_subject_probes_in(replaced, subject_dump) or probe_nodes,
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
    target_subject_dumps = frozenset(ast.dump(probe.subject) for probe in all_probes)
    if len(target_subject_dumps) != 1:
        return None
    (target_dump,) = target_subject_dumps
    match_index: int | None = None
    leading: ParsedAbsenceProbe | None = None
    for idx, value in enumerate(test.values[1:], start=1):
        candidate = _leading_unconditional_probe(value)
        if candidate is None or ast.dump(candidate.subject) != target_dump:
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
    allow_found_walrus: bool = False,
    non_none_callees: frozenset[str] = frozenset(),
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
    subject_dump = ast.dump(subject)
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

    if isinstance(op, ast.And):
        if stmt.orelse:
            if (
                _always_terminates(stmt.body)
                and _has_any_return(stmt.body)
                and (fallback := _trivial_pure_return(stmt.orelse)) is not None
            ):
                return PrefixGuardedAndReturningAbsenceBranchSite(
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
                    replaced_nodes=(stmt,),
                    guard_expr=guard_expr,
                    fallback_return=fallback,
                )
            return None
        tail = container[index + 1 :]
        if (
            tail
            and _always_terminates(stmt.body)
            and _has_any_return(stmt.body)
            and (fallback := _trivial_pure_return(tail)) is not None
        ):
            return PrefixGuardedAndReturningAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
                    *tail,
                ),
                is_whole_function=False,
                replaced_nodes=tuple(container[index:]),
                guard_expr=guard_expr,
                fallback_return=fallback,
            )
        if (
            allow_found_walrus
            and _always_terminates(stmt.body)
            and _has_value_return(stmt.body)
            and not _has_break_or_continue(stmt.body)
            and not _has_break_or_yield(stmt.body)
            and _all_returns_proved_non_none(
                (ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),),
                subject_dump,
                non_none_callees,
            )
            and not _is_vacuous_attr_return(stmt, subject_dump)
        ):
            return PrefixGuardedAndNonNoneReturnAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(test=remaining_test, body=list(stmt.body), orelse=[]),
                    ast.Return(value=ast.Constant(value=None)),
                ),
                is_whole_function=False,
                replaced_nodes=(stmt,),
                guard_expr=guard_expr,
            )
        if (
            in_loop_body
            and not _has_any_return(stmt.body)
            and not _has_non_continue_loop_escape(stmt.body)
            and _has_break_or_continue(stmt.body)
        ):
            return PrefixGuardedAndLoopContinueAbsenceBranchSite(
                subject=subject,
                probe_nodes=probe_nodes,
                method_body=(
                    ast.If(
                        test=remaining_test,
                        body=list(_rewrite_loop_continues_to_true(stmt.body)),
                        orelse=[],
                    ),
                    ast.Return(value=ast.Constant(value=False)),
                ),
                is_whole_function=False,
                replaced_nodes=(stmt,),
                guard_expr=guard_expr,
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
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


def _is_trivial_short_circuit_expr(expr: ast.expr, subject: ast.expr) -> bool:
    """Whether `expr` is a literal or non-subject local name safe at a short-circuiting `IfExp`."""

    if _is_trivial_constant_expr(expr):
        return True
    return isinstance(expr, ast.Name) and ast.dump(expr) != ast.dump(subject)


def _extract_prefix_guarded_absence_ternary_site(
    node: ast.IfExp,
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
    allow_found_walrus: bool = False,
    non_none_callees: frozenset[str] = frozenset(),
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
    if isinstance(op, ast.Or) and _is_trivial_short_circuit_expr(
        node.body, probes[0].subject
    ):
        return PrefixGuardedOrTernaryAbsenceBranchSite(
            subject=probes[0].subject,
            probe_nodes=tuple(probe.node for probe in probes),
            method_body=(ast.Return(value=inner_ifexp),),
            is_whole_function=False,
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.body,
        )
    if isinstance(op, ast.And) and _is_trivial_short_circuit_expr(
        node.orelse, probes[0].subject
    ):
        return PrefixGuardedAndTernaryAbsenceBranchSite(
            subject=probes[0].subject,
            probe_nodes=tuple(probe.node for probe in probes),
            method_body=(ast.Return(value=inner_ifexp),),
            is_whole_function=False,
            node=node,
            guard_expr=guard_expr,
            short_circuit_expr=node.orelse,
        )
    if isinstance(op, ast.And) and allow_found_walrus:
        subject = probes[0].subject
        subject_dump = ast.dump(subject)
        guard_dump = ast.dump(guard_expr)
        non_none_dumps = _is_not_none_guard_dumps(remaining_test)
        if _expr_proved_non_none(
            node.body, non_none_dumps, subject_dump, non_none_callees
        ) and not _is_bare_subject_attr(node.body, subject_dump):
            # Fold chained nested `B if SAME_GUARD and subj.b is [not] None else C` arms
            # into one owner method so all probes on `subj` in the chain collapse together.
            chain_arms: list[tuple[ast.expr, ast.expr]] = [(remaining_test, node.body)]
            chain_probes: list[ast.Compare] = [probe.node for probe in probes]
            tail_expr: ast.expr = node.orelse
            while isinstance(tail_expr, ast.IfExp):
                next_lines = frozenset(
                    range(
                        tail_expr.test.lineno,
                        (tail_expr.test.end_lineno or tail_expr.test.lineno) + 1,
                    )
                )
                next_split = _split_prefix_guarded_boolop(
                    tail_expr.test, next_lines, subject_source=subject_source
                )
                if next_split is None:
                    break
                next_op, next_guard, next_rem, next_probes = next_split
                if (
                    not isinstance(next_op, ast.And)
                    or ast.dump(next_guard) != guard_dump
                    or ast.dump(next_probes[0].subject) != subject_dump
                ):
                    break
                next_non_none = _is_not_none_guard_dumps(next_rem)
                if not _expr_proved_non_none(
                    tail_expr.body, next_non_none, subject_dump, non_none_callees
                ) or _is_bare_subject_attr(tail_expr.body, subject_dump):
                    break
                chain_arms.append((next_rem, tail_expr.body))
                chain_probes.extend(probe.node for probe in next_probes)
                tail_expr = tail_expr.orelse
            method_expr: ast.expr = ast.Constant(value=None)
            for arm_test, arm_body in reversed(chain_arms):
                method_expr = ast.IfExp(
                    test=arm_test, body=arm_body, orelse=method_expr
                )
            return PrefixGuardedAndNonNoneTernaryAbsenceBranchSite(
                subject=subject,
                probe_nodes=tuple(chain_probes),
                method_body=(ast.Return(value=method_expr),),
                is_whole_function=False,
                node=node,
                guard_expr=guard_expr,
                short_circuit_expr=tail_expr,
            )
    return None


def _extract_composite_ternary_site(
    function: FunctionNode,
    body: list[ast.stmt],
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
) -> CompositeTernaryAbsenceBranchSite | None:
    """Locate the smallest enclosing composite expression (`Tuple`/`List`/`Dict`/`Call`)
    containing >=2 non-boolean `IfExp`s probing the same subject when one sits on `target_lines`."""

    whole_return_expr: ast.expr | None = None
    if len(body) == 1 and isinstance(body[0], ast.Return):
        whole_return_expr = body[0].value

    best: tuple[int, CompositeTernaryAbsenceBranchSite] | None = None
    for node in ast.walk(function):
        if not isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Call)):
            continue
        if any(
            isinstance(
                sub,
                (
                    ast.NamedExpr,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            )
            for sub in ast.walk(node)
        ):
            continue
        ifexps = [sub for sub in ast.walk(node) if isinstance(sub, ast.IfExp)]
        if len(ifexps) < 2:
            continue
        matched_subject: ast.expr | None = None
        matched_dump: str | None = None
        probe_nodes: list[ast.Compare] = []
        valid_ifexp_count = 0
        hits_target = False
        valid = True
        for ifexp in ifexps:
            if (
                (_is_bool_constant(ifexp.body) and _is_bool_constant(ifexp.orelse))
                or _is_zero_one_tuple_pair(ifexp.body, ifexp.orelse)
            ):
                continue
            leading = _leading_unconditional_probe(ifexp.test)
            if leading is None:
                continue
            if subject_source is not None and ast.unparse(leading.subject) != subject_source:
                continue
            cand_dump = ast.dump(leading.subject)
            if matched_dump is None:
                matched_subject = leading.subject
                matched_dump = cand_dump
            elif cand_dump != matched_dump:
                continue
            ifexp_probes = tuple(
                probe
                for sub in ast.walk(ifexp.test)
                if (probe := _parse_absence_probe(sub)) is not None
                and ast.dump(probe.subject) == matched_dump
            )
            if not ifexp_probes:
                continue
            valid_ifexp_count += 1
            for probe in ifexp_probes:
                probe_nodes.append(probe.node)
                if probe.node.lineno in target_lines:
                    hits_target = True
        if not hits_target or valid_ifexp_count < 2 or matched_subject is None or matched_dump is None:
            continue
        # Ensure every absence probe on this subject inside `node` lives in the collected `IfExp`s.
        collected_ids = {id(p) for p in probe_nodes}
        for sub in ast.walk(node):
            probe = _parse_absence_probe(sub)
            if probe is not None and ast.dump(probe.subject) == matched_dump and id(probe.node) not in collected_ids:
                valid = False
                break
        if not valid:
            continue
        span = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno
        candidate = CompositeTernaryAbsenceBranchSite(
            subject=matched_subject,
            probe_nodes=tuple(probe_nodes),
            method_body=(ast.Return(value=node),),
            is_whole_function=node is whole_return_expr,
            node=node,
        )
        if best is None or span < best[0]:
            best = (span, candidate)
    return None if best is None else best[1]


def _is_statically_bool_expr(expr: ast.expr) -> bool:
    """Whether `expr` always evaluates to `bool` (safe for `-> bool` annotation)."""

    match expr:
        case ast.Compare() | ast.UnaryOp(op=ast.Not()):
            return True
        case ast.Constant(value=bool()):
            return True
        case ast.BoolOp(values=values):
            return all(_is_statically_bool_expr(val) for val in values)
        case ast.Call(func=ast.Name(id="bool" | "isinstance" | "issubclass" | "any" | "all")):
            return True
    return False


def _boolop_is_domain_predicate(expr: ast.BoolOp, subject: ast.expr) -> bool:
    """True when `expr` pairs an absence probe with real domain logic rather than a
    zero-arg pure own-field boolean (which would relocate into OwnStateBranchAnswer)."""

    if len(expr.values) < 2:
        return False
    if any(
        isinstance(
            sub,
            (
                ast.NamedExpr,
                ast.Yield,
                ast.YieldFrom,
                ast.Lambda,
            ),
        )
        for sub in ast.walk(expr)
    ):
        return False
    subject_dump = ast.dump(subject)

    def _has_external_work(node: ast.AST) -> bool:
        if isinstance(node, ast.expr) and ast.dump(node) == subject_dump:
            return False
        match node:
            case ast.Call(func=ast.Name(id=name)) if name not in PURE_BUILTINS:
                return True
            case ast.Call(func=ast.Attribute()):
                return True
            case ast.Name(id=name, ctx=ast.Load()) if name not in PURE_BUILTINS:
                return True
        return any(_has_external_work(child) for child in ast.iter_child_nodes(node))

    return _has_external_work(expr)


def _extract_boolop_absence_site(
    function: FunctionNode,
    body: list[ast.stmt],
    target_lines: frozenset[int],
    *,
    subject_source: str | None = None,
) -> BoolOpAbsenceBranchSite | None:
    """Locate a pushable compound `BoolOp` (optionally prefix-guarded) probing `subject.attr is [not] None`."""

    whole_return_expr: ast.expr | None = None
    if len(body) == 1 and isinstance(body[0], ast.Return):
        whole_return_expr = body[0].value

    for node in ast.walk(function):
        if not isinstance(node, ast.BoolOp) or len(node.values) < 2:
            continue
        probes = _unconditional_absence_probes_in(
            node, target_lines, subject_source=subject_source
        )
        if probes and _boolop_is_domain_predicate(node, probes[0].subject):
            site = BoolOpAbsenceBranchSite(
                subject=probes[0].subject,
                probe_nodes=tuple(probe.node for probe in probes),
                method_body=(ast.Return(value=node),),
                is_whole_function=node is whole_return_expr,
                node=node,
                is_bool_expr=_is_statically_bool_expr(node),
            )
            if site.subject_allowed():
                return site
        split = _split_prefix_guarded_boolop(
            node, target_lines, subject_source=subject_source
        )
        if split is not None:
            op, guard_expr, remaining_test, split_probes = split
            if isinstance(remaining_test, ast.BoolOp) and _boolop_is_domain_predicate(
                remaining_test, split_probes[0].subject
            ):
                guarded_site = PrefixGuardedBoolOpAbsenceBranchSite(
                    subject=split_probes[0].subject,
                    probe_nodes=tuple(probe.node for probe in split_probes),
                    method_body=(ast.Return(value=remaining_test),),
                    is_whole_function=False,
                    node=node,
                    is_bool_expr=_is_statically_bool_expr(remaining_test),
                    guard_expr=guard_expr,
                    op_keyword="and" if isinstance(op, ast.And) else "or",
                )
                if guarded_site.subject_allowed():
                    return guarded_site
    return None


def _enclosing_comprehension_locals(
    function: FunctionNode,
    target: ast.AST,
    subject: ast.expr,
) -> frozenset[str]:
    """Return comprehension loop variables in scope around `target` (excluding `subject` if a Name)."""

    target_id = id(target)
    subject_name = subject.id if isinstance(subject, ast.Name) else None
    found: set[str] = set()
    for node in ast.walk(function):
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ) and any(id(sub) == target_id for sub in ast.walk(node)):
            for gen in node.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        if sub.id != subject_name:
                            found.add(sub.id)
    return frozenset(found)


def find_absence_branch_site(
    function: FunctionNode,
    source_text: str,
    test_line: int,
    *,
    subject_source: str | None = None,
    non_none_callees: frozenset[str] = frozenset(),
) -> AbsenceBranchSite | None:
    """Locate a pushable `if`, `elif`, or non-boolean `IfExp` probing `subject.attr is [not] None` on `test_line`."""

    target_lines = frozenset((test_line,))
    allow_found_walrus = not _uses_local_name(function, "_found")
    body = _body_without_docstring(function)
    for container, index, stmt, in_loop_body in _statement_list_children(body):
        if isinstance(stmt, ast.If):
            is_elif = _line_is_elif(source_text, stmt.lineno)
            extract_container = [stmt] if is_elif else container
            extract_index = 0 if is_elif else index
            extract_in_loop = False if is_elif else in_loop_body
            probes = _unconditional_absence_probes_in(
                stmt.test, target_lines, subject_source=subject_source
            )
            extracted: AbsenceBranchSite | None = None
            if probes:
                extracted = _extract_absence_if_site(
                    extract_container,
                    extract_index,
                    stmt,
                    body,
                    probes,
                    in_loop_body=extract_in_loop,
                    allow_found_walrus=allow_found_walrus,
                    non_none_callees=non_none_callees,
                )
            if extracted is None:
                extracted = _extract_prefix_guarded_absence_if_site(
                    extract_container,
                    extract_index,
                    stmt,
                    body,
                    target_lines,
                    subject_source=subject_source,
                    in_loop_body=extract_in_loop,
                    allow_found_walrus=allow_found_walrus,
                    non_none_callees=non_none_callees,
                )
            if extracted is not None:
                if is_elif:
                    if extracted.replaced_ast_nodes() == (stmt,):
                        elif_site = ElifAbsenceBranchSite(
                            subject=extracted.subject,
                            probe_nodes=extracted.probe_nodes,
                            method_body=extracted.method_body,
                            is_whole_function=False,
                            inner=extracted,
                        )
                        if elif_site.subject_allowed():
                            return elif_site
                elif extracted.subject_allowed():
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
                if extracted is not None and extracted.subject_allowed():
                    return extracted
            extracted = _extract_prefix_guarded_absence_ternary_site(
                ifexp,
                target_lines,
                subject_source=subject_source,
                allow_found_walrus=allow_found_walrus,
                non_none_callees=non_none_callees,
            )
            if extracted is not None and extracted.subject_allowed():
                return extracted
    composite = _extract_composite_ternary_site(
        function, body, target_lines, subject_source=subject_source
    )
    if composite is not None and composite.subject_allowed():
        return composite
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp):
            probes = _unconditional_absence_probes_in(
                node.test, target_lines, subject_source=subject_source
            )
            if probes:
                extracted = _extract_absence_ternary_site(
                    node, probes, is_whole_return=False
                )
                if extracted is not None and extracted.subject_allowed():
                    return extracted
            extracted = _extract_prefix_guarded_absence_ternary_site(
                node,
                target_lines,
                subject_source=subject_source,
                allow_found_walrus=allow_found_walrus,
                non_none_callees=non_none_callees,
            )
            if extracted is not None and extracted.subject_allowed():
                return extracted
    return _extract_boolop_absence_site(
        function, body, target_lines, subject_source=subject_source
    )


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
            or variadic_parameters(args)
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
    def owner_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        owner_geometry: SourceTextGeometry,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
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

    def owner_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        owner_geometry: SourceTextGeometry,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        del proof, snapshot, owner_geometry, rationale
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

    def _resolved_annotations(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        rationale: str,
    ) -> ResolvedMethodAnnotations:
        source_module = snapshot.parsed_module_for_source_path(proof.source_path)
        return ResolvedMethodAnnotations.for_push(
            snapshot,
            source_module,
            proof.test_line,
            proof.enclosure,
            proof.function,
            proof.outer_receiver_aliases,
            proof.captured_args,
            (proof.owner,),
            propagates_function_return_type=proof.site.propagates_function_return_type,
            is_void_method=proof.site.is_void_method,
            constant_bool_return=proof.site.returns_bool,
            rationale=rationale,
        )

    def owner_edits(
        self,
        proof: ForeignAbsenceBranchProof,
        snapshot: CodemodSourceSnapshot,
        owner_geometry: SourceTextGeometry,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        params = tuple(arg.param_name for arg in proof.captured_args)
        method_annotations = self._resolved_annotations(proof, snapshot, rationale)
        method_source = _render_method(
            InstanceDispatchKind(),
            self.method_name,
            params,
            proof.method_body,
            is_async=proof.is_async,
            docstring=self.docstring,
            param_annotations=method_annotations.param_map,
            return_annotation=method_annotations.return_annotation,
        )
        return _class_append_edits(
            proof.owner.file_path,
            owner_geometry,
            proof.owner.node,
            method_source,
            member_names=(self.method_name,),
            rationale=rationale,
        )

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
        method_annotations = self._resolved_annotations(proof, snapshot, rationale)
        return (
            *method_annotations.import_edits,
            *_cross_module_imports_for_body(
                snapshot,
                source_module,
                proof.test_line,
                proof.owner,
                proof.method_body,
                local_exclusions,
                rationale=rationale,
            ),
        )

    def rationale(self, proof: ForeignAbsenceBranchProof) -> str:
        return (
            f"Push foreign absence branch on `{ast.unparse(proof.site.subject)}` in "
            f"`{proof.function.name}` onto `{proof.owner.qualname}.{self.method_name}`."
        )


def _upgrade_to_assigning_site(
    function: FunctionNode,
    site: AbsenceBranchSite,
    escaping: Sequence[str],
) -> AbsenceBranchSite | None:
    """Promote a void absence branch assigning one escaping local into `target = subject.method(...)`."""

    if len(escaping) != 1:
        return None
    target_name = escaping[0]
    if not site.method_body or not isinstance(site.method_body[0], ast.If):
        return None
    if any(
        isinstance(node, (ast.Global, ast.Nonlocal)) and target_name in node.names
        for node in ast.walk(function)
    ):
        return None
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Del)
        and node.id == target_name
        for stmt in site.method_body
        for node in ast.walk(stmt)
    ):
        return None

    is_subject_name = isinstance(site.subject, ast.Name) and site.subject.id == target_name
    replaced = site.replaced_ast_nodes()
    if len(site.method_body) > 1:
        if (
            not is_subject_name
            and replaced
            and _is_definitely_bound_before(function, replaced[0], target_name)
        ):
            new_body = (
                *site.method_body,
                ast.Return(value=ast.Name(id=target_name, ctx=ast.Load())),
            )
            return site.with_assigning_target(
                target_name, new_body, reassigned_target=target_name
            )
        return None
    if_stmt = site.method_body[0]

    # Definite single-target assignment across all arms of an `if ... else:` without pre-reads.
    if (
        if_stmt.orelse
        and not is_subject_name
        and _definitely_assigns_in_block((if_stmt,), target_name)
        and not _reads_before_assign_in_block((if_stmt,), target_name)
    ):
        new_body = (if_stmt, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load())))
        return site.with_assigning_target(target_name, new_body, reassigned_target=None)

    # Single-branch conditional rebind or augassign on an already-bound local.
    if (
        not if_stmt.orelse
        and replaced
        and _is_definitely_bound_before(function, replaced[0], target_name)
    ):
        body = if_stmt.body
        if (
            body
            and isinstance(body[-1], ast.Assign)
            and len(body[-1].targets) == 1
            and isinstance(body[-1].targets[0], ast.Name)
            and body[-1].targets[0].id == target_name
            and target_name not in _names_stored_in(body[:-1])
        ):
            rewritten_if = ast.If(
                test=if_stmt.test,
                body=[*body[:-1], ast.Return(value=body[-1].value)],
                orelse=[],
            )
            new_body = (
                rewritten_if,
                ast.Return(value=ast.Name(id=target_name, ctx=ast.Load())),
            )
            return site.with_assigning_target(target_name, new_body, reassigned_target=None)
        if not is_subject_name:
            new_body = (if_stmt, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load())))
            return site.with_assigning_target(
                target_name, new_body, reassigned_target=target_name
            )
    return None


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
    enclosure: FunctionEnclosure = ModuleEnclosure()
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = ()

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
        source_path, enclosure, function = TypeTestDispatchProof.locate_function(
            snapshot, target, test_line=test_line
        )
        source_text = snapshot.sources_by_file_path[source_path]
        non_none_callees = _non_none_callee_names(
            snapshot, source_path, enclosure, function
        )
        site = find_absence_branch_site(
            function,
            source_text,
            test_line,
            subject_source=subject_source,
            non_none_callees=non_none_callees,
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
                upgraded = _upgrade_to_assigning_site(function, site, escaping)
                if upgraded is not None:
                    site = upgraded
                else:
                    raise NeedsInput(
                        f"`{function.name}` assigns outer local(s) {', '.join(escaping)} inside the absence branch",
                        NeededInput.SOURCE_CHANGE,
                    )

        adapter = site.as_dispatch_adapter()
        enclosing_locals = enclosure.enclosing_local_names() | _enclosing_comprehension_locals(
            function, site.replaced_ast_nodes()[0], site.subject
        )
        outer_receiver_aliases = _outer_receiver_aliases(
            function, adapter, (site.method_body,), enclosing_locals=enclosing_locals
        )
        rewritten_body = _rewrite_subject_to_receiver(
            site.method_body,
            site.subject,
            "self",
            outer_receiver_aliases=outer_receiver_aliases,
        )
        captured_args, is_async = _analyze_arm_bodies(
            function,
            adapter,
            (rewritten_body,),
            outer_receiver_aliases=outer_receiver_aliases,
            enclosing_locals=enclosing_locals,
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
            enclosure=enclosure,
            outer_receiver_aliases=outer_receiver_aliases,
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
        owner_edits = self.method_target.owner_edits(
            self, snapshot, owner_geometry, rationale
        )

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

        consumer_edit = consumer_geometry.nominal_edit(
            file_path=self.source_path,
            replacements=(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=rep_start,
                    end_offset=rep_end,
                    replacement_source=replacement,
                ),
            ),
            rationale=rationale,
        )
        return (*owner_edits, consumer_edit, *import_edits)


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


def _module_imports_ast(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            if any(alias.name in ("ast", "_ast") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("ast", "_ast"):
                return True
    return False


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
            if ast_evaluation.needs == (NeededInput.OPERATION,) or not any(
                _module_imports_ast(
                    snapshot.parsed_module_for_source_path(ev.file_path).module
                )
                for ev in finding.evidence
            ):
                subject = (
                    metrics.subject_source
                    if metrics is not None and metrics.subject_source
                    else "receiver"
                )
                return self.stopped_evaluation(
                    NeedsInput(
                        f"probed receiver `{subject}` has no resolved scanned owner class",
                        NeededInput.SOURCE_CHANGE,
                    )
                )
            return ast_evaluation
        classes = snapshot.compact_class_index.classes_by_symbol
        if owner_symbol not in classes:
            if owner_symbol.startswith(("ast.", "_ast.")):
                return ast_evaluation
            return self.stopped_evaluation(
                NeedsInput(
                    f"owner `{owner_symbol}` is outside the scanned repository",
                    NeededInput.SOURCE_CHANGE,
                )
            )

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
                snapshot, target, test_line=probe_lines[0]
            )
        except ValueError as error:
            return self.stopped_evaluation(error)
        source_text = snapshot.sources_by_file_path[primary.file_path]
        non_none_callees = _non_none_callee_names(
            snapshot, primary.file_path, _enclosure, function
        )
        site = find_absence_branch_site(
            function,
            source_text,
            probe_lines[0],
            subject_source=subject_source,
            non_none_callees=non_none_callees,
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
    "AssigningAbsenceBranchSite",
    "ForeignAbsenceBranchProof",
    "ForeignAbsenceProbeFindingRecipeSynthesizer",
    "ParsedAbsenceProbe",
    "PrefixGuardedAndAssigningAbsenceBranchSite",
    "PushForeignAbsenceBranchOperation",
    "ReturningAbsenceBranchSite",
    "TernaryAbsenceBranchSite",
    "VoidAbsenceBranchSite",
    "find_absence_branch_site",
)
