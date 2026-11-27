"""Replace a threaded `T | None` parameter with a `Held` / `Absent` nominal family.

When a function or method accepts `param: T | None`, probes `param is [not] None`
inside its own body, and all call sites are proved closed, introducing a nominal
family (`Base`, `AbsentBase`, `HeldBase(value: T)`) moves the two branch bodies
onto the family and replaces every call site's `None` / value argument with the
family member decided once by the caller.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.ast_field_decoders import (
    declared_annotations,
    optional_values,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
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
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.models import (
    RefactorFinding,
    ThreadedAbsenceMetrics,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import ReceiverSpan
from nominal_refactor_advisor.refactor_concepts import ThreadedAbsenceFamilyConcept
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
)
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    CapturedArgument,
    DispatchSite,
    FunctionEnclosure,
    FunctionNode,
    InstanceDispatchKind,
    OuterReceiverAlias,
    ResolvedMethodAnnotations,
    TypeTestDispatchProof,
    _always_terminates,
    _always_void_terminates,
    _analyze_arm_bodies,
    _body_without_docstring,
    _definitely_assigns_in_block,
    _ends_with_continue_only,
    _has_any_return,
    _has_break_or_continue,
    _has_future_annotations,
    _has_loop_escape,
    _has_non_continue_loop_escape,
    _has_value_return,
    _is_definitely_bound_before,
    _names_referenced_outside,
    _names_stored_in,
    _outer_receiver_aliases,
    _reads_before_assign_in_block,
    _render_indented_statements,
    _render_method,
    _rewrite_loop_continues_to_returns,
    _statement_list_children,
)


def _node_covers_line(node: ast.AST, line: int) -> bool:
    start = getattr(node, "lineno", None)
    if start is None:
        return False
    end = getattr(node, "end_lineno", start) or start
    return start <= line <= end


def _is_none_const(node: ast.AST) -> bool:
    match node:
        case ast.Constant(value=None):
            return True
    return False


def _parse_param_probe(
    node: ast.AST, param_name: str
) -> tuple[bool, ast.Compare] | None:
    """Match `param is [not] None` or `None is [not] param` -> `(is_none, node)`."""

    match node:
        case ast.Compare(
            left=ast.Name(id=name),
            ops=[ast.Is() | ast.IsNot() as op],
            comparators=[comp],
        ) if name == param_name and _is_none_const(comp):
            return (isinstance(op, ast.Is), node)
        case ast.Compare(
            left=comp,
            ops=[ast.Is() | ast.IsNot() as op],
            comparators=[ast.Name(id=name)],
        ) if name == param_name and _is_none_const(comp):
            return (isinstance(op, ast.Is), node)
    return None


def _strip_none_from_annotation(annotation: ast.expr) -> ast.expr | None:
    """Remove `None` from `X | None`, `Optional[X]`, or `Union[X, None]`."""

    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        parts: list[ast.expr] = []

        def collect(expr: ast.expr) -> None:
            if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
                collect(expr.left)
                collect(expr.right)
            else:
                parts.append(expr)

        collect(annotation)
        non_none = [p for p in parts if not _is_none_const(p)]
        if len(non_none) == len(parts) or not non_none:
            return None
        result = copy.deepcopy(non_none[0])
        for extra in non_none[1:]:
            result = ast.BinOp(left=result, op=ast.BitOr(), right=copy.deepcopy(extra))
        return result
    if isinstance(annotation, ast.Subscript):
        base_name = (
            annotation.value.id
            if isinstance(annotation.value, ast.Name)
            else (
                annotation.value.attr
                if isinstance(annotation.value, ast.Attribute)
                else None
            )
        )
        if base_name == "Optional":
            return copy.deepcopy(annotation.slice)
        if base_name == "Union" and isinstance(annotation.slice, ast.Tuple):
            elts = [e for e in annotation.slice.elts if not _is_none_const(e)]
            if len(elts) == len(annotation.slice.elts) or not elts:
                return None
            if len(elts) == 1:
                return copy.deepcopy(elts[0])
            return ast.Subscript(
                value=copy.deepcopy(annotation.value),
                slice=ast.Tuple(
                    elts=[copy.deepcopy(e) for e in elts], ctx=ast.Load()
                ),
                ctx=ast.Load(),
            )
    return None


def _is_noop_body(statements: Sequence[ast.stmt]) -> bool:
    if not statements:
        return True
    if len(statements) != 1:
        return False
    match statements[0]:
        case ast.Pass():
            return True
        case ast.Return(value=None):
            return True
        case ast.Return(value=ast.Constant(value=None)):
            return True
    return False


def _rewrite_param_to_self_value(
    statements: tuple[ast.stmt, ...],
    param_name: str,
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...],
) -> tuple[ast.stmt, ...]:
    outer_map = {item.source_name: item.param_name for item in outer_receiver_aliases}

    class _Rewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load):
                if node.id == param_name:
                    return ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="value",
                        ctx=ast.Load(),
                    )
                if node.id in outer_map:
                    return ast.Name(id=outer_map[node.id], ctx=ast.Load())
            return node

    rewriter = _Rewriter()
    return tuple(
        ast.fix_missing_locations(rewriter.visit(copy.deepcopy(stmt)))
        for stmt in statements
    )


def _rewrite_outer_aliases_only(
    statements: tuple[ast.stmt, ...],
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...],
) -> tuple[ast.stmt, ...]:
    if not outer_receiver_aliases:
        return tuple(copy.deepcopy(stmt) for stmt in statements)
    outer_map = {item.source_name: item.param_name for item in outer_receiver_aliases}

    class _Rewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in outer_map:
                return ast.Name(id=outer_map[node.id], ctx=ast.Load())
            return node

    rewriter = _Rewriter()
    return tuple(
        ast.fix_missing_locations(rewriter.visit(copy.deepcopy(stmt)))
        for stmt in statements
    )


def _name_used_in(name: str, nodes: Sequence[ast.AST]) -> bool:
    for root in nodes:
        for child in ast.walk(root):
            if isinstance(child, ast.Name) and child.id == name:
                return True
    return False


def _strip_trailing_bare_return(
    statements: Sequence[ast.stmt],
) -> tuple[ast.stmt, ...]:
    if statements and isinstance(statements[-1], ast.Return):
        if not optional_values(statements[-1]):
            return tuple(statements[:-1])
    return tuple(statements)


def _strip_trailing_continue(statements: Sequence[ast.stmt]) -> tuple[ast.stmt, ...]:
    if statements and isinstance(statements[-1], ast.Continue):
        return tuple(statements[:-1])
    return tuple(statements)


@dataclass(frozen=True)
class _ThreadedDispatchAdapter(DispatchSite):
    """Adapter letting `type_test_dispatch` capture/alias helpers operate on a threaded site."""

    _reassigned_target: str | None = None

    @property
    def reassigned_target(self) -> str | None:
        return self._reassigned_target

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class ThreadedAbsenceBranchSite(ABC):
    """One `param is [not] None` branch normalized into absent and held method bodies."""

    param_name: str
    probe_node: ast.Compare
    absent_body: tuple[ast.stmt, ...]
    held_body: tuple[ast.stmt, ...]
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
    def reassigned_target(self) -> str | None:
        return None

    def with_assigning_target(
        self,
        target_name: str,
        absent_body: tuple[ast.stmt, ...],
        held_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> ThreadedAbsenceBranchSite | None:
        del target_name, absent_body, held_body, reassigned_target
        return None

    def as_dispatch_adapter(self) -> DispatchSite:
        return _ThreadedDispatchAdapter(
            subject=ast.Name(id=self.param_name, ctx=ast.Load()),
            kind=InstanceDispatchKind(),
            arms=(),
            default_body=(),
            is_whole_function=self.is_whole_function,
            test_nodes=(self.probe_node,),
            _reassigned_target=self.reassigned_target,
        )


@dataclass(frozen=True)
class ReturningThreadedSite(ThreadedAbsenceBranchSite):
    """Returning `if param is [not] None:` branch replaced by `return param.method(...)`."""

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
class VoidThreadedSite(ThreadedAbsenceBranchSite):
    """Void or raising `if param is [not] None:` branch replaced by `param.method(...)`."""

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
        absent_body: tuple[ast.stmt, ...],
        held_body: tuple[ast.stmt, ...],
        *,
        reassigned_target: str | None = None,
    ) -> ThreadedAbsenceBranchSite | None:
        return AssigningThreadedSite(
            param_name=self.param_name,
            probe_node=self.probe_node,
            absent_body=absent_body,
            held_body=held_body,
            is_whole_function=False,
            replaced_nodes=self.replaced_nodes,
            target_name=target_name,
            _reassigned_target=reassigned_target,
        )


@dataclass(frozen=True)
class AssigningThreadedSite(ThreadedAbsenceBranchSite):
    """Branch computing or updating one local, replaced by `target = param.method(...)`."""

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
class TernaryThreadedSite(ThreadedAbsenceBranchSite):
    """Inline `a if param is [not] None else b` or compound boolean expression replaced by `param.method(...)`."""

    node: ast.expr

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        return call_expr

    def replaced_ast_nodes(self) -> tuple[ast.AST, ...]:
        return (self.node,)


@dataclass(frozen=True)
class PrefixGuardedAndThreadedSite(ThreadedAbsenceBranchSite):
    """Void `if PREFIX and param is [not] None: body` replaced by
    `if PREFIX:\\n    param.method(...)`."""

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

    @property
    def propagates_function_return_type(self) -> bool:
        return False

    @property
    def is_void_method(self) -> bool:
        return True


@dataclass(frozen=True)
class PrefixGuardedOrThreadedSite(ThreadedAbsenceBranchSite):
    """`if PREFIX or param is [not] None: early_exit` replaced by
    `if PREFIX:\\n    <early_exit>\\n[return ]param.method(...)`."""

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

    @property
    def propagates_function_return_type(self) -> bool:
        return self.returns_value

    @property
    def is_void_method(self) -> bool:
        return not self.returns_value


def _extract_threaded_if_site(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    param_name: str,
    probe_line: int,
    *,
    in_loop_body: bool,
) -> ThreadedAbsenceBranchSite | None:
    direct = _parse_param_probe(stmt.test, param_name)
    if direct is not None and _node_covers_line(stmt.test, probe_line):
        is_none, probe_node = direct
        if stmt.orelse:
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            absent_body = tuple(stmt.body) if is_none else tuple(stmt.orelse)
            held_body = tuple(stmt.orelse) if is_none else tuple(stmt.body)
            if _has_any_return(replaced):
                if _always_terminates(stmt.body) and _always_terminates(stmt.orelse):
                    return ReturningThreadedSite(
                        param_name=param_name,
                        probe_node=probe_node,
                        absent_body=absent_body,
                        held_body=held_body,
                        is_whole_function=is_whole,
                        replaced_nodes=replaced,
                    )
                return None
            if _has_loop_escape(replaced):
                return None
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )

        tail = container[index + 1 :]
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
            and not _has_break_or_continue((stmt, *tail))
        ):
            replaced = tuple(container[index:])
            is_whole = container is whole_fn_body and index == 0
            hit_body = (
                tuple(stmt.body)
                if _always_terminates(stmt.body)
                else (*stmt.body, *tail)
            )
            absent_body = hit_body if is_none else tuple(tail)
            held_body = tuple(tail) if is_none else hit_body
            if _has_any_return(replaced):
                return ReturningThreadedSite(
                    param_name=param_name,
                    probe_node=probe_node,
                    absent_body=absent_body,
                    held_body=held_body,
                    is_whole_function=is_whole,
                    replaced_nodes=replaced,
                )
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            stripped = _strip_trailing_bare_return(stmt.body)
            absent_body = stripped if is_none else tuple(tail)
            held_body = tuple(tail) if is_none else stripped
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=index == 0,
                replaced_nodes=tuple(container[index:]),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            stripped = _strip_trailing_continue(stmt.body)
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            absent_body = stripped if is_none else rewritten_tail
            held_body = rewritten_tail if is_none else stripped
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=False,
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            absent_body = tuple(stmt.body) if is_none else ()
            held_body = () if is_none else tuple(stmt.body)
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                replaced_nodes=(stmt,),
            )
        return None

    # Compound BoolOp `if PREFIX and param is [not] None:` / `if param is [not] None and|or SUFFIX:`
    if not isinstance(stmt.test, ast.BoolOp) or len(stmt.test.values) < 2:
        return None
    match_idx: int | None = None
    matched_probe: tuple[bool, ast.Compare] | None = None
    for idx, value in enumerate(stmt.test.values):
        if not _node_covers_line(value, probe_line):
            continue
        parsed = _parse_param_probe(value, param_name)
        if parsed is not None:
            match_idx = idx
            matched_probe = parsed
            break
    if match_idx is None or matched_probe is None:
        return None
    is_none, probe_node = matched_probe
    prefix_values = stmt.test.values[:match_idx]
    suffix_values = stmt.test.values[match_idx + 1 :]
    if any(_name_used_in(param_name, (v,)) for v in prefix_values):
        return None
    guard_expr: ast.expr | None = (
        None
        if not prefix_values
        else (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=stmt.test.op, values=prefix_values)
        )
    )
    suffix_expr: ast.expr | None = (
        None
        if not suffix_values
        else (
            suffix_values[0]
            if len(suffix_values) == 1
            else ast.BoolOp(op=stmt.test.op, values=suffix_values)
        )
    )
    if isinstance(stmt.test.op, ast.And):
        if guard_expr is not None:
            if stmt.orelse or _has_any_return(stmt.body) or _has_loop_escape(stmt.body):
                return None
            hit_body: tuple[ast.stmt, ...] = (
                (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
                if suffix_expr is not None
                else tuple(stmt.body)
            )
            absent_body = hit_body if is_none else ()
            held_body = () if is_none else hit_body
            return PrefixGuardedAndThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=False,
                replaced_nodes=(stmt,),
                guard_expr=guard_expr,
            )
        if suffix_expr is None:
            return None
        if stmt.orelse:
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            cond_body: tuple[ast.stmt, ...] = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=list(stmt.orelse)),
            )
            absent_body = cond_body if is_none else tuple(stmt.orelse)
            held_body = tuple(stmt.orelse) if is_none else cond_body
            if _has_any_return(replaced):
                if _always_terminates(stmt.body) and _always_terminates(stmt.orelse):
                    return ReturningThreadedSite(
                        param_name=param_name,
                        probe_node=probe_node,
                        absent_body=absent_body,
                        held_body=held_body,
                        is_whole_function=is_whole,
                        replaced_nodes=replaced,
                    )
                return None
            if _has_loop_escape(replaced):
                return None
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        tail = container[index + 1 :]
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
            and not _has_break_or_continue((stmt, *tail))
        ):
            replaced = tuple(container[index:])
            is_whole = container is whole_fn_body and index == 0
            cond_body = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            absent_body = cond_body if is_none else tuple(tail)
            held_body = tuple(tail) if is_none else cond_body
            if _has_any_return(replaced):
                return ReturningThreadedSite(
                    param_name=param_name,
                    probe_node=probe_node,
                    absent_body=absent_body,
                    held_body=held_body,
                    is_whole_function=is_whole,
                    replaced_nodes=replaced,
                )
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            cond_body = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            absent_body = cond_body if is_none else tuple(tail)
            held_body = tuple(tail) if is_none else cond_body
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=index == 0,
                replaced_nodes=tuple(container[index:]),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            cond_body = (
                ast.If(
                    test=suffix_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            absent_body = cond_body if is_none else rewritten_tail
            held_body = rewritten_tail if is_none else cond_body
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=False,
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            hit_body = (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
            absent_body = hit_body if is_none else ()
            held_body = () if is_none else hit_body
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                replaced_nodes=(stmt,),
            )
        return None

    if isinstance(stmt.test.op, ast.Or):
        if guard_expr is not None:
            if stmt.orelse:
                return None
            tail = container[index + 1 :]
            if (
                tail
                and _always_terminates(stmt.body)
                and _always_terminates(tail)
                and not _has_break_or_continue((stmt, *tail))
            ):
                fallback_body: tuple[ast.stmt, ...] = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                absent_body = tuple(stmt.body) if is_none else fallback_body
                held_body = fallback_body if is_none else tuple(stmt.body)
                replaced = tuple(container[index:])
                return PrefixGuardedOrThreadedSite(
                    param_name=param_name,
                    probe_node=probe_node,
                    absent_body=absent_body,
                    held_body=held_body,
                    is_whole_function=False,
                    replaced_nodes=replaced,
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                    returns_value=_has_any_return(replaced),
                )
            return None
        if suffix_expr is None:
            return None
        if stmt.orelse:
            replaced = (stmt,)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            fallback_body = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=list(stmt.orelse)),
            )
            absent_body = tuple(stmt.body) if is_none else fallback_body
            held_body = fallback_body if is_none else tuple(stmt.body)
            if _has_any_return(replaced):
                if _always_terminates(stmt.body) and _always_terminates(stmt.orelse):
                    return ReturningThreadedSite(
                        param_name=param_name,
                        probe_node=probe_node,
                        absent_body=absent_body,
                        held_body=held_body,
                        is_whole_function=is_whole,
                        replaced_nodes=replaced,
                    )
                return None
            if _has_loop_escape(replaced):
                return None
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        tail = container[index + 1 :]
        if (
            tail
            and _always_terminates(tail)
            and (_always_terminates(stmt.body) or _has_any_return(stmt.body))
            and not _has_break_or_continue((stmt, *tail))
        ):
            replaced = tuple(container[index:])
            is_whole = container is whole_fn_body and index == 0
            hit_body = (
                tuple(stmt.body)
                if _always_terminates(stmt.body)
                else (*stmt.body, *tail)
            )
            fallback_body = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            absent_body = hit_body if is_none else fallback_body
            held_body = fallback_body if is_none else hit_body
            if _has_any_return(replaced):
                return ReturningThreadedSite(
                    param_name=param_name,
                    probe_node=probe_node,
                    absent_body=absent_body,
                    held_body=held_body,
                    is_whole_function=is_whole,
                    replaced_nodes=replaced,
                )
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=is_whole,
                replaced_nodes=replaced,
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            stripped = _strip_trailing_bare_return(stmt.body)
            fallback_body = (
                ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            absent_body = stripped if is_none else fallback_body
            held_body = fallback_body if is_none else stripped
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=index == 0,
                replaced_nodes=tuple(container[index:]),
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
                ast.If(
                    test=suffix_expr,
                    body=list(stripped),
                    orelse=list(rewritten_tail),
                ),
            )
            absent_body = stripped if is_none else fallback_body
            held_body = fallback_body if is_none else stripped
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=False,
                replaced_nodes=tuple(container[index:]),
            )
        if not _has_any_return(stmt.body) and not _has_loop_escape(stmt.body):
            fallback_body = (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
            absent_body = tuple(stmt.body) if is_none else fallback_body
            held_body = fallback_body if is_none else tuple(stmt.body)
            return VoidThreadedSite(
                param_name=param_name,
                probe_node=probe_node,
                absent_body=absent_body,
                held_body=held_body,
                is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                replaced_nodes=(stmt,),
            )
    return None


def _extract_threaded_ifexp_site(
    node: ast.IfExp,
    param_name: str,
    probe_line: int,
    *,
    is_whole_function: bool,
) -> TernaryThreadedSite | None:
    if not _node_covers_line(node.test, probe_line):
        return None
    direct = _parse_param_probe(node.test, param_name)
    if direct is not None:
        is_none, probe_node = direct
        absent_expr = node.body if is_none else node.orelse
        held_expr = node.orelse if is_none else node.body
        return TernaryThreadedSite(
            param_name=param_name,
            probe_node=probe_node,
            absent_body=(ast.Return(value=absent_expr),),
            held_body=(ast.Return(value=held_expr),),
            is_whole_function=is_whole_function,
            node=node,
        )
    if isinstance(node.test, ast.BoolOp) and len(node.test.values) >= 2:
        first_val = node.test.values[0]
        if not _node_covers_line(first_val, probe_line):
            return None
        parsed = _parse_param_probe(first_val, param_name)
        if parsed is None:
            return None
        is_none, probe_node = parsed
        suffix_values = node.test.values[1:]
        suffix_expr: ast.expr = (
            suffix_values[0]
            if len(suffix_values) == 1
            else ast.BoolOp(op=node.test.op, values=suffix_values)
        )
        deferred = ast.IfExp(test=suffix_expr, body=node.body, orelse=node.orelse)
        if isinstance(node.test.op, ast.Or):
            absent_expr = node.body if is_none else deferred
            held_expr = deferred if is_none else node.body
        else:
            absent_expr = deferred if is_none else node.orelse
            held_expr = node.orelse if is_none else deferred
        return TernaryThreadedSite(
            param_name=param_name,
            probe_node=probe_node,
            absent_body=(ast.Return(value=absent_expr),),
            held_body=(ast.Return(value=held_expr),),
            is_whole_function=is_whole_function,
            node=node,
        )
    return None


def _extract_threaded_boolop_expr_site(
    expr: ast.expr,
    param_name: str,
    probe_line: int,
    *,
    is_whole_function: bool,
) -> TernaryThreadedSite | None:
    """Match `param is [not] None and|or SUFFIX` or `bool(param is [not] None and|or SUFFIX)`."""

    wrapped_bool = False
    target_boolop: ast.BoolOp | None = None
    match expr:
        case ast.Call(
            func=ast.Name(id="bool"),
            args=[ast.BoolOp() as inner_op],
            keywords=[],
        ):
            wrapped_bool = True
            target_boolop = inner_op
        case ast.BoolOp() as inner_op:
            target_boolop = inner_op
    if target_boolop is None or len(target_boolop.values) < 2:
        return None
    first_val = target_boolop.values[0]
    if not _node_covers_line(first_val, probe_line):
        return None
    parsed = _parse_param_probe(first_val, param_name)
    if parsed is None:
        return None
    is_none, probe_node = parsed
    suffix_values = target_boolop.values[1:]
    suffix_expr: ast.expr = (
        suffix_values[0]
        if len(suffix_values) == 1
        else ast.BoolOp(op=target_boolop.op, values=suffix_values)
    )
    continue_expr: ast.expr = (
        ast.Call(
            func=ast.Name(id="bool", ctx=ast.Load()),
            args=[suffix_expr],
            keywords=[],
        )
        if wrapped_bool
        else suffix_expr
    )
    if isinstance(target_boolop.op, ast.Or):
        short_circuit = ast.Constant(value=True)
        absent_expr = short_circuit if is_none else continue_expr
        held_expr = continue_expr if is_none else short_circuit
    else:
        short_circuit = ast.Constant(value=False)
        absent_expr = continue_expr if is_none else short_circuit
        held_expr = short_circuit if is_none else continue_expr
    return TernaryThreadedSite(
        param_name=param_name,
        probe_node=probe_node,
        absent_body=(ast.Return(value=absent_expr),),
        held_body=(ast.Return(value=held_expr),),
        is_whole_function=is_whole_function,
        node=expr,
    )


def find_threaded_absence_site(
    function: FunctionNode,
    param_name: str,
    probe_line: int,
) -> ThreadedAbsenceBranchSite | None:
    """Locate a pushable `if`, `IfExp`, or compound `BoolOp` branching on `param_name is [not] None`."""

    body = _body_without_docstring(function)
    for container, index, stmt, in_loop_body in _statement_list_children(body):
        if isinstance(stmt, ast.If):
            extracted = _extract_threaded_if_site(
                container,
                index,
                stmt,
                body,
                param_name,
                probe_line,
                in_loop_body=in_loop_body,
            )
            if extracted is not None:
                return extracted
        if isinstance(stmt, ast.Return):
            for ret_val in optional_values(stmt):
                is_whole = container is body and len(body) == 1
                if isinstance(ret_val, ast.IfExp):
                    site = _extract_threaded_ifexp_site(
                        ret_val, param_name, probe_line, is_whole_function=is_whole
                    )
                    if site is not None:
                        return site
                bool_site = _extract_threaded_boolop_expr_site(
                    ret_val, param_name, probe_line, is_whole_function=is_whole
                )
                if bool_site is not None:
                    return bool_site
    for node in ast.walk(function):
        if isinstance(node, ast.IfExp):
            site = _extract_threaded_ifexp_site(
                node, param_name, probe_line, is_whole_function=False
            )
            if site is not None:
                return site
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bool":
            bool_site = _extract_threaded_boolop_expr_site(
                node, param_name, probe_line, is_whole_function=False
            )
            if bool_site is not None:
                return bool_site
    for node in ast.walk(function):
        if isinstance(node, ast.BoolOp):
            bool_site = _extract_threaded_boolop_expr_site(
                node, param_name, probe_line, is_whole_function=False
            )
            if bool_site is not None:
                return bool_site
    return None


def _upgrade_threaded_to_assigning(
    function: FunctionNode,
    site: ThreadedAbsenceBranchSite,
    escaping: Sequence[str],
) -> ThreadedAbsenceBranchSite | None:
    if len(escaping) != 1:
        return None
    target_name = escaping[0]
    if any(
        isinstance(node, (ast.Global, ast.Nonlocal)) and target_name in node.names
        for node in ast.walk(function)
    ):
        return None
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Del)
        and node.id == target_name
        for body in (site.absent_body, site.held_body)
        for stmt in body
        for node in ast.walk(stmt)
    ):
        return None
    both_definite = (
        bool(site.absent_body)
        and bool(site.held_body)
        and _definitely_assigns_in_block(site.absent_body, target_name)
        and _definitely_assigns_in_block(site.held_body, target_name)
        and not _reads_before_assign_in_block(site.absent_body, target_name)
        and not _reads_before_assign_in_block(site.held_body, target_name)
    )
    if both_definite:
        return site.with_assigning_target(
            target_name,
            (*site.absent_body, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load()))),
            (*site.held_body, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load()))),
            reassigned_target=None,
        )
    replaced = site.replaced_ast_nodes()
    if replaced and _is_definitely_bound_before(function, replaced[0], target_name):
        return site.with_assigning_target(
            target_name,
            (*site.absent_body, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load()))),
            (*site.held_body, ast.Return(value=ast.Name(id=target_name, ctx=ast.Load()))),
            reassigned_target=target_name,
        )
    return None


class _CallArgWrapping(ABC):
    """How one caller's argument expression is wrapped into the new nominal family."""

    @abstractmethod
    def render(self, source: str, family_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def needed_symbols(self, family_name: str) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def uses_from_optional(self) -> bool:
        return False


@dataclass(frozen=True)
class _AbsentArgWrap(_CallArgWrapping):
    """Literal `None` becomes `Absent<Family>()`."""

    def render(self, source: str, family_name: str) -> str:
        del source
        return f"Absent{family_name}()"

    def needed_symbols(self, family_name: str) -> tuple[str, ...]:
        return (f"Absent{family_name}",)


@dataclass(frozen=True)
class _HeldArgWrap(_CallArgWrapping):
    """Provably non-None expression `expr` becomes `Held<Family>(expr)`."""

    expr: ast.expr

    def render(self, source: str, family_name: str) -> str:
        geometry = SourceTextGeometry(source)
        start, end = geometry.required_node_offsets(self.expr)
        return f"Held{family_name}({source[start:end]})"

    def needed_symbols(self, family_name: str) -> tuple[str, ...]:
        return (f"Held{family_name}",)


@dataclass(frozen=True)
class _ConditionalArgWrap(_CallArgWrapping):
    """`A if cond else None` or `None if cond else A` becomes `Held(A) if cond else Absent()`."""

    test: ast.expr
    present_expr: ast.expr
    absent_when_true: bool

    def render(self, source: str, family_name: str) -> str:
        geometry = SourceTextGeometry(source)
        t_start, t_end = geometry.required_node_offsets(self.test)
        p_start, p_end = geometry.required_node_offsets(self.present_expr)
        test_src = source[t_start:t_end]
        held_src = f"Held{family_name}({source[p_start:p_end]})"
        absent_src = f"Absent{family_name}()"
        if self.absent_when_true:
            return f"{absent_src} if {test_src} else {held_src}"
        return f"{held_src} if {test_src} else {absent_src}"

    def needed_symbols(self, family_name: str) -> tuple[str, ...]:
        return (f"Absent{family_name}", f"Held{family_name}")


@dataclass(frozen=True)
class _FromOptionalArgWrap(_CallArgWrapping):
    """Dynamic `T | None` expression becomes `<Family>.from_optional(expr)`."""

    expr: ast.expr

    def render(self, source: str, family_name: str) -> str:
        geometry = SourceTextGeometry(source)
        start, end = geometry.required_node_offsets(self.expr)
        return f"{family_name}.from_optional({source[start:end]})"

    def needed_symbols(self, family_name: str) -> tuple[str, ...]:
        return (family_name,)

    @property
    def uses_from_optional(self) -> bool:
        return True


@dataclass(frozen=True)
class _RewrittenCallArg:
    file_path: str
    arg_expr: ast.expr
    wrapping: _CallArgWrapping


def _classify_call_args_batched(
    snapshot: CodemodSourceSnapshot,
    raw_args: Sequence[tuple[str, ast.expr]],
) -> tuple[_RewrittenCallArg, ...]:
    """Classify all caller argument expressions with at most one batched type-checker query."""

    pending_spans: list[ReceiverSpan] = []
    span_by_index: dict[int, ReceiverSpan] = {}
    pre_wrapped: dict[int, _CallArgWrapping] = {}
    for idx, (file_path, expr) in enumerate(raw_args):
        if _is_none_const(expr):
            pre_wrapped[idx] = _AbsentArgWrap()
        elif isinstance(expr, ast.IfExp) and _is_none_const(expr.orelse) and not _is_none_const(expr.body):
            pre_wrapped[idx] = _ConditionalArgWrap(
                test=expr.test, present_expr=expr.body, absent_when_true=False
            )
        elif isinstance(expr, ast.IfExp) and _is_none_const(expr.body) and not _is_none_const(expr.orelse):
            pre_wrapped[idx] = _ConditionalArgWrap(
                test=expr.test, present_expr=expr.orelse, absent_when_true=True
            )
        elif isinstance(
            expr,
            (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.JoinedStr),
        ):
            pre_wrapped[idx] = _HeldArgWrap(expr=expr)
        else:
            span = ReceiverSpan.of(file_path, expr)
            span_by_index[idx] = span
            pending_spans.append(span)

    inferred = (
        snapshot.type_checker.receiver_types(tuple(dict.fromkeys(pending_spans)))
        if pending_spans
        else {}
    )
    results: list[_RewrittenCallArg] = []
    for idx, (file_path, expr) in enumerate(raw_args):
        wrapping = pre_wrapped.get(idx)
        if wrapping is None:
            answer = inferred.get(span_by_index[idx])
            if answer is not None and answer.is_provably_non_none():
                wrapping = _HeldArgWrap(expr=expr)
            else:
                wrapping = _FromOptionalArgWrap(expr=expr)
        results.append(
            _RewrittenCallArg(file_path=file_path, arg_expr=expr, wrapping=wrapping)
        )
    return tuple(results)


def _locate_parameter(
    function: FunctionNode,
    param_name: str,
    *,
    skip_first_receiver: bool,
) -> tuple[ast.arg, ast.expr, int | None, ast.expr | None]:
    """Return `(arg_node, annotation, positional_index_for_call, default_expr_or_none)`."""

    args = function.args
    positional = (*args.posonlyargs, *args.args)
    defaults_offset = len(positional) - len(args.defaults)
    for idx, arg in enumerate(positional):
        if arg.arg != param_name:
            continue
        anns = tuple(declared_annotations(arg))
        if not anns:
            raise NeedsInput(
                f"parameter `{param_name}` of `{function.name}` has no type annotation",
                NeededInput.PROOF,
            )
        call_pos = idx - 1 if skip_first_receiver else idx
        if call_pos < 0:
            raise NeedsInput(
                f"`{param_name}` is the receiver parameter of `{function.name}`",
                NeededInput.PROOF,
            )
        default_expr = (
            args.defaults[idx - defaults_offset] if idx >= defaults_offset else None
        )
        return arg, anns[0], call_pos, default_expr
    for idx, arg in enumerate(args.kwonlyargs):
        if arg.arg != param_name:
            continue
        anns = tuple(declared_annotations(arg))
        if not anns:
            raise NeedsInput(
                f"parameter `{param_name}` of `{function.name}` has no type annotation",
                NeededInput.PROOF,
            )
        default_expr = args.kw_defaults[idx]
        return arg, anns[0], None, default_expr
    raise NeedsInput(
        f"`{function.name}` declares no parameter `{param_name}`",
        NeededInput.SOURCE_CHANGE,
    )


def _extract_call_argument(
    call: ast.Call,
    param_name: str,
    call_pos: int | None,
    has_default: bool,
) -> ast.expr | None:
    """Return the argument passed for `param_name`, `None` if omitted with default,
    or raise `NeedsInput` if the call uses `*args`/`**kwargs` splats."""

    if any(isinstance(arg, ast.Starred) for arg in call.args) or CallKeywords.of(
        call.keywords
    ).splats:
        raise NeedsInput(
            f"call at line {call.lineno} uses `*args` or `**kwargs`; cannot prove `{param_name}` argument",
            NeededInput.PROOF,
        )
    for kw in CallKeywords.of(call.keywords).named:
        if kw.name == param_name:
            return kw.value
    if call_pos is not None and call_pos < len(call.args):
        return call.args[call_pos]
    if has_default:
        return None
    raise NeedsInput(
        f"call at line {call.lineno} does not supply `{param_name}`",
        NeededInput.PROOF,
    )


def _module_exports_name(module: ast.Module, name: str) -> bool:
    for stmt in module.body:
        match stmt:
            case ast.Assign(
                targets=[ast.Name(id="__all__")],
                value=ast.Tuple(elts=elts) | ast.List(elts=elts),
            ):
                for elt in elts:
                    if isinstance(elt, ast.Constant) and elt.value == name:
                        return True
    return False


def _top_level_enclosing_stmt(module: ast.Module, line: int) -> ast.stmt | None:
    for stmt in module.body:
        start = getattr(stmt, "lineno", 0)
        end = getattr(stmt, "end_lineno", start) or start
        if start <= line <= end:
            return stmt
    return None


def _module_top_level_bound_names(module: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for stmt in module.body:
        match stmt:
            case ast.ClassDef(name=n) | ast.FunctionDef(name=n) | ast.AsyncFunctionDef(name=n):
                names.add(n)
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            case ast.AnnAssign(target=ast.Name(id=n)):
                names.add(n)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                for alias in aliases:
                    names.add(alias.asname or alias.name.split(".", 1)[0])
    return frozenset(names)


def _is_abstract_or_stub_body(function: FunctionNode) -> bool:
    """Whether `function` is an abstract/stub declaration with no concrete execution path."""

    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return True
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, (ast.Pass, ast.Raise)):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            return True
    return False


def _locate_class_method(
    mod: ParsedModule,
    class_symbol: str,
    method_name: str,
) -> tuple[ast.ClassDef, FunctionNode] | None:
    for _cls_idx, cls_node in mod.syntax_index.indexed_nodes_of_type(ast.ClassDef):
        if f"{mod.module_name}.{cls_node.name}" != class_symbol:
            continue
        for stmt in cls_node.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == method_name
            ):
                return cls_node, stmt
    return None


def _is_passive_override(function: FunctionNode, param_name: str) -> bool:
    """Whether `function` is a stub or never loads `param_name` (unused or `del`-only)."""

    if _is_abstract_or_stub_body(function):
        return True
    for stmt in function.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Name)
                and node.id == param_name
                and isinstance(node.ctx, ast.Load)
            ):
                return False
    return True


@dataclass(frozen=True)
class _PassiveOverrideParamEdit:
    """Parameter annotation/default rewrite for a passive (stub/unused/del-only) override."""

    file_path: str
    param_annotations: tuple[ast.expr, ...]
    default_exprs: tuple[ast.expr, ...]

    def needed_family_names(self, family_name: str) -> tuple[str, ...]:
        names: list[str] = []
        if self.param_annotations:
            names.append(family_name)
        if self.default_exprs:
            names.append(f"Absent{family_name}")
        return tuple(names)

    def span_replacements(
        self, geometry: SourceTextGeometry, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        reps: list[SourceTextSpanReplacement] = []
        for ann in self.param_annotations:
            a_start, a_end = geometry.required_node_offsets(ann)
            reps.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=a_start,
                    end_offset=a_end,
                    replacement_source=family_name,
                )
            )
        for default_expr in self.default_exprs:
            d_start, d_end = geometry.required_node_offsets(default_expr)
            reps.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=d_start,
                    end_offset=d_end,
                    replacement_source=f"Absent{family_name}()",
                )
            )
        return tuple(reps)


@dataclass(frozen=True)
class _ResolvedThreadedHierarchy:
    """Connected single-function or single-brancher overriding hierarchy for a threaded parameter."""

    family_anchor_line: int
    passive_overrides: tuple[_PassiveOverrideParamEdit, ...]
    reaching_symbols: frozenset[str]
    owner_qualnames_by_file: Mapping[str, frozenset[str]]


class _OverrideParamPlacement(ABC):
    """Expected positional or keyword-only placement of a parameter on hierarchy overrides."""

    @abstractmethod
    def require_positional(self, idx: int, func_name: str, param_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def require_keyword_only(self, func_name: str, param_name: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class _PositionalPlacement(_OverrideParamPlacement):
    """Parameter must appear at positional index `index` on every override."""

    index: int

    def require_positional(self, idx: int, func_name: str, param_name: str) -> None:
        if idx != self.index:
            raise NeedsInput(
                f"override `{func_name}` places `{param_name}` at a different parameter position",
                NeededInput.PROOF,
            )

    def require_keyword_only(self, func_name: str, param_name: str) -> None:
        raise NeedsInput(
            f"override `{func_name}` makes `{param_name}` keyword-only",
            NeededInput.PROOF,
        )


@dataclass(frozen=True)
class _KeywordOnlyPlacement(_OverrideParamPlacement):
    """Parameter must remain keyword-only on every override."""

    def require_positional(self, idx: int, func_name: str, param_name: str) -> None:
        del idx
        raise NeedsInput(
            f"override `{func_name}` places `{param_name}` at a different parameter position",
            NeededInput.PROOF,
        )

    def require_keyword_only(self, func_name: str, param_name: str) -> None:
        del func_name, param_name


def _locate_passive_override_param(
    function: FunctionNode,
    param_name: str,
    placement: _OverrideParamPlacement,
    *,
    skip_first_receiver: bool,
) -> tuple[tuple[ast.expr, ...], tuple[ast.expr, ...]]:
    """Verify `function` accepts `param_name` compatibly and return its annotation/default tuples."""

    args = function.args
    pos_args = list(args.posonlyargs) + list(args.args)
    offset = 1 if skip_first_receiver and pos_args else 0
    pos_params = pos_args[offset:]
    pos_defaults_start = len(pos_params) - len(args.defaults)
    for idx, arg in enumerate(pos_params):
        if arg.arg != param_name:
            continue
        placement.require_positional(idx, function.name, param_name)
        anns = declared_annotations(arg)
        if anns and _strip_none_from_annotation(anns[0]) is None:
            raise NeedsInput(
                f"override `{function.name}` annotation `{ast.unparse(anns[0])}` on `{param_name}` is not `T | None`",
                NeededInput.PROOF,
            )
        default_idx = idx - pos_defaults_start
        if default_idx >= 0:
            default_expr = args.defaults[default_idx]
            if not _is_none_const(default_expr):
                raise NeedsInput(
                    f"override `{function.name}` parameter `{param_name}` has non-None default `{ast.unparse(default_expr)}`",
                    NeededInput.SOURCE_CHANGE,
                )
            return anns, (default_expr,)
        return anns, ()

    for idx, arg in enumerate(args.kwonlyargs):
        if arg.arg != param_name:
            continue
        placement.require_keyword_only(function.name, param_name)
        anns = declared_annotations(arg)
        if anns and _strip_none_from_annotation(anns[0]) is None:
            raise NeedsInput(
                f"override `{function.name}` annotation `{ast.unparse(anns[0])}` on `{param_name}` is not `T | None`",
                NeededInput.PROOF,
            )
        kw_default = args.kw_defaults[idx]
        if kw_default is not None:
            if not _is_none_const(kw_default):
                raise NeedsInput(
                    f"override `{function.name}` parameter `{param_name}` has non-None default `{ast.unparse(kw_default)}`",
                    NeededInput.SOURCE_CHANGE,
                )
            return anns, (kw_default,)
        return anns, ()

    if variadic_parameters(args):
        return (), ()
    raise NeedsInput(
        f"override `{function.name}` does not accept parameter `{param_name}`",
        NeededInput.PROOF,
    )


def _resolve_threaded_hierarchy(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
    param_name: str,
    placement: _OverrideParamPlacement,
) -> _ResolvedThreadedHierarchy:
    """Resolve a standalone function or a single-brancher overriding hierarchy."""

    owner_symbol = enclosure.declaring_class_symbol(snapshot)
    if owner_symbol is None:
        return _ResolvedThreadedHierarchy(
            family_anchor_line=function.lineno,
            passive_overrides=(),
            reaching_symbols=frozenset(),
            owner_qualnames_by_file={},
        )

    index = snapshot.compact_class_index
    classes = index.classes_by_symbol
    if owner_symbol not in classes:
        raise NeedsInput(
            f"`{owner_symbol}` is not in the scanned class index",
            NeededInput.PROOF,
        )

    declared_syms: set[str] = {owner_symbol}
    queue = [owner_symbol]
    while queue:
        sym = queue.pop()
        for other_sym in (*index.ancestor_symbols(sym), *index.descendant_symbols(sym)):
            if other_sym in declared_syms:
                continue
            decl = classes.get(other_sym)
            if decl is not None and function.name in decl.member_binding_names:
                declared_syms.add(other_sym)
                queue.append(other_sym)

    is_async = isinstance(function, ast.AsyncFunctionDef)
    skip_first = enclosure.class_receiver_symbol(snapshot) is not None
    passive_overrides: list[_PassiveOverrideParamEdit] = []
    anchor_lines: list[int] = [function.lineno]
    qualnames_by_file: dict[str, set[str]] = {
        source_path: {classes[owner_symbol].qualname}
    }

    for sym in sorted(declared_syms):
        if sym == owner_symbol:
            continue
        decl = classes[sym]
        cmod = snapshot.parsed_module_for_source_path(decl.file_path)
        found = _locate_class_method(cmod, sym, function.name)
        if found is None:
            raise NeedsInput(
                f"`{decl.qualname}` also declares `{function.name}`; overriding hierarchy is not rewritten",
                NeededInput.PROOF,
            )
        cls_node, other_fn = found
        if isinstance(other_fn, ast.AsyncFunctionDef) != is_async:
            raise NeedsInput(
                f"`{decl.qualname}.{function.name}` mixes sync and async overrides",
                NeededInput.PROOF,
            )
        other_is_static = any(
            isinstance(dec, ast.Name) and dec.id == "staticmethod"
            for dec in other_fn.decorator_list
        )
        if other_is_static == skip_first:
            raise NeedsInput(
                f"`{decl.qualname}.{function.name}` mixes staticmethod and bound-method overrides",
                NeededInput.PROOF,
            )
        if not _is_passive_override(other_fn, param_name):
            raise NeedsInput(
                f"`{decl.qualname}` also uses `{param_name}` in `{function.name}`; "
                "overriding hierarchy is not rewritten",
                NeededInput.PROOF,
            )
        anns, defs = _locate_passive_override_param(
            other_fn,
            param_name,
            placement,
            skip_first_receiver=skip_first,
        )
        passive_overrides.append(
            _PassiveOverrideParamEdit(
                file_path=decl.file_path,
                param_annotations=anns,
                default_exprs=defs,
            )
        )
        qualnames_by_file.setdefault(decl.file_path, set()).add(decl.qualname)
        if decl.file_path == source_path:
            anchor_lines.append(cls_node.lineno)

    reaching_symbols = frozenset().union(
        *(frozenset((sym, *index.descendant_symbols(sym))) for sym in declared_syms)
    )
    return _ResolvedThreadedHierarchy(
        family_anchor_line=min(anchor_lines),
        passive_overrides=tuple(passive_overrides),
        reaching_symbols=reaching_symbols,
        owner_qualnames_by_file={
            fp: frozenset(names) for fp, names in qualnames_by_file.items()
        },
    )


def _collect_closed_call_args(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
    param_name: str,
    call_pos: int | None,
    has_default: bool,
    hierarchy: _ResolvedThreadedHierarchy,
) -> tuple[_RewrittenCallArg, ...]:
    """Prove every reference to `function` across the scanned repo is a direct call
    and collect every call site's argument expression for `param_name`."""

    is_private = function.name.startswith("_")
    bindings = snapshot.module_binding_proof
    source_module = snapshot.parsed_module_for_source_path(source_path)
    raw_args: list[tuple[str, ast.expr]] = []
    owner_symbol = enclosure.declaring_class_symbol(snapshot)

    if owner_symbol is None:
        if enclosure.enclosing_function_chain():
            raise NeedsInput(
                f"`{function.name}` is nested inside another function; cannot prove closed callers",
                NeededInput.PROOF,
            )
        if _module_exports_name(source_module.module, function.name):
            raise NeedsInput(
                f"`{function.name}` is listed in `__all__`; callers are not closed",
                NeededInput.PROOF,
            )
        target_symbol = f"{source_module.module_name}.{function.name}"
        for mod in snapshot.parsed_modules:
            if function.name not in mod.source:
                continue
            node_index = mod.fact(ModuleNodeIndex)
            candidates: tuple[ast.Name | ast.Attribute, ...] = (
                *(
                    name_node
                    for name_node in node_index.names_by_id.get(function.name, ())
                    if isinstance(name_node.ctx, ast.Load)
                ),
                *node_index.attributes_by_attr.get(function.name, ()),
            )
            for node in candidates:
                witness = bindings.reference_or_builtin_witness_at(
                    mod, node, line=node.lineno
                )
                if witness is None or witness.qualified_name != target_symbol:
                    continue
                parent = node_index.parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    raise NeedsInput(
                        f"{mod.file_path}:{node.lineno} references `{function.name}` as a value; callers are not closed",
                        NeededInput.PROOF,
                    )
                arg_expr = _extract_call_argument(
                    parent, param_name, call_pos, has_default
                )
                if arg_expr is not None:
                    raw_args.append((mod.file_path, arg_expr))
        if not is_private and not raw_args:
            raise NeedsInput(
                f"public function `{function.name}` has no scanned callers; cannot prove callers are closed",
                NeededInput.PROOF,
            )
        return _classify_call_args_batched(snapshot, raw_args)

    index = snapshot.compact_class_index
    classes = index.classes_by_symbol
    reaching_symbols = hierarchy.reaching_symbols
    owner_qualname = classes[owner_symbol].qualname
    reaching_files = frozenset(
        (
            source_path,
            *(
                classes[sym].file_path
                for sym in reaching_symbols
                if sym in classes
            ),
        )
    )

    # Pass 1: collect definite hits and defer receiver-type queries for one batched mypy call.
    definite_hits: list[tuple[str, ast.Attribute, ast.Call | None]] = []
    deferred_checks: list[tuple[str, ast.Attribute, ast.Call | None, ReceiverSpan]] = []
    for mod in snapshot.parsed_modules:
        if function.name not in mod.source:
            continue
        node_index = mod.fact(ModuleNodeIndex)
        attr_nodes = node_index.attributes_by_attr.get(function.name, ())
        if not attr_nodes:
            continue
        reaching_class_ranges: list[tuple[int, int]] = []
        if mod.file_path in reaching_files:
            for _cls_idx, cls_node in mod.syntax_index.indexed_nodes_of_type(
                ast.ClassDef
            ):
                sym = f"{mod.module_name}.{cls_node.name}"
                if sym in reaching_symbols:
                    reaching_class_ranges.append(
                        (cls_node.lineno, cls_node.end_lineno or cls_node.lineno)
                    )
        local_owner_qualnames = hierarchy.owner_qualnames_by_file.get(
            mod.file_path, frozenset()
        )

        for node in attr_nodes:
            parent = node_index.parents.get(node)
            parent_call = (
                parent if isinstance(parent, ast.Call) and parent.func is node else None
            )
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                if any(lo <= node.lineno <= hi for lo, hi in reaching_class_ranges):
                    definite_hits.append((mod.file_path, node, parent_call))
                    continue
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in local_owner_qualnames
            ):
                definite_hits.append((mod.file_path, node, parent_call))
                continue
            witness = bindings.reference_or_builtin_witness_at(
                mod, node.value, line=node.lineno
            )
            if witness is not None and witness.qualified_name in reaching_symbols:
                definite_hits.append((mod.file_path, node, parent_call))
                continue
            deferred_checks.append(
                (
                    mod.file_path,
                    node,
                    parent_call,
                    ReceiverSpan.of(mod.file_path, node.value),
                )
            )

    inferred_receivers = (
        snapshot.type_checker.receiver_types(
            tuple(dict.fromkeys(span for _, _, _, span in deferred_checks))
        )
        if deferred_checks
        else {}
    )
    all_hits = list(definite_hits)
    for file_path, node, parent_call, span in deferred_checks:
        answer = inferred_receivers.get(span)
        if answer is None:
            raise NeedsInput(
                f"{file_path}:{node.lineno} accesses `.{function.name}` on an unproved receiver",
                NeededInput.PROOF,
            )
        if answer.reaches_method_owner(
            reaching_symbols,
            file_path=file_path,
            line=node.lineno,
            method_name=function.name,
        ):
            all_hits.append((file_path, node, parent_call))

    for file_path, node, parent_call in all_hits:
        if parent_call is None:
            raise NeedsInput(
                f"{file_path}:{node.lineno} references `.{function.name}` as a value; callers are not closed",
                NeededInput.PROOF,
            )
        arg_expr = _extract_call_argument(
            parent_call, param_name, call_pos, has_default
        )
        if arg_expr is not None:
            raw_args.append((file_path, arg_expr))

    if not is_private and not raw_args:
        raise NeedsInput(
            f"public method `{owner_qualname}.{function.name}` has no scanned callers; cannot prove callers are closed",
            NeededInput.PROOF,
        )
    return _classify_call_args_batched(snapshot, raw_args)


@dataclass(frozen=True)
class ThreadedAbsenceProof:
    """A threaded `T | None` parameter proved safe to replace with a Held/Absent family."""

    source_path: str
    probe_line: int
    function: FunctionNode
    enclosure: FunctionEnclosure
    param_arg: ast.arg
    param_annotation: ast.expr
    inner_type: ast.expr
    default_expr: ast.expr | None
    site: ThreadedAbsenceBranchSite
    family_name: str
    family_docstring: str
    method_name: str
    method_docstring: str | None
    absent_body: tuple[ast.stmt, ...]
    held_body: tuple[ast.stmt, ...]
    captured_args: tuple[CapturedArgument, ...]
    is_async: bool
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...]
    call_args: tuple[_RewrittenCallArg, ...]
    passive_overrides: tuple[_PassiveOverrideParamEdit, ...]
    insert_before_stmt: ast.stmt

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        parameter_name: str,
        probe_line: int,
        family_name: str,
        family_docstring: str,
        method_name: str,
        method_docstring: str | None = None,
    ) -> ThreadedAbsenceProof:
        for label, ident in (("family_name", family_name), ("method_name", method_name)):
            if (
                not ident.isidentifier()
                or keyword_module.iskeyword(ident)
                or (ident.startswith("__") and ident.endswith("__"))
            ):
                raise NeedsInput(
                    f"`{ident}` is not a valid non-dunder {label}",
                    NeededInput.OWNER_PICK,
                )
        source_path, enclosure, function = TypeTestDispatchProof.locate_function(
            snapshot, target, test_line=probe_line
        )
        if function.name.startswith("__") and function.name.endswith("__"):
            raise NeedsInput(
                f"`{function.name}` is a dunder method whose signature is fixed by Python",
                NeededInput.PROOF,
            )
        skip_first = enclosure.class_receiver_symbol(snapshot) is not None
        param_arg, param_ann, call_pos, default_expr = _locate_parameter(
            function, parameter_name, skip_first_receiver=skip_first
        )
        if default_expr is not None and not _is_none_const(default_expr):
            raise NeedsInput(
                f"parameter `{parameter_name}` has non-None default `{ast.unparse(default_expr)}`",
                NeededInput.SOURCE_CHANGE,
            )
        inner_type = _strip_none_from_annotation(param_ann)
        if inner_type is None:
            raise NeedsInput(
                f"annotation `{ast.unparse(param_ann)}` on `{parameter_name}` is not a `T | None` union",
                NeededInput.PROOF,
            )

        site = find_threaded_absence_site(function, parameter_name, probe_line)
        if site is None:
            raise NeedsInput(
                f"absence probe on `{parameter_name}` at {source_path}:{probe_line} is not a pushable branch shape",
                NeededInput.SOURCE_CHANGE,
            )

        if site.checks_local_store_escapes:
            stored = _names_stored_in((*site.absent_body, *site.held_body))
            used_outside = _names_referenced_outside(function, site.replaced_ast_nodes())
            escaping = sorted(stored & used_outside)
            if escaping:
                upgraded = _upgrade_threaded_to_assigning(function, site, escaping)
                if upgraded is not None:
                    site = upgraded
                else:
                    raise NeedsInput(
                        f"`{function.name}` assigns outer local(s) {', '.join(escaping)} inside the absence branch",
                        NeededInput.SOURCE_CHANGE,
                    )

        # Soundness: `parameter_name` must not be referenced outside the replaced branch,
        # must not be read in the absent arm, and must not be stored/deleted anywhere in `function`.
        outside_names = _names_referenced_outside(function, site.replaced_ast_nodes())
        if parameter_name in outside_names:
            raise NeedsInput(
                f"parameter `{parameter_name}` is still referenced outside the pushed absence branch",
                NeededInput.SOURCE_CHANGE,
            )
        if _name_used_in(parameter_name, site.absent_body):
            raise NeedsInput(
                f"absent branch still reads `{parameter_name}`",
                NeededInput.SOURCE_CHANGE,
            )
        if parameter_name in _names_stored_in(function.body):
            raise NeedsInput(
                f"`{function.name}` rebinds `{parameter_name}`",
                NeededInput.SOURCE_CHANGE,
            )

        adapter = site.as_dispatch_adapter()
        enclosing_locals = enclosure.enclosing_local_names()
        outer_receiver_aliases = _outer_receiver_aliases(
            function,
            adapter,
            (site.absent_body, site.held_body),
            enclosing_locals=enclosing_locals,
        )
        rewritten_absent = _rewrite_outer_aliases_only(
            site.absent_body, outer_receiver_aliases
        )
        rewritten_held = _rewrite_param_to_self_value(
            site.held_body, parameter_name, outer_receiver_aliases
        )
        captured_args, is_async = _analyze_arm_bodies(
            function,
            adapter,
            (rewritten_absent, rewritten_held),
            outer_receiver_aliases=outer_receiver_aliases,
            enclosing_locals=enclosing_locals,
        )

        source_module = snapshot.parsed_module_for_source_path(source_path)
        existing_top = _module_top_level_bound_names(source_module.module)
        new_family_names = (
            family_name,
            f"Absent{family_name}",
            f"Held{family_name}",
        )
        for new_name in new_family_names:
            if new_name in existing_top:
                raise NeedsInput(
                    f"`{new_name}` is already bound in {source_path}",
                    NeededInput.OWNER_PICK,
                )

        placement: _OverrideParamPlacement = (
            _KeywordOnlyPlacement()
            if call_pos is None
            else _PositionalPlacement(call_pos)
        )
        hierarchy = _resolve_threaded_hierarchy(
            snapshot,
            source_path,
            enclosure,
            function,
            parameter_name,
            placement,
        )

        insert_before = _top_level_enclosing_stmt(
            source_module.module, hierarchy.family_anchor_line
        )
        if insert_before is None:
            raise NeedsInput(
                f"cannot locate top-level declaration enclosing `{function.name}`",
                NeededInput.PROOF,
            )

        call_args = _collect_closed_call_args(
            snapshot,
            source_path,
            enclosure,
            function,
            parameter_name,
            call_pos,
            default_expr is not None,
            hierarchy,
        )

        # Verify cross-module imports for external callers and passive overrides are cycle-free.
        import_graph = snapshot.module_import_graph
        external_imports_needed: list[tuple[str, tuple[str, ...]]] = [
            (item.file_path, item.wrapping.needed_symbols(family_name))
            for item in call_args
            if item.file_path != source_path
        ]
        external_imports_needed.extend(
            (override.file_path, override.needed_family_names(family_name))
            for override in hierarchy.passive_overrides
            if override.file_path != source_path
        )
        for ext_file, sym_names in external_imports_needed:
            ext_mod = snapshot.parsed_module_for_source_path(ext_file)
            ext_bound = _module_top_level_bound_names(ext_mod.module)
            for sym_name in sym_names:
                if sym_name in ext_bound:
                    raise NeedsInput(
                        f"`{sym_name}` is already bound in {ext_file}",
                        NeededInput.OWNER_PICK,
                    )
                try:
                    import_graph.required_import_source(
                        importing_file_path=ext_file,
                        imported_file_path=source_path,
                        imported_name=sym_name,
                    )
                except ValueError as error:
                    raise NeedsInput(str(error), NeededInput.SOURCE_CHANGE) from error

        return cls(
            source_path=source_path,
            probe_line=probe_line,
            function=function,
            enclosure=enclosure,
            param_arg=param_arg,
            param_annotation=param_ann,
            inner_type=inner_type,
            default_expr=default_expr,
            site=site,
            family_name=family_name,
            family_docstring=family_docstring,
            method_name=method_name,
            method_docstring=method_docstring,
            absent_body=rewritten_absent,
            held_body=rewritten_held,
            captured_args=captured_args,
            is_async=is_async,
            outer_receiver_aliases=outer_receiver_aliases,
            call_args=call_args,
            passive_overrides=hierarchy.passive_overrides,
            insert_before_stmt=insert_before,
        )

    def rationale(self) -> str:
        return (
            f"Replace threaded optional parameter `{self.param_arg.arg}` in "
            f"`{self.function.name}` with `{self.family_name}` "
            f"(`Absent{self.family_name}` / `Held{self.family_name}`)."
        )

    def _render_family_classes(
        self,
        snapshot: CodemodSourceSnapshot,
        source_module_future_ann: bool,
        param_annotations: dict[str, ast.expr],
        return_annotation: ast.expr | None,
    ) -> tuple[str, bool]:
        params = tuple(arg.param_name for arg in self.captured_args)
        absent_is_noop = _is_noop_body(self.absent_body)
        needs_abstractmethod = not absent_is_noop
        needs_from_optional = any(item.wrapping.uses_from_optional for item in self.call_args)

        base_lines = [
            f"class {self.family_name}(ABC):",
            f'    """{self.family_docstring}"""',
            "",
        ]
        if absent_is_noop:
            base_method = _render_method(
                InstanceDispatchKind(),
                self.method_name,
                params,
                (ast.Pass(),),
                is_async=self.is_async,
                docstring=self.method_docstring,
                param_annotations=param_annotations,
                return_annotation=return_annotation,
            )
        else:
            base_lines.append("    @abstractmethod")
            base_method = _render_method(
                InstanceDispatchKind(),
                self.method_name,
                params,
                (ast.Expr(value=ast.Constant(value=...)),),
                is_async=self.is_async,
                docstring=self.method_docstring,
                param_annotations=param_annotations,
                return_annotation=return_annotation,
            )
        for line in base_method.splitlines():
            base_lines.append(f"    {line}" if line else "")

        if needs_from_optional:
            orig_ann_src = ast.unparse(self.param_annotation)
            ret_type = (
                self.family_name
                if source_module_future_ann
                else repr(self.family_name)
            )
            base_lines.extend(
                [
                    "",
                    "    @classmethod",
                    f"    def from_optional(cls, value: {orig_ann_src}) -> {ret_type}:",
                    f"        return Absent{self.family_name}() if value is None else Held{self.family_name}(value)",
                ]
            )

        absent_lines = [
            "",
            "",
            "@dataclass(frozen=True)",
            f"class Absent{self.family_name}({self.family_name}):",
            f'    """Absent `{self.family_name}` carrying no value."""',
        ]
        if not absent_is_noop:
            absent_lines.append("")
            absent_method = _render_method(
                InstanceDispatchKind(),
                self.method_name,
                params,
                self.absent_body,
                is_async=self.is_async,
                docstring=None,
                param_annotations=param_annotations,
                return_annotation=return_annotation,
            )
            for line in absent_method.splitlines():
                absent_lines.append(f"    {line}" if line else "")

        inner_type_src = ast.unparse(self.inner_type)
        held_body = self.held_body if self.held_body else (ast.Pass(),)
        held_lines = [
            "",
            "",
            "@dataclass(frozen=True)",
            f"class Held{self.family_name}({self.family_name}):",
            f'    """Held `{self.family_name}` carrying its present value."""',
            "",
            f"    value: {inner_type_src}",
            "",
        ]
        held_method = _render_method(
            InstanceDispatchKind(),
            self.method_name,
            params,
            held_body,
            is_async=self.is_async,
            docstring=None,
            param_annotations=param_annotations,
            return_annotation=return_annotation,
        )
        for line in held_method.splitlines():
            held_lines.append(f"    {line}" if line else "")

        del snapshot
        return "\n".join((*base_lines, *absent_lines, *held_lines)) + "\n\n\n", needs_abstractmethod

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        source_module = snapshot.parsed_module_for_source_path(self.source_path)
        future_ann = _has_future_annotations(source_module.module)
        source_geometry = SourceTextGeometry(
            snapshot.sources_by_file_path[self.source_path]
        )
        resolved_anns = ResolvedMethodAnnotations.for_push(
            snapshot,
            source_module,
            self.probe_line,
            self.enclosure,
            self.function,
            self.outer_receiver_aliases,
            self.captured_args,
            (),
            propagates_function_return_type=self.site.propagates_function_return_type,
            is_void_method=self.site.is_void_method,
            rationale=rationale,
        )
        family_block, needs_abstractmethod = self._render_family_classes(
            snapshot,
            future_ann,
            resolved_anns.param_map,
            resolved_anns.return_annotation,
        )

        # Build non-overlapping replacements in `source_path`.
        replacements_by_file: dict[str, list[SourceTextSpanReplacement]] = {}

        first_anchor: ast.AST = (
            self.insert_before_stmt.decorator_list[0]
            if isinstance(
                self.insert_before_stmt,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            and self.insert_before_stmt.decorator_list
            else self.insert_before_stmt
        )
        ins_start, _ = source_geometry.required_node_offsets(first_anchor)
        ins_line_start = source_geometry.source.rfind("\n", 0, ins_start) + 1
        replacements_by_file.setdefault(self.source_path, []).append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=ins_line_start,
                end_offset=ins_line_start,
                replacement_source=family_block,
            )
        )

        # Rewrite parameter annotation `X | None` -> `FamilyName` (and `= None` -> `= AbsentFamilyName()`).
        ann_start, ann_end = source_geometry.required_node_offsets(self.param_annotation)
        replacements_by_file[self.source_path].append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=ann_start,
                end_offset=ann_end,
                replacement_source=self.family_name,
            )
        )
        if self.default_expr is not None:
            def_start, def_end = source_geometry.required_node_offsets(self.default_expr)
            replacements_by_file[self.source_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=def_start,
                    end_offset=def_end,
                    replacement_source=f"Absent{self.family_name}()",
                )
            )

        # Rewrite the absence branch inside `function` into `param.method(...)`.
        call_args_str = ", ".join(arg.call_expr for arg in self.captured_args)
        raw_call = f"{self.param_arg.arg}.{self.method_name}({call_args_str})"
        call_expr = f"await {raw_call}" if self.is_async else raw_call
        rep_start, rep_end = self.site.replaced_offsets(source_geometry)
        line_start = source_geometry.source.rfind("\n", 0, rep_start) + 1
        line_indent = source_geometry.source[line_start:rep_start]
        if not line_indent.isspace():
            line_indent = ""
        replacements_by_file[self.source_path].append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=rep_start,
                end_offset=rep_end,
                replacement_source=self.site.render_call_replacement(
                    call_expr, line_indent
                ),
            )
        )

        # Rewrite every caller's argument expression and collect cross-module imports.
        needed_caller_imports: dict[tuple[str, str], ModuleImportMutation] = {}
        import_graph = snapshot.module_import_graph
        for item in self.call_args:
            caller_source = snapshot.sources_by_file_path[item.file_path]
            caller_geom = SourceTextGeometry(caller_source)
            a_start, a_end = caller_geom.required_node_offsets(item.arg_expr)
            replacements_by_file.setdefault(item.file_path, []).append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=a_start,
                    end_offset=a_end,
                    replacement_source=item.wrapping.render(
                        caller_source, self.family_name
                    ),
                )
            )
            if item.file_path != self.source_path:
                for sym_name in item.wrapping.needed_symbols(self.family_name):
                    key = (item.file_path, sym_name)
                    if key not in needed_caller_imports:
                        needed_caller_imports[key] = ModuleImportMutation.from_source(
                            file_path=item.file_path,
                            import_source=import_graph.required_import_source(
                                importing_file_path=item.file_path,
                                imported_file_path=self.source_path,
                                imported_name=sym_name,
                            ),
                            rationale=rationale,
                        )

        # Rewrite parameter annotations/defaults on passive overrides (stub/unused/del-only).
        for override in self.passive_overrides:
            ov_geom = SourceTextGeometry(snapshot.sources_by_file_path[override.file_path])
            replacements_by_file.setdefault(override.file_path, []).extend(
                override.span_replacements(ov_geom, self.family_name)
            )
            if override.file_path != self.source_path:
                for sym_name in override.needed_family_names(self.family_name):
                    key = (override.file_path, sym_name)
                    if key not in needed_caller_imports:
                        needed_caller_imports[key] = ModuleImportMutation.from_source(
                            file_path=override.file_path,
                            import_source=import_graph.required_import_source(
                                importing_file_path=override.file_path,
                                imported_file_path=self.source_path,
                                imported_name=sym_name,
                            ),
                            rationale=rationale,
                        )

        all_edits: list[NominalSourceEdit] = []
        # Ensure `ABC`, `abstractmethod` (when needed), and `dataclass` are imported in `source_path`.
        bound_in_source = _module_top_level_bound_names(source_module.module)
        abc_needed = [
            name
            for name in (("ABC", "abstractmethod") if needs_abstractmethod else ("ABC",))
            if name not in bound_in_source
        ]
        if abc_needed:
            all_edits.append(
                ModuleImportMutation.from_source(
                    file_path=self.source_path,
                    import_source=f"from abc import {', '.join(abc_needed)}",
                    rationale=rationale,
                )
            )
        if "dataclass" not in bound_in_source:
            all_edits.append(
                ModuleImportMutation.from_source(
                    file_path=self.source_path,
                    import_source="from dataclasses import dataclass",
                    rationale=rationale,
                )
            )
        all_edits.extend(needed_caller_imports.values())
        for file_path, reps in replacements_by_file.items():
            geom = SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            all_edits.append(
                geom.nominal_edit(
                    file_path=file_path,
                    replacements=tuple(
                        sorted(reps, key=lambda r: (r.start_offset, r.end_offset))
                    ),
                    rationale=rationale,
                )
            )
        return tuple(all_edits)


@dataclass(frozen=True, kw_only=True)
class ThreadedAbsenceToFamilyOperation(RepositorySourceReprovedOperation):
    """Replace a threaded `T | None` parameter with a `Held` / `Absent` nominal family."""

    parameter_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    probe_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    family_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    family_docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    method_docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )

    def __post_init__(self) -> None:
        if self.probe_line < 1:
            raise ValueError("probe_line must be a positive line number")
        for label, value in (
            ("parameter_name", self.parameter_name),
            ("family_name", self.family_name),
            ("method_name", self.method_name),
        ):
            if not value.isidentifier() or keyword_module.iskeyword(value):
                raise ValueError(f"{label} must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        proof = ThreadedAbsenceProof.proved(
            snapshot,
            self.target,
            self.parameter_name,
            self.probe_line,
            self.family_name,
            self.family_docstring,
            self.method_name,
            self.method_docstring,
        )
        return proof.edits(snapshot, self.rationale_text(proof.rationale()))


class ThreadedAbsenceFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    PrewarmTypeCheckerMixin,
    FindingRecipeSynthesizer,
    ThreadedAbsenceFamilyConcept,
    ABC,
):
    """Turn a proved threaded-absence finding into an owner-picked Held/Absent family codemod."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not finding.evidence:
            return self.rejected_evaluation(
                "threaded-absence refactoring requires source context and finding evidence"
            )
        if not isinstance(finding.metrics, ThreadedAbsenceMetrics):
            return self.rejected_evaluation(
                "threaded-absence finding is missing structured metrics"
            )
        metrics = finding.metrics
        snapshot = context.execution_snapshot()
        primary = finding.evidence[0]
        target = SourceRewriteTarget(
            file_path=primary.file_path, qualname=primary.symbol
        )
        probe_line = metrics.probe_lines[0] if metrics.probe_lines else primary.line
        pick = metrics.picked_base
        if pick is None:
            try:
                probe_proof = ThreadedAbsenceProof.proved(
                    snapshot,
                    target,
                    metrics.parameter_name,
                    probe_line,
                    "__NraThreadedAbsenceFamily",
                    "Nominal held/absent family.",
                    "__nra_threaded_absence_method",
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            default_method_hint = (
                f" (defaults to `{probe_proof.function.name}` if `hoisted_name` is omitted)"
                if probe_proof.site.is_whole_function
                else "; also supply `hoisted_name` for the method name"
            )
            return self.stopped_evaluation(
                NeedsInput(
                    f"threaded absence on `{metrics.parameter_name}` in `{metrics.function_symbol}` "
                    f"can become a held/absent family; supply `owner` `{metrics.function_symbol}` "
                    f"with `name` (family base class) and `judgement` (one-line docstring)"
                    f"{default_method_hint} in `--codemod-owner-picks`",
                    NeededInput.OWNER_PICK,
                )
            )
        family_name = pick.name
        family_docstring = pick.judgement
        method_name = pick.hoisted_name or primary.symbol.rsplit(".", 1)[-1]
        method_docstring = pick.hoisted_judgement
        try:
            proof = ThreadedAbsenceProof.proved(
                snapshot,
                target,
                metrics.parameter_name,
                probe_line,
                family_name,
                family_docstring,
                method_name,
                method_docstring,
            )
            operation = ThreadedAbsenceToFamilyOperation(
                target=target,
                parameter_name=metrics.parameter_name,
                probe_line=probe_line,
                family_name=family_name,
                family_docstring=family_docstring,
                method_name=method_name,
                method_docstring=method_docstring,
            )
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=proof.rationale(),
            ).with_proved_operation(operation, snapshot)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "AssigningThreadedSite",
    "PrefixGuardedAndThreadedSite",
    "PrefixGuardedOrThreadedSite",
    "ReturningThreadedSite",
    "TernaryThreadedSite",
    "ThreadedAbsenceBranchSite",
    "ThreadedAbsenceFindingRecipeSynthesizer",
    "ThreadedAbsenceProof",
    "ThreadedAbsenceToFamilyOperation",
    "VoidThreadedSite",
    "find_threaded_absence_site",
)
