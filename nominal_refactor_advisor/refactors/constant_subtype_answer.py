"""Push branches on a family's constant subtype answer onto polymorphic methods of that family.

When every member of a class family answers a zero-argument question with a
bool or str constant (`ClassVar[bool]`, class bool attribute, property, or
zero-argument method returning a constant) and consumers branch on that answer,
moving each consumer's branch onto the family base and overriding only on the
members whose resolved constant answer differs eliminates the branch. When no
other reads of the constant answer remain in the repository, the constant
declaration itself is deleted in the same recipe.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
from abc import ABC
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.codemod_payload import (
    BooleanPayloadValueCodec,
    CodemodPayloadRecord,
    OptionalStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
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
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.models import (
    ConstantSubtypeAnswerMetrics,
    RefactorFinding,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import ConstantSubtypeAnswerConcept
from nominal_refactor_advisor.refactors.class_overlap import (
    obsolete_import_edits,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
)
from nominal_refactor_advisor.refactors.restated_overrides import NamespaceReaders
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    BooleanPredicateDispatchSite,
    BoolOpSuffixDispatchSite,
    CapturedArgument,
    ClassEnclosure,
    DispatchKind,
    DispatchSite,
    FunctionEnclosure,
    FunctionNode,
    InstanceDispatchKind,
    OrElseGuardedStatementDispatchSite,
    OuterReceiverAlias,
    PrefixGuardedOrDispatchSite,
    PrefixGuardedStatementDispatchSite,
    ResolvedMethodAnnotations,
    ReturningStatementDispatchSite,
    StatementDispatchSite,
    TernaryDispatchSite,
    TypeTestArm,
    TypeTestDispatchProof,
    _always_terminates,
    _always_void_terminates,
    _analyze_arm_bodies,
    _body_without_docstring,
    _class_append_edits,
    _cross_module_imports_for_body,
    _ends_with_continue_only,
    _has_any_return,
    _has_non_continue_loop_escape,
    _has_value_return,
    _is_pure_subject,
    _outer_receiver_aliases,
    _render_method,
    _rewrite_loop_continues_to_returns,
    _rewrite_subject_to_receiver,
    _statement_list_children,
    _strip_trailing_bare_return,
    _strip_trailing_continue,
)
from nominal_refactor_advisor.relocation_observations import (
    ClassBodyReading,
    declares_abstract,
)

ConstantValue = bool | str


@dataclass(frozen=True)
class ParsedAnswerTest:
    """One branch test reading `subject.<member_name>[()]` against a bool or str constant."""

    subject: ast.expr
    tested_value: ConstantValue
    negated: bool
    test_node: ast.expr


@dataclass(frozen=True)
class AnswerBranchSite:
    """A consumer branch normalized into `(value -> body)` arms plus a default body."""

    dispatch: DispatchSite
    arm_values: tuple[ConstantValue, ...]


ComprehensionNode = ast.GeneratorExp | ast.ListComp | ast.SetComp


def _fresh_comp_item_name(function: FunctionNode, comp: ComprehensionNode) -> str:
    used = {
        node.id
        for root in (function, comp)
        for node in ast.walk(root)
        if isinstance(node, ast.Name)
    }
    if "_item" not in used:
        return "_item"
    index = 1
    while f"_item_{index}" in used:
        index += 1
    return f"_item_{index}"


@dataclass(frozen=True)
class SumComprehensionDispatchSite(DispatchSite):
    """`sum(elt for ... if [PREFIX and] [not] subject.flag)` replaced by `sum(subject.method(...) for ... [if PREFIX])`."""

    comp: ast.GeneratorExp
    remaining_last_ifs: tuple[ast.expr, ...]

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.comp)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        new_comp = copy.deepcopy(self.comp)
        new_comp.elt = ast.parse(call_expr, mode="eval").body
        new_comp.generators[-1].ifs = [copy.deepcopy(expr) for expr in self.remaining_last_ifs]
        return ast.unparse(ast.fix_missing_locations(new_comp))


@dataclass(frozen=True)
class ComprehensionFilterDispatchSite(DispatchSite):
    """Filtered `GeneratorExp`/`ListComp`/`SetComp` replaced by flattening `for _item in subject.method(...)`."""

    comp: ComprehensionNode
    remaining_last_ifs: tuple[ast.expr, ...]
    trailing_ifs: tuple[ast.expr, ...]
    item_name: str

    def replaced_offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.comp)

    def render_call_replacement(self, call_expr: str, line_indent: str = "") -> str:
        del line_indent
        new_comp = copy.deepcopy(self.comp)
        new_comp.elt = ast.Name(id=self.item_name, ctx=ast.Load())
        new_comp.generators[-1].ifs = [copy.deepcopy(expr) for expr in self.remaining_last_ifs]
        new_comp.generators.append(
            ast.comprehension(
                target=ast.Name(id=self.item_name, ctx=ast.Store()),
                iter=ast.parse(call_expr, mode="eval").body,
                ifs=[copy.deepcopy(expr) for expr in self.trailing_ifs],
                is_async=0,
            )
        )
        return ast.unparse(ast.fix_missing_locations(new_comp))


def _extract_answer_read(expr: ast.expr, member_name: str) -> ast.expr | None:
    """Return `subject` when `expr` is `subject.<member_name>` or `subject.<member_name>()`."""

    match expr:
        case ast.Attribute(value=subject, attr=attr) if (
            attr == member_name and _is_pure_subject(subject)
        ):
            return subject
        case ast.Call(
            func=ast.Attribute(value=subject, attr=attr),
            args=[],
            keywords=[],
        ) if attr == member_name and _is_pure_subject(subject):
            return subject
    return None


def _parse_answer_test(
    expr: ast.expr, member_name: str
) -> ParsedAnswerTest | None:
    negated = False
    inner = expr
    match expr:
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            negated = True
            inner = operand

    subject = _extract_answer_read(inner, member_name)
    if subject is not None:
        return ParsedAnswerTest(subject, True, negated, expr)

    match inner:
        case ast.Compare(
            left=left,
            ops=[ast.Is() | ast.IsNot() | ast.Eq() | ast.NotEq() as op],
            comparators=[ast.Constant(value=bool() | str() as constant)],
        ):
            subject = _extract_answer_read(left, member_name)
            if subject is None:
                return None
            is_negated = negated ^ isinstance(op, (ast.IsNot, ast.NotEq))
            return ParsedAnswerTest(subject, constant, is_negated, expr)
        case ast.Compare(
            left=ast.Constant(value=bool() | str() as constant),
            ops=[ast.Is() | ast.IsNot() | ast.Eq() | ast.NotEq() as op],
            comparators=[right],
        ):
            subject = _extract_answer_read(right, member_name)
            if subject is None:
                return None
            is_negated = negated ^ isinstance(op, (ast.IsNot, ast.NotEq))
            return ParsedAnswerTest(subject, constant, is_negated, expr)
    return None


def _dummy_arm(body: tuple[ast.stmt, ...]) -> TypeTestArm:
    return TypeTestArm((), body, False)


def _build_answer_statement_site(
    first_test: ParsedAnswerTest,
    kind: DispatchKind,
    arms: tuple[TypeTestArm, ...],
    arm_values: tuple[ConstantValue, ...],
    default_body: tuple[ast.stmt, ...],
    replaced_nodes: tuple[ast.stmt, ...],
    is_whole_function: bool,
    test_nodes: tuple[ast.AST, ...],
) -> AnswerBranchSite | None:
    all_bodies = (*(arm.body for arm in arms), default_body)
    if any(_has_any_return(body) for body in all_bodies):
        if not all(_always_terminates(body) for body in all_bodies):
            return None
        return AnswerBranchSite(
            ReturningStatementDispatchSite(
                subject=first_test.subject,
                kind=kind,
                arms=arms,
                default_body=default_body,
                is_whole_function=is_whole_function,
                test_nodes=test_nodes,
                replaced_nodes=replaced_nodes,
            ),
            arm_values,
        )
    return AnswerBranchSite(
        StatementDispatchSite(
            subject=first_test.subject,
            kind=kind,
            arms=arms,
            default_body=default_body,
            is_whole_function=is_whole_function,
            test_nodes=test_nodes,
            replaced_nodes=replaced_nodes,
        ),
        arm_values,
    )


def _extract_compound_answer_if(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    test_line: int,
    member_name: str,
    kind: DispatchKind,
    in_loop_body: bool,
) -> AnswerBranchSite | None:
    if not isinstance(stmt.test, ast.BoolOp) or len(stmt.test.values) < 2:
        return None
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
        return None
    first = _parse_answer_test(stmt.test.values[0], member_name)
    if first is None:
        match_index: int | None = None
        matched: ParsedAnswerTest | None = None
        for idx, value in enumerate(stmt.test.values[1:], start=1):
            parsed = _parse_answer_test(value, member_name)
            if parsed is not None and value.lineno == test_line:
                match_index = idx
                matched = parsed
                break
        if match_index is None or matched is None:
            return None
        prefix_values = stmt.test.values[:match_index]
        guard_expr: ast.expr = (
            prefix_values[0]
            if len(prefix_values) == 1
            else ast.BoolOp(op=stmt.test.op, values=prefix_values)
        )
        suffix_values = stmt.test.values[match_index + 1 :]
        suffix_expr: ast.expr | None = (
            None
            if not suffix_values
            else (
                suffix_values[0]
                if len(suffix_values) == 1
                else ast.BoolOp(op=stmt.test.op, values=suffix_values)
            )
        )
        if isinstance(stmt.test.op, ast.Or) and (stmt.orelse or _has_any_return(stmt.body) or _always_terminates(stmt.body) or (in_loop_body and _ends_with_continue_only(stmt.body))):
            if stmt.orelse:
                if not (_has_any_return((stmt,)) and _always_terminates(stmt.body) and _always_terminates(stmt.orelse)):
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
                arm_body = fallback_body if matched.negated else tuple(stmt.body)
                default_body = tuple(stmt.body) if matched.negated else fallback_body
                return AnswerBranchSite(
                    PrefixGuardedOrDispatchSite(
                        subject=matched.subject,
                        kind=kind,
                        arms=(_dummy_arm(arm_body),),
                        default_body=default_body,
                        is_whole_function=False,
                        test_nodes=(matched.test_node,),
                        replaced_nodes=(stmt,),
                        guard_expr=guard_expr,
                        guard_body=tuple(stmt.body),
                        returns_value=True,
                    ),
                    (matched.tested_value,),
                )
            tail = container[index + 1 :]
            if tail and _always_terminates(stmt.body) and _always_terminates(tail):
                replaced = tuple(container[index:])
                fallback_body = (
                    (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]), *tail)
                    if suffix_expr is not None
                    else tuple(tail)
                )
                arm_body = fallback_body if matched.negated else tuple(stmt.body)
                default_body = tuple(stmt.body) if matched.negated else fallback_body
                return AnswerBranchSite(
                    PrefixGuardedOrDispatchSite(
                        subject=matched.subject,
                        kind=kind,
                        arms=(_dummy_arm(arm_body),),
                        default_body=default_body,
                        is_whole_function=False,
                        test_nodes=(matched.test_node,),
                        replaced_nodes=replaced,
                        guard_expr=guard_expr,
                        guard_body=tuple(stmt.body),
                        returns_value=_has_any_return(replaced),
                    ),
                    (matched.tested_value,),
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
                arm_body = fallback_body if matched.negated else stripped
                default_body = stripped if matched.negated else fallback_body
                return AnswerBranchSite(
                    PrefixGuardedOrDispatchSite(
                        subject=matched.subject,
                        kind=kind,
                        arms=(_dummy_arm(arm_body),),
                        default_body=default_body,
                        is_whole_function=False,
                        test_nodes=(matched.test_node,),
                        replaced_nodes=tuple(container[index:]),
                        guard_expr=guard_expr,
                        guard_body=tuple(stmt.body),
                        returns_value=False,
                    ),
                    (matched.tested_value,),
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
                arm_body = fallback_body if matched.negated else stripped
                default_body = stripped if matched.negated else fallback_body
                return AnswerBranchSite(
                    PrefixGuardedOrDispatchSite(
                        subject=matched.subject,
                        kind=kind,
                        arms=(_dummy_arm(arm_body),),
                        default_body=default_body,
                        is_whole_function=False,
                        test_nodes=(matched.test_node,),
                        replaced_nodes=tuple(container[index:]),
                        guard_expr=guard_expr,
                        guard_body=tuple(stmt.body),
                        returns_value=False,
                    ),
                    (matched.tested_value,),
                )
            if stmt.orelse or _has_any_return(stmt.body):
                return None
        if stmt.orelse or _has_any_return(stmt.body):
            return None
        if isinstance(stmt.test.op, ast.And):
            if (
                container is whole_fn_body
                and len(whole_fn_body) == 1
                and all(_is_pure_subject(val) for val in prefix_values)
            ):
                other_values = [*prefix_values, *suffix_values]
                other_expr: ast.expr = (
                    other_values[0]
                    if len(other_values) == 1
                    else ast.BoolOp(op=ast.And(), values=other_values)
                )
                conditional_body: tuple[ast.stmt, ...] = (
                    ast.If(test=other_expr, body=list(stmt.body), orelse=[]),
                )
                arm_body = (ast.Pass(),) if matched.negated else conditional_body
                default_body = conditional_body if matched.negated else (ast.Pass(),)
                return AnswerBranchSite(
                    StatementDispatchSite(
                        subject=matched.subject,
                        kind=kind,
                        arms=(_dummy_arm(arm_body),),
                        default_body=default_body,
                        is_whole_function=True,
                        test_nodes=(matched.test_node,),
                        replaced_nodes=(stmt,),
                    ),
                    (matched.tested_value,),
                )
            matched_body: tuple[ast.stmt, ...] = (
                (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
                if suffix_expr is not None
                else tuple(stmt.body)
            )
            arm_body = (ast.Pass(),) if matched.negated else matched_body
            default_body = matched_body if matched.negated else (ast.Pass(),)
            return AnswerBranchSite(
                PrefixGuardedStatementDispatchSite(
                    subject=matched.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=default_body,
                    is_whole_function=False,
                    test_nodes=(matched.test_node,),
                    replaced_nodes=(stmt,),
                    guard_expr=guard_expr,
                ),
                (matched.tested_value,),
            )
        if isinstance(stmt.test.op, ast.Or):
            fallback_body: tuple[ast.stmt, ...] = (
                (ast.If(test=suffix_expr, body=list(stmt.body), orelse=[]),)
                if suffix_expr is not None
                else (ast.Pass(),)
            )
            arm_body = fallback_body if matched.negated else tuple(stmt.body)
            default_body = tuple(stmt.body) if matched.negated else fallback_body
            return AnswerBranchSite(
                OrElseGuardedStatementDispatchSite(
                    subject=matched.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=default_body,
                    is_whole_function=False,
                    test_nodes=(matched.test_node,),
                    replaced_nodes=(stmt,),
                    guard_expr=guard_expr,
                    guard_body=tuple(stmt.body),
                ),
                (matched.tested_value,),
            )
        return None

    rest_values = stmt.test.values[1:]
    rest_expr: ast.expr = (
        rest_values[0]
        if len(rest_values) == 1
        else ast.BoolOp(op=stmt.test.op, values=rest_values)
    )
    if isinstance(stmt.test.op, ast.And):
        if stmt.orelse:
            cond_body: tuple[ast.stmt, ...] = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)),
            )
            arm_body = tuple(stmt.orelse) if first.negated else cond_body
            default_body = cond_body if first.negated else tuple(stmt.orelse)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm(arm_body),),
                (first.tested_value,),
                default_body,
                (stmt,),
                is_whole,
                (first.test_node,),
            )
        tail = container[index + 1 :]
        if _always_terminates(stmt.body) and tail and _always_terminates(tail):
            cond_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body = tuple(tail) if first.negated else cond_body
            default_body = cond_body if first.negated else tuple(tail)
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm(arm_body),),
                (first.tested_value,),
                default_body,
                tuple(container[index:]),
                container is whole_fn_body and index == 0,
                (first.test_node,),
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            cond_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            arm_body = tuple(tail) if first.negated else cond_body
            default_body = cond_body if first.negated else tuple(tail)
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=default_body,
                    is_whole_function=index == 0,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
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
                    test=rest_expr,
                    body=list(_strip_trailing_continue(stmt.body)),
                    orelse=list(rewritten_tail),
                ),
            )
            arm_body = rewritten_tail if first.negated else cond_body
            default_body = cond_body if first.negated else rewritten_tail
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=default_body,
                    is_whole_function=False,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
            )
        if not _has_any_return(stmt.body):
            cond_body = (ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),)
            arm_body = (ast.Pass(),) if first.negated else cond_body
            default_body = cond_body if first.negated else (ast.Pass(),)
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=default_body,
                    is_whole_function=is_whole,
                    test_nodes=(first.test_node,),
                    replaced_nodes=(stmt,),
                ),
                (first.tested_value,),
            )
        return None
    if isinstance(stmt.test.op, ast.Or) and first.negated:
        if stmt.orelse:
            arm_stmt = ast.If(
                test=rest_expr, body=list(stmt.body), orelse=list(stmt.orelse)
            )
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm((arm_stmt,)),),
                (first.tested_value,),
                tuple(stmt.body),
                (stmt,),
                is_whole,
                (first.test_node,),
            )
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise):
            arm_stmt = ast.If(test=rest_expr, body=list(stmt.body), orelse=[])
            is_whole = container is whole_fn_body and len(whole_fn_body) == 1
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm((arm_stmt,)),),
                    default_body=tuple(stmt.body),
                    is_whole_function=is_whole,
                    test_nodes=(first.test_node,),
                    replaced_nodes=(stmt,),
                ),
                (first.tested_value,),
            )
        tail = container[index + 1 :]
        if _always_terminates(stmt.body) and tail and _always_terminates(tail):
            arm_body = (
                ast.If(test=rest_expr, body=list(stmt.body), orelse=[]),
                *tail,
            )
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm(arm_body),),
                (first.tested_value,),
                tuple(stmt.body),
                tuple(container[index:]),
                container is whole_fn_body and index == 0,
                (first.test_node,),
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
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=_strip_trailing_bare_return(stmt.body),
                    is_whole_function=index == 0,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
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
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(arm_body),),
                    default_body=_strip_trailing_continue(stmt.body),
                    is_whole_function=False,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
            )
    return None


def _extract_answer_if(
    container: list[ast.stmt],
    index: int,
    stmt: ast.If,
    whole_fn_body: list[ast.stmt],
    test_line: int,
    member_name: str,
    kind: DispatchKind,
    in_loop_body: bool,
) -> AnswerBranchSite | None:
    first = _parse_answer_test(stmt.test, member_name)
    if first is None:
        return _extract_compound_answer_if(
            container, index, stmt, whole_fn_body, test_line, member_name, kind, in_loop_body
        )
    subject_dump = ast.dump(first.subject)
    if first.negated:
        if stmt.orelse:
            if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                return None
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm(tuple(stmt.orelse)),),
                (first.tested_value,),
                tuple(stmt.body),
                (stmt,),
                container is whole_fn_body and len(whole_fn_body) == 1,
                (first.test_node,),
            )
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise):
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm((ast.Pass(),)),),
                    default_body=tuple(stmt.body),
                    is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                    test_nodes=(first.test_node,),
                    replaced_nodes=(stmt,),
                ),
                (first.tested_value,),
            )
        tail = container[index + 1 :]
        if tail and _always_terminates(stmt.body) and _always_terminates(tail):
            return _build_answer_statement_site(
                first,
                kind,
                (_dummy_arm(tuple(tail)),),
                (first.tested_value,),
                tuple(stmt.body),
                tuple(container[index:]),
                container is whole_fn_body and index == 0,
                (first.test_node,),
            )
        if (
            container is whole_fn_body
            and tail
            and _always_void_terminates(stmt.body)
            and not _has_value_return(tail)
        ):
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(tuple(tail)),),
                    default_body=_strip_trailing_bare_return(stmt.body),
                    is_whole_function=index == 0,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
            )
        if (
            in_loop_body
            and tail
            and _ends_with_continue_only(stmt.body)
            and not _has_non_continue_loop_escape(tail)
        ):
            rewritten_tail = _rewrite_loop_continues_to_returns(tail)
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm(rewritten_tail),),
                    default_body=_strip_trailing_continue(stmt.body),
                    is_whole_function=False,
                    test_nodes=(first.test_node,),
                    replaced_nodes=tuple(container[index:]),
                ),
                (first.tested_value,),
            )
        if not _has_any_return(stmt.body):
            return AnswerBranchSite(
                StatementDispatchSite(
                    subject=first.subject,
                    kind=kind,
                    arms=(_dummy_arm((ast.Pass(),)),),
                    default_body=tuple(stmt.body),
                    is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                    test_nodes=(first.test_node,),
                    replaced_nodes=(stmt,),
                ),
                (first.tested_value,),
            )
        return None

    arms_list: list[TypeTestArm] = [_dummy_arm(tuple(stmt.body))]
    values_list: list[ConstantValue] = [first.tested_value]
    test_nodes: list[ast.AST] = [first.test_node]
    current_orelse = stmt.orelse
    while len(current_orelse) == 1 and isinstance(current_orelse[0], ast.If):
        branch = current_orelse[0]
        branch_test = _parse_answer_test(branch.test, member_name)
        if (
            branch_test is None
            or branch_test.negated
            or ast.dump(branch_test.subject) != subject_dump
        ):
            return None
        arms_list.append(_dummy_arm(tuple(branch.body)))
        values_list.append(branch_test.tested_value)
        test_nodes.append(branch_test.test_node)
        current_orelse = branch.orelse

    if current_orelse:
        return _build_answer_statement_site(
            first,
            kind,
            tuple(arms_list),
            tuple(values_list),
            tuple(current_orelse),
            (stmt,),
            container is whole_fn_body and len(whole_fn_body) == 1,
            tuple(test_nodes),
        )
    tail = container[index + 1 :]
    if all(_always_terminates(arm.body) for arm in arms_list) and tail and _always_terminates(tail):
        return _build_answer_statement_site(
            first,
            kind,
            tuple(arms_list),
            tuple(values_list),
            tuple(tail),
            tuple(container[index:]),
            container is whole_fn_body and index == 0,
            tuple(test_nodes),
        )
    if (
        container is whole_fn_body
        and tail
        and all(_always_void_terminates(arm.body) for arm in arms_list)
        and not _has_value_return(tail)
    ):
        stripped_arms = tuple(_dummy_arm(_strip_trailing_bare_return(arm.body)) for arm in arms_list)
        return AnswerBranchSite(
            StatementDispatchSite(
                subject=first.subject,
                kind=kind,
                arms=stripped_arms,
                default_body=tuple(tail),
                is_whole_function=index == 0,
                test_nodes=tuple(test_nodes),
                replaced_nodes=tuple(container[index:]),
            ),
            tuple(values_list),
        )
    if (
        in_loop_body
        and tail
        and all(_ends_with_continue_only(arm.body) for arm in arms_list)
        and not _has_non_continue_loop_escape(tail)
    ):
        rewritten_tail = _rewrite_loop_continues_to_returns(tail)
        stripped_arms = tuple(_dummy_arm(_strip_trailing_continue(arm.body)) for arm in arms_list)
        return AnswerBranchSite(
            StatementDispatchSite(
                subject=first.subject,
                kind=kind,
                arms=stripped_arms,
                default_body=rewritten_tail,
                is_whole_function=False,
                test_nodes=tuple(test_nodes),
                replaced_nodes=tuple(container[index:]),
            ),
            tuple(values_list),
        )
    if not any(_has_any_return(arm.body) for arm in arms_list):
        return AnswerBranchSite(
            StatementDispatchSite(
                subject=first.subject,
                kind=kind,
                arms=tuple(arms_list),
                default_body=(ast.Pass(),),
                is_whole_function=container is whole_fn_body and len(whole_fn_body) == 1,
                test_nodes=tuple(test_nodes),
                replaced_nodes=(stmt,),
            ),
            tuple(values_list),
        )
    return None


def _is_pure_independent_guard(expr: ast.expr, subject_root_name: str) -> bool:
    """Whether `expr` is side-effect-free and does not reference `subject_root_name`
    (so evaluating the subject before `expr` preserves short-circuit semantics)."""

    match expr:
        case ast.Constant():
            return True
        case ast.Name(id=name):
            return name != subject_root_name
        case ast.Attribute(value=value):
            return _is_pure_independent_guard(value, subject_root_name)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return _is_pure_independent_guard(operand, subject_root_name)
        case ast.BoolOp(values=values):
            return all(_is_pure_independent_guard(v, subject_root_name) for v in values)
        case ast.Compare(left=left, comparators=comparators):
            return _is_pure_independent_guard(left, subject_root_name) and all(
                _is_pure_independent_guard(c, subject_root_name) for c in comparators
            )
    return False


def _subject_root_name(expr: ast.expr) -> str | None:
    current = expr
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _extract_answer_ternary(
    node: ast.IfExp,
    is_whole_return: bool,
    member_name: str,
    kind: DispatchKind,
) -> AnswerBranchSite | None:
    parsed = _parse_answer_test(node.test, member_name)
    if parsed is not None:
        if parsed.negated:
            arm_body: tuple[ast.stmt, ...] = (ast.Return(value=node.orelse),)
            default_body: tuple[ast.stmt, ...] = (ast.Return(value=node.body),)
        else:
            arm_body = (ast.Return(value=node.body),)
            default_body = (ast.Return(value=node.orelse),)
        return AnswerBranchSite(
            TernaryDispatchSite(
                subject=parsed.subject,
                kind=kind,
                arms=(_dummy_arm(arm_body),),
                default_body=default_body,
                is_whole_function=is_whole_return,
                test_nodes=(parsed.test_node,),
                node=node,
            ),
            (parsed.tested_value,),
        )
    if isinstance(node.test, ast.BoolOp) and len(node.test.values) >= 2:
        match_idx: int | None = None
        matched: ParsedAnswerTest | None = None
        for idx, value in enumerate(node.test.values):
            candidate = _parse_answer_test(value, member_name)
            if candidate is not None:
                match_idx = idx
                matched = candidate
                break
        if match_idx is None or matched is None:
            return None
        if match_idx > 0:
            root_name = _subject_root_name(matched.subject)
            if root_name is None or not all(
                _is_pure_independent_guard(val, root_name)
                for val in node.test.values[:match_idx]
            ):
                return None
        rest_values = [
            *node.test.values[:match_idx],
            *node.test.values[match_idx + 1 :],
        ]
        rest_expr: ast.expr = (
            rest_values[0]
            if len(rest_values) == 1
            else ast.BoolOp(op=node.test.op, values=rest_values)
        )
        rest_ifexp = ast.IfExp(test=rest_expr, body=node.body, orelse=node.orelse)
        if isinstance(node.test.op, ast.Or):
            if matched.negated:
                arm_body = (ast.Return(value=rest_ifexp),)
                default_body = (ast.Return(value=node.body),)
            else:
                arm_body = (ast.Return(value=node.body),)
                default_body = (ast.Return(value=rest_ifexp),)
        else:
            if matched.negated:
                arm_body = (ast.Return(value=node.orelse),)
                default_body = (ast.Return(value=rest_ifexp),)
            else:
                arm_body = (ast.Return(value=rest_ifexp),)
                default_body = (ast.Return(value=node.orelse),)
        return AnswerBranchSite(
            TernaryDispatchSite(
                subject=matched.subject,
                kind=kind,
                arms=(_dummy_arm(arm_body),),
                default_body=default_body,
                is_whole_function=is_whole_return,
                test_nodes=(matched.test_node,),
                node=node,
            ),
            (matched.tested_value,),
        )
    return None


def _extract_answer_boolop(
    expr: ast.expr,
    is_whole_return: bool,
    test_line: int,
    member_name: str,
    kind: DispatchKind,
) -> AnswerBranchSite | None:
    if not isinstance(expr, ast.BoolOp) or len(expr.values) < 2:
        return None
    match_idx: int | None = None
    matched: ParsedAnswerTest | None = None
    for idx, value in enumerate(expr.values):
        if value.lineno != test_line:
            continue
        candidate = _parse_answer_test(value, member_name)
        if candidate is not None:
            match_idx = idx
            matched = candidate
            break
    if match_idx is None or matched is None:
        return None
    rest_values = expr.values[match_idx + 1 :]
    if not rest_values:
        # Don't turn bare `x.flag` at the end of a BoolOp into another constant bool method.
        return None
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
        return AnswerBranchSite(
            BooleanPredicateDispatchSite(
                subject=matched.subject,
                kind=kind,
                arms=(_dummy_arm(arm_body),),
                default_body=default_body,
                is_whole_function=is_whole_return,
                test_nodes=(matched.test_node,),
                predicate_expr=expr,
            ),
            (matched.tested_value,),
        )
    return AnswerBranchSite(
        BoolOpSuffixDispatchSite(
            subject=matched.subject,
            kind=kind,
            arms=(_dummy_arm(arm_body),),
            default_body=default_body,
            is_whole_function=False,
            test_nodes=(matched.test_node,),
            predicate_expr=expr.values[match_idx],
            end_expr=expr.values[-1],
        ),
        (matched.tested_value,),
    )


def _match_comp_and_filter(
    if_expr: ast.expr, test_line: int, member_name: str
) -> tuple[ParsedAnswerTest, ast.expr | None, ast.expr | None] | None:
    """Match `[PREFIX and] [not] subject.flag [and SUFFIX]` on `test_line`."""

    direct = _parse_answer_test(if_expr, member_name)
    if direct is not None and if_expr.lineno == test_line:
        return (direct, None, None)
    if (
        isinstance(if_expr, ast.BoolOp)
        and isinstance(if_expr.op, ast.And)
        and len(if_expr.values) >= 2
    ):
        for idx, value in enumerate(if_expr.values):
            if value.lineno != test_line:
                continue
            parsed = _parse_answer_test(value, member_name)
            if parsed is None:
                continue
            prefix_values = if_expr.values[:idx]
            suffix_values = if_expr.values[idx + 1 :]
            prefix_expr: ast.expr | None = (
                None
                if not prefix_values
                else (
                    prefix_values[0]
                    if len(prefix_values) == 1
                    else ast.BoolOp(op=ast.And(), values=prefix_values)
                )
            )
            suffix_expr: ast.expr | None = (
                None
                if not suffix_values
                else (
                    suffix_values[0]
                    if len(suffix_values) == 1
                    else ast.BoolOp(op=ast.And(), values=suffix_values)
                )
            )
            return (parsed, prefix_expr, suffix_expr)
    return None


def _has_named_expr(nodes: Sequence[ast.AST]) -> bool:
    return any(
        isinstance(child, ast.NamedExpr)
        for root in nodes
        for child in ast.walk(root)
    )


def _extract_sum_comprehension(
    call: ast.Call,
    test_line: int,
    member_name: str,
    kind: DispatchKind,
) -> AnswerBranchSite | None:
    match call:
        case ast.Call(
            func=ast.Name(id="sum"),
            args=[ast.GeneratorExp() as comp],
            keywords=[],
        ):
            pass
        case _:
            return None
    if not comp.generators or any(gen.is_async for gen in comp.generators):
        return None
    last_gen = comp.generators[-1]
    if not last_gen.ifs or _has_named_expr((comp.elt, *last_gen.ifs)):
        return None
    matched = _match_comp_and_filter(last_gen.ifs[-1], test_line, member_name)
    if matched is None:
        return None
    parsed, prefix_expr, suffix_expr = matched
    if suffix_expr is not None:
        return None
    remaining_ifs: list[ast.expr] = list(last_gen.ifs[:-1])
    if prefix_expr is not None:
        remaining_ifs.append(prefix_expr)
    present_body: tuple[ast.stmt, ...] = (ast.Return(value=comp.elt),)
    absent_body: tuple[ast.stmt, ...] = (ast.Return(value=ast.Constant(value=0)),)
    arm_body = absent_body if parsed.negated else present_body
    default_body = present_body if parsed.negated else absent_body
    return AnswerBranchSite(
        SumComprehensionDispatchSite(
            subject=parsed.subject,
            kind=kind,
            arms=(_dummy_arm(arm_body),),
            default_body=default_body,
            is_whole_function=False,
            test_nodes=(parsed.test_node,),
            comp=comp,
            remaining_last_ifs=tuple(remaining_ifs),
        ),
        (parsed.tested_value,),
    )


def _extract_comprehension_filter(
    function: FunctionNode,
    comp: ComprehensionNode,
    test_line: int,
    member_name: str,
    kind: DispatchKind,
) -> AnswerBranchSite | None:
    if not comp.generators or any(gen.is_async for gen in comp.generators):
        return None
    last_gen = comp.generators[-1]
    if not last_gen.ifs or _has_named_expr((comp.elt, *last_gen.ifs)):
        return None
    matched_idx: int | None = None
    matched: tuple[ParsedAnswerTest, ast.expr | None, ast.expr | None] | None = None
    for idx, if_expr in enumerate(last_gen.ifs):
        candidate = _match_comp_and_filter(if_expr, test_line, member_name)
        if candidate is not None:
            matched_idx = idx
            matched = candidate
            break
    if matched_idx is None or matched is None:
        return None
    parsed, prefix_expr, suffix_expr = matched
    remaining_ifs: list[ast.expr] = list(last_gen.ifs[:matched_idx])
    if prefix_expr is not None:
        remaining_ifs.append(prefix_expr)
    trailing_ifs: list[ast.expr] = []
    if suffix_expr is not None:
        trailing_ifs.append(suffix_expr)
    trailing_ifs.extend(last_gen.ifs[matched_idx + 1 :])
    present_body: tuple[ast.stmt, ...] = (
        ast.Return(value=ast.Tuple(elts=[comp.elt], ctx=ast.Load())),
    )
    absent_body: tuple[ast.stmt, ...] = (
        ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load())),
    )
    arm_body = absent_body if parsed.negated else present_body
    default_body = present_body if parsed.negated else absent_body
    return AnswerBranchSite(
        ComprehensionFilterDispatchSite(
            subject=parsed.subject,
            kind=kind,
            arms=(_dummy_arm(arm_body),),
            default_body=default_body,
            is_whole_function=False,
            test_nodes=(parsed.test_node,),
            comp=comp,
            remaining_last_ifs=tuple(remaining_ifs),
            trailing_ifs=tuple(trailing_ifs),
            item_name=_fresh_comp_item_name(function, comp),
        ),
        (parsed.tested_value,),
    )


def find_answer_dispatch_site(
    function: FunctionNode,
    test_line: int,
    member_name: str,
    kind: DispatchKind,
) -> AnswerBranchSite | None:
    """Locate the normalized constant-answer branch in `function` on `test_line`."""

    body = _body_without_docstring(function)
    for container, index, stmt, in_loop_body in _statement_list_children(body):
        if isinstance(stmt, ast.If) and (
            stmt.test.lineno == test_line
            or (
                isinstance(stmt.test, ast.BoolOp)
                and any(value.lineno == test_line for value in stmt.test.values)
            )
        ):
            extracted = _extract_answer_if(
                container,
                index,
                stmt,
                body,
                test_line,
                member_name,
                kind,
                in_loop_body,
            )
            if extracted is not None:
                return extracted
        if isinstance(stmt, ast.Return):
            match stmt:
                case ast.Return(value=ast.IfExp() as ifexp) if (
                    ifexp.test.lineno == test_line
                    or (
                        isinstance(ifexp.test, ast.BoolOp)
                        and any(value.lineno == test_line for value in ifexp.test.values)
                    )
                ):
                    extracted = _extract_answer_ternary(
                        ifexp, container is body and len(body) == 1, member_name, kind
                    )
                    if extracted is not None:
                        return extracted
                case ast.Return(value=ast.BoolOp() as boolop) if any(
                    value.lineno == test_line for value in boolop.values
                ):
                    extracted = _extract_answer_boolop(
                        boolop,
                        container is body and len(body) == 1,
                        test_line,
                        member_name,
                        kind,
                    )
                    if extracted is not None:
                        return extracted
    branch_tests = {
        id(node.test)
        for node in ast.walk(function)
        if isinstance(node, (ast.If, ast.IfExp))
    }
    comp_ifs = {
        id(if_expr)
        for node in ast.walk(function)
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp))
        for gen in node.generators
        for if_expr in gen.ifs
    }
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            extracted = _extract_sum_comprehension(node, test_line, member_name, kind)
            if extracted is not None:
                return extracted
    for node in ast.walk(function):
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            extracted = _extract_comprehension_filter(
                function, node, test_line, member_name, kind
            )
            if extracted is not None:
                return extracted
        if isinstance(node, ast.IfExp) and (
            node.test.lineno == test_line
            or (
                isinstance(node.test, ast.BoolOp)
                and any(value.lineno == test_line for value in node.test.values)
            )
        ):
            extracted = _extract_answer_ternary(node, False, member_name, kind)
            if extracted is not None:
                return extracted
        if (
            isinstance(node, ast.BoolOp)
            and id(node) not in branch_tests
            and id(node) not in comp_ifs
            and any(value.lineno == test_line for value in node.values)
        ):
            extracted = _extract_answer_boolop(node, False, test_line, member_name, kind)
            if extracted is not None:
                return extracted
    return None


@dataclass(frozen=True)
class ConstantAnswerFamily:
    """One closed scanned family and the constant value every member resolves for `member_name`."""

    base: ResolvedClassTarget
    base_symbol: str
    member_name: str
    family_symbols: tuple[str, ...]
    resolved_answers: dict[str, ConstantValue]
    declaring_statements: dict[str, tuple[ResolvedClassTarget, ast.stmt]]
    is_class_level: bool

    @classmethod
    def resolve(
        cls,
        snapshot: CodemodSourceSnapshot,
        base_target: SourceRewriteTarget,
        member_name: str,
    ) -> ConstantAnswerFamily:
        base = ResolvedClassTarget.from_rewrite_target(snapshot, base_target)
        base_symbol = base.required_symbol(snapshot)
        verdicts = snapshot.class_overlap_verdicts
        index = verdicts.class_index
        classes = index.classes_by_symbol
        if base_symbol not in classes:
            raise NeedsInput(f"`{base_symbol}` is not in the scanned class index", NeededInput.PROOF)

        descendants = sorted(
            index.descendant_symbols(base_symbol),
            key=lambda sym: (len(index.ancestor_symbols(sym)), sym),
        )
        family_symbols = (base_symbol, *descendants)
        declared_constants: dict[str, ConstantValue] = {}
        declaring_statements: dict[str, tuple[ResolvedClassTarget, ast.stmt]] = {}
        uninitialized_annotations: set[str] = set()
        class_level_markers = 0
        instance_level_markers = 0

        for sym in family_symbols:
            indexed = classes.get(sym)
            if indexed is None:
                continue
            target = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=indexed.file_path, qualname=indexed.qualname),
            )
            parsed = snapshot.parsed_module_for_source_path(indexed.file_path)
            bindings = ModuleNominalBindingAuthority(parsed).snapshots_before((target.node.lineno,))[
                target.node.lineno
            ]
            reading = ClassBodyReading.of(target.node, bindings)
            if member_name in reading.computed:
                raise NeedsInput(
                    f"`{indexed.qualname}.{member_name}` is computed, not a constant answer",
                    NeededInput.PROOF,
                )
            for answer in reading.answers:
                if answer.member == member_name:
                    declared_constants[sym] = ast.literal_eval(answer.value)
                    break
            for statement in target.node.body:
                match statement:
                    case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) if (
                        name == member_name
                    ):
                        declaring_statements[sym] = (target, statement)
                        is_classmethod = any(
                            isinstance(dec, ast.Name) and dec.id == "classmethod"
                            for dec in statement.decorator_list
                        )
                        if is_classmethod:
                            class_level_markers += 1
                        elif not declares_abstract(statement, bindings):
                            instance_level_markers += 1
                    case ast.Assign(targets=[ast.Name(id=name)]) if name == member_name:
                        declaring_statements[sym] = (target, statement)
                        class_level_markers += 1
                    case ast.AnnAssign(target=ast.Name(id=name), value=value) if (
                        name == member_name
                    ):
                        declaring_statements[sym] = (target, statement)
                        class_level_markers += 1
                        if value is None:
                            uninitialized_annotations.add(sym)

        if len(declared_constants) < 2:
            raise NeedsInput(
                f"`{base.qualname}.{member_name}` has fewer than two constant answers in its family",
                NeededInput.PROOF,
            )

        resolved_answers: dict[str, ConstantValue] = {}
        for sym in family_symbols:
            order = verdicts.linearized(sym)
            found_value: ConstantValue | None = None
            for ancestor in order:
                if ancestor in declared_constants:
                    found_value = declared_constants[ancestor]
                    break
            if found_value is not None:
                resolved_answers[sym] = found_value
                continue
            indexed = classes[sym]
            # An abstract class with unimplemented abstract methods, or a declaration-only
            # base with subclasses (`flag: bool`), cannot provide a value for `member_name` itself.
            has_abstract = bool(indexed.abstract_method_names) or any(
                ancestor in classes and bool(classes[ancestor].abstract_method_names)
                for ancestor in order
            )
            is_uninitialized_base = sym in uninitialized_annotations and bool(
                index.descendant_symbols(sym)
            )
            if not has_abstract and not is_uninitialized_base:
                raise NeedsInput(
                    f"concrete class `{indexed.qualname}` inherits no constant answer for `{member_name}`",
                    NeededInput.PROOF,
                )

        return cls(
            base=base,
            base_symbol=base_symbol,
            member_name=member_name,
            family_symbols=family_symbols,
            resolved_answers=resolved_answers,
            declaring_statements=declaring_statements,
            is_class_level=class_level_markers > 0 and instance_level_markers == 0,
        )


def _method_header_text(
    function: FunctionNode, source_geometry: SourceTextGeometry
) -> str:
    """Extract the decorators + `def ...:` header of `function` verbatim from source."""

    first_anchor: ast.AST = (
        function.decorator_list[0] if function.decorator_list else function
    )
    start, _ = source_geometry.required_node_offsets(first_anchor)
    line_start = source_geometry.source.rfind("\n", 0, start) + 1
    first_body = function.body[0]
    body_start, _ = source_geometry.required_node_offsets(first_body)
    colon_idx = source_geometry.source.rfind(":", start, body_start)
    if colon_idx < 0:
        raise NeedsInput(
            f"cannot locate header colon for `{function.name}`",
            NeededInput.PROOF,
        )
    raw = source_geometry.source[line_start : colon_idx + 1]
    base_indent = function.col_offset
    dedented: list[str] = []
    for line in raw.splitlines():
        if len(line) >= base_indent and line[:base_indent].isspace():
            dedented.append(line[base_indent:])
        else:
            dedented.append(line.lstrip())
    return "\n".join(dedented)


def _render_in_place_override(
    header_text: str,
    body: tuple[ast.stmt, ...],
) -> str:
    lines = header_text.splitlines()
    for stmt in body:
        for raw_line in ast.unparse(ast.fix_missing_locations(stmt)).splitlines():
            lines.append(f"    {raw_line}" if raw_line else "")
    return "\n".join(lines)


def _render_block_at_indent(
    body: tuple[ast.stmt, ...], indent: str
) -> str:
    rendered_lines: list[str] = []
    for stmt in body:
        for raw_line in ast.unparse(ast.fix_missing_locations(stmt)).splitlines():
            rendered_lines.append(raw_line)
    return "\n".join(
        f"{indent}{line}" if idx > 0 and line else line
        for idx, line in enumerate(rendered_lines)
    )


def _contains_super_call(statements: Sequence[ast.stmt]) -> bool:
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super":
                return True
    return False


def _resolve_dispatch_kind(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    function: FunctionNode,
    enclosure: FunctionEnclosure,
    subject: ast.expr,
) -> DispatchKind:
    positional = (*function.args.posonlyargs, *function.args.args)
    first_param = positional[0].arg if positional else None
    return enclosure.dispatch_kind_for_receiver(subject, first_param, function, source_path, snapshot)


def _statement_full_line_span(
    geometry: SourceTextGeometry, statement: ast.stmt
) -> tuple[int, int, str]:
    """Return `(line_start, end_offset, indent)` covering `statement` and its decorators."""

    first_anchor: ast.AST = (
        statement.decorator_list[0]
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and statement.decorator_list
        else statement
    )
    start, _ = geometry.required_node_offsets(first_anchor)
    _, end = geometry.required_node_offsets(statement)
    line_start = geometry.source.rfind("\n", 0, start) + 1
    raw_prefix = geometry.source[line_start:start]
    indent = raw_prefix[: len(raw_prefix) - len(raw_prefix.lstrip(" \t"))]
    if not indent and statement.col_offset > 0:
        indent = " " * statement.col_offset
    return line_start, end, indent


def _indent_methods(methods: Sequence[str], indent: str) -> str:
    joined = "\n\n".join(method.strip("\n") for method in methods)
    return "\n".join(f"{indent}{line}" if line else "" for line in joined.splitlines())


@dataclass(frozen=True)
class ConstantAnswerPushConsumer(CodemodPayloadRecord):
    """One consumer branch site to push onto its constant-answer family."""

    target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    test_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )

    def __post_init__(self) -> None:
        if self.test_line < 1:
            raise ValueError("test_line must be a positive line number")
        if not self.method_name.isidentifier() or keyword_module.iskeyword(self.method_name):
            raise ValueError("method_name must be a Python identifier")


@dataclass(frozen=True)
class ConstantAnswerPushProof:
    """One consumer branch on a constant subtype answer proved safe to push onto its family."""

    source_path: str
    test_line: int
    function: FunctionNode
    enclosure: FunctionEnclosure
    family: ConstantAnswerFamily
    owner: ResolvedClassTarget
    branch_site: AnswerBranchSite
    in_place_base_method: bool
    base_body: tuple[ast.stmt, ...]
    overrides: tuple[tuple[ResolvedClassTarget, tuple[ast.stmt, ...]], ...]
    captured_args: tuple[CapturedArgument, ...]
    is_async: bool
    method_name: str
    docstring: str | None
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = ()

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        test_line: int,
        base_target: SourceRewriteTarget,
        member_name: str,
        method_name: str,
        docstring: str | None = None,
        family: ConstantAnswerFamily | None = None,
    ) -> ConstantAnswerPushProof:
        if (
            not method_name.isidentifier()
            or keyword_module.iskeyword(method_name)
            or (method_name.startswith("__") and method_name.endswith("__"))
        ):
            raise NeedsInput(
                f"`{method_name}` is not a valid non-dunder method name",
                NeededInput.OWNER_PICK,
            )
        resolved_family = (
            family
            if family is not None
            else ConstantAnswerFamily.resolve(snapshot, base_target, member_name)
        )
        source_path, enclosure, function = TypeTestDispatchProof.locate_function(
            snapshot, target, test_line=test_line
        )
        raw_site = find_answer_dispatch_site(
            function, test_line, member_name, InstanceDispatchKind()
        )
        if raw_site is None:
            raise NeedsInput(
                f"branch on `{member_name}` at {source_path}:{test_line} is not a pushable dispatch shape",
                NeededInput.SOURCE_CHANGE,
            )
        kind = _resolve_dispatch_kind(
            snapshot, source_path, function, enclosure, raw_site.dispatch.subject
        )
        branch_site = AnswerBranchSite(
            replace(raw_site.dispatch, kind=kind),
            raw_site.arm_values,
        )

        site = branch_site.dispatch
        positional = (*function.args.posonlyargs, *function.args.args)
        first_param = positional[0].arg if positional else None
        enclosure_symbol = (
            enclosure.class_receiver_symbol(snapshot)
        )
        in_place = (
            enclosure_symbol is not None
            and enclosure_symbol in resolved_family.family_symbols
            and isinstance(site.subject, ast.Name)
            and site.subject.id == first_param
            and site.is_whole_function
            and method_name == function.name
        )

        verdicts = snapshot.class_overlap_verdicts
        index = verdicts.class_index
        classes = index.classes_by_symbol
        if in_place and enclosure_symbol is not None:
            owner_symbol = enclosure_symbol
            owner_target = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(
                    file_path=classes[owner_symbol].file_path,
                    qualname=classes[owner_symbol].qualname,
                ),
            )
            sub_descendants = sorted(
                index.descendant_symbols(owner_symbol),
                key=lambda sym: (len(index.ancestor_symbols(sym)), sym),
            )
            scope_symbols = (owner_symbol, *sub_descendants)
        else:
            owner_symbol = resolved_family.base_symbol
            owner_target = resolved_family.base
            scope_symbols = resolved_family.family_symbols

        # Ensure the method name is free on the target sub-hierarchy (or only on owner itself when in-place).
        for sym in scope_symbols:
            indexed = classes[sym]
            target_cls = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=indexed.file_path, qualname=indexed.qualname),
            )
            if in_place and sym == owner_symbol:
                continue
            if method_name in target_cls.bound_names:
                raise NeedsInput(
                    f"`{target_cls.qualname}` already binds `{method_name}`",
                    NeededInput.OWNER_PICK if not in_place else NeededInput.SOURCE_CHANGE,
                )

        # Build the raw body for each constant value.
        body_by_value: dict[ConstantValue, tuple[ast.stmt, ...]] = {}
        for value, arm in zip(branch_site.arm_values, site.arms, strict=True):
            body_by_value[value] = arm.body

        def raw_body_for(value: ConstantValue) -> tuple[ast.stmt, ...]:
            return body_by_value.get(value, site.default_body)

        if owner_symbol in resolved_family.resolved_answers:
            base_value = resolved_family.resolved_answers[owner_symbol]
        else:
            scoped_counts = Counter(
                resolved_family.resolved_answers[sym]
                for sym in scope_symbols
                if sym in resolved_family.resolved_answers
            )
            if not scoped_counts:
                raise NeedsInput(
                    f"no member under `{owner_target.qualname}` resolves a constant answer for `{member_name}`",
                    NeededInput.PROOF,
                )
            base_value = max(
                scoped_counts,
                key=lambda val: (scoped_counts[val], 0 if val is False else 1, str(val)),
            )

        assigned: dict[str, ConstantValue] = {owner_symbol: base_value}
        override_symbols: list[tuple[str, ConstantValue]] = []
        for sym in scope_symbols[1:]:
            if sym not in resolved_family.resolved_answers:
                continue
            order = verdicts.linearized(sym)
            inherited_value = next(
                (assigned[anc] for anc in order[1:] if anc in assigned),
                None,
            )
            wanted = resolved_family.resolved_answers[sym]
            if inherited_value != wanted:
                assigned[sym] = wanted
                override_symbols.append((sym, wanted))

        if not override_symbols:
            raise NeedsInput(
                f"every member of `{owner_target.qualname}` takes the same branch for `{member_name}` at {source_path}:{test_line}",
                NeededInput.PROOF,
            )

        # Rewrite bodies cleanly for either in-place or pushed family method.
        outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = ()
        if in_place:
            all_raw = (raw_body_for(base_value), *(raw_body_for(v) for _, v in override_symbols))
            if any(_contains_super_call(body) for body in all_raw):
                raise NeedsInput(
                    f"`{function.name}` calls `super()`, which cannot move onto subclasses automatically",
                    NeededInput.SOURCE_CHANGE,
                )
            base_body = raw_body_for(base_value)
            overrides = tuple(
                (
                    ResolvedClassTarget.from_rewrite_target(
                        snapshot,
                        SourceRewriteTarget(
                            file_path=classes[sym].file_path,
                            qualname=classes[sym].qualname,
                        ),
                    ),
                    raw_body_for(val),
                )
                for sym, val in override_symbols
            )
            captured_args: tuple[CapturedArgument, ...] = ()
            is_async = isinstance(function, ast.AsyncFunctionDef)
            local_exclusions = frozenset(
                arg.arg
                for arg in ast.walk(function.args)
                if isinstance(arg, ast.arg)
            )
        else:
            receiver = site.kind.receiver
            raw_bodies = (
                raw_body_for(base_value),
                *(raw_body_for(val) for _, val in override_symbols),
            )
            enclosing_locals = enclosure.enclosing_local_names()
            outer_receiver_aliases = _outer_receiver_aliases(
                function, site, raw_bodies, enclosing_locals=enclosing_locals
            )
            base_body = _rewrite_subject_to_receiver(
                raw_body_for(base_value),
                site.subject,
                receiver,
                outer_receiver_aliases=outer_receiver_aliases,
            )
            overrides = tuple(
                (
                    ResolvedClassTarget.from_rewrite_target(
                        snapshot,
                        SourceRewriteTarget(
                            file_path=classes[sym].file_path,
                            qualname=classes[sym].qualname,
                        ),
                    ),
                    _rewrite_subject_to_receiver(
                        raw_body_for(val),
                        site.subject,
                        receiver,
                        outer_receiver_aliases=outer_receiver_aliases,
                    ),
                )
                for sym, val in override_symbols
            )
            captured_args, is_async = _analyze_arm_bodies(
                function,
                site,
                (base_body, *(body for _, body in overrides)),
                outer_receiver_aliases=outer_receiver_aliases,
                enclosing_locals=enclosing_locals,
            )
            local_exclusions = frozenset(
                (receiver, *(arg.param_name for arg in captured_args))
            )

        parsed_source = snapshot.parsed_module_for_source_path(source_path)
        rationale_probe = f"Push `{member_name}` branch onto `{owner_target.qualname}.{method_name}`."
        _cross_module_imports_for_body(
            snapshot,
            parsed_source,
            test_line,
            owner_target,
            base_body,
            local_exclusions,
            rationale_probe,
        )
        for target_cls, override_body in overrides:
            _cross_module_imports_for_body(
                snapshot,
                parsed_source,
                test_line,
                target_cls,
                override_body,
                local_exclusions,
                rationale_probe,
            )

        return cls(
            source_path=source_path,
            test_line=test_line,
            function=function,
            enclosure=enclosure,
            family=resolved_family,
            owner=owner_target,
            branch_site=branch_site,
            in_place_base_method=in_place,
            base_body=base_body,
            overrides=overrides,
            captured_args=captured_args,
            is_async=is_async,
            method_name=method_name,
            docstring=docstring.strip() if docstring else None,
            outer_receiver_aliases=outer_receiver_aliases,
        )

    def consumed_attribute_ids(self) -> frozenset[int]:
        member = self.family.member_name
        return frozenset(
            id(node)
            for test_node in self.branch_site.dispatch.test_nodes
            for node in ast.walk(test_node)
            if isinstance(node, ast.Attribute) and node.attr == member
        )

    def consumer_replacement(
        self, geometry: SourceTextGeometry
    ) -> tuple[int, int, str]:
        site = self.branch_site.dispatch
        start, end = site.replaced_offsets(geometry)
        line_start = geometry.source.rfind("\n", 0, start) + 1
        line_indent = geometry.source[line_start:start]
        if not line_indent.isspace():
            line_indent = ""
        if self.in_place_base_method:
            return start, end, _render_block_at_indent(self.base_body, line_indent)
        call_args = ", ".join(arg.call_expr for arg in self.captured_args)
        call = f"{ast.unparse(site.subject)}.{self.method_name}({call_args})"
        if self.is_async:
            call = f"await {call}"
        return start, end, site.render_call_replacement(call, line_indent)

    def _resolved_annotations(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> ResolvedMethodAnnotations:
        site = self.branch_site.dispatch
        parsed_source = snapshot.parsed_module_for_source_path(self.source_path)
        all_destinations = (
            self.family.base,
            *(target_cls for target_cls, _body in self.overrides),
        )
        return ResolvedMethodAnnotations.for_push(
            snapshot,
            parsed_source,
            self.test_line,
            self.enclosure,
            self.function,
            self.outer_receiver_aliases,
            self.captured_args,
            all_destinations,
            propagates_function_return_type=site.propagates_function_return_type,
            is_void_method=site.is_void_method,
            constant_bool_return=site.returns_constant_bool,
            rationale=rationale,
        )

    def appended_methods(
        self,
        snapshot: CodemodSourceSnapshot,
        source_geometry: SourceTextGeometry,
        rationale: str = "",
    ) -> tuple[tuple[str, ResolvedClassTarget, str], ...]:
        """Return `(class_symbol, target_cls, rendered_method)` entries this push adds."""

        entries: list[tuple[str, ResolvedClassTarget, str]] = []
        if self.in_place_base_method:
            header_text = _method_header_text(self.function, source_geometry)
            for target_cls, override_body in self.overrides:
                entries.append(
                    (
                        target_cls.required_symbol(snapshot),
                        target_cls,
                        _render_in_place_override(header_text, override_body),
                    )
                )
            return tuple(entries)

        site = self.branch_site.dispatch
        param_names = tuple(arg.param_name for arg in self.captured_args)
        method_annotations = self._resolved_annotations(snapshot, rationale)
        param_annotations = method_annotations.param_map
        base_method = _render_method(
            site.kind,
            self.method_name,
            param_names,
            self.base_body,
            is_async=self.is_async,
            docstring=self.docstring,
            param_annotations=param_annotations,
            return_annotation=method_annotations.return_annotation,
        )
        entries.append((self.family.base_symbol, self.family.base, base_method))
        for target_cls, override_body in self.overrides:
            override_method = _render_method(
                site.kind,
                self.method_name,
                param_names,
                override_body,
                is_async=self.is_async,
                param_annotations=param_annotations,
                return_annotation=method_annotations.return_annotation,
            )
            entries.append((target_cls.required_symbol(snapshot), target_cls, override_method))
        return tuple(entries)

    def cross_module_import_edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        if self.in_place_base_method:
            local_exclusions = frozenset(
                arg.arg
                for arg in ast.walk(self.function.args)
                if isinstance(arg, ast.arg)
            )
            annotation_imports: tuple[NominalSourceEdit, ...] = ()
        else:
            param_names = tuple(arg.param_name for arg in self.captured_args)
            local_exclusions = frozenset((self.branch_site.dispatch.kind.receiver, *param_names))
            annotation_imports = self._resolved_annotations(
                snapshot, rationale
            ).import_edits
        parsed_source = snapshot.parsed_module_for_source_path(self.source_path)
        extra: list[NominalSourceEdit] = [
            *annotation_imports,
            *_cross_module_imports_for_body(
                snapshot,
                parsed_source,
                self.test_line,
                self.owner,
                self.base_body,
                local_exclusions,
                rationale,
            ),
        ]
        for target_cls, override_body in self.overrides:
            extra.extend(
                _cross_module_imports_for_body(
                    snapshot,
                    parsed_source,
                    self.test_line,
                    target_cls,
                    override_body,
                    local_exclusions,
                    rationale,
                )
            )
        return tuple(extra)


def _can_delete_constant_answer(
    snapshot: CodemodSourceSnapshot,
    family: ConstantAnswerFamily,
    consumed_attribute_ids: frozenset[int],
) -> bool:
    for parsed in snapshot.parsed_modules:
        if family.member_name not in parsed.source:
            continue
        for node in parsed.fact(ModuleNodeIndex).attributes_by_attr.get(
            family.member_name, ()
        ):
            if id(node) not in consumed_attribute_ids:
                return False
    classes = snapshot.overlap_class_index.classes_by_symbol
    for sym in family.declaring_statements:
        try:
            NamespaceReaders(snapshot).require_unread(
                classes[sym], frozenset((family.member_name,))
            )
        except ValueError:
            return False
        indexed = classes[sym]
        if (
            sym == family.base_symbol
            and indexed.abstract_method_names == frozenset((family.member_name,))
        ):
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class PushConstantSubtypeAnswerOperation(RepositorySourceReprovedOperation):
    """Push consumer branches on a family's constant subtype answer onto the family,
    and delete the constant answer declarations atomically when no reads remain."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    consumers: tuple[ConstantAnswerPushConsumer, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ConstantAnswerPushConsumer)
    )
    delete_if_unread: bool = codemod_payload_field(
        BooleanPayloadValueCodec(), default=True
    )

    def __post_init__(self) -> None:
        if not self.member_name.isidentifier() or keyword_module.iskeyword(self.member_name):
            raise ValueError("member_name must be a Python identifier")
        if not self.consumers:
            raise ValueError("consumers must be non-empty")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        family = ConstantAnswerFamily.resolve(snapshot, self.target, self.member_name)
        proofs: list[ConstantAnswerPushProof] = []
        seen_method_names: set[str] = set()
        for consumer in self.consumers:
            if consumer.method_name in seen_method_names:
                raise NeedsInput(
                    f"duplicate method name `{consumer.method_name}` for consumers of `{family.base.qualname}.{self.member_name}`",
                    NeededInput.OWNER_PICK,
                )
            seen_method_names.add(consumer.method_name)
            proofs.append(
                ConstantAnswerPushProof.proved(
                    snapshot,
                    consumer.target,
                    consumer.test_line,
                    self.target,
                    self.member_name,
                    consumer.method_name,
                    consumer.docstring,
                    family=family,
                )
            )

        consumed_ids = frozenset().union(*(proof.consumed_attribute_ids() for proof in proofs))
        delete_answer = self.delete_if_unread and _can_delete_constant_answer(
            snapshot, family, consumed_ids
        )
        method_list = ", ".join(f"`{proof.method_name}`" for proof in proofs)
        suffix = f" and deletes unread `{self.member_name}`" if delete_answer else ""
        rationale = self.rationale_text(
            f"Pushes `{family.base.qualname}.{self.member_name}` branch(es) onto {method_list}{suffix}."
        )

        geometries: dict[str, SourceTextGeometry] = {}
        replacements_by_file: dict[str, list[tuple[int, int, str]]] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        def add_span(file_path: str, start: int, end: int, text: str) -> None:
            replacements_by_file.setdefault(file_path, []).append((start, end, text))

        appended_by_class: dict[str, tuple[ResolvedClassTarget, list[tuple[str, str]]]] = {}
        extra_edits: list[NominalSourceEdit] = []
        for proof in proofs:
            source_geom = geometry(proof.source_path)
            start, end, replacement = proof.consumer_replacement(source_geom)
            add_span(proof.source_path, start, end, replacement)
            for sym, target_cls, rendered in proof.appended_methods(
                snapshot, source_geom, rationale
            ):
                bucket = appended_by_class.setdefault(sym, (target_cls, []))
                bucket[1].append((proof.method_name, rendered))
            extra_edits.extend(proof.cross_module_import_edits(snapshot, rationale))

        dropped_by_file: dict[str, list[ast.AST]] = {}
        for sym in family.family_symbols:
            appended_entry = appended_by_class.get(sym)
            decl_entry = family.declaring_statements.get(sym) if delete_answer else None
            if decl_entry is None and appended_entry is None:
                continue
            if decl_entry is None and appended_entry is not None:
                target_cls, methods = appended_entry
                for method_name, rendered in methods:
                    extra_edits.extend(
                        _class_append_edits(
                            target_cls.file_path,
                            geometry(target_cls.file_path),
                            target_cls.node,
                            rendered,
                            member_names=(method_name,),
                            rationale=rationale,
                        )
                    )
            elif decl_entry is not None and appended_entry is None:
                target_cls, stmt = decl_entry
                deletion = StatementDeletionSource(
                    source=snapshot.sources_by_file_path[target_cls.file_path],
                    node=target_cls.node,
                    file_path=target_cls.file_path,
                )
                for span_repl in deletion.replacements_for_statements((stmt,)):
                    add_span(
                        target_cls.file_path,
                        span_repl.start_offset,
                        span_repl.end_offset,
                        span_repl.replacement_source,
                    )
                dropped_by_file.setdefault(target_cls.file_path, []).append(stmt)
            elif decl_entry is not None and appended_entry is not None:
                target_cls, stmt = decl_entry
                _cls, methods = appended_entry
                line_start, end, indent = _statement_full_line_span(
                    geometry(target_cls.file_path), stmt
                )
                add_span(
                    target_cls.file_path,
                    line_start,
                    end,
                    _indent_methods([rendered for _name, rendered in methods], indent),
                )
                dropped_by_file.setdefault(target_cls.file_path, []).append(stmt)

        for file_path, dropped in dropped_by_file.items():
            extra_edits.extend(obsolete_import_edits(snapshot, file_path, tuple(dropped), rationale))

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


@dataclass(frozen=True, kw_only=True)
class DeleteConstantSubtypeAnswerOperation(RepositorySourceReprovedOperation):
    """Delete an unread constant subtype answer from its declaring family once consumers are pushed."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.member_name.isidentifier() or keyword_module.iskeyword(self.member_name):
            raise ValueError("member_name must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        family = ConstantAnswerFamily.resolve(snapshot, self.target, self.member_name)
        for parsed in snapshot.parsed_modules:
            if self.member_name not in parsed.source:
                continue
            for node in parsed.fact(ModuleNodeIndex).attributes_by_attr.get(
                self.member_name, ()
            ):
                raise NeedsInput(
                    f"{parsed.file_path}:{node.lineno} still reads `.{self.member_name}`",
                    NeededInput.SOURCE_CHANGE,
                )
        classes = snapshot.overlap_class_index.classes_by_symbol
        for sym in family.declaring_statements:
            NamespaceReaders(snapshot).require_unread(
                classes[sym], frozenset((self.member_name,))
            )
            indexed = classes[sym]
            if (
                sym == family.base_symbol
                and indexed.abstract_method_names == frozenset((self.member_name,))
            ):
                raise NeedsInput(
                    f"deleting `{self.member_name}` would remove the sole `@abstractmethod` on `{indexed.qualname}`",
                    NeededInput.PROOF,
                )

        rationale = self.rationale_text(
            f"Deletes unread constant subtype answer `{self.member_name}` from `{family.base.qualname}` family."
        )
        edits: list[NominalSourceEdit] = []
        by_file: dict[str, list[ast.AST]] = {}
        for _sym, (target_cls, stmt) in family.declaring_statements.items():
            deletion = StatementDeletionSource(
                source=snapshot.sources_by_file_path[target_cls.file_path],
                node=target_cls.node,
                file_path=target_cls.file_path,
            )
            edits.extend(
                deletion.physical_edits(
                    file_path=target_cls.file_path,
                    replacements=deletion.replacements_for_statements((stmt,)),
                    rationale=rationale,
                )
            )
            by_file.setdefault(target_cls.file_path, []).append(stmt)
        for file_path, dropped in by_file.items():
            edits.extend(obsolete_import_edits(snapshot, file_path, tuple(dropped), rationale))
        return tuple(edits)


class ConstantSubtypeAnswerFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
    ConstantSubtypeAnswerConcept,
    ABC,
):
    """Push branch consumers of a constant subtype answer onto its family and delete it when unread."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not isinstance(finding.metrics, ConstantSubtypeAnswerMetrics):
            return self.rejected_evaluation(
                "pushing a constant subtype answer needs source context and family metrics"
            )
        metrics = finding.metrics
        if not metrics.consumer_locations:
            return self.rejected_evaluation("finding carries no branch-read consumers")
        if metrics.declaring_family.startswith("open ") or ", " in metrics.declaring_family:
            return self.stopped_evaluation(
                NeedsInput(
                    f"family `{metrics.declaring_family}` is open or spans multiple roots",
                    NeededInput.PROOF,
                )
            )
        snapshot = context.execution_snapshot()
        classes = snapshot.overlap_class_index.classes_by_symbol
        base_indexed = classes.get(metrics.declaring_family)
        if base_indexed is None:
            return self.rejected_evaluation(
                f"declaring family `{metrics.declaring_family}` left the source"
            )
        base_target = source_rewrite_target(base_indexed)
        try:
            family = ConstantAnswerFamily.resolve(snapshot, base_target, metrics.member_name)
        except ValueError as error:
            return self.stopped_evaluation(error)

        pick = finding.metrics.picked_base
        consumers: list[ConstantAnswerPushConsumer] = []
        for consumer in metrics.consumer_locations:
            consumer_target = SourceRewriteTarget(
                file_path=consumer.file_path, qualname=consumer.symbol
            )
            try:
                _path, _enclosure, function = TypeTestDispatchProof.locate_function(
                    snapshot, consumer_target
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            raw_site = find_answer_dispatch_site(
                function, consumer.line, metrics.member_name, InstanceDispatchKind()
            )
            if raw_site is None:
                return self.stopped_evaluation(
                    NeedsInput(
                        f"branch on `{metrics.member_name}` at {consumer.file_path}:{consumer.line} "
                        "is not a pushable dispatch shape",
                        NeededInput.SOURCE_CHANGE,
                    )
                )
            is_dunder = function.name.startswith("__") and function.name.endswith("__")
            if raw_site.dispatch.is_whole_function and not is_dunder:
                method_name = function.name
                docstring: str | None = ast.get_docstring(function)
            elif pick is not None and len(metrics.consumer_locations) == 1:
                method_name = pick.name
                docstring = pick.judgement
            else:
                probe_name = "__nra_probe_answer_method"
                try:
                    ConstantAnswerPushProof.proved(
                        snapshot,
                        consumer_target,
                        consumer.line,
                        base_target,
                        metrics.member_name,
                        probe_name,
                        None,
                        family=family,
                    )
                except ValueError as error:
                    return self.stopped_evaluation(error)
                return self.stopped_evaluation(
                    NeedsInput(
                        f"embedded branch on `{metrics.member_name}` in `{consumer.symbol}` "
                        f"can push onto `{base_indexed.qualname}`; supply `owner` "
                        f"`{metrics.consumer_symbol}` with a method name and one-line judgement "
                        "in `--codemod-owner-picks`",
                        NeededInput.OWNER_PICK,
                    )
                )
            consumers.append(
                ConstantAnswerPushConsumer(
                    target=consumer_target,
                    test_line=consumer.line,
                    method_name=method_name,
                    docstring=docstring,
                )
            )

        operation = PushConstantSubtypeAnswerOperation(
            target=base_target,
            member_name=metrics.member_name,
            consumers=tuple(consumers),
            delete_if_unread=True,
        )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-push-constant-subtype-answer",
            reason=(
                f"Pushes branch consumer(s) of `{base_indexed.qualname}.{metrics.member_name}` "
                "onto polymorphic methods on the family."
            ),
        ).with_operation(operation)
        try:
            sim = recipe.simulate(snapshot)
            if not sim.is_clean:
                return self.stopped_evaluation(
                    NeedsInput(
                        f"pushing `{metrics.member_name}` consumers produced conflicting edits",
                        NeededInput.SOURCE_CHANGE,
                    )
                )
        except ValueError as error:
            return self.stopped_evaluation(error)

        return self.executable_evaluation(recipe)


__all__ = (
    "AnswerBranchSite",
    "ConstantAnswerFamily",
    "ConstantAnswerPushConsumer",
    "ConstantAnswerPushProof",
    "ConstantSubtypeAnswerFindingRecipeSynthesizer",
    "DeleteConstantSubtypeAnswerOperation",
    "OrElseGuardedStatementDispatchSite",
    "ParsedAnswerTest",
    "PushConstantSubtypeAnswerOperation",
    "find_answer_dispatch_site",
)
