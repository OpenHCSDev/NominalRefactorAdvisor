"""Split a frozen record's derived bool state into a subtype and push consumer branches onto it.

When a frozen dataclass declares a `-> bool` property or zero-argument method
derived from its own fields (`self.field is not None`, `self.verdict == "INTERVENE"`,
etc.) and callers branch on it, deciding the subtype once at construction (`Record`
for the default/False state and `@dataclass(frozen=True) class Subtype(Record)` for
the True state) moves every consumer branch onto polymorphic methods of the family
and deletes (or, when non-branching data reads remain, constantizes) the derived bool.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from nominal_refactor_advisor.ast_field_decoders import optional_values
from nominal_refactor_advisor.ast_projection import CallKeywords, ExpressionReference
from nominal_refactor_advisor.ast_tools import AstParentIndex
from nominal_refactor_advisor.class_index import (
    PROPERTY_DECORATORS,
    ModuleNominalBindingAuthority,
)
from nominal_refactor_advisor.class_symbols import ClassSymbolResolutionAuthority
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    BooleanPayloadValueCodec,
    PayloadRecordArrayValueCodec,
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
    OwnStateBranchAnswerMetrics,
    RefactorFinding,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import ReceiverSpan, ReceiverType
from nominal_refactor_advisor.refactor_concepts import OwnStateBranchAnswerConcept
from nominal_refactor_advisor.refactors.class_overlap import (
    obsolete_import_edits,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    AnswerBranchSite,
    ConstantAnswerPushConsumer,
    _contains_super_call,
    _indent_methods,
    _method_header_text,
    _render_in_place_override,
    _statement_full_line_span,
    _subject_allowed_for_site,
    find_answer_dispatch_site,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
)
from nominal_refactor_advisor.refactors.restated_overrides import NamespaceReaders
from nominal_refactor_advisor.refactors.threaded_absence import (
    _module_exports_name,
    _module_top_level_bound_names,
)
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    CapturedArgument,
    FunctionEnclosure,
    FunctionNode,
    InstanceDispatchKind,
    OuterReceiverAlias,
    ResolvedMethodAnnotations,
    TypeTestDispatchProof,
    _analyze_arm_bodies,
    _class_append_edits,
    _cross_module_imports_for_body,
    _has_future_annotations,
    _names_referenced_outside,
    _names_stored_in,
    _outer_receiver_aliases,
    _render_method,
    _rewrite_subject_to_receiver,
    _upgrade_dispatch_site_to_assigning,
)
from nominal_refactor_advisor.relocation_observations import (
    calls_only_pure_builtins,
    statements_after_docstring,
    takes_only_receiver,
)


_NON_DATACLASS_BASE_SYMBOLS = frozenset(
    (
        "builtins.object",
        "abc.ABC",
        "typing.Generic",
    )
)


class _FieldDefault(ABC):
    """Whether a frozen dataclass field is required, carries a default expression, or uses a factory."""

    @abstractmethod
    def bind_missing(
        self, field_name: str, file_path: str, line: int
    ) -> ast.expr | None:
        raise NotImplementedError

    def require_factory_compatible(
        self, record_qualname: str, field_name: str, factory_name: str
    ) -> None:
        del record_qualname, field_name, factory_name

    def render_default_suffix(self) -> str:
        return ""


@dataclass(frozen=True)
class _RequiredFieldDefault(_FieldDefault):
    """Required field with no default."""

    def bind_missing(
        self, field_name: str, file_path: str, line: int
    ) -> ast.expr | None:
        raise NeedsInput(
            f"constructor call at {file_path}:{line} omits required field `{field_name}`",
            NeededInput.PROOF,
        )


@dataclass(frozen=True)
class _ExprFieldDefault(_FieldDefault):
    """Field declaring a default expression."""

    expr: ast.expr

    def bind_missing(
        self, field_name: str, file_path: str, line: int
    ) -> ast.expr | None:
        del field_name, file_path, line
        return self.expr

    def render_default_suffix(self) -> str:
        return f" = {ast.unparse(self.expr)}"


@dataclass(frozen=True)
class _FactoryFieldDefault(_FieldDefault):
    """Field declaring `field(default_factory=...)`."""

    def bind_missing(
        self, field_name: str, file_path: str, line: int
    ) -> ast.expr | None:
        del field_name, file_path, line
        return None

    def require_factory_compatible(
        self, record_qualname: str, field_name: str, factory_name: str
    ) -> None:
        raise NeedsInput(
            f"field `{field_name}` on `{record_qualname}` uses `default_factory`; cannot synthesize `{factory_name}`",
            NeededInput.PROOF,
        )


@dataclass(frozen=True)
class _DataclassFieldSpec:
    """One `__init__` parameter derived from a frozen dataclass field declaration."""

    name: str
    annotation: ast.expr
    default: _FieldDefault
    kw_only: bool


def _is_none_const(node: ast.AST) -> bool:
    match node:
        case ast.Constant(value=None):
            return True
    return False


def _is_kw_only_annotation(annotation: ast.expr) -> bool:
    ref = ExpressionReference.decoded(annotation).terminal_name
    return ref == "KW_ONLY"


def _is_classvar_or_initvar(annotation: ast.expr) -> bool:
    match annotation:
        case ast.Subscript(value=base):
            return ExpressionReference.decoded(base).terminal_name in (
                "ClassVar",
                "InitVar",
            )
    return ExpressionReference.decoded(annotation).terminal_name in (
        "ClassVar",
        "InitVar",
    )


def _find_frozen_dataclass_decorator(
    class_node: ast.ClassDef,
    bindings_snapshot,
) -> tuple[ast.Call, bool] | None:
    """Return `(decorator_call, class_kw_only)` when `class_node` is `@dataclass(frozen=True, ...)`."""

    for decorator in class_node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(decorator.func)
        ).attribute_chain
        if parts is None or not bindings_snapshot.reference_for(parts).has_bound_name(
            "dataclasses.dataclass"
        ):
            continue
        frozen = False
        kw_only = False
        for kw in CallKeywords.of(decorator.keywords).named:
            if kw.name == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                frozen = True
            elif kw.name == "kw_only" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                kw_only = True
        if frozen:
            return decorator, kw_only
    return None


def _extract_dataclass_fields(
    snapshot: CodemodSourceSnapshot,
    record_symbol: str,
) -> tuple[tuple[_DataclassFieldSpec, ...], ast.Call]:
    """Extract ordered `__init__` fields and the `@dataclass(frozen=True, ...)` decorator."""

    verdicts = snapshot.class_overlap_verdicts
    index = verdicts.class_index
    classes = index.classes_by_symbol
    order = verdicts.linearized(record_symbol)

    fields_by_name: dict[str, _DataclassFieldSpec] = {}
    record_decorator: ast.Call | None = None

    for sym in reversed(order):
        if sym in _NON_DATACLASS_BASE_SYMBOLS:
            continue
        indexed = classes.get(sym)
        if indexed is None:
            raise NeedsInput(
                f"ancestor `{sym}` of `{record_symbol}` is outside the scanned repository",
                NeededInput.PROOF,
            )
        cls_target = ResolvedClassTarget.from_rewrite_target(
            snapshot,
            SourceRewriteTarget(file_path=indexed.file_path, qualname=indexed.qualname),
        )
        parsed = snapshot.parsed_module_for_source_path(indexed.file_path)
        bindings = ModuleNominalBindingAuthority(parsed).snapshot_before(
            cls_target.node.lineno
        )
        for stmt in cls_target.node.body:
            match stmt:
                case ast.FunctionDef(name="__init__" | "__new__") | ast.AsyncFunctionDef(
                    name="__init__" | "__new__"
                ):
                    raise NeedsInput(
                        f"`{cls_target.qualname}` declares custom `{stmt.name}`; cannot synthesize subtype factory",
                        NeededInput.PROOF,
                    )
        dc_info = _find_frozen_dataclass_decorator(cls_target.node, bindings)
        if sym == record_symbol:
            if dc_info is None:
                raise NeedsInput(
                    f"`{cls_target.qualname}` is not a `@dataclass(frozen=True)`",
                    NeededInput.PROOF,
                )
            record_decorator = dc_info[0]
        class_kw_only = dc_info[1] if dc_info is not None else False
        after_kw_only_sentinel = False
        for stmt in cls_target.node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(
                stmt.target, ast.Name
            ):
                continue
            if _is_kw_only_annotation(stmt.annotation):
                after_kw_only_sentinel = True
                continue
            if _is_classvar_or_initvar(stmt.annotation):
                continue
            if dc_info is None:
                raise NeedsInput(
                    f"non-dataclass ancestor `{cls_target.qualname}` declares annotated attribute `{stmt.target.id}`",
                    NeededInput.PROOF,
                )
            field_name = stmt.target.id
            init_enabled = True
            field_kw_only = class_kw_only or after_kw_only_sentinel
            field_default: _FieldDefault = _RequiredFieldDefault()
            values = tuple(optional_values(stmt))
            if values:
                raw_value = values[0]
                if isinstance(raw_value, ast.Call):
                    callee_parts = ExpressionReference.decoded(
                        raw_value.func
                    ).attribute_chain
                    is_dc_field = (
                        callee_parts is not None
                        and bindings.reference_for(callee_parts).has_bound_name(
                            "dataclasses.field"
                        )
                    )
                else:
                    is_dc_field = False
                if is_dc_field and isinstance(raw_value, ast.Call):
                    if CallKeywords.of(raw_value.keywords).splats or raw_value.args:
                        raise NeedsInput(
                            f"field `{field_name}` on `{cls_target.qualname}` uses complex `field(...)` arguments",
                            NeededInput.PROOF,
                        )
                    for kw in CallKeywords.of(raw_value.keywords).named:
                        if kw.name == "init" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            init_enabled = False
                        elif kw.name == "kw_only" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            field_kw_only = True
                        elif kw.name == "default":
                            field_default = _ExprFieldDefault(kw.value)
                        elif kw.name == "default_factory":
                            field_default = _FactoryFieldDefault()
                else:
                    field_default = _ExprFieldDefault(raw_value)
            if not init_enabled:
                fields_by_name.pop(field_name, None)
                continue
            fields_by_name[field_name] = _DataclassFieldSpec(
                name=field_name,
                annotation=stmt.annotation,
                default=field_default,
                kw_only=field_kw_only,
            )

    if record_decorator is None:
        raise NeedsInput(
            f"`{record_symbol}` is not a `@dataclass(frozen=True)`",
            NeededInput.PROOF,
        )
    # Python dataclasses place positional fields first (in MRO order), then kw_only fields.
    pos_fields = [f for f in fields_by_name.values() if not f.kw_only]
    kw_fields = [f for f in fields_by_name.values() if f.kw_only]
    return (*pos_fields, *kw_fields), record_decorator


class _SelfFieldRewriter(ast.NodeTransformer):
    """Rewrite `self.<init_field>` to `<init_field>` inside a record's predicate expression."""

    def __init__(self, init_field_names: frozenset[str]) -> None:
        self._init_field_names = init_field_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        match node:
            case ast.Attribute(value=ast.Name(id="self"), attr=attr) if (
                attr in self._init_field_names
            ):
                return ast.copy_location(ast.Name(id=attr, ctx=ast.Load()), node)
        return self.generic_visit(node)


def _rewrite_predicate_to_params(
    predicate_expr: ast.expr,
    init_field_names: frozenset[str],
    record_qualname: str,
    member_name: str,
) -> ast.expr:
    rewritten = _SelfFieldRewriter(init_field_names).visit(copy.deepcopy(predicate_expr))
    ast.fix_missing_locations(rewritten)
    for node in ast.walk(rewritten):
        if isinstance(node, ast.Name) and node.id == "self":
            raise NeedsInput(
                f"`{record_qualname}.{member_name}` reads non-init attribute or method off `self`",
                NeededInput.PROOF,
            )
    return rewritten


def _bind_ctor_args(
    call: ast.Call,
    fields: Sequence[_DataclassFieldSpec],
    file_path: str,
) -> dict[str, ast.expr | None]:
    """Bind call arguments (or field defaults) to each `__init__` field name."""

    if any(isinstance(arg, ast.Starred) for arg in call.args) or CallKeywords.of(
        call.keywords
    ).splats:
        raise NeedsInput(
            f"constructor call at {file_path}:{call.lineno} uses `*args` or `**kwargs`; cannot classify statically or forward safely",
            NeededInput.PROOF,
        )
    pos_fields = [f for f in fields if not f.kw_only]
    bound: dict[str, ast.expr | None] = {}
    for idx, arg in enumerate(call.args):
        if idx >= len(pos_fields):
            raise NeedsInput(
                f"constructor call at {file_path}:{call.lineno} passes too many positional arguments",
                NeededInput.PROOF,
            )
        bound[pos_fields[idx].name] = arg
    for kw in CallKeywords.of(call.keywords).named:
        bound[kw.name] = kw.value
    for spec in fields:
        if spec.name not in bound:
            bound[spec.name] = spec.default.bind_missing(
                spec.name, file_path, call.lineno
            )
    return bound


class _StaticValue:
    """Marker wrapper separating a known Python constant from unknown expressions."""

    def __init__(self, value: object) -> None:
        self.value = value


def _static_literal(expr: ast.expr) -> _StaticValue | None:
    match expr:
        case ast.Constant(value=val):
            return _StaticValue(val)
        case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() | float() as val)):
            return _StaticValue(-val)
        case ast.Tuple(elts=elts):
            items = [_static_literal(elt) for elt in elts]
            if all(item is not None for item in items):
                return _StaticValue(tuple(item.value for item in items if item is not None))
        case ast.List(elts=elts):
            items = [_static_literal(elt) for elt in elts]
            if all(item is not None for item in items):
                return _StaticValue([item.value for item in items if item is not None])
    return None


def _is_expr_provably_non_none(
    expr: ast.expr,
    file_path: str,
    inferred_types: Mapping[ReceiverSpan, ReceiverType],
) -> bool:
    if _is_none_const(expr):
        return False
    if isinstance(
        expr,
        (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.JoinedStr),
    ):
        return True
    answer = inferred_types.get(ReceiverSpan.of(file_path, expr))
    return answer is not None and answer.is_provably_non_none()


def _eval_predicate_statically(
    expr: ast.expr,
    bound: Mapping[str, ast.expr | None],
    file_path: str,
    inferred_types: Mapping[ReceiverSpan, ReceiverType],
) -> bool | None:
    """Return `True`/`False` when `expr` is statically determined by `bound`, else `None`."""

    match expr:
        case ast.Constant(value=val):
            return bool(val)
        case ast.Name(id=name) if name in bound:
            arg = bound[name]
            if arg is None:
                return None
            lit = _static_literal(arg)
            return None if lit is None else bool(lit.value)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            inner = _eval_predicate_statically(operand, bound, file_path, inferred_types)
            return None if inner is None else (not inner)
        case ast.BoolOp(op=ast.And(), values=values):
            all_true = True
            for val in values:
                res = _eval_predicate_statically(val, bound, file_path, inferred_types)
                if res is False:
                    return False
                if res is None:
                    all_true = False
            return True if all_true else None
        case ast.BoolOp(op=ast.Or(), values=values):
            all_false = True
            for val in values:
                res = _eval_predicate_statically(val, bound, file_path, inferred_types)
                if res is True:
                    return True
                if res is None:
                    all_false = False
            return False if all_false else None
        case ast.Compare(
            left=ast.Name(id=name),
            ops=[ast.Is() | ast.IsNot() as op],
            comparators=[ast.Constant(value=None)],
        ) if name in bound:
            arg = bound[name]
            if arg is None:
                return None
            is_is = isinstance(op, ast.Is)
            if _is_none_const(arg):
                return is_is
            if _is_expr_provably_non_none(arg, file_path, inferred_types):
                return not is_is
            return None
        case ast.Compare(
            left=ast.Name(id=name),
            ops=[ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot() as op],
            comparators=[ast.Constant(value=const)],
        ) if name in bound:
            arg = bound[name]
            if arg is None:
                return None
            lit = _static_literal(arg)
            if lit is not None:
                if isinstance(op, ast.Eq):
                    return bool(lit.value == const)
                if isinstance(op, ast.NotEq):
                    return bool(lit.value != const)
                if isinstance(op, ast.Is):
                    return bool(lit.value is const)
                return bool(lit.value is not const)
            return None
        case ast.Call(func=ast.Name(id="bool"), args=[arg], keywords=[]):
            return _eval_predicate_statically(arg, bound, file_path, inferred_types)
    return None


def _collect_none_check_param_names(expr: ast.expr) -> frozenset[str]:
    names: set[str] = set()
    for node in ast.walk(expr):
        match node:
            case ast.Compare(
                left=ast.Name(id=name),
                ops=[ast.Is() | ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ):
                names.add(name)
    return frozenset(names)


class _CtorRewrite(ABC):
    """How one closed construction site of `Record` is rewritten."""

    @abstractmethod
    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        subtype_name: str,
        factory_name: str,
    ) -> tuple[str, SourceTextSpanReplacement] | None:
        raise NotImplementedError

    @abstractmethod
    def needs_factory(self) -> bool:
        return False

    def needed_cross_module_subtype_file(self, record_file_path: str) -> str | None:
        del record_file_path
        return None


@dataclass(frozen=True)
class _KeepBaseCtor(_CtorRewrite):
    """Statically `False` construction: stays `Record(...)` / `cls(...)`."""

    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        subtype_name: str,
        factory_name: str,
    ) -> tuple[str, SourceTextSpanReplacement] | None:
        del snapshot, subtype_name, factory_name
        return None

    def needs_factory(self) -> bool:
        return False


@dataclass(frozen=True)
class _RewriteToSubtypeCtor(_CtorRewrite):
    """Statically `True` construction: rewrites callee to `<TrueSubtype>`."""

    file_path: str
    callee: ast.expr

    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        subtype_name: str,
        factory_name: str,
    ) -> tuple[str, SourceTextSpanReplacement] | None:
        del factory_name
        geom = SourceTextGeometry(snapshot.sources_by_file_path[self.file_path])
        start, end = geom.required_node_offsets(self.callee)
        return (
            self.file_path,
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=subtype_name,
            ),
        )

    def needs_factory(self) -> bool:
        return False

    def needed_cross_module_subtype_file(self, record_file_path: str) -> str | None:
        return None if self.file_path == record_file_path else self.file_path


@dataclass(frozen=True)
class _RewriteToFactoryCtor(_CtorRewrite):
    """Dynamic construction: rewrites callee to `<callee>.<factory_name>`."""

    file_path: str
    callee: ast.expr

    def replacement(
        self,
        snapshot: CodemodSourceSnapshot,
        subtype_name: str,
        factory_name: str,
    ) -> tuple[str, SourceTextSpanReplacement] | None:
        del subtype_name
        source = snapshot.sources_by_file_path[self.file_path]
        geom = SourceTextGeometry(source)
        start, end = geom.required_node_offsets(self.callee)
        return (
            self.file_path,
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=f"{source[start:end]}.{factory_name}",
            ),
        )

    def needs_factory(self) -> bool:
        return True


def _is_in_annotation_or_type_check(node: ast.AST, parents: AstParentIndex) -> bool:
    """True when `node` sits inside a type annotation or the 2nd arg of `isinstance`/`issubclass`."""

    child: ast.AST = node
    parent = parents.get(child)
    while parent is not None:
        match parent:
            case ast.AnnAssign(annotation=ann) if ann is child:
                return True
            case ast.arg(annotation=ann) if ann is child:
                return True
            case (
                ast.FunctionDef(returns=ret) | ast.AsyncFunctionDef(returns=ret)
            ) if ret is child:
                return True
            case ast.TypeAlias(value=val) if val is child:
                return True
            case ast.Call(
                func=ast.Name(id="isinstance" | "issubclass"),
                args=[_, second],
            ) if second is child:
                return True
        child = parent
        parent = parents.get(child)
    return False


def _replace_call_mutates_predicate_fields(
    call: ast.Call,
    predicate_fields: frozenset[str],
) -> bool:
    """True when `dataclasses.replace(obj, ...)` has `**kwargs` splats or touches any field in `predicate_fields`."""

    kw = CallKeywords.of(call.keywords)
    if kw.splats:
        return True
    return any(named.name in predicate_fields for named in kw.named)


def _collect_closed_constructions(
    snapshot: CodemodSourceSnapshot,
    record_target: ResolvedClassTarget,
    record_symbol: str,
    fields: Sequence[_DataclassFieldSpec],
    rewritten_pred: ast.expr,
) -> tuple[_CtorRewrite, ...]:
    """Prove every reference to `record_symbol` is closed and classify every construction site."""

    record_module = snapshot.parsed_module_for_source_path(record_target.file_path)
    if _module_exports_name(record_module.module, record_target.qualname):
        raise NeedsInput(
            f"`{record_target.qualname}` is listed in `__all__`; construction sites are not closed",
            NeededInput.PROOF,
        )

    init_field_names = frozenset(spec.name for spec in fields)
    predicate_fields = frozenset(
        node.id
        for node in ast.walk(rewritten_pred)
        if isinstance(node, ast.Name) and node.id in init_field_names
    )

    bindings = snapshot.module_binding_proof
    raw_ctors: list[tuple[str, ast.Call, dict[str, ast.expr | None]]] = []

    # Check classmethod `cls(...)` calls inside `record_target.node` and `dataclasses.replace` on `self`.
    record_bindings = ModuleNominalBindingAuthority(record_module)
    for stmt in record_target.node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_classmethod = any(
            (isinstance(dec, ast.Name) and dec.id == "classmethod")
            or (isinstance(dec, ast.Attribute) and dec.attr == "classmethod")
            for dec in stmt.decorator_list
        )
        positional = (*stmt.args.posonlyargs, *stmt.args.args)
        first_name = positional[0].arg if positional else None
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            # Refuse `dataclasses.replace(self, ...)` when it mutates a predicate field.
            callee_parts = ExpressionReference.decoded(node.func).attribute_chain
            if (
                callee_parts is not None
                and record_bindings.snapshot_before(node.lineno)
                .reference_for(callee_parts)
                .has_bound_name("dataclasses.replace")
                and _replace_call_mutates_predicate_fields(node, predicate_fields)
            ):
                raise NeedsInput(
                    f"{record_target.file_path}:{node.lineno} calls `dataclasses.replace` on predicate field(s) inside `{record_target.qualname}`",
                    NeededInput.PROOF,
                )
            if (
                is_classmethod
                and first_name is not None
                and isinstance(node.func, ast.Name)
                and node.func.id == first_name
            ):
                bound = _bind_ctor_args(node, fields, record_target.file_path)
                raw_ctors.append((record_target.file_path, node, bound))

    record_short_name = record_target.qualname.rsplit(".", 1)[-1]
    pending_replace_checks: list[tuple[str, ast.Call, ReceiverSpan]] = []
    for mod in snapshot.parsed_modules:
        has_record_name = record_short_name in mod.source
        has_replace_name = "replace" in mod.source
        if not has_record_name and not has_replace_name:
            continue
        node_index = mod.fact(ModuleNodeIndex)

        if has_replace_name:
            mod_bindings: ModuleNominalBindingAuthority | None = None
            for call in node_index.calls:
                if not call.args or not _replace_call_mutates_predicate_fields(
                    call, predicate_fields
                ):
                    continue
                callee_parts = ExpressionReference.decoded(call.func).attribute_chain
                if callee_parts is None:
                    continue
                if mod_bindings is None:
                    mod_bindings = ModuleNominalBindingAuthority(mod)
                if (
                    mod_bindings.snapshot_before(call.lineno)
                    .reference_for(callee_parts)
                    .has_bound_name("dataclasses.replace")
                ):
                    pending_replace_checks.append(
                        (mod.file_path, call, ReceiverSpan.of(mod.file_path, call.args[0]))
                    )

        if not has_record_name:
            continue
        candidates: tuple[ast.Name | ast.Attribute, ...] = (
            *(
                name_node
                for name_node in node_index.names_by_id.get(record_short_name, ())
                if isinstance(name_node.ctx, ast.Load)
            ),
            *node_index.attributes_by_attr.get(record_short_name, ()),
        )
        for node in candidates:
            if _is_in_annotation_or_type_check(node, node_index.parents):
                continue
            witness = bindings.reference_or_builtin_witness_at(
                mod, node, line=node.lineno
            )
            if witness is None or witness.qualified_name != record_symbol:
                continue
            parent = node_index.parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                bound = _bind_ctor_args(parent, fields, mod.file_path)
                raw_ctors.append((mod.file_path, parent, bound))
                continue
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if parent.attr == "__new__":
                    raise NeedsInput(
                        f"{mod.file_path}:{node.lineno} accesses `{record_target.qualname}.__new__`",
                        NeededInput.PROOF,
                    )
                continue
            raise NeedsInput(
                f"{mod.file_path}:{node.lineno} references `{record_target.qualname}` as a value; construction sites are not closed",
                NeededInput.PROOF,
            )

    if pending_replace_checks:
        replace_types = snapshot.type_checker.receiver_types(
            tuple(dict.fromkeys(span for _, _, span in pending_replace_checks))
        )
        for file_path, call, span in pending_replace_checks:
            ans = replace_types.get(span)
            if ans is not None and ans.reaches_method_owner(
                frozenset((record_symbol,)),
                file_path=file_path,
                line=call.lineno,
                method_name="__replace__",
            ):
                raise NeedsInput(
                    f"{file_path}:{call.lineno} calls `dataclasses.replace` on predicate field(s) of `{record_target.qualname}`",
                    NeededInput.PROOF,
                )

    if not raw_ctors:
        raise NeedsInput(
            f"`{record_target.qualname}` has no scanned construction sites; cannot prove construction is closed",
            NeededInput.PROOF,
        )

    # Batch type-checker queries for any parameter compared with None in the predicate.
    none_params = _collect_none_check_param_names(rewritten_pred)
    pending_spans: list[ReceiverSpan] = []
    for file_path, _call, bound in raw_ctors:
        for param_name in none_params:
            arg = bound.get(param_name)
            if arg is None or _is_none_const(arg) or isinstance(
                arg,
                (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.JoinedStr),
            ):
                continue
            pending_spans.append(ReceiverSpan.of(file_path, arg))
    inferred_types = (
        snapshot.type_checker.receiver_types(tuple(dict.fromkeys(pending_spans)))
        if pending_spans
        else {}
    )

    rewrites: list[_CtorRewrite] = []
    for file_path, call, bound in raw_ctors:
        static_res = _eval_predicate_statically(
            rewritten_pred, bound, file_path, inferred_types
        )
        if static_res is False:
            rewrites.append(_KeepBaseCtor())
        elif static_res is True:
            rewrites.append(_RewriteToSubtypeCtor(file_path=file_path, callee=call.func))
        else:
            rewrites.append(_RewriteToFactoryCtor(file_path=file_path, callee=call.func))
    return tuple(rewrites)


class _AncestorMemberDecl(ABC):
    """Whether an ancestor of `Record` declares a constant-`False` default for `member_name`."""

    @abstractmethod
    def owner_symbol(self, record_symbol: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def owner_target(self, record_target: ResolvedClassTarget) -> ResolvedClassTarget:
        raise NotImplementedError

    def symbols_to_check(self, record_symbol: str) -> tuple[str, ...]:
        return (record_symbol,)

    def apply_delete(
        self,
        snapshot: CodemodSourceSnapshot,
        ancestor_methods: tuple[tuple[str, str], ...],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[str, str], ...]:
        del snapshot, geometry, add_span, dropped_by_file
        return ancestor_methods

    def apply_constantize_on_record(
        self,
        snapshot: CodemodSourceSnapshot,
        record_target: ResolvedClassTarget,
        member_stmt: ast.FunctionDef,
        record_methods: tuple[tuple[str, str], ...],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[str, str], ...]:
        del snapshot, dropped_by_file
        geom = geometry(record_target.file_path)
        body_stmts = statements_after_docstring(member_stmt.body)
        first_start, _ = geom.required_node_offsets(body_stmts[0])
        _, last_end = geom.required_node_offsets(body_stmts[-1])
        add_span(record_target.file_path, first_start, last_end, "return False")
        return record_methods


@dataclass(frozen=True)
class _NoAncestorMemberDecl(_AncestorMemberDecl):
    """`Record` itself is the topmost declarer of `member_name`."""

    def owner_symbol(self, record_symbol: str) -> str:
        return record_symbol

    def owner_target(self, record_target: ResolvedClassTarget) -> ResolvedClassTarget:
        return record_target


@dataclass(frozen=True)
class _ConstantFalseAncestorDecl(_AncestorMemberDecl):
    """An ancestor of `Record` declares `member_name` returning constant `False`."""

    symbol: str
    target: ResolvedClassTarget
    stmt: ast.FunctionDef

    def owner_symbol(self, record_symbol: str) -> str:
        del record_symbol
        return self.symbol

    def owner_target(self, record_target: ResolvedClassTarget) -> ResolvedClassTarget:
        del record_target
        return self.target

    def symbols_to_check(self, record_symbol: str) -> tuple[str, ...]:
        return (record_symbol, self.symbol)

    def apply_delete(
        self,
        snapshot: CodemodSourceSnapshot,
        ancestor_methods: tuple[tuple[str, str], ...],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[str, str], ...]:
        if ancestor_methods:
            a_ls, a_end, a_indent = _statement_full_line_span(
                geometry(self.target.file_path), self.stmt
            )
            add_span(
                self.target.file_path,
                a_ls,
                a_end,
                _indent_methods([r for _n, r in ancestor_methods], a_indent),
            )
            dropped_by_file.setdefault(self.target.file_path, []).append(self.stmt)
            return ()
        deletion = StatementDeletionSource(
            source=snapshot.sources_by_file_path[self.target.file_path],
            node=self.target.node,
            file_path=self.target.file_path,
        )
        for span_repl in deletion.replacements_for_statements((self.stmt,)):
            add_span(
                self.target.file_path,
                span_repl.start_offset,
                span_repl.end_offset,
                span_repl.replacement_source,
            )
        dropped_by_file.setdefault(self.target.file_path, []).append(self.stmt)
        return ()

    def apply_constantize_on_record(
        self,
        snapshot: CodemodSourceSnapshot,
        record_target: ResolvedClassTarget,
        member_stmt: ast.FunctionDef,
        record_methods: tuple[tuple[str, str], ...],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[str, str], ...]:
        if record_methods:
            line_start, end, indent = _statement_full_line_span(
                geometry(record_target.file_path), member_stmt
            )
            add_span(
                record_target.file_path,
                line_start,
                end,
                _indent_methods([r for _n, r in record_methods], indent),
            )
            dropped_by_file.setdefault(record_target.file_path, []).append(member_stmt)
            return ()
        deletion = StatementDeletionSource(
            source=snapshot.sources_by_file_path[record_target.file_path],
            node=record_target.node,
            file_path=record_target.file_path,
        )
        for span_repl in deletion.replacements_for_statements((member_stmt,)):
            add_span(
                record_target.file_path,
                span_repl.start_offset,
                span_repl.end_offset,
                span_repl.replacement_source,
            )
        dropped_by_file.setdefault(record_target.file_path, []).append(member_stmt)
        return ()


class _MemberDisposition(ABC):
    """How the original `-> bool` predicate declaration is updated after pushing its branch consumers."""

    @abstractmethod
    def apply_with_appended(
        self,
        snapshot: CodemodSourceSnapshot,
        record_target: ResolvedClassTarget,
        member_stmt: ast.FunctionDef,
        ancestor_decl: _AncestorMemberDecl,
        record_methods_to_append: Sequence[tuple[str, str]],
        ancestor_methods_to_append: Sequence[tuple[str, str]],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        """Apply member deletion/constantization, returning remaining `(record_methods, ancestor_methods)` to append."""
        raise NotImplementedError

    @abstractmethod
    def subtype_override_method(self, member_name: str, is_property: bool) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def summary_verb(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class _DeleteOwnStateMember(_MemberDisposition):
    """Zero remaining reads: delete `member_name` from `Record` (and constant-False ancestor if any)."""

    def apply_with_appended(
        self,
        snapshot: CodemodSourceSnapshot,
        record_target: ResolvedClassTarget,
        member_stmt: ast.FunctionDef,
        ancestor_decl: _AncestorMemberDecl,
        record_methods_to_append: Sequence[tuple[str, str]],
        ancestor_methods_to_append: Sequence[tuple[str, str]],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        rem_record = tuple(record_methods_to_append)
        if rem_record:
            line_start, end, indent = _statement_full_line_span(
                geometry(record_target.file_path), member_stmt
            )
            add_span(
                record_target.file_path,
                line_start,
                end,
                _indent_methods([r for _n, r in rem_record], indent),
            )
            dropped_by_file.setdefault(record_target.file_path, []).append(member_stmt)
            rem_record = ()
        else:
            deletion = StatementDeletionSource(
                source=snapshot.sources_by_file_path[record_target.file_path],
                node=record_target.node,
                file_path=record_target.file_path,
            )
            for span_repl in deletion.replacements_for_statements((member_stmt,)):
                add_span(
                    record_target.file_path,
                    span_repl.start_offset,
                    span_repl.end_offset,
                    span_repl.replacement_source,
                )
            dropped_by_file.setdefault(record_target.file_path, []).append(member_stmt)

        rem_anc = ancestor_decl.apply_delete(
            snapshot,
            tuple(ancestor_methods_to_append),
            geometry,
            add_span,
            dropped_by_file,
        )
        return rem_record, rem_anc

    def subtype_override_method(self, member_name: str, is_property: bool) -> str | None:
        del member_name, is_property
        return None

    def summary_verb(self) -> str:
        return "deletes"


@dataclass(frozen=True)
class _ConstantizeOwnStateMember(_MemberDisposition):
    """Non-branching data reads remain: replace probing body with constant `False` / `True`."""

    def apply_with_appended(
        self,
        snapshot: CodemodSourceSnapshot,
        record_target: ResolvedClassTarget,
        member_stmt: ast.FunctionDef,
        ancestor_decl: _AncestorMemberDecl,
        record_methods_to_append: Sequence[tuple[str, str]],
        ancestor_methods_to_append: Sequence[tuple[str, str]],
        geometry,
        add_span,
        dropped_by_file: dict[str, list[ast.AST]],
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        rem_record = ancestor_decl.apply_constantize_on_record(
            snapshot,
            record_target,
            member_stmt,
            tuple(record_methods_to_append),
            geometry,
            add_span,
            dropped_by_file,
        )
        return rem_record, tuple(ancestor_methods_to_append)

    def subtype_override_method(self, member_name: str, is_property: bool) -> str | None:
        if is_property:
            return f"@property\ndef {member_name}(self) -> bool:\n    return True"
        return f"def {member_name}(self) -> bool:\n    return True"

    def summary_verb(self) -> str:
        return "constantizes"


@dataclass(frozen=True)
class _OwnStateConsumerProof:
    """One proved consumer branch pushed onto `base_owner` (`False` arm) and `TrueSubtype` (`True` arm)."""

    source_path: str
    test_line: int
    function: FunctionNode
    enclosure: FunctionEnclosure
    branch_site: AnswerBranchSite
    in_place_base_method: bool
    false_body: tuple[ast.stmt, ...]
    true_body: tuple[ast.stmt, ...]
    captured_args: tuple[CapturedArgument, ...]
    is_async: bool
    method_name: str
    docstring: str | None
    outer_receiver_aliases: tuple[OuterReceiverAlias, ...]

    def consumed_attribute_ids(self, member_name: str) -> frozenset[int]:
        return frozenset(
            id(node)
            for test_node in self.branch_site.dispatch.test_nodes
            for node in ast.walk(test_node)
            if isinstance(node, ast.Attribute) and node.attr == member_name
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
            rendered = site.render_in_place_default(self.false_body, line_indent)
            assert rendered is not None
            return start, end, rendered
        call_args = ", ".join(arg.call_expr for arg in self.captured_args)
        subject_text = ast.unparse(site.subject)
        match site.subject:
            case ast.Name() | ast.Attribute() | ast.Call() | ast.Subscript():
                pass
            case _:
                subject_text = f"({subject_text})"
        call = f"{subject_text}.{self.method_name}({call_args})"
        if self.is_async:
            call = f"await {call}"
        return start, end, site.render_call_replacement(call, line_indent)

    def render_methods_and_imports(
        self,
        snapshot: CodemodSourceSnapshot,
        base_owner_target: ResolvedClassTarget,
        record_target: ResolvedClassTarget,
        rationale: str,
    ) -> tuple[str | None, str, tuple[NominalSourceEdit, ...]]:
        """Return `(base_method_to_append_or_none, subtype_method, cross_module_import_edits)`."""

        source_geom = SourceTextGeometry(snapshot.sources_by_file_path[self.source_path])
        parsed_source = snapshot.parsed_module_for_source_path(self.source_path)
        if self.in_place_base_method:
            header_text = _method_header_text(self.function, source_geom)
            subtype_rendered = _render_in_place_override(header_text, self.true_body)
            return None, subtype_rendered, ()

        site = self.branch_site.dispatch
        param_names = tuple(arg.param_name for arg in self.captured_args)
        destinations = (
            (base_owner_target,)
            if base_owner_target.file_path == record_target.file_path
            else (base_owner_target, record_target)
        )
        resolved_anns = ResolvedMethodAnnotations.for_push(
            snapshot,
            parsed_source,
            self.test_line,
            self.enclosure,
            self.function,
            self.outer_receiver_aliases,
            self.captured_args,
            destinations,
            propagates_function_return_type=site.propagates_function_return_type,
            is_void_method=site.is_void_method,
            constant_bool_return=site.returns_constant_bool,
            rationale=rationale,
        )
        base_method = _render_method(
            site.kind,
            self.method_name,
            param_names,
            self.false_body,
            is_async=self.is_async,
            docstring=self.docstring,
            param_annotations=resolved_anns.param_map,
            return_annotation=resolved_anns.return_annotation,
        )
        subtype_method = _render_method(
            site.kind,
            self.method_name,
            param_names,
            self.true_body,
            is_async=self.is_async,
            param_annotations=resolved_anns.param_map,
            return_annotation=resolved_anns.return_annotation,
        )
        local_exclusions = frozenset((site.kind.receiver, *param_names))
        imports: list[NominalSourceEdit] = [
            *resolved_anns.import_edits,
            *_cross_module_imports_for_body(
                snapshot,
                parsed_source,
                self.test_line,
                base_owner_target,
                self.false_body,
                local_exclusions,
                rationale,
            ),
            *_cross_module_imports_for_body(
                snapshot,
                parsed_source,
                self.test_line,
                record_target,
                self.true_body,
                local_exclusions,
                rationale,
            ),
        ]
        return base_method, subtype_method, tuple(imports)


def _resolve_constant_false_ancestor(
    snapshot: CodemodSourceSnapshot,
    record_symbol: str,
    member_name: str,
) -> _AncestorMemberDecl:
    """Resolve whether an ancestor of `record_symbol` declares `member_name` as constant `False`."""

    index = snapshot.overlap_class_index
    classes = index.classes_by_symbol
    declaring_ancestors: list[str] = []
    for anc in index.ancestor_symbols(record_symbol):
        indexed = classes.get(anc)
        if indexed is None:
            continue
        if member_name in indexed.member_binding_names:
            declaring_ancestors.append(anc)
    if not declaring_ancestors:
        return _NoAncestorMemberDecl()
    if len(declaring_ancestors) > 1:
        raise NeedsInput(
            f"multiple ancestors of `{record_symbol}` declare `{member_name}`",
            NeededInput.PROOF,
        )
    anc_sym = declaring_ancestors[0]
    anc_indexed = classes[anc_sym]
    anc_target = ResolvedClassTarget.from_rewrite_target(
        snapshot,
        SourceRewriteTarget(
            file_path=anc_indexed.file_path, qualname=anc_indexed.qualname
        ),
    )
    anc_stmt: ast.FunctionDef | None = None
    for stmt in anc_target.node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == member_name:
            anc_stmt = stmt
            break
    if anc_stmt is None:
        raise NeedsInput(
            f"ancestor `{anc_target.qualname}` declares `{member_name}` as non-method; cannot push onto it",
            NeededInput.PROOF,
        )
    match statements_after_docstring(anc_stmt.body):
        case [ast.Return(value=ast.Constant(value=False))]:
            pass
        case _:
            raise NeedsInput(
                f"ancestor `{anc_target.qualname}.{member_name}` is not a constant `False` default",
                NeededInput.PROOF,
            )
    # Ensure no other descendant of `anc_sym` (other than `record_symbol`) overrides `member_name`.
    for desc in index.descendant_symbols(anc_sym):
        if desc == record_symbol:
            continue
        desc_indexed = classes.get(desc)
        if desc_indexed is not None and member_name in desc_indexed.member_binding_names:
            raise NeedsInput(
                f"sibling `{desc_indexed.qualname}` also overrides `{member_name}`",
                NeededInput.PROOF,
            )
    return _ConstantFalseAncestorDecl(anc_sym, anc_target, anc_stmt)


def _prove_consumer(
    snapshot: CodemodSourceSnapshot,
    consumer: ConstantAnswerPushConsumer,
    member_name: str,
    base_owner_symbol: str,
    base_owner_target: ResolvedClassTarget,
    record_symbol: str,
    record_target: ResolvedClassTarget,
) -> _OwnStateConsumerProof:
    method_name = consumer.method_name
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
        snapshot, consumer.target, test_line=consumer.test_line
    )
    raw_site = find_answer_dispatch_site(
        function, consumer.test_line, member_name, InstanceDispatchKind()
    )
    if raw_site is None:
        raise NeedsInput(
            f"branch on `{member_name}` at {source_path}:{consumer.test_line} is not a pushable dispatch shape",
            NeededInput.SOURCE_CHANGE,
        )
    site = raw_site.dispatch
    if not _subject_allowed_for_site(site):
        raise NeedsInput(
            f"branch subject `{ast.unparse(site.subject)}` at {source_path}:{consumer.test_line} is not safely rewritable",
            NeededInput.SOURCE_CHANGE,
        )
    positional = (*function.args.posonlyargs, *function.args.args)
    first_param = positional[0].arg if positional else None
    enclosure_symbol = enclosure.class_receiver_symbol(snapshot)
    in_place = (
        enclosure_symbol == record_symbol
        and base_owner_symbol == record_symbol
        and isinstance(site.subject, ast.Name)
        and site.subject.id == first_param
        and site.is_whole_function
        and method_name == function.name
    )
    if not in_place and site.checks_local_store_escapes:
        replaced_nodes = site.replaced_ast_nodes()
        if replaced_nodes:
            stored = _names_stored_in(
                tuple(
                    stmt
                    for body in (*(arm.body for arm in site.arms), site.default_body)
                    for stmt in body
                )
            )
            used_outside = _names_referenced_outside(function, replaced_nodes)
            escaping = sorted(stored & used_outside)
            if escaping:
                upgraded = _upgrade_dispatch_site_to_assigning(
                    function, site, escaping
                )
                if upgraded is not None:
                    site = upgraded
                    raw_site = AnswerBranchSite(site, raw_site.arm_values)

    if not in_place:
        for target_cls in (base_owner_target, record_target):
            if method_name in target_cls.bound_names:
                raise NeedsInput(
                    f"`{target_cls.qualname}` already binds `{method_name}`",
                    NeededInput.OWNER_PICK,
                )

    body_by_value: dict[object, tuple[ast.stmt, ...]] = {}
    for value, arm in zip(raw_site.arm_values, site.arms, strict=True):
        body_by_value[value] = arm.body
    raw_true = body_by_value.get(True, site.default_body)
    raw_false = body_by_value.get(False, site.default_body)
    if ast.dump(ast.Module(body=list(raw_true), type_ignores=[])) == ast.dump(
        ast.Module(body=list(raw_false), type_ignores=[])
    ):
        raise NeedsInput(
            f"both arms of `{member_name}` at {source_path}:{consumer.test_line} are identical",
            NeededInput.PROOF,
        )

    outer_receiver_aliases: tuple[OuterReceiverAlias, ...] = ()
    if in_place:
        if _contains_super_call(raw_false) or _contains_super_call(raw_true):
            raise NeedsInput(
                f"`{function.name}` calls `super()`, which cannot move onto subtypes automatically",
                NeededInput.SOURCE_CHANGE,
            )
        false_body = raw_false
        if site.render_in_place_default(false_body) is None:
            raise NeedsInput(
                f"`{function.name}` branch shape at {source_path}:{consumer.test_line} cannot render in place",
                NeededInput.SOURCE_CHANGE,
            )
        true_body = raw_true
        captured_args: tuple[CapturedArgument, ...] = ()
        is_async = isinstance(function, ast.AsyncFunctionDef)
    else:
        receiver = site.kind.receiver
        enclosing_locals = enclosure.enclosing_local_names()
        outer_receiver_aliases = _outer_receiver_aliases(
            function,
            site,
            (raw_false, raw_true),
            enclosing_locals=enclosing_locals,
        )
        false_body = _rewrite_subject_to_receiver(
            raw_false,
            site.subject,
            receiver,
            outer_receiver_aliases=outer_receiver_aliases,
        )
        true_body = _rewrite_subject_to_receiver(
            raw_true,
            site.subject,
            receiver,
            outer_receiver_aliases=outer_receiver_aliases,
        )
        captured_args, is_async = _analyze_arm_bodies(
            function,
            site,
            (false_body, true_body),
            outer_receiver_aliases=outer_receiver_aliases,
            enclosing_locals=enclosing_locals,
        )
        local_exclusions = frozenset(
            (receiver, *(arg.param_name for arg in captured_args))
        )
        parsed_source = snapshot.parsed_module_for_source_path(source_path)
        rationale_probe = f"Push `{member_name}` branch onto `{base_owner_target.qualname}.{method_name}`."
        _cross_module_imports_for_body(
            snapshot,
            parsed_source,
            consumer.test_line,
            base_owner_target,
            false_body,
            local_exclusions,
            rationale_probe,
        )
        _cross_module_imports_for_body(
            snapshot,
            parsed_source,
            consumer.test_line,
            record_target,
            true_body,
            local_exclusions,
            rationale_probe,
        )

    return _OwnStateConsumerProof(
        source_path=source_path,
        test_line=consumer.test_line,
        function=function,
        enclosure=enclosure,
        branch_site=raw_site,
        in_place_base_method=in_place,
        false_body=false_body,
        true_body=true_body,
        captured_args=captured_args,
        is_async=is_async,
        method_name=method_name,
        docstring=consumer.docstring.strip() if consumer.docstring else None,
        outer_receiver_aliases=outer_receiver_aliases,
    )


@dataclass(frozen=True)
class OwnStateBranchAnswerProof:
    """Complete proof to split a frozen dataclass's derived bool into a subtype."""

    record_symbol: str
    record_target: ResolvedClassTarget
    record_decorator: ast.Call
    fields: tuple[_DataclassFieldSpec, ...]
    member_name: str
    member_stmt: ast.FunctionDef
    is_property: bool
    rewritten_predicate: ast.expr
    base_owner_symbol: str
    base_owner_target: ResolvedClassTarget
    ancestor_decl: _AncestorMemberDecl
    subtype_name: str
    subtype_docstring: str
    factory_name: str
    ctor_rewrites: tuple[_CtorRewrite, ...]
    consumers: tuple[_OwnStateConsumerProof, ...]
    disposition: _MemberDisposition

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        member_name: str,
        subtype_name: str,
        subtype_docstring: str,
        consumers: Sequence[ConstantAnswerPushConsumer],
        *,
        factory_name: str = "classified",
        delete_if_unread: bool = True,
    ) -> OwnStateBranchAnswerProof:
        for label, ident in (
            ("subtype_name", subtype_name),
            ("factory_name", factory_name),
        ):
            if (
                not ident.isidentifier()
                or keyword_module.iskeyword(ident)
                or (ident.startswith("__") and ident.endswith("__"))
            ):
                raise NeedsInput(
                    f"`{ident}` is not a valid non-dunder {label}",
                    NeededInput.OWNER_PICK,
                )
        if not consumers:
            raise NeedsInput(
                f"no branch consumers supplied for `{member_name}`",
                NeededInput.PROOF,
            )

        record_target = ResolvedClassTarget.from_rewrite_target(snapshot, target)
        record_symbol = record_target.required_symbol(snapshot)
        index = snapshot.overlap_class_index
        if index.descendant_symbols(record_symbol):
            raise NeedsInput(
                f"`{record_target.qualname}` already has subclasses; cannot split directly into one subtype",
                NeededInput.SOURCE_CHANGE,
            )

        record_module = snapshot.parsed_module_for_source_path(record_target.file_path)
        existing_top = _module_top_level_bound_names(record_module.module)
        if subtype_name in existing_top:
            raise NeedsInput(
                f"`{subtype_name}` is already bound in {record_target.file_path}",
                NeededInput.OWNER_PICK,
            )

        fields, record_decorator = _extract_dataclass_fields(snapshot, record_symbol)
        member_stmt: ast.FunctionDef | None = None
        for stmt in record_target.node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == member_name:
                member_stmt = stmt
                break
        if member_stmt is None or not takes_only_receiver(member_stmt):
            raise NeedsInput(
                f"`{record_target.qualname}` declares no zero-argument `{member_name}` method/property",
                NeededInput.PROOF,
            )
        bindings = ModuleNominalBindingAuthority(record_module).snapshot_before(
            record_target.node.lineno
        )
        body_stmts = statements_after_docstring(member_stmt.body)
        match body_stmts:
            case [ast.Return(value=ast.expr() as pred_expr)] if calls_only_pure_builtins(
                pred_expr, bindings
            ):
                pass
            case _:
                raise NeedsInput(
                    f"`{record_target.qualname}.{member_name}` is not a single pure boolean return expression",
                    NeededInput.PROOF,
                )
        is_property = bindings.decorated_by(member_stmt, PROPERTY_DECORATORS)
        init_field_names = frozenset(spec.name for spec in fields)
        rewritten_pred = _rewrite_predicate_to_params(
            pred_expr, init_field_names, record_target.qualname, member_name
        )

        ancestor_decl = _resolve_constant_false_ancestor(
            snapshot, record_symbol, member_name
        )
        base_owner_symbol = ancestor_decl.owner_symbol(record_symbol)
        base_owner_target = ancestor_decl.owner_target(record_target)

        ctor_rewrites = _collect_closed_constructions(
            snapshot, record_target, record_symbol, fields, rewritten_pred
        )
        needs_factory = any(rw.needs_factory() for rw in ctor_rewrites)
        if needs_factory:
            if factory_name in record_target.bound_names:
                raise NeedsInput(
                    f"`{record_target.qualname}` already binds `{factory_name}`",
                    NeededInput.OWNER_PICK,
                )
            for spec in fields:
                spec.default.require_factory_compatible(
                    record_target.qualname, spec.name, factory_name
                )

        # Verify cycle-free cross-module imports for any statically-True external constructor call.
        import_graph = snapshot.module_import_graph
        for rw in ctor_rewrites:
            ext_file = rw.needed_cross_module_subtype_file(record_target.file_path)
            if ext_file is None:
                continue
            ext_mod = snapshot.parsed_module_for_source_path(ext_file)
            if subtype_name in _module_top_level_bound_names(ext_mod.module):
                raise NeedsInput(
                    f"`{subtype_name}` is already bound in caller {ext_file}",
                    NeededInput.OWNER_PICK,
                )
            try:
                import_graph.required_import_source(
                    importing_file_path=ext_file,
                    imported_file_path=record_target.file_path,
                    imported_name=subtype_name,
                )
            except ValueError as error:
                raise NeedsInput(str(error), NeededInput.SOURCE_CHANGE) from error

        proved_consumers: list[_OwnStateConsumerProof] = []
        seen_names: set[str] = set()
        for consumer in consumers:
            if consumer.method_name in seen_names:
                raise NeedsInput(
                    f"duplicate method name `{consumer.method_name}` for consumers of `{record_target.qualname}.{member_name}`",
                    NeededInput.OWNER_PICK,
                )
            seen_names.add(consumer.method_name)
            proved_consumers.append(
                _prove_consumer(
                    snapshot,
                    consumer,
                    member_name,
                    base_owner_symbol,
                    base_owner_target,
                    record_symbol,
                    record_target,
                )
            )

        consumed_ids = frozenset().union(
            *(c.consumed_attribute_ids(member_name) for c in proved_consumers)
        )
        can_delete = delete_if_unread and _can_delete_own_state_member(
            snapshot,
            ancestor_decl.symbols_to_check(record_symbol),
            member_name,
            consumed_ids,
        )
        disposition: _MemberDisposition = (
            _DeleteOwnStateMember() if can_delete else _ConstantizeOwnStateMember()
        )

        return cls(
            record_symbol=record_symbol,
            record_target=record_target,
            record_decorator=record_decorator,
            fields=fields,
            member_name=member_name,
            member_stmt=member_stmt,
            is_property=is_property,
            rewritten_predicate=rewritten_pred,
            base_owner_symbol=base_owner_symbol,
            base_owner_target=base_owner_target,
            ancestor_decl=ancestor_decl,
            subtype_name=subtype_name,
            subtype_docstring=subtype_docstring,
            factory_name=factory_name,
            ctor_rewrites=ctor_rewrites,
            consumers=tuple(proved_consumers),
            disposition=disposition,
        )

    def rationale(self) -> str:
        methods = ", ".join(f"`{c.method_name}`" for c in self.consumers)
        return (
            f"Splits `{self.record_target.qualname}` via `{self.subtype_name}`, "
            f"pushes `{self.member_name}` branch(es) onto {methods}, "
            f"and {self.disposition.summary_verb()} `{self.member_name}`."
        )

    def _render_factory_method(self, future_ann: bool) -> str:
        pos_parts: list[str] = ["cls"]
        kw_parts: list[str] = []
        fwd_parts: list[str] = []
        for spec in self.fields:
            ann_src = ast.unparse(spec.annotation)
            part = f"{spec.name}: {ann_src}{spec.default.render_default_suffix()}"
            if spec.kw_only:
                kw_parts.append(part)
            else:
                pos_parts.append(part)
            fwd_parts.append(f"{spec.name}={spec.name}")
        if kw_parts:
            sig = ", ".join((*pos_parts, "*", *kw_parts))
        else:
            sig = ", ".join(pos_parts)
        ret_type = (
            self.record_target.qualname
            if future_ann
            else repr(self.record_target.qualname)
        )
        fwd = ", ".join(fwd_parts)
        pred_src = ast.unparse(self.rewritten_predicate)
        return (
            "@classmethod\n"
            f"def {self.factory_name}({sig}) -> {ret_type}:\n"
            f"    return (\n"
            f"        {self.subtype_name}({fwd})\n"
            f"        if {pred_src}\n"
            f"        else cls({fwd})\n"
            f"    )"
        )

    def _render_subtype_class(
        self,
        record_source: str,
        subtype_methods: Sequence[str],
    ) -> str:
        geom = SourceTextGeometry(record_source)
        dec_start, dec_end = geom.required_node_offsets(self.record_decorator)
        decorator_src = f"@{record_source[dec_start:dec_end]}"
        lines = [
            "",
            "",
            decorator_src,
            f"class {self.subtype_name}({self.record_target.qualname}):",
            f'    """{self.subtype_docstring}"""',
        ]
        for rendered in subtype_methods:
            lines.append("")
            for line in rendered.splitlines():
                lines.append(f"    {line}" if line else "")
        return "\n".join(lines) + "\n"

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        geometries: dict[str, SourceTextGeometry] = {}
        replacements_by_file: dict[str, list[SourceTextSpanReplacement]] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        def add_span(file_path: str, start: int, end: int, text: str) -> None:
            replacements_by_file.setdefault(file_path, []).append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start, end_offset=end, replacement_source=text
                )
            )

        extra_edits: list[NominalSourceEdit] = []
        base_methods_to_append: list[tuple[str, str]] = []
        subtype_methods: list[str] = []

        for consumer in self.consumers:
            c_geom = geometry(consumer.source_path)
            start, end, replacement = consumer.consumer_replacement(c_geom)
            add_span(consumer.source_path, start, end, replacement)
            base_rendered, sub_rendered, imports = consumer.render_methods_and_imports(
                snapshot, self.base_owner_target, self.record_target, rationale
            )
            if base_rendered is not None:
                base_methods_to_append.append((consumer.method_name, base_rendered))
            subtype_methods.append(sub_rendered)
            extra_edits.extend(imports)

        const_override = self.disposition.subtype_override_method(
            self.member_name, self.is_property
        )
        if const_override is not None:
            subtype_methods.insert(0, const_override)

        # Rewrite closed construction sites and add cross-module imports for static-True callers.
        import_graph = snapshot.module_import_graph
        imported_subtype_files: set[str] = set()
        for rw in self.ctor_rewrites:
            res = rw.replacement(snapshot, self.subtype_name, self.factory_name)
            if res is not None:
                c_file, span_repl = res
                replacements_by_file.setdefault(c_file, []).append(span_repl)
            ext_file = rw.needed_cross_module_subtype_file(self.record_target.file_path)
            if ext_file is not None and ext_file not in imported_subtype_files:
                imported_subtype_files.add(ext_file)
                extra_edits.append(
                    ModuleImportMutation.from_source(
                        file_path=ext_file,
                        import_source=import_graph.required_import_source(
                            importing_file_path=ext_file,
                            imported_file_path=self.record_target.file_path,
                            imported_name=self.subtype_name,
                        ),
                        rationale=rationale,
                    )
                )

        record_module = snapshot.parsed_module_for_source_path(
            self.record_target.file_path
        )
        record_methods_to_append: list[tuple[str, str]] = []
        if any(rw.needs_factory() for rw in self.ctor_rewrites):
            record_methods_to_append.append(
                (
                    self.factory_name,
                    self._render_factory_method(
                        _has_future_annotations(record_module.module)
                    ),
                )
            )
        if self.base_owner_symbol == self.record_symbol:
            record_methods_to_append.extend(base_methods_to_append)
            ancestor_methods_to_append: list[tuple[str, str]] = []
        else:
            ancestor_methods_to_append = list(base_methods_to_append)

        dropped_by_file: dict[str, list[ast.AST]] = {}
        remaining_record_methods, remaining_ancestor_methods = (
            self.disposition.apply_with_appended(
                snapshot,
                self.record_target,
                self.member_stmt,
                self.ancestor_decl,
                record_methods_to_append,
                ancestor_methods_to_append,
                geometry,
                add_span,
                dropped_by_file,
            )
        )

        rec_geom = geometry(self.record_target.file_path)
        _, rec_end = rec_geom.required_node_offsets(self.record_target.node)
        nl_after = rec_geom.source.find("\n", rec_end)
        insert_offset = nl_after + 1 if nl_after != -1 else len(rec_geom.source)
        subtype_block = self._render_subtype_class(rec_geom.source, subtype_methods)
        if remaining_record_methods:
            rec_start, _ = rec_geom.required_node_offsets(self.record_target.node)
            rec_line_start = rec_geom.source.rfind("\n", 0, rec_start) + 1
            rec_indent = rec_geom.source[rec_line_start:rec_start]
            if not rec_indent.isspace():
                rec_indent = ""
            member_indent = f"{rec_indent}    "
            appended_block = (
                "\n"
                + _indent_methods(
                    [rendered for _name, rendered in remaining_record_methods],
                    member_indent,
                )
                + "\n"
            )
            add_span(
                self.record_target.file_path,
                insert_offset,
                insert_offset,
                appended_block + subtype_block,
            )
        else:
            add_span(
                self.record_target.file_path,
                insert_offset,
                insert_offset,
                subtype_block,
            )

        for method_name, rendered in remaining_ancestor_methods:
            extra_edits.extend(
                _class_append_edits(
                    self.base_owner_target.file_path,
                    geometry(self.base_owner_target.file_path),
                    self.base_owner_target.node,
                    rendered,
                    member_names=(method_name,),
                    rationale=rationale,
                )
            )

        for file_path, dropped in dropped_by_file.items():
            extra_edits.extend(
                obsolete_import_edits(snapshot, file_path, tuple(dropped), rationale)
            )

        span_edits = tuple(
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    sorted(reps, key=lambda r: (r.start_offset, r.end_offset))
                ),
                rationale=rationale,
            )
            for file_path, reps in replacements_by_file.items()
        )
        return (*span_edits, *extra_edits)


def _can_delete_own_state_member(
    snapshot: CodemodSourceSnapshot,
    symbols: tuple[str, ...],
    member_name: str,
    consumed_attribute_ids: frozenset[int],
) -> bool:
    for parsed in snapshot.parsed_modules:
        if member_name not in parsed.source:
            continue
        for node in parsed.fact(ModuleNodeIndex).attributes_by_attr.get(member_name, ()):
            if id(node) not in consumed_attribute_ids:
                return False
    classes = snapshot.overlap_class_index.classes_by_symbol
    for sym in symbols:
        try:
            NamespaceReaders(snapshot).require_unread(
                classes[sym], frozenset((member_name,))
            )
        except ValueError:
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class OwnStateBranchAnswerOperation(RepositorySourceReprovedOperation):
    """Split a frozen record's derived bool state into a subtype and push consumer branches onto it."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    subtype_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    subtype_docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    consumers: tuple[ConstantAnswerPushConsumer, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(ConstantAnswerPushConsumer)
    )
    factory_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec(), default="classified"
    )
    delete_if_unread: bool = codemod_payload_field(
        BooleanPayloadValueCodec(), default=True
    )

    def __post_init__(self) -> None:
        for label, val in (
            ("member_name", self.member_name),
            ("subtype_name", self.subtype_name),
            ("factory_name", self.factory_name),
        ):
            if not val.isidentifier() or keyword_module.iskeyword(val):
                raise ValueError(f"{label} must be a Python identifier")
        if not self.consumers:
            raise ValueError("consumers must be non-empty")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        proof = OwnStateBranchAnswerProof.proved(
            snapshot,
            self.target,
            self.member_name,
            self.subtype_name,
            self.subtype_docstring,
            self.consumers,
            factory_name=self.factory_name,
            delete_if_unread=self.delete_if_unread,
        )
        return proof.edits(snapshot, self.rationale_text(proof.rationale()))


class OwnStateBranchAnswerFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    PrewarmTypeCheckerMixin,
    FindingRecipeSynthesizer,
    OwnStateBranchAnswerConcept,
    ABC,
):
    """Turn a proved own-state branch answer finding into an owner-picked subtype split codemod."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not isinstance(
            finding.metrics, OwnStateBranchAnswerMetrics
        ):
            return self.rejected_evaluation(
                "splitting an own-state branch answer needs source context and structured metrics"
            )
        metrics = finding.metrics
        if not metrics.consumer_locations:
            return self.rejected_evaluation("finding carries no branch-read consumers")
        snapshot = context.execution_snapshot()
        classes = snapshot.overlap_class_index.classes_by_symbol
        record_indexed = classes.get(metrics.record_symbol)
        if record_indexed is None:
            return self.rejected_evaluation(
                f"record `{metrics.record_symbol}` left the source"
            )
        record_target = source_rewrite_target(record_indexed)
        pick = metrics.picked_base

        # Inspect consumers to see which ones are auto-named whole functions vs embedded.
        embedded_indices: list[int] = []
        consumer_entries: list[tuple[SourceRewriteTarget, int, str | None, str | None]] = []
        for idx, consumer in enumerate(metrics.consumer_locations):
            consumer_target = SourceRewriteTarget(
                file_path=consumer.file_path, qualname=consumer.symbol
            )
            try:
                _path, _enclosure, function = TypeTestDispatchProof.locate_function(
                    snapshot, consumer_target, test_line=consumer.line
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
                consumer_entries.append(
                    (consumer_target, consumer.line, function.name, ast.get_docstring(function))
                )
            else:
                embedded_indices.append(idx)
                consumer_entries.append((consumer_target, consumer.line, None, None))

        if len(embedded_indices) > 1:
            probe_consumers = tuple(
                ConstantAnswerPushConsumer(
                    target=c_target,
                    test_line=c_line,
                    method_name=c_name or f"__nra_own_state_method_{idx}",
                    docstring=c_doc,
                )
                for idx, (c_target, c_line, c_name, c_doc) in enumerate(consumer_entries)
            )
            try:
                OwnStateBranchAnswerProof.proved(
                    snapshot,
                    record_target,
                    metrics.member_name,
                    "__NraOwnStateSubtype",
                    "Nominal state subtype.",
                    probe_consumers,
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            return self.stopped_evaluation(
                NeedsInput(
                    f"`{metrics.record_qualname}.{metrics.member_name}` has {len(embedded_indices)} "
                    "embedded branch consumers; supply per-consumer method names via JSON replay",
                    NeededInput.OWNER_PICK,
                )
            )

        needs_method_name = len(embedded_indices) == 1
        if pick is None or (needs_method_name and not pick.hoisted_name):
            probe_consumers = tuple(
                ConstantAnswerPushConsumer(
                    target=c_target,
                    test_line=c_line,
                    method_name=c_name or "__nra_own_state_method",
                    docstring=c_doc,
                )
                for c_target, c_line, c_name, c_doc in consumer_entries
            )
            try:
                OwnStateBranchAnswerProof.proved(
                    snapshot,
                    record_target,
                    metrics.member_name,
                    "__NraOwnStateSubtype",
                    "Nominal state subtype.",
                    probe_consumers,
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            method_hint = (
                " and `hoisted_name` (pushed method name)"
                if needs_method_name
                else ""
            )
            return self.stopped_evaluation(
                NeedsInput(
                    f"own-state predicate `{metrics.record_qualname}.{metrics.member_name}` can split into a subtype; "
                    f"supply `owner` `{metrics.record_qualname}.{metrics.member_name}` with `name` (True-state subtype class), "
                    f"`judgement` (one-line docstring){method_hint} in `--codemod-owner-picks`",
                    NeededInput.OWNER_PICK,
                )
            )

        resolved_consumers = tuple(
            ConstantAnswerPushConsumer(
                target=c_target,
                test_line=c_line,
                method_name=c_name or (pick.hoisted_name or ""),
                docstring=c_doc if c_name is not None else pick.hoisted_judgement,
            )
            for c_target, c_line, c_name, c_doc in consumer_entries
        )
        operation = OwnStateBranchAnswerOperation(
            target=record_target,
            member_name=metrics.member_name,
            subtype_name=pick.name,
            subtype_docstring=pick.judgement,
            consumers=resolved_consumers,
        )
        try:
            proof = OwnStateBranchAnswerProof.proved(
                snapshot,
                record_target,
                metrics.member_name,
                pick.name,
                pick.judgement,
                resolved_consumers,
            )
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=proof.rationale(),
            ).with_proved_operation(operation, snapshot)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "OwnStateBranchAnswerFindingRecipeSynthesizer",
    "OwnStateBranchAnswerOperation",
    "OwnStateBranchAnswerProof",
)
