"""Replace a 0/1 tuple return (`()` or `(x,)`) with a `Held` / `Absent` nominal family.

When a function or method returns only `()` or `(x,)` (with both arities) and all
callers are proved closed to iteration or `for`-loop dispatch, introducing a nominal
family (`Base`, `AbsentBase`, `HeldBase(value: T)`) replaces the 0/1 tuple with an
explicit present/absent type, pushes clean `for x in fn(...):` consumer bodies onto
the family as polymorphic methods (leaving `AbsentBase` empty), and implements
`__iter__` (`iter(())` on `Base`, `yield self.value` on `HeldBase`) whenever
unpacking/comprehension/iterable consumers remain.
"""

from __future__ import annotations

import ast
import copy
import keyword as keyword_module
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.ast_field_decoders import (
    ParameterAnnotations,
    declared_annotations,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_projection import CallKeywords, ExpressionReference
from nominal_refactor_advisor.ast_tools import AstParentIndex, ParsedModule
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
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
from nominal_refactor_advisor.models import RefactorFinding, ZeroOneTupleMetrics
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import ReceiverSpan
from nominal_refactor_advisor.refactor_concepts import ZeroOneTupleFamilyConcept
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    _contains_super_call,
    _statement_full_line_span,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    PrewarmTypeCheckerMixin,
)
from nominal_refactor_advisor.refactors.threaded_absence import (
    _is_abstract_or_stub_body,
    _locate_class_method,
    _module_exports_name,
    _module_top_level_bound_names,
    _top_level_enclosing_stmt,
)
from nominal_refactor_advisor.refactors.type_test_dispatch import (
    FunctionEnclosure,
    FunctionNode,
    InstanceDispatchKind,
    TypeTestDispatchProof,
    _render_method,
)
from nominal_refactor_advisor.relocation_observations import function_nodes


_PURE_ITER_BUILTINS = frozenset(("tuple", "list", "set", "frozenset", "iter"))
_SINGLE_ELEM_CONTAINER_NAMES = frozenset(("Iterable", "Sequence", "Collection"))


def _extract_zero_one_element_type(returns: ast.expr) -> ast.expr:
    """Extract `T` from `tuple[T, ...]` or `Iterable[T]`/`Sequence[T]`/`Collection[T]`."""

    match returns:
        case ast.Subscript(
            value=wrapper,
            slice=ast.Tuple(elts=[elem_type, ast.Constant(value=val)]),
        ) if (
            ExpressionReference.decoded(wrapper).terminal_name in ("tuple", "Tuple")
            and val is Ellipsis
        ):
            return elem_type
        case ast.Subscript(value=wrapper, slice=slice_expr) if (
            ExpressionReference.decoded(wrapper).terminal_name
            in _SINGLE_ELEM_CONTAINER_NAMES
            and not isinstance(slice_expr, ast.Tuple)
        ):
            return slice_expr
    raise NeedsInput(
        f"return annotation `{ast.unparse(returns)}` is not `tuple[T, ...]` or `Iterable[T]`/`Sequence[T]`",
        NeededInput.PROOF,
    )


class _ZeroOneReturnRewrite(ABC):
    """One `return ()` or `return (expr,)` rewrite inside the producer function."""

    @property
    @abstractmethod
    def tuple_node(self) -> ast.Tuple:
        raise NotImplementedError

    @abstractmethod
    def replacement_text(
        self,
        source_geometry: SourceTextGeometry,
        absent_name: str,
        held_name: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def arity(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class _AbsentTupleReturn(_ZeroOneReturnRewrite):
    """`return ()` -> `return AbsentFamily()`."""

    node: ast.Tuple

    @property
    def tuple_node(self) -> ast.Tuple:
        return self.node

    def replacement_text(
        self,
        source_geometry: SourceTextGeometry,
        absent_name: str,
        held_name: str,
    ) -> str:
        del source_geometry, held_name
        return f"{absent_name}()"

    def arity(self) -> int:
        return 0


@dataclass(frozen=True)
class _HeldTupleReturn(_ZeroOneReturnRewrite):
    """`return (expr,)` -> `return HeldFamily(expr)`."""

    node: ast.Tuple
    item_expr: ast.expr

    @property
    def tuple_node(self) -> ast.Tuple:
        return self.node

    def replacement_text(
        self,
        source_geometry: SourceTextGeometry,
        absent_name: str,
        held_name: str,
    ) -> str:
        del absent_name
        item_start, item_end = source_geometry.required_node_offsets(self.item_expr)
        item_src = source_geometry.source[item_start:item_end]
        if isinstance(self.item_expr, ast.NamedExpr):
            item_src = f"({item_src})"
        return f"{held_name}({item_src})"

    def arity(self) -> int:
        return 1


def _collect_override_zero_one_returns(
    function: FunctionNode,
    qualname: str,
) -> tuple[_ZeroOneReturnRewrite, ...]:
    """Prove every `return` of one hierarchy override returns `()` or `(x,)` (or stub)."""

    if _is_abstract_or_stub_body(function):
        return ()
    rewrites: list[_ZeroOneReturnRewrite] = []
    for child in function_nodes(function):
        match child:
            case ast.Return(value=ast.Tuple(elts=elements) as tup) if (
                len(elements) <= 1
                and not any(isinstance(el, ast.Starred) for el in elements)
            ):
                if not elements:
                    rewrites.append(_AbsentTupleReturn(node=tup))
                else:
                    rewrites.append(_HeldTupleReturn(node=tup, item_expr=elements[0]))
            case ast.Return():
                raise NeedsInput(
                    f"override `{qualname}` at line {child.lineno} returns a non-0/1 value; "
                    "overriding hierarchy is not pure 0/1",
                    NeededInput.PROOF,
                )
    if not rewrites:
        raise NeedsInput(
            f"override `{qualname}` has no return statement; overriding hierarchy is not pure 0/1",
            NeededInput.PROOF,
        )
    return tuple(rewrites)


@dataclass(frozen=True)
class _HierarchyOverrideMember:
    """One function or method declaration participating in a 0/1-tuple family rewrite."""

    file_path: str
    class_symbol: str
    function: FunctionNode
    return_annotations: tuple[ast.expr, ...]
    return_rewrites: tuple[_ZeroOneReturnRewrite, ...]

    def needed_family_names(
        self, family_name: str, absent_name: str, held_name: str
    ) -> tuple[str, ...]:
        names: list[str] = []
        if self.return_annotations:
            names.append(family_name)
        arities = {rw.arity() for rw in self.return_rewrites}
        if 0 in arities:
            names.append(absent_name)
        if 1 in arities:
            names.append(held_name)
        return tuple(names)

    def span_replacements(
        self,
        geometry: SourceTextGeometry,
        family_name: str,
        absent_name: str,
        held_name: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        reps: list[SourceTextSpanReplacement] = []
        for ret_ann in self.return_annotations:
            ret_start, ret_end = geometry.required_node_offsets(ret_ann)
            reps.append(
                SourceTextSpanReplacement(
                    start_offset=ret_start,
                    end_offset=ret_end,
                    replacement_source=family_name,
                )
            )
        for rw in self.return_rewrites:
            t_start, t_end = geometry.required_node_offsets(rw.tuple_node)
            reps.append(
                SourceTextSpanReplacement(
                    start_offset=t_start,
                    end_offset=t_end,
                    replacement_source=rw.replacement_text(
                        geometry, absent_name, held_name
                    ),
                )
            )
        return tuple(reps)


@dataclass(frozen=True)
class _ResolvedZeroOneHierarchy:
    """Connected single-function or overriding-method hierarchy proved pure 0/1."""

    family_file_path: str
    family_anchor_line: int
    element_type: ast.expr
    members: tuple[_HierarchyOverrideMember, ...]
    reaching_symbols: frozenset[str]
    owner_qualnames_by_file: Mapping[str, frozenset[str]]


def _element_type_assignable(
    snapshot: CodemodSourceSnapshot,
    sub_mod: ParsedModule,
    sub_elem: ast.expr,
    sub_line: int,
    root_mod: ParsedModule,
    root_elem: ast.expr,
    root_line: int,
) -> bool:
    """Whether `sub_elem` matches or is a scanned subclass of `root_elem`."""

    if ast.unparse(sub_elem) == ast.unparse(root_elem):
        return True
    bindings = snapshot.module_binding_proof
    sub_w = bindings.reference_or_builtin_witness_at(sub_mod, sub_elem, line=sub_line)
    root_w = bindings.reference_or_builtin_witness_at(root_mod, root_elem, line=root_line)
    if sub_w is None or root_w is None:
        return False
    if sub_w.qualified_name == root_w.qualified_name:
        return True
    index = snapshot.compact_class_index
    return root_w.qualified_name in index.ancestor_symbols(sub_w.qualified_name)


def _resolve_zero_one_hierarchy(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
    primary_return_annotation: ast.expr,
    primary_element_type: ast.expr,
) -> _ResolvedZeroOneHierarchy:
    """Resolve either a standalone function or an entire pure-0/1 overriding hierarchy."""

    owner_symbol = enclosure.declaring_class_symbol(snapshot)
    if owner_symbol is None:
        rewrites = _collect_zero_one_returns(function)
        return _ResolvedZeroOneHierarchy(
            family_file_path=source_path,
            family_anchor_line=function.lineno,
            element_type=primary_element_type,
            members=(
                _HierarchyOverrideMember(
                    file_path=source_path,
                    class_symbol="",
                    function=function,
                    return_annotations=(primary_return_annotation,),
                    return_rewrites=rewrites,
                ),
            ),
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

    sorted_syms = tuple(sorted(declared_syms))
    roots = tuple(
        sym
        for sym in sorted_syms
        if not any(anc in declared_syms for anc in index.ancestor_symbols(sym))
    )
    root_files = {classes[sym].file_path for sym in roots}
    if len(root_files) != 1:
        raise NeedsInput(
            f"overriding hierarchy for `{function.name}` has root declarations across multiple modules; "
            "cannot place a single family cleanly",
            NeededInput.SOURCE_CHANGE,
        )
    family_file_path = next(iter(root_files))
    family_mod = snapshot.parsed_module_for_source_path(family_file_path)
    is_async = isinstance(function, ast.AsyncFunctionDef)

    located: list[tuple[str, str, ast.ClassDef, FunctionNode]] = []
    for sym in sorted_syms:
        decl = classes[sym]
        cmod = snapshot.parsed_module_for_source_path(decl.file_path)
        found = _locate_class_method(cmod, sym, function.name)
        if found is None:
            raise NeedsInput(
                f"could not locate `def {function.name}` on `{decl.qualname}`; overriding hierarchy is not rewritten",
                NeededInput.PROOF,
            )
        cls_node, method_fn = found
        if isinstance(method_fn, ast.AsyncFunctionDef) != is_async:
            raise NeedsInput(
                f"`{decl.qualname}.{function.name}` mixes sync and async overrides",
                NeededInput.PROOF,
            )
        located.append((sym, decl.file_path, cls_node, method_fn))

    # Determine family element type from the root declaration in `family_file_path`.
    family_element_type: ast.expr | None = None
    family_element_line: int = function.lineno
    root_set = frozenset(roots)
    for sym, _fp, _cls_node, method_fn in located:
        if sym not in root_set:
            continue
        ret_anns = declared_annotations(method_fn)
        if ret_anns:
            family_element_type = _extract_zero_one_element_type(ret_anns[0])
            family_element_line = method_fn.lineno
            break
    if family_element_type is None:
        if family_file_path != source_path:
            raise NeedsInput(
                f"root declaration of `{function.name}` in `{family_file_path}` has no return annotation",
                NeededInput.PROOF,
            )
        family_element_type = primary_element_type
        family_element_line = function.lineno

    members: list[_HierarchyOverrideMember] = []
    root_anchor_lines: list[int] = []
    qualnames_by_file: dict[str, set[str]] = {}
    for sym, file_path, cls_node, method_fn in located:
        decl = classes[sym]
        if sym in root_set:
            root_anchor_lines.append(cls_node.lineno)
        qualnames_by_file.setdefault(file_path, set()).add(decl.qualname)
        ret_anns = declared_annotations(method_fn)
        if ret_anns:
            member_elem = _extract_zero_one_element_type(ret_anns[0])
            member_mod = snapshot.parsed_module_for_source_path(file_path)
            if not _element_type_assignable(
                snapshot,
                member_mod,
                member_elem,
                method_fn.lineno,
                family_mod,
                family_element_type,
                family_element_line,
            ):
                raise NeedsInput(
                    f"override `{decl.qualname}.{function.name}` element type `{ast.unparse(member_elem)}` "
                    f"is not compatible with `{ast.unparse(family_element_type)}`",
                    NeededInput.PROOF,
                )
        member_rewrites = _collect_override_zero_one_returns(
            method_fn, f"{decl.qualname}.{function.name}"
        )
        members.append(
            _HierarchyOverrideMember(
                file_path=file_path,
                class_symbol=sym,
                function=method_fn,
                return_annotations=ret_anns,
                return_rewrites=member_rewrites,
            )
        )

    all_arities = {rw.arity() for m in members for rw in m.return_rewrites}
    if all_arities != {0, 1}:
        raise NeedsInput(
            f"`{function.name}` hierarchy does not return both `()` and `(x,)`",
            NeededInput.PROOF,
        )

    reaching_symbols = frozenset().union(
        *(frozenset((sym, *index.descendant_symbols(sym))) for sym in declared_syms)
    )
    return _ResolvedZeroOneHierarchy(
        family_file_path=family_file_path,
        family_anchor_line=min(root_anchor_lines),
        element_type=family_element_type,
        members=tuple(members),
        reaching_symbols=reaching_symbols,
        owner_qualnames_by_file={
            fp: frozenset(names) for fp, names in qualnames_by_file.items()
        },
    )


def _collect_zero_one_returns(
    function: FunctionNode,
) -> tuple[_ZeroOneReturnRewrite, ...]:
    """Prove every `return` of `function` returns `()` or `(x,)` with both arities."""

    rewrites: list[_ZeroOneReturnRewrite] = []
    for child in function_nodes(function):
        match child:
            case ast.Return(value=ast.Tuple(elts=elements) as tup) if (
                len(elements) <= 1
                and not any(isinstance(el, ast.Starred) for el in elements)
            ):
                if not elements:
                    rewrites.append(_AbsentTupleReturn(node=tup))
                else:
                    rewrites.append(_HeldTupleReturn(node=tup, item_expr=elements[0]))
            case ast.Return():
                raise NeedsInput(
                    f"return at line {child.lineno} in `{function.name}` is not a 0/1 tuple",
                    NeededInput.PROOF,
                )
    arities = {rw.arity() for rw in rewrites}
    if arities != {0, 1}:
        raise NeedsInput(
            f"`{function.name}` does not return both `()` and `(x,)`",
            NeededInput.PROOF,
        )
    return tuple(rewrites)


class _ZeroOneConsumer(ABC):
    """How one closed call site consumes the 0/1 tuple result."""

    @abstractmethod
    def requires_iter(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pushed_loop(self) -> _PushedForLoopConsumer | None:
        raise NotImplementedError


@dataclass(frozen=True)
class _IterableProtocolConsumer(_ZeroOneConsumer):
    """A call site that consumes the result purely through Python's iteration protocol."""

    file_path: str
    line: int

    def requires_iter(self) -> bool:
        return True

    def pushed_loop(self) -> _PushedForLoopConsumer | None:
        return None


@dataclass(frozen=True)
class _PushedForLoopConsumer(_ZeroOneConsumer):
    """A `for x in fn(...): <body>` site whose body moves cleanly onto `Held<Family>`."""

    file_path: str
    for_stmt: ast.For
    call_expr: ast.expr
    loop_var: str
    rewritten_body: tuple[ast.stmt, ...]
    param_names: tuple[str, ...]
    call_arg_exprs: tuple[str, ...]
    param_annotations: Mapping[str, ast.expr]

    def requires_iter(self) -> bool:
        return False

    def pushed_loop(self) -> _PushedForLoopConsumer | None:
        return self


class _PushedLoopPlan(ABC):
    """Whether the generated family carries a pushed `for`-loop consumer method."""

    @abstractmethod
    def render_base_methods(
        self,
        method_name: str,
        method_docstring: str | None,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def render_held_methods(
        self,
        method_name: str,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def call_site_replacements(
        self,
        snapshot: CodemodSourceSnapshot,
        method_name: str,
    ) -> tuple[tuple[str, SourceTextSpanReplacement], ...]:
        raise NotImplementedError

    @abstractmethod
    def owner_pick_hint(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class _NoPushedLoop(_PushedLoopPlan):
    """No direct `for`-loop body was pushed; consumers iterate through `__iter__`."""

    def render_base_methods(
        self,
        method_name: str,
        method_docstring: str | None,
    ) -> tuple[str, ...]:
        del method_name, method_docstring
        return ()

    def render_held_methods(self, method_name: str) -> tuple[str, ...]:
        del method_name
        return ()

    def call_site_replacements(
        self,
        snapshot: CodemodSourceSnapshot,
        method_name: str,
    ) -> tuple[tuple[str, SourceTextSpanReplacement], ...]:
        del snapshot, method_name
        return ()

    def owner_pick_hint(self) -> str:
        return ""


@dataclass(frozen=True)
class _ActivePushedLoop(_PushedLoopPlan):
    """One pushed `for x in fn(...): <body>` consumer rewritten to `fn(...).<method>(...)`."""

    consumer: _PushedForLoopConsumer

    def render_base_methods(
        self,
        method_name: str,
        method_docstring: str | None,
    ) -> tuple[str, ...]:
        base_method = _render_method(
            InstanceDispatchKind(),
            method_name,
            self.consumer.param_names,
            (ast.Return(value=ast.Constant(value=None)),),
            is_async=False,
            docstring=method_docstring,
            param_annotations=self.consumer.param_annotations,
            return_annotation=ast.Constant(value=None),
        )
        return ("", *(f"    {line}" if line else "" for line in base_method.splitlines()))

    def render_held_methods(self, method_name: str) -> tuple[str, ...]:
        held_method = _render_method(
            InstanceDispatchKind(),
            method_name,
            self.consumer.param_names,
            self.consumer.rewritten_body,
            is_async=False,
            docstring=None,
            param_annotations=self.consumer.param_annotations,
            return_annotation=ast.Constant(value=None),
        )
        return ("", *(f"    {line}" if line else "" for line in held_method.splitlines()))

    def call_site_replacements(
        self,
        snapshot: CodemodSourceSnapshot,
        method_name: str,
    ) -> tuple[tuple[str, SourceTextSpanReplacement], ...]:
        geom = SourceTextGeometry(snapshot.sources_by_file_path[self.consumer.file_path])
        call_start, call_end = geom.required_node_offsets(self.consumer.call_expr)
        call_src = geom.source[call_start:call_end]
        if isinstance(self.consumer.call_expr, ast.Await):
            call_src = f"({call_src})"
        span_start, span_end, indent = _statement_full_line_span(
            geom, self.consumer.for_stmt
        )
        args_text = ", ".join(self.consumer.call_arg_exprs)
        replacement = f"{indent}{call_src}.{method_name}({args_text})\n"
        return (
            (
                self.consumer.file_path,
                SourceTextSpanReplacement(
                    start_offset=span_start,
                    end_offset=span_end,
                    replacement_source=replacement,
                ),
            ),
        )

    def owner_pick_hint(self) -> str:
        return "; also supply `hoisted_name` for the pushed `for`-loop method (defaults to `apply`)"


def _enclosing_function_of(
    node: ast.AST, parents: AstParentIndex
) -> FunctionNode | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parents.get(cur)
    return None


def _loop_body_escapes_or_suspends(stmts: Sequence[ast.stmt]) -> bool:
    """True when `stmts` contains `return`/`break`/`continue`/`yield`/`await`/`global`/`nonlocal` or `super()`."""

    stack: list[ast.AST] = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(
            node,
            (
                ast.Return,
                ast.Break,
                ast.Continue,
                ast.Yield,
                ast.YieldFrom,
                ast.Await,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return _contains_super_call(stmts)


def _function_param_and_local_annotations(
    fn: FunctionNode,
) -> dict[str, ast.expr]:
    anns: dict[str, ast.expr] = {
        param.arg: param.annotation
        for param in ParameterAnnotations.of(
            (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
        ).annotated
    }
    for child in function_nodes(fn):
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            anns.setdefault(child.target.id, child.annotation)
    return anns


def _rewrite_loop_body_names(
    stmts: Sequence[ast.stmt],
    loop_var: str,
    self_alias: str | None,
) -> tuple[ast.stmt, ...]:
    """Rewrite `loop_var` -> `self.value` and optional outer `self` -> `self_alias`."""

    class _Rewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load):
                if node.id == loop_var:
                    return ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="value",
                        ctx=ast.Load(),
                    )
                if self_alias is not None and node.id == "self":
                    return ast.Name(id=self_alias, ctx=ast.Load())
            return node

    rewritten: list[ast.stmt] = []
    transformer = _Rewriter()
    for stmt in stmts:
        cloned = copy.deepcopy(stmt)
        transformed = transformer.visit(cloned)
        if not isinstance(transformed, ast.stmt):
            raise NeedsInput("loop body rewrite produced non-statement", NeededInput.PROOF)
        ast.fix_missing_locations(transformed)
        rewritten.append(transformed)
    return tuple(rewritten)


def _try_build_pushed_for_loop(
    caller_mod: ParsedModule,
    caller_node_index: ModuleNodeIndex,
    for_stmt: ast.For,
    call_expr: ast.expr,
    source_path: str,
) -> _ZeroOneConsumer:
    """Push `for x in fn(...): <body>` onto the family when sound; otherwise fall back to `__iter__`."""

    fallback = _IterableProtocolConsumer(file_path=caller_mod.file_path, line=for_stmt.lineno)
    if (
        for_stmt.orelse
        or not isinstance(for_stmt.target, ast.Name)
        or caller_mod.file_path != source_path
    ):
        return fallback
    caller_fn = _enclosing_function_of(for_stmt, caller_node_index.parents)
    if caller_fn is None or _loop_body_escapes_or_suspends(for_stmt.body):
        return fallback

    loop_var = for_stmt.target.id
    # Collect names bound or referenced in the enclosing function outside `for_stmt`.
    for_node_ids = {id(n) for n in ast.walk(for_stmt)}
    fn_param_names = {
        arg.arg
        for arg in (
            *caller_fn.args.posonlyargs,
            *caller_fn.args.args,
            *caller_fn.args.kwonlyargs,
            *variadic_parameters(caller_fn.args),
        )
    }

    outer_bound: set[str] = set(fn_param_names)
    outer_reads: set[str] = set()
    for node in function_nodes(caller_fn):
        if id(node) in for_node_ids:
            continue
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                outer_bound.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                outer_reads.add(node.id)

    body_stores: set[str] = set()
    body_loads_ordered: list[str] = []
    for stmt in for_stmt.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    body_stores.add(node.id)
                elif isinstance(node.ctx, ast.Load) and node.id not in body_loads_ordered:
                    body_loads_ordered.append(node.id)

    if loop_var in body_stores:
        return fallback
    # Refuse if loop body rebinds any variable visible outside the loop.
    if body_stores & (outer_bound | outer_reads):
        return fallback

    captured_raw = tuple(
        name
        for name in body_loads_ordered
        if name != loop_var and name not in body_stores and name in outer_bound
    )
    self_alias: str | None = None
    if "self" in captured_raw:
        used_names = set(captured_raw) | body_stores | {loop_var, "self"}
        candidate = "owner"
        suffix = 1
        while candidate in used_names:
            candidate = f"owner_{suffix}"
            suffix += 1
        self_alias = candidate

    param_names = tuple(
        self_alias if (name == "self" and self_alias is not None) else name
        for name in captured_raw
    )
    all_anns = _function_param_and_local_annotations(caller_fn)
    param_anns: dict[str, ast.expr] = {}
    for raw_name, param_name in zip(captured_raw, param_names, strict=True):
        if raw_name in all_anns:
            param_anns[param_name] = all_anns[raw_name]

    rewritten_body = _rewrite_loop_body_names(for_stmt.body, loop_var, self_alias)
    return _PushedForLoopConsumer(
        file_path=caller_mod.file_path,
        for_stmt=for_stmt,
        call_expr=call_expr,
        loop_var=loop_var,
        rewritten_body=rewritten_body,
        param_names=param_names,
        call_arg_exprs=captured_raw,
        param_annotations=param_anns,
    )


def _classify_consumer_use(
    caller_mod: ParsedModule,
    caller_node_index: ModuleNodeIndex,
    call_node: ast.Call,
    *,
    is_async: bool,
    function_name: str,
    source_path: str,
) -> _ZeroOneConsumer:
    """Prove one call site only uses iteration or a pushable `for` loop."""

    parent = caller_node_index.parents.get(call_node)
    consumer_expr: ast.expr = call_node
    if is_async:
        if not (isinstance(parent, ast.Await) and parent.value is call_node):
            raise NeedsInput(
                f"{caller_mod.file_path}:{call_node.lineno} does not await `{function_name}(...)` directly",
                NeededInput.PROOF,
            )
        consumer_expr = parent
        parent = caller_node_index.parents.get(consumer_expr)

    if isinstance(parent, ast.For) and parent.iter is consumer_expr:
        return _try_build_pushed_for_loop(
            caller_mod, caller_node_index, parent, consumer_expr, source_path
        )
    if isinstance(parent, ast.Starred) and parent.value is consumer_expr:
        return _IterableProtocolConsumer(file_path=caller_mod.file_path, line=call_node.lineno)
    if isinstance(parent, ast.comprehension) and parent.iter is consumer_expr:
        return _IterableProtocolConsumer(file_path=caller_mod.file_path, line=call_node.lineno)
    if isinstance(parent, ast.YieldFrom) and parent.value is consumer_expr:
        return _IterableProtocolConsumer(file_path=caller_mod.file_path, line=call_node.lineno)
    if isinstance(parent, ast.IfExp) and (
        parent.body is consumer_expr or parent.orelse is consumer_expr
    ):
        grandparent = caller_node_index.parents.get(parent)
        if (
            (isinstance(grandparent, ast.Starred) and grandparent.value is parent)
            or (isinstance(grandparent, ast.For) and grandparent.iter is parent)
            or (isinstance(grandparent, ast.comprehension) and grandparent.iter is parent)
            or (isinstance(grandparent, ast.YieldFrom) and grandparent.value is parent)
        ):
            return _IterableProtocolConsumer(
                file_path=caller_mod.file_path, line=call_node.lineno
            )
    if isinstance(parent, ast.Call):
        kw = CallKeywords.of(parent.keywords)
        if not kw.named and not kw.splats and parent.args == [consumer_expr]:
            if (
                isinstance(parent.func, ast.Name)
                and parent.func.id in _PURE_ITER_BUILTINS
            ):
                return _IterableProtocolConsumer(
                    file_path=caller_mod.file_path, line=call_node.lineno
                )
            if (
                isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "extend"
            ):
                return _IterableProtocolConsumer(
                    file_path=caller_mod.file_path, line=call_node.lineno
                )

    parent_name = type(parent).__name__ if parent is not None else "None"
    raise NeedsInput(
        f"{caller_mod.file_path}:{call_node.lineno} uses `{function_name}(...)` in a non-iterable context ({parent_name}); callers are not closed to iteration",
        NeededInput.PROOF,
    )


def _collect_closed_consumers(
    snapshot: CodemodSourceSnapshot,
    source_path: str,
    enclosure: FunctionEnclosure,
    function: FunctionNode,
    hierarchy: _ResolvedZeroOneHierarchy,
) -> tuple[_ZeroOneConsumer, ...]:
    """Prove every reference to `function` (or its overriding hierarchy) is a closed iterable/loop call."""

    is_private = function.name.startswith("_")
    is_async = isinstance(function, ast.AsyncFunctionDef)
    bindings = snapshot.module_binding_proof
    source_module = snapshot.parsed_module_for_source_path(source_path)
    owner_symbol = enclosure.declaring_class_symbol(snapshot)
    consumers: list[_ZeroOneConsumer] = []

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
                consumers.append(
                    _classify_consumer_use(
                        mod,
                        node_index,
                        parent,
                        is_async=is_async,
                        function_name=function.name,
                        source_path=hierarchy.family_file_path,
                    )
                )
        if not is_private and not consumers:
            raise NeedsInput(
                f"public function `{function.name}` has no scanned callers; cannot prove callers are closed",
                NeededInput.PROOF,
            )
        return tuple(consumers)

    index = snapshot.compact_class_index
    classes = index.classes_by_symbol
    reaching_symbols = hierarchy.reaching_symbols
    owner_qualname = classes[owner_symbol].qualname
    reaching_files = frozenset(
        (
            hierarchy.family_file_path,
            *(
                classes[sym].file_path
                for sym in reaching_symbols
                if sym in classes
            ),
        )
    )

    definite_hits: list[tuple[ParsedModule, ModuleNodeIndex, ast.Attribute, ast.Call | None]] = []
    deferred_checks: list[
        tuple[ParsedModule, ModuleNodeIndex, ast.Attribute, ast.Call | None, ReceiverSpan]
    ] = []
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
                    definite_hits.append((mod, node_index, node, parent_call))
                    continue
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in local_owner_qualnames
            ):
                definite_hits.append((mod, node_index, node, parent_call))
                continue
            witness = bindings.reference_or_builtin_witness_at(
                mod, node.value, line=node.lineno
            )
            if witness is not None and witness.qualified_name in reaching_symbols:
                definite_hits.append((mod, node_index, node, parent_call))
                continue
            deferred_checks.append(
                (
                    mod,
                    node_index,
                    node,
                    parent_call,
                    ReceiverSpan.of(mod.file_path, node.value),
                )
            )

    inferred_receivers = (
        snapshot.type_checker.receiver_types(
            tuple(dict.fromkeys(span for _, _, _, _, span in deferred_checks))
        )
        if deferred_checks
        else {}
    )
    all_hits = list(definite_hits)
    for mod, node_index, node, parent_call, span in deferred_checks:
        answer = inferred_receivers.get(span)
        if answer is None:
            raise NeedsInput(
                f"{mod.file_path}:{node.lineno} accesses `.{function.name}` on an unproved receiver",
                NeededInput.PROOF,
            )
        if answer.reaches_method_owner(
            reaching_symbols,
            file_path=mod.file_path,
            line=node.lineno,
            method_name=function.name,
        ):
            all_hits.append((mod, node_index, node, parent_call))

    for mod, node_index, node, parent_call in all_hits:
        if parent_call is None:
            raise NeedsInput(
                f"{mod.file_path}:{node.lineno} references `.{function.name}` as a value; callers are not closed",
                NeededInput.PROOF,
            )
        consumers.append(
            _classify_consumer_use(
                mod,
                node_index,
                parent_call,
                is_async=is_async,
                function_name=function.name,
                source_path=hierarchy.family_file_path,
            )
        )

    if not is_private and not consumers:
        raise NeedsInput(
            f"public method `{owner_qualname}.{function.name}` has no scanned callers; cannot prove callers are closed",
            NeededInput.PROOF,
        )
    return tuple(consumers)


@dataclass(frozen=True)
class ZeroOneTupleProof:
    """A 0/1 tuple producer (or overriding hierarchy) proved safe to replace with a Held/Absent family."""

    family_file_path: str
    function: FunctionNode
    enclosure: FunctionEnclosure
    element_type: ast.expr
    members: tuple[_HierarchyOverrideMember, ...]
    cross_module_imports: tuple[tuple[str, str], ...]
    pushed_loop_plan: _PushedLoopPlan
    needs_iter: bool
    family_name: str
    family_docstring: str
    method_name: str
    method_docstring: str | None
    insert_before_stmt: ast.stmt

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        family_name: str,
        family_docstring: str,
        method_name: str = "apply",
        method_docstring: str | None = None,
    ) -> ZeroOneTupleProof:
        for label, ident in (("family_name", family_name), ("method_name", method_name)):
            if (
                not ident.isidentifier()
                or keyword_module.iskeyword(ident)
                or (ident.startswith("__") and ident.endswith("__"))
            ):
                raise ValueError(f"{label} `{ident}` is not a valid non-dunder identifier")
        if not family_docstring.strip():
            raise ValueError("family_docstring must be non-empty")

        source_path, enclosure, function = TypeTestDispatchProof.locate_function(
            snapshot, target
        )
        if function.name.startswith("__") and function.name.endswith("__"):
            raise NeedsInput(
                f"`{function.name}` is a dunder method whose signature is fixed by Python",
                NeededInput.PROOF,
            )
        returns_tuple = declared_annotations(function)
        if not returns_tuple:
            raise NeedsInput(
                f"`{target.qualname}` has no return annotation; annotate `tuple[T, ...]` or `Iterable[T]` first",
                NeededInput.PROOF,
            )
        return_annotation = returns_tuple[0]
        element_type = _extract_zero_one_element_type(return_annotation)

        hierarchy = _resolve_zero_one_hierarchy(
            snapshot,
            source_path,
            enclosure,
            function,
            return_annotation,
            element_type,
        )
        family_file_path = hierarchy.family_file_path
        family_module = snapshot.parsed_module_for_source_path(family_file_path)

        existing_names = _module_top_level_bound_names(family_module.module)
        absent_name = f"Absent{family_name}"
        held_name = f"Held{family_name}"
        for candidate_name in (family_name, absent_name, held_name):
            if candidate_name in existing_names:
                raise NeedsInput(
                    f"`{candidate_name}` is already bound in `{family_file_path}`; pick another family name",
                    NeededInput.OWNER_PICK,
                )

        # Verify cycle-free cross-module imports for overrides declared outside `family_file_path`.
        import_graph = snapshot.module_import_graph
        needed_by_ext_file: dict[str, list[str]] = {}
        for member in hierarchy.members:
            if member.file_path == family_file_path:
                continue
            ext_names = needed_by_ext_file.setdefault(member.file_path, [])
            for name in member.needed_family_names(family_name, absent_name, held_name):
                if name not in ext_names:
                    ext_names.append(name)

        cross_module_imports: list[tuple[str, str]] = []
        for ext_file, needed_names in sorted(needed_by_ext_file.items()):
            ext_mod = snapshot.parsed_module_for_source_path(ext_file)
            ext_bound = _module_top_level_bound_names(ext_mod.module)
            for name in needed_names:
                if name in ext_bound:
                    raise NeedsInput(
                        f"`{name}` is already bound in override module `{ext_file}`; pick another family name",
                        NeededInput.OWNER_PICK,
                    )
                try:
                    import_src = import_graph.required_import_source(
                        importing_file_path=ext_file,
                        imported_file_path=family_file_path,
                        imported_name=name,
                    )
                except ValueError as error:
                    raise NeedsInput(str(error), NeededInput.SOURCE_CHANGE) from error
                cross_module_imports.append((ext_file, import_src))

        insert_before = _top_level_enclosing_stmt(
            family_module.module, hierarchy.family_anchor_line
        )
        if insert_before is None:
            raise NeedsInput(
                f"could not locate top-level anchor for `{target.qualname}`",
                NeededInput.PROOF,
            )

        consumers = _collect_closed_consumers(
            snapshot, source_path, enclosure, function, hierarchy
        )
        pushed_candidates = [
            pushed
            for consumer in consumers
            if (pushed := consumer.pushed_loop()) is not None
        ]
        if len(pushed_candidates) == 1:
            pushed_loop_plan: _PushedLoopPlan = _ActivePushedLoop(
                consumer=pushed_candidates[0]
            )
            needs_iter = any(consumer.requires_iter() for consumer in consumers)
        else:
            pushed_loop_plan = _NoPushedLoop()
            needs_iter = True

        return cls(
            family_file_path=family_file_path,
            function=function,
            enclosure=enclosure,
            element_type=hierarchy.element_type,
            members=hierarchy.members,
            cross_module_imports=tuple(cross_module_imports),
            pushed_loop_plan=pushed_loop_plan,
            needs_iter=needs_iter,
            family_name=family_name,
            family_docstring=family_docstring,
            method_name=method_name,
            method_docstring=method_docstring,
            insert_before_stmt=insert_before,
        )

    def rationale(self) -> str:
        return (
            f"Replace 0/1 tuple returns in `{self.function.name}` with nominal family "
            f"`{self.family_name}` (`Absent{self.family_name}` / `Held{self.family_name}`)"
        )

    def _render_family_classes(self) -> str:
        elem_src = ast.unparse(self.element_type)
        base_lines = [
            f"class {self.family_name}(ABC):",
            f'    """{self.family_docstring}"""',
        ]
        if self.needs_iter:
            base_lines.extend(
                [
                    "",
                    f"    def __iter__(self) -> Iterator[{elem_src}]:",
                    "        return iter(())",
                ]
            )
        base_lines.extend(
            self.pushed_loop_plan.render_base_methods(
                self.method_name, self.method_docstring
            )
        )

        absent_lines = [
            "",
            "",
            "@dataclass(frozen=True)",
            f"class Absent{self.family_name}({self.family_name}):",
            f'    """Absent `{self.family_name}` carrying no value."""',
        ]

        held_lines = [
            "",
            "",
            "@dataclass(frozen=True)",
            f"class Held{self.family_name}({self.family_name}):",
            f'    """Held `{self.family_name}` carrying its present value."""',
            "",
            f"    value: {elem_src}",
        ]
        if self.needs_iter:
            held_lines.extend(
                [
                    "",
                    f"    def __iter__(self) -> Iterator[{elem_src}]:",
                    "        yield self.value",
                ]
            )
        held_lines.extend(self.pushed_loop_plan.render_held_methods(self.method_name))

        return "\n".join((*base_lines, *absent_lines, *held_lines)) + "\n\n\n"

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        family_module = snapshot.parsed_module_for_source_path(self.family_file_path)
        geometries: dict[str, SourceTextGeometry] = {
            self.family_file_path: SourceTextGeometry(
                snapshot.sources_by_file_path[self.family_file_path]
            )
        }
        family_geometry = geometries[self.family_file_path]
        family_block = self._render_family_classes()
        insert_offset, _insert_end, _insert_indent = _statement_full_line_span(
            family_geometry, self.insert_before_stmt
        )

        replacements_by_file: dict[str, list[SourceTextSpanReplacement]] = {
            self.family_file_path: [
                SourceTextSpanReplacement(
                    start_offset=insert_offset,
                    end_offset=insert_offset,
                    replacement_source=family_block,
                )
            ]
        }

        absent_name = f"Absent{self.family_name}"
        held_name = f"Held{self.family_name}"
        for member in self.members:
            geom = geometries.get(member.file_path)
            if geom is None:
                geom = SourceTextGeometry(snapshot.sources_by_file_path[member.file_path])
                geometries[member.file_path] = geom
            replacements_by_file.setdefault(member.file_path, []).extend(
                member.span_replacements(
                    geom, self.family_name, absent_name, held_name
                )
            )

        for file_path, rep in self.pushed_loop_plan.call_site_replacements(
            snapshot, self.method_name
        ):
            replacements_by_file.setdefault(file_path, []).append(rep)

        extra_edits: list[NominalSourceEdit] = []
        bindings = snapshot.module_binding_proof
        required_imports: list[tuple[str, tuple[str, ...], str]] = [
            ("ABC", ("abc.ABC",), "from abc import ABC"),
            (
                "dataclass",
                ("dataclasses.dataclass",),
                "from dataclasses import dataclass",
            ),
        ]
        if self.needs_iter:
            required_imports.append(
                (
                    "Iterator",
                    ("collections.abc.Iterator", "typing.Iterator"),
                    "from collections.abc import Iterator",
                )
            )
        for name, expected_qnames, import_src in required_imports:
            witness = bindings.reference_or_builtin_witness_at(
                family_module,
                ast.Name(id=name, ctx=ast.Load()),
                line=self.insert_before_stmt.lineno,
            )
            if witness is None:
                extra_edits.append(
                    ModuleImportMutation.from_source(
                        file_path=self.family_file_path,
                        import_source=import_src,
                        rationale=rationale,
                    )
                )
            elif witness.qualified_name not in expected_qnames:
                raise NeedsInput(
                    f"`{name}` is bound to `{witness.qualified_name}` in `{self.family_file_path}`",
                    NeededInput.SOURCE_CHANGE,
                )

        for ext_file, import_src in self.cross_module_imports:
            extra_edits.append(
                ModuleImportMutation.from_source(
                    file_path=ext_file,
                    import_source=import_src,
                    rationale=rationale,
                )
            )

        span_edits = tuple(
            SourceTextGeometry(snapshot.sources_by_file_path[file_path]).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    sorted(reps, key=lambda item: (item.start_offset, item.end_offset))
                ),
                rationale=rationale,
            )
            for file_path, reps in replacements_by_file.items()
        )
        return (*extra_edits, *span_edits)


@dataclass(frozen=True, kw_only=True)
class ZeroOneTupleToFamilyOperation(RepositorySourceReprovedOperation):
    """Replace a 0/1 tuple returning function with a Held/Absent nominal family."""

    family_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    family_docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    method_name: str = codemod_payload_field(
        RequiredStringPayloadValueCodec(), default="apply"
    )
    method_docstring: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )

    def __post_init__(self) -> None:
        for label, ident in (
            ("family_name", self.family_name),
            ("method_name", self.method_name),
        ):
            if (
                not ident.isidentifier()
                or keyword_module.iskeyword(ident)
                or (ident.startswith("__") and ident.endswith("__"))
            ):
                raise ValueError(f"{label} `{ident}` must be a valid non-dunder identifier")
        if not self.family_docstring.strip():
            raise ValueError("family_docstring must be non-empty")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        proof = ZeroOneTupleProof.proved(
            snapshot,
            self.target,
            self.family_name,
            self.family_docstring,
            self.method_name,
            self.method_docstring,
        )
        return proof.edits(snapshot, self.rationale_text(proof.rationale()))


class ZeroOneTupleFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    PrewarmTypeCheckerMixin,
    FindingRecipeSynthesizer,
    ZeroOneTupleFamilyConcept,
    ABC,
):
    """Turn a proved 0/1 tuple finding into an owner-picked Held/Absent family codemod."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not finding.evidence:
            return self.rejected_evaluation(
                "0/1 tuple refactoring requires source context and finding evidence"
            )
        if not isinstance(finding.metrics, ZeroOneTupleMetrics):
            return self.rejected_evaluation(
                "0/1 tuple finding is missing structured metrics"
            )
        metrics = finding.metrics
        snapshot = context.execution_snapshot()
        primary = finding.evidence[0]
        target = SourceRewriteTarget(
            file_path=primary.file_path, qualname=primary.symbol
        )
        pick = metrics.picked_base
        if pick is None:
            try:
                probe_proof = ZeroOneTupleProof.proved(
                    snapshot,
                    target,
                    "_NraZeroOneTupleFamily",
                    "Nominal held/absent family.",
                    "apply",
                )
            except ValueError as error:
                return self.stopped_evaluation(error)
            loop_hint = probe_proof.pushed_loop_plan.owner_pick_hint()
            return self.stopped_evaluation(
                NeedsInput(
                    f"0/1 tuple return in `{metrics.function_symbol}` can become a held/absent family; "
                    f"supply `owner` `{metrics.function_symbol}` with `name` (family base class) "
                    f"and `judgement` (one-line docstring){loop_hint} in `--codemod-owner-picks`",
                    NeededInput.OWNER_PICK,
                )
            )
        family_name = pick.name
        family_docstring = pick.judgement
        method_name = pick.hoisted_name or "apply"
        method_docstring = pick.hoisted_judgement
        try:
            proof = ZeroOneTupleProof.proved(
                snapshot,
                target,
                family_name,
                family_docstring,
                method_name,
                method_docstring,
            )
            operation = ZeroOneTupleToFamilyOperation(
                target=target,
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
    "ZeroOneTupleFindingRecipeSynthesizer",
    "ZeroOneTupleProof",
    "ZeroOneTupleToFamilyOperation",
)
