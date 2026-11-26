"""Hoist shared leading or trailing steps from sibling overrides into an ancestor template method.

When two or more independent subclasses override an inherited method differently
and share identical leading and/or trailing statements containing at least one
restated receiver step, hoisting those statements onto the declaring ancestor's
method turns the ancestor into the template method and rewrites each sibling to
implement only its differing hook body (or deletes overrides whose body becomes
empty).
"""

from __future__ import annotations

import ast
import keyword as keyword_module
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.ast_field_decoders import (
    declared_annotations,
    kwarg_parameters,
    optional_values,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.class_index import (
    PER_CLASS_NAMES,
    ClassBodyDeclarationKind,
    ClassBodyModuleScope,
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
    reads_class_cell_or_private_name,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.class_method_facets import CompactClassMethod
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.class_syntax_facets import _trim_leading_docstring
from nominal_refactor_advisor.codemod_declaration_source import ClassBodySourceAuthority
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
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
from nominal_refactor_advisor.codemod_source_edits import (
    Include,
    NominalSourceEdit,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import SiblingOverrideStepConcept
from nominal_refactor_advisor.refactors.class_creation import (
    MroWalk,
    declared_names,
)
from nominal_refactor_advisor.refactors.class_overlap import (
    ClassReferenceSites,
    OverlapClass,
    obsolete_import_edits,
    require_static_c3,
    required_name_imports,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor.refactors.restated_overrides import (
    MemberDeletionCreationProof,
    NamespaceReaders,
    RestatedOverrideDeletion,
    next_binder,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

_ALLOWED_SIBLING_DECORATORS = frozenset(("override",))
_ABSTRACT_DECORATORS = frozenset(("abstractmethod",))
_KIND_DECORATORS = frozenset(("classmethod", "property", "cached_property"))


@dataclass(frozen=True)
class OmittedStep:
    """A call every overriding class makes but one, which reads the member it is made on."""

    call: str
    omitting: CompactIndexedClass


@dataclass(frozen=True)
class SiblingOverrideStepCandidate:
    """One inherited method, the classes overriding it differently, and the steps they share."""

    declaring: CompactIndexedClass
    """The nearest class every one of them inherits from that declares the method."""
    name: str
    overriding: tuple[tuple[CompactIndexedClass, int], ...]
    """Each overriding class, with the line of its override."""
    restated: tuple[str, ...]
    """The calls through the receiver that every override makes, by member path."""
    omitted: tuple[OmittedStep, ...]

    @property
    def unanswered(self) -> bool:
        """Whether the overridden declaration is an abstract hook: the ancestor runs no
        sequence of its own, so every step of it is written in the overriding classes."""

        return self.name in self.declaring.abstract_method_names

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(indexed.file_path, line, f"{indexed.qualname}.{self.name}")
            for indexed, line in self.overriding
        )


def _made_by_every_other(
    steps: dict[str, frozenset[str]], symbol: str
) -> frozenset[str]:
    """The calls every override but `symbol`'s makes and `symbol`'s does not."""

    others = [calls for other, calls in steps.items() if other != symbol]
    return frozenset.intersection(*others) - steps[symbol]


def _members_read(
    index,
    methods: dict[str, dict[str, CompactClassMethod]],
    symbol: str,
    read: dict[str, frozenset[str]],
) -> frozenset[str]:
    """The members of the receiver that some method of the class, or of a scanned
    ancestor, reads: what the class has to call a step on. Kept in `read` by class."""

    if symbol not in read:
        read[symbol] = frozenset(
            member
            for owner in (symbol, *index.ancestor_symbols(symbol))
            for method in methods.get(owner, {}).values()
            for member in method.receiver_steps.members
        )
    return read[symbol]


@dataclass(frozen=True)
class _SiblingOverrideStepsBaseline:
    by_path: dict[str, CompactModuleClassProjection]
    class_index: CompactClassFamilyIndex
    methods_by_symbol: dict[str, dict[str, CompactClassMethod]]
    overriding_by_symbol: dict[str, tuple[tuple[str, str], ...]]
    candidates_by_declaring: dict[
        str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideStepCandidate], ...]
    ]
    declared_names: frozenset[str]


@ScanCache.cached
def _base_sibling_override_steps_slots() -> dict[bool, _SiblingOverrideStepsBaseline]:
    return {}


def _symbol_step_overrides(
    verdicts: ClassOverlapVerdicts,
    methods: dict[str, dict[str, CompactClassMethod]],
    symbol: str,
) -> tuple[tuple[str, str], ...]:
    index, bodies = verdicts.class_index, verdicts.bodies
    mine = methods.get(symbol)
    if not mine:
        return ()
    indexed_self = index.class_for(symbol)
    if indexed_self is None:
        return ()
    ancestors = tuple(
        ancestor
        for ancestor in index.ancestor_symbols(symbol)
        if ancestor in bodies.declarations_by_symbol
    )
    if not ancestors:
        return ()
    ancestor_decls = tuple(
        (ancestor, bodies.declarations_of(ancestor)) for ancestor in ancestors
    )
    abstract = indexed_self.abstract_method_names
    emitted: list[tuple[str, str]] = []
    for name, declaration in bodies.declarations_of(symbol).items():
        if (
            name in PER_CLASS_NAMES
            or name in abstract
            or name not in mine
            or not declaration.kind.is_method
        ):
            continue
        for ancestor, decls in ancestor_decls:
            if name in decls:
                emitted.append((ancestor, name))
    return tuple(emitted)


def _step_candidates_for_declaring(
    verdicts: ClassOverlapVerdicts,
    methods: dict[str, dict[str, CompactClassMethod]],
    declared: frozenset[str],
    read: dict[str, frozenset[str]],
    declaring: str,
    by_name: dict[str, list[str]],
) -> tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideStepCandidate], ...]:
    index, bodies = verdicts.class_index, verdicts.bodies
    declaring_class = index.classes_by_symbol[declaring]
    emitted: list[tuple[tuple[str, frozenset[str]], SiblingOverrideStepCandidate]] = []
    for name, symbols in by_name.items():
        symbol_set = frozenset(symbols)
        independent = [
            symbol
            for symbol in symbols
            if symbol_set.isdisjoint(index.ancestor_symbols(symbol))
        ]
        if len(independent) < 2:
            continue
        declarations = [bodies.declarations_of(symbol)[name] for symbol in independent]
        if all(
            declaration.same_declaration_as(declarations[0])
            for declaration in declarations[1:]
        ):
            continue  # one declaration: `SiblingOverrideDetector`'s finding
        steps = {
            symbol: frozenset(methods[symbol][name].receiver_steps.calls) - {name}
            for symbol in independent
        }
        restated = frozenset.intersection(*steps.values())
        omitted = (
            tuple(
                OmittedStep(call, index.classes_by_symbol[symbol])
                for symbol in sorted(independent)
                for call in sorted(_made_by_every_other(steps, symbol))
                if "." in call
                and call.rsplit(".", 1)[1] in declared
                and (member := call.split(".", 1)[0])
                not in {made.split(".", 1)[0] for made in steps[symbol]}
                and member in _members_read(index, methods, symbol, read)
            )
            if restated or len(independent) >= 3
            else ()
        )
        if not restated and not omitted:
            continue
        emitted.append(
            (
                (name, frozenset(independent)),
                SiblingOverrideStepCandidate(
                    declaring=declaring_class,
                    name=name,
                    overriding=tuple(
                        sorted(
                            (
                                (index.classes_by_symbol[symbol], methods[symbol][name].line)
                                for symbol in independent
                            ),
                            key=lambda found: (found[0].file_path, found[1]),
                        )
                    ),
                    restated=tuple(sorted(restated)),
                    omitted=omitted,
                ),
            )
        )
    return tuple(emitted)


def _merge_nearest_step_candidates(
    index: CompactClassFamilyIndex,
    candidates_by_declaring: dict[
        str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideStepCandidate], ...]
    ],
) -> tuple[SiblingOverrideStepCandidate, ...]:
    nearest: dict[tuple[str, frozenset[str]], SiblingOverrideStepCandidate] = {}
    for declaring, entries in candidates_by_declaring.items():
        declaring_ancestors: tuple[str, ...] | None = None
        for key, candidate in entries:
            held = nearest.get(key)
            if held is None:
                nearest[key] = candidate
            else:
                if declaring_ancestors is None:
                    declaring_ancestors = index.ancestor_symbols(declaring)
                if held.declaring.symbol in declaring_ancestors:
                    nearest[key] = candidate
    return tuple(
        sorted(
            nearest.values(),
            key=lambda candidate: (
                candidate.evidence[0].file_path,
                candidate.evidence[0].line,
                candidate.name,
            ),
        )
    )


def sibling_override_step_candidates(
    verdicts: ClassOverlapVerdicts,
    projections: Sequence[CompactModuleClassProjection],
) -> tuple[SiblingOverrideStepCandidate, ...]:
    """Every inherited method overridden differently by independent siblings with a shared
    or omitted receiver step."""

    index, bodies = verdicts.class_index, verdicts.bodies
    cur_projs = tuple(projections)
    settled_key = not any(p.has_unsettled_field_declarations for p in cur_projs)
    if ScanCache.active() is not None:
        slots = _base_sibling_override_steps_slots()
        base = slots.get(settled_key)
        if base is not None:
            base_by_path = base.by_path
            cur_by_path = {p.file_path: p for p in cur_projs}
            changed = tuple(
                p for p in cur_projs if base_by_path.get(p.file_path) is not p
            )
            removed = tuple(
                p for path, p in base_by_path.items() if path not in cur_by_path
            )
            if not changed and not removed:
                return _merge_nearest_step_candidates(index, base.candidates_by_declaring)
            if len(changed) + len(removed) <= 64:
                methods: dict[str, dict[str, CompactClassMethod]] = dict(
                    base.methods_by_symbol
                )
                affected_direct: set[str] = set()
                for p in removed:
                    for c in p.classes:
                        affected_direct.add(c.symbol)
                        methods.pop(c.symbol, None)
                    for m in p.class_methods:
                        affected_direct.add(m.class_symbol)
                        methods.pop(m.class_symbol, None)
                for p in changed:
                    old_p = base_by_path.get(p.file_path)
                    if old_p is not None:
                        for c in old_p.classes:
                            affected_direct.add(c.symbol)
                            methods.pop(c.symbol, None)
                        for m in old_p.class_methods:
                            affected_direct.add(m.class_symbol)
                            methods.pop(m.class_symbol, None)
                    for c in p.classes:
                        affected_direct.add(c.symbol)
                    for m in p.class_methods:
                        affected_direct.add(m.class_symbol)
                        methods.setdefault(m.class_symbol, {})[m.method_name] = m
                declared = frozenset(name for owned in methods.values() for name in owned)
                affected_symbols: set[str] = set(affected_direct)
                for sym in affected_direct:
                    affected_symbols.update(base.class_index.descendant_symbols(sym))
                    affected_symbols.update(index.descendant_symbols(sym))
                overriding_by_symbol = dict(base.overriding_by_symbol)
                dirty_declaring: set[str] = set()
                if declared != base.declared_names:
                    dirty_declaring.update(base.candidates_by_declaring)
                for sym in affected_symbols:
                    old_pairs = overriding_by_symbol.pop(sym, ())
                    for anc, _name in old_pairs:
                        dirty_declaring.add(anc)
                    if sym in bodies.declarations_by_symbol:
                        new_pairs = _symbol_step_overrides(verdicts, methods, sym)
                        if new_pairs:
                            overriding_by_symbol[sym] = new_pairs
                            for anc, _name in new_pairs:
                                dirty_declaring.add(anc)
                dirty_declaring.update(affected_symbols)
                # Also mark declaring ancestors whose candidates contain any affected symbol
                # (in case _members_read changed for a descendant without changing overriding pairs).
                for sym in affected_symbols:
                    for anc in index.ancestor_symbols(sym):
                        if anc in base.candidates_by_declaring:
                            dirty_declaring.add(anc)
                dirty_by_declaring: dict[str, dict[str, list[str]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                for sym, pairs in overriding_by_symbol.items():
                    for anc, name in pairs:
                        if anc in dirty_declaring:
                            dirty_by_declaring[anc][name].append(sym)
                candidates_by_declaring = dict(base.candidates_by_declaring)
                read: dict[str, frozenset[str]] = {}
                for declaring in dirty_declaring:
                    by_name = dirty_by_declaring.get(declaring)
                    if by_name and declaring in index.classes_by_symbol:
                        cands = _step_candidates_for_declaring(
                            verdicts, methods, declared, read, declaring, by_name
                        )
                        if cands:
                            candidates_by_declaring[declaring] = cands
                        else:
                            candidates_by_declaring.pop(declaring, None)
                    else:
                        candidates_by_declaring.pop(declaring, None)
                return _merge_nearest_step_candidates(index, candidates_by_declaring)

    methods_full: dict[str, dict[str, CompactClassMethod]] = defaultdict(dict)
    for projection in cur_projs:
        for method in projection.class_methods:
            methods_full[method.class_symbol][method.method_name] = method
    declared_full = frozenset(name for owned in methods_full.values() for name in owned)
    overriding_by_sym_full: dict[str, tuple[tuple[str, str], ...]] = {}
    by_declaring_full: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for symbol in bodies.declarations_by_symbol:
        pairs = _symbol_step_overrides(verdicts, methods_full, symbol)
        if pairs:
            overriding_by_sym_full[symbol] = pairs
            for anc, name in pairs:
                by_declaring_full[anc][name].append(symbol)
    read_full: dict[str, frozenset[str]] = {}
    candidates_by_decl_full: dict[
        str, tuple[tuple[tuple[str, frozenset[str]], SiblingOverrideStepCandidate], ...]
    ] = {}
    for declaring, by_name in by_declaring_full.items():
        cands = _step_candidates_for_declaring(
            verdicts, methods_full, declared_full, read_full, declaring, by_name
        )
        if cands:
            candidates_by_decl_full[declaring] = cands
    if ScanCache.active() is not None and len(cur_projs) >= 20:
        slots = _base_sibling_override_steps_slots()
        if settled_key not in slots:
            slots[settled_key] = _SiblingOverrideStepsBaseline(
                by_path={p.file_path: p for p in cur_projs},
                class_index=index,
                methods_by_symbol=dict(methods_full),
                overriding_by_symbol=overriding_by_sym_full,
                candidates_by_declaring=candidates_by_decl_full,
                declared_names=declared_full,
            )
    return _merge_nearest_step_candidates(index, candidates_by_decl_full)


def _decorator_terminal_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _statement_source_block(
    source_lines: Sequence[str], statement: ast.stmt, indent: str
) -> str:
    """Extract `statement`'s exact source lines, dedented and re-indented to `indent`."""

    start_line = statement.lineno
    if (
        isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
        and statement.decorator_list
    ):
        start_line = min(
            start_line, *(dec.lineno for dec in statement.decorator_list)
        )
    end_line = statement.end_lineno or statement.lineno
    raw = "".join(source_lines[start_line - 1 : end_line])
    dedented = textwrap.dedent(raw)
    rendered_lines = [
        f"{indent}{line}" if line.strip() else line
        for line in dedented.splitlines(keepends=True)
    ]
    text = "".join(rendered_lines)
    return text if text.endswith("\n") else f"{text}\n"


def _leading_docstring_stmt(method: FunctionNode) -> ast.Expr | None:
    if not method.body:
        return None
    match method.body[0]:
        case ast.Expr(value=ast.Constant(value=str())) as doc_stmt:
            return doc_stmt
    return None


@dataclass(frozen=True)
class ExtraHookParams:
    """Extra parameters forwarded from hoisted template setup into the subclass hook."""

    names: tuple[str, ...] = ()
    keyword_only: bool = False
    annotations: tuple[tuple[str, ast.expr], ...] = ()

    @classmethod
    def for_method(
        cls,
        method: FunctionNode,
        names: Sequence[str],
        annotations: Mapping[str, ast.expr] = (),  # type: ignore[assignment]
    ) -> ExtraHookParams | None:
        ordered = tuple(dict.fromkeys(names))
        if not ordered:
            return cls()
        existing = _all_parameter_names(method)
        for name in ordered:
            if (
                not name.isidentifier()
                or keyword_module.iskeyword(name)
                or name in existing
            ):
                return None
        args = method.args
        keyword_only = bool(args.defaults or args.vararg or args.kwonlyargs)
        if keyword_only and kwarg_parameters(args):
            return None
        ann_map = dict(annotations) if annotations else {}
        ann_pairs = tuple((n, ann_map[n]) for n in ordered if n in ann_map)
        return cls(names=ordered, keyword_only=keyword_only, annotations=ann_pairs)

    def with_arguments(self, args: ast.arguments) -> ast.arguments:
        if not self.names:
            return args
        ann_by_name = dict(self.annotations)
        added = [ast.arg(arg=name, annotation=ann_by_name.get(name)) for name in self.names]
        if self.keyword_only:
            return ast.arguments(
                posonlyargs=list(args.posonlyargs),
                args=list(args.args),
                vararg=args.vararg,
                kwonlyargs=[*args.kwonlyargs, *added],
                kw_defaults=[*args.kw_defaults, *(None for _ in added)],
                kwarg=args.kwarg,
                defaults=list(args.defaults),
            )
        return ast.arguments(
            posonlyargs=list(args.posonlyargs),
            args=[*args.args, *added],
            vararg=args.vararg,
            kwonlyargs=list(args.kwonlyargs),
            kw_defaults=list(args.kw_defaults),
            kwarg=args.kwarg,
            defaults=list(args.defaults),
        )

    def forwarded_call_args(self) -> tuple[str, ...]:
        if self.keyword_only:
            return tuple(f"{name}={name}" for name in self.names)
        return self.names


def _method_header_signature(
    method: FunctionNode,
    name: str,
    source_lines: Sequence[str] | None = None,
    *,
    extra_params: ExtraHookParams = ExtraHookParams(),
    preserve_return_annotation: bool = True,
) -> str:
    """Render `def <name>(...) -> Ret:`, preserving original source spacing when available."""

    if (
        not extra_params.names
        and preserve_return_annotation
        and source_lines is not None
        and method.body
    ):
        first_body = method.body[0]
        body_start_line = first_body.lineno
        if (
            isinstance(
                first_body, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            )
            and first_body.decorator_list
        ):
            body_start_line = min(
                body_start_line, *(dec.lineno for dec in first_body.decorator_list)
            )
        if body_start_line > method.lineno:
            header_raw = "".join(source_lines[method.lineno - 1 : body_start_line - 1])
            header_stripped = textwrap.dedent(header_raw).rstrip()
            if header_stripped.endswith(":"):
                marker = f"def {method.name}"
                if marker in header_stripped:
                    return header_stripped.replace(marker, f"def {name}", 1)
    prefix = "async def" if isinstance(method, ast.AsyncFunctionDef) else "def"
    args_src = ast.unparse(extra_params.with_arguments(method.args))
    returns_annotations = (
        tuple(declared_annotations(method)) if preserve_return_annotation else ()
    )
    returns_src = (
        f" -> {ast.unparse(returns_annotations[0])}" if returns_annotations else ""
    )
    return f"{prefix} {name}({args_src}){returns_src}:"


class MethodKind(ABC):
    """How the hoisted method and its hook are decorated (plain, `@classmethod`, `@property`, `@cached_property`)."""

    def template_decorators(self, method: FunctionNode) -> tuple[str, ...]:
        return tuple(
            ast.unparse(dec)
            for dec in method.decorator_list
            if _decorator_terminal_name(dec) not in _ABSTRACT_DECORATORS
        )

    def hook_decorators(self, abstract_decorators: tuple[str, ...]) -> tuple[str, ...]:
        return abstract_decorators

    def sibling_hook_decorators(self, method: FunctionNode) -> tuple[str, ...]:
        return tuple(ast.unparse(dec) for dec in method.decorator_list)

    @classmethod
    def of(
        cls,
        owner: OverlapClass,
        method: FunctionNode,
        *,
        allow_abstract: bool,
    ) -> MethodKind:
        allowed = (
            _ALLOWED_SIBLING_DECORATORS | _ABSTRACT_DECORATORS | _KIND_DECORATORS
            if allow_abstract
            else _ALLOWED_SIBLING_DECORATORS | _KIND_DECORATORS
        )
        found_kinds: list[str] = []
        for decorator in method.decorator_list:
            name = _decorator_terminal_name(decorator)
            if name not in allowed:
                raise NeedsInput(
                    f"`{owner.qualname}.{method.name}` has decorator `@{ast.unparse(decorator)}`",
                    NeededInput.PROOF,
                )
            if name in _KIND_DECORATORS:
                found_kinds.append(name)
        if len(found_kinds) > 1:
            raise NeedsInput(
                f"`{owner.qualname}.{method.name}` combines multiple method descriptors",
                NeededInput.PROOF,
            )
        if found_kinds == ["classmethod"]:
            return ClassMethodKind()
        if found_kinds == ["property"]:
            return PropertyMethodKind()
        if found_kinds == ["cached_property"]:
            return CachedPropertyMethodKind()
        return PlainMethodKind()

    def is_cached_property(self) -> bool:
        """Whether this method kind is a cached property."""
        return False


@dataclass(frozen=True)
class PlainMethodKind(MethodKind):
    """Plain instance method with no descriptor decorator."""


@dataclass(frozen=True)
class ClassMethodKind(MethodKind):
    """`@classmethod` method whose hook is also a `@classmethod`."""

    def hook_decorators(self, abstract_decorators: tuple[str, ...]) -> tuple[str, ...]:
        return ("classmethod", *abstract_decorators)


@dataclass(frozen=True)
class PropertyMethodKind(MethodKind):
    """`@property` getter whose per-subclass hook is a plain helper method."""

    def sibling_hook_decorators(self, method: FunctionNode) -> tuple[str, ...]:
        return tuple(
            ast.unparse(dec)
            for dec in method.decorator_list
            if _decorator_terminal_name(dec) != "property"
        )


@dataclass(frozen=True)
class CachedPropertyMethodKind(MethodKind):
    """`@cached_property` getter whose per-subclass hook is a plain helper method."""

    def template_decorators(self, method: FunctionNode) -> tuple[str, ...]:
        kept = tuple(
            ast.unparse(dec)
            for dec in method.decorator_list
            if _decorator_terminal_name(dec) not in (_ABSTRACT_DECORATORS | {"property"})
        )
        if any(
            _decorator_terminal_name(dec) == "cached_property"
            for dec in method.decorator_list
        ):
            return kept
        return ("cached_property", *kept)

    def sibling_hook_decorators(self, method: FunctionNode) -> tuple[str, ...]:
        return tuple(
            ast.unparse(dec)
            for dec in method.decorator_list
            if _decorator_terminal_name(dec) != "cached_property"
        )

    def is_cached_property(self) -> bool:
        return True


def _forwarded_hook_call(
    method: FunctionNode,
    hook_name: str,
    *,
    extra_params: ExtraHookParams = ExtraHookParams(),
) -> str:
    """Render `[await ]<receiver>.<hook_name>(<forwarded arguments>)`."""

    args = method.args
    positional = (*args.posonlyargs, *args.args)
    receiver = positional[0].arg
    call_args: list[str] = [param.arg for param in positional[1:]]
    if not extra_params.keyword_only:
        call_args.extend(extra_params.forwarded_call_args())
    match args:
        case ast.arguments(vararg=ast.arg(arg=vararg_name)):
            call_args.append(f"*{vararg_name}")
    for kw_param in args.kwonlyargs:
        call_args.append(f"{kw_param.arg}={kw_param.arg}")
    if extra_params.keyword_only:
        call_args.extend(extra_params.forwarded_call_args())
    match args:
        case ast.arguments(kwarg=ast.arg(arg=kwarg_name)):
            call_args.append(f"**{kwarg_name}")
    call = f"{receiver}.{hook_name}({', '.join(call_args)})"
    return f"await {call}" if isinstance(method, ast.AsyncFunctionDef) else call


def _parameter_signature_key(method: FunctionNode) -> tuple[object, ...]:
    """Structural parameter identity (names, kinds, defaults) ignoring annotations."""

    args = method.args
    vararg_key: tuple[str, ...] = ()
    kwarg_key: tuple[str, ...] = ()
    match args:
        case ast.arguments(vararg=ast.arg(arg=vararg_name)):
            vararg_key = (vararg_name,)
    match args:
        case ast.arguments(kwarg=ast.arg(arg=kwarg_name)):
            kwarg_key = (kwarg_name,)
    return (
        type(method),
        tuple(p.arg for p in args.posonlyargs),
        tuple(p.arg for p in args.args),
        vararg_key,
        tuple(p.arg for p in args.kwonlyargs),
        kwarg_key,
        tuple(ast.dump(d, include_attributes=False) for d in args.defaults),
        tuple(
            None if d is None else ast.dump(d, include_attributes=False)
            for d in args.kw_defaults
        ),
    )


def _all_parameter_names(method: FunctionNode) -> frozenset[str]:
    args = method.args
    return frozenset(
        p.arg
        for p in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *variadic_parameters(args),
        )
    )


def _contains_node_type(
    statements: Sequence[ast.stmt], node_types: tuple[type[ast.AST], ...]
) -> bool:
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, node_types):
                return True
    return False


def _has_return_statement(statements: Sequence[ast.stmt]) -> bool:
    return any(
        isinstance(node, ast.Return)
        for statement in statements
        for node in ast.walk(statement)
    )


def _read_names(statements: Sequence[ast.stmt]) -> frozenset[str]:
    return frozenset(
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _receiver_call_path(callee: ast.expr, receiver_name: str) -> str | None:
    parts: list[str] = []
    current = callee
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not parts:
        return None
    match current:
        case ast.Name(id=name) if name == receiver_name:
            return ".".join(reversed(parts))
    return None


def _statement_calls_any_step(
    statements: Sequence[ast.stmt],
    receiver_name: str,
    restated_calls: frozenset[str],
) -> bool:
    for statement in statements:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and _receiver_call_path(node.func, receiver_name) in restated_calls
            ):
                return True
    return False


def _returns_none_annotation(method: FunctionNode) -> bool:
    for annotation in declared_annotations(method):
        match annotation:
            case ast.Constant(value=None):
                return True
    return False


def _calls_super_member(node: ast.AST, member_name: str) -> bool:
    for subnode in ast.walk(node):
        match subnode:
            case ast.Attribute(
                value=ast.Call(func=ast.Name(id="super")),
                attr=attr,
            ) if attr == member_name:
                return True
    return False


def _has_non_none_return(statements: Sequence[ast.stmt]) -> bool:
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Return) and optional_values(node):
                return True
    return False


def _ordered_definite_bindings(
    statements: Sequence[ast.stmt],
) -> tuple[tuple[str, ...], dict[str, ast.expr]]:
    """Names definitely assigned at the top level of `statements`, in order, with
    any explicit `AnnAssign` annotations."""

    names: list[str] = []
    annotations: dict[str, ast.expr] = {}

    def record_target(target: ast.expr) -> None:
        match target:
            case ast.Name(id=bound):
                if bound not in names:
                    names.append(bound)
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for elt in elts:
                    record_target(elt)
            case ast.Starred(value=inner):
                record_target(inner)

    for statement in statements:
        match statement:
            case ast.Assign(targets=targets):
                for target in targets:
                    record_target(target)
            case ast.AnnAssign(target=ast.Name(id=bound), value=value, annotation=ann) if (
                value is not None
            ):
                if bound not in names:
                    names.append(bound)
                annotations[bound] = ann
            case ast.AugAssign(target=ast.Name(id=bound)):
                if bound not in names:
                    names.append(bound)
    return tuple(names), annotations


def _leading_unconditional_step_expr(
    statement: ast.stmt,
    receiver_name: str,
    restated_calls: frozenset[str],
) -> ast.expr | None:
    """First `self.<step>(...)` (or `await self.<step>(...)`) evaluated unconditionally
    at the start of `statement` before any impure call or short-circuit barrier."""

    found: ast.expr | None = None
    barrier_hit = False

    def walk(node: ast.AST | None) -> None:
        nonlocal found, barrier_hit
        if barrier_hit or node is None:
            return
        call_node: ast.Call | None = None
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            call_node = node.value
        elif isinstance(node, ast.Call):
            call_node = node
        if (
            call_node is not None
            and _receiver_call_path(call_node.func, receiver_name) in restated_calls
        ):
            found = node  # type: ignore[assignment]
            barrier_hit = True
            return
        match node:
            case ast.Constant() | ast.Name():
                return
            case ast.Attribute(value=inner):
                walk(inner)
            case ast.Subscript(value=value, slice=slice_expr):
                walk(value)
                walk(slice_expr)
                barrier_hit = True
            case ast.Slice(lower=lower, upper=upper, step=step):
                walk(lower)
                walk(upper)
                walk(step)
            case ast.UnaryOp(operand=operand):
                walk(operand)
            case ast.BinOp(left=left, op=op, right=right):
                walk(left)
                if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
                    barrier_hit = True
                    return
                walk(right)
            case ast.Tuple(elts=elts) | ast.List(elts=elts) | ast.Set(elts=elts):
                for elt in elts:
                    walk(elt)
            case ast.Starred(value=inner):
                walk(inner)
            case ast.Dict(keys=keys, values=values):
                for key, value in zip(keys, values, strict=True):
                    if key is not None:
                        walk(key)
                    walk(value)
            case ast.JoinedStr(values=values):
                for part in values:
                    walk(part)
            case ast.FormattedValue(value=value, format_spec=spec):
                walk(value)
                if spec is not None:
                    walk(spec)
            case ast.Call(func=func, args=args, keywords=keywords):
                walk(func)
                for arg in args:
                    walk(arg)
                for kw in keywords:
                    walk(kw.value)
                barrier_hit = True
            case ast.Return(value=value) | ast.Expr(value=value) | ast.Assign(value=value):
                walk(value)
            case ast.AnnAssign(value=value):
                walk(value)
            case ast.Yield(value=value) | ast.YieldFrom(value=value):
                walk(value)
                barrier_hit = True
            case ast.If(test=test) | ast.IfExp(test=test):
                walk(test)
                barrier_hit = True
            case ast.BoolOp(values=values) if values:
                walk(values[0])
                barrier_hit = True
            case ast.Compare(left=left, comparators=comparators):
                walk(left)
                if comparators:
                    walk(comparators[0])
                barrier_hit = True
            case (
                ast.GeneratorExp(generators=generators)
                | ast.ListComp(generators=generators)
                | ast.SetComp(generators=generators)
                | ast.DictComp(generators=generators)
            ) if generators:
                walk(generators[0].iter)
                barrier_hit = True
            case _:
                barrier_hit = True

    walk(statement)
    if found is None:
        return None
    if _contains_node_type(
        (ast.Expr(value=found),),
        (
            ast.Yield,
            ast.YieldFrom,
            ast.NamedExpr,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
        ),
    ):
        return None
    target_dump = ast.dump(found, include_attributes=False)
    occurrences = sum(
        1
        for subnode in ast.walk(statement)
        if isinstance(subnode, type(found))
        and ast.dump(subnode, include_attributes=False) == target_dump
    )
    if occurrences != 1:
        return None
    return found


def _render_statement_with_expr_replaced(
    source: str,
    statement: ast.stmt,
    target_expr: ast.expr,
    replacement_text: str,
    indent: str,
) -> str:
    """Render `statement` with `target_expr`'s exact source span replaced by `replacement_text`."""

    geometry = SourceTextGeometry(source)
    stmt_start, stmt_end = geometry.line_span_offsets(
        statement.lineno, statement.end_lineno or statement.lineno
    )
    expr_start, expr_end = geometry.required_node_offsets(target_expr)
    rel_start = expr_start - stmt_start
    rel_end = expr_end - stmt_start
    raw = source[stmt_start:stmt_end]
    spliced = f"{raw[:rel_start]}{replacement_text}{raw[rel_end:]}"
    dedented = textwrap.dedent(spliced)
    rendered_lines = [
        f"{indent}{line}" if line.strip() else line
        for line in dedented.splitlines(keepends=True)
    ]
    text = "".join(rendered_lines)
    return text if text.endswith("\n") else f"{text}\n"


def _render_expr_source(source: str, expr: ast.expr) -> str:
    geometry = SourceTextGeometry(source)
    start, end = geometry.required_node_offsets(expr)
    return source[start:end]


def _render_safe_expr(source: str, expr: ast.expr) -> str:
    text = _render_expr_source(source, expr)
    return f"({text})" if "\n" in text else text


def _join_boolop_sources(
    source: str, op: ast.boolop, operands: Sequence[ast.expr]
) -> str:
    word = " and " if isinstance(op, ast.And) else " or "
    parts: list[str] = []
    for operand in operands:
        text = _render_expr_source(source, operand)
        if isinstance(
            operand, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)
        ) or "\n" in text:
            text = f"({text})"
        parts.append(text)
    joined = word.join(parts)
    return f"({joined})" if "\n" in joined else joined


def _render_boolop_atom(source: str, operand: ast.expr) -> str:
    text = _render_expr_source(source, operand)
    if isinstance(
        operand, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)
    ) or "\n" in text:
        return f"({text})"
    return text


class HookDeclaration(ABC):
    """How the ancestor declares the per-subclass hook method."""

    body_line: ClassVar[str]

    def abstract_decorators(self) -> tuple[str, ...]:
        return ()

    def obsolete_declaring_decorators(
        self, declaring_method: FunctionNode
    ) -> tuple[ast.expr, ...]:
        return tuple(
            decorator
            for decorator in declaring_method.decorator_list
            if _decorator_terminal_name(decorator) in _ABSTRACT_DECORATORS
        )

    def render(
        self,
        method: FunctionNode,
        hook_name: str,
        indent: str,
        method_kind: MethodKind,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams = ExtraHookParams(),
        preserve_return_annotation: bool = True,
    ) -> str:
        header = _method_header_signature(
            method,
            hook_name,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=preserve_return_annotation,
        )
        indented_header = "\n".join(
            f"{indent}{line}" if line.strip() else line
            for line in header.splitlines()
        )
        dec_prefix = "".join(
            f"{indent}@{dec}\n"
            for dec in method_kind.hook_decorators(self.abstract_decorators())
        )
        return f"{dec_prefix}{indented_header}\n{indent}    {self.body_line}\n"


@dataclass(frozen=True)
class AbstractHookDeclaration(HookDeclaration):
    """All overriding siblings still have a non-empty middle body; keep `@abstractmethod`."""

    body_line: ClassVar[str] = "..."
    decorator_source: str

    def abstract_decorators(self) -> tuple[str, ...]:
        return (self.decorator_source,)

    def obsolete_declaring_decorators(
        self, declaring_method: FunctionNode
    ) -> tuple[ast.expr, ...]:
        del declaring_method
        return ()


@dataclass(frozen=True)
class NoOpHookDeclaration(HookDeclaration):
    """At least one sibling has an empty middle body after hoisting; default hook is `pass`."""

    body_line: ClassVar[str] = "pass"


@dataclass(frozen=True)
class EmptyGeneratorHookDeclaration(HookDeclaration):
    """At least one generator sibling has an empty middle body after hoisting; default hook yields nothing."""

    body_line: ClassVar[str] = "yield from ()"


@dataclass(frozen=True)
class NotImplementedHookDeclaration(HookDeclaration):
    """Every sibling has a non-empty middle body on an ancestor without `@abstractmethod`."""

    body_line: ClassVar[str] = "raise NotImplementedError"


@dataclass(frozen=True)
class HoistedStepExpression:
    """An identical unconditional leading step call hoisted into a local variable on the template."""

    var_name: str
    expr: ast.expr


class SiblingHookBody(ABC):
    """How one sibling's differing middle body renders inside its hook method."""

    @abstractmethod
    def render_header(
        self,
        method: FunctionNode,
        hook_name: str,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str: ...

    @abstractmethod
    def render_body(
        self,
        source: str,
        source_lines: Sequence[str],
        inner_indent: str,
    ) -> list[str]: ...


@dataclass(frozen=True)
class DirectMiddleHookBody(SiblingHookBody):
    """Renders the sibling's remaining middle statements directly."""

    middle: tuple[ast.stmt, ...]
    leading_expr_replacement: tuple[ast.expr, str] | None = None

    def render_header(
        self,
        method: FunctionNode,
        hook_name: str,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return _method_header_signature(
            method,
            hook_name,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=True,
        )

    def render_body(
        self,
        source: str,
        source_lines: Sequence[str],
        inner_indent: str,
    ) -> list[str]:
        pieces: list[str] = []
        for index, statement in enumerate(self.middle):
            if index == 0 and self.leading_expr_replacement is not None:
                target_expr, var_name = self.leading_expr_replacement
                pieces.append(
                    _render_statement_with_expr_replaced(
                        source, statement, target_expr, var_name, inner_indent
                    )
                )
            else:
                pieces.append(
                    _statement_source_block(source_lines, statement, inner_indent)
                )
        return pieces


@dataclass(frozen=True)
class ReturningExprHookBody(SiblingHookBody):
    """Renders optional leading prep statements followed by `return <expr>`."""

    prep: tuple[ast.stmt, ...]
    return_expr: ast.expr

    def render_header(
        self,
        method: FunctionNode,
        hook_name: str,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return _method_header_signature(
            method,
            hook_name,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=False,
        )

    def render_body(
        self,
        source: str,
        source_lines: Sequence[str],
        inner_indent: str,
    ) -> list[str]:
        pieces = [
            _statement_source_block(source_lines, statement, inner_indent)
            for statement in self.prep
        ]
        pieces.append(f"{inner_indent}return {_render_safe_expr(source, self.return_expr)}\n")
        return pieces


@dataclass(frozen=True)
class ReturningTupleHookBody(SiblingHookBody):
    """Renders optional leading prep statements followed by `return (arg0, arg1, ...)`."""

    prep: tuple[ast.stmt, ...]
    args: tuple[ast.expr, ...]

    def render_header(
        self,
        method: FunctionNode,
        hook_name: str,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return _method_header_signature(
            method,
            hook_name,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=False,
        )

    def render_body(
        self,
        source: str,
        source_lines: Sequence[str],
        inner_indent: str,
    ) -> list[str]:
        pieces = [
            _statement_source_block(source_lines, statement, inner_indent)
            for statement in self.prep
        ]
        if not self.args:
            tuple_src = "()"
        elif len(self.args) == 1:
            tuple_src = f"({_render_safe_expr(source, self.args[0])},)"
        else:
            rendered = ", ".join(_render_safe_expr(source, arg) for arg in self.args)
            tuple_src = f"({rendered})"
        pieces.append(f"{inner_indent}return {tuple_src}\n")
        return pieces


@dataclass(frozen=True)
class ReturningBoolOpHookBody(SiblingHookBody):
    """Renders optional leading prep statements followed by `return <operands joined by op>`."""

    prep: tuple[ast.stmt, ...]
    op: ast.boolop
    operands: tuple[ast.expr, ...]

    def render_header(
        self,
        method: FunctionNode,
        hook_name: str,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return _method_header_signature(
            method,
            hook_name,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=False,
        )

    def render_body(
        self,
        source: str,
        source_lines: Sequence[str],
        inner_indent: str,
    ) -> list[str]:
        pieces = [
            _statement_source_block(source_lines, statement, inner_indent)
            for statement in self.prep
        ]
        expr_text = (
            _render_safe_expr(source, self.operands[0])
            if len(self.operands) == 1
            else _join_boolop_sources(source, self.op, self.operands)
        )
        pieces.append(f"{inner_indent}return {expr_text}\n")
        return pieces


class SiblingBodyTemplate(ABC):
    """How the ancestor's template method invokes the sibling hook."""

    def render_hook_declaration(
        self,
        hook_declaration: HookDeclaration,
        declaring_method: FunctionNode,
        hook_name: str,
        indent: str,
        method_kind: MethodKind,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return hook_declaration.render(
            declaring_method,
            hook_name,
            indent,
            method_kind,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=False,
        )

    @abstractmethod
    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str: ...

    @abstractmethod
    def import_checked_statements(self) -> tuple[ast.stmt, ...]: ...


@dataclass(frozen=True)
class DirectHookTemplate(SiblingBodyTemplate):
    """Calls or returns/yields from the hook directly."""

    forward_return: bool
    is_generator: bool

    def render_hook_declaration(
        self,
        hook_declaration: HookDeclaration,
        declaring_method: FunctionNode,
        hook_name: str,
        indent: str,
        method_kind: MethodKind,
        source_lines: Sequence[str],
        *,
        extra_params: ExtraHookParams,
    ) -> str:
        return hook_declaration.render(
            declaring_method,
            hook_name,
            indent,
            method_kind,
            source_lines,
            extra_params=extra_params,
            preserve_return_annotation=True,
        )

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source, hoist_source_lines
        if self.is_generator:
            return (
                f"{inner_indent}return (yield from {hook_call})\n"
                if self.forward_return
                else f"{inner_indent}yield from {hook_call}\n"
            )
        return (
            f"{inner_indent}return {hook_call}\n"
            if self.forward_return
            else f"{inner_indent}{hook_call}\n"
        )

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return ()


@dataclass(frozen=True)
class ReturnCallTemplate(SiblingBodyTemplate):
    """Every sibling ends with the same receiver call differing in a single argument slot."""

    template_stmt: ast.stmt
    template_call: ast.Call
    varying_slot_expr: ast.expr
    import_checked_stmt: ast.stmt
    hoist_var_name: str | None = None

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source_lines
        if self.hoist_var_name is not None:
            rendered_call = _render_statement_with_expr_replaced(
                hoist_source,
                self.template_stmt,
                self.varying_slot_expr,
                self.hoist_var_name,
                inner_indent,
            )
            return f"{inner_indent}{self.hoist_var_name} = {hook_call}\n{rendered_call}"
        return _render_statement_with_expr_replaced(
            hoist_source,
            self.template_stmt,
            self.varying_slot_expr,
            hook_call,
            inner_indent,
        )

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return (self.import_checked_stmt,)


@dataclass(frozen=True)
class VariadicReturnCallTemplate(SiblingBodyTemplate):
    """Every sibling ends with the same receiver call differing in positional arity or
    multiple positional arguments, unpacked via `*self._hook(...)`."""

    template_call: ast.Call
    is_return: bool
    is_await: bool
    fixed_keywords: tuple[ast.keyword, ...]
    import_checked_stmt: ast.stmt
    hoist_var_name: str | None = None

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source_lines
        func_src = _render_expr_source(hoist_source, self.template_call.func)
        star_arg = (
            f"*{self.hoist_var_name}"
            if self.hoist_var_name is not None
            else f"*{hook_call}"
        )
        kw_parts = [
            f"{kw.name}={_render_safe_expr(hoist_source, kw.value)}"
            for kw in CallKeywords.of(self.fixed_keywords).named
        ]
        all_args = ", ".join((star_arg, *kw_parts))
        call_text = f"await {func_src}({all_args})" if self.is_await else f"{func_src}({all_args})"
        stmt_text = f"return {call_text}" if self.is_return else call_text
        if self.hoist_var_name is not None:
            return (
                f"{inner_indent}{self.hoist_var_name} = {hook_call}\n"
                f"{inner_indent}{stmt_text}\n"
            )
        return f"{inner_indent}{stmt_text}\n"

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return (self.import_checked_stmt,)


@dataclass(frozen=True)
class BoolOpTemplate(SiblingBodyTemplate):
    """Every sibling returns a short-circuit `and`/`or` sharing leading or trailing operands."""

    op: ast.boolop
    shared_operands: tuple[ast.expr, ...]
    hook_first: bool

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source_lines
        word = " and " if isinstance(self.op, ast.And) else " or "
        shared_parts = [_render_boolop_atom(hoist_source, expr) for expr in self.shared_operands]
        ordered = [hook_call, *shared_parts] if self.hook_first else [*shared_parts, hook_call]
        joined = word.join(ordered)
        rendered = f"({joined})" if "\n" in joined else joined
        return f"{inner_indent}return {rendered}\n"

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return tuple(ast.Expr(value=expr) for expr in self.shared_operands)


@dataclass(frozen=True)
class TrailingGuardedStepTemplate(SiblingBodyTemplate):
    """Every sibling ends with `if <cond_i>: <identical step body>`."""

    guarded_body: tuple[ast.stmt, ...]

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source
        body_indent = f"{inner_indent}    "
        pieces = [f"{inner_indent}if {hook_call}:\n"]
        for statement in self.guarded_body:
            pieces.append(
                _statement_source_block(hoist_source_lines, statement, body_indent)
            )
        return "".join(pieces)

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return self.guarded_body


@dataclass(frozen=True)
class GuardedEarlyReturnSuffixTemplate(SiblingBodyTemplate):
    """Every sibling has `if <cond_i>: <identical early return>` before an identical trailing step."""

    early_return_body: tuple[ast.stmt, ...]
    shared_tail: tuple[ast.stmt, ...]

    def render_template_call(
        self,
        hoist_source: str,
        hoist_source_lines: Sequence[str],
        hook_call: str,
        inner_indent: str,
    ) -> str:
        del hoist_source
        body_indent = f"{inner_indent}    "
        pieces = [f"{inner_indent}if {hook_call}:\n"]
        for statement in self.early_return_body:
            pieces.append(
                _statement_source_block(hoist_source_lines, statement, body_indent)
            )
        for statement in self.shared_tail:
            pieces.append(
                _statement_source_block(hoist_source_lines, statement, inner_indent)
            )
        return "".join(pieces)

    def import_checked_statements(self) -> tuple[ast.stmt, ...]:
        return (*self.early_return_body, *self.shared_tail)


class SiblingRewrite(ABC):
    """How one overriding sibling method is updated after hoisting shared steps."""

    child: OverlapClass
    method: FunctionNode

    @abstractmethod
    def edits(
        self,
        snapshot: CodemodSourceSnapshot,
        declaring_file_path: str,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class DeleteEmptySiblingOverride(SiblingRewrite):
    """All statements of this sibling override were hoisted; delete the override outright."""

    child: OverlapClass
    method: FunctionNode

    def edits(
        self,
        snapshot: CodemodSourceSnapshot,
        declaring_file_path: str,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        deletion = StatementDeletionSource(
            source=snapshot.sources_by_file_path[self.child.file_path],
            node=self.child.node,
            file_path=self.child.file_path,
        )
        result: list[NominalSourceEdit] = list(
            deletion.physical_edits(
                file_path=self.child.file_path,
                replacements=deletion.replacements_for_statements((self.method,)),
                rationale=rationale,
            )
        )
        if self.child.file_path != declaring_file_path:
            result.extend(
                obsolete_import_edits(
                    snapshot, self.child.file_path, (self.method,), rationale
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class RewriteSiblingToHook(SiblingRewrite):
    """Rename this sibling override to the hook and keep only its unique middle statements."""

    child: OverlapClass
    method: FunctionNode
    hook_body: SiblingHookBody
    dropped: tuple[ast.stmt, ...]
    hook_name: str
    method_kind: MethodKind
    extra_params: ExtraHookParams = ExtraHookParams()

    def edits(
        self,
        snapshot: CodemodSourceSnapshot,
        declaring_file_path: str,
        rationale: str,
    ) -> tuple[NominalSourceEdit, ...]:
        source = snapshot.sources_by_file_path[self.child.file_path]
        source_lines = source.splitlines(keepends=True)
        body = ClassBodySourceAuthority(self.child.node, source)
        indent = body.indentation
        inner_indent = f"{indent}    "
        start, end = body.geometry.node_span_offsets(
            SourceNodeSpan(self.method, Include)
        )
        pieces: list[str] = [
            f"{indent}@{dec}\n"
            for dec in self.method_kind.sibling_hook_decorators(self.method)
        ]
        header = self.hook_body.render_header(
            self.method,
            self.hook_name,
            source_lines,
            extra_params=self.extra_params,
        )
        for line in header.splitlines():
            pieces.append(f"{indent}{line}\n" if line.strip() else "\n")
        docstring = _leading_docstring_stmt(self.method)
        if docstring is not None:
            pieces.append(_statement_source_block(source_lines, docstring, inner_indent))
        pieces.extend(self.hook_body.render_body(source, source_lines, inner_indent))
        replacement = SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source="".join(pieces),
        )
        result: list[NominalSourceEdit] = list(
            body.geometry.physical_edits(
                file_path=self.child.file_path,
                replacements=(replacement,),
                rationale=rationale,
            )
        )
        if self.child.file_path != declaring_file_path and self.dropped:
            result.extend(
                obsolete_import_edits(
                    snapshot, self.child.file_path, self.dropped, rationale
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class _SiblingMethod:
    owner: OverlapClass
    method: FunctionNode
    statements: tuple[ast.stmt, ...]
    scope: ClassBodyModuleScope
    class_bound_names: frozenset[str]


def _snapshot_sibling_override_step_candidates(
    snapshot: CodemodSourceSnapshot,
) -> tuple[SiblingOverrideStepCandidate, ...]:
    """Compute sibling-override-step candidates once per ``CodemodSourceSnapshot``."""

    return snapshot.retained_operation_derivation(
        (SiblingOverrideStepCandidate,),
        lambda: sibling_override_step_candidates(
            snapshot.class_overlap_verdicts,
            snapshot.compact_class_projections,
        ),
    )


def _snapshot_sibling_override_step_by_evidence(
    snapshot: CodemodSourceSnapshot,
) -> dict[frozenset[tuple[str, str]], SiblingOverrideStepCandidate]:
    return snapshot.retained_operation_derivation(
        (SiblingOverrideStepCandidate, "by_evidence"),
        lambda: {
            frozenset(
                (indexed.file_path, f"{indexed.qualname}.{cand.name}")
                for indexed, _line in cand.overriding
            ): cand
            for cand in _snapshot_sibling_override_step_candidates(snapshot)
        },
    )


@dataclass(frozen=True)
class SiblingOverrideStepHoist:
    """A sibling-override step hoist proved safe onto `declaring`."""

    candidate: SiblingOverrideStepCandidate
    declaring: OverlapClass
    declaring_method: FunctionNode
    method_kind: MethodKind
    hook_name: str
    hook_declaration: HookDeclaration
    prefix: tuple[ast.stmt, ...]
    suffix: tuple[ast.stmt, ...]
    hoist_source_owner: OverlapClass
    rewrites: tuple[SiblingRewrite, ...]
    body_template: SiblingBodyTemplate
    extra_params: ExtraHookParams = ExtraHookParams()
    hoisted_step_expr: HoistedStepExpression | None = None

    @classmethod
    def proved(
        cls,
        snapshot: CodemodSourceSnapshot,
        declaring_target,
        member_name: str,
        hook_name: str,
    ) -> SiblingOverrideStepHoist:
        declaring = OverlapClass.require(snapshot, declaring_target)
        if member_name.startswith("__") and member_name.endswith("__"):
            raise NeedsInput(
                f"`{declaring.qualname}.{member_name}` is a dunder method; NRA needs "
                "a proof that splitting it into a template method leaves class creation "
                "and callers unchanged",
                NeededInput.PROOF,
            )
        verdicts = snapshot.class_overlap_verdicts
        matches = [
            candidate
            for candidate in _snapshot_sibling_override_step_candidates(snapshot)
            if candidate.declaring.symbol == declaring.symbol
            and candidate.name == member_name
        ]
        if len(matches) != 1:
            raise NeedsInput(
                f"expected one sibling-override-step group for `{declaring.qualname}.{member_name}`, "
                f"found {len(matches)}",
                NeededInput.SOURCE_CHANGE,
            )
        (candidate,) = matches
        if not candidate.restated:
            raise NeedsInput(
                f"`{declaring.qualname}.{member_name}` has only omitted steps and no shared "
                "step across every override",
                NeededInput.SOURCE_CHANGE,
            )
        declaring_ancestry = require_static_c3(snapshot, declaring.facts)
        declaring_method, declaring_kind = cls._require_single_method(
            declaring, member_name, allow_abstract=True
        )
        declaring_sig = _parameter_signature_key(declaring_method)
        positional = (
            *declaring_method.args.posonlyargs,
            *declaring_method.args.args,
        )
        if not positional:
            raise NeedsInput(
                f"`{declaring.qualname}.{member_name}` takes no receiver parameter",
                NeededInput.PROOF,
            )
        receiver_name = positional[0].arg
        param_names = _all_parameter_names(declaring_method)

        siblings, method_kind = cls._load_siblings(
            snapshot, candidate, member_name, declaring_sig, declaring_kind
        )
        is_generator = any(
            _contains_node_type(sibling.statements, (ast.Yield, ast.YieldFrom))
            for sibling in siblings
        )
        if is_generator and isinstance(declaring_method, ast.AsyncFunctionDef):
            raise NeedsInput(
                f"`{declaring.qualname}.{member_name}` is an async generator method; NRA needs "
                "an async-generator delegation proof before splitting it",
                NeededInput.PROOF,
            )
        restated_calls = frozenset(candidate.restated)
        (
            prefix_len,
            suffix_len,
            prefix_extra_names,
            prefix_extra_annotations,
        ) = cls._shared_prefix_and_suffix(
            declaring_method, siblings, param_names
        )
        prefix = siblings[0].statements[:prefix_len]
        suffix = (
            siblings[0].statements[len(siblings[0].statements) - suffix_len :]
            if suffix_len
            else ()
        )
        raw_middles = tuple(
            sibling.statements[
                prefix_len : len(sibling.statements) - suffix_len
            ]
            for sibling in siblings
        )
        if not any(raw_middles):
            raise NeedsInput(
                f"every override of `{declaring.qualname}.{member_name}` is identical; "
                "lift the whole method instead",
                NeededInput.SOURCE_CHANGE,
            )

        hoisted_edges = (*prefix, *suffix)
        has_edge_step = bool(hoisted_edges) and _statement_calls_any_step(
            hoisted_edges, receiver_name, restated_calls
        )
        hoisted_step_expr: HoistedStepExpression | None = None
        sibling_step_exprs: tuple[ast.expr, ...] = ()
        matched_body_template: SiblingBodyTemplate | None = None
        matched_hook_bodies: tuple[SiblingHookBody, ...] = ()
        extra_names = list(prefix_extra_names)
        extra_annotations = dict(prefix_extra_annotations)

        if not has_edge_step:
            prefix_locals = frozenset(
                (*param_names, *LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prefix))
            )
            step_match = cls._detect_hoisted_step_expr(
                siblings,
                raw_middles,
                prefix_locals,
                receiver_name,
                restated_calls,
            )
            if step_match is not None:
                hoisted_step_expr, sibling_step_exprs = step_match
                extra_names.append(hoisted_step_expr.var_name)
            elif not suffix and not is_generator:
                template_match = cls._detect_body_template(
                    siblings,
                    raw_middles,
                    prefix_locals,
                    receiver_name,
                    restated_calls,
                )
                if template_match is not None:
                    matched_body_template, matched_hook_bodies = template_match

        if (
            not has_edge_step
            and hoisted_step_expr is None
            and matched_body_template is None
        ):
            raise NeedsInput(
                f"shared step(s) in `{siblings[0].owner.qualname}` and its siblings are not "
                "independent leading or trailing statements safe to hoist into a template method",
                NeededInput.SOURCE_CHANGE,
            )

        extra_params = ExtraHookParams.for_method(
            declaring_method, extra_names, extra_annotations
        )
        if extra_params is None:
            raise NeedsInput(
                f"shared step(s) in `{siblings[0].owner.qualname}` and its siblings bind "
                "locals that cannot be forwarded safely into the hook signature",
                NeededInput.SOURCE_CHANGE,
            )

        effective_middles: list[tuple[ast.stmt, ...]] = []
        leading_replacements: list[tuple[ast.expr, str] | None] = []
        extra_dropped: list[tuple[ast.stmt, ...]] = []
        if hoisted_step_expr is not None:
            step_var = hoisted_step_expr.var_name
            for middle, step_expr in zip(raw_middles, sibling_step_exprs, strict=True):
                first_stmt = middle[0]
                match first_stmt:
                    case ast.Assign(targets=[ast.Name(id=bound)], value=val) if (
                        bound == step_var and val is step_expr
                    ):
                        effective_middles.append(middle[1:])
                        leading_replacements.append(None)
                        extra_dropped.append((first_stmt,))
                    case ast.Expr(value=val) if val is step_expr:
                        effective_middles.append(middle[1:])
                        leading_replacements.append(None)
                        extra_dropped.append((first_stmt,))
                    case _:
                        effective_middles.append(middle)
                        leading_replacements.append((step_expr, step_var))
                        extra_dropped.append((ast.Expr(value=step_expr),))
        else:
            for middle in raw_middles:
                effective_middles.append(middle)
                leading_replacements.append(None)
                extra_dropped.append(())

        middles = tuple(effective_middles)
        if not any(middles):
            raise NeedsInput(
                f"every override of `{declaring.qualname}.{member_name}` is identical; "
                "lift the whole method instead",
                NeededInput.SOURCE_CHANGE,
            )

        cls._require_dataflow_safe(declaring, member_name, prefix, middles, suffix)
        cls._require_mro_descendants_and_hook_safe(
            snapshot,
            verdicts,
            candidate,
            declaring,
            declaring_ancestry,
            tuple(sibling.owner for sibling in siblings),
            member_name,
            hook_name,
        )
        for sibling in siblings:
            child = sibling.owner
            ancestry = require_static_c3(snapshot, child.facts)
            order = verdicts.linearized(child.symbol)
            names = frozenset((member_name, hook_name))
            proof = MemberDeletionCreationProof(
                snapshot,
                child.facts,
                ancestry,
                f"`{child.qualname}.{member_name}`",
                names,
                frozenset(),
                MroWalk(order, order, child.symbol, names),
            )
            RestatedOverrideDeletion.require_once_per_class(
                proof, child, sibling.method
            )
            RestatedOverrideDeletion.require_plain_decorators(proof, child)
            proof.require_unchanged()
            NamespaceReaders(snapshot).require_unread(child.facts, names)

        declaring_order = verdicts.linearized(declaring.symbol)
        declaring_names = frozenset((member_name, hook_name))
        declaring_proof = MemberDeletionCreationProof(
            snapshot,
            declaring.facts,
            declaring_ancestry,
            f"`{declaring.qualname}.{member_name}`",
            declaring_names,
            frozenset(),
            MroWalk(declaring_order, declaring_order, declaring.symbol, declaring_names),
        )
        RestatedOverrideDeletion.require_plain_decorators(declaring_proof, declaring)
        declaring_proof.require_unchanged()
        NamespaceReaders(snapshot).require_unread(declaring.facts, declaring_names)

        hook_declaration = cls._select_hook_declaration(
            declaring_method,
            has_empty_middle=any(not middle for middle in middles),
            is_generator=is_generator,
        )
        if matched_body_template is not None:
            body_template: SiblingBodyTemplate = matched_body_template
        else:
            if is_generator:
                forward_return = (not suffix) and any(
                    _has_non_none_return(middle) for middle in middles
                )
            else:
                any_middle_returns = any(
                    _has_return_statement(middle) for middle in middles
                )
                forward_return = (not suffix) and (
                    any_middle_returns or not _returns_none_annotation(declaring_method)
                )
            body_template = DirectHookTemplate(
                forward_return=forward_return,
                is_generator=is_generator,
            )

        rewrites_list: list[SiblingRewrite] = []
        for idx, (sibling, middle) in enumerate(zip(siblings, middles, strict=True)):
            if not middle:
                rewrites_list.append(
                    DeleteEmptySiblingOverride(sibling.owner, sibling.method)
                )
            else:
                dropped_stmts = (
                    *sibling.statements[:prefix_len],
                    *(
                        sibling.statements[len(sibling.statements) - suffix_len :]
                        if suffix_len
                        else ()
                    ),
                    *extra_dropped[idx],
                )
                hook_body: SiblingHookBody = (
                    matched_hook_bodies[idx]
                    if matched_body_template is not None
                    else DirectMiddleHookBody(
                        middle=middle,
                        leading_expr_replacement=leading_replacements[idx],
                    )
                )
                rewrites_list.append(
                    RewriteSiblingToHook(
                        sibling.owner,
                        sibling.method,
                        hook_body,
                        dropped_stmts,
                        hook_name,
                        method_kind,
                        extra_params=extra_params,
                    )
                )
        return cls(
            candidate=candidate,
            declaring=declaring,
            declaring_method=declaring_method,
            method_kind=method_kind,
            hook_name=hook_name,
            hook_declaration=hook_declaration,
            prefix=prefix,
            suffix=suffix,
            hoist_source_owner=siblings[0].owner,
            rewrites=tuple(rewrites_list),
            body_template=body_template,
            extra_params=extra_params,
            hoisted_step_expr=hoisted_step_expr,
        )

    @staticmethod
    def _require_single_method(
        owner: OverlapClass, member_name: str, *, allow_abstract: bool
    ) -> tuple[FunctionNode, MethodKind]:
        statements = [
            statement
            for statement in owner.node.body
            if member_name in declared_names(statement)
        ]
        if len(statements) != 1 or not isinstance(
            statements[0], ast.FunctionDef | ast.AsyncFunctionDef
        ):
            raise NeedsInput(
                f"`{owner.qualname}.{member_name}` is not a single method definition",
                NeededInput.PROOF,
            )
        method = statements[0]
        return method, MethodKind.of(owner, method, allow_abstract=allow_abstract)

    @classmethod
    def _load_siblings(
        cls,
        snapshot: CodemodSourceSnapshot,
        candidate: SiblingOverrideStepCandidate,
        member_name: str,
        declaring_sig: tuple[object, ...],
        declaring_kind: MethodKind,
    ) -> tuple[tuple[_SiblingMethod, ...], MethodKind]:
        siblings: list[_SiblingMethod] = []
        sibling_kinds: list[MethodKind] = []
        for indexed, _line in candidate.overriding:
            child = OverlapClass.for_symbol(snapshot, indexed.symbol)
            method, child_kind = cls._require_single_method(
                child, member_name, allow_abstract=False
            )
            sibling_kinds.append(child_kind)
            if _parameter_signature_key(method) != declaring_sig:
                raise NeedsInput(
                    f"`{child.qualname}.{member_name}` has a different parameter signature "
                    f"from `{candidate.declaring.qualname}.{member_name}`",
                    NeededInput.PROOF,
                )
            parsed = snapshot.parsed_module_for_source_path(child.file_path)
            scope = ClassBodyModuleScope.for_lines(parsed, (child.node.lineno,))[
                child.node.lineno
            ]
            class_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(child.node.body)
            trimmed = tuple(_trim_leading_docstring(list(method.body)))
            siblings.append(
                _SiblingMethod(child, method, trimmed, scope, class_bound)
            )
        effective_kind = declaring_kind
        if (
            isinstance(declaring_kind, PropertyMethodKind)
            and candidate.unanswered
            and sibling_kinds
            and all(kind.is_cached_property() for kind in sibling_kinds)
        ):
            effective_kind = CachedPropertyMethodKind()
        for sibling, child_kind in zip(siblings, sibling_kinds, strict=True):
            if child_kind != effective_kind:
                raise NeedsInput(
                    f"`{sibling.owner.qualname}.{member_name}` has a different descriptor kind "
                    f"from `{candidate.declaring.qualname}.{member_name}`",
                    NeededInput.PROOF,
                )
        return tuple(siblings), effective_kind

    @classmethod
    def _statements_semantically_equal(
        cls,
        first: _SiblingMethod,
        other: _SiblingMethod,
        stmt_a: ast.stmt,
        stmt_b: ast.stmt,
        local_names: frozenset[str],
    ) -> bool:
        if ast.dump(stmt_a, include_attributes=False) != ast.dump(
            stmt_b, include_attributes=False
        ):
            return False
        if reads_class_cell_or_private_name(stmt_a) or reads_class_cell_or_private_name(
            stmt_b
        ):
            return False
        if _contains_node_type(
            (stmt_a,), (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)
        ):
            return False
        free_names = sorted(
            {
                reference.id
                for reference in ModuleLexicalDependencyProjection.from_module(
                    ast.Module(body=[stmt_a], type_ignores=[])
                ).external_name_references
                if reference.id not in local_names
            }
        )
        for name in free_names:
            if name in first.class_bound_names or name in other.class_bound_names:
                return False
            witness_a = first.scope.witness_for(name)
            witness_b = other.scope.witness_for(name)
            if witness_a is not None or witness_b is not None:
                if witness_a != witness_b:
                    return False
            elif (
                first.scope.deferred_for(name) is not None
                or other.scope.deferred_for(name) is not None
            ):
                return False
        return True

    @classmethod
    def _shared_prefix_and_suffix(
        cls,
        declaring_method: FunctionNode,
        siblings: tuple[_SiblingMethod, ...],
        param_names: frozenset[str],
    ) -> tuple[int, int, tuple[str, ...], dict[str, ast.expr]]:
        first = siblings[0]
        min_len = min(len(sibling.statements) for sibling in siblings)
        prefix_len = 0
        prefix_locals = set(param_names)
        while prefix_len < min_len:
            stmt_0 = first.statements[prefix_len]
            if all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    stmt_0,
                    sibling.statements[prefix_len],
                    frozenset(prefix_locals),
                )
                for sibling in siblings[1:]
            ):
                prefix_locals.update(
                    LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((stmt_0,))
                )
                prefix_len += 1
            else:
                break

        max_suffix = min_len - prefix_len
        suffix_len = 0
        while suffix_len < max_suffix:
            stmt_0 = first.statements[len(first.statements) - 1 - suffix_len]
            if all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    stmt_0,
                    sibling.statements[len(sibling.statements) - 1 - suffix_len],
                    param_names,
                )
                for sibling in siblings[1:]
            ):
                suffix_len += 1
            else:
                break

        # A suffix cannot be hoisted when any middle body has an early `return`
        # or binds a local name read by the suffix: drop suffix if unsafe.
        if suffix_len > 0:
            candidate_suffix = first.statements[len(first.statements) - suffix_len :]
            suffix_reads = _read_names(candidate_suffix)
            middles = tuple(
                sibling.statements[
                    prefix_len : len(sibling.statements) - suffix_len
                ]
                for sibling in siblings
            )
            if any(
                _has_return_statement(middle)
                or bool(
                    suffix_reads
                    & LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(middle)
                )
                for middle in middles
            ):
                suffix_len = 0

        # When a shared prefix binds locals read by any middle body, forward
        # definitely-assigned top-level bindings into the hook; drop prefix only
        # when a conditionally-bound local leaks into a middle or cannot be forwarded.
        forwarded_names: tuple[str, ...] = ()
        forwarded_annotations: dict[str, ast.expr] = {}
        if prefix_len > 0:
            candidate_prefix = first.statements[:prefix_len]
            prefix_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(candidate_prefix)
            if prefix_bound:
                middles = tuple(
                    sibling.statements[
                        prefix_len : len(sibling.statements) - suffix_len
                    ]
                    for sibling in siblings
                )
                middle_reads: set[str] = set()
                for middle in middles:
                    middle_reads.update(_read_names(middle))
                needed_from_prefix = prefix_bound & middle_reads
                if needed_from_prefix:
                    definite_names, definite_annotations = _ordered_definite_bindings(
                        candidate_prefix
                    )
                    if not (needed_from_prefix <= set(definite_names)):
                        prefix_len = 0
                    else:
                        extra_names = tuple(
                            name
                            for name in definite_names
                            if name in needed_from_prefix and name not in param_names
                        )
                        if (
                            extra_names
                            and ExtraHookParams.for_method(
                                declaring_method, extra_names, definite_annotations
                            )
                            is None
                        ):
                            prefix_len = 0
                        else:
                            forwarded_names = extra_names
                            forwarded_annotations = {
                                name: definite_annotations[name]
                                for name in extra_names
                                if name in definite_annotations
                            }
        return prefix_len, suffix_len, forwarded_names, forwarded_annotations

    @classmethod
    def _detect_hoisted_step_expr(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[HoistedStepExpression, tuple[ast.expr, ...]] | None:
        if any(not middle for middle in raw_middles):
            return None
        exprs: list[ast.expr] = []
        for middle in raw_middles:
            found = _leading_unconditional_step_expr(
                middle[0], receiver_name, restated_calls
            )
            if found is None:
                return None
            exprs.append(found)
        first = siblings[0]
        first_stmt = ast.Expr(value=exprs[0])
        for sibling, expr in zip(siblings[1:], exprs[1:], strict=True):
            if not cls._statements_semantically_equal(
                first, sibling, first_stmt, ast.Expr(value=expr), prefix_locals
            ):
                return None
        # Choose a clean, collision-free parameter name for the hoisted step value.
        assigned_names: set[str] = set()
        for middle, expr in zip(raw_middles, exprs, strict=True):
            match middle[0]:
                case ast.Assign(targets=[ast.Name(id=bound)], value=val) if (
                    val is expr
                ):
                    assigned_names.add(bound)
        forbidden: set[str] = set(prefix_locals)
        for middle, expr in zip(raw_middles, exprs, strict=True):
            match middle[0]:
                case ast.Assign(targets=[ast.Name(id=bound)], value=val) if (
                    val is expr and len(assigned_names) == 1
                ):
                    rest = middle[1:]
                case _:
                    rest = middle
            forbidden.update(_read_names(rest))
            forbidden.update(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(rest))
        if len(assigned_names) == 1:
            preferred = next(iter(assigned_names))
            if (
                preferred.isidentifier()
                and not keyword_module.iskeyword(preferred)
                and preferred not in prefix_locals
            ):
                # Ensure no non-assigning sibling reads or binds `preferred` as a different name.
                conflict = False
                for middle, expr in zip(raw_middles, exprs, strict=True):
                    match middle[0]:
                        case ast.Assign(targets=[ast.Name(id=bound)], value=val) if (
                            bound == preferred and val is expr
                        ):
                            continue
                    if preferred in _read_names(middle) or preferred in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(middle):
                        conflict = True
                        break
                if not conflict:
                    return (
                        HoistedStepExpression(var_name=preferred, expr=exprs[0]),
                        tuple(exprs),
                    )
        call_node = (
            exprs[0].value
            if isinstance(exprs[0], ast.Await) and isinstance(exprs[0].value, ast.Call)
            else exprs[0]
        )
        call_path = (
            _receiver_call_path(call_node.func, receiver_name)
            if isinstance(call_node, ast.Call)
            else None
        )
        raw_base = (call_path or "step").rsplit(".", maxsplit=1)[-1].lstrip("_") or "step"
        candidates = (
            raw_base,
            f"{raw_base}_value",
            f"_{raw_base}",
            f"_{raw_base}_value",
        )
        var_name = next(
            (
                cand
                for cand in candidates
                if cand.isidentifier()
                and not keyword_module.iskeyword(cand)
                and cand not in forbidden
            ),
            None,
        )
        if var_name is None:
            return None
        return HoistedStepExpression(var_name=var_name, expr=exprs[0]), tuple(exprs)

    @classmethod
    def _detect_body_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[SiblingBodyTemplate, tuple[SiblingHookBody, ...]] | None:
        for detector in (
            cls._detect_return_call_template,
            cls._detect_variadic_return_call_template,
            cls._detect_boolop_template,
            cls._detect_trailing_guarded_step_template,
            cls._detect_guarded_early_return_suffix_template,
        ):
            matched = detector(
                siblings,
                raw_middles,
                prefix_locals,
                receiver_name,
                restated_calls,
            )
            if matched is not None:
                return matched
        return None

    @classmethod
    def _detect_return_call_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[ReturnCallTemplate, tuple[SiblingHookBody, ...]] | None:
        calls: list[ast.Call] = []
        wrappers: list[tuple[type[ast.stmt], bool]] = []
        preps: list[tuple[ast.stmt, ...]] = []
        for middle in raw_middles:
            if not middle:
                return None
            prep = middle[:-1]
            if _has_return_statement(prep) or _contains_node_type(
                prep, (ast.Yield, ast.YieldFrom)
            ):
                return None
            preps.append(prep)
            last = middle[-1]
            match last:
                case ast.Return(value=ast.Await(value=ast.Call() as call)):
                    calls.append(call)
                    wrappers.append((ast.Return, True))
                case ast.Return(value=ast.Call() as call):
                    calls.append(call)
                    wrappers.append((ast.Return, False))
                case ast.Expr(value=ast.Await(value=ast.Call() as call)):
                    calls.append(call)
                    wrappers.append((ast.Expr, True))
                case ast.Expr(value=ast.Call() as call):
                    calls.append(call)
                    wrappers.append((ast.Expr, False))
                case _:
                    return None
        if len(set(wrappers)) != 1:
            return None
        first_call = calls[0]
        if _receiver_call_path(first_call.func, receiver_name) not in restated_calls:
            return None
        if any(isinstance(arg, ast.Starred) for arg in first_call.args) or CallKeywords.of(
            first_call.keywords
        ).splats:
            return None
        kw_names = tuple(kw.arg for kw in first_call.keywords)
        first = siblings[0]
        for sibling, call in zip(siblings[1:], calls[1:], strict=True):
            if len(call.args) != len(first_call.args):
                return None
            if tuple(kw.arg for kw in call.keywords) != kw_names:
                return None
            if any(isinstance(arg, ast.Starred) for arg in call.args):
                return None
            if not cls._statements_semantically_equal(
                first,
                sibling,
                ast.Expr(value=first_call.func),
                ast.Expr(value=call.func),
                prefix_locals,
            ):
                return None
        diff_pos: list[int] = []
        for i in range(len(first_call.args)):
            if not all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    ast.Expr(value=first_call.args[i]),
                    ast.Expr(value=call.args[i]),
                    prefix_locals,
                )
                for sibling, call in zip(siblings[1:], calls[1:], strict=True)
            ):
                diff_pos.append(i)
        diff_kw: list[int] = []
        for j in range(len(first_call.keywords)):
            if not all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    ast.Expr(value=first_call.keywords[j].value),
                    ast.Expr(value=call.keywords[j].value),
                    prefix_locals,
                )
                for sibling, call in zip(siblings[1:], calls[1:], strict=True)
            ):
                diff_kw.append(j)
        if len(diff_pos) + len(diff_kw) != 1:
            return None
        prep_bound = frozenset().union(
            *(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prep) for prep in preps)
        )
        if diff_pos:
            slot_index = diff_pos[0]
            slot_exprs = tuple(call.args[slot_index] for call in calls)
            fixed_exprs: list[ast.expr] = [
                *(arg for idx, arg in enumerate(first_call.args) if idx != slot_index),
                *(kw.value for kw in first_call.keywords),
            ]
            has_earlier_args = slot_index > 0
            sanitized_args = [
                ast.Constant(value=None) if idx == slot_index else arg
                for idx, arg in enumerate(first_call.args)
            ]
            sanitized_call = ast.Call(
                func=first_call.func,
                args=sanitized_args,
                keywords=list(first_call.keywords),
            )
        else:
            kw_index = diff_kw[0]
            slot_exprs = tuple(call.keywords[kw_index].value for call in calls)
            fixed_exprs = [
                *first_call.args,
                *(
                    kw.value
                    for idx, kw in enumerate(first_call.keywords)
                    if idx != kw_index
                ),
            ]
            has_earlier_args = bool(first_call.args) or kw_index > 0
            sanitized_keywords = [
                ast.keyword(arg=kw.arg, value=ast.Constant(value=None))
                if idx == kw_index
                else kw
                for idx, kw in enumerate(first_call.keywords)
            ]
            sanitized_call = ast.Call(
                func=first_call.func,
                args=list(first_call.args),
                keywords=sanitized_keywords,
            )
        if prep_bound and (
            _read_names(tuple(ast.Expr(value=expr) for expr in (first_call.func, *fixed_exprs)))
            & prep_bound
        ):
            return None
        has_prep = any(bool(prep) for prep in preps)
        hoist_var_name: str | None = None
        if has_prep:
            if has_earlier_args:
                return None
            forbidden = prefix_locals | _read_names((raw_middles[0][-1],))
            hoist_var_name = next(
                (
                    name
                    for name in ("_step_value", "_step_arg", "_hook_value")
                    if name not in forbidden
                ),
                None,
            )
            if hoist_var_name is None:
                return None
        template = ReturnCallTemplate(
            template_stmt=raw_middles[0][-1],
            template_call=first_call,
            varying_slot_expr=slot_exprs[0],
            import_checked_stmt=ast.Expr(value=sanitized_call),
            hoist_var_name=hoist_var_name,
        )
        hook_bodies = tuple(
            ReturningExprHookBody(prep=prep, return_expr=slot_expr)
            for prep, slot_expr in zip(preps, slot_exprs, strict=True)
        )
        return template, hook_bodies

    @classmethod
    def _detect_variadic_return_call_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[VariadicReturnCallTemplate, tuple[SiblingHookBody, ...]] | None:
        calls: list[ast.Call] = []
        wrappers: list[tuple[type[ast.stmt], bool]] = []
        preps: list[tuple[ast.stmt, ...]] = []
        for middle in raw_middles:
            if not middle:
                return None
            prep = middle[:-1]
            if _has_return_statement(prep) or _contains_node_type(
                prep, (ast.Yield, ast.YieldFrom)
            ):
                return None
            preps.append(prep)
            last = middle[-1]
            match last:
                case ast.Return(value=ast.Await(value=ast.Call() as call)):
                    calls.append(call)
                    wrappers.append((ast.Return, True))
                case ast.Return(value=ast.Call() as call):
                    calls.append(call)
                    wrappers.append((ast.Return, False))
                case ast.Expr(value=ast.Await(value=ast.Call() as call)):
                    calls.append(call)
                    wrappers.append((ast.Expr, True))
                case ast.Expr(value=ast.Call() as call):
                    calls.append(call)
                    wrappers.append((ast.Expr, False))
                case _:
                    return None
        if len(set(wrappers)) != 1:
            return None
        first_call = calls[0]
        if _receiver_call_path(first_call.func, receiver_name) not in restated_calls:
            return None
        if any(
            any(isinstance(arg, ast.Starred) for arg in call.args)
            or CallKeywords.of(call.keywords).splats
            for call in calls
        ):
            return None
        kw_names = tuple(kw.arg for kw in first_call.keywords)
        first = siblings[0]
        for sibling, call in zip(siblings[1:], calls[1:], strict=True):
            if tuple(kw.arg for kw in call.keywords) != kw_names:
                return None
            if not cls._statements_semantically_equal(
                first,
                sibling,
                ast.Expr(value=first_call.func),
                ast.Expr(value=call.func),
                prefix_locals,
            ):
                return None
            for kw_a, kw_b in zip(first_call.keywords, call.keywords, strict=True):
                if not cls._statements_semantically_equal(
                    first,
                    sibling,
                    ast.Expr(value=kw_a.value),
                    ast.Expr(value=kw_b.value),
                    prefix_locals,
                ):
                    return None
        arg_counts = {len(call.args) for call in calls}
        if len(arg_counts) == 1 and len(first_call.args) < 2:
            return None
        prep_bound = frozenset().union(
            *(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prep) for prep in preps)
        )
        fixed_exprs = (first_call.func, *(kw.value for kw in first_call.keywords))
        if prep_bound and (
            _read_names(tuple(ast.Expr(value=expr) for expr in fixed_exprs)) & prep_bound
        ):
            return None
        has_prep = any(bool(prep) for prep in preps)
        hoist_var_name: str | None = None
        if has_prep:
            forbidden = prefix_locals | _read_names((raw_middles[0][-1],))
            hoist_var_name = next(
                (
                    name
                    for name in ("_step_args", "_step_values", "_hook_args")
                    if name not in forbidden
                ),
                None,
            )
            if hoist_var_name is None:
                return None
        wrapper_cls, is_await = wrappers[0]
        sanitized_call = ast.Call(
            func=first_call.func,
            args=[],
            keywords=list(first_call.keywords),
        )
        template = VariadicReturnCallTemplate(
            template_call=first_call,
            is_return=wrapper_cls is ast.Return,
            is_await=is_await,
            fixed_keywords=tuple(first_call.keywords),
            import_checked_stmt=ast.Expr(value=sanitized_call),
            hoist_var_name=hoist_var_name,
        )
        hook_bodies = tuple(
            ReturningTupleHookBody(prep=prep, args=tuple(call.args))
            for prep, call in zip(preps, calls, strict=True)
        )
        return template, hook_bodies

    @classmethod
    def _detect_boolop_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[BoolOpTemplate, tuple[SiblingHookBody, ...]] | None:
        boolops: list[ast.BoolOp] = []
        preps: list[tuple[ast.stmt, ...]] = []
        for middle in raw_middles:
            if not middle:
                return None
            prep = middle[:-1]
            if _has_return_statement(prep) or _contains_node_type(
                prep, (ast.Yield, ast.YieldFrom)
            ):
                return None
            match middle[-1]:
                case ast.Return(value=ast.BoolOp() as boolop) if len(boolop.values) >= 2:
                    boolops.append(boolop)
                    preps.append(prep)
                case _:
                    return None
        op_type = type(boolops[0].op)
        if any(type(b.op) is not op_type for b in boolops[1:]):
            return None
        first = siblings[0]
        first_values = boolops[0].values
        min_operands = min(len(b.values) for b in boolops)
        prep_bound = frozenset().union(
            *(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prep) for prep in preps)
        )

        # Try shared trailing operands: `return <hook> and/or <shared_tail>`.
        tail_len = 0
        while tail_len < min_operands - 1:
            expr_0 = first_values[len(first_values) - 1 - tail_len]
            if _read_names((ast.Expr(value=expr_0),)) & prep_bound:
                break
            if all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    ast.Expr(value=expr_0),
                    ast.Expr(value=b.values[len(b.values) - 1 - tail_len]),
                    prefix_locals,
                )
                for sibling, b in zip(siblings[1:], boolops[1:], strict=True)
            ):
                tail_len += 1
            else:
                break
        if tail_len > 0:
            shared_tail = tuple(first_values[len(first_values) - tail_len :])
            if _statement_calls_any_step(
                tuple(ast.Expr(value=e) for e in shared_tail),
                receiver_name,
                restated_calls,
            ):
                template = BoolOpTemplate(
                    op=boolops[0].op,
                    shared_operands=shared_tail,
                    hook_first=True,
                )
                hook_bodies = tuple(
                    ReturningBoolOpHookBody(
                        prep=prep,
                        op=b.op,
                        operands=tuple(b.values[: len(b.values) - tail_len]),
                    )
                    for prep, b in zip(preps, boolops, strict=True)
                )
                return template, hook_bodies

        # Try shared leading operands (only when no sibling has prep statements):
        # `return <shared_head> and/or <hook>`.
        if any(bool(prep) for prep in preps):
            return None
        head_len = 0
        while head_len < min_operands - 1:
            expr_0 = first_values[head_len]
            if all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    ast.Expr(value=expr_0),
                    ast.Expr(value=b.values[head_len]),
                    prefix_locals,
                )
                for sibling, b in zip(siblings[1:], boolops[1:], strict=True)
            ):
                head_len += 1
            else:
                break
        if head_len > 0:
            shared_head = tuple(first_values[:head_len])
            if _statement_calls_any_step(
                tuple(ast.Expr(value=e) for e in shared_head),
                receiver_name,
                restated_calls,
            ):
                template = BoolOpTemplate(
                    op=boolops[0].op,
                    shared_operands=shared_head,
                    hook_first=False,
                )
                hook_bodies = tuple(
                    ReturningBoolOpHookBody(
                        prep=(),
                        op=b.op,
                        operands=tuple(b.values[head_len:]),
                    )
                    for b in boolops
                )
                return template, hook_bodies
        return None

    @classmethod
    def _detect_trailing_guarded_step_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[TrailingGuardedStepTemplate, tuple[SiblingHookBody, ...]] | None:
        preps: list[tuple[ast.stmt, ...]] = []
        conditions: list[ast.expr] = []
        bodies: list[tuple[ast.stmt, ...]] = []
        for middle in raw_middles:
            if not middle:
                return None
            prep = middle[:-1]
            if _has_return_statement(prep) or _contains_node_type(
                prep, (ast.Yield, ast.YieldFrom)
            ):
                return None
            match middle[-1]:
                case ast.If(test=cond, body=if_body, orelse=[]) if if_body and not _contains_node_type(
                    (ast.Expr(value=cond),), (ast.NamedExpr, ast.Yield, ast.YieldFrom)
                ):
                    preps.append(prep)
                    conditions.append(cond)
                    bodies.append(tuple(if_body))
                case _:
                    return None
        first_body = bodies[0]
        if _contains_node_type(first_body, (ast.Break, ast.Continue)):
            return None
        if not _statement_calls_any_step(first_body, receiver_name, restated_calls):
            return None
        prep_bound = frozenset().union(
            *(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prep) for prep in preps)
        )
        if _read_names(first_body) & prep_bound:
            return None
        first = siblings[0]
        for sibling, other_body in zip(siblings[1:], bodies[1:], strict=True):
            if len(other_body) != len(first_body):
                return None
            local_names = set(prefix_locals)
            for stmt_a, stmt_b in zip(first_body, other_body, strict=True):
                if not cls._statements_semantically_equal(
                    first, sibling, stmt_a, stmt_b, frozenset(local_names)
                ):
                    return None
                local_names.update(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((stmt_a,)))
        template = TrailingGuardedStepTemplate(guarded_body=first_body)
        hook_bodies = tuple(
            ReturningExprHookBody(prep=prep, return_expr=cond)
            for prep, cond in zip(preps, conditions, strict=True)
        )
        return template, hook_bodies

    @classmethod
    def _detect_guarded_early_return_suffix_template(
        cls,
        siblings: tuple[_SiblingMethod, ...],
        raw_middles: tuple[tuple[ast.stmt, ...], ...],
        prefix_locals: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[GuardedEarlyReturnSuffixTemplate, tuple[SiblingHookBody, ...]] | None:
        min_len = min(len(middle) for middle in raw_middles)
        if min_len < 2:
            return None
        first = siblings[0]
        first_middle = raw_middles[0]
        tail_len = 0
        while tail_len < min_len - 1:
            stmt_0 = first_middle[len(first_middle) - 1 - tail_len]
            if all(
                cls._statements_semantically_equal(
                    first,
                    sibling,
                    stmt_0,
                    middle[len(middle) - 1 - tail_len],
                    prefix_locals,
                )
                for sibling, middle in zip(siblings[1:], raw_middles[1:], strict=True)
            ):
                tail_len += 1
            else:
                break
        if tail_len == 0:
            return None
        shared_tail = first_middle[len(first_middle) - tail_len :]
        if _contains_node_type(shared_tail, (ast.Break, ast.Continue)):
            return None
        if not _statement_calls_any_step(shared_tail, receiver_name, restated_calls):
            return None
        preps: list[tuple[ast.stmt, ...]] = []
        conditions: list[ast.expr] = []
        early_bodies: list[tuple[ast.stmt, ...]] = []
        for middle in raw_middles:
            before_tail = middle[: len(middle) - tail_len]
            if not before_tail:
                return None
            prep = before_tail[:-1]
            if _has_return_statement(prep) or _contains_node_type(
                prep, (ast.Yield, ast.YieldFrom)
            ):
                return None
            match before_tail[-1]:
                case ast.If(test=cond, body=if_body, orelse=[]) if (
                    len(if_body) == 1
                    and isinstance(if_body[0], ast.Return)
                    and not _contains_node_type(
                        (ast.Expr(value=cond),), (ast.NamedExpr, ast.Yield, ast.YieldFrom)
                    )
                ):
                    preps.append(prep)
                    conditions.append(cond)
                    early_bodies.append(tuple(if_body))
                case _:
                    return None
        first_early = early_bodies[0]
        prep_bound = frozenset().union(
            *(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(prep) for prep in preps)
        )
        if (_read_names(first_early) | _read_names(shared_tail)) & prep_bound:
            return None
        for sibling, other_early in zip(siblings[1:], early_bodies[1:], strict=True):
            if not cls._statements_semantically_equal(
                first, sibling, first_early[0], other_early[0], prefix_locals
            ):
                return None
        template = GuardedEarlyReturnSuffixTemplate(
            early_return_body=first_early,
            shared_tail=shared_tail,
        )
        hook_bodies = tuple(
            ReturningExprHookBody(prep=prep, return_expr=cond)
            for prep, cond in zip(preps, conditions, strict=True)
        )
        return template, hook_bodies

    @staticmethod
    def _require_dataflow_safe(
        declaring: OverlapClass,
        member_name: str,
        prefix: tuple[ast.stmt, ...],
        middles: tuple[tuple[ast.stmt, ...], ...],
        suffix: tuple[ast.stmt, ...],
    ) -> None:
        if _contains_node_type(prefix, (ast.Break, ast.Continue)):
            raise NeedsInput(
                f"leading step in `{declaring.qualname}.{member_name}` contains loop control flow",
                NeededInput.PROOF,
            )
        if suffix and any(_has_return_statement(middle) for middle in middles):
            raise NeedsInput(
                f"an override of `{declaring.qualname}.{member_name}` returns early before the trailing step",
                NeededInput.PROOF,
            )

    @classmethod
    def _require_mro_descendants_and_hook_safe(
        cls,
        snapshot: CodemodSourceSnapshot,
        verdicts: ClassOverlapVerdicts,
        candidate: SiblingOverrideStepCandidate,
        declaring: OverlapClass,
        declaring_ancestry,
        children: tuple[OverlapClass, ...],
        member_name: str,
        hook_name: str,
    ) -> None:
        del declaring_ancestry
        class_index = verdicts.class_index
        declaring_order = verdicts.linearized(declaring.symbol)
        for ancestor_symbol in declaring_order:
            ancestor = class_index.class_for(ancestor_symbol)
            if ancestor is not None and (
                hook_name in ancestor.direct_declared_member_names
                or hook_name in ancestor.method_names
            ):
                raise NeedsInput(
                    f"`{ancestor.qualname}` already binds hook name `{hook_name}`",
                    NeededInput.SOURCE_CHANGE,
                )

        if not candidate.unanswered:
            cls._require_declaring_not_directly_used(snapshot, declaring, member_name)

        overriding_symbols = frozenset(child.symbol for child in children)
        for child in children:
            order = verdicts.linearized(child.symbol)
            if declaring.symbol not in order:
                raise NeedsInput(
                    f"`{declaring.qualname}` is not in `{child.qualname}`'s MRO",
                    NeededInput.PROOF,
                )
            between = order[1 : order.index(declaring.symbol)]
            intervening = next_binder(verdicts, between, member_name)
            if intervening is not None:
                raise NeedsInput(
                    f"`{intervening}` binds `{member_name}` between `{child.qualname}` "
                    f"and `{declaring.qualname}`",
                    NeededInput.PROOF,
                )

        for desc_symbol in class_index.descendant_symbols(declaring.symbol):
            desc = OverlapClass.for_symbol(snapshot, desc_symbol)
            require_static_c3(snapshot, desc.facts)
            order = verdicts.linearized(desc_symbol)
            if hook_name in desc.target.bound_names:
                raise NeedsInput(
                    f"`{desc.qualname}` already binds hook name `{hook_name}`",
                    NeededInput.SOURCE_CHANGE,
                )
            if _calls_super_member(desc.node, member_name):
                raise NeedsInput(
                    f"`{desc.qualname}` calls `super().{member_name}()`, which hoisting "
                    f"onto `{declaring.qualname}` would change",
                    NeededInput.PROOF,
                )
            if desc_symbol in overriding_symbols:
                continue
            if declaring.symbol not in order:
                continue
            before_declaring = order[: order.index(declaring.symbol)]
            if any(symbol in overriding_symbols for symbol in before_declaring):
                continue
            binder = next_binder(verdicts, before_declaring, member_name)
            if binder is None:
                if cls._remains_abstract(verdicts, order, member_name):
                    continue
                raise NeedsInput(
                    f"`{desc.qualname}` inherits `{declaring.qualname}.{member_name}` "
                    "without overriding it",
                    NeededInput.PROOF,
                )

    @staticmethod
    def _require_declaring_not_directly_used(
        snapshot: CodemodSourceSnapshot,
        declaring: OverlapClass,
        member_name: str,
    ) -> None:
        if declaring.facts.abstract_method_names:
            return
        allowed_roles = frozenset((ClassReferenceRole.TYPE_TEST,))
        for site in ClassReferenceSites(snapshot, frozenset((declaring.symbol,))).sites:
            if site.is_base or site.is_annotation or site.role in allowed_roles:
                continue
            raise NeedsInput(
                f"{site.source.file_path}:{site.node.lineno} uses `{declaring.qualname}` "
                f"directly, which currently answers `{member_name}` itself",
                NeededInput.PROOF,
            )

    @staticmethod
    def _remains_abstract(
        verdicts: ClassOverlapVerdicts,
        order: tuple[str, ...],
        hoisted_member: str,
    ) -> bool:
        class_index = verdicts.class_index
        if not any(
            symbol in ("abc.ABC", "abc.ABCMeta")
            or (
                (indexed := class_index.class_for(symbol)) is not None
                and any(
                    meta in ("abc.ABCMeta", "ABCMeta")
                    for meta in indexed.metaclass_names
                )
            )
            for symbol in order
        ):
            return False
        abstract_names: set[str] = set()
        for symbol in order:
            indexed = class_index.class_for(symbol)
            if indexed is not None:
                abstract_names.update(indexed.abstract_method_names)
        abstract_names.discard(hoisted_member)
        for abstract_name in abstract_names:
            first = next_binder(verdicts, order, abstract_name)
            if first is None:
                continue
            owner = class_index.class_for(first)
            if owner is not None and abstract_name in owner.abstract_method_names:
                return True
        return False

    @staticmethod
    def _select_hook_declaration(
        declaring_method: FunctionNode,
        *,
        has_empty_middle: bool,
        is_generator: bool = False,
    ) -> HookDeclaration:
        if has_empty_middle:
            return (
                EmptyGeneratorHookDeclaration()
                if is_generator
                else NoOpHookDeclaration()
            )
        for decorator in declaring_method.decorator_list:
            if _decorator_terminal_name(decorator) in _ABSTRACT_DECORATORS:
                return AbstractHookDeclaration(ast.unparse(decorator))
        return NotImplementedHookDeclaration()

    def edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        declaring = self.declaring
        declaring_source = snapshot.sources_by_file_path[declaring.file_path]
        declaring_lines = declaring_source.splitlines(keepends=True)
        hoist_source = snapshot.sources_by_file_path[
            self.hoist_source_owner.file_path
        ]
        hoist_source_lines = hoist_source.splitlines(keepends=True)
        body = ClassBodySourceAuthority(declaring.node, declaring_source)
        indent = body.indentation
        inner_indent = f"{indent}    "
        start, end = body.geometry.node_span_offsets(
            SourceNodeSpan(self.declaring_method, Include)
        )

        template_pieces: list[str] = [
            f"{indent}@{dec}\n"
            for dec in self.method_kind.template_decorators(self.declaring_method)
        ]
        header = _method_header_signature(
            self.declaring_method, self.candidate.name, declaring_lines
        )
        for line in header.splitlines():
            template_pieces.append(f"{indent}{line}\n" if line.strip() else "\n")
        docstring = _leading_docstring_stmt(self.declaring_method)
        if docstring is not None:
            template_pieces.append(
                _statement_source_block(declaring_lines, docstring, inner_indent)
            )
        for statement in self.prefix:
            template_pieces.append(
                _statement_source_block(hoist_source_lines, statement, inner_indent)
            )
        if self.hoisted_step_expr is not None:
            expr_src = _render_expr_source(hoist_source, self.hoisted_step_expr.expr)
            template_pieces.append(
                f"{inner_indent}{self.hoisted_step_expr.var_name} = {expr_src}\n"
            )
        hook_call = _forwarded_hook_call(
            self.declaring_method, self.hook_name, extra_params=self.extra_params
        )
        call_line = self.body_template.render_template_call(
            hoist_source,
            hoist_source_lines,
            hook_call,
            inner_indent,
        )
        template_pieces.append(call_line)
        for statement in self.suffix:
            template_pieces.append(
                _statement_source_block(hoist_source_lines, statement, inner_indent)
            )
        template_pieces.append("\n")
        template_pieces.append(
            self.body_template.render_hook_declaration(
                self.hook_declaration,
                self.declaring_method,
                self.hook_name,
                indent,
                self.method_kind,
                declaring_lines,
                extra_params=self.extra_params,
            )
        )

        replacement = SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source="".join(template_pieces),
        )
        hoisted_statements: list[ast.stmt] = [*self.prefix, *self.suffix]
        if self.hoisted_step_expr is not None:
            hoisted_statements.append(ast.Expr(value=self.hoisted_step_expr.expr))
        hoisted_statements.extend(self.body_template.import_checked_statements())
        all_edits: list[NominalSourceEdit] = [
            *body.geometry.physical_edits(
                file_path=declaring.file_path,
                replacements=(replacement,),
                rationale=rationale,
            ),
            *required_name_imports(
                snapshot,
                tuple(hoisted_statements),
                self.hoist_source_owner,
                declaring,
                rationale,
            ),
        ]
        if self.method_kind.is_cached_property() and not any(
            _decorator_terminal_name(dec) == "cached_property"
            for dec in self.declaring_method.decorator_list
        ):
            declaring_parsed = snapshot.parsed_module_for_source_path(
                declaring.file_path
            )
            declaring_scope = ClassBodyModuleScope.for_lines(
                declaring_parsed, (declaring.node.lineno,)
            )[declaring.node.lineno]
            if declaring_scope.witness_for("cached_property") is None:
                all_edits.append(
                    ModuleImportMutation.from_source(
                        file_path=declaring.file_path,
                        import_source="from functools import cached_property\n",
                        scope=ModuleImportScope.RUNTIME,
                        rationale=rationale,
                    )
                )
        dropped_abstract = self.hook_declaration.obsolete_declaring_decorators(
            self.declaring_method
        )
        if dropped_abstract:
            all_edits.extend(
                obsolete_import_edits(
                    snapshot, declaring.file_path, dropped_abstract, rationale
                )
            )
        for rewrite in self.rewrites:
            all_edits.extend(rewrite.edits(snapshot, declaring.file_path, rationale))
        return tuple(all_edits)

    def rationale(self) -> str:
        siblings = ", ".join(
            f"`{rewrite.child.qualname}`" for rewrite in self.rewrites
        )
        return (
            f"Hoists shared step(s) of `{self.candidate.name}` into template method "
            f"`{self.declaring.qualname}.{self.candidate.name}` calling "
            f"`{self.hook_name}` across {siblings}."
        )


@dataclass(frozen=True, kw_only=True)
class HoistSiblingOverrideStepOperation(RepositorySourceReprovedOperation):
    """Hoist identical leading/trailing step(s) from sibling overrides into their
    declaring ancestor's template method and route differing bodies through a hook."""

    member_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    hook_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        for field_name, value in (
            ("member_name", self.member_name),
            ("hook_name", self.hook_name),
        ):
            if not value.isidentifier() or keyword_module.iskeyword(value):
                raise ValueError(f"{field_name} must be a Python identifier")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        hoist = SiblingOverrideStepHoist.proved(
            snapshot, self.target, self.member_name, self.hook_name
        )
        return hoist.edits(snapshot, self.rationale_text(hoist.rationale()))


class SiblingOverrideStepFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    SiblingOverrideStepConcept,
):
    """Hoist shared leading or trailing steps of sibling overrides into their ancestor."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        if context is None or not finding.evidence:
            return self.rejected_evaluation(
                "hoisting a sibling override step needs source context and sibling evidence"
            )
        snapshot = context.execution_snapshot()
        evidence_keys = frozenset(
            (location.file_path, location.symbol) for location in finding.evidence
        )
        candidate = _snapshot_sibling_override_step_by_evidence(snapshot).get(
            evidence_keys
        )
        if candidate is None:
            return self.rejected_evaluation("the sibling override step left the source")
        hook_name = f"_{candidate.name}_step"
        operation = HoistSiblingOverrideStepOperation(
            target=source_rewrite_target(candidate.declaring),
            member_name=candidate.name,
            hook_name=hook_name,
        )
        try:
            recipe = RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Every override of `{candidate.declaring.qualname}.{candidate.name}` "
                    f"shares hoistable step(s); move them once into "
                    f"`{candidate.declaring.qualname}.{candidate.name}` calling `{hook_name}`."
                ),
            ).with_proved_operation(operation, context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(recipe)


__all__ = (
    "AbstractHookDeclaration",
    "BoolOpTemplate",
    "CachedPropertyMethodKind",
    "ClassMethodKind",
    "DeleteEmptySiblingOverride",
    "DirectHookTemplate",
    "DirectMiddleHookBody",
    "EmptyGeneratorHookDeclaration",
    "ExtraHookParams",
    "GuardedEarlyReturnSuffixTemplate",
    "HoistSiblingOverrideStepOperation",
    "HoistedStepExpression",
    "HookDeclaration",
    "MethodKind",
    "NoOpHookDeclaration",
    "NotImplementedHookDeclaration",
    "OmittedStep",
    "PlainMethodKind",
    "PropertyMethodKind",
    "ReturnCallTemplate",
    "ReturningBoolOpHookBody",
    "ReturningExprHookBody",
    "ReturningTupleHookBody",
    "RewriteSiblingToHook",
    "SiblingBodyTemplate",
    "SiblingHookBody",
    "SiblingOverrideStepCandidate",
    "SiblingOverrideStepFindingRecipeSynthesizer",
    "SiblingOverrideStepHoist",
    "SiblingRewrite",
    "TrailingGuardedStepTemplate",
    "VariadicReturnCallTemplate",
    "sibling_override_step_candidates",
)
