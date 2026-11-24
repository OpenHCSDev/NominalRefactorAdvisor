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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.ast_field_decoders import (
    declared_annotations,
    variadic_parameters,
)
from nominal_refactor_advisor.class_index import (
    PER_CLASS_NAMES,
    ClassBodyDeclarationKind,
    ClassBodyModuleScope,
    CompactIndexedClass,
    CompactModuleClassProjection,
    reads_class_cell_or_private_name,
)
from nominal_refactor_advisor.class_method_facets import CompactClassMethod
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.class_syntax_facets import _trim_leading_docstring
from nominal_refactor_advisor.codemod_declaration_source import ClassBodySourceAuthority
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
_KIND_DECORATORS = frozenset(("classmethod", "property"))


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


def sibling_override_step_candidates(
    verdicts: ClassOverlapVerdicts,
    projections: Sequence[CompactModuleClassProjection],
) -> tuple[SiblingOverrideStepCandidate, ...]:
    """Every inherited method overridden differently by independent siblings with a shared
    or omitted receiver step."""

    index, bodies = verdicts.class_index, verdicts.bodies
    methods: dict[str, dict[str, CompactClassMethod]] = defaultdict(dict)
    for projection in projections:
        for method in projection.class_methods:
            methods[method.class_symbol][method.method_name] = method
    declared = frozenset(name for owned in methods.values() for name in owned)
    overriding: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for symbol in bodies.declarations_by_symbol:
        mine = methods.get(symbol)
        if not mine:
            continue
        indexed_self = index.class_for(symbol)
        if indexed_self is None:
            continue
        abstract = indexed_self.abstract_method_names
        for name, declaration in bodies.declarations_of(symbol).items():
            if (
                name in PER_CLASS_NAMES
                or name in abstract
                or name not in mine
                or declaration.kind is not ClassBodyDeclarationKind.METHOD
            ):
                continue
            for ancestor in index.ancestor_symbols(symbol):
                if (
                    ancestor in bodies.declarations_by_symbol
                    and name in bodies.declarations_of(ancestor)
                ):
                    overriding[(ancestor, name)].append(symbol)
    nearest: dict[tuple[str, frozenset[str]], SiblingOverrideStepCandidate] = {}
    read: dict[str, frozenset[str]] = {}
    for (declaring, name), symbols in overriding.items():
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
        key = (name, frozenset(independent))
        held = nearest.get(key)
        if held is None or held.declaring.symbol in index.ancestor_symbols(declaring):
            nearest[key] = SiblingOverrideStepCandidate(
                declaring=index.classes_by_symbol[declaring],
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
            )
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


def _method_header_signature(
    method: FunctionNode,
    name: str,
    source_lines: Sequence[str] | None = None,
) -> str:
    """Render `def <name>(...) -> Ret:`, preserving original source spacing when available."""

    if source_lines is not None and method.body:
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
    args_src = ast.unparse(method.args)
    returns_annotations = tuple(declared_annotations(method))
    returns_src = (
        f" -> {ast.unparse(returns_annotations[0])}" if returns_annotations else ""
    )
    return f"{prefix} {name}({args_src}){returns_src}:"


class MethodKind(ABC):
    """How the hoisted method and its hook are decorated (plain, `@classmethod`, `@property`)."""

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
        return PlainMethodKind()


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


def _forwarded_hook_call(method: FunctionNode, hook_name: str) -> str:
    """Render `[await ]<receiver>.<hook_name>(<forwarded arguments>)`."""

    args = method.args
    positional = (*args.posonlyargs, *args.args)
    receiver = positional[0].arg
    call_args: list[str] = [param.arg for param in positional[1:]]
    match args:
        case ast.arguments(vararg=ast.arg(arg=vararg_name)):
            call_args.append(f"*{vararg_name}")
    for kw_param in args.kwonlyargs:
        call_args.append(f"{kw_param.arg}={kw_param.arg}")
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
    ) -> str:
        header = _method_header_signature(method, hook_name, source_lines)
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
class NotImplementedHookDeclaration(HookDeclaration):
    """Every sibling has a non-empty middle body on an ancestor without `@abstractmethod`."""

    body_line: ClassVar[str] = "raise NotImplementedError"


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
    middle: tuple[ast.stmt, ...]
    dropped: tuple[ast.stmt, ...]
    hook_name: str
    method_kind: MethodKind

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
        header = _method_header_signature(self.method, self.hook_name, source_lines)
        for line in header.splitlines():
            pieces.append(f"{indent}{line}\n" if line.strip() else "\n")
        docstring = _leading_docstring_stmt(self.method)
        if docstring is not None:
            pieces.append(_statement_source_block(source_lines, docstring, inner_indent))
        for statement in self.middle:
            pieces.append(_statement_source_block(source_lines, statement, inner_indent))
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
    forward_return: bool

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
        declaring_method, method_kind = cls._require_single_method(
            declaring, member_name, allow_abstract=True
        )
        cls._require_non_generator_method(declaring, declaring_method)
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

        siblings = cls._load_siblings(
            snapshot, candidate, member_name, declaring_sig, method_kind
        )
        prefix_len, suffix_len = cls._shared_prefix_and_suffix(
            siblings, param_names, receiver_name, frozenset(candidate.restated)
        )
        prefix = siblings[0].statements[:prefix_len]
        suffix = (
            siblings[0].statements[len(siblings[0].statements) - suffix_len :]
            if suffix_len
            else ()
        )
        middles = tuple(
            sibling.statements[
                prefix_len : len(sibling.statements) - suffix_len
            ]
            for sibling in siblings
        )
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
            declaring_method, has_empty_middle=any(not middle for middle in middles)
        )
        any_middle_returns = any(_has_return_statement(middle) for middle in middles)
        forward_return = (not suffix) and (
            any_middle_returns or not _returns_none_annotation(declaring_method)
        )
        rewrites = tuple(
            DeleteEmptySiblingOverride(sibling.owner, sibling.method)
            if not middle
            else RewriteSiblingToHook(
                sibling.owner,
                sibling.method,
                middle,
                (
                    *sibling.statements[:prefix_len],
                    *(
                        sibling.statements[len(sibling.statements) - suffix_len :]
                        if suffix_len
                        else ()
                    ),
                ),
                hook_name,
                method_kind,
            )
            for sibling, middle in zip(siblings, middles, strict=True)
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
            rewrites=rewrites,
            forward_return=forward_return,
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

    @staticmethod
    def _require_non_generator_method(
        owner: OverlapClass, method: FunctionNode
    ) -> None:
        if _contains_node_type(method.body, (ast.Yield, ast.YieldFrom)):
            raise NeedsInput(
                f"`{owner.qualname}.{method.name}` is a generator method; NRA needs "
                "a generator-delegation proof before splitting it",
                NeededInput.PROOF,
            )

    @classmethod
    def _load_siblings(
        cls,
        snapshot: CodemodSourceSnapshot,
        candidate: SiblingOverrideStepCandidate,
        member_name: str,
        declaring_sig: tuple[object, ...],
        declaring_kind: MethodKind,
    ) -> tuple[_SiblingMethod, ...]:
        siblings: list[_SiblingMethod] = []
        for indexed, _line in candidate.overriding:
            child = OverlapClass.for_symbol(snapshot, indexed.symbol)
            method, child_kind = cls._require_single_method(
                child, member_name, allow_abstract=False
            )
            if child_kind != declaring_kind:
                raise NeedsInput(
                    f"`{child.qualname}.{member_name}` has a different descriptor kind "
                    f"from `{candidate.declaring.qualname}.{member_name}`",
                    NeededInput.PROOF,
                )
            cls._require_non_generator_method(child, method)
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
        return tuple(siblings)

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
        siblings: tuple[_SiblingMethod, ...],
        param_names: frozenset[str],
        receiver_name: str,
        restated_calls: frozenset[str],
    ) -> tuple[int, int]:
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

        # A prefix cannot be hoisted when it binds a local name read by any middle or suffix.
        if prefix_len > 0:
            candidate_prefix = first.statements[:prefix_len]
            prefix_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(candidate_prefix)
            if prefix_bound:
                remainders = tuple(
                    sibling.statements[prefix_len:] for sibling in siblings
                )
                if any(prefix_bound & _read_names(rem) for rem in remainders):
                    prefix_len = 0

        hoisted = (
            *first.statements[:prefix_len],
            *(
                first.statements[len(first.statements) - suffix_len :]
                if suffix_len
                else ()
            ),
        )
        if not hoisted or not _statement_calls_any_step(
            hoisted, receiver_name, restated_calls
        ):
            raise NeedsInput(
                f"shared step(s) in `{first.owner.qualname}` and its siblings are not "
                "independent leading or trailing statements safe to hoist into a template method",
                NeededInput.SOURCE_CHANGE,
            )
        return prefix_len, suffix_len

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
        declaring_method: FunctionNode, *, has_empty_middle: bool
    ) -> HookDeclaration:
        if has_empty_middle:
            return NoOpHookDeclaration()
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
        hoist_source_lines = snapshot.sources_by_file_path[
            self.hoist_source_owner.file_path
        ].splitlines(keepends=True)
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
        hook_call = _forwarded_hook_call(self.declaring_method, self.hook_name)
        call_line = (
            f"{inner_indent}return {hook_call}\n"
            if self.forward_return
            else f"{inner_indent}{hook_call}\n"
        )
        template_pieces.append(call_line)
        for statement in self.suffix:
            template_pieces.append(
                _statement_source_block(hoist_source_lines, statement, inner_indent)
            )
        template_pieces.append("\n")
        template_pieces.append(
            self.hook_declaration.render(
                self.declaring_method,
                self.hook_name,
                indent,
                self.method_kind,
                declaring_lines,
            )
        )

        replacement = SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=end,
            replacement_source="".join(template_pieces),
        )
        hoisted_statements = (*self.prefix, *self.suffix)
        all_edits: list[NominalSourceEdit] = [
            *body.geometry.physical_edits(
                file_path=declaring.file_path,
                replacements=(replacement,),
                rationale=rationale,
            ),
            *required_name_imports(
                snapshot,
                hoisted_statements,
                self.hoist_source_owner,
                declaring,
                rationale,
            ),
        ]
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
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason=(
                    f"Every override of `{candidate.declaring.qualname}.{candidate.name}` "
                    f"shares hoistable step(s); move them once into "
                    f"`{candidate.declaring.qualname}.{candidate.name}` calling `{hook_name}`."
                ),
            ).with_operation(operation)
        )


__all__ = (
    "AbstractHookDeclaration",
    "ClassMethodKind",
    "DeleteEmptySiblingOverride",
    "HoistSiblingOverrideStepOperation",
    "HookDeclaration",
    "MethodKind",
    "NoOpHookDeclaration",
    "NotImplementedHookDeclaration",
    "OmittedStep",
    "PlainMethodKind",
    "PropertyMethodKind",
    "RewriteSiblingToHook",
    "SiblingOverrideStepCandidate",
    "SiblingOverrideStepFindingRecipeSynthesizer",
    "SiblingOverrideStepHoist",
    "SiblingRewrite",
    "sibling_override_step_candidates",
)
