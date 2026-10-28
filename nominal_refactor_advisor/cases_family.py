"""An enum that declares behaviour becomes the class family its members describe.

``CasesFamily`` is the ``Cases`` input form with its own end state: the enum's
name stays as an abstract root, each member becomes a subclass carrying its data
as class attributes, and every method that read ``self`` to tell members apart
splits into one body per member, decided once by folding the member's literals
into its tests. Branches on member data outside the enum become calls of a new
family method; the names for those methods are questions the caller answers.

Members are classes, so identity (``status is Hit``), hashing and pickling keep
their meaning. The root the family inherits is the target repository's own
declaration (``DeclaredRoot``): it names the attribute members carry their wire
value in and the classmethod that decodes one, so ``X(value)`` becomes
``X.<decoder>(value)`` and wire names survive.
"""

from __future__ import annotations

import ast
import builtins
import operator
import textwrap
import tokenize
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .ast_field_decoders import declared_annotations
from .value_expression import CompactValueExpression
from .ast_projection import StatementValue
from .lexical_bindings import ImportBoundNameProjection
from .ast_tools import (
    AstParentIndex,
    BuiltinCallName,
    ParsedModule,
    PythonSourcePathPolicy,
)
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement
from .descent import (
    Answers,
    Cases,
    Descent,
    EnumDeclaration,
    Member,
    MethodName,
    ModuleNames,
    Question,
    Rewrite,
    TailName,
)
from .declared_root import DeclaredRoot
from .semantic_descent import AuthorityClaim, SemanticAuthorityKind

ITERATING_CALLS = frozenset({"tuple", "list", "set", "frozenset", "sorted"})
BOOLEAN_CALLS = frozenset(
    (*BuiltinCallName.type_test_names(), BuiltinCallName.BOOL, "callable", "hasattr")
)


class CallbackName(Question):
    @property
    def prompt(self) -> str:
        return "parameter name for the fall-through the family calls back"


class Unknown:
    """A value folding cannot decide from one member's literals."""


UNKNOWN = Unknown()


class Comparison(ABC, metaclass=AutoRegisterMeta):
    """One Python comparison operator, registered under the ``ast`` node it decodes:
    its answer for two known operands, and the operator that answers its negation
    exactly, if one does."""

    __registry__: ClassVar[dict[type[ast.cmpop], type[Comparison]]] = {}
    __registry_key__ = "node"
    __skip_if_no_key__ = True
    node: ClassVar[type[ast.cmpop]]
    inverse: ClassVar[type[ast.cmpop] | None] = None

    @classmethod
    def of(cls, op: ast.cmpop) -> type[Comparison]:
        """The boundary: the member decoding one parsed operator."""

        return cls.__registry__[type(op)]

    @staticmethod
    @abstractmethod
    def apply(left: object, right: object) -> object:
        """The operator applied to two operands."""

    @classmethod
    @abstractmethod
    def evaluated(cls, left: object, right: object) -> object:
        """The comparison's value, or UNKNOWN when the operands cannot decide it."""


class ValueComparison(Comparison, ABC):
    """Decided by the operands' values; operands it cannot compare are UNKNOWN."""

    @classmethod
    def evaluated(cls, left: object, right: object) -> object:
        try:
            return cls.apply(left, right)
        except TypeError:
            return UNKNOWN


class IdentityComparison(Comparison, ABC):
    """``is``/``is not``: decided only between None and bools, which are singletons."""

    @classmethod
    def evaluated(cls, left: object, right: object) -> object:
        if not all(value is None or isinstance(value, bool) for value in (left, right)):
            return UNKNOWN
        return cls.apply(left, right)


class EqualComparison(ValueComparison):
    node = ast.Eq
    apply = staticmethod(operator.eq)


class NotEqualComparison(ValueComparison):
    node = ast.NotEq
    apply = staticmethod(operator.ne)


class LessComparison(ValueComparison):
    node = ast.Lt
    apply = staticmethod(operator.lt)


class LessEqualComparison(ValueComparison):
    node = ast.LtE
    apply = staticmethod(operator.le)


class GreaterComparison(ValueComparison):
    node = ast.Gt
    apply = staticmethod(operator.gt)


class GreaterEqualComparison(ValueComparison):
    node = ast.GtE
    apply = staticmethod(operator.ge)


class ContainsComparison(ValueComparison):
    node = ast.In
    inverse = ast.NotIn

    @staticmethod
    def apply(left: object, right: object) -> object:
        return left in right


class NotContainsComparison(ValueComparison):
    node = ast.NotIn
    inverse = ast.In

    @staticmethod
    def apply(left: object, right: object) -> object:
        return left not in right


class IsComparison(IdentityComparison):
    node = ast.Is
    inverse = ast.IsNot
    apply = staticmethod(operator.is_)


class IsNotComparison(IdentityComparison):
    node = ast.IsNot
    inverse = ast.Is
    apply = staticmethod(operator.is_not)


def handback(module: ParsedModule, node: ast.AST, reason: str) -> ValueError:
    return ValueError(f"handback: {module.file_path}:{node.lineno}: {reason}")


def body_without_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        return statements[1:]
    return statements


def docstring(statements: list[ast.stmt]) -> ast.Expr | None:
    if body_without_docstring(statements) is statements:
        return None
    first = statements[0]
    return first if isinstance(first, ast.Expr) else None


def decorator_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(ast.unparse(decorator) for decorator in function.decorator_list)


def is_name(node: ast.AST | None, names: Iterable[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in set(names)


def loaded_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
    return frozenset(
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def stored_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
    return frozenset(
        node.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del)
    )


def module_bound_names(module: ast.Module) -> frozenset[str]:
    """Names one module binds at top level: definitions, imports and assignments."""

    names: set[str] = set()
    for statement in module.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(statement.name)
        elif isinstance(statement, ast.Import | ast.ImportFrom):
            names.update(ImportBoundNameProjection(statement).names())
        elif isinstance(statement, ast.If):
            names.update(module_bound_names(ast.Module(statement.body, [])))
        else:
            names.update(stored_names([statement]))
    return frozenset(names)


def literal_constant(value: object) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


def negated(node: ast.expr) -> ast.expr:
    """``not node`` read as a test, with one comparison inverted where exact."""

    if isinstance(node, ast.Constant):
        return ast.Constant(not node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return node.operand
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        inverse = Comparison.of(node.ops[0]).inverse
        if inverse is not None:
            return ast.Compare(node.left, [inverse()], list(node.comparators))
    return ast.UnaryOp(ast.Not(), node)


def rebuilt(node: ast.AST, renames: dict[int, str]) -> ast.AST:
    """A copy of ``node`` with each renamed node replaced by the source it becomes."""

    if id(node) in renames:
        return ast.parse(renames[id(node)], mode="eval").body
    fields: dict[str, object] = {}
    for name, value in ast.iter_fields(node):
        if isinstance(value, list):
            fields[name] = [
                rebuilt(item, renames) if isinstance(item, ast.AST) else item
                for item in value
            ]
        elif isinstance(value, ast.AST):
            fields[name] = rebuilt(value, renames)
        else:
            fields[name] = value
    return type(node)(**fields)


def plain_lambda(node: ast.expr) -> bool:
    """A lambda of positional names whose body binds no name of its own."""

    return (
        isinstance(node, ast.Lambda)
        and not node.args.posonlyargs
        and node.args.vararg is None
        and not node.args.kwonlyargs
        and node.args.kwarg is None
        and not node.args.defaults
        and not any(
            isinstance(
                part,
                ast.Lambda
                | ast.NamedExpr
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            )
            for part in ast.walk(node.body)
        )
    )


def applied(function: ast.Lambda, arguments: list[str]) -> str:
    """``function``'s body with each parameter replaced by its argument's source."""

    bound = {
        parameter.arg: argument
        for parameter, argument in zip(function.args.args, arguments)
    }
    body = rebuilt(
        function.body,
        {
            id(node): bound[node.id]
            for node in ast.walk(function.body)
            if isinstance(node, ast.Name) and node.id in bound
        },
    )
    assert isinstance(body, ast.expr)
    source = ast.unparse(body)
    atomic = ast.Name | ast.Attribute | ast.Call | ast.Constant | ast.Subscript
    return source if isinstance(body, atomic) else f"({source})"


def is_boolean(node: ast.expr, boolean_names: frozenset[str]) -> bool:
    """Whether ``node`` evaluates to a ``bool``, read from its syntax alone."""

    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, bool)
    if isinstance(node, ast.BoolOp):
        return all(is_boolean(value, boolean_names) for value in node.values)
    if isinstance(node, ast.Name):
        return node.id in boolean_names
    return isinstance(node, ast.Call) and is_name(node.func, BOOLEAN_CALLS)


def type_positions(annotation: ast.expr) -> Iterator[ast.AST]:
    """The nodes of ``annotation`` in type position: ``Annotated`` metadata are values."""

    pending: list[ast.AST] = [annotation]
    while pending:
        node = pending.pop()
        yield node
        if (
            isinstance(node, ast.Subscript)
            and (
                is_name(node.value, ["Annotated"])
                or isinstance(node.value, ast.Attribute)
                and node.value.attr == "Annotated"
            )
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            pending.extend((node.value, node.slice.elts[0]))
        else:
            pending.extend(ast.iter_child_nodes(node))


def member_type(root: str, source: str) -> str:
    """Annotation ``source`` with ``root``, a type of members, read as ``type[root]``.

    String annotations are rewritten inside their quotes.
    """

    tree = ast.parse(source, mode="eval")
    lines = SourceLines(source)
    parents = AstParentIndex(tree).parent_by_node
    replacements: list[tuple[int, int, str]] = []
    for part in type_positions(tree.body):
        if is_name(part, [root]):
            parent = parents.get(part)
            if isinstance(parent, ast.Subscript) and is_name(parent.value, ["type"]):
                continue
            replacements.append((*lines.offsets(part), f"type[{root}]"))
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            inner = member_type(root, part.value)
            if inner != part.value:
                replacements.append((*lines.offsets(part), f'"{inner}"'))
    for start, end, text in sorted(replacements, reverse=True):
        source = source[:start] + text + source[end:]
    return source


@dataclass
class Folding:
    """One member's reading of a block: decided tests folded, every read renamed.

    ``literal`` answers a read the member decides; ``renames`` maps a node to the
    source it becomes. Tests are rebuilt from the AST, every other statement keeps
    its own source text.
    """

    module: ParsedModule
    rewrite: Rewrite
    literal: Callable[[ast.AST], object]
    renames: dict[int, str]
    boolean_names: frozenset[str]

    def expression(self, node: ast.expr) -> ast.expr:
        copy = rebuilt(node, self.renames)
        return copy if isinstance(copy, ast.expr) else node

    def decides(self, node: ast.AST) -> bool:
        return any(self.literal(part) is not UNKNOWN for part in ast.walk(node))

    def value(self, node: ast.expr) -> ast.expr:
        known = self.literal(node)
        if known is not UNKNOWN and literal_constant(known):
            return ast.Constant(known)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        return self.expression(node)

    def truth(self, node: ast.expr) -> ast.expr:
        """``node`` read only for its truth, with every decided part folded."""

        known = self.literal(node)
        if known is not UNKNOWN:
            return ast.Constant(bool(known))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return negated(self.truth(node.operand))
        if isinstance(node, ast.BoolOp):
            conjunction = isinstance(node.op, ast.And)
            parts: list[ast.expr] = []
            for value in node.values:
                part = self.truth(value)
                if isinstance(part, ast.Constant):
                    if bool(part.value) != conjunction:
                        return ast.Constant(not conjunction)
                    continue
                parts.append(part)
            if not parts:
                return ast.Constant(conjunction)
            if len(parts) == 1:
                return parts[0]
            return ast.BoolOp(type(node.op)(), parts)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        return self.expression(node)

    def compare(self, node: ast.Compare) -> ast.expr:
        operands = [self.value(node.left), *map(self.value, node.comparators)]
        if all(isinstance(operand, ast.Constant) for operand in operands):
            known = self.evaluated_chain(
                [
                    operand.value
                    for operand in operands
                    if isinstance(operand, ast.Constant)
                ],
                node.ops,
            )
            if known is not UNKNOWN:
                return ast.Constant(bool(known))
        if len(node.ops) == 1 and isinstance(
            node.ops[0], ast.Is | ast.IsNot | ast.Eq | ast.NotEq
        ):
            left, right = operands
            for constant, other in ((left, right), (right, left)):
                if (
                    isinstance(constant, ast.Constant)
                    and isinstance(constant.value, bool)
                    and is_boolean(other, self.boolean_names)
                ):
                    positive = (
                        isinstance(node.ops[0], ast.Is | ast.Eq) == constant.value
                    )
                    return other if positive else negated(other)
        return ast.Compare(operands[0], list(node.ops), operands[1:])

    @staticmethod
    def evaluated_chain(values: list[object], ops: list[ast.cmpop]) -> object:
        for left, op, right in zip(values, ops, values[1:]):
            result = Comparison.of(op).evaluated(left, right)
            if result is UNKNOWN:
                return UNKNOWN
            if not result:
                return False
        return True

    def block(self, statements: list[ast.stmt]) -> tuple[list[str], bool]:
        """Source lines of ``statements`` and whether every path through them ends."""

        lines: list[str] = []
        for statement in statements:
            emitted, ends = self.statement(statement)
            lines.extend(emitted)
            if ends:
                return lines, True
        return lines, False

    def statement(self, statement: ast.stmt) -> tuple[list[str], bool]:
        if isinstance(statement, ast.If) and self.decides(statement):
            return self.chain(statement, "if")
        if (
            isinstance(statement, ast.Expr)
            and id(statement.value) in self.renames
            and literal_value(
                ast.parse(self.renames[id(statement.value)], mode="eval").body
            )
            is not UNKNOWN
        ):
            # An applied lambda that reduced to a constant does nothing here.
            return [], False
        source = self.rewrite.statements_source(
            self.module,
            [statement],
            [
                (node, self.renames[id(node)])
                for node in outermost(statement, self.renames)
            ],
        )
        return source.rstrip("\n").split("\n"), isinstance(
            statement, ast.Return | ast.Raise
        )

    def chain(self, statement: ast.If, keyword: str) -> tuple[list[str], bool]:
        test = self.truth(statement.test)
        if isinstance(test, ast.Constant):
            chosen = statement.body if test.value else statement.orelse
            if not test.value and len(chosen) == 1 and isinstance(chosen[0], ast.If):
                # An ``elif`` node's source starts at ``elif``: never slice it.
                return self.chain(chosen[0], keyword)
            if keyword == "if":
                return self.block(chosen)
            lines, ends = self.block(chosen)
            return (["else:", *indented(lines)], ends) if lines else ([], False)
        body, body_ends = self.block(statement.body)
        lines = [f"{keyword} {ast.unparse(test)}:", *indented(body or ["pass"])]
        if not statement.orelse:
            return lines, False
        if len(statement.orelse) == 1 and isinstance(statement.orelse[0], ast.If):
            rest, rest_ends = self.chain(statement.orelse[0], "elif")
            return lines + rest, body_ends and rest_ends
        orelse, orelse_ends = self.block(statement.orelse)
        if not orelse:
            return lines, False
        return lines + ["else:", *indented(orelse)], body_ends and orelse_ends


def indented(lines: list[str], prefix: str = "    ") -> list[str]:
    return [f"{prefix}{line}" if line else line for line in lines]


def outermost(root: ast.AST, renames: dict[int, str]) -> Iterator[ast.AST]:
    """The renamed nodes under ``root`` that no other renamed node encloses."""

    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in renames:
            yield node
            continue
        pending.extend(ast.iter_child_nodes(node))


def parameter_sources(
    arguments: ast.arguments,
    first: str | None,
    annotation_text: Callable[[ast.expr], str],
    default_text: Callable[[ast.expr], str],
) -> list[str]:
    """Each parameter as Black renders it, the first renamed to ``first``.

    ``annotation_text`` renders an annotation, ``default_text`` a default value:
    a default is a value, never a type to rewrite.
    """

    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    )
    defaults.extend(arguments.defaults)
    sources: list[str] = []
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        name = first if index == 0 and first is not None else argument.arg
        sources.append(
            argument_source(
                name, argument.annotation, default, annotation_text, default_text
            )
        )
        if index + 1 == len(arguments.posonlyargs):
            sources.append("/")
    if arguments.vararg is not None:
        sources.append(
            "*"
            + argument_source(
                arguments.vararg.arg,
                arguments.vararg.annotation,
                None,
                annotation_text,
                default_text,
            )
        )
    elif arguments.kwonlyargs:
        sources.append("*")
    for argument, kw_default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        sources.append(
            argument_source(
                argument.arg,
                argument.annotation,
                kw_default,
                annotation_text,
                default_text,
            )
        )
    if arguments.kwarg is not None:
        sources.append(
            "**"
            + argument_source(
                arguments.kwarg.arg,
                arguments.kwarg.annotation,
                None,
                annotation_text,
                default_text,
            )
        )
    return sources


def argument_source(
    name: str,
    annotation: ast.expr | None,
    default: ast.expr | None,
    annotation_text: Callable[[ast.expr], str],
    default_text: Callable[[ast.expr], str],
) -> str:
    if annotation is None:
        return name if default is None else f"{name}={default_text(default)}"
    source = f"{name}: {annotation_text(annotation)}"
    return source if default is None else f"{source} = {default_text(default)}"


def segment(module: ParsedModule, node: ast.expr) -> str:
    """``node``'s own source on one line, else its unparsed form."""

    found = ast.get_source_segment(module.source, node)
    return found if found is not None and "\n" not in found else ast.unparse(node)


def signature(
    name: str,
    parameters: list[str],
    returns: str | None,
    indent: str,
    line_length: int,
) -> list[str]:
    """``def`` header lines, exploded one parameter per line past the line length."""

    suffix = "" if returns is None else f" -> {returns}"
    line = f"def {name}({', '.join(parameters)}){suffix}:"
    if len(indent) + len(line) <= line_length:
        return [line]
    return [
        f"def {name}(",
        *(f"    {parameter}," for parameter in parameters),
        f"){suffix}:",
    ]



def authored_header(
    module: ParsedModule,
    function: ast.FunctionDef,
    first: str | None,
    annotation_text: Callable[[ast.expr], str],
) -> list[str]:
    """``function``'s own ``def`` header lines, as its author laid them out, with
    the receiver renamed to ``first`` and each annotation rendered by
    ``annotation_text``."""

    geometry = SourceTextGeometry(module.source)
    start = geometry.required_node_offsets(function)[0]
    arguments = function.args
    receiver = (
        None
        if first is None
        else next(iter([*arguments.posonlyargs, *arguments.args]), None)
    )
    renamed = (
        []
        if receiver is None or receiver.arg == first
        else [
            SourceTextSpanReplacement.from_offsets(
                start_offset=geometry.required_node_offsets(receiver)[0],
                end_offset=geometry.required_node_offsets(receiver)[0]
                + len(receiver.arg),
                replacement_source=first,
            )
        ]
    )
    annotations = [
        annotation
        for annotation in (
            *(
                argument.annotation
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    arguments.vararg,
                    *arguments.kwonlyargs,
                    arguments.kwarg,
                )
                if argument is not None
            ),
            function.returns,
        )
        if annotation is not None
    ]
    retyped = [
        SourceTextSpanReplacement.from_offsets(
            start_offset=geometry.required_node_offsets(annotation)[0],
            end_offset=geometry.required_node_offsets(annotation)[1],
            replacement_source=annotation_text(annotation),
        )
        for annotation in annotations
        if annotation_text(annotation) != segment(module, annotation)
    ]
    text = geometry.source_with_replacements_in_span(
        start, header_end(module.source, start), [*renamed, *retyped]
    )
    head, *rest = text.split("\n")
    indent = " " * function.col_offset
    return [head, *(line.removeprefix(indent) for line in rest)]


def header_end(source: str, start: int) -> int:
    """The offset just past the colon closing the ``def`` header at ``start``."""

    depth = 0
    lines = source[start:].splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    for token in tokenize.generate_tokens(iter(lines).__next__):
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth -= 1
        elif token.string == ":" and depth == 0:
            row, column = token.end
            return start + line_offsets[row - 1] + column
    raise ValueError("def header has no closing colon")


@dataclass(frozen=True)
class Datum:
    """One attribute every member class declares: its name, annotation and store."""

    name: str
    annotation: str | None
    stored: str


@dataclass
class MethodBodies:
    """One family method: its header, each member's folded body and the unfolded one.

    The root takes the body most members share and each other member overrides
    it, unless two members would carry the same override: that copy restates a
    sibling, so the root keeps the unfolded body, reading each member's own data,
    and no member overrides. A method that applies inlined data has no unfolded
    body: that data no longer exists.
    """

    name: str
    header: list[str]
    bodies: dict[str, list[str]]
    generic: list[str] | None
    docstring: list[str] = field(default_factory=list)

    def root_body(self, members: tuple[Member, ...]) -> list[str]:
        counts = Counter(tuple(self.bodies[member.name]) for member in members)
        most = max(counts.values())
        root = next(
            self.bodies[member.name]
            for member in members
            if counts[tuple(self.bodies[member.name])] == most
        )
        if self.generic is not None and any(
            count > 1 for body, count in counts.items() if list(body) != root
        ):
            return self.generic
        return root

    def override(self, member: Member, members: tuple[Member, ...]) -> list[str] | None:
        root = self.root_body(members)
        body = self.bodies[member.name]
        return None if root is self.generic or body == root else body

    def source(self, body: list[str], with_docstring: bool) -> list[str]:
        lines = [
            *self.header,
            *indented(
                [*(self.docstring if with_docstring else []), *(body or ["pass"])]
            ),
        ]
        return indented(lines)


@dataclass
class CaseModel:
    """The enum read as a family: per-member data, literals and folded properties."""

    enum: EnumDeclaration
    declared: DeclaredRoot
    data: list[Datum]
    expressions: dict[str, dict[str, ast.expr]]
    literals: dict[str, dict[str, object]]
    folded: dict[str, str | None]
    forwarded: dict[str, str]
    called: frozenset[str]
    inlined: dict[str, dict[str, ast.Lambda]] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return self.enum.name

    @property
    def members(self) -> tuple[Member, ...]:
        return self.enum.members

    @cached_property
    def class_names(self) -> dict[str, str]:
        return {member.name: member.class_name for member in self.members}

    @property
    def decided_names(self) -> frozenset[str]:
        """Attributes every member answers with a literal: reads a fold decides."""

        return frozenset(
            name
            for name in (*(datum.name for datum in self.data), *self.folded)
            if all(name in self.literals[member.name] for member in self.members)
        )

    @property
    def decided_reads(self) -> frozenset[str]:
        """The attribute names a read of a decided datum spells: its own, or the
        stored field it forwards (``value``, when the root carries it elsewhere)."""

        decided = self.decided_names
        return decided | {
            stored for stored, name in self.forwarded.items() if name in decided
        }

    def attribute_name(self, attribute: str) -> str:
        return self.forwarded.get(attribute, attribute)


class CasesFamily(Cases, Descent):
    """An enum that declares behaviour, read through its members' data and identity.

    The enum's name becomes an abstract root; members become its subclasses.
    """

    def admitted(self, enum: EnumDeclaration) -> EnumDeclaration:
        """Every method has a place: on the root, or split across the members."""

        return enum

    def claims(self, snapshot: CodemodSourceSnapshot) -> tuple[AuthorityClaim, ...]:
        enum = self.enum(snapshot)
        return (
            AuthorityClaim(
                claimed_symbol=enum.name,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
                file_path=enum.module.file_path,
                qualname=enum.name,
            ),
        )

    def rewrite(self, snapshot: CodemodSourceSnapshot) -> Rewrite:
        enum = self.enum(snapshot)
        model = self.model(enum, snapshot)
        rewrite = Rewrite(snapshot)
        answer = Answers(self.names)
        aliases = self.aliases(snapshot, enum, enum.module, enum.name)
        self.require_free_class_names(model)
        base = self.root_base(model)
        methods = [self.method(model, rewrite, function) for function in enum.methods]
        handled: set[int] = {id(node) for node in ast.walk(enum.node)}
        for site in self.sites(snapshot, model, aliases, answer):
            methods.append(site.install(self, model, rewrite, answer, handled))
        rewrite.replace(
            enum.module, enum.node, self.family_source(model, rewrite, methods, base)
        )
        for module, names in aliases.items():
            self.rewrite_reads(rewrite, module, names, model, handled)
        ValueReads(model, snapshot, aliases).rewrite(rewrite, handled)
        self.rewrite_called(rewrite, snapshot, model, handled)
        answer.require_complete()
        if self.imported_root(model) is None:
            rewrite.ensure_import(
                enum.module, model.declared.module, (model.declared.name,)
            )
        self.rebind_imports(rewrite, aliases, enum, enum.module)
        return rewrite

    @staticmethod
    def imported_root(model: CaseModel) -> str | None:
        """The local name the enum's module binds the declared root to, if any."""

        declared = model.declared
        module = model.enum.module
        if module.module_name == declared.module:
            return declared.name
        for statement in module.module.body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            source = module.module_path_identity.resolve_import_from_module(
                imported_module=statement.module, level=statement.level
            )
            for alias in statement.names:
                if source == declared.module and alias.name == declared.name:
                    return ImportBoundNameProjection(statement).alias_bound_name(alias)
        return None

    def root_base(self, model: CaseModel) -> str:
        """The declared root as the family's module names it once imported."""

        local = self.imported_root(model)
        if local is not None:
            return local
        if model.declared.name in module_bound_names(model.enum.module.module):
            raise handback(
                model.enum.module,
                model.enum.node,
                f"{model.declared.name}, the declared family root, is already"
                " bound here",
            )
        return model.declared.name

    # -- the enum read as a family -------------------------------------------------

    def model(
        self, enum: EnumDeclaration, snapshot: CodemodSourceSnapshot
    ) -> CaseModel:
        module = enum.module
        declared = DeclaredRoot.declared_for(Path(module.file_path))
        if enum.node.decorator_list or enum.node.keywords:
            raise handback(
                module,
                enum.node,
                f"enum {enum.name} is decorated or takes class keywords",
            )
        for statement in body_without_docstring(enum.node.body):
            if isinstance(statement, ast.AsyncFunctionDef):
                raise handback(
                    module, statement, "async enum method; no family form derived"
                )
            if not isinstance(
                statement, ast.Assign | ast.AnnAssign | ast.FunctionDef
            ) or (
                isinstance(statement, ast.AnnAssign)
                and StatementValue.of(statement).supplies_value
            ):
                raise handback(
                    module,
                    statement,
                    "enum body statement is neither member, field, nor method",
                )
            if isinstance(statement, ast.Assign) and not all(
                isinstance(argument, ast.expr)
                for member in enum.members
                for argument in member.arguments
            ):
                raise handback(module, statement, "enum member is not a value tuple")
        data, expressions = self.data(enum, declared)
        literals = {
            member.name: {
                name: value
                for name, expression in expressions[member.name].items()
                if (value := literal_value(expression)) is not UNKNOWN
            }
            for member in enum.members
        }
        forwarded = self.forwarded(enum, data)
        if declared.value != "value":
            if "value" in forwarded:
                raise handback(
                    module,
                    enum.node,
                    f"property {forwarded['value']} forwards value, which"
                    f" {declared.name} carries as {declared.value}",
                )
            forwarded["value"] = declared.value
        data = [
            Datum(
                forwarded.get(datum.stored, datum.name), datum.annotation, datum.stored
            )
            for datum in data
        ]
        for member in enum.members:
            literals[member.name] = {
                forwarded.get(name, name): value
                for name, value in literals[member.name].items()
            }
        model = CaseModel(
            enum, declared, data, expressions, literals, {}, forwarded, frozenset()
        )
        self.fold_properties(model)
        model.called = frozenset(
            function.name
            for function in enum.methods
            if isinstance(function, ast.FunctionDef)
            and decorator_names(function) == ("property",)
            and function.name not in model.folded
            and function.name not in forwarded.values()
        )
        model.inlined = self.inlined(model, snapshot)
        model.data = [datum for datum in model.data if datum.name not in model.inlined]
        return model

    @staticmethod
    def inlined(
        model: CaseModel, snapshot: CodemodSourceSnapshot
    ) -> dict[str, dict[str, ast.Lambda]]:
        """Data every member answers with a lambda, only called by the enum itself.

        Members are classes now, so each call becomes that member's lambda body
        and the data goes: a stored function beside its one caller restates it.
        Reads are matched by attribute name.
        """

        enum_nodes = {id(node) for node in ast.walk(model.enum.node)}
        inlined: dict[str, dict[str, ast.Lambda]] = {}
        for datum in model.data:
            functions = {
                member.name: model.expressions[member.name][datum.stored]
                for member in model.members
            }
            if not all(map(plain_lambda, functions.values())):
                continue
            lambdas = {
                name: function
                for name, function in functions.items()
                if isinstance(function, ast.Lambda)
            }
            arities = {len(function.args.args) for function in lambdas.values()}
            if len(arities) != 1:
                continue
            arity = arities.pop()
            names = {datum.name, datum.stored}
            if all(
                id(node) in enum_nodes
                and (
                    not isinstance(node.ctx, ast.Load)
                    or (
                        isinstance(parent, ast.Call)
                        and parent.func is node
                        and len(parent.args) == arity
                        and not parent.keywords
                        and not any(
                            isinstance(argument, ast.Starred)
                            for argument in parent.args
                        )
                    )
                )
                for module in snapshot.parsed_modules
                for parent in ast.walk(module.module)
                for node in ast.iter_child_nodes(parent)
                if isinstance(node, ast.Attribute) and node.attr in names
            ):
                inlined[datum.name] = lambdas
        return inlined

    def data(
        self, enum: EnumDeclaration, declared: DeclaredRoot
    ) -> tuple[list[Datum], dict[str, dict[str, ast.expr]]]:
        """``value`` then each field ``__new__`` stores, with every member's expression."""

        module = enum.module
        for member in enum.members:
            if not (
                member.arguments
                and isinstance(member.arguments[0], ast.Constant)
                and isinstance(member.arguments[0].value, str)
            ):
                raise handback(
                    module,
                    member.node,
                    f"member {member.name}'s value is not a string literal;"
                    f" {declared.name} carries a str {declared.value}",
                )
        annotations = {
            statement.target.id: ast.unparse(statement.annotation)
            for statement in enum.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        new = enum.new
        if new is None:
            if not all(len(member.arguments) == 1 for member in enum.members):
                raise handback(module, enum.node, "tuple members without __new__")
            return [Datum("value", "str", "value")], {
                member.name: {"value": member.arguments[0]} for member in enum.members
            }
        body = new.body
        parameters = [argument.arg for argument in new.args.args[1:]]
        stores: dict[str, str] = {}
        if not (
            len(body) >= 2
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and isinstance(body[-1], ast.Return)
            and is_name(body[-1].value, [body[0].targets[0].id])
        ):
            raise handback(module, new, "__new__ is not create, store fields, return")
        created = body[0].targets[0].id
        for statement in body[1:-1]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Attribute)
                and is_name(statement.targets[0].value, [created])
                and is_name(statement.value, parameters)
                and isinstance(statement.value, ast.Name)
            ):
                raise handback(
                    module, statement, "__new__ statement does not store one parameter"
                )
            stores[statement.value.id] = statement.targets[0].attr
        if set(stores) != set(parameters) or "_value_" not in stores.values():
            raise handback(
                module,
                new,
                "__new__ does not store every parameter once, value included",
            )
        value_parameter = next(
            name for name, attribute in stores.items() if attribute == "_value_"
        )
        if parameters[0] != value_parameter:
            raise handback(
                module, new, "the member value is not __new__'s first parameter"
            )
        annotations_by_parameter = {
            argument.arg: (
                None
                if argument.annotation is None
                else ast.unparse(argument.annotation)
            )
            for argument in new.args.args[1:]
        }
        data = [Datum("value", annotations_by_parameter[value_parameter], "value")]
        data.extend(
            Datum(
                stores[parameter],
                annotations.get(stores[parameter], annotations_by_parameter[parameter]),
                stores[parameter],
            )
            for parameter in parameters[1:]
        )
        expressions: dict[str, dict[str, ast.expr]] = {}
        for member in enum.members:
            fields = dict(member.fields)
            values: dict[str, ast.expr] = {"value": member.arguments[0]}
            for parameter in parameters[1:]:
                expression = fields[parameter]
                if expression is None:
                    raise handback(
                        module,
                        member.node,
                        f"member {member.name} leaves {parameter} unset",
                    )
                values[stores[parameter]] = expression
            expressions[member.name] = values
        return data, expressions

    @staticmethod
    def forwarded(enum: EnumDeclaration, data: list[Datum]) -> dict[str, str]:
        """Stored field -> the property that only returns it, which takes its name."""

        stored = {datum.stored for datum in data}
        forwarded: dict[str, str] = {}
        for function in enum.methods:
            if not (
                isinstance(function, ast.FunctionDef)
                and decorator_names(function) == ("property",)
            ):
                continue
            body = body_without_docstring(function.body)
            receiver = function.args.args[0].arg
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and isinstance(body[0].value, ast.Attribute)
                and is_name(body[0].value.value, [receiver])
                and body[0].value.attr in stored
                and body[0].value.attr not in forwarded
            ):
                forwarded[body[0].value.attr] = function.name
        return forwarded

    def fold_properties(self, model: CaseModel) -> None:
        """Properties whose one return every member answers with a literal become data."""

        pending = [
            function
            for function in model.enum.methods
            if isinstance(function, ast.FunctionDef)
            and decorator_names(function) == ("property",)
            and function.name not in model.forwarded.values()
        ]
        progressed = True
        while progressed:
            progressed = False
            for function in list(pending):
                body = body_without_docstring(function.body)
                if (
                    len(body) != 1
                    or not isinstance(body[0], ast.Return)
                    or not StatementValue.of(body[0]).supplies_value
                ):
                    pending.remove(function)
                    continue
                receiver = function.args.args[0].arg
                values = {
                    member.name: self.evaluated(model, member, receiver, body[0].value)
                    for member in model.members
                }
                if any(value is UNKNOWN for value in values.values()):
                    continue
                for member_name, value in values.items():
                    model.literals[member_name][function.name] = value
                model.folded[function.name] = (
                    None if function.returns is None else ast.unparse(function.returns)
                )
                pending.remove(function)
                progressed = True

    def evaluated(
        self, model: CaseModel, member: Member, receiver: str, node: ast.expr
    ) -> object:
        """``node``'s exact value for ``member``, or UNKNOWN."""

        known = self.member_literal(model, member, receiver, node)
        if known is not UNKNOWN:
            return known
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self.evaluated(model, member, receiver, node.operand)
            return UNKNOWN if inner is UNKNOWN else not inner
        if isinstance(node, ast.BoolOp):
            result: object = UNKNOWN
            for value in node.values:
                result = self.evaluated(model, member, receiver, value)
                if result is UNKNOWN:
                    return UNKNOWN
                if bool(result) != isinstance(node.op, ast.And):
                    return result
            return result
        if isinstance(node, ast.Compare):
            values = [
                self.evaluated(model, member, receiver, operand)
                for operand in (node.left, *node.comparators)
            ]
            if any(value is UNKNOWN for value in values):
                return UNKNOWN
            return Folding.evaluated_chain(values, node.ops)
        return UNKNOWN

    def member_literal(
        self, model: CaseModel, member: Member, receiver: str, node: ast.AST
    ) -> object:
        """What ``member`` answers for a read of itself through ``receiver``."""

        if (
            isinstance(node, ast.Attribute)
            and is_name(node.value, [receiver])
            and isinstance(node.ctx, ast.Load)
        ):
            name = model.attribute_name(node.attr)
            return model.literals[member.name].get(name, UNKNOWN)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and is_name(node.left, [receiver])
        ):
            members = self.member_refs(model, node.comparators[0], receiver)
            if members is None:
                return UNKNOWN
            operation = node.ops[0]
            if isinstance(operation, ast.Is | ast.Eq | ast.In):
                return member.name in members
            if isinstance(operation, ast.IsNot | ast.NotEq | ast.NotIn):
                return member.name not in members
        return UNKNOWN

    @staticmethod
    def member_refs(
        model: CaseModel, node: ast.AST, receiver: str
    ) -> frozenset[str] | None:
        """Members ``node`` names: ``E.X``, ``cls.X`` or ``self.X``, alone or in a literal."""

        elements = (
            node.elts if isinstance(node, ast.Tuple | ast.List | ast.Set) else [node]
        )
        names = [
            element.attr
            for element in elements
            if isinstance(element, ast.Attribute)
            and is_name(element.value, [model.root, receiver, "cls"])
            and element.attr in model.class_names
        ]
        return frozenset(names) if len(names) == len(elements) else None

    @staticmethod
    def require_free_class_names(model: CaseModel) -> None:
        bound = module_bound_names(model.enum.module.module)
        for member in model.members:
            if member.class_name in bound:
                raise handback(
                    model.enum.module,
                    member.node,
                    f"{member.class_name} is already bound in the enum's module",
                )

    # -- the enum's own methods ----------------------------------------------------

    def method(
        self,
        model: CaseModel,
        rewrite: Rewrite,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> MethodBodies | None:
        """One enum method on the family; ``None`` for a property that became data."""

        module = model.enum.module
        if not isinstance(function, ast.FunctionDef):
            raise handback(module, function, "async enum method")
        decorators = decorator_names(function)
        if function.name in model.folded or function.name in model.forwarded.values():
            return None
        if decorators not in ((), ("property",), ("classmethod",), ("staticmethod",)):
            raise handback(module, function, f"enum method decorated {decorators}")
        static = decorators == ("staticmethod",)
        receiver = None if static else function.args.args[0].arg
        first = None if static else "cls"
        header_decorator = ["@staticmethod"] if static else ["@classmethod"]
        parameters = parameter_sources(
            function.args,
            first,
            lambda node: member_type(model.root, segment(module, node)),
            lambda node: segment(module, node),
        )
        returns = (
            None
            if function.returns is None
            else member_type(model.root, segment(module, function.returns))
        )
        authored = authored_header(
            module,
            function,
            first,
            lambda node: member_type(model.root, segment(module, node)),
        )
        header = [
            *header_decorator,
            *(
                authored
                if all(
                    len(f"    {line}") <= model.declared.line_length
                    for line in authored
                )
                else signature(
                    function.name,
                    parameters,
                    returns,
                    "    ",
                    model.declared.line_length,
                )
            ),
        ]
        statements = body_without_docstring(function.body)
        documentation = docstring(function.body)
        renames = self.method_renames(
            model, function, receiver, decorators == ("classmethod",)
        )
        applications = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and receiver is not None
            and is_name(node.func.value, [receiver])
            and model.attribute_name(node.func.attr) in model.inlined
        ]
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module,
                rewrite,
                (
                    (
                        lambda node, member=member: self.member_literal(
                            model, member, receiver, node
                        )
                    )
                    if receiver is not None and decorators != ("classmethod",)
                    else (lambda node: UNKNOWN)
                ),
                renames
                | {
                    id(call): applied(
                        model.inlined[model.attribute_name(call.func.attr)][
                            member.name
                        ],
                        [
                            ast.unparse(rebuilt(argument, renames))
                            for argument in call.args
                        ],
                    )
                    for call in applications
                    if isinstance(call.func, ast.Attribute)
                },
                self.boolean_names(function),
            )
            bodies[member.name], _ends = folding.block(statements)
        generic = (
            None
            if applications
            else Folding(
                module,
                rewrite,
                lambda node: UNKNOWN,
                renames,
                self.boolean_names(function),
            ).block(statements)[0]
        )
        return MethodBodies(
            function.name,
            header,
            bodies,
            generic,
            (
                []
                if documentation is None
                else rewrite.statements_source(module, [documentation])
                .rstrip("\n")
                .split("\n")
            ),
        )

    def method_renames(
        self,
        model: CaseModel,
        function: ast.FunctionDef,
        receiver: str | None,
        is_classmethod: bool,
    ) -> dict[int, str]:
        """Member references become classes, the receiver ``cls``, and reads of
        properties now methods become calls."""

        renames: dict[int, str] = {}
        parents = AstParentIndex(function).parent_by_node
        for statement in function.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    if (
                        is_name(
                            node.value, [model.root, *([receiver] if receiver else [])]
                        )
                        and node.attr in model.class_names
                    ):
                        renames[id(node)] = model.class_names[node.attr]
                        continue
                    if receiver is not None and is_name(node.value, [receiver]):
                        name = model.attribute_name(node.attr)
                        parent = parents.get(node)
                        called = name in model.called and not (
                            isinstance(parent, ast.Call) and parent.func is node
                        )
                        if name != node.attr or called:
                            renames[id(node)] = f"cls.{name}" + ("()" if called else "")
                        continue
                if (
                    isinstance(node, ast.Name)
                    and receiver is not None
                    and node.id == receiver
                    and id(node) not in renames
                ):
                    parent = parents.get(node)
                    if is_classmethod and self.iterated(node, parent):
                        renames[id(node)] = f"{model.root}.__subclasses__()"
                    elif not is_classmethod and isinstance(node.ctx, ast.Load):
                        if not (
                            isinstance(parent, ast.Attribute) and id(parent) in renames
                        ):
                            renames[id(node)] = "cls"
                    elif (
                        is_classmethod
                        and isinstance(parent, ast.Call)
                        and parent.func is node
                    ):
                        renames[id(node)] = f"cls.{model.declared.decoder}"
        return renames

    @staticmethod
    def iterated(node: ast.AST, parent: ast.AST | None) -> bool:
        if isinstance(parent, ast.For | ast.comprehension) and parent.iter is node:
            return True
        return (
            isinstance(parent, ast.Call)
            and is_name(parent.func, ITERATING_CALLS)
            and parent.args == [node]
        )

    def boolean_names(self, function: ast.FunctionDef) -> frozenset[str]:
        """Parameters annotated ``bool`` and locals bound once to a boolean expression."""

        names = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if is_name(argument.annotation, ["bool"])
        }
        assignments: dict[str, list[ast.expr]] = {}
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments.setdefault(node.targets[0].id, []).append(node.value)
        for name, values in assignments.items():
            if len(values) == 1 and is_boolean(values[0], frozenset(names)):
                names.add(name)
        return frozenset(names)

    # -- branches on member data outside the enum ----------------------------------

    def sites(
        self,
        snapshot: CodemodSourceSnapshot,
        model: CaseModel,
        aliases: dict[ParsedModule, ModuleNames],
        answer: Answers,
    ) -> list[Site]:
        """Every production block whose tests read member data through a field."""

        found: list[Site] = []
        decided = model.decided_reads
        for module in sorted(aliases, key=lambda module: module.file_path):
            if PythonSourcePathPolicy.is_test_path(module.path):
                continue
            names = aliases[module]
            for owner in module.module.body:
                if not isinstance(owner, ast.ClassDef) or owner.name == model.root:
                    continue
                fields = {
                    statement.target.id: statement
                    for statement in owner.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                }
                held = {
                    name
                    for name, statement in fields.items()
                    if is_name(statement.annotation, names.enum)
                }
                for function in owner.body:
                    if not isinstance(function, ast.FunctionDef):
                        continue
                    self.refuse_member_comparisons(module, function, model, names)
                    if (
                        not held
                        or "staticmethod" in decorator_names(function)
                        or not function.args.args
                    ):
                        continue
                    receiver = function.args.args[0].arg
                    found.extend(
                        self.function_sites(
                            module, owner, function, receiver, held, fields, decided
                        )
                    )
        for site in found:
            site.ask(answer)
        return found

    def refuse_member_comparisons(
        self,
        module: ParsedModule,
        function: ast.FunctionDef,
        model: CaseModel,
        names: ModuleNames,
    ) -> None:
        for node in ast.walk(function):
            if isinstance(node, ast.If | ast.While | ast.IfExp | ast.match_case):
                tests = (
                    [node.guard] if isinstance(node, ast.match_case) else [node.test]
                )
                for test in tests:
                    for part in ast.walk(test) if test is not None else ():
                        if (
                            isinstance(part, ast.Attribute)
                            and is_name(part.value, names.enum)
                            and part.attr in model.class_names
                        ):
                            raise handback(
                                module,
                                part,
                                f"member {part.attr} compared outside the family",
                            )

    def function_sites(
        self,
        module: ParsedModule,
        owner: ast.ClassDef,
        function: ast.FunctionDef,
        receiver: str,
        held: set[str],
        fields: dict[str, ast.AnnAssign],
        decided: frozenset[str],
    ) -> Iterator[Site]:
        body = body_without_docstring(function.body)

        def reads(statement: ast.stmt) -> str | None:
            for node in ast.walk(statement):
                tests = (
                    [node.test]
                    if isinstance(node, ast.If | ast.While | ast.IfExp)
                    else []
                )
                for test in tests:
                    for part in ast.walk(test):
                        if (
                            isinstance(part, ast.Attribute)
                            and part.attr in decided
                            and isinstance(part.value, ast.Attribute)
                            and part.value.attr in held
                            and is_name(part.value.value, [receiver])
                        ):
                            return part.value.attr
            return None

        index = 0
        while index < len(body):
            held_field = reads(body[index])
            if held_field is None:
                index += 1
                continue
            if not isinstance(body[index], ast.If):
                raise handback(
                    module,
                    body[index],
                    "member data decided inside a compound statement",
                )
            end = index + 1
            while (
                end < len(body)
                and isinstance(body[end], ast.If)
                and reads(body[end]) == held_field
            ):
                end += 1
            # A returning run moves whole with the rest of the function, which
            # only runs when the run did not return: together they are one
            # decision. A guard keeps the rest on its class as the tail method.
            # Otherwise each if decides its own fact and becomes its own method,
            # so no override repeats a sibling's answer to another decision.
            returning = any(
                isinstance(node, ast.Return)
                for statement in body[index:end]
                for node in ast.walk(statement)
            )
            if returning and not Site.guards(body, index, end):
                end = len(body)
            for start, stop in (
                [(index, end)]
                if returning
                else [(position, position + 1) for position in range(index, end)]
            ):
                yield Site.decoded(
                    module,
                    owner,
                    function,
                    receiver,
                    held_field,
                    fields,
                    body,
                    start,
                    stop,
                )
            index = end

    # -- the family source ---------------------------------------------------------

    def family_source(
        self,
        model: CaseModel,
        rewrite: Rewrite,
        methods: list[MethodBodies | None],
        base: str,
    ) -> str:
        enum = model.enum
        module = enum.module
        rewrite_methods = [method for method in methods if method is not None]
        documentation = docstring(enum.node.body)
        declared = model.declared
        root: list[str] = [f"class {model.root}({base}):"]
        annotated = [
            *(
                (datum.name, datum.annotation)
                for datum in model.data
                if datum.name != declared.value
            ),
            *model.folded.items(),
        ]
        if documentation is not None:
            root.extend(
                indented(
                    rewrite.statements_source(module, [documentation])
                    .rstrip("\n")
                    .split("\n")
                )
            )
            # Annotations follow after a blank line; each method brings its own.
            if annotated:
                root.append("")
        root.extend(
            f"    {name}: {member_type(model.root, annotation or 'object')}"
            for name, annotation in annotated
        )
        for method in rewrite_methods:
            root.append("")
            root.extend(method.source(method.root_body(model.members), True))
        if len(root) == 1:
            # The declared root carries the value; a root with nothing else is empty.
            root.append("    pass")
        classes = ["\n".join(root)]
        for member in model.members:
            value = member.arguments[0]
            assert isinstance(value, ast.Constant) and isinstance(value.value, str)
            keywords, carried = declared.carrier.member(
                declared.value, member.class_name, value.value, segment(module, value)
            )
            lines = [
                f"class {member.class_name}({', '.join([model.root, *keywords])}):",
                *indented(carried),
            ]
            for datum in model.data:
                if datum.name == declared.value:
                    continue
                expression = model.expressions[member.name][datum.stored]
                lines.extend(indented(self.assignment(module, datum.name, expression)))
            for name in model.folded:
                lines.append(f"    {name} = {model.literals[member.name][name]!r}")
            for method in rewrite_methods:
                body = method.override(member, model.members)
                if body is not None:
                    lines.append("")
                    lines.extend(method.source(body, False))
            if len(lines) == 1:
                lines.append("    pass")
            classes.append("\n".join(lines))
        return "\n\n\n".join(classes)

    @staticmethod
    def assignment(module: ParsedModule, name: str, expression: ast.expr) -> list[str]:
        segment = ast.get_source_segment(module.source, expression)
        if segment is None:
            segment = ast.unparse(expression)
        first, *rest = segment.split("\n")
        return (
            [f"{name} = {first}", *textwrap.dedent("\n".join(rest)).split("\n")]
            if rest
            else [f"{name} = {first}"]
        )

    # -- reads of the enum everywhere else -----------------------------------------

    def rewrite_reads(
        self,
        rewrite: Rewrite,
        module: ParsedModule,
        names: ModuleNames,
        model: CaseModel,
        handled: set[int],
    ) -> None:
        """Member references, annotations and iteration; the root keeps the enum's name."""

        parents = AstParentIndex(module.module).parent_by_node
        annotations = {
            id(node): annotation
            for annotated in ast.walk(module.module)
            for annotation in annotations_of(annotated)
            for node in type_positions(annotation)
        }
        bound = module_bound_names(module.module)
        local_names = set(names.enum)
        if module is model.enum.module:
            local_names.add(model.root)
        for node in ast.walk(module.module):
            if id(node) in handled:
                continue
            if (
                id(node) in annotations
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                text = (
                    member_type(model.root, node.value)
                    if model.root in local_names
                    else node.value
                )
                if text != node.value:
                    rewrite.replace(module, node, f'"{text}"')
                    handled.add(id(node))
                continue
            if not is_name(node, local_names):
                continue
            parent = parents.get(node)
            if (
                isinstance(parent, ast.Attribute)
                and parent.attr in model.class_names
                and id(parent) not in handled
            ):
                member_class = model.class_names[parent.attr]
                if module is not model.enum.module and member_class in bound:
                    raise handback(
                        module,
                        parent,
                        f"{member_class} is already bound in this module",
                    )
                rewrite.replace(module, parent, member_class)
                handled.add(id(parent))
                continue
            if id(node) in annotations:
                if isinstance(parent, ast.Subscript) and is_name(
                    parent.value, ["type"]
                ):
                    continue
                rewrite.replace(module, node, f"type[{node.id}]")
                continue
            if self.iterated(node, parent):
                rewrite.replace(module, node, f"{node.id}.__subclasses__()")
                continue
            if (
                isinstance(parent, ast.Call)
                and parent.func is node
                and len(parent.args) == 1
                and not isinstance(parent.args[0], ast.Starred)
                and not parent.keywords
            ):
                # An enum call decodes a value and passes a member through; the
                # declared decoder is only declared for values.
                if not provably_text(parent.args[0], parent, parents):
                    raise handback(
                        module,
                        parent,
                        f"{node.id}(...) also passes a member through, and"
                        f" {model.declared.decoder} is declared for values: decode"
                        " the argument once where it enters",
                    )
                rewrite.replace(module, node, f"{node.id}.{model.declared.decoder}")
                continue
            if isinstance(parent, ast.Call) and (
                parent.func is node
                or (is_name(parent.func, BuiltinCallName.type_test_names()))
            ):
                raise handback(
                    module,
                    node,
                    f"enum {node.id} called or instance-checked; members are classes now",
                )

    def rewrite_called(
        self,
        rewrite: Rewrite,
        snapshot: CodemodSourceSnapshot,
        model: CaseModel,
        handled: set[int],
    ) -> None:
        """A property that is not data per member becomes a method; its reads, calls.

        Reads are matched by attribute name, so no other class may declare the name.
        """

        if not model.called:
            return
        for module in snapshot.parsed_modules:
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.ClassDef)
                    and node is not model.enum.node
                    and any(
                        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                        and statement.name in model.called
                        or isinstance(statement, ast.AnnAssign | ast.Assign)
                        and stored_names([statement]) & model.called
                        for statement in node.body
                    )
                ):
                    raise handback(
                        module,
                        node,
                        f"{node.name} also declares one of {sorted(model.called)}",
                    )
        for module in snapshot.parsed_modules:
            parents = AstParentIndex(module.module).parent_by_node
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in model.called
                    and isinstance(node.ctx, ast.Load)
                    and id(node) not in handled
                ):
                    parent = parents.get(node)
                    if isinstance(parent, ast.Call) and parent.func is node:
                        raise handback(module, node, f"{node.attr} is already called")
                    end = rewrite.geometry(module).required_node_offsets(node)[1]
                    rewrite.span(module, end, end, "()")
                    handled.add(id(node))


def provably_text(
    argument: ast.expr, call: ast.Call, parents: dict[ast.AST, ast.AST]
) -> bool:
    """``argument`` is a string literal or a name declared ``str``: a value, never
    a member."""

    if isinstance(argument, ast.Constant):
        return isinstance(argument.value, str)
    return isinstance(argument, ast.Name) and is_name(
        declared_type(argument.id, call, parents), ["str"]
    )


def declared_type(
    name: str, node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.expr | None:
    """The type ``name`` is declared with where ``node`` reads it: a parameter's
    annotation, or the value type of the mapping a ``match`` captured it from."""

    parent = parents.get(node)
    if parent is None:
        return None
    if isinstance(parent, ast.match_case):
        subject = parents.get(parent)
        captured = isinstance(subject, ast.Match) and any(
            isinstance(pattern, ast.MatchMapping)
            and any(
                isinstance(value, ast.MatchAs)
                and value.pattern is None
                and value.name == name
                for value in pattern.patterns
            )
            for pattern in ast.walk(parent.pattern)
        )
        if captured and isinstance(subject.subject, ast.Name):
            return mapping_value_type(
                declared_type(subject.subject.id, subject, parents)
            )
    if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        arguments = parent.args
        return next(
            (
                parameter.annotation
                for parameter in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
                if parameter.arg == name
            ),
            None,
        )
    return declared_type(name, parent, parents)


def mapping_value_type(annotation: ast.expr | None) -> ast.expr | None:
    match annotation:
        case ast.Subscript(
            value=ast.Name(id="dict" | "Mapping"), slice=ast.Tuple(elts=[_, value])
        ):
            return value
    return None


TYPE_WRAPPERS = frozenset(
    {"Annotated", "ClassVar", "Final", "Literal", "Optional", "Union", "None"}
)


def held_type(annotation: ast.expr) -> ast.expr:
    """The type an annotation holds: ``Annotated`` metadata and ``| None`` dropped,
    a string annotation read as its expression."""

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return held_type(ast.parse(annotation.value, mode="eval").body)
        except SyntaxError:
            return annotation
    if (
        isinstance(annotation, ast.Subscript)
        and (
            is_name(annotation.value, ["Annotated"])
            or isinstance(annotation.value, ast.Attribute)
            and annotation.value.attr == "Annotated"
        )
        and isinstance(annotation.slice, ast.Tuple)
        and annotation.slice.elts
    ):
        return held_type(annotation.slice.elts[0])
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        sides = [
            side
            for side in (annotation.left, annotation.right)
            if not CompactValueExpression.project(side).value_is_none_literal
        ]
        if len(sides) == 1:
            return held_type(sides[0])
    return annotation


def scope_walk(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of one scope: nested functions are scopes of their own."""

    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            pending.extend(ast.iter_child_nodes(node))


def may_carry_value(annotation: ast.expr) -> bool:
    """Whether a value of this annotation may have a ``value`` attribute of its own:
    any type it names that is neither a builtin nor a typing wrapper."""

    builtin = set(dir(builtins)) | TYPE_WRAPPERS
    return any(
        isinstance(node, ast.Attribute)
        and node.attr not in builtin
        or isinstance(node, ast.Name)
        and node.id not in builtin
        for node in type_positions(held_type(annotation))
    )


CONTAINERS = frozenset(
    {"tuple", "list", "set", "frozenset", "Sequence", "Iterable", "Collection"}
)


@dataclass(frozen=True)
class Typed:
    """An annotation read in the module that wrote it."""

    module: ParsedModule
    annotation: ast.expr


@dataclass(frozen=True)
class Scope:
    """Where a read sits: its innermost function and that function's class."""

    function: ast.FunctionDef | ast.AsyncFunctionDef | None
    owner: ast.ClassDef | None


@dataclass
class DeclaredTypes:
    """Static types read from declarations alone: annotations, the receiver, a name
    bound once, and loops over annotated containers. Anything else is unknown.

    Classes are found by name across the snapshot; a name two classes share is
    unknown.
    """

    snapshot: CodemodSourceSnapshot

    @cached_property
    def classes(self) -> dict[str, list[tuple[ParsedModule, ast.ClassDef]]]:
        found: dict[str, list[tuple[ParsedModule, ast.ClassDef]]] = {}
        for module in self.snapshot.parsed_modules:
            for node in ast.walk(module.module):
                if isinstance(node, ast.ClassDef):
                    found.setdefault(node.name, []).append((module, node))
        return found

    def class_of(self, typed: Typed | None) -> tuple[ParsedModule, ast.ClassDef] | None:
        if typed is None:
            return None
        held = held_type(typed.annotation)
        if not isinstance(held, ast.Name):
            return None
        found = self.classes.get(held.id, [])
        return found[0] if len(found) == 1 else None

    def field(
        self, owner: tuple[ParsedModule, ast.ClassDef] | None, name: str, depth: int
    ) -> Typed | None:
        if owner is None or depth > 20:
            return None
        module, node = owner
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and is_name(
                statement.target, [name]
            ):
                return Typed(module, statement.annotation)
        for base in node.bases:
            found = self.field(self.class_of(Typed(module, base)), name, depth + 1)
            if found is not None:
                return found
        return None

    @staticmethod
    def element(typed: Typed | None, index: int | None) -> Typed | None:
        """The element type of a container annotation, by position for a tuple."""

        if typed is None:
            return None
        held = held_type(typed.annotation)
        if not (isinstance(held, ast.Subscript) and is_name(held.value, CONTAINERS)):
            return None
        elements = (
            held.slice.elts if isinstance(held.slice, ast.Tuple) else [held.slice]
        )
        if (
            len(elements) == 2
            and isinstance(elements[1], ast.Constant)
            and elements[1].value is Ellipsis
        ):
            return Typed(typed.module, elements[0])
        if is_name(held.value, ["tuple"]):
            if index is None or index >= len(elements):
                return None
            return Typed(typed.module, elements[index])
        return Typed(typed.module, elements[0]) if len(elements) == 1 else None

    @staticmethod
    def scopes(module: ParsedModule) -> dict[int, Scope]:
        found: dict[int, Scope] = {}
        owners = {
            id(statement): node
            for node in ast.walk(module.module)
            if isinstance(node, ast.ClassDef)
            for statement in node.body
        }
        for function in ast.walk(module.module):
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                scope = Scope(function, owners.get(id(function)))
                for node in scope_walk(function):
                    found[id(node)] = scope
        return found

    def of(
        self, module: ParsedModule, node: ast.expr, scope: Scope, depth: int = 0
    ) -> Typed | None:
        if depth > 20:
            return None
        if isinstance(node, ast.Attribute):
            base = self.of(module, node.value, scope, depth + 1)
            return self.field(self.class_of(base), node.attr, depth + 1)
        if not isinstance(node, ast.Name) or scope.function is None:
            return None
        function = scope.function
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if (
            scope.owner is not None
            and arguments
            and arguments[0].arg == node.id
            and "staticmethod" not in decorator_names(function)
        ):
            return Typed(module, ast.Name(scope.owner.name, ast.Load()))
        for argument in arguments:
            if argument.arg == node.id:
                return (
                    None
                    if argument.annotation is None
                    else Typed(module, argument.annotation)
                )
        bindings = [
            binding
            for binding in scope_walk(function)
            if node.id
            in stored_names(
                [binding.target]
                if isinstance(binding, ast.For | ast.comprehension | ast.AnnAssign)
                else binding.targets if isinstance(binding, ast.Assign) else []
            )
        ]
        if len(bindings) != 1:
            return None
        (binding,) = bindings
        if isinstance(binding, ast.AnnAssign):
            return Typed(module, binding.annotation)
        if isinstance(binding, ast.Assign):
            if len(binding.targets) != 1:
                return None
            target, value = binding.targets[0], binding.value
            if isinstance(target, ast.Name):
                return self.of(module, value, scope, depth + 1)
            if (
                isinstance(target, ast.Tuple)
                and isinstance(value, ast.Tuple)
                and len(target.elts) == len(value.elts)
            ):
                for element, source in zip(target.elts, value.elts):
                    if is_name(element, [node.id]):
                        return self.of(module, source, scope, depth + 1)
            return None
        assert isinstance(binding, ast.For | ast.comprehension)
        item = self.element(self.of(module, binding.iter, scope, depth + 1), None)
        if isinstance(binding.target, ast.Name):
            return item
        if isinstance(binding.target, ast.Tuple):
            for index, element in enumerate(binding.target.elts):
                if is_name(element, [node.id]):
                    return self.element(item, index)
        return None


@dataclass
class ValueReads:
    """Reads of a member's value, renamed to the attribute the declared root carries
    it in when that is not ``value``.

    A ``.value`` read is a member's when its subject is a member reference, an enum
    decode, a name a parameter annotated with the enum or a loop over it binds, the
    receiver's own field declared holding the enum, or any field of that name some
    class declares holding the enum. That last match is by name (closed world): a
    field of the same name declared with a type that may carry its own ``value``
    hands the read back.
    """

    model: CaseModel
    snapshot: CodemodSourceSnapshot
    aliases: dict[ParsedModule, ModuleNames]

    def enum_names(self, module: ParsedModule) -> frozenset[str]:
        names = self.aliases.get(module)
        return frozenset() if names is None else names.enum

    def holds_member(self, module: ParsedModule, annotation: ast.expr | None) -> bool:
        return annotation is not None and is_name(
            held_type(annotation), self.enum_names(module)
        )

    @cached_property
    def declarations(self) -> list[tuple[ParsedModule, ast.ClassDef, ast.AnnAssign]]:
        return [
            (module, owner, statement)
            for module in self.snapshot.parsed_modules
            for owner in ast.walk(module.module)
            if isinstance(owner, ast.ClassDef)
            for statement in owner.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]

    @cached_property
    def held_fields(self) -> frozenset[str]:
        return frozenset(
            statement.target.id
            for module, _owner, statement in self.declarations
            if isinstance(statement.target, ast.Name)
            and self.holds_member(module, statement.annotation)
        )

    @cached_property
    def ambiguous_fields(self) -> frozenset[str]:
        return frozenset(
            statement.target.id
            for module, _owner, statement in self.declarations
            if isinstance(statement.target, ast.Name)
            and statement.target.id in self.held_fields
            and not self.holds_member(module, statement.annotation)
            and may_carry_value(statement.annotation)
        )

    def held_subjects(self, module: ParsedModule) -> frozenset[int]:
        """Nodes of ``module`` that hold a member: typed names and own fields."""

        names = self.enum_names(module)
        held: set[int] = set()
        scopes = [
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        ]
        for scope in [module.module, *scopes]:
            typed = {
                argument.arg
                for argument in (
                    [
                        *scope.args.posonlyargs,
                        *scope.args.args,
                        *scope.args.kwonlyargs,
                    ]
                    if not isinstance(scope, ast.Module)
                    else []
                )
                if self.holds_member(module, argument.annotation)
            }
            typed.update(
                loop.target.id
                for loop in scope_walk(scope)
                if isinstance(loop, ast.For | ast.AsyncFor | ast.comprehension)
                and is_name(loop.iter, names)
                and isinstance(loop.target, ast.Name)
            )
            held.update(
                id(node)
                for node in scope_walk(scope)
                if isinstance(node, ast.Name) and node.id in typed
            )
        for owner in ast.walk(module.module):
            if not isinstance(owner, ast.ClassDef):
                continue
            fields = {
                statement.target.id
                for statement in owner.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and self.holds_member(module, statement.annotation)
            }
            for function in owner.body:
                if not (
                    isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
                    and function.args.args
                    and fields
                ):
                    continue
                receiver = function.args.args[0].arg
                held.update(
                    id(node)
                    for node in ast.walk(function)
                    if isinstance(node, ast.Attribute)
                    and node.attr in fields
                    and is_name(node.value, [receiver])
                )
        return frozenset(held)

    @cached_property
    def types(self) -> DeclaredTypes:
        return DeclaredTypes(self.snapshot)

    def is_member(
        self,
        module: ParsedModule,
        subject: ast.expr,
        held: frozenset[int],
        scope: Scope,
    ) -> bool:
        names = self.enum_names(module)
        if id(subject) in held:
            return True
        if isinstance(subject, ast.Call):
            return is_name(subject.func, names)
        if not isinstance(subject, ast.Attribute):
            return False
        if is_name(subject.value, names):
            return subject.attr in self.model.class_names
        if subject.attr not in self.held_fields:
            return False
        typed = self.types.of(module, subject, scope)
        if typed is not None:
            return self.holds_member(typed.module, typed.annotation)
        if subject.attr in self.ambiguous_fields:
            raise handback(
                module,
                subject,
                f"{ast.unparse(subject)}.value: field {subject.attr} is declared"
                " holding the enum and holding another type that may carry value",
            )
        return True

    def rewrite(self, rewrite: Rewrite, handled: set[int]) -> None:
        attribute = self.model.declared.value
        if attribute == "value":
            return
        for module in self.snapshot.parsed_modules:
            held = self.held_subjects(module)
            scopes = DeclaredTypes.scopes(module)
            for node in ast.walk(module.module):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "value"
                    and isinstance(node.ctx, ast.Load)
                    and id(node) not in handled
                    and self.is_member(
                        module,
                        node.value,
                        held,
                        scopes.get(id(node), Scope(None, None)),
                    )
                ):
                    end = rewrite.geometry(module).required_node_offsets(node)[1]
                    rewrite.span(module, end - len("value"), end, attribute)
                    handled.add(id(node))


def literal_value(expression: ast.expr) -> object:
    try:
        value = ast.literal_eval(expression)
    except ValueError:
        return UNKNOWN
    return value if literal_constant(value) else UNKNOWN


def annotations_of(node: ast.AST) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.arg):
        return declared_annotations(node)
    if isinstance(node, ast.AnnAssign):
        return (node.annotation,)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return declared_annotations(node)
    return ()


class SourceLines:
    """Offsets of nodes parsed from one source string."""

    def __init__(self, source: str) -> None:
        self.starts = [0]
        for line in source.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.source = source

    def offsets(self, node: ast.AST) -> tuple[int, int]:
        start = self.starts[node.lineno - 1] + len(
            self.source[self.starts[node.lineno - 1] :]
            .encode()[: node.col_offset]
            .decode()
        )
        end = self.starts[node.end_lineno - 1] + len(
            self.source[self.starts[node.end_lineno - 1] :]
            .encode()[: node.end_col_offset]
            .decode()
        )
        return start, end


@dataclass
class Site:
    """A run of a reader's ``if`` statements deciding member data through one field."""

    module: ParsedModule
    owner: ast.ClassDef
    function: ast.FunctionDef
    receiver: str
    field_name: str
    fields: dict[str, ast.AnnAssign]
    body: list[ast.stmt]
    start: int
    end: int
    method: str = ""

    @classmethod
    def decoded(
        cls,
        module: ParsedModule,
        owner: ast.ClassDef,
        function: ast.FunctionDef,
        receiver: str,
        field_name: str,
        fields: dict[str, ast.AnnAssign],
        body: list[ast.stmt],
        start: int,
        end: int,
    ) -> Site:
        """A guard, or a block."""

        if cls.guards(body, start, end):
            return Guard(
                module, owner, function, receiver, field_name, fields, body, start, end
            )
        return Block(
            module, owner, function, receiver, field_name, fields, body, start, end
        )

    @staticmethod
    def guards(body: list[ast.stmt], start: int, end: int) -> bool:
        """A lone ``if ...: return`` with statements after it."""

        statement = body[start]
        return (
            end == start + 1
            and end < len(body)
            and isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and literal_value(statement.body[0].value or ast.Constant(None)) is None
        )

    @property
    def statements(self) -> list[ast.stmt]:
        return self.body[self.start : self.end]

    @property
    def node(self) -> ast.stmt:
        return self.body[self.start]

    def ask(self, answer: Answers) -> None:
        self.method = answer(MethodName(self.module, self.node))

    def field_read(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == self.field_name
            and is_name(node.value, [self.receiver])
        )

    def literal(self, model: CaseModel, member: Member) -> Callable[[ast.AST], object]:
        def read(node: ast.AST) -> object:
            if isinstance(node, ast.Attribute) and self.field_read(node.value):
                return model.literals[member.name].get(
                    model.attribute_name(node.attr), UNKNOWN
                )
            return UNKNOWN

        return read

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        raise NotImplementedError

    def resolvable(self, model: CaseModel, annotation: str) -> str:
        """``annotation``, if every name in it is bound where the family lives."""

        bound = (
            module_bound_names(model.enum.module.module)
            | {model.root}
            | set(dir(builtins))
        )
        missing = sorted(loaded_names([ast.parse(annotation, mode="eval")]) - bound)
        if missing:
            raise handback(
                self.module,
                self.node,
                f"annotation {annotation!r} names {missing}, unbound in the family's module",
            )
        return member_type(model.root, annotation)

    def family_call(self, returns: bool, arguments: list[str], col_offset: int) -> str:
        """The call to the family method; past 88 columns, one argument per line."""

        prefix = ("return " if returns else "") + (
            f"{self.receiver}.{self.field_name}.{self.method}("
        )
        call = f"{prefix}{', '.join(arguments)})"
        if col_offset + len(call) <= 88:
            return call + "\n"
        return "\n".join(
            [prefix, *(f"    {argument}," for argument in arguments), ")\n"]
        )

    def replace_statements(
        self,
        rewrite: Rewrite,
        statements: list[ast.stmt],
        source: str,
        handled: set[int],
    ) -> None:
        geometry = rewrite.geometry(self.module)
        start = geometry.line_offsets[statements[0].lineno - 1]
        end = geometry.required_node_offsets(statements[-1])[1]
        indent = " " * statements[0].col_offset
        rewrite.span(
            self.module, start, end, textwrap.indent(source, indent).rstrip("\n")
        )
        handled.update(
            id(node) for statement in statements for node in ast.walk(statement)
        )


@dataclass
class Block(Site):
    """The run moves whole: free reads become parameters, its returns the method's."""

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        statements = self.statements
        module = self.module
        if any(
            isinstance(
                node, ast.Yield | ast.YieldFrom | ast.Await | ast.Global | ast.Nonlocal
            )
            for statement in statements
            for node in ast.walk(statement)
        ):
            raise handback(
                module, self.node, "branch yields, awaits or rebinds an outer scope"
            )
        returns = any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        )
        if stored_names(statements) & loaded_names(self.body[self.end :]):
            raise handback(
                module, self.node, "branch binds a name the function reads after it"
            )
        renames: dict[int, str] = {}
        parameters: dict[str, str] = {}
        arguments: dict[str, str] = {}
        prefix = self.body[: self.start]
        bound_before = stored_names(prefix) | {
            argument.arg
            for argument in (
                *self.function.args.args[1:],
                *self.function.args.kwonlyargs,
            )
        }
        boolean_names = family.boolean_names(self.function)
        local_annotations = {
            argument.arg: argument.annotation
            for argument in (
                *self.function.args.args[1:],
                *self.function.args.kwonlyargs,
            )
        }
        parents = AstParentIndex(
            ast.Module(body=list(statements), type_ignores=[])
        ).parent_by_node
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute) and self.field_read(node.value):
                    renames[id(node)] = f"cls.{model.attribute_name(node.attr)}"
                elif self.field_read(node):
                    parent = parents.get(node)
                    if not (
                        isinstance(parent, ast.Attribute) and id(parent) in renames
                    ):
                        renames[id(node)] = "cls"
                elif isinstance(node, ast.Attribute) and is_name(
                    node.value, [self.receiver]
                ):
                    if node.attr not in self.fields or not isinstance(
                        node.ctx, ast.Load
                    ):
                        raise handback(
                            module,
                            node,
                            f"branch reads {self.receiver}.{node.attr}, not a declared field",
                        )
                    renames[id(node)] = node.attr
                    parameters.setdefault(
                        node.attr, ast.unparse(self.fields[node.attr].annotation)
                    )
                    arguments.setdefault(node.attr, f"{self.receiver}.{node.attr}")
                elif is_name(node, [self.receiver]):
                    parent = parents.get(node)
                    if not (
                        isinstance(parent, ast.Attribute)
                        and (id(parent) in renames or self.field_read(parent))
                    ):
                        raise handback(
                            module, node, f"branch reads {self.receiver} itself"
                        )
                elif (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in bound_before
                ):
                    annotation = local_annotations.get(node.id)
                    if annotation is not None:
                        parameters.setdefault(node.id, ast.unparse(annotation))
                    elif node.id in boolean_names:
                        parameters.setdefault(node.id, "bool")
                    else:
                        raise handback(
                            module,
                            node,
                            f"branch reads local {node.id!r} of undeclared type",
                        )
                    arguments.setdefault(node.id, node.id)
        family_bound = module_bound_names(model.enum.module.module) | set(dir(builtins))
        free = (
            loaded_names(statements)
            - stored_names(statements)
            - set(parameters)
            - {self.receiver}
            - family_bound
        )
        if free:
            raise handback(
                module,
                self.node,
                f"branch reads {sorted(free)}, unbound in the family's module",
            )
        if "cls" in parameters:
            raise handback(module, self.node, "branch parameters collide")
        family_returns = (
            self.resolvable(model, ast.unparse(self.function.returns))
            if returns and self.function.returns is not None
            else "None"
        )
        header = [
            "@classmethod",
            *signature(
                self.method,
                [
                    "cls",
                    *(
                        f"{name}: {self.resolvable(model, annotation)}"
                        for name, annotation in parameters.items()
                    ),
                ],
                family_returns,
                "    ",
                model.declared.line_length,
            ),
        ]
        booleans = boolean_names | {
            name for name, annotation in parameters.items() if annotation == "bool"
        }
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module, rewrite, self.literal(model, member), renames, booleans
            )
            bodies[member.name], _ends = folding.block(statements)
        generic, _ends = Folding(
            module, rewrite, lambda node: UNKNOWN, renames, booleans
        ).block(statements)
        call = self.family_call(
            returns, list(arguments.values()), statements[0].col_offset
        )
        self.replace_statements(rewrite, statements, call, handled)
        return MethodBodies(self.method, header, bodies, generic)


@dataclass
class Guard(Site):
    """``if decided: return`` before the rest: the family runs the rest or not.

    The rest stays on its class as the tail method; members that fall through
    call it back through the parameter the family method takes.
    """

    tail: str = ""
    callback: str = ""

    def ask(self, answer: Answers) -> None:
        super().ask(answer)
        self.tail = answer(TailName(self.module, self.node))
        self.callback = answer(CallbackName(self.module, self.node))

    def install(
        self,
        family: CasesFamily,
        model: CaseModel,
        rewrite: Rewrite,
        answer: Answers,
        handled: set[int],
    ) -> MethodBodies:
        module = self.module
        guard = self.node
        assert isinstance(guard, ast.If)
        rest = self.body[self.end :]
        if (
            len(self.function.args.args) != 1
            or self.function.args.kwonlyargs
            or self.function.args.vararg
            or self.function.args.kwarg
        ):
            raise handback(
                module,
                guard,
                "guarded function takes parameters; the tail callback would need them",
            )
        if stored_names(self.body[: self.start]) & loaded_names(rest):
            raise handback(
                module, guard, "the rest reads a local bound before the guard"
            )
        if model.root in loaded_names(rest):
            raise handback(module, guard, "the rest reads the enum itself")
        if any(
            statement.name == self.tail
            for statement in self.owner.body
            if isinstance(statement, ast.FunctionDef)
        ):
            raise handback(
                module, guard, f"{self.owner.name} already declares {self.tail}"
            )
        returns = (
            "None"
            if self.function.returns is None
            else self.resolvable(model, ast.unparse(self.function.returns))
        )
        bodies: dict[str, list[str]] = {}
        for member in model.members:
            folding = Folding(
                module, rewrite, self.literal(model, member), {}, frozenset()
            )
            test = folding.truth(guard.test)
            if not isinstance(test, ast.Constant):
                raise handback(
                    module, guard, "guard is not decided by member data alone"
                )
            bodies[member.name] = (
                ["return"] if test.value else [f"return {self.callback}()"]
            )
        renames = {
            id(node): f"cls.{model.attribute_name(node.attr)}"
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Attribute) and self.field_read(node.value)
        }
        generic = [
            f"if {ast.unparse(rebuilt(guard.test, renames))}:",
            "    return",
            f"return {self.callback}()",
        ]
        callable_name = next(
            (
                ImportBoundNameProjection(statement).alias_bound_name(alias)
                for statement in model.enum.module.module.body
                if isinstance(statement, ast.ImportFrom)
                and statement.module in {"typing", "collections.abc"}
                for alias in statement.names
                if alias.name == "Callable"
            ),
            None,
        )
        if callable_name is None:
            rewrite.ensure_import(model.enum.module, "collections.abc", ("Callable",))
            callable_name = "Callable"
        header = [
            "@classmethod",
            *signature(
                self.method,
                ["cls", f"{self.callback}: {callable_name}[[], {returns}]"],
                returns,
                "    ",
                model.declared.line_length,
            ),
        ]
        tail_source = rewrite.statements_source(module, rest)
        indent = " " * self.function.col_offset
        tail = textwrap.indent(
            "\n".join(
                signature(
                    self.tail,
                    [self.receiver],
                    (
                        None
                        if self.function.returns is None
                        else ast.unparse(self.function.returns)
                    ),
                    indent,
                    model.declared.line_length,
                )
            )
            + "\n"
            + textwrap.indent(tail_source, "    "),
            indent,
        )
        call = self.family_call(
            True, [f"{self.receiver}.{self.tail}"], guard.col_offset
        )
        geometry = rewrite.geometry(module)
        start = geometry.line_offsets[guard.lineno - 1]
        end = geometry.required_node_offsets(self.function)[1]
        rewrite.span(
            module,
            start,
            end,
            textwrap.indent(call, " " * guard.col_offset) + "\n" + tail.rstrip("\n"),
        )
        handled.update(
            id(node)
            for statement in self.body[self.start :]
            for node in ast.walk(statement)
        )
        return MethodBodies(self.method, header, bodies, generic)
